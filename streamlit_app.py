import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import lancedb
import plotly.express as px

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------
# 설정
# ---------------------------------------------------------
OUTPUT_DIR = "output"
EMBEDDING_DIR = os.path.join(OUTPUT_DIR, "embeddings", "text")
TEXT_UNIT_FILE = os.path.join(OUTPUT_DIR, "text_units.parquet")

LLM_MODEL = "gpt-4o-mini"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------------------------------------------------------
# Embeddings Load (LanceDB)
# ---------------------------------------------------------
@st.cache_resource
def load_embeddings_lancedb():
    if not os.path.exists(EMBEDDING_DIR):
        st.error("❌ embeddings/text 디렉토리가 존재하지 않습니다.")
        st.stop()

    db = lancedb.connect(EMBEDDING_DIR)
    table_name = db.table_names()[0]
    table = db.open_table(table_name)

    df_embedding = table.to_pandas()
    df_embedding.rename(columns={"vector": "embedding"}, inplace=True)

    return df_embedding


# ---------------------------------------------------------
# text_units.parquet Load
# ---------------------------------------------------------
@st.cache_resource
def load_text_units():
    if not os.path.exists(TEXT_UNIT_FILE):
        st.error("❌ text_units.parquet 파일이 없습니다.")
        st.stop()
    return pd.read_parquet(TEXT_UNIT_FILE)


# ---------------------------------------------------------
# Semantic Search
# ---------------------------------------------------------
def semantic_search(query, k=5):
    q_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    df_text = load_text_units()
    df_emb = load_embeddings_lancedb()

    df = df_text.merge(df_emb, on="id", how="inner")

    vectors = np.vstack(df["embedding"].values)
    scores = cosine_similarity([q_emb], vectors)[0]
    df["score"] = scores

    return df.sort_values("score", ascending=False).head(k)


# ---------------------------------------------------------
# 처분 추천 LLM
# ---------------------------------------------------------
def recommend_action(case_text):
    prompt = f"""
너는 감사 전문가다.

다음 감사 사례 요약을 기반으로,
1) 위반내용 요약
2) 관련 근거 규정
3) 처분 수위 추천(주의/경고/문책 등)
4) 그 이유

를 작성해라.

감사 요약:
{case_text}
"""

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return resp.choices[0].message.content


# ---------------------------------------------------------
# 검색 성능 평가 지표
# ---------------------------------------------------------
def hit_rate(results, ground_truth_ids):
    return 1 if results["id"].isin(ground_truth_ids).any() else 0


def precision_at_k(results, ground_truth_ids, k=5):
    top_k = results.head(k)
    hit = top_k["id"].isin(ground_truth_ids).sum()
    return hit / k


def recall_at_k(results, ground_truth_ids, k=5):
    top_k = results.head(k)
    relevant = top_k["id"].isin(ground_truth_ids).sum()
    return relevant / len(ground_truth_ids)


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
st.set_page_config(page_title="감사 유사사례 검색 · 처분 추천", layout="wide")

st.title("🔎 감사 유사사례 검색 · 근거 · 처분 추천")

tab1, tab2, tab3 = st.tabs(["🔍 유사사례 검색", "⚖ 처분 추천", "📊 검색 성능 평가"])

# TAB 1
with tab1:
    st.subheader("🔍 유사사례 검색")
    query = st.text_input("검색어 입력", placeholder="예: 예산 부적정 집행")

    if st.button("검색 실행"):
        results = semantic_search(query, k=5)
        st.success("검색 완료!")

        for _, row in results.iterrows():
            st.markdown("---")
            st.markdown(f"### 📄 사례 ID: `{row['id']}` | 점수: **{row['score']:.4f}**")
            st.write(row["text"])

# TAB 2
with tab2:
    st.subheader("⚖ AI 처분 추천")
    case_text = st.text_area("사례 내용", height=200)

    if st.button("추천 생성"):
        with st.spinner("AI 분석 중..."):
            st.write(recommend_action(case_text))

# TAB 3
with tab3:
    st.subheader("📊 검색 성능 평가")
    ground_truth = st.text_input("정답 ID (쉼표 구분)", placeholder="예: 12, 55, 88")

    query_eval = st.text_input("검색어 입력", key="eval_query")

    if st.button("평가 실행"):
        gt = [int(x.strip()) for x in ground_truth.split(",")]

        res = semantic_search(query_eval, k=10)

        st.write(f"Precision@5: {precision_at_k(res, gt):.3f}")
        st.write(f"Recall@5: {recall_at_k(res, gt):.3f}")
        st.write(f"HitRate: {hit_rate(res, gt)}")
