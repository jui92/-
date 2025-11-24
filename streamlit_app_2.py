import os
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from openai import OpenAI
import streamlit as st


# =========================================================
# 0. 환경 설정
# =========================================================
OUTPUT_DIR = "output"  # GraphRAG 인덱싱 결과 폴더

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

st.set_page_config(page_title="Audit GraphRAG QA", layout="wide")


# =========================================================
# 1. 데이터 로딩 유틸
# =========================================================
def _detect_embedding_column(df: pd.DataFrame) -> str:
    """
    text_units.parquet 에서 임베딩 벡터가 들어있는 컬럼명을 자동 탐지.
    기본적으로 'embedding' 을 기대하지만, 없으면
    'vector', 'embeddings' 등의 후보를 순차적으로 검사하고,
    그래도 없으면 리스트 타입(object) 컬럼 중 하나를 선택.
    """
    candidate_cols = ["embedding", "embeddings", "vector", "vectors"]
    for col in candidate_cols:
        if col in df.columns:
            return col

    # fallback: 리스트 타입 컬럼 찾기
    for col in df.columns:
        if df[col].dtype == "object" and isinstance(df[col].dropna().iloc[0], (list, tuple)):
            return col

    raise ValueError("임베딩 벡터 컬럼을 찾지 못했습니다. text_units.parquet 구조를 확인해주세요.")


@st.cache_resource
def load_graph_index(output_dir: str) -> Dict[str, Any]:
    """
    GraphRAG 인덱싱 결과(output 디렉토리)를 읽어서
    검색에 필요한 DataFrame/벡터를 메모리에 올린다.
    """
    data: Dict[str, Any] = {}

    # 필수 파일들 로딩
    text_units_path = os.path.join(output_dir, "text_units.parquet")
    documents_path = os.path.join(output_dir, "documents.parquet")
    entities_path = os.path.join(output_dir, "entities.parquet")
    relationships_path = os.path.join(output_dir, "relationships.parquet")
    communities_path = os.path.join(output_dir, "communities.parquet")

    if not os.path.exists(text_units_path):
        raise FileNotFoundError(f"{text_units_path} 를 찾을 수 없습니다.")
    if not os.path.exists(documents_path):
        raise FileNotFoundError(f"{documents_path} 를 찾을 수 없습니다.")

    text_units = pd.read_parquet(text_units_path)
    documents = pd.read_parquet(documents_path)
    entities = pd.read_parquet(entities_path) if os.path.exists(entities_path) else None
    relationships = (
        pd.read_parquet(relationships_path) if os.path.exists(relationships_path) else None
    )
    communities = (
        pd.read_parquet(communities_path) if os.path.exists(communities_path) else None
    )

    # 임베딩 컬럼 자동 탐지 및 넘파이 배열로 변환
    emb_col = _detect_embedding_column(text_units)
    embeddings_list = text_units[emb_col].tolist()
    embeddings = np.array(embeddings_list, dtype=np.float32)

    # case id / text / document id 등 컬럼명 유연 처리
    id_col = "id" if "id" in text_units.columns else text_units.columns[0]
    text_col_candidates = ["text", "content", "body"]
    text_col = next((c for c in text_col_candidates if c in text_units.columns), None)
    if text_col is None:
        # fallback: 가장 텍스트스러운 object 컬럼
        obj_cols = [c for c in text_units.columns if text_units[c].dtype == "object"]
        text_col = obj_cols[0]

    # 문서 human-readable id, title
    doc_id_col = "id" if "id" in documents.columns else documents.columns[0]
    doc_hr_id_col = (
        "human_readable_id" if "human_readable_id" in documents.columns else doc_id_col
    )
    doc_title_candidates = ["title", "name", "label", "doc_title"]
    doc_title_col = next((c for c in doc_title_candidates if c in documents.columns), None)

    data["text_units"] = text_units
    data["documents"] = documents
    data["entities"] = entities
    data["relationships"] = relationships
    data["communities"] = communities
    data["embeddings"] = embeddings
    data["id_col"] = id_col
    data["text_col"] = text_col
    data["doc_id_col"] = doc_id_col
    data["doc_hr_id_col"] = doc_hr_id_col
    data["doc_title_col"] = doc_title_col

    return data


# =========================================================
# 2. 검색 / 추천 로직
# =========================================================
def get_embedding(text: str) -> np.ndarray:
    """OpenAI 임베딩 호출."""
    if client is None:
        raise RuntimeError("OPENAI_API_KEY 가 설정되지 않았습니다.")
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return np.array(resp.data[0].embedding, dtype=np.float32)


def semantic_search(
    query: str,
    data: Dict[str, Any],
    top_k: int = 10,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    쿼리를 임베딩하고 text_units 와 코사인 유사도 계산 후,
    상위 top_k 결과와 점수 반환.
    """
    q_vec = get_embedding(query)
    doc_vecs = data["embeddings"]  # (N, d)

    # 코사인 유사도
    dot = np.dot(doc_vecs, q_vec)
    doc_norm = np.linalg.norm(doc_vecs, axis=1)
    q_norm = np.linalg.norm(q_vec) + 1e-8
    scores = dot / (doc_norm * q_norm + 1e-8)

    top_idx = np.argsort(scores)[-top_k:][::-1]
    top_scores = scores[top_idx]
    top_df = data["text_units"].iloc[top_idx].copy()
    top_df["similarity"] = top_scores

    return top_df, top_scores


def build_case_summary(
    row: pd.Series,
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    text_unit 한 건에 대해 관련 문서/커뮤니티/엔티티 정보를 모아
    LLM에게 넘기기 좋은 구조로 정리.
    """
    id_col = data["id_col"]
    text_col = data["text_col"]
    doc_id_col = data["doc_id_col"]
    doc_hr_id_col = data["doc_hr_id_col"]
    doc_title_col = data["doc_title_col"]

    text = row[text_col]
    # document_ids 컬럼이 list 형태로 있을 가능성이 큼
    doc_ids = []
    if "document_ids" in row:
        val = row["document_ids"]
        if isinstance(val, (list, tuple)):
            doc_ids = val
        elif pd.notna(val):
            doc_ids = [val]

    doc_meta_list = []
    if len(doc_ids) > 0:
        docs = data["documents"]
        for did in doc_ids:
            sub = docs[docs[doc_id_col] == did]
            if len(sub) == 0:
                continue
            d = sub.iloc[0]
            doc_meta_list.append(
                {
                    "id": d[doc_id_col],
                    "human_readable_id": d.get(doc_hr_id_col, d[doc_id_col]),
                    "title": d.get(doc_title_col, None),
                }
            )

    return {
        "text_id": row[id_col],
        "text": text,
        "documents": doc_meta_list,
    }


def llm_analyze_case(
    query: str,
    case_info: Dict[str, Any],
) -> str:
    """
    단일 유사사례에 대해:
    - 쟁점 요약
    - 유사점/차이점
    - 근거 기반 처분(주의/경고/중징계 등) 추천
    을 LLM으로 생성.
    """
    if client is None:
        raise RuntimeError("OPENAI_API_KEY 가 설정되지 않았습니다.")

    docs_str = ""
    for d in case_info["documents"]:
        docs_str += f"- 문서ID: {d.get('human_readable_id', d.get('id'))}, 제목: {d.get('title')}\n"

    prompt = f"""
당신은 공공기관 내부감사 전문가입니다.

[사용자 질의]
{query}

[후보 유사사례의 본문]
{case_info['text']}

[후보 유사사례의 문서 메타데이터]
{docs_str if docs_str else '(메타데이터 없음)'}

위 정보를 기반으로 아래 형식을 꼭 지켜서 한국어로 답변하세요.

1. 사건 개요 요약 (3~5줄)
2. 쟁점 및 위법/부당 소지
3. 과거 유사사례와의 유사점/차이점 (알 수 있는 범위에서)
4. 권고되는 처분 수준
   - 예: "주의", "경고", "주의 및 제도개선 권고", "중징계 검토" 등
5. 처분 수준에 대한 근거 설명
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "너는 한국의 공공기관 내부감사 전문가이다."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content


def llm_recommend_overall(
    query: str,
    top_cases: List[Dict[str, Any]],
) -> str:
    """
    상위 여러 건의 유사사례를 한 번에 보고
    '종합적 처분·조치 권고'를 만들어 줌.
    """
    if client is None:
        raise RuntimeError("OPENAI_API_KEY 가 설정되지 않았습니다.")

    cases_str = ""
    for i, c in enumerate(top_cases, 1):
        docs_str = ""
        for d in c["documents"]:
            docs_str += f"- 문서ID: {d.get('human_readable_id', d.get('id'))}, 제목: {d.get('title')}\n"
        cases_str += f"\n[사례 {i}]\n본문: {c['text'][:800]}\n문서 메타데이터:\n{docs_str or '(없음)'}\n"

    prompt = f"""
당신은 공공기관 내부감사·징계 전문가입니다.

[사용자 질의]
{query}

[상위 유사사례 모음]
{cases_str}

위 정보를 종합해서 아래 내용을 한국어로 작성하세요.

1. 공통적으로 나타나는 문제 유형 (bullet 형식)
2. 법령/내규 위반 소지가 큰 포인트
3. 종합적인 처분 수준 권고 (예: 주의, 경고, 경고+제도개선, 중징계 검토 등)
4. 왜 그 수준이 적절한지에 대한 설명
5. 향후 유사사례 예방을 위한 제도 개선 또는 내부통제 강화 방안 (3~5개)
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "너는 감사/조치 전문가이다."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content


# =========================================================
# 3. 평가 지표 계산 (Precision@k, Recall@k, MRR, HitRate)
# =========================================================
def precision_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
    retrieved_k = retrieved[:k]
    if not retrieved_k:
        return 0.0
    rel_set = set(relevant)
    hit = sum(1 for r in retrieved_k if r in rel_set)
    return hit / len(retrieved_k)


def recall_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    rel_set = set(relevant)
    hit = sum(1 for r in retrieved_k if r in rel_set)
    return hit / len(rel_set)


def mrr_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
    rel_set = set(relevant)
    for i, r in enumerate(retrieved[:k]):
        if r in rel_set:
            return 1.0 / (i + 1)
    return 0.0


def hit_rate_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
    rel_set = set(relevant)
    return float(any(r in rel_set for r in retrieved[:k]))


# =========================================================
# 4. Streamlit UI
# =========================================================
st.title("📘 감사 유사사례 검색 · 근거 그래프 · 처분 추천")

st.markdown(
    """
이 앱은 **GraphRAG 인덱싱 결과(output 폴더)** 를 기반으로

1. 🔍 유사 감사사례 검색  
2. 🧠 사례 분석 + 설명가능한 처분 추천  
3. 📊 검색 성능 평가 지표(Precision@k, Recall@k, MRR, HitRate)  

를 지원합니다.
"""
)

if client is None:
    st.error("OPENAI_API_KEY 환경 변수가 설정되어 있지 않습니다. .env 또는 시스템 환경변수를 확인하세요.")

# 데이터 로딩 (한 번만)
try:
    data = load_graph_index(OUTPUT_DIR)
except Exception as e:
    st.error(f"GraphRAG output 로딩 실패: {e}")
    st.stop()

tab_search, tab_overall, tab_eval = st.tabs(
    ["🔍 유사사례 검색 & 단일 사례 분석", "🧠 종합 처분 추천", "📊 검색 성능 평가"]
)

# ---------------------------------------------
# 4-1. 유사사례 검색 & 단일 사례 분석
# ---------------------------------------------
with tab_search:
    st.subheader("🔍 유사사례 검색")

    query = st.text_input(
        "감사사례를 검색해보세요. (예: 예산 부적정 집행, 용역계약 위법 등)",
        key="query_input",
    )

    top_k = st.slider("검색 결과 개수 (Top-K)", min_value=3, max_value=30, value=10, step=1)

    if st.button("검색 실행", type="primary"):
        if not query.strip():
            st.warning("검색어를 입력하세요.")
        else:
            with st.spinner("임베딩 검색 중..."):
                try:
                    top_df, scores = semantic_search(query, data, top_k=top_k)
                except Exception as e:
                    st.error(f"검색 중 오류가 발생했습니다: {e}")
                else:
                    st.session_state["last_query"] = query
                    st.session_state["last_results"] = top_df
                    st.success(f"{len(top_df)}건의 유사사례를 찾았습니다.")

    # 검색 결과 표시
    last_results: pd.DataFrame = st.session_state.get("last_results")
    last_query: str = st.session_state.get("last_query", "")

    if last_results is not None:
        st.markdown("---")
        st.markdown("### 🔎 검색 결과 (유사도 순)")

        for idx, row in last_results.iterrows():
            case_info = build_case_summary(row, data)
            sim = row["similarity"]

            # 카드 스타일 출력
            with st.container(border=True):
                doc_ids_str = ", ".join(
                    d.get("human_readable_id", str(d.get("id"))) for d in case_info["documents"]
                ) or "문서 정보 없음"

                st.markdown(
                    f"**사례 ID:** `{case_info['text_id']}` &nbsp;&nbsp; "
                    f"**연관 문서:** {doc_ids_str} &nbsp;&nbsp; "
                    f"**유사도:** `{sim:.3f}`"
                )
                st.markdown(
                    f"<div style='font-size:0.9rem;'>{case_info['text'][:300]}...</div>",
                    unsafe_allow_html=True,
                )

                # 상세 분석(LLM 호출)
                with st.expander("🧠 이 사례 기반 상세 분석 & 처분 추천 보기"):
                    if st.button("이 사례 분석 실행", key=f"analyze_{case_info['text_id']}"):
                        with st.spinner("LLM이 사례를 분석 중입니다..."):
                            try:
                                analysis = llm_analyze_case(last_query, case_info)
                                st.markdown(analysis)
                            except Exception as e:
                                st.error(f"LLM 분석 중 오류: {e}")

# ---------------------------------------------
# 4-2. 여러 사례 기반 종합 처분 추천
# ---------------------------------------------
with tab_overall:
    st.subheader("🧠 상위 유사사례 기반 종합 처분 추천")

    last_results: pd.DataFrame = st.session_state.get("last_results")
    last_query: str = st.session_state.get("last_query", "")

    if last_results is None:
        st.info("먼저 [🔍 유사사례 검색] 탭에서 검색을 한 번 실행해주세요.")
    else:
        st.markdown(f"**현재 기준 질의:** `{last_query}`")
        num_cases = st.slider(
            "종합 분석에 사용할 상위 사례 개수",
            min_value=3,
            max_value=min(20, len(last_results)),
            value=min(5, len(last_results)),
            step=1,
        )

        if st.button("종합 처분 추천 생성", type="primary"):
            with st.spinner("상위 사례들을 종합해 처분 추천을 생성 중입니다..."):
                try:
                    top_cases_info = [
                        build_case_summary(row, data)
                        for _, row in last_results.head(num_cases).iterrows()
                    ]
                    summary = llm_recommend_overall(last_query, top_cases_info)
                    st.markdown(summary)
                except Exception as e:
                    st.error(f"종합 추천 생성 중 오류: {e}")

# ---------------------------------------------
# 4-3. 검색 성능 평가 탭
# ---------------------------------------------
with tab_eval:
    st.subheader("📊 검색 성능 평가 (Precision@k / Recall@k / MRR / HitRate)")

    st.markdown(
        """
**사용 방법**

1. 아래 형식의 JSON 파일을 업로드 합니다.

```json
[
  {
    "query": "예산 부적정 집행",
    "relevant_ids": ["DOC-001", "DOC-005"],
    "k": 10
  },
  {
    "query": "용역 계약 지연",
    "relevant_ids": ["DOC-010"],
    "k": 10
  }
]
""")