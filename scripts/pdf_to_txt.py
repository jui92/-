import os
import sys
import pdfplumber

def convert_pdf_to_txt(pdf_path, txt_path):
    """PDF → TXT 변환 함수"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"[ERROR] 변환 실패 → {pdf_path} | {e}")
        return False

    # 저장
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    return True


def main(pdf_dir, txt_temp_dir, final_txt_dir):
    print(f"[INFO] PDF 폴더: {pdf_dir}")
    print(f"[INFO] TXT 생성 폴더(temp): {txt_temp_dir}")
    print(f"[INFO] TXT 이동 폴더(final): {final_txt_dir}")

    # 폴더 생성
    os.makedirs(txt_temp_dir, exist_ok=True)
    os.makedirs(final_txt_dir, exist_ok=True)

    # PDF 목록 읽어오기
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    print(f"[INFO] 변환 대상 PDF 총 {len(pdf_files)}개")

    converted_count = 0
    skipped_count = 0

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        txt_filename = os.path.splitext(pdf_file)[0] + ".txt"
        final_txt_path = os.path.join(final_txt_dir, txt_filename)

        # 이미 존재하는 TXT는 건너뛰기 (옵션 B의 핵심)
        if os.path.exists(final_txt_path):
            skipped_count += 1
            print(f"[SKIP] 이미 존재 → {txt_filename}")
            continue

        print(f"[INFO] 변환 중 → {pdf_file}")
        tmp_txt_path = os.path.join(txt_temp_dir, txt_filename)

        if convert_pdf_to_txt(pdf_path, tmp_txt_path):
            # 최종 txt 폴더로 이동
            os.replace(tmp_txt_path, final_txt_path)
            converted_count += 1
        else:
            print(f"[ERROR] 변환 실패: {pdf_file}")

    print("===============================================")
    print(f"[INFO] 새로 변환된 TXT: {converted_count}개")
    print(f"[INFO] 스킵된 PDF (이미 TXT 존재): {skipped_count}개")
    print("===============================================")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("사용법: python pdf_to_txt.py <PDF_DIR> <TEMP_TXT_DIR> <FINAL_TXT_DIR>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3])
