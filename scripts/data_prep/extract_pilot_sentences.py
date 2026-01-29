"""
문제집 OCR 데이터에서 파일럿 라벨링용 50문장 추출
"""
import json
import os
import re
from pathlib import Path
from html import unescape
from collections import defaultdict

# 엔티티 관련 키워드 (다양성 확보용)
ENTITY_KEYWORDS = {
    "Drug": ["약제", "항생제", "투여", "처방", "복용", "주사", "정제", "캡슐", "mg",
             "세팔로스포린", "아미노글리코사이드", "인슐린", "알부민", "진통제"],
    "Procedure": ["수술", "시술", "치료", "검사", "절개", "봉합", "주사", "투여",
                  "전기자극", "물리치료", "재활", "마취"],
    "Disease": ["질환", "질병", "증상", "감염", "폐렴", "당뇨", "고혈압", "암",
                "통증", "염증", "방광", "신경"],
    "Exam": ["검사", "촬영", "초음파", "MRI", "CT", "X선", "혈액", "소변"],
    "Code": ["코드", "분류번호", "EDI", "KCD", "수가"],
    "Material": ["재료", "기구", "장비", "붕대", "거즈", "카테터", "튜브"],
    "Benefit": ["급여", "비급여", "선별급여", "본인부담", "요양급여", "건강보험"],
    "Criteria": ["1일", "1회", "주", "이내", "이상", "까지", "한하여", "산정", "인정"]
}


def extract_text_from_html(html_content: str) -> str:
    """HTML에서 텍스트 추출 (정규식 사용)"""
    if not html_content:
        return ""
    # HTML 태그 제거
    text = re.sub(r'<br\s*/?>', ' ', html_content)  # <br> -> space
    text = re.sub(r'<[^>]+>', '', text)  # 모든 태그 제거
    text = unescape(text)  # HTML 엔티티 디코딩
    # 줄바꿈 및 여러 공백 정리
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_entity_types(text: str) -> list:
    """텍스트에 포함된 엔티티 타입 반환"""
    found_types = []
    for entity_type, keywords in ENTITY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                found_types.append(entity_type)
                break
    return found_types


def is_valid_sentence(text: str) -> bool:
    """유효한 문장인지 확인"""
    if not text or len(text) < 20:
        return False
    # 페이지 번호, 정답만 있는 경우 제외
    if re.match(r'^[\d\s①②③④⑤○]+$', text):
        return False
    if "ANSWER" in text and len(text) < 50:
        return False
    # 헤더만 있는 경우 제외
    if text in ["예상 문제", "해설", "보험심사청구관리사 자격시험 예상문제집"]:
        return False
    return True


def extract_sentences_from_json(json_path: Path) -> list:
    """JSON 파일에서 문장 추출"""
    sentences = []

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 전체 HTML에서 추출
    html_content = data.get('content', {}).get('html', '')
    if not html_content:
        return sentences

    # 개별 요소에서 추출
    elements = data.get('elements', [])
    for elem in elements:
        category = elem.get('category', '')
        # paragraph, list, heading1만 추출 (header, footer 제외)
        if category not in ['paragraph', 'list', 'heading1']:
            continue

        elem_html = elem.get('content', {}).get('html', '')
        text = extract_text_from_html(elem_html)

        if is_valid_sentence(text):
            entity_types = get_entity_types(text)
            sentences.append({
                'text': text,
                'source_file': str(json_path.name),
                'page_num': data.get('page_num', 0),
                'category': category,
                'entity_types': entity_types
            })

    return sentences


def select_diverse_sentences(all_sentences: list, target_count: int = 50) -> list:
    """다양한 엔티티 타입을 포함한 문장 선별"""
    # 엔티티 타입별로 그룹화
    by_type = defaultdict(list)
    for sent in all_sentences:
        for etype in sent['entity_types']:
            by_type[etype].append(sent)

    selected = []
    selected_texts = set()

    # 각 엔티티 타입에서 최소 5개씩 선택
    for etype in ENTITY_KEYWORDS.keys():
        candidates = by_type.get(etype, [])
        count = 0
        for sent in candidates:
            if sent['text'] not in selected_texts and count < 6:
                selected.append(sent)
                selected_texts.add(sent['text'])
                count += 1

    # 나머지는 복합 엔티티 우선으로 채움
    remaining = [s for s in all_sentences if s['text'] not in selected_texts]
    remaining.sort(key=lambda x: len(x['entity_types']), reverse=True)

    for sent in remaining:
        if len(selected) >= target_count:
            break
        if sent['text'] not in selected_texts:
            selected.append(sent)
            selected_texts.add(sent['text'])

    return selected[:target_count]


def main():
    repo_root = Path(__file__).parent.parent.parent
    external_root = Path(os.environ.get("SCRAPE_HUB_ROOT", "C:/Jimin/scrape-hub"))
    base_path = external_root / "data/khima/book_ocr/upstage_results"
    output_path = repo_root / "data/pilot_labeling"

    all_sentences = []

    # 모든 JSON 파일 처리
    for book_dir in base_path.iterdir():
        if not book_dir.is_dir():
            continue

        for json_file in book_dir.glob("*_upstage.json"):
            sentences = extract_sentences_from_json(json_file)
            all_sentences.extend(sentences)

    print(f"총 추출된 문장: {len(all_sentences)}")

    # 엔티티 타입별 통계
    type_counts = defaultdict(int)
    for sent in all_sentences:
        for etype in sent['entity_types']:
            type_counts[etype] += 1

    print("\n엔티티 타입별 분포:")
    for etype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {etype}: {count}")

    # 50개 선별
    selected = select_diverse_sentences(all_sentences, 50)

    print(f"\n선별된 문장: {len(selected)}")

    # 선별된 문장의 엔티티 분포
    selected_types = defaultdict(int)
    for sent in selected:
        for etype in sent['entity_types']:
            selected_types[etype] += 1

    print("\n선별된 문장의 엔티티 타입 분포:")
    for etype, count in sorted(selected_types.items(), key=lambda x: -x[1]):
        print(f"  {etype}: {count}")

    # JSONL로 저장 (라벨링용)
    output_file = output_path / "pilot_sentences_50.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, sent in enumerate(selected, 1):
            record = {
                'id': i,
                'text': sent['text'],
                'source_file': sent['source_file'],
                'page_num': sent['page_num'],
                'detected_types': sent['entity_types'],
                'entities': []  # 수동 라벨링할 필드
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"\n저장 완료: {output_file}")

    # 샘플 출력
    print("\n=== 샘플 문장 (처음 5개) ===")
    for sent in selected[:5]:
        print(f"\n[{sent['entity_types']}]")
        print(f"  {sent['text'][:100]}...")


if __name__ == "__main__":
    main()
