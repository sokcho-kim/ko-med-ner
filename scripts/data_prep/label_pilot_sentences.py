"""
파일럿 50문장 NER 라벨링
규칙 기반 + 사전 매칭으로 엔티티 추출
"""
import json
import re
from pathlib import Path

# 엔티티 사전 (정확 매칭용)
ENTITY_DICT = {
    "Disease": [
        # 암
        "간암", "대장암", "유방암", "위암", "자궁경부암", "폐암", "전립선암",
        "혈액암", "갑상선암", "췌장암", "암",
        # 질환
        "질병", "질환", "부상", "감염", "폐렴", "당뇨", "고혈압", "통증", "염증",
        "질병군", "환자군",
    ],
    "Procedure": [
        # 치료/시술
        "수술", "시술", "치료", "마취", "전신마취", "재활", "예방", "입원", "간호", "이송",
        "진찰", "처치", "진료", "진료행위",
        # 물리치료
        "물리치료", "한냉치료", "표층열치료", "경피적전기신경자극치료", "간섭파전류치료",
        "자외선치료",
        # 기타
        "마취관리", "마취유지", "기본마취", "마취료", "마취유지료", "기본마취료",
    ],
    "Drug": [
        "약제", "약제비", "항생제", "진통제", "인슐린", "알부민",
        "세팔로스포린", "아미노글리코사이드", "의약품", "퇴장방지의약품",
        "처방", "투여", "복용",
    ],
    "Exam": [
        # 검사
        "검사", "검진", "건강검진", "암검진", "일반건강검진", "영유아건강검진",
        "혈액검사", "소변검사", "검체검사", "골밀도검사", "초음파검사", "자궁경부세포검사",
        # 영상
        "MRI", "CT", "X선", "X-ray", "흉부CT", "촬영",
        "방사선", "핵의학", "초음파",
        # 검사료
        "검사료", "초음파 검사료", "검체 검사료",
    ],
    "Code": [
        # 수가 코드 패턴은 정규식으로 별도 처리
        "수가", "수가제", "행위별수가제", "포괄수가제", "신포괄수가제", "총액계약제", "인두제",
        "DRG", "Fee-for-service",
    ],
    "Material": [
        "재료", "치료재료", "치료재료비", "치료재료대", "재료비",
        "기구", "장비", "시설", "붕대", "거즈", "카테터", "튜브",
        "보조기기", "보장구", "장애인보장구",
        "Infusion Pump",
    ],
    "Benefit": [
        # 급여 유형
        "급여", "비급여", "선별급여", "요양급여", "보험급여", "현물급여", "현금급여",
        "장제급여", "출산급여", "의료급여",
        # 보험
        "건강보험", "국민건강보험", "사회보험", "고용보험", "산재보험", "민간보험",
        # 제도
        "NHI", "NHS", "NHIS",
        # 기관
        "건강보험심사평가원", "국민건강보험공단", "보건복지부",
    ],
    "Criteria": [
        # 빈도/횟수
        "1일", "1회", "2회", "1년", "2년", "15분", "1시간",
        "1일 1회", "1일 2회", "주 1회", "주 2회",
        "매", "마다", "이내", "이상", "이하", "미만", "초과",
        # 산정 관련
        "산정", "인정", "별도", "별도로 산정", "별도 산정",
        # 비율
        "50%", "90%", "100%", "5%", "95%",
    ],
}

# 코드 패턴 (정규식)
CODE_PATTERNS = [
    r'[A-Z]{1,2}\d{3,4}',  # KK057, L7990
    r'[A-Z]\d{2}\.\d{1,2}',  # M54.56
    r'Q\d{4}',  # Q2662
]


def find_entities(text: str) -> list:
    """텍스트에서 엔티티 찾기"""
    entities = []
    found_spans = set()  # 중복 방지

    # 1. 사전 기반 매칭
    for label, keywords in ENTITY_DICT.items():
        for keyword in keywords:
            # 긴 키워드부터 매칭 (부분 매칭 방지)
            pass

        # 길이순 정렬 (긴 것 먼저)
        sorted_keywords = sorted(keywords, key=len, reverse=True)
        for keyword in sorted_keywords:
            for match in re.finditer(re.escape(keyword), text):
                start, end = match.start(), match.end()
                span_key = (start, end)

                # 이미 더 긴 엔티티에 포함된 경우 스킵
                is_overlapping = False
                for (s, e) in found_spans:
                    if start >= s and end <= e:
                        is_overlapping = True
                        break

                if not is_overlapping:
                    entities.append({
                        "text": keyword,
                        "label": label,
                        "start": start,
                        "end": end
                    })
                    found_spans.add(span_key)

    # 2. 코드 패턴 매칭
    for pattern in CODE_PATTERNS:
        for match in re.finditer(pattern, text):
            start, end = match.start(), match.end()
            span_key = (start, end)
            if span_key not in found_spans:
                entities.append({
                    "text": match.group(),
                    "label": "Code",
                    "start": start,
                    "end": end
                })
                found_spans.add(span_key)

    # 3. 숫자+단위 패턴 (Criteria)
    criteria_patterns = [
        r'\d+일\s*\d+회',  # 1일 2회
        r'\d+년마다\s*\d+회',  # 2년마다 1회
        r'만\s*\d+세\s*이상',  # 만 20세 이상
        r'\d+%',  # 50%
    ]
    for pattern in criteria_patterns:
        for match in re.finditer(pattern, text):
            start, end = match.start(), match.end()
            span_key = (start, end)
            if span_key not in found_spans:
                entities.append({
                    "text": match.group(),
                    "label": "Criteria",
                    "start": start,
                    "end": end
                })
                found_spans.add(span_key)

    # 위치순 정렬
    entities.sort(key=lambda x: x["start"])

    return entities


def main():
    repo_root = Path(__file__).parent.parent.parent
    input_path = repo_root / "data/pilot_labeling/pilot_sentences_50.jsonl"
    output_path = repo_root / "data/pilot_labeling/pilot_sentences_50_labeled.jsonl"

    results = []
    label_counts = {}

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())
            text = record['text']

            # 엔티티 추출
            entities = find_entities(text)

            # 결과 저장
            record['entities'] = entities
            results.append(record)

            # 통계
            for ent in entities:
                label = ent['label']
                label_counts[label] = label_counts.get(label, 0) + 1

    # 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"라벨링 완료: {len(results)}개 문장")
    print(f"저장 위치: {output_path}")

    # 통계 출력
    print("\n=== 엔티티 라벨 분포 ===")
    total = sum(label_counts.values())
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count} ({count/total*100:.1f}%)")
    print(f"  총 엔티티: {total}개")

    # 샘플 출력
    print("\n=== 샘플 (처음 3개) ===")
    for record in results[:3]:
        print(f"\n[ID {record['id']}] {record['text'][:80]}...")
        for ent in record['entities'][:5]:
            print(f"  - {ent['text']} [{ent['label']}]")
        if len(record['entities']) > 5:
            print(f"  ... 외 {len(record['entities'])-5}개")


if __name__ == "__main__":
    main()
