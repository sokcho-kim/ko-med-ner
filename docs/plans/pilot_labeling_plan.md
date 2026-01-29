# 파일럿 라벨링 계획

> 작성일: 2026-01-06
> 목적: E2E 파이프라인 검증을 위한 50문장 라벨링

---

## 1. 목적

**50개 라벨링은 'NER 학습'용이 아니라 'E2E 파이프라인 검증'용**

| 구분 | 목적 | 산출물 |
|------|------|--------|
| 파일럿 라벨링 | 스키마 검증, 경계 케이스 발견 | 라벨링된 50문장 |
| Entity Linking | NER→Neo4j 연결 검증 | 링킹 결과 + 미매칭 분석 |
| Cypher 템플릿 | 질의 패턴→쿼리 변환 검증 | 템플릿 10개 |

---

## 2. 데이터 소스

### 2.1 문제집 OCR 데이터

| 항목 | 값 |
|------|---|
| 소스 경로 | `data/khima/book_ocr/upstage_results/` |
| 책 수 | 3권 (S377341106825120315020, 130, 180) |
| 총 페이지 | 약 200페이지 |
| OCR 엔진 | Upstage Document Parse |

### 2.2 추출 통계

| 항목 | 값 |
|------|---|
| 총 추출 문장 | 1,674개 |
| 선별 문장 | 50개 |
| 추출 스크립트 | `project/ner/scripts/extract_pilot_sentences.py` |

---

## 3. 선정 기준

### 3.1 문장 유효성

```python
def is_valid_sentence(text: str) -> bool:
    # 최소 20자 이상
    # 페이지 번호, 정답만 있는 경우 제외
    # 헤더만 있는 경우 제외
    # paragraph, list, heading1 카테고리만 포함
```

### 3.2 엔티티 다양성

각 엔티티 타입에서 최소 5~6개씩 선택 후, 복합 엔티티(여러 라벨 포함) 우선

| 우선순위 | 기준 |
|----------|------|
| 1순위 | 8개 라벨 각각 최소 6개 확보 |
| 2순위 | 복합 엔티티 (3개+ 라벨 포함) |
| 3순위 | 제도/청구 관련 핵심 문장 |

---

## 4. 선정 결과

### 4.1 엔티티 타입 분포 (전체 1,674문장)

| 라벨 | 개수 | 비율 |
|------|-----|------|
| Criteria | 825 | 49.3% |
| Procedure | 562 | 33.6% |
| Benefit | 367 | 21.9% |
| Drug | 264 | 15.8% |
| Disease | 246 | 14.7% |
| Code | 200 | 11.9% |
| Exam | 184 | 11.0% |
| Material | 114 | 6.8% |

### 4.2 엔티티 타입 분포 (선별 50문장)

| 라벨 | 개수 | 비율 | 목표 달성 |
|------|-----|------|----------|
| Benefit | 22 | 44% | ✅ |
| Disease | 21 | 42% | ✅ |
| Material | 16 | 32% | ✅ |
| Procedure | 16 | 32% | ✅ |
| Criteria | 15 | 30% | ✅ |
| Drug | 14 | 28% | ✅ |
| Code | 13 | 26% | ✅ |
| Exam | 10 | 20% | ✅ |

### 4.3 복합 엔티티 분포

| 라벨 수 | 문장 수 | 예시 |
|--------|--------|------|
| 7개 | 1 | Drug, Procedure, Disease, Code, Material, Benefit, Criteria |
| 6개 | 1 | Drug, Procedure, Disease, Exam, Material, Criteria |
| 5개 | 4 | Drug, Disease, Code, Material, Criteria 등 |
| 4개 | 4 | Drug, Procedure, Disease, Exam 등 |
| 3개 | 8 | Procedure, Disease, Exam 등 |
| 2개 | 15 | Drug + Code, Disease + Criteria 등 |
| 1개 | 17 | Benefit, Code, Criteria 등 |

---

## 5. 출력 파일

### 5.1 파일 위치

```
project/ner/data/pilot_labeling/pilot_sentences_50.jsonl
```

### 5.2 파일 형식

```json
{
  "id": 1,
  "text": "문장 텍스트",
  "source_file": "page_006_upstage.json",
  "page_num": 6,
  "detected_types": ["Drug", "Procedure"],
  "entities": []  // 수동 라벨링할 필드
}
```

### 5.3 라벨링 후 형식

```json
{
  "id": 1,
  "text": "요양급여 : 피보험자의 질병·부상에 대하여...",
  "entities": [
    {"text": "요양급여", "label": "Benefit", "start": 0, "end": 4},
    {"text": "질병", "label": "Disease", "start": 15, "end": 17},
    {"text": "부상", "label": "Disease", "start": 18, "end": 20}
  ]
}
```

---

## 6. 라벨링 스키마 (8개)

| 라벨 | 정의 | 예시 |
|------|------|------|
| **Disease** | 상병, 질환, 진단명 | 요추부 염좌, 고지질혈증, 폐암 |
| **Procedure** | 시술, 수술, 치료법 | 추나요법, 위루술, 마취 |
| **Drug** | 약제 (성분/용량 포함) | 우루사, 알부민, 인슐린 |
| **Exam** | 검사, 진단행위 | MRI, 초음파, 혈액검사 |
| **Code** | 수가코드, 상병코드 | Mt070, M54.56, KK057 |
| **Material** | 재료, 의료기기 | NPWT, 탄력붕대, 카테터 |
| **Benefit** | 급여유형, 기준, 제도 | 선별급여, 비급여, 요양급여 |
| **Criteria** | 인정기준, 산정기준 | "1일 1회", "주 2회 이내" |

---

## 7. 샘플 문장 (10개)

### 7.1 복합 엔티티 (5개+)

**#49** (7개 라벨)
```
① 정상군 : 특정 질병군별 입원일수가 5~95% 사이의 환자
② 열외군 : 하단열외군 - 특정 질병군별 입원일수가 하위 5% 미만인 환자
...
⑤ 전액 본인부담 : 신포괄수가제 내에서도 건강보험의 적용을 받지 못하는 항목은 '비급여'...
```
- Disease: 질병군
- Procedure: 수술, 시술
- Drug: 약제
- Code: 수가
- Material: 치료재료
- Benefit: 비급여, 건강보험
- Criteria: 5~95%, 하위 5%

**#50** (6개 라벨)
```
1) 각 장의 산정지침 또는 분류항목의 "주"에서 별도로 산정할 수 있도록 규정한 약제비, 치료재료대 등
2) 생혈
3) 퇴장방지의약품 사용장려비
...
```
- Drug: 약제비, 퇴장방지의약품
- Procedure: 검사료, 초음파 검사료
- Material: 치료재료대
- Exam: 검체 검사료, 골밀도검사
- Code: KK057, KK058
- Criteria: 별도로 산정

### 7.2 제도/청구 관련

**#10**
```
① 요양급여 : 피보험자 및 피부양자의 질병·부상·출산 등에 대하여 실시하며,
요양급여의 종류는 진찰, 검사, 약제, 치료재료의 지급, 처치·수술 기타의 치료,
예방, 재활, 입원, 간호, 이송 등
```

**#19**
```
① 요양급여 : 보험급여 중 현물급여의 대표적인 형태로, 진찰·검사·약제·처치·수술 등을 의미
② 보험급여 : 건강보험공단이 가입자에게 제공하는 모든 급여(현물 + 현금)를 통칭
```

### 7.3 검사/암검진 관련

**#21**
```
① 간암, 대장암, 유방암, 위암, 전립선암, 혈액암
② 간암, 대장암, 유방암, 위암, 자궁경부암, 폐암
③ 갑상선암, 대장암, 유방암, 위암, 전립선암, 혈액암
```

**#24**
```
① 유방암은 저선량 흉부CT검사로 시행
② 간암은 혈액검사와 X-ray 검사로 시행
③ 자궁경부암은 자궁경부세포검사로 시행
```

---

## 8. 라벨링 결과 (2026-01-06 완료)

### 8.1 라벨링 방법

| 항목 | 값 |
|------|---|
| 방법 | 규칙 기반 + 사전 매칭 |
| 스크립트 | `project/ner/scripts/label_pilot_sentences.py` |
| 출력 파일 | `project/ner/data/pilot_labeling/pilot_sentences_50_labeled.jsonl` |

### 8.2 추출된 엔티티 통계

| 라벨 | 개수 | 비율 |
|------|-----|------|
| Benefit | 114 | 23.5% |
| Procedure | 85 | 17.5% |
| Disease | 70 | 14.4% |
| Exam | 62 | 12.8% |
| Criteria | 57 | 11.7% |
| Code | 48 | 9.9% |
| Material | 31 | 6.4% |
| Drug | 19 | 3.9% |
| **총계** | **486** | 100% |

### 8.3 문장당 평균 엔티티

- 평균: 9.7개/문장
- 최소: 1개
- 최대: 31개 (복합 제도 설명 문장)

### 8.4 라벨링 예시

**문장 #10** (요양급여 정의)
```json
{
  "text": "① 요양급여 : 피보험자 및 피부양자의 질병·부상·출산 등에 대하여 실시하며...",
  "entities": [
    {"text": "요양급여", "label": "Benefit", "start": 2, "end": 6},
    {"text": "질병", "label": "Disease", "start": 22, "end": 24},
    {"text": "부상", "label": "Disease", "start": 25, "end": 27},
    {"text": "진찰", "label": "Procedure", "start": 54, "end": 56},
    {"text": "검사", "label": "Exam", "start": 58, "end": 60},
    {"text": "약제", "label": "Drug", "start": 62, "end": 64},
    {"text": "수술", "label": "Procedure", "start": 80, "end": 82},
    {"text": "90%", "label": "Criteria", "start": 323, "end": 326}
  ]
}
```

**문장 #21** (6대 암검진 - 질병 집중)
```json
{
  "text": "① 간암, 대장암, 유방암, 위암, 전립선암, 혈액암...",
  "entities": [
    {"text": "간암", "label": "Disease"},
    {"text": "대장암", "label": "Disease"},
    {"text": "유방암", "label": "Disease"},
    {"text": "위암", "label": "Disease"},
    {"text": "자궁경부암", "label": "Disease"},
    {"text": "폐암", "label": "Disease"}
    // 총 30개 암종 엔티티
  ]
}
```

### 8.5 사전 구성

```python
ENTITY_DICT = {
    "Disease": ["간암", "대장암", "폐암", "질병", "질환", ...],  # 25개
    "Procedure": ["수술", "시술", "치료", "마취", ...],  # 22개
    "Drug": ["약제", "항생제", "인슐린", ...],  # 13개
    "Exam": ["검사", "검진", "MRI", "CT", ...],  # 23개
    "Code": ["수가", "DRG", "행위별수가제", ...],  # 9개
    "Material": ["재료", "치료재료", "보조기기", ...],  # 14개
    "Benefit": ["급여", "비급여", "요양급여", ...],  # 21개
    "Criteria": ["1일", "1회", "산정", "이상", ...],  # 16개
}
```

---

## 9. Entity Linking 결과 (2026-01-06)

### 9.1 Drug 링킹 결과

| 항목 | 값 |
|------|---|
| Drug 엔티티 수 | 19개 |
| 유니크 텍스트 | 5개 |
| Gazetteer 크기 | 24,738개 (제품명 22,870 + 일반명 1,868) |

### 9.2 매칭 통계

| 상태 | 개수 | 비율 |
|------|-----|------|
| exact_match | 0 | 0% |
| partial_match | 0 | 0% |
| fuzzy_match | 0 | 0% |
| **no_match** | **19** | **100%** |

### 9.3 추출된 Drug 엔티티

```
처방, 약제, 약제비, 의약품, 퇴장방지의약품
```

### 9.4 문제점 발견

> **핵심 발견**: 문제집에 **실제 약품명이 없음**

| 분류 | 예시 | 링킹 가능 |
|------|------|----------|
| 실제 약품명 | 아스피린, 타이레놀, 알부민 | ✅ 가능 |
| 일반 용어 | 약제, 의약품, 처방 | ❌ 불가 |
| 제도 용어 | 약제비, 퇴장방지의약품 | ❌ 불가 |

### 9.5 스키마 개선 제안

| 현재 | 개선안 | 이유 |
|------|--------|------|
| Drug (통합) | **DrugName** + **DrugTerm** | 실제 약품명 vs 제도용어 구분 |

```
DrugName  - 실제 약품명 (알부민, 인슐린, 글리아타민)
DrugTerm  - 약제 관련 용어 (약제, 약제비, 의약품)
```

### 9.6 Disease 링킹 결과

| 항목 | 값 |
|------|---|
| Disease 엔티티 수 | 70개 |
| 유니크 텍스트 | 17개 |
| KCD Gazetteer | 59,323개 |
| **총 매칭률** | **82.9%** |

| 상태 | 개수 | 비율 |
|------|-----|------|
| exact_match | 0 | 0% |
| partial_match | 54 | 77.1% |
| fuzzy_match | 4 | 5.7% |
| no_match | 12 | 17.1% |

**매칭 성공 예시:**
- 간암 → C22.0 간암의 악성 신생물
- 위암 → C16.9 위암 NOS
- 유방암 → D03.5 유방의 제자리흑색종

**매칭 실패:**
- 환자군, 대장암, 혈액암, 췌장암 (KCD 용어 불일치)

---

## 9.7 사용 가능한 데이터 소스

### Gazetteer 후보

| 소스 | 경로 | 크기 | 용도 |
|------|------|------|------|
| **약품 마스터** | `data/pharmalex_unity/merged_pharma_data_active.csv` | 22,870 제품 | Drug 링킹 |
| **KCD-9** | `data/kssc/kcd-9th/normalized/kcd9_full.json` | 54,125 코드 | Disease 링킹 |
| **상병마스터** | `data/hira_master/배포용 상병마스터_250908.xlsx` | - | Disease 보완 |
| **고시** | `data/cg_parsed/고시_20251101.xlsx` | 21MB | Procedure/Criteria |
| **심사지침** | `data/cg_parsed/심사지침_20251101.xlsx` | 640KB | Criteria |
| **행정해석** | `data/cg_parsed/행정해석_20251101.xlsx` | 12MB | 사례 기반 |

### 법령/제도 데이터

| 소스 | 경로 | 용도 |
|------|------|------|
| 의료급여법 | `data/likms/laws/의료급여법.json` | Benefit 용어 |
| 국민건강보험법 | `data/likms/laws/국민건강보험법.json` | Benefit 용어 |
| 요양급여기준규칙 | `data/likms/laws/국민건강보험_요양급여의_기준에_관한_규칙.json` | 제도 용어 |
| HIRA 심사기준 | `data/hira_rulesvc/parsed/*.json` | 심사 용어 |

### PDF 파싱 데이터

| 소스 | 경로 | 용도 |
|------|------|------|
| 건강보험요양급여비용 | `data/hira/parsed/group_a/2025년 1월판...` | Procedure/Code |
| 요양급여 세부사항(약제) | `data/hira/parsed/group_a/요양급여의 적용기준...` | Drug 급여기준 |
| 자동차보험진료수가 | `data/hira/parsed/group_a/자동차보험진료수가...` | Procedure |

### 9.8 Gazetteer 구축 결과 (2026-01-06)

#### 구축 완료

| Gazetteer | 개수 | 소스 |
|-----------|------|------|
| **Disease** | 53,944개 | KCD-9 + 상병마스터 + 암별칭 |
| **Drug** | 24,738개 | pharmalex_unity |
| **Procedure** | 0개 | (고시 파싱 필요) |
| **총계** | **78,682개** | |

- 스크립트: `project/ner/scripts/build_gazetteer.py`
- 출력: `project/ner/data/gazetteer/`

#### 암 별칭 추가

```
대장암 → C189    혈액암 → C959    췌장암 → C259
결장암 → C189    백혈병 → C959    폐암 → C349
```

#### 남은 확장 작업

```
Procedure: + 고시 → 시술/수술명 + 수가코드
Criteria: + 심사지침 → 산정기준 패턴
Code: + 건강보험요양급여비용 → EDI 코드
```

---

## 10. 다음 단계

| 순서 | 작업 | 상태 | 예상 산출물 |
|------|------|------|------------|
| 1 | 규칙 기반 라벨링 | ✅ 완료 | 50문장 × 486 엔티티 |
| 2 | Drug Entity Linking | ✅ 완료 | **0% 매칭** (문제점 발견) |
| 3 | Disease Entity Linking | ✅ 완료 | **82.9% 매칭** |
| 4 | Gazetteer 확장 | ✅ 부분완료 | Disease 53,944개, Drug 24,738개 |
| 5 | Disease 링킹 재테스트 | ⏳ 대기 | 새 Gazetteer로 재검증 |
| 6 | Procedure Gazetteer | ⏳ 대기 | 고시 파싱 필요 |
| 7 | Cypher 템플릿 작성 | ⏳ 대기 | 상위 10개 질의 패턴 |
| 8 | E2E 파이프라인 검증 | ⏳ 대기 | 오류/개선점 리포트 |

---

## 11. 관련 문서

| 문서 | 경로 |
|------|------|
| 질의 정의 | `project/ner/docs/plans/query_definition.md` |
| 추출 스크립트 | `project/ner/scripts/extract_pilot_sentences.py` |
| 라벨링 스크립트 | `project/ner/scripts/label_pilot_sentences.py` |
| Drug 링킹 스크립트 | `project/ner/scripts/entity_linking_drug.py` |
| Disease 링킹 스크립트 | `project/ner/scripts/entity_linking_disease.py` |
| 파일럿 데이터 (원본) | `project/ner/data/pilot_labeling/pilot_sentences_50.jsonl` |
| 파일럿 데이터 (라벨링) | `project/ner/data/pilot_labeling/pilot_sentences_50_labeled.jsonl` |
| Drug 링킹 결과 | `project/ner/data/pilot_labeling/drug_linking_result.jsonl` |
| Disease 링킹 결과 | `project/ner/data/pilot_labeling/disease_linking_result.jsonl` |
| Gazetteer 구축 스크립트 | `project/ner/scripts/build_gazetteer.py` |
| Disease Gazetteer (통합) | `project/ner/data/gazetteer/disease_gazetteer.json` |
| Drug Gazetteer (통합) | `project/ner/data/gazetteer/drug_gazetteer.json` |
| Drug 원본 | `data/pharmalex_unity/merged_pharma_data_active.csv` |
| Disease 원본 (KCD) | `data/kssc/kcd-9th/normalized/kcd9_full.json` |
| Disease 원본 (상병마스터) | `data/hira_master/배포용 상병마스터_250908(2).xlsx` |
