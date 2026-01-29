# Disease Gazetteer v2.0 변경 로그

> 작성일: 2026-01-07
> 목적: Gazetteer v1.0 → v2.0 변경 내역 문서화

---

## 1. 변경 요약

| 항목 | v1.0 | v2.0 | 변화 |
|------|------|------|------|
| 버전 | 1.0 | **2.0** | - |
| 암 별칭 수 | 15개 (하드코딩) | **101개** (CSV 로드) | +86개 |
| 암 별칭 소스 | 수동 입력 | NCC 71개 암종 기반 | 검증 완료 |
| Disease 총 엔트리 | 53,944개 | **54,029개** | +85개 |

---

## 2. 변경 상세

### 2.1 암 별칭 확장

**기존 (v1.0)**: 15개 주요 암만 하드코딩
```python
cancer_aliases = {
    "위암": "C169",
    "폐암": "C349",
    # ... 15개
}
```

**변경 (v2.0)**: 101개 암종 CSV에서 로드
```python
CANCER_MAPPING_FILE = "neo4j/data/bridges/cancer_kcd_mapping_verified.csv"
# 101개 암종 자동 로드
```

### 2.2 새로 추가된 암종 예시

| 암종 | KCD 코드 | 비고 |
|------|----------|------|
| 가성점액종 | C48.2 | 신규 |
| 간내 담도암 | C22.1 | 신규 |
| 교모세포종 | C71.9 | 신규 |
| 균상식육종 | C84.0 | 신규 |
| 난소상피암 | C56 | 신규 |
| 비소세포폐암 | C34.9 | 신규 (폐암과 동일 코드) |
| 소세포폐암 | C34.9 | 신규 (폐암과 동일 코드) |
| 미만성 거대B세포림프종 | C83.3 | 신규 |
| 골수이형성증후군 | D46.9 | 신규 (D코드) |
| 뇌하수체선종 | D35.2 | 신규 (양성 종양) |

### 2.3 수정된 기존 매핑

| 암종 | v1.0 코드 | v2.0 코드 | 비고 |
|------|----------|----------|------|
| 전립선암 | D075 (제자리암) | **C61** | 악성으로 수정 |
| 갑상선암 | C739 | **C73** | .9 제거 |

---

## 3. 파일 구조 변경

### 3.1 build_gazetteer.py 수정

```python
# 추가된 상수
CANCER_MAPPING_FILE = Path("neo4j/data/bridges/cancer_kcd_mapping_verified.csv")

# 변경된 로직
if CANCER_MAPPING_FILE.exists():
    with open(CANCER_MAPPING_FILE, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # CSV에서 암 별칭 로드
```

### 3.2 disease_gazetteer.json 구조

```json
{
  "version": "2.0",
  "total": 54029,
  "sources": ["KCD-9", "상병마스터", "NCC암별칭"],
  "cancer_aliases_count": 101,
  "entries": {
    "위암": {
      "code": "C169",
      "name_kr": "위암",
      "name_en": "",
      "source": "NCC암별칭",
      "kcd_original": "C16.9",
      "kcd_name": "상세불명의 위의 악성 신생물"
    },
    ...
  }
}
```

---

## 4. 데이터 흐름

```
cancer_kcd_mapping_verified.csv (101개 검증된 매핑)
    ↓
build_gazetteer.py (CSV 로드)
    ↓
disease_gazetteer.json (v2.0)
    ↓
entity_linking_disease.py (Entity Linking 수행)
```

---

## 5. 검증 결과

### 5.1 Entity Linking 테스트 (50문장)

| 지표 | v1.0 | v2.0 |
|------|------|------|
| 총 매칭률 | 82.9% | **97.1%** |
| exact_match | 0% | **57.1%** |
| no_match | 17.1% | **2.9%** |

### 5.2 개선 효과

- "대장암", "혈액암", "췌장암" 등 v1.0에서 no_match였던 항목이 exact_match로 개선
- 암 통용명 인식률 대폭 향상

---

## 6. 파일 위치

| 파일 | 경로 |
|------|------|
| 스크립트 | `project/ner/scripts/build_gazetteer.py` |
| 암 별칭 매핑 | `neo4j/data/bridges/cancer_kcd_mapping_verified.csv` |
| Gazetteer 출력 | `project/ner/data/gazetteer/disease_gazetteer.json` |
| 이 문서 | `project/ner/docs/gazetteer_v2_changelog.md` |

---

## 7. 참고 문서

- `project/ner/docs/cancer_kcd_mapping_analysis.md`: 매핑 검증 과정
- `project/ner/docs/cancer_alias_mapping_spec.md`: 암 별칭 매핑 명세
- `project/ner/docs/data_formats.md`: 데이터 형식 설명
- `project/ner/docs/neo4j_schema_gap_analysis.md`: Neo4j 스키마 갭 분석

---

# Procedure Gazetteer v1.0 구축 로그

> 작성일: 2026-01-07

---

## P1. 데이터 소스

### P1.1 EDI 수가 데이터 (SNOMED-CT 매핑)

| 파일 | 카테고리 | 엔트리 수 |
|------|----------|----------|
| 9장_처치및수술료 | 처치수술 | 2,598개 |
| 2장_검사 | 검사 | 4,110개 |
| 7장_이학요법료 | 이학요법 | 77개 |
| 1장_기본진료료 | 기본진료 | 563개 |
| 3장_영상진단및방사선 | 영상진단 | 4개 |
| 18장_응급의료수가 | 응급의료 | 250개 |
| **총계** | - | **7,602개** |

### P1.2 데이터 위치

```
data/hins/downloads/edi/
├── 9장_19_20용어매핑테이블(처치및수술료)_(심평원코드-SNOMED_CT).xlsx
├── 2장_19_20용어매핑테이블(검사)_(심평원코드-SNOMED_CT).xlsx
└── ...
```

---

## P2. Gazetteer 구조

```json
{
  "version": "1.0",
  "total": 7602,
  "sources": ["EDI수가"],
  "categories": ["처치수술", "검사", "이학요법", ...],
  "entries": {
    "표층열치료": {
      "code": "MM010",
      "name": "표층열치료",
      "category": "이학요법",
      "source": "EDI수가"
    }
  }
}
```

---

## P3. 샘플 엔트리

| 카테고리 | 코드 | 이름 |
|----------|------|------|
| 처치수술 | M0031 | 피부 및 피하조직, 근육내 이물제거술 |
| 검사 | B0001 | 임상병리검사종합검증료 |
| 이학요법 | MM010 | 표층열치료 |
| 응급의료 | T6120020 | 8자형 석고 |

---

## P4. 파일 위치

| 파일 | 경로 |
|------|------|
| Gazetteer | `project/ner/data/gazetteer/procedure_gazetteer.json` |
| 원본 데이터 | `data/hins/downloads/edi/*.xlsx` |
