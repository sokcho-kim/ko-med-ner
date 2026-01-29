# NER 데이터 형식 명세

> 최종 수정: 2026-01-07

---

## 1. Gazetteer 데이터

### 1.1 개요

| 항목 | 값 |
|------|---|
| 목적 | NER Entity Linking을 위한 정규화된 엔티티 사전 |
| 위치 | `project/ner/data/gazetteer/` |
| 생성 스크립트 | `project/ner/scripts/build_gazetteer.py` |
| 생성 일자 | 2026-01-06 |

### 1.2 파일 목록

| 파일명 | 엔티티 수 | 소스 |
|--------|----------|------|
| `disease_gazetteer.json` | 53,944 | KCD-9, 상병마스터, 별칭 |
| `drug_gazetteer.json` | 24,738 | pharmalex_unity |
| `procedure_gazetteer.json` | 0 | (미구축) |

---

## 2. Disease Gazetteer

### 2.1 형식 비교: 원본 vs 통합 Gazetteer

#### 원본: `kcd9_full.json`

| 항목 | 값 |
|------|---|
| 경로 | `data/kssc/kcd-9th/normalized/kcd9_full.json` |
| 구조 | `codes` 배열 |
| 총 코드 | 54,125 |
| 소스 | KCD-9 단일 |

```json
{
  "version": "KCD-9",
  "release_date": "2025-10-31",
  "revision": "2차 정오",
  "total_codes": 54125,
  "generated_at": "2025-11-05T17:59:20.999452",
  "codes": [
    {
      "code": "A00",
      "name_kr": "콜레라",
      "name_en": "Cholera",
      "is_header": true,
      "classification": "소",
      "symbol": "",
      "note": "",
      "is_lowest": false,
      "is_domestic": false,
      "is_oriental": false,
      "is_additional": false
    }
  ]
}
```

**필드 설명:**

| 필드 | 타입 | 설명 |
|------|------|------|
| `code` | string | KCD 코드 (예: A00, A00.0) |
| `name_kr` | string | 한글 질병명 |
| `name_en` | string | 영문 질병명 |
| `is_header` | bool | 분류 헤더 여부 |
| `classification` | string | 분류 수준 (대/중/소) |
| `symbol` | string | 특수 기호 (†, * 등) |
| `note` | string | 주석 |
| `is_lowest` | bool | 최하위 코드 여부 |
| `is_domestic` | bool | 한국 고유 코드 여부 |
| `is_oriental` | bool | 한의학 코드 여부 |
| `is_additional` | bool | 추가 코드 여부 |

---

#### 통합 Gazetteer: `disease_gazetteer.json`

| 항목 | 값 |
|------|---|
| 경로 | `project/ner/data/gazetteer/disease_gazetteer.json` |
| 구조 | `entries` 딕셔너리 (이름 = 키) |
| 총 엔트리 | 53,944 |
| 소스 | KCD-9 + 상병마스터 + 암별칭 |

```json
{
  "version": "1.0",
  "total": 53944,
  "sources": ["KCD-9", "상병마스터", "별칭"],
  "entries": {
    "콜레라": {
      "code": "A00",
      "name_kr": "콜레라",
      "name_en": "Cholera",
      "source": "KCD-9"
    },
    "대장암": {
      "code": "C189",
      "name_kr": "대장암",
      "name_en": "",
      "source": "별칭"
    }
  }
}
```

**필드 설명:**

| 필드 | 타입 | 설명 |
|------|------|------|
| `code` | string | KCD 코드 |
| `name_kr` | string | 한글 질병명 |
| `name_en` | string | 영문 질병명 (없으면 빈 문자열) |
| `source` | string | 데이터 출처 (KCD-9 / 상병마스터 / 별칭) |

---

### 2.2 형식 차이 요약

| 비교 항목 | `kcd9_full.json` | `disease_gazetteer.json` |
|-----------|------------------|--------------------------|
| 최상위 구조 | `codes` (배열) | `entries` (딕셔너리) |
| 키 방식 | 인덱스 기반 | 질병명(소문자) 기반 |
| 조회 복잡도 | O(n) 순회 | O(1) 해시 |
| 소스 통합 | 단일 (KCD-9) | 다중 (KCD-9 + 상병마스터 + 별칭) |
| 메타데이터 | 풍부 (10개 필드) | 최소 (4개 필드) |
| 용도 | 원본 보존, 분석용 | Entity Linking 최적화 |

### 2.3 데이터 소스별 기여

| 소스 | 기여 | 비고 |
|------|------|------|
| KCD-9 | 52,682개 | 기본 질병 분류 |
| 상병마스터 | 1,247개 | 중복 제외 후 추가분 |
| 암별칭 | 15개 | 대장암, 혈액암, 췌장암 등 일반 용어 |

### 2.4 암별칭 목록

KCD에 없는 일반 용어를 수동 매핑:

| 일반 용어 | KCD 코드 | KCD 정식 명칭 |
|-----------|----------|---------------|
| 대장암 | C189 | 결장의 악성 신생물, 상세불명 |
| 결장암 | C189 | 결장의 악성 신생물, 상세불명 |
| 혈액암 | C959 | 상세불명의 백혈병 |
| 백혈병 | C959 | 상세불명의 백혈병 |
| 췌장암 | C259 | 췌장의 악성 신생물, 상세불명 |
| 폐암 | C349 | 기관지 및 폐의 악성 신생물, 상세불명 |

---

## 3. Drug Gazetteer

### 3.1 형식

| 항목 | 값 |
|------|---|
| 경로 | `project/ner/data/gazetteer/drug_gazetteer.json` |
| 구조 | `entries` 딕셔너리 (제품명 = 키) |
| 총 엔트리 | 24,738 |
| 소스 | pharmalex_unity |

```json
{
  "version": "1.0",
  "total": 24738,
  "sources": ["pharmalex_unity"],
  "entries": {
    "타이레놀정500밀리그람": {
      "product_name": "타이레놀정500밀리그람",
      "ingredient": "아세트아미노펜",
      "company": "한국얀센",
      "source": "pharmalex_unity"
    }
  }
}
```

---

## 4. Entity Linking 스크립트

### 4.1 현재 상태

| 스크립트 | 사용 데이터 | 상태 |
|----------|-------------|------|
| `entity_linking_disease.py` | `kcd9_full.json` (원본) | 업데이트 필요 |
| `entity_linking_drug.py` | - | 확인 필요 |

### 4.2 업데이트 계획

`entity_linking_disease.py`를 새 Gazetteer 형식에 맞게 수정:

```python
# Before (원본 형식)
KCD_FILE = Path("data/kssc/kcd-9th/normalized/kcd9_full.json")
for item in data.get('codes', []):
    code = item.get('code', '')

# After (Gazetteer 형식)
GAZETTEER_FILE = Path("project/ner/data/gazetteer/disease_gazetteer.json")
for name, info in data.get('entries', {}).items():
    code = info.get('code', '')
```

---

## 5. 변경 이력

| 일자 | 변경 내용 |
|------|----------|
| 2026-01-06 | Gazetteer 초기 구축 (build_gazetteer.py) |
| 2026-01-07 | 형식 문서화 |
