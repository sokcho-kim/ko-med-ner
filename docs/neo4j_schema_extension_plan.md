# Neo4j 스키마 확장 계획

> 작성일: 2026-01-07
> 목적: NER → Graph Query 파이프라인 구축을 위한 스키마 확장 방안

---

## 1. 현황 분석

### 1.1 현재 스키마 (As-Is)

```
Drug ───TARGETS───> Biomarker ───TESTED_BY───> Test
```

| 노드 | 데이터 파일 | 레코드 수 |
|------|------------|----------|
| Drug | anticancer_normalized_v2.csv | 1,001개 (154성분) |
| Biomarker | biomarkers_extracted_v2.json | 23개 |
| Test | biomarker_test_mappings_v2_code_based.json | 134개 관계 |

### 1.2 NER 추출 엔티티

| 엔티티 | Gazetteer | 레코드 수 | 코드 체계 |
|--------|-----------|----------|----------|
| Disease | disease_gazetteer.json | 54,029개 | KCD-9 |
| Drug | drug_gazetteer.json | ~50,000개 | 제품코드 |
| Procedure | procedure_gazetteer.json | 7,602개 | EDI 수가코드 |

### 1.3 갭

| 항목 | 현재 | 필요 | 상태 |
|------|------|------|------|
| Disease 노드 | 없음 | 필요 | **GAP** |
| Drug-Disease 관계 | 없음 | 필요 | **GAP** |
| Procedure 노드 | 없음 | 선택 | GAP |

---

## 2. 보유 데이터 분석

### 2.1 Drug-Disease 관계 소스: HIRA Regimen

**파일**: `neo4j/data/bridges/hira_regimens_normalized.json`

```json
{
  "regimen_id": "REGIMEN_제2025_210호_3DRUG",
  "cancer_name": "자궁암",
  "kcd_codes": ["C53", "C54", "C55"],
  "drugs": [
    {"name": "Dostarlimab", "atc_code": "L01FF07"},
    {"name": "Paclitaxel", "atc_code": "L01CD01"},
    {"name": "Carboplatin", "atc_code": "L01XA02"}
  ],
  "regimen_type": "병용요법",
  "line": "1차",
  "purpose": "고식적요법",
  "announcement_no": "제2025-210호"
}
```

| 항목 | 값 |
|------|------|
| 총 레지멘 수 | 38개 |
| 완전 매핑 | 35개 (92.1%) |
| 고유 암종 | ~15개 |
| 고유 약물 | ~50개 |

**핵심**: HIRA 고시 기반 공식 급여 적응증 데이터

### 2.2 Disease 노드 소스: Cancer-KCD 매핑

**파일**: `neo4j/data/bridges/cancer_kcd_mapping_verified.csv`

| 컬럼 | 설명 |
|------|------|
| cancer_name | 암종명 (위암, 폐암 등) |
| kcd_code | KCD-9 코드 (C16.9 등) |
| kcd_name_kr | KCD 한글명 |

- 검증 완료: 101개 암종-KCD 매핑

### 2.3 Biomarker-Cancer 소스

**파일**: `neo4j/data/bridges/biomarker_cancer_kcd_mapping.csv`

| 바이오마커 | 암종 | KCD |
|-----------|------|-----|
| HER2 | 위암 | C97 |
| BRAF | 대장암 | C19, C20 |
| EGFR | 폐암 | C34 |
| AR | 전립선암 | C61 |

---

## 3. 목표 스키마 (To-Be)

### 3.1 스키마 다이어그램

```
                    Disease
                       │
          ┌────────────┼────────────┐
          │            │            │
       TREATS    HAS_BIOMARKER   DIAGNOSED_BY
          │            │            │
          ▼            ▼            ▼
   Drug ───TARGETS──> Biomarker    Procedure
                          │
                      TESTED_BY
                          │
                          ▼
                        Test
```

### 3.2 신규 노드

#### Disease 노드

```cypher
CREATE (:Disease {
  // 식별자
  disease_id: "DIS_C169",
  kcd_code: "C16.9",

  // 명칭
  name_kr: "위암",
  name_en: "Gastric cancer",
  kcd_name: "상세불명의 위의 악성 신생물",

  // 분류
  category: "악성신생물",
  chapter: "C00-C97",

  // 메타데이터
  source: "KCD-9"
})
```

#### Procedure 노드 (Phase 2)

```cypher
CREATE (:Procedure {
  procedure_id: "PROC_M0031",
  edi_code: "M0031",
  name: "피부 및 피하조직, 근육내 이물제거술",
  category: "처치수술",
  source: "EDI수가"
})
```

### 3.3 신규 관계

#### Drug-Disease (TREATS)

```cypher
// HIRA Regimen 기반
CREATE (d:Drug)-[:TREATS {
  regimen_id: "REGIMEN_제2025_210호_3DRUG",
  regimen_type: "병용요법",        // 병용/단독
  line: "1차",                     // 1차/2차/3차
  purpose: "고식적요법",           // 고식적/완치적
  announcement_no: "제2025-210호",
  announcement_date: "2025.10.1."
}]->(dis:Disease)
```

#### Disease-Biomarker (HAS_BIOMARKER)

```cypher
CREATE (dis:Disease)-[:HAS_BIOMARKER {
  clinical_significance: "치료 반응 예측",
  frequency: "10-15%"
}]->(b:Biomarker)
```

---

## 4. 구현 계획

### Phase 1: Disease 노드 및 Drug-Disease 관계 (필수)

**목표**: NER Disease 엔티티 → Graph Query 가능

| 단계 | 작업 | 입력 | 출력 |
|------|------|------|------|
| 1.1 | Disease 노드 생성 | cancer_kcd_mapping_verified.csv | Disease 노드 101개 |
| 1.2 | Drug-Disease 관계 | hira_regimens_normalized.json | TREATS 관계 ~150개 |
| 1.3 | Biomarker-Disease 관계 | biomarker_cancer_kcd_mapping.csv | HAS_BIOMARKER 11개 |

**예상 결과**:
- Disease 노드: 101개
- TREATS 관계: ~150개 (38 레지멘 × 평균 4 약물)
- HAS_BIOMARKER: 11개

### Phase 2: Procedure 노드 (선택)

**목표**: NER Procedure 엔티티 → Graph Query 가능

| 단계 | 작업 | 입력 | 출력 |
|------|------|------|------|
| 2.1 | Procedure 노드 생성 | procedure_gazetteer.json | Procedure 노드 7,602개 |
| 2.2 | Disease-Procedure 관계 | (별도 데이터 필요) | DIAGNOSED_BY 관계 |

**주의**: Disease-Procedure 관계 데이터가 현재 없음. 수동 매핑 또는 외부 소스 필요.

### Phase 3: 통합 검증

| 단계 | 작업 |
|------|------|
| 3.1 | E2E 파이프라인 테스트 |
| 3.2 | Cypher 템플릿 검증 |
| 3.3 | 성능 최적화 (인덱스) |

---

## 5. 구현 스크립트 명세

### 5.1 Disease 노드 생성 스크립트

**파일**: `neo4j/scripts/create_disease_nodes.py`

```python
# 입력
INPUT_FILE = "neo4j/data/bridges/cancer_kcd_mapping_verified.csv"

# 출력 (Cypher 또는 직접 로드)
OUTPUT_CYPHER = "neo4j/queries/create_disease_nodes.cypher"

# 로직
1. CSV 로드 (101개 암종-KCD 매핑)
2. 중복 KCD 코드 처리 (동일 KCD에 여러 암종명)
3. Disease 노드 CREATE 문 생성
4. 인덱스 생성 (kcd_code, name_kr)
```

### 5.2 Drug-Disease 관계 생성 스크립트

**파일**: `neo4j/scripts/create_treats_relations.py`

```python
# 입력
REGIMEN_FILE = "neo4j/data/bridges/hira_regimens_normalized.json"
DRUG_FILE = "neo4j/data/bridges/anticancer_normalized_v2.csv"

# 로직
1. 레지멘 로드 (38개)
2. 각 레지멘의 drugs → 기존 Drug 노드 매칭 (ATC 코드)
3. 각 레지멘의 kcd_codes → Disease 노드 매칭
4. TREATS 관계 생성 (관계 속성 포함)

# 관계 속성
- regimen_id
- regimen_type (병용/단독)
- line (1차/2차/3차)
- purpose (고식적/완치적)
- announcement_no
```

### 5.3 Cypher 로드 쿼리

```cypher
// Disease 노드 생성
LOAD CSV WITH HEADERS FROM 'file:///cancer_kcd_mapping_verified.csv' AS row
CREATE (:Disease {
  disease_id: 'DIS_' + replace(row.kcd_code, '.', ''),
  kcd_code: row.kcd_code,
  name_kr: row.cancer_name,
  kcd_name: row.kcd_name_kr,
  source: 'KCD-9'
});

// 인덱스 생성
CREATE INDEX disease_kcd_code FOR (d:Disease) ON (d.kcd_code);
CREATE INDEX disease_name_kr FOR (d:Disease) ON (d.name_kr);
```

---

## 6. 데이터 흐름

### Phase 1 완료 후 데이터 흐름

```
NER 텍스트 입력
    ↓
"위암 환자에게 Dostarlimab 투여"
    ↓
NER 추출: [Disease: 위암, Drug: Dostarlimab]
    ↓
Entity Linking (Gazetteer)
    ↓
[Disease: C16.9, Drug: L01FF07]
    ↓
Cypher Query 생성
    ↓
MATCH (dis:Disease {kcd_code: 'C16.9'})<-[:TREATS]-(d:Drug)
WHERE d.atc_code STARTS WITH 'L01'
RETURN d.name, dis.name_kr
    ↓
Graph 조회 결과
```

---

## 7. 예상 쿼리 패턴

### 7.1 질환별 치료제 조회

```cypher
// 위암 치료제 조회
MATCH (d:Drug)-[t:TREATS]->(dis:Disease {name_kr: '위암'})
RETURN d.brand_name_short AS 약물명,
       t.line AS 치료라인,
       t.regimen_type AS 요법유형,
       t.announcement_no AS 고시번호
ORDER BY t.line;
```

### 7.2 약물 적응증 조회

```cypher
// Paclitaxel 적응증 조회
MATCH (d:Drug {normalized_name: 'paclitaxel'})-[t:TREATS]->(dis:Disease)
RETURN dis.name_kr AS 적응증,
       dis.kcd_code AS KCD코드,
       t.line AS 치료라인
```

### 7.3 바이오마커 관련 질환 및 약물

```cypher
// HER2 관련 전체 경로
MATCH (d:Drug)-[:TARGETS]->(b:Biomarker {name_en: 'HER2'})<-[:HAS_BIOMARKER]-(dis:Disease)
RETURN dis.name_kr AS 암종,
       d.brand_name_short AS 표적치료제,
       b.name_ko AS 바이오마커
```

---

## 8. 리스크 및 대응

| 리스크 | 영향 | 대응 |
|--------|------|------|
| Drug 노드 매칭 실패 | TREATS 관계 누락 | ATC 코드 + 성분명 복합 매칭 |
| KCD 코드 형식 불일치 | Disease 링킹 실패 | C16.9 ↔ C169 양방향 지원 |
| 레지멘 데이터 부족 (38개) | 관계 희소 | HIRA 고시 추가 파싱 |
| Procedure 관계 데이터 없음 | Phase 2 지연 | Phase 1 우선 완료 |

---

## 9. 작업 순서 및 의존성

```
[Phase 1]
1. create_disease_nodes.py 작성
   ↓
2. Disease 노드 101개 생성
   ↓
3. create_treats_relations.py 작성
   ↓
4. TREATS 관계 생성
   ↓
5. HAS_BIOMARKER 관계 생성
   ↓
6. Cypher 템플릿 작성 (NER 기반 5개 패턴)
   ↓
7. E2E 파이프라인 테스트

[Phase 2] (선택)
8. Procedure 노드 생성
9. Disease-Procedure 관계 (데이터 확보 후)
```

---

## 10. 파일 위치

| 항목 | 경로 |
|------|------|
| 이 문서 | `project/ner/docs/neo4j_schema_extension_plan.md` |
| 갭 분석 | `project/ner/docs/neo4j_schema_gap_analysis.md` |
| Disease 소스 | `neo4j/data/bridges/cancer_kcd_mapping_verified.csv` |
| TREATS 소스 | `neo4j/data/bridges/hira_regimens_normalized.json` |
| 기존 Drug 노드 | `neo4j/data/bridges/anticancer_normalized_v2.csv` |

---

## 11. 결론

1. **Phase 1 필수**: Disease 노드 + Drug-Disease(TREATS) 관계 구축
2. **데이터 확보 완료**: cancer_kcd_mapping_verified.csv (101개), hira_regimens_normalized.json (38개)
3. **Phase 2 선택**: Procedure 노드는 Disease-Procedure 관계 데이터 확보 후 진행
4. **예상 작업량**: 스크립트 2개 + Cypher 템플릿 5개
