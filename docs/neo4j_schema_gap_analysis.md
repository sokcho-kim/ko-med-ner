# Neo4j 스키마 vs NER 엔티티 갭 분석

> 작성일: 2026-01-07
> 목적: 현재 Neo4j 그래프 스키마와 NER 추출 엔티티 간 불일치 분석 및 개선 방향 제시

---

## 1. 문제 요약

| 구분 | 현재 상태 | 필요 상태 |
|------|-----------|-----------|
| NER 추출 엔티티 | Disease, Drug, Procedure | - |
| Neo4j 노드 | Biomarker, Drug, Test | Disease, Drug, Procedure |
| 갭 | Disease 노드 없음, Procedure 노드 없음 | 스키마 확장 필요 |

**결론**: NER → Graph Query 파이프라인 구축을 위해 Neo4j 스키마 확장이 필요함

---

## 2. 현재 Neo4j 스키마 분석

### 2.1 노드 유형

```cypher
// 현재 존재하는 노드
(:Biomarker)  // 바이오마커 (예: EGFR, ALK, HER2)
(:Drug)       // 약물 (예: 게피티닙, 엘로티닙)
(:Test)       // 검사 (예: FISH, IHC, NGS)
```

### 2.2 관계 유형

```cypher
// 현재 관계
(:Drug)-[:TARGETS]->(:Biomarker)     // 약물이 바이오마커를 표적으로 함
(:Biomarker)-[:TESTED_BY]->(:Test)   // 바이오마커가 검사로 측정됨
```

### 2.3 현재 스키마 다이어그램

```
   Drug ──TARGETS──> Biomarker ──TESTED_BY──> Test
```

### 2.4 기존 sample_queries.cypher 쿼리 카테고리

| # | 카테고리 | 설명 |
|---|----------|------|
| 1 | 바이오마커별 표적치료제 조회 | EGFR 표적 약물 등 |
| 2 | 검사별 바이오마커 조회 | NGS로 검사 가능한 바이오마커 |
| 3 | 약물별 표적 바이오마커 조회 | 게피티닙 표적 |
| 4 | 전체 표적치료제-바이오마커 관계 | 전체 조회 |
| 5 | 검사 방법 통계 | 검사별 바이오마커 수 |
| 6 | 경로 탐색 | Drug → Biomarker → Test |
| 7 | 바이오마커 없는 약물 찾기 | 고아 노드 |
| 8 | 다중 약물 표적 바이오마커 | 허브 바이오마커 |
| 9 | 특정 검사로 약물 추천 | Test 기반 추천 |
| 10 | 복합 조건 쿼리 | 여러 조건 결합 |

---

## 3. NER 파이프라인 요구사항

### 3.1 NER 추출 엔티티 (Gazetteer 기반)

| 엔티티 | Gazetteer | 엔트리 수 | 코드 체계 |
|--------|-----------|-----------|-----------|
| **Disease** | disease_gazetteer.json | 54,029개 | KCD-9 |
| **Drug** | drug_gazetteer.json | ~50,000개 | 제품코드 |
| **Procedure** | procedure_gazetteer.json | 7,602개 | EDI 수가코드 |

### 3.2 NER → Graph Query 파이프라인 목표

```
텍스트 입력
    ↓
NER (GLiNER/KoELECTRA)
    ↓
Entity Linking (Gazetteer)
    ↓
[Disease: KCD-9, Drug: 제품코드, Procedure: EDI코드]
    ↓
Cypher Query 생성
    ↓
Neo4j 그래프 조회
    ↓
지식 기반 답변 생성
```

---

## 4. 갭 분석

### 4.1 노드 갭

| NER 엔티티 | Neo4j 노드 | 상태 | 조치 |
|------------|------------|------|------|
| Disease | (없음) | **GAP** | 신규 추가 필요 |
| Drug | Drug | OK | 속성 확장 검토 |
| Procedure | Test (부분) | **GAP** | Procedure 노드 신규 또는 Test 확장 |

### 4.2 관계 갭

현재 그래프는 Drug-Biomarker-Test 중심이나, NER 파이프라인에서 필요한 관계:

| 필요 관계 | 현재 상태 | 비고 |
|-----------|-----------|------|
| Drug -[:TREATS]-> Disease | 없음 | 핵심 관계, 추가 필요 |
| Disease -[:DIAGNOSED_BY]-> Procedure | 없음 | 진단 관계 |
| Disease -[:HAS_BIOMARKER]-> Biomarker | 없음 | 질환-바이오마커 연결 |
| Procedure -[:TESTS]-> Biomarker | 있음 (역방향) | TESTED_BY로 존재 |

### 4.3 코드 체계 갭

| 엔티티 | Gazetteer 코드 | Neo4j 현재 | 조치 |
|--------|----------------|------------|------|
| Disease | KCD-9 (C169 등) | - | 노드에 kcd_code 속성 추가 |
| Drug | 제품코드 | 이름 기반 | product_code 속성 추가 |
| Procedure | EDI 수가코드 | - | edi_code 속성 추가 |

---

## 5. 개선 방향

### 5.1 스키마 확장안 (권장)

```
                              Disease
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
                 TREATS    HAS_BIOMARKER  DIAGNOSED_BY
                    │            │            │
                    ▼            ▼            ▼
   Drug ───TARGETS───> Biomarker <───TESTS─── Procedure
                           │
                           │
                       TESTED_BY
                           │
                           ▼
                         Test
```

### 5.2 신규 노드 정의

```cypher
// Disease 노드
CREATE (:Disease {
  kcd_code: "C169",        // KCD-9 코드 (PK)
  name_kr: "위암",
  name_en: "Gastric cancer",
  category: "악성신생물",
  source: "KCD-9"
})

// Procedure 노드
CREATE (:Procedure {
  edi_code: "M0031",       // EDI 수가코드 (PK)
  name: "피부 및 피하조직, 근육내 이물제거술",
  category: "처치수술",
  source: "EDI수가"
})
```

### 5.3 신규 관계 정의

```cypher
// 약물-질환 치료 관계
CREATE (d:Drug)-[:TREATS {
  indication: "1차 치료",
  evidence_level: "1A"
}]->(dis:Disease)

// 질환-바이오마커 관계
CREATE (dis:Disease)-[:HAS_BIOMARKER {
  frequency: "30%",
  clinical_significance: "예후인자"
}]->(b:Biomarker)

// 질환-시술 진단 관계
CREATE (dis:Disease)-[:DIAGNOSED_BY {
  stage: "확진",
  priority: 1
}]->(p:Procedure)

// 시술-바이오마커 검사 관계
CREATE (p:Procedure)-[:TESTS]->(b:Biomarker)
```

---

## 6. Cypher 템플릿 재정의 필요성

### 6.1 기존 템플릿 문제

- 기존 10개 카테고리는 Drug-Biomarker-Test 스키마 기반
- NER 추출 결과 (Disease, Drug, Procedure)와 직접 연결 불가
- Disease 엔티티를 활용하는 쿼리 패턴 부재

### 6.2 새로운 템플릿 요구사항

NER 결과를 활용한 쿼리 패턴 필요:

| # | 패턴 | 입력 | 출력 |
|---|------|------|------|
| 1 | 질환별 치료제 조회 | Disease | Drug 목록 |
| 2 | 질환별 진단 검사 | Disease | Procedure 목록 |
| 3 | 질환 관련 바이오마커 | Disease | Biomarker 목록 |
| 4 | 약물 적응증 조회 | Drug | Disease 목록 |
| 5 | 복합 경로 탐색 | Disease + Drug | 관련 정보 |

---

## 7. 작업 우선순위

### 7.1 즉시 필요 (E2E 파이프라인 구축 전)

1. **Disease 노드 추가**: KCD-9 기반 질환 노드 생성
2. **Drug-Disease 관계 구축**: 적응증 데이터 연결
3. **Cypher 템플릿 재작성**: NER 엔티티 기반 쿼리 패턴

### 7.2 후속 작업

1. Procedure 노드 추가 또는 Test 노드 확장
2. 관계 속성 풍부화 (근거수준, 빈도 등)
3. 다중 홉 쿼리 최적화

---

## 8. 파일 위치

| 항목 | 경로 |
|------|------|
| 현재 스키마 쿼리 | `neo4j/queries/sample_queries.cypher` |
| 이 문서 | `project/ner/docs/neo4j_schema_gap_analysis.md` |
| Disease Gazetteer | `project/ner/data/gazetteer/disease_gazetteer.json` |
| Procedure Gazetteer | `project/ner/data/gazetteer/procedure_gazetteer.json` |

---

## 9. 결론

1. **현재 상태**: Neo4j 그래프는 Drug-Biomarker-Test 중심으로, NER 추출 엔티티(Disease, Drug, Procedure)와 직접 매핑 불가
2. **필요 조치**: Disease 노드 추가 및 Drug-Disease, Disease-Biomarker 관계 구축
3. **Cypher 템플릿**: 기존 10개 카테고리는 유지하되, NER 엔티티 기반 신규 패턴 추가 필요
4. **"10개" 근거**: 기존 sample_queries.cypher의 10개 카테고리는 Drug-Biomarker-Test 스키마 기준이며, NER 파이프라인용은 별도 정의 필요
