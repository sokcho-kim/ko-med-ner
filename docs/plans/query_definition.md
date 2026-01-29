# NER 질의(Query) 정의

> 작성일: 2026-01-06
> 목적: GraphRAG 기반 **삭감대응 / 심사청구 보조**

---

## 1. 프로젝트 목적

**삭감대응 및 심사청구 보조를 위한 GraphRAG**

| 기능 | 설명 |
|------|------|
| 급여기준 조회 | 시술/약제의 인정기준, 급여 여부 |
| 동시산정 확인 | A와 B 동시 청구 가능 여부 |
| 상병-코드 매핑 | 상병명 ↔ 상병코드(KCD) 매핑 |
| 삭감 사유 분석 | 왜 삭감되었는지, 어떻게 대응할지 |

---

## 2. 스키마 비교: 문제집 vs 사용자 질의

### 2.1 문제집 기반 스키마 (13개)

```
# 임상 (4개)
Disease     - 질병, 증상
Body        - 신체 부위
Treatment   - 치료법
Symptom     - 증상

# 약학 (3개)
Drug        - 의약품명
Ingredient  - 성분명
Dosage      - 용량

# 제도/청구 (6개)
Benefit     - 급여 유형
Procedure   - 진료행위
Fee         - 수가
Code        - 의료코드
Organization - 기관명
Regulation  - 법령
```

**장점**: 포괄적, 이론적 완성도
**단점**: 실제 질의와 괴리, 일부 라벨 사용빈도 낮음

---

### 2.2 사용자 질의 기반 스키마 (7개)

```
Procedure   - 시술/수술 (추나, 위루술)
Drug        - 약제 (우루사, 알부민)
Exam        - 검사 (MRI, 초음파)
Disease     - 상병 (요추부 염좌)
Code        - 수가/상병코드 (Mt070, M54.56)
Material    - 재료 (NPWT, 탄력붕대)
Policy      - 제도/기준 (선별급여)
```

**장점**: 실제 사용 패턴 반영
**단점**: 사용자가 미숙해 질의가 단편적

---

### 2.3 비교 매트릭스

| 문제집 스키마 | 사용자 질의 | 삭감대응 필요도 | 통합 |
|--------------|------------|----------------|------|
| Disease | Disease | ⭐⭐⭐ 높음 | ✅ **Disease** |
| Body | - | ⭐ 낮음 | ❌ 제외 |
| Treatment | Procedure | ⭐⭐⭐ 높음 | ✅ **Procedure** |
| Symptom | - | ⭐ 낮음 | ❌ 제외 |
| Drug | Drug | ⭐⭐⭐ 높음 | ✅ **Drug** |
| Ingredient | - | ⭐⭐ 중간 | ⚠️ Drug에 포함 |
| Dosage | - | ⭐⭐ 중간 | ⚠️ Drug에 포함 |
| Benefit | Policy | ⭐⭐⭐ 높음 | ✅ **Benefit** |
| Procedure | - | - | → Treatment와 통합 |
| Fee | - | ⭐⭐ 중간 | ⚠️ Code에 포함 |
| Code | Code | ⭐⭐⭐ 높음 | ✅ **Code** |
| Organization | - | ⭐ 낮음 | ❌ 제외 |
| Regulation | Policy | ⭐⭐ 중간 | → Benefit에 통합 |
| - | Exam | ⭐⭐⭐ 높음 | ✅ **Exam** (추가) |
| - | Material | ⭐⭐ 중간 | ✅ **Material** (추가) |

---

## 3. 통합 스키마 (8개)

### 3.1 핵심 엔티티

| 라벨 | 정의 | 예시 | 삭감대응 용도 |
|------|------|------|--------------|
| **Disease** | 상병, 질환, 진단명 | 요추부 염좌, 고지질혈증, 추간판탈출증 | 적응증 확인, 상병-시술 매칭 |
| **Procedure** | 시술, 수술, 치료법 | 추나요법, 위루술, 척추후궁절제술 | 급여기준, 동시산정 |
| **Drug** | 약제 (성분/용량 포함) | 우루사, 알부민 500mg, 글리아타민 | 투여기준, 병용투여 |
| **Exam** | 검사, 진단행위 | MRI, 초음파, 골밀도검사 | 검사 급여기준 |
| **Code** | 수가코드, 상병코드 | Mt070, U2233, M54.56, V191 | 청구서 작성, 코드 매핑 |
| **Material** | 재료, 의료기기 | NPWT, 탄력붕대, Gastrostomy tube | 재료대 청구 |
| **Benefit** | 급여유형, 기준, 제도 | 선별급여, 비급여, 본인부담률 | 급여 여부 판단 |
| **Criteria** | 인정기준, 산정기준 | "1일 1회", "주 2회 이내" | 삭감 방어 근거 |

### 3.2 Code 타입 구분 (속성)

Code 라벨은 단일하되 `code_type` 속성으로 구분:

```
Code {
  value: "M54.56"
  type: "KCD"      # KCD (상병) / EDI (수가) / ATC (약품) / 재료
}
```

### 3.3 Criteria 라벨링 가이드

| 패턴 | 예시 | 라벨링 방법 |
|------|------|------------|
| 숫자+단위+빈도 | "1일 3회", "주 2회" | span 전체를 Criteria |
| 조건문 | "~한 경우에 한하여" | Criteria + Benefit 분리 |
| 긴 문장 | "동일 마취하 연속 수술 시 50%" | 핵심 조건만 Criteria |

---

## 4. Neo4j 스키마와의 호환성

### 4.1 기존 Neo4j 노드

```
Drug (50만+), Disease (54K), Procedure (1.5K),
Cancer (100), Test (19), Guideline (9K), Document (600)
```

### 4.2 NER → Neo4j 매핑

| NER 라벨 | Neo4j 노드 | 호환성 | 비고 |
|----------|-----------|--------|------|
| Disease | Disease | ✅ 일치 | - |
| Procedure | Procedure | ✅ 일치 | - |
| Drug | Drug | ✅ 일치 | - |
| Exam | Test | ⚠️ 유사 | 이름만 다름, 매핑 가능 |
| Code | (속성) | ⚠️ 변환 | kcd_code, code_kor 등 속성으로 존재 |
| Material | ❌ 없음 | 🔧 추가 필요 | 새 노드 타입 필요 |
| Benefit | Guideline | ⚠️ 파생 | Guideline에서 추출 |
| Criteria | Guideline | ⚠️ 파생 | Guideline 내용에서 추출 |

### 4.3 필요한 Neo4j 변경사항

#### 추가 필요 (신규)

```cypher
// Material 노드 추가
(:Material {
  name: String,
  code: String,
  category: String  // NPWT, 붕대, 튜브 등
})

// CONCURRENT_RULE 엣지 추가 (동시산정)
(p1:Procedure)-[:CONCURRENT_RULE {
  decision: "ALLOW" | "EXCLUDE" | "REDUCE",
  rate: Float,           // 0.5 = 50% 산정
  condition: String,     // "동일 마취하 연속 수술"
  source_doc: String,    // "고시 제2024-XX호"
  effective_date: Date
}]->(p2:Procedure)
```

#### 변경 불필요

- Disease, Procedure, Drug, Test → 기존 구조 유지
- Guideline → Benefit/Criteria 추출 소스로 활용

---

## 5. 동시산정 규칙 데이터

### 5.1 cg_parsed 분석 결과

| 소스 | 동시산정 관련 |
|------|-------------|
| 심사지침 | 44건 |
| **고시** | **647건** |

### 5.2 동시산정 패턴 분류

| 패턴 | 건수 | Neo4j decision |
|------|------|----------------|
| 50% 산정 | 131건 | REDUCE (rate: 0.5) |
| 100% + 50% | 74건 | REDUCE (rate: 0.5) |
| 주된 수술/검사 | 69건 | EXCLUDE (부수술) |
| 별도 산정 불가 | 68건 | EXCLUDE |
| 일련의 과정 | 27건 | EXCLUDE |
| 동시 산정 불가 | 15건 | EXCLUDE |

### 5.3 동시산정 엣지 설계

```cypher
// 별도 산정 불가 (EXCLUSIVE)
CREATE (p1:Procedure {name: "심초음파"})
       -[:CONCURRENT_RULE {
         decision: "EXCLUDE",
         condition: "심초음파 실시 시 산소포화도 검사 별도 산정 불가",
         source_doc: "고시 제2024-195호"
       }]->
       (p2:Procedure {name: "산소포화도 검사"})

// 50% 감액 (REDUCE)
CREATE (p1:Procedure {name: "주수술"})
       -[:CONCURRENT_RULE {
         decision: "REDUCE",
         rate: 0.5,
         condition: "동일 마취하 연속 수술",
         source_doc: "산정지침(5)"
       }]->
       (p2:Procedure {name: "부수술"})
```

### 5.4 동시산정은 NER이 아닌 규칙 DB

> **핵심**: 동시산정 관계는 NER로 추출하지 않음

| 역할 | 담당 |
|------|------|
| NER | 텍스트에서 "추나요법", "분구침술" 엔티티 추출 |
| **규칙 DB** | Procedure ↔ Procedure 간 CONCURRENT_RULE 엣지 관리 |
| 파싱 | cg_parsed 고시 647건에서 규칙 추출 → Neo4j |

---

## 6. 사용자 질의 유형별 매핑

| 질의 패턴 | 필요 엔티티 | 필요 관계 |
|----------|------------|----------|
| "~급여기준" | Procedure/Drug/Exam | → Guideline (HAS_CRITERIA) |
| "~인정상병" | Drug + Disease | INDICATED_FOR |
| "~상병코드" | Disease | kcd_code 속성 |
| "동시산정 가능?" | Procedure + Procedure | CONCURRENT_RULE |
| "~재료 청구" | Procedure + Material | USES |

---

## 7. 데이터 소스

| 소스 | 위치 | 용도 |
|------|------|------|
| 사용자 질의 | `data/user_qa/` | 평가용 + 패턴 분석 (202건) |
| khima 문제집 | `data/khima/book_ocr/` | 실버셋 (제도/청구 용어) |
| cg_parsed 고시 | `data/cg_parsed/고시_20251101.xlsx` | **동시산정 규칙 647건** |
| cg_parsed 심사지침 | `data/cg_parsed/심사지침_20251101.xlsx` | 심사지침 228건 |
| Gazetteer | `data/pharmalex/` 등 | Drug, Code 사전 |

---

## 8. 결론

### 최종 NER 스키마 (8개)

```
Disease, Procedure, Drug, Exam, Code, Material, Benefit, Criteria
```

### Neo4j 호환성

| 항목 | 상태 |
|------|------|
| 기존 노드 (Drug, Disease, Procedure, Test) | ✅ 호환 |
| Material 노드 | 🔧 추가 필요 |
| CONCURRENT_RULE 엣지 | 🔧 추가 필요 |
| Code | ⚠️ 속성으로 처리 (code_type 구분) |

### 구조 변경 요약

```
Neo4j 추가 작업:
1. Material 노드 타입 신규 생성
2. CONCURRENT_RULE 엣지 타입 추가
3. cg_parsed 고시 647건 → 동시산정 규칙 파싱/적재
```

---

## 9. 다음 단계: E2E 파이프라인 검증

### 9.1 핵심 원칙

> **50개 라벨링은 'NER 학습'용이 아니라 'E2E 파이프라인 검증'용**

| 구분 | 목적 | 산출물 |
|------|------|--------|
| 파일럿 라벨링 | 스키마 검증, 경계 케이스 발견 | 라벨링된 50문장 |
| Entity Linking | NER→Neo4j 연결 검증 | 링킹 결과 + 미매칭 분석 |
| Cypher 템플릿 | 질의 패턴→쿼리 변환 검증 | 템플릿 10개 |

### 9.2 통합 작업 단위

```
┌─────────────────────────────────────────────────────────┐
│              E2E 파이프라인 검증 (한 덩어리)              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌───────────────────────────────────────────────────┐ │
│  │ Step 1: 파일럿 라벨링 (50개)                       │ │
│  │ - 소스: 문제집 (data/khima/book_ocr/)             │ │
│  │ - 산출물: 엔티티 + 링킹 결과 포함                  │ │
│  └───────────────────┬───────────────────────────────┘ │
│                      │                                  │
│                      ▼                                  │
│  ┌───────────────────────────────────────────────────┐ │
│  │ Step 2: Top Intent / Top 엔티티 조합 분석          │ │
│  │ - 빈도 분석: 어떤 라벨 조합이 많은가               │ │
│  │ - 패턴 추출: 질의 유형별 필요 엔티티               │ │
│  └───────────────────┬───────────────────────────────┘ │
│                      │                                  │
│                      ▼                                  │
│  ┌───────────────────────────────────────────────────┐ │
│  │ Step 3: Cypher 템플릿 10개 확정                    │ │
│  │ - 가장 빈번한 조합 우선                           │ │
│  │ - Neo4j 스키마와 매핑                             │ │
│  └───────────────────────────────────────────────────┘ │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 9.3 Entity Linking 전략

#### Drug 도메인 우선

| 이유 | 설명 |
|------|------|
| 사전 풍부 | 150만+ 엔트리 (pharmalex_unity + 약가마스터) |
| 빈도 높음 | 사용자 질의에서 가장 많이 언급 |
| 가치 높음 | 약제 급여기준 조회가 핵심 유즈케이스 |
| 검증 용이 | 표준화된 코드 체계 (EDI, ATC) |

#### Entity Linking 모듈 구조

```
┌─────────────────────────────────────────────────────────┐
│                  Entity Linking 모듈                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Input: NER 추출 결과 (text, label, span)               │
│                      │                                  │
│                      ▼                                  │
│  ┌─────────────────────────────────────────────────┐   │
│  │ 1. 사전 매칭 (Gazetteer Lookup)                  │   │
│  │    - 정확 매칭 (exact match)                     │   │
│  │    - 부분 매칭 (contains/prefix)                 │   │
│  └─────────────────────┬───────────────────────────┘   │
│                        │                                │
│                        ▼                                │
│  ┌─────────────────────────────────────────────────┐   │
│  │ 2. 유사도 스코어링 (Similarity)                  │   │
│  │    - 편집 거리 (Levenshtein)                     │   │
│  │    - 음절 유사도                                 │   │
│  │    - 의미 유사도 (임베딩 기반, optional)         │   │
│  └─────────────────────┬───────────────────────────┘   │
│                        │                                │
│                        ▼                                │
│  ┌─────────────────────────────────────────────────┐   │
│  │ 3. 임계값 필터링 (Threshold)                     │   │
│  │    - score >= 0.8: 자동 링킹                     │   │
│  │    - 0.5 <= score < 0.8: 후보 제시               │   │
│  │    - score < 0.5: 미매칭                         │   │
│  └─────────────────────┬───────────────────────────┘   │
│                        │                                │
│                        ▼                                │
│  Output: Neo4j 노드 ID 또는 후보 리스트                 │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 9.4 작업 순서

| 순서 | 작업 | 소스/도구 | 산출물 |
|------|------|----------|--------|
| 1 | 문제집에서 50문장 선정 | `data/khima/book_ocr/` | 원문 50개 |
| 2 | NER 라벨링 (8개 라벨) | 수동 + GPT 보조 | 라벨링 JSONL |
| 3 | Drug 엔티티 링킹 | pharmalex_unity 사전 | 링킹 결과 |
| 4 | 질의 패턴 분석 | 라벨링 결과 집계 | Top 엔티티 조합 |
| 5 | Cypher 템플릿 작성 | Neo4j 스키마 참조 | 템플릿 10개 |
| 6 | E2E 검증 | 파이프라인 실행 | 오류/개선점 |

### 9.5 완료 기준

| 항목 | 완료 조건 |
|------|----------|
| 라벨링 | 50문장 × 8라벨 스키마 적용 완료 |
| 링킹 | Drug 엔티티 80%+ Neo4j 노드 매칭 |
| 템플릿 | 상위 10개 질의 패턴 커버 |
| 파이프라인 | 텍스트→NER→링킹→Cypher→결과 흐름 검증 |

---

## 10. 후속 작업 (파이프라인 검증 후)

| 순서 | 작업 | 의존성 |
|------|------|--------|
| 1 | Neo4j Material 노드 추가 | 파이프라인 검증 완료 |
| 2 | Neo4j CONCURRENT_RULE 엣지 추가 | Material 노드 |
| 3 | cg_parsed 647건 규칙 파싱/적재 | CONCURRENT_RULE 엣지 |
| 4 | 대규모 실버셋 생성 (5,000개) | 라벨링 가이드라인 확정 |
| 5 | NER 모델 풀 파인튜닝 | 실버셋 + 인코더 선택 |
| 6 | 전체 시스템 통합 | 모든 컴포넌트 |

---

## 11. 관련 문서

| 문서 | 경로 |
|------|------|
| Neo4j 스키마 | `meta/designs/neo4j_schema_diagram.md` |
| Knowledge Graph 계획 | `project/knowledge_graph/docs/plans/knowledge_graph_neo4j_plan.md` |
| 전체 로드맵 | `project/ner/docs/plans/ner_project_roadmap.md` |
| 사용자 질의 | `data/user_qa/사용자질의응답_251209.xlsx` |
