# Cypher 템플릿 검증 리포트

**검증일**: 2026-01-07

## 현재 구현 상태

### 구현된 쿼리 패턴 (8개)

| # | 패턴 | 메서드 | E2E 사용 | 비고 |
|---|------|--------|----------|------|
| 1 | Disease → Drug | `disease_to_drugs(kcd_code, line)` | ✅ | 차수 필터 지원 |
| 2 | Disease → Drug (이름) | `disease_to_drugs_by_name(name)` | ✅ | 부분 매칭 |
| 3 | Drug → Disease | `drug_to_diseases(atc_code)` | ✅ | |
| 4 | Drug → Disease (이름) | `drug_to_diseases_by_name(name)` | ✅ | |
| 5 | Drug → Biomarker | `drug_to_biomarkers(atc_code)` | ❌ | E2E 미사용 |
| 6 | Biomarker → Test | `biomarker_to_tests(name)` | ✅ | |
| 7 | Full Path | `full_treatment_path(kcd_code)` | ❌ | E2E 미사용 |
| 8 | Cancer → Treatment | `cancer_to_treatments(name)` | ❌ | Cancer 노드 필요 |

### 보조 쿼리 (2개)

| 메서드 | 용도 |
|--------|------|
| `drug_usage_stats(name)` | 약물 사용 통계 |
| `regimen_details(id)` | Regimen 상세 정보 |

## 문제점

### 1. E2E에서 미사용 쿼리
- `drug_to_biomarkers()`: 약물 입력 시 표적 바이오마커도 조회 가능
- `full_treatment_path()`: Disease 입력 시 전체 경로 한번에 조회 가능

### 2. 누락된 역방향 쿼리
| 필요한 쿼리 | 용도 |
|-------------|------|
| `biomarker_to_drugs()` | "HER2 양성이면 어떤 약 쓰나요?" |
| `test_to_biomarkers()` | "이 검사로 뭘 측정하나요?" |

### 3. 헬퍼 함수 중복
- `generate_queries_from_entities()` 함수가 이미 존재
- E2E에서 동일 로직 직접 구현 → 중복

## 개선 계획

### Phase 1: 역방향 쿼리 추가
```python
def biomarker_to_drugs(self, biomarker_name: str) -> CypherQuery:
    """바이오마커 표적 약물 조회"""
    # Biomarker ←[TARGETS]- Drug

def test_to_biomarkers(self, edi_code: str) -> CypherQuery:
    """검사로 측정 가능한 바이오마커 조회"""
    # Test ←[TESTED_BY]- Biomarker
```

### Phase 2: E2E 파이프라인 개선
- Drug 엔티티 처리 시 `drug_to_biomarkers()` 추가 호출
- Disease 엔티티 처리 시 옵션으로 `full_treatment_path()` 사용

### Phase 3: 코드 정리
- E2E에서 `generate_queries_from_entities()` 헬퍼 함수 활용

## 그래프 스키마 참조

```
Disease -[TREATED_BY]-> Regimen -[INCLUDES]-> Drug
                                              |
                                              v
                                         [TARGETS]
                                              |
                                              v
                                          Biomarker
                                              |
                                              v
                                         [TESTED_BY]
                                              |
                                              v
                                            Test
```
