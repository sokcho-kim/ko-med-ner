# GLiNER2 테스트 결과

> 테스트일: 2026-01-06

---

## 개요

GLiNER2를 로컬에 설치하고 한국어 의료 NER 태스크에 적용 가능한지 테스트함.

---

## 환경 설정

### 설치

```bash
cd C:/Jimin/GLiNER2
uv venv .venv
uv pip install -e . --python .venv/Scripts/python.exe
```

### 설치된 패키지

- `gliner2==1.2.3`
- `gliner==0.2.24` (의존성)
- 기반 모델: `microsoft/deberta-v3-base` (205M 파라미터)

---

## 테스트 결과

### 1. 영어 테스트 - 성공

```python
text = "Apple CEO Tim Cook announced the new iPhone 15 in Cupertino."
entities = extractor.extract_entities(text, ['company', 'person', 'product', 'location'])
```

**결과:**
```python
{
    'company': ['Apple'],
    'person': ['Tim Cook'],
    'product': ['iPhone 15'],
    'location': ['Cupertino']
}
```

✅ **완벽하게 작동**

---

### 2. 한국어 의료 테스트 - 부분 성공

#### 테스트 데이터

| # | 텍스트 |
|---|--------|
| 1 | 제2형 당뇨병 환자의 간 기능이 악화되어 인슐린 주사로 치료를 변경했다. |
| 2 | 고혈압으로 인한 심장 기능 저하가 관찰되어 항고혈압제를 투여하였다. |
| 3 | 폐렴 환자에게 항생제 치료를 시작했으며 폐 기능 검사를 예정하였다. |
| 4 | 만성 신부전으로 투석 치료를 받고 있는 환자의 신장 상태를 모니터링 중이다. |

#### GLiNER2 (fastino/gliner2-base-v1) 결과

| Sample | Disease | Body | Treatment |
|--------|---------|------|-----------|
| 1 | 제2형 당뇨병 ✓ | ✗ | ✗ |
| 2 | ✗ | ✗ | ✗ |
| 3 | 폐렴 ✓ | 폐 ✓ | ✗ |
| 4 | ✗ | ✗ | ✗ |

**인식률: 매우 낮음 (4/12 = 33%)**

#### GLiNER v2.1 (urchade/gliner_multi-v2.1) 결과 비교

| Sample | Disease | Body | Treatment |
|--------|---------|------|-----------|
| 1 | 제2형 당뇨병 ✓ | 간 ✓ | 인슐린 주사 ✓ |
| 2 | 고혈압 ✓ | 심장 ✓ | 항고혈압제 ✓ |
| 3 | 폐렴 ✓ | ✗ | 항생제 치료 ✓ |
| 4 | 만성 신부전 ✓ | 신장 ✓ | 투석 치료 ✓ |

**인식률: 높음 (11/12 = 92%)**

---

## 모델 비교

| 항목 | GLiNER2 | GLiNER v2.1 |
|------|---------|-------------|
| 패키지 | `gliner2` | `gliner` |
| 모델 | `fastino/gliner2-base-v1` | `urchade/gliner_multi-v2.1` |
| 인코더 | DeBERTa-v3-base | XLM-RoBERTa |
| 파라미터 | 205M | 560M |
| 영어 성능 | ✅ 우수 | ✅ 우수 |
| 한국어 성능 | ❌ 부족 | ✅ 우수 |
| 스키마 인터페이스 | ✅ 지원 (설명 포함) | ❌ 미지원 |
| 파인튜닝 필요 | 선택적 (LoRA) | 선택적 |

---

## 핵심 발견

### GLiNER2의 장점

1. **스키마 기반 인터페이스**: 엔티티 설명을 자연어로 제공 가능
2. **다중 태스크**: NER, Classification, Relation Extraction, JSON Extraction 통합
3. **CPU 최적화**: GPU 없이도 빠른 추론
4. **파인튜닝 불필요**: Zero-shot으로 사용 가능 (영어 한정)

### GLiNER2의 한계

1. **한국어 지원 부족**: 기본 모델(DeBERTa-v3-base)이 영어 중심
2. **다국어 모델 부재**: 현재 한국어 특화 GLiNER2 모델 없음

### 권장 사항

| 언어 | 권장 모델 |
|------|----------|
| 영어 | GLiNER2 (`fastino/gliner2-base-v1`) |
| 한국어/다국어 | GLiNER v2.1 (`urchade/gliner_multi-v2.1`) |

---

## 다음 단계

1. **한국어 NER**: GLiNER v2.1 (다국어) 모델 사용
2. **영어 NER**: GLiNER2의 스키마 인터페이스 활용
3. **파인튜닝 고려**: 한국어 의료 도메인에 맞춘 LoRA 어댑터 학습

---

## 코드 예시

### GLiNER2 (영어)

```python
from gliner2 import GLiNER2

extractor = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

# 설명 포함 방식 (정확도 향상)
entities = extractor.extract_entities(text, {
    "drug": "Pharmaceutical drugs, medications",
    "disease": "Medical conditions, illnesses"
})
```

### GLiNER v2.1 (한국어/다국어)

```python
from gliner import GLiNER

model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")

entities = model.predict_entities(
    text,
    ["Disease", "Body", "Treatment"],
    threshold=0.3
)
```

---

## 참고

- GLiNER2 GitHub: https://github.com/urchade/GLiNER (GLiNER2 브랜치)
- 모델: https://huggingface.co/fastino/gliner2-base-v1
- 다국어 모델: https://huggingface.co/urchade/gliner_multi-v2.1
