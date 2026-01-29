# GLiNER2 논문 분석

> 원본: `GLiNER2_ An Efficient Multi-Task Information Extraction System with Schema-Driven Interface.pdf`
> 출처: arXiv:2507.18546v1 [cs.CL] 24 Jul 2025
> 분석일: 2025-12-24

---

## 기본 정보

| 항목 | 내용 |
|------|------|
| 제목 | GLiNER2: An Efficient Multi-Task Information Extraction System with Schema-Driven Interface |
| 저자 | Urchade Zaratiana, Gil Pasternak, Oliver Boyd, George Hurn-Maloney, Ash Lewis |
| 소속 | Fastino AI |
| 이메일 | {uz,gil,o8,g,ash}@fastino.ai |
| arXiv | 2507.18546v1 |
| 발표일 | 2025-07-24 |
| 학회 | EMNLP 2025 System Demonstrations |
| 라이선스 | Apache 2.0 |
| GitHub | https://github.com/fastino-ai/GLiNER2 |

---

## Abstract 요약

Information Extraction(IE)은 NLP의 근본적인 작업이지만, 기존 솔루션은 태스크별로 특화된 모델이 필요하거나 계산 비용이 높은 LLM에 의존한다. GLiNER2는 **단일 205M 파라미터 모델**로 NER, 텍스트 분류, 계층적 구조 추출을 통합하는 프레임워크를 제시한다.

---

## 1. Introduction - 문제 제기

### LLM의 배포 문제점

1. **GPU 필요**: Llama-2-7b 같은 "작은" 모델도 GPU 가속 필요
2. **프라이버시**: API 사용 시 PII, 금융 데이터 등 민감 정보 노출 위험
3. **규정 준수**: GDPR, HIPAA 등 on-premises 배포 요구
4. **비용**: API 비용이 연구자, 스타트업, 개발도상국에 부담

### GLiNER의 성공과 한계

- GLiNER: Zero-shot NER을 위한 소형 트랜스포머 인코더
- 성공: CPU에서 효율적 실행, PII 마스킹 도메인에서 인기
- 한계: NER만 지원, 다른 IE 태스크는 별도 모델 필요

### 파생 모델들 (Fragmentation 문제)

| 모델 | 용도 |
|------|------|
| GLiREL | 관계 추출 |
| GLiClass | Zero-shot 텍스트 분류 |
| GLiNER-BioMed | 바이오메디컬 NER |
| OpenBioNER | 경량 바이오메디컬 NER |
| GLiDRE | 프랑스어 문서 레벨 관계 추출 |

→ **문제**: 각각 별도 모델 개발/배포 필요

---

## 2. System Design

### Core Innovation: Unified Input Formulation

```
[Task Prompt] ⊕ [SEP] ⊕ [Input Text]
```

- `[Task Prompt]`: 추출할 것 지정 (엔티티 타입, 분류 레이블 등)
- `[SEP]`: 구분자 토큰
- `[Input Text]`: 분석할 텍스트

### 4가지 통합 태스크

#### 1. Entity Recognition
- Label description 지원 (자연어 정의로 의미적 이해 향상)
- Nested/Overlapping spans 지원

#### 2. Hierarchical Structure Extraction
- Parent-child 관계 캡처
- 복잡한 중첩 정보 추출

#### 3. Text Classification
- Single-label / Multi-label 지원
- Label description 지원

#### 4. Task Composition
- **단일 forward pass**에서 여러 추출 태스크 동시 실행
- 공유된 contextual understanding

---

## 3. Experiments

### 3.1 Training Data

| 구분 | 건수 | 비고 |
|------|------|------|
| Real-world | 135,698 | News, Wikipedia, Legal, PubMed, ArXiv |
| Synthetic | 118,636 | Email, SMS, Resume, Social media, E-commerce |
| **Total** | **254,334** | GPT-4o로 자동 어노테이션 |

### 3.2 Zero-shot Classification 성능 (Table 2)

| Dataset | Task Type | # Labels | GPT-4o | GLiClass | DeBERTa-v3 | **GLiNER2** |
|---------|-----------|----------|--------|----------|------------|-------------|
| SNIPS | Intent | 7 | 0.97 | 0.80 | 0.77 | **0.83** |
| Banking77 | Intent | 77 | 0.78 | 0.21 | 0.42 | **0.70** |
| Amazon Intent | Intent | 31 | 0.72 | 0.51 | 0.59 | 0.53 |
| SST-2 | Sentiment | 2 | 0.94 | 0.90 | 0.92 | 0.86 |
| IMDB | Sentiment | 2 | 0.95 | 0.92 | 0.89 | **0.87** |
| AG News | Topic | 4 | 0.85 | 0.68 | 0.68 | **0.74** |
| 20 Newsgroups | Topic | 20 | 0.68 | 0.36 | 0.54 | 0.49 |
| **Average** | - | - | 0.84 | 0.63 | 0.69 | **0.72** |

**결론**: 오픈소스 중 최고 평균 정확도 (0.72)

### 3.3 Zero-shot NER 성능 - CrossNER (Table 3)

| Dataset | GPT-4o | GLiNER-M | GLiNER2 |
|---------|--------|----------|---------|
| AI | 0.547 | 0.518 | 0.526 |
| Literature | 0.561 | 0.597 | 0.564 |
| Music | 0.736 | 0.694 | 0.632 |
| Politics | 0.632 | 0.686 | 0.679 |
| Science | 0.518 | 0.581 | 0.547 |
| **Average** | **0.599** | 0.615 | 0.590 |

**결론**: GPT-4o와 거의 동등 (0.590 vs 0.599), 멀티태스크 모델임을 감안하면 인상적

### 3.4 CPU Latency (Table 4)

| #Labels | GPT-4o | DeBERTa | GLiClass | GLiNER2 |
|---------|--------|---------|----------|---------|
| 5 | 358ms | 1714ms | 137ms | **130ms** |
| 10 | 382ms | 3404ms | 131ms | 132ms |
| 20 | 425ms | 6758ms | 140ms | 163ms |
| 50 | 463ms | 16897ms | 190ms | 208ms |
| **Speedup** | 1.00× | 0.10× | 2.75× | **2.62×** |

**결론**:
- GPT-4o 대비 **2.6배 빠름**
- DeBERTa 대비 **26배 빠름** (DeBERTa는 레이블당 별도 forward pass)
- 레이블 수에 관계없이 **단일 forward pass**

---

## 4. Artifacts

### 4.1 Python Package

```python
# 설치
pip install gliner2

# 로드
from gliner2 import GLiNER2
extractor = GLiNER2.from_pretrained("gliner/gliner2-base")
```

### Named Entity Recognition

```python
text = "Apple Inc. CEO Tim Cook announced new products in Cupertino."
entities = ["company", "person", "location", "product"]
results = extractor.extract_entities(text, entities)
# {'entities': {'company': ['Apple Inc.'],
#               'person': ['Tim Cook'],
#               'location': ['Cupertino']}}

# Label description 사용
entity_descriptions = {
    "company": "Business organizations and corporations",
    "person": "Names of individuals including executives",
    "location": "Geographical places including cities"
}
results = extractor.extract_entities(text, entity_descriptions)
```

### Hierarchical Structure Extraction

```python
product_schema = {
    "product": [
        "name::str::Product name and model",
        "price::str::Product cost",
        "features::list::Key product features",
        "category::[electronics|software|hardware]::str"
    ]
}
results = extractor.extract_json(text, product_schema)
```

### Text Classification

```python
# 단순 분류
labels = ["positive", "negative", "neutral"]
results = extractor.classify_text(text, {"sentiment": labels})

# Multi-label with descriptions
tasks = {
    "aspects": {
        "labels": ["acting", "plot", "visuals", "music"],
        "multi_label": True,
        "descriptions": {
            "acting": "Quality of character performances",
            "plot": "Story structure and narrative"
        }
    }
}
results = extractor.classify_text(text, tasks)
```

### Task Composition (핵심 기능)

```python
# 단일 forward pass에서 NER + Classification + Structure Extraction
schema = (extractor.create_schema()
    # NER
    .entities(["person", "company", "product", "location", "price"])

    # Classification
    .classification("sentiment", ["positive", "negative", "neutral"])
    .classification("urgency", ["low", "medium", "high"])

    # Structure Extraction
    .structure("product_info")
        .field("name", dtype="str", description="Product name")
        .field("price", dtype="str", description="Product cost")
        .field("features", dtype="list", description="Key features")
)

results = extractor.extract(text, schema)
```

---

## Appendix A: Architecture Details

### Special Token Vocabulary

| Token | 역할 |
|-------|------|
| `[P]` (Prompt) | 태스크 시작 |
| `[E]` (Entity) | NER 엔티티 타입 |
| `[C]` (Child) | 계층 구조 자식 필드 |
| `[L]` (Label) | 분류 레이블 |
| `[SEP]` | 세그먼트 구분 |

### NER Input Format

```
[P] entities ([E] e1 [E] e2 ... [E] en) [SEP] x1, x2, ..., xN
```

**Scoring**:
```
score(si, ej) = sim(h_si, h_ej)
```
- `h_si`: span representation
- `h_ej`: entity type embedding
- `sim`: dot product + sigmoid

### Hierarchical Structure Extraction

```
[P] parent ([C] a1 [C] a2 ... [C] am) [SEP] x1, x2, ..., xN
```

**2단계 프로세스**:
1. MLP가 `[P]` 토큰으로 인스턴스 개수 K 예측 (0-19 클래스)
2. K개의 인스턴스별로 각 속성 추출

### Text Classification

```
[P] task ([L] l1 [L] l2 ... [L] lk) [SEP] x1, x2, ..., xN
```

**Scoring**:
```
logit_i = MLP(h_li)
```
- Single-label: softmax
- Multi-label: sigmoid

---

## Appendix B: Experimental Setup

### Training Hyperparameters (Table 5)

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 5 |
| Optimizer | AdamW |
| LR (backbone) | 1×10⁻⁵ |
| LR (task layers) | 2×10⁻⁵ |
| Weight decay | 0.01 |
| Warmup steps | 1,000 |
| Gradient clipping | 1.0 |

### Training Data Distribution (Table 6)

| Domain | Count |
|--------|-------|
| Law | 19,798 |
| PubMed | 16,400 |
| Wikipedia | 17,909 |
| ArXiv | 7,135 |
| News | 74,456 |
| Synthetic | 118,636 |
| **Total** | **254,334** |

---

## 의료 도메인 활용 시사점

### 강점

1. **PubMed 데이터 포함**: 16,400건의 의료 논문으로 학습
2. **Label description**: 의료 엔티티에 상세 설명 추가 가능
3. **계층 구조 추출**: 심사사례의 복잡한 구조 추출에 적합
4. **CPU 실행**: GPU 없이도 빠른 추론 가능
5. **프라이버시**: 로컬 실행으로 민감한 의료 데이터 보호

### 관련 연구

- **GLiNER-BioMed** (Yazdani et al., 2025): 바이오메디컬 특화 버전
- **OpenBioNER** (Cocchieri et al., 2025): 경량 바이오메디컬 NER

### 의료 NER 스키마 예시

```python
# 의료 도메인 NER + 관계 + 구조 추출
schema = (extractor.create_schema()
    # 엔티티 추출
    .entities({
        "약제명": "의약품의 상품명 또는 성분명",
        "질병명": "질환, 증상, 진단명",
        "고시번호": "건강보험 요양급여 고시 번호 (예: 제2024-177호)",
        "행위코드": "의료 행위 코드 (예: E7070, C8904)",
        "치료재료": "스텐트, 인공관절 등 의료 재료"
    })

    # 분류
    .classification("심사결과", ["승인", "불승인", "조정"])
    .classification("카테고리", ["약제", "행위", "치료재료", "기결정"])

    # 구조 추출
    .structure("심사사례")
        .field("환자정보", dtype="str")
        .field("청구내역", dtype="str")
        .field("결정사유", dtype="str")
        .field("관련고시", dtype="list")
)
```

---

## 인용

```bibtex
@inproceedings{gliner2-2025,
    title = "GLiNER2: An Efficient Multi-Task Information Extraction System with Schema-Driven Interface",
    author = "Zaratiana, Urchade and Pasternak, Gil and Boyd, Oliver and Hurn-Maloney, George and Lewis, Ash",
    booktitle = "Proceedings of EMNLP 2025: System Demonstrations",
    year = "2025"
}
```

---

## 참고 링크

- GitHub: https://github.com/fastino-ai/GLiNER2
- 원본 GLiNER: https://github.com/urchade/GLiNER
- GLiREL: https://github.com/jackboyla/GLiREL
- GLiClass: https://huggingface.co/knowledgator/GLiClass
