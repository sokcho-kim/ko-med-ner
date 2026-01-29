# GLiNER2: Unified Schema-Based Information Extraction

> 출처: https://github.com/fastino-ai/GLiNER2
> 스크래핑 일시: 2025-12-24

---

GLiNER2 is an open-source framework that consolidates entity recognition, text classification, structured data extraction, and relation extraction into a single 205M parameter model optimized for CPU inference.

## Installation & Quick Start

Install via pip:

```bash
pip install gliner2
```

Basic usage:

```python
from gliner2 import GLiNER2

extractor = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
text = "Apple CEO Tim Cook announced iPhone 15 in Cupertino."
result = extractor.extract_entities(text,
    ["company", "person", "product", "location"])
```

## Core Capabilities

### Four Main Tasks

| Task | Description |
|------|-------------|
| Entity Extraction | NER with optional descriptions |
| Text Classification | single/multi-label |
| Structured Data Extraction | JSON parsing |
| Relation Extraction | entity relationships |

### Key Features

- "One Model, Four Tasks" handling
- CPU-first design; no GPU required
- Local processing; no external dependencies
- Confidence scores and character span support
- Batch processing capabilities

## Available Models

| Model | Parameters | Description |
|-------|------------|-------------|
| fastino/gliner2-base-v1 | 205M | 기본 모델 |
| fastino/gliner2-large-v1 | 340M | 대형 모델 |
| GLiNER XL 1B | - | API 기반 |

## Training Custom Models

Supports JSONL training data format with input/output fields.

### Features

- Standard fine-tuning
- LoRA adapters (~5MB vs ~450MB)
- Parameter-efficient training
- Validation and early stopping

### Example Training Workflow

```python
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

config = TrainingConfig(output_dir="./output", num_epochs=10)
trainer = GLiNER2Trainer(model, config)
trainer.train(train_data=examples)
```

## Multi-task Schema Example

```python
schema = (extractor.create_schema()
    .entities(["person", "company", "product"])
    .classification("sentiment", ["positive", "negative", "neutral"])
    .relations(["works_for", "manufactures"])
    .structure("product_info")
        .field("name", dtype="str")
        .field("price", dtype="float")
)

result = extractor.extract(text, schema)
```

## Advanced Features

### Regex Validators

```python
schema = (extractor.create_schema()
    .entities(["email"])
    .regex_validator("email", r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
)
```

### Confidence Thresholds

```python
result = extractor.extract(text, schema, confidence_threshold=0.8)
```

### Batch Processing

```python
texts = ["Text 1...", "Text 2...", "Text 3..."]
results = extractor.extract_batch(texts, schema)
```

## License

Apache License 2.0

## Citation

Published at EMNLP 2025 Demonstrations.

```bibtex
@inproceedings{gliner2-2025,
    title = "GLiNER2: Unified Schema-Based Information Extraction",
    booktitle = "Proceedings of EMNLP 2025: System Demonstrations",
    year = "2025"
}
```

---

## 의료 도메인 활용 예시 (커스텀)

```python
# 의료 NER + 관계 추출 스키마
schema = (extractor.create_schema()
    .entities(["약제명", "질병명", "고시번호", "행위코드", "치료재료"])
    .relations(["TREATS", "APPROVES", "APPLIES_TO", "AMENDS"])
    .structure("심사사례")
        .field("환자정보", dtype="str")
        .field("심사결과", dtype="str")
        .field("결정사유", dtype="str")
)
```

## GLiNER2 vs GLiNER1 비교

| 항목 | GLiNER1 | GLiNER2 |
|------|---------|---------|
| 기능 | NER만 | NER + 분류 + 관계 + 구조화 |
| 파라미터 | ~1.33GB | 205M / 340M |
| 멀티태스크 | ❌ | ✅ |
| LoRA 지원 | ❌ | ✅ |
| CPU 최적화 | △ | ✅ |
