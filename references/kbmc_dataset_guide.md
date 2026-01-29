# KBMC 데이터셋 가이드

> **출처**: https://huggingface.co/datasets/SungJoo/KBMC
> **라이센스**: Apache 2.0 (상업용/학술용 가능)
> **최종 확인**: 2026-01-05

---

## 개요

| 항목 | 내용 |
|------|------|
| 이름 | Korean Bio-Medical Corpus (KBMC) |
| 용도 | 한국어 의료 NER (Named Entity Recognition) |
| 크기 | 6,150 문장, 1.52 MB |
| 형식 | Parquet, CSV |
| 저자 | Sungjoo Byun et al. (SNU, KAIST) |

---

## 데이터 구조

### 컬럼

| 컬럼명 | 설명 | 예시 |
|--------|------|------|
| **Sentence** | 의료 한국어 문장 (공백 토큰화) | `전신 적 다한증 은 신체 전체 에...` |
| **Tags** | BIO 태그 (공백 구분) | `Disease-B Disease-I Disease-I O O O...` |

### 라벨 체계 (7개)

| 라벨 | 설명 | 개수 | 비율 |
|------|------|------|------|
| **O** | Outside (개체명 아님) | 대부분 | ~80% |
| **Disease-B** | 질병 시작 | 10,595 | 6.9% |
| **Disease-I** | 질병 내부 | 10,089 | 6.6% |
| **Body-B** | 신체부위 시작 | 5,215 | 3.4% |
| **Body-I** | 신체부위 내부 | 1,158 | 0.8% |
| **Treatment-B** | 치료법 시작 | 1,193 | 0.8% |
| **Treatment-I** | 치료법 내부 | 839 | 0.5% |

### 예시 데이터

```
Sentence: 전신 적 다한증 은 신체 전체 에 힘 이 빠져서 일상 생활 이 어려워지는 질환 으로 , 근육 통증 과 무기 력 감 이 동반 됩니다 .
Tags:     Disease-B Disease-I Disease-I O O O O O O O O O O O O O O Disease-B Disease-I O Disease-B Disease-I Disease-I O O O O
```

---

## 다운로드 방법

### 1. Hugging Face Datasets (권장)

```python
from datasets import load_dataset

# 데이터셋 로드
dataset = load_dataset("SungJoo/KBMC")

# 확인
print(f"Train size: {len(dataset['train'])}")
print(dataset['train'][0])
```

### 2. Pandas

```python
import pandas as pd

# Parquet 직접 로드
df = pd.read_parquet("hf://datasets/SungJoo/KBMC/default/train/0000.parquet")
print(df.head())
```

### 3. CLI

```bash
# huggingface-cli 사용
huggingface-cli download SungJoo/KBMC --repo-type dataset
```

---

## 전처리 예시

### 토큰-라벨 정렬

```python
def parse_kbmc_sample(sample):
    """KBMC 샘플을 토큰-라벨 쌍으로 변환"""
    tokens = sample['Sentence'].split()
    tags = sample['Tags'].split()

    assert len(tokens) == len(tags), "토큰-태그 길이 불일치"

    return {
        'tokens': tokens,
        'tags': tags
    }

# 사용
parsed = parse_kbmc_sample(dataset['train'][0])
for token, tag in zip(parsed['tokens'], parsed['tags']):
    if tag != 'O':
        print(f"{token} -> {tag}")
```

### 라벨 매핑

```python
# 라벨 → ID 매핑
label_list = ['O', 'Disease-B', 'Disease-I', 'Body-B', 'Body-I', 'Treatment-B', 'Treatment-I']
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

# 태그 시퀀스를 ID로 변환
def tags_to_ids(tags):
    return [label2id[tag] for tag in tags.split()]
```

---

## 학습 설정 (기존 노트북 참고)

```python
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)

# 모델 & 토크나이저
model_name = "monologg/koelectra-base-v3-discriminator"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

# 학습 설정
training_args = TrainingArguments(
    output_dir="./kbmc-ner",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)
```

---

## 기대 성능 (논문 기준)

| 모델 | Disease F1 | Body F1 | Treatment F1 | Avg F1 |
|------|------------|---------|--------------|--------|
| KM-BERT | 98.04 | 98.13 | 98.53 | 88.53 |
| KoELECTRA | 98.05 | 97.72 | 96.56 | 88.86 |
| KoBERT | 98.25 | 98.22 | 98.18 | 88.70 |

---

## 인용

```bibtex
@article{byun2024kbmc,
    title={Korean Bio-Medical Corpus (KBMC) for Medical Named Entity Recognition},
    author={Byun, Sungjoo and Hong, Jiseung and Park, Sumin and Jang, Dongjun and Seo, Jean and Kim, Minseok and Oh, Chaeyoung and Shin, Hyopil},
    journal={arXiv preprint arXiv:2403.16158},
    year={2024}
}
```

---

## 관련 파일

| 파일 | 위치 |
|------|------|
| KBMC 논문 분석 | `references/kbmc_paper_analysis.md` |
| 기존 NER 노트북 | `notebooks/NER기반_의료_용어_추출기.ipynb` |
| GLiNER2 비교 | `references/gliner2_paper_analysis.md` |

---

## 다음 단계

1. [ ] 데이터셋 다운로드 (`load_dataset("SungJoo/KBMC")`)
2. [ ] 로컬 환경에서 KoELECTRA 학습
3. [ ] GLiNER2 zero-shot 테스트 (같은 데이터)
4. [ ] 성능 비교 (F1 기준)
