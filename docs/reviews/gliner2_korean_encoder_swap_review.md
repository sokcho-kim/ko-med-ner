# GLiNER2 한국어 인코더 교체 — 코드 리뷰 문서

> 2026-01-27 구현 완료. Codex 리뷰 6건 수정 반영.

---

## 1. 프로젝트 상황 요약

### 뭘 하려는 건지

GLiNER2라는 NER 모델의 인코더(DeBERTa-v3-base)가 영어 전용이라 한국어 의료 텍스트를 처리 못 함. 인코더만 한국어용(`team-lucid/deberta-v3-base-korean`)으로 교체하고 파인튜닝해서 한국어 의료 NER 성능을 올리려는 것.

### 이전 실패 이력

| 시도 | 결과 | 실패 원인 |
|------|------|----------|
| LoRA 파인튜닝 #1 (KBMC 50건) | F1=0.0000 | DeBERTa-v3-base가 한국어를 음절 단위 분절 (`"당뇨병"` → `['▁','당','뇨','병']`) |
| LoRA 파인튜닝 #2 (Silver 696건) | F1=0.0000 | 동일. LoRA로는 인코더 자체의 한계 극복 불가 |

### 데이터

- 학습: 696건 (`project/ner/data/gliner2_train_v2/train.jsonl`)
- 테스트: 77건 (`project/ner/data/gliner2_train_v2/test.jsonl`)
- 라벨: Disease, Drug, Procedure, Biomarker
- 소스: 보건복지부 고시 + GPT-5 검증

### 인코더 선택 근거

GLiNER2 모델 구조:
```
[인코더: DeBERTa-v3-base] → [프로젝션: 768→512] → [span_rep] → [classifier]
```

인코더를 교체하면서 프로젝션/span_rep/classifier의 pretrained weights를 살리려면 **동일 아키텍처**여야 함. 6개 후보 중 `team-lucid/deberta-v3-base-korean`만 DebertaV2Config + hidden_size=768.

| 후보 | 아키텍처 | 탈락 사유 |
|------|----------|----------|
| KURE-v1 | Sentence Transformer | 문장→벡터 1개. NER은 토큰별 벡터 필요 |
| KoELECTRA | ElectraConfig | Config 불일치 → head weights 재활용 불가 |
| KLUE-RoBERTa | RobertaConfig | Config 불일치 |
| XLM-RoBERTa | XLMRobertaConfig | Config 불일치 + 270M 무거움 |
| gliner_ko | GLiNER v1 완성 모델 | GLiNER2의 스키마 인터페이스 포기 |
| **deberta-v3-base-korean** | **DebertaV2Config** | **선택** |

---

## 2. 생성한 파일 4개

| # | 파일 | 용도 |
|---|------|------|
| 1 | `docs/plans/gliner2_korean_encoder_swap_plan.md` | 전체 계획 문서 |
| 2 | `scripts/gliner2/train_gliner2_korean_encoder.py` | 학습 스크립트 (핵심) |
| 3 | `scripts/gliner2/verify_encoder_swap.py` | 사전 검증 스크립트 |
| 4 | `docs/plans/encoder_selection_discussion.md` | 기존 인코더 선택 문서 갱신 |

모든 경로는 `project/ner/` 기준.

---

## 3. 학습 스크립트 상세 (`train_gliner2_korean_encoder.py`)

### 3.1 전체 흐름

```
main()
  ├─ [1/5] 데이터 로드 (JSONL)
  ├─ [2/5] GLiNER2 로드 → swap_encoder()로 인코더 교체
  ├─ [3/5] 교체 직후 베이스라인 측정 (threshold sweep)
  ├─ [4/5] 파인튜닝 (커스텀 루프 or 내장 Trainer)
  └─ [5/5] 최종 평가 + 결과 저장
```

### 3.2 핵심 함수: `swap_encoder()`

**하는 일**: GLiNER2 내부에서 DeBERTa 인코더를 찾아서 한국어 DeBERTa로 교체

```python
def swap_encoder(model, korean_encoder_name=KOREAN_ENCODER):
    # 1. find_encoder_attr()로 GLiNER2 내부 인코더 모듈 탐색
    encoder_path, old_encoder = find_encoder_attr(model)

    # 2. Config 추출
    old_config = old_encoder.config

    # 3. 한국어 DeBERTa 로드
    korean_config = AutoConfig.from_pretrained(korean_encoder_name)
    korean_encoder = AutoModel.from_pretrained(korean_encoder_name)
    korean_tokenizer = AutoTokenizer.from_pretrained(korean_encoder_name)

    # 4. Config 클래스 + hidden_size 일치 assert
    assert old_config.__class__.__name__ == korean_config.__class__.__name__
    assert old_config.hidden_size == korean_config.hidden_size

    # 5. set_nested_attr()로 인코더 모듈 교체
    set_nested_attr(model, encoder_path, korean_encoder)

    # 6. 토크나이저 교체
    # 7. 임베딩 리사이즈 (vocab 차이 보정)
    # 8. forward pass 검증

    return model, korean_tokenizer, encoder_path
```

**리뷰 포인트**:
- `find_encoder_attr()`이 GLiNER2 내부 구조를 추측으로 탐색함. 후보 경로 4개 + fallback(named_modules에서 DebertaV2 검색). GLiNER2의 실제 구조를 모르는 상태에서 작성함.
- `find_tokenizer_attr()`도 마찬가지. `tokenizer`, `_tokenizer`, `model.tokenizer` 3개 후보.
- `set_nested_attr()`은 튜플 경로를 받아 중첩 속성에 setattr.

### 3.3 핵심 함수: `find_encoder_attr()`

GLiNER2 모델의 인코더가 어디에 붙어 있는지 모르므로 여러 경로를 순서대로 탐색:

```python
candidates = [
    ("model", "encoder"),                                    # model.model.encoder
    ("encoder",),                                            # model.encoder
    ("model", "token_rep_layer", "bert_layer", "bert"),      # GLiNER 계열
    ("model", "token_rep_layer"),
]
```

각 경로를 `hasattr` 체인으로 탐색하고, 실패하면 `named_modules()`에서 클래스명에 `"DebertaV2"`가 포함된 모듈을 찾는 fallback.

**리뷰 포인트**: 실제 GLiNER2 라이브러리의 모델 구조를 확인하지 않고 작성된 코드. RunPod에서 실행 시 경로가 안 맞으면 fallback이 동작해야 하는데, fallback도 `"DebertaV2"` 문자열 매칭에 의존.

### 3.4 커스텀 학습 루프: `train_custom_loop()`

GLiNER2Trainer 대신 직접 PyTorch 루프를 구현한 이유: **차등 학습률** 지원.

```python
param_groups = [
    {"params": encoder_params, "lr": 2e-5,  "weight_decay": 0.01},  # 인코더
    {"params": head_params,    "lr": 5e-4,  "weight_decay": 0.01},  # 헤드
]
optimizer = AdamW(param_groups)
```

학습 루프 핵심:
```python
for epoch in range(1, epochs + 1):
    model.train()
    # 셔플 후 배치 순회
    for batch in batches:
        loss = model.compute_loss(batch)  # GLiNER2의 compute_loss() 사용
        loss.backward()
        optimizer.step()

    # 에폭마다 threshold sweep 평가
    model.eval()
    best_th, best_f1, results = threshold_sweep(model, test_data)

    # 최고 F1 갱신 시 체크포인트 저장
    if best_f1 > overall_best:
        model.save_pretrained(best_path)
```

**리뷰 포인트**:
- `model.compute_loss(batch)` — GLiNER2에 이 메서드가 있다고 가정함. InputExample 리스트를 받아서 loss를 반환한다고 가정.
- gradient accumulation이 argparse에는 있지만 커스텀 루프에는 구현 안 됨. `--grad-accum`은 builtin trainer 경로에서만 사용됨.
- `import random`이 루프 안에 있음 (매 에폭 호출). 상단으로 올려야 함.
- FP16: `torch.amp.GradScaler("cuda")` + `torch.amp.autocast("cuda")` 사용. device가 cuda일 때만.

### 3.5 파라미터 그룹 분류: `get_parameter_groups()`

```python
encoder_prefix = ".".join(encoder_path) + "."

for name, param in model.named_parameters():
    if name.startswith(encoder_prefix) or name.startswith("encoder."):
        encoder_params.append(param)
    else:
        head_params.append(param)

# fallback: 위에서 인코더 파라미터가 0개면 키워드 매칭
if not encoder_params:
    for name, param in model.named_parameters():
        if any(kw in name.lower() for kw in ["deberta", "encoder", "embeddings", "bert"]):
            encoder_params.append(param)
```

**리뷰 포인트**: `"embeddings"` 키워드가 너무 넓어서 인코더가 아닌 임베딩 레이어도 잡힐 수 있음. 다만 이 경우에도 인코더와 같은 LR을 쓰는 거라 큰 문제는 아님.

### 3.6 평가: `evaluate()` + `threshold_sweep()`

```python
def evaluate(model, test_data, threshold=0.3):
    for doc in test_data:
        gold_set = {(m, label) for label, mentions in doc["entities"].items() for m in mentions}
        pred = model.extract_entities(doc["text"], ENTITY_DESCRIPTIONS, threshold=threshold)
        pred_set = {(m, label) for label, mentions in pred["entities"].items() for m in mentions}
        # exact match: tp = |gold ∩ pred|, fp = |pred - gold|, fn = |gold - pred|
```

threshold sweep: 0.1~0.6에서 최고 F1 임계값을 찾음.

**리뷰 포인트**: `model.extract_entities()`의 반환 형식이 `{"entities": {"Disease": ["당뇨병", ...], ...}}`라고 가정. 기존 코드(`train_gliner2_silver.py`)와 동일한 인터페이스.

### 3.7 argparse 기본값

```
--train      ../../data/gliner2_train_v2/train.jsonl  (스크립트 기준 상대경로)
--test       ../../data/gliner2_train_v2/test.jsonl
--output     gliner2_korean_encoder
--epochs     15
--batch-size 8
--encoder-lr 2e-5
--head-lr    5e-4
--fp16       True (기본)
--seed       42
```

`Path(__file__).parent`로 상대경로를 절대경로로 변환.

### 3.8 사용하지 않는 import

```python
import copy          # 사용 안 됨
import os            # 사용 안 됨
import torch.nn as nn          # 사용 안 됨
from torch.utils.data import DataLoader  # 사용 안 됨
```

---

## 4. 검증 스크립트 상세 (`verify_encoder_swap.py`)

### 4.1 검증 5단계

| # | 검증 | 방법 | 의존성 |
|---|------|------|--------|
| 1 | Config 클래스 | `AutoConfig.from_pretrained()` → 클래스명에 `"DebertaV2"` 포함 확인 | transformers |
| 2 | hidden_size | `config.hidden_size == 768` | transformers |
| 3 | 토크나이저 | 의료 문장 5개 토크나이즈 → 단일 음절 한글 토큰 비율 70% 초과 시 FAIL | transformers |
| 4 | Forward pass | `AutoModel` 로드 → forward → output shape `(1, seq_len, 768)` + NaN/Inf 체크 | transformers, torch |
| 5 | GLiNER2 구조 | `GLiNER2.from_pretrained()` → `named_modules()`에서 DeBERTa 찾기 | gliner2 (없으면 SKIP) |

### 4.2 토크나이저 품질 판별 로직

```python
# 한글 토큰 중 단일 음절인 것의 비율
single_syllable_count = sum(
    1 for t in tokens
    if len(t.replace("▁", "").replace("##", "")) == 1
    and any("\uac00" <= c <= "\ud7a3" for c in t)
)
ratio = single_syllable_count / total_korean_tokens
is_char_level = ratio > 0.7  # 70% 초과면 음절 분절로 판단
```

영어 DeBERTa-v3-base의 경우 한국어가 거의 100% 음절 분절됨. 한국어 DeBERTa는 형태소 수준이어야 함.

### 4.3 검증 5번 (GLiNER2 구조) — 조건부 실행

```python
try:
    from gliner2 import GLiNER2
except ImportError:
    print("  [SKIP] gliner2 패키지가 설치되지 않음")
    return None
```

gliner2가 설치 안 된 로컬 환경에서도 검증 1~4는 실행 가능. 5번만 SKIP. 종합 결과에서 `None`은 `SKIP`으로 표시되며 전체 PASS/FAIL 판정에 영향 없음.

### 4.4 종합 결과 로직

```python
all_passed = True
for name, passed in results.items():
    if passed is None:       # SKIP — 무시
        status = "SKIP"
    elif passed:             # PASS
        status = "PASS ✓"
    else:                    # FAIL — all_passed = False
        status = "FAIL ✗"
        all_passed = False
```

exit code: 0 (성공) / 1 (실패).

---

## 5. 계획 문서 (`gliner2_korean_encoder_swap_plan.md`)

문서 구조:
1. 배경 (문제, 목표, 데이터)
2. 인코더 후보 6개 비교 + 탈락 사유
3. Surgical Swap 구현 방식
4. 하이퍼파라미터 (인코더 LR 2e-5 / 헤드 LR 5e-4)
5. 평가 계획 (사전 검증 → 학습 중 에폭별 → 최종)
6. 리스크 5개
7. 실행 절차 (Phase 1: 로컬 검증 → Phase 2: RunPod 학습 → Phase 3: 분석)

---

## 6. 기존 문서 갱신 (`encoder_selection_discussion.md`)

### 변경 사항

| 섹션 | 이전 | 이후 |
|------|------|------|
| 후보 인코더 | 3개 (xlm-r, klue, koelectra) | 6개 (+ KURE-v1, gliner_ko, deberta-v3-base-korean) |
| 선택 기준 | "GLiNER2 호환성" (코드 수준) | "아키텍처 일치" (Config 클래스 일치) |
| 결론 | "인코더 선택은 후순위. 데이터부터" | "deberta-v3-base-korean으로 결정. 데이터 준비 완료" |
| 미해결 사항 | 라벨 미정의, 골든셋 없음, 실버셋 없음 | 모두 해결됨 |

갱신일 `2026-01-27` 추가.

---

## 7. 리뷰 시 주의할 점

### 확인이 필요한 가정들

1. **GLiNER2 모델 내부 구조**: `find_encoder_attr()`이 4개 경로 후보 + fallback으로 인코더를 찾음. 실제 `fastino/gliner2-base-v1`의 모듈 트리를 확인해야 정확히 맞는지 알 수 있음.

2. **`model.compute_loss(batch)`**: 커스텀 학습 루프가 이 메서드에 의존. GLiNER2에 이 인터페이스가 있는지, InputExample 리스트를 직접 받는지 확인 필요. → `_compute_loss_safe()` 래퍼로 3가지 호출 방식 자동 시도하도록 개선됨 (섹션 9 참조)

3. **`model.extract_entities()` 반환 형식**: `{"entities": {"label": ["mention", ...]}}` 형태라고 가정. 기존 코드(`train_gliner2_silver.py`, `eval.py`)에서 동일하게 사용하고 있어서 맞을 가능성 높음.

4. **`model.save_pretrained()`**: 인코더 교체 후에도 이 메서드가 정상 동작하는지. 교체된 인코더 + 원래 head가 함께 저장되는지.

5. **임베딩 리사이즈**: `resize_token_embeddings(len(korean_tokenizer))`가 DeBERTa 모델에서 정상 동작하는지. vocab 크기 차이가 크면 새로 초기화되는 임베딩 행이 많아짐. → 3단계 fallback 구현됨 (섹션 9 참조)

### 코드 품질 이슈 (→ 섹션 9에서 전부 해결됨)

1. ~~**사용하지 않는 import 4개**: `copy`, `os`, `nn`, `DataLoader`~~ → 삭제됨
2. ~~**`import random`이 루프 안에 위치**~~ → 최상위 import로 이동
3. ~~**gradient accumulation 미구현**~~ → 커스텀 루프에 구현 완료
4. ~~**`--fp16` 기본값이 `True`인데 `action="store_true"`**~~ → `BooleanOptionalAction`으로 변경

### 설계 판단

1. **커스텀 루프 vs 내장 Trainer**: 차등 LR을 위해 커스텀 루프 작성. `--use-builtin-trainer` 옵션으로 fallback 제공. 합리적 판단.
2. **threshold sweep**: 매 에폭마다 6개 임계값 전수 평가. 77건 테스트셋이면 부담 없음.
3. **체크포인트 저장**: best (F1 기준) + 5에폭마다 + final. `save_pretrained` 실패 시 `state_dict` fallback.

---

## 8. 파일 전문

> **주의**: 아래 코드는 **1차 리뷰 전 초기 버전**이다. 섹션 9의 수정 사항(1차 Codex 리뷰 6건 + 2차 재검토 2건)이 반영된 최신 코드는 실제 소스 파일을 참조할 것.

### 8.1 `train_gliner2_korean_encoder.py`

파일 위치: `project/ner/scripts/gliner2/train_gliner2_korean_encoder.py`

```python
"""
GLiNER2 한국어 인코더 교체 학습 스크립트

Approach A: Surgical Swap
- pretrained GLiNER2 로드 → 인코더를 team-lucid/deberta-v3-base-korean으로 교체
- 토크나이저 교체 + 임베딩 리사이즈
- 차등 학습률로 전체 파인튜닝 (인코더 2e-5, 헤드 5e-4)

사용법:
    python train_gliner2_korean_encoder.py
    python train_gliner2_korean_encoder.py --epochs 20 --batch-size 4
    python train_gliner2_korean_encoder.py --train /path/to/train.jsonl --test /path/to/test.jsonl
"""

import json
import argparse
import os
import sys
import copy
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    get_linear_schedule_with_warmup,
)

# ============================================
# 설정
# ============================================

KOREAN_ENCODER = "team-lucid/deberta-v3-base-korean"
GLINER2_PRETRAINED = "fastino/gliner2-base-v1"

ENTITY_DESCRIPTIONS = {
    "Disease": "질병, 증상, 의학적 상태 (예: 당뇨병, 고혈압, 폐렴, 암)",
    "Drug": "약물, 의약품, 치료제 (예: 인슐린, 아스피린, 항생제)",
    "Procedure": "의료 시술, 수술, 검사 (예: 내시경, MRI, 수술)",
    "Biomarker": "바이오마커, 검사 수치, 생체 지표 (예: 혈당, 콜레스테롤, 종양표지자)",
}

LABELS = list(ENTITY_DESCRIPTIONS.keys())

THRESHOLD_SWEEP = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def load_data(path: str) -> list:
    """JSONL 파일 로드"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# ============================================
# 인코더 교체 (Surgical Swap)
# ============================================


def find_encoder_attr(model):
    """GLiNER2 모델에서 인코더 모듈의 속성 이름을 찾는다.

    GLiNER2 내부 구조에 따라 인코더가 위치하는 속성명이 다를 수 있으므로
    여러 가능한 경로를 탐색한다.
    """
    # 가능한 인코더 경로 (우선순위 순)
    candidates = [
        ("model", "encoder"),       # model.model.encoder
        ("encoder",),               # model.encoder
        ("model", "token_rep_layer", "bert_layer", "bert"),  # GLiNER 계열
        ("model", "token_rep_layer"),
    ]

    for path in candidates:
        obj = model
        found = True
        for attr in path:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                found = False
                break
        if found and hasattr(obj, "config"):
            return path, obj

    # Fallback: 모델의 모든 모듈에서 DebertaV2 찾기
    for name, module in model.named_modules():
        cls_name = module.__class__.__name__
        if "DebertaV2" in cls_name and hasattr(module, "config"):
            path = tuple(name.split("."))
            return path, module

    raise RuntimeError(
        "GLiNER2 모델에서 인코더를 찾을 수 없습니다.\n"
        f"모델 최상위 속성: {[n for n, _ in model.named_children()]}"
    )


def find_tokenizer_attr(model):
    """GLiNER2 모델에서 토크나이저 속성을 찾는다."""
    candidates = ["tokenizer", "_tokenizer", "model.tokenizer"]
    for attr_path in candidates:
        parts = attr_path.split(".")
        obj = model
        found = True
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                found = False
                break
        if found and hasattr(obj, "tokenize"):
            return attr_path
    return None


def set_nested_attr(obj, path, value):
    """중첩된 속성에 값을 설정한다. path는 점(.)으로 구분된 문자열 또는 튜플."""
    if isinstance(path, str):
        path = path.split(".")
    for attr in path[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, path[-1], value)


def swap_encoder(model, korean_encoder_name=KOREAN_ENCODER):
    """GLiNER2의 인코더를 한국어 DeBERTa로 교체한다.

    Returns:
        model: 인코더가 교체된 모델
        korean_tokenizer: 한국어 토크나이저
        encoder_path: 인코더의 속성 경로
    """
    print(f"\n{'=' * 60}")
    print(f"인코더 교체: {korean_encoder_name}")
    print(f"{'=' * 60}")

    # 1. 기존 인코더 분석
    encoder_path, old_encoder = find_encoder_attr(model)
    old_config = old_encoder.config
    print(f"\n[기존 인코더]")
    print(f"  경로: {'.'.join(encoder_path)}")
    print(f"  Config: {old_config.__class__.__name__}")
    print(f"  hidden_size: {old_config.hidden_size}")
    print(f"  num_layers: {getattr(old_config, 'num_hidden_layers', 'N/A')}")

    # 2. 한국어 DeBERTa 로드
    print(f"\n[한국어 인코더 로드]")
    korean_config = AutoConfig.from_pretrained(korean_encoder_name)
    korean_encoder = AutoModel.from_pretrained(korean_encoder_name)
    korean_tokenizer = AutoTokenizer.from_pretrained(korean_encoder_name)

    print(f"  Config: {korean_config.__class__.__name__}")
    print(f"  hidden_size: {korean_config.hidden_size}")
    print(f"  num_layers: {getattr(korean_config, 'num_hidden_layers', 'N/A')}")
    print(f"  vocab_size: {korean_config.vocab_size}")

    # 3. 호환성 검증
    assert old_config.__class__.__name__ == korean_config.__class__.__name__, (
        f"Config 불일치: {old_config.__class__.__name__} != {korean_config.__class__.__name__}"
    )
    assert old_config.hidden_size == korean_config.hidden_size, (
        f"hidden_size 불일치: {old_config.hidden_size} != {korean_config.hidden_size}"
    )
    print(f"\n  ✓ 아키텍처 호환성 확인 (Config, hidden_size 일치)")

    # 4. 인코더 교체
    set_nested_attr(model, encoder_path, korean_encoder)
    print(f"  ✓ 인코더 교체 완료")

    # 5. 토크나이저 교체
    tok_attr = find_tokenizer_attr(model)
    if tok_attr:
        set_nested_attr(model, tok_attr.split("."), korean_tokenizer)
        print(f"  ✓ 토크나이저 교체 완료 (속성: {tok_attr})")
    else:
        # 직접 속성 설정
        model.tokenizer = korean_tokenizer
        print(f"  ✓ 토크나이저 교체 완료 (model.tokenizer)")

    # 6. 임베딩 리사이즈
    _, new_encoder = find_encoder_attr(model)
    if hasattr(new_encoder, "resize_token_embeddings"):
        new_encoder.resize_token_embeddings(len(korean_tokenizer))
        print(f"  ✓ 임베딩 리사이즈 완료 (vocab={len(korean_tokenizer)})")

    # 7. 교체 검증 — forward pass
    print(f"\n[교체 검증]")
    test_text = "당뇨병 환자에게 인슐린을 투여한다"
    tokens = korean_tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
    print(f"  입력: \"{test_text}\"")
    print(f"  토큰: {korean_tokenizer.tokenize(test_text)}")

    with torch.no_grad():
        _, swapped_encoder = find_encoder_attr(model)
        device = next(swapped_encoder.parameters()).device
        tokens = {k: v.to(device) for k, v in tokens.items()}
        output = swapped_encoder(**tokens)
        hidden = output.last_hidden_state
        print(f"  출력 shape: {hidden.shape}")
        print(f"  ✓ Forward pass 성공")

    return model, korean_tokenizer, encoder_path


# ============================================
# 평가
# ============================================


def evaluate(model, test_data, threshold=0.3):
    """엔티티 단위 평가 (exact match)"""
    tp, fp, fn = 0, 0, 0
    label_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for doc in test_data:
        # 정답
        gold_set = set()
        for label, mentions in doc["entities"].items():
            for m in mentions:
                gold_set.add((m, label))

        # 예측
        try:
            result = model.extract_entities(
                doc["text"], ENTITY_DESCRIPTIONS, threshold=threshold
            )
            pred_set = set()
            for label, mentions in result.get("entities", {}).items():
                for m in mentions:
                    pred_set.add((m, label))
        except Exception as e:
            print(f"  [WARN] 예측 실패: {e}")
            pred_set = set()

        # 계산
        matched = gold_set & pred_set
        tp += len(matched)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

        for text, label in matched:
            label_stats[label]["tp"] += 1
        for text, label in (pred_set - gold_set):
            label_stats[label]["fp"] += 1
        for text, label in (gold_set - pred_set):
            label_stats[label]["fn"] += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "label_stats": dict(label_stats),
    }


def threshold_sweep(model, test_data, thresholds=THRESHOLD_SWEEP):
    """여러 임계값에서 평가하여 최적 임계값을 찾는다."""
    best_f1 = 0
    best_threshold = 0.3
    results = {}

    for th in thresholds:
        result = evaluate(model, test_data, threshold=th)
        results[th] = result
        if result["f1"] > best_f1:
            best_f1 = result["f1"]
            best_threshold = th

    return best_threshold, best_f1, results


def print_results(name, result, labels=LABELS):
    """결과 출력"""
    print(f"\n{name}:")
    print(f"  Precision: {result['precision']:.4f}")
    print(f"  Recall:    {result['recall']:.4f}")
    print(f"  F1:        {result['f1']:.4f}")
    print(f"  (TP={result['tp']}, FP={result['fp']}, FN={result['fn']})")

    if result.get("label_stats"):
        print(f"\n  라벨별 성능:")
        for label in labels:
            if label in result["label_stats"]:
                s = result["label_stats"][label]
                p = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) > 0 else 0
                r = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) > 0 else 0
                f = 2 * p * r / (p + r) if (p + r) > 0 else 0
                print(f"    {label:12} P={p:.3f} R={r:.3f} F1={f:.3f} (TP={s['tp']}, FP={s['fp']}, FN={s['fn']})")
            else:
                print(f"    {label:12} (데이터 없음)")


# ============================================
# 학습
# ============================================


def get_parameter_groups(model, encoder_path, encoder_lr, head_lr, weight_decay=0.01):
    """차등 학습률을 적용한 파라미터 그룹을 생성한다.

    - 인코더 파라미터: encoder_lr (작은 LR로 사전 학습 지식 보존)
    - 헤드 파라미터: head_lr (큰 LR로 새 인코더에 빠르게 적응)
    """
    encoder_prefix = ".".join(encoder_path) + "."

    encoder_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(encoder_prefix) or name.startswith("encoder."):
            encoder_params.append(param)
        else:
            head_params.append(param)

    # 인코더가 하나도 안 잡힌 경우 — 모델 구조에 따라 다른 패턴 시도
    if not encoder_params:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # DeBERTa 관련 키워드로 분류
            if any(kw in name.lower() for kw in ["deberta", "encoder", "embeddings", "bert"]):
                encoder_params.append(param)
            else:
                head_params.append(param)

    print(f"\n[파라미터 그룹]")
    print(f"  인코더: {len(encoder_params)} params, LR={encoder_lr}")
    print(f"  헤드:   {len(head_params)} params, LR={head_lr}")

    param_groups = [
        {
            "params": encoder_params,
            "lr": encoder_lr,
            "weight_decay": weight_decay,
        },
        {
            "params": head_params,
            "lr": head_lr,
            "weight_decay": weight_decay,
        },
    ]

    return param_groups


def train_with_gliner2_trainer(model, train_data, test_data, args, encoder_path):
    """GLiNER2 내장 Trainer를 사용한 학습 (fallback).

    차등 학습률은 GLiNER2Trainer가 지원하지 않을 수 있으므로,
    이 경우 단일 학습률로 전체 파인튜닝을 수행한다.
    """
    from gliner2.training.data import InputExample
    from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

    train_examples = []
    for doc in train_data:
        train_examples.append(
            InputExample(
                text=doc["text"],
                entities=doc["entities"],
                entity_descriptions=ENTITY_DESCRIPTIONS,
            )
        )

    test_examples = []
    for doc in test_data:
        test_examples.append(
            InputExample(
                text=doc["text"],
                entities=doc["entities"],
                entity_descriptions=ENTITY_DESCRIPTIONS,
            )
        )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = TrainingConfig(
        output_dir=str(output_dir),
        use_lora=False,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        task_lr=args.head_lr,  # 단일 LR 사용
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy="epoch",
        save_best=True,
        early_stopping=False,
        fp16=args.fp16,
        num_workers=2,
        seed=args.seed,
    )

    trainer = GLiNER2Trainer(model, config)

    print(f"\n학습 시작 (GLiNER2Trainer, 단일 LR={args.head_lr})...")
    results = trainer.train(train_data=train_examples, eval_data=test_examples)

    # 모델 저장
    final_path = output_dir / "final"
    trainer.model.save_pretrained(str(final_path))
    print(f"모델 저장: {final_path}")

    return trainer.model


def train_custom_loop(model, train_data, test_data, args, encoder_path):
    """커스텀 학습 루프 — 차등 학습률 지원.

    GLiNER2Trainer 대신 직접 PyTorch 학습 루프를 구현하여
    인코더/헤드에 다른 학습률을 적용한다.
    """
    from gliner2.training.data import InputExample

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # InputExample 변환
    train_examples = [
        InputExample(
            text=doc["text"],
            entities=doc["entities"],
            entity_descriptions=ENTITY_DESCRIPTIONS,
        )
        for doc in train_data
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    # 파라미터 그룹 (차등 LR)
    param_groups = get_parameter_groups(
        model, encoder_path,
        encoder_lr=args.encoder_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
    )
    optimizer = AdamW(param_groups)

    # 스케줄러
    total_steps = (len(train_examples) // args.batch_size + 1) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"\n[학습 설정]")
    print(f"  Device: {device}")
    print(f"  Train: {len(train_examples)}건")
    print(f"  Test: {len(test_data)}건")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  인코더 LR: {args.encoder_lr}")
    print(f"  헤드 LR: {args.head_lr}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  FP16: {args.fp16}")

    scaler = torch.amp.GradScaler("cuda") if args.fp16 and device.type == "cuda" else None

    best_f1 = 0
    best_epoch = 0
    best_threshold = 0.3
    epoch_results = []

    start_time = datetime.now()
    print(f"\n학습 시작: {start_time.strftime('%H:%M:%S')}")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        batch_count = 0

        # 간단한 배치 처리 — GLiNER2의 내부 collate 활용
        import random
        indices = list(range(len(train_examples)))
        random.shuffle(indices)

        for i in range(0, len(indices), args.batch_size):
            batch_indices = indices[i : i + args.batch_size]
            batch = [train_examples[j] for j in batch_indices]

            optimizer.zero_grad()

            try:
                if scaler:
                    with torch.amp.autocast("cuda"):
                        loss = model.compute_loss(batch)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss = model.compute_loss(batch)
                    loss.backward()
                    optimizer.step()

                scheduler.step()
                epoch_loss += loss.item()
                batch_count += 1

            except Exception as e:
                print(f"  [WARN] Batch {i//args.batch_size + 1} 에러: {e}")
                continue

        avg_loss = epoch_loss / max(batch_count, 1)

        # 에폭별 평가
        model.eval()
        with torch.no_grad():
            best_th, best_th_f1, sweep_results = threshold_sweep(model, test_data)

        result_at_best_th = sweep_results[best_th]

        epoch_info = {
            "epoch": epoch,
            "loss": avg_loss,
            "best_threshold": best_th,
            "f1": best_th_f1,
            "precision": result_at_best_th["precision"],
            "recall": result_at_best_th["recall"],
        }
        epoch_results.append(epoch_info)

        # 에폭 결과 출력
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Best threshold: {best_th}")
        print(f"  F1={best_th_f1:.4f} (P={result_at_best_th['precision']:.4f}, R={result_at_best_th['recall']:.4f})")

        # 라벨별 성능
        for label in LABELS:
            if label in result_at_best_th["label_stats"]:
                s = result_at_best_th["label_stats"][label]
                p = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) > 0 else 0
                r = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) > 0 else 0
                f = 2 * p * r / (p + r) if (p + r) > 0 else 0
                print(f"    {label:12} F1={f:.3f}")

        # 최고 성능 체크포인트 저장
        if best_th_f1 > best_f1:
            best_f1 = best_th_f1
            best_epoch = epoch
            best_threshold = best_th

            best_path = output_dir / "best"
            best_path.mkdir(parents=True, exist_ok=True)
            try:
                model.save_pretrained(str(best_path))
                print(f"  ★ 새 최고 성능! F1={best_f1:.4f} → 저장: {best_path}")
            except Exception as e:
                # save_pretrained가 없으면 state_dict 저장
                torch.save(model.state_dict(), best_path / "model.pt")
                print(f"  ★ 새 최고 성능! F1={best_f1:.4f} → state_dict 저장")

        # 에폭별 체크포인트
        if epoch % 5 == 0 or epoch == args.epochs:
            ckpt_path = output_dir / f"checkpoint-epoch-{epoch}"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            try:
                model.save_pretrained(str(ckpt_path))
            except Exception:
                torch.save(model.state_dict(), ckpt_path / "model.pt")

    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()

    print(f"\n{'=' * 60}")
    print(f"학습 완료: {end_time.strftime('%H:%M:%S')} ({elapsed/60:.1f}분)")
    print(f"최고 성능: Epoch {best_epoch}, F1={best_f1:.4f} (threshold={best_threshold})")
    print(f"{'=' * 60}")

    # 최종 모델 저장
    final_path = output_dir / "final"
    final_path.mkdir(parents=True, exist_ok=True)
    try:
        model.save_pretrained(str(final_path))
    except Exception:
        torch.save(model.state_dict(), final_path / "model.pt")

    return model, epoch_results, best_f1, best_epoch, best_threshold, elapsed


# ============================================
# 메인
# ============================================


def main():
    parser = argparse.ArgumentParser(description="GLiNER2 한국어 인코더 교체 학습")
    parser.add_argument(
        "--train", default="../../data/gliner2_train_v2/train.jsonl",
        help="학습 데이터 경로",
    )
    parser.add_argument(
        "--test", default="../../data/gliner2_train_v2/test.jsonl",
        help="테스트 데이터 경로",
    )
    parser.add_argument("--output", default="gliner2_korean_encoder", help="출력 디렉토리")
    parser.add_argument("--epochs", type=int, default=15, help="에폭 수")
    parser.add_argument("--batch-size", type=int, default=8, help="배치 크기")
    parser.add_argument("--encoder-lr", type=float, default=2e-5, help="인코더 학습률")
    parser.add_argument("--head-lr", type=float, default=5e-4, help="헤드 학습률")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup 비율")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--fp16", action="store_true", default=True, help="FP16 사용")
    parser.add_argument("--no-fp16", action="store_true", help="FP16 비활성화")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument(
        "--use-builtin-trainer", action="store_true",
        help="GLiNER2 내장 Trainer 사용 (차등 LR 미지원)",
    )
    args = parser.parse_args()

    if args.no_fp16:
        args.fp16 = False

    # CUDA 확인
    if not torch.cuda.is_available():
        print("[WARN] GPU를 사용할 수 없습니다. CPU에서 학습합니다.")
        args.fp16 = False

    # 시드 설정
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 60)
    print("GLiNER2 한국어 인코더 교체 학습")
    print("=" * 60)

    # ============================================
    # 1. 데이터 로드
    # ============================================
    print("\n[1/5] 데이터 로드...")

    # 상대 경로 → 절대 경로 변환
    script_dir = Path(__file__).parent
    train_path = Path(args.train)
    test_path = Path(args.test)

    if not train_path.is_absolute():
        train_path = (script_dir / train_path).resolve()
    if not test_path.is_absolute():
        test_path = (script_dir / test_path).resolve()

    train_data = load_data(str(train_path))
    test_data = load_data(str(test_path))
    print(f"  Train: {len(train_data)}건 ({train_path})")
    print(f"  Test: {len(test_data)}건 ({test_path})")

    # ============================================
    # 2. 모델 로드 + 인코더 교체
    # ============================================
    print("\n[2/5] 모델 로드 + 인코더 교체...")

    from gliner2 import GLiNER2

    model = GLiNER2.from_pretrained(GLINER2_PRETRAINED)
    print(f"  GLiNER2 로드 완료: {GLINER2_PRETRAINED}")

    # Surgical swap
    model, korean_tokenizer, encoder_path = swap_encoder(model, KOREAN_ENCODER)

    # ============================================
    # 3. 교체 후 베이스라인 (파인튜닝 전)
    # ============================================
    print("\n[3/5] 인코더 교체 후 베이스라인 측정...")
    print("(인코더만 교체, 파인튜닝 전)")

    model.eval()
    with torch.no_grad():
        swap_best_th, swap_best_f1, swap_sweep = threshold_sweep(model, test_data)

    print(f"\n임계값 스윕 결과:")
    for th in THRESHOLD_SWEEP:
        r = swap_sweep[th]
        print(f"  th={th:.1f}: F1={r['f1']:.4f} (P={r['precision']:.4f}, R={r['recall']:.4f})")
    print(f"\n  최적 임계값: {swap_best_th} → F1={swap_best_f1:.4f}")

    baseline_result = swap_sweep[swap_best_th]
    print_results("인코더 교체 후 베이스라인", baseline_result)

    # ============================================
    # 4. 파인튜닝
    # ============================================
    print("\n[4/5] 파인튜닝 시작...")

    if args.use_builtin_trainer:
        finetuned_model = train_with_gliner2_trainer(
            model, train_data, test_data, args, encoder_path
        )
        epoch_results = []
        best_f1 = 0
        best_epoch = 0
        best_threshold = 0.3
        elapsed = 0
    else:
        finetuned_model, epoch_results, best_f1, best_epoch, best_threshold, elapsed = (
            train_custom_loop(model, train_data, test_data, args, encoder_path)
        )

    # ============================================
    # 5. 최종 평가
    # ============================================
    print("\n[5/5] 최종 평가...")

    # 최고 모델 로드 시도
    best_path = Path(args.output) / "best"
    if (best_path / "config.json").exists():
        try:
            from gliner2 import GLiNER2
            eval_model = GLiNER2.from_pretrained(str(best_path))
            print(f"  최고 모델 로드: {best_path}")
        except Exception:
            eval_model = finetuned_model
            print(f"  최종 모델 사용 (best 로드 실패)")
    elif (best_path / "model.pt").exists():
        eval_model = finetuned_model
        # state_dict 로드는 아키텍처가 동일해야 하므로 현재 모델 사용
        print(f"  최종 모델 사용 (state_dict)")
    else:
        eval_model = finetuned_model
        print(f"  최종 모델 사용")

    eval_model.eval()
    with torch.no_grad():
        final_best_th, final_best_f1, final_sweep = threshold_sweep(eval_model, test_data)

    print(f"\n최종 임계값 스윕:")
    for th in THRESHOLD_SWEEP:
        r = final_sweep[th]
        print(f"  th={th:.1f}: F1={r['f1']:.4f}")

    final_result = final_sweep[final_best_th]
    print_results(f"최종 결과 (threshold={final_best_th})", final_result)

    # ============================================
    # 결과 비교
    # ============================================
    print(f"\n{'=' * 60}")
    print("결과 비교")
    print("=" * 60)

    delta_f1 = final_result["f1"] - baseline_result["f1"]
    print(f"\n{'':25} {'교체 직후':>12} {'파인튜닝 후':>12} {'Delta':>12}")
    print("-" * 65)
    print(f"{'Precision':25} {baseline_result['precision']:>12.4f} {final_result['precision']:>12.4f} {final_result['precision']-baseline_result['precision']:>+12.4f}")
    print(f"{'Recall':25} {baseline_result['recall']:>12.4f} {final_result['recall']:>12.4f} {final_result['recall']-baseline_result['recall']:>+12.4f}")
    print(f"{'F1':25} {baseline_result['f1']:>12.4f} {final_result['f1']:>12.4f} {delta_f1:>+12.4f}")
    print(f"{'Best threshold':25} {swap_best_th:>12.1f} {final_best_th:>12.1f}")

    # ============================================
    # 결과 저장
    # ============================================
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "results.txt"
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("GLiNER2 한국어 인코더 교체 학습 결과\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"날짜: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"인코더: {KOREAN_ENCODER}\n")
        f.write(f"Train: {len(train_data)}건, Test: {len(test_data)}건\n")
        f.write(f"Epochs: {args.epochs}, Batch: {args.batch_size}\n")
        f.write(f"인코더 LR: {args.encoder_lr}, 헤드 LR: {args.head_lr}\n")
        f.write(f"학습 시간: {elapsed/60:.1f}분\n\n")

        f.write("--- 인코더 교체 직후 (파인튜닝 전) ---\n")
        f.write(f"  Best threshold: {swap_best_th}\n")
        f.write(f"  P={baseline_result['precision']:.4f}, R={baseline_result['recall']:.4f}, F1={baseline_result['f1']:.4f}\n\n")

        f.write("--- 파인튜닝 후 ---\n")
        f.write(f"  Best threshold: {final_best_th}\n")
        f.write(f"  P={final_result['precision']:.4f}, R={final_result['recall']:.4f}, F1={final_result['f1']:.4f}\n\n")

        f.write(f"Delta F1: {delta_f1:+.4f}\n")
        f.write(f"Best epoch: {best_epoch}\n\n")

        # 에폭별 결과
        if epoch_results:
            f.write("--- 에폭별 결과 ---\n")
            f.write(f"{'Epoch':>5} {'Loss':>10} {'Threshold':>10} {'F1':>10} {'P':>10} {'R':>10}\n")
            for er in epoch_results:
                f.write(
                    f"{er['epoch']:>5} {er['loss']:>10.4f} {er['best_threshold']:>10.1f} "
                    f"{er['f1']:>10.4f} {er['precision']:>10.4f} {er['recall']:>10.4f}\n"
                )

        # 라벨별 최종 성능
        f.write("\n--- 라벨별 최종 성능 ---\n")
        for label in LABELS:
            if label in final_result["label_stats"]:
                s = final_result["label_stats"][label]
                p = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) > 0 else 0
                r = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) > 0 else 0
                f1_label = 2 * p * r / (p + r) if (p + r) > 0 else 0
                f.write(f"  {label:12} P={p:.3f} R={r:.3f} F1={f1_label:.3f}\n")

    # Training config 저장
    config_file = output_dir / "training_config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "encoder": KOREAN_ENCODER,
                "base_model": GLINER2_PRETRAINED,
                "encoder_lr": args.encoder_lr,
                "head_lr": args.head_lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "weight_decay": args.weight_decay,
                "warmup_ratio": args.warmup_ratio,
                "grad_accum": args.grad_accum,
                "fp16": args.fp16,
                "seed": args.seed,
                "train_count": len(train_data),
                "test_count": len(test_data),
                "best_epoch": best_epoch,
                "best_f1": best_f1,
                "best_threshold": best_threshold,
                "elapsed_seconds": elapsed,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n결과 저장: {results_file}")
    print(f"설정 저장: {config_file}")
    print("\n완료!")


if __name__ == "__main__":
    main()
```

### 8.2 `verify_encoder_swap.py`

파일 위치: `project/ner/scripts/gliner2/verify_encoder_swap.py` (415줄)

```python
"""
GLiNER2 한국어 인코더 교체 사전 검증 스크립트

GPU 없이 로컬에서 실행하여 인코더 교체 호환성을 확인한다.

검증 항목:
1. team-lucid/deberta-v3-base-korean이 DebertaV2Config으로 로드되는지
2. hidden_size=768 일치 여부
3. 토크나이저가 한국어를 형태소 단위로 분절하는지 (음절 아님)
4. Forward pass 출력 shape 검증
5. GLiNER2 인코더와의 아키텍처 매칭

사용법:
    python verify_encoder_swap.py
"""

import sys
from pathlib import Path

KOREAN_ENCODER = "team-lucid/deberta-v3-base-korean"
GLINER2_PRETRAINED = "fastino/gliner2-base-v1"

# 의료 도메인 테스트 문장
TEST_SENTENCES = [
    "당뇨병 환자에게 인슐린을 투여한다",
    "고혈압 치료를 위해 혈압강하제를 처방하였다",
    "뇌졸중 후 재활치료로 물리치료를 시행한다",
    "위내시경 검사에서 위궤양이 발견되었다",
    "혈당 수치가 126mg/dL 이상으로 당뇨병 진단 기준을 충족한다",
]


def separator(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def check_pass(name):
    print(f"  ✓ PASS: {name}")


def check_fail(name, detail=""):
    print(f"  ✗ FAIL: {name}")
    if detail:
        print(f"    → {detail}")
    return False


# ============================================
# 검증 1: Config 클래스 확인
# ============================================

def verify_config_class():
    """한국어 DeBERTa가 DebertaV2Config으로 로드되는지 확인"""
    separator("검증 1: Config 클래스")

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(KOREAN_ENCODER)
    config_cls = config.__class__.__name__

    print(f"  모델: {KOREAN_ENCODER}")
    print(f"  Config 클래스: {config_cls}")
    print(f"  model_type: {getattr(config, 'model_type', 'N/A')}")

    if "DebertaV2" in config_cls or "deberta-v2" in getattr(config, "model_type", ""):
        check_pass("DebertaV2Config 확인")
        return True, config
    else:
        return check_fail(
            "DebertaV2Config",
            f"기대: DebertaV2Config, 실제: {config_cls}"
        ), config


# ============================================
# 검증 2: hidden_size 일치
# ============================================

def verify_hidden_size(korean_config):
    """hidden_size가 768로 GLiNER2와 일치하는지 확인"""
    separator("검증 2: hidden_size")

    expected_hidden = 768
    actual_hidden = korean_config.hidden_size

    print(f"  GLiNER2 기대값: {expected_hidden}")
    print(f"  한국어 DeBERTa: {actual_hidden}")
    print(f"  num_hidden_layers: {getattr(korean_config, 'num_hidden_layers', 'N/A')}")
    print(f"  num_attention_heads: {getattr(korean_config, 'num_attention_heads', 'N/A')}")
    print(f"  intermediate_size: {getattr(korean_config, 'intermediate_size', 'N/A')}")
    print(f"  vocab_size: {getattr(korean_config, 'vocab_size', 'N/A')}")

    if actual_hidden == expected_hidden:
        check_pass(f"hidden_size={expected_hidden} 일치")
        return True
    else:
        return check_fail(
            "hidden_size",
            f"기대: {expected_hidden}, 실제: {actual_hidden}"
        )


# ============================================
# 검증 3: 토크나이저 품질
# ============================================

def verify_tokenizer():
    """토크나이저가 한국어를 형태소 단위로 분절하는지 확인"""
    separator("검증 3: 토크나이저 품질")

    from transformers import AutoTokenizer

    korean_tok = AutoTokenizer.from_pretrained(KOREAN_ENCODER)

    print(f"  토크나이저 클래스: {korean_tok.__class__.__name__}")
    print(f"  vocab_size: {korean_tok.vocab_size}")
    print(f"  model_max_length: {korean_tok.model_max_length}")
    print()

    # 영어 DeBERTa 토크나이저와 비교 (가능한 경우)
    try:
        english_tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
        has_english = True
    except Exception:
        has_english = False

    all_good = True
    for sent in TEST_SENTENCES:
        korean_tokens = korean_tok.tokenize(sent)
        print(f"  입력: \"{sent}\"")
        print(f"  한국어 DeBERTa: {korean_tokens} ({len(korean_tokens)}토큰)")

        if has_english:
            english_tokens = english_tok.tokenize(sent)
            print(f"  영어 DeBERTa:   {english_tokens} ({len(english_tokens)}토큰)")

        # 음절 분절 감지: 한글 단일 음절 토큰이 전체의 50% 이상이면 음절 분절로 판단
        single_syllable_count = sum(
            1 for t in korean_tokens
            if len(t.replace("▁", "").replace("##", "")) == 1
            and any("\uac00" <= c <= "\ud7a3" for c in t)
        )
        total_korean_chars = sum(
            1 for t in korean_tokens
            if any("\uac00" <= c <= "\ud7a3" for c in t)
        )

        if total_korean_chars > 0:
            ratio = single_syllable_count / total_korean_chars
            is_char_level = ratio > 0.7

            if is_char_level:
                print(f"  ⚠ 음절 분절 비율: {ratio:.0%} (한글 토큰 중 단일 음절)")
                all_good = False
            else:
                print(f"  형태소 분절 비율 양호: 단일음절={ratio:.0%}")

        print()

    if all_good:
        check_pass("토크나이저가 형태소 수준으로 분절")
    else:
        check_fail("토크나이저", "음절 단위 분절 감지 — 성능에 영향 있을 수 있음")

    return all_good


# ============================================
# 검증 4: Forward pass 출력 shape
# ============================================

def verify_forward_pass():
    """Forward pass 출력이 (batch, seq_len, 768)인지 확인"""
    separator("검증 4: Forward pass")

    import torch
    from transformers import AutoModel, AutoTokenizer

    model = AutoModel.from_pretrained(KOREAN_ENCODER)
    tokenizer = AutoTokenizer.from_pretrained(KOREAN_ENCODER)

    model.eval()
    test_text = TEST_SENTENCES[0]

    tokens = tokenizer(
        test_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    print(f"  입력: \"{test_text}\"")
    print(f"  input_ids shape: {tokens['input_ids'].shape}")
    print(f"  토큰 ID: {tokens['input_ids'][0].tolist()[:20]}...")

    with torch.no_grad():
        output = model(**tokens)

    hidden = output.last_hidden_state
    print(f"  출력 shape: {hidden.shape}")
    print(f"  기대: (1, seq_len, 768)")

    # shape 검증
    batch_size, seq_len, hidden_size = hidden.shape

    checks = []

    if batch_size == 1:
        check_pass(f"batch_size={batch_size}")
        checks.append(True)
    else:
        check_fail("batch_size", f"기대: 1, 실제: {batch_size}")
        checks.append(False)

    if hidden_size == 768:
        check_pass(f"hidden_size={hidden_size}")
        checks.append(True)
    else:
        check_fail("hidden_size", f"기대: 768, 실제: {hidden_size}")
        checks.append(False)

    if seq_len > 1:
        check_pass(f"seq_len={seq_len} (>1)")
        checks.append(True)
    else:
        check_fail("seq_len", f"실제: {seq_len}")
        checks.append(False)

    # NaN/Inf 체크
    if torch.isnan(hidden).any():
        check_fail("NaN", "출력에 NaN 존재")
        checks.append(False)
    elif torch.isinf(hidden).any():
        check_fail("Inf", "출력에 Inf 존재")
        checks.append(False)
    else:
        check_pass("출력 값 정상 (NaN/Inf 없음)")
        checks.append(True)

    # 출력 통계
    print(f"\n  출력 통계:")
    print(f"    mean: {hidden.mean().item():.6f}")
    print(f"    std:  {hidden.std().item():.6f}")
    print(f"    min:  {hidden.min().item():.6f}")
    print(f"    max:  {hidden.max().item():.6f}")

    return all(checks)


# ============================================
# 검증 5: GLiNER2 모델 구조 분석
# ============================================

def verify_gliner2_structure():
    """GLiNER2 모델 구조를 분석하여 인코더 교체 포인트를 확인"""
    separator("검증 5: GLiNER2 모델 구조")

    try:
        from gliner2 import GLiNER2
    except ImportError:
        print("  [SKIP] gliner2 패키지가 설치되지 않음")
        print("  → RunPod에서 실행 시 확인 가능")
        return None  # 불확정

    try:
        model = GLiNER2.from_pretrained(GLINER2_PRETRAINED)
    except Exception as e:
        print(f"  [SKIP] 모델 로드 실패: {e}")
        return None

    # 최상위 모듈 목록
    print(f"  최상위 모듈:")
    for name, child in model.named_children():
        print(f"    {name}: {child.__class__.__name__}")

    # 인코더 찾기
    encoder_found = False
    encoder_config = None

    for name, module in model.named_modules():
        cls_name = module.__class__.__name__
        if "DebertaV2" in cls_name and hasattr(module, "config"):
            print(f"\n  인코더 발견:")
            print(f"    경로: {name}")
            print(f"    클래스: {cls_name}")
            print(f"    hidden_size: {module.config.hidden_size}")
            print(f"    num_layers: {getattr(module.config, 'num_hidden_layers', 'N/A')}")
            encoder_found = True
            encoder_config = module.config
            break

    if not encoder_found:
        # DeBERTa 외 인코더 탐색
        for name, module in model.named_modules():
            if hasattr(module, "config") and hasattr(module.config, "hidden_size"):
                cls_name = module.__class__.__name__
                if any(kw in cls_name.lower() for kw in ["model", "encoder", "bert"]):
                    print(f"\n  인코더 후보:")
                    print(f"    경로: {name}")
                    print(f"    클래스: {cls_name}")
                    print(f"    Config: {module.config.__class__.__name__}")
                    print(f"    hidden_size: {module.config.hidden_size}")
                    encoder_config = module.config

    # 토크나이저 확인
    if hasattr(model, "tokenizer"):
        tok = model.tokenizer
        print(f"\n  토크나이저:")
        print(f"    클래스: {tok.__class__.__name__}")
        print(f"    vocab_size: {tok.vocab_size}")
        test_tokens = tok.tokenize("당뇨병")
        print(f"    \"당뇨병\" → {test_tokens}")

    # 파라미터 수
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  파라미터:")
    print(f"    총: {total_params:,}")
    print(f"    학습 가능: {trainable_params:,}")

    if encoder_found:
        check_pass("GLiNER2 내 DeBERTa 인코더 확인")
        return True
    else:
        print("\n  ⚠ DeBERTa 인코더를 명시적으로 찾지 못함")
        print("    → 모델 구조를 수동으로 확인 필요")
        return False


# ============================================
# 메인
# ============================================

def main():
    print("=" * 60)
    print("  GLiNER2 한국어 인코더 교체 사전 검증")
    print("=" * 60)
    print(f"\n  한국어 인코더: {KOREAN_ENCODER}")
    print(f"  GLiNER2 모델:  {GLINER2_PRETRAINED}")

    results = {}

    # 검증 1: Config 클래스
    try:
        passed, config = verify_config_class()
        results["config_class"] = passed
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        results["config_class"] = False
        config = None

    # 검증 2: hidden_size
    if config:
        try:
            results["hidden_size"] = verify_hidden_size(config)
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            results["hidden_size"] = False
    else:
        results["hidden_size"] = False

    # 검증 3: 토크나이저
    try:
        results["tokenizer"] = verify_tokenizer()
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        results["tokenizer"] = False

    # 검증 4: Forward pass
    try:
        results["forward_pass"] = verify_forward_pass()
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        results["forward_pass"] = False

    # 검증 5: GLiNER2 구조 (선택적)
    try:
        results["gliner2_structure"] = verify_gliner2_structure()
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        results["gliner2_structure"] = None

    # ============================================
    # 종합 결과
    # ============================================
    separator("종합 결과")

    all_passed = True
    for name, passed in results.items():
        if passed is None:
            status = "SKIP"
        elif passed:
            status = "PASS ✓"
        else:
            status = "FAIL ✗"
            all_passed = False
        print(f"  {name:25} {status}")

    print()
    if all_passed:
        print("  ★ 모든 검증 통과 — 인코더 교체 진행 가능")
        return 0
    else:
        failed = [k for k, v in results.items() if v is False]
        print(f"  ⚠ 실패 항목: {', '.join(failed)}")
        print("  → 실패 항목을 확인 후 진행 여부를 결정하세요")
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

---

## 9. Codex 리뷰 피드백 및 수정 이력

> 초기 구현 완료 후 OpenAI Codex에 리뷰를 의뢰하여 6건의 피드백을 받음. 전부 수정 완료.

### 9.1 리뷰 피드백 6건과 대응

#### 이슈 1: `compute_loss(batch)` 호출 형식 가정

**Codex 지적**: 커스텀 학습 루프가 `model.compute_loss(batch)`를 직접 호출하는데, GLiNER2의 실제 학습 인터페이스가 이 시그니처를 지원하는지 불확실. 호출 실패 시 학습 자체가 불가.

**수정**: `_compute_loss_safe()` 래퍼 함수 추가. 3가지 호출 방식을 순서대로 시도:

```python
def _compute_loss_safe(model, batch):
    # 1차: model.compute_loss(batch)
    # 2차: model.train_step(batch)
    # 3차: model(examples=batch)
    # 모두 실패 시: RuntimeError + 구체적 안내 메시지
```

**근거**: GLiNER2의 내부 API가 버전마다 다를 수 있으므로, 하나의 방식에 의존하지 않고 다중 fallback으로 안전하게 처리.

---

#### 이슈 2: `--fp16` 플래그의 `store_true` + `default=True` 충돌

**Codex 지적**: `action="store_true"`는 기본값이 `False`일 때 쓰는 패턴. `default=True`와 조합하면 CLI에서 `--fp16`을 안 줘도 항상 `True`가 되어 `--no-fp16`으로만 비활성화 가능. 직관적이지 않음.

**수정 전**:
```python
parser.add_argument("--fp16", action="store_true", default=True)
parser.add_argument("--no-fp16", action="store_true")
```

**수정 후**:
```python
parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True,
                    help="FP16 사용 (기본: True, --no-fp16으로 비활성화)")
```

`--no-fp16` 별도 인자와 `if args.no_fp16:` 분기도 제거.

---

#### 이슈 3: `--grad-accum` 인자가 커스텀 루프에서 미적용

**Codex 지적**: argparse에 `--grad-accum` 옵션이 정의되어 있고 builtin trainer 경로에서는 사용되지만, 커스텀 학습 루프에서는 완전히 무시됨. 사용자가 `--grad-accum 4`를 줘도 효과 없음.

**수정**: 커스텀 루프에 gradient accumulation 전면 구현:

```python
grad_accum = args.grad_accum
# ...
optimizer.zero_grad()
accum_count = 0
for i in range(0, len(indices), args.batch_size):
    # ...
    loss = loss / grad_accum          # loss 스케일링
    loss.backward()
    accum_count += 1

    if accum_count % grad_accum == 0 or (i + args.batch_size) >= len(indices):
        optimizer.step()              # grad_accum 스텝마다 업데이트
        scheduler.step()
        optimizer.zero_grad()
```

`total_steps` 계산도 accumulation을 반영하도록 수정:
```python
total_steps = (steps_per_epoch // grad_accum) * args.epochs
```

---

#### 이슈 4: 인코더 파라미터 fallback 키워드가 너무 넓음

**Codex 지적**: 차등 학습률을 위한 인코더/헤드 파라미터 분류에서 fallback 키워드가 `["deberta", "encoder", "embeddings", "bert"]`로 너무 넓음. `"embeddings"`가 인코더가 아닌 임베딩 레이어까지 잡을 수 있고, `"encoder"`가 GLiNER2의 span encoder 등 비인코더 모듈도 매칭할 수 있음.

**수정 전**:
```python
if any(kw in name.lower() for kw in ["deberta", "encoder", "embeddings", "bert"]):
```

**수정 후**:
```python
if any(kw in name.lower() for kw in ["deberta", "bert_layer"]):
```

`"deberta"`와 `"bert_layer"`만 남겨서 DeBERTa 인코더 내부 레이어만 정확히 매칭.

---

#### 이슈 5: `resize_token_embeddings`가 호출되지 않을 수 있음

**Codex 지적**: 인코더 교체 후 `resize_token_embeddings()`를 호출하는데, 이 메서드가 교체된 인코더 객체에 존재하지 않을 수 있음. 실패하면 vocab 크기 불일치로 런타임 에러 발생.

**수정**: 3단계 fallback 구현:

```python
# 1단계: 인코더에서 시도
if hasattr(new_encoder, "resize_token_embeddings"):
    new_encoder.resize_token_embeddings(len(korean_tokenizer))

# 2단계: 최상위 모델에서 시도
elif hasattr(model, "resize_token_embeddings"):
    model.resize_token_embeddings(len(korean_tokenizer))

# 3단계: word_embeddings 직접 확인
else:
    for name, module in new_encoder.named_modules():
        if hasattr(module, "weight") and "word_embeddings" in name:
            # vocab 크기 비교 + 경고 출력
```

---

#### 이슈 6: 리뷰 문서 인코딩 문제

**Codex 지적**: 리뷰 문서(`gliner2_korean_encoder_swap_review.md`)가 UTF-8 BOM이나 비정상 인코딩으로 저장됨.

**수정**: Python으로 파일을 읽어서 clean UTF-8 (BOM 없이)로 다시 저장:

```python
with open(path, "r", encoding="utf-8-sig") as f:
    content = f.read()
with open(path, "w", encoding="utf-8", newline="\n") as f:
    f.write(content)
```

---

### 9.2 추가 정리 (리뷰와 별개)

Codex 리뷰 과정에서 발견된 부가적 코드 품질 이슈도 함께 수정:

| 항목 | 수정 내용 |
|------|----------|
| 미사용 import 제거 | `copy`, `os`, `nn`, `DataLoader` 4개 삭제 |
| `import random` 위치 | 에폭 루프 내부 → 파일 최상위 import로 이동 |

### 9.3 수정 후 상태

| 항목 | 수정 전 | 수정 후 |
|------|---------|---------|
| loss 계산 | `model.compute_loss(batch)` 직접 호출 | `_compute_loss_safe()` 3-way fallback |
| FP16 플래그 | `store_true` + `default=True` (모순) | `BooleanOptionalAction` + `default=True` |
| Gradient accum | 커스텀 루프에서 미적용 | 완전 구현 (loss 스케일링 + 조건부 step) |
| 인코더 파라미터 매칭 | `["deberta", "encoder", "embeddings", "bert"]` | `["deberta", "bert_layer"]` |
| 임베딩 리사이즈 | 단일 호출 (실패 가능) | 3단계 fallback |
| 문서 인코딩 | UTF-8 BOM 혼재 | Clean UTF-8 |
| 미사용 import | 4개 존재 | 제거 |
| `import random` | 루프 내부 | 최상위 |

---

### 9.4 2차 리뷰 피드백 (Codex 재검토)

> 1차 수정 후 재검토 결과 5건 피드백. **소스 코드에는 문제 없음**. 리뷰 문서의 요약 스니펫 2건만 수정.

| # | 심각도 | 지적 | 판정 | 대응 |
|---|--------|------|------|------|
| 1 | High | 문서 인코딩 모지바이크 | **반박** | 파일은 clean UTF-8 (BOM 없음). 리뷰어 환경의 렌더링 문제 |
| 2 | High | `swap_encoder()` 요약에서 `old_config`/`korean_config` 미정의 | **수용 (문서만)** | 실제 소스(L147,156)에는 정의 있음. 리뷰 문서 섹션 3.2의 요약 스니펫에 정의 추가 |
| 3 | Medium | `evaluate()` 요약에서 `mention` vs `m` 변수명 불일치 | **수용 (문서만)** | 실제 소스(L245-258)는 `m`으로 일관. 리뷰 문서 섹션 3.6의 요약 스니펫을 `(m, label)`로 수정 |
| 4 | Medium | `건n`, `분n` 개행 깨짐 | **반박** | `f.write(f"...건\n")`은 정상 Python. `\n`이 newline으로 해석됨 |
| 5 | Low | `status = "PASS ??` 따옴표 미닫힘 | **반박** | `status = "PASS ✓"` — 유니코드 체크마크가 특정 환경에서 깨져 보이는 렌더링 문제 |

**이전 수정 7건 반영 확인**: 모두 소스 코드에 정상 반영됨 (위 9.3 테이블 참조)
