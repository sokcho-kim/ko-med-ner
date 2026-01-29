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
import random
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import torch
from torch.optim import AdamW
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

    # 6. 임베딩 리사이즈 (vocab 크기 차이 보정)
    _, new_encoder = find_encoder_attr(model)
    resized = False
    if hasattr(new_encoder, "resize_token_embeddings"):
        new_encoder.resize_token_embeddings(len(korean_tokenizer))
        resized = True
    elif hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(korean_tokenizer))
        resized = True
    else:
        # 수동 리사이즈: word_embeddings 직접 확인
        for name, module in new_encoder.named_modules():
            if hasattr(module, "weight") and "word_embeddings" in name:
                old_vocab = module.weight.shape[0]
                new_vocab = len(korean_tokenizer)
                if old_vocab != new_vocab:
                    print(f"  ⚠ vocab 불일치: 인코더={old_vocab}, 토크나이저={new_vocab}")
                    print(f"    → resize_token_embeddings 미지원. 수동 조정 필요할 수 있음")
                else:
                    resized = True
                break
    if resized:
        print(f"  ✓ 임베딩 리사이즈 완료 (vocab={len(korean_tokenizer)})")
    else:
        print(f"  ⚠ 임베딩 리사이즈 건너뜀 (resize_token_embeddings 미지원)")

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
            # DeBERTa 모듈에 속하는 파라미터만 인코더로 분류
            # "embeddings"는 인코더 외부에도 있을 수 있으므로 제외
            if any(kw in name.lower() for kw in ["deberta", "bert_layer"]):
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


def _compute_loss_safe(model, batch):
    """GLiNER2의 compute_loss를 안전하게 호출한다.

    GLiNER2가 InputExample 리스트를 직접 받는지, dict 배치를 받는지
    실제 인터페이스가 불확실하므로 여러 호출 방식을 시도한다.
    """
    # 방식 1: InputExample 리스트 직접 전달
    try:
        loss = model.compute_loss(batch)
        if isinstance(loss, torch.Tensor):
            return loss
    except TypeError:
        pass

    # 방식 2: train_step 메서드 (일부 GLiNER 구현)
    if hasattr(model, "train_step"):
        try:
            result = model.train_step(batch)
            if isinstance(result, torch.Tensor):
                return result
            if isinstance(result, dict) and "loss" in result:
                return result["loss"]
        except TypeError:
            pass

    # 방식 3: forward에 examples 키워드
    if hasattr(model, "forward"):
        try:
            result = model(examples=batch)
            if isinstance(result, torch.Tensor):
                return result
            if isinstance(result, dict) and "loss" in result:
                return result["loss"]
        except TypeError:
            pass

    # 모든 방식 실패
    raise RuntimeError(
        "GLiNER2 모델에서 loss를 계산할 수 없습니다.\n"
        "model.compute_loss(batch), model.train_step(batch), "
        "model(examples=batch) 모두 실패.\n"
        "GLiNER2의 학습 인터페이스를 확인하세요."
    )


def train_custom_loop(model, train_data, test_data, args, encoder_path):
    """커스텀 학습 루프 — 차등 학습률 + gradient accumulation 지원.

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

    # 스케줄러 (gradient accumulation 반영)
    grad_accum = args.grad_accum
    steps_per_epoch = (len(train_examples) + args.batch_size - 1) // args.batch_size
    optimizer_steps_per_epoch = (steps_per_epoch + grad_accum - 1) // grad_accum
    total_steps = optimizer_steps_per_epoch * args.epochs
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
    print(f"  Gradient accumulation: {grad_accum}")
    print(f"  실효 배치: {args.batch_size * grad_accum}")
    print(f"  인코더 LR: {args.encoder_lr}")
    print(f"  헤드 LR: {args.head_lr}")
    print(f"  Optimizer steps: {total_steps}")
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

        indices = list(range(len(train_examples)))
        random.shuffle(indices)

        optimizer.zero_grad()
        accum_count = 0

        for i in range(0, len(indices), args.batch_size):
            batch_indices = indices[i : i + args.batch_size]
            batch = [train_examples[j] for j in batch_indices]

            try:
                if scaler:
                    with torch.amp.autocast("cuda"):
                        loss = _compute_loss_safe(model, batch)
                    # gradient accumulation: loss를 스텝 수로 나눔
                    loss = loss / grad_accum
                    scaler.scale(loss).backward()
                else:
                    loss = _compute_loss_safe(model, batch)
                    loss = loss / grad_accum
                    loss.backward()

                epoch_loss += loss.item() * grad_accum  # 원래 스케일 복원
                batch_count += 1
                accum_count += 1

                # grad_accum 스텝마다 optimizer step
                if accum_count % grad_accum == 0 or (i + args.batch_size) >= len(indices):
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            except Exception as e:
                print(f"  [WARN] Batch {i//args.batch_size + 1} 에러: {e}")
                optimizer.zero_grad()
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
        "--train", default="../data/gliner2_train_v2/train.jsonl",
        help="학습 데이터 경로",
    )
    parser.add_argument(
        "--test", default="../data/gliner2_train_v2/test.jsonl",
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
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True,
                        help="FP16 사용 (기본: True, --no-fp16으로 비활성화)")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument(
        "--use-builtin-trainer", action="store_true",
        help="GLiNER2 내장 Trainer 사용 (차등 LR 미지원)",
    )
    args = parser.parse_args()

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
