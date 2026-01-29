"""
GLiNER2 Silver-set 파인튜닝 스크립트

사용법:
    python train_gliner2_silver.py
    python train_gliner2_silver.py --epochs 20 --batch-size 8
    python train_gliner2_silver.py --full-finetune  # LoRA 대신 전체 파인튜닝
"""

import json
import argparse
import os
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# ============================================
# 설정
# ============================================

ENTITY_DESCRIPTIONS = {
    "Disease": "질병, 증상, 의학적 상태 (예: 당뇨병, 고혈압, 폐렴, 암)",
    "Drug": "약물, 의약품, 치료제 (예: 인슐린, 아스피린, 항생제)",
    "Procedure": "의료 시술, 수술, 검사 (예: 내시경, MRI, 수술)",
    "Biomarker": "바이오마커, 검사 수치, 생체 지표 (예: 혈당, 콜레스테롤, 종양표지자)"
}

LABELS = list(ENTITY_DESCRIPTIONS.keys())


def load_data(path: str) -> list:
    """JSONL 파일 로드"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def convert_to_input_examples(data: list):
    """GLiNER2 InputExample로 변환"""
    from gliner2.training.data import InputExample

    examples = []
    for doc in data:
        example = InputExample(
            text=doc['text'],
            entities=doc['entities'],
            entity_descriptions=ENTITY_DESCRIPTIONS
        )
        examples.append(example)
    return examples


def evaluate(model, test_examples, threshold=0.3):
    """엔티티 단위 평가"""
    tp, fp, fn = 0, 0, 0
    label_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

    for ex in test_examples:
        # 정답
        gold_set = set()
        for label, mentions in ex.entities.items():
            for m in mentions:
                gold_set.add((m, label))

        # 예측
        result = model.extract_entities(ex.text, ENTITY_DESCRIPTIONS, threshold=threshold)
        pred_set = set()
        for label, mentions in result.get('entities', {}).items():
            for m in mentions:
                pred_set.add((m, label))

        # 계산
        matched = gold_set & pred_set
        tp += len(matched)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

        for text, label in matched:
            label_stats[label]['tp'] += 1
        for text, label in (pred_set - gold_set):
            label_stats[label]['fp'] += 1
        for text, label in (gold_set - pred_set):
            label_stats[label]['fn'] += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp, 'fp': fp, 'fn': fn,
        'label_stats': dict(label_stats)
    }


def print_results(name, result):
    """결과 출력"""
    print(f"\n{name}:")
    print(f"  Precision: {result['precision']:.4f}")
    print(f"  Recall:    {result['recall']:.4f}")
    print(f"  F1:        {result['f1']:.4f}")
    print(f"  (TP={result['tp']}, FP={result['fp']}, FN={result['fn']})")

    if result.get('label_stats'):
        print("\n  라벨별 성능:")
        for label in LABELS:
            if label in result['label_stats']:
                s = result['label_stats'][label]
                p = s['tp'] / (s['tp'] + s['fp']) if (s['tp'] + s['fp']) > 0 else 0
                r = s['tp'] / (s['tp'] + s['fn']) if (s['tp'] + s['fn']) > 0 else 0
                f = 2 * p * r / (p + r) if (p + r) > 0 else 0
                print(f"    {label:12} P={p:.3f} R={r:.3f} F1={f:.3f}")


def main():
    parser = argparse.ArgumentParser(description='GLiNER2 Silver-set 파인튜닝')
    parser.add_argument('--train', default='train.jsonl', help='학습 데이터')
    parser.add_argument('--test', default='test.jsonl', help='테스트 데이터')
    parser.add_argument('--output', default='gliner2_silver', help='출력 디렉토리')
    parser.add_argument('--epochs', type=int, default=15, help='에폭 수')
    parser.add_argument('--batch-size', type=int, default=8, help='배치 크기')
    parser.add_argument('--lr', type=float, default=5e-4, help='학습률')
    parser.add_argument('--lora-r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--full-finetune', action='store_true', help='LoRA 대신 전체 파인튜닝')
    parser.add_argument('--threshold', type=float, default=0.3, help='예측 임계값')
    parser.add_argument('--grad-accum', type=int, default=1, help='Gradient accumulation steps')
    args = parser.parse_args()

    print("=" * 60)
    print("GLiNER2 Silver-set 파인튜닝")
    print("=" * 60)
    print(f"\n설정:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  LoRA: {'OFF (Full finetune)' if args.full_finetune else f'ON (r={args.lora_r})'}")

    # ============================================
    # 1. 데이터 로드
    # ============================================
    print("\n[1/5] 데이터 로드...")
    train_data = load_data(args.train)
    test_data = load_data(args.test)
    print(f"  Train: {len(train_data)}건")
    print(f"  Test: {len(test_data)}건")

    # InputExample 변환
    train_examples = convert_to_input_examples(train_data)
    test_examples = convert_to_input_examples(test_data)

    # ============================================
    # 2. 모델 로드
    # ============================================
    print("\n[2/5] 모델 로드...")
    from gliner2 import GLiNER2

    model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    print("  fastino/gliner2-base-v1 로드 완료")

    # ============================================
    # 3. 베이스라인 측정
    # ============================================
    print("\n[3/5] 베이스라인 측정 (파인튜닝 전)...")
    baseline_result = evaluate(model, test_examples, threshold=args.threshold)
    print_results("베이스라인", baseline_result)

    # ============================================
    # 4. 파인튜닝
    # ============================================
    print("\n[4/5] 파인튜닝 시작...")
    from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

    # 출력 디렉토리
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = TrainingConfig(
        output_dir=str(output_dir),

        # LoRA 설정
        use_lora=not args.full_finetune,
        lora_r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.1,
        lora_target_modules=["encoder", "span_rep", "classifier"],

        # 학습 설정
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        task_lr=args.lr,
        warmup_ratio=0.1,
        gradient_accumulation_steps=args.grad_accum,

        # 평가
        eval_strategy="epoch",
        save_best=True,
        early_stopping=False,

        # 기타
        fp16=True,  # GPU에서는 활성화
        num_workers=2,
        seed=42
    )

    # 모델 다시 로드 (깨끗한 상태)
    model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    trainer = GLiNER2Trainer(model, config)

    start_time = datetime.now()
    print(f"  학습 시작: {start_time.strftime('%H:%M:%S')}")

    results = trainer.train(train_data=train_examples, eval_data=test_examples)

    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    print(f"  학습 완료: {end_time.strftime('%H:%M:%S')} ({elapsed/60:.1f}분)")

    # 학습된 모델 저장
    final_path = output_dir / "final"
    trainer.model.save_pretrained(str(final_path))
    print(f"  모델 저장: {final_path}")

    # ============================================
    # 5. 평가
    # ============================================
    print("\n[5/5] 파인튜닝 후 평가...")

    # 저장된 모델 로드 (final 사용)
    if (final_path / "config.json").exists():
        finetuned_model = GLiNER2.from_pretrained(str(final_path))
        print(f"  최종 모델 로드: {final_path}")
    else:
        finetuned_model = trainer.model
        print("  trainer 모델 사용")

    finetuned_result = evaluate(finetuned_model, test_examples, threshold=args.threshold)
    print_results("파인튜닝 후", finetuned_result)

    # ============================================
    # 결과 비교
    # ============================================
    print("\n" + "=" * 60)
    print("결과 비교")
    print("=" * 60)

    delta_f1 = finetuned_result['f1'] - baseline_result['f1']
    print(f"\n{'':20} {'Baseline':>12} {'Finetuned':>12} {'Delta':>12}")
    print("-" * 60)
    print(f"{'Precision':20} {baseline_result['precision']:>12.4f} {finetuned_result['precision']:>12.4f} {finetuned_result['precision']-baseline_result['precision']:>+12.4f}")
    print(f"{'Recall':20} {baseline_result['recall']:>12.4f} {finetuned_result['recall']:>12.4f} {finetuned_result['recall']-baseline_result['recall']:>+12.4f}")
    print(f"{'F1':20} {baseline_result['f1']:>12.4f} {finetuned_result['f1']:>12.4f} {delta_f1:>+12.4f}")

    # 결과 저장
    results_file = output_dir / "results.txt"
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("GLiNER2 Silver-set 파인튜닝 결과\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"날짜: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Train: {len(train_examples)}건, Test: {len(test_examples)}건\n")
        f.write(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}\n")
        f.write(f"LoRA: {'OFF' if args.full_finetune else f'ON (r={args.lora_r})'}\n")
        f.write(f"학습 시간: {elapsed/60:.1f}분\n\n")

        f.write("베이스라인:\n")
        f.write(f"  P={baseline_result['precision']:.4f}, R={baseline_result['recall']:.4f}, F1={baseline_result['f1']:.4f}\n\n")

        f.write("파인튜닝 후:\n")
        f.write(f"  P={finetuned_result['precision']:.4f}, R={finetuned_result['recall']:.4f}, F1={finetuned_result['f1']:.4f}\n\n")

        f.write(f"Delta F1: {delta_f1:+.4f}\n")

    print(f"\n결과 저장: {results_file}")
    print("\n완료!")


if __name__ == "__main__":
    main()
