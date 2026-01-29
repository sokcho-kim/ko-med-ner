"""
GLiNER2 LoRA 파인튜닝 테스트 (KBMC 데이터)
- 목적: LoRA 파인튜닝이 한국어 의료 NER에 효과가 있는지 검증
- 데이터: SungJoo/KBMC (50개 샘플)
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
from collections import defaultdict

# ============================================
# 1. 설정
# ============================================
TRAIN_SIZE = 40
TEST_SIZE = 10
LABELS = ["Disease", "Body", "Treatment"]
ENTITY_DESCRIPTIONS = {
    "Disease": "질병, 증상, 의학적 상태 (예: 당뇨병, 고혈압, 폐렴)",
    "Body": "신체 부위, 장기, 해부학적 구조 (예: 간, 심장, 폐)",
    "Treatment": "치료법, 시술, 약물 투여 (예: 인슐린 주사, 항생제 치료)"
}

print("=" * 60)
print("GLiNER2 LoRA 파인튜닝 테스트")
print("=" * 60)

# ============================================
# 2. 데이터 로드
# ============================================
print("\n[Step 1/5] KBMC 데이터 로드...")
from datasets import load_dataset

dataset = load_dataset("SungJoo/KBMC")
full_data = dataset['train']
print(f"전체 데이터: {len(full_data)}개")

# 샘플링 (50개만)
sample_indices = list(range(TRAIN_SIZE + TEST_SIZE))
sampled_data = full_data.select(sample_indices)
print(f"샘플 데이터: {len(sampled_data)}개")

# ============================================
# 3. GLiNER2 형식으로 변환
# ============================================
print("\n[Step 2/5] GLiNER2 형식 변환...")

def parse_kbmc_to_gliner2(sentence, tags):
    """KBMC BIO 형식을 GLiNER2 형식으로 변환"""
    tokens = sentence.split()
    tag_list = tags.split()

    # 텍스트 복원 (공백 제거)
    text = ''.join(tokens)

    # 엔티티 추출
    entities_by_type = defaultdict(list)
    current_entity = None
    current_type = None

    for token, tag in zip(tokens, tag_list):
        if tag.endswith('-B'):
            if current_entity:
                entities_by_type[current_type].append(current_entity)
            current_type = tag.replace('-B', '')
            current_entity = token
        elif tag.endswith('-I') and current_entity:
            current_entity += token
        else:
            if current_entity:
                entities_by_type[current_type].append(current_entity)
                current_entity = None
                current_type = None

    if current_entity:
        entities_by_type[current_type].append(current_entity)

    return text, dict(entities_by_type)

# 변환
from gliner2.training.data import InputExample

train_examples = []
test_examples = []

for i, sample in enumerate(sampled_data):
    text, entities = parse_kbmc_to_gliner2(sample['Sentence'], sample['Tags'])

    # 빈 엔티티 제외하고 InputExample 생성
    if entities:
        example = InputExample(
            text=text,
            entities=entities,
            entity_descriptions=ENTITY_DESCRIPTIONS
        )

        if i < TRAIN_SIZE:
            train_examples.append(example)
        else:
            test_examples.append(example)

print(f"Train: {len(train_examples)}개, Test: {len(test_examples)}개")

# 샘플 출력
print("\n변환 샘플:")
ex = train_examples[0]
print(f"  Text: {ex.text[:50]}...")
print(f"  Entities: {ex.entities}")

# ============================================
# 4. 베이스라인 측정 (파인튜닝 전)
# ============================================
print("\n[Step 3/5] 베이스라인 측정 (파인튜닝 전)...")
from gliner2 import GLiNER2

model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

def evaluate(model, test_examples, threshold=0.5):
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

baseline_result = evaluate(model, test_examples, threshold=0.3)
print(f"\n베이스라인 결과:")
print(f"  Precision: {baseline_result['precision']:.4f}")
print(f"  Recall:    {baseline_result['recall']:.4f}")
print(f"  F1:        {baseline_result['f1']:.4f}")
print(f"  (TP={baseline_result['tp']}, FP={baseline_result['fp']}, FN={baseline_result['fn']})")

# ============================================
# 5. LoRA 파인튜닝
# ============================================
print("\n[Step 4/5] LoRA 파인튜닝...")
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

config = TrainingConfig(
    output_dir="./gliner2_korean_medical",

    # LoRA 설정
    use_lora=True,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    lora_target_modules=["encoder", "span_rep", "classifier"],

    # 학습 설정
    num_epochs=10,  # 테스트용으로 줄임
    batch_size=4,
    task_lr=5e-4,
    warmup_ratio=0.1,

    # 평가
    eval_strategy="epoch",
    save_best=True,

    # 기타
    fp16=False,  # CPU에서는 비활성화
    num_workers=0,
    seed=42
)

# 모델 다시 로드 (깨끗한 상태에서 시작)
model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

trainer = GLiNER2Trainer(model, config)

print(f"학습 시작... (epochs={config.num_epochs}, batch={config.batch_size})")
results = trainer.train(train_data=train_examples)

print(f"\n학습 완료!")
print(f"  Total steps: {results.get('total_steps', 'N/A')}")

# ============================================
# 6. 파인튜닝 후 평가
# ============================================
print("\n[Step 5/5] 파인튜닝 후 평가...")

# 베스트 모델 로드
try:
    finetuned_model = GLiNER2.from_pretrained("./gliner2_korean_medical/best")
except:
    finetuned_model = model  # 베스트가 없으면 현재 모델 사용

finetuned_result = evaluate(finetuned_model, test_examples, threshold=0.3)
print(f"\n파인튜닝 후 결과:")
print(f"  Precision: {finetuned_result['precision']:.4f}")
print(f"  Recall:    {finetuned_result['recall']:.4f}")
print(f"  F1:        {finetuned_result['f1']:.4f}")
print(f"  (TP={finetuned_result['tp']}, FP={finetuned_result['fp']}, FN={finetuned_result['fn']})")

# ============================================
# 7. 결과 비교
# ============================================
print("\n" + "=" * 60)
print("결과 비교")
print("=" * 60)

print(f"\n{'':20} {'Baseline':>12} {'Finetuned':>12} {'Delta':>12}")
print("-" * 60)
print(f"{'Precision':20} {baseline_result['precision']:>12.4f} {finetuned_result['precision']:>12.4f} {finetuned_result['precision']-baseline_result['precision']:>+12.4f}")
print(f"{'Recall':20} {baseline_result['recall']:>12.4f} {finetuned_result['recall']:>12.4f} {finetuned_result['recall']-baseline_result['recall']:>+12.4f}")
print(f"{'F1':20} {baseline_result['f1']:>12.4f} {finetuned_result['f1']:>12.4f} {finetuned_result['f1']-baseline_result['f1']:>+12.4f}")

# 결과 저장
output_file = "./gliner2_lora_test_results.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write("GLiNER2 LoRA 파인튜닝 테스트 결과\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Train: {len(train_examples)}개, Test: {len(test_examples)}개\n")
    f.write(f"Epochs: {config.num_epochs}\n")
    f.write(f"LoRA r: {config.lora_r}, alpha: {config.lora_alpha}\n\n")
    f.write("베이스라인:\n")
    f.write(f"  P={baseline_result['precision']:.4f}, R={baseline_result['recall']:.4f}, F1={baseline_result['f1']:.4f}\n\n")
    f.write("파인튜닝 후:\n")
    f.write(f"  P={finetuned_result['precision']:.4f}, R={finetuned_result['recall']:.4f}, F1={finetuned_result['f1']:.4f}\n\n")
    f.write(f"Delta F1: {finetuned_result['f1']-baseline_result['f1']:+.4f}\n")

print(f"\n결과 저장: {output_file}")
print("\n완료!")
