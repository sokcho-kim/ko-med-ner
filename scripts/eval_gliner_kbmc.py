"""
GLiNER Zero-shot KBMC 평가 스크립트
- 모델: urchade/gliner_multi-v2.1
- 데이터: SungJoo/KBMC (동일 test set 615개)
- 목적: KoELECTRA와 공정 비교
"""

import warnings
warnings.filterwarnings('ignore')

from datasets import load_dataset
from gliner import GLiNER
from collections import defaultdict
import re

# ============================================
# 1. 설정
# ============================================
MODEL_NAME = "urchade/gliner_multi-v2.1"
LABELS = ["Disease", "Body", "Treatment"]
THRESHOLD = 0.3

print("=" * 50)
print("GLiNER Zero-shot KBMC 평가")
print("=" * 50)

# ============================================
# 2. 데이터 로드 (동일 split)
# ============================================
print("\n[1/4] 데이터 로드 중...")
dataset = load_dataset("SungJoo/KBMC")
dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)
test_set = dataset['test']
print(f"Test set: {len(test_set)} 문장")

# ============================================
# 3. 모델 로드
# ============================================
print("\n[2/4] GLiNER 모델 로드 중...")
model = GLiNER.from_pretrained(MODEL_NAME)
print(f"모델 로드 완료: {MODEL_NAME}")

# ============================================
# 4. 라벨 매핑
# ============================================
# KBMC: Disease-B, Disease-I, Body-B, Body-I, Treatment-B, Treatment-I, O
# GLiNER: Disease, Body, Treatment

def parse_kbmc_entities(sentence, tags):
    """KBMC 형식에서 엔티티 추출"""
    tokens = sentence.split()
    tag_list = tags.split()

    entities = []
    current_entity = None
    current_type = None
    start_idx = 0

    char_pos = 0
    for i, (token, tag) in enumerate(zip(tokens, tag_list)):
        if tag.endswith('-B'):
            # 이전 엔티티 저장
            if current_entity:
                entities.append({
                    'text': current_entity,
                    'label': current_type,
                })
            # 새 엔티티 시작
            current_type = tag.replace('-B', '')
            current_entity = token
        elif tag.endswith('-I') and current_entity:
            # 엔티티 계속
            current_entity += token
        else:
            # O 태그
            if current_entity:
                entities.append({
                    'text': current_entity,
                    'label': current_type,
                })
                current_entity = None
                current_type = None

    # 마지막 엔티티
    if current_entity:
        entities.append({
            'text': current_entity,
            'label': current_type,
        })

    return entities

def extract_gliner_entities(model, text, labels, threshold=0.3):
    """GLiNER로 엔티티 추출"""
    entities = model.predict_entities(text, labels, threshold=threshold)
    return [{'text': e['text'], 'label': e['label']} for e in entities]

# ============================================
# 5. 평가
# ============================================
print("\n[3/4] 평가 중...")

# 엔티티 단위 평가
true_positives = 0
false_positives = 0
false_negatives = 0

# 라벨별 통계
label_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

for idx, sample in enumerate(test_set):
    if idx % 100 == 0:
        print(f"  진행: {idx}/{len(test_set)}")

    sentence = sample['Sentence']
    tags = sample['Tags']

    # 정답 엔티티
    gold_entities = parse_kbmc_entities(sentence, tags)
    gold_set = set((e['text'], e['label']) for e in gold_entities)

    # 공백 유지한 텍스트로 예측 (이전 테스트에서 더 나은 결과)
    pred_entities = extract_gliner_entities(model, sentence, LABELS, THRESHOLD)
    pred_set = set((e['text'], e['label']) for e in pred_entities)

    # TP, FP, FN 계산
    tp = gold_set & pred_set
    fp = pred_set - gold_set
    fn = gold_set - pred_set

    true_positives += len(tp)
    false_positives += len(fp)
    false_negatives += len(fn)

    # 라벨별 통계
    for text, label in tp:
        label_stats[label]['tp'] += 1
    for text, label in fp:
        label_stats[label]['fp'] += 1
    for text, label in fn:
        label_stats[label]['fn'] += 1

# ============================================
# 6. 결과 계산
# ============================================
print("\n[4/4] 결과 계산...")

precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print("\n" + "=" * 50)
print("GLiNER Zero-shot 평가 결과")
print("=" * 50)

print(f"\n전체 결과:")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1:        {f1:.4f}")

print(f"\n라벨별 결과:")
for label in LABELS:
    stats = label_stats[label]
    tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0
    print(f"  {label}: P={p:.4f}, R={r:.4f}, F1={f:.4f} (TP={tp}, FP={fp}, FN={fn})")

print(f"\n통계:")
print(f"  True Positives:  {true_positives}")
print(f"  False Positives: {false_positives}")
print(f"  False Negatives: {false_negatives}")

# 결과 저장
with open("gliner_kbmc_results.txt", "w", encoding="utf-8") as f:
    f.write("GLiNER Zero-shot KBMC 평가 결과\n")
    f.write("=" * 40 + "\n")
    f.write(f"Model: {MODEL_NAME}\n")
    f.write(f"Test Size: {len(test_set)}\n")
    f.write(f"Threshold: {THRESHOLD}\n")
    f.write(f"\n결과:\n")
    f.write(f"  Precision: {precision:.4f}\n")
    f.write(f"  Recall:    {recall:.4f}\n")
    f.write(f"  F1:        {f1:.4f}\n")
    f.write(f"\n라벨별:\n")
    for label in LABELS:
        stats = label_stats[label]
        tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        f.write(f"  {label}: P={p:.4f}, R={r:.4f}, F1={f:.4f}\n")

print("\n결과 저장: gliner_kbmc_results.txt")
print("\n완료!")
