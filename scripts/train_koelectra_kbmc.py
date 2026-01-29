"""
KoELECTRA + KBMC Baseline 학습 스크립트
- 모델: monologg/koelectra-base-v3-discriminator
- 데이터: SungJoo/KBMC
- 목적: 한국어 의료 NER baseline 구축
"""

import os
import warnings
warnings.filterwarnings('ignore')

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np

# ============================================
# 1. 설정
# ============================================
MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
OUTPUT_DIR = "./koelectra-kbmc-baseline"
EPOCHS = 3
BATCH_SIZE = 16  # GPU용
MAX_LENGTH = 128

# 라벨 정의
LABEL_LIST = ['O', 'Disease-B', 'Disease-I', 'Body-B', 'Body-I', 'Treatment-B', 'Treatment-I']
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}

print("=" * 50)
print("KoELECTRA + KBMC Baseline 학습")
print("=" * 50)

# ============================================
# 2. 데이터 로드 및 분리
# ============================================
print("\n[1/5] 데이터 로드 중...")
dataset = load_dataset("SungJoo/KBMC")

# Train/Test 분리 (90/10)
dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)
print(f"Train: {len(dataset['train'])}, Test: {len(dataset['test'])}")

# ============================================
# 3. 토크나이저 로드
# ============================================
print("\n[2/5] 토크나이저 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ============================================
# 4. 데이터 전처리
# ============================================
print("\n[3/5] 데이터 전처리 중...")

def tokenize_and_align_labels(examples):
    """KBMC 데이터를 토크나이저에 맞게 정렬"""

    # 토큰과 태그 분리
    all_tokens = [sent.split() for sent in examples['Sentence']]
    all_tags = [tags.split() for tags in examples['Tags']]

    # 토크나이징
    tokenized_inputs = tokenizer(
        all_tokens,
        truncation=True,
        max_length=MAX_LENGTH,
        is_split_into_words=True,
        padding='max_length'
    )

    labels = []
    for i, (tokens, tags) in enumerate(zip(all_tokens, all_tags)):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                # 특수 토큰 ([CLS], [SEP], [PAD])
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # 새로운 단어의 첫 번째 토큰
                if word_idx < len(tags):
                    label_ids.append(LABEL2ID.get(tags[word_idx], 0))
                else:
                    label_ids.append(0)  # O 태그
            else:
                # 같은 단어의 서브워드
                if word_idx < len(tags):
                    current_tag = tags[word_idx]
                    # B- 태그를 I- 태그로 변환
                    if current_tag.endswith('-B'):
                        i_tag = current_tag.replace('-B', '-I')
                        label_ids.append(LABEL2ID.get(i_tag, 0))
                    else:
                        label_ids.append(LABEL2ID.get(current_tag, 0))
                else:
                    label_ids.append(0)

            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# 전처리 적용
tokenized_train = dataset['train'].map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dataset['train'].column_names
)

tokenized_test = dataset['test'].map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dataset['test'].column_names
)

print(f"전처리 완료: Train {len(tokenized_train)}, Test {len(tokenized_test)}")

# ============================================
# 5. 모델 로드
# ============================================
print("\n[4/5] 모델 로드 중...")
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL_LIST),
    id2label=ID2LABEL,
    label2id=LABEL2ID
)
print(f"모델 로드 완료: {MODEL_NAME}")

# ============================================
# 6. 평가 함수
# ============================================
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    # -100 제외하고 실제 라벨만 추출
    true_labels = []
    true_predictions = []

    for pred, label in zip(predictions, labels):
        for p, l in zip(pred, label):
            if l != -100:
                true_labels.append(l)
                true_predictions.append(p)

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, true_predictions, average='weighted', zero_division=0
    )
    accuracy = accuracy_score(true_labels, true_predictions)

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# ============================================
# 7. 학습 설정
# ============================================
print("\n[5/5] 학습 시작...")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none",  # wandb 등 비활성화
    fp16=True,  # GPU 가속
)

data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# 학습 실행
train_result = trainer.train()

# ============================================
# 8. 결과 출력
# ============================================
print("\n" + "=" * 50)
print("학습 완료!")
print("=" * 50)

# 최종 평가
eval_results = trainer.evaluate()
print(f"\n최종 평가 결과:")
print(f"  Accuracy:  {eval_results['eval_accuracy']:.4f}")
print(f"  F1:        {eval_results['eval_f1']:.4f}")
print(f"  Precision: {eval_results['eval_precision']:.4f}")
print(f"  Recall:    {eval_results['eval_recall']:.4f}")

# 모델 저장
trainer.save_model(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
print(f"\n모델 저장: {OUTPUT_DIR}/final")

# 결과 저장
with open(f"{OUTPUT_DIR}/results.txt", "w", encoding="utf-8") as f:
    f.write("KoELECTRA + KBMC Baseline 결과\n")
    f.write("=" * 40 + "\n")
    f.write(f"Model: {MODEL_NAME}\n")
    f.write(f"Epochs: {EPOCHS}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
    f.write(f"Train Size: {len(dataset['train'])}\n")
    f.write(f"Test Size: {len(dataset['test'])}\n")
    f.write("\n결과:\n")
    f.write(f"  Accuracy:  {eval_results['eval_accuracy']:.4f}\n")
    f.write(f"  F1:        {eval_results['eval_f1']:.4f}\n")
    f.write(f"  Precision: {eval_results['eval_precision']:.4f}\n")
    f.write(f"  Recall:    {eval_results['eval_recall']:.4f}\n")

print("\n완료!")
