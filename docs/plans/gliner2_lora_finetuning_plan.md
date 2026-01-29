# GLiNER2 LoRA 파인튜닝 계획

> 작성일: 2026-01-06

---

## 1. 목적

GLiNER2의 스키마 기반 인터페이스를 활용하면서, LoRA 파인튜닝으로 한국어 의료 NER 성능을 개선할 수 있는지 검증한다.

### 배경

| 항목 | 현황 |
|------|------|
| GLiNER2 기본 모델 | 영어 최적화 (DeBERTa-v3-base) |
| 한국어 인식률 | 33% (테스트 결과) |
| GLiNER v2.1 (다국어) | 92% |
| GLiNER2 장점 | 스키마 인터페이스, 엔티티 설명 포함 |

**가설**: LoRA 파인튜닝으로 한국어 의료 도메인에 적응하면, GLiNER2의 구조적 장점을 유지하면서 한국어 성능을 개선할 수 있다.

---

## 2. 테스트 범위

### 이번 테스트 (Phase 1)

- **목표**: LoRA 파인튜닝 파이프라인이 작동하는지 확인
- **데이터**: KBMC 샘플 (~50개)
- **평가**: Before/After 비교

### 실제 학습 (Phase 2) - 추후 진행

- 학습 데이터 구축 방안 검토 후 결정
- Gazetteer 활용 실버셋 생성
- 대규모 데이터로 본격 학습

---

## 3. 테스트 데이터 준비

### 3.1 KBMC 데이터 형식

원본 (KBMC):
```json
{
  "text": "제2형 당뇨병 환자의 간 기능이 악화되어 인슐린 주사로 치료를 변경했다.",
  "entities": [
    {"type": "Disease", "text": "제2형 당뇨병", "start": 0, "end": 7},
    {"type": "Body", "text": "간", "start": 12, "end": 13},
    {"type": "Treatment", "text": "인슐린 주사", "start": 24, "end": 30}
  ]
}
```

### 3.2 GLiNER2 학습 형식으로 변환

```python
from gliner2.training.data import InputExample

example = InputExample(
    text="제2형 당뇨병 환자의 간 기능이 악화되어 인슐린 주사로 치료를 변경했다.",
    entities={
        "Disease": ["제2형 당뇨병"],
        "Body": ["간"],
        "Treatment": ["인슐린 주사"]
    },
    entity_descriptions={
        "Disease": "질병, 증상, 의학적 상태",
        "Body": "신체 부위, 장기, 해부학적 구조",
        "Treatment": "치료법, 시술, 약물 투여"
    }
)
```

---

## 4. 학습 설정

### 4.1 LoRA 설정

```python
from gliner2.training.trainer import TrainingConfig

config = TrainingConfig(
    output_dir="./gliner2_korean_medical",

    # LoRA 설정
    use_lora=True,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    lora_target_modules=["encoder", "span_rep", "classifier"],

    # 학습 설정
    num_epochs=20,
    batch_size=8,
    task_lr=5e-4,
    warmup_ratio=0.1,

    # 평가
    eval_strategy="epoch",
    save_best=True,
    early_stopping=True,
    early_stopping_patience=5
)
```

### 4.2 하이퍼파라미터 선택 근거

| 파라미터 | 값 | 이유 |
|----------|-----|------|
| lora_r | 16 | 기본값, 충분한 표현력 |
| lora_alpha | 32 | 2 * r (권장) |
| lora_dropout | 0.1 | 과적합 방지 |
| num_epochs | 20 | 소량 데이터에 충분 |
| batch_size | 8 | 메모리 제약 고려 |
| task_lr | 5e-4 | LoRA 권장 학습률 |

---

## 5. 평가 계획

### 5.1 테스트 데이터

학습에 사용하지 않은 10개 샘플로 평가

### 5.2 평가 지표

| 지표 | 설명 |
|------|------|
| Entity-level F1 | 엔티티 단위 정확도 |
| Type-level F1 | Disease, Body, Treatment 각각 |
| Before/After 비교 | 파인튜닝 전후 성능 변화 |

### 5.3 성공 기준

- **최소**: 파인튜닝 후 인식률 50% 이상 (현재 33%)
- **목표**: 인식률 70% 이상
- **이상적**: GLiNER v2.1 수준 (90%+)

---

## 6. 작업 절차

### Step 1: 데이터 준비
- [ ] KBMC 샘플 데이터 로드
- [ ] GLiNER2 InputExample 형식으로 변환
- [ ] Train/Test 분할 (40/10)

### Step 2: 베이스라인 측정
- [ ] 파인튜닝 전 GLiNER2로 테스트셋 평가
- [ ] 결과 기록

### Step 3: LoRA 파인튜닝
- [ ] TrainingConfig 설정
- [ ] 학습 실행
- [ ] 체크포인트 저장

### Step 4: 평가
- [ ] 파인튜닝 후 테스트셋 평가
- [ ] Before/After 비교
- [ ] 결과 문서화

### Step 5: 결론 도출
- [ ] LoRA 파인튜닝 효과 판단
- [ ] 다음 단계 결정 (데이터 확보 vs 인코더 교체)

---

## 7. 예상 리스크

| 리스크 | 영향 | 대응 |
|--------|------|------|
| LoRA만으로 성능 개선 부족 | 인코더 자체가 한국어 토큰화 못함 | 인코더 교체 검토 |
| 과적합 | 소량 데이터로 일반화 어려움 | 테스트셋 분리, early stopping |
| 메모리 부족 | 학습 불가 | batch_size 축소, gradient accumulation |

---

## 8. 다음 단계 (Phase 2 준비)

LoRA 테스트 결과에 따라:

### 성공 시
1. 학습 데이터 대규모 구축
2. Gazetteer 활용 실버셋 생성
3. 본격 파인튜닝

### 실패 시
1. 한국어 인코더로 교체 검토
   - `klue/roberta-base`
   - `monologg/koelectra-base-v3`
2. 또는 GLiNER v2.1 기반으로 방향 전환

---

## 9. 관련 문서

- `project/ner/docs/study/gliner2.md` - GLiNER2 개요
- `project/ner/docs/study/gliner2_test_results.md` - 테스트 결과
- `C:/Jimin/GLiNER2/tutorial/9-training.md` - 학습 튜토리얼
- `C:/Jimin/GLiNER2/tutorial/10-lora_adapters.md` - LoRA 가이드

---

## 10. 파일 구조 (예정)

```
project/ner/
├── docs/
│   ├── study/
│   │   ├── gliner2.md
│   │   └── gliner2_test_results.md
│   └── plans/
│       └── gliner2_lora_finetuning_plan.md  (본 문서)
├── data/
│   └── gliner2_test/
│       ├── train.jsonl
│       └── test.jsonl
├── scripts/
│   └── gliner2/
│       ├── prepare_data.py
│       ├── train_lora.py
│       └── evaluate.py
└── models/
    └── gliner2_korean_medical/
        └── (체크포인트)
```
