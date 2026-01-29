# GLiNER2 한국어 인코더 교체 계획

> 작성일: 2026-01-27

---

## 1. 배경

### 1.1 문제 정의

GLiNER2의 기본 인코더 DeBERTa-v3-base는 영어 중심 토크나이저를 사용하여 한국어를 음절 단위로 분절한다.

```
"당뇨병" → ['▁', '당', '뇨', '병']  (음절 분절 — 의미 정보 손실)
```

이로 인해:
- 베이스라인 F1 = 0.0833 (KBMC 데이터)
- LoRA 파인튜닝 2회 시도 결과: F1 = 0.0000 (완전 실패)
- 인코더 자체가 한국어 토큰을 의미 단위로 처리하지 못하므로, 위에 어떤 어댑터를 올려도 근본 한계를 해결할 수 없음

### 1.2 목표

GLiNER2의 pretrained head weights(프로젝션, span representation, classifier)를 최대한 재활용하면서, 인코더만 한국어용으로 교체하여 한국어 의료 NER 성능을 개선한다.

**목표 F1**: 0.50 이상 (베이스라인 0.387 대비 13%p 이상 개선)

### 1.3 데이터 현황

| 데이터셋 | Train | Test | 라벨 |
|----------|-------|------|------|
| gliner2_train_v2 | 696건 | 77건 | Disease, Drug, Procedure, Biomarker |

소스: `cg_parsed_sampled_1000_verified_final_v2_clean.jsonl` (GPT-5 검증 완료)

---

## 2. 인코더 후보 비교

### 2.1 후보 6개 체계적 비교

| # | 후보 | 아키텍처 | 파라미터 | hidden_size | Config 클래스 |
|---|------|----------|----------|-------------|--------------|
| 1 | KURE-v1 | Sentence Transformer | - | 1024 (문장 벡터) | - |
| 2 | KoELECTRA-base-v3 | ELECTRA | 110M | 768 | ElectraConfig |
| 3 | KLUE-RoBERTa-base | RoBERTa | 111M | 768 | RobertaConfig |
| 4 | XLM-RoBERTa-base | RoBERTa | 270M | 768 | XLMRobertaConfig |
| 5 | gliner_ko | GLiNER v1 완성 모델 | ~110M | 768 | - |
| 6 | **deberta-v3-base-korean** | **DeBERTa V2** | **131M** | **768** | **DebertaV2Config** |

### 2.2 탈락 사유

| 후보 | 판정 | 탈락 사유 |
|------|------|----------|
| **KURE-v1** | ❌ 탈락 | Sentence Transformer (문장 → 벡터 1개). NER은 **토큰별 벡터**가 필요. 구조적 불가 |
| **KoELECTRA** | ❌ 탈락 | ELECTRA 아키텍처 ≠ DeBERTa. pretrained head weights 재활용 불가. 696건으로 head를 from-scratch 학습 불가능 |
| **KLUE-RoBERTa** | ❌ 탈락 | RoBERTa 아키텍처 ≠ DeBERTa. 동일 사유 |
| **XLM-RoBERTa** | ❌ 탈락 | RoBERTa 아키텍처 (270M, 무거움). GLiNER v2.1에서 한국어 F1~65% 한계. 아키텍처 불일치로 head 재활용 불가 |
| **gliner_ko** | ❌ 탈락 | GLiNER v1 완성 모델. GLiNER2의 스키마 인터페이스(엔티티 설명, 다중 태스크) 포기해야 함 |
| **deberta-v3-base-korean** | ✅ **선택** | 동일 아키텍처 (DebertaV2Config, hidden=768, 12 layers). head weights 재활용 가능 |

### 2.3 핵심 논점: 아키텍처 일치의 중요성

GLiNER2의 pretrained 모델 구조:
```
[인코더: DeBERTa-v3-base] → [프로젝션: 768→512] → [span_rep] → [classifier]
```

인코더를 교체하면 프로젝션/span_rep/classifier의 입력 분포가 바뀐다. 이때:
- **동일 아키텍처** (DeBERTa → DeBERTa): 출력 분포가 유사 → head weights를 초기화로 활용 가능 → 소량 파인튜닝으로 수렴
- **다른 아키텍처** (DeBERTa → RoBERTa/ELECTRA): 출력 분포가 근본적으로 다름 → head weights 무의미 → 전체 from-scratch 필요 → 696건으로 불가능

**결론**: `team-lucid/deberta-v3-base-korean`이 유일한 실현 가능한 선택

---

## 3. 구현 방식: Approach A (Surgical Swap)

### 3.1 방식 비교

| 방식 | 설명 | 장단점 |
|------|------|--------|
| **A: Surgical swap** ✅ | pretrained GLiNER2 로드 → 인코더만 교체 → 전체 파인튜닝 | head weights 활용 가능. 696건 소량 데이터에 유리 |
| B: From scratch | 한국어 인코더로 새 모델 생성 → 전체 학습 | 모든 weight 랜덤 시작. 696건으로 부족 |

### 3.2 Surgical Swap 절차

```python
# 1. pretrained GLiNER2 로드 (인코더 + head 모두 학습된 상태)
model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

# 2. 한국어 DeBERTa 인코더 로드
korean_encoder = AutoModel.from_pretrained("team-lucid/deberta-v3-base-korean")
korean_tokenizer = AutoTokenizer.from_pretrained("team-lucid/deberta-v3-base-korean")

# 3. 인코더만 교체 (head weights는 유지)
model.model.encoder = korean_encoder

# 4. 토크나이저 교체
model.tokenizer = korean_tokenizer

# 5. 임베딩 리사이즈 (vocab 크기 차이 보정)
model.model.encoder.resize_token_embeddings(len(korean_tokenizer))

# 6. 차등 학습률로 전체 파인튜닝
```

### 3.3 왜 Full Fine-tune인가 (LoRA 아님)

| 근거 | 설명 |
|------|------|
| 이전 LoRA 실패 | 2회 시도 모두 F1=0.0000 |
| 인코더 교체 상황 | 새 인코더의 출력 분포에 head를 적응시켜야 함 |
| LoRA의 한계 | 저랭크 제약이 인코더-head 간 분포 차이 보정에 부적합 |
| 차등 학습률 | 인코더는 작게(2e-5), head는 크게(5e-4) → 인코더 한국어 지식 보존 + head 적응 |

---

## 4. 하이퍼파라미터

### 4.1 학습 설정

| 파라미터 | 값 | 근거 |
|----------|-----|------|
| **인코더 LR** | 2e-5 | 한국어 DeBERTa의 사전 학습 지식 보존 |
| **헤드 LR** | 5e-4 | 새 인코더 출력에 빠르게 적응 |
| Epochs | 15 | 기존 실험과 동일, early stopping 병행 |
| Batch size | 8 | T4 16GB 기준 |
| Warmup ratio | 0.1 | 첫 10% 스텝 동안 LR 워밍업 |
| Weight decay | 0.01 | 과적합 방지 |
| Max sequence length | 512 | 의료 텍스트 길이 고려 |
| FP16 | True | GPU 메모리 효율화 |
| Gradient accumulation | 2 | 실효 배치 = 16 |
| Seed | 42 | 재현성 |

### 4.2 평가 설정

| 파라미터 | 값 |
|----------|-----|
| 평가 주기 | 매 에폭 |
| 임계값 스윕 | 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 |
| 메트릭 | Entity-level F1 (exact match) |
| 라벨별 리포트 | Disease, Drug, Procedure, Biomarker |
| Best model 기준 | 최고 F1 달성 에폭 저장 |

---

## 5. 평가 계획

### 5.1 사전 검증 (GPU 불필요)

`verify_encoder_swap.py`로 로컬에서 확인:
1. `team-lucid/deberta-v3-base-korean`이 DebertaV2Config으로 로드되는지
2. hidden_size=768 일치 여부
3. 토크나이저가 한국어를 형태소 단위로 분절하는지 (음절 아님)
4. Forward pass 출력 shape 검증

### 5.2 학습 중 평가

매 에폭마다:
- 임계값 스윕 (0.1~0.6) → 최적 임계값의 F1 기록
- 라벨별 P/R/F1 출력
- 오버피팅 감지 (train loss ↓ + test F1 ↓ 시 조기 중단)

### 5.3 최종 평가

| 메트릭 | 목표 |
|--------|------|
| Overall F1 | ≥ 0.50 |
| Disease F1 | ≥ 0.55 |
| Drug F1 | ≥ 0.45 |
| Procedure F1 | ≥ 0.40 |
| Biomarker F1 | ≥ 0.35 |

---

## 6. 리스크

| 리스크 | 확률 | 영향 | 대응 |
|--------|------|------|------|
| 한국어 DeBERTa의 출력 분포가 영어 DeBERTa와 너무 다름 | 중 | 높음 | 초기 에폭에서 F1 추이 확인. 수렴 안 되면 LR 조정 |
| 696건 과적합 | 높음 | 중 | Early stopping, weight decay, dropout 강화 |
| 토크나이저 vocab 차이로 임베딩 리사이즈 실패 | 낮음 | 높음 | verify_encoder_swap.py로 사전 검증 |
| GLiNER2 내부 구조가 단순 인코더 교체를 허용하지 않음 | 중 | 높음 | 모델 구조 사전 분석 후 교체 포인트 확인 |
| GPU OOM | 낮음 | 중 | Gradient accumulation 증가, batch size 감소 |

---

## 7. 실행 절차

### Phase 1: 사전 검증 (로컬)
1. `verify_encoder_swap.py` 실행
2. 아키텍처 호환성, 토크나이저 품질 확인
3. 통과하면 Phase 2 진행

### Phase 2: 학습 (RunPod)
1. RunPod에 환경 세팅 (T4/A10 GPU)
2. 데이터 업로드: `gliner2_train_v2/train.jsonl`, `test.jsonl`
3. `train_gliner2_korean_encoder.py` 실행
4. F1 추이 모니터링

### Phase 3: 분석
1. 최적 체크포인트 선택
2. 라벨별 성능 분석
3. 에러 케이스 검토
4. 결과 리포트 작성

---

## 8. 산출물

| 파일 | 설명 |
|------|------|
| `scripts/gliner2/verify_encoder_swap.py` | 사전 검증 스크립트 |
| `scripts/gliner2/train_gliner2_korean_encoder.py` | 학습 스크립트 |
| `docs/plans/gliner2_korean_encoder_swap_plan.md` | 본 계획 문서 |
| `docs/plans/encoder_selection_discussion.md` | 인코더 선택 논의 (업데이트) |

---

## 9. 관련 문서

| 문서 | 참조 내용 |
|------|----------|
| `docs/reports/gliner2_lora_test_20260106.md` | LoRA 실패 분석 |
| `docs/plans/gliner2_full_finetuning_estimate.md` | 비용/리소스 견적 |
| `docs/plans/gliner2_lora_finetuning_plan.md` | 이전 LoRA 계획 (참고용) |
| `data/gliner2_train_v2/meta.json` | 학습 데이터 메타정보 |
