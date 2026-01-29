# KBMC Baseline 비교 실험 리포트

> **실험일**: 2026-01-05
> **목적**: 한국어 의료 NER에서 KoELECTRA fine-tuning vs GLiNER zero-shot 비교

---

## 실험 개요

| 항목 | 내용 |
|------|------|
| 데이터셋 | SungJoo/KBMC (6,150 문장) |
| 라벨 | Disease, Body, Treatment (BIO 태깅) |
| Train/Test | 5,535 / 615 (90/10 split, seed=42) |

---

## 모델 비교

### 1. KoELECTRA Fine-tuning

| 항목 | 내용 |
|------|------|
| 모델 | monologg/koelectra-base-v3-discriminator |
| 방식 | Supervised fine-tuning |
| Epochs | 3 |
| Batch Size | 16 |
| 학습 환경 | RunPod RTX 3090 |
| 학습 시간 | 72초 |

**결과:**

| 지표 | 값 |
|------|-----|
| **F1** | **0.9706 (97.06%)** |
| Precision | 0.9711 |
| Recall | 0.9704 |
| Accuracy | 0.9704 |

### 2. GLiNER Zero-shot

| 항목 | 내용 |
|------|------|
| 모델 | urchade/gliner_multi-v2.1 |
| 방식 | Zero-shot (라벨만 제공) |
| Threshold | 0.3 |
| 테스트 환경 | RunPod RTX 3090 |

**결과:**

| 지표 | 값 |
|------|-----|
| **F1** | **0.3401 (34.01%)** |
| Precision | 0.3267 |
| Recall | 0.3547 |

**라벨별 F1:**

| 라벨 | F1 | TP | FP | FN |
|------|-----|-----|-----|-----|
| Disease | 27.81% | 241 | 485 | 766 |
| Body | 55.07% | 288 | 307 | 163 |
| Treatment | 12.06% | 31 | 362 | 90 |

---

## 비교 요약

| 모델 | F1 | 장점 | 단점 |
|------|-----|------|------|
| **KoELECTRA** | **97.06%** | 높은 정확도 | 학습 데이터 필요 |
| GLiNER | 34.01% | 학습 불필요 | 한국어 성능 낮음 |

**차이: 63.05%p** (KoELECTRA 압도적 우위)

---

## 분석

### KoELECTRA 성공 요인
1. 한국어 특화 사전학습 (KoELECTRA)
2. BIO 태깅으로 정확한 경계 학습
3. KBMC 데이터의 공백 토큰화 형식

### GLiNER 실패 요인
1. 한국어 토크나이징 문제 (교착어 특성)
2. 의료 도메인 지식 부족
3. 엔티티 경계 인식 실패 (문장 전체를 하나의 엔티티로)

---

## 결론

| 결론 | 내용 |
|------|------|
| **한국어 의료 NER** | Fine-tuning 필수 |
| **GLiNER zero-shot** | 한국어 의료 도메인에서 실용 수준 미달 |
| **권장 접근** | KoELECTRA + 도메인 데이터 fine-tuning |

---

## 다음 단계

1. [ ] GLiNER 파인튜닝으로 성능 개선 가능한지 실험
2. [ ] 다른 한국어 NER 모델 비교 (KoBERT, KoBART 등)
3. [ ] 실제 Graph RAG 파이프라인에 KoELECTRA 통합

---

## 관련 파일

| 파일 | 경로 |
|------|------|
| 학습 스크립트 | `scripts/train/train_koelectra_kbmc.py` |
| 클라우드 노트북 | `notebooks/train_koelectra_kbmc_cloud.ipynb` |
| RunPod 가이드 | `docs/runpod_guide.md` |
| GLiNER 테스트 | `docs/reports/gliner_zeroshot_test_20260105.md` |
| KoELECTRA 결과 | `models/kbmc/koelectra-baseline-results.txt` |

---

## 비용

| 항목 | 비용 |
|------|------|
| RunPod RTX 3090 | ~$0.02 (2분 사용) |
| 총 비용 | **~25원** |
