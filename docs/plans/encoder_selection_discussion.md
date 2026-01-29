# GLiNER2 인코더 선택 논의

> 작성일: 2026-01-06
> 갱신일: 2026-01-27

---

## 1. 배경

GLiNER2 LoRA 파인튜닝 테스트 결과, 기본 인코더(DeBERTa-v3-base)가 한국어를 처리하지 못해 실패.
→ 인코더 교체 필요

### 1.1 이전 실험 결과

| 실험 | 인코더 | 결과 |
|------|--------|------|
| LoRA 파인튜닝 (KBMC 50건) | DeBERTa-v3-base | F1=0.0000 (실패) |
| LoRA 파인튜닝 (Silver 696건) | DeBERTa-v3-base | F1=0.0000 (실패) |
| GLiNER v2.1 zero-shot | XLM-RoBERTa | F1≈0.65 (참고) |

**근본 원인**: DeBERTa-v3-base 토크나이저가 한국어를 음절 단위로 분절
```
"당뇨병" → ['▁', '당', '뇨', '병']  (의미 정보 손실)
```

### 1.2 현재 데이터 현황

| 항목 | 상태 |
|------|------|
| 라벨 정의 | ✅ Disease, Drug, Procedure, Biomarker |
| 학습 데이터 | ✅ 696건 (gliner2_train_v2/train.jsonl) |
| 테스트 데이터 | ✅ 77건 (gliner2_train_v2/test.jsonl) |
| 데이터 품질 | ✅ GPT-5 검증 완료 |

---

## 2. 후보 인코더 (6개)

### 2.1 전체 후보 목록

| # | 인코더 | 아키텍처 | 파라미터 | hidden | Config 클래스 |
|---|--------|----------|----------|--------|--------------|
| 1 | `upskyy/kure-v1` | Sentence Transformer | - | 1024 (문장) | - |
| 2 | `monologg/koelectra-base-v3` | ELECTRA | 110M | 768 | ElectraConfig |
| 3 | `klue/roberta-base` | RoBERTa | 111M | 768 | RobertaConfig |
| 4 | `xlm-roberta-base` | RoBERTa | 270M | 768 | XLMRobertaConfig |
| 5 | `urchade/gliner_multi` (gliner_ko) | GLiNER v1 | ~110M | 768 | - |
| 6 | `team-lucid/deberta-v3-base-korean` | **DeBERTa V2** | 131M | **768** | **DebertaV2Config** |

### 2.2 체계적 탈락 분석

| 후보 | 판정 | 탈락 사유 |
|------|------|----------|
| **KURE-v1** | ❌ | Sentence Transformer — 문장→벡터 1개. NER은 **토큰별 벡터** 필요. 구조적 불가 |
| **KoELECTRA** | ❌ | ELECTRA ≠ DeBERTa. pretrained head weights 재활용 불가. 696건으로 head를 from-scratch 학습 불가능 |
| **KLUE-RoBERTa** | ❌ | RoBERTa ≠ DeBERTa. 동일 사유 |
| **XLM-RoBERTa** | ❌ | RoBERTa 아키텍처 (270M, 무거움). GLiNER v2.1에서 한국어 F1≈65% 한계. 아키텍처 불일치로 head 재활용 불가 |
| **gliner_ko** | ❌ | GLiNER v1 완성 모델. GLiNER2의 스키마 인터페이스(엔티티 설명, 다중 태스크) 포기해야 함 |
| **deberta-v3-base-korean** | **✅** | 동일 아키텍처 (DebertaV2Config, hidden=768, 12 layers). head weights 재활용 가능 |

---

## 3. 선택 기준 (갱신)

| 기준 | 중요도 | 설명 |
|------|--------|------|
| **아키텍처 일치** | **최상** | GLiNER2 pretrained head weights 재활용 가능 여부 |
| 한국어 토큰화 품질 | 높음 | 의료 용어 분절 정확도 (형태소 vs 음절) |
| GLiNER2 호환성 | 높음 | surgical swap으로 작동하는가 |
| 모델 크기 | 중간 | 추론 속도, 메모리 |
| 사전 학습 도메인 | 중간 | 의료 도메인 친화성 |

**핵심 통찰**: 이전 문서에서는 "GLiNER2 호환성"을 코드 수준 호환으로 봤지만, 실제로는 **아키텍처 레벨 일치**가 핵심. GLiNER2의 pretrained 모델은 인코더(DeBERTa) + 프로젝션(768→512) + span_rep + classifier가 함께 학습됨. 인코더를 교체하면서 나머지 weight를 살리려면 **동일 Config 클래스**여야 한다.

---

## 4. 결정: `team-lucid/deberta-v3-base-korean`

### 4.1 선택 근거

1. **DebertaV2Config**: GLiNER2의 인코더와 동일한 Config 클래스
2. **hidden_size=768**: 프로젝션 레이어 입력 차원 일치
3. **12 layers**: 동일한 레이어 수
4. **한국어 형태소 분절**: 음절이 아닌 형태소 단위 토큰화
5. **131M 파라미터**: 합리적인 크기 (XLM-R의 270M 대비 경량)

### 4.2 구현 방식: Surgical Swap

```
pretrained GLiNER2 로드 → 인코더만 한국어 DeBERTa로 교체 → 전체 파인튜닝
```

- head weights(프로젝션, span_rep, classifier)는 유지
- 인코더만 교체하여 한국어 토큰 임베딩 확보
- 차등 학습률: 인코더 2e-5 / 헤드 5e-4
- Full fine-tune (LoRA 2회 실패 근거)

---

## 5. 이전 미해결 사항 해결 현황

| 이전 미해결 사항 | 현재 상태 |
|----------------|----------|
| 질의(Query) 정의 | ✅ 해결 — Disease, Drug, Procedure, Biomarker |
| 골든셋 구축 | ✅ 해결 — 77건 테스트셋 (GPT-5 검증) |
| 실버셋 구축 | ✅ 해결 — 696건 학습셋 (Gazetteer + GPT-5 검증) |
| 인코더 선택 | ✅ 결정 — deberta-v3-base-korean |

---

## 6. 다음 단계

1. **사전 검증** — `verify_encoder_swap.py`로 호환성 확인
2. **학습** — `train_gliner2_korean_encoder.py`로 RunPod 학습
3. **목표** — F1 ≥ 0.50 (베이스라인 0.387 대비 개선)

---

## 7. 관련 문서

| 문서 | 설명 |
|------|------|
| `gliner2_korean_encoder_swap_plan.md` | 전체 교체 계획 (상세) |
| `gliner2_full_finetuning_estimate.md` | 비용/리소스 견적 |
| `gliner2_lora_finetuning_plan.md` | 이전 LoRA 계획 (실패) |
| `../reports/gliner2_lora_test_20260106.md` | LoRA 실패 분석 |
