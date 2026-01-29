# 한국어 의료 NER 벤치마크 구축 계획

> **작성일**: 2026-01-05
> **목적**: 논문급 평가 체계 구축 + 실제 파이프라인 적용

---

## 현재 상태

### 완료된 실험

| 모델 | 방식 | F1 | 테스트셋 |
|------|------|-----|----------|
| KoELECTRA | Fine-tuning | **97.06%** | KBMC 615개 |
| GLiNER | Zero-shot | 34.01% | KBMC 615개 |

### 한계점

- 단일 데이터셋 (KBMC)만 사용
- Cross-validation 미실시
- 다른 모델과 비교 부족
- 외부 검증 없음

---

## 벤치마크 구축 계획

### 1. 데이터셋 확보

| 데이터셋 | 출처 | 상태 | 우선순위 |
|----------|------|------|----------|
| KBMC | HuggingFace | ✓ 완료 | - |
| NCBI Disease (한국어) | 번역/구축 | 탐색 필요 | 중 |
| 약학정보원 | 크롤링 | 확인 필요 | 중 |
| 건보심평원 공공데이터 | data.go.kr | 확인 필요 | 높음 |
| 자체 구축 (EMR 형식) | 수동 | 고려 중 | 낮음 |

### 2. 비교 모델

| 모델 | 방식 | 상태 | 우선순위 |
|------|------|------|----------|
| KoELECTRA | Fine-tuning | ✓ 완료 | - |
| GLiNER | Zero-shot | ✓ 완료 | - |
| KoBERT | Fine-tuning | 예정 | 높음 |
| KoBART | Fine-tuning | 예정 | 중 |
| GPT-4 | Few-shot | 예정 | 높음 |
| Claude | Few-shot | 예정 | 높음 |
| GLiNER | Fine-tuning | 예정 | 중 |

### 3. 평가 메트릭

| 메트릭 | 설명 | 구현 |
|--------|------|------|
| **Strict Entity F1** | 정확히 일치하는 엔티티 | 완료 |
| **Partial Entity F1** | 부분 일치 허용 | 예정 |
| **Token F1** | 토큰 단위 평가 | 예정 |
| **라벨별 F1** | Disease/Body/Treatment 각각 | 완료 |
| **Macro F1** | 라벨별 평균 | 예정 |
| **Micro F1** | 전체 평균 | 완료 |

### 4. 평가 프레임워크 구조

```
project/ner/
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py          # 평가 메트릭 함수들
│   ├── data_loader.py      # 데이터셋 로더
│   ├── model_wrapper.py    # 모델별 래퍼
│   └── benchmark.py        # 벤치마크 실행기
├── datasets/
│   ├── kbmc/               # KBMC 데이터
│   └── [other]/            # 추가 데이터셋
├── models/
│   ├── koelectra/
│   ├── kobert/
│   ├── gliner/
│   └── llm_fewshot/
└── results/
    └── benchmark_YYYYMMDD.json
```

---

## 실행 계획

### Phase 1: 기반 구축 (현재)
- [x] KoELECTRA baseline
- [x] GLiNER zero-shot 비교
- [ ] 평가 프레임워크 코드화
- [ ] 결과 자동 리포트 생성

### Phase 2: 모델 확장
- [ ] KoBERT fine-tuning
- [ ] GPT-4 few-shot 테스트
- [ ] Claude few-shot 테스트
- [ ] GLiNER fine-tuning

### Phase 3: 데이터 확장
- [ ] 추가 데이터셋 탐색/확보
- [ ] Cross-dataset 평가
- [ ] 일반화 성능 검증

### Phase 4: 논문화
- [ ] 실험 결과 정리
- [ ] 분석 및 인사이트 도출
- [ ] 논문 초안 작성

---

## 예상 결과 테이블 (논문용)

| Model | Method | KBMC F1 | Dataset2 F1 | Avg |
|-------|--------|---------|-------------|-----|
| KoELECTRA | Fine-tune | 97.06 | - | - |
| KoBERT | Fine-tune | - | - | - |
| GLiNER | Zero-shot | 34.01 | - | - |
| GLiNER | Fine-tune | - | - | - |
| GPT-4 | Few-shot | - | - | - |
| Claude | Few-shot | - | - | - |

---

## 참고 자료

- KBMC 논문: Korean Bio-Medical Corpus
- GLiNER 논문: Generalist Model for NER
- KoELECTRA: 한국어 ELECTRA

---

## 관련 파일

| 파일 | 경로 |
|------|------|
| KoELECTRA 학습 | `scripts/train/train_koelectra_kbmc.py` |
| GLiNER 평가 | `scripts/eval/eval_gliner_kbmc.py` |
| 비교 리포트 | `docs/reports/kbmc_baseline_comparison_20260105.md` |
| RunPod 가이드 | `docs/runpod_guide.md` |
