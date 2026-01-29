# 한국어 의료 NER 연구 체크리스트

> 집에서 조사할 것들

---

## 0. 이전 모델 비교 (최우선)

### Google Drive 모델 확인
- [ ] Google Drive에서 `kbmc-ner-final` 폴더 찾기
- [ ] 모델 파일 있는지 확인 (`config.json`, `pytorch_model.bin` 등)
- [ ] 이전 학습 결과 (F1) 확인 (노트북 출력 or 저장된 파일)

### 동일 조건 평가
```python
# Colab에서 실행
from google.colab import drive
drive.mount('/content/drive')

from transformers import AutoModelForTokenClassification, AutoTokenizer
from datasets import load_dataset

# 이전 모델 로드
model_path = "/content/drive/MyDrive/경로/kbmc-ner-final"  # 경로 수정
model = AutoModelForTokenClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 동일 test set (seed=42)
dataset = load_dataset("SungJoo/KBMC")
dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)
test_set = dataset['test']  # 615개
```

### 비교 항목
- [ ] 이전 모델 F1 vs 오늘 모델 F1 (97.06%)
- [ ] 하이퍼파라미터 차이 확인 (epochs, batch size, lr)
- [ ] train/test split 동일한지 (seed=42?)

---

## 1. 데이터셋 탐색

### 한국어 의료 NER 데이터셋
- [ ] **KLUE NER** - 일반 도메인이지만 구조 참고
  - https://klue-benchmark.com/
- [ ] **모두의 말뭉치 (국립국어원)**
  - 의료 관련 있는지 확인
  - https://corpus.korean.go.kr/
- [ ] **AIHub 의료 데이터** ⭐ 우선 확인
  - https://aihub.or.kr/
  - **필수의료지식** - NER 태깅 여부 확인
  - **전문의료지식** - NER 태깅 여부 확인
  - 데이터 형식, 라벨 종류, 크기, 라이선스 확인
- [ ] **건강보험심사평가원**
  - https://opendata.hira.or.kr/
  - 질병명, 약품명 데이터
- [ ] **약학정보원**
  - https://www.health.kr/
  - 약물 정보 크롤링 가능?
- [ ] **식약처 의약품 데이터**
  - https://nedrug.mfds.go.kr/
  - 의약품 명칭 사전

### 영어 의료 NER (번역 고려)
- [ ] **NCBI Disease Corpus**
- [ ] **BC5CDR** (Chemical-Disease Relations)
- [ ] **i2b2 NER datasets**
- [ ] **MedMentions**

---

## 2. 비교 모델 조사

### Fine-tuning 기반
- [ ] **KoBERT** - SKT 한국어 BERT
  - https://github.com/SKTBrain/KoBERT
- [ ] **KoBART** - SKT 한국어 BART
- [ ] **KcELECTRA** - 한국어 댓글 ELECTRA
- [ ] **KLUE-RoBERTa** - KLUE 벤치마크 모델

### Zero-shot / Few-shot
- [ ] **GPT-4** few-shot 프롬프팅
  - 비용 계산 필요
- [ ] **Claude** few-shot 프롬프팅
- [ ] **GLiNER fine-tuning**
  - KBMC로 파인튜닝하면 성능 개선?
- [ ] **UniNER** - 범용 NER 모델
  - https://github.com/universal-ner/universal-ner

---

## 3. 평가 방법론

### 메트릭
- [ ] **seqeval** 라이브러리
  - NER 표준 평가 도구
  - pip install seqeval
- [ ] **Strict vs Lenient matching**
  - 정확 일치 vs 부분 일치
- [ ] **CoNLL evaluation script**
  - 표준 NER 평가 스크립트

### 실험 설계
- [ ] **K-fold cross-validation**
  - 5-fold가 표준
- [ ] **Statistical significance test**
  - McNemar's test, paired t-test
- [ ] **Error analysis 방법론**
  - 어떤 유형의 오류가 많은지

---

## 4. 관련 논문

### 한국어 NER
- [ ] KLUE 논문 (Korean Language Understanding Evaluation)
- [ ] KoELECTRA 논문
- [ ] 한국어 의료 NER 관련 논문 (Google Scholar 검색)

### 의료 NER 일반
- [ ] BioBERT 논문
- [ ] PubMedBERT 논문
- [ ] GLiNER / GLiNER2 논문 (이미 있음)

### 벤치마크 관련
- [ ] BLURB (Biomedical Language Understanding and Reasoning Benchmark)
- [ ] BigBIO (다국어 의료 NLP 벤치마크)

---

## 5. 기술적 조사

### 프레임워크
- [ ] **Hugging Face datasets** - 데이터 로딩
- [ ] **seqeval** - NER 평가
- [ ] **Weights & Biases** - 실험 추적
- [ ] **Optuna** - 하이퍼파라미터 튜닝

### 인프라
- [ ] **RunPod** vs **Lambda Labs** vs **Vast.ai** 비교
- [ ] **Colab Pro** 가격/성능

---

## 6. 확인할 질문들

### 데이터 관련
- [ ] KBMC 외에 공개된 한국어 의료 NER 데이터 있나?
- [ ] AIHub 의료 데이터 NER 용도로 쓸 수 있나?
- [ ] 직접 라벨링해야 하나? (비용/시간)

### 모델 관련
- [ ] GLiNER 한국어로 fine-tuning한 사례 있나?
- [ ] 한국어 Bio-BERT 같은 거 있나?
- [ ] LLM few-shot이 fine-tuning을 이길 수 있나?

### 평가 관련
- [ ] 의료 NER에서 표준 벤치마크가 뭔지?
- [ ] 논문에서 보통 어떤 메트릭 보고하는지?
- [ ] 테스트셋 크기 615개가 충분한지?

---

## 7. 우선순위 정리

### 이번 주 (높음)
1. [ ] AIHub 의료 데이터 확인
2. [ ] KLUE NER 구조 파악
3. [ ] GPT-4 few-shot 테스트 비용 계산

### 다음 주 (중간)
4. [ ] KoBERT 학습 코드 준비
5. [ ] seqeval 기반 평가 프레임워크
6. [ ] 관련 논문 2-3개 읽기

### 나중에 (낮음)
7. [ ] 추가 데이터셋 구축/번역
8. [ ] 하이퍼파라미터 최적화
9. [ ] 논문 초안

---

## 링크 모음

| 이름 | URL |
|------|-----|
| KBMC | https://huggingface.co/datasets/SungJoo/KBMC |
| KLUE | https://klue-benchmark.com/ |
| AIHub | https://aihub.or.kr/ |
| 모두의말뭉치 | https://corpus.korean.go.kr/ |
| 건보심평원 | https://opendata.hira.or.kr/ |
| GLiNER | https://github.com/urchade/GLiNER |
| KoELECTRA | https://github.com/monologg/KoELECTRA |
| seqeval | https://github.com/chakki-works/seqeval |
