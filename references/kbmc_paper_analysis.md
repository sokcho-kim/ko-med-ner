# KBMC 논문 분석

> 원본: `Korean Bio-Medical Corpus (KBMC) for Medical Named Entity Recognition.pdf`
> 출처: arXiv:2403.16158v1 [cs.CL] 24 Mar 2024
> 분석일: 2025-12-25

---

## 기본 정보

| 항목 | 내용 |
|------|------|
| 제목 | Korean Bio-Medical Corpus (KBMC) for Medical Named Entity Recognition |
| 저자 | Sungjoo Byun, Jiseung Hong, Sumin Park, Dongjun Jang, Jean Seo, Minseok Kim, Chaeyoung Oh, Hyopil Shin |
| 소속 | Seoul National University, KAIST |
| arXiv | 2403.16158v1 |
| 발표일 | 2024-03-24 |
| 키워드 | Medical NER, Korean NER dataset, Domain-specific, Data construction with LLM |

---

## Abstract 요약

NER(Named Entity Recognition)은 의료 NLP에서 핵심적인 역할을 한다. 그러나 **한국어 의료 NER 데이터셋은 오픈소스로 공개된 것이 없었다**. 이 문제를 해결하기 위해 ChatGPT를 활용하여 **KBMC (Korean Bio-Medical Corpus)**를 구축했다. KBMC 데이터셋을 사용하면 일반 한국어 NER 데이터셋으로 학습한 모델 대비 **의료 NER 성능이 약 20% 향상**된다.

---

## 1. Introduction - 문제 제기

### Medical NER의 역할

1. **의료 용어 처리**: 의료 전문용어와 은어를 식별하고 처리
2. **비정형 데이터 정보 추출**: 비정형 의료 데이터셋에서 정보 추출/인코딩
3. **민감 정보 익명화**: 환자 특정 정보 식별 및 익명화

### 데이터 부족 문제

| 문제점 | 설명 |
|--------|------|
| 레이블링 비용 | Disease, Body, Treatment 등 특정 엔티티 카테고리에 대한 광범위한 레이블링 필요 |
| 전문 지식 필요 | 의료 도메인은 전문가 수준의 지식 필요 |
| 저자원 언어 | 한국어 같은 저자원 언어에서 더욱 심각 |
| **오픈소스 부재** | **한국어 의료 NER 데이터셋이 오픈소스로 공개된 것이 없음** |

### 연구 기여

1. **KBMC 공개**: 최초의 오픈소스 한국어 의료 NER 데이터셋
2. **의료 데이터 처리 기여**: 민감 데이터 익명화, 비정형 의료 데이터 재구성에 활용 가능

---

## 2. Related Work

### Medical NER 연구 동향

| 접근법 | 모델/기술 | 참고 |
|--------|-----------|------|
| 전통적 방법 | LSTM | Liu et al., 2017; Lyu et al., 2017 |
| BERT 기반 | BioBERT | Lee et al., 2019 |
| 벤치마크 | BLUE (Biomedical Language Understanding Evaluation) | Peng et al., 2019 |
| 툴킷 | SpaCy, Apache Spark, Flair | Eyre et al., 2021 |

### Medical NER 데이터셋

| 데이터셋 | 언어 | 특징 |
|----------|------|------|
| i2b2 | 영어 | Medical concept extraction challenge |
| n2c2 | 영어 | Adverse drug events, medication extraction |
| SemClinBER | 포르투갈어 | 포르투갈어 의료 NER |
| NCBI-disease | 영어 | PubMed abstracts 질병명 |
| BC5CDR | 영어 | Chemical-Disease Relation |
| **KBMC** | **한국어** | **최초 오픈소스 한국어 의료 NER** |

---

## 3. KBMC 데이터셋 구축

### 3.1 데이터 구축 과정

```
┌─────────────────────────────────────────────────────────────┐
│                    KBMC 구축 파이프라인                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Text Source                                             │
│     ├── KOSTOM (Korean Standard Terminology of Medicine)    │
│     │   ├── KCD 8차 개정 (한국표준질병분류)                   │
│     │   └── 의료 현장 로컬 용어                              │
│     └── ChatGPT API (gpt-3.5-turbo)                         │
│         └── 프롬프트: "주어진 의료 용어를 포함하는            │
│             20단어 이상의 한국어 문장 생성"                   │
│                                                             │
│  2. Automatic Pre-annotation                                │
│     ├── OKT (Open-source Korean Text) 토크나이저            │
│     ├── 어휘 목록 기반 스팬 매칭                             │
│     └── B-I-O 태그 자동 부여                                 │
│                                                             │
│  3. Human Annotation                                        │
│     ├── 4명의 어노테이터                                     │
│     ├── 1명의 리뷰어 (품질 검증)                             │
│     └── KOSTOM 표준 준수                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Pre-annotation 알고리즘

```
입력: 문장 집합 W = {w1, w2, ..., wN}
출력: BIO 태깅된 토큰 시퀀스

1. OKT로 토크나이징: Ŵ = {x1, x2, ..., xM}
2. 엔티티 유형별 어휘 목록: E = {Disease, Body, Treatment}
3. 각 엔티티 타입 e ∈ E에 대해:
   - 어휘 목록과 매칭되는 스팬 집합 Se 탐지
   - 스팬의 첫 토큰: B-e 태그
   - 스팬의 나머지 토큰: I-e 태그
```

### 3.2 데이터셋 통계

| 항목 | 수치 |
|------|------|
| 총 문장 수 | **6,150** |
| 총 토큰 수 | **153,971** |
| 고유 질병명 | 4,162 |
| 고유 신체 부위 | 841 |
| 고유 치료법 | 396 |

### 레이블 분포

| Named Entity | Scheme | 개수 |
|--------------|--------|------|
| **Disease** | B (Begin) | 10,595 |
| | I (Inside) | 10,089 |
| **Body** | B (Begin) | 5,215 |
| | I (Inside) | 1,158 |
| **Treatment** | B (Begin) | 1,193 |
| | I (Inside) | 839 |

### 3.3 토크나이징 전략

| 기존 방식 | KBMC 방식 |
|-----------|-----------|
| 어절(word) 단위 | **형태소(morpheme) 단위** |
| 명사+조사 결합 | 명사와 조사 분리 |
| 부정확한 태깅 | **정확한 태깅** |

**이유**: 한국어는 교착어(agglutinative language)이므로 형태소 단위 토크나이징이 더 정확함

### 어노테이션 예시

| 한국어 문장 | 번역 | NER 태그 |
|-------------|------|----------|
| 전신 적 다한증 은 신체 전체 에... | Systemic myasthenia is a condition... | Disease-B Disease-I Disease-I O O O... |
| 췌장암 이란 췌장 에 생긴 암세포... | Pancreatic cancer refers to... | Disease-B O Body-B O O O O Disease-B... |
| 버킷 림프종 은 림프절 에서... | Burkitt lymphoma is a malignant... | Disease-B Disease-I O Body-B O O O... Treatment-B Treatment-I... |

---

## 4. 실험

### 4.1 실험 모델

| 모델 | 유형 | 특징 |
|------|------|------|
| **KM-BERT** | 도메인 특화 | 한국어 의료 코퍼스로 사전학습 |
| KR-BERT | 일반 | 한국어 특화 소형 모델 |
| KoBERT | 일반 | SKT Brain |
| KR-ELECTRA | 일반 | 한국어 ELECTRA |
| KoELECTRA v3 | 일반 | monologg |
| BiLSTM-CRF | 전통적 | Bidirectional LSTM + CRF |

### 4.2 실험 결과

#### 일반 데이터셋만 사용 (Naver NER Dataset)

| Model | Avg.F1 (General) | Medical NE | F1 (Medical) |
|-------|------------------|------------|--------------|
| KM-BERT | 87.08 | TRM | 75.35 |
| KR-BERT | 86.51 | TRM | 75.26 |
| KoBERT | 88.01 | TRM | 78.21 |
| KR-ELECTRA | 87.62 | TRM | 76.25 |
| KoELECTRA | 88.00 | TRM | 76.58 |
| BiLSTM-CRF | 55.23 | TRM | 42.23 |

**문제점**: TRM 레이블이 의료 용어와 IT 용어를 모두 포함 → 의료 용어 정확한 식별 어려움

#### KBMC 적용 후 성능

| Model | Avg.F1 | Disease F1 | Body F1 | Treatment F1 |
|-------|--------|------------|---------|--------------|
| KM-BERT | 88.53 (+1.45) | **98.04** (+22.69) | **98.13** (+22.78) | **98.53** (+23.18) |
| KR-BERT | 87.48 (+0.97) | 98.04 (+22.78) | 98.32 (+23.06) | 97.82 (+22.56) |
| KoBERT | 88.70 (+0.69) | 98.25 (+20.04) | 98.22 (+20.01) | 98.18 (+19.97) |
| KR-ELECTRA | 88.63 (+1.01) | 98.21 (+21.96) | 98.31 (+22.06) | 98.53 (+22.28) |
| KoELECTRA | 88.86 (+0.86) | 98.05 (+21.47) | 97.72 (+21.14) | 96.56 (+19.98) |
| BiLSTM-CRF | 56.68 (+1.45) | 88.18 (+45.95) | 81.44 (+39.21) | 61.14 (+18.91) |

### 핵심 성능 향상

```
┌─────────────────────────────────────────────────────────────┐
│                    성능 향상 요약                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  TRM (일반 데이터셋)     →    Disease/Body/Treatment (KBMC) │
│                                                             │
│       ~75-78%          →         ~98%                       │
│                                                             │
│              약 20% 포인트 성능 향상                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 MedSpaCy 적용 테스트

| 지표 | 수치 |
|------|------|
| Avg.F1 | **95.69** |
| Precision | 97.02 |
| Recall | 95.52 |

→ MedSpaCy + ko_core_news_md 조합에서도 우수한 성능

---

## 5. 데이터셋 통합 전략

### Naver NER 데이터셋과 KBMC 통합

```
┌─────────────────────────────────────────────────────────────┐
│              데이터셋 통합 전략                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Original Naver NER (90,000 문장)                           │
│  ├── 14개 Named Entities                                    │
│  │   (PER, FLD, NUM, DAT, ORG, TRM, LOC, EVT, ...)         │
│  └── TRM: 의료 + IT 용어 혼합 (문제)                        │
│                                                             │
│  ↓ TRM 포함 문장 12,426개 제외                              │
│                                                             │
│  Combined Dataset                                           │
│  ├── Naver NER (13개 일반 NE)                               │
│  └── KBMC (3개 의료 NE: Disease, Body, Treatment)           │
│                                                             │
│  = 총 16개 Named Entities                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Limitations

| 한계점 | 설명 |
|--------|------|
| 데이터 제한 | 한국어 의료 데이터 가용성 부족 |
| 태스크 제한 | NER만 지원, QA 등 다른 downstream 태스크 미지원 |
| 비교 제한 | 비교 가능한 한국어 NER 데이터셋이 Naver 데이터셋뿐 |
| 모델 제한 | 도메인 특화 모델이 KM-BERT뿐 |

---

## 7. 의료 도메인 활용 시사점

### KBMC의 강점

1. **최초 오픈소스**: 한국어 의료 NER 데이터셋 최초 공개
2. **ChatGPT 활용**: LLM을 활용한 효율적 데이터 구축 방법론
3. **형태소 단위 토크나이징**: 한국어 특성을 고려한 정확한 어노테이션
4. **표준 준수**: KOSTOM (한국표준의학용어) 기반

### 우리 프로젝트 적용 가능성

| 항목 | KBMC | 우리 프로젝트 (cg_parsed) |
|------|------|---------------------------|
| 엔티티 | Disease, Body, Treatment | 약제명, 질병명, 고시번호, 행위코드 |
| 데이터 출처 | KOSTOM + ChatGPT 생성 | HIRA 고시/사례/행정해석 |
| 활용 | 의료 NER | Graph RAG 질의 분석 |

### GLiNER2와 KBMC 비교

| 항목 | KBMC | GLiNER2 |
|------|------|---------|
| 접근법 | Fine-tuning 데이터셋 | Zero-shot 모델 |
| 엔티티 | 3개 (Disease, Body, Treatment) | 무제한 (스키마 정의) |
| 한국어 | **네이티브 지원** | 제한적 지원 |
| 관계 추출 | X | O |
| 구조화 추출 | X | O |

### 하이브리드 접근 제안

```python
# 1단계: GLiNER2로 제로샷 추출
# 2단계: KBMC 데이터로 한국어 의료 엔티티 파인튜닝
# 3단계: cg_parsed 데이터로 추가 파인튜닝 (고시번호, 행위코드 등)

pipeline = {
    "base_model": "GLiNER2",
    "fine_tuning_data": [
        "KBMC (Disease, Body, Treatment)",
        "cg_parsed (고시번호, 행위코드, 약제명)"
    ],
    "output": "Korean Medical NER for Graph RAG"
}
```

---

## 8. 데이터셋 접근

### 관련 링크

| 리소스 | URL |
|--------|-----|
| Naver NER Dataset | https://github.com/naver/nlp-challenge/tree/master/missions/ner |
| KOSTOM | https://www.hins.or.kr/index.es?sid=a1 |
| KCD (한국표준질병분류) | https://www.koicd.kr/kcd/kcds.do |
| OKT 토크나이저 | https://github.com/open-korean-text/open-korean-text |
| KoBERT | https://github.com/SKTBrain/KoBERT |
| KR-ELECTRA | https://github.com/snunlp/KR-ELECTRA |
| KoELECTRA | https://github.com/monologg/KoELECTRA |
| MedSpaCy | https://github.com/medspacy/medspacy |

---

## 인용

```bibtex
@article{byun2024kbmc,
    title={Korean Bio-Medical Corpus (KBMC) for Medical Named Entity Recognition},
    author={Byun, Sungjoo and Hong, Jiseung and Park, Sumin and Jang, Dongjun and Seo, Jean and Kim, Minseok and Oh, Chaeyoung and Shin, Hyopil},
    journal={arXiv preprint arXiv:2403.16158},
    year={2024}
}
```

---

## 핵심 인사이트 요약

| 핵심 포인트 | 내용 |
|-------------|------|
| **문제** | 한국어 의료 NER 오픈소스 데이터셋 부재 |
| **해결책** | KBMC - ChatGPT + KOSTOM 기반 구축 |
| **성능** | 의료 NER F1 **약 20% 향상** (75% → 98%) |
| **규모** | 6,150 문장, 153,971 토큰 |
| **엔티티** | Disease (질병), Body (신체), Treatment (치료) |
| **토크나이징** | 형태소 단위 (OKT) - 한국어 특성 반영 |
| **활용** | Fine-tuning 데이터셋으로 활용 |

---

## 다음 단계 (우리 프로젝트)

1. [ ] KBMC 데이터셋 다운로드 가능 여부 확인
2. [ ] GLiNER2 + KBMC 하이브리드 파인튜닝 계획
3. [ ] cg_parsed 데이터로 추가 파인튜닝 데이터 준비
4. [ ] 의료 도메인 특화 엔티티 정의 (고시번호, 행위코드 등)
