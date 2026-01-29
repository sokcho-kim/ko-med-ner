# GLiNER Zero-shot 테스트 결과

> **테스트일**: 2026-01-05
> **모델**: urchade/gliner_multi-v2.1
> **데이터**: KBMC (SungJoo/KBMC)
> **목적**: 한국어 의료 NER에서 GLiNER zero-shot 성능 평가

---

## 테스트 환경

| 항목 | 내용 |
|------|------|
| 모델 | `urchade/gliner_multi-v2.1` |
| 베이스 | microsoft/mdeberta-v3-base |
| 데이터셋 | KBMC (6,150 문장) |
| 라벨 | Disease, Body, Treatment |
| Threshold | 0.3 |

---

## 테스트 방법

```python
from gliner import GLiNER
from datasets import load_dataset

model = GLiNER.from_pretrained('urchade/gliner_multi-v2.1')
dataset = load_dataset('SungJoo/KBMC')

labels = ['Disease', 'Body', 'Treatment']

# 공백 제거하여 원문 복원 후 예측
text = sample['Sentence'].replace(' ', '')
entities = model.predict_entities(text, labels, threshold=0.3)
```

---

## 테스트 결과 (10개 샘플)

### 샘플 1
- **입력**: 전신적다한증은신체전체에힘이빠져서일상생활이어려워지는질환으로,근육통증과무기력감이동반됩니다.
- **정답**: 전신적다한증(Disease), 근육통증(Disease), 무기력감(Disease)
- **예측**: 전신적다한증은신체전체에힘이빠져서일상생활이어려워지는질환으로(Disease, 0.78)
- **결과**: ❌ 엔티티 경계 과확장

### 샘플 2
- **입력**: 췌장암이란췌장에생긴암세포로이루어진종괴...
- **정답**: 췌장암(Disease), 췌장(Body), 종괴(Disease)
- **예측**: 췌장암이란췌장에생긴암세포로이루어진종괴(Disease, 0.76)
- **결과**: ❌ 엔티티 경계 과확장

### 샘플 3
- **입력**: 폐기능저하로인한호흡곤란기침천식발작...
- **정답**: 폐기능저하로인한호흡곤란기침천식발작(Disease)
- **예측**: (없음)
- **결과**: ❌ 미검출

### 샘플 4
- **입력**: 개방성귀인두관은귀속에서발생하는질환으로...
- **정답**: 개방성귀인두관(Disease), 귀(Body)
- **예측**: 개방성귀인두관은귀속에서발생하는질환으로(Body, 0.59)
- **결과**: ❌ 라벨 오류 + 경계 과확장

### 샘플 5
- **입력**: 치아의유착증은치아가이상적인위치에...
- **정답**: 치아의유착증(Disease), 치아(Body)
- **예측**: 치아의유착증은치아가...질환입니다(Disease, 0.79)
- **결과**: ❌ 엔티티 경계 과확장

### 샘플 6
- **입력**: 호르몬장애는신체의호르몬분비에이상이생겨서...
- **정답**: 호르몬장애(Disease), 생리불순(Disease), 생리통(Disease)
- **예측**:
  - 호르몬장애는신체의호르몬분비에이상이생겨서생기는질환으로(Disease, 0.88)
  - 여성의경우에는생리불순(Body, 0.75) ← 라벨 오류
  - 생리통(Disease, 0.88) ✅
- **결과**: ⚠️ 부분 성공 (생리통 검출)

### 샘플 7
- **입력**: 범혈구감소증은혈액내범혈구의수가감소하여...
- **정답**: 범혈구감소증(Disease), 혈액(Body), 범혈구(Body)
- **예측**:
  - 범혈구감소증은혈액내범혈구의수가감소하여면역력이약해지는질환으로(Disease, 0.90)
  - 체내감염에취약해지고출혈이발생할수있으며(Body, 0.56)
- **결과**: ❌ 엔티티 경계 과확장

### 샘플 8
- **입력**: 담낭염을동반한담관결석으로인해복부가심하게아프고...
- **정답**: 담낭염(Disease), 담관결석(Disease), 복부(Body)
- **예측**: 담낭염을동반한담관결석으로인해복부가심하게아프고(Disease, 0.94)
- **결과**: ❌ 엔티티 경계 과확장

### 샘플 9
- **입력**: 쿠퍼샘의농양은피부에생기는염증으로...
- **정답**: 쿠퍼샘의농양(Disease), 피부(Body), 염증(Disease)
- **예측**: 쿠퍼샘의농양은피부에생기는염증으로통증과붓기를동반하여치료가필요한질환입니다(Disease, 0.73)
- **결과**: ❌ 엔티티 경계 과확장

### 샘플 10
- **입력**: 안면신경통은안면의일부분에서발생하는신경통으로...
- **정답**: 안면신경통(Disease), 안면(Body), 신경통(Disease)
- **예측**:
  - 안면신경통은안면의일부분에서발생하는신경통으로(Disease, 0.86)
  - 얼굴의한쪽면에갑작스럽게발생하며심한통증과함께얼굴의근육이뻣뻣해지는증상을보입니다(Body, 0.57)
- **결과**: ❌ 엔티티 경계 과확장

---

## 문제점 분석

### 1. 엔티티 경계 과확장 (주요 문제)

| 현상 | 원인 추정 |
|------|----------|
| 문장 전체를 하나의 엔티티로 인식 | 한국어 토크나이징 문제 |
| "질환으로", "입니다" 등 포함 | 엔티티 종료 지점 인식 실패 |

**예시**:
```
정답: 전신적다한증 (Disease)
예측: 전신적다한증은신체전체에힘이빠져서일상생활이어려워지는질환으로 (Disease)
```

### 2. 세부 엔티티 미검출

| 현상 | 원인 추정 |
|------|----------|
| 개별 엔티티 구분 못함 | 하나의 큰 엔티티로 병합 |
| Body, Treatment 검출률 낮음 | Disease에 편향 |

**예시**:
```
정답: 췌장암(Disease), 췌장(Body), 종괴(Disease)
예측: 췌장암이란췌장에생긴암세포로이루어진종괴 (Disease만)
```

### 3. 라벨 혼동

| 현상 | 빈도 |
|------|------|
| Body를 Disease로 예측 | 다수 |
| Disease를 Body로 예측 | 일부 (샘플 4, 6) |

---

## 정량적 평가 (추정)

| 지표 | 추정치 | 비고 |
|------|--------|------|
| Precision | ~10-20% | 경계 과확장으로 낮음 |
| Recall | ~30-40% | 일부 엔티티는 포함됨 |
| **F1** | **~20-30%** | Zero-shot 한계 |

### 비교 (참고)

| 모델 | F1 (KBMC) |
|------|-----------|
| GLiNER zero-shot | ~20-30% (추정) |
| KoELECTRA fine-tuned | ~98% (논문) |
| KoBERT fine-tuned | ~98% (논문) |

---

## 원인 분석

### 1. 한국어 특성

| 특성 | 영향 |
|------|------|
| 교착어 (조사 결합) | "다한증**은**", "췌장**에**" → 경계 불명확 |
| 공백 없는 복합어 | "전신적다한증" vs "전신 적 다한증" |
| 형태소 분리 필요 | GLiNER는 subword 기반 |

### 2. Zero-shot 한계

| 한계 | 설명 |
|------|------|
| 도메인 지식 부족 | 의료 용어 경계 학습 안 됨 |
| 한국어 학습 부족 | 영어/다국어 위주 학습 |
| BIO 태깅 미지원 | 토큰 단위 태깅 불가 |

### 3. 입력 형식 문제

| 문제 | 설명 |
|------|------|
| 공백 제거 입력 | 원문 복원 시 토큰 경계 손실 |
| 토큰화된 입력 미사용 | KBMC는 공백 토큰화된 형식 |

---

## 개선 방안

### 방안 1: 토큰화된 입력 사용

```python
# 공백 제거 대신 공백 유지
text = sample['Sentence']  # "전신 적 다한증 은 신체 전체 에..."
entities = model.predict_entities(text, labels)
```

### 방안 2: GLiNER 파인튜닝

```python
from gliner import GLiNERConfig, GLiNER

# KBMC 데이터로 파인튜닝
# 한국어 의료 엔티티 경계 학습
```

### 방안 3: 후처리 추가

```python
# 예측 결과에서 조사 제거
def postprocess(entity_text):
    # "다한증은" → "다한증"
    suffixes = ['은', '는', '이', '가', '을', '를', '에', '의', '로']
    for suffix in suffixes:
        if entity_text.endswith(suffix):
            return entity_text[:-len(suffix)]
    return entity_text
```

### 방안 4: KoELECTRA baseline 먼저 구축

- Fine-tuned 모델과 정확한 비교
- F1 ~98% baseline 확보 후 GLiNER 개선

---

## 결론

| 항목 | 결과 |
|------|------|
| **GLiNER zero-shot 한국어 의료 NER** | 실용 수준 미달 |
| **주요 문제** | 엔티티 경계 과확장, 세부 엔티티 미검출 |
| **추정 F1** | ~20-30% (vs KoELECTRA 98%) |
| **권장 다음 단계** | KoELECTRA baseline 구축 또는 GLiNER 파인튜닝 |

---

## 다음 단계

1. [ ] 토큰화된 입력으로 GLiNER 재테스트
2. [ ] KoELECTRA + KBMC 파인튜닝 (baseline)
3. [ ] GLiNER 파인튜닝 실험
4. [ ] 정확한 F1 비교

---

## 관련 파일

| 파일 | 위치 |
|------|------|
| KBMC 데이터셋 가이드 | `references/kbmc_dataset_guide.md` |
| KBMC 논문 분석 | `references/kbmc_paper_analysis.md` |
| GLiNER2 논문 분석 | `references/gliner2_paper_analysis.md` |
| 기존 NER 노트북 | `notebooks/NER기반_의료_용어_추출기.ipynb` |
