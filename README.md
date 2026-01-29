# ko-med-ner

Fine-tuning GLiNER2 for Korean Medical Entity Recognition and Knowledge Base (KCD, SNOMED-CT) Mapping.

## Entity Types

| Type | 설명 | 예시 |
|------|------|------|
| Disease | 질병, 증상, 의학적 상태 | 당뇨병, 고혈압, 폐렴 |
| Drug | 약물, 의약품, 치료제 | 인슐린, 아스피린, 항생제 |
| Procedure | 의료 시술, 수술, 검사 | 내시경, MRI, 수술 |
| Biomarker | 바이오마커, 검사 수치 | 혈당, 콜레스테롤, 종양표지자 |

## 프로젝트 구조

```
ko-med-ner/
├── docker/                # RunPod 학습 환경 Docker
├── configs/               # 실험 설정 YAML
├── data/                  # 학습/평가 데이터
├── scripts/               # 학습, 평가, 검증 스크립트
│   └── data_prep/         # 데이터 전처리 스크립트
├── results/               # 실험 결과
├── models/                # 모델 체크포인트 (gitignore)
├── docs/                  # 문서 (계획, 리포트, 가이드)
├── references/            # 논문, 선행연구 분석
└── notebooks/             # 탐색/실험 노트북
```

## 실행 환경

### Docker (RunPod)

```bash
# 이미지 빌드
cd docker
docker build -t gliner2-train .

# 컨테이너 실행
docker run --gpus all -v $(pwd):/workspace gliner2-train bash
```

### 로컬 환경

```bash
uv venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -r docker/requirements.txt
```

## 주요 스크립트

| 스크립트 | 설명 |
|---------|------|
| `scripts/train_gliner2_korean_encoder.py` | 한국어 인코더 교체 학습 (Surgical Swap) |
| `scripts/train_gliner2_silver.py` | LoRA/Full 파인튜닝 |
| `scripts/verify_encoder_swap.py` | 인코더 교체 사전 검증 (7-step) |
| `scripts/eval.py` | 모델 평가 |
| `scripts/convert_to_gliner2.py` | 데이터 형식 변환 |

## 데이터

- `data/gliner2_train_v2/`: GPT-5 검증 데이터 (696 train, 77 test)
- `data/gazetteer/`: 엔티티 사전 (disease, drug, procedure, biomarker)
- `data/silver_set/`: 전처리 중간 데이터
- `data/pilot_labeling/`: 파일럿 라벨링 데이터
