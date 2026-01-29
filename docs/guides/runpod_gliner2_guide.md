# RunPod GLiNER2 학습 가이드

> Silver-set 데이터로 GLiNER2 파인튜닝

---

## 0. 준비물 확인

로컬에 준비되어 있어야 함:

```
project/ner/data/gliner2_train/
├── train.jsonl    (159건)
├── test.jsonl     (17건)
└── meta.json
```

없으면 먼저 실행:
```bash
python project/ner/scripts/gliner2/convert_to_gliner2.py
```

---

## 1. Pod 생성

### RunPod 접속
https://runpod.io → 로그인

### GPU 선택 (권장)

| GPU | 시간당 | VRAM | 추천 |
|-----|--------|------|------|
| RTX 3090 | $0.22 | 24GB | 가성비 최고 |
| RTX 4090 | $0.44 | 24GB | 빠름 |
| A100 40GB | $1.89 | 40GB | 대용량 배치 |

### 설정
1. **Pods** → **+ Deploy**
2. GPU: **RTX 3090** 선택
3. Template: **RunPod Pytorch 2.x**
4. **Deploy On-Demand**

---

## 2. SSH 접속 정보 확인

Pod 생성 후 **Connect** 버튼 클릭

```
ssh root@<IP> -p <PORT> -i ~/.ssh/id_rsa
```

**예시** (실제 값으로 대체):
- IP: `213.173.96.17`
- PORT: `22057`

---

## 3. 파일 전송 (로컬 → RunPod)

**PowerShell에서 실행** (경로는 D드라이브 기준):

```powershell
# 변수 설정 (실제 값으로 변경)
$IP = "213.173.96.17"
$PORT = "22057"

# 데이터 파일 전송
scp -P $PORT D:/scrape-hub/project/ner/data/gliner2_train/train.jsonl root@${IP}:~/
scp -P $PORT D:/scrape-hub/project/ner/data/gliner2_train/test.jsonl root@${IP}:~/
scp -P $PORT D:/scrape-hub/project/ner/data/gliner2_train/meta.json root@${IP}:~/

# 학습 스크립트 전송
scp -P $PORT D:/scrape-hub/project/ner/scripts/gliner2/train_gliner2_silver.py root@${IP}:~/
```

**Git Bash / Mac / Linux**:
```bash
IP="213.173.96.17"
PORT="22057"

scp -P $PORT /d/scrape-hub/project/ner/data/gliner2_train/*.jsonl root@$IP:~/
scp -P $PORT /d/scrape-hub/project/ner/data/gliner2_train/meta.json root@$IP:~/
scp -P $PORT /d/scrape-hub/project/ner/scripts/gliner2/train_gliner2_silver.py root@$IP:~/
```

---

## 4. SSH 접속

```bash
ssh root@<IP> -p <PORT>
```

예시:
```bash
ssh root@213.173.96.17 -p 22057
```

---

## 5. 환경 설정 (RunPod 안에서)

```bash
# 1. 패키지 설치
pip install --upgrade pip
pip install gliner2 transformers datasets accelerate

# 2. GPU 확인
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"

# 3. 파일 확인
ls -la *.jsonl
head -1 train.jsonl
```

---

## 6. 학습 실행

```bash
# 기본 실행
python train_gliner2_silver.py

# 옵션 지정
python train_gliner2_silver.py --epochs 20 --batch-size 8 --lr 5e-4
```

**예상 시간**: RTX 3090 기준 159건 × 20 epochs ≈ 10-20분

---

## 7. 학습 모니터링

```bash
# 실시간 GPU 사용량
watch -n 1 nvidia-smi

# 별도 터미널에서 로그 확인
tail -f gliner2_silver/training.log
```

---

## 8. 결과 다운로드 (로컬에서)

```powershell
# 결과 텍스트
scp -P $PORT root@${IP}:~/gliner2_silver/results.txt ./

# 모델 전체 (LoRA 어댑터)
scp -rP $PORT root@${IP}:~/gliner2_silver/best ./gliner2_silver_model/
```

---

## 9. Pod 종료 (필수!)

**학습 완료 후 반드시 종료**

| 옵션 | 설명 |
|------|------|
| **Stop** | 일시정지 (볼륨 유지, 소액 과금) |
| **Terminate** | 완전 삭제 (과금 중지) |

---

## 비용 예상

| GPU | 시간당 | 20분 학습 |
|-----|--------|-----------|
| RTX 3090 | $0.22 | ~$0.07 |
| RTX 4090 | $0.44 | ~$0.15 |

---

## 전체 플로우 요약

```bash
# === 로컬 (PowerShell) ===
$IP = "YOUR_IP"
$PORT = "YOUR_PORT"

# 파일 전송
scp -P $PORT D:/scrape-hub/project/ner/data/gliner2_train/*.jsonl root@${IP}:~/
scp -P $PORT D:/scrape-hub/project/ner/scripts/gliner2/train_gliner2_silver.py root@${IP}:~/

# SSH 접속
ssh root@$IP -p $PORT

# === RunPod 안에서 ===
pip install gliner2 transformers datasets accelerate
python train_gliner2_silver.py

# === 로컬에서 결과 다운로드 ===
scp -rP $PORT root@${IP}:~/gliner2_silver ./
```

---

## 문제 해결

### gliner2 import 에러
```bash
pip uninstall gliner gliner2
pip install gliner2
```

### CUDA OOM (메모리 부족)
```bash
# batch_size 줄이기
python train_gliner2_silver.py --batch-size 4

# 또는 gradient accumulation
python train_gliner2_silver.py --batch-size 2 --grad-accum 4
```

### SSH 접속 실패
```bash
# known_hosts 초기화 (IP 변경됐을 때)
ssh-keygen -R <IP>
```

---

## 다음 단계

1. 학습 완료 후 테스트셋 평가
2. 결과 좋으면 전체 데이터(1000건)로 재학습
3. 실서비스 적용 검토
