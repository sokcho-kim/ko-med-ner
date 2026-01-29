# RunPod GPU 학습 가이드

> KoELECTRA + KBMC 학습용

---

## 1. 사전 준비 (1회만)

### SSH 키 생성 (없으면)
```bash
ssh-keygen -t rsa -b 4096
# 엔터 3번 (기본값)
```

### SSH 공개키 확인
```bash
cat ~/.ssh/id_rsa.pub
```

### RunPod에 키 등록
1. https://runpod.io 로그인
2. **Settings** → **SSH Public Keys** → **Update Public Key**
3. 공개키 붙여넣기 → 저장

---

## 2. Pod 생성

1. **Pods** → **+ Deploy**
2. GPU 선택: **RTX 3090** (~$0.22/hr, 가성비)
3. Template: **RunPod Pytorch 2.x**
4. **Deploy On-Demand** 클릭

---

## 3. SSH 정보 확인

Pod 생성 후 **Connect** 클릭하면 SSH 명령어 나옴:

```
ssh root@<IP> -p <PORT> -i ~/.ssh/id_rsa
```

예시:
- IP: `213.173.96.17`
- PORT: `22057`

---

## 4. 파일 전송 (로컬 → RunPod)

```bash
scp -P <PORT> C:/Jimin/scrape-hub/project/ner/scripts/train/train_koelectra_kbmc.py root@<IP>:~/
```

예시:
```bash
scp -P 22057 C:/Jimin/scrape-hub/project/ner/scripts/train/train_koelectra_kbmc.py root@213.173.96.17:~/
```

---

## 5. RunPod 접속

```bash
ssh root@<IP> -p <PORT>
```

예시:
```bash
ssh root@213.173.96.17 -p 22057
```

---

## 6. 학습 실행 (RunPod 안에서)

```bash
# 패키지 설치 (torch 2.6+ 필요)
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets accelerate scikit-learn

# 학습 실행
python train_koelectra_kbmc.py
```

> **참고**: torch 2.6 미만이면 보안 취약점으로 모델 로드 실패함

**예상 시간: 5-10분** (RTX 3090 기준)

---

## 7. 결과 확인 (RunPod 안에서)

```bash
# 결과 파일 확인
cat koelectra-kbmc-baseline/results.txt
```

---

## 8. 결과 다운로드 (로컬에서)

```bash
# 결과 텍스트만
scp -P <PORT> root@<IP>:~/koelectra-kbmc-baseline/results.txt ./

# 모델 전체 (선택)
scp -rP <PORT> root@<IP>:~/koelectra-kbmc-baseline/final ./koelectra-model/
```

---

## 9. Pod 종료 (중요!)

**안 쓰면 반드시 Stop/Terminate**
- Stop: 일시정지 (볼륨 유지, 약간 과금)
- Terminate: 완전 삭제 (과금 중지)

---

## 비용 참고

| GPU | 시간당 | 10분 학습 |
|-----|--------|-----------|
| RTX 3090 | $0.22 | ~$0.04 |
| RTX 4090 | $0.44 | ~$0.07 |
| A100 | $1.89 | ~$0.32 |

---

## 문제 해결

### SSH 접속 안 됨
```bash
# 키 권한 확인
chmod 600 ~/.ssh/id_rsa

# known_hosts 초기화
ssh-keygen -R <IP>
```

### 패키지 설치 오류
```bash
pip install --upgrade pip
pip install transformers datasets accelerate scikit-learn
```

### CUDA 오류
```bash
# GPU 확인
nvidia-smi

# PyTorch CUDA 확인
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 전체 명령어 요약

```bash
# 1. 파일 전송
scp -P <PORT> C:/Jimin/scrape-hub/project/ner/scripts/train/train_koelectra_kbmc.py root@<IP>:~/

# 2. SSH 접속
ssh root@<IP> -p <PORT>

# 3. 학습 (RunPod 안에서)
pip install transformers datasets accelerate scikit-learn
python train_koelectra_kbmc.py

# 4. 결과 다운로드 (로컬에서)
scp -P <PORT> root@<IP>:~/koelectra-kbmc-baseline/results.txt ./
```
