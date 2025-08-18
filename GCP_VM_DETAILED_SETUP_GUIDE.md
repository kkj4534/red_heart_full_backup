# GCP VM 상세 설정 가이드 - Red Heart AI 프로젝트

## 1. 환경 호환성 분석

### 현재 로컬 환경 (WSL)
```
OS: WSL2 Ubuntu (Linux 커널 6.6.87.2)
Python: 3.10+ (venv 사용)
Conda: 서브프로세스용
GPU: 로컬 8GB (사용 불가)
Storage: 130GB 프로젝트
```

### GCP VM 권장 OS 선택

#### ✅ 최적 선택: Deep Learning VM Image
```
이미지 이름: c0-deeplearning-common-gpu-v20240922-debian-11-py310
- Debian 11 (WSL Ubuntu와 99% 호환)
- Python 3.10 기본 설치 (venv 완벽 지원)
- Conda (Miniconda) 사전 설치됨
- CUDA 12.1 + cuDNN 8.9
- PyTorch 2.1.0 사전 설치
```

#### 대안: Ubuntu 20.04 LTS
```
이미지 이름: ubuntu-2004-focal-v20240307
- WSL과 동일한 패키지 관리자 (apt)
- Python 3.8 기본 (3.10 수동 설치 필요)
- CUDA 수동 설치 필요
- venv/conda 수동 설치 필요
```

## 2. VM 생성 상세 설정

### 2.1 기본 정보
```
Name: redheart-training-vm
Region: asia-northeast3 (Seoul)
Zone: asia-northeast3-b
Series: N1
Machine type: n1-standard-8
- vCPUs: 8
- Memory: 30 GB
- 이유: 130GB 데이터 처리 + 다중 프로세스 지원
```

### 2.2 GPU 구성
```
GPU type: NVIDIA T4
Number of GPUs: 1
GPU memory: 16 GB GDDR6

설정 옵션:
□ Install NVIDIA GPU driver automatically
  → Deep Learning VM: 체크 불필요 (이미 설치됨)
  → Ubuntu: 체크 필수

□ Enable Virtual Workstation (NVIDIA GRID)
  → 체크하지 마세요 (딥러닝에 불필요, 추가 비용)
```

### 2.3 부트 디스크 상세
```
Operating system: Deep Learning on Linux
Version: Debian 11 based Deep Learning VM M118
Boot disk type: Balanced persistent disk
Size: 250 GB (권장)
  - 시스템: 20GB
  - 프로젝트: 130GB
  - 임시 파일/캐시: 50GB
  - 체크포인트: 50GB

암호화: Google-managed encryption key (기본)
□ Delete boot disk when instance is deleted: 체크
```

### 2.4 네트워킹 상세
```
Network interface:
- Network: default
- Subnetwork: default (asia-northeast3)
- Primary internal IPv4 address: Ephemeral (automatic)
- External IPv4 address: Ephemeral (필수!)
  → None 선택시 SSH 접속 불가

IP forwarding: Off
Network tags: (선택사항)
- deep-learning
- gpu-vm
- jupyter

Hostname: (비워두기 - 자동 생성)
```

### 2.5 방화벽 규칙
```
✅ Allow HTTP traffic (포트 80)
  - Jupyter Notebook 접속용
✅ Allow HTTPS traffic (포트 443)
  - 보안 연결용

추가 포트 오픈 (필요시):
- 8888: Jupyter
- 6006: TensorBoard
- 22: SSH (기본 오픈)

커스텀 방화벽 규칙 생성:
gcloud compute firewall-rules create allow-jupyter \
  --allow tcp:8888 \
  --source-ranges 0.0.0.0/0 \
  --target-tags deep-learning
```

## 3. Identity and API access

### 3.1 Service account
```
Service account: Compute Engine default service account
또는
Create new service account:
- Name: redheart-vm-sa
- Role: 
  - Storage Admin (GCS 접근)
  - Logging Admin (로그 쓰기)
  - Monitoring Metric Writer (모니터링)
```

### 3.2 Access scopes
```
✅ Allow full access to all Cloud APIs
이유:
- gsutil 사용 (Cloud Storage)
- gcloud 명령어
- BigQuery 접근 (필요시)
- Cloud Logging
```

## 4. Management 설정

### 4.1 Metadata
```
키-값 쌍 추가:

enable-oslogin: TRUE
  → Google 계정으로 SSH 접속

startup-script: |
  #!/bin/bash
  # 시작 스크립트
  echo "VM Started at $(date)" >> /home/startup.log
  
  # Swap 메모리 추가 (선택)
  sudo fallocate -l 32G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  
  # 필수 디렉토리 생성
  mkdir -p /home/$USER/project
  mkdir -p /home/$USER/checkpoints
  
  # GPU 상태 확인
  nvidia-smi >> /home/startup.log
```

### 4.2 Availability policies
```
Preemptibility: Standard (not Spot)
  → Spot VM: 60-91% 저렴하지만 24시간 제한
  → Standard: 안정적, 중단 없음

On host maintenance: Terminate
  → GPU VM은 live migration 불가

Automatic restart: On
  → 예기치 않은 종료시 자동 재시작

Node affinity: None needed
```

### 4.3 SSH Keys
```
방법 1: 브라우저 SSH (가장 쉬움)
- 설정 불필요

방법 2: 로컬 SSH 키 등록
ssh-keygen -t rsa -b 4096 -C "your-email@example.com"
cat ~/.ssh/id_rsa.pub

위 내용을 SSH Keys 섹션에 붙여넣기
```

## 5. Observability

### 5.1 Ops Agent
```
✅ Install Ops Agent for Monitoring and Logging
포함 내용:
- CPU/Memory/Disk 메트릭
- GPU 사용률 (nvidia-smi 기반)
- 시스템 로그
- 애플리케이션 로그

대시보드에서 확인:
- Monitoring > Dashboards > VM Instance
```

### 5.2 Cloud Logging
```
✅ Enable Cloud Logging
로그 레벨: Info
로그 보관: 30일 (기본)
```

## 6. Security

### 6.1 Shielded VM
```
□ Turn on Secure Boot: Off (호환성)
□ Turn on vTPM: Off
□ Turn on Integrity monitoring: Off

→ 딥러닝 워크로드에는 불필요
```

### 6.2 Confidential VM
```
Confidential VM service: Disabled
→ 성능 오버헤드 있음, 불필요
```

## 7. Sole-tenancy
```
Node affinity labels: None
→ 공유 하드웨어 사용 (비용 절감)
```

## 8. 환경 설정 스크립트

### VM 생성 후 실행할 초기 설정
```bash
#!/bin/bash

# 1. 시스템 업데이트
sudo apt update && sudo apt upgrade -y

# 2. Python 환경 확인
python3 --version  # 3.10 확인
pip3 --version

# 3. venv 생성 (WSL과 동일)
python3 -m venv ~/redheart_env
source ~/redheart_env/bin/activate

# 4. Conda 환경 생성 (서브프로세스용)
conda create -n subprocess_env python=3.10 -y
conda activate subprocess_env

# 5. 필수 패키지 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate bitsandbytes

# 6. CUDA 확인
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# 7. 프로젝트 디렉토리 준비
mkdir -p ~/project/linux_red_heart
mkdir -p ~/checkpoints
mkdir -p ~/logs

# 8. Git 설정 (선택)
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
```

## 9. 데이터 전송 전략

### 9.1 대용량 파일 전송 (130GB)
```bash
# 로컬에서 압축 (시간 단축)
tar -czf redheart_project.tar.gz \
  --exclude='*.pyc' \
  --exclude='__pycache__' \
  --exclude='.git' \
  linux_red_heart/

# Google Cloud Storage 경유 (가장 빠름)
# 1. 버킷 생성
gsutil mb -l asia-northeast3 gs://redheart-data/

# 2. 업로드 (병렬 처리)
gsutil -m -o GSUtil:parallel_composite_upload_threshold=150M \
  cp redheart_project.tar.gz gs://redheart-data/

# 3. VM에서 다운로드
gsutil cp gs://redheart-data/redheart_project.tar.gz .
tar -xzf redheart_project.tar.gz

# 예상 시간:
# 압축: 30-60분
# 업로드: 60-120분 (네트워크 속도 의존)
# 다운로드: 10-20분 (GCP 내부 네트워크)
```

### 9.2 직접 SCP (작은 파일)
```bash
# 단일 파일
gcloud compute scp file.py instance-name:~/project/ \
  --zone=asia-northeast3-b

# 디렉토리
gcloud compute scp -r ./linux_red_heart instance-name:~/project/ \
  --zone=asia-northeast3-b
```

## 10. 비용 최적화

### 10.1 예상 비용 (서울 리전)
```
구성요소별 시간당 비용:
- n1-standard-8: $0.38/시간
- NVIDIA T4: $0.35/시간
- 250GB 디스크: $0.014/시간
- External IP: $0.004/시간
- 네트워크 egress: $0.12/GB

총합: 약 $0.75/시간 (₩1,000)

월간 예상 (100시간 사용):
- VM + GPU: $75
- 스토리지: $10
- 네트워크: $5
총: 약 $90 (₩120,000)
```

### 10.2 비용 절감 팁
```
1. Spot VM 사용 (60% 할인)
   - 단점: 24시간 제한, 중단 가능
   
2. 사용하지 않을 때 STOP
   - 디스크 비용만 과금
   
3. 스냅샷 생성 후 VM 삭제
   - 필요시 스냅샷에서 복구
   
4. Committed use discount (1년 약정시 37% 할인)
```

## 11. 문제 해결

### 11.1 GPU 할당 실패
```
오류: "Quota 'GPUS_ALL_REGIONS' exceeded"
해결:
1. IAM & Admin > Quotas
2. Filter: gpu
3. EDIT QUOTAS 클릭
4. 1로 증가 요청
```

### 11.2 SSH 접속 실패
```
오류: "Permission denied (publickey)"
해결:
1. 브라우저 SSH 사용
2. 또는 gcloud compute ssh --troubleshoot
```

### 11.3 디스크 공간 부족
```
확인: df -h
해결:
1. VM 정지
2. 디스크 크기 증가
3. VM 시작 후 파티션 확장
sudo growpart /dev/sda 1
sudo resize2fs /dev/sda1
```

## 12. 체크리스트

### VM 생성 전
- [ ] GPU 할당량 확인
- [ ] 예산 설정
- [ ] 데이터 압축 준비

### VM 생성 시
- [ ] 서울 리전 선택
- [ ] n1-standard-8 선택
- [ ] T4 GPU 추가
- [ ] Deep Learning VM 선택
- [ ] 250GB 디스크 설정
- [ ] External IP 활성화
- [ ] HTTP/HTTPS 방화벽 오픈
- [ ] Cloud API 전체 액세스

### VM 생성 후
- [ ] SSH 접속 확인
- [ ] nvidia-smi 실행
- [ ] Python 환경 설정
- [ ] 프로젝트 업로드
- [ ] 테스트 실행

---

작성일: 2024년
프로젝트: Red Heart AI (666M 파라미터)
환경: WSL → GCP VM 마이그레이션