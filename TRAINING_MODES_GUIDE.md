# Red Heart AI 통합 학습 시스템 - 실행 모드 가이드

## 📋 개요

Red Heart AI 730M 파라미터 모델의 다양한 학습 및 테스트 모드를 제공합니다.
모든 명령은 `run_learning.sh` 스크립트를 통해 실행되며, 가상환경과 의존성이 자동으로 관리됩니다.

## 🚀 Quick Start

```bash
# 빠른 테스트 (2 에폭)
bash run_learning.sh unified-test

# 샘플 제한 테스트 (3 에폭)
SAMPLES=3 bash run_learning.sh unified-test

# 전체 학습 (60 에폭)
bash run_learning.sh unified-train

# nohup으로 백그라운드 실행 (20분 타임아웃)
nohup timeout 1200 bash run_learning.sh unified-test --samples 3 --debug --verbose &
```

## 🔧 실행 모드 상세

### 1. unified-test (테스트 모드)
**용도**: 시스템 검증 및 빠른 테스트
```bash
# 기본 테스트 (2 에폭)
bash run_learning.sh unified-test

# 샘플 수 지정 (N 에폭)
SAMPLES=5 bash run_learning.sh unified-test

# 디버그 모드
bash run_learning.sh unified-test --debug --verbose
```

**특징**:
- 730M 파라미터 모델 사용
- LR 스윕 포함 (5개 LR 테스트)
- Sweet Spot 탐지 활성화
- Parameter Crossover 포함
- Advanced Training Techniques 활성화
- 메모리 효율적 (배치 사이즈 2)

### 2. unified-train (전체 학습 모드)
**용도**: 60 에폭 전체 학습
```bash
# 기본 60 에폭 학습
bash run_learning.sh unified-train

# 커스텀 에폭 수
bash run_learning.sh unified-train --epochs 30

# 체크포인트에서 재개
bash run_learning.sh unified-train --resume training/checkpoints_final/checkpoint_epoch_0020.pt
```

**특징**:
- 60 에폭 학습 (약 2-3일 소요)
- 짝수 에폭마다 체크포인트 저장 (총 30개)
- 모듈별 Sweet Spot 자동 탐지
- 학습 완료 후 Parameter Crossover 실행
- 최종 crossover_final.pth 생성

### 3. unified-test-v1 (레거시 모드)
**용도**: 기존 800M/320M 시스템 테스트
```bash
bash run_learning.sh unified-test-v1
```

### 4. train-local (로컬 학습 테스트)
**용도**: GPU 메모리 사용량 모니터링과 함께 소규모 샘플 테스트
```bash
bash run_learning.sh train-local
SAMPLES=10 bash run_learning.sh train-local
```

### 5. validate (시스템 검증)
**용도**: 학습 없이 시스템 무결성 검사
```bash
bash run_learning.sh validate
```

## 📊 학습 설정

### 기본 하이퍼파라미터
```python
# 모델 설정
model_params = 730_000_000  # 730M 파라미터
hidden_dim = 1280
num_layers = 18
num_heads = 20

# 학습 설정
total_epochs = 60
micro_batch_size = 2  # GPU 메모리 절약
gradient_accumulation = 32  # 유효 배치 = 64
base_lr = 1e-4

# LR 스윕
lr_sweep_range = (1e-5, 1e-2)
lr_sweep_points = 5

# Advanced Training
label_smoothing = 0.1
rdrop_alpha = 1.0
ema_decay = 0.999
```

### 메모리 관리
- **초기 배치 사이즈**: 2 (안정성 우선)
- **OOM 핸들링**: 메모리 부족 시 자동 정리 (폴백 비활성화)
- **Gradient Accumulation**: 32 스텝 (유효 배치 = 64)
- **Dynamic Swap Manager**: 자동 활성화 가능

## 🎯 권장 워크플로우

### 1단계: 환경 검증
```bash
# 시스템 상태 확인
bash run_learning.sh validate
```

### 2단계: 컴포넌트 테스트
```bash
# 빠른 컴포넌트 테스트
python training/test_unified_training.py --quick
```

### 3단계: 미니 학습 테스트
```bash
# 2-3 에폭 테스트
SAMPLES=3 bash run_learning.sh unified-test --debug
```

### 4단계: 본 학습
```bash
# 백그라운드 실행 (nohup)
nohup bash run_learning.sh unified-train > training.log 2>&1 &

# 또는 screen/tmux 사용
screen -S training
bash run_learning.sh unified-train
# Ctrl+A, D로 detach
```

### 5단계: 모니터링
```bash
# 로그 모니터링
tail -f training.log

# GPU 사용량 모니터링
watch -n 1 nvidia-smi

# 체크포인트 확인
ls -lah training/checkpoints_final/
```

## 🔍 디버그 옵션

### --debug 플래그
상세한 디버그 정보 출력
```bash
bash run_learning.sh unified-test --debug
```

### --verbose 플래그
모든 로그 메시지 출력
```bash
bash run_learning.sh unified-test --verbose
```

### --samples N
테스트용 샘플/에폭 수 제한
```bash
SAMPLES=5 bash run_learning.sh unified-test
```

## 📁 출력 디렉토리 구조

```
training/
├── checkpoints_final/        # 체크포인트 저장
│   ├── checkpoint_epoch_0002_lr_0.000100_*.pt
│   ├── checkpoint_epoch_0004_lr_0.000095_*.pt
│   └── crossover_final.pth   # 최종 모델
├── lr_sweep_results/          # LR 스윕 결과
│   ├── lr_sweep_*.json
│   └── lr_sweep_plot_*.png
├── sweet_spot_analysis/       # Sweet Spot 분석
│   ├── sweet_spot_analysis_*.json
│   ├── sweet_spot_report_*.txt
│   └── plots/
├── oom_stats/                 # OOM 통계
│   └── oom_stats_*.json
└── logs/                      # 학습 로그
    └── training_*.log
```

## ⚠️ 주의사항

1. **메모리 요구사항**
   - GPU: 최소 8GB VRAM (RTX 2070S 이상)
   - RAM: 최소 16GB (권장 32GB)
   - 디스크: 최소 50GB 여유 공간

2. **학습 시간**
   - 테스트 모드 (2 에폭): 약 1-2시간
   - 전체 학습 (60 에폭): 약 2-3일

3. **배치 사이즈**
   - 기본값 2 유지 권장 (안정성)
   - OOM 발생 시 메모리 정리 후 재시도

4. **체크포인트**
   - 짝수 에폭마다 자동 저장
   - 최대 30개 유지 (오래된 것 자동 삭제)

## 🛠️ 문제 해결

### OOM (Out of Memory) 에러
```bash
# 메모리 상태 확인
python -c "import torch; print(torch.cuda.memory_summary())"

# 캐시 정리
python -c "import torch; torch.cuda.empty_cache()"

# 배치 사이즈 줄이기 (이미 2로 설정됨)
bash run_learning.sh unified-test --batch-size 1
```

### 가상환경 문제
```bash
# 가상환경 재생성
rm -rf red_heart_env
python3 -m venv red_heart_env
source red_heart_env/bin/activate
pip install -r requirements_venv.txt
```

### 체크포인트 복구
```bash
# 최신 체크포인트에서 재개
LATEST=$(ls -t training/checkpoints_final/*.pt | head -1)
bash run_learning.sh unified-train --resume $LATEST
```

## 📈 성능 모니터링

### 실시간 메트릭 확인
```python
# Python 스크립트로 메트릭 확인
import json
with open('training/checkpoints_final/metadata.json') as f:
    metadata = json.load(f)
    for checkpoint in metadata:
        print(f"Epoch {checkpoint['epoch']}: Loss={checkpoint['metrics']['loss']:.4f}")
```

### Sweet Spot 분석 확인
```bash
# 최신 Sweet Spot 리포트 확인
cat training/sweet_spot_analysis/sweet_spot_report_*.txt
```

### 학습 곡선 시각화
```python
# 별도 스크립트 실행
python training/plot_training_curves.py
```

## 🎯 최종 모델 사용

학습 완료 후:
```python
import torch

# Crossover 모델 로드
model = torch.load('training/checkpoints_final/crossover_final.pth')

# 또는 특정 체크포인트 로드
checkpoint = torch.load('training/checkpoints_final/checkpoint_epoch_0060.pt')
model.load_state_dict(checkpoint['model_state'])
```

## 📚 추가 참고자료

- [RED_HEART_AI_FINAL_TRAINING_STRATEGY.md](RED_HEART_AI_FINAL_TRAINING_STRATEGY.md) - 전체 학습 전략
- [requirements_venv.txt](requirements_venv.txt) - Python 의존성
- [training/test_unified_training.py](training/test_unified_training.py) - 단위 테스트

---

**마지막 업데이트**: 2025-08-18
**모델 버전**: 730M (최종)
**문서 버전**: 1.0