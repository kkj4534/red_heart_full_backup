# Red Heart AI 환경 복구 가이드

## 🚨 환경 복구가 필요한 경우

1. **의존성 충돌** 발생 시
2. **CUDA 버전 불일치** 문제
3. **FAISS import 오류**
4. **환경 초기화** 필요 시

## 📦 현재 환경 구조

### 두 개의 독립 환경
```
├── venv (red_heart_env/)
│   ├── Python 3.12.3
│   ├── PyTorch 2.5.1+cu121
│   ├── Transformers 4.53.1
│   └── 172개 패키지
│
└── conda (faiss-test)
    ├── Python 3.12.3
    ├── FAISS GPU 1.11.0
    ├── CUDA 11.8.89
    └── 105개 패키지
```

## 🔧 완전 복구 절차

### 1단계: 기존 환경 백업
```bash
# venv 패키지 목록 백업
source red_heart_env/bin/activate
pip freeze > backup_venv_$(date +%Y%m%d_%H%M%S).txt

# conda 패키지 목록 백업
conda list --export > backup_conda_$(date +%Y%m%d_%H%M%S).txt
conda env export > backup_conda_env_$(date +%Y%m%d_%H%M%S).yml
```

### 2단계: 환경 제거 (필요시)
```bash
# venv 제거
deactivate
rm -rf red_heart_env/

# conda 환경 제거
conda deactivate
conda env remove -n faiss-test
```

### 3단계: Conda 환경 재생성
```bash
# conda 환경 생성
conda env create -f requirements_conda_complete.yml

# 또는 수동 생성
conda create -n faiss-test python=3.12.3
conda activate faiss-test
conda install -c pytorch faiss-gpu=1.11.0
conda install -c nvidia cuda-cudart=11.8.89
```

### 4단계: venv 환경 재생성
```bash
# venv 생성
python3.12 -m venv red_heart_env
source red_heart_env/bin/activate

# pip 업그레이드
pip install --upgrade pip setuptools wheel

# 패키지 설치
pip install -r requirements_venv_complete.txt

# PyTorch CUDA 버전 확인 필요시
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### 5단계: 통합 스크립트 설정
```bash
# activate 스크립트 수정
cat >> red_heart_env/bin/activate << 'EOF'

# Red Heart AI 통합 환경 활성화
source ~/miniconda3/etc/profile.d/conda.sh
conda activate faiss-test

echo "✅ Red Heart AI 통합 환경 활성화 완료"
echo "   - venv: $VIRTUAL_ENV"
echo "   - conda: faiss-test (함께 활성화됨)"
echo "   - python: $(which python)"
echo "   - 환경 분리: faiss→conda subprocess, 나머지→venv"
EOF
```

## 🔍 버전 충돌 해결

### 주요 충돌 패키지와 해결책

| 패키지 | venv | conda | 해결 방법 |
|--------|------|-------|-----------|
| h11 | 0.16.0 | 0.9.0 | venv 버전 사용 |
| httpx | 0.28.1 | 0.13.3 | venv 버전 사용 |
| PyYAML | 5.1.2 | 6.0.2 | 각 환경별 독립 사용 |
| setuptools | 70.2.0 | 80.9.0 | 각 환경별 독립 사용 |
| typing-extensions | 4.12.2 | 4.14.1 | venv 버전 우선 |

### 충돌 발생 시 수동 해결
```bash
# venv에서 특정 버전 강제 설치
pip install --force-reinstall package==version

# conda에서 특정 버전 고정
conda install package=version
```

## 🧪 환경 검증

### 기본 검증
```bash
# 환경 활성화
source red_heart_env/bin/activate

# Python 버전
python --version  # 3.12.3

# 핵심 패키지 임포트 테스트
python -c "
import torch
import transformers
import faiss
import numpy as np
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Transformers: {transformers.__version__}')
print(f'NumPy: {np.__version__}')
print('FAISS GPU: OK')
"
```

### 전체 모듈 테스트
```bash
python -c "
from unified_system import UnifiedSystem
from module_selector import ModuleSelector, ExecutionMode

selector = ModuleSelector()
selector.set_mode(ExecutionMode.TRAINING)
selector.print_summary()

print('✅ 모든 모듈 로드 성공')
"
```

## 📋 패키지 버전 고정 목록

### 절대 변경 금지
- `numpy==1.26.4` (FAISS 호환성)
- `faiss-gpu==1.11.0` (CUDA 11.4.4)
- `torch==2.5.1+cu121` (CUDA 12.1)

### 주의 필요
- `transformers==4.53.1` (모델 가중치 호환)
- `sentence-transformers==5.0.0` (임베딩 호환)
- `llama_cpp_python==0.3.12` (CUDA 지원)

## 🔄 부분 복구

### venv만 복구
```bash
deactivate
rm -rf red_heart_env/
python3.12 -m venv red_heart_env
source red_heart_env/bin/activate
pip install -r requirements_venv_complete.txt
```

### conda FAISS만 재설치
```bash
conda activate faiss-test
conda remove faiss-gpu
conda install -c pytorch faiss-gpu=1.11.0
```

### 특정 패키지 그룹만 복구
```bash
# ML 코어만
pip install torch==2.5.1+cu121 transformers==4.53.1 sentence-transformers==5.0.0

# NLP 한국어만
pip install konlpy==0.6.0 kss==6.0.0 soynlp==0.0.493

# 시각화만
pip install matplotlib==3.10.3 seaborn==0.13.2 plotly==5.24.1
```

## ⚠️ 주의사항

1. **CUDA 버전 혼재**
   - PyTorch: CUDA 12.1
   - FAISS: CUDA 11.4.4
   - 동시 사용 가능하나 메모리 오버헤드 있음

2. **NumPy 버전**
   - 반드시 1.26.4 유지
   - 2.x 버전 설치 시 FAISS 오류

3. **환경 변수**
   ```bash
   export CUDA_HOME=/usr/local/cuda-11.8
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
   ```

4. **메모리 제한**
   - GPU: 8GB VRAM
   - 학습 시 gradient_accumulation_steps=16 필수

## 📁 관련 파일

| 파일명 | 용도 |
|--------|------|
| `requirements_complete.txt` | 전체 패키지 목록 및 설명 |
| `requirements_venv_complete.txt` | venv 전용 (172개) |
| `requirements_conda_complete.yml` | conda 환경 정의 |
| `requirements_venv_new.txt` | 현재 venv 스냅샷 |
| `requirements_conda_export.txt` | 현재 conda 스냅샷 |

## 🆘 문제 해결

### ImportError: No module named 'faiss'
```bash
conda activate faiss-test
conda install -c pytorch faiss-gpu=1.11.0
```

### CUDA out of memory
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### Segmentation fault (FAISS)
```bash
# NumPy 버전 확인
pip show numpy  # 1.26.4여야 함
pip install --force-reinstall numpy==1.26.4
```

### 환경 활성화 실패
```bash
# 수동 활성화
source red_heart_env/bin/activate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate faiss-test
```

---
*마지막 업데이트: 2025-08-18*
*작성자: Red Heart AI Team*