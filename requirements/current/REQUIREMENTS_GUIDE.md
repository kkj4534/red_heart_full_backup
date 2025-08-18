# Red Heart AI 환경 설정 가이드

## 📦 환경 구성
- **venv**: Python 3.12.3 기반 가상환경
- **conda**: faiss-test 환경 (FAISS GPU 전용)
- **통합 활성화**: 두 환경이 자동으로 함께 활성화됨

## 🚀 빠른 설치

### 1. 환경 활성화
```bash
source red_heart_env/bin/activate
# ✅ venv와 conda(faiss-test) 환경이 함께 활성화됩니다
```

### 2. 패키지 설치
```bash
# venv 패키지 설치 (메인 환경)
pip install -r requirements_venv_new.txt

# conda 패키지는 이미 설치되어 있음 (faiss-gpu 포함)
# 필요시: conda install --file requirements_conda_export.txt
```

## 📋 주요 패키지 버전 (2025-08-18 기준)

### 핵심 ML/AI
- **torch**: 2.5.1+cu121
- **transformers**: 4.53.1
- **sentence-transformers**: 5.0.0
- **faiss-gpu**: 1.11.0 (conda)
- **llama-cpp-python**: 0.3.12

### 과학 계산
- **numpy**: 1.26.4 (FAISS 호환)
- **scipy**: 1.16.0
- **pandas**: 2.3.0
- **scikit-learn**: 1.7.0

### NLP
- **konlpy**: 0.6.0
- **nltk**: 3.9.1
- **kss**: 6.0.0
- **soynlp**: 0.0.493

## 🔧 환경 분리 전략

### venv (pip)
- 일반 Python 패키지
- ML/AI 라이브러리
- NLP 도구
- 시각화 라이브러리

### conda
- FAISS GPU (CUDA 11.4.4)
- CUDA 런타임
- MKL 최적화 라이브러리

## ⚠️ 주의사항

1. **NumPy 버전 고정**: 1.26.4 (FAISS 호환성)
2. **CUDA 버전**: 12.1 (PyTorch) / 11.4.4 (FAISS)
3. **환경 활성화**: 항상 `source red_heart_env/bin/activate` 사용

## 📁 관련 파일

- `requirements.txt`: 통합 요구사항 (참조용)
- `requirements_venv_new.txt`: venv 패키지 목록 (172개)
- `requirements_conda_export.txt`: conda 패키지 목록 (105개)
- `requirements_backup_*.txt`: 이전 버전 백업

## 🔄 업데이트 방법

```bash
# 1. 환경 활성화
source red_heart_env/bin/activate

# 2. 현재 패키지 목록 추출
pip freeze > requirements_venv_new.txt
conda list --export > requirements_conda_export.txt

# 3. 백업
cp requirements.txt requirements_backup_$(date +%Y%m%d).txt

# 4. 업데이트
pip install --upgrade -r requirements_venv_new.txt
```

## 🧪 환경 검증

```bash
# Python 버전 확인
python --version  # Python 3.12.3

# 주요 패키지 임포트 테스트
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import faiss; print('FAISS GPU: OK')"
```

## 📊 메모리 사용량

- **학습 모드**: ~6.5GB VRAM (730M 모델)
- **추론 모드**: ~7.5GB VRAM (전체 모듈)
- **RAM 권장**: 32GB+
- **스왑**: DSM 활성화시 RAM으로 오프로드 가능

---
*마지막 업데이트: 2025-08-18 18:35*