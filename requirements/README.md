# Requirements 관리 체계

## 📂 폴더 구조

```
requirements/
├── current/          # 현재 사용 중인 최신 의존성 (2025-08-18)
├── previous/         # 이전 버전 의존성 (2025-06~08)
├── archive/          # 오래된 백업 (2025-07-04 이전)
└── README.md         # 이 문서
```

## 🚀 현재 사용 중 (current/)

### 핵심 파일
- **requirements_main.txt** - 메인 통합 의존성 (업데이트됨)
- **requirements_complete.txt** - 전체 패키지 상세 목록 (주석 포함)
- **requirements_venv_complete.txt** - venv 환경 전용 (172개 패키지)
- **requirements_conda_complete.yml** - conda 환경 정의 (105개 패키지)

### 스냅샷
- **requirements_venv_new.txt** - 현재 venv 환경 스냅샷 (pip freeze)
- **requirements_conda_export.txt** - 현재 conda 환경 스냅샷 (conda list --export)

### 문서
- **REQUIREMENTS_GUIDE.md** - 환경 설정 가이드
- **ENVIRONMENT_RECOVERY_GUIDE.md** - 환경 복구 가이드

## 📦 빠른 설치

### 전체 환경 설치
```bash
# 1. conda 환경 생성
conda env create -f requirements/current/requirements_conda_complete.yml

# 2. venv 활성화
source red_heart_env/bin/activate

# 3. pip 패키지 설치
pip install -r requirements/current/requirements_venv_complete.txt
```

### 간단 설치 (메인 파일만)
```bash
pip install -r requirements/current/requirements_main.txt
```

## 🔍 파일별 용도

| 파일명 | 용도 | 패키지 수 | 환경 |
|--------|------|-----------|------|
| `requirements_main.txt` | 기본 통합 의존성 | ~100개 | 통합 |
| `requirements_complete.txt` | 전체 상세 목록 | 277개 | 통합 |
| `requirements_venv_complete.txt` | venv 전용 | 172개 | venv |
| `requirements_conda_complete.yml` | conda 환경 정의 | 105개 | conda |
| `requirements_venv_new.txt` | 현재 venv 스냅샷 | 172개 | venv |
| `requirements_conda_export.txt` | 현재 conda 스냅샷 | 105개 | conda |

## ⚠️ 중요 버전 정보

### 고정 버전 (절대 변경 금지)
- **numpy**: 1.26.4 (FAISS 호환성)
- **faiss-gpu**: 1.11.0 (CUDA 11.4.4)
- **torch**: 2.5.1+cu121 (CUDA 12.1)

### 주요 패키지 버전
- **Python**: 3.12.3
- **transformers**: 4.53.1
- **sentence-transformers**: 5.0.0
- **llama-cpp-python**: 0.3.12
- **scikit-learn**: 1.7.0
- **pandas**: 2.3.0

## 🔧 환경 분리 전략

### venv (pip)
- 일반 Python 패키지
- ML/AI 프레임워크 (PyTorch, Transformers)
- NLP 도구 (konlpy, nltk)
- 시각화 (matplotlib, seaborn)

### conda
- FAISS GPU (CUDA 11.4.4)
- CUDA 런타임 라이브러리
- MKL 최적화
- 시스템 레벨 의존성

## 🔄 버전 충돌 해결

| 패키지 | venv | conda | 사용 버전 |
|--------|------|-------|-----------|
| h11 | 0.16.0 | 0.9.0 | venv (0.16.0) |
| httpx | 0.28.1 | 0.13.3 | venv (0.28.1) |
| PyYAML | 5.1.2 | 6.0.2 | 각 환경 독립 |
| setuptools | 70.2.0 | 80.9.0 | 각 환경 독립 |

## 📝 이전 버전 (previous/)

- **2025-08-06**: requirements_unified.txt, requirements_venv.txt
- **2025-07-15**: requirements_conda.txt, requirements_system.txt
- **2025-07-04**: requirements_advanced.txt, requirements_verified.txt
- **2025-06-29**: requirements_minimal.txt

## 🗄️ 아카이브 (archive/)

2025-07-04 이전의 오래된 백업 파일들:
- 초기 requirements.txt
- requirements_advanced.txt
- requirements_minimal.txt
- requirements_unified.txt

## 🆘 문제 해결

### 환경 복구가 필요한 경우
```bash
# 전체 복구 가이드 참조
cat requirements/current/ENVIRONMENT_RECOVERY_GUIDE.md
```

### 특정 버전으로 롤백
```bash
# 이전 버전 확인
ls requirements/previous/

# 특정 버전으로 복구
pip install -r requirements/previous/requirements_venv.txt
```

## 📅 업데이트 이력

| 날짜 | 변경사항 |
|------|----------|
| 2025-08-18 | 전체 재구성, venv/conda 분리 관리 |
| 2025-08-06 | unified 버전 생성 |
| 2025-07-15 | conda/system 분리 |
| 2025-07-04 | advanced/verified 버전 |
| 2025-06-29 | 최초 minimal 버전 |

---
*마지막 정리: 2025-08-18 18:50*
*관리자: Red Heart AI Team*