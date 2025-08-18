# Red Heart AI 의존성 현황

**작성일자**: 2025-08-18 02:00 KST  
**환경 구성**: venv + conda 분리 운영

---

## 1. venv 환경 (`red_heart_env`)
**경로**: `/mnt/c/large_project/linux_red_heart/red_heart_env`  
**Python**: 3.12  
**용도**: 메인 실행 환경 (transformers, torch, 기타 ML 라이브러리)

### 핵심 패키지
| 패키지 | 설치 버전 | requirements 요구사항 | 상태 |
|--------|-----------|---------------------|------|
| torch | 2.5.1+cu121 | >=2.0.0,<3.0.0 | ✅ |
| transformers | 4.53.1 | >=4.30.0,<5.0.0 | ✅ |
| numpy | 1.26.4 | >=1.24.0,<2.0.0 | ✅ |
| sentence-transformers | 5.0.0 | >=2.2.0 | ✅ |
| rapidfuzz | 3.13.0 | (미지정) | ✅ 설치됨 |
| scipy | 1.16.0 | >=1.10.0 | ✅ |
| pandas | 2.3.1 | >=2.0.0 | ✅ |
| thinc | 8.3.4 | (자동 설치) | ✅ |

### 특이사항
- `rapidfuzz`가 requirements_venv.txt에 누락되어 있었으나 이미 설치됨
- NumPy 1.x 버전으로 고정 (FAISS 호환성)
- thinc 8.3.4는 NumPy 1.x와 호환

---

## 2. conda 환경 (`faiss-test`)
**경로**: `/home/kkj/miniconda3/envs/faiss-test`  
**Python**: 3.12  
**용도**: FAISS 벡터 DB 전용 (subprocess 실행)

### 핵심 패키지
| 패키지 | 설치 버전 | requirements 요구사항 | 상태 |
|--------|-----------|---------------------|------|
| faiss | 1.11.0 | faiss-cpu>=1.11.0 | ✅ |
| numpy | 1.26.4 | >=1.24.0,<2.0.0 | ✅ 수정됨 |
| scipy | 1.16.0 | >=1.10.0 | ✅ |
| spacy | 3.8.7 | >=3.4.0 | ✅ |
| thinc | 8.3.4 | (spacy 의존성) | ✅ 다운그레이드됨 |
| blis | 1.2.1 | (thinc 의존성) | ✅ 다운그레이드됨 |

### 수정 내역
1. **NumPy 2.3.2 → 1.26.4**: FAISS 호환성 문제 해결
2. **thinc 8.3.6 → 8.3.4**: NumPy 1.x 호환성 확보
3. **blis 1.3.0 → 1.2.1**: thinc 의존성 맞춤

---

## 3. 환경 분리 전략

### 실행 방식
- **venv**: 메인 Python 프로세스 실행
- **conda**: FAISS 관련 작업시 subprocess로 격리 실행
- **목적**: 의존성 충돌 방지, 메모리 관리 최적화

### FAISS 사용 현황
- **현재**: 코드에서 직접 사용하는 곳 없음
- **과거**: GPU 메모리 관리에 사용 (백업 파일에서 확인)
- **run_learning.sh**: 환경 체크시에만 import 테스트

---

## 4. 의존성 문제 해결 기록

### 해결된 문제
- ✅ conda NumPy 버전 충돌 (2.x → 1.x 다운그레이드)
- ✅ thinc-NumPy 호환성 (8.3.6 → 8.3.4)
- ✅ rapidfuzz venv 설치 확인

### 권장 사항
1. `requirements_venv.txt`에 `rapidfuzz>=3.13.0` 추가 필요
2. FAISS가 실제로 사용되지 않으면 conda 환경 체크 제거 고려
3. 모든 패키지를 NumPy 1.x 기반으로 통일 유지

---

## 5. 테스트 명령어

### venv 환경 테스트
```bash
source red_heart_env/bin/activate
python -c "import torch, transformers, numpy, rapidfuzz; print('venv OK')"
```

### conda 환경 테스트  
```bash
conda run -n faiss-test python -c "import faiss, numpy, spacy; print('conda OK')"
```

### 통합 환경 테스트
```bash
./run_learning.sh validate
```