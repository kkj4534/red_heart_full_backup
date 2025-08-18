# Red Heart AI run_learning.sh 스크립트 정리

## 📋 개요
Red Heart AI 통합 학습 시스템의 진입점 스크립트로, 다양한 실행 모드와 환경 설정을 자동으로 관리합니다.

## 🔧 환경 설정
- **듀얼 환경 시스템**: conda (faiss) + venv (transformers, torch)
- **자동 환경 검증**: 필수 패키지 확인 및 자동 설치
- **스마트 활성화**: 환경 상태 체크 후 필요시에만 설정

## 📊 실행 모드 및 인자

### 1. 학습 모드 (Training)

#### `train` | `training`
- **설명**: 완전한 학습 기능 (NO FALLBACK)
- **특징**: DSP, 칼만필터 등 모든 모듈 온전히 학습
- **실행**: `python unified_training_v2.py --mode train [추가인자]`
- **용도**: 정식 학습

#### `train-local` | `train-test` 
- **설명**: 로컬 학습 테스트 모드
- **특징**: 
  - 소규모 샘플로 학습 가능성 검증
  - GPU 메모리 사용량 모니터링
- **실행**: `python unified_training_v2.py --mode train --max-samples ${SAMPLES:-3} --debug --verbose [추가인자]`
- **용도**: 개발/디버깅

#### `train-cloud` | `train-full`
- **설명**: 클라우드 학습 모드
- **특징**:
  - 전체 데이터셋 학습
  - 체크포인트 자동 저장
- **실행**: `python unified_training_v2.py --mode train --full-dataset --checkpoint-interval 1000 [추가인자]`
- **용도**: 프로덕션 학습

#### `train-validate`
- **설명**: 학습 검증 모드
- **특징**: 학습된 모델 성능 평가
- **실행**: `python unified_training_v2.py --mode eval --load-checkpoint [추가인자]`
- **용도**: 모델 평가

### 2. 메인 시스템 모드

#### `main` | `advanced`
- **설명**: Red Heart AI 메인 시스템 (main.py)
- **특징**:
  - 모든 고급 AI 모듈 통합
  - module_bridge_coordinator 활용
  - XAI, 베이지안, 반사실적 추론 등 전체 기능
- **실행**: `python main.py --mode advanced [추가인자]`
- **용도**: 전체 기능 통합 실행

### 3. 운용 모드 (Production)

#### `production` | `prod`
- **설명**: 운용 모드 - main.py 전체 시스템
- **특징**:
  - 모든 고급 분석 모듈 통합
  - XAI, 시계열, 베이지안 등 전체 기능
- **실행**: `python main.py --mode production [추가인자]`
- **용도**: 프로덕션 운영

#### `production-advanced` | `prod-adv`
- **설명**: 고급 운용 모드
- **특징**:
  - main.py + 추가 고급 모듈
  - XAI 피드백, 시계열 전파, 베이지안 추론
- **실행**: `python main.py --mode advanced --enable-xai --enable-temporal --enable-bayesian [추가인자]`
- **용도**: 고급 분석 운영

#### `production-oss` | `prod-oss`
- **설명**: OSS 20B 통합 운용 모드
- **특징**: OSS 모델과 연동 분석
- **실행**: `python main.py --mode production --oss-integration [추가인자]`
- **용도**: 대규모 언어 모델 통합

### 4. 고급 AI 분석 모드

#### `xai` | `explain`
- **설명**: XAI 설명 가능 AI 모드
- **특징**: 의사결정 투명성 제공
- **실행**: `python main.py --mode xai [추가인자]`
- **용도**: 의사결정 설명

#### `temporal` | `time-series`
- **설명**: 시계열 사건 전파 분석
- **특징**: 장기적 영향 예측
- **실행**: `python main.py --mode temporal [추가인자]`
- **용도**: 시간축 분석

#### `bayesian`
- **설명**: 베이지안 추론 모드
- **특징**: 불확실성 정량화
- **실행**: `python main.py --mode bayesian [추가인자]`
- **용도**: 확률적 추론

#### `counterfactual` | `what-if`
- **설명**: 반사실적 추론 모드
- **특징**: '만약' 시나리오 분석
- **실행**: `python main.py --mode counterfactual [추가인자]`
- **용도**: 대안 시나리오 분석

#### `ethics` | `multi-ethics`
- **설명**: 다차원 윤리 시스템
- **특징**: 복합적 윤리 판단
- **실행**: `python main.py --mode ethics [추가인자]`
- **용도**: 윤리적 판단

### 5. MCP 준비 모드

#### `mcp-prepare` | `mcp-init`
- **설명**: MCP 서비스 준비 모드
- **특징**:
  - API 엔드포인트 초기화
  - 인터페이스 스켈레톤 생성
- **실행**: `python mcp_service_init.py [추가인자]`
- **용도**: MCP 서비스 초기화

### 6. 통합 시스템 모드 (Unified)

#### `unified` | `800m` | `v2`
- **설명**: 320M v2 통합 시스템 실행
- **특징**:
  - 104M 공유 백본 + 174M 전문 헤드 + 40M 전문모듈
  - LLM 전처리 + 3단계 워크플로우
  - Gate 9 최적화 버전
- **실행**: `python unified_training_v2.py --mode train [추가인자]`
- **용도**: v2 시스템 실행

#### `unified-train`
- **설명**: 320M v2 통합 시스템 훈련 모드
- **실행**: `python unified_training_v2.py --mode train [추가인자]`
- **용도**: v2 학습

#### `unified-test` ⭐ **[권장 테스트 모드]**
- **설명**: 320M v2 통합 시스템 학습 테스트 모드
- **특징**:
  - **그래디언트 체크 (NaN, 끊김 검증)**
  - **텐서 계산 무결성 확인**
  - **파라미터 업데이트 없음 (베이스라인 회귀 방지)**
- **실행**: `python unified_training_v2.py --mode train-test --max-samples ${SAMPLES:-3} --no-param-update --debug --verbose [추가인자]`
- **용도**: 안전한 테스트 및 검증
- **예시**: 
  ```bash
  timeout 1200 bash run_learning.sh unified-test --samples 3 --debug --verbose > test_gate9_$(date +%Y%m%d_%H%M%S).txt 2>&1 &
  ```

#### `unified-test-v1` | `unified-test-800m`
- **설명**: 기존 800M 통합 시스템 테스트 모드
- **실행**: `python unified_system_main.py --mode test [추가인자]`
- **용도**: 레거시 시스템 테스트

#### `unified-monitor`
- **설명**: 학습 모니터링 모드
- **특징**: 실시간 학습 상태 추적
- **용도**: 학습 진행 상황 모니터링

### 7. 특수 모드

#### `auto`
- **설명**: 자동 모드 선택
- **특징**: 환경에 따라 최적 모드 자동 선택
- **용도**: 기본 실행

#### `interactive`
- **설명**: 대화형 모드
- **특징**: 실시간 상호작용
- **용도**: 인터랙티브 테스트

## 🎯 공통 인자

### 기본 인자
- `--samples N`: 학습/테스트 샘플 수 (기본값: 3)
- `--epochs N`: 학습 에포크 수
- `--batch-size N`: 배치 크기
- `--learning-rate F`: 학습률

### 디버그 인자
- `--debug`: 디버그 모드 활성화
- `--verbose`: 상세 로그 출력
- `--log-level LEVEL`: 로그 레벨 설정 (DEBUG/INFO/WARNING/ERROR)

### 특수 인자
- `--no-param-update`: 파라미터 업데이트 비활성화 (테스트용)
- `--checkpoint-interval N`: 체크포인트 저장 간격
- `--load-checkpoint PATH`: 체크포인트 로드
- `--full-dataset`: 전체 데이터셋 사용
- `--max-samples N`: 최대 샘플 수 제한

## 💡 사용 예시

### 1. 빠른 테스트 (파라미터 업데이트 없이)
```bash
bash run_learning.sh unified-test --samples 3 --debug
```

### 2. 로컬 학습 테스트
```bash
bash run_learning.sh train-local --samples 5 --epochs 2 --verbose
```

### 3. 프로덕션 운영
```bash
bash run_learning.sh production --enable-xai --enable-temporal
```

### 4. XAI 분석 모드
```bash
bash run_learning.sh xai --input "윤리적 판단이 필요한 상황"
```

### 5. 백그라운드 테스트 실행
```bash
timeout 1200 bash run_learning.sh unified-test --samples 3 --debug --verbose > test_$(date +%Y%m%d_%H%M%S).txt 2>&1 &
```

## 📌 주의사항

1. **unified-test 모드 사용 권장**: 테스트 시 `--no-param-update` 플래그로 파라미터 업데이트를 방지하여 안전한 테스트 가능
2. **환경 자동 설정**: 첫 실행 시 필요한 환경이 자동으로 설정됨
3. **타임아웃 설정**: 장시간 실행 시 `timeout` 명령어 사용 권장
4. **로그 저장**: 중요한 실행은 로그 파일로 리다이렉션 권장

## 🔄 환경 변수

- `SAMPLES`: 기본 샘플 수 (기본값: 3)
- `PYTHONPATH`: Python 경로 자동 설정
- `CUDA_VISIBLE_DEVICES`: GPU 선택 (자동 감지)

## 📈 모니터링

학습 진행 상황은 다음과 같이 모니터링 가능:
```bash
# 실시간 로그 확인
tail -f test_gate9_*.txt

# GPU 사용량 확인
nvidia-smi -l 1

# 프로세스 확인
ps aux | grep python
```