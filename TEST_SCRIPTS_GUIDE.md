# 🧪 Red Heart AI 테스트 스크립트 가이드

## 📊 구현 완성도 현황
- **MD 문서 요구사항 구현**: 100% 완료 ✅
- **전체 시스템 파라미터**: 약 970M
- **최적 체크포인트**: Epoch 50 (Sweet Spot 분석 결과)

## 🚀 주요 테스트 스크립트

### 1. quick_test_md_requirements.py
**목적**: MD 문서 요구사항 구현 상태 검증

```bash
# 가상환경 활성화 후 실행
source red_heart_env/bin/activate
python3 quick_test_md_requirements.py
```

**검증 항목** (12/12 완료):
1. ✅ 메모리 모드 4개 (LIGHT/MEDIUM/HEAVY/MCP)
2. ✅ 3뷰 시나리오 시스템 (낙관/중도/비관)
3. ✅ 5개 윤리 엔진 (공리주의/의무론/덕윤리/돌봄윤리/정의론)
4. ✅ 비선형 워크플로우 (analyze_ethical_dilemma)
5. ✅ 메모리 스왑 매니저 (LLM ↔ Red Heart)
6. ✅ LLM 통합 (AdvancedLLMEngine)
7. ✅ 후회 시스템 대안 생성 (suggest_alternatives)
8. ✅ 경험 DB 저장 (store_experience)
9. ✅ MCP 서버 (RedHeartMCPServer)
10. ✅ 시간적 전파 분석 (TemporalEventPropagationAnalyzer)
11. ✅ MEDIUM 모드 600M 재설계
12. ✅ 정합성 판단 (_calculate_plausibility)

---

### 2. simple_inference_test.py
**목적**: 빠른 추론 테스트

```bash
# 기본 테스트 (LIGHT 모드만)
python3 simple_inference_test.py

# MEDIUM 모드 추가
python3 simple_inference_test.py --medium

# HEAVY 모드 추가
python3 simple_inference_test.py --heavy

# 윤리적 딜레마 분석
python3 simple_inference_test.py --dilemma

# 모든 테스트 실행
python3 simple_inference_test.py --all

# 디버그 모드
python3 simple_inference_test.py --debug
```

**테스트 내용**:
- 각 모드별 초기화 시간
- GPU 메모리 사용량
- 활성 모듈 확인
- 추론 시간 및 결과
- 윤리적 딜레마 분석 (HEAVY 모드)

---

### 3. test_modes.sh
**목적**: 모드별 상세 테스트 (bash 스크립트)

```bash
# 실행 권한 부여
chmod +x test_modes.sh

# 모든 모드 테스트
./test_modes.sh

# 특정 모드만 테스트
./test_modes.sh light
./test_modes.sh medium
./test_modes.sh heavy
./test_modes.sh mcp
```

**특징**:
- 자동 가상환경 활성화
- GPU 정보 확인
- 각 모드별 활성 모듈 상태
- 추론 및 딜레마 분석 테스트

---

## 📦 메모리 모드 사양

### LIGHT 모드 (230M)
- **용도**: 빠른 프로토타이핑
- **활성 모듈**: 기본 모듈만
- **초기화 시간**: ~15초
- **GPU 메모리**: ~2.5GB

### MEDIUM 모드 (600M) - MD 문서 재설계
- **용도**: 균형잡힌 일반 사용
- **활성 모듈**:
  - DSP Simulator ✅
  - Kalman Filter ✅
  - Phase Networks ✅
  - 3-View Scenario ✅
  - Multi-Ethics System ✅
- **초기화 시간**: ~30초
- **GPU 메모리**: ~4GB

### HEAVY 모드 (970M)
- **용도**: 심층 분석 (동적 스왑)
- **활성 모듈**: 전체 모듈
- **특별 기능**:
  - Neural Analyzers (368M)
  - Advanced Wrappers (112M)
  - Meta Integration (40M)
  - 모든 윤리 시스템
  - 비선형 워크플로우
- **초기화 시간**: ~2분
- **GPU 메모리**: ~7GB (스왑 활용)

### MCP 모드
- **용도**: Claude 통합 서버
- **기반**: HEAVY 모드 + MCP 서버
- **특별 기능**: 외부 API 제공

---

## 🔧 main_unified.py 실행 옵션

### 체크포인트 옵션 (개선됨)
```bash
# 기본 (epoch 50 자동 검색)
python3 main_unified.py --text "테스트 텍스트"

# 특정 에폭 지정
python3 main_unified.py --epoch 40 --text "테스트"

# 직접 경로 지정 (우선순위 높음)
python3 main_unified.py --checkpoint path/to/checkpoint.pt --text "테스트"
```

### 메모리 모드 옵션
```bash
# 자동 모드 선택 (기본)
python3 main_unified.py --text "테스트"

# 특정 모드 강제
python3 main_unified.py --memory-mode heavy --text "테스트"
```

### 모듈 비활성화 옵션
```bash
# Neural Analyzers 비활성화 (메모리 절약)
python3 main_unified.py --no-neural --text "테스트"

# 여러 모듈 비활성화
python3 main_unified.py --no-neural --no-wrappers --text "테스트"
```

### LLM 통합 옵션
```bash
# LLM 없이 (기본)
python3 main_unified.py --text "테스트"

# 로컬 LLM 사용
python3 main_unified.py --llm local --text "테스트"

# MCP 서버 모드
python3 main_unified.py --llm mcp
```

---

## 💾 체크포인트 관리

### 현재 상황
- **디스크 사용량**: 약 180GB (30개 체크포인트 × 6GB)
- **실제 로딩**: 6GB (epoch 50 체크포인트만)
- **최적 에폭**: 50 (sweet spot 분석 결과)

### 개선 사항
```python
# 이전: 하드코딩된 경로
checkpoint_path = "training/checkpoints_final/checkpoint_epoch_0050_lr_0.000009_20250827_154403.pt"

# 현재: 에폭 번호로 자동 검색
checkpoint_epoch = 50  # 변경 가능
# 자동으로 checkpoint_epoch_0050_*.pt 검색
```

### 디스크 공간 절약 방법
```bash
# 불필요한 체크포인트 제거 (50번 제외)
cd training/checkpoints_final
mkdir backup
mv checkpoint_epoch_00[2-4]*.pt backup/  # 20-49 백업
# 필요시 rm -rf backup/
```

---

## 🛠️ 문제 해결

### ImportError: No module named 'numpy'
```bash
# 가상환경 활성화 필수
source red_heart_env/bin/activate
```

### CUDA out of memory
```bash
# LIGHT 모드로 시작
python3 simple_inference_test.py

# 또는 모듈 비활성화
python3 main_unified.py --no-neural --no-wrappers --text "테스트"
```

### 체크포인트 로딩 실패
```bash
# 최신 체크포인트 자동 사용
python3 main_unified.py --epoch 0 --text "테스트"
# epoch 0은 최신 체크포인트를 자동 선택
```

---

## 📈 성능 메트릭

### 초기화 시간
- LIGHT: ~15초
- MEDIUM: ~30초
- HEAVY: ~2분 (체크포인트 6GB 로딩 포함)

### GPU 메모리 사용량
- LIGHT: 2.5GB
- MEDIUM: 4GB
- HEAVY: 7GB (동적 스왑 활용)

### 추론 속도
- LIGHT: ~0.5초/쿼리
- MEDIUM: ~1초/쿼리
- HEAVY: ~2초/쿼리

---

## 🔍 검증 완료 사항

✅ **MD 문서 요구사항**: 100% 구현 완료
✅ **메모리 모드**: 4개 모드 정상 작동
✅ **윤리 시스템**: 5개 엔진 통합
✅ **비선형 워크플로우**: 구현 및 테스트
✅ **LLM 통합**: 메모리 스왑 포함
✅ **체크포인트 최적화**: 에폭 기반 자동 검색

---

## 📞 지원

문제 발생 시:
1. 가상환경 활성화 확인
2. GPU 메모리 확인 (`nvidia-smi`)
3. 디버그 모드 실행 (`--debug`)
4. 로그 파일 확인 (`logs/unified_system.log`)

작성일: 2025-08-30
버전: 1.0