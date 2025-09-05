# Red Heart AI 통합 시스템 사용 설명서

## 📋 개요
Red Heart AI 통합 시스템은 50 epoch으로 학습된 730M 파라미터 모델을 활용하는 추론 시스템입니다.

## 🎯 주요 변경 사항

### 기존 시스템 (main.py) → 새로운 시스템 (main_unified.py)

#### 문제점
- **500.6M 파라미터 (68.6%)가 완전히 미사용**
- 구버전 모듈 사용 (AdvancedEmotionAnalyzer 등)
- 학습된 체크포인트 미활용

#### 해결
- **730M 전체 파라미터 활용**
- UnifiedModel 기반 통합 아키텍처
- 50 epoch 체크포인트 자동 로드

## 🏗️ 시스템 구성

### 핵심 컴포넌트 (730M)
```
1. UnifiedModel (243.6M)
   - Backbone: 90.6M
   - Emotion Head: 38.3M
   - Bentham Head: 38.3M
   - Regret Head: 38.3M ✨ NEW
   - SURD Head: 38.3M

2. Neural Analyzers (368M) ✨ NEW
   - NeuralEmotionAnalyzer: 102M
   - NeuralBenthamCalculator: 120M
   - NeuralRegretAnalyzer: 111M
   - NeuralSURDAnalyzer: 35M

3. Advanced Wrappers (112M) ✨ NEW
   - EmotionAnalyzerWrapper: 48M
   - BenthamCalculatorWrapper: 20M
   - SemanticAnalyzerWrapper: 20M
   - SURDAnalyzerWrapper: 24M

4. DSP Components (16.3M) ✨ NEW
   - EmotionDSPSimulator: 14M
   - DynamicKalmanFilter: 2.3M

5. Phase Networks (4.3M) ✨ NEW
   - Phase0ProjectionNet
   - Phase2CommunityNet
   - HierarchicalEmotionIntegrator
```

## 🚀 빠른 시작

### 1. 기본 추론
```bash
# 간단한 텍스트 분석
./run_inference.sh inference --text "분석할 텍스트"

# Python 직접 실행
python main_unified.py --text "분석할 텍스트"
```

### 2. 대화형 데모
```bash
./run_inference.sh demo
```

### 3. 시스템 테스트
```bash
# 간단한 테스트
python test_unified_system.py

# 전체 테스트
./run_inference.sh test --verbose
```

### 4. 운용 모드
```bash
# 완전한 운용 모드
./run_inference.sh production --text "운용 텍스트"

# 경량 모드 (빠른 추론)
./run_inference.sh light --text "빠른 분석"
```

## ⚙️ 고급 사용법

### 모듈 선택적 활성화
```bash
# Neural Analyzers 없이 실행
python main_unified.py --no-neural --text "텍스트"

# DSP 시뮬레이터 없이 실행
python main_unified.py --no-dsp --text "텍스트"

# 최소 모드 (UnifiedModel만)
python main_unified.py \
    --no-neural \
    --no-wrappers \
    --no-dsp \
    --no-phase \
    --text "텍스트"
```

### LLM 통합
```bash
# 로컬 LLM (HelpingAI 9B)
./run_inference.sh llm-local --text "텍스트"

# Claude API
./run_inference.sh llm-claude --text "텍스트"
```

### 체크포인트 선택
```bash
# 특정 체크포인트 사용
python main_unified.py \
    --checkpoint training/checkpoints_final/checkpoint_epoch_0050_lr_0.000009_20250827_154403.pt \
    --text "텍스트"
```

## 📊 성능 비교

| 항목 | 기존 (main.py) | 신규 (main_unified.py) |
|------|---------------|----------------------|
| 사용 파라미터 | 230M (31.4%) | 730M (100%) |
| Neural Analyzers | ❌ | ✅ (368M) |
| Advanced Wrappers | ❌ | ✅ (112M) |
| DSP/Kalman | ❌ | ✅ (16.3M) |
| Regret Circuit | ❌ | ✅ |
| 체크포인트 활용 | ❌ | ✅ (50 epoch) |

## 🔧 문제 해결

### NumPy 없음 오류
```bash
pip install numpy
```

### CUDA 메모리 부족
```bash
# CPU 모드로 실행
python main_unified.py --device cpu --text "텍스트"

# 경량 모드로 실행
./run_inference.sh light --text "텍스트"
```

### 체크포인트 로드 실패
```bash
# 가장 최신 체크포인트 자동 검색
python main_unified.py --text "텍스트"
```

## 📁 주요 파일

- `main_unified.py`: 통합 추론 시스템 (신규)
- `run_inference.sh`: 운용 스크립트 (신규)
- `test_unified_system.py`: 시스템 테스트 (신규)
- `main.py`: 기존 시스템 (구버전)
- `training/checkpoints_final/`: 학습된 체크포인트

## 🎯 권장 사용 시나리오

1. **완전한 분석이 필요한 경우**
   ```bash
   ./run_inference.sh production --text "중요한 윤리적 결정"
   ```

2. **빠른 응답이 필요한 경우**
   ```bash
   ./run_inference.sh light --text "간단한 질문"
   ```

3. **LLM 보강이 필요한 경우**
   ```bash
   ./run_inference.sh llm-local --text "복잡한 상황"
   ```

4. **테스트/개발 중**
   ```bash
   ./run_inference.sh demo
   ```

## 📈 메트릭

학습 완료 상태 (50 epoch):
- 최종 Loss: 0.1268
- DSP Loss: 0.968 (37.5% 감소)
- DSP Accuracy: 99.9%
- Kalman Loss: 0.582 (6.4% 감소)
- Kalman Accuracy: 100%

## 🚧 향후 계획

1. MCP (Model Context Protocol) 완전 구현
2. 웹 인터페이스 추가
3. REST API 서버 구현
4. 더 많은 LLM 모델 지원
5. 실시간 스트리밍 분석

## 📞 지원

문제가 발생하면:
1. `test_unified_system.py` 실행하여 상태 확인
2. `./run_inference.sh monitor`로 시스템 모니터링
3. 로그 파일 확인: `logs/`

---
작성일: 2025-08-28
버전: 1.0 (730M 통합 시스템)