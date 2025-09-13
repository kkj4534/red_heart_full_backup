# 🚀 Red Heart AI 통합 시스템 명령어 가이드

## 📦 환경 설정

### 1. 가상환경 활성화
```bash
source red_heart_env/bin/activate
```

### 2. 필수 패키지 설치 (수동)
```bash
pip install numpy torch transformers sentence-transformers
pip install matplotlib seaborn pandas jinja2 markdown
```

## 🎮 실행 명령어 (V2.0 개편)

### 🔥 핵심 실행 명령어 (개편된 아키텍처)

#### I/O Pipeline + WorkflowDSM + Claude API (권장)
```bash
source red_heart_env/bin/activate && \
python3 main_unified.py \
  --memory-mode medium \
  --mode inference \
  --text "AI 윤리적 문제 해결" \
  --epochs 50 \
  --use-io-pipeline \
  --use-workflow-dsm \
  --llm claude \
  --debug \
  > mainunified_claude_workflow_$(date +%Y%m%d_%H%M%S).txt 2>&1 &
```

#### Local LLM + 자동 번역기
```bash
source red_heart_env/bin/activate && \
python3 main_unified.py \
  --memory-mode medium \
  --mode inference \
  --text "복잡한 철학적 질문" \
  --epochs 50 \
  --use-io-pipeline \
  --use-workflow-dsm \
  --llm local \
  --debug \
  > mainunified_local_workflow_$(date +%Y%m%d_%H%M%S).txt 2>&1 &
```

### 기존 스크립트 사용 (호환성 유지)

#### 기본 추론
```bash
./run_unified.sh inference --text "분석할 텍스트"
```

#### 대화형 모드
```bash
./run_unified.sh interactive
```

#### 운용 모드 (모든 기능)
```bash
./run_unified.sh production --text "텍스트"
```

#### 벤치마크
```bash
./run_unified.sh benchmark --samples 100
```

#### 정밀 매핑 테스트
```bash
./run_unified.sh mapping
```

#### 유휴 학습 실행
```bash
./run_unified.sh idle
```

### 직접 Python 실행

#### 가상환경 활성화 후 실행
```bash
source red_heart_env/bin/activate && python main_unified.py --text "분석할 텍스트"
```

#### 대화형 모드
```bash
source red_heart_env/bin/activate && python main_unified.py --mode interactive
```

#### 운용 모드
```bash
source red_heart_env/bin/activate && python main_unified.py --mode production --text "텍스트"
```

## 🎚️ 메모리 모드별 실행

### MINIMAL (최소 - 2GB VRAM)
```bash
./run_unified.sh minimal --text "텍스트"
```

### LIGHT (경량 - 4GB VRAM)
```bash
./run_unified.sh light --text "텍스트"
```

### NORMAL (일반 - 6GB VRAM)
```bash
./run_unified.sh normal --text "텍스트"
```

### HEAVY (고성능 - 8GB VRAM)
```bash
./run_unified.sh heavy --text "텍스트"
```

### ULTRA (울트라 - 10GB VRAM)
```bash
./run_unified.sh ultra --text "텍스트"
```

### EXTREME (익스트림 - 12GB+ VRAM)
```bash
./run_unified.sh extreme --text "텍스트"
```

## 🧪 테스트 및 검증

### 통합 테스트
```bash
./run_unified.sh test
```

### 정밀 매핑 테스트
```bash
python test_emotion_bentham_mapping.py
```

### 최종 통합 테스트
```bash
python test_final_integration.py
```

### 시스템 상태 확인
```bash
./run_unified.sh status
```

## 📊 벤치마크

### 기본 벤치마크
```bash
./run_unified.sh benchmark
```

### 상세 벤치마크 (100개 샘플)
```bash
python benchmark_unified.py --samples 100 --memory-mode normal --plot
```

### 메모리 모드 비교
```bash
python benchmark_unified.py --compare-modes --samples 50
```

## 🤖 LLM 통합

### 로컬 LLM 사용
```bash
./run_unified.sh llm-local --text "분석할 텍스트"
```

### Claude API 사용
```bash
./run_unified.sh llm-claude --text "분석할 텍스트"
```

## 🌙 유휴 학습

### 유휴 학습 시작
```bash
./run_unified.sh idle
```

### 백그라운드 실행
```bash
nohup ./run_unified.sh idle > idle_learning.log 2>&1 &
```

## 🔧 고급 옵션

### 특정 체크포인트 사용
```bash
python main_unified.py --checkpoint training/checkpoints_final/best_model.pt --text "텍스트"
```

### 모듈 선택적 비활성화
```bash
python main_unified.py --no-neural --no-wrappers --text "빠른 분석"
```

### 상세 로그 출력
```bash
python main_unified.py --text "텍스트" --verbose --debug
```

### 배치 처리
```bash
python main_unified.py --batch-size 8 --text "텍스트1" "텍스트2" "텍스트3"
```

## 📝 주요 Python 명령어

### main_unified.py 직접 실행
```python
from main_unified import UnifiedInferenceSystem, InferenceConfig
import asyncio

async def run():
    config = InferenceConfig()
    config.memory_mode = MemoryMode.NORMAL
    
    system = UnifiedInferenceSystem(config)
    await system.initialize()
    
    result = await system.analyze("분석할 텍스트")
    print(result)
    
    await system.cleanup()

asyncio.run(run())
```

### 정밀 매퍼 단독 사용
```python
from semantic_emotion_bentham_mapper import SemanticEmotionBenthamMapper

mapper = SemanticEmotionBenthamMapper()
emotion = {
    'valence': 0.7,
    'arousal': 0.6,
    'dominance': 0.5,
    'certainty': 0.8,
    'surprise': 0.2,
    'anticipation': 0.7
}
bentham = mapper.map_emotion_to_bentham(emotion)
print(bentham)
```

## 🚨 문제 해결

### NumPy 오류 시
```bash
pip install numpy==1.26.0
```

### CUDA 오류 시
```bash
export CUDA_VISIBLE_DEVICES=0
python main_unified.py --device cpu --text "텍스트"
```

### 메모리 부족 시
```bash
./run_unified.sh minimal --text "텍스트"
```

## 📈 모니터링

### GPU 사용량 확인
```bash
watch -n 1 nvidia-smi
```

### 실시간 로그 확인
```bash
python main_unified.py --text "텍스트" --verbose 2>&1 | tee output.log
```

### 프로세스 확인
```bash
ps aux | grep -E "(main_unified|idle_learner)"
```

## 🎯 권장 실행 순서

1. **환경 활성화**
   ```bash
   source red_heart_env/bin/activate
   ```

2. **시스템 상태 확인**
   ```bash
   ./run_unified.sh status
   ```

3. **간단한 테스트**
   ```bash
   ./run_unified.sh test
   ```

4. **실제 사용**
   ```bash
   ./run_unified.sh production --text "분석할 텍스트" --verbose
   ```

## 📌 빠른 시작

### 가장 간단한 실행
```bash
./run_unified.sh inference --text "오늘은 정말 행복한 날이야!"
```

### 대화형 분석
```bash
./run_unified.sh interactive
```

### 성능 테스트
```bash
./run_unified.sh benchmark --samples 10
```

---

**최종 업데이트**: 2025-08-29
**버전**: 2.0 (정밀 매핑 + 유휴 학습 통합)