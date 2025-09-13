# 🚀 Red Heart AI 통합 시스템 명령어 가이드 V2.0

## 📌 개편된 아키텍처 핵심 명령어

### 🎯 메인 추론 명령어 (권장)

#### 1. I/O Pipeline + WorkflowDSM + Claude API
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

#### 2. Local LLM + 자동 번역기 + I/O Pipeline
```bash
source red_heart_env/bin/activate && \
python3 main_unified.py \
  --memory-mode medium \
  --mode inference \
  --text "복잡한 철학적 질문에 대한 분석" \
  --epochs 50 \
  --use-io-pipeline \
  --use-workflow-dsm \
  --llm local \
  --local-model-path "models/llama-3.2-ko.gguf" \
  --debug \
  > mainunified_local_workflow_$(date +%Y%m%d_%H%M%S).txt 2>&1 &
```

#### 3. GPT API + 통합 임베딩
```bash
source red_heart_env/bin/activate && \
python3 main_unified.py \
  --memory-mode medium \
  --mode inference \
  --text "인공지능의 창의성과 인간 예술" \
  --epochs 50 \
  --use-io-pipeline \
  --use-workflow-dsm \
  --llm gpt \
  --gpt-model "gpt-4" \
  --enriched-embedding \
  --debug \
  > mainunified_gpt_workflow_$(date +%Y%m%d_%H%M%S).txt 2>&1 &
```

### 🔧 상세 파라미터 설명

#### 필수 파라미터
- `--memory-mode`: 메모리 프로파일 선택
  - `minimal`: 2GB VRAM (기본 기능만)
  - `light`: 4GB VRAM (경량 모델)
  - `medium`: 6GB VRAM (균형잡힌 성능) **[권장]**
  - `heavy`: 8GB VRAM (고성능)
  - `ultra`: 10GB VRAM (울트라)
  - `extreme`: 12GB+ VRAM (최대 성능)

- `--mode`: 실행 모드
  - `inference`: 추론 모드 (단일 텍스트 분석)
  - `interactive`: 대화형 모드
  - `production`: 운용 모드 (모든 기능)
  - `test`: 테스트 모드
  - `benchmark`: 벤치마크 모드

- `--epochs`: 데이터셋 학습 에폭 수 (기본값: 10)
  - 높을수록 정밀도 향상, 처리 시간 증가

#### 개편된 시스템 전용 파라미터
- `--use-io-pipeline`: I/O Pipeline 활성화 (비동기 처리)
- `--use-workflow-dsm`: WorkflowDSM 활성화 (2레벨 메모리 관리)
- `--enriched-embedding`: LLM 분석 결과 통합 임베딩
- `--llm`: LLM 백엔드 선택
  - `local`: 로컬 GGUF 모델 (자동 번역기 활성화)
  - `claude`: Claude API
  - `gpt`: OpenAI GPT API
  - `perplexity`: Perplexity API
  - `mcp`: MCP 프로토콜

### 📊 전체 워크플로우 테스트

#### 1. 완전 통합 테스트 (모든 기능 활성화)
```bash
source red_heart_env/bin/activate && \
python3 main_unified.py \
  --memory-mode medium \
  --mode production \
  --text "딥러닝의 미래와 AGI 가능성" \
  --epochs 50 \
  --batch-size 4 \
  --use-io-pipeline \
  --use-workflow-dsm \
  --llm claude \
  --enriched-embedding \
  --enable-circuit \
  --enable-meta-learning \
  --checkpoint training/checkpoints_final/best_model.pt \
  --verbose \
  --debug \
  > production_full_test_$(date +%Y%m%d_%H%M%S).txt 2>&1 &
```

#### 2. 벤치마크 모드 (성능 측정)
```bash
source red_heart_env/bin/activate && \
python3 main_unified.py \
  --mode benchmark \
  --samples 100 \
  --memory-mode medium \
  --use-io-pipeline \
  --use-workflow-dsm \
  --llm claude \
  --plot \
  --compare-modes \
  > benchmark_results_$(date +%Y%m%d_%H%M%S).txt 2>&1 &
```

### 🔄 워크플로우 단계별 실행

#### Phase 1: LLM 초기 분석
```bash
source red_heart_env/bin/activate && \
python3 main_unified.py \
  --memory-mode medium \
  --mode inference \
  --text "분석할 텍스트" \
  --phase llm-only \
  --llm claude \
  --debug
```

#### Phase 2: 통합 임베딩
```bash
source red_heart_env/bin/activate && \
python3 main_unified.py \
  --memory-mode medium \
  --mode inference \
  --text "분석할 텍스트" \
  --phase embedding-only \
  --enriched-embedding \
  --debug
```

#### Phase 3: RedHeart 코어 분석
```bash
source red_heart_env/bin/activate && \
python3 main_unified.py \
  --memory-mode medium \
  --mode inference \
  --text "분석할 텍스트" \
  --phase redheart-only \
  --epochs 50 \
  --debug
```

#### Phase 4: Circuit 브레이커
```bash
source red_heart_env/bin/activate && \
python3 main_unified.py \
  --memory-mode medium \
  --mode inference \
  --text "분석할 텍스트" \
  --phase circuit-only \
  --enable-circuit \
  --debug
```

### 🐛 디버깅 및 모니터링

#### 1. 실시간 로그 모니터링
```bash
# 터미널 1: 실행
source red_heart_env/bin/activate && \
python3 main_unified.py \
  --memory-mode medium \
  --mode inference \
  --text "테스트 텍스트" \
  --epochs 50 \
  --use-io-pipeline \
  --use-workflow-dsm \
  --llm claude \
  --debug \
  2>&1 | tee realtime_log_$(date +%Y%m%d_%H%M%S).txt

# 터미널 2: 모니터링
tail -f realtime_log_*.txt | grep -E "(Phase|Loading|Unloading|Memory|Error)"
```

#### 2. GPU 메모리 모니터링
```bash
# 별도 터미널에서 실행
watch -n 1 'nvidia-smi | grep -E "(MiB|%)" && echo "---" && ps aux | grep main_unified'
```

#### 3. WorkflowDSM 상태 확인
```bash
source red_heart_env/bin/activate && \
python3 -c "
from dynamic_swap_manager import WorkflowDSM
import asyncio

async def check():
    dsm = WorkflowDSM()
    await dsm.initialize()
    print(f'Current Phase: {dsm.current_phase}')
    print(f'Memory Usage: {dsm.get_memory_status()}')
    await dsm.cleanup()

asyncio.run(check())
"
```

### 🚨 트러블슈팅

#### I/O Pipeline 실패 시
```bash
# 동기 모드로 폴백 (비권장, 디버깅용)
source red_heart_env/bin/activate && \
python3 main_unified.py \
  --memory-mode minimal \
  --mode inference \
  --text "테스트" \
  --epochs 10 \
  --no-io-pipeline \
  --device cpu \
  --debug
```

#### 메모리 부족 시
```bash
# 최소 메모리 모드 + CPU 사용
source red_heart_env/bin/activate && \
python3 main_unified.py \
  --memory-mode minimal \
  --mode inference \
  --text "간단한 텍스트" \
  --epochs 5 \
  --batch-size 1 \
  --device cpu \
  --no-io-pipeline \
  --debug
```

#### SentenceTransformer 서버 문제 시
```bash
# 서버 재시작
pkill -f sentence_transformer_server
sleep 2

# 단독 실행 테스트
source red_heart_env/bin/activate && \
python3 -c "
from sentence_transformer_singleton import get_sentence_transformer
st = get_sentence_transformer()
result = st.encode(['테스트 문장'])
print(f'Embedding shape: {result.shape}')
"
```

### 📝 배치 처리 명령어

#### 여러 텍스트 동시 처리
```bash
source red_heart_env/bin/activate && \
cat texts.txt | while read line; do
  python3 main_unified.py \
    --memory-mode medium \
    --mode inference \
    --text "$line" \
    --epochs 50 \
    --use-io-pipeline \
    --use-workflow-dsm \
    --llm claude \
    --output-dir results/ \
    >> batch_log_$(date +%Y%m%d).txt 2>&1
  sleep 2  # GPU 메모리 정리 대기
done &
```

### 🎯 권장 실행 순서

1. **시스템 상태 확인**
```bash
nvidia-smi
ps aux | grep -E "(sentence_transformer|main_unified)"
```

2. **가상환경 활성화**
```bash
source red_heart_env/bin/activate
```

3. **간단한 테스트**
```bash
python3 main_unified.py --mode test --samples 1 --epochs 1 --memory-mode minimal --debug
```

4. **메인 실행 (개편된 시스템)**
```bash
python3 main_unified.py \
  --memory-mode medium \
  --mode inference \
  --text "AI의 윤리적 문제와 해결 방안" \
  --epochs 50 \
  --use-io-pipeline \
  --use-workflow-dsm \
  --llm claude \
  --debug \
  > output_$(date +%Y%m%d_%H%M%S).txt 2>&1 &
```

5. **로그 확인**
```bash
tail -f output_*.txt
```

---

**최종 업데이트**: 2025-09-08
**버전**: 2.0 (I/O Pipeline + WorkflowDSM 통합)
**주요 변경사항**: 
- I/O Pipeline 비동기 처리 추가
- WorkflowDSM 2레벨 메모리 관리
- LLM 타입별 자동 번역기 설정
- 통합 임베딩 (원본 + LLM 분석 결과)