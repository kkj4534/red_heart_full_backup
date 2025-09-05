# 🚨 RedHeart AI Critical Issues Deep Analysis Report

## 🔥 Executive Summary
시스템이 GPU OOM(Out of Memory)로 인한 치명적 실패를 경험하고 있음. 근본 원인은 잘못된 메모리 관리와 비효율적인 모델 로딩 전략.

## 📊 핵심 문제 분석

### 1. SentenceTransformer Subprocess GPU 메모리 누수 (800MB+ 낭비)
**문제점:**
- `sentence_transformer_singleton.py`가 각 모델마다 별도 subprocess 생성
- 각 subprocess가 GPU에 독립적으로 모델 로드 (400MB × 2 = 800MB)
- MEDIUM 모드에서도 GPU 사용 (FORCE_CPU_INIT 무시)

**코드 분석:**
```python
# advanced_emotion_analyzer.py:338
self.device = get_device()  # MEDIUM 모드에서도 'cuda' 반환

# advanced_emotion_analyzer.py:872-881
self.embedders['multilingual'] = get_sentence_transformer(
    multilingual_model,
    device=str(self.device)  # 'cuda' 전달
)
self.embedders['korean'] = get_sentence_transformer(
    korean_embedding_model,
    device=str(self.device)  # 'cuda' 전달
)
```

**결과:**
- 두 개의 독립 subprocess가 GPU에 모델 로드
- DSM이 이 subprocess 메모리를 관리할 수 없음
- GPU 사용률 82.3% → 99.4% 급증

### 2. 모듈 동시 로딩 문제
**문제점:**
- `main_unified.py:initialize()`에서 모든 모듈 동시 초기화
- Workflow phase 무관하게 전체 로드
- 메모리 압박 발생

**코드 분석:**
```python
# main_unified.py:380-400
await self._load_unified_model()       # 250M
await self._load_translator()          # CPU (OK)
await self._load_neural_analyzers()    # 368M (GPU)
await self._load_advanced_wrappers()   # 112M (일부 GPU)
await self._load_dsp_components()      # 추가 메모리
await self._load_phase_networks()      # 추가 메모리
```

### 3. DSM 언로드 실패
**문제점:**
- CRITICAL priority 모델 (backbone, heads) 언로드 불가
- SentenceTransformer subprocess 모델 관리 불가
- 실제 해제 가능한 모델 없음 (0.0MB freed)

**코드 분석:**
```python
# dynamic_swap_manager.py:1873-1875
if model_info.priority == SwapPriority.CRITICAL:
    logger.debug(f"[CRITICAL 보호] {name}은 언로드 불가")
    continue
```

### 4. FORCE_CPU_INIT 부분 적용
**문제점:**
- AdvancedBenthamCalculator는 FORCE_CPU_INIT 준수 ✅
- AdvancedEmotionAnalyzer는 무시 ❌
- 일관성 없는 CPU/GPU 정책

### 5. Perplexity API JSON 파싱 실패
**문제점:**
- 응답 형식 불일치로 JSON 파싱 에러 반복
- 에러 핸들링 부재로 시스템 중단

## 🛠️ 종합 해결 방안

### Phase 1: 즉시 수정 (Critical)

#### 1.1 SentenceTransformer MEDIUM 모드 CPU 강제
```python
# advanced_emotion_analyzer.py 수정
def __init__(self):
    # ... 기존 코드 ...
    
    # MEDIUM 모드 체크
    import os
    if os.environ.get('FORCE_CPU_INIT', '0') == '1':
        self.device = torch.device('cpu')
        logger.info("📌 FORCE_CPU_INIT: AdvancedEmotionAnalyzer CPU 모드")
    else:
        from config import get_device
        self.device = get_device()
    
    # ... 나머지 코드 ...
    
    # 임베딩 모델 로드 시 device 강제
    device_str = 'cpu' if os.environ.get('FORCE_CPU_INIT', '0') == '1' else str(self.device)
    
    self.embedders['multilingual'] = get_sentence_transformer(
        multilingual_model,
        device=device_str  # CPU/GPU 분기
    )
```

#### 1.2 SentenceTransformer Subprocess 제거
```python
# sentence_transformer_singleton.py 대체 구현
class SentenceTransformerManager:
    def get_model(self, model_name: str, device: str = None):
        # subprocess 대신 직접 로드
        if device == 'cpu' or device == 'cpu:0':
            # CPU에서 직접 로드
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name, device='cpu')
        else:
            # GPU 로드는 DSM 통해 관리
            model = self._load_with_dsm(model_name, device)
        return model
```

### Phase 2: Workflow 기반 순차 로딩

#### 2.1 Lazy Loading 구현
```python
# main_unified.py 수정
async def initialize(self):
    # 필수 모듈만 초기화
    await self._load_unified_model()  # backbone/heads 필수
    
    # 나머지는 lazy loading
    self.pending_modules = {
        'neural_analyzers': self._load_neural_analyzers,
        'advanced_wrappers': self._load_advanced_wrappers,
        # ...
    }
    
async def _ensure_module_loaded(self, module_name: str):
    """필요 시점에 모듈 로드"""
    if module_name in self.pending_modules:
        await self.pending_modules[module_name]()
        del self.pending_modules[module_name]
```

#### 2.2 Workflow Phase 정의
```python
# workflow_phases.py
class WorkflowPhase(Enum):
    INIT = "init"           # backbone만
    EMBED = "embed"         # +embedding models
    ANALYZE = "analyze"     # +analyzers
    INTEGRATE = "integrate" # +integration
    GENERATE = "generate"   # +LLM
```

### Phase 3: DSM 개선

#### 3.1 Priority 재조정
```python
# CRITICAL을 세분화
class SwapPriority(Enum):
    BACKBONE = 5     # 절대 언로드 불가
    PRIMARY = 4      # 워크플로우 핵심 (조건부 언로드)
    SECONDARY = 3    # 보조 모듈 (언로드 가능)
    AUXILIARY = 2    # 부가 기능 (우선 언로드)
    TEMPORARY = 1    # 임시 (즉시 언로드)
```

#### 3.2 Subprocess 모델 관리
```python
# DSM에 subprocess 모델 추가
def register_subprocess_model(self, process_id: int, model_name: str, size_mb: float):
    """subprocess 모델 등록 및 추적"""
    self.subprocess_models[process_id] = {
        'name': model_name,
        'size_mb': size_mb,
        'can_terminate': True
    }
```

### Phase 4: 메모리 목표 설정

#### 4.1 MEDIUM 모드 메모리 배분
```yaml
Total GPU: 8192 MB
Reserved: 1500 MB (OS/Driver)
Available: 6692 MB

Allocation:
- Backbone: 250 MB (BACKBONE priority)
- Heads: 250 MB (BACKBONE priority)
- Current Phase Models: 1500 MB (PRIMARY)
- Buffer: 500 MB
- Dynamic: 4192 MB (SECONDARY/AUXILIARY)
```

## 📝 Action Items

### 즉시 실행 (오늘)
1. [ ] AdvancedEmotionAnalyzer FORCE_CPU_INIT 수정
2. [ ] SentenceTransformer subprocess → 직접 로드 변경
3. [ ] Perplexity API 에러 핸들링 추가

### 단기 (1-2일)
1. [ ] Lazy loading 구현
2. [ ] Workflow phase 시스템 구현
3. [ ] DSM priority 세분화

### 중기 (3-5일)
1. [ ] WAUP 정책 완전 구현
2. [ ] 메모리 프로파일링 도구 추가
3. [ ] 자동 메모리 최적화 시스템

## 🎯 예상 결과
- GPU 메모리 사용: 99.4% → 85% 목표
- 모델 로드 시간: 50% 단축
- OOM 에러: 완전 제거
- 추론 속도: 30% 개선

## 📊 측정 지표
```python
# 수정 전
GPU Usage: 99.4% (7.7GB/8.0GB)
Models Loaded: 15개 동시
Subprocess: 2개 (800MB)
DSM Freed: 0.0MB

# 목표
GPU Usage: 85% (6.8GB/8.0GB)
Models Loaded: 5-7개 (phase별)
Subprocess: 0개
DSM Freed: 500MB+
```

## 🔍 검증 방법
```bash
# 테스트 스크립트
python3 main_unified.py \
  --memory-mode medium \
  --mode inference \
  --text "테스트" \
  --debug \
  --monitor-memory
```

---
*작성일: 2025-09-04*
*작성자: Claude 4 Extended Thinking*
*검토 필요: GPU 메모리 관리 전문가*