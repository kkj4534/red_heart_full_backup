# Red Heart I/O 파이프라인 구현 계획

## 📋 프로젝트 개요

### 배경
Red Heart 시스템은 "통합의 저주"에 걸려 있음:
- 모든 모듈이 강하게 결합되어 분리 불가능
- Claude API 독립 실행이 구조적으로 불가능
- 3개의 다른 메모리 관리 시스템이 혼재
- 730M 파라미터 목표 미달성 (실제 625M)

### 해결 방향
사용자와 합의된 방향:
1. **모듈 간 I/O 관리를 통한 분리화** - 모놀리식 구조는 유지하되 비동기 파이프라인화
2. **통합 메모리 관리** - 3개 시스템을 UnifiedMemoryManager로 통합
3. **3중 감정 처리 유지** - 의도적 설계이므로 보존
4. **LLM 독립성 확보** - 플러그인 시스템으로 LLM/API/MCP 교체 가능
5. **모듈 경량화 없음** - 성능상 중요하므로 유지
6. **인터페이스 표준화** - 모듈 간 통신 프로토콜 통일

## 🏗️ 아키텍처 설계

### 1. 전체 시스템 구조
```
┌─────────────────────────────────────────────────┐
│                  Main Entry Point                │
│                 (main_unified.py)                │
└──────────────────┬──────────────────────────────┘
                   │
       ┌───────────┴───────────┐
       │    IOPipeline Core    │
       │  (비동기 큐 시스템)    │
       └───┬───────────────┬───┘
           │               │
    ┌──────┴──────┐  ┌─────┴──────┐
    │   Input     │  │   Output    │
    │   Queue     │  │   Queue     │
    └──────┬──────┘  └─────┬──────┘
           │               │
┌──────────┴───────────────┴──────────┐
│     UnifiedMemoryManager (UMM)      │
│  (메모리 조율 및 Phase 관리)          │
└──────────┬───────────────────────────┘
           │
    ┌──────┴──────────────────┐
    │    Module Handlers       │
    ├──────────────────────────┤
    │ • LLM Plugin Handler     │
    │ • UnifiedModel Handler   │
    │ • Neural Analyzer Handler│
    │ • Circuit Handler        │
    │ • Advanced Wrapper Handler│
    └──────────────────────────┘
```

### 2. 데이터 플로우 (비동기적 동기 제어)
```
[동기적 순차 처리 워크플로우]
사용자 입력 → LLM 초기 분석 → GPU 스왑 → Red Heart 처리 → GPU 스왑 → Circuit → GPU 스왑 → LLM 최종
     ↓            ↓            ↓              ↓            ↓          ↓         ↓            ↓
  IOPipeline   Plugin시스템  wait_step    DSM 활성화   wait_step   분석     wait_step    요약

[DSM 철학 적용]
- 각 화살표에서 wait_for_step()으로 동기화
- GPU 스왑 시점에서 CPU/GPU 작업 완료 확인
- 비대칭 처리 방지: 모든 모듈 완료 후 다음 진행
```

### 3. 메모리 관리 전략
| Phase | 모듈 | GPU 사용 | 스왑 전략 |
|-------|------|----------|-----------|
| LLM 초기 | LLM Engine | ~4GB | 사용 후 RAM 스왑 |
| Red Heart | UnifiedModel | ~3GB | DSM 동적 관리 |
| Red Heart | Neural Analyzers | ~1GB | 필요시 로드 |
| Circuit | EmotionEthicsRegret | ~1GB | 사용 후 RAM 스왑 |
| LLM 최종 | LLM Engine | ~4GB | 사용 후 해제 |

## 📝 상세 구현 TODO 리스트

### 🔴 긴급 수정 사항 (즉시 처리)

#### TODO-001: NeuralAnalyzers 초기화 버그 수정
```python
# 위치: main_unified.py 라인 620-714
# 문제: analyzer = analyzer.to('cpu') 변수 재할당 버그
# 수정:
self.neural_analyzers[name] = analyzer.to('cpu')
```
- [ ] main_unified.py 라인 669, 681, 693, 706 수정
- [ ] 변수 재할당 대신 딕셔너리 직접 업데이트
- [ ] 테스트: `python3 main_unified.py --mode inference --text "test"`

#### TODO-002: UnboundLocalError 'os' 변수 수정
```python
# 위치: main_unified.py 상단 import 섹션
# 문제: 중복된 조건부 import 제거
# 수정: os 모듈을 무조건 import
```
- [ ] 중복 import 문 제거
- [ ] os 모듈 기본 import로 변경
- [ ] protobuf 의존성 확인

#### TODO-003: Advanced Wrappers LLM 의존성 제거
```python
# 위치: main_unified.py 라인 1612
# 문제: LLM 초기 분석이 Advanced Wrappers에 의존
# 수정: 독립 함수 생성
```
- [ ] `_llm_initial_analysis_independent()` 함수 생성
- [ ] Advanced Wrappers 체크 제거
- [ ] 직접 LLM 엔진 사용하도록 수정

#### TODO-004: 외부 모델 중복 로드 긴급 수정 🔴
```python
# 위치: advanced_bentham_calculator.py 라인 185-218
# 문제: AutoModel.from_pretrained 직접 호출로 싱글톤 패턴 우회
# 영향: GPU OOM 발생 (692-1384MB 메모리 낭비)
```
- [ ] advanced_bentham_calculator를 sentence_transformer_singleton 사용하도록 수정
- [ ] hf_model_wrapper의 과대 메모리 추정 수정 (800MB → 346MB)
- [ ] 기존 AutoModel 코드 제거
- [ ] 테스트: GPU 메모리 사용량 측정

#### TODO-005: 번역 모델 조건부 로드
```python
# 위치: local_translator.py
# 문제: opus-mt-ko-en 모델이 항상 로드됨 (~300MB)
# 수정: 필요시에만 로드
```
- [ ] lazy loading 패턴 적용
- [ ] API 모드에서는 로드 스킵
- [ ] 영어 전용 모델 + 한국어 텍스트일 때만 로드

#### TODO-006: SentenceTransformer 차원 불일치 문제 수정 🔴
```python
# 문제 1: 차원 불일치
# - UnifiedModel 기대: 768차원
# - all-MiniLM-L6-v2: 384차원 (불일치!)
# - ko-sroberta-multitask: 768차원 (일치)

# 문제 2: 과도한 패딩
# - max_seq_length=512로 패딩 → 메모리 낭비
# - 실제 필요: 단일 문장 임베딩 (1차원)
```
- [ ] all-MiniLM-L6-v2를 768차원 모델로 교체
- [ ] 또는 384→768 프로젝션 레이어 추가
- [ ] max_seq_length 패딩 제거 (단일 임베딩만 사용)
- [ ] 메모리 사용량 측정 (예상 절감: 512배)

### 🟡 Phase 1: 기반 구조 (1주차)

#### TODO-100: IOPipeline 클래스 구현 (비동기적 동기 제어)
```python
# 파일: io_pipeline.py (신규 생성)
# DSM 철학: 비동기 큐 기반이지만 스텝별 동기 보장
class IOPipeline:
    def __init__(self):
        self.input_queue = asyncio.Queue()
        self.output_queue = asyncio.Queue()
        self.handlers = {}
        self.processing_tasks = []
        # 동기화 메커니즘 추가
        self.step_barriers = {}  # 스텝별 완료 대기
        self.cpu_gpu_sync = asyncio.Lock()  # CPU/GPU 비대칭 방지
```
- [ ] 기본 클래스 구조 생성
- [ ] 입력/출력 큐 구현
- [ ] 모듈 핸들러 등록 시스템
- [ ] 비동기 처리 루프 구현
- [ ] **스텝별 동기화 배리어 구현** (비대칭 방지)
- [ ] **CPU/GPU 작업 완료 동기화**
- [ ] 에러 처리 및 재시도 로직
- [ ] 큐 오버플로우 방지 로직
- [ ] 우선순위 큐 지원 추가

#### TODO-101: 모듈 라우팅 시스템
```python
# 파일: io_pipeline.py
async def route_to_module(self, task):
    handler = self.handlers.get(task['module'])
    if not handler:
        raise ValueError(f"Unknown module: {task['module']}")
    return await handler(task['data'])
```
- [ ] 모듈별 라우터 정의
- [ ] 동적 핸들러 등록/해제
- [ ] 라우팅 테이블 관리
- [ ] 모듈 상태 모니터링

#### TODO-102: 표준 데이터 구조 정의
```python
# 파일: data_structures.py (신규 생성)
@dataclass
class TaskMessage:
    module: str
    task_type: str
    data: Dict
    priority: int = 0
    timestamp: float = field(default_factory=time.time)
```
- [ ] TaskMessage 클래스 정의
- [ ] ResultMessage 클래스 정의
- [ ] EmotionData 표준화
- [ ] BenthamResult 표준화
- [ ] SURDMetrics 표준화
- [ ] 직렬화/역직렬화 메서드

#### TODO-103: UnifiedMemoryManager 구현 (DSM 통합)
```python
# 파일: unified_memory_manager.py (신규 생성)
class UnifiedMemoryManager:
    """DSM 철학: GPU 사용량 기반 동적 스왑 + 의존성 보장"""
    
    def __init__(self, config):
        self.strategy = self._determine_strategy(config)
        self.memory_state = {}
        self.phase_transitions = []
        # DSM 핵심 기능
        self.dependency_graph = {}  # 모듈 의존성
        self.priority_queue = []    # 우선순위 기반 스왑
        
    async def synchronous_swap(self, step_id: str):
        """동기적 GPU 스왑 (CPU/GPU 완료 대기)"""
        # 1. 현재 GPU 모듈 언로드
        await self._unload_gpu_modules()
        # 2. CPU 작업 완료 대기
        await self._wait_cpu_completion()
        # 3. 다음 스텝 모듈 로드
        await self._load_next_modules(step_id)
```
- [ ] 기본 클래스 구조
- [ ] 3개 기존 시스템 통합
  - [ ] SystemSwapManager 통합 (Local LLM용)
  - [ ] DynamicSwapManager 통합 (Red Heart DSM)
  - [ ] DirectGPUManager 통합 (Claude API용)
- [ ] **DSM 동기적 스왑 메커니즘 구현**
- [ ] **모듈 의존성 그래프 관리**
- [ ] **우선순위 기반 GPU 할당**
- [ ] Phase별 메모리 전략 정의
- [ ] 메모리 모니터링 시스템
- [ ] GPU/RAM 스왑 메서드
- [ ] 메모리 압력 감지 시스템

### 🟡 Phase 2: LLM 플러그인 시스템 (2주차)

#### TODO-200: LLMPlugin 추상 클래스
```python
# 파일: llm_plugins/base.py (신규 생성)
from abc import ABC, abstractmethod

class LLMPlugin(ABC):
    @abstractmethod
    async def initialize(self, config: Dict):
        pass
    
    @abstractmethod
    async def analyze_initial(self, text: str) -> Dict:
        pass
    
    @abstractmethod
    async def summarize_final(self, results: Dict) -> str:
        pass
```
- [ ] 추상 클래스 정의
- [ ] 표준 인터페이스 정의
- [ ] 설정 관리 시스템
- [ ] 플러그인 레지스트리

#### TODO-201: ClaudeLLMPlugin 구현
```python
# 파일: llm_plugins/claude_plugin.py (신규 생성)
class ClaudeLLMPlugin(LLMPlugin):
    async def initialize(self, config: Dict):
        self.api_key = config['api_key']
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
```
- [ ] Claude API 클라이언트 초기화
- [ ] analyze_initial 구현
  - [ ] 프롬프트 템플릿 정의
  - [ ] JSON 응답 파싱
  - [ ] 에러 처리
- [ ] summarize_final 구현
- [ ] API 레이트 리밋 처리
- [ ] 재시도 로직
- [ ] Advanced Wrappers 의존성 완전 제거

#### TODO-202: LocalLLMPlugin 구현
```python
# 파일: llm_plugins/local_plugin.py (신규 생성)
class LocalLLMPlugin(LLMPlugin):
    ENGLISH_ONLY_MODELS = ['dolphin-llama3', 'mistral-7b']
    
    async def initialize(self, config: Dict):
        self.model_path = config['model_path']
        self.need_translation = config['model_name'] in self.ENGLISH_ONLY_MODELS
```
- [ ] Local 모델 로더 구현
- [ ] 조건부 번역 모듈 초기화
- [ ] Dolphin-Llama3 통합
- [ ] 메모리 효율적 로딩
- [ ] 텍스트 생성 파이프라인

#### TODO-203: 번역 모듈 조건부 로드
```python
# 수정: main_unified.py
if self.config.llm_mode == "local" and self._needs_translation():
    await self._init_translator()
```
- [ ] 영어 전용 모델 리스트 정의
- [ ] 한국어 감지 로직 구현
- [ ] 조건부 초기화 로직
- [ ] 번역 캐싱 시스템

### 🟡 Phase 3: Red Heart 모듈 I/O 분리 (3주차)

#### TODO-300: RedHeartCore I/O 래퍼
```python
# 파일: red_heart_io.py (신규 생성)
class RedHeartCore:
    def __init__(self, io_pipeline: IOPipeline):
        self.pipeline = io_pipeline
        self.unified_model = None
        self.processing_loop_task = None
```
- [ ] 기본 클래스 구조
- [ ] 비동기 처리 루프
- [ ] UnifiedModel 래퍼 메서드
- [ ] 입력 큐 모니터링
- [ ] 출력 큐 전송

#### TODO-301: UnifiedModel I/O 분리
```python
# 수정: training/unified_training_final.py
async def process_async(self, task_message: TaskMessage):
    # 비동기 처리로 변환
    result = await self._run_in_executor(self.forward, task_message.data)
    return ResultMessage(module='unified_model', data=result)
```
- [ ] forward 메서드 비동기 래퍼
- [ ] 배치 처리 지원
- [ ] 메모리 효율적 처리
- [ ] 결과 직렬화

#### TODO-302: Neural Analyzers I/O 분리
```python
# 수정: analyzer_neural_modules.py
class NeuralAnalyzersIO:
    async def process_batch(self, tasks: List[TaskMessage]):
        # 배치 처리로 효율성 향상
        pass
```
- [ ] 각 Analyzer별 I/O 래퍼
- [ ] 병렬 처리 구현
- [ ] 결과 집계 시스템
- [ ] 에러 격리

#### TODO-303: EmotionEthicsRegretCircuit I/O 분리
```python
# 수정: emotion_ethics_regret_circuit.py
async def process_circuit_async(self, emotion_data):
    # 7단계 처리를 비동기로
    pass
```
- [ ] 경험 DB 비동기 조회
- [ ] 7단계 파이프라인 비동기화
- [ ] 반사실 시나리오 병렬 처리
- [ ] 결과 스트리밍

### 🟡 Phase 4: 통합 및 최적화 (4주차)

#### TODO-400: main_unified.py 리팩토링
```python
# 주요 변경사항
class UnifiedSystem:
    def __init__(self):
        self.io_pipeline = IOPipeline()
        self.memory_manager = UnifiedMemoryManager()
        self.llm_plugin = None
```
- [ ] 기존 동기 코드 제거
- [ ] IOPipeline 통합
- [ ] UnifiedMemoryManager 통합
- [ ] LLM 플러그인 시스템 통합
- [ ] Phase별 처리 로직 재구성
- [ ] 에러 처리 개선

#### TODO-401: SentenceTransformer 중복 제거
```python
# 단일 인스턴스로 통합
self.sentence_transformer = SentenceTransformer(
    'jhgan/ko-sroberta-multitask',
    cache_folder='./models/sentence_transformers'
)
```
- [ ] AdvancedEmotionAnalyzer 수정
- [ ] main_unified._tokenize() 통합
- [ ] 캐시 폴더 통일
- [ ] 메모리 사용량 50% 감소 확인

#### TODO-402: claude_inference.py 제거
- [ ] 코드 백업
- [ ] main_unified.py로 통합
- [ ] 테스트 케이스 이전
- [ ] 파일 제거

### 🟢 Phase 5: 테스트 및 검증 (5주차)

#### TODO-500: 단위 테스트
```python
# 파일: tests/test_io_pipeline.py
class TestIOPipeline:
    async def test_queue_overflow(self):
        # 큐 오버플로우 테스트
        pass
```
- [ ] IOPipeline 테스트
- [ ] UnifiedMemoryManager 테스트
- [ ] LLM 플러그인 테스트
- [ ] 데이터 구조 테스트

#### TODO-501: 통합 테스트
```bash
# 테스트 시나리오
1. Local LLM (Dolphin) 모드
2. Claude API 모드
3. GPT API 모드
4. 메모리 압력 상황
```
- [ ] Local LLM 전체 플로우 테스트
- [ ] Claude API 독립 실행 테스트
- [ ] 메모리 스왑 동작 테스트
- [ ] 에러 복구 테스트

#### TODO-502: 성능 벤치마크
```python
# 측정 항목
- 처리 시간 (기존 vs 신규)
- 메모리 사용량
- GPU 사용률
- 큐 처리량
```
- [ ] 처리 시간 비교
- [ ] 메모리 프로파일링
- [ ] GPU 사용률 모니터링
- [ ] 병목 지점 분석

#### TODO-503: 메모리 누수 테스트
```bash
# 장시간 실행 테스트
python3 main_unified.py --mode inference --continuous --hours 24
```
- [ ] 24시간 연속 실행
- [ ] 메모리 증가 모니터링
- [ ] 리소스 누수 탐지
- [ ] 자동 복구 검증

## 🚨 위험 요소 및 대응 방안

### 위험 1: 비동기 전환시 성능 저하
- **위험도**: 높음
- **대응**: 배치 처리 및 파이프라이닝으로 처리량 유지

### 위험 2: 메모리 관리 통합 실패
- **위험도**: 중간
- **대응**: 단계적 통합, 기존 시스템 백업 유지

### 위험 3: LLM 플러그인 호환성
- **위험도**: 낮음
- **대응**: 표준 인터페이스 엄격 준수

### 위험 4: 기존 코드와의 충돌
- **위험도**: 중간
- **대응**: 점진적 마이그레이션, 피처 플래그 사용

## 📊 성공 지표

1. **독립 실행**: Claude API만으로 추론 가능
2. **메모리 효율**: 8GB VRAM 내에서 안정적 실행
   - 외부 모델 중복 제거로 1.4GB 절감
   - SentenceTransformer 패딩 제거로 512배 메모리 절감
   - GPU OOM 발생률 0%
3. **동기적 제어**: 비대칭 처리 0건
   - DSM 동기적 스왑 100% 성공
   - CPU/GPU 작업 완료 동기화 보장
   - 스텝별 wait_for_step() 정상 작동
4. **성능 유지**: 기존 대비 ±10% 이내 처리 시간
5. **확장성**: 새 LLM 추가시 플러그인만 구현
6. **안정성**: 24시간 연속 실행 가능

## 🔄 마이그레이션 전략

### Step 1: 병렬 실행 (1-2주)
- 기존 시스템과 새 시스템 병렬 운영
- 피처 플래그로 전환 제어
- A/B 테스트 수행

### Step 2: 점진적 전환 (3-4주)
- 모듈별 순차 전환
- LLM 플러그인 먼저 적용
- Red Heart Core 마지막 전환

### Step 3: 기존 코드 제거 (5주)
- 안정성 확인 후 제거
- 백업 유지
- 롤백 계획 수립

## 📅 일정 계획

| 주차 | 작업 내용 | 완료 기준 |
|------|-----------|-----------|
| 1주차 | 기반 구조 구축 | IOPipeline, UMM 구현 완료 |
| 2주차 | LLM 플러그인 | Claude/Local 플러그인 동작 |
| 3주차 | Red Heart I/O | 모든 모듈 I/O 분리 완료 |
| 4주차 | 통합 및 최적화 | main_unified.py 리팩토링 |
| 5주차 | 테스트 및 배포 | 모든 테스트 통과 |

## 🔍 모니터링 및 로깅

### 로깅 전략
```python
# 구조화된 로깅
logger.info({
    'event': 'task_processed',
    'module': 'unified_model',
    'duration': 1.23,
    'memory_used': '2.5GB'
})
```

### 메트릭 수집
- 큐 크기 및 처리량
- 모듈별 처리 시간
- 메모리 사용량 추이
- 에러율 및 재시도 횟수

## 📚 참고 문서

1. **기존 분석 문서**
   - WORKFLOW_ANALYSIS.md: 시스템 복잡도 분석
   - CLAUDE.md: 프로젝트 개발 지침

2. **관련 파일**
   - main_unified.py: 메인 진입점 (3327줄)
   - training/unified_training_final.py: UnifiedModel (1000+줄)
   - analyzer_neural_modules.py: Neural Analyzers (511줄)
   - emotion_ethics_regret_circuit.py: Circuit (1071줄)

3. **외부 참조**
   - AsyncIO 공식 문서
   - PyTorch 메모리 관리 가이드
   - Anthropic Claude API 문서

## ✅ 체크리스트 (대화 압축 후 연속성용)

### 즉시 확인 사항
- [ ] venv 환경 활성화: `source red_heart_env/bin/activate`
- [ ] 현재 브랜치: master
- [ ] 작업 디렉토리: `/mnt/c/large_project/linux_red_heart`

### 현재 상태
- [ ] NeuralAnalyzers 초기화 버그 수정 여부
- [ ] Advanced Wrappers LLM 의존성 제거 여부
- [ ] IOPipeline 클래스 생성 여부
- [ ] UnifiedMemoryManager 구현 여부
- [ ] LLM 플러그인 시스템 구현 여부

### 다음 작업
1. 긴급 버그 수정 (TODO-001, 002, 003)
2. IOPipeline 기본 구조 구현
3. UnifiedMemoryManager 설계
4. ClaudeLLMPlugin 프로토타입

---
*작성일: 2025-01-07*
*최종 업데이트: 2025-01-07 - DSM 철학 및 비동기적 동기 제어 개념 추가*
*작성자: Claude*
*목적: I/O 파이프라인 구현을 통한 Red Heart 시스템 개선 (동기적 제어 보장)*