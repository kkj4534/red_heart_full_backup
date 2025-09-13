# Red Heart AI 워크플로우 아키텍처 v2.0
_통합 GPU 관리 및 최적화된 추론 파이프라인_

## 📌 핵심 설계 원칙

### 1. 워크플로우 순서 (Sequential GPU Exclusive)
```
Text Input → LLM 초기 분석 → 통합 임베딩 → RedHeart 처리 → Circuit 분석 → LLM 최종 종합
```

### 2. GPU 메모리 관리 전략 (8GB VRAM 제약)
- **Phase별 GPU 독점**: 한 번에 하나의 주요 컴포넌트만 GPU 사용
- **2단계 DSM**: 워크플로우 레벨 + RedHeart 내부 레벨
- **싱글톤 패턴**: SentenceTransformer RAM 상주

## 🔄 상세 워크플로우

### Phase 0: 입력 및 번역
```python
# 입력 텍스트 수신
original_text = "분석할 텍스트"

# LLM 타입별 자동 번역기 설정
if llm_mode == "local":
    use_translator = True  # 로컬 LLM은 영어만 지원
elif llm_mode in ["gpt", "claude", "perplexity"]:
    use_translator = False  # API는 한국어 직접 지원

# 한국어 감지 및 번역 (필요시)
if use_translator and is_korean(original_text):
    text = translator.translate_ko_to_en(original_text)
else:
    text = original_text
```

### Phase 1: LLM 초기 분석
```python
# 워크플로우 DSM으로 LLM 로드
await workflow_dsm.load_phase('llm')  # GPU: 4GB

# LLM 초기 분석
llm_prompt = f"""
Analyze: {text}
Provide:
1. Emotional states (joy, sadness, anger, fear, surprise, disgust, neutral)
2. Three possible scenarios
3. Ethical considerations
4. Potential regret factors
Format: JSON
"""

llm_response = await llm.generate(llm_prompt)
llm_analysis = json.loads(llm_response)

# LLM은 GPU에 유지 (다음 Phase에서 언로드)
```

### Phase 2: 통합 임베딩 생성
```python
# 풍부한 컨텍스트 구성 (원본 + LLM 분석)
enriched_text = f"""
원본 텍스트: {original_text}
감정 분석: {llm_analysis['emotions']}
시나리오: {llm_analysis['scenarios']}
윤리적 고려사항: {llm_analysis['ethics']}
후회 요인: {llm_analysis['regret_factors']}
"""

# SentenceTransformer 싱글톤 (RAM 상주)
# GPU 연산 필요시 텐서만 이동
embeddings = sentence_transformer.encode(enriched_text)  # 896차원

# LLM 언로드
await workflow_dsm.unload_phase('llm')  # GPU: 0GB
```

### Phase 3: RedHeart 초기화 및 준비
```python
# RedHeart 모듈 순차 로드
await workflow_dsm.load_phase('redheart')  # GPU: 0GB → 3GB

# 순차적 초기화 (DSM 관리 하)
components = [
    ('unified_model', 625, SwapPriority.CRITICAL),
    ('neural_analyzers', 368, SwapPriority.HIGH),
    ('advanced_wrappers', 112, SwapPriority.MEDIUM),
    ('dsp_simulator', 14, SwapPriority.LOW),
    ('phase_networks', 80, SwapPriority.MEDIUM)
]

for name, size_mb, priority in components:
    await redheart_dsm.load_component(name, size_mb, priority)
    
# 초기화 완료 후 RAM으로 부분 스왑
await redheart_dsm.optimize_memory()  # GPU: 2GB (핵심만 유지)
```

### Phase 4: RedHeart 처리 (DSM 동적 관리)
```python
# RedHeart 내부 DSM이 자동으로 GPU 관리
redheart_result = await redheart.process_with_dsm(
    embeddings=embeddings,
    llm_hints={
        'emotions': llm_analysis['emotions'],
        'scenarios': llm_analysis['scenarios']
    }
)

# DSM 동작 예시
# Step 1: emotion → GPU 로드 → 처리 → 결과 저장
# Step 2: emotion → RAM, bentham → GPU → 처리
# Step 3: bentham → RAM, counterfactual → GPU → 처리
# ...각 Step마다 필요한 모듈만 GPU 사용
```

### Phase 5: Circuit 분석 (GPU 여유 확보 후)
```python
# RedHeart 핵심 모듈만 RAM으로 스왑
await redheart_dsm.swap_to_ram(['unified_model', 'neural_analyzers'])
# GPU: 1GB (Circuit용 공간 확보)

# EmotionEthicsRegretCircuit 로드 및 실행
circuit_context = CircuitDecisionContext(
    scenario_text=text,
    emotion_data=redheart_result['emotion'],
    bentham_scores=redheart_result['bentham'],
    llm_scenarios=llm_analysis['scenarios']
)

circuit_result = await circuit.process_ethical_decision(circuit_context)

# Circuit 언로드
await workflow_dsm.unload_component('circuit')
```

### Phase 6: LLM 최종 종합
```python
# RedHeart 완전 언로드, LLM 재로드
await workflow_dsm.swap_phases('redheart', 'llm')  # GPU: 4GB

# 최종 종합 프롬프트
final_prompt = f"""
원본: {original_text}
RedHeart 분석: {redheart_result}
Circuit 판단: {circuit_result}

위 분석을 종합하여 최종 윤리적 평가를 제공하세요.
"""

final_response = await llm.generate(final_prompt)
```

## 🏗️ DSM 2단계 구조

### Level 1: 워크플로우 DSM
```python
class WorkflowDSM:
    """전체 Phase 간 GPU 관리"""
    
    phases = {
        'llm': {'size': 4096, 'priority': HIGH},
        'redheart': {'size': 3072, 'priority': CRITICAL},
        'circuit': {'size': 1024, 'priority': MEDIUM}
    }
    
    async def load_phase(self, phase_name):
        # 현재 GPU 사용 Phase 언로드
        await self.unload_current()
        # 새 Phase 로드
        await self.load_to_gpu(phase_name)
        
    async def swap_phases(self, from_phase, to_phase):
        # 원자적 Phase 교체
        await self.unload_phase(from_phase)
        await self.load_phase(to_phase)
```

### Level 2: RedHeart DSM
```python
class RedHeartDSM(DynamicSwapManager):
    """RedHeart 내부 모듈 간 세밀한 GPU 관리"""
    
    async def process_step(self, step_name, required_modules):
        # 필요한 모듈만 GPU 로드
        for module in required_modules:
            if not self.is_on_gpu(module):
                await self.swap_in(module)
        
        # 불필요한 모듈 RAM 스왑
        for module in self.loaded_modules:
            if module not in required_modules:
                await self.swap_out(module)
                
        # OOM 방지 - 우선순위 기반 추가 스왑
        if self.gpu_memory_usage > 0.9:
            await self.emergency_swap()
```

## 📊 메모리 사용 프로파일

| Phase | GPU 사용 | RAM 사용 | 주요 컴포넌트 |
|-------|----------|----------|--------------|
| 입력/번역 | 0GB | 0.3GB | Translator |
| LLM 초기 | 4GB | 1.2GB | LLM Engine + SentenceTransformer |
| 임베딩 | 0.5GB | 5.2GB | SentenceTransformer (GPU 연산) |
| RedHeart 준비 | 2GB | 3GB | 핵심 모듈만 GPU |
| RedHeart 실행 | 2-4GB (동적) | 1-3GB (동적) | DSM 자동 관리 |
| Circuit | 1GB | 6GB | Circuit + 최소 RedHeart |
| LLM 최종 | 4GB | 3GB | LLM Engine |

## 🔧 구현 체크리스트

### 필수 구현
- [ ] 워크플로우 DSM 구현 (`workflow_dsm.py`)
- [ ] LLM 결과 포함 통합 임베딩
- [ ] Phase별 GPU 독점 메커니즘
- [ ] Circuit 독립 실행 지원

### 기존 시스템 수정
- [ ] `main_unified.py`: 워크플로우 순서 변경
- [ ] `dynamic_swap_manager.py`: 2단계 구조 지원
- [ ] `sentence_transformer_singleton.py`: GPU 텐서 이동 최적화

### 검증
- [ ] 8GB GPU에서 전체 워크플로우 실행
- [ ] 각 Phase별 메모리 프로파일링
- [ ] DSM 스왑 횟수 및 오버헤드 측정

## 🚀 실행 명령어

### 기본 실행 (I/O Pipeline 활성화)
```bash
python3 main_unified.py \
    --mode inference \
    --text "분석할 텍스트" \
    --memory-mode medium \
    --llm local \
    --use-io-pipeline \
    --debug
```

### 워크플로우 DSM 모니터링
```bash
python3 main_unified.py \
    --mode inference \
    --text "분석할 텍스트" \
    --memory-mode medium \
    --llm local \
    --use-io-pipeline \
    --monitor-dsm \
    --verbose
```

## ⚠️ 주의사항

1. **SentenceTransformer는 언로드하지 않음** (싱글톤 RAM 상주)
2. **DSM 동기화**: CPU 작업 완료까지 대기 필수
3. **Phase 전환 시 GPU 캐시 정리** 필수
4. **LLM 타입별 번역기 자동 설정** 확인

## 📈 성능 목표

- **추론 시간**: 200ms 이하 (텍스트 100단어 기준)
- **GPU 최대 사용량**: 5GB (여유 3GB 확보)
- **스왑 오버헤드**: 전체 시간의 10% 이하
- **메모리 OOM**: 0% (완전 방지)

---
*최종 업데이트: 2025-01-07*
*작성자: Red Heart AI Team*
*버전: 2.0 (통합 워크플로우 아키텍처)*