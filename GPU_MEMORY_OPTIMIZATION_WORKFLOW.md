# Red Heart AI GPU 메모리 최적화 워크플로우

## 📋 전체 시스템 아키텍처

### 🚀 메인 실행 플로우
```
run_learning.sh → unified_system_main.py → UnifiedSystemOrchestrator → initialize_system()
```

### 🧠 핵심 메모리 관리 파일들

#### 1. **config.py** - 마스터 메모리 오케스트레이터
- **MasterMemoryOrchestrator 클래스**: 모든 GPU 메모리 관리의 중앙 통제소
- **intelligent_unload_model()**: PyTorch 2025 모범 사례 적용한 진짜 언로드
- **_emergency_intelligent_cleanup()**: 105% 메모리 폭주 대응 시스템
- **_discover_and_register_gpu_models()**: 실제 GPU 상주 모델 스캔 시스템

#### 2. **unified_system_orchestrator.py** - 시스템 초기화 관리
- **_initialize_core_components()**: 746라인에서 헤드 초기화 호출
- **initialize_unified_memory_system()**: 710-715라인에서 메모리 시스템 활성화
- **_log_gpu_memory_state()**: 실시간 메모리 모니터링

#### 3. **head_compatibility_interface.py** - 헤드 초기화 최적화
- **initialize_all_heads()**: GPU 순차적 초기화 + 즉시 스왑 시스템
- **85% 예측 기반 관리**: 다음 모델 로딩 전 메모리 예측 및 사전 스왑
- **헤드 우선순위 시스템**: 메모리 사용량 기준 순차 로딩

---

## 🔄 실제 워크플로우 단계

### Phase 1: 시스템 시작 (unified_system_main.py)
```python
1. UnifiedSystemRunner 생성
2. UnifiedSystemOrchestrator(config) 초기화
3. await orchestrator.initialize_system() 호출 ← 메모리 관리 시작점
```

### Phase 2: 핵심 컴포넌트 초기화 (unified_system_orchestrator.py)
```python
1. initialize_unified_memory_system() → MasterMemoryOrchestrator 활성화
2. RedHeartUnifiedBackbone() → 300M 백본 GPU 로드 (21.5% 사용)
3. RedHeartDynamicSwapManager() → 스왑 시스템 준비
4. HeadCompatibilityManager() → 헤드 관리자 생성
5. await head_compatibility_manager.initialize_all_heads() ← 핵심 최적화 지점
```

### Phase 3: GPU 순차적 헤드 초기화 (head_compatibility_interface.py)
```python
# 헤드 초기화 순서 (메모리 사용량 오름차순)
for head_type, estimated_mb in [
    (META_INTEGRATION, 40MB),    # 가장 작음
    (SEMANTIC_SURD, 80MB),
    (BENTHAM_FROMM, 120MB), 
    (REGRET_LEARNING, 120MB),
    (EMOTION_EMPATHY, 140MB)     # 가장 큼
]:
    # Step 1: 메모리 예측 (85% 초과 예상 시 사전 스왑)
    predicted_usage = current_usage + (estimated_mb / 80)  # 8GB GPU 기준
    if predicted_usage > 85:
        await master_orch._emergency_intelligent_cleanup()
    
    # Step 2: GPU에서 헤드 초기화
    await adapter.initialize_head()
    
    # Step 3: 즉시 RAM 스왑 (백본만 GPU 상주)
    pytorch_network.to('cpu')
    
    # Step 4: CUDA 캐시 정리
    torch.cuda.empty_cache()
```

---

## 🛠️ 메모리 관리 핵심 메커니즘

### 1. **PyTorch 2025 모범 사례 언로드 시퀀스** (config.py)
```python
# intelligent_unload_model() 메서드
model_instance.to('cpu')              # Step 1: GPU→CPU 이동
del self.active_model_refs[model_id]  # Step 2: WeakRef 제거  
del model_instance                    # Step 3: Python 객체 삭제 (핵심!)
model_instance = None
gc.collect()                          # Step 4: 가비지 컬렉션
torch.cuda.empty_cache()              # Step 5: CUDA 캐시 정리
torch.cuda.synchronize()
```

### 2. **85% 예측 기반 사전 스왑**
- 다음 모델 로딩 시 85% 초과 예상되면 **사전에 스왑 실행**
- 메모리 폭발 방지를 위한 **예방적 조치**

### 3. **긴급 정리 시스템 강화**
```python
# _emergency_intelligent_cleanup() 메서드
1. 실제 GPU 상주 모델 전면 스캔
2. PyTorch 메모리 스냅샷으로 실제 상태 확인
3. CRITICAL 외 모든 모델 강제 언로드 (75% 목표)
4. 3회 반복 가비지 컬렉션
5. CUDA 캐시 완전 정리
```

---

## 📊 현재 메모리 사용률 결과

### ✅ **개선된 메모리 상태**:
- **이전**: 105% 메모리 폭주 (8.4GB/8GB)
- **현재**: 21.5% 안정적 운영 (1.721GB/8GB)
- **백본만 GPU 상주**: 300M 파라미터 (1.685GB)
- **헤드들**: 필요시 동적 로딩 (RAM에서 대기)

### 🧠 **백본 메모리 구성**:
```
총 244개 GPU 텐서, 1.685GB:
1. [50000, 1280] Embedding - 244.1MB (19%)
2. [8192, 1, 1280] Attention - 40.0MB  
3. [2560, 2560] Linear - 25.0MB
4. [5120, 1280] FFN - 25.0MB × 2개
```

---

## 🎯 추가 최적화 가능 영역

### 1. **더 공격적인 메모리 관리**
- 현재 85% → **80% 또는 75%**로 임계치 하향
- 백본 일부 레이어도 **선택적 스워핑** 고려

### 2. **헤드 초기화 시간 단축**
- **병렬 초기화 + 순차 스왑** 전략
- **사전 컴파일된 헤드** 활용

### 3. **메모리 압축 활용**
- **ModelCompressor** 클래스 활성화
- RAM 저장 시 **압축**, GPU 로딩 시 **실시간 해제**

---

## 🔧 주요 설정 파라미터

### config.py 설정
```python
MEMORY_THRESHOLD = 85  # GPU 메모리 임계치 (%)
TARGET_CLEANUP_USAGE = 75  # 긴급 정리 목표 (%)
EMERGENCY_CLEANUP_TIMEOUT = 120.0  # 정리 타임아웃 (초)
```

### head_compatibility_interface.py 설정
```python
head_priority_order = [
    (META_INTEGRATION, 40),    # MB
    (SEMANTIC_SURD, 80), 
    (BENTHAM_FROMM, 120),
    (REGRET_LEARNING, 120),
    (EMOTION_EMPATHY, 140)
]
```

---

## ⚡ 성능 최적화 요약

### 🎯 **현재 상태 (2025-07-23 업데이트)**
1. **현재 메모리 사용률**: 21.5% (1.721GB/8GB) - **너무 보수적!**
2. **문제점**: GPU 메모리 **과소 활용** (78.5% 여유)
3. **개선 방향**: **85% 근접선까지 적극적 활용**으로 성능 최대화 필요

### 🚀 **공격적 GPU 활용 전략**
1. **목표 메모리 사용률**: **80-85%** (6.4-6.8GB)
2. **현재 여유 공간**: 6.3GB → **추가 모델 상주 가능**
3. **헤드 상주 전략**: RAM 스왑 대신 **GPU 동시 상주**
4. **추가 모델 로딩**: 캐시 모델, 프리로드 헤드 등

### 📊 **이론적 최대 활용 계산**
```
현재: 백본 21.5% (1.721GB)
추가 가능: 78.5% (6.279GB) 여유

헤드 총합: 475MB (35+75+115+115+135)
→ 모든 헤드 GPU 상주해도 27.3%만 사용

추가 활용 가능: 85% - 27.3% = 57.7% (4.6GB)
→ 더 많은 모델/캐시를 GPU에 상주 가능!
```

### 🔥 **85% 근접 활용을 위한 전략**
1. **모든 헤드 GPU 상주**: 스왑 대신 동시 로딩
2. **프리로드 시스템**: 다음 사용 예상 모델들 사전 로딩
3. **캐시 레이어**: 자주 사용되는 중간 결과 GPU 캐시
4. **배치 처리**: 여러 헤드 동시 실행으로 처리량 최대화

이 시스템은 **8GB GPU를 85% 근접선까지 적극 활용**하여 **최대 성능**을 달성하도록 재설계되었습니다.