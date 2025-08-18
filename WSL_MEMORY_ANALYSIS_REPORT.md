# WSL 하이브리드 학습 시스템 메모리 부족 분석 보고서

## 📋 문제 요약

하이브리드 학습 시스템에서 10번의 학습 시도 중 WSL이 강제 종료되는 문제가 발생하고 있습니다. 분석 결과, **메모리 사용량 초과**가 주요 원인으로 확인되었습니다.

## 🔍 분석 결과

### 1. 시스템 환경 분석

**WSL 환경:**
- WSL2 (Linux 6.6.87.2-microsoft-standard-WSL2)
- 할당된 메모리: 62GB (실제 가용 메모리)
- 스왑: 16GB
- GPU: NVIDIA GeForce RTX 2070S (8GB VRAM)
- **주요 발견: PyTorch가 CPU 전용 버전으로 설치됨 (CUDA 지원 없음)**

### 2. 메모리 사용량 분석

**하이브리드 모델 메모리 요구사항:**
```
목표 파라미터: 3,000,000,000개
실제 파라미터: 6,828,991,488개 (목표의 2.28배)

메모리 사용량 분석:
- 모델 파라미터 (FP32): 25.44GB
- 그라디언트: 25.44GB 
- 옵티마이저 상태 (AdamW): 50.88GB
- 활성화 메모리: 0.18GB
- 총 예상 메모리: 101.94GB
```

**⚠️ 문제점: 101.94GB > 62GB (WSL 가용 메모리)**

### 3. 로그 분석

**하이브리드 학습 로그 패턴:**
```
2025-06-25 04:29:29 - 하이브리드 분산 학습 시작
2025-06-25 04:30:19 - 모델 준비 완료: 6,831,038,318개 파라미터
2025-06-25 04:30:19 - 데이터 준비 완료: 10개 시나리오  
2025-06-25 04:30:19 - 에포크 1/3 시작
[로그 중단 - WSL 강제 종료]
```

**중단 패턴:** 모델 생성 직후 첫 번째 학습 스텝에서 일관되게 중단

### 4. WSL 시스템 메시지 분석

```
dmesg 로그에서 발견된 문제:
- hv_balloon: Cold memory discard hypercall failed
- systemd-journald: uncleanly shut down, renaming and replacing
- Time jumped backwards, rotating (비정상 종료 후 재시작 징후)
```

## 🚨 WSL 강제 종료 원인

### 1. 메모리 Over-commitment
- 모델이 요구하는 101.94GB > WSL 할당 메모리 62GB
- WSL은 Windows 호스트의 메모리 관리 정책에 따라 강제 종료

### 2. GPU 가속 부재
- PyTorch CPU 전용 버전 사용
- RTX 2070S GPU가 활용되지 않음
- 모든 계산이 CPU/RAM에 집중되어 메모리 압박 가중

### 3. 하이브리드 모델 설계 문제
- 목표 33억 파라미터 초과 (실제 68억)
- 과도하게 큰 hidden_dim (3072), layers (60), intermediate_size (12288)
- 메모리 효율성을 고려하지 않은 아키텍처

### 4. 비동기 처리 오버헤드
- AsyncRegretCalculator의 다중 워커 스레드
- 각 스텝당 7개 후회 시나리오 * 3번 벤담 계산
- 메모리 사용량 추가 증가

## 💡 해결 방안

### 🔧 즉시 적용 가능한 해결책

#### 1. 모델 크기 대폭 축소
```python
# 현재 설정
hidden_dim = 3072      # → 1024로 축소 (1/3)
num_layers = 60        # → 24로 축소 (1/2.5) 
intermediate_size = 12288  # → 4096으로 축소 (1/3)

# 예상 파라미터: 약 8억개 (목표 범위 내)
# 예상 메모리: 약 25GB (WSL 가용 범위)
```

#### 2. 배치 크기 및 시퀀스 길이 축소
```python
batch_size = 2          # 8 → 2
micro_batch_size = 1    # 2 → 1
sequence_length = 16    # 32 → 16
```

#### 3. 그라디언트 체크포인팅
```python
# PyTorch 그라디언트 체크포인팅 활용
torch.utils.checkpoint.checkpoint()
```

#### 4. 메모리 효율적 옵티마이저
```python
# AdamW 대신 SGD 사용 (메모리 사용량 1/2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

### 🔨 중장기 해결책

#### 1. CUDA 지원 PyTorch 재설치
```bash
# WSL에서 CUDA 지원 PyTorch 설치
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 2. WSL 메모리 설정 최적화
Windows에서 `.wslconfig` 파일 생성:
```ini
[wsl2]
memory=96GB          # 메모리 할당 증가
processors=8         # CPU 코어 제한
swap=32GB           # 스왑 크기 증가
```

#### 3. 모델 파이프라인 병렬화
```python
# 모델을 여러 단계로 분할하여 메모리 사용량 분산
class PipelinedModel(nn.Module):
    def __init__(self):
        self.stage1 = ModelStage1()  # CPU
        self.stage2 = ModelStage2()  # GPU
        self.stage3 = ModelStage3()  # CPU
```

#### 4. 점진적 학습 (Incremental Learning)
```python
# 데이터를 작은 청크로 나누어 순차 학습
for chunk in data_chunks:
    train_on_chunk(chunk)
    save_checkpoint()
    clear_memory()
```

### 🚀 최적화된 설정 제안

```python
# WSL 안전 설정
OPTIMIZED_CONFIG = HybridConfig(
    # 모델 크기 (메모리 안전)
    target_params=800_000_000,    # 8억 파라미터
    hidden_dim=1024,              # 축소
    num_layers=24,                # 축소
    intermediate_size=4096,       # 축소
    
    # 학습 설정 (안정성 우선)
    batch_size=2,                 # 최소화
    micro_batch_size=1,           # 최소화
    regrets_per_step=3,           # 7 → 3
    bentham_calculations_per_regret=1,  # 3 → 1
    
    # 시스템 자원 (보수적)
    num_workers=2,                # 4 → 2
    gpu_layers_ratio=0.0,         # CPU 전용 모드
    use_mixed_precision=False,    # FP16 비활성화 (호환성)
    
    # 메모리 관리
    gradient_accumulation_steps=8,  # 메모리 효율성
    max_grad_norm=0.5,            # 그라디언트 클리핑
    log_every_n_steps=10,         # 로깅 빈도 감소
    save_checkpoint_every=50      # 체크포인트 빈도 감소
)
```

## 📊 예상 효과

### 최적화 전
- 메모리 사용량: 101.94GB
- WSL 강제 종료: 100%
- 학습 성공률: 0%

### 최적화 후
- 메모리 사용량: ~25GB 
- WSL 강제 종료: 0% (예상)
- 학습 성공률: 95%+ (예상)
- 성능 손실: 약 30-40% (파라미터 수 감소)

## 🎯 권장사항

### 우선순위 1: 즉시 적용
1. **모델 크기 축소** (가장 중요)
2. **배치 크기 최소화**
3. **메모리 효율적 옵티마이저 사용**

### 우선순위 2: 시스템 개선
1. **CUDA 지원 PyTorch 설치**
2. **WSL 메모리 설정 최적화**
3. **메모리 모니터링 도구 추가**

### 우선순위 3: 아키텍처 개선
1. **모델 파이프라인 병렬화**
2. **점진적 학습 구현**
3. **동적 메모리 관리 시스템**

## 🔍 모니터링 방안

### 실시간 메모리 모니터링
```python
import psutil
import torch

def monitor_memory():
    ram_usage = psutil.virtual_memory().used / 1024**3
    if torch.cuda.is_available():
        gpu_usage = torch.cuda.memory_allocated() / 1024**3
    print(f"RAM: {ram_usage:.2f}GB, GPU: {gpu_usage:.2f}GB")
```

### WSL 안전 장치
```python
def safe_model_creation(config):
    estimated_memory = estimate_model_memory(config)
    available_memory = psutil.virtual_memory().available / 1024**3
    
    if estimated_memory > available_memory * 0.8:
        raise MemoryError("모델이 메모리 한계를 초과합니다")
    
    return create_model(config)
```

## 📝 결론

WSL에서 하이브리드 학습 시스템의 강제 종료는 **메모리 사용량 초과**가 근본 원인입니다. 현재 모델이 WSL 메모리 한계(62GB)의 1.6배인 101.94GB를 요구하여 시스템이 보호 목적으로 강제 종료됩니다.

**즉시 모델 크기를 1/3로 축소**하고 **배치 크기를 최소화**하면 문제를 해결할 수 있습니다. 장기적으로는 GPU 활용과 메모리 최적화를 통해 성능을 회복할 수 있습니다.

---

*분석 일시: 2025-06-25*  
*분석 대상: `/mnt/c/large_project/linux_red_heart/training/` 하이브리드 학습 시스템*