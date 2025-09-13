#!/usr/bin/env python3
"""
Unified Memory Manager - 통합 메모리 관리 시스템

3개의 분산된 메모리 시스템을 통합:
- SystemSwapManager (Local LLM용)
- DynamicSwapManager (API 모드용)  
- DirectGPUManager (Claude 모드용)

DSM 철학 적용으로 통일된 인터페이스 제공
"""

import torch
import gc
import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import psutil
import threading

logger = logging.getLogger(__name__)


class MemoryMode(Enum):
    """메모리 관리 모드"""
    SYSTEM_SWAP = "system_swap"    # Local LLM 모드
    DYNAMIC_SWAP = "dynamic_swap"   # API 모드
    DIRECT_GPU = "direct_gpu"       # Claude 모드
    UNIFIED = "unified"             # 통합 모드 (새로운)


class SwapPriority(Enum):
    """모델 스왑 우선순위"""
    CRITICAL = 1  # 절대 스왑 안함
    HIGH = 2      # 가능한 유지
    MEDIUM = 3    # 필요시 스왑
    LOW = 4       # 적극적 스왑


@dataclass
class ModelInfo:
    """모델 정보"""
    name: str
    model: Any
    size_mb: float
    priority: SwapPriority
    device: str = "cpu"
    last_used: float = field(default_factory=time.time)
    usage_count: int = 0
    is_locked: bool = False  # GPU 점유 잠금


@dataclass
class MemoryState:
    """메모리 상태"""
    gpu_total_mb: float
    gpu_used_mb: float
    gpu_free_mb: float
    ram_total_mb: float
    ram_used_mb: float
    ram_free_mb: float
    swap_total_mb: float
    swap_used_mb: float
    models_in_gpu: List[str]
    models_in_ram: List[str]


class UnifiedMemoryManager:
    """
    통합 메모리 관리자
    
    특징:
    - 3개 시스템 통합 인터페이스
    - DSM 철학 적용 (동기적 스왑)
    - 자동 모드 전환
    - 통계 및 모니터링
    """
    
    def __init__(self, mode: MemoryMode = MemoryMode.UNIFIED):
        """
        Args:
            mode: 메모리 관리 모드
        """
        self.mode = mode
        self.models: Dict[str, ModelInfo] = {}
        self.gpu_lock = threading.Lock()
        self.stats = {
            'swaps_to_gpu': 0,
            'swaps_to_ram': 0,
            'cache_clears': 0,
            'oom_recoveries': 0
        }
        
        # GPU 가용성 확인
        self.has_gpu = torch.cuda.is_available()
        if self.has_gpu:
            self.device = torch.device('cuda')
            self.gpu_properties = torch.cuda.get_device_properties(0)
            self.gpu_total_mb = self.gpu_properties.total_memory / 1024 / 1024
            logger.info(f"GPU 감지: {self.gpu_properties.name} ({self.gpu_total_mb:.0f}MB)")
        else:
            self.device = torch.device('cpu')
            self.gpu_total_mb = 0
            logger.warning("GPU 사용 불가 - CPU 모드로 실행")
            
        # 메모리 임계값 설정
        self.gpu_threshold_mb = self.gpu_total_mb * 0.85 if self.has_gpu else 0
        self.ram_threshold_mb = psutil.virtual_memory().total / 1024 / 1024 * 0.8
        
        logger.info(f"UnifiedMemoryManager 초기화 (모드: {mode.value})")
        logger.info(f"GPU 임계값: {self.gpu_threshold_mb:.0f}MB")
        logger.info(f"RAM 임계값: {self.ram_threshold_mb:.0f}MB")
        
    def register_model(self, name: str, model: Any, 
                      priority: SwapPriority = SwapPriority.MEDIUM,
                      estimated_size_mb: Optional[float] = None) -> bool:
        """모델 등록
        
        Args:
            name: 모델 이름
            model: 모델 객체
            priority: 스왑 우선순위
            estimated_size_mb: 예상 크기 (MB)
            
        Returns:
            등록 성공 여부
        """
        try:
            # 크기 추정
            if estimated_size_mb is None:
                estimated_size_mb = self._estimate_model_size(model)
                
            # 현재 디바이스 확인
            current_device = "cpu"
            if hasattr(model, 'device'):
                current_device = str(model.device)
            elif hasattr(model, 'parameters'):
                try:
                    param = next(model.parameters())
                    current_device = str(param.device)
                except StopIteration:
                    pass
                    
            model_info = ModelInfo(
                name=name,
                model=model,
                size_mb=estimated_size_mb,
                priority=priority,
                device=current_device
            )
            
            self.models[name] = model_info
            logger.info(f"모델 등록: {name} ({estimated_size_mb:.0f}MB, {current_device})")
            return True
            
        except Exception as e:
            logger.error(f"모델 등록 실패 ({name}): {e}")
            return False
            
    def _estimate_model_size(self, model: Any) -> float:
        """모델 크기 추정 (MB)"""
        total_params = 0
        
        if hasattr(model, 'parameters'):
            for param in model.parameters():
                total_params += param.numel()
                
        # 파라미터당 4바이트 (float32) + 그래디언트/버퍼 여유분
        size_mb = (total_params * 4 * 1.5) / (1024 * 1024)
        return max(size_mb, 10)  # 최소 10MB
        
    async def request_gpu(self, model_name: str, timeout: float = 30.0) -> bool:
        """GPU 메모리 요청 (DSM 동기적 스왑)
        
        Args:
            model_name: 모델 이름
            timeout: 대기 시간
            
        Returns:
            GPU 할당 성공 여부
        """
        if not self.has_gpu:
            logger.warning(f"GPU 없음 - {model_name} CPU 유지")
            return False
            
        if model_name not in self.models:
            logger.error(f"등록되지 않은 모델: {model_name}")
            return False
            
        model_info = self.models[model_name]
        
        # 이미 GPU에 있으면 성공
        if 'cuda' in model_info.device:
            model_info.last_used = time.time()
            model_info.usage_count += 1
            return True
            
        # GPU 메모리 확보
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.gpu_lock:
                if self._ensure_gpu_memory(model_info.size_mb):
                    # GPU로 이동
                    try:
                        model_info.model = model_info.model.to(self.device)
                        model_info.device = str(self.device)
                        model_info.last_used = time.time()
                        model_info.usage_count += 1
                        self.stats['swaps_to_gpu'] += 1
                        
                        logger.info(f"✅ {model_name} GPU 할당 성공")
                        return True
                        
                    except RuntimeError as e:
                        if 'out of memory' in str(e):
                            logger.warning(f"GPU OOM - 재시도: {model_name}")
                            self._handle_gpu_oom()
                        else:
                            raise
                            
            # 재시도 전 대기
            await asyncio.sleep(0.5)
            
        logger.error(f"GPU 할당 실패 (타임아웃): {model_name}")
        return False
        
    def _ensure_gpu_memory(self, required_mb: float) -> bool:
        """GPU 메모리 확보
        
        Args:
            required_mb: 필요한 메모리 (MB)
            
        Returns:
            메모리 확보 성공 여부
        """
        if not self.has_gpu:
            return False
            
        current_state = self.get_memory_state()
        
        # 여유 메모리 충분하면 성공
        if current_state.gpu_free_mb >= required_mb * 1.2:  # 20% 여유
            return True
            
        # 스왑 가능한 모델 찾기
        swappable = [
            (name, info) for name, info in self.models.items()
            if 'cuda' in info.device and not info.is_locked
        ]
        
        # 우선순위와 사용 시간 기준 정렬
        swappable.sort(key=lambda x: (x[1].priority.value, -x[1].last_used))
        
        # 필요한 만큼 스왑
        freed_mb = 0
        for name, info in swappable:
            if freed_mb >= required_mb:
                break
                
            # RAM으로 스왑
            try:
                info.model = info.model.cpu()
                info.device = "cpu"
                freed_mb += info.size_mb
                self.stats['swaps_to_ram'] += 1
                logger.info(f"스왑: {name} → RAM ({info.size_mb:.0f}MB 확보)")
                
            except Exception as e:
                logger.error(f"스왑 실패 ({name}): {e}")
                
        # 캐시 정리
        self.clear_gpu_cache()
        
        # 다시 확인
        current_state = self.get_memory_state()
        return current_state.gpu_free_mb >= required_mb
        
    def release_gpu(self, model_name: str) -> bool:
        """GPU 메모리 해제
        
        Args:
            model_name: 모델 이름
            
        Returns:
            해제 성공 여부
        """
        if model_name not in self.models:
            return False
            
        model_info = self.models[model_name]
        
        if 'cuda' not in model_info.device:
            return True  # 이미 CPU에 있음
            
        try:
            with self.gpu_lock:
                model_info.model = model_info.model.cpu()
                model_info.device = "cpu"
                self.stats['swaps_to_ram'] += 1
                self.clear_gpu_cache()
                
            logger.info(f"GPU 해제: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"GPU 해제 실패 ({model_name}): {e}")
            return False
            
    def lock_model(self, model_name: str):
        """모델 잠금 (스왑 방지)"""
        if model_name in self.models:
            self.models[model_name].is_locked = True
            
    def unlock_model(self, model_name: str):
        """모델 잠금 해제"""
        if model_name in self.models:
            self.models[model_name].is_locked = False
            
    def clear_gpu_cache(self):
        """GPU 캐시 정리"""
        if self.has_gpu:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            self.stats['cache_clears'] += 1
            
    def _handle_gpu_oom(self):
        """GPU OOM 처리"""
        logger.warning("GPU OOM 처리 중...")
        
        # 모든 잠금 해제
        for info in self.models.values():
            info.is_locked = False
            
        # 우선순위 낮은 모델부터 스왑
        models_on_gpu = [
            (name, info) for name, info in self.models.items()
            if 'cuda' in info.device
        ]
        
        models_on_gpu.sort(key=lambda x: (x[1].priority.value, -x[1].last_used))
        
        # 절반 스왑
        swap_count = max(1, len(models_on_gpu) // 2)
        for name, info in models_on_gpu[:swap_count]:
            try:
                info.model = info.model.cpu()
                info.device = "cpu"
                logger.info(f"OOM 스왑: {name} → RAM")
            except:
                pass
                
        self.clear_gpu_cache()
        self.stats['oom_recoveries'] += 1
        
    def get_memory_state(self) -> MemoryState:
        """현재 메모리 상태 조회"""
        # GPU 메모리
        gpu_used = 0
        gpu_free = self.gpu_total_mb
        
        if self.has_gpu:
            gpu_used = torch.cuda.memory_allocated() / 1024 / 1024
            gpu_free = self.gpu_total_mb - gpu_used
            
        # RAM 메모리
        ram_info = psutil.virtual_memory()
        ram_total = ram_info.total / 1024 / 1024
        ram_used = ram_info.used / 1024 / 1024
        ram_free = ram_info.available / 1024 / 1024
        
        # 스왑 메모리
        swap_info = psutil.swap_memory()
        swap_total = swap_info.total / 1024 / 1024
        swap_used = swap_info.used / 1024 / 1024
        
        # 모델 위치
        models_in_gpu = [
            name for name, info in self.models.items()
            if 'cuda' in info.device
        ]
        
        models_in_ram = [
            name for name, info in self.models.items()
            if 'cpu' in info.device
        ]
        
        return MemoryState(
            gpu_total_mb=self.gpu_total_mb,
            gpu_used_mb=gpu_used,
            gpu_free_mb=gpu_free,
            ram_total_mb=ram_total,
            ram_used_mb=ram_used,
            ram_free_mb=ram_free,
            swap_total_mb=swap_total,
            swap_used_mb=swap_used,
            models_in_gpu=models_in_gpu,
            models_in_ram=models_in_ram
        )
        
    def get_stats(self) -> Dict:
        """통계 조회"""
        state = self.get_memory_state()
        return {
            **self.stats,
            'total_models': len(self.models),
            'models_in_gpu': len(state.models_in_gpu),
            'models_in_ram': len(state.models_in_ram),
            'gpu_usage_percent': (state.gpu_used_mb / self.gpu_total_mb * 100) 
                                if self.gpu_total_mb > 0 else 0
        }
        
    async def optimize_layout(self):
        """메모리 레이아웃 최적화
        
        우선순위와 사용 패턴에 따라 모델 재배치
        """
        logger.info("메모리 레이아웃 최적화 시작...")
        
        # 사용 빈도순 정렬
        sorted_models = sorted(
            self.models.items(),
            key=lambda x: (-x[1].usage_count, x[1].priority.value)
        )
        
        # GPU 공간 계산
        available_gpu = self.gpu_threshold_mb
        
        # 우선순위 높은 모델부터 GPU 할당
        for name, info in sorted_models:
            if available_gpu >= info.size_mb:
                if 'cpu' in info.device:
                    success = await self.request_gpu(name)
                    if success:
                        available_gpu -= info.size_mb
            else:
                # GPU 공간 부족하면 RAM으로
                if 'cuda' in info.device and info.priority == SwapPriority.LOW:
                    self.release_gpu(name)
                    
        logger.info("메모리 레이아웃 최적화 완료")
        
    def __str__(self) -> str:
        """상태 문자열"""
        state = self.get_memory_state()
        stats = self.get_stats()
        
        return f"""
UnifiedMemoryManager 상태
========================
모드: {self.mode.value}
모델: {stats['total_models']}개 (GPU: {stats['models_in_gpu']}, RAM: {stats['models_in_ram']})
GPU: {state.gpu_used_mb:.0f}/{self.gpu_total_mb:.0f}MB ({stats['gpu_usage_percent']:.1f}%)
RAM: {state.ram_used_mb:.0f}/{state.ram_total_mb:.0f}MB
스왑: GPU→RAM {stats['swaps_to_ram']}회, RAM→GPU {stats['swaps_to_gpu']}회
OOM 복구: {stats['oom_recoveries']}회
"""


# 싱글톤 인스턴스
_instance: Optional[UnifiedMemoryManager] = None
_lock = threading.Lock()


def get_unified_memory_manager(mode: MemoryMode = MemoryMode.UNIFIED) -> UnifiedMemoryManager:
    """싱글톤 인스턴스 조회"""
    global _instance
    
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = UnifiedMemoryManager(mode)
                
    return _instance


# 테스트 코드
if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(level=logging.INFO)
    
    async def test_manager():
        """메모리 관리자 테스트"""
        manager = get_unified_memory_manager()
        
        # 더미 모델 생성
        class DummyModel:
            def __init__(self, size):
                self.weight = torch.randn(size, size)
                
            def to(self, device):
                self.weight = self.weight.to(device)
                return self
                
            def cpu(self):
                self.weight = self.weight.cpu()
                return self
                
            @property
            def device(self):
                return self.weight.device
                
        # 모델 등록
        model1 = DummyModel(1000)
        model2 = DummyModel(2000)
        model3 = DummyModel(1500)
        
        manager.register_model("model1", model1, SwapPriority.HIGH)
        manager.register_model("model2", model2, SwapPriority.MEDIUM)
        manager.register_model("model3", model3, SwapPriority.LOW)
        
        # GPU 요청 테스트
        print("GPU 요청 테스트...")
        await manager.request_gpu("model1")
        await manager.request_gpu("model2")
        
        # 상태 출력
        print(manager)
        
        # 레이아웃 최적화
        await manager.optimize_layout()
        
        # 최종 상태
        print(manager)
        
    asyncio.run(test_manager())