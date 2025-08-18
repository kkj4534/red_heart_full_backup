"""
워크플로우 인식 메모리 관리자 - Red Heart AI
Workflow-Aware Memory Manager for Red Heart AI

중요한 워크플로우의 연계성을 이해하고 보호하는 지능적 메모리 관리
"""

import torch
import torch.nn as nn
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
from datetime import datetime
import weakref
from collections import defaultdict, deque
import gc
import time

logger = logging.getLogger(__name__)

class WorkflowStage(Enum):
    """워크플로우 단계"""
    INITIALIZATION = "initialization"
    DATA_LOADING = "data_loading"
    BACKBONE_FORWARD = "backbone_forward"
    HEAD_PROCESSING = "head_processing"
    SYNERGY_COMPUTATION = "synergy_computation"
    LOSS_COMPUTATION = "loss_computation"
    BACKWARD_PASS = "backward_pass"
    OPTIMIZATION = "optimization"
    EVALUATION = "evaluation"
    FINALIZATION = "finalization"

class TaskDependency:
    """태스크 의존성 정의"""
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.dependencies: Set[str] = set()
        self.dependents: Set[str] = set()
        self.required_models: Set[str] = set()
        self.produces_data: Set[str] = set()
        self.consumes_data: Set[str] = set()
        
    def add_dependency(self, dep_task_id: str):
        """의존성 추가"""
        self.dependencies.add(dep_task_id)
        
    def add_dependent(self, dep_task_id: str):
        """이 태스크에 의존하는 태스크 추가"""
        self.dependents.add(dep_task_id)

@dataclass
class WorkflowUnit:
    """워크플로우 실행 단위"""
    unit_id: str
    stage: WorkflowStage
    tasks: List[str] = field(default_factory=list)
    required_models: Set[str] = field(default_factory=set)
    active_models: Set[str] = field(default_factory=set)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    estimated_memory_mb: float = 0.0
    actual_memory_mb: float = 0.0
    is_active: bool = False
    is_protected: bool = False  # 언로드 보호
    
    def add_task(self, task_id: str, required_models: Set[str]):
        """태스크 추가"""
        self.tasks.append(task_id)
        self.required_models.update(required_models)
        
    def activate(self):
        """워크플로우 단위 활성화"""
        self.is_active = True
        self.is_protected = True
        self.start_time = datetime.now()
        
    def deactivate(self):
        """워크플로우 단위 비활성화"""
        self.is_active = False
        self.is_protected = False
        self.end_time = datetime.now()

class WorkflowTracker:
    """워크플로우 추적기"""
    
    def __init__(self):
        self.current_stage = WorkflowStage.INITIALIZATION
        self.active_units: Dict[str, WorkflowUnit] = {}
        self.completed_units: deque = deque(maxlen=100)
        self.task_dependencies: Dict[str, TaskDependency] = {}
        self.stage_history: deque = deque(maxlen=50)
        self.model_usage_map: Dict[str, Set[str]] = defaultdict(set)  # model -> units
        
    def register_task_dependency(self, task_id: str, dependencies: List[str], 
                                required_models: List[str]):
        """태스크 의존성 등록"""
        task_dep = TaskDependency(task_id)
        task_dep.required_models.update(required_models)
        
        for dep in dependencies:
            task_dep.add_dependency(dep)
            if dep in self.task_dependencies:
                self.task_dependencies[dep].add_dependent(task_id)
                
        self.task_dependencies[task_id] = task_dep
        
    def start_workflow_unit(self, unit_id: str, stage: WorkflowStage, 
                           required_models: Set[str]) -> WorkflowUnit:
        """워크플로우 단위 시작"""
        unit = WorkflowUnit(
            unit_id=unit_id,
            stage=stage,
            required_models=required_models
        )
        unit.activate()
        
        self.active_units[unit_id] = unit
        self.current_stage = stage
        self.stage_history.append((stage, datetime.now()))
        
        # 모델 사용 맵 업데이트
        for model in required_models:
            self.model_usage_map[model].add(unit_id)
            
        logger.info(f"워크플로우 단위 시작: {unit_id} (단계: {stage.value})")
        return unit
        
    def complete_workflow_unit(self, unit_id: str):
        """워크플로우 단위 완료"""
        if unit_id in self.active_units:
            unit = self.active_units[unit_id]
            unit.deactivate()
            
            # 모델 사용 맵 정리
            for model in unit.required_models:
                self.model_usage_map[model].discard(unit_id)
                
            self.completed_units.append(unit)
            del self.active_units[unit_id]
            
            logger.info(f"워크플로우 단위 완료: {unit_id}")
            
    def get_protected_models(self) -> Set[str]:
        """보호되어야 할 모델 목록 반환"""
        protected_models = set()
        
        for unit in self.active_units.values():
            if unit.is_protected:
                protected_models.update(unit.required_models)
                
        return protected_models
        
    def predict_next_stage(self) -> Optional[WorkflowStage]:
        """다음 워크플로우 단계 예측"""
        stage_transitions = {
            WorkflowStage.INITIALIZATION: WorkflowStage.DATA_LOADING,
            WorkflowStage.DATA_LOADING: WorkflowStage.BACKBONE_FORWARD,
            WorkflowStage.BACKBONE_FORWARD: WorkflowStage.HEAD_PROCESSING,
            WorkflowStage.HEAD_PROCESSING: WorkflowStage.SYNERGY_COMPUTATION,
            WorkflowStage.SYNERGY_COMPUTATION: WorkflowStage.LOSS_COMPUTATION,
            WorkflowStage.LOSS_COMPUTATION: WorkflowStage.BACKWARD_PASS,
            WorkflowStage.BACKWARD_PASS: WorkflowStage.OPTIMIZATION,
            WorkflowStage.OPTIMIZATION: WorkflowStage.EVALUATION,
            WorkflowStage.EVALUATION: WorkflowStage.DATA_LOADING,  # 다음 배치
        }
        
        return stage_transitions.get(self.current_stage)

class WorkflowAwareMemoryManager:
    """워크플로우 인식 메모리 관리자"""
    
    def __init__(self, memory_threshold_mb: float = 6500.0):  # ~6.5GB for 8GB GPU
        self.memory_threshold_mb = memory_threshold_mb
        self.workflow_tracker = WorkflowTracker()
        self.model_registry: Dict[str, Tuple[nn.Module, float]] = {}  # name -> (model, size_mb)
        self.gpu_models: Set[str] = set()
        self.swap_history: deque = deque(maxlen=1000)
        self.memory_predictions: deque = deque(maxlen=100)
        self.memory_allocations: Dict[str, Tuple[float, float]] = {}  # module -> (size_mb, timestamp)
        
        # 메인 루프 참조
        self.main_loop = None  # 오케스트레이터가 주입할 메인 이벤트 루프
        
        # 워크플로우별 메모리 프로파일
        self.workflow_memory_profiles = {
            WorkflowStage.BACKBONE_FORWARD: 1500.0,  # MB
            WorkflowStage.HEAD_PROCESSING: 2000.0,
            WorkflowStage.SYNERGY_COMPUTATION: 500.0,
            WorkflowStage.BACKWARD_PASS: 2500.0,
            WorkflowStage.OPTIMIZATION: 1000.0,
        }
        
        # GPU 모델 레지스트리 (HF 래퍼와 공유)
        self._gpu_models = {}  # model_id -> {model, size_mb, owner, last_used}
        
        # HF 래퍼 통합
        try:
            from hf_model_wrapper import get_hf_wrapper
            hf_wrapper = get_hf_wrapper()
            hf_wrapper.set_memory_manager(self)
            logger.info("✅ HF 모델 래퍼와 메모리 매니저 통합 완료")
        except ImportError:
            logger.warning("⚠️ HF 모델 래퍼를 찾을 수 없음 - 수동 모델 추적 모드")
        
    def register_model(self, name: str, model: nn.Module):
        """모델 등록"""
        size_mb = self._calculate_model_size(model)
        self.model_registry[name] = (model, size_mb)
        logger.info(f"모델 등록: {name} ({size_mb:.1f}MB)")
        
    def _calculate_model_size(self, model: nn.Module) -> float:
        """모델 크기 계산 (MB)"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.numel() * param.element_size()
            
        for buffer in model.buffers():
            buffer_size += buffer.numel() * buffer.element_size()
            
        total_size_mb = (param_size + buffer_size) / (1024 * 1024)
        return total_size_mb
        
    def get_current_gpu_usage(self) -> float:
        """현재 GPU 메모리 사용량 (MB)"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            return allocated
        return 0.0
        
    def predict_memory_requirement(self, stage: WorkflowStage, 
                                 required_models: Set[str]) -> float:
        """메모리 요구사항 예측"""
        base_memory = self.workflow_memory_profiles.get(stage, 500.0)
        
        model_memory = 0.0
        for model_name in required_models:
            if model_name in self.model_registry:
                _, size_mb = self.model_registry[model_name]
                model_memory += size_mb
                
        total_requirement = base_memory + model_memory
        
        # 히스토리 기반 보정
        if len(self.memory_predictions) > 10:
            recent_errors = []
            for pred in list(self.memory_predictions)[-10:]:
                if pred['stage'] == stage:
                    error = pred['actual'] - pred['predicted']
                    recent_errors.append(error)
                    
            if recent_errors:
                avg_error = sum(recent_errors) / len(recent_errors)
                total_requirement += avg_error
                
        return total_requirement
        
    async def prepare_for_workflow(self, unit_id: str, stage: WorkflowStage,
                                  required_models: Set[str]) -> bool:
        """워크플로우를 위한 메모리 준비"""
        logger.info(f"워크플로우 준비: {unit_id} (단계: {stage.value})")
        
        # 현재 GPU 사용량 확인
        current_usage = self.get_current_gpu_usage()
        required_memory = self.predict_memory_requirement(stage, required_models)
        
        logger.info(f"현재 GPU 사용량: {current_usage:.1f}MB, "
                   f"예상 필요량: {required_memory:.1f}MB, "
                   f"임계값: {self.memory_threshold_mb:.1f}MB")
        
        # 메모리 부족 예상시 스왑 준비
        if current_usage + required_memory > self.memory_threshold_mb:
            await self._prepare_memory_space(required_memory)
            
        # 워크플로우 단위 시작
        unit = self.workflow_tracker.start_workflow_unit(unit_id, stage, required_models)
        
        # 필요한 모델들 GPU로 로드
        for model_name in required_models:
            if model_name not in self.gpu_models:
                await self._load_model_to_gpu(model_name)
                
        return True
        
    async def _prepare_memory_space(self, required_mb: float):
        """필요한 메모리 공간 확보"""
        protected_models = self.workflow_tracker.get_protected_models()
        current_usage = self.get_current_gpu_usage()
        
        # 언로드 가능한 모델 찾기
        unloadable_models = []
        for model_name in self.gpu_models:
            if model_name not in protected_models:
                _, size_mb = self.model_registry.get(model_name, (None, 0))
                if size_mb > 0:
                    # 모델 사용 빈도 확인
                    usage_count = len(self.workflow_tracker.model_usage_map.get(model_name, set()))
                    unloadable_models.append((model_name, size_mb, usage_count))
                    
        # 사용 빈도가 낮은 순으로 정렬
        unloadable_models.sort(key=lambda x: (x[2], -x[1]))
        
        # 필요한 만큼 언로드
        freed_memory = 0.0
        for model_name, size_mb, _ in unloadable_models:
            if current_usage - freed_memory + required_mb < self.memory_threshold_mb:
                break
                
            await self._unload_model_from_gpu(model_name)
            freed_memory += size_mb
            
        logger.info(f"메모리 확보 완료: {freed_memory:.1f}MB 해제")
        
    async def _load_model_to_gpu(self, model_name: str):
        """모델을 GPU로 로드"""
        if model_name in self.model_registry:
            model, size_mb = self.model_registry[model_name]
            model.cuda()
            self.gpu_models.add(model_name)
            logger.debug(f"모델 GPU 로드: {model_name} ({size_mb:.1f}MB)")
            
    async def _unload_model_from_gpu(self, model_name: str):
        """모델을 GPU에서 언로드"""
        if model_name in self.gpu_models:
            model, size_mb = self.model_registry[model_name]
            model.cpu()
            self.gpu_models.remove(model_name)
            
            # 즉시 GPU 메모리 정리
            torch.cuda.empty_cache()
            
            logger.debug(f"모델 GPU 언로드: {model_name} ({size_mb:.1f}MB)")
            
    def complete_workflow(self, unit_id: str):
        """워크플로우 완료"""
        self.workflow_tracker.complete_workflow_unit(unit_id)
        
    def get_workflow_status(self) -> Dict[str, Any]:
        """워크플로우 상태 반환"""
        return {
            'current_stage': self.workflow_tracker.current_stage.value,
            'active_units': len(self.workflow_tracker.active_units),
            'protected_models': list(self.workflow_tracker.get_protected_models()),
            'gpu_models': list(self.gpu_models),
            'gpu_usage_mb': self.get_current_gpu_usage(),
            'memory_threshold_mb': self.memory_threshold_mb
        }
        
    async def optimize_for_next_stage(self):
        """다음 단계를 위한 최적화"""
        next_stage = self.workflow_tracker.predict_next_stage()
        if next_stage:
            logger.info(f"다음 단계 예측: {next_stage.value}")
            
            # 다음 단계에서 필요할 모델 예측 (히스토리 기반)
            # 실제 구현에서는 더 정교한 예측 로직 필요
            
    def emergency_cleanup(self):
        """긴급 메모리 정리"""
        logger.warning("긴급 메모리 정리 시작!")
        
        # 보호되지 않은 모든 모델 언로드
        protected_models = self.workflow_tracker.get_protected_models()
        models_to_unload = [m for m in self.gpu_models if m not in protected_models]
        
        for model_name in models_to_unload:
            model, _ = self.model_registry[model_name]
            model.cpu()
            self.gpu_models.remove(model_name)
            
        torch.cuda.empty_cache()
        gc.collect()
        
        freed_memory = self.get_current_gpu_usage()
        logger.warning(f"긴급 정리 완료: {len(models_to_unload)}개 모델 언로드")
        
    def _estimate_bundle_mb(self, module_name: str, deps: List[str]) -> float:
        """
        모듈과 의존성의 총 메모리 크기 추정 (NO FALLBACK - 실측치만 사용)
        
        Args:
            module_name: 모듈 이름
            deps: 의존성 모듈 리스트
            
        Returns:
            float: 추정 메모리 크기 (MB)
        """
        total_mb = 0.0
        
        # NO FALLBACK - MODULE_SPECS의 추정치 사용 금지
        # DSM에 등록된 실제 크기 확인
        from dynamic_swap_manager import get_swap_manager
        swap_manager = get_swap_manager()
        
        if swap_manager and module_name in swap_manager.models:
            actual_size = swap_manager.models[module_name].size_mb
            if actual_size > 0:
                total_mb += actual_size
                logger.debug(f"  {module_name}: {actual_size}MB (DSM 실측)")
        
        # 의존성 모듈들의 실제 크기
        for dep in deps:
            if swap_manager and dep in swap_manager.models:
                dep_size = swap_manager.models[dep].size_mb
                if dep_size > 0:
                    total_mb += dep_size
                    logger.debug(f"  의존성 {dep}: {dep_size}MB (DSM 실측)")
        
        # 2. HF 래퍼 실측치 반영
        if hasattr(self, '_gpu_models'):
            for model_id, model_info in self._gpu_models.items():
                if (model_info.get('owner') == module_name or 
                    model_info.get('owner') in deps):
                    actual_mb = model_info.get('size_mb', 0)
                    # 실측치가 있으면 스펙보다 우선 (90% 적용)
                    total_mb = max(total_mb, actual_mb * 0.9)
                    logger.debug(f"  {model_id}: {actual_mb}MB (실측, owner={model_info.get('owner')})")
        
        # 3. 오버헤드 버퍼 추가
        overhead_mb = max(100, 0.05 * total_mb)
        total_mb += overhead_mb
        
        logger.info(f"[GPU MEM] 번들 크기 추정: {module_name} + deps = {total_mb:.1f}MB (오버헤드 {overhead_mb:.1f}MB 포함)")
        return total_mb

    async def request_gpu(self, module_name: str, required_mb: float = None, 
                         deps: List[str] = None, target_util: float = 0.85,
                         must_succeed: bool = True, timeout: float = 30.0) -> bool:
        """
        GPU 공간 요청 - 의존성 단위 승인
        
        Args:
            module_name: 모듈 이름
            required_mb: 필요한 메모리 (MB), None이면 자동 추정
            deps: 의존성 모듈 리스트
            target_util: 목표 GPU 사용률 (0.85 = 85%)
            
        Returns:
            bool: 할당 성공 여부
        """
        from config import get_gpu_memory_info
        
        deps = deps or []
        
        # required_mb가 없으면 자동 추정
        if required_mb is None:
            required_mb = self._estimate_bundle_mb(module_name, deps)
        else:
            # 의존성이 있으면 번들 크기 계산
            if deps:
                bundle_mb = self._estimate_bundle_mb(module_name, deps)
                required_mb = max(required_mb, bundle_mb)
        
        # 현재 GPU 사용률 확인 (empty_cache 후 측정)
        torch.cuda.empty_cache()
        gpu_info = get_gpu_memory_info()
        if not gpu_info:
            logger.warning("[GPU MEM] GPU 정보를 가져올 수 없습니다")
            return False
            
        current_usage = gpu_info['usage_percent'] / 100.0
        total_mb = gpu_info['total_mb']
        
        # reserved 메모리도 고려한 실제 여유 공간 계산
        actual_free = gpu_info['free_mb']
        reserved_free = max(0, gpu_info['cached_mb'] - gpu_info['allocated_mb'])
        total_available = actual_free + reserved_free
        
        logger.info(f"[GPU QUEUE] {module_name}: {required_mb:.1f}MB 요청 (의존성 {len(deps)}개)")
        logger.info(f"[GPU MEM] 현재: {current_usage*100:.1f}% 사용, {total_available:.1f}MB 실제 여유")
        
        # 공간이 부족하면 free_gpu_space_intelligent 호출
        if total_available < required_mb:
            logger.info(f"[GPU MEM] 공간 부족 - {required_mb:.1f}MB 필요, {total_available:.1f}MB 여유")
            
            # DynamicSwapManager의 지능적 공간 확보 시도
            from dynamic_swap_manager import get_swap_manager
            swap_manager = get_swap_manager()
            if swap_manager:
                success = await swap_manager.free_gpu_space_intelligent(
                    required_mb=required_mb,
                    target_usage=target_util,
                    hard_timeout_s=timeout
                )
                if not success:
                    logger.error(f"[GPU MEM] 공간 확보 실패: {module_name}")
                    if must_succeed:
                        raise RuntimeError(f"[GPU] 필수 모듈 공간 확보 실패(owner={module_name}, need≈{required_mb}MB, target={target_util})")
                    return False
                logger.info(f"[GPU MEM] 공간 확보 성공: {required_mb:.1f}MB")
            else:
                logger.error("[GPU MEM] SwapManager를 찾을 수 없습니다")
                return False
        
        # 메모리 할당 기록
        self.memory_allocations[module_name] = (required_mb, time.time())
        logger.info(f"[GPU MEM] ✅ {module_name} 할당 완료: {required_mb:.1f}MB")
        return True
        
    def set_boot_completed(self):
        """부팅 완료 표시 - deprecated, 호환성을 위해 빈 함수로 유지"""
        logger.info("🚀 set_boot_completed 호출됨 (deprecated)")
    
    def request_gpu_blocking(self, module_name: str, required_mb: float = None,
                           deps: List[str] = None, target_util: float = 0.85,
                           timeout: float = 30.0, is_required: bool = False) -> bool:
        """
        동기 GPU 메모리 요청 - 메모리가 확보될 때까지 대기
        
        Args:
            module_name: 모듈 이름
            required_mb: 필요한 메모리 (MB)
            deps: 의존성 모듈 리스트
            target_util: 목표 GPU 사용률
            timeout: 최대 대기 시간 (초)
            is_required: 필수 모듈 여부 (True면 실패 시 예외 발생)
            
        Returns:
            bool: 할당 성공 여부
        """
        import asyncio
        from config import get_gpu_memory_info
        
        # GPU 할당 정책 검사 (부팅 플래그 제거)
        
        start_time = time.time()
        deps = deps or []
        
        # required_mb가 없으면 추정
        if required_mb is None:
            required_mb = self._estimate_bundle_mb(module_name, deps)
            
        logger.info(f"[GPU BLOCKING] {module_name}: {required_mb:.1f}MB 동기 요청 시작")
        
        # 이벤트 루프 처리
        try:
            # 스레드 안전성: main_loop를 사용하거나, 현재 실행 중인 루프 확인
            loop = None
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # 스레드에서 실행 중이면 main_loop 사용
                loop = self.main_loop
            
            if loop and loop.is_running():
                # 이미 루프가 실행 중이면 폴링 방식 사용
                while time.time() - start_time < timeout:
                    # empty_cache 호출 후 재측정으로 실제 여유 공간 확인
                    torch.cuda.empty_cache()
                    gpu_info = get_gpu_memory_info()
                    
                    if gpu_info:
                        # reserved 메모리도 고려한 실제 여유 공간 계산
                        # empty_cache 후이므로 reserved-allocated가 실제 사용 가능한 공간
                        actual_free = gpu_info['free_mb']
                        reserved_free = max(0, gpu_info['cached_mb'] - gpu_info['allocated_mb'])
                        total_available = actual_free + reserved_free
                        
                        logger.debug(f"[GPU BLOCKING] 여유 공간: free={actual_free:.1f}MB, reserved_free={reserved_free:.1f}MB, total={total_available:.1f}MB")
                        
                        if total_available >= required_mb:
                            # 메모리가 충분하면 할당 기록하고 성공 반환
                            self.memory_allocations[module_name] = (required_mb, time.time())
                            logger.info(f"[GPU BLOCKING] ✅ {module_name} 할당 성공 (폴링)")
                            return True
                    
                    # 50ms 대기
                    time.sleep(0.05)
                    
                    # 주기적으로 공간 확보 시도
                    if int((time.time() - start_time) * 10) % 10 == 0:  # 1초마다
                        from dynamic_swap_manager import get_swap_manager
                        swap_manager = get_swap_manager()
                        if swap_manager:
                            # 동기적으로 공간 확보 시도
                            logger.debug(f"[GPU BLOCKING] 공간 확보 시도...")
                            # TODO: free_gpu_space_intelligent를 동기 버전으로 호출
                
                # 타임아웃
                msg = f"[GPU BLOCKING] ❌ {module_name} 할당 실패: 타임아웃 ({timeout}초)"
                logger.error(msg)
                if is_required:
                    raise RuntimeError(msg)
                return False
                
            else:
                # 루프가 없거나 실행 중이 아닐 때
                if loop is None:
                    # 새 이벤트 루프 생성
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        success = loop.run_until_complete(
                            self.request_gpu(module_name, required_mb, deps, target_util, 
                                        must_succeed=is_required, timeout=timeout)
                        )
                    finally:
                        loop.close()
                else:
                    # 루프는 있지만 실행 중이 아닌 경우
                    success = loop.run_until_complete(
                        self.request_gpu(module_name, required_mb, deps, target_util, 
                                        must_succeed=is_required, timeout=timeout)
                    )
                
                if not success and is_required:
                    msg = f"[GPU BLOCKING] ❌ 필수 모듈 {module_name} 메모리 할당 실패"
                    raise RuntimeError(msg)
                return success
                
        except Exception as e:
            msg = f"[GPU BLOCKING] ❌ {module_name} 할당 중 오류: {e}"
            logger.error(msg)
            if is_required:
                raise RuntimeError(msg) from e
            return False
        
    def register_cpu_preloaded(self, module_name: str, estimated_mb: float):
        """
        CPU 프리로드 모듈 등록
        
        Args:
            module_name: 모듈 이름
            estimated_mb: 예상 메모리 크기 (MB)
        """
        # CPU 프리로드 모듈 추적
        if not hasattr(self, 'cpu_preloaded_models'):
            self.cpu_preloaded_models = {}
            
        self.cpu_preloaded_models[module_name] = estimated_mb
        logger.info(f"[CPU 프리로드] {module_name} 등록 ({estimated_mb}MB)")
        
        # GPU 목록에서 제거 (있다면)
        if module_name in self.gpu_models:
            self.gpu_models.remove(module_name)