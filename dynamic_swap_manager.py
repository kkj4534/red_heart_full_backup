"""
Red Heart 동적 스왑 매니저 - LLM 스타일 RAM 스왑 시스템
Dynamic Swap Manager for Red Heart - LLM-style RAM Swap System

800M 파라미터를 8GB GPU에서 효율적으로 처리:
- 300M 백본: 항상 GPU 상주
- 500M 헤드들: 필요시에만 GPU 스왑
- 예측적 프리로딩으로 오버헤드 최소화
- 압축 스왑으로 속도 향상
"""

import asyncio
import logging
import time
import torch
import torch.nn as nn
import pickle
import gzip
import threading
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import psutil
import gc
from collections import deque, defaultdict
from enum import Enum
import io
import queue
import weakref

from config import ADVANCED_CONFIG, get_gpu_memory_info, get_smart_device
from workflow_aware_memory_manager import (
    WorkflowAwareMemoryManager, WorkflowStage, WorkflowTracker
)

# 로거 설정
logger = logging.getLogger(__name__)

class SwapLocation(Enum):
    """스왑 위치"""
    GPU = "gpu"
    RAM = "ram"
    DISK = "disk"

class SwapPriority(Enum):
    """스왑 우선순위"""
    CRITICAL = "critical"    # 백본 (항상 GPU)
    HIGH = "high"           # 현재 처리 중인 헤드
    MEDIUM = "medium"       # 곧 필요할 헤드
    LOW = "low"             # 당분간 불필요한 헤드
    WORKFLOW = "workflow"   # 워크플로우 보호 모델
    DYNAMIC = "dynamic"     # 동적 점수 기반 우선순위

@dataclass
class SwapableModel:
    """스왑 가능한 모델 정보"""
    name: str
    model: Optional[nn.Module]  # 실제 nn.Module만 허용 (NO FALLBACK)
    location: SwapLocation
    priority: SwapPriority
    last_access: datetime
    size_mb: float
    compressed_data: Optional[bytes] = None
    access_count: int = 0
    swap_count: int = 0
    status: str = 'ready'  # 항상 'ready' (NO FALLBACK)
    last_bind_attempt: Optional[datetime] = None
    priority_score: float = 50.0  # 워크플로우 기반 동적 점수 (0-100)
    workflow_group: Optional[str] = None  # 연계된 모듈 그룹
    avoid_unload: bool = False  # 점수가 매우 높을 때 언로드 회피 플래그
    
    def __post_init__(self):
        self.last_access = datetime.now()

class TaskSequencePredictor:
    """태스크 시퀀스 예측기"""
    
    def __init__(self, history_size: int = 1000):
        self.history = deque(maxlen=history_size)
        self.patterns = defaultdict(lambda: defaultdict(int))
        self.window_size = 3
        
    def record_task(self, task: str):
        """태스크 기록"""
        self.history.append((task, datetime.now()))
        self._update_patterns()
        
    def _update_patterns(self):
        """패턴 업데이트"""
        if len(self.history) < self.window_size + 1:
            return
            
        # 최근 window_size개 태스크를 기반으로 다음 태스크 예측 패턴 학습
        recent_tasks = [item[0] for item in list(self.history)[-self.window_size-1:]]
        context = tuple(recent_tasks[:-1])
        next_task = recent_tasks[-1]
        
        self.patterns[context][next_task] += 1
        
    def predict_next_tasks(self, current_task: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """다음 태스크들 예측"""
        if len(self.history) < self.window_size:
            return []
            
        # 최근 컨텍스트 구성
        recent_tasks = [item[0] for item in list(self.history)[-self.window_size+1:]]
        recent_tasks.append(current_task)
        context = tuple(recent_tasks)
        
        # 예측
        predictions = self.patterns.get(context, {})
        if not predictions:
            return []
            
        # 확률 계산 및 정렬
        total_count = sum(predictions.values())
        sorted_predictions = sorted(
            [(task, count/total_count) for task, count in predictions.items()],
            key=lambda x: x[1], reverse=True
        )
        
        return sorted_predictions[:top_k]

class ModelCompressor:
    """모델 압축기"""
    
    @staticmethod
    def compress_model(model: nn.Module) -> bytes:
        """모델을 압축된 바이트로 변환"""
        start_time = time.time()
        
        # 모델 상태를 바이트로 직렬화
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        
        # gzip 압축
        compressed_data = gzip.compress(buffer.getvalue())
        
        compression_time = time.time() - start_time
        original_size = len(buffer.getvalue())
        compressed_size = len(compressed_data)
        compression_ratio = original_size / compressed_size
        
        logger.debug(f"모델 압축 완료: {original_size/1024/1024:.1f}MB -> "
                    f"{compressed_size/1024/1024:.1f}MB "
                    f"(비율: {compression_ratio:.1f}x, 시간: {compression_time:.3f}s)")
        
        return compressed_data
    
    @staticmethod
    def decompress_model(compressed_data: bytes, model_template: nn.Module) -> nn.Module:
        """압축된 데이터에서 모델 복원"""
        start_time = time.time()
        
        # 압축 해제
        decompressed_data = gzip.decompress(compressed_data)
        
        # 모델 상태 로드
        buffer = io.BytesIO(decompressed_data)
        state_dict = torch.load(buffer, map_location='cpu')
        
        # 새 모델 인스턴스 생성 및 상태 로드
        model_template.load_state_dict(state_dict)
        
        decompression_time = time.time() - start_time
        logger.debug(f"모델 압축 해제 완료: 시간 {decompression_time:.3f}s")
        
        return model_template

class AsyncModelSwapper:
    """비동기 모델 스왑퍼"""
    
    def __init__(self, max_workers: int = 2):
        self.max_workers = max_workers
        self.swap_queue = asyncio.Queue()
        self.workers = []
        self.running = False
        
    async def start(self):
        """스왑퍼 시작"""
        self.running = True
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._swap_worker(f"worker_{i}"))
            self.workers.append(worker)
        logger.info(f"비동기 스왑퍼 시작: {self.max_workers}개 워커")
        
    async def stop(self):
        """스왑퍼 중지"""
        self.running = False
        
        # 모든 워커에 중지 신호
        for _ in self.workers:
            await self.swap_queue.put(None)
            
        # 워커들 완료 대기
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        logger.info("비동기 스왑퍼 중지")
        
    async def _swap_worker(self, worker_name: str):
        """스왑 워커"""
        logger.debug(f"스왑 워커 {worker_name} 시작")
        
        while self.running:
            try:
                # 스왑 작업 대기
                swap_task = await asyncio.wait_for(self.swap_queue.get(), timeout=1.0)
                
                if swap_task is None:  # 중지 신호
                    break
                    
                # 스왑 실행
                await swap_task()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"스왑 워커 {worker_name} 오류: {str(e)}")
                
        logger.debug(f"스왑 워커 {worker_name} 종료")
        
    async def schedule_swap(self, swap_coroutine):
        """스왑 작업 예약"""
        await self.swap_queue.put(swap_coroutine)

class RedHeartDynamicSwapManager:
    """
    Red Heart 동적 스왑 매니저
    
    LLM 스타일 적극적 RAM 스왑으로 800M 파라미터를 8GB GPU에서 처리
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or ADVANCED_CONFIG.get('dynamic_swap_config', {})
        
        # 기본 우선순위 테이블 - 동적 우선순위 지원
        # 초기값만 제공, 실제로는 워크플로우 단계별 priority_score로 동적 조정됨
        self.DEFAULT_PRIORITIES = {
            "unified_backbone": SwapPriority.HIGH,  # 기본 HIGH, 워크플로우에서 점수 95로 상향
            "translator": SwapPriority.HIGH,
            "emotion_analyzer": SwapPriority.HIGH,
            "emotion_empathy_head": SwapPriority.HIGH,  # emotion_analyzer와 연동
            "semantic_analyzer": SwapPriority.HIGH,
            "bentham_calculator": SwapPriority.HIGH,
            "regret_analyzer": SwapPriority.MEDIUM,
            "neural_components": SwapPriority.LOW,
            "meta_integration": SwapPriority.MEDIUM,
            "surd_analyzer": SwapPriority.MEDIUM,
            "bayesian_engine": SwapPriority.LOW,
            "llm_engine": SwapPriority.LOW,
            "experience_database": SwapPriority.LOW,
            "hierarchical_emotion": SwapPriority.MEDIUM,
            "usage_pattern_analyzer": SwapPriority.LOW,
            # 헤드들
            "emotion_empathy_head": SwapPriority.HIGH,
            "bentham_fromm_head": SwapPriority.HIGH,
            "semantic_surd_head": SwapPriority.HIGH,
            "regret_learning_head": SwapPriority.MEDIUM,
            "meta_integration_head": SwapPriority.MEDIUM,
        }
        
        # 스왑 관리
        self.models: Dict[str, SwapableModel] = {}
        self.gpu_resident_models: Dict[str, nn.Module] = {}
        self.ram_models: Dict[str, SwapableModel] = {}
        
        # 예측 및 최적화 컴포넌트
        self.task_predictor = TaskSequencePredictor()
        self.model_compressor = ModelCompressor()
        self.async_swapper = AsyncModelSwapper()
        
        # 워크플로우 인식 메모리 관리자 통합
        self.workflow_memory_manager = WorkflowAwareMemoryManager(
            memory_threshold_mb=self.config.get('memory_threshold_mb', 6500.0)
        )
        
        # 설정
        self.memory_threshold = self.config.get('memory_threshold', 0.65)
        self.swap_timeout = self.config.get('swap_timeout', 2.0)
        self.compression_enabled = self.config.get('compression_enabled', True)
        self.async_swap = self.config.get('async_swap', True)
        self.preload_prediction = self.config.get('preload_prediction', True)
        self.workflow_aware = self.config.get('workflow_aware', True)  # 워크플로우 인식 모드
        
        # 통계
        self.stats = {
            'total_swaps': 0,
            'successful_swaps': 0,
            'failed_swaps': 0,
            'total_swap_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'preload_hits': 0,
            'preload_misses': 0
        }
        
        # 백그라운드 작업
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        
        logger.info("RedHeartDynamicSwapManager 초기화 완료")
        
    async def initialize(self):
        """스왑 매니저 초기화"""
        if self.async_swap:
            await self.async_swapper.start()
            
        # 백그라운드 정리 작업 시작
        cleanup_task = asyncio.create_task(self._background_cleanup())
        self._background_tasks.append(cleanup_task)
        
        logger.info("동적 스왑 매니저 초기화 완료")
        
    async def shutdown(self):
        """스왑 매니저 종료"""
        self._shutdown_event.set()
        
        # 백그라운드 작업들 정리
        for task in self._background_tasks:
            task.cancel()
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        if self.async_swap:
            await self.async_swapper.stop()
            
        logger.info("동적 스왑 매니저 종료 완료")
        
    def register_model(self, name: str, model: nn.Module, priority: SwapPriority = None):
        """모델 등록 - NO FALLBACK 정책 (nn.Module 필수)
        
        Args:
            name: 모델 이름
            model: 실제 nn.Module 인스턴스 (필수)
            priority: 우선순위 (None이면 DEFAULT_PRIORITIES에서 조회)
        """
        # NO FALLBACK - model이 None이거나 nn.Module이 아니면 즉시 예외
        if model is None:
            raise RuntimeError(f"[DSM] {name} 등록 실패: model is None (NO FALLBACK)")
        if not isinstance(model, nn.Module):
            raise RuntimeError(f"[DSM] {name} 등록 실패: nn.Module이 아님 ({type(model).__name__}) - NO FALLBACK")
            
        # 우선순위 자동 설정
        if priority is None:
            priority = self.DEFAULT_PRIORITIES.get(name, SwapPriority.MEDIUM)
            logger.debug(f"DEFAULT_PRIORITIES에서 {name}의 우선순위 설정: {priority.value}")
            
        # 실제 모델 크기 계산 (NO 추정치, NO 메타 등록)
        size_mb = self._calculate_model_size(model)
        
        # size_mb 검증 (0 이하면 등록 불가)
        if size_mb <= 0:
            logger.error(f"[DSM] {name} 파라미터 없음: size_mb={size_mb:.1f}MB")
            raise RuntimeError(f"[DSM] {name} 등록 실패: 파라미터 없음 (size={size_mb:.1f}MB)")
        
        logger.info(f"[DSM] {name} 크기: {size_mb:.1f}MB (mgr_id={id(self)})")
        
        # 모델의 현재 device를 확인하여 location 결정
        is_cuda = any(p.is_cuda for p in model.parameters()) if hasattr(model, 'parameters') else False
        location = SwapLocation.GPU if is_cuda else SwapLocation.RAM
        
        swapable_model = SwapableModel(
            name=name,
            model=model,  # 항상 nn.Module
            location=location,  # 실제 device에 따라 설정
            priority=priority,
            last_access=datetime.now(),
            size_mb=size_mb,
            status='ready'  # 항상 ready (NO deferred)
        )
            
        # 기존 키가 있으면 교체 로깅
        if name in self.models:
            old_model = self.models[name]
            old_size = old_model.size_mb
            old_priority = old_model.priority.name if old_model.priority else "NONE"
            new_priority = priority.name if priority else "NONE"
            logger.info(f"[DSM] 교체 등록: {name} ({old_size:.1f}MB→{size_mb:.1f}MB, {old_priority}→{new_priority})")
        
        self.models[name] = swapable_model
        
        # location과 gpu_resident_models를 일관되게 관리
        if is_cuda:
            self.gpu_resident_models[name] = model
            logger.info(f"   ✅ {name}을 GPU에 등록 (location=GPU, size={size_mb:.1f}MB)")
        else:
            self.ram_models[name] = swapable_model
            # GPU에서 제거 (혹시 있을 수 있음)
            self.gpu_resident_models.pop(name, None)
            logger.info(f"   💾 {name}을 RAM에 등록 (location=RAM, size={size_mb:.1f}MB)")
        
        # master_model_registry에도 등록 (언로드 후보로 사용)
        from config import get_master_orchestrator, ModelPriority
        orchestrator = get_master_orchestrator()
        if orchestrator:
            # SwapPriority를 ModelPriority로 매핑
            priority_map = {
                SwapPriority.CRITICAL: ModelPriority.CRITICAL,
                SwapPriority.HIGH: ModelPriority.HIGH,
                SwapPriority.MEDIUM: ModelPriority.MEDIUM,
                SwapPriority.LOW: ModelPriority.LOW
            }
            model_priority = priority_map.get(priority, ModelPriority.MEDIUM)
            
            orchestrator.master_model_registry[name] = {
                'device': 'cuda' if is_cuda else 'cpu',
                'memory_mb': size_mb,
                'priority': model_priority,
                'access_count': 0,
                'last_access': datetime.now()
            }
            logger.debug(f"   master_model_registry에 {name} 등록 (priority={model_priority})")
        
        # 압축 데이터 생성 (백그라운드에서)
        if self.compression_enabled:
            asyncio.create_task(self._compress_model_background(name))
            
        logger.info(f"[REGISTER HEAD] {name} ({size_mb:.1f}MB, 우선순위: {priority.value})")
    
    def update_workflow_priorities(self, workflow_stage: WorkflowStage, required_models: List[str], 
                                 related_groups: Dict[str, List[str]] = None):
        """워크플로우 기반 동적 우선순위 업데이트
        
        Args:
            workflow_stage: 현재 워크플로우 단계
            required_models: 현 단계에서 필요한 모델들
            related_groups: 연계된 모듈 그룹 {그룹명: [모델들]}
        """
        related_groups = related_groups or {}
        
        # 워크플로우 단계별 기본 점수
        stage_base_scores = {
            WorkflowStage.INITIALIZATION: 30.0,
            WorkflowStage.DATA_LOADING: 40.0,
            WorkflowStage.BACKBONE_FORWARD: 85.0,  # 백본 단계에서는 높은 점수
            WorkflowStage.HEAD_PROCESSING: 80.0,
            WorkflowStage.SYNERGY_COMPUTATION: 75.0,
            WorkflowStage.LOSS_COMPUTATION: 70.0,
            WorkflowStage.BACKWARD_PASS: 85.0,
            WorkflowStage.OPTIMIZATION: 60.0,
            WorkflowStage.EVALUATION: 50.0,
            WorkflowStage.FINALIZATION: 30.0
        }
        
        base_score = stage_base_scores.get(workflow_stage, 50.0)
        
        # 모든 모델의 점수를 초기화 (기본 낮은 점수)
        for name, model in self.models.items():
            model.priority_score = 20.0  # 사용 안 하는 모델은 낮은 점수
        
        # 필요한 모델들에 높은 점수 부여
        for model_name in required_models:
            if model_name in self.models:
                self.models[model_name].priority_score = base_score
                self.models[model_name].workflow_group = f"{workflow_stage.value}_primary"
                
                # 백본은 항상 더 높은 점수
                if model_name == "unified_backbone" and workflow_stage in [
                    WorkflowStage.BACKBONE_FORWARD, WorkflowStage.BACKWARD_PASS
                ]:
                    self.models[model_name].priority_score = 95.0
        
        # 연계된 그룹에 같은 점수 부여
        for group_name, group_models in related_groups.items():
            group_score = base_score - 5.0  # 주 모델보다 약간 낮은 점수
            for model_name in group_models:
                if model_name in self.models:
                    self.models[model_name].priority_score = group_score
                    self.models[model_name].workflow_group = group_name
        
        # 점수 기반으로 우선순위 재설정 (순수 점수제)
        for name, model in self.models.items():
            # 점수만으로 우선순위 결정 (CRITICAL 제거, 동적 시스템 일관화)
            if model.priority_score >= 90.0:
                # 90점 이상은 극히 중요 (백본 등) - 하지만 절대 언로드 불가는 아님
                model.priority = SwapPriority.HIGH
                # 점수가 매우 높으면 언로드 회피 플래그 설정
                model.avoid_unload = True
            elif model.priority_score >= 70.0:
                model.priority = SwapPriority.HIGH
                model.avoid_unload = False
            elif model.priority_score >= 50.0:
                model.priority = SwapPriority.MEDIUM
                model.avoid_unload = False
            elif model.priority_score >= 30.0:
                model.priority = SwapPriority.LOW
                model.avoid_unload = False
            else:
                model.priority = SwapPriority.LOW
                model.avoid_unload = False
        
        logger.info(f"[워크플로우] {workflow_stage.value} 단계 우선순위 업데이트")
        logger.debug(f"  필수 모델: {required_models[:5]} (점수: {base_score})")
        logger.debug(f"  연계 그룹: {list(related_groups.keys())[:3]}")
    
    def get_model_priority_score(self, model_name: str) -> float:
        """모델의 현재 우선순위 점수 반환"""
        if model_name in self.models:
            return self.models[model_name].priority_score
        return 0.0
    
    def get_workflow_aware_unload_candidates(self, required_mb: float, 
                                            exclude_models: Set[str] = None) -> List[str]:
        """워크플로우 인식 언로드 후보 선정
        
        점수가 낮은 모델부터 언로드 후보로 선정
        """
        exclude_models = exclude_models or set()
        candidates = []
        
        for name, model in self.gpu_resident_models.items():
            if name not in exclude_models:
                model_info = self.models[name]
                score = model_info.priority_score
                size_mb = model_info.size_mb
                
                # avoid_unload 플래그가 설정된 모델은 최대한 회피 (점수 90+ 모델)
                # 하지만 절대 금지는 아님 (동적 시스템)
                if not model_info.avoid_unload:
                    candidates.append((name, score, size_mb))
                else:
                    # avoid_unload는 최후의 선택지로 남겨둠
                    logger.debug(f"  {name} 언로드 회피 (점수: {score:.1f}, avoid_unload=True)")
        
        # 점수가 낮은 순으로 정렬
        candidates.sort(key=lambda x: x[1])
        
        # 필요한 메모리만큼만 선택
        selected = []
        freed_mb = 0.0
        
        for name, score, size_mb in candidates:
            if freed_mb >= required_mb:
                break
            selected.append(name)
            freed_mb += size_mb
            logger.debug(f"  언로드 후보: {name} (점수: {score:.1f}, 크기: {size_mb:.1f}MB)")
        
        return selected
        
    async def load_model_to_gpu(self, name: str, timeout: float = None) -> nn.Module:
        """모델을 GPU로 로드"""
        if name not in self.models:
            raise ValueError(f"등록되지 않은 모델: {name}")
            
        timeout = timeout or self.swap_timeout
        start_time = time.time()
        
        # 이미 GPU에 있는 경우
        if name in self.gpu_resident_models:
            self._update_access_stats(name, hit=True)
            return self.gpu_resident_models[name]
            
        self._update_access_stats(name, hit=False)
        
        # GPU 메모리 확인 및 정리
        # GPU 사용률이 높으면 HIGH 우선순위도 언로드 허용
        gpu_info = get_gpu_memory_info()
        allow_high_unload = gpu_info and gpu_info['usage_percent'] >= 85.0
        await self._ensure_gpu_memory(self.models[name].size_mb, allow_high_priority_unload=allow_high_unload)
        
        try:
            # 모델을 GPU로 이동
            model = self.models[name].model
            device = get_smart_device(memory_required_mb=self.models[name].size_mb * 1.2)
            
            if device.type == 'cuda':
                model = model.to(device)
                model.eval()  # 추론 모드로 설정
                
                # GPU 상주 모델로 등록
                self.gpu_resident_models[name] = model
                self.models[name].location = SwapLocation.GPU
                
                # RAM에서 제거
                if name in self.ram_models:
                    del self.ram_models[name]
                    
                swap_time = time.time() - start_time
                self._update_swap_stats(True, swap_time)
                
                logger.debug(f"모델 GPU 로드 완료: {name} ({swap_time:.3f}s)")
                
                # 태스크 예측기에 기록
                self.task_predictor.record_task(name)
                
                # 예측적 프리로딩
                if self.preload_prediction:
                    asyncio.create_task(self._predictive_preload(name))
                    
                return model
            else:
                # GPU 메모리 부족 - CPU에서 실행
                logger.warning(f"GPU 메모리 부족으로 CPU에서 실행: {name}")
                return model.to('cpu')
                
        except Exception as e:
            self._update_swap_stats(False, time.time() - start_time)
            logger.error(f"모델 GPU 로드 실패: {name}, 오류: {str(e)}")
            raise
            
    async def ensure_on_gpu(self, name: str, required_mb: float = None) -> bool:
        """해당 모델이 DSM에만 있으면 GPU로 승격. 이미 GPU면 no-op.
        
        Args:
            name: 모델 이름
            required_mb: 필요한 메모리 (MB), None이면 모델 크기 사용
            
        Returns:
            bool: 성공 시 True, 실패 시 False
        """
        if name not in self.models:
            logger.warning(f"ensure_on_gpu: 레지스트리에 없음: {name}")
            return False
        
        # 이미 GPU에 있으면 성공
        if name in self.gpu_resident_models:
            logger.debug(f"ensure_on_gpu: {name}는 이미 GPU에 있음")
            return True
        
        # 필요시 메모리 확보
        if required_mb is None:
            required_mb = self.models[name].size_mb
        
        try:
            # GPU로 로드
            await self.load_model_to_gpu(name)
            logger.info(f"ensure_on_gpu: {name} GPU 승격 완료")
            return True
        except Exception as e:
            logger.error(f"ensure_on_gpu: {name} GPU 승격 실패: {e}")
            return False
    
    async def unload_model_from_gpu(self, name: str):
        """모델을 GPU에서 언로드"""
        if name not in self.gpu_resident_models:
            logger.warning(f"언로드 요청된 모델 {name}이 gpu_resident_models에 없음")
            # models에서 찾아보기
            if name in self.models and self.models[name].model is not None:
                model = self.models[name].model
                if hasattr(model, 'device') and str(model.device).startswith('cuda'):
                    logger.info(f"models에서 {name} 발견, GPU에서 언로드 시도")
                else:
                    return
            else:
                return
        else:
            model = self.gpu_resident_models[name]
            
        start_time = time.time()
        
        try:
            # GPU 메모리 사용량 측정 (언로드 전)
            from config import get_gpu_memory_info
            before_info = get_gpu_memory_info()
            before_mb = before_info['allocated_mb'] if before_info else 0
            
            # GPU에서 모델 제거
            if hasattr(model, 'to'):
                model = model.to('cpu')
            elif hasattr(model, 'cpu'):
                model = model.cpu()
            else:
                logger.warning(f"{name}: to() 또는 cpu() 메서드 없음")
            
            # 모든 하위 모듈도 CPU로 이동
            if hasattr(model, 'modules'):
                for submodule in model.modules():
                    if hasattr(submodule, 'to'):
                        submodule.to('cpu')
            
            # 상태 업데이트
            if name in self.gpu_resident_models:
                del self.gpu_resident_models[name]
            
            if name in self.models:
                self.models[name].model = model
                self.models[name].location = SwapLocation.RAM
                self.ram_models[name] = self.models[name]
            
            # GPU 메모리 정리 (더 적극적으로)
            if torch.cuda.is_available():
                # 참조 제거
                del model
                # 가비지 컬렉션
                import gc
                gc.collect()
                # GPU 캐시 정리
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()  # CUDA 11+: OS에 메모리 반환
                
            # GPU 메모리 사용량 측정 (언로드 후)
            after_info = get_gpu_memory_info()
            after_mb = after_info['allocated_mb'] if after_info else 0
            freed_mb = before_mb - after_mb
                
            unload_time = time.time() - start_time
            logger.info(f"✅ {name} GPU 언로드 완료 ({unload_time:.3f}s, {freed_mb:.1f}MB 해제)")
            
        except Exception as e:
            logger.error(f"모델 GPU 언로드 실패: {name}, 오류: {str(e)}")
            raise
            
    async def get_model(self, name: str) -> nn.Module:
        """모델 가져오기 (필요시 자동 로드)"""
        self.models[name].last_access = datetime.now()
        self.models[name].access_count += 1
        
        if name in self.gpu_resident_models:
            return self.gpu_resident_models[name]
        else:
            return await self.load_model_to_gpu(name)
            
    async def load_head_to_gpu(self, head_name: str, timeout: float = None) -> nn.Module:
        """헤드 특화 GPU 로딩 메소드 - 헤드별 최적화 및 시너지 고려"""
        logger.info(f"🔍 load_head_to_gpu('{head_name}')")
        logger.info(f"🔍 DSM keys(pre): {sorted(list(self.models.keys()))[:50]}")
        
        # 헤드가 등록되지 않은 경우 즉시 에러
        if head_name not in self.models:
            logger.error(f"[DSM] 등록되지 않은 헤드 요청: {head_name} (mgr_id={id(self)})")
            logger.error(f"[DSM] 현재 keys(models): {sorted(list(self.models.keys()))[:50]}")
            raise ValueError(f"등록되지 않은 헤드: {head_name}")
        
        # 모델이 None인 경우도 에러 (NO FALLBACK)
        if self.models[head_name].model is None:
            raise RuntimeError(f"[NO-FALLBACK] {head_name}: model is None")
            
        timeout = timeout or self.swap_timeout
        start_time = time.time()
        
        logger.debug(f"헤드 GPU 로딩 시작: {head_name}")
        
        # 이미 GPU에 있는 경우
        if head_name in self.gpu_resident_models:
            self._update_access_stats(head_name, hit=True)
            logger.debug(f"헤드 이미 GPU에 상주: {head_name}")
            return self.gpu_resident_models[head_name]
            
        self._update_access_stats(head_name, hit=False)
        
        # NO FALLBACK - lazy 바인딩 제거
        # 모델이 이미 등록되어 있어야 함
        if self.models[head_name].model is None:
            raise RuntimeError(f"헤드 {head_name}의 모델이 None입니다 (NO FALLBACK 정책)")
        
        # 헤드 특화 메모리 확보 (일반 모델보다 보수적으로)
        head_size_mb = self.models[head_name].size_mb
        # 현재 로드 중인 헤드는 언로드에서 제외
        # GPU 사용률이 높으면 HIGH 우선순위도 언로드 허용
        gpu_info = get_gpu_memory_info()
        allow_high_unload = gpu_info and gpu_info['usage_percent'] >= 85.0
        await self._ensure_gpu_memory(head_size_mb * 1.1, exclude_models={head_name}, 
                                     allow_high_priority_unload=allow_high_unload)  # 10% 여유
        
        # 후보/결과 요약
        try:
            mem = get_gpu_memory_info()
            candidates = [name for name, model in self.models.items() 
                         if model.location == SwapLocation.GPU and model.priority != SwapPriority.CRITICAL]
            logger.info(f"[G9] after-ensure free_mb={mem.get('free_mb', 'NA') if mem else 'NA'} "
                       f"alloc={mem.get('allocated_mb', 'NA') if mem else 'NA'}MB "
                       f"candidates={candidates[:10]}")
        except Exception:
            pass
        
        try:
            # 헤드를 GPU로 이동
            head_model = self.models[head_name].model
            if head_model is None:
                raise ValueError(f"헤드 모델이 None입니다: {head_name}")
            device = get_smart_device(memory_required_mb=head_size_mb * 1.2)
            
            if device.type == 'cuda':
                # 헤드 특화 최적화
                head_model = head_model.to(device)
                head_model.eval()  # 헤드는 항상 추론 모드
                
                # 헤드 특화 메모리 최적화
                if hasattr(head_model, 'half') and ADVANCED_CONFIG.get('use_fp16', True):
                    head_model = head_model.half()  # FP16 사용
                
                # GPU 상주 헤드로 등록
                self.gpu_resident_models[head_name] = head_model
                self.models[head_name].location = SwapLocation.GPU
                
                # RAM에서 제거
                if head_name in self.ram_models:
                    del self.ram_models[head_name]
                    
                swap_time = time.time() - start_time
                self._update_swap_stats(True, swap_time)
                
                logger.info(f"헤드 GPU 로딩 완료: {head_name} ({swap_time:.3f}s, {head_size_mb:.1f}MB)")
                
                # 헤드 사용 패턴 기록
                self.task_predictor.record_task(f"head_{head_name}")
                
                # 헤드 특화 예측적 프리로딩 (관련 헤드들)
                if self.preload_prediction:
                    asyncio.create_task(self._predictive_head_preload(head_name))
                    
                return head_model
            else:
                # NO FALLBACK - GPU 메모리 부족 시 즉시 실패
                logger.error(f"[DSM] {head_name} GPU 로딩 실패: device={device.type} (NO FALLBACK)")
                raise RuntimeError(f"[DSM] {head_name} GPU 로딩 실패: GPU 메모리 부족 (NO FALLBACK)")
                
        except Exception as e:
            self._update_swap_stats(False, time.time() - start_time)
            logger.error(f"헤드 GPU 로딩 실패: {head_name}, 오류: {str(e)}")
            
            # 순수 재시도 방식 (fallback 없음)
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                retry_count += 1
                logger.info(f"헤드 GPU 로딩 재시도 {retry_count}/{max_retries}: {head_name}")
                
                try:
                    # 메모리 강제 정리 후 재시도
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    await asyncio.sleep(0.5)  # 짧은 대기
                    
                    # 더 보수적인 메모리 확보 (현재 헤드 제외)
                    # 재시도 시에는 더 적극적으로 언로드
                    gpu_info = get_gpu_memory_info()
                    allow_high_unload = gpu_info and gpu_info['usage_percent'] >= 80.0  # 재시도 시 더 낮은 임계치
                    await self._ensure_gpu_memory(head_size_mb * 1.5, exclude_models={head_name},
                                                 allow_high_priority_unload=allow_high_unload)
                    
                    head_model = self.models[head_name].model
                    device = get_smart_device(memory_required_mb=head_size_mb * 1.3)
                    
                    if device.type == 'cuda':
                        head_model = head_model.to(device)
                        head_model.eval()
                        
                        self.gpu_resident_models[head_name] = head_model
                        self.models[head_name].location = SwapLocation.GPU
                        
                        if head_name in self.ram_models:
                            del self.ram_models[head_name]
                            
                        total_time = time.time() - start_time
                        self._update_swap_stats(True, total_time)
                        
                        logger.info(f"헤드 GPU 로딩 재시도 성공: {head_name} ({total_time:.3f}s)")
                        return head_model
                        
                except Exception as retry_error:
                    logger.warning(f"헤드 GPU 로딩 재시도 {retry_count} 실패: {head_name}, 오류: {str(retry_error)}")
                    
                    if retry_count == max_retries:
                        logger.error(f"헤드 GPU 로딩 최대 재시도 횟수 초과: {head_name}")
                        raise retry_error
            
            raise e
            
    def log_memory_reconciliation(self, tag: str = ""):
        """DSM 추적 메모리 vs 실제 GPU 메모리 비교"""
        from config import get_gpu_memory_info
        
        info = get_gpu_memory_info() or {}
        allocated_mb = info.get('allocated_mb', 0.0)
        
        # DSM이 추적하는 GPU 메모리 총합
        dsm_tracked = sum(
            self.models[name].size_mb 
            for name in self.gpu_resident_models.keys()
            if name in self.models
        )
        
        # 추적되지 않는 메모리 (백본, 캐시 등)
        untracked_mb = max(0.0, allocated_mb - dsm_tracked)
        untracked_pct = (untracked_mb / allocated_mb * 100) if allocated_mb > 0 else 0
        
        logger.critical(f"[MEM RECON {tag}] "
                       f"torch_alloc={allocated_mb:.1f}MB, "
                       f"dsm_tracked={dsm_tracked:.1f}MB, "
                       f"untracked={untracked_mb:.1f}MB ({untracked_pct:.1f}% 미추적)")
        
        # 큰 미추적 메모리가 있으면 경고
        if untracked_pct > 50:
            logger.warning(f"⚠️ DSM 미추적 메모리가 50% 이상: {untracked_mb:.1f}MB")
            logger.warning(f"   DSM GPU 모델: {list(self.gpu_resident_models.keys())[:10]}")
    
    async def _ensure_gpu_memory(self, required_mb: float, max_attempts: int = 3, 
                                exclude_models: set = None, allow_high_priority_unload: bool = False):
        """GPU 메모리 확보 - fail-fast 구현
        
        Args:
            required_mb: 필요한 메모리 (MB)
            max_attempts: 최대 시도 횟수 (기본 3회)
            exclude_models: 언로드에서 제외할 모델 이름 집합
            allow_high_priority_unload: HIGH 우선순위 모델도 언로드 허용 여부
        """
        exclude_models = exclude_models or set()
        
        # 스왑 시작 전 DSM 상태 로깅
        logger.info(f"[DSM 스왑 시작] 필요: {required_mb:.1f}MB, 제외: {list(exclude_models)[:5]}")
        logger.info(f"[DSM 현황] models.keys: {list(self.models.keys())[:20]}")
        logger.info(f"[DSM 현황] gpu_resident: {list(self.gpu_resident_models.keys())[:20]}")
        
        for attempt in range(max_attempts):
            gpu_info = get_gpu_memory_info()
            if gpu_info is None:
                raise RuntimeError("[DSM] GPU 정보를 가져올 수 없음")
                
            available_mb = gpu_info['free_mb']
            
            # 여유 메모리가 충분한 경우
            if available_mb >= required_mb * 1.2:  # 20% 마진
                logger.info(f"✅ [DSM] need={required_mb:.1f}MB / free={available_mb:.1f}MB → 충분함")
                return
                
            logger.info(f"[시도 {attempt+1}/{max_attempts}] need={required_mb:.1f}MB / free={available_mb:.1f}MB → 스왑 필요")
            
            # LRU 기반으로 모델들 언로드
            models_to_unload = []
            
            # ✅ 후보 산정 (GPU 상주 + CRITICAL 제외 + exclude 제외)
            logger.info(f"[DSM 후보 산정] GPU 상주 모델 수: {len(self.gpu_resident_models)}, HIGH 언로드 허용: {allow_high_priority_unload}")
            for name, model in list(self.gpu_resident_models.items()):
                model_info = self.models[name]
                
                if name in exclude_models:
                    logger.debug(f"   - {name}: 제외 (exclude_models에 포함)")
                elif model_info.avoid_unload and not allow_high_priority_unload:
                    # 동적 시스템: avoid_unload 플래그로 판단 (점수 90+ 모델)
                    logger.debug(f"   - {name}: 제외 (avoid_unload=True, 점수={model_info.priority_score:.1f})")
                elif model_info.priority == SwapPriority.HIGH and not allow_high_priority_unload:
                    # HIGH 우선순위도 점수 기반으로 판단
                    logger.debug(f"   - {name}: 제외 (HIGH, 점수={model_info.priority_score:.1f}, allow_high=False)")
                else:
                    # 워크플로우 점수 고려
                    priority_score = model_info.priority_score
                    models_to_unload.append((
                        name, 
                        model_info.last_access, 
                        model_info.priority.value,
                        model_info.size_mb,
                        priority_score  # 워크플로우 기반 동적 점수 추가
                    ))
                    logger.debug(f"   - {name}: 후보 추가 ({model_info.size_mb:.1f}MB, 점수={priority_score:.1f})")
                    
            # 메타 등록된 모델도 후보에 포함 (deferred 상태)
            for name, swapable in self.models.items():
                if (name not in exclude_models and 
                    name not in self.gpu_resident_models and 
                    swapable.status == 'deferred' and
                    not swapable.avoid_unload):  # 동적 시스템: avoid_unload 플래그로 판단
                    # 메타 등록 모델도 언로드 후보 리스트에 포함 (잠재적 공간 확보용)
                    logger.debug(f"   메타 등록 모델도 후보에 추가: {name} ({swapable.size_mb:.1f}MB, 점수={swapable.priority_score:.1f})")
                    
            # 언로드 후보가 없으면 여유 메모리 확인
            if not models_to_unload:
                if available_mb >= required_mb:
                    logger.info(f"[DSM PASS] 후보 0개지만 여유 {available_mb:.1f}MB ≥ 필요 {required_mb:.1f}MB → 언로드 생략")
                    return
                    
                logger.warning(f"[G9] ensure_gpu_memory: 후보 0개 (exclude={list(exclude_models)}) "
                               f"free_mb={available_mb:.1f}")
                logger.error(f"[DSM FAIL] 언로드 후보 0개")
                logger.error(f"   - GPU 상주 모델: {list(self.gpu_resident_models.keys())[:10]}")
                logger.error(f"   - 제외 모델: {exclude_models}")
                
                # 상세 디버깅 정보 추가
                logger.warning(f"[G9] DSM keys: {sorted(list(self.models.keys()))[:50]}")
                logger.warning(f"[G9] GPU-resident keys: {sorted(list(self.gpu_resident_models.keys()))[:50]}")
                
                # 우선순위 정보 추가
                priorities_info = {}
                for k in list(self.models.keys())[:20]:
                    priorities_info[k] = self.models[k].priority.name if self.models[k].priority else "NONE"
                logger.warning(f"[G9] priorities: {priorities_info}")
                logger.error(f"   - 등록된 모델 수: {len(self.models)}")
                logger.error(f"   - avoid_unload 모델 수: {sum(1 for m in self.models.values() if m.avoid_unload)}")
                logger.error(f"   - 점수 90+ 모델 수: {sum(1 for m in self.models.values() if m.priority_score >= 90.0)}")
                logger.error(f"   - 현재 free_mb: {available_mb:.1f}, 필요: {required_mb:.1f}MB")
                raise RuntimeError(f"GPU 메모리 확보 불가: 언로드 가능한 모델 없음 (필요: {required_mb:.1f}MB)")
                
            # ⭐ 개선된 정렬: 큰 모델 우선, 낮은 점수 우선, 오래된 접근 우선
            # 기존: (우선순위값, 마지막접근시간) -> 작은 값 우선
            # 신규: (크기 내림차순, 동적점수 오름차순, 우선순위값, 마지막접근)
            models_to_unload.sort(key=lambda x: (-x[3], x[4] if len(x) > 4 else 50.0, x[2], x[1]))
            logger.info(f"[G9] unload candidates: {[c[0] for c in models_to_unload[:20]]} ... "
                        f"need~{required_mb:.1f}MB")
            
            # 필요한 만큼 언로드
            freed_memory = 0
            unloaded_count = 0
            
            # 튜플 언패킹 (5개 요소: name, last_access, priority, size_mb, priority_score)
            for model_info in models_to_unload:
                name = model_info[0]
                size_mb = model_info[3]
                if freed_memory >= required_mb:
                    break
                    
                try:
                    await self.unload_model_from_gpu(name)
                    freed_memory += size_mb
                    unloaded_count += 1
                    logger.info(f"   언로드: {name} ({size_mb:.1f}MB) - 누적 해제: {freed_memory:.1f}MB")
                except Exception as e:
                    logger.warning(f"   언로드 실패: {name} - {e}")
                    
            # 언로드 결과 확인
            if unloaded_count == 0:
                logger.error(f"[DSM] 모델을 언로드했지만 메모리가 해제되지 않음")
                raise RuntimeError(f"GPU 메모리 확보 실패: 언로드 효과 없음")
                
            # 언로드 결과 한 줄 요약
            unloaded_names = [m[0] for m in models_to_unload[:unloaded_count]]
            logger.info(f"[DSM 스왑] candidates={len(models_to_unload)} / unloaded={unloaded_count} / freed={freed_memory:.1f}MB / models={unloaded_names[:3]}")
            
            # 메모리 추적 로깅
            self.log_memory_reconciliation(f"after_unload_try{attempt+1}")
            
            # 강화된 동기화: 언로드 후 실제 메모리 해제 보장
            if torch.cuda.is_available():
                # 1. 캐시 비우기
                torch.cuda.empty_cache()
                # 2. GPU 동기화 (모든 CUDA 작업 완료 대기)
                torch.cuda.synchronize()
                # 3. Python GC 실행
                gc.collect()
                # 4. 짧은 대기 (allocator 반영 시간)
                await asyncio.sleep(0.2)
                
                # 5. 언로드 후 실제 메모리 변화 측정
                gpu_info_after = get_gpu_memory_info()
                if gpu_info_after:
                    logger.info(f"[DSM 동기화] 언로드 후 GPU 사용률: {gpu_info_after['usage_percent']:.1f}% "
                              f"(여유: {gpu_info_after['free_mb']:.1f}MB)")
                
            # 다음 시도 전 추가 대기
            if attempt < max_attempts - 1:
                await asyncio.sleep(0.3)
        
        # 모든 시도 실패
        logger.error(f"[DSM 스왑 실패] 최종 상태:")
        logger.error(f"   - models.keys: {list(self.models.keys())[:20]}")
        logger.error(f"   - gpu_resident: {list(self.gpu_resident_models.keys())[:20]}")
        raise RuntimeError(f"[DSM] GPU 메모리 확보 실패: {max_attempts}회 시도 후에도 {required_mb:.1f}MB 확보 불가")
            
    async def _predictive_preload(self, current_task: str):
        """예측적 프리로딩"""
        try:
            predictions = self.task_predictor.predict_next_tasks(current_task, top_k=2)
            
            for next_task, probability in predictions:
                if probability > 0.3 and next_task in self.models:  # 30% 이상 확률
                    if next_task not in self.gpu_resident_models:
                        # 백그라운드에서 프리로드
                        asyncio.create_task(self._background_preload(next_task))
                        
        except Exception as e:
            logger.debug(f"예측적 프리로딩 오류: {str(e)}")
            
    async def _background_preload(self, model_name: str):
        """백그라운드 프리로딩"""
        try:
            # 잠시 대기 (현재 작업 완료 후)
            await asyncio.sleep(0.5)
            
            # GPU 메모리 여유 확인
            gpu_info = get_gpu_memory_info()
            if gpu_info and gpu_info['usage_percent'] < 55:  # 55% 미만 사용 시에만
                await self.load_model_to_gpu(model_name)
                self.stats['preload_hits'] += 1
                logger.debug(f"예측적 프리로딩 성공: {model_name}")
            else:
                self.stats['preload_misses'] += 1
                
        except Exception as e:
            self.stats['preload_misses'] += 1
            logger.debug(f"예측적 프리로딩 실패: {model_name}, {str(e)}")
            
    async def _compress_model_background(self, name: str):
        """백그라운드 모델 압축"""
        try:
            if name in self.models:
                model = self.models[name].model
                compressed_data = self.model_compressor.compress_model(model)
                self.models[name].compressed_data = compressed_data
                logger.debug(f"모델 압축 완료: {name}")
        except Exception as e:
            logger.debug(f"모델 압축 실패: {name}, {str(e)}")
    
    async def _predictive_head_preload(self, current_head: str):
        """헤드 특화 예측적 프리로딩"""
        try:
            # 헤드 간 관련성 기반 예측 (간단한 휴리스틱)
            related_heads = self._get_related_heads(current_head)
            
            for related_head in related_heads:
                if related_head in self.models and related_head not in self.gpu_resident_models:
                    # 백그라운드에서 관련 헤드 프리로드
                    asyncio.create_task(self._background_head_preload(related_head))
                    
        except Exception as e:
            logger.debug(f"헤드 예측적 프리로딩 오류: {str(e)}")
            
    def _get_related_heads(self, head_name: str) -> List[str]:
        """관련 헤드 목록 반환 (헤드 간 시너지 고려)"""
        related_map = {
            'emotion_head': ['empathy_head', 'sentiment_head'],
            'empathy_head': ['emotion_head', 'social_head'],
            'bentham_head': ['ethical_head', 'utility_head'],
            'semantic_head': ['linguistic_head', 'context_head'],
            'surd_head': ['causal_head', 'uncertainty_head']
        }
        
        return related_map.get(head_name, [])
    
    def _get_dynamic_preload_threshold(self, gpu_usage_percent: float) -> float:
        """GPU 사용률에 따른 동적 프리로딩 임계값 계산"""
        if gpu_usage_percent < 60:
            # 60% 미만: 적극적 프리로딩 허용
            return 75.0
        elif gpu_usage_percent < 70:
            # 60-70%: 보수적 프리로딩 
            return 65.0  
        elif gpu_usage_percent < 80:
            # 70-80%: 매우 제한적 프리로딩
            return 60.0
        else:
            # 80% 초과: 프리로딩 금지
            return 0.0
    
    async def _smart_memory_management(self, current_usage_percent: float):
        """스마트 메모리 관리 - 80% 초과시 적극적 언로드"""
        if current_usage_percent > 80:
            logger.info(f"GPU 사용률 위험 수준: {current_usage_percent:.1f}%, 적극적 언로드 시작")
            
            # 우선순위 낮은 모델들 언로드 (CRITICAL 제외)
            models_to_unload = []
            for name, model in list(self.gpu_resident_models.items()):
                if self.models[name].priority != SwapPriority.CRITICAL:
                    models_to_unload.append((
                        name, 
                        self.models[name].last_access,
                        self.models[name].priority
                    ))
            
            # 접근 시간과 우선순위 기준 정렬 (오래되고 우선순위 낮은 것부터)
            models_to_unload.sort(key=lambda x: (x[2].value, x[1]))
            
            # 75% 이하로 떨어질 때까지 언로드
            target_usage = 75.0
            unloaded_count = 0
            
            for name, _, priority in models_to_unload:
                if current_usage_percent <= target_usage:
                    break
                    
                await self.unload_model_from_gpu(name)
                unloaded_count += 1
                
                # 언로드 후 사용률 재확인
                gpu_info = get_gpu_memory_info()
                if gpu_info:
                    current_usage_percent = gpu_info['usage_percent']
                    logger.debug(f"언로드 후 GPU 사용률: {current_usage_percent:.1f}%")
            
            if unloaded_count > 0:
                logger.info(f"스마트 메모리 관리 완료: {unloaded_count}개 모델 언로드")
                
            # 추가 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
    
    async def _background_head_preload(self, head_name: str):
        """백그라운드 헤드 프리로딩 - 동적 임계값 적용"""
        try:
            # 잠시 대기 (현재 헤드 로딩 완료 후)
            await asyncio.sleep(1.0)
            
            # GPU 메모리 상태 확인
            gpu_info = get_gpu_memory_info()
            if not gpu_info:
                self.stats['preload_misses'] += 1
                return
                
            current_usage = gpu_info['usage_percent']
            
            # 80% 초과시 스마트 메모리 관리 수행
            if current_usage > 80:
                await self._smart_memory_management(current_usage)
                self.stats['preload_misses'] += 1
                return
            
            # 동적 임계값 계산
            threshold = self._get_dynamic_preload_threshold(current_usage)
            
            if threshold > 0 and current_usage < threshold:
                await self.load_head_to_gpu(head_name)
                self.stats['preload_hits'] += 1
                logger.debug(f"헤드 예측적 프리로딩 성공: {head_name} (사용률: {current_usage:.1f}%, 임계값: {threshold:.1f}%)")
            else:
                self.stats['preload_misses'] += 1
                logger.debug(f"헤드 예측적 프리로딩 제한: {head_name} (사용률: {current_usage:.1f}%, 임계값: {threshold:.1f}%)")
                
        except Exception as e:
            self.stats['preload_misses'] += 1
            logger.debug(f"헤드 예측적 프리로딩 실패: {head_name}, {str(e)}")
            
    async def _background_cleanup(self):
        """백그라운드 정리 작업"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(30)  # 30초마다 실행
                
                # 메모리 정리
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                # 통계 로깅
                if hasattr(self, 'stats'):
                    hit_rate = self.stats['cache_hits'] / max(1, self.stats['cache_hits'] + self.stats['cache_misses'])
                    logger.debug(f"스왑 통계 - 히트율: {hit_rate:.1%}, 총 스왑: {self.stats['total_swaps']}")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"백그라운드 정리 오류: {str(e)}")
                
    def _calculate_model_size(self, model: nn.Module) -> float:
        """모델 크기 계산 (MB)"""
        if model is None:
            return 0.0
            
        # wrapper 클래스 처리: 내부의 실제 nn.Module 찾기
        actual_model = model
        
        # parameters() 메서드가 없으면 내부 모델 찾기
        if not hasattr(model, 'parameters'):
            # 일반적인 속성명들 시도
            for attr_name in ['model', 'models', 'network', 'net', '_model', '_network']:
                if hasattr(model, attr_name):
                    candidate = getattr(model, attr_name)
                    # dict인 경우 첫 번째 값 시도
                    if isinstance(candidate, dict) and len(candidate) > 0:
                        candidate = next(iter(candidate.values()))
                    # nn.Module인지 확인
                    if hasattr(candidate, 'parameters'):
                        actual_model = candidate
                        logger.debug(f"Wrapper 클래스에서 실제 모델 찾음: {attr_name}")
                        break
            else:
                # NO FALLBACK - nn.Module이 아니면 등록 불가
                raise RuntimeError(f"nn.Module이 아닌 객체는 등록할 수 없음: {type(model).__name__}")
        
        try:
            param_size = sum(p.numel() * p.element_size() for p in actual_model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in actual_model.buffers())
            return (param_size + buffer_size) / (1024 ** 2)
        except Exception as e:
            logger.error(f"모델 크기 계산 실패: {e}")
            # 실패 시에도 GPU 모델이면 추정치 반환
            if hasattr(model, 'device') and str(model.device).startswith('cuda'):
                return 700.0
            return 0.0
        
    def _update_access_stats(self, name: str, hit: bool):
        """접근 통계 업데이트"""
        if hit:
            self.stats['cache_hits'] += 1
        else:
            self.stats['cache_misses'] += 1
            
    def _update_swap_stats(self, success: bool, swap_time: float):
        """스왑 통계 업데이트"""
        self.stats['total_swaps'] += 1
        self.stats['total_swap_time'] += swap_time
        
        if success:
            self.stats['successful_swaps'] += 1
        else:
            self.stats['failed_swaps'] += 1
            
    def get_stats(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        hit_rate = self.stats['cache_hits'] / max(1, self.stats['cache_hits'] + self.stats['cache_misses'])
        avg_swap_time = self.stats['total_swap_time'] / max(1, self.stats['total_swaps'])
        
        return {
            **self.stats,
            'cache_hit_rate': hit_rate,
            'average_swap_time': avg_swap_time,
            'gpu_resident_models': list(self.gpu_resident_models.keys()),
            'ram_models': list(self.ram_models.keys()),
            'total_models': len(self.models)
        }
        
    def get_memory_status(self) -> Dict[str, Any]:
        """메모리 상태 반환"""
        gpu_info = get_gpu_memory_info()
        
        gpu_model_sizes = sum(
            self.models[name].size_mb for name in self.gpu_resident_models
        )
        
        ram_model_sizes = sum(
            model.size_mb for model in self.ram_models.values()
        )
        
        return {
            'gpu_info': gpu_info,
            'gpu_model_total_mb': gpu_model_sizes,
            'ram_model_total_mb': ram_model_sizes,
            'gpu_utilization_percent': gpu_info['usage_percent'] if gpu_info else 0,
            'models_on_gpu': len(self.gpu_resident_models),
            'models_on_ram': len(self.ram_models)
        }
        
    async def free_gpu_space_intelligent(self, required_mb: int, 
                                          target_usage: float = 0.85, 
                                          hard_timeout_s: float = 45.0) -> bool:
        """
        지능적 GPU 공간 확보 - LRU + 우선순위 기반
        목표 사용률에 도달할 때까지 우선순위 언로드를 반복
        
        Args:
            required_mb: 필요한 메모리 (MB)
            target_usage: 목표 GPU 사용률 (0.0~1.0)
            hard_timeout_s: 타임아웃 시간 (초)
            
        Returns:
            bool: 성공 시 True, 타임아웃/실패 시 False
        """
        logger.info(f"[지능적 GPU 공간 확보] {required_mb}MB 필요, 목표 사용률: {target_usage*100:.1f}%")
        
        start_time = time.time()
        
        # 타임아웃까지 반복
        while True:
            # 타임아웃 체크
            if (time.time() - start_time) > hard_timeout_s:
                logger.warning(f"[타임아웃] {hard_timeout_s}초 초과")
                return False
            
            # 현재 GPU 상태 확인
            gpu_info = get_gpu_memory_info()
            if not gpu_info:
                logger.error("GPU 정보를 가져올 수 없습니다")
                return False
                
            current_free_mb = gpu_info['free_mb']
            current_usage = gpu_info['usage_percent'] / 100.0  # 0.0~1.0 비율로 변환
            logger.info(f"[현재 GPU] 여유: {current_free_mb}MB, 사용률: {current_usage*100:.1f}%")
            
            # 목표 사용률 달성 및 충분한 공간 확인
            if current_usage <= target_usage and current_free_mb >= required_mb:
                logger.info(f"[목표 달성] 사용률: {current_usage*100:.1f}% <= {target_usage*100:.1f}%, 여유: {current_free_mb}MB >= {required_mb}MB")
                return True
            
            # 해제해야 할 메모리 (int 캐스팅으로 안정화) + safety_margin
            safety_margin_mb = 256  # 여유 버퍼 확보
            need_to_free_mb = int(max(required_mb - current_free_mb + safety_margin_mb, 
                                      (current_usage - target_usage) * gpu_info['total_mb']))  # 목표 사용률 고려
            logger.info(f"[해제 필요] {need_to_free_mb}MB")
            
            # 언로드 후보 선정 (LRU + 우선순위)
            unload_candidates = []
            
            # 워크플로우 보호 모델 가져오기
            protected_models = set()
            if self.workflow_aware and hasattr(self.workflow_memory_manager, 'workflow_tracker'):
                protected_models = self.workflow_memory_manager.workflow_tracker.get_protected_models()
                logger.debug(f"[워크플로우 보호] {len(protected_models)}개 모델")
            
            # GPU에 있는 모델들 중 후보 선정
            for name, model_info in self.models.items():
                if name not in self.gpu_resident_models:
                    continue
                    
                # CRITICAL 우선순위는 언로드 불가 (백본과 translator만)
                if model_info.priority == SwapPriority.CRITICAL:
                    logger.debug(f"[CRITICAL 보호] {name}은 언로드 불가")
                    continue
                    
                # 워크플로우 보호 모델은 건너뛰기
                if name in protected_models:
                    continue
                    
                # device_policy가 cpu_preload인 모델 우선 언로드
                is_cpu_preload = getattr(model_info, 'device_policy', '') == 'cpu_preload'
                
                unload_candidates.append({
                    'name': name,
                    'size_mb': model_info.size_mb,
                    'priority': model_info.priority.value,
                    'last_access': model_info.last_access,
                    'is_cpu_preload': is_cpu_preload
                })
            
            # 후보가 없으면 더 이상 언로드 불가 - 무한 루프 방지
            if not unload_candidates:
                logger.warning("[언로드 불가] 더 이상 언로드할 수 있는 모델이 없습니다")
                logger.warning(f"[DSM] GPU 사용률 {current_usage*100:.1f}%로 목표 {target_usage*100:.1f}% 달성 실패")
                return False
            
            # 우선순위 안정적 변환을 위한 헬퍼 함수
            def _prio_rank(p):
                s = str(getattr(p, "value", p)).lower()
                if "critical" in s: return 0
                if "high" in s: return 1
                if "mid" in s or "medium" in s: return 2
                if "low" in s: return 3
                try:
                    return int(p)  # 숫자형이면 그대로
                except Exception:
                    return 4
            
            # 정렬: cpu_preload > LOW 우선순위 > LRU
            unload_candidates.sort(key=lambda x: (
                not x['is_cpu_preload'],  # cpu_preload 먼저
                _prio_rank(x['priority']),  # 우선순위 (안정화된 정렬)
                x['last_access']          # 오래된 것 먼저
            ))
            
            # 첫 번째 후보 하나만 언로드 (반복 루프에서 점진적으로 처리)
            candidate = unload_candidates[0]
            name = candidate['name']
            size_mb = candidate['size_mb']
            
            logger.info(f"[언로드] {name} ({size_mb}MB) - {candidate['priority']}")
            
            try:
                # GPU에서 CPU로 이동
                await self.unload_model_from_gpu(name)
                
                # GPU 메모리 정리
                torch.cuda.empty_cache()
                gc.collect()
                
                # 상태 확인
                gpu_info = get_gpu_memory_info()
                logger.info(f"[언로드 후] 여유: {gpu_info['free_mb']}MB (+{size_mb}MB)")
                logger.info(f"[GPU MEM] after swap-out {name}: alloc={gpu_info['allocated_mb']/1024:.1f}GB reserved={gpu_info['cached_mb']/1024:.1f}GB util={gpu_info['usage_percent']:.1f}%")
                
                # 다음 루프로 계속
                await asyncio.sleep(0.1)  # 짧은 대기로 시스템 안정화
                # break 제거 - while 루프 처음으로 돌아가 usage 재평가
                
            except Exception as e:
                logger.error(f"{name} 언로드 실패: {e}")
                # 실패한 경우 다음 후보로 계속 시도
            
            # 루프 끝에서 사용률이 여전히 높으면 CRITICAL 외 모델 강제 언로드
            # (while 루프 안에 있어야 함 - 들여쓰기 수정)
            gpu_info = get_gpu_memory_info()
            if gpu_info:
                current_usage = gpu_info['usage_percent'] / 100.0
                current_free_mb = gpu_info['free_mb']
                
                # 목표 달성 확인 (강제 언로드 후 재확인)
                if current_usage <= target_usage and current_free_mb >= required_mb:
                    logger.info(f"[목표 달성] 사용률: {current_usage*100:.1f}% <= {target_usage*100:.1f}%, 여유: {current_free_mb}MB >= {required_mb}MB")
                    return True
                
                # 아직 목표 미달성시 강제 언로드
                if current_usage > target_usage:
                    logger.warning(f"[강제 언로드] 사용률 {current_usage*100:.1f}% > 목표 {target_usage*100:.1f}%")
                    
                    # CRITICAL이 아닌 첫 번째 모델 강제 언로드
                    for name, model_info in self.models.items():
                        if name not in self.gpu_resident_models:
                            continue
                        if model_info.priority != SwapPriority.CRITICAL:
                            logger.info(f"[강제 언로드] {name} (priority: {model_info.priority.value})")
                            await self.unload_model_from_gpu(name)
                            # while 루프 처음으로 돌아가 usage 재평가
                            logger.debug(f"[DSM] Re-check GPU usage after forced unload")
                            break  # for 루프만 탈출, while 루프는 계속
        
    @classmethod
    def get_instance(cls):
        """싱글톤 인스턴스 반환"""
        if not hasattr(cls, '_instance'):
            cls._instance = cls()
        return cls._instance

# 컨텍스트 매니저
class SwapContext:
    """스왑 컨텍스트 매니저 - with 문으로 사용"""
    
    def __init__(self, swap_manager: RedHeartDynamicSwapManager, model_names: List[str]):
        self.swap_manager = swap_manager
        self.model_names = model_names
        self.loaded_models = {}
        
    async def __aenter__(self):
        """컨텍스트 진입"""
        for name in self.model_names:
            model = await self.swap_manager.get_model(name)
            self.loaded_models[name] = model
        return self.loaded_models
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 종료"""
        # 필요에 따라 모델들을 언로드할 수 있지만, 
        # 예측적 프리로딩을 위해 즉시 언로드하지 않음
        pass

# 사용 예시 함수
async def example_usage():
    """동적 스왑 매니저 사용 예시"""
    swap_manager = RedHeartDynamicSwapManager()
    await swap_manager.initialize()
    
    try:
        # 가상의 모델 등록
        dummy_model = nn.Linear(1000, 1000)
        swap_manager.register_model("test_model", dummy_model, SwapPriority.HIGH)
        
        # 모델 사용
        async with SwapContext(swap_manager, ["test_model"]) as models:
            test_model = models["test_model"]
            # 모델 사용...
            
        # 통계 출력
        stats = swap_manager.get_stats()
        memory_status = swap_manager.get_memory_status()
        
        print("스왑 통계:", stats)
        print("메모리 상태:", memory_status)
        
    finally:
        await swap_manager.shutdown()

# 호환성을 위한 alias 제공
DynamicSwapManager = RedHeartDynamicSwapManager

# 전역 싱글톤 인스턴스
_swap_mgr = None

def set_swap_manager(inst: Union["RedHeartDynamicSwapManager", "DynamicSwapManager"]):
    """외부(오케스트레이터)에서 생성한 DSM을 전역 공유 인스턴스로 고정."""
    global _swap_mgr
    _swap_mgr = inst
    logger.info(f"[DSM] Global swap manager set -> id={id(inst)}")

def get_swap_manager() -> DynamicSwapManager:
    """전역 스왑 매니저 인스턴스 반환 (외부 호출부 기대 심볼)"""
    global _swap_mgr
    if _swap_mgr is None:
        _swap_mgr = DynamicSwapManager.get_instance()
    return _swap_mgr

__all__ = ["RedHeartDynamicSwapManager", "DynamicSwapManager", "get_swap_manager"]

if __name__ == "__main__":
    asyncio.run(example_usage())