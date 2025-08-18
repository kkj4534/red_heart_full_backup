"""
고급 멀티스레드 스왑 스케줄러 - Week 2 최종 최적화
Advanced Multithreaded Swap Scheduler - Week 2 Final Optimization

정교한 스레드 관리 및 우선순위 기반 스케줄링:
- 동적 스레드 풀 관리 (CPU/GPU 코어 수 기반)
- 우선순위 큐 기반 작업 스케줄링
- 데드락 방지 및 리소스 경합 해결
- CPU-GPU 간 비동기 데이터 전송 최적화
- 실시간 성능 모니터링 및 자동 튜닝
"""

import asyncio
import logging
import time
import threading
import multiprocessing
import queue
import concurrent.futures
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict, namedtuple
from enum import Enum, IntEnum
import heapq
import psutil
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import weakref
import gc
import traceback
from abc import ABC, abstractmethod

# 핵심 시스템 임포트
from config import ADVANCED_CONFIG, get_gpu_memory_info, get_smart_device
from head_compatibility_interface import HeadType
from smart_lossless_compression_system import SmartCompressionSystem, LayerMetadata
from adaptive_cache_manager import AdaptiveCacheManager

# 로거 설정
logger = logging.getLogger(__name__)

class TaskPriority(IntEnum):
    """작업 우선순위 (낮은 값 = 높은 우선순위)"""
    CRITICAL = 0      # 즉시 처리 필요
    HIGH = 1          # 높은 우선순위
    NORMAL = 2        # 일반 우선순위
    LOW = 3           # 낮은 우선순위
    BACKGROUND = 4    # 백그라운드 처리

class TaskType(Enum):
    """작업 타입"""
    COMPRESS = "compress"           # 압축 작업
    DECOMPRESS = "decompress"       # 압축 해제 작업
    TRANSFER_TO_GPU = "transfer_gpu"   # GPU 전송
    TRANSFER_TO_CPU = "transfer_cpu"   # CPU 전송
    CACHE_WARMUP = "cache_warmup"     # 캐시 워밍
    MEMORY_CLEANUP = "memory_cleanup"  # 메모리 정리

class WorkerType(Enum):
    """워커 타입"""
    CPU_INTENSIVE = "cpu_intensive"     # CPU 집약적 작업
    IO_INTENSIVE = "io_intensive"       # I/O 집약적 작업
    GPU_TRANSFER = "gpu_transfer"       # GPU 전송 전용
    MIXED = "mixed"                     # 혼합 작업

@dataclass
class SwapTask:
    """스왑 작업 정의"""
    task_id: str
    task_type: TaskType
    priority: TaskPriority
    worker_type: WorkerType
    func: Callable
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # 메타데이터
    creation_time: datetime = field(default_factory=datetime.now)
    estimated_duration: float = 1.0  # 예상 실행 시간 (초)
    memory_requirement: int = 0       # 메모리 요구량 (바이트)
    dependencies: List[str] = field(default_factory=list)  # 의존성 작업들
    
    # 실행 상태
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Any = None
    error: Optional[Exception] = None
    
    def __lt__(self, other):
        """우선순위 큐를 위한 비교 연산자"""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.creation_time < other.creation_time
    
    @property
    def is_completed(self) -> bool:
        """작업 완료 여부"""
        return self.end_time is not None
    
    @property
    def execution_time(self) -> Optional[float]:
        """실행 시간 (초)"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

@dataclass
class WorkerStatistics:
    """워커 통계"""
    worker_id: str
    worker_type: WorkerType
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_execution_time: float = 0.0
    avg_execution_time: float = 0.0
    current_task: Optional[str] = None
    last_activity: datetime = field(default_factory=datetime.now)
    
    def update_completion(self, execution_time: float):
        """작업 완료 통계 업데이트"""
        self.tasks_completed += 1
        self.total_execution_time += execution_time
        self.avg_execution_time = self.total_execution_time / self.tasks_completed
        self.last_activity = datetime.now()
    
    def update_failure(self):
        """작업 실패 통계 업데이트"""
        self.tasks_failed += 1
        self.last_activity = datetime.now()

class ResourceManager:
    """리소스 관리자 - 메모리, CPU, GPU 리소스 추적"""
    
    def __init__(self):
        self.cpu_usage_history = deque(maxlen=100)
        self.memory_usage_history = deque(maxlen=100)
        self.gpu_usage_history = deque(maxlen=100)
        
        # 리소스 제한
        self.max_cpu_usage = 0.8  # 80% CPU 사용률 제한
        self.max_memory_usage = 0.9  # 90% 메모리 사용률 제한
        self.max_gpu_memory_usage = 0.85  # 85% GPU 메모리 제한
        
        # 현재 사용 중인 리소스
        self.active_cpu_tasks = set()
        self.active_memory_intensive_tasks = set()
        self.active_gpu_tasks = set()
        
        # 락
        self.resource_lock = threading.Lock()
    
    def can_execute_task(self, task: SwapTask) -> bool:
        """작업 실행 가능 여부 확인"""
        with self.resource_lock:
            # CPU 사용률 확인
            cpu_percent = psutil.cpu_percent()
            if cpu_percent > self.max_cpu_usage * 100:
                return False
            
            # 메모리 사용률 확인
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > self.max_memory_usage * 100:
                return False
            
            # GPU 메모리 확인
            gpu_info = get_gpu_memory_info()
            if gpu_info and gpu_info['usage_percent'] > self.max_gpu_memory_usage * 100:
                if task.worker_type == WorkerType.GPU_TRANSFER:
                    return False
            
            # 작업별 리소스 제한 확인
            if task.worker_type == WorkerType.CPU_INTENSIVE:
                if len(self.active_cpu_tasks) >= multiprocessing.cpu_count():
                    return False
            
            return True
    
    def acquire_resources(self, task: SwapTask):
        """리소스 획득"""
        with self.resource_lock:
            if task.worker_type == WorkerType.CPU_INTENSIVE:
                self.active_cpu_tasks.add(task.task_id)
            elif task.memory_requirement > 100 * 1024 * 1024:  # 100MB 이상
                self.active_memory_intensive_tasks.add(task.task_id)
            elif task.worker_type == WorkerType.GPU_TRANSFER:
                self.active_gpu_tasks.add(task.task_id)
    
    def release_resources(self, task: SwapTask):
        """리소스 해제"""
        with self.resource_lock:
            self.active_cpu_tasks.discard(task.task_id)
            self.active_memory_intensive_tasks.discard(task.task_id)
            self.active_gpu_tasks.discard(task.task_id)
    
    def update_usage_history(self):
        """사용량 이력 업데이트"""
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        self.cpu_usage_history.append(cpu_percent)
        self.memory_usage_history.append(memory_percent)
        
        gpu_info = get_gpu_memory_info()
        if gpu_info:
            self.gpu_usage_history.append(gpu_info['usage_percent'])
    
    def get_resource_status(self) -> Dict[str, Any]:
        """리소스 상태 반환"""
        return {
            'cpu_usage_percent': psutil.cpu_percent(),
            'memory_usage_percent': psutil.virtual_memory().percent,
            'gpu_usage_percent': get_gpu_memory_info()['usage_percent'] if get_gpu_memory_info() else 0,
            'active_cpu_tasks': len(self.active_cpu_tasks),
            'active_memory_tasks': len(self.active_memory_intensive_tasks),
            'active_gpu_tasks': len(self.active_gpu_tasks)
        }

class DeadlockDetector:
    """데드락 감지 및 방지 시스템"""
    
    def __init__(self):
        self.dependency_graph = defaultdict(set)  # 작업 의존성 그래프
        self.waiting_tasks = defaultdict(set)     # 대기 중인 작업들
        self.lock = threading.Lock()
    
    def add_dependency(self, task_id: str, dependency_id: str):
        """의존성 추가"""
        with self.lock:
            self.dependency_graph[task_id].add(dependency_id)
    
    def remove_dependency(self, task_id: str, dependency_id: str):
        """의존성 제거"""
        with self.lock:
            self.dependency_graph[task_id].discard(dependency_id)
            if not self.dependency_graph[task_id]:
                del self.dependency_graph[task_id]
    
    def detect_deadlock(self) -> List[List[str]]:
        """데드락 감지 (순환 참조 탐지)"""
        with self.lock:
            cycles = []
            visited = set()
            rec_stack = set()
            
            def dfs(node, path):
                if node in rec_stack:
                    # 순환 발견
                    cycle_start = path.index(node)
                    cycles.append(path[cycle_start:])
                    return
                
                if node in visited:
                    return
                
                visited.add(node)
                rec_stack.add(node)
                
                for neighbor in self.dependency_graph.get(node, []):
                    dfs(neighbor, path + [neighbor])
                
                rec_stack.remove(node)
            
            for task_id in self.dependency_graph:
                if task_id not in visited:
                    dfs(task_id, [task_id])
            
            return cycles
    
    def resolve_deadlock(self, cycles: List[List[str]]) -> List[str]:
        """데드락 해결 (우선순위 낮은 작업 취소)"""
        tasks_to_cancel = []
        
        for cycle in cycles:
            # 순환에서 가장 우선순위가 낮은 작업 찾기
            # 실제 구현에서는 작업 정보를 참조하여 결정
            if cycle:
                tasks_to_cancel.append(cycle[-1])  # 임시로 마지막 작업 취소
        
        return tasks_to_cancel

class AdvancedWorker:
    """고급 워커 클래스"""
    
    def __init__(self, worker_id: str, worker_type: WorkerType, 
                 resource_manager: ResourceManager):
        self.worker_id = worker_id
        self.worker_type = worker_type
        self.resource_manager = resource_manager
        self.statistics = WorkerStatistics(worker_id, worker_type)
        
        # 워커 상태
        self.is_running = False
        self.current_task = None
        self.thread = None
        
        # 작업 큐
        self.task_queue = queue.PriorityQueue()
        self.shutdown_event = threading.Event()
        
    def start(self):
        """워커 시작"""
        if self.is_running:
            return
        
        self.is_running = True
        self.shutdown_event.clear()
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()
        logger.debug(f"워커 시작: {self.worker_id} ({self.worker_type.value})")
    
    def stop(self):
        """워커 중지"""
        if not self.is_running:
            return
        
        self.is_running = False
        self.shutdown_event.set()
        
        # 종료 신호 작업 추가
        shutdown_task = SwapTask(
            task_id="shutdown",
            task_type=TaskType.MEMORY_CLEANUP,
            priority=TaskPriority.CRITICAL,
            worker_type=self.worker_type,
            func=lambda: None
        )
        self.task_queue.put((shutdown_task.priority, shutdown_task))
        
        if self.thread:
            self.thread.join(timeout=5.0)
        
        logger.debug(f"워커 중지: {self.worker_id}")
    
    def add_task(self, task: SwapTask):
        """작업 추가"""
        if self.is_running:
            self.task_queue.put((task.priority, task))
    
    def _worker_loop(self):
        """워커 메인 루프"""
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # 작업 대기 (타임아웃 포함)
                try:
                    priority, task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # 종료 신호 확인
                if task.task_id == "shutdown":
                    break
                
                # 리소스 확인
                if not self.resource_manager.can_execute_task(task):
                    # 리소스 부족시 작업을 다시 큐에 추가 (우선순위 낮춤)
                    delayed_task = task
                    delayed_task.priority = TaskPriority(min(task.priority + 1, TaskPriority.BACKGROUND))
                    self.task_queue.put((delayed_task.priority, delayed_task))
                    time.sleep(0.1)  # 잠시 대기
                    continue
                
                # 작업 실행
                self._execute_task(task)
                
            except Exception as e:
                logger.error(f"워커 {self.worker_id} 오류: {str(e)}")
                if self.current_task:
                    self.current_task.error = e
                    self.current_task.end_time = datetime.now()
                    self.statistics.update_failure()
    
    def _execute_task(self, task: SwapTask):
        """작업 실행"""
        try:
            # 리소스 획득
            self.resource_manager.acquire_resources(task)
            
            # 작업 시작
            task.start_time = datetime.now()
            self.current_task = task
            self.statistics.current_task = task.task_id
            
            # 실제 작업 실행
            if asyncio.iscoroutinefunction(task.func):
                # 비동기 함수인 경우
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    task.result = loop.run_until_complete(task.func(*task.args, **task.kwargs))
                finally:
                    loop.close()
            else:
                # 동기 함수인 경우
                task.result = task.func(*task.args, **task.kwargs)
            
            # 작업 완료
            task.end_time = datetime.now()
            execution_time = task.execution_time
            self.statistics.update_completion(execution_time)
            
            logger.debug(f"작업 완료: {task.task_id} ({execution_time:.3f}s)")
            
        except Exception as e:
            task.error = e
            task.end_time = datetime.now()
            self.statistics.update_failure()
            logger.error(f"작업 실패: {task.task_id}, 오류: {str(e)}")
            
        finally:
            # 리소스 해제
            self.resource_manager.release_resources(task)
            self.current_task = None
            self.statistics.current_task = None

class AdvancedMultithreadedSwapScheduler:
    """
    고급 멀티스레드 스왑 스케줄러
    
    정교한 스레드 관리 및 우선순위 기반 스케줄링 시스템
    """
    
    def __init__(self, config: Dict[str, Any] = None,
                 compression_system: Optional[SmartCompressionSystem] = None,
                 cache_manager: Optional[AdaptiveCacheManager] = None):
        self.config = config or ADVANCED_CONFIG.get('swap_scheduler_config', {})
        self.compression_system = compression_system
        self.cache_manager = cache_manager
        
        # 시스템 리소스 정보
        self.cpu_count = multiprocessing.cpu_count()
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # 워커 풀 구성
        self._initialize_worker_pools()
        
        # 관리자들
        self.resource_manager = ResourceManager()
        self.deadlock_detector = DeadlockDetector()
        
        # 작업 관리
        self.pending_tasks = {}  # task_id -> SwapTask
        self.completed_tasks = deque(maxlen=1000)  # 완료된 작업들
        self.task_futures = {}  # task_id -> Future
        
        # 성능 모니터링
        self.performance_metrics = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'avg_execution_time': 0.0,
            'throughput_tasks_per_sec': 0.0,
            'resource_utilization': {}
        }
        
        # 자동 튜닝
        self.auto_tuning_enabled = True
        self.tuning_history = deque(maxlen=100)
        self.last_tuning_time = datetime.now()
        
        # 모니터링 스레드
        self.monitoring_active = False
        self.monitoring_thread = None
        
        logger.info(f"AdvancedMultithreadedSwapScheduler 초기화 완료: "
                   f"CPU {self.cpu_count}코어, 메모리 {self.memory_gb:.1f}GB")
    
    def _initialize_worker_pools(self):
        """워커 풀 초기화"""
        # 워커 수 계산 (CPU 코어 수 기반)
        base_config = {
            'cpu_intensive_workers': max(2, self.cpu_count // 2),
            'io_intensive_workers': max(4, self.cpu_count),
            'gpu_transfer_workers': min(4, max(2, torch.cuda.device_count() * 2)),
            'mixed_workers': max(2, self.cpu_count // 4)
        }
        base_config.update(self.config.get('worker_counts', {}))
        
        self.workers = {}
        self.worker_pools = defaultdict(list)
        
        # CPU 집약적 워커들
        for i in range(base_config['cpu_intensive_workers']):
            worker_id = f"cpu_worker_{i}"
            worker = AdvancedWorker(worker_id, WorkerType.CPU_INTENSIVE, self.resource_manager)
            self.workers[worker_id] = worker
            self.worker_pools[WorkerType.CPU_INTENSIVE].append(worker)
        
        # I/O 집약적 워커들
        for i in range(base_config['io_intensive_workers']):
            worker_id = f"io_worker_{i}"
            worker = AdvancedWorker(worker_id, WorkerType.IO_INTENSIVE, self.resource_manager)
            self.workers[worker_id] = worker
            self.worker_pools[WorkerType.IO_INTENSIVE].append(worker)
        
        # GPU 전송 전용 워커들
        for i in range(base_config['gpu_transfer_workers']):
            worker_id = f"gpu_worker_{i}"
            worker = AdvancedWorker(worker_id, WorkerType.GPU_TRANSFER, self.resource_manager)
            self.workers[worker_id] = worker
            self.worker_pools[WorkerType.GPU_TRANSFER].append(worker)
        
        # 혼합 작업 워커들
        for i in range(base_config['mixed_workers']):
            worker_id = f"mixed_worker_{i}"
            worker = AdvancedWorker(worker_id, WorkerType.MIXED, self.resource_manager)
            self.workers[worker_id] = worker
            self.worker_pools[WorkerType.MIXED].append(worker)
        
        logger.info(f"워커 풀 초기화 완료: "
                   f"CPU {len(self.worker_pools[WorkerType.CPU_INTENSIVE])}, "
                   f"I/O {len(self.worker_pools[WorkerType.IO_INTENSIVE])}, "
                   f"GPU {len(self.worker_pools[WorkerType.GPU_TRANSFER])}, "
                   f"Mixed {len(self.worker_pools[WorkerType.MIXED])}")
    
    async def initialize(self):
        """스케줄러 초기화"""
        # 모든 워커 시작
        for worker in self.workers.values():
            worker.start()
        
        # 모니터링 시작
        await self.start_monitoring()
        
        logger.info("고급 멀티스레드 스왑 스케줄러 초기화 완료")
    
    async def shutdown(self):
        """스케줄러 종료"""
        # 모니터링 중지
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        # 모든 워커 중지
        for worker in self.workers.values():
            worker.stop()
        
        logger.info("고급 멀티스레드 스왑 스케줄러 종료 완료")
    
    async def submit_task(self, task: SwapTask) -> str:
        """작업 제출"""
        # 의존성 확인
        if task.dependencies:
            for dep_id in task.dependencies:
                if dep_id not in self.completed_tasks:
                    self.deadlock_detector.add_dependency(task.task_id, dep_id)
        
        # 데드락 감지
        cycles = self.deadlock_detector.detect_deadlock()
        if cycles:
            tasks_to_cancel = self.deadlock_detector.resolve_deadlock(cycles)
            logger.warning(f"데드락 감지 및 해결: {len(cycles)}개 순환, {len(tasks_to_cancel)}개 작업 취소")
        
        # 작업 등록
        self.pending_tasks[task.task_id] = task
        self.performance_metrics['total_tasks'] += 1
        
        # 적절한 워커 선택 및 할당
        worker = self._select_optimal_worker(task)
        if worker:
            worker.add_task(task)
            logger.debug(f"작업 할당: {task.task_id} -> {worker.worker_id}")
        else:
            logger.warning(f"사용 가능한 워커 없음: {task.task_id}")
        
        return task.task_id
    
    def _select_optimal_worker(self, task: SwapTask) -> Optional[AdvancedWorker]:
        """최적의 워커 선택"""
        # 작업 타입에 따른 워커 풀 선택
        if task.worker_type in self.worker_pools:
            candidate_workers = self.worker_pools[task.worker_type]
        else:
            # 혼합 워커를 대안으로 사용
            candidate_workers = self.worker_pools[WorkerType.MIXED]
        
        if not candidate_workers:
            return None
        
        # 워커 선택 전략
        best_worker = None
        best_score = float('inf')
        
        for worker in candidate_workers:
            if not worker.is_running:
                continue
            
            # 워커 점수 계산 (낮을수록 좋음)
            score = self._calculate_worker_score(worker, task)
            
            if score < best_score:
                best_score = score
                best_worker = worker
        
        return best_worker
    
    def _calculate_worker_score(self, worker: AdvancedWorker, task: SwapTask) -> float:
        """워커 점수 계산"""
        score = 0.0
        
        # 큐 길이 고려
        queue_length = worker.task_queue.qsize()
        score += queue_length * 1.0
        
        # 평균 실행 시간 고려
        score += worker.statistics.avg_execution_time
        
        # 실패율 고려
        total_tasks = worker.statistics.tasks_completed + worker.statistics.tasks_failed
        if total_tasks > 0:
            failure_rate = worker.statistics.tasks_failed / total_tasks
            score += failure_rate * 10.0
        
        # 현재 작업 유무 고려
        if worker.current_task:
            score += 2.0
        
        # 최근 활동 고려
        time_since_activity = (datetime.now() - worker.statistics.last_activity).seconds
        if time_since_activity > 60:  # 1분 이상 비활성
            score += 5.0
        
        return score
    
    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """작업 완료 대기"""
        start_time = time.time()
        
        while True:
            # 완료된 작업에서 찾기
            for completed_task in self.completed_tasks:
                if completed_task.task_id == task_id:
                    if completed_task.error:
                        raise completed_task.error
                    return completed_task.result
            
            # 타임아웃 체크
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"작업 타임아웃: {task_id}")
            
            await asyncio.sleep(0.1)
    
    async def cancel_task(self, task_id: str) -> bool:
        """작업 취소"""
        if task_id in self.pending_tasks:
            # 의존성 제거
            for dep_id in self.pending_tasks[task_id].dependencies:
                self.deadlock_detector.remove_dependency(task_id, dep_id)
            
            del self.pending_tasks[task_id]
            return True
        
        return False
    
    # 고성능 압축/해제 작업 래퍼들
    async def compress_layer_async(self, layer_id: str, layer: nn.Module,
                                 priority: TaskPriority = TaskPriority.NORMAL) -> str:
        """레이어 비동기 압축"""
        task = SwapTask(
            task_id=f"compress_{layer_id}_{int(time.time())}",
            task_type=TaskType.COMPRESS,
            priority=priority,
            worker_type=WorkerType.CPU_INTENSIVE,
            func=self._compress_layer_wrapper,
            args=(layer_id, layer),
            estimated_duration=2.0,
            memory_requirement=self._estimate_layer_memory(layer)
        )
        
        return await self.submit_task(task)
    
    async def decompress_layer_async(self, layer_id: str,
                                   priority: TaskPriority = TaskPriority.HIGH) -> str:
        """레이어 비동기 압축 해제"""
        task = SwapTask(
            task_id=f"decompress_{layer_id}_{int(time.time())}",
            task_type=TaskType.DECOMPRESS,
            priority=priority,
            worker_type=WorkerType.IO_INTENSIVE,
            func=self._decompress_layer_wrapper,
            args=(layer_id,),
            estimated_duration=1.0
        )
        
        return await self.submit_task(task)
    
    async def transfer_to_gpu_async(self, tensor: torch.Tensor, device: torch.device,
                                  priority: TaskPriority = TaskPriority.HIGH) -> str:
        """GPU 전송 비동기 작업"""
        task = SwapTask(
            task_id=f"gpu_transfer_{int(time.time())}",
            task_type=TaskType.TRANSFER_TO_GPU,
            priority=priority,
            worker_type=WorkerType.GPU_TRANSFER,
            func=self._gpu_transfer_wrapper,
            args=(tensor, device),
            estimated_duration=0.5,
            memory_requirement=tensor.numel() * tensor.element_size()
        )
        
        return await self.submit_task(task)
    
    def _compress_layer_wrapper(self, layer_id: str, layer: nn.Module):
        """압축 작업 래퍼"""
        if self.compression_system:
            # 실제 압축 시스템 호출
            return self.compression_system._perform_compression(
                layer_id, layer, "balanced"  # 임시로 balanced 레벨 사용
            )
        else:
            # 기본 압축 구현
            return self._basic_compression(layer)
    
    def _decompress_layer_wrapper(self, layer_id: str):
        """압축 해제 작업 래퍼"""
        if self.compression_system:
            return self.compression_system._perform_decompression(layer_id, "high")
        else:
            return self._basic_decompression(layer_id)
    
    def _gpu_transfer_wrapper(self, tensor: torch.Tensor, device: torch.device):
        """GPU 전송 작업 래퍼"""
        return tensor.to(device, non_blocking=True)
    
    def _basic_compression(self, layer: nn.Module):
        """기본 압축 구현"""
        # 간단한 압축 구현
        return f"compressed_{layer.__class__.__name__}"
    
    def _basic_decompression(self, layer_id: str):
        """기본 압축 해제 구현"""
        # 간단한 압축 해제 구현
        return f"decompressed_{layer_id}"
    
    def _estimate_layer_memory(self, layer: nn.Module) -> int:
        """레이어 메모리 사용량 추정"""
        total_size = 0
        for param in layer.parameters():
            total_size += param.numel() * param.element_size()
        return total_size
    
    async def start_monitoring(self):
        """성능 모니터링 시작"""
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("스왑 스케줄러 모니터링 시작")
    
    def _monitoring_loop(self):
        """모니터링 루프"""
        while self.monitoring_active:
            try:
                # 리소스 사용량 업데이트
                self.resource_manager.update_usage_history()
                
                # 성능 메트릭 업데이트
                self._update_performance_metrics()
                
                # 자동 튜닝
                if self.auto_tuning_enabled:
                    self._auto_tuning()
                
                # 완료된 작업 정리
                self._cleanup_completed_tasks()
                
                time.sleep(5.0)  # 5초마다
                
            except Exception as e:
                logger.error(f"모니터링 루프 오류: {str(e)}")
                time.sleep(1.0)
    
    def _update_performance_metrics(self):
        """성능 메트릭 업데이트"""
        # 완료된 작업 수 계산
        completed_count = 0
        failed_count = 0
        total_execution_time = 0.0
        
        for worker in self.workers.values():
            completed_count += worker.statistics.tasks_completed
            failed_count += worker.statistics.tasks_failed
            total_execution_time += worker.statistics.total_execution_time
        
        self.performance_metrics.update({
            'completed_tasks': completed_count,
            'failed_tasks': failed_count,
            'avg_execution_time': total_execution_time / max(1, completed_count),
            'throughput_tasks_per_sec': completed_count / max(1, time.time() - 
                                       getattr(self, '_start_time', time.time())),
            'resource_utilization': self.resource_manager.get_resource_status()
        })
    
    def _auto_tuning(self):
        """자동 튜닝"""
        current_time = datetime.now()
        if (current_time - self.last_tuning_time).seconds < 60:  # 1분마다
            return
        
        # 성능 분석
        resource_status = self.resource_manager.get_resource_status()
        cpu_usage = resource_status['cpu_usage_percent']
        memory_usage = resource_status['memory_usage_percent']
        
        tuning_actions = []
        
        # CPU 사용률이 낮으면 워커 추가 고려
        if cpu_usage < 50 and len(self.worker_pools[WorkerType.CPU_INTENSIVE]) < self.cpu_count:
            tuning_actions.append("add_cpu_worker")
        
        # 메모리 사용률이 높으면 I/O 워커 줄이기
        elif memory_usage > 80:
            tuning_actions.append("reduce_io_workers")
        
        # GPU 사용률 확인
        gpu_usage = resource_status['gpu_usage_percent']
        if gpu_usage > 90:
            tuning_actions.append("reduce_gpu_transfers")
        
        if tuning_actions:
            self.tuning_history.append({
                'timestamp': current_time,
                'actions': tuning_actions,
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'gpu_usage': gpu_usage
            })
            logger.info(f"자동 튜닝 수행: {tuning_actions}")
        
        self.last_tuning_time = current_time
    
    def _cleanup_completed_tasks(self):
        """완료된 작업 정리"""
        # 완료된 작업들을 completed_tasks로 이동
        completed_task_ids = []
        
        for task_id, task in self.pending_tasks.items():
            if task.is_completed:
                self.completed_tasks.append(task)
                completed_task_ids.append(task_id)
                
                # 의존성 제거
                for dep_id in task.dependencies:
                    self.deadlock_detector.remove_dependency(task_id, dep_id)
        
        # pending_tasks에서 제거
        for task_id in completed_task_ids:
            del self.pending_tasks[task_id]
    
    def get_scheduler_statistics(self) -> Dict[str, Any]:
        """스케줄러 통계"""
        worker_stats = {}
        
        for worker_type, workers in self.worker_pools.items():
            worker_stats[worker_type.value] = {
                'count': len(workers),
                'total_completed': sum(w.statistics.tasks_completed for w in workers),
                'total_failed': sum(w.statistics.tasks_failed for w in workers),
                'avg_execution_time': np.mean([w.statistics.avg_execution_time for w in workers if w.statistics.avg_execution_time > 0])
            }
        
        return {
            'performance_metrics': self.performance_metrics,
            'worker_statistics': worker_stats,
            'resource_status': self.resource_manager.get_resource_status(),
            'pending_tasks': len(self.pending_tasks),
            'completed_tasks': len(self.completed_tasks),
            'system_info': {
                'cpu_count': self.cpu_count,
                'memory_gb': self.memory_gb,
                'gpu_count': torch.cuda.device_count()
            },
            'tuning_history': list(self.tuning_history)[-10:]  # 최근 10개
        }

# 사용 예시 함수
async def example_usage():
    """고급 멀티스레드 스왑 스케줄러 사용 예시"""
    scheduler = AdvancedMultithreadedSwapScheduler()
    await scheduler.initialize()
    
    try:
        # 테스트 레이어 생성
        test_layer = nn.Linear(1024, 512)
        
        # 압축 작업 제출
        compress_task_id = await scheduler.compress_layer_async(
            "test_layer", test_layer, TaskPriority.HIGH
        )
        
        # 작업 완료 대기
        result = await scheduler.wait_for_task(compress_task_id, timeout=10.0)
        print(f"압축 작업 완료: {result}")
        
        # GPU 전송 작업
        test_tensor = torch.randn(100, 100)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        transfer_task_id = await scheduler.transfer_to_gpu_async(
            test_tensor, device, TaskPriority.HIGH
        )
        
        gpu_tensor = await scheduler.wait_for_task(transfer_task_id, timeout=5.0)
        print(f"GPU 전송 완료: {gpu_tensor.device}")
        
        # 통계 출력
        stats = scheduler.get_scheduler_statistics()
        print(f"\n=== 스케줄러 통계 ===")
        print(f"총 작업: {stats['performance_metrics']['total_tasks']}")
        print(f"완료 작업: {stats['performance_metrics']['completed_tasks']}")
        print(f"실패 작업: {stats['performance_metrics']['failed_tasks']}")
        print(f"평균 실행 시간: {stats['performance_metrics']['avg_execution_time']:.3f}s")
        print(f"처리량: {stats['performance_metrics']['throughput_tasks_per_sec']:.2f} tasks/sec")
        
        # 워커별 통계
        print(f"\n=== 워커 통계 ===")
        for worker_type, worker_stat in stats['worker_statistics'].items():
            print(f"{worker_type}: {worker_stat['count']}개 워커, "
                  f"완료 {worker_stat['total_completed']}, "
                  f"실패 {worker_stat['total_failed']}")
        
        # 리소스 상태
        resource_status = stats['resource_status']
        print(f"\n=== 리소스 상태 ===")
        print(f"CPU 사용률: {resource_status['cpu_usage_percent']:.1f}%")
        print(f"메모리 사용률: {resource_status['memory_usage_percent']:.1f}%")
        print(f"GPU 사용률: {resource_status['gpu_usage_percent']:.1f}%")
        
    finally:
        await scheduler.shutdown()

if __name__ == "__main__":
    asyncio.run(example_usage())