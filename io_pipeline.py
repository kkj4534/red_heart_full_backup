#!/usr/bin/env python3
"""
Red Heart I/O Pipeline System

DSM 철학 적용: "비동기 기반 동기 스왑 처리"
- 비동기 큐를 사용하지만 스텝별 동기화 보장
- CPU/GPU 비대칭 처리 방지
- 모듈 간 독립성 확보
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import torch
import heapq
from data_structures import TaskMessage, ResultMessage, Priority, TaskType, ModuleType

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """파이프라인 처리 단계"""
    INIT = "init"
    LLM_INITIAL = "llm_initial"
    GPU_SWAP_1 = "gpu_swap_1"
    RED_HEART = "red_heart"
    GPU_SWAP_2 = "gpu_swap_2"
    CIRCUIT = "circuit"
    GPU_SWAP_3 = "gpu_swap_3"
    LLM_FINAL = "llm_final"
    COMPLETE = "complete"


class StepBarrier:
    """스텝별 동기화 장벽 (DSM 철학 구현)"""
    
    def __init__(self, name: str):
        self.name = name
        self.event = asyncio.Event()
        self.waiting_count = 0
        self.completed = False
        
    async def wait(self):
        """장벽에서 대기"""
        self.waiting_count += 1
        logger.debug(f"[{self.name}] 대기 중... (대기자: {self.waiting_count})")
        await self.event.wait()
        self.waiting_count -= 1
        
    def release(self):
        """장벽 해제"""
        if not self.completed:
            self.completed = True
            self.event.set()
            logger.debug(f"[{self.name}] 장벽 해제! (대기자: {self.waiting_count}명 해제)")
            
    def reset(self):
        """장벽 재설정"""
        self.completed = False
        self.event.clear()
        self.waiting_count = 0


class IOPipeline:
    """
    Red Heart I/O 파이프라인
    
    특징:
    - 비동기 큐 기반 모듈 간 통신
    - DSM 철학: 스텝별 동기화로 순차 처리 보장
    - CPU/GPU 비대칭 방지
    - 모듈 독립성 확보
    """
    
    def __init__(self, max_queue_size: int = 100, max_retries: int = 3):
        """
        Args:
            max_queue_size: 각 큐의 최대 크기
            max_retries: 최대 재시도 횟수
        """
        # 입출력 큐
        self.input_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self.output_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        
        # 우선순위 큐 (heapq 사용)
        self.priority_queue: List[tuple] = []  # (priority, timestamp, task)
        self.priority_lock = asyncio.Lock()
        
        # 스테이지별 큐 (모듈 간 전달용)
        self.stage_queues: Dict[PipelineStage, asyncio.Queue] = {
            stage: asyncio.Queue(maxsize=max_queue_size)
            for stage in PipelineStage
        }
        
        # 스텝 장벽 (DSM 철학 구현)
        self.step_barriers: Dict[str, StepBarrier] = {}
        
        # CPU/GPU 동기화 락
        self.cpu_gpu_sync = asyncio.Lock()
        
        # 모듈 핸들러 등록
        self.module_handlers: Dict[PipelineStage, Callable] = {}
        
        # 통계
        self.stats = defaultdict(lambda: {
            'processed': 0,
            'errors': 0,
            'total_time': 0.0,
            'avg_time': 0.0
        })
        
        # 실행 상태
        self.running = False
        self.workers: List[asyncio.Task] = []
        
        # 재시도 설정
        self.max_retries = max_retries
        self.retry_delays = [1.0, 2.0, 5.0]  # 재시도 지연 시간 (초)
        
        logger.info(f"IOPipeline 초기화 완료 (최대 재시도: {max_retries})")
        
    def register_handler(self, stage: PipelineStage, handler: Callable):
        """모듈 핸들러 등록
        
        Args:
            stage: 파이프라인 스테이지
            handler: 비동기 핸들러 함수
        """
        self.module_handlers[stage] = handler
        logger.info(f"핸들러 등록: {stage.value} -> {handler.__name__}")
        
    async def submit_task(self, task: TaskMessage) -> str:
        """작업 제출 및 결과 대기
        
        Args:
            task: TaskMessage 객체
            
        Returns:
            작업 ID
        """
        task_id = task.session_id or f"task_{time.time():.6f}"
        
        # 우선순위에 따라 큐 선택
        if task.priority <= Priority.HIGH.value:
            # 높은 우선순위는 우선순위 큐로
            async with self.priority_lock:
                heapq.heappush(self.priority_queue, 
                             (task.priority, task.timestamp, task))
            logger.info(f"우선순위 작업 제출: {task_id} (우선순위: {task.priority})")
        else:
            # 일반 우선순위는 일반 큐로
            await self.input_queue.put(task)
            logger.info(f"일반 작업 제출: {task_id}")
        
        return task_id
        
    async def get_result(self, task_id: str, timeout: float = 60.0) -> Optional[ResultMessage]:
        """결과 조회
        
        Args:
            task_id: 작업 ID
            timeout: 대기 시간
            
        Returns:
            완료된 ResultMessage 또는 None
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # 출력 큐 확인
            try:
                result = await asyncio.wait_for(
                    self.output_queue.get(),
                    timeout=1.0
                )
                
                if isinstance(result, ResultMessage) and result.task_id == task_id:
                    return result
                else:
                    # 다른 작업이면 다시 큐에 넣기
                    await self.output_queue.put(result)
                    
            except asyncio.TimeoutError:
                continue
                
        logger.warning(f"작업 {task_id} 타임아웃")
        return None
        
    async def wait_for_step(self, step_name: str):
        """스텝 완료 대기 (DSM 철학)
        
        모든 관련 작업이 완료될 때까지 대기
        """
        if step_name not in self.step_barriers:
            self.step_barriers[step_name] = StepBarrier(step_name)
            
        barrier = self.step_barriers[step_name]
        await barrier.wait()
        
    def complete_step(self, step_name: str):
        """스텝 완료 신호"""
        if step_name in self.step_barriers:
            self.step_barriers[step_name].release()
            
    async def _ensure_gpu_sync(self):
        """GPU/CPU 동기화 보장"""
        async with self.cpu_gpu_sync:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                logger.debug("GPU 동기화 완료")
                
    async def _process_with_retry(self, task: TaskMessage, handler: Callable) -> ResultMessage:
        """재시도 로직이 포함된 태스크 처리
        
        Args:
            task: 처리할 TaskMessage
            handler: 핸들러 함수
            
        Returns:
            ResultMessage
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # 핸들러 실행
                result = await handler(task.data, task.metadata)
                
                # 성공적인 결과 반환
                return ResultMessage(
                    module=task.module,
                    task_type=task.task_type,
                    data=result,
                    success=True,
                    processing_time=time.time() - task.timestamp,
                    session_id=task.session_id,
                    task_id=task.session_id
                )
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"처리 실패 (시도 {attempt + 1}/{self.max_retries}): {e}")
                
                # 재시도 전 대기
                if attempt < self.max_retries - 1:
                    delay = self.retry_delays[min(attempt, len(self.retry_delays) - 1)]
                    await asyncio.sleep(delay)
        
        # 모든 재시도 실패
        return ResultMessage(
            module=task.module,
            task_type=task.task_type,
            data={},
            success=False,
            error=f"Max retries exceeded: {last_error}",
            processing_time=time.time() - task.timestamp,
            session_id=task.session_id,
            task_id=task.session_id
        )
    
    async def _process_stage(self, stage: PipelineStage):
        """스테이지 처리 워커
        
        Args:
            stage: 처리할 스테이지
        """
        queue = self.stage_queues[stage]
        handler = self.module_handlers.get(stage)
        
        while self.running:
            try:
                # 우선순위 큐 먼저 확인
                task = None
                async with self.priority_lock:
                    if self.priority_queue:
                        # 우선순위 큐에서 가져오기
                        _, _, task = heapq.heappop(self.priority_queue)
                
                # 우선순위 큐가 비어있으면 일반 큐에서 가져오기
                if task is None:
                    task = await asyncio.wait_for(queue.get(), timeout=1.0)
                
                if not handler:
                    # 핸들러가 없으면 에러 결과 생성
                    result = ResultMessage(
                        module=ModuleType.UNIFIED_MODEL,  # 기본값
                        task_type=TaskType.UNIFIED,  # 기본값
                        data={},
                        success=False,
                        error=f"No handler for stage {stage.value}",
                        session_id=task.session_id if isinstance(task, TaskMessage) else None
                    )
                    await self.output_queue.put(result)
                    continue
                    
                # GPU 스왑 단계면 동기화
                if "GPU_SWAP" in stage.value:
                    await self._ensure_gpu_sync()
                    
                # 재시도 로직이 포함된 핸들러 실행
                if isinstance(task, TaskMessage):
                    result = await self._process_with_retry(task, handler)
                    
                    # 통계 업데이트
                    if result.success:
                        self.stats[stage]['processed'] += 1
                        self.stats[stage]['total_time'] += result.processing_time or 0
                        self.stats[stage]['avg_time'] = (
                            self.stats[stage]['total_time'] / 
                            self.stats[stage]['processed']
                        )
                    else:
                        self.stats[stage]['errors'] += 1
                    
                    # 결과를 출력 큐로
                    await self.output_queue.put(result)
                else:
                    # 레거시 코드 호환성 (삭제 예정)
                    logger.warning(f"레거시 task 형식 감지: {type(task)}")
                    continue
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"워커 오류 ({stage.value}): {e}")
                
    def _get_next_stage(self, current: PipelineStage) -> Optional[PipelineStage]:
        """다음 스테이지 결정"""
        stage_order = [
            PipelineStage.INIT,
            PipelineStage.LLM_INITIAL,
            PipelineStage.GPU_SWAP_1,
            PipelineStage.RED_HEART,
            PipelineStage.GPU_SWAP_2,
            PipelineStage.CIRCUIT,
            PipelineStage.GPU_SWAP_3,
            PipelineStage.LLM_FINAL,
            PipelineStage.COMPLETE
        ]
        
        try:
            current_idx = stage_order.index(current)
            if current_idx < len(stage_order) - 1:
                return stage_order[current_idx + 1]
        except ValueError:
            pass
            
        return None
        
    async def start(self):
        """파이프라인 시작"""
        if self.running:
            logger.warning("파이프라인이 이미 실행 중")
            return
            
        self.running = True
        
        # 입력 큐 처리 워커
        async def input_worker():
            while self.running:
                try:
                    task = await asyncio.wait_for(
                        self.input_queue.get(),
                        timeout=1.0
                    )
                    # 첫 스테이지로 전달
                    task.stage = PipelineStage.LLM_INITIAL
                    await self.stage_queues[PipelineStage.LLM_INITIAL].put(task)
                except asyncio.TimeoutError:
                    continue
                    
        self.workers.append(asyncio.create_task(input_worker()))
        
        # 각 스테이지별 워커 시작
        for stage in PipelineStage:
            if stage not in [PipelineStage.INIT, PipelineStage.COMPLETE]:
                worker = asyncio.create_task(self._process_stage(stage))
                self.workers.append(worker)
                
        logger.info(f"파이프라인 시작 (워커: {len(self.workers)}개)")
        
    async def stop(self):
        """파이프라인 중지"""
        self.running = False
        
        # 모든 워커 종료 대기
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)
            self.workers.clear()
            
        logger.info("파이프라인 중지")
        
    def get_stats(self) -> Dict:
        """통계 조회"""
        return dict(self.stats)
        
    async def __aenter__(self):
        """async with 지원"""
        await self.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """async with 종료"""
        await self.stop()


# 테스트용 예제 핸들러
async def example_llm_handler(data: Dict, metadata: Dict) -> Dict:
    """LLM 핸들러 예제"""
    logger.info(f"LLM 처리: {data.get('text', '')[:50]}...")
    await asyncio.sleep(0.1)  # 처리 시뮬레이션
    return {'llm_result': 'analyzed'}


async def example_red_heart_handler(data: Dict, metadata: Dict) -> Dict:
    """Red Heart 핸들러 예제"""
    logger.info(f"Red Heart 처리: {data.get('text', '')[:50]}...")
    await asyncio.sleep(0.2)  # 처리 시뮬레이션
    return {'red_heart_result': 'processed'}


# 테스트 코드
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    async def test_pipeline():
        """파이프라인 테스트"""
        pipeline = IOPipeline()
        
        # 핸들러 등록
        pipeline.register_handler(PipelineStage.LLM_INITIAL, example_llm_handler)
        pipeline.register_handler(PipelineStage.RED_HEART, example_red_heart_handler)
        
        async with pipeline:
            # 작업 제출
            task_id = await pipeline.submit_task("테스트 텍스트입니다.")
            
            # 결과 대기
            result = await pipeline.get_result(task_id)
            if result:
                print(f"결과: {result.data}")
                print(f"소요 시간: {result.completed_at - result.created_at:.2f}초")
            
            # 통계 출력
            stats = pipeline.get_stats()
            for stage, stat in stats.items():
                if stat['processed'] > 0:
                    print(f"{stage.value}: {stat['processed']}건 처리, 평균 {stat['avg_time']:.3f}초")
    
    asyncio.run(test_pipeline())