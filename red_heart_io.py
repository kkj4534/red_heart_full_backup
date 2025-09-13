#!/usr/bin/env python3
"""
Red Heart Core I/O 래퍼

UnifiedModel과 다른 모듈들을 IOPipeline과 통합
비동기 처리를 통해 모듈 간 독립성 확보
"""

import asyncio
import logging
import time
import torch
import traceback
from typing import Dict, Any, Optional, List, Union
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json

# 로컬 임포트
from io_pipeline import IOPipeline, PipelineStage
from data_structures import (
    TaskMessage, ResultMessage, EmotionData, BenthamResult, SURDMetrics,
    Priority, TaskType, ModuleType
)
from unified_memory_manager import UnifiedMemoryManager, MemoryMode, SwapPriority

logger = logging.getLogger(__name__)


class RedHeartCore:
    """
    Red Heart 코어 I/O 래퍼
    
    UnifiedModel과 다른 분석 모듈들을 IOPipeline과 통합하여
    비동기적으로 처리하는 중앙 처리 시스템
    """
    
    def __init__(self, 
                 io_pipeline: IOPipeline,
                 unified_model: Optional[Any] = None,
                 config: Optional[Dict] = None):
        """
        Args:
            io_pipeline: I/O 파이프라인
            unified_model: UnifiedModel 인스턴스
            config: 설정 딕셔너리
        """
        self.pipeline = io_pipeline
        self.unified_model = unified_model
        self.config = config or {}
        
        # 처리 루프 태스크
        self.processing_loop_task = None
        self.running = False
        
        # 모듈 레지스트리
        self.modules = {}
        
        # 스레드 풀 (CPU 바운드 작업용)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 메모리 관리자
        self.memory_manager = UnifiedMemoryManager(MemoryMode.UNIFIED)
        
        # 통계
        self.stats = {
            'processed': 0,
            'errors': 0,
            'total_latency': 0.0,
            'emotion_tasks': 0,
            'bentham_tasks': 0,
            'surd_tasks': 0
        }
        
        # 핸들러 등록
        self._register_handlers()
        
        logger.info("RedHeartCore 초기화 완료")
    
    def _register_handlers(self):
        """파이프라인 핸들러 등록"""
        # LLM 초기 분석
        self.pipeline.register_handler(
            PipelineStage.LLM_INITIAL,
            self._handle_llm_initial
        )
        
        # Red Heart 처리 (UnifiedModel)
        self.pipeline.register_handler(
            PipelineStage.RED_HEART,
            self._handle_red_heart
        )
        
        # Circuit 처리
        self.pipeline.register_handler(
            PipelineStage.CIRCUIT,
            self._handle_circuit
        )
        
        # LLM 최종 요약
        self.pipeline.register_handler(
            PipelineStage.LLM_FINAL,
            self._handle_llm_final
        )
        
        logger.info("파이프라인 핸들러 등록 완료")
    
    def register_module(self, name: str, module: Any, priority: SwapPriority = SwapPriority.MEDIUM):
        """모듈 등록
        
        Args:
            name: 모듈 이름
            module: 모듈 인스턴스
            priority: 메모리 스왑 우선순위
        """
        self.modules[name] = module
        
        # 메모리 관리자에 등록
        if hasattr(module, 'parameters'):
            self.memory_manager.register_model(name, module, priority)
        
        logger.info(f"모듈 등록: {name}")
    
    async def _handle_llm_initial(self, data: Dict, metadata: Dict) -> Dict:
        """LLM 초기 분석 처리
        
        Args:
            data: 입력 데이터
            metadata: 메타데이터
            
        Returns:
            분석 결과
        """
        try:
            text = data.get('text', '')
            
            # LLM 플러그인 시스템 사용
            if 'llm_plugin_manager' in self.modules:
                llm_manager = self.modules['llm_plugin_manager']
                
                # 초기 분석 요청
                analysis = await llm_manager.analyze_initial(text)
                
                return {
                    'llm_initial': analysis,
                    'text': text
                }
            else:
                # 폴백: 기본 분석
                return {
                    'llm_initial': {
                        'emotions': ['neutral'],
                        'themes': ['general'],
                        'intent': 'unknown'
                    },
                    'text': text
                }
                
        except Exception as e:
            logger.error(f"LLM 초기 분석 오류: {e}")
            return {'error': str(e)}
    
    async def _handle_red_heart(self, data: Dict, metadata: Dict) -> Dict:
        """Red Heart (UnifiedModel) 처리
        
        Args:
            data: 입력 데이터
            metadata: 메타데이터
            
        Returns:
            UnifiedModel 처리 결과
        """
        try:
            if not self.unified_model:
                logger.warning("UnifiedModel이 등록되지 않음")
                return {'error': 'UnifiedModel not registered'}
            
            text = data.get('text', '')
            llm_initial = data.get('llm_initial', {})
            
            # GPU 메모리 요청
            await self.memory_manager.request_gpu('unified_model', timeout=30.0)
            
            # 비동기 처리를 위한 래퍼
            result = await self._run_unified_model_async(text, llm_initial)
            
            # 통계 업데이트
            self.stats['processed'] += 1
            if 'emotion' in result:
                self.stats['emotion_tasks'] += 1
            if 'bentham' in result:
                self.stats['bentham_tasks'] += 1
            
            return {
                'unified_result': result,
                **data  # 이전 데이터 전달
            }
            
        except Exception as e:
            logger.error(f"Red Heart 처리 오류: {e}")
            self.stats['errors'] += 1
            return {'error': str(e)}
    
    async def _run_unified_model_async(self, text: str, llm_context: Dict) -> Dict:
        """UnifiedModel 비동기 실행
        
        CPU 바운드 작업을 별도 스레드에서 실행
        
        Args:
            text: 입력 텍스트
            llm_context: LLM 컨텍스트
            
        Returns:
            처리 결과
        """
        loop = asyncio.get_event_loop()
        
        # UnifiedModel forward를 별도 스레드에서 실행
        def run_model():
            try:
                # 실제 UnifiedModel 호출
                if hasattr(self.unified_model, 'process'):
                    return self.unified_model.process(text, llm_context)
                elif hasattr(self.unified_model, 'forward'):
                    # 텐서 변환 필요시
                    return self.unified_model.forward(text)
                else:
                    # 폴백
                    return {
                        'emotion': {'primary': 'neutral', 'confidence': 0.5},
                        'bentham': {'final_score': 0.5}
                    }
            except Exception as e:
                logger.error(f"모델 실행 오류: {e}")
                return {'error': str(e)}
        
        # 스레드 풀에서 실행
        result = await loop.run_in_executor(self.executor, run_model)
        
        return result
    
    async def _handle_circuit(self, data: Dict, metadata: Dict) -> Dict:
        """Circuit 처리 (감정-윤리-후회 통합)
        
        Args:
            data: 입력 데이터  
            metadata: 메타데이터
            
        Returns:
            Circuit 처리 결과
        """
        try:
            unified_result = data.get('unified_result', {})
            
            # EmotionEthicsRegretCircuit 처리
            if 'emotion_circuit' in self.modules:
                circuit = self.modules['emotion_circuit']
                
                # Circuit 컨텍스트 생성
                circuit_context = {
                    'emotion': unified_result.get('emotion', {}),
                    'bentham': unified_result.get('bentham', {}),
                    'text': data.get('text', '')
                }
                
                # Circuit 처리
                circuit_result = await self._run_in_executor(
                    circuit.process,
                    circuit_context
                )
                
                return {
                    'circuit_result': circuit_result,
                    **data
                }
            else:
                # Circuit 없으면 패스스루
                return data
                
        except Exception as e:
            logger.error(f"Circuit 처리 오류: {e}")
            return {'error': str(e)}
    
    async def _handle_llm_final(self, data: Dict, metadata: Dict) -> Dict:
        """LLM 최종 요약 처리
        
        Args:
            data: 입력 데이터
            metadata: 메타데이터
            
        Returns:
            최종 요약 결과
        """
        try:
            # 모든 결과 수집
            results = {
                'llm_initial': data.get('llm_initial', {}),
                'unified': data.get('unified_result', {}),
                'circuit': data.get('circuit_result', {})
            }
            
            # LLM 최종 요약
            if 'llm_plugin_manager' in self.modules:
                llm_manager = self.modules['llm_plugin_manager']
                
                summary = await llm_manager.summarize_final(results)
                
                return {
                    'final_summary': summary,
                    'all_results': results
                }
            else:
                # 폴백: 간단한 요약
                return {
                    'final_summary': "처리 완료",
                    'all_results': results
                }
                
        except Exception as e:
            logger.error(f"LLM 최종 요약 오류: {e}")
            return {'error': str(e)}
    
    async def _run_in_executor(self, func, *args, **kwargs):
        """스레드 풀에서 함수 실행
        
        CPU 바운드 작업을 별도 스레드에서 실행
        
        Args:
            func: 실행할 함수
            *args: 위치 인자
            **kwargs: 키워드 인자
            
        Returns:
            함수 실행 결과
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            func,
            *args,
            **kwargs
        )
    
    async def process_task(self, task: TaskMessage) -> ResultMessage:
        """단일 작업 처리
        
        Args:
            task: 작업 메시지
            
        Returns:
            결과 메시지
        """
        start_time = time.time()
        
        try:
            # 파이프라인에 작업 제출
            task_id = await self.pipeline.submit_task(task)
            
            # 결과 대기 (타임아웃 적용)
            timeout = task.timeout or 60.0
            result = await self.pipeline.get_result(task_id, timeout=timeout)
            
            if result:
                # 성공
                processing_time = time.time() - start_time
                
                return ResultMessage(
                    module=ModuleType.UNIFIED_MODEL,
                    task_type=task.task_type,
                    data=result.data if hasattr(result, 'data') else result,
                    success=True,
                    processing_time=processing_time,
                    session_id=task.session_id,
                    task_id=task_id,
                    metadata=result.metadata if hasattr(result, 'metadata') else {}
                )
            else:
                # 타임아웃
                return ResultMessage(
                    module=ModuleType.UNIFIED_MODEL,
                    task_type=task.task_type,
                    data={},
                    success=False,
                    error='Processing timeout',
                    session_id=task.session_id,
                    task_id=task_id
                )
                
        except Exception as e:
            logger.error(f"작업 처리 오류: {e}\n{traceback.format_exc()}")
            
            return ResultMessage(
                module=ModuleType.UNIFIED_MODEL,
                task_type=task.task_type,
                data={},
                success=False,
                error=str(e),
                session_id=task.session_id,
                task_id=task.session_id or 'unknown'
            )
    
    async def process_batch(self, tasks: List[TaskMessage]) -> List[ResultMessage]:
        """배치 작업 처리
        
        Args:
            tasks: 작업 메시지 리스트
            
        Returns:
            결과 메시지 리스트
        """
        # 병렬 처리
        results = await asyncio.gather(
            *[self.process_task(task) for task in tasks],
            return_exceptions=True
        )
        
        # 예외 처리
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # 에러 결과 생성
                processed_results.append(
                    ResultMessage(
                        module=ModuleType.UNIFIED_MODEL,
                        task_type=tasks[i].task_type,
                        data={},
                        success=False,
                        error=str(result),
                        session_id=tasks[i].session_id,
                        task_id=tasks[i].session_id or f"batch_{i}"
                    )
                )
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def start_processing_loop(self):
        """비동기 처리 루프 시작"""
        if self.running:
            logger.warning("처리 루프가 이미 실행 중")
            return
        
        self.running = True
        self.processing_loop_task = asyncio.create_task(self._processing_loop())
        
        # 파이프라인 시작
        await self.pipeline.start()
        
        logger.info("RedHeartCore 처리 루프 시작")
    
    async def _processing_loop(self):
        """내부 처리 루프
        
        입력 큐를 모니터링하고 작업을 처리
        """
        while self.running:
            try:
                # 입력 큐에서 작업 가져오기 (논블로킹)
                await asyncio.sleep(0.1)  # 짧은 대기
                
                # 여기서는 외부에서 process_task/process_batch를 호출하는 방식 사용
                # 실제 큐 모니터링은 IOPipeline이 담당
                
            except Exception as e:
                logger.error(f"처리 루프 오류: {e}")
                await asyncio.sleep(1.0)  # 에러 시 대기
    
    async def stop_processing_loop(self):
        """처리 루프 중지"""
        self.running = False
        
        # 처리 루프 태스크 취소
        if self.processing_loop_task:
            self.processing_loop_task.cancel()
            try:
                await self.processing_loop_task
            except asyncio.CancelledError:
                pass
        
        # 파이프라인 중지
        await self.pipeline.stop()
        
        # 스레드 풀 종료
        self.executor.shutdown(wait=True)
        
        logger.info("RedHeartCore 처리 루프 중지")
    
    def get_stats(self) -> Dict:
        """통계 조회
        
        Returns:
            처리 통계
        """
        pipeline_stats = self.pipeline.get_stats()
        
        return {
            **self.stats,
            'pipeline': pipeline_stats,
            'memory': self.memory_manager.get_stats(),
            'modules': list(self.modules.keys())
        }
    
    async def __aenter__(self):
        """async with 지원"""
        await self.start_processing_loop()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """async with 종료"""
        await self.stop_processing_loop()


# 유틸리티 함수
async def create_red_heart_core(config: Dict) -> RedHeartCore:
    """RedHeartCore 생성 헬퍼
    
    Args:
        config: 설정 딕셔너리
        
    Returns:
        초기화된 RedHeartCore 인스턴스
    """
    # IOPipeline 생성
    pipeline = IOPipeline(max_queue_size=config.get('queue_size', 100))
    
    # RedHeartCore 생성
    core = RedHeartCore(pipeline, config=config)
    
    # UnifiedModel 로드 (있다면)
    if 'unified_model_path' in config:
        # TODO: UnifiedModel 로드 로직
        pass
    
    # 기타 모듈 등록
    # TODO: 필요한 모듈들 등록
    
    return core


# 테스트 코드
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    async def test_red_heart_core():
        """RedHeartCore 테스트"""
        
        # 설정
        config = {
            'queue_size': 50,
            'memory_mode': 'unified'
        }
        
        # Core 생성
        core = await create_red_heart_core(config)
        
        async with core:
            # 테스트 작업 생성
            task1 = TaskMessage(
                module=ModuleType.UNIFIED_MODEL,
                task_type=TaskType.EMOTION,
                data={'text': '오늘은 정말 행복한 날이야!'},
                priority=Priority.HIGH.value,
                session_id='test_001'
            )
            
            task2 = TaskMessage(
                module=ModuleType.UNIFIED_MODEL,
                task_type=TaskType.BENTHAM,
                data={'text': '도덕적 딜레마 상황입니다.'},
                priority=Priority.NORMAL.value,
                session_id='test_002'
            )
            
            # 단일 작업 처리
            result1 = await core.process_task(task1)
            print(f"Result 1: {result1.to_dict()}")
            
            # 배치 처리
            results = await core.process_batch([task1, task2])
            for i, result in enumerate(results):
                print(f"Batch Result {i+1}: {result.to_dict()}")
            
            # 통계 출력
            stats = core.get_stats()
            print(f"\n통계: {json.dumps(stats, indent=2, ensure_ascii=False)}")
    
    # 테스트 실행
    asyncio.run(test_red_heart_core())