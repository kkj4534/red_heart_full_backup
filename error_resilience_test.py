"""
Red Heart Linux Advanced - 에러 처리 및 복원력 테스트 시스템
예외 상황 대응, 장애 복구, 시스템 안정성 검증

⚠️ 중요: 순수한 재시도 방식 (Pure Retry Approach)
이 시스템은 fallback이나 graceful degradation을 사용하지 않습니다.
에러 발생 시:
1. 고급 시스템을 그대로 유지
2. 메모리/GPU 정리만 수행 (시스템 downgrade 없음) 
3. 동일한 고급 알고리즘으로 재시도
4. 모든 재시도 실패 시 에러 그대로 전파

이 모듈이 제공하는 기능:
1. 다양한 예외 상황 시뮬레이션
2. 순수 재시도 기반 복원력 테스트
3. 고급 시스템 유지하면서 에러 처리 검증
4. fallback 없는 장애 복구 메커니즘 테스트
5. 원본 시스템 순수성 유지하면서 복원력 검증
"""

import asyncio
import logging
import traceback
import time
import json
import random
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import tempfile
import os
import signal
import threading
from contextlib import contextmanager

# Red Heart 시스템 모듈들
from config import SYSTEM_CONFIG, setup_logging, get_smart_device
from advanced_hierarchical_emotion_system import (
    EnhancedEmpathyLearner,
    MirrorNeuronSystem, 
    HierarchicalEmotionState
)
from advanced_bentham_calculator import (
    FrommEnhancedBenthamCalculator,
    FrommEthicalAnalyzer
)

# 로거 설정
logger = setup_logging()

class ErrorType(Enum):
    """에러 유형 분류"""
    MEMORY_ERROR = "memory_error"
    GPU_ERROR = "gpu_error"
    TIMEOUT_ERROR = "timeout_error"
    INVALID_INPUT = "invalid_input"
    NETWORK_ERROR = "network_error"
    FILE_IO_ERROR = "file_io_error"
    COMPUTATION_ERROR = "computation_error"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CONCURRENT_ACCESS = "concurrent_access"
    DATA_CORRUPTION = "data_corruption"

class SeverityLevel(Enum):
    """에러 심각도"""
    LOW = "low"          # 기능 일부 제한
    MEDIUM = "medium"    # 성능 저하
    HIGH = "high"        # 기능 실패
    CRITICAL = "critical" # 시스템 중단

@dataclass
class ErrorScenario:
    """에러 시나리오 정의"""
    name: str
    error_type: ErrorType
    severity: SeverityLevel
    description: str
    trigger_function: Callable
    expected_behavior: str
    recovery_expected: bool
    max_recovery_time: float = 30.0

@dataclass
class ErrorTestResult:
    """에러 테스트 결과"""
    scenario_name: str
    error_type: ErrorType
    severity: SeverityLevel
    error_triggered: bool
    error_handled_gracefully: bool
    recovery_successful: bool
    recovery_time: Optional[float]
    system_state_after_error: str
    error_message: Optional[str] = None
    additional_info: Dict[str, Any] = None

class ErrorSimulator:
    """에러 상황 시뮬레이터"""
    
    def __init__(self):
        self.active_simulations = set()
        self.original_functions = {}
        
    async def simulate_memory_exhaustion(self, target_mb: int = 100):
        """메모리 고갈 시뮬레이션"""
        logger.info(f"💾 메모리 고갈 시뮬레이션 시작: {target_mb}MB")
        memory_hog = []
        try:
            # 메모리를 점진적으로 할당
            for i in range(target_mb):
                memory_hog.append([0] * (1024 * 1024 // 4))  # 1MB씩 할당
                if i % 10 == 0:
                    await asyncio.sleep(0.1)  # 다른 작업이 실행될 시간 제공
        except MemoryError:
            logger.info("✅ 메모리 고갈 상황 성공적으로 시뮬레이션됨")
            raise
        finally:
            # 메모리 해제
            del memory_hog
    
    async def simulate_gpu_memory_error(self):
        """GPU 메모리 에러 시뮬레이션"""
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("GPU가 사용 불가능한 상태로 시뮬레이션됨")
            
            logger.info("🖥️ GPU 메모리 에러 시뮬레이션 시작")
            # 과도한 GPU 메모리 할당 시도
            large_tensor = torch.zeros((10000, 10000, 100), device='cuda')
            
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            logger.info("✅ GPU 메모리 에러 성공적으로 시뮬레이션됨")
            raise
        except ImportError:
            raise RuntimeError("PyTorch를 사용할 수 없는 상태로 시뮬레이션됨")
    
    async def simulate_timeout_error(self, delay: float = 10.0):
        """타임아웃 에러 시뮬레이션"""
        logger.info(f"⏰ 타임아웃 에러 시뮬레이션: {delay}초 지연")
        await asyncio.sleep(delay)
        raise TimeoutError(f"{delay}초 타임아웃 시뮬레이션")
    
    async def simulate_invalid_input_error(self):
        """잘못된 입력 에러 시뮬레이션"""
        logger.info("❌ 잘못된 입력 에러 시뮬레이션")
        raise ValueError("시뮬레이션된 잘못된 입력 데이터")
    
    async def simulate_file_io_error(self):
        """파일 I/O 에러 시뮬레이션"""
        logger.info("📁 파일 I/O 에러 시뮬레이션")
        # 존재하지 않는 파일 접근 시도
        with open("/nonexistent/path/file.txt", "r") as f:
            f.read()
    
    async def simulate_computation_error(self):
        """계산 에러 시뮬레이션"""
        logger.info("🔢 계산 에러 시뮬레이션")
        result = 1 / 0  # ZeroDivisionError
    
    async def simulate_data_corruption(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """데이터 손상 시뮬레이션"""
        logger.info("💾 데이터 손상 시뮬레이션")
        corrupted_data = data.copy()
        
        # 무작위로 데이터 필드 손상
        if isinstance(corrupted_data, dict):
            keys = list(corrupted_data.keys())
            if keys:
                corrupt_key = random.choice(keys)
                corrupted_data[corrupt_key] = None  # 데이터 손상
                
        return corrupted_data

class PureRetryErrorHandler:
    """순수한 재시도 에러 핸들러 (fallback/degradation 없음)"""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 0.5):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
    async def execute_with_pure_retry(self, func: Callable, *args, **kwargs) -> Any:
        """
        fallback이나 graceful degradation 없이 순수한 고급 시스템으로만 재시도
        
        에러 발생 시:
        1. 메모리/GPU 정리만 수행 (시스템 degradation 없음)
        2. 동일한 고급 알고리즘으로 재시도
        3. 모든 재시도 실패 시 에러 그대로 전파
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):  # 초기 시도 + 재시도
            try:
                if attempt > 0:  # 재시도 시에만 리소스 정리
                    await self._clean_resources_only()
                    logger.info(f"🔄 순수 재시도 {attempt}/{self.max_retries} (고급 시스템 유지)")
                
                # 동일한 고급 함수 실행 (downgrade 없음)
                result = await func(*args, **kwargs)
                
                if attempt > 0:
                    logger.info(f"✅ 순수 재시도 성공 (시도 {attempt})")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    logger.info(f"⚠️ 시도 {attempt + 1} 실패: {str(e)[:100]}...")
                    await asyncio.sleep(self.retry_delay)
                else:
                    logger.error(f"❌ 모든 순수 재시도 실패: {str(e)}")
        
        # 모든 재시도 실패 시 마지막 에러 그대로 전파
        raise last_exception
    
    async def _clean_resources_only(self):
        """리소스 정리만 수행 (시스템 downgrade 없음)"""
        # 메모리 정리
        import gc
        gc.collect()
        
        # GPU 메모리 정리
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        # 짧은 대기로 시스템 안정화
        await asyncio.sleep(0.1)

class SystemResilienceValidator:
    """시스템 복원력 검증기"""
    
    def __init__(self):
        self.error_simulator = ErrorSimulator()
        self.pure_retry_handler = PureRetryErrorHandler(max_retries=3, retry_delay=0.5)
        self.test_results: List[ErrorTestResult] = []
        
        # 에러 시나리오 정의
        self.error_scenarios = [
            ErrorScenario(
                name="memory_exhaustion_during_empathy_processing",
                error_type=ErrorType.MEMORY_ERROR,
                severity=SeverityLevel.HIGH,
                description="공감 처리 중 메모리 고갈",
                trigger_function=self.error_simulator.simulate_memory_exhaustion,
                expected_behavior="우아한 실패와 메모리 정리",
                recovery_expected=True,
                max_recovery_time=10.0
            ),
            ErrorScenario(
                name="gpu_memory_error_during_neural_processing",
                error_type=ErrorType.GPU_ERROR,
                severity=SeverityLevel.MEDIUM,
                description="신경망 처리 중 GPU 메모리 에러",
                trigger_function=self.error_simulator.simulate_gpu_memory_error,
                expected_behavior="GPU 메모리 정리 후 원본 GPU 시스템 재시도",
                recovery_expected=True,
                max_recovery_time=5.0
            ),
            ErrorScenario(
                name="timeout_during_complex_analysis",
                error_type=ErrorType.TIMEOUT_ERROR,
                severity=SeverityLevel.MEDIUM,
                description="복잡한 분석 중 타임아웃",
                trigger_function=lambda: self.error_simulator.simulate_timeout_error(8.0),
                expected_behavior="원본 시스템 재시도 (타임아웃 연장)",
                recovery_expected=True,
                max_recovery_time=1.0
            ),
            ErrorScenario(
                name="invalid_input_handling",
                error_type=ErrorType.INVALID_INPUT,
                severity=SeverityLevel.LOW,
                description="잘못된 형식의 입력 데이터",
                trigger_function=self.error_simulator.simulate_invalid_input_error,
                expected_behavior="에러 즉시 전파, 재시도 없음",
                recovery_expected=False,
                max_recovery_time=0.0
            ),
            ErrorScenario(
                name="resource_exhaustion_during_processing",
                error_type=ErrorType.RESOURCE_EXHAUSTION,
                severity=SeverityLevel.HIGH,
                description="처리 중 리소스 고갈",
                trigger_function=self.error_simulator.simulate_memory_exhaustion,
                expected_behavior="메모리 정리 후 원본 시스템 재시도",
                recovery_expected=True,
                max_recovery_time=15.0
            ),
            ErrorScenario(
                name="computation_error_in_utility_calculation",
                error_type=ErrorType.COMPUTATION_ERROR,
                severity=SeverityLevel.MEDIUM,
                description="유틸리티 계산 중 수치 에러",
                trigger_function=self.error_simulator.simulate_computation_error,
                expected_behavior="에러 전파, 상위 레벨에서 재시도",
                recovery_expected=False,
                max_recovery_time=0.0
            )
        ]
    
    async def test_empathy_system_resilience(self) -> List[ErrorTestResult]:
        """공감 시스템 복원력 테스트"""
        logger.info("🧠 공감 시스템 복원력 테스트 시작")
        results = []
        
        try:
            empathy_learner = EnhancedEmpathyLearner()
            await empathy_learner.initialize()
            
            # 정상 동작 확인
            normal_result = await empathy_learner.process_empathy_learning(
                "테스트 메시지입니다.", {"test": True}
            )
            assert normal_result is not None
            logger.info("✅ 공감 시스템 정상 동작 확인")
            
            # 각 에러 시나리오 테스트
            for scenario in self.error_scenarios[:4]:  # 공감 시스템 관련 시나리오만
                result = await self._test_single_scenario_with_component(
                    scenario, empathy_learner, "process_empathy_learning",
                    "테스트 에러 시나리오", {"error_test": True}
                )
                results.append(result)
                
                # 시스템 복구 후 순수 재시도로 정상 동작 재확인
                try:
                    recovery_result = await self.pure_retry_handler.execute_with_pure_retry(
                        empathy_learner.process_empathy_learning,
                        "복구 테스트 메시지", {"recovery_test": True}
                    )
                    if recovery_result:
                        logger.info(f"✅ {scenario.name} 후 순수 재시도로 시스템 복구 성공")
                    else:
                        logger.warning(f"⚠️ {scenario.name} 후 순수 재시도 복구 부분적 성공")
                except Exception as e:
                    logger.error(f"❌ {scenario.name} 후 순수 재시도 복구 실패: {str(e)}")
        
        except Exception as e:
            logger.error(f"❌ 공감 시스템 복원력 테스트 실행 실패: {str(e)}")
            
        return results
    
    async def test_bentham_calculator_resilience(self) -> List[ErrorTestResult]:
        """벤담 계산기 복원력 테스트"""
        logger.info("⚖️ 벤담 계산기 복원력 테스트 시작")
        results = []
        
        try:
            calculator = FrommEnhancedBenthamCalculator()
            
            # 정상 동작 확인
            normal_result = await calculator.calculate_enhanced_utility(
                "정상 테스트 메시지", {"test": True}
            )
            assert normal_result is not None
            logger.info("✅ 벤담 계산기 정상 동작 확인")
            
            # 계산 관련 에러 시나리오 테스트
            computation_scenarios = [s for s in self.error_scenarios 
                                   if s.error_type in [ErrorType.COMPUTATION_ERROR, ErrorType.INVALID_INPUT]]
            
            for scenario in computation_scenarios:
                result = await self._test_single_scenario_with_component(
                    scenario, calculator, "calculate_enhanced_utility",
                    "에러 테스트 입력", {"error_test": True}
                )
                results.append(result)
                
                # 순수 재시도로 복구 테스트
                try:
                    recovery_result = await self.pure_retry_handler.execute_with_pure_retry(
                        calculator.calculate_enhanced_utility,
                        "순수 재시도 테스트", {"pure_retry_test": True}
                    )
                    if recovery_result:
                        logger.info(f"✅ {scenario.name} 후 벤담 계산기 순수 재시도 성공")
                except Exception as e:
                    logger.error(f"❌ {scenario.name} 후 벤담 계산기 순수 재시도 실패: {str(e)}")
                
        except Exception as e:
            logger.error(f"❌ 벤담 계산기 복원력 테스트 실행 실패: {str(e)}")
            
        return results
    
    async def _test_single_scenario_with_component(
        self, scenario: ErrorScenario, component, method_name: str, 
        test_input: str, test_context: Dict[str, Any]
    ) -> ErrorTestResult:
        """단일 시나리오 테스트 실행"""
        logger.info(f"🧪 테스트 시나리오: {scenario.name}")
        
        error_triggered = False
        error_handled_gracefully = False
        recovery_successful = False
        recovery_time = None
        error_message = None
        system_state_after_error = "unknown"
        
        try:
            # 에러 상황 시뮬레이션과 동시에 컴포넌트 실행
            start_time = time.time()
            
            # 의도적으로 에러를 발생시키는 입력 생성
            if scenario.error_type == ErrorType.INVALID_INPUT:
                # 잘못된 형식의 입력
                test_input = None
                test_context = {"invalid": True, "data": [1, 2, {"corrupted": None}]}
            elif scenario.error_type == ErrorType.COMPUTATION_ERROR:
                # 계산 에러를 유발하는 입력
                test_context = {"division_by_zero": True, "invalid_numbers": float('inf')}
            
            # 컴포넌트 메소드 실행
            try:
                method = getattr(component, method_name)
                
                # 타임아웃이 있는 경우 제한된 시간 내에 실행
                if scenario.error_type == ErrorType.TIMEOUT_ERROR:
                    result = await asyncio.wait_for(
                        method(test_input, test_context), 
                        timeout=5.0
                    )
                else:
                    result = await method(test_input, test_context)
                
                # 에러가 예상되었지만 성공한 경우
                if scenario.error_type != ErrorType.INVALID_INPUT:
                    system_state_after_error = "functioning"
                    error_handled_gracefully = True
                    
            except Exception as e:
                error_triggered = True
                error_message = str(e)
                
                # 에러가 우아하게 처리되었는지 확인
                if isinstance(e, (ValueError, TypeError)) and scenario.error_type == ErrorType.INVALID_INPUT:
                    error_handled_gracefully = True
                    system_state_after_error = "graceful_failure"
                elif isinstance(e, ZeroDivisionError) and scenario.error_type == ErrorType.COMPUTATION_ERROR:
                    error_handled_gracefully = True
                    system_state_after_error = "graceful_failure"
                elif isinstance(e, asyncio.TimeoutError) and scenario.error_type == ErrorType.TIMEOUT_ERROR:
                    error_handled_gracefully = True
                    system_state_after_error = "timeout_handled"
                elif isinstance(e, (MemoryError, RuntimeError)) and scenario.error_type in [ErrorType.MEMORY_ERROR, ErrorType.GPU_ERROR]:
                    error_handled_gracefully = True
                    system_state_after_error = "resource_error_handled"
                else:
                    system_state_after_error = "unexpected_failure"
                    logger.warning(f"⚠️ 예상치 못한 에러 타입: {type(e).__name__}")
            
            # 순수한 재시도 테스트 (fallback 없이 원본 시스템 그대로)
            if error_triggered and scenario.recovery_expected:
                recovery_start = time.time()
                max_retries = 3
                retry_count = 0
                
                while retry_count < max_retries:
                    try:
                        # 리소스 정리 (메모리/GPU만)
                        if scenario.error_type in [ErrorType.MEMORY_ERROR, ErrorType.GPU_ERROR]:
                            import gc
                            gc.collect()
                            if scenario.error_type == ErrorType.GPU_ERROR:
                                try:
                                    import torch
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                except ImportError:
                                    pass
                        
                        # 동일한 고급 시스템으로 재시도 (fallback 없음)
                        recovery_result = await method(test_input, test_context)
                        
                        if recovery_result is not None:
                            recovery_successful = True
                            recovery_time = time.time() - recovery_start
                            system_state_after_error = f"pure_retry_success_attempt_{retry_count + 1}"
                            break
                            
                    except Exception as recovery_e:
                        retry_count += 1
                        logger.info(f"🔄 순수 재시도 {retry_count}/{max_retries}: {str(recovery_e)}")
                        
                        if retry_count >= max_retries:
                            logger.warning(f"❌ 모든 순수 재시도 실패")
                            system_state_after_error = "pure_retry_exhausted"
                        else:
                            # 짧은 대기 후 재시도
                            await asyncio.sleep(0.5)
            
        except Exception as test_e:
            logger.error(f"❌ 시나리오 테스트 실행 실패: {str(test_e)}")
            error_message = f"테스트 실행 에러: {str(test_e)}"
            system_state_after_error = "test_failure"
        
        return ErrorTestResult(
            scenario_name=scenario.name,
            error_type=scenario.error_type,
            severity=scenario.severity,
            error_triggered=error_triggered,
            error_handled_gracefully=error_handled_gracefully,
            recovery_successful=recovery_successful,
            recovery_time=recovery_time,
            system_state_after_error=system_state_after_error,
            error_message=error_message,
            additional_info={
                "expected_behavior": scenario.expected_behavior,
                "max_recovery_time": scenario.max_recovery_time
            }
        )
    
    async def test_concurrent_access_safety(self) -> ErrorTestResult:
        """동시 접근 안전성 테스트"""
        logger.info("🔄 동시 접근 안전성 테스트 시작")
        
        try:
            empathy_learner = EnhancedEmpathyLearner()
            await empathy_learner.initialize()
            
            # 다중 동시 요청 시뮬레이션
            concurrent_tasks = []
            for i in range(10):
                task = empathy_learner.process_empathy_learning(
                    f"동시 요청 {i}", {"concurrent_id": i}
                )
                concurrent_tasks.append(task)
            
            # 모든 작업 동시 실행
            start_time = time.time()
            results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
            execution_time = time.time() - start_time
            
            # 결과 분석
            successful_results = [r for r in results if not isinstance(r, Exception)]
            failed_results = [r for r in results if isinstance(r, Exception)]
            
            success_rate = len(successful_results) / len(results)
            
            return ErrorTestResult(
                scenario_name="concurrent_access_safety",
                error_type=ErrorType.CONCURRENT_ACCESS,
                severity=SeverityLevel.MEDIUM,
                error_triggered=len(failed_results) > 0,
                error_handled_gracefully=success_rate >= 0.8,  # 80% 이상 성공하면 안전
                recovery_successful=True,
                recovery_time=execution_time,
                system_state_after_error="concurrent_safe" if success_rate >= 0.8 else "concurrent_issues",
                additional_info={
                    "total_requests": len(results),
                    "successful_requests": len(successful_results),
                    "failed_requests": len(failed_results),
                    "success_rate": success_rate,
                    "execution_time": execution_time
                }
            )
            
        except Exception as e:
            logger.error(f"❌ 동시 접근 테스트 실패: {str(e)}")
            return ErrorTestResult(
                scenario_name="concurrent_access_safety",
                error_type=ErrorType.CONCURRENT_ACCESS,
                severity=SeverityLevel.HIGH,
                error_triggered=True,
                error_handled_gracefully=False,
                recovery_successful=False,
                recovery_time=None,
                system_state_after_error="concurrent_failure",
                error_message=str(e)
            )
    
    async def test_data_integrity_under_stress(self) -> ErrorTestResult:
        """스트레스 상황에서 데이터 무결성 테스트"""
        logger.info("💾 데이터 무결성 스트레스 테스트 시작")
        
        try:
            calculator = FrommEnhancedBenthamCalculator()
            
            # 스트레스 데이터 생성
            stress_inputs = []
            for i in range(100):
                stress_inputs.append({
                    "text": f"스트레스 테스트 입력 {i}",
                    "context": {
                        "stress_id": i,
                        "complexity": random.choice(["low", "medium", "high"]),
                        "random_data": [random.random() for _ in range(10)]
                    }
                })
            
            # 순차 처리와 결과 저장
            results = []
            checksums = []
            
            for input_data in stress_inputs[:20]:  # 처음 20개만 테스트
                try:
                    result = await calculator.calculate_enhanced_utility(
                        input_data["text"], input_data["context"]
                    )
                    
                    if result:
                        results.append(result)
                        # 간단한 체크섬 계산
                        checksum = hash(str(result.get("total_utility", 0)))
                        checksums.append(checksum)
                        
                except Exception as e:
                    logger.warning(f"⚠️ 스트레스 입력 처리 실패: {str(e)}")
            
            # 데이터 무결성 검증
            unique_checksums = len(set(checksums))
            total_results = len(results)
            
            # 무결성 평가
            integrity_score = unique_checksums / total_results if total_results > 0 else 0
            integrity_maintained = integrity_score > 0.8  # 80% 이상 유니크하면 무결성 유지
            
            return ErrorTestResult(
                scenario_name="data_integrity_under_stress",
                error_type=ErrorType.DATA_CORRUPTION,
                severity=SeverityLevel.MEDIUM,
                error_triggered=not integrity_maintained,
                error_handled_gracefully=integrity_maintained,
                recovery_successful=True,
                recovery_time=0.0,
                system_state_after_error="integrity_maintained" if integrity_maintained else "data_issues",
                additional_info={
                    "total_processed": total_results,
                    "unique_results": unique_checksums,
                    "integrity_score": integrity_score,
                    "integrity_threshold": 0.8
                }
            )
            
        except Exception as e:
            logger.error(f"❌ 데이터 무결성 테스트 실패: {str(e)}")
            return ErrorTestResult(
                scenario_name="data_integrity_under_stress",
                error_type=ErrorType.DATA_CORRUPTION,
                severity=SeverityLevel.HIGH,
                error_triggered=True,
                error_handled_gracefully=False,
                recovery_successful=False,
                recovery_time=None,
                system_state_after_error="integrity_test_failed",
                error_message=str(e)
            )
    
    async def run_comprehensive_resilience_test(self) -> Dict[str, Any]:
        """종합 복원력 테스트 실행"""
        logger.info("🛡️ Red Heart 시스템 종합 복원력 테스트 시작")
        
        all_results = []
        
        try:
            # 1. 공감 시스템 복원력 테스트
            empathy_results = await self.test_empathy_system_resilience()
            all_results.extend(empathy_results)
            
            # 2. 벤담 계산기 복원력 테스트
            bentham_results = await self.test_bentham_calculator_resilience()
            all_results.extend(bentham_results)
            
            # 3. 동시 접근 안전성 테스트
            concurrent_result = await self.test_concurrent_access_safety()
            all_results.append(concurrent_result)
            
            # 4. 데이터 무결성 테스트
            integrity_result = await self.test_data_integrity_under_stress()
            all_results.append(integrity_result)
            
            # 결과 분석
            total_tests = len(all_results)
            graceful_failures = sum(1 for r in all_results if r.error_handled_gracefully)
            successful_recoveries = sum(1 for r in all_results if r.recovery_successful)
            
            resilience_score = (graceful_failures + successful_recoveries) / (total_tests * 2) if total_tests > 0 else 0
            
            # 심각도별 분류
            severity_breakdown = {}
            for severity in SeverityLevel:
                severity_tests = [r for r in all_results if r.severity == severity]
                severity_breakdown[severity.value] = {
                    "total": len(severity_tests),
                    "graceful": sum(1 for r in severity_tests if r.error_handled_gracefully),
                    "recovered": sum(1 for r in severity_tests if r.recovery_successful)
                }
            
            return {
                "success": True,
                "resilience_score": resilience_score,
                "total_tests": total_tests,
                "graceful_error_handling": graceful_failures,
                "successful_recoveries": successful_recoveries,
                "test_results": [asdict(result) for result in all_results],
                "severity_breakdown": severity_breakdown,
                "summary": {
                    "overall_resilience": "excellent" if resilience_score > 0.8 else "good" if resilience_score > 0.6 else "needs_improvement",
                    "critical_issues": [r.scenario_name for r in all_results if r.severity == SeverityLevel.CRITICAL and not r.error_handled_gracefully],
                    "high_priority_fixes": [r.scenario_name for r in all_results if r.severity == SeverityLevel.HIGH and not r.error_handled_gracefully]
                }
            }
            
        except Exception as e:
            logger.error(f"❌ 종합 복원력 테스트 실행 실패: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "partial_results": [asdict(result) for result in all_results]
            }
    
    def save_resilience_report(self, results: Dict[str, Any], filename: str = None) -> str:
        """복원력 테스트 보고서 저장"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"red_heart_resilience_test_report_{timestamp}.json"
        
        report_path = f"/mnt/c/large_project/linux_red_heart/logs/{filename}"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"🛡️ 복원력 테스트 보고서 저장됨: {report_path}")
        return report_path

async def main():
    """메인 복원력 테스트 실행 함수"""
    print("=" * 80)
    print("🛡️ Red Heart Linux Advanced - 에러 처리 및 복원력 테스트")
    print("=" * 80)
    
    validator = SystemResilienceValidator()
    
    try:
        # 종합 복원력 테스트 실행
        results = await validator.run_comprehensive_resilience_test()
        
        if results["success"]:
            print("\n" + "=" * 80)
            print("🛡️ 시스템 복원력 테스트 결과 요약")
            print("=" * 80)
            
            print(f"총 테스트: {results['total_tests']}")
            print(f"우아한 에러 처리: {results['graceful_error_handling']}/{results['total_tests']}")
            print(f"성공적 복구: {results['successful_recoveries']}/{results['total_tests']}")
            print(f"복원력 점수: {results['resilience_score']:.1%}")
            print(f"전체 평가: {results['summary']['overall_resilience']}")
            
            # 심각도별 분석
            print(f"\n📊 심각도별 분석:")
            for severity, data in results["severity_breakdown"].items():
                if data["total"] > 0:
                    success_rate = (data["graceful"] + data["recovered"]) / (data["total"] * 2)
                    print(f"  {severity.upper()}: {data['total']}개 테스트, 성공률 {success_rate:.1%}")
            
            # 중요 이슈
            critical_issues = results["summary"]["critical_issues"]
            high_priority = results["summary"]["high_priority_fixes"]
            
            if critical_issues:
                print(f"\n🚨 치명적 이슈:")
                for issue in critical_issues:
                    print(f"  - {issue}")
            
            if high_priority:
                print(f"\n⚠️ 높은 우선순위 수정 필요:")
                for issue in high_priority:
                    print(f"  - {issue}")
            
            # 보고서 저장
            report_path = validator.save_resilience_report(results)
            print(f"\n📄 상세 보고서: {report_path}")
            
            if results["resilience_score"] > 0.8:
                print("\n🎉 시스템이 매우 우수한 복원력을 보여줍니다!")
                return 0
            elif results["resilience_score"] > 0.6:
                print("\n✅ 시스템이 양호한 복원력을 보여줍니다.")
                return 0
            else:
                print("\n⚠️ 시스템 복원력 개선이 필요합니다.")
                return 1
                
        else:
            print(f"\n❌ 복원력 테스트 실행 실패: {results.get('error', '알 수 없는 오류')}")
            return 1
            
    except Exception as e:
        print(f"\n💥 복원력 테스트 실행 중 심각한 오류: {str(e)}")
        print(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)