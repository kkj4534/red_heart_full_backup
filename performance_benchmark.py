"""
Red Heart Linux Advanced - 성능 벤치마크 및 메모리 최적화 시스템
GPU/CPU 사용량, 메모리 효율성, 처리 속도 최적화 분석

이 모듈은 다음 기능을 제공합니다:
1. 실시간 시스템 리소스 모니터링
2. 각 구성 요소별 성능 프로파일링
3. 메모리 누수 감지 및 최적화 제안
4. GPU 메모리 관리 및 동적 할당 최적화
5. 병목 지점 분석 및 성능 개선 권장사항
"""

import asyncio
import time
import psutil
import threading
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import gc
import numpy as np
from collections import defaultdict, deque
import tracemalloc

try:
    import torch
    import torch.profiler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

from config import (
    SYSTEM_CONFIG, 
    get_smart_device, 
    get_gpu_memory_info,
    gpu_model_context,
    setup_logging
)

# 로거 설정
logger = setup_logging()

@dataclass
class ResourceSnapshot:
    """시스템 리소스 스냅샷"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    gpu_memory_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None
    disk_io_read: float = 0
    disk_io_write: float = 0
    network_io_sent: float = 0
    network_io_recv: float = 0

@dataclass
class PerformanceMetrics:
    """성능 메트릭 데이터"""
    component_name: str
    execution_time: float
    memory_peak_mb: float
    memory_growth_mb: float
    cpu_usage_avg: float
    gpu_memory_peak_mb: Optional[float] = None
    gpu_utilization_avg: Optional[float] = None
    throughput_ops_per_sec: Optional[float] = None
    error_count: int = 0
    cache_hit_rate: Optional[float] = None

@dataclass
class OptimizationRecommendation:
    """최적화 권장사항"""
    category: str  # memory, cpu, gpu, io, algorithm
    priority: str  # high, medium, low
    description: str
    estimated_improvement: str
    implementation_effort: str  # low, medium, high
    code_location: Optional[str] = None

class SystemResourceMonitor:
    """실시간 시스템 리소스 모니터링"""
    
    def __init__(self, sampling_interval: float = 0.5):
        self.sampling_interval = sampling_interval
        self.is_monitoring = False
        self.resource_history: deque = deque(maxlen=1000)  # 최근 1000개 샘플 유지
        self.monitor_thread = None
        self.start_time = None
        
    def start_monitoring(self):
        """모니터링 시작"""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.start_time = datetime.now()
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("🔍 시스템 리소스 모니터링 시작")
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("⏹️ 시스템 리소스 모니터링 중지")
    
    def _monitoring_loop(self):
        """모니터링 루프"""
        last_disk_io = psutil.disk_io_counters()
        last_net_io = psutil.net_io_counters()
        
        while self.is_monitoring:
            try:
                # 기본 시스템 메트릭
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                
                # 디스크 I/O 변화량
                current_disk_io = psutil.disk_io_counters()
                disk_read_delta = current_disk_io.read_bytes - last_disk_io.read_bytes
                disk_write_delta = current_disk_io.write_bytes - last_disk_io.write_bytes
                last_disk_io = current_disk_io
                
                # 네트워크 I/O 변화량
                current_net_io = psutil.net_io_counters()
                net_sent_delta = current_net_io.bytes_sent - last_net_io.bytes_sent
                net_recv_delta = current_net_io.bytes_recv - last_net_io.bytes_recv
                last_net_io = current_net_io
                
                # GPU 메트릭 (사용 가능한 경우)
                gpu_memory_mb = None
                gpu_utilization = None
                
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    gpu_info = get_gpu_memory_info()
                    if gpu_info:
                        gpu_memory_mb = gpu_info['allocated_mb']
                        gpu_utilization = gpu_info['usage_percent']
                
                # 스냅샷 생성
                snapshot = ResourceSnapshot(
                    timestamp=datetime.now(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_mb=memory.used / (1024 * 1024),
                    gpu_memory_mb=gpu_memory_mb,
                    gpu_utilization=gpu_utilization,
                    disk_io_read=disk_read_delta / (1024 * 1024),  # MB/s
                    disk_io_write=disk_write_delta / (1024 * 1024),  # MB/s
                    network_io_sent=net_sent_delta / (1024 * 1024),  # MB/s
                    network_io_recv=net_recv_delta / (1024 * 1024)  # MB/s
                )
                
                self.resource_history.append(snapshot)
                
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                logger.error(f"모니터링 중 오류: {str(e)}")
                time.sleep(self.sampling_interval)
    
    def get_current_stats(self) -> Dict[str, Any]:
        """현재 리소스 통계"""
        if not self.resource_history:
            return {}
        
        recent_snapshots = list(self.resource_history)[-20:]  # 최근 20개
        
        return {
            "current": asdict(recent_snapshots[-1]) if recent_snapshots else {},
            "avg_last_20": {
                "cpu_percent": np.mean([s.cpu_percent for s in recent_snapshots]),
                "memory_percent": np.mean([s.memory_percent for s in recent_snapshots]),
                "memory_mb": np.mean([s.memory_mb for s in recent_snapshots]),
                "gpu_memory_mb": np.mean([s.gpu_memory_mb for s in recent_snapshots if s.gpu_memory_mb]),
                "gpu_utilization": np.mean([s.gpu_utilization for s in recent_snapshots if s.gpu_utilization])
            },
            "peak": {
                "cpu_percent": max(s.cpu_percent for s in recent_snapshots),
                "memory_mb": max(s.memory_mb for s in recent_snapshots),
                "gpu_memory_mb": max(s.gpu_memory_mb for s in recent_snapshots if s.gpu_memory_mb) if any(s.gpu_memory_mb for s in recent_snapshots) else None
            }
        }

class ComponentProfiler:
    """개별 구성 요소 성능 프로파일링"""
    
    def __init__(self, component_name: str, monitor: SystemResourceMonitor):
        self.component_name = component_name
        self.monitor = monitor
        self.start_time = None
        self.start_memory = None
        self.start_gpu_memory = None
        self.start_snapshot = None
        self.operation_count = 0
        self.error_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # 메모리 추적 시작
        tracemalloc.start()
        
    def __enter__(self):
        """프로파일링 시작"""
        self.start_time = time.time()
        self.operation_count = 0
        self.error_count = 0
        
        # 초기 리소스 상태 캡처
        if self.monitor.resource_history:
            self.start_snapshot = self.monitor.resource_history[-1]
        
        # 메모리 기준점 설정
        self.start_memory = tracemalloc.get_traced_memory()[0]
        
        # GPU 메모리 기준점
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_info = get_gpu_memory_info()
            self.start_gpu_memory = gpu_info['allocated_mb'] if gpu_info else 0
        
        logger.info(f"🔍 {self.component_name} 프로파일링 시작")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """프로파일링 종료 및 메트릭 계산"""
        execution_time = time.time() - self.start_time
        
        # 메모리 사용량 계산
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        memory_growth = (current_memory - self.start_memory) / (1024 * 1024)  # MB
        memory_peak = peak_memory / (1024 * 1024)  # MB
        
        # GPU 메모리 계산
        gpu_memory_peak = None
        if TORCH_AVAILABLE and torch.cuda.is_available() and self.start_gpu_memory is not None:
            gpu_info = get_gpu_memory_info()
            current_gpu_memory = gpu_info['allocated_mb'] if gpu_info else 0
            gpu_memory_peak = max(current_gpu_memory, self.start_gpu_memory)
        
        # CPU 사용률 평균 계산
        cpu_usage_avg = 0
        gpu_utilization_avg = None
        
        if self.monitor.resource_history and self.start_snapshot:
            relevant_snapshots = [
                s for s in self.monitor.resource_history 
                if s.timestamp >= self.start_snapshot.timestamp
            ]
            if relevant_snapshots:
                cpu_usage_avg = np.mean([s.cpu_percent for s in relevant_snapshots])
                gpu_utils = [s.gpu_utilization for s in relevant_snapshots if s.gpu_utilization]
                if gpu_utils:
                    gpu_utilization_avg = np.mean(gpu_utils)
        
        # 처리량 계산
        throughput = self.operation_count / execution_time if execution_time > 0 else 0
        
        # 캐시 히트율 계산
        cache_hit_rate = None
        if self.cache_hits + self.cache_misses > 0:
            cache_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses)
        
        self.metrics = PerformanceMetrics(
            component_name=self.component_name,
            execution_time=execution_time,
            memory_peak_mb=memory_peak,
            memory_growth_mb=memory_growth,
            cpu_usage_avg=cpu_usage_avg,
            gpu_memory_peak_mb=gpu_memory_peak,
            gpu_utilization_avg=gpu_utilization_avg,
            throughput_ops_per_sec=throughput,
            error_count=self.error_count,
            cache_hit_rate=cache_hit_rate
        )
        
        logger.info(f"✅ {self.component_name} 프로파일링 완료: "
                   f"{execution_time:.3f}s, {memory_peak:.1f}MB peak, "
                   f"{throughput:.1f} ops/s")
    
    def record_operation(self):
        """작업 하나 완료 기록"""
        self.operation_count += 1
    
    def record_error(self):
        """에러 발생 기록"""
        self.error_count += 1
    
    def record_cache_hit(self):
        """캐시 히트 기록"""
        self.cache_hits += 1
    
    def record_cache_miss(self):
        """캐시 미스 기록"""
        self.cache_misses += 1

class PerformanceBenchmarkSuite:
    """종합 성능 벤치마크 및 최적화 분석"""
    
    def __init__(self):
        self.monitor = SystemResourceMonitor(sampling_interval=0.2)
        self.component_metrics: List[PerformanceMetrics] = []
        self.optimization_recommendations: List[OptimizationRecommendation] = []
        
        # 벤치마크 시나리오
        self.benchmark_scenarios = [
            {
                "name": "small_empathy_analysis",
                "text": "오늘 기분이 좋지 않아요.",
                "context": {"complexity": "low"},
                "expected_time": 2.0,
                "iterations": 10
            },
            {
                "name": "medium_semantic_analysis", 
                "text": "회사에서 승진 기회가 생겼지만 이사를 가야 하는 상황이라 가족과 상의가 필요합니다.",
                "context": {"complexity": "medium", "domains": ["career", "family"]},
                "expected_time": 5.0,
                "iterations": 5
            },
            {
                "name": "complex_moral_dilemma",
                "text": "환경 보호를 위해 플라스틱 사용을 줄이고 싶지만, 경제적 여건상 친환경 제품을 구매하기 어려운 상황입니다. 또한 직장에서는 환경에 해로운 사업을 하고 있어서 윤리적 갈등을 겪고 있습니다.",
                "context": {"complexity": "high", "moral_weight": 0.9, "domains": ["environment", "economics", "ethics"]},
                "expected_time": 8.0,
                "iterations": 3
            },
            {
                "name": "batch_processing_test",
                "texts": [
                    "친구와 약속을 잡았어요.",
                    "오늘 날씨가 좋네요.", 
                    "새로운 프로젝트를 시작했습니다.",
                    "가족과 함께 시간을 보냈어요.",
                    "건강한 식사를 했습니다."
                ],
                "context": {"batch_mode": True},
                "expected_time": 3.0,
                "iterations": 5
            }
        ]
    
    async def benchmark_empathy_learning_system(self) -> PerformanceMetrics:
        """공감 학습 시스템 벤치마크"""
        from advanced_hierarchical_emotion_system import EnhancedEmpathyLearner
        
        with ComponentProfiler("EnhancedEmpathyLearner", self.monitor) as profiler:
            empathy_learner = EnhancedEmpathyLearner()
            await empathy_learner.initialize()
            
            # 각 시나리오 실행
            for scenario in self.benchmark_scenarios[:3]:  # 배치 테스트 제외
                for i in range(scenario["iterations"]):
                    try:
                        start_time = time.time()
                        result = await empathy_learner.process_empathy_learning(
                            scenario["text"], scenario["context"]
                        )
                        
                        # 성능 기대치 검증
                        elapsed = time.time() - start_time
                        if elapsed > scenario["expected_time"]:
                            logger.warning(f"⚠️ {scenario['name']} 예상보다 느림: {elapsed:.2f}s > {scenario['expected_time']}s")
                        
                        profiler.record_operation()
                        
                        # 결과 유효성 검증
                        assert result is not None
                        assert 'empathy_score' in result
                        assert 0 <= result['empathy_score'] <= 1
                        
                    except Exception as e:
                        profiler.record_error()
                        logger.error(f"공감 학습 벤치마크 오류: {str(e)}")
        
        return profiler.metrics
    
    async def benchmark_bentham_calculator(self) -> PerformanceMetrics:
        """벤담 계산기 벤치마크"""
        from advanced_bentham_calculator import FrommEnhancedBenthamCalculator
        
        with ComponentProfiler("FrommEnhancedBenthamCalculator", self.monitor) as profiler:
            calculator = FrommEnhancedBenthamCalculator()
            
            for scenario in self.benchmark_scenarios[:3]:
                for i in range(scenario["iterations"]):
                    try:
                        start_time = time.time()
                        result = await calculator.calculate_enhanced_utility(
                            scenario["text"], scenario["context"]
                        )
                        
                        elapsed = time.time() - start_time
                        profiler.record_operation()
                        
                        # 결과 검증
                        assert result is not None
                        assert 'total_utility' in result
                        assert result['total_utility'] >= 0
                        
                    except Exception as e:
                        profiler.record_error()
                        logger.error(f"벤담 계산기 벤치마크 오류: {str(e)}")
        
        return profiler.metrics
    
    async def benchmark_memory_efficiency(self) -> Dict[str, Any]:
        """메모리 효율성 벤치마크"""
        logger.info("💾 메모리 효율성 벤치마크 시작")
        
        # 가비지 컬렉션 강제 실행
        gc.collect()
        initial_memory = psutil.virtual_memory().used / (1024 * 1024)
        
        # 초기 GPU 메모리
        initial_gpu_memory = 0
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gpu_info = get_gpu_memory_info()
            initial_gpu_memory = gpu_info['allocated_mb'] if gpu_info else 0
        
        # 메모리 사용량 추적
        memory_samples = []
        gpu_memory_samples = []
        
        # 대량 처리 시뮬레이션
        from advanced_hierarchical_emotion_system import EnhancedEmpathyLearner
        empathy_learner = EnhancedEmpathyLearner()
        await empathy_learner.initialize()
        
        for i in range(50):  # 50회 연속 처리
            # 현재 메모리 사용량 기록
            current_memory = psutil.virtual_memory().used / (1024 * 1024)
            memory_samples.append(current_memory - initial_memory)
            
            if TORCH_AVAILABLE and torch.cuda.is_available():
                gpu_info = get_gpu_memory_info()
                current_gpu = gpu_info['allocated_mb'] if gpu_info else 0
                gpu_memory_samples.append(current_gpu - initial_gpu_memory)
            
            # 처리 실행
            await empathy_learner.process_empathy_learning(
                f"테스트 문장 {i}: 다양한 감정과 상황을 포함한 복잡한 시나리오",
                {"iteration": i}
            )
            
            # 주기적 가비지 컬렉션
            if i % 10 == 0:
                gc.collect()
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # 최종 가비지 컬렉션
        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        final_memory = psutil.virtual_memory().used / (1024 * 1024)
        memory_leak = final_memory - initial_memory
        
        final_gpu_memory = 0
        gpu_memory_leak = 0
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_info = get_gpu_memory_info()
            final_gpu_memory = gpu_info['allocated_mb'] if gpu_info else 0
            gpu_memory_leak = final_gpu_memory - initial_gpu_memory
        
        return {
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "memory_leak_mb": memory_leak,
            "peak_memory_usage_mb": max(memory_samples) if memory_samples else 0,
            "avg_memory_usage_mb": np.mean(memory_samples) if memory_samples else 0,
            "initial_gpu_memory_mb": initial_gpu_memory,
            "final_gpu_memory_mb": final_gpu_memory, 
            "gpu_memory_leak_mb": gpu_memory_leak,
            "peak_gpu_memory_mb": max(gpu_memory_samples) if gpu_memory_samples else 0,
            "memory_stability": memory_leak < 50,  # 50MB 이하면 안정적
            "gpu_memory_stability": gpu_memory_leak < 100  # 100MB 이하면 안정적
        }
    
    def analyze_bottlenecks(self) -> List[OptimizationRecommendation]:
        """병목 지점 분석 및 최적화 권장사항"""
        recommendations = []
        
        if not self.component_metrics:
            return recommendations
        
        # 실행 시간 분석
        slow_components = [m for m in self.component_metrics if m.execution_time > 5.0]
        for component in slow_components:
            recommendations.append(OptimizationRecommendation(
                category="performance",
                priority="high",
                description=f"{component.component_name}의 실행 시간이 {component.execution_time:.2f}초로 느립니다.",
                estimated_improvement="30-50% 속도 향상 가능",
                implementation_effort="medium",
                code_location=f"{component.component_name} 클래스"
            ))
        
        # 메모리 사용량 분석
        memory_heavy_components = [m for m in self.component_metrics if m.memory_peak_mb > 500]
        for component in memory_heavy_components:
            recommendations.append(OptimizationRecommendation(
                category="memory",
                priority="high",
                description=f"{component.component_name}이 {component.memory_peak_mb:.1f}MB의 메모리를 사용합니다.",
                estimated_improvement="20-40% 메모리 절약 가능",
                implementation_effort="medium",
                code_location=f"{component.component_name} 메모리 관리"
            ))
        
        # GPU 메모리 분석
        gpu_heavy_components = [m for m in self.component_metrics 
                               if m.gpu_memory_peak_mb and m.gpu_memory_peak_mb > 1000]
        for component in gpu_heavy_components:
            recommendations.append(OptimizationRecommendation(
                category="gpu",
                priority="medium",
                description=f"{component.component_name}이 {component.gpu_memory_peak_mb:.1f}MB의 GPU 메모리를 사용합니다.",
                estimated_improvement="GPU 메모리 효율성 개선 가능",
                implementation_effort="low",
                code_location="GPU 메모리 관리 설정"
            ))
        
        # CPU 사용률 분석
        cpu_intensive_components = [m for m in self.component_metrics if m.cpu_usage_avg > 80]
        for component in cpu_intensive_components:
            recommendations.append(OptimizationRecommendation(
                category="cpu",
                priority="medium", 
                description=f"{component.component_name}이 평균 {component.cpu_usage_avg:.1f}%의 CPU를 사용합니다.",
                estimated_improvement="멀티프로세싱으로 부하 분산 가능",
                implementation_effort="high",
                code_location="병렬 처리 구현"
            ))
        
        # 에러율 분석
        error_prone_components = [m for m in self.component_metrics if m.error_count > 0]
        for component in error_prone_components:
            recommendations.append(OptimizationRecommendation(
                category="reliability",
                priority="high",
                description=f"{component.component_name}에서 {component.error_count}회의 에러가 발생했습니다.",
                estimated_improvement="시스템 안정성 향상",
                implementation_effort="medium",
                code_location=f"{component.component_name} 에러 처리"
            ))
        
        # 캐시 히트율 분석
        low_cache_components = [m for m in self.component_metrics 
                               if m.cache_hit_rate and m.cache_hit_rate < 0.7]
        for component in low_cache_components:
            recommendations.append(OptimizationRecommendation(
                category="algorithm",
                priority="medium",
                description=f"{component.component_name}의 캐시 히트율이 {component.cache_hit_rate:.1%}로 낮습니다.",
                estimated_improvement="캐시 효율성 개선으로 성능 향상",
                implementation_effort="low",
                code_location="캐시 전략 개선"
            ))
        
        return recommendations
    
    async def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """종합 성능 벤치마크 실행"""
        logger.info("🚀 Red Heart 시스템 종합 성능 벤치마크 시작")
        
        # 모니터링 시작
        self.monitor.start_monitoring()
        
        try:
            benchmark_start = time.time()
            
            # 1. 공감 학습 시스템 벤치마크
            logger.info("🧠 공감 학습 시스템 벤치마크...")
            empathy_metrics = await self.benchmark_empathy_learning_system()
            self.component_metrics.append(empathy_metrics)
            
            # 2. 벤담 계산기 벤치마크
            logger.info("⚖️ 벤담 계산기 벤치마크...")
            bentham_metrics = await self.benchmark_bentham_calculator()
            self.component_metrics.append(bentham_metrics)
            
            # 3. 메모리 효율성 벤치마크
            logger.info("💾 메모리 효율성 벤치마크...")
            memory_analysis = await self.benchmark_memory_efficiency()
            
            # 4. 병목 지점 분석
            logger.info("🔍 병목 지점 분석...")
            self.optimization_recommendations = self.analyze_bottlenecks()
            
            total_benchmark_time = time.time() - benchmark_start
            
            # 최종 시스템 상태
            final_stats = self.monitor.get_current_stats()
            
            return {
                "success": True,
                "total_benchmark_time": total_benchmark_time,
                "component_metrics": [asdict(m) for m in self.component_metrics],
                "memory_analysis": memory_analysis,
                "optimization_recommendations": [asdict(r) for r in self.optimization_recommendations],
                "system_stats": final_stats,
                "summary": {
                    "total_components_tested": len(self.component_metrics),
                    "avg_execution_time": np.mean([m.execution_time for m in self.component_metrics]),
                    "total_memory_peak": sum(m.memory_peak_mb for m in self.component_metrics),
                    "total_errors": sum(m.error_count for m in self.component_metrics),
                    "optimization_opportunities": len(self.optimization_recommendations)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ 벤치마크 실행 중 오류: {str(e)}")
            return {"success": False, "error": str(e)}
            
        finally:
            self.monitor.stop_monitoring()
    
    def save_benchmark_report(self, results: Dict[str, Any], filename: str = None) -> str:
        """벤치마크 보고서 저장"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"red_heart_performance_benchmark_{timestamp}.json"
        
        report_path = f"/mnt/c/large_project/linux_red_heart/logs/{filename}"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"📊 벤치마크 보고서 저장됨: {report_path}")
        return report_path

async def main():
    """메인 벤치마크 실행 함수"""
    print("=" * 80)
    print("🚀 Red Heart Linux Advanced - 성능 벤치마크 스위트")
    print("=" * 80)
    
    benchmark_suite = PerformanceBenchmarkSuite()
    
    try:
        # 종합 벤치마크 실행
        results = await benchmark_suite.run_comprehensive_benchmark()
        
        if results["success"]:
            print("\n" + "=" * 80)
            print("📊 성능 벤치마크 결과 요약")
            print("=" * 80)
            
            summary = results["summary"]
            print(f"테스트된 구성 요소: {summary['total_components_tested']}")
            print(f"평균 실행시간: {summary['avg_execution_time']:.3f}초")
            print(f"총 메모리 피크: {summary['total_memory_peak']:.1f}MB")
            print(f"총 에러 수: {summary['total_errors']}")
            print(f"최적화 기회: {summary['optimization_opportunities']}개")
            
            # 메모리 분석 결과
            memory = results["memory_analysis"]
            print(f"\n💾 메모리 분석:")
            print(f"  메모리 누수: {memory['memory_leak_mb']:.1f}MB")
            print(f"  GPU 메모리 누수: {memory['gpu_memory_leak_mb']:.1f}MB")
            print(f"  메모리 안정성: {'✅' if memory['memory_stability'] else '❌'}")
            print(f"  GPU 안정성: {'✅' if memory['gpu_memory_stability'] else '❌'}")
            
            # 최적화 권장사항
            recommendations = results["optimization_recommendations"]
            if recommendations:
                print(f"\n🔧 주요 최적화 권장사항:")
                for rec in recommendations[:5]:  # 상위 5개만 표시
                    priority_emoji = "🔴" if rec["priority"] == "high" else "🟡" if rec["priority"] == "medium" else "🟢"
                    print(f"  {priority_emoji} [{rec['category']}] {rec['description']}")
            
            # 보고서 저장
            report_path = benchmark_suite.save_benchmark_report(results)
            print(f"\n📄 상세 보고서: {report_path}")
            
            print("\n🎉 성능 벤치마크가 성공적으로 완료되었습니다!")
            
        else:
            print(f"\n❌ 벤치마크 실행 실패: {results.get('error', '알 수 없는 오류')}")
            return 1
            
    except Exception as e:
        print(f"\n💥 벤치마크 실행 중 심각한 오류: {str(e)}")
        print(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)