#!/usr/bin/env python3
"""
Red Heart AI 통합 시스템 벤치마크
성능 측정 및 메모리 사용량 분석
"""

import asyncio
import time
import torch
import psutil
import numpy as np
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging
from dataclasses import dataclass, field
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# 프로젝트 경로 추가
import sys
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from main_unified import UnifiedInferenceSystem, InferenceConfig, MemoryMode

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger('Benchmark')


@dataclass
class BenchmarkResult:
    """벤치마크 결과"""
    memory_mode: str
    num_iterations: int
    batch_size: int
    
    # 시간 측정
    total_time: float = 0.0
    avg_time_per_request: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    std_time: float = 0.0
    percentile_50: float = 0.0
    percentile_95: float = 0.0
    percentile_99: float = 0.0
    
    # 메모리 측정
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    min_memory_mb: float = float('inf')
    max_memory_mb: float = 0.0
    
    # GPU 측정 (있을 경우)
    peak_gpu_memory_mb: float = 0.0
    avg_gpu_memory_mb: float = 0.0
    
    # 처리량
    throughput: float = 0.0  # requests per second
    
    # 모듈별 시간
    module_times: Dict[str, float] = field(default_factory=dict)
    
    # 오류율
    error_count: int = 0
    error_rate: float = 0.0
    
    # 개별 측정값
    individual_times: List[float] = field(default_factory=list)
    individual_memories: List[float] = field(default_factory=list)


class UnifiedSystemBenchmark:
    """통합 시스템 벤치마크"""
    
    def __init__(self, memory_mode: MemoryMode, batch_size: int = 1):
        self.memory_mode = memory_mode
        self.batch_size = batch_size
        self.system = None
        self.process = psutil.Process()
        
        # 테스트 텍스트 세트
        self.test_texts = [
            "오늘 날씨가 정말 좋아서 기분이 좋습니다.",
            "시험에 떨어져서 너무 속상하고 우울합니다.",
            "친구들과 함께 시간을 보내니 행복해요.",
            "미래가 불확실해서 걱정이 많이 됩니다.",
            "새로운 프로젝트를 시작하게 되어 설레고 기대됩니다.",
            "실수를 해서 후회가 되고 자책감이 듭니다.",
            "가족과 함께 보내는 시간이 소중하고 감사합니다.",
            "일이 잘 풀리지 않아서 답답하고 화가 납니다.",
            "목표를 달성해서 뿌듯하고 자랑스럽습니다.",
            "혼자 있으니 외롭고 쓸쓸한 기분이 듭니다.",
            "좋은 소식을 들어서 기쁘고 신이 납니다.",
            "실망스러운 결과에 좌절감을 느낍니다.",
            "도전적인 과제를 앞두고 긴장되지만 동기부여가 됩니다.",
            "평화로운 순간을 즐기며 편안함을 느낍니다.",
            "예상치 못한 선물을 받아서 놀랍고 감동적입니다."
        ]
        
    async def initialize(self):
        """시스템 초기화"""
        logger.info(f"벤치마크 시스템 초기화 (모드: {self.memory_mode.value})")
        
        config = InferenceConfig(
            memory_mode=self.memory_mode,
            auto_memory_mode=False,
            batch_size=self.batch_size,
            enable_monitoring=False,  # 모니터링 비활성화로 순수 성능 측정
            verbose=False
        )
        
        self.system = UnifiedInferenceSystem(config)
        await self.system.initialize()
        
        # 워밍업
        await self._warmup()
        
        logger.info("벤치마크 시스템 초기화 완료")
    
    async def _warmup(self):
        """워밍업 실행"""
        logger.info("워밍업 중...")
        for i in range(3):
            await self.system.analyze(self.test_texts[0])
        logger.info("워밍업 완료")
    
    def _get_memory_usage(self) -> Tuple[float, float]:
        """현재 메모리 사용량 (MB)"""
        # CPU 메모리
        cpu_memory = self.process.memory_info().rss / 1024 / 1024
        
        # GPU 메모리
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
        
        return cpu_memory, gpu_memory
    
    async def run_benchmark(self, num_iterations: int = 100) -> BenchmarkResult:
        """벤치마크 실행"""
        logger.info(f"벤치마크 시작: {num_iterations}회 반복")
        
        result = BenchmarkResult(
            memory_mode=self.memory_mode.value,
            num_iterations=num_iterations,
            batch_size=self.batch_size
        )
        
        # 측정값 저장
        times = []
        memories = []
        gpu_memories = []
        errors = 0
        
        # 진행 표시
        progress_interval = max(1, num_iterations // 10)
        
        # 벤치마크 실행
        start_total = time.time()
        
        for i in range(num_iterations):
            # 텍스트 선택 (순환)
            text = self.test_texts[i % len(self.test_texts)]
            
            # 메모리 측정 (시작)
            mem_before, gpu_mem_before = self._get_memory_usage()
            
            # 시간 측정
            start = time.perf_counter()
            
            try:
                # 분석 실행
                output = await self.system.analyze(text)
                
                # 시간 기록
                elapsed = time.perf_counter() - start
                times.append(elapsed)
                
                # 모듈별 시간 기록 (첫 번째 실행에서만)
                if i == 0 and 'processing_time' in output:
                    result.module_times['total'] = output['processing_time']
                
            except Exception as e:
                logger.warning(f"반복 {i} 실패: {e}")
                errors += 1
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            
            # 메모리 측정 (종료)
            mem_after, gpu_mem_after = self._get_memory_usage()
            memories.append(mem_after)
            gpu_memories.append(gpu_mem_after)
            
            # 진행 상황 출력
            if (i + 1) % progress_interval == 0:
                progress = ((i + 1) / num_iterations) * 100
                logger.info(f"  진행: {progress:.0f}% ({i+1}/{num_iterations})")
        
        total_time = time.time() - start_total
        
        # 통계 계산
        times_array = np.array(times)
        memories_array = np.array(memories)
        gpu_memories_array = np.array(gpu_memories) if gpu_memories else np.array([0])
        
        result.total_time = total_time
        result.avg_time_per_request = np.mean(times_array)
        result.min_time = np.min(times_array)
        result.max_time = np.max(times_array)
        result.std_time = np.std(times_array)
        result.percentile_50 = np.percentile(times_array, 50)
        result.percentile_95 = np.percentile(times_array, 95)
        result.percentile_99 = np.percentile(times_array, 99)
        
        result.avg_memory_mb = np.mean(memories_array)
        result.min_memory_mb = np.min(memories_array)
        result.max_memory_mb = np.max(memories_array)
        result.peak_memory_mb = np.max(memories_array)
        
        result.avg_gpu_memory_mb = np.mean(gpu_memories_array)
        result.peak_gpu_memory_mb = np.max(gpu_memories_array)
        
        result.throughput = num_iterations / total_time
        result.error_count = errors
        result.error_rate = errors / num_iterations
        
        result.individual_times = times
        result.individual_memories = memories.tolist()
        
        return result
    
    def print_results(self, result: BenchmarkResult):
        """결과 출력"""
        print("\n" + "=" * 70)
        print(f"📊 벤치마크 결과 - {result.memory_mode} 모드")
        print("=" * 70)
        
        print(f"\n⏱️ 시간 성능:")
        print(f"  총 실행 시간: {result.total_time:.2f}초")
        print(f"  평균 처리 시간: {result.avg_time_per_request*1000:.2f}ms")
        print(f"  최소 시간: {result.min_time*1000:.2f}ms")
        print(f"  최대 시간: {result.max_time*1000:.2f}ms")
        print(f"  표준편차: {result.std_time*1000:.2f}ms")
        print(f"  중앙값 (P50): {result.percentile_50*1000:.2f}ms")
        print(f"  P95: {result.percentile_95*1000:.2f}ms")
        print(f"  P99: {result.percentile_99*1000:.2f}ms")
        print(f"  처리량: {result.throughput:.2f} req/s")
        
        print(f"\n💾 메모리 사용:")
        print(f"  평균 CPU 메모리: {result.avg_memory_mb:.1f}MB")
        print(f"  최대 CPU 메모리: {result.peak_memory_mb:.1f}MB")
        print(f"  최소 CPU 메모리: {result.min_memory_mb:.1f}MB")
        
        if result.avg_gpu_memory_mb > 0:
            print(f"  평균 GPU 메모리: {result.avg_gpu_memory_mb:.1f}MB")
            print(f"  최대 GPU 메모리: {result.peak_gpu_memory_mb:.1f}MB")
        
        if result.error_count > 0:
            print(f"\n⚠️ 오류:")
            print(f"  오류 횟수: {result.error_count}")
            print(f"  오류율: {result.error_rate*100:.2f}%")
        
        print("=" * 70)
    
    def save_results(self, result: BenchmarkResult, filepath: Path):
        """결과 저장"""
        result_dict = {
            'timestamp': datetime.now().isoformat(),
            'memory_mode': result.memory_mode,
            'num_iterations': result.num_iterations,
            'batch_size': result.batch_size,
            'metrics': {
                'avg_time_ms': result.avg_time_per_request * 1000,
                'min_time_ms': result.min_time * 1000,
                'max_time_ms': result.max_time * 1000,
                'std_time_ms': result.std_time * 1000,
                'p50_ms': result.percentile_50 * 1000,
                'p95_ms': result.percentile_95 * 1000,
                'p99_ms': result.percentile_99 * 1000,
                'throughput_rps': result.throughput,
                'avg_memory_mb': result.avg_memory_mb,
                'peak_memory_mb': result.peak_memory_mb,
                'avg_gpu_memory_mb': result.avg_gpu_memory_mb,
                'peak_gpu_memory_mb': result.peak_gpu_memory_mb,
                'error_rate': result.error_rate
            },
            'raw_data': {
                'times': result.individual_times[:100],  # 처음 100개만
                'memories': result.individual_memories[:100]
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        logger.info(f"결과 저장: {filepath}")
    
    def plot_results(self, result: BenchmarkResult, save_path: Path = None):
        """결과 시각화"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # 1. 응답 시간 분포
            axes[0, 0].hist(np.array(result.individual_times) * 1000, bins=30, edgecolor='black')
            axes[0, 0].axvline(result.avg_time_per_request * 1000, color='red', linestyle='--', label='평균')
            axes[0, 0].axvline(result.percentile_95 * 1000, color='orange', linestyle='--', label='P95')
            axes[0, 0].set_xlabel('응답 시간 (ms)')
            axes[0, 0].set_ylabel('빈도')
            axes[0, 0].set_title('응답 시간 분포')
            axes[0, 0].legend()
            
            # 2. 시간별 응답 시간 추이
            axes[0, 1].plot(np.array(result.individual_times) * 1000)
            axes[0, 1].set_xlabel('요청 번호')
            axes[0, 1].set_ylabel('응답 시간 (ms)')
            axes[0, 1].set_title('응답 시간 추이')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. 메모리 사용량 추이
            axes[1, 0].plot(result.individual_memories)
            axes[1, 0].axhline(result.avg_memory_mb, color='red', linestyle='--', label='평균')
            axes[1, 0].set_xlabel('요청 번호')
            axes[1, 0].set_ylabel('메모리 (MB)')
            axes[1, 0].set_title('메모리 사용량 추이')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. 성능 요약
            axes[1, 1].axis('off')
            summary_text = f"""
            성능 요약 ({result.memory_mode} 모드)
            
            평균 응답: {result.avg_time_per_request*1000:.1f}ms
            P95: {result.percentile_95*1000:.1f}ms
            P99: {result.percentile_99*1000:.1f}ms
            
            처리량: {result.throughput:.1f} req/s
            
            평균 메모리: {result.avg_memory_mb:.0f}MB
            최대 메모리: {result.peak_memory_mb:.0f}MB
            
            오류율: {result.error_rate*100:.1f}%
            """
            axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')
            
            plt.suptitle(f'Red Heart AI 벤치마크 결과 - {result.memory_mode} 모드', fontsize=14)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150)
                logger.info(f"그래프 저장: {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("matplotlib 없음 - 시각화 건너뜀")


async def compare_memory_modes(num_iterations: int = 50):
    """메모리 모드별 비교 벤치마크"""
    
    modes_to_test = [
        MemoryMode.MINIMAL,
        MemoryMode.LIGHT,
        MemoryMode.NORMAL,
        MemoryMode.HEAVY
    ]
    
    results = {}
    
    for mode in modes_to_test:
        logger.info(f"\n{'='*70}")
        logger.info(f"테스트 모드: {mode.value}")
        logger.info(f"{'='*70}")
        
        try:
            benchmark = UnifiedSystemBenchmark(mode)
            await benchmark.initialize()
            
            result = await benchmark.run_benchmark(num_iterations)
            benchmark.print_results(result)
            
            # 결과 저장
            filepath = Path(f"benchmark_{mode.value}_{int(time.time())}.json")
            benchmark.save_results(result, filepath)
            
            # 시각화
            plot_path = Path(f"benchmark_{mode.value}_{int(time.time())}.png")
            benchmark.plot_results(result, plot_path)
            
            results[mode.value] = result
            
        except Exception as e:
            logger.error(f"{mode.value} 모드 벤치마크 실패: {e}")
            continue
        
        # 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 모드 간 대기
        await asyncio.sleep(5)
    
    # 비교 요약
    print("\n" + "=" * 70)
    print("📊 메모리 모드별 비교")
    print("=" * 70)
    
    print(f"\n{'모드':<10} {'평균(ms)':<12} {'P95(ms)':<12} {'처리량(req/s)':<15} {'메모리(MB)':<12}")
    print("-" * 70)
    
    for mode_name, result in results.items():
        print(f"{mode_name:<10} "
              f"{result.avg_time_per_request*1000:<12.1f} "
              f"{result.percentile_95*1000:<12.1f} "
              f"{result.throughput:<15.1f} "
              f"{result.peak_memory_mb:<12.0f}")
    
    return results


async def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='Red Heart AI 벤치마크')
    parser.add_argument('--memory-mode', default='normal',
                       choices=['minimal', 'light', 'normal', 'heavy', 'ultra', 'extreme'],
                       help='메모리 모드')
    parser.add_argument('--num-iterations', type=int, default=100,
                       help='반복 횟수')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='배치 크기')
    parser.add_argument('--compare', action='store_true',
                       help='모든 메모리 모드 비교')
    
    args = parser.parse_args()
    
    if args.compare:
        # 모드 비교
        await compare_memory_modes(args.num_iterations)
    else:
        # 단일 모드 벤치마크
        mode = MemoryMode[args.memory_mode.upper()]
        benchmark = UnifiedSystemBenchmark(mode, args.batch_size)
        
        await benchmark.initialize()
        result = await benchmark.run_benchmark(args.num_iterations)
        
        benchmark.print_results(result)
        
        # 결과 저장
        filepath = Path(f"benchmark_{mode.value}_{int(time.time())}.json")
        benchmark.save_results(result, filepath)
        
        # 시각화
        plot_path = Path(f"benchmark_{mode.value}_{int(time.time())}.png")
        benchmark.plot_results(result, plot_path)


if __name__ == '__main__':
    asyncio.run(main())