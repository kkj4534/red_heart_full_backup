"""
OOM (Out of Memory) Handler
메모리 부족 시 자동으로 배치 사이즈 조정 및 DSM 활성화
"""

import torch
import gc
import os
import psutil
import logging
from typing import Dict, Any, Optional, Tuple, Callable
from functools import wraps
import traceback
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class OOMHandler:
    """
    OOM 핸들링 시스템
    - GPU/CPU 메모리 모니터링
    - 배치 사이즈 자동 조정
    - Dynamic Swap Manager 활성화
    - Gradient Accumulation 조정
    """
    
    def __init__(self,
                 initial_batch_size: int = 4,
                 min_batch_size: int = 1,
                 gradient_accumulation: int = 16,
                 memory_threshold: float = 0.85,
                 enable_dsm: bool = True):
        """
        Args:
            initial_batch_size: 초기 배치 사이즈
            min_batch_size: 최소 배치 사이즈
            gradient_accumulation: Gradient Accumulation 스텝
            memory_threshold: 메모리 임계값 (0.85 = 85%)
            enable_dsm: Dynamic Swap Manager 사용 여부
        """
        self.initial_batch_size = initial_batch_size
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.gradient_accumulation = gradient_accumulation
        self.memory_threshold = memory_threshold
        self.enable_dsm = enable_dsm
        
        # 메모리 통계
        self.oom_count = 0
        self.batch_size_history = [initial_batch_size]
        self.memory_usage_history = []
        
        # DSM 상태
        self.dsm_active = False
        self.dsm_manager = None
        
        # GPU 사용 가능 여부 확인
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.has_gpu = torch.cuda.is_available()
        
        logger.info("✅ OOM Handler 초기화")
        logger.info(f"  - 초기 배치 사이즈: {initial_batch_size}")
        logger.info(f"  - 최소 배치 사이즈: {min_batch_size}")
        logger.info(f"  - Gradient Accumulation: {gradient_accumulation}")
        logger.info(f"  - 메모리 임계값: {memory_threshold * 100}%")
        logger.info(f"  - 디바이스: {self.device}")
    
    def check_memory_status(self) -> Dict[str, Any]:
        """
        현재 메모리 상태 확인
        
        Returns:
            메모리 상태 정보
        """
        status = {
            'timestamp': datetime.now().isoformat(),
            'cpu': {},
            'gpu': {}
        }
        
        # CPU 메모리
        cpu_mem = psutil.virtual_memory()
        status['cpu'] = {
            'total_gb': cpu_mem.total / (1024**3),
            'available_gb': cpu_mem.available / (1024**3),
            'used_gb': cpu_mem.used / (1024**3),
            'percent': cpu_mem.percent
        }
        
        # GPU 메모리
        if self.has_gpu:
            try:
                gpu_mem_allocated = torch.cuda.memory_allocated() / (1024**3)
                gpu_mem_reserved = torch.cuda.memory_reserved() / (1024**3)
                gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                status['gpu'] = {
                    'total_gb': gpu_mem_total,
                    'allocated_gb': gpu_mem_allocated,
                    'reserved_gb': gpu_mem_reserved,
                    'free_gb': gpu_mem_total - gpu_mem_reserved,
                    'percent': (gpu_mem_reserved / gpu_mem_total) * 100
                }
            except Exception as e:
                logger.warning(f"GPU 메모리 확인 실패: {e}")
                status['gpu'] = {'error': str(e)}
        
        return status
    
    def is_memory_critical(self) -> bool:
        """
        메모리가 임계 상태인지 확인
        
        Returns:
            True if memory is critical
        """
        status = self.check_memory_status()
        
        # CPU 메모리 체크
        if status['cpu']['percent'] > self.memory_threshold * 100:
            return True
        
        # GPU 메모리 체크
        if self.has_gpu and 'percent' in status['gpu']:
            if status['gpu']['percent'] > self.memory_threshold * 100:
                return True
        
        return False
    
    def handle_oom(self, exception: Exception) -> bool:
        """
        OOM 예외 처리
        
        Args:
            exception: 발생한 예외
            
        Returns:
            복구 성공 여부
        """
        self.oom_count += 1
        logger.warning(f"⚠️ OOM 발생! (#{self.oom_count})")
        logger.warning(f"   예외: {str(exception)}")
        
        # 메모리 정리
        self._clear_memory()
        
        # # 배치 사이즈 감소 (주석 처리: 배치 사이즈 2에서 더 이상 감소하지 않음)
        # if self.current_batch_size > self.min_batch_size:
        #     old_batch_size = self.current_batch_size
        #     self.current_batch_size = max(self.min_batch_size, self.current_batch_size // 2)
        #     self.batch_size_history.append(self.current_batch_size)
        #     
        #     # Gradient Accumulation 조정 (유효 배치 사이즈 유지)
        #     effective_batch = old_batch_size * self.gradient_accumulation
        #     self.gradient_accumulation = effective_batch // self.current_batch_size
        #     
        #     logger.info(f"  📉 배치 사이즈 조정: {old_batch_size} → {self.current_batch_size}")
        #     logger.info(f"  📊 Gradient Accumulation 조정: {self.gradient_accumulation}")
        #     
        #     return True
        
        # 배치 사이즈 감소 대신 바로 False 반환
        logger.warning("  ⚠️ OOM 발생: 배치 사이즈 2 유지 (폴백 비활성화)")
        return False
        
        # DSM 활성화 시도
        if self.enable_dsm and not self.dsm_active:
            if self._activate_dsm():
                return True
        
        # 복구 실패
        logger.error("  ❌ OOM 복구 실패: 최소 배치 사이즈 도달")
        return False
    
    def _clear_memory(self):
        """메모리 정리"""
        if self.has_gpu:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        gc.collect()
        
        logger.info("  🧹 메모리 정리 완료")
    
    def _activate_dsm(self) -> bool:
        """Dynamic Swap Manager 활성화"""
        try:
            # DSM 설정
            os.environ['DSM_SYNC_MODE'] = 'true'  # 이벤트 루프 없는 환경 대비
            
            # DSM import 시도
            try:
                from dynamic_swap_manager import get_swap_manager
                self.dsm_manager = get_swap_manager()
                self.dsm_active = True
                logger.info("  ✅ Dynamic Swap Manager 활성화")
                return True
            except ImportError:
                logger.warning("  ⚠️ DSM 모듈을 찾을 수 없습니다")
                return False
                
        except Exception as e:
            logger.error(f"  ❌ DSM 활성화 실패: {e}")
            return False
    
    def safe_forward_pass(self, func: Callable) -> Callable:
        """
        Forward pass를 OOM-safe하게 래핑하는 데코레이터
        
        Args:
            func: Forward pass 함수
            
        Returns:
            래핑된 함수
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    # 메모리 상태 체크
                    if self.is_memory_critical():
                        logger.warning("  ⚠️ 메모리 임계 상태 감지")
                        self._clear_memory()
                    
                    # Forward pass 실행
                    result = func(*args, **kwargs)
                    return result
                    
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower() or 'cuda' in str(e).lower():
                        retry_count += 1
                        
                        if self.handle_oom(e):
                            logger.info(f"  🔄 재시도 {retry_count}/{max_retries}")
                            # 배치 사이즈가 조정되었으므로 함수 인자 수정 필요
                            # 이 부분은 실제 구현에서 조정 필요
                            continue
                        else:
                            raise
                    else:
                        raise
                        
            raise RuntimeError(f"OOM 복구 실패: {max_retries}회 재시도 후에도 실패")
            
        return wrapper
    
    def adjust_dataloader(self, dataloader: Any) -> Any:
        """
        현재 배치 사이즈에 맞게 DataLoader 조정
        
        Args:
            dataloader: 원본 DataLoader
            
        Returns:
            조정된 DataLoader
        """
        if hasattr(dataloader, 'batch_size'):
            if dataloader.batch_size != self.current_batch_size:
                logger.info(f"  📊 DataLoader 배치 사이즈 조정: {dataloader.batch_size} → {self.current_batch_size}")
                
                # 새 DataLoader 생성
                from torch.utils.data import DataLoader
                new_dataloader = DataLoader(
                    dataset=dataloader.dataset,
                    batch_size=self.current_batch_size,
                    shuffle=dataloader.shuffle if hasattr(dataloader, 'shuffle') else False,
                    num_workers=dataloader.num_workers if hasattr(dataloader, 'num_workers') else 0,
                    pin_memory=dataloader.pin_memory if hasattr(dataloader, 'pin_memory') else False,
                    drop_last=dataloader.drop_last if hasattr(dataloader, 'drop_last') else False
                )
                return new_dataloader
        
        return dataloader
    
    def get_effective_batch_size(self) -> int:
        """
        유효 배치 사이즈 계산
        
        Returns:
            유효 배치 사이즈 (배치 사이즈 * gradient accumulation)
        """
        return self.current_batch_size * self.gradient_accumulation
    
    def log_memory_stats(self, step: int, phase: str = 'train'):
        """
        메모리 통계 로깅
        
        Args:
            step: 현재 스텝
            phase: 학습 단계 ('train', 'val', 'test')
        """
        status = self.check_memory_status()
        
        self.memory_usage_history.append({
            'step': step,
            'phase': phase,
            'batch_size': self.current_batch_size,
            'cpu_percent': status['cpu']['percent'],
            'gpu_percent': status['gpu'].get('percent', 0) if self.has_gpu else 0,
            'timestamp': status['timestamp']
        })
        
        # 주기적으로 상세 로그
        if step % 100 == 0:
            logger.info(f"📊 메모리 상태 (Step {step}):")
            logger.info(f"   - CPU: {status['cpu']['percent']:.1f}% ({status['cpu']['used_gb']:.1f}/{status['cpu']['total_gb']:.1f} GB)")
            if self.has_gpu and 'percent' in status['gpu']:
                logger.info(f"   - GPU: {status['gpu']['percent']:.1f}% ({status['gpu']['allocated_gb']:.1f}/{status['gpu']['total_gb']:.1f} GB)")
            logger.info(f"   - 배치 사이즈: {self.current_batch_size}")
            logger.info(f"   - OOM 발생 횟수: {self.oom_count}")
    
    def save_stats(self, output_dir: str = "training/oom_stats"):
        """
        OOM 통계 저장
        
        Args:
            output_dir: 출력 디렉토리
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        stats = {
            'initial_batch_size': self.initial_batch_size,
            'final_batch_size': self.current_batch_size,
            'min_batch_size': self.min_batch_size,
            'gradient_accumulation': self.gradient_accumulation,
            'oom_count': self.oom_count,
            'batch_size_history': self.batch_size_history,
            'memory_usage_history': self.memory_usage_history[-1000:],  # 최근 1000개만
            'dsm_active': self.dsm_active,
            'timestamp': timestamp
        }
        
        output_file = output_dir / f"oom_stats_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"📊 OOM 통계 저장: {output_file}")
        
        # 요약 리포트
        report_file = output_dir / f"oom_report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("OOM Handler Report\n")
            f.write(f"Generated: {timestamp}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"초기 배치 사이즈: {self.initial_batch_size}\n")
            f.write(f"최종 배치 사이즈: {self.current_batch_size}\n")
            f.write(f"OOM 발생 횟수: {self.oom_count}\n")
            f.write(f"DSM 활성화: {self.dsm_active}\n")
            f.write(f"유효 배치 사이즈: {self.get_effective_batch_size()}\n")
            
            if self.batch_size_history:
                f.write("\n배치 사이즈 변경 이력:\n")
                for i, bs in enumerate(self.batch_size_history):
                    f.write(f"  [{i}] {bs}\n")
        
        return str(output_file)
    
    def reset(self):
        """핸들러 상태 초기화"""
        self.current_batch_size = self.initial_batch_size
        self.batch_size_history = [self.initial_batch_size]
        self.oom_count = 0
        self.memory_usage_history = []
        self.dsm_active = False
        self._clear_memory()
        
        logger.info("🔄 OOM Handler 초기화 완료")