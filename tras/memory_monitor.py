#!/usr/bin/env python3
"""
실시간 메모리 모니터링 안전장치
Real-time Memory Monitoring Safety Guard
"""

import psutil
import torch
import time
import threading
import logging
from typing import Dict, Optional, Callable

class MemoryMonitor:
    """실시간 메모리 모니터링 및 안전장치"""
    
    def __init__(self, max_memory_gb: float = 70.0, warning_threshold: float = 0.9, critical_threshold: float = 0.95):
        self.max_memory_gb = max_memory_gb
        self.max_memory_bytes = max_memory_gb * 1024**3
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        
        self.monitoring = False
        self.monitor_thread = None
        self.callback_functions = []
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def add_callback(self, callback: Callable[[Dict], None]):
        """메모리 임계값 초과시 호출할 콜백 함수 추가"""
        self.callback_functions.append(callback)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """현재 메모리 사용량 반환 (GB 단위)"""
        # 시스템 RAM
        memory = psutil.virtual_memory()
        ram_used_gb = memory.used / 1024**3
        ram_total_gb = memory.total / 1024**3
        ram_percent = memory.percent
        
        # GPU 메모리 (가능한 경우)
        gpu_used_gb = 0.0
        gpu_total_gb = 0.0
        gpu_percent = 0.0
        
        if torch.cuda.is_available():
            gpu_used_bytes = torch.cuda.memory_allocated()
            gpu_total_bytes = torch.cuda.get_device_properties(0).total_memory
            gpu_used_gb = gpu_used_bytes / 1024**3
            gpu_total_gb = gpu_total_bytes / 1024**3
            gpu_percent = (gpu_used_gb / gpu_total_gb) * 100 if gpu_total_gb > 0 else 0
        
        return {
            'ram_used_gb': ram_used_gb,
            'ram_total_gb': ram_total_gb,
            'ram_percent': ram_percent,
            'gpu_used_gb': gpu_used_gb,
            'gpu_total_gb': gpu_total_gb,
            'gpu_percent': gpu_percent,
            'total_used_gb': ram_used_gb + gpu_used_gb,
            'budget_used_percent': (ram_used_gb / self.max_memory_gb) * 100
        }
    
    def check_memory_safety(self) -> Dict[str, any]:
        """메모리 안전성 검사"""
        usage = self.get_memory_usage()
        
        status = {
            'safe': True,
            'level': 'safe',
            'message': 'Memory usage within safe limits',
            'usage': usage
        }
        
        budget_usage = usage['budget_used_percent'] / 100
        
        if budget_usage >= self.critical_threshold:
            status.update({
                'safe': False,
                'level': 'critical',
                'message': f"CRITICAL: Memory usage {budget_usage:.1%} exceeds critical threshold {self.critical_threshold:.1%}"
            })
        elif budget_usage >= self.warning_threshold:
            status.update({
                'safe': True,
                'level': 'warning',
                'message': f"WARNING: Memory usage {budget_usage:.1%} exceeds warning threshold {self.warning_threshold:.1%}"
            })
        
        return status
    
    def _monitor_loop(self):
        """모니터링 루프"""
        while self.monitoring:
            try:
                status = self.check_memory_safety()
                
                if status['level'] in ['warning', 'critical']:
                    self.logger.warning(status['message'])
                    usage = status['usage']
                    self.logger.info(f"RAM: {usage['ram_used_gb']:.2f}/{usage['ram_total_gb']:.2f}GB ({usage['ram_percent']:.1f}%)")
                    
                    if torch.cuda.is_available():
                        self.logger.info(f"GPU: {usage['gpu_used_gb']:.2f}/{usage['gpu_total_gb']:.2f}GB ({usage['gpu_percent']:.1f}%)")
                    
                    # 콜백 함수들 호출
                    for callback in self.callback_functions:
                        try:
                            callback(status)
                        except Exception as e:
                            self.logger.error(f"Callback error: {e}")
                
                time.sleep(1)  # 1초마다 검사
                
            except Exception as e:
                self.logger.error(f"Monitor loop error: {e}")
                time.sleep(5)
    
    def start_monitoring(self):
        """모니터링 시작"""
        if self.monitoring:
            self.logger.warning("Monitoring already started")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info(f"🔍 Memory monitoring started (Budget: {self.max_memory_gb}GB, Warning: {self.warning_threshold:.1%}, Critical: {self.critical_threshold:.1%})")
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        
        self.logger.info("🔍 Memory monitoring stopped")
    
    def emergency_cleanup(self):
        """응급 메모리 정리"""
        self.logger.warning("🚨 Emergency memory cleanup initiated!")
        
        # PyTorch 캐시 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.info("✅ GPU cache cleared")
        
        # 가비지 컬렉션 강제 실행
        import gc
        gc.collect()
        self.logger.info("✅ Garbage collection completed")
        
        # 정리 후 상태 확인
        usage = self.get_memory_usage()
        self.logger.info(f"📊 After cleanup - RAM: {usage['ram_used_gb']:.2f}GB, Budget usage: {usage['budget_used_percent']:.1f}%")

def create_memory_guard(max_memory_gb: float = 70.0) -> MemoryMonitor:
    """메모리 가드 생성 및 기본 콜백 설정"""
    monitor = MemoryMonitor(max_memory_gb)
    
    def emergency_callback(status):
        """응급 상황 콜백"""
        if status['level'] == 'critical':
            monitor.emergency_cleanup()
            
            # 여전히 위험하면 경고
            new_status = monitor.check_memory_safety()
            if new_status['level'] == 'critical':
                monitor.logger.error("🆘 CRITICAL: Emergency cleanup failed to reduce memory usage!")
                monitor.logger.error("🛑 Consider stopping training immediately to prevent system crash!")
    
    monitor.add_callback(emergency_callback)
    return monitor

if __name__ == "__main__":
    # 테스트
    print("🧪 Memory Monitor Test")
    monitor = create_memory_guard(70.0)
    monitor.start_monitoring()
    
    try:
        # 5초간 모니터링 테스트
        time.sleep(5)
        status = monitor.check_memory_safety()
        print(f"📊 Current status: {status['level']} - {status['message']}")
        print(f"📈 Memory usage: {status['usage']}")
    finally:
        monitor.stop_monitoring()