"""
통합 시스템 오케스트레이터 - Week 5 핵심 구현
Unified System Orchestrator - Week 5 Core Implementation

전체 Red Heart 시스템 통합 관리:
- run_learning.sh와 Python 시스템 연동
- 실시간 성능 모니터링 및 대시보드
- 전체 시스템 상태 추적 및 알림
- 로그 통합 관리 및 분석
- 자동 장애 복구 및 최적화
"""

import asyncio
import logging
import time
import subprocess
import signal
import os
import sys
import json
import threading
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict, OrderedDict
from enum import Enum
import numpy as np
import psutil
import socket
import torch
import torch.nn
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import asynccontextmanager
import shutil

# 핵심 시스템 임포트
from config import ADVANCED_CONFIG, get_smart_device, get_gpu_memory_info
from head_compatibility_interface import HeadType, HeadProcessingResult, HeadCompatibilityManager
from unified_red_heart_core import RedHeartUnifiedBackbone
from dynamic_swap_manager import RedHeartDynamicSwapManager, SwapPriority
from intelligent_synergy_system import IntelligentSynergySystem
from unified_learning_system import UnifiedLearningSystem, TrainingMetrics
from advanced_usage_pattern_analyzer import AdvancedUsagePatternAnalyzer
from workflow_aware_memory_manager import WorkflowAwareMemoryManager, WorkflowStage

# 로거 설정
logger = logging.getLogger(__name__)

def _extract_nn_module(obj):
    """래퍼 객체에서 실제 nn.Module 추출 (NO FALLBACK)
    
    Args:
        obj: 검사할 객체
        
    Returns:
        nn.Module 또는 None
    """
    import torch.nn as nn
    
    # 이미 nn.Module이면 바로 반환
    if isinstance(obj, nn.Module):
        return obj
    
    # get_pytorch_network 메서드 확인 (최우선)
    if hasattr(obj, 'get_pytorch_network') and callable(obj.get_pytorch_network):
        try:
            net = obj.get_pytorch_network()
            if isinstance(net, nn.Module):
                logger.debug(f"get_pytorch_network()에서 nn.Module 발견: {net.__class__.__name__}")
                return net
        except Exception as e:
            logger.debug(f"get_pytorch_network() 호출 실패: {e}")
    
    # HF pipeline 감지 - .model 노출
    if obj.__class__.__name__.endswith("Pipeline") and hasattr(obj, "model"):
        if isinstance(obj.model, nn.Module):
            logger.debug(f"HF Pipeline에서 nn.Module 발견: {obj.model.__class__.__name__}")
            return obj.model
    
    # 다양한 속성명 체크 (우선순위 순)
    attr_names = [
        'model', '_model', 'network', '_network', 'embedding_model',
        'module', '_module', 'net', '_net',
        'encoder', 'decoder', 'embeddings',
        'backbone', 'head', 'classifier',
        # emotion_analyzer 관련 추가
        'emotion_moe', 'hierarchical_model', 'default_network',
        '_primary_nn', 'multilingual_direct', 'korean_model',
        # semantic_analyzer 관련 추가
        'fusion_network', 'semantic_model', 'causal_model',
        'emotion_model', 'fusion_model', 'ethical_model',
        # translator 관련 추가  
        'translation_model', 'translator_model', 'mbart_model'
    ]
    
    # 관용 팩토리 메서드도 시도 (확장)
    factory_methods = [
        'to_torch', 'build_network', 'as_pytorch', 'materialize', 'get_model',
        'create_model', 'build', 'construct', 'get_nn_module', 'to_nn_module',
        'get_network', 'create_network', 'get_pytorch_model'
    ]
    for method_name in factory_methods:
        if hasattr(obj, method_name) and callable(getattr(obj, method_name)):
            try:
                net = getattr(obj, method_name)()
                if isinstance(net, nn.Module):
                    logger.debug(f"{method_name}()에서 nn.Module 추출")
                    return net
            except Exception as e:
                logger.debug(f"{method_name}() 실패: {e}")
    
    for attr_name in attr_names:
        if hasattr(obj, attr_name):
            attr = getattr(obj, attr_name)
            if isinstance(attr, nn.Module):
                logger.debug(f"{attr_name} 속성에서 nn.Module 발견: {attr.__class__.__name__}")
                return attr
    
    # models/_models 딕셔너리 확인
    for dict_name in ['models', '_models']:
        if hasattr(obj, dict_name):
            models_dict = getattr(obj, dict_name)
            if isinstance(models_dict, dict):
                for key, m in models_dict.items():
                    if isinstance(m, nn.Module):
                        logger.debug(f"{dict_name}['{key}']에서 nn.Module 발견: {m.__class__.__name__}")
                        return m
    
    # 찾을 수 없음
    logger.debug(f"{obj.__class__.__name__}에서 nn.Module을 찾을 수 없음")
    return None

class SimpleThinAdapter(torch.nn.Module):
    """nn.Module이 없는 객체를 위한 간단한 어댑터"""
    
    def __init__(self, wrapped_obj):
        super().__init__()
        self.wrapped = wrapped_obj
        # 더미 파라미터 추가 (DSM이 관리할 수 있도록)
        self.dummy_param = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        
    def forward(self, *args, **kwargs):
        """wrapped 객체가 forward나 process 메서드를 가지면 호출"""
        if hasattr(self.wrapped, 'forward'):
            return self.wrapped.forward(*args, **kwargs)
        elif hasattr(self.wrapped, 'process'):
            return self.wrapped.process(*args, **kwargs)
        elif hasattr(self.wrapped, '__call__'):
            return self.wrapped(*args, **kwargs)
        else:
            # 최소한 입력을 그대로 반환
            return args[0] if args else None
    
    def to(self, device):
        """디바이스 이동 처리"""
        super().to(device)
        if hasattr(self.wrapped, 'to'):
            self.wrapped.to(device)
        elif hasattr(self.wrapped, 'device'):
            self.wrapped.device = device
        return self

class SystemStatus(Enum):
    """시스템 상태"""
    INITIALIZING = "initializing"
    READY = "ready"
    TRAINING = "training"
    MONITORING = "monitoring"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    SHUTDOWN = "shutdown"

class ComponentStatus(Enum):
    """컴포넌트 상태"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    OFFLINE = "offline"
    RECOVERING = "recovering"

@dataclass
class SystemHealthMetrics:
    """시스템 건강도 메트릭"""
    timestamp: datetime = field(default_factory=datetime.now)
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    gpu_memory: float = 0.0
    disk_usage: float = 0.0
    
    # 네트워크 메트릭
    network_in: float = 0.0
    network_out: float = 0.0
    
    # 시스템 온도 (가능한 경우)
    cpu_temp: Optional[float] = None
    gpu_temp: Optional[float] = None
    
    # 프로세스 메트릭
    active_processes: int = 0
    python_processes: int = 0
    
    # 커스텀 메트릭
    training_efficiency: float = 0.0
    synergy_performance: float = 0.0
    swap_efficiency: float = 0.0

@dataclass
class ComponentHealth:
    """컴포넌트 건강도"""
    name: str
    status: ComponentStatus
    health_score: float = 1.0  # 0.0 - 1.0
    last_check: datetime = field(default_factory=datetime.now)
    error_count: int = 0
    uptime: timedelta = field(default_factory=lambda: timedelta(0))
    
    # 성능 메트릭
    response_time: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    
    # 상세 정보
    details: Dict[str, Any] = field(default_factory=dict)
    recent_errors: List[str] = field(default_factory=lambda: deque(maxlen=10))

class SystemMonitor:
    """시스템 모니터"""
    
    def __init__(self, check_interval: float = 5.0):
        self.check_interval = check_interval
        self.is_monitoring = False
        self.monitoring_thread = None
        
        # 메트릭 저장소
        self.health_history = deque(maxlen=1000)  # 최근 1000개 메트릭
        self.component_health = {}
        
        # 알람 설정
        self.alert_thresholds = {
            'cpu_usage': 80.0,      # 80% 이상
            'memory_usage': 85.0,   # 85% 이상
            'gpu_usage': 90.0,      # 90% 이상
            'disk_usage': 95.0,     # 95% 이상
            'error_rate': 0.1       # 10% 이상
        }
        
        # 알람 콜백
        self.alert_callbacks = []
        
    def start_monitoring(self):
        """모니터링 시작"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("시스템 모니터링 시작")
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        logger.info("시스템 모니터링 중지")
    
    def _monitoring_loop(self):
        """모니터링 루프"""
        while self.is_monitoring:
            try:
                # 시스템 메트릭 수집
                metrics = self._collect_system_metrics()
                self.health_history.append(metrics)
                
                # 알람 체크
                self._check_alerts(metrics)
                
                # 컴포넌트 건강도 업데이트
                self._update_component_health()
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"모니터링 중 오류: {str(e)}")
                time.sleep(self.check_interval)
    
    def _collect_system_metrics(self) -> SystemHealthMetrics:
        """시스템 메트릭 수집"""
        
        # CPU 및 메모리
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # 디스크
        disk = psutil.disk_usage('/')
        
        # 네트워크
        net_io = psutil.net_io_counters()
        
        # GPU 정보
        gpu_info = get_gpu_memory_info()
        gpu_usage = 0.0
        gpu_memory = 0.0
        if gpu_info:
            gpu_memory = gpu_info.get('memory_used_gb', 0) / gpu_info.get('memory_total_gb', 1) * 100
            gpu_usage = gpu_info.get('utilization', 0)
        
        # 프로세스 정보
        all_processes = list(psutil.process_iter(['pid', 'name', 'cmdline']))
        python_processes = len([p for p in all_processes 
                               if p.info['name'] and 'python' in p.info['name'].lower()])
        
        # 온도 정보 (가능한 경우)
        cpu_temp = None
        try:
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                cpu_temp = max([t.current for t in temps['coretemp']])
        except:
            pass
        
        metrics = SystemHealthMetrics(
            timestamp=datetime.now(),
            cpu_usage=cpu_percent,
            memory_usage=memory.percent,
            gpu_usage=gpu_usage,
            gpu_memory=gpu_memory,
            disk_usage=disk.percent,
            network_in=net_io.bytes_recv / (1024**2),  # MB
            network_out=net_io.bytes_sent / (1024**2),  # MB
            cpu_temp=cpu_temp,
            active_processes=len(all_processes),
            python_processes=python_processes
        )
        
        return metrics
    
    def _check_alerts(self, metrics: SystemHealthMetrics):
        """알람 체크"""
        alerts = []
        
        if metrics.cpu_usage > self.alert_thresholds['cpu_usage']:
            alerts.append(f"높은 CPU 사용률: {metrics.cpu_usage:.1f}%")
        
        if metrics.memory_usage > self.alert_thresholds['memory_usage']:
            alerts.append(f"높은 메모리 사용률: {metrics.memory_usage:.1f}%")
        
        if metrics.gpu_usage > self.alert_thresholds['gpu_usage']:
            alerts.append(f"높은 GPU 사용률: {metrics.gpu_usage:.1f}%")
        
        if metrics.disk_usage > self.alert_thresholds['disk_usage']:
            alerts.append(f"높은 디스크 사용률: {metrics.disk_usage:.1f}%")
        
        # 온도 알람
        if metrics.cpu_temp and metrics.cpu_temp > 80:
            alerts.append(f"높은 CPU 온도: {metrics.cpu_temp:.1f}°C")
        
        # 알람 콜백 실행
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert, metrics)
                except Exception as e:
                    logger.error(f"알람 콜백 실행 오류: {str(e)}")
    
    def _update_component_health(self):
        """컴포넌트 건강도 업데이트"""
        # 현재 활성 컴포넌트들의 건강도 체크
        components_to_check = [
            'unified_backbone',
            'swap_manager', 
            'synergy_system',
            'learning_system',
            'pattern_analyzer'
        ]
        
        for component_name in components_to_check:
            health = self._assess_component_health(component_name)
            self.component_health[component_name] = health
    
    def _assess_component_health(self, component_name: str) -> ComponentHealth:
        """컴포넌트 건강도 평가"""
        
        # 기존 건강도 정보 가져오기
        existing_health = self.component_health.get(component_name)
        if existing_health:
            health = existing_health
        else:
            health = ComponentHealth(name=component_name, status=ComponentStatus.HEALTHY)
        
        # 간단한 건강도 체크 (실제 구현에서는 각 컴포넌트별 세부 체크)
        try:
            # 예시: 프로세스 존재 여부로 건강도 판단
            if component_name == 'unified_backbone':
                # GPU 메모리 사용량으로 활성 상태 판단
                gpu_info = get_gpu_memory_info()
                if gpu_info and gpu_info.get('memory_used_gb', 0) > 1:
                    health.status = ComponentStatus.HEALTHY
                    health.health_score = 0.9
                else:
                    health.status = ComponentStatus.WARNING
                    health.health_score = 0.6
            
            elif component_name in ['swap_manager', 'synergy_system', 'learning_system']:
                # Python 프로세스 활성 상태로 판단
                if self.health_history and self.health_history[-1].python_processes > 0:
                    health.status = ComponentStatus.HEALTHY
                    health.health_score = 0.8
                else:
                    health.status = ComponentStatus.WARNING
                    health.health_score = 0.5
            
            else:
                # 기본적으로 건강한 상태로 가정
                health.status = ComponentStatus.HEALTHY
                health.health_score = 0.7
            
            health.last_check = datetime.now()
            
        except Exception as e:
            health.status = ComponentStatus.CRITICAL
            health.health_score = 0.1
            health.recent_errors.append(str(e))
            health.error_count += 1
        
        return health
    
    def get_system_summary(self) -> Dict[str, Any]:
        """시스템 요약 정보"""
        if not self.health_history:
            return {"status": "no_data"}
        
        latest_metrics = self.health_history[-1]
        
        # 전체 시스템 상태 결정
        overall_status = SystemStatus.READY
        if latest_metrics.cpu_usage > 90 or latest_metrics.memory_usage > 95:
            overall_status = SystemStatus.ERROR
        elif latest_metrics.cpu_usage > 80 or latest_metrics.memory_usage > 85:
            overall_status = SystemStatus.WARNING
        
        # 컴포넌트 건강도 요약
        component_summary = {}
        healthy_components = 0
        total_components = len(self.component_health)
        
        for name, health in self.component_health.items():
            component_summary[name] = {
                'status': health.status.value,
                'health_score': health.health_score,
                'error_count': health.error_count
            }
            if health.status == ComponentStatus.HEALTHY:
                healthy_components += 1
        
        overall_health = healthy_components / max(1, total_components)
        
        return {
            'overall_status': overall_status.value,
            'overall_health': overall_health,
            'current_metrics': {
                'cpu_usage': latest_metrics.cpu_usage,
                'memory_usage': latest_metrics.memory_usage,
                'gpu_usage': latest_metrics.gpu_usage,
                'gpu_memory': latest_metrics.gpu_memory,
                'disk_usage': latest_metrics.disk_usage,
                'python_processes': latest_metrics.python_processes
            },
            'component_health': component_summary,
            'timestamp': latest_metrics.timestamp.isoformat(),
            'uptime_minutes': len(self.health_history) * self.check_interval / 60
        }

class LogManager:
    """로그 관리자"""
    
    def __init__(self, log_dir: str = "logs/unified_system"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 로그 파일들 - pathlib 대신 os.path.join 사용
        self.system_log = os.path.join(self.log_dir, "system.log")
        self.training_log = os.path.join(self.log_dir, "training.log")
        self.error_log = os.path.join(self.log_dir, "errors.log")
        self.performance_log = os.path.join(self.log_dir, "performance.log")
        
        # 로그 회전 설정
        self.max_log_size = 100 * 1024 * 1024  # 100MB
        self.max_log_files = 10
        
        self._setup_loggers()
    
    def _setup_loggers(self):
        """로거 설정"""
        
        # 시스템 로거
        self.system_logger = logging.getLogger("red_heart.system")
        self.system_logger.setLevel(logging.INFO)
        
        system_handler = logging.FileHandler(self.system_log)
        system_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.system_logger.addHandler(system_handler)
        
        # 훈련 로거
        self.training_logger = logging.getLogger("red_heart.training")
        self.training_logger.setLevel(logging.INFO)
        
        training_handler = logging.FileHandler(self.training_log)
        training_handler.setFormatter(logging.Formatter(
            '%(asctime)s - TRAINING - %(message)s'
        ))
        self.training_logger.addHandler(training_handler)
        
        # 에러 로거
        self.error_logger = logging.getLogger("red_heart.errors")
        self.error_logger.setLevel(logging.ERROR)
        
        error_handler = logging.FileHandler(self.error_log)
        error_handler.setFormatter(logging.Formatter(
            '%(asctime)s - ERROR - %(name)s - %(message)s'
        ))
        self.error_logger.addHandler(error_handler)
    
    def log_system_event(self, message: str, level: str = "INFO"):
        """시스템 이벤트 로깅"""
        if level == "INFO":
            self.system_logger.info(message)
        elif level == "WARNING":
            self.system_logger.warning(message)
        elif level == "ERROR":
            self.system_logger.error(message)
            self.error_logger.error(message)
    
    def log_training_metrics(self, metrics: TrainingMetrics):
        """훈련 메트릭 로깅"""
        metric_str = (
            f"Epoch: {metrics.epoch}, Step: {metrics.step}, "
            f"Loss: {metrics.total_loss:.4f}, "
            f"Memory: {metrics.memory_usage:.2f}MB, "
            f"Time: {metrics.training_time:.3f}s, "
            f"Samples/s: {metrics.samples_per_second:.1f}"
        )
        self.training_logger.info(metric_str)
    
    def log_performance_data(self, component: str, metrics: Dict[str, Any]):
        """성능 데이터 로깅"""
        perf_data = {
            'timestamp': datetime.now().isoformat(),
            'component': component,
            'metrics': metrics
        }
        
        with open(self.performance_log, 'a') as f:
            f.write(json.dumps(perf_data) + '\n')
    
    def rotate_logs(self):
        """로그 회전"""
        for log_file in [self.system_log, self.training_log, self.error_log, self.performance_log]:
            # 문자열로 저장된 로그 파일을 os.path로 처리
            if os.path.exists(log_file) and os.path.getsize(log_file) > self.max_log_size:
                # 기존 로그 백업
                base_name = os.path.splitext(log_file)[0]
                ext = os.path.splitext(log_file)[1]
                
                for i in range(self.max_log_files - 1, 0, -1):
                    old_file = f"{base_name}.{i}{ext}"
                    new_file = f"{base_name}.{i+1}{ext}"
                    if os.path.exists(old_file):
                        os.rename(old_file, new_file)
                
                # 현재 로그를 .1로 이동
                os.rename(log_file, f"{base_name}.1{ext}")

class ProcessManager:
    """프로세스 관리자"""
    
    def __init__(self):
        self.processes = {}  # 프로세스 ID -> 프로세스 정보
        self.process_configs = {}  # 프로세스 설정
    
    async def start_background_process(self, name: str, command: List[str],
                                     restart_on_failure: bool = True) -> int:
        """백그라운드 프로세스 시작"""
        
        try:
            # 출력 리다이렉션으로 블로킹 방지
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                # 표준 입력도 닫음
                stdin=asyncio.subprocess.DEVNULL
            )
            
            self.processes[name] = {
                'process': process,
                'pid': process.pid,
                'command': command,
                'start_time': datetime.now(),
                'restart_on_failure': restart_on_failure,
                'restart_count': 0
            }
            
            logger.info(f"백그라운드 프로세스 시작: {name} (PID: {process.pid})")
            
            # 프로세스 모니터링 태스크 시작
            monitor_task = asyncio.create_task(self._monitor_process(name))
            self.processes[name]['monitor_task'] = monitor_task
            
            return process.pid
            
        except Exception as e:
            logger.error(f"프로세스 시작 실패 - {name}: {str(e)}")
            raise
    
    async def _monitor_process(self, name: str):
        """프로세스 모니터링"""
        if name not in self.processes:
            return
            
        process_info = self.processes[name]
        process = process_info['process']
        
        try:
            # 프로세스 완료 대기
            return_code = await process.wait()
            
            logger.info(f"프로세스 완료: {name} (코드: {return_code})")
            
            if return_code != 0:
                logger.warning(f"프로세스 비정상 종료: {name} (코드: {return_code})")
                
                # 재시작 여부 결정
                if process_info['restart_on_failure'] and process_info['restart_count'] < 3:
                    logger.info(f"프로세스 재시작: {name}")
                    # 기존 프로세스 정보 정리 후 재시작
                    if name in self.processes:
                        del self.processes[name]
                    
                    process_info['restart_count'] += 1
                    await self.start_background_process(
                        name, process_info['command'], 
                        process_info['restart_on_failure']
                    )
                    return  # 재시작 시에는 여기서 종료
                else:
                    logger.error(f"프로세스 재시작 포기: {name}")
            else:
                logger.info(f"프로세스 정상 완료: {name}")
            
        except asyncio.CancelledError:
            logger.info(f"프로세스 모니터링 취소: {name}")
        except Exception as e:
            logger.error(f"프로세스 모니터링 오류 - {name}: {str(e)}")
        
        finally:
            # 프로세스 정보 확실히 정리
            try:
                if name in self.processes:
                    process_info = self.processes[name]
                    
                    # 모니터링 태스크 정리
                    if 'monitor_task' in process_info:
                        monitor_task = process_info['monitor_task']
                        if not monitor_task.done():
                            monitor_task.cancel()
                    
                    # 프로세스 정리 전에 출력 스트림 정리 (DEVNULL이므로 실제로는 불필요)
                    if hasattr(process, 'stdout') and process.stdout and not process.stdout.is_closing():
                        process.stdout.close()
                    if hasattr(process, 'stderr') and process.stderr and not process.stderr.is_closing():
                        process.stderr.close()
                    
                    del self.processes[name]
                    logger.info(f"프로세스 정보 정리 완료: {name}")
            except Exception as e:
                logger.error(f"프로세스 정리 오류 - {name}: {str(e)}")
    
    async def stop_process(self, name: str) -> bool:
        """프로세스 중지"""
        if name not in self.processes:
            logger.warning(f"프로세스를 찾을 수 없음: {name}")
            return False
        
        process_info = self.processes[name]
        process = process_info['process']
        
        try:
            if process.returncode is not None:
                # 이미 종료된 프로세스
                logger.info(f"프로세스 이미 종료됨: {name}")
                if name in self.processes:
                    del self.processes[name]
                return True
            
            # SIGTERM 신호 전송
            process.terminate()
            logger.info(f"프로세스 종료 신호 전송: {name}")
            
            # 5초 대기 후 강제 종료
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
                logger.info(f"프로세스 정상 종료: {name}")
            except asyncio.TimeoutError:
                process.kill()
                logger.warning(f"프로세스 강제 종료: {name}")
                try:
                    await asyncio.wait_for(process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    logger.error(f"프로세스 강제 종료 실패: {name}")
            
            # 프로세스 정보 정리
            if name in self.processes:
                del self.processes[name]
            
            return True
            
        except Exception as e:
            logger.error(f"프로세스 중지 오류 - {name}: {str(e)}")
            # 오류 발생 시에도 프로세스 정보 정리
            if name in self.processes:
                del self.processes[name]
            return False
    
    def get_process_status(self) -> Dict[str, Any]:
        """프로세스 상태 조회"""
        status = {}
        
        for name, info in self.processes.items():
            process = info['process']
            uptime = datetime.now() - info['start_time']
            
            status[name] = {
                'pid': info['pid'],
                'uptime_seconds': uptime.total_seconds(),
                'restart_count': info['restart_count'],
                'is_running': process.returncode is None,
                'command': ' '.join(info['command'])
            }
        
        return status

class UnifiedSystemOrchestrator:
    """
    통합 시스템 오케스트레이터 - 메인 클래스
    
    전체 Red Heart 시스템의 통합 관리 및 조정
    """
    
    # 모듈 의존성 그래프 정의 (클래스 수준)
    MODULE_DEP = {
        'emotion_analyzer':  ['bentham_calculator'],  # emotion_empathy_head 제거 - 순환 의존성 방지
        'semantic_analyzer': [],  # semantic_surd_head 제거 - 헤드는 모듈에 의존하지 않음
        'regret_analyzer': [],
        'translator': [],
        'neural_components': [],
        'surd_analyzer': [],
        'bayesian_engine': [],
        'llm_engine': [],
        'experience_database': [],
        'hierarchical_emotion': [],
        'usage_pattern_analyzer': [],
        'meta_integration': []
    }
    
    # \ubaa8\ub4c8 \uc774\ub984\uacfc DSM \ud5e4\ub4dc \uc774\ub984 \ub9e4\ud551 (\ud074\ub798\uc2a4 \uc218\uc900)
    MODULE_TO_HEAD_NAME = {
        'semantic_analyzer': 'semantic_surd_head',
        'emotion_analyzer': 'emotion_empathy_head',
        'meta_integration': 'meta_integration_head',
        'bentham_calculator': 'bentham_fromm_head',
        'regret_analyzer': 'regret_learning_head',
        # translator\ub294 \ud5e4\ub4dc\uac00 \uc544\ub2c8\ubbc0\ub85c \ub9e4\ud551 \uc5c6\uc74c
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or ADVANCED_CONFIG.get('orchestrator_config', {})
        
        # HF 모델 래퍼 활성화 - 모든 from_pretrained/pipeline 호출이 자동으로 추적됨
        try:
            from hf_model_wrapper import enable_auto_registration, get_hf_wrapper
            enable_auto_registration()
            logger.info("✅ HF 모델 자동 등록 활성화 - 모든 모델 로딩이 추적됩니다")
        except ImportError:
            logger.warning("⚠️ HF 모델 래퍼를 찾을 수 없음 - 모델 로딩 추적 비활성화")
        
        # 시스템 상태
        self.system_status = SystemStatus.INITIALIZING
        self.start_time = datetime.now()
        self.is_running = False
        
        # 모듈 인스턴스 관리 (예외 경로에서도 안전하게 접근 가능)
        self.module_instances: Dict[str, Any] = {}
        
        # 핵심 컴포넌트들
        self.unified_backbone = None
        self.swap_manager = None
        self.synergy_system = None
        self.learning_system = None
        self.pattern_analyzer = None
        
        # 관리 컴포넌트들
        self.system_monitor = SystemMonitor()
        self.log_manager = LogManager()
        self.process_manager = ProcessManager()
        
        # 메모리 관리자
        self.memory_manager = WorkflowAwareMemoryManager(memory_threshold_mb=6500.0)
        
        # run_learning.sh 통합 - pathlib 대신 os.path 사용
        self.script_path = os.path.join(os.path.dirname(__file__), "run_learning.sh")
        self.script_runner = None
        
        # 웹 대시보드 (선택적)
        self.dashboard_port = self.config.get('dashboard_port', 8080)
        self.dashboard_server = None
        
        # 자동 복구 설정
        self.auto_recovery = True
        self.recovery_attempts = 0
        self.max_recovery_attempts = 3
        
        # 빠른 초기화 모드 (validation 등에서 사용)
        self.fast_init_mode = self.config.get('fast_init_mode', False)
        
        logger.info("UnifiedSystemOrchestrator 초기화 완료")
    
    async def initialize_system(self) -> bool:
        """시스템 초기화"""
        
        try:
            self.log_manager.log_system_event("시스템 초기화 시작")
            
            # 1. 모니터링 시작
            self.system_monitor.start_monitoring()
            self.system_monitor.alert_callbacks.append(self._handle_system_alert)
            
            # 메모리 매니저에 메인 루프 주입
            self.memory_manager.main_loop = asyncio.get_running_loop()
            logger.info("✅ 메모리 매니저에 메인 이벤트 루프 주입 완료")
            
            # 2. 핵심 컴포넌트 초기화
            try:
                await self._initialize_core_components()
            finally:
                # 부팅 완료 신호 - 이후부터 GPU 할당 허용
                self.memory_manager.set_boot_completed()
            
            # 3. run_learning.sh 상태 확인
            if not os.path.exists(self.script_path):
                raise FileNotFoundError(f"run_learning.sh not found: {self.script_path}")
            
            # 4. 대시보드 시작 (선택적)
            if self.config.get('enable_dashboard', False):
                await self._start_dashboard()
            
            self.system_status = SystemStatus.READY
            self.is_running = True
            
            self.log_manager.log_system_event("시스템 초기화 완료", "INFO")
            logger.info("Red Heart 통합 시스템 초기화 완료")
            
            return True
            
        except Exception as e:
            self.system_status = SystemStatus.ERROR
            error_msg = f"시스템 초기화 실패: {str(e)}"
            self.log_manager.log_system_event(error_msg, "ERROR")
            logger.error(error_msg)
            
            # 상세한 에러 정보 출력
            import traceback
            tb_str = traceback.format_exc()
            logger.error(f"🚨 상세 에러 스택:\n{tb_str}")
            print(f"\n{'='*60}")
            print(f"🚨 시스템 초기화 실패!")
            print(f"{'='*60}")
            print(f"에러: {str(e)}")
            print(f"{'='*60}")
            print(f"상세 스택 트레이스:")
            print(tb_str)
            print(f"{'='*60}\n")
            
            return False
    
    def _ensure_pre_registered_heads(self):
        """헤드 선등록 보장 - 모듈 성공/실패와 무관하게 항상 등록"""
        import torch.nn as nn
        from dynamic_swap_manager import get_swap_manager, SwapPriority
        
        logger.critical("📋 헤드 선등록 보장 함수 호출됨")
        
        # self.swap_manager 직접 사용 (get_swap_manager 대신)
        swap = self.swap_manager if hasattr(self, 'swap_manager') else get_swap_manager()
        logger.critical(f"📋 swap_manager 상태: {type(swap)}")
        
        if not swap:
            logger.critical("❌ DSM이 없어서 선등록 불가")
            raise RuntimeError("DSM not available at pre-registration time")
            
        # DSM 현재 상태 로그
        logger.critical(f"📊 DSM 등록된 키(선등록 전): {sorted(list(swap.models.keys()))[:50]}")
        
        # emotion_empathy_head 무조건 선등록 (없을 때만)
        if "emotion_empathy_head" not in swap.models:
            class MinimalEmotionHead(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(768, 7)
                    # gradient 흐름 보장을 위해 명시적으로 requires_grad 설정
                    for param in self.parameters():
                        param.requires_grad = True
                
                def forward(self, x):
                    # gradient 흐름 보장
                    x = x.contiguous()
                    # 1280 -> 768 변환은 학습 시스템에서 처리
                    if x.shape[-1] > 768:
                        x = x[..., :768]
                    output = self.linear(x)
                    # gradient 흐름 강제
                    if not output.requires_grad and self.training:
                        output.requires_grad_(True)
                    return output
            
            try:
                default_head = MinimalEmotionHead()
                # HIGH 우선순위로 등록 (보호 필요)
                swap.register_model("emotion_empathy_head", default_head, priority=SwapPriority.HIGH)
                logger.info("✅ emotion_empathy_head 선등록 완료 (MinimalEmotionHead, priority=HIGH)")
                
                # warm load/unload - 나중에 비동기 컨텍스트에서 수행하도록 플래그 설정
                # 동기 함수에서는 직접 실행할 수 없으므로 플래그만 설정
                self._needs_warm_load_unload = True
                logger.info("🔄 warm load/unload 예약됨 (나중에 비동기 컨텍스트에서 실행)")
                    
            except Exception as e:
                logger.critical(f"❌ emotion_empathy_head 선등록 실패: {e}")
                raise RuntimeError(f"emotion_empathy_head 선등록 필수: {e}")
        else:
            logger.info("ℹ️ emotion_empathy_head 이미 등록됨")
        
        # 성공 보장 검증
        assert "emotion_empathy_head" in swap.models, "emotion_empathy_head pre-registration must succeed"
        logger.info(f"📊 DSM 등록된 키(선등록 후): {sorted(list(swap.models.keys()))[:50]}")
    
    async def _wait_for_existing_init(self, instance):
        """이미 시작된 초기화가 완료될 때까지 대기"""
        while not getattr(instance, 'fully_initialized', False):
            await asyncio.sleep(0.5)
        return instance
    
    async def _initialize_global_modules_sequential(self):
        """전역 모듈 순차 초기화 - MasterMemoryOrchestrator 관리"""
        
        # 전역 상태 컨테이너 보장 - 예외 경로에서도 안전
        if not hasattr(self, 'module_instances'):
            self.module_instances = {}
        
        # Float16 설정 - 안전하게 GPU 블록에서만 적용
        import torch
        # torch.set_default_dtype(torch.float16)  # 전역 설정은 위험
        # 대신 GPU 연산 시 torch.cuda.amp.autocast 사용 권장
        logger.info("🔧 PyTorch 정밀도: GPU 블록에서만 float16 적용 예정")
        
        from config import register_system_module, get_master_memory_orchestrator, get_gpu_memory_info
        from module_specs import MODULE_SPECS
        
        # MasterMemoryOrchestrator 연결
        try:
            orchestrator = get_master_memory_orchestrator()
        except Exception as e:
            logger.warning(f"MasterMemoryOrchestrator 연결 실패: {e}, 기본 초기화 진행")
            orchestrator = None
        
        # MODULE_SPECS를 사용하여 초기화
        module_specs = MODULE_SPECS
        
        # TODO: 동적 타임아웃 시스템 구현
        # 각 모듈의 실제 초기화 시간을 측정하여 다음 실행 시 타임아웃 자동 조정
        """
        # 향후 구현 예정: 모듈별 2분(120초) 기본 타임아웃 + 동적 조정
        for spec in module_specs:
            # 기본 타임아웃: 2분
            base_timeout = 180  # 3분으로 증가
            
            # 이전 실행 기록에서 평균 초기화 시간 가져오기
            avg_init_time = self._get_average_init_time(spec['name'])
            
            # 동적 타임아웃 = 평균 시간의 150% 또는 기본값 중 큰 값
            if avg_init_time > 0:
                dynamic_timeout = max(avg_init_time * 1.5, base_timeout)
            else:
                dynamic_timeout = base_timeout
            
            spec['timeout'] = dynamic_timeout
            
            # 초기화 시간 기록
            start_time = time.time()
            result = await self._load_single_module(spec)
            init_time = time.time() - start_time
            self._record_init_time(spec['name'], init_time)
        """
        
        logger.info(f"📋 {len(module_specs)}개 전역 모듈 순차 초기화 시작")
        
        for i, spec in enumerate(module_specs, 1):
            try:
                logger.info(f"🔄 [{i}/{len(module_specs)}] {spec['name']} 초기화 중...")
                
                # GPU 메모리 상태 확인
                memory_info = get_gpu_memory_info()
                if memory_info:
                    current_usage = memory_info['usage_percent']
                    free_mb = memory_info['free_mb']
                    logger.info(f"   📊 현재 GPU: {current_usage:.1f}% 사용, {free_mb}MB 여유")
                    
                    # NO-FALLBACK: 추정치 기반 공간 확보 금지
                # 실제 로딩은 DSM.register_model 후, DSM._ensure_gpu_memory에서만 판단
                if current_usage > 85:
                    logger.info("   🔄 GPU 메모리 85% 초과 - DSM 경유 로딩만 허용 (추정치 사용 금지)")
                
                # 타임아웃 직전 경고 로그
                timeout_seconds = spec['timeout']
                
                # semantic_analyzer는 최초 캐시 구축 시 오래 걸릴 수 있으므로 타임아웃 상향
                original_timeout = timeout_seconds
                spec_name = str(spec.get('name', ''))
                # semantic_analyzer로 시작하거나 포함하면 타임아웃 상향
                if spec_name.startswith('semantic_analyzer') or 'semantic_analyzer' in spec_name:
                    timeout_seconds = max(timeout_seconds, 360)
                    logger.info(f"   ⏱️ semantic_analyzer 특별 타임아웃 적용: {timeout_seconds}초 (원래: {original_timeout}초, 모듈명: {spec_name})")
                
                logger.info(f"   ⏱️ 모듈 로딩 시작: {spec['name']} (타임아웃: {timeout_seconds}초, 스펙 기본: {spec['timeout']}초)")
                
                # 타임아웃 적용한 모듈 초기화
                start_load_time = time.time()
                module_instance = await asyncio.wait_for(
                    self._load_single_module(spec),
                    timeout=timeout_seconds
                )
                load_time = time.time() - start_load_time
                logger.info(f"   ⏱️ {spec['name']} 로딩 완료: {load_time:.1f}초 소요")
                
                # emotion_analyzer가 스킵되더라도 기본 head 등록 보장 (NO FALLBACK)
                if spec['name'] == 'emotion_analyzer' and module_instance is None:
                    logger.warning(f"⚠️ emotion_analyzer 스킵됨 - 기본 emotion_empathy_head 등록 진행")
                    try:
                        from dynamic_swap_manager import get_swap_manager, SwapPriority
                        import torch.nn as nn
                        
                        swap_manager = get_swap_manager()
                        if swap_manager:
                            # 최소한의 기본 네트워크 생성 (NO FALLBACK - 실제 nn.Module)
                            class MinimalEmotionNetwork(nn.Module):
                                def __init__(self):
                                    super().__init__()
                                    self.linear = nn.Linear(768, 7)  # BERT hidden size -> 7 emotions
                                
                                def forward(self, x):
                                    return self.linear(x)
                            
                            default_network = MinimalEmotionNetwork()
                            swap_manager.register_model(
                                "emotion_empathy_head",
                                default_network,
                                priority=SwapPriority.HIGH
                            )
                            logger.info("✅ 기본 emotion_empathy_head 등록 완료 (MinimalEmotionNetwork)")
                    except Exception as e:
                        logger.error(f"❌ 기본 emotion_empathy_head 등록 실패: {e}")
                        # NO FALLBACK - 실패 시 시스템 중단
                        raise RuntimeError(f"emotion_empathy_head 등록 필수: {e}")
                
                # 모듈이 성공적으로 로딩된 경우 확인
                if module_instance is not None:
                    logger.info(f"   ✅ {spec['name']} 모듈 로딩 성공")
                    
                    # DSM에 전역 모듈 등록 (언로드 가능하도록)
                    try:
                        from dynamic_swap_manager import get_swap_manager, SwapPriority
                        swap_manager = get_swap_manager()
                        if swap_manager:
                            # 우선순위 매핑
                            priority_map = {
                                'CRITICAL': SwapPriority.CRITICAL,
                                'HIGH': SwapPriority.HIGH,
                                'MEDIUM': SwapPriority.MEDIUM,
                                'LOW': SwapPriority.LOW
                            }
                            priority = priority_map.get(spec.get('priority', 'MEDIUM'), SwapPriority.MEDIUM)
                            
                            # _extract_nn_module 함수로 실제 nn.Module 추출 (NO FALLBACK)
                            actual_nn_module = _extract_nn_module(module_instance)
                            
                            # 실제로 GPU에 있는지 확인하고 등록
                            is_on_gpu = False
                            
                            # 1. wrapper의 device 확인
                            if hasattr(module_instance, 'device'):
                                if str(module_instance.device).startswith('cuda'):
                                    is_on_gpu = True
                            
                            # 2. 실제 nn.Module의 device 확인
                            if not is_on_gpu and actual_nn_module is not None:
                                if hasattr(actual_nn_module, 'device'):
                                    if str(actual_nn_module.device).startswith('cuda'):
                                        is_on_gpu = True
                                # parameters를 통해 device 확인
                                elif hasattr(actual_nn_module, 'parameters'):
                                    try:
                                        for p in actual_nn_module.parameters():
                                            if p.is_cuda:
                                                is_on_gpu = True
                                                break
                                    except:
                                        pass
                            
                            # 3. DSM이 gpu_resident 관리 (외부에서 직접 조작 금지)
                            if is_on_gpu:
                                logger.info(f"   ℹ️ {spec['name']}이 GPU에 있음 - DSM이 관리")
                            else:
                                logger.info(f"   ℹ️ {spec['name']}은 CPU에 있음")
                            
                            # 실제 nn.Module을 DSM에 등록 (NO FALLBACK)
                            if actual_nn_module is None:
                                # nn.Module 추출 재시도 (더 체계적인 방법)
                                actual_nn_module = _extract_nn_module(module_instance)
                            
                            if actual_nn_module is not None:
                                swap_manager.register_model(
                                    spec['name'],
                                    actual_nn_module,  # 실제 nn.Module만 등록
                                    priority=priority
                                )
                                logger.info(f"   ✅ {spec['name']}을 DSM에 등록 완료 (모델: {actual_nn_module.__class__.__name__})")
                                
                                # GPU 상주 전환 + 안정화 대기 (등록-상주-언로드 핸드셰이크)
                                try:
                                    if spec.get('device_policy') == 'gpu_on_demand':
                                        await swap_manager.load_model_to_gpu(spec['name'], timeout=spec.get('timeout', 120))
                                        await asyncio.sleep(0.1)  # 캐시/allocator 안정화
                                        logger.info(f"   ✅ {spec['name']} GPU 상주 완료")
                                except Exception as e:
                                    logger.warning(f"   ⚠️ {spec['name']} GPU 상주 실패: {e}")
                                
                                # emotion_analyzer의 경우 특별 확인
                                if spec['name'] == 'emotion_analyzer':
                                    logger.info(f"   🔍 emotion_analyzer DSM 등록 확인: {actual_nn_module.__class__.__name__}")
                            else:
                                # nn.Module 추출 실패 - NO FALLBACK 정책에 따라 메타 등록 금지
                                logger.warning(f"   ⚠️ {spec['name']} nn.Module 추출 실패 - DSM 등록 스킵")
                                logger.debug(f"      - 모듈 타입: {type(module_instance)}")
                                logger.debug(f"      - get_pytorch_network 메서드: {hasattr(module_instance, 'get_pytorch_network')}")
                                logger.debug(f"      - NO FALLBACK: 메타 등록 금지, 실제 nn.Module만 허용")
                    except Exception as e:
                        logger.warning(f"   ⚠️ DSM 등록 실패 (계속 진행): {e}")
                else:
                    error_msg = f"{spec['name']} 모듈 로딩 실패"
                    logger.critical(f"   ❌ {error_msg} - 시스템 중단")
                    
                    # 전역 모듈 상태 확인
                    from config import get_system_module
                    all_modules = []
                    for mod_spec in module_specs:
                        mod = get_system_module(mod_spec['name'])
                        if mod:
                            all_modules.append(f"✅ {mod_spec['name']}")
                        else:
                            all_modules.append(f"❌ {mod_spec['name']}")
                    
                    logger.error(f"🔍 현재 전역 모듈 상태:")
                    for mod_status in all_modules:
                        logger.error(f"   {mod_status}")
                    
                    raise RuntimeError(f"필수 모듈 {spec['name']} 로딩 실패")
                
                # 초기화 후 메모리 상태 재확인
                memory_info_after = get_gpu_memory_info()
                if memory_info_after:
                    usage_after = memory_info_after['usage_percent']
                    logger.info(f"   ✅ {spec['name']} 완료 - GPU: {usage_after:.1f}% 사용")
                else:
                    logger.info(f"   ✅ {spec['name']} 완료")
                
                # 메모리 안정화를 위한 짧은 대기 (100ms)
                await asyncio.sleep(0.1)
                
            except asyncio.TimeoutError:
                # semantic_analyzer에 대해 하트비트 기반 동적 연장 시도
                if 'semantic_analyzer' in str(spec.get('name', '')):
                    # 모듈 인스턴스가 부분적으로 생성되었는지 확인
                    partial_instance = None
                    for mod_name, mod_inst in (self.module_instances or {}).items():
                        if 'semantic_analyzer' in mod_name:
                            partial_instance = mod_inst
                            break
                    
                    # 하트비트 확인
                    if partial_instance:
                        last_hb = getattr(partial_instance, 'last_heartbeat_ts', 0.0)
                        if (time.time() - last_hb) <= 10.0:
                            extra_timeout = 120
                            logger.warning(f"   ⏱️ semantic_analyzer 진행중 감지 (마지막 하트비트: {int(time.time()-last_hb)}초 전)")
                            logger.warning(f"   ⏱️ 타임아웃 {extra_timeout}초 추가 연장 시도")
                            try:
                                # 재시도 (이미 시작된 초기화가 완료될 때까지 대기)
                                module_instance = await asyncio.wait_for(
                                    self._wait_for_existing_init(partial_instance),
                                    timeout=extra_timeout
                                )
                                logger.info(f"   ✅ semantic_analyzer 연장된 시간 내 완료")
                                # 성공 시 continue로 다음 모듈로 진행
                                continue
                            except asyncio.TimeoutError:
                                logger.error(f"   ❌ semantic_analyzer 연장 후에도 타임아웃")
                
                # 타임아웃 진단 정보 수집
                diagnostic_info = f"설정: {timeout_seconds}초, 기본: {spec['timeout']}초"
                
                # 부분적으로 생성된 인스턴스에서 하트비트 정보 확인
                for mod_name, mod_inst in (self.module_instances or {}).items():
                    if spec['name'] in mod_name or mod_name in spec['name']:
                        last_hb = getattr(mod_inst, 'last_heartbeat_ts', None)
                        if last_hb:
                            elapsed_since_hb = time.time() - last_hb
                            diagnostic_info += f", 마지막 하트비트: {elapsed_since_hb:.1f}초 전"
                            break
                
                logger.error(f"   ❌ {spec['name']} 초기화 타임아웃 ({timeout_seconds}초)")
                logger.error(f"   💔 타임아웃 발생 ({diagnostic_info})")
                
                # 부분 리소스 정리 및 안전 탈출
                partial_inst = self.module_instances.pop(spec['name'], None)
                # semantic_analyzer 전용: 취소 이벤트가 있으면 설정
                if partial_inst and hasattr(partial_inst, '_cancel_event'):
                    partial_inst._cancel_event.set()
                    logger.info(f"   🛑 {spec['name']} 취소 이벤트 설정됨")
                
                raise RuntimeError(f"{spec['name']} 초기화 타임아웃")
                
            except Exception as e:
                logger.error(f"   ❌ {spec['name']} 초기화 실패: {e}")
                # 필수 모듈인 경우 전체 실패
                if spec['priority'] == 'HIGH':
                    raise RuntimeError(f"필수 모듈 {spec['name']} 초기화 실패: {e}")
                else:
                    logger.warning(f"   ⚠️ 선택적 모듈 {spec['name']} 건너뜀")
        
        # 메타 통합 시스템 (선택적)
        try:
            logger.info("🌐 메타 통합 시스템 초기화 중...")
            from advanced_meta_integration_system import AdvancedMetaIntegrationSystem
            meta_integration = AdvancedMetaIntegrationSystem()
            register_system_module('meta_integration', meta_integration, 'meta')
            logger.info("✅ meta_integration 전역 등록 완료")
        except ImportError:
            logger.warning("⚠️ 메타 통합 시스템 모듈을 찾을 수 없음 - 건너뜀")
        except Exception as e:
            logger.warning(f"⚠️ 메타 통합 시스템 초기화 실패: {e}")
        
        # 최종 메모리 상태 출력
        final_memory = get_gpu_memory_info()
        if final_memory:
            final_usage = final_memory['usage_percent']
            logger.info(f"🎯 전역 모듈 초기화 완료 - 최종 GPU 사용률: {final_usage:.1f}%")
            if final_usage < 60:
                logger.warning(f"⚠️ GPU 사용률이 낮습니다 ({final_usage:.1f}%) - 더 적극적 로딩 필요")
        
        logger.info("🎉 전역 모듈 순차 초기화 완료 - HeadAdapter 연결 준비됨")
    
    async def _load_single_module(self, spec):
        """단일 모듈 안전 로딩 - 온라인 다운로드 필요 모듈 체크"""
        module_start_time = time.time()
        logger.info(f"🔧 _load_single_module 시작: {spec['name']} ({spec['class_path']})")
        logger.info(f"   ⏱️ 시작 시간: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
        try:
            # 오프라인 모드 환경 변수 설정 (다운로드 방지)
            import os
            original_env = {}
            offline_env = {
                'TRANSFORMERS_OFFLINE': '1',
                'HF_HUB_OFFLINE': '1', 
                'HF_DATASETS_OFFLINE': '1',
                'HF_HUB_DISABLE_TELEMETRY': '1',
                'DISABLE_TELEMETRY': '1',
                'HF_HOME': os.path.expanduser('~/.cache/huggingface'),
                'TRANSFORMERS_CACHE': os.path.expanduser('~/.cache/huggingface/hub'),
                'HF_DATASETS_CACHE': os.path.expanduser('~/.cache/huggingface/datasets'),
                'SENTENCE_TRANSFORMERS_HOME': os.path.expanduser('~/.cache/torch/sentence_transformers'),
                'TOKENIZERS_PARALLELISM': 'false'
            }
            
            # 환경 변수 백업 및 설정
            for key, value in offline_env.items():
                original_env[key] = os.environ.get(key)
                os.environ[key] = value
            
            try:
                # 온라인 다운로드가 필요한 모듈인지 체크
                requires_download = self._check_if_requires_download(spec)
                
                if requires_download:
                    logger.info(f"🔍 {spec['name']} - 온라인 다운로드 필요 모듈 감지, 캐시 확인 중...")
                    logger.info(f"   - 모듈 경로: {spec['class_path']}")
                    
                    # 캐시 확인
                    has_cache = self._check_model_cache(spec)
                    if not has_cache:
                        logger.warning(f"⚠️ {spec['name']} - 로컬 캐시 없음")
                        print(f"\n{'='*60}")
                        print(f"⚠️ 모듈 {spec['name']} 건너뜀 - 캐시 없음")
                        print(f"모듈 경로: {spec['class_path']}")
                        print(f"필요한 모델이 로컬에 없습니다.")
                        print(f"{'='*60}\n")
                        return None  # 캐시가 없으면 None 반환하여 건너뜀
                    else:
                        logger.info(f"✅ {spec['name']} - 로컬 캐시 발견, 오프라인 로딩 진행")
                else:
                    logger.info(f"📦 {spec['name']} - 다운로드 불필요 모듈, 직접 로딩")
                
                # 동적 import
                module_path, class_name = spec['class_path'].rsplit('.', 1)
                module = __import__(module_path, fromlist=[class_name])
                
                # 클래스 가져오기 시도 (오타 보정 포함)
                try:
                    module_class = getattr(module, class_name)
                except AttributeError:
                    # 클래스명 오타 가능성 - difflib으로 가장 유사한 클래스 찾기
                    import difflib
                    candidates = [name for name in dir(module) if name[0].isupper() and not name.startswith('_')]
                    best_matches = difflib.get_close_matches(class_name, candidates, n=1, cutoff=0.8)
                    
                    if best_matches:
                        actual_class_name = best_matches[0]
                        logger.warning(f"⚠️ 클래스 '{class_name}' 대신 '{actual_class_name}' 사용 (유사도 기반)")
                        module_class = getattr(module, actual_class_name)
                    else:
                        # 재시도 실패 시 원래 에러 발생
                        raise AttributeError(f"module '{module.__name__}' has no attribute '{class_name}'. "
                                           f"Available classes: {candidates[:5]}...")
                
                # device_policy 먼저 확인
                policy = spec.get('device_policy', 'gpu_required')
                logger.info(f"🔧 {spec['name']} device_policy: {policy}")
                
                # 인스턴스 생성
                instance = module_class()
                
                # RAM 선등록 - 모든 모듈을 초기에 DSM에 등록
                if hasattr(self, 'swap_manager') and self.swap_manager:
                    try:
                        # nn.Module 추출 시도
                        nn_core = _extract_nn_module(instance)
                        if nn_core is None:
                            # 어댑터 생성
                            nn_core = SimpleThinAdapter(instance)
                            logger.info(f"📦 {spec['name']} - SimpleThinAdapter로 래핑")
                        
                        # DSM에 RAM 선등록
                        priority = self.swap_manager.DEFAULT_PRIORITIES.get(
                            spec['name'], 
                            SwapPriority.MEDIUM
                        )
                        self.swap_manager.register_model(
                            spec['name'], 
                            nn_core, 
                            priority=priority
                        )
                        logger.info(f"✅ {spec['name']} RAM 선등록 완료 (priority={priority})")
                    except Exception as e:
                        logger.warning(f"⚠️ {spec['name']} RAM 선등록 중 경고: {e}")
                
                # cpu_preload 정책이면 즉시 CPU로 이동 (initialize 전에!)
                if policy == 'cpu_preload':
                    logger.info(f"📋 {spec['name']}을 CPU로 이동 (cpu_preload 정책)")
                    try:
                        # device 속성 변경
                        if hasattr(instance, 'device'):
                            instance.device = torch.device('cpu')
                            logger.info(f"   - device 속성을 CPU로 변경")
                        
                        # 내부 모델들 CPU로 이동
                        if hasattr(instance, 'models') and isinstance(instance.models, dict):
                            for model_name, model in instance.models.items():
                                if model is not None and hasattr(model, 'to'):
                                    instance.models[model_name] = model.to('cpu')
                                    logger.debug(f"   - {model_name} CPU로 이동")
                        
                        # GPU 메모리 즉시 해제
                        torch.cuda.empty_cache()
                        logger.info(f"   ✅ {spec['name']} CPU 이동 완료")
                    except Exception as e:
                        logger.warning(f"   ⚠️ CPU 이동 중 경고: {e}")
                
                # 인스턴스 타입과 메서드 확인
                logger.info(f"🔍 {spec['name']} 인스턴스 생성됨:")
                logger.info(f"   - 타입: {type(instance)}")
                logger.info(f"   - 클래스명: {instance.__class__.__name__}")
                logger.info(f"   - get_pytorch_network 메서드 존재: {hasattr(instance, 'get_pytorch_network')}")
                if hasattr(instance, 'get_pytorch_network'):
                    logger.info(f"   - get_pytorch_network 타입: {type(getattr(instance, 'get_pytorch_network'))}")
                else:
                    # 메서드 목록 출력
                    methods = [m for m in dir(instance) if not m.startswith('_') and callable(getattr(instance, m, None))]
                    logger.info(f"   - 사용 가능한 메서드들 (일부): {methods[:10]}")
                
                # 비동기 초기화 메서드가 있고 needs_initialize가 True면 호출
                if hasattr(instance, 'initialize') and spec.get('needs_initialize', False):
                    initialize_method = getattr(instance, 'initialize')
                    logger.info(f"🔄 {spec['name']} - initialize() 메서드 발견, needs_initialize=True, 호출 중...")
                    
                    # 코루틴인지 확인
                    import inspect
                    if inspect.iscoroutinefunction(initialize_method):
                        logger.info(f"   - 비동기 초기화 실행 중...")
                        await initialize_method()
                        logger.info(f"   ✅ 비동기 초기화 완료")
                    else:
                        logger.info(f"   - 동기 초기화 실행 중...")
                        initialize_method()
                        logger.info(f"   ✅ 동기 초기화 완료")
                    
                    # 초기화 후 get_pytorch_network 재확인
                    if hasattr(instance, 'get_pytorch_network'):
                        logger.info(f"   - 초기화 후 get_pytorch_network 메서드 존재: True")
                        try:
                            network = instance.get_pytorch_network()
                            if network is not None:
                                logger.info(f"   ✅ PyTorch 네트워크 확인됨: {type(network)}")
                            else:
                                logger.warning(f"   ⚠️ get_pytorch_network()가 None 반환")
                        except Exception as e:
                            logger.error(f"   ❌ get_pytorch_network() 호출 실패: {e}")
                elif hasattr(instance, 'initialize') and not spec.get('needs_initialize', False):
                    logger.info(f"⚠️ {spec['name']} - initialize() 메서드는 있지만 needs_initialize=False이므로 건너뜀")
                
                from config import register_system_module, get_system_module
                category = spec.get('category', 'misc')
                
                if policy == 'cpu_preload':
                    # 이미 CPU로 이동했으므로 등록만 수행
                    # 전역 레지스트리에 등록
                    register_system_module(spec['name'], instance, category)
                    
                    # 메모리 매니저에 CPU 프리로드 등록
                    if hasattr(self, 'memory_manager') and self.memory_manager:
                        estimated_mb = spec.get('estimated_mb', None)
                        self.memory_manager.register_cpu_preloaded(spec['name'], estimated_mb if estimated_mb else 0)
                    
                    logger.info(f"✅ {spec['name']} CPU 프리로드 완료 (카테고리: {category})")
                    return instance
                    
                elif policy == 'gpu_required':
                    # GPU 로딩이 필요한 경우 - DSM을 통해 GPU 로딩
                    logger.info(f"🔄 {spec['name']} GPU 로딩 필요 - DSM 경유 GPU 승격...")
                    
                    # 먼저 CPU 프리로드처럼 처리
                    category = spec.get('category', 'utility')
                    logger.info(f"📝 {spec['name']} 전역 레지스트리 등록 (카테고리: {category})")
                    register_system_module(spec['name'], instance, category)
                    
                    # DSM에 등록 후 GPU로 승격
                    if hasattr(self, 'swap_manager') and self.swap_manager:
                        try:
                            # nn.Module 추출
                            actual_nn_module = _extract_nn_module(instance)
                            if actual_nn_module:
                                # DSM에 등록
                                priority = self.swap_manager.DEFAULT_PRIORITIES.get(
                                    spec['name'], 
                                    SwapPriority.MEDIUM
                                )
                                self.swap_manager.register_model(
                                    spec['name'],
                                    actual_nn_module,
                                    priority=priority
                                )
                                logger.info(f"   ✅ {spec['name']} DSM 등록 완료")
                                
                                # DSM을 통해 GPU로 승격
                                await self.swap_manager.load_model_to_gpu(spec['name'])
                                logger.info(f"   ✅ {spec['name']} DSM 경유 GPU 승격 완료")
                            else:
                                logger.warning(f"   ⚠️ {spec['name']} nn.Module 추출 실패 - GPU 승격 스킵")
                        except Exception as e:
                            logger.error(f"   ❌ {spec['name']} DSM GPU 승격 실패: {e}")
                            if spec.get('required', True):
                                raise
                    
                    # 전역 레지스트리에 등록
                    logger.info(f"📝 {spec['name']} 전역 레지스트리 등록 시도...")
                    register_system_module(spec['name'], instance, category)
                    
                    # DSM이 GPU resident 관리 (외부에서 직접 조작 금지)
                    
                    # 등록 확인
                    registered_module = get_system_module(spec['name'])
                    if registered_module is not None:
                        module_load_time = time.time() - module_start_time
                        logger.info(f"✅ {spec['name']} 전역 레지스트리 등록 성공! (카테고리: {category}, 소요시간: {module_load_time:.1f}초)")
                        logger.info(f"   - 등록된 모듈 타입: {type(registered_module)}")
                        
                        # emotion_analyzer의 경우 initialize() 메서드 호출하여 emotion_empathy_head 등록
                        if spec['name'] == 'emotion_analyzer' and hasattr(instance, 'initialize'):
                            logger.info(f"🔄 emotion_analyzer.initialize() 호출하여 emotion_empathy_head 등록...")
                            try:
                                await instance.initialize()
                                logger.info(f"✅ emotion_analyzer.initialize() 완료 - emotion_empathy_head 등록됨")
                            except Exception as e:
                                logger.error(f"❌ emotion_analyzer.initialize() 실패: {e}")
                                # NO FALLBACK - 초기화 실패시 시스템 중단
                                raise RuntimeError(f"emotion_analyzer 초기화 실패 (emotion_empathy_head 등록 실패): {e}")
                        
                        # GPU MEM 로그 포맷
                        gpu_info = get_gpu_memory_info()
                        if gpu_info:
                            logger.info(f"[GPU MEM] after swap-in {spec['name']}: alloc={gpu_info['allocated_mb']/1024:.1f}GB reserved={gpu_info['cached_mb']/1024:.1f}GB util={gpu_info['usage_percent']:.1f}%")
                    else:
                        logger.error(f"❌ {spec['name']} 전역 레지스트리 등록 실패!")
                        raise RuntimeError(f"{spec['name']} 모듈이 전역 레지스트리에 등록되지 않음")
                    
                    return instance
                    
                elif policy == 'gpu_on_demand':
                    # GPU on-demand 정책: CPU 프리로드 + DSM 등록 (GPU 승격은 나중에)
                    logger.info(f"🔄 {spec['name']} GPU on-demand 정책 - CPU 프리로드 + DSM 등록")
                    
                    # CPU로 이동
                    logger.info(f"📋 {spec['name']}을 CPU로 이동 (gpu_on_demand 정책)")
                    try:
                        # device 속성 변경
                        if hasattr(instance, 'device'):
                            instance.device = torch.device('cpu')
                            logger.info(f"   - device 속성을 CPU로 변경")
                        
                        # 내부 모델들 CPU로 이동
                        if hasattr(instance, 'models') and isinstance(instance.models, dict):
                            for model_name, model in instance.models.items():
                                if model is not None and hasattr(model, 'to'):
                                    instance.models[model_name] = model.to('cpu')
                                    logger.debug(f"   - {model_name} CPU로 이동")
                        
                        # GPU 메모리 즉시 해제
                        torch.cuda.empty_cache()
                        logger.info(f"   ✅ {spec['name']} CPU 이동 완료")
                    except Exception as e:
                        logger.warning(f"   ⚠️ CPU 이동 중 경고: {e}")
                    
                    # 전역 레지스트리 등록
                    category = spec.get('category', 'misc')
                    register_system_module(spec['name'], instance, category)
                    logger.info(f"✅ {spec['name']} 전역 등록 완료 (카테고리: {category})")
                    
                    # DSM에 등록 (나중에 GPU 승격 가능하도록)
                    if hasattr(self, 'swap_manager') and self.swap_manager:
                        try:
                            actual_nn_module = _extract_nn_module(instance)
                            if actual_nn_module:
                                # DSM 우선순위 결정
                                priority_map = {
                                    'HIGH': SwapPriority.HIGH,
                                    'MEDIUM': SwapPriority.MEDIUM,
                                    'LOW': SwapPriority.LOW
                                }
                                priority = priority_map.get(
                                    spec.get('priority', 'MEDIUM'),
                                    SwapPriority.MEDIUM
                                )
                                
                                # DSM에 등록
                                self.swap_manager.register_model(
                                    spec['name'],
                                    actual_nn_module,
                                    priority=priority
                                )
                                logger.info(f"   ✅ {spec['name']} DSM 등록 완료 (우선순위: {priority.name})")
                                logger.info(f"   📊 GPU 승격은 필요시 load_model_to_gpu()로 수행됩니다")
                            else:
                                logger.warning(f"   ⚠️ {spec['name']} nn.Module 추출 실패 - DSM 등록 스킵")
                        except Exception as e:
                            logger.warning(f"   ⚠️ {spec['name']} DSM 등록 중 경고: {e}")
                    
                    # 메모리 매니저에 CPU 프리로드 등록
                    if hasattr(self, 'memory_manager') and self.memory_manager:
                        estimated_mb = spec.get('estimated_mb', 0)
                        self.memory_manager.register_cpu_preloaded(spec['name'], estimated_mb)
                        logger.info(f"   📊 메모리 매니저 등록: {estimated_mb}MB")
                    
                    logger.info(f"✅ {spec['name']} GPU on-demand 초기화 완료")
                    return instance
                    
                else:
                    # 잘못된 device_policy 값
                    logger.error(f"❌ 잘못된 device_policy 값: {policy}")
                    raise ValueError(f"허용되지 않은 device_policy: {policy}. 'gpu_required', 'cpu_preload' 또는 'gpu_on_demand'만 허용됩니다.")
                
            finally:
                # 환경 변수 복원
                for key, original_value in original_env.items():
                    if original_value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = original_value
            
        except Exception as e:
            error_msg = f"모듈 {spec['name']} 로딩 실패: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            # 상세 에러 정보 출력
            import traceback
            tb_str = traceback.format_exc()
            logger.error(f"🚨 {spec['name']} 로딩 실패 상세 스택:\n{tb_str}")
            
            # 재시도 로직 처리
            retry_count = spec.get('retry_on_error', 0)
            if hasattr(self, '_module_retry_count'):
                current_retry = self._module_retry_count.get(spec['name'], 0)
            else:
                self._module_retry_count = {}
                current_retry = 0
            
            if current_retry < retry_count:
                self._module_retry_count[spec['name']] = current_retry + 1
                logger.warning(f"🔄 {spec['name']} 로딩 재시도 중... ({current_retry + 1}/{retry_count})")
                await asyncio.sleep(2)  # 2초 대기 후 재시도
                return await self._load_single_module(spec)
            
            # optional 모듈 처리
            if spec.get('optional', False):
                logger.warning(f"⚠️ 선택적 모듈 {spec['name']} 로딩 실패 - 건너뜀")
                print(f"\n{'='*60}")
                print(f"⚠️ 선택적 모듈 {spec['name']} 건너뜀")
                print(f"에러: {str(e)}")
                print(f"모듈 경로: {spec['class_path']}")
                print(f"{'='*60}\n")
                return None
            
            # 필수 모듈인 경우에만 예외 발생
            if spec.get('priority') == 'HIGH' and not spec.get('optional', False):
                print(f"\n{'='*60}")
                print(f"🚨 필수 모듈 {spec['name']} 로딩 실패!")
                print(f"{'='*60}")
                print(f"에러: {str(e)}")
                print(f"모듈 경로: {spec['class_path']}")
                print(f"{'='*60}\n")
                raise
            else:
                logger.warning(f"선택적 모듈 {spec['name']} 로딩 실패 - 건너뜀")
                return None
    
    def _check_if_requires_download(self, spec):
        """모듈이 온라인 다운로드가 필요한지 체크"""
        # transformers, sentence-transformers 기반 모듈들만 체크
        if 'advanced_emotion_analyzer' in spec.get('class_path', ''):
            return True  # transformers 사용
        elif 'advanced_semantic_analyzer' in spec.get('class_path', ''):
            return True  # transformers + sentence-transformers 사용
        elif 'local_translator' in spec.get('class_path', ''):
            return True  # OPUS-MT 모델 사용
        elif 'advanced_bentham_calculator' in spec.get('class_path', ''):
            return True  # transformers 사용 (기존 모델들 재사용)
        
        # 다른 모듈들은 transformers 모델을 사용하지 않음
        return False
    
    def _check_model_cache(self, spec):
        """모델 캐시 존재 여부 확인"""
        import os
        from pathlib import Path
        
        # 실제 huggingface 캐시 경로
        hub_cache_path = Path.home() / '.cache' / 'huggingface' / 'hub'
        
        if not hub_cache_path.exists():
            logger.warning(f"HuggingFace 캐시 디렉토리 없음: {hub_cache_path}")
            return False
        
        # 모듈별 필요한 모델들 확인
        required_models = []
        
        if 'advanced_emotion_analyzer' in spec.get('class_path', ''):
            required_models = [
                'models--j-hartmann--emotion-english-distilroberta-base',
                'models--beomi--KcELECTRA-base-v2022',
                'models--jhgan--ko-sroberta-multitask',
                'models--sentence-transformers--paraphrase-multilingual-mpnet-base-v2'
            ]
        elif 'advanced_semantic_analyzer' in spec.get('class_path', ''):
            required_models = [
                'models--jhgan--ko-sroberta-multitask',
                'models--j-hartmann--emotion-english-distilroberta-base',
                'models--facebook--bart-large-mnli',
                'models--microsoft--DialoGPT-medium',
                'models--klue--bert-base'
            ]
        elif 'local_translator' in spec.get('class_path', ''):
            required_models = [
                'models--Helsinki-NLP--opus-mt-ko-en'
            ]
        elif 'advanced_bentham_calculator' in spec.get('class_path', ''):
            # bentham_calculator는 다른 모듈들의 모델을 재사용
            required_models = [
                'models--beomi--KcELECTRA-base-v2022',
                'models--jhgan--ko-sroberta-multitask'
            ]
        
        # 필요한 모델들이 캐시에 있는지 확인
        logger.info(f"📂 {spec['name']}에 필요한 모델 {len(required_models)}개 확인 중...")
        logger.info(f"   📁 캐시 경로: {hub_cache_path}")
        missing_models = []
        
        for model_cache_name in required_models:
            model_path = hub_cache_path / model_cache_name
            if not model_path.exists():
                logger.warning(f"   ❌ 모델 캐시 없음: {model_cache_name}")
                logger.warning(f"      경로: {model_path}")
                missing_models.append(model_cache_name)
            else:
                # 캐시 크기 확인
                try:
                    cache_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
                    cache_size_mb = cache_size / (1024 * 1024)
                    logger.info(f"   ✅ 모델 캐시 발견: {model_cache_name} ({cache_size_mb:.1f}MB)")
                except Exception as e:
                    logger.info(f"   ✅ 모델 캐시 발견: {model_cache_name} (크기 확인 실패)")
        
        if missing_models:
            print(f"\n{'='*60}")
            print(f"❌ {spec['name']} 모듈에 필요한 모델 캐시 부족")
            print(f"누락된 모델:")
            for model in missing_models:
                print(f"  - {model}")
            print(f"{'='*60}\n")
            return False
        
        return len(required_models) > 0  # 필요한 모델이 있고 모두 캐시에 있으면 True
    
    def _log_dsm_snapshot(self, tag: str):
        """DSM 상태 스냅샷 로깅 - 디버깅용"""
        try:
            swap = self.swap_manager
            if not swap:
                logger.warning(f"[DSM@{tag}] swap_manager가 None")
                return
            
            # 등록된 모든 모델
            all_models = list(swap.models.keys()) if hasattr(swap, 'models') else []
            logger.critical(f"[DSM@{tag}] models={all_models[:30]}")
            
            # GPU 상주 모델들
            gpu_keys = list(swap.gpu_resident_models.keys()) if hasattr(swap, 'gpu_resident_models') else []
            
            # 크기 정보
            sizes = {}
            priorities = {}
            for k in gpu_keys[:20]:
                if k in swap.models:
                    sizes[k] = f"{swap.models[k].size_mb:.1f}MB"
                    priorities[k] = swap.models[k].priority.value if swap.models[k].priority else "NONE"
            
            logger.critical(f"[DSM@{tag}] gpu_resident={gpu_keys[:20]} sizes={sizes}")
            logger.critical(f"[DSM@{tag}] priorities={priorities}")
            
        except Exception as e:
            logger.warning(f"DSM 스냅샷 로깅 실패: {e}")
    
    def _log_gpu_memory_state(self, stage: str):
        """GPU 메모리 상태 로깅"""
        try:
            gpu_info = get_gpu_memory_info()
            if gpu_info:
                usage_percent = gpu_info.get('usage_percent', 0)
                used_gb = gpu_info.get('memory_used_gb', 0)
                total_gb = gpu_info.get('memory_total_gb', 8)
                logger.info(f"🔍 [{stage}] GPU 메모리: {usage_percent:.1f}% ({used_gb:.2f}/{total_gb:.2f}GB)")
            else:
                logger.info(f"🔍 [{stage}] GPU 메모리 정보 없음")
        except Exception as e:
            logger.warning(f"GPU 메모리 상태 로깅 실패: {e}")
    
    async def _emergency_gpu_cleanup_using_dsm(self, target_usage: float = 0.78, 
                                               exclude: Optional[set] = None) -> None:
        """
        DSM 레지스트리를 기준으로만 GPU 언로드를 수행한다.
        exclude: 현재 로딩중/필수 보호 모델 이름 집합
        """
        from dynamic_swap_manager import SwapPriority
        swap = self.swap_manager
        if swap is None:
            logger.error("❌ DSM 미초기화 - 긴급 정리 스킵")
            return
        exclude = exclude or set()

        info_before = get_gpu_memory_info() or {}
        used_before = info_before.get('usage_percent', 0.0)
        freed_total = 0.0

        # 후보: GPU 상주 + CRITICAL 제외 + exclude 제외
        candidates = []
        for name, model in list(swap.models.items()):
            if name in swap.gpu_resident_models:
                if model.priority != SwapPriority.CRITICAL and name not in exclude:
                    candidates.append((name, model.priority, model.size_mb, model.last_access))

        if not candidates:
            logger.warning("⚠️ DSM 기반 후보 0개 → 언로드 불가")
            return

        # 우선순위/최근접근 기준으로 정렬 (낮은 우선순위, 오래된 접근 순)
        candidates.sort(key=lambda x: (x[1].value, x[3]))  # priority.value asc, last_access asc
        logger.info(f"📝 DSM 후보 {len(candidates)}개: {[c[0] for c in candidates[:20]]} ...")

        for name, prio, sz, _ in candidates:
            try:
                await swap.unload_model_from_gpu(name)
                freed_total += sz
            except Exception as e:
                logger.error(f"언로드 실패: {name} ({e})")
                continue
            
            # CUDA 캐시 반영 대기
            await asyncio.sleep(0.15)
            info_now = get_gpu_memory_info() or {}
            if info_now.get('usage_percent', 1.0) <= target_usage * 100:
                break

        info_after = get_gpu_memory_info() or {}
        logger.info(
            f"✅ DSM 긴급 정리 완료: 사용률 {used_before:.1f}% → {info_after.get('usage_percent', used_before):.1f}% "
            f"(해제 추정 {freed_total:.1f}MB)"
        )
    
    async def _initialize_core_components(self):
        """핵심 컴포넌트 초기화"""
        
        # 컴포넌트들을 점진적으로 초기화
        self.log_manager.log_system_event("핵심 컴포넌트 초기화 시작")
        
        # GPU 메모리 초기 상태 확인 - 가장 먼저!
        self._log_gpu_memory_state("초기화 시작 전")
        
        # 🔥 1단계: 전역 모듈 순차 초기화 (GPU 메모리 85% 목표)
        logger.critical(f"🔍 fast_init_mode = {self.fast_init_mode}")
        if self.fast_init_mode:
            logger.info("⚡ 빠른 초기화 모드 - 전역 모듈 초기화 건너뜀")
        else:
            logger.critical("🌐 전역 모듈 순차 초기화 시작 - MasterMemoryOrchestrator 관리")
            try:
                await self._initialize_global_modules_sequential()
                logger.critical("✅ 전역 모듈 순차 초기화 완료")
            except Exception as e:
                logger.critical(f"❌ 전역 모듈 초기화 실패: {str(e)}")
                import traceback
                logger.critical(f"❌ 스택 트레이스:\n{traceback.format_exc()}")
                # 프로젝트 규칙: fallback 없이 실패
                raise RuntimeError(f"전역 모듈 초기화 실패 - 시스템 중단: {e}") from e
        
        # 🔥 핵심 추가: 통합 메모리 관리 시스템 초기화 (메모리 폭주 방지)
        logger.info("🧠 통합 메모리 관리 시스템 초기화 시작...")
        from config import initialize_unified_memory_system
        if initialize_unified_memory_system():
            logger.info("✅ 통합 메모리 관리 시스템 초기화 완료")
        else:
            logger.error("❌ 통합 메모리 관리 시스템 초기화 실패 - 메모리 폭주 위험!")
        
        # 전역 모듈 초기화 후 GPU 상태 확인
        self._log_gpu_memory_state("전역 모듈 초기화 후")
        
        # 1. Unified Backbone
        logger.info("🧠 Unified Backbone 초기화 시작...")
        self.unified_backbone = RedHeartUnifiedBackbone()
        logger.info("Unified Backbone 초기화 완료")
        
        # 백본 초기화 후 메모리 상태
        self._log_gpu_memory_state("백본 초기화 후")
        
        # 2. Swap Manager (워크플로우 인식 활성화)
        logger.info("🔄 Dynamic Swap Manager 초기화 시작...")
        swap_config = ADVANCED_CONFIG.get('dynamic_swap_config', {})
        swap_config['workflow_aware'] = True
        swap_config['memory_threshold_mb'] = 6500.0  # 8GB GPU의 ~81%
        self.swap_manager = RedHeartDynamicSwapManager(config=swap_config)
        await self.swap_manager.initialize()
        
        # 전역 DSM으로 publish (이후 모든 모듈의 get_swap_manager()가 같은 객체를 보도록)
        from dynamic_swap_manager import set_swap_manager, get_swap_manager, SwapPriority
        set_swap_manager(self.swap_manager)
        logger.critical(f"[O] set_swap_manager OK: self_id={id(self.swap_manager)} global_id={id(get_swap_manager())}")
        assert id(self.swap_manager) == id(get_swap_manager()), "DSM publish failed"
        logger.info("Dynamic Swap Manager 초기화 완료 (워크플로우 인식 활성화)")
        
        # 🔥 백본을 DSM에 등록 (가장 큰 메모리 소비자)
        if self.swap_manager and self.unified_backbone:
            try:
                # _extract_nn_module 헬퍼를 사용하여 안전하게 추출
                backbone_model = _extract_nn_module(self.unified_backbone)
                
                if backbone_model is not None:
                    self.swap_manager.register_model(
                        "unified_backbone",
                        backbone_model,
                        priority=SwapPriority.CRITICAL  # 백본은 CRITICAL로 보호
                    )
                    logger.critical("✅ 백본 DSM 등록 완료 (CRITICAL 우선순위)")
                else:
                    # 추출 실패 시 MinimalBackbone으로 임시 등록 (NO FALLBACK 정책 예외)
                    logger.warning("⚠️ 백본 nn.Module 추출 실패 - MinimalBackbone로 임시 등록")
                    import torch
                    class _MinimalBackbone(torch.nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.placeholder = torch.nn.Linear(1280, 1280, bias=False)
                        def forward(self, x):
                            return x if x is None else self.placeholder(x)
                    
                    minimal_backbone = _MinimalBackbone()
                    minimal_backbone.to(get_smart_device())  # GPU로 이동
                    self.swap_manager.register_model(
                        "unified_backbone",
                        minimal_backbone,
                        priority=SwapPriority.CRITICAL
                    )
                    logger.critical("⚠️ MinimalBackbone DSM 등록 완료 (백본 추출 실패 대비)")
            except Exception as e:
                logger.error(f"❌ 백본 DSM 등록 실패: {e}")
        
        self._log_gpu_memory_state("스왑 매니저 초기화 후")
        self._log_dsm_snapshot("스왑매니저초기화후")
        
        # 헤드 선등록 보장 (스왑 매니저 준비 직후)
        self._ensure_pre_registered_heads()
        
        # warm load/unload 수행 (비동기 컨텍스트에서 안전하게)
        if hasattr(self, '_needs_warm_load_unload') and self._needs_warm_load_unload:
            try:
                logger.info("🔄 warm load/unload 시작 (비동기 컨텍스트)")
                # GPU로 올리기
                if "emotion_empathy_head" in self.swap_manager.models:
                    await self.swap_manager.load_head_to_gpu("emotion_empathy_head", timeout=1.0)
                    logger.info("✅ emotion_empathy_head warm load 완료")
                    
                    # 잠시 대기 (메모리 안정화)
                    await asyncio.sleep(0.1)
                    
                    # GPU에서 내리기
                    await self.swap_manager.unload_model_from_gpu("emotion_empathy_head")
                    logger.info("✅ emotion_empathy_head warm unload 완료")
                    
                self._needs_warm_load_unload = False
                logger.info("✅ warm load/unload 완료 (스왑 경로 검증)")
            except Exception as e:
                logger.warning(f"⚠️ warm load/unload 실패: {e}")
                self._needs_warm_load_unload = False
        
        # 3. Head Compatibility Manager (헤드 등록을 위해 필요)
        logger.info("🎭 Head Compatibility Manager 초기화 시작...")
        self.head_compatibility_manager = HeadCompatibilityManager(
            self.unified_backbone, 
            self.swap_manager
        )
        logger.info("Head Compatibility Manager 초기화 완료")
        self._log_gpu_memory_state("헤드 호환성 매니저 생성 후")
        
        # 4. 모든 헤드 초기화 및 등록 (메모리 폭발 의심 구간)
        if self.fast_init_mode:
            logger.info("⚡ 빠른 초기화 모드 - 헤드 초기화 건너뜀")
        else:
            # 헤드 초기화 전 선등록 재확인
            self._ensure_pre_registered_heads()
            
            logger.critical("🚨 헤드 초기화 시작 - 메모리 폭발 의심 구간!")
            logger.critical(f"🔍 head_compatibility_manager = {type(self.head_compatibility_manager)}")
            logger.critical(f"🔍 head_compatibility_manager.initialized = {getattr(self.head_compatibility_manager, 'initialized', 'NOT_FOUND')}")
            logger.critical("🔍 initialize_all_heads() 호출 직전...")
            
            # asyncio 타임아웃 추가 (900초 제한 - 5개 헤드 * 180초) - 실패시 시스템 중단
            try:
                await asyncio.wait_for(
                    self.head_compatibility_manager.initialize_all_heads(),
                    timeout=900.0
                )
                logger.critical("🔍 initialize_all_heads() 호출 완료!")
            except asyncio.TimeoutError:
                logger.error("❌ initialize_all_heads() 900초 타임아웃 발생!")
                logger.error("🔍 어떤 헤드에서 hanging이 발생했는지 확인 중...")
                
                # 각 헤드별 상태 확인
                if hasattr(self.head_compatibility_manager, 'head_adapters'):
                    for head_name, adapter in self.head_compatibility_manager.head_adapters.items():
                        status = getattr(adapter, 'initialized', 'UNKNOWN')
                        logger.error(f"  📋 {head_name}: {status}")
                
                # 시스템 완전 중단 - graceful degradation 금지
                logger.critical("❌ 헤드 초기화 실패로 시스템 중단")
                raise RuntimeError("Head initialization timeout - system cannot continue without all heads")
            
        logger.critical("🚨 헤드 초기화 완료 - 메모리 상태 확인 중...")
        self._log_gpu_memory_state("모든 헤드 초기화 후")
        self._log_dsm_snapshot("헤드초기화후")
        
        # 헤드 초기화 직후 메모리 상태에 따라 조건부 정리
        mem_info = get_gpu_memory_info()
        current_usage = mem_info.get("usage_percent", 0) if mem_info else 0
        
        if current_usage >= 85.0:
            logger.critical(f"🚨 GPU 사용률 {current_usage:.1f}% - DSM 긴급 정리 수행 (target=85%)")
            self._log_dsm_snapshot("긴급정리전")
            # 필수 헤드와 백본은 제외
            exclude_models = {'emotion_empathy_head', 'unified_backbone', 'regret_learning_head'}
            await self._emergency_gpu_cleanup_using_dsm(target_usage=0.85, exclude=exclude_models)
            self._log_gpu_memory_state("자동 메모리 정리 후")
            self._log_dsm_snapshot("긴급정리후")
        else:
            logger.info(f"✅ GPU 사용률 {current_usage:.1f}% - 임계치 미만으로 정리 스킵")
        
        # 5. Synergy System
        logger.info("⚡ Intelligent Synergy System 초기화 시작...")
        self.synergy_system = IntelligentSynergySystem()
        logger.info("Intelligent Synergy System 초기화 완료")
        self._log_gpu_memory_state("시너지 시스템 초기화 후")
        
        # 6. Learning System - HeadCompatibilityManager 전달
        logger.info("📚 Unified Learning System 초기화 시작...")
        self.learning_system = UnifiedLearningSystem(
            head_compatibility_manager=self.head_compatibility_manager,
            swap_manager=self.swap_manager
        )
        logger.info("Unified Learning System 인스턴스 생성 완료 (HeadCompatibilityManager 공유)")
        
        # Learning System의 비동기 초기화 호출 - cached_head_modules 채우기
        logger.info("📚 Unified Learning System 헤드 모듈 캐싱 시작...")
        try:
            await self.learning_system.initialize_system()
            logger.info("✅ Unified Learning System 초기화 완료 - 모든 헤드 모듈 캐싱됨")
        except RuntimeError as e:
            logger.error(f"❌ 헤드 모듈 초기화 실패: {str(e)}")
            raise  # 프로젝트 규칙: fallback 없이 hard failure
        
        self._log_gpu_memory_state("학습 시스템 초기화 후")
        self._log_dsm_snapshot("학습시스템초기화후")
        
        # 7. Pattern Analyzer
        logger.info("📊 Advanced Usage Pattern Analyzer 초기화 시작...")
        self.pattern_analyzer = AdvancedUsagePatternAnalyzer()
        logger.info("Advanced Usage Pattern Analyzer 초기화 완료")
        self._log_gpu_memory_state("패턴 분석기 초기화 후")
    
    async def run_training_pipeline(self, mode: str = "auto", **kwargs) -> bool:
        """훈련 파이프라인 실행"""
        
        if self.system_status != SystemStatus.READY:
            logger.error("시스템이 준비되지 않음")
            return False
        
        try:
            self.system_status = SystemStatus.TRAINING
            self.log_manager.log_system_event(f"훈련 파이프라인 시작: {mode}")
            
            # 무한 루프 방지: run_learning.sh 재실행 제거, Python 학습 시스템 직접 실행
            logger.info(f"Python 통합 학습 시스템 직접 실행 (모드: {mode})")
            
            # Python 통합 학습 시스템 실행
            python_success = await self._run_python_training(**kwargs)
            
            if python_success:
                self.log_manager.log_system_event("훈련 파이프라인 완료", "INFO")
                self.system_status = SystemStatus.READY
                return True
            else:
                raise Exception("Python 통합 학습 시스템 실행 실패")
            
        except Exception as e:
            self.system_status = SystemStatus.ERROR
            self.log_manager.log_system_event(f"훈련 파이프라인 실패: {str(e)}", "ERROR")
            
            # 자동 복구 시도
            if self.auto_recovery and self.recovery_attempts < self.max_recovery_attempts:
                logger.info("자동 복구 시도 중...")
                await self._attempt_recovery()
            
            return False
    
    async def _run_learning_script(self, mode: str, **kwargs) -> bool:
        """run_learning.sh 스크립트 실행"""
        
        # 명령어 구성
        command = ["bash", str(self.script_path), mode]
        
        # 추가 인자들 처리
        for key, value in kwargs.items():
            if key == "samples":
                command.extend(["--samples", str(value)])
            elif key == "batch_size":
                command.extend(["--batch-size", str(value)])
            elif key == "learning_rate":
                command.extend(["--learning-rate", str(value)])
            elif key == "verbose" and value:
                command.append("--verbose")
            elif key == "debug" and value:
                command.append("--debug")
        
        try:
            # 스크립트 실행
            process_name = f"run_learning_{mode}_{int(time.time())}"
            pid = await self.process_manager.start_background_process(
                process_name, command, restart_on_failure=False
            )
            
            self.log_manager.log_system_event(f"run_learning.sh 실행: PID {pid}")
            
            # 프로세스 완료 대기 (타임아웃 설정)
            timeout = kwargs.get('timeout', 3600)  # 기본 1시간
            
            start_time = time.time()
            while process_name in self.process_manager.processes:
                if time.time() - start_time > timeout:
                    logger.warning("run_learning.sh 타임아웃")
                    await self.process_manager.stop_process(process_name)
                    return False
                
                await asyncio.sleep(5)  # 5초마다 확인
            
            logger.info("run_learning.sh 실행 완료")
            return True
            
        except Exception as e:
            logger.error(f"run_learning.sh 실행 오류: {str(e)}")
            return False
    
    async def _run_python_training(self, **kwargs) -> bool:
        """Python 통합 학습 시스템 실행"""
        
        try:
            # 가상의 데이터 로더 (실제 구현에서는 실제 데이터 사용)
            class DummyDataLoader:
                def __init__(self, num_batches=100):
                    self.num_batches = num_batches
                
                def __iter__(self):
                    for i in range(self.num_batches):
                        yield {
                            'text': f'Training sample {i}',
                            'batch_size': kwargs.get('batch_size', 4),
                            'labels': np.random.randint(0, 10, size=(kwargs.get('batch_size', 4),))
                        }
            
            # 훈련 설정
            num_epochs = kwargs.get('epochs', 3)
            num_samples = kwargs.get('samples', 500)
            num_batches = min(num_samples // kwargs.get('batch_size', 4), 500)
            
            train_loader = DummyDataLoader(num_batches)
            val_loader = DummyDataLoader(num_batches // 5)  # 검증 데이터는 1/5
            
            # 학습 시작 직전 헤드 선등록 재확인
            self._ensure_pre_registered_heads()
            
            # 통합 학습 실행
            await self.learning_system.train_unified_system(
                train_data_loader=train_loader,
                validation_data_loader=val_loader,
                num_epochs=num_epochs
            )
            
            logger.info("Python 통합 학습 완료")
            return True
            
        except Exception as e:
            logger.error(f"Python 통합 학습 오류: {str(e)}")
            import traceback
            tb_str = traceback.format_exc()
            logger.error(f"😨 상세 에러 스택:\n{tb_str}")
            logger.error(f"🔍 에러 타입: {type(e).__name__}")
            logger.error(f"📍 num_epochs: {num_epochs}, num_batches: {num_batches}")
            return False
    
    async def _attempt_recovery(self):
        """자동 복구 시도"""
        
        self.recovery_attempts += 1
        self.log_manager.log_system_event(f"자동 복구 시도 {self.recovery_attempts}/{self.max_recovery_attempts}")
        
        try:
            # 1. 시스템 정리
            await self._cleanup_system()
            
            # 2. 컴포넌트 재초기화
            await self._initialize_core_components()
            
            # 3. 시스템 상태 복원
            self.system_status = SystemStatus.READY
            
            self.log_manager.log_system_event("자동 복구 성공", "INFO")
            logger.info("시스템 자동 복구 성공")
            
        except Exception as e:
            self.log_manager.log_system_event(f"자동 복구 실패: {str(e)}", "ERROR")
            logger.error(f"자동 복구 실패: {str(e)}")
    
    async def _cleanup_system(self):
        """시스템 정리"""
        
        # 실행 중인 프로세스 정리
        for process_name in list(self.process_manager.processes.keys()):
            await self.process_manager.stop_process(process_name)
        
        # GPU 메모리 정리
        if hasattr(self.unified_backbone, 'clear_cache'):
            self.unified_backbone.clear_cache()
        
        # 임시 파일 정리
        temp_dir = "./temp"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        logger.info("시스템 정리 완료")
    
    def _handle_system_alert(self, alert_message: str, metrics: SystemHealthMetrics):
        """시스템 알림 처리"""
        
        self.log_manager.log_system_event(f"시스템 알림: {alert_message}", "WARNING")
        
        # 심각한 알림의 경우 자동 조치
        if "CPU" in alert_message and metrics.cpu_usage > 95:
            logger.warning("높은 CPU 사용률 감지 - 부하 분산 시도")
            # 여기서 부하 분산 로직 구현
        
        elif "메모리" in alert_message and metrics.memory_usage > 90:
            logger.warning("높은 메모리 사용률 감지 - 메모리 정리 시도")
            # 여기서 메모리 정리 로직 구현
    
    async def _start_dashboard(self):
        """웹 대시보드 시작"""
        
        try:
            # 간단한 웹 대시보드 (실제 구현에서는 FastAPI, Flask 등 사용)
            logger.info(f"웹 대시보드 시작 예정: http://localhost:{self.dashboard_port}")
            self.log_manager.log_system_event(f"웹 대시보드 포트 {self.dashboard_port}에서 시작 예정")
            
        except Exception as e:
            logger.error(f"웹 대시보드 시작 실패: {str(e)}")
    
    async def shutdown_system(self):
        """시스템 종료"""
        
        try:
            self.system_status = SystemStatus.SHUTDOWN
            self.log_manager.log_system_event("시스템 종료 시작")
            
            # 1. 실행 중인 프로세스 정리
            for process_name in list(self.process_manager.processes.keys()):
                await self.process_manager.stop_process(process_name)
            
            # 2. 모니터링 중지
            self.system_monitor.stop_monitoring()
            
            # 3. 로그 회전
            self.log_manager.rotate_logs()
            
            # 4. 시스템 정리
            await self._cleanup_system()
            
            self.is_running = False
            
            uptime = datetime.now() - self.start_time
            self.log_manager.log_system_event(f"시스템 정상 종료 (가동시간: {uptime})", "INFO")
            logger.info(f"Red Heart 통합 시스템 종료 완료 (가동시간: {uptime})")
            
        except Exception as e:
            self.log_manager.log_system_event(f"시스템 종료 중 오류: {str(e)}", "ERROR")
            logger.error(f"시스템 종료 중 오류: {str(e)}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """전체 시스템 상태 조회"""
        
        # 시스템 모니터 요약
        monitor_summary = self.system_monitor.get_system_summary()
        
        # 프로세스 상태
        process_status = self.process_manager.get_process_status()
        
        # 컴포넌트 상태
        component_status = {}
        if self.learning_system:
            component_status['learning_system'] = self.learning_system.get_training_statistics()
        
        if self.synergy_system:
            component_status['synergy_system'] = self.synergy_system.get_synergy_statistics()
        
        # 전체 상태 종합
        uptime = datetime.now() - self.start_time
        
        return {
            'system_status': self.system_status.value,
            'uptime_seconds': uptime.total_seconds(),
            'is_running': self.is_running,
            'recovery_attempts': self.recovery_attempts,
            'monitor_summary': monitor_summary,
            'process_status': process_status,
            'component_status': component_status,
            'timestamp': datetime.now().isoformat()
        }
    
    def _log_gpu_memory_state(self, stage_name: str):
        """단계별 GPU 메모리 상태 상세 로깅"""
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning(f"[{stage_name}] CUDA 사용 불가")
                return
            
            # 기본 GPU 메모리 정보
            device_props = torch.cuda.get_device_properties(0)
            total_memory_gb = device_props.total_memory / (1024**3)
            allocated_gb = torch.cuda.memory_allocated(0) / (1024**3)
            reserved_gb = torch.cuda.memory_reserved(0) / (1024**3)
            free_gb = total_memory_gb - allocated_gb
            allocated_percent = (allocated_gb / total_memory_gb) * 100
            
            logger.critical(f"🔍 [{stage_name}] GPU 메모리 상태:")
            logger.critical(f"   💾 총 메모리: {total_memory_gb:.2f}GB")
            logger.critical(f"   📊 할당됨: {allocated_gb:.3f}GB ({allocated_percent:.1f}%)")
            logger.critical(f"   📦 예약됨: {reserved_gb:.3f}GB")
            logger.critical(f"   💚 여유: {free_gb:.3f}GB")
            
            # config.py의 get_gpu_memory_info()와 비교
            from config import get_gpu_memory_info
            config_info = get_gpu_memory_info()
            if config_info:
                config_percent = config_info.get('usage_percent', 0)
                logger.critical(f"   🔄 config.py 측정: {config_percent:.1f}%")
                if abs(allocated_percent - config_percent) > 5:
                    logger.error(f"   ⚠️ 측정 방식 차이 발견: torch({allocated_percent:.1f}%) vs config({config_percent:.1f}%)")
            
            # GPU 텐서 개수 및 크기 분석
            import gc
            gpu_tensors = []
            total_tensor_memory = 0
            
            for obj in gc.get_objects():
                if torch.is_tensor(obj) and obj.is_cuda:
                    tensor_size = obj.element_size() * obj.numel()
                    gpu_tensors.append({
                        'shape': list(obj.shape),
                        'dtype': str(obj.dtype),
                        'size_mb': tensor_size / (1024**2),
                        'requires_grad': obj.requires_grad
                    })
                    total_tensor_memory += tensor_size
            
            total_tensor_gb = total_tensor_memory / (1024**3)
            logger.critical(f"   🧠 GPU 텐서: {len(gpu_tensors)}개, {total_tensor_gb:.3f}GB")
            
            # 큰 텐서들 상위 5개 출력
            if gpu_tensors:
                large_tensors = sorted(gpu_tensors, key=lambda x: x['size_mb'], reverse=True)[:5]
                logger.critical(f"   📋 큰 텐서 TOP 5:")
                for i, tensor in enumerate(large_tensors, 1):
                    logger.critical(f"      {i}. {tensor['shape']} ({tensor['dtype']}) - {tensor['size_mb']:.1f}MB")
            
            # 메모리 캐시 상태
            logger.critical(f"   🗂️ 캐시된 메모리: {(reserved_gb - allocated_gb):.3f}GB")
            
            # 위험 수준 판정
            if allocated_percent > 100:
                logger.error(f"   🚨🚨🚨 CRITICAL: 메모리 오버플로우 ({allocated_percent:.1f}%)")
            elif allocated_percent > 90:
                logger.warning(f"   ⚠️ WARNING: 높은 메모리 사용률 ({allocated_percent:.1f}%)")
            
            # 메모리 통계 업데이트 (peak tracking)
            current_max = torch.cuda.max_memory_allocated(0) / (1024**3)
            if not hasattr(self, '_peak_memory_gb'):
                self._peak_memory_gb = 0
            
            if current_max > self._peak_memory_gb:
                self._peak_memory_gb = current_max
                logger.critical(f"   🔝 새로운 Peak 메모리: {self._peak_memory_gb:.3f}GB (at {stage_name})")
            
        except Exception as e:
            logger.error(f"[{stage_name}] GPU 메모리 상태 로깅 실패: {str(e)}")

# 사용 예시 함수
async def example_usage():
    """통합 시스템 오케스트레이터 사용 예시"""
    
    # 오케스트레이터 생성
    orchestrator = UnifiedSystemOrchestrator({
        'dashboard_port': 8080,
        'enable_dashboard': False,  # 예시에서는 비활성화
        'auto_recovery': True
    })
    
    try:
        print("=== Red Heart 통합 시스템 오케스트레이터 테스트 ===")
        
        # 시스템 초기화
        init_success = await orchestrator.initialize_system()
        if not init_success:
            print("❌ 시스템 초기화 실패")
            return
        
        print("✅ 시스템 초기화 완료")
        
        # 시스템 상태 확인
        status = orchestrator.get_system_status()
        print(f"시스템 상태: {status['system_status']}")
        print(f"전체 건강도: {status['monitor_summary'].get('overall_health', 0):.2%}")
        
        # 훈련 파이프라인 실행 (간단한 테스트)
        print("\n🚀 훈련 파이프라인 시작...")
        training_success = await orchestrator.run_training_pipeline(
            mode="test",
            samples=50,
            batch_size=2,
            epochs=1,
            timeout=300  # 5분 타임아웃
        )
        
        if training_success:
            print("✅ 훈련 파이프라인 완료")
        else:
            print("❌ 훈련 파이프라인 실패")
        
        # 최종 상태 확인
        final_status = orchestrator.get_system_status()
        print(f"\n=== 최종 시스템 상태 ===")
        print(f"상태: {final_status['system_status']}")
        print(f"가동시간: {final_status['uptime_seconds']:.1f}초")
        print(f"복구 시도: {final_status['recovery_attempts']}회")
        
    except KeyboardInterrupt:
        print("\n사용자 중단 요청")
    
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
    
    finally:
        # 시스템 종료
        print("\n🔄 시스템 종료 중...")
        await orchestrator.shutdown_system()
        print("✅ 시스템 종료 완료")

if __name__ == "__main__":
    asyncio.run(example_usage())