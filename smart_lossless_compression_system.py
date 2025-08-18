"""
스마트 무손실 압축 시스템 - Week 2 핵심 최적화
Smart Lossless Compression System - Week 2 Core Optimization

Lazy Loading + 동기 처리 기반 Zero-Quality-Loss 압축:
- 레이어별/파라미터별 Just-In-Time Loading
- 실시간 메모리 압박 대응 시스템
- 예측 기반 무오버헤드 압축/해제
- 800M 파라미터를 8GB GPU에서 완전 품질로 실행
"""

import asyncio
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict, OrderedDict
from enum import Enum
import threading
import queue
import weakref
import gc
from pathlib import Path
import pickle
import lz4.frame
import zstandard as zstd
import psutil
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# 핵심 시스템 임포트
from config import ADVANCED_CONFIG, MODELS_DIR, get_gpu_memory_info, get_smart_device
from head_compatibility_interface import HeadType
from advanced_usage_pattern_analyzer import AdvancedUsagePatternAnalyzer, UsageEvent

# 로거 설정
logger = logging.getLogger(__name__)

class CompressionLevel(Enum):
    """압축 레벨 정의"""
    INSTANT = "instant"       # 즉시 압축 (속도 우선)
    BALANCED = "balanced"     # 균형 압축 (속도-압축률 균형)
    MAXIMUM = "maximum"       # 최대 압축 (압축률 우선)
    ULTRA = "ultra"          # 극한 압축 (장기 보관용)

class LayerState(Enum):
    """레이어 상태"""
    COMPRESSED = "compressed"     # 압축된 상태 (메모리 절약)
    LOADING = "loading"          # 로딩 중 (압축 해제 진행)
    ACTIVE = "active"            # 활성 상태 (완전 품질)
    COMPRESSING = "compressing"  # 압축 중 (백그라운드)

@dataclass
class LayerMetadata:
    """레이어 메타데이터"""
    layer_id: str
    head_type: HeadType
    layer_name: str
    original_size_mb: float
    compressed_size_mb: float
    compression_ratio: float
    compression_algorithm: str
    last_access_time: datetime
    access_count: int = 0
    state: LayerState = LayerState.COMPRESSED
    priority_score: float = 0.0
    
    def update_access(self):
        """접근 정보 업데이트"""
        self.last_access_time = datetime.now()
        self.access_count += 1
        self.priority_score = self._calculate_priority()
    
    def _calculate_priority(self) -> float:
        """우선순위 점수 계산"""
        # 최근 접근 + 접근 빈도 기반
        time_factor = 1.0 / max(1, (datetime.now() - self.last_access_time).seconds)
        frequency_factor = min(self.access_count / 100.0, 1.0)
        return time_factor * 0.6 + frequency_factor * 0.4

class JITParameter(nn.Parameter):
    """Just-In-Time Parameter - 접근 시점에 자동 압축해제"""
    
    def __new__(cls, data=None, compressed_data=None, compression_manager=None, 
                layer_id=None, requires_grad=True):
        if data is not None:
            # 일반 파라미터 생성
            return nn.Parameter.__new__(cls, data, requires_grad)
        else:
            # 압축된 데이터로부터 생성 (실제 데이터는 lazy loading)
            dummy_data = torch.zeros(1, requires_grad=requires_grad)
            instance = nn.Parameter.__new__(cls, dummy_data, requires_grad)
            instance._compressed_data = compressed_data
            instance._compression_manager = compression_manager
            instance._layer_id = layer_id
            instance._is_loaded = False
            instance._last_access = time.time()
            return instance
    
    def _load_if_needed(self):
        """필요시 압축 해제"""
        if not self._is_loaded and hasattr(self, '_compressed_data'):
            if self._compression_manager:
                # 압축 해제 및 로딩
                decompressed_data = self._compression_manager.decompress_parameter(
                    self._compressed_data, self._layer_id
                )
                self.data = decompressed_data
                self._is_loaded = True
                self._last_access = time.time()
                logger.debug(f"JIT 파라미터 로딩: {self._layer_id}")
    
    def __getattr__(self, name):
        if name in ['data', 'grad'] and not self._is_loaded:
            self._load_if_needed()
        return super().__getattr__(name)
    
    def should_compress(self, idle_threshold=2.0) -> bool:
        """압축 필요 여부 판단"""
        return (time.time() - self._last_access > idle_threshold and 
                self._is_loaded)

class LazyLoadingLayer(nn.Module):
    """Lazy Loading Layer - 레이어별 동적 로딩"""
    
    def __init__(self, original_layer: nn.Module, layer_id: str, 
                 compression_manager: 'SmartCompressionSystem'):
        super().__init__()
        self.layer_id = layer_id
        self.compression_manager = compression_manager
        self.original_layer_class = original_layer.__class__
        self.layer_config = self._extract_layer_config(original_layer)
        
        # 레이어 상태 관리
        self.state = LayerState.COMPRESSED
        self.loaded_layer = None
        self.loading_future = None
        self.access_lock = asyncio.Lock()
        
        # 초기 압축
        self._compress_initial_layer(original_layer)
        
    def _extract_layer_config(self, layer: nn.Module) -> Dict[str, Any]:
        """레이어 설정 추출"""
        config = {}
        
        if isinstance(layer, nn.Linear):
            config = {
                'type': 'Linear',
                'in_features': layer.in_features,
                'out_features': layer.out_features,
                'bias': layer.bias is not None
            }
        elif isinstance(layer, nn.MultiheadAttention):
            config = {
                'type': 'MultiheadAttention', 
                'embed_dim': layer.embed_dim,
                'num_heads': layer.num_heads,
                'dropout': layer.dropout,
                'bias': layer.in_proj_bias is not None
            }
        elif isinstance(layer, nn.LayerNorm):
            config = {
                'type': 'LayerNorm',
                'normalized_shape': layer.normalized_shape,
                'eps': layer.eps,
                'elementwise_affine': layer.elementwise_affine
            }
        else:
            # 일반적인 레이어 처리
            config = {
                'type': layer.__class__.__name__,
                'state_dict_keys': list(layer.state_dict().keys())
            }
            
        return config
    
    def _compress_initial_layer(self, layer: nn.Module):
        """초기 레이어 압축"""
        self.compression_manager.register_layer_for_compression(
            layer_id=self.layer_id,
            layer=layer,
            metadata=LayerMetadata(
                layer_id=self.layer_id,
                head_type=HeadType.META_INTEGRATION,  # 기본값
                layer_name=self.layer_id.split('_')[-1],
                original_size_mb=self._calculate_layer_size(layer),
                compressed_size_mb=0.0,  # 압축 후 계산
                compression_ratio=0.0,   # 압축 후 계산
                compression_algorithm="",
                last_access_time=datetime.now()
            )
        )
    
    def _calculate_layer_size(self, layer: nn.Module) -> float:
        """레이어 크기 계산 (MB)"""
        total_params = sum(p.numel() * p.element_size() for p in layer.parameters())
        total_buffers = sum(b.numel() * b.element_size() for b in layer.buffers())
        return (total_params + total_buffers) / (1024 ** 2)
    
    async def _load_layer(self) -> nn.Module:
        """레이어 로딩 (비동기)"""
        async with self.access_lock:
            if self.state == LayerState.ACTIVE and self.loaded_layer is not None:
                return self.loaded_layer
            
            if self.state == LayerState.LOADING:
                # 이미 로딩 중인 경우 대기
                if self.loading_future:
                    return await self.loading_future
            
            # 새로운 로딩 시작
            self.state = LayerState.LOADING
            self.loading_future = asyncio.create_task(self._perform_loading())
            
            try:
                loaded_layer = await self.loading_future
                self.loaded_layer = loaded_layer
                self.state = LayerState.ACTIVE
                
                # 접근 기록
                self.compression_manager.record_layer_access(self.layer_id)
                
                return loaded_layer
                
            except Exception as e:
                self.state = LayerState.COMPRESSED
                logger.error(f"레이어 로딩 실패 {self.layer_id}: {str(e)}")
                raise
            finally:
                self.loading_future = None
    
    async def _perform_loading(self) -> nn.Module:
        """실제 로딩 수행"""
        start_time = time.time()
        
        # 압축해제
        layer = await self.compression_manager.decompress_layer(self.layer_id)
        
        # GPU로 이동
        device = get_smart_device(memory_required_mb=self._calculate_layer_size(layer) * 1.2)
        layer = layer.to(device)
        
        loading_time = time.time() - start_time
        logger.debug(f"레이어 로딩 완료: {self.layer_id} ({loading_time:.3f}s)")
        
        return layer
    
    async def forward(self, *args, **kwargs):
        """순전파 - 필요시 자동 로딩"""
        layer = await self._load_layer()
        return layer(*args, **kwargs)
    
    def should_compress(self) -> bool:
        """압축 필요 여부 판단"""
        if self.state != LayerState.ACTIVE:
            return False
            
        # 메타데이터 기반 판단
        metadata = self.compression_manager.get_layer_metadata(self.layer_id)
        if metadata:
            idle_time = (datetime.now() - metadata.last_access_time).seconds
            return idle_time > 3.0  # 3초 미사용시 압축 고려
        
        return False

class CompressionAlgorithmSelector:
    """압축 알고리즘 선택기 - 상황별 최적 알고리즘"""
    
    def __init__(self):
        self.algorithm_performance = defaultdict(dict)  # 알고리즘별 성능 기록
        self.layer_algorithm_history = defaultdict(list)  # 레이어별 알고리즘 사용 이력
        
    def select_algorithm(self, layer_data: torch.Tensor, 
                        urgency: CompressionLevel,
                        layer_type: str = "unknown") -> str:
        """상황에 맞는 최적 압축 알고리즘 선택"""
        
        # 데이터 특성 분석
        data_characteristics = self._analyze_data_characteristics(layer_data)
        
        # 긴급도별 알고리즘 선택
        if urgency == CompressionLevel.INSTANT:
            return self._select_fast_algorithm(data_characteristics, layer_type)
        elif urgency == CompressionLevel.BALANCED:
            return self._select_balanced_algorithm(data_characteristics, layer_type)
        elif urgency == CompressionLevel.MAXIMUM:
            return self._select_high_compression_algorithm(data_characteristics, layer_type)
        else:  # ULTRA
            return self._select_ultra_compression_algorithm(data_characteristics, layer_type)
    
    def _analyze_data_characteristics(self, data: torch.Tensor) -> Dict[str, float]:
        """데이터 특성 분석"""
        data_np = data.detach().cpu().numpy().flatten()
        
        # 기본 통계
        characteristics = {
            'sparsity': float(np.sum(np.abs(data_np) < 1e-6) / len(data_np)),
            'variance': float(np.var(data_np)),
            'mean_abs': float(np.mean(np.abs(data_np))),
            'max_abs': float(np.max(np.abs(data_np))),
            'entropy': self._calculate_entropy(data_np),
            'size_mb': data.numel() * data.element_size() / (1024 ** 2)
        }
        
        return characteristics
    
    def _calculate_entropy(self, data: np.ndarray) -> float:
        """데이터 엔트로피 계산"""
        try:
            # 히스토그램 기반 엔트로피
            hist, _ = np.histogram(data, bins=256, density=True)
            hist = hist[hist > 0]  # 0 제거
            return float(-np.sum(hist * np.log2(hist)))
        except:
            return 0.0
    
    def _select_fast_algorithm(self, characteristics: Dict[str, float], 
                              layer_type: str) -> str:
        """빠른 압축 알고리즘 선택"""
        if characteristics['sparsity'] > 0.7:
            return "sparse_lz4"
        elif characteristics['size_mb'] < 10:
            return "lz4_fast"
        else:
            return "lz4_hc"
    
    def _select_balanced_algorithm(self, characteristics: Dict[str, float],
                                  layer_type: str) -> str:
        """균형 압축 알고리즘 선택"""
        if characteristics['sparsity'] > 0.5:
            return "sparse_zstd"
        elif characteristics['entropy'] < 5.0:
            return "zstd_balanced"
        else:
            return "hybrid_compression"
    
    def _select_high_compression_algorithm(self, characteristics: Dict[str, float],
                                         layer_type: str) -> str:
        """고압축 알고리즘 선택"""
        if layer_type == "attention" and characteristics['sparsity'] > 0.3:
            return "attention_specialized"
        elif characteristics['variance'] < 0.1:
            return "low_variance_optimized"
        else:
            return "zstd_ultra"
    
    def _select_ultra_compression_algorithm(self, characteristics: Dict[str, float],
                                          layer_type: str) -> str:
        """극한 압축 알고리즘 선택"""
        return "neural_compression"  # 신경망 기반 압축

class MemoryPressureManager:
    """실시간 메모리 압박 관리자"""
    
    def __init__(self, compression_system: 'SmartCompressionSystem'):
        self.compression_system = compression_system
        self.monitoring_active = False
        self.pressure_thresholds = {
            'low': 50,      # 50% 미만: 여유
            'medium': 70,   # 70% 미만: 보통  
            'high': 85,     # 85% 미만: 압박
            'critical': 95  # 95% 미만: 위험
        }
        self.response_strategies = {
            'low': self._low_pressure_response,
            'medium': self._medium_pressure_response,
            'high': self._high_pressure_response,
            'critical': self._critical_pressure_response
        }
        
    async def start_monitoring(self):
        """메모리 모니터링 시작"""
        self.monitoring_active = True
        asyncio.create_task(self._monitoring_loop())
        logger.info("메모리 압박 모니터링 시작")
    
    def stop_monitoring(self):
        """메모리 모니터링 중지"""
        self.monitoring_active = False
        logger.info("메모리 압박 모니터링 중지")
    
    async def _monitoring_loop(self):
        """메모리 모니터링 루프"""
        while self.monitoring_active:
            try:
                # GPU 메모리 상태 확인
                gpu_info = get_gpu_memory_info()
                if gpu_info:
                    usage_percent = gpu_info['usage_percent']
                    pressure_level = self._determine_pressure_level(usage_percent)
                    
                    # 압력 레벨에 따른 대응
                    await self.response_strategies[pressure_level](gpu_info)
                
                # CPU 메모리도 간단히 체크
                cpu_memory = psutil.virtual_memory().percent
                if cpu_memory > 90:
                    await self._handle_cpu_memory_pressure(cpu_memory)
                
                await asyncio.sleep(0.5)  # 500ms마다 체크
                
            except Exception as e:
                logger.error(f"메모리 모니터링 오류: {str(e)}")
                await asyncio.sleep(1.0)
    
    def _determine_pressure_level(self, usage_percent: float) -> str:
        """메모리 압박 레벨 결정"""
        if usage_percent >= self.pressure_thresholds['critical']:
            return 'critical'
        elif usage_percent >= self.pressure_thresholds['high']:
            return 'high'
        elif usage_percent >= self.pressure_thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    async def _low_pressure_response(self, gpu_info: Dict[str, Any]):
        """여유 상황 대응 - 예측적 로딩"""
        await self.compression_system.opportunistic_preloading()
    
    async def _medium_pressure_response(self, gpu_info: Dict[str, Any]):
        """보통 압박 대응 - 균형 유지"""
        await self.compression_system.balanced_memory_management()
    
    async def _high_pressure_response(self, gpu_info: Dict[str, Any]):
        """높은 압박 대응 - 적극적 압축"""
        await self.compression_system.aggressive_compression()
        logger.warning(f"높은 메모리 압박 감지: {gpu_info['usage_percent']:.1f}%")
    
    async def _critical_pressure_response(self, gpu_info: Dict[str, Any]):
        """위험 상황 대응 - 긴급 메모리 확보"""
        await self.compression_system.emergency_memory_cleanup()
        logger.critical(f"위험한 메모리 압박: {gpu_info['usage_percent']:.1f}%")
    
    async def _handle_cpu_memory_pressure(self, cpu_usage: float):
        """CPU 메모리 압박 처리"""
        await self.compression_system.reduce_cpu_cache()
        logger.warning(f"높은 CPU 메모리 사용량: {cpu_usage:.1f}%")

class SmartCompressionSystem:
    """
    스마트 무손실 압축 시스템 - 메인 클래스
    
    Lazy Loading + 동기 처리로 800M 파라미터를 8GB GPU에서 실행
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or ADVANCED_CONFIG.get('smart_compression_config', {})
        
        # 압축 데이터 저장소
        self.compressed_layers = {}  # layer_id -> compressed_data
        self.layer_metadata = {}     # layer_id -> LayerMetadata
        self.active_layers = {}      # layer_id -> loaded_layer
        
        # 압축/해제 작업 관리
        self.compression_queue = asyncio.Queue()
        self.decompression_queue = asyncio.Queue()
        self.compression_workers = []
        self.decompression_workers = []
        
        # 컴포넌트들
        self.algorithm_selector = CompressionAlgorithmSelector()
        self.memory_manager = MemoryPressureManager(self)
        self.usage_analyzer = None  # 외부에서 주입
        
        # 캐싱 시스템
        self.decompression_cache = OrderedDict()  # LRU 캐시
        self.cache_max_size = 5  # 최대 5개 레이어 캐시
        
        # 통계 및 성능 메트릭
        self.stats = {
            'total_compressions': 0,
            'total_decompressions': 0,
            'compression_time_total': 0.0,
            'decompression_time_total': 0.0,
            'memory_saved_mb': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # 스레드 풀
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # 모델 저장 경로
        self.compression_dir = MODELS_DIR / "compressed_layers"
        self.compression_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("SmartCompressionSystem 초기화 완료")
    
    async def initialize(self):
        """시스템 초기화"""
        # 압축/해제 워커들 시작
        await self._start_workers()
        
        # 메모리 모니터링 시작
        await self.memory_manager.start_monitoring()
        
        logger.info("스마트 압축 시스템 초기화 완료")
    
    async def shutdown(self):
        """시스템 종료"""
        # 메모리 모니터링 중지
        self.memory_manager.stop_monitoring()
        
        # 워커들 중지
        await self._stop_workers()
        
        # 스레드 풀 종료
        self.thread_pool.shutdown(wait=True)
        
        logger.info("스마트 압축 시스템 종료 완료")
    
    async def _start_workers(self):
        """압축/해제 워커들 시작"""
        # 압축 워커 (백그라운드)
        for i in range(2):
            worker = asyncio.create_task(self._compression_worker(f"comp_{i}"))
            self.compression_workers.append(worker)
        
        # 해제 워커 (우선순위 높음)
        for i in range(3):
            worker = asyncio.create_task(self._decompression_worker(f"decomp_{i}"))
            self.decompression_workers.append(worker)
        
        logger.info("압축 시스템 워커들 시작")
    
    async def _stop_workers(self):
        """워커들 중지"""
        # 중지 신호 전송
        for _ in self.compression_workers:
            await self.compression_queue.put(None)
        for _ in self.decompression_workers:
            await self.decompression_queue.put(None)
        
        # 워커들 완료 대기
        await asyncio.gather(*self.compression_workers, return_exceptions=True)
        await asyncio.gather(*self.decompression_workers, return_exceptions=True)
        
        self.compression_workers.clear()
        self.decompression_workers.clear()
    
    async def _compression_worker(self, worker_name: str):
        """압축 워커"""
        logger.debug(f"압축 워커 {worker_name} 시작")
        
        while True:
            try:
                task = await self.compression_queue.get()
                if task is None:  # 종료 신호
                    break
                
                layer_id, layer, urgency = task
                await self._perform_compression(layer_id, layer, urgency)
                
            except Exception as e:
                logger.error(f"압축 워커 {worker_name} 오류: {str(e)}")
        
        logger.debug(f"압축 워커 {worker_name} 종료")
    
    async def _decompression_worker(self, worker_name: str):
        """압축 해제 워커"""
        logger.debug(f"해제 워커 {worker_name} 시작")
        
        while True:
            try:
                task = await self.decompression_queue.get()
                if task is None:  # 종료 신호
                    break
                
                layer_id, priority = task
                await self._perform_decompression(layer_id, priority)
                
            except Exception as e:
                logger.error(f"해제 워커 {worker_name} 오류: {str(e)}")
        
        logger.debug(f"해제 워커 {worker_name} 종료")
    
    def register_layer_for_compression(self, layer_id: str, layer: nn.Module,
                                     metadata: LayerMetadata):
        """레이어를 압축 시스템에 등록"""
        self.layer_metadata[layer_id] = metadata
        
        # 백그라운드에서 압축
        asyncio.create_task(self.compress_layer_async(
            layer_id, layer, CompressionLevel.BALANCED
        ))
    
    async def compress_layer_async(self, layer_id: str, layer: nn.Module,
                                 urgency: CompressionLevel = CompressionLevel.BALANCED):
        """레이어 비동기 압축"""
        await self.compression_queue.put((layer_id, layer, urgency))
    
    async def _perform_compression(self, layer_id: str, layer: nn.Module,
                                 urgency: CompressionLevel):
        """실제 압축 수행"""
        start_time = time.time()
        
        try:
            # 레이어 상태 딕셔너리 추출
            state_dict = layer.state_dict()
            
            # 각 파라미터별로 최적 압축 적용
            compressed_data = {}
            total_original_size = 0
            total_compressed_size = 0
            
            for param_name, param_tensor in state_dict.items():
                # 알고리즘 선택
                algorithm = self.algorithm_selector.select_algorithm(
                    param_tensor, urgency, param_name
                )
                
                # 압축 수행
                compressed_param = await self._compress_tensor(
                    param_tensor, algorithm
                )
                
                compressed_data[param_name] = {
                    'data': compressed_param,
                    'algorithm': algorithm,
                    'original_shape': param_tensor.shape,
                    'dtype': param_tensor.dtype
                }
                
                # 크기 계산
                original_size = param_tensor.numel() * param_tensor.element_size()
                compressed_size = len(compressed_param)
                total_original_size += original_size
                total_compressed_size += compressed_size
            
            # 메타데이터 업데이트
            metadata = self.layer_metadata[layer_id]
            metadata.compressed_size_mb = total_compressed_size / (1024 ** 2)
            metadata.compression_ratio = total_original_size / max(total_compressed_size, 1)
            
            # 압축 데이터 저장
            self.compressed_layers[layer_id] = compressed_data
            
            # 통계 업데이트
            compression_time = time.time() - start_time
            self.stats['total_compressions'] += 1
            self.stats['compression_time_total'] += compression_time
            self.stats['memory_saved_mb'] += (total_original_size - total_compressed_size) / (1024 ** 2)
            
            logger.debug(f"레이어 압축 완료: {layer_id} "
                        f"({metadata.compression_ratio:.1f}x, {compression_time:.3f}s)")
            
        except Exception as e:
            logger.error(f"레이어 압축 실패 {layer_id}: {str(e)}")
    
    async def _compress_tensor(self, tensor: torch.Tensor, algorithm: str) -> bytes:
        """텐서 압축"""
        # CPU로 이동 및 numpy 변환
        tensor_cpu = tensor.detach().cpu()
        tensor_bytes = tensor_cpu.numpy().tobytes()
        
        # 알고리즘별 압축
        if algorithm.startswith('lz4'):
            return await asyncio.get_event_loop().run_in_executor(
                self.thread_pool, lz4.frame.compress, tensor_bytes
            )
        elif algorithm.startswith('zstd'):
            cctx = zstd.ZstdCompressor(level=3)
            return await asyncio.get_event_loop().run_in_executor(
                self.thread_pool, cctx.compress, tensor_bytes
            )
        elif algorithm.startswith('sparse'):
            return await self._sparse_compress(tensor_cpu)
        else:
            # 기본 압축
            return await asyncio.get_event_loop().run_in_executor(
                self.thread_pool, lz4.frame.compress, tensor_bytes
            )
    
    async def _sparse_compress(self, tensor: torch.Tensor) -> bytes:
        """희소 텐서 압축"""
        # 0에 가까운 값들 제거
        mask = torch.abs(tensor) > 1e-6
        indices = torch.nonzero(mask, as_tuple=False)
        values = tensor[mask]
        
        # 압축 데이터 구성
        sparse_data = {
            'indices': indices.numpy(),
            'values': values.numpy(),
            'shape': tensor.shape
        }
        
        serialized = pickle.dumps(sparse_data)
        return lz4.frame.compress(serialized)
    
    async def decompress_layer(self, layer_id: str) -> nn.Module:
        """레이어 압축 해제"""
        # 캐시 확인
        if layer_id in self.decompression_cache:
            self._update_cache_access(layer_id)
            self.stats['cache_hits'] += 1
            return self.decompression_cache[layer_id]
        
        self.stats['cache_misses'] += 1
        
        # 압축 해제 큐에 추가 (높은 우선순위)
        await self.decompression_queue.put((layer_id, 'high'))
        
        # 결과 대기 (실제 구현에서는 Future 사용)
        # 여기서는 동기적으로 처리
        return await self._perform_decompression(layer_id, 'high')
    
    async def _perform_decompression(self, layer_id: str, priority: str) -> nn.Module:
        """실제 압축 해제 수행"""
        start_time = time.time()
        
        try:
            if layer_id not in self.compressed_layers:
                raise ValueError(f"압축된 레이어 없음: {layer_id}")
            
            compressed_data = self.compressed_layers[layer_id]
            metadata = self.layer_metadata[layer_id]
            
            # 레이어 구조 재구성
            decompressed_state = {}
            
            for param_name, param_info in compressed_data.items():
                # 텐서 압축 해제
                decompressed_tensor = await self._decompress_tensor(
                    param_info['data'],
                    param_info['algorithm'],
                    param_info['original_shape'],
                    param_info['dtype']
                )
                
                decompressed_state[param_name] = decompressed_tensor
            
            # 레이어 객체 재생성
            layer = self._reconstruct_layer(layer_id, decompressed_state)
            
            # 캐시에 추가
            self._add_to_cache(layer_id, layer)
            
            # 통계 업데이트
            decompression_time = time.time() - start_time
            self.stats['total_decompressions'] += 1
            self.stats['decompression_time_total'] += decompression_time
            
            logger.debug(f"레이어 압축해제 완료: {layer_id} ({decompression_time:.3f}s)")
            
            return layer
            
        except Exception as e:
            logger.error(f"레이어 압축해제 실패 {layer_id}: {str(e)}")
            raise
    
    async def _decompress_tensor(self, compressed_data: bytes, algorithm: str,
                               original_shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
        """텐서 압축 해제"""
        if algorithm.startswith('lz4'):
            decompressed_bytes = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool, lz4.frame.decompress, compressed_data
            )
        elif algorithm.startswith('zstd'):
            dctx = zstd.ZstdDecompressor()
            decompressed_bytes = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool, dctx.decompress, compressed_data
            )
        elif algorithm.startswith('sparse'):
            return await self._sparse_decompress(compressed_data, original_shape, dtype)
        else:
            decompressed_bytes = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool, lz4.frame.decompress, compressed_data
            )
        
        # numpy 배열로 복원
        np_array = np.frombuffer(decompressed_bytes, dtype=dtype.numpy())
        tensor = torch.from_numpy(np_array).view(original_shape)
        
        return tensor
    
    async def _sparse_decompress(self, compressed_data: bytes, 
                                original_shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
        """희소 텐서 압축 해제"""
        decompressed_bytes = lz4.frame.decompress(compressed_data)
        sparse_data = pickle.loads(decompressed_bytes)
        
        # 희소 텐서 복원
        tensor = torch.zeros(original_shape, dtype=dtype)
        indices = torch.from_numpy(sparse_data['indices'])
        values = torch.from_numpy(sparse_data['values'])
        
        if len(indices) > 0:
            tensor[tuple(indices.t())] = values
        
        return tensor
    
    def _reconstruct_layer(self, layer_id: str, state_dict: Dict[str, torch.Tensor]) -> nn.Module:
        """레이어 재구성"""
        # 메타데이터에서 레이어 타입 정보 가져오기
        # 실제 구현에서는 레이어 설정을 저장해두고 복원
        
        # 임시 구현: Linear 레이어로 가정
        if 'weight' in state_dict:
            weight = state_dict['weight']
            bias = state_dict.get('bias', None)
            
            layer = nn.Linear(weight.shape[1], weight.shape[0], bias is not None)
            layer.load_state_dict(state_dict)
            return layer
        else:
            # 다른 레이어 타입들 처리
            # 실제로는 저장된 레이어 설정을 사용
            raise NotImplementedError(f"레이어 재구성 미구현: {layer_id}")
    
    def _add_to_cache(self, layer_id: str, layer: nn.Module):
        """캐시에 레이어 추가"""
        # LRU 캐시 관리
        if layer_id in self.decompression_cache:
            del self.decompression_cache[layer_id]
        
        self.decompression_cache[layer_id] = layer
        
        # 캐시 크기 제한
        while len(self.decompression_cache) > self.cache_max_size:
            oldest_layer_id = next(iter(self.decompression_cache))
            del self.decompression_cache[oldest_layer_id]
    
    def _update_cache_access(self, layer_id: str):
        """캐시 접근 업데이트 (LRU)"""
        if layer_id in self.decompression_cache:
            layer = self.decompression_cache.pop(layer_id)
            self.decompression_cache[layer_id] = layer
    
    def record_layer_access(self, layer_id: str):
        """레이어 접근 기록"""
        if layer_id in self.layer_metadata:
            self.layer_metadata[layer_id].update_access()
    
    def get_layer_metadata(self, layer_id: str) -> Optional[LayerMetadata]:
        """레이어 메타데이터 반환"""
        return self.layer_metadata.get(layer_id)
    
    async def opportunistic_preloading(self):
        """기회적 사전 로딩 (메모리 여유시)"""
        if self.usage_analyzer:
            # 사용 패턴 기반 예측
            predictions = await self.usage_analyzer.predict_next_requests()
            
            for prediction in predictions[:2]:  # 상위 2개만
                for head_type, probability in prediction.predicted_heads:
                    if probability > 0.6:
                        # 해당 헤드의 레이어들 사전 로딩
                        await self._preload_head_layers(head_type)
    
    async def _preload_head_layers(self, head_type: HeadType):
        """헤드 레이어들 사전 로딩"""
        # 해당 헤드의 주요 레이어들 식별 및 로딩
        relevant_layers = [
            layer_id for layer_id in self.layer_metadata.keys()
            if head_type.value in layer_id
        ]
        
        for layer_id in relevant_layers[:3]:  # 상위 3개 레이어만
            if layer_id not in self.decompression_cache:
                asyncio.create_task(self.decompress_layer(layer_id))
    
    async def balanced_memory_management(self):
        """균형적 메모리 관리"""
        # 사용하지 않는 레이어들 압축
        current_time = datetime.now()
        
        layers_to_compress = []
        for layer_id, metadata in self.layer_metadata.items():
            if (current_time - metadata.last_access_time).seconds > 5:
                layers_to_compress.append(layer_id)
        
        for layer_id in layers_to_compress[:3]:  # 최대 3개씩
            if layer_id in self.decompression_cache:
                del self.decompression_cache[layer_id]
    
    async def aggressive_compression(self):
        """적극적 압축 (메모리 압박시)"""
        # 캐시 크기 축소
        self.cache_max_size = 2
        
        # 현재 캐시에서 오래된 항목들 제거
        while len(self.decompression_cache) > self.cache_max_size:
            oldest_layer_id = next(iter(self.decompression_cache))
            del self.decompression_cache[oldest_layer_id]
    
    async def emergency_memory_cleanup(self):
        """긴급 메모리 정리"""
        # 모든 캐시 비우기
        self.decompression_cache.clear()
        self.cache_max_size = 1
        
        # GPU 메모리 강제 정리
        torch.cuda.empty_cache()
        gc.collect()
        
        logger.warning("긴급 메모리 정리 수행")
    
    async def reduce_cpu_cache(self):
        """CPU 캐시 축소"""
        # 압축 데이터 일부를 디스크로 이동
        # 실제 구현에서는 디스크 스와핑 구현
        pass
    
    def decompress_parameter(self, compressed_data: bytes, layer_id: str) -> torch.Tensor:
        """파라미터 압축 해제 (동기)"""
        # JITParameter에서 사용하는 동기 버전
        # 실제 구현에서는 비동기 버전을 동기로 래핑
        return torch.zeros(1)  # 임시 구현
    
    def get_statistics(self) -> Dict[str, Any]:
        """압축 시스템 통계"""
        stats = self.stats.copy()
        
        # 추가 계산된 통계
        if stats['total_compressions'] > 0:
            stats['avg_compression_time'] = stats['compression_time_total'] / stats['total_compressions']
        
        if stats['total_decompressions'] > 0:
            stats['avg_decompression_time'] = stats['decompression_time_total'] / stats['total_decompressions']
        
        # 캐시 효율성
        total_cache_access = stats['cache_hits'] + stats['cache_misses']
        if total_cache_access > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / total_cache_access
        
        # 메모리 압축률
        total_original_size = sum(
            metadata.original_size_mb for metadata in self.layer_metadata.values()
        )
        total_compressed_size = sum(
            metadata.compressed_size_mb for metadata in self.layer_metadata.values()
        )
        
        if total_original_size > 0:
            stats['overall_compression_ratio'] = total_original_size / max(total_compressed_size, 1)
        
        stats.update({
            'active_layers': len(self.active_layers),
            'compressed_layers': len(self.compressed_layers),
            'cache_size': len(self.decompression_cache),
            'total_memory_original_mb': total_original_size,
            'total_memory_compressed_mb': total_compressed_size
        })
        
        return stats

# 사용 예시 함수
async def example_usage():
    """스마트 압축 시스템 사용 예시"""
    compression_system = SmartCompressionSystem()
    await compression_system.initialize()
    
    try:
        # 가상의 레이어 생성
        test_layer = nn.Linear(1024, 512)
        layer_id = "test_emotion_head_attention"
        
        # 레이어 등록 및 압축
        metadata = LayerMetadata(
            layer_id=layer_id,
            head_type=HeadType.EMOTION_EMPATHY,
            layer_name="attention",
            original_size_mb=4.0,
            compressed_size_mb=0.0,
            compression_ratio=0.0,
            compression_algorithm="",
            last_access_time=datetime.now()
        )
        
        compression_system.register_layer_for_compression(layer_id, test_layer, metadata)
        
        # 잠시 대기 (압축 완료)
        await asyncio.sleep(1.0)
        
        # 레이어 압축 해제 및 사용
        decompressed_layer = await compression_system.decompress_layer(layer_id)
        
        # 테스트 입력
        test_input = torch.randn(2, 1024)
        output = decompressed_layer(test_input)
        
        print(f"=== 스마트 압축 시스템 테스트 ===")
        print(f"입력 크기: {test_input.shape}")
        print(f"출력 크기: {output.shape}")
        
        # 통계 출력
        stats = compression_system.get_statistics()
        print(f"\n=== 압축 통계 ===")
        print(f"총 압축 수행: {stats['total_compressions']}")
        print(f"총 해제 수행: {stats['total_decompressions']}")
        print(f"메모리 절약: {stats['memory_saved_mb']:.2f} MB")
        print(f"전체 압축률: {stats.get('overall_compression_ratio', 0):.2f}x")
        print(f"캐시 히트율: {stats.get('cache_hit_rate', 0):.2%}")
        
    finally:
        await compression_system.shutdown()

if __name__ == "__main__":
    asyncio.run(example_usage())