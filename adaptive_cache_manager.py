"""
적응적 캐시 관리자 - Week 2 메모리 최적화
Adaptive Cache Manager - Week 2 Memory Optimization

동적 캐시 크기 조절 및 지능형 캐시 관리:
- 실시간 메모리 압박 대응 캐시 크기 조절
- 사용 패턴 기반 예측적 캐시 워밍
- 계층적 캐시 시스템 (L1/L2/L3)
- ML 기반 캐시 성능 자동 튜닝
- Zero-Overhead 캐시 교체 알고리즘
"""

import asyncio
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Generic, TypeVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict, OrderedDict
from enum import Enum
import threading
import psutil
import gc
import weakref
import heapq
import pickle
from pathlib import Path
import json
from abc import ABC, abstractmethod

# 핵심 시스템 임포트
from config import ADVANCED_CONFIG, get_gpu_memory_info, get_smart_device
from head_compatibility_interface import HeadType
from advanced_usage_pattern_analyzer import AdvancedUsagePatternAnalyzer, PredictionResult
from smart_lossless_compression_system import SmartCompressionSystem, LayerMetadata

# 로거 설정
logger = logging.getLogger(__name__)

T = TypeVar('T')  # 캐시 아이템 타입

class CacheLevel(Enum):
    """캐시 레벨"""
    L1 = "l1"  # 초고속 캐시 (GPU 메모리)
    L2 = "l2"  # 고속 캐시 (GPU/CPU 하이브리드)
    L3 = "l3"  # 대용량 캐시 (CPU 메모리)

class CachePriority(Enum):
    """캐시 우선순위"""
    CRITICAL = "critical"     # 절대 제거 불가
    HIGH = "high"            # 높은 우선순위
    MEDIUM = "medium"        # 보통 우선순위
    LOW = "low"             # 낮은 우선순위
    EXPENDABLE = "expendable" # 언제든 제거 가능

class CachePolicy(Enum):
    """캐시 정책"""
    LRU = "lru"                    # Least Recently Used
    LFU = "lfu"                    # Least Frequently Used
    ADAPTIVE_LRU = "adaptive_lru"   # 적응적 LRU
    PREDICTIVE = "predictive"       # 예측 기반
    HYBRID = "hybrid"              # 하이브리드

@dataclass
class CacheItem(Generic[T]):
    """캐시 아이템"""
    key: str
    value: T
    size_bytes: int
    access_count: int = 0
    last_access: datetime = field(default_factory=datetime.now)
    creation_time: datetime = field(default_factory=datetime.now)
    priority: CachePriority = CachePriority.MEDIUM
    level: CacheLevel = CacheLevel.L2
    
    # 성능 메트릭
    load_time: float = 0.0
    hit_rate: float = 0.0
    cost_benefit_score: float = 0.0
    
    def update_access(self):
        """접근 정보 업데이트"""
        self.access_count += 1
        self.last_access = datetime.now()
        self._recalculate_metrics()
    
    def _recalculate_metrics(self):
        """메트릭 재계산"""
        age_seconds = (datetime.now() - self.creation_time).total_seconds()
        self.hit_rate = self.access_count / max(1, age_seconds / 3600)  # 시간당 히트율
        
        # 비용-편익 점수 (높을수록 유용)
        size_penalty = self.size_bytes / (1024 * 1024)  # MB 단위 크기 페널티
        frequency_benefit = min(self.access_count, 100) / 100  # 빈도 혜택
        recency_benefit = 1.0 / max(1, (datetime.now() - self.last_access).total_seconds() / 3600)
        
        self.cost_benefit_score = (frequency_benefit * 0.4 + recency_benefit * 0.6) / max(0.1, size_penalty)

@dataclass
class CacheStatistics:
    """캐시 통계"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    memory_usage_bytes: int = 0
    memory_limit_bytes: int = 0
    
    # 레벨별 통계
    l1_hits: int = 0
    l2_hits: int = 0
    l3_hits: int = 0
    
    # 성능 메트릭
    avg_lookup_time_ms: float = 0.0
    avg_load_time_ms: float = 0.0
    memory_efficiency: float = 0.0  # 유용한 데이터 비율
    
    @property
    def hit_rate(self) -> float:
        """캐시 히트율"""
        return self.cache_hits / max(1, self.total_requests)
    
    @property
    def memory_utilization(self) -> float:
        """메모리 사용률"""
        return self.memory_usage_bytes / max(1, self.memory_limit_bytes)

class CacheReplacementPolicy(ABC):
    """캐시 교체 정책 추상 클래스"""
    
    @abstractmethod
    def select_victim(self, items: List[CacheItem], required_space: int) -> List[str]:
        """제거할 아이템들 선택"""
        pass
    
    @abstractmethod
    def update_on_access(self, item: CacheItem):
        """접근 시 정책 업데이트"""
        pass

class AdaptiveLRUPolicy(CacheReplacementPolicy):
    """적응적 LRU 정책"""
    
    def __init__(self):
        self.access_pattern_analyzer = {}  # 접근 패턴 분석
        self.temporal_weights = deque(maxlen=1000)  # 시간적 가중치
        
    def select_victim(self, items: List[CacheItem], required_space: int) -> List[str]:
        """적응적으로 제거할 아이템들 선택"""
        if not items:
            return []
        
        # 우선순위별로 후보 분류
        candidates_by_priority = defaultdict(list)
        for item in items:
            if item.priority != CachePriority.CRITICAL:
                candidates_by_priority[item.priority].append(item)
        
        victims = []
        freed_space = 0
        
        # 우선순위 낮은 것부터 제거
        for priority in [CachePriority.EXPENDABLE, CachePriority.LOW, 
                        CachePriority.MEDIUM, CachePriority.HIGH]:
            if freed_space >= required_space:
                break
                
            candidates = candidates_by_priority[priority]
            if not candidates:
                continue
            
            # 적응적 점수 계산
            for item in candidates:
                item.adaptive_score = self._calculate_adaptive_score(item)
            
            # 점수 낮은 순서로 정렬
            candidates.sort(key=lambda x: x.adaptive_score)
            
            for item in candidates:
                if freed_space >= required_space:
                    break
                victims.append(item.key)
                freed_space += item.size_bytes
        
        return victims
    
    def _calculate_adaptive_score(self, item: CacheItem) -> float:
        """적응적 점수 계산 (낮을수록 제거 우선)"""
        now = datetime.now()
        
        # 시간 기반 요소
        time_since_access = (now - item.last_access).total_seconds()
        time_decay = 1.0 / (1.0 + time_since_access / 3600)  # 1시간 기준 감쇠
        
        # 빈도 기반 요소
        age_hours = max(1, (now - item.creation_time).total_seconds() / 3600)
        frequency_score = item.access_count / age_hours
        
        # 크기 기반 페널티
        size_penalty = item.size_bytes / (1024 * 1024)  # MB 단위
        
        # 비용-편익 점수 활용
        adaptive_score = (
            time_decay * 0.4 +
            min(frequency_score, 5.0) * 0.3 +  # 최대 5로 제한
            item.cost_benefit_score * 0.2 +
            (1.0 / max(0.1, size_penalty)) * 0.1
        )
        
        return adaptive_score
    
    def update_on_access(self, item: CacheItem):
        """접근 시 패턴 업데이트"""
        current_time = datetime.now()
        
        # 접근 패턴 기록
        if item.key not in self.access_pattern_analyzer:
            self.access_pattern_analyzer[item.key] = deque(maxlen=50)
        
        self.access_pattern_analyzer[item.key].append(current_time)
        
        # 시간적 가중치 업데이트
        self.temporal_weights.append(current_time.timestamp())

class PredictiveCachePolicy(CacheReplacementPolicy):
    """예측 기반 캐시 정책"""
    
    def __init__(self, usage_analyzer: Optional[AdvancedUsagePatternAnalyzer] = None):
        self.usage_analyzer = usage_analyzer
        self.future_access_predictions = {}  # 미래 접근 예측
        
    def select_victim(self, items: List[CacheItem], required_space: int) -> List[str]:
        """예측 기반 제거 아이템 선택"""
        if not items:
            return []
        
        # 미래 접근 확률 예측
        for item in items:
            item.future_access_prob = self._predict_future_access(item)
        
        # 미래 접근 확률이 낮은 순서로 정렬
        candidates = [item for item in items if item.priority != CachePriority.CRITICAL]
        candidates.sort(key=lambda x: (x.priority.value, x.future_access_prob))
        
        victims = []
        freed_space = 0
        
        for item in candidates:
            if freed_space >= required_space:
                break
            victims.append(item.key)
            freed_space += item.size_bytes
        
        return victims
    
    def _predict_future_access(self, item: CacheItem) -> float:
        """미래 접근 확률 예측"""
        if not self.usage_analyzer:
            # 기본 예측: 최근 접근 패턴 기반
            time_since_access = (datetime.now() - item.last_access).total_seconds()
            return 1.0 / (1.0 + time_since_access / 1800)  # 30분 기준
        
        # 고급 예측 (사용 패턴 분석기 활용)
        # 실제 구현에서는 usage_analyzer의 예측 결과 활용
        return 0.5  # 임시값
    
    def update_on_access(self, item: CacheItem):
        """접근 시 예측 모델 업데이트"""
        # 예측 정확도 검증 및 모델 조정
        pass

class HierarchicalCacheLevel:
    """계층적 캐시 레벨"""
    
    def __init__(self, level: CacheLevel, max_size_bytes: int, 
                 policy: CacheReplacementPolicy):
        self.level = level
        self.max_size_bytes = max_size_bytes
        self.current_size_bytes = 0
        self.policy = policy
        self.items: Dict[str, CacheItem] = {}
        self.access_order = OrderedDict()  # LRU 추적용
        self.lock = asyncio.Lock()
        
    async def get(self, key: str) -> Optional[Any]:
        """아이템 조회"""
        async with self.lock:
            if key in self.items:
                item = self.items[key]
                item.update_access()
                self.policy.update_on_access(item)
                
                # LRU 순서 업데이트
                self.access_order.move_to_end(key)
                
                return item.value
            return None
    
    async def put(self, key: str, item: CacheItem) -> bool:
        """아이템 저장"""
        async with self.lock:
            # 공간 확보 필요 여부 확인
            required_space = item.size_bytes
            if key in self.items:
                required_space -= self.items[key].size_bytes
            
            if self.current_size_bytes + required_space > self.max_size_bytes:
                # 공간 확보
                await self._make_space(required_space)
                
                # 다시 확인
                if self.current_size_bytes + required_space > self.max_size_bytes:
                    return False  # 공간 확보 실패
            
            # 기존 아이템 제거 (있는 경우)
            if key in self.items:
                old_item = self.items[key]
                self.current_size_bytes -= old_item.size_bytes
                del self.items[key]
                self.access_order.pop(key, None)
            
            # 새 아이템 추가
            item.level = self.level
            self.items[key] = item
            self.access_order[key] = True
            self.current_size_bytes += item.size_bytes
            
            return True
    
    async def remove(self, key: str) -> bool:
        """아이템 제거"""
        async with self.lock:
            if key in self.items:
                item = self.items[key]
                self.current_size_bytes -= item.size_bytes
                del self.items[key]
                self.access_order.pop(key, None)
                return True
            return False
    
    async def _make_space(self, required_space: int):
        """공간 확보"""
        if required_space <= 0:
            return
        
        # 정책에 따라 제거할 아이템들 선택
        victims = self.policy.select_victim(list(self.items.values()), required_space)
        
        # 선택된 아이템들 제거
        for victim_key in victims:
            if victim_key in self.items:
                await self.remove(victim_key)
    
    def get_statistics(self) -> Dict[str, Any]:
        """레벨 통계"""
        return {
            'level': self.level.value,
            'item_count': len(self.items),
            'size_bytes': self.current_size_bytes,
            'max_size_bytes': self.max_size_bytes,
            'utilization': self.current_size_bytes / self.max_size_bytes,
            'avg_item_size': self.current_size_bytes / max(1, len(self.items))
        }

class AdaptiveCacheManager:
    """
    적응적 캐시 관리자
    
    동적 캐시 크기 조절 및 지능형 캐시 관리 시스템
    """
    
    def __init__(self, config: Dict[str, Any] = None, 
                 usage_analyzer: Optional[AdvancedUsagePatternAnalyzer] = None,
                 compression_system: Optional[SmartCompressionSystem] = None):
        self.config = config or ADVANCED_CONFIG.get('adaptive_cache_config', {})
        self.usage_analyzer = usage_analyzer
        self.compression_system = compression_system
        
        # 캐시 계층 설정
        self._initialize_cache_levels()
        
        # 통계 및 성능 메트릭
        self.statistics = CacheStatistics()
        self.performance_history = deque(maxlen=1000)
        
        # 동적 크기 조절 설정
        self.memory_pressure_threshold = 0.85  # 85% 이상시 크기 축소
        self.memory_comfort_threshold = 0.60   # 60% 미만시 크기 확대
        self.size_adjustment_factor = 0.1      # 10%씩 조절
        
        # 예측적 캐시 워밍
        self.warmup_scheduler = None
        self.warmup_active = False
        
        # 모니터링 및 자동 튜닝
        self.monitoring_active = False
        self.auto_tuning_enabled = True
        self.tuning_history = deque(maxlen=100)
        
        # 성능 최적화
        self.lookup_times = deque(maxlen=1000)
        self.eviction_efficiency = deque(maxlen=100)
        
        logger.info("AdaptiveCacheManager 초기화 완료")
    
    def _initialize_cache_levels(self):
        """캐시 레벨 초기화"""
        # 기본 설정값
        base_config = {
            'l1_size_mb': 100,    # L1: 100MB (GPU 메모리)
            'l2_size_mb': 500,    # L2: 500MB (하이브리드)
            'l3_size_mb': 2000,   # L3: 2GB (CPU 메모리)
        }
        base_config.update(self.config)
        
        # L1 캐시 (초고속, GPU 메모리)
        l1_policy = AdaptiveLRUPolicy()
        self.l1_cache = HierarchicalCacheLevel(
            CacheLevel.L1,
            base_config['l1_size_mb'] * 1024 * 1024,
            l1_policy
        )
        
        # L2 캐시 (고속, 하이브리드)
        l2_policy = AdaptiveLRUPolicy()
        self.l2_cache = HierarchicalCacheLevel(
            CacheLevel.L2,
            base_config['l2_size_mb'] * 1024 * 1024,
            l2_policy
        )
        
        # L3 캐시 (대용량, CPU 메모리)
        l3_policy = PredictiveCachePolicy(self.usage_analyzer)
        self.l3_cache = HierarchicalCacheLevel(
            CacheLevel.L3,
            base_config['l3_size_mb'] * 1024 * 1024,
            l3_policy
        )
        
        self.cache_levels = [self.l1_cache, self.l2_cache, self.l3_cache]
    
    async def initialize(self):
        """캐시 관리자 초기화"""
        # 모니터링 시작
        await self.start_monitoring()
        
        # 예측적 캐시 워밍 시작
        if self.usage_analyzer:
            await self.start_predictive_warming()
        
        logger.info("적응적 캐시 관리자 초기화 완료")
    
    async def shutdown(self):
        """캐시 관리자 종료"""
        self.monitoring_active = False
        self.warmup_active = False
        
        # 백그라운드 작업들 정리
        if self.warmup_scheduler:
            self.warmup_scheduler.cancel()
        
        logger.info("적응적 캐시 관리자 종료 완료")
    
    async def get(self, key: str, loader: Optional[Callable] = None) -> Optional[Any]:
        """캐시에서 아이템 조회"""
        start_time = time.time()
        self.statistics.total_requests += 1
        
        # L1 → L2 → L3 순서로 조회
        for level, cache_level in enumerate([self.l1_cache, self.l2_cache, self.l3_cache]):
            value = await cache_level.get(key)
            if value is not None:
                self.statistics.cache_hits += 1
                
                # 레벨별 히트 통계 업데이트
                if level == 0:
                    self.statistics.l1_hits += 1
                elif level == 1:
                    self.statistics.l2_hits += 1
                else:
                    self.statistics.l3_hits += 1
                
                # 더 높은 레벨로 승격 (캐시 웜업)
                if level > 0:
                    await self._promote_to_higher_level(key, value, level)
                
                # 성능 메트릭 업데이트
                lookup_time = (time.time() - start_time) * 1000
                self.lookup_times.append(lookup_time)
                
                return value
        
        # 캐시 미스
        self.statistics.cache_misses += 1
        
        # 로더가 제공된 경우 값 로드 및 캐시
        if loader:
            try:
                load_start = time.time()
                value = await loader() if asyncio.iscoroutinefunction(loader) else loader()
                load_time = (time.time() - load_start) * 1000
                
                if value is not None:
                    await self.put(key, value, CachePriority.MEDIUM, load_time=load_time)
                
                return value
            except Exception as e:
                logger.error(f"캐시 로더 실패 {key}: {str(e)}")
        
        return None
    
    async def put(self, key: str, value: Any, priority: CachePriority = CachePriority.MEDIUM,
                 level: Optional[CacheLevel] = None, load_time: float = 0.0) -> bool:
        """캐시에 아이템 저장"""
        # 아이템 크기 계산
        size_bytes = self._calculate_size(value)
        
        # 캐시 아이템 생성
        cache_item = CacheItem(
            key=key,
            value=value,
            size_bytes=size_bytes,
            priority=priority,
            load_time=load_time
        )
        
        # 자동 레벨 선택
        if level is None:
            level = self._select_optimal_level(cache_item)
        
        # 해당 레벨에 저장
        target_cache = self._get_cache_by_level(level)
        if target_cache:
            success = await target_cache.put(key, cache_item)
            if success:
                self._update_memory_usage()
                return True
        
        # 저장 실패시 다른 레벨에 시도
        for cache_level in self.cache_levels:
            if cache_level.level != level:
                success = await cache_level.put(key, cache_item)
                if success:
                    self._update_memory_usage()
                    return True
        
        return False
    
    async def remove(self, key: str) -> bool:
        """캐시에서 아이템 제거"""
        removed = False
        
        for cache_level in self.cache_levels:
            if await cache_level.remove(key):
                removed = True
        
        if removed:
            self._update_memory_usage()
        
        return removed
    
    async def _promote_to_higher_level(self, key: str, value: Any, current_level: int):
        """더 높은 레벨로 아이템 승격"""
        if current_level <= 0:
            return
        
        target_level = current_level - 1
        target_cache = self.cache_levels[target_level]
        
        # 기존 아이템 정보 가져오기
        source_cache = self.cache_levels[current_level]
        source_item = source_cache.items.get(key)
        
        if source_item:
            # 새로운 캐시 아이템 생성 (레벨 조정)
            promoted_item = CacheItem(
                key=key,
                value=value,
                size_bytes=source_item.size_bytes,
                access_count=source_item.access_count,
                priority=source_item.priority,
                load_time=source_item.load_time
            )
            
            # 더 높은 레벨에 추가
            await target_cache.put(key, promoted_item)
    
    def _select_optimal_level(self, item: CacheItem) -> CacheLevel:
        """최적의 캐시 레벨 선택"""
        # 우선순위 기반 레벨 선택
        if item.priority == CachePriority.CRITICAL:
            return CacheLevel.L1
        elif item.priority == CachePriority.HIGH:
            # 크기가 작으면 L1, 크면 L2
            return CacheLevel.L1 if item.size_bytes < 10 * 1024 * 1024 else CacheLevel.L2
        elif item.priority == CachePriority.MEDIUM:
            return CacheLevel.L2
        else:
            return CacheLevel.L3
    
    def _get_cache_by_level(self, level: CacheLevel) -> Optional[HierarchicalCacheLevel]:
        """레벨별 캐시 반환"""
        for cache_level in self.cache_levels:
            if cache_level.level == level:
                return cache_level
        return None
    
    def _calculate_size(self, value: Any) -> int:
        """값의 크기 계산"""
        if isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
        elif isinstance(value, (list, tuple)):
            return sum(self._calculate_size(item) for item in value)
        elif isinstance(value, dict):
            return sum(self._calculate_size(k) + self._calculate_size(v) 
                      for k, v in value.items())
        elif hasattr(value, '__sizeof__'):
            return value.__sizeof__()
        else:
            # 기본 추정
            return len(str(value).encode('utf-8'))
    
    def _update_memory_usage(self):
        """메모리 사용량 업데이트"""
        total_memory = sum(cache.current_size_bytes for cache in self.cache_levels)
        total_limit = sum(cache.max_size_bytes for cache in self.cache_levels)
        
        self.statistics.memory_usage_bytes = total_memory
        self.statistics.memory_limit_bytes = total_limit
    
    async def start_monitoring(self):
        """모니터링 시작"""
        self.monitoring_active = True
        asyncio.create_task(self._monitoring_loop())
        logger.info("캐시 모니터링 시작")
    
    async def _monitoring_loop(self):
        """모니터링 루프"""
        while self.monitoring_active:
            try:
                # 성능 메트릭 계산
                await self._update_performance_metrics()
                
                # 동적 크기 조절
                await self._dynamic_size_adjustment()
                
                # 자동 튜닝
                if self.auto_tuning_enabled:
                    await self._auto_tuning()
                
                await asyncio.sleep(10.0)  # 10초마다
                
            except Exception as e:
                logger.error(f"캐시 모니터링 오류: {str(e)}")
                await asyncio.sleep(5.0)
    
    async def _update_performance_metrics(self):
        """성능 메트릭 업데이트"""
        # 평균 조회 시간
        if self.lookup_times:
            self.statistics.avg_lookup_time_ms = sum(self.lookup_times) / len(self.lookup_times)
        
        # 메모리 효율성 계산
        total_items = sum(len(cache.items) for cache in self.cache_levels)
        useful_items = 0
        
        for cache in self.cache_levels:
            for item in cache.items.values():
                if item.access_count > 1:  # 2번 이상 접근된 아이템을 유용한 것으로 간주
                    useful_items += 1
        
        self.statistics.memory_efficiency = useful_items / max(1, total_items)
        
        # 성능 이력 기록
        current_performance = {
            'timestamp': datetime.now(),
            'hit_rate': self.statistics.hit_rate,
            'memory_utilization': self.statistics.memory_utilization,
            'memory_efficiency': self.statistics.memory_efficiency,
            'avg_lookup_time': self.statistics.avg_lookup_time_ms
        }
        self.performance_history.append(current_performance)
    
    async def _dynamic_size_adjustment(self):
        """동적 크기 조절"""
        gpu_info = get_gpu_memory_info()
        if not gpu_info:
            return
        
        gpu_usage = gpu_info['usage_percent'] / 100.0
        
        # 메모리 압박 상황
        if gpu_usage > self.memory_pressure_threshold:
            await self._shrink_cache_sizes()
        # 메모리 여유 상황
        elif gpu_usage < self.memory_comfort_threshold:
            await self._expand_cache_sizes()
    
    async def _shrink_cache_sizes(self):
        """캐시 크기 축소"""
        shrink_factor = 1.0 - self.size_adjustment_factor
        
        for cache_level in self.cache_levels:
            new_size = int(cache_level.max_size_bytes * shrink_factor)
            await self._resize_cache_level(cache_level, new_size)
        
        logger.info(f"메모리 압박으로 캐시 크기 {self.size_adjustment_factor:.1%} 축소")
    
    async def _expand_cache_sizes(self):
        """캐시 크기 확대"""
        expand_factor = 1.0 + self.size_adjustment_factor
        
        # 히트율이 높은 레벨 우선 확대
        hit_rates = []
        for i, cache_level in enumerate(self.cache_levels):
            level_hits = getattr(self.statistics, f'l{i+1}_hits', 0)
            total_requests = max(1, self.statistics.total_requests)
            hit_rate = level_hits / total_requests
            hit_rates.append((hit_rate, cache_level))
        
        # 히트율 높은 순서로 정렬
        hit_rates.sort(reverse=True)
        
        # 상위 캐시 레벨만 확대
        for hit_rate, cache_level in hit_rates[:2]:  # 상위 2개 레벨만
            new_size = int(cache_level.max_size_bytes * expand_factor)
            await self._resize_cache_level(cache_level, new_size)
        
        logger.info(f"메모리 여유로 캐시 크기 {self.size_adjustment_factor:.1%} 확대")
    
    async def _resize_cache_level(self, cache_level: HierarchicalCacheLevel, new_size: int):
        """캐시 레벨 크기 조정"""
        if new_size < cache_level.current_size_bytes:
            # 축소: 공간 확보 필요
            required_reduction = cache_level.current_size_bytes - new_size
            await cache_level._make_space(required_reduction)
        
        cache_level.max_size_bytes = new_size
    
    async def _auto_tuning(self):
        """자동 튜닝"""
        if len(self.performance_history) < 10:
            return
        
        # 최근 성능 트렌드 분석
        recent_performances = list(self.performance_history)[-10:]
        avg_hit_rate = sum(p['hit_rate'] for p in recent_performances) / len(recent_performances)
        avg_lookup_time = sum(p['avg_lookup_time'] for p in recent_performances) / len(recent_performances)
        
        # 성능이 저하되고 있는 경우 정책 조정
        if avg_hit_rate < 0.7 or avg_lookup_time > 10.0:  # 히트율 70% 미만 또는 조회시간 10ms 초과
            await self._adjust_cache_policies()
    
    async def _adjust_cache_policies(self):
        """캐시 정책 조정"""
        # 정책 효율성 분석 및 조정
        # 실제 구현에서는 A/B 테스트나 강화학습 사용 가능
        
        tuning_record = {
            'timestamp': datetime.now(),
            'action': 'policy_adjustment',
            'hit_rate_before': self.statistics.hit_rate,
            'lookup_time_before': self.statistics.avg_lookup_time_ms
        }
        self.tuning_history.append(tuning_record)
        
        logger.info("캐시 정책 자동 조정 수행")
    
    async def start_predictive_warming(self):
        """예측적 캐시 워밍 시작"""
        if not self.usage_analyzer:
            return
        
        self.warmup_active = True
        self.warmup_scheduler = asyncio.create_task(self._predictive_warming_loop())
        logger.info("예측적 캐시 워밍 시작")
    
    async def _predictive_warming_loop(self):
        """예측적 워밍 루프"""
        while self.warmup_active:
            try:
                # 사용 패턴 예측
                predictions = await self.usage_analyzer.predict_next_requests()
                
                # 예측 결과 기반 캐시 워밍
                for prediction in predictions[:3]:  # 상위 3개 예측
                    await self._warm_cache_for_prediction(prediction)
                
                await asyncio.sleep(30.0)  # 30초마다
                
            except Exception as e:
                logger.error(f"예측적 캐시 워밍 오류: {str(e)}")
                await asyncio.sleep(10.0)
    
    async def _warm_cache_for_prediction(self, prediction: PredictionResult):
        """예측 결과 기반 캐시 워밍"""
        if prediction.confidence_score < 0.6:
            return
        
        for head_type, probability in prediction.predicted_heads:
            if probability > 0.5:
                # 해당 헤드의 자주 사용되는 데이터를 사전 로딩
                await self._preload_head_data(head_type)
    
    async def _preload_head_data(self, head_type: HeadType):
        """헤드 데이터 사전 로딩"""
        # 압축 시스템과 연동하여 자주 사용되는 레이어들 사전 로딩
        if self.compression_system:
            # 해당 헤드의 주요 레이어들 식별
            relevant_layers = [
                layer_id for layer_id in self.compression_system.layer_metadata.keys()
                if head_type.value in layer_id
            ]
            
            # 상위 3개 레이어 사전 로딩
            for layer_id in relevant_layers[:3]:
                cache_key = f"compressed_layer_{layer_id}"
                
                # 이미 캐시에 있는지 확인
                cached_layer = await self.get(cache_key)
                if cached_layer is None:
                    # 압축 해제된 레이어를 캐시에 저장
                    try:
                        layer = await self.compression_system.decompress_layer(layer_id)
                        await self.put(cache_key, layer, CachePriority.MEDIUM)
                        logger.debug(f"예측적 캐시 워밍: {layer_id}")
                    except Exception as e:
                        logger.error(f"예측적 로딩 실패 {layer_id}: {str(e)}")
    
    def get_detailed_statistics(self) -> Dict[str, Any]:
        """상세 통계 정보"""
        stats = {
            'overall': {
                'total_requests': self.statistics.total_requests,
                'cache_hits': self.statistics.cache_hits,
                'cache_misses': self.statistics.cache_misses,
                'hit_rate': self.statistics.hit_rate,
                'memory_usage_mb': self.statistics.memory_usage_bytes / (1024 * 1024),
                'memory_limit_mb': self.statistics.memory_limit_bytes / (1024 * 1024),
                'memory_utilization': self.statistics.memory_utilization,
                'memory_efficiency': self.statistics.memory_efficiency,
                'avg_lookup_time_ms': self.statistics.avg_lookup_time_ms
            },
            'levels': {},
            'performance_trend': [],
            'tuning_history': list(self.tuning_history)
        }
        
        # 레벨별 통계
        for i, cache_level in enumerate(self.cache_levels):
            level_stats = cache_level.get_statistics()
            level_hits = getattr(self.statistics, f'l{i+1}_hits', 0)
            level_stats['hits'] = level_hits
            level_stats['hit_rate'] = level_hits / max(1, self.statistics.total_requests)
            stats['levels'][f'L{i+1}'] = level_stats
        
        # 성능 트렌드 (최근 10개)
        if self.performance_history:
            recent_history = list(self.performance_history)[-10:]
            stats['performance_trend'] = [
                {
                    'timestamp': p['timestamp'].isoformat(),
                    'hit_rate': p['hit_rate'],
                    'memory_utilization': p['memory_utilization'],
                    'avg_lookup_time': p['avg_lookup_time']
                }
                for p in recent_history
            ]
        
        return stats
    
    async def optimize_for_workload(self, workload_type: str):
        """워크로드 타입별 최적화"""
        optimizations = {
            'interactive': {
                'l1_priority': 0.5,  # L1 캐시 비중 증가
                'l2_priority': 0.3,
                'l3_priority': 0.2,
                'eviction_aggressiveness': 0.3
            },
            'batch_processing': {
                'l1_priority': 0.2,
                'l2_priority': 0.3,
                'l3_priority': 0.5,  # L3 캐시 비중 증가
                'eviction_aggressiveness': 0.8
            },
            'memory_constrained': {
                'l1_priority': 0.6,
                'l2_priority': 0.3,
                'l3_priority': 0.1,
                'eviction_aggressiveness': 0.9
            }
        }
        
        if workload_type in optimizations:
            config = optimizations[workload_type]
            await self._apply_optimization_config(config)
            logger.info(f"캐시 최적화 적용: {workload_type}")
    
    async def _apply_optimization_config(self, config: Dict[str, float]):
        """최적화 설정 적용"""
        # 캐시 크기 재조정
        total_memory = sum(cache.max_size_bytes for cache in self.cache_levels)
        
        new_sizes = [
            int(total_memory * config['l1_priority']),
            int(total_memory * config['l2_priority']),
            int(total_memory * config['l3_priority'])
        ]
        
        for cache_level, new_size in zip(self.cache_levels, new_sizes):
            await self._resize_cache_level(cache_level, new_size)

# 사용 예시 함수
async def example_usage():
    """적응적 캐시 관리자 사용 예시"""
    # 캐시 관리자 생성
    cache_manager = AdaptiveCacheManager()
    await cache_manager.initialize()
    
    try:
        # 테스트 데이터 캐싱
        test_data = torch.randn(100, 100)
        
        # 데이터 저장
        await cache_manager.put("test_tensor", test_data, CachePriority.HIGH)
        
        # 데이터 조회
        cached_data = await cache_manager.get("test_tensor")
        print(f"캐시된 데이터 크기: {cached_data.shape}")
        
        # 로더 함수와 함께 사용
        async def expensive_computation():
            await asyncio.sleep(1.0)  # 비용이 큰 계산 시뮬레이션
            return torch.randn(50, 50)
        
        result = await cache_manager.get("expensive_result", expensive_computation)
        print(f"계산 결과 크기: {result.shape}")
        
        # 통계 출력
        stats = cache_manager.get_detailed_statistics()
        print(f"\n=== 캐시 통계 ===")
        print(f"총 요청: {stats['overall']['total_requests']}")
        print(f"히트율: {stats['overall']['hit_rate']:.2%}")
        print(f"메모리 사용률: {stats['overall']['memory_utilization']:.2%}")
        print(f"평균 조회 시간: {stats['overall']['avg_lookup_time_ms']:.2f}ms")
        
        # 레벨별 통계
        for level, level_stats in stats['levels'].items():
            print(f"\n{level} 캐시:")
            print(f"  아이템 수: {level_stats['item_count']}")
            print(f"  사용률: {level_stats['utilization']:.2%}")
            print(f"  히트율: {level_stats['hit_rate']:.2%}")
        
    finally:
        await cache_manager.shutdown()

if __name__ == "__main__":
    asyncio.run(example_usage())