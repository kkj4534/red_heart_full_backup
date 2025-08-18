#!/usr/bin/env python3
"""
Dynamic GPU Memory Manager for Red Heart AI System
동적 GPU 메모리 관리자 - 모델별 맞춤형 메모리 할당

Features:
- 모델별 개별 메모리 할당 전략
- klue/bert-base 등 오버헤드 위험 모델 안정성 보장
- 메인 학습 파이프라인 동적 최적화
- 실시간 메모리 모니터링 및 조정
"""

import torch
import gc
import time
import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from contextlib import contextmanager
import threading
from enum import Enum

class ModelRiskLevel(Enum):
    """모델 오버헤드 위험도 분류"""
    LOW = "low"           # 안전한 모델들
    MEDIUM = "medium"     # 중간 위험도
    HIGH = "high"         # klue/bert-base 등 오버헤드 위험 모델

@dataclass
class GPUMemoryProfile:
    """GPU 메모리 프로필"""
    model_name: str
    risk_level: ModelRiskLevel
    base_allocation: float      # 기본 할당량 (비율)
    max_allocation: float       # 최대 할당량 (비율)
    min_allocation: float       # 최소 할당량 (비율)
    priority: int              # 우선순위 (1=최고)
    stable_mode: bool          # 안정 모드 사용 여부

class DynamicGPUManager:
    """동적 GPU 메모리 관리자"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)
        
        # GPU 총 메모리 정보
        if torch.cuda.is_available():
            self.total_memory = torch.cuda.get_device_properties(0).total_memory
            self.total_memory_gb = self.total_memory / (1024**3)
        else:
            self.total_memory = 0
            self.total_memory_gb = 0
        
        # 모델별 메모리 프로필 설정
        self.memory_profiles = self._initialize_memory_profiles()
        
        # 현재 로드된 모델 추적
        self.loaded_models: Dict[str, torch.nn.Module] = {}
        self.model_memory_usage: Dict[str, float] = {}
        
        # 동적 할당을 위한 메모리 풀
        self.reserved_memory = 0.1  # 10% 시스템 예약
        self.available_memory_ratio = 1.0 - self.reserved_memory
        
        # 스레드 안전성
        self.lock = threading.Lock()
        
        self.logger.info(f"Dynamic GPU Manager initialized - Total GPU Memory: {self.total_memory_gb:.1f}GB")
    
    def _initialize_memory_profiles(self) -> Dict[str, GPUMemoryProfile]:
        """모델별 메모리 프로필 초기화"""
        profiles = {
            # 오버헤드 위험 모델들 - 안정성 우선
            'klue/bert-base': GPUMemoryProfile(
                model_name='klue/bert-base',
                risk_level=ModelRiskLevel.HIGH,
                base_allocation=0.15,      # 15% 고정 (현재 설정 유지)
                max_allocation=0.20,       # 최대 20%
                min_allocation=0.10,       # 최소 10%
                priority=1,                # 최고 우선순위
                stable_mode=True           # 안정 모드 활성화
            ),
            'beomi/KcELECTRA-base-v2022': GPUMemoryProfile(
                model_name='beomi/KcELECTRA-base-v2022',
                risk_level=ModelRiskLevel.HIGH,
                base_allocation=0.15,
                max_allocation=0.20,
                min_allocation=0.10,
                priority=2,
                stable_mode=True
            ),
            
            # 중간 위험도 모델들
            'j-hartmann/emotion-english-distilroberta-base': GPUMemoryProfile(
                model_name='j-hartmann/emotion-english-distilroberta-base',
                risk_level=ModelRiskLevel.MEDIUM,
                base_allocation=0.25,
                max_allocation=0.35,
                min_allocation=0.15,
                priority=3,
                stable_mode=False
            ),
            
            # 안전한 모델들 - 동적 최적화 적용
            'main_learning_pipeline': GPUMemoryProfile(
                model_name='main_learning_pipeline',
                risk_level=ModelRiskLevel.LOW,
                base_allocation=0.40,      # 40% 기본 할당
                max_allocation=0.70,       # 최대 70% 까지 확장 가능
                min_allocation=0.20,       # 최소 20% 보장
                priority=4,
                stable_mode=False          # 동적 최적화 활성화
            ),
            'neural_networks': GPUMemoryProfile(
                model_name='neural_networks',
                risk_level=ModelRiskLevel.LOW,
                base_allocation=0.30,
                max_allocation=0.60,
                min_allocation=0.15,
                priority=5,
                stable_mode=False
            ),
            
            # 새로 추가된 모듈들
            'legal_expert_system': GPUMemoryProfile(
                model_name='legal_expert_system',
                risk_level=ModelRiskLevel.LOW,
                base_allocation=0.10,      # 10% 기본 할당 (내장 모델 기반)
                max_allocation=0.15,       # 최대 15%
                min_allocation=0.05,       # 최소 5%
                priority=6,
                stable_mode=False          # 동적 최적화 활성화
            ),
            'three_view_scenario_system': GPUMemoryProfile(
                model_name='three_view_scenario_system',
                risk_level=ModelRiskLevel.LOW,
                base_allocation=0.15,      # 15% 기본 할당 (신경망 기반)
                max_allocation=0.25,       # 최대 25%
                min_allocation=0.08,       # 최소 8%
                priority=7,
                stable_mode=False          # 동적 최적화 활성화
            ),
            'phase_controller_hook': GPUMemoryProfile(
                model_name='phase_controller_hook',
                risk_level=ModelRiskLevel.LOW,
                base_allocation=0.05,      # 5% 기본 할당 (성능 모니터링)
                max_allocation=0.10,       # 최대 10%
                min_allocation=0.02,       # 최소 2%
                priority=8,
                stable_mode=False          # 동적 최적화 활성화
            )
        }
        return profiles
    
    @contextmanager
    def allocate_memory(self, model_name: str, dynamic_boost: bool = False):
        """
        모델별 메모리 할당 컨텍스트 매니저
        
        Args:
            model_name: 모델 이름
            dynamic_boost: 동적 부스트 활성화 여부
        """
        profile = self.memory_profiles.get(model_name)
        if not profile:
            # 알려지지 않은 모델은 중간 위험도로 처리
            profile = GPUMemoryProfile(
                model_name=model_name,
                risk_level=ModelRiskLevel.MEDIUM,
                base_allocation=0.25,
                max_allocation=0.35,
                min_allocation=0.15,
                priority=10,
                stable_mode=True
            )
        
        with self.lock:
            try:
                # 메모리 할당
                allocated_memory = self._allocate_model_memory(profile, dynamic_boost)
                self.model_memory_usage[model_name] = allocated_memory
                
                self.logger.info(f"🔧 {model_name} 메모리 할당: {allocated_memory:.1%} ({allocated_memory * self.total_memory_gb:.1f}GB)")
                
                yield allocated_memory
                
            finally:
                # 메모리 해제
                self._release_model_memory(model_name)
                self.logger.info(f"🗑️ {model_name} 메모리 해제 완료")
    
    def _allocate_model_memory(self, profile: GPUMemoryProfile, dynamic_boost: bool) -> float:
        """실제 메모리 할당 로직"""
        if not torch.cuda.is_available():
            return 0.0
        
        # 현재 메모리 상태 확인
        current_allocated = torch.cuda.memory_allocated() / self.total_memory
        current_reserved = torch.cuda.memory_reserved() / self.total_memory
        available_ratio = self.available_memory_ratio - current_reserved
        
        # 할당량 결정 - 리미트 접근 문제 해결
        if profile.stable_mode:
            # 안정 모드: 고정 할당
            target_allocation = profile.base_allocation
            self.logger.info(f"🔒 {profile.model_name}: 안정 모드 - 고정 할당 {target_allocation:.1%}")
        else:
            # 동적 모드: 상황에 따라 조정 (더 보수적 접근)
            if dynamic_boost and available_ratio > 0.6:  # 0.5 → 0.6으로 상향
                # 여유 메모리가 충분하면 확장 할당 (더 보수적)
                target_allocation = min(profile.max_allocation, available_ratio * 0.7)  # 0.8 → 0.7로 감소
                self.logger.info(f"⚡ {profile.model_name}: 동적 부스트 - 확장 할당 {target_allocation:.1%}")
            elif available_ratio > 0.3:  # 최소 여유분 체크 추가
                # 기본 할당 (더 보수적)
                target_allocation = min(profile.base_allocation, available_ratio * 0.5)  # 0.6 → 0.5로 감소
                self.logger.info(f"🔧 {profile.model_name}: 기본 할당 {target_allocation:.1%}")
            else:
                # 메모리 부족 시 최소 할당
                target_allocation = min(profile.min_allocation, available_ratio * 0.8)
                self.logger.warning(f"⚠️ {profile.model_name}: 메모리 부족 - 최소 할당 {target_allocation:.1%}")
        
        # 최소/최대 제한 적용
        target_allocation = max(profile.min_allocation, target_allocation)
        target_allocation = min(profile.max_allocation, target_allocation)
        
        # PyTorch 메모리 설정
        try:
            if target_allocation > 0:
                # 메모리 캐시 정리
                torch.cuda.empty_cache()
                gc.collect()
                
                # 메모리 할당량 설정 (실제 할당은 모델 로딩 시)
                self.logger.info(f"🎯 최종 할당: {target_allocation:.1%} ({target_allocation * self.total_memory_gb:.1f}GB)")
                
        except Exception as e:
            self.logger.warning(f"메모리 할당 중 오류: {e}")
            # 안전한 기본값으로 fallback
            target_allocation = profile.min_allocation
        
        return target_allocation
    
    def _release_model_memory(self, model_name: str):
        """모델 메모리 해제"""
        if model_name in self.model_memory_usage:
            del self.model_memory_usage[model_name]
        
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
        
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def get_memory_status(self) -> Dict[str, float]:
        """현재 GPU 메모리 상태 반환"""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        allocated_bytes = torch.cuda.memory_allocated()
        reserved_bytes = torch.cuda.memory_reserved()
        total_bytes = self.total_memory
        
        return {
            "total_gb": total_bytes / (1024**3),
            "allocated_gb": allocated_bytes / (1024**3),
            "reserved_gb": reserved_bytes / (1024**3),
            "available_gb": (total_bytes - reserved_bytes) / (1024**3),
            "utilization_percent": (allocated_bytes / total_bytes) * 100,
            "active_models": list(self.model_memory_usage.keys()),
            "model_allocations": self.model_memory_usage.copy()
        }
    
    def optimize_for_learning(self) -> bool:
        """학습 최적화 모드 활성화"""
        try:
            # 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # 학습 전용 메모리 풀 설정
            memory_status = self.get_memory_status()
            available_memory = memory_status.get("available_gb", 0)
            
            if available_memory > 4.0:  # 4GB 이상 여유
                self.logger.info(f"🚀 학습 최적화 모드 활성화 - 사용 가능 메모리: {available_memory:.1f}GB")
                return True
            else:
                self.logger.warning(f"⚠️ 메모리 부족으로 학습 최적화 제한 - 사용 가능: {available_memory:.1f}GB")
                return False
                
        except Exception as e:
            self.logger.error(f"학습 최적화 중 오류: {e}")
            return False
    
    def emergency_cleanup(self):
        """비상 메모리 정리"""
        self.logger.warning("🚨 비상 메모리 정리 실행")
        
        # 모든 모델 해제
        for model_name in list(self.loaded_models.keys()):
            self._release_model_memory(model_name)
        
        # 강제 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        gc.collect()
        self.logger.info("✅ 비상 메모리 정리 완료")

# 전역 GPU 관리자 인스턴스
_gpu_manager = None

def get_gpu_manager() -> DynamicGPUManager:
    """전역 GPU 관리자 인스턴스 반환"""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = DynamicGPUManager()
    return _gpu_manager

# 편의 함수들
def allocate_gpu_memory(model_name: str, dynamic_boost: bool = False):
    """GPU 메모리 할당 (컨텍스트 매니저)"""
    return get_gpu_manager().allocate_memory(model_name, dynamic_boost)

def get_gpu_status() -> Dict[str, float]:
    """현재 GPU 상태 조회"""
    return get_gpu_manager().get_memory_status()

def optimize_gpu_for_learning() -> bool:
    """학습용 GPU 최적화"""
    return get_gpu_manager().optimize_for_learning()

def emergency_gpu_cleanup():
    """비상 GPU 메모리 정리"""
    get_gpu_manager().emergency_cleanup()