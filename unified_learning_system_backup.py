"""
통합 학습 시스템 - Week 4 핵심 구현
Unified Learning System - Week 4 Core Implementation

메모리 효율적인 800M 파라미터 통합 훈련 시스템:
- 공유 백본을 통한 다중 헤드 동시 학습
- 그래디언트 체크포인팅 및 혼합 정밀도 (FP16) 훈련
- 동적 배치 크기 조절 및 메모리 최적화
- 헤드별 학습 스케줄링 및 손실 함수 균형
- 효율적 백프로파게이션 및 그래디언트 누적
"""

import asyncio
import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict, OrderedDict
from enum import Enum
import numpy as np
# pathlib 제거 - WSL 호환성을 위해 os.path 사용
import json
import threading
from abc import ABC, abstractmethod
import math
import gc
import psutil
import os

# 핵심 시스템 임포트
from config import ADVANCED_CONFIG, get_smart_device, get_gpu_memory_info, ModelPriority, get_priority_based_device, _force_swap_low_priority_models, _emergency_gpu_cleanup
from head_compatibility_interface import HeadType, HeadProcessingResult
from unified_red_heart_core import RedHeartUnifiedBackbone, UnifiedRepresentation
from dynamic_swap_manager import RedHeartDynamicSwapManager
from intelligent_synergy_system import IntelligentSynergySystem

# 로거 설정
logger = logging.getLogger(__name__)

class LearningPhase(Enum):
    """학습 단계"""
    WARM_UP = "warm_up"                    # 워밍업 단계
    COLLABORATIVE = "collaborative"        # 협력 학습 단계
    SPECIALIZED = "specialized"            # 전문화 학습 단계
    INTEGRATION = "integration"            # 통합 학습 단계
    FINE_TUNING = "fine_tuning"           # 파인튜닝 단계

class TrainingStrategy(Enum):
    """훈련 전략"""
    ROUND_ROBIN = "round_robin"           # 순차적 헤드 훈련
    PARALLEL = "parallel"                 # 병렬 헤드 훈련
    PRIORITY_BASED = "priority_based"     # 우선순위 기반 훈련
    ADAPTIVE = "adaptive"                 # 적응적 훈련

@dataclass
class TrainingMetrics:
    """훈련 메트릭"""
    epoch: int = 0
    step: int = 0
    total_loss: float = 0.0
    head_losses: Dict[HeadType, float] = field(default_factory=dict)
    learning_rates: Dict[str, float] = field(default_factory=dict)
    memory_usage: float = 0.0
    gpu_utilization: float = 0.0
    training_time: float = 0.0
    gradient_norm: float = 0.0
    synergy_gain: float = 0.0
    
    # 효율성 메트릭
    samples_per_second: float = 0.0
    memory_efficiency: float = 0.0  # 메모리 사용 효율성
    convergence_rate: float = 0.0   # 수렴 속도

@dataclass
class HeadTrainingConfig:
    """헤드별 훈련 설정"""
    head_type: HeadType
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip_value: float = 1.0
    loss_weight: float = 1.0
    update_frequency: int = 1  # N번의 스텝마다 업데이트
    freeze_until_epoch: int = 0  # 이 에포크까지 동결
    
    # 스케줄링 설정
    warmup_steps: int = 1000
    scheduler_type: str = "cosine"  # cosine, linear, exponential
    
    # 정규화 설정
    dropout_rate: float = 0.1
    label_smoothing: float = 0.0

class MemoryEfficientTrainer(nn.Module):
    """메모리 효율적 훈련기"""
    
    def __init__(self, unified_backbone: RedHeartUnifiedBackbone,
                 head_configs: Dict[HeadType, HeadTrainingConfig],
                 unified_system=None):  # UnifiedLearningSystem 참조 추가
        super().__init__()
        
        self.unified_backbone = unified_backbone
        self.head_configs = head_configs
        self.unified_system = unified_system  # cached_head_modules 접근을 위해
        
        # 훈련 상태 - 메모리 안전성을 위해 ROUND_ROBIN 고정
        self.current_phase = LearningPhase.WARM_UP
        self.training_strategy = TrainingStrategy.ROUND_ROBIN
        logger.info(f"훈련 전략 설정: {self.training_strategy.value} (메모리 안전성 우선)")
        
        # 메모리 최적화 설정
        self.use_gradient_checkpointing = True
        self.use_mixed_precision = True
        self.gradient_accumulation_steps = 4
        self.max_batch_size = 8
        self.adaptive_batch_sizing = True
        
        # 그래디언트 스케일러 (혼합 정밀도용)
        self.scaler = GradScaler()
        
        # 옵티마이저 및 스케줄러
        self.optimizers = {}
        self.schedulers = {}
        self._initialize_optimizers()
        
        # 훈련 메트릭
        self.metrics_history = deque(maxlen=1000)
        self.current_metrics = TrainingMetrics()
        
        # 동적 배치 크기 관리
        self.adaptive_batch_manager = AdaptiveBatchSizeManager()
        
        # 메모리 모니터링
        self.memory_monitor = MemoryMonitor()
        
        # 백본을 메모리 모니터에 등록
        self.memory_monitor.register_gpu_module("unified_backbone", self.unified_backbone)
        
        logger.info("MemoryEfficientTrainer 초기화 완료")
    
    def forward_with_checkpointing(self, batch_data: Dict[str, Any], 
                                  active_heads: List[HeadType]) -> Dict[str, Any]:
        """그래디언트 체크포인팅을 사용한 순전파"""
        
        # 1. 백본 순전파
        backbone_output = self.unified_backbone(batch_data)
        
        # 2. 헤드별 처리
        head_outputs = {}
        for head_type in active_heads:
            if self.unified_system and hasattr(self.unified_system, 'cached_head_modules'):
                head_module = self.unified_system.cached_head_modules.get(head_type)
                if head_module is not None:
                    try:
                        # 백본 출력을 헤드에 전달
                        head_input = backbone_output.shared_embedding
                        logger.debug(f"헤드 {head_type.value} - 백본 출력 shape: {head_input.shape}")
                        
                        # 헤드 forward 실행 (forward 메서드가 내부적으로 input_adapter 처리)
                        if hasattr(head_module, 'forward'):
                            logger.debug(f"헤드 {head_type.value} - forward 실행")
                            try:
                                head_output = head_module.forward(head_input)
                                head_outputs[head_type] = head_output
                                logger.debug(f"헤드 {head_type.value} - forward 성공, 출력 shape: {head_output.shape}")
                            except RuntimeError as re:
                                if "mat1 and mat2" in str(re):
                                    logger.error(f"Shape mismatch 상세 정보:")
                                    logger.error(f"  - 헤드 타입: {head_type.value}")
                                    logger.error(f"  - 헤드 클래스: {head_module.__class__.__name__}")
                                    logger.error(f"  - 입력 shape: {head_input.shape}")
                                    
                                    # PyTorch 네트워크 구조 상세 분석
                                    if hasattr(head_module, 'get_pytorch_network'):
                                        pytorch_net = head_module.get_pytorch_network()
                                        if pytorch_net:
                                            logger.error(f"  - PyTorch 네트워크 타입: {type(pytorch_net).__name__}")
                                            
                                            # Sequential이나 다른 컨테이너인 경우
                                            if hasattr(pytorch_net, 'modules'):
                                                for idx, module in enumerate(pytorch_net.modules()):
                                                    if isinstance(module, torch.nn.Linear):
                                                        logger.error(f"    - Linear 레이어 {idx}: in_features={module.in_features}, out_features={module.out_features}")
                                            
                                            # 모든 파라미터의 shape 출력
                                            for name, param in pytorch_net.named_parameters():
                                                if 'weight' in name:
                                                    logger.error(f"    - {name}: shape={param.shape}")
                                raise
                        else:
                            logger.warning(f"헤드 {head_type.value}에 forward 메서드가 없음")
                    except Exception as e:
                        logger.error(f"헤드 {head_type.value} 순전파 오류: {str(e)}")
                        logger.error(f"오류 타입: {type(e).__name__}")
                        import traceback
                        logger.error(f"트레이스백:\n{traceback.format_exc()}")
                        logger.error(f"입력 shape: {head_input.shape if hasattr(head_input, 'shape') else 'N/A'}")
                        
                        # 더 자세한 디버깅 정보
                        if hasattr(head_module, '__class__'):
                            logger.error(f"헤드 클래스: {head_module.__class__.__name__}")
                        if hasattr(head_module, 'get_pytorch_network'):
                            pytorch_net = head_module.get_pytorch_network()
                            if pytorch_net:
                                logger.error(f"PyTorch 네트워크 타입: {type(pytorch_net)}")
        
        return {
            'backbone_output': backbone_output,
            'head_outputs': head_outputs
        }
    
    def _initialize_optimizers(self):
        """옵티마이저 및 스케줄러 초기화"""
        
        # 공유 백본용 옵티마이저
        backbone_params = list(self.unified_backbone.parameters())
        self.optimizers['backbone'] = torch.optim.AdamW(
            backbone_params,
            lr=1e-4,
            weight_decay=1e-5,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 백본을 학습 모드로 설정하고 gradient 연결 강화
        self.unified_backbone.train()
        
        # 백본 파라미터들의 requires_grad 확인 및 강화
        backbone_grad_params = sum(1 for p in self.unified_backbone.parameters() if p.requires_grad)
        backbone_total_params = sum(1 for p in self.unified_backbone.parameters())
        
        if backbone_grad_params != backbone_total_params:
            logger.warning(f"백본 파라미터 gradient 설정 불일치: {backbone_grad_params}/{backbone_total_params}")
            for param in self.unified_backbone.parameters():
                param.requires_grad_(True)
        
        logger.info(f"백본 옵티마이저 초기화: {backbone_total_params}개 파라미터 중 {backbone_grad_params}개 gradient 활성화")
        
        # 헤드별 옵티마이저 (나중에 헤드가 로드될 때 추가)
        for head_type, config in self.head_configs.items():
            # 헤드별 파라미터는 실제 헤드가 로드될 때 설정
            pass
        
        # 스케줄러 설정
        self.schedulers['backbone'] = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizers['backbone'],
            T_0=1000,  # 첫 번째 재시작까지의 스텝 수
            T_mult=2,  # 재시작 주기 배수
            eta_min=1e-6
        )
    
    def add_head_optimizer(self, head_type: HeadType, head_module: nn.Module):
        """헤드별 옵티마이저 추가"""
        config = self.head_configs[head_type]
        
        self.optimizers[head_type.value] = torch.optim.AdamW(
            head_module.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # 헤드를 메모리 모니터에 등록
        self.memory_monitor.register_gpu_module(f"head_{head_type.value}", head_module)
        
        if config.scheduler_type == "cosine":
            self.schedulers[head_type.value] = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizers[head_type.value],
                T_max=10000,
                eta_min=config.learning_rate * 0.01
            )
        elif config.scheduler_type == "linear":
            self.schedulers[head_type.value] = torch.optim.lr_scheduler.LinearLR(
                self.optimizers[head_type.value],
                start_factor=0.1,
                total_iters=config.warmup_steps
            )
    
    def forward_with_checkpointing(self, input_data: Dict[str, Any],
                                 active_heads: List[HeadType]) -> Dict[str, torch.Tensor]:
        """체크포인팅을 사용한 순전파"""
        
        # 입력 데이터 준비
        input_text = input_data.get('text', '')
        batch_size = input_data.get('batch_size', 1)
        
        # 토크나이징 (가상의 토크나이징)
        input_ids = torch.randint(0, 30000, (batch_size, 128), device=self.unified_backbone.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        
        # 백본이 학습 모드인지 확인하고 설정
        if not self.unified_backbone.training:
            logger.warning("백본이 eval 모드였습니다. 학습 모드로 변경합니다.")
            self.unified_backbone.train()
        
        # input_ids는 정수형이므로 requires_grad 설정하지 않음 (오류 방지)
        logger.debug(f"input_ids dtype: {input_ids.dtype}, shape: {input_ids.shape}")
        logger.debug(f"백본 training 모드: {self.unified_backbone.training}")
        
        # 백본 파라미터들의 gradient 상태 확인
        backbone_grad_params = sum(1 for p in self.unified_backbone.parameters() if p.requires_grad)
        backbone_total_params = sum(1 for p in self.unified_backbone.parameters())
        logger.debug(f"백본 gradient 파라미터: {backbone_grad_params}/{backbone_total_params}")
        
        if self.use_gradient_checkpointing:
            # 그래디언트 체크포인팅 사용
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            
            # 백본 처리 (체크포인팅 적용)
            unified_repr = checkpoint(
                create_custom_forward(self.unified_backbone),
                input_ids,
                attention_mask
            )
        else:
            # 일반 순전파
            unified_repr = self.unified_backbone(input_ids, attention_mask)
        
        # 각 헤드별 출력 계산 - 백본만 테스트 모드
        head_outputs = {}
        for head_type in active_heads:
            # 캐시된 실제 헤드 모듈 사용
            if self.unified_system is not None:
                real_head = self.unified_system.cached_head_modules.get(head_type)
            else:
                real_head = None
            
            if real_head is None:
                raise RuntimeError(f"실제 헤드 모듈 {head_type.value}을 찾을 수 없습니다. 시스템 초기화가 제대로 완료되지 않았습니다.")
            
            # 실제 헤드를 통한 순전파 - CPU에서 계산하여 메모리 문제 해결
            head_input = unified_repr.shared_embedding  # 백본 출력을 헤드 입력으로 사용
            
            # 103% GPU 메모리 문제 해결: gradient 연결을 유지하면서 CPU에서 계산
            # 헤드 모듈을 CPU로 이동 (이미 CPU에 있어야 함)
            if next(real_head.parameters()).device.type != 'cpu':
                logger.warning(f"헤드 {head_type.value}이 GPU에 있습니다. CPU로 이동합니다.")
                real_head = real_head.cpu()
            
            # 헤드 입력을 CPU로 이동 (gradient 연결 유지)
            head_input_cpu = head_input.cpu()  # detach() 제거하여 gradient 연결 유지
            logger.debug(f"헤드 {head_type.value}: gradient 연결 유지하며 CPU로 이동")
            
            # gradient 연결 상태 확인 및 분석
            logger.debug(f"백본 출력 분석 - requires_grad: {head_input_cpu.requires_grad}, dtype: {head_input_cpu.dtype}, shape: {head_input_cpu.shape}")
            logger.debug(f"백본 출력 grad_fn: {head_input_cpu.grad_fn}")
            
            if not head_input_cpu.requires_grad:
                # 백본 파라미터들의 requires_grad 상태 재확인
                backbone_params_requiring_grad = [p for p in self.unified_backbone.parameters() if p.requires_grad]
                logger.warning(f"백본 출력이 requires_grad=False입니다!")
                logger.warning(f"백본 requires_grad 파라미터: {len(backbone_params_requiring_grad)}개")
                
                # 실제 gradient 연결 문제 해결
                if len(backbone_params_requiring_grad) > 0:
                    logger.warning("백본 training 모드 및 gradient 설정을 강제로 복구합니다.")
                    
                    # 백본을 명시적으로 training 모드로 설정
                    self.unified_backbone.train()
                    
                    # 모든 백본 파라미터의 requires_grad 재설정
                    for param in self.unified_backbone.parameters():
                        param.requires_grad_(True)
                    
                    # 백본을 다시 실행하여 gradient가 연결된 출력 생성
                    logger.warning("백본 순전파를 재실행합니다.")
                    if self.use_gradient_checkpointing:
                        unified_repr = checkpoint(
                            create_custom_forward(self.unified_backbone),
                            input_ids,
                            attention_mask
                        )
                    else:
                        unified_repr = self.unified_backbone(input_ids, attention_mask)
                    
                    head_input = unified_repr.shared_embedding
                    
                    # 재실행 후에도 gradient가 없으면 치명적 오류
                    if not head_input.requires_grad:
                        logger.error("백본 순전파 재실행 후에도 gradient가 연결되지 않음")
                        raise RuntimeError("백본 gradient 연결 실패 - 백본 구현 결함")
                    else:
                        logger.info("백본 gradient 연결 복구 성공")
                else:
                    logger.error("백본의 모든 파라미터가 requires_grad=False입니다.")
                    # 백본 파라미터들을 강제로 활성화
                    for param in self.unified_backbone.parameters():
                        param.requires_grad_(True)
                    logger.warning("백본 파라미터 requires_grad를 강제로 True로 설정했습니다.")
                    raise RuntimeError("백본 파라미터 gradient 설정 문제 - 재시도 필요")
            else:
                logger.debug(f"헤드 {head_type.value}: gradient 연결 정상")
            
            # 헤드 모듈이 학습 모드인지 확인
            if hasattr(real_head, 'training'):
                if not real_head.training:
                    logger.warning(f"헤드 {head_type.value}이 eval 모드입니다. 학습 모드로 변경합니다.")
                    real_head.train()
            
            # 헤드 파라미터의 requires_grad 확인
            head_params_with_grad = sum(1 for p in real_head.parameters() if p.requires_grad)
            head_params_total = sum(1 for p in real_head.parameters())
            if head_params_with_grad == 0:
                logger.error(f"헤드 {head_type.value}의 모든 파라미터가 requires_grad=False입니다")
                raise RuntimeError(f"헤드 {head_type.value} 파라미터에 gradient가 연결되지 않음")
            
            logger.debug(f"헤드 {head_type.value}: {head_params_with_grad}/{head_params_total} 파라미터가 requires_grad=True")
            
            # 실제 헤드 순전파 (CPU에서 수행)
            head_output_cpu = real_head(head_input_cpu)
            
            # 결과를 다시 GPU로 이동 (loss 계산을 위해)
            head_output = head_output_cpu.to(head_input.device)  # 백본과 같은 디바이스로 이동
            
            # 출력의 gradient 연결 확인
            if not head_output.requires_grad:
                logger.error(f"헤드 {head_type.value} 출력이 requires_grad=False입니다")
                raise RuntimeError(f"헤드 {head_type.value} 출력에 gradient가 연결되지 않음")
            
            logger.debug(f"헤드 {head_type.value}: CPU 계산 완료 후 GPU로 이동 ({head_output_cpu.device} → {head_output.device})")
            head_outputs[head_type.value] = head_output
        
        return {
            'unified_representation': unified_repr,
            'head_outputs': head_outputs
        }
    
    async def train_step(self, batch_data: Dict[str, Any],
                        active_heads: List[HeadType]) -> TrainingMetrics:
        """단일 훈련 스텝"""
        
        step_start_time = time.time()
        
        # 메모리 사용량 체크
        memory_before = self.memory_monitor.get_memory_usage()
        
        # 동적 배치 크기 조절
        if self.adaptive_batch_sizing:
            batch_size = self.adaptive_batch_manager.get_optimal_batch_size(
                memory_before, len(active_heads)
            )
            batch_data['batch_size'] = batch_size
        
        # 혼합 정밀도 컨텍스트
        with autocast(enabled=self.use_mixed_precision):
            # 순전파
            outputs = self.forward_with_checkpointing(batch_data, active_heads)
            
            # 손실 계산
            losses = self._calculate_losses(outputs, batch_data, active_heads)
            total_loss = sum(losses.values()) / len(losses)
        
        # 역전파 (그래디언트 누적 사용)
        scaled_loss = total_loss / self.gradient_accumulation_steps
        
        if self.use_mixed_precision:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        # 그래디언트 업데이트 (누적 스텝마다)
        if (self.current_metrics.step + 1) % self.gradient_accumulation_steps == 0:
            await self._update_gradients(active_heads)
        
        # 메트릭 업데이트
        memory_after = self.memory_monitor.get_memory_usage()
        step_time = time.time() - step_start_time
        
        self._update_metrics(
            losses, total_loss, memory_after - memory_before, 
            step_time, batch_data.get('batch_size', 1)
        )
        
        return self.current_metrics
    
    async def _update_gradients(self, active_heads: List[HeadType]):
        """그래디언트 업데이트"""
        
        # 그래디언트 클리핑
        total_norm = 0.0
        
        if self.use_mixed_precision:
            # 백본 그래디언트 클리핑
            self.scaler.unscale_(self.optimizers['backbone'])
            backbone_norm = torch.nn.utils.clip_grad_norm_(
                self.unified_backbone.parameters(), 1.0
            )
            total_norm += backbone_norm
            
            # 헤드별 그래디언트 클리핑
            for head_type in active_heads:
                if head_type.value in self.optimizers:
                    self.scaler.unscale_(self.optimizers[head_type.value])
                    config = self.head_configs[head_type]
                    # 실제 헤드 파라미터가 있다면 클리핑
                    # head_norm = torch.nn.utils.clip_grad_norm_(head_params, config.gradient_clip_value)
                    # total_norm += head_norm
        
        # 옵티마이저 스텝
        if self.use_mixed_precision:
            self.scaler.step(self.optimizers['backbone'])
            for head_type in active_heads:
                if head_type.value in self.optimizers:
                    self.scaler.step(self.optimizers[head_type.value])
            self.scaler.update()
        else:
            self.optimizers['backbone'].step()
            for head_type in active_heads:
                if head_type.value in self.optimizers:
                    self.optimizers[head_type.value].step()
        
        # 스케줄러 업데이트
        for scheduler in self.schedulers.values():
            scheduler.step()
        
        # 그래디언트 초기화
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()
        
        self.current_metrics.gradient_norm = total_norm
    
    def _calculate_losses(self, outputs: Dict[str, torch.Tensor],
                         batch_data: Dict[str, Any],
                         active_heads: List[HeadType]) -> Dict[str, torch.Tensor]:
        """손실 함수 계산"""
        losses = {}
        
        # 각 헤드별 손실 계산
        for head_type in active_heads:
            config = self.head_configs[head_type]
            
            # 가상의 타겟 생성 (실제 구현에서는 batch_data에서 추출)
            batch_size = batch_data.get('batch_size', 1)
            
            if head_type == HeadType.EMOTION_EMPATHY:
                # 감정 분류 손실
                target = torch.randint(0, 10, (batch_size,), device=outputs['head_outputs'][head_type.value].device)
                logits = outputs['head_outputs'][head_type.value][:, :10]  # 10개 감정 클래스
                loss = F.cross_entropy(logits, target, label_smoothing=config.label_smoothing)
                
            elif head_type == HeadType.BENTHAM_FROMM:
                # 윤리 점수 회귀 손실
                target = torch.rand(batch_size, 1, device=outputs['head_outputs'][head_type.value].device)
                pred = outputs['head_outputs'][head_type.value][:, :1]
                loss = F.mse_loss(pred, target)
                
            elif head_type == HeadType.SEMANTIC_SURD:
                # 의미 유사도 손실
                target = torch.rand(batch_size, 768, device=outputs['head_outputs'][head_type.value].device)
                pred = outputs['head_outputs'][head_type.value]
                loss = F.cosine_embedding_loss(pred, target, torch.ones(batch_size, device=pred.device))
                
            elif head_type == HeadType.REGRET_LEARNING:
                # 후회 예측 손실
                target = torch.rand(batch_size, 1, device=outputs['head_outputs'][head_type.value].device)
                pred = outputs['head_outputs'][head_type.value][:, :1]
                loss = F.smooth_l1_loss(pred, target)
                
            else:
                # 기본 손실
                target = torch.randn_like(outputs['head_outputs'][head_type.value])
                loss = F.mse_loss(outputs['head_outputs'][head_type.value], target)
            
            # 가중치 적용
            losses[head_type.value] = loss * config.loss_weight
        
        return losses
    
    def _update_metrics(self, losses: Dict[str, torch.Tensor], total_loss: torch.Tensor,
                       memory_delta: float, step_time: float, batch_size: int):
        """메트릭 업데이트"""
        
        self.current_metrics.step += 1
        self.current_metrics.total_loss = float(total_loss.item())
        self.current_metrics.head_losses = {k: float(v.item()) for k, v in losses.items()}
        self.current_metrics.memory_usage = memory_delta
        self.current_metrics.training_time = step_time
        self.current_metrics.samples_per_second = batch_size / step_time if step_time > 0 else 0.0
        
        # 학습률 기록
        for name, optimizer in self.optimizers.items():
            self.current_metrics.learning_rates[name] = optimizer.param_groups[0]['lr']
        
        # 메트릭 히스토리에 추가
        metrics_copy = TrainingMetrics(
            epoch=self.current_metrics.epoch,
            step=self.current_metrics.step,
            total_loss=self.current_metrics.total_loss,
            head_losses=self.current_metrics.head_losses.copy(),
            learning_rates=self.current_metrics.learning_rates.copy(),
            memory_usage=self.current_metrics.memory_usage,
            training_time=self.current_metrics.training_time,
            samples_per_second=self.current_metrics.samples_per_second
        )
        self.metrics_history.append(metrics_copy)

class AdaptiveBatchSizeManager:
    """적응적 배치 크기 관리자 - 103% GPU 메모리 문제 대응"""
    
    def __init__(self, initial_batch_size: int = 4, max_batch_size: int = 16):
        self.current_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.min_batch_size = 1
        
        # 메모리 사용량 기록
        self.memory_history = deque(maxlen=10)
        self.oom_count = 0
        self.successful_steps = 0
        
        # 적응 파라미터 (현실적 조정 - GPU threshold 85% 복원에 맞춤)
        self.memory_threshold = 0.65  # GPU 메모리 65% 사용시 배치 크기 감소 (50% → 65%)
        self.increase_threshold = 0.50  # GPU 메모리 50% 미만시 배치 크기 증가 고려 (35% → 50%)
        
    def get_optimal_batch_size(self, current_memory_usage: float, num_active_heads: int) -> int:
        """우선순위 시스템과 통합된 최적 배치 크기 계산"""
        
        # GPU 메모리 정보 가져오기
        memory_info = get_gpu_memory_info()
        if memory_info:
            memory_utilization = memory_info.get('usage_percent', 50) / 100.0
        else:
            memory_utilization = 0.5  # 기본값
        
        self.memory_history.append(memory_utilization)
        
        # 우선순위 시스템과 통합된 단계별 배치 크기 조정
        # 70% 미만: 정상 운영
        if memory_utilization < 0.70:
            target_batch_size = 4  # 기본 배치 크기
            
        # 70-75%: 경고 레벨 - 배치 크기 감소
        elif memory_utilization < 0.75:
            target_batch_size = 2
            logger.warning(f"GPU 메모리 경고 레벨({memory_utilization:.1%}) - 배치 크기 2로 감소")
            
        # 75-80%: 위험 레벨 - 최소 배치 크기
        elif memory_utilization < 0.80:
            target_batch_size = 1
            logger.error(f"GPU 메모리 위험 레벨({memory_utilization:.1%}) - 배치 크기 1로 최소화")
            
        # 80% 초과: 긴급 상황 - 배치 처리 중단
        else:
            target_batch_size = 1
            logger.critical(f"GPU 메모리 긴급 상황({memory_utilization:.1%}) - 최소 배치로 학습 계속")
        
        # 활성 헤드 수에 따른 추가 조정
        if num_active_heads > 2:
            target_batch_size = max(1, target_batch_size // 2)
            logger.info(f"다중 헤드 활성({num_active_heads}개) - 배치 크기 추가 감소: {target_batch_size}")
        
        # 점진적 변경 (갑작스런 변화 방지)
        if target_batch_size > self.current_batch_size:
            self.current_batch_size = min(target_batch_size, self.current_batch_size + 1)
        elif target_batch_size < self.current_batch_size:
            self.current_batch_size = target_batch_size  # 감소는 즉시 적용
        
        # 범위 제한
        self.current_batch_size = max(self.min_batch_size, 
                                     min(self.max_batch_size, self.current_batch_size))
        
        self.successful_steps += 1
        return self.current_batch_size
    
    def report_oom(self):
        """Out of Memory 발생 보고"""
        self.oom_count += 1
        if self.current_batch_size > self.min_batch_size:
            self.current_batch_size = max(self.min_batch_size, self.current_batch_size // 2)
            logger.error(f"OOM 발생, 배치 크기 대폭 감소: {self.current_batch_size}")

class MemoryMonitor:
    """강화된 메모리 모니터 - GPU 모듈별 상세 추적"""
    
    def __init__(self):
        self.cpu_memory_history = deque(maxlen=100)
        self.gpu_memory_history = deque(maxlen=100)
        
        # GPU 모듈 추적을 위한 추가 저장소
        self.gpu_module_registry = {}  # 모듈명 -> 모듈 객체
        self.memory_snapshots = deque(maxlen=50)  # 상세 메모리 스냅샷
        self.high_memory_alerts = []  # 고메모리 사용 알림 기록
        
        # 디버그 설정
        self.debug_threshold = 0.90  # 90% 이상에서 상세 로그
        self.critical_threshold = 1.0  # 100% 이상에서 긴급 대응
        
    def get_memory_usage(self) -> Dict[str, float]:
        """현재 메모리 사용량 반환 및 고메모리 사용 시 자동 분석"""
        
        # CPU 메모리
        cpu_memory = psutil.virtual_memory()
        cpu_usage_gb = (cpu_memory.total - cpu_memory.available) / (1024**3)
        
        # GPU 메모리 (상세 분석)
        gpu_info = get_gpu_memory_info()
        gpu_usage_gb = gpu_info.get('memory_used_gb', 0) if gpu_info else 0
        gpu_total_gb = gpu_info.get('memory_total_gb', 8) if gpu_info else 8
        gpu_percent = (gpu_usage_gb / gpu_total_gb * 100) if gpu_info else 0
        
        # torch.cuda 직접 조회로 더 정확한 수치 얻기
        import torch
        if torch.cuda.is_available():
            allocated_gb = torch.cuda.memory_allocated(0) / (1024**3)
            reserved_gb = torch.cuda.memory_reserved(0) / (1024**3)
            free_gb = (torch.cuda.get_device_properties(0).total_memory / (1024**3)) - allocated_gb
            torch_gpu_percent = (allocated_gb / (torch.cuda.get_device_properties(0).total_memory / (1024**3))) * 100
            
            # 더 정확한 수치로 업데이트
            gpu_usage_gb = allocated_gb
            gpu_percent = torch_gpu_percent
            
            usage = {
                'cpu_memory_gb': cpu_usage_gb,
                'gpu_memory_gb': gpu_usage_gb,
                'cpu_percent': cpu_memory.percent,
                'gpu_percent': gpu_percent,
                # 추가 상세 정보
                'gpu_allocated_gb': allocated_gb,
                'gpu_reserved_gb': reserved_gb,
                'gpu_free_gb': free_gb,
                'gpu_total_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3)
            }
        else:
            usage = {
                'cpu_memory_gb': cpu_usage_gb,
                'gpu_memory_gb': gpu_usage_gb,
                'cpu_percent': cpu_memory.percent,
                'gpu_percent': gpu_percent
            }
        
        self.cpu_memory_history.append(usage['cpu_memory_gb'])
        self.gpu_memory_history.append(usage['gpu_memory_gb'])
        
        # 90% 이상이면 상세 분석 수행
        if usage['gpu_percent'] >= (self.debug_threshold * 100):
            self._analyze_high_memory_usage(usage)
        
        # 100% 이상이면 긴급 대응
        if usage['gpu_percent'] >= (self.critical_threshold * 100):
            self._emergency_memory_response(usage)
        
        return usage
    
    def get_memory_efficiency(self) -> float:
        """메모리 효율성 계산"""
        if len(self.gpu_memory_history) < 2:
            return 1.0
        
        # 메모리 사용량의 안정성을 효율성 지표로 사용
        recent_usage = list(self.gpu_memory_history)[-10:]
        if len(recent_usage) < 2:
            return 1.0
        
        std_dev = np.std(recent_usage)
        mean_usage = np.mean(recent_usage)
        
        # 표준편차가 작을수록 효율적 (안정적)
        efficiency = 1.0 / (1.0 + std_dev / max(0.1, mean_usage))
        return min(1.0, efficiency)
    
    def register_gpu_module(self, name: str, module):
        """GPU 모듈 등록 (추적용)"""
        self.gpu_module_registry[name] = module
        logger.debug(f"GPU 모듈 등록: {name}")
    
    def _analyze_high_memory_usage(self, usage: Dict[str, float]):
        """고메모리 사용 상세 분석"""
        
        timestamp = datetime.now()
        gpu_percent = usage['gpu_percent']
        
        logger.critical(f"🚨 HIGH GPU MEMORY ALERT: {gpu_percent:.1f}% 사용 중!")
        logger.critical(f"   할당된 메모리: {usage.get('gpu_allocated_gb', 0):.2f}GB")
        logger.critical(f"   예약된 메모리: {usage.get('gpu_reserved_gb', 0):.2f}GB")
        logger.critical(f"   여유 메모리: {usage.get('gpu_free_gb', 0):.2f}GB")
        logger.critical(f"   전체 메모리: {usage.get('gpu_total_gb', 8):.2f}GB")
        
        # torch.cuda 상세 정보
        import torch
        if torch.cuda.is_available():
            logger.critical("📊 TORCH.CUDA 상세 정보:")
            logger.critical(f"   memory_allocated(): {torch.cuda.memory_allocated(0) / (1024**3):.3f}GB")
            logger.critical(f"   memory_reserved(): {torch.cuda.memory_reserved(0) / (1024**3):.3f}GB")
            logger.critical(f"   max_memory_allocated(): {torch.cuda.max_memory_allocated(0) / (1024**3):.3f}GB")
            logger.critical(f"   max_memory_reserved(): {torch.cuda.max_memory_reserved(0) / (1024**3):.3f}GB")
            
            # 메모리 사용률 계산 방식들 비교
            device_props = torch.cuda.get_device_properties(0)
            total_memory = device_props.total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)
            reserved_memory = torch.cuda.memory_reserved(0) / (1024**3)
            
            allocated_percent = (allocated_memory / total_memory) * 100
            reserved_percent = (reserved_memory / total_memory) * 100
            
            logger.critical(f"🔍 메모리 사용률 계산 비교:")
            logger.critical(f"   할당 기준: {allocated_percent:.2f}%")
            logger.critical(f"   예약 기준: {reserved_percent:.2f}%")
            logger.critical(f"   config.py 기준: {usage.get('gpu_percent', 0):.2f}%")
        
        # 등록된 GPU 모듈들의 메모리 사용량 분석
        self._debug_gpu_modules()
        
        # 메모리 스냅샷 저장
        snapshot = {
            'timestamp': timestamp,
            'gpu_percent': gpu_percent,
            'usage_details': usage.copy(),
            'module_count': len(self.gpu_module_registry)
        }
        self.memory_snapshots.append(snapshot)
        self.high_memory_alerts.append(snapshot)
        
        # 최근 메모리 패턴 분석
        if len(self.gpu_memory_history) >= 5:
            recent_usage = list(self.gpu_memory_history)[-5:]
            avg_usage = sum(recent_usage) / len(recent_usage)
            trend = "증가" if recent_usage[-1] > recent_usage[0] else "감소"
            logger.critical(f"📈 최근 메모리 패턴: 평균 {avg_usage:.2f}GB, {trend} 추세")
    
    def _emergency_memory_response(self, usage: Dict[str, float]):
        """100% 이상 메모리 사용 시 긴급 대응"""
        
        gpu_percent = usage['gpu_percent']
        
        logger.error(f"🚨🚨🚨 CRITICAL GPU MEMORY OVERFLOW: {gpu_percent:.1f}%!!")
        logger.error("긴급 메모리 정리 시작...")
        
        # torch 캐시 정리
        import torch
        if torch.cuda.is_available():
            logger.error("🧹 torch.cuda.empty_cache() 실행")
            torch.cuda.empty_cache()
            
            # 정리 후 상태 재확인
            after_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            after_reserved = torch.cuda.memory_reserved(0) / (1024**3)
            after_percent = (after_allocated / (torch.cuda.get_device_properties(0).total_memory / (1024**3))) * 100
            
            logger.error(f"🔄 정리 후 상태: {after_percent:.1f}% (할당: {after_allocated:.2f}GB, 예약: {after_reserved:.2f}GB)")
        
        # 가비지 컬렉션
        import gc
        logger.error("🗑️  gc.collect() 실행")
        gc.collect()
        
        # 위험 수준 기록
        self.high_memory_alerts.append({
            'timestamp': datetime.now(),
            'type': 'CRITICAL_OVERFLOW',
            'gpu_percent': gpu_percent,
            'emergency_response': True
        })
    
    def _debug_gpu_modules(self):
        """등록된 GPU 모듈들의 상세 메모리 분석"""
        
        if not self.gpu_module_registry:
            logger.warning("🔍 등록된 GPU 모듈이 없습니다. 추적 불가.")
            return
        
        logger.critical(f"🔍 GPU 모듈별 메모리 분석 ({len(self.gpu_module_registry)}개 모듈):")
        
        import torch
        total_params = 0
        
        for name, module in self.gpu_module_registry.items():
            try:
                if hasattr(module, 'parameters'):
                    # PyTorch 모듈인 경우
                    param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
                    param_memory_mb = sum(p.numel() * p.element_size() for p in module.parameters()) / (1024**2)
                    device = next(module.parameters()).device if param_count > 0 else "unknown"
                    
                    logger.critical(f"   📦 {name}:")
                    logger.critical(f"      파라미터: {param_count:,}개")
                    logger.critical(f"      메모리: {param_memory_mb:.2f}MB")
                    logger.critical(f"      디바이스: {device}")
                    logger.critical(f"      훈련 모드: {getattr(module, 'training', 'unknown')}")
                    
                    total_params += param_count
                
                else:
                    logger.critical(f"   ❓ {name}: PyTorch 모듈 아님")
            
            except Exception as e:
                logger.critical(f"   ❌ {name}: 분석 실패 - {str(e)}")
        
        logger.critical(f"🔢 전체 추적된 파라미터: {total_params:,}개")
        
        # 현재 GPU 텐서들 분석 시도
        try:
            if torch.cuda.is_available():
                logger.critical("🧠 GPU 텐서 분석:")
                import gc
                gpu_tensors = []
                for obj in gc.get_objects():
                    if torch.is_tensor(obj) and obj.is_cuda:
                        gpu_tensors.append(obj)
                
                total_tensor_memory = sum(tensor.element_size() * tensor.numel() for tensor in gpu_tensors) / (1024**3)
                logger.critical(f"   GPU 텐서 개수: {len(gpu_tensors)}개")
                logger.critical(f"   GPU 텐서 메모리: {total_tensor_memory:.3f}GB")
        
        except Exception as e:
            logger.warning(f"GPU 텐서 분석 실패: {str(e)}")
    
    def get_debug_summary(self) -> Dict[str, Any]:
        """디버그 요약 정보 반환"""
        return {
            'registered_modules': len(self.gpu_module_registry),
            'high_memory_alerts': len(self.high_memory_alerts),
            'memory_snapshots': len(self.memory_snapshots),
            'latest_alert': self.high_memory_alerts[-1] if self.high_memory_alerts else None,
            'debug_threshold': self.debug_threshold,
            'critical_threshold': self.critical_threshold
        }

class UnifiedLearningScheduler:
    """통합 학습 스케줄러"""
    
    def __init__(self, head_configs: Dict[HeadType, HeadTrainingConfig]):
        self.head_configs = head_configs
        self.current_phase = LearningPhase.WARM_UP
        self.phase_progress = 0.0
        
        # 스케줄링 상태
        self.step_count = 0
        self.epoch_count = 0
        self.phase_step_count = 0
        
        # 학습 전략 매개변수
        self.strategy_params = {
            TrainingStrategy.ROUND_ROBIN: {'cycle_length': 4},
            TrainingStrategy.PARALLEL: {'weight_balance': 0.5},
            TrainingStrategy.PRIORITY_BASED: {'priority_weights': {}},
            TrainingStrategy.ADAPTIVE: {'adaptation_rate': 0.1}
        }
        
    def get_active_heads(self, current_strategy: TrainingStrategy, 
                        available_heads: List[HeadType]) -> List[HeadType]:
        """현재 전략에 따른 활성 헤드 선택"""
        
        if current_strategy == TrainingStrategy.ROUND_ROBIN:
            # 순차적 헤드 훈련
            cycle_length = self.strategy_params[TrainingStrategy.ROUND_ROBIN]['cycle_length']
            cycle_position = self.step_count % (len(available_heads) * cycle_length)
            head_index = cycle_position // cycle_length
            return [available_heads[head_index]]
        
        elif current_strategy == TrainingStrategy.PARALLEL:
            # 모든 헤드 병렬 훈련
            return available_heads
        
        elif current_strategy == TrainingStrategy.PRIORITY_BASED:
            # 우선순위 기반 선택
            priorities = self._calculate_head_priorities(available_heads)
            # 상위 50% 헤드 선택
            sorted_heads = sorted(priorities.items(), key=lambda x: x[1], reverse=True)
            top_heads = [head for head, _ in sorted_heads[:max(1, len(sorted_heads)//2)]]
            return top_heads
        
        elif current_strategy == TrainingStrategy.ADAPTIVE:
            # 적응적 선택 (성능 기반)
            return self._adaptive_head_selection(available_heads)
        
        return available_heads  # 기본값
    
    def _calculate_head_priorities(self, available_heads: List[HeadType]) -> Dict[HeadType, float]:
        """헤드별 우선순위 계산"""
        priorities = {}
        
        for head_type in available_heads:
            config = self.head_configs[head_type]
            
            # 기본 우선순위는 학습률과 손실 가중치에 기반
            base_priority = config.learning_rate * config.loss_weight
            
            # 현재 에포크와 freeze_until_epoch 고려
            if self.epoch_count < config.freeze_until_epoch:
                priority = 0.0  # 동결된 헤드
            else:
                # 워밍업 완료 정도에 따른 우선순위
                warmup_factor = min(1.0, self.step_count / max(1, config.warmup_steps))
                priority = base_priority * warmup_factor
            
            priorities[head_type] = priority
        
        return priorities
    
    def _adaptive_head_selection(self, available_heads: List[HeadType]) -> List[HeadType]:
        """적응적 헤드 선택"""
        
        # 현재 단계와 성능을 고려한 동적 선택
        if self.current_phase == LearningPhase.WARM_UP:
            # 워밍업 단계: 기본 헤드들만
            basic_heads = [HeadType.EMOTION_EMPATHY, HeadType.BENTHAM_FROMM]
            return [h for h in basic_heads if h in available_heads]
        
        elif self.current_phase == LearningPhase.COLLABORATIVE:
            # 협력 학습: 모든 헤드
            return available_heads
        
        elif self.current_phase == LearningPhase.SPECIALIZED:
            # 전문화 학습: 교대로 전문화
            cycle_length = 3
            head_index = (self.phase_step_count // cycle_length) % len(available_heads)
            return [available_heads[head_index]]
        
        elif self.current_phase == LearningPhase.INTEGRATION:
            # 통합 학습: 시너지 효과가 높은 조합
            synergistic_combinations = [
                [HeadType.EMOTION_EMPATHY, HeadType.BENTHAM_FROMM],
                [HeadType.BENTHAM_FROMM, HeadType.REGRET_LEARNING],
                [HeadType.SEMANTIC_SURD, HeadType.META_INTEGRATION]
            ]
            combination_index = (self.phase_step_count // 5) % len(synergistic_combinations)
            selected_combination = synergistic_combinations[combination_index]
            return [h for h in selected_combination if h in available_heads]
        
        return available_heads
    
    def update_phase(self, performance_metrics: Dict[str, float]):
        """학습 단계 업데이트"""
        self.step_count += 1
        self.phase_step_count += 1
        
        # 단계 전환 조건 확인
        should_advance = False
        
        if self.current_phase == LearningPhase.WARM_UP:
            # 워밍업 완료 조건: 1000 스텝 또는 손실 안정화
            if (self.phase_step_count >= 1000 or 
                performance_metrics.get('loss_stability', 0) > 0.8):
                should_advance = True
                next_phase = LearningPhase.COLLABORATIVE
        
        elif self.current_phase == LearningPhase.COLLABORATIVE:
            # 협력 학습 완료 조건: 시너지 효과 확인
            if (self.phase_step_count >= 2000 or
                performance_metrics.get('synergy_gain', 0) > 0.15):
                should_advance = True
                next_phase = LearningPhase.SPECIALIZED
        
        elif self.current_phase == LearningPhase.SPECIALIZED:
            # 전문화 완료 조건: 각 헤드별 성능 안정화
            if (self.phase_step_count >= 3000 or
                performance_metrics.get('head_stability', 0) > 0.9):
                should_advance = True
                next_phase = LearningPhase.INTEGRATION
        
        elif self.current_phase == LearningPhase.INTEGRATION:
            # 통합 학습 완료 조건: 전체 성능 수렴
            if (self.phase_step_count >= 2000 or
                performance_metrics.get('convergence_rate', 0) > 0.95):
                should_advance = True
                next_phase = LearningPhase.FINE_TUNING
        
        if should_advance:
            logger.info(f"학습 단계 전환: {self.current_phase.value} → {next_phase.value}")
            self.current_phase = next_phase
            self.phase_step_count = 0
            self.phase_progress = 0.0

class RealTimeMemoryMonitor:
    """실시간 GPU 메모리 모니터링 및 스왑 제어"""
    
    def __init__(self):
        self.monitoring_enabled = True
        self.warning_threshold = 70  # 70% 이상시 경고
        self.force_swap_threshold = 75  # 75% 이상시 강제 스왑
        self.emergency_threshold = 80  # 80% 이상시 긴급 정리
        self.memory_history = deque(maxlen=10)
        
    def check_memory_and_act(self, step_count: int = 0):
        """메모리 상태 확인 및 필요시 스왑 액션 수행"""
        if not self.monitoring_enabled:
            return True
            
        memory_info = get_gpu_memory_info()
        if memory_info is None:
            return True
            
        usage_percent = memory_info['usage_percent']
        self.memory_history.append(usage_percent)
        
        # 70% 미만: 정상 운영
        if usage_percent < self.warning_threshold:
            if step_count % 20 == 0:  # 20스텝마다 정상 상태 로그
                logger.debug(f"GPU 메모리 정상: {usage_percent:.1f}% 사용중")
            return True
        
        # 70-75%: 경고 및 예방적 스왑 준비
        elif usage_percent < self.force_swap_threshold:
            logger.warning(f"GPU 메모리 경고 레벨: {usage_percent:.1f}% - 예방적 스왑 준비")
            self._prepare_preventive_swap()
            return True
        
        # 75-80%: 강제 스왑 수행
        elif usage_percent < self.emergency_threshold:
            logger.error(f"GPU 메모리 위험 레벨: {usage_percent:.1f}% - 강제 스왑 수행")
            self._perform_force_swap()
            return True
        
        # 80% 초과: 긴급 정리
        else:
            logger.critical(f"GPU 메모리 긴급 상황: {usage_percent:.1f}% - 긴급 정리 수행")
            self._perform_emergency_cleanup()
            return False  # 학습 일시 중단 신호
    
    def _prepare_preventive_swap(self):
        """예방적 스왑 준비"""
        # LOW 우선순위 모델들 CPU 이동 준비
        logger.info("LOW 우선순위 모델들 CPU 이동 준비 중...")
    
    def _perform_force_swap(self):
        """강제 스왑 수행"""
        logger.info("낮은 우선순위 모델들 강제 스왑 수행 중...")
        _force_swap_low_priority_models()
        
        # 스왑 후 메모리 재확인
        memory_info = get_gpu_memory_info()
        if memory_info:
            logger.info(f"강제 스왑 완료: GPU 메모리 사용률 {memory_info['usage_percent']:.1f}%")
    
    def _perform_emergency_cleanup(self):
        """긴급 GPU 메모리 정리"""
        logger.critical("긴급 GPU 메모리 정리 수행 중...")
        _emergency_gpu_cleanup()
        
        # 정리 후 메모리 재확인
        memory_info = get_gpu_memory_info()
        if memory_info:
            logger.critical(f"긴급 정리 완료: GPU 메모리 사용률 {memory_info['usage_percent']:.1f}%")
    
    def get_memory_trend(self) -> str:
        """메모리 사용률 추세 반환"""
        if len(self.memory_history) < 3:
            return "insufficient_data"
        
        recent_avg = sum(self.memory_history[-3:]) / 3
        older_avg = sum(self.memory_history[:-3]) / max(1, len(self.memory_history) - 3)
        
        if recent_avg > older_avg + 5:
            return "increasing"
        elif recent_avg < older_avg - 5:
            return "decreasing"
        else:
            return "stable"

class UnifiedLearningSystem:
    """
    통합 학습 시스템 - 메인 클래스
    
    800M 파라미터 다중 헤드 통합 훈련 시스템
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or ADVANCED_CONFIG.get('unified_learning_config', {})
        
        # 헤드 훈련 설정
        self.head_configs = self._initialize_head_configs()
        
        # 핵심 컴포넌트들
        self.unified_backbone = RedHeartUnifiedBackbone()
        
        # 백본 training 모드 및 gradient 설정 강화
        self.unified_backbone.train()
        for param in self.unified_backbone.parameters():
            param.requires_grad_(True)
        logger.info(f"백본 gradient 설정 완료: {sum(1 for p in self.unified_backbone.parameters() if p.requires_grad)}/{sum(1 for p in self.unified_backbone.parameters())} 파라미터")
        
        self.trainer = MemoryEfficientTrainer(self.unified_backbone, self.head_configs, self)
        self.scheduler = UnifiedLearningScheduler(self.head_configs)
        
        # 스왑 매니저 및 시너지 시스템
        self.swap_manager = RedHeartDynamicSwapManager()
        self.synergy_system = IntelligentSynergySystem()
        
        # 헤드 호환성 매니저 - 헤드들을 스왑 매니저에 등록
        from head_compatibility_interface import HeadCompatibilityManager
        self.head_compatibility_manager = HeadCompatibilityManager(
            self.unified_backbone, self.swap_manager
        )
        
        # 실시간 메모리 모니터링 시스템
        self.memory_monitor = RealTimeMemoryMonitor()
        self.last_memory_check = 0
        self.memory_check_interval = 5  # 5스텝마다 메모리 확인
        
        # 훈련 상태
        self.is_training = False
        self.training_thread = None
        self.training_stats = {}
        
        # 체크포인트 관리
        self.checkpoint_dir = "checkpoints/unified_learning"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 실제 헤드 모듈 캐시
        self.cached_head_modules = {}
        
        # 초기화 완료 플래그
        self.initialized = False
        
        logger.info("UnifiedLearningSystem 초기화 완료")
    
    async def initialize_system(self):
        """시스템 비동기 초기화 - 스왑 매니저 및 헤드들을 사전 초기화"""
        if self.initialized:
            return
        
        logger.info("통합 학습 시스템 비동기 초기화 시작...")
        
        # 1. 스왑 매니저 먼저 초기화
        await self.swap_manager.initialize()
        logger.info("동적 스왑 매니저 초기화 완료")
        
        # 2. 헤드 호환성 매니저 초기화 (헤드들을 스왑 매니저에 등록)
        logger.info("헤드 호환성 매니저 초기화 중...")
        await self.head_compatibility_manager.initialize_all_heads()
        logger.info("모든 헤드 초기화 완료")
        
        # 3. 실제 헤드 모듈들을 캐시
        logger.info("실제 헤드 모듈 캐싱 중...")
        for head_type in self.head_configs.keys():
            try:
                real_head = await self._get_real_head_module(head_type)
                if real_head is not None:
                    self.cached_head_modules[head_type] = real_head
                    logger.info(f"헤드 {head_type.value} 캐싱 완료")
                else:
                    logger.warning(f"헤드 {head_type.value} 캐싱 실패: None 반환")
            except Exception as e:
                logger.error(f"헤드 {head_type.value} 캐싱 오류: {str(e)}")
        
        logger.info(f"헤드 모듈 캐싱 완료: {len(self.cached_head_modules)}/{len(self.head_configs)}개")
        
        # 4. 모든 헤드가 성공적으로 캐싱되었는지 검증 (fallback 없음)
        if len(self.cached_head_modules) != len(self.head_configs):
            missing_heads = [head_type.value for head_type in self.head_configs.keys() 
                           if head_type not in self.cached_head_modules]
            logger.error(f"헤드 초기화 실패: {missing_heads} 헤드들을 로딩할 수 없습니다.")
            logger.error("프로젝트 규칙에 따라 fallback 처리는 금지됩니다. 시스템을 종료합니다.")
            raise RuntimeError(f"필수 헤드 초기화 실패: {missing_heads}. HeadCompatibilityManager 확인 필요.")
        
        # 5. 각 헤드의 PyTorch 모듈 검증
        for head_type, head_module in self.cached_head_modules.items():
            if not isinstance(head_module, nn.Module):
                logger.error(f"헤드 {head_type.value}이 유효한 PyTorch 모듈이 아닙니다: {type(head_module)}")
                raise RuntimeError(f"헤드 {head_type.value} 모듈 타입 오류")
            
            # 파라미터 수 검증
            param_count = sum(p.numel() for p in head_module.parameters())
            if param_count == 0:
                logger.error(f"헤드 {head_type.value}에 파라미터가 없습니다.")
                raise RuntimeError(f"헤드 {head_type.value} 파라미터 부재")
            
            logger.info(f"헤드 {head_type.value} 검증 완료: {param_count:,}개 파라미터")
        
        # 6. 초기화 완료 플래그 설정
        self.initialized = True
        logger.info("통합 학습 시스템 비동기 초기화 완료 - 모든 헤드 검증됨")
    
    def _initialize_head_configs(self) -> Dict[HeadType, HeadTrainingConfig]:
        """헤드별 훈련 설정 초기화"""
        configs = {}
        
        # 감정 공감 헤드
        configs[HeadType.EMOTION_EMPATHY] = HeadTrainingConfig(
            head_type=HeadType.EMOTION_EMPATHY,
            learning_rate=3e-4,
            weight_decay=1e-4,
            loss_weight=1.2,  # 감정 처리 중요도 높음
            warmup_steps=500,
            scheduler_type="cosine",
            dropout_rate=0.1
        )
        
        # 벤담-프롬 윤리 헤드
        configs[HeadType.BENTHAM_FROMM] = HeadTrainingConfig(
            head_type=HeadType.BENTHAM_FROMM,
            learning_rate=2e-4,
            weight_decay=2e-4,
            loss_weight=1.5,  # 윤리 판단 가장 중요
            warmup_steps=800,
            scheduler_type="cosine",
            dropout_rate=0.15
        )
        
        # 의미 SURD 헤드
        configs[HeadType.SEMANTIC_SURD] = HeadTrainingConfig(
            head_type=HeadType.SEMANTIC_SURD,
            learning_rate=1e-4,
            weight_decay=1e-5,
            loss_weight=0.8,
            warmup_steps=1000,
            scheduler_type="linear",
            dropout_rate=0.05
        )
        
        # 후회 학습 헤드
        configs[HeadType.REGRET_LEARNING] = HeadTrainingConfig(
            head_type=HeadType.REGRET_LEARNING,
            learning_rate=2.5e-4,
            weight_decay=1e-4,
            loss_weight=1.0,
            warmup_steps=600,
            scheduler_type="cosine",
            dropout_rate=0.12
        )
        
        # 메타 통합 헤드
        configs[HeadType.META_INTEGRATION] = HeadTrainingConfig(
            head_type=HeadType.META_INTEGRATION,
            learning_rate=1.5e-4,
            weight_decay=5e-5,
            loss_weight=1.1,
            freeze_until_epoch=2,  # 다른 헤드들이 어느 정도 학습된 후 시작
            warmup_steps=1200,
            scheduler_type="cosine",
            dropout_rate=0.08
        )
        
        return configs
    
    async def train_unified_system(self, train_data_loader, 
                                 validation_data_loader=None,
                                 num_epochs: int = 10,
                                 training_strategy: TrainingStrategy = TrainingStrategy.ROUND_ROBIN):
        """통합 시스템 훈련"""
        
        logger.info(f"통합 학습 시작: {num_epochs} 에포크, 전략: {training_strategy.value}")
        
        # 시스템 초기화 (처음 훈련시에만)
        if not self.initialized:
            await self.initialize_system()
        
        self.is_training = True
        self.trainer.training_strategy = training_strategy
        
        try:
            for epoch in range(num_epochs):
                self.trainer.current_metrics.epoch = epoch
                self.scheduler.epoch_count = epoch
                
                # 에포크 시작
                epoch_start_time = time.time()
                epoch_losses = []
                epoch_metrics = []
                
                logger.info(f"에포크 {epoch+1}/{num_epochs} 시작")
                
                # 에포크용 활성 헤드 초기화
                available_heads = list(self.head_configs.keys())
                epoch_active_heads = self.scheduler.get_active_heads(training_strategy, available_heads)
                
                # 배치별 훈련
                for batch_idx, batch_data in enumerate(train_data_loader):
                    if not self.is_training:
                        break
                    
                    # 배치별 활성 헤드 선택 (동적 스케줄링 지원)
                    active_heads = self.scheduler.get_active_heads(training_strategy, available_heads)
                    
                    # 헤드 로딩 (스왑 매니저 사용)
                    await self._load_active_heads(active_heads)
                    
                    try:
                        # 훈련 스텝 실행
                        step_metrics = await self.trainer.train_step(batch_data, active_heads)
                        epoch_metrics.append(step_metrics)
                        epoch_losses.append(step_metrics.total_loss)
                        
                        # 실시간 메모리 모니터링 및 스왑 제어
                        total_steps = epoch * len(train_data_loader) + batch_idx
                        if total_steps % self.memory_check_interval == 0:
                            memory_ok = self.memory_monitor.check_memory_and_act(total_steps)
                            if not memory_ok:
                                logger.warning(f"메모리 부족으로 학습 스텝 {total_steps} 일시 정지")
                                # 긴급 정리 후 잠시 대기
                                await asyncio.sleep(2)
                                continue
                        
                        # 스케줄러 업데이트
                        performance_metrics = {
                            'loss_stability': self._calculate_loss_stability(epoch_losses),
                            'synergy_gain': step_metrics.synergy_gain,
                            'convergence_rate': self._calculate_convergence_rate(epoch_metrics)
                        }
                        self.scheduler.update_phase(performance_metrics)
                        
                        # 로깅 (매 100 스텝마다)
                        if batch_idx % 100 == 0:
                            await self._log_training_progress(step_metrics, batch_idx, active_heads)
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            logger.error("GPU 메모리 부족 - 배치 크기 조정")
                            self.trainer.adaptive_batch_manager.report_oom()
                            torch.cuda.empty_cache()
                            gc.collect()
                            continue
                        else:
                            raise
                
                # 에포크 완료 처리
                epoch_time = time.time() - epoch_start_time
                await self._complete_epoch(epoch, epoch_metrics, epoch_time)
                
                # 검증 (선택적)
                if validation_data_loader is not None:
                    # 검증을 위한 헤드 선택 (에포크 전체에서 사용된 헤드들)
                    await self._validate_model(validation_data_loader, epoch_active_heads)
                
                # 체크포인트 저장
                if (epoch + 1) % 2 == 0:  # 2 에포크마다 저장
                    await self._save_checkpoint(epoch)
        
        except Exception as e:
            logger.error(f"훈련 중 오류 발생: {str(e)}")
            raise
        
        finally:
            self.is_training = False
            logger.info("통합 학습 완료")
    
    async def _load_active_heads(self, active_heads: List[HeadType]):
        """활성 헤드들 로딩"""
        
        for head_type in active_heads:
            # 스왑 매니저를 통해 헤드 로딩
            await self.swap_manager.load_head_to_gpu(head_type.value)
            
            # 옵티마이저 등록 (처음 로딩시)
            if head_type.value not in self.trainer.optimizers:
                # HeadCompatibilityManager를 통해 실제 헤드 모듈 가져오기
                real_head = await self._get_real_head_module(head_type)
                if real_head is not None:
                    self.trainer.add_head_optimizer(head_type, real_head)
                else:
                    logger.warning(f"헤드 {head_type.value}의 실제 모듈을 가져올 수 없어 스킵합니다")
    
    async def _get_real_head_module(self, head_type: HeadType) -> Optional[nn.Module]:
        """HeadCompatibilityManager를 통해 헤드 어댑터 가져오기"""
        try:
            # 헤드 어댑터 가져오기
            head_adapter = self.head_compatibility_manager.head_adapters.get(head_type)
            if head_adapter is None:
                logger.error(f"헤드 어댑터를 찾을 수 없음: {head_type.value}")
                return None
            
            # 헤드 어댑터 초기화 확인
            if not head_adapter.initialized:
                await head_adapter.initialize_head()
            
            # 헤드 어댑터가 forward 메서드를 가지고 있는지 확인
            if hasattr(head_adapter, 'forward'):
                logger.info(f"헤드 어댑터 전체 반환: {head_type.value}")
                return head_adapter  # 헤드 어댑터 전체를 반환 (input_adapter 포함)
            else:
                # forward 메서드가 없으면 PyTorch 네트워크만 반환
                pytorch_network = None
                if hasattr(head_adapter, 'get_pytorch_network'):
                    pytorch_network = head_adapter.get_pytorch_network()
                
                if pytorch_network is not None:
                    logger.info(f"PyTorch 네트워크만 반환: {head_type.value}")
                    return pytorch_network
                else:
                    logger.warning(f"헤드 {head_type.value}에서 사용 가능한 모듈을 찾을 수 없음")
                    return None
                
        except Exception as e:
            logger.error(f"헤드 모듈 가져오기 실패 {head_type.value}: {str(e)}")
            return None
    
    def _calculate_loss_stability(self, recent_losses: List[float], window_size: int = 50) -> float:
        """손실 안정성 계산"""
        if len(recent_losses) < window_size:
            return 0.0
        
        recent_window = recent_losses[-window_size:]
        std_dev = np.std(recent_window)
        mean_loss = np.mean(recent_window)
        
        # 표준편차가 작을수록 안정적
        stability = 1.0 / (1.0 + std_dev / max(0.001, mean_loss))
        return min(1.0, stability)
    
    def _calculate_convergence_rate(self, metrics_history: List[TrainingMetrics]) -> float:
        """수렴 속도 계산"""
        if len(metrics_history) < 10:
            return 0.0
        
        recent_losses = [m.total_loss for m in metrics_history[-10:]]
        
        # 손실 감소 추세 분석
        if len(recent_losses) >= 2:
            loss_trend = recent_losses[-1] - recent_losses[0]
            if loss_trend < 0:  # 손실 감소 중
                convergence = min(1.0, abs(loss_trend) / recent_losses[0])
            else:
                convergence = 0.0
        else:
            convergence = 0.0
        
        return convergence
    
    async def _log_training_progress(self, metrics: TrainingMetrics, 
                                   batch_idx: int, active_heads: List[HeadType]):
        """훈련 진행상황 로깅"""
        
        head_names = [h.value for h in active_heads]
        memory_info = self.trainer.memory_monitor.get_memory_usage()
        
        logger.info(
            f"스텝 {metrics.step}, 배치 {batch_idx}: "
            f"손실={metrics.total_loss:.4f}, "
            f"활성헤드={head_names}, "
            f"GPU={memory_info['gpu_percent']:.1f}%, "
            f"처리속도={metrics.samples_per_second:.1f} samples/s"
        )
        
        # 헤드별 손실 로깅
        for head_name, loss in metrics.head_losses.items():
            logger.debug(f"  {head_name} 손실: {loss:.4f}")
    
    async def _complete_epoch(self, epoch: int, epoch_metrics: List[TrainingMetrics], 
                            epoch_time: float):
        """에포크 완료 처리"""
        
        # 에포크 통계 계산
        avg_loss = np.mean([m.total_loss for m in epoch_metrics])
        avg_memory = np.mean([m.memory_usage for m in epoch_metrics])
        total_samples = sum(m.samples_per_second * m.training_time for m in epoch_metrics)
        
        # 효율성 메트릭
        memory_efficiency = self.trainer.memory_monitor.get_memory_efficiency()
        
        epoch_stats = {
            'epoch': epoch,
            'avg_loss': avg_loss,
            'epoch_time': epoch_time,
            'total_samples': total_samples,
            'avg_memory_usage': avg_memory,
            'memory_efficiency': memory_efficiency,
            'learning_phase': self.scheduler.current_phase.value,
            'training_strategy': self.trainer.training_strategy.value
        }
        
        self.training_stats[f'epoch_{epoch}'] = epoch_stats
        
        logger.info(
            f"에포크 {epoch} 완료: "
            f"평균손실={avg_loss:.4f}, "
            f"시간={epoch_time:.1f}s, "
            f"메모리효율={memory_efficiency:.2%}, "
            f"학습단계={self.scheduler.current_phase.value}"
        )
    
    async def _validate_model(self, validation_data_loader, active_heads: List[HeadType]):
        """모델 검증"""
        
        logger.info("모델 검증 시작...")
        
        self.trainer.eval()
        validation_losses = []
        
        try:
            with torch.no_grad():
                for batch_data in validation_data_loader:
                    # 검증 순전파
                    outputs = self.trainer.forward_with_checkpointing(batch_data, active_heads)
                    
                    # 검증 손실 계산
                    losses = self.trainer._calculate_losses(outputs, batch_data, active_heads)
                    total_loss = sum(losses.values()) / len(losses)
                    validation_losses.append(float(total_loss.item()))
        
        finally:
            self.trainer.train()  # 훈련 모드로 복원
        
        avg_val_loss = np.mean(validation_losses)
        logger.info(f"검증 완료: 평균 검증 손실 = {avg_val_loss:.4f}")
        
        return avg_val_loss
    
    async def _save_checkpoint(self, epoch: int):
        """체크포인트 저장"""
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f"unified_model_epoch_{epoch}.pth")
        
        checkpoint = {
            'epoch': epoch,
            'unified_backbone_state_dict': self.unified_backbone.state_dict(),
            'optimizers_state_dict': {name: opt.state_dict() for name, opt in self.trainer.optimizers.items()},
            'schedulers_state_dict': {name: sch.state_dict() for name, sch in self.trainer.schedulers.items()},
            'training_stats': self.training_stats,
            'current_phase': self.scheduler.current_phase.value,
            'step_count': self.scheduler.step_count,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"체크포인트 저장: {checkpoint_path}")
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """훈련 통계 반환"""
        
        current_metrics = self.trainer.current_metrics
        memory_stats = self.trainer.memory_monitor.get_memory_usage()
        
        stats = {
            'current_metrics': {
                'epoch': current_metrics.epoch,
                'step': current_metrics.step,
                'total_loss': current_metrics.total_loss,
                'head_losses': current_metrics.head_losses,
                'learning_rates': current_metrics.learning_rates,
                'samples_per_second': current_metrics.samples_per_second,
                'memory_efficiency': self.trainer.memory_monitor.get_memory_efficiency()
            },
            'memory_stats': memory_stats,
            'training_phase': self.scheduler.current_phase.value,
            'training_strategy': self.trainer.training_strategy.value,
            'batch_size': self.trainer.adaptive_batch_manager.current_batch_size,
            'epoch_stats': self.training_stats
        }
        
        return stats

# 사용 예시 함수
async def example_usage():
    """통합 학습 시스템 사용 예시"""
    
    # 가상의 데이터 로더 생성
    class DummyDataLoader:
        def __init__(self, num_batches=100):
            self.num_batches = num_batches
        
        def __iter__(self):
            for i in range(self.num_batches):
                yield {
                    'text': f'샘플 텍스트 {i}',
                    'batch_size': 4,
                    'labels': torch.randint(0, 10, (4,))
                }
    
    # 통합 학습 시스템 생성
    learning_system = UnifiedLearningSystem()
    
    # 데이터 로더
    train_loader = DummyDataLoader(num_batches=500)
    val_loader = DummyDataLoader(num_batches=50)
    
    print("=== 통합 학습 시스템 테스트 ===")
    
    # 훈련 실행
    await learning_system.train_unified_system(
        train_data_loader=train_loader,
        validation_data_loader=val_loader,
        num_epochs=3,
        training_strategy=TrainingStrategy.ROUND_ROBIN
    )
    
    # 훈련 통계 출력
    stats = learning_system.get_training_statistics()
    print(f"\n=== 훈련 완료 통계 ===")
    print(f"최종 손실: {stats['current_metrics']['total_loss']:.4f}")
    print(f"훈련 단계: {stats['training_phase']}")
    print(f"메모리 효율성: {stats['current_metrics']['memory_efficiency']:.2%}")
    print(f"처리 속도: {stats['current_metrics']['samples_per_second']:.1f} samples/s")
    
    # 헤드별 손실
    print(f"\n=== 헤드별 손실 ===")
    for head_name, loss in stats['current_metrics']['head_losses'].items():
        print(f"{head_name}: {loss:.4f}")

if __name__ == "__main__":
    asyncio.run(example_usage())