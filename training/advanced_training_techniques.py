"""
Advanced Training Techniques
Label Smoothing, R-Drop, EMA, LLRD 등 고급 학습 기법 통합
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from copy import deepcopy
from collections import defaultdict

logger = logging.getLogger(__name__)


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss
    과적합 방지 및 일반화 성능 향상
    """
    
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        """
        Args:
            num_classes: 클래스 개수
            smoothing: 스무딩 정도 (0.0 ~ 1.0)
        """
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 예측값 (batch_size, num_classes)
            target: 타겟 라벨 (batch_size)
            
        Returns:
            Label smoothed loss
        """
        if self.smoothing == 0:
            return F.cross_entropy(pred, target)
        
        pred = pred.log_softmax(dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


class RDropLoss(nn.Module):
    """
    R-Drop: Regularized Dropout for Neural Networks
    동일 입력에 대한 두 번의 forward pass 결과를 일관되게 유지
    """
    
    def __init__(self, alpha: float = 1.0, kl_weight: float = 1.0):
        """
        Args:
            alpha: R-Drop 강도
            kl_weight: KL divergence 가중치
        """
        super().__init__()
        self.alpha = alpha
        self.kl_weight = kl_weight
        
    def forward(self, 
                logits1: torch.Tensor, 
                logits2: torch.Tensor,
                labels: torch.Tensor,
                base_loss_fn: nn.Module = None) -> torch.Tensor:
        """
        Args:
            logits1: 첫 번째 forward pass 결과
            logits2: 두 번째 forward pass 결과
            labels: 타겟 라벨
            base_loss_fn: 기본 손실 함수
            
        Returns:
            R-Drop loss
        """
        if base_loss_fn is None:
            base_loss_fn = nn.CrossEntropyLoss()
        
        # 기본 손실
        loss1 = base_loss_fn(logits1, labels)
        loss2 = base_loss_fn(logits2, labels)
        base_loss = (loss1 + loss2) / 2
        
        # KL divergence
        p = F.log_softmax(logits1, dim=-1)
        q = F.log_softmax(logits2, dim=-1)
        
        kl_loss1 = F.kl_div(p, q.exp(), reduction='batchmean')
        kl_loss2 = F.kl_div(q, p.exp(), reduction='batchmean')
        kl_loss = (kl_loss1 + kl_loss2) / 2
        
        # 전체 손실
        total_loss = base_loss + self.alpha * self.kl_weight * kl_loss
        
        return total_loss


class ExponentialMovingAverage:
    """
    EMA (Exponential Moving Average)
    모델 파라미터의 지수 이동 평균을 유지하여 안정성 향상
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        """
        Args:
            model: 대상 모델
            decay: EMA decay rate
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # 초기 shadow 파라미터 생성
        self.register()
        
    def register(self):
        """모델 파라미터를 shadow에 등록"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Shadow 파라미터 업데이트"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Shadow 파라미터를 모델에 적용"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        """원본 파라미터로 복원"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self) -> Dict:
        """EMA 상태 저장"""
        return {
            'shadow': self.shadow,
            'decay': self.decay
        }
    
    def load_state_dict(self, state_dict: Dict):
        """EMA 상태 로드"""
        self.shadow = state_dict['shadow']
        self.decay = state_dict['decay']


class LayerWiseLRDecay:
    """
    LLRD (Layer-wise Learning Rate Decay)
    레이어별로 다른 학습률 적용 (깊은 레이어일수록 낮은 학습률)
    """
    
    def __init__(self,
                 model: nn.Module,
                 base_lr: float = 1e-4,
                 decay_rate: float = 0.95,  # 8개 레이어에 적절한 값 (0.95^7 = 0.7)
                 num_layers: Optional[int] = None):
        """
        Args:
            model: 대상 모델
            base_lr: 기본 학습률
            decay_rate: 레이어별 감쇠율
            num_layers: 전체 레이어 수 (None이면 자동 계산)
        """
        self.model = model
        self.base_lr = base_lr
        self.decay_rate = decay_rate
        
        # 레이어 수 계산
        if num_layers is None:
            self.num_layers = self._count_layers()
        else:
            self.num_layers = num_layers
        
        # 파라미터 그룹 생성
        self.param_groups = self._create_param_groups()
        
        logger.info(f"✅ LLRD 초기화: {self.num_layers}개 레이어")
        logger.info(f"  - Base LR: {base_lr:.1e}")
        logger.info(f"  - Decay Rate: {decay_rate}")
    
    def _count_layers(self) -> int:
        """모델의 레이어 수 계산 - 주요 encoder 레이어만 카운트"""
        import re
        layer_indices = set()
        
        # transformer_encoder.layers.숫자 패턴으로 정확히 매칭
        for name, _ in self.model.named_modules():
            # backbone.transformer_encoder.layers.0 형식
            match = re.search(r'transformer_encoder\.layers\.(\d+)$', name)
            if match:
                layer_indices.add(int(match.group(1)))
        
        # 레이어가 없으면 기본값 8 사용 (Red Heart 백본 기본값)
        layer_count = len(layer_indices) if layer_indices else 8
        return layer_count
    
    def _create_param_groups(self) -> List[Dict]:
        """레이어별 파라미터 그룹 생성"""
        param_groups = []
        
        # 레이어별로 그룹화
        layer_params = defaultdict(list)
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # 레이어 인덱스 추정
            layer_idx = self._get_layer_index(name)
            layer_params[layer_idx].append(param)
        
        # 각 레이어에 대한 파라미터 그룹 생성
        for layer_idx in sorted(layer_params.keys()):
            # 깊은 레이어일수록 낮은 학습률 (LLRD 원칙)
            lr_scale = self.decay_rate ** (self.num_layers - layer_idx - 1)
            layer_lr = self.base_lr * lr_scale
            
            param_groups.append({
                'params': layer_params[layer_idx],
                'lr': layer_lr,
                'layer_idx': layer_idx
            })
            
            logger.debug(f"  Layer {layer_idx}: LR={layer_lr:.1e} ({len(layer_params[layer_idx])} params)")
        
        return param_groups
    
    def _get_layer_index(self, param_name: str) -> int:
        """파라미터 이름으로부터 레이어 인덱스 추정"""
        # 간단한 휴리스틱: 숫자 추출
        import re
        numbers = re.findall(r'\d+', param_name)
        if numbers:
            return min(int(numbers[0]), self.num_layers - 1)
        return 0
    
    def get_optimizer(self, optimizer_class=torch.optim.AdamW, **kwargs) -> torch.optim.Optimizer:
        """LLRD가 적용된 옵티마이저 생성"""
        return optimizer_class(self.param_groups, **kwargs)


class MixupAugmentation:
    """
    Mixup Data Augmentation
    두 샘플을 선형 보간하여 학습 데이터 증강
    """
    
    def __init__(self, alpha: float = 0.2):
        """
        Args:
            alpha: Beta 분포의 파라미터
        """
        self.alpha = alpha
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Args:
            x: 입력 데이터 (batch_size, ...)
            y: 타겟 라벨 (batch_size, ...)
            
        Returns:
            mixed_x, y_a, y_b, lam
        """
        batch_size = x.size(0)
        
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index]
        
        return mixed_x, y, y[index], lam


class AdvancedTrainingManager:
    """
    고급 학습 기법 통합 관리자
    """
    
    def __init__(self,
                 enable_label_smoothing: bool = True,
                 enable_rdrop: bool = True,
                 enable_ema: bool = True,
                 enable_llrd: bool = True,
                 enable_mixup: bool = False,
                 label_smoothing: float = 0.1,
                 rdrop_alpha: float = 1.0,
                 ema_decay: float = 0.999,
                 llrd_decay: float = 0.8,
                 mixup_alpha: float = 0.2):
        """
        Args:
            enable_*: 각 기법 활성화 여부
            *: 각 기법의 하이퍼파라미터
        """
        self.config = {
            'label_smoothing': {'enabled': enable_label_smoothing, 'smoothing': label_smoothing},
            'rdrop': {'enabled': enable_rdrop, 'alpha': rdrop_alpha},
            'ema': {'enabled': enable_ema, 'decay': ema_decay},
            'llrd': {'enabled': enable_llrd, 'decay': llrd_decay},
            'mixup': {'enabled': enable_mixup, 'alpha': mixup_alpha}
        }
        
        # 컴포넌트 초기화
        self.label_smoothing_loss = None
        self.rdrop_loss = None
        self.ema = None
        self.llrd = None
        self.mixup = None
        
        logger.info("✅ Advanced Training Manager 초기화")
        for technique, cfg in self.config.items():
            if cfg['enabled']:
                logger.info(f"  - {technique}: 활성화")
    
    def initialize(self, model: nn.Module, num_classes: int = 6, base_lr: float = 1e-4):
        """
        컴포넌트 초기화
        
        Args:
            model: 학습 모델
            num_classes: 클래스 수
            base_lr: 기본 학습률
        """
        # Label Smoothing
        if self.config['label_smoothing']['enabled']:
            self.label_smoothing_loss = LabelSmoothingLoss(
                num_classes=num_classes,
                smoothing=self.config['label_smoothing']['smoothing']
            )
        
        # R-Drop
        if self.config['rdrop']['enabled']:
            self.rdrop_loss = RDropLoss(alpha=self.config['rdrop']['alpha'])
        
        # EMA
        if self.config['ema']['enabled']:
            self.ema = ExponentialMovingAverage(
                model=model,
                decay=self.config['ema']['decay']
            )
        
        # LLRD
        if self.config['llrd']['enabled']:
            self.llrd = LayerWiseLRDecay(
                model=model,
                base_lr=base_lr,
                decay_rate=self.config['llrd']['decay']
            )
        
        # Mixup
        if self.config['mixup']['enabled']:
            self.mixup = MixupAugmentation(alpha=self.config['mixup']['alpha'])
        
        logger.info("✅ Advanced Training 컴포넌트 초기화 완료")
    
    def compute_loss(self, 
                    model: nn.Module,
                    inputs: torch.Tensor,
                    labels: torch.Tensor,
                    base_loss_fn: Optional[nn.Module] = None) -> torch.Tensor:
        """
        통합 손실 계산
        
        Args:
            model: 모델
            inputs: 입력 데이터
            labels: 라벨
            base_loss_fn: 기본 손실 함수
            
        Returns:
            최종 손실
        """
        total_loss = 0
        
        # Mixup 적용
        if self.mixup and self.config['mixup']['enabled'] and model.training:
            inputs, labels_a, labels_b, lam = self.mixup(inputs, labels)
            
            # Forward pass
            outputs = model(inputs)
            
            # Mixup loss
            if self.label_smoothing_loss and self.config['label_smoothing']['enabled']:
                loss_a = self.label_smoothing_loss(outputs, labels_a)
                loss_b = self.label_smoothing_loss(outputs, labels_b)
            else:
                loss_fn = base_loss_fn or nn.CrossEntropyLoss()
                loss_a = loss_fn(outputs, labels_a)
                loss_b = loss_fn(outputs, labels_b)
            
            total_loss = lam * loss_a + (1 - lam) * loss_b
            
        # R-Drop 적용
        elif self.rdrop_loss and self.config['rdrop']['enabled'] and model.training:
            # 두 번의 forward pass
            outputs1 = model(inputs)
            outputs2 = model(inputs)
            
            # R-Drop loss
            if self.label_smoothing_loss and self.config['label_smoothing']['enabled']:
                base_fn = self.label_smoothing_loss
            else:
                base_fn = base_loss_fn or nn.CrossEntropyLoss()
            
            total_loss = self.rdrop_loss(outputs1, outputs2, labels, base_fn)
            
        # 일반 forward
        else:
            outputs = model(inputs)
            
            if self.label_smoothing_loss and self.config['label_smoothing']['enabled']:
                total_loss = self.label_smoothing_loss(outputs, labels)
            else:
                loss_fn = base_loss_fn or nn.CrossEntropyLoss()
                total_loss = loss_fn(outputs, labels)
        
        return total_loss
    
    def step(self):
        """학습 스텝 후 업데이트"""
        # EMA 업데이트
        if self.ema and self.config['ema']['enabled']:
            self.ema.update()
    
    def apply_ema(self):
        """EMA 파라미터 적용"""
        if self.ema and self.config['ema']['enabled']:
            self.ema.apply_shadow()
    
    def restore_ema(self):
        """EMA 파라미터 복원"""
        if self.ema and self.config['ema']['enabled']:
            self.ema.restore()
    
    def get_optimizer(self, model: nn.Module, **kwargs) -> torch.optim.Optimizer:
        """최적화기 생성"""
        if self.llrd and self.config['llrd']['enabled']:
            return self.llrd.get_optimizer(**kwargs)
        else:
            return torch.optim.AdamW(model.parameters(), **kwargs)
    
    def state_dict(self) -> Dict:
        """상태 저장"""
        state = {'config': self.config}
        
        if self.ema:
            state['ema'] = self.ema.state_dict()
        
        return state
    
    def load_state_dict(self, state_dict: Dict):
        """상태 로드"""
        self.config = state_dict['config']
        
        if 'ema' in state_dict and self.ema:
            self.ema.load_state_dict(state_dict['ema'])