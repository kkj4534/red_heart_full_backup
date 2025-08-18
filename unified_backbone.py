#!/usr/bin/env python3
"""
Red Heart AI 통합 백본 네트워크
15M 파라미터 - 효율적인 공유 표현 학습
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class RedHeartUnifiedBackbone(nn.Module):
    """
    통합 백본 네트워크 (68M 파라미터 - 330M의 1.364배)
    - 강화된 공통 특징 추출 및 태스크별 프로젝션
    - 확장된 트랜스포머 아키텍처 (8층)
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # 설정 (68M 파라미터를 위한 조정)
        self.input_dim = config.get('input_dim', 768)
        self.hidden_dim = config.get('d_model', 896)  # 768의 1.17배
        self.num_layers = config.get('num_layers', 8)  # 8층
        self.num_heads = config.get('num_heads', 14)  # 14 헤드
        self.dropout = config.get('dropout', 0.1)
        self.task_dim = config.get('task_dim', 896)  # 태스크 출력 차원
        
        # 입력 프로젝션 (1.2M)
        self.input_projection = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
        
        # 트랜스포머 인코더 레이어 (약 42M)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=config.get('feedforward_dim', 3584),  # 896*4
            dropout=self.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN for stability
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
        
        # 태스크별 프로젝션 (각 0.8M × 4 = 3.2M)
        self.task_projections = nn.ModuleDict({
            'emotion': self._create_task_projection(),
            'bentham': self._create_task_projection(),
            'regret': self._create_task_projection(),
            'surd': self._create_task_projection()
        })
        
        # 태스크별 특화 레이어 (각 0.5M × 4 = 2M)
        self.task_specialization = nn.ModuleDict({
            'emotion': self._create_task_specialization(),
            'bentham': self._create_task_specialization(),
            'regret': self._create_task_specialization(),
            'surd': self._create_task_specialization()
        })
        
        # 태스크별 어텐션 가중치 (1M)
        self.task_attention = nn.ModuleDict({
            'emotion': self._create_task_attention(),
            'bentham': self._create_task_attention(),
            'regret': self._create_task_attention(),
            'surd': self._create_task_attention()
        })
        
        # 잔차 연결을 위한 skip connection
        self.skip_projection = nn.Linear(self.input_dim, self.task_dim)
        
        self._init_weights()
        
        # 파라미터 수 계산 및 로깅
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"백본 총 파라미터: {total_params:,} ({total_params/1e6:.2f}M)")
        logger.info(f"학습 가능 파라미터: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    def _create_task_projection(self) -> nn.Module:
        """태스크별 프로젝션 레이어 생성"""
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.task_dim),
            nn.LayerNorm(self.task_dim)
        )
    
    def _create_task_specialization(self) -> nn.Module:
        """태스크별 특화 레이어"""
        return nn.Sequential(
            nn.Linear(self.task_dim, self.task_dim),
            nn.LayerNorm(self.task_dim),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
    
    def _create_task_attention(self) -> nn.Module:
        """태스크별 어텐션 메커니즘"""
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def _init_weights(self):
        """가중치 초기화"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        x: torch.Tensor,
        task: Optional[str] = None,
        return_all_tasks: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        순전파
        
        Args:
            x: 입력 텐서 (batch_size, seq_len, input_dim) 또는 (batch_size, input_dim)
            task: 특정 태스크 ('emotion', 'bentham', 'regret', 'surd')
            return_all_tasks: 모든 태스크 출력 반환 여부
        
        Returns:
            태스크별 특징 딕셔너리
        """
        # 입력 차원 조정
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_dim)
        
        batch_size, seq_len, _ = x.shape
        
        # Skip connection을 위한 원본 저장
        skip = self.skip_projection(x.mean(dim=1))  # (batch, task_dim)
        
        # 입력 프로젝션
        x = self.input_projection(x)  # (batch, seq_len, hidden_dim)
        
        # 트랜스포머 인코딩
        encoded = self.transformer_encoder(x)  # (batch, seq_len, hidden_dim)
        
        # 시퀀스 차원 집계 (평균 풀링 + 어텐션)
        outputs = {}
        
        for task_name in self.task_projections.keys():
            if task and task_name != task and not return_all_tasks:
                continue
            
            # 태스크별 어텐션 가중치 계산
            attention_weights = self.task_attention[task_name](encoded)  # (batch, seq_len, 1)
            attention_weights = F.softmax(attention_weights, dim=1)
            
            # 어텐션 적용 집계
            weighted_encoded = (encoded * attention_weights).sum(dim=1)  # (batch, hidden_dim)
            
            # 태스크별 프로젝션
            task_features = self.task_projections[task_name](weighted_encoded)  # (batch, task_dim)
            
            # 태스크별 특화
            task_features = self.task_specialization[task_name](task_features)
            
            # Skip connection 추가
            task_features = task_features + skip
            
            outputs[task_name] = task_features
        
        return outputs
    
    def get_task_features(self, x: torch.Tensor, task: str) -> torch.Tensor:
        """특정 태스크의 특징만 추출"""
        outputs = self.forward(x, task=task)
        return outputs[task]
    
    def freeze_backbone(self):
        """백본 가중치 고정 (헤드만 학습)"""
        for name, param in self.named_parameters():
            if 'task_projections' not in name:
                param.requires_grad = False
    
    def unfreeze_backbone(self):
        """백본 가중치 학습 가능"""
        for param in self.parameters():
            param.requires_grad = True


class MultiTaskHead(nn.Module):
    """
    멀티태스크 헤드 베이스 클래스
    각 태스크별 헤드가 상속받아 구현
    """
    
    def __init__(self, input_dim: int = 384, output_dim: int = 7):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("서브클래스에서 구현 필요")
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """태스크별 손실 계산"""
        raise NotImplementedError("서브클래스에서 구현 필요")