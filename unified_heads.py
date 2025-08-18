#!/usr/bin/env python3
"""
Red Heart AI 통합 태스크 헤드
80M 파라미터 - 태스크별 특화 학습
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import logging
from unified_backbone import MultiTaskHead

logger = logging.getLogger(__name__)


class EmotionHead(MultiTaskHead):
    """
    감정 분석 헤드 (30M 파라미터 - 22M의 1.364배)
    - 7개 기본 감정 + 문화적 감정 처리
    - MoE 11개 전문가 시스템
    """
    
    def __init__(self, input_dim: int = 896):
        super().__init__(input_dim, output_dim=7)
        
        # 기본 감정 처리 레이어 (7M)
        self.base_emotion = nn.Sequential(
            nn.Linear(input_dim, 1536),
            nn.LayerNorm(1536),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1536, 1152),
            nn.LayerNorm(1152),
            nn.GELU(),
            nn.Linear(1152, 768),
            nn.LayerNorm(768)
        )
        
        # MoE 전문가 시스템 (11M)
        self.num_experts = 11
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(768, 576),
                nn.LayerNorm(576),
                nn.GELU(),
                nn.Linear(576, 384),
                nn.GELU(),
                nn.Linear(384, 7)
            ) for _ in range(self.num_experts)
        ])
        
        # 게이팅 네트워크 (1.5M)
        self.gating = nn.Sequential(
            nn.Linear(768, 384),
            nn.GELU(),
            nn.Linear(384, self.num_experts),
            nn.Softmax(dim=-1)
        )
        
        # 계층적 감정 처리 (5.5M)
        self.hierarchical = nn.ModuleList([
            nn.Sequential(
                nn.Linear(768, 576),
                nn.LayerNorm(576),
                nn.GELU(),
                nn.Linear(576, 384),
                nn.LayerNorm(384),
                nn.GELU(),
                nn.Linear(384, 192),
                nn.Linear(192, 7)
            ) for _ in range(3)  # 3개 계층
        ])
        
        # 문화적 감정 적응 (3M)
        self.cultural_emotion = nn.Sequential(
            nn.Linear(768, 576),
            nn.LayerNorm(576),
            nn.GELU(),
            nn.Linear(576, 384),
            nn.GELU(),
            nn.Linear(384, 3)  # 정, 한, 체면
        )
        
        # 어텐션 메커니즘 (2M)
        self.attention = nn.MultiheadAttention(
            embed_dim=768,
            num_heads=12,
            dropout=0.1,
            batch_first=True
        )
        
        # 최종 출력 레이어 (1M)
        self.output_layer = nn.Sequential(
            nn.Linear(768 + 7 * 3 + 3, 384),  # base + hierarchical + cultural
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Linear(384, 7)
        )
        
        self._log_params()
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """순전파"""
        batch_size = x.shape[0]
        
        # 기본 감정 처리
        base_features = self.base_emotion(x)  # (batch, 768)
        
        # MoE 처리
        gate_weights = self.gating(base_features)  # (batch, num_experts)
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_out = expert(base_features)  # (batch, 7)
            weighted_out = expert_out * gate_weights[:, i:i+1]
            expert_outputs.append(weighted_out)
        moe_output = torch.stack(expert_outputs, dim=1).sum(dim=1)  # (batch, 7)
        
        # 계층적 처리
        hierarchical_outputs = []
        for layer in self.hierarchical:
            h_out = layer(base_features)  # (batch, 7)
            hierarchical_outputs.append(h_out)
        hierarchical_concat = torch.cat(hierarchical_outputs, dim=-1)  # (batch, 21)
        
        # 문화적 감정
        cultural = self.cultural_emotion(base_features)  # (batch, 3)
        
        # 어텐션 적용
        attn_out, _ = self.attention(
            base_features.unsqueeze(1),
            base_features.unsqueeze(1),
            base_features.unsqueeze(1)
        )
        attn_features = attn_out.squeeze(1)  # (batch, 768)
        
        # 최종 결합
        combined = torch.cat([
            attn_features,
            hierarchical_concat,
            cultural
        ], dim=-1)  # (batch, 768 + 21 + 3)
        
        output = self.output_layer(combined)  # (batch, 7)
        
        return {
            'emotions': output,
            'moe_output': moe_output,
            'cultural': cultural,
            'attention_features': attn_features
        }
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Focal Loss for emotion (Joy 편향 해결)"""
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** 2.0) * ce_loss  # gamma=2.0
        return focal_loss.mean()
    
    def _log_params(self):
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"감정 헤드 파라미터: {total:,} ({total/1e6:.2f}M)")


class BenthamHead(MultiTaskHead):
    """
    벤담 윤리 헤드 (27M 파라미터 - 20M의 1.364배)
    - 10요소 공리주의 계산
    - 6개 윤리 전문가
    """
    
    def __init__(self, input_dim: int = 896):
        super().__init__(input_dim, output_dim=10)
        
        # 기본 윤리 처리 (5.5M)
        self.base_ethics = nn.Sequential(
            nn.Linear(input_dim, 1152),
            nn.LayerNorm(1152),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1152, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Linear(768, 576)
        )
        
        # MoE 윤리 전문가 (8M)
        self.ethical_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(576, 768),
                nn.LayerNorm(768),
                nn.GELU(),
                nn.Linear(768, 384),
                nn.GELU(),
                nn.Linear(384, 10)
            ) for _ in range(6)  # 6개 전문가
        ])
        
        # 6층 가중치 시스템 (5.5M)
        self.weight_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(576, 384),
                nn.LayerNorm(384),
                nn.GELU(),
                nn.Linear(384, 192),
                nn.Linear(192, 6)
            ) for _ in range(6)  # 6개 가중치 레이어
        ])
        
        # 법률 위험 평가 (4M)
        self.legal_risk = nn.ModuleList([
            nn.Sequential(
                nn.Linear(576, 384),
                nn.LayerNorm(384),
                nn.GELU(),
                nn.Linear(384, 192),
                nn.Linear(192, 1)
            ) for _ in range(5)  # 5개 도메인
        ])
        
        # 시간적 분석 LSTM (3M)
        self.temporal = nn.LSTM(
            input_size=576,
            hidden_size=384,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # 최종 출력 (1.5M)
        self.output_layer = nn.Sequential(
            nn.Linear(576 + 10 * 6 + 6 * 6 + 5 + 384, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Linear(768, 384),
            nn.Linear(384, 10)
        )
        
        self._log_params()
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """순전파"""
        batch_size = x.shape[0]
        
        # 기본 처리
        base_features = self.base_ethics(x)  # (batch, 576)
        
        # MoE 윤리 전문가
        expert_outputs = []
        for expert in self.ethical_experts:
            expert_out = expert(base_features)
            expert_outputs.append(expert_out)
        experts_concat = torch.cat(expert_outputs, dim=-1)  # (batch, 80)
        
        # 6층 가중치
        weight_outputs = []
        for layer in self.weight_layers:
            weight_out = layer(base_features)
            weight_outputs.append(weight_out)
        weights_concat = torch.cat(weight_outputs, dim=-1)  # (batch, 36)
        
        # 법률 위험
        legal_outputs = []
        for assessor in self.legal_risk:
            legal_out = assessor(base_features)
            legal_outputs.append(legal_out)
        legal_concat = torch.cat(legal_outputs, dim=-1)  # (batch, 5)
        
        # 시간적 분석
        temporal_in = base_features.unsqueeze(1)  # (batch, 1, 768)
        temporal_out, _ = self.temporal(temporal_in)
        temporal_features = temporal_out.squeeze(1)  # (batch, 512)
        
        # 최종 결합
        combined = torch.cat([
            base_features,
            experts_concat,
            weights_concat,
            legal_concat,
            temporal_features
        ], dim=-1)
        
        output = self.output_layer(combined)
        
        return {
            'bentham_scores': output,
            'expert_outputs': expert_outputs,
            'weights': weights_concat,
            'legal_risk': legal_concat
        }
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """벤담 손실 (MSE + 극단값 페널티)"""
        mse_loss = F.mse_loss(predictions, targets)
        
        # 극단값 페널티 (0.3~2.5 범위 벗어나면)
        penalty = torch.relu(predictions - 2.5) + torch.relu(0.3 - predictions)
        penalty_loss = penalty.mean()
        
        return mse_loss + 0.1 * penalty_loss
    
    def _log_params(self):
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"벤담 헤드 파라미터: {total:,} ({total/1e6:.2f}M)")


class RegretHead(MultiTaskHead):
    """
    후회 분석 헤드 (30M 파라미터 - 22M의 1.364배)
    - 반사실적 추론
    - 3뷰 시나리오 (낙관/중도/비관)
    """
    
    def __init__(self, input_dim: int = 896):
        super().__init__(input_dim, output_dim=1)
        
        # 기본 후회 처리 (7M)
        self.base_regret = nn.Sequential(
            nn.Linear(input_dim, 1536),
            nn.LayerNorm(1536),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1536, 1152),
            nn.LayerNorm(1152),
            nn.GELU(),
            nn.Linear(1152, 768)
        )
        
        # 3뷰 시나리오 네트워크 (8M)
        self.scenario_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(768, 768),
                nn.LayerNorm(768),
                nn.GELU(),
                nn.Linear(768, 576),
                nn.GELU(),
                nn.Linear(576, 384),
                nn.Linear(384, 1)
            ) for _ in range(3)  # 낙관, 중도, 비관
        ])
        
        # 반사실적 추론 GRU (5.5M)
        self.counterfactual = nn.GRU(
            input_size=768,
            hidden_size=576,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        self.counterfactual_attention = nn.MultiheadAttention(
            embed_dim=576,
            num_heads=12,
            dropout=0.1,
            batch_first=True
        )
        
        # 불확실성 정량화 (2M)
        self.uncertainty = nn.ModuleList([
            nn.Sequential(
                nn.Linear(768, 384),
                nn.LayerNorm(384),
                nn.GELU(),
                nn.Dropout(0.2),  # 높은 dropout으로 불확실성 모델링
                nn.Linear(384, 192),
                nn.Linear(192, 1)
            ) for _ in range(5)  # 5개 dropout 네트워크
        ])
        
        # 시간 전파 LSTM (2M)
        self.temporal_propagation = nn.LSTM(
            input_size=768,
            hidden_size=384,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # 최종 출력 (1.5M)
        self.output_layer = nn.Sequential(
            nn.Linear(768 + 3 + 576 + 5 + 384, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Linear(768, 384),
            nn.Linear(384, 1)
        )
        
        self._log_params()
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """순전파"""
        batch_size = x.shape[0]
        
        # 기본 처리
        base_features = self.base_regret(x)  # (batch, 768)
        
        # 3뷰 시나리오
        scenario_outputs = []
        for network in self.scenario_networks:
            scenario_out = network(base_features)
            scenario_outputs.append(scenario_out)
        scenarios = torch.cat(scenario_outputs, dim=-1)  # (batch, 3)
        
        # 반사실적 추론
        cf_in = base_features.unsqueeze(1)  # (batch, 1, 768)
        cf_out, _ = self.counterfactual(cf_in)
        cf_features = cf_out.squeeze(1)  # (batch, 576)
        
        # 어텐션 적용
        cf_attn, _ = self.counterfactual_attention(
            cf_features.unsqueeze(1),
            cf_features.unsqueeze(1),
            cf_features.unsqueeze(1)
        )
        cf_final = cf_attn.squeeze(1)  # (batch, 576)
        
        # 불확실성 정량화
        uncertainty_outputs = []
        for network in self.uncertainty:
            unc_out = network(base_features)
            uncertainty_outputs.append(unc_out)
        uncertainty = torch.cat(uncertainty_outputs, dim=-1)  # (batch, 5)
        
        # 시간 전파
        temporal_in = base_features.unsqueeze(1)
        temporal_out, _ = self.temporal_propagation(temporal_in)
        temporal_features = temporal_out.squeeze(1)  # (batch, 384)
        
        # 최종 결합
        combined = torch.cat([
            base_features,
            scenarios,
            cf_final,
            uncertainty,
            temporal_features
        ], dim=-1)
        
        output = self.output_layer(combined)
        
        return {
            'regret_score': output,
            'scenarios': scenarios,
            'counterfactual': cf_final,
            'uncertainty': uncertainty.std(dim=-1, keepdim=True)
        }
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """후회 손실 (Huber Loss - 이상치에 강건)"""
        return F.smooth_l1_loss(predictions, targets)
    
    def _log_params(self):
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"후회 헤드 파라미터: {total:,} ({total/1e6:.2f}M)")


class SURDHead(MultiTaskHead):
    """
    SURD 분석 헤드 (22M 파라미터 - 16M의 1.364배)
    - 정보이론 기반 인과분석
    - PID 분해 (Synergy, Unique, Redundant, Deterministic)
    """
    
    def __init__(self, input_dim: int = 896):
        super().__init__(input_dim, output_dim=4)  # S, U, R, D
        
        # 기본 인과 처리 (4M)
        self.base_causal = nn.Sequential(
            nn.Linear(input_dim, 1152),
            nn.LayerNorm(1152),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1152, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Linear(768, 576)
        )
        
        # PID 분해 네트워크 (5.5M)
        self.pid_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(576, 576),
                nn.LayerNorm(576),
                nn.GELU(),
                nn.Linear(576, 384),
                nn.GELU(),
                nn.Linear(384, 192),
                nn.Linear(192, 96),
                nn.Linear(96, 1)
            ) for _ in range(4)  # S, U, R, D
        ])
        
        # 그래프 신경망 (4M)
        self.graph_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(576, 576),
                nn.LayerNorm(576),
                nn.GELU(),
                nn.Linear(576, 576)
            ) for _ in range(4)
        ])
        
        # 상호정보 추정 (4M)
        self.mutual_info = nn.ModuleList([
            nn.Sequential(
                nn.Linear(576, 576),
                nn.LayerNorm(576),
                nn.GELU(),
                nn.Linear(576, 384),
                nn.GELU(),
                nn.Linear(384, 192),
                nn.Linear(192, 1)
            ) for _ in range(3)  # 3개 추정기
        ])
        
        # 어텐션 메커니즘 (3M)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=576,
            num_heads=12,
            dropout=0.1,
            batch_first=True
        )
        
        # 최종 출력 (1.5M)
        self.output_layer = nn.Sequential(
            nn.Linear(576 + 4 + 576 + 3, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Linear(768, 384),
            nn.Linear(384, 4)
        )
        
        self._log_params()
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """순전파"""
        batch_size = x.shape[0]
        
        # 기본 처리
        base_features = self.base_causal(x)  # (batch, 576)
        
        # PID 분해
        pid_outputs = []
        for network in self.pid_networks:
            pid_out = network(base_features)
            pid_outputs.append(pid_out)
        pid_values = torch.cat(pid_outputs, dim=-1)  # (batch, 4)
        
        # 그래프 처리
        graph_features = base_features
        for layer in self.graph_layers:
            graph_features = graph_features + layer(graph_features)  # 잔차 연결
        
        # 상호정보 추정
        mi_outputs = []
        for estimator in self.mutual_info:
            mi_out = estimator(base_features)
            mi_outputs.append(mi_out)
        mi_values = torch.cat(mi_outputs, dim=-1)  # (batch, 3)
        
        # 어텐션 적용
        attn_in = base_features.unsqueeze(1)
        attn_out, _ = self.self_attention(attn_in, attn_in, attn_in)
        attn_features = attn_out.squeeze(1)  # (batch, 768)
        
        # 최종 결합
        combined = torch.cat([
            attn_features,
            pid_values,
            graph_features,
            mi_values
        ], dim=-1)
        
        output = self.output_layer(combined)
        
        return {
            'surd_values': output,  # S, U, R, D
            'pid_raw': pid_values,
            'mutual_info': mi_values,
            'graph_features': graph_features
        }
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """SURD 손실 (MSE + 정규화)"""
        mse_loss = F.mse_loss(predictions, targets)
        
        # SURD 값의 합이 1에 가까워지도록 정규화
        sum_constraint = (predictions.sum(dim=-1) - 1.0) ** 2
        
        return mse_loss + 0.1 * sum_constraint.mean()
    
    def _log_params(self):
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"SURD 헤드 파라미터: {total:,} ({total/1e6:.2f}M)")


def create_all_heads() -> Dict[str, MultiTaskHead]:
    """모든 헤드 생성"""
    heads = {
        'emotion': EmotionHead(),
        'bentham': BenthamHead(),
        'regret': RegretHead(),
        'surd': SURDHead()
    }
    
    total_params = sum(
        sum(p.numel() for p in head.parameters())
        for head in heads.values()
    )
    
    logger.info(f"헤드 총 파라미터: {total_params:,} ({total_params/1e6:.2f}M)")
    logger.info(f"목표: 80M, 실제: {total_params/1e6:.2f}M")
    
    return heads