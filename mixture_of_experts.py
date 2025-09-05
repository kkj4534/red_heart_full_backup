"""
Mixture of Experts (MoE) Implementation
감정/윤리/후회 분석을 위한 전문가 시스템
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger('RedHeart.MixtureOfExperts')

@dataclass
class ExpertOutput:
    """전문가 출력 결과"""
    expert_id: str
    output: torch.Tensor
    confidence: float
    weight: float
    metadata: Dict[str, Any]

@dataclass
class MoEResult:
    """MoE 전체 결과"""
    final_output: torch.Tensor
    expert_outputs: List[ExpertOutput]
    gating_weights: torch.Tensor
    total_experts_used: int
    diversity_score: float
    metadata: Dict[str, Any]

class Expert(nn.Module, ABC):
    """전문가 베이스 클래스"""
    
    def __init__(self, expert_id: str, input_dim: int, output_dim: int):
        super(Expert, self).__init__()
        self.expert_id = expert_id
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """전문가 포워드 패스"""
        pass
    
    @abstractmethod
    def get_confidence(self, x: torch.Tensor, output: torch.Tensor) -> float:
        """전문가 신뢰도 계산"""
        pass

class LinearExpert(Expert):
    """선형 전문가"""
    
    def __init__(self, expert_id: str, input_dim: int, output_dim: int, hidden_dim: int = None):
        super(LinearExpert, self).__init__(expert_id, input_dim, output_dim)
        
        if hidden_dim is None:
            hidden_dim = max(input_dim // 2, output_dim * 2)
            
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # 신뢰도 계산용
        self.confidence_head = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 중간 특성 저장
        for i, layer in enumerate(self.network):
            x = layer(x)
            if i == len(self.network) - 3:  # hidden_dim // 2 레이어
                self.last_hidden = x.clone()
        return x
    
    def get_confidence(self, x: torch.Tensor, output: torch.Tensor) -> float:
        if hasattr(self, 'last_hidden'):
            confidence_logit = self.confidence_head(self.last_hidden)
            confidence = torch.sigmoid(confidence_logit).mean().item()
        else:
            # fallback: 출력의 분산 기반 신뢰도
            variance = torch.var(output, dim=-1).mean().item()
            confidence = 1.0 / (1.0 + variance)
        
        return max(0.1, min(0.9, confidence))

class SpecializedExpert(Expert):
    """특화된 전문가 (도메인별)"""
    
    def __init__(self, expert_id: str, input_dim: int, output_dim: int, 
                 specialization: str, expertise_weights: Optional[torch.Tensor] = None):
        super(SpecializedExpert, self).__init__(expert_id, input_dim, output_dim)
        
        self.specialization = specialization
        
        # 전문 분야에 따른 가중치
        if expertise_weights is not None:
            self.expertise_weights = nn.Parameter(expertise_weights)
        else:
            self.expertise_weights = nn.Parameter(torch.ones(input_dim))
        
        # 특화된 네트워크 구조
        hidden_dim = max(input_dim, output_dim * 2)
        
        # 특화 레이어
        self.specialization_layer = nn.Linear(input_dim, hidden_dim)
        
        # 메인 네트워크
        self.main_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # 신뢰도 계산
        self.confidence_network = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 전문 분야 가중치 적용
        weighted_input = x * self.expertise_weights
        
        # 특화 레이어
        specialized_features = F.relu(self.specialization_layer(weighted_input))
        
        # 메인 네트워크
        for i, layer in enumerate(self.main_network):
            specialized_features = layer(specialized_features)
            if i == len(self.main_network) - 3:  # LayerNorm 레이어 이후
                self.confidence_features = specialized_features.clone()
        
        return specialized_features
    
    def get_confidence(self, x: torch.Tensor, output: torch.Tensor) -> float:
        if hasattr(self, 'confidence_features'):
            confidence = self.confidence_network(self.confidence_features).mean().item()
        else:
            # fallback
            confidence = 0.5
        
        return max(0.1, min(0.9, confidence))

class GatingNetwork(nn.Module):
    """게이팅 네트워크 - 전문가 선택"""
    
    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int = None):
        super(GatingNetwork, self).__init__()
        
        if hidden_dim is None:
            hidden_dim = max(input_dim // 2, num_experts * 4)
        
        self.input_dim = input_dim
        self.num_experts = num_experts
        
        # 게이팅 네트워크
        self.gating_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_experts)
        )
        
        # 다양성 촉진을 위한 레이어 - input_dim에서 직접 연결
        self.diversity_layer = nn.Linear(input_dim, num_experts)
        
    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        게이팅 가중치 계산
        
        Args:
            x: 입력 텐서
            temperature: 소프트맥스 온도 (낮을수록 더 선택적)
            
        Returns:
            전문가별 가중치
        """
        # 게이팅 로짓 계산
        gating_logits = self.gating_network(x)
        
        # 다양성 보너스 - 이제 x를 직접 사용 가능
        diversity_bonus = self.diversity_layer(x) * 0.1
        
        # 최종 로짓
        final_logits = gating_logits + diversity_bonus
        
        # 온도 스케일링 적용
        scaled_logits = final_logits / temperature
        
        # 소프트맥스로 가중치 계산
        weights = F.softmax(scaled_logits, dim=-1)
        
        return weights

class MixtureOfExperts(nn.Module):
    """Mixture of Experts 메인 클래스"""
    
    def __init__(self, experts: List[Expert], gating_network: GatingNetwork, 
                 top_k: int = None, load_balancing: bool = True):
        super(MixtureOfExperts, self).__init__()
        
        self.experts = nn.ModuleList(experts)
        self.gating_network = gating_network
        self.num_experts = len(experts)
        
        # Top-K 전문가만 사용 (메모리 효율성)
        self.top_k = min(top_k or self.num_experts, self.num_experts)
        
        # 로드 밸런싱
        self.load_balancing = load_balancing
        self.expert_usage_count = torch.zeros(self.num_experts)
        
        # 성능 추적
        self.performance_tracker = {}
        
    def forward(self, x: torch.Tensor, temperature: float = 1.0, 
                return_expert_outputs: bool = False) -> Union[torch.Tensor, MoEResult]:
        """
        MoE 포워드 패스
        
        Args:
            x: 입력 텐서 [batch_size, input_dim]
            temperature: 게이팅 온도
            return_expert_outputs: 전문가별 상세 결과 반환 여부
            
        Returns:
            최종 출력 또는 MoEResult
        """
        batch_size = x.size(0)
        
        # 게이팅 가중치 계산
        gating_weights = self.gating_network(x, temperature)
        
        # Top-K 전문가 선택
        if self.top_k < self.num_experts:
            top_k_weights, top_k_indices = torch.topk(gating_weights, self.top_k, dim=-1)
            top_k_weights = F.softmax(top_k_weights, dim=-1)
        else:
            top_k_weights = gating_weights
            top_k_indices = torch.arange(self.num_experts).expand(batch_size, -1).to(x.device)
        
        # 전문가별 출력 계산
        expert_outputs = []
        final_output = torch.zeros(batch_size, self.experts[0].output_dim).to(x.device)
        
        for i in range(self.top_k):
            for batch_idx in range(batch_size):
                expert_idx = top_k_indices[batch_idx, i].item()
                weight = top_k_weights[batch_idx, i].item()
                
                if weight > 0.01:  # 최소 임계값
                    expert = self.experts[expert_idx]
                    expert_output = expert(x[batch_idx:batch_idx+1])
                    confidence = expert.get_confidence(x[batch_idx:batch_idx+1], expert_output)
                    
                    # 가중 출력 누적
                    weighted_output = expert_output * weight * confidence
                    final_output[batch_idx] += weighted_output.squeeze(0)
                    
                    # 전문가 사용 통계 업데이트
                    self.expert_usage_count[expert_idx] += 1
                    
                    if return_expert_outputs:
                        expert_outputs.append(ExpertOutput(
                            expert_id=expert.expert_id,
                            output=expert_output.squeeze(0),
                            confidence=confidence,
                            weight=weight,
                            metadata={'batch_idx': batch_idx, 'expert_idx': expert_idx}
                        ))
        
        if return_expert_outputs:
            # 다양성 점수 계산
            diversity_score = self._calculate_diversity_score(gating_weights)
            
            return MoEResult(
                final_output=final_output,
                expert_outputs=expert_outputs,
                gating_weights=gating_weights,
                total_experts_used=len(set(eo.metadata['expert_idx'] for eo in expert_outputs)),
                diversity_score=diversity_score,
                metadata={
                    'top_k': self.top_k,
                    'temperature': temperature,
                    'expert_usage_count': self.expert_usage_count.tolist()
                }
            )
        else:
            return final_output
    
    def _calculate_diversity_score(self, gating_weights: torch.Tensor) -> float:
        """전문가 사용의 다양성 점수 계산"""
        # 엔트로피 기반 다양성
        mean_weights = gating_weights.mean(dim=0)
        entropy = -torch.sum(mean_weights * torch.log(mean_weights + 1e-8))
        max_entropy = torch.log(torch.tensor(float(self.num_experts)))
        
        diversity_score = (entropy / max_entropy).item()
        return diversity_score
    
    def get_load_balancing_loss(self) -> torch.Tensor:
        """로드 밸런싱 손실 계산"""
        if not self.load_balancing:
            return torch.tensor(0.0)
        
        # 전문가 사용 분포의 분산을 최소화
        usage_variance = torch.var(self.expert_usage_count.float())
        return usage_variance * 0.01  # 작은 가중치로 추가
    
    def reset_usage_stats(self):
        """사용 통계 초기화"""
        self.expert_usage_count.zero_()
    
    def get_expert_statistics(self) -> Dict[str, Any]:
        """전문가 사용 통계"""
        total_usage = self.expert_usage_count.sum().item()
        if total_usage == 0:
            return {}
        
        usage_percentages = (self.expert_usage_count / total_usage * 100).tolist()
        
        return {
            'expert_usage_count': self.expert_usage_count.tolist(),
            'expert_usage_percentage': usage_percentages,
            'total_invocations': total_usage,
            'most_used_expert': torch.argmax(self.expert_usage_count).item(),
            'least_used_expert': torch.argmin(self.expert_usage_count).item()
        }

def create_emotion_moe(input_dim: int, output_dim: int, num_experts: int = 4) -> MixtureOfExperts:
    """감정 분석용 MoE 생성"""
    experts = []
    
    # 감정별 특화 전문가
    emotion_specializations = ['joy_trust', 'sadness_fear', 'anger_disgust', 'surprise_anticipation']
    
    for i in range(num_experts):
        if i < len(emotion_specializations):
            # 특화된 전문가
            specialization = emotion_specializations[i]
            expert = SpecializedExpert(
                expert_id=f"emotion_expert_{specialization}",
                input_dim=input_dim,
                output_dim=output_dim,
                specialization=specialization
            )
        else:
            # 일반 전문가
            expert = LinearExpert(
                expert_id=f"emotion_expert_general_{i}",
                input_dim=input_dim,
                output_dim=output_dim
            )
        experts.append(expert)
    
    # 게이팅 네트워크
    gating_network = GatingNetwork(input_dim, num_experts)
    
    return MixtureOfExperts(experts, gating_network, top_k=3)

def create_ethics_moe(input_dim: int, output_dim: int, num_experts: int = 4) -> MixtureOfExperts:
    """윤리 분석용 MoE 생성"""
    experts = []
    
    # 윤리 이론별 특화 전문가
    ethics_specializations = ['utilitarian', 'deontological', 'virtue_ethics', 'care_ethics']
    
    for i in range(num_experts):
        if i < len(ethics_specializations):
            specialization = ethics_specializations[i]
            expert = SpecializedExpert(
                expert_id=f"ethics_expert_{specialization}",
                input_dim=input_dim,
                output_dim=output_dim,
                specialization=specialization
            )
        else:
            expert = LinearExpert(
                expert_id=f"ethics_expert_general_{i}",
                input_dim=input_dim,
                output_dim=output_dim
            )
        experts.append(expert)
    
    gating_network = GatingNetwork(input_dim, num_experts)
    
    return MixtureOfExperts(experts, gating_network, top_k=3)

def create_regret_moe(input_dim: int, output_dim: int, num_experts: int = 3) -> MixtureOfExperts:
    """후회 분석용 MoE 생성"""
    experts = []
    
    # 후회 유형별 특화 전문가
    regret_specializations = ['action_regret', 'inaction_regret', 'outcome_regret']
    
    for i in range(num_experts):
        if i < len(regret_specializations):
            specialization = regret_specializations[i]
            expert = SpecializedExpert(
                expert_id=f"regret_expert_{specialization}",
                input_dim=input_dim,
                output_dim=output_dim,
                specialization=specialization
            )
        else:
            expert = LinearExpert(
                expert_id=f"regret_expert_general_{i}",
                input_dim=input_dim,
                output_dim=output_dim
            )
        experts.append(expert)
    
    gating_network = GatingNetwork(input_dim, num_experts)
    
    return MixtureOfExperts(experts, gating_network, top_k=2)