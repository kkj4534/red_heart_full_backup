"""
Phase0/Phase2 신경망 모듈
학습 가능한 파라미터를 가진 신경망 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List


class Phase0ProjectionNet(nn.Module):
    """Phase0: 타자→자신 감정 투영 학습 네트워크
    
    타자의 감정을 자신의 감정 공간으로 투영하는 비선형 변환 학습
    """
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 512, output_dim: int = 7):
        super().__init__()
        
        # 투영 네트워크 (2M 파라미터)
        self.projection_layers = nn.Sequential(
            # 입력 처리
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            
            # 중간 변환 레이어
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            
            # 차원 축소
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            
            # 감정 투영
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # 캘리브레이션 파라미터 (학습 가능)
        self.calibration_bias = nn.Parameter(torch.zeros(output_dim))
        self.calibration_scale = nn.Parameter(torch.ones(output_dim))
        
        # 비선형 투영 함수의 학습 가능 파라미터
        self.projection_weight = nn.Parameter(torch.randn(output_dim, output_dim) * 0.1)
        
    def forward(self, other_emotion: torch.Tensor, context_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        타자 감정을 자신 감정으로 투영
        
        Args:
            other_emotion: 타자의 감정 벡터 [batch_size, 7] 또는 임베딩 [batch_size, 768]
            context_embedding: 문맥 임베딩 (선택) [batch_size, 768]
            
        Returns:
            투영된 자신의 감정 벡터 [batch_size, 7]
        """
        # 입력 차원 확인
        if other_emotion.shape[-1] == 7:
            # 감정 벡터를 높은 차원으로 확장
            expanded = F.pad(other_emotion, (0, 761), mode='constant', value=0)
            if context_embedding is not None:
                expanded = expanded + context_embedding * 0.1
            x = expanded
        else:
            # 이미 높은 차원 입력
            x = other_emotion
            if context_embedding is not None:
                x = x + context_embedding * 0.1
        
        # 투영 네트워크 통과
        projected = self.projection_layers(x)
        
        # 캘리브레이션 적용
        calibrated = projected * self.calibration_scale + self.calibration_bias
        
        # 비선형 투영 적용
        final = torch.matmul(torch.sigmoid(calibrated), self.projection_weight)
        
        # 정규화 (감정 합 = 1)
        final = F.softmax(final, dim=-1)
        
        return final


class Phase2CommunityNet(nn.Module):
    """Phase2: 개인→공동체 감정 패턴 학습 네트워크
    
    개인들의 감정을 집계하여 공동체 수준의 감정 패턴을 학습
    """
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 512, output_dim: int = 10, max_individuals: int = 100):
        super().__init__()
        
        # 개인 감정 인코더 (1M)
        self.individual_encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # 어텐션 메커니즘 (0.5M)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 공동체 패턴 디코더 (1M)
        self.community_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # 문화적 맥락 인코더
        self.cultural_encoder = nn.ModuleDict({
            'korean': self._create_cultural_encoder(hidden_dim),
            'western': self._create_cultural_encoder(hidden_dim),
            'eastern': self._create_cultural_encoder(hidden_dim),
            'global': self._create_cultural_encoder(hidden_dim)
        })
        
        # 시간적 동태 모델링
        self.temporal_dynamics = nn.GRU(
            input_size=output_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True
        )
        
    def _create_cultural_encoder(self, hidden_dim: int) -> nn.Module:
        """문화별 인코더 생성"""
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU()
        )
    
    def forward(self, 
                individual_emotions: torch.Tensor,
                cultural_context: Optional[str] = None,
                temporal_history: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        개인 감정들을 공동체 감정 패턴으로 변환
        
        Args:
            individual_emotions: 개인들의 감정 [batch_size, num_individuals, 768]
            cultural_context: 문화적 맥락 ('korean', 'western', 'eastern', 'global')
            temporal_history: 시간적 이력 [batch_size, time_steps, output_dim]
            
        Returns:
            공동체 감정 패턴 [batch_size, output_dim]
        """
        batch_size = individual_emotions.shape[0]
        
        # 개인 감정 인코딩
        encoded, (hidden, _) = self.individual_encoder(individual_emotions)
        
        # 셀프 어텐션으로 중요한 개인 감정 추출
        attended, _ = self.attention(encoded, encoded, encoded)
        
        # 평균 풀링으로 집계
        aggregated = torch.mean(attended, dim=1)  # [batch_size, hidden_dim]
        
        # 문화적 맥락 적용
        if cultural_context and cultural_context in self.cultural_encoder:
            cultural_features = self.cultural_encoder[cultural_context](aggregated)
            aggregated = aggregated + F.pad(cultural_features, (0, aggregated.shape[-1] - cultural_features.shape[-1]))
        
        # 공동체 패턴 디코딩
        community_pattern = self.community_decoder(aggregated)
        
        # 시간적 동태 적용
        if temporal_history is not None:
            temporal_out, _ = self.temporal_dynamics(temporal_history)
            # 마지막 시간 단계의 출력 사용
            temporal_features = temporal_out[:, -1, :]
            # 차원 맞추기
            temporal_features = F.pad(temporal_features, (0, community_pattern.shape[-1] - temporal_features.shape[-1]))
            community_pattern = community_pattern + temporal_features * 0.3
        
        return community_pattern


class HierarchicalEmotionIntegrator(nn.Module):
    """Phase 0-1-2 통합 모듈
    
    3단계 계층적 감정 시스템을 통합하여 학습
    """
    
    def __init__(self):
        super().__init__()
        
        # Phase별 네트워크
        self.phase0_net = Phase0ProjectionNet()
        self.phase1_net = None  # Phase1은 이미 구현됨 (EmpathyNet)
        self.phase2_net = Phase2CommunityNet()
        
        # 통합 레이어
        self.integration_layer = nn.Sequential(
            nn.Linear(7 + 10, 32),  # Phase0(7) + Phase2(10)
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 7)  # 최종 7차원 감정
        )
        
    def forward(self, 
                other_emotion: torch.Tensor,
                individual_emotions: torch.Tensor,
                context: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """
        통합 감정 처리
        
        Args:
            other_emotion: 타자 감정
            individual_emotions: 개인들 감정
            context: 추가 맥락 정보
            
        Returns:
            각 Phase별 출력과 통합 결과
        """
        # Phase 0: 타자→자신 투영
        phase0_out = self.phase0_net(other_emotion)
        
        # Phase 2: 개인→공동체
        cultural_context = context.get('culture', 'global') if context else 'global'
        phase2_out = self.phase2_net(individual_emotions, cultural_context)
        
        # 통합
        combined = torch.cat([phase0_out, phase2_out], dim=-1)
        integrated = self.integration_layer(combined)
        
        return {
            'phase0_projection': phase0_out,
            'phase2_community': phase2_out,
            'integrated_emotion': integrated
        }