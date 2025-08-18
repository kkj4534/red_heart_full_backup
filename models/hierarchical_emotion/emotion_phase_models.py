"""
계층적 감정 시스템 모델 - Phase별 감정 학습 모델
Hierarchical Emotion System Models for Phase-based Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle
from datetime import datetime

@dataclass
class EmotionPhaseConfig:
    """감정 페이즈 설정"""
    phase_id: int
    input_dim: int = 768  # sentence transformer embedding 차원
    hidden_dims: List[int] = None
    output_dim: int = 6   # 6차원 감정 벡터
    dropout_rate: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = 32
    
    def __post_init__(self):
        if self.hidden_dims is None:
            # Phase별로 복잡도 증가
            base_dims = [512, 256, 128]
            self.hidden_dims = [dim + (self.phase_id * 64) for dim in base_dims]


class EmotionTransformerBlock(nn.Module):
    """감정 변환을 위한 트랜스포머 블록"""
    
    def __init__(self, d_model: int, nhead: int = 8, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class Phase0EmotionModel(nn.Module):
    """Phase 0: 타자 감정 → 자신 투영 모델"""
    
    def __init__(self, config: EmotionPhaseConfig):
        super().__init__()
        self.config = config
        
        # 입력 임베딩 처리
        self.input_projection = nn.Linear(config.input_dim, config.hidden_dims[0])
        
        # 타자 감정 투영 (6차원 → hidden_dims[0] 차원)
        self.other_emotion_projection = nn.Linear(6, config.hidden_dims[0])
        
        # 입력 차원 동적 조정
        self.input_adaptation = None
        
        # 타자 감정 인코더
        self.other_emotion_encoder = nn.Sequential(
            nn.Linear(config.hidden_dims[0], config.hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dims[1], config.hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )
        
        # 자아 투영 메커니즘
        self.self_projection_layer = nn.Sequential(
            nn.Linear(config.hidden_dims[2], config.hidden_dims[2]),
            nn.Tanh(),  # 감정의 부호를 고려한 활성화
            nn.Linear(config.hidden_dims[2], config.output_dim)
        )
        
        # 캘리브레이션 레이어
        self.calibration_weights = nn.Parameter(torch.ones(config.output_dim))
        self.calibration_bias = nn.Parameter(torch.zeros(config.output_dim))
        
    def forward(self, text_embedding: torch.Tensor, other_emotion: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 입력 차원 동적 조정
        if text_embedding.shape[-1] != self.config.input_dim:
            if self.input_adaptation is None:
                self.input_adaptation = nn.Linear(text_embedding.shape[-1], self.config.input_dim).to(text_embedding.device)
            text_embedding = self.input_adaptation(text_embedding)
        
        # 텍스트 임베딩 투영
        projected_input = self.input_projection(text_embedding)
        
        # 타자 감정을 같은 차원으로 투영
        projected_other_emotion = self.other_emotion_projection(other_emotion)
        
        # 타자 감정 정보와 결합
        combined_input = projected_input + projected_other_emotion
        
        # 타자 감정 인코딩
        encoded_other = self.other_emotion_encoder(combined_input)
        
        # 자아 투영
        self_emotion = self.self_projection_layer(encoded_other)
        
        # 캘리브레이션 적용
        calibrated_emotion = self_emotion * self.calibration_weights + self.calibration_bias
        
        return {
            'emotion_vector': torch.tanh(calibrated_emotion),  # [-1, 1] 범위로 정규화
            'other_encoded': encoded_other,
            'projection_strength': torch.sigmoid(self_emotion).mean(dim=-1)
        }


class Phase1EmpathyModel(nn.Module):
    """Phase 1: 후회 기반 공감 학습 모델"""
    
    def __init__(self, config: EmotionPhaseConfig):
        super().__init__()
        self.config = config
        
        # 후회 임베딩 처리
        self.regret_encoder = nn.Sequential(
            nn.Linear(config.input_dim + 6, config.hidden_dims[0]),  # +6 for regret dimensions
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )
        
        # 공감 어텐션 메커니즘
        self.empathy_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dims[0], 
            num_heads=8, 
            dropout=config.dropout_rate,
            batch_first=True
        )
        
        # 트랜스포머 블록
        self.transformer_block = EmotionTransformerBlock(
            d_model=config.hidden_dims[0],
            nhead=8,
            dim_feedforward=config.hidden_dims[1],
            dropout=config.dropout_rate
        )
        
        # 공감 예측 헤드
        self.empathy_predictor = nn.Sequential(
            nn.Linear(config.hidden_dims[0], config.hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dims[1], config.hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(config.hidden_dims[2], config.output_dim)
        )
        
        # 후회 강도 예측
        self.regret_intensity_predictor = nn.Linear(config.hidden_dims[0], 1)
        
    def forward(self, text_embedding: torch.Tensor, regret_vector: torch.Tensor, 
                phase0_emotion: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        # 후회 정보와 텍스트 임베딩 결합
        regret_input = torch.cat([text_embedding, regret_vector], dim=-1)
        encoded_regret = self.regret_encoder(regret_input)
        
        # Phase 0 감정을 query로 사용하는 어텐션
        if encoded_regret.dim() == 2:
            encoded_regret = encoded_regret.unsqueeze(1)  # [batch, 1, dim]
        if phase0_emotion.dim() == 2:
            # Phase0 감정을 같은 차원으로 확장
            if not hasattr(self, 'phase0_projection'):
                self.phase0_projection = nn.Linear(phase0_emotion.shape[-1], self.config.hidden_dims[0]).to(phase0_emotion.device)
            phase0_emotion = self.phase0_projection(phase0_emotion)
            phase0_emotion = phase0_emotion.unsqueeze(1)
            
        attended_empathy, attention_weights = self.empathy_attention(
            phase0_emotion, encoded_regret, encoded_regret
        )
        
        # 트랜스포머 처리
        transformed_empathy = self.transformer_block(attended_empathy)
        
        # 공감 감정 예측
        empathy_emotion = self.empathy_predictor(transformed_empathy.squeeze(1))
        
        # 후회 강도 예측
        regret_intensity = torch.sigmoid(self.regret_intensity_predictor(transformed_empathy.squeeze(1)))
        
        return {
            'empathy_emotion': torch.tanh(empathy_emotion),
            'regret_intensity': regret_intensity,
            'attention_weights': attention_weights,
            'encoded_regret': encoded_regret.squeeze(1)
        }


class Phase2CommunityModel(nn.Module):
    """Phase 2: 공동체 감정 확장 모델"""
    
    def __init__(self, config: EmotionPhaseConfig):
        super().__init__()
        self.config = config
        
        # 다중 관점 인코더
        self.multi_perspective_encoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.input_dim, config.hidden_dims[0]),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate),
                nn.Linear(config.hidden_dims[0], config.hidden_dims[1])
            ) for _ in range(3)  # 개인, 가족, 사회 관점
        ])
        
        # 관점 통합 어텐션
        self.perspective_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dims[1],
            num_heads=12,  # 더 복잡한 어텐션
            dropout=config.dropout_rate,
            batch_first=True
        )
        
        # 공동체 감정 확장 네트워크
        self.community_expansion = nn.Sequential(
            nn.Linear(config.hidden_dims[1] * 3, config.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dims[0], config.hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(config.hidden_dims[1], config.output_dim * 3)  # 개인/가족/사회 각각
        )
        
        # 계층적 감정 통합 (동적 차원 계산)
        self.hierarchical_integrator = None  # forward에서 동적 생성
        
    def forward(self, text_embedding: torch.Tensor, 
                phase0_emotion: torch.Tensor, phase1_empathy: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        # 다중 관점 인코딩
        perspectives = []
        for encoder in self.multi_perspective_encoder:
            perspective = encoder(text_embedding)
            perspectives.append(perspective)
        
        # 관점들을 하나의 텐서로 스택
        stacked_perspectives = torch.stack(perspectives, dim=1)  # [batch, 3, dim]
        
        # 관점 간 어텐션
        attended_perspectives, perspective_weights = self.perspective_attention(
            stacked_perspectives, stacked_perspectives, stacked_perspectives
        )
        
        # 공동체 감정 확장
        flattened_perspectives = attended_perspectives.reshape(attended_perspectives.size(0), -1)
        expanded_emotions = self.community_expansion(flattened_perspectives)
        
        # 개인/가족/사회 감정으로 분할
        personal_emotion = expanded_emotions[:, :self.config.output_dim]
        family_emotion = expanded_emotions[:, self.config.output_dim:self.config.output_dim*2]
        social_emotion = expanded_emotions[:, self.config.output_dim*2:]
        
        # 이전 Phase 결과와 통합
        combined_emotions = torch.cat([
            phase0_emotion, phase1_empathy, personal_emotion, 
            family_emotion, social_emotion
        ], dim=-1)
        
        # 계층적 통합 (동적 생성)
        if self.hierarchical_integrator is None:
            input_dim = combined_emotions.shape[-1]
            self.hierarchical_integrator = nn.Sequential(
                nn.Linear(input_dim, self.config.hidden_dims[1]),
                nn.ReLU(),
                nn.Linear(self.config.hidden_dims[1], self.config.output_dim)
            ).to(combined_emotions.device)
        
        integrated_emotion = self.hierarchical_integrator(combined_emotions)
        
        return {
            'integrated_emotion': torch.tanh(integrated_emotion),
            'personal_emotion': torch.tanh(personal_emotion),
            'family_emotion': torch.tanh(family_emotion),
            'social_emotion': torch.tanh(social_emotion),
            'perspective_weights': perspective_weights
        }


class HierarchicalEmotionModel(nn.Module):
    """전체 계층적 감정 시스템 통합 모델"""
    
    def __init__(self, input_dim: int = 768):
        super().__init__()
        
        # 각 Phase별 설정
        self.phase0_config = EmotionPhaseConfig(phase_id=0, input_dim=input_dim)
        self.phase1_config = EmotionPhaseConfig(phase_id=1, input_dim=input_dim)
        self.phase2_config = EmotionPhaseConfig(phase_id=2, input_dim=input_dim)
        
        # 각 Phase 모델
        self.phase0_model = Phase0EmotionModel(self.phase0_config)
        self.phase1_model = Phase1EmpathyModel(self.phase1_config)
        self.phase2_model = Phase2CommunityModel(self.phase2_config)
        
        # 최종 통합 레이어
        self.final_integrator = nn.Sequential(
            nn.Linear(6 * 5, 256),  # 모든 감정 벡터 통합
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # 최종 6차원 감정 벡터
        )
        
        # 감정 차원별 가중치
        self.emotion_dimension_weights = nn.Parameter(torch.ones(6))
        
    def forward(self, text_embedding: torch.Tensor, other_emotion: torch.Tensor = None,
                regret_vector: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        
        # 기본값 설정 (단일 입력으로 호출될 때)
        batch_size = text_embedding.shape[0]
        
        if other_emotion is None:
            other_emotion = torch.zeros(batch_size, 6, device=text_embedding.device)
        if regret_vector is None:
            regret_vector = torch.zeros(batch_size, 6, device=text_embedding.device)  # 6차원으로 수정
        
        # Phase 0: 타자 감정 투영
        phase0_output = self.phase0_model(text_embedding, other_emotion)
        
        # Phase 1: 공감 학습 (regret_vector 차원 확인)
        if regret_vector.shape[-1] != 6:
            # regret_vector를 6차원으로 패딩 또는 자르기
            if regret_vector.shape[-1] > 6:
                regret_vector = regret_vector[:, :6]
            else:
                padding = torch.zeros(batch_size, 6 - regret_vector.shape[-1], device=text_embedding.device)
                regret_vector = torch.cat([regret_vector, padding], dim=-1)
        
        phase1_output = self.phase1_model(
            text_embedding, regret_vector, phase0_output['emotion_vector']
        )
        
        # Phase 2: 공동체 확장
        phase2_output = self.phase2_model(
            text_embedding, phase0_output['emotion_vector'], phase1_output['empathy_emotion']
        )
        
        # 모든 감정 벡터 수집
        all_emotions = torch.cat([
            phase0_output['emotion_vector'],
            phase1_output['empathy_emotion'],
            phase2_output['personal_emotion'],
            phase2_output['family_emotion'],
            phase2_output['social_emotion']
        ], dim=-1)
        
        # 최종 통합
        final_emotion = self.final_integrator(all_emotions)
        
        # 차원별 가중치 적용
        weighted_emotion = final_emotion * self.emotion_dimension_weights
        
        return {
            'final_emotion': torch.tanh(weighted_emotion),
            'phase0_emotion': phase0_output['emotion_vector'],
            'phase1_empathy': phase1_output['empathy_emotion'],
            'phase2_integrated': phase2_output['integrated_emotion'],
            'regret_intensity': phase1_output['regret_intensity'],
            'perspective_weights': phase2_output['perspective_weights'],
            'emotion_evolution': {
                'phase0': phase0_output,
                'phase1': phase1_output,
                'phase2': phase2_output
            }
        }


class EmotionModelManager:
    """감정 모델 관리자"""
    
    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.training_history = []
        
    def save_model(self, model: HierarchicalEmotionModel, epoch: int, 
                   loss: float, metrics: Dict[str, float]):
        """모델 저장"""
        model_path = self.models_dir / f"hierarchical_emotion_epoch_{epoch}.pth"
        
        save_data = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'loss': loss,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'model_config': {
                'input_dim': 768,
                'model_class': 'HierarchicalEmotionModel'
            }
        }
        
        torch.save(save_data, model_path)
        
        # 훈련 이력 저장
        history_path = self.models_dir / "training_history.json"
        self.training_history.append({
            'epoch': epoch,
            'loss': loss,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })
        
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, indent=2, ensure_ascii=False)
            
    def load_model(self, model_path: Optional[Path] = None) -> HierarchicalEmotionModel:
        """모델 로드"""
        if model_path is None:
            # 가장 최신 모델 찾기
            model_files = list(self.models_dir.glob("hierarchical_emotion_epoch_*.pth"))
            if not model_files:
                raise FileNotFoundError("저장된 모델이 없습니다.")
            
            model_path = max(model_files, key=lambda p: p.stat().st_mtime)
        
        checkpoint = torch.load(model_path, map_location='cpu')
        
        model = HierarchicalEmotionModel(
            input_dim=checkpoint['model_config']['input_dim']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model = model
        return model
    
    def get_training_stats(self) -> Dict[str, Any]:
        """훈련 통계 반환"""
        if not self.training_history:
            return {}
        
        losses = [h['loss'] for h in self.training_history]
        return {
            'total_epochs': len(self.training_history),
            'best_loss': min(losses),
            'latest_loss': losses[-1],
            'improvement': losses[0] - losses[-1] if len(losses) > 1 else 0,
            'training_history': self.training_history[-10:]  # 최근 10개 에포크
        }


# 감정 차원 정의
EMOTION_DIMENSIONS = {
    'valence': 0,      # 감정가 (긍정/부정)
    'arousal': 1,      # 각성도 (활성화/비활성화)
    'dominance': 2,    # 지배감 (통제/무력감)
    'certainty': 3,    # 확실성 (확신/불확실)
    'surprise': 4,     # 놀라움 (예상/예상외)
    'anticipation': 5  # 기대감 (기대/무관심)
}

def create_emotion_model(input_dim: int = 768, device: str = 'cpu') -> HierarchicalEmotionModel:
    """감정 모델 생성 헬퍼 함수"""
    model = HierarchicalEmotionModel(input_dim=input_dim)
    model.to(device)
    return model

def emotion_vector_to_dict(emotion_vector: torch.Tensor) -> Dict[str, float]:
    """감정 벡터를 딕셔너리로 변환"""
    if emotion_vector.dim() > 1:
        emotion_vector = emotion_vector.squeeze()
    
    return {
        name: float(emotion_vector[idx])
        for name, idx in EMOTION_DIMENSIONS.items()
    }