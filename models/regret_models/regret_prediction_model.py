"""
후회 예측 및 학습 모델
Regret Prediction and Learning Models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle
from datetime import datetime
from enum import Enum

class RegretType(Enum):
    """후회 유형"""
    ACTION_REGRET = "action"           # 행동에 대한 후회
    INACTION_REGRET = "inaction"       # 무행동에 대한 후회
    TIMING_REGRET = "timing"           # 타이밍에 대한 후회
    CHOICE_REGRET = "choice"           # 선택에 대한 후회
    EMPATHY_REGRET = "empathy"         # 공감 부족에 대한 후회
    PREDICTION_REGRET = "prediction"   # 예측 실패에 대한 후회

@dataclass
class RegretContext:
    """후회 맥락 정보"""
    decision_quality: float
    outcome_deviation: float
    time_pressure: float
    information_completeness: float
    emotional_involvement: float
    social_consequences: float
    reversibility: float

class RegretIntensityPredictor(nn.Module):
    """후회 강도 예측 모델"""
    
    def __init__(self, input_dim: int = 768, context_dim: int = 7):
        super().__init__()
        
        # 텍스트 임베딩 처리
        self.text_encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 맥락 정보 처리
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # 결합 레이어
        self.fusion_layer = nn.Sequential(
            nn.Linear(256 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # 후회 강도 예측 헤드
        self.intensity_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 0-1 범위로 정규화
        )
        
        # 후회 유형 분류 헤드
        self.type_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, len(RegretType)),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, text_embedding: torch.Tensor, 
                context_features: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        
        # 기본값 설정 (단일 입력으로 호출될 때)
        if context_features is None:
            batch_size = text_embedding.shape[0]
            context_features = torch.zeros(batch_size, 7, device=text_embedding.device)
        
        # context_features 차원이 7이 아닌 경우 조정
        if context_features.shape[-1] != 7:
            # 차원이 다르면 선형 변환으로 맞춤
            if not hasattr(self, 'context_adapter'):
                self.context_adapter = nn.Linear(context_features.shape[-1], 7).to(context_features.device)
            context_features = self.context_adapter(context_features)
        
        # 텍스트 인코딩
        text_encoded = self.text_encoder(text_embedding)
        
        # 맥락 인코딩
        context_encoded = self.context_encoder(context_features)
        
        # 특징 결합
        fused_features = torch.cat([text_encoded, context_encoded], dim=-1)
        fused_features = self.fusion_layer(fused_features)
        
        # 후회 강도 예측
        intensity = self.intensity_head(fused_features)
        
        # 후회 유형 분류
        regret_type_probs = self.type_classifier(fused_features)
        
        return {
            'regret_intensity': intensity.squeeze(-1),
            'regret_type_probs': regret_type_probs,
            'fused_features': fused_features
        }

class CounterfactualGenerator(nn.Module):
    """반사실적 시나리오 생성 모델"""
    
    def __init__(self, input_dim: int = 768, latent_dim: int = 256):
        super().__init__()
        
        # 인코더: 원본 시나리오 → 잠재 공간
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2)  # mean과 logvar
        )
        
        # 디코더: 잠재 공간 → 대안 시나리오
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Tanh()
        )
        
        # 변화 강도 조절기
        self.change_modulator = nn.Sequential(
            nn.Linear(latent_dim + 1, 128),  # +1 for change intensity
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.Tanh()
        )
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """VAE 재매개화 트릭"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, scenario_embedding: torch.Tensor, 
                change_intensity: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        
        if change_intensity is None:
            change_intensity = torch.ones(scenario_embedding.size(0), 1, 
                                        device=scenario_embedding.device) * 0.5
        
        # 인코딩
        encoded = self.encoder(scenario_embedding)
        mu, logvar = torch.chunk(encoded, 2, dim=-1)
        
        # 재매개화
        z = self.reparameterize(mu, logvar)
        
        # 변화 강도 적용
        change_input = torch.cat([z, change_intensity], dim=-1)
        modulated_z = self.change_modulator(change_input)
        
        # 디코딩
        counterfactual = self.decoder(modulated_z)
        
        return {
            'counterfactual_embedding': counterfactual,
            'original_embedding': scenario_embedding,
            'latent_z': z,
            'mu': mu,
            'logvar': logvar,
            'change_intensity': change_intensity
        }

class RegretLearningModel(nn.Module):
    """후회 기반 학습 모델"""
    
    def __init__(self, input_dim: int = 768, memory_size: int = 1000):
        super().__init__()
        
        self.memory_size = memory_size
        self.experience_memory = []
        
        # 후회 예측기
        self.regret_predictor = RegretIntensityPredictor(input_dim)
        
        # 반사실적 생성기
        self.counterfactual_generator = CounterfactualGenerator(input_dim)
        
        # 의사결정 품질 평가기
        self.decision_evaluator = nn.Sequential(
            nn.Linear(input_dim * 2 + 1, 512),  # 원본 + 반사실적 + 후회강도
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # 학습률 조절기
        self.learning_rate_controller = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, scenario_embedding: torch.Tensor, 
                context_features: torch.Tensor,
                actual_outcome: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        # 1. 후회 예측
        regret_prediction = self.regret_predictor(scenario_embedding, context_features)
        
        # 2. 반사실적 시나리오 생성
        counterfactual_output = self.counterfactual_generator(
            scenario_embedding, 
            regret_prediction['regret_intensity'].unsqueeze(-1)
        )
        
        # 3. 의사결정 품질 평가
        decision_input = torch.cat([
            scenario_embedding,
            counterfactual_output['counterfactual_embedding'],
            regret_prediction['regret_intensity'].unsqueeze(-1)
        ], dim=-1)
        
        decision_quality = self.decision_evaluator(decision_input)
        
        # 4. 학습률 조절
        adaptive_lr = self.learning_rate_controller(regret_prediction['fused_features'])
        
        result = {
            'regret_intensity': regret_prediction['regret_intensity'],
            'regret_type_probs': regret_prediction['regret_type_probs'],
            'counterfactual_scenario': counterfactual_output['counterfactual_embedding'],
            'decision_quality': decision_quality.squeeze(-1),
            'adaptive_learning_rate': adaptive_lr.squeeze(-1),
            'latent_representation': counterfactual_output['latent_z']
        }
        
        # 실제 결과가 있는 경우 후회 계산
        if actual_outcome is not None:
            result['actual_regret'] = self._calculate_actual_regret(
                regret_prediction['regret_intensity'],
                decision_quality.squeeze(-1),
                actual_outcome
            )
        
        return result
    
    def _calculate_actual_regret(self, predicted_regret: torch.Tensor,
                               decision_quality: torch.Tensor,
                               actual_outcome: torch.Tensor) -> torch.Tensor:
        """실제 후회 계산"""
        # 예측과 실제 결과의 차이
        prediction_error = torch.abs(predicted_regret - actual_outcome)
        
        # 의사결정 품질에 따른 가중치
        quality_weight = 1.0 - decision_quality
        
        # 실제 후회 = 예측 오차 × 품질 가중치
        actual_regret = prediction_error * quality_weight
        
        return actual_regret
    
    def add_experience(self, experience: Dict[str, Any]):
        """경험 메모리에 추가"""
        if len(self.experience_memory) >= self.memory_size:
            self.experience_memory.pop(0)  # FIFO
        
        self.experience_memory.append(experience)
    
    def get_similar_experiences(self, current_embedding: torch.Tensor, 
                              top_k: int = 5) -> List[Dict[str, Any]]:
        """유사한 경험 검색"""
        if not self.experience_memory:
            return []
        
        similarities = []
        for exp in self.experience_memory:
            if 'embedding' in exp:
                sim = F.cosine_similarity(
                    current_embedding.unsqueeze(0),
                    exp['embedding'].unsqueeze(0)
                ).item()
                similarities.append((sim, exp))
        
        # 유사도 순으로 정렬하여 상위 k개 반환
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [exp for _, exp in similarities[:top_k]]

class RegretModelManager:
    """후회 모델 관리자"""
    
    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.training_stats = {}
        
    def save_model(self, model: RegretLearningModel, epoch: int, 
                   metrics: Dict[str, float]):
        """모델 저장"""
        model_path = self.models_dir / f"regret_model_epoch_{epoch}.pth"
        
        save_data = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'metrics': metrics,
            'experience_memory': model.experience_memory,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(save_data, model_path)
        
        # 통계 저장
        stats_path = self.models_dir / "training_stats.json"
        self.training_stats[epoch] = {
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_stats, f, indent=2, ensure_ascii=False)
            
    def load_model(self, model_path: Optional[Path] = None) -> RegretLearningModel:
        """모델 로드"""
        if model_path is None:
            model_files = list(self.models_dir.glob("regret_model_epoch_*.pth"))
            if not model_files:
                raise FileNotFoundError("저장된 후회 모델이 없습니다.")
            
            model_path = max(model_files, key=lambda p: p.stat().st_mtime)
        
        checkpoint = torch.load(model_path, map_location='cpu')
        
        model = RegretLearningModel()
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 경험 메모리 복원
        if 'experience_memory' in checkpoint:
            model.experience_memory = checkpoint['experience_memory']
        
        self.model = model
        return model

class RegretAnalysisTools:
    """후회 분석 도구들"""
    
    @staticmethod
    def analyze_regret_patterns(regret_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """후회 패턴 분석"""
        if not regret_history:
            return {}
        
        # 후회 유형별 빈도
        type_counts = {}
        intensities = []
        
        for entry in regret_history:
            regret_type = entry.get('dominant_type', 'unknown')
            type_counts[regret_type] = type_counts.get(regret_type, 0) + 1
            intensities.append(entry.get('intensity', 0))
        
        return {
            'type_distribution': type_counts,
            'average_intensity': np.mean(intensities),
            'intensity_trend': np.polyfit(range(len(intensities)), intensities, 1)[0],
            'peak_regret': max(intensities),
            'total_episodes': len(regret_history)
        }
    
    @staticmethod
    def generate_regret_insights(regret_patterns: Dict[str, Any]) -> List[str]:
        """후회 패턴 기반 인사이트 생성"""
        insights = []
        
        # 가장 빈번한 후회 유형
        if regret_patterns.get('type_distribution'):
            most_common = max(regret_patterns['type_distribution'].items(), 
                            key=lambda x: x[1])
            insights.append(f"가장 빈번한 후회 유형: {most_common[0]} ({most_common[1]}회)")
        
        # 후회 강도 트렌드
        trend = regret_patterns.get('intensity_trend', 0)
        if trend > 0.01:
            insights.append("후회 강도가 시간에 따라 증가하는 경향을 보입니다.")
        elif trend < -0.01:
            insights.append("후회 강도가 시간에 따라 감소하는 경향을 보입니다.")
        else:
            insights.append("후회 강도가 안정적인 수준을 유지하고 있습니다.")
        
        # 평균 후회 강도 평가
        avg_intensity = regret_patterns.get('average_intensity', 0)
        if avg_intensity > 0.7:
            insights.append("전반적으로 높은 수준의 후회를 경험하고 있습니다.")
        elif avg_intensity < 0.3:
            insights.append("후회 수준이 상대적으로 낮은 편입니다.")
        
        return insights

def create_regret_context(decision_data: Dict[str, Any]) -> RegretContext:
    """의사결정 데이터로부터 후회 맥락 생성"""
    return RegretContext(
        decision_quality=decision_data.get('quality_score', 0.5),
        outcome_deviation=decision_data.get('outcome_deviation', 0.5),
        time_pressure=decision_data.get('time_pressure', 0.5),
        information_completeness=decision_data.get('info_completeness', 0.5),
        emotional_involvement=decision_data.get('emotional_level', 0.5),
        social_consequences=decision_data.get('social_impact', 0.5),
        reversibility=decision_data.get('reversibility', 0.5)
    )

def regret_context_to_tensor(context: RegretContext) -> torch.Tensor:
    """후회 맥락을 텐서로 변환"""
    return torch.tensor([
        context.decision_quality,
        context.outcome_deviation,
        context.time_pressure,
        context.information_completeness,
        context.emotional_involvement,
        context.social_consequences,
        context.reversibility
    ], dtype=torch.float32)