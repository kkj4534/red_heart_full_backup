"""
반사실적 추론 모델들
Counterfactual Reasoning Models
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
import copy
import random

class CounterfactualType(Enum):
    """반사실적 시나리오 유형"""
    ALTERNATIVE_ACTION = "alternative_action"     # 대안 행동
    DIFFERENT_TIMING = "different_timing"         # 다른 타이밍
    REVERSED_DECISION = "reversed_decision"       # 반대 결정
    MODIFIED_CONTEXT = "modified_context"         # 수정된 맥락
    ENHANCED_INFORMATION = "enhanced_information" # 추가 정보
    DIFFERENT_VALUES = "different_values"         # 다른 가치관

@dataclass
class CounterfactualConfig:
    """반사실적 추론 설정"""
    input_dim: int = 768
    hidden_dims: List[int] = None
    latent_dim: int = 256
    num_scenarios: int = 5
    scenario_diversity_weight: float = 0.3
    plausibility_weight: float = 0.4
    impact_weight: float = 0.3
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256, 128]
        # 최소 3개 차원 보장
        while len(self.hidden_dims) < 3:
            self.hidden_dims.append(self.hidden_dims[-1] // 2 if self.hidden_dims else 128)

@dataclass
class CounterfactualScenario:
    """반사실적 시나리오"""
    scenario_type: CounterfactualType
    original_embedding: torch.Tensor
    counterfactual_embedding: torch.Tensor
    plausibility_score: float
    impact_score: float
    confidence: float
    explanation: str
    key_changes: List[str]

class VariationalCounterfactualEncoder(nn.Module):
    """변분 반사실적 인코더"""
    
    def __init__(self, config: CounterfactualConfig):
        super().__init__()
        self.config = config
        
        # 인코더 네트워크 (VAE)
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_dims[0], config.hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_dims[1], config.latent_dim * 2)  # mean과 logvar
        )
        
        # 시나리오 타입별 특화 인코더 (안전한 인덱스 사용)
        safe_hidden_dim = config.hidden_dims[-1] if len(config.hidden_dims) > 0 else 128
        self.type_encoders = nn.ModuleDict({
            scenario_type.value: nn.Sequential(
                nn.Linear(config.latent_dim, safe_hidden_dim),
                nn.ReLU(),
                nn.Linear(safe_hidden_dim, config.latent_dim)
            ) for scenario_type in CounterfactualType
        })
        
        # 디코더 네트워크
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_dims[1], config.hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(config.hidden_dims[0], config.input_dim),
            nn.Tanh()
        )
        
        # 가능성 점수 예측기
        self.plausibility_predictor = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(config.hidden_dims[2], 1),
            nn.Sigmoid()
        )
        
        # 영향도 점수 예측기
        self.impact_predictor = nn.Sequential(
            nn.Linear(config.input_dim * 2, config.hidden_dims[1]),  # 원본과 반사실적 연결
            nn.ReLU(),
            nn.Linear(config.hidden_dims[1], config.hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(config.hidden_dims[2], 1),
            nn.Sigmoid()
        )
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """VAE 재매개화 트릭"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, original_embedding: torch.Tensor, 
                scenario_type: CounterfactualType,
                diversity_factor: float = 1.0) -> Dict[str, torch.Tensor]:
        
        # 원본 임베딩 인코딩
        encoded = self.encoder(original_embedding)
        mu, logvar = torch.chunk(encoded, 2, dim=-1)
        
        # 재매개화
        z = self.reparameterize(mu, logvar)
        
        # 시나리오 타입별 변형
        type_encoder = self.type_encoders[scenario_type.value]
        modified_z = type_encoder(z)
        
        # 다양성 인자 적용
        if diversity_factor != 1.0:
            noise = torch.randn_like(modified_z) * (diversity_factor - 1.0) * 0.1
            modified_z = modified_z + noise
        
        # 반사실적 임베딩 생성
        counterfactual_embedding = self.decoder(modified_z)
        
        # 가능성 점수 예측
        plausibility = self.plausibility_predictor(modified_z)
        
        # 영향도 점수 예측
        combined_input = torch.cat([original_embedding, counterfactual_embedding], dim=-1)
        impact = self.impact_predictor(combined_input)
        
        return {
            'counterfactual_embedding': counterfactual_embedding,
            'latent_z': modified_z,
            'mu': mu,
            'logvar': logvar,
            'plausibility': plausibility.squeeze(-1),
            'impact': impact.squeeze(-1)
        }

class CounterfactualGenerator(nn.Module):
    """반사실적 시나리오 생성기"""
    
    def __init__(self, config: CounterfactualConfig):
        super().__init__()
        self.config = config
        
        # 각 시나리오 타입별 인코더
        self.encoders = nn.ModuleDict({
            scenario_type.value: VariationalCounterfactualEncoder(config)
            for scenario_type in CounterfactualType
        })
        
        # 시나리오 품질 평가기
        self.quality_evaluator = nn.Sequential(
            nn.Linear(config.input_dim * 2 + 2, config.hidden_dims[1]),  # +2 for plausibility & impact
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_dims[1], config.hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(config.hidden_dims[2], 1),
            nn.Sigmoid()
        )
        
        # 시나리오 선택기 (어텐션 기반)
        self.scenario_selector = nn.MultiheadAttention(
            embed_dim=config.input_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
    def generate_multiple_scenarios(self, original_embedding: torch.Tensor,
                                  num_scenarios: Optional[int] = None) -> Dict[str, Any]:
        """다중 반사실적 시나리오 생성"""
        if num_scenarios is None:
            num_scenarios = self.config.num_scenarios
        
        scenarios = []
        scenario_embeddings = []
        
        # 각 시나리오 타입별로 생성
        for scenario_type in CounterfactualType:
            encoder = self.encoders[scenario_type.value]
            
            # 다양성을 위해 여러 번 생성
            for diversity in [0.5, 1.0, 1.5]:
                output = encoder(original_embedding, scenario_type, diversity)
                
                # 품질 평가
                quality_input = torch.cat([
                    original_embedding,
                    output['counterfactual_embedding'],
                    output['plausibility'].unsqueeze(-1),
                    output['impact'].unsqueeze(-1)
                ], dim=-1)
                
                quality_score = self.quality_evaluator(quality_input)
                
                scenario = {
                    'type': scenario_type,
                    'embedding': output['counterfactual_embedding'],
                    'plausibility': output['plausibility'],
                    'impact': output['impact'],
                    'quality': quality_score.squeeze(-1),
                    'diversity': diversity,
                    'latent': output['latent_z']
                }
                
                scenarios.append(scenario)
                scenario_embeddings.append(output['counterfactual_embedding'])
        
        # 상위 N개 시나리오 선택
        if scenarios:
            def get_quality_score(scenario):
                quality = scenario['quality']
                if torch.is_tensor(quality):
                    if quality.numel() == 1:
                        return quality.item()
                    else:
                        return quality.mean().item()
                else:
                    return float(quality)
            
            scenarios.sort(key=get_quality_score, reverse=True)
            top_scenarios = scenarios[:num_scenarios]
        else:
            top_scenarios = []
        
        # 선택된 시나리오들의 임베딩
        if top_scenarios:
            try:
                selected_embeddings = torch.stack([s['embedding'] for s in top_scenarios], dim=1)
                
                # 어텐션 기반 시나리오 선택
                original_expanded = original_embedding.unsqueeze(1)
                attended_scenarios, attention_weights = self.scenario_selector(
                    original_expanded, selected_embeddings, selected_embeddings
                )
            except Exception as e:
                # 스택 실패 시 더미 데이터 생성
                attended_scenarios = original_embedding.unsqueeze(1)
                attention_weights = torch.ones(original_embedding.shape[0], 1, 1, device=original_embedding.device)
        else:
            # 시나리오가 없는 경우 더미 시나리오 생성
            dummy_scenario = {
                'type': CounterfactualType.ALTERNATIVE_ACTION,
                'embedding': original_embedding,
                'plausibility': torch.tensor(0.5, device=original_embedding.device),
                'impact': torch.tensor(0.5, device=original_embedding.device),
                'quality': torch.tensor(0.5, device=original_embedding.device),
                'diversity': 1.0,
                'latent': torch.zeros(original_embedding.shape[0], self.config.latent_dim, device=original_embedding.device)
            }
            top_scenarios = [dummy_scenario]
            attended_scenarios = original_embedding.unsqueeze(1)
            attention_weights = torch.ones(original_embedding.shape[0], 1, 1, device=original_embedding.device)
        
        return {
            'scenarios': top_scenarios,
            'attended_scenarios': attended_scenarios,
            'attention_weights': attention_weights,
            'scenario_count': len(top_scenarios)
        }

class CounterfactualExplainer(nn.Module):
    """반사실적 설명 생성기"""
    
    def __init__(self, config: CounterfactualConfig, vocab_size: int = 1000):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        
        # 특징 차이 분석기
        self.difference_analyzer = nn.Sequential(
            nn.Linear(config.input_dim * 2, config.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_dims[0], config.hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(config.hidden_dims[1], config.hidden_dims[2])
        )
        
        # 설명 생성 네트워크 (간단한 시퀀스 모델)
        self.explanation_generator = nn.LSTM(
            input_size=config.hidden_dims[2],
            hidden_size=config.hidden_dims[1],
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )
        
        # 출력 투영 (어휘 크기로)
        self.output_projection = nn.Linear(config.hidden_dims[1], vocab_size)
        
        # 중요도 점수 생성기
        self.importance_scorer = nn.Sequential(
            nn.Linear(config.hidden_dims[2], config.hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(config.hidden_dims[2], config.input_dim),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, original_embedding: torch.Tensor, 
                counterfactual_embedding: torch.Tensor,
                max_length: int = 50) -> Dict[str, torch.Tensor]:
        
        # 특징 차이 분석
        combined_input = torch.cat([original_embedding, counterfactual_embedding], dim=-1)
        difference_features = self.difference_analyzer(combined_input)
        
        # 중요도 점수 계산
        importance_scores = self.importance_scorer(difference_features)
        
        # 설명 생성 준비
        sequence_input = difference_features.unsqueeze(1).repeat(1, max_length, 1)
        
        # LSTM을 통한 설명 생성
        lstm_output, _ = self.explanation_generator(sequence_input)
        
        # 어휘 확률 분포
        vocab_logits = self.output_projection(lstm_output)
        vocab_probs = F.softmax(vocab_logits, dim=-1)
        
        return {
            'difference_features': difference_features,
            'importance_scores': importance_scores,
            'vocab_probabilities': vocab_probs,
            'explanation_embedding': lstm_output.mean(dim=1)
        }

class AdvancedCounterfactualModel(nn.Module):
    """고급 반사실적 추론 모델"""
    
    def __init__(self, config: CounterfactualConfig):
        super().__init__()
        self.config = config
        
        # 반사실적 생성기
        self.generator = CounterfactualGenerator(config)
        
        # 설명 생성기
        self.explainer = CounterfactualExplainer(config)
        
        # 일관성 검사기 (동적 차원)
        self.consistency_checker = None  # forward에서 동적 생성
        
        # 신뢰도 추정기
        self.confidence_estimator = nn.Sequential(
            nn.Linear(config.input_dim + 3, config.hidden_dims[2]),  # +3 for quality metrics
            nn.ReLU(),
            nn.Linear(config.hidden_dims[2], 1),
            nn.Sigmoid()
        )
        
    def forward(self, original_embedding: torch.Tensor, 
                context_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        
        # 다중 반사실적 시나리오 생성
        generation_output = self.generator.generate_multiple_scenarios(original_embedding)
        
        results = []
        
        # 시나리오 안전성 검사
        scenarios = generation_output.get('scenarios', [])
        if not scenarios:
            # 빈 시나리오 리스트인 경우 더미 결과 반환
            return {
                'counterfactual_scenarios': [],
                'generation_summary': generation_output,
                'best_scenario': None
            }
        
        for scenario in scenarios:
            # 설명 생성
            explanation_output = self.explainer(
                original_embedding, 
                scenario['embedding']
            )
            
            # 일관성 검사 (동적 생성)
            consistency_input = torch.cat([
                original_embedding,
                scenario['embedding'],
                explanation_output['explanation_embedding']
            ], dim=-1)
            
            if self.consistency_checker is None:
                input_dim = consistency_input.shape[-1]
                self.consistency_checker = nn.Sequential(
                    nn.Linear(input_dim, self.config.hidden_dims[1]),
                    nn.ReLU(),
                    nn.Linear(self.config.hidden_dims[1], self.config.hidden_dims[2]),
                    nn.ReLU(),
                    nn.Linear(self.config.hidden_dims[2], 1),
                    nn.Sigmoid()
                ).to(consistency_input.device)
            
            consistency_score = self.consistency_checker(consistency_input)
            
            # 신뢰도 추정
            confidence_input = torch.cat([
                original_embedding,
                scenario['plausibility'].unsqueeze(-1),
                scenario['impact'].unsqueeze(-1),
                consistency_score
            ], dim=-1)
            confidence = self.confidence_estimator(confidence_input)
            
            # 결과 정리
            result = {
                'scenario_type': scenario['type'],
                'counterfactual_embedding': scenario['embedding'],
                'plausibility': scenario['plausibility'],
                'impact': scenario['impact'],
                'quality': scenario['quality'],
                'consistency': consistency_score.squeeze(-1),
                'confidence': confidence.squeeze(-1),
                'explanation_features': explanation_output['difference_features'],
                'importance_scores': explanation_output['importance_scores']
            }
            
            results.append(result)
        
        # 최고 시나리오 선택
        best_scenario = None
        if results:
            def get_confidence_score(result):
                confidence = result['confidence']
                if torch.is_tensor(confidence):
                    if confidence.numel() == 1:
                        return confidence.item()
                    else:
                        return confidence.mean().item()
                else:
                    return float(confidence)
            
            best_scenario = max(results, key=get_confidence_score)
        
        return {
            'counterfactual_scenarios': results,
            'generation_summary': generation_output,
            'best_scenario': best_scenario
        }

class CounterfactualAnalysisTools:
    """반사실적 분석 도구들"""
    
    @staticmethod
    def analyze_scenario_distribution(scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """시나리오 분포 분석"""
        type_counts = {}
        quality_scores = []
        confidence_scores = []
        
        for scenario in scenarios:
            scenario_type = scenario.get('scenario_type')
            if scenario_type:
                type_value = getattr(scenario_type, 'value', str(scenario_type))
                type_counts[type_value] = type_counts.get(type_value, 0) + 1
            
            if isinstance(scenario['quality'], torch.Tensor):
                quality_scores.append(scenario['quality'].item())
            else:
                quality_scores.append(scenario['quality'])
                
            if isinstance(scenario['confidence'], torch.Tensor):
                confidence_scores.append(scenario['confidence'].item())
            else:
                confidence_scores.append(scenario['confidence'])
        
        return {
            'type_distribution': type_counts,
            'average_quality': np.mean(quality_scores) if quality_scores else 0,
            'average_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'quality_variance': np.var(quality_scores) if quality_scores else 0,
            'total_scenarios': len(scenarios)
        }
    
    @staticmethod
    def rank_scenarios(scenarios: List[Dict[str, Any]], 
                      weights: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """시나리오 순위 매기기"""
        if weights is None:
            weights = {
                'plausibility': 0.3,
                'impact': 0.3,
                'quality': 0.2,
                'confidence': 0.2
            }
        
        scored_scenarios = []
        
        for scenario in scenarios:
            score = 0
            for metric, weight in weights.items():
                if metric in scenario:
                    value = scenario[metric]
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    score += value * weight
            
            scenario_copy = copy.deepcopy(scenario)
            scenario_copy['composite_score'] = score
            scored_scenarios.append(scenario_copy)
        
        # 점수순으로 정렬
        return sorted(scored_scenarios, key=lambda x: x['composite_score'], reverse=True)
    
    @staticmethod
    def generate_insights(scenarios: List[Dict[str, Any]]) -> List[str]:
        """반사실적 분석 인사이트 생성"""
        insights = []
        
        if not scenarios:
            return ["분석할 시나리오가 없습니다."]
        
        # 가장 유망한 시나리오
        if scenarios:
            best_scenario = max(scenarios, key=lambda x: x.get('composite_score', 0))
            scenario_type = getattr(best_scenario.get('scenario_type'), 'value', 'unknown')
            insights.append(f"가장 유망한 대안: {scenario_type}")
        else:
            insights.append("분석 가능한 시나리오가 없습니다.")
        
        # 평균 품질 분석
        avg_quality = np.mean([s.get('quality', 0) for s in scenarios])
        if avg_quality > 0.7:
            insights.append("전반적으로 높은 품질의 대안들이 제시되었습니다.")
        elif avg_quality < 0.4:
            insights.append("제시된 대안들의 품질이 상대적으로 낮습니다.")
        
        # 시나리오 다양성 분석
        types = set()
        for s in scenarios:
            scenario_type = s.get('scenario_type')
            if scenario_type:
                types.add(getattr(scenario_type, 'value', str(scenario_type)))
        
        if len(types) > 3:
            insights.append("다양한 유형의 대안들이 탐색되었습니다.")
        
        return insights

class CounterfactualModelManager:
    """반사실적 모델 관리자"""
    
    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.config = None
        
    def save_model(self, model: AdvancedCounterfactualModel, config: CounterfactualConfig,
                   epoch: int, metrics: Dict[str, float]):
        """모델 저장"""
        model_path = self.models_dir / f"counterfactual_model_epoch_{epoch}.pth"
        config_path = self.models_dir / "counterfactual_config.json"
        
        # 모델 저장
        save_data = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        torch.save(save_data, model_path)
        
        # 설정 저장
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(config), f, indent=2, ensure_ascii=False)
            
    def load_model(self, model_path: Optional[Path] = None) -> AdvancedCounterfactualModel:
        """모델 로드"""
        # 설정 로드
        config_path = self.models_dir / "counterfactual_config.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            self.config = CounterfactualConfig(**config_dict)
        else:
            self.config = CounterfactualConfig()
        
        if model_path is None:
            model_files = list(self.models_dir.glob("counterfactual_model_epoch_*.pth"))
            if not model_files:
                raise FileNotFoundError("저장된 반사실적 모델이 없습니다.")
            
            model_path = max(model_files, key=lambda p: p.stat().st_mtime)
        
        checkpoint = torch.load(model_path, map_location='cpu')
        
        model = AdvancedCounterfactualModel(self.config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model = model
        return model

def create_counterfactual_config(**kwargs) -> CounterfactualConfig:
    """반사실적 추론 설정 생성 헬퍼"""
    return CounterfactualConfig(**kwargs)

def generate_counterfactual_explanation(scenario: Dict[str, Any]) -> str:
    """반사실적 시나리오 설명 생성"""
    scenario_type = scenario['scenario_type'].value
    confidence = scenario.get('confidence', 0)
    impact = scenario.get('impact', 0)
    
    type_descriptions = {
        'alternative_action': "다른 행동을 선택했다면",
        'different_timing': "다른 시점에 결정했다면",
        'reversed_decision': "반대 결정을 내렸다면",
        'modified_context': "상황이 달랐다면",
        'enhanced_information': "더 많은 정보가 있었다면",
        'different_values': "다른 가치관을 가졌다면"
    }
    
    base_description = type_descriptions.get(scenario_type, "다른 선택을 했다면")
    
    if confidence > 0.8:
        confidence_desc = "매우 높은 확신으로"
    elif confidence > 0.6:
        confidence_desc = "높은 확신으로"
    elif confidence > 0.4:
        confidence_desc = "중간 정도의 확신으로"
    else:
        confidence_desc = "낮은 확신으로"
    
    if impact > 0.8:
        impact_desc = "결과에 큰 변화가 있었을 것"
    elif impact > 0.6:
        impact_desc = "결과에 상당한 변화가 있었을 것"
    elif impact > 0.4:
        impact_desc = "결과에 어느 정도 변화가 있었을 것"
    else:
        impact_desc = "결과에 제한적인 변화가 있었을 것"
    
    return f"{base_description} {confidence_desc} {impact_desc}입니다."