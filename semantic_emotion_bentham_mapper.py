#!/usr/bin/env python3
"""
의미론적 감정-벤담 정밀 매핑 시스템
Semantic Emotion-Bentham Precision Mapping System

감정의 6차원과 벤담의 10차원을 의미론적으로 연결하는 정밀 매핑
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger('RedHeart.SemanticMapper')

# 감정 차원 정의 (6차원)
EMOTION_DIMENSIONS = {
    'valence': 0,      # 감정가 (긍정/부정) -1 ~ 1
    'arousal': 1,      # 각성도 (활성화/비활성화) -1 ~ 1
    'dominance': 2,    # 지배감 (통제/무력감) -1 ~ 1
    'certainty': 3,    # 확실성 (확신/불확실) 0 ~ 1
    'surprise': 4,     # 놀라움 (예상/예상외) 0 ~ 1
    'anticipation': 5  # 기대감 (기대/무관심) 0 ~ 1
}

# 벤담 차원 정의 (10차원)
BENTHAM_DIMENSIONS = {
    'intensity': 0,            # 강도
    'duration': 1,             # 지속성
    'certainty': 2,            # 확실성
    'propinquity': 3,          # 근접성
    'fecundity': 4,            # 생산성
    'purity': 5,               # 순수성
    'extent': 6,               # 범위
    'external_cost': 7,        # 외부 비용
    'redistribution_effect': 8, # 재분배 효과
    'self_damage': 9           # 자기 피해
}


class SemanticEmotionBenthamMapper:
    """의미론적 연결 기반 정밀 매핑"""
    
    def __init__(self):
        # 의미론적 매핑 규칙 (각 벤담 차원에 대한 감정 차원 기여도)
        self.mapping_rules = {
            'intensity': [
                ('arousal', 0.6),      # 각성도가 강도의 주요 요인
                ('valence', 0.3),      # 감정가의 절댓값도 강도에 기여
                ('dominance', 0.1)     # 지배감도 약간 영향
            ],
            'duration': [
                ('dominance', 0.5),    # 통제감이 높으면 지속적
                ('certainty', 0.3),    # 확실하면 오래 지속
                ('arousal', -0.2)      # 과도한 각성은 짧게 지속
            ],
            'certainty': [
                ('certainty', 0.8),    # 감정 확실성이 직접 매핑
                ('surprise', -0.2)     # 놀라움은 불확실성 증가
            ],
            'propinquity': [
                ('anticipation', 0.7), # 기대감이 시간적 근접성
                ('arousal', 0.3)       # 각성도도 즉시성에 영향
            ],
            'fecundity': [
                ('valence', 0.4),      # 긍정 감정이 생산적
                ('anticipation', 0.4), # 기대감이 미래 생산성
                ('dominance', 0.2)     # 통제감이 생산성 증가
            ],
            'purity': [
                ('valence', 0.5),      # 명확한 감정가
                ('certainty', 0.3),    # 확실성이 순수성
                ('surprise', -0.2)     # 놀라움은 혼란 유발
            ],
            'extent': [
                ('dominance', 0.4),    # 지배감이 영향 범위
                ('valence', 0.3),      # 감정의 전파력
                ('arousal', 0.3)       # 각성도의 확산력
            ],
            'external_cost': [
                ('valence', -0.5),     # 부정 감정이 외부 비용
                ('arousal', 0.3),      # 과도한 각성도 비용
                ('surprise', 0.2)      # 예상치 못한 것의 비용
            ],
            'redistribution_effect': [
                ('dominance', -0.4),   # 무력감이 재분배 필요성
                ('valence', 0.3),      # 긍정적이면 나눔 의지
                ('certainty', 0.3)     # 확실한 상황에서 재분배
            ],
            'self_damage': [
                ('valence', -0.6),     # 부정 감정이 자기 손상
                ('dominance', -0.3),   # 무력감도 자기 손상
                ('certainty', -0.1)    # 불확실성도 약간 영향
            ]
        }
        
        logger.info("SemanticEmotionBenthamMapper 초기화 완료")
    
    def _normalize_emotion_vector(self, emotion_data: Union[Dict, List, np.ndarray, torch.Tensor]) -> Dict[str, float]:
        """감정 벡터를 정규화된 딕셔너리로 변환"""
        
        # Dict 형태인 경우
        if isinstance(emotion_data, dict):
            normalized = {}
            for dim_name in EMOTION_DIMENSIONS.keys():
                if dim_name in emotion_data:
                    normalized[dim_name] = float(emotion_data[dim_name])
                else:
                    # scores 리스트가 있는 경우
                    if 'scores' in emotion_data and isinstance(emotion_data['scores'], (list, np.ndarray)):
                        scores = emotion_data['scores']
                        dim_idx = EMOTION_DIMENSIONS[dim_name]
                        if len(scores) > dim_idx:
                            normalized[dim_name] = float(scores[dim_idx])
                        else:
                            normalized[dim_name] = 0.0
                    else:
                        normalized[dim_name] = 0.0
            return normalized
        
        # List/Array/Tensor 형태인 경우
        elif isinstance(emotion_data, (list, np.ndarray, torch.Tensor)):
            if isinstance(emotion_data, torch.Tensor):
                emotion_data = emotion_data.detach().cpu().numpy()
            elif isinstance(emotion_data, list):
                emotion_data = np.array(emotion_data)
            
            normalized = {}
            for dim_name, dim_idx in EMOTION_DIMENSIONS.items():
                if len(emotion_data) > dim_idx:
                    normalized[dim_name] = float(emotion_data[dim_idx])
                else:
                    normalized[dim_name] = 0.0
            return normalized
        
        else:
            # 기본값
            return {dim: 0.0 for dim in EMOTION_DIMENSIONS.keys()}
    
    def map_emotion_to_bentham(self, emotion_data: Union[Dict, List, np.ndarray, torch.Tensor]) -> Dict[str, float]:
        """6차원 감정을 10차원 벤담으로 의미론적 변환"""
        
        # 감정 벡터 정규화
        emotion_vec = self._normalize_emotion_vector(emotion_data)
        
        # 벤담 벡터 초기화
        bentham_params = {}
        
        # 각 벤담 차원에 대해 의미론적 매핑 적용
        for bentham_name, mappings in self.mapping_rules.items():
            value = 0.0
            
            # 관련 감정 차원들의 가중 합
            for emotion_name, weight in mappings:
                emotion_value = emotion_vec.get(emotion_name, 0.0)
                
                # valence의 절댓값 처리 (강도 계산 시)
                if bentham_name in ['intensity', 'purity', 'extent'] and emotion_name == 'valence':
                    emotion_value = abs(emotion_value)
                
                value += emotion_value * weight
            
            # Sigmoid로 0~1 범위 정규화
            normalized_value = 1 / (1 + np.exp(-value))
            bentham_params[bentham_name] = normalized_value
        
        return bentham_params
    
    def map_with_hierarchy(self, emotion_data: Union[Dict, List], hierarchy_level: str = 'self') -> Dict[str, float]:
        """계층적 감정 처리를 고려한 매핑 (공동체>타자>자아)"""
        
        # 기본 매핑
        bentham_params = self.map_emotion_to_bentham(emotion_data)
        
        # 계층별 가중치 조정
        if hierarchy_level == 'community':
            # 공동체 레벨: 범위와 재분배 강조
            bentham_params['extent'] = min(1.0, bentham_params['extent'] * 1.5)
            bentham_params['redistribution_effect'] = min(1.0, bentham_params['redistribution_effect'] * 1.3)
            bentham_params['self_damage'] = bentham_params['self_damage'] * 0.5  # 자기 손상 덜 중요
            
        elif hierarchy_level == 'other':
            # 타자 레벨: 생산성과 순수성 강조
            bentham_params['fecundity'] = min(1.0, bentham_params['fecundity'] * 1.2)
            bentham_params['purity'] = min(1.0, bentham_params['purity'] * 1.1)
            
        else:  # self
            # 자아 레벨: 지속성과 자기 손상 강조
            bentham_params['duration'] = min(1.0, bentham_params['duration'] * 1.1)
            bentham_params['self_damage'] = min(1.0, bentham_params['self_damage'] * 1.2)
        
        return bentham_params


class NeuralEmotionBenthamAdapter(nn.Module):
    """학습 가능한 감정→벤담 변환 신경망"""
    
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        
        # 확장 레이어
        self.expand_layer = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # 변환 레이어 (2층)
        self.transform_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 출력 레이어
        self.output_layer = nn.Linear(hidden_dim, 10)
        
        # 직접 연결 (잔차)
        self.direct_mapping = nn.Linear(6, 10)
        
        # 게이팅 메커니즘
        self.gate = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Linear(16, 10),
            nn.Sigmoid()
        )
        
        # 의미론적 매핑 초기화 (사전 지식 주입)
        self._initialize_with_semantic_knowledge()
    
    def _initialize_with_semantic_knowledge(self):
        """의미론적 사전 지식으로 가중치 초기화"""
        # SemanticMapper의 규칙을 초기 가중치로 사용
        mapper = SemanticEmotionBenthamMapper()
        
        # 직접 매핑 레이어에 의미론적 규칙 반영
        with torch.no_grad():
            for b_idx, b_name in enumerate(BENTHAM_DIMENSIONS.keys()):
                if b_name in mapper.mapping_rules:
                    for e_name, weight in mapper.mapping_rules[b_name]:
                        e_idx = EMOTION_DIMENSIONS[e_name]
                        self.direct_mapping.weight[b_idx, e_idx] = weight * 0.5
    
    def forward(self, emotion_vector: torch.Tensor) -> torch.Tensor:
        """감정 벡터를 벤담 벡터로 변환"""
        
        # 차원 체크
        if emotion_vector.dim() == 1:
            emotion_vector = emotion_vector.unsqueeze(0)
        
        # 주 변환 경로
        expanded = self.expand_layer(emotion_vector)
        transformed = self.transform_layers(expanded)
        main_output = torch.sigmoid(self.output_layer(transformed))
        
        # 직접 매핑 경로 (잔차)
        direct_output = torch.sigmoid(self.direct_mapping(emotion_vector))
        
        # 게이팅으로 두 경로 혼합
        gate_weights = self.gate(emotion_vector)
        output = gate_weights * main_output + (1 - gate_weights) * direct_output
        
        return output


def create_precision_mapper(mode: str = 'semantic') -> Union[SemanticEmotionBenthamMapper, NeuralEmotionBenthamAdapter]:
    """정밀 매퍼 생성 팩토리 함수"""
    
    if mode == 'semantic':
        return SemanticEmotionBenthamMapper()
    elif mode == 'neural':
        return NeuralEmotionBenthamAdapter()
    else:
        raise ValueError(f"Unknown mapper mode: {mode}")


# 테스트 코드
if __name__ == '__main__':
    # 의미론적 매퍼 테스트
    print("=" * 50)
    print("의미론적 감정→벤담 매퍼 테스트")
    print("=" * 50)
    
    mapper = SemanticEmotionBenthamMapper()
    
    # 테스트 감정 데이터
    test_emotions = {
        'valence': 0.7,      # 긍정적
        'arousal': 0.5,      # 중간 각성
        'dominance': 0.8,    # 높은 통제감
        'certainty': 0.9,    # 매우 확실
        'surprise': 0.1,     # 예상됨
        'anticipation': 0.6  # 어느정도 기대
    }
    
    # 매핑 수행
    bentham_result = mapper.map_emotion_to_bentham(test_emotions)
    
    print("\n입력 감정:")
    for k, v in test_emotions.items():
        print(f"  {k:15s}: {v:.2f}")
    
    print("\n출력 벤담 파라미터:")
    for k, v in bentham_result.items():
        print(f"  {k:20s}: {v:.3f}")
    
    # 계층별 매핑 테스트
    print("\n" + "=" * 50)
    print("계층별 매핑 차이")
    print("=" * 50)
    
    for level in ['self', 'other', 'community']:
        result = mapper.map_with_hierarchy(test_emotions, level)
        print(f"\n{level.upper()} 레벨:")
        print(f"  extent: {result['extent']:.3f}")
        print(f"  redistribution_effect: {result['redistribution_effect']:.3f}")
        print(f"  self_damage: {result['self_damage']:.3f}")
    
    # 신경망 어댑터 테스트
    print("\n" + "=" * 50)
    print("신경망 어댑터 테스트")
    print("=" * 50)
    
    adapter = NeuralEmotionBenthamAdapter()
    emotion_tensor = torch.tensor([[0.7, 0.5, 0.8, 0.9, 0.1, 0.6]], dtype=torch.float32)
    
    with torch.no_grad():
        neural_output = adapter(emotion_tensor)
        print(f"\n신경망 출력 형태: {neural_output.shape}")
        print("벤담 파라미터:")
        for idx, name in enumerate(BENTHAM_DIMENSIONS.keys()):
            print(f"  {name:20s}: {neural_output[0, idx].item():.3f}")