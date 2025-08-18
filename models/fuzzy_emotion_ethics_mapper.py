#!/usr/bin/env python3
"""
퍼지 감정-윤리 매핑 시스템 (Fuzzy Emotion-Ethics Mapper)
Fuzzy Logic 기반 감정 → 윤리 판단 매핑

docs/연동에 대한 문제.txt 개선사항 구현:
- 퍼지 감정 → 윤리 판단 매핑
- 명확한 이산적 선택이 아닌, 감정 흐름의 정도에 따른 윤리 판단 변화
- 행복: 0.2 / 슬픔: 0.7 → 선택지 중 "위로성 높은" 선택에 점수 부여
- Fuzzy Logic 기반으로 감정과 판단 사이 연속적 연결
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import logging
import json
from enum import Enum
from collections import defaultdict

class EmotionDimension(Enum):
    """감정 차원 정의"""
    VALENCE = "valence"           # 감정가 (긍정/부정)
    AROUSAL = "arousal"           # 각성도 (활성화/비활성화)
    DOMINANCE = "dominance"       # 지배감 (통제/무력감)
    CERTAINTY = "certainty"       # 확실성 (확신/불확실)
    SURPRISE = "surprise"         # 놀라움 (예상/예상외)
    ANTICIPATION = "anticipation" # 기대감 (기대/무관심)

class EthicsCategory(Enum):
    """윤리 카테고리 정의"""
    RULE_BASED = "rule_based"         # 규칙 기반 윤리
    CONSEQUENCE = "consequence"       # 결과 기반 윤리
    VIRTUE = "virtue"                # 덕윤리
    CARE = "care"                    # 배려 윤리
    JUSTICE = "justice"              # 정의 윤리
    AUTONOMY = "autonomy"            # 자율성 존중

@dataclass
class FuzzyRule:
    """퍼지 규칙 정의"""
    emotion_conditions: Dict[EmotionDimension, Tuple[float, float]]  # (min, max) 범위
    ethics_output: Dict[EthicsCategory, float]  # 윤리 카테고리별 가중치
    rule_name: str
    confidence: float = 1.0
    description: str = ""
    
    def evaluate(self, emotion_state: Dict[EmotionDimension, float]) -> float:
        """규칙 적용도 계산"""
        membership_degrees = []
        
        for emotion_dim, (min_val, max_val) in self.emotion_conditions.items():
            if emotion_dim in emotion_state:
                emotion_value = emotion_state[emotion_dim]
                # 삼각형 멤버십 함수
                membership = self._triangular_membership(emotion_value, min_val, max_val)
                membership_degrees.append(membership)
        
        if not membership_degrees:
            return 0.0
        
        # 최소값 사용 (AND 연산)
        rule_strength = min(membership_degrees) * self.confidence
        return rule_strength
    
    def _triangular_membership(self, value: float, min_val: float, max_val: float) -> float:
        """삼각형 멤버십 함수"""
        if value < min_val or value > max_val:
            return 0.0
        
        center = (min_val + max_val) / 2
        
        if value <= center:
            # 상승 구간
            return (value - min_val) / (center - min_val) if center != min_val else 1.0
        else:
            # 하강 구간
            return (max_val - value) / (max_val - center) if max_val != center else 1.0

class FuzzyEmotionEthicsMapper:
    """퍼지 감정-윤리 매핑 시스템"""
    
    def __init__(self):
        self.fuzzy_rules = []
        self.emotion_history = []
        self.ethics_decision_history = []
        
        # 기본 퍼지 규칙들 초기화
        self._initialize_default_rules()
        
        # 학습 가능한 규칙 가중치
        self.rule_weights = nn.Parameter(torch.ones(len(self.fuzzy_rules)))
        
        # 감정-윤리 연결 강도 매트릭스
        self.connection_matrix = self._initialize_connection_matrix()
        
        self.logger = logging.getLogger('FuzzyEmotionEthicsMapper')
    
    def _initialize_default_rules(self):
        """기본 퍼지 규칙들 초기화"""
        
        # 규칙 1: 높은 슬픔 + 낮은 각성 → 배려 윤리 강화
        self.fuzzy_rules.append(FuzzyRule(
            emotion_conditions={
                EmotionDimension.VALENCE: (-1.0, -0.3),      # 부정적
                EmotionDimension.AROUSAL: (0.0, 0.4)         # 낮은 각성
            },
            ethics_output={
                EthicsCategory.CARE: 1.5,
                EthicsCategory.VIRTUE: 1.2,
                EthicsCategory.RULE_BASED: 0.8,
                EthicsCategory.CONSEQUENCE: 0.7
            },
            rule_name="sadness_care_enhancement",
            description="슬픔 주도 상태에서 배려와 위로 우선"
        ))
        
        # 규칙 2: 높은 분노 + 높은 각성 → 정의 윤리 강화
        self.fuzzy_rules.append(FuzzyRule(
            emotion_conditions={
                EmotionDimension.VALENCE: (-1.0, -0.2),      # 부정적
                EmotionDimension.AROUSAL: (0.6, 1.0),        # 높은 각성
                EmotionDimension.DOMINANCE: (0.4, 1.0)       # 높은 지배감
            },
            ethics_output={
                EthicsCategory.JUSTICE: 1.6,
                EthicsCategory.RULE_BASED: 1.3,
                EthicsCategory.CARE: 0.6,
                EthicsCategory.CONSEQUENCE: 1.1
            },
            rule_name="anger_justice_enhancement",
            description="분노 주도 상태에서 정의와 공정성 우선"
        ))
        
        # 규칙 3: 높은 기쁨 + 높은 각성 → 공동체/협력 우선
        self.fuzzy_rules.append(FuzzyRule(
            emotion_conditions={
                EmotionDimension.VALENCE: (0.3, 1.0),        # 긍정적
                EmotionDimension.AROUSAL: (0.5, 1.0)         # 높은 각성
            },
            ethics_output={
                EthicsCategory.CARE: 1.4,
                EthicsCategory.VIRTUE: 1.3,
                EthicsCategory.CONSEQUENCE: 1.2,
                EthicsCategory.AUTONOMY: 1.1
            },
            rule_name="joy_cooperation_enhancement",
            description="기쁨 주도 상태에서 공동체 협력 우선"
        ))
        
        # 규칙 4: 높은 평온 + 낮은 각성 → 이성적 숙고 우선
        self.fuzzy_rules.append(FuzzyRule(
            emotion_conditions={
                EmotionDimension.VALENCE: (0.1, 1.0),        # 약간 긍정적
                EmotionDimension.AROUSAL: (0.0, 0.3),        # 낮은 각성
                EmotionDimension.CERTAINTY: (0.4, 1.0)       # 높은 확실성
            },
            ethics_output={
                EthicsCategory.RULE_BASED: 1.5,
                EthicsCategory.VIRTUE: 1.3,
                EthicsCategory.JUSTICE: 1.2,
                EthicsCategory.CARE: 1.0
            },
            rule_name="calm_rational_enhancement",
            description="평온 주도 상태에서 이성적 숙고 우선"
        ))
        
        # 규칙 5: 높은 불확실성 → 신중한 접근
        self.fuzzy_rules.append(FuzzyRule(
            emotion_conditions={
                EmotionDimension.CERTAINTY: (0.0, 0.4),      # 낮은 확실성
                EmotionDimension.SURPRISE: (0.5, 1.0)        # 높은 놀라움
            },
            ethics_output={
                EthicsCategory.RULE_BASED: 1.4,              # 규칙 따르기
                EthicsCategory.CARE: 1.3,                    # 신중한 배려
                EthicsCategory.CONSEQUENCE: 0.7,             # 결과 예측 어려움
                EthicsCategory.VIRTUE: 1.1
            },
            rule_name="uncertainty_caution_enhancement",
            description="불확실 상황에서 신중한 규칙 기반 접근"
        ))
        
        # 규칙 6: 높은 기대감 → 미래 지향적 판단
        self.fuzzy_rules.append(FuzzyRule(
            emotion_conditions={
                EmotionDimension.ANTICIPATION: (0.6, 1.0),   # 높은 기대감
                EmotionDimension.VALENCE: (0.0, 1.0)         # 중성~긍정
            },
            ethics_output={
                EthicsCategory.CONSEQUENCE: 1.5,             # 결과 중시
                EthicsCategory.AUTONOMY: 1.3,                # 자율성 존중
                EthicsCategory.VIRTUE: 1.1,
                EthicsCategory.CARE: 1.0
            },
            rule_name="anticipation_future_oriented",
            description="기대감 높은 상태에서 미래 결과 중시"
        ))
    
    def _initialize_connection_matrix(self) -> torch.Tensor:
        """감정-윤리 연결 매트릭스 초기화"""
        num_emotions = len(EmotionDimension)
        num_ethics = len(EthicsCategory)
        
        # 랜덤 초기화 후 정규화
        matrix = torch.randn(num_emotions, num_ethics) * 0.1 + 0.5
        matrix = torch.clamp(matrix, 0.0, 2.0)
        
        return nn.Parameter(matrix)
    
    def map_emotion_to_ethics(self, emotion_vector: torch.Tensor) -> Dict[str, Any]:
        """감정 벡터를 윤리 판단 가중치로 매핑"""
        
        # 입력 검증 및 변환
        if emotion_vector.dim() == 1:
            emotion_vector = emotion_vector.unsqueeze(0)
        
        batch_size = emotion_vector.shape[0]
        emotion_dim = emotion_vector.shape[1]
        
        # 감정 상태 딕셔너리 생성
        emotion_states = []
        for i in range(batch_size):
            emotion_state = {}
            for j, emotion_dim_enum in enumerate(EmotionDimension):
                if j < emotion_dim:
                    emotion_state[emotion_dim_enum] = float(emotion_vector[i, j])
            emotion_states.append(emotion_state)
        
        # 퍼지 규칙 적용
        batch_ethics_weights = []
        batch_rule_activations = []
        
        for emotion_state in emotion_states:
            ethics_weights, rule_activations = self._apply_fuzzy_rules(emotion_state)
            batch_ethics_weights.append(ethics_weights)
            batch_rule_activations.append(rule_activations)
        
        # 연결 매트릭스 적용 (학습 가능한 매핑)
        matrix_weights = self._apply_connection_matrix(emotion_vector)
        
        # 퍼지 규칙과 매트릭스 결과 통합
        final_weights = self._integrate_mappings(batch_ethics_weights, matrix_weights)
        
        return {
            'ethics_weights': final_weights,
            'rule_activations': batch_rule_activations,
            'matrix_weights': matrix_weights,
            'emotion_states': emotion_states
        }
    
    def _apply_fuzzy_rules(self, emotion_state: Dict[EmotionDimension, float]) -> Tuple[Dict[EthicsCategory, float], Dict[str, float]]:
        """퍼지 규칙 적용"""
        
        # 각 규칙의 활성화 정도 계산
        rule_activations = {}
        aggregated_ethics = defaultdict(float)
        total_activation = 0.0
        
        for i, rule in enumerate(self.fuzzy_rules):
            activation = rule.evaluate(emotion_state)
            rule_weight = float(self.rule_weights[i])
            weighted_activation = activation * rule_weight
            
            rule_activations[rule.rule_name] = activation
            total_activation += weighted_activation
            
            # 규칙의 윤리 출력을 가중 평균에 기여
            for ethics_cat, weight in rule.ethics_output.items():
                aggregated_ethics[ethics_cat] += weight * weighted_activation
        
        # 정규화
        if total_activation > 0:
            for ethics_cat in aggregated_ethics:
                aggregated_ethics[ethics_cat] /= total_activation
        else:
            # 기본값 설정
            for ethics_cat in EthicsCategory:
                aggregated_ethics[ethics_cat] = 1.0
        
        return dict(aggregated_ethics), rule_activations
    
    def _apply_connection_matrix(self, emotion_vector: torch.Tensor) -> torch.Tensor:
        """연결 매트릭스 적용"""
        # emotion_vector: [batch_size, emotion_dim]
        # connection_matrix: [emotion_dim, ethics_dim]
        
        # 매트릭스 곱셈으로 직접 매핑
        matrix_output = torch.matmul(emotion_vector, self.connection_matrix)
        
        # Sigmoid 활성화로 0~2 범위 매핑
        matrix_output = torch.sigmoid(matrix_output) * 2.0
        
        return matrix_output
    
    def _integrate_mappings(self, fuzzy_weights: List[Dict[EthicsCategory, float]], 
                          matrix_weights: torch.Tensor) -> torch.Tensor:
        """퍼지 규칙과 매트릭스 결과 통합"""
        
        batch_size = len(fuzzy_weights)
        num_ethics = len(EthicsCategory)
        
        # 퍼지 규칙 결과를 텐서로 변환
        fuzzy_tensor = torch.zeros(batch_size, num_ethics)
        
        for i, weights_dict in enumerate(fuzzy_weights):
            for j, ethics_cat in enumerate(EthicsCategory):
                fuzzy_tensor[i, j] = weights_dict.get(ethics_cat, 1.0)
        
        # 가중 평균 (퍼지 규칙: 0.7, 매트릭스: 0.3)
        integrated = fuzzy_tensor * 0.7 + matrix_weights * 0.3
        
        # 최종 정규화 (0.1 ~ 2.0 범위)
        integrated = torch.clamp(integrated, 0.1, 2.0)
        
        return integrated
    
    def adaptive_rule_learning(self, emotion_vector: torch.Tensor, 
                             ethics_feedback: torch.Tensor,
                             learning_rate: float = 0.01):
        """적응적 규칙 학습"""
        
        # 현재 매핑 결과 획득
        mapping_result = self.map_emotion_to_ethics(emotion_vector)
        predicted_weights = mapping_result['ethics_weights']
        
        # 손실 계산 (예측 vs 피드백)
        loss = F.mse_loss(predicted_weights, ethics_feedback)
        
        # 규칙 가중치 업데이트
        loss.backward(retain_graph=True)
        
        with torch.no_grad():
            self.rule_weights.data -= learning_rate * self.rule_weights.grad
            self.rule_weights.data = torch.clamp(self.rule_weights.data, 0.1, 2.0)
            self.rule_weights.grad.zero_()
            
            # 연결 매트릭스 업데이트
            if self.connection_matrix.grad is not None:
                self.connection_matrix.data -= learning_rate * self.connection_matrix.grad
                self.connection_matrix.data = torch.clamp(self.connection_matrix.data, 0.0, 2.0)
                self.connection_matrix.grad.zero_()
        
        self.logger.debug(f"규칙 학습 완료, 손실: {loss.item():.4f}")
        
        return loss.item()
    
    def get_dominant_emotion_pattern(self, emotion_vector: torch.Tensor) -> Dict[str, Any]:
        """지배적 감정 패턴 분석"""
        
        if emotion_vector.dim() == 1:
            emotion_vector = emotion_vector.unsqueeze(0)
        
        # 감정 차원별 강도 분석
        emotion_intensities = {}
        for i, emotion_dim in enumerate(EmotionDimension):
            if i < emotion_vector.shape[1]:
                intensity = float(emotion_vector[0, i])
                emotion_intensities[emotion_dim.value] = intensity
        
        # 지배적 감정 식별
        dominant_emotion = max(emotion_intensities.items(), key=lambda x: abs(x[1]))
        
        # 감정 조합 패턴 분석
        patterns = self._analyze_emotion_combinations(emotion_intensities)
        
        return {
            'dominant_emotion': dominant_emotion[0],
            'dominant_intensity': dominant_emotion[1],
            'emotion_intensities': emotion_intensities,
            'combination_patterns': patterns,
            'complexity_score': self._calculate_emotional_complexity(emotion_intensities)
        }
    
    def _analyze_emotion_combinations(self, emotion_intensities: Dict[str, float]) -> List[str]:
        """감정 조합 패턴 분석"""
        patterns = []
        
        valence = emotion_intensities.get('valence', 0.0)
        arousal = emotion_intensities.get('arousal', 0.0)
        dominance = emotion_intensities.get('dominance', 0.0)
        
        # 기본 조합 패턴들
        if valence > 0.5 and arousal > 0.5:
            patterns.append("excited_positive")
        elif valence > 0.5 and arousal < 0.3:
            patterns.append("calm_positive")
        elif valence < -0.5 and arousal > 0.5:
            patterns.append("agitated_negative")
        elif valence < -0.5 and arousal < 0.3:
            patterns.append("depressed_withdrawn")
        
        # 지배감 관련 패턴
        if dominance > 0.7:
            patterns.append("assertive_controlling")
        elif dominance < 0.3:
            patterns.append("submissive_powerless")
        
        # 복잡한 패턴들
        if abs(valence) < 0.3 and arousal > 0.6:
            patterns.append("neutral_aroused_alert")
        
        return patterns
    
    def _calculate_emotional_complexity(self, emotion_intensities: Dict[str, float]) -> float:
        """감정 복잡성 점수 계산"""
        
        # 활성화된 감정 차원 수
        active_dimensions = sum(1 for intensity in emotion_intensities.values() if abs(intensity) > 0.3)
        
        # 감정 강도의 분산
        intensities = list(emotion_intensities.values())
        variance = np.var(intensities) if intensities else 0.0
        
        # 복잡성 점수 (0~1)
        complexity = (active_dimensions / len(emotion_intensities)) * 0.7 + min(variance, 1.0) * 0.3
        
        return float(complexity)
    
    def explain_ethics_mapping(self, emotion_vector: torch.Tensor) -> str:
        """윤리 매핑 설명 생성"""
        
        mapping_result = self.map_emotion_to_ethics(emotion_vector)
        emotion_pattern = self.get_dominant_emotion_pattern(emotion_vector)
        
        # 활성화된 규칙들
        active_rules = []
        for rule_name, activation in mapping_result['rule_activations'][0].items():
            if activation > 0.3:
                rule = next(r for r in self.fuzzy_rules if r.rule_name == rule_name)
                active_rules.append((rule, activation))
        
        # 설명 텍스트 생성
        explanation = f"주요 감정: {emotion_pattern['dominant_emotion']} (강도: {emotion_pattern['dominant_intensity']:.2f})\n"
        explanation += f"감정 복잡성: {emotion_pattern['complexity_score']:.2f}\n\n"
        
        explanation += "활성화된 퍼지 규칙들:\n"
        for rule, activation in active_rules:
            explanation += f"- {rule.description} (활성화: {activation:.2f})\n"
        
        # 윤리 가중치 결과
        ethics_weights = mapping_result['ethics_weights'][0]
        explanation += "\n윤리 판단 가중치:\n"
        for i, ethics_cat in enumerate(EthicsCategory):
            weight = float(ethics_weights[i])
            explanation += f"- {ethics_cat.value}: {weight:.2f}\n"
        
        return explanation
    
    def save_fuzzy_system(self, path: Path):
        """퍼지 시스템 저장"""
        
        # 규칙들을 직렬화 가능한 형태로 변환
        rules_data = []
        for rule in self.fuzzy_rules:
            rule_data = {
                'emotion_conditions': {
                    dim.value: condition for dim, condition in rule.emotion_conditions.items()
                },
                'ethics_output': {
                    cat.value: weight for cat, weight in rule.ethics_output.items()
                },
                'rule_name': rule.rule_name,
                'confidence': rule.confidence,
                'description': rule.description
            }
            rules_data.append(rule_data)
        
        save_data = {
            'fuzzy_rules': rules_data,
            'rule_weights': self.rule_weights.detach().cpu().numpy().tolist(),
            'connection_matrix': self.connection_matrix.detach().cpu().numpy().tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"퍼지 시스템 저장: {path}")
    
    def load_fuzzy_system(self, path: Path):
        """퍼지 시스템 로드"""
        
        with open(path, 'r', encoding='utf-8') as f:
            save_data = json.load(f)
        
        # 규칙 가중치 복원
        self.rule_weights.data = torch.tensor(save_data['rule_weights'])
        
        # 연결 매트릭스 복원
        self.connection_matrix.data = torch.tensor(save_data['connection_matrix'])
        
        self.logger.info(f"퍼지 시스템 로드: {path}")

def create_fuzzy_emotion_ethics_mapper() -> FuzzyEmotionEthicsMapper:
    """퍼지 감정-윤리 매퍼 생성 헬퍼 함수"""
    return FuzzyEmotionEthicsMapper()

# 사용 예시
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 퍼지 매퍼 생성
    mapper = create_fuzzy_emotion_ethics_mapper()
    
    # 테스트 감정 벡터들
    test_emotions = [
        torch.tensor([0.2, 0.7, 0.3, 0.6, 0.1, 0.4]),  # 행복하지만 불안
        torch.tensor([-0.7, 0.2, 0.1, 0.8, 0.3, 0.2]), # 슬프고 차분함
        torch.tensor([-0.5, 0.8, 0.9, 0.3, 0.7, 0.1]), # 분노와 지배감
        torch.tensor([0.8, 0.3, 0.7, 0.9, 0.1, 0.8])   # 매우 긍정적이고 확신
    ]
    
    for i, emotion_vec in enumerate(test_emotions):
        print(f"\n=== 테스트 {i+1} ===")
        
        # 윤리 매핑 수행
        mapping_result = mapper.map_emotion_to_ethics(emotion_vec)
        
        # 감정 패턴 분석
        pattern = mapper.get_dominant_emotion_pattern(emotion_vec)
        
        print(f"지배적 감정: {pattern['dominant_emotion']}")
        print(f"감정 복잡성: {pattern['complexity_score']:.2f}")
        print(f"윤리 가중치: {mapping_result['ethics_weights'][0]}")
        
        # 설명 생성
        explanation = mapper.explain_ethics_mapping(emotion_vec)
        print(f"\n설명:\n{explanation}")
    
    # 적응적 학습 테스트
    print("\n=== 적응적 학습 테스트 ===")
    test_emotion = torch.tensor([[0.0, 0.5, 0.5, 0.7, 0.2, 0.3]])
    target_ethics = torch.tensor([[1.5, 1.2, 0.8, 1.3, 1.0, 0.9]])  # 목표 윤리 가중치
    
    initial_loss = mapper.adaptive_rule_learning(test_emotion, target_ethics, 0.01)
    print(f"초기 손실: {initial_loss:.4f}")
    
    # 몇 번 더 학습
    for epoch in range(5):
        loss = mapper.adaptive_rule_learning(test_emotion, target_ethics, 0.01)
        print(f"Epoch {epoch+1} 손실: {loss:.4f}")
    
    # 학습 후 결과 확인
    final_result = mapper.map_emotion_to_ethics(test_emotion)
    print(f"학습 후 윤리 가중치: {final_result['ethics_weights'][0]}")
    print(f"목표 윤리 가중치: {target_ethics[0]}")