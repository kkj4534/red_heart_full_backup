"""
퍼지 로직 감정-윤리 매핑 (Fuzzy Logic Emotion-Ethics Mapper)
Fuzzy Logic Emotion-Ethics Mapping Module

감정 전환의 자연스러운 처리와 연속적인 윤리 판단을 위한 퍼지 로직 기반 
매핑 시스템을 구현하여 인간적이고 직관적인 의사결정을 지원합니다.

핵심 기능:
1. 감정 임계점의 경계 모호성 해결
2. 연속적 감정-윤리 매핑
3. 적응적 퍼지 멤버십 함수
4. 언어적 변수 기반 감정 표현
"""

import numpy as np
import torch
import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import math

try:
    from config import ADVANCED_CONFIG, DEVICE
except ImportError:
    # config.py에 문제가 있을 경우 기본값 사용
    ADVANCED_CONFIG = {'enable_gpu': False}
    DEVICE = 'cpu'
    print("⚠️  config.py 임포트 실패, 기본값 사용")
from data_models import EmotionData

logger = logging.getLogger('FuzzyEmotionEthicsMapper')

class FuzzySet:
    """퍼지 집합 클래스"""
    
    def __init__(self, name: str, membership_function: Callable[[float], float]):
        self.name = name
        self.membership_function = membership_function
    
    def membership(self, x: float) -> float:
        """멤버십 값 계산"""
        return np.clip(self.membership_function(x), 0.0, 1.0)

class LinguisticVariable:
    """언어적 변수 클래스"""
    
    def __init__(self, name: str, universe: Tuple[float, float], fuzzy_sets: List[FuzzySet]):
        self.name = name
        self.universe = universe  # (min, max)
        self.fuzzy_sets = {fs.name: fs for fs in fuzzy_sets}
    
    def fuzzify(self, value: float) -> Dict[str, float]:
        """값을 퍼지화"""
        return {name: fs.membership(value) for name, fs in self.fuzzy_sets.items()}
    
    def defuzzify(self, fuzzy_values: Dict[str, float], method: str = 'centroid') -> float:
        """역퍼지화"""
        if method == 'centroid':
            return self._centroid_defuzzify(fuzzy_values)
        elif method == 'max_membership':
            return self._max_membership_defuzzify(fuzzy_values)
        else:
            raise ValueError(f"Unknown defuzzification method: {method}")
    
    def _centroid_defuzzify(self, fuzzy_values: Dict[str, float]) -> float:
        """중심점 방법으로 역퍼지화"""
        # 각 퍼지 집합의 중심점 추정
        centroids = {
            'very_low': self.universe[0] + 0.1 * (self.universe[1] - self.universe[0]),
            'low': self.universe[0] + 0.3 * (self.universe[1] - self.universe[0]),
            'medium': self.universe[0] + 0.5 * (self.universe[1] - self.universe[0]),
            'high': self.universe[0] + 0.7 * (self.universe[1] - self.universe[0]),
            'very_high': self.universe[0] + 0.9 * (self.universe[1] - self.universe[0])
        }
        
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for set_name, membership in fuzzy_values.items():
            if set_name in centroids and membership > 0:
                weighted_sum += centroids[set_name] * membership
                weight_sum += membership
        
        return weighted_sum / weight_sum if weight_sum > 0 else 0.5

    def _max_membership_defuzzify(self, fuzzy_values: Dict[str, float]) -> float:
        """최대 멤버십 방법으로 역퍼지화"""
        max_set = max(fuzzy_values.keys(), key=lambda k: fuzzy_values[k])
        
        # 최대 멤버십 집합의 대표값 반환
        set_representatives = {
            'very_low': 0.1,
            'low': 0.3,
            'medium': 0.5,
            'high': 0.7,
            'very_high': 0.9
        }
        
        return set_representatives.get(max_set, 0.5)

@dataclass
class FuzzyEmotionState:
    """퍼지 감정 상태"""
    linguistic_representation: Dict[str, float] = field(default_factory=dict)
    numerical_values: Dict[str, float] = field(default_factory=dict)
    certainty_level: float = 0.7
    transition_momentum: float = 0.0

@dataclass
class FuzzyEthicsMapping:
    """퍼지 윤리 매핑 결과"""
    ethics_weights: Dict[str, float] = field(default_factory=dict)
    confidence_levels: Dict[str, float] = field(default_factory=dict)
    linguistic_explanations: List[str] = field(default_factory=list)
    mapping_quality: float = 0.0

class FuzzyEmotionEthicsMapper:
    """퍼지 로직 감정-윤리 매핑 시스템"""
    
    def __init__(self):
        self.logger = logger
        
        # 언어적 변수 정의
        self.linguistic_variables = self._initialize_linguistic_variables()
        
        # 퍼지 규칙 베이스
        self.fuzzy_rules = self._initialize_fuzzy_rules()
        
        # 감정 전환 추적
        self.emotion_history = []
        self.transition_patterns = {}
        
        # 적응적 파라미터
        self.adaptation_memory = {
            'successful_mappings': [],
            'failed_mappings': [],
            'user_feedback_history': []
        }
        
        # 매핑 통계
        self.mapping_statistics = {
            'total_mappings': 0,
            'successful_mappings': 0,
            'average_confidence': 0.0,
            'average_quality': 0.0
        }
        
        self.logger.info("퍼지 로직 감정-윤리 매핑 시스템 초기화 완료")
    
    def _initialize_linguistic_variables(self) -> Dict[str, LinguisticVariable]:
        """언어적 변수 초기화"""
        
        variables = {}
        
        # 감정 강도 변수
        emotion_intensity_sets = [
            FuzzySet('barely_felt', self._triangular_mf(0.0, 0.0, 0.3)),
            FuzzySet('noticeable', self._triangular_mf(0.1, 0.3, 0.5)),
            FuzzySet('moderate', self._triangular_mf(0.3, 0.5, 0.7)),
            FuzzySet('strong', self._triangular_mf(0.5, 0.7, 0.9)),
            FuzzySet('overwhelming', self._triangular_mf(0.7, 1.0, 1.0))
        ]
        variables['emotion_intensity'] = LinguisticVariable(
            'emotion_intensity', (0.0, 1.0), emotion_intensity_sets
        )
        
        # 윤리적 중요도 변수
        ethics_importance_sets = [
            FuzzySet('irrelevant', self._triangular_mf(0.0, 0.0, 0.25)),
            FuzzySet('somewhat_important', self._triangular_mf(0.1, 0.35, 0.6)),
            FuzzySet('important', self._triangular_mf(0.4, 0.65, 0.85)),
            FuzzySet('very_important', self._triangular_mf(0.7, 0.9, 1.0)),
            FuzzySet('critical', self._triangular_mf(0.85, 1.0, 1.0))
        ]
        variables['ethics_importance'] = LinguisticVariable(
            'ethics_importance', (0.0, 1.0), ethics_importance_sets
        )
        
        # 확신도 변수
        confidence_sets = [
            FuzzySet('very_uncertain', self._triangular_mf(0.0, 0.0, 0.3)),
            FuzzySet('uncertain', self._triangular_mf(0.1, 0.3, 0.5)),
            FuzzySet('moderate_confidence', self._triangular_mf(0.3, 0.5, 0.7)),
            FuzzySet('confident', self._triangular_mf(0.5, 0.7, 0.9)),
            FuzzySet('very_confident', self._triangular_mf(0.7, 1.0, 1.0))
        ]
        variables['confidence'] = LinguisticVariable(
            'confidence', (0.0, 1.0), confidence_sets
        )
        
        return variables
    
    def _triangular_mf(self, a: float, b: float, c: float) -> Callable[[float], float]:
        """삼각형 멤버십 함수 생성"""
        def membership(x: float) -> float:
            if x <= a or x >= c:
                return 0.0
            elif a < x <= b:
                return (x - a) / (b - a) if b != a else 1.0
            else:  # b < x < c
                return (c - x) / (c - b) if c != b else 1.0
        
        return membership
    
    def _gaussian_mf(self, mean: float, std: float) -> Callable[[float], float]:
        """가우시안 멤버십 함수 생성"""
        def membership(x: float) -> float:
            return math.exp(-0.5 * ((x - mean) / std) ** 2)
        
        return membership
    
    def _initialize_fuzzy_rules(self) -> List[Dict[str, Any]]:
        """퍼지 규칙 베이스 초기화"""
        
        rules = [
            # 기쁨 관련 규칙
            {
                'id': 'joy_care',
                'condition': 'IF emotion_intensity IS strong AND emotion_type IS joy',
                'conclusion': 'THEN care_harm IS very_important',
                'confidence': 0.8,
                'linguistic': "기쁠 때는 타인에 대한 돌봄이 매우 중요해집니다"
            },
            {
                'id': 'joy_fairness',
                'condition': 'IF emotion_intensity IS moderate AND emotion_type IS joy',
                'conclusion': 'THEN fairness IS important',
                'confidence': 0.7,
                'linguistic': "적당히 기쁠 때는 공정성을 중시하게 됩니다"
            },
            
            # 슬픔 관련 규칙
            {
                'id': 'sadness_care',
                'condition': 'IF emotion_intensity IS strong AND emotion_type IS sadness',
                'conclusion': 'THEN care_harm IS critical',
                'confidence': 0.9,
                'linguistic': "슬플 때는 돌봄과 해악 방지가 극도로 중요해집니다"
            },
            {
                'id': 'sadness_authority',
                'condition': 'IF emotion_intensity IS moderate AND emotion_type IS sadness',
                'conclusion': 'THEN authority IS somewhat_important',
                'confidence': 0.5,
                'linguistic': "슬플 때는 권위에 대한 관심이 낮아집니다"
            },
            
            # 분노 관련 규칙
            {
                'id': 'anger_fairness',
                'condition': 'IF emotion_intensity IS strong AND emotion_type IS anger',
                'conclusion': 'THEN fairness IS critical',
                'confidence': 0.9,
                'linguistic': "화날 때는 공정성이 극도로 중요해집니다"
            },
            {
                'id': 'anger_loyalty',
                'condition': 'IF emotion_intensity IS moderate AND emotion_type IS anger',
                'conclusion': 'THEN loyalty IS important',
                'confidence': 0.7,
                'linguistic': "화날 때는 충성심을 중시하게 됩니다"
            },
            
            # 두려움 관련 규칙
            {
                'id': 'fear_authority',
                'condition': 'IF emotion_intensity IS strong AND emotion_type IS fear',
                'conclusion': 'THEN authority IS very_important',
                'confidence': 0.8,
                'linguistic': "두려울 때는 권위와 질서를 중시하게 됩니다"
            },
            {
                'id': 'fear_care',
                'condition': 'IF emotion_intensity IS overwhelming AND emotion_type IS fear',
                'conclusion': 'THEN care_harm IS critical',
                'confidence': 0.9,
                'linguistic': "극도로 두려울 때는 안전과 보호가 최우선이 됩니다"
            },
            
            # 사랑 관련 규칙
            {
                'id': 'love_care',
                'condition': 'IF emotion_intensity IS strong AND emotion_type IS love',
                'conclusion': 'THEN care_harm IS critical',
                'confidence': 0.9,
                'linguistic': "사랑할 때는 돌봄이 가장 중요해집니다"
            },
            {
                'id': 'love_loyalty',
                'condition': 'IF emotion_intensity IS moderate AND emotion_type IS love',
                'conclusion': 'THEN loyalty IS very_important',
                'confidence': 0.8,
                'linguistic': "사랑할 때는 충성심이 매우 중요해집니다"
            },
            
            # 복합 감정 규칙
            {
                'id': 'mixed_balanced',
                'condition': 'IF emotion_intensity IS moderate AND emotion_certainty IS low',
                'conclusion': 'THEN ALL_ethics IS important',
                'confidence': 0.6,
                'linguistic': "감정이 복잡할 때는 모든 윤리적 가치를 균형있게 고려해야 합니다"
            }
        ]
        
        return rules
    
    def map_emotion_to_ethics(
        self,
        emotion_data: EmotionData,
        context: Dict[str, Any] = None
    ) -> FuzzyEthicsMapping:
        """감정을 윤리적 가중치로 매핑"""
        
        start_time = time.time()
        
        # 1단계: 감정 퍼지화
        fuzzy_emotion = self._fuzzify_emotion(emotion_data)
        
        # 2단계: 퍼지 규칙 적용
        activated_rules = self._apply_fuzzy_rules(fuzzy_emotion, emotion_data, context)
        
        # 3단계: 윤리 가중치 추론
        ethics_weights = self._infer_ethics_weights(activated_rules)
        
        # 4단계: 신뢰도 계산
        confidence_levels = self._calculate_confidence_levels(activated_rules, fuzzy_emotion)
        
        # 5단계: 언어적 설명 생성
        linguistic_explanations = self._generate_linguistic_explanations(activated_rules)
        
        # 6단계: 매핑 품질 평가
        mapping_quality = self._evaluate_mapping_quality(
            fuzzy_emotion, activated_rules, ethics_weights, confidence_levels
        )
        
        # 결과 생성
        mapping_result = FuzzyEthicsMapping(
            ethics_weights=ethics_weights,
            confidence_levels=confidence_levels,
            linguistic_explanations=linguistic_explanations,
            mapping_quality=mapping_quality
        )
        
        # 감정 히스토리 업데이트
        self._update_emotion_history(emotion_data, mapping_result)
        
        # 통계 업데이트
        self._update_mapping_statistics(mapping_result)
        
        processing_time = time.time() - start_time
        
        self.logger.debug(
            f"퍼지 매핑 완료: 품질 {mapping_quality:.3f}, "
            f"활성 규칙 {len(activated_rules)}개, 처리시간 {processing_time:.3f}초"
        )
        
        return mapping_result
    
    def _fuzzify_emotion(self, emotion_data: EmotionData) -> FuzzyEmotionState:
        """감정 데이터를 퍼지화"""
        
        # 감정 강도 계산 (VAD 벡터의 크기)
        emotion_magnitude = math.sqrt(
            emotion_data.valence**2 + 
            emotion_data.arousal**2 + 
            emotion_data.dominance**2
        ) / math.sqrt(3)  # 정규화
        
        # 언어적 표현으로 변환
        intensity_var = self.linguistic_variables['emotion_intensity']
        linguistic_intensity = intensity_var.fuzzify(emotion_magnitude)
        
        # 확신도 퍼지화
        confidence_var = self.linguistic_variables['confidence']
        linguistic_confidence = confidence_var.fuzzify(emotion_data.confidence)
        
        # 수치적 값들
        numerical_values = {
            'valence': emotion_data.valence,
            'arousal': emotion_data.arousal,
            'dominance': emotion_data.dominance,
            'magnitude': emotion_magnitude,
            'confidence': emotion_data.confidence
        }
        
        # 언어적 표현 통합
        linguistic_representation = {
            **{f"intensity_{k}": v for k, v in linguistic_intensity.items()},
            **{f"confidence_{k}": v for k, v in linguistic_confidence.items()}
        }
        
        return FuzzyEmotionState(
            linguistic_representation=linguistic_representation,
            numerical_values=numerical_values,
            certainty_level=emotion_data.confidence
        )
    
    def _apply_fuzzy_rules(
        self,
        fuzzy_emotion: FuzzyEmotionState,
        emotion_data: EmotionData,
        context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """퍼지 규칙 적용"""
        
        activated_rules = []
        
        # 주요 감정 타입 결정
        emotion_type = self._determine_emotion_type(emotion_data)
        
        for rule in self.fuzzy_rules:
            # 규칙 조건 평가
            activation_level = self._evaluate_rule_condition(
                rule, fuzzy_emotion, emotion_type, context
            )
            
            if activation_level > 0.1:  # 최소 활성화 임계값
                activated_rule = {
                    'rule': rule,
                    'activation_level': activation_level,
                    'weighted_confidence': rule['confidence'] * activation_level
                }
                activated_rules.append(activated_rule)
        
        # 활성화 수준으로 정렬
        activated_rules.sort(key=lambda r: r['activation_level'], reverse=True)
        
        return activated_rules
    
    def _determine_emotion_type(self, emotion_data: EmotionData) -> str:
        """VAD 값을 기반으로 감정 타입 결정"""
        
        valence = emotion_data.valence
        arousal = emotion_data.arousal
        dominance = emotion_data.dominance
        
        # 간단한 휴리스틱 기반 감정 분류
        if valence > 0.5:
            if arousal > 0.5:
                return 'joy' if dominance > 0.5 else 'excitement'
            else:
                return 'contentment' if dominance > 0.5 else 'love'
        elif valence < -0.3:
            if arousal > 0.5:
                return 'anger' if dominance > 0.5 else 'fear'
            else:
                return 'sadness' if dominance < 0.5 else 'disgust'
        else:
            if arousal > 0.7:
                return 'surprise'
            else:
                return 'neutral'
    
    def _evaluate_rule_condition(
        self,
        rule: Dict[str, Any],
        fuzzy_emotion: FuzzyEmotionState,
        emotion_type: str,
        context: Dict[str, Any] = None
    ) -> float:
        """규칙 조건 평가"""
        
        activation = 0.0
        
        # 규칙 ID 기반 평가 (간단한 버전)
        rule_id = rule['id']
        
        if rule_id.startswith(emotion_type):
            # 감정 타입이 일치하는 경우
            intensity_activation = 0.0
            
            # 강도 조건 확인
            if 'strong' in rule['condition']:
                intensity_activation = fuzzy_emotion.linguistic_representation.get('intensity_strong', 0.0)
            elif 'moderate' in rule['condition']:
                intensity_activation = fuzzy_emotion.linguistic_representation.get('intensity_moderate', 0.0)
            elif 'overwhelming' in rule['condition']:
                intensity_activation = fuzzy_emotion.linguistic_representation.get('intensity_overwhelming', 0.0)
            else:
                # 기본적으로 모든 강도 고려
                intensity_activation = max(
                    fuzzy_emotion.linguistic_representation.get('intensity_moderate', 0.0),
                    fuzzy_emotion.linguistic_representation.get('intensity_strong', 0.0),
                    fuzzy_emotion.linguistic_representation.get('intensity_overwhelming', 0.0)
                )
            
            activation = intensity_activation
            
        # 복합 감정 규칙 처리
        elif rule_id == 'mixed_balanced':
            # 감정 확실성이 낮을 때 활성화
            uncertainty_activation = fuzzy_emotion.linguistic_representation.get('confidence_uncertain', 0.0)
            moderate_intensity = fuzzy_emotion.linguistic_representation.get('intensity_moderate', 0.0)
            activation = min(uncertainty_activation, moderate_intensity)
        
        # 맥락 기반 조정
        if context and activation > 0:
            # 상황적 맥락이 규칙 활성화에 영향
            situational_factor = context.get('situational_intensity', 1.0)
            activation *= situational_factor
        
        return np.clip(activation, 0.0, 1.0)
    
    def _infer_ethics_weights(self, activated_rules: List[Dict[str, Any]]) -> Dict[str, float]:
        """활성화된 규칙들로부터 윤리 가중치 추론"""
        
        # 윤리 카테고리별 가중 합계
        ethics_weights = {
            'care_harm': 0.0,
            'fairness_cheating': 0.0,
            'loyalty_betrayal': 0.0,
            'authority_subversion': 0.0,
            'sanctity_degradation': 0.0
        }
        
        total_weights = {category: 0.0 for category in ethics_weights.keys()}
        
        for activated_rule in activated_rules:
            rule = activated_rule['rule']
            activation = activated_rule['activation_level']
            confidence = activated_rule['weighted_confidence']
            
            # 규칙 결론에서 윤리 카테고리와 중요도 추출
            conclusion = rule['conclusion']
            
            # 간단한 파싱 (실제로는 더 정교한 파서 필요)
            if 'care_harm' in conclusion:
                importance = self._extract_importance_level(conclusion)
                weight_contribution = importance * activation * confidence
                ethics_weights['care_harm'] += weight_contribution
                total_weights['care_harm'] += activation * confidence
                
            elif 'fairness' in conclusion:
                importance = self._extract_importance_level(conclusion)
                weight_contribution = importance * activation * confidence
                ethics_weights['fairness_cheating'] += weight_contribution
                total_weights['fairness_cheating'] += activation * confidence
                
            elif 'loyalty' in conclusion:
                importance = self._extract_importance_level(conclusion)
                weight_contribution = importance * activation * confidence
                ethics_weights['loyalty_betrayal'] += weight_contribution
                total_weights['loyalty_betrayal'] += activation * confidence
                
            elif 'authority' in conclusion:
                importance = self._extract_importance_level(conclusion)
                weight_contribution = importance * activation * confidence
                ethics_weights['authority_subversion'] += weight_contribution
                total_weights['authority_subversion'] += activation * confidence
                
            elif 'ALL_ethics' in conclusion:
                # 모든 윤리에 균등하게 적용
                importance = self._extract_importance_level(conclusion)
                weight_contribution = importance * activation * confidence / len(ethics_weights)
                for category in ethics_weights.keys():
                    ethics_weights[category] += weight_contribution
                    total_weights[category] += activation * confidence / len(ethics_weights)
        
        # 정규화
        for category in ethics_weights.keys():
            if total_weights[category] > 0:
                ethics_weights[category] /= total_weights[category]
            else:
                ethics_weights[category] = 0.5  # 기본값
            
            # 범위 제한
            ethics_weights[category] = np.clip(ethics_weights[category], 0.1, 1.0)
        
        return ethics_weights
    
    def _extract_importance_level(self, conclusion: str) -> float:
        """결론에서 중요도 수준 추출"""
        
        if 'critical' in conclusion:
            return 1.0
        elif 'very_important' in conclusion:
            return 0.9
        elif 'important' in conclusion:
            return 0.7
        elif 'somewhat_important' in conclusion:
            return 0.5
        elif 'irrelevant' in conclusion:
            return 0.1
        else:
            return 0.6  # 기본값
    
    def _calculate_confidence_levels(
        self,
        activated_rules: List[Dict[str, Any]],
        fuzzy_emotion: FuzzyEmotionState
    ) -> Dict[str, float]:
        """신뢰도 수준 계산"""
        
        confidence_levels = {}
        
        # 전체 신뢰도
        if activated_rules:
            overall_confidence = np.mean([rule['weighted_confidence'] for rule in activated_rules])
        else:
            overall_confidence = 0.5
        
        confidence_levels['overall'] = overall_confidence
        
        # 감정 기반 신뢰도
        emotion_confidence = fuzzy_emotion.certainty_level
        confidence_levels['emotion_based'] = emotion_confidence
        
        # 규칙 활성화 기반 신뢰도
        if activated_rules:
            max_activation = max(rule['activation_level'] for rule in activated_rules)
            activation_confidence = max_activation
        else:
            activation_confidence = 0.0
        
        confidence_levels['activation_based'] = activation_confidence
        
        # 통합 신뢰도
        integrated_confidence = (
            overall_confidence * 0.4 +
            emotion_confidence * 0.3 +
            activation_confidence * 0.3
        )
        
        confidence_levels['integrated'] = integrated_confidence
        
        return confidence_levels
    
    def _generate_linguistic_explanations(self, activated_rules: List[Dict[str, Any]]) -> List[str]:
        """언어적 설명 생성"""
        
        explanations = []
        
        # 상위 3개 활성화된 규칙의 설명 포함
        top_rules = activated_rules[:3]
        
        for i, activated_rule in enumerate(top_rules):
            rule = activated_rule['rule']
            activation = activated_rule['activation_level']
            
            linguistic_desc = rule.get('linguistic', '규칙이 적용되었습니다')
            confidence_desc = f" (확신도: {activation:.2f})"
            
            explanations.append(f"{i+1}. {linguistic_desc}{confidence_desc}")
        
        # 전체 상황 요약
        if len(activated_rules) > 3:
            explanations.append(f"추가로 {len(activated_rules) - 3}개의 규칙이 더 적용되었습니다.")
        
        if not activated_rules:
            explanations.append("현재 감정 상태에서 명확한 윤리적 지침을 찾기 어렵습니다.")
        
        return explanations
    
    def _evaluate_mapping_quality(
        self,
        fuzzy_emotion: FuzzyEmotionState,
        activated_rules: List[Dict[str, Any]],
        ethics_weights: Dict[str, float],
        confidence_levels: Dict[str, float]
    ) -> float:
        """매핑 품질 평가"""
        
        quality_score = 0.0
        
        # 1. 규칙 활성화 품질 (30%)
        if activated_rules:
            rule_quality = np.mean([rule['activation_level'] for rule in activated_rules])
            quality_score += rule_quality * 0.3
        
        # 2. 신뢰도 품질 (25%)
        confidence_quality = confidence_levels.get('integrated', 0.5)
        quality_score += confidence_quality * 0.25
        
        # 3. 감정 명확성 품질 (25%)
        emotion_clarity = fuzzy_emotion.certainty_level
        quality_score += emotion_clarity * 0.25
        
        # 4. 윤리 가중치 균형성 (20%)
        weights_values = list(ethics_weights.values())
        if weights_values:
            # 과도한 편중 방지 (적당한 분산 선호)
            weights_std = np.std(weights_values)
            balance_score = 1.0 - min(weights_std / 0.5, 1.0)  # 표준편차가 0.5를 넘으면 페널티
            quality_score += balance_score * 0.2
        
        return np.clip(quality_score, 0.0, 1.0)
    
    def _update_emotion_history(self, emotion_data: EmotionData, mapping_result: FuzzyEthicsMapping):
        """감정 히스토리 업데이트"""
        
        history_entry = {
            'timestamp': time.time(),
            'emotion': {
                'valence': emotion_data.valence,
                'arousal': emotion_data.arousal,
                'dominance': emotion_data.dominance,
                'confidence': emotion_data.confidence
            },
            'mapping_quality': mapping_result.mapping_quality,
            'ethics_weights': mapping_result.ethics_weights.copy()
        }
        
        self.emotion_history.append(history_entry)
        
        # 히스토리 크기 제한
        if len(self.emotion_history) > 100:
            self.emotion_history = self.emotion_history[-100:]
    
    def _update_mapping_statistics(self, mapping_result: FuzzyEthicsMapping):
        """매핑 통계 업데이트"""
        
        self.mapping_statistics['total_mappings'] += 1
        
        if mapping_result.mapping_quality > 0.6:
            self.mapping_statistics['successful_mappings'] += 1
        
        # 평균 신뢰도 업데이트
        total_count = self.mapping_statistics['total_mappings']
        current_avg_conf = self.mapping_statistics['average_confidence']
        new_confidence = mapping_result.confidence_levels.get('integrated', 0.5)
        new_avg_conf = (current_avg_conf * (total_count - 1) + new_confidence) / total_count
        self.mapping_statistics['average_confidence'] = new_avg_conf
        
        # 평균 품질 업데이트
        current_avg_quality = self.mapping_statistics['average_quality']
        new_avg_quality = (current_avg_quality * (total_count - 1) + mapping_result.mapping_quality) / total_count
        self.mapping_statistics['average_quality'] = new_avg_quality
    
    def smooth_emotion_transition(
        self,
        current_emotion: EmotionData,
        target_emotion: EmotionData,
        transition_speed: float = 0.3
    ) -> EmotionData:
        """감정 전환 스무딩"""
        
        # 퍼지 멤버십 기반 부드러운 전환
        smoothed_valence = self._fuzzy_smooth_transition(
            current_emotion.valence, target_emotion.valence, transition_speed
        )
        
        smoothed_arousal = self._fuzzy_smooth_transition(
            current_emotion.arousal, target_emotion.arousal, transition_speed
        )
        
        smoothed_dominance = self._fuzzy_smooth_transition(
            current_emotion.dominance, target_emotion.dominance, transition_speed
        )
        
        # 신뢰도는 더 보수적으로 변화
        smoothed_confidence = self._fuzzy_smooth_transition(
            current_emotion.confidence, target_emotion.confidence, transition_speed * 0.5
        )
        
        return EmotionData(
            valence=smoothed_valence,
            arousal=smoothed_arousal,
            dominance=smoothed_dominance,
            confidence=smoothed_confidence
        )
    
    def _fuzzy_smooth_transition(self, current: float, target: float, speed: float) -> float:
        """퍼지 멤버십 기반 부드러운 전환"""
        
        # 변화량 계산
        difference = target - current
        
        # 퍼지 멤버십으로 변화 강도 조절
        if abs(difference) < 0.1:
            # 작은 변화: 부드럽게
            transition_factor = speed * 0.5
        elif abs(difference) < 0.3:
            # 중간 변화: 보통 속도
            transition_factor = speed
        else:
            # 큰 변화: 빠르게 하지만 제한
            transition_factor = speed * 1.5
        
        # 경계 클램핑으로 자연스러운 제한
        change = difference * transition_factor
        new_value = current + change
        
        return np.clip(new_value, -1.0, 1.0)
    
    def get_mapping_analytics(self) -> Dict[str, Any]:
        """매핑 분석 정보 반환"""
        
        analytics = {
            'statistics': self.mapping_statistics.copy(),
            'emotion_history_length': len(self.emotion_history),
            'fuzzy_rules_count': len(self.fuzzy_rules),
            'linguistic_variables': list(self.linguistic_variables.keys())
        }
        
        # 최근 감정 트렌드
        if len(self.emotion_history) >= 5:
            recent_emotions = self.emotion_history[-5:]
            analytics['recent_emotion_trend'] = {
                'avg_valence': np.mean([e['emotion']['valence'] for e in recent_emotions]),
                'avg_arousal': np.mean([e['emotion']['arousal'] for e in recent_emotions]),
                'avg_dominance': np.mean([e['emotion']['dominance'] for e in recent_emotions]),
                'avg_quality': np.mean([e['mapping_quality'] for e in recent_emotions])
            }
        
        # 윤리 가중치 트렌드
        if len(self.emotion_history) >= 3:
            recent_mappings = self.emotion_history[-3:]
            ethics_trends = {}
            
            for category in ['care_harm', 'fairness_cheating', 'loyalty_betrayal', 
                           'authority_subversion', 'sanctity_degradation']:
                values = [mapping['ethics_weights'].get(category, 0.5) for mapping in recent_mappings]
                ethics_trends[category] = {
                    'mean': np.mean(values),
                    'trend': values[-1] - values[0] if len(values) > 1 else 0.0
                }
            
            analytics['ethics_trends'] = ethics_trends
        
        return analytics


# 테스트 및 데모 함수
def test_fuzzy_emotion_ethics_mapper():
    """퍼지 로직 감정-윤리 매핑 테스트"""
    print("🌟 퍼지 로직 감정-윤리 매핑 테스트 시작")
    
    # 매퍼 초기화
    mapper = FuzzyEmotionEthicsMapper()
    
    # 테스트 감정 데이터들
    test_emotions = [
        EmotionData(valence=0.8, arousal=0.7, dominance=0.6, confidence=0.9),  # 기쁨
        EmotionData(valence=-0.6, arousal=0.8, dominance=0.4, confidence=0.8),  # 분노
        EmotionData(valence=-0.7, arousal=0.3, dominance=0.2, confidence=0.7),  # 슬픔
        EmotionData(valence=-0.3, arousal=0.9, dominance=0.2, confidence=0.6),  # 두려움
        EmotionData(valence=0.6, arousal=0.2, dominance=0.8, confidence=0.8),   # 사랑
    ]
    
    emotion_names = ['기쁨', '분노', '슬픔', '두려움', '사랑']
    
    print(f"테스트 감정 수: {len(test_emotions)}개")
    
    # 각 감정에 대한 매핑 테스트
    for i, (emotion, name) in enumerate(zip(test_emotions, emotion_names)):
        print(f"\n--- {name} 감정 매핑 테스트 ---")
        print(f"VAD: ({emotion.valence:.2f}, {emotion.arousal:.2f}, {emotion.dominance:.2f})")
        
        # 매핑 실행
        mapping_result = mapper.map_emotion_to_ethics(emotion)
        
        print(f"📊 매핑 결과:")
        print(f"- 매핑 품질: {mapping_result.mapping_quality:.3f}")
        print(f"- 통합 신뢰도: {mapping_result.confidence_levels.get('integrated', 0.0):.3f}")
        
        print(f"⚖️ 윤리 가중치:")
        for ethics, weight in mapping_result.ethics_weights.items():
            print(f"  • {ethics}: {weight:.3f}")
        
        print(f"💭 언어적 설명:")
        for j, explanation in enumerate(mapping_result.linguistic_explanations, 1):
            print(f"  {j}. {explanation}")
    
    # 감정 전환 스무딩 테스트
    print(f"\n🔄 감정 전환 스무딩 테스트")
    
    current_emotion = test_emotions[0]  # 기쁨
    target_emotion = test_emotions[2]   # 슬픔
    
    print(f"현재 감정: 기쁨 VAD({current_emotion.valence:.2f}, {current_emotion.arousal:.2f}, {current_emotion.dominance:.2f})")
    print(f"목표 감정: 슬픔 VAD({target_emotion.valence:.2f}, {target_emotion.arousal:.2f}, {target_emotion.dominance:.2f})")
    
    # 여러 단계의 전환 시뮬레이션
    transition_emotion = current_emotion
    for step in range(5):
        transition_emotion = mapper.smooth_emotion_transition(
            transition_emotion, target_emotion, transition_speed=0.3
        )
        print(f"  단계 {step+1}: VAD({transition_emotion.valence:.2f}, {transition_emotion.arousal:.2f}, {transition_emotion.dominance:.2f})")
    
    # 분석 정보
    analytics = mapper.get_mapping_analytics()
    print(f"\n📈 매핑 분석:")
    print(f"- 총 매핑 수: {analytics['statistics']['total_mappings']}")
    print(f"- 성공적 매핑: {analytics['statistics']['successful_mappings']}")
    print(f"- 평균 신뢰도: {analytics['statistics']['average_confidence']:.3f}")
    print(f"- 평균 품질: {analytics['statistics']['average_quality']:.3f}")
    print(f"- 퍼지 규칙 수: {analytics['fuzzy_rules_count']}")
    print(f"- 언어적 변수: {', '.join(analytics['linguistic_variables'])}")
    
    if 'recent_emotion_trend' in analytics:
        trend = analytics['recent_emotion_trend']
        print(f"- 최근 감정 트렌드:")
        print(f"  평균 Valence: {trend['avg_valence']:.3f}")
        print(f"  평균 Arousal: {trend['avg_arousal']:.3f}")
        print(f"  평균 품질: {trend['avg_quality']:.3f}")
    
    if 'ethics_trends' in analytics:
        print(f"- 윤리 가중치 트렌드:")
        for category, trend_data in analytics['ethics_trends'].items():
            print(f"  {category}: 평균 {trend_data['mean']:.3f}, 변화 {trend_data['trend']:+.3f}")
    
    print("✅ 퍼지 로직 감정-윤리 매핑 테스트 완료")
    
    return mapper


if __name__ == "__main__":
    test_fuzzy_emotion_ethics_mapper()
