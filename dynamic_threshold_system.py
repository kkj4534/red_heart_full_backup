"""
Dynamic Threshold System - 동적 임계값 조정 시스템
후회 분석에서 상황별 동적 임계값 계산
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import math

logger = logging.getLogger('RedHeart.DynamicThreshold')

@dataclass
class DynamicThresholdResult:
    """동적 임계값 계산 결과"""
    threshold: float
    relative_regret: float
    absolute_regret: float
    stakeholder_penalty: float
    context_complexity: float
    confidence: float
    calculation_metadata: Dict[str, Any]

class DynamicThresholdCalculator:
    """동적 임계값 계산기"""
    
    def __init__(self):
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 기본 임계값 설정
        self.base_threshold = 0.3
        self.min_threshold = 0.1
        self.max_threshold = 0.8
        
        # 상황별 가중치
        self.context_weights = {
            'stakeholder_count': 0.3,
            'time_pressure': 0.2,
            'uncertainty_level': 0.2,
            'ethical_complexity': 0.3
        }
    
    def calculate_stakeholder_penalty(self, context: Dict[str, Any]) -> float:
        """이해관계자 페널티 계산
        
        Args:
            context: 상황 컨텍스트
            
        Returns:
            float: 이해관계자 페널티 (0.0 ~ 1.0)
        """
        penalty_factors = []
        
        # 1. 이해관계자 수
        stakeholder_count = context.get('affected_count', 1)
        if stakeholder_count > 1:
            # 로그 스케일로 정규화
            count_penalty = min(np.log(stakeholder_count) / np.log(100), 1.0)
            penalty_factors.append(count_penalty * 0.4)
        
        # 2. 취약계층 포함 여부
        vulnerable_involved = context.get('vulnerable_protection', 0.0)
        if vulnerable_involved > 0.5:
            penalty_factors.append(vulnerable_involved * 0.3)
        
        # 3. 사회적 파급효과
        social_impact = context.get('social_impact', 0.5)
        penalty_factors.append(social_impact * 0.2)
        
        # 4. 장기적 영향
        long_term_impact = context.get('long_term_impact', 0.5)
        penalty_factors.append(long_term_impact * 0.1)
        
        # 가중 평균
        if penalty_factors:
            stakeholder_penalty = np.mean(penalty_factors)
        else:
            stakeholder_penalty = 0.0
        
        return min(max(stakeholder_penalty, 0.0), 1.0)
    
    def calculate_context_complexity(self, context: Dict[str, Any]) -> float:
        """상황 복잡도 계산
        
        Args:
            context: 상황 컨텍스트
            
        Returns:
            float: 상황 복잡도 (0.0 ~ 1.0)
        """
        complexity_factors = []
        
        # 1. 윤리적 복잡도
        ethical_complexity = context.get('ethical_complexity', 0.5)
        complexity_factors.append(ethical_complexity * 0.3)
        
        # 2. 불확실성 수준
        uncertainty_level = context.get('uncertainty_level', 0.5)
        complexity_factors.append(uncertainty_level * 0.3)
        
        # 3. 시간 압박도
        time_pressure = context.get('time_pressure', 0.5)
        complexity_factors.append(time_pressure * 0.2)
        
        # 4. 정보 부족도
        information_quality = context.get('information_quality', 0.5)
        info_shortage = 1.0 - information_quality
        complexity_factors.append(info_shortage * 0.2)
        
        # 가중 평균
        if complexity_factors:
            complexity = np.mean(complexity_factors)
        else:
            complexity = 0.5
        
        return min(max(complexity, 0.0), 1.0)
    
    def calculate_dynamic_threshold(self, delta_utility: float, 
                                  context: Dict[str, Any]) -> DynamicThresholdResult:
        """동적 임계값 계산
        
        Args:
            delta_utility: 유틸리티 차이
            context: 상황 컨텍스트
            
        Returns:
            DynamicThresholdResult: 동적 임계값 결과
        """
        try:
            # 1. 이해관계자 페널티 계산
            stakeholder_penalty = self.calculate_stakeholder_penalty(context)
            
            # 2. 상황 복잡도 계산
            context_complexity = self.calculate_context_complexity(context)
            
            # 3. 동적 임계값 계산 (sigmoid 기반)
            # threshold = sigmoid(Δutility / (1 + stakeholder_penalty))
            normalized_delta = abs(delta_utility) / (1.0 + stakeholder_penalty)
            sigmoid_threshold = 1.0 / (1.0 + np.exp(-normalized_delta))
            
            # 4. 상황 복잡도 적용
            complexity_adjustment = 1.0 + (context_complexity - 0.5) * 0.4
            adjusted_threshold = sigmoid_threshold * complexity_adjustment
            
            # 5. 범위 제한
            final_threshold = np.clip(adjusted_threshold, self.min_threshold, self.max_threshold)
            
            # 6. 상대적 후회 계산
            # 현재 상황에서의 최대 가능 유틸리티 차이 추정
            max_possible_delta = self._estimate_max_utility_delta(context)
            if max_possible_delta > 0:
                relative_regret = abs(delta_utility) / max_possible_delta
            else:
                relative_regret = 0.0
            
            # 7. 절대적 후회 계산
            absolute_regret = abs(delta_utility)
            
            # 8. 신뢰도 계산
            confidence = self._calculate_threshold_confidence(
                delta_utility, stakeholder_penalty, context_complexity, context
            )
            
            # 9. 계산 메타데이터
            calculation_metadata = {
                'base_threshold': self.base_threshold,
                'sigmoid_threshold': sigmoid_threshold,
                'complexity_adjustment': complexity_adjustment,
                'max_possible_delta': max_possible_delta,
                'timestamp': datetime.now().isoformat(),
                'method': 'sigmoid_based_dynamic'
            }
            
            return DynamicThresholdResult(
                threshold=final_threshold,
                relative_regret=relative_regret,
                absolute_regret=absolute_regret,
                stakeholder_penalty=stakeholder_penalty,
                context_complexity=context_complexity,
                confidence=confidence,
                calculation_metadata=calculation_metadata
            )
            
        except Exception as e:
            self.logger.error(f"동적 임계값 계산 실패: {e}")
            # 폴백: 기본 임계값 반환
            return DynamicThresholdResult(
                threshold=self.base_threshold,
                relative_regret=0.0,
                absolute_regret=abs(delta_utility),
                stakeholder_penalty=0.0,
                context_complexity=0.5,
                confidence=0.5,
                calculation_metadata={'error': str(e), 'fallback': True}
            )
    
    def _estimate_max_utility_delta(self, context: Dict[str, Any]) -> float:
        """최대 가능 유틸리티 차이 추정
        
        Args:
            context: 상황 컨텍스트
            
        Returns:
            float: 추정된 최대 유틸리티 차이
        """
        # 상황별 최대 유틸리티 차이 추정
        estimation_factors = []
        
        # 1. 옵션 수에 따른 차이
        option_count = context.get('option_count', 2)
        if option_count > 1:
            # 더 많은 옵션 = 더 큰 차이 가능성
            option_factor = min(np.log(option_count) / np.log(10), 1.0)
            estimation_factors.append(option_factor)
        
        # 2. 영향 범위에 따른 차이
        affected_count = context.get('affected_count', 1)
        if affected_count > 1:
            impact_factor = min(np.log(affected_count) / np.log(1000), 1.0)
            estimation_factors.append(impact_factor)
        
        # 3. 시간 범위에 따른 차이
        duration = context.get('duration_seconds', 3600)
        if duration > 3600:  # 1시간 이상
            time_factor = min(np.log(duration) / np.log(31536000), 1.0)  # 1년 기준
            estimation_factors.append(time_factor)
        
        # 4. 불확실성에 따른 차이
        uncertainty = context.get('uncertainty_level', 0.5)
        uncertainty_factor = uncertainty
        estimation_factors.append(uncertainty_factor)
        
        # 가중 평균으로 최대 차이 추정
        if estimation_factors:
            max_delta = np.mean(estimation_factors)
        else:
            max_delta = 0.5  # 기본값
        
        return max_delta
    
    def _calculate_threshold_confidence(self, delta_utility: float, 
                                      stakeholder_penalty: float,
                                      context_complexity: float,
                                      context: Dict[str, Any]) -> float:
        """임계값 계산 신뢰도 계산
        
        Args:
            delta_utility: 유틸리티 차이
            stakeholder_penalty: 이해관계자 페널티
            context_complexity: 상황 복잡도
            context: 상황 컨텍스트
            
        Returns:
            float: 신뢰도 (0.0 ~ 1.0)
        """
        confidence_factors = []
        
        # 1. 정보 품질
        information_quality = context.get('information_quality', 0.5)
        confidence_factors.append(information_quality * 0.3)
        
        # 2. 불확실성 (역수)
        uncertainty_level = context.get('uncertainty_level', 0.5)
        certainty = 1.0 - uncertainty_level
        confidence_factors.append(certainty * 0.3)
        
        # 3. 데이터 완전성
        data_completeness = context.get('data_completeness', 0.5)
        confidence_factors.append(data_completeness * 0.2)
        
        # 4. 계산 안정성 (극단값이 아닐수록 높은 신뢰도)
        delta_stability = 1.0 - min(abs(delta_utility), 1.0)
        confidence_factors.append(delta_stability * 0.2)
        
        # 가중 평균
        if confidence_factors:
            confidence = np.mean(confidence_factors)
        else:
            confidence = 0.5
        
        return min(max(confidence, 0.0), 1.0)
    
    def update_threshold_based_on_feedback(self, actual_regret: float,
                                         predicted_threshold: float,
                                         context: Dict[str, Any]) -> float:
        """피드백 기반 임계값 조정
        
        Args:
            actual_regret: 실제 후회 값
            predicted_threshold: 예측된 임계값
            context: 상황 컨텍스트
            
        Returns:
            float: 조정된 임계값
        """
        # 예측 오차 계산
        prediction_error = abs(actual_regret - predicted_threshold)
        
        # 학습률 계산 (상황 복잡도에 따라 조정)
        context_complexity = self.calculate_context_complexity(context)
        learning_rate = 0.01 * (1.0 + context_complexity)
        
        # 임계값 조정
        if actual_regret > predicted_threshold:
            # 실제 후회가 더 큰 경우: 임계값 증가
            adjustment = learning_rate * prediction_error
        else:
            # 실제 후회가 더 작은 경우: 임계값 감소
            adjustment = -learning_rate * prediction_error
        
        adjusted_threshold = predicted_threshold + adjustment
        
        # 범위 제한
        return np.clip(adjusted_threshold, self.min_threshold, self.max_threshold)

# 전역 계산기 인스턴스
dynamic_threshold_calculator = DynamicThresholdCalculator()