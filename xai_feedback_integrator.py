"""
XAI 피드백 통합기 (XAI Feedback Integrator)
XAI Feedback Integration Module

해석 가능성(XAI) 결과를 시스템 개선으로 연결하는 피드백 루프를 구현하여
모델의 투명성을 활용한 지속적 학습과 성능 향상을 달성합니다.

핵심 기능:
1. XAI 해석 결과의 구조화된 분석
2. 해석 결과 기반 자동 정책 조정
3. 피드백 루프를 통한 점진적 모델 개선
4. 사용자 이해도 기반 설명 품질 최적화
"""

import numpy as np
import torch
import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from pathlib import Path
import threading

try:
    from config import ADVANCED_CONFIG, DEVICE
except ImportError:
    # config.py에 문제가 있을 경우 기본값 사용
    ADVANCED_CONFIG = {'enable_gpu': False}
    DEVICE = 'cpu'
    print("⚠️  config.py 임포트 실패, 기본값 사용")
from data_models import EmotionData

logger = logging.getLogger('XAIFeedbackIntegrator')

@dataclass
class XAIInterpretation:
    """XAI 해석 결과 데이터 클래스"""
    interpretation_id: str
    decision_id: str
    timestamp: float = field(default_factory=time.time)
    
    # 특성 중요도 (Feature Importance)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    # 주의 가중치 (Attention Weights)
    attention_weights: Dict[str, float] = field(default_factory=dict)
    
    # 반사실적 설명 (Counterfactual Explanations)
    counterfactuals: List[Dict[str, Any]] = field(default_factory=list)
    
    # 규칙 기반 설명 (Rule-based Explanations)
    rule_explanations: List[str] = field(default_factory=list)
    
    # 신뢰도 및 불확실성
    explanation_confidence: float = 0.7
    model_uncertainty: float = 0.3
    
    # 사용자 피드백
    user_understanding: Optional[float] = None  # 0-1 (이해도)
    user_agreement: Optional[float] = None      # 0-1 (동의도)
    explanation_quality: Optional[float] = None # 0-1 (설명 품질)

@dataclass
class FeedbackAction:
    """피드백 기반 액션"""
    action_type: str  # 'weight_adjustment', 'rule_modification', 'parameter_tuning'
    target_component: str  # 대상 컴포넌트
    adjustment_details: Dict[str, Any] = field(default_factory=dict)
    expected_impact: float = 0.0
    confidence: float = 0.5
    reasoning: str = ""

@dataclass
class IntegrationResult:
    """통합 결과"""
    interpretation: XAIInterpretation
    identified_issues: List[str] = field(default_factory=list)
    recommended_actions: List[FeedbackAction] = field(default_factory=list)
    system_improvements: Dict[str, float] = field(default_factory=dict)
    feedback_quality: float = 0.0
    integration_success: bool = False

class XAIFeedbackIntegrator:
    """XAI 피드백 통합기"""
    
    def __init__(self):
        self.logger = logger
        
        # 해석 결과 저장소
        self.interpretations_history = deque(maxlen=1000)
        self.feedback_actions_history = deque(maxlen=500)
        
        # 피드백 통계
        self.feedback_statistics = {
            'total_interpretations': 0,
            'successful_integrations': 0,
            'average_user_understanding': 0.0,
            'average_explanation_quality': 0.0,
            'improvement_rate': 0.0
        }
        
        # 특성 중요도 추적
        self.feature_importance_tracker = defaultdict(list)
        self.attention_patterns_tracker = defaultdict(list)
        
        # 시스템 컴포넌트 매핑
        self.component_mappings = {
            'emotion_analyzer': ['valence', 'arousal', 'dominance', 'emotion_confidence'],
            'bentham_calculator': ['intensity', 'duration', 'certainty', 'propinquity', 'fecundity', 'purity', 'extent'],
            'regret_system': ['anticipated_regret', 'regret_intensity', 'risk_aversion'],
            'ethics_policy': ['care_harm', 'fairness_cheating', 'loyalty_betrayal', 'authority_subversion'],
            'phase_controller': ['exploration_rate', 'safety_threshold', 'confidence_threshold']
        }
        
        # 개선 임계값
        self.improvement_thresholds = {
            'feature_importance_change': 0.1,
            'attention_drift': 0.15,
            'user_understanding_drop': 0.2,
            'explanation_quality_drop': 0.25,
            'model_uncertainty_increase': 0.3
        }
        
        # 적응적 학습 파라미터
        self.adaptation_config = {
            'learning_rate': 0.02,
            'momentum': 0.9,
            'feedback_weight': 0.7,
            'stability_factor': 0.8,
            'min_feedback_count': 3
        }
        
        self.logger.info("XAI 피드백 통합기 초기화 완료")
    
    def integrate_xai_feedback(
        self, 
        interpretation: XAIInterpretation,
        system_components: Dict[str, Any]
    ) -> IntegrationResult:
        """XAI 해석 결과를 시스템 피드백으로 통합"""
        
        start_time = time.time()
        
        # 해석 결과 저장
        self.interpretations_history.append(interpretation)
        self.feedback_statistics['total_interpretations'] += 1
        
        # 1단계: 특성 중요도 분석
        feature_analysis = self._analyze_feature_importance(interpretation)
        
        # 2단계: 주의 패턴 분석
        attention_analysis = self._analyze_attention_patterns(interpretation)
        
        # 3단계: 사용자 피드백 분석
        user_feedback_analysis = self._analyze_user_feedback(interpretation)
        
        # 4단계: 이상 패턴 탐지
        anomaly_detection = self._detect_interpretation_anomalies(interpretation)
        
        # 5단계: 개선 액션 생성
        improvement_actions = self._generate_improvement_actions(
            interpretation, feature_analysis, attention_analysis, 
            user_feedback_analysis, anomaly_detection
        )
        
        # 6단계: 시스템 컴포넌트 업데이트
        system_improvements = self._apply_system_improvements(
            improvement_actions, system_components
        )
        
        # 7단계: 피드백 품질 평가
        feedback_quality = self._evaluate_feedback_quality(
            interpretation, improvement_actions, system_improvements
        )
        
        # 결과 생성
        result = IntegrationResult(
            interpretation=interpretation,
            identified_issues=anomaly_detection,
            recommended_actions=improvement_actions,
            system_improvements=system_improvements,
            feedback_quality=feedback_quality,
            integration_success=len(system_improvements) > 0
        )
        
        # 통계 업데이트
        self._update_integration_statistics(result)
        
        # 액션 히스토리 저장
        self.feedback_actions_history.extend(improvement_actions)
        
        processing_time = time.time() - start_time
        
        self.logger.info(
            f"XAI 피드백 통합 완료: {len(improvement_actions)}개 액션, "
            f"품질 점수 {feedback_quality:.3f}, 처리시간 {processing_time:.3f}초"
        )
        
        return result
    
    def _analyze_feature_importance(self, interpretation: XAIInterpretation) -> Dict[str, Any]:
        """특성 중요도 분석"""
        
        analysis = {
            'high_importance_features': [],
            'low_importance_features': [],
            'importance_changes': {},
            'stability_score': 0.0
        }
        
        # 현재 특성 중요도
        current_importance = interpretation.feature_importance
        
        # 중요도별 특성 분류
        for feature, importance in current_importance.items():
            if importance > 0.7:
                analysis['high_importance_features'].append(feature)
            elif importance < 0.3:
                analysis['low_importance_features'].append(feature)
        
        # 과거 중요도와 비교
        for feature, importance in current_importance.items():
            if feature in self.feature_importance_tracker:
                past_importances = self.feature_importance_tracker[feature]
                if past_importances:
                    avg_past_importance = np.mean(past_importances[-10:])  # 최근 10개 평균
                    change = importance - avg_past_importance
                    analysis['importance_changes'][feature] = change
            
            # 현재 중요도 저장
            self.feature_importance_tracker[feature].append(importance)
        
        # 안정성 점수 계산
        if analysis['importance_changes']:
            changes = list(analysis['importance_changes'].values())
            stability_score = 1.0 - np.std(changes)  # 변화량의 표준편차가 낮을수록 안정적
            analysis['stability_score'] = max(0.0, stability_score)
        
        return analysis
    
    def _analyze_attention_patterns(self, interpretation: XAIInterpretation) -> Dict[str, Any]:
        """주의 패턴 분석"""
        
        analysis = {
            'attention_focus': [],
            'attention_drift': {},
            'pattern_consistency': 0.0,
            'unexpected_patterns': []
        }
        
        current_attention = interpretation.attention_weights
        
        # 주의 집중 영역 식별
        if current_attention:
            sorted_attention = sorted(current_attention.items(), key=lambda x: x[1], reverse=True)
            top_3_attention = sorted_attention[:3]
            analysis['attention_focus'] = [item[0] for item in top_3_attention]
        
        # 주의 패턴 변화 분석
        for component, weight in current_attention.items():
            if component in self.attention_patterns_tracker:
                past_weights = self.attention_patterns_tracker[component]
                if past_weights:
                    avg_past_weight = np.mean(past_weights[-5:])  # 최근 5개 평균
                    drift = weight - avg_past_weight
                    if abs(drift) > self.improvement_thresholds['attention_drift']:
                        analysis['attention_drift'][component] = drift
            
            # 현재 주의 가중치 저장
            self.attention_patterns_tracker[component].append(weight)
        
        # 예상치 못한 패턴 탐지
        for component, weight in current_attention.items():
            # 일반적으로 중요하지 않은 컴포넌트가 높은 주의를 받는 경우
            if component in ['temporal_urgency', 'external_pressure'] and weight > 0.8:
                analysis['unexpected_patterns'].append(f"높은 주의: {component} ({weight:.3f})")
        
        return analysis
    
    def _analyze_user_feedback(self, interpretation: XAIInterpretation) -> Dict[str, Any]:
        """사용자 피드백 분석"""
        
        analysis = {
            'feedback_available': False,
            'understanding_level': 0.5,
            'agreement_level': 0.5,
            'quality_rating': 0.5,
            'improvement_needed': False
        }
        
        # 사용자 피드백 존재 여부
        if (interpretation.user_understanding is not None or 
            interpretation.user_agreement is not None or 
            interpretation.explanation_quality is not None):
            analysis['feedback_available'] = True
        
        # 피드백 값 설정
        if interpretation.user_understanding is not None:
            analysis['understanding_level'] = interpretation.user_understanding
        
        if interpretation.user_agreement is not None:
            analysis['agreement_level'] = interpretation.user_agreement
        
        if interpretation.explanation_quality is not None:
            analysis['quality_rating'] = interpretation.explanation_quality
        
        # 개선 필요성 판단
        if (analysis['understanding_level'] < 0.6 or 
            analysis['quality_rating'] < 0.6):
            analysis['improvement_needed'] = True
        
        return analysis
    
    def _detect_interpretation_anomalies(self, interpretation: XAIInterpretation) -> List[str]:
        """해석 결과 이상 패턴 탐지"""
        
        anomalies = []
        
        # 1. 특성 중요도 이상
        importance_values = list(interpretation.feature_importance.values())
        if importance_values:
            if max(importance_values) < 0.3:  # 모든 특성이 낮은 중요도
                anomalies.append("모든 특성의 중요도가 낮음 - 모델 신뢰성 검토 필요")
            
            if len([v for v in importance_values if v > 0.8]) > len(importance_values) * 0.7:
                anomalies.append("과도하게 많은 특성이 높은 중요도 - 과적합 가능성")
        
        # 2. 모델 불확실성 이상
        if interpretation.model_uncertainty > 0.7:
            anomalies.append("높은 모델 불확실성 - 추가 학습 데이터 필요")
        
        # 3. 설명 신뢰도 이상
        if interpretation.explanation_confidence < 0.4:
            anomalies.append("낮은 설명 신뢰도 - XAI 모델 개선 필요")
        
        # 4. 사용자 피드백 이상
        if (interpretation.user_understanding is not None and 
            interpretation.user_understanding < 0.3):
            anomalies.append("사용자 이해도 매우 낮음 - 설명 방식 재검토 필요")
        
        # 5. 반사실적 설명 부족
        if len(interpretation.counterfactuals) == 0:
            anomalies.append("반사실적 설명 부족 - 대안 시나리오 생성 필요")
        
        return anomalies
    
    def _generate_improvement_actions(
        self,
        interpretation: XAIInterpretation,
        feature_analysis: Dict[str, Any],
        attention_analysis: Dict[str, Any],
        user_feedback_analysis: Dict[str, Any],
        anomalies: List[str]
    ) -> List[FeedbackAction]:
        """개선 액션 생성"""
        
        actions = []
        
        # 1. 특성 중요도 기반 액션
        if feature_analysis['importance_changes']:
            for feature, change in feature_analysis['importance_changes'].items():
                if abs(change) > self.improvement_thresholds['feature_importance_change']:
                    component = self._map_feature_to_component(feature)
                    if component:
                        action = FeedbackAction(
                            action_type='weight_adjustment',
                            target_component=component,
                            adjustment_details={
                                'feature': feature,
                                'importance_change': change,
                                'adjustment_magnitude': min(abs(change) * 0.5, 0.1)
                            },
                            expected_impact=abs(change) * 0.3,
                            confidence=0.7,
                            reasoning=f"특성 {feature}의 중요도가 {change:.3f} 변화"
                        )
                        actions.append(action)
        
        # 2. 주의 패턴 기반 액션
        if attention_analysis['attention_drift']:
            for component, drift in attention_analysis['attention_drift'].items():
                action = FeedbackAction(
                    action_type='parameter_tuning',
                    target_component=component,
                    adjustment_details={
                        'attention_drift': drift,
                        'tuning_direction': 'increase' if drift < 0 else 'decrease',
                        'tuning_magnitude': min(abs(drift) * 0.3, 0.05)
                    },
                    expected_impact=abs(drift) * 0.2,
                    confidence=0.6,
                    reasoning=f"주의 패턴이 {drift:.3f} 변화한 {component} 조정"
                )
                actions.append(action)
        
        # 3. 사용자 피드백 기반 액션
        if user_feedback_analysis['improvement_needed']:
            if user_feedback_analysis['understanding_level'] < 0.6:
                action = FeedbackAction(
                    action_type='rule_modification',
                    target_component='explanation_generator',
                    adjustment_details={
                        'understanding_level': user_feedback_analysis['understanding_level'],
                        'modification_type': 'simplify_explanations',
                        'complexity_reduction': 0.3
                    },
                    expected_impact=0.4,
                    confidence=0.8,
                    reasoning="사용자 이해도 향상을 위한 설명 단순화"
                )
                actions.append(action)
            
            if user_feedback_analysis['quality_rating'] < 0.6:
                action = FeedbackAction(
                    action_type='rule_modification',
                    target_component='explanation_generator',
                    adjustment_details={
                        'quality_rating': user_feedback_analysis['quality_rating'],
                        'modification_type': 'enhance_detail',
                        'detail_enhancement': 0.4
                    },
                    expected_impact=0.3,
                    confidence=0.7,
                    reasoning="설명 품질 향상을 위한 상세도 증가"
                )
                actions.append(action)
        
        # 4. 이상 패턴 기반 액션
        for anomaly in anomalies:
            if "모든 특성의 중요도가 낮음" in anomaly:
                action = FeedbackAction(
                    action_type='parameter_tuning',
                    target_component='feature_selector',
                    adjustment_details={
                        'tuning_type': 'feature_selection_threshold',
                        'threshold_adjustment': -0.1
                    },
                    expected_impact=0.5,
                    confidence=0.6,
                    reasoning="특성 선택 임계값 조정으로 더 많은 특성 활용"
                )
                actions.append(action)
            
            elif "높은 모델 불확실성" in anomaly:
                action = FeedbackAction(
                    action_type='parameter_tuning',
                    target_component='uncertainty_estimator',
                    adjustment_details={
                        'tuning_type': 'confidence_calibration',
                        'calibration_strength': 0.2
                    },
                    expected_impact=0.4,
                    confidence=0.5,
                    reasoning="불확실성 추정 모델 보정"
                )
                actions.append(action)
        
        return actions
    
    def _map_feature_to_component(self, feature: str) -> Optional[str]:
        """특성을 시스템 컴포넌트에 매핑"""
        
        for component, features in self.component_mappings.items():
            if feature in features:
                return component
        
        # 키워드 기반 매핑
        if any(keyword in feature.lower() for keyword in ['emotion', 'feeling', 'sentiment']):
            return 'emotion_analyzer'
        elif any(keyword in feature.lower() for keyword in ['bentham', 'pleasure', 'pain', 'utility']):
            return 'bentham_calculator'
        elif any(keyword in feature.lower() for keyword in ['regret', 'counterfactual']):
            return 'regret_system'
        elif any(keyword in feature.lower() for keyword in ['ethics', 'moral', 'fairness']):
            return 'ethics_policy'
        elif any(keyword in feature.lower() for keyword in ['phase', 'exploration', 'execution']):
            return 'phase_controller'
        
        return None
    
    def _apply_system_improvements(
        self,
        actions: List[FeedbackAction],
        system_components: Dict[str, Any]
    ) -> Dict[str, float]:
        """시스템 개선 적용"""
        
        improvements = {}
        
        for action in actions:
            component_name = action.target_component
            
            if component_name in system_components:
                component = system_components[component_name]
                
                try:
                    # 가중치 조정 액션
                    if action.action_type == 'weight_adjustment':
                        improvement = self._apply_weight_adjustment(component, action)
                        if improvement > 0:
                            improvements[f"{component_name}_weight_adjustment"] = improvement
                    
                    # 파라미터 튜닝 액션
                    elif action.action_type == 'parameter_tuning':
                        improvement = self._apply_parameter_tuning(component, action)
                        if improvement > 0:
                            improvements[f"{component_name}_parameter_tuning"] = improvement
                    
                    # 규칙 수정 액션
                    elif action.action_type == 'rule_modification':
                        improvement = self._apply_rule_modification(component, action)
                        if improvement > 0:
                            improvements[f"{component_name}_rule_modification"] = improvement
                
                except Exception as e:
                    self.logger.warning(f"액션 적용 실패 ({component_name}): {e}")
                    continue
        
        return improvements
    
    def _apply_weight_adjustment(self, component: Any, action: FeedbackAction) -> float:
        """가중치 조정 적용"""
        
        details = action.adjustment_details
        
        # 컴포넌트에 adjust_weights 메서드가 있는지 확인
        if hasattr(component, 'adjust_weights'):
            try:
                adjustment_result = component.adjust_weights(
                    feature=details.get('feature'),
                    magnitude=details.get('adjustment_magnitude', 0.05)
                )
                return action.expected_impact
            except Exception as e:
                self.logger.warning(f"가중치 조정 실패: {e}")
                return 0.0
        
        # 다른 방법으로 가중치 조정 시도
        elif hasattr(component, 'config') and isinstance(component.config, dict):
            feature = details.get('feature')
            magnitude = details.get('adjustment_magnitude', 0.05)
            
            if feature in component.config:
                old_value = component.config[feature]
                component.config[feature] = np.clip(old_value + magnitude, 0.0, 1.0)
                return action.expected_impact * 0.7
        
        return 0.0
    
    def _apply_parameter_tuning(self, component: Any, action: FeedbackAction) -> float:
        """파라미터 튜닝 적용"""
        
        details = action.adjustment_details
        
        # 컴포넌트에 tune_parameters 메서드가 있는지 확인
        if hasattr(component, 'tune_parameters'):
            try:
                tuning_result = component.tune_parameters(
                    tuning_type=details.get('tuning_type'),
                    magnitude=details.get('tuning_magnitude', 0.05)
                )
                return action.expected_impact
            except Exception as e:
                self.logger.warning(f"파라미터 튜닝 실패: {e}")
                return 0.0
        
        # 기본 파라미터 조정
        elif hasattr(component, 'learning_config') and isinstance(component.learning_config, dict):
            tuning_type = details.get('tuning_type')
            magnitude = details.get('tuning_magnitude', 0.05)
            
            if tuning_type == 'learning_rate' and 'learning_rate' in component.learning_config:
                old_lr = component.learning_config['learning_rate']
                component.learning_config['learning_rate'] = np.clip(old_lr * (1 + magnitude), 0.001, 0.1)
                return action.expected_impact * 0.5
        
        return 0.0
    
    def _apply_rule_modification(self, component: Any, action: FeedbackAction) -> float:
        """규칙 수정 적용"""
        
        details = action.adjustment_details
        
        # 컴포넌트에 modify_rules 메서드가 있는지 확인
        if hasattr(component, 'modify_rules'):
            try:
                modification_result = component.modify_rules(
                    modification_type=details.get('modification_type'),
                    parameters=details
                )
                return action.expected_impact
            except Exception as e:
                self.logger.warning(f"규칙 수정 실패: {e}")
                return 0.0
        
        # 설명 생성기 특별 처리
        elif hasattr(component, 'explanation_config'):
            mod_type = details.get('modification_type')
            
            if mod_type == 'simplify_explanations':
                component.explanation_config['complexity_level'] = max(
                    component.explanation_config.get('complexity_level', 0.5) - 0.2, 0.1
                )
                return action.expected_impact * 0.6
            
            elif mod_type == 'enhance_detail':
                component.explanation_config['detail_level'] = min(
                    component.explanation_config.get('detail_level', 0.5) + 0.2, 1.0
                )
                return action.expected_impact * 0.6
        
        return 0.0
    
    def _evaluate_feedback_quality(
        self,
        interpretation: XAIInterpretation,
        actions: List[FeedbackAction],
        improvements: Dict[str, float]
    ) -> float:
        """피드백 품질 평가"""
        
        quality_score = 0.0
        
        # 1. 해석 결과 품질 (30%)
        interpretation_quality = (
            interpretation.explanation_confidence * 0.5 +
            (1.0 - interpretation.model_uncertainty) * 0.5
        )
        quality_score += interpretation_quality * 0.3
        
        # 2. 액션 품질 (40%)
        if actions:
            action_confidence = np.mean([action.confidence for action in actions])
            action_impact = np.mean([action.expected_impact for action in actions])
            action_quality = (action_confidence + action_impact) / 2.0
            quality_score += action_quality * 0.4
        else:
            quality_score += 0.2  # 액션이 없어도 최소 점수
        
        # 3. 개선 효과 (30%)
        if improvements:
            improvement_quality = min(np.mean(list(improvements.values())), 1.0)
            quality_score += improvement_quality * 0.3
        
        return np.clip(quality_score, 0.0, 1.0)
    
    def _update_integration_statistics(self, result: IntegrationResult):
        """통합 통계 업데이트"""
        
        if result.integration_success:
            self.feedback_statistics['successful_integrations'] += 1
        
        # 사용자 이해도 업데이트
        if result.interpretation.user_understanding is not None:
            current_understanding = self.feedback_statistics['average_user_understanding']
            total_count = self.feedback_statistics['total_interpretations']
            new_understanding = (
                current_understanding * (total_count - 1) + result.interpretation.user_understanding
            ) / total_count
            self.feedback_statistics['average_user_understanding'] = new_understanding
        
        # 설명 품질 업데이트
        if result.interpretation.explanation_quality is not None:
            current_quality = self.feedback_statistics['average_explanation_quality']
            total_count = self.feedback_statistics['total_interpretations']
            new_quality = (
                current_quality * (total_count - 1) + result.interpretation.explanation_quality
            ) / total_count
            self.feedback_statistics['average_explanation_quality'] = new_quality
        
        # 개선률 업데이트
        if result.system_improvements:
            improvement_count = len(result.system_improvements)
            self.feedback_statistics['improvement_rate'] = (
                self.feedback_statistics['successful_integrations'] / 
                max(self.feedback_statistics['total_interpretations'], 1)
            )
    
    def get_feedback_analytics(self) -> Dict[str, Any]:
        """피드백 분석 정보 반환"""
        
        return {
            'statistics': self.feedback_statistics.copy(),
            'recent_interpretations': len(self.interpretations_history),
            'recent_actions': len(self.feedback_actions_history),
            'feature_importance_trends': {
                feature: np.mean(values[-10:]) if values else 0.0
                for feature, values in self.feature_importance_tracker.items()
            },
            'attention_patterns_trends': {
                component: np.mean(values[-5:]) if values else 0.0
                for component, values in self.attention_patterns_tracker.items()
            },
            'top_improvement_areas': self._identify_top_improvement_areas(),
            'system_stability': self._calculate_system_stability()
        }
    
    def _identify_top_improvement_areas(self) -> List[str]:
        """주요 개선 영역 식별"""
        
        improvement_areas = []
        
        # 최근 액션들 분석
        recent_actions = list(self.feedback_actions_history)[-20:]  # 최근 20개
        
        # 액션 타입별 빈도
        action_type_counts = defaultdict(int)
        target_component_counts = defaultdict(int)
        
        for action in recent_actions:
            action_type_counts[action.action_type] += 1
            target_component_counts[action.target_component] += 1
        
        # 가장 빈번한 액션 타입
        if action_type_counts:
            top_action_type = max(action_type_counts.keys(), key=lambda k: action_type_counts[k])
            improvement_areas.append(f"주요 액션 타입: {top_action_type}")
        
        # 가장 문제가 많은 컴포넌트
        if target_component_counts:
            top_problem_component = max(target_component_counts.keys(), key=lambda k: target_component_counts[k])
            improvement_areas.append(f"개선 필요 컴포넌트: {top_problem_component}")
        
        # 사용자 피드백 기반 개선 영역
        if self.feedback_statistics['average_user_understanding'] < 0.7:
            improvement_areas.append("설명 이해도 향상 필요")
        
        if self.feedback_statistics['average_explanation_quality'] < 0.7:
            improvement_areas.append("설명 품질 향상 필요")
        
        return improvement_areas
    
    def _calculate_system_stability(self) -> float:
        """시스템 안정성 계산"""
        
        if len(self.interpretations_history) < 5:
            return 0.5  # 데이터 부족
        
        # 최근 해석 결과들의 일관성 분석
        recent_interpretations = list(self.interpretations_history)[-10:]
        
        # 설명 신뢰도의 일관성
        confidence_scores = [interp.explanation_confidence for interp in recent_interpretations]
        confidence_stability = 1.0 - np.std(confidence_scores)
        
        # 모델 불확실성의 일관성
        uncertainty_scores = [interp.model_uncertainty for interp in recent_interpretations]
        uncertainty_stability = 1.0 - np.std(uncertainty_scores)
        
        # 전체 안정성 점수
        stability_score = (confidence_stability + uncertainty_stability) / 2.0
        
        return np.clip(stability_score, 0.0, 1.0)
    
    def save_feedback_state(self, filepath: str):
        """피드백 상태 저장"""
        
        state_data = {
            'feedback_statistics': self.feedback_statistics,
            'interpretations_count': len(self.interpretations_history),
            'actions_count': len(self.feedback_actions_history),
            'feature_importance_summary': {
                feature: {
                    'mean': float(np.mean(values)) if values else 0.0,
                    'std': float(np.std(values)) if values else 0.0,
                    'count': len(values)
                }
                for feature, values in self.feature_importance_tracker.items()
            },
            'attention_patterns_summary': {
                component: {
                    'mean': float(np.mean(values)) if values else 0.0,
                    'std': float(np.std(values)) if values else 0.0,
                    'count': len(values)
                }
                for component, values in self.attention_patterns_tracker.items()
            },
            'analytics': self.get_feedback_analytics()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"XAI 피드백 상태를 {filepath}에 저장 완료")


# 테스트 및 데모 함수
def test_xai_feedback_integrator():
    """XAI 피드백 통합기 테스트"""
    print("🔍 XAI 피드백 통합기 테스트 시작")
    
    # 통합기 초기화
    integrator = XAIFeedbackIntegrator()
    
    # 테스트 해석 결과 생성
    test_interpretation = XAIInterpretation(
        interpretation_id="test_interp_1",
        decision_id="decision_123",
        feature_importance={
            'valence': 0.8,
            'arousal': 0.6,
            'intensity': 0.9,
            'care_harm': 0.7,
            'fairness_cheating': 0.4
        },
        attention_weights={
            'emotion_analyzer': 0.7,
            'bentham_calculator': 0.8,
            'ethics_policy': 0.6,
            'regret_system': 0.3
        },
        counterfactuals=[
            {'scenario': 'if valence was higher', 'probability': 0.8},
            {'scenario': 'if intensity was lower', 'probability': 0.6}
        ],
        rule_explanations=[
            "높은 감정 강도로 인한 결정",
            "돌봄-해악 원칙이 주요 영향"
        ],
        explanation_confidence=0.75,
        model_uncertainty=0.25,
        user_understanding=0.6,
        user_agreement=0.8,
        explanation_quality=0.7
    )
    
    # 가상의 시스템 컴포넌트
    mock_components = {
        'emotion_analyzer': type('MockComponent', (), {
            'config': {'valence_weight': 0.8, 'arousal_weight': 0.6},
            'adjust_weights': lambda self, feature, magnitude: True
        })(),
        'bentham_calculator': type('MockComponent', (), {
            'learning_config': {'learning_rate': 0.01},
            'tune_parameters': lambda self, tuning_type, magnitude: True
        })(),
        'explanation_generator': type('MockComponent', (), {
            'explanation_config': {'complexity_level': 0.7, 'detail_level': 0.6},
            'modify_rules': lambda self, modification_type, parameters: True
        })()
    }
    
    print(f"해석 결과 생성 완료:")
    print(f"- 특성 중요도: {len(test_interpretation.feature_importance)}개")
    print(f"- 주의 가중치: {len(test_interpretation.attention_weights)}개")
    print(f"- 반사실적 설명: {len(test_interpretation.counterfactuals)}개")
    print(f"- 사용자 이해도: {test_interpretation.user_understanding:.3f}")
    
    # 피드백 통합 실행
    result = integrator.integrate_xai_feedback(test_interpretation, mock_components)
    
    # 결과 출력
    print(f"\n📊 피드백 통합 결과:")
    print(f"- 통합 성공: {'예' if result.integration_success else '아니오'}")
    print(f"- 탐지된 이슈: {len(result.identified_issues)}개")
    print(f"- 권장 액션: {len(result.recommended_actions)}개")
    print(f"- 시스템 개선: {len(result.system_improvements)}개")
    print(f"- 피드백 품질: {result.feedback_quality:.3f}")
    
    if result.identified_issues:
        print(f"\n⚠️ 탐지된 이슈들:")
        for i, issue in enumerate(result.identified_issues, 1):
            print(f"  {i}. {issue}")
    
    if result.recommended_actions:
        print(f"\n🔧 권장 액션들:")
        for i, action in enumerate(result.recommended_actions, 1):
            print(f"  {i}. {action.action_type} -> {action.target_component}")
            print(f"     예상 영향: {action.expected_impact:.3f}, 신뢰도: {action.confidence:.3f}")
            print(f"     이유: {action.reasoning}")
    
    if result.system_improvements:
        print(f"\n📈 시스템 개선들:")
        for improvement, value in result.system_improvements.items():
            print(f"  - {improvement}: {value:.3f}")
    
    # 추가 해석 결과로 테스트 (시간에 따른 변화 시뮬레이션)
    print(f"\n🔄 시간에 따른 변화 시뮬레이션")
    
    for i in range(3):
        # 변화된 해석 결과 생성
        modified_interpretation = XAIInterpretation(
            interpretation_id=f"test_interp_{i+2}",
            decision_id=f"decision_{i+124}",
            feature_importance={
                'valence': 0.8 + (i * 0.1),
                'arousal': 0.6 - (i * 0.05),
                'intensity': 0.9 - (i * 0.1),
                'care_harm': 0.7 + (i * 0.08),
                'fairness_cheating': 0.4 + (i * 0.15)
            },
            attention_weights={
                'emotion_analyzer': 0.7 - (i * 0.1),
                'bentham_calculator': 0.8 + (i * 0.05),
                'ethics_policy': 0.6 + (i * 0.1),
                'regret_system': 0.3 + (i * 0.2)
            },
            explanation_confidence=0.75 - (i * 0.1),
            model_uncertainty=0.25 + (i * 0.1),
            user_understanding=0.6 + (i * 0.1),
            explanation_quality=0.7 + (i * 0.05)
        )
        
        # 통합 실행
        iter_result = integrator.integrate_xai_feedback(modified_interpretation, mock_components)
        print(f"  반복 {i+1}: 액션 {len(iter_result.recommended_actions)}개, "
              f"품질 {iter_result.feedback_quality:.3f}")
    
    # 분석 정보
    analytics = integrator.get_feedback_analytics()
    print(f"\n📈 피드백 분석:")
    print(f"- 총 해석 수: {analytics['statistics']['total_interpretations']}")
    print(f"- 성공적 통합: {analytics['statistics']['successful_integrations']}")
    print(f"- 평균 사용자 이해도: {analytics['statistics']['average_user_understanding']:.3f}")
    print(f"- 평균 설명 품질: {analytics['statistics']['average_explanation_quality']:.3f}")
    print(f"- 시스템 안정성: {analytics['system_stability']:.3f}")
    
    if analytics['top_improvement_areas']:
        print(f"- 주요 개선 영역:")
        for area in analytics['top_improvement_areas']:
            print(f"  • {area}")
    
    print("✅ XAI 피드백 통합기 테스트 완료")
    
    return integrator, result


if __name__ == "__main__":
    test_xai_feedback_integrator()
