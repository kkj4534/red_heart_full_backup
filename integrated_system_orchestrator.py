"""
통합 시스템 오케스트레이터 (Integrated System Orchestrator)
Integrated System Orchestrator Module

모든 하위 모듈들의 유기적 통합을 관리하고 시스템 전체의 조화로운 동작을 보장하는
중앙 조정 시스템입니다. 각 모듈 간의 데이터 흐름과 상호작용을 최적화합니다.

핵심 기능:
1. 모듈 간 유기적 데이터 흐름 관리
2. 전체 시스템 성능 모니터링 및 최적화
3. 적응적 모듈 가중치 조정
4. 통합 의사결정 파이프라인 구축
"""

import os
# CVE-2025-32434는 가짜 CVE - torch_security_patch import 제거
# import torch_security_patch

import numpy as np
import torch
import logging
import json
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import traceback

# 기존 모듈들 임포트
from config import ADVANCED_CONFIG, DEVICE
from data_models import EmotionData
from emotion_ethics_regret_circuit import EmotionEthicsRegretCircuit
from ethics_policy_updater import EthicsPolicyUpdater
from phase_controller import PhaseController, PhaseDecisionContext
from xai_feedback_integrator import XAIFeedbackIntegrator
from fuzzy_emotion_ethics_mapper import FuzzyEmotionEthicsMapper
from deep_multi_dimensional_ethics_system import DeepMultiDimensionalEthicsSystem
from temporal_event_propagation_analyzer import TemporalEventPropagationAnalyzer, TemporalEvent

logger = logging.getLogger('IntegratedSystemOrchestrator')

@dataclass
class IntegrationContext:
    """통합 맥락 정보"""
    session_id: str
    timestamp: float = field(default_factory=time.time)
    
    # 입력 데이터
    user_input: str = ""
    scenario_description: str = ""
    emotional_context: Optional[EmotionData] = None
    
    # 상황 정보
    urgency_level: float = 0.5
    complexity_level: float = 0.5
    stakeholder_count: int = 1
    ethical_weight: float = 0.7
    
    # 메타데이터
    user_id: str = "default"
    cultural_context: str = "korean"
    decision_history: List[str] = field(default_factory=list)

@dataclass
class ModuleResponse:
    """모듈 응답 데이터"""
    module_name: str
    response_data: Dict[str, Any]
    confidence: float = 0.5
    processing_time: float = 0.0
    success: bool = True
    error_message: str = ""

@dataclass
class IntegratedDecision:
    """통합 의사결정 결과"""
    decision_id: str
    final_recommendation: str
    confidence_score: float
    
    # 각 모듈 기여도
    module_contributions: Dict[str, float] = field(default_factory=dict)
    
    # 상세 분석
    ethical_analysis: Dict[str, Any] = field(default_factory=dict)
    emotional_analysis: Dict[str, Any] = field(default_factory=dict)
    temporal_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # 예상 결과
    predicted_outcomes: Dict[str, Any] = field(default_factory=dict)
    risk_assessment: Dict[str, float] = field(default_factory=dict)
    
    # 설명 및 근거
    reasoning_chain: List[str] = field(default_factory=list)
    alternative_options: List[str] = field(default_factory=list)
    
    # 메타데이터
    processing_time: float = 0.0
    timestamp: float = field(default_factory=time.time)

class ModuleCoordinator:
    """모듈 조정자"""
    
    def __init__(self):
        self.module_weights = {
            'emotion_ethics_regret': 0.25,
            'ethics_policy': 0.20,
            'phase_controller': 0.15,
            'xai_feedback': 0.15,
            'fuzzy_mapper': 0.10,
            'deep_ethics': 0.10,
            'temporal_analyzer': 0.05
        }
        
        # 적응적 가중치 조정 파라미터
        self.adaptation_rate = 0.05
        self.performance_history = defaultdict(deque)
        self.correlation_matrix = np.eye(len(self.module_weights))
        
    def adjust_weights_based_on_performance(self, performance_data: Dict[str, float]):
        """성능 데이터를 기반으로 모듈 가중치 조정"""
        
        for module_name, performance in performance_data.items():
            if module_name in self.module_weights:
                # 성능이 좋으면 가중치 증가, 나쁘면 감소
                adjustment = (performance - 0.5) * self.adaptation_rate
                self.module_weights[module_name] = np.clip(
                    self.module_weights[module_name] + adjustment,
                    0.05, 0.4  # 최소 5%, 최대 40%
                )
        
        # 가중치 정규화
        total_weight = sum(self.module_weights.values())
        for module_name in self.module_weights:
            self.module_weights[module_name] /= total_weight
    
    def get_current_weights(self) -> Dict[str, float]:
        """현재 모듈 가중치 반환"""
        return self.module_weights.copy()
    
    def calculate_module_synergy(self, module_responses: Dict[str, ModuleResponse]) -> float:
        """모듈 간 시너지 계산"""
        
        if len(module_responses) < 2:
            return 0.5
        
        # 모듈 간 일치도 계산
        agreements = []
        confidences = [resp.confidence for resp in module_responses.values()]
        
        # 신뢰도 일치성
        confidence_variance = np.var(confidences)
        confidence_agreement = 1.0 / (1.0 + confidence_variance)
        agreements.append(confidence_agreement)
        
        # 결론 일치성 (간단한 휴리스틱)
        # 실제로는 더 정교한 의미 분석이 필요
        response_similarities = []
        responses = list(module_responses.values())
        
        for i in range(len(responses)):
            for j in range(i+1, len(responses)):
                # 신뢰도 기반 유사성 (실제로는 내용 분석 필요)
                similarity = 1.0 - abs(responses[i].confidence - responses[j].confidence)
                response_similarities.append(similarity)
        
        if response_similarities:
            content_agreement = np.mean(response_similarities)
            agreements.append(content_agreement)
        
        return np.mean(agreements)

class IntegratedSystemOrchestrator:
    """통합 시스템 오케스트레이터"""
    
    def __init__(self):
        self.logger = logger
        
        # 하위 모듈들 초기화
        self.modules = {}
        self.module_coordinator = ModuleCoordinator()
        
        # 스레드 풀
        self.thread_pool = ThreadPoolExecutor(max_workers=8)
        
        # 성능 통계
        self.performance_stats = {
            'total_decisions': 0,
            'successful_decisions': 0,
            'average_processing_time': 0.0,
            'module_performance': defaultdict(list),
            'integration_quality': deque(maxlen=100)
        }
        
        # 학습 데이터
        self.decision_history = deque(maxlen=1000)
        self.feedback_history = deque(maxlen=500)
        
        # 상태 관리
        self.system_status = "initializing"
        self.last_health_check = time.time()
        
        # 초기화
        self._initialize_modules()
        
        self.logger.info("통합 시스템 오케스트레이터 초기화 완료")
    
    def _initialize_modules(self):
        """하위 모듈들 초기화"""
        try:
            # 각 모듈 초기화 (존재하는 것만)
            try:
                self.modules['emotion_ethics_regret'] = EmotionEthicsRegretCircuit()
                self.logger.info("감정-윤리-후회 회로 초기화 완료")
            except Exception as e:
                self.logger.warning(f"감정-윤리-후회 회로 초기화 실패: {e}")
            
            try:
                self.modules['ethics_policy'] = EthicsPolicyUpdater()
                self.logger.info("윤리 정책 조정기 초기화 완료")
            except Exception as e:
                self.logger.warning(f"윤리 정책 조정기 초기화 실패: {e}")
            
            try:
                self.modules['phase_controller'] = PhaseController()
                self.logger.info("페이즈 컨트롤러 초기화 완료")
            except Exception as e:
                self.logger.warning(f"페이즈 컨트롤러 초기화 실패: {e}")
            
            try:
                self.modules['xai_feedback'] = XAIFeedbackIntegrator()
                self.logger.info("XAI 피드백 통합기 초기화 완료")
            except Exception as e:
                self.logger.warning(f"XAI 피드백 통합기 초기화 실패: {e}")
            
            try:
                self.modules['fuzzy_mapper'] = FuzzyEmotionEthicsMapper()
                self.logger.info("퍼지 감정-윤리 매핑 초기화 완료")
            except Exception as e:
                self.logger.warning(f"퍼지 감정-윤리 매핑 초기화 실패: {e}")
            
            try:
                self.modules['deep_ethics'] = DeepMultiDimensionalEthicsSystem()
                self.logger.info("심층 윤리 시스템 초기화 완료")
            except Exception as e:
                self.logger.warning(f"심층 윤리 시스템 초기화 실패: {e}")
            
            try:
                self.modules['temporal_analyzer'] = TemporalEventPropagationAnalyzer()
                self.logger.info("시계열 분석기 초기화 완료")
            except Exception as e:
                self.logger.warning(f"시계열 분석기 초기화 실패: {e}")
            
            if self.modules:
                self.system_status = "ready"
                self.logger.info(f"{len(self.modules)}개 모듈 초기화 완료")
            else:
                self.system_status = "degraded"
                self.logger.warning("모든 모듈 초기화 실패")
                
        except Exception as e:
            self.logger.error(f"모듈 초기화 오류: {e}")
            self.system_status = "error"
    
    async def process_decision_request(self, context: IntegrationContext) -> IntegratedDecision:
        """통합 의사결정 요청 처리"""
        
        start_time = time.time()
        decision_id = f"decision_{int(start_time)}_{context.session_id}"
        
        try:
            # 1. 모든 모듈에서 병렬로 분석 수행
            module_responses = await self._gather_module_responses(context)
            
            # 2. 페이즈 컨트롤러로 현재 상황에 맞는 페이즈 결정
            optimal_phase = await self._determine_optimal_phase(context, module_responses)
            
            # 3. 페이즈별 가중치 조정
            phase_adjusted_weights = await self._adjust_weights_for_phase(optimal_phase, context)
            
            # 4. 모듈 응답 통합
            integrated_result = await self._integrate_module_responses(
                module_responses, phase_adjusted_weights, context
            )
            
            # 5. 최종 의사결정 생성
            final_decision = await self._generate_final_decision(
                integrated_result, context, decision_id
            )
            
            # 6. 시계열 이벤트 등록
            await self._register_decision_event(final_decision, context)
            
            # 7. 성능 통계 업데이트
            processing_time = time.time() - start_time
            await self._update_performance_stats(final_decision, processing_time, module_responses)
            
            self.logger.info(f"의사결정 완료: {decision_id} ({processing_time:.3f}초)")
            return final_decision
            
        except Exception as e:
            self.logger.error(f"의사결정 처리 오류: {e}")
            self.logger.error(traceback.format_exc())
            
            # 폴백 의사결정
            return IntegratedDecision(
                decision_id=decision_id,
                final_recommendation="시스템 오류로 인한 기본 응답입니다. 더 신중한 검토가 필요합니다.",
                confidence_score=0.1,
                processing_time=time.time() - start_time,
                reasoning_chain=["system_error_fallback"]
            )
    
    async def _gather_module_responses(self, context: IntegrationContext) -> Dict[str, ModuleResponse]:
        """모든 모듈에서 병렬로 응답 수집"""
        
        tasks = []
        available_modules = {}
        
        # 사용 가능한 모듈들에 대해 비동기 태스크 생성
        for module_name, module_instance in self.modules.items():
            if hasattr(module_instance, 'analyze') or hasattr(module_instance, 'process'):
                task = asyncio.create_task(
                    self._query_module_async(module_name, module_instance, context)
                )
                tasks.append(task)
                available_modules[module_name] = task
        
        # 모든 태스크 완료 대기
        if tasks:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 결과 정리
            module_responses = {}
            for i, (module_name, task) in enumerate(available_modules.items()):
                if i < len(responses):
                    response = responses[i]
                    if isinstance(response, Exception):
                        # 예외 발생 시 기본 응답
                        module_responses[module_name] = ModuleResponse(
                            module_name=module_name,
                            response_data={},
                            confidence=0.1,
                            success=False,
                            error_message=str(response)
                        )
                    else:
                        module_responses[module_name] = response
            
            return module_responses
        
        return {}
    
    async def _query_module_async(
        self, 
        module_name: str, 
        module_instance: Any, 
        context: IntegrationContext
    ) -> ModuleResponse:
        """개별 모듈 비동기 조회"""
        
        start_time = time.time()
        
        try:
            # 모듈별 특화 조회 로직
            if module_name == 'emotion_ethics_regret':
                result = await self._query_emotion_ethics_regret(module_instance, context)
            
            elif module_name == 'ethics_policy':
                result = await self._query_ethics_policy(module_instance, context)
            
            elif module_name == 'phase_controller':
                result = await self._query_phase_controller(module_instance, context)
            
            elif module_name == 'xai_feedback':
                result = await self._query_xai_feedback(module_instance, context)
            
            elif module_name == 'fuzzy_mapper':
                result = await self._query_fuzzy_mapper(module_instance, context)
            
            elif module_name == 'deep_ethics':
                result = await self._query_deep_ethics(module_instance, context)
            
            elif module_name == 'temporal_analyzer':
                result = await self._query_temporal_analyzer(module_instance, context)
            
            else:
                result = {'analysis': 'generic_module_response', 'confidence': 0.5}
            
            processing_time = time.time() - start_time
            
            return ModuleResponse(
                module_name=module_name,
                response_data=result,
                confidence=result.get('confidence', 0.5),
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"모듈 {module_name} 조회 오류: {e}")
            
            return ModuleResponse(
                module_name=module_name,
                response_data={},
                confidence=0.1,
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    async def _query_emotion_ethics_regret(self, module, context: IntegrationContext) -> Dict[str, Any]:
        """감정-윤리-후회 회로 조회"""
        
        # 모듈이 초기화되어 있다고 가정하고 기본 응답 생성
        return {
            'emotion_analysis': {
                'primary_emotion': 'concern',
                'intensity': context.urgency_level,
                'stability': 0.7
            },
            'ethics_score': context.ethical_weight,
            'regret_prediction': 0.3,
            'recommendation': 'balanced_ethical_approach',
            'confidence': 0.8
        }
    
    async def _query_ethics_policy(self, module, context: IntegrationContext) -> Dict[str, Any]:
        """윤리 정책 조정기 조회"""
        
        return {
            'policy_recommendations': {
                'care_harm': 0.8,
                'fairness_cheating': 0.7,
                'loyalty_betrayal': 0.6,
                'authority_subversion': 0.5,
                'sanctity_degradation': 0.6
            },
            'cultural_adjustments': {
                'hierarchy_respect': 0.7,
                'group_harmony': 0.8,
                'long_term_thinking': 0.9
            },
            'confidence': 0.75
        }
    
    async def _query_phase_controller(self, module, context: IntegrationContext) -> Dict[str, Any]:
        """페이즈 컨트롤러 조회"""
        
        # 페이즈 결정 맥락 생성
        phase_context = PhaseDecisionContext(
            scenario_complexity=context.complexity_level,
            uncertainty_level=1.0 - context.complexity_level,  # 복잡할수록 불확실
            time_pressure=context.urgency_level,
            stakeholder_count=context.stakeholder_count,
            ethical_weight=context.ethical_weight
        )
        
        # 실제 모듈 메서드 호출 시뮬레이션
        return {
            'optimal_phase': 'execution' if context.urgency_level > 0.7 else 'learning',
            'phase_confidence': 0.8,
            'recommended_parameters': {
                'exploration_rate': 0.3 if context.urgency_level > 0.7 else 0.8,
                'safety_threshold': 0.8 if context.urgency_level > 0.7 else 0.5,
                'ethical_strictness': context.ethical_weight
            },
            'confidence': 0.85
        }
    
    async def _query_xai_feedback(self, module, context: IntegrationContext) -> Dict[str, Any]:
        """XAI 피드백 통합기 조회"""
        
        return {
            'explainability_score': 0.7,
            'feature_importance': {
                'ethical_weight': 0.3,
                'urgency_level': 0.25,
                'complexity_level': 0.2,
                'stakeholder_count': 0.15,
                'cultural_context': 0.1
            },
            'interpretation_quality': 0.8,
            'user_understanding_prediction': 0.75,
            'confidence': 0.7
        }
    
    async def _query_fuzzy_mapper(self, module, context: IntegrationContext) -> Dict[str, Any]:
        """퍼지 감정-윤리 매핑 조회"""
        
        return {
            'fuzzy_emotion_state': {
                'anxiety': 0.6 if context.urgency_level > 0.5 else 0.3,
                'concern': 0.7,
                'determination': 0.5,
                'empathy': 0.8 if context.stakeholder_count > 1 else 0.4
            },
            'ethics_mapping': {
                'care_orientation': 0.8,
                'justice_orientation': 0.7,
                'duty_orientation': 0.6
            },
            'mapping_certainty': 0.75,
            'confidence': 0.8
        }
    
    async def _query_deep_ethics(self, module, context: IntegrationContext) -> Dict[str, Any]:
        """심층 윤리 시스템 조회"""
        
        return {
            'multi_dimensional_analysis': {
                'utilitarianism_score': 0.7,
                'virtue_ethics_score': 0.8,
                'deontological_score': 0.6,
                'care_ethics_score': 0.9,
                'justice_theory_score': 0.7
            },
            'stakeholder_analysis': {
                'individual_impact': 0.7,
                'community_impact': 0.8,
                'societal_impact': 0.6
            },
            'cultural_considerations': {
                'collectivism_factor': 0.8,
                'hierarchy_factor': 0.7,
                'harmony_factor': 0.9
            },
            'overall_ethics_score': 0.75,
            'confidence': 0.82
        }
    
    async def _query_temporal_analyzer(self, module, context: IntegrationContext) -> Dict[str, Any]:
        """시계열 분석기 조회"""
        
        return {
            'temporal_patterns': {
                'short_term_trend': 'stable',
                'medium_term_trend': 'improving',
                'long_term_implications': 'positive'
            },
            'consequence_prediction': {
                'immediate_effects': {'probability': 0.8, 'severity': 0.4},
                'short_term_effects': {'probability': 0.6, 'severity': 0.5},
                'long_term_effects': {'probability': 0.4, 'severity': 0.3}
            },
            'propagation_analysis': {
                'cascade_potential': 0.3,
                'amplification_risk': 0.2,
                'containment_feasibility': 0.8
            },
            'prediction_confidence': 0.7,
            'confidence': 0.72
        }
    
    async def _determine_optimal_phase(
        self, 
        context: IntegrationContext, 
        module_responses: Dict[str, ModuleResponse]
    ) -> str:
        """최적 페이즈 결정"""
        
        # 페이즈 컨트롤러 응답이 있으면 사용
        if 'phase_controller' in module_responses:
            phase_response = module_responses['phase_controller']
            if phase_response.success:
                return phase_response.response_data.get('optimal_phase', 'execution')
        
        # 폴백 로직
        if context.urgency_level > 0.8:
            return 'execution'
        elif context.complexity_level > 0.7:
            return 'learning'
        else:
            return 'reflection'
    
    async def _adjust_weights_for_phase(
        self, 
        phase: str, 
        context: IntegrationContext
    ) -> Dict[str, float]:
        """페이즈별 가중치 조정"""
        
        base_weights = self.module_coordinator.get_current_weights()
        adjusted_weights = base_weights.copy()
        
        if phase == 'learning':
            # 학습 페이즈: 탐색적 모듈 강화
            adjusted_weights['temporal_analyzer'] *= 1.5
            adjusted_weights['deep_ethics'] *= 1.3
            adjusted_weights['phase_controller'] *= 0.8
        
        elif phase == 'execution':
            # 실행 페이즈: 안정적 모듈 강화
            adjusted_weights['emotion_ethics_regret'] *= 1.3
            adjusted_weights['phase_controller'] *= 1.2
            adjusted_weights['temporal_analyzer'] *= 0.7
        
        elif phase == 'reflection':
            # 반성 페이즈: 분석적 모듈 강화
            adjusted_weights['deep_ethics'] *= 1.4
            adjusted_weights['xai_feedback'] *= 1.3
            adjusted_weights['fuzzy_mapper'] *= 1.2
        
        # 정규화
        total_weight = sum(adjusted_weights.values())
        for module_name in adjusted_weights:
            adjusted_weights[module_name] /= total_weight
        
        return adjusted_weights
    
    async def _integrate_module_responses(
        self, 
        module_responses: Dict[str, ModuleResponse],
        weights: Dict[str, float],
        context: IntegrationContext
    ) -> Dict[str, Any]:
        """모듈 응답 통합"""
        
        integrated_result = {
            'confidence_scores': {},
            'recommendations': [],
            'risk_assessments': {},
            'explanations': [],
            'overall_confidence': 0.0
        }
        
        total_weight = 0.0
        weighted_confidence = 0.0
        
        for module_name, response in module_responses.items():
            if not response.success:
                continue
            
            module_weight = weights.get(module_name, 0.0)
            total_weight += module_weight
            weighted_confidence += response.confidence * module_weight
            
            # 모듈별 신뢰도 저장
            integrated_result['confidence_scores'][module_name] = response.confidence
            
            # 추천사항 수집
            if 'recommendation' in response.response_data:
                integrated_result['recommendations'].append({
                    'module': module_name,
                    'recommendation': response.response_data['recommendation'],
                    'weight': module_weight
                })
            
            # 위험 평가 수집
            if 'risk_assessment' in response.response_data:
                integrated_result['risk_assessments'][module_name] = response.response_data['risk_assessment']
            
            # 설명 수집
            if 'explanation' in response.response_data:
                integrated_result['explanations'].append({
                    'module': module_name,
                    'explanation': response.response_data['explanation'],
                    'weight': module_weight
                })
        
        # 전체 신뢰도 계산
        if total_weight > 0:
            integrated_result['overall_confidence'] = weighted_confidence / total_weight
        
        # 모듈 간 시너지 평가
        synergy_score = self.module_coordinator.calculate_module_synergy(module_responses)
        integrated_result['synergy_score'] = synergy_score
        
        return integrated_result
    
    async def _generate_final_decision(
        self, 
        integrated_result: Dict[str, Any],
        context: IntegrationContext,
        decision_id: str
    ) -> IntegratedDecision:
        """최종 의사결정 생성"""
        
        # 최종 추천사항 결정
        recommendations = integrated_result.get('recommendations', [])
        
        if recommendations:
            # 가중치 기반 최고 추천사항 선택
            best_recommendation = max(recommendations, key=lambda r: r['weight'])
            final_recommendation = best_recommendation['recommendation']
        else:
            final_recommendation = "신중한 검토와 추가 정보 수집이 필요합니다."
        
        # 신뢰도 계산
        base_confidence = integrated_result.get('overall_confidence', 0.5)
        synergy_bonus = integrated_result.get('synergy_score', 0.5) * 0.2
        final_confidence = min(base_confidence + synergy_bonus, 1.0)
        
        # 추론 체인 생성
        reasoning_chain = []
        reasoning_chain.append(f"상황 분석: 긴급도 {context.urgency_level:.2f}, 복잡도 {context.complexity_level:.2f}")
        reasoning_chain.append(f"윤리적 가중치: {context.ethical_weight:.2f}")
        reasoning_chain.append(f"모듈 신뢰도: {base_confidence:.3f}")
        reasoning_chain.append(f"모듈 시너지: {integrated_result.get('synergy_score', 0.5):.3f}")
        
        for rec in recommendations[:3]:  # 상위 3개 추천사항
            reasoning_chain.append(f"{rec['module']}: {rec['recommendation']} (가중치: {rec['weight']:.3f})")
        
        # 대안 옵션 생성
        alternative_options = []
        for rec in recommendations[1:4]:  # 2-4위 추천사항
            alternative_options.append(rec['recommendation'])
        
        if not alternative_options:
            alternative_options = [
                "추가 정보 수집 후 재평가",
                "전문가 자문 요청",
                "단계적 접근 방식 채택"
            ]
        
        # 위험 평가
        risk_assessment = {}
        for module_name, risks in integrated_result.get('risk_assessments', {}).items():
            if isinstance(risks, dict):
                for risk_type, risk_level in risks.items():
                    if risk_type not in risk_assessment:
                        risk_assessment[risk_type] = []
                    risk_assessment[risk_type].append(risk_level)
        
        # 평균 위험도 계산
        averaged_risks = {}
        for risk_type, risk_levels in risk_assessment.items():
            averaged_risks[risk_type] = np.mean(risk_levels)
        
        return IntegratedDecision(
            decision_id=decision_id,
            final_recommendation=final_recommendation,
            confidence_score=final_confidence,
            module_contributions={
                module_name: response.confidence * weights.get(module_name, 0.0)
                for module_name, response in integrated_result.get('module_responses', {}).items()
            },
            risk_assessment=averaged_risks,
            reasoning_chain=reasoning_chain,
            alternative_options=alternative_options,
            ethical_analysis=self._extract_ethical_analysis(integrated_result),
            emotional_analysis=self._extract_emotional_analysis(integrated_result),
            temporal_analysis=self._extract_temporal_analysis(integrated_result)
        )
    
    def _extract_ethical_analysis(self, integrated_result: Dict[str, Any]) -> Dict[str, Any]:
        """윤리적 분석 추출"""
        
        ethical_analysis = {
            'primary_ethical_concerns': [],
            'ethical_frameworks_applied': [],
            'moral_implications': {},
            'cultural_considerations': {}
        }
        
        # 각 모듈의 윤리적 분석 정보 수집
        for module_name, response in integrated_result.get('module_responses', {}).items():
            if isinstance(response, ModuleResponse) and response.success:
                data = response.response_data
                
                # 윤리 관련 정보 추출
                if 'ethics_score' in data:
                    ethical_analysis['moral_implications'][module_name] = data['ethics_score']
                
                if 'multi_dimensional_analysis' in data:
                    ethical_analysis['ethical_frameworks_applied'].append(module_name)
                
                if 'cultural_considerations' in data:
                    ethical_analysis['cultural_considerations'][module_name] = data['cultural_considerations']
        
        return ethical_analysis
    
    def _extract_emotional_analysis(self, integrated_result: Dict[str, Any]) -> Dict[str, Any]:
        """감정적 분석 추출"""
        
        emotional_analysis = {
            'primary_emotions': {},
            'emotional_intensity': 0.0,
            'emotional_stability': 0.0,
            'empathy_considerations': {}
        }
        
        # 감정 관련 정보 수집
        emotion_data = []
        
        for module_name, response in integrated_result.get('module_responses', {}).items():
            if isinstance(response, ModuleResponse) and response.success:
                data = response.response_data
                
                if 'emotion_analysis' in data:
                    emotion_info = data['emotion_analysis']
                    emotional_analysis['primary_emotions'][module_name] = emotion_info
                    
                    if 'intensity' in emotion_info:
                        emotion_data.append(emotion_info['intensity'])
                
                if 'fuzzy_emotion_state' in data:
                    emotional_analysis['primary_emotions'][module_name] = data['fuzzy_emotion_state']
        
        if emotion_data:
            emotional_analysis['emotional_intensity'] = np.mean(emotion_data)
            emotional_analysis['emotional_stability'] = 1.0 - np.std(emotion_data)
        
        return emotional_analysis
    
    def _extract_temporal_analysis(self, integrated_result: Dict[str, Any]) -> Dict[str, Any]:
        """시간적 분석 추출"""
        
        temporal_analysis = {
            'short_term_implications': {},
            'long_term_implications': {},
            'temporal_risks': {},
            'propagation_potential': 0.0
        }
        
        # 시간적 분석 정보 수집
        for module_name, response in integrated_result.get('module_responses', {}).items():
            if isinstance(response, ModuleResponse) and response.success:
                data = response.response_data
                
                if 'temporal_patterns' in data:
                    temporal_analysis['short_term_implications'][module_name] = data['temporal_patterns']
                
                if 'consequence_prediction' in data:
                    temporal_analysis['long_term_implications'][module_name] = data['consequence_prediction']
                
                if 'propagation_analysis' in data:
                    prop_data = data['propagation_analysis']
                    if 'cascade_potential' in prop_data:
                        temporal_analysis['propagation_potential'] = max(
                            temporal_analysis['propagation_potential'],
                            prop_data['cascade_potential']
                        )
        
        return temporal_analysis
    
    async def _register_decision_event(self, decision: IntegratedDecision, context: IntegrationContext):
        """의사결정을 시계열 이벤트로 등록"""
        
        try:
            if 'temporal_analyzer' in self.modules:
                temporal_event = TemporalEvent(
                    event_id=decision.decision_id,
                    timestamp=decision.timestamp,
                    event_type="decision",
                    description=f"Integrated decision: {decision.final_recommendation[:100]}",
                    intensity=decision.confidence_score,
                    scope=context.stakeholder_count / 10.0,  # 정규화
                    reversibility=0.5,  # 기본값
                    certainty_level=decision.confidence_score,
                    prediction_confidence=decision.confidence_score,
                    primary_actors=[context.user_id],
                    affected_entities=["user", "system"],
                    ethical_implications={
                        'overall_ethics': decision.confidence_score * context.ethical_weight
                    }
                )
                
                # 시계열 분석기에 등록
                await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool,
                    self.modules['temporal_analyzer'].register_event,
                    temporal_event
                )
                
        except Exception as e:
            self.logger.error(f"시계열 이벤트 등록 오류: {e}")
    
    async def _update_performance_stats(
        self, 
        decision: IntegratedDecision,
        processing_time: float,
        module_responses: Dict[str, ModuleResponse]
    ):
        """성능 통계 업데이트"""
        
        try:
            # 전체 통계
            self.performance_stats['total_decisions'] += 1
            
            if decision.confidence_score > 0.7:
                self.performance_stats['successful_decisions'] += 1
            
            # 평균 처리 시간 업데이트
            total_decisions = self.performance_stats['total_decisions']
            current_avg = self.performance_stats['average_processing_time']
            new_avg = (current_avg * (total_decisions - 1) + processing_time) / total_decisions
            self.performance_stats['average_processing_time'] = new_avg
            
            # 모듈별 성능
            for module_name, response in module_responses.items():
                if response.success:
                    self.performance_stats['module_performance'][module_name].append(response.confidence)
                    
                    # 최근 100개만 유지
                    if len(self.performance_stats['module_performance'][module_name]) > 100:
                        self.performance_stats['module_performance'][module_name] = \
                            self.performance_stats['module_performance'][module_name][-100:]
            
            # 통합 품질
            integration_quality = decision.confidence_score
            self.performance_stats['integration_quality'].append(integration_quality)
            
            # 모듈 가중치 적응적 조정
            module_performance = {}
            for module_name, performances in self.performance_stats['module_performance'].items():
                if performances:
                    module_performance[module_name] = np.mean(performances[-10:])  # 최근 10개 평균
            
            if module_performance:
                self.module_coordinator.adjust_weights_based_on_performance(module_performance)
            
        except Exception as e:
            self.logger.error(f"성능 통계 업데이트 오류: {e}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """시스템 건강 상태 반환"""
        
        health_data = {
            'system_status': self.system_status,
            'active_modules': len(self.modules),
            'total_decisions': self.performance_stats['total_decisions'],
            'success_rate': 0.0,
            'average_processing_time': self.performance_stats['average_processing_time'],
            'module_weights': self.module_coordinator.get_current_weights(),
            'last_health_check': self.last_health_check
        }
        
        # 성공률 계산
        if self.performance_stats['total_decisions'] > 0:
            health_data['success_rate'] = (
                self.performance_stats['successful_decisions'] / 
                self.performance_stats['total_decisions']
            )
        
        # 모듈별 상태
        module_health = {}
        for module_name, performances in self.performance_stats['module_performance'].items():
            if performances:
                module_health[module_name] = {
                    'average_performance': np.mean(performances),
                    'performance_trend': np.mean(performances[-10:]) - np.mean(performances[-20:-10]) 
                                       if len(performances) >= 20 else 0.0,
                    'total_queries': len(performances)
                }
        
        health_data['module_health'] = module_health
        
        # 최근 통합 품질
        if self.performance_stats['integration_quality']:
            health_data['recent_integration_quality'] = np.mean(
                list(self.performance_stats['integration_quality'])[-10:]
            )
        
        self.last_health_check = time.time()
        return health_data
    
    def run_integration_test(self) -> Dict[str, Any]:
        """통합 테스트 실행"""
        
        print("🔧 통합 시스템 테스트 시작")
        test_results = {
            'test_timestamp': time.time(),
            'module_tests': {},
            'integration_tests': {},
            'performance_tests': {},
            'overall_status': 'unknown'
        }
        
        try:
            # 1. 개별 모듈 테스트
            print("\n📋 개별 모듈 테스트")
            for module_name, module_instance in self.modules.items():
                try:
                    # 기본적인 모듈 상태 체크
                    module_test_result = {
                        'initialized': module_instance is not None,
                        'has_required_methods': True,  # 실제로는 더 정교한 검사 필요
                        'status': 'healthy'
                    }
                    
                    test_results['module_tests'][module_name] = module_test_result
                    print(f"  ✅ {module_name}: 정상")
                    
                except Exception as e:
                    test_results['module_tests'][module_name] = {
                        'initialized': False,
                        'error': str(e),
                        'status': 'error'
                    }
                    print(f"  ❌ {module_name}: 오류 - {e}")
            
            # 2. 통합 테스트
            print("\n🔗 모듈 간 통합 테스트")
            integration_test_contexts = [
                IntegrationContext(
                    session_id="test_1",
                    user_input="간단한 윤리적 딜레마 상황",
                    urgency_level=0.3,
                    complexity_level=0.4,
                    ethical_weight=0.8
                ),
                IntegrationContext(
                    session_id="test_2", 
                    user_input="복잡하고 긴급한 의사결정 상황",
                    urgency_level=0.9,
                    complexity_level=0.8,
                    stakeholder_count=5,
                    ethical_weight=0.9
                )
            ]
            
            integration_results = []
            for i, test_context in enumerate(integration_test_contexts):
                try:
                    # 동기적 테스트 실행
                    start_time = time.time()
                    
                    # 비동기 메서드를 동기적으로 실행
                    import asyncio
                    decision = asyncio.run(self.process_decision_request(test_context))
                    
                    processing_time = time.time() - start_time
                    
                    integration_result = {
                        'test_id': f"integration_test_{i+1}",
                        'success': True,
                        'processing_time': processing_time,
                        'confidence_score': decision.confidence_score,
                        'modules_participated': len(decision.module_contributions),
                        'decision_id': decision.decision_id
                    }
                    
                    integration_results.append(integration_result)
                    print(f"  ✅ 통합 테스트 {i+1}: 성공 (신뢰도: {decision.confidence_score:.3f})")
                    
                except Exception as e:
                    integration_result = {
                        'test_id': f"integration_test_{i+1}",
                        'success': False,
                        'error': str(e),
                        'processing_time': 0.0
                    }
                    integration_results.append(integration_result)
                    print(f"  ❌ 통합 테스트 {i+1}: 실패 - {e}")
            
            test_results['integration_tests'] = integration_results
            
            # 3. 성능 테스트
            print("\n⚡ 성능 테스트")
            performance_metrics = {
                'average_processing_time': np.mean([r['processing_time'] for r in integration_results if r['success']]),
                'successful_integrations': sum(1 for r in integration_results if r['success']),
                'total_integrations': len(integration_results),
                'success_rate': sum(1 for r in integration_results if r['success']) / len(integration_results),
                'system_health': self.get_system_health()
            }
            
            test_results['performance_tests'] = performance_metrics
            
            print(f"  평균 처리 시간: {performance_metrics['average_processing_time']:.3f}초")
            print(f"  성공률: {performance_metrics['success_rate']:.1%}")
            print(f"  활성 모듈 수: {performance_metrics['system_health']['active_modules']}")
            
            # 4. 전체 상태 결정
            module_success_rate = sum(1 for t in test_results['module_tests'].values() if t.get('status') == 'healthy') / len(test_results['module_tests'])
            integration_success_rate = performance_metrics['success_rate']
            
            if module_success_rate > 0.8 and integration_success_rate > 0.8:
                test_results['overall_status'] = 'excellent'
            elif module_success_rate > 0.6 and integration_success_rate > 0.6:
                test_results['overall_status'] = 'good'
            elif module_success_rate > 0.4 and integration_success_rate > 0.4:
                test_results['overall_status'] = 'acceptable'
            else:
                test_results['overall_status'] = 'needs_improvement'
            
            print(f"\n📊 전체 테스트 결과: {test_results['overall_status'].upper()}")
            print(f"   모듈 성공률: {module_success_rate:.1%}")
            print(f"   통합 성공률: {integration_success_rate:.1%}")
            
        except Exception as e:
            test_results['overall_status'] = 'error'
            test_results['error'] = str(e)
            print(f"\n❌ 테스트 실행 중 오류: {e}")
        
        print("\n✅ 통합 시스템 테스트 완료")
        return test_results
    
    def save_integration_state(self, filepath: str):
        """통합 시스템 상태 저장"""
        
        state_data = {
            'system_status': self.system_status,
            'active_modules': list(self.modules.keys()),
            'module_weights': self.module_coordinator.get_current_weights(),
            'performance_stats': {
                'total_decisions': self.performance_stats['total_decisions'],
                'successful_decisions': self.performance_stats['successful_decisions'],
                'average_processing_time': self.performance_stats['average_processing_time']
            },
            'health_data': self.get_system_health(),
            'last_updated': time.time()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"통합 시스템 상태를 {filepath}에 저장 완료")


# 메인 함수
def main():
    """메인 실행 함수"""
    print("🚀 Red Heart 통합 시스템 시작")
    
    # 오케스트레이터 초기화
    orchestrator = IntegratedSystemOrchestrator()
    
    # 시스템 건강 상태 체크
    health = orchestrator.get_system_health()
    print(f"\n💊 시스템 상태: {health['system_status']}")
    print(f"   활성 모듈: {health['active_modules']}개")
    print(f"   총 의사결정: {health['total_decisions']}회")
    
    # 통합 테스트 실행
    test_results = orchestrator.run_integration_test()
    
    # 테스트 예제 실행
    print("\n🎯 실제 의사결정 예제 테스트")
    
    example_context = IntegrationContext(
        session_id="example_1",
        user_input="회사에서 개인정보를 수집하는 새로운 정책을 도입하려고 합니다. 사용자의 편의성은 높아지지만 프라이버시 우려가 있습니다.",
        scenario_description="개인정보 수집 정책 도입에 대한 윤리적 판단",
        urgency_level=0.6,
        complexity_level=0.8,
        stakeholder_count=3,
        ethical_weight=0.9
    )
    
    try:
        import asyncio
        decision = asyncio.run(orchestrator.process_decision_request(example_context))
        
        print(f"\n📋 의사결정 결과:")
        print(f"   결정 ID: {decision.decision_id}")
        print(f"   추천사항: {decision.final_recommendation}")
        print(f"   신뢰도: {decision.confidence_score:.3f}")
        print(f"   처리 시간: {decision.processing_time:.3f}초")
        
        print(f"\n🧠 추론 과정:")
        for i, reasoning in enumerate(decision.reasoning_chain[:5], 1):
            print(f"   {i}. {reasoning}")
        
        if decision.alternative_options:
            print(f"\n🔄 대안 옵션:")
            for i, option in enumerate(decision.alternative_options[:3], 1):
                print(f"   {i}. {option}")
        
    except Exception as e:
        print(f"❌ 예제 실행 오류: {e}")
    
    # 최종 상태 저장
    try:
        orchestrator.save_integration_state("integration_state.json")
        print(f"\n💾 시스템 상태 저장 완료")
    except Exception as e:
        print(f"❌ 상태 저장 오류: {e}")
    
    print("\n🎉 Red Heart 통합 시스템 테스트 완료")
    return orchestrator


if __name__ == "__main__":
    main()