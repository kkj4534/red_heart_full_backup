"""
감정-윤리-후회 삼각 회로 (Emotion-Ethics-Regret Triangle Circuit)
인간적 윤리 판단 과정을 모델링한 유기적 상호작용 시스템

핵심 원칙:
1. 감정 우선순위: 공동체 > 타자 > 자아 (치명적 손실 시 우선순위 역전)
2. 윤리적 추론: 감정을 바탕으로 한 가치 판단
3. 후회는 학습: 직접 개입 아닌 미묘한 편향으로 작용
4. 손실 억제 우선: 기쁨보다 슬픔을 우선시 (영구 손실 원리)
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import time
from pathlib import Path

# 고급 모듈 필수 로드 (의존성 문제 해결 후)
from advanced_bentham_calculator import AdvancedBenthamCalculator
from advanced_emotion_analyzer import AdvancedEmotionAnalyzer  
from advanced_regret_analyzer import AdvancedRegretAnalyzer
ADVANCED_MODULES_AVAILABLE = True
from data_models import EmotionData, EmotionState, EmotionIntensity

logger = logging.getLogger('RedHeart.EmotionEthicsRegretCircuit')

@dataclass
class CircuitDecisionContext:
    """회로 의사결정 맥락"""
    scenario_text: str
    proposed_action: str
    
    # 다층 감정 입력
    community_emotion: Optional[EmotionData] = None
    other_emotion: Optional[EmotionData] = None
    self_emotion: Optional[EmotionData] = None
    
    # 맥락 정보
    stakeholders: List[str] = None
    social_context: Dict[str, Any] = None
    temporal_urgency: float = 0.5
    
    # 과거 경험
    past_regret_memory: Optional[Dict[str, float]] = None
    similar_decisions_history: List[Dict] = None

@dataclass
class CircuitDecisionResult:
    """회로 의사결정 결과"""
    final_ethical_score: float
    confidence: float
    
    # 단계별 결과
    integrated_emotion: EmotionData
    ethical_values: Dict[str, float]
    bentham_result: Any  # EnhancedHedonicResult
    predicted_regret: Dict[str, float]
    
    # 메타 정보
    critical_loss_detected: bool
    emotion_conflict_resolved: str
    reasoning_trace: List[str]
    processing_time: float

class EmotionEthicsRegretCircuit:
    """감정-윤리-후회 삼각 회로 관리자"""
    
    def __init__(self):
        """회로 초기화"""
        self.logger = logger
        
        # 핵심 모듈들 (고급 모듈 우선)
        if ADVANCED_MODULES_AVAILABLE:
            self.emotion_analyzer = AdvancedEmotionAnalyzer()
            self.bentham_calculator = AdvancedBenthamCalculator()
            self.regret_analyzer = AdvancedRegretAnalyzer()
            print("✅ 고급 모듈들로 회로 초기화 완료")
        else:
            raise RuntimeError("고급 모듈들이 필요합니다. 기본 모듈로는 시스템이 불완전합니다.")
        
        # 경험 데이터베이스 시스템 초기화
        try:
            from advanced_experience_database import AdvancedExperienceDatabase
            self.experience_db = AdvancedExperienceDatabase()
            self.experience_enabled = True
            print("✅ 경험 데이터베이스 시스템 초기화 완료")
        except ImportError as e:
            print(f"⚠️ 경험 데이터베이스 시스템 없음: {e}")
            self.experience_db = None
            self.experience_enabled = False
        
        # 회로 상태
        self.decision_history = []
        self.learning_memory = {
            'regret_patterns': {},
            'successful_decisions': {},
            'emotion_adaptations': {}
        }
        
        # 성능 추적
        self.performance_metrics = {
            'total_decisions': 0,
            'average_processing_time': 0.0,
            'emotion_conflict_rate': 0.0,
            'critical_loss_rate': 0.0
        }
        
        self.logger.info("감정-윤리-후회 삼각 회로 초기화 완료")
    
    async def _try_experience_based_decision(self, 
                                           context: CircuitDecisionContext,
                                           reasoning_trace: List[str]) -> Optional[CircuitDecisionResult]:
        """경험 기반 의사결정 시도"""
        
        if not self.experience_enabled:
            reasoning_trace.append("경험 시스템 비활성화됨")
            return None
        
        try:
            # 현재 상황을 경험 검색 쿼리로 변환
            from advanced_experience_database import ExperienceQuery
            
            query = ExperienceQuery(
                query_text=f"{context.scenario_text} {context.proposed_action}",
                category_filter="ethical_decision",
                similarity_threshold=0.75,  # 높은 유사도 요구
                max_results=5,
                boost_recent=True
            )
            
            # 유사 경험 검색
            similar_experiences = await self.experience_db.search_experiences(query)
            
            if not similar_experiences or len(similar_experiences) == 0:
                reasoning_trace.append("유사 경험 없음 - 검색 결과 0건")
                return None
            
            # 최고 유사도 경험 확인
            best_experience = similar_experiences[0]
            if best_experience['similarity'] < 0.8:
                reasoning_trace.append(f"유사도 부족 ({best_experience['similarity']:.3f} < 0.8)")
                return None
            
            reasoning_trace.append(
                f"유사 경험 발견: {len(similar_experiences)}건, "
                f"최고 유사도: {best_experience['similarity']:.3f}"
            )
            
            # 경험 기반 의사결정 실행
            return await self._make_experience_based_decision(
                context, similar_experiences, reasoning_trace
            )
            
        except Exception as e:
            reasoning_trace.append(f"경험 검색 실패: {e}")
            return None
    
    async def _make_experience_based_decision(self,
                                           context: CircuitDecisionContext,
                                           similar_experiences: List[Dict],
                                           reasoning_trace: List[str]) -> CircuitDecisionResult:
        """유사 경험을 바탕으로 의사결정"""
        
        start_time = time.time()
        
        # 경험들로부터 패턴 추출
        ethical_patterns = []
        regret_patterns = []
        confidence_scores = []
        
        for exp in similar_experiences:
            if 'ethical_score' in exp['metadata']:
                ethical_patterns.append(exp['metadata']['ethical_score'])
            if 'regret_score' in exp['metadata']:
                regret_patterns.append(exp['metadata']['regret_score'])
            confidence_scores.append(exp['similarity'])
        
        # 가중 평균 계산 (유사도 기반 가중치)
        if ethical_patterns:
            weighted_ethical_score = np.average(ethical_patterns, weights=confidence_scores[:len(ethical_patterns)])
        else:
            # 경험에 윤리 점수가 없으면 중간값
            weighted_ethical_score = 0.5
        
        if regret_patterns:
            weighted_regret_score = np.average(regret_patterns, weights=confidence_scores[:len(regret_patterns)])
        else:
            weighted_regret_score = 0.3  # 기본 후회 수준
        
        # 경험 기반 신뢰도 계산
        experience_confidence = np.mean(confidence_scores)
        
        # 간단한 감정 분석 (경험 기반이므로 빠른 처리)
        basic_emotion = self.emotion_analyzer.analyze_emotion(
            f"{context.scenario_text} {context.proposed_action}", 
            language='ko'
        )
        
        reasoning_trace.append(
            f"경험 기반 점수: 윤리={weighted_ethical_score:.3f}, "
            f"후회={weighted_regret_score:.3f}, 신뢰도={experience_confidence:.3f}"
        )
        
        # 경험 기반 결과 생성
        return CircuitDecisionResult(
            final_ethical_score=weighted_ethical_score,
            confidence=experience_confidence,
            integrated_emotion=basic_emotion,
            ethical_values={
                'care_harm': weighted_ethical_score,
                'fairness': weighted_ethical_score * 0.9,
                'loyalty': weighted_ethical_score * 0.8
            },
            bentham_result=None,  # 경험 기반에서는 간소화
            predicted_regret={
                'anticipated_regret': weighted_regret_score,
                'regret_intensity': weighted_regret_score * 0.8,
                'confidence': experience_confidence
            },
            critical_loss_detected=False,
            emotion_conflict_resolved='experience_based',
            reasoning_trace=reasoning_trace,
            processing_time=time.time() - start_time
        )

    async def process_ethical_decision(self, 
                                     context: CircuitDecisionContext) -> CircuitDecisionResult:
        """인간적 윤리 판단 과정을 통한 의사결정"""
        
        start_time = time.time()
        reasoning_trace = []
        
        try:
            # 0단계: 경험 기반 의사결정 시도
            reasoning_trace.append("0단계: 경험 데이터베이스 검색 시작")
            experience_result = await self._try_experience_based_decision(context, reasoning_trace)
            
            if experience_result is not None:
                # 유사 경험 발견 - 경험 기반 의사결정
                reasoning_trace.append("✅ 유사 경험 발견 - 경험 기반 의사결정 적용")
                return experience_result
            else:
                # 경험 없음 - 사고실험 모드로 전환
                reasoning_trace.append("💭 유사 경험 없음 - 사고실험 모드 활성화")
            
            # 1단계: 깊이 있는 사고실험 모드 - 다각도 관점 분석
            reasoning_trace.append("1단계: 심층 사고실험 시작 - 다각도 관점 분석")
            stakeholder_perspectives = await self._analyze_stakeholder_perspectives(context, reasoning_trace)
            
            # 2단계: 반사실적 시나리오 탐구
            reasoning_trace.append("2단계: 반사실적 시나리오 심층 탐구")
            counterfactual_scenarios = await self._explore_counterfactual_scenarios(context, reasoning_trace)
            
            # 3단계: 다층 감정 분석 및 통합
            reasoning_trace.append("3단계: 이해관계자별 감정 분석 및 통합")
            integrated_emotion, emotion_meta = await self._analyze_and_integrate_emotions(
                context, reasoning_trace, stakeholder_perspectives
            )
            
            # 4단계: 윤리적 가치 추론 (반사실적 시나리오 반영)
            reasoning_trace.append("4단계: 윤리적 가치 추론 (반사실적 분석 반영)")
            ethical_values = await self._perform_ethical_reasoning(
                integrated_emotion, context, reasoning_trace, counterfactual_scenarios
            )
            
            # 5단계: 벤담 계산 (윤리적 가치 반영)
            reasoning_trace.append("5단계: 윤리적 벤담 계산 시작")
            bentham_result = await self._calculate_ethical_bentham(
                integrated_emotion, ethical_values, context, reasoning_trace
            )
            
            # 6단계: 후회 예측 및 학습 편향 추출
            reasoning_trace.append("6단계: 후회 예측 및 학습 편향 시작")
            predicted_regret, learning_insights = await self._predict_regret_and_learning(
                context, bentham_result, reasoning_trace, counterfactual_scenarios
            )
            
            # 7단계: 결과 통합 및 신뢰도 계산
            reasoning_trace.append("7단계: 결과 통합 및 평가")
            final_result = self._integrate_final_result(
                integrated_emotion, ethical_values, bentham_result, 
                predicted_regret, emotion_meta, reasoning_trace, start_time
            )
            
            # 6단계: 학습 메모리 업데이트 (결과 저장 및 경험 축적)
            await self._update_learning_memory(context, final_result)
            
            # 7단계: 경험 데이터베이스에 저장 (미래 참조용)
            await self._store_experience_for_future(context, final_result)
            
            # 성능 메트릭 업데이트
            self._update_performance_metrics(final_result, emotion_meta)
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"윤리적 의사결정 처리 실패: {e}")
            raise RuntimeError(f"윤리적 의사결정 시스템 실패: {e}. 폴백 없이 명확한 실패.")
    
    async def _analyze_and_integrate_emotions(self, 
                                            context: CircuitDecisionContext,
                                            reasoning_trace: List[str]) -> Tuple[EmotionData, Dict]:
        """다층 감정 분석 및 통합"""
        
        emotion_meta = {
            'critical_loss_detected': False,
            'emotion_conflict_type': 'none',
            'emotion_sources_used': []
        }
        
        # 텍스트 기반 감정 분석 (자아 감정으로 사용)
        if not context.self_emotion:
            combined_text = f"{context.scenario_text} {context.proposed_action}"
            context.self_emotion = self.emotion_analyzer.analyze_emotion(
                combined_text, language='ko'
            )
            reasoning_trace.append(f"자아 감정 분석 완료: {context.self_emotion.primary_emotion.value}")
        
        # 공동체 감정 추론 (사회적 맥락 기반)
        if not context.community_emotion and context.social_context:
            context.community_emotion = await self._infer_community_emotion(
                context, reasoning_trace
            )
        
        # 타자 감정 추론 (이해관계자 기반)
        if not context.other_emotion and context.stakeholders:
            context.other_emotion = await self._infer_other_emotion(
                context, reasoning_trace
            )
        
        # 고급 감정 통합 (벤담 계산기의 메서드 활용)
        integrated_emotion = self.bentham_calculator._integrate_emotion_hierarchy(
            context.community_emotion, context.other_emotion, context.self_emotion
        )
        
        # 메타 정보 수집
        emotion_meta['emotion_sources_used'] = [
            'community' if context.community_emotion else None,
            'other' if context.other_emotion else None,
            'self' if context.self_emotion else None
        ]
        emotion_meta['emotion_sources_used'] = [x for x in emotion_meta['emotion_sources_used'] if x]
        
        # 치명적 손실 탐지
        critical_loss = self.bentham_calculator._detect_critical_emotional_loss(
            context.community_emotion, context.other_emotion, context.self_emotion
        )
        emotion_meta['critical_loss_detected'] = critical_loss['any_critical']
        
        if emotion_meta['critical_loss_detected']:
            reasoning_trace.append("⚠️ 치명적 감정 손실 탐지됨 - 손실 억제 모드 활성화")
            
        reasoning_trace.append(
            f"감정 통합 완료: {integrated_emotion.primary_emotion.value} "
            f"(출처: {', '.join(emotion_meta['emotion_sources_used'])})"
        )
        
        return integrated_emotion, emotion_meta
    
    async def _infer_community_emotion(self, 
                                     context: CircuitDecisionContext,
                                     reasoning_trace: List[str]) -> EmotionData:
        """사회적 맥락을 기반으로 공동체 감정 추론"""
        
        # 사회적 맥락 키워드 분석
        social_keywords = context.social_context.get('keywords', [])
        impact_scope = context.social_context.get('impact_scope', 'individual')
        
        # 공동체 관심사 매핑
        community_concern_mapping = {
            'safety': EmotionState.FEAR,
            'injustice': EmotionState.ANGER, 
            'loss': EmotionState.SADNESS,
            'celebration': EmotionState.JOY,
            'uncertainty': EmotionState.FEAR,
            'achievement': EmotionState.JOY
        }
        
        # 기본 공동체 감정
        community_emotion = EmotionState.NEUTRAL
        intensity = EmotionIntensity.MODERATE
        
        # 키워드 기반 감정 추론
        for keyword in social_keywords:
            if keyword in community_concern_mapping:
                community_emotion = community_concern_mapping[keyword]
                break
        
        # 영향 범위에 따른 강도 조정
        if impact_scope in ['society', 'national']:
            if community_emotion in [EmotionState.FEAR, EmotionState.SADNESS]:
                intensity = EmotionIntensity.VERY_STRONG
            elif community_emotion == EmotionState.JOY:
                intensity = EmotionIntensity.STRONG
        
        reasoning_trace.append(f"공동체 감정 추론: {community_emotion.value} (범위: {impact_scope})")
        
        return EmotionData(
            primary_emotion=community_emotion,
            intensity=intensity,
            confidence=0.7,
            language='ko',
            processing_method='community_inference',
            dominance=0.5
        )
    
    async def _infer_other_emotion(self, 
                                 context: CircuitDecisionContext,
                                 reasoning_trace: List[str]) -> EmotionData:
        """이해관계자를 기반으로 타자 감정 추론"""
        
        # 이해관계자 유형별 감정 패턴
        stakeholder_emotion_patterns = {
            'vulnerable': EmotionState.FEAR,      # 취약계층
            'affected': EmotionState.SADNESS,     # 직접 영향받는 사람들
            'beneficiary': EmotionState.JOY,      # 수혜자
            'competitor': EmotionState.ANGER,     # 경쟁자
            'observer': EmotionState.NEUTRAL      # 관찰자
        }
        
        # 주요 이해관계자 식별
        primary_stakeholders = context.stakeholders[:3]  # 상위 3개만
        
        # 가장 취약하거나 영향을 많이 받는 집단의 감정을 우선
        other_emotion = EmotionState.NEUTRAL
        intensity = EmotionIntensity.MODERATE
        
        for stakeholder in primary_stakeholders:
            if 'vulnerable' in stakeholder.lower() or 'victim' in stakeholder.lower():
                other_emotion = EmotionState.FEAR
                intensity = EmotionIntensity.STRONG
                break
            elif 'affected' in stakeholder.lower() or 'impact' in stakeholder.lower():
                other_emotion = EmotionState.SADNESS
                intensity = EmotionIntensity.MODERATE
            elif 'benefit' in stakeholder.lower():
                other_emotion = EmotionState.JOY
                intensity = EmotionIntensity.MODERATE
        
        reasoning_trace.append(f"타자 감정 추론: {other_emotion.value} (주요 이해관계자: {primary_stakeholders[0] if primary_stakeholders else 'None'})")
        
        return EmotionData(
            primary_emotion=other_emotion,
            intensity=intensity,
            confidence=0.6,
            language='ko',
            processing_method='stakeholder_inference',
            dominance=0.5
        )
    
    async def _perform_ethical_reasoning(self, 
                                       integrated_emotion: EmotionData,
                                       context: CircuitDecisionContext,
                                       reasoning_trace: List[str]) -> Dict[str, float]:
        """감정을 바탕으로 한 윤리적 가치 추론"""
        
        # 벤담 계산기의 윤리적 추론 활용
        ethical_values = self.bentham_calculator._perform_ethical_reasoning(
            integrated_emotion, {'text_description': context.scenario_text}
        )
        
        # 시급성에 따른 조정
        if context.temporal_urgency > 0.8:
            ethical_values['care_harm'] += 0.1  # 긴급할 때 안전 우선
            ethical_values['authority'] += 0.05  # 신속한 결정 필요
        
        # 이해관계자 수에 따른 공정성 조정
        if context.stakeholders and len(context.stakeholders) > 5:
            ethical_values['fairness'] += 0.1  # 많은 이해관계자 시 공정성 중시
        
        # 극단값 방지
        for key in ethical_values:
            ethical_values[key] = max(0.1, min(0.9, ethical_values[key]))
        
        reasoning_trace.append(
            f"윤리적 가치 추론 완료: "
            f"돌봄={ethical_values['care_harm']:.2f}, "
            f"공정성={ethical_values['fairness']:.2f}, "
            f"충성={ethical_values['loyalty']:.2f}"
        )
        
        return ethical_values
    
    async def _calculate_ethical_bentham(self, 
                                       integrated_emotion: EmotionData,
                                       ethical_values: Dict[str, float],
                                       context: CircuitDecisionContext,
                                       reasoning_trace: List[str]) -> Any:
        """윤리적 가치를 반영한 벤담 계산"""
        
        # 과거 후회 메모리에서 학습 편향 추출
        past_regret_memory = context.past_regret_memory or {}
        
        # 벤담 계산 입력 데이터 구성
        bentham_input = {
            'input_values': {
                'intensity': 0.7,  # 기본값들
                'duration': 0.6,
                'certainty': 0.8,
                'propinquity': 0.7,
                'fecundity': 0.5,
                'purity': 0.6,
                'extent': min(1.0, len(context.stakeholders) / 10.0) if context.stakeholders else 0.5
            },
            'text_description': f"{context.scenario_text} {context.proposed_action}",
            'language': 'ko',
            'affected_count': len(context.stakeholders) if context.stakeholders else 1,
            'duration_seconds': 3600 * (1 + context.temporal_urgency),  # 시급성 반영
            'information_quality': 0.8,
            'uncertainty_level': 1.0 - integrated_emotion.confidence,
            'social_context': context.social_context or {}
        }
        
        # 윤리적 벤담 계산 실행
        bentham_result = self.bentham_calculator.calculate_with_ethical_reasoning(
            bentham_input,
            community_emotion=context.community_emotion,
            other_emotion=context.other_emotion,
            self_emotion=context.self_emotion,
            past_regret_memory=past_regret_memory
        )
        
        reasoning_trace.append(
            f"벤담 계산 완료: 최종점수={bentham_result.final_score:.3f}, "
            f"기본점수={bentham_result.base_score:.3f}, "
            f"신뢰도={bentham_result.confidence_score:.3f}"
        )
        
        return bentham_result
    
    async def _predict_regret_and_learning(self, 
                                         context: CircuitDecisionContext,
                                         bentham_result: Any,
                                         reasoning_trace: List[str]) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """후회 예측 및 학습 인사이트 추출"""
        
        # 후회 분석 입력 구성
        regret_input = {
            'id': f"decision_{int(time.time())}",
            'scenario': context.scenario_text,
            'action': context.proposed_action,
            'context': {
                'bentham_score': bentham_result.final_score,
                'confidence': bentham_result.confidence_score,
                'stakeholders': context.stakeholders or [],
                'temporal_urgency': context.temporal_urgency
            }
        }
        
        # outcome_data 구성 (벤담 결과 기반으로 단순화)
        outcome_data = {
            'utility_score': bentham_result.final_score,  # 벤담 점수를 유틸리티로 사용
            'satisfaction': bentham_result.confidence_score,  # 신뢰도를 만족도로 사용
            'success_rating': bentham_result.confidence_score,  # 벤담 신뢰도를 성공도로 사용
            'emotional_impact': 0.5,  # 기본값
            'decision_timestamp': time.time(),
            'stakeholder_count': len(context.stakeholders) if context.stakeholders else 1
        }
        
        # 후회 분석 실행 (outcome_data와 함께)
        regret_metrics = await self.regret_analyzer.analyze_regret(regret_input, outcome_data)
        
        # 예측된 후회 정보
        predicted_regret = {
            'anticipated_regret': regret_metrics.anticipated_regret,
            'regret_intensity': regret_metrics.regret_intensity,
            'regret_duration': regret_metrics.regret_duration,
            'confidence': regret_metrics.model_confidence
        }
        
        # 학습 인사이트
        learning_insights = {
            'risk_aversion_tendency': regret_metrics.anticipated_regret * 0.5,
            'decision_pattern_match': self._find_similar_decisions(context),
            'improvement_suggestions': self._generate_improvement_suggestions(regret_metrics)
        }
        
        reasoning_trace.append(
            f"후회 예측 완료: 예상후회={predicted_regret['anticipated_regret']:.3f}, "
            f"후회강도={predicted_regret['regret_intensity']:.3f}"
        )
        
        return predicted_regret, learning_insights
    
    def _integrate_final_result(self, 
                              integrated_emotion: EmotionData,
                              ethical_values: Dict[str, float],
                              bentham_result: Any,
                              predicted_regret: Dict[str, float],
                              emotion_meta: Dict,
                              reasoning_trace: List[str],
                              start_time: float) -> CircuitDecisionResult:
        """최종 결과 통합"""
        
        # 최종 윤리적 점수 계산
        final_ethical_score = bentham_result.final_score
        
        # 후회를 고려한 신뢰도 조정
        regret_adjusted_confidence = bentham_result.confidence_score * (1 - predicted_regret['anticipated_regret'] * 0.3)
        
        # 감정 충돌 유형 결정
        emotion_conflict_type = 'none'
        if len(emotion_meta['emotion_sources_used']) > 1:
            if emotion_meta['critical_loss_detected']:
                emotion_conflict_type = 'critical_loss_resolved'
            else:
                emotion_conflict_type = 'standard_integration'
        
        processing_time = time.time() - start_time
        
        reasoning_trace.append(f"최종 통합 완료: 처리시간={processing_time:.3f}초")
        
        return CircuitDecisionResult(
            final_ethical_score=final_ethical_score,
            confidence=regret_adjusted_confidence,
            integrated_emotion=integrated_emotion,
            ethical_values=ethical_values,
            bentham_result=bentham_result,
            predicted_regret=predicted_regret,
            critical_loss_detected=emotion_meta['critical_loss_detected'],
            emotion_conflict_resolved=emotion_conflict_type,
            reasoning_trace=reasoning_trace,
            processing_time=processing_time
        )
    
    def _find_similar_decisions(self, context: CircuitDecisionContext) -> float:
        """유사한 과거 결정 찾기"""
        # 실제로는 더 복잡한 유사도 계산 필요
        if context.similar_decisions_history:
            return len(context.similar_decisions_history) / 10.0
        return 0.0
    
    def _generate_improvement_suggestions(self, regret_metrics) -> List[str]:
        """개선 제안 생성"""
        suggestions = []
        
        if regret_metrics.uncertainty_estimate > 0.7:
            suggestions.append("더 많은 정보 수집이 필요합니다")
        
        if regret_metrics.anticipated_regret > 0.6:
            suggestions.append("대안 옵션을 더 탐색해보세요")
        
        if regret_metrics.model_confidence < 0.5:
            suggestions.append("전문가 의견을 구하는 것을 고려하세요")
        
        return suggestions
    
    async def _update_learning_memory(self, 
                                    context: CircuitDecisionContext,
                                    result: CircuitDecisionResult):
        """학습 메모리 업데이트"""
        
        # 의사결정 기록 저장
        decision_record = {
            'timestamp': time.time(),
            'context': context,
            'result': result,
            'ethical_score': result.final_ethical_score,
            'predicted_regret': result.predicted_regret['anticipated_regret']
        }
        
        self.decision_history.append(decision_record)
        
        # 최근 100개 결정만 유지
        if len(self.decision_history) > 100:
            self.decision_history = self.decision_history[-100:]
    
    async def _store_experience_for_future(self, 
                                         context: CircuitDecisionContext,
                                         result: CircuitDecisionResult):
        """미래 참조용 경험 데이터베이스 저장"""
        
        if not self.experience_enabled:
            return
        
        try:
            from advanced_experience_database import AdvancedExperience
            
            # 경험 데이터 구성
            experience_text = f"{context.scenario_text} {context.proposed_action}"
            
            experience = AdvancedExperience(
                text=experience_text,
                category="ethical_decision",
                metadata={
                    'ethical_score': result.final_ethical_score,
                    'regret_score': result.predicted_regret.get('anticipated_regret', 0.0),
                    'confidence': result.confidence,
                    'stakeholder_count': len(context.stakeholders) if context.stakeholders else 1,
                    'temporal_urgency': context.temporal_urgency,
                    'emotion_type': result.integrated_emotion.primary_emotion.value,
                    'processing_time': result.processing_time,
                    'reasoning_steps': len(result.reasoning_trace)
                },
                importance_score=result.final_ethical_score * result.confidence  # 윤리점수 × 신뢰도
            )
            
            # 경험 데이터베이스에 저장
            await self.experience_db.add_experience(experience)
            
            self.logger.debug(f"경험 저장 완료: 윤리점수={result.final_ethical_score:.3f}")
            
        except Exception as e:
            self.logger.warning(f"경험 저장 실패: {e}")
        
        # 패턴 학습 (간단한 버전)
        if result.final_ethical_score > 0.7:
            emotion_key = result.integrated_emotion.primary_emotion.value
            if emotion_key not in self.learning_memory['successful_decisions']:
                self.learning_memory['successful_decisions'][emotion_key] = []
            self.learning_memory['successful_decisions'][emotion_key].append(result.final_ethical_score)
    
    def _update_performance_metrics(self, 
                                  result: CircuitDecisionResult,
                                  emotion_meta: Dict):
        """성능 메트릭 업데이트"""
        
        self.performance_metrics['total_decisions'] += 1
        
        # 평균 처리 시간 업데이트
        total = self.performance_metrics['total_decisions']
        old_avg = self.performance_metrics['average_processing_time']
        new_avg = (old_avg * (total - 1) + result.processing_time) / total
        self.performance_metrics['average_processing_time'] = new_avg
        
        # 감정 충돌률 업데이트
        if result.emotion_conflict_resolved != 'none':
            conflict_count = self.performance_metrics['emotion_conflict_rate'] * (total - 1) + 1
            self.performance_metrics['emotion_conflict_rate'] = conflict_count / total
        
        # 치명적 손실률 업데이트
        if result.critical_loss_detected:
            loss_count = self.performance_metrics['critical_loss_rate'] * (total - 1) + 1
            self.performance_metrics['critical_loss_rate'] = loss_count / total
    
    

    def get_circuit_status(self) -> Dict[str, Any]:
        """회로 상태 정보 반환"""
        return {
            'performance_metrics': self.performance_metrics,
            'learning_memory_size': {
                'decision_history': len(self.decision_history),
                'successful_patterns': len(self.learning_memory['successful_decisions']),
                'regret_patterns': len(self.learning_memory['regret_patterns'])
            },
            'recent_decisions': len([d for d in self.decision_history if time.time() - d['timestamp'] < 3600])
        }


# 테스트 함수
async def test_emotion_ethics_regret_circuit():
    """감정-윤리-후회 회로 테스트"""
    
    circuit = EmotionEthicsRegretCircuit()
    
    # 테스트 시나리오
    test_context = CircuitDecisionContext(
        scenario_text="지역 공원을 개발해서 상업시설을 짓는 것을 고려하고 있습니다. 이는 경제적 이익을 가져다주지만 환경과 주민들의 휴식공간에 영향을 줄 수 있습니다.",
        proposed_action="공원 일부를 상업시설로 개발한다",
        stakeholders=["지역주민", "환경보호단체", "개발업체", "지방정부", "미래세대"],
        social_context={
            'impact_scope': 'community',
            'keywords': ['development', 'environment', 'economic']
        },
        temporal_urgency=0.6,
        past_regret_memory={'average_regret': 0.4}
    )
    
    print("=== 감정-윤리-후회 삼각 회로 테스트 ===")
    
    # 회로 처리 실행
    result = await circuit.process_ethical_decision(test_context)
    
    # 결과 출력
    print(f"\n📊 최종 결과:")
    print(f"- 윤리적 점수: {result.final_ethical_score:.3f}")
    print(f"- 신뢰도: {result.confidence:.3f}")
    print(f"- 처리 시간: {result.processing_time:.3f}초")
    print(f"- 치명적 손실 탐지: {'예' if result.critical_loss_detected else '아니오'}")
    
    print(f"\n🎭 통합된 감정:")
    print(f"- 주요 감정: {result.integrated_emotion.primary_emotion.value}")
    print(f"- 강도: {result.integrated_emotion.intensity.value}")
    print(f"- 신뢰도: {result.integrated_emotion.confidence:.3f}")
    
    print(f"\n⚖️ 윤리적 가치:")
    for key, value in result.ethical_values.items():
        print(f"- {key}: {value:.3f}")
    
    print(f"\n😔 예측된 후회:")
    for key, value in result.predicted_regret.items():
        print(f"- {key}: {value:.3f}")
    
    print(f"\n🔍 추론 과정:")
    for i, trace in enumerate(result.reasoning_trace, 1):
        print(f"{i}. {trace}")
    
    # 회로 상태 확인
    status = circuit.get_circuit_status()
    print(f"\n📈 회로 상태:")
    print(f"- 총 결정 수: {status['performance_metrics']['total_decisions']}")
    print(f"- 평균 처리 시간: {status['performance_metrics']['average_processing_time']:.3f}초")
    
    return result

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_emotion_ethics_regret_circuit())
