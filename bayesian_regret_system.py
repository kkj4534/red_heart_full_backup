"""
베이지안 후회 알고리즘 시스템 - 자기 성찰 및 시나리오 확장
상황-판단-행위-결과 로깅에서 오류 분석 및 경험 학습

Features:
- 베이지안 확률 오류 분석
- 예측 vs 실제 결과 비교
- 가중치 오류 진단 및 수정
- 확률별 하위 시나리오 자동 생성
- 시나리오별 3개 행위 시뮬레이션
- 인과관계 예상 및 경험 DB 업데이트
"""

import asyncio
import json
import logging
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path
from collections import defaultdict
import uuid
import copy

# 기존 시스템과 호환
from data_models import Decision, EthicalSituation, DecisionLog, HedonicValues, Experience, EmotionState
from config import SYSTEM_CONFIG
import utils

logger = logging.getLogger('RedHeart.BayesianRegret')

@dataclass
class DecisionRecord:
    """상황-판단-행위-결과 기록"""
    record_id: str
    timestamp: datetime
    
    # 1. 상황 (Situation)
    situation: EthicalSituation
    context: Dict[str, Any]
    stakeholders: List[str]
    
    # 2. 판단 (Judgment) 
    predicted_outcomes: Dict[str, float]  # 예측한 결과들
    bayesian_priors: Dict[str, float]     # 사전 확률
    confidence_level: float               # 확신도
    chosen_weights: Dict[str, float]      # 선택한 가중치들
    
    # 3. 행위 (Action)
    chosen_action: str
    alternative_actions: List[str]
    action_reasoning: str
    
    # 4. 결과 (Result) - 나중에 업데이트
    actual_outcomes: Optional[Dict[str, float]] = None
    regret_intensity: Optional[float] = None
    outcome_timestamp: Optional[datetime] = None
    
    # 분석 결과
    analysis_completed: bool = False
    error_analysis: Optional[Dict[str, Any]] = None
    learned_lessons: List[str] = field(default_factory=list)

@dataclass
class RegretAnalysis:
    """후회 분석 결과"""
    record_id: str = ""
    
    # 오류 분석
    bayesian_error: float = 0.0              # 베이지안 예측 오차
    prediction_errors: Dict[str, float] = field(default_factory=dict) # 각 결과별 예측 오차
    weight_errors: Dict[str, float] = field(default_factory=dict)     # 가중치 설정 오차
    
    # 원인 분석
    primary_error_source: str = ""          # 주요 오류 원인
    error_factors: List[str] = field(default_factory=list)           # 오류 요인들
    
    # 개선 방안
    suggested_weight_adjustments: Dict[str, float] = field(default_factory=dict)
    improved_priors: Dict[str, float] = field(default_factory=dict)
    confidence_adjustment: float = 0.0
    
    # ⭐ 새로운 인지적 후회 필드들
    cognitive_regret_factors: Dict[str, Any] = field(default_factory=dict)
    missed_causal_relationships: List[str] = field(default_factory=list)
    judgment_biases: Dict[str, float] = field(default_factory=dict)
    cognitive_regret_score: float = 0.0
    surd_llm_insights: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    confidence_level: float = 0.95

@dataclass
class CounterfactualScenario:
    """반사실적 시나리오"""
    scenario_id: str
    base_record_id: str
    
    # 변경된 조건
    modified_situation: EthicalSituation
    modified_context: Dict[str, Any]
    probability_weight: float          # 이 시나리오가 발생할 확률
    
    # 시뮬레이션된 3개 행위
    simulated_actions: List[Dict[str, Any]]  # 각각 action, predicted_result, confidence
    
    # 학습 내용
    causal_insights: List[str]
    updated_priors: Dict[str, float]

class BayesianRegretSystem:
    """베이지안 후회 알고리즘 메인 시스템"""
    
    def __init__(self, experience_db_path: str = "data/experience_db"):
        self.experience_db_path = Path(experience_db_path)
        self.experience_db_path.mkdir(exist_ok=True)
        
        # 기록 저장소
        self.decision_records: Dict[str, DecisionRecord] = {}
        self.regret_analyses: Dict[str, RegretAnalysis] = {}
        self.counterfactual_scenarios: Dict[str, List[CounterfactualScenario]] = {}
        
        # 학습된 지식
        self.learned_priors: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.weight_adjustments: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.causal_patterns: Dict[str, Union[List[str], Dict[str, Any]]] = defaultdict(list)
        # 인지적 학습을 위한 특별 카테고리 초기화
        self.causal_patterns['cognitive_learning'] = {}
        self.causal_patterns['cognitive_blindspots'] = {}  
        self.causal_patterns['missed_factors'] = []
        
        # 성능 추적
        self.prediction_accuracy_history: List[float] = []
        self.regret_reduction_history: List[float] = []
        
        # 기존 시스템 연동을 위한 설정
        self.integration_config = SYSTEM_CONFIG.get('regret_learning', {
            'min_outcome_wait_hours': 24,
            'confidence_threshold': 0.7,
            'max_scenarios_per_analysis': 5,
            'learning_rate': 0.1
        })
        
        logger.info("베이지안 후회 시스템 초기화 완료")
    
    async def record_decision(
        self,
        situation: EthicalSituation,
        context: Dict[str, Any],
        stakeholders: List[str],
        predicted_outcomes: Dict[str, float],
        bayesian_priors: Dict[str, float],
        confidence_level: float,
        chosen_weights: Dict[str, float],
        chosen_action: str,
        alternative_actions: List[str],
        action_reasoning: str
    ) -> str:
        """의사결정 기록 (1-2-3단계)"""
        
        record_id = f"decision_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        record = DecisionRecord(
            record_id=record_id,
            timestamp=datetime.now(),
            situation=situation,
            context=context,
            stakeholders=stakeholders,
            predicted_outcomes=predicted_outcomes,
            bayesian_priors=bayesian_priors,
            confidence_level=confidence_level,
            chosen_weights=chosen_weights,
            chosen_action=chosen_action,
            alternative_actions=alternative_actions,
            action_reasoning=action_reasoning
        )
        
        self.decision_records[record_id] = record
        await self._save_decision_record(record)
        
        logger.info(f"의사결정 기록됨: {record_id}")
        return record_id
    
    async def update_outcome(
        self,
        record_id: str,
        actual_outcomes: Dict[str, float],
        regret_intensity: float
    ) -> bool:
        """결과 업데이트 (4단계) 및 자동 분석 트리거"""
        
        if record_id not in self.decision_records:
            logger.error(f"기록을 찾을 수 없음: {record_id}")
            return False
        
        record = self.decision_records[record_id]
        record.actual_outcomes = actual_outcomes
        record.regret_intensity = regret_intensity
        record.outcome_timestamp = datetime.now()
        
        await self._save_decision_record(record)
        
        # 자동으로 후회 분석 시작
        await self.analyze_regret(record_id)
        
        logger.info(f"결과 업데이트 및 분석 완료: {record_id}")
        return True
    
    async def analyze_cognitive_regret_with_surd(self, record_id: str, 
                                               surd_analysis_result = None) -> RegretAnalysis:
        """SURD + LLM 해석을 활용한 인지적 후회 분석"""
        
        if record_id not in self.decision_records:
            raise ValueError(f"의사결정 기록을 찾을 수 없습니다: {record_id}")
            
        record = self.decision_records[record_id]
        start_time = time.time()
        
        try:
            # 1. 기본 후회 분석 수행
            basic_regret = await self.analyze_regret(record_id)
            
            # 2. SURD + LLM 해석을 통한 인지적 후회 분석
            cognitive_regret_factors = {}
            if surd_analysis_result and hasattr(surd_analysis_result, 'llm_interpretation'):
                cognitive_regret_factors = self._analyze_cognitive_blind_spots(
                    record, surd_analysis_result
                )
            
            # 3. 놓친 인과관계 분석
            missed_causal_relationships = self._identify_missed_causal_factors(
                record, surd_analysis_result
            )
            
            # 4. 판단 과정 편향 분석
            judgment_biases = self._analyze_judgment_biases(
                record, surd_analysis_result
            )
            
            # 5. 인지적 후회 점수 계산
            cognitive_regret_score = self._calculate_cognitive_regret_score(
                cognitive_regret_factors, missed_causal_relationships, judgment_biases
            )
            
            # 6. 통합 후회 분석 결과 생성
            enhanced_regret = RegretAnalysis(
                regret_id=f"cognitive_{record_id}_{int(time.time())}",
                record_id=record_id,
                bayesian_error=basic_regret.bayesian_error,
                prediction_errors=basic_regret.prediction_errors,
                weight_errors=basic_regret.weight_errors,
                confidence_level=basic_regret.confidence_level,
                # ⭐ 새로운 인지적 후회 필드들
                cognitive_regret_factors=cognitive_regret_factors,
                missed_causal_relationships=missed_causal_relationships,
                judgment_biases=judgment_biases,
                cognitive_regret_score=cognitive_regret_score,
                surd_llm_insights=self._extract_surd_llm_insights(surd_analysis_result),
                processing_time=time.time() - start_time
            )
            
            # 7. 학습 메모리 업데이트
            await self._update_cognitive_learning_memory(enhanced_regret)
            
            logger.info(f"인지적 후회 분석 완료: {record_id}")
            return enhanced_regret
            
        except Exception as e:
            logger.error(f"인지적 후회 분석 실패: {e}")
            # fallback to basic regret analysis
            return await self.analyze_regret(record_id)

    async def analyze_regret(self, record_id: str) -> RegretAnalysis:
        """베이지안 후회 분석 수행"""
        
        record = self.decision_records[record_id]
        if not record.actual_outcomes:
            raise ValueError("실제 결과가 없어서 분석할 수 없습니다")
        
        logger.info(f"후회 분석 시작: {record_id}")
        
        # 1. 베이지안 확률 오류 분석
        bayesian_error = self._calculate_bayesian_error(record)
        
        # 2. 예측 오차 계산
        prediction_errors = self._calculate_prediction_errors(record)
        
        # 3. 가중치 오류 분석
        weight_errors = self._analyze_weight_errors(record)
        
        # 4. 주요 오류 원인 식별
        primary_error_source, error_factors = self._identify_error_source(
            bayesian_error, prediction_errors, weight_errors
        )
        
        # 5. 개선 방안 생성
        weight_adjustments = await self._generate_weight_adjustments(record, weight_errors)
        improved_priors = await self._generate_improved_priors(record, prediction_errors)
        confidence_adjustment = self._calculate_confidence_adjustment(record)
        
        # 분석 결과 저장
        analysis = RegretAnalysis(
            record_id=record_id,
            bayesian_error=bayesian_error,
            prediction_errors=prediction_errors,
            weight_errors=weight_errors,
            primary_error_source=primary_error_source,
            error_factors=error_factors,
            suggested_weight_adjustments=weight_adjustments,
            improved_priors=improved_priors,
            confidence_adjustment=confidence_adjustment
        )
        
        self.regret_analyses[record_id] = analysis
        
        # 6. 반사실적 시나리오 생성
        await self._generate_counterfactual_scenarios(record_id, analysis)
        
        # 7. 경험 데이터베이스 업데이트
        await self._update_experience_database(record_id, analysis)
        
        record.analysis_completed = True
        await self._save_decision_record(record)
        
        logger.info(f"후회 분석 완료: {record_id}")
        return analysis
    
    def _calculate_bayesian_error(self, record: DecisionRecord) -> float:
        """베이지안 확률 예측 오차 계산"""
        
        total_error = 0.0
        count = 0
        
        for outcome_key, predicted_prob in record.bayesian_priors.items():
            if outcome_key in record.actual_outcomes:
                actual_value = record.actual_outcomes[outcome_key]
                # 베이지안 오차 = |P(예측) - 실제발생확률| 
                error = abs(predicted_prob - actual_value)
                total_error += error
                count += 1
        
        return total_error / count if count > 0 else 0.0
    
    def _calculate_prediction_errors(self, record: DecisionRecord) -> Dict[str, float]:
        """각 결과별 예측 오차 계산"""
        
        errors = {}
        
        for outcome_key, predicted_value in record.predicted_outcomes.items():
            if outcome_key in record.actual_outcomes:
                actual_value = record.actual_outcomes[outcome_key]
                error = abs(predicted_value - actual_value)
                errors[outcome_key] = error
        
        return errors
    
    def _analyze_weight_errors(self, record: DecisionRecord) -> Dict[str, float]:
        """가중치 설정 오류 분석"""
        
        weight_errors = {}
        
        # 실제 결과를 보고 어떤 가중치가 잘못되었는지 분석
        for weight_key, used_weight in record.chosen_weights.items():
            # 이 가중치가 결과에 미친 실제 영향도 계산
            actual_impact = self._calculate_actual_weight_impact(record, weight_key)
            
            # 사용한 가중치 vs 실제 영향도 차이
            error = abs(used_weight - actual_impact)
            weight_errors[weight_key] = error
        
        return weight_errors
    
    def _calculate_actual_weight_impact(self, record: DecisionRecord, weight_key: str) -> float:
        """특정 가중치가 실제 결과에 미친 영향도 계산"""
        
        # 간단한 휴리스틱 - 실제로는 더 복잡한 인과 분석 필요
        if weight_key in record.actual_outcomes:
            return record.actual_outcomes[weight_key]
        
        # 가중치 타입별로 실제 영향도 추정
        if 'stakeholder' in weight_key.lower():
            stakeholder_satisfaction = record.actual_outcomes.get('stakeholder_satisfaction', 0.5)
            return stakeholder_satisfaction
        elif 'ethical' in weight_key.lower():
            ethical_score = record.actual_outcomes.get('ethical_score', 0.5)
            return ethical_score
        elif 'consequence' in weight_key.lower():
            consequence_score = record.actual_outcomes.get('consequence_score', 0.5)
            return consequence_score
        
        return 0.5  # 기본값
    
    def _identify_error_source(
        self, 
        bayesian_error: float,
        prediction_errors: Dict[str, float],
        weight_errors: Dict[str, float]
    ) -> Tuple[str, List[str]]:
        """주요 오류 원인 식별"""
        
        error_factors = []
        
        # 오류 크기별 분석
        if bayesian_error > 0.3:
            error_factors.append("베이지안 사전 확률 오류")
        
        avg_prediction_error = np.mean(list(prediction_errors.values()))
        if avg_prediction_error > 0.2:
            error_factors.append("결과 예측 오류")
        
        avg_weight_error = np.mean(list(weight_errors.values()))
        if avg_weight_error > 0.25:
            error_factors.append("가중치 설정 오류")
        
        # 주요 오류 원인 결정
        max_error = max(bayesian_error, avg_prediction_error, avg_weight_error)
        
        if max_error == bayesian_error:
            primary_source = "베이지안 확률 오류"
        elif max_error == avg_prediction_error:
            primary_source = "예측 상황 오류"
        else:
            primary_source = "가중치 설정 오류"
        
        return primary_source, error_factors
    
    async def _generate_weight_adjustments(
        self, 
        record: DecisionRecord,
        weight_errors: Dict[str, float]
    ) -> Dict[str, float]:
        """가중치 조정 제안 생성"""
        
        adjustments = {}
        learning_rate = self.integration_config['learning_rate']
        
        for weight_key, error in weight_errors.items():
            current_weight = record.chosen_weights.get(weight_key, 0.5)
            actual_impact = self._calculate_actual_weight_impact(record, weight_key)
            
            # 학습률을 적용한 점진적 조정
            adjustment = learning_rate * (actual_impact - current_weight)
            new_weight = np.clip(current_weight + adjustment, 0.0, 1.0)
            
            adjustments[weight_key] = new_weight
        
        return adjustments
    
    async def _generate_improved_priors(
        self,
        record: DecisionRecord,
        prediction_errors: Dict[str, float]
    ) -> Dict[str, float]:
        """개선된 사전 확률 생성"""
        
        improved_priors = {}
        learning_rate = self.integration_config['learning_rate']
        
        for outcome_key, error in prediction_errors.items():
            current_prior = record.bayesian_priors.get(outcome_key, 0.5)
            actual_value = record.actual_outcomes.get(outcome_key, 0.5)
            
            # 베이지안 업데이트
            # P(new) = P(old) + learning_rate * (actual - P(old))
            improved_prior = current_prior + learning_rate * (actual_value - current_prior)
            improved_prior = np.clip(improved_prior, 0.01, 0.99)
            
            improved_priors[outcome_key] = improved_prior
        
        return improved_priors
    
    def _calculate_confidence_adjustment(self, record: DecisionRecord) -> float:
        """신뢰도 조정 계산"""
        
        # 예측이 정확했다면 신뢰도 증가, 틀렸다면 감소
        total_error = 0.0
        count = 0
        
        for key, predicted in record.predicted_outcomes.items():
            if key in record.actual_outcomes:
                error = abs(predicted - record.actual_outcomes[key])
                total_error += error
                count += 1
        
        avg_error = total_error / count if count > 0 else 0.5
        
        # 오차가 클수록 신뢰도 감소
        confidence_adjustment = record.confidence_level - (avg_error * 0.5)
        return np.clip(confidence_adjustment, 0.1, 0.9)
    
    async def _generate_counterfactual_scenarios(
        self,
        record_id: str,
        analysis: RegretAnalysis
    ) -> List[CounterfactualScenario]:
        """반사실적 시나리오 생성"""
        
        record = self.decision_records[record_id]
        scenarios = []
        
        # 분석 결과를 바탕으로 가능성 분기 생성
        num_scenarios = min(
            self.integration_config['max_scenarios_per_analysis'],
            3 + len(analysis.error_factors)
        )
        
        for i in range(num_scenarios):
            scenario = await self._create_single_counterfactual_scenario(
                record, analysis, scenario_index=i
            )
            scenarios.append(scenario)
        
        self.counterfactual_scenarios[record_id] = scenarios
        
        logger.info(f"반사실적 시나리오 {len(scenarios)}개 생성: {record_id}")
        return scenarios
    
    async def _create_single_counterfactual_scenario(
        self,
        record: DecisionRecord,
        analysis: RegretAnalysis,
        scenario_index: int
    ) -> CounterfactualScenario:
        """단일 반사실적 시나리오 생성"""
        
        scenario_id = f"{record.record_id}_cf_{scenario_index}"
        
        # 상황 변형 (오류 분석을 바탕으로)
        modified_situation, modified_context = self._modify_situation_based_on_errors(
            record.situation, record.context, analysis, scenario_index
        )
        
        # 확률 가중치 계산
        probability_weight = self._calculate_scenario_probability(analysis, scenario_index)
        
        # 3개 행위 시뮬레이션
        simulated_actions = await self._simulate_three_actions(
            modified_situation, modified_context, record
        )
        
        # 인과관계 분석
        causal_insights = self._analyze_causal_relationships(
            record, modified_situation, simulated_actions
        )
        
        # 업데이트된 사전 확률
        updated_priors = self._calculate_updated_priors(
            record.bayesian_priors, analysis.improved_priors, scenario_index
        )
        
        scenario = CounterfactualScenario(
            scenario_id=scenario_id,
            base_record_id=record.record_id,
            modified_situation=modified_situation,
            modified_context=modified_context,
            probability_weight=probability_weight,
            simulated_actions=simulated_actions,
            causal_insights=causal_insights,
            updated_priors=updated_priors
        )
        
        return scenario
    
    def _modify_situation_based_on_errors(
        self,
        original_situation: EthicalSituation,
        original_context: Dict[str, Any],
        analysis: RegretAnalysis,
        scenario_index: int
    ) -> Tuple[EthicalSituation, Dict[str, Any]]:
        """오류 분석을 바탕으로 상황 변형"""
        
        modified_situation = copy.deepcopy(original_situation)
        modified_context = copy.deepcopy(original_context)
        
        # 주요 오류 원인에 따라 다른 변형 적용
        if analysis.primary_error_source == "베이지안 확률 오류":
            # 확률적 요소가 다르게 작용한 상황
            modified_context['uncertainty_level'] = modified_context.get('uncertainty_level', 0.5) + 0.2
            modified_context['information_availability'] = max(0.1, 
                modified_context.get('information_availability', 0.7) - 0.3)
        
        elif analysis.primary_error_source == "예측 상황 오류":
            # 예상과 다른 외부 요인이 개입한 상황
            modified_context['external_factors'] = modified_context.get('external_factors', [])
            modified_context['external_factors'].append(f"unexpected_factor_{scenario_index}")
            
        elif analysis.primary_error_source == "가중치 설정 오류":
            # 이해관계자들의 영향력이 다른 상황
            modified_context['stakeholder_influence'] = {
                stakeholder: np.random.uniform(0.1, 0.9) 
                for stakeholder in modified_context.get('stakeholders', [])
            }
        
        return modified_situation, modified_context
    
    def _calculate_scenario_probability(
        self,
        analysis: RegretAnalysis,
        scenario_index: int
    ) -> float:
        """시나리오 발생 확률 계산"""
        
        # 오류 크기를 바탕으로 확률 계산
        base_probability = 1.0 / (scenario_index + 2)  # 점진적 감소
        
        # 주요 오류가 클수록 해당 시나리오의 확률 증가
        error_magnitude = max(analysis.prediction_errors.values()) if analysis.prediction_errors else 0.1
        probability_boost = error_magnitude * 0.5
        
        final_probability = min(base_probability + probability_boost, 0.8)
        return final_probability
    
    async def _simulate_three_actions(
        self,
        situation: EthicalSituation,
        context: Dict[str, Any],
        original_record: DecisionRecord
    ) -> List[Dict[str, Any]]:
        """3개 행위 시뮬레이션"""
        
        actions = []
        
        # 1. 원래 선택한 행위 (개선된 버전)
        improved_original = await self._simulate_improved_action(
            original_record.chosen_action, situation, context, original_record
        )
        actions.append(improved_original)
        
        # 2. 첫 번째 대안 행위
        if original_record.alternative_actions:
            alternative_1 = await self._simulate_alternative_action(
                original_record.alternative_actions[0], situation, context, original_record
            )
            actions.append(alternative_1)
        
        # 3. 완전히 새로운 행위 (AI 생성)
        novel_action = await self._generate_novel_action(situation, context, original_record)
        actions.append(novel_action)
        
        return actions
    
    async def _simulate_improved_action(
        self,
        original_action: str,
        situation: EthicalSituation,
        context: Dict[str, Any],
        original_record: DecisionRecord
    ) -> Dict[str, Any]:
        """개선된 원래 행위 시뮬레이션"""
        
        # 학습된 가중치를 적용한 개선된 버전
        improved_weights = {}
        for key, value in original_record.chosen_weights.items():
            if key in self.weight_adjustments:
                improved_weights[key] = self.weight_adjustments[key].get(key, value)
            else:
                improved_weights[key] = value
        
        # 개선된 결과 예측
        predicted_result = await self._predict_action_outcome(
            original_action, situation, context, improved_weights
        )
        
        return {
            'action': f"개선된 {original_action}",
            'predicted_result': predicted_result,
            'confidence': min(original_record.confidence_level + 0.1, 0.9),
            'reasoning': f"학습된 가중치를 적용하여 개선된 {original_action}"
        }
    
    async def _simulate_alternative_action(
        self,
        alternative_action: str,
        situation: EthicalSituation,
        context: Dict[str, Any],
        original_record: DecisionRecord
    ) -> Dict[str, Any]:
        """대안 행위 시뮬레이션"""
        
        # 대안 행위에 대한 결과 예측
        predicted_result = await self._predict_action_outcome(
            alternative_action, situation, context, original_record.chosen_weights
        )
        
        return {
            'action': alternative_action,
            'predicted_result': predicted_result,
            'confidence': original_record.confidence_level * 0.8,  # 약간 낮은 신뢰도
            'reasoning': f"원래 고려했던 대안: {alternative_action}"
        }
    
    async def _generate_novel_action(
        self,
        situation: EthicalSituation,
        context: Dict[str, Any],
        original_record: DecisionRecord
    ) -> Dict[str, Any]:
        """완전히 새로운 행위 생성"""
        
        # 간단한 휴리스틱으로 새로운 행위 생성
        # 실제로는 더 정교한 AI 생성 로직 필요
        
        novel_actions = [
            "모든 이해관계자와 추가 논의 후 결정",
            "전문가 자문을 구한 후 단계적 접근",
            "임시방편적 해결 후 장기적 계획 수립",
            "문제를 공개하고 집단 지성 활용",
            "결정을 보류하고 추가 정보 수집"
        ]
        
        # 상황에 맞는 행위 선택 (단순화)
        selected_action = np.random.choice(novel_actions)
        
        predicted_result = await self._predict_action_outcome(
            selected_action, situation, context, original_record.chosen_weights
        )
        
        return {
            'action': selected_action,
            'predicted_result': predicted_result,
            'confidence': 0.6,  # 새로운 행위라서 낮은 신뢰도
            'reasoning': f"새로운 접근법: {selected_action}"
        }
    
    async def _predict_action_outcome(
        self,
        action: str,
        situation: EthicalSituation,
        context: Dict[str, Any],
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """행위 결과 예측"""
        
        # 간단한 휴리스틱 기반 예측
        # 실제로는 더 정교한 예측 모델 필요
        
        base_outcomes = {
            'stakeholder_satisfaction': 0.5,
            'ethical_score': 0.5,
            'consequence_score': 0.5,
            'long_term_impact': 0.5
        }
        
        # 행위 특성에 따른 조정
        if '논의' in action or '자문' in action:
            base_outcomes['stakeholder_satisfaction'] += 0.2
            base_outcomes['ethical_score'] += 0.1
        
        if '단계적' in action or '장기적' in action:
            base_outcomes['long_term_impact'] += 0.3
            base_outcomes['consequence_score'] += 0.2
        
        if '공개' in action or '투명' in action:
            base_outcomes['ethical_score'] += 0.3
            base_outcomes['stakeholder_satisfaction'] -= 0.1
        
        # 가중치 적용
        for outcome_key in base_outcomes:
            if outcome_key in weights:
                base_outcomes[outcome_key] *= (1 + weights[outcome_key] - 0.5)
        
        # 값 범위 제한
        for key in base_outcomes:
            base_outcomes[key] = np.clip(base_outcomes[key], 0.0, 1.0)
        
        return base_outcomes
    
    def _analyze_causal_relationships(
        self,
        original_record: DecisionRecord,
        modified_situation: EthicalSituation,
        simulated_actions: List[Dict[str, Any]]
    ) -> List[str]:
        """인과관계 분석"""
        
        insights = []
        
        # 상황 변화가 결과에 미치는 영향 분석
        if hasattr(modified_situation, 'urgency') and hasattr(original_record.situation, 'urgency'):
            if modified_situation.urgency != original_record.situation.urgency:
                insights.append(f"긴급도 변화가 행위 선택에 중요한 영향을 미침")
        
        # 행위별 결과 차이 분석
        result_variances = []
        for action_data in simulated_actions:
            result = action_data['predicted_result']
            variance = np.var(list(result.values()))
            result_variances.append(variance)
        
        if max(result_variances) - min(result_variances) > 0.1:
            insights.append("행위 선택이 결과에 매우 큰 영향을 미침")
        
        # 이해관계자 영향 분석
        stakeholder_impacts = []
        for action_data in simulated_actions:
            stakeholder_satisfaction = action_data['predicted_result'].get('stakeholder_satisfaction', 0.5)
            stakeholder_impacts.append(stakeholder_satisfaction)
        
        if max(stakeholder_impacts) - min(stakeholder_impacts) > 0.3:
            insights.append("이해관계자 만족도가 행위에 따라 크게 달라짐")
        
        return insights
    
    def _calculate_updated_priors(
        self,
        original_priors: Dict[str, float],
        improved_priors: Dict[str, float],
        scenario_index: int
    ) -> Dict[str, float]:
        """업데이트된 사전 확률 계산"""
        
        updated = {}
        
        for key, original_value in original_priors.items():
            if key in improved_priors:
                # 시나리오 인덱스에 따라 점진적 적용
                weight = 1.0 / (scenario_index + 1)
                updated_value = original_value + weight * (improved_priors[key] - original_value)
                updated[key] = np.clip(updated_value, 0.01, 0.99)
            else:
                updated[key] = original_value
        
        return updated
    
    async def _update_experience_database(
        self,
        record_id: str,
        analysis: RegretAnalysis
    ) -> bool:
        """경험 데이터베이스 업데이트"""
        
        try:
            record = self.decision_records[record_id]
            
            # 학습된 가중치 저장
            situation_type = getattr(record.situation, 'situation_type', 'general')
            if situation_type not in self.weight_adjustments:
                self.weight_adjustments[situation_type] = {}
            
            for weight_key, new_weight in analysis.suggested_weight_adjustments.items():
                self.weight_adjustments[situation_type][weight_key] = new_weight
            
            # 개선된 사전 확률 저장
            if situation_type not in self.learned_priors:
                self.learned_priors[situation_type] = {}
            
            for prior_key, new_prior in analysis.improved_priors.items():
                self.learned_priors[situation_type][prior_key] = new_prior
            
            # 인과관계 패턴 저장
            if record_id in self.counterfactual_scenarios:
                for scenario in self.counterfactual_scenarios[record_id]:
                    for insight in scenario.causal_insights:
                        if insight not in self.causal_patterns[situation_type]:
                            self.causal_patterns[situation_type].append(insight)
            
            # 성능 지표 업데이트
            prediction_accuracy = 1.0 - analysis.bayesian_error
            self.prediction_accuracy_history.append(prediction_accuracy)
            
            regret_reduction = max(0.0, 1.0 - record.regret_intensity)
            self.regret_reduction_history.append(regret_reduction)
            
            # 파일로 저장
            await self._save_learned_knowledge()
            
            logger.info(f"경험 데이터베이스 업데이트 완료: {record_id}")
            return True
            
        except Exception as e:
            logger.error(f"경험 데이터베이스 업데이트 실패: {str(e)}")
            return False
    
    async def get_learned_weights(self, situation_type: str) -> Dict[str, float]:
        """학습된 가중치 조회"""
        return self.weight_adjustments.get(situation_type, {})
    
    async def get_learned_priors(self, situation_type: str) -> Dict[str, float]:
        """학습된 사전 확률 조회"""
        return self.learned_priors.get(situation_type, {})
    
    async def get_causal_patterns(self, situation_type: str) -> List[str]:
        """학습된 인과관계 패턴 조회"""
        return self.causal_patterns.get(situation_type, [])
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 조회"""
        
        recent_accuracy = np.mean(self.prediction_accuracy_history[-10:]) if self.prediction_accuracy_history else 0.0
        recent_regret_reduction = np.mean(self.regret_reduction_history[-10:]) if self.regret_reduction_history else 0.0
        
        return {
            'total_decisions': len(self.decision_records),
            'analyzed_decisions': len(self.regret_analyses),
            'recent_prediction_accuracy': recent_accuracy,
            'recent_regret_reduction': recent_regret_reduction,
            'learned_situation_types': list(self.learned_priors.keys()),
            'total_causal_insights': sum(len(patterns) for patterns in self.causal_patterns.values())
        }
    
    async def _save_decision_record(self, record: DecisionRecord):
        """의사결정 기록 저장"""
        file_path = self.experience_db_path / f"{record.record_id}.json"
        
        # datetime 객체를 문자열로 변환
        record_dict = asdict(record)
        record_dict['timestamp'] = record.timestamp.isoformat()
        if record.outcome_timestamp:
            record_dict['outcome_timestamp'] = record.outcome_timestamp.isoformat()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(record_dict, f, ensure_ascii=False, indent=2)
    
    async def _save_learned_knowledge(self):
        """학습된 지식 저장"""
        knowledge_file = self.experience_db_path / "learned_knowledge.json"
        
        knowledge = {
            'learned_priors': dict(self.learned_priors),
            'weight_adjustments': dict(self.weight_adjustments),
            'causal_patterns': dict(self.causal_patterns),
            'prediction_accuracy_history': self.prediction_accuracy_history,
            'regret_reduction_history': self.regret_reduction_history,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(knowledge_file, 'w', encoding='utf-8') as f:
            json.dump(knowledge, f, ensure_ascii=False, indent=2)
    
    async def load_learned_knowledge(self):
        """저장된 학습 지식 로드"""
        knowledge_file = self.experience_db_path / "learned_knowledge.json"
        
        if knowledge_file.exists():
            try:
                with open(knowledge_file, 'r', encoding='utf-8') as f:
                    knowledge = json.load(f)
                
                self.learned_priors = defaultdict(dict, knowledge.get('learned_priors', {}))
                self.weight_adjustments = defaultdict(dict, knowledge.get('weight_adjustments', {}))
                self.causal_patterns = defaultdict(list, knowledge.get('causal_patterns', {}))
                self.prediction_accuracy_history = knowledge.get('prediction_accuracy_history', [])
                self.regret_reduction_history = knowledge.get('regret_reduction_history', [])
                
                logger.info("학습된 지식 로드 완료")
                
            except Exception as e:
                logger.error(f"학습된 지식 로드 실패: {str(e)}")

    # ===== SURD + LLM 기반 인지적 후회 분석 헬퍼 메소드들 =====
    
    def _analyze_cognitive_blind_spots(self, record: DecisionRecord, 
                                     surd_analysis_result) -> Dict[str, Any]:
        """SURD + LLM 결과를 바탕으로 인지적 맹점 분석"""
        
        cognitive_factors = {}
        
        if not surd_analysis_result or not hasattr(surd_analysis_result, 'llm_interpretation'):
            return cognitive_factors
            
        llm_interpretation = surd_analysis_result.llm_interpretation
        
        # 1. 감정적 맹점 분석
        sentiment_data = getattr(surd_analysis_result, 'sentiment', {})
        if sentiment_data and isinstance(sentiment_data, dict):
            emotion_bias = sentiment_data.get('confidence', 0.0) - 0.5
            if abs(emotion_bias) > 0.3:  # 강한 감정적 편향
                cognitive_factors['emotional_bias'] = {
                    'intensity': abs(emotion_bias),
                    'direction': 'positive' if emotion_bias > 0 else 'negative',
                    'impact': '의사결정 과정에서 감정적 편향이 객관적 판단을 방해'
                }
        
        # 2. 유틸리티 계산 맹점 분석
        utility_data = getattr(surd_analysis_result, 'utility', {})
        if utility_data and isinstance(utility_data, dict):
            utility_confidence = utility_data.get('confidence', 0.5)
            if utility_confidence < 0.4:  # 낮은 유틸리티 확신도
                cognitive_factors['utility_uncertainty'] = {
                    'confidence_level': utility_confidence,
                    'impact': '유틸리티 계산 과정에서 불확실성으로 인한 오판단'
                }
        
        # 3. 합리성 추론 맹점 분석  
        rationality_data = getattr(surd_analysis_result, 'rationality', {})
        if rationality_data and isinstance(rationality_data, dict):
            rationality_score = rationality_data.get('score', 0.5)
            if rationality_score < 0.6:  # 낮은 합리성 점수
                cognitive_factors['rationality_gaps'] = {
                    'score': rationality_score,
                    'impact': '논리적 추론 과정에서 합리성 부족으로 인한 판단 오류'
                }
        
        # 4. LLM 해석 기반 추가 맹점
        if isinstance(llm_interpretation, dict):
            causal_relationships = llm_interpretation.get('causal_relationships', [])
            if causal_relationships:
                missed_factors = []
                for relationship in causal_relationships:
                    if isinstance(relationship, dict):
                        confidence = relationship.get('confidence', 0.0)
                        if confidence > 0.7:  # 높은 확신도 인과관계가 놓쳤을 가능성
                            missed_factors.append(relationship.get('factor', 'unknown'))
                
                if missed_factors:
                    cognitive_factors['missed_high_confidence_factors'] = {
                        'factors': missed_factors,
                        'impact': '의사결정 시 고려하지 못한 중요 인과요인들'
                    }
        
        return cognitive_factors
    
    def _identify_missed_causal_factors(self, record: DecisionRecord, 
                                      surd_analysis_result) -> List[str]:
        """놓친 인과관계 요인들 식별"""
        
        missed_factors = []
        
        if not surd_analysis_result or not hasattr(surd_analysis_result, 'llm_interpretation'):
            return missed_factors
            
        llm_interpretation = surd_analysis_result.llm_interpretation
        
        # 1. LLM이 식별한 인과관계에서 놓친 요인들
        if isinstance(llm_interpretation, dict):
            causal_relationships = llm_interpretation.get('causal_relationships', [])
            for relationship in causal_relationships:
                if isinstance(relationship, dict):
                    factor = relationship.get('factor', '')
                    confidence = relationship.get('confidence', 0.0)
                    
                    # 높은 확신도이지만 원래 의사결정에서 고려되지 않은 요인
                    if confidence > 0.6 and factor:
                        # 원래 예측 결과나 가중치에 포함되지 않았는지 확인
                        original_factors = set()
                        original_factors.update(record.predicted_outcomes.keys())
                        original_factors.update(record.chosen_weights.keys())
                        
                        if factor not in original_factors:
                            missed_factors.append(f"LLM 고신뢰도 요인: {factor} (신뢰도: {confidence:.2f})")
        
        # 2. SURD 분석에서 드러난 놓친 윤리적 차원
        deontology_data = getattr(surd_analysis_result, 'deontology', {})
        if deontology_data and isinstance(deontology_data, dict):
            deontology_score = deontology_data.get('score', 0.5)
            if deontology_score > 0.7:  # 높은 의무론적 중요성
                # 원래 의사결정에서 의무론적 고려가 부족했다면
                ethical_weight = record.chosen_weights.get('ethical_duty', 0.0)
                if ethical_weight < 0.5:
                    missed_factors.append(f"의무론적 윤리 고려 부족 (SURD 점수: {deontology_score:.2f})")
        
        # 3. 상황별 맥락에서 놓친 이해관계자 고려
        stakeholder_analysis = llm_interpretation.get('stakeholder_impact', {}) if isinstance(llm_interpretation, dict) else {}
        if stakeholder_analysis:
            for stakeholder, impact in stakeholder_analysis.items():
                if isinstance(impact, dict) and impact.get('severity', 0.0) > 0.6:
                    if stakeholder not in record.stakeholders:
                        missed_factors.append(f"미고려 이해관계자: {stakeholder} (영향도: {impact.get('severity', 0.0):.2f})")
        
        return missed_factors
    
    def _analyze_judgment_biases(self, record: DecisionRecord, 
                               surd_analysis_result) -> Dict[str, float]:
        """의사결정 과정의 인지 편향 분석"""
        
        biases = {}
        
        if not surd_analysis_result:
            return biases
            
        # 1. 확증 편향 (Confirmation Bias) 분석
        confidence_level = record.confidence_level
        actual_accuracy = 1.0 - self.regret_analyses.get(record.record_id, 
                                                         RegretAnalysis()).bayesian_error
        confidence_accuracy_gap = abs(confidence_level - actual_accuracy)
        if confidence_accuracy_gap > 0.3:
            biases['confirmation_bias'] = confidence_accuracy_gap
        
        # 2. 앵커링 편향 (Anchoring Bias) 분석
        # 첫 번째 예측값과 나머지 예측값들의 가중 평균 차이로 추정
        if len(record.predicted_outcomes) > 1:
            predicted_values = list(record.predicted_outcomes.values())
            first_prediction = predicted_values[0]
            avg_other_predictions = np.mean(predicted_values[1:])
            anchoring_bias = abs(first_prediction - avg_other_predictions) / max(first_prediction, avg_other_predictions, 0.1)
            biases['anchoring_bias'] = min(anchoring_bias, 1.0)
        
        # 3. 감정적 편향 (SURD 감정 분석 기반)
        sentiment_data = getattr(surd_analysis_result, 'sentiment', {})
        if sentiment_data and isinstance(sentiment_data, dict):
            sentiment_confidence = sentiment_data.get('confidence', 0.5)
            sentiment_polarity = sentiment_data.get('label', '')
            
            # 강한 감정적 반응이 있는 경우 편향 가능성 높음
            if sentiment_confidence > 0.7 and sentiment_polarity in ['POSITIVE', 'NEGATIVE']:
                emotional_bias_score = (sentiment_confidence - 0.5) * 2  # 0.5~1.0 범위를 0~1.0으로 변환
                biases['emotional_bias'] = emotional_bias_score
        
        # 4. 과신 편향 (Overconfidence Bias)
        if hasattr(surd_analysis_result, 'llm_interpretation') and surd_analysis_result.llm_interpretation:
            llm_data = surd_analysis_result.llm_interpretation
            if isinstance(llm_data, dict):
                uncertainty_indicators = llm_data.get('uncertainty_indicators', [])
                if uncertainty_indicators and len(uncertainty_indicators) > 2:
                    # LLM이 많은 불확실성을 지적했는데 높은 확신도를 보인 경우
                    if confidence_level > 0.8:
                        overconfidence_score = (confidence_level - 0.5) * len(uncertainty_indicators) * 0.1
                        biases['overconfidence_bias'] = min(overconfidence_score, 1.0)
        
        # 5. 손실 회피 편향 (Loss Aversion)
        # 가중치에서 리스크 관련 요소가 과도하게 높은 경우
        risk_weights = {k: v for k, v in record.chosen_weights.items() 
                       if 'risk' in k.lower() or 'loss' in k.lower() or 'negative' in k.lower()}
        if risk_weights:
            avg_risk_weight = np.mean(list(risk_weights.values()))
            if avg_risk_weight > 0.7:  # 과도한 리스크 회피
                biases['loss_aversion'] = (avg_risk_weight - 0.5) * 2
        
        return biases
    
    def _calculate_cognitive_regret_score(self, cognitive_factors: Dict[str, Any],
                                        missed_factors: List[str],
                                        judgment_biases: Dict[str, float]) -> float:
        """인지적 후회 종합 점수 계산"""
        
        score_components = []
        
        # 1. 인지적 맹점 점수 (0~1)
        if cognitive_factors:
            blindspot_scores = []
            for factor_type, factor_data in cognitive_factors.items():
                if isinstance(factor_data, dict):
                    intensity = factor_data.get('intensity', 0.0)
                    if isinstance(intensity, (int, float)):
                        blindspot_scores.append(min(intensity, 1.0))
                    else:
                        blindspot_scores.append(0.5)  # 기본값
            
            if blindspot_scores:
                avg_blindspot_score = np.mean(blindspot_scores)
                score_components.append(avg_blindspot_score * 0.4)  # 40% 가중치
        
        # 2. 놓친 인과요인 점수 (0~1)
        if missed_factors:
            # 놓친 요인의 수를 정규화 (최대 5개로 가정)
            missed_factor_score = min(len(missed_factors) / 5.0, 1.0)
            score_components.append(missed_factor_score * 0.3)  # 30% 가중치
        
        # 3. 판단 편향 점수 (0~1)
        if judgment_biases:
            bias_scores = list(judgment_biases.values())
            if bias_scores:
                avg_bias_score = np.mean(bias_scores)
                score_components.append(avg_bias_score * 0.3)  # 30% 가중치
        
        # 종합 점수 계산
        if score_components:
            total_score = sum(score_components)
            return min(total_score, 1.0)
        else:
            return 0.0
    
    def _extract_surd_llm_insights(self, surd_analysis_result) -> Dict[str, Any]:
        """SURD + LLM 분석 결과에서 핵심 인사이트 추출"""
        
        insights = {}
        
        if not surd_analysis_result:
            return insights
            
        # 1. SURD 각 차원별 핵심 점수
        insights['surd_scores'] = {
            'sentiment': getattr(surd_analysis_result, 'sentiment', {}).get('confidence', 0.0) if hasattr(surd_analysis_result, 'sentiment') else 0.0,
            'utility': getattr(surd_analysis_result, 'utility', {}).get('confidence', 0.0) if hasattr(surd_analysis_result, 'utility') else 0.0,
            'rationality': getattr(surd_analysis_result, 'rationality', {}).get('score', 0.0) if hasattr(surd_analysis_result, 'rationality') else 0.0,
            'deontology': getattr(surd_analysis_result, 'deontology', {}).get('score', 0.0) if hasattr(surd_analysis_result, 'deontology') else 0.0
        }
        
        # 2. LLM 해석 핵심 내용
        if hasattr(surd_analysis_result, 'llm_interpretation') and surd_analysis_result.llm_interpretation:
            llm_data = surd_analysis_result.llm_interpretation
            if isinstance(llm_data, dict):
                insights['llm_interpretation'] = {
                    'causal_relationships_count': len(llm_data.get('causal_relationships', [])),
                    'uncertainty_indicators_count': len(llm_data.get('uncertainty_indicators', [])),
                    'stakeholder_impact_count': len(llm_data.get('stakeholder_impact', {})),
                    'key_insights': llm_data.get('key_insights', [])[:3]  # 상위 3개만
                }
        
        # 3. 종합 신뢰도 점수
        surd_values = list(insights['surd_scores'].values())
        if surd_values:
            insights['overall_confidence'] = np.mean(surd_values)
        else:
            insights['overall_confidence'] = 0.0
            
        # 4. 분석 완성도
        completed_dimensions = sum(1 for score in surd_values if score > 0.1)
        insights['analysis_completeness'] = completed_dimensions / 4.0  # 4개 차원 기준
        
        return insights
    
    async def _update_cognitive_learning_memory(self, enhanced_regret: RegretAnalysis):
        """인지적 학습 메모리 업데이트"""
        
        try:
            record_id = enhanced_regret.record_id
            
            # 1. 인지적 편향 패턴 학습
            if enhanced_regret.judgment_biases:
                cognitive_bias_key = f"cognitive_biases_{record_id[:8]}"  # 앞 8자리로 키 생성
                self.causal_patterns['cognitive_learning'][cognitive_bias_key] = {
                    'biases': enhanced_regret.judgment_biases,
                    'cognitive_score': enhanced_regret.cognitive_regret_score,
                    'timestamp': datetime.now().isoformat()
                }
            
            # 2. 놓친 인과요인 패턴 학습  
            if enhanced_regret.missed_causal_relationships:
                for missed_factor in enhanced_regret.missed_causal_relationships:
                    if missed_factor not in self.causal_patterns['missed_factors']:
                        self.causal_patterns['missed_factors'].append(missed_factor)
            
            # 3. 인지적 맹점 패턴 학습
            if enhanced_regret.cognitive_regret_factors:
                blindspot_key = f"blindspots_{record_id[:8]}"
                self.causal_patterns['cognitive_blindspots'][blindspot_key] = {
                    'factors': enhanced_regret.cognitive_regret_factors,
                    'surd_insights': enhanced_regret.surd_llm_insights,
                    'timestamp': datetime.now().isoformat()
                }
            
            # 4. 학습 메모리 크기 제한 (최근 100개만 유지)
            for pattern_type in ['cognitive_learning', 'cognitive_blindspots']:
                if pattern_type in self.causal_patterns:
                    pattern_dict = self.causal_patterns[pattern_type]
                    if isinstance(pattern_dict, dict) and len(pattern_dict) > 100:
                        # 가장 오래된 50개 제거
                        sorted_items = sorted(pattern_dict.items(), 
                                            key=lambda x: x[1].get('timestamp', '') if isinstance(x[1], dict) else '')
                        items_to_keep = dict(sorted_items[-50:])
                        self.causal_patterns[pattern_type] = items_to_keep
            
            # 5. 지식 저장
            await self._save_learned_knowledge()
            
            logger.info(f"인지적 학습 메모리 업데이트 완료: {record_id}")
            
        except Exception as e:
            logger.error(f"인지적 학습 메모리 업데이트 실패: {str(e)}")

    # ===== 하향 반사실적 조건 (Downward Counterfactual) 구현 =====
    
    async def generate_downward_counterfactual_reinforcement(self, record_id: str, 
                                                           regret_analysis: RegretAnalysis) -> Dict[str, Any]:
        """하향 반사실적 조건을 통한 미약한 강화 생성
        
        최선의 결과가 보이지 않을 때, 더 나쁜 상황들을 제시하여 
        현재 선택에 대한 심리적 위안을 제공하는 시스템
        """
        
        if record_id not in self.decision_records:
            raise ValueError(f"의사결정 기록을 찾을 수 없습니다: {record_id}")
            
        record = self.decision_records[record_id]
        
        # 1. 하향 반사실적 생성 필요성 판단
        reinforcement_needed = self._assess_reinforcement_need(record, regret_analysis)
        
        if not reinforcement_needed['needed']:
            return {
                'reinforcement_generated': False,
                'reason': reinforcement_needed['reason'],
                'current_regret_level': regret_analysis.bayesian_error
            }
        
        # 2. 하향 반사실적 시나리오 생성
        downward_scenarios = await self._generate_downward_scenarios(record, regret_analysis)
        
        # 3. 각 시나리오에 대한 결과 예측
        scenario_outcomes = await self._predict_worse_outcomes(record, downward_scenarios)
        
        # 4. 심리적 위안 효과 계산
        comfort_metrics = self._calculate_psychological_comfort(
            record, regret_analysis, scenario_outcomes
        )
        
        # 5. 강화 메시지 생성
        reinforcement_messages = self._generate_reinforcement_messages(
            scenario_outcomes, comfort_metrics
        )
        
        # 6. 결과 구성
        reinforcement_result = {
            'reinforcement_generated': True,
            'trigger_condition': reinforcement_needed['trigger'],
            'downward_scenarios': downward_scenarios,
            'scenario_outcomes': scenario_outcomes,
            'comfort_metrics': comfort_metrics,
            'reinforcement_messages': reinforcement_messages,
            'effectiveness_score': comfort_metrics.get('overall_comfort', 0.0),
            'recommended_duration': self._calculate_reinforcement_duration(comfort_metrics)
        }
        
        # 7. 학습 메모리에 저장
        await self._store_reinforcement_pattern(record_id, reinforcement_result)
        
        logger.info(f"하향 반사실적 강화 생성 완료: {record_id}")
        return reinforcement_result
    
    def _assess_reinforcement_need(self, record: DecisionRecord, 
                                 regret_analysis: RegretAnalysis) -> Dict[str, Any]:
        """하향 반사실적 강화 필요성 평가"""
        
        assessment = {
            'needed': False,
            'reason': '',
            'trigger': '',
            'severity': 0.0
        }
        
        # 1. 높은 후회 강도 (주요 트리거)
        if regret_analysis.bayesian_error > 0.6:
            assessment.update({
                'needed': True,
                'trigger': 'high_regret_intensity',
                'reason': '높은 후회 강도로 인한 심리적 위안 필요',
                'severity': regret_analysis.bayesian_error
            })
            return assessment
        
        # 2. 인지적 후회 점수가 높은 경우
        if hasattr(regret_analysis, 'cognitive_regret_score') and regret_analysis.cognitive_regret_score > 0.5:
            assessment.update({
                'needed': True,
                'trigger': 'high_cognitive_regret',
                'reason': '인지적 후회로 인한 자기 비난 완화 필요',
                'severity': regret_analysis.cognitive_regret_score
            })
            return assessment
        
        # 3. 예측 오차가 클 때 (예상과 현실의 큰 차이)
        prediction_error_avg = np.mean(list(regret_analysis.prediction_errors.values())) if regret_analysis.prediction_errors else 0.0
        if prediction_error_avg > 0.5:
            assessment.update({
                'needed': True,
                'trigger': 'high_prediction_error',
                'reason': '예측 실패로 인한 실망감 완화 필요',
                'severity': prediction_error_avg
            })
            return assessment
        
        # 4. 실제 후회 강도가 기록되어 있고 높은 경우
        if record.regret_intensity and record.regret_intensity > 0.7:
            assessment.update({
                'needed': True,
                'trigger': 'high_actual_regret',
                'reason': '실제 경험한 높은 후회 강도',
                'severity': record.regret_intensity
            })
            return assessment
        
        # 5. 강화 불필요한 경우
        assessment['reason'] = '현재 후회 수준이 관리 가능한 범위 내'
        return assessment
    
    async def _generate_downward_scenarios(self, record: DecisionRecord, 
                                         regret_analysis: RegretAnalysis) -> List[Dict[str, Any]]:
        """더 나쁜 결과를 가져올 뻔한 시나리오들 생성"""
        
        scenarios = []
        
        # 1. 선택하지 않은 대안들이 더 나빴을 시나리오
        for alt_action in record.alternative_actions:
            worse_scenario = {
                'type': 'alternative_action_worse',
                'description': f"{alt_action}을 선택했다면",
                'action': alt_action,
                'worse_factors': self._identify_why_alternative_worse(record, alt_action, regret_analysis)
            }
            scenarios.append(worse_scenario)
        
        # 2. 외부 조건이 더 악화되었을 시나리오
        external_worse_scenario = {
            'type': 'external_conditions_worse',
            'description': '외부 조건이 더 악화되었다면',
            'worse_factors': self._generate_worse_external_conditions(record)
        }
        scenarios.append(external_worse_scenario)
        
        # 3. 이해관계자 반응이 더 부정적이었을 시나리오
        if record.stakeholders:
            stakeholder_worse_scenario = {
                'type': 'stakeholder_reaction_worse',
                'description': '이해관계자들 반응이 더 부정적이었다면',
                'worse_factors': self._generate_worse_stakeholder_reactions(record)
            }
            scenarios.append(stakeholder_worse_scenario)
        
        # 4. 정보가 더 부족했을 시나리오
        information_worse_scenario = {
            'type': 'information_shortage',
            'description': '정보가 더 부족했다면',
            'worse_factors': self._generate_information_shortage_effects(record)
        }
        scenarios.append(information_worse_scenario)
        
        # 5. 시간 압박이 더 심했을 시나리오
        time_pressure_scenario = {
            'type': 'time_pressure_worse',
            'description': '시간 압박이 더 심했다면',
            'worse_factors': self._generate_time_pressure_effects(record)
        }
        scenarios.append(time_pressure_scenario)
        
        return scenarios
    
    def _identify_why_alternative_worse(self, record: DecisionRecord, 
                                      alt_action: str, regret_analysis: RegretAnalysis) -> List[str]:
        """대안 행동이 더 나빴을 이유들 생성"""
        
        worse_factors = []
        
        # 실제 결과와 예측을 비교하여 대안의 리스크 요소 추정
        chosen_action = record.chosen_action
        
        # 1. 더 높은 리스크 요소
        worse_factors.append(f"{alt_action}은 {chosen_action}보다 높은 실패 확률을 가짐")
        
        # 2. 이해관계자 피해
        if record.stakeholders:
            affected_stakeholders = record.stakeholders[:2]  # 상위 2명만
            worse_factors.append(f"{', '.join(affected_stakeholders)}에게 더 큰 피해를 줄 가능성")
        
        # 3. 장기적 부작용
        worse_factors.append(f"{alt_action}은 예상치 못한 장기적 부작용 초래 가능성")
        
        # 4. 경제적 손실
        worse_factors.append(f"더 큰 비용이나 기회비용 발생 가능성")
        
        return worse_factors
    
    def _generate_worse_external_conditions(self, record: DecisionRecord) -> List[str]:
        """더 악화된 외부 조건들 생성"""
        return [
            "경제적 상황이 더 불안정했다면 선택지가 더 제한적이었을 것",
            "정치적/사회적 환경이 더 적대적이었다면 더 큰 저항에 직면",
            "기술적 제약이나 자원 부족이 더 심각했다면 실행 불가능",
            "경쟁자나 반대 세력의 방해가 더 심했다면 실패 확률 증가"
        ]
    
    def _generate_worse_stakeholder_reactions(self, record: DecisionRecord) -> List[str]:
        """더 부정적인 이해관계자 반응들 생성"""
        return [
            "주요 이해관계자들의 협력 거부로 계획 전체 무산 가능성",
            "언론이나 여론의 더 강한 반발로 인한 신뢰도 급락",
            "파트너나 동료들의 지지 철회로 고립 상황 초래",
            "상급자나 의사결정권자의 승인 거부로 프로젝트 중단"
        ]
    
    def _generate_information_shortage_effects(self, record: DecisionRecord) -> List[str]:
        """정보 부족으로 인한 더 나쁜 효과들 생성"""
        return [
            "핵심 정보 부족으로 완전히 잘못된 방향으로 진행",
            "숨겨진 리스크나 함정을 발견하지 못해 더 큰 실패",
            "경쟁 정보나 시장 동향 파악 실패로 기회 완전 상실",
            "법적, 규제적 변화를 놓쳐 심각한 법적 문제 야기"
        ]
    
    def _generate_time_pressure_effects(self, record: DecisionRecord) -> List[str]:
        """시간 압박으로 인한 더 나쁜 효과들 생성"""
        return [
            "급하게 결정하여 중대한 실수나 누락 발생",
            "충분한 검토 없이 진행하여 예측 불가능한 부작용",
            "이해관계자 설득이나 합의 과정 생략으로 나중에 큰 갈등",
            "품질 검증이나 테스트 단계 생략으로 치명적 결함 발생"
        ]
    
    async def _predict_worse_outcomes(self, record: DecisionRecord, 
                                    scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """각 하향 시나리오의 더 나쁜 결과들 예측"""
        
        scenario_outcomes = {}
        
        for i, scenario in enumerate(scenarios):
            scenario_id = f"downward_{i+1}"
            
            # 현재 실제 결과와 비교하여 얼마나 더 나빴을지 계산
            current_outcomes = record.actual_outcomes or {}
            
            predicted_worse_outcomes = {}
            for outcome_key, current_value in current_outcomes.items():
                if isinstance(current_value, (int, float)):
                    # 시나리오 타입에 따라 다른 악화 정도 적용
                    if scenario['type'] == 'alternative_action_worse':
                        worse_value = current_value * 0.6  # 40% 더 나쁨
                    elif scenario['type'] == 'external_conditions_worse':
                        worse_value = current_value * 0.4  # 60% 더 나쁨
                    elif scenario['type'] == 'stakeholder_reaction_worse':
                        worse_value = current_value * 0.5  # 50% 더 나쁨
                    else:
                        worse_value = current_value * 0.7  # 30% 더 나쁨
                    
                    predicted_worse_outcomes[outcome_key] = worse_value
                else:
                    predicted_worse_outcomes[outcome_key] = "훨씬 더 부정적"
            
            scenario_outcomes[scenario_id] = {
                'scenario': scenario,
                'predicted_outcomes': predicted_worse_outcomes,
                'improvement_over_worse': self._calculate_improvement_over_worse(
                    current_outcomes, predicted_worse_outcomes
                )
            }
        
        return scenario_outcomes
    
    def _calculate_improvement_over_worse(self, current: Dict[str, Any], 
                                        worse: Dict[str, Any]) -> float:
        """현재 결과가 더 나쁜 시나리오보다 얼마나 나은지 계산"""
        
        improvements = []
        
        for key in current.keys():
            if key in worse and isinstance(current[key], (int, float)) and isinstance(worse[key], (int, float)):
                if worse[key] != 0:
                    improvement = (current[key] - worse[key]) / abs(worse[key])
                    improvements.append(max(improvement, 0.0))  # 음수는 0으로
        
        return np.mean(improvements) if improvements else 0.3  # 기본 30% 개선
    
    def _calculate_psychological_comfort(self, record: DecisionRecord,
                                       regret_analysis: RegretAnalysis,
                                       scenario_outcomes: Dict[str, Any]) -> Dict[str, Any]:
        """심리적 위안 효과 계산"""
        
        comfort_metrics = {}
        
        # 1. 전체 위안 점수 계산
        improvement_scores = [
            outcome['improvement_over_worse'] 
            for outcome in scenario_outcomes.values()
        ]
        overall_comfort = np.mean(improvement_scores) if improvement_scores else 0.0
        
        # 2. 후회 감소 효과
        original_regret = regret_analysis.bayesian_error
        estimated_regret_reduction = min(overall_comfort * 0.3, original_regret * 0.5)
        
        # 3. 자기 효능감 회복
        self_efficacy_boost = overall_comfort * 0.4  # 40% 적용
        
        # 4. 감정적 안정성 향상
        emotional_stability = overall_comfort * 0.6  # 60% 적용
        
        comfort_metrics = {
            'overall_comfort': overall_comfort,
            'regret_reduction_estimate': estimated_regret_reduction,
            'self_efficacy_boost': self_efficacy_boost,
            'emotional_stability_improvement': emotional_stability,
            'confidence_restoration': (self_efficacy_boost + emotional_stability) / 2
        }
        
        return comfort_metrics
    
    def _generate_reinforcement_messages(self, scenario_outcomes: Dict[str, Any],
                                       comfort_metrics: Dict[str, Any]) -> List[str]:
        """강화 메시지 생성"""
        
        messages = []
        
        # 1. 주요 위안 메시지
        overall_comfort = comfort_metrics.get('overall_comfort', 0.0)
        if overall_comfort > 0.4:
            messages.append(f"현재 상황이 가능했던 더 나쁜 시나리오들보다 {overall_comfort:.1%} 더 나은 결과입니다.")
        
        # 2. 구체적 시나리오별 메시지
        for scenario_id, outcome in scenario_outcomes.items():
            improvement = outcome['improvement_over_worse']
            scenario_desc = outcome['scenario']['description']
            if improvement > 0.2:
                messages.append(f"{scenario_desc}, 현재보다 {improvement:.1%} 더 나쁜 상황이 될 뻔했습니다.")
        
        # 3. 자기 효능감 강화 메시지
        efficacy_boost = comfort_metrics.get('self_efficacy_boost', 0.0)
        if efficacy_boost > 0.3:
            messages.append("주어진 제약 조건에서 합리적인 판단을 내렸으며, 더 나쁜 결과를 피했습니다.")
        
        # 4. 학습 기회 강조 메시지
        messages.append("이 경험을 통해 미래 의사결정에서 더 나은 결과를 얻을 수 있는 인사이트를 확보했습니다.")
        
        return messages
    
    def _calculate_reinforcement_duration(self, comfort_metrics: Dict[str, Any]) -> int:
        """강화 효과 지속 시간 계산 (시간 단위)"""
        
        base_duration = 24  # 기본 24시간
        
        comfort_level = comfort_metrics.get('overall_comfort', 0.0)
        
        # 위안 효과에 따라 지속 시간 조정
        if comfort_level > 0.7:
            return base_duration * 3  # 72시간
        elif comfort_level > 0.5:
            return base_duration * 2  # 48시간
        elif comfort_level > 0.3:
            return int(base_duration * 1.5)  # 36시간
        else:
            return base_duration  # 24시간
    
    async def _store_reinforcement_pattern(self, record_id: str, 
                                         reinforcement_result: Dict[str, Any]):
        """강화 패턴을 학습 메모리에 저장"""
        
        try:
            # 강화 패턴 키 생성
            pattern_key = f"reinforcement_{record_id[:8]}_{int(time.time())}"
            
            # 패턴 데이터 구성
            pattern_data = {
                'trigger_condition': reinforcement_result['trigger_condition'],
                'effectiveness_score': reinforcement_result['effectiveness_score'],
                'scenarios_used': len(reinforcement_result['downward_scenarios']),
                'comfort_achieved': reinforcement_result['comfort_metrics']['overall_comfort'],
                'timestamp': datetime.now().isoformat()
            }
            
            # 강화 패턴 저장
            if 'reinforcement_patterns' not in self.causal_patterns:
                self.causal_patterns['reinforcement_patterns'] = {}
            
            self.causal_patterns['reinforcement_patterns'][pattern_key] = pattern_data
            
            # 메모리 크기 제한 (최근 50개만 유지)
            if len(self.causal_patterns['reinforcement_patterns']) > 50:
                sorted_items = sorted(
                    self.causal_patterns['reinforcement_patterns'].items(),
                    key=lambda x: x[1].get('timestamp', '')
                )
                self.causal_patterns['reinforcement_patterns'] = dict(sorted_items[-50:])
            
            # 지식 저장
            await self._save_learned_knowledge()
            
            logger.info(f"강화 패턴 저장 완료: {pattern_key}")
            
        except Exception as e:
            logger.error(f"강화 패턴 저장 실패: {str(e)}")

    # ===== 후회 기반 시스템 역전파 피드백 메커니즘 =====
    
    async def propagate_regret_feedback_to_system(self, regret_analysis: RegretAnalysis,
                                                record_id: str) -> Dict[str, Any]:
        """후회 분석 결과를 전체 시스템에 역전파하여 학습 업데이트"""
        
        if record_id not in self.decision_records:
            raise ValueError(f"의사결정 기록을 찾을 수 없습니다: {record_id}")
            
        record = self.decision_records[record_id]
        
        # 1. 피드백 분석 및 전략 수립
        feedback_analysis = self._analyze_feedback_requirements(regret_analysis, record)
        
        # 2. 벤담 계산기 가중치 조정
        bentham_updates = await self._update_bentham_calculator_weights(
            regret_analysis, record, feedback_analysis
        )
        
        # 3. SURD 분석기 패턴 조정
        surd_updates = await self._update_surd_analysis_patterns(
            regret_analysis, record, feedback_analysis
        )
        
        # 4. 감정 분석기 민감도 조정
        emotion_updates = await self._update_emotion_analyzer_sensitivity(
            regret_analysis, record, feedback_analysis
        )
        
        # 5. 의미 분석기 가중치 조정
        semantic_updates = await self._update_semantic_analyzer_weights(
            regret_analysis, record, feedback_analysis
        )
        
        # 6. 전역 시스템 파라미터 조정
        global_updates = await self._update_global_system_parameters(
            regret_analysis, record, feedback_analysis
        )
        
        # 7. 역전파 결과 통합
        propagation_result = {
            'record_id': record_id,
            'feedback_analysis': feedback_analysis,
            'bentham_updates': bentham_updates,
            'surd_updates': surd_updates,
            'emotion_updates': emotion_updates,
            'semantic_updates': semantic_updates,
            'global_updates': global_updates,
            'propagation_success': True,
            'affected_components': [],
            'learning_improvements': {}
        }
        
        # 8. 성공적으로 업데이트된 컴포넌트 추적
        if bentham_updates['updates_applied']:
            propagation_result['affected_components'].append('bentham_calculator')
        if surd_updates['updates_applied']:
            propagation_result['affected_components'].append('surd_analyzer')
        if emotion_updates['updates_applied']:
            propagation_result['affected_components'].append('emotion_analyzer')
        if semantic_updates['updates_applied']:
            propagation_result['affected_components'].append('semantic_analyzer')
        if global_updates['updates_applied']:
            propagation_result['affected_components'].append('global_parameters')
        
        # 9. 학습 개선도 계산
        propagation_result['learning_improvements'] = self._calculate_learning_improvements(
            regret_analysis, propagation_result
        )
        
        # 10. 피드백 패턴 저장
        await self._store_feedback_propagation_pattern(propagation_result)
        
        logger.info(f"후회 피드백 시스템 역전파 완료: {len(propagation_result['affected_components'])}개 컴포넌트 업데이트")
        return propagation_result
    
    def _analyze_feedback_requirements(self, regret_analysis: RegretAnalysis,
                                     record: DecisionRecord) -> Dict[str, Any]:
        """피드백 요구사항 및 전략 분석"""
        
        feedback_analysis = {
            'primary_error_type': regret_analysis.primary_error_source,
            'error_severity': regret_analysis.bayesian_error,
            'required_adjustments': {},
            'priority_components': [],
            'adjustment_magnitude': 'moderate'
        }
        
        # 1. 오류 유형별 주요 조정 대상 식별
        if regret_analysis.primary_error_source == "베이지안 확률 오류":
            feedback_analysis['priority_components'] = ['bentham_calculator', 'global_parameters']
            feedback_analysis['required_adjustments']['probability_weights'] = True
            feedback_analysis['required_adjustments']['confidence_calibration'] = True
            
        elif regret_analysis.primary_error_source == "예측 상황 오류":
            feedback_analysis['priority_components'] = ['surd_analyzer', 'semantic_analyzer']
            feedback_analysis['required_adjustments']['prediction_accuracy'] = True
            feedback_analysis['required_adjustments']['context_sensitivity'] = True
            
        elif regret_analysis.primary_error_source == "가중치 설정 오류":
            feedback_analysis['priority_components'] = ['bentham_calculator', 'emotion_analyzer']
            feedback_analysis['required_adjustments']['weight_balancing'] = True
            feedback_analysis['required_adjustments']['emotional_weighting'] = True
        
        # 2. 조정 강도 결정
        if regret_analysis.bayesian_error > 0.8:
            feedback_analysis['adjustment_magnitude'] = 'major'
        elif regret_analysis.bayesian_error > 0.6:
            feedback_analysis['adjustment_magnitude'] = 'moderate'
        else:
            feedback_analysis['adjustment_magnitude'] = 'minor'
        
        # 3. 인지적 후회 기반 추가 조정
        if hasattr(regret_analysis, 'cognitive_regret_score') and regret_analysis.cognitive_regret_score > 0.5:
            feedback_analysis['priority_components'].append('emotion_analyzer')
            feedback_analysis['required_adjustments']['cognitive_bias_correction'] = True
        
        return feedback_analysis
    
    async def _update_bentham_calculator_weights(self, regret_analysis: RegretAnalysis,
                                               record: DecisionRecord,
                                               feedback_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """벤담 계산기 가중치 업데이트"""
        
        updates = {
            'updates_applied': False,
            'weight_adjustments': {},
            'new_parameters': {},
            'improvement_estimate': 0.0
        }
        
        try:
            # 1. 가중치 설정 오류 기반 조정
            if 'weight_balancing' in feedback_analysis['required_adjustments']:
                for weight_key, error_magnitude in regret_analysis.weight_errors.items():
                    if error_magnitude > 0.3:  # 30% 이상 오차
                        adjustment_factor = min(error_magnitude * 0.5, 0.3)  # 최대 30% 조정
                        
                        # 오차 방향에 따라 가중치 조정
                        if weight_key in record.chosen_weights:
                            current_weight = record.chosen_weights[weight_key]
                            if error_magnitude > 0:  # 과소평가된 경우
                                new_weight = min(current_weight + adjustment_factor, 1.0)
                            else:  # 과대평가된 경우
                                new_weight = max(current_weight - adjustment_factor, 0.1)
                            
                            updates['weight_adjustments'][weight_key] = {
                                'previous': current_weight,
                                'adjusted': new_weight,
                                'change': new_weight - current_weight
                            }
            
            # 2. 경험 기반 가중치 학습 업데이트
            situation_type = getattr(record.situation, 'situation_type', 'general')
            if situation_type not in self.weight_adjustments:
                self.weight_adjustments[situation_type] = {}
            
            for weight_key, adjustment in updates['weight_adjustments'].items():
                self.weight_adjustments[situation_type][weight_key] = adjustment['adjusted']
            
            # 3. 확신도 조정 매개변수 업데이트
            if 'confidence_calibration' in feedback_analysis['required_adjustments']:
                confidence_error = abs(record.confidence_level - (1.0 - regret_analysis.bayesian_error))
                if confidence_error > 0.2:
                    confidence_adjustment = -confidence_error * 0.5  # 과신 억제
                    updates['new_parameters']['confidence_adjustment'] = confidence_adjustment
            
            updates['updates_applied'] = len(updates['weight_adjustments']) > 0 or len(updates['new_parameters']) > 0
            updates['improvement_estimate'] = len(updates['weight_adjustments']) * 0.1  # 가중치당 10% 개선 추정
            
        except Exception as e:
            logger.error(f"벤담 계산기 가중치 업데이트 실패: {e}")
        
        return updates
    
    async def _update_surd_analysis_patterns(self, regret_analysis: RegretAnalysis,
                                           record: DecisionRecord,
                                           feedback_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """SURD 분석 패턴 업데이트"""
        
        updates = {
            'updates_applied': False,
            'pattern_adjustments': {},
            'sensitivity_changes': {},
            'improvement_estimate': 0.0
        }
        
        try:
            # 1. 예측 정확도 개선을 위한 SURD 가중치 조정
            if 'prediction_accuracy' in feedback_analysis['required_adjustments']:
                # 인지적 후회 점수가 높으면 감정/합리성 차원 가중치 증가
                if hasattr(regret_analysis, 'cognitive_regret_score') and regret_analysis.cognitive_regret_score > 0.5:
                    updates['pattern_adjustments']['sentiment_weight'] = 1.2  # 20% 증가
                    updates['pattern_adjustments']['rationality_weight'] = 1.3  # 30% 증가
                
                # 베이지안 오류가 크면 유틸리티/의무론 차원 가중치 조정
                if regret_analysis.bayesian_error > 0.6:
                    updates['pattern_adjustments']['utility_weight'] = 1.1  # 10% 증가
                    updates['pattern_adjustments']['deontology_weight'] = 1.15  # 15% 증가
            
            # 2. 맥락 민감도 조정
            if 'context_sensitivity' in feedback_analysis['required_adjustments']:
                # 상황별 SURD 분석 민감도 학습
                situation_type = getattr(record.situation, 'situation_type', 'general')
                prediction_errors_avg = np.mean(list(regret_analysis.prediction_errors.values())) if regret_analysis.prediction_errors else 0.0
                
                if prediction_errors_avg > 0.4:
                    sensitivity_adjustment = 1 + (prediction_errors_avg * 0.5)
                    updates['sensitivity_changes'][situation_type] = sensitivity_adjustment
            
            # 3. LLM 해석 가중치 조정
            if hasattr(regret_analysis, 'surd_llm_insights') and regret_analysis.surd_llm_insights:
                llm_confidence = regret_analysis.surd_llm_insights.get('overall_confidence', 0.5)
                if llm_confidence < 0.4:  # LLM 해석 신뢰도가 낮았던 경우
                    updates['pattern_adjustments']['llm_interpretation_weight'] = 0.8  # 20% 감소
                elif llm_confidence > 0.8:  # LLM 해석이 매우 정확했던 경우
                    updates['pattern_adjustments']['llm_interpretation_weight'] = 1.2  # 20% 증가
            
            updates['updates_applied'] = len(updates['pattern_adjustments']) > 0 or len(updates['sensitivity_changes']) > 0
            updates['improvement_estimate'] = len(updates['pattern_adjustments']) * 0.08  # 패턴당 8% 개선 추정
            
        except Exception as e:
            logger.error(f"SURD 분석 패턴 업데이트 실패: {e}")
        
        return updates
    
    async def _update_emotion_analyzer_sensitivity(self, regret_analysis: RegretAnalysis,
                                                 record: DecisionRecord,
                                                 feedback_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """감정 분석기 민감도 업데이트"""
        
        updates = {
            'updates_applied': False,
            'sensitivity_adjustments': {},
            'bias_corrections': {},
            'improvement_estimate': 0.0
        }
        
        try:
            # 1. 감정적 가중치 오류 기반 민감도 조정
            if 'emotional_weighting' in feedback_analysis['required_adjustments']:
                # 감정 관련 가중치 오차 분석
                emotion_weight_errors = {k: v for k, v in regret_analysis.weight_errors.items() 
                                       if 'emotion' in k.lower() or 'feeling' in k.lower()}
                
                for weight_key, error in emotion_weight_errors.items():
                    if abs(error) > 0.25:
                        sensitivity_adjustment = 1 - (error * 0.3)  # 오차에 반비례 조정
                        updates['sensitivity_adjustments'][weight_key] = sensitivity_adjustment
            
            # 2. 인지 편향 보정
            if 'cognitive_bias_correction' in feedback_analysis['required_adjustments']:
                if hasattr(regret_analysis, 'judgment_biases') and regret_analysis.judgment_biases:
                    for bias_type, bias_strength in regret_analysis.judgment_biases.items():
                        if bias_strength > 0.4:
                            # 편향 유형별 보정 강도 설정
                            if bias_type == 'emotional_bias':
                                updates['bias_corrections']['emotion_dampening'] = 1 - bias_strength * 0.4
                            elif bias_type == 'confirmation_bias':
                                updates['bias_corrections']['confirmation_resistance'] = 1 + bias_strength * 0.3
                            elif bias_type == 'anchoring_bias':
                                updates['bias_corrections']['anchoring_resistance'] = 1 + bias_strength * 0.25
            
            # 3. 상황별 감정 가중치 학습
            situation_type = getattr(record.situation, 'situation_type', 'general')
            if regret_analysis.bayesian_error > 0.5:
                # 후회가 클 때 감정 분석의 가중치 재조정
                emotion_contribution = self._estimate_emotion_contribution_to_error(regret_analysis, record)
                if emotion_contribution > 0.3:
                    updates['sensitivity_adjustments'][f'{situation_type}_emotion_weight'] = 1 - emotion_contribution * 0.4
            
            updates['updates_applied'] = len(updates['sensitivity_adjustments']) > 0 or len(updates['bias_corrections']) > 0
            updates['improvement_estimate'] = len(updates['sensitivity_adjustments']) * 0.06  # 조정당 6% 개선 추정
            
        except Exception as e:
            logger.error(f"감정 분석기 민감도 업데이트 실패: {e}")
        
        return updates
    
    def _estimate_emotion_contribution_to_error(self, regret_analysis: RegretAnalysis,
                                              record: DecisionRecord) -> float:
        """감정이 오류에 기여한 정도 추정"""
        
        emotion_contribution = 0.0
        
        # 1. 감정 편향 점수 기반
        if hasattr(regret_analysis, 'judgment_biases') and 'emotional_bias' in regret_analysis.judgment_biases:
            emotion_contribution += regret_analysis.judgment_biases['emotional_bias'] * 0.4
        
        # 2. 확신도와 실제 정확도 차이에서 감정적 과신 추정
        confidence_accuracy_gap = abs(record.confidence_level - (1.0 - regret_analysis.bayesian_error))
        if confidence_accuracy_gap > 0.3:
            emotion_contribution += confidence_accuracy_gap * 0.3
        
        # 3. 감정 관련 가중치 오류
        emotion_weight_errors = [abs(v) for k, v in regret_analysis.weight_errors.items() 
                               if 'emotion' in k.lower() or 'feeling' in k.lower()]
        if emotion_weight_errors:
            emotion_contribution += np.mean(emotion_weight_errors) * 0.3
        
        return min(emotion_contribution, 1.0)
    
    async def _update_semantic_analyzer_weights(self, regret_analysis: RegretAnalysis,
                                              record: DecisionRecord,
                                              feedback_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """의미 분석기 가중치 업데이트"""
        
        updates = {
            'updates_applied': False,
            'semantic_weights': {},
            'context_adjustments': {},
            'improvement_estimate': 0.0
        }
        
        try:
            # 1. 맥락 민감도 조정
            if 'context_sensitivity' in feedback_analysis['required_adjustments']:
                # 예측 오차가 큰 경우 의미 분석 가중치 증가
                prediction_errors_avg = np.mean(list(regret_analysis.prediction_errors.values())) if regret_analysis.prediction_errors else 0.0
                if prediction_errors_avg > 0.4:
                    updates['semantic_weights']['context_analysis_weight'] = 1 + prediction_errors_avg * 0.5
                    updates['semantic_weights']['stakeholder_analysis_weight'] = 1 + prediction_errors_avg * 0.3
            
            # 2. 놓친 인과관계 기반 조정
            if hasattr(regret_analysis, 'missed_causal_relationships') and regret_analysis.missed_causal_relationships:
                missed_factors_count = len(regret_analysis.missed_causal_relationships)
                if missed_factors_count > 2:
                    # 많은 인과관계를 놓쳤다면 의미 분석 강화
                    causal_weight_boost = min(missed_factors_count * 0.1, 0.5)
                    updates['semantic_weights']['causal_relationship_weight'] = 1 + causal_weight_boost
            
            # 3. 상황별 의미 분석 패턴 조정
            situation_type = getattr(record.situation, 'situation_type', 'general')
            if regret_analysis.bayesian_error > 0.6:
                # 큰 오류 시 해당 상황 유형에서 의미 분석 강화
                updates['context_adjustments'][situation_type] = {
                    'semantic_depth': 1.3,  # 30% 증가
                    'relationship_analysis': 1.2  # 20% 증가
                }
            
            updates['updates_applied'] = len(updates['semantic_weights']) > 0 or len(updates['context_adjustments']) > 0
            updates['improvement_estimate'] = len(updates['semantic_weights']) * 0.07  # 가중치당 7% 개선 추정
            
        except Exception as e:
            logger.error(f"의미 분석기 가중치 업데이트 실패: {e}")
        
        return updates
    
    async def _update_global_system_parameters(self, regret_analysis: RegretAnalysis,
                                             record: DecisionRecord,
                                             feedback_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """전역 시스템 파라미터 업데이트"""
        
        updates = {
            'updates_applied': False,
            'parameter_changes': {},
            'system_adjustments': {},
            'improvement_estimate': 0.0
        }
        
        try:
            # 1. 확률 보정 파라미터 조정
            if 'probability_weights' in feedback_analysis['required_adjustments']:
                bayesian_error = regret_analysis.bayesian_error
                if bayesian_error > 0.5:
                    # 베이지안 오류가 클 때 전역 확률 보정 강화
                    updates['parameter_changes']['probability_calibration_factor'] = 1 - bayesian_error * 0.3
                    updates['parameter_changes']['uncertainty_penalty'] = 1 + bayesian_error * 0.4
            
            # 2. 의사결정 시간 제한 조정
            if hasattr(record, 'decision_time') and record.decision_time:
                decision_speed_error = self._analyze_decision_speed_impact(regret_analysis, record)
                if decision_speed_error > 0.3:
                    if regret_analysis.bayesian_error > 0.6:
                        # 빠른 결정이 오류로 이어진 경우 시간 제한 완화
                        updates['parameter_changes']['decision_time_multiplier'] = 1.5
                    else:
                        # 느린 결정도 오류가 발생한 경우 시간 제한 강화
                        updates['parameter_changes']['decision_time_multiplier'] = 0.8
            
            # 3. 컴포넌트 간 가중치 재조정
            component_effectiveness = self._analyze_component_effectiveness(regret_analysis, feedback_analysis)
            for component, effectiveness in component_effectiveness.items():
                if effectiveness < 0.6:  # 효과성이 낮은 컴포넌트
                    updates['system_adjustments'][f'{component}_weight'] = effectiveness + 0.2
                elif effectiveness > 0.9:  # 매우 효과적인 컴포넌트
                    updates['system_adjustments'][f'{component}_weight'] = min(effectiveness + 0.1, 1.0)
            
            # 4. 학습률 조정
            regret_severity = feedback_analysis['error_severity']
            if regret_severity > 0.7:
                updates['parameter_changes']['learning_rate'] = 1.5  # 큰 오류 시 학습률 증가
            elif regret_severity < 0.3:
                updates['parameter_changes']['learning_rate'] = 0.8  # 작은 오류 시 학습률 감소
            
            updates['updates_applied'] = len(updates['parameter_changes']) > 0 or len(updates['system_adjustments']) > 0
            updates['improvement_estimate'] = len(updates['parameter_changes']) * 0.05  # 파라미터당 5% 개선 추정
            
        except Exception as e:
            logger.error(f"전역 시스템 파라미터 업데이트 실패: {e}")
        
        return updates
    
    def _analyze_decision_speed_impact(self, regret_analysis: RegretAnalysis,
                                     record: DecisionRecord) -> float:
        """의사결정 속도가 오류에 미친 영향 분석"""
        
        # 간단한 휴리스틱: 높은 확신도 + 높은 오류 = 성급한 결정
        confidence_error_product = record.confidence_level * regret_analysis.bayesian_error
        return min(confidence_error_product, 1.0)
    
    def _analyze_component_effectiveness(self, regret_analysis: RegretAnalysis,
                                       feedback_analysis: Dict[str, Any]) -> Dict[str, float]:
        """각 컴포넌트의 효과성 분석"""
        
        effectiveness = {
            'bentham_calculator': 0.7,  # 기본값
            'surd_analyzer': 0.7,
            'emotion_analyzer': 0.7,
            'semantic_analyzer': 0.7
        }
        
        # 주요 오류 원인에 따라 컴포넌트 효과성 조정
        primary_error = regret_analysis.primary_error_source
        
        if primary_error == "베이지안 확률 오류":
            effectiveness['bentham_calculator'] *= 0.6  # 효과성 감소
        elif primary_error == "예측 상황 오류":
            effectiveness['surd_analyzer'] *= 0.6
            effectiveness['semantic_analyzer'] *= 0.6
        elif primary_error == "가중치 설정 오류":
            effectiveness['bentham_calculator'] *= 0.7
            effectiveness['emotion_analyzer'] *= 0.7
        
        # 인지적 후회가 높으면 감정 분석 효과성 감소
        if hasattr(regret_analysis, 'cognitive_regret_score') and regret_analysis.cognitive_regret_score > 0.6:
            effectiveness['emotion_analyzer'] *= 0.8
        
        return effectiveness
    
    def _calculate_learning_improvements(self, regret_analysis: RegretAnalysis,
                                       propagation_result: Dict[str, Any]) -> Dict[str, float]:
        """학습 개선도 계산"""
        
        improvements = {}
        
        # 각 컴포넌트별 개선 추정치 합산
        component_updates = [
            propagation_result['bentham_updates'],
            propagation_result['surd_updates'],
            propagation_result['emotion_updates'],
            propagation_result['semantic_updates'],
            propagation_result['global_updates']
        ]
        
        total_improvement = sum(update.get('improvement_estimate', 0.0) for update in component_updates)
        
        improvements = {
            'total_estimated_improvement': min(total_improvement, 0.5),  # 최대 50% 개선
            'components_updated': len(propagation_result['affected_components']),
            'error_reduction_potential': min(regret_analysis.bayesian_error * total_improvement, 0.3),
            'learning_acceleration': total_improvement * 2  # 학습 가속도
        }
        
        return improvements
    
    async def _store_feedback_propagation_pattern(self, propagation_result: Dict[str, Any]):
        """피드백 전파 패턴 저장"""
        
        try:
            pattern_key = f"feedback_{propagation_result['record_id'][:8]}_{int(time.time())}"
            
            pattern_data = {
                'affected_components': propagation_result['affected_components'],
                'total_improvement': propagation_result['learning_improvements']['total_estimated_improvement'],
                'error_reduction': propagation_result['learning_improvements']['error_reduction_potential'],
                'feedback_type': propagation_result['feedback_analysis']['primary_error_type'],
                'timestamp': datetime.now().isoformat()
            }
            
            # 피드백 패턴 저장
            if 'feedback_propagation' not in self.causal_patterns:
                self.causal_patterns['feedback_propagation'] = {}
            
            self.causal_patterns['feedback_propagation'][pattern_key] = pattern_data
            
            # 메모리 크기 제한 (최근 100개만 유지)
            if len(self.causal_patterns['feedback_propagation']) > 100:
                sorted_items = sorted(
                    self.causal_patterns['feedback_propagation'].items(),
                    key=lambda x: x[1].get('timestamp', '')
                )
                self.causal_patterns['feedback_propagation'] = dict(sorted_items[-100:])
            
            # 지식 저장
            await self._save_learned_knowledge()
            
            logger.info(f"피드백 전파 패턴 저장 완료: {pattern_key}")
            
        except Exception as e:
            logger.error(f"피드백 전파 패턴 저장 실패: {str(e)}")

# 기존 시스템과의 통합을 위한 래퍼 클래스
class RegretSystemIntegration:
    """기존 Red Heart 시스템과의 통합 클래스 - 업그레이드된 후회 시스템 통합"""
    
    def __init__(self):
        self.bayesian_system = BayesianRegretSystem()
        self.integration_active = False
        self.component_status = {
            'bayesian_regret': False,
            'cognitive_analysis': False,
            'downward_counterfactual': False,
            'feedback_propagation': False
        }
    
    async def initialize(self):
        """업그레이드된 후회 시스템 통합 초기화"""
        try:
            # 1. 베이지안 후회 시스템 초기화
            await self.bayesian_system.load_learned_knowledge()
            self.component_status['bayesian_regret'] = True
            logger.info("베이지안 후회 시스템 초기화 완료")
            
            # 2. 시스템 상태 검증
            self._verify_system_integrity()
            
            # 3. 통합 상태 활성화
            self.integration_active = True
            
            logger.info("업그레이드된 후회 시스템 통합 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"후회 시스템 통합 초기화 실패: {e}")
            return False
    
    def _verify_system_integrity(self):
        """시스템 무결성 검증"""
        
        # 필수 컴포넌트 검증
        required_attributes = [
            'decision_records', 'regret_analyses', 'causal_patterns',
            'learned_priors', 'weight_adjustments'
        ]
        
        for attr in required_attributes:
            if not hasattr(self.bayesian_system, attr):
                raise RuntimeError(f"필수 컴포넌트 누락: {attr}")
        
        # 인지적 후회 분석 기능 검증
        if hasattr(self.bayesian_system, 'analyze_cognitive_regret_with_surd'):
            self.component_status['cognitive_analysis'] = True
        
        # 하향 반사실적 기능 검증
        if hasattr(self.bayesian_system, 'generate_downward_counterfactual_reinforcement'):
            self.component_status['downward_counterfactual'] = True
        
        # 피드백 전파 기능 검증
        if hasattr(self.bayesian_system, 'propagate_regret_feedback_to_system'):
            self.component_status['feedback_propagation'] = True
    
    async def process_comprehensive_regret_analysis(self, decision_log: DecisionLog,
                                                  outcome_data: Optional[Dict[str, Any]] = None,
                                                  surd_analysis_result = None) -> Dict[str, Any]:
        """업그레이드된 후회 시스템의 종합 분석 처리"""
        
        if not self.integration_active:
            raise RuntimeError("후회 시스템 통합이 초기화되지 않았습니다")
        
        analysis_results = {
            'record_id': '',
            'basic_regret_analysis': None,
            'cognitive_regret_analysis': None,
            'downward_counterfactual': None,
            'feedback_propagation': None,
            'system_improvements': {},
            'processing_success': False
        }
        
        try:
            # 1. 기본 의사결정 기록 및 후회 분석
            record_id = await self.bayesian_system.record_decision(
                situation=decision_log.situation,
                context=decision_log.context or {},
                stakeholders=getattr(decision_log, 'stakeholders', []),
                predicted_outcomes=getattr(decision_log, 'predicted_outcomes', {}),
                bayesian_priors=getattr(decision_log, 'bayesian_priors', {}),
                confidence_level=getattr(decision_log, 'confidence_level', 0.7),
                chosen_weights=getattr(decision_log, 'chosen_weights', {}),
                chosen_action=decision_log.chosen_action,
                alternative_actions=getattr(decision_log, 'alternative_actions', []),
                action_reasoning=getattr(decision_log, 'action_reasoning', "")
            )
            analysis_results['record_id'] = record_id
            
            # 결과 업데이트 (있는 경우)
            if outcome_data:
                actual_outcomes = outcome_data.get('actual_outcomes', {})
                regret_intensity = outcome_data.get('regret_intensity', 0.0)
                await self.bayesian_system.update_outcome(record_id, actual_outcomes, regret_intensity)
            
            # 기본 후회 분석
            if self.component_status['bayesian_regret']:
                basic_regret = await self.bayesian_system.analyze_regret(record_id)
                analysis_results['basic_regret_analysis'] = basic_regret
                logger.info(f"기본 후회 분석 완료: 오류율 {basic_regret.bayesian_error:.3f}")
            
            # 2. 인지적 후회 분석 (SURD + LLM 통합)
            if self.component_status['cognitive_analysis'] and surd_analysis_result:
                cognitive_regret = await self.bayesian_system.analyze_cognitive_regret_with_surd(
                    record_id, surd_analysis_result
                )
                analysis_results['cognitive_regret_analysis'] = cognitive_regret
                logger.info(f"인지적 후회 분석 완료: 점수 {cognitive_regret.cognitive_regret_score:.3f}")
            
            # 3. 하향 반사실적 강화 생성 (필요한 경우)
            regret_for_reinforcement = analysis_results['cognitive_regret_analysis'] or analysis_results['basic_regret_analysis']
            if self.component_status['downward_counterfactual'] and regret_for_reinforcement:
                downward_reinforcement = await self.bayesian_system.generate_downward_counterfactual_reinforcement(
                    record_id, regret_for_reinforcement
                )
                analysis_results['downward_counterfactual'] = downward_reinforcement
                
                if downward_reinforcement['reinforcement_generated']:
                    logger.info(f"하향 반사실적 강화 생성: 효과 {downward_reinforcement['effectiveness_score']:.3f}")
            
            # 4. 시스템 전체 피드백 전파
            if self.component_status['feedback_propagation'] and regret_for_reinforcement:
                feedback_result = await self.bayesian_system.propagate_regret_feedback_to_system(
                    regret_for_reinforcement, record_id
                )
                analysis_results['feedback_propagation'] = feedback_result
                analysis_results['system_improvements'] = feedback_result.get('learning_improvements', {})
                
                logger.info(f"시스템 피드백 전파 완료: {len(feedback_result['affected_components'])}개 컴포넌트 업데이트")
            
            analysis_results['processing_success'] = True
            logger.info(f"종합 후회 분석 완료: {record_id}")
            
        except Exception as e:
            logger.error(f"종합 후회 분석 실패: {e}")
            analysis_results['error'] = str(e)
        
        return analysis_results
    
    async def process_decision_with_learning(
        self,
        decision_log: DecisionLog,
        outcome_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """의사결정 처리 및 학습 (기존 시스템 호환)"""
        
        # DecisionLog를 BayesianRegretSystem 형식으로 변환
        record_id = await self.bayesian_system.record_decision(
            situation=decision_log.situation,
            context=decision_log.context or {},
            stakeholders=getattr(decision_log, 'stakeholders', []),
            predicted_outcomes=getattr(decision_log, 'predicted_outcomes', {}),
            bayesian_priors=getattr(decision_log, 'bayesian_priors', {}),
            confidence_level=getattr(decision_log, 'confidence_level', 0.7),
            chosen_weights=getattr(decision_log, 'chosen_weights', {}),
            chosen_action=decision_log.chosen_action,
            alternative_actions=getattr(decision_log, 'alternative_actions', []),
            action_reasoning=getattr(decision_log, 'action_reasoning', "")
        )
        
        # 결과가 있으면 즉시 분석
        if outcome_data:
            actual_outcomes = outcome_data.get('actual_outcomes', {})
            regret_intensity = outcome_data.get('regret_intensity', 0.0)
            
            await self.bayesian_system.update_outcome(
                record_id, actual_outcomes, regret_intensity
            )
        
        return record_id
    
    async def run_system_integration_test(self) -> Dict[str, Any]:
        """통합 시스템 테스트 실행"""
        
        test_results = {
            'test_passed': False,
            'component_tests': {},
            'integration_tests': {},
            'performance_metrics': {},
            'recommendations': []
        }
        
        try:
            logger.info("후회 시스템 통합 테스트 시작")
            
            # 1. 개별 컴포넌트 테스트
            test_results['component_tests'] = await self._test_individual_components()
            
            # 2. 통합 워크플로우 테스트
            test_results['integration_tests'] = await self._test_integration_workflows()
            
            # 3. 성능 메트릭 수집
            test_results['performance_metrics'] = await self._collect_performance_metrics()
            
            # 4. 시스템 상태 종합 평가
            overall_success = self._evaluate_overall_test_results(test_results)
            test_results['test_passed'] = overall_success
            
            # 5. 개선 권장사항 생성
            test_results['recommendations'] = self._generate_improvement_recommendations(test_results)
            
            logger.info(f"후회 시스템 통합 테스트 완료: {'성공' if overall_success else '실패'}")
            
        except Exception as e:
            logger.error(f"통합 테스트 실행 실패: {e}")
            test_results['error'] = str(e)
        
        return test_results
    
    async def _test_individual_components(self) -> Dict[str, Any]:
        """개별 컴포넌트 테스트"""
        
        component_tests = {}
        
        # 베이지안 후회 시스템 테스트
        component_tests['bayesian_regret'] = await self._test_bayesian_regret_component()
        
        # 인지적 후회 분석 테스트
        if self.component_status['cognitive_analysis']:
            component_tests['cognitive_analysis'] = await self._test_cognitive_analysis_component()
        
        # 하향 반사실적 테스트
        if self.component_status['downward_counterfactual']:
            component_tests['downward_counterfactual'] = await self._test_downward_counterfactual_component()
        
        # 피드백 전파 테스트
        if self.component_status['feedback_propagation']:
            component_tests['feedback_propagation'] = await self._test_feedback_propagation_component()
        
        return component_tests
    
    async def _test_bayesian_regret_component(self) -> Dict[str, Any]:
        """베이지안 후회 시스템 컴포넌트 테스트"""
        
        test_result = {'success': False, 'details': {}}
        
        try:
            # 테스트 데이터 생성
            from data_models import EthicalSituation
            test_situation = EthicalSituation(
                situation_id="test_001",
                description="테스트 윤리적 상황",
                stakeholders=["테스트_이해관계자1", "테스트_이해관계자2"],
                situation_type="test"
            )
            
            # 의사결정 기록 테스트
            record_id = await self.bayesian_system.record_decision(
                situation=test_situation,
                context={"test": True},
                stakeholders=["테스트_이해관계자1"],
                predicted_outcomes={"결과1": 0.7, "결과2": 0.3},
                bayesian_priors={"사전확률1": 0.6},
                confidence_level=0.8,
                chosen_weights={"가중치1": 0.5, "가중치2": 0.5},
                chosen_action="테스트_행동",
                alternative_actions=["대안행동1", "대안행동2"],
                action_reasoning="테스트 추론"
            )
            test_result['details']['record_creation'] = True
            
            # 결과 업데이트 테스트
            await self.bayesian_system.update_outcome(
                record_id, 
                {"결과1": 0.5, "결과2": 0.4}, 
                0.6
            )
            test_result['details']['outcome_update'] = True
            
            # 후회 분석 테스트
            regret_analysis = await self.bayesian_system.analyze_regret(record_id)
            test_result['details']['regret_analysis'] = regret_analysis is not None
            test_result['details']['bayesian_error'] = regret_analysis.bayesian_error if regret_analysis else 0.0
            
            test_result['success'] = True
            
        except Exception as e:
            test_result['error'] = str(e)
        
        return test_result
    
    async def _test_cognitive_analysis_component(self) -> Dict[str, Any]:
        """인지적 후회 분석 컴포넌트 테스트"""
        
        test_result = {'success': False, 'details': {}}
        
        try:
            # 기존 분석 결과가 있는지 확인
            if self.bayesian_system.regret_analyses:
                record_id = list(self.bayesian_system.regret_analyses.keys())[0]
                basic_regret = self.bayesian_system.regret_analyses[record_id]
                
                # 모의 SURD 결과 생성
                mock_surd_result = type('MockSURD', (), {
                    'sentiment': {'confidence': 0.7, 'label': 'POSITIVE'},
                    'utility': {'confidence': 0.6, 'score': 0.8},
                    'rationality': {'score': 0.5},
                    'deontology': {'score': 0.7},
                    'llm_interpretation': {
                        'causal_relationships': [
                            {'factor': '테스트_요인', 'confidence': 0.8}
                        ],
                        'uncertainty_indicators': ['불확실성1', '불확실성2'],
                        'stakeholder_impact': {'이해관계자1': {'severity': 0.7}}
                    }
                })()
                
                # 인지적 후회 분석 실행
                cognitive_result = await self.bayesian_system.analyze_cognitive_regret_with_surd(
                    record_id, mock_surd_result
                )
                
                test_result['details']['cognitive_analysis_executed'] = True
                test_result['details']['cognitive_regret_score'] = cognitive_result.cognitive_regret_score
                test_result['details']['missed_factors_count'] = len(cognitive_result.missed_causal_relationships)
                test_result['success'] = True
            else:
                test_result['details']['no_existing_analysis'] = True
                test_result['success'] = False
                
        except Exception as e:
            test_result['error'] = str(e)
        
        return test_result
    
    async def _test_downward_counterfactual_component(self) -> Dict[str, Any]:
        """하향 반사실적 컴포넌트 테스트"""
        
        test_result = {'success': False, 'details': {}}
        
        try:
            if self.bayesian_system.regret_analyses:
                record_id = list(self.bayesian_system.regret_analyses.keys())[0]
                regret_analysis = self.bayesian_system.regret_analyses[record_id]
                
                # 높은 후회 시나리오 시뮬레이션
                regret_analysis.bayesian_error = 0.7  # 높은 오류율 설정
                
                downward_result = await self.bayesian_system.generate_downward_counterfactual_reinforcement(
                    record_id, regret_analysis
                )
                
                test_result['details']['reinforcement_generated'] = downward_result['reinforcement_generated']
                if downward_result['reinforcement_generated']:
                    test_result['details']['scenarios_count'] = len(downward_result['downward_scenarios'])
                    test_result['details']['effectiveness_score'] = downward_result['effectiveness_score']
                
                test_result['success'] = True
            else:
                test_result['success'] = False
                test_result['details']['no_existing_analysis'] = True
                
        except Exception as e:
            test_result['error'] = str(e)
        
        return test_result
    
    async def _test_feedback_propagation_component(self) -> Dict[str, Any]:
        """피드백 전파 컴포넌트 테스트"""
        
        test_result = {'success': False, 'details': {}}
        
        try:
            if self.bayesian_system.regret_analyses:
                record_id = list(self.bayesian_system.regret_analyses.keys())[0]
                regret_analysis = self.bayesian_system.regret_analyses[record_id]
                
                feedback_result = await self.bayesian_system.propagate_regret_feedback_to_system(
                    regret_analysis, record_id
                )
                
                test_result['details']['propagation_success'] = feedback_result['propagation_success']
                test_result['details']['affected_components'] = feedback_result['affected_components']
                test_result['details']['total_improvement'] = feedback_result['learning_improvements']['total_estimated_improvement']
                
                test_result['success'] = True
            else:
                test_result['success'] = False
                test_result['details']['no_existing_analysis'] = True
                
        except Exception as e:
            test_result['error'] = str(e)
        
        return test_result
    
    async def _test_integration_workflows(self) -> Dict[str, Any]:
        """통합 워크플로우 테스트"""
        
        integration_tests = {}
        
        try:
            # 전체 파이프라인 테스트
            from data_models import DecisionLog, EthicalSituation
            
            test_decision = DecisionLog(
                decision_id="integration_test_001",
                situation=EthicalSituation(
                    situation_id="int_test_situation",
                    description="통합 테스트 상황",
                    stakeholders=["통합테스트_이해관계자"],
                    situation_type="integration_test"
                ),
                chosen_action="통합테스트_행동",
                context={"integration_test": True}
            )
            
            # 종합 분석 실행
            comprehensive_result = await self.process_comprehensive_regret_analysis(
                test_decision,
                outcome_data={
                    'actual_outcomes': {'결과1': 0.4, '결과2': 0.6},
                    'regret_intensity': 0.5
                }
            )
            
            integration_tests['comprehensive_analysis'] = {
                'success': comprehensive_result['processing_success'],
                'components_activated': sum(1 for k, v in comprehensive_result.items() 
                                          if k.endswith('_analysis') or k.endswith('_counterfactual') or k.endswith('_propagation') 
                                          and v is not None)
            }
            
        except Exception as e:
            integration_tests['comprehensive_analysis'] = {
                'success': False,
                'error': str(e)
            }
        
        return integration_tests
    
    async def _collect_performance_metrics(self) -> Dict[str, Any]:
        """성능 메트릭 수집"""
        
        metrics = {}
        
        try:
            # 시스템 상태 메트릭
            performance_summary = await self.bayesian_system.get_performance_summary()
            metrics['system_performance'] = performance_summary
            
            # 컴포넌트 활성화 상태
            metrics['component_status'] = self.component_status.copy()
            metrics['integration_active'] = self.integration_active
            
            # 메모리 사용량 (근사치)
            metrics['memory_usage'] = {
                'decision_records': len(self.bayesian_system.decision_records),
                'regret_analyses': len(self.bayesian_system.regret_analyses),
                'causal_patterns': len(self.bayesian_system.causal_patterns)
            }
            
        except Exception as e:
            metrics['error'] = str(e)
        
        return metrics
    
    def _evaluate_overall_test_results(self, test_results: Dict[str, Any]) -> bool:
        """전체 테스트 결과 평가"""
        
        # 필수 컴포넌트 테스트 성공 여부 확인
        component_tests = test_results.get('component_tests', {})
        
        # 베이지안 후회 시스템은 필수
        if not component_tests.get('bayesian_regret', {}).get('success', False):
            return False
        
        # 활성화된 컴포넌트들의 테스트 성공 여부 확인
        for component, enabled in self.component_status.items():
            if enabled and component in component_tests:
                if not component_tests[component].get('success', False):
                    return False
        
        # 통합 테스트 성공 여부 확인
        integration_tests = test_results.get('integration_tests', {})
        if not integration_tests.get('comprehensive_analysis', {}).get('success', False):
            return False
        
        return True
    
    def _generate_improvement_recommendations(self, test_results: Dict[str, Any]) -> List[str]:
        """개선 권장사항 생성"""
        
        recommendations = []
        
        # 실패한 컴포넌트에 대한 권장사항
        component_tests = test_results.get('component_tests', {})
        for component, result in component_tests.items():
            if not result.get('success', False):
                if 'error' in result:
                    recommendations.append(f"{component} 컴포넌트 오류 수정 필요: {result['error']}")
                else:
                    recommendations.append(f"{component} 컴포넌트 기능 점검 필요")
        
        # 성능 기반 권장사항
        performance = test_results.get('performance_metrics', {})
        if 'system_performance' in performance:
            sys_perf = performance['system_performance']
            if sys_perf.get('recent_prediction_accuracy', 0) < 0.7:
                recommendations.append("예측 정확도 개선 필요 (현재 70% 미만)")
            if sys_perf.get('total_decisions', 0) < 10:
                recommendations.append("더 많은 의사결정 데이터 수집 권장")
        
        # 통합 테스트 기반 권장사항
        integration_tests = test_results.get('integration_tests', {})
        comp_analysis = integration_tests.get('comprehensive_analysis', {})
        if comp_analysis.get('components_activated', 0) < 3:
            recommendations.append("더 많은 시스템 컴포넌트 활성화 고려")
        
        if not recommendations:
            recommendations.append("시스템이 정상적으로 작동 중입니다.")
        
        return recommendations
    
    async def get_learning_insights(self, situation_type: str = "general") -> Dict[str, Any]:
        """학습 인사이트 조회 (기존 시스템 호환)"""
        
        weights = await self.bayesian_system.get_learned_weights(situation_type)
        priors = await self.bayesian_system.get_learned_priors(situation_type)
        patterns = await self.bayesian_system.get_causal_patterns(situation_type)
        performance = await self.bayesian_system.get_performance_summary()
        
        return {
            'learned_weights': weights,
            'learned_priors': priors,
            'causal_patterns': patterns,
            'performance_summary': performance
        }