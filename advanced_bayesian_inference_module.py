"""
고급 베이지안 추론 모듈 - Linux 전용
Advanced Bayesian Inference Module for Linux

독립적 베이지안 추론 시스템:
- 사전 확률 학습 및 업데이트
- 문학 데이터 기반 믿음 네트워크
- 동적 추론 및 불확실성 정량화
"""

import os
import json
import time
import asyncio
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
import uuid
from collections import defaultdict, deque
from enum import Enum
import pickle
from scipy import stats
from scipy.special import logsumexp

# 시스템 설정
from config import ADVANCED_CONFIG, CACHE_DIR, MODELS_DIR, LOGS_DIR

# 로깅 설정
logger = logging.getLogger(__name__)

class BeliefType(Enum):
    """믿음/신념 유형"""
    FACTUAL = "factual"  # 사실적 믿음
    MORAL = "moral"  # 도덕적 믿음
    EMOTIONAL = "emotional"  # 감정적 믿음
    SOCIAL = "social"  # 사회적 믿음
    PREDICTIVE = "predictive"  # 예측적 믿음

@dataclass
class BayesianNode:
    """베이지안 네트워크 노드"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    belief_type: BeliefType = BeliefType.FACTUAL
    
    # 확률 분포
    prior: Dict[str, float] = field(default_factory=dict)  # 사전 확률
    likelihood: Dict[str, Dict[str, float]] = field(default_factory=dict)  # 우도
    posterior: Dict[str, float] = field(default_factory=dict)  # 사후 확률
    
    # 부모/자식 노드
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    
    # 메타데이터
    evidence_count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    confidence: float = 0.5
    
    # 문학적 맥락
    literary_sources: List[str] = field(default_factory=list)
    narrative_contexts: Dict[str, float] = field(default_factory=dict)

@dataclass
class Evidence:
    """증거/관찰 데이터"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_id: str = ""
    value: Any = None
    strength: float = 1.0  # 증거의 강도/신뢰도
    source: str = ""  # 증거 출처
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class InferenceResult:
    """추론 결과"""
    query_node: str
    posterior_distribution: Dict[str, float]
    confidence: float
    uncertainty: float  # 엔트로피 기반
    influential_factors: List[Tuple[str, float]]  # (노드, 영향도)
    reasoning_path: List[str]
    computation_time: float

class LiteraryBeliefNetwork:
    """문학 기반 믿음 네트워크"""
    
    def __init__(self):
        self.belief_patterns = self._load_literary_belief_patterns()
        self.narrative_priors = self._initialize_narrative_priors()
        
    def _load_literary_belief_patterns(self) -> Dict[str, Dict[str, Any]]:
        """문학적 믿음 패턴 로드"""
        return {
            'tragic_fate': {
                'description': '비극적 운명에 대한 믿음',
                'prior': {'inevitable': 0.7, 'changeable': 0.3},
                'sources': ['오이디푸스', '햄릿', '맥베스'],
                'update_rate': 0.1
            },
            'redemption': {
                'description': '구원/회복 가능성에 대한 믿음',
                'prior': {'possible': 0.6, 'impossible': 0.4},
                'sources': ['죄와 벌', '레미제라블', '크리스마스 캐럴'],
                'update_rate': 0.15
            },
            'love_conquers': {
                'description': '사랑의 극복력에 대한 믿음',
                'prior': {'strong': 0.5, 'weak': 0.5},
                'sources': ['로미오와 줄리엣', '춘향전', '어린왕자'],
                'update_rate': 0.2
            },
            'karma': {
                'description': '인과응보에 대한 믿음',
                'prior': {'exists': 0.8, 'random': 0.2},
                'sources': ['홍길동전', '장화홍련전', '심청전'],
                'update_rate': 0.12
            }
        }
    
    def _initialize_narrative_priors(self) -> Dict[str, float]:
        """서사 구조별 사전 확률"""
        return {
            'hero_journey': 0.7,  # 영웅의 여정
            'tragedy': 0.6,  # 비극
            'comedy': 0.5,  # 희극
            'romance': 0.65,  # 로맨스
            'moral_tale': 0.75  # 교훈담
        }
    
    def get_literary_prior(self, belief_type: str, context: Dict[str, Any]) -> Dict[str, float]:
        """문학적 맥락 기반 사전 확률 반환"""
        if belief_type in self.belief_patterns:
            pattern = self.belief_patterns[belief_type]
            prior = pattern['prior'].copy()
            
            # 서사 구조에 따른 조정
            if 'narrative_type' in context:
                narrative = context['narrative_type']
                if narrative in self.narrative_priors:
                    adjustment = self.narrative_priors[narrative] - 0.5
                    for key in prior:
                        prior[key] = np.clip(prior[key] + adjustment * 0.2, 0.1, 0.9)
            
            return prior
        
        # 기본 균등 분포
        return {'true': 0.5, 'false': 0.5}

class AdvancedBayesianInference:
    """고급 베이지안 추론 시스템"""
    
    def __init__(self):
        """시스템 초기화"""
        self.config = ADVANCED_CONFIG.get('bayesian_inference', {
            'smoothing_factor': 0.01,  # 라플라스 스무딩
            'evidence_decay': 0.95,  # 증거 시간 감쇠
            'min_confidence': 0.1,
            'max_iterations': 100,
            'convergence_threshold': 0.001
        })
        
        # 네트워크 구조
        self.nodes: Dict[str, BayesianNode] = {}
        self.edges: List[Tuple[str, str]] = []  # (parent, child)
        
        # 증거 저장소
        self.evidence_history: List[Evidence] = []
        self.evidence_cache: Dict[str, List[Evidence]] = defaultdict(list)
        
        # 문학적 믿음 네트워크
        self.literary_network = LiteraryBeliefNetwork()
        
        # 추론 캐시
        self.inference_cache = {}
        self.cache_ttl = 300  # 5분
        
        # 학습 메트릭
        self.learning_metrics = {
            'total_inferences': 0,
            'successful_predictions': 0,
            'belief_updates': 0,
            'accuracy_history': []
        }
        
        # 기본 네트워크 구축
        self._build_default_network()
        
        logger.info("고급 베이지안 추론 시스템이 초기화되었습니다.")
    
    def _build_default_network(self):
        """기본 베이지안 네트워크 구축"""
        # 감정-행동 네트워크
        self.add_node(BayesianNode(
            name="emotional_state",
            belief_type=BeliefType.EMOTIONAL,
            prior={"positive": 0.5, "negative": 0.5}
        ))
        
        self.add_node(BayesianNode(
            name="action_tendency",
            belief_type=BeliefType.PREDICTIVE,
            prior={"approach": 0.5, "avoid": 0.5}
        ))
        
        self.add_edge("emotional_state", "action_tendency")
        
        # 도덕적 판단 네트워크
        self.add_node(BayesianNode(
            name="moral_judgment",
            belief_type=BeliefType.MORAL,
            prior={"right": 0.6, "wrong": 0.4}
        ))
        
        self.add_node(BayesianNode(
            name="social_approval",
            belief_type=BeliefType.SOCIAL,
            prior={"approve": 0.5, "disapprove": 0.5}
        ))
        
        self.add_edge("moral_judgment", "social_approval")
        
        # 예측 네트워크
        self.add_node(BayesianNode(
            name="outcome_prediction",
            belief_type=BeliefType.PREDICTIVE,
            prior={"success": 0.5, "failure": 0.5}
        ))
        
        self.add_edge("action_tendency", "outcome_prediction")
        self.add_edge("emotional_state", "outcome_prediction")
        
        # 후회 예측 네트워크
        self.add_node(BayesianNode(
            name="regret_prediction",
            belief_type=BeliefType.PREDICTIVE,
            prior={"low": 0.3, "medium": 0.4, "high": 0.3}
        ))
        
        self.add_edge("emotional_state", "regret_prediction")
        self.add_edge("action_tendency", "regret_prediction")
        self.add_edge("outcome_prediction", "regret_prediction")
    
    def add_node(self, node: BayesianNode):
        """노드 추가"""
        self.nodes[node.id] = node
        if node.name:
            # 이름으로도 접근 가능하게
            self.nodes[node.name] = node
    
    def add_edge(self, parent: str, child: str):
        """엣지(의존성) 추가"""
        parent_node = self.nodes.get(parent)
        child_node = self.nodes.get(child)
        
        if parent_node and child_node:
            self.edges.append((parent_node.id, child_node.id))
            parent_node.children.append(child_node.id)
            child_node.parents.append(parent_node.id)
            
            # 조건부 확률 테이블 초기화
            self._initialize_cpt(parent_node, child_node)
    
    def _initialize_cpt(self, parent: BayesianNode, child: BayesianNode):
        """조건부 확률 테이블 초기화"""
        # 간단한 초기화: 부모 상태에 따라 자식 상태 영향
        if not child.likelihood:
            child.likelihood = {}
        
        for parent_state in parent.prior.keys():
            if parent_state not in child.likelihood:
                child.likelihood[parent_state] = {}
            
            # 기본적으로 부모와 유사한 경향
            for child_state in child.prior.keys():
                if parent_state == child_state or \
                   (parent_state in ["positive", "right", "success"] and 
                    child_state in ["positive", "approach", "approve", "success"]):
                    child.likelihood[parent_state][child_state] = 0.7
                else:
                    child.likelihood[parent_state][child_state] = 0.3
    
    async def add_evidence(self, evidence: Evidence):
        """증거 추가"""
        try:
            # 증거 검증
            if evidence.node_id not in self.nodes:
                logger.warning(f"존재하지 않는 노드에 대한 증거: {evidence.node_id}")
                return
            
            # 증거 저장
            self.evidence_history.append(evidence)
            self.evidence_cache[evidence.node_id].append(evidence)
            
            # 믿음 업데이트
            await self._update_beliefs(evidence)
            
            # 캐시 무효화
            self.inference_cache.clear()
            
            self.learning_metrics['belief_updates'] += 1
            
            logger.debug(f"증거 추가됨: 노드={evidence.node_id}, 값={evidence.value}")
            
        except Exception as e:
            logger.error(f"증거 추가 실패: {e}")
    
    async def _update_beliefs(self, evidence: Evidence):
        """베이지안 업데이트"""
        node = self.nodes[evidence.node_id]
        
        # 시간 가중치 계산 (최신 증거일수록 높은 가중치)
        time_weight = self._calculate_time_weight(evidence.timestamp)
        weighted_strength = evidence.strength * time_weight
        
        # 베이지안 업데이트
        if not node.posterior:
            node.posterior = node.prior.copy()
        
        # 우도 계산
        likelihoods = {}
        for state in node.prior.keys():
            if state == evidence.value:
                likelihoods[state] = weighted_strength
            else:
                likelihoods[state] = 1.0 - weighted_strength
        
        # 사후 확률 계산
        new_posterior = {}
        normalization = 0.0
        
        for state in node.prior.keys():
            new_posterior[state] = node.posterior[state] * likelihoods[state]
            normalization += new_posterior[state]
        
        # 정규화
        if normalization > 0:
            for state in new_posterior:
                new_posterior[state] /= normalization
                # 스무딩 적용
                new_posterior[state] = (
                    new_posterior[state] * (1 - self.config['smoothing_factor']) +
                    self.config['smoothing_factor'] / len(node.prior)
                )
        
        node.posterior = new_posterior
        node.evidence_count += 1
        node.last_updated = datetime.now()
        
        # 신뢰도 업데이트
        node.confidence = min(1.0, node.confidence + 0.05)
        
        # 자식 노드들에 전파
        await self._propagate_beliefs(node)
    
    async def _propagate_beliefs(self, updated_node: BayesianNode):
        """믿음 전파 (Belief Propagation)"""
        # 메시지 패싱을 통한 자식 노드 업데이트
        for child_id in updated_node.children:
            child_node = self.nodes[child_id]
            
            # 부모로부터의 메시지 계산
            message = self._calculate_message(updated_node, child_node)
            
            # 자식 노드의 믿음 업데이트
            await self._update_child_belief(child_node, updated_node, message)
    
    def _calculate_message(self, parent: BayesianNode, child: BayesianNode) -> Dict[str, float]:
        """부모에서 자식으로의 메시지 계산"""
        message = {}
        
        for child_state in child.prior.keys():
            prob = 0.0
            for parent_state in parent.posterior.keys():
                # P(child|parent) * P(parent)
                likelihood = child.likelihood.get(parent_state, {}).get(child_state, 0.5)
                prob += likelihood * parent.posterior[parent_state]
            
            message[child_state] = prob
        
        return message
    
    async def _update_child_belief(self, child: BayesianNode, parent: BayesianNode, message: Dict[str, float]):
        """자식 노드의 믿음 업데이트"""
        if not child.posterior:
            child.posterior = child.prior.copy()
        
        # 모든 부모로부터의 메시지 결합
        combined_message = message.copy()
        
        # 다른 부모들의 영향도 고려
        for other_parent_id in child.parents:
            if other_parent_id != parent.id:
                other_parent = self.nodes[other_parent_id]
                if other_parent.posterior:
                    other_message = self._calculate_message(other_parent, child)
                    for state in combined_message:
                        combined_message[state] *= other_message.get(state, 1.0)
        
        # 사후 확률 업데이트
        normalization = sum(combined_message.values())
        if normalization > 0:
            for state in child.posterior:
                child.posterior[state] = combined_message.get(state, 0.0) / normalization
        
        child.last_updated = datetime.now()
    
    def _calculate_time_weight(self, timestamp: datetime) -> float:
        """시간 가중치 계산"""
        age = (datetime.now() - timestamp).total_seconds()
        decay_factor = self.config['evidence_decay'] ** (age / 86400)  # 일 단위 감쇠
        return max(0.1, decay_factor)
    
    async def infer(self, 
                   query_node: str,
                   given_evidence: Dict[str, Any] = None,
                   context: Dict[str, Any] = None) -> InferenceResult:
        """베이지안 추론 수행"""
        start_time = time.time()
        
        try:
            # 캐시 확인
            cache_key = f"{query_node}_{str(given_evidence)}_{str(context)}"
            if cache_key in self.inference_cache:
                cached_result, cache_time = self.inference_cache[cache_key]
                if time.time() - cache_time < self.cache_ttl:
                    return cached_result
            
            # 쿼리 노드 확인
            if query_node not in self.nodes:
                raise ValueError(f"존재하지 않는 노드: {query_node}")
            
            target_node = self.nodes[query_node]
            
            # 임시 증거 추가
            temp_evidence_ids = []
            if given_evidence:
                for node_name, value in given_evidence.items():
                    if node_name in self.nodes:
                        temp_evidence = Evidence(
                            node_id=node_name,
                            value=value,
                            strength=0.9,
                            source="inference_query"
                        )
                        await self.add_evidence(temp_evidence)
                        temp_evidence_ids.append(temp_evidence.id)
            
            # 문학적 맥락 적용
            if context and 'literary_context' in context:
                await self._apply_literary_context(target_node, context['literary_context'])
            
            # Junction Tree 또는 Variable Elimination으로 추론
            posterior = await self._exact_inference(target_node)
            
            # 불확실성 계산 (엔트로피)
            uncertainty = self._calculate_entropy(posterior)
            
            # 영향력 있는 요인 분석
            influential_factors = await self._analyze_influential_factors(target_node)
            
            # 추론 경로 추적
            reasoning_path = self._trace_reasoning_path(target_node)
            
            # 결과 생성
            result = InferenceResult(
                query_node=query_node,
                posterior_distribution=posterior,
                confidence=target_node.confidence,
                uncertainty=uncertainty,
                influential_factors=influential_factors,
                reasoning_path=reasoning_path,
                computation_time=time.time() - start_time
            )
            
            # 캐시 저장
            self.inference_cache[cache_key] = (result, time.time())
            
            # 임시 증거 제거
            self.evidence_history = [
                e for e in self.evidence_history 
                if e.id not in temp_evidence_ids
            ]
            
            self.learning_metrics['total_inferences'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"추론 실패: {e}")
            # 기본 결과 반환
            return InferenceResult(
                query_node=query_node,
                posterior_distribution=target_node.prior.copy(),
                confidence=0.0,
                uncertainty=1.0,
                influential_factors=[],
                reasoning_path=[],
                computation_time=time.time() - start_time
            )
    
    async def _apply_literary_context(self, node: BayesianNode, literary_context: Dict[str, Any]):
        """문학적 맥락 적용"""
        # 문학적 사전 확률 가져오기
        if 'belief_type' in literary_context:
            literary_prior = self.literary_network.get_literary_prior(
                literary_context['belief_type'],
                literary_context
            )
            
            # 기존 사전 확률과 결합
            for state in node.prior:
                if state in literary_prior:
                    # 가중 평균
                    weight = literary_context.get('weight', 0.3)
                    node.prior[state] = (
                        node.prior[state] * (1 - weight) +
                        literary_prior[state] * weight
                    )
        
        # 문학적 출처 추가
        if 'source' in literary_context:
            node.literary_sources.append(literary_context['source'])
    
    async def _exact_inference(self, target_node: BayesianNode) -> Dict[str, float]:
        """정확한 추론 (Variable Elimination)"""
        # 단순한 경우: 부모가 없거나 적은 경우
        if not target_node.parents:
            return target_node.posterior if target_node.posterior else target_node.prior
        
        # 복잡한 경우: 모든 부모의 영향 고려
        posterior = {}
        
        for state in target_node.prior:
            prob = target_node.prior[state]
            
            # 각 부모의 영향 계산
            for parent_id in target_node.parents:
                parent = self.nodes[parent_id]
                parent_dist = parent.posterior if parent.posterior else parent.prior
                
                # 가중합
                parent_influence = 0.0
                for parent_state, parent_prob in parent_dist.items():
                    likelihood = target_node.likelihood.get(
                        parent_state, {}
                    ).get(state, 0.5)
                    parent_influence += likelihood * parent_prob
                
                prob *= parent_influence
            
            posterior[state] = prob
        
        # 정규화
        total = sum(posterior.values())
        if total > 0:
            posterior = {k: v/total for k, v in posterior.items()}
        
        return posterior
    
    def _calculate_entropy(self, distribution: Dict[str, float]) -> float:
        """엔트로피 계산 (불확실성 측정)"""
        entropy = 0.0
        for prob in distribution.values():
            if prob > 0:
                entropy -= prob * np.log(prob)
        
        # 0-1로 정규화
        max_entropy = np.log(len(distribution))
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    async def _analyze_influential_factors(self, target_node: BayesianNode) -> List[Tuple[str, float]]:
        """영향력 있는 요인 분석"""
        factors = []
        
        # 직접 부모들의 영향력
        for parent_id in target_node.parents:
            parent = self.nodes[parent_id]
            
            # 상호 정보량으로 영향력 측정
            influence = await self._calculate_mutual_information(parent, target_node)
            factors.append((parent.name or parent.id, influence))
        
        # 최근 증거의 영향력
        recent_evidence = self.evidence_cache.get(target_node.id, [])[-5:]
        for evidence in recent_evidence:
            factors.append((f"evidence_{evidence.source}", evidence.strength))
        
        # 영향력 순으로 정렬
        factors.sort(key=lambda x: x[1], reverse=True)
        
        return factors[:5]  # 상위 5개
    
    async def _calculate_mutual_information(self, node1: BayesianNode, node2: BayesianNode) -> float:
        """상호 정보량 계산"""
        mi = 0.0
        
        dist1 = node1.posterior if node1.posterior else node1.prior
        dist2 = node2.posterior if node2.posterior else node2.prior
        
        # 간단한 근사: 독립성 가정 하에서의 차이
        for state1 in dist1:
            for state2 in dist2:
                joint_prob = dist1[state1] * dist2[state2]  # 독립 가정
                
                # 실제 조건부 확률이 있으면 사용
                if state1 in node2.likelihood:
                    actual_prob = node2.likelihood[state1].get(state2, joint_prob)
                    joint_prob = dist1[state1] * actual_prob
                
                if joint_prob > 0:
                    mi += joint_prob * np.log(
                        joint_prob / (dist1[state1] * dist2[state2])
                    )
        
        return max(0.0, mi)
    
    def _trace_reasoning_path(self, target_node: BayesianNode) -> List[str]:
        """추론 경로 추적"""
        path = []
        visited = set()
        
        def dfs(node_id: str, depth: int = 0):
            if depth > 5 or node_id in visited:  # 최대 깊이 제한
                return
            
            visited.add(node_id)
            node = self.nodes[node_id]
            
            # 경로에 추가
            path.append(f"{'  ' * depth}{node.name or node.id}")
            
            # 부모 노드 탐색
            for parent_id in node.parents:
                dfs(parent_id, depth + 1)
        
        dfs(target_node.id)
        return path
    
    async def explain_inference(self, result: InferenceResult) -> str:
        """추론 결과 설명 생성"""
        explanation = []
        
        explanation.append(f"=== {result.query_node}에 대한 베이지안 추론 결과 ===\n")
        
        # 사후 확률 분포
        explanation.append("사후 확률 분포:")
        for state, prob in sorted(result.posterior_distribution.items(), 
                                 key=lambda x: x[1], reverse=True):
            explanation.append(f"  - {state}: {prob:.3f}")
        
        # 신뢰도와 불확실성
        explanation.append(f"\n신뢰도: {result.confidence:.3f}")
        explanation.append(f"불확실성 (엔트로피): {result.uncertainty:.3f}")
        
        # 주요 영향 요인
        if result.influential_factors:
            explanation.append("\n주요 영향 요인:")
            for factor, influence in result.influential_factors:
                explanation.append(f"  - {factor}: {influence:.3f}")
        
        # 추론 경로
        if result.reasoning_path:
            explanation.append("\n추론 경로:")
            for step in result.reasoning_path:
                explanation.append(f"  {step}")
        
        explanation.append(f"\n계산 시간: {result.computation_time:.3f}초")
        
        return "\n".join(explanation)
    
    async def update_from_outcome(self, 
                                prediction_node: str,
                                predicted_value: Any,
                                actual_value: Any,
                                context: Dict[str, Any] = None):
        """실제 결과를 통한 학습"""
        try:
            # 예측 정확도 계산
            is_correct = predicted_value == actual_value
            
            # 증거로 추가
            evidence = Evidence(
                node_id=prediction_node,
                value=actual_value,
                strength=1.0,
                source="actual_outcome",
                context=context or {}
            )
            await self.add_evidence(evidence)
            
            # 학습 메트릭 업데이트
            if is_correct:
                self.learning_metrics['successful_predictions'] += 1
            
            accuracy = (self.learning_metrics['successful_predictions'] / 
                       max(1, self.learning_metrics['total_inferences']))
            self.learning_metrics['accuracy_history'].append(accuracy)
            
            logger.info(f"예측 결과 학습: 정확도={accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"결과 학습 실패: {e}")
    
    def visualize_network(self) -> Dict[str, Any]:
        """네트워크 시각화 데이터 생성"""
        visualization = {
            'nodes': [],
            'edges': [],
            'clusters': defaultdict(list)
        }
        
        # 노드 정보
        for node_id, node in self.nodes.items():
            if isinstance(node, BayesianNode):
                node_data = {
                    'id': node.id,
                    'name': node.name,
                    'type': node.belief_type.value,
                    'confidence': node.confidence,
                    'evidence_count': node.evidence_count
                }
                
                # 현재 믿음 상태
                if node.posterior:
                    max_state = max(node.posterior.items(), key=lambda x: x[1])
                    node_data['current_belief'] = {
                        'state': max_state[0],
                        'probability': max_state[1]
                    }
                
                visualization['nodes'].append(node_data)
                visualization['clusters'][node.belief_type.value].append(node.id)
        
        # 엣지 정보
        for parent_id, child_id in self.edges:
            visualization['edges'].append({
                'source': parent_id,
                'target': child_id,
                'strength': 0.5  # 기본값, 실제로는 상호정보량 등으로 계산 가능
            })
        
        return visualization
    
    async def save_network(self, filepath: str = None):
        """네트워크 저장"""
        if filepath is None:
            filepath = os.path.join(
                MODELS_DIR,
                f"bayesian_network_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            )
        
        try:
            network_data = {
                'nodes': {k: asdict(v) if isinstance(v, BayesianNode) else v 
                         for k, v in self.nodes.items()},
                'edges': self.edges,
                'evidence_history': [asdict(e) for e in self.evidence_history[-1000:]],
                'learning_metrics': self.learning_metrics,
                'config': self.config
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(network_data, f)
            
            logger.info(f"베이지안 네트워크가 저장되었습니다: {filepath}")
            
        except Exception as e:
            logger.error(f"네트워크 저장 실패: {e}")
    
    async def load_network(self, filepath: str):
        """네트워크 로드"""
        try:
            with open(filepath, 'rb') as f:
                network_data = pickle.load(f)
            
            # 노드 복원
            self.nodes = {}
            for node_id, node_data in network_data['nodes'].items():
                if isinstance(node_data, dict) and 'belief_type' in node_data:
                    node_data['belief_type'] = BeliefType(node_data['belief_type'])
                    node = BayesianNode(**node_data)
                    self.nodes[node_id] = node
            
            # 엣지 복원
            self.edges = network_data['edges']
            
            # 증거 히스토리 복원
            self.evidence_history = [
                Evidence(**e) for e in network_data.get('evidence_history', [])
            ]
            
            # 학습 메트릭 복원
            self.learning_metrics = network_data.get('learning_metrics', {})
            
            logger.info(f"베이지안 네트워크가 로드되었습니다: {filepath}")
            
        except Exception as e:
            logger.error(f"네트워크 로드 실패: {e}")

    def infer_sync(self, 
                   query_node: str,
                   given_evidence: Dict[str, Any] = None,
                   context: Dict[str, Any] = None) -> InferenceResult:
        """동기 버전 베이지안 추론"""
        import asyncio
        try:
            # 기존 이벤트 루프가 있는지 확인
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 이미 실행 중인 루프가 있으면 새 스레드에서 실행
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        lambda: asyncio.run(self.infer(query_node, given_evidence, context))
                    )
                    return future.result()
            else:
                # 루프가 없으면 직접 실행
                return loop.run_until_complete(self.infer(query_node, given_evidence, context))
        except RuntimeError:
            # 이벤트 루프 문제가 있으면 새로운 루프 생성
            return asyncio.run(self.infer(query_node, given_evidence, context))

async def test_bayesian_inference():
    """베이지안 추론 시스템 테스트"""
    system = AdvancedBayesianInference()
    
    logger.info("=== 베이지안 추론 시스템 테스트 ===")
    
    # 1. 증거 추가 테스트
    logger.info("\n1. 증거 추가")
    
    # 감정 상태에 대한 증거
    await system.add_evidence(Evidence(
        node_id="emotional_state",
        value="negative",
        strength=0.8,
        source="literary_analysis",
        context={"text": "햄릿의 고뇌"}
    ))
    
    # 도덕적 판단에 대한 증거
    await system.add_evidence(Evidence(
        node_id="moral_judgment",
        value="wrong",
        strength=0.7,
        source="character_action",
        context={"action": "복수 계획"}
    ))
    
    # 2. 추론 테스트
    logger.info("\n2. 행동 경향 추론")
    
    result1 = await system.infer(
        "action_tendency",
        given_evidence={"emotional_state": "negative"},
        context={"literary_context": {"belief_type": "tragic_fate", "source": "hamlet"}}
    )
    
    explanation1 = await system.explain_inference(result1)
    logger.info(explanation1)
    
    # 3. 결과 예측 추론
    logger.info("\n3. 결과 예측 추론")
    
    result2 = await system.infer(
        "outcome_prediction",
        given_evidence={
            "emotional_state": "negative",
            "action_tendency": "avoid"
        }
    )
    
    explanation2 = await system.explain_inference(result2)
    logger.info(explanation2)
    
    # 4. 학습 테스트
    logger.info("\n4. 실제 결과로부터 학습")
    
    # 예측과 실제 결과 비교
    predicted = max(result2.posterior_distribution.items(), key=lambda x: x[1])[0]
    actual = "failure"
    
    await system.update_from_outcome(
        "outcome_prediction",
        predicted,
        actual,
        context={"scenario": "tragic_ending"}
    )
    
    # 5. 네트워크 시각화 데이터
    logger.info("\n5. 네트워크 구조")
    viz_data = system.visualize_network()
    logger.info(f"노드 수: {len(viz_data['nodes'])}")
    logger.info(f"엣지 수: {len(viz_data['edges'])}")
    logger.info(f"믿음 유형별 클러스터: {dict(viz_data['clusters'])}")
    
    # 6. 저장
    await system.save_network()
    
    return system

if __name__ == "__main__":
    # 테스트 실행
    asyncio.run(test_bayesian_inference())