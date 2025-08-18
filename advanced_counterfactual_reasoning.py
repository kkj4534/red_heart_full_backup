"""
고급 반사실적 추론 시스템 - Linux 전용
Advanced Counterfactual Reasoning System for Linux

문학 데이터 기반 "만약 다른 선택을 했다면?" 시나리오 생성 및 분석
기존 Rumbaugh 기반 구조를 문학적 맥락으로 정교화
"""

import os
import json
import time
import asyncio
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
import uuid
import re
from collections import defaultdict, deque
from enum import Enum
import copy

# 고급 분석 라이브러리
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
# SentenceTransformer는 sentence_transformer_singleton을 통해 사용
import networkx as nx

from config import ADVANCED_CONFIG, MODELS_DIR
from data_models import (
    Decision, EthicalSituation, HedonicValues, Experience,
    EmotionData, EmotionState, DecisionLog
)
from advanced_bentham_calculator import AdvancedBenthamCalculator

logger = logging.getLogger('RedHeart.AdvancedCounterfactual')

class ScenarioType(Enum):
    """시나리오 유형"""
    MORAL_DILEMMA = "moral_dilemma"          # 도덕적 딜레마
    INTERPERSONAL = "interpersonal"          # 인간관계
    SACRIFICE = "sacrifice"                  # 희생/포기
    LOYALTY_CONFLICT = "loyalty_conflict"    # 충성심 갈등
    TRUTH_VS_KINDNESS = "truth_vs_kindness" # 진실 vs 친절
    DUTY_VS_DESIRE = "duty_vs_desire"       # 의무 vs 욕망

class ConfidenceLevel(Enum):
    """확신도 수준"""
    VERY_LOW = 0.2    # 매우 낮음
    LOW = 0.4         # 낮음  
    MEDIUM = 0.6      # 보통
    HIGH = 0.8        # 높음
    VERY_HIGH = 0.95  # 매우 높음

@dataclass
class LiteraryContext:
    """문학적 맥락 정보"""
    literary_work: str = ""           # 작품명
    author: str = ""                  # 작가
    genre: str = ""                   # 장르
    cultural_period: str = ""         # 문화적 시대
    narrative_perspective: str = ""    # 서술 관점
    themes: List[str] = field(default_factory=list)        # 주제들
    character_archetypes: List[str] = field(default_factory=list) # 인물 유형

@dataclass
class SituationHypothesis:
    """향상된 상황 가설"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    base_situation: Dict[str, Any] = field(default_factory=dict)
    hypothesized_factors: Dict[str, Any] = field(default_factory=dict)
    
    # 문학적 요소들
    literary_context: LiteraryContext = field(default_factory=LiteraryContext)
    narrative_elements: Dict[str, Any] = field(default_factory=dict)
    
    # 개선된 분석
    plausibility_score: float = 0.5
    evidence_support: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    causal_chains: List[Dict[str, Any]] = field(default_factory=list)
    
    # 감정적/도덕적 차원
    emotional_impact: Dict[str, float] = field(default_factory=dict)
    moral_weight: float = 0.5
    cultural_relevance: float = 0.5
    
    def get_complete_situation(self) -> Dict[str, Any]:
        """기본 상황과 가설을 합친 완전한 상황 반환"""
        complete = copy.deepcopy(self.base_situation)
        complete.update(self.hypothesized_factors)
        complete['literary_context'] = asdict(self.literary_context)
        complete['narrative_elements'] = self.narrative_elements
        return complete

@dataclass
class ActionCandidate:
    """향상된 행위 후보"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    action_type: str = ""
    description: str = ""
    
    # 문학적 분석
    literary_precedents: List[str] = field(default_factory=list)  # 문학적 전례
    character_motivation: Dict[str, Any] = field(default_factory=dict)
    narrative_function: str = ""  # 서사적 기능
    
    # 윤리적/실용적 분석
    ethical_constraints: List[str] = field(default_factory=list)
    feasibility_score: float = 1.0
    expected_outcomes: Dict[str, Any] = field(default_factory=dict)
    stakeholders_affected: List[str] = field(default_factory=list)
    
    # 감정적 차원
    emotional_consequences: Dict[str, float] = field(default_factory=dict)
    empathy_requirement: float = 0.5
    
    # 장기적 영향
    short_term_effects: List[str] = field(default_factory=list)
    long_term_effects: List[str] = field(default_factory=list)
    ripple_effects: List[str] = field(default_factory=list)

@dataclass
class CounterfactualScenario:
    """향상된 반사실적 시나리오"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    hypothesis: SituationHypothesis = field(default_factory=SituationHypothesis)
    action: ActionCandidate = field(default_factory=ActionCandidate)
    
    # 결과 예측
    hedonic_score: float = 0.0
    moral_score: float = 0.0
    practical_score: float = 0.0
    
    # 상세 분석
    detailed_outcomes: Dict[str, Any] = field(default_factory=dict)
    causal_pathways: List[Dict[str, Any]] = field(default_factory=list)
    alternative_endings: List[str] = field(default_factory=list)
    
    # 문학적 분석
    narrative_coherence: float = 0.5
    thematic_relevance: float = 0.5
    character_development: Dict[str, float] = field(default_factory=dict)
    
    # 메타 분석
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    scenario_type: ScenarioType = ScenarioType.MORAL_DILEMMA
    learning_value: float = 0.5  # 이 시나리오에서 배울 수 있는 가치

@dataclass
class CounterfactualResult:
    """향상된 반사실적 추론 결과"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # 주요 결과
    selected_scenario: Optional[CounterfactualScenario] = None
    alternative_scenarios: List[CounterfactualScenario] = field(default_factory=list)
    
    # 분석 과정
    all_hypotheses: List[SituationHypothesis] = field(default_factory=list)
    all_action_candidates: Dict[str, List[ActionCandidate]] = field(default_factory=dict)
    
    # 의사결정 근거
    decision_rationale: str = ""
    selection_criteria: Dict[str, float] = field(default_factory=dict)
    trade_offs: List[str] = field(default_factory=list)
    
    # 성능 지표
    computation_time: float = 0.0
    confidence_score: float = 0.0
    scenarios_explored: int = 0
    
    # 학습 요소
    lessons_learned: List[str] = field(default_factory=list)
    patterns_discovered: List[str] = field(default_factory=list)
    cultural_insights: List[str] = field(default_factory=list)

@dataclass
class RegretMemory:
    """향상된 후회 메모리"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # 기본 정보
    original_option: ActionCandidate = field(default_factory=ActionCandidate)
    alternative_option: ActionCandidate = field(default_factory=ActionCandidate)
    
    # 후회 분석
    regret_intensity: float = 0.0
    regret_type: str = ""  # "action", "inaction", "timing", "method"
    
    # 상세 결과
    actual_outcome: Dict[str, Any] = field(default_factory=dict)
    imagined_outcome: Dict[str, Any] = field(default_factory=dict)
    emotional_impact: Dict[str, float] = field(default_factory=dict)
    
    # 문학적/문화적 맥락
    literary_parallels: List[str] = field(default_factory=list)
    cultural_expectations: Dict[str, float] = field(default_factory=dict)
    
    # 메타데이터
    counterfactual_assumptions: List[str] = field(default_factory=list)
    memory_weight: float = 0.5  # 0.0(망상) ~ 1.0(현실 기반)
    created_at: datetime = field(default_factory=datetime.now)
    literary_source: str = ""

class LiteraryHypothesisGenerator:
    """문학 기반 상황 가설 생성기"""
    
    def __init__(self):
        self.embedding_model = None
        self._initialize_embedding_model()
        
        # 문학적 패턴 데이터베이스
        self.literary_patterns = self._load_literary_patterns()
        self.character_archetypes = self._load_character_archetypes()
        self.narrative_structures = self._load_narrative_structures()
        
        # 분석 도구
        self.scenario_classifier = self._initialize_scenario_classifier()
        
        logger.info("문학 기반 가설 생성기가 초기화되었습니다.")
    
    def _initialize_embedding_model(self):
        """임베딩 모델 초기화"""
        try:
            from sentence_transformer_singleton import get_sentence_transformer
            
            self.embedding_model = get_sentence_transformer(
                'paraphrase-multilingual-mpnet-base-v2',
                device='cpu'
            )
        except Exception as e:
            logger.warning(f"임베딩 모델 로드 실패: {e}")
            self.embedding_model = None
    
    def _load_literary_patterns(self) -> Dict[str, List[str]]:
        """문학적 패턴 로드"""
        return {
            'tragic_irony': [
                "운명의 아이러니", "예상치 못한 결과", "선한 의도의 악한 결과"
            ],
            'moral_growth': [
                "시련을 통한 성장", "가치관의 변화", "자아 발견"
            ],
            'sacrifice_redemption': [
                "자기희생", "타인을 위한 포기", "구원의 행위"
            ],
            'love_duty_conflict': [
                "사랑과 의무의 갈등", "개인과 사회", "마음과 이성"
            ]
        }
    
    def _load_character_archetypes(self) -> Dict[str, Dict[str, Any]]:
        """캐릭터 원형 로드"""
        return {
            'hero': {
                'traits': ['용기', '정의감', '희생정신'],
                'motivations': ['선의 실현', '타인 보호', '명예'],
                'typical_dilemmas': ['개인 vs 공익', '원칙 vs 현실']
            },
            'mentor': {
                'traits': ['지혜', '경험', '인도력'],
                'motivations': ['제자 성장', '지식 전수', '올바른 길 안내'],
                'typical_dilemmas': ['진실 공개 여부', '개입 vs 자율성']
            },
            'innocent': {
                'traits': ['순수', '신뢰', '낙관'],
                'motivations': ['선한 믿음', '조화', '평화'],
                'typical_dilemmas': ['배신 당했을 때', '현실 직면']
            }
        }
    
    def _load_narrative_structures(self) -> Dict[str, List[str]]:
        """서사 구조 로드"""
        return {
            'three_act': ['설정', '갈등', '해결'],
            'heros_journey': ['소명', '시련', '변화', '귀환'],
            'tragedy': ['상승', '절정', '몰락', '파멸'],
            'comedy': ['혼란', '오해', '깨달음', '화해']
        }
    
    def _initialize_scenario_classifier(self):
        """시나리오 분류기 초기화"""
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=50, random_state=42)
    
    async def generate_hypotheses(self, base_situation: Dict[str, Any],
                                 literary_context: LiteraryContext,
                                 num_hypotheses: int = 5) -> List[SituationHypothesis]:
        """문학적 맥락을 고려한 상황 가설 생성"""
        try:
            hypotheses = []
            
            # 1. 기본 시나리오 분석
            scenario_type = await self._classify_scenario_type(base_situation, literary_context)
            
            # 2. 문학적 패턴 기반 가설 생성
            pattern_hypotheses = await self._generate_pattern_based_hypotheses(
                base_situation, literary_context, scenario_type, num_hypotheses // 2
            )
            hypotheses.extend(pattern_hypotheses)
            
            # 3. 캐릭터 원형 기반 가설 생성
            archetype_hypotheses = await self._generate_archetype_based_hypotheses(
                base_situation, literary_context, scenario_type, num_hypotheses // 2
            )
            hypotheses.extend(archetype_hypotheses)
            
            # 4. 변형 가설 생성 (창의적 시나리오)
            if len(hypotheses) < num_hypotheses:
                creative_hypotheses = await self._generate_creative_variations(
                    base_situation, literary_context, num_hypotheses - len(hypotheses)
                )
                hypotheses.extend(creative_hypotheses)
            
            # 5. 가설 타당성 검증 및 점수 계산
            for hypothesis in hypotheses:
                await self._validate_and_score_hypothesis(hypothesis, literary_context)
            
            # 6. 타당성 순으로 정렬
            hypotheses.sort(key=lambda h: h.plausibility_score, reverse=True)
            
            return hypotheses[:num_hypotheses]
            
        except Exception as e:
            logger.error(f"가설 생성 실패: {e}")
            return []
    
    async def _classify_scenario_type(self, situation: Dict[str, Any], 
                                    context: LiteraryContext) -> ScenarioType:
        """시나리오 유형 분류"""
        try:
            # 키워드 기반 분류
            situation_text = str(situation)
            
            if any(word in situation_text.lower() for word in ['도덕', '윤리', '옳다', '그르다']):
                return ScenarioType.MORAL_DILEMMA
            elif any(word in situation_text.lower() for word in ['관계', '친구', '가족', '사랑']):
                return ScenarioType.INTERPERSONAL
            elif any(word in situation_text.lower() for word in ['희생', '포기', '버리다']):
                return ScenarioType.SACRIFICE
            elif any(word in situation_text.lower() for word in ['충성', '배신', '의리']):
                return ScenarioType.LOYALTY_CONFLICT
            elif any(word in situation_text.lower() for word in ['진실', '거짓말', '친절']):
                return ScenarioType.TRUTH_VS_KINDNESS
            elif any(word in situation_text.lower() for word in ['의무', '욕망', '원하다']):
                return ScenarioType.DUTY_VS_DESIRE
            else:
                return ScenarioType.MORAL_DILEMMA  # 기본값
                
        except Exception as e:
            logger.error(f"시나리오 분류 실패: {e}")
            return ScenarioType.MORAL_DILEMMA
    
    async def _generate_pattern_based_hypotheses(self, base_situation: Dict[str, Any],
                                               context: LiteraryContext, 
                                               scenario_type: ScenarioType,
                                               count: int) -> List[SituationHypothesis]:
        """문학적 패턴 기반 가설 생성"""
        hypotheses = []
        
        try:
            # 관련 문학 패턴 선택
            relevant_patterns = self._select_relevant_patterns(scenario_type, context)
            
            for i, pattern in enumerate(relevant_patterns[:count]):
                hypothesis = SituationHypothesis(
                    base_situation=base_situation,
                    literary_context=context
                )
                
                # 패턴에 따른 가설적 요인 추가
                hypothesis.hypothesized_factors = await self._apply_literary_pattern(
                    base_situation, pattern, context
                )
                
                # 증거 및 가정 추가
                hypothesis.evidence_support = [f"문학적 패턴: {pattern}"]
                hypothesis.assumptions = [f"{pattern} 패턴의 전형적 전개를 따른다고 가정"]
                
                # 서사적 요소 추가
                hypothesis.narrative_elements = {
                    'pattern_type': pattern,
                    'expected_arc': self._get_pattern_arc(pattern),
                    'tension_points': self._identify_tension_points(pattern, base_situation)
                }
                
                hypotheses.append(hypothesis)
            
            return hypotheses
            
        except Exception as e:
            logger.error(f"패턴 기반 가설 생성 실패: {e}")
            return []
    
    def _select_relevant_patterns(self, scenario_type: ScenarioType, 
                                context: LiteraryContext) -> List[str]:
        """관련 문학 패턴 선택"""
        all_patterns = list(self.literary_patterns.keys())
        
        # 시나리오 유형에 따른 패턴 필터링
        if scenario_type == ScenarioType.MORAL_DILEMMA:
            return ['tragic_irony', 'moral_growth']
        elif scenario_type == ScenarioType.SACRIFICE:
            return ['sacrifice_redemption', 'tragic_irony']
        elif scenario_type == ScenarioType.INTERPERSONAL:
            return ['love_duty_conflict', 'moral_growth']
        else:
            return all_patterns[:2]  # 기본값
    
    async def _apply_literary_pattern(self, base_situation: Dict[str, Any],
                                    pattern: str, context: LiteraryContext) -> Dict[str, Any]:
        """문학적 패턴을 상황에 적용"""
        hypothesized_factors = {}
        
        if pattern == 'tragic_irony':
            hypothesized_factors.update({
                'hidden_consequence': '선한 의도가 예상치 못한 결과를 가져올 가능성',
                'ironic_reversal': '상황이 정반대로 전개될 수 있음',
                'fate_intervention': '운명적 개입이나 우연의 일치 가능성'
            })
        
        elif pattern == 'moral_growth':
            hypothesized_factors.update({
                'learning_opportunity': '이 경험을 통한 도덕적 성장 가능성',
                'value_clarification': '자신의 진정한 가치관 발견 기회',
                'wisdom_gain': '시행착오를 통한 지혜 획득'
            })
        
        elif pattern == 'sacrifice_redemption':
            hypothesized_factors.update({
                'redemptive_action': '희생을 통한 구원이나 회복 가능성',
                'greater_good': '개인의 희생이 더 큰 선을 가져올 수 있음',
                'moral_debt': '과거의 잘못을 만회할 기회'
            })
        
        elif pattern == 'love_duty_conflict':
            hypothesized_factors.update({
                'emotional_priority': '감정과 이성 중 어느 것을 우선시할지',
                'social_expectation': '사회적 기대와 개인적 욕구의 충돌',
                'long_term_happiness': '장기적 행복과 단기적 만족의 갈등'
            })
        
        return hypothesized_factors
    
    def _get_pattern_arc(self, pattern: str) -> List[str]:
        """패턴의 서사적 전개 구조"""
        arcs = {
            'tragic_irony': ['희망적 시작', '숨겨진 징조', '아이러니 전개', '비극적 결말'],
            'moral_growth': ['도덕적 혼란', '시련과 고민', '깨달음의 순간', '성장된 모습'],
            'sacrifice_redemption': ['죄책감/부채감', '희생의 결심', '구원적 행위', '회복/화해'],
            'love_duty_conflict': ['갈등 인식', '고민과 번민', '선택의 순간', '결과 수용']
        }
        return arcs.get(pattern, ['시작', '전개', '절정', '결말'])
    
    def _identify_tension_points(self, pattern: str, situation: Dict[str, Any]) -> List[str]:
        """긴장감 포인트 식별"""
        # 패턴별 긴장감 요소들
        tension_points = []
        
        if pattern == 'tragic_irony':
            tension_points = ['예상치 못한 변수', '선의의 역설', '운명의 개입']
        elif pattern == 'moral_growth':
            tension_points = ['가치관 충돌', '자아 의심', '타인의 판단']
        
        return tension_points

class AdvancedActionCandidateGenerator:
    """고급 행위 후보 생성기"""
    
    def __init__(self):
        self.ethical_frameworks = self._load_ethical_frameworks()
        self.cultural_norms = self._load_cultural_norms()
        self.literary_precedents = self._load_literary_precedents()
        
        logger.info("고급 행위 후보 생성기가 초기화되었습니다.")
    
    def _load_ethical_frameworks(self) -> Dict[str, Dict[str, Any]]:
        """윤리적 프레임워크 로드"""
        return {
            'consequentialist': {
                'focus': '결과 중심',
                'key_principles': ['최대 효용', '전체 행복'],
                'decision_criteria': ['예상 결과', '영향 범위', '확률']
            },
            'deontological': {
                'focus': '의무 중심', 
                'key_principles': ['도덕 법칙', '의무 준수'],
                'decision_criteria': ['원칙 일치', '보편적 적용 가능성']
            },
            'virtue_ethics': {
                'focus': '덕목 중심',
                'key_principles': ['인격 완성', '덕목 실천'],
                'decision_criteria': ['덕목 체현', '인격적 성장']
            }
        }
    
    def _load_cultural_norms(self) -> Dict[str, List[str]]:
        """문화적 규범 로드"""
        return {
            'korean_traditional': [
                '효도', '존경', '겸손', '조화', '집단 우선', '체면 중시'
            ],
            'western_individual': [
                '자율성', '개인 권리', '자기 실현', '공정성', '투명성'
            ],
            'confucian': [
                '인의예지', '충효', '수신제가', '사회 질서', '학습'
            ]
        }
    
    def _load_literary_precedents(self) -> Dict[str, List[Dict[str, Any]]]:
        """문학적 전례 로드"""
        return {
            'moral_dilemma': [
                {
                    'work': '춘향전',
                    'situation': '권력과 사랑의 갈등',
                    'action': '신의 지키기',
                    'outcome': '고난 후 보상'
                },
                {
                    'work': '햄릿',
                    'situation': '복수의 의무',
                    'action': '진실 추구',
                    'outcome': '비극적 정의'
                }
            ],
            'sacrifice': [
                {
                    'work': '엄마를 부탁해',
                    'situation': '가족을 위한 희생',
                    'action': '자기 포기',
                    'outcome': '사랑의 완성'
                }
            ]
        }
    
    async def generate_action_candidates(self, hypothesis: SituationHypothesis,
                                       max_candidates: int = 6) -> List[ActionCandidate]:
        """상황 가설에 대한 행위 후보들 생성"""
        try:
            candidates = []
            
            # 1. 윤리적 프레임워크별 행위 생성
            ethical_actions = await self._generate_ethical_framework_actions(hypothesis)
            candidates.extend(ethical_actions)
            
            # 2. 문화적 규범 기반 행위 생성
            cultural_actions = await self._generate_cultural_norm_actions(hypothesis)
            candidates.extend(cultural_actions)
            
            # 3. 문학적 전례 기반 행위 생성
            literary_actions = await self._generate_literary_precedent_actions(hypothesis)
            candidates.extend(literary_actions)
            
            # 4. 창의적 대안 행위 생성
            creative_actions = await self._generate_creative_alternatives(hypothesis)
            candidates.extend(creative_actions)
            
            # 5. 중복 제거 및 품질 평가
            unique_candidates = await self._deduplicate_and_evaluate(candidates)
            
            # 6. 실행 가능성 및 결과 예측
            for candidate in unique_candidates:
                await self._analyze_feasibility_and_outcomes(candidate, hypothesis)
            
            # 7. 상위 후보 선택
            unique_candidates.sort(key=lambda c: c.feasibility_score, reverse=True)
            
            return unique_candidates[:max_candidates]
            
        except Exception as e:
            logger.error(f"행위 후보 생성 실패: {e}")
            return []
    
    async def _generate_ethical_framework_actions(self, hypothesis: SituationHypothesis) -> List[ActionCandidate]:
        """윤리적 프레임워크 기반 행위 생성"""
        actions = []
        
        for framework_name, framework in self.ethical_frameworks.items():
            action = ActionCandidate(
                action_type=f"{framework_name}_action",
                description=f"{framework['focus']}에 따른 행위",
                narrative_function=f"{framework_name} 윤리학적 해결"
            )
            
            # 프레임워크별 구체적 행위 정의
            if framework_name == 'consequentialist':
                action.description = "가장 많은 사람에게 가장 큰 행복을 가져다 줄 행위 선택"
                action.expected_outcomes['utilitarian_benefit'] = 0.8
                
            elif framework_name == 'deontological':
                action.description = "도덕적 의무와 원칙에 따른 올바른 행위 실행"
                action.expected_outcomes['moral_righteousness'] = 0.9
                
            elif framework_name == 'virtue_ethics':
                action.description = "덕목을 체현하고 인격을 완성하는 행위 선택"
                action.expected_outcomes['character_growth'] = 0.85
            
            actions.append(action)
        
        return actions
    
    async def _generate_cultural_norm_actions(self, hypothesis: SituationHypothesis) -> List[ActionCandidate]:
        """문화적 규범 기반 행위 생성"""
        actions = []
        
        # 한국 전통 문화 기반 행위
        korean_action = ActionCandidate(
            action_type="korean_traditional",
            description="한국 전통 가치관에 따른 행위 (효도, 예의, 조화 중시)",
            narrative_function="전통적 해결책"
        )
        korean_action.cultural_relevance = 0.9
        korean_action.expected_outcomes['social_harmony'] = 0.8
        actions.append(korean_action)
        
        # 개인주의적 행위
        individual_action = ActionCandidate(
            action_type="individualistic",
            description="개인의 자율성과 권리를 우선시하는 행위",
            narrative_function="개인주의적 해결책"
        )
        individual_action.expected_outcomes['personal_autonomy'] = 0.85
        actions.append(individual_action)
        
        return actions
    
    async def _generate_literary_precedent_actions(self, hypothesis: SituationHypothesis) -> List[ActionCandidate]:
        """문학적 전례 기반 행위 생성"""
        actions = []
        
        # 상황 유형에 맞는 문학적 전례 찾기
        situation_type = self._classify_situation_for_precedents(hypothesis)
        precedents = self.literary_precedents.get(situation_type, [])
        
        for precedent in precedents[:2]:  # 상위 2개만
            action = ActionCandidate(
                action_type="literary_precedent",
                description=f"'{precedent['work']}'의 {precedent['action']} 방식을 따른 행위",
                narrative_function="문학적 전례 적용"
            )
            
            action.literary_precedents = [precedent['work']]
            action.expected_outcomes['narrative_satisfaction'] = 0.7
            action.character_motivation = {
                'literary_inspiration': precedent['work'],
                'archetypal_pattern': precedent['action']
            }
            
            actions.append(action)
        
        return actions
    
    def _classify_situation_for_precedents(self, hypothesis: SituationHypothesis) -> str:
        """문학적 전례를 위한 상황 분류"""
        situation_text = str(hypothesis.base_situation).lower()
        
        if any(word in situation_text for word in ['희생', '포기', '버리다']):
            return 'sacrifice'
        elif any(word in situation_text for word in ['도덕', '윤리', '딜레마']):
            return 'moral_dilemma'
        else:
            return 'moral_dilemma'  # 기본값
    
    async def _generate_creative_alternatives(self, hypothesis: SituationHypothesis) -> List[ActionCandidate]:
        """창의적 대안 행위 생성"""
        actions = []
        
        # 타협적 해결책
        compromise_action = ActionCandidate(
            action_type="compromise",
            description="상충하는 가치들 사이의 창의적 타협안 모색",
            narrative_function="제3의 길"
        )
        compromise_action.expected_outcomes['balanced_solution'] = 0.75
        actions.append(compromise_action)
        
        # 시간 지연 전략
        delay_action = ActionCandidate(
            action_type="temporal_delay",
            description="즉시 결정하지 않고 시간을 두고 상황 변화 관찰",
            narrative_function="시간의 지혜 활용"
        )
        delay_action.expected_outcomes['information_gain'] = 0.6
        actions.append(delay_action)
        
        return actions
    
    async def _deduplicate_and_evaluate(self, candidates: List[ActionCandidate]) -> List[ActionCandidate]:
        """중복 제거 및 평가"""
        unique_candidates = []
        seen_descriptions = set()
        
        for candidate in candidates:
            # 유사한 설명의 중복 제거
            if candidate.description not in seen_descriptions:
                unique_candidates.append(candidate)
                seen_descriptions.add(candidate.description)
        
        return unique_candidates
    
    async def _analyze_feasibility_and_outcomes(self, candidate: ActionCandidate, 
                                              hypothesis: SituationHypothesis):
        """실행 가능성 및 결과 분석"""
        try:
            # 실행 가능성 계산
            candidate.feasibility_score = await self._calculate_feasibility(candidate, hypothesis)
            
            # 감정적 결과 예측
            candidate.emotional_consequences = await self._predict_emotional_outcomes(candidate, hypothesis)
            
            # 장단기 효과 분석
            candidate.short_term_effects = await self._analyze_short_term_effects(candidate)
            candidate.long_term_effects = await self._analyze_long_term_effects(candidate)
            
        except Exception as e:
            logger.error(f"실행 가능성 분석 실패: {e}")
    
    async def _calculate_feasibility(self, candidate: ActionCandidate, 
                                   hypothesis: SituationHypothesis) -> float:
        """실행 가능성 계산"""
        base_feasibility = 0.7  # 기본 실행 가능성
        
        # 문화적 수용성 고려
        if candidate.action_type == "korean_traditional":
            base_feasibility += 0.2
        
        # 윤리적 제약 고려
        constraint_penalty = len(candidate.ethical_constraints) * 0.1
        
        return max(0.0, min(1.0, base_feasibility - constraint_penalty))
    
    async def _predict_emotional_outcomes(self, candidate: ActionCandidate, 
                                        hypothesis: SituationHypothesis) -> Dict[str, float]:
        """감정적 결과 예측"""
        emotional_outcomes = {
            'satisfaction': 0.5,
            'guilt': 0.3,
            'pride': 0.4,
            'regret': 0.2,
            'peace': 0.6
        }
        
        # 행위 유형에 따른 감정 조정
        if candidate.action_type.startswith('deontological'):
            emotional_outcomes['pride'] += 0.3
            emotional_outcomes['guilt'] -= 0.2
        
        return emotional_outcomes
    
    async def _analyze_short_term_effects(self, candidate: ActionCandidate) -> List[str]:
        """단기 효과 분석"""
        effects = []
        
        if candidate.action_type == "compromise":
            effects.extend(["부분적 만족", "추가 논의 필요", "임시적 해결"])
        elif candidate.action_type == "temporal_delay":
            effects.extend(["결정 연기", "추가 정보 수집", "불확실성 지속"])
        
        return effects
    
    async def _analyze_long_term_effects(self, candidate: ActionCandidate) -> List[str]:
        """장기 효과 분석"""
        effects = []
        
        if candidate.action_type.startswith('virtue'):
            effects.extend(["인격적 성장", "타인의 존경", "내적 만족"])
        elif candidate.action_type.startswith('consequentialist'):
            effects.extend(["실용적 결과", "효율성 증대", "최대 효용"])
        
        return effects

class AdvancedCounterfactualReasoning:
    """고급 반사실적 추론 시스템 메인 클래스"""
    
    def __init__(self):
        """시스템 초기화"""
        self.config = ADVANCED_CONFIG.get('counterfactual', {})
        self.models_dir = os.path.join(MODELS_DIR, 'counterfactual_models')
        os.makedirs(self.models_dir, exist_ok=True)
        
        # 구성 요소들 초기화
        self.hypothesis_generator = LiteraryHypothesisGenerator()
        self.action_generator = AdvancedActionCandidateGenerator()
        
        # 분석 도구들
        self.scenario_evaluator = self._initialize_scenario_evaluator()
        self.outcome_predictor = self._initialize_outcome_predictor()
        
        # 벤담 계산기 초기화
        from advanced_bentham_calculator import AdvancedBenthamCalculator
        self.bentham_calculator = AdvancedBenthamCalculator()
        
        # 메모리 시스템
        self.regret_memories = deque(maxlen=1000)
        self.learning_patterns = {}
        
        # 성능 추적
        self.performance_metrics = {
            'total_scenarios_generated': 0,
            'successful_predictions': 0,
            'average_confidence': 0.0,
            'processing_times': []
        }
        
        logger.info("고급 반사실적 추론 시스템이 초기화되었습니다.")
    
    def _initialize_scenario_evaluator(self):
        """시나리오 평가기 초기화"""
        # 간단한 평가 함수로 시작
        return {
            'hedonic_weight': 0.4,
            'moral_weight': 0.3,
            'practical_weight': 0.3
        }
    
    def _initialize_outcome_predictor(self):
        """결과 예측기 초기화"""
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=100, random_state=42)
    
    async def analyze_counterfactual_scenarios(self, 
                                             base_situation: Dict[str, Any],
                                             literary_context: LiteraryContext = None,
                                             options: Dict[str, Any] = None) -> CounterfactualResult:
        """
        메인 반사실적 시나리오 분석
        
        Args:
            base_situation: 기본 상황
            literary_context: 문학적 맥락
            options: 분석 옵션
            
        Returns:
            반사실적 추론 결과
        """
        start_time = time.time()
        session_id = str(uuid.uuid4())
        
        try:
            # 기본값 설정
            if literary_context is None:
                literary_context = LiteraryContext()
            
            if options is None:
                options = {
                    'num_hypotheses': 5,
                    'max_actions_per_hypothesis': 3,
                    'confidence_threshold': 0.6
                }
            
            logger.info(f"반사실적 분석 시작: {session_id}")
            
            # 1단계: 상황 가설 생성
            hypotheses = await self.hypothesis_generator.generate_hypotheses(
                base_situation, 
                literary_context,
                options.get('num_hypotheses', 5)
            )
            
            # 2단계: 각 가설에 대한 행위 후보 생성
            all_scenarios = []
            action_candidates_by_hypothesis = {}
            
            for hypothesis in hypotheses:
                action_candidates = await self.action_generator.generate_action_candidates(
                    hypothesis,
                    options.get('max_actions_per_hypothesis', 3)
                )
                
                action_candidates_by_hypothesis[hypothesis.id] = action_candidates
                
                # 3단계: 시나리오 생성 및 평가
                for action in action_candidates:
                    scenario = await self._create_and_evaluate_scenario(hypothesis, action)
                    all_scenarios.append(scenario)
            
            # 4단계: 최적 시나리오 선택
            selected_scenario = await self._select_optimal_scenario(
                all_scenarios, options.get('confidence_threshold', 0.6)
            )
            
            # 5단계: 결과 구성
            result = CounterfactualResult(
                session_id=session_id,
                selected_scenario=selected_scenario,
                alternative_scenarios=[s for s in all_scenarios if s != selected_scenario],
                all_hypotheses=hypotheses,
                all_action_candidates=action_candidates_by_hypothesis,
                decision_rationale=await self._generate_decision_rationale(selected_scenario, all_scenarios),
                computation_time=time.time() - start_time,
                confidence_score=selected_scenario.confidence_intervals.get('overall', (0.5, 0.8))[0] if selected_scenario else 0.0,
                scenarios_explored=len(all_scenarios)
            )
            
            # 6단계: 학습 및 메모리 업데이트
            await self._update_learning_patterns(result)
            
            # 성능 메트릭 업데이트
            self._update_performance_metrics(result)
            
            logger.info(f"반사실적 분석 완료: {len(all_scenarios)}개 시나리오 탐색")
            return result
            
        except Exception as e:
            logger.error(f"반사실적 분석 실패: {e}")
            return CounterfactualResult(
                session_id=session_id,
                computation_time=time.time() - start_time,
                decision_rationale=f"분석 실패: {str(e)}"
            )
    
    async def _create_and_evaluate_scenario(self, hypothesis: SituationHypothesis, 
                                          action: ActionCandidate) -> CounterfactualScenario:
        """시나리오 생성 및 평가"""
        scenario = CounterfactualScenario(
            hypothesis=hypothesis,
            action=action
        )
        
        # 결과 점수들 계산
        scenario.hedonic_score = await self._calculate_hedonic_score(hypothesis, action)
        scenario.moral_score = await self._calculate_moral_score(hypothesis, action)
        scenario.practical_score = await self._calculate_practical_score(hypothesis, action)
        
        # 상세 결과 예측
        scenario.detailed_outcomes = await self._predict_detailed_outcomes(hypothesis, action)
        
        # 인과 경로 분석
        scenario.causal_pathways = await self._analyze_causal_pathways(hypothesis, action)
        
        # 문학적 분석
        scenario.narrative_coherence = await self._assess_narrative_coherence(hypothesis, action)
        scenario.thematic_relevance = await self._assess_thematic_relevance(hypothesis, action)
        
        # 신뢰 구간 계산
        scenario.confidence_intervals = await self._calculate_confidence_intervals(scenario)
        
        # 학습 가치 평가
        scenario.learning_value = await self._assess_learning_value(scenario)
        
        return scenario
    
    async def _calculate_hedonic_score(self, hypothesis: SituationHypothesis, 
                                     action: ActionCandidate) -> float:
        """벤담 계산기를 활용한 쾌락 점수 계산"""
        try:
            # 행위 후보를 벤담 계산기 입력 형식으로 변환
            bentham_input = {
                'action_description': action.action_description,
                'context': {
                    'situation': hypothesis.situation_description,
                    'emotional_state': hypothesis.emotional_state,
                    'literary_context': hypothesis.literary_parallels,
                    'cultural_context': hypothesis.cultural_expectations
                },
                'expected_outcomes': action.expected_outcomes,
                'emotional_consequences': action.emotional_consequences,
                'time_horizon': action.time_horizon,
                'stakeholders': action.stakeholders,
                'confidence_level': action.confidence_level
            }
            
            # 벤담 계산기로 정교한 쾌락 점수 계산
            bentham_result = await self.bentham_calculator.calculate_enhanced(
                bentham_input,
                use_ai_weights=True,
                context_analysis=True
            )
            
            # 벤담 결과를 0-1 범위로 정규화
            hedonic_score = bentham_result.get('hedonic_score', 0.5)
            
            # 추가적인 문학적 맥락 고려
            literary_bonus = 0.0
            if hypothesis.literary_parallels:
                # 문학적 패턴이 있으면 보너스 (긍정적 결말 패턴)
                positive_patterns = ['redemption', 'growth', 'sacrifice_reward', 'wisdom_gained']
                for pattern in hypothesis.literary_parallels:
                    if any(pos in pattern.lower() for pos in positive_patterns):
                        literary_bonus += 0.05
            
            final_score = min(1.0, hedonic_score + literary_bonus)
            
            logger.debug(f"벤담 기반 쾌락 점수 계산 완료: {final_score:.3f}")
            return max(0.0, final_score)
            
        except Exception as e:
            logger.error(f"벤담 기반 쾌락 점수 계산 실패: {e}")
            # 폴백: 기존 방식으로 계산
            return await self._calculate_hedonic_score_fallback(hypothesis, action)
    
    async def _calculate_hedonic_score_fallback(self, hypothesis: SituationHypothesis, 
                                              action: ActionCandidate) -> float:
        """벤담 계산기 실패 시 폴백 쾌락 점수 계산"""
        try:
            # 감정적 결과 기반 계산
            emotional_outcomes = action.emotional_consequences
            
            positive_emotions = emotional_outcomes.get('satisfaction', 0) + emotional_outcomes.get('pride', 0) + emotional_outcomes.get('peace', 0)
            negative_emotions = emotional_outcomes.get('guilt', 0) + emotional_outcomes.get('regret', 0)
            
            hedonic_score = (positive_emotions - negative_emotions) / 3.0
            
            # 예상 결과의 긍정성 고려
            outcome_positivity = sum(v for k, v in action.expected_outcomes.items() if 'positive' in k or 'benefit' in k or 'growth' in k)
            
            # 최종 점수 (0-1 범위로 정규화)
            final_score = (hedonic_score * 0.7 + outcome_positivity * 0.3)
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            logger.error(f"폴백 쾌락 점수 계산 실패: {e}")
            return 0.5
    
    async def _calculate_moral_score(self, hypothesis: SituationHypothesis, 
                                   action: ActionCandidate) -> float:
        """도덕 점수 계산"""
        try:
            moral_score = 0.5  # 기본값
            
            # 윤리적 프레임워크 기반 평가
            if action.action_type.startswith('deontological'):
                moral_score += 0.3  # 의무론적 행위는 높은 도덕 점수
            elif action.action_type.startswith('virtue'):
                moral_score += 0.25  # 덕윤리 기반 행위
            elif action.action_type.startswith('consequentialist'):
                moral_score += 0.2   # 결과주의 행위
            
            # 문화적 규범 준수도 고려
            if action.action_type == 'korean_traditional':
                moral_score += 0.15
            
            # 윤리적 제약 위반 패널티
            constraint_penalty = len(action.ethical_constraints) * 0.1
            moral_score -= constraint_penalty
            
            return max(0.0, min(1.0, moral_score))
            
        except Exception as e:
            logger.error(f"도덕 점수 계산 실패: {e}")
            return 0.5
    
    async def _calculate_practical_score(self, hypothesis: SituationHypothesis, 
                                       action: ActionCandidate) -> float:
        """실용 점수 계산"""
        try:
            # 실행 가능성이 주요 지표
            practical_score = action.feasibility_score
            
            # 단기/장기 효과 고려
            short_term_bonus = len(action.short_term_effects) * 0.05
            long_term_bonus = len(action.long_term_effects) * 0.1
            
            practical_score += short_term_bonus + long_term_bonus
            
            # 이해관계자 수 고려
            stakeholder_factor = min(len(action.stakeholders_affected) * 0.1, 0.3)
            practical_score += stakeholder_factor
            
            return max(0.0, min(1.0, practical_score))
            
        except Exception as e:
            logger.error(f"실용 점수 계산 실패: {e}")
            return 0.5
    
    async def _predict_detailed_outcomes(self, hypothesis: SituationHypothesis, 
                                       action: ActionCandidate) -> Dict[str, Any]:
        """상세 결과 예측"""
        outcomes = {
            'immediate_results': action.short_term_effects,
            'long_term_results': action.long_term_effects,
            'emotional_impact': action.emotional_consequences,
            'social_consequences': [],
            'personal_growth': {},
            'relationship_changes': {}
        }
        
        # 사회적 결과 예측
        if action.action_type == 'korean_traditional':
            outcomes['social_consequences'] = ['사회적 인정', '전통 가치 보존']
        elif action.action_type == 'individualistic':
            outcomes['social_consequences'] = ['개인적 자유 확대', '사회적 논란 가능']
        
        # 개인적 성장 예측
        if action.action_type.startswith('virtue'):
            outcomes['personal_growth'] = {
                'moral_development': 0.8,
                'wisdom_gain': 0.7,
                'character_strength': 0.75
            }
        
        return outcomes
    
    async def _analyze_causal_pathways(self, hypothesis: SituationHypothesis, 
                                     action: ActionCandidate) -> List[Dict[str, Any]]:
        """인과 경로 분석"""
        pathways = []
        
        # 기본 인과 경로: 행위 → 즉시 결과 → 장기 결과
        basic_pathway = {
            'pathway_id': str(uuid.uuid4()),
            'sequence': [
                f"행위: {action.description}",
                f"즉시 결과: {', '.join(action.short_term_effects[:2])}",
                f"장기 결과: {', '.join(action.long_term_effects[:2])}"
            ],
            'probability': action.feasibility_score,
            'impact_strength': 0.7
        }
        pathways.append(basic_pathway)
        
        # 감정적 인과 경로
        if action.emotional_consequences:
            emotional_pathway = {
                'pathway_id': str(uuid.uuid4()),
                'sequence': [
                    f"행위: {action.description}",
                    f"감정적 반응: {max(action.emotional_consequences, key=action.emotional_consequences.get)}",
                    f"행동 변화: 감정 기반 후속 행동"
                ],
                'probability': 0.6,
                'impact_strength': 0.5
            }
            pathways.append(emotional_pathway)
        
        return pathways
    
    async def _assess_narrative_coherence(self, hypothesis: SituationHypothesis, 
                                        action: ActionCandidate) -> float:
        """서사적 일관성 평가"""
        coherence = 0.5  # 기본값
        
        # 문학적 전례가 있으면 일관성 높음
        if action.literary_precedents:
            coherence += 0.3
        
        # 캐릭터 동기와 행위의 일치성
        if action.character_motivation:
            coherence += 0.2
        
        # 서사적 기능의 명확성
        if action.narrative_function:
            coherence += 0.15
        
        return min(1.0, coherence)
    
    async def _assess_thematic_relevance(self, hypothesis: SituationHypothesis, 
                                       action: ActionCandidate) -> float:
        """주제적 관련성 평가"""
        relevance = 0.5  # 기본값
        
        # 문학적 맥락의 주제와 일치성
        if hypothesis.literary_context.themes:
            theme_match_count = 0
            for theme in hypothesis.literary_context.themes:
                if theme in action.description.lower():
                    theme_match_count += 1
            
            if theme_match_count > 0:
                relevance += min(theme_match_count * 0.2, 0.4)
        
        return min(1.0, relevance)
    
    async def _calculate_confidence_intervals(self, scenario: CounterfactualScenario) -> Dict[str, Tuple[float, float]]:
        """신뢰 구간 계산"""
        intervals = {}
        
        # 각 점수에 대한 신뢰 구간 (±0.1 범위로 단순화)
        intervals['hedonic'] = (
            max(0.0, scenario.hedonic_score - 0.1),
            min(1.0, scenario.hedonic_score + 0.1)
        )
        
        intervals['moral'] = (
            max(0.0, scenario.moral_score - 0.1),
            min(1.0, scenario.moral_score + 0.1)
        )
        
        intervals['practical'] = (
            max(0.0, scenario.practical_score - 0.1),
            min(1.0, scenario.practical_score + 0.1)
        )
        
        # 전체 신뢰도
        overall_score = (scenario.hedonic_score + scenario.moral_score + scenario.practical_score) / 3
        intervals['overall'] = (
            max(0.0, overall_score - 0.15),
            min(1.0, overall_score + 0.15)
        )
        
        return intervals
    
    async def _assess_learning_value(self, scenario: CounterfactualScenario) -> float:
        """학습 가치 평가"""
        learning_value = 0.5  # 기본값
        
        # 복잡성이 높을수록 학습 가치 높음
        complexity = len(scenario.causal_pathways) * 0.1
        learning_value += complexity
        
        # 도덕적 딜레마가 있으면 학습 가치 높음
        if scenario.scenario_type == ScenarioType.MORAL_DILEMMA:
            learning_value += 0.2
        
        # 문학적 일관성이 높으면 학습 가치 높음
        learning_value += scenario.narrative_coherence * 0.3
        
        return min(1.0, learning_value)
    
    async def _select_optimal_scenario(self, scenarios: List[CounterfactualScenario], 
                                     confidence_threshold: float) -> Optional[CounterfactualScenario]:
        """최적 시나리오 선택"""
        if not scenarios:
            return None
        
        # 각 시나리오의 종합 점수 계산
        for scenario in scenarios:
            weights = self.scenario_evaluator
            scenario.overall_score = (
                scenario.hedonic_score * weights['hedonic_weight'] +
                scenario.moral_score * weights['moral_weight'] +
                scenario.practical_score * weights['practical_weight']
            )
        
        # 신뢰도 필터링
        confident_scenarios = [
            s for s in scenarios 
            if s.confidence_intervals.get('overall', (0, 0))[0] >= confidence_threshold
        ]
        
        if not confident_scenarios:
            confident_scenarios = scenarios  # 신뢰도 조건을 만족하는 것이 없으면 전체에서 선택
        
        # 최고 점수 시나리오 선택
        return max(confident_scenarios, key=lambda s: getattr(s, 'overall_score', 0))
    
    async def _generate_decision_rationale(self, selected: Optional[CounterfactualScenario], 
                                         all_scenarios: List[CounterfactualScenario]) -> str:
        """의사결정 근거 생성"""
        if not selected:
            return "적절한 시나리오를 찾지 못했습니다."
        
        rationale_parts = [
            f"선택된 시나리오: {selected.action.description}",
            f"종합 점수: {getattr(selected, 'overall_score', 0):.3f}",
            f"쾌락 점수: {selected.hedonic_score:.3f}",
            f"도덕 점수: {selected.moral_score:.3f}",
            f"실용 점수: {selected.practical_score:.3f}"
        ]
        
        # 선택 이유
        if hasattr(selected, 'overall_score'):
            avg_score = np.mean([getattr(s, 'overall_score', 0) for s in all_scenarios])
            if selected.overall_score > avg_score + 0.1:
                rationale_parts.append("평균보다 현저히 높은 종합 점수로 인해 선택되었습니다.")
        
        # 특별한 장점
        if selected.narrative_coherence > 0.8:
            rationale_parts.append("높은 서사적 일관성을 보입니다.")
        
        if selected.learning_value > 0.7:
            rationale_parts.append("높은 학습 가치를 제공합니다.")
        
        return " ".join(rationale_parts)
    
    async def _update_learning_patterns(self, result: CounterfactualResult):
        """학습 패턴 업데이트"""
        try:
            if result.selected_scenario:
                # 선택된 시나리오의 패턴 기록
                pattern_key = f"{result.selected_scenario.scenario_type.value}_{result.selected_scenario.action.action_type}"
                
                if pattern_key not in self.learning_patterns:
                    self.learning_patterns[pattern_key] = {
                        'count': 0,
                        'avg_hedonic': 0.0,
                        'avg_moral': 0.0,
                        'avg_practical': 0.0,
                        'success_rate': 0.0
                    }
                
                pattern = self.learning_patterns[pattern_key]
                pattern['count'] += 1
                
                # 이동 평균 업데이트
                alpha = 1.0 / pattern['count']
                pattern['avg_hedonic'] = (1 - alpha) * pattern['avg_hedonic'] + alpha * result.selected_scenario.hedonic_score
                pattern['avg_moral'] = (1 - alpha) * pattern['avg_moral'] + alpha * result.selected_scenario.moral_score
                pattern['avg_practical'] = (1 - alpha) * pattern['avg_practical'] + alpha * result.selected_scenario.practical_score
                
                # 성공률 업데이트 (높은 종합 점수를 성공으로 간주)
                is_success = getattr(result.selected_scenario, 'overall_score', 0) > 0.7
                pattern['success_rate'] = (1 - alpha) * pattern['success_rate'] + alpha * (1.0 if is_success else 0.0)
            
        except Exception as e:
            logger.error(f"학습 패턴 업데이트 실패: {e}")
    
    def _update_performance_metrics(self, result: CounterfactualResult):
        """성능 메트릭 업데이트"""
        self.performance_metrics['total_scenarios_generated'] += result.scenarios_explored
        self.performance_metrics['processing_times'].append(result.computation_time)
        
        if result.selected_scenario:
            self.performance_metrics['successful_predictions'] += 1
        
        # 평균 신뢰도 업데이트
        if self.performance_metrics['total_scenarios_generated'] > 0:
            total_confidence = sum(
                [result.confidence_score] + 
                [getattr(s, 'overall_score', 0) for s in result.alternative_scenarios]
            )
            self.performance_metrics['average_confidence'] = total_confidence / result.scenarios_explored
    
    async def add_regret_memory(self, regret_memory: RegretMemory):
        """후회 메모리 추가"""
        try:
            self.regret_memories.append(regret_memory)
            
            # 유사한 상황에서의 학습 업데이트
            await self._update_from_regret(regret_memory)
            
            logger.info(f"후회 메모리 추가됨: {regret_memory.id}")
            
        except Exception as e:
            logger.error(f"후회 메모리 추가 실패: {e}")
    
    async def _update_from_regret(self, regret_memory: RegretMemory):
        """후회로부터 학습"""
        try:
            # 후회 강도가 높은 경우 해당 패턴의 가중치 조정
            if regret_memory.regret_intensity > 0.7:
                action_type = regret_memory.original_option.action_type
                
                # 해당 행위 유형의 가중치 감소
                for pattern_key, pattern_data in self.learning_patterns.items():
                    if action_type in pattern_key:
                        pattern_data['success_rate'] *= 0.9  # 성공률 감소
            
        except Exception as e:
            logger.error(f"후회 학습 실패: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 반환"""
        processing_times = self.performance_metrics['processing_times']
        
        return {
            'total_scenarios': self.performance_metrics['total_scenarios_generated'],
            'successful_predictions': self.performance_metrics['successful_predictions'],
            'success_rate': (
                self.performance_metrics['successful_predictions'] / 
                max(1, len(processing_times))
            ),
            'average_confidence': self.performance_metrics['average_confidence'],
            'average_processing_time': np.mean(processing_times) if processing_times else 0.0,
            'learning_patterns_count': len(self.learning_patterns),
            'regret_memories_count': len(self.regret_memories)
        }

def create_advanced_counterfactual_reasoning() -> AdvancedCounterfactualReasoning:
    """고급 반사실적 추론 시스템 생성"""
    return AdvancedCounterfactualReasoning()

# 테스트 코드
if __name__ == "__main__":
    import asyncio
    
    async def test_counterfactual_reasoning():
        """반사실적 추론 시스템 테스트"""
        
        # 시스템 초기화
        reasoning_system = create_advanced_counterfactual_reasoning()
        
        # 테스트 상황
        base_situation = {
            'description': '친구가 다른 친구의 비밀을 털어놓았을 때, 그 비밀을 당사자에게 말해야 할지 고민되는 상황',
            'stakeholders': ['나', '비밀을 말한 친구', '비밀의 당사자'],
            'constraints': ['친구관계 유지', '신뢰 보호', '진실 추구'],
            'context': '대학교 기숙사에서 발생한 상황'
        }
        
        # 문학적 맥락
        literary_context = LiteraryContext(
            literary_work="현대 한국 소설",
            themes=["우정", "진실", "도덕적 딜레마"],
            cultural_period="현대",
            narrative_perspective="1인칭"
        )
        
        print("=== 고급 반사실적 추론 시스템 테스트 ===\n")
        
        # 분석 실행
        result = await reasoning_system.analyze_counterfactual_scenarios(
            base_situation, literary_context
        )
        
        # 결과 출력
        print(f"세션 ID: {result.session_id}")
        print(f"처리 시간: {result.computation_time:.3f}초")
        print(f"탐색된 시나리오: {result.scenarios_explored}개")
        print(f"신뢰도: {result.confidence_score:.3f}\n")
        
        if result.selected_scenario:
            selected = result.selected_scenario
            print("=== 선택된 시나리오 ===")
            print(f"행위: {selected.action.description}")
            print(f"쾌락 점수: {selected.hedonic_score:.3f}")
            print(f"도덕 점수: {selected.moral_score:.3f}")
            print(f"실용 점수: {selected.practical_score:.3f}")
            print(f"서사적 일관성: {selected.narrative_coherence:.3f}")
            print(f"학습 가치: {selected.learning_value:.3f}")
            
            print(f"\n=== 예상 결과 ===")
            for key, value in selected.detailed_outcomes.items():
                print(f"{key}: {value}")
            
            print(f"\n=== 인과 경로 ===")
            for pathway in selected.causal_pathways:
                print(f"경로 {pathway['pathway_id'][:8]}:")
                for step in pathway['sequence']:
                    print(f"  → {step}")
                print(f"  확률: {pathway['probability']:.3f}")
        
        print(f"\n=== 의사결정 근거 ===")
        print(result.decision_rationale)
        
        print(f"\n=== 대안 시나리오들 ===")
        for i, alt in enumerate(result.alternative_scenarios[:3]):
            print(f"{i+1}. {alt.action.description}")
            print(f"   종합점수: {getattr(alt, 'overall_score', 0):.3f}")
        
        # 성능 요약
        performance = reasoning_system.get_performance_summary()
        print(f"\n=== 성능 요약 ===")
        for key, value in performance.items():
            print(f"{key}: {value}")
        
        print("\n테스트 완료!")
    
    # 비동기 테스트 실행
    asyncio.run(test_counterfactual_reasoning())