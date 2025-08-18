"""
심층 사고용 다차원 윤리 추론 시스템
Deep Multi-Dimensional Ethics Reasoning System

다양한 윤리적 관점을 통합하여 깊이 있는 윤리적 추론을 수행하는 시스템입니다.
공리주의, 덕 윤리학, 의무론적 윤리학, 돌봄 윤리학, 정의론 등을 종합적으로 고려합니다.

핵심 기능:
1. 다차원 윤리학파 통합 추론
2. 문화적 맥락 고려 윤리 판단
3. 이해관계자 관점 다각도 분석
4. 장단기 결과 예측 기반 윤리 평가
"""

import numpy as np
import torch
import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

try:
    from config import ADVANCED_CONFIG, DEVICE
except ImportError:
    # config.py에 문제가 있을 경우 기본값 사용
    ADVANCED_CONFIG = {'enable_gpu': False}
    DEVICE = 'cpu'
    print("⚠️  config.py 임포트 실패, 기본값 사용")
from data_models import EmotionData
from mixture_of_experts import create_ethics_moe, MixtureOfExperts

logger = logging.getLogger('DeepMultiDimensionalEthics')

class EthicsSchool(Enum):
    """윤리학파"""
    UTILITARIANISM = "utilitarianism"      # 공리주의
    VIRTUE_ETHICS = "virtue_ethics"        # 덕 윤리학
    DEONTOLOGICAL = "deontological"        # 의무론적 윤리학
    CARE_ETHICS = "care_ethics"            # 돌봄 윤리학
    JUSTICE_THEORY = "justice_theory"      # 정의론
    NARRATIVE_ETHICS = "narrative_ethics"  # 서사 윤리학
    FEMINIST_ETHICS = "feminist_ethics"    # 페미니스트 윤리학
    ENVIRONMENTAL_ETHICS = "environmental_ethics"  # 환경 윤리학

@dataclass
class StakeholderPerspective:
    """이해관계자 관점"""
    stakeholder_id: str
    name: str
    role: str
    power_level: float  # 0-1, 권력/영향력 수준
    vulnerability: float  # 0-1, 취약성 수준
    
    # 관점별 가치
    values: Dict[str, float] = field(default_factory=dict)
    concerns: List[str] = field(default_factory=list)
    
    # 예상 영향
    expected_benefits: float = 0.0
    expected_harms: float = 0.0
    
    # 의견 가중치
    voice_weight: float = 1.0

@dataclass
class CulturalContext:
    """문화적 맥락"""
    culture_id: str
    cultural_values: Dict[str, float] = field(default_factory=dict)
    social_norms: List[str] = field(default_factory=list)
    moral_priorities: Dict[str, float] = field(default_factory=dict)
    
    # 문화적 특성
    individualism_collectivism: float = 0.5  # 0: 집단주의, 1: 개인주의
    power_distance: float = 0.5  # 권력 거리
    uncertainty_avoidance: float = 0.5  # 불확실성 회피
    long_term_orientation: float = 0.5  # 장기 지향성

@dataclass
class EthicalDilemma:
    """윤리적 딜레마"""
    dilemma_id: str
    scenario: str
    context: str
    
    # 딜레마 특성
    complexity_level: float = 0.5
    urgency_level: float = 0.5
    reversibility: float = 0.5  # 결정의 가역성
    
    # 관련 정보
    stakeholders: List[StakeholderPerspective] = field(default_factory=list)
    cultural_context: Optional[CulturalContext] = None
    available_options: List[str] = field(default_factory=list)
    
    # 제약사항
    legal_constraints: List[str] = field(default_factory=list)
    resource_constraints: Dict[str, float] = field(default_factory=dict)
    time_constraints: float = 1.0  # 무제한 시간 = 1.0

@dataclass
class EthicsReasoning:
    """윤리 추론 결과"""
    school: EthicsSchool
    reasoning_process: List[str] = field(default_factory=list)
    ethical_score: float = 0.0
    confidence: float = 0.0
    key_principles: List[str] = field(default_factory=list)
    potential_conflicts: List[str] = field(default_factory=list)

@dataclass
class IntegratedEthicsResult:
    """통합 윤리 추론 결과"""
    dilemma: EthicalDilemma
    school_reasonings: Dict[EthicsSchool, EthicsReasoning] = field(default_factory=dict)
    
    # 통합 결과
    overall_recommendation: str = ""
    confidence_score: float = 0.0
    ethical_consensus: float = 0.0  # 학파 간 합의 정도
    
    # 상세 분석
    stakeholder_analysis: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    cultural_considerations: List[str] = field(default_factory=list)
    temporal_analysis: Dict[str, float] = field(default_factory=dict)
    
    # 메타 정보
    processing_time: float = 0.0
    reasoning_depth: int = 0

class EthicsReasoningEngine(ABC):
    """윤리 추론 엔진 추상 클래스"""
    
    @abstractmethod
    def reason(self, dilemma: EthicalDilemma) -> EthicsReasoning:
        """윤리적 추론 수행"""
        pass
    
    @abstractmethod
    def get_school(self) -> EthicsSchool:
        """윤리학파 반환"""
        pass

class UtilitarianEngine(EthicsReasoningEngine):
    """공리주의 추론 엔진"""
    
    def get_school(self) -> EthicsSchool:
        return EthicsSchool.UTILITARIANISM
    
    def reason(self, dilemma: EthicalDilemma) -> EthicsReasoning:
        """공리주의적 추론"""
        
        reasoning_process = ["공리주의적 관점에서 분석 시작"]
        
        # 1. 전체 효용 계산
        total_utility = 0.0
        stakeholder_count = len(dilemma.stakeholders)
        
        if stakeholder_count > 0:
            for stakeholder in dilemma.stakeholders:
                net_utility = stakeholder.expected_benefits - stakeholder.expected_harms
                # 취약성 가중치 적용
                weighted_utility = net_utility * (1.0 + stakeholder.vulnerability)
                total_utility += weighted_utility
            
            average_utility = total_utility / stakeholder_count
            reasoning_process.append(f"평균 효용 계산: {average_utility:.3f}")
        else:
            average_utility = 0.5
            reasoning_process.append("이해관계자 정보 부족으로 기본 효용 적용")
        
        # 2. 최대 행복 원칙 적용
        reasoning_process.append("최대 다수의 최대 행복 원칙 적용")
        
        # 3. 결과 기반 평가
        if average_utility > 0.6:
            recommendation = "공리주의적 관점에서 긍정적 결과 예상"
            ethical_score = min(average_utility, 1.0)
        elif average_utility < 0.4:
            recommendation = "공리주의적 관점에서 부정적 결과 예상"
            ethical_score = max(average_utility, 0.0)
        else:
            recommendation = "공리주의적 관점에서 중립적 결과 예상"
            ethical_score = 0.5
        
        reasoning_process.append(recommendation)
        
        # 잠재적 갈등
        conflicts = []
        if stakeholder_count > 5:
            conflicts.append("다수의 이해관계자로 인한 효용 계산 복잡성")
        
        # 취약 계층 보호 필요성
        vulnerable_stakeholders = [s for s in dilemma.stakeholders if s.vulnerability > 0.7]
        if vulnerable_stakeholders:
            conflicts.append(f"{len(vulnerable_stakeholders)}명의 취약 계층 특별 고려 필요")
        
        return EthicsReasoning(
            school=self.get_school(),
            reasoning_process=reasoning_process,
            ethical_score=ethical_score,
            confidence=0.8 if stakeholder_count > 0 else 0.5,
            key_principles=["최대 다수의 최대 행복", "결과 중심 판단", "효용 최대화"],
            potential_conflicts=conflicts
        )

class VirtueEthicsEngine(EthicsReasoningEngine):
    """덕 윤리학 추론 엔진"""
    
    def get_school(self) -> EthicsSchool:
        return EthicsSchool.VIRTUE_ETHICS
    
    def reason(self, dilemma: EthicalDilemma) -> EthicsReasoning:
        """덕 윤리학적 추론"""
        
        reasoning_process = ["덕 윤리학적 관점에서 분석 시작"]
        
        # 핵심 덕목들
        virtues = {
            'courage': 0.0,      # 용기
            'justice': 0.0,      # 정의
            'temperance': 0.0,   # 절제
            'wisdom': 0.0,       # 지혜
            'compassion': 0.0,   # 연민
            'integrity': 0.0,    # 성실성
            'humility': 0.0      # 겸손
        }
        
        # 시나리오 분석으로 덕목 평가
        scenario_lower = dilemma.scenario.lower()
        
        # 용기 평가
        if any(word in scenario_lower for word in ['위험', '도전', '어려운', '곤란']):
            virtues['courage'] = 0.8
            reasoning_process.append("상황이 용기를 요구함")
        
        # 정의 평가
        if any(word in scenario_lower for word in ['공정', '공평', '평등', '차별']):
            virtues['justice'] = 0.9
            reasoning_process.append("정의로운 판단이 핵심")
        
        # 절제 평가
        if any(word in scenario_lower for word in ['욕심', '탐욕', '과도']):
            virtues['temperance'] = 0.7
            reasoning_process.append("절제가 필요한 상황")
        
        # 지혜 평가
        if dilemma.complexity_level > 0.7:
            virtues['wisdom'] = 0.8
            reasoning_process.append("복잡한 상황으로 지혜가 요구됨")
        
        # 연민 평가
        vulnerable_count = len([s for s in dilemma.stakeholders if s.vulnerability > 0.6])
        if vulnerable_count > 0:
            virtues['compassion'] = min(0.9, 0.5 + vulnerable_count * 0.2)
            reasoning_process.append(f"취약 계층 {vulnerable_count}명으로 연민이 중요")
        
        # 성실성 평가 (기본값 높게)
        virtues['integrity'] = 0.8
        reasoning_process.append("성실성은 모든 윤리적 행동의 기반")
        
        # 겸손 평가
        if any(word in scenario_lower for word in ['권력', '지위', '우월']):
            virtues['humility'] = 0.7
            reasoning_process.append("권력 관련 상황에서 겸손이 필요")
        
        # 전체 덕목 점수 계산
        virtue_scores = [score for score in virtues.values() if score > 0]
        if virtue_scores:
            ethical_score = np.mean(virtue_scores)
            confidence = len(virtue_scores) / len(virtues)  # 평가된 덕목 비율
        else:
            ethical_score = 0.5
            confidence = 0.3
        
        # 핵심 원칙
        key_principles = []
        for virtue, score in virtues.items():
            if score > 0.6:
                key_principles.append(virtue)
        
        if not key_principles:
            key_principles = ["성실성", "품성 중심 판단"]
        
        # 잠재적 갈등
        conflicts = []
        high_virtues = [v for v, s in virtues.items() if s > 0.7]
        if len(high_virtues) > 3:
            conflicts.append("다수 덕목 간 우선순위 결정 필요")
        
        if virtues['justice'] > 0.7 and virtues['compassion'] > 0.7:
            conflicts.append("정의와 연민 사이의 균형 필요")
        
        return EthicsReasoning(
            school=self.get_school(),
            reasoning_process=reasoning_process,
            ethical_score=ethical_score,
            confidence=confidence,
            key_principles=key_principles,
            potential_conflicts=conflicts
        )

class DeontologicalEngine(EthicsReasoningEngine):
    """의무론적 윤리학 추론 엔진"""
    
    def get_school(self) -> EthicsSchool:
        return EthicsSchool.DEONTOLOGICAL
    
    def reason(self, dilemma: EthicalDilemma) -> EthicsReasoning:
        """의무론적 추론"""
        
        reasoning_process = ["의무론적 관점에서 분석 시작"]
        
        # 핵심 의무들
        duties = {
            'no_harm': 0.0,           # 해를 끼치지 말라
            'truth_telling': 0.0,     # 진실을 말하라
            'promise_keeping': 0.0,   # 약속을 지켜라
            'respect_autonomy': 0.0,  # 자율성을 존중하라
            'fairness': 0.0,          # 공정하게 대하라
            'respect_dignity': 0.0    # 인간 존엄성을 존중하라
        }
        
        scenario_lower = dilemma.scenario.lower()
        
        # 무해 원칙
        if any(word in scenario_lower for word in ['해롭', '피해', '손상', '위험']):
            duties['no_harm'] = 0.9
            reasoning_process.append("무해 원칙(do no harm) 적용")
        
        # 진실 의무
        if any(word in scenario_lower for word in ['거짓', '속임', '진실', '정직']):
            duties['truth_telling'] = 0.8
            reasoning_process.append("진실 의무 확인")
        
        # 약속 준수
        if any(word in scenario_lower for word in ['약속', '계약', '합의', '서약']):
            duties['promise_keeping'] = 0.8
            reasoning_process.append("약속 준수 의무 확인")
        
        # 자율성 존중
        autonomous_stakeholders = len([s for s in dilemma.stakeholders if s.vulnerability < 0.3])
        if autonomous_stakeholders > 0:
            duties['respect_autonomy'] = 0.8
            reasoning_process.append(f"{autonomous_stakeholders}명의 자율적 개인에 대한 존중")
        
        # 공정성
        if any(word in scenario_lower for word in ['공정', '평등', '차별']):
            duties['fairness'] = 0.9
            reasoning_process.append("공정성 의무 적용")
        
        # 인간 존엄성 (항상 높은 점수)
        duties['respect_dignity'] = 0.9
        reasoning_process.append("인간 존엄성 존중은 절대적 의무")
        
        # 칸트의 정언명령 적용
        reasoning_process.append("칸트의 정언명령 원칙 적용")
        
        # 보편화 가능성 테스트
        if dilemma.complexity_level < 0.5:
            universalizability = 0.8
            reasoning_process.append("행동 원칙의 보편화 가능성 높음")
        else:
            universalizability = 0.6
            reasoning_process.append("복잡한 상황으로 보편화 가능성 제한적")
        
        # 전체 의무 준수 점수
        duty_scores = [score for score in duties.values() if score > 0]
        if duty_scores:
            ethical_score = np.mean(duty_scores) * universalizability
            confidence = 0.9  # 의무론은 확실한 규칙 기반
        else:
            ethical_score = 0.7  # 기본적으로 의무 준수 지향
            confidence = 0.7
        
        # 핵심 원칙
        key_principles = ["정언명령", "의무 기반 판단", "보편화 가능성"]
        active_duties = [duty for duty, score in duties.items() if score > 0.6]
        key_principles.extend(active_duties)
        
        # 잠재적 갈등
        conflicts = []
        if len(active_duties) > 3:
            conflicts.append("다수 의무 간 우선순위 충돌")
        
        if duties['no_harm'] > 0.7 and duties['truth_telling'] > 0.7:
            conflicts.append("진실 말하기와 해악 방지 의무 간 갈등 가능")
        
        return EthicsReasoning(
            school=self.get_school(),
            reasoning_process=reasoning_process,
            ethical_score=ethical_score,
            confidence=confidence,
            key_principles=key_principles,
            potential_conflicts=conflicts
        )

class CareEthicsEngine(EthicsReasoningEngine):
    """돌봄 윤리학 추론 엔진"""
    
    def get_school(self) -> EthicsSchool:
        return EthicsSchool.CARE_ETHICS
    
    def reason(self, dilemma: EthicalDilemma) -> EthicsReasoning:
        """돌봄 윤리학적 추론"""
        
        reasoning_process = ["돌봄 윤리학적 관점에서 분석 시작"]
        
        # 돌봄 중심 가치들
        care_values = {
            'responsiveness': 0.0,    # 반응성
            'responsibility': 0.0,    # 책임감
            'competence': 0.0,        # 능력
            'attentiveness': 0.0,     # 주의깊음
            'trust': 0.0              # 신뢰
        }
        
        # 관계 중심 분석
        relationships = []
        vulnerable_stakeholders = [s for s in dilemma.stakeholders if s.vulnerability > 0.5]
        powerful_stakeholders = [s for s in dilemma.stakeholders if s.power_level > 0.7]
        
        # 반응성 평가
        if vulnerable_stakeholders:
            care_values['responsiveness'] = min(0.9, 0.5 + len(vulnerable_stakeholders) * 0.2)
            reasoning_process.append(f"취약 계층 {len(vulnerable_stakeholders)}명에 대한 반응성 중요")
            relationships.append("취약자-보호자 관계")
        
        # 책임감 평가
        if powerful_stakeholders:
            care_values['responsibility'] = min(0.9, 0.6 + len(powerful_stakeholders) * 0.1)
            reasoning_process.append(f"권력자 {len(powerful_stakeholders)}명의 책임감 중요")
            relationships.append("권력자-약자 관계")
        
        # 능력 평가
        if dilemma.resource_constraints:
            available_resources = np.mean(list(dilemma.resource_constraints.values()))
            care_values['competence'] = available_resources
            reasoning_process.append(f"돌봄 능력 평가: 자원 가용성 {available_resources:.2f}")
        else:
            care_values['competence'] = 0.7
        
        # 주의깊음 평가
        care_values['attentiveness'] = 0.8  # 돌봄 윤리는 항상 세심함 요구
        reasoning_process.append("상황에 대한 세심한 주의 필요")
        
        # 신뢰 평가
        scenario_lower = dilemma.scenario.lower()
        if any(word in scenario_lower for word in ['신뢰', '믿음', '의존']):
            care_values['trust'] = 0.8
            reasoning_process.append("신뢰 관계 중요성 확인")
        else:
            care_values['trust'] = 0.6
        
        # 돌봄 관계 네트워크 분석
        reasoning_process.append("돌봄 관계 네트워크 분석")
        if len(relationships) > 0:
            reasoning_process.append(f"식별된 관계: {', '.join(relationships)}")
        
        # 맥락적 접근
        reasoning_process.append("추상적 원칙보다 구체적 맥락 중시")
        
        # 전체 돌봄 점수
        care_scores = list(care_values.values())
        ethical_score = np.mean(care_scores)
        
        # 취약자 보호 가중치
        if vulnerable_stakeholders:
            vulnerability_bonus = len(vulnerable_stakeholders) * 0.05
            ethical_score = min(1.0, ethical_score + vulnerability_bonus)
            reasoning_process.append(f"취약자 보호 가중치 적용: +{vulnerability_bonus:.2f}")
        
        confidence = 0.8 if vulnerable_stakeholders else 0.6
        
        # 핵심 원칙
        key_principles = ["관계 중심 윤리", "취약자 보호", "돌봄과 배려"]
        active_values = [value for value, score in care_values.items() if score > 0.6]
        key_principles.extend(active_values)
        
        # 잠재적 갈등
        conflicts = []
        if len(powerful_stakeholders) > 0 and len(vulnerable_stakeholders) > 0:
            conflicts.append("권력 불균형 상황에서 돌봄 관계 복잡성")
        
        if dilemma.resource_constraints and np.mean(list(dilemma.resource_constraints.values())) < 0.5:
            conflicts.append("제한된 자원으로 인한 돌봄 능력 제약")
        
        return EthicsReasoning(
            school=self.get_school(),
            reasoning_process=reasoning_process,
            ethical_score=ethical_score,
            confidence=confidence,
            key_principles=key_principles,
            potential_conflicts=conflicts
        )

class DeepMultiDimensionalEthicsSystem:
    """심층 다차원 윤리 추론 시스템"""
    
    def __init__(self):
        self.logger = logger
        
        # 윤리 추론 엔진들
        self.reasoning_engines = {
            EthicsSchool.UTILITARIANISM: UtilitarianEngine(),
            EthicsSchool.VIRTUE_ETHICS: VirtueEthicsEngine(),
            EthicsSchool.DEONTOLOGICAL: DeontologicalEngine(),
            EthicsSchool.CARE_ETHICS: CareEthicsEngine()
        }
        
        # 학파별 가중치 (문화적/상황적으로 조정 가능)
        self.school_weights = {
            EthicsSchool.UTILITARIANISM: 0.3,
            EthicsSchool.VIRTUE_ETHICS: 0.25,
            EthicsSchool.DEONTOLOGICAL: 0.25,
            EthicsSchool.CARE_ETHICS: 0.2
        }
        
        # 추론 히스토리
        self.reasoning_history = []
        
        # 문화적 적응 메모리
        self.cultural_adaptations = {}
        
        # Mixture of Experts for 윤리 분석
        self.moe_enabled = True
        if self.moe_enabled:
            try:
                # 윤리 분석용 MoE 초기화
                ethics_input_dim = 512  # 윤리적 맥락 임베딩 차원
                ethics_output_dim = len(EthicsSchool)  # 윤리학파 수
                
                self.ethics_moe = create_ethics_moe(
                    input_dim=ethics_input_dim,
                    output_dim=ethics_output_dim,
                    num_experts=4
                )
                
                # GPU 사용 가능시 이동
                if torch.cuda.is_available() and ADVANCED_CONFIG.get('enable_gpu', False):
                    self.ethics_moe = self.ethics_moe.cuda()
                
                self.logger.info("윤리 분석용 MoE 시스템 초기화 완료 (4개 전문가)")
            except Exception as e:
                self.logger.warning(f"윤리 MoE 초기화 실패, 기본 시스템 사용: {e}")
                self.moe_enabled = False
        
        self.logger.info("심층 다차원 윤리 추론 시스템 초기화 완료")
    
    def comprehensive_ethical_analysis(self, dilemma: EthicalDilemma) -> IntegratedEthicsResult:
        """종합적 윤리 분석"""
        
        start_time = time.time()
        
        # 1단계: 각 윤리학파별 추론
        school_reasonings = {}
        for school, engine in self.reasoning_engines.items():
            try:
                reasoning = engine.reason(dilemma)
                school_reasonings[school] = reasoning
                self.logger.debug(f"{school.value} 추론 완료: 점수 {reasoning.ethical_score:.3f}")
            except Exception as e:
                self.logger.error(f"{school.value} 추론 실패: {e}")
                continue
        
        # 2단계: MoE 기반 윤리학파 가중치 조정
        if self.moe_enabled:
            school_reasonings = self._apply_moe_ethics_analysis(dilemma, school_reasonings)
        
        # 3단계: 이해관계자 관점 분석
        stakeholder_analysis = self._analyze_stakeholder_perspectives(dilemma, school_reasonings)
        
        # 3단계: 문화적 맥락 고려
        cultural_considerations = self._consider_cultural_context(dilemma, school_reasonings)
        
        # 4단계: 시간적 분석 (단기/장기 영향)
        temporal_analysis = self._analyze_temporal_implications(dilemma, school_reasonings)
        
        # 5단계: 학파 간 합의 및 갈등 분석
        ethical_consensus = self._calculate_ethical_consensus(school_reasonings)
        
        # 6단계: 통합 추천 생성
        overall_recommendation = self._generate_integrated_recommendation(
            dilemma, school_reasonings, stakeholder_analysis, cultural_considerations
        )
        
        # 7단계: 신뢰도 계산
        confidence_score = self._calculate_overall_confidence(school_reasonings, ethical_consensus)
        
        # 결과 생성
        result = IntegratedEthicsResult(
            dilemma=dilemma,
            school_reasonings=school_reasonings,
            overall_recommendation=overall_recommendation,
            confidence_score=confidence_score,
            ethical_consensus=ethical_consensus,
            stakeholder_analysis=stakeholder_analysis,
            cultural_considerations=cultural_considerations,
            temporal_analysis=temporal_analysis,
            processing_time=time.time() - start_time,
            reasoning_depth=len(school_reasonings)
        )
        
        # 히스토리 저장
        self.reasoning_history.append(result)
        if len(self.reasoning_history) > 100:
            self.reasoning_history = self.reasoning_history[-100:]
        
        self.logger.info(
            f"종합 윤리 분석 완료: {len(school_reasonings)}개 학파, "
            f"합의도 {ethical_consensus:.3f}, 신뢰도 {confidence_score:.3f}"
        )
        
        return result
    
    def _apply_moe_ethics_analysis(self, dilemma: EthicalDilemma, 
                                 school_reasonings: Dict[EthicsSchool, Any]) -> Dict[EthicsSchool, Any]:
        """
        MoE 기반 윤리학파 분석 및 가중치 조정
        
        Args:
            dilemma: 윤리적 딜레마
            school_reasonings: 기본 윤리학파별 추론 결과
            
        Returns:
            MoE로 보정된 윤리학파별 추론 결과
        """
        try:
            # 윤리적 맥락 임베딩 생성
            context_embedding = self._create_ethics_context_embedding(dilemma)
            
            if context_embedding is None:
                return school_reasonings
            
            # MoE 추론
            moe_result = self.ethics_moe(context_embedding, temperature=0.7, return_expert_outputs=True)
            
            # MoE 결과를 윤리학파 우선순위로 변환
            ethics_probs = torch.softmax(moe_result.final_output, dim=-1).squeeze(0)
            
            # 윤리학파 매핑
            ethics_schools = list(EthicsSchool)
            
            # MoE 기반 학파별 가중치 조정
            enhanced_reasonings = {}
            
            for i, school in enumerate(ethics_schools):
                if school in school_reasonings and i < len(ethics_probs):
                    original_reasoning = school_reasonings[school]
                    moe_weight = ethics_probs[i].item()
                    
                    # 원본 점수와 MoE 가중치 결합
                    original_score = getattr(original_reasoning, 'ethical_score', 0.5)
                    enhanced_score = original_score * (0.7 + 0.3 * moe_weight)
                    
                    # 새로운 추론 결과 생성 (원본 복사 후 수정)
                    enhanced_reasoning = original_reasoning
                    enhanced_reasoning.ethical_score = enhanced_score
                    enhanced_reasoning.confidence *= (0.8 + 0.2 * moe_weight)
                    
                    # MoE 메타데이터 추가
                    if hasattr(enhanced_reasoning, 'metadata'):
                        enhanced_reasoning.metadata.update({
                            'moe_weight': moe_weight,
                            'moe_enhanced': True,
                            'original_score': original_score
                        })
                    
                    enhanced_reasonings[school] = enhanced_reasoning
                    
                    self.logger.debug(f"{school.value} MoE 가중치: {moe_weight:.3f}, "
                                    f"조정된 점수: {enhanced_score:.3f}")
                
                elif school in school_reasonings:
                    # MoE 결과가 없는 경우 원본 유지
                    enhanced_reasonings[school] = school_reasonings[school]
            
            # MoE 다양성 정보 로깅
            self.logger.info(f"윤리 MoE 분석 완료 - 전문가 {moe_result.total_experts_used}개 사용, "
                           f"다양성 점수: {moe_result.diversity_score:.3f}")
            
            return enhanced_reasonings
            
        except Exception as e:
            self.logger.error(f"윤리 MoE 분석 실패: {e}")
            return school_reasonings
    
    def _create_ethics_context_embedding(self, dilemma: EthicalDilemma) -> Optional[torch.Tensor]:
        """
        윤리적 맥락 임베딩 생성
        
        Args:
            dilemma: 윤리적 딜레마
            
        Returns:
            윤리적 맥락 임베딩 텐서
        """
        try:
            # 딜레마의 핵심 요소들을 벡터로 변환
            context_features = []
            
            # 1. 딜레마 복잡도
            complexity = getattr(dilemma, 'complexity_level', 0.5)
            context_features.extend([complexity])
            
            # 2. 이해관계자 수
            stakeholder_count = len(getattr(dilemma, 'stakeholders', []))
            normalized_stakeholder_count = min(stakeholder_count / 10.0, 1.0)
            context_features.extend([normalized_stakeholder_count])
            
            # 3. 시간적 긴급성
            urgency = getattr(dilemma, 'urgency_level', 0.5)
            context_features.extend([urgency])
            
            # 4. 문화적 민감성
            cultural_sensitivity = getattr(dilemma, 'cultural_sensitivity', 0.5)
            context_features.extend([cultural_sensitivity])
            
            # 5. 결과의 가역성
            reversibility = getattr(dilemma, 'consequence_reversibility', 0.5)
            context_features.extend([reversibility])
            
            # 6. 개인 vs 집단 영향
            personal_vs_collective = getattr(dilemma, 'personal_vs_collective_impact', 0.5)
            context_features.extend([personal_vs_collective])
            
            # 벡터를 원하는 차원으로 확장 (512차원)
            while len(context_features) < 512:
                # 기존 특성들을 변형하여 확장
                base_idx = len(context_features) % 6
                if base_idx < len(context_features):
                    # 기존 값에 작은 변형 추가
                    variation = context_features[base_idx] * (0.9 + 0.2 * np.random.random())
                    context_features.append(variation)
                else:
                    context_features.append(0.5)  # 기본값
            
            # 텐서로 변환
            embedding = torch.tensor(context_features[:512], dtype=torch.float32)
            
            # 정규화
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)
            
            # 배치 차원 추가
            embedding = embedding.unsqueeze(0)
            
            # GPU로 이동 (필요시)
            if torch.cuda.is_available() and ADVANCED_CONFIG.get('enable_gpu', False):
                embedding = embedding.cuda()
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"윤리 맥락 임베딩 생성 실패: {e}")
            return None
    
    def _analyze_stakeholder_perspectives(
        self,
        dilemma: EthicalDilemma,
        school_reasonings: Dict[EthicsSchool, EthicsReasoning]
    ) -> Dict[str, Dict[str, Any]]:
        """이해관계자 관점 분석"""
        
        stakeholder_analysis = {}
        
        for stakeholder in dilemma.stakeholders:
            analysis = {
                'perspective': stakeholder.name,
                'vulnerability_level': stakeholder.vulnerability,
                'power_level': stakeholder.power_level,
                'expected_impact': stakeholder.expected_benefits - stakeholder.expected_harms,
                'voice_weight': stakeholder.voice_weight,
                'ethical_priorities': {}
            }
            
            # 취약성에 따른 우선순위
            if stakeholder.vulnerability > 0.7:
                analysis['ethical_priorities']['care_ethics'] = 0.9
                analysis['ethical_priorities']['protection_needed'] = True
            
            # 권력 수준에 따른 책임
            if stakeholder.power_level > 0.7:
                analysis['ethical_priorities']['responsibility'] = 0.8
                analysis['ethical_priorities']['leadership_expected'] = True
            
            # 윤리학파별 관심도
            for school, reasoning in school_reasonings.items():
                if school == EthicsSchool.CARE_ETHICS and stakeholder.vulnerability > 0.5:
                    analysis['ethical_priorities'][school.value] = reasoning.ethical_score
                elif school == EthicsSchool.UTILITARIANISM:
                    # 모든 이해관계자가 공리주의적 계산에 포함
                    analysis['ethical_priorities'][school.value] = reasoning.ethical_score
                elif school == EthicsSchool.DEONTOLOGICAL and stakeholder.power_level > 0.6:
                    # 권력 있는 이해관계자는 의무 준수 중요
                    analysis['ethical_priorities'][school.value] = reasoning.ethical_score
            
            stakeholder_analysis[stakeholder.stakeholder_id] = analysis
        
        return stakeholder_analysis
    
    def _consider_cultural_context(
        self,
        dilemma: EthicalDilemma,
        school_reasonings: Dict[EthicsSchool, EthicsReasoning]
    ) -> List[str]:
        """문화적 맥락 고려"""
        
        considerations = []
        
        if dilemma.cultural_context:
            culture = dilemma.cultural_context
            
            # 개인주의 vs 집단주의
            if culture.individualism_collectivism < 0.3:
                considerations.append("집단주의 문화에서 공동체 이익 우선시")
                # 공리주의와 돌봄 윤리 가중치 증가
                self.school_weights[EthicsSchool.UTILITARIANISM] *= 1.2
                self.school_weights[EthicsSchool.CARE_ETHICS] *= 1.3
                
            elif culture.individualism_collectivism > 0.7:
                considerations.append("개인주의 문화에서 개인 권리와 자유 중시")
                # 덕 윤리와 의무론 가중치 증가
                self.school_weights[EthicsSchool.VIRTUE_ETHICS] *= 1.2
                self.school_weights[EthicsSchool.DEONTOLOGICAL] *= 1.1
            
            # 권력 거리
            if culture.power_distance > 0.7:
                considerations.append("높은 권력 거리 문화에서 위계 질서 중시")
            elif culture.power_distance < 0.3:
                considerations.append("낮은 권력 거리 문화에서 평등 중시")
            
            # 불확실성 회피
            if culture.uncertainty_avoidance > 0.7:
                considerations.append("불확실성 회피 문화에서 명확한 규칙 선호")
                # 의무론적 윤리 가중치 증가
                self.school_weights[EthicsSchool.DEONTOLOGICAL] *= 1.2
            
            # 장기 지향성
            if culture.long_term_orientation > 0.7:
                considerations.append("장기 지향 문화에서 미래 결과 중시")
            
            # 문화적 가치 반영
            for value, importance in culture.cultural_values.items():
                if importance > 0.7:
                    considerations.append(f"문화적 가치 '{value}' 높은 중요성")
        
        else:
            considerations.append("문화적 맥락 정보 없음 - 일반적 윤리 원칙 적용")
        
        return considerations
    
    def _analyze_temporal_implications(
        self,
        dilemma: EthicalDilemma,
        school_reasonings: Dict[EthicsSchool, EthicsReasoning]
    ) -> Dict[str, float]:
        """시간적 영향 분석"""
        
        temporal_analysis = {
            'immediate_impact': 0.0,    # 즉시 영향
            'short_term_impact': 0.0,   # 단기 영향 (1개월-1년)
            'long_term_impact': 0.0,    # 장기 영향 (1년 이상)
            'reversibility': dilemma.reversibility,
            'urgency_factor': dilemma.urgency_level
        }
        
        # 각 윤리학파의 시간적 가중치
        temporal_weights = {
            EthicsSchool.UTILITARIANISM: {'immediate': 0.3, 'short': 0.4, 'long': 0.3},
            EthicsSchool.VIRTUE_ETHICS: {'immediate': 0.2, 'short': 0.3, 'long': 0.5},
            EthicsSchool.DEONTOLOGICAL: {'immediate': 0.5, 'short': 0.3, 'long': 0.2},
            EthicsSchool.CARE_ETHICS: {'immediate': 0.4, 'short': 0.4, 'long': 0.2}
        }
        
        # 각 시간대별 영향 계산
        for school, reasoning in school_reasonings.items():
            if school in temporal_weights:
                weights = temporal_weights[school]
                score = reasoning.ethical_score
                
                temporal_analysis['immediate_impact'] += score * weights['immediate']
                temporal_analysis['short_term_impact'] += score * weights['short']
                temporal_analysis['long_term_impact'] += score * weights['long']
        
        # 정규화
        num_schools = len(school_reasonings)
        if num_schools > 0:
            temporal_analysis['immediate_impact'] /= num_schools
            temporal_analysis['short_term_impact'] /= num_schools
            temporal_analysis['long_term_impact'] /= num_schools
        
        return temporal_analysis
    
    def _calculate_ethical_consensus(self, school_reasonings: Dict[EthicsSchool, EthicsReasoning]) -> float:
        """윤리학파 간 합의도 계산"""
        
        if len(school_reasonings) < 2:
            return 1.0  # 학파가 하나뿐이면 합의도 최대
        
        scores = [reasoning.ethical_score for reasoning in school_reasonings.values()]
        
        # 점수 분산으로 합의도 측정 (분산이 낮을수록 합의도 높음)
        score_variance = np.var(scores)
        consensus = 1.0 - min(score_variance * 4, 1.0)  # 분산 4배 후 역산
        
        return np.clip(consensus, 0.0, 1.0)
    
    def _generate_integrated_recommendation(
        self,
        dilemma: EthicalDilemma,
        school_reasonings: Dict[EthicsSchool, EthicsReasoning],
        stakeholder_analysis: Dict[str, Dict[str, Any]],
        cultural_considerations: List[str]
    ) -> str:
        """통합 추천 생성"""
        
        recommendations = []
        
        # 1. 학파별 핵심 메시지
        high_scoring_schools = [(school, reasoning) for school, reasoning in school_reasonings.items() 
                               if reasoning.ethical_score > 0.6]
        
        if high_scoring_schools:
            recommendations.append("윤리적 관점별 분석 결과:")
            for school, reasoning in high_scoring_schools:
                recommendations.append(f"- {school.value}: {reasoning.reasoning_process[-1]}")
        
        # 2. 이해관계자 고려사항
        vulnerable_stakeholders = [sid for sid, analysis in stakeholder_analysis.items() 
                                 if analysis['vulnerability_level'] > 0.6]
        
        if vulnerable_stakeholders:
            recommendations.append(f"특별 고려 대상: {len(vulnerable_stakeholders)}명의 취약 계층")
        
        # 3. 문화적 권고사항
        if cultural_considerations:
            recommendations.append("문화적 맥락 고려사항:")
            for consideration in cultural_considerations[:2]:  # 상위 2개만
                recommendations.append(f"- {consideration}")
        
        # 4. 최종 권고
        overall_scores = [reasoning.ethical_score for reasoning in school_reasonings.values()]
        if overall_scores:
            avg_score = np.mean(overall_scores)
            if avg_score > 0.7:
                final_rec = "윤리적으로 바람직한 결정으로 판단됩니다."
            elif avg_score > 0.5:
                final_rec = "신중한 고려 하에 진행 가능한 결정입니다."
            else:
                final_rec = "윤리적 우려가 있어 재검토가 필요합니다."
        else:
            final_rec = "추가 정보 수집 후 재평가가 필요합니다."
        
        recommendations.append(f"\n종합 권고: {final_rec}")
        
        return "\n".join(recommendations)
    
    def _calculate_overall_confidence(
        self,
        school_reasonings: Dict[EthicsSchool, EthicsReasoning],
        consensus: float
    ) -> float:
        """전체 신뢰도 계산"""
        
        if not school_reasonings:
            return 0.0
        
        # 각 학파의 신뢰도 평균
        avg_confidence = np.mean([reasoning.confidence for reasoning in school_reasonings.values()])
        
        # 합의도를 신뢰도에 반영
        overall_confidence = avg_confidence * (0.7 + 0.3 * consensus)
        
        return np.clip(overall_confidence, 0.0, 1.0)
    
    def get_ethics_analytics(self) -> Dict[str, Any]:
        """윤리 분석 정보 반환"""
        
        if not self.reasoning_history:
            return {"message": "분석 히스토리가 없습니다."}
        
        recent_analyses = self.reasoning_history[-10:]
        
        analytics = {
            'total_analyses': len(self.reasoning_history),
            'average_confidence': np.mean([analysis.confidence_score for analysis in recent_analyses]),
            'average_consensus': np.mean([analysis.ethical_consensus for analysis in recent_analyses]),
            'average_processing_time': np.mean([analysis.processing_time for analysis in recent_analyses]),
            'school_performance': {},
            'common_conflicts': []
        }
        
        # 학파별 성능
        for school in EthicsSchool:
            school_scores = []
            for analysis in recent_analyses:
                if school in analysis.school_reasonings:
                    school_scores.append(analysis.school_reasonings[school].ethical_score)
            
            if school_scores:
                analytics['school_performance'][school.value] = {
                    'average_score': np.mean(school_scores),
                    'consistency': 1.0 - np.std(school_scores),
                    'usage_rate': len(school_scores) / len(recent_analyses)
                }
        
        # 공통 갈등 요소
        all_conflicts = []
        for analysis in recent_analyses:
            for reasoning in analysis.school_reasonings.values():
                all_conflicts.extend(reasoning.potential_conflicts)
        
        if all_conflicts:
            from collections import Counter
            conflict_counts = Counter(all_conflicts)
            analytics['common_conflicts'] = [conflict for conflict, count in conflict_counts.most_common(3)]
        
        return analytics


# 테스트 및 데모 함수
def test_deep_multi_dimensional_ethics():
    """심층 다차원 윤리 시스템 테스트"""
    print("🧠 심층 다차원 윤리 추론 시스템 테스트 시작")
    
    # 시스템 초기화
    ethics_system = DeepMultiDimensionalEthicsSystem()
    
    # 테스트 이해관계자들
    stakeholders = [
        StakeholderPerspective(
            stakeholder_id="employees",
            name="직원들",
            role="근로자",
            power_level=0.3,
            vulnerability=0.7,
            expected_benefits=0.2,
            expected_harms=0.8,
            voice_weight=0.8
        ),
        StakeholderPerspective(
            stakeholder_id="shareholders",
            name="주주들",
            role="투자자",
            power_level=0.9,
            vulnerability=0.2,
            expected_benefits=0.8,
            expected_harms=0.1,
            voice_weight=0.6
        ),
        StakeholderPerspective(
            stakeholder_id="customers",
            name="고객들",
            role="소비자",
            power_level=0.5,
            vulnerability=0.4,
            expected_benefits=0.6,
            expected_harms=0.3,
            voice_weight=0.7
        )
    ]
    
    # 문화적 맥락
    cultural_context = CulturalContext(
        culture_id="korean",
        cultural_values={"hierarchy_respect": 0.8, "group_harmony": 0.9},
        individualism_collectivism=0.3,  # 집단주의 성향
        power_distance=0.7,
        long_term_orientation=0.8
    )
    
    # 테스트 딜레마
    test_dilemma = EthicalDilemma(
        dilemma_id="corporate_layoff",
        scenario="경제적 어려움으로 인해 회사가 직원 30%를 해고해야 하는 상황입니다. "
                "이는 회사의 생존을 위해 필요하지만 많은 가정에 경제적 타격을 줄 것입니다.",
        context="글로벌 경제 침체로 인한 구조조정",
        complexity_level=0.8,
        urgency_level=0.7,
        reversibility=0.3,
        stakeholders=stakeholders,
        cultural_context=cultural_context,
        available_options=[
            "전체 직원 30% 해고",
            "임금 삭감으로 해고 최소화",
            "단계적 구조조정",
            "사업부 매각"
        ],
        resource_constraints={"financial": 0.3, "time": 0.4, "human": 0.6}
    )
    
    print(f"테스트 딜레마: {test_dilemma.scenario}")
    print(f"복잡도: {test_dilemma.complexity_level}, 긴급도: {test_dilemma.urgency_level}")
    print(f"이해관계자: {len(stakeholders)}명")
    
    # 종합 윤리 분석 실행
    result = ethics_system.comprehensive_ethical_analysis(test_dilemma)
    
    # 결과 출력
    print(f"\n📊 종합 분석 결과:")
    print(f"- 처리 시간: {result.processing_time:.3f}초")
    print(f"- 추론 깊이: {result.reasoning_depth}개 학파")
    print(f"- 윤리적 합의도: {result.ethical_consensus:.3f}")
    print(f"- 전체 신뢰도: {result.confidence_score:.3f}")
    
    print(f"\n🏫 학파별 추론 결과:")
    for school, reasoning in result.school_reasonings.items():
        print(f"\n--- {school.value} ---")
        print(f"윤리 점수: {reasoning.ethical_score:.3f}")
        print(f"신뢰도: {reasoning.confidence:.3f}")
        print(f"핵심 원칙: {', '.join(reasoning.key_principles[:3])}")
        print(f"추론 과정:")
        for i, process in enumerate(reasoning.reasoning_process[-3:], 1):
            print(f"  {i}. {process}")
        
        if reasoning.potential_conflicts:
            print(f"잠재적 갈등: {', '.join(reasoning.potential_conflicts)}")
    
    print(f"\n👥 이해관계자 분석:")
    for stakeholder_id, analysis in result.stakeholder_analysis.items():
        print(f"- {analysis['perspective']}: "
              f"취약성 {analysis['vulnerability_level']:.2f}, "
              f"권력 {analysis['power_level']:.2f}, "
              f"예상 영향 {analysis['expected_impact']:.2f}")
    
    print(f"\n🌍 문화적 고려사항:")
    for consideration in result.cultural_considerations:
        print(f"- {consideration}")
    
    print(f"\n⏰ 시간적 분석:")
    for time_aspect, value in result.temporal_analysis.items():
        print(f"- {time_aspect}: {value:.3f}")
    
    print(f"\n💡 종합 권고사항:")
    print(result.overall_recommendation)
    
    # 분석 정보
    analytics = ethics_system.get_ethics_analytics()
    print(f"\n📈 시스템 분석:")
    print(f"- 총 분석 수: {analytics['total_analyses']}")
    print(f"- 평균 신뢰도: {analytics['average_confidence']:.3f}")
    print(f"- 평균 합의도: {analytics['average_consensus']:.3f}")
    
    if analytics['school_performance']:
        print(f"- 학파별 성능:")
        for school, perf in analytics['school_performance'].items():
            print(f"  {school}: 평균 점수 {perf['average_score']:.3f}, "
                  f"일관성 {perf['consistency']:.3f}")
    
    print("✅ 심층 다차원 윤리 추론 시스템 테스트 완료")
    
    return ethics_system, result


if __name__ == "__main__":
    test_deep_multi_dimensional_ethics()
