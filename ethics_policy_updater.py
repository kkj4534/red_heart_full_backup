"""
윤리 정책 자동 조정기 (Ethics Policy Auto-Updater)
Ethics Policy Auto-Updater Module

경험 데이터베이스를 기반으로 윤리 가중치를 지속적으로 학습하고 조정하여
개인화된 윤리 정책과 문화적 맥락을 반영한 적응적 윤리 판단 시스템을 구현합니다.

핵심 기능:
1. 경험 기반 윤리 가중치 자동 조정
2. 개인-공동체 균형 계수 도입
3. 문화적 맥락 고려 다차원 윤리 계산
4. 사용자 피드백 기반 정책 개선
"""

import numpy as np
import torch
import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, deque
import sqlite3

try:
    from config import ADVANCED_CONFIG, DEVICE, DATA_DIR
    import os
    # pathlib 대신 os.path 사용 (WSL 호환성)
    EXPERIENCE_DB_DIR = os.path.join(DATA_DIR, "experience_db")
except ImportError:
    # config.py에 문제가 있을 경우 기본값 사용
    ADVANCED_CONFIG = {'enable_gpu': False}
    DEVICE = 'cpu'
    from pathlib import Path
    EXPERIENCE_DB_DIR = Path("data/experience_db")
    print("⚠️  config.py 임포트 실패, 기본값 사용")
from data_models import EmotionData

logger = logging.getLogger('EthicsPolicyUpdater')

@dataclass
class EthicsPolicy:
    """윤리 정책 데이터 클래스"""
    policy_id: str
    user_id: str = "default"
    
    # 기본 윤리 가중치 (도덕 기반 이론)
    care_harm: float = 0.8
    fairness_cheating: float = 0.7
    loyalty_betrayal: float = 0.6
    authority_subversion: float = 0.5
    sanctity_degradation: float = 0.6
    
    # 개인-공동체 균형 계수
    individual_weight: float = 0.4
    community_weight: float = 0.6
    
    # 문화적 맥락 가중치
    cultural_context: Dict[str, float] = field(default_factory=lambda: {
        'hierarchy_respect': 0.7,  # 위계 존중
        'group_harmony': 0.8,     # 집단 조화
        'face_saving': 0.6,       # 체면 중시
        'long_term_thinking': 0.9 # 장기적 사고
    })
    
    # 정책 메타데이터
    creation_time: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    update_count: int = 0
    confidence_score: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'policy_id': self.policy_id,
            'user_id': self.user_id,
            'care_harm': self.care_harm,
            'fairness_cheating': self.fairness_cheating,
            'loyalty_betrayal': self.loyalty_betrayal,
            'authority_subversion': self.authority_subversion,
            'sanctity_degradation': self.sanctity_degradation,
            'individual_weight': self.individual_weight,
            'community_weight': self.community_weight,
            'cultural_context': self.cultural_context,
            'creation_time': self.creation_time,
            'last_updated': self.last_updated,
            'update_count': self.update_count,
            'confidence_score': self.confidence_score
        }

@dataclass
class EthicsExperience:
    """윤리적 경험 데이터"""
    experience_id: str
    scenario: str
    decision_made: str
    outcome_rating: float  # -1.0 ~ 1.0 (매우 나쁨 ~ 매우 좋음)
    
    # 맥락 정보
    emotion_state: EmotionData
    stakeholders: List[str]
    cultural_context: str
    decision_urgency: float
    
    # 결과 정보
    actual_regret: float
    user_satisfaction: float
    moral_correctness: float  # 외부 평가자 또는 전문가 평가
    
    # 메타데이터
    timestamp: float = field(default_factory=time.time)
    feedback_quality: float = 0.8  # 피드백 품질 점수
    decision: Optional[str] = None  # 호환성을 위한 별칭

@dataclass
class PolicyUpdateResult:
    """정책 업데이트 결과"""
    old_policy: EthicsPolicy
    new_policy: EthicsPolicy
    update_magnitude: float
    convergence_achieved: bool
    improvement_areas: List[str]
    confidence_change: float
    reasoning_trace: List[str] = field(default_factory=list)

class EthicsPolicyUpdater:
    """윤리 정책 자동 조정기"""
    
    def __init__(self, db_path: Optional[str] = None):
        self.logger = logger
        
        # 데이터베이스 설정
        self.db_path = db_path or str(EXPERIENCE_DB_DIR / "ethics_policy.db")
        self._init_database()
        
        # 학습 설정
        self.learning_config = {
            'learning_rate': 0.01,
            'momentum': 0.9,
            'decay_rate': 0.99,
            'min_experiences_for_update': 5,
            'convergence_threshold': 0.001,
            'max_update_magnitude': 0.1,
            'experience_weight_decay': 0.95,  # 오래된 경험의 가중치 감소
            'cultural_adaptation_rate': 0.05
        }
        
        # 메모리 버퍼
        self.experience_buffer = deque(maxlen=1000)
        self.recent_updates = deque(maxlen=50)
        
        # 통계 추적
        self.update_statistics = {
            'total_updates': 0,
            'successful_updates': 0,
            'average_improvement': 0.0,
            'convergence_rate': 0.0
        }
        
        # 다차원 윤리 계산기
        self.multi_ethics_calculator = MultiEthicsCalculator()
        
        self.logger.info("윤리 정책 자동 조정기 초기화 완료")
    
    def _init_database(self):
        """데이터베이스 초기화"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # 정책 테이블
            conn.execute('''
                CREATE TABLE IF NOT EXISTS ethics_policies (
                    policy_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    policy_data TEXT,
                    creation_time REAL,
                    last_updated REAL,
                    update_count INTEGER,
                    confidence_score REAL
                )
            ''')
            
            # 경험 테이블
            conn.execute('''
                CREATE TABLE IF NOT EXISTS ethics_experiences (
                    experience_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    scenario TEXT,
                    decision_made TEXT,
                    outcome_rating REAL,
                    actual_regret REAL,
                    user_satisfaction REAL,
                    moral_correctness REAL,
                    experience_data TEXT,
                    timestamp REAL,
                    feedback_quality REAL
                )
            ''')
            
            # 업데이트 히스토리 테이블
            conn.execute('''
                CREATE TABLE IF NOT EXISTS policy_updates (
                    update_id TEXT PRIMARY KEY,
                    policy_id TEXT,
                    old_policy_data TEXT,
                    new_policy_data TEXT,
                    update_magnitude REAL,
                    improvement_score REAL,
                    timestamp REAL
                )
            ''')
            
            conn.commit()
    
    def add_experience(self, experience: EthicsExperience):
        """새로운 윤리적 경험 추가"""
        
        # 메모리 버퍼에 추가
        self.experience_buffer.append(experience)
        
        # 데이터베이스에 저장
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO ethics_experiences 
                (experience_id, user_id, scenario, decision_made, outcome_rating,
                 actual_regret, user_satisfaction, moral_correctness, 
                 experience_data, timestamp, feedback_quality)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                experience.experience_id,
                "default",  # 사용자 ID
                experience.scenario,
                experience.decision_made,
                experience.outcome_rating,
                experience.actual_regret,
                experience.user_satisfaction,
                experience.moral_correctness,
                json.dumps(experience.__dict__, default=str),
                experience.timestamp,
                experience.feedback_quality
            ))
            conn.commit()
        
        self.logger.info(f"새로운 윤리적 경험 추가: {experience.experience_id}")
    
    def get_policy(self, user_id: str = "default") -> EthicsPolicy:
        """사용자별 윤리 정책 조회"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT policy_data FROM ethics_policies 
                WHERE user_id = ? 
                ORDER BY last_updated DESC LIMIT 1
            ''', (user_id,))
            
            row = cursor.fetchone()
            
            if row:
                policy_data = json.loads(row[0])
                policy = EthicsPolicy(**policy_data)
                return policy
            else:
                # 기본 정책 생성
                default_policy = EthicsPolicy(
                    policy_id=f"default_{user_id}_{int(time.time())}",
                    user_id=user_id
                )
                self.save_policy(default_policy)
                return default_policy
    
    def save_policy(self, policy: EthicsPolicy):
        """정책 저장"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO ethics_policies 
                (policy_id, user_id, policy_data, creation_time, 
                 last_updated, update_count, confidence_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                policy.policy_id,
                policy.user_id,
                json.dumps(policy.to_dict()),
                policy.creation_time,
                policy.last_updated,
                policy.update_count,
                policy.confidence_score
            ))
            conn.commit()
    
    def update_policy_from_experiences(
        self, 
        user_id: str = "default",
        min_experiences: Optional[int] = None
    ) -> PolicyUpdateResult:
        """경험을 바탕으로 정책 업데이트"""
        
        # 기존 정책 로드
        old_policy = self.get_policy(user_id)
        
        # 최근 경험들 수집
        experiences = self._get_recent_experiences(
            user_id, 
            min_experiences or self.learning_config['min_experiences_for_update']
        )
        
        if len(experiences) < (min_experiences or self.learning_config['min_experiences_for_update']):
            self.logger.warning(f"경험 데이터 부족: {len(experiences)}개 (최소 {min_experiences}개 필요)")
            return self._create_no_update_result(old_policy, "insufficient_data")
        
        reasoning_trace = [f"정책 업데이트 시작: {len(experiences)}개 경험 분석"]
        
        # 1단계: 경험 분석 및 가중치 계산
        experience_weights = self._calculate_experience_weights(experiences)
        reasoning_trace.append(f"경험 가중치 계산 완료: 평균 품질 {np.mean(experience_weights):.3f}")
        
        # 2단계: 윤리 가중치 조정
        new_ethics_weights = self._update_ethics_weights(
            old_policy, experiences, experience_weights, reasoning_trace
        )
        
        # 3단계: 개인-공동체 균형 조정
        new_balance = self._update_individual_community_balance(
            old_policy, experiences, experience_weights, reasoning_trace
        )
        
        # 4단계: 문화적 맥락 조정
        new_cultural_context = self._update_cultural_context(
            old_policy, experiences, experience_weights, reasoning_trace
        )
        
        # 5단계: 새로운 정책 생성
        new_policy = EthicsPolicy(
            policy_id=f"{user_id}_{int(time.time())}",
            user_id=user_id,
            care_harm=new_ethics_weights['care_harm'],
            fairness_cheating=new_ethics_weights['fairness_cheating'],
            loyalty_betrayal=new_ethics_weights['loyalty_betrayal'],
            authority_subversion=new_ethics_weights['authority_subversion'],
            sanctity_degradation=new_ethics_weights['sanctity_degradation'],
            individual_weight=new_balance['individual'],
            community_weight=new_balance['community'],
            cultural_context=new_cultural_context,
            last_updated=time.time(),
            update_count=old_policy.update_count + 1
        )
        
        # 6단계: 업데이트 크기 및 수렴 체크
        update_magnitude = self._calculate_update_magnitude(old_policy, new_policy)
        convergence_achieved = update_magnitude < self.learning_config['convergence_threshold']
        
        # 7단계: 신뢰도 업데이트
        new_policy.confidence_score = self._update_confidence_score(
            old_policy, experiences, update_magnitude
        )
        
        # 8단계: 개선 영역 식별
        improvement_areas = self._identify_improvement_areas(
            old_policy, new_policy, experiences
        )
        
        reasoning_trace.append(
            f"정책 업데이트 완료: 크기 {update_magnitude:.4f}, "
            f"수렴 {'달성' if convergence_achieved else '미달성'}"
        )
        
        # 정책 저장
        self.save_policy(new_policy)
        
        # 업데이트 결과 생성
        result = PolicyUpdateResult(
            old_policy=old_policy,
            new_policy=new_policy,
            update_magnitude=update_magnitude,
            convergence_achieved=convergence_achieved,
            improvement_areas=improvement_areas,
            confidence_change=new_policy.confidence_score - old_policy.confidence_score,
            reasoning_trace=reasoning_trace
        )
        
        # 업데이트 히스토리 저장
        self._save_update_history(result)
        
        # 통계 업데이트
        self._update_statistics(result)
        
        return result
    
    def _get_recent_experiences(self, user_id: str, min_count: int) -> List[EthicsExperience]:
        """최근 경험 데이터 조회"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT experience_data FROM ethics_experiences 
                WHERE user_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (user_id, min_count * 2))  # 여유롭게 가져오기
            
            experiences = []
            for row in cursor.fetchall():
                try:
                    exp_data = json.loads(row[0])
                    # EmotionData 복원
                    if 'emotion_state' in exp_data and isinstance(exp_data['emotion_state'], dict):
                        emotion_dict = exp_data['emotion_state']
                        # 간단한 EmotionData 복원 (실제로는 더 정교한 방법 필요)
                        exp_data['emotion_state'] = EmotionData(
                            valence=emotion_dict.get('valence', 0.0),
                            arousal=emotion_dict.get('arousal', 0.0),
                            dominance=emotion_dict.get('dominance', 0.0),
                            confidence=emotion_dict.get('confidence', 0.5)
                        )
                    
                    experience = EthicsExperience(**exp_data)
                    experiences.append(experience)
                except Exception as e:
                    self.logger.warning(f"경험 데이터 복원 실패: {e}")
                    continue
            
            return experiences[:min_count * 3]  # 최대 3배까지
    
    def _calculate_experience_weights(self, experiences: List[EthicsExperience]) -> np.ndarray:
        """경험별 가중치 계산"""
        
        weights = []
        current_time = time.time()
        
        for exp in experiences:
            # 시간 가중치 (최근일수록 높음)
            time_weight = np.exp(-(current_time - exp.timestamp) / (86400 * 30))  # 30일 기준
            
            # 피드백 품질 가중치
            quality_weight = exp.feedback_quality
            
            # 결과 신뢰도 가중치 (만족도와 도덕적 정확성 기반)
            result_weight = (exp.user_satisfaction + exp.moral_correctness) / 2.0
            
            # 전체 가중치
            total_weight = time_weight * quality_weight * result_weight
            weights.append(total_weight)
        
        # 정규화
        weights = np.array(weights)
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(len(weights)) / len(weights)
        
        return weights
    
    def _update_ethics_weights(
        self, 
        old_policy: EthicsPolicy,
        experiences: List[EthicsExperience],
        weights: np.ndarray,
        reasoning_trace: List[str]
    ) -> Dict[str, float]:
        """윤리 가중치 업데이트"""
        
        # 현재 가중치
        current_weights = {
            'care_harm': old_policy.care_harm,
            'fairness_cheating': old_policy.fairness_cheating,
            'loyalty_betrayal': old_policy.loyalty_betrayal,
            'authority_subversion': old_policy.authority_subversion,
            'sanctity_degradation': old_policy.sanctity_degradation
        }
        
        # 경험 기반 조정 계산
        adjustments = defaultdict(float)
        
        for i, exp in enumerate(experiences):
            exp_weight = weights[i]
            
            # 결과 평가 (-1: 매우 나쁨, 1: 매우 좋음)
            outcome_score = exp.outcome_rating
            
            # 시나리오 분석 기반 윤리 가중치 조정
            scenario_ethics = self._analyze_scenario_ethics(exp.scenario)
            
            for ethics_type, relevance in scenario_ethics.items():
                if ethics_type in current_weights:
                    # 결과가 좋았다면 해당 윤리 가중치 증가, 나빴다면 감소
                    adjustment = outcome_score * relevance * exp_weight * self.learning_config['learning_rate']
                    adjustments[ethics_type] += adjustment
        
        # 새로운 가중치 계산
        new_weights = {}
        total_adjustment = 0.0
        
        for ethics_type, current_weight in current_weights.items():
            adjustment = adjustments[ethics_type]
            new_weight = current_weight + adjustment
            
            # 범위 제한 (0.1 ~ 0.9)
            new_weight = np.clip(new_weight, 0.1, 0.9)
            new_weights[ethics_type] = new_weight
            total_adjustment += abs(adjustment)
        
        reasoning_trace.append(
            f"윤리 가중치 조정: 총 변화량 {total_adjustment:.4f}"
        )
        
        return new_weights
    
    def _analyze_scenario_ethics(self, scenario: str) -> Dict[str, float]:
        """시나리오에서 윤리 관련성 분석"""
        
        # 키워드 기반 간단한 분석 (실제로는 더 정교한 NLP 필요)
        ethics_keywords = {
            'care_harm': ['안전', '위험', '해로운', '보호', '돌봄', '피해', '상처'],
            'fairness_cheating': ['공정', '불공정', '차별', '평등', '공평', '부당'],
            'loyalty_betrayal': ['배신', '충성', '신뢰', '배반', '약속', '의리'],
            'authority_subversion': ['권위', '규칙', '법', '질서', '복종', '반항'],
            'sanctity_degradation': ['신성', '순수', '더러운', '깨끗', '존엄', '모독']
        }
        
        scenario_lower = scenario.lower()
        relevance_scores = {}
        
        for ethics_type, keywords in ethics_keywords.items():
            score = 0.0
            for keyword in keywords:
                if keyword in scenario_lower:
                    score += 1.0
            
            # 정규화 (0-1 범위)
            relevance_scores[ethics_type] = min(score / len(keywords), 1.0)
        
        return relevance_scores
    
    def _update_individual_community_balance(
        self,
        old_policy: EthicsPolicy,
        experiences: List[EthicsExperience],
        weights: np.ndarray,
        reasoning_trace: List[str]
    ) -> Dict[str, float]:
        """개인-공동체 균형 조정"""
        
        current_individual = old_policy.individual_weight
        current_community = old_policy.community_weight
        
        # 경험 기반 균형 조정
        balance_adjustments = []
        
        for i, exp in enumerate(experiences):
            exp_weight = weights[i]
            
            # 이해관계자 수에 따른 공동체 중요도
            stakeholder_count = len(exp.stakeholders) if exp.stakeholders else 1
            community_importance = min(stakeholder_count / 10.0, 1.0)
            
            # 결과 기반 조정
            if exp.outcome_rating > 0.5:  # 좋은 결과
                if community_importance > 0.7:
                    # 공동체 중심 결정이 성공적이었음
                    balance_adjustments.append(('community', exp_weight * 0.1))
                else:
                    # 개인 중심 결정이 성공적이었음
                    balance_adjustments.append(('individual', exp_weight * 0.1))
            elif exp.outcome_rating < -0.5:  # 나쁜 결과
                if community_importance > 0.7:
                    # 공동체 중심 결정이 실패했음
                    balance_adjustments.append(('individual', exp_weight * 0.05))
                else:
                    # 개인 중심 결정이 실패했음
                    balance_adjustments.append(('community', exp_weight * 0.05))
        
        # 조정 적용
        individual_adjustment = sum(adj for direction, adj in balance_adjustments if direction == 'individual')
        community_adjustment = sum(adj for direction, adj in balance_adjustments if direction == 'community')
        
        new_individual = current_individual + individual_adjustment - community_adjustment
        new_community = current_community + community_adjustment - individual_adjustment
        
        # 정규화 (합이 1이 되도록)
        total = new_individual + new_community
        if total > 0:
            new_individual /= total
            new_community /= total
        else:
            new_individual, new_community = 0.4, 0.6  # 기본값
        
        reasoning_trace.append(
            f"개인-공동체 균형 조정: {current_individual:.3f}->{new_individual:.3f}, "
            f"{current_community:.3f}->{new_community:.3f}"
        )
        
        return {
            'individual': new_individual,
            'community': new_community
        }
    
    def _update_cultural_context(
        self,
        old_policy: EthicsPolicy,
        experiences: List[EthicsExperience],
        weights: np.ndarray,
        reasoning_trace: List[str]
    ) -> Dict[str, float]:
        """문화적 맥락 가중치 업데이트"""
        
        current_context = old_policy.cultural_context.copy()
        
        # 경험에서 문화적 맥락 분석
        cultural_factors = {
            'hierarchy_respect': 0.0,
            'group_harmony': 0.0,
            'face_saving': 0.0,
            'long_term_thinking': 0.0
        }
        
        for i, exp in enumerate(experiences):
            exp_weight = weights[i]
            
            # 시나리오에서 문화적 요소 추출
            scenario_lower = exp.scenario.lower()
            
            # 위계 존중
            if any(word in scenario_lower for word in ['상사', '선배', '권위', '지위']):
                if exp.outcome_rating > 0:
                    cultural_factors['hierarchy_respect'] += exp_weight * 0.1
                else:
                    cultural_factors['hierarchy_respect'] -= exp_weight * 0.05
            
            # 집단 조화
            if any(word in scenario_lower for word in ['팀', '조직', '단체', '협력']):
                if exp.outcome_rating > 0:
                    cultural_factors['group_harmony'] += exp_weight * 0.1
                else:
                    cultural_factors['group_harmony'] -= exp_weight * 0.05
            
            # 체면 중시
            if any(word in scenario_lower for word in ['체면', '명예', '평판', '이미지']):
                if exp.outcome_rating > 0:
                    cultural_factors['face_saving'] += exp_weight * 0.1
                else:
                    cultural_factors['face_saving'] -= exp_weight * 0.05
            
            # 장기적 사고
            if any(word in scenario_lower for word in ['미래', '장기', '지속', '세대']):
                if exp.outcome_rating > 0:
                    cultural_factors['long_term_thinking'] += exp_weight * 0.1
                else:
                    cultural_factors['long_term_thinking'] -= exp_weight * 0.05
        
        # 문화적 맥락 업데이트
        new_context = {}
        for factor, adjustment in cultural_factors.items():
            current_value = current_context.get(factor, 0.7)
            new_value = current_value + adjustment * self.learning_config['cultural_adaptation_rate']
            new_context[factor] = np.clip(new_value, 0.1, 1.0)
        
        reasoning_trace.append(f"문화적 맥락 업데이트 완료")
        
        return new_context
    
    def _calculate_update_magnitude(self, old_policy: EthicsPolicy, new_policy: EthicsPolicy) -> float:
        """정책 업데이트 크기 계산"""
        
        # 윤리 가중치 변화
        ethics_changes = [
            abs(new_policy.care_harm - old_policy.care_harm),
            abs(new_policy.fairness_cheating - old_policy.fairness_cheating),
            abs(new_policy.loyalty_betrayal - old_policy.loyalty_betrayal),
            abs(new_policy.authority_subversion - old_policy.authority_subversion),
            abs(new_policy.sanctity_degradation - old_policy.sanctity_degradation)
        ]
        
        # 균형 변화
        balance_changes = [
            abs(new_policy.individual_weight - old_policy.individual_weight),
            abs(new_policy.community_weight - old_policy.community_weight)
        ]
        
        # 문화적 맥락 변화
        cultural_changes = []
        for key in old_policy.cultural_context:
            old_val = old_policy.cultural_context.get(key, 0.7)
            new_val = new_policy.cultural_context.get(key, 0.7)
            cultural_changes.append(abs(new_val - old_val))
        
        # 전체 변화량 (가중 평균)
        total_change = (
            np.mean(ethics_changes) * 0.5 +
            np.mean(balance_changes) * 0.3 +
            np.mean(cultural_changes) * 0.2
        )
        
        return total_change
    
    def _update_confidence_score(
        self,
        old_policy: EthicsPolicy,
        experiences: List[EthicsExperience],
        update_magnitude: float
    ) -> float:
        """신뢰도 점수 업데이트"""
        
        # 경험 품질 기반 신뢰도
        experience_quality = np.mean([exp.feedback_quality for exp in experiences])
        
        # 결과 일관성 기반 신뢰도
        outcome_variance = np.var([exp.outcome_rating for exp in experiences])
        consistency_score = max(0.0, 1.0 - outcome_variance)
        
        # 업데이트 크기 기반 안정성
        stability_score = max(0.0, 1.0 - update_magnitude * 10)
        
        # 경험 수 기반 신뢰도
        sample_score = min(len(experiences) / 20.0, 1.0)
        
        # 전체 신뢰도 계산
        new_confidence = (
            old_policy.confidence_score * 0.3 +
            experience_quality * 0.25 +
            consistency_score * 0.25 +
            stability_score * 0.1 +
            sample_score * 0.1
        )
        
        return np.clip(new_confidence, 0.1, 1.0)
    
    def _identify_improvement_areas(
        self,
        old_policy: EthicsPolicy,
        new_policy: EthicsPolicy,
        experiences: List[EthicsExperience]
    ) -> List[str]:
        """개선 영역 식별"""
        
        areas = []
        
        # 낮은 만족도 경험 분석
        low_satisfaction_exps = [exp for exp in experiences if exp.user_satisfaction < 0.3]
        if len(low_satisfaction_exps) > len(experiences) * 0.3:
            areas.append("사용자 만족도 개선 필요")
        
        # 높은 후회 경험 분석
        high_regret_exps = [exp for exp in experiences if exp.actual_regret > 0.7]
        if len(high_regret_exps) > len(experiences) * 0.2:
            areas.append("후회 최소화 전략 필요")
        
        # 도덕적 정확성 분석
        low_moral_exps = [exp for exp in experiences if exp.moral_correctness < 0.4]
        if len(low_moral_exps) > len(experiences) * 0.2:
            areas.append("도덕적 판단 정확성 향상 필요")
        
        # 신뢰도 분석
        if new_policy.confidence_score < 0.6:
            areas.append("정책 신뢰도 향상 필요")
        
        return areas
    
    def _create_no_update_result(self, policy: EthicsPolicy, reason: str) -> PolicyUpdateResult:
        """업데이트 없는 결과 생성"""
        return PolicyUpdateResult(
            old_policy=policy,
            new_policy=policy,
            update_magnitude=0.0,
            convergence_achieved=True,
            improvement_areas=[],
            confidence_change=0.0,
            reasoning_trace=[f"업데이트 미실행: {reason}"]
        )
    
    def _save_update_history(self, result: PolicyUpdateResult):
        """업데이트 히스토리 저장"""
        
        update_id = f"update_{int(time.time())}"
        improvement_score = 1.0 if len(result.improvement_areas) == 0 else 0.5
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO policy_updates 
                (update_id, policy_id, old_policy_data, new_policy_data,
                 update_magnitude, improvement_score, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                update_id,
                result.new_policy.policy_id,
                json.dumps(result.old_policy.to_dict()),
                json.dumps(result.new_policy.to_dict()),
                result.update_magnitude,
                improvement_score,
                time.time()
            ))
            conn.commit()
    
    def _update_statistics(self, result: PolicyUpdateResult):
        """통계 업데이트"""
        
        self.update_statistics['total_updates'] += 1
        
        if len(result.improvement_areas) == 0:
            self.update_statistics['successful_updates'] += 1
        
        # 평균 개선도 업데이트
        improvement = 1.0 if len(result.improvement_areas) == 0 else 0.0
        total = self.update_statistics['total_updates']
        current_avg = self.update_statistics['average_improvement']
        new_avg = (current_avg * (total - 1) + improvement) / total
        self.update_statistics['average_improvement'] = new_avg
        
        # 수렴률 업데이트
        convergence = 1.0 if result.convergence_achieved else 0.0
        current_conv = self.update_statistics['convergence_rate']
        new_conv = (current_conv * (total - 1) + convergence) / total
        self.update_statistics['convergence_rate'] = new_conv
    
    def get_analytics(self, user_id: str = "default") -> Dict[str, Any]:
        """분석 정보 반환 (호환성을 위한 별칭)"""
        return self.get_policy_analytics(user_id)
    
    def get_policy_analytics(self, user_id: str = "default") -> Dict[str, Any]:
        """정책 분석 정보 반환"""
        
        policy = self.get_policy(user_id)
        
        # 최근 업데이트 히스토리
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT update_magnitude, improvement_score, timestamp 
                FROM policy_updates 
                WHERE policy_id LIKE ? 
                ORDER BY timestamp DESC 
                LIMIT 10
            ''', (f"{user_id}%",))
            
            recent_updates = cursor.fetchall()
        
        return {
            'current_policy': policy.to_dict(),
            'update_statistics': self.update_statistics,
            'recent_updates': [
                {
                    'magnitude': row[0],
                    'improvement': row[1],
                    'timestamp': row[2]
                } for row in recent_updates
            ],
            'experience_buffer_size': len(self.experience_buffer)
        }


class MultiEthicsCalculator:
    """다차원 윤리 계산기 (공리주의 + 덕윤리 + 의무론)"""
    
    def __init__(self):
        self.ethics_weights = {
            'utilitarian': 0.4,    # 공리주의 (기존 벤담 계산)
            'virtue_ethics': 0.3,  # 덕 윤리학
            'deontological': 0.3   # 의무론적 윤리학
        }
    
    def calculate_multi_dimensional_ethics(
        self,
        scenario: str,
        stakeholders: List[str],
        consequences: Dict[str, float],
        moral_rules: List[str],
        virtues_involved: List[str]
    ) -> Dict[str, float]:
        """다차원 윤리 점수 계산"""
        
        # 1. 공리주의적 계산 (결과 기반)
        utilitarian_score = self._calculate_utilitarian_score(consequences, stakeholders)
        
        # 2. 덕 윤리학적 계산 (성품 기반)
        virtue_score = self._calculate_virtue_score(virtues_involved, scenario)
        
        # 3. 의무론적 계산 (규칙 기반)
        deontological_score = self._calculate_deontological_score(moral_rules, scenario)
        
        # 4. 통합 점수 계산
        integrated_score = (
            utilitarian_score * self.ethics_weights['utilitarian'] +
            virtue_score * self.ethics_weights['virtue_ethics'] +
            deontological_score * self.ethics_weights['deontological']
        )
        
        return {
            'utilitarian': utilitarian_score,
            'virtue_ethics': virtue_score,
            'deontological': deontological_score,
            'integrated': integrated_score
        }
    
    def _calculate_utilitarian_score(self, consequences: Dict[str, float], stakeholders: List[str]) -> float:
        """공리주의적 점수 계산"""
        if not consequences:
            return 0.5
        
        # 결과의 전체 효용 계산
        total_utility = sum(consequences.values())
        stakeholder_count = len(stakeholders) if stakeholders else 1
        
        # 평균 효용으로 정규화
        average_utility = total_utility / max(stakeholder_count, 1)
        
        # 0-1 범위로 변환
        return np.clip((average_utility + 1) / 2, 0.0, 1.0)
    
    def _calculate_virtue_score(self, virtues: List[str], scenario: str) -> float:
        """덕 윤리학적 점수 계산"""
        if not virtues:
            return 0.5
        
        # 덕목별 점수 매핑
        virtue_scores = {
            'courage': 0.8,      # 용기
            'justice': 0.9,      # 정의
            'temperance': 0.7,   # 절제
            'wisdom': 0.8,       # 지혜
            'compassion': 0.9,   # 연민
            'honesty': 0.8,      # 정직
            'integrity': 0.9,    # 성실성
            'humility': 0.7      # 겸손
        }
        
        # 시나리오에서 요구되는 덕목들의 평균 점수
        relevant_scores = [virtue_scores.get(virtue.lower(), 0.5) for virtue in virtues]
        
        return np.mean(relevant_scores) if relevant_scores else 0.5
    
    def _calculate_deontological_score(self, moral_rules: List[str], scenario: str) -> float:
        """의무론적 점수 계산"""
        if not moral_rules:
            return 0.5
        
        # 도덕 규칙별 준수 점수
        rule_compliance = {
            'no_harm': 0.9,           # 해를 끼치지 말라
            'truth_telling': 0.8,     # 진실을 말하라
            'promise_keeping': 0.8,   # 약속을 지켜라
            'respect_autonomy': 0.9,  # 자율성을 존중하라
            'fairness': 0.9,          # 공정하게 대하라
            'respect_dignity': 1.0    # 인간 존엄성을 존중하라
        }
        
        # 관련 규칙들의 평균 준수 점수
        relevant_scores = [rule_compliance.get(rule.lower(), 0.5) for rule in moral_rules]
        
        return np.mean(relevant_scores) if relevant_scores else 0.5


# 테스트 및 데모 함수
def test_ethics_policy_updater():
    """윤리 정책 자동 조정기 테스트"""
    print("🔧 윤리 정책 자동 조정기 테스트 시작")
    
    # 조정기 초기화
    updater = EthicsPolicyUpdater()
    
    # 테스트 경험 데이터 생성
    test_experiences = [
        EthicsExperience(
            experience_id="exp_1",
            scenario="팀 프로젝트에서 동료가 기여하지 않았지만 공정한 평가를 받았습니다.",
            decision_made="동료에게 직접 이야기하고 상황을 개선하려 노력했습니다.",
            outcome_rating=0.7,
            emotion_state=EmotionData(valence=0.2, arousal=0.6, dominance=0.5, confidence=0.8),
            stakeholders=["동료", "팀원들", "프로젝트 성과"],
            cultural_context="직장",
            decision_urgency=0.6,
            actual_regret=0.3,
            user_satisfaction=0.8,
            moral_correctness=0.9
        ),
        EthicsExperience(
            experience_id="exp_2",
            scenario="상사가 부당한 요구를 했지만 거부하기 어려운 상황이었습니다.",
            decision_made="정중하게 대안을 제시하며 거부했습니다.",
            outcome_rating=0.4,
            emotion_state=EmotionData(valence=-0.3, arousal=0.8, dominance=0.3, confidence=0.7),
            stakeholders=["상사", "자신", "조직"],
            cultural_context="위계질서",
            decision_urgency=0.8,
            actual_regret=0.5,
            user_satisfaction=0.6,
            moral_correctness=0.8
        )
    ]
    
    # 경험 추가
    for exp in test_experiences:
        updater.add_experience(exp)
    
    # 초기 정책 확인
    initial_policy = updater.get_policy()
    print(f"초기 정책 신뢰도: {initial_policy.confidence_score:.3f}")
    
    # 정책 업데이트 실행
    update_result = updater.update_policy_from_experiences(min_experiences=2)
    
    # 결과 출력
    print(f"\n📊 정책 업데이트 결과:")
    print(f"- 업데이트 크기: {update_result.update_magnitude:.4f}")
    print(f"- 수렴 달성: {'예' if update_result.convergence_achieved else '아니오'}")
    print(f"- 신뢰도 변화: {update_result.confidence_change:.3f}")
    print(f"- 개선 영역: {', '.join(update_result.improvement_areas)}")
    
    print(f"\n⚖️ 윤리 가중치 변화:")
    old_policy = update_result.old_policy
    new_policy = update_result.new_policy
    
    ethics_types = ['care_harm', 'fairness_cheating', 'loyalty_betrayal', 
                   'authority_subversion', 'sanctity_degradation']
    
    for ethics_type in ethics_types:
        old_val = getattr(old_policy, ethics_type)
        new_val = getattr(new_policy, ethics_type)
        change = new_val - old_val
        print(f"- {ethics_type}: {old_val:.3f} → {new_val:.3f} ({change:+.3f})")
    
    print(f"\n🏛️ 개인-공동체 균형 변화:")
    print(f"- 개인: {old_policy.individual_weight:.3f} → {new_policy.individual_weight:.3f}")
    print(f"- 공동체: {old_policy.community_weight:.3f} → {new_policy.community_weight:.3f}")
    
    print(f"\n🌏 문화적 맥락 변화:")
    for key in old_policy.cultural_context:
        old_val = old_policy.cultural_context[key]
        new_val = new_policy.cultural_context[key]
        change = new_val - old_val
        print(f"- {key}: {old_val:.3f} → {new_val:.3f} ({change:+.3f})")
    
    # 분석 정보
    analytics = updater.get_policy_analytics()
    print(f"\n📈 정책 분석:")
    print(f"- 총 업데이트: {analytics['update_statistics']['total_updates']}")
    print(f"- 성공률: {analytics['update_statistics']['average_improvement']:.3f}")
    print(f"- 수렴률: {analytics['update_statistics']['convergence_rate']:.3f}")
    
    return updater, update_result


if __name__ == "__main__":
    test_ethics_policy_updater()
