"""
고급 후회 학습 시스템 - Linux 전용
Advanced Regret Learning System for Linux

문학 기반 시계열 후회 학습:
- 페이즈 전환: 학습 횟수 & 후회 임계값 기반
- 다층적 후회: 상위 페이즈에서 하위 페이즈 후회도 지속 반영
- 시계열 패턴: 문학 데이터 I/O를 통한 시간적 학습
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
from collections import deque, defaultdict
from enum import Enum
import pickle
import torch
import torch.nn as nn

# 시스템 설정
from config import ADVANCED_CONFIG, CACHE_DIR, MODELS_DIR, LOGS_DIR

# 로깅 설정
logger = logging.getLogger(__name__)

class RegretType(Enum):
    """후회 유형"""
    ACTION_REGRET = "action"  # 행위에 대한 후회
    INACTION_REGRET = "inaction"  # 무행위에 대한 후회
    TIMING_REGRET = "timing"  # 시기에 대한 후회
    CHOICE_REGRET = "choice"  # 선택에 대한 후회
    EMPATHY_REGRET = "empathy"  # 공감 실패 후회
    PREDICTION_REGRET = "prediction"  # 예측 실패 후회

class LearningPhase(Enum):
    """학습 페이즈"""
    PHASE_0 = 0  # 자신 감정 캘리브레이션
    PHASE_1 = 1  # 타인 공감 학습
    PHASE_2 = 2  # 공동체 이해

@dataclass
class RegretMemory:
    """후회 메모리"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    regret_type: RegretType = RegretType.ACTION_REGRET
    intensity: float = 0.0  # 0.0 ~ 1.0
    
    # 상황 정보
    situation_context: Dict[str, Any] = field(default_factory=dict)
    chosen_action: Dict[str, Any] = field(default_factory=dict)
    alternative_actions: List[Dict[str, Any]] = field(default_factory=list)
    
    # 결과 정보
    actual_outcome: Dict[str, Any] = field(default_factory=dict)
    counterfactual_outcomes: List[Dict[str, Any]] = field(default_factory=list)
    
    # 학습 정보
    learning_phase: LearningPhase = LearningPhase.PHASE_0
    learned_pattern: Dict[str, Any] = field(default_factory=dict)
    update_weights: Dict[str, float] = field(default_factory=dict)
    
    # 시계열 정보
    timestamp: datetime = field(default_factory=datetime.now)
    temporal_distance: float = 0.0  # 사건으로부터의 시간 거리
    decay_factor: float = 1.0  # 시간에 따른 감소
    
    # 문학적 참조
    literary_reference: str = ""
    narrative_pattern: str = ""

@dataclass
class PhaseTransition:
    """페이즈 전환 정보"""
    from_phase: LearningPhase
    to_phase: LearningPhase
    transition_time: datetime
    trigger_condition: str  # "threshold" or "count"
    metrics_at_transition: Dict[str, float]
    
@dataclass
class RegretLearningState:
    """후회 학습 상태"""
    current_phase: LearningPhase = LearningPhase.PHASE_0
    phase_learning_counts: Dict[LearningPhase, int] = field(default_factory=dict)
    phase_regret_levels: Dict[LearningPhase, List[float]] = field(default_factory=lambda: {
        LearningPhase.PHASE_0: [],
        LearningPhase.PHASE_1: [],
        LearningPhase.PHASE_2: []
    })
    phase_transitions: List[PhaseTransition] = field(default_factory=list)
    active_learning_phases: Set[LearningPhase] = field(default_factory=lambda: {LearningPhase.PHASE_0})

class AdvancedRegretLearningSystem:
    """고급 후회 학습 시스템"""
    
    def __init__(self):
        """시스템 초기화"""
        self.config = ADVANCED_CONFIG.get('regret_learning', {
            'phase_transition_threshold': 0.3,  # 후회 임계값
            'min_learning_count': 50,  # 최소 학습 횟수
            'temporal_window': 100,  # 시계열 윈도우
            'decay_rate': 0.95,  # 시간 감쇠율
            'multi_phase_weights': {  # 다층 학습 가중치
                LearningPhase.PHASE_0: 1.0,
                LearningPhase.PHASE_1: 0.7,
                LearningPhase.PHASE_2: 0.5
            }
        })
        
        # 학습 상태
        self.learning_state = RegretLearningState()
        
        # 메모리 시스템
        self.regret_memories = {
            phase: deque(maxlen=1000) for phase in LearningPhase
        }
        self.temporal_patterns = defaultdict(list)
        
        # 학습 모델
        self.regret_models = {
            phase: self._initialize_phase_model(phase) for phase in LearningPhase
        }
        
        # 문학 패턴 데이터베이스
        self.literary_patterns = self._load_literary_patterns()
        
        # 성능 추적
        self.performance_metrics = {
            'total_regrets': 0,
            'phase_accuracies': {phase: [] for phase in LearningPhase},
            'learning_curves': {phase: [] for phase in LearningPhase}
        }
        
        logger.info("고급 후회 학습 시스템이 초기화되었습니다.")
    
    def _initialize_phase_model(self, phase: LearningPhase) -> Dict[str, Any]:
        """페이즈별 모델 초기화"""
        if phase == LearningPhase.PHASE_0:
            return {
                'calibration_factors': defaultdict(lambda: 1.0),
                'emotion_mappings': {},
                'self_understanding': 0.0
            }
        elif phase == LearningPhase.PHASE_1:
            return {
                'empathy_patterns': defaultdict(list),
                'prediction_models': {},
                'other_understanding': 0.0
            }
        else:  # PHASE_2
            return {
                'community_dynamics': defaultdict(dict),
                'consensus_patterns': {},
                'collective_understanding': 0.0
            }
    
    def _load_literary_patterns(self) -> Dict[str, Dict[str, Any]]:
        """문학적 후회 패턴 로드"""
        return {
            'tragic_choice': {
                'description': '비극적 선택에 대한 후회',
                'intensity_range': (0.7, 1.0),
                'temporal_pattern': 'persistent',
                'examples': ['햄릿의 복수 지연', '로미오의 성급한 자살']
            },
            'missed_opportunity': {
                'description': '놓친 기회에 대한 후회',
                'intensity_range': (0.5, 0.8),
                'temporal_pattern': 'growing',
                'examples': ['춘향전의 이별', '그레이트 개츠비의 재회']
            },
            'moral_compromise': {
                'description': '도덕적 타협에 대한 후회',
                'intensity_range': (0.6, 0.9),
                'temporal_pattern': 'fluctuating',
                'examples': ['맥베스의 왕위 찬탈', '파우스트의 영혼 거래']
            },
            'empathy_failure': {
                'description': '공감 실패에 대한 후회',
                'intensity_range': (0.4, 0.7),
                'temporal_pattern': 'delayed',
                'examples': ['리어왕의 딸들 오해', '오이디푸스의 운명']
            }
        }
    
    async def process_regret(self,
                           situation: Dict[str, Any],
                           outcome: Dict[str, Any],
                           alternatives: List[Dict[str, Any]],
                           literary_context: Dict[str, Any] = None) -> RegretMemory:
        """
        후회 처리 및 학습
        
        Args:
            situation: 상황 정보
            outcome: 실제 결과
            alternatives: 대안적 선택들
            literary_context: 문학적 맥락
            
        Returns:
            생성된 후회 메모리
        """
        try:
            # 후회 강도 계산
            regret_intensity = await self._calculate_regret_intensity(
                situation, outcome, alternatives
            )
            
            # 후회 유형 분류
            regret_type = self._classify_regret_type(
                situation, outcome, alternatives
            )
            
            # 문학적 패턴 매칭
            narrative_pattern = self._match_literary_pattern(
                regret_type, regret_intensity, literary_context
            )
            
            # 후회 메모리 생성
            regret_memory = RegretMemory(
                regret_type=regret_type,
                intensity=regret_intensity,
                situation_context=situation,
                chosen_action=situation.get('chosen_action', {}),
                alternative_actions=alternatives,
                actual_outcome=outcome,
                counterfactual_outcomes=await self._generate_counterfactuals(
                    situation, alternatives
                ),
                learning_phase=self.learning_state.current_phase,
                literary_reference=literary_context.get('source', '') if literary_context else '',
                narrative_pattern=narrative_pattern
            )
            
            # 다층적 학습 수행
            await self._multi_phase_learning(regret_memory)
            
            # 페이즈 전환 체크
            await self._check_phase_transition()
            
            # 시계열 패턴 업데이트
            self._update_temporal_patterns(regret_memory)
            
            # 성능 메트릭 업데이트
            self._update_performance_metrics(regret_memory)
            
            logger.info(f"후회 처리 완료: 유형={regret_type.value}, 강도={regret_intensity:.3f}")
            return regret_memory
            
        except Exception as e:
            logger.error(f"후회 처리 실패: {e}")
            return RegretMemory()
    
    async def _calculate_regret_intensity(self,
                                        situation: Dict[str, Any],
                                        outcome: Dict[str, Any],
                                        alternatives: List[Dict[str, Any]]) -> float:
        """후회 강도 계산"""
        try:
            # 기본 후회 (결과의 부정성)
            outcome_negativity = outcome.get('negativity', 0.5)
            base_regret = outcome_negativity
            
            # 대안 비교 후회
            alternative_regret = 0.0
            if alternatives:
                # 최선의 대안과 비교
                best_alternative_value = max(
                    alt.get('expected_value', 0.0) for alt in alternatives
                )
                actual_value = outcome.get('actual_value', 0.0)
                alternative_regret = max(0, best_alternative_value - actual_value)
            
            # 예측 오차 후회
            if 'expected_outcome' in situation and 'actual_outcome' in outcome:
                prediction_error = abs(
                    situation['expected_outcome'] - outcome['actual_outcome']
                )
                prediction_regret = prediction_error * 0.5
            else:
                prediction_regret = 0.0
            
            # 시간적 요인 (늦은 깨달음)
            temporal_factor = 1.0
            if 'decision_time' in situation and 'realization_time' in outcome:
                time_delay = (outcome['realization_time'] - situation['decision_time']).total_seconds()
                temporal_factor = 1.0 + (time_delay / 86400) * 0.1  # 하루당 10% 증가
            
            # 종합 후회 강도
            total_regret = (
                base_regret * 0.4 +
                alternative_regret * 0.4 +
                prediction_regret * 0.2
            ) * temporal_factor
            
            return np.clip(total_regret, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"후회 강도 계산 실패: {e}")
            return 0.5
    
    def _classify_regret_type(self,
                            situation: Dict[str, Any],
                            outcome: Dict[str, Any],
                            alternatives: List[Dict[str, Any]]) -> RegretType:
        """후회 유형 분류"""
        # 행위 vs 무행위
        if situation.get('action_taken', True):
            if outcome.get('timing_issue', False):
                return RegretType.TIMING_REGRET
            elif alternatives and len(alternatives) > 1:
                return RegretType.CHOICE_REGRET
            else:
                return RegretType.ACTION_REGRET
        else:
            return RegretType.INACTION_REGRET
        
        # 특수 케이스
        if outcome.get('empathy_failure', False):
            return RegretType.EMPATHY_REGRET
        elif outcome.get('prediction_failure', False):
            return RegretType.PREDICTION_REGRET
        
        return RegretType.ACTION_REGRET
    
    def _match_literary_pattern(self,
                              regret_type: RegretType,
                              intensity: float,
                              literary_context: Dict[str, Any]) -> str:
        """문학적 패턴 매칭"""
        best_match = 'general_regret'
        best_score = 0.0
        
        for pattern_name, pattern_info in self.literary_patterns.items():
            # 강도 범위 체크
            min_intensity, max_intensity = pattern_info['intensity_range']
            if min_intensity <= intensity <= max_intensity:
                score = 1.0 - abs(intensity - (min_intensity + max_intensity) / 2)
                
                # 문학적 맥락 일치도
                if literary_context and 'genre' in literary_context:
                    if literary_context['genre'] in pattern_info.get('genres', []):
                        score += 0.3
                
                if score > best_score:
                    best_score = score
                    best_match = pattern_name
        
        return best_match
    
    async def _generate_counterfactuals(self,
                                      situation: Dict[str, Any],
                                      alternatives: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """반사실적 결과 생성"""
        counterfactuals = []
        
        for alt in alternatives[:3]:  # 최대 3개
            # 간단한 시뮬레이션
            counterfactual = {
                'action': alt.get('action', 'unknown'),
                'estimated_outcome': alt.get('expected_value', 0.5),
                'confidence': alt.get('confidence', 0.5),
                'reasoning': alt.get('reasoning', '')
            }
            counterfactuals.append(counterfactual)
        
        return counterfactuals
    
    async def _multi_phase_learning(self, regret_memory: RegretMemory):
        """다층적 학습 - 모든 활성 페이즈에서 학습"""
        
        # 현재 페이즈 학습
        primary_learning = await self._learn_in_phase(
            regret_memory, 
            self.learning_state.current_phase,
            weight=1.0
        )
        
        # 하위 페이즈들도 함께 학습 (감소된 가중치로)
        for phase in self.learning_state.active_learning_phases:
            if phase != self.learning_state.current_phase:
                weight = self.config['multi_phase_weights'].get(phase, 0.5)
                
                # 중요한 상황이 아니면 더 낮은 가중치
                if not regret_memory.situation_context.get('is_critical', False):
                    weight *= 0.7
                
                await self._learn_in_phase(regret_memory, phase, weight)
        
        # 학습 횟수 업데이트
        current_count = self.learning_state.phase_learning_counts.get(
            self.learning_state.current_phase, 0
        )
        self.learning_state.phase_learning_counts[
            self.learning_state.current_phase
        ] = current_count + 1
        
        # 후회 수준 기록
        self.learning_state.phase_regret_levels[
            self.learning_state.current_phase
        ].append(regret_memory.intensity)
    
    async def _learn_in_phase(self,
                            regret_memory: RegretMemory,
                            phase: LearningPhase,
                            weight: float) -> Dict[str, Any]:
        """특정 페이즈에서의 학습"""
        model = self.regret_models[phase]
        learning_result = {}
        
        if phase == LearningPhase.PHASE_0:
            # 자기 이해 학습
            emotion_key = self._extract_emotion_key(regret_memory.situation_context)
            current_factor = model['calibration_factors'][emotion_key]
            
            # 후회 기반 캘리브레이션 조정
            adjustment = regret_memory.intensity * weight * 0.1
            model['calibration_factors'][emotion_key] = current_factor - adjustment
            
            # 자기 이해도 업데이트
            model['self_understanding'] = min(1.0, 
                model['self_understanding'] + (1 - regret_memory.intensity) * weight * 0.01
            )
            
            learning_result = {
                'phase': 'PHASE_0',
                'calibration_update': adjustment,
                'self_understanding': model['self_understanding']
            }
            
        elif phase == LearningPhase.PHASE_1:
            # 타인 이해 학습
            pattern_key = self._extract_pattern_key(regret_memory.situation_context)
            
            # 공감 패턴 업데이트
            model['empathy_patterns'][pattern_key].append({
                'regret_intensity': regret_memory.intensity,
                'context': regret_memory.situation_context,
                'timestamp': regret_memory.timestamp
            })
            
            # 예측 모델 개선
            if regret_memory.regret_type == RegretType.EMPATHY_REGRET:
                model['other_understanding'] = min(1.0,
                    model['other_understanding'] + (1 - regret_memory.intensity) * weight * 0.02
                )
            
            learning_result = {
                'phase': 'PHASE_1',
                'pattern_update': pattern_key,
                'other_understanding': model['other_understanding']
            }
            
        else:  # PHASE_2
            # 공동체 이해 학습
            community_key = self._extract_community_key(regret_memory.situation_context)
            
            # 집단 역학 업데이트
            if community_key not in model['community_dynamics']:
                model['community_dynamics'][community_key] = {
                    'consensus_failures': 0,
                    'successful_predictions': 0
                }
            
            if regret_memory.intensity > 0.5:
                model['community_dynamics'][community_key]['consensus_failures'] += weight
            else:
                model['community_dynamics'][community_key]['successful_predictions'] += weight
            
            # 집단 이해도 업데이트
            model['collective_understanding'] = min(1.0,
                model['collective_understanding'] + (1 - regret_memory.intensity) * weight * 0.015
            )
            
            learning_result = {
                'phase': 'PHASE_2',
                'community_update': community_key,
                'collective_understanding': model['collective_understanding']
            }
        
        # 메모리에 저장
        self.regret_memories[phase].append(regret_memory)
        
        return learning_result
    
    async def _check_phase_transition(self):
        """페이즈 전환 체크"""
        current_phase = self.learning_state.current_phase
        
        # 이미 최고 페이즈면 전환 없음
        if current_phase == LearningPhase.PHASE_2:
            return
        
        # 학습 횟수 체크
        learning_count = self.learning_state.phase_learning_counts.get(current_phase, 0)
        if learning_count < self.config['min_learning_count']:
            return
        
        # 최근 후회 수준 체크
        recent_regrets = self.learning_state.phase_regret_levels[current_phase][-20:]
        if not recent_regrets:
            return
        
        avg_recent_regret = np.mean(recent_regrets)
        
        # 전환 조건: 평균 후회가 임계값 이하
        if avg_recent_regret < self.config['phase_transition_threshold']:
            # 다음 페이즈로 전환
            next_phase = LearningPhase(current_phase.value + 1)
            
            # 전환 기록
            transition = PhaseTransition(
                from_phase=current_phase,
                to_phase=next_phase,
                transition_time=datetime.now(),
                trigger_condition='threshold',
                metrics_at_transition={
                    'learning_count': learning_count,
                    'avg_regret': avg_recent_regret,
                    'model_understanding': self._get_phase_understanding(current_phase)
                }
            )
            
            self.learning_state.phase_transitions.append(transition)
            self.learning_state.current_phase = next_phase
            self.learning_state.active_learning_phases.add(next_phase)
            
            logger.info(f"페이즈 전환: {current_phase.name} → {next_phase.name}")
            logger.info(f"전환 시 평균 후회: {avg_recent_regret:.3f}")
    
    def _get_phase_understanding(self, phase: LearningPhase) -> float:
        """페이즈별 이해도 반환"""
        model = self.regret_models[phase]
        
        if phase == LearningPhase.PHASE_0:
            return model['self_understanding']
        elif phase == LearningPhase.PHASE_1:
            return model['other_understanding']
        else:
            return model['collective_understanding']
    
    def _update_temporal_patterns(self, regret_memory: RegretMemory):
        """시계열 패턴 업데이트"""
        pattern_key = regret_memory.narrative_pattern
        
        # 시간 창 내의 패턴 저장
        self.temporal_patterns[pattern_key].append({
            'timestamp': regret_memory.timestamp,
            'intensity': regret_memory.intensity,
            'phase': regret_memory.learning_phase,
            'decay': regret_memory.decay_factor
        })
        
        # 오래된 패턴 제거
        cutoff_time = datetime.now() - timedelta(days=30)
        self.temporal_patterns[pattern_key] = [
            p for p in self.temporal_patterns[pattern_key]
            if p['timestamp'] > cutoff_time
        ]
    
    def _update_performance_metrics(self, regret_memory: RegretMemory):
        """성능 메트릭 업데이트"""
        self.performance_metrics['total_regrets'] += 1
        
        # 페이즈별 정확도 (낮은 후회 = 높은 정확도)
        phase = regret_memory.learning_phase
        accuracy = 1.0 - regret_memory.intensity
        self.performance_metrics['phase_accuracies'][phase].append(accuracy)
        
        # 학습 곡선
        if len(self.performance_metrics['phase_accuracies'][phase]) > 10:
            recent_accuracy = np.mean(
                self.performance_metrics['phase_accuracies'][phase][-10:]
            )
            self.performance_metrics['learning_curves'][phase].append(recent_accuracy)
    
    def _extract_emotion_key(self, context: Dict[str, Any]) -> str:
        """감정 키 추출 (Phase 0)"""
        emotion_type = context.get('emotion_type', 'neutral')
        situation_type = context.get('situation_type', 'general')
        return f"{emotion_type}|{situation_type}"
    
    def _extract_pattern_key(self, context: Dict[str, Any]) -> str:
        """패턴 키 추출 (Phase 1)"""
        relationship = context.get('relationship_type', 'stranger')
        interaction = context.get('interaction_type', 'neutral')
        return f"{relationship}|{interaction}"
    
    def _extract_community_key(self, context: Dict[str, Any]) -> str:
        """커뮤니티 키 추출 (Phase 2)"""
        group_type = context.get('group_type', 'general')
        cultural_context = context.get('cultural_context', 'default')
        return f"{group_type}|{cultural_context}"
    
    async def generate_regret_report(self) -> Dict[str, Any]:
        """후회 학습 리포트 생성"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'current_phase': self.learning_state.current_phase.name,
            'active_phases': [p.name for p in self.learning_state.active_learning_phases],
            'total_regrets_processed': self.performance_metrics['total_regrets'],
            'phase_transitions': len(self.learning_state.phase_transitions)
        }
        
        # 페이즈별 상태
        phase_states = {}
        for phase in LearningPhase:
            model = self.regret_models[phase]
            recent_regrets = self.learning_state.phase_regret_levels[phase][-50:]
            
            phase_states[phase.name] = {
                'learning_count': self.learning_state.phase_learning_counts.get(phase, 0),
                'understanding_level': self._get_phase_understanding(phase),
                'avg_recent_regret': np.mean(recent_regrets) if recent_regrets else 0.0,
                'memory_count': len(self.regret_memories[phase])
            }
        
        report['phase_states'] = phase_states
        
        # 시계열 패턴 요약
        pattern_summary = {}
        for pattern, entries in self.temporal_patterns.items():
            if entries:
                recent_entries = entries[-10:]
                pattern_summary[pattern] = {
                    'count': len(entries),
                    'avg_intensity': np.mean([e['intensity'] for e in recent_entries]),
                    'trend': self._calculate_trend([e['intensity'] for e in recent_entries])
                }
        
        report['temporal_patterns'] = pattern_summary
        
        # 학습 곡선
        learning_curves = {}
        for phase, curve in self.performance_metrics['learning_curves'].items():
            if curve:
                learning_curves[phase.name] = {
                    'current_performance': curve[-1] if curve else 0.0,
                    'improvement_rate': self._calculate_improvement_rate(curve)
                }
        
        report['learning_curves'] = learning_curves
        
        return report
    
    def _calculate_trend(self, values: List[float]) -> str:
        """추세 계산"""
        if len(values) < 2:
            return 'stable'
        
        # 선형 회귀로 추세 계산
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return 'increasing'
        elif slope < -0.01:
            return 'decreasing'
        else:
            return 'stable'
    
    def _calculate_improvement_rate(self, curve: List[float]) -> float:
        """개선율 계산"""
        if len(curve) < 2:
            return 0.0
        
        # 초기 대비 현재 성능
        initial_performance = np.mean(curve[:5]) if len(curve) >= 5 else curve[0]
        current_performance = np.mean(curve[-5:]) if len(curve) >= 5 else curve[-1]
        
        if initial_performance > 0:
            return (current_performance - initial_performance) / initial_performance
        else:
            return 0.0
    
    async def save_state(self, filepath: str = None):
        """학습 상태 저장"""
        if filepath is None:
            filepath = os.path.join(
                MODELS_DIR,
                f"regret_learning_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            )
        
        try:
            state_data = {
                'learning_state': asdict(self.learning_state),
                'regret_models': self.regret_models,
                'temporal_patterns': dict(self.temporal_patterns),
                'performance_metrics': self.performance_metrics,
                'config': self.config
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(state_data, f)
            
            logger.info(f"학습 상태가 저장되었습니다: {filepath}")
            
        except Exception as e:
            logger.error(f"학습 상태 저장 실패: {e}")
    
    async def load_state(self, filepath: str):
        """학습 상태 로드"""
        try:
            with open(filepath, 'rb') as f:
                state_data = pickle.load(f)
            
            # 상태 복원
            self.learning_state = RegretLearningState(**state_data['learning_state'])
            self.regret_models = state_data['regret_models']
            self.temporal_patterns = defaultdict(list, state_data['temporal_patterns'])
            self.performance_metrics = state_data['performance_metrics']
            
            logger.info(f"학습 상태가 로드되었습니다: {filepath}")
            
        except Exception as e:
            logger.error(f"학습 상태 로드 실패: {e}")
    
    def get_pytorch_network(self) -> Optional[torch.nn.Module]:
        """PyTorch 네트워크 반환 (HeadAdapter와의 호환성)"""
        # AdvancedRegretLearningSystem은 전통적인 딥러닝 모델 대신
        # 딕셔너리 기반 통계 모델을 사용하므로 None 반환
        logger.warning("AdvancedRegretLearningSystem: PyTorch 네트워크가 없음 (딕셔너리 기반 모델)")
        return None
    
    async def analyze(self, counterfactuals: Any = None, 
                      bentham_score: Dict[str, float] = None) -> Dict[str, Any]:
        """
        후회 분석 (간단한 분석 인터페이스)
        
        Args:
            counterfactuals: 반사실적 시나리오
            bentham_score: 벤담 점수
            
        Returns:
            분석 결과
        """
        result = {
            'regret_level': 0.0,
            'learning_insights': [],
            'suggested_alternatives': [],
            'confidence': 0.0
        }
        
        try:
            # 1. 반사실 시나리오가 있으면 분석
            if counterfactuals:
                if hasattr(counterfactuals, 'scenarios'):
                    # 시나리오별 후회 수준 계산
                    regret_scores = []
                    for scenario in counterfactuals.scenarios[:3]:
                        # 간단한 후회 점수 계산
                        scenario_score = getattr(scenario, 'hedonic_score', 0.5)
                        current_score = bentham_score.get('final_score', 0.5) if bentham_score else 0.5
                        regret = max(0, scenario_score - current_score)
                        regret_scores.append(regret)
                    
                    result['regret_level'] = np.mean(regret_scores) if regret_scores else 0.0
                
                # 2. 학습 인사이트 생성
                if result['regret_level'] > 0.3:
                    result['learning_insights'].append("높은 후회 수준 - 대안적 선택 고려 필요")
                    result['learning_insights'].append("더 나은 결과를 위한 행동 패턴 재검토")
                elif result['regret_level'] > 0.1:
                    result['learning_insights'].append("중간 수준 후회 - 부분적 개선 가능")
                else:
                    result['learning_insights'].append("낮은 후회 - 현재 선택이 적절함")
            
            # 3. 벤담 점수 기반 분석
            if bentham_score:
                score = bentham_score.get('final_score', 0.5)
                if score < 0.3:
                    result['learning_insights'].append("벤담 점수 낮음 - 윤리적 재고려 필요")
                elif score > 0.7:
                    result['learning_insights'].append("벤담 점수 높음 - 긍정적 결과 예상")
            
            # 4. 대안 제시 (suggest_alternatives 활용)
            analysis_data = {
                'regret_level': result['regret_level'],
                'bentham_score': bentham_score,
                'counterfactuals': counterfactuals
            }
            result['suggested_alternatives'] = await self.suggest_alternatives(analysis_data)
            
            # 5. 신뢰도 계산
            result['confidence'] = 0.7 if counterfactuals else 0.3
            
            return result
            
        except Exception as e:
            logger.error(f"후회 분석 실패: {e}")
            return result
    
    async def suggest_alternatives(self, analysis_data: Dict[str, Any]) -> List[str]:
        """후회 분석 기반 대안 시나리오 제안
        
        MD 문서 사양: Phase 2에서 후회 분석으로 추가 시나리오 제안
        높은 후회 가능성이 있는 시나리오들을 찾아 대안 생성
        """
        alternatives = []
        
        try:
            # 현재 분석에서 후회 요소 추출
            bentham_score = analysis_data.get('bentham', {})
            emotion_data = analysis_data.get('emotion', {})
            
            # 후회 유형별 대안 생성
            
            # 1. 강도(intensity) 기반 대안
            if bentham_score.get('intensity', 0) < 0.5:
                alternatives.append("더 적극적인 개입을 통해 영향력을 높이는 방안")
            
            # 2. 지속성(duration) 기반 대안
            if bentham_score.get('duration', 0) < 0.3:
                alternatives.append("장기적 관점에서 지속 가능한 해결책 모색")
            
            # 3. 확실성(certainty) 기반 대안
            if bentham_score.get('certainty', 0) < 0.6:
                alternatives.append("더 많은 정보 수집 후 신중한 결정")
            
            # 4. 범위(extent) 기반 대안
            if bentham_score.get('extent', 0) < 0.4:
                alternatives.append("더 많은 이해관계자를 고려한 포괄적 접근")
            
            # 5. 감정 데이터 기반 대안
            if emotion_data:
                primary_emotion = emotion_data.get('primary_emotion', '')
                if 'fear' in str(primary_emotion).lower():
                    alternatives.append("두려움을 극복하고 용기 있는 선택 고려")
                elif 'anger' in str(primary_emotion).lower():
                    alternatives.append("감정을 가라앉히고 이성적 판단 추구")
                elif 'sadness' in str(primary_emotion).lower():
                    alternatives.append("긍정적 측면을 찾아 희망적 대안 모색")
            
            # 6. 과거 학습 기반 대안 (메모리에서)
            for phase, memories in self.regret_memories.items():
                if memories:
                    # 최근 유사 상황에서 낮은 후회를 보인 대안 찾기
                    low_regret_memories = [m for m in memories if m.intensity < 0.3]
                    if low_regret_memories:
                        latest = low_regret_memories[-1]
                        if hasattr(latest, 'alternative_action') and latest.alternative_action:
                            alternatives.append(f"과거 성공 사례 참고: {latest.alternative_action}")
            
            # 중복 제거 및 제한
            alternatives = list(dict.fromkeys(alternatives))[:6]  # 최대 6개
            
            # 대안이 없으면 기본 대안 제공
            if not alternatives:
                alternatives = [
                    "현재 선택을 재검토하고 다른 관점 고려",
                    "단계적 접근을 통한 위험 최소화",
                    "협력적 해결 방안 모색"
                ]
            
        except Exception as e:
            logger.error(f"대안 생성 실패: {e}")
            alternatives = ["대안 생성 중 오류 발생"]
        
        return alternatives

async def test_regret_learning_system():
    """후회 학습 시스템 테스트"""
    system = AdvancedRegretLearningSystem()
    
    # 테스트 시나리오 생성
    test_scenarios = []
    
    # Phase 0 학습을 위한 시나리오 (자기 감정 캘리브레이션)
    for i in range(60):  # 최소 학습 횟수보다 많게
        intensity = max(0.1, 0.8 - i * 0.01)  # 점진적으로 후회 감소
        scenario = {
            'situation': {
                'emotion_type': 'sadness',
                'situation_type': 'loss',
                'chosen_action': {'type': 'suppress_emotion'},
                'is_critical': i % 10 == 0  # 10번에 한 번은 중요한 상황
            },
            'outcome': {
                'negativity': intensity,
                'actual_value': 0.3,
                'empathy_failure': False
            },
            'alternatives': [
                {'action': 'express_emotion', 'expected_value': 0.7},
                {'action': 'seek_support', 'expected_value': 0.8}
            ],
            'literary_context': {
                'source': 'hamlet',
                'genre': 'tragedy'
            }
        }
        test_scenarios.append(scenario)
    
    # Phase 1 학습을 위한 시나리오 (타인 공감)
    for i in range(60):
        intensity = max(0.1, 0.7 - i * 0.008)
        scenario = {
            'situation': {
                'relationship_type': 'friend',
                'interaction_type': 'conflict',
                'chosen_action': {'type': 'misunderstand_other'},
                'is_critical': i % 15 == 0
            },
            'outcome': {
                'negativity': intensity,
                'actual_value': 0.4,
                'empathy_failure': True,
                'prediction_failure': True
            },
            'alternatives': [
                {'action': 'listen_actively', 'expected_value': 0.8},
                {'action': 'ask_clarification', 'expected_value': 0.75}
            ],
            'literary_context': {
                'source': 'pride_and_prejudice',
                'genre': 'romance'
            }
        }
        test_scenarios.append(scenario)
    
    # 시나리오 처리
    logger.info("=== 후회 학습 시스템 테스트 시작 ===")
    
    for idx, scenario in enumerate(test_scenarios):
        regret_memory = await system.process_regret(
            situation=scenario['situation'],
            outcome=scenario['outcome'],
            alternatives=scenario['alternatives'],
            literary_context=scenario['literary_context']
        )
        
        # 주기적 상태 체크
        if (idx + 1) % 20 == 0:
            report = await system.generate_regret_report()
            logger.info(f"\n--- 진행 상황 ({idx + 1}/{len(test_scenarios)}) ---")
            logger.info(f"현재 페이즈: {report['current_phase']}")
            logger.info(f"활성 페이즈: {report['active_phases']}")
            
            for phase, state in report['phase_states'].items():
                logger.info(f"{phase}: 이해도={state['understanding_level']:.3f}, "
                          f"평균후회={state['avg_recent_regret']:.3f}")
    
    # 최종 리포트
    final_report = await system.generate_regret_report()
    logger.info("\n=== 최종 학습 결과 ===")
    logger.info(json.dumps(final_report, indent=2, ensure_ascii=False))
    
    # 상태 저장
    await system.save_state()
    
    return system

if __name__ == "__main__":
    # 테스트 실행
    asyncio.run(test_regret_learning_system())