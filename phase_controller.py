"""
페이즈 컨트롤러 (Phase Controller)
Phase Controller Module

학습(Learning), 실행(Execution), 반성(Reflection) 페이즈를 명시적으로 분기 처리하여
각 페이즈별 최적화된 정책과 파라미터를 적용하는 적응적 시스템을 구현합니다.

핵심 기능:
1. 페이즈별 차등 정책 적용
2. 동적 페이즈 전환 체크포인트
3. 페이즈별 성능 모니터링
4. 컨텍스트 기반 페이즈 선택
"""

import numpy as np
import torch
import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
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

logger = logging.getLogger('PhaseController')

class Phase(Enum):
    """시스템 페이즈 정의"""
    LEARNING = "learning"         # 학습 페이즈 - 탐색적, 실험적
    EXECUTION = "execution"       # 실행 페이즈 - 보수적, 안정적
    REFLECTION = "reflection"     # 반성 페이즈 - 분석적, 개선적
    TRANSITION = "transition"     # 전환 페이즈 - 페이즈 간 전환

@dataclass
class PhaseConfig:
    """페이즈별 설정"""
    phase: Phase
    
    # 탐색/활용 균형
    exploration_rate: float = 0.5
    exploitation_rate: float = 0.5
    
    # 안전성 임계값
    safety_threshold: float = 0.5
    risk_tolerance: float = 0.5
    
    # 학습 파라미터
    learning_rate: float = 0.01
    memory_weight: float = 0.5
    
    # 의사결정 파라미터
    confidence_threshold: float = 0.7
    consensus_requirement: float = 0.6
    
    # 시간 제약
    max_processing_time: float = 5.0
    urgency_factor: float = 1.0
    
    # 피드백 감도
    regret_sensitivity: float = 0.5
    reward_sensitivity: float = 0.5
    
    # 윤리적 엄격성
    ethical_strictness: float = 0.7
    moral_flexibility: float = 0.3

@dataclass
class PhaseTransitionCriteria:
    """페이즈 전환 기준"""
    
    # 성능 기반 전환
    performance_threshold: float = 0.8
    consistency_requirement: float = 0.7
    improvement_rate: float = 0.05
    
    # 시간 기반 전환
    min_phase_duration: float = 300.0  # 5분
    max_phase_duration: float = 3600.0  # 1시간
    
    # 경험 기반 전환
    min_experiences: int = 10
    experience_quality_threshold: float = 0.7
    
    # 환경 기반 전환
    stability_indicator: float = 0.8
    novelty_threshold: float = 0.3
    
    # 사용자 피드백 기반
    user_satisfaction_threshold: float = 0.8
    intervention_count_limit: int = 3

@dataclass
class PhaseState:
    """현재 페이즈 상태"""
    current_phase: Phase
    phase_start_time: float
    phase_duration: float = 0.0
    
    # 페이즈 성능 지표
    success_rate: float = 0.0
    average_confidence: float = 0.5
    error_rate: float = 0.0
    user_satisfaction: float = 0.5
    
    # 전환 준비도
    transition_readiness: float = 0.0
    next_recommended_phase: Optional[Phase] = None
    
    # 메타데이터
    decision_count: int = 0
    last_update_time: float = field(default_factory=time.time)
    phase_history: List[Tuple[Phase, float]] = field(default_factory=list)

@dataclass
class PhaseDecisionContext:
    """페이즈별 의사결정 맥락"""
    scenario_complexity: float = 0.5
    uncertainty_level: float = 0.5
    time_pressure: float = 0.5
    stakeholder_count: int = 1
    ethical_weight: float = 0.7
    
    # 과거 경험
    similar_scenarios: int = 0
    past_success_rate: float = 0.5
    
    # 현재 상황
    available_information: float = 0.8
    user_confidence: float = 0.7
    external_pressure: float = 0.3

class PhaseController:
    """페이즈 컨트롤러"""
    
    def __init__(self):
        self.logger = logger
        
        # 현재 상태
        self.current_state = PhaseState(
            current_phase=Phase.LEARNING,
            phase_start_time=time.time()
        )
        
        # 페이즈별 설정
        self.phase_configs = self._initialize_phase_configs()
        
        # 전환 기준
        self.transition_criteria = PhaseTransitionCriteria()
        
        # 성능 추적
        self.performance_history = {
            Phase.LEARNING: [],
            Phase.EXECUTION: [],
            Phase.REFLECTION: []
        }
        
        # 페이즈별 통계
        self.phase_statistics = {
            phase: {
                'total_time': 0.0,
                'total_decisions': 0,
                'success_count': 0,
                'average_performance': 0.0
            } for phase in Phase
        }
        
        # 동적 조정 파라미터
        self.adaptation_memory = {
            'successful_transitions': [],
            'failed_transitions': [],
            'optimal_phase_durations': {}
        }
        
        # 스레드 안전성
        self.state_lock = threading.Lock()
        
        self.logger.info("페이즈 컨트롤러 초기화 완료")
    
    def _initialize_phase_configs(self) -> Dict[Phase, PhaseConfig]:
        """페이즈별 설정 초기화"""
        return {
            Phase.LEARNING: PhaseConfig(
                phase=Phase.LEARNING,
                exploration_rate=0.8,      # 높은 탐색
                exploitation_rate=0.2,     # 낮은 활용
                safety_threshold=0.3,      # 낮은 안전 임계값
                risk_tolerance=0.8,        # 높은 위험 허용
                learning_rate=0.05,        # 높은 학습률
                memory_weight=0.3,         # 낮은 기억 가중치
                confidence_threshold=0.5,  # 낮은 신뢰도 요구
                regret_sensitivity=0.9,    # 높은 후회 감도
                ethical_strictness=0.6     # 중간 윤리 엄격성
            ),
            
            Phase.EXECUTION: PhaseConfig(
                phase=Phase.EXECUTION,
                exploration_rate=0.1,      # 낮은 탐색
                exploitation_rate=0.9,     # 높은 활용
                safety_threshold=0.8,      # 높은 안전 임계값
                risk_tolerance=0.2,        # 낮은 위험 허용
                learning_rate=0.01,        # 낮은 학습률
                memory_weight=0.8,         # 높은 기억 가중치
                confidence_threshold=0.8,  # 높은 신뢰도 요구
                regret_sensitivity=0.3,    # 낮은 후회 감도
                ethical_strictness=0.9     # 높은 윤리 엄격성
            ),
            
            Phase.REFLECTION: PhaseConfig(
                phase=Phase.REFLECTION,
                exploration_rate=0.5,      # 중간 탐색
                exploitation_rate=0.5,     # 중간 활용
                safety_threshold=0.6,      # 중간 안전 임계값
                risk_tolerance=0.4,        # 중간 위험 허용
                learning_rate=0.03,        # 중간 학습률
                memory_weight=0.9,         # 높은 기억 가중치
                confidence_threshold=0.6,  # 중간 신뢰도 요구
                regret_sensitivity=0.7,    # 높은 후회 감도
                ethical_strictness=0.8     # 높은 윤리 엄격성
            )
        }
    
    def get_current_phase_config(self) -> PhaseConfig:
        """현재 페이즈 설정 반환"""
        with self.state_lock:
            return self.phase_configs[self.current_state.current_phase]
    
    def determine_optimal_phase(self, context: PhaseDecisionContext) -> Phase:
        """맥락을 기반으로 최적 페이즈 결정"""
        
        # 페이즈별 적합도 점수 계산
        phase_scores = {}
        
        # 학습 페이즈 적합도
        learning_score = self._calculate_learning_phase_score(context)
        phase_scores[Phase.LEARNING] = learning_score
        
        # 실행 페이즈 적합도
        execution_score = self._calculate_execution_phase_score(context)
        phase_scores[Phase.EXECUTION] = execution_score
        
        # 반성 페이즈 적합도
        reflection_score = self._calculate_reflection_phase_score(context)
        phase_scores[Phase.REFLECTION] = reflection_score
        
        # 최고 점수 페이즈 선택
        optimal_phase = max(phase_scores.keys(), key=lambda p: phase_scores[p])
        
        self.logger.debug(
            f"페이즈 적합도 점수: "
            f"학습={learning_score:.3f}, "
            f"실행={execution_score:.3f}, "
            f"반성={reflection_score:.3f} "
            f"-> 선택: {optimal_phase.value}"
        )
        
        return optimal_phase
    
    def _calculate_learning_phase_score(self, context: PhaseDecisionContext) -> float:
        """학습 페이즈 적합도 점수 계산"""
        score = 0.0
        
        # 높은 불확실성 -> 학습 페이즈 선호
        score += context.uncertainty_level * 0.3
        
        # 낮은 과거 경험 -> 학습 페이즈 선호
        if context.similar_scenarios < 5:
            score += 0.3
        else:
            score += max(0.0, 0.3 - (context.similar_scenarios - 5) * 0.05)
        
        # 복잡한 시나리오 -> 학습 페이즈 선호
        score += context.scenario_complexity * 0.2
        
        # 낮은 시간 압박 -> 학습 페이즈 선호
        score += (1.0 - context.time_pressure) * 0.2
        
        return np.clip(score, 0.0, 1.0)
    
    def _calculate_execution_phase_score(self, context: PhaseDecisionContext) -> float:
        """실행 페이즈 적합도 점수 계산"""
        score = 0.0
        
        # 높은 확실성 -> 실행 페이즈 선호
        score += (1.0 - context.uncertainty_level) * 0.3
        
        # 많은 과거 경험 -> 실행 페이즈 선호
        if context.similar_scenarios >= 10:
            score += 0.3
        elif context.similar_scenarios >= 5:
            score += 0.2
        
        # 높은 과거 성공률 -> 실행 페이즈 선호
        score += context.past_success_rate * 0.2
        
        # 높은 시간 압박 -> 실행 페이즈 선호
        score += context.time_pressure * 0.2
        
        return np.clip(score, 0.0, 1.0)
    
    def _calculate_reflection_phase_score(self, context: PhaseDecisionContext) -> float:
        """반성 페이즈 적합도 점수 계산"""
        score = 0.0
        
        # 높은 윤리적 가중치 -> 반성 페이즈 선호
        score += context.ethical_weight * 0.3
        
        # 중간 정도의 경험과 불확실성 -> 반성 페이즈 선호
        if 3 <= context.similar_scenarios <= 8:
            score += 0.2
        
        if 0.3 <= context.uncertainty_level <= 0.7:
            score += 0.2
        
        # 많은 이해관계자 -> 반성 페이즈 선호
        stakeholder_factor = min(context.stakeholder_count / 10.0, 1.0)
        score += stakeholder_factor * 0.3
        
        return np.clip(score, 0.0, 1.0)
    
    def check_phase_transition_needed(self) -> Tuple[bool, Optional[Phase]]:
        """페이즈 전환 필요성 체크"""
        
        with self.state_lock:
            current_time = time.time()
            phase_duration = current_time - self.current_state.phase_start_time
            
            # 시간 기반 전환 체크
            if phase_duration > self.transition_criteria.max_phase_duration:
                return True, self._suggest_next_phase()
            
            # 성능 기반 전환 체크
            if self.current_state.success_rate < (1.0 - self.transition_criteria.performance_threshold):
                if phase_duration > self.transition_criteria.min_phase_duration:
                    return True, self._suggest_next_phase()
            
            # 사용자 만족도 기반 전환 체크
            if self.current_state.user_satisfaction < self.transition_criteria.user_satisfaction_threshold:
                if phase_duration > self.transition_criteria.min_phase_duration:
                    return True, self._suggest_next_phase()
            
            # 경험 충분성 체크
            if (self.current_state.current_phase == Phase.LEARNING and 
                self.current_state.decision_count >= self.transition_criteria.min_experiences):
                if self.current_state.success_rate > self.transition_criteria.performance_threshold:
                    return True, Phase.EXECUTION
            
            return False, None
    
    def _suggest_next_phase(self) -> Phase:
        """다음 페이즈 제안"""
        current = self.current_state.current_phase
        
        # 순환적 페이즈 전환 로직
        if current == Phase.LEARNING:
            # 학습에서 실행으로 (충분한 학습 후)
            if self.current_state.success_rate > 0.7:
                return Phase.EXECUTION
            else:
                return Phase.REFLECTION  # 성과가 부족하면 반성
        
        elif current == Phase.EXECUTION:
            # 실행에서 반성으로 (성과 검토)
            if self.current_state.success_rate < 0.6:
                return Phase.REFLECTION
            else:
                return Phase.LEARNING  # 새로운 학습 기회 탐색
        
        elif current == Phase.REFLECTION:
            # 반성에서 학습으로 (개선 방향 도출 후)
            return Phase.LEARNING
        
        return Phase.LEARNING  # 기본값
    
    def transition_to_phase(self, target_phase: Phase, reason: str = "manual") -> bool:
        """페이즈 전환 실행"""
        
        with self.state_lock:
            old_phase = self.current_state.current_phase
            
            if old_phase == target_phase:
                self.logger.debug(f"이미 {target_phase.value} 페이즈입니다.")
                return False
            
            # 현재 페이즈 통계 업데이트
            self._update_phase_statistics(old_phase)
            
            # 페이즈 전환 실행
            self.current_state.current_phase = target_phase
            self.current_state.phase_start_time = time.time()
            self.current_state.phase_duration = 0.0
            
            # 페이즈 히스토리 업데이트
            self.current_state.phase_history.append((old_phase, time.time()))
            
            # 성능 지표 초기화
            self.current_state.success_rate = 0.0
            self.current_state.average_confidence = 0.5
            self.current_state.error_rate = 0.0
            self.current_state.decision_count = 0
            
            self.logger.info(
                f"페이즈 전환: {old_phase.value} -> {target_phase.value} (이유: {reason})"
            )
            
            return True
    
    def apply_phase_policy(
        self, 
        base_decision_params: Dict[str, Any],
        context: PhaseDecisionContext
    ) -> Dict[str, Any]:
        """페이즈별 정책을 기본 의사결정 파라미터에 적용"""
        
        current_config = self.get_current_phase_config()
        modified_params = base_decision_params.copy()
        
        # 탐색/활용 균형 조정
        if 'exploration_weight' in modified_params:
            modified_params['exploration_weight'] *= current_config.exploration_rate
        
        if 'exploitation_weight' in modified_params:
            modified_params['exploitation_weight'] *= current_config.exploitation_rate
        
        # 안전성 임계값 조정
        if 'safety_threshold' in modified_params:
            modified_params['safety_threshold'] = max(
                modified_params['safety_threshold'],
                current_config.safety_threshold
            )
        
        # 학습률 조정
        if 'learning_rate' in modified_params:
            modified_params['learning_rate'] *= current_config.learning_rate / 0.01  # 정규화
        
        # 신뢰도 임계값 조정
        if 'confidence_threshold' in modified_params:
            modified_params['confidence_threshold'] = max(
                modified_params['confidence_threshold'],
                current_config.confidence_threshold
            )
        
        # 윤리적 엄격성 조정
        if 'ethical_strictness' in modified_params:
            modified_params['ethical_strictness'] *= current_config.ethical_strictness
        
        # 페이즈별 특별 조정
        self._apply_phase_specific_adjustments(modified_params, current_config, context)
        
        return modified_params
    
    def _apply_phase_specific_adjustments(
        self,
        params: Dict[str, Any],
        config: PhaseConfig,
        context: PhaseDecisionContext
    ):
        """페이즈별 특별 조정"""
        
        if config.phase == Phase.LEARNING:
            # 학습 페이즈: 실험적, 탐색적 조정
            if 'risk_tolerance' in params:
                params['risk_tolerance'] *= 1.5  # 위험 허용도 증가
            
            if 'novelty_bonus' in params:
                params['novelty_bonus'] *= 2.0  # 새로운 시도 보상 증가
        
        elif config.phase == Phase.EXECUTION:
            # 실행 페이즈: 안정적, 보수적 조정
            if 'uncertainty_penalty' in params:
                params['uncertainty_penalty'] *= 1.5  # 불확실성 페널티 증가
            
            if 'consistency_weight' in params:
                params['consistency_weight'] *= 1.3  # 일관성 가중치 증가
        
        elif config.phase == Phase.REFLECTION:
            # 반성 페이즈: 신중하고 분석적 조정
            if 'analysis_depth' in params:
                params['analysis_depth'] *= 1.4  # 분석 깊이 증가
            
            if 'stakeholder_consideration' in params:
                params['stakeholder_consideration'] *= 1.2  # 이해관계자 고려 증가
    
    def record_decision_outcome(
        self,
        decision_success: bool,
        confidence: float,
        user_satisfaction: float,
        processing_time: float
    ):
        """의사결정 결과 기록"""
        
        with self.state_lock:
            self.current_state.decision_count += 1
            
            # 성공률 업데이트
            current_success = self.current_state.success_rate
            decision_weight = 1.0 / self.current_state.decision_count
            
            if decision_success:
                new_success_rate = current_success + (1.0 - current_success) * decision_weight
            else:
                new_success_rate = current_success * (1.0 - decision_weight)
            
            self.current_state.success_rate = new_success_rate
            
            # 평균 신뢰도 업데이트
            current_conf = self.current_state.average_confidence
            new_conf = current_conf + (confidence - current_conf) * decision_weight
            self.current_state.average_confidence = new_conf
            
            # 사용자 만족도 업데이트
            current_sat = self.current_state.user_satisfaction
            new_sat = current_sat + (user_satisfaction - current_sat) * decision_weight
            self.current_state.user_satisfaction = new_sat
            
            # 상태 업데이트 시간 기록
            self.current_state.last_update_time = time.time()
            
            # 성능 히스토리에 추가
            performance_record = {
                'timestamp': time.time(),
                'success': decision_success,
                'confidence': confidence,
                'user_satisfaction': user_satisfaction,
                'processing_time': processing_time
            }
            
            current_phase = self.current_state.current_phase
            self.performance_history[current_phase].append(performance_record)
            
            # 히스토리 크기 제한
            if len(self.performance_history[current_phase]) > 100:
                self.performance_history[current_phase] = self.performance_history[current_phase][-100:]
    
    def _update_phase_statistics(self, phase: Phase):
        """페이즈 통계 업데이트"""
        
        if phase in self.performance_history and self.performance_history[phase]:
            recent_records = self.performance_history[phase][-50:]  # 최근 50개
            
            # 평균 성능 계산
            if recent_records:
                avg_performance = np.mean([r['success'] for r in recent_records])
                self.phase_statistics[phase]['average_performance'] = avg_performance
                
                total_decisions = self.phase_statistics[phase]['total_decisions']
                success_count = sum(r['success'] for r in recent_records)
                
                self.phase_statistics[phase]['total_decisions'] = total_decisions + len(recent_records)
                self.phase_statistics[phase]['success_count'] += success_count
        
        # 페이즈 시간 누적
        phase_duration = time.time() - self.current_state.phase_start_time
        self.phase_statistics[phase]['total_time'] += phase_duration
    
    def get_analytics(self) -> Dict[str, Any]:
        """분석 정보 반환 (호환성을 위한 별칭)"""
        return self.get_phase_analytics()
    
    def get_phase_analytics(self) -> Dict[str, Any]:
        """페이즈 분석 정보 반환"""
        
        with self.state_lock:
            current_time = time.time()
            current_duration = current_time - self.current_state.phase_start_time
            
            return {
                'current_state': {
                    'phase': self.current_state.current_phase.value,
                    'duration': current_duration,
                    'success_rate': self.current_state.success_rate,
                    'average_confidence': self.current_state.average_confidence,
                    'user_satisfaction': self.current_state.user_satisfaction,
                    'decision_count': self.current_state.decision_count
                },
                'phase_statistics': {
                    phase.value: stats for phase, stats in self.phase_statistics.items()
                },
                'transition_readiness': self.current_state.transition_readiness,
                'recent_transitions': self.current_state.phase_history[-5:],
                'performance_trends': self._calculate_performance_trends()
            }
    
    def _calculate_performance_trends(self) -> Dict[str, float]:
        """성능 트렌드 계산"""
        trends = {}
        
        for phase, records in self.performance_history.items():
            if len(records) >= 10:
                recent_10 = records[-10:]
                previous_10 = records[-20:-10] if len(records) >= 20 else records[:-10]
                
                recent_avg = np.mean([r['success'] for r in recent_10])
                previous_avg = np.mean([r['success'] for r in previous_10]) if previous_10 else recent_avg
                
                trend = recent_avg - previous_avg
                trends[phase.value] = trend
            else:
                trends[phase.value] = 0.0
        
        return trends
    
    def save_phase_state(self, filepath: str):
        """페이즈 상태 저장"""
        
        state_data = {
            'current_state': {
                'current_phase': self.current_state.current_phase.value,
                'phase_start_time': self.current_state.phase_start_time,
                'success_rate': self.current_state.success_rate,
                'average_confidence': self.current_state.average_confidence,
                'user_satisfaction': self.current_state.user_satisfaction,
                'decision_count': self.current_state.decision_count,
                'phase_history': [(p.value, t) for p, t in self.current_state.phase_history]
            },
            'phase_statistics': {
                phase.value: stats for phase, stats in self.phase_statistics.items()
            },
            'performance_history_summary': {
                phase.value: len(records) for phase, records in self.performance_history.items()
            },
            'analytics': self.get_phase_analytics()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"페이즈 상태를 {filepath}에 저장 완료")


# 자동 페이즈 관리자
class AutoPhaseManager:
    """자동 페이즈 관리자 - 주기적으로 페이즈 최적화"""
    
    def __init__(self, phase_controller: PhaseController, check_interval: float = 60.0):
        self.phase_controller = phase_controller
        self.check_interval = check_interval
        self.running = False
        self.thread = None
        self.logger = logging.getLogger('AutoPhaseManager')
    
    def start(self):
        """자동 페이즈 관리 시작"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._management_loop, daemon=True)
        self.thread.start()
        self.logger.info("자동 페이즈 관리 시작")
    
    def stop(self):
        """자동 페이즈 관리 중지"""
        self.running = False
        if self.thread:
            self.thread.join()
        self.logger.info("자동 페이즈 관리 중지")
    
    def _management_loop(self):
        """관리 루프"""
        while self.running:
            try:
                # 페이즈 전환 필요성 체크
                needs_transition, suggested_phase = self.phase_controller.check_phase_transition_needed()
                
                if needs_transition and suggested_phase:
                    success = self.phase_controller.transition_to_phase(
                        suggested_phase, 
                        reason="auto_optimization"
                    )
                    
                    if success:
                        self.logger.info(f"자동 페이즈 전환 완료: {suggested_phase.value}")
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"자동 페이즈 관리 오류: {e}")
                time.sleep(self.check_interval)


# 테스트 및 데모 함수
def test_phase_controller():
    """페이즈 컨트롤러 테스트"""
    print("🎛️ 페이즈 컨트롤러 테스트 시작")
    
    # 컨트롤러 초기화
    controller = PhaseController()
    
    # 테스트 시나리오
    test_contexts = [
        PhaseDecisionContext(
            scenario_complexity=0.8,
            uncertainty_level=0.9,
            time_pressure=0.2,
            stakeholder_count=2,
            similar_scenarios=1,
            past_success_rate=0.3
        ),
        PhaseDecisionContext(
            scenario_complexity=0.4,
            uncertainty_level=0.2,
            time_pressure=0.8,
            stakeholder_count=1,
            similar_scenarios=15,
            past_success_rate=0.9
        ),
        PhaseDecisionContext(
            scenario_complexity=0.6,
            uncertainty_level=0.5,
            time_pressure=0.4,
            stakeholder_count=8,
            ethical_weight=0.9,
            similar_scenarios=5,
            past_success_rate=0.6
        )
    ]
    
    # 각 맥락에 대한 최적 페이즈 결정
    for i, context in enumerate(test_contexts, 1):
        print(f"\n--- 시나리오 {i} ---")
        print(f"복잡도: {context.scenario_complexity:.2f}, "
              f"불확실성: {context.uncertainty_level:.2f}, "
              f"시간압박: {context.time_pressure:.2f}")
        
        # 최적 페이즈 결정
        optimal_phase = controller.determine_optimal_phase(context)
        print(f"🎯 최적 페이즈: {optimal_phase.value}")
        
        # 페이즈 전환
        controller.transition_to_phase(optimal_phase, f"scenario_{i}")
        
        # 페이즈별 정책 적용
        base_params = {
            'exploration_weight': 0.5,
            'safety_threshold': 0.6,
            'learning_rate': 0.01,
            'confidence_threshold': 0.7,
            'ethical_strictness': 0.8
        }
        
        modified_params = controller.apply_phase_policy(base_params, context)
        
        print(f"📊 조정된 파라미터:")
        for key, value in modified_params.items():
            original = base_params.get(key, 0)
            change = value - original
            print(f"  {key}: {original:.3f} → {value:.3f} ({change:+.3f})")
        
        # 가상의 의사결정 결과 기록
        import random
        success = random.random() > 0.3
        confidence = random.uniform(0.5, 0.9)
        satisfaction = random.uniform(0.4, 0.9)
        
        controller.record_decision_outcome(success, confidence, satisfaction, 2.5)
        
        print(f"🔄 결과 기록: 성공={success}, 신뢰도={confidence:.3f}, 만족도={satisfaction:.3f}")
    
    # 페이즈 분석 정보
    analytics = controller.get_phase_analytics()
    print(f"\n📈 페이즈 분석:")
    print(f"현재 페이즈: {analytics['current_state']['phase']}")
    print(f"성공률: {analytics['current_state']['success_rate']:.3f}")
    print(f"평균 신뢰도: {analytics['current_state']['average_confidence']:.3f}")
    print(f"사용자 만족도: {analytics['current_state']['user_satisfaction']:.3f}")
    
    # 자동 페이즈 관리자 테스트
    print(f"\n🤖 자동 페이즈 관리자 테스트")
    auto_manager = AutoPhaseManager(controller, check_interval=1.0)
    auto_manager.start()
    
    # 잠시 대기 후 중지
    time.sleep(3)
    auto_manager.stop()
    
    print("✅ 페이즈 컨트롤러 테스트 완료")
    
    return controller


if __name__ == "__main__":
    test_phase_controller()
