#!/usr/bin/env python3
"""
페이즈 컨트롤러 (Phase Controller)
학습/실행/반응 페이즈 명시적 분기 처리

docs/추가 조정.txt 개선사항 구현:
- 학습/실행 페이즈 간 경계 흐림 해결
- phase_controller.py 같은 모듈 추가해 명시적 분기 처리
- 페이즈별 loss function, reward function 분기 적용
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import logging
from enum import Enum
import threading
import queue
from collections import defaultdict, deque

class PhaseType(Enum):
    """페이즈 타입 정의"""
    LEARNING = "learning"           # 학습 페이즈
    EXECUTION = "execution"         # 실행 페이즈  
    REACTION = "reaction"           # 반응 페이즈
    REFLECTION = "reflection"       # 성찰 페이즈 (추가)
    ADAPTATION = "adaptation"       # 적응 페이즈 (추가)

@dataclass
class PhaseConfig:
    """페이즈별 설정"""
    phase_type: PhaseType
    
    # 손실 함수 설정
    loss_function_weights: Dict[str, float] = None
    reward_function_weights: Dict[str, float] = None
    
    # 최적화 설정
    learning_rate_multiplier: float = 1.0
    gradient_clipping: float = 1.0
    regularization_strength: float = 0.01
    
    # 모델 동작 설정
    enable_exploration: bool = True
    exploration_rate: float = 0.1
    confidence_threshold: float = 0.7
    uncertainty_tolerance: float = 0.3
    
    # 메모리 및 처리 설정
    max_context_length: int = 1024
    batch_processing: bool = True
    real_time_processing: bool = False
    
    def __post_init__(self):
        if self.loss_function_weights is None:
            self.loss_function_weights = self._get_default_loss_weights()
        if self.reward_function_weights is None:
            self.reward_function_weights = self._get_default_reward_weights()
    
    def _get_default_loss_weights(self) -> Dict[str, float]:
        """페이즈별 기본 손실 함수 가중치"""
        defaults = {
            PhaseType.LEARNING: {
                'classification_loss': 1.0,
                'regret_loss': 0.8,
                'ethics_loss': 0.9,
                'emotion_loss': 0.7,
                'consistency_loss': 0.5
            },
            PhaseType.EXECUTION: {
                'classification_loss': 0.3,
                'regret_loss': 1.2,
                'ethics_loss': 1.5,
                'emotion_loss': 1.0,
                'consistency_loss': 0.8
            },
            PhaseType.REACTION: {
                'classification_loss': 0.1,
                'regret_loss': 1.5,
                'ethics_loss': 1.2,
                'emotion_loss': 1.3,
                'consistency_loss': 1.0
            },
            PhaseType.REFLECTION: {
                'classification_loss': 0.2,
                'regret_loss': 1.0,
                'ethics_loss': 0.8,
                'emotion_loss': 0.9,
                'consistency_loss': 1.2
            },
            PhaseType.ADAPTATION: {
                'classification_loss': 0.6,
                'regret_loss': 1.1,
                'ethics_loss': 1.0,
                'emotion_loss': 0.8,
                'consistency_loss': 0.9
            }
        }
        return defaults.get(self.phase_type, defaults[PhaseType.LEARNING])
    
    def _get_default_reward_weights(self) -> Dict[str, float]:
        """페이즈별 기본 보상 함수 가중치"""
        defaults = {
            PhaseType.LEARNING: {
                'accuracy_reward': 1.0,
                'exploration_reward': 0.8,
                'ethics_alignment_reward': 0.7,
                'regret_minimization_reward': 0.6,
                'emotional_appropriateness_reward': 0.5
            },
            PhaseType.EXECUTION: {
                'accuracy_reward': 1.2,
                'exploration_reward': 0.3,
                'ethics_alignment_reward': 1.5,
                'regret_minimization_reward': 1.3,
                'emotional_appropriateness_reward': 1.0
            },
            PhaseType.REACTION: {
                'accuracy_reward': 0.8,
                'exploration_reward': 0.2,
                'ethics_alignment_reward': 1.0,
                'regret_minimization_reward': 1.5,
                'emotional_appropriateness_reward': 1.4
            },
            PhaseType.REFLECTION: {
                'accuracy_reward': 0.5,
                'exploration_reward': 0.1,
                'ethics_alignment_reward': 0.8,
                'regret_minimization_reward': 1.2,
                'emotional_appropriateness_reward': 1.0
            },
            PhaseType.ADAPTATION: {
                'accuracy_reward': 0.9,
                'exploration_reward': 1.0,
                'ethics_alignment_reward': 1.1,
                'regret_minimization_reward': 0.8,
                'emotional_appropriateness_reward': 0.7
            }
        }
        return defaults.get(self.phase_type, defaults[PhaseType.LEARNING])

class PhaseTransitionManager:
    """페이즈 전환 관리자"""
    
    def __init__(self):
        self.current_phase = PhaseType.LEARNING
        self.phase_history = deque(maxlen=100)
        self.transition_triggers = defaultdict(list)
        self.phase_statistics = defaultdict(lambda: defaultdict(int))
        
        # 기본 전환 규칙 설정
        self._setup_default_transition_rules()
    
    def _setup_default_transition_rules(self):
        """기본 전환 규칙 설정"""
        # 학습 → 실행 전환 조건
        self.add_transition_trigger(
            from_phase=PhaseType.LEARNING,
            to_phase=PhaseType.EXECUTION,
            condition=lambda metrics: metrics.get('training_accuracy', 0) > 0.8
        )
        
        # 실행 → 반응 전환 조건
        self.add_transition_trigger(
            from_phase=PhaseType.EXECUTION,
            to_phase=PhaseType.REACTION,
            condition=lambda metrics: metrics.get('regret_score', 0) > 0.6
        )
        
        # 반응 → 성찰 전환 조건
        self.add_transition_trigger(
            from_phase=PhaseType.REACTION,
            to_phase=PhaseType.REFLECTION,
            condition=lambda metrics: metrics.get('emotion_intensity', 0) > 0.7
        )
        
        # 성찰 → 적응 전환 조건
        self.add_transition_trigger(
            from_phase=PhaseType.REFLECTION,
            to_phase=PhaseType.ADAPTATION,
            condition=lambda metrics: metrics.get('insight_clarity', 0) > 0.5
        )
        
        # 적응 → 학습 전환 조건 (순환)
        self.add_transition_trigger(
            from_phase=PhaseType.ADAPTATION,
            to_phase=PhaseType.LEARNING,
            condition=lambda metrics: metrics.get('adaptation_complete', False)
        )
    
    def add_transition_trigger(self, from_phase: PhaseType, to_phase: PhaseType, 
                             condition: callable, priority: int = 1):
        """전환 트리거 추가"""
        self.transition_triggers[from_phase].append({
            'to_phase': to_phase,
            'condition': condition,
            'priority': priority
        })
        
        # 우선순위별 정렬
        self.transition_triggers[from_phase].sort(key=lambda x: x['priority'], reverse=True)
    
    def evaluate_transition(self, current_metrics: Dict[str, Any]) -> Optional[PhaseType]:
        """전환 평가 및 새 페이즈 반환"""
        current_triggers = self.transition_triggers.get(self.current_phase, [])
        
        for trigger in current_triggers:
            try:
                if trigger['condition'](current_metrics):
                    new_phase = trigger['to_phase']
                    self._record_transition(self.current_phase, new_phase, current_metrics)
                    return new_phase
            except Exception as e:
                logging.getLogger('PhaseTransitionManager').warning(
                    f"전환 조건 평가 오류: {e}"
                )
        
        return None
    
    def transition_to_phase(self, new_phase: PhaseType, metrics: Dict[str, Any] = None):
        """강제 페이즈 전환"""
        if metrics is None:
            metrics = {}
        
        self._record_transition(self.current_phase, new_phase, metrics)
        self.current_phase = new_phase
    
    def _record_transition(self, from_phase: PhaseType, to_phase: PhaseType, 
                          metrics: Dict[str, Any]):
        """전환 기록"""
        transition_record = {
            'from_phase': from_phase,
            'to_phase': to_phase,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics.copy(),
            'trigger_reason': self._identify_trigger_reason(from_phase, to_phase, metrics)
        }
        
        self.phase_history.append(transition_record)
        self.phase_statistics[from_phase.value][to_phase.value] += 1
        
        logging.getLogger('PhaseTransitionManager').info(
            f"페이즈 전환: {from_phase.value} → {to_phase.value}"
        )
    
    def _identify_trigger_reason(self, from_phase: PhaseType, to_phase: PhaseType,
                               metrics: Dict[str, Any]) -> str:
        """전환 원인 식별"""
        # 단순화된 원인 식별 로직
        key_metrics = []
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and value > 0.5:
                key_metrics.append(f"{key}={value:.2f}")
        
        return f"Triggered by: {', '.join(key_metrics[:3])}"
    
    def get_phase_statistics(self) -> Dict[str, Any]:
        """페이즈 통계 반환"""
        return {
            'current_phase': self.current_phase.value,
            'transition_counts': dict(self.phase_statistics),
            'recent_transitions': list(self.phase_history)[-10:],
            'phase_duration_stats': self._calculate_phase_durations()
        }
    
    def _calculate_phase_durations(self) -> Dict[str, float]:
        """페이즈별 지속 시간 통계"""
        if len(self.phase_history) < 2:
            return {}
        
        durations = defaultdict(list)
        
        for i in range(1, len(self.phase_history)):
            prev_transition = self.phase_history[i-1]
            curr_transition = self.phase_history[i]
            
            prev_time = datetime.fromisoformat(prev_transition['timestamp'])
            curr_time = datetime.fromisoformat(curr_transition['timestamp'])
            
            duration = (curr_time - prev_time).total_seconds()
            phase = prev_transition['to_phase'].value
            durations[phase].append(duration)
        
        # 평균 지속 시간 계산
        avg_durations = {}
        for phase, times in durations.items():
            avg_durations[phase] = sum(times) / len(times)
        
        return avg_durations

class PhaseLossFunction:
    """페이즈별 손실 함수"""
    
    def __init__(self, phase_config: PhaseConfig):
        self.phase_config = phase_config
        self.loss_weights = phase_config.loss_function_weights
        
    def compute_loss(self, predictions: Dict[str, torch.Tensor], 
                    targets: Dict[str, torch.Tensor],
                    additional_info: Dict[str, Any] = None) -> Dict[str, torch.Tensor]:
        """페이즈별 손실 계산"""
        losses = {}
        total_loss = torch.tensor(0.0, requires_grad=True)
        
        # 분류 손실
        if 'classification' in predictions and 'classification' in targets:
            cls_loss = F.cross_entropy(predictions['classification'], targets['classification'])
            losses['classification_loss'] = cls_loss
            total_loss = total_loss + cls_loss * self.loss_weights.get('classification_loss', 1.0)
        
        # 후회 손실
        if 'regret' in predictions and 'regret' in targets:
            regret_loss = F.mse_loss(predictions['regret'], targets['regret'])
            losses['regret_loss'] = regret_loss
            total_loss = total_loss + regret_loss * self.loss_weights.get('regret_loss', 1.0)
        
        # 윤리 손실
        if 'ethics' in predictions and 'ethics' in targets:
            ethics_loss = self._compute_ethics_loss(predictions['ethics'], targets['ethics'])
            losses['ethics_loss'] = ethics_loss
            total_loss = total_loss + ethics_loss * self.loss_weights.get('ethics_loss', 1.0)
        
        # 감정 손실
        if 'emotion' in predictions and 'emotion' in targets:
            emotion_loss = F.mse_loss(predictions['emotion'], targets['emotion'])
            losses['emotion_loss'] = emotion_loss
            total_loss = total_loss + emotion_loss * self.loss_weights.get('emotion_loss', 1.0)
        
        # 일관성 손실
        if additional_info and 'previous_predictions' in additional_info:
            consistency_loss = self._compute_consistency_loss(
                predictions, additional_info['previous_predictions']
            )
            losses['consistency_loss'] = consistency_loss
            total_loss = total_loss + consistency_loss * self.loss_weights.get('consistency_loss', 1.0)
        
        losses['total_loss'] = total_loss
        return losses
    
    def _compute_ethics_loss(self, pred_ethics: torch.Tensor, target_ethics: torch.Tensor) -> torch.Tensor:
        """윤리 손실 계산 (페이즈별 특화)"""
        base_loss = F.mse_loss(pred_ethics, target_ethics)
        
        # 실행 페이즈에서는 윤리적 일관성 강조
        if self.phase_config.phase_type == PhaseType.EXECUTION:
            # 극단값 패널티 (더 보수적 결정 선호)
            extreme_penalty = torch.mean(torch.abs(pred_ethics)) * 0.1
            base_loss = base_loss + extreme_penalty
        
        # 반응 페이즈에서는 감정-윤리 정렬 강조
        elif self.phase_config.phase_type == PhaseType.REACTION:
            # 감정과 윤리 판단 간 조화 검사 (예시)
            ethics_emotion_alignment = torch.mean(torch.abs(pred_ethics[:, 0] - pred_ethics[:, 1]))
            base_loss = base_loss + ethics_emotion_alignment * 0.05
        
        return base_loss
    
    def _compute_consistency_loss(self, current_pred: Dict[str, torch.Tensor],
                                previous_pred: Dict[str, torch.Tensor]) -> torch.Tensor:
        """일관성 손실 계산"""
        consistency_loss = torch.tensor(0.0)
        count = 0
        
        for key in current_pred:
            if key in previous_pred:
                # 예측 간 차이 패널티
                diff = F.mse_loss(current_pred[key], previous_pred[key])
                consistency_loss = consistency_loss + diff
                count += 1
        
        if count > 0:
            consistency_loss = consistency_loss / count
        
        return consistency_loss

class PhaseRewardFunction:
    """페이즈별 보상 함수"""
    
    def __init__(self, phase_config: PhaseConfig):
        self.phase_config = phase_config
        self.reward_weights = phase_config.reward_function_weights
    
    def compute_reward(self, predictions: Dict[str, torch.Tensor],
                      outcomes: Dict[str, Any],
                      context: Dict[str, Any] = None) -> Dict[str, float]:
        """페이즈별 보상 계산"""
        rewards = {}
        total_reward = 0.0
        
        # 정확도 보상
        if 'accuracy' in outcomes:
            accuracy_reward = float(outcomes['accuracy']) * self.reward_weights.get('accuracy_reward', 1.0)
            rewards['accuracy_reward'] = accuracy_reward
            total_reward += accuracy_reward
        
        # 탐색 보상 (학습 페이즈에서 중요)
        if self.phase_config.enable_exploration:
            exploration_reward = self._compute_exploration_reward(predictions, context)
            exploration_reward *= self.reward_weights.get('exploration_reward', 1.0)
            rewards['exploration_reward'] = exploration_reward
            total_reward += exploration_reward
        
        # 윤리 정렬 보상
        if 'ethics_alignment' in outcomes:
            ethics_reward = float(outcomes['ethics_alignment']) * self.reward_weights.get('ethics_alignment_reward', 1.0)
            rewards['ethics_alignment_reward'] = ethics_reward
            total_reward += ethics_reward
        
        # 후회 최소화 보상
        if 'regret_score' in outcomes:
            regret_reward = (1.0 - float(outcomes['regret_score'])) * self.reward_weights.get('regret_minimization_reward', 1.0)
            rewards['regret_minimization_reward'] = regret_reward
            total_reward += regret_reward
        
        # 감정적 적절성 보상
        if 'emotional_appropriateness' in outcomes:
            emotion_reward = float(outcomes['emotional_appropriateness']) * self.reward_weights.get('emotional_appropriateness_reward', 1.0)
            rewards['emotional_appropriateness_reward'] = emotion_reward
            total_reward += emotion_reward
        
        rewards['total_reward'] = total_reward
        return rewards
    
    def _compute_exploration_reward(self, predictions: Dict[str, torch.Tensor],
                                  context: Dict[str, Any] = None) -> float:
        """탐색 보상 계산"""
        exploration_reward = 0.0
        
        # 예측 분산을 탐색 지표로 사용
        for key, pred in predictions.items():
            if pred.numel() > 1:
                variance = torch.var(pred).item()
                exploration_reward += variance * 0.1
        
        # 불확실성이 높은 상황에서 탐색 보상 증가
        if context and 'uncertainty' in context:
            uncertainty_bonus = float(context['uncertainty']) * 0.2
            exploration_reward += uncertainty_bonus
        
        return min(exploration_reward, 1.0)  # 최대값 제한

class PhaseController:
    """메인 페이즈 컨트롤러"""
    
    def __init__(self):
        self.transition_manager = PhaseTransitionManager()
        self.phase_configs = {}
        self.loss_functions = {}
        self.reward_functions = {}
        
        # 기본 페이즈 설정 초기화
        self._initialize_default_phases()
        
        # 현재 활성 구성요소
        self.current_loss_function = None
        self.current_reward_function = None
        
        # 메트릭 추적
        self.phase_metrics = defaultdict(lambda: defaultdict(list))
        self.transition_history = []
        
        self.logger = logging.getLogger('PhaseController')
        self._update_active_components()
    
    def _initialize_default_phases(self):
        """기본 페이즈들 초기화"""
        for phase_type in PhaseType:
            config = PhaseConfig(phase_type=phase_type)
            self.phase_configs[phase_type] = config
            self.loss_functions[phase_type] = PhaseLossFunction(config)
            self.reward_functions[phase_type] = PhaseRewardFunction(config)
    
    def _update_active_components(self):
        """현재 페이즈에 맞는 활성 구성요소 업데이트"""
        current_phase = self.transition_manager.current_phase
        self.current_loss_function = self.loss_functions[current_phase]
        self.current_reward_function = self.reward_functions[current_phase]
        self.logger.debug(f"활성 구성요소 업데이트: {current_phase.value}")
    
    def update_phase_config(self, phase_type: PhaseType, config: PhaseConfig):
        """페이즈 설정 업데이트"""
        self.phase_configs[phase_type] = config
        self.loss_functions[phase_type] = PhaseLossFunction(config)
        self.reward_functions[phase_type] = PhaseRewardFunction(config)
        
        # 현재 페이즈라면 활성 구성요소 업데이트
        if phase_type == self.transition_manager.current_phase:
            self._update_active_components()
    
    def process_step(self, predictions: Dict[str, torch.Tensor],
                    targets: Dict[str, torch.Tensor],
                    outcomes: Dict[str, Any],
                    context: Dict[str, Any] = None) -> Dict[str, Any]:
        """단계 처리 - 페이즈별 손실/보상 계산 및 전환 평가"""
        
        # 현재 페이즈에서 손실 계산
        losses = self.current_loss_function.compute_loss(
            predictions, targets, context
        )
        
        # 현재 페이즈에서 보상 계산
        rewards = self.current_reward_function.compute_reward(
            predictions, outcomes, context
        )
        
        # 메트릭 준비
        step_metrics = {
            'total_loss': float(losses['total_loss']),
            'total_reward': rewards['total_reward'],
            'current_phase': self.transition_manager.current_phase.value
        }
        
        # 외부 메트릭 통합
        if context:
            step_metrics.update(context)
        if outcomes:
            step_metrics.update(outcomes)
        
        # 페이즈 전환 평가
        new_phase = self.transition_manager.evaluate_transition(step_metrics)
        if new_phase:
            self.transition_to_phase(new_phase, step_metrics)
        
        # 메트릭 기록
        current_phase = self.transition_manager.current_phase
        for key, value in step_metrics.items():
            if isinstance(value, (int, float)):
                self.phase_metrics[current_phase.value][key].append(value)
        
        return {
            'losses': losses,
            'rewards': rewards,
            'phase_info': {
                'current_phase': current_phase.value,
                'phase_changed': new_phase is not None,
                'new_phase': new_phase.value if new_phase else None
            },
            'metrics': step_metrics
        }
    
    def transition_to_phase(self, new_phase: PhaseType, metrics: Dict[str, Any] = None):
        """페이즈 전환 수행"""
        old_phase = self.transition_manager.current_phase
        self.transition_manager.transition_to_phase(new_phase, metrics)
        self._update_active_components()
        
        # 전환 이력 기록
        transition_record = {
            'from': old_phase.value,
            'to': new_phase.value,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics or {}
        }
        self.transition_history.append(transition_record)
        
        self.logger.info(f"페이즈 전환 완료: {old_phase.value} → {new_phase.value}")
    
    def get_current_phase(self) -> PhaseType:
        """현재 페이즈 반환"""
        return self.transition_manager.current_phase
    
    def get_phase_statistics(self) -> Dict[str, Any]:
        """페이즈 통계 반환"""
        base_stats = self.transition_manager.get_phase_statistics()
        
        # 페이즈별 성능 메트릭 추가
        phase_performance = {}
        for phase, metrics in self.phase_metrics.items():
            phase_performance[phase] = {}
            for metric_name, values in metrics.items():
                if values:
                    phase_performance[phase][metric_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'latest': values[-1],
                        'trend': np.polyfit(range(len(values)), values, 1)[0] if len(values) > 1 else 0
                    }
        
        base_stats['phase_performance'] = phase_performance
        base_stats['recent_transitions'] = self.transition_history[-10:]
        
        return base_stats
    
    def reset_phase_metrics(self):
        """페이즈 메트릭 초기화"""
        self.phase_metrics.clear()
        self.transition_history.clear()
        self.logger.info("페이즈 메트릭 초기화 완료")
    
    def save_phase_state(self, path: Path):
        """페이즈 상태 저장"""
        state_data = {
            'current_phase': self.transition_manager.current_phase.value,
            'phase_configs': {
                phase.value: asdict(config) 
                for phase, config in self.phase_configs.items()
            },
            'transition_history': self.transition_history,
            'phase_statistics': self.get_phase_statistics(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            import json
            json.dump(state_data, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"페이즈 상태 저장: {path}")
    
    def load_phase_state(self, path: Path):
        """페이즈 상태 로드"""
        with open(path, 'r', encoding='utf-8') as f:
            import json
            state_data = json.load(f)
        
        # 현재 페이즈 복원
        current_phase_value = state_data['current_phase']
        current_phase = PhaseType(current_phase_value)
        self.transition_manager.current_phase = current_phase
        
        # 전환 이력 복원
        self.transition_history = state_data.get('transition_history', [])
        
        # 활성 구성요소 업데이트
        self._update_active_components()
        
        self.logger.info(f"페이즈 상태 로드: {path}, 현재 페이즈: {current_phase.value}")

def create_phase_controller() -> PhaseController:
    """페이즈 컨트롤러 생성 헬퍼 함수"""
    return PhaseController()

# 사용 예시
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 페이즈 컨트롤러 생성
    controller = create_phase_controller()
    
    # 가상의 학습 루프
    for step in range(50):
        # 가상의 예측/타겟/결과
        predictions = {
            'classification': torch.randn(4, 10),
            'regret': torch.rand(4, 1),
            'ethics': torch.rand(4, 3),
            'emotion': torch.rand(4, 6)
        }
        
        targets = {
            'classification': torch.randint(0, 10, (4,)),
            'regret': torch.rand(4, 1),
            'ethics': torch.rand(4, 3), 
            'emotion': torch.rand(4, 6)
        }
        
        outcomes = {
            'accuracy': np.random.uniform(0.6, 0.9),
            'regret_score': np.random.uniform(0.2, 0.8),
            'ethics_alignment': np.random.uniform(0.5, 0.9),
            'emotional_appropriateness': np.random.uniform(0.4, 0.8)
        }
        
        context = {
            'step': step,
            'uncertainty': np.random.uniform(0.1, 0.6)
        }
        
        # 단계 처리
        result = controller.process_step(predictions, targets, outcomes, context)
        
        # 로그 출력 (가끔)
        if step % 10 == 0:
            current_phase = controller.get_current_phase()
            print(f"Step {step}: Phase = {current_phase.value}, "
                  f"Loss = {result['losses']['total_loss']:.3f}, "
                  f"Reward = {result['rewards']['total_reward']:.3f}")
    
    # 최종 통계 출력
    stats = controller.get_phase_statistics()
    print(f"\n=== 페이즈 통계 ===")
    print(f"현재 페이즈: {stats['current_phase']}")
    print(f"전환 횟수: {stats['transition_counts']}")
    print(f"최근 전환: {len(stats['recent_transitions'])}회")