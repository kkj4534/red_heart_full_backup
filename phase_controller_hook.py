"""
PhaseController Hook System for Red Heart AI
예측 오차 기반 자동 파인튜닝 시스템

핵심 기능:
1. 예측 오차 모니터링
2. 성능 임계값 기반 자동 파인튜닝
3. 적응형 학습률 조정
4. 모델 성능 추적 및 최적화
5. 오류 패턴 분석 및 대응
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from datetime import datetime, timedelta
from collections import deque, defaultdict
import threading
import asyncio
from pathlib import Path

logger = logging.getLogger('RedHeart.PhaseControllerHook')

class PhaseType(Enum):
    """학습 단계 유형"""
    TRAINING = "training"
    VALIDATION = "validation"
    INFERENCE = "inference"
    FINE_TUNING = "fine_tuning"
    EVALUATION = "evaluation"

class PerformanceMetric(Enum):
    """성능 지표"""
    ACCURACY = "accuracy"
    LOSS = "loss"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    REGRET_PREDICTION_ERROR = "regret_prediction_error"
    BENTHAM_CALCULATION_ERROR = "bentham_calculation_error"
    ETHICS_CLASSIFICATION_ERROR = "ethics_classification_error"

@dataclass
class PerformanceSnapshot:
    """성능 스냅샷"""
    timestamp: datetime
    phase_type: PhaseType
    metrics: Dict[str, float]
    
    # 모델별 성능
    model_performances: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # 오차 분석
    error_patterns: Dict[str, List[float]] = field(default_factory=dict)
    prediction_errors: List[float] = field(default_factory=list)
    
    # 시스템 상태
    memory_usage: float = 0.0
    gpu_utilization: float = 0.0
    processing_time: float = 0.0
    
    # 메타데이터
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TuningAction:
    """튜닝 액션"""
    action_type: str
    target_model: str
    parameters: Dict[str, Any]
    priority: int
    timestamp: datetime
    
    # 예상 효과
    expected_improvement: float = 0.0
    confidence: float = 0.5
    
    # 실행 상태
    executed: bool = False
    execution_time: Optional[datetime] = None
    actual_improvement: Optional[float] = None

class ErrorPatternAnalyzer:
    """오류 패턴 분석기"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.error_history = deque(maxlen=window_size)
        self.pattern_cache = {}
        
    def add_error(self, error_value: float, context: Dict[str, Any]):
        """오류 추가"""
        self.error_history.append({
            'error': error_value,
            'timestamp': datetime.now(),
            'context': context
        })
        
        # 패턴 캐시 무효화
        self.pattern_cache.clear()
    
    def detect_patterns(self) -> Dict[str, Any]:
        """오류 패턴 탐지"""
        if len(self.error_history) < 10:
            return {}
        
        cache_key = f"patterns_{len(self.error_history)}"
        if cache_key in self.pattern_cache:
            return self.pattern_cache[cache_key]
        
        errors = [entry['error'] for entry in self.error_history]
        contexts = [entry['context'] for entry in self.error_history]
        
        patterns = {}
        
        # 1. 추세 분석
        if len(errors) >= 20:
            recent_errors = errors[-20:]
            older_errors = errors[-40:-20] if len(errors) >= 40 else errors[:-20]
            
            recent_avg = np.mean(recent_errors)
            older_avg = np.mean(older_errors)
            
            patterns['trend'] = {
                'direction': 'increasing' if recent_avg > older_avg else 'decreasing',
                'magnitude': abs(recent_avg - older_avg),
                'significance': abs(recent_avg - older_avg) / (older_avg + 1e-8)
            }
        
        # 2. 분산 분석
        error_std = np.std(errors)
        error_mean = np.mean(errors)
        patterns['stability'] = {
            'coefficient_of_variation': error_std / (error_mean + 1e-8),
            'is_stable': error_std < error_mean * 0.2
        }
        
        # 3. 이상치 탐지
        q75, q25 = np.percentile(errors, [75, 25])
        iqr = q75 - q25
        outlier_threshold = q75 + 1.5 * iqr
        
        outliers = [e for e in errors if e > outlier_threshold]
        patterns['outliers'] = {
            'count': len(outliers),
            'percentage': len(outliers) / len(errors) * 100,
            'threshold': outlier_threshold
        }
        
        # 4. 주기성 분석
        if len(errors) >= 50:
            # 단순 자기상관 분석
            autocorr = np.corrcoef(errors[:-1], errors[1:])[0, 1]
            patterns['autocorrelation'] = {
                'lag_1': autocorr,
                'has_pattern': abs(autocorr) > 0.3
            }
        
        # 5. 컨텍스트 기반 패턴
        context_patterns = self._analyze_context_patterns(contexts, errors)
        patterns['context_patterns'] = context_patterns
        
        self.pattern_cache[cache_key] = patterns
        return patterns
    
    def _analyze_context_patterns(self, contexts: List[Dict[str, Any]], 
                                 errors: List[float]) -> Dict[str, Any]:
        """컨텍스트 기반 패턴 분석"""
        
        # 컨텍스트 키별 오류 분포
        context_errors = defaultdict(list)
        
        for context, error in zip(contexts, errors):
            for key, value in context.items():
                if isinstance(value, (int, float, bool)):
                    context_errors[key].append((value, error))
                elif isinstance(value, str):
                    context_errors[f"{key}_{value}"].append((1.0, error))
        
        patterns = {}
        
        for context_key, value_error_pairs in context_errors.items():
            if len(value_error_pairs) >= 5:
                values = [pair[0] for pair in value_error_pairs]
                errors = [pair[1] for pair in value_error_pairs]
                
                # 상관관계 분석
                if len(set(values)) > 1:  # 값이 다양해야 상관관계 분석 가능
                    correlation = np.corrcoef(values, errors)[0, 1]
                    if abs(correlation) > 0.3:
                        patterns[context_key] = {
                            'correlation': correlation,
                            'impact': 'high' if abs(correlation) > 0.6 else 'moderate',
                            'avg_error': np.mean(errors)
                        }
        
        return patterns

class AdaptiveLearningRateScheduler:
    """적응형 학습률 스케줄러"""
    
    def __init__(self, initial_lr: float = 1e-4, 
                 patience: int = 10, 
                 factor: float = 0.5,
                 min_lr: float = 1e-6):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        
        self.best_metric = float('inf')
        self.patience_counter = 0
        self.lr_history = []
        
    def step(self, current_metric: float) -> float:
        """학습률 업데이트"""
        
        if current_metric < self.best_metric:
            self.best_metric = current_metric
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
            if self.patience_counter >= self.patience:
                # 학습률 감소
                new_lr = max(self.current_lr * self.factor, self.min_lr)
                
                if new_lr < self.current_lr:
                    self.current_lr = new_lr
                    self.patience_counter = 0
                    
                    logger.info(f"학습률 감소: {self.current_lr:.6f}")
        
        self.lr_history.append(self.current_lr)
        return self.current_lr
    
    def get_recommended_lr(self, error_patterns: Dict[str, Any]) -> float:
        """오류 패턴 기반 학습률 권장"""
        
        recommended_lr = self.current_lr
        
        # 추세 기반 조정
        if 'trend' in error_patterns:
            trend = error_patterns['trend']
            if trend['direction'] == 'increasing' and trend['significance'] > 0.1:
                # 오류 증가 추세 → 학습률 감소
                recommended_lr = self.current_lr * 0.8
            elif trend['direction'] == 'decreasing' and trend['significance'] > 0.1:
                # 오류 감소 추세 → 학습률 약간 증가
                recommended_lr = self.current_lr * 1.1
        
        # 안정성 기반 조정
        if 'stability' in error_patterns:
            stability = error_patterns['stability']
            if not stability['is_stable']:
                # 불안정 → 학습률 감소
                recommended_lr = min(recommended_lr, self.current_lr * 0.9)
        
        # 이상치 기반 조정
        if 'outliers' in error_patterns:
            outliers = error_patterns['outliers']
            if outliers['percentage'] > 20:
                # 이상치 많음 → 학습률 감소
                recommended_lr = min(recommended_lr, self.current_lr * 0.7)
        
        return max(recommended_lr, self.min_lr)

class PhaseControllerHook:
    """단계 제어 훅 시스템"""
    
    def __init__(self, models: Dict[str, nn.Module], 
                 performance_threshold: float = 0.8,
                 error_threshold: float = 0.1,
                 monitoring_window: int = 50):
        
        self.models = models
        self.performance_threshold = performance_threshold
        self.error_threshold = error_threshold
        self.monitoring_window = monitoring_window
        
        # GPU 메모리 관리자
        try:
            from dynamic_gpu_manager import DynamicGPUManager
            self.gpu_manager = DynamicGPUManager()
            self.gpu_optimization_enabled = True
            logger.info("PhaseController Hook GPU 메모리 관리 활성화")
        except ImportError:
            self.gpu_manager = None
            self.gpu_optimization_enabled = False
            logger.warning("Dynamic GPU Manager를 찾을 수 없습니다. 기본 메모리 관리 사용")
        
        # 성능 모니터링
        self.performance_history = deque(maxlen=monitoring_window)
        self.error_analyzer = ErrorPatternAnalyzer(window_size=monitoring_window)
        
        # 학습률 스케줄러
        self.lr_schedulers = {
            model_name: AdaptiveLearningRateScheduler()
            for model_name in models.keys()
        }
        
        # 튜닝 액션
        self.pending_actions = []
        self.executed_actions = []
        
        # 성능 임계값
        self.thresholds = {
            PerformanceMetric.ACCURACY: 0.85,
            PerformanceMetric.LOSS: 0.3,
            PerformanceMetric.REGRET_PREDICTION_ERROR: 0.15,
            PerformanceMetric.BENTHAM_CALCULATION_ERROR: 0.1,
            PerformanceMetric.ETHICS_CLASSIFICATION_ERROR: 0.2
        }
        
        # 튜닝 전략
        self.tuning_strategies = {
            'learning_rate_adjustment': self._adjust_learning_rate,
            'model_architecture_tuning': self._tune_model_architecture,
            'regularization_adjustment': self._adjust_regularization,
            'batch_size_optimization': self._optimize_batch_size,
            'early_stopping_adjustment': self._adjust_early_stopping
        }
        
        # 실행 상태
        self.is_monitoring = False
        self.monitoring_thread = None
        
        logger.info("PhaseController Hook 시스템 초기화 완료")
    
    def start_monitoring(self):
        """모니터링 시작"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info("성능 모니터링 시작")
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        
        logger.info("성능 모니터링 중지")
    
    def _monitoring_loop(self):
        """모니터링 루프"""
        while self.is_monitoring:
            try:
                # 성능 체크
                self._check_performance()
                
                # 튜닝 액션 실행
                self._execute_pending_actions()
                
                # 1초 대기
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"모니터링 루프 오류: {e}")
                time.sleep(5.0)
    
    def record_performance(self, phase_type: PhaseType, 
                          metrics: Dict[str, float],
                          model_performances: Optional[Dict[str, Dict[str, float]]] = None,
                          context: Optional[Dict[str, Any]] = None):
        """성능 기록"""
        
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            phase_type=phase_type,
            metrics=metrics,
            model_performances=model_performances or {},
            metadata=context or {}
        )
        
        self.performance_history.append(snapshot)
        
        # 오류 패턴 분석용 데이터 추가
        if 'error' in metrics:
            self.error_analyzer.add_error(metrics['error'], context or {})
        
        logger.debug(f"성능 기록: {phase_type.value}, 메트릭: {metrics}")
    
    def _check_performance(self):
        """성능 체크 및 튜닝 액션 생성"""
        
        if len(self.performance_history) < 10:
            return
        
        # 최근 성능 분석
        recent_snapshots = list(self.performance_history)[-10:]
        
        # 각 메트릭별 성능 체크
        for metric_name, threshold in self.thresholds.items():
            metric_key = metric_name.value
            
            # 메트릭 값 추출
            recent_values = []
            for snapshot in recent_snapshots:
                if metric_key in snapshot.metrics:
                    recent_values.append(snapshot.metrics[metric_key])
            
            if len(recent_values) >= 5:
                avg_value = np.mean(recent_values)
                
                # 임계값 확인
                if (metric_name in [PerformanceMetric.LOSS, 
                                  PerformanceMetric.REGRET_PREDICTION_ERROR,
                                  PerformanceMetric.BENTHAM_CALCULATION_ERROR,
                                  PerformanceMetric.ETHICS_CLASSIFICATION_ERROR] and 
                    avg_value > threshold):
                    
                    # 성능 저하 → 튜닝 액션 생성
                    self._generate_tuning_actions(metric_name, avg_value, threshold)
                    
                elif (metric_name in [PerformanceMetric.ACCURACY, 
                                    PerformanceMetric.PRECISION, 
                                    PerformanceMetric.RECALL, 
                                    PerformanceMetric.F1_SCORE] and 
                      avg_value < threshold):
                    
                    # 성능 저하 → 튜닝 액션 생성
                    self._generate_tuning_actions(metric_name, avg_value, threshold)
    
    def _generate_tuning_actions(self, metric: PerformanceMetric, 
                               current_value: float, threshold: float):
        """튜닝 액션 생성"""
        
        # 오류 패턴 분석
        error_patterns = self.error_analyzer.detect_patterns()
        
        # 성능 차이 계산
        performance_gap = abs(current_value - threshold)
        
        # 우선순위 결정
        priority = min(10, max(1, int(performance_gap * 10)))
        
        # 튜닝 전략 선택
        actions = []
        
        # 1. 학습률 조정
        if 'trend' in error_patterns:
            for model_name, scheduler in self.lr_schedulers.items():
                recommended_lr = scheduler.get_recommended_lr(error_patterns)
                
                if recommended_lr != scheduler.current_lr:
                    actions.append(TuningAction(
                        action_type='learning_rate_adjustment',
                        target_model=model_name,
                        parameters={'learning_rate': recommended_lr},
                        priority=priority,
                        timestamp=datetime.now(),
                        expected_improvement=performance_gap * 0.3,
                        confidence=0.7
                    ))
        
        # 2. 모델 아키텍처 조정
        if performance_gap > 0.2:
            actions.append(TuningAction(
                action_type='model_architecture_tuning',
                target_model='all',
                parameters={'adjustment_type': 'increase_complexity'},
                priority=priority - 1,
                timestamp=datetime.now(),
                expected_improvement=performance_gap * 0.5,
                confidence=0.6
            ))
        
        # 3. 정규화 조정
        if 'stability' in error_patterns and not error_patterns['stability']['is_stable']:
            actions.append(TuningAction(
                action_type='regularization_adjustment',
                target_model='all',
                parameters={'increase_regularization': True},
                priority=priority,
                timestamp=datetime.now(),
                expected_improvement=performance_gap * 0.4,
                confidence=0.8
            ))
        
        # 기존 액션과 중복 제거
        for action in actions:
            if not self._is_duplicate_action(action):
                self.pending_actions.append(action)
                logger.info(f"튜닝 액션 생성: {action.action_type} for {action.target_model}")
    
    def _is_duplicate_action(self, new_action: TuningAction) -> bool:
        """중복 액션 확인"""
        for existing_action in self.pending_actions:
            if (existing_action.action_type == new_action.action_type and
                existing_action.target_model == new_action.target_model and
                not existing_action.executed):
                return True
        return False
    
    def _execute_pending_actions(self):
        """대기 중인 액션 실행"""
        
        if not self.pending_actions:
            return
        
        # 우선순위 정렬
        self.pending_actions.sort(key=lambda x: x.priority, reverse=True)
        
        # 최대 3개 액션 실행
        actions_to_execute = self.pending_actions[:3]
        
        for action in actions_to_execute:
            try:
                success = self._execute_action(action)
                
                if success:
                    action.executed = True
                    action.execution_time = datetime.now()
                    self.executed_actions.append(action)
                    
                    logger.info(f"튜닝 액션 실행 완료: {action.action_type}")
                else:
                    logger.warning(f"튜닝 액션 실행 실패: {action.action_type}")
                
            except Exception as e:
                logger.error(f"튜닝 액션 실행 오류: {e}")
            
            finally:
                self.pending_actions.remove(action)
    
    def _execute_action(self, action: TuningAction) -> bool:
        """액션 실행"""
        
        strategy = self.tuning_strategies.get(action.action_type)
        if not strategy:
            logger.warning(f"알 수 없는 튜닝 전략: {action.action_type}")
            return False
        
        try:
            return strategy(action)
        except Exception as e:
            logger.error(f"튜닝 전략 실행 오류: {e}")
            return False
    
    def _adjust_learning_rate(self, action: TuningAction) -> bool:
        """학습률 조정"""
        
        model_name = action.target_model
        new_lr = action.parameters.get('learning_rate')
        
        if model_name in self.models and new_lr is not None:
            # 모델의 옵티마이저 학습률 조정
            model = self.models[model_name]
            
            # 모델에 연결된 옵티마이저가 있다면 조정
            if hasattr(model, 'optimizer'):
                for param_group in model.optimizer.param_groups:
                    param_group['lr'] = new_lr
                
                # 스케줄러 업데이트
                if model_name in self.lr_schedulers:
                    self.lr_schedulers[model_name].current_lr = new_lr
                
                logger.info(f"모델 {model_name} 학습률 조정: {new_lr}")
                return True
        
        return False
    
    def _tune_model_architecture(self, action: TuningAction) -> bool:
        """모델 아키텍처 조정"""
        
        adjustment_type = action.parameters.get('adjustment_type', 'increase_complexity')
        
        # 실제 구현에서는 모델 아키텍처를 동적으로 조정
        # 여기서는 로깅만 수행
        logger.info(f"모델 아키텍처 조정: {adjustment_type}")
        
        return True
    
    def _adjust_regularization(self, action: TuningAction) -> bool:
        """정규화 조정"""
        
        increase_regularization = action.parameters.get('increase_regularization', True)
        
        # 실제 구현에서는 정규화 파라미터 조정
        # 여기서는 로깅만 수행
        logger.info(f"정규화 조정: {'증가' if increase_regularization else '감소'}")
        
        return True
    
    def _optimize_batch_size(self, action: TuningAction) -> bool:
        """배치 크기 최적화"""
        
        # 실제 구현에서는 배치 크기 동적 조정
        # 여기서는 로깅만 수행
        logger.info("배치 크기 최적화")
        
        return True
    
    def _adjust_early_stopping(self, action: TuningAction) -> bool:
        """조기 종료 조정"""
        
        # 실제 구현에서는 조기 종료 파라미터 조정
        # 여기서는 로깅만 수행
        logger.info("조기 종료 파라미터 조정")
        
        return True
    
    def get_performance_report(self) -> Dict[str, Any]:
        """성능 보고서 생성"""
        
        if not self.performance_history:
            return {}
        
        # 최근 성능 통계
        recent_snapshots = list(self.performance_history)[-20:]
        
        report = {
            'summary': {
                'total_snapshots': len(self.performance_history),
                'monitoring_window': self.monitoring_window,
                'recent_snapshots': len(recent_snapshots)
            },
            'performance_trends': {},
            'executed_actions': len(self.executed_actions),
            'pending_actions': len(self.pending_actions),
            'error_patterns': self.error_analyzer.detect_patterns()
        }
        
        # 메트릭별 추세 분석
        for metric_name in ['accuracy', 'loss', 'regret_prediction_error']:
            values = []
            for snapshot in recent_snapshots:
                if metric_name in snapshot.metrics:
                    values.append(snapshot.metrics[metric_name])
            
            if values:
                report['performance_trends'][metric_name] = {
                    'current': values[-1],
                    'average': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'trend': 'improving' if len(values) > 1 and values[-1] > values[0] else 'stable'
                }
        
        return report
    
    def reset_monitoring(self):
        """모니터링 리셋"""
        self.performance_history.clear()
        self.error_analyzer = ErrorPatternAnalyzer(window_size=self.monitoring_window)
        self.pending_actions.clear()
        self.executed_actions.clear()
        
        logger.info("모니터링 시스템 리셋 완료")