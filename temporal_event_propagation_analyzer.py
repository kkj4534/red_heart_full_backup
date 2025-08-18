"""
시계열 사건 전파 분석기 (Temporal Event Propagation Analyzer)
Temporal Event Propagation Analysis Module

시간에 따른 사건의 전파 패턴을 분석하고 학습을 통해 미래 결과를 예측하는
지능형 시계열 분석 시스템을 구현하여 의사결정의 장기적 영향을 고려합니다.

핵심 기능:
1. 다층 시계열 사건 모델링 (초, 분, 시, 일, 월 단위)
2. 인과관계 기반 사건 전파 패턴 학습
3. 확률적 미래 예측 및 시나리오 생성
4. 감정-윤리 회로와의 유기적 통합
"""

import numpy as np
import torch
import torch.nn as nn
import logging
import json
import time
import math
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from pathlib import Path
import threading
from datetime import datetime, timedelta

try:
    from config import ADVANCED_CONFIG, DEVICE
except ImportError:
    # config.py에 문제가 있을 경우 기본값 사용
    ADVANCED_CONFIG = {'enable_gpu': False}
    DEVICE = 'cpu'
    print("⚠️  config.py 임포트 실패, 기본값 사용")
from data_models import EmotionData

logger = logging.getLogger('TemporalEventPropagationAnalyzer')

class TemporalScale(Enum):
    """시간 척도"""
    IMMEDIATE = "immediate"      # 즉시 (초~분)
    SHORT_TERM = "short_term"    # 단기 (분~시)
    MEDIUM_TERM = "medium_term"  # 중기 (시~일)
    LONG_TERM = "long_term"      # 장기 (일~월)
    GENERATIONAL = "generational" # 세대 (월~년)

@dataclass
class TemporalEvent:
    """시간적 사건 클래스"""
    event_id: str
    timestamp: float
    event_type: str
    description: str
    
    # 사건 속성
    intensity: float = 0.5      # 사건 강도 (0~1)
    scope: float = 0.5          # 영향 범위 (0~1)
    reversibility: float = 0.5   # 가역성 (0~1, 높을수록 되돌리기 쉬움)
    
    # 관련 엔티티
    primary_actors: List[str] = field(default_factory=list)
    affected_entities: List[str] = field(default_factory=list)
    
    # 감정적 맥락
    emotion_state: Optional[EmotionData] = None
    ethical_implications: Dict[str, float] = field(default_factory=dict)
    
    # 인과관계
    causal_antecedents: List[str] = field(default_factory=list)  # 원인 사건들
    expected_consequences: List[str] = field(default_factory=list)  # 예상 결과
    
    # 불확실성
    certainty_level: float = 0.7
    prediction_confidence: float = 0.5

@dataclass
class PropagationPath:
    """전파 경로"""
    path_id: str
    source_event: str
    target_event: str
    propagation_delay: float  # 전파 지연 시간 (초)
    
    # 전파 특성
    transmission_strength: float = 0.5  # 전파 강도
    decay_rate: float = 0.1            # 감쇠율
    amplification_factor: float = 1.0   # 증폭 인수
    
    # 전파 메커니즘
    propagation_type: str = "direct"    # direct, indirect, cascade
    mediating_factors: List[str] = field(default_factory=list)
    
    # 학습된 가중치
    learned_weight: float = 1.0
    confidence: float = 0.5

@dataclass
class TemporalPrediction:
    """시간적 예측"""
    prediction_id: str
    target_time: float
    predicted_events: List[TemporalEvent]
    
    # 예측 품질
    confidence_score: float = 0.5
    uncertainty_range: Tuple[float, float] = (0.0, 1.0)
    
    # 시나리오 분석
    best_case_scenario: Dict[str, Any] = field(default_factory=dict)
    worst_case_scenario: Dict[str, Any] = field(default_factory=dict)
    most_likely_scenario: Dict[str, Any] = field(default_factory=dict)
    
    # 예측 근거
    contributing_patterns: List[str] = field(default_factory=list)
    historical_precedents: List[str] = field(default_factory=list)

class TemporalPatternDetector:
    """시간 패턴 탐지기"""
    
    def __init__(self, window_sizes: List[int] = None):
        if window_sizes is None:
            window_sizes = [5, 10, 20, 50, 100]  # 다양한 시간 윈도우
        
        self.window_sizes = window_sizes
        self.detected_patterns = defaultdict(list)
        self.pattern_confidence = defaultdict(float)
        
    def detect_cyclic_patterns(self, events: List[TemporalEvent]) -> Dict[str, Any]:
        """주기적 패턴 탐지"""
        if len(events) < 10:
            return {}
        
        # 이벤트 타입별 시간 간격 분석
        type_timestamps = defaultdict(list)
        for event in events:
            type_timestamps[event.event_type].append(event.timestamp)
        
        cyclic_patterns = {}
        
        for event_type, timestamps in type_timestamps.items():
            if len(timestamps) < 3:
                continue
                
            timestamps.sort()
            intervals = np.diff(timestamps)
            
            if len(intervals) > 0:
                # 주기성 분석
                mean_interval = np.mean(intervals)
                std_interval = np.std(intervals)
                
                # 주기성 점수 (낮은 표준편차 = 높은 주기성)
                periodicity_score = 1.0 / (1.0 + std_interval / max(mean_interval, 1e-6))
                
                if periodicity_score > 0.7:  # 임계값
                    cyclic_patterns[event_type] = {
                        'period': mean_interval,
                        'variability': std_interval,
                        'periodicity_score': periodicity_score,
                        'sample_count': len(intervals)
                    }
        
        return cyclic_patterns
    
    def detect_cascade_patterns(self, events: List[TemporalEvent]) -> List[Dict[str, Any]]:
        """연쇄 반응 패턴 탐지"""
        cascade_patterns = []
        
        # 시간순 정렬
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        for i, seed_event in enumerate(sorted_events):
            # 시드 이벤트 후 일정 시간 내 발생한 이벤트들 찾기
            cascade_window = 3600  # 1시간
            cascade_events = []
            
            for j in range(i + 1, len(sorted_events)):
                follow_event = sorted_events[j]
                time_diff = follow_event.timestamp - seed_event.timestamp
                
                if time_diff > cascade_window:
                    break
                
                # 연관성 체크 (공통 액터, 유사한 타입 등)
                relevance_score = self._calculate_event_relevance(seed_event, follow_event)
                
                if relevance_score > 0.5:
                    cascade_events.append({
                        'event': follow_event,
                        'delay': time_diff,
                        'relevance': relevance_score
                    })
            
            if len(cascade_events) >= 2:  # 최소 2개 이상의 후속 이벤트
                cascade_patterns.append({
                    'seed_event': seed_event,
                    'cascade_events': cascade_events,
                    'cascade_strength': np.mean([ce['relevance'] for ce in cascade_events]),
                    'average_delay': np.mean([ce['delay'] for ce in cascade_events])
                })
        
        return cascade_patterns
    
    def _calculate_event_relevance(self, event1: TemporalEvent, event2: TemporalEvent) -> float:
        """두 이벤트 간 연관성 계산"""
        relevance = 0.0
        
        # 공통 액터
        common_actors = set(event1.primary_actors) & set(event2.primary_actors)
        if common_actors:
            relevance += 0.4
        
        # 공통 영향 대상
        common_affected = set(event1.affected_entities) & set(event2.affected_entities)
        if common_affected:
            relevance += 0.3
        
        # 이벤트 타입 유사성
        if event1.event_type == event2.event_type:
            relevance += 0.2
        elif event1.event_type in event2.expected_consequences:
            relevance += 0.5
        
        # 강도 유사성
        intensity_similarity = 1.0 - abs(event1.intensity - event2.intensity)
        relevance += intensity_similarity * 0.1
        
        return min(relevance, 1.0)

class TemporalLearningNetwork(nn.Module):
    """시간적 학습 신경망"""
    
    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # LSTM for temporal sequence modeling
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Attention mechanism for important events
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1
        )
        
        # Prediction heads
        self.event_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim)  # Predict next event features
        )
        
        self.timing_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),  # Predict time until next event
            nn.Softplus()  # Ensure positive timing
        )
        
        self.confidence_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Confidence score 0-1
        )
        
    def forward(self, event_sequences: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            event_sequences: (batch_size, seq_len, input_dim)
        
        Returns:
            next_event_features: (batch_size, input_dim)
            next_event_timing: (batch_size, 1) 
            prediction_confidence: (batch_size, 1)
        """
        batch_size, seq_len, _ = event_sequences.shape
        
        # LSTM encoding
        lstm_out, (hidden, cell) = self.lstm(event_sequences)
        
        # Attention over sequence
        attended_out, attention_weights = self.attention(
            lstm_out.transpose(0, 1),  # (seq_len, batch_size, hidden_dim)
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1)
        )
        
        # Use last timestep for prediction
        final_representation = attended_out[-1]  # (batch_size, hidden_dim)
        
        # Predictions
        next_event = self.event_predictor(final_representation)
        next_timing = self.timing_predictor(final_representation)
        confidence = self.confidence_predictor(final_representation)
        
        return next_event, next_timing, confidence

class TemporalEventPropagationAnalyzer:
    """시계열 사건 전파 분석기"""
    
    def __init__(self):
        self.logger = logger
        
        # 이벤트 저장소
        self.events_database = deque(maxlen=10000)  # 최근 10,000개 이벤트
        self.propagation_paths = {}
        
        # 패턴 탐지기
        self.pattern_detector = TemporalPatternDetector()
        
        # 학습 네트워크
        self.learning_network = TemporalLearningNetwork().to(DEVICE)
        self.optimizer = torch.optim.Adam(self.learning_network.parameters(), lr=0.001)
        
        # 시간 척도별 분석기
        self.scale_analyzers = {
            scale: self._create_scale_analyzer(scale) for scale in TemporalScale
        }
        
        # 예측 캐시
        self.prediction_cache = {}
        self.cache_duration = 300  # 5분 캐시
        
        # 성능 통계
        self.performance_stats = {
            'total_predictions': 0,
            'accurate_predictions': 0,
            'average_confidence': 0.0,
            'prediction_errors': deque(maxlen=1000)
        }
        
        # 실시간 모니터링
        self.monitoring_active = False
        self.monitoring_thread = None
        
        self.logger.info("시계열 사건 전파 분석기 초기화 완료")
    
    def _create_scale_analyzer(self, scale: TemporalScale) -> Dict[str, Any]:
        """시간 척도별 분석기 생성"""
        scale_configs = {
            TemporalScale.IMMEDIATE: {
                'window_size': 60,      # 1분
                'prediction_horizon': 300,  # 5분
                'decay_rate': 0.5,
                'pattern_sensitivity': 0.9
            },
            TemporalScale.SHORT_TERM: {
                'window_size': 3600,    # 1시간
                'prediction_horizon': 14400,  # 4시간
                'decay_rate': 0.3,
                'pattern_sensitivity': 0.7
            },
            TemporalScale.MEDIUM_TERM: {
                'window_size': 86400,   # 1일
                'prediction_horizon': 604800,  # 1주
                'decay_rate': 0.1,
                'pattern_sensitivity': 0.5
            },
            TemporalScale.LONG_TERM: {
                'window_size': 604800,  # 1주
                'prediction_horizon': 2592000,  # 1달
                'decay_rate': 0.05,
                'pattern_sensitivity': 0.3
            },
            TemporalScale.GENERATIONAL: {
                'window_size': 2592000, # 1달
                'prediction_horizon': 31536000,  # 1년
                'decay_rate': 0.01,
                'pattern_sensitivity': 0.1
            }
        }
        
        return scale_configs[scale]
    
    def register_event(self, event: TemporalEvent) -> bool:
        """새로운 이벤트 등록"""
        try:
            # 이벤트 검증
            if not self._validate_event(event):
                self.logger.warning(f"유효하지 않은 이벤트: {event.event_id}")
                return False
            
            # 데이터베이스에 추가
            self.events_database.append(event)
            
            # 실시간 패턴 업데이트
            self._update_patterns_incrementally(event)
            
            # 전파 경로 학습
            self._learn_propagation_paths(event)
            
            # 예측 캐시 무효화 (관련 예측만)
            self._invalidate_related_predictions(event)
            
            self.logger.debug(f"이벤트 등록 완료: {event.event_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"이벤트 등록 오류: {e}")
            return False
    
    def _validate_event(self, event: TemporalEvent) -> bool:
        """이벤트 유효성 검증"""
        if not event.event_id or not event.event_type:
            return False
        
        if not (0 <= event.intensity <= 1):
            return False
        
        if not (0 <= event.scope <= 1):
            return False
        
        if event.timestamp <= 0:
            return False
        
        return True
    
    def analyze_temporal_patterns(self, scale: TemporalScale = None) -> Dict[str, Any]:
        """시간적 패턴 분석"""
        if scale:
            return self._analyze_single_scale(scale)
        
        # 모든 척도에 대해 분석
        all_patterns = {}
        
        for temporal_scale in TemporalScale:
            scale_patterns = self._analyze_single_scale(temporal_scale)
            all_patterns[temporal_scale.value] = scale_patterns
        
        # 크로스 스케일 패턴 분석
        cross_scale_patterns = self._analyze_cross_scale_patterns()
        all_patterns['cross_scale'] = cross_scale_patterns
        
        return all_patterns
    
    def _analyze_single_scale(self, scale: TemporalScale) -> Dict[str, Any]:
        """단일 시간 척도 패턴 분석"""
        analyzer_config = self.scale_analyzers[scale]
        window_size = analyzer_config['window_size']
        
        # 해당 시간 윈도우 내 이벤트 필터링
        current_time = time.time()
        window_start = current_time - window_size
        
        scale_events = [
            event for event in self.events_database
            if event.timestamp >= window_start
        ]
        
        if len(scale_events) < 5:
            return {'insufficient_data': True, 'event_count': len(scale_events)}
        
        # 패턴 탐지
        cyclic_patterns = self.pattern_detector.detect_cyclic_patterns(scale_events)
        cascade_patterns = self.pattern_detector.detect_cascade_patterns(scale_events)
        
        # 통계 분석
        event_frequency = self._calculate_event_frequency(scale_events, window_size)
        intensity_trends = self._analyze_intensity_trends(scale_events)
        
        return {
            'scale': scale.value,
            'event_count': len(scale_events),
            'time_window': window_size,
            'cyclic_patterns': cyclic_patterns,
            'cascade_patterns': cascade_patterns,
            'event_frequency': event_frequency,
            'intensity_trends': intensity_trends,
            'pattern_confidence': analyzer_config['pattern_sensitivity']
        }
    
    def _analyze_cross_scale_patterns(self) -> Dict[str, Any]:
        """크로스 스케일 패턴 분석"""
        cross_patterns = {}
        
        # 단기와 장기 이벤트 간 연관성
        recent_events = list(self.events_database)[-50:]  # 최근 50개
        
        if len(recent_events) < 10:
            return {'insufficient_data': True}
        
        # 강도 증폭/감쇠 패턴
        intensity_evolution = [event.intensity for event in recent_events]
        if len(intensity_evolution) > 1:
            intensity_trend = np.polyfit(range(len(intensity_evolution)), intensity_evolution, 1)[0]
            cross_patterns['intensity_trend'] = float(intensity_trend)
        
        # 이벤트 타입 전환 패턴
        type_transitions = self._analyze_type_transitions(recent_events)
        cross_patterns['type_transitions'] = type_transitions
        
        # 감정-윤리 상관관계
        emotion_ethics_correlation = self._analyze_emotion_ethics_correlation(recent_events)
        cross_patterns['emotion_ethics_correlation'] = emotion_ethics_correlation
        
        return cross_patterns
    
    def predict_future_events(
        self, 
        prediction_horizon: float,
        scale: TemporalScale = TemporalScale.SHORT_TERM,
        confidence_threshold: float = 0.6
    ) -> TemporalPrediction:
        """미래 이벤트 예측"""
        
        # 캐시 확인
        cache_key = f"{prediction_horizon}_{scale.value}_{confidence_threshold}"
        if cache_key in self.prediction_cache:
            cached_prediction, cache_time = self.prediction_cache[cache_key]
            if time.time() - cache_time < self.cache_duration:
                return cached_prediction
        
        # 예측 실행
        prediction = self._generate_prediction(prediction_horizon, scale, confidence_threshold)
        
        # 캐시 저장
        self.prediction_cache[cache_key] = (prediction, time.time())
        
        # 성능 통계 업데이트
        self.performance_stats['total_predictions'] += 1
        
        return prediction
    
    def _generate_prediction(
        self,
        prediction_horizon: float,
        scale: TemporalScale,
        confidence_threshold: float
    ) -> TemporalPrediction:
        """예측 생성"""
        
        current_time = time.time()
        target_time = current_time + prediction_horizon
        
        # 학습 데이터 준비
        training_data = self._prepare_training_data(scale)
        
        if len(training_data) < 10:
            # 데이터 부족 시 규칙 기반 예측
            return self._rule_based_prediction(target_time, scale)
        
        # 신경망 기반 예측
        neural_prediction = self._neural_network_prediction(training_data, prediction_horizon)
        
        # 패턴 기반 예측
        pattern_prediction = self._pattern_based_prediction(target_time, scale)
        
        # 예측 결합
        combined_prediction = self._combine_predictions(
            neural_prediction, 
            pattern_prediction, 
            target_time
        )
        
        # 신뢰도 검증
        if combined_prediction.confidence_score < confidence_threshold:
            # 신뢰도 부족 시 보수적 예측
            combined_prediction = self._conservative_prediction(target_time, scale)
        
        return combined_prediction
    
    def _prepare_training_data(self, scale: TemporalScale) -> List[torch.Tensor]:
        """학습 데이터 준비"""
        analyzer_config = self.scale_analyzers[scale]
        window_size = analyzer_config['window_size']
        
        # 시간 윈도우 내 이벤트 추출
        current_time = time.time()
        relevant_events = [
            event for event in self.events_database
            if current_time - event.timestamp <= window_size * 2  # 충분한 데이터
        ]
        
        if len(relevant_events) < 20:
            return []
        
        # 이벤트를 특성 벡터로 변환
        event_features = []
        for event in relevant_events:
            features = self._event_to_features(event)
            event_features.append(features)
        
        # 시계열 시퀀스 생성
        sequence_length = 10
        sequences = []
        
        for i in range(len(event_features) - sequence_length):
            sequence = torch.stack(event_features[i:i+sequence_length])
            sequences.append(sequence)
        
        return sequences
    
    def _event_to_features(self, event: TemporalEvent) -> torch.Tensor:
        """이벤트를 특성 벡터로 변환"""
        features = []
        
        # 기본 속성
        features.extend([
            event.intensity,
            event.scope,
            event.reversibility,
            event.certainty_level,
            event.prediction_confidence
        ])
        
        # 시간 특성 (주기성을 위한 사인/코사인 인코딩)
        time_of_day = (event.timestamp % 86400) / 86400  # 하루 중 시간
        day_of_week = ((event.timestamp // 86400) % 7) / 7  # 요일
        
        features.extend([
            math.sin(2 * math.pi * time_of_day),
            math.cos(2 * math.pi * time_of_day),
            math.sin(2 * math.pi * day_of_week),
            math.cos(2 * math.pi * day_of_week)
        ])
        
        # 이벤트 타입 원핫 인코딩 (상위 10개 타입)
        common_types = ['decision', 'consequence', 'feedback', 'external', 'internal', 
                       'social', 'economic', 'ethical', 'emotional', 'other']
        
        type_vector = [1.0 if event.event_type == t else 0.0 for t in common_types]
        features.extend(type_vector)
        
        # 감정 상태 (있을 경우)
        if event.emotion_state:
            emotion_features = [
                getattr(event.emotion_state, attr, 0.0) 
                for attr in ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust']
            ]
            features.extend(emotion_features)
        else:
            features.extend([0.0] * 6)
        
        # 윤리적 함의
        ethics_features = [
            event.ethical_implications.get('harm', 0.0),
            event.ethical_implications.get('fairness', 0.0),
            event.ethical_implications.get('loyalty', 0.0),
            event.ethical_implications.get('authority', 0.0),
            event.ethical_implications.get('sanctity', 0.0)
        ]
        features.extend(ethics_features)
        
        # 액터 및 영향 수
        features.extend([
            len(event.primary_actors) / 10.0,  # 정규화
            len(event.affected_entities) / 10.0,
            len(event.causal_antecedents) / 5.0,
            len(event.expected_consequences) / 5.0
        ])
        
        # 패딩으로 64차원 맞추기
        while len(features) < 64:
            features.append(0.0)
        
        return torch.tensor(features[:64], dtype=torch.float32).to(DEVICE)
    
    def _neural_network_prediction(
        self,
        training_data: List[torch.Tensor],
        prediction_horizon: float
    ) -> TemporalPrediction:
        """신경망 기반 예측"""
        
        if not training_data:
            return self._empty_prediction()
        
        try:
            # 학습 데이터를 배치로 변환
            batch_size = min(len(training_data), 32)
            batch = torch.stack(training_data[:batch_size])
            
            # 예측 실행
            with torch.no_grad():
                self.learning_network.eval()
                next_event_features, next_timing, confidence = self.learning_network(batch)
            
            # 예측 결과를 이벤트로 변환
            predicted_events = []
            current_time = time.time()
            
            for i in range(batch_size):
                features = next_event_features[i].cpu().numpy()
                timing_offset = next_timing[i].item()
                pred_confidence = confidence[i].item()
                
                # 특성에서 이벤트 속성 추출
                predicted_event = TemporalEvent(
                    event_id=f"pred_{int(current_time)}_{i}",
                    timestamp=current_time + timing_offset,
                    event_type=self._features_to_event_type(features),
                    description=f"Neural network predicted event at +{timing_offset:.1f}s",
                    intensity=min(max(features[0], 0.0), 1.0),
                    scope=min(max(features[1], 0.0), 1.0),
                    reversibility=min(max(features[2], 0.0), 1.0),
                    certainty_level=pred_confidence,
                    prediction_confidence=pred_confidence
                )
                
                predicted_events.append(predicted_event)
            
            # 예측 객체 생성
            prediction = TemporalPrediction(
                prediction_id=f"neural_{int(current_time)}",
                target_time=current_time + prediction_horizon,
                predicted_events=predicted_events,
                confidence_score=float(confidence.mean().item()),
                contributing_patterns=['neural_network_analysis']
            )
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"신경망 예측 오류: {e}")
            return self._empty_prediction()
    
    def _features_to_event_type(self, features: np.ndarray) -> str:
        """특성 벡터에서 이벤트 타입 추출"""
        type_start_idx = 9  # 타입 원핫 인코딩 시작 위치
        type_features = features[type_start_idx:type_start_idx+10]
        
        common_types = ['decision', 'consequence', 'feedback', 'external', 'internal', 
                       'social', 'economic', 'ethical', 'emotional', 'other']
        
        max_idx = np.argmax(type_features)
        return common_types[max_idx]
    
    def _pattern_based_prediction(
        self,
        target_time: float,
        scale: TemporalScale
    ) -> TemporalPrediction:
        """패턴 기반 예측"""
        
        # 현재 패턴 분석
        current_patterns = self._analyze_single_scale(scale)
        
        if current_patterns.get('insufficient_data'):
            return self._empty_prediction()
        
        predicted_events = []
        confidence_scores = []
        
        # 주기적 패턴 기반 예측
        cyclic_patterns = current_patterns.get('cyclic_patterns', {})
        for event_type, pattern_info in cyclic_patterns.items():
            next_occurrence = self._predict_next_cyclic_event(
                event_type, pattern_info, target_time
            )
            if next_occurrence:
                predicted_events.append(next_occurrence)
                confidence_scores.append(pattern_info['periodicity_score'])
        
        # 연쇄 반응 패턴 기반 예측
        cascade_patterns = current_patterns.get('cascade_patterns', [])
        for cascade in cascade_patterns:
            cascade_prediction = self._predict_cascade_continuation(cascade, target_time)
            if cascade_prediction:
                predicted_events.extend(cascade_prediction)
                confidence_scores.extend([0.7] * len(cascade_prediction))
        
        # 전체 신뢰도 계산
        overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.3
        
        prediction = TemporalPrediction(
            prediction_id=f"pattern_{int(target_time)}",
            target_time=target_time,
            predicted_events=predicted_events,
            confidence_score=overall_confidence,
            contributing_patterns=list(cyclic_patterns.keys()) + ['cascade_analysis']
        )
        
        return prediction
    
    def _predict_next_cyclic_event(
        self,
        event_type: str,
        pattern_info: Dict[str, Any],
        target_time: float
    ) -> Optional[TemporalEvent]:
        """다음 주기적 이벤트 예측"""
        
        period = pattern_info['period']
        periodicity_score = pattern_info['periodicity_score']
        
        # 마지막 해당 타입 이벤트 찾기
        last_event = None
        for event in reversed(list(self.events_database)):
            if event.event_type == event_type:
                last_event = event
                break
        
        if not last_event:
            return None
        
        # 다음 발생 시간 예측
        time_since_last = time.time() - last_event.timestamp
        periods_passed = time_since_last / period
        next_period_start = (int(periods_passed) + 1) * period
        next_occurrence_time = last_event.timestamp + next_period_start
        
        # 목표 시간 내에 있는지 확인
        if next_occurrence_time > target_time:
            return None
        
        # 예측 이벤트 생성
        predicted_event = TemporalEvent(
            event_id=f"cyclic_{event_type}_{int(next_occurrence_time)}",
            timestamp=next_occurrence_time,
            event_type=event_type,
            description=f"Predicted cyclic {event_type} event",
            intensity=last_event.intensity,  # 이전 강도와 유사하다고 가정
            scope=last_event.scope,
            reversibility=last_event.reversibility,
            certainty_level=periodicity_score,
            prediction_confidence=periodicity_score,
            expected_consequences=[f"continuation_of_{event_type}_pattern"]
        )
        
        return predicted_event
    
    def _predict_cascade_continuation(
        self,
        cascade: Dict[str, Any],
        target_time: float
    ) -> List[TemporalEvent]:
        """연쇄 반응 지속 예측"""
        
        seed_event = cascade['seed_event']
        cascade_events = cascade['cascade_events']
        average_delay = cascade['average_delay']
        
        if not cascade_events:
            return []
        
        # 마지막 연쇄 이벤트 시간
        last_cascade_time = max(ce['event'].timestamp for ce in cascade_events)
        
        # 다음 연쇄 이벤트 예측
        next_cascade_time = last_cascade_time + average_delay
        
        if next_cascade_time > target_time:
            return []
        
        # 연쇄 강도 계산 (감쇠 고려)
        cascade_strength = cascade['cascade_strength']
        time_decay = math.exp(-(next_cascade_time - seed_event.timestamp) / (average_delay * 3))
        effective_strength = cascade_strength * time_decay
        
        if effective_strength < 0.3:  # 너무 약한 연쇄는 무시
            return []
        
        # 예측 이벤트 생성
        predicted_event = TemporalEvent(
            event_id=f"cascade_{seed_event.event_id}_{int(next_cascade_time)}",
            timestamp=next_cascade_time,
            event_type=f"cascade_{seed_event.event_type}",
            description=f"Predicted cascade event from {seed_event.event_id}",
            intensity=seed_event.intensity * effective_strength,
            scope=seed_event.scope * 0.8,  # 범위는 약간 감소
            reversibility=seed_event.reversibility,
            certainty_level=effective_strength,
            prediction_confidence=effective_strength,
            causal_antecedents=[seed_event.event_id],
            primary_actors=seed_event.primary_actors.copy(),
            affected_entities=seed_event.affected_entities.copy()
        )
        
        return [predicted_event]
    
    def _combine_predictions(
        self,
        neural_prediction: TemporalPrediction,
        pattern_prediction: TemporalPrediction,
        target_time: float
    ) -> TemporalPrediction:
        """예측 결합"""
        
        combined_events = []
        combined_confidence_scores = []
        combined_patterns = []
        
        # 신경망 예측 추가
        if neural_prediction.predicted_events:
            combined_events.extend(neural_prediction.predicted_events)
            combined_confidence_scores.append(neural_prediction.confidence_score)
            combined_patterns.extend(neural_prediction.contributing_patterns)
        
        # 패턴 예측 추가
        if pattern_prediction.predicted_events:
            combined_events.extend(pattern_prediction.predicted_events)
            combined_confidence_scores.append(pattern_prediction.confidence_score)
            combined_patterns.extend(pattern_prediction.contributing_patterns)
        
        # 중복 제거 및 정렬
        combined_events = sorted(combined_events, key=lambda e: e.timestamp)
        
        # 전체 신뢰도 계산
        if combined_confidence_scores:
            overall_confidence = np.mean(combined_confidence_scores)
        else:
            overall_confidence = 0.3
        
        # 시나리오 분석
        scenarios = self._generate_scenarios(combined_events)
        
        combined_prediction = TemporalPrediction(
            prediction_id=f"combined_{int(target_time)}",
            target_time=target_time,
            predicted_events=combined_events,
            confidence_score=overall_confidence,
            contributing_patterns=list(set(combined_patterns)),
            best_case_scenario=scenarios['best_case'],
            worst_case_scenario=scenarios['worst_case'],
            most_likely_scenario=scenarios['most_likely']
        )
        
        return combined_prediction
    
    def _generate_scenarios(self, predicted_events: List[TemporalEvent]) -> Dict[str, Dict[str, Any]]:
        """시나리오 생성"""
        
        if not predicted_events:
            return {
                'best_case': {'total_impact': 0.0, 'risk_level': 0.0},
                'worst_case': {'total_impact': 0.0, 'risk_level': 0.0},
                'most_likely': {'total_impact': 0.0, 'risk_level': 0.0}
            }
        
        # 이벤트별 영향도 계산
        positive_impacts = []
        negative_impacts = []
        risk_levels = []
        
        for event in predicted_events:
            # 간단한 영향도 모델
            impact = event.intensity * event.scope
            risk = (1.0 - event.reversibility) * event.intensity
            
            if event.event_type in ['consequence', 'feedback', 'ethical']:
                positive_impacts.append(impact)
            else:
                negative_impacts.append(impact)
            
            risk_levels.append(risk)
        
        total_positive = sum(positive_impacts) if positive_impacts else 0.0
        total_negative = sum(negative_impacts) if negative_impacts else 0.0
        total_risk = np.mean(risk_levels) if risk_levels else 0.0
        
        scenarios = {
            'best_case': {
                'total_impact': total_positive * 1.3 - total_negative * 0.7,
                'risk_level': total_risk * 0.5,
                'description': 'Optimistic scenario with positive outcomes'
            },
            'worst_case': {
                'total_impact': total_positive * 0.7 - total_negative * 1.3,
                'risk_level': total_risk * 1.5,
                'description': 'Pessimistic scenario with negative outcomes'
            },
            'most_likely': {
                'total_impact': total_positive - total_negative,
                'risk_level': total_risk,
                'description': 'Most probable scenario based on patterns'
            }
        }
        
        return scenarios
    
    def evaluate_prediction_accuracy(self, prediction: TemporalPrediction) -> Dict[str, float]:
        """예측 정확도 평가"""
        
        if time.time() < prediction.target_time:
            return {'error': 'prediction_not_yet_due'}
        
        # 예측 시점 이후 실제 발생한 이벤트들
        actual_events = [
            event for event in self.events_database
            if prediction.target_time - 300 <= event.timestamp <= prediction.target_time + 300
        ]
        
        if not actual_events:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'temporal_accuracy': 0.0,
                'confidence_calibration': prediction.confidence_score
            }
        
        # 예측된 이벤트와 실제 이벤트 매칭
        matched_predictions = 0
        temporal_errors = []
        
        for pred_event in prediction.predicted_events:
            best_match = None
            best_similarity = 0.0
            
            for actual_event in actual_events:
                similarity = self._calculate_event_similarity(pred_event, actual_event)
                if similarity > best_similarity and similarity > 0.5:  # 임계값
                    best_similarity = similarity
                    best_match = actual_event
            
            if best_match:
                matched_predictions += 1
                temporal_error = abs(pred_event.timestamp - best_match.timestamp)
                temporal_errors.append(temporal_error)
        
        # 정확도 지표 계산
        precision = matched_predictions / len(prediction.predicted_events) if prediction.predicted_events else 0.0
        recall = matched_predictions / len(actual_events) if actual_events else 0.0
        temporal_accuracy = 1.0 / (1.0 + np.mean(temporal_errors)) if temporal_errors else 0.0
        
        # 성능 통계 업데이트
        if precision > 0.6:  # 임계값 이상이면 정확한 예측으로 간주
            self.performance_stats['accurate_predictions'] += 1
        
        error_rate = 1.0 - precision
        self.performance_stats['prediction_errors'].append(error_rate)
        
        return {
            'precision': precision,
            'recall': recall,
            'temporal_accuracy': temporal_accuracy,
            'confidence_calibration': abs(prediction.confidence_score - precision),
            'matched_events': matched_predictions,
            'total_predicted': len(prediction.predicted_events),
            'total_actual': len(actual_events)
        }
    
    def _calculate_event_similarity(self, event1: TemporalEvent, event2: TemporalEvent) -> float:
        """두 이벤트 간 유사도 계산"""
        similarity = 0.0
        
        # 이벤트 타입
        if event1.event_type == event2.event_type:
            similarity += 0.4
        
        # 강도 유사성
        intensity_sim = 1.0 - abs(event1.intensity - event2.intensity)
        similarity += intensity_sim * 0.2
        
        # 범위 유사성
        scope_sim = 1.0 - abs(event1.scope - event2.scope)
        similarity += scope_sim * 0.2
        
        # 시간적 근접성
        time_diff = abs(event1.timestamp - event2.timestamp)
        time_sim = math.exp(-time_diff / 300)  # 5분 기준 지수 감쇠
        similarity += time_sim * 0.2
        
        return min(similarity, 1.0)
    
    def _update_patterns_incrementally(self, new_event: TemporalEvent):
        """증분적 패턴 업데이트"""
        try:
            # 최근 이벤트들과의 연관성 체크
            recent_events = list(self.events_database)[-20:]
            
            for recent_event in recent_events:
                if recent_event.event_id == new_event.event_id:
                    continue
                
                relevance = self.pattern_detector._calculate_event_relevance(recent_event, new_event)
                
                if relevance > 0.6:
                    # 새로운 전파 경로 발견
                    path_id = f"{recent_event.event_id}_to_{new_event.event_id}"
                    propagation_delay = new_event.timestamp - recent_event.timestamp
                    
                    if propagation_delay > 0:  # 순서가 맞는 경우
                        self.propagation_paths[path_id] = PropagationPath(
                            path_id=path_id,
                            source_event=recent_event.event_id,
                            target_event=new_event.event_id,
                            propagation_delay=propagation_delay,
                            transmission_strength=relevance,
                            learned_weight=1.0,
                            confidence=relevance
                        )
        
        except Exception as e:
            self.logger.error(f"패턴 업데이트 오류: {e}")
    
    def _learn_propagation_paths(self, new_event: TemporalEvent):
        """전파 경로 학습"""
        # 신경망 학습을 위한 데이터 준비
        if len(self.events_database) < 50:
            return
        
        try:
            # 학습 데이터 준비
            training_sequences = self._prepare_training_data(TemporalScale.SHORT_TERM)
            
            if len(training_sequences) < 5:
                return
            
            # 미니배치 학습
            batch_size = min(len(training_sequences), 8)
            batch = torch.stack(training_sequences[:batch_size])
            
            # 타겟 데이터 (다음 이벤트)
            target_event_features = self._event_to_features(new_event)
            target_timing = torch.tensor([300.0], dtype=torch.float32).to(DEVICE)  # 기본 5분
            target_confidence = torch.tensor([new_event.certainty_level], dtype=torch.float32).to(DEVICE)
            
            # 순전파
            self.learning_network.train()
            pred_features, pred_timing, pred_confidence = self.learning_network(batch)
            
            # 손실 계산
            feature_loss = nn.MSELoss()(pred_features[0], target_event_features)
            timing_loss = nn.MSELoss()(pred_timing[0], target_timing)
            confidence_loss = nn.MSELoss()(pred_confidence[0], target_confidence)
            
            total_loss = feature_loss + timing_loss + confidence_loss
            
            # 역전파
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            self.logger.debug(f"신경망 학습 완료: loss={total_loss.item():.4f}")
            
        except Exception as e:
            self.logger.error(f"전파 경로 학습 오류: {e}")
    
    def get_analytics_dashboard(self) -> Dict[str, Any]:
        """분석 대시보드 데이터"""
        
        current_time = time.time()
        
        # 기본 통계
        total_events = len(self.events_database)
        recent_events = [e for e in self.events_database if current_time - e.timestamp <= 3600]
        
        # 성능 통계
        accuracy_rate = (
            self.performance_stats['accurate_predictions'] / 
            max(self.performance_stats['total_predictions'], 1)
        )
        
        # 패턴 분석
        all_patterns = self.analyze_temporal_patterns()
        
        # 예측 성능
        recent_errors = list(self.performance_stats['prediction_errors'])[-10:]
        error_trend = np.mean(recent_errors) if recent_errors else 0.5
        
        dashboard = {
            'overview': {
                'total_events': total_events,
                'recent_events_1h': len(recent_events),
                'total_patterns': len(self.propagation_paths),
                'accuracy_rate': accuracy_rate,
                'error_trend': error_trend
            },
            'temporal_patterns': all_patterns,
            'performance_metrics': {
                'total_predictions': self.performance_stats['total_predictions'],
                'accurate_predictions': self.performance_stats['accurate_predictions'],
                'average_confidence': self.performance_stats['average_confidence'],
                'recent_error_rate': error_trend
            },
            'propagation_analysis': {
                'total_paths': len(self.propagation_paths),
                'strongest_paths': self._get_strongest_propagation_paths(5),
                'average_delay': self._calculate_average_propagation_delay()
            },
            'prediction_cache': {
                'cached_predictions': len(self.prediction_cache),
                'cache_hit_potential': min(len(self.prediction_cache) / 10.0, 1.0)
            }
        }
        
        return dashboard
    
    def _get_strongest_propagation_paths(self, count: int) -> List[Dict[str, Any]]:
        """가장 강한 전파 경로들"""
        if not self.propagation_paths:
            return []
        
        sorted_paths = sorted(
            self.propagation_paths.values(),
            key=lambda p: p.transmission_strength * p.confidence,
            reverse=True
        )
        
        return [
            {
                'path_id': path.path_id,
                'source': path.source_event,
                'target': path.target_event,
                'strength': path.transmission_strength,
                'delay': path.propagation_delay,
                'confidence': path.confidence
            }
            for path in sorted_paths[:count]
        ]
    
    def _calculate_average_propagation_delay(self) -> float:
        """평균 전파 지연시간 계산"""
        if not self.propagation_paths:
            return 0.0
        
        delays = [path.propagation_delay for path in self.propagation_paths.values()]
        return float(np.mean(delays))
    
    def _calculate_event_frequency(self, events: List[TemporalEvent], window_size: float) -> Dict[str, float]:
        """이벤트 빈도 계산"""
        type_counts = defaultdict(int)
        
        for event in events:
            type_counts[event.event_type] += 1
        
        # 시간당 빈도로 변환
        hours = window_size / 3600.0
        frequencies = {
            event_type: count / hours 
            for event_type, count in type_counts.items()
        }
        
        return frequencies
    
    def _analyze_intensity_trends(self, events: List[TemporalEvent]) -> Dict[str, float]:
        """강도 트렌드 분석"""
        if len(events) < 5:
            return {'trend': 0.0, 'volatility': 0.0}
        
        intensities = [event.intensity for event in sorted(events, key=lambda e: e.timestamp)]
        
        # 선형 트렌드
        x = np.arange(len(intensities))
        trend_slope = np.polyfit(x, intensities, 1)[0] if len(intensities) > 1 else 0.0
        
        # 변동성
        volatility = np.std(intensities)
        
        return {
            'trend': float(trend_slope),
            'volatility': float(volatility),
            'average_intensity': float(np.mean(intensities)),
            'max_intensity': float(np.max(intensities)),
            'min_intensity': float(np.min(intensities))
        }
    
    def _analyze_type_transitions(self, events: List[TemporalEvent]) -> Dict[str, Dict[str, float]]:
        """이벤트 타입 전환 분석"""
        transitions = defaultdict(lambda: defaultdict(int))
        total_transitions = defaultdict(int)
        
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        for i in range(len(sorted_events) - 1):
            current_type = sorted_events[i].event_type
            next_type = sorted_events[i + 1].event_type
            
            transitions[current_type][next_type] += 1
            total_transitions[current_type] += 1
        
        # 확률로 변환
        transition_probabilities = {}
        for source_type, targets in transitions.items():
            total = total_transitions[source_type]
            if total > 0:
                transition_probabilities[source_type] = {
                    target_type: count / total
                    for target_type, count in targets.items()
                }
        
        return transition_probabilities
    
    def _analyze_emotion_ethics_correlation(self, events: List[TemporalEvent]) -> Dict[str, float]:
        """감정-윤리 상관관계 분석"""
        emotion_ethics_pairs = []
        
        for event in events:
            if event.emotion_state and event.ethical_implications:
                # 감정 강도 계산
                emotion_intensity = np.mean([
                    getattr(event.emotion_state, attr, 0.0)
                    for attr in ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust']
                ])
                
                # 윤리적 중요도 계산
                ethics_importance = np.mean(list(event.ethical_implications.values()))
                
                emotion_ethics_pairs.append((emotion_intensity, ethics_importance))
        
        if len(emotion_ethics_pairs) < 3:
            return {'correlation': 0.0, 'sample_count': len(emotion_ethics_pairs)}
        
        emotions, ethics = zip(*emotion_ethics_pairs)
        correlation = np.corrcoef(emotions, ethics)[0, 1]
        
        return {
            'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
            'sample_count': len(emotion_ethics_pairs),
            'average_emotion_intensity': float(np.mean(emotions)),
            'average_ethics_importance': float(np.mean(ethics))
        }
    
    def _rule_based_prediction(self, target_time: float, scale: TemporalScale) -> TemporalPrediction:
        """규칙 기반 예측 (데이터 부족 시)"""
        current_time = time.time()
        
        # 기본적인 규칙 기반 예측
        predicted_events = []
        
        # 최근 이벤트 패턴 기반
        recent_events = list(self.events_database)[-10:]
        if recent_events:
            # 가장 빈번한 이벤트 타입
            type_counts = defaultdict(int)
            for event in recent_events:
                type_counts[event.event_type] += 1
            
            most_common_type = max(type_counts.keys(), key=type_counts.get)
            
            # 간단한 예측 이벤트 생성
            predicted_event = TemporalEvent(
                event_id=f"rule_based_{int(target_time)}",
                timestamp=current_time + (target_time - current_time) / 2,
                event_type=most_common_type,
                description=f"Rule-based prediction of {most_common_type}",
                intensity=0.5,
                scope=0.4,
                reversibility=0.6,
                certainty_level=0.3,
                prediction_confidence=0.3
            )
            
            predicted_events.append(predicted_event)
        
        return TemporalPrediction(
            prediction_id=f"rule_based_{int(target_time)}",
            target_time=target_time,
            predicted_events=predicted_events,
            confidence_score=0.3,
            contributing_patterns=['rule_based_fallback']
        )
    
    def _conservative_prediction(self, target_time: float, scale: TemporalScale) -> TemporalPrediction:
        """보수적 예측 (신뢰도 부족 시)"""
        # 매우 보수적인 예측 - 불확실성 명시
        return TemporalPrediction(
            prediction_id=f"conservative_{int(target_time)}",
            target_time=target_time,
            predicted_events=[],
            confidence_score=0.1,
            uncertainty_range=(0.0, 1.0),
            contributing_patterns=['conservative_fallback'],
            most_likely_scenario={'description': 'High uncertainty - insufficient data for reliable prediction'}
        )
    
    def _empty_prediction(self) -> TemporalPrediction:
        """빈 예측 (오류 시)"""
        return TemporalPrediction(
            prediction_id=f"empty_{int(time.time())}",
            target_time=time.time(),
            predicted_events=[],
            confidence_score=0.0
        )
    
    def _invalidate_related_predictions(self, event: TemporalEvent):
        """관련 예측 캐시 무효화"""
        # 이벤트와 관련된 예측들을 캐시에서 제거
        keys_to_remove = []
        
        for cache_key in self.prediction_cache.keys():
            # 간단한 휴리스틱: 이벤트 타입이 포함된 예측들
            if event.event_type in cache_key or 'all' in cache_key:
                keys_to_remove.append(cache_key)
        
        for key in keys_to_remove:
            del self.prediction_cache[key]

    def save_state(self, filepath: str):
        """상태 저장"""
        state_data = {
            'events_count': len(self.events_database),
            'propagation_paths_count': len(self.propagation_paths),
            'performance_stats': self.performance_stats,
            'cache_size': len(self.prediction_cache),
            'analytics': self.get_analytics_dashboard()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"시계열 분석기 상태를 {filepath}에 저장 완료")


# 테스트 및 데모 함수
def test_temporal_analyzer():
    """시계열 분석기 테스트"""
    print("⏰ 시계열 사건 전파 분석기 테스트 시작")
    
    # 분석기 초기화
    analyzer = TemporalEventPropagationAnalyzer()
    
    # 테스트 이벤트 생성
    current_time = time.time()
    test_events = []
    
    # 주기적 이벤트 시뮬레이션
    for i in range(20):
        event = TemporalEvent(
            event_id=f"test_decision_{i}",
            timestamp=current_time - (20-i) * 300,  # 5분 간격
            event_type="decision",
            description=f"Test decision event {i}",
            intensity=0.6 + np.random.normal(0, 0.1),
            scope=0.5,
            reversibility=0.7,
            primary_actors=["user", "system"],
            affected_entities=["user", "community"],
            ethical_implications={
                'harm': np.random.uniform(0.2, 0.8),
                'fairness': np.random.uniform(0.3, 0.9)
            }
        )
        test_events.append(event)
    
    # 연쇄 반응 이벤트 시뮬레이션
    for i in range(10):
        base_time = current_time - (10-i) * 600  # 10분 간격
        
        # 시드 이벤트
        seed_event = TemporalEvent(
            event_id=f"seed_{i}",
            timestamp=base_time,
            event_type="external",
            description=f"External trigger {i}",
            intensity=0.8,
            scope=0.9,
            reversibility=0.3
        )
        test_events.append(seed_event)
        
        # 후속 이벤트들
        for j in range(3):
            follow_event = TemporalEvent(
                event_id=f"follow_{i}_{j}",
                timestamp=base_time + (j+1) * 60,  # 1분씩 지연
                event_type="consequence",
                description=f"Consequence {j} of {seed_event.event_id}",
                intensity=0.8 - j * 0.2,
                scope=0.7,
                reversibility=0.5,
                causal_antecedents=[seed_event.event_id]
            )
            test_events.append(follow_event)
    
    # 이벤트 등록
    print(f"📝 {len(test_events)}개 테스트 이벤트 등록 중...")
    for event in test_events:
        analyzer.register_event(event)
    
    # 패턴 분석
    print("\n🔍 시간적 패턴 분석")
    patterns = analyzer.analyze_temporal_patterns()
    
    for scale, pattern_data in patterns.items():
        if scale != 'cross_scale' and not pattern_data.get('insufficient_data'):
            print(f"\n--- {scale.upper()} 척도 ---")
            print(f"이벤트 수: {pattern_data['event_count']}")
            
            cyclic = pattern_data.get('cyclic_patterns', {})
            if cyclic:
                print(f"주기적 패턴: {len(cyclic)}개")
                for event_type, pattern_info in cyclic.items():
                    print(f"  {event_type}: 주기 {pattern_info['period']:.1f}초, "
                          f"주기성 {pattern_info['periodicity_score']:.3f}")
            
            cascade = pattern_data.get('cascade_patterns', [])
            if cascade:
                print(f"연쇄 반응 패턴: {len(cascade)}개")
                for i, casc in enumerate(cascade[:3]):  # 상위 3개만 표시
                    print(f"  패턴 {i+1}: 강도 {casc['cascade_strength']:.3f}, "
                          f"평균 지연 {casc['average_delay']:.1f}초")
    
    # 미래 예측
    print("\n🔮 미래 이벤트 예측")
    prediction_horizons = [300, 1800, 3600]  # 5분, 30분, 1시간
    
    for horizon in prediction_horizons:
        prediction = analyzer.predict_future_events(
            prediction_horizon=horizon,
            scale=TemporalScale.SHORT_TERM
        )
        
        print(f"\n--- {horizon//60}분 후 예측 ---")
        print(f"예측 이벤트 수: {len(prediction.predicted_events)}")
        print(f"신뢰도: {prediction.confidence_score:.3f}")
        print(f"기여 패턴: {', '.join(prediction.contributing_patterns)}")
        
        for i, pred_event in enumerate(prediction.predicted_events[:3]):  # 상위 3개
            time_until = pred_event.timestamp - current_time
            print(f"  이벤트 {i+1}: {pred_event.event_type} "
                  f"(+{time_until:.0f}초, 강도 {pred_event.intensity:.2f})")
    
    # 분석 대시보드
    print("\n📊 분석 대시보드")
    dashboard = analyzer.get_analytics_dashboard()
    
    overview = dashboard['overview']
    print(f"전체 이벤트: {overview['total_events']}")
    print(f"최근 1시간 이벤트: {overview['recent_events_1h']}")
    print(f"탐지된 패턴: {overview['total_patterns']}")
    print(f"예측 정확도: {overview['accuracy_rate']:.3f}")
    
    performance = dashboard['performance_metrics']
    print(f"총 예측 수: {performance['total_predictions']}")
    print(f"정확한 예측: {performance['accurate_predictions']}")
    
    propagation = dashboard['propagation_analysis']
    print(f"전파 경로: {propagation['total_paths']}")
    print(f"평균 전파 지연: {propagation['average_delay']:.1f}초")
    
    print("\n✅ 시계열 사건 전파 분석기 테스트 완료")
    
    return analyzer


if __name__ == "__main__":
    test_temporal_analyzer()
