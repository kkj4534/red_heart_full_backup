"""
고급 사용 패턴 분석기 - Week 2 스왑 최적화
Advanced Usage Pattern Analyzer - Week 2 Swap Optimization

머신러닝 기반 사용자 요청 패턴 분석 및 예측:
- LSTM 기반 시퀀스 예측 모델
- 시간대별/컨텍스트별 패턴 학습
- 실시간 적응형 예측 시스템
- 에러 기반 자가 보정 메커니즘
"""

import asyncio
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import json
import pickle
# pathlib 제거 - WSL 호환성을 위해 os.path 사용
import os
import threading
import queue
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
import joblib

# 핵심 시스템 임포트
from config import ADVANCED_CONFIG, MODELS_DIR, CACHE_DIR, get_gpu_memory_info
from head_compatibility_interface import HeadType

# 로거 설정
logger = logging.getLogger(__name__)

class PatternType(Enum):
    """패턴 타입 정의"""
    SEQUENTIAL = "sequential"      # 순차적 패턴 (A→B→C)
    CYCLIC = "cyclic"             # 순환적 패턴 (A→B→A→B)
    CONTEXTUAL = "contextual"     # 상황별 패턴 (시간/주제 의존)
    BURST = "burst"               # 집중 사용 패턴
    RANDOM = "random"             # 무작위 패턴

class ContextType(Enum):
    """컨텍스트 타입"""
    EMOTIONAL_ANALYSIS = "emotional"
    ETHICAL_REASONING = "ethical" 
    SEMANTIC_UNDERSTANDING = "semantic"
    REGRET_LEARNING = "regret"
    META_INTEGRATION = "meta"
    MIXED_USAGE = "mixed"

@dataclass
class UsageEvent:
    """사용 이벤트 데이터"""
    timestamp: datetime
    head_type: HeadType
    context_type: ContextType
    request_text: str
    processing_time: float
    memory_usage: float
    cache_hit: bool
    user_session_id: str = "default"
    text_length: int = 0
    complexity_score: float = 0.0
    
    def __post_init__(self):
        self.text_length = len(self.request_text)
        self.complexity_score = self._calculate_complexity()
    
    def _calculate_complexity(self) -> float:
        """텍스트 복잡도 계산"""
        text = self.request_text.lower()
        
        # 기본 복잡도 지표들
        word_count = len(text.split())
        char_count = len(text)
        unique_words = len(set(text.split()))
        
        # 복잡한 단어 패턴 감지
        complex_patterns = ['감정', '윤리', '도덕', '철학', '관계', '공감', '후회']
        pattern_count = sum(1 for pattern in complex_patterns if pattern in text)
        
        # 정규화된 복잡도 점수 (0-1)
        complexity = min(1.0, (
            word_count * 0.1 + 
            (unique_words / max(1, word_count)) * 0.3 +
            pattern_count * 0.4 +
            min(char_count / 1000, 1.0) * 0.2
        ))
        
        return complexity

@dataclass
class PredictionResult:
    """예측 결과"""
    predicted_heads: List[Tuple[HeadType, float]]  # (헤드, 확률)
    confidence_score: float
    predicted_time: datetime
    context_prediction: ContextType
    memory_requirement: float
    processing_time_estimate: float
    cache_strategy: str
    
class LSTMPatternPredictor(nn.Module):
    """LSTM 기반 패턴 예측 모델"""
    
    def __init__(self, input_dim: int = 20, hidden_dim: int = 128, 
                 num_layers: int = 2, num_heads: int = 5):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=0.2
        )
        
        # 어텐션 메커니즘
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, batch_first=True
        )
        
        # 예측 헤드들
        self.head_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_heads)
        )
        
        self.time_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.context_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, len(ContextType))
        )
        
        # 메모리 요구량 예측기
        self.memory_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # 0-1 범위로 정규화
        )
        
    def forward(self, x, hidden=None):
        """순전파"""
        batch_size = x.size(0)
        
        # LSTM 처리
        lstm_out, hidden = self.lstm(x, hidden)
        
        # 어텐션 적용
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 마지막 타임스텝 추출
        final_output = attn_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # 각 예측 헤드 실행
        head_logits = self.head_classifier(final_output)
        time_pred = self.time_predictor(final_output)
        context_logits = self.context_classifier(final_output)
        memory_pred = self.memory_predictor(final_output)
        
        return {
            'head_logits': head_logits,
            'time_prediction': time_pred,
            'context_logits': context_logits,
            'memory_prediction': memory_pred,
            'attention_weights': attn_weights,
            'hidden_state': hidden
        }

class AdvancedUsagePatternAnalyzer:
    """
    고급 사용 패턴 분석기
    
    머신러닝 기반 사용자 요청 패턴 분석 및 예측 시스템
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or ADVANCED_CONFIG.get('usage_pattern_config', {})
        
        # 데이터 저장소
        self.usage_history = deque(maxlen=10000)  # 최근 10,000개 이벤트
        self.session_patterns = defaultdict(list)  # 세션별 패턴
        self.temporal_patterns = defaultdict(list)  # 시간대별 패턴
        
        # 머신러닝 모델들
        self.lstm_predictor = None
        self.pattern_classifier = None
        self.complexity_estimator = None
        
        # 데이터 전처리기들
        self.feature_scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.context_encoder = LabelEncoder()
        
        # 패턴 분석 파라미터
        self.sequence_length = 10  # LSTM 입력 시퀀스 길이
        self.prediction_horizon = 5  # 예측 범위 (다음 5개 요청)
        self.min_pattern_length = 3  # 최소 패턴 길이
        
        # 실시간 예측 캐시
        self.prediction_cache = {}
        self.cache_timeout = 300  # 5분 캐시 유효기간
        
        # 모델 학습 설정
        self.model_retrain_interval = 1000  # 1000개 이벤트마다 재학습
        self.last_training_count = 0
        
        # 성능 메트릭
        self.prediction_accuracy = deque(maxlen=1000)
        self.cache_hit_rate = deque(maxlen=1000)
        
        # 파일 경로
        self.models_dir = os.path.join(MODELS_DIR, "usage_patterns")
        os.makedirs(self.models_dir, exist_ok=True)
        
        # 초기화
        self._initialize_models()
        self._load_models()
        
        logger.info("AdvancedUsagePatternAnalyzer 초기화 완료")
    
    def _initialize_models(self):
        """모델들 초기화"""
        # LSTM 예측 모델
        self.lstm_predictor = LSTMPatternPredictor(
            input_dim=20,  # 피처 수
            hidden_dim=128,
            num_layers=2,
            num_heads=len(HeadType)
        )
        
        # 패턴 분류기 (전통적 ML)
        self.pattern_classifier = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # 복잡도 추정기
        self.complexity_estimator = RandomForestRegressor(
            n_estimators=50,
            max_depth=8,
            random_state=42
        )
    
    def _load_models(self):
        """저장된 모델들 로드"""
        try:
            # LSTM 모델 로드
            lstm_path = os.path.join(self.models_dir, "lstm_predictor.pth")
            if os.path.exists(lstm_path):
                self.lstm_predictor.load_state_dict(torch.load(lstm_path, map_location='cpu'))
                logger.info("LSTM 예측 모델 로드 완료")
            
            # 전통적 ML 모델들 로드
            pattern_path = os.path.join(self.models_dir, "pattern_classifier.joblib")
            if os.path.exists(pattern_path):
                self.pattern_classifier = joblib.load(pattern_path)
                logger.info("패턴 분류기 로드 완료")
            
            complexity_path = os.path.join(self.models_dir, "complexity_estimator.joblib")
            if os.path.exists(complexity_path):
                self.complexity_estimator = joblib.load(complexity_path)
                logger.info("복잡도 추정기 로드 완료")
            
            # 전처리기들 로드
            scaler_path = os.path.join(self.models_dir, "feature_scaler.joblib")
            if os.path.exists(scaler_path):
                self.feature_scaler = joblib.load(scaler_path)
                
        except Exception as e:
            logger.warning(f"모델 로드 중 오류: {str(e)}")
    
    def _save_models(self):
        """모델들 저장"""
        try:
            # LSTM 모델 저장
            lstm_path = os.path.join(self.models_dir, "lstm_predictor.pth")
            torch.save(self.lstm_predictor.state_dict(), lstm_path)
            
            # 전통적 ML 모델들 저장
            joblib.dump(self.pattern_classifier, os.path.join(self.models_dir, "pattern_classifier.joblib"))
            joblib.dump(self.complexity_estimator, os.path.join(self.models_dir, "complexity_estimator.joblib"))
            joblib.dump(self.feature_scaler, os.path.join(self.models_dir, "feature_scaler.joblib"))
            
            logger.info("모든 모델 저장 완료")
            
        except Exception as e:
            logger.error(f"모델 저장 중 오류: {str(e)}")
    
    def record_usage_event(self, event: UsageEvent):
        """사용 이벤트 기록"""
        self.usage_history.append(event)
        self.session_patterns[event.user_session_id].append(event)
        
        # 시간대별 패턴 기록
        hour = event.timestamp.hour
        self.temporal_patterns[hour].append(event)
        
        # 주기적 모델 재학습 체크
        if len(self.usage_history) - self.last_training_count >= self.model_retrain_interval:
            asyncio.create_task(self._retrain_models())
            self.last_training_count = len(self.usage_history)
        
        logger.debug(f"사용 이벤트 기록: {event.head_type.value} @ {event.timestamp}")
    
    async def predict_next_requests(self, session_id: str = "default", 
                                  context: Optional[str] = None) -> List[PredictionResult]:
        """다음 요청들 예측"""
        try:
            # 캐시 확인
            cache_key = f"{session_id}_{context}_{datetime.now().strftime('%Y%m%d%H%M')}"
            if cache_key in self.prediction_cache:
                cache_entry = self.prediction_cache[cache_key]
                if datetime.now() - cache_entry['timestamp'] < timedelta(seconds=self.cache_timeout):
                    self.cache_hit_rate.append(1.0)
                    return cache_entry['predictions']
            
            self.cache_hit_rate.append(0.0)
            
            # 최근 패턴 분석
            recent_events = list(self.usage_history)[-self.sequence_length:]
            if len(recent_events) < self.min_pattern_length:
                return self._get_default_predictions()
            
            # 피처 추출
            features = self._extract_features(recent_events, context)
            
            # LSTM 기반 예측
            lstm_predictions = await self._lstm_predict(features)
            
            # 전통적 ML 기반 예측
            traditional_predictions = self._traditional_predict(features)
            
            # 앙상블 예측 결합
            combined_predictions = self._combine_predictions(
                lstm_predictions, traditional_predictions
            )
            
            # 결과 캐싱
            self.prediction_cache[cache_key] = {
                'predictions': combined_predictions,
                'timestamp': datetime.now()
            }
            
            return combined_predictions
            
        except Exception as e:
            logger.error(f"예측 중 오류: {str(e)}")
            return self._get_default_predictions()
    
    def _extract_features(self, events: List[UsageEvent], context: Optional[str] = None) -> np.ndarray:
        """이벤트들로부터 피처 추출"""
        if not events:
            return np.zeros((1, 20))
        
        features = []
        
        for i, event in enumerate(events):
            feature_vector = [
                # 시간 피처들
                event.timestamp.hour / 24.0,  # 시간 (정규화)
                event.timestamp.weekday() / 7.0,  # 요일 (정규화)
                
                # 헤드 타입 (원핫 인코딩)
                float(event.head_type == HeadType.EMOTION_EMPATHY),
                float(event.head_type == HeadType.BENTHAM_FROMM),
                float(event.head_type == HeadType.SEMANTIC_SURD),
                float(event.head_type == HeadType.REGRET_LEARNING),
                float(event.head_type == HeadType.META_INTEGRATION),
                
                # 컨텍스트 피처들
                float(event.context_type == ContextType.EMOTIONAL_ANALYSIS),
                float(event.context_type == ContextType.ETHICAL_REASONING),
                float(event.context_type == ContextType.SEMANTIC_UNDERSTANDING),
                float(event.context_type == ContextType.REGRET_LEARNING),
                float(event.context_type == ContextType.META_INTEGRATION),
                
                # 성능 피처들
                min(event.processing_time / 10.0, 1.0),  # 처리 시간 (정규화)
                min(event.memory_usage / 1000.0, 1.0),   # 메모리 사용량 (정규화)
                float(event.cache_hit),                   # 캐시 히트
                
                # 텍스트 피처들
                min(event.text_length / 1000.0, 1.0),   # 텍스트 길이 (정규화)
                event.complexity_score,                   # 복잡도 점수
                
                # 시퀀스 위치
                i / max(1, len(events) - 1),             # 시퀀스 내 위치
                
                # 간격 피처 (이전 이벤트와의 시간 간격)
                0.0 if i == 0 else min((event.timestamp - events[i-1].timestamp).seconds / 3600.0, 1.0)
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    async def _lstm_predict(self, features: np.ndarray) -> Dict[str, Any]:
        """LSTM 모델을 사용한 예측"""
        try:
            # 텐서 변환
            feature_tensor = torch.FloatTensor(features).unsqueeze(0)  # (1, seq_len, features)
            
            # 모델 예측
            self.lstm_predictor.eval()
            with torch.no_grad():
                output = self.lstm_predictor(feature_tensor)
            
            # 결과 해석
            head_probs = F.softmax(output['head_logits'], dim=-1).squeeze().numpy()
            context_probs = F.softmax(output['context_logits'], dim=-1).squeeze().numpy()
            
            return {
                'head_probabilities': head_probs,
                'context_probabilities': context_probs,
                'time_prediction': output['time_prediction'].item(),
                'memory_prediction': output['memory_prediction'].item(),
                'confidence': float(torch.max(F.softmax(output['head_logits'], dim=-1)).item())
            }
            
        except Exception as e:
            logger.error(f"LSTM 예측 오류: {str(e)}")
            return self._get_default_lstm_output()
    
    def _traditional_predict(self, features: np.ndarray) -> Dict[str, Any]:
        """전통적 ML 모델을 사용한 예측"""
        try:
            # 최근 피처만 사용 (마지막 이벤트)
            last_features = features[-1:] if len(features) > 0 else np.zeros((1, 20))
            
            # 패턴 분류
            if hasattr(self.pattern_classifier, 'predict'):
                pattern_prediction = self.pattern_classifier.predict(last_features)[0]
            else:
                pattern_prediction = 0.5
            
            # 복잡도 추정
            if hasattr(self.complexity_estimator, 'predict'):
                complexity_prediction = self.complexity_estimator.predict(last_features)[0]
            else:
                complexity_prediction = 0.5
            
            return {
                'pattern_score': pattern_prediction,
                'complexity_score': complexity_prediction,
                'confidence': 0.7  # 기본 신뢰도
            }
            
        except Exception as e:
            logger.error(f"전통적 ML 예측 오류: {str(e)}")
            return {'pattern_score': 0.5, 'complexity_score': 0.5, 'confidence': 0.5}
    
    def _combine_predictions(self, lstm_pred: Dict[str, Any], 
                           traditional_pred: Dict[str, Any]) -> List[PredictionResult]:
        """LSTM과 전통적 ML 예측 결과 결합"""
        try:
            results = []
            
            # LSTM 결과에서 상위 헤드들 추출
            head_types = list(HeadType)
            head_probs = lstm_pred.get('head_probabilities', np.ones(len(head_types)) / len(head_types))
            
            # 상위 3개 헤드 선택
            top_indices = np.argsort(head_probs)[-3:][::-1]
            
            for i, idx in enumerate(top_indices):
                head_type = head_types[idx]
                probability = float(head_probs[idx])
                
                # 컨텍스트 예측
                context_probs = lstm_pred.get('context_probabilities', np.ones(len(ContextType)) / len(ContextType))
                predicted_context = list(ContextType)[np.argmax(context_probs)]
                
                # 메모리 및 시간 예측
                base_memory = lstm_pred.get('memory_prediction', 0.5)
                complexity_factor = traditional_pred.get('complexity_score', 0.5)
                memory_requirement = min(base_memory * (1 + complexity_factor), 1.0)
                
                processing_time_estimate = 1.0 + complexity_factor * 2.0  # 1-3초 범위
                
                # 캐시 전략 결정
                cache_strategy = self._determine_cache_strategy(head_type, probability, complexity_factor)
                
                # 신뢰도 계산
                lstm_confidence = lstm_pred.get('confidence', 0.5)
                traditional_confidence = traditional_pred.get('confidence', 0.5)
                combined_confidence = (lstm_confidence * 0.7 + traditional_confidence * 0.3) * probability
                
                result = PredictionResult(
                    predicted_heads=[(head_type, probability)],
                    confidence_score=combined_confidence,
                    predicted_time=datetime.now() + timedelta(seconds=10 * (i + 1)),
                    context_prediction=predicted_context,
                    memory_requirement=memory_requirement,
                    processing_time_estimate=processing_time_estimate,
                    cache_strategy=cache_strategy
                )
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"예측 결합 오류: {str(e)}")
            return self._get_default_predictions()
    
    def _determine_cache_strategy(self, head_type: HeadType, probability: float, 
                                complexity: float) -> str:
        """캐시 전략 결정"""
        if probability > 0.8 and complexity < 0.3:
            return "aggressive_preload"
        elif probability > 0.6:
            return "standard_preload"
        elif complexity > 0.7:
            return "lazy_load"
        else:
            return "no_cache"
    
    def _get_default_predictions(self) -> List[PredictionResult]:
        """기본 예측 결과 반환"""
        default_heads = [HeadType.EMOTION_EMPATHY, HeadType.BENTHAM_FROMM, HeadType.SEMANTIC_SURD]
        results = []
        
        for i, head_type in enumerate(default_heads):
            result = PredictionResult(
                predicted_heads=[(head_type, 0.33)],
                confidence_score=0.5,
                predicted_time=datetime.now() + timedelta(seconds=10 * (i + 1)),
                context_prediction=ContextType.MIXED_USAGE,
                memory_requirement=0.5,
                processing_time_estimate=2.0,
                cache_strategy="standard_preload"
            )
            results.append(result)
        
        return results
    
    def _get_default_lstm_output(self) -> Dict[str, Any]:
        """기본 LSTM 출력"""
        num_heads = len(HeadType)
        num_contexts = len(ContextType)
        
        return {
            'head_probabilities': np.ones(num_heads) / num_heads,
            'context_probabilities': np.ones(num_contexts) / num_contexts,
            'time_prediction': 0.0,
            'memory_prediction': 0.5,
            'confidence': 0.5
        }
    
    async def _retrain_models(self):
        """모델들 재학습"""
        try:
            logger.info("사용 패턴 모델 재학습 시작...")
            
            if len(self.usage_history) < 100:  # 최소 데이터 요구량
                return
            
            # 학습 데이터 준비
            training_data = self._prepare_training_data()
            
            if training_data is None:
                return
            
            # 전통적 ML 모델 학습
            await self._train_traditional_models(training_data)
            
            # LSTM 모델 학습 (별도 스레드에서)
            asyncio.create_task(self._train_lstm_model(training_data))
            
            # 모델 저장
            self._save_models()
            
            logger.info("사용 패턴 모델 재학습 완료")
            
        except Exception as e:
            logger.error(f"모델 재학습 중 오류: {str(e)}")
    
    def _prepare_training_data(self) -> Optional[Dict[str, Any]]:
        """학습 데이터 준비"""
        try:
            events = list(self.usage_history)
            if len(events) < self.sequence_length + 1:
                return None
            
            sequences = []
            targets = []
            
            # 슬라이딩 윈도우로 시퀀스 생성
            for i in range(len(events) - self.sequence_length):
                sequence_events = events[i:i + self.sequence_length]
                target_event = events[i + self.sequence_length]
                
                sequence_features = self._extract_features(sequence_events)
                target_features = self._event_to_target(target_event)
                
                sequences.append(sequence_features)
                targets.append(target_features)
            
            return {
                'sequences': np.array(sequences),
                'targets': np.array(targets),
                'num_samples': len(sequences)
            }
            
        except Exception as e:
            logger.error(f"학습 데이터 준비 오류: {str(e)}")
            return None
    
    def _event_to_target(self, event: UsageEvent) -> Dict[str, Any]:
        """이벤트를 타겟 형태로 변환"""
        head_target = [0] * len(HeadType)
        head_target[list(HeadType).index(event.head_type)] = 1
        
        context_target = [0] * len(ContextType)
        context_target[list(ContextType).index(event.context_type)] = 1
        
        return {
            'head': head_target,
            'context': context_target,
            'memory': min(event.memory_usage / 1000.0, 1.0),
            'processing_time': min(event.processing_time / 10.0, 1.0)
        }
    
    async def _train_traditional_models(self, training_data: Dict[str, Any]):
        """전통적 ML 모델 학습"""
        try:
            sequences = training_data['sequences']
            targets = training_data['targets']
            
            # 마지막 시퀀스 스텝만 사용 (2D 형태로 변환)
            X = sequences[:, -1, :]  # (num_samples, features)
            
            # 패턴 분류기 학습 (메모리 사용량 예측)
            y_memory = np.array([t['memory'] for t in targets])
            self.pattern_classifier.fit(X, y_memory)
            
            # 복잡도 추정기 학습 (처리 시간 예측)
            y_time = np.array([t['processing_time'] for t in targets])
            self.complexity_estimator.fit(X, y_time)
            
            logger.info("전통적 ML 모델 학습 완료")
            
        except Exception as e:
            logger.error(f"전통적 ML 모델 학습 오류: {str(e)}")
    
    async def _train_lstm_model(self, training_data: Dict[str, Any]):
        """LSTM 모델 학습 (비동기)"""
        try:
            sequences = training_data['sequences']
            targets = training_data['targets']
            
            # 텐서 변환
            X = torch.FloatTensor(sequences)
            y_head = torch.FloatTensor([t['head'] for t in targets])
            y_context = torch.FloatTensor([t['context'] for t in targets])
            y_memory = torch.FloatTensor([t['memory'] for t in targets]).unsqueeze(1)
            
            # 옵티마이저 설정
            optimizer = torch.optim.Adam(self.lstm_predictor.parameters(), lr=0.001)
            criterion_ce = nn.CrossEntropyLoss()
            criterion_mse = nn.MSELoss()
            
            # 학습 루프
            self.lstm_predictor.train()
            for epoch in range(10):  # 간단한 학습
                optimizer.zero_grad()
                
                output = self.lstm_predictor(X)
                
                # 손실 계산
                head_loss = criterion_ce(output['head_logits'], y_head.argmax(dim=1))
                context_loss = criterion_ce(output['context_logits'], y_context.argmax(dim=1))
                memory_loss = criterion_mse(output['memory_prediction'], y_memory)
                
                total_loss = head_loss + context_loss + memory_loss
                
                total_loss.backward()
                optimizer.step()
                
                if epoch % 5 == 0:
                    logger.debug(f"LSTM 학습 Epoch {epoch}: Loss = {total_loss.item():.4f}")
            
            logger.info("LSTM 모델 학습 완료")
            
        except Exception as e:
            logger.error(f"LSTM 모델 학습 오류: {str(e)}")
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """패턴 분석 통계"""
        try:
            if not self.usage_history:
                return {'error': '데이터 없음'}
            
            # 기본 통계
            total_events = len(self.usage_history)
            unique_sessions = len(self.session_patterns)
            
            # 헤드 사용 빈도
            head_counts = defaultdict(int)
            for event in self.usage_history:
                head_counts[event.head_type.value] += 1
            
            # 시간대별 패턴
            hourly_counts = defaultdict(int)
            for event in self.usage_history:
                hourly_counts[event.timestamp.hour] += 1
            
            # 성능 지표
            avg_cache_hit_rate = np.mean(self.cache_hit_rate) if self.cache_hit_rate else 0.0
            avg_prediction_accuracy = np.mean(self.prediction_accuracy) if self.prediction_accuracy else 0.0
            
            return {
                'total_events': total_events,
                'unique_sessions': unique_sessions,
                'head_usage_frequency': dict(head_counts),
                'hourly_usage_pattern': dict(hourly_counts),
                'cache_hit_rate': avg_cache_hit_rate,
                'prediction_accuracy': avg_prediction_accuracy,
                'models_trained': hasattr(self.pattern_classifier, 'feature_importances_')
            }
            
        except Exception as e:
            logger.error(f"통계 계산 오류: {str(e)}")
            return {'error': str(e)}

# 사용 예시 함수
async def example_usage():
    """고급 사용 패턴 분석기 사용 예시"""
    analyzer = AdvancedUsagePatternAnalyzer()
    
    # 가상의 사용 이벤트들 기록
    test_events = [
        UsageEvent(
            timestamp=datetime.now() - timedelta(minutes=10),
            head_type=HeadType.EMOTION_EMPATHY,
            context_type=ContextType.EMOTIONAL_ANALYSIS,
            request_text="기분이 우울할 때 어떻게 해야 할까요?",
            processing_time=2.5,
            memory_usage=150.0,
            cache_hit=False
        ),
        UsageEvent(
            timestamp=datetime.now() - timedelta(minutes=5),
            head_type=HeadType.BENTHAM_FROMM,
            context_type=ContextType.ETHICAL_REASONING,
            request_text="이 선택이 윤리적으로 옳은 것일까요?",
            processing_time=3.2,
            memory_usage=200.0,
            cache_hit=True
        )
    ]
    
    # 이벤트들 기록
    for event in test_events:
        analyzer.record_usage_event(event)
    
    # 예측 수행
    predictions = await analyzer.predict_next_requests("test_session")
    
    print("=== 사용 패턴 예측 결과 ===")
    for i, pred in enumerate(predictions):
        print(f"\n예측 {i+1}:")
        print(f"  헤드: {pred.predicted_heads[0][0].value}")
        print(f"  확률: {pred.predicted_heads[0][1]:.3f}")
        print(f"  신뢰도: {pred.confidence_score:.3f}")
        print(f"  예상 메모리: {pred.memory_requirement:.2f}")
        print(f"  캐시 전략: {pred.cache_strategy}")
    
    # 통계 출력
    stats = analyzer.get_pattern_statistics()
    print(f"\n=== 패턴 분석 통계 ===")
    print(f"총 이벤트: {stats['total_events']}")
    print(f"캐시 히트율: {stats['cache_hit_rate']:.2%}")

if __name__ == "__main__":
    asyncio.run(example_usage())