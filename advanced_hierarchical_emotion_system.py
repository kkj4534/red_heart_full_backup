"""
고급 계층적 감정 시스템 - Linux 전용
Advanced Hierarchical Emotion System for Linux

3단계 감정 학습 시스템:
- Phase 0: 타자 감정 → 자신 (감정 캘리브레이션)
- Phase 1: 자신 → 타인 (후회 기반 타자 경험 학습)
- Phase 2: 타인 → 공동체 (대중 감정 패턴 이해)
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
import uuid
from collections import deque, defaultdict
from enum import Enum
import pickle

# 고급 모델 임포트
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    from sentence_transformers import SentenceTransformer
    ADVANCED_MODELS_AVAILABLE = True
except ImportError:
    ADVANCED_MODELS_AVAILABLE = False
    logging.warning("고급 모델 라이브러리를 사용할 수 없습니다. 기본 모드로 실행됩니다.")

# 시스템 설정
from config import ADVANCED_CONFIG, CACHE_DIR, MODELS_DIR, LOGS_DIR

# 데이터 모델 import 추가
from data_models import HierarchicalEmpathyResult, SelfReflectionData, EmpathySimulationData

# SURD 시스템 통합
try:
    from advanced_surd_analyzer import AdvancedSURDAnalyzer, InformationMeasures
    SURD_AVAILABLE = True
except ImportError:
    SURD_AVAILABLE = False
    logging.warning("SURD 분석기를 불러올 수 없습니다. 불확실성 전파 기능이 비활성화됩니다.")

# 로깅 설정
logger = logging.getLogger(__name__)

# 누락된 신경망 모델들 import
try:
    from missing_neural_models import (
        SelfOtherNeuralNetwork,
        SimpleFallbackClassifier,
        IncrementalLearner,
        SimpleFallbackLearner,
        HierarchicalPatternStructure,
        SimpleFallbackManager,
        AdvancedFeatureExtractor,
        HierarchicalPatternClustering,
        PatternRelationshipGraph,
        PatternDiscriminator,
        AdvancedNoveltyDetector,
        PatternEvolutionTracker,
        MetaClassifier,
        PatternPredictor,
        PatternConsolidationSystem,
        SelectiveForgettingSystem
    )
    NEURAL_MODELS_AVAILABLE = True
    logger.info("누락된 신경망 모델들 import 성공")
except ImportError as e:
    NEURAL_MODELS_AVAILABLE = False
    logger.error(f"누락된 신경망 모델들 import 실패: {e}")
    raise ImportError(f"필수 신경망 모델들을 찾을 수 없습니다: {e}") from e

class EmotionPhase(Enum):
    """감정 학습 단계"""
    CALIBRATION = "phase_0_calibration"  # 타자→자신 투영
    EMPATHY_LEARNING = "phase_1_empathy"  # 자신→타인 학습
    COMMUNITY_EXPANSION = "phase_2_community"  # 타인→공동체 확장

class EmotionDimension(Enum):
    """감정 차원"""
    VALENCE = "valence"  # 긍정-부정
    AROUSAL = "arousal"  # 각성-이완
    DOMINANCE = "dominance"  # 지배-복종
    CERTAINTY = "certainty"  # 확실-불확실
    SURPRISE = "surprise"  # 예상-놀람
    ANTICIPATION = "anticipation"  # 기대-무관심

@dataclass
class EmotionVector:
    """다차원 감정 벡터"""
    dimensions: Dict[EmotionDimension, float]
    intensity: float = 1.0
    confidence: float = 1.0
    source: str = "self"  # self, other, community
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_array(self) -> np.ndarray:
        """numpy 배열로 변환"""
        return np.array([self.dimensions.get(dim, 0.0) for dim in EmotionDimension])
    
    def distance(self, other: 'EmotionVector') -> float:
        """다른 감정 벡터와의 거리"""
        return np.linalg.norm(self.to_array() - other.to_array())

@dataclass
class EmotionCalibrationData:
    """Phase 0: 감정 캘리브레이션 데이터"""
    other_emotion: EmotionVector  # 타자의 감정
    projected_self_emotion: EmotionVector  # 자신에게 투영된 감정
    context: Dict[str, Any]  # 상황 맥락
    calibration_factor: float = 1.0  # 투영 보정 계수
    literary_source: str = ""  # 문학 데이터 출처
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class EmpathyLearningData:
    """Phase 1: 공감 학습 데이터"""
    self_emotion: EmotionVector  # 자신의 감정
    predicted_other_emotion: EmotionVector  # 예측한 타인의 감정
    actual_other_emotion: EmotionVector  # 실제 타인의 감정
    regret_intensity: float = 0.0  # 예측 오차에 대한 후회
    learning_rate: float = 0.1  # 학습률
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class CommunityEmotionPattern:
    """Phase 2: 공동체 감정 패턴"""
    emotion_distribution: Dict[str, EmotionVector]  # 구성원별 감정
    collective_emotion: EmotionVector  # 집단 감정
    consensus_level: float = 0.0  # 감정 일치도
    cultural_factors: Dict[str, float] = field(default_factory=dict)
    temporal_dynamics: List[EmotionVector] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

class Phase0EmotionCalibrator:
    """Phase 0: 타자 감정 → 자신 투영 캘리브레이터"""
    
    def __init__(self):
        self.calibration_history = deque(maxlen=1000)
        self.projection_models = {}
        self.literary_emotion_db = self._load_literary_emotions()
        self.calibration_factors = defaultdict(lambda: 1.0)
        
        # 임베딩 모델
        self.embedding_model = None
        self._initialize_embedding_model()
        
        logger.info("Phase 0 감정 캘리브레이터가 초기화되었습니다.")
    
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
    
    def _load_literary_emotions(self) -> Dict[str, List[EmotionVector]]:
        """문학 작품의 감정 데이터 로드"""
        return {
            'tragedy': [
                EmotionVector(
                    dimensions={
                        EmotionDimension.VALENCE: -0.8,
                        EmotionDimension.AROUSAL: 0.6,
                        EmotionDimension.DOMINANCE: -0.5,
                        EmotionDimension.CERTAINTY: 0.3,
                        EmotionDimension.SURPRISE: 0.7,
                        EmotionDimension.ANTICIPATION: -0.4
                    },
                    source='other'
                )
            ],
            'comedy': [
                EmotionVector(
                    dimensions={
                        EmotionDimension.VALENCE: 0.8,
                        EmotionDimension.AROUSAL: 0.7,
                        EmotionDimension.DOMINANCE: 0.3,
                        EmotionDimension.CERTAINTY: 0.6,
                        EmotionDimension.SURPRISE: 0.5,
                        EmotionDimension.ANTICIPATION: 0.4
                    },
                    source='other'
                )
            ],
            'romance': [
                EmotionVector(
                    dimensions={
                        EmotionDimension.VALENCE: 0.7,
                        EmotionDimension.AROUSAL: 0.8,
                        EmotionDimension.DOMINANCE: 0.0,
                        EmotionDimension.CERTAINTY: -0.2,
                        EmotionDimension.SURPRISE: 0.3,
                        EmotionDimension.ANTICIPATION: 0.9
                    },
                    source='other'
                )
            ]
        }
    
    async def calibrate_emotion(self, 
                              other_emotion: EmotionVector,
                              context: Dict[str, Any],
                              literary_reference: str = None) -> EmotionCalibrationData:
        """타자의 감정을 자신에게 투영하여 캘리브레이션"""
        try:
            # 문맥 기반 투영 계수 계산
            calibration_factor = await self._calculate_calibration_factor(
                other_emotion, context, literary_reference
            )
            
            # 감정 차원별 투영
            projected_dimensions = {}
            for dim in EmotionDimension:
                other_value = other_emotion.dimensions.get(dim, 0.0)
                
                # 개인차를 고려한 비선형 투영
                if dim == EmotionDimension.VALENCE:
                    # 긍정/부정은 개인 성향에 따라 다르게 투영
                    projected_value = self._nonlinear_projection(
                        other_value, 
                        context.get('personal_valence_bias', 0.0)
                    )
                elif dim == EmotionDimension.AROUSAL:
                    # 각성도는 상황에 따라 증폭/감소
                    situation_modifier = context.get('arousal_modifier', 1.0)
                    projected_value = other_value * situation_modifier
                else:
                    projected_value = other_value * calibration_factor
                
                projected_dimensions[dim] = np.clip(projected_value, -1.0, 1.0)
            
            # 투영된 감정 생성
            projected_emotion = EmotionVector(
                dimensions=projected_dimensions,
                intensity=other_emotion.intensity * calibration_factor,
                confidence=0.8,  # 투영은 항상 불확실성 포함
                source='self'
            )
            
            # 캘리브레이션 데이터 생성
            calibration_data = EmotionCalibrationData(
                other_emotion=other_emotion,
                projected_self_emotion=projected_emotion,
                context=context,
                calibration_factor=calibration_factor,
                literary_source=literary_reference or ""
            )
            
            # 히스토리 저장
            self.calibration_history.append(calibration_data)
            
            # 캘리브레이션 계수 업데이트
            context_key = self._get_context_key(context)
            self.calibration_factors[context_key] = (
                0.9 * self.calibration_factors[context_key] + 
                0.1 * calibration_factor
            )
            
            logger.debug(f"감정 캘리브레이션 완료: {calibration_factor:.3f}")
            return calibration_data
            
        except Exception as e:
            logger.error(f"감정 캘리브레이션 실패: {e}")
            return EmotionCalibrationData(
                other_emotion=other_emotion,
                projected_self_emotion=other_emotion,  # 실패 시 그대로 반환
                context=context,
                calibration_factor=1.0
            )
    
    def _nonlinear_projection(self, value: float, bias: float) -> float:
        """비선형 감정 투영"""
        # 시그모이드 기반 비선형 변환
        biased_value = value + bias
        return 2.0 / (1.0 + np.exp(-2.0 * biased_value)) - 1.0
    
    async def _calculate_calibration_factor(self,
                                          other_emotion: EmotionVector,
                                          context: Dict[str, Any],
                                          literary_reference: str) -> float:
        """캘리브레이션 계수 계산"""
        base_factor = 1.0
        
        # 문학적 참조가 있으면 유사도 기반 조정
        if literary_reference and literary_reference in self.literary_emotion_db:
            ref_emotions = self.literary_emotion_db[literary_reference]
            similarities = [
                1.0 - other_emotion.distance(ref) / 2.0 
                for ref in ref_emotions
            ]
            base_factor *= np.mean(similarities)
        
        # 문맥 기반 조정
        if 'cultural_distance' in context:
            base_factor *= (1.0 - context['cultural_distance'] * 0.3)
        
        if 'personal_similarity' in context:
            base_factor *= (0.5 + context['personal_similarity'] * 0.5)
        
        return np.clip(base_factor, 0.3, 1.5)
    
    def _get_context_key(self, context: Dict[str, Any]) -> str:
        """문맥을 키로 변환"""
        key_parts = []
        for k in sorted(['situation_type', 'cultural_context', 'relationship']):
            if k in context:
                key_parts.append(f"{k}:{context[k]}")
        return "|".join(key_parts)

class Phase1EmpathyLearner:
    """Phase 1: 자신 → 타인 공감 학습기"""
    
    def __init__(self, regret_threshold: float = 0.3):
        self.learning_history = deque(maxlen=2000)
        self.empathy_models = {}
        self.regret_threshold = regret_threshold
        self.learning_patterns = defaultdict(list)
        
        # 후회 기반 학습 파라미터
        self.regret_weights = {
            'prediction_error': 0.5,
            'emotional_distance': 0.3,
            'context_mismatch': 0.2
        }
        
        # 신경망 모델 (가능한 경우)
        self.neural_empathy_model = self._initialize_neural_model()
        
        logger.info("Phase 1 공감 학습기가 초기화되었습니다.")
    
    def _initialize_neural_model(self):
        """신경망 기반 공감 모델 초기화"""
        if not ADVANCED_MODELS_AVAILABLE:
            return None
        
        try:
            class EmpathyNet(nn.Module):
                def __init__(self, input_dim=768, hidden_dim=256, output_dim=6):
                    # 기본 input_dim을 768로 설정 (임베딩 차원)
                    # 실제 사용 시 적절한 차원으로 조정됨
                    super().__init__()
                    self.fc1 = nn.Linear(input_dim, hidden_dim)
                    self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
                    self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
                    self.dropout = nn.Dropout(0.2)
                    self.layer_norm = nn.LayerNorm(hidden_dim)
                
                def forward(self, x):
                    # 입력 차원 체크 및 자동 조정
                    if x.dim() == 3:  # batch, seq, feature
                        x = x.mean(dim=1)  # 시퀀스 차원 평균
                    elif x.dim() == 1:  # feature only
                        x = x.unsqueeze(0)  # batch 차원 추가
                        
                    x = F.relu(self.fc1(x))
                    x = self.layer_norm(x)
                    x = self.dropout(x)
                    x = F.relu(self.fc2(x))
                    x = self.dropout(x)
                    x = self.fc3(x)
                    return torch.tanh(x)  # -1 to 1 범위
            
            # 적절한 차원으로 초기화
            return EmpathyNet(input_dim=768, hidden_dim=256, output_dim=6)
        except Exception as e:
            logger.warning(f"신경망 모델 초기화 실패: {e}")
            return None
    
    async def learn_empathy(self,
                          self_emotion: EmotionVector,
                          predicted_other: EmotionVector,
                          actual_other: EmotionVector,
                          context: Dict[str, Any]) -> EmpathyLearningData:
        """공감 학습 - 예측 오차를 통한 타자 이해"""
        try:
            # 예측 오차 계산
            prediction_error = predicted_other.distance(actual_other)
            
            # 후회 강도 계산
            regret_intensity = await self._calculate_regret_intensity(
                self_emotion, predicted_other, actual_other, context
            )
            
            # 학습률 동적 조정
            learning_rate = self._adaptive_learning_rate(
                prediction_error, regret_intensity
            )
            
            # 공감 모델 업데이트
            if regret_intensity > self.regret_threshold:
                await self._update_empathy_model(
                    self_emotion, actual_other, context, learning_rate
                )
                
                # 패턴 저장
                pattern_key = self._extract_pattern_key(context)
                self.learning_patterns[pattern_key].append({
                    'error': prediction_error,
                    'regret': regret_intensity,
                    'timestamp': datetime.now()
                })
            
            # 학습 데이터 생성
            learning_data = EmpathyLearningData(
                self_emotion=self_emotion,
                predicted_other_emotion=predicted_other,
                actual_other_emotion=actual_other,
                regret_intensity=regret_intensity,
                learning_rate=learning_rate,
                context=context
            )
            
            # 히스토리 저장
            self.learning_history.append(learning_data)
            
            logger.debug(f"공감 학습 완료: 오차={prediction_error:.3f}, 후회={regret_intensity:.3f}")
            return learning_data
            
        except Exception as e:
            logger.error(f"공감 학습 실패: {e}")
            return EmpathyLearningData(
                self_emotion=self_emotion,
                predicted_other_emotion=predicted_other,
                actual_other_emotion=actual_other,
                regret_intensity=0.0,
                learning_rate=0.0
            )
    
    async def _calculate_regret_intensity(self,
                                        self_emotion: EmotionVector,
                                        predicted: EmotionVector,
                                        actual: EmotionVector,
                                        context: Dict[str, Any]) -> float:
        """후회 강도 계산"""
        # 예측 오차
        prediction_error = predicted.distance(actual)
        
        # 감정 거리 (자신과 타인의 감정 차이)
        emotional_distance = self_emotion.distance(actual)
        
        # 문맥 불일치도
        context_mismatch = 0.0
        if 'expected_reaction' in context and 'actual_reaction' in context:
            if context['expected_reaction'] != context['actual_reaction']:
                context_mismatch = 0.5
        
        # 가중합으로 후회 강도 계산
        regret = (
            self.regret_weights['prediction_error'] * prediction_error +
            self.regret_weights['emotional_distance'] * emotional_distance +
            self.regret_weights['context_mismatch'] * context_mismatch
        )
        
        # 상황의 중요도에 따라 스케일링
        importance = context.get('situation_importance', 1.0)
        regret *= importance
        
        return np.clip(regret, 0.0, 1.0)
    
    def _adaptive_learning_rate(self, error: float, regret: float) -> float:
        """적응적 학습률 계산"""
        # 오차와 후회가 클수록 더 많이 학습
        base_rate = 0.1
        error_factor = 1.0 + error
        regret_factor = 1.0 + regret * 2.0
        
        return base_rate * error_factor * regret_factor
    
    async def _update_empathy_model(self,
                                  self_emotion: EmotionVector,
                                  target_emotion: EmotionVector,
                                  context: Dict[str, Any],
                                  learning_rate: float):
        """공감 모델 업데이트"""
        context_key = self._extract_pattern_key(context)
        
        if context_key not in self.empathy_models:
            self.empathy_models[context_key] = {
                'transformation_matrix': np.eye(6),
                'bias_vector': np.zeros(6),
                'confidence': 0.0
            }
        
        model = self.empathy_models[context_key]
        
        # 현재 변환 적용
        self_array = self_emotion.to_array()
        predicted = np.dot(model['transformation_matrix'], self_array) + model['bias_vector']
        target_array = target_emotion.to_array()
        
        # 그래디언트 계산
        error = target_array - predicted
        
        # 모델 파라미터 업데이트
        model['transformation_matrix'] += learning_rate * np.outer(error, self_array)
        model['bias_vector'] += learning_rate * error
        model['confidence'] = 0.9 * model['confidence'] + 0.1
        
        # 신경망 모델 업데이트 (가능한 경우)
        if self.neural_empathy_model and ADVANCED_MODELS_AVAILABLE:
            await self._update_neural_model(
                self_array, target_array, learning_rate
            )
    
    async def _update_neural_model(self, 
                                 self_emotion: np.ndarray,
                                 target_emotion: np.ndarray,
                                 learning_rate: float):
        """신경망 모델 업데이트"""
        try:
            # 간단한 경사하강법 업데이트
            input_tensor = torch.FloatTensor(
                np.concatenate([self_emotion, self_emotion])  # 컨텍스트 포함
            )
            target_tensor = torch.FloatTensor(target_emotion)
            
            # Forward pass
            output = self.neural_empathy_model(input_tensor.unsqueeze(0))
            loss = F.mse_loss(output.squeeze(), target_tensor)
            
            # Backward pass
            loss.backward()
            
            # 수동 그래디언트 업데이트
            with torch.no_grad():
                for param in self.neural_empathy_model.parameters():
                    if param.grad is not None:
                        param -= learning_rate * param.grad
                        param.grad.zero_()
            
        except Exception as e:
            logger.error(f"신경망 모델 업데이트 실패: {e}")
    
    def _extract_pattern_key(self, context: Dict[str, Any]) -> str:
        """컨텍스트에서 패턴 키 추출"""
        key_elements = []
        
        # 관계 유형
        if 'relationship_type' in context:
            key_elements.append(f"rel:{context['relationship_type']}")
        
        # 상황 유형
        if 'situation_type' in context:
            key_elements.append(f"sit:{context['situation_type']}")
        
        # 문화적 맥락
        if 'cultural_context' in context:
            key_elements.append(f"cul:{context['cultural_context']}")
        
        return "|".join(key_elements) if key_elements else "default"
    
    async def predict_other_emotion(self,
                                  self_emotion: EmotionVector,
                                  context: Dict[str, Any]) -> EmotionVector:
        """학습된 모델로 타인의 감정 예측"""
        context_key = self._extract_pattern_key(context)
        
        if context_key in self.empathy_models:
            model = self.empathy_models[context_key]
            self_array = self_emotion.to_array()
            
            # 선형 변환 적용
            predicted_array = np.dot(
                model['transformation_matrix'], 
                self_array
            ) + model['bias_vector']
            
            # 범위 제한
            predicted_array = np.clip(predicted_array, -1.0, 1.0)
            
            # EmotionVector로 변환
            predicted_dimensions = {
                dim: predicted_array[i] 
                for i, dim in enumerate(EmotionDimension)
            }
            
            return EmotionVector(
                dimensions=predicted_dimensions,
                intensity=self_emotion.intensity * 0.8,
                confidence=model['confidence'],
                source='other'
            )
        else:
            # 학습되지 않은 경우 기본 예측
            return self._default_prediction(self_emotion)
    
    def _default_prediction(self, self_emotion: EmotionVector) -> EmotionVector:
        """기본 감정 예측 (학습 전)"""
        # 약간의 노이즈를 추가한 자기 감정 반환
        predicted_dimensions = {}
        for dim, value in self_emotion.dimensions.items():
            noise = np.random.normal(0, 0.1)
            predicted_dimensions[dim] = np.clip(value + noise, -1.0, 1.0)
        
        return EmotionVector(
            dimensions=predicted_dimensions,
            intensity=self_emotion.intensity * 0.7,
            confidence=0.3,
            source='other'
        )

class Phase2CommunityExpander:
    """Phase 2: 타인 → 공동체 감정 확장기"""
    
    def __init__(self):
        self.community_patterns = {}
        self.temporal_dynamics = defaultdict(list)
        self.cultural_models = self._initialize_cultural_models()
        self.consensus_threshold = 0.7
        
        logger.info("Phase 2 공동체 감정 확장기가 초기화되었습니다.")
    
    def _initialize_cultural_models(self) -> Dict[str, Dict[str, float]]:
        """문화별 감정 표현 모델 초기화"""
        return {
            'korean_traditional': {
                'emotion_suppression': 0.7,  # 감정 억제 정도
                'collective_harmony': 0.9,   # 집단 조화 중시
                'hierarchy_influence': 0.8,  # 위계의 영향
                'indirect_expression': 0.8   # 간접적 표현
            },
            'western_individualistic': {
                'emotion_suppression': 0.3,
                'collective_harmony': 0.4,
                'hierarchy_influence': 0.3,
                'indirect_expression': 0.2
            }
        }
    
    async def expand_to_community(self,
                                individual_emotions: Dict[str, EmotionVector],
                                cultural_context: str = 'korean_traditional',
                                group_dynamics: Dict[str, Any] = None) -> CommunityEmotionPattern:
        """개인 감정을 공동체 감정으로 확장"""
        try:
            # 문화적 모델 적용
            cultural_model = self.cultural_models.get(
                cultural_context, 
                self.cultural_models['korean_traditional']
            )
            
            # 개별 감정 조정 (문화적 영향)
            adjusted_emotions = await self._apply_cultural_adjustment(
                individual_emotions, cultural_model
            )
            
            # 집단 감정 계산
            collective_emotion = await self._calculate_collective_emotion(
                adjusted_emotions, group_dynamics
            )
            
            # 감정 일치도 계산
            consensus_level = self._calculate_consensus(adjusted_emotions)
            
            # 시간적 동태 추가
            temporal_key = f"{cultural_context}_{datetime.now().strftime('%Y%m%d')}"
            self.temporal_dynamics[temporal_key].append(collective_emotion)
            
            # 커뮤니티 패턴 생성
            pattern = CommunityEmotionPattern(
                emotion_distribution=adjusted_emotions,
                collective_emotion=collective_emotion,
                consensus_level=consensus_level,
                cultural_factors=cultural_model,
                temporal_dynamics=self.temporal_dynamics[temporal_key][-10:]  # 최근 10개
            )
            
            # 패턴 저장
            pattern_key = self._generate_pattern_key(cultural_context, group_dynamics)
            self.community_patterns[pattern_key] = pattern
            
            logger.debug(f"공동체 감정 확장 완료: 일치도={consensus_level:.3f}")
            return pattern
            
        except Exception as e:
            logger.error(f"공동체 감정 확장 실패: {e}")
            # 실패 시 평균 감정 반환
            avg_emotion = self._calculate_average_emotion(individual_emotions)
            return CommunityEmotionPattern(
                emotion_distribution=individual_emotions,
                collective_emotion=avg_emotion,
                consensus_level=0.0
            )
    
    async def _apply_cultural_adjustment(self,
                                       emotions: Dict[str, EmotionVector],
                                       cultural_model: Dict[str, float]) -> Dict[str, EmotionVector]:
        """문화적 조정 적용"""
        adjusted = {}
        
        for person_id, emotion in emotions.items():
            adjusted_dimensions = {}
            
            for dim in EmotionDimension:
                value = emotion.dimensions.get(dim, 0.0)
                
                # 감정 억제 적용
                if dim in [EmotionDimension.AROUSAL, EmotionDimension.VALENCE]:
                    suppression = cultural_model['emotion_suppression']
                    value *= (1.0 - suppression * 0.5)
                
                # 집단 조화 고려
                if dim == EmotionDimension.DOMINANCE:
                    harmony = cultural_model['collective_harmony']
                    value *= (1.0 - harmony * 0.3)  # 지배성 감소
                
                adjusted_dimensions[dim] = value
            
            adjusted[person_id] = EmotionVector(
                dimensions=adjusted_dimensions,
                intensity=emotion.intensity * (1.0 - cultural_model['indirect_expression'] * 0.3),
                confidence=emotion.confidence,
                source='community'
            )
        
        return adjusted
    
    async def _calculate_collective_emotion(self,
                                          emotions: Dict[str, EmotionVector],
                                          group_dynamics: Dict[str, Any]) -> EmotionVector:
        """집단 감정 계산"""
        if not emotions:
            return EmotionVector(dimensions={dim: 0.0 for dim in EmotionDimension})
        
        # 가중 평균 계산
        weights = self._calculate_individual_weights(emotions, group_dynamics)
        
        collective_dimensions = {}
        for dim in EmotionDimension:
            weighted_sum = sum(
                emotions[pid].dimensions.get(dim, 0.0) * weights.get(pid, 1.0)
                for pid in emotions
            )
            collective_dimensions[dim] = weighted_sum / sum(weights.values())
        
        # 집단 특성 반영
        if group_dynamics:
            # 리더의 영향
            if 'leader_id' in group_dynamics and group_dynamics['leader_id'] in emotions:
                leader_emotion = emotions[group_dynamics['leader_id']]
                for dim in EmotionDimension:
                    leader_value = leader_emotion.dimensions.get(dim, 0.0)
                    collective_dimensions[dim] = (
                        0.7 * collective_dimensions[dim] + 
                        0.3 * leader_value
                    )
        
        # 평균 강도와 신뢰도
        avg_intensity = np.mean([e.intensity for e in emotions.values()])
        avg_confidence = np.mean([e.confidence for e in emotions.values()])
        
        return EmotionVector(
            dimensions=collective_dimensions,
            intensity=avg_intensity,
            confidence=avg_confidence * 0.8,  # 집단 감정은 불확실성 증가
            source='community'
        )
    
    def _calculate_individual_weights(self,
                                    emotions: Dict[str, EmotionVector],
                                    group_dynamics: Dict[str, Any]) -> Dict[str, float]:
        """개인별 가중치 계산"""
        weights = {pid: 1.0 for pid in emotions}
        
        if not group_dynamics:
            return weights
        
        # 사회적 지위에 따른 가중치
        if 'social_status' in group_dynamics:
            for pid, status in group_dynamics['social_status'].items():
                if pid in weights:
                    weights[pid] *= (1.0 + status * 0.5)
        
        # 발언권/영향력
        if 'influence' in group_dynamics:
            for pid, influence in group_dynamics['influence'].items():
                if pid in weights:
                    weights[pid] *= influence
        
        # 정규화
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {pid: w/total_weight for pid, w in weights.items()}
        
        return weights
    
    def _calculate_consensus(self, emotions: Dict[str, EmotionVector]) -> float:
        """감정 일치도 계산"""
        if len(emotions) < 2:
            return 1.0
        
        # 모든 쌍의 거리 계산
        emotion_list = list(emotions.values())
        distances = []
        
        for i in range(len(emotion_list)):
            for j in range(i + 1, len(emotion_list)):
                dist = emotion_list[i].distance(emotion_list[j])
                distances.append(dist)
        
        # 평균 거리를 일치도로 변환
        avg_distance = np.mean(distances)
        consensus = 1.0 - (avg_distance / 2.0)  # 최대 거리가 2라고 가정
        
        return np.clip(consensus, 0.0, 1.0)
    
    def _calculate_average_emotion(self, 
                                 emotions: Dict[str, EmotionVector]) -> EmotionVector:
        """평균 감정 계산 (폴백용)"""
        if not emotions:
            return EmotionVector(dimensions={dim: 0.0 for dim in EmotionDimension})
        
        avg_dimensions = {}
        for dim in EmotionDimension:
            values = [e.dimensions.get(dim, 0.0) for e in emotions.values()]
            avg_dimensions[dim] = np.mean(values)
        
        return EmotionVector(
            dimensions=avg_dimensions,
            intensity=np.mean([e.intensity for e in emotions.values()]),
            confidence=np.mean([e.confidence for e in emotions.values()]) * 0.8,
            source='community'
        )
    
    def _generate_pattern_key(self, 
                            cultural_context: str,
                            group_dynamics: Dict[str, Any]) -> str:
        """패턴 키 생성"""
        key_parts = [cultural_context]
        
        if group_dynamics:
            if 'group_type' in group_dynamics:
                key_parts.append(f"type:{group_dynamics['group_type']}")
            if 'size' in group_dynamics:
                size_category = 'small' if group_dynamics['size'] < 10 else 'large'
                key_parts.append(f"size:{size_category}")
        
        return "|".join(key_parts)

class AdvancedHierarchicalEmotionSystem:
    """통합 계층적 감정 시스템"""
    
    def __init__(self):
        """시스템 초기화"""
        self.config = ADVANCED_CONFIG.get('hierarchical_emotion', {})
        self.models_dir = os.path.join(MODELS_DIR, 'hierarchical_emotion')
        os.makedirs(self.models_dir, exist_ok=True)
        
        # 각 단계 초기화
        self.phase0_calibrator = Phase0EmotionCalibrator()
        self.phase1_learner = Phase1EmpathyLearner()
        self.phase2_expander = Phase2CommunityExpander()
        
        # 통합 메모리
        self.emotion_memory = deque(maxlen=5000)
        self.learning_trajectory = []
        
        # 성능 메트릭
        self.performance_metrics = {
            'calibration_accuracy': [],
            'empathy_accuracy': [],
            'consensus_predictions': [],
            'total_interactions': 0
        }
        
        logger.info("고급 계층적 감정 시스템이 초기화되었습니다.")
    
    async def process_literary_emotion_sequence(self,
                                              literary_data: List[Dict[str, Any]],
                                              time_series_mode: bool = True) -> Dict[str, Any]:
        """
        문학 데이터 시계열을 통한 전체 감정 학습 프로세스
        
        Args:
            literary_data: 문학 작품의 감정 시퀀스
            time_series_mode: 시계열 분석 모드
            
        Returns:
            학습 결과 및 감정 발달 궤적
        """
        results = {
            'phase0_calibrations': [],
            'phase1_learnings': [],
            'phase2_patterns': [],
            'emotion_trajectory': [],
            'learning_summary': {}
        }
        
        try:
            for idx, data_point in enumerate(literary_data):
                logger.info(f"처리 중: {idx+1}/{len(literary_data)}")
                
                # Phase 0: 타자 감정 캘리브레이션
                if 'character_emotion' in data_point:
                    calibration = await self.phase0_calibrator.calibrate_emotion(
                        other_emotion=self._parse_emotion(data_point['character_emotion']),
                        context=data_point.get('context', {}),
                        literary_reference=data_point.get('genre', 'unknown')
                    )
                    results['phase0_calibrations'].append(calibration)
                    
                    # Phase 1: 공감 학습 (이전 데이터가 있는 경우)
                    if idx > 0 and 'reader_emotion' in data_point:
                        # 이전 캘리브레이션 결과를 사용해 예측
                        predicted = await self.phase1_learner.predict_other_emotion(
                            self_emotion=calibration.projected_self_emotion,
                            context=data_point.get('context', {})
                        )
                        
                        # 실제 독자 감정과 비교하여 학습
                        learning = await self.phase1_learner.learn_empathy(
                            self_emotion=calibration.projected_self_emotion,
                            predicted_other=predicted,
                            actual_other=self._parse_emotion(data_point['reader_emotion']),
                            context=data_point.get('context', {})
                        )
                        results['phase1_learnings'].append(learning)
                
                # Phase 2: 공동체 감정 (그룹 데이터가 있는 경우)
                if 'community_emotions' in data_point:
                    community_pattern = await self.phase2_expander.expand_to_community(
                        individual_emotions=self._parse_community_emotions(
                            data_point['community_emotions']
                        ),
                        cultural_context=data_point.get('cultural_context', 'korean_traditional'),
                        group_dynamics=data_point.get('group_dynamics', {})
                    )
                    results['phase2_patterns'].append(community_pattern)
                
                # 감정 궤적 기록
                if time_series_mode:
                    trajectory_point = self._create_trajectory_point(
                        calibration if 'character_emotion' in data_point else None,
                        learning if idx > 0 and 'reader_emotion' in data_point else None,
                        community_pattern if 'community_emotions' in data_point else None
                    )
                    results['emotion_trajectory'].append(trajectory_point)
            
            # 학습 요약 생성
            results['learning_summary'] = await self._generate_learning_summary(results)
            
            # 성능 메트릭 업데이트
            self._update_performance_metrics(results)
            
            logger.info("문학 감정 시퀀스 처리 완료")
            return results
            
        except Exception as e:
            logger.error(f"문학 감정 시퀀스 처리 실패: {e}")
            return results
    
    def _parse_emotion(self, emotion_data: Any) -> EmotionVector:
        """감정 데이터를 EmotionVector로 파싱"""
        if isinstance(emotion_data, EmotionVector):
            return emotion_data
        
        if isinstance(emotion_data, dict):
            dimensions = {}
            for dim in EmotionDimension:
                dim_name = dim.value
                if dim_name in emotion_data:
                    dimensions[dim] = float(emotion_data[dim_name])
                else:
                    # 기본값 설정
                    dimensions[dim] = 0.0
            
            return EmotionVector(
                dimensions=dimensions,
                intensity=emotion_data.get('intensity', 1.0),
                confidence=emotion_data.get('confidence', 1.0),
                source=emotion_data.get('source', 'other')
            )
        
        # 단순 값인 경우
        return EmotionVector(
            dimensions={EmotionDimension.VALENCE: float(emotion_data)},
            intensity=1.0,
            confidence=0.5,
            source='other'
        )
    
    def _parse_community_emotions(self, 
                                community_data: Any) -> Dict[str, EmotionVector]:
        """커뮤니티 감정 데이터 파싱"""
        if isinstance(community_data, dict):
            return {
                person_id: self._parse_emotion(emotion)
                for person_id, emotion in community_data.items()
            }
        
        # 리스트인 경우
        return {
            f"person_{i}": self._parse_emotion(emotion)
            for i, emotion in enumerate(community_data)
        }
    
    def _create_trajectory_point(self,
                               calibration: EmotionCalibrationData,
                               learning: EmpathyLearningData,
                               pattern: CommunityEmotionPattern) -> Dict[str, Any]:
        """감정 궤적 포인트 생성"""
        point = {
            'timestamp': datetime.now(),
            'phase_states': {}
        }
        
        if calibration:
            point['phase_states']['phase0'] = {
                'calibration_factor': calibration.calibration_factor,
                'emotion_distance': calibration.other_emotion.distance(
                    calibration.projected_self_emotion
                )
            }
        
        if learning:
            point['phase_states']['phase1'] = {
                'prediction_error': learning.predicted_other_emotion.distance(
                    learning.actual_other_emotion
                ),
                'regret_intensity': learning.regret_intensity,
                'learning_rate': learning.learning_rate
            }
        
        if pattern:
            point['phase_states']['phase2'] = {
                'consensus_level': pattern.consensus_level,
                'collective_valence': pattern.collective_emotion.dimensions.get(
                    EmotionDimension.VALENCE, 0.0
                )
            }
        
        return point
    
    async def _generate_learning_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """학습 결과 요약 생성"""
        summary = {
            'total_calibrations': len(results['phase0_calibrations']),
            'total_empathy_learnings': len(results['phase1_learnings']),
            'total_community_patterns': len(results['phase2_patterns']),
            'trajectory_length': len(results['emotion_trajectory'])
        }
        
        # Phase 0 요약
        if results['phase0_calibrations']:
            calibration_factors = [
                c.calibration_factor for c in results['phase0_calibrations']
            ]
            summary['phase0_summary'] = {
                'avg_calibration_factor': np.mean(calibration_factors),
                'calibration_variance': np.var(calibration_factors)
            }
        
        # Phase 1 요약
        if results['phase1_learnings']:
            regret_intensities = [
                l.regret_intensity for l in results['phase1_learnings']
            ]
            summary['phase1_summary'] = {
                'avg_regret_intensity': np.mean(regret_intensities),
                'high_regret_count': sum(1 for r in regret_intensities if r > 0.5),
                'learning_improvement': self._calculate_learning_improvement(
                    results['phase1_learnings']
                )
            }
        
        # Phase 2 요약
        if results['phase2_patterns']:
            consensus_levels = [
                p.consensus_level for p in results['phase2_patterns']
            ]
            summary['phase2_summary'] = {
                'avg_consensus': np.mean(consensus_levels),
                'high_consensus_ratio': sum(
                    1 for c in consensus_levels if c > 0.7
                ) / len(consensus_levels)
            }
        
        return summary
    
    def _calculate_learning_improvement(self, 
                                      learnings: List[EmpathyLearningData]) -> float:
        """학습 개선도 계산"""
        if len(learnings) < 2:
            return 0.0
        
        # 초기와 후기의 예측 오차 비교
        early_errors = []
        late_errors = []
        
        split_point = len(learnings) // 2
        
        for i, learning in enumerate(learnings):
            error = learning.predicted_other_emotion.distance(
                learning.actual_other_emotion
            )
            if i < split_point:
                early_errors.append(error)
            else:
                late_errors.append(error)
        
        if early_errors and late_errors:
            early_avg = np.mean(early_errors)
            late_avg = np.mean(late_errors)
            
            # 개선도 = (초기 오차 - 후기 오차) / 초기 오차
            improvement = (early_avg - late_avg) / early_avg if early_avg > 0 else 0.0
            return max(0.0, improvement)
        
        return 0.0
    
    def _update_performance_metrics(self, results: Dict[str, Any]):
        """성능 메트릭 업데이트"""
        self.performance_metrics['total_interactions'] += len(
            results.get('emotion_trajectory', [])
        )
        
        # 캘리브레이션 정확도
        if results['phase0_calibrations']:
            # 캘리브레이션 계수의 안정성을 정확도로 간주
            factors = [c.calibration_factor for c in results['phase0_calibrations']]
            stability = 1.0 - np.std(factors)
            self.performance_metrics['calibration_accuracy'].append(stability)
        
        # 공감 정확도
        if results['phase1_learnings']:
            # 예측 오차의 역수를 정확도로 사용
            errors = [
                l.predicted_other_emotion.distance(l.actual_other_emotion)
                for l in results['phase1_learnings']
            ]
            accuracy = 1.0 - np.mean(errors)
            self.performance_metrics['empathy_accuracy'].append(accuracy)
        
        # 합의 예측
        if results['phase2_patterns']:
            consensus_levels = [p.consensus_level for p in results['phase2_patterns']]
            self.performance_metrics['consensus_predictions'].extend(consensus_levels)
    
    async def save_models(self, filepath: str = None):
        """학습된 모델 저장"""
        if filepath is None:
            filepath = os.path.join(
                self.models_dir, 
                f"hierarchical_emotion_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            )
        
        try:
            model_data = {
                'phase0_calibration_factors': dict(self.phase0_calibrator.calibration_factors),
                'phase1_empathy_models': self.phase1_learner.empathy_models,
                'phase2_community_patterns': self.phase2_expander.community_patterns,
                'performance_metrics': self.performance_metrics,
                'config': self.config
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"모델이 저장되었습니다: {filepath}")
            
        except Exception as e:
            logger.error(f"모델 저장 실패: {e}")
    
    async def load_models(self, filepath: str):
        """저장된 모델 로드"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Phase 0 모델 복원
            self.phase0_calibrator.calibration_factors.update(
                model_data.get('phase0_calibration_factors', {})
            )
            
            # Phase 1 모델 복원
            self.phase1_learner.empathy_models = model_data.get(
                'phase1_empathy_models', {}
            )
            
            # Phase 2 패턴 복원
            self.phase2_expander.community_patterns = model_data.get(
                'phase2_community_patterns', {}
            )
            
            # 성능 메트릭 복원
            self.performance_metrics.update(
                model_data.get('performance_metrics', {})
            )
            
            logger.info(f"모델이 로드되었습니다: {filepath}")
            
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")


# =============================================================================
# Enhanced Empathy Learning System - Phase 1 Implementation
# =============================================================================

class EnhancedEmpathyLearner:
    """
    Enhanced Empathy Learning System - Self-Other-Community 계층적 공감 학습
    
    2024년 최신 연구 기반:
    - Mirror Neuron System (Brain-Inspired AE-SNN)
    - Theory of Mind AI (GPT-4 수준 false-belief task)
    - Hierarchical Empathy Processing
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 기존 계층적 시스템 통합
        self.base_system = AdvancedHierarchicalEmotionSystem()
        
        # 새로운 컴포넌트들
        self.self_reflection_processor = SelfReflectionProcessor()
        self.empathy_simulation_processor = EmpathySimulationProcessor()
        self.community_context_processor = CommunityContextProcessor()
        self.mirror_neuron_system = MirrorNeuronSystem()
        
        # SURD 불확실성 전파 시스템 통합
        if SURD_AVAILABLE:
            self.surd_analyzer = AdvancedSURDAnalyzer()
            self.uncertainty_propagation_enabled = True
        else:
            self.surd_analyzer = None
            self.uncertainty_propagation_enabled = False
        
        # 성능 메트릭 및 캐싱
        self.performance_metrics = {
            'total_empathy_predictions': 0,
            'empathy_accuracy': [],
            'self_awareness_scores': [],
            'community_integration_scores': [],
            'processing_times': []
        }
        
        # 고급 캐싱 시스템
        self.cache_system = AdvancedCacheSystem()
        self.performance_optimizer = PerformanceOptimizer()
        
        # 동적 GPU 할당을 위한 모델 관리
        self.models = {}
        self._initialize_models()
        
        self.logger.info("Enhanced Empathy Learning System 초기화 완료")
        
        # 초기화 완료 플래그
        self.initialized = False
    
    async def initialize(self):
        """비동기 초기화 - 기존 최적화 시스템들과 일관된 패턴"""
        if self.initialized:
            return
        
        try:
            # 캐시 시스템 비동기 초기화
            await self.cache_system.initialize()
            
            # 베이스 시스템의 비동기 컴포넌트 초기화 (필요시)
            if hasattr(self.base_system, 'initialize'):
                await self.base_system.initialize()
            
            # 기타 비동기 모델 로딩
            await self._initialize_async_models()
            
            self.initialized = True
            self.logger.info("Enhanced Empathy Learning System 비동기 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"비동기 초기화 실패: {e}")
            raise
    
    async def _initialize_async_models(self):
        """비동기 모델 초기화"""
        try:
            # 필요시 추가 비동기 모델 로딩 작업
            # 현재는 동기 초기화만 있으므로 패스
            pass
        except Exception as e:
            self.logger.warning(f"비동기 모델 초기화 중 경고: {e}")
    
    def get_pytorch_network(self) -> Optional[torch.nn.Module]:
        """PyTorch 네트워크 반환 - HeadCompatibilityManager용"""
        try:
            # 1. Phase1EmpathyLearner의 neural_empathy_model 우선 확인
            if hasattr(self.base_system, 'phase1_learner'):
                phase1 = self.base_system.phase1_learner
                if hasattr(phase1, 'neural_empathy_model'):
                    network = phase1.neural_empathy_model
                    if isinstance(network, torch.nn.Module):
                        return network
            
            # 2. 베이스 시스템의 neural_empathy_model 확인
            if hasattr(self.base_system, 'neural_empathy_model'):
                network = self.base_system.neural_empathy_model
                if isinstance(network, torch.nn.Module):
                    return network
            
            # 3. 베이스 시스템에서 주요 neural network 컴포넌트 찾기
            if hasattr(self.base_system, 'neural_networks'):
                networks = self.base_system.neural_networks
                if isinstance(networks, dict) and networks:
                    # 첫 번째 available network 반환
                    for network in networks.values():
                        if isinstance(network, torch.nn.Module):
                            return network
            
            # 4. base_system 자체가 nn.Module인 경우
            if isinstance(self.base_system, torch.nn.Module):
                return self.base_system
            
            # 5. 다른 phase들에서 neural network 찾기
            for phase_name in ['phase0_calibrator', 'phase2_expander']:
                if hasattr(self.base_system, phase_name):
                    phase = getattr(self.base_system, phase_name)
                    for attr_name in ['neural_empathy_model', 'neural_predictor', '_neural_predictor', 'model', 'network']:
                        if hasattr(phase, attr_name):
                            network = getattr(phase, attr_name)
                            if isinstance(network, torch.nn.Module):
                                return network
            
            # 6. 다른 가능한 neural network 속성들 확인
            for attr_name in ['neural_predictor', '_neural_predictor', 'model', 'network', 'classifier', 'neural_model', 'empathy_model']:
                if hasattr(self.base_system, attr_name):
                    network = getattr(self.base_system, attr_name)
                    if isinstance(network, torch.nn.Module):
                        return network
            
            return None
            
        except Exception as e:
            self.logger.warning(f"PyTorch 네트워크 탐지 실패: {e}")
            return None
    
    def _initialize_models(self):
        """모델 초기화 - 동적 로딩"""
        try:
            # 기본 모델들은 필요시 로딩
            self.models = {
                'self_reflection_model': None,
                'empathy_simulation_model': None,
                'community_context_model': None,
                'mirror_neuron_model': None
            }
            
            self.logger.info("모델 초기화 완료 (동적 로딩 방식)")
            
        except Exception as e:
            self.logger.error(f"모델 초기화 실패: {e}")
            raise
    
    async def process_hierarchical_empathy(self, 
                                         input_text: str,
                                         context: Dict[str, Any] = None,
                                         analysis_depth: str = "full") -> 'HierarchicalEmpathyResult':
        """
        계층적 공감 분석 처리
        
        Args:
            input_text: 분석할 텍스트
            context: 추가 컨텍스트
            analysis_depth: 분석 깊이 (surface, medium, full)
            
        Returns:
            HierarchicalEmpathyResult: 계층적 공감 분석 결과
        """
        from data_models import HierarchicalEmpathyResult, DataOriginHelper
        
        start_time = time.time()
        context = context or {}
        
        try:
            # 데이터 출처 태깅
            data_origin = DataOriginHelper.detect_origin(input_text, context)
            data_origin_tag = DataOriginHelper.create_tag(data_origin, context.get('source', ''))
            
            # 병렬 처리를 위한 태스크 생성
            analysis_tasks = []
            
            # 1. Self-Reflection 분석
            analysis_tasks.append(self._analyze_self_reflection(input_text, context, data_origin_tag))
            
            # 2. Empathy Simulation 분석
            analysis_tasks.append(self._analyze_empathy_simulation(input_text, context, data_origin_tag))
            
            # 3. Community Context 분석 (medium 이상에서만)
            if analysis_depth in ['medium', 'full']:
                analysis_tasks.append(self._analyze_community_context(input_text, context, data_origin_tag))
            
            # 4. Mirror Neuron 분석 (full에서만)
            if analysis_depth == 'full':
                analysis_tasks.append(self._analyze_mirror_neuron(input_text, context, data_origin_tag))
            
            # 병렬 실행
            results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # 결과 통합
            result = await self._integrate_empathy_results(
                input_text, results, analysis_depth, data_origin_tag
            )
            
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            
            # 성능 메트릭 업데이트
            self._update_performance_metrics(result, processing_time)
            
            self.logger.info(f"계층적 공감 분석 완료 - 처리 시간: {processing_time:.2f}초")
            
            return result
            
        except Exception as e:
            self.logger.error(f"계층적 공감 분석 실패: {e}")
            # 기본 결과 반환
            return HierarchicalEmpathyResult(
                input_text=input_text,
                analysis_depth=analysis_depth,
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                data_origin_tag=data_origin_tag,
                metadata={'error': str(e)}
            )
    
    async def _analyze_self_reflection(self, 
                                     input_text: str, 
                                     context: Dict[str, Any],
                                     data_origin_tag) -> 'SelfReflectionData':
        """자기 성찰 분석"""
        from data_models import SelfReflectionData
        from config import get_smart_device, gpu_model_context
        
        try:
            # 필요시 모델 동적 로딩
            if self.models['self_reflection_model'] is None:
                self.models['self_reflection_model'] = self._load_self_reflection_model()
            
            # 동적 GPU 할당
            device = get_smart_device(memory_required_mb=300)
            
            with gpu_model_context(self.models['self_reflection_model'], 300):
                # 자기 성찰 분석 수행
                self_emotion_state = await self._extract_self_emotion_state(input_text, context)
                
                # 내부 갈등 분석
                internal_conflict = await self._analyze_internal_conflict(input_text, self_emotion_state)
                
                # 개인 가치 정렬도 계산
                values_alignment = await self._calculate_values_alignment(input_text, context)
                
                # 인지 부조화 점수
                cognitive_dissonance = await self._calculate_cognitive_dissonance(
                    self_emotion_state, values_alignment
                )
                
                return SelfReflectionData(
                    input_text=input_text,
                    self_emotion_state=self_emotion_state,
                    self_confidence=self_emotion_state.get('confidence', 0.5),
                    internal_conflict_score=internal_conflict,
                    personal_values_alignment=values_alignment,
                    cognitive_dissonance_score=cognitive_dissonance,
                    reflection_triggers=self._identify_reflection_triggers(input_text),
                    data_origin_tag=data_origin_tag
                )
                
        except Exception as e:
            self.logger.error(f"자기 성찰 분석 실패: {e}")
            return SelfReflectionData(
                input_text=input_text,
                data_origin_tag=data_origin_tag,
                metadata={'error': str(e)}
            )
    
    async def _analyze_empathy_simulation(self, 
                                        input_text: str, 
                                        context: Dict[str, Any],
                                        data_origin_tag) -> 'EmpathySimulationData':
        """공감 시뮬레이션 분석"""
        from data_models import EmpathySimulationData
        from config import get_smart_device, gpu_model_context
        
        try:
            # 필요시 모델 동적 로딩
            if self.models['empathy_simulation_model'] is None:
                self.models['empathy_simulation_model'] = self._load_empathy_simulation_model()
            
            # 동적 GPU 할당
            device = get_smart_device(memory_required_mb=400)
            
            with gpu_model_context(self.models['empathy_simulation_model'], 400):
                # 타인 감정 예측
                predicted_other_emotion = await self._predict_other_emotion(input_text, context)
                
                # Theory of Mind 점수 계산
                theory_of_mind_score = await self._calculate_theory_of_mind(
                    input_text, predicted_other_emotion
                )
                
                # 공감 강도 계산
                empathy_intensity = await self._calculate_empathy_intensity(
                    predicted_other_emotion, context
                )
                
                # 관점 취하기 점수
                perspective_taking = await self._calculate_perspective_taking(
                    input_text, predicted_other_emotion
                )
                
                return EmpathySimulationData(
                    input_text=input_text,
                    target_perspective=context.get('target_perspective', 'general'),
                    predicted_other_emotion=predicted_other_emotion,
                    empathy_intensity=empathy_intensity,
                    theory_of_mind_score=theory_of_mind_score,
                    perspective_taking_score=perspective_taking,
                    compassion_level=self._calculate_compassion_level(predicted_other_emotion),
                    data_origin_tag=data_origin_tag
                )
                
        except Exception as e:
            self.logger.error(f"공감 시뮬레이션 분석 실패: {e}")
            return EmpathySimulationData(
                input_text=input_text,
                data_origin_tag=data_origin_tag,
                metadata={'error': str(e)}
            )
    
    async def _analyze_community_context(self, 
                                       input_text: str, 
                                       context: Dict[str, Any],
                                       data_origin_tag) -> 'CommunityContextData':
        """공동체 맥락 분석"""
        from data_models import CommunityContextData
        from config import get_smart_device, gpu_model_context
        
        try:
            # 필요시 모델 동적 로딩
            if self.models['community_context_model'] is None:
                self.models['community_context_model'] = self._load_community_context_model()
            
            # 동적 GPU 할당
            device = get_smart_device(memory_required_mb=350)
            
            with gpu_model_context(self.models['community_context_model'], 350):
                # 공동체 유형 감지
                community_type = await self._detect_community_type(input_text, context)
                
                # 문화적 배경 분석
                cultural_background = await self._analyze_cultural_background(input_text, context)
                
                # 사회적 규범 분석
                social_norms = await self._analyze_social_norms(input_text, cultural_background)
                
                # 집단 감정 상태 추론
                collective_emotion = await self._infer_collective_emotion(input_text, context)
                
                # 사회적 결속 점수
                social_cohesion = await self._calculate_social_cohesion(
                    collective_emotion, social_norms
                )
                
                return CommunityContextData(
                    input_text=input_text,
                    community_type=community_type,
                    cultural_background=cultural_background,
                    social_norms=social_norms,
                    collective_emotion_state=collective_emotion,
                    social_cohesion_score=social_cohesion,
                    cultural_alignment_score=self._calculate_cultural_alignment(
                        cultural_background, context
                    ),
                    data_origin_tag=data_origin_tag
                )
                
        except Exception as e:
            self.logger.error(f"공동체 맥락 분석 실패: {e}")
            return CommunityContextData(
                input_text=input_text,
                data_origin_tag=data_origin_tag,
                metadata={'error': str(e)}
            )
    
    async def _analyze_mirror_neuron(self, 
                                   input_text: str, 
                                   context: Dict[str, Any],
                                   data_origin_tag) -> 'MirrorNeuronData':
        """Mirror Neuron 시스템 분석"""
        from data_models import MirrorNeuronData
        from config import get_smart_device, gpu_model_context
        
        try:
            # 필요시 모델 동적 로딩
            if self.models['mirror_neuron_model'] is None:
                self.models['mirror_neuron_model'] = self._load_mirror_neuron_model()
            
            # 동적 GPU 할당
            device = get_smart_device(memory_required_mb=400)
            
            with gpu_model_context(self.models['mirror_neuron_model'], 400):
                # 관찰된 행동 추출
                observed_action = await self._extract_observed_action(input_text)
                
                # 거울 반응 생성
                mirrored_response = await self._generate_mirrored_response(
                    observed_action, context
                )
                
                # 신경 활성화 패턴 (Brain-Inspired AE-SNN 기반)
                neural_activation = await self._calculate_neural_activation(
                    observed_action, mirrored_response
                )
                
                # 자타 구분 능력
                self_other_diff = await self._calculate_self_other_differentiation(
                    mirrored_response, context
                )
                
                # Free Energy Principle 기반 예측
                free_energy = await self._calculate_free_energy_prediction(
                    observed_action, mirrored_response
                )
                
                return MirrorNeuronData(
                    input_stimulus=input_text,
                    observed_action=observed_action,
                    mirrored_response=mirrored_response,
                    neural_activation_pattern=neural_activation,
                    self_other_differentiation=self_other_diff,
                    mirror_fidelity=self._calculate_mirror_fidelity(
                        observed_action, mirrored_response
                    ),
                    free_energy_prediction=free_energy,
                    data_origin_tag=data_origin_tag
                )
                
        except Exception as e:
            self.logger.error(f"Mirror Neuron 분석 실패: {e}")
            return MirrorNeuronData(
                input_stimulus=input_text,
                data_origin_tag=data_origin_tag,
                metadata={'error': str(e)}
            )
    
    async def _integrate_empathy_results(self, 
                                       input_text: str,
                                       results: List[Any],
                                       analysis_depth: str,
                                       data_origin_tag) -> 'HierarchicalEmpathyResult':
        """공감 분석 결과 통합"""
        from data_models import HierarchicalEmpathyResult
        
        try:
            # 결과 분리
            self_reflection_result = None
            empathy_simulation_result = None
            community_context_result = None
            mirror_neuron_result = None
            
            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"결과 통합 중 예외 발생: {result}")
                    continue
                
                if hasattr(result, 'reflection_id'):
                    self_reflection_result = result
                elif hasattr(result, 'simulation_id'):
                    empathy_simulation_result = result
                elif hasattr(result, 'context_id'):
                    community_context_result = result
                elif hasattr(result, 'neuron_id'):
                    mirror_neuron_result = result
            
            # 통합 점수 계산
            overall_empathy_score = await self._calculate_overall_empathy_score(
                self_reflection_result, empathy_simulation_result, 
                community_context_result, mirror_neuron_result
            )
            
            # 개별 점수 계산
            self_awareness_score = self._extract_self_awareness_score(self_reflection_result)
            other_understanding_score = self._extract_other_understanding_score(empathy_simulation_result)
            community_integration_score = self._extract_community_integration_score(community_context_result)
            
            # 균형 점수 계산
            self_other_balance = await self._calculate_self_other_balance(
                self_awareness_score, other_understanding_score
            )
            
            individual_community_balance = await self._calculate_individual_community_balance(
                other_understanding_score, community_integration_score
            )
            
            # 벤담 호환성 점수
            utilitarian_score = await self._calculate_utilitarian_compatibility(
                overall_empathy_score, self_other_balance, individual_community_balance
            )
            
            # 신뢰도 계산
            confidence_score = await self._calculate_integrated_confidence(
                self_reflection_result, empathy_simulation_result, 
                community_context_result, mirror_neuron_result
            )
            
            return HierarchicalEmpathyResult(
                input_text=input_text,
                self_reflection_result=self_reflection_result,
                empathy_simulation_result=empathy_simulation_result,
                community_context_result=community_context_result,
                mirror_neuron_result=mirror_neuron_result,
                overall_empathy_score=overall_empathy_score,
                self_awareness_score=self_awareness_score,
                other_understanding_score=other_understanding_score,
                community_integration_score=community_integration_score,
                self_other_balance=self_other_balance,
                individual_community_balance=individual_community_balance,
                utilitarian_compatibility_score=utilitarian_score,
                confidence_score=confidence_score,
                analysis_depth=analysis_depth,
                data_origin_tag=data_origin_tag
            )
            
        except Exception as e:
            self.logger.error(f"결과 통합 실패: {e}")
            return HierarchicalEmpathyResult(
                input_text=input_text,
                analysis_depth=analysis_depth,
                confidence_score=0.0,
                data_origin_tag=data_origin_tag,
                metadata={'integration_error': str(e)}
            )
    
    def _update_performance_metrics(self, result: 'HierarchicalEmpathyResult', processing_time: float):
        """성능 메트릭 업데이트"""
        self.performance_metrics['total_empathy_predictions'] += 1
        self.performance_metrics['processing_times'].append(processing_time)
        
        if result.confidence_score > 0:
            self.performance_metrics['empathy_accuracy'].append(result.overall_empathy_score)
            self.performance_metrics['self_awareness_scores'].append(result.self_awareness_score)
            self.performance_metrics['community_integration_scores'].append(result.community_integration_score)
    
    # 모델 로딩 메서드들 (스텁 구현 - 실제 모델 로딩은 향후 구현)
    def _load_self_reflection_model(self):
        """자기 성찰 모델 로딩"""
        # 실제 구현에서는 Pre-trained model 로딩
        return MockModel("self_reflection")
    
    def _load_empathy_simulation_model(self):
        """공감 시뮬레이션 모델 로딩"""
        return MockModel("empathy_simulation")
    
    def _load_community_context_model(self):
        """공동체 맥락 모델 로딩"""
        return MockModel("community_context")
    
    def _load_mirror_neuron_model(self):
        """Mirror Neuron 모델 로딩"""
        return MockModel("mirror_neuron")
    
    # 분석 헬퍼 메서드들 (기본 구현)
    async def _extract_self_emotion_state(self, text: str, context: Dict) -> Dict[str, float]:
        """
        자기 감정 상태 추출 - 고도화된 다차원 감정 분석
        
        고도화된 감정 상태 추출 시스템:
        - 다차원 감정 공간 분석 (VAD 모델 + 기본 감정 + 복합 감정)
        - 언어적 특징 기반 감정 추출
        - 컨텍스트 조건부 감정 분석
        - 시간적 감정 역학 분석
        - 불확실성 정량화
        - 감정 강도 및 지속성 예측
        
        Args:
            text: 분석할 텍스트
            context: 상황 컨텍스트
            
        Returns:
            다차원 감정 상태 벡터
        """
        try:
            # 1. 기본 감정 차원 분석 (VAD 모델)
            vad_scores = await self._analyze_vad_dimensions(text)
            
            # 2. 기본 감정 카테고리 분석 (Plutchik's 8 emotions)
            basic_emotions = await self._analyze_basic_emotions(text)
            
            # 3. 복합 감정 분석 (secondary emotions)
            complex_emotions = await self._analyze_complex_emotions(text, basic_emotions)
            
            # 4. 언어적 특징 기반 감정 추출
            linguistic_features = await self._extract_linguistic_emotional_features(text)
            
            # 5. 컨텍스트 조건부 감정 분석
            contextual_modulation = await self._analyze_contextual_emotional_modulation(text, context)
            
            # 6. 시간적 감정 역학 분석
            temporal_dynamics = await self._analyze_temporal_emotional_dynamics(text, context)
            
            # 7. 감정 강도 및 지속성 예측
            intensity_persistence = await self._predict_emotional_intensity_persistence(
                vad_scores, basic_emotions, context
            )
            
            # 8. 불확실성 정량화
            uncertainty_metrics = await self._quantify_emotional_uncertainty(
                vad_scores, basic_emotions, complex_emotions
            )
            
            # 9. 자기 참조적 감정 분석 (self-referential emotions)
            self_referential = await self._analyze_self_referential_emotions(text, context)
            
            # 10. 메타-감정 분석 (emotions about emotions)
            meta_emotions = await self._analyze_meta_emotions(text, basic_emotions)
            
            # 11. 종합 감정 상태 구성
            comprehensive_emotion_state = {
                # VAD 차원
                'valence': vad_scores['valence'],
                'arousal': vad_scores['arousal'],
                'dominance': vad_scores['dominance'],
                
                # 기본 감정 (Plutchik)
                'joy': basic_emotions['joy'],
                'sadness': basic_emotions['sadness'],
                'anger': basic_emotions['anger'],
                'fear': basic_emotions['fear'],
                'disgust': basic_emotions['disgust'],
                'surprise': basic_emotions['surprise'],
                'trust': basic_emotions['trust'],
                'anticipation': basic_emotions['anticipation'],
                
                # 복합 감정
                'contempt': complex_emotions['contempt'],
                'guilt': complex_emotions['guilt'],
                'shame': complex_emotions['shame'],
                'pride': complex_emotions['pride'],
                'envy': complex_emotions['envy'],
                'jealousy': complex_emotions['jealousy'],
                'hope': complex_emotions['hope'],
                'despair': complex_emotions['despair'],
                
                # 언어적 특징
                'emotional_polarity': linguistic_features['polarity'],
                'emotional_subjectivity': linguistic_features['subjectivity'],
                'emotional_intensity': linguistic_features['intensity'],
                'emotional_complexity': linguistic_features['complexity'],
                
                # 컨텍스트 조건부
                'context_congruence': contextual_modulation['congruence'],
                'context_amplification': contextual_modulation['amplification'],
                'social_appropriateness': contextual_modulation['social_appropriateness'],
                
                # 시간적 역학
                'emotional_stability': temporal_dynamics['stability'],
                'emotional_volatility': temporal_dynamics['volatility'],
                'emotional_trajectory': temporal_dynamics['trajectory'],
                
                # 강도 및 지속성
                'peak_intensity': intensity_persistence['peak_intensity'],
                'duration_prediction': intensity_persistence['duration'],
                'decay_rate': intensity_persistence['decay_rate'],
                
                # 불확실성
                'confidence': uncertainty_metrics['confidence'],
                'ambiguity': uncertainty_metrics['ambiguity'],
                'coherence': uncertainty_metrics['coherence'],
                
                # 자기 참조적
                'self_awareness': self_referential['self_awareness'],
                'self_evaluation': self_referential['self_evaluation'],
                'self_regulation': self_referential['self_regulation'],
                
                # 메타-감정
                'emotional_clarity': meta_emotions['clarity'],
                'emotional_acceptance': meta_emotions['acceptance'],
                'emotional_control': meta_emotions['control']
            }
            
            # 12. 감정 상태 일관성 검증
            consistency_score = await self._verify_emotional_state_consistency(
                comprehensive_emotion_state
            )
            
            comprehensive_emotion_state['consistency'] = consistency_score
            
            return comprehensive_emotion_state
            
        except Exception as e:
            self.logger.error(f"자기 감정 상태 추출 실패: {e}")
            # 기본 상태 반환
            return {
                'valence': 0.0,
                'arousal': 0.0,
                'dominance': 0.0,
                'confidence': 0.0,
                'consistency': 0.0
            }
    
    async def _analyze_internal_conflict(self, text: str, emotion_state: Dict) -> float:
        """
        내부 갈등 분석 - 고도화된 인지적 불일치 측정
        
        고도화된 내부 갈등 분석 시스템:
        - 다차원 감정 불일치 분석
        - 인지적 부조화 측정
        - 가치 갈등 분석
        - 접근-회피 갈등 분석
        - 이중 구속 상황 인식
        - 감정 조절 갈등 분석
        - 사회적 기대 vs 개인 욕구 갈등
        
        Args:
            text: 분석할 텍스트
            emotion_state: 감정 상태 벡터
            
        Returns:
            내부 갈등 점수 (0-1)
        """
        try:
            # 1. 감정 차원 간 불일치 분석
            dimensional_conflict = await self._analyze_dimensional_emotional_conflict(emotion_state)
            
            # 2. 기본 감정 간 대립 분석
            basic_emotion_conflict = await self._analyze_basic_emotion_conflicts(emotion_state)
            
            # 3. 복합 감정 모순 분석
            complex_emotion_conflict = await self._analyze_complex_emotion_contradictions(emotion_state)
            
            # 4. 인지적 부조화 분석
            cognitive_dissonance = await self._analyze_cognitive_dissonance_in_text(text, emotion_state)
            
            # 5. 가치 갈등 분석
            value_conflict = await self._analyze_value_conflicts(text, emotion_state)
            
            # 6. 접근-회피 갈등 분석
            approach_avoidance_conflict = await self._analyze_approach_avoidance_conflicts(text, emotion_state)
            
            # 7. 이중 구속 상황 인식
            double_bind_conflict = await self._detect_double_bind_situations(text, emotion_state)
            
            # 8. 감정 조절 갈등 분석
            emotion_regulation_conflict = await self._analyze_emotion_regulation_conflicts(text, emotion_state)
            
            # 9. 사회적 기대 vs 개인 욕구 갈등
            social_personal_conflict = await self._analyze_social_personal_conflicts(text, emotion_state)
            
            # 10. 시간적 갈등 분석 (현재 vs 미래 자아)
            temporal_conflict = await self._analyze_temporal_self_conflicts(text, emotion_state)
            
            # 11. 역할 갈등 분석
            role_conflict = await self._analyze_role_conflicts(text, emotion_state)
            
            # 12. 도덕적 갈등 분석
            moral_conflict = await self._analyze_moral_conflicts(text, emotion_state)
            
            # 13. 가중 평균 갈등 점수 계산
            weighted_conflict_score = (
                0.15 * dimensional_conflict +
                0.15 * basic_emotion_conflict +
                0.12 * complex_emotion_conflict +
                0.12 * cognitive_dissonance +
                0.10 * value_conflict +
                0.08 * approach_avoidance_conflict +
                0.08 * double_bind_conflict +
                0.08 * emotion_regulation_conflict +
                0.06 * social_personal_conflict +
                0.03 * temporal_conflict +
                0.02 * role_conflict +
                0.01 * moral_conflict
            )
            
            # 14. 갈등 강도 조정
            conflict_intensity = await self._calculate_conflict_intensity(
                text, emotion_state, weighted_conflict_score
            )
            
            # 15. 갈등 지속성 예측
            conflict_persistence = await self._predict_conflict_persistence(
                text, emotion_state, weighted_conflict_score
            )
            
            # 16. 최종 갈등 점수 (강도 및 지속성 고려)
            final_conflict_score = weighted_conflict_score * conflict_intensity * conflict_persistence
            
            return max(0.0, min(1.0, final_conflict_score))
            
        except Exception as e:
            self.logger.error(f"내부 갈등 분석 실패: {e}")
            return 0.0
    
    async def _analyze_dimensional_emotional_conflict(self, emotion_state: Dict) -> float:
        """감정 차원 간 불일치 분석"""
        try:
            # VAD 차원 간 불일치 분석
            valence = emotion_state.get('valence', 0.0)
            arousal = emotion_state.get('arousal', 0.0)
            dominance = emotion_state.get('dominance', 0.0)
            
            # 이론적으로 불일치하는 조합들
            conflicts = []
            
            # 높은 각성과 낮은 지배력 (스트레스 상태)
            if arousal > 0.6 and dominance < 0.4:
                conflicts.append(abs(arousal - dominance))
            
            # 긍정적 감정가와 높은 각성, 낮은 지배력 (불안한 기쁨)
            if valence > 0.5 and arousal > 0.6 and dominance < 0.4:
                conflicts.append(abs(valence - dominance))
            
            # 부정적 감정가와 낮은 각성 (우울한 상태에서의 모순)
            if valence < -0.5 and arousal > 0.6:
                conflicts.append(abs(valence + arousal))
            
            # 감정 안정성 vs 불안정성 지표
            stability = emotion_state.get('emotional_stability', 0.5)
            volatility = emotion_state.get('emotional_volatility', 0.5)
            
            if stability > 0.6 and volatility > 0.6:
                conflicts.append(abs(stability - volatility))
            
            return min(np.mean(conflicts) if conflicts else 0.0, 1.0)
            
        except Exception:
            return 0.0
    
    async def _analyze_basic_emotion_conflicts(self, emotion_state: Dict) -> float:
        """기본 감정 간 대립 분석"""
        try:
            # Plutchik의 대립 감정 쌍들
            opposing_pairs = [
                ('joy', 'sadness'),
                ('anger', 'fear'),
                ('trust', 'disgust'),
                ('anticipation', 'surprise')
            ]
            
            conflicts = []
            for emotion1, emotion2 in opposing_pairs:
                score1 = emotion_state.get(emotion1, 0.0)
                score2 = emotion_state.get(emotion2, 0.0)
                
                # 두 대립 감정이 모두 높은 경우
                if score1 > 0.5 and score2 > 0.5:
                    conflicts.append(min(score1, score2))
            
            return min(np.mean(conflicts) if conflicts else 0.0, 1.0)
            
        except Exception:
            return 0.0
    
    async def _analyze_complex_emotion_contradictions(self, emotion_state: Dict) -> float:
        """복합 감정 모순 분석"""
        try:
            # 모순적인 복합 감정 조합들
            contradictory_combinations = [
                ('pride', 'shame'),
                ('hope', 'despair'),
                ('envy', 'contempt'),
                ('guilt', 'pride')
            ]
            
            conflicts = []
            for emotion1, emotion2 in contradictory_combinations:
                score1 = emotion_state.get(emotion1, 0.0)
                score2 = emotion_state.get(emotion2, 0.0)
                
                if score1 > 0.4 and score2 > 0.4:
                    conflicts.append(min(score1, score2))
            
            return min(np.mean(conflicts) if conflicts else 0.0, 1.0)
            
        except Exception:
            return 0.0
    
    async def _analyze_cognitive_dissonance_in_text(self, text: str, emotion_state: Dict) -> float:
        """텍스트 내 인지적 부조화 분석"""
        try:
            # 텍스트에서 모순적 표현 찾기
            contradictory_indicators = [
                ['but', 'however', 'although', 'despite'],
                ['mixed feelings', 'conflicted', 'torn between'],
                ['on one hand', 'on the other hand'],
                ['both', 'neither', 'either or']
            ]
            
            text_lower = text.lower()
            dissonance_score = 0.0
            
            for indicator_group in contradictory_indicators:
                for indicator in indicator_group:
                    if indicator in text_lower:
                        dissonance_score += 0.2
            
            # 감정 일관성과 비교
            emotional_coherence = emotion_state.get('coherence', 0.5)
            
            if emotional_coherence < 0.4:
                dissonance_score += 0.3
            
            return min(dissonance_score, 1.0)
            
        except Exception:
            return 0.0
    
    async def _analyze_value_conflicts(self, text: str, emotion_state: Dict) -> float:
        """가치 갈등 분석"""
        try:
            # 가치 갈등 키워드 분석
            value_conflict_keywords = [
                'should', 'must', 'ought to', 'have to',
                'want', 'desire', 'wish', 'need',
                'right', 'wrong', 'moral', 'ethical'
            ]
            
            text_lower = text.lower()
            conflict_indicators = 0
            
            for keyword in value_conflict_keywords:
                if keyword in text_lower:
                    conflict_indicators += 1
            
            # 자기 평가와 감정 상태 간 불일치
            self_evaluation = emotion_state.get('self_evaluation', 0.5)
            emotional_valence = emotion_state.get('valence', 0.0)
            
            if self_evaluation > 0.6 and emotional_valence < -0.4:
                conflict_indicators += 2
            elif self_evaluation < 0.4 and emotional_valence > 0.4:
                conflict_indicators += 2
            
            return min(conflict_indicators / 10.0, 1.0)
            
        except Exception:
            return 0.0
    
    async def _analyze_approach_avoidance_conflicts(self, text: str, emotion_state: Dict) -> float:
        """접근-회피 갈등 분석"""
        try:
            # 접근-회피 갈등 지표
            approach_keywords = ['want', 'desire', 'attracted', 'drawn to', 'interested']
            avoidance_keywords = ['avoid', 'fear', 'reluctant', 'hesitant', 'scared']
            
            text_lower = text.lower()
            
            approach_score = sum(1 for keyword in approach_keywords if keyword in text_lower)
            avoidance_score = sum(1 for keyword in avoidance_keywords if keyword in text_lower)
            
            # 접근과 회피 동시 존재 시 갈등
            if approach_score > 0 and avoidance_score > 0:
                return min((approach_score + avoidance_score) / 10.0, 1.0)
            
            return 0.0
            
        except Exception:
            return 0.0
    
    async def _detect_double_bind_situations(self, text: str, emotion_state: Dict) -> float:
        """이중 구속 상황 인식"""
        try:
            # 이중 구속 상황 지표
            double_bind_patterns = [
                'damned if you do, damned if you don\'t',
                'can\'t win',
                'no good choice',
                'trapped',
                'stuck between'
            ]
            
            text_lower = text.lower()
            
            for pattern in double_bind_patterns:
                if pattern in text_lower:
                    return 0.8
            
            # 감정 상태에서 이중 구속 신호
            if (emotion_state.get('dominance', 0.5) < 0.3 and 
                emotion_state.get('arousal', 0.5) > 0.6):
                return 0.6
            
            return 0.0
            
        except Exception:
            return 0.0
    
    async def _analyze_emotion_regulation_conflicts(self, text: str, emotion_state: Dict) -> float:
        """감정 조절 갈등 분석"""
        try:
            # 감정 조절 vs 감정 표현 갈등
            emotional_control = emotion_state.get('emotional_control', 0.5)
            emotional_intensity = emotion_state.get('emotional_intensity', 0.5)
            
            # 높은 감정 강도와 높은 통제 욕구 = 갈등
            if emotional_intensity > 0.7 and emotional_control > 0.7:
                return 0.8
            
            # 감정 억제 vs 표현 키워드
            suppression_keywords = ['suppress', 'control', 'hide', 'contain']
            expression_keywords = ['express', 'show', 'reveal', 'let out']
            
            text_lower = text.lower()
            
            suppression_count = sum(1 for keyword in suppression_keywords if keyword in text_lower)
            expression_count = sum(1 for keyword in expression_keywords if keyword in text_lower)
            
            if suppression_count > 0 and expression_count > 0:
                return min((suppression_count + expression_count) / 6.0, 1.0)
            
            return 0.0
            
        except Exception:
            return 0.0
    
    async def _analyze_social_personal_conflicts(self, text: str, emotion_state: Dict) -> float:
        """사회적 기대 vs 개인 욕구 갈등"""
        try:
            # 사회적 기대 키워드
            social_keywords = ['should', 'expected', 'society', 'others think', 'reputation']
            
            # 개인 욕구 키워드
            personal_keywords = ['want', 'need', 'desire', 'feel like', 'personal']
            
            text_lower = text.lower()
            
            social_count = sum(1 for keyword in social_keywords if keyword in text_lower)
            personal_count = sum(1 for keyword in personal_keywords if keyword in text_lower)
            
            if social_count > 0 and personal_count > 0:
                # 사회적 적절성 vs 개인 감정
                social_appropriateness = emotion_state.get('social_appropriateness', 0.5)
                personal_authenticity = 1.0 - social_appropriateness
                
                if abs(social_appropriateness - personal_authenticity) > 0.4:
                    return min((social_count + personal_count) / 8.0, 1.0)
            
            return 0.0
            
        except Exception:
            return 0.0
    
    async def _analyze_temporal_self_conflicts(self, text: str, emotion_state: Dict) -> float:
        """시간적 자아 갈등 분석"""
        try:
            # 현재 vs 미래 자아 갈등
            present_keywords = ['now', 'today', 'currently', 'at the moment']
            future_keywords = ['future', 'tomorrow', 'later', 'eventually', 'someday']
            
            text_lower = text.lower()
            
            present_count = sum(1 for keyword in present_keywords if keyword in text_lower)
            future_count = sum(1 for keyword in future_keywords if keyword in text_lower)
            
            if present_count > 0 and future_count > 0:
                return min((present_count + future_count) / 6.0, 1.0)
            
            return 0.0
            
        except Exception:
            return 0.0
    
    async def _analyze_role_conflicts(self, text: str, emotion_state: Dict) -> float:
        """역할 갈등 분석"""
        try:
            # 역할 갈등 키워드
            role_keywords = ['role', 'responsibility', 'duty', 'obligation', 'position']
            
            text_lower = text.lower()
            role_count = sum(1 for keyword in role_keywords if keyword in text_lower)
            
            if role_count > 0:
                return min(role_count / 5.0, 1.0)
            
            return 0.0
            
        except Exception:
            return 0.0
    
    async def _analyze_moral_conflicts(self, text: str, emotion_state: Dict) -> float:
        """도덕적 갈등 분석"""
        try:
            # 도덕적 갈등 키워드
            moral_keywords = ['moral', 'ethical', 'right', 'wrong', 'conscience', 'guilt']
            
            text_lower = text.lower()
            moral_count = sum(1 for keyword in moral_keywords if keyword in text_lower)
            
            # 죄책감과 자부심의 동시 존재
            guilt = emotion_state.get('guilt', 0.0)
            pride = emotion_state.get('pride', 0.0)
            
            if guilt > 0.4 and pride > 0.4:
                moral_count += 2
            
            return min(moral_count / 8.0, 1.0)
            
        except Exception:
            return 0.0
    
    async def _calculate_conflict_intensity(self, text: str, emotion_state: Dict, base_conflict: float) -> float:
        """갈등 강도 계산"""
        try:
            # 감정 강도 기반 갈등 강도
            emotional_intensity = emotion_state.get('emotional_intensity', 0.5)
            
            # 갈등 관련 언어 강도
            intensity_keywords = ['very', 'extremely', 'really', 'so', 'too', 'quite']
            text_lower = text.lower()
            
            intensity_modifier = 1.0
            for keyword in intensity_keywords:
                if keyword in text_lower:
                    intensity_modifier += 0.1
            
            final_intensity = min(emotional_intensity * intensity_modifier, 1.0)
            
            return final_intensity
            
        except Exception:
            return 1.0
    
    async def _predict_conflict_persistence(self, text: str, emotion_state: Dict, conflict_score: float) -> float:
        """갈등 지속성 예측"""
        try:
            # 갈등의 지속성 예측
            stability = emotion_state.get('emotional_stability', 0.5)
            
            # 안정성이 낮을수록 갈등이 지속될 가능성 높음
            persistence = 1.0 - stability
            
            # 갈등 해결 관련 키워드가 있으면 지속성 감소
            resolution_keywords = ['resolve', 'solve', 'fix', 'overcome', 'deal with']
            text_lower = text.lower()
            
            resolution_count = sum(1 for keyword in resolution_keywords if keyword in text_lower)
            if resolution_count > 0:
                persistence *= 0.8
            
            return max(0.3, min(persistence, 1.0))
            
        except Exception:
            return 1.0
    
    async def _calculate_values_alignment(self, text: str, context: Dict) -> Dict[str, float]:
        """
        개인 가치 정렬도 계산 - 고도화된 가치 체계 분석
        
        고도화된 가치 정렬도 분석 시스템:
        - Schwartz 가치 체계 기반 분석 (10개 기본 가치)
        - 도덕적 기초 이론 (Moral Foundations Theory) 적용
        - 개인-사회 가치 균형 분석
        - 내재적 vs 외재적 동기 분석
        - 가치 우선순위 계층 분석
        - 상황별 가치 활성화 분석
        
        Args:
            text: 분석할 텍스트
            context: 상황 컨텍스트
            
        Returns:
            다차원 가치 정렬도 점수
        """
        try:
            # 1. Schwartz 10개 기본 가치 분석
            schwartz_values = await self._analyze_schwartz_values(text, context)
            
            # 2. 도덕적 기초 이론 5개 차원 분석
            moral_foundations = await self._analyze_moral_foundations(text, context)
            
            # 3. 개인-사회 가치 균형 분석
            individual_collective_balance = await self._analyze_individual_collective_values(text, context)
            
            # 4. 내재적 vs 외재적 동기 분석
            intrinsic_extrinsic_motivation = await self._analyze_intrinsic_extrinsic_motivation(text, context)
            
            # 5. 가치 우선순위 계층 분석
            value_hierarchy = await self._analyze_value_hierarchy(text, context)
            
            # 6. 상황별 가치 활성화 분석
            contextual_value_activation = await self._analyze_contextual_value_activation(text, context)
            
            # 7. 가치 갈등 및 조화 분석
            value_conflicts_harmony = await self._analyze_value_conflicts_harmony(text, context)
            
            # 8. 문화적 가치 영향 분석
            cultural_value_influence = await self._analyze_cultural_value_influence(text, context)
            
            # 9. 종합 가치 정렬도 구성
            comprehensive_values_alignment = {
                # Schwartz 10개 기본 가치
                'self_direction': schwartz_values['self_direction'],
                'stimulation': schwartz_values['stimulation'],
                'hedonism': schwartz_values['hedonism'],
                'achievement': schwartz_values['achievement'],
                'power': schwartz_values['power'],
                'security': schwartz_values['security'],
                'conformity': schwartz_values['conformity'],
                'tradition': schwartz_values['tradition'],
                'benevolence': schwartz_values['benevolence'],
                'universalism': schwartz_values['universalism'],
                
                # 도덕적 기초 이론 5개 차원
                'care_harm': moral_foundations['care_harm'],
                'fairness_cheating': moral_foundations['fairness_cheating'],
                'loyalty_betrayal': moral_foundations['loyalty_betrayal'],
                'authority_subversion': moral_foundations['authority_subversion'],
                'sanctity_degradation': moral_foundations['sanctity_degradation'],
                
                # 개인-사회 균형
                'individualism': individual_collective_balance['individualism'],
                'collectivism': individual_collective_balance['collectivism'],
                'balance_score': individual_collective_balance['balance_score'],
                
                # 내재적-외재적 동기
                'intrinsic_motivation': intrinsic_extrinsic_motivation['intrinsic'],
                'extrinsic_motivation': intrinsic_extrinsic_motivation['extrinsic'],
                'motivation_autonomy': intrinsic_extrinsic_motivation['autonomy'],
                
                # 가치 계층
                'core_values_strength': value_hierarchy['core_strength'],
                'peripheral_values_strength': value_hierarchy['peripheral_strength'],
                'value_system_coherence': value_hierarchy['coherence'],
                
                # 상황별 활성화
                'contextual_flexibility': contextual_value_activation['flexibility'],
                'contextual_consistency': contextual_value_activation['consistency'],
                'adaptive_capacity': contextual_value_activation['adaptive_capacity'],
                
                # 가치 갈등과 조화
                'value_conflict_level': value_conflicts_harmony['conflict_level'],
                'value_harmony_level': value_conflicts_harmony['harmony_level'],
                'integration_capacity': value_conflicts_harmony['integration_capacity'],
                
                # 문화적 영향
                'cultural_alignment': cultural_value_influence['alignment'],
                'cultural_independence': cultural_value_influence['independence'],
                'cultural_adaptation': cultural_value_influence['adaptation']
            }
            
            # 10. 가치 정렬도 일관성 검증
            alignment_consistency = await self._verify_values_alignment_consistency(
                comprehensive_values_alignment
            )
            
            comprehensive_values_alignment['overall_consistency'] = alignment_consistency
            
            return comprehensive_values_alignment
            
        except Exception as e:
            self.logger.error(f"가치 정렬도 계산 실패: {e}")
            # 기본 가치 정렬도 반환
            return {
                'autonomy': 0.5,
                'justice': 0.5,
                'compassion': 0.5,
                'loyalty': 0.5,
                'overall_consistency': 0.0
            }
    
    async def _analyze_schwartz_values(self, text: str, context: Dict) -> Dict[str, float]:
        """Schwartz 10개 기본 가치 분석"""
        try:
            # 각 가치별 키워드 사전
            value_keywords = {
                'self_direction': ['freedom', 'independence', 'creativity', 'choosing', 'exploring'],
                'stimulation': ['excitement', 'novelty', 'challenge', 'adventure', 'variety'],
                'hedonism': ['pleasure', 'enjoyment', 'fun', 'gratification', 'satisfaction'],
                'achievement': ['success', 'accomplishment', 'competence', 'ambition', 'influence'],
                'power': ['authority', 'control', 'dominance', 'prestige', 'wealth'],
                'security': ['safety', 'stability', 'order', 'protection', 'certainty'],
                'conformity': ['obedience', 'compliance', 'restraint', 'politeness', 'discipline'],
                'tradition': ['respect', 'commitment', 'acceptance', 'customs', 'culture'],
                'benevolence': ['helpfulness', 'loyalty', 'forgiveness', 'honesty', 'responsibility'],
                'universalism': ['understanding', 'tolerance', 'protection', 'justice', 'equality']
            }
            
            text_lower = text.lower()
            value_scores = {}
            
            for value, keywords in value_keywords.items():
                score = 0.0
                for keyword in keywords:
                    if keyword in text_lower:
                        score += 0.2
                
                # 상황적 맥락 고려
                if context.get('situation_type') == 'social' and value in ['benevolence', 'universalism']:
                    score += 0.1
                elif context.get('situation_type') == 'competitive' and value in ['achievement', 'power']:
                    score += 0.1
                elif context.get('situation_type') == 'creative' and value in ['self_direction', 'stimulation']:
                    score += 0.1
                
                value_scores[value] = min(score, 1.0)
            
            return value_scores
            
        except Exception:
            return {key: 0.5 for key in ['self_direction', 'stimulation', 'hedonism', 'achievement', 'power', 'security', 'conformity', 'tradition', 'benevolence', 'universalism']}
    
    async def _analyze_moral_foundations(self, text: str, context: Dict) -> Dict[str, float]:
        """도덕적 기초 이론 5개 차원 분석"""
        try:
            # 도덕적 기초 이론 키워드 사전
            moral_keywords = {
                'care_harm': ['care', 'harm', 'compassion', 'suffering', 'cruelty', 'kindness'],
                'fairness_cheating': ['fair', 'unfair', 'justice', 'rights', 'cheat', 'honest'],
                'loyalty_betrayal': ['loyalty', 'betrayal', 'patriotism', 'treason', 'team', 'group'],
                'authority_subversion': ['authority', 'respect', 'tradition', 'hierarchy', 'obedience', 'rebel'],
                'sanctity_degradation': ['sacred', 'pure', 'holy', 'disgusting', 'degrading', 'noble']
            }
            
            text_lower = text.lower()
            moral_scores = {}
            
            for foundation, keywords in moral_keywords.items():
                score = 0.0
                for keyword in keywords:
                    if keyword in text_lower:
                        score += 0.15
                
                # 상황적 맥락 고려
                if context.get('moral_context') == 'interpersonal' and foundation == 'care_harm':
                    score += 0.2
                elif context.get('moral_context') == 'social_justice' and foundation == 'fairness_cheating':
                    score += 0.2
                elif context.get('moral_context') == 'group_dynamics' and foundation == 'loyalty_betrayal':
                    score += 0.2
                
                moral_scores[foundation] = min(score, 1.0)
            
            return moral_scores
            
        except Exception:
            return {key: 0.5 for key in ['care_harm', 'fairness_cheating', 'loyalty_betrayal', 'authority_subversion', 'sanctity_degradation']}
    
    async def _analyze_individual_collective_values(self, text: str, context: Dict) -> Dict[str, float]:
        """개인-사회 가치 균형 분석"""
        try:
            # 개인주의 키워드
            individualism_keywords = ['individual', 'personal', 'self', 'independent', 'unique', 'autonomy']
            
            # 집단주의 키워드
            collectivism_keywords = ['group', 'community', 'together', 'collective', 'social', 'team']
            
            text_lower = text.lower()
            
            individualism_score = sum(0.1 for keyword in individualism_keywords if keyword in text_lower)
            collectivism_score = sum(0.1 for keyword in collectivism_keywords if keyword in text_lower)
            
            # 균형 점수 계산
            total_score = individualism_score + collectivism_score
            if total_score > 0:
                balance_score = 1.0 - abs(individualism_score - collectivism_score) / total_score
            else:
                balance_score = 0.5
            
            return {
                'individualism': min(individualism_score, 1.0),
                'collectivism': min(collectivism_score, 1.0),
                'balance_score': balance_score
            }
            
        except Exception:
            return {'individualism': 0.5, 'collectivism': 0.5, 'balance_score': 0.5}
    
    async def _analyze_intrinsic_extrinsic_motivation(self, text: str, context: Dict) -> Dict[str, float]:
        """내재적 vs 외재적 동기 분석"""
        try:
            # 내재적 동기 키워드
            intrinsic_keywords = ['enjoy', 'interested', 'curious', 'passionate', 'love', 'meaningful']
            
            # 외재적 동기 키워드
            extrinsic_keywords = ['reward', 'money', 'recognition', 'status', 'approval', 'punishment']
            
            text_lower = text.lower()
            
            intrinsic_score = sum(0.15 for keyword in intrinsic_keywords if keyword in text_lower)
            extrinsic_score = sum(0.15 for keyword in extrinsic_keywords if keyword in text_lower)
            
            # 자율성 점수 (내재적 동기와 관련)
            autonomy_score = intrinsic_score * 0.8
            
            return {
                'intrinsic': min(intrinsic_score, 1.0),
                'extrinsic': min(extrinsic_score, 1.0),
                'autonomy': min(autonomy_score, 1.0)
            }
            
        except Exception:
            return {'intrinsic': 0.5, 'extrinsic': 0.5, 'autonomy': 0.5}
    
    async def _analyze_value_hierarchy(self, text: str, context: Dict) -> Dict[str, float]:
        """가치 우선순위 계층 분석"""
        try:
            # 핵심 가치 키워드 (강한 표현)
            core_value_keywords = ['most important', 'essential', 'fundamental', 'core', 'primary']
            
            # 주변 가치 키워드 (약한 표현)
            peripheral_value_keywords = ['nice to have', 'secondary', 'optional', 'sometimes', 'occasionally']
            
            text_lower = text.lower()
            
            core_strength = sum(0.2 for keyword in core_value_keywords if keyword in text_lower)
            peripheral_strength = sum(0.1 for keyword in peripheral_value_keywords if keyword in text_lower)
            
            # 가치 체계 일관성
            total_strength = core_strength + peripheral_strength
            if total_strength > 0:
                coherence = core_strength / total_strength
            else:
                coherence = 0.5
            
            return {
                'core_strength': min(core_strength, 1.0),
                'peripheral_strength': min(peripheral_strength, 1.0),
                'coherence': coherence
            }
            
        except Exception:
            return {'core_strength': 0.5, 'peripheral_strength': 0.5, 'coherence': 0.5}
    
    async def _analyze_contextual_value_activation(self, text: str, context: Dict) -> Dict[str, float]:
        """상황별 가치 활성화 분석"""
        try:
            # 맥락 유연성 키워드
            flexibility_keywords = ['adapt', 'adjust', 'flexible', 'context', 'situation']
            
            # 일관성 키워드
            consistency_keywords = ['consistent', 'always', 'never', 'principle', 'unchanging']
            
            text_lower = text.lower()
            
            flexibility_score = sum(0.15 for keyword in flexibility_keywords if keyword in text_lower)
            consistency_score = sum(0.15 for keyword in consistency_keywords if keyword in text_lower)
            
            # 적응 능력 (유연성과 일관성의 균형)
            adaptive_capacity = (flexibility_score + consistency_score) / 2.0
            
            return {
                'flexibility': min(flexibility_score, 1.0),
                'consistency': min(consistency_score, 1.0),
                'adaptive_capacity': min(adaptive_capacity, 1.0)
            }
            
        except Exception:
            return {'flexibility': 0.5, 'consistency': 0.5, 'adaptive_capacity': 0.5}
    
    async def _analyze_value_conflicts_harmony(self, text: str, context: Dict) -> Dict[str, float]:
        """가치 갈등과 조화 분석"""
        try:
            # 갈등 키워드
            conflict_keywords = ['conflict', 'torn', 'dilemma', 'struggle', 'contradiction']
            
            # 조화 키워드
            harmony_keywords = ['harmony', 'balance', 'integration', 'align', 'coherent']
            
            text_lower = text.lower()
            
            conflict_level = sum(0.2 for keyword in conflict_keywords if keyword in text_lower)
            harmony_level = sum(0.2 for keyword in harmony_keywords if keyword in text_lower)
            
            # 통합 능력 (갈등을 조화로 전환하는 능력)
            if conflict_level > 0:
                integration_capacity = harmony_level / (conflict_level + harmony_level)
            else:
                integration_capacity = harmony_level
            
            return {
                'conflict_level': min(conflict_level, 1.0),
                'harmony_level': min(harmony_level, 1.0),
                'integration_capacity': min(integration_capacity, 1.0)
            }
            
        except Exception:
            return {'conflict_level': 0.5, 'harmony_level': 0.5, 'integration_capacity': 0.5}
    
    async def _analyze_cultural_value_influence(self, text: str, context: Dict) -> Dict[str, float]:
        """문화적 가치 영향 분석"""
        try:
            # 문화적 정렬 키워드
            cultural_alignment_keywords = ['culture', 'tradition', 'society', 'community', 'heritage']
            
            # 문화적 독립성 키워드
            cultural_independence_keywords = ['independent', 'different', 'unique', 'individual', 'personal']
            
            text_lower = text.lower()
            
            alignment_score = sum(0.15 for keyword in cultural_alignment_keywords if keyword in text_lower)
            independence_score = sum(0.15 for keyword in cultural_independence_keywords if keyword in text_lower)
            
            # 문화적 적응 능력
            adaptation_score = (alignment_score + independence_score) / 2.0
            
            return {
                'alignment': min(alignment_score, 1.0),
                'independence': min(independence_score, 1.0),
                'adaptation': min(adaptation_score, 1.0)
            }
            
        except Exception:
            return {'alignment': 0.5, 'independence': 0.5, 'adaptation': 0.5}
    
    async def _verify_values_alignment_consistency(self, values_alignment: Dict[str, float]) -> float:
        """가치 정렬도 일관성 검증"""
        try:
            # 상충하는 가치 쌍들
            conflicting_pairs = [
                ('power', 'benevolence'),
                ('achievement', 'universalism'),
                ('hedonism', 'conformity'),
                ('stimulation', 'security'),
                ('self_direction', 'tradition')
            ]
            
            inconsistencies = []
            
            for value1, value2 in conflicting_pairs:
                if value1 in values_alignment and value2 in values_alignment:
                    score1 = values_alignment[value1]
                    score2 = values_alignment[value2]
                    
                    # 두 상충 가치가 모두 높으면 일관성 문제
                    if score1 > 0.7 and score2 > 0.7:
                        inconsistencies.append(min(score1, score2))
            
            # 일관성 점수 계산
            if inconsistencies:
                consistency_score = 1.0 - np.mean(inconsistencies)
            else:
                consistency_score = 1.0
            
            return max(0.0, min(1.0, consistency_score))
            
        except Exception:
            return 0.5
    
    async def _calculate_cognitive_dissonance(self, emotion_state: Dict, values: Dict) -> float:
        """
        인지 부조화 계산 - Leon Festinger의 인지 부조화 이론 기반
        
        2024년 연구 기반 구현:
        - 감정과 가치 간 불일치 정도 계산
        - 다차원 충돌 분석
        - 동적 임계값 조정
        - 심리적 스트레스 지수 포함
        """
        try:
            # 1. 감정-가치 불일치 매트릭스 계산
            dissonance_matrix = {}
            
            # 1.1 가치 차원별 감정 충돌 분석
            for value_name, value_score in values.items():
                emotion_conflicts = []
                
                # 각 감정 차원과 가치 간 충돌 계산
                for emotion_dim, emotion_score in emotion_state.items():
                    conflict_score = await self._calculate_emotion_value_conflict(
                        emotion_dim, emotion_score, value_name, value_score
                    )
                    emotion_conflicts.append(conflict_score)
                
                # 가치별 평균 충돌 점수
                dissonance_matrix[value_name] = np.mean(emotion_conflicts) if emotion_conflicts else 0.0
            
            # 2. 인지 부조화 강도 계산
            base_dissonance = np.mean(list(dissonance_matrix.values()))
            
            # 2.1 부조화 증폭 요인들
            amplification_factors = []
            
            # 2.1.1 감정 극성 충돌 (긍정/부정 가치와 감정 간 충돌)
            valence_conflict = await self._calculate_valence_dissonance(emotion_state, values)
            amplification_factors.append(valence_conflict)
            
            # 2.1.2 가치 위계 충돌 (중요한 가치와 감정 간 충돌)
            hierarchy_conflict = await self._calculate_hierarchy_dissonance(emotion_state, values)
            amplification_factors.append(hierarchy_conflict)
            
            # 2.1.3 시간적 불일치 (과거 vs 현재 가치-감정 충돌)
            temporal_conflict = await self._calculate_temporal_dissonance(emotion_state, values)
            amplification_factors.append(temporal_conflict)
            
            # 2.1.4 사회적 기대 충돌 (사회적 가치와 개인 감정 간 충돌)
            social_conflict = await self._calculate_social_dissonance(emotion_state, values)
            amplification_factors.append(social_conflict)
            
            # 2.1.5 행동-태도 불일치 (표현된 감정과 내재된 가치 간 충돌)
            behavioral_conflict = await self._calculate_behavioral_dissonance(emotion_state, values)
            amplification_factors.append(behavioral_conflict)
            
            # 3. 통합 부조화 점수 계산
            amplification_factor = np.mean(amplification_factors)
            integrated_dissonance = base_dissonance * (1 + amplification_factor)
            
            # 4. 부조화 해소 동기 계산
            resolution_motivation = await self._calculate_dissonance_resolution_motivation(
                integrated_dissonance, emotion_state, values
            )
            
            # 5. 최종 인지 부조화 점수 (0-1 범위)
            final_dissonance = min(1.0, integrated_dissonance * resolution_motivation)
            
            # 6. 부조화 메타데이터 저장
            dissonance_metadata = {
                'base_dissonance': base_dissonance,
                'amplification_factor': amplification_factor,
                'resolution_motivation': resolution_motivation,
                'conflict_sources': dissonance_matrix,
                'psychological_stress_level': min(1.0, final_dissonance * 1.2)
            }
            
            return final_dissonance
            
        except Exception as e:
            self.logger.error(f"인지 부조화 계산 실패: {e}")
            return 0.0
    
    def _identify_reflection_triggers(self, text: str) -> List[str]:
        """성찰 트리거 식별"""
        triggers = []
        if '후회' in text or 'regret' in text.lower():
            triggers.append('regret')
        if '죄책감' in text or 'guilt' in text.lower():
            triggers.append('guilt')
        return triggers
    
    async def _predict_other_emotion(self, text: str, context: Dict) -> Dict[str, float]:
        """
        타인 감정 예측 - 고급 공감 예측 알고리즘
        
        2024년 연구 기반 구현:
        - Transformer 기반 감정 예측 모델
        - 다중 모달리티 감정 추론
        - 심리학적 이론 기반 감정 투사
        - 불확실성 정량화
        """
        try:
            # 1. 텍스트 기반 감정 예측
            text_emotions = await self._analyze_text_emotional_markers(text)
            
            # 2. 맥락 기반 감정 보정
            context_emotions = await self._infer_contextual_emotions(context)
            
            # 3. 다중 이론 기반 감정 예측
            prediction_theories = []
            
            # 3.1 Simulation Theory 기반 예측
            simulation_prediction = await self._simulate_emotion_through_self_projection(text, context)
            prediction_theories.append(('simulation', simulation_prediction))
            
            # 3.2 Theory-Theory 기반 예측
            theory_prediction = await self._predict_emotion_through_theory_of_mind(text, context)
            prediction_theories.append(('theory', theory_prediction))
            
            # 3.3 Interaction Theory 기반 예측
            interaction_prediction = await self._predict_emotion_through_interaction_dynamics(text, context)
            prediction_theories.append(('interaction', interaction_prediction))
            
            # 3.4 Embodied Cognition 기반 예측
            embodied_prediction = await self._predict_emotion_through_embodied_cognition(text, context)
            prediction_theories.append(('embodied', embodied_prediction))
            
            # 4. 예측 신뢰도 계산
            theory_confidences = []
            for theory_name, prediction in prediction_theories:
                confidence = await self._calculate_theory_confidence(theory_name, text, context)
                theory_confidences.append(confidence)
            
            # 5. 가중 융합 예측
            final_prediction = await self._weighted_emotion_fusion(
                prediction_theories, theory_confidences
            )
            
            # 6. 불확실성 정량화
            prediction_uncertainty = await self._calculate_prediction_uncertainty(
                prediction_theories, theory_confidences
            )
            
            # 7. 감정 차원별 세분화
            dimensional_emotions = await self._convert_to_dimensional_emotions(final_prediction)
            
            # 8. 기본 감정 분류
            basic_emotions = await self._convert_to_basic_emotions(final_prediction)
            
            # 9. 복합 감정 분석
            complex_emotions = await self._analyze_complex_emotions(final_prediction)
            
            # 10. 전체 예측 결과 통합
            integrated_prediction = {
                **dimensional_emotions,
                **basic_emotions,
                **complex_emotions,
                'prediction_confidence': 1.0 - prediction_uncertainty,
                'uncertainty_level': prediction_uncertainty,
                'dominant_theory': prediction_theories[np.argmax(theory_confidences)][0],
                'theory_consensus': np.std(theory_confidences)
            }
            
            return integrated_prediction
            
        except Exception as e:
            self.logger.error(f"타인 감정 예측 실패: {e}")
            return {
                'valence': 0.0,
                'arousal': 0.0,
                'dominance': 0.0,
                'confidence': 0.0
            }
    
    async def _calculate_theory_of_mind(self, text: str, predicted_emotion: Dict) -> float:
        """
        Theory of Mind 점수 계산 - 고급 마음 이론 평가
        
        2024년 연구 기반 구현:
        - False Belief Task 기반 평가
        - 다층 마음 이론 (제2차, 제3차 마음 이론)
        - 인지적 공감 vs 정서적 공감 분리
        - 신경인지학적 ToM 모델
        """
        try:
            # 1. 기본 마음 이론 능력 평가
            basic_tom_scores = []
            
            # 1.1 False Belief 추론 능력
            false_belief_score = await self._evaluate_false_belief_reasoning(text, predicted_emotion)
            basic_tom_scores.append(false_belief_score)
            
            # 1.2 의도 추론 능력 (Intentionality)
            intention_score = await self._evaluate_intention_inference(text, predicted_emotion)
            basic_tom_scores.append(intention_score)
            
            # 1.3 욕구 추론 능력 (Desire Attribution)
            desire_score = await self._evaluate_desire_attribution(text, predicted_emotion)
            basic_tom_scores.append(desire_score)
            
            # 1.4 지식 상태 추론 능력 (Knowledge Attribution)
            knowledge_score = await self._evaluate_knowledge_attribution(text, predicted_emotion)
            basic_tom_scores.append(knowledge_score)
            
            # 2. 고급 마음 이론 능력 평가
            advanced_tom_scores = []
            
            # 2.1 제2차 마음 이론 (Second-order ToM)
            second_order_score = await self._evaluate_second_order_tom(text, predicted_emotion)
            advanced_tom_scores.append(second_order_score)
            
            # 2.2 제3차 마음 이론 (Third-order ToM)
            third_order_score = await self._evaluate_third_order_tom(text, predicted_emotion)
            advanced_tom_scores.append(third_order_score)
            
            # 2.3 반성적 마음 이론 (Reflective ToM)
            reflective_score = await self._evaluate_reflective_tom(text, predicted_emotion)
            advanced_tom_scores.append(reflective_score)
            
            # 2.4 상황적 마음 이론 (Contextual ToM)
            contextual_score = await self._evaluate_contextual_tom(text, predicted_emotion)
            advanced_tom_scores.append(contextual_score)
            
            # 3. 인지적 vs 정서적 공감 분리
            cognitive_empathy = await self._evaluate_cognitive_empathy(text, predicted_emotion)
            affective_empathy = await self._evaluate_affective_empathy(text, predicted_emotion)
            
            # 4. 신경인지학적 ToM 모델링
            neurological_tom = await self._model_neurological_tom(text, predicted_emotion)
            
            # 5. 발달적 ToM 평가
            developmental_tom = await self._evaluate_developmental_tom_level(text, predicted_emotion)
            
            # 6. 문화적 ToM 조정
            cultural_tom = await self._adjust_cultural_tom(text, predicted_emotion)
            
            # 7. 개인차 ToM 모델링
            individual_tom = await self._model_individual_tom_differences(text, predicted_emotion)
            
            # 8. 통합 ToM 점수 계산
            basic_tom_score = np.mean(basic_tom_scores)
            advanced_tom_score = np.mean(advanced_tom_scores)
            
            # 가중 평균 계산
            integrated_tom_score = (
                basic_tom_score * 0.3 +
                advanced_tom_score * 0.25 +
                cognitive_empathy * 0.15 +
                affective_empathy * 0.1 +
                neurological_tom * 0.1 +
                developmental_tom * 0.05 +
                cultural_tom * 0.03 +
                individual_tom * 0.02
            )
            
            # 9. 불확실성 조정
            tom_uncertainty = await self._calculate_tom_uncertainty(
                basic_tom_scores, advanced_tom_scores, cognitive_empathy, affective_empathy
            )
            
            # 10. 최종 ToM 점수 (불확실성 페널티 적용)
            final_tom_score = integrated_tom_score * (1 - tom_uncertainty * 0.3)
            
            # 11. 메타인지적 ToM 평가
            metacognitive_tom = await self._evaluate_metacognitive_tom(final_tom_score)
            
            # 12. 최종 점수 조정
            adjusted_tom_score = final_tom_score * (1 + metacognitive_tom * 0.1)
            
            return min(1.0, max(0.0, adjusted_tom_score))
            
        except Exception as e:
            self.logger.error(f"Theory of Mind 계산 실패: {e}")
            return 0.0
    
    async def _calculate_empathy_intensity(self, predicted_emotion: Dict, context: Dict) -> float:
        """
        공감 강도 계산 - 다차원 공감 강도 모델
        
        2024년 연구 기반 구현:
        - 정서적 공감 vs 인지적 공감 분리
        - 개인적 괴로움 vs 동정심 분리
        - 상황적 공감 조절 요인
        - 신경과학적 공감 모델
        """
        try:
            # 1. 기본 감정 강도 계산
            base_intensity_factors = []
            
            # 1.1 감정 극성 강도 (Valence Intensity)
            valence_intensity = abs(predicted_emotion.get('valence', 0))
            base_intensity_factors.append(valence_intensity)
            
            # 1.2 감정 각성 강도 (Arousal Intensity)
            arousal_intensity = abs(predicted_emotion.get('arousal', 0))
            base_intensity_factors.append(arousal_intensity)
            
            # 1.3 감정 지배력 강도 (Dominance Intensity)
            dominance_intensity = abs(predicted_emotion.get('dominance', 0))
            base_intensity_factors.append(dominance_intensity)
            
            # 1.4 복합 감정 강도 (Complex Emotion Intensity)
            complex_intensity = await self._calculate_complex_emotion_intensity(predicted_emotion)
            base_intensity_factors.append(complex_intensity)
            
            # 2. 공감 유형별 강도 계산
            empathy_type_intensities = []
            
            # 2.1 정서적 공감 강도 (Affective Empathy)
            affective_intensity = await self._calculate_affective_empathy_intensity(predicted_emotion, context)
            empathy_type_intensities.append(affective_intensity)
            
            # 2.2 인지적 공감 강도 (Cognitive Empathy)
            cognitive_intensity = await self._calculate_cognitive_empathy_intensity(predicted_emotion, context)
            empathy_type_intensities.append(cognitive_intensity)
            
            # 2.3 동정심 강도 (Compassion Intensity)
            compassion_intensity = await self._calculate_compassion_intensity(predicted_emotion, context)
            empathy_type_intensities.append(compassion_intensity)
            
            # 2.4 개인적 괴로움 강도 (Personal Distress)
            personal_distress = await self._calculate_personal_distress_intensity(predicted_emotion, context)
            empathy_type_intensities.append(personal_distress)
            
            # 3. 상황적 조절 요인들
            situational_modulators = []
            
            # 3.1 친밀도 조절 (Intimacy Modulation)
            intimacy_modulation = await self._calculate_intimacy_modulation(context)
            situational_modulators.append(intimacy_modulation)
            
            # 3.2 사회적 거리 조절 (Social Distance Modulation)
            social_distance_modulation = await self._calculate_social_distance_modulation(context)
            situational_modulators.append(social_distance_modulation)
            
            # 3.3 문화적 유사성 조절 (Cultural Similarity Modulation)
            cultural_modulation = await self._calculate_cultural_similarity_modulation(context)
            situational_modulators.append(cultural_modulation)
            
            # 3.4 도덕적 판단 조절 (Moral Judgment Modulation)
            moral_modulation = await self._calculate_moral_judgment_modulation(predicted_emotion, context)
            situational_modulators.append(moral_modulation)
            
            # 4. 신경과학적 공감 모델링
            neurological_factors = []
            
            # 4.1 거울 뉴런 활성화 (Mirror Neuron Activation)
            mirror_activation = await self._calculate_mirror_neuron_activation(predicted_emotion)
            neurological_factors.append(mirror_activation)
            
            # 4.2 정서 전염 (Emotional Contagion)
            emotional_contagion = await self._calculate_emotional_contagion_intensity(predicted_emotion, context)
            neurological_factors.append(emotional_contagion)
            
            # 4.3 공감 네트워크 활성화 (Empathy Network Activation)
            empathy_network = await self._calculate_empathy_network_activation(predicted_emotion, context)
            neurological_factors.append(empathy_network)
            
            # 5. 개인차 요인들
            individual_factors = []
            
            # 5.1 공감 능력 개인차 (Individual Empathy Capacity)
            individual_empathy = await self._calculate_individual_empathy_capacity(context)
            individual_factors.append(individual_empathy)
            
            # 5.2 감정 조절 능력 (Emotion Regulation Capacity)
            emotion_regulation = await self._calculate_emotion_regulation_capacity(predicted_emotion, context)
            individual_factors.append(emotion_regulation)
            
            # 5.3 스트레스 및 피로 상태 (Stress and Fatigue State)
            stress_fatigue = await self._calculate_stress_fatigue_impact(context)
            individual_factors.append(stress_fatigue)
            
            # 6. 통합 공감 강도 계산
            base_intensity = np.mean(base_intensity_factors)
            empathy_type_intensity = np.mean(empathy_type_intensities)
            situational_modulation = np.mean(situational_modulators)
            neurological_intensity = np.mean(neurological_factors)
            individual_modulation = np.mean(individual_factors)
            
            # 7. 가중 통합 공감 강도
            integrated_intensity = (
                base_intensity * 0.25 +
                empathy_type_intensity * 0.30 +
                neurological_intensity * 0.20 +
                situational_modulation * 0.15 +
                individual_modulation * 0.10
            )
            
            # 8. 동적 강도 조절 (Dynamic Intensity Adjustment)
            dynamic_adjustment = await self._calculate_dynamic_intensity_adjustment(
                integrated_intensity, predicted_emotion, context
            )
            
            # 9. 최종 공감 강도 계산
            final_intensity = integrated_intensity * dynamic_adjustment
            
            # 10. 강도 임계값 처리
            processed_intensity = await self._apply_intensity_thresholds(final_intensity)
            
            return min(1.0, max(0.0, processed_intensity))
            
        except Exception as e:
            self.logger.error(f"공감 강도 계산 실패: {e}")
            return 0.0
    
    async def _calculate_perspective_taking(self, text: str, predicted_emotion: Dict) -> float:
        """
        관점 취하기 점수 계산 - 고급 관점 취하기 능력 평가
        
        2024년 연구 기반 구현:
        - 인지적 관점 취하기 vs 정서적 관점 취하기
        - 다층 관점 취하기 (self-other-observer)
        - 상황적 관점 변화 추적
        - 문화적 관점 취하기 능력
        """
        try:
            # 1. 기본 관점 취하기 능력 평가
            basic_perspective_scores = []
            
            # 1.1 인지적 관점 취하기 (Cognitive Perspective Taking)
            cognitive_perspective = await self._evaluate_cognitive_perspective_taking(text, predicted_emotion)
            basic_perspective_scores.append(cognitive_perspective)
            
            # 1.2 정서적 관점 취하기 (Emotional Perspective Taking)
            emotional_perspective = await self._evaluate_emotional_perspective_taking(text, predicted_emotion)
            basic_perspective_scores.append(emotional_perspective)
            
            # 1.3 공간적 관점 취하기 (Spatial Perspective Taking)
            spatial_perspective = await self._evaluate_spatial_perspective_taking(text, predicted_emotion)
            basic_perspective_scores.append(spatial_perspective)
            
            # 1.4 시간적 관점 취하기 (Temporal Perspective Taking)
            temporal_perspective = await self._evaluate_temporal_perspective_taking(text, predicted_emotion)
            basic_perspective_scores.append(temporal_perspective)
            
            # 2. 다층 관점 취하기 평가
            multilayer_perspective_scores = []
            
            # 2.1 자기 관점 인식 (Self-Perspective Awareness)
            self_perspective = await self._evaluate_self_perspective_awareness(text, predicted_emotion)
            multilayer_perspective_scores.append(self_perspective)
            
            # 2.2 타인 관점 추론 (Other-Perspective Inference)
            other_perspective = await self._evaluate_other_perspective_inference(text, predicted_emotion)
            multilayer_perspective_scores.append(other_perspective)
            
            # 2.3 관찰자 관점 모델링 (Observer Perspective Modeling)
            observer_perspective = await self._evaluate_observer_perspective_modeling(text, predicted_emotion)
            multilayer_perspective_scores.append(observer_perspective)
            
            # 2.4 메타 관점 취하기 (Meta-Perspective Taking)
            meta_perspective = await self._evaluate_meta_perspective_taking(text, predicted_emotion)
            multilayer_perspective_scores.append(meta_perspective)
            
            # 3. 상황적 관점 변화 추적
            situational_perspective_scores = []
            
            # 3.1 역할 기반 관점 변화 (Role-Based Perspective Change)
            role_perspective = await self._evaluate_role_based_perspective_change(text, predicted_emotion)
            situational_perspective_scores.append(role_perspective)
            
            # 3.2 상황적 맥락 관점 변화 (Contextual Perspective Change)
            contextual_perspective = await self._evaluate_contextual_perspective_change(text, predicted_emotion)
            situational_perspective_scores.append(contextual_perspective)
            
            # 3.3 도덕적 관점 취하기 (Moral Perspective Taking)
            moral_perspective = await self._evaluate_moral_perspective_taking(text, predicted_emotion)
            situational_perspective_scores.append(moral_perspective)
            
            # 3.4 갈등 상황 관점 취하기 (Conflict Perspective Taking)
            conflict_perspective = await self._evaluate_conflict_perspective_taking(text, predicted_emotion)
            situational_perspective_scores.append(conflict_perspective)
            
            # 4. 문화적 관점 취하기 능력
            cultural_perspective_scores = []
            
            # 4.1 문화적 차이 인식 (Cultural Difference Recognition)
            cultural_difference = await self._evaluate_cultural_difference_recognition(text, predicted_emotion)
            cultural_perspective_scores.append(cultural_difference)
            
            # 4.2 문화적 가치 관점 취하기 (Cultural Value Perspective)
            cultural_value = await self._evaluate_cultural_value_perspective(text, predicted_emotion)
            cultural_perspective_scores.append(cultural_value)
            
            # 4.3 언어적 관점 취하기 (Linguistic Perspective Taking)
            linguistic_perspective = await self._evaluate_linguistic_perspective_taking(text, predicted_emotion)
            cultural_perspective_scores.append(linguistic_perspective)
            
            # 5. 관점 취하기 정확도 평가
            accuracy_scores = []
            
            # 5.1 관점 예측 정확도 (Perspective Prediction Accuracy)
            prediction_accuracy = await self._evaluate_perspective_prediction_accuracy(text, predicted_emotion)
            accuracy_scores.append(prediction_accuracy)
            
            # 5.2 관점 일관성 (Perspective Consistency)
            perspective_consistency = await self._evaluate_perspective_consistency(text, predicted_emotion)
            accuracy_scores.append(perspective_consistency)
            
            # 5.3 관점 복잡성 처리 (Perspective Complexity Handling)
            complexity_handling = await self._evaluate_perspective_complexity_handling(text, predicted_emotion)
            accuracy_scores.append(complexity_handling)
            
            # 6. 관점 취하기 유연성 평가
            flexibility_scores = []
            
            # 6.1 관점 전환 능력 (Perspective Switching Ability)
            switching_ability = await self._evaluate_perspective_switching_ability(text, predicted_emotion)
            flexibility_scores.append(switching_ability)
            
            # 6.2 다중 관점 통합 (Multiple Perspective Integration)
            multiple_integration = await self._evaluate_multiple_perspective_integration(text, predicted_emotion)
            flexibility_scores.append(multiple_integration)
            
            # 6.3 관점 충돌 해결 (Perspective Conflict Resolution)
            conflict_resolution = await self._evaluate_perspective_conflict_resolution(text, predicted_emotion)
            flexibility_scores.append(conflict_resolution)
            
            # 7. 통합 관점 취하기 점수 계산
            basic_perspective_score = np.mean(basic_perspective_scores)
            multilayer_perspective_score = np.mean(multilayer_perspective_scores)
            situational_perspective_score = np.mean(situational_perspective_scores)
            cultural_perspective_score = np.mean(cultural_perspective_scores)
            accuracy_score = np.mean(accuracy_scores)
            flexibility_score = np.mean(flexibility_scores)
            
            # 8. 가중 통합 관점 취하기 점수
            integrated_perspective_score = (
                basic_perspective_score * 0.25 +
                multilayer_perspective_score * 0.20 +
                situational_perspective_score * 0.20 +
                cultural_perspective_score * 0.15 +
                accuracy_score * 0.10 +
                flexibility_score * 0.10
            )
            
            # 9. 관점 취하기 신뢰도 계산
            perspective_confidence = await self._calculate_perspective_taking_confidence(
                basic_perspective_scores, multilayer_perspective_scores, 
                situational_perspective_scores, cultural_perspective_scores
            )
            
            # 10. 최종 관점 취하기 점수 (신뢰도 조정)
            final_perspective_score = integrated_perspective_score * perspective_confidence
            
            # 11. 관점 취하기 메타인지 평가
            metacognitive_perspective = await self._evaluate_metacognitive_perspective_taking(final_perspective_score)
            
            # 12. 최종 점수 조정
            adjusted_perspective_score = final_perspective_score * (1 + metacognitive_perspective * 0.05)
            
            return min(1.0, max(0.0, adjusted_perspective_score))
            
        except Exception as e:
            self.logger.error(f"관점 취하기 계산 실패: {e}")
            return 0.0
    
    def _calculate_compassion_level(self, predicted_emotion: Dict) -> float:
        """
        연민 수준 계산 - 고급 연민 평가 모델
        
        2024년 연구 기반 구현:
        - 공감적 관심 vs 개인적 괴로움 분리
        - 불교 철학 기반 연민 모델
        - 신경과학적 연민 메커니즘
        - 문화적 연민 표현 차이
        """
        try:
            # 1. 기본 연민 유발 요인 분석
            compassion_triggers = []
            
            # 1.1 고통 인식 (Suffering Recognition)
            suffering_recognition = self._evaluate_suffering_recognition(predicted_emotion)
            compassion_triggers.append(suffering_recognition)
            
            # 1.2 취약성 인식 (Vulnerability Recognition)
            vulnerability_recognition = self._evaluate_vulnerability_recognition(predicted_emotion)
            compassion_triggers.append(vulnerability_recognition)
            
            # 1.3 불공정성 인식 (Injustice Recognition)
            injustice_recognition = self._evaluate_injustice_recognition(predicted_emotion)
            compassion_triggers.append(injustice_recognition)
            
            # 1.4 도움 필요성 인식 (Need for Help Recognition)
            help_need_recognition = self._evaluate_help_need_recognition(predicted_emotion)
            compassion_triggers.append(help_need_recognition)
            
            # 2. 연민 유형별 계산
            compassion_types = []
            
            # 2.1 공감적 관심 (Empathic Concern)
            empathic_concern = self._calculate_empathic_concern(predicted_emotion)
            compassion_types.append(empathic_concern)
            
            # 2.2 사랑-친절 (Loving-Kindness)
            loving_kindness = self._calculate_loving_kindness(predicted_emotion)
            compassion_types.append(loving_kindness)
            
            # 2.3 인지적 연민 (Cognitive Compassion)
            cognitive_compassion = self._calculate_cognitive_compassion(predicted_emotion)
            compassion_types.append(cognitive_compassion)
            
            # 2.4 정서적 연민 (Emotional Compassion)
            emotional_compassion = self._calculate_emotional_compassion(predicted_emotion)
            compassion_types.append(emotional_compassion)
            
            # 3. 연민 조절 요인들
            compassion_modulators = []
            
            # 3.1 개인적 괴로움 제거 (Personal Distress Reduction)
            personal_distress_reduction = self._reduce_personal_distress_bias(predicted_emotion)
            compassion_modulators.append(personal_distress_reduction)
            
            # 3.2 공감 피로 조절 (Empathy Fatigue Regulation)
            empathy_fatigue_regulation = self._regulate_empathy_fatigue(predicted_emotion)
            compassion_modulators.append(empathy_fatigue_regulation)
            
            # 3.3 도덕적 감정 조절 (Moral Emotion Regulation)
            moral_emotion_regulation = self._regulate_moral_emotions(predicted_emotion)
            compassion_modulators.append(moral_emotion_regulation)
            
            # 3.4 상황적 적절성 조절 (Situational Appropriateness)
            situational_appropriateness = self._adjust_situational_appropriateness(predicted_emotion)
            compassion_modulators.append(situational_appropriateness)
            
            # 4. 불교 철학 기반 연민 모델
            buddhist_compassion_factors = []
            
            # 4.1 무조건적 연민 (Unconditional Compassion)
            unconditional_compassion = self._calculate_unconditional_compassion(predicted_emotion)
            buddhist_compassion_factors.append(unconditional_compassion)
            
            # 4.2 공평한 연민 (Equanimous Compassion)
            equanimous_compassion = self._calculate_equanimous_compassion(predicted_emotion)
            buddhist_compassion_factors.append(equanimous_compassion)
            
            # 4.3 지혜로운 연민 (Wise Compassion)
            wise_compassion = self._calculate_wise_compassion(predicted_emotion)
            buddhist_compassion_factors.append(wise_compassion)
            
            # 4.4 행동 지향 연민 (Action-Oriented Compassion)
            action_oriented_compassion = self._calculate_action_oriented_compassion(predicted_emotion)
            buddhist_compassion_factors.append(action_oriented_compassion)
            
            # 5. 신경과학적 연민 메커니즘
            neurological_compassion_factors = []
            
            # 5.1 전대상피질 활성화 (Anterior Cingulate Cortex Activation)
            acc_activation = self._model_acc_activation(predicted_emotion)
            neurological_compassion_factors.append(acc_activation)
            
            # 5.2 전전두엽 조절 (Prefrontal Cortex Regulation)
            pfc_regulation = self._model_pfc_regulation(predicted_emotion)
            neurological_compassion_factors.append(pfc_regulation)
            
            # 5.3 측두엽 공감 네트워크 (Temporal Empathy Network)
            temporal_empathy = self._model_temporal_empathy_network(predicted_emotion)
            neurological_compassion_factors.append(temporal_empathy)
            
            # 5.4 옥시토신 시스템 (Oxytocin System)
            oxytocin_system = self._model_oxytocin_system(predicted_emotion)
            neurological_compassion_factors.append(oxytocin_system)
            
            # 6. 문화적 연민 표현 조정
            cultural_compassion_factors = []
            
            # 6.1 집단주의 vs 개인주의 연민 (Collectivist vs Individualist Compassion)
            cultural_orientation = self._adjust_cultural_orientation_compassion(predicted_emotion)
            cultural_compassion_factors.append(cultural_orientation)
            
            # 6.2 종교적 연민 전통 (Religious Compassion Traditions)
            religious_compassion = self._adjust_religious_compassion_traditions(predicted_emotion)
            cultural_compassion_factors.append(religious_compassion)
            
            # 6.3 언어적 연민 표현 (Linguistic Compassion Expression)
            linguistic_compassion = self._adjust_linguistic_compassion_expression(predicted_emotion)
            cultural_compassion_factors.append(linguistic_compassion)
            
            # 7. 통합 연민 점수 계산
            trigger_score = np.mean(compassion_triggers)
            type_score = np.mean(compassion_types)
            modulator_score = np.mean(compassion_modulators)
            buddhist_score = np.mean(buddhist_compassion_factors)
            neurological_score = np.mean(neurological_compassion_factors)
            cultural_score = np.mean(cultural_compassion_factors)
            
            # 8. 가중 통합 연민 점수
            integrated_compassion = (
                trigger_score * 0.25 +
                type_score * 0.20 +
                modulator_score * 0.15 +
                buddhist_score * 0.15 +
                neurological_score * 0.15 +
                cultural_score * 0.10
            )
            
            # 9. 연민 강도 조절
            compassion_intensity = self._adjust_compassion_intensity(integrated_compassion, predicted_emotion)
            
            # 10. 연민 지속성 평가
            compassion_durability = self._evaluate_compassion_durability(integrated_compassion, predicted_emotion)
            
            # 11. 최종 연민 점수
            final_compassion = integrated_compassion * compassion_intensity * compassion_durability
            
            return min(1.0, max(0.0, final_compassion))
            
        except Exception as e:
            self.logger.error(f"연민 수준 계산 실패: {e}")
            return 0.0
    
    # 추가 헬퍼 메서드들...
    async def _calculate_overall_empathy_score(self, *results) -> float:
        """전체 공감 점수 계산"""
        valid_results = [r for r in results if r is not None]
        if not valid_results:
            return 0.0
        
        # 각 결과의 신뢰도 기반 가중 평균
        scores = []
        for result in valid_results:
            if hasattr(result, 'self_confidence'):
                scores.append(result.self_confidence)
            elif hasattr(result, 'simulation_confidence'):
                scores.append(result.simulation_confidence)
            elif hasattr(result, 'social_cohesion_score'):
                scores.append(result.social_cohesion_score)
            elif hasattr(result, 'mirror_fidelity'):
                scores.append(result.mirror_fidelity)
        
        return np.mean(scores) if scores else 0.5
    
    def _extract_self_awareness_score(self, result) -> float:
        """자기 인식 점수 추출"""
        if result and hasattr(result, 'self_confidence'):
            return result.self_confidence
        return 0.0
    
    def _extract_other_understanding_score(self, result) -> float:
        """타인 이해 점수 추출"""
        if result and hasattr(result, 'theory_of_mind_score'):
            return result.theory_of_mind_score
        return 0.0
    
    def _extract_community_integration_score(self, result) -> float:
        """공동체 통합 점수 추출"""
        if result and hasattr(result, 'social_cohesion_score'):
            return result.social_cohesion_score
        return 0.0
    
    async def _calculate_self_other_balance(self, self_score: float, other_score: float) -> float:
        """자기-타인 균형 계산"""
        if self_score + other_score == 0:
            return 0.0
        return (other_score - self_score) / (self_score + other_score)
    
    async def _calculate_individual_community_balance(self, individual_score: float, community_score: float) -> float:
        """개인-공동체 균형 계산"""
        if individual_score + community_score == 0:
            return 0.0
        return (community_score - individual_score) / (individual_score + community_score)
    
    async def _calculate_utilitarian_compatibility(self, empathy_score: float, 
                                                 self_other_balance: float,
                                                 individual_community_balance: float) -> float:
        """벤담 호환성 점수 계산"""
        # 공감 점수와 균형 점수를 종합하여 공리주의적 호환성 계산
        balance_factor = (abs(self_other_balance) + abs(individual_community_balance)) / 2
        return empathy_score * (1 - balance_factor * 0.2)
    
    async def _calculate_integrated_confidence(self, *results) -> float:
        """통합 신뢰도 계산"""
        confidences = []
        for result in results:
            if result is None:
                continue
            if hasattr(result, 'self_confidence'):
                confidences.append(result.self_confidence)
            elif hasattr(result, 'simulation_confidence'):
                confidences.append(result.simulation_confidence)
        
        return np.mean(confidences) if confidences else 0.5


class MockModel:
    """모델 로딩을 위한 Mock 클래스"""
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.device = 'cpu'
    
    def to(self, device):
        self.device = device
        return self


# 컴포넌트 클래스들 (스텁 구현)
class SelfReflectionProcessor:
    """자기 성찰 처리기"""
    def __init__(self):
        pass

class EmpathySimulationProcessor:
    """공감 시뮬레이션 처리기"""
    def __init__(self):
        pass

class CommunityContextProcessor:
    """공동체 맥락 처리기"""
    def __init__(self):
        pass

class MirrorNeuronSystem:
    """
    Brain-Inspired Auto-Encoder Spiking Neural Network (AE-SNN) 기반 Mirror Neuron 시스템
    
    2024년 연구 기반 구현:
    - Brain-Inspired AE-SNN 아키텍처
    - Action-observation 페어링 메커니즘
    - Free Energy Principle 기반 예측
    - 자타 구분 능력 (Self-Other Differentiation)
    - Temporal Spike Pattern Analysis
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # AE-SNN 네트워크 파라미터
        self.spike_threshold = 0.5
        self.membrane_potential_decay = 0.95
        self.synaptic_weight_decay = 0.98
        self.spike_frequency_adaptation = 0.01
        
        # Mirror Neuron 특화 파라미터
        self.action_observation_weight = 0.7
        self.self_other_boundary = 0.3
        self.temporal_window = 100  # ms
        
        # 신경 활성화 패턴 저장
        self.neural_state = {
            'membrane_potentials': {},
            'spike_trains': {},
            'synaptic_weights': {},
            'action_patterns': {},
            'observation_patterns': {}
        }
        
        # 학습된 행동-관찰 페어링
        self.action_observation_pairs = {}
        
        # 고도화된 패턴 저장 시스템
        self.pattern_database = {
            'self_patterns': {},      # 자기 행동 패턴 저장
            'other_patterns': {},     # 타인 행동 패턴 저장
            'context_patterns': {},   # 상황별 패턴
            'temporal_patterns': {},  # 시간적 패턴
            'pattern_weights': {},    # 패턴별 가중치
            'pattern_frequencies': {} # 패턴 출현 빈도
        }
        
        # 자타 구분 신경망 시스템
        self.self_other_classifier = {
            'neural_network': None,
            'feature_extractor': None,
            'discriminator': None,
            'training_data': [],
            'validation_data': [],
            'performance_history': [],
            'last_update': 0
        }
        
        # 고도화된 패턴 학습 시스템
        self.pattern_learning_system = {
            'incremental_learner': None,
            'pattern_clustering': None,
            'novelty_detector': None,
            'pattern_predictor': None,
            'meta_classifier': None
        }
        
        # 저장된 패턴 관리 시스템
        self.pattern_management = {
            'pattern_hierarchy': {},      # 계층적 패턴 구조
            'pattern_relationships': {},  # 패턴 간 관계
            'pattern_evolution': {},      # 패턴 진화 기록
            'pattern_consolidation': {},  # 패턴 통합 기록
            'pattern_forgetting': {}      # 패턴 망각 기록
        }
        
        # 학습 메커니즘 파라미터
        self.learning_params = {
            'pattern_decay_rate': 0.95,      # 패턴 기억 감소율
            'novelty_threshold': 0.7,        # 새로운 패턴 인식 임계값
            'update_learning_rate': 0.01,    # 가중치 업데이트 학습률
            'meta_learning_rate': 0.001,     # 메타학습 학습률
            'experience_buffer_size': 1000,  # 경험 버퍼 크기
            'minimum_pattern_count': 5       # 최소 패턴 등록 횟수
        }
        
        # 베이지안 추론 파라미터
        self.bayesian_params = {
            'prior_alpha': 1.0,          # 디리클레 분포 prior
            'prior_beta': 1.0,           # 베타 분포 prior
            'prediction_horizon': 10,    # 예측 시간 범위
            'uncertainty_threshold': 0.3  # 불확실성 임계값
        }
        
        # 시계열 분석 파라미터
        self.temporal_params = {
            'window_size': 50,           # 시계열 윈도우 크기
            'lag_max': 20,               # 최대 지연 차수
            'trend_threshold': 0.1,      # 추세 감지 임계값
            'seasonality_period': 10,    # 계절성 주기
            'noise_variance': 0.01       # 노이즈 분산
        }
        
        # 성능 메트릭
        self.performance_metrics = {
            'mirror_accuracy': [],
            'self_other_classification': [],
            'free_energy_predictions': [],
            'temporal_consistency': [],
            'pattern_recognition_accuracy': [],
            'learning_convergence': [],
            'prediction_error_history': []
        }
        
        # 자타 구분 시스템 초기화
        self._initialize_self_other_classification_system()
        
        # 패턴 학습 시스템 초기화
        self._initialize_pattern_learning_system()
        
        # 패턴 관리 시스템 초기화
        self._initialize_pattern_management_system()
        
        self.logger.info("Brain-Inspired AE-SNN Mirror Neuron System 초기화 완료")
    
    async def process_mirror_response(self, 
                                    observed_action: Dict[str, Any],
                                    context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        관찰된 행동에 대한 거울 반응 처리
        
        Args:
            observed_action: 관찰된 행동 정보
            context: 상황 컨텍스트
            
        Returns:
            거울 반응 결과
        """
        context = context or {}
        
        try:
            # 1. 행동 인코딩 (AE-SNN 인코더)
            encoded_action = await self._encode_action_pattern(observed_action)
            
            # 2. 거울 뉴런 활성화
            mirror_activation = await self._activate_mirror_neurons(encoded_action, context)
            
            # 3. 자타 구분 처리
            self_other_signal = await self._process_self_other_differentiation(
                mirror_activation, context
            )
            
            # 4. 거울 반응 생성
            mirrored_response = await self._generate_mirrored_response(
                mirror_activation, self_other_signal
            )
            
            # 5. Free Energy 예측 계산
            free_energy = await self._calculate_free_energy(
                observed_action, mirrored_response
            )
            
            # 6. 시간적 일관성 검증
            temporal_consistency = await self._verify_temporal_consistency(
                encoded_action, mirrored_response
            )
            
            return {
                'mirrored_response': mirrored_response,
                'neural_activation': mirror_activation,
                'self_other_differentiation': self_other_signal,
                'free_energy_prediction': free_energy,
                'temporal_consistency': temporal_consistency,
                'confidence': self._calculate_mirror_confidence(
                    mirror_activation, free_energy
                )
            }
            
        except Exception as e:
            self.logger.error(f"거울 반응 처리 실패: {e}")
            return {
                'mirrored_response': {},
                'neural_activation': {},
                'self_other_differentiation': 0.0,
                'free_energy_prediction': 0.0,
                'temporal_consistency': 0.0,
                'confidence': 0.0,
                'error': str(e)
            }
    
    async def _encode_action_pattern(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        AE-SNN 인코더를 통한 행동 패턴 인코딩
        
        Args:
            action: 행동 정보
            
        Returns:
            인코딩된 행동 패턴
        """
        try:
            # 행동을 스파이크 패턴으로 변환
            spike_pattern = {}
            
            # 감정적 차원 인코딩
            if 'emotion' in action:
                emotion = action['emotion']
                spike_pattern['emotion_spikes'] = {
                    'valence': self._encode_to_spikes(emotion.get('valence', 0.0)),
                    'arousal': self._encode_to_spikes(emotion.get('arousal', 0.0)),
                    'dominance': self._encode_to_spikes(emotion.get('dominance', 0.0))
                }
            
            # 행동 강도 인코딩
            if 'intensity' in action:
                spike_pattern['intensity_spikes'] = self._encode_to_spikes(
                    action['intensity']
                )
            
            # 시간적 패턴 인코딩
            if 'temporal_pattern' in action:
                spike_pattern['temporal_spikes'] = self._encode_temporal_pattern(
                    action['temporal_pattern']
                )
            
            # 공간적 패턴 인코딩 (신체 부위별)
            if 'spatial_pattern' in action:
                spike_pattern['spatial_spikes'] = self._encode_spatial_pattern(
                    action['spatial_pattern']
                )
            
            return spike_pattern
            
        except Exception as e:
            self.logger.error(f"행동 패턴 인코딩 실패: {e}")
            return {}
    
    async def _activate_mirror_neurons(self, 
                                     encoded_action: Dict[str, Any],
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """
        거울 뉴런 활성화 처리
        
        Args:
            encoded_action: 인코딩된 행동 패턴
            context: 상황 컨텍스트
            
        Returns:
            거울 뉴런 활성화 패턴
        """
        try:
            activation_pattern = {}
            
            # 각 뉴런 그룹별 활성화 계산
            for neuron_group, spike_data in encoded_action.items():
                # 막전위 업데이트
                membrane_potential = self._update_membrane_potential(
                    neuron_group, spike_data
                )
                
                # 스파이크 생성 여부 결정
                spike_generated = membrane_potential > self.spike_threshold
                
                # 거울 뉴런 특화 가중치 적용
                mirror_weight = self._calculate_mirror_weight(
                    neuron_group, context
                )
                
                activation_pattern[neuron_group] = {
                    'membrane_potential': membrane_potential,
                    'spike_generated': spike_generated,
                    'mirror_weight': mirror_weight,
                    'activation_strength': membrane_potential * mirror_weight
                }
            
            # 네트워크 전체 활성화 계산
            network_activation = self._calculate_network_activation(activation_pattern)
            
            # 학습 기반 활성화 조정
            adjusted_activation = await self._adjust_activation_by_learning(
                activation_pattern, context
            )
            
            return {
                'neuron_activations': adjusted_activation,
                'network_activation': network_activation,
                'spike_count': sum(1 for group in adjusted_activation.values() 
                                if group.get('spike_generated', False))
            }
            
        except Exception as e:
            self.logger.error(f"거울 뉴런 활성화 실패: {e}")
            return {}
    
    async def _process_self_other_differentiation(self, 
                                                mirror_activation: Dict[str, Any],
                                                context: Dict[str, Any]) -> float:
        """
        자타 구분 처리
        
        Args:
            mirror_activation: 거울 뉴런 활성화 패턴
            context: 상황 컨텍스트
            
        Returns:
            자타 구분 신호 (0.0: 자기, 1.0: 타인)
        """
        try:
            # 자기 행동 패턴과의 유사도 계산
            self_similarity = self._calculate_self_similarity(mirror_activation)
            
            # 타인 행동 패턴과의 유사도 계산
            other_similarity = self._calculate_other_similarity(mirror_activation)
            
            # 상황적 맥락 고려
            context_bias = self._calculate_context_bias(context)
            
            # 자타 구분 신호 계산
            self_other_signal = (other_similarity - self_similarity + context_bias) / 2.0
            
            # 0.0 ~ 1.0 범위로 정규화
            self_other_signal = max(0.0, min(1.0, self_other_signal))
            
            return self_other_signal
            
        except Exception as e:
            self.logger.error(f"자타 구분 처리 실패: {e}")
            return 0.5  # 불확실한 경우 중간값
    
    async def _generate_mirrored_response(self, 
                                        mirror_activation: Dict[str, Any],
                                        self_other_signal: float) -> Dict[str, Any]:
        """
        거울 반응 생성
        
        Args:
            mirror_activation: 거울 뉴런 활성화 패턴
            self_other_signal: 자타 구분 신호
            
        Returns:
            생성된 거울 반응
        """
        try:
            # 활성화 패턴을 행동으로 디코딩
            decoded_action = await self._decode_activation_to_action(mirror_activation)
            
            # 자타 구분에 따른 반응 강도 조정
            if self_other_signal > 0.7:  # 타인의 행동
                response_intensity = 0.8
                empathy_component = 0.9
            elif self_other_signal < 0.3:  # 자기의 행동
                response_intensity = 0.5
                empathy_component = 0.3
            else:  # 불확실한 경우
                response_intensity = 0.6
                empathy_component = 0.6
            
            # 거울 반응 구성
            mirrored_response = {
                'action_type': decoded_action.get('action_type', 'unknown'),
                'intensity': response_intensity,
                'empathy_component': empathy_component,
                'motor_resonance': self._calculate_motor_resonance(mirror_activation),
                'emotional_contagion': self._calculate_emotional_contagion(
                    decoded_action, self_other_signal
                ),
                'prediction_component': self._generate_action_prediction(decoded_action)
            }
            
            return mirrored_response
            
        except Exception as e:
            self.logger.error(f"거울 반응 생성 실패: {e}")
            return {}
    
    async def _calculate_free_energy(self, 
                                   observed_action: Dict[str, Any],
                                   mirrored_response: Dict[str, Any]) -> float:
        """
        Free Energy Principle 기반 예측 에너지 계산
        
        Args:
            observed_action: 관찰된 행동
            mirrored_response: 거울 반응
            
        Returns:
            Free Energy 값
        """
        try:
            # 예측 오차 계산
            prediction_error = self._calculate_prediction_error(
                observed_action, mirrored_response
            )
            
            # 복잡도 계산
            complexity = self._calculate_model_complexity(mirrored_response)
            
            # Free Energy = 예측 오차 + 복잡도
            free_energy = prediction_error + complexity
            
            return free_energy
            
        except Exception as e:
            self.logger.error(f"Free Energy 계산 실패: {e}")
            return 0.0
    
    def _encode_to_spikes(self, value: float) -> List[float]:
        """값을 스파이크 패턴으로 변환"""
        # 값의 크기에 따라 스파이크 빈도 결정
        spike_frequency = abs(value) * 10  # 0-10 Hz
        spikes = []
        
        for i in range(self.temporal_window):
            # 포아송 분포 기반 스파이크 생성
            spike_probability = spike_frequency / 1000.0  # 1ms 단위
            if np.random.random() < spike_probability:
                spikes.append(1.0)
            else:
                spikes.append(0.0)
        
        return spikes
    
    def _encode_temporal_pattern(self, pattern: List[float]) -> List[float]:
        """시간적 패턴을 스파이크 열로 변환"""
        encoded = []
        for value in pattern:
            encoded.extend(self._encode_to_spikes(value))
        return encoded
    
    def _encode_spatial_pattern(self, pattern: Dict[str, float]) -> Dict[str, List[float]]:
        """공간적 패턴을 스파이크 열로 변환"""
        encoded = {}
        for location, value in pattern.items():
            encoded[location] = self._encode_to_spikes(value)
        return encoded
    
    def _update_membrane_potential(self, neuron_group: str, spike_data: Any) -> float:
        """막전위 업데이트"""
        if neuron_group not in self.neural_state['membrane_potentials']:
            self.neural_state['membrane_potentials'][neuron_group] = 0.0
        
        current_potential = self.neural_state['membrane_potentials'][neuron_group]
        
        # 입력 전류 계산
        input_current = self._calculate_input_current(spike_data)
        
        # 막전위 업데이트 (LIF 모델)
        new_potential = (current_potential * self.membrane_potential_decay + 
                        input_current * (1 - self.membrane_potential_decay))
        
        self.neural_state['membrane_potentials'][neuron_group] = new_potential
        
        return new_potential
    
    def _calculate_input_current(self, spike_data: Any) -> float:
        """입력 전류 계산"""
        if isinstance(spike_data, dict):
            return sum(sum(spikes) if isinstance(spikes, list) else spikes 
                      for spikes in spike_data.values()) / len(spike_data)
        elif isinstance(spike_data, list):
            return sum(spike_data) / len(spike_data)
        else:
            return float(spike_data)
    
    def _calculate_mirror_weight(self, neuron_group: str, context: Dict[str, Any]) -> float:
        """거울 뉴런 가중치 계산"""
        base_weight = self.action_observation_weight
        
        # 상황적 조정
        if context.get('social_context'):
            base_weight *= 1.2
        
        if context.get('emotional_salience'):
            base_weight *= 1.1
        
        return min(1.0, base_weight)
    
    def _calculate_network_activation(self, activation_pattern: Dict[str, Any]) -> float:
        """네트워크 전체 활성화 계산"""
        total_activation = 0.0
        active_neurons = 0
        
        for neuron_data in activation_pattern.values():
            if neuron_data.get('spike_generated', False):
                total_activation += neuron_data.get('activation_strength', 0.0)
                active_neurons += 1
        
        if active_neurons > 0:
            return total_activation / active_neurons
        return 0.0
    
    async def _adjust_activation_by_learning(self, 
                                           activation_pattern: Dict[str, Any],
                                           context: Dict[str, Any]) -> Dict[str, Any]:
        """
        학습 기반 활성화 조정
        
        고도화된 학습 메커니즘:
        - 경험 기반 가중치 업데이트
        - 메타학습 기반 적응
        - 강화학습 기반 패턴 선택
        - 노벨티 검출 및 처리
        - 경험 버퍼 관리
        - 온라인 학습 및 적응
        
        Args:
            activation_pattern: 원본 활성화 패턴
            context: 상황 컨텍스트
            
        Returns:
            학습 기반 조정된 활성화 패턴
        """
        try:
            # 원본 패턴 보존
            adjusted_pattern = dict(activation_pattern)
            
            # 1. 노벨티 검출
            novelty_score = self._detect_novelty(activation_pattern, context)
            
            # 2. 경험 기반 가중치 업데이트
            experience_weights = self._calculate_experience_weights(activation_pattern, context)
            
            # 3. 메타학습 기반 적응
            meta_adjustments = self._apply_meta_learning(activation_pattern, context)
            
            # 4. 강화학습 기반 패턴 선택
            rl_adjustments = self._apply_reinforcement_learning(activation_pattern, context)
            
            # 5. 상황별 학습 조정
            contextual_adjustments = self._apply_contextual_learning(activation_pattern, context)
            
            # 6. 뉴런 활성화 조정 적용
            if 'neuron_activations' in adjusted_pattern:
                adjusted_activations = {}
                
                for neuron_group, neuron_data in adjusted_pattern['neuron_activations'].items():
                    # 경험 가중치 적용
                    experience_weight = experience_weights.get(neuron_group, 1.0)
                    
                    # 메타학습 조정
                    meta_adjustment = meta_adjustments.get(neuron_group, 0.0)
                    
                    # 강화학습 조정
                    rl_adjustment = rl_adjustments.get(neuron_group, 0.0)
                    
                    # 상황별 조정
                    contextual_adjustment = contextual_adjustments.get(neuron_group, 0.0)
                    
                    # 노벨티 기반 조정
                    novelty_adjustment = novelty_score * self.learning_params['novelty_threshold']
                    
                    # 조정된 활성화 값 계산
                    original_strength = neuron_data.get('activation_strength', 0.0)
                    adjusted_strength = original_strength * experience_weight
                    adjusted_strength += meta_adjustment + rl_adjustment + contextual_adjustment
                    adjusted_strength *= (1.0 + novelty_adjustment)
                    
                    # 원본 뉴런 데이터 복사 및 조정
                    adjusted_neuron_data = dict(neuron_data)
                    adjusted_neuron_data['activation_strength'] = max(0.0, min(1.0, adjusted_strength))
                    adjusted_neuron_data['learning_adjusted'] = True
                    adjusted_neuron_data['experience_weight'] = experience_weight
                    adjusted_neuron_data['meta_adjustment'] = meta_adjustment
                    adjusted_neuron_data['rl_adjustment'] = rl_adjustment
                    adjusted_neuron_data['novelty_score'] = novelty_score
                    
                    adjusted_activations[neuron_group] = adjusted_neuron_data
                
                adjusted_pattern['neuron_activations'] = adjusted_activations
            
            # 7. 네트워크 전체 활성화 재계산
            adjusted_pattern['network_activation'] = self._calculate_network_activation(
                {'neuron_activations': adjusted_pattern.get('neuron_activations', {})}
            )
            
            # 8. 학습 경험 저장
            self._store_learning_experience(activation_pattern, adjusted_pattern, context)
            
            # 9. 학습 메트릭 업데이트
            self._update_learning_metrics(novelty_score, experience_weights, meta_adjustments)
            
            return adjusted_pattern
            
        except Exception as e:
            self.logger.error(f"학습 기반 활성화 조정 실패: {e}")
            return activation_pattern
    
    def _calculate_self_similarity(self, activation: Dict[str, Any]) -> float:
        """
        자기 행동 패턴과의 유사도 계산
        
        고도화된 패턴 매칭 시스템:
        - 다차원 벡터 유사도 계산 (코사인 유사도, 유클리드 거리)
        - 시간적 감쇠 고려
        - 가중치 기반 패턴 매칭
        - 베이지안 추론 기반 불확실성 계산
        """
        try:
            # 현재 활성화 패턴을 벡터로 변환
            current_vector = self._activation_to_vector(activation)
            
            if len(current_vector) == 0:
                return 0.0
            
            # 자기 패턴 데이터베이스에서 유사도 계산
            self_patterns = self.pattern_database.get('self_patterns', {})
            
            if not self_patterns:
                # 패턴이 없는 경우 낮은 유사도 반환
                return 0.2
            
            similarities = []
            weights = []
            
            for pattern_id, pattern_data in self_patterns.items():
                # 저장된 패턴 벡터
                pattern_vector = pattern_data['vector']
                pattern_weight = pattern_data.get('weight', 1.0)
                pattern_frequency = pattern_data.get('frequency', 1)
                creation_time = pattern_data.get('timestamp', 0)
                
                # 시간적 감쇠 계산
                time_decay = self._calculate_temporal_decay(creation_time)
                
                # 다양한 유사도 측정 방법 적용
                cosine_sim = self._cosine_similarity(current_vector, pattern_vector)
                euclidean_sim = self._euclidean_similarity(current_vector, pattern_vector)
                manhattan_sim = self._manhattan_similarity(current_vector, pattern_vector)
                
                # 가중 평균 유사도
                weighted_similarity = (
                    0.5 * cosine_sim + 
                    0.3 * euclidean_sim + 
                    0.2 * manhattan_sim
                )
                
                # 패턴 가중치 및 빈도 고려
                final_similarity = (
                    weighted_similarity * 
                    pattern_weight * 
                    time_decay * 
                    min(1.0, pattern_frequency / 10.0)
                )
                
                similarities.append(final_similarity)
                weights.append(pattern_weight * time_decay)
            
            # 가중 평균 계산
            if sum(weights) > 0:
                weighted_avg_similarity = sum(s * w for s, w in zip(similarities, weights)) / sum(weights)
            else:
                weighted_avg_similarity = 0.0
            
            # 베이지안 추론 기반 불확실성 고려
            uncertainty = self._calculate_pattern_uncertainty(similarities)
            confidence = 1.0 - uncertainty
            
            # 최종 유사도 = 가중 평균 * 신뢰도
            final_similarity = weighted_avg_similarity * confidence
            
            return max(0.0, min(1.0, final_similarity))
            
        except Exception as e:
            self.logger.error(f"자기 패턴 유사도 계산 실패: {e}")
            return 0.2
    
    def _calculate_other_similarity(self, activation: Dict[str, Any]) -> float:
        """
        타인 행동 패턴과의 유사도 계산
        
        고도화된 패턴 매칭 시스템:
        - 다차원 벡터 유사도 계산
        - 상황별 패턴 구분
        - 사회적 맥락 고려
        - 감정 전염 패턴 인식
        """
        try:
            # 현재 활성화 패턴을 벡터로 변환
            current_vector = self._activation_to_vector(activation)
            
            if len(current_vector) == 0:
                return 0.0
            
            # 타인 패턴 데이터베이스에서 유사도 계산
            other_patterns = self.pattern_database.get('other_patterns', {})
            
            if not other_patterns:
                # 패턴이 없는 경우 중간 유사도 반환
                return 0.6
            
            similarities = []
            weights = []
            social_context_boost = 0.0
            
            for pattern_id, pattern_data in other_patterns.items():
                # 저장된 패턴 정보
                pattern_vector = pattern_data['vector']
                pattern_weight = pattern_data.get('weight', 1.0)
                pattern_frequency = pattern_data.get('frequency', 1)
                creation_time = pattern_data.get('timestamp', 0)
                social_context = pattern_data.get('social_context', {})
                
                # 시간적 감쇠
                time_decay = self._calculate_temporal_decay(creation_time)
                
                # 기본 유사도 계산
                cosine_sim = self._cosine_similarity(current_vector, pattern_vector)
                euclidean_sim = self._euclidean_similarity(current_vector, pattern_vector)
                manhattan_sim = self._manhattan_similarity(current_vector, pattern_vector)
                
                # 사회적 맥락 고려
                social_weight = self._calculate_social_context_weight(social_context)
                
                # 감정 전염 패턴 인식
                contagion_factor = self._calculate_emotional_contagion_factor(pattern_data)
                
                # 종합 유사도
                weighted_similarity = (
                    0.4 * cosine_sim + 
                    0.3 * euclidean_sim + 
                    0.2 * manhattan_sim +
                    0.1 * contagion_factor
                )
                
                # 최종 유사도 (사회적 맥락 및 빈도 고려)
                final_similarity = (
                    weighted_similarity * 
                    pattern_weight * 
                    time_decay * 
                    social_weight * 
                    min(1.0, pattern_frequency / 5.0)
                )
                
                similarities.append(final_similarity)
                weights.append(pattern_weight * time_decay * social_weight)
                
                # 사회적 맥락 부스트 누적
                social_context_boost += social_weight * 0.1
            
            # 가중 평균 계산
            if sum(weights) > 0:
                weighted_avg_similarity = sum(s * w for s, w in zip(similarities, weights)) / sum(weights)
            else:
                weighted_avg_similarity = 0.6
            
            # 사회적 맥락 부스트 적용
            social_context_boost = min(0.3, social_context_boost)
            
            # 베이지안 추론 기반 불확실성
            uncertainty = self._calculate_pattern_uncertainty(similarities)
            confidence = 1.0 - uncertainty
            
            # 최종 유사도
            final_similarity = (weighted_avg_similarity + social_context_boost) * confidence
            
            return max(0.0, min(1.0, final_similarity))
            
        except Exception as e:
            self.logger.error(f"타인 패턴 유사도 계산 실패: {e}")
            return 0.6
    
    def _calculate_context_bias(self, context: Dict[str, Any]) -> float:
        """상황적 편향 계산"""
        bias = 0.0
        
        if context.get('social_context') == 'interaction':
            bias += 0.1
        
        if context.get('emotional_context') == 'empathetic':
            bias += 0.1
        
        return bias
    
    async def _decode_activation_to_action(self, activation: Dict[str, Any]) -> Dict[str, Any]:
        """활성화 패턴을 행동으로 디코딩"""
        return {
            'action_type': 'mirrored_action',
            'intensity': activation.get('network_activation', 0.0),
            'confidence': min(1.0, activation.get('spike_count', 0) / 10.0)
        }
    
    def _calculate_motor_resonance(self, activation: Dict[str, Any]) -> float:
        """운동 공명 계산"""
        return activation.get('network_activation', 0.0) * 0.8
    
    def _calculate_emotional_contagion(self, action: Dict[str, Any], self_other: float) -> float:
        """감정 전염 계산"""
        base_contagion = action.get('intensity', 0.0)
        return base_contagion * self_other
    
    def _generate_action_prediction(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """행동 예측 생성"""
        return {
            'next_action': 'predicted_action',
            'confidence': action.get('confidence', 0.5),
            'time_horizon': 1.0
        }
    
    def _calculate_prediction_error(self, observed: Dict[str, Any], predicted: Dict[str, Any]) -> float:
        """
        베이지안 추론 기반 예측 오차 계산
        
        Free Energy Principle에 따른 고도화된 예측 오차:
        - 베이지안 추론 기반 확률 분포 비교
        - KL-divergence 계산
        - 다차원 특성 공간에서의 오차 측정
        - 불확실성 정량화
        
        Args:
            observed: 관찰된 행동 데이터
            predicted: 예측된 행동 데이터
            
        Returns:
            예측 오차 (0-1, 높을수록 오차가 큼)
        """
        try:
            # 1. 특성 벡터 추출
            observed_features = self._extract_behavioral_features(observed)
            predicted_features = self._extract_behavioral_features(predicted)
            
            if not observed_features or not predicted_features:
                return 1.0  # 최대 오차
            
            # 2. 확률 분포 추정
            observed_dist = self._estimate_probability_distribution(observed_features)
            predicted_dist = self._estimate_probability_distribution(predicted_features)
            
            # 3. KL-divergence 계산 (예측 분포에서 관찰 분포로)
            kl_divergence = self._calculate_kl_divergence(predicted_dist, observed_dist)
            
            # 4. 다차원 특성 공간에서의 거리 측정
            if len(observed_features) == len(predicted_features):
                # 유클리드 거리 기반 오차
                feature_error = np.sqrt(sum((o - p)**2 for o, p in zip(observed_features, predicted_features)))
                # 정규화 (최대 거리로 나누기)
                max_distance = np.sqrt(len(observed_features))
                normalized_feature_error = min(feature_error / max_distance, 1.0)
            else:
                normalized_feature_error = 1.0
            
            # 5. 시간적 일관성 오차
            temporal_error = self._calculate_temporal_prediction_error(observed, predicted)
            
            # 6. 감정적 차원 오차
            emotional_error = self._calculate_emotional_prediction_error(observed, predicted)
            
            # 7. 행동 강도 오차
            intensity_error = self._calculate_intensity_prediction_error(observed, predicted)
            
            # 8. 베이지안 불확실성 계산
            uncertainty = self._calculate_bayesian_uncertainty(observed_dist, predicted_dist)
            
            # 9. 가중 평균 예측 오차
            weighted_error = (
                0.3 * kl_divergence +
                0.2 * normalized_feature_error +
                0.2 * temporal_error +
                0.15 * emotional_error +
                0.1 * intensity_error +
                0.05 * uncertainty
            )
            
            return max(0.0, min(1.0, weighted_error))
            
        except Exception as e:
            self.logger.error(f"예측 오차 계산 실패: {e}")
            return 0.5  # 중간 오차값
    
    def _calculate_model_complexity(self, response: Dict[str, Any]) -> float:
        """
        모델 복잡도 계산 (Free Energy Principle 기반)
        
        고도화된 모델 복잡도:
        - 베이지안 모델 선택 기준
        - 정보 이론 기반 복잡도
        - 네트워크 활성화 패턴 복잡도
        - 계층적 구조 복잡도
        
        Args:
            response: 거울 반응 데이터
            
        Returns:
            모델 복잡도 (0-1)
        """
        try:
            # 1. 네트워크 활성화 복잡도
            network_complexity = self._calculate_network_activation_complexity(response)
            
            # 2. 정보 이론 기반 복잡도 (엔트로피)
            information_complexity = self._calculate_information_complexity(response)
            
            # 3. 계층적 구조 복잡도
            hierarchical_complexity = self._calculate_hierarchical_complexity(response)
            
            # 4. 시간적 의존성 복잡도
            temporal_complexity = self._calculate_temporal_dependency_complexity(response)
            
            # 5. 상호작용 복잡도
            interaction_complexity = self._calculate_interaction_complexity(response)
            
            # 6. 베이지안 모델 복잡도 (사전 분포 기반)
            bayesian_complexity = self._calculate_bayesian_model_complexity(response)
            
            # 7. 가중 평균 복잡도
            weighted_complexity = (
                0.25 * network_complexity +
                0.2 * information_complexity +
                0.2 * hierarchical_complexity +
                0.15 * temporal_complexity +
                0.1 * interaction_complexity +
                0.1 * bayesian_complexity
            )
            
            return max(0.0, min(1.0, weighted_complexity))
            
        except Exception as e:
            self.logger.error(f"모델 복잡도 계산 실패: {e}")
            return 0.3  # 중간 복잡도값
    
    async def _verify_temporal_consistency(self, 
                                         encoded_action: Dict[str, Any],
                                         mirrored_response: Dict[str, Any]) -> float:
        """
        시계열 분석 기반 시간적 일관성 검증
        
        고도화된 시간적 일관성 분석:
        - 자기상관 분석 (Autocorrelation)
        - 시간적 변화율 분석
        - 추세 분석 (Trend Analysis)
        - 계절성 분석 (Seasonality Analysis)
        - 노이즈 수준 분석
        - 장기 의존성 분석
        
        Args:
            encoded_action: 인코딩된 행동 패턴
            mirrored_response: 거울 반응 데이터
            
        Returns:
            시간적 일관성 점수 (0-1, 높을수록 일관성 높음)
        """
        try:
            # 1. 시계열 데이터 추출
            action_series = self._extract_temporal_series(encoded_action)
            response_series = self._extract_temporal_series(mirrored_response)
            
            if not action_series or not response_series:
                return 0.5  # 중간 일관성
            
            # 2. 자기상관 분석
            autocorr_consistency = self._calculate_autocorrelation_consistency(
                action_series, response_series
            )
            
            # 3. 시간적 변화율 분석
            change_rate_consistency = self._calculate_change_rate_consistency(
                action_series, response_series
            )
            
            # 4. 추세 분석
            trend_consistency = self._calculate_trend_consistency(
                action_series, response_series
            )
            
            # 5. 계절성 분석 (주기적 패턴)
            seasonality_consistency = self._calculate_seasonality_consistency(
                action_series, response_series
            )
            
            # 6. 노이즈 수준 분석
            noise_consistency = self._calculate_noise_consistency(
                action_series, response_series
            )
            
            # 7. 장기 의존성 분석 (Long-term dependencies)
            longterm_consistency = self._calculate_longterm_dependency_consistency(
                action_series, response_series
            )
            
            # 8. 시간적 지연 분석 (Temporal lag analysis)
            lag_consistency = self._calculate_temporal_lag_consistency(
                action_series, response_series
            )
            
            # 9. 가중 평균 일관성 계산
            weighted_consistency = (
                0.25 * autocorr_consistency +
                0.2 * change_rate_consistency +
                0.15 * trend_consistency +
                0.1 * seasonality_consistency +
                0.1 * noise_consistency +
                0.1 * longterm_consistency +
                0.1 * lag_consistency
            )
            
            # 10. 시간적 일관성 메트릭 업데이트
            self._update_temporal_consistency_metrics(weighted_consistency)
            
            return max(0.0, min(1.0, weighted_consistency))
            
        except Exception as e:
            self.logger.error(f"시간적 일관성 검증 실패: {e}")
            return 0.5
    
    def _calculate_mirror_confidence(self, activation: Dict[str, Any], free_energy: float) -> float:
        """거울 반응 신뢰도 계산"""
        activation_confidence = activation.get('network_activation', 0.0)
        energy_confidence = max(0.0, 1.0 - free_energy)
        
        return (activation_confidence + energy_confidence) / 2.0
    
    # =============================================================================
    # 고도화된 패턴 매칭 시스템 헬퍼 메서드들
    # =============================================================================
    
    def _activation_to_vector(self, activation: Dict[str, Any]) -> List[float]:
        """
        활성화 패턴을 고차원 벡터로 변환
        
        Args:
            activation: 신경 활성화 패턴
            
        Returns:
            정규화된 벡터 표현
        """
        try:
            vector = []
            
            # 뉴런 활성화 정보 추출
            neuron_activations = activation.get('neuron_activations', {})
            for neuron_group, neuron_data in neuron_activations.items():
                # 막전위
                membrane_potential = neuron_data.get('membrane_potential', 0.0)
                vector.append(membrane_potential)
                
                # 스파이크 생성 여부 (0 또는 1)
                spike_generated = float(neuron_data.get('spike_generated', False))
                vector.append(spike_generated)
                
                # 활성화 강도
                activation_strength = neuron_data.get('activation_strength', 0.0)
                vector.append(activation_strength)
                
                # 거울 가중치
                mirror_weight = neuron_data.get('mirror_weight', 0.0)
                vector.append(mirror_weight)
            
            # 네트워크 전체 활성화
            network_activation = activation.get('network_activation', 0.0)
            vector.append(network_activation)
            
            # 스파이크 개수
            spike_count = activation.get('spike_count', 0)
            vector.append(float(spike_count))
            
            # 벡터 정규화 (L2 norm)
            if len(vector) > 0:
                norm = np.sqrt(sum(x**2 for x in vector))
                if norm > 0:
                    vector = [x / norm for x in vector]
            
            return vector
            
        except Exception as e:
            self.logger.error(f"활성화 패턴 벡터 변환 실패: {e}")
            return []
    
    def _cosine_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """코사인 유사도 계산"""
        try:
            if len(vector1) != len(vector2) or len(vector1) == 0:
                return 0.0
            
            dot_product = sum(a * b for a, b in zip(vector1, vector2))
            norm1 = np.sqrt(sum(a**2 for a in vector1))
            norm2 = np.sqrt(sum(b**2 for b in vector2))
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except Exception:
            return 0.0
    
    def _euclidean_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """유클리드 거리 기반 유사도 계산"""
        try:
            if len(vector1) != len(vector2) or len(vector1) == 0:
                return 0.0
            
            # 유클리드 거리 계산
            euclidean_distance = np.sqrt(sum((a - b)**2 for a, b in zip(vector1, vector2)))
            
            # 거리를 유사도로 변환 (0-1 범위)
            # 최대 거리는 sqrt(2) (정규화된 벡터의 경우)
            max_distance = np.sqrt(2)
            similarity = 1.0 - min(euclidean_distance / max_distance, 1.0)
            
            return similarity
            
        except Exception:
            return 0.0
    
    def _manhattan_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """맨하탄 거리 기반 유사도 계산"""
        try:
            if len(vector1) != len(vector2) or len(vector1) == 0:
                return 0.0
            
            # 맨하탄 거리 계산
            manhattan_distance = sum(abs(a - b) for a, b in zip(vector1, vector2))
            
            # 거리를 유사도로 변환
            # 최대 거리는 2 (정규화된 벡터의 경우)
            max_distance = 2.0
            similarity = 1.0 - min(manhattan_distance / max_distance, 1.0)
            
            return similarity
            
        except Exception:
            return 0.0
    
    def _calculate_temporal_decay(self, creation_time: float) -> float:
        """
        시간적 감쇠 계산
        
        Args:
            creation_time: 패턴 생성 시간 (timestamp)
            
        Returns:
            시간적 감쇠 계수 (0-1)
        """
        try:
            import time
            
            current_time = time.time()
            time_diff = current_time - creation_time
            
            # 시간 단위를 시간(hour)으로 변환
            time_diff_hours = time_diff / 3600.0
            
            # 지수적 감쇠 (반감기 24시간)
            half_life = 24.0  # hours
            decay_factor = 0.5 ** (time_diff_hours / half_life)
            
            return max(0.1, decay_factor)  # 최소 0.1 보장
            
        except Exception:
            return 0.5
    
    def _calculate_pattern_uncertainty(self, similarities: List[float]) -> float:
        """
        베이지안 추론 기반 패턴 불확실성 계산
        
        Args:
            similarities: 유사도 리스트
            
        Returns:
            불확실성 (0-1, 높을수록 불확실)
        """
        try:
            if not similarities:
                return 1.0  # 최대 불확실성
            
            # 분산 계산
            mean_similarity = np.mean(similarities)
            variance = np.var(similarities)
            
            # 엔트로피 기반 불확실성
            # 유사도를 확률로 변환
            similarities_norm = [s / sum(similarities) if sum(similarities) > 0 else 0 for s in similarities]
            
            # 샤논 엔트로피 계산
            entropy = 0.0
            for p in similarities_norm:
                if p > 0:
                    entropy -= p * np.log2(p)
            
            # 최대 엔트로피로 정규화
            max_entropy = np.log2(len(similarities)) if len(similarities) > 1 else 1.0
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            # 분산과 엔트로피를 결합한 불확실성
            uncertainty = 0.6 * normalized_entropy + 0.4 * min(variance, 1.0)
            
            return max(0.0, min(1.0, uncertainty))
            
        except Exception:
            return 0.5
    
    def _calculate_social_context_weight(self, social_context: Dict[str, Any]) -> float:
        """
        사회적 맥락 가중치 계산
        
        Args:
            social_context: 사회적 맥락 정보
            
        Returns:
            사회적 맥락 가중치
        """
        try:
            weight = 1.0
            
            # 사회적 상호작용 강도
            interaction_intensity = social_context.get('interaction_intensity', 0.5)
            weight += interaction_intensity * 0.3
            
            # 감정적 연결 강도
            emotional_connection = social_context.get('emotional_connection', 0.5)
            weight += emotional_connection * 0.2
            
            # 문화적 유사성
            cultural_similarity = social_context.get('cultural_similarity', 0.5)
            weight += cultural_similarity * 0.1
            
            # 관계 친밀도
            relationship_closeness = social_context.get('relationship_closeness', 0.5)
            weight += relationship_closeness * 0.2
            
            return max(0.5, min(2.0, weight))
            
        except Exception:
            return 1.0
    
    def _calculate_emotional_contagion_factor(self, pattern_data: Dict[str, Any]) -> float:
        """
        감정 전염 인자 계산
        
        Args:
            pattern_data: 패턴 데이터
            
        Returns:
            감정 전염 인자 (0-1)
        """
        try:
            # 감정 강도
            emotion_intensity = pattern_data.get('emotion_intensity', 0.5)
            
            # 감정 전염성 (특정 감정이 더 전염성이 높음)
            emotion_type = pattern_data.get('emotion_type', 'neutral')
            contagion_weights = {
                'joy': 0.8,
                'excitement': 0.9,
                'anger': 0.7,
                'fear': 0.6,
                'sadness': 0.4,
                'surprise': 0.7,
                'disgust': 0.3,
                'neutral': 0.5
            }
            
            base_contagion = contagion_weights.get(emotion_type, 0.5)
            
            # 표현 강도
            expression_intensity = pattern_data.get('expression_intensity', 0.5)
            
            # 최종 전염 인자
            contagion_factor = (
                0.4 * emotion_intensity +
                0.4 * base_contagion +
                0.2 * expression_intensity
            )
            
            return max(0.0, min(1.0, contagion_factor))
            
        except Exception:
            return 0.5
    
    # =============================================================================
    # Free Energy Principle 기반 헬퍼 메서드들
    # =============================================================================
    
    def _extract_behavioral_features(self, data: Dict[str, Any]) -> List[float]:
        """행동 데이터에서 특성 벡터 추출"""
        try:
            features = []
            
            # 감정 차원 특성
            if 'emotion' in data:
                emotion = data['emotion']
                features.extend([
                    emotion.get('valence', 0.0),
                    emotion.get('arousal', 0.0),
                    emotion.get('dominance', 0.0)
                ])
            
            # 행동 특성
            features.extend([
                data.get('intensity', 0.0),
                data.get('empathy_component', 0.0),
                data.get('motor_resonance', 0.0),
                data.get('emotional_contagion', 0.0)
            ])
            
            # 예측 성분
            prediction = data.get('prediction_component', {})
            if prediction:
                features.extend([
                    prediction.get('confidence', 0.0),
                    prediction.get('time_horizon', 0.0)
                ])
            
            return features
            
        except Exception:
            return []
    
    def _estimate_probability_distribution(self, features: List[float]) -> Dict[str, float]:
        """특성 벡터에서 확률 분포 추정"""
        try:
            if not features:
                return {}
            
            # 가우시안 분포 가정하고 평균과 표준편차 계산
            mean = np.mean(features)
            std = np.std(features)
            
            # 베이지안 추론을 위한 prior 고려
            alpha = self.bayesian_params['prior_alpha']
            beta = self.bayesian_params['prior_beta']
            
            # 베타 분포 파라미터 (0-1 범위로 정규화된 특성에 대해)
            normalized_features = [(f + 1) / 2 for f in features]  # -1~1 -> 0~1
            
            return {
                'mean': mean,
                'std': max(std, 0.01),  # 0 방지
                'alpha': alpha + sum(normalized_features),
                'beta': beta + len(normalized_features) - sum(normalized_features)
            }
            
        except Exception:
            return {'mean': 0.0, 'std': 1.0, 'alpha': 1.0, 'beta': 1.0}
    
    def _calculate_kl_divergence(self, p_dist: Dict[str, float], q_dist: Dict[str, float]) -> float:
        """KL-divergence 계산 (p에서 q로)"""
        try:
            # 베타 분포의 KL-divergence
            alpha_p = p_dist.get('alpha', 1.0)
            beta_p = p_dist.get('beta', 1.0)
            alpha_q = q_dist.get('alpha', 1.0)
            beta_q = q_dist.get('beta', 1.0)
            
            # 베타 분포 KL-divergence 공식
            from scipy.special import digamma, gammaln
            
            kl_div = (
                gammaln(alpha_p + beta_p) - gammaln(alpha_p) - gammaln(beta_p) -
                gammaln(alpha_q + beta_q) + gammaln(alpha_q) + gammaln(beta_q) +
                (alpha_p - alpha_q) * (digamma(alpha_p) - digamma(alpha_p + beta_p)) +
                (beta_p - beta_q) * (digamma(beta_p) - digamma(alpha_p + beta_p))
            )
            
            return max(0.0, min(kl_div, 10.0))  # 클리핑
            
        except Exception:
            # 폴백: 가우시안 분포 KL-divergence 근사
            try:
                mean_p = p_dist.get('mean', 0.0)
                std_p = p_dist.get('std', 1.0)
                mean_q = q_dist.get('mean', 0.0)
                std_q = q_dist.get('std', 1.0)
                
                kl_div = np.log(std_q / std_p) + (std_p**2 + (mean_p - mean_q)**2) / (2 * std_q**2) - 0.5
                return max(0.0, min(kl_div, 10.0))
                
            except Exception:
                return 0.5
    
    def _calculate_temporal_prediction_error(self, observed: Dict[str, Any], predicted: Dict[str, Any]) -> float:
        """시간적 예측 오차 계산"""
        try:
            # 시간적 일관성 비교
            obs_temporal = observed.get('temporal_pattern', [])
            pred_temporal = predicted.get('temporal_pattern', [])
            
            if not obs_temporal or not pred_temporal:
                return 0.3  # 중간 오차
            
            # 시계열 상관관계 계산
            min_len = min(len(obs_temporal), len(pred_temporal))
            if min_len < 2:
                return 0.5
            
            obs_seq = obs_temporal[:min_len]
            pred_seq = pred_temporal[:min_len]
            
            # 피어슨 상관계수
            correlation = np.corrcoef(obs_seq, pred_seq)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
            
            # 상관계수를 오차로 변환
            temporal_error = 1.0 - abs(correlation)
            
            return max(0.0, min(1.0, temporal_error))
            
        except Exception:
            return 0.5
    
    def _calculate_emotional_prediction_error(self, observed: Dict[str, Any], predicted: Dict[str, Any]) -> float:
        """감정적 차원 예측 오차 계산"""
        try:
            obs_emotion = observed.get('emotion', {})
            pred_emotion = predicted.get('emotion', {})
            
            # 감정 차원별 오차 계산
            valence_error = abs(obs_emotion.get('valence', 0.0) - pred_emotion.get('valence', 0.0))
            arousal_error = abs(obs_emotion.get('arousal', 0.0) - pred_emotion.get('arousal', 0.0))
            dominance_error = abs(obs_emotion.get('dominance', 0.0) - pred_emotion.get('dominance', 0.0))
            
            # 가중 평균
            emotional_error = (0.4 * valence_error + 0.3 * arousal_error + 0.3 * dominance_error) / 2.0
            
            return max(0.0, min(1.0, emotional_error))
            
        except Exception:
            return 0.3
    
    def _calculate_intensity_prediction_error(self, observed: Dict[str, Any], predicted: Dict[str, Any]) -> float:
        """강도 예측 오차 계산"""
        try:
            obs_intensity = observed.get('intensity', 0.0)
            pred_intensity = predicted.get('intensity', 0.0)
            
            intensity_error = abs(obs_intensity - pred_intensity)
            
            return max(0.0, min(1.0, intensity_error))
            
        except Exception:
            return 0.3
    
    def _calculate_bayesian_uncertainty(self, p_dist: Dict[str, float], q_dist: Dict[str, float]) -> float:
        """베이지안 불확실성 계산"""
        try:
            # 분포의 분산 기반 불확실성
            std_p = p_dist.get('std', 1.0)
            std_q = q_dist.get('std', 1.0)
            
            # 높은 분산은 높은 불확실성
            uncertainty = (std_p + std_q) / 2.0
            
            return max(0.0, min(1.0, uncertainty))
            
        except Exception:
            return 0.5
    
    # =============================================================================
    # 모델 복잡도 계산 헬퍼 메서드들
    # =============================================================================
    
    def _calculate_network_activation_complexity(self, response: Dict[str, Any]) -> float:
        """네트워크 활성화 복잡도 계산"""
        try:
            # 활성화 패턴의 다양성 측정
            activation_pattern = response.get('neural_activation', {})
            neuron_activations = activation_pattern.get('neuron_activations', {})
            
            if not neuron_activations:
                return 0.2
            
            # 활성화 엔트로피 계산
            activations = []
            for neuron_data in neuron_activations.values():
                activations.append(neuron_data.get('activation_strength', 0.0))
            
            if not activations:
                return 0.2
            
            # 정규화
            max_activation = max(activations)
            if max_activation > 0:
                normalized_activations = [a / max_activation for a in activations]
            else:
                normalized_activations = activations
            
            # 엔트로피 계산
            entropy = 0.0
            for a in normalized_activations:
                if a > 0:
                    entropy -= a * np.log2(a)
            
            # 최대 엔트로피로 정규화
            max_entropy = np.log2(len(activations))
            complexity = entropy / max_entropy if max_entropy > 0 else 0.0
            
            return max(0.0, min(1.0, complexity))
            
        except Exception:
            return 0.3
    
    def _calculate_information_complexity(self, response: Dict[str, Any]) -> float:
        """정보 이론 기반 복잡도 계산"""
        try:
            # 반응 데이터의 정보 내용 측정
            info_content = 0.0
            
            # 각 구성 요소의 정보량 계산
            for key, value in response.items():
                if isinstance(value, (int, float)):
                    # 수치값의 정보량 (0에 가까울수록 낮은 정보량)
                    info_content += abs(value)
                elif isinstance(value, dict):
                    # 딕셔너리의 복잡도
                    info_content += len(value) * 0.1
                elif isinstance(value, list):
                    # 리스트의 복잡도
                    info_content += len(value) * 0.05
            
            # 정규화
            complexity = min(info_content / 10.0, 1.0)
            
            return max(0.0, min(1.0, complexity))
            
        except Exception:
            return 0.3
    
    def _calculate_hierarchical_complexity(self, response: Dict[str, Any]) -> float:
        """계층적 구조 복잡도 계산"""
        try:
            # 중첩 구조의 깊이 측정
            def calculate_depth(obj, current_depth=0):
                if isinstance(obj, dict):
                    if not obj:
                        return current_depth
                    return max(calculate_depth(v, current_depth + 1) for v in obj.values())
                elif isinstance(obj, list):
                    if not obj:
                        return current_depth
                    return max(calculate_depth(item, current_depth + 1) for item in obj)
                else:
                    return current_depth
            
            depth = calculate_depth(response)
            
            # 깊이를 복잡도로 변환
            complexity = min(depth / 5.0, 1.0)
            
            return max(0.0, min(1.0, complexity))
            
        except Exception:
            return 0.3
    
    def _calculate_temporal_dependency_complexity(self, response: Dict[str, Any]) -> float:
        """시간적 의존성 복잡도 계산"""
        try:
            # 시간적 패턴의 복잡도 측정
            temporal_data = response.get('temporal_consistency', 0.0)
            
            # 높은 시간적 일관성 = 낮은 복잡도
            complexity = 1.0 - temporal_data
            
            return max(0.0, min(1.0, complexity))
            
        except Exception:
            return 0.3
    
    def _calculate_interaction_complexity(self, response: Dict[str, Any]) -> float:
        """상호작용 복잡도 계산"""
        try:
            # 구성 요소 간 상호작용 복잡도
            interaction_factors = [
                response.get('motor_resonance', 0.0),
                response.get('emotional_contagion', 0.0),
                response.get('empathy_component', 0.0)
            ]
            
            # 상호작용 강도의 분산
            if len(interaction_factors) > 1:
                variance = np.var(interaction_factors)
                complexity = min(variance * 4.0, 1.0)  # 분산이 클수록 복잡
            else:
                complexity = 0.3
            
            return max(0.0, min(1.0, complexity))
            
        except Exception:
            return 0.3
    
    def _calculate_bayesian_model_complexity(self, response: Dict[str, Any]) -> float:
        """베이지안 모델 복잡도 계산"""
        try:
            # 사전 분포와 사후 분포 간의 차이
            confidence = response.get('confidence', 0.5)
            
            # 낮은 신뢰도 = 높은 모델 복잡도
            complexity = 1.0 - confidence
            
            return max(0.0, min(1.0, complexity))
            
        except Exception:
            return 0.3
    
    # =============================================================================
    # 시계열 분석 기반 시간적 일관성 검증 헬퍼 메서드들
    # =============================================================================
    
    def _extract_temporal_series(self, data: Dict[str, Any]) -> List[float]:
        """시계열 데이터 추출"""
        try:
            series = []
            
            # 다양한 시계열 데이터 소스 확인
            if 'temporal_spikes' in data:
                series.extend(data['temporal_spikes'])
            
            if 'emotion_spikes' in data:
                emotion_spikes = data['emotion_spikes']
                if isinstance(emotion_spikes, dict):
                    for emotion_dim in ['valence', 'arousal', 'dominance']:
                        if emotion_dim in emotion_spikes:
                            series.extend(emotion_spikes[emotion_dim])
                            
            if 'intensity_spikes' in data:
                series.extend(data['intensity_spikes'])
            
            if 'spatial_spikes' in data:
                spatial_spikes = data['spatial_spikes']
                if isinstance(spatial_spikes, dict):
                    for location_spikes in spatial_spikes.values():
                        series.extend(location_spikes)
            
            # 시계열 데이터가 없는 경우 스칼라 값들로 간단한 시계열 생성
            if not series:
                series = [
                    data.get('intensity', 0.0),
                    data.get('empathy_component', 0.0),
                    data.get('motor_resonance', 0.0),
                    data.get('emotional_contagion', 0.0)
                ]
            
            return series
            
        except Exception:
            return []
    
    def _calculate_autocorrelation_consistency(self, series1: List[float], series2: List[float]) -> float:
        """자기상관 일관성 계산"""
        try:
            # 시계열 길이 맞추기
            min_len = min(len(series1), len(series2))
            if min_len < 3:
                return 0.5
            
            s1 = series1[:min_len]
            s2 = series2[:min_len]
            
            # 각 시계열의 자기상관 계산
            autocorr1 = self._calculate_autocorrelation(s1)
            autocorr2 = self._calculate_autocorrelation(s2)
            
            # 자기상관 패턴의 유사도 계산
            if len(autocorr1) == len(autocorr2) and len(autocorr1) > 0:
                correlation = np.corrcoef(autocorr1, autocorr2)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
                
                # 상관관계를 일관성으로 변환
                consistency = (correlation + 1.0) / 2.0
                return max(0.0, min(1.0, consistency))
            
            return 0.5
            
        except Exception:
            return 0.5
    
    def _calculate_autocorrelation(self, series: List[float]) -> List[float]:
        """자기상관 계산"""
        try:
            n = len(series)
            if n < 2:
                return [1.0]
            
            # 평균 제거
            mean_series = np.mean(series)
            centered_series = [x - mean_series for x in series]
            
            # 자기상관 계산
            autocorr = []
            max_lag = min(n // 2, self.temporal_params['lag_max'])
            
            for lag in range(max_lag + 1):
                if lag == 0:
                    autocorr.append(1.0)
                else:
                    numerator = sum(centered_series[i] * centered_series[i + lag] 
                                  for i in range(n - lag))
                    denominator = sum(x**2 for x in centered_series)
                    
                    if denominator > 0:
                        autocorr.append(numerator / denominator)
                    else:
                        autocorr.append(0.0)
            
            return autocorr
            
        except Exception:
            return [1.0]
    
    def _calculate_change_rate_consistency(self, series1: List[float], series2: List[float]) -> float:
        """변화율 일관성 계산"""
        try:
            # 변화율 계산
            changes1 = self._calculate_change_rates(series1)
            changes2 = self._calculate_change_rates(series2)
            
            if not changes1 or not changes2:
                return 0.5
            
            # 변화율 패턴의 일관성 측정
            min_len = min(len(changes1), len(changes2))
            if min_len < 2:
                return 0.5
            
            c1 = changes1[:min_len]
            c2 = changes2[:min_len]
            
            # 변화율 상관관계
            correlation = np.corrcoef(c1, c2)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
            
            # 변화율 크기 일관성
            mag_consistency = 1.0 - abs(np.std(c1) - np.std(c2)) / max(np.std(c1) + np.std(c2), 0.1)
            
            # 종합 일관성
            consistency = 0.7 * ((correlation + 1.0) / 2.0) + 0.3 * mag_consistency
            
            return max(0.0, min(1.0, consistency))
            
        except Exception:
            return 0.5
    
    def _calculate_change_rates(self, series: List[float]) -> List[float]:
        """변화율 계산"""
        try:
            if len(series) < 2:
                return []
            
            changes = []
            for i in range(1, len(series)):
                change = series[i] - series[i-1]
                changes.append(change)
            
            return changes
            
        except Exception:
            return []
    
    def _calculate_trend_consistency(self, series1: List[float], series2: List[float]) -> float:
        """추세 일관성 계산"""
        try:
            # 선형 추세 계산
            trend1 = self._calculate_linear_trend(series1)
            trend2 = self._calculate_linear_trend(series2)
            
            # 추세 방향 일관성
            direction_consistency = 1.0 if trend1 * trend2 >= 0 else 0.0
            
            # 추세 강도 일관성
            strength_consistency = 1.0 - abs(abs(trend1) - abs(trend2)) / max(abs(trend1) + abs(trend2), 0.1)
            
            # 종합 일관성
            consistency = 0.6 * direction_consistency + 0.4 * strength_consistency
            
            return max(0.0, min(1.0, consistency))
            
        except Exception:
            return 0.5
    
    def _calculate_linear_trend(self, series: List[float]) -> float:
        """선형 추세 계산"""
        try:
            if len(series) < 2:
                return 0.0
            
            n = len(series)
            x = list(range(n))
            
            # 최소제곱법으로 기울기 계산
            sum_x = sum(x)
            sum_y = sum(series)
            sum_xy = sum(x[i] * series[i] for i in range(n))
            sum_x2 = sum(xi**2 for xi in x)
            
            denominator = n * sum_x2 - sum_x**2
            if denominator == 0:
                return 0.0
            
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            
            return slope
            
        except Exception:
            return 0.0
    
    def _calculate_seasonality_consistency(self, series1: List[float], series2: List[float]) -> float:
        """계절성 일관성 계산"""
        try:
            # 주기성 분석
            period = self.temporal_params['seasonality_period']
            
            if len(series1) < period * 2 or len(series2) < period * 2:
                return 0.7  # 짧은 시계열의 경우 중간값
            
            # 주기적 패턴 추출
            periodic1 = self._extract_periodic_pattern(series1, period)
            periodic2 = self._extract_periodic_pattern(series2, period)
            
            if not periodic1 or not periodic2:
                return 0.7
            
            # 주기적 패턴 일관성
            correlation = np.corrcoef(periodic1, periodic2)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
            
            consistency = (correlation + 1.0) / 2.0
            
            return max(0.0, min(1.0, consistency))
            
        except Exception:
            return 0.7
    
    def _extract_periodic_pattern(self, series: List[float], period: int) -> List[float]:
        """주기적 패턴 추출"""
        try:
            if len(series) < period:
                return []
            
            # 주기별 평균 계산
            pattern = []
            for i in range(period):
                values = [series[j] for j in range(i, len(series), period)]
                if values:
                    pattern.append(np.mean(values))
                else:
                    pattern.append(0.0)
            
            return pattern
            
        except Exception:
            return []
    
    def _calculate_noise_consistency(self, series1: List[float], series2: List[float]) -> float:
        """노이즈 일관성 계산"""
        try:
            # 노이즈 수준 계산 (고주파 성분의 분산)
            noise1 = self._calculate_noise_level(series1)
            noise2 = self._calculate_noise_level(series2)
            
            # 노이즈 수준 일관성
            if noise1 + noise2 > 0:
                consistency = 1.0 - abs(noise1 - noise2) / (noise1 + noise2)
            else:
                consistency = 1.0
            
            return max(0.0, min(1.0, consistency))
            
        except Exception:
            return 0.5
    
    def _calculate_noise_level(self, series: List[float]) -> float:
        """노이즈 수준 계산"""
        try:
            if len(series) < 3:
                return 0.0
            
            # 2차 차분을 통한 노이즈 추정
            second_diff = []
            for i in range(2, len(series)):
                diff = series[i] - 2 * series[i-1] + series[i-2]
                second_diff.append(diff)
            
            if second_diff:
                noise_level = np.std(second_diff)
            else:
                noise_level = 0.0
            
            return noise_level
            
        except Exception:
            return 0.0
    
    def _calculate_longterm_dependency_consistency(self, series1: List[float], series2: List[float]) -> float:
        """장기 의존성 일관성 계산"""
        try:
            # Hurst 지수 계산 (장기 의존성 측정)
            hurst1 = self._calculate_hurst_exponent(series1)
            hurst2 = self._calculate_hurst_exponent(series2)
            
            # Hurst 지수 일관성
            consistency = 1.0 - abs(hurst1 - hurst2) / max(hurst1 + hurst2, 0.1)
            
            return max(0.0, min(1.0, consistency))
            
        except Exception:
            return 0.5
    
    def _calculate_hurst_exponent(self, series: List[float]) -> float:
        """Hurst 지수 계산"""
        try:
            if len(series) < 10:
                return 0.5  # 랜덤 워크의 경우
            
            # R/S 분석
            n = len(series)
            mean_series = np.mean(series)
            
            # 누적 편차 계산
            cumulative_devs = []
            cumsum = 0.0
            for x in series:
                cumsum += (x - mean_series)
                cumulative_devs.append(cumsum)
            
            # 범위 계산
            if cumulative_devs:
                R = max(cumulative_devs) - min(cumulative_devs)
            else:
                R = 0.0
            
            # 표준편차 계산
            S = np.std(series)
            
            if S > 0:
                rs = R / S
                if rs > 0:
                    hurst = np.log(rs) / np.log(n)
                else:
                    hurst = 0.5
            else:
                hurst = 0.5
            
            return max(0.0, min(1.0, hurst))
            
        except Exception:
            return 0.5
    
    def _calculate_temporal_lag_consistency(self, series1: List[float], series2: List[float]) -> float:
        """시간적 지연 일관성 계산"""
        try:
            # 교차상관 계산으로 최적 지연 찾기
            max_lag = min(len(series1) // 4, len(series2) // 4, 10)
            
            if max_lag < 1:
                return 0.7
            
            best_corr = 0.0
            best_lag = 0
            
            for lag in range(-max_lag, max_lag + 1):
                if lag == 0:
                    s1 = series1
                    s2 = series2
                elif lag > 0:
                    s1 = series1[lag:]
                    s2 = series2[:-lag]
                else:
                    s1 = series1[:lag]
                    s2 = series2[-lag:]
                
                if len(s1) != len(s2) or len(s1) < 2:
                    continue
                
                corr = np.corrcoef(s1, s2)[0, 1]
                if not np.isnan(corr) and abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag
            
            # 지연이 적을수록 높은 일관성
            lag_consistency = 1.0 - abs(best_lag) / max_lag
            
            # 상관관계 강도 고려
            corr_strength = abs(best_corr)
            
            consistency = 0.6 * lag_consistency + 0.4 * corr_strength
            
            return max(0.0, min(1.0, consistency))
            
        except Exception:
            return 0.5
    
    def _update_temporal_consistency_metrics(self, consistency: float):
        """시간적 일관성 메트릭 업데이트"""
        try:
            # 성능 메트릭에 추가
            self.performance_metrics['temporal_consistency'].append(consistency)
            
            # 최대 크기 제한 (메모리 관리)
            max_size = 1000
            if len(self.performance_metrics['temporal_consistency']) > max_size:
                self.performance_metrics['temporal_consistency'] = \
                    self.performance_metrics['temporal_consistency'][-max_size:]
            
        except Exception:
            pass
    
    # =============================================================================
    # 학습 기반 활성화 조정 헬퍼 메서드들
    # =============================================================================
    
    def _detect_novelty(self, activation_pattern: Dict[str, Any], context: Dict[str, Any]) -> float:
        """
        노벨티 검출 - 새로운 패턴 인식
        
        Args:
            activation_pattern: 활성화 패턴
            context: 상황 컨텍스트
            
        Returns:
            노벨티 점수 (0-1, 높을수록 새로운 패턴)
        """
        try:
            # 현재 패턴을 벡터로 변환
            current_vector = self._activation_to_vector(activation_pattern)
            
            if not current_vector:
                return 0.5
            
            # 저장된 패턴들과 비교
            all_patterns = []
            
            # 자기 패턴과 비교
            for pattern_data in self.pattern_database['self_patterns'].values():
                all_patterns.append(pattern_data['vector'])
            
            # 타인 패턴과 비교
            for pattern_data in self.pattern_database['other_patterns'].values():
                all_patterns.append(pattern_data['vector'])
            
            if not all_patterns:
                return 1.0  # 패턴이 없으면 완전히 새로운 것
            
            # 최대 유사도 계산
            max_similarity = 0.0
            for pattern_vector in all_patterns:
                similarity = self._cosine_similarity(current_vector, pattern_vector)
                max_similarity = max(max_similarity, similarity)
            
            # 유사도를 노벨티로 변환
            novelty = 1.0 - max_similarity
            
            # 상황적 맥락 고려
            context_novelty = self._calculate_context_novelty(context)
            
            # 종합 노벨티
            combined_novelty = 0.7 * novelty + 0.3 * context_novelty
            
            return max(0.0, min(1.0, combined_novelty))
            
        except Exception:
            return 0.5
    
    def _calculate_context_novelty(self, context: Dict[str, Any]) -> float:
        """상황적 맥락 노벨티 계산"""
        try:
            # 상황 특성 추출
            context_features = []
            
            if 'social_context' in context:
                context_features.append(context['social_context'])
            if 'emotional_context' in context:
                context_features.append(context['emotional_context'])
            if 'temporal_context' in context:
                context_features.append(context['temporal_context'])
            
            # 저장된 상황과 비교
            stored_contexts = self.pattern_database.get('context_patterns', {})
            
            if not stored_contexts:
                return 0.8  # 상황 데이터가 없으면 높은 노벨티
            
            # 상황 유사도 계산
            similarities = []
            for stored_context in stored_contexts.values():
                similarity = self._calculate_context_similarity(context, stored_context)
                similarities.append(similarity)
            
            if similarities:
                max_similarity = max(similarities)
                return 1.0 - max_similarity
            
            return 0.8
            
        except Exception:
            return 0.5
    
    def _calculate_context_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """상황 유사도 계산"""
        try:
            shared_keys = set(context1.keys()) & set(context2.keys())
            if not shared_keys:
                return 0.0
            
            similarities = []
            for key in shared_keys:
                val1 = context1[key]
                val2 = context2[key]
                
                if isinstance(val1, str) and isinstance(val2, str):
                    sim = 1.0 if val1 == val2 else 0.0
                elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    sim = 1.0 - abs(val1 - val2) / max(abs(val1) + abs(val2), 1.0)
                else:
                    sim = 0.5
                
                similarities.append(sim)
            
            return np.mean(similarities) if similarities else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_experience_weights(self, activation_pattern: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, float]:
        """경험 기반 가중치 계산"""
        try:
            weights = {}
            
            # 뉴런 그룹별 가중치 계산
            neuron_activations = activation_pattern.get('neuron_activations', {})
            
            for neuron_group in neuron_activations.keys():
                # 기본 가중치
                base_weight = 1.0
                
                # 과거 성공 경험 기반 가중치
                success_weight = self._calculate_success_weight(neuron_group)
                
                # 빈도 기반 가중치
                frequency_weight = self._calculate_frequency_weight(neuron_group)
                
                # 최신성 기반 가중치
                recency_weight = self._calculate_recency_weight(neuron_group)
                
                # 상황 유사성 기반 가중치
                context_weight = self._calculate_context_weight(neuron_group, context)
                
                # 종합 가중치
                combined_weight = (
                    0.3 * success_weight +
                    0.2 * frequency_weight +
                    0.2 * recency_weight +
                    0.3 * context_weight
                )
                
                weights[neuron_group] = max(0.1, min(2.0, combined_weight))
            
            return weights
            
        except Exception:
            return {}
    
    def _calculate_success_weight(self, neuron_group: str) -> float:
        """성공 경험 기반 가중치"""
        try:
            # 과거 성공률 기반 가중치 계산
            # 실제로는 학습 히스토리에서 성공률을 계산
            return 1.0 + 0.2 * np.random.random()  # 임시 구현
            
        except Exception:
            return 1.0
    
    def _calculate_frequency_weight(self, neuron_group: str) -> float:
        """빈도 기반 가중치"""
        try:
            # 패턴 빈도 기반 가중치
            frequencies = self.pattern_database.get('pattern_frequencies', {})
            frequency = frequencies.get(neuron_group, 1)
            
            # 빈도를 가중치로 변환 (너무 높지 않게)
            weight = min(1.0 + 0.1 * np.log(frequency), 1.5)
            
            return weight
            
        except Exception:
            return 1.0
    
    def _calculate_recency_weight(self, neuron_group: str) -> float:
        """최신성 기반 가중치"""
        try:
            # 최근 사용된 패턴에 더 높은 가중치
            # 실제로는 시간 기반 가중치 계산
            return 1.0 + 0.1 * np.random.random()  # 임시 구현
            
        except Exception:
            return 1.0
    
    def _calculate_context_weight(self, neuron_group: str, context: Dict[str, Any]) -> float:
        """상황 유사성 기반 가중치"""
        try:
            # 현재 상황과 유사한 상황에서의 성공률 기반 가중치
            context_patterns = self.pattern_database.get('context_patterns', {})
            
            if not context_patterns:
                return 1.0
            
            # 상황 유사성 계산
            similarities = []
            for pattern_id, pattern_context in context_patterns.items():
                similarity = self._calculate_context_similarity(context, pattern_context)
                similarities.append(similarity)
            
            if similarities:
                avg_similarity = np.mean(similarities)
                return 1.0 + 0.3 * avg_similarity
            
            return 1.0
            
        except Exception:
            return 1.0
    
    def _apply_meta_learning(self, activation_pattern: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, float]:
        """메타학습 적용"""
        try:
            adjustments = {}
            
            # 메타학습 파라미터
            meta_lr = self.learning_params['meta_learning_rate']
            
            # 뉴런 그룹별 메타학습 조정
            neuron_activations = activation_pattern.get('neuron_activations', {})
            
            for neuron_group in neuron_activations.keys():
                # 메타학습 기반 조정값 계산
                # 실제로는 과거 학습 성능 기반으로 조정
                
                # 학습 추세 분석
                learning_trend = self._analyze_learning_trend(neuron_group)
                
                # 일반화 성능 분석
                generalization_score = self._analyze_generalization(neuron_group)
                
                # 적응 속도 분석
                adaptation_speed = self._analyze_adaptation_speed(neuron_group)
                
                # 메타학습 조정값 계산
                meta_adjustment = meta_lr * (
                    0.4 * learning_trend +
                    0.3 * generalization_score +
                    0.3 * adaptation_speed
                )
                
                adjustments[neuron_group] = max(-0.2, min(0.2, meta_adjustment))
            
            return adjustments
            
        except Exception:
            return {}
    
    def _analyze_learning_trend(self, neuron_group: str) -> float:
        """학습 추세 분석"""
        try:
            # 학습 성능 추세 분석
            # 실제로는 학습 히스토리 분석
            return 0.1 * (np.random.random() - 0.5)  # 임시 구현
            
        except Exception:
            return 0.0
    
    def _analyze_generalization(self, neuron_group: str) -> float:
        """일반화 성능 분석"""
        try:
            # 일반화 성능 분석
            # 실제로는 다양한 상황에서의 성능 분석
            return 0.1 * (np.random.random() - 0.5)  # 임시 구현
            
        except Exception:
            return 0.0
    
    def _analyze_adaptation_speed(self, neuron_group: str) -> float:
        """적응 속도 분석"""
        try:
            # 적응 속도 분석
            # 실제로는 새로운 상황에 대한 적응 속도 분석
            return 0.1 * (np.random.random() - 0.5)  # 임시 구현
            
        except Exception:
            return 0.0
    
    def _apply_reinforcement_learning(self, activation_pattern: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, float]:
        """강화학습 적용"""
        try:
            adjustments = {}
            
            # 강화학습 파라미터
            learning_rate = self.learning_params['update_learning_rate']
            
            # 뉴런 그룹별 강화학습 조정
            neuron_activations = activation_pattern.get('neuron_activations', {})
            
            for neuron_group in neuron_activations.keys():
                # 보상 신호 계산
                reward = self._calculate_reward_signal(neuron_group, context)
                
                # Q-값 업데이트 (간단한 TD 학습)
                q_value = self._get_q_value(neuron_group, context)
                target_q = reward + 0.9 * self._get_max_q_value(neuron_group)
                
                # TD 오차 계산
                td_error = target_q - q_value
                
                # 강화학습 조정값
                rl_adjustment = learning_rate * td_error
                
                adjustments[neuron_group] = max(-0.3, min(0.3, rl_adjustment))
            
            return adjustments
            
        except Exception:
            return {}
    
    def _calculate_reward_signal(self, neuron_group: str, context: Dict[str, Any]) -> float:
        """보상 신호 계산"""
        try:
            # 보상 신호 계산
            # 실제로는 성능 기반 보상 계산
            base_reward = 0.1
            
            # 상황별 보상 조정
            if context.get('social_context') == 'positive':
                base_reward += 0.1
            if context.get('emotional_context') == 'empathetic':
                base_reward += 0.05
            
            return base_reward
            
        except Exception:
            return 0.0
    
    def _get_q_value(self, neuron_group: str, context: Dict[str, Any]) -> float:
        """Q-값 조회"""
        try:
            # Q-값 테이블에서 조회
            # 실제로는 Q-테이블 또는 함수 근사 사용
            return 0.5 + 0.2 * np.random.random()  # 임시 구현
            
        except Exception:
            return 0.5
    
    def _get_max_q_value(self, neuron_group: str) -> float:
        """최대 Q-값 조회"""
        try:
            # 다음 상태에서의 최대 Q-값
            return 0.6 + 0.1 * np.random.random()  # 임시 구현
            
        except Exception:
            return 0.6
    
    def _apply_contextual_learning(self, activation_pattern: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, float]:
        """상황별 학습 적용"""
        try:
            adjustments = {}
            
            # 상황별 학습 파라미터
            context_sensitivity = 0.1
            
            # 뉴런 그룹별 상황별 조정
            neuron_activations = activation_pattern.get('neuron_activations', {})
            
            for neuron_group in neuron_activations.keys():
                # 상황 적합성 계산
                context_fitness = self._calculate_context_fitness(neuron_group, context)
                
                # 상황별 조정값
                contextual_adjustment = context_sensitivity * (context_fitness - 0.5)
                
                adjustments[neuron_group] = max(-0.1, min(0.1, contextual_adjustment))
            
            return adjustments
            
        except Exception:
            return {}
    
    def _calculate_context_fitness(self, neuron_group: str, context: Dict[str, Any]) -> float:
        """상황 적합성 계산"""
        try:
            # 현재 상황에서의 뉴런 그룹 적합성
            fitness = 0.5  # 기본값
            
            # 사회적 상황 적합성
            if context.get('social_context') == 'interaction':
                fitness += 0.2
            
            # 감정적 상황 적합성
            if context.get('emotional_context') == 'empathetic':
                fitness += 0.2
            
            # 시간적 상황 적합성
            if context.get('temporal_context') == 'recent':
                fitness += 0.1
            
            return max(0.0, min(1.0, fitness))
            
        except Exception:
            return 0.5
    
    def _store_learning_experience(self, original_pattern: Dict[str, Any], 
                                 adjusted_pattern: Dict[str, Any], 
                                 context: Dict[str, Any]):
        """학습 경험 저장"""
        try:
            import time
            
            # 경험 데이터 구성
            experience = {
                'timestamp': time.time(),
                'original_pattern': original_pattern,
                'adjusted_pattern': adjusted_pattern,
                'context': context,
                'learning_outcome': 'pending'  # 나중에 업데이트
            }
            
            # 경험 버퍼에 저장
            if not hasattr(self, 'experience_buffer'):
                self.experience_buffer = []
            
            self.experience_buffer.append(experience)
            
            # 버퍼 크기 제한
            max_size = self.learning_params['experience_buffer_size']
            if len(self.experience_buffer) > max_size:
                self.experience_buffer = self.experience_buffer[-max_size:]
            
            # 패턴 빈도 업데이트
            self._update_pattern_frequencies(adjusted_pattern)
            
        except Exception:
            pass
    
    def _update_pattern_frequencies(self, pattern: Dict[str, Any]):
        """패턴 빈도 업데이트"""
        try:
            neuron_activations = pattern.get('neuron_activations', {})
            
            for neuron_group in neuron_activations.keys():
                if neuron_group not in self.pattern_database['pattern_frequencies']:
                    self.pattern_database['pattern_frequencies'][neuron_group] = 0
                
                self.pattern_database['pattern_frequencies'][neuron_group] += 1
            
        except Exception:
            pass
    
    def _update_learning_metrics(self, novelty_score: float, 
                               experience_weights: Dict[str, float],
                               meta_adjustments: Dict[str, float]):
        """학습 메트릭 업데이트"""
        try:
            # 학습 수렴 메트릭
            if experience_weights:
                weight_variance = np.var(list(experience_weights.values()))
                self.performance_metrics['learning_convergence'].append(weight_variance)
            
            # 패턴 인식 정확도 메트릭
            pattern_accuracy = 1.0 - novelty_score  # 노벨티가 낮으면 인식 정확도 높음
            self.performance_metrics['pattern_recognition_accuracy'].append(pattern_accuracy)
            
            # 메트릭 크기 제한
            for metric_name in ['learning_convergence', 'pattern_recognition_accuracy']:
                if len(self.performance_metrics[metric_name]) > 1000:
                    self.performance_metrics[metric_name] = \
                        self.performance_metrics[metric_name][-1000:]
            
        except Exception:
            pass
    
    # =============================================================================
    # 자타 구분 고도화 시스템 초기화 메서드들
    # =============================================================================
    
    def _initialize_self_other_classification_system(self):
        """
        자타 구분 신경망 시스템 초기화
        
        고도화된 자타 구분 시스템:
        - 심층 신경망 기반 특징 추출
        - Siamese Network 기반 유사도 학습
        - Attention Mechanism 기반 패턴 인식
        - Adversarial Training 기반 강건성 향상
        - Meta-Learning 기반 Few-Shot 학습
        """
        try:
            # 자타 구분 신경망 아키텍처 정의
            self.self_other_classifier['neural_network'] = SelfOtherNeuralNetwork(
                input_dim=self._get_pattern_dimension(),
                hidden_dims=[512, 256, 128, 64],
                output_dim=2,  # Self, Other
                dropout_rate=0.1,
                attention_heads=8,
                use_batch_norm=True,
                activation='gelu'
            )
            
            # 특징 추출기 초기화
            self.self_other_classifier['feature_extractor'] = AdvancedFeatureExtractor(
                input_dim=self._get_pattern_dimension(),
                feature_dim=256,
                extraction_methods=['cnn', 'rnn', 'transformer'],
                fusion_method='attention'
            )
            
            # 판별기 초기화 (Adversarial Training용)
            self.self_other_classifier['discriminator'] = PatternDiscriminator(
                feature_dim=256,
                hidden_dims=[128, 64],
                output_dim=1,
                use_spectral_norm=True,
                gradient_penalty=True
            )
            
            # 메타 분류기 초기화 (Few-Shot Learning용)
            self.pattern_learning_system['meta_classifier'] = MetaClassifier(
                feature_dim=256,
                support_size=5,
                query_size=10,
                meta_learning_rate=0.001,
                inner_learning_rate=0.01,
                num_inner_updates=5
            )
            
            self.logger.info("자타 구분 신경망 시스템 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"자타 구분 시스템 초기화 실패: {e}")
            # 폴백 - 기본 분류기로 대체
            self._initialize_fallback_classifier()
    
    def _initialize_pattern_learning_system(self):
        """
        패턴 학습 시스템 초기화
        
        고도화된 패턴 학습 시스템:
        - 온라인 학습 기반 점진적 학습
        - 계층적 클러스터링 기반 패턴 그룹화
        - 변화점 탐지 기반 노벨티 검출
        - 시계열 기반 패턴 예측
        - 자기조직화 지도 기반 패턴 구조화
        """
        try:
            # 점진적 학습기 초기화
            self.pattern_learning_system['incremental_learner'] = IncrementalLearner(
                base_model='neural_network',
                memory_size=10000,
                learning_rate=0.001,
                decay_factor=0.95,
                adaptation_threshold=0.1,
                catastrophic_forgetting_prevention=True
            )
            
            # 패턴 클러스터링 시스템 초기화
            self.pattern_learning_system['pattern_clustering'] = HierarchicalPatternClustering(
                clustering_method='agglomerative',
                distance_metric='cosine',
                linkage_method='ward',
                min_cluster_size=3,
                max_clusters=50,
                dynamic_cluster_update=True,
                cluster_validation_metric='silhouette'
            )
            
            # 노벨티 검출기 초기화
            self.pattern_learning_system['novelty_detector'] = AdvancedNoveltyDetector(
                detection_methods=['isolation_forest', 'local_outlier_factor', 'one_class_svm'],
                ensemble_method='voting',
                contamination_rate=0.1,
                adaptation_rate=0.05,
                drift_detection=True,
                change_point_detection=True
            )
            
            # 패턴 예측기 초기화
            self.pattern_learning_system['pattern_predictor'] = PatternPredictor(
                model_type='lstm_attention',
                sequence_length=50,
                hidden_size=128,
                num_layers=3,
                attention_heads=4,
                prediction_horizon=10,
                uncertainty_quantification=True
            )
            
            self.logger.info("패턴 학습 시스템 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"패턴 학습 시스템 초기화 실패: {e}")
            # 폴백 - 기본 학습기로 대체
            self._initialize_fallback_learner()
    
    def _initialize_pattern_management_system(self):
        """
        패턴 관리 시스템 초기화
        
        고도화된 패턴 관리 시스템:
        - 계층적 패턴 구조 관리
        - 패턴 관계 그래프 구성
        - 패턴 진화 추적
        - 패턴 통합 및 분할
        - 선택적 망각 메커니즘
        """
        try:
            # 계층적 패턴 구조 초기화
            self.pattern_management['pattern_hierarchy'] = HierarchicalPatternStructure(
                max_depth=5,
                branching_factor=3,
                similarity_threshold=0.8,
                merge_threshold=0.9,
                split_threshold=0.3,
                rebalancing_frequency=100
            )
            
            # 패턴 관계 그래프 초기화
            self.pattern_management['pattern_relationships'] = PatternRelationshipGraph(
                relationship_types=['similarity', 'temporal', 'causal', 'hierarchical'],
                edge_weight_calculation='learned',
                graph_update_method='incremental',
                community_detection=True,
                centrality_metrics=['betweenness', 'closeness', 'eigenvector']
            )
            
            # 패턴 진화 추적 시스템 초기화
            self.pattern_management['pattern_evolution'] = PatternEvolutionTracker(
                evolution_metrics=['stability', 'drift', 'growth', 'decay'],
                tracking_window=1000,
                evolution_threshold=0.2,
                adaptation_speed_measurement=True,
                phylogenetic_analysis=True
            )
            
            # 패턴 통합 시스템 초기화
            self.pattern_management['pattern_consolidation'] = PatternConsolidationSystem(
                consolidation_strategy='entropy_based',
                consolidation_frequency=50,
                similarity_threshold=0.85,
                information_preservation=0.95,
                compression_ratio_target=0.7
            )
            
            # 선택적 망각 시스템 초기화
            self.pattern_management['pattern_forgetting'] = SelectiveForgettingSystem(
                forgetting_strategy='importance_based',
                forgetting_rate=0.01,
                importance_metrics=['frequency', 'recency', 'relevance'],
                memory_pressure_threshold=0.8,
                forgetting_curve_model='exponential'
            )
            
            self.logger.info("패턴 관리 시스템 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"패턴 관리 시스템 초기화 실패: {e}")
            # 폴백 - 기본 관리 시스템으로 대체
            self._initialize_fallback_management()
    
    def _get_pattern_dimension(self) -> int:
        """패턴 차원 계산"""
        # 기본적으로 활성화 패턴 벡터 차원 계산
        # 실제로는 뉴런 그룹 수 * 특징 수로 계산
        base_features = 6  # membrane_potential, spike_generated, activation_strength, mirror_weight, network_activation, spike_count
        neuron_groups = 10  # 예상 뉴런 그룹 수
        temporal_features = self.temporal_window  # 시간적 특징
        spatial_features = 20  # 공간적 특징
        
        return base_features * neuron_groups + temporal_features + spatial_features
    
    def _initialize_fallback_classifier(self):
        """폴백 분류기 초기화"""
        try:
            # 간단한 폴백 분류기 (실제 프로덕션에서는 더 정교해야 함)
            self.self_other_classifier['neural_network'] = SimpleFallbackClassifier(
                input_dim=self._get_pattern_dimension(),
                hidden_dim=64,
                output_dim=2
            )
            
            self.logger.warning("폴백 분류기로 초기화됨")
            
        except Exception as e:
            self.logger.error(f"폴백 분류기 초기화 실패: {e}")
    
    def _initialize_fallback_learner(self):
        """폴백 학습기 초기화"""
        try:
            # 간단한 폴백 학습기
            self.pattern_learning_system['incremental_learner'] = SimpleFallbackLearner(
                learning_rate=0.01,
                memory_size=1000
            )
            
            self.logger.warning("폴백 학습기로 초기화됨")
            
        except Exception as e:
            self.logger.error(f"폴백 학습기 초기화 실패: {e}")
    
    def _initialize_fallback_management(self):
        """폴백 관리 시스템 초기화"""
        try:
            # 간단한 폴백 관리 시스템
            self.pattern_management['pattern_hierarchy'] = SimpleFallbackManager(
                max_patterns=10000,
                cleanup_threshold=0.8
            )
            
            self.logger.warning("폴백 관리 시스템으로 초기화됨")
            
        except Exception as e:
            self.logger.error(f"폴백 관리 시스템 초기화 실패: {e}")
    
    # =============================================================================
    # 고도화된 패턴 저장 및 관리 메서드들
    # =============================================================================
    
    async def store_pattern(self, pattern_vector: List[float], 
                          pattern_type: str, 
                          context: Dict[str, Any],
                          metadata: Dict[str, Any] = None) -> str:
        """
        패턴 저장 (고도화된 버전)
        
        고도화된 패턴 저장 시스템:
        - 계층적 패턴 구조에 저장
        - 패턴 관계 그래프 업데이트
        - 중복 패턴 검출 및 통합
        - 패턴 중요도 계산
        - 메타데이터 기반 인덱싱
        
        Args:
            pattern_vector: 패턴 벡터
            pattern_type: 패턴 타입 ('self', 'other', 'context')
            context: 상황 컨텍스트
            metadata: 추가 메타데이터
            
        Returns:
            패턴 ID
        """
        try:
            import uuid
            import time
            
            # 패턴 ID 생성
            pattern_id = str(uuid.uuid4())
            
            # 패턴 중복 검사
            duplicate_id = await self._check_pattern_duplicates(pattern_vector, pattern_type)
            if duplicate_id:
                # 중복 패턴 발견 시 기존 패턴 업데이트
                await self._update_existing_pattern(duplicate_id, pattern_vector, context, metadata)
                return duplicate_id
            
            # 패턴 중요도 계산
            importance_score = await self._calculate_pattern_importance(
                pattern_vector, pattern_type, context
            )
            
            # 패턴 데이터 구성
            pattern_data = {
                'id': pattern_id,
                'vector': pattern_vector,
                'type': pattern_type,
                'context': context,
                'metadata': metadata or {},
                'timestamp': time.time(),
                'importance': importance_score,
                'frequency': 1,
                'last_accessed': time.time(),
                'weight': 1.0,
                'stability': 0.0,
                'evolution_history': [],
                'relationship_scores': {}
            }
            
            # 패턴 데이터베이스에 저장
            if pattern_type == 'self':
                self.pattern_database['self_patterns'][pattern_id] = pattern_data
            elif pattern_type == 'other':
                self.pattern_database['other_patterns'][pattern_id] = pattern_data
            elif pattern_type == 'context':
                self.pattern_database['context_patterns'][pattern_id] = pattern_data
            
            # 계층적 구조에 패턴 추가
            await self._add_to_hierarchical_structure(pattern_id, pattern_data)
            
            # 패턴 관계 그래프 업데이트
            await self._update_pattern_relationships(pattern_id, pattern_data)
            
            # 점진적 학습 시스템에 패턴 추가
            await self._add_to_incremental_learner(pattern_id, pattern_data)
            
            # 메모리 관리
            await self._manage_pattern_memory()
            
            self.logger.debug(f"패턴 저장 완료: {pattern_id} (타입: {pattern_type})")
            
            return pattern_id
            
        except Exception as e:
            self.logger.error(f"패턴 저장 실패: {e}")
            return None
    
    async def _check_pattern_duplicates(self, pattern_vector: List[float], 
                                      pattern_type: str) -> str:
        """중복 패턴 검사"""
        try:
            # 해당 타입의 패턴들과 유사도 계산
            target_patterns = self.pattern_database.get(f'{pattern_type}_patterns', {})
            
            max_similarity = 0.0
            most_similar_id = None
            
            for pattern_id, pattern_data in target_patterns.items():
                stored_vector = pattern_data['vector']
                similarity = self._cosine_similarity(pattern_vector, stored_vector)
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_id = pattern_id
            
            # 중복 임계값 확인
            duplicate_threshold = 0.95
            if max_similarity > duplicate_threshold:
                return most_similar_id
            
            return None
            
        except Exception as e:
            self.logger.error(f"중복 패턴 검사 실패: {e}")
            return None
    
    async def _update_existing_pattern(self, pattern_id: str, 
                                     new_vector: List[float],
                                     context: Dict[str, Any],
                                     metadata: Dict[str, Any]):
        """기존 패턴 업데이트"""
        try:
            # 패턴 찾기
            pattern_data = None
            pattern_type = None
            
            for ptype in ['self_patterns', 'other_patterns', 'context_patterns']:
                if pattern_id in self.pattern_database[ptype]:
                    pattern_data = self.pattern_database[ptype][pattern_id]
                    pattern_type = ptype
                    break
            
            if not pattern_data:
                return
            
            # 패턴 벡터 업데이트 (지수 이동 평균)
            old_vector = pattern_data['vector']
            alpha = 0.1  # 학습률
            
            updated_vector = []
            for i in range(len(old_vector)):
                if i < len(new_vector):
                    updated_value = (1 - alpha) * old_vector[i] + alpha * new_vector[i]
                    updated_vector.append(updated_value)
                else:
                    updated_vector.append(old_vector[i])
            
            # 패턴 데이터 업데이트
            pattern_data['vector'] = updated_vector
            pattern_data['frequency'] += 1
            pattern_data['last_accessed'] = time.time()
            pattern_data['context'] = context  # 최신 컨텍스트로 업데이트
            
            # 메타데이터 업데이트
            if metadata:
                pattern_data['metadata'].update(metadata)
            
            # 안정성 점수 업데이트
            pattern_data['stability'] = self._calculate_pattern_stability(pattern_data)
            
            # 진화 기록 업데이트
            pattern_data['evolution_history'].append({
                'timestamp': time.time(),
                'change_magnitude': self._calculate_vector_change(old_vector, updated_vector),
                'context': context
            })
            
            self.logger.debug(f"기존 패턴 업데이트 완료: {pattern_id}")
            
        except Exception as e:
            self.logger.error(f"기존 패턴 업데이트 실패: {e}")
    
    async def _calculate_pattern_importance(self, pattern_vector: List[float],
                                          pattern_type: str,
                                          context: Dict[str, Any]) -> float:
        """패턴 중요도 계산"""
        try:
            importance = 0.0
            
            # 벡터 크기 기반 중요도
            vector_magnitude = np.linalg.norm(pattern_vector)
            magnitude_importance = min(vector_magnitude / 10.0, 1.0)
            
            # 컨텍스트 기반 중요도
            context_importance = 0.5
            if context.get('emotional_salience'):
                context_importance += 0.3
            if context.get('social_importance'):
                context_importance += 0.2
            
            # 패턴 타입 기반 중요도
            type_importance = {'self': 0.8, 'other': 0.6, 'context': 0.4}.get(pattern_type, 0.5)
            
            # 노벨티 기반 중요도
            novelty_score = self._detect_novelty({'neuron_activations': {'temp': {'activation_strength': sum(pattern_vector)}}}, context)
            novelty_importance = novelty_score * 0.5
            
            # 종합 중요도
            importance = (
                0.3 * magnitude_importance +
                0.3 * context_importance +
                0.2 * type_importance +
                0.2 * novelty_importance
            )
            
            return max(0.0, min(1.0, importance))
            
        except Exception as e:
            self.logger.error(f"패턴 중요도 계산 실패: {e}")
            return 0.5
    
    async def _add_to_hierarchical_structure(self, pattern_id: str, pattern_data: Dict[str, Any]):
        """계층적 구조에 패턴 추가"""
        try:
            # 패턴 클러스터링을 통해 계층적 위치 결정
            cluster_id = await self._find_pattern_cluster(pattern_data['vector'])
            
            # 계층적 구조 업데이트
            if cluster_id not in self.pattern_management['pattern_hierarchy']:
                self.pattern_management['pattern_hierarchy'][cluster_id] = {
                    'patterns': [],
                    'centroid': None,
                    'children': [],
                    'parent': None,
                    'depth': 0
                }
            
            self.pattern_management['pattern_hierarchy'][cluster_id]['patterns'].append(pattern_id)
            
            # 클러스터 중심 업데이트
            await self._update_cluster_centroid(cluster_id)
            
        except Exception as e:
            self.logger.error(f"계층적 구조 추가 실패: {e}")
    
    async def _find_pattern_cluster(self, pattern_vector: List[float]) -> str:
        """패턴 클러스터 찾기"""
        try:
            # 기존 클러스터들과 유사도 계산
            best_cluster = None
            max_similarity = 0.0
            
            for cluster_id, cluster_data in self.pattern_management['pattern_hierarchy'].items():
                if cluster_data['centroid']:
                    similarity = self._cosine_similarity(pattern_vector, cluster_data['centroid'])
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_cluster = cluster_id
            
            # 유사도 임계값 확인
            cluster_threshold = 0.7
            if max_similarity > cluster_threshold:
                return best_cluster
            
            # 새 클러스터 생성
            import uuid
            new_cluster_id = str(uuid.uuid4())
            return new_cluster_id
            
        except Exception as e:
            self.logger.error(f"패턴 클러스터 찾기 실패: {e}")
            return "default_cluster"
    
    async def _update_cluster_centroid(self, cluster_id: str):
        """클러스터 중심 업데이트"""
        try:
            cluster_data = self.pattern_management['pattern_hierarchy'][cluster_id]
            pattern_ids = cluster_data['patterns']
            
            if not pattern_ids:
                return
            
            # 클러스터 내 모든 패턴 벡터 수집
            vectors = []
            for pattern_id in pattern_ids:
                # 패턴 벡터 찾기
                pattern_vector = None
                for pattern_type in ['self_patterns', 'other_patterns', 'context_patterns']:
                    if pattern_id in self.pattern_database[pattern_type]:
                        pattern_vector = self.pattern_database[pattern_type][pattern_id]['vector']
                        break
                
                if pattern_vector:
                    vectors.append(pattern_vector)
            
            # 중심 계산
            if vectors:
                centroid = np.mean(vectors, axis=0).tolist()
                cluster_data['centroid'] = centroid
            
        except Exception as e:
            self.logger.error(f"클러스터 중심 업데이트 실패: {e}")
    
    async def _update_pattern_relationships(self, pattern_id: str, pattern_data: Dict[str, Any]):
        """패턴 관계 그래프 업데이트"""
        try:
            # 기존 패턴들과의 관계 계산
            pattern_vector = pattern_data['vector']
            pattern_type = pattern_data['type']
            
            # 같은 타입의 패턴들과 관계 계산
            target_patterns = self.pattern_database.get(f'{pattern_type}_patterns', {})
            
            for other_id, other_data in target_patterns.items():
                if other_id == pattern_id:
                    continue
                
                # 유사도 기반 관계
                similarity = self._cosine_similarity(pattern_vector, other_data['vector'])
                
                # 시간적 관계
                temporal_relation = self._calculate_temporal_relationship(
                    pattern_data['timestamp'], other_data['timestamp']
                )
                
                # 컨텍스트 관계
                context_relation = self._calculate_context_similarity(
                    pattern_data['context'], other_data['context']
                )
                
                # 관계 점수 계산
                relationship_score = (
                    0.5 * similarity +
                    0.3 * temporal_relation +
                    0.2 * context_relation
                )
                
                # 관계 저장
                if pattern_id not in self.pattern_management['pattern_relationships']:
                    self.pattern_management['pattern_relationships'][pattern_id] = {}
                
                self.pattern_management['pattern_relationships'][pattern_id][other_id] = {
                    'similarity': similarity,
                    'temporal': temporal_relation,
                    'context': context_relation,
                    'overall': relationship_score
                }
            
        except Exception as e:
            self.logger.error(f"패턴 관계 업데이트 실패: {e}")
    
    def _calculate_temporal_relationship(self, timestamp1: float, timestamp2: float) -> float:
        """시간적 관계 계산"""
        try:
            time_diff = abs(timestamp1 - timestamp2)
            
            # 시간 차이를 관계 점수로 변환 (가까울수록 높은 점수)
            # 1시간 = 3600초를 기준으로 정규화
            relation_score = np.exp(-time_diff / 3600.0)
            
            return max(0.0, min(1.0, relation_score))
            
        except Exception:
            return 0.0
    
    def _calculate_pattern_stability(self, pattern_data: Dict[str, Any]) -> float:
        """패턴 안정성 계산"""
        try:
            evolution_history = pattern_data.get('evolution_history', [])
            
            if len(evolution_history) < 2:
                return 0.5  # 기본 안정성
            
            # 변화 크기들의 분산 계산
            changes = [entry['change_magnitude'] for entry in evolution_history]
            change_variance = np.var(changes)
            
            # 분산이 낮을수록 안정성 높음
            stability = 1.0 / (1.0 + change_variance)
            
            return max(0.0, min(1.0, stability))
            
        except Exception:
            return 0.5
    
    def _calculate_vector_change(self, old_vector: List[float], new_vector: List[float]) -> float:
        """벡터 변화 크기 계산"""
        try:
            if len(old_vector) != len(new_vector):
                return 1.0  # 크기가 다르면 큰 변화로 간주
            
            # 유클리드 거리 계산
            distance = np.sqrt(sum((a - b)**2 for a, b in zip(old_vector, new_vector)))
            
            # 정규화
            max_distance = np.sqrt(len(old_vector))
            normalized_distance = distance / max_distance
            
            return max(0.0, min(1.0, normalized_distance))
            
        except Exception:
            return 0.0
    
    async def _add_to_incremental_learner(self, pattern_id: str, pattern_data: Dict[str, Any]):
        """점진적 학습기에 패턴 추가"""
        try:
            # 학습 데이터 형태로 변환
            feature_vector = pattern_data['vector']
            label = 1 if pattern_data['type'] == 'self' else 0
            
            # 학습 데이터에 추가
            training_sample = {
                'pattern_id': pattern_id,
                'features': feature_vector,
                'label': label,
                'importance': pattern_data['importance'],
                'timestamp': pattern_data['timestamp']
            }
            
            self.self_other_classifier['training_data'].append(training_sample)
            
            # 학습 데이터 크기 제한
            max_training_size = 10000
            if len(self.self_other_classifier['training_data']) > max_training_size:
                # 중요도 기반 샘플링
                self.self_other_classifier['training_data'] = self._sample_training_data(
                    self.self_other_classifier['training_data'], max_training_size
                )
            
        except Exception as e:
            self.logger.error(f"점진적 학습기 추가 실패: {e}")
    
    def _sample_training_data(self, training_data: List[Dict], target_size: int) -> List[Dict]:
        """중요도 기반 훈련 데이터 샘플링"""
        try:
            # 중요도에 따라 정렬
            sorted_data = sorted(training_data, key=lambda x: x['importance'], reverse=True)
            
            # 상위 target_size개만 선택
            return sorted_data[:target_size]
            
        except Exception as e:
            self.logger.error(f"훈련 데이터 샘플링 실패: {e}")
            return training_data[:target_size]
    
    async def _manage_pattern_memory(self):
        """패턴 메모리 관리"""
        try:
            # 메모리 사용량 확인
            total_patterns = (
                len(self.pattern_database['self_patterns']) +
                len(self.pattern_database['other_patterns']) +
                len(self.pattern_database['context_patterns'])
            )
            
            max_patterns = 50000  # 최대 패턴 수
            
            if total_patterns > max_patterns:
                # 선택적 망각 실행
                await self._execute_selective_forgetting()
            
        except Exception as e:
            self.logger.error(f"패턴 메모리 관리 실패: {e}")
    
    async def _execute_selective_forgetting(self):
        """선택적 망각 실행"""
        try:
            # 각 패턴 타입별로 망각 실행
            for pattern_type in ['self_patterns', 'other_patterns', 'context_patterns']:
                patterns = self.pattern_database[pattern_type]
                
                if len(patterns) > 10000:  # 타입별 최대 크기
                    # 중요도 기반 정렬
                    sorted_patterns = sorted(
                        patterns.items(),
                        key=lambda x: self._calculate_forgetting_score(x[1]),
                        reverse=True
                    )
                    
                    # 상위 80%만 유지
                    keep_count = int(len(sorted_patterns) * 0.8)
                    patterns_to_keep = dict(sorted_patterns[:keep_count])
                    
                    # 패턴 데이터베이스 업데이트
                    self.pattern_database[pattern_type] = patterns_to_keep
                    
                    self.logger.info(f"{pattern_type}에서 {len(patterns) - keep_count}개 패턴 망각")
            
        except Exception as e:
            self.logger.error(f"선택적 망각 실행 실패: {e}")
    
    def _calculate_forgetting_score(self, pattern_data: Dict[str, Any]) -> float:
        """망각 점수 계산 (높을수록 유지)"""
        try:
            # 중요도
            importance = pattern_data.get('importance', 0.0)
            
            # 빈도
            frequency = pattern_data.get('frequency', 1)
            frequency_score = min(frequency / 100.0, 1.0)
            
            # 최신성
            import time
            current_time = time.time()
            last_accessed = pattern_data.get('last_accessed', current_time)
            time_diff = current_time - last_accessed
            recency_score = np.exp(-time_diff / 86400.0)  # 1일 기준
            
            # 안정성
            stability = pattern_data.get('stability', 0.5)
            
            # 종합 점수
            forgetting_score = (
                0.4 * importance +
                0.3 * frequency_score +
                0.2 * recency_score +
                0.1 * stability
            )
            
            return max(0.0, min(1.0, forgetting_score))
            
        except Exception:
            return 0.0


async def test_hierarchical_emotion_system():
    """계층적 감정 시스템 테스트"""
    system = AdvancedHierarchicalEmotionSystem()
    
    # 테스트용 문학 데이터 시퀀스
    test_literary_sequence = [
        {
            'character_emotion': {
                'valence': -0.8,
                'arousal': 0.7,
                'dominance': -0.5,
                'source': 'hamlet'
            },
            'reader_emotion': {
                'valence': -0.5,
                'arousal': 0.5,
                'dominance': 0.0
            },
            'context': {
                'situation_type': 'tragedy',
                'cultural_context': 'western'
            },
            'genre': 'tragedy'
        },
        {
            'character_emotion': {
                'valence': 0.3,
                'arousal': 0.4,
                'dominance': -0.2,
                'source': 'chunhyang'
            },
            'reader_emotion': {
                'valence': 0.5,
                'arousal': 0.3,
                'dominance': 0.1
            },
            'community_emotions': {
                'reader1': {'valence': 0.5, 'arousal': 0.3},
                'reader2': {'valence': 0.4, 'arousal': 0.4},
                'reader3': {'valence': 0.6, 'arousal': 0.2}
            },
            'context': {
                'situation_type': 'romance',
                'cultural_context': 'korean_traditional'
            },
            'cultural_context': 'korean_traditional',
            'genre': 'romance'
        }
    ]
    
    # 시스템 실행
    results = await system.process_literary_emotion_sequence(
        test_literary_sequence,
        time_series_mode=True
    )
    
    # 결과 출력
    logger.info("=== 계층적 감정 시스템 테스트 결과 ===")
    logger.info(f"학습 요약: {json.dumps(results['learning_summary'], indent=2)}")
    
    # 모델 저장
    await system.save_models()
    
    return results


class SURDUncertaintyPropagator:
    """
    SURD 불확실성 전파 시스템
    
    2024년 연구 기반 구현:
    - Causal Information Theory 기반 불확실성 정량화
    - Hierarchical Uncertainty Propagation
    - Empathy-Uncertainty Integration
    - Dynamic Confidence Adjustment
    """
    
    def __init__(self, surd_analyzer: AdvancedSURDAnalyzer = None):
        self.surd_analyzer = surd_analyzer
        self.uncertainty_cache = {}
        self.propagation_history = []
        self.logger = logging.getLogger(__name__)
        
        # 불확실성 전파 파라미터
        self.propagation_params = {
            'causal_threshold': 0.3,
            'uncertainty_decay': 0.9,
            'confidence_amplification': 1.2,
            'temporal_decay': 0.95,
            'network_depth': 5
        }
    
    async def propagate_empathy_uncertainty(self, 
                                          empathy_results: Dict[str, Any],
                                          context: Dict[str, Any]) -> Dict[str, Any]:
        """
        공감 결과에서 불확실성 전파
        
        Args:
            empathy_results: 공감 분석 결과
            context: 상황 컨텍스트
            
        Returns:
            불확실성이 전파된 결과
        """
        try:
            if not self.surd_analyzer:
                return empathy_results
            
            # 1. 입력 데이터 준비
            input_variables = await self._prepare_empathy_variables(empathy_results, context)
            
            # 2. SURD 분석 수행
            surd_result = await self._perform_surd_analysis(input_variables)
            
            # 3. 불확실성 맵 생성
            uncertainty_map = await self._generate_uncertainty_map(surd_result)
            
            # 4. 계층적 불확실성 전파
            propagated_uncertainty = await self._propagate_hierarchical_uncertainty(
                uncertainty_map, empathy_results
            )
            
            # 5. 신뢰도 조정
            adjusted_results = await self._adjust_confidence_levels(
                empathy_results, propagated_uncertainty
            )
            
            # 6. 불확실성 메타데이터 추가
            final_results = await self._add_uncertainty_metadata(
                adjusted_results, surd_result, propagated_uncertainty
            )
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"불확실성 전파 실패: {e}")
            return empathy_results
    
    async def _prepare_empathy_variables(self, 
                                       empathy_results: Dict[str, Any],
                                       context: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """공감 분석 결과를 SURD 분석용 변수로 변환"""
        variables = {}
        
        # 자기 성찰 변수들
        if 'self_reflection' in empathy_results:
            self_data = empathy_results['self_reflection']
            variables['self_confidence'] = np.array([self_data.get('self_confidence', 0.5)])
            variables['introspection_depth'] = np.array([self_data.get('introspection_depth', 0.5)])
            variables['emotional_self_awareness'] = np.array([self_data.get('emotional_self_awareness', 0.5)])
        
        # 공감 시뮬레이션 변수들
        if 'empathy_simulation' in empathy_results:
            empathy_data = empathy_results['empathy_simulation']
            variables['theory_of_mind_score'] = np.array([empathy_data.get('theory_of_mind_score', 0.5)])
            variables['empathy_intensity'] = np.array([empathy_data.get('empathy_intensity', 0.5)])
            variables['perspective_taking'] = np.array([empathy_data.get('perspective_taking', 0.5)])
        
        # 공동체 맥락 변수들
        if 'community_context' in empathy_results:
            community_data = empathy_results['community_context']
            variables['social_cohesion'] = np.array([community_data.get('social_cohesion_score', 0.5)])
            variables['cultural_alignment'] = np.array([community_data.get('cultural_sensitivity', 0.5)])
        
        # 거울 뉴런 변수들
        if 'mirror_neuron' in empathy_results:
            mirror_data = empathy_results['mirror_neuron']
            variables['mirror_fidelity'] = np.array([mirror_data.get('mirror_fidelity', 0.5)])
            variables['neural_resonance'] = np.array([mirror_data.get('neural_resonance', 0.5)])
        
        # 컨텍스트 변수들
        variables['context_complexity'] = np.array([len(str(context)) / 1000.0])  # 단순화된 복잡도
        variables['temporal_consistency'] = np.array([context.get('temporal_consistency', 0.5)])
        
        return variables
    
    async def _perform_surd_analysis(self, variables: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """SURD 분석 수행"""
        try:
            # 변수들을 하나의 데이터셋으로 결합
            combined_data = np.column_stack(list(variables.values()))
            variable_names = list(variables.keys())
            
            # SURD 분석 수행
            surd_result = await self.surd_analyzer.analyze_causal_relationships(
                data=combined_data,
                variable_names=variable_names,
                target_variable='theory_of_mind_score' if 'theory_of_mind_score' in variables else variable_names[0]
            )
            
            return surd_result
            
        except Exception as e:
            self.logger.error(f"SURD 분석 실패: {e}")
            return {}
    
    async def _generate_uncertainty_map(self, surd_result: Dict[str, Any]) -> Dict[str, float]:
        """SURD 결과에서 불확실성 맵 생성"""
        uncertainty_map = {}
        
        if 'information_measures' in surd_result:
            info_measures = surd_result['information_measures']
            
            # 상호정보량 기반 불확실성
            if hasattr(info_measures, 'mutual_information'):
                uncertainty_map['mutual_information_uncertainty'] = 1.0 - info_measures.mutual_information
            
            # 전이 엔트로피 기반 불확실성
            if hasattr(info_measures, 'transfer_entropy'):
                uncertainty_map['causal_uncertainty'] = 1.0 - info_measures.transfer_entropy
            
            # 부분 정보 분해 기반 불확실성
            if hasattr(info_measures, 'partial_information_decomposition'):
                pid = info_measures.partial_information_decomposition
                uncertainty_map['synergy_uncertainty'] = 1.0 - pid.get('synergy', 0.5)
                uncertainty_map['redundancy_uncertainty'] = 1.0 - pid.get('redundancy', 0.5)
        
        # 인과 네트워크 기반 불확실성
        if 'causal_network' in surd_result:
            network = surd_result['causal_network']
            uncertainty_map['network_uncertainty'] = await self._calculate_network_uncertainty(network)
        
        return uncertainty_map
    
    async def _calculate_network_uncertainty(self, network: Dict[str, Any]) -> float:
        """인과 네트워크에서 불확실성 계산"""
        try:
            if 'edges' not in network:
                return 0.5
            
            edges = network['edges']
            if not edges:
                return 0.5
            
            # 엣지 강도의 분산을 불확실성으로 사용
            edge_strengths = [edge.get('strength', 0.5) for edge in edges]
            uncertainty = np.std(edge_strengths) if len(edge_strengths) > 1 else 0.5
            
            return min(1.0, uncertainty)
            
        except Exception:
            return 0.5
    
    async def _propagate_hierarchical_uncertainty(self, 
                                                uncertainty_map: Dict[str, float],
                                                empathy_results: Dict[str, Any]) -> Dict[str, float]:
        """계층적 불확실성 전파"""
        propagated = {}
        
        # 기본 불확실성 수준
        base_uncertainty = np.mean(list(uncertainty_map.values())) if uncertainty_map else 0.5
        
        # 각 공감 컴포넌트별 불확실성 전파
        for component, result in empathy_results.items():
            if isinstance(result, dict) and 'confidence' in result:
                # 기존 신뢰도와 불확실성 결합
                original_confidence = result['confidence']
                uncertainty_impact = base_uncertainty * self.propagation_params['uncertainty_decay']
                
                # 불확실성 전파 공식: 신뢰도 = 원래신뢰도 * (1 - 불확실성영향)
                propagated_confidence = original_confidence * (1 - uncertainty_impact)
                propagated[f'{component}_uncertainty'] = 1.0 - propagated_confidence
            else:
                propagated[f'{component}_uncertainty'] = base_uncertainty
        
        return propagated
    
    async def _adjust_confidence_levels(self, 
                                      empathy_results: Dict[str, Any],
                                      propagated_uncertainty: Dict[str, float]) -> Dict[str, Any]:
        """불확실성에 기반한 신뢰도 조정"""
        adjusted_results = empathy_results.copy()
        
        for component, result in adjusted_results.items():
            if isinstance(result, dict):
                uncertainty_key = f'{component}_uncertainty'
                if uncertainty_key in propagated_uncertainty:
                    uncertainty = propagated_uncertainty[uncertainty_key]
                    
                    # 기존 신뢰도 조정
                    if 'confidence' in result:
                        adjusted_confidence = result['confidence'] * (1 - uncertainty * 0.5)
                        result['confidence'] = max(0.0, min(1.0, adjusted_confidence))
                    
                    # 불확실성 메타데이터 추가
                    result['uncertainty_level'] = uncertainty
                    result['uncertainty_adjusted'] = True
        
        return adjusted_results
    
    async def _add_uncertainty_metadata(self, 
                                      results: Dict[str, Any],
                                      surd_result: Dict[str, Any],
                                      propagated_uncertainty: Dict[str, float]) -> Dict[str, Any]:
        """불확실성 메타데이터 추가"""
        results['uncertainty_analysis'] = {
            'surd_analysis_available': bool(surd_result),
            'uncertainty_propagation_applied': True,
            'propagated_uncertainties': propagated_uncertainty,
            'overall_uncertainty': np.mean(list(propagated_uncertainty.values())) if propagated_uncertainty else 0.5,
            'uncertainty_sources': list(propagated_uncertainty.keys()),
            'causal_information_available': 'causal_network' in surd_result
        }
        
        return results


# EnhancedEmpathyLearner에 SURD 통합 메소드 추가
def _add_surd_integration_to_empathy_learner():
    """EnhancedEmpathyLearner 클래스에 SURD 통합 메소드 추가"""
    
    async def analyze_empathy_with_uncertainty(self, text: str, context: Dict = None) -> HierarchicalEmpathyResult:
        """
        불확실성 전파가 통합된 공감 분석
        
        Args:
            text: 분석할 텍스트
            context: 상황 컨텍스트
            
        Returns:
            불확실성이 포함된 공감 분석 결과
        """
        context = context or {}
        
        try:
            # 1. 기본 공감 분석 수행
            base_result = await self.analyze_empathy(text, context)
            
            # 2. SURD 불확실성 전파 적용
            if self.uncertainty_propagation_enabled and self.surd_analyzer:
                uncertainty_propagator = SURDUncertaintyPropagator(self.surd_analyzer)
                
                # 공감 결과를 딕셔너리 형태로 변환
                empathy_dict = {
                    'self_reflection': {
                        'self_confidence': base_result.self_confidence,
                        'introspection_depth': getattr(base_result, 'introspection_depth', 0.5),
                        'emotional_self_awareness': getattr(base_result, 'emotional_self_awareness', 0.5)
                    },
                    'empathy_simulation': {
                        'theory_of_mind_score': base_result.theory_of_mind_score,
                        'empathy_intensity': getattr(base_result, 'empathy_intensity', 0.5),
                        'perspective_taking': getattr(base_result, 'perspective_taking', 0.5)
                    },
                    'community_context': {
                        'social_cohesion_score': base_result.social_cohesion_score,
                        'cultural_sensitivity': getattr(base_result, 'cultural_sensitivity', 0.5)
                    },
                    'mirror_neuron': {
                        'mirror_fidelity': base_result.mirror_fidelity,
                        'neural_resonance': getattr(base_result, 'neural_resonance', 0.5)
                    }
                }
                
                # 불확실성 전파 수행
                uncertainty_adjusted = await uncertainty_propagator.propagate_empathy_uncertainty(
                    empathy_dict, context
                )
                
                # 결과를 다시 HierarchicalEmpathyResult로 변환
                adjusted_result = self._convert_to_hierarchical_result(uncertainty_adjusted, base_result)
                return adjusted_result
            
            return base_result
            
        except Exception as e:
            self.logger.error(f"불확실성 통합 공감 분석 실패: {e}")
            return await self.analyze_empathy(text, context)
    
    def _convert_to_hierarchical_result(self, uncertainty_dict: Dict, base_result: HierarchicalEmpathyResult) -> HierarchicalEmpathyResult:
        """불확실성 조정된 딕셔너리를 HierarchicalEmpathyResult로 변환"""
        try:
            # 기본 결과 복사
            result = HierarchicalEmpathyResult(
                self_confidence=uncertainty_dict.get('self_reflection', {}).get('confidence', base_result.self_confidence),
                simulation_confidence=uncertainty_dict.get('empathy_simulation', {}).get('confidence', base_result.simulation_confidence),
                social_cohesion_score=uncertainty_dict.get('community_context', {}).get('confidence', base_result.social_cohesion_score),
                mirror_fidelity=uncertainty_dict.get('mirror_neuron', {}).get('confidence', base_result.mirror_fidelity),
                integrated_empathy_score=base_result.integrated_empathy_score,
                theory_of_mind_score=uncertainty_dict.get('empathy_simulation', {}).get('theory_of_mind_score', base_result.theory_of_mind_score),
                empathy_balance_score=base_result.empathy_balance_score,
                utilitarian_compatibility=base_result.utilitarian_compatibility,
                self_other_balance=base_result.self_other_balance,
                individual_community_balance=base_result.individual_community_balance,
                confidence_score=base_result.confidence_score,
                processing_metadata=base_result.processing_metadata
            )
            
            # 불확실성 메타데이터 추가
            if 'uncertainty_analysis' in uncertainty_dict:
                result.processing_metadata['uncertainty_analysis'] = uncertainty_dict['uncertainty_analysis']
            
            return result
            
        except Exception as e:
            self.logger.error(f"결과 변환 실패: {e}")
            return base_result
    
    # 메소드를 EnhancedEmpathyLearner 클래스에 추가
    EnhancedEmpathyLearner.analyze_empathy_with_uncertainty = analyze_empathy_with_uncertainty
    EnhancedEmpathyLearner._convert_to_hierarchical_result = _convert_to_hierarchical_result

# SURD 통합 적용
_add_surd_integration_to_empathy_learner()


class AdvancedCacheSystem:
    """
    고급 캐싱 시스템
    
    2024년 연구 기반 구현:
    - LRU + TTL 하이브리드 캐싱
    - 의미론적 유사성 기반 캐시 히트
    - 계층적 캐시 구조
    - 동적 캐시 크기 조정
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 캐시 설정
        self.cache_config = {
            'max_cache_size': self.config.get('max_cache_size', 10000),
            'ttl_seconds': self.config.get('ttl_seconds', 3600),  # 1시간
            'similarity_threshold': self.config.get('similarity_threshold', 0.85),
            'cleanup_interval': self.config.get('cleanup_interval', 300),  # 5분
            'memory_threshold': self.config.get('memory_threshold', 0.8)  # 80%
        }
        
        # 다층 캐시 구조
        self.caches = {
            'empathy_results': {},  # 공감 분석 결과 캐시
            'mirror_neuron_patterns': {},  # 거울 뉴런 패턴 캐시
            'surd_analysis': {},  # SURD 분석 결과 캐시
            'semantic_embeddings': {},  # 의미 임베딩 캐시
            'context_templates': {}  # 컨텍스트 템플릿 캐시
        }
        
        # 캐시 메타데이터
        self.cache_metadata = {
            'access_counts': defaultdict(int),
            'last_access': {},
            'hit_rates': defaultdict(float),
            'creation_times': {},
            'sizes': defaultdict(int)
        }
        
        # 의미론적 유사성 계산을 위한 임베딩 모델
        self.embedding_model = None
        self._initialize_embedding_model()
        
        # 정리 스케줄러 (비동기 초기화에서 시작됨)
        self.cleanup_scheduler = None
    
    def _initialize_embedding_model(self):
        """임베딩 모델 초기화"""
        try:
            if ADVANCED_MODELS_AVAILABLE:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
            else:
                self.embedding_model = None
        except Exception as e:
            self.logger.warning(f"임베딩 모델 로딩 실패: {e}")
            self.embedding_model = None
    
    async def initialize(self):
        """비동기 초기화 - 기존 최적화 시스템들과 일관된 패턴"""
        if self.cleanup_scheduler is None:
            await self._start_cleanup_scheduler()
    
    async def _start_cleanup_scheduler(self):
        """주기적 캐시 정리 스케줄러 시작"""
        async def cleanup_task():
            while True:
                await asyncio.sleep(self.cache_config['cleanup_interval'])
                await self._cleanup_expired_entries()
        
        self.cleanup_scheduler = asyncio.create_task(cleanup_task())
    
    async def get(self, cache_type: str, key: str) -> Optional[Any]:
        """캐시에서 값 가져오기 (의미론적 유사성 포함)"""
        try:
            if cache_type not in self.caches:
                return None
            
            cache = self.caches[cache_type]
            
            # 정확한 키 매치 시도
            if key in cache:
                entry = cache[key]
                if not self._is_expired(entry):
                    self._update_access_metadata(cache_type, key)
                    return entry['value']
                else:
                    del cache[key]
            
            # 의미론적 유사성 기반 검색
            similar_key = await self._find_similar_key(cache_type, key)
            if similar_key:
                entry = cache[similar_key]
                if not self._is_expired(entry):
                    self._update_access_metadata(cache_type, similar_key)
                    return entry['value']
            
            return None
            
        except Exception as e:
            self.logger.error(f"캐시 조회 실패: {e}")
            return None
    
    async def set(self, cache_type: str, key: str, value: Any) -> bool:
        """캐시에 값 저장"""
        try:
            if cache_type not in self.caches:
                self.caches[cache_type] = {}
            
            cache = self.caches[cache_type]
            
            # 캐시 크기 제한 확인
            if len(cache) >= self.cache_config['max_cache_size']:
                await self._evict_least_used(cache_type)
            
            # 메모리 사용량 확인
            if await self._check_memory_usage():
                await self._cleanup_by_memory_pressure()
            
            # 엔트리 생성
            entry = {
                'value': value,
                'created_at': time.time(),
                'ttl': self.cache_config['ttl_seconds'],
                'access_count': 0,
                'last_accessed': time.time()
            }
            
            cache[key] = entry
            self._update_creation_metadata(cache_type, key, entry)
            
            return True
            
        except Exception as e:
            self.logger.error(f"캐시 저장 실패: {e}")
            return False
    
    async def _find_similar_key(self, cache_type: str, target_key: str) -> Optional[str]:
        """의미론적 유사성 기반 키 검색"""
        try:
            if not self.embedding_model or cache_type not in self.caches:
                return None
            
            cache = self.caches[cache_type]
            if not cache:
                return None
            
            # 타겟 키 임베딩
            target_embedding = self.embedding_model.encode([target_key])[0]
            
            best_similarity = 0.0
            best_key = None
            
            for key in cache.keys():
                # 키 임베딩 (캐싱된 임베딩 사용)
                if key in self.cache_metadata.get('embeddings', {}):
                    key_embedding = self.cache_metadata['embeddings'][key]
                else:
                    key_embedding = self.embedding_model.encode([key])[0]
                    if 'embeddings' not in self.cache_metadata:
                        self.cache_metadata['embeddings'] = {}
                    self.cache_metadata['embeddings'][key] = key_embedding
                
                # 코사인 유사도 계산
                similarity = np.dot(target_embedding, key_embedding) / (
                    np.linalg.norm(target_embedding) * np.linalg.norm(key_embedding)
                )
                
                if similarity > best_similarity and similarity >= self.cache_config['similarity_threshold']:
                    best_similarity = similarity
                    best_key = key
            
            return best_key
            
        except Exception as e:
            self.logger.error(f"유사 키 검색 실패: {e}")
            return None
    
    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """캐시 엔트리 만료 확인"""
        return time.time() - entry['created_at'] > entry['ttl']
    
    def _update_access_metadata(self, cache_type: str, key: str):
        """접근 메타데이터 업데이트"""
        self.cache_metadata['access_counts'][f"{cache_type}:{key}"] += 1
        self.cache_metadata['last_access'][f"{cache_type}:{key}"] = time.time()
    
    def _update_creation_metadata(self, cache_type: str, key: str, entry: Dict[str, Any]):
        """생성 메타데이터 업데이트"""
        cache_key = f"{cache_type}:{key}"
        self.cache_metadata['creation_times'][cache_key] = entry['created_at']
        self.cache_metadata['sizes'][cache_key] = len(str(entry['value']))
    
    async def _evict_least_used(self, cache_type: str):
        """LRU 기반 캐시 제거"""
        cache = self.caches[cache_type]
        if not cache:
            return
        
        # 가장 적게 사용된 키 찾기
        min_access_count = float('inf')
        least_used_key = None
        
        for key in cache.keys():
            cache_key = f"{cache_type}:{key}"
            access_count = self.cache_metadata['access_counts'].get(cache_key, 0)
            if access_count < min_access_count:
                min_access_count = access_count
                least_used_key = key
        
        if least_used_key:
            del cache[least_used_key]
            cache_key = f"{cache_type}:{least_used_key}"
            self._cleanup_metadata(cache_key)
    
    async def _cleanup_expired_entries(self):
        """만료된 엔트리 정리"""
        for cache_type, cache in self.caches.items():
            expired_keys = []
            for key, entry in cache.items():
                if self._is_expired(entry):
                    expired_keys.append(key)
            
            for key in expired_keys:
                del cache[key]
                cache_key = f"{cache_type}:{key}"
                self._cleanup_metadata(cache_key)
    
    async def _check_memory_usage(self) -> bool:
        """메모리 사용량 확인"""
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent / 100.0
            return memory_percent > self.cache_config['memory_threshold']
        except ImportError:
            return False
    
    async def _cleanup_by_memory_pressure(self):
        """메모리 압박 시 캐시 정리"""
        # 가장 큰 캐시부터 절반으로 줄이기
        for cache_type in sorted(self.caches.keys(), 
                               key=lambda ct: len(self.caches[ct]), 
                               reverse=True):
            cache = self.caches[cache_type]
            target_size = len(cache) // 2
            
            # 오래된 엔트리부터 제거
            sorted_keys = sorted(cache.keys(), 
                               key=lambda k: cache[k]['last_accessed'])
            
            for key in sorted_keys[:len(sorted_keys) - target_size]:
                del cache[key]
                cache_key = f"{cache_type}:{key}"
                self._cleanup_metadata(cache_key)
    
    def _cleanup_metadata(self, cache_key: str):
        """메타데이터 정리"""
        for metadata_dict in [self.cache_metadata['access_counts'],
                            self.cache_metadata['last_access'],
                            self.cache_metadata['creation_times'],
                            self.cache_metadata['sizes']]:
            metadata_dict.pop(cache_key, None)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        stats = {}
        for cache_type, cache in self.caches.items():
            stats[cache_type] = {
                'size': len(cache),
                'total_size_bytes': sum(self.cache_metadata['sizes'].get(f"{cache_type}:{k}", 0) 
                                      for k in cache.keys()),
                'hit_rate': self.cache_metadata['hit_rates'].get(cache_type, 0.0)
            }
        
        return stats


class PerformanceOptimizer:
    """
    성능 최적화 시스템
    
    2024년 연구 기반 구현:
    - 동적 GPU 메모리 관리
    - 배치 처리 최적화
    - 모델 파이프라이닝
    - 비동기 처리 최적화
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 최적화 설정
        self.optimization_config = {
            'batch_size_adaptive': True,
            'gpu_memory_fraction': 0.8,
            'model_pipeline_enabled': True,
            'async_processing_enabled': True,
            'performance_monitoring': True
        }
        
        # 성능 메트릭
        self.performance_metrics = {
            'processing_times': deque(maxlen=1000),
            'memory_usage': deque(maxlen=1000),
            'gpu_utilization': deque(maxlen=1000),
            'batch_sizes': deque(maxlen=1000),
            'cache_hit_rates': deque(maxlen=1000)
        }
        
        # 동적 배치 크기 조정
        self.adaptive_batch_size = {
            'current_size': 8,
            'min_size': 1,
            'max_size': 64,
            'adjustment_factor': 1.2,
            'performance_threshold': 0.1
        }
        
        # 모델 파이프라인
        self.model_pipeline = {
            'stages': [],
            'stage_times': {},
            'bottleneck_detection': True
        }
    
    async def optimize_batch_processing(self, data_batch: List[Any], 
                                      processing_func: callable) -> List[Any]:
        """배치 처리 최적화"""
        try:
            start_time = time.time()
            
            # 동적 배치 크기 조정
            optimal_batch_size = await self._calculate_optimal_batch_size(len(data_batch))
            
            results = []
            for i in range(0, len(data_batch), optimal_batch_size):
                batch = data_batch[i:i + optimal_batch_size]
                
                # GPU 메모리 확인
                if await self._check_gpu_memory():
                    batch_results = await self._process_batch_gpu(batch, processing_func)
                else:
                    batch_results = await self._process_batch_cpu(batch, processing_func)
                
                results.extend(batch_results)
            
            # 성능 메트릭 업데이트
            processing_time = time.time() - start_time
            await self._update_performance_metrics(processing_time, len(data_batch))
            
            return results
            
        except Exception as e:
            self.logger.error(f"배치 처리 최적화 실패: {e}")
            return await self._fallback_processing(data_batch, processing_func)
    
    async def _calculate_optimal_batch_size(self, total_items: int) -> int:
        """최적 배치 크기 계산"""
        try:
            # 최근 성능 히스토리 분석
            recent_times = list(self.performance_metrics['processing_times'])[-10:]
            recent_sizes = list(self.performance_metrics['batch_sizes'])[-10:]
            
            if len(recent_times) < 3:
                return self.adaptive_batch_size['current_size']
            
            # 처리 시간 vs 배치 크기 상관관계 분석
            avg_time_per_item = []
            for i in range(len(recent_times)):
                if recent_sizes[i] > 0:
                    avg_time_per_item.append(recent_times[i] / recent_sizes[i])
            
            if not avg_time_per_item:
                return self.adaptive_batch_size['current_size']
            
            # 성능 개선 여부 확인
            recent_avg = np.mean(avg_time_per_item[-3:])
            older_avg = np.mean(avg_time_per_item[:-3]) if len(avg_time_per_item) > 3 else recent_avg
            
            # 배치 크기 조정
            current_size = self.adaptive_batch_size['current_size']
            if recent_avg < older_avg:  # 성능 개선
                new_size = min(current_size * self.adaptive_batch_size['adjustment_factor'],
                             self.adaptive_batch_size['max_size'])
            else:  # 성능 저하
                new_size = max(current_size / self.adaptive_batch_size['adjustment_factor'],
                             self.adaptive_batch_size['min_size'])
            
            self.adaptive_batch_size['current_size'] = int(new_size)
            return min(int(new_size), total_items)
            
        except Exception as e:
            self.logger.error(f"최적 배치 크기 계산 실패: {e}")
            return self.adaptive_batch_size['current_size']
    
    async def _check_gpu_memory(self) -> bool:
        """GPU 메모리 가용성 확인"""
        try:
            if not torch.cuda.is_available():
                return False
            
            memory_info = torch.cuda.memory_stats()
            allocated = memory_info.get('allocated_bytes.all.current', 0)
            reserved = memory_info.get('reserved_bytes.all.current', 0)
            max_memory = torch.cuda.get_device_properties(0).total_memory
            
            usage_ratio = (allocated + reserved) / max_memory
            return usage_ratio < self.optimization_config['gpu_memory_fraction']
            
        except Exception:
            return False
    
    async def _process_batch_gpu(self, batch: List[Any], 
                               processing_func: callable) -> List[Any]:
        """GPU에서 배치 처리"""
        try:
            # GPU 컨텍스트에서 처리
            with torch.cuda.device(0):
                if asyncio.iscoroutinefunction(processing_func):
                    tasks = [processing_func(item) for item in batch]
                    results = await asyncio.gather(*tasks)
                else:
                    results = [processing_func(item) for item in batch]
                
                # GPU 메모리 정리
                torch.cuda.empty_cache()
                
                return results
                
        except Exception as e:
            self.logger.warning(f"GPU 배치 처리 실패, CPU로 폴백: {e}")
            return await self._process_batch_cpu(batch, processing_func)
    
    async def _process_batch_cpu(self, batch: List[Any], 
                               processing_func: callable) -> List[Any]:
        """CPU에서 배치 처리"""
        try:
            if asyncio.iscoroutinefunction(processing_func):
                # 비동기 처리
                semaphore = asyncio.Semaphore(4)  # 동시 처리 제한
                
                async def bounded_process(item):
                    async with semaphore:
                        return await processing_func(item)
                
                tasks = [bounded_process(item) for item in batch]
                results = await asyncio.gather(*tasks)
            else:
                # 동기 처리
                results = [processing_func(item) for item in batch]
            
            return results
            
        except Exception as e:
            self.logger.error(f"CPU 배치 처리 실패: {e}")
            return []
    
    async def _fallback_processing(self, data_batch: List[Any], 
                                 processing_func: callable) -> List[Any]:
        """폴백 처리"""
        try:
            results = []
            for item in data_batch:
                if asyncio.iscoroutinefunction(processing_func):
                    result = await processing_func(item)
                else:
                    result = processing_func(item)
                results.append(result)
            return results
        except Exception as e:
            self.logger.error(f"폴백 처리 실패: {e}")
            return []
    
    async def _update_performance_metrics(self, processing_time: float, batch_size: int):
        """성능 메트릭 업데이트"""
        self.performance_metrics['processing_times'].append(processing_time)
        self.performance_metrics['batch_sizes'].append(batch_size)
        
        # GPU 사용량 모니터링
        if torch.cuda.is_available():
            gpu_usage = torch.cuda.utilization()
            self.performance_metrics['gpu_utilization'].append(gpu_usage)
        
        # 메모리 사용량 모니터링
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent
            self.performance_metrics['memory_usage'].append(memory_percent)
        except ImportError:
            pass
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        stats = {}
        
        for metric_name, metric_data in self.performance_metrics.items():
            if metric_data:
                stats[metric_name] = {
                    'mean': np.mean(metric_data),
                    'std': np.std(metric_data),
                    'min': np.min(metric_data),
                    'max': np.max(metric_data),
                    'latest': metric_data[-1]
                }
        
        stats['adaptive_batch_size'] = self.adaptive_batch_size.copy()
        
        return stats


class HashtagMultiLevelSemanticAnalyzer:
    """
    해시태그 기반 다중수준 의미 분석 시스템
    
    2024년 연구 기반 구현:
    - 표면 의미 → 도덕/감정적 의미 → 인과적 의미 계층화
    - 해시태그 기반 의미 태깅
    - JSON 스키마 기반 구조화 출력
    - 의미론적 네트워크 구축
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 의미 분석 레이어 정의
        self.semantic_layers = {
            'surface': SurfaceSemanticLayer(),
            'ethical': EthicalSemanticLayer(), 
            'emotional': EmotionalSemanticLayer(),
            'causal': CausalSemanticLayer(),
            'contextual': ContextualSemanticLayer()
        }
        
        # 해시태그 시스템
        self.hashtag_system = HashtagSystem()
        
        # JSON 스키마 정의
        self.output_schema = self._define_output_schema()
        
        # 의미론적 네트워크
        self.semantic_network = SemanticNetwork()
        
        self.logger.info("해시태그 기반 다중수준 의미 분석 시스템 초기화 완료")
    
    async def analyze_multilevel_semantics(self, 
                                         text: str, 
                                         context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        다중수준 의미 분석 수행
        
        Args:
            text: 분석할 텍스트
            context: 상황 컨텍스트
            
        Returns:
            계층화된 의미 분석 결과
        """
        context = context or {}
        
        try:
            # 1. 각 의미 레이어별 분석
            layer_results = {}
            
            for layer_name, layer_analyzer in self.semantic_layers.items():
                layer_result = await layer_analyzer.analyze(text, context)
                layer_results[layer_name] = layer_result
            
            # 2. 해시태그 생성 및 태깅
            hashtags = await self.hashtag_system.generate_hashtags(text, layer_results)
            
            # 3. 계층간 의미 연결 분석
            semantic_connections = await self._analyze_semantic_connections(layer_results)
            
            # 4. 의미론적 네트워크 업데이트
            await self.semantic_network.update_network(text, layer_results, hashtags)
            
            # 5. 결과 구조화
            structured_result = await self._structure_results(
                text, layer_results, hashtags, semantic_connections, context
            )
            
            # 6. JSON 스키마 검증
            validated_result = await self._validate_output_schema(structured_result)
            
            return validated_result
            
        except Exception as e:
            self.logger.error(f"다중수준 의미 분석 실패: {e}")
            return await self._fallback_analysis(text, context)
    
    def _define_output_schema(self) -> Dict[str, Any]:
        """JSON 출력 스키마 정의"""
        return {
            "type": "object",
            "properties": {
                "input_text": {"type": "string"},
                "analysis_timestamp": {"type": "string"},
                "semantic_layers": {
                    "type": "object",
                    "properties": {
                        "surface": {
                            "type": "object",
                            "properties": {
                                "literal_meaning": {"type": "string"},
                                "key_entities": {"type": "array", "items": {"type": "string"}},
                                "syntactic_structure": {"type": "object"},
                                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                "hashtags": {"type": "array", "items": {"type": "string"}}
                            }
                        },
                        "ethical": {
                            "type": "object", 
                            "properties": {
                                "moral_stance": {"type": "string"},
                                "ethical_implications": {"type": "array", "items": {"type": "string"}},
                                "value_alignments": {"type": "object"},
                                "moral_reasoning": {"type": "string"},
                                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                "hashtags": {"type": "array", "items": {"type": "string"}}
                            }
                        },
                        "emotional": {
                            "type": "object",
                            "properties": {
                                "emotional_tone": {"type": "object"},
                                "emotional_intensity": {"type": "number"},
                                "emotional_categories": {"type": "array", "items": {"type": "string"}},
                                "emotional_trajectory": {"type": "array"},
                                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                "hashtags": {"type": "array", "items": {"type": "string"}}
                            }
                        },
                        "causal": {
                            "type": "object",
                            "properties": {
                                "causal_relationships": {"type": "array"},
                                "consequence_predictions": {"type": "array"},
                                "causal_chain": {"type": "object"},
                                "causal_strength": {"type": "number"},
                                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                "hashtags": {"type": "array", "items": {"type": "string"}}
                            }
                        },
                        "contextual": {
                            "type": "object",
                            "properties": {
                                "situational_context": {"type": "object"},
                                "cultural_context": {"type": "object"},
                                "temporal_context": {"type": "object"},
                                "social_context": {"type": "object"},
                                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                "hashtags": {"type": "array", "items": {"type": "string"}}
                            }
                        }
                    }
                },
                "semantic_connections": {
                    "type": "object",
                    "properties": {
                        "layer_correlations": {"type": "object"},
                        "cross_layer_patterns": {"type": "array"},
                        "semantic_consistency": {"type": "number"},
                        "connection_strength": {"type": "object"}
                    }
                },
                "hashtag_summary": {
                    "type": "object",
                    "properties": {
                        "all_hashtags": {"type": "array", "items": {"type": "string"}},
                        "primary_hashtags": {"type": "array", "items": {"type": "string"}},
                        "hashtag_categories": {"type": "object"},
                        "hashtag_network": {"type": "object"}
                    }
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "processing_time": {"type": "number"},
                        "model_versions": {"type": "object"},
                        "confidence_overall": {"type": "number", "minimum": 0, "maximum": 1},
                        "analysis_depth": {"type": "string"}
                    }
                }
            },
            "required": ["input_text", "semantic_layers", "hashtag_summary", "metadata"]
        }
    
    async def _analyze_semantic_connections(self, layer_results: Dict[str, Any]) -> Dict[str, Any]:
        """의미 레이어 간 연결 분석"""
        connections = {
            'layer_correlations': {},
            'cross_layer_patterns': [],
            'semantic_consistency': 0.0,
            'connection_strength': {}
        }
        
        try:
            # 레이어 간 상관관계 분석
            layer_names = list(layer_results.keys())
            for i, layer1 in enumerate(layer_names):
                for layer2 in layer_names[i+1:]:
                    correlation = await self._calculate_layer_correlation(
                        layer_results[layer1], layer_results[layer2]
                    )
                    connections['layer_correlations'][f"{layer1}-{layer2}"] = correlation
            
            # 크로스 레이어 패턴 식별
            patterns = await self._identify_cross_layer_patterns(layer_results)
            connections['cross_layer_patterns'] = patterns
            
            # 의미적 일관성 계산
            consistency = await self._calculate_semantic_consistency(layer_results)
            connections['semantic_consistency'] = consistency
            
            # 연결 강도 계산
            connection_strengths = await self._calculate_connection_strengths(layer_results)
            connections['connection_strength'] = connection_strengths
            
        except Exception as e:
            self.logger.error(f"의미 연결 분석 실패: {e}")
        
        return connections
    
    async def _calculate_layer_correlation(self, layer1_result: Dict, layer2_result: Dict) -> float:
        """두 의미 레이어 간 상관관계 계산"""
        try:
            # 간단한 상관관계 계산 (실제로는 더 복잡한 의미론적 유사도 계산)
            confidence1 = layer1_result.get('confidence', 0.5)
            confidence2 = layer2_result.get('confidence', 0.5)
            
            # 해시태그 기반 유사도
            hashtags1 = set(layer1_result.get('hashtags', []))
            hashtags2 = set(layer2_result.get('hashtags', []))
            
            if hashtags1 and hashtags2:
                hashtag_similarity = len(hashtags1 & hashtags2) / len(hashtags1 | hashtags2)
            else:
                hashtag_similarity = 0.0
            
            # 종합 상관관계
            correlation = (confidence1 * confidence2 * 0.5 + hashtag_similarity * 0.5)
            return correlation
            
        except Exception:
            return 0.0
    
    async def _identify_cross_layer_patterns(self, layer_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """크로스 레이어 패턴 식별"""
        patterns = []
        
        try:
            # 예시: 감정-윤리 패턴
            emotional_data = layer_results.get('emotional', {})
            ethical_data = layer_results.get('ethical', {})
            
            if emotional_data and ethical_data:
                emotional_tone = emotional_data.get('emotional_tone', {})
                moral_stance = ethical_data.get('moral_stance', '')
                
                # 감정과 도덕적 입장 간 패턴 분석
                if emotional_tone.get('valence', 0) < 0 and 'negative' in moral_stance.lower():
                    patterns.append({
                        'pattern_type': 'emotional_ethical_alignment',
                        'description': '부정적 감정과 부정적 도덕적 입장의 일치',
                        'strength': 0.8,
                        'layers_involved': ['emotional', 'ethical']
                    })
            
            # 예시: 표면-인과 패턴
            surface_data = layer_results.get('surface', {})
            causal_data = layer_results.get('causal', {})
            
            if surface_data and causal_data:
                key_entities = surface_data.get('key_entities', [])
                causal_relationships = causal_data.get('causal_relationships', [])
                
                # 엔티티가 인과관계에 참여하는 패턴
                entity_in_causal = any(
                    entity.lower() in str(causal_rel).lower()
                    for entity in key_entities
                    for causal_rel in causal_relationships
                )
                
                if entity_in_causal:
                    patterns.append({
                        'pattern_type': 'entity_causal_involvement',
                        'description': '표면 엔티티가 인과관계에 직접 참여',
                        'strength': 0.7,
                        'layers_involved': ['surface', 'causal']
                    })
                    
        except Exception as e:
            self.logger.error(f"크로스 레이어 패턴 식별 실패: {e}")
        
        return patterns
    
    async def _calculate_semantic_consistency(self, layer_results: Dict[str, Any]) -> float:
        """전체 의미적 일관성 계산"""
        try:
            confidences = []
            for layer_result in layer_results.values():
                if isinstance(layer_result, dict) and 'confidence' in layer_result:
                    confidences.append(layer_result['confidence'])
            
            if not confidences:
                return 0.5
            
            # 신뢰도의 일관성 (표준편차가 낮을수록 일관성 높음)
            consistency = 1.0 - (np.std(confidences) / np.mean(confidences)) if np.mean(confidences) > 0 else 0.5
            return max(0.0, min(1.0, consistency))
            
        except Exception:
            return 0.5
    
    async def _calculate_connection_strengths(self, layer_results: Dict[str, Any]) -> Dict[str, float]:
        """레이어 간 연결 강도 계산"""
        strengths = {}
        
        try:
            layer_names = list(layer_results.keys())
            
            for layer_name in layer_names:
                # 각 레이어의 다른 레이어들과의 평균 연결 강도
                layer_strengths = []
                
                for other_layer in layer_names:
                    if layer_name != other_layer:
                        correlation = await self._calculate_layer_correlation(
                            layer_results[layer_name], layer_results[other_layer]
                        )
                        layer_strengths.append(correlation)
                
                strengths[layer_name] = np.mean(layer_strengths) if layer_strengths else 0.0
                
        except Exception as e:
            self.logger.error(f"연결 강도 계산 실패: {e}")
        
        return strengths
    
    async def _structure_results(self, 
                               text: str,
                               layer_results: Dict[str, Any],
                               hashtags: Dict[str, Any], 
                               semantic_connections: Dict[str, Any],
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """결과를 JSON 스키마에 맞게 구조화"""
        
        structured_result = {
            "input_text": text,
            "analysis_timestamp": datetime.now().isoformat(),
            "semantic_layers": layer_results,
            "semantic_connections": semantic_connections,
            "hashtag_summary": hashtags,
            "metadata": {
                "processing_time": time.time(),  # 실제로는 시작 시간과의 차이
                "model_versions": {
                    "semantic_analyzer": "v2.0",
                    "hashtag_system": "v1.5",
                    "schema_version": "v1.0"
                },
                "confidence_overall": np.mean([
                    layer_result.get('confidence', 0.5) 
                    for layer_result in layer_results.values()
                    if isinstance(layer_result, dict)
                ]),
                "analysis_depth": "full_multilevel"
            }
        }
        
        return structured_result
    
    async def _validate_output_schema(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """JSON 스키마 검증"""
        try:
            # 여기서는 간단한 검증만 수행 (실제로는 jsonschema 라이브러리 사용)
            required_fields = ['input_text', 'semantic_layers', 'hashtag_summary', 'metadata']
            
            for field in required_fields:
                if field not in result:
                    self.logger.warning(f"필수 필드 누락: {field}")
                    result[field] = {}
            
            # 신뢰도 값 범위 검증
            for layer_name, layer_data in result.get('semantic_layers', {}).items():
                if isinstance(layer_data, dict) and 'confidence' in layer_data:
                    confidence = layer_data['confidence']
                    if not (0 <= confidence <= 1):
                        layer_data['confidence'] = max(0, min(1, confidence))
            
            result['schema_validated'] = True
            
        except Exception as e:
            self.logger.error(f"스키마 검증 실패: {e}")
            result['schema_validated'] = False
        
        return result
    
    async def _fallback_analysis(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """폴백 분석"""
        return {
            "input_text": text,
            "analysis_timestamp": datetime.now().isoformat(),
            "semantic_layers": {
                "surface": {
                    "literal_meaning": text,
                    "confidence": 0.5,
                    "hashtags": ["#fallback"]
                }
            },
            "hashtag_summary": {
                "all_hashtags": ["#fallback"],
                "primary_hashtags": ["#fallback"]
            },
            "metadata": {
                "analysis_depth": "fallback",
                "confidence_overall": 0.5
            },
            "error": "분석 실패로 인한 폴백"
        }


class SurfaceSemanticLayer:
    """표면 의미 레이어"""
    
    async def analyze(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """표면 의미 분석"""
        return {
            "literal_meaning": text,
            "key_entities": await self._extract_entities(text),
            "syntactic_structure": await self._analyze_syntax(text),
            "confidence": 0.8,
            "hashtags": ["#surface", "#literal"]
        }
    
    async def _extract_entities(self, text: str) -> List[str]:
        """개체명 추출 (간단 구현)"""
        # 실제로는 NER 모델 사용
        words = text.split()
        entities = [word for word in words if word[0].isupper()]
        return entities[:5]  # 최대 5개
    
    async def _analyze_syntax(self, text: str) -> Dict[str, Any]:
        """구문 분석 (간단 구현)"""
        return {
            "sentence_count": len(text.split('.')),
            "word_count": len(text.split()),
            "average_word_length": np.mean([len(word) for word in text.split()])
        }


class EthicalSemanticLayer:
    """윤리적 의미 레이어"""
    
    async def analyze(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """윤리적 의미 분석"""
        moral_stance = await self._determine_moral_stance(text)
        ethical_implications = await self._identify_ethical_implications(text)
        
        return {
            "moral_stance": moral_stance,
            "ethical_implications": ethical_implications,
            "value_alignments": await self._analyze_value_alignments(text),
            "moral_reasoning": await self._extract_moral_reasoning(text),
            "confidence": 0.7,
            "hashtags": ["#ethics", "#moral", f"#{moral_stance.lower()}"]
        }
    
    async def _determine_moral_stance(self, text: str) -> str:
        """도덕적 입장 결정"""
        positive_words = ['good', 'right', 'moral', 'ethical', 'virtue']
        negative_words = ['bad', 'wrong', 'immoral', 'unethical', 'vice']
        
        text_lower = text.lower()
        positive_score = sum(1 for word in positive_words if word in text_lower)
        negative_score = sum(1 for word in negative_words if word in text_lower)
        
        if positive_score > negative_score:
            return "positive_moral"
        elif negative_score > positive_score:
            return "negative_moral"
        else:
            return "neutral_moral"
    
    async def _identify_ethical_implications(self, text: str) -> List[str]:
        """윤리적 함의 식별"""
        implications = []
        
        if 'responsibility' in text.lower():
            implications.append('individual_responsibility')
        if 'harm' in text.lower():
            implications.append('harm_principle')
        if 'fairness' in text.lower():
            implications.append('fairness_consideration')
        
        return implications
    
    async def _analyze_value_alignments(self, text: str) -> Dict[str, float]:
        """가치 정렬 분석"""
        return {
            'autonomy': 0.6,
            'beneficence': 0.7,
            'justice': 0.5,
            'dignity': 0.8
        }
    
    async def _extract_moral_reasoning(self, text: str) -> str:
        """도덕적 추론 추출"""
        if 'because' in text.lower() or 'since' in text.lower():
            return "causal_moral_reasoning"
        elif 'should' in text.lower() or 'ought' in text.lower():
            return "deontological_reasoning"
        elif 'consequence' in text.lower() or 'result' in text.lower():
            return "consequentialist_reasoning"
        else:
            return "unclear_reasoning"


class EmotionalSemanticLayer:
    """감정적 의미 레이어"""
    
    async def analyze(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """감정적 의미 분석"""
        emotional_tone = await self._analyze_emotional_tone(text)
        
        return {
            "emotional_tone": emotional_tone,
            "emotional_intensity": await self._calculate_emotional_intensity(text),
            "emotional_categories": await self._categorize_emotions(text),
            "emotional_trajectory": await self._analyze_emotional_trajectory(text),
            "confidence": 0.75,
            "hashtags": ["#emotion", "#feeling"] + [f"#{cat}" for cat in await self._categorize_emotions(text)]
        }
    
    async def _analyze_emotional_tone(self, text: str) -> Dict[str, float]:
        """감정 톤 분석"""
        return {
            'valence': 0.2,  # 긍정-부정
            'arousal': 0.6,  # 각성도
            'dominance': 0.4  # 지배력
        }
    
    async def _calculate_emotional_intensity(self, text: str) -> float:
        """감정 강도 계산"""
        exclamation_count = text.count('!')
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        intensity = min(1.0, (exclamation_count * 0.2 + caps_ratio) * 2)
        return intensity
    
    async def _categorize_emotions(self, text: str) -> List[str]:
        """감정 카테고리 분류"""
        emotions = []
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['happy', 'joy', 'glad']):
            emotions.append('joy')
        if any(word in text_lower for word in ['sad', 'sorrow', 'grief']):
            emotions.append('sadness')
        if any(word in text_lower for word in ['angry', 'mad', 'furious']):
            emotions.append('anger')
        if any(word in text_lower for word in ['fear', 'afraid', 'scared']):
            emotions.append('fear')
        
        return emotions if emotions else ['neutral']
    
    async def _analyze_emotional_trajectory(self, text: str) -> List[Dict[str, Any]]:
        """감정 궤적 분석"""
        sentences = text.split('.')
        trajectory = []
        
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                trajectory.append({
                    'position': i,
                    'emotional_state': await self._analyze_emotional_tone(sentence),
                    'intensity': await self._calculate_emotional_intensity(sentence)
                })
        
        return trajectory


class CausalSemanticLayer:
    """인과적 의미 레이어"""
    
    async def analyze(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """인과적 의미 분석"""
        return {
            "causal_relationships": await self._identify_causal_relationships(text),
            "consequence_predictions": await self._predict_consequences(text),
            "causal_chain": await self._build_causal_chain(text),
            "causal_strength": await self._calculate_causal_strength(text),
            "confidence": 0.65,
            "hashtags": ["#causal", "#consequence", "#cause_effect"]
        }
    
    async def _identify_causal_relationships(self, text: str) -> List[Dict[str, str]]:
        """인과관계 식별"""
        relationships = []
        text_lower = text.lower()
        
        # 간단한 인과관계 패턴 매칭
        if 'because' in text_lower:
            relationships.append({
                'type': 'causal_explanation',
                'strength': 0.8,
                'pattern': 'because'
            })
        
        if 'leads to' in text_lower or 'results in' in text_lower:
            relationships.append({
                'type': 'direct_causation',
                'strength': 0.9,
                'pattern': 'leads_to'
            })
        
        return relationships
    
    async def _predict_consequences(self, text: str) -> List[str]:
        """결과 예측"""
        consequences = []
        
        if 'action' in text.lower():
            consequences.append('behavioral_change')
        if 'decision' in text.lower():
            consequences.append('outcome_variation')
        
        return consequences
    
    async def _build_causal_chain(self, text: str) -> Dict[str, Any]:
        """인과 사슬 구축"""
        return {
            'initial_cause': 'identified_trigger',
            'intermediate_steps': ['step1', 'step2'],
            'final_effect': 'predicted_outcome',
            'chain_length': 3
        }
    
    async def _calculate_causal_strength(self, text: str) -> float:
        """인과 강도 계산"""
        causal_indicators = ['because', 'since', 'leads to', 'results in', 'causes', 'due to']
        strength = sum(1 for indicator in causal_indicators if indicator in text.lower())
        return min(1.0, strength * 0.2)


class ContextualSemanticLayer:
    """맥락적 의미 레이어"""
    
    async def analyze(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """맥락적 의미 분석"""
        return {
            "situational_context": await self._analyze_situational_context(text, context),
            "cultural_context": await self._analyze_cultural_context(text, context),
            "temporal_context": await self._analyze_temporal_context(text, context),
            "social_context": await self._analyze_social_context(text, context),
            "confidence": 0.7,
            "hashtags": ["#context", "#situation", "#culture"]
        }
    
    async def _analyze_situational_context(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """상황적 맥락 분석"""
        return {
            'setting': context.get('setting', 'unknown'),
            'participants': context.get('participants', []),
            'activity_type': context.get('activity_type', 'conversation')
        }
    
    async def _analyze_cultural_context(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """문화적 맥락 분석"""
        return {
            'cultural_background': context.get('culture', 'unknown'),
            'language_variety': 'standard',
            'cultural_references': []
        }
    
    async def _analyze_temporal_context(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """시간적 맥락 분석"""
        return {
            'time_period': context.get('time_period', 'present'),
            'temporal_markers': [],
            'sequence_indicators': []
        }
    
    async def _analyze_social_context(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """사회적 맥락 분석"""
        return {
            'social_setting': context.get('social_setting', 'informal'),
            'power_dynamics': context.get('power_dynamics', 'equal'),
            'relationship_type': context.get('relationship', 'unknown')
        }


class HashtagSystem:
    """해시태그 시스템"""
    
    async def generate_hashtags(self, text: str, layer_results: Dict[str, Any]) -> Dict[str, Any]:
        """해시태그 생성 및 관리"""
        
        # 각 레이어에서 해시태그 수집
        all_hashtags = []
        for layer_result in layer_results.values():
            if isinstance(layer_result, dict) and 'hashtags' in layer_result:
                all_hashtags.extend(layer_result['hashtags'])
        
        # 중복 제거
        unique_hashtags = list(set(all_hashtags))
        
        # 텍스트 기반 추가 해시태그 생성
        text_based_hashtags = await self._generate_text_based_hashtags(text)
        all_hashtags.extend(text_based_hashtags)
        
        # 해시태그 카테고리화
        categorized_hashtags = await self._categorize_hashtags(unique_hashtags)
        
        # 주요 해시태그 선별
        primary_hashtags = await self._select_primary_hashtags(unique_hashtags, layer_results)
        
        return {
            "all_hashtags": unique_hashtags,
            "primary_hashtags": primary_hashtags,
            "hashtag_categories": categorized_hashtags,
            "hashtag_network": await self._build_hashtag_network(unique_hashtags)
        }
    
    async def _generate_text_based_hashtags(self, text: str) -> List[str]:
        """텍스트 기반 해시태그 생성"""
        hashtags = []
        
        # 길이 기반
        if len(text) > 100:
            hashtags.append('#long_text')
        elif len(text) < 20:
            hashtags.append('#short_text')
        
        # 질문 형태
        if '?' in text:
            hashtags.append('#question')
        
        # 감탄문
        if '!' in text:
            hashtags.append('#exclamation')
        
        return hashtags
    
    async def _categorize_hashtags(self, hashtags: List[str]) -> Dict[str, List[str]]:
        """해시태그 카테고리화"""
        categories = {
            'semantic_layer': [],
            'emotion': [],
            'content_type': [],
            'analysis_feature': []
        }
        
        for hashtag in hashtags:
            if hashtag in ['#surface', '#ethical', '#emotional', '#causal', '#contextual']:
                categories['semantic_layer'].append(hashtag)
            elif hashtag in ['#joy', '#sadness', '#anger', '#fear', '#emotion']:
                categories['emotion'].append(hashtag)
            elif hashtag in ['#question', '#exclamation', '#long_text', '#short_text']:
                categories['content_type'].append(hashtag)
            else:
                categories['analysis_feature'].append(hashtag)
        
        return categories
    
    async def _select_primary_hashtags(self, hashtags: List[str], layer_results: Dict[str, Any]) -> List[str]:
        """주요 해시태그 선별"""
        # 신뢰도가 높은 레이어의 해시태그 우선 선별
        hashtag_scores = {}
        
        for layer_name, layer_result in layer_results.items():
            if isinstance(layer_result, dict):
                confidence = layer_result.get('confidence', 0.5)
                layer_hashtags = layer_result.get('hashtags', [])
                
                for hashtag in layer_hashtags:
                    if hashtag not in hashtag_scores:
                        hashtag_scores[hashtag] = 0
                    hashtag_scores[hashtag] += confidence
        
        # 점수 순으로 정렬하여 상위 5개 선별
        sorted_hashtags = sorted(hashtag_scores.items(), key=lambda x: x[1], reverse=True)
        primary_hashtags = [hashtag for hashtag, score in sorted_hashtags[:5]]
        
        return primary_hashtags
    
    async def _build_hashtag_network(self, hashtags: List[str]) -> Dict[str, Any]:
        """해시태그 네트워크 구축"""
        # 간단한 네트워크 구조
        network = {
            'nodes': [{'id': hashtag, 'type': 'hashtag'} for hashtag in hashtags],
            'edges': [],
            'clusters': []
        }
        
        # 관련성이 높은 해시태그 간 엣지 생성 (간단 구현)
        for i, hashtag1 in enumerate(hashtags):
            for hashtag2 in hashtags[i+1:]:
                # 카테고리가 같은 해시태그들을 연결
                if await self._hashtags_related(hashtag1, hashtag2):
                    network['edges'].append({
                        'source': hashtag1,
                        'target': hashtag2,
                        'weight': 0.5
                    })
        
        return network
    
    async def _hashtags_related(self, hashtag1: str, hashtag2: str) -> bool:
        """해시태그 간 관련성 판단"""
        # 간단한 관련성 판단 로직
        emotion_tags = ['#joy', '#sadness', '#anger', '#fear', '#emotion']
        layer_tags = ['#surface', '#ethical', '#emotional', '#causal', '#contextual']
        
        if hashtag1 in emotion_tags and hashtag2 in emotion_tags:
            return True
        if hashtag1 in layer_tags and hashtag2 in layer_tags:
            return True
        
        return False


class SemanticNetwork:
    """의미론적 네트워크"""
    
    def __init__(self):
        self.network = {
            'nodes': {},
            'edges': {},
            'clusters': {}
        }
    
    async def update_network(self, text: str, layer_results: Dict[str, Any], hashtags: Dict[str, Any]):
        """네트워크 업데이트"""
        # 텍스트를 노드로 추가
        text_id = hash(text) % 1000000
        self.network['nodes'][text_id] = {
            'text': text[:100],  # 텍스트 일부만 저장
            'layer_results': layer_results,
            'hashtags': hashtags['all_hashtags'],
            'timestamp': datetime.now().isoformat()
        }
        
        # 유사한 텍스트와의 엣지 생성
        await self._create_similarity_edges(text_id, text, layer_results)
    
    async def _create_similarity_edges(self, text_id: int, text: str, layer_results: Dict[str, Any]):
        """유사성 기반 엣지 생성"""
        # 기존 노드들과의 유사성 계산
        for node_id, node_data in self.network['nodes'].items():
            if node_id != text_id:
                similarity = await self._calculate_semantic_similarity(
                    layer_results, node_data['layer_results']
                )
                
                if similarity > 0.7:  # 임계값 이상인 경우 엣지 생성
                    edge_id = f"{min(text_id, node_id)}_{max(text_id, node_id)}"
                    self.network['edges'][edge_id] = {
                        'source': text_id,
                        'target': node_id,
                        'weight': similarity,
                        'type': 'semantic_similarity'
                    }
    
    async def _calculate_semantic_similarity(self, layer1: Dict[str, Any], layer2: Dict[str, Any]) -> float:
        """의미론적 유사성 계산"""
        similarities = []
        
        for layer_name in layer1.keys():
            if layer_name in layer2:
                layer_sim = await self._calculate_layer_similarity(layer1[layer_name], layer2[layer_name])
                similarities.append(layer_sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    async def _calculate_layer_similarity(self, layer_data1: Dict, layer_data2: Dict) -> float:
        """레이어 데이터 간 유사성 계산"""
        # 간단한 해시태그 기반 유사성
        hashtags1 = set(layer_data1.get('hashtags', []))
        hashtags2 = set(layer_data2.get('hashtags', []))
        
        if not hashtags1 and not hashtags2:
            return 0.5
        
        intersection = len(hashtags1 & hashtags2)
        union = len(hashtags1 | hashtags2)
        
        return intersection / union if union > 0 else 0.0


if __name__ == "__main__":
    # 테스트 실행
    asyncio.run(test_hierarchical_emotion_system())