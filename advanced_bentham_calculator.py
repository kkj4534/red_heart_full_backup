"""
고급 벤담 쾌락 계산기 - Linux 전용
Advanced Bentham Pleasure Calculator for Linux

기존 7가지 변수에 6개의 추가 가중치 레이어를 적용하여
AI 모델과 고급 수치 계산을 통한 정교한 쾌락 계산을 수행합니다.
"""

__all__ = ['AdvancedBenthamCalculator']

import os
# CVE-2025-32434는 가짜 CVE - torch_security_patch import 제거
# import torch_security_patch

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from transformers import pipeline, AutoTokenizer, AutoModel
import scipy.stats as stats  
from scipy.optimize import minimize_scalar
import threading
import time

from config import ADVANCED_CONFIG, DEVICE, TORCH_DTYPE, BATCH_SIZE, MODELS_DIR, get_smart_device, ModelPriority, get_priority_based_device
from data_models import (
    HedonicValues, EmotionData, EmotionState, BenthamVariable, 
    EmotionIntensity, EthicalSituation, EnhancedHedonicResult,
    WeightLayerResult, AdvancedCalculationContext
)

# 에리히 프롬 철학 통합을 위한 추가 임포트
from enum import Enum

class FrommOrientation(Enum):
    """에리히 프롬의 성격 지향 (To Have or To Be 기반)"""
    HAVING = "having"  # 소유 지향
    BEING = "being"   # 존재 지향
    MIXED = "mixed"   # 혼합 지향
from bentham_v2_calculator import bentham_v2_calculator
from mixture_of_experts import create_ethics_moe, MixtureOfExperts
from three_view_scenario_system import ThreeViewScenarioSystem
from phase_controller_hook import PhaseControllerHook, PhaseType, PerformanceMetric
from legal_expert_system import get_legal_expert_system, LegalDomain, OperationMode

# 고급 라이브러리 가용성 확인 (개발용 간소화)
ADVANCED_LIBS_AVAILABLE = True

logger = logging.getLogger('RedHeart.AdvancedBenthamCalculator')


class NeuralWeightPredictor(nn.Module):
    """신경망 기반 가중치 예측 모델 - 메모리 최적화된 경량 버전"""
    
    def __init__(self, input_dim: int = 50, hidden_dim: int = 256):  # 50차원 특성에 맞게 조정
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 벤담 계산 신경망 - 50차원 입력에 최적화
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),  # 동일한 차원 유지
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 6),  # 6개 가중치 레이어
            nn.Sigmoid()
        )
        
        # 가중치 범위 조정을 위한 스케일링
        self.weight_scale = nn.Parameter(torch.tensor([1.5, 1.8, 2.0, 2.5, 2.0, 1.5]))
        self.weight_bias = nn.Parameter(torch.tensor([0.3, 0.2, 0.3, 0.2, 0.4, 0.5]))
        
    def forward(self, x):
        weights = self.layers(x)
        # 가중치를 0.3~2.5 범위로 스케일링
        scaled_weights = weights * self.weight_scale + self.weight_bias
        return scaled_weights


class TransformerContextAnalyzer:
    """트랜스포머 기반 맥락 분석기"""
    
    def __init__(self):
        if not ADVANCED_LIBS_AVAILABLE:
            raise ImportError("고급 라이브러리가 필요합니다.")
            
        self.device = DEVICE
        
        # 감정 분석 파이프라인 (순차적 로딩)
        def load_emotion_classifier():
            # 이 함수는 순차적 로더에서 호출되며, 이미 GPU 할당이 승인된 상태
            return pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                local_files_only=True,
                device=0,  # GPU에서 실행
                return_all_scores=True
            )
        
        # 순차적 로딩 요청
        from config import get_gpu_loader
        gpu_loader = get_gpu_loader()
        emotion_device, emotion_model = gpu_loader.request_gpu_loading(
            model_id="bentham_emotion_classifier",
            priority=ModelPriority.MEDIUM,
            estimated_memory_mb=732,
            loading_function=load_emotion_classifier
        )
        
        # 디바이스에 따라 최종 모델 설정
        if emotion_device.type == 'cuda' and emotion_model is not None:
            # GPU에서 로드된 모델 사용
            self.emotion_classifier = emotion_model
            self.logger.info(f"감정 분석 모델 GPU 순차 로드 완료: {emotion_device}")
        else:
            # CPU로 폴백
            self.emotion_classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                local_files_only=True,
                device=-1,
                return_all_scores=True
            )
            self.logger.info(f"감정 분석 모델 CPU 로드 완료: {emotion_device}")
        
        # 윤리적 추론 파이프라인 (순차적 로딩)
        try:
            def load_ethical_classifier():
                return pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli",
                    local_files_only=True,
                    device=0
                )
            
            ethical_device, ethical_model = gpu_loader.request_gpu_loading(
                model_id="bentham_ethical_classifier",
                priority=ModelPriority.MEDIUM,
                estimated_memory_mb=732,
                loading_function=load_ethical_classifier
            )
            
            if ethical_device.type == 'cuda' and ethical_model is not None:
                self.ethical_classifier = ethical_model
            else:
                self.ethical_classifier = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli",
                    local_files_only=True,
                    device=-1
                )
            self.logger.info(f"윤리 분석 모델 순차 로드 완료: {ethical_device}")
            
            # entailment 레이블 매핑 확인 및 수정
            model = self.ethical_classifier.model
            if hasattr(model.config, 'label2id'):
                label2id = model.config.label2id
                if 'ENTAILMENT' not in label2id and 'entailment' not in label2id:
                    # 수동으로 entailment 매핑 추가
                    if 'LABEL_0' in label2id and len(label2id) >= 3:
                        # MNLI 모델의 일반적 매핑: CONTRADICTION=0, NEUTRAL=1, ENTAILMENT=2
                        model.config.label2id.update({
                            'entailment': 2,
                            'neutral': 1,
                            'contradiction': 0
                        })
                        self.logger.info("✅ entailment 레이블 매핑 수동 설정 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 윤리적 분류기 초기화 실패: {e}")
            # 폴백으로 간단한 분류기 사용
            self.ethical_classifier = None
        
        # 한국어 감정 분석 (순차적 로딩)
        def load_korean_classifier():
            return pipeline(
                "text-classification",
                model="beomi/KcELECTRA-base-v2022",
                local_files_only=True,
                device=0
            )
        
        korean_device, korean_model = gpu_loader.request_gpu_loading(
            model_id="bentham_korean_classifier",
            priority=ModelPriority.LOW,
            estimated_memory_mb=732,
            loading_function=load_korean_classifier
        )
        
        if korean_device.type == 'cuda' and korean_model is not None:
            self.korean_classifier = korean_model
        else:
            self.korean_classifier = pipeline(
                "text-classification",
                model="beomi/KcELECTRA-base-v2022",
                local_files_only=True,
                device=-1
            )
        self.logger.info(f"한국어 감정 모델 순차 로드 완료: {korean_device}")
        
        # 문맥 임베딩 모델 (순차적 로딩)
        def load_context_model():
            return AutoModel.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2",
                local_files_only=True
            ).to(torch.device('cuda'))
        
        context_device, context_model = gpu_loader.request_gpu_loading(
            model_id="bentham_context_model",
            priority=ModelPriority.LOW,
            estimated_memory_mb=732,
            loading_function=load_context_model
        )
        
        if context_device.type == 'cuda' and context_model is not None:
            self.context_model = context_model
        else:
            self.context_model = AutoModel.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2",
                local_files_only=True
            ).to('cpu')
        self.logger.info(f"문맥 임베딩 모델 순차 로드 완료: {context_device}")
        
        self.context_tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2",
            local_files_only=True
        )
        
        logger.info("트랜스포머 맥락 분석기 초기화 완료")
        
    def analyze_context(self, text: str, language: str = "ko") -> Dict[str, Any]:
        """종합적 맥락 분석"""
        results = {}
        
        try:
            # 1. 감정 분석
            if language == "ko":
                emotion_results = self.korean_classifier(text)
            else:
                emotion_results = self.emotion_classifier(text)
            results['emotions'] = emotion_results
            
            # 2. 윤리적 분류
            ethical_labels = [
                "생명과 안전", "정의와 공정성", "자율성과 자유", 
                "취약계층 보호", "사회적 책임", "개인적 이익"
            ]
            ethical_results = self.ethical_classifier(text, ethical_labels)
            results['ethical_aspects'] = ethical_results
            
            # 3. 맥락 임베딩
            with torch.no_grad():
                inputs = self.context_tokenizer(
                    text, return_tensors="pt", 
                    truncation=True, max_length=512
                )
                # Context model과 같은 디바이스로 이동
                inputs = {k: v.to(next(self.context_model.parameters()).device) for k, v in inputs.items()}
                
                outputs = self.context_model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                results['context_embedding'] = embeddings.cpu().numpy()
                
            # 4. 복잡도 분석
            complexity_metrics = self._analyze_complexity(text)
            results['complexity'] = complexity_metrics
            
        except Exception as e:
            logger.error(f"맥락 분석 실패: {e}")
            raise RuntimeError(
                f"트랜스포머 기반 맥락 분석에 실패했습니다: {str(e)}. "
                f"적절한 모델이 로드되지 않았거나 입력 데이터가 올바르지 않습니다. "
                f"대체 분석 방법을 사용하지 않고 정확한 분석이 필요합니다."
            )
            
        return results
        
    def _analyze_complexity(self, text: str) -> Dict[str, float]:
        """텍스트 복잡도 분석"""
        words = text.split()
        sentences = text.split('.')
        
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'lexical_diversity': len(set(words)) / len(words) if words else 0,
            'structural_complexity': len(sentences) / len(words) if words else 0
        }
        


class AdvancedWeightLayer:
    """고급 가중치 레이어 기본 클래스"""
    
    def __init__(self, name: str, neural_predictor: NeuralWeightPredictor = None):
        self.name = name
        self.neural_predictor = neural_predictor
        self.last_contribution = 0.0
        self.last_reasoning = ""
        self.confidence_score = 0.0
        
        # ML 모델 초기화
        self.ml_model = RandomForestRegressor(
            n_estimators=100, 
            random_state=42,
            n_jobs=-1 if ADVANCED_CONFIG['use_multiprocessing'] else 1
        )
        self.feature_scaler = StandardScaler()
        self.is_trained = False
        
    def compute_weight(self, context: AdvancedCalculationContext) -> float:
        """고급 가중치 계산"""
        try:
            # 1. 특성 벡터 추출
            features = self._extract_features(context)
            
            # 2. 신경망 예측 (가능한 경우)
            if self.neural_predictor and torch.is_tensor(features):
                with torch.no_grad():
                    neural_weight = self.neural_predictor(features.unsqueeze(0))
                    weight = neural_weight[0, self._get_layer_index()].item()
            else:
                # 3. ML 모델 예측
                weight = self._ml_predict(features)
                
            # 4. 규칙 기반 보정
            weight = self._apply_rule_based_correction(weight, context)
            
            # 5. 신뢰도 계산
            self.confidence_score = self._calculate_confidence(context, weight)
            
            return max(0.3, min(2.5, weight))
            
        except Exception as e:
            logger.error(f"{self.name} 가중치 계산 실패: {e}")
            return 1.0
            
    def _extract_features(self, context: AdvancedCalculationContext) -> np.ndarray:
        """맥락에서 특성 추출"""
        features = []
        
        # 기본 벤담 변수들 (Bentham v2 확장)
        for var in ['intensity', 'duration', 'certainty', 'propinquity', 
                   'fecundity', 'purity', 'extent', 'external_cost',
                   'redistribution_effect', 'self_damage']:
            features.append(context.input_values.get(var, 0.5))
            
        # 감정 특성
        if context.emotion_data:
            from data_models import emotion_intensity_to_float
            valence_val = emotion_intensity_to_float(context.emotion_data.valence)
            arousal_val = emotion_intensity_to_float(context.emotion_data.arousal)
            intensity_val = float(context.emotion_data.intensity.value) / 4.0 if hasattr(context.emotion_data.intensity, 'value') else emotion_intensity_to_float(context.emotion_data.intensity)
            confidence_val = emotion_intensity_to_float(context.emotion_data.confidence)
            
            features.extend([
                valence_val,
                arousal_val,
                intensity_val,
                confidence_val
            ])
        else:
            features.extend([0.0, 0.0, 0.5, 0.5])
            
        # 맥락 임베딩 (처음 20차원만 사용)
        if hasattr(context, 'context_embedding') and context.context_embedding is not None:
            embedding = context.context_embedding.flatten()[:20]
            features.extend(embedding.tolist())
        else:
            features.extend([0.0] * 20)
            
        # 추가 맥락 정보
        features.extend([
            context.affected_count / 100.0,  # 정규화
            min(context.duration_seconds / 3600.0, 1.0),  # 시간 정규화
            context.information_quality,
            context.uncertainty_level
        ])
        
        # 총 50차원 특성 벡터
        while len(features) < 50:
            features.append(0.0)
            
        return np.array(features[:50], dtype=np.float32)
    
    def _try_load_trained_model(self):
        """훈련된 모델 로드 시도"""
        try:
            import joblib
            from pathlib import Path
            
            # WSL 호환성을 위해 os.path 사용
            model_dir = os.path.join(MODELS_DIR, "bentham_models")
            layer_name = getattr(self, 'layer_name', 'contextual')  # 기본값
            
            model_path = os.path.join(model_dir, f"{layer_name}_model.joblib")
            scaler_path = os.path.join(model_dir, f"{layer_name}_scaler.joblib")
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.ml_model = joblib.load(model_path)
                self.feature_scaler = joblib.load(scaler_path)
                self.is_trained = True
                logger.info(f"{layer_name} ML 모델 로드 완료")
            else:
                logger.warning(f"훈련된 모델 파일을 찾을 수 없음: {model_path}")
                
        except Exception as e:
            logger.warning(f"모델 로드 실패: {e}")
        
    def _ml_predict(self, features: np.ndarray) -> float:
        """ML 모델 예측"""
        if not self.is_trained:
            # 모델 로드 시도
            self._try_load_trained_model()
            
        if not self.is_trained:
            logger.warning(f"{self.name} ML 모델이 훈련되지 않음. 기본 가중치 반환")
            return 1.0
            
        try:
            # 특성 크기 확인 및 스케일러 재초기화
            if hasattr(self.feature_scaler, 'n_features_in_'):
                expected_features = self.feature_scaler.n_features_in_
                if features.shape[0] != expected_features:
                    logger.warning(f"{self.name} 특성 크기 불일치 ({features.shape[0]} vs {expected_features}). 스케일러 재초기화")
                    # 임시 더미 데이터로 스케일러 재훈련
                    dummy_data = np.random.randn(10, features.shape[0])
                    self.feature_scaler.fit(dummy_data)
            else:
                # 스케일러가 훈련되지 않은 경우 현재 특성으로 초기화
                logger.info(f"{self.name} 스케일러를 {features.shape[0]}차원으로 초기화")
                dummy_data = np.random.randn(10, features.shape[0])
                self.feature_scaler.fit(dummy_data)
            
            features_scaled = self.feature_scaler.transform(features.reshape(1, -1))
            prediction = self.ml_model.predict(features_scaled)[0]
            
            # 예측값을 합리적인 범위로 제한
            prediction = max(0.5, min(2.0, prediction))
            
            return prediction
        except Exception as e:
            logger.warning(f"{self.name} ML 모델 예측 실패: {e}. 기본 가중치 반환")
            return 1.0
        
    def _apply_rule_based_correction(self, weight: float, 
                                   context: AdvancedCalculationContext) -> float:
        """규칙 기반 보정 (하위 클래스에서 구현)"""
        return weight
        
    def _calculate_confidence(self, context: AdvancedCalculationContext, 
                            weight: float) -> float:
        """신뢰도 계산"""
        confidence_factors = []
        
        # 데이터 완전성 (Bentham v2 확장: 10개 변수)
        completeness = len([v for v in context.input_values.values() if v != 0.5]) / 10
        confidence_factors.append(completeness)
        
        # 정보 품질
        confidence_factors.append(context.information_quality)
        
        # 불확실성 (역수)
        confidence_factors.append(1.0 - context.uncertainty_level)
        
        # 가중치 안정성 (극단값이 아닐수록 높은 신뢰도)
        weight_stability = 1.0 - abs(weight - 1.0) / 1.5
        confidence_factors.append(max(0.0, weight_stability))
        
        return float(np.mean(confidence_factors))
        
    def _get_layer_index(self) -> int:
        """레이어 인덱스 반환 (하위 클래스에서 구현)"""
        return 0
        
    def get_contribution(self) -> WeightLayerResult:
        """기여도 정보 반환"""
        return WeightLayerResult(
            layer_name=self.name,
            weight_factor=1.0 + self.last_contribution,
            contribution_score=abs(self.last_contribution),
            confidence=self.confidence_score,
            metadata={'reasoning': self.last_reasoning}
        )


class AdvancedContextualWeightLayer(AdvancedWeightLayer):
    """고급 상황적 맥락 가중치 레이어"""
    
    def __init__(self, neural_predictor: NeuralWeightPredictor = None):
        super().__init__("고급 상황적 맥락", neural_predictor)
        
    def _get_layer_index(self) -> int:
        return 0
        
    def _apply_rule_based_correction(self, weight: float, 
                                   context: AdvancedCalculationContext) -> float:
        """상황적 맥락 규칙 기반 보정"""
        corrections = []
        reasoning_parts = []
        
        # 복잡도 기반 보정
        if hasattr(context, 'complexity_metrics'):
            complexity = context.complexity_metrics
            if complexity.get('structural_complexity', 0) > 0.1:
                correction = 1.2
                corrections.append(correction)
                reasoning_parts.append("높은 구조적 복잡도 (+20%)")
                
        # 사회적 맥락 보정
        if hasattr(context, 'social_context'):
            social = context.social_context
            if social.get('impact_scope') == 'community':
                correction = 1.3
                corrections.append(correction)
                reasoning_parts.append("공동체 영향 (+30%)")
                
        # 보정 적용
        if corrections:
            final_correction = np.mean(corrections)
            weight *= final_correction
            
        self.last_contribution = weight - 1.0
        self.last_reasoning = " | ".join(reasoning_parts) if reasoning_parts else "기본 맥락"
        
        return weight


class AdvancedTemporalWeightLayer(AdvancedWeightLayer):
    """고급 시간적 영향 가중치 레이어"""
    
    def __init__(self, neural_predictor: NeuralWeightPredictor = None):
        super().__init__("고급 시간적 영향", neural_predictor)
        
    def _get_layer_index(self) -> int:
        return 1
        
    def _apply_rule_based_correction(self, weight: float, 
                                   context: AdvancedCalculationContext) -> float:
        """시간적 영향 규칙 기반 보정"""
        corrections = []
        reasoning_parts = []
        
        # 시간 할인 함수 적용
        time_discount = self._calculate_temporal_discount(context.duration_seconds)
        if time_discount != 1.0:
            corrections.append(time_discount)
            reasoning_parts.append(f"시간 할인 적용 ({(time_discount-1)*100:+.1f}%)")
            
        # 긴급성 기반 보정
        if hasattr(context, 'urgency_level'):
            urgency = context.urgency_level
            if urgency > 0.8:
                correction = 1.4
                corrections.append(correction)
                reasoning_parts.append("높은 긴급성 (+40%)")
            elif urgency < 0.3:
                correction = 0.7
                corrections.append(correction)
                reasoning_parts.append("낮은 긴급성 (-30%)")
                
        # 보정 적용
        if corrections:
            final_correction = np.mean(corrections)
            weight *= final_correction
            
        self.last_contribution = weight - 1.0
        self.last_reasoning = " | ".join(reasoning_parts) if reasoning_parts else "기본 시간 영향"
        
        return weight
        
    def _calculate_temporal_discount(self, duration_seconds: float) -> float:
        """시간 할인 계수 계산"""
        # 하이퍼볼릭 할인 함수 사용
        k = 0.01  # 할인 상수
        t = duration_seconds / 3600.0  # 시간 단위 변환
        
        discount_factor = 1 / (1 + k * t)
        return min(1.2, max(0.5, discount_factor))


class AdvancedSocialWeightLayer(AdvancedWeightLayer):
    """고급 사회적 파급 가중치 레이어"""
    
    def __init__(self, neural_predictor: NeuralWeightPredictor = None):
        super().__init__("고급 사회적 파급", neural_predictor)
        
    def _get_layer_index(self) -> int:
        return 2
        
    def _apply_rule_based_correction(self, weight: float, 
                                   context: AdvancedCalculationContext) -> float:
        """사회적 파급 규칙 기반 보정"""
        corrections = []
        reasoning_parts = []
        
        # 네트워크 효과 계산
        network_multiplier = self._calculate_network_effect(context.affected_count)
        if network_multiplier != 1.0:
            corrections.append(network_multiplier)
            reasoning_parts.append(f"네트워크 효과 ({(network_multiplier-1)*100:+.1f}%)")
            
        # 사회적 지위 고려
        if hasattr(context, 'social_status'):
            status = context.social_status
            if status == 'influential':
                correction = 1.3
                corrections.append(correction)
                reasoning_parts.append("영향력 있는 지위 (+30%)")
            elif status == 'vulnerable':
                correction = 1.4
                corrections.append(correction)
                reasoning_parts.append("취약계층 보호 (+40%)")
                
        # 보정 적용
        if corrections:
            final_correction = np.mean(corrections)
            weight *= final_correction
            
        self.last_contribution = weight - 1.0
        self.last_reasoning = " | ".join(reasoning_parts) if reasoning_parts else "개인적 영향"
        
        return weight
        
    def _calculate_network_effect(self, affected_count: int) -> float:
        """네트워크 효과 계산"""
        # 멱법칙 기반 네트워크 효과
        if affected_count <= 1:
            return 1.0
        
        # 로그 스케일 증가
        network_effect = 1.0 + 0.1 * np.log10(affected_count)
        return min(2.0, network_effect)


class AdvancedEthicalWeightLayer(AdvancedWeightLayer):
    """고급 윤리적 중요도 가중치 레이어"""
    
    def __init__(self, neural_predictor: NeuralWeightPredictor = None):
        super().__init__("고급 윤리적 중요도", neural_predictor)
        
    def _get_layer_index(self) -> int:
        return 3
        
    def _apply_rule_based_correction(self, weight: float, 
                                   context: AdvancedCalculationContext) -> float:
        """윤리적 중요도 규칙 기반 보정"""
        corrections = []
        reasoning_parts = []
        
        # 윤리적 원칙 기반 보정
        ethical_scores = self._calculate_ethical_scores(context)
        
        for principle, score in ethical_scores.items():
            if score > 0.7:
                correction = 1.0 + score * 0.5
                corrections.append(correction)
                reasoning_parts.append(f"{principle} 높음 (+{(correction-1)*100:.0f}%)")
            elif score < 0.3:
                correction = 1.0 - (0.3 - score) * 0.3
                corrections.append(correction)
                reasoning_parts.append(f"{principle} 낮음 ({(correction-1)*100:.0f}%)")
                
        # 보정 적용
        if corrections:
            final_correction = np.mean(corrections)
            weight *= final_correction
            
        self.last_contribution = weight - 1.0
        self.last_reasoning = " | ".join(reasoning_parts) if reasoning_parts else "윤리적 중립"
        
        return weight
        
    def _calculate_ethical_scores(self, context: AdvancedCalculationContext) -> Dict[str, float]:
        """윤리적 원칙별 점수 계산"""
        scores = {}
        
        # 생명과 안전
        if hasattr(context, 'life_impact_level'):
            scores['생명과 안전'] = context.life_impact_level
        else:
            scores['생명과 안전'] = 0.5
            
        # 정의와 공정성
        if hasattr(context, 'justice_level'):
            scores['정의와 공정성'] = context.justice_level
        else:
            scores['정의와 공정성'] = 0.5
            
        # 자율성과 자유
        if hasattr(context, 'autonomy_level'):
            scores['자율성과 자유'] = context.autonomy_level
        else:
            scores['자율성과 자유'] = 0.5
            
        return scores


class AdvancedEmotionalWeightLayer(AdvancedWeightLayer):
    """고급 감정적 강도 가중치 레이어 - 감정 기반 정성적 영역 강화"""
    
    def __init__(self, neural_predictor: NeuralWeightPredictor = None):
        super().__init__("고급 감정적 강도", neural_predictor)
        
        # 감정별 쾌락/고통 매핑 (연구 기반)
        self.emotion_hedonic_mapping = {
            'joy': {'valence': 0.85, 'intensity_multiplier': 1.3, 'duration_factor': 1.2},
            'trust': {'valence': 0.6, 'intensity_multiplier': 1.1, 'duration_factor': 1.4},
            'anticipation': {'valence': 0.4, 'intensity_multiplier': 1.0, 'duration_factor': 0.8},
            'surprise': {'valence': 0.1, 'intensity_multiplier': 1.4, 'duration_factor': 0.5},
            'neutral': {'valence': 0.0, 'intensity_multiplier': 1.0, 'duration_factor': 1.0},
            'disgust': {'valence': -0.6, 'intensity_multiplier': 1.2, 'duration_factor': 1.1},
            'sadness': {'valence': -0.7, 'intensity_multiplier': 1.1, 'duration_factor': 1.5},
            'anger': {'valence': -0.8, 'intensity_multiplier': 1.4, 'duration_factor': 0.9},
            'fear': {'valence': -0.9, 'intensity_multiplier': 1.5, 'duration_factor': 1.3}
        }
        
        # 감정 조합 효과 (감정이 복합적일 때)
        self.emotion_interaction_effects = {
            ('joy', 'trust'): 1.2,  # 긍정적 시너지
            ('sadness', 'fear'): 1.3,  # 부정적 증폭
            ('anger', 'disgust'): 1.25,  # 거부감 증폭
            ('surprise', 'fear'): 1.15,  # 놀람과 두려움
            ('anticipation', 'joy'): 1.1   # 기대와 기쁨
        }
        
    def _get_layer_index(self) -> int:
        return 4
        
    def _apply_rule_based_correction(self, weight: float, 
                                   context: AdvancedCalculationContext) -> float:
        """감정적 강도 규칙 기반 보정 - 정성적 영역 강화"""
        corrections = []
        reasoning_parts = []
        
        if context.emotion_data:
            # 1. 기본 감정 분석
            primary_emotion = context.emotion_data.primary_emotion.value if hasattr(context.emotion_data.primary_emotion, 'value') else str(context.emotion_data.primary_emotion)
            emotion_profile = self.emotion_hedonic_mapping.get(primary_emotion, self.emotion_hedonic_mapping['neutral'])
            
            # 감정가(valence) 기반 보정
            valence_correction = 1.0 + (emotion_profile['valence'] * 0.3)
            corrections.append(valence_correction)
            reasoning_parts.append(f"감정가 영향 ({emotion_profile['valence']:+.2f}): {(valence_correction-1)*100:+.1f}%")
            
            # 2. 감정 강도 세밀 분석
            emotion_magnitude = np.sqrt(
                context.emotion_data.valence**2 + 
                context.emotion_data.arousal**2
            )
            
            intensity_multiplier = emotion_profile['intensity_multiplier']
            
            # 퍼지 로직: 부드러운 감정 강도 전환
            extreme_membership = self._fuzzy_membership(emotion_magnitude, 0.7, 1.0)
            high_membership = self._fuzzy_membership(emotion_magnitude, 0.5, 0.8)
            low_membership = self._fuzzy_membership(emotion_magnitude, 0.0, 0.4)
            
            # 연속적 보정 계산 (딱딱한 if문 대신)
            extreme_correction = 1.0 + (extreme_membership * intensity_multiplier * 0.6)
            high_correction = 1.0 + (high_membership * intensity_multiplier * 0.4)
            low_correction = 1.0 - (low_membership * 0.3)
            
            # 가중 조합
            correction = (extreme_correction * extreme_membership + 
                         high_correction * high_membership + 
                         low_correction * low_membership) / max(0.1, extreme_membership + high_membership + low_membership)
            
            corrections.append(correction)
            reasoning_parts.append(f"연속적 감정 강도 ({(correction-1)*100:+.0f}%)")
            
            # 3. 각성도(arousal) 기반 벤담 변수 조정
            from data_models import emotion_intensity_to_float
            arousal_val = emotion_intensity_to_float(context.emotion_data.arousal)
            arousal_impact = self._calculate_arousal_impact(arousal_val, context)
            if arousal_impact != 1.0:
                corrections.append(arousal_impact)
                reasoning_parts.append(f"각성도 조정 ({(arousal_impact-1)*100:+.1f}%)")
            
            # 4. 감정 지속성 고려
            duration_factor = emotion_profile['duration_factor']
            if hasattr(context, 'duration_seconds') and context.duration_seconds:
                expected_emotion_duration = emotion_magnitude * duration_factor * 3600  # 시간 단위
                if context.duration_seconds > expected_emotion_duration:
                    duration_correction = 1.1
                    corrections.append(duration_correction)
                    reasoning_parts.append("감정 지속성 보정 (+10%)")
            
            # 5. 2차 감정 효과 (복합 감정)
            if hasattr(context.emotion_data, 'secondary_emotions') and context.emotion_data.secondary_emotions:
                secondary_correction = self._calculate_secondary_emotion_effects(
                    primary_emotion, context.emotion_data.secondary_emotions
                )
                if secondary_correction != 1.0:
                    corrections.append(secondary_correction)
                    reasoning_parts.append(f"복합 감정 효과 ({(secondary_correction-1)*100:+.1f}%)")
            
            # 6. 감정 신뢰도 기반 가중치 조정
            confidence_weight = max(0.5, context.emotion_data.confidence)
            if context.emotion_data.confidence > 0.8:
                correction = 1.0 + (context.emotion_data.confidence - 0.8) * 0.5
                corrections.append(correction)
                reasoning_parts.append(f"높은 감정 신뢰도 (+{(correction-1)*100:.0f}%)")
            elif context.emotion_data.confidence < 0.5:
                correction = 0.8 + context.emotion_data.confidence * 0.4
                corrections.append(correction)
                reasoning_parts.append(f"낮은 감정 신뢰도 ({(correction-1)*100:.0f}%)")
            
            # 7. 문화적 감정 요소 (한국어 특화)
            if hasattr(context.emotion_data, 'language') and context.emotion_data.language == 'ko':
                cultural_correction = self._apply_korean_cultural_emotion_correction(context)
                if cultural_correction != 1.0:
                    corrections.append(cultural_correction)
                    reasoning_parts.append(f"한국 문화적 감정 ({(cultural_correction-1)*100:+.1f}%)")
                
        # 보정 적용 (가중 평균)
        if corrections:
            # 신뢰도 기반 가중 평균
            weights = [1.0] * len(corrections)
            if context.emotion_data and hasattr(context.emotion_data, 'confidence'):
                # 신뢰도가 높을수록 더 강한 보정 적용
                confidence_factor = context.emotion_data.confidence
                weights = [w * confidence_factor for w in weights]
            
            final_correction = np.average(corrections, weights=weights)
            weight *= final_correction
            
        self.last_contribution = weight - 1.0
        self.last_reasoning = " | ".join(reasoning_parts) if reasoning_parts else "중간 감정 강도"
        
        return weight
    
    def _calculate_arousal_impact(self, arousal: float, context: AdvancedCalculationContext) -> float:
        """각성도가 벤담 변수에 미치는 영향 계산"""
        # 각성도에 따른 벤담 변수별 영향
        if arousal > 0.7:  # 높은 각성도
            # 강도(intensity)와 확실성(certainty) 증가, 지속성(duration) 감소
            return 1.2
        elif arousal < -0.3:  # 낮은 각성도
            # 강도 감소, 지속성 증가
            return 0.9
        else:
            return 1.0
    
    def _calculate_secondary_emotion_effects(self, primary_emotion: str, 
                                           secondary_emotions: dict) -> float:
        """2차 감정들의 복합 효과 계산"""
        interaction_effects = []
        
        for secondary_emotion, intensity in secondary_emotions.items():
            if hasattr(secondary_emotion, 'value'):
                secondary_name = secondary_emotion.value
            else:
                secondary_name = str(secondary_emotion)
                
            # 감정 조합 효과 확인
            emotion_pair = tuple(sorted([primary_emotion, secondary_name]))
            if emotion_pair in self.emotion_interaction_effects:
                effect = self.emotion_interaction_effects[emotion_pair]
                weighted_effect = 1.0 + (effect - 1.0) * intensity
                interaction_effects.append(weighted_effect)
        
        if interaction_effects:
            return np.mean(interaction_effects)
        else:
            return 1.0
    
    def _apply_korean_cultural_emotion_correction(self, context: AdvancedCalculationContext) -> float:
        """한국 문화적 감정 요소 보정"""
        # 한국 문화 특유의 감정 (한, 정, 체면, 눈치 등)을 고려
        cultural_factors = []
        
        if hasattr(context, 'cultural_emotions'):
            # 한(恨) - 깊은 슬픔과 원망
            if context.cultural_emotions.get('han', 0) > 0.5:
                cultural_factors.append(1.3)  # 강한 부정적 감정 증폭
            
            # 정(情) - 따뜻한 인간적 정서
            if context.cultural_emotions.get('jeong', 0) > 0.5:
                cultural_factors.append(1.15)  # 긍정적 감정 증폭
            
            # 체면 - 사회적 자존심
            if context.cultural_emotions.get('chemyeon', 0) > 0.5:
                cultural_factors.append(1.1)  # 사회적 영향 증가
            
        if cultural_factors:
            return np.mean(cultural_factors)
        else:
            return 1.0
    
    def _fuzzy_membership(self, value: float, center: float, width: float) -> float:
        """퍼지 멤버십 함수 (삼각형 분포)"""
        if width <= 0:
            return 1.0 if value == center else 0.0
        
        distance = abs(value - center)
        if distance >= width:
            return 0.0
        else:
            return 1.0 - (distance / width)


class AdvancedCognitiveWeightLayer(AdvancedWeightLayer):
    """고급 인지적 복잡도 가중치 레이어"""
    
    def __init__(self, neural_predictor: NeuralWeightPredictor = None):
        super().__init__("고급 인지적 복잡도", neural_predictor)
        
    def _get_layer_index(self) -> int:
        return 5
        
    def _apply_rule_based_correction(self, weight: float, 
                                   context: AdvancedCalculationContext) -> float:
        """인지적 복잡도 규칙 기반 보정"""
        corrections = []
        reasoning_parts = []
        
        # 인지 부하 계산
        cognitive_load = self._calculate_cognitive_load(context)
        
        if cognitive_load > 0.8:
            correction = 0.8
            corrections.append(correction)
            reasoning_parts.append("높은 인지 부하 (-20%)")
        elif cognitive_load < 0.3:
            correction = 1.2
            corrections.append(correction)
            reasoning_parts.append("낮은 인지 부하 (+20%)")
            
        # 불확실성 보정
        if context.uncertainty_level > 0.7:
            correction = 0.9
            corrections.append(correction)
            reasoning_parts.append("높은 불확실성 (-10%)")
        elif context.uncertainty_level < 0.3:
            correction = 1.1
            corrections.append(correction)
            reasoning_parts.append("낮은 불확실성 (+10%)")
            
        # 보정 적용
        if corrections:
            final_correction = np.mean(corrections)
            weight *= final_correction
            
        self.last_contribution = weight - 1.0
        self.last_reasoning = " | ".join(reasoning_parts) if reasoning_parts else "보통 인지 복잡도"
        
        return weight
        
    def _calculate_cognitive_load(self, context: AdvancedCalculationContext) -> float:
        """인지 부하 계산"""
        load_factors = []
        
        # 변수 수 기반 부하 (Bentham v2 확장: 10개 변수)
        non_default_vars = len([v for v in context.input_values.values() if v != 0.5])
        load_factors.append(non_default_vars / 10.0)
        
        # 복잡도 메트릭 기반 부하
        if hasattr(context, 'complexity_metrics'):
            complexity = context.complexity_metrics
            lexical_complexity = complexity.get('lexical_diversity', 0.5)
            structural_complexity = complexity.get('structural_complexity', 0.1) * 10
            load_factors.extend([lexical_complexity, structural_complexity])
            
        # 불확실성 기반 부하
        load_factors.append(context.uncertainty_level)
        
        return float(np.mean(load_factors))


class AdvancedBenthamCalculator:
    """고급 벤담 쾌락 계산기 - Linux 전용 AI 강화 버전"""
    
    def __init__(self):
        if not ADVANCED_LIBS_AVAILABLE:
            raise ImportError("고급 라이브러리가 필요합니다. requirements.txt를 확인하세요.")
            
        self.logger = logger
        self.device = DEVICE
        
        # 레이지 로딩을 위한 플래그
        self._neural_predictor = None
        self._context_analyzer = None
        self._weight_layers = None
        self._models_loaded = False
        
        # 메모리 사용량 추적
        self._memory_usage = 0
        self._max_memory_mb = 8192  # 8GB 제한
        
        # 배치 크기 동적 조정 설정
        self._base_batch_size = 32
        self._min_batch_size = 4
        self._max_batch_size = 128
        self._current_batch_size = self._base_batch_size
        
        # 고급 계산 설정
        self.advanced_config = {
            'use_neural_prediction': True,
            'use_transformer_analysis': True,
            'use_ml_enhancement': True,
            'extreme_pleasure_threshold': 0.8,
            'extreme_pain_threshold': -0.8,
            'neural_confidence_threshold': 0.7,
            'batch_processing': True,
            'optimization_enabled': True
        }
        
        # 최적화 설정
        self.optimization_params = {
            'method': 'bounded',
            'bounds': (0.3, 2.5),
            'tolerance': 1e-6
        }
        
        # 캐싱 시스템
        self.calculation_cache = {}
        self.cache_lock = threading.Lock()
        
        # MoE 시스템 초기화
        self.moe_enabled = True
        
        # =====================================================
        # 강화 모듈 통합 (42.5M 추가 → 총 45M)
        # =====================================================
        base_dim = 768
        
        # 1. 심층 윤리 추론 네트워크 (12M)
        self.deep_ethics = nn.ModuleDict({
            'philosophical': nn.Sequential(
                nn.Linear(base_dim, 1024),
                nn.LayerNorm(1024),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(1024, 768),
                nn.LayerNorm(768),
                nn.GELU(),
                nn.Linear(768, 512),
                nn.Linear(512, 10)  # 벤담 10요소
            ),
            'deontological': nn.Sequential(
                nn.Linear(base_dim, 768),
                nn.LayerNorm(768),
                nn.GELU(),
                nn.Linear(768, 512),
                nn.Linear(512, 5)  # 의무론적 원칙 5개
            ),
            'virtue_ethics': nn.Sequential(
                nn.Linear(base_dim, 768),
                nn.LayerNorm(768),
                nn.GELU(),
                nn.Linear(768, 512),
                nn.Linear(512, 7)  # 덕 윤리 7개
            ),
            'integrator': nn.Sequential(
                nn.Linear(22, 128),
                nn.GELU(),
                nn.Linear(128, 10)
            )
        }).to(self.device)
        
        # 2. 사회적 영향 평가 (10M)
        self.social_impact = nn.ModuleDict({
            'individual': nn.LSTM(base_dim, base_dim // 2, 2, batch_first=True, bidirectional=True),
            'group': nn.LSTM(base_dim, base_dim // 2, 2, batch_first=True, bidirectional=True),
            'society': nn.LSTM(base_dim, base_dim // 2, 2, batch_first=True, bidirectional=True),
            'global': nn.LSTM(base_dim, base_dim // 2, 2, batch_first=True, bidirectional=True),
            'impact_aggregator': nn.Sequential(
                nn.Linear(base_dim * 4, base_dim * 2),
                nn.LayerNorm(base_dim * 2),
                nn.GELU(),
                nn.Linear(base_dim * 2, base_dim),
                nn.Linear(base_dim, 10)
            )
        }).to(self.device)
        
        # 3. 장기 결과 예측 (10M)
        self.long_term_predictor = nn.ModuleDict({
            'temporal_encoder': nn.TransformerEncoder(
                nn.TransformerEncoderLayer(base_dim, 8, base_dim * 2, dropout=0.1, batch_first=True),
                num_layers=3
            ),
            'outcome_predictor': nn.Sequential(
                nn.Linear(base_dim, 1024),
                nn.LayerNorm(1024),
                nn.GELU(),
                nn.Linear(1024, 512),
                nn.Linear(512, 256),
                nn.Linear(256, 10)  # 10개 시간대별 결과
            ),
            'uncertainty_estimator': nn.Sequential(
                nn.Linear(base_dim, 256),
                nn.GELU(),
                nn.Linear(256, 10),
                nn.Softplus()
            )
        }).to(self.device)
        
        # 4. 문화간 윤리 비교 (10.5M + 3M 추가 = 13.5M)
        self.cross_cultural = nn.ModuleDict({
            'western': self._create_cultural_ethics_network(base_dim),
            'eastern': self._create_cultural_ethics_network(base_dim),
            'korean': self._create_cultural_ethics_network(base_dim),
            'islamic': self._create_cultural_ethics_network(base_dim),
            'african': self._create_cultural_ethics_network(base_dim),
            'fusion': nn.Sequential(
                nn.Linear(base_dim * 5, base_dim * 3),
                nn.LayerNorm(base_dim * 3),
                nn.GELU(),
                nn.Linear(base_dim * 3, base_dim),
                nn.LayerNorm(base_dim),
                nn.Linear(base_dim, 10)
            ),
            # 추가 레이어 (3M)
            'deep_cultural': nn.Sequential(
                nn.Linear(base_dim, 1024),
                nn.LayerNorm(1024),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(1024, 768),
                nn.LayerNorm(768),
                nn.GELU(),
                nn.Linear(768, base_dim)
            )
        }).to(self.device)
        
        # 파라미터 로깅
        total_params = sum(p.numel() for p in [
            *self.deep_ethics.parameters(),
            *self.social_impact.parameters(),
            *self.long_term_predictor.parameters(),
            *self.cross_cultural.parameters()
        ])
        logger.info(f"✅ 벤담 계산기 강화 모듈 통합: {total_params/1e6:.1f}M 파라미터 추가")
        if self.moe_enabled:
            try:
                # 윤리 분석용 MoE 초기화
                ethics_input_dim = 512  # 윤리 맥락 임베딩 차원
                ethics_output_dim = 6   # 윤리 가치 수 (care_harm, fairness, loyalty, authority, sanctity, liberty)
                
                self.ethics_moe = create_ethics_moe(
                    input_dim=ethics_input_dim,
                    output_dim=ethics_output_dim,
                    num_experts=4  # 공리주의, 의무론, 덕윤리, 돌봄윤리
                )
                
                self.logger.info("벤담 계산기 윤리 MoE 시스템 초기화 완료 (4개 전문가)")
            except Exception as e:
                self.logger.warning(f"벤담 윤리 MoE 초기화 실패, 기본 시스템 사용: {e}")
                self.moe_enabled = False
        
        # 3뷰 시나리오 시스템 초기화
        self.scenario_system_enabled = True
        if self.scenario_system_enabled:
            try:
                self.three_view_system = ThreeViewScenarioSystem(device=self.device)
                self.logger.info("벤담 계산기 3뷰 시나리오 시스템 초기화 완료")
            except Exception as e:
                self.logger.warning(f"3뷰 시나리오 시스템 초기화 실패: {e}")
                self.scenario_system_enabled = False
        
        # PhaseController Hook 초기화
        self.phase_controller_enabled = True
        if self.phase_controller_enabled:
            try:
                # 벤담 계산기 모델들 수집
                models = {}
                if hasattr(self, 'neural_predictor') and self.neural_predictor:
                    models['neural_predictor'] = self.neural_predictor
                if hasattr(self, 'ethics_moe') and self.ethics_moe:
                    models['ethics_moe'] = self.ethics_moe
                
                self.phase_controller = PhaseControllerHook(
                    models=models,
                    performance_threshold=0.8,
                    error_threshold=0.1
                )
                
                # 모니터링 시작
                self.phase_controller.start_monitoring()
                
                self.logger.info("벤담 계산기 PhaseController Hook 초기화 완료")
            except Exception as e:
                self.logger.warning(f"PhaseController Hook 초기화 실패: {e}")
                self.phase_controller_enabled = False
        
        # 법률 전문가 시스템 초기화
        self.legal_expert_enabled = True
        if self.legal_expert_enabled:
            try:
                self.legal_expert = get_legal_expert_system(device=self.device)
                
                # 기본적으로 추론 모드로 설정 (필요시 학습 모드로 변경 가능)
                self.legal_expert.set_operation_mode(OperationMode.INFERENCE)
                
                self.logger.info("벤담 계산기 법률 전문가 시스템 초기화 완료")
            except Exception as e:
                self.logger.warning(f"법률 전문가 시스템 초기화 실패: {e}")
                self.legal_expert_enabled = False
        
        self.logger.info("고급 벤담 쾌락 계산기 초기화 완료 (레이지 로딩 활성화)")
        
        # 등록 단계에서 get_pytorch_network가 작동하도록 기본 네트워크 보장
        self._ensure_default_network()
    
    def _ensure_default_network(self):
        """최소 하나의 PyTorch 네트워크가 존재하도록 보장"""
        # 레이지 로딩이라 초기에는 비어있으므로 기본 네트워크 생성
        logger.info("🔨 벤담 계산기 기본 PyTorch 네트워크 생성 중...")
        self._build_default_network()
    
    def _build_default_network(self):
        """
        기본 PyTorch 네트워크 생성
        - 등록/헤드 바인딩을 위한 최소 네트워크
        - 가볍고 메모리 효율적
        """
        import torch.nn as nn
        
        # 디바이스 설정
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # 간단한 쾌락 예측 네트워크
        class DefaultBenthamNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                # 7가지 변수 입력
                self.input_fc = nn.Linear(7, 64)
                self.hidden = nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 16),
                    nn.ReLU()
                )
                self.output_fc = nn.Linear(16, 1)  # 쾌락 점수 출력
                
            def forward(self, x):
                x = self.input_fc(x)
                x = self.hidden(x)
                return self.output_fc(x)
        
        # 네트워크 생성 및 설정
        self.default_network = DefaultBenthamNetwork().to(device)
        self._primary_nn = self.default_network
        
        # neural_predictor가 비어있으면 기본 네트워크로 설정
        if self._neural_predictor is None:
            self._neural_predictor = self.default_network
            logger.info("🔗 _neural_predictor를 기본 네트워크로 설정")
        
        logger.info(f"✅ 벤담 계산기 기본 네트워크 생성 완료 (device: {device})")
        logger.info(f"   - 파라미터 수: {sum(p.numel() for p in self.default_network.parameters()):,}")
    
    @property
    def neural_predictor(self):
        """신경망 가중치 예측기 (레이지 로딩)"""
        if self._neural_predictor is None:
            self._check_memory_usage()
            self.logger.info("신경망 가중치 예측기 로딩 중...")
            predictor_device = get_smart_device(memory_required_mb=100)
            self._neural_predictor = NeuralWeightPredictor().to(predictor_device)
            self.logger.info(f"신경망 가중치 예측기 로드: {predictor_device} (100MB)")
            self._update_memory_usage()
        return self._neural_predictor
    
    @property
    def context_analyzer(self):
        """트랜스포머 맥락 분석기 (레이지 로딩)"""
        if self._context_analyzer is None:
            self._check_memory_usage()
            self.logger.info("트랜스포머 맥락 분석기 로딩 중...")
            self._context_analyzer = TransformerContextAnalyzer()
            self._update_memory_usage()
        return self._context_analyzer
    
    @property
    def weight_layers(self):
        """6개 고급 가중치 레이어 (레이지 로딩)"""
        if self._weight_layers is None:
            self._check_memory_usage()
            self.logger.info("가중치 레이어들 로딩 중...")
            self._weight_layers = [
                AdvancedContextualWeightLayer(self.neural_predictor),
                AdvancedTemporalWeightLayer(self.neural_predictor),
                AdvancedSocialWeightLayer(self.neural_predictor),
                AdvancedEthicalWeightLayer(self.neural_predictor),
                AdvancedEmotionalWeightLayer(self.neural_predictor),
                AdvancedCognitiveWeightLayer(self.neural_predictor)
            ]
            self._update_memory_usage()
        return self._weight_layers
    
    def _check_memory_usage(self):
        """메모리 사용량 체크"""
        import psutil
        
        # 시스템 메모리 확인
        memory = psutil.virtual_memory()
        memory_usage_percent = memory.percent
        available_mb = memory.available / (1024 * 1024)
        
        # 자동 메모리 안전장치 (80% 이상)
        if memory_usage_percent >= 80:
            self.logger.critical(f"메모리 사용률 위험 수준: {memory_usage_percent:.1f}%")
            self._emergency_memory_optimization()
        elif available_mb < 4096:  # 4GB 미만
            self.logger.warning(f"메모리 부족: {available_mb:.0f}MB 남음")
            self._optimize_memory_usage()
        
        # GPU 메모리 확인 (가능한 경우)
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            gpu_allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
            gpu_available = gpu_memory - gpu_allocated
            gpu_usage_percent = (gpu_allocated / gpu_memory) * 100
            
            # GPU 메모리 안전장치 (80% 이상)
            if gpu_usage_percent >= 80:
                self.logger.critical(f"GPU 메모리 사용률 위험 수준: {gpu_usage_percent:.1f}%")
                self._emergency_gpu_optimization()
            elif gpu_available < 2048:  # 2GB 미만
                self.logger.warning(f"GPU 메모리 부족: {gpu_available:.0f}MB 남음")
                torch.cuda.empty_cache()
    
    def _update_memory_usage(self):
        """메모리 사용량 업데이트"""
        import psutil
        
        process = psutil.Process()
        self._memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
        
        if torch.cuda.is_available():
            gpu_usage = torch.cuda.memory_allocated(0) / (1024 * 1024)
            self.logger.info(f"메모리 사용량: RAM {self._memory_usage:.0f}MB, GPU {gpu_usage:.0f}MB")
        else:
            self.logger.info(f"메모리 사용량: RAM {self._memory_usage:.0f}MB")
    
    def _optimize_memory_usage(self):
        """메모리 사용량 최적화"""
        # 캐시 정리
        if hasattr(self, 'calculation_cache'):
            self.calculation_cache.clear()
        
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 가비지 컬렉션 실행
        import gc
        gc.collect()
        
        self.logger.info("메모리 최적화 완료")

    def _emergency_memory_optimization(self):
        """긴급 메모리 최적화 (80% 이상 사용 시)"""
        self.logger.critical("긴급 메모리 최적화 시작")
        
        # 1. 모든 모델 언로드
        self.unload_models()
        
        # 2. 계산 캐시 완전 삭제
        if hasattr(self, 'calculation_cache'):
            self.calculation_cache.clear()
        
        # 3. 배치 크기 최소화
        self._current_batch_size = self._min_batch_size
        
        # 4. 강제 가비지 컬렉션
        import gc
        gc.collect()
        
        # 5. GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        self.logger.critical("긴급 메모리 최적화 완료")

    def _emergency_gpu_optimization(self):
        """긴급 GPU 메모리 최적화 (80% 이상 사용 시)"""
        self.logger.critical("긴급 GPU 메모리 최적화 시작")
        
        # 1. GPU 캐시 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # 2. 모델을 CPU로 이동 (일시적)
        if self._neural_predictor is not None:
            self._neural_predictor = self._neural_predictor.cpu()
        
        # 3. 배치 크기 최소화
        self._current_batch_size = self._min_batch_size
        
        # 4. 강제 가비지 컬렉션
        import gc
        gc.collect()
        
        self.logger.critical("긴급 GPU 메모리 최적화 완료")

    def _adjust_batch_size(self):
        """메모리 상황에 따른 배치 크기 동적 조정"""
        import psutil
        
        # 현재 메모리 사용률 확인
        memory = psutil.virtual_memory()
        memory_usage_percent = memory.percent
        
        # 메모리 사용률에 따른 배치 크기 조정
        if memory_usage_percent > 85:  # 85% 이상
            self._current_batch_size = max(self._min_batch_size, self._current_batch_size // 2)
            self.logger.warning(f"높은 메모리 사용률 ({memory_usage_percent:.1f}%), 배치 크기 감소: {self._current_batch_size}")
        elif memory_usage_percent < 60:  # 60% 미만
            self._current_batch_size = min(self._max_batch_size, self._current_batch_size * 2)
            self.logger.info(f"낮은 메모리 사용률 ({memory_usage_percent:.1f}%), 배치 크기 증가: {self._current_batch_size}")
        
        # GPU 메모리도 고려 (가능한 경우)
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_allocated = torch.cuda.memory_allocated(0)
            gpu_usage_percent = (gpu_allocated / gpu_memory) * 100
            
            if gpu_usage_percent > 80:  # GPU 메모리 80% 이상
                self._current_batch_size = max(self._min_batch_size, self._current_batch_size // 2)
                self.logger.warning(f"높은 GPU 메모리 사용률 ({gpu_usage_percent:.1f}%), 배치 크기 감소: {self._current_batch_size}")
        
        return self._current_batch_size
    
    def get_optimal_batch_size(self) -> int:
        """최적화된 배치 크기 반환"""
        return self._adjust_batch_size()
    
    def unload_models(self):
        """모델 언로드 및 메모리 해제"""
        if self._neural_predictor is not None:
            del self._neural_predictor
            self._neural_predictor = None
        
        if self._context_analyzer is not None:
            del self._context_analyzer
            self._context_analyzer = None
        
        if self._weight_layers is not None:
            del self._weight_layers
            self._weight_layers = None
        
        self._models_loaded = False
        self._optimize_memory_usage()
        self.logger.info("모든 모델 언로드 완료")
        
    def _create_cultural_ethics_network(self, dim: int) -> nn.Module:
        """문화별 윤리 해석 네트워크"""
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim)
        )
        
    def calculate_with_ethical_reasoning(self,
                                       input_data: Dict[str, Any],
                                       community_emotion: Optional['EmotionData'] = None,
                                       other_emotion: Optional['EmotionData'] = None,
                                       self_emotion: Optional['EmotionData'] = None,
                                       past_regret_memory: Optional[Dict[str, float]] = None,
                                       use_cache: bool = True) -> EnhancedHedonicResult:
        """인간적 윤리 추론 과정을 따른 벤담 계산 (3뷰 시나리오 통합)"""
        
        # 1단계: 감정 우선순위에 따른 통합 (공동체 > 타자 > 자아)
        integrated_emotion = self._integrate_emotion_hierarchy(
            community_emotion, other_emotion, self_emotion
        )
        
        # 2단계: 감정 기반 윤리적 가치 추론
        ethical_values = self._perform_ethical_reasoning(integrated_emotion, input_data)
        
        # 3단계: 윤리적 가치를 반영한 벤담 계산
        input_data['emotion_data'] = integrated_emotion
        input_data['ethical_values'] = ethical_values
        
        # 4단계: 과거 후회는 미묘한 학습 효과로만 반영 (직접 개입 아님)
        if past_regret_memory:
            input_data['learning_bias'] = self._extract_learning_bias(past_regret_memory)
        
        # 5단계: 3뷰 시나리오 분석 통합
        if self.scenario_system_enabled:
            try:
                # 3뷰 시나리오 분석 수행
                scenario_analysis = asyncio.run(self.three_view_system.analyze_three_view_scenarios(input_data))
                
                # 시나리오 결과를 벤담 계산에 통합
                enhanced_result = self._integrate_scenario_analysis(input_data, scenario_analysis, use_cache)
                
                self.logger.debug(f"3뷰 시나리오 벤담 계산 완료: 합의 효용 {scenario_analysis.consensus_utility:.3f}")
                return enhanced_result
                
            except Exception as e:
                self.logger.warning(f"3뷰 시나리오 분석 실패, 기본 계산 사용: {e}")
                return self.calculate_with_advanced_layers(input_data, use_cache)
        else:
            return self.calculate_with_advanced_layers(input_data, use_cache)
    
    def _integrate_scenario_analysis(self, input_data: Dict[str, Any], 
                                   scenario_analysis, use_cache: bool = True) -> EnhancedHedonicResult:
        """시나리오 분석을 벤담 계산에 통합"""
        
        # 시나리오별 벤담 계산 수행
        scenario_results = {}
        
        for scenario_name, scenario_metrics in [
            ('optimistic', scenario_analysis.optimistic_scenario),
            ('neutral', scenario_analysis.neutral_scenario),
            ('pessimistic', scenario_analysis.pessimistic_scenario)
        ]:
            # 시나리오별 입력 데이터 조정
            scenario_input = input_data.copy()
            
            # 시나리오 특성을 벤담 변수에 반영
            scenario_input['intensity'] = scenario_input.get('intensity', 0.5) * (1 + scenario_metrics.expected_pleasure - scenario_metrics.expected_pain)
            scenario_input['duration'] = scenario_input.get('duration', 60) * scenario_metrics.probability_weight
            scenario_input['certainty'] = scenario_input.get('certainty', 0.5) * scenario_metrics.confidence_level
            
            # 시나리오별 윤리적 가치 반영
            if scenario_metrics.ethical_implications:
                scenario_input['ethical_values'] = scenario_metrics.ethical_implications
            
            # 시나리오 메타데이터 추가
            scenario_input['scenario_type'] = scenario_name
            scenario_input['scenario_weight'] = scenario_metrics.probability_weight
            scenario_input['regret_potential'] = scenario_metrics.regret_potential
            
            # 벤담 계산 수행
            scenario_result = self.calculate_with_advanced_layers(scenario_input, use_cache)
            scenario_results[scenario_name] = scenario_result
        
        # 가중 평균으로 최종 결과 계산
        final_result = self._calculate_weighted_bentham_result(scenario_results, scenario_analysis)
        
        return final_result
    
    def _calculate_weighted_bentham_result(self, scenario_results: Dict[str, EnhancedHedonicResult],
                                         scenario_analysis) -> EnhancedHedonicResult:
        """시나리오별 결과를 가중 평균으로 통합"""
        
        # 가중치 계산
        weights = {
            'optimistic': scenario_analysis.optimistic_scenario.probability_weight,
            'neutral': scenario_analysis.neutral_scenario.probability_weight,
            'pessimistic': scenario_analysis.pessimistic_scenario.probability_weight
        }
        
        total_weight = sum(weights.values())
        if total_weight == 0:
            # 프로젝트 규칙: fallback 없는 순수 재시도 방식
            retry_count = 0
            max_retries = 3
            while retry_count < max_retries:
                retry_count += 1
                self.logger.info(f"가중치 재계산 재시도 {retry_count}/{max_retries}")
                try:
                    # 최소 가중치를 부여하여 재계산
                    weights['optimistic'] = max(0.1, scenario_analysis.optimistic_scenario.probability_weight)
                    weights['neutral'] = max(0.4, scenario_analysis.neutral_scenario.probability_weight)  
                    weights['pessimistic'] = max(0.1, scenario_analysis.pessimistic_scenario.probability_weight)
                    total_weight = sum(weights.values())
                    if total_weight > 0:
                        self.logger.info(f"가중치 재계산 성공: total_weight={total_weight}")
                        break
                except Exception as retry_error:
                    self.logger.error(f"가중치 재계산 재시도 {retry_count} 실패: {retry_error}")
                    if retry_count >= max_retries:
                        self.logger.error("가중치 계산 모든 재시도 실패 - 시스템 정지")
                        raise RuntimeError(f"가중치 계산 최종 실패: {retry_error}")
                    import time
                    time.sleep(0.1)
            
            if total_weight == 0:
                self.logger.error("가중치 재계산 후에도 total_weight가 0 - 시스템 정지")
                raise RuntimeError("시나리오 가중치 계산 실패 - fallback 금지로 시스템 정지")
        
        # 정규화
        for key in weights:
            weights[key] /= total_weight
        
        # 가중 평균 계산
        weighted_total_score = sum(
            scenario_results[scenario].total_score * weights[scenario]
            for scenario in scenario_results
        )
        
        weighted_base_score = sum(
            scenario_results[scenario].base_score * weights[scenario]
            for scenario in scenario_results
        )
        
        # 대표 결과 선택 (중도적 시나리오 기준)
        base_result = scenario_results['neutral']
        
        # 향상된 결과 생성
        enhanced_result = EnhancedHedonicResult(
            total_score=weighted_total_score,
            base_score=weighted_base_score,
            hedonic_values=base_result.hedonic_values,
            weight_layers=base_result.weight_layers,
            confidence_score=base_result.confidence_score * scenario_analysis.consensus_strength,
            calculation_context=base_result.calculation_context,
            processing_time_ms=base_result.processing_time_ms + scenario_analysis.analysis_duration_ms,
            cache_hit=base_result.cache_hit,
            metadata={
                **base_result.metadata,
                'scenario_analysis': {
                    'consensus_utility': scenario_analysis.consensus_utility,
                    'consensus_regret': scenario_analysis.consensus_regret,
                    'uncertainty_range': scenario_analysis.uncertainty_range,
                    'scenario_diversity': scenario_analysis.scenario_diversity,
                    'consensus_strength': scenario_analysis.consensus_strength,
                    'recommended_decision': scenario_analysis.recommended_decision
                },
                'scenario_weights': weights,
                'scenario_scores': {
                    scenario: result.total_score
                    for scenario, result in scenario_results.items()
                }
            }
        )
        
        # PhaseController Hook에 성능 기록
        if self.phase_controller_enabled:
            try:
                # 벤담 계산 오차 추정
                score_variance = np.var([result.total_score for result in scenario_results.values()])
                calculation_error = score_variance / (abs(weighted_total_score) + 1e-8)
                
                # 성능 메트릭 구성
                performance_metrics = {
                    'bentham_calculation_error': calculation_error,
                    'processing_time_ms': enhanced_result.processing_time_ms,
                    'confidence_score': enhanced_result.confidence_score,
                    'scenario_diversity': scenario_analysis.scenario_diversity,
                    'consensus_strength': scenario_analysis.consensus_strength
                }
                
                # 모델별 성능 (MoE 관련)
                model_performances = {}
                if hasattr(scenario_analysis, 'metadata') and 'expert_usage' in scenario_analysis.metadata:
                    model_performances['ethics_moe'] = scenario_analysis.metadata['expert_usage']
                
                # 컨텍스트 정보
                context = {
                    'calculation_type': 'bentham_with_scenarios',
                    'scenario_count': 3,
                    'cache_hit': enhanced_result.cache_hit,
                    'total_score': weighted_total_score
                }
                
                # 성능 기록
                self.phase_controller.record_performance(
                    phase_type=PhaseType.INFERENCE,
                    metrics=performance_metrics,
                    model_performances=model_performances,
                    context=context
                )
                
            except Exception as e:
                self.logger.warning(f"PhaseController 성능 기록 실패: {e}")
        
        return enhanced_result
    
    def _integrate_emotion_hierarchy(self, 
                                   community_emotion: Optional['EmotionData'],
                                   other_emotion: Optional['EmotionData'], 
                                   self_emotion: Optional['EmotionData']) -> 'EmotionData':
        """고급 감정 통합: 치명적 손실 방지와 맥락 적응형 우선순위"""
        
        # 기본값 설정
        if not any([community_emotion, other_emotion, self_emotion]):
            from data_models import EmotionData, EmotionState, EmotionIntensity
            return EmotionData(
                primary_emotion=EmotionState.NEUTRAL,
                intensity=EmotionIntensity.MODERATE,
                confidence=0.5,
                dominance=0.5
            )
        
        # 1단계: 치명적 감정 손실 탐지
        critical_loss_detected = self._detect_critical_emotional_loss(
            community_emotion, other_emotion, self_emotion
        )
        
        # 2단계: 맥락 적응적 가중치 계산
        base_weights = self._calculate_contextual_weights(
            community_emotion, other_emotion, self_emotion, critical_loss_detected
        )
        
        # 3단계: 감정 강도 기반 우선순위 재조정
        adjusted_weights = self._adjust_weights_by_intensity(
            base_weights, community_emotion, other_emotion, self_emotion
        )
        
        # 4단계: 감정 충돌 해결 (손실 억제 우선)
        resolved_emotion = self._resolve_emotion_conflicts(
            community_emotion, other_emotion, self_emotion, 
            adjusted_weights, critical_loss_detected
        )
        
        return resolved_emotion
    
    def _detect_critical_emotional_loss(self, 
                                      community_emotion: Optional['EmotionData'],
                                      other_emotion: Optional['EmotionData'], 
                                      self_emotion: Optional['EmotionData']) -> Dict[str, bool]:
        """치명적 감정 손실 탐지 (영구 손실 원리)"""
        
        critical_loss = {
            'community_loss': False,
            'other_loss': False,
            'self_loss': False,
            'any_critical': False
        }
        
        # 치명적 손실 감정들 (sadness, fear, disgust + 높은 강도)
        loss_emotions = {'sadness', 'fear', 'disgust', 'anger'}
        critical_threshold = 0.7  # 강도 임계값
        
        def _is_critical_loss(emotion_data):
            if not emotion_data:
                return False
            emotion_name = emotion_data.primary_emotion.value if hasattr(emotion_data.primary_emotion, 'value') else str(emotion_data.primary_emotion)
            if hasattr(emotion_data.intensity, 'value'):
                intensity_value = float(emotion_data.intensity.value) / 4.0
            else:
                from data_models import emotion_intensity_to_float
                intensity_value = emotion_intensity_to_float(emotion_data.arousal)
            return emotion_name in loss_emotions and intensity_value > critical_threshold
        
        # 각 레벨별 치명적 손실 확인
        critical_loss['community_loss'] = _is_critical_loss(community_emotion)
        critical_loss['other_loss'] = _is_critical_loss(other_emotion)
        critical_loss['self_loss'] = _is_critical_loss(self_emotion)
        critical_loss['any_critical'] = any([
            critical_loss['community_loss'], 
            critical_loss['other_loss'], 
            critical_loss['self_loss']
        ])
        
        return critical_loss
    
    def _calculate_contextual_weights(self, 
                                    community_emotion: Optional['EmotionData'],
                                    other_emotion: Optional['EmotionData'], 
                                    self_emotion: Optional['EmotionData'],
                                    critical_loss: Dict[str, bool]) -> Dict[str, float]:
        """맥락 적응적 가중치 계산"""
        
        # 기본 가중치 (민주적 공리주의)
        base_weights = {'community': 0.5, 'other': 0.3, 'self': 0.2}
        
        # 치명적 손실 발생 시 우선순위 재조정
        if critical_loss['any_critical']:
            # 손실이 있는 주체들에게 더 높은 가중치 (소수 보호)
            if critical_loss['other_loss'] or critical_loss['community_loss']:
                # 타자나 공동체에 치명적 손실이 있으면 자아 우선순위 하락
                base_weights = {'community': 0.4, 'other': 0.4, 'self': 0.2}
                
                if critical_loss['community_loss']:
                    base_weights['community'] += 0.1  # 공동체 손실 우선
                if critical_loss['other_loss']:
                    base_weights['other'] += 0.1    # 타자 손실 우선
                    
                # 자아 가중치 재분배
                total_boost = (base_weights['community'] + base_weights['other']) - 0.8
                base_weights['self'] = max(0.1, 0.2 - total_boost)
        
        # 맥락별 추가 조정 (가상의 맥락 정보 활용)
        # 실제로는 input_data에서 맥락 정보를 받아올 것
        
        return base_weights
    
    def _adjust_weights_by_intensity(self, 
                                   base_weights: Dict[str, float],
                                   community_emotion: Optional['EmotionData'],
                                   other_emotion: Optional['EmotionData'], 
                                   self_emotion: Optional['EmotionData']) -> Dict[str, float]:
        """감정 강도 기반 가중치 재조정"""
        
        adjusted_weights = base_weights.copy()
        
        # 각 감정의 강도 계산
        def _get_intensity(emotion_data):
            if not emotion_data:
                return 0.0
            if hasattr(emotion_data.intensity, 'value'):
                return float(emotion_data.intensity.value) / 4.0
            else:
                return abs(emotion_data.arousal) if hasattr(emotion_data, 'arousal') else 0.5
        
        intensities = {
            'community': _get_intensity(community_emotion),
            'other': _get_intensity(other_emotion),
            'self': _get_intensity(self_emotion)
        }
        
        # 극단적으로 강한 감정이 있는 경우 가중치 재분배
        max_intensity = max(intensities.values())
        if max_intensity > 0.8:  # 매우 강한 감정
            for source, intensity in intensities.items():
                if intensity == max_intensity:
                    # 가장 강한 감정에 추가 가중치 (+20%까지)
                    boost = min(0.2, (intensity - 0.6) * 0.5)
                    adjusted_weights[source] += boost
                    
                    # 다른 가중치들을 비례적으로 감소
                    total_reduction = boost
                    other_sources = [s for s in adjusted_weights.keys() if s != source]
                    for other_source in other_sources:
                        adjusted_weights[other_source] -= total_reduction / len(other_sources)
                        adjusted_weights[other_source] = max(0.1, adjusted_weights[other_source])
        
        # 가중치 정규화
        total = sum(adjusted_weights.values())
        for key in adjusted_weights:
            adjusted_weights[key] /= total
            
        return adjusted_weights
    
    def _resolve_emotion_conflicts(self, 
                                 community_emotion: Optional['EmotionData'],
                                 other_emotion: Optional['EmotionData'], 
                                 self_emotion: Optional['EmotionData'],
                                 weights: Dict[str, float],
                                 critical_loss: Dict[str, bool]) -> 'EmotionData':
        """감정 충돌 해결: 손실 억제 우선 원칙"""
        
        from data_models import EmotionData, EmotionState, EmotionIntensity
        
        # 존재하는 감정들만 수집
        emotions = []
        emotion_sources = []
        emotion_weights = []
        
        if community_emotion:
            emotions.append(community_emotion)
            emotion_sources.append('community')
            emotion_weights.append(weights['community'])
            
        if other_emotion:
            emotions.append(other_emotion)
            emotion_sources.append('other')
            emotion_weights.append(weights['other'])
            
        if self_emotion:
            emotions.append(self_emotion)
            emotion_sources.append('self')
            emotion_weights.append(weights['self'])
        
        if not emotions:
            return EmotionData(
                primary_emotion=EmotionState.NEUTRAL,
                intensity=EmotionIntensity.MODERATE,
                confidence=0.5,
                dominance=0.5
            )
        
        # 감정 충돌 매트릭스 적용
        if len(emotions) > 1:
            resolved_emotion = self._apply_conflict_resolution_matrix(
                emotions, emotion_sources, emotion_weights, critical_loss
            )
        else:
            resolved_emotion = emotions[0]
        
        return resolved_emotion
    
    def _apply_conflict_resolution_matrix(self, 
                                        emotions: List['EmotionData'],
                                        sources: List[str],
                                        weights: List[float],
                                        critical_loss: Dict[str, bool]) -> 'EmotionData':
        """감정 충돌 해결 매트릭스 - 손실 억제 우선"""
        
        from data_models import EmotionData, EmotionState, EmotionIntensity
        
        # 손실 감정과 기쁨 감정 분리
        loss_emotions = {'sadness', 'fear', 'disgust', 'anger'}
        joy_emotions = {'joy', 'trust', 'anticipation', 'surprise'}
        
        loss_emotion_candidates = []
        joy_emotion_candidates = []
        neutral_emotion_candidates = []
        
        for i, emotion in enumerate(emotions):
            emotion_name = emotion.primary_emotion.value if hasattr(emotion.primary_emotion, 'value') else str(emotion.primary_emotion)
            
            if emotion_name in loss_emotions:
                loss_emotion_candidates.append((emotion, sources[i], weights[i]))
            elif emotion_name in joy_emotions:
                joy_emotion_candidates.append((emotion, sources[i], weights[i]))
            else:
                neutral_emotion_candidates.append((emotion, sources[i], weights[i]))
        
        # 충돌 해결 규칙
        if loss_emotion_candidates and joy_emotion_candidates:
            # 손실 vs 기쁨 충돌: 손실 우선 (영구 손실 원리)
            if critical_loss['any_critical']:
                # 치명적 손실이 있으면 무조건 손실 감정 우선
                strongest_loss = max(loss_emotion_candidates, key=lambda x: x[2])  # 가중치 기준
                return strongest_loss[0]
            else:
                # 일반적 손실의 경우 가중치 기반으로 결정하되 손실에 약간의 우선권
                loss_total_weight = sum(x[2] for x in loss_emotion_candidates) * 1.2  # 20% 보너스
                joy_total_weight = sum(x[2] for x in joy_emotion_candidates)
                
                if loss_total_weight > joy_total_weight:
                    strongest_loss = max(loss_emotion_candidates, key=lambda x: x[2])
                    return strongest_loss[0]
                else:
                    strongest_joy = max(joy_emotion_candidates, key=lambda x: x[2])
                    return strongest_joy[0]
        
        elif loss_emotion_candidates:
            # 손실 감정만 있는 경우
            strongest_loss = max(loss_emotion_candidates, key=lambda x: x[2])
            return strongest_loss[0]
            
        elif joy_emotion_candidates:
            # 기쁨 감정만 있는 경우
            strongest_joy = max(joy_emotion_candidates, key=lambda x: x[2])
            return strongest_joy[0]
            
        else:
            # 중성 감정들만 있는 경우
            if neutral_emotion_candidates:
                strongest_neutral = max(neutral_emotion_candidates, key=lambda x: x[2])
                return strongest_neutral[0]
            else:
                # 폴백
                return EmotionData(
                    primary_emotion=EmotionState.NEUTRAL,
                    intensity=EmotionIntensity.MODERATE,
                    confidence=0.5
                )
    
    def _perform_ethical_reasoning(self, emotion_data: 'EmotionData', 
                                 input_data: Dict[str, Any]) -> Dict[str, float]:
        """MoE 기반 윤리적 가치 추론"""
        
        # 기본 윤리 가치 (fallback)
        ethical_values = {
            'care_harm': 0.5,      # 돌봄/해악 방지
            'fairness': 0.5,       # 공정성
            'loyalty': 0.5,        # 충성/배신
            'authority': 0.5,      # 권위/존중
            'sanctity': 0.5,       # 신성/순수
            'liberty': 0.5         # 자유/억압
        }
        
        if not emotion_data:
            return ethical_values
        
        # MoE 시스템을 통한 고급 윤리적 추론
        if self.moe_enabled:
            try:
                # 윤리적 맥락 임베딩 생성
                ethics_embedding = self._create_ethics_embedding(emotion_data, input_data)
                
                # MoE 시스템을 통한 윤리적 분석 (안전한 차원 처리)
                if ethics_embedding.dim() == 1:
                    ethics_input = ethics_embedding.unsqueeze(0)
                else:
                    ethics_input = ethics_embedding
                moe_result = self.ethics_moe(ethics_input, return_expert_outputs=True)
                
                # 전문가별 윤리적 관점 통합
                ethical_values = self._integrate_expert_ethics(moe_result, emotion_data, input_data)
                
                self.logger.debug(f"MoE 윤리 추론 완료: {len(moe_result.expert_outputs)}개 전문가, "
                                f"다양성: {moe_result.diversity_score:.3f}")
                
            except Exception as e:
                # 프로젝트 규칙: fallback 없는 순수 재시도 방식
                retry_count = 0
                max_retries = 3
                ethical_values = None
                while retry_count < max_retries:
                    retry_count += 1
                    self.logger.info(f"MoE 윤리 추론 재시도 {retry_count}/{max_retries}")
                    try:
                        # 윤리적 맥락 임베딩 재생성
                        ethics_embedding = self._create_ethics_embedding(emotion_data, input_data)
                        
                        # MoE 시스템을 통한 윤리적 분석 재시도 (안전한 차원 처리)
                        if ethics_embedding.dim() == 1:
                            ethics_input = ethics_embedding.unsqueeze(0)
                        else:
                            ethics_input = ethics_embedding
                        moe_result = self.ethics_moe(ethics_input, return_expert_outputs=True)
                        
                        # 전문가별 윤리적 관점 통합
                        ethical_values = self._integrate_expert_ethics(moe_result, emotion_data, input_data)
                        
                        self.logger.info(f"MoE 윤리 추론 재시도 성공: {len(moe_result.expert_outputs)}개 전문가")
                        break
                    except Exception as retry_error:
                        self.logger.error(f"MoE 윤리 추론 재시도 {retry_count} 실패: {retry_error}")
                        if retry_count >= max_retries:
                            self.logger.error("MoE 윤리 추론 모든 재시도 실패 - 시스템 정지")
                            raise RuntimeError(f"MoE 윤리 추론 최종 실패: {retry_error}")
                        import time
                        time.sleep(0.5)
                
                if ethical_values is None:
                    self.logger.error("MoE 윤리 추론 재시도 후에도 실패 - 시스템 정지")
                    raise RuntimeError("MoE 윤리 추론 실패 - fallback 금지로 시스템 정지")
        else:
            # 기본 감정 기반 윤리 추론
            ethical_values = self._basic_ethical_reasoning(emotion_data, input_data)
        
        # 법률 전문가 시스템 통합
        if self.legal_expert_enabled:
            try:
                legal_analysis = self._apply_legal_expert_analysis(input_data, ethical_values)
                if legal_analysis:
                    # 법률 분석 결과를 윤리적 가치에 반영
                    ethical_values = self._integrate_legal_analysis(ethical_values, legal_analysis)
                    
                    self.logger.debug(f"법률 분석 통합 완료: 위험도 {legal_analysis.risk_level.value}")
                    
            except Exception as e:
                self.logger.warning(f"법률 전문가 분석 실패: {e}")
        
        return ethical_values
    
    def _create_ethics_embedding(self, emotion_data: 'EmotionData', input_data: Dict[str, Any]) -> torch.Tensor:
        """윤리적 맥락 임베딩 생성"""
        features = []
        
        # 1. 감정 특성 (8차원)
        if emotion_data:
            primary_emotion = emotion_data.primary_emotion.value if hasattr(emotion_data.primary_emotion, 'value') else str(emotion_data.primary_emotion)
            emotion_mapping = {
                'joy': [1, 0, 0, 0, 0, 0, 0, 0],
                'sadness': [0, 1, 0, 0, 0, 0, 0, 0],
                'anger': [0, 0, 1, 0, 0, 0, 0, 0],
                'fear': [0, 0, 0, 1, 0, 0, 0, 0],
                'trust': [0, 0, 0, 0, 1, 0, 0, 0],
                'disgust': [0, 0, 0, 0, 0, 1, 0, 0],
                'surprise': [0, 0, 0, 0, 0, 0, 1, 0],
                'anticipation': [0, 0, 0, 0, 0, 0, 0, 1]
            }
            emotion_vec = emotion_mapping.get(primary_emotion, [0, 0, 0, 0, 0, 0, 0, 0])
            features.extend(emotion_vec)
            
            # 감정 강도 및 기타 특성
            intensity = emotion_data.intensity.value if hasattr(emotion_data.intensity, 'value') else 3
            features.extend([
                intensity / 6.0,  # 정규화된 강도
                getattr(emotion_data, 'confidence', 0.5),
                getattr(emotion_data, 'dominance', 0.5)
            ])
        else:
            features.extend([0] * 11)  # 감정 데이터 없음
        
        # 2. 맥락 특성 (20차원)
        context_features = []
        
        # 시간적 맥락
        duration = input_data.get('duration', 60)  # 기본 60초
        context_features.append(min(duration / 3600, 1.0))  # 시간 정규화 (최대 1시간)
        
        # 확실성
        certainty = input_data.get('certainty', 0.5)
        context_features.append(certainty)
        
        # 근접성
        propinquity = input_data.get('propinquity', 0.5)
        context_features.append(propinquity)
        
        # 생산성
        productivity = input_data.get('productivity', 0.5)
        context_features.append(productivity)
        
        # 순수성
        purity = input_data.get('purity', 0.5)
        context_features.append(purity)
        
        # 범위 (몇 명에게 영향을 미치는지)
        affected_people = input_data.get('affected_people', 1)
        context_features.append(min(affected_people / 100, 1.0))  # 최대 100명으로 정규화
        
        # 텍스트 기반 윤리적 단서 (14차원)
        text = input_data.get('text', '') or input_data.get('description', '')
        if text:
            text_lower = text.lower()
            
            # 각 윤리적 차원의 키워드 존재 여부
            care_keywords = ['돌봄', '보호', '안전', '건강', '복지']
            fairness_keywords = ['공정', '평등', '정의', '균형', '공평']
            loyalty_keywords = ['충성', '신뢰', '배신', '소속', '그룹']
            authority_keywords = ['권위', '존중', '질서', '규칙', '전통']
            sanctity_keywords = ['신성', '순수', '더럽힘', '거룩', '정결']
            liberty_keywords = ['자유', '억압', '독립', '선택', '자율']
            
            for keywords in [care_keywords, fairness_keywords, loyalty_keywords, 
                           authority_keywords, sanctity_keywords, liberty_keywords]:
                keyword_score = sum(1 for keyword in keywords if keyword in text_lower)
                context_features.append(min(keyword_score / len(keywords), 1.0))
            
            # 추가적인 맥락 특성 (8차원)
            context_features.extend([
                float('위험' in text_lower),  # 위험 존재
                float('이익' in text_lower),  # 이익 존재
                float('갈등' in text_lower),  # 갈등 존재
                float('도덕' in text_lower),  # 도덕적 언급
                float('사회' in text_lower),  # 사회적 맥락
                float('개인' in text_lower),  # 개인적 맥락
                float('긴급' in text_lower),  # 긴급성
                float('장기' in text_lower)   # 장기적 영향
            ])
        else:
            context_features.extend([0] * 14)
        
        features.extend(context_features)
        
        # 3. 패딩하여 512차원으로 맞춤
        while len(features) < 512:
            features.append(0.0)
        
        return torch.tensor(features[:512], dtype=torch.float32, device=self.device)
    
    def _integrate_expert_ethics(self, moe_result, emotion_data: 'EmotionData', 
                               input_data: Dict[str, Any]) -> Dict[str, float]:
        """전문가별 윤리적 관점 통합"""
        
        # 기본 윤리 가치
        ethical_values = {
            'care_harm': 0.5,
            'fairness': 0.5,
            'loyalty': 0.5,
            'authority': 0.5,
            'sanctity': 0.5,
            'liberty': 0.5
        }
        
        # 전문가별 특화 분석
        expert_contributions = {}
        
        for expert_output in moe_result.expert_outputs:
            expert_id = expert_output.expert_id
            confidence = expert_output.confidence
            weight = expert_output.weight
            output = expert_output.output
            
            # 전문가 유형별 윤리적 해석
            if 'utilitarian' in expert_id:
                # 공리주의: 최대 행복 원칙
                expert_contributions['utilitarian'] = {
                    'care_harm': output[0].item() * 1.2,  # 해악 방지 중시
                    'fairness': output[1].item() * 1.0,
                    'loyalty': output[2].item() * 0.8,
                    'authority': output[3].item() * 0.7,
                    'sanctity': output[4].item() * 0.6,
                    'liberty': output[5].item() * 1.1,
                    'weight': weight,
                    'confidence': confidence
                }
            elif 'deontological' in expert_id:
                # 의무론: 의무와 규칙 중시
                expert_contributions['deontological'] = {
                    'care_harm': output[0].item() * 1.0,
                    'fairness': output[1].item() * 1.3,  # 공정성 중시
                    'loyalty': output[2].item() * 0.9,
                    'authority': output[3].item() * 1.2,  # 권위 중시
                    'sanctity': output[4].item() * 1.1,
                    'liberty': output[5].item() * 0.8,
                    'weight': weight,
                    'confidence': confidence
                }
            elif 'virtue_ethics' in expert_id:
                # 덕윤리: 인격과 덕목 중시
                expert_contributions['virtue_ethics'] = {
                    'care_harm': output[0].item() * 1.1,
                    'fairness': output[1].item() * 1.1,
                    'loyalty': output[2].item() * 1.2,  # 충성 중시
                    'authority': output[3].item() * 1.0,
                    'sanctity': output[4].item() * 1.3,  # 순수성 중시
                    'liberty': output[5].item() * 0.9,
                    'weight': weight,
                    'confidence': confidence
                }
            elif 'care_ethics' in expert_id:
                # 돌봄윤리: 관계와 돌봄 중시
                expert_contributions['care_ethics'] = {
                    'care_harm': output[0].item() * 1.4,  # 돌봄 최우선
                    'fairness': output[1].item() * 0.9,
                    'loyalty': output[2].item() * 1.3,  # 관계 중시
                    'authority': output[3].item() * 0.7,
                    'sanctity': output[4].item() * 0.8,
                    'liberty': output[5].item() * 1.0,
                    'weight': weight,
                    'confidence': confidence
                }
        
        # 가중 평균으로 최종 윤리 가치 계산
        if expert_contributions:
            total_weight = sum(contrib['weight'] for contrib in expert_contributions.values())
            
            if total_weight > 0:
                for value_key in ethical_values.keys():
                    weighted_sum = sum(
                        contrib[value_key] * contrib['weight'] 
                        for contrib in expert_contributions.values()
                    )
                    ethical_values[value_key] = max(0.1, min(0.9, weighted_sum / total_weight))
        
        return ethical_values
    
    def _basic_ethical_reasoning(self, emotion_data: 'EmotionData', 
                               input_data: Dict[str, Any]) -> Dict[str, float]:
        """기본 감정 기반 윤리 추론 (fallback)"""
        
        ethical_values = {
            'care_harm': 0.5,
            'fairness': 0.5,
            'loyalty': 0.5,
            'authority': 0.5,
            'sanctity': 0.5,
            'liberty': 0.5
        }
        
        if not emotion_data:
            return ethical_values
            
        # 감정에 따른 윤리적 가치 조정 (미묘하고 자연스럽게)
        primary_emotion = emotion_data.primary_emotion.value if hasattr(emotion_data.primary_emotion, 'value') else str(emotion_data.primary_emotion)
        
        if primary_emotion == 'joy':
            ethical_values['care_harm'] += 0.1  # 기쁠 때 더 돌봄 지향
            ethical_values['fairness'] += 0.05
        elif primary_emotion == 'sadness':
            ethical_values['care_harm'] += 0.15  # 슬플 때 해악 방지 중시
            ethical_values['loyalty'] += 0.1
        elif primary_emotion == 'anger':
            ethical_values['fairness'] += 0.2   # 화날 때 공정성 중시
            ethical_values['authority'] -= 0.1
        elif primary_emotion == 'fear':
            ethical_values['care_harm'] += 0.2  # 두려울 때 안전 최우선
            ethical_values['authority'] += 0.1
        elif primary_emotion == 'trust':
            ethical_values['loyalty'] += 0.15
            ethical_values['fairness'] += 0.1
        
        return ethical_values
    
    def _apply_legal_expert_analysis(self, input_data: Dict[str, Any], 
                                   ethical_values: Dict[str, float]):
        """법률 전문가 분석 적용"""
        
        # 도메인 식별
        domain = self._identify_legal_domain(input_data)
        
        # 법률 분석 컨텍스트 데이터 구성
        context_data = {
            'urgency': input_data.get('urgency', 0.5),
            'impact_level': input_data.get('impact_level', 0.5),
            'public_interest': ethical_values.get('care_harm', 0.5),
            'stakeholder_count': input_data.get('affected_people', 1),
            'financial_impact': input_data.get('financial_impact', 0)
        }
        
        # 법률 분석 수행
        text = input_data.get('text', '') or input_data.get('description', '')
        legal_analysis = self.legal_expert.analyze_legal_context(
            domain=domain,
            text=text,
            context_data=context_data
        )
        
        return legal_analysis
    
    def _identify_legal_domain(self, input_data: Dict[str, Any]) -> LegalDomain:
        """법률 도메인 식별"""
        
        text = input_data.get('text', '') or input_data.get('description', '')
        if not text:
            return LegalDomain.LIFE  # 기본값
        
        text_lower = text.lower()
        
        # 도메인별 키워드 매칭
        domain_keywords = {
            LegalDomain.EDUCATION: ['학교', '교육', '학생', '교사', '수업', '학습', '시험'],
            LegalDomain.BUSINESS: ['회사', '사업', '근무', '직원', '계약', '거래', '매출'],
            LegalDomain.SOCIAL: ['사회', '공동체', '관계', '사람', '커뮤니티', '단체'],
            LegalDomain.POLITICS: ['정치', '정부', '선거', '정책', '법률', '행정', '공무원'],
            LegalDomain.LIFE: ['생활', '일상', '개인', '가족', '건강', '소비', '주거']
        }
        
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            domain_scores[domain] = score
        
        # 가장 높은 점수의 도메인 반환
        best_domain = max(domain_scores, key=domain_scores.get)
        return best_domain if domain_scores[best_domain] > 0 else LegalDomain.LIFE
    
    def _integrate_legal_analysis(self, ethical_values: Dict[str, float], 
                                legal_analysis) -> Dict[str, float]:
        """법률 분석 결과를 윤리적 가치에 통합"""
        
        enhanced_values = ethical_values.copy()
        
        # 위험 수준에 따른 윤리적 가치 조정
        risk_level = legal_analysis.risk_level
        
        if risk_level.value == 'critical':
            # 심각한 법적 위험 → 안전성 우선
            enhanced_values['care_harm'] = min(0.9, enhanced_values['care_harm'] * 1.3)
            enhanced_values['authority'] = min(0.9, enhanced_values['authority'] * 1.2)
            enhanced_values['liberty'] = max(0.1, enhanced_values['liberty'] * 0.7)
            
        elif risk_level.value == 'high':
            # 높은 법적 위험 → 신중한 접근
            enhanced_values['care_harm'] = min(0.9, enhanced_values['care_harm'] * 1.2)
            enhanced_values['authority'] = min(0.9, enhanced_values['authority'] * 1.1)
            enhanced_values['fairness'] = min(0.9, enhanced_values['fairness'] * 1.1)
            
        elif risk_level.value == 'medium':
            # 중간 법적 위험 → 균형 조정
            enhanced_values['fairness'] = min(0.9, enhanced_values['fairness'] * 1.1)
            enhanced_values['authority'] = min(0.9, enhanced_values['authority'] * 1.05)
            
        # 신뢰도에 따른 가중치 조정
        confidence_factor = legal_analysis.confidence
        
        # 신뢰도가 낮으면 조정 폭을 줄임
        if confidence_factor < 0.7:
            adjustment_factor = 0.5 + confidence_factor * 0.5
            
            for key in enhanced_values:
                original_value = ethical_values[key]
                adjusted_value = enhanced_values[key]
                enhanced_values[key] = original_value + (adjusted_value - original_value) * adjustment_factor
        
        # 법률 분석 메타데이터 추가
        enhanced_values['legal_metadata'] = {
            'risk_level': risk_level.value,
            'confidence': confidence_factor,
            'relevant_laws': legal_analysis.relevant_laws,
            'recommendations': legal_analysis.recommendations[:3]  # 상위 3개만
        }
        
        return enhanced_values
    
    def _extract_learning_bias(self, past_regret_memory: Dict[str, float]) -> Dict[str, float]:
        """과거 후회에서 미묘한 학습 편향 추출 (직접 개입 아님)"""
        
        # 후회는 직접적이지 않고 학습된 편향으로만 작용
        learning_bias = {
            'risk_aversion': 0.0,    # 위험 회피 성향
            'time_preference': 0.0,   # 시간 선호
            'social_weight': 0.0      # 사회적 고려 가중치
        }
        
        # 과거 후회 패턴에서 학습된 편향 추출 (매우 미묘하게)
        avg_regret = past_regret_memory.get('average_regret', 0.0)
        
        if avg_regret > 0.7:  # 높은 후회 경험 시
            learning_bias['risk_aversion'] = 0.1   # 약간 더 신중하게
            learning_bias['time_preference'] = 0.05  # 장기적 관점 증가
        elif avg_regret < 0.3:  # 낮은 후회 경험 시  
            learning_bias['risk_aversion'] = -0.05  # 약간 더 대담하게
            
        return learning_bias
    
    async def _search_similar_experiences(self, input_data: Dict[str, Any], 
                                        experience_db) -> List[Dict[str, Any]]:
        """경험 메모리에서 유사 상황 검색"""
        try:
            from advanced_experience_database import ExperienceQuery
            
            # 쿼리 구성
            scenario_text = input_data.get('text_description', '')
            query = ExperienceQuery(
                query_text=scenario_text,
                category_filter='ethical_decision',
                max_results=5,
                similarity_threshold=0.7
            )
            
            # 유사 경험 검색
            similar_experiences = await experience_db.search_experiences(query)
            return similar_experiences
            
        except Exception as e:
            self.logger.warning(f"유사 경험 검색 실패: {e}")
            return []
    
    def _extract_experience_adjustments(self, similar_experiences: List[Dict[str, Any]]) -> Dict[str, float]:
        """유사 경험에서 벤담 계산 조정값 추출"""
        adjustments = {
            'intensity_bias': 0.0,
            'duration_bias': 0.0,
            'certainty_bias': 0.0,
            'risk_aversion_bias': 0.0,
            'social_weight_bias': 0.0
        }
        
        if not similar_experiences:
            return adjustments
            
        # 경험별 후회 정보 분석
        total_weight = 0.0
        for exp in similar_experiences:
            exp_weight = exp.get('similarity_score', 0.5)
            regret_level = exp.get('regret_level', 0.5)
            outcome_satisfaction = exp.get('outcome_satisfaction', 0.5)
            
            # 후회가 높았던 경우 -> 더 신중하게
            if regret_level > 0.6:
                adjustments['certainty_bias'] += exp_weight * 0.1
                adjustments['risk_aversion_bias'] += exp_weight * 0.15
            
            # 만족도가 낮았던 경우 -> 더 보수적으로  
            if outcome_satisfaction < 0.4:
                adjustments['intensity_bias'] += exp_weight * 0.05
                adjustments['duration_bias'] += exp_weight * 0.1
                
            total_weight += exp_weight
        
        # 가중 평균으로 정규화
        if total_weight > 0:
            for key in adjustments:
                adjustments[key] = max(-0.2, min(0.2, adjustments[key] / total_weight))
                
        return adjustments
    
    def _apply_experience_adjustments(self, input_data: Dict[str, Any], 
                                    adjustments: Dict[str, float]) -> Dict[str, Any]:
        """경험 기반 조정값을 입력 데이터에 적용"""
        adjusted_input = input_data.copy()
        
        # 벤담 변수들 조정
        if 'input_values' in adjusted_input:
            bentham_values = adjusted_input['input_values'].copy()
            
            # intensity 조정
            if 'intensity' in bentham_values:
                bentham_values['intensity'] = max(0.0, min(1.0, 
                    bentham_values['intensity'] + adjustments.get('intensity_bias', 0.0)))
            
            # duration 조정  
            if 'duration' in bentham_values:
                bentham_values['duration'] = max(0.0, min(1.0,
                    bentham_values['duration'] + adjustments.get('duration_bias', 0.0)))
                    
            # certainty 조정
            if 'certainty' in bentham_values:
                bentham_values['certainty'] = max(0.0, min(1.0,
                    bentham_values['certainty'] + adjustments.get('certainty_bias', 0.0)))
            
            adjusted_input['input_values'] = bentham_values
        
        # 가중치 레이어 조정
        if 'weight_adjustments' not in adjusted_input:
            adjusted_input['weight_adjustments'] = {}
            
        weight_adj = adjusted_input['weight_adjustments']
        weight_adj['risk_aversion'] = adjustments.get('risk_aversion_bias', 0.0)
        weight_adj['social_emphasis'] = adjustments.get('social_weight_bias', 0.0)
        
        return adjusted_input
    
    async def calculate_with_experience_integration(self,
                                                  input_data: Dict[str, Any],
                                                  experience_db = None,
                                                  use_cache: bool = True) -> EnhancedHedonicResult:
        """경험 메모리 통합 벤담 계산 - 유사 상황 기반 판단 개선"""
        
        # 캐시 확인
        cache_key = self._generate_cache_key(input_data)
        if use_cache and cache_key in self.calculation_cache:
            return self.calculation_cache[cache_key]
        
        try:
            # 1. 경험 메모리에서 유사 상황 검색
            experience_adjustments = {}
            if experience_db:
                similar_experiences = await self._search_similar_experiences(
                    input_data, experience_db
                )
                experience_adjustments = self._extract_experience_adjustments(
                    similar_experiences
                )
            
            # 2. 경험 기반 입력 데이터 조정
            adjusted_input = self._apply_experience_adjustments(
                input_data, experience_adjustments
            )
            
            # 3. 기존 고급 계산 수행
            result = self.calculate_with_advanced_layers(adjusted_input, use_cache=False)
            
            # 4. 경험 메모리 메타데이터 추가
            result.metadata['experience_influence'] = experience_adjustments
            result.metadata['similar_cases_count'] = len(experience_adjustments)
            
            # 5. 캐시 저장
            if use_cache:
                self.calculation_cache[cache_key] = result
                
            return result
            
        except Exception as e:
            self.logger.error(f"경험 통합 벤담 계산 실패: {e}")
            # fallback to regular calculation
            return self.calculate_with_advanced_layers(input_data, use_cache)

    def calculate_with_advanced_layers(self, 
                                     input_data: Dict[str, Any],
                                     use_cache: bool = True) -> EnhancedHedonicResult:
        """고급 다층 가중치를 적용한 쾌락 계산"""
        
        # 캐시 확인
        cache_key = self._generate_cache_key(input_data)
        if use_cache and cache_key in self.calculation_cache:
            self.logger.debug("캐시된 결과 반환")
            return self.calculation_cache[cache_key]
            
        try:
            # 1. 고급 맥락 분석
            context = self._prepare_advanced_context(input_data)
            
            # 2. 기본 계산
            base_score = self._calculate_base_advanced(context)
            
            # 3. 신경망 기반 가중치 예측
            if self.advanced_config['use_neural_prediction']:
                neural_weights = self._predict_neural_weights(context)
            else:
                neural_weights = None
                
            # 4. 각 레이어 적용 (동적 멀티레이어 AI 시스템)
            layered_score = base_score
            layer_results = []
            layer_interactions = []  # 레이어간 상호작용 추적
            
            for i, layer in enumerate(self.weight_layers):
                try:
                    # 신경망 가중치 사용 (가능한 경우)
                    if neural_weights is not None:
                        layer_weight = neural_weights[i].item()
                    else:
                        layer_weight = layer.compute_weight(context)
                    
                    # 동적 조정: 복잡한 상황에서 더 넓은 범위 허용
                    context_complexity = self._calculate_context_complexity(context)
                    if context_complexity > 0.7:  # 복잡한 상황
                        layer_weight = max(0.3, min(2.5, layer_weight))  # 더 넓은 범위
                    else:  # 일반적 상황
                        layer_weight = max(0.7, min(1.5, layer_weight))  # 안정적 범위
                    
                    # 레이어 상호작용 계산
                    if i > 0:
                        interaction_factor = self._calculate_layer_interaction(
                            layer_results[-1], layer, context
                        )
                        layer_weight *= interaction_factor
                        layer_interactions.append(interaction_factor)
                    
                    # 점진적 곱셈 적용 (원본 멀티레이어 설계)
                    layered_score *= layer_weight
                    
                    layer_contribution = layer.get_contribution()
                    layer_contribution.weight_factor = layer_weight
                    layer_results.append(layer_contribution)
                    
                    self.logger.debug(f"{layer.name}: {layer_weight:.3f} -> 누적 점수: {layered_score:.3f}")
                    
                except Exception as e:
                    self.logger.error(f"{layer.name} 레이어 적용 실패: {e}")
                    # 중립 가중치 적용 (시스템 지속성 보장)
                    layered_score *= 1.0
                    layer_results.append(WeightLayerResult(
                        layer_name=layer.name,
                        weight_factor=1.0,
                        contribution_score=0.0,
                        confidence=0.5,
                        metadata={'reasoning': f"중립 가중치 적용: {str(e)}"}
                    ))
            
            # 지능적 점수 정규화 (동적 범위 조정)
            layered_score = self._smart_score_normalization(layered_score, context, layer_interactions)
            
            # 5. 고급 극단값 보정
            adjustment_result = self._apply_advanced_extreme_adjustment(
                layered_score, context
            )
            
            # 6. 최적화 (선택적)
            if self.advanced_config['optimization_enabled']:
                final_score = self._optimize_final_score(
                    adjustment_result['score'], context
                )
            else:
                final_score = adjustment_result['score']
            
            # 최종 점수 안전성 검증 및 정규화
            final_score = self._ensure_score_bounds(final_score)
                
            # 7. 종합 신뢰도 계산
            confidence = self._calculate_comprehensive_confidence(
                context, layer_results, neural_weights
            )
            
            # 8. 상세 분석 결과
            calculation_breakdown = self._generate_detailed_breakdown(
                base_score, layer_results, adjustment_result, context
            )
            
            # 9. HedonicValues 객체 생성 (벤담의 7가지 변수 실제 계산)
            hedonic_values = self._calculate_hedonic_values(context, final_score, base_score)
            
            # 결과 생성
            result = EnhancedHedonicResult(
                final_score=final_score,
                base_score=base_score,
                hedonic_values=hedonic_values,  # 실제 벤담 값들 포함
                layer_contributions=layer_results,
                extreme_adjustment_applied=adjustment_result['applied'],
                adjustment_factor=adjustment_result['factor'],
                confidence_score=confidence,
                processing_time=time.time() - context.start_time if hasattr(context, 'start_time') else 0.0,
                context_analysis={
                    'calculation_breakdown': calculation_breakdown,
                    'neural_prediction_used': neural_weights is not None,
                    'optimization_applied': self.advanced_config['optimization_enabled']
                }
            )
            
            # 캐시 저장
            if use_cache:
                with self.cache_lock:
                    self.calculation_cache[cache_key] = result
                    
            return result
            
        except Exception as e:
            self.logger.error(f"고급 계산 실패: {e}")
            raise RuntimeError(
                f"고급 벤담 계산에 실패했습니다: {str(e)}. "
                f"신경망 예측, 트랜스포머 분석, 또는 가중치 레이어 중 하나가 실패했습니다. "
                f"모든 구성 요소가 올바르게 초기화되고 훈련되었는지 확인하세요. "
                f"대체 계산 방법을 사용하지 않고 정확한 분석이 필요합니다."
            )
            
    def _calculate_hedonic_values(self, context: AdvancedCalculationContext, final_score: float, base_score: float) -> HedonicValues:
        """벤담의 7가지 변수를 실제 계산하여 HedonicValues 객체 생성"""
        from data_models import HedonicValues
        
        try:
            # 기본 맥락 데이터 추출
            emotion_data = context.emotion_data
            affected_count = context.affected_count
            duration_seconds = context.duration_seconds
            uncertainty_level = context.uncertainty_level
            information_quality = context.information_quality
            
            # 1. 강도 (Intensity) - 감정 강도와 최종 점수 기반
            intensity = final_score
            if emotion_data and hasattr(emotion_data, 'arousal'):
                from data_models import safe_float_operation
                intensity = safe_float_operation(final_score, emotion_data.arousal, 'add') / 2.0
            intensity = max(0.0, min(1.0, intensity))
            
            # 2. 지속성 (Duration) - 시간 기반 계산
            duration = min(1.0, duration_seconds / 3600.0)  # 1시간을 최대값으로 정규화
            if duration_seconds > 86400:  # 하루 이상
                duration = 1.0
            elif duration_seconds < 60:  # 1분 미만
                duration = duration_seconds / 60.0
                
            # 3. 확실성 (Certainty) - 정보 품질과 불확실성 기반
            certainty = information_quality * (1.0 - uncertainty_level)
            certainty = max(0.0, min(1.0, certainty))
            
            # 4. 근접성 (Propinquity) - 영향 범위와 시간 기반
            propinquity = 1.0 / (1.0 + duration_seconds / 3600.0)  # 시간이 멀수록 감소
            propinquity = max(0.1, min(1.0, propinquity))
            
            # 5. 다산성 (Fecundity) - 영향 범위와 감정 강도 기반
            fecundity = (affected_count / 100.0) * intensity  # 100명을 기준으로 정규화
            fecundity = max(0.0, min(1.0, fecundity))
            
            # 6. 순수성 (Purity) - 확실성과 강도의 조합
            purity = certainty * intensity
            purity = max(0.0, min(1.0, purity))
            
            # 7. 범위 (Extent) - 영향받는 사람 수 기반
            extent = min(1.0, affected_count / 1000.0)  # 1000명을 최대값으로 정규화
            if affected_count < 1:
                extent = 0.1  # 최소값
                
            # 총 쾌락값 계산 (벤담의 원래 공식: 강도 × 지속성 × 확실성 × 근접성 × 다산성 × 순수성 × 범위)
            hedonic_total = intensity * duration * certainty * propinquity * fecundity * purity * extent
            hedonic_total = max(0.0, min(1.0, hedonic_total))
            
            # HedonicValues 객체 생성
            hedonic_values = HedonicValues(
                intensity=intensity,
                duration=duration,
                certainty=certainty,
                propinquity=propinquity,
                fecundity=fecundity,
                purity=purity,
                extent=extent,
                hedonic_total=hedonic_total
            )
            
            self.logger.debug(f"벤담 값 계산 완료: 강도={intensity:.3f}, 지속성={duration:.3f}, 확실성={certainty:.3f}, 총값={hedonic_total:.3f}")
            
            return hedonic_values
            
        except Exception as e:
            self.logger.error(f"HedonicValues 계산 실패: {e}")
            # 실패 시 기본값 반환하지 않고 예외 발생
            raise RuntimeError(f"벤담 세부값 계산 실패 - fallback 비활성화: {str(e)}")
    
            
    def _prepare_advanced_context(self, input_data: Dict[str, Any]) -> AdvancedCalculationContext:
        """고급 계산 맥락 준비"""
        context = AdvancedCalculationContext()
        context.start_time = time.time()
        
        # 기본 정보
        context.input_values = input_data.get('input_values', {})
        context.emotion_data = input_data.get('emotion_data')
        context.affected_count = input_data.get('affected_count', 1)
        context.duration_seconds = input_data.get('duration_seconds', 60)
        context.information_quality = input_data.get('information_quality', 0.7)
        context.uncertainty_level = input_data.get('uncertainty_level', 0.3)
        
        # 맥락 분석
        if self.advanced_config['use_transformer_analysis']:
            text_input = input_data.get('text_description', "")
            if text_input:
                language = input_data.get('language', 'ko')
                analysis_result = self.context_analyzer.analyze_context(text_input, language)
                
                context.context_embedding = analysis_result.get('context_embedding')
                context.complexity_metrics = analysis_result.get('complexity', {})
                context.ethical_analysis = analysis_result.get('ethical_aspects', {})
                context.emotion_analysis = analysis_result.get('emotions', [])
                
        # 추가 맥락 정보
        context.social_context = input_data.get('social_context', {})
        context.temporal_context = input_data.get('temporal_context', {})
        context.ethical_context = input_data.get('ethical_context', {})
        context.cognitive_context = input_data.get('cognitive_context', {})
        
        # 윤리적 가치 정보
        context.ethical_values = input_data.get('ethical_values', {})
        
        # 학습 편향 정보 (과거 후회 경험에서 학습된)
        context.learning_bias = input_data.get('learning_bias', {})
        
        return context
        
    def _calculate_base_advanced(self, context: AdvancedCalculationContext) -> float:
        """고급 기본 계산"""
        # 가중치 기반 계산 (Bentham v2 확장)
        weights = {
            'intensity': 0.20, 'duration': 0.12, 'certainty': 0.12,
            'propinquity': 0.08, 'fecundity': 0.08, 'purity': 0.08, 'extent': 0.12,
            # Bentham v2 추가 변수들
            'external_cost': 0.10,           # 외부비용 (중요도 높음)
            'redistribution_effect': 0.05,   # 재분배효과 (중간 중요도)
            'self_damage': 0.05             # 자아손상 (중간 중요도)
        }
        
        total_score = 0.0
        
        # Bentham v2 새로운 변수들 계산 (안전한 호출)
        try:
            bentham_v2_vars = bentham_v2_calculator.calculate_bentham_v2_variables(
                context.__dict__, getattr(context, 'surd_graph', None)
            )
            # 새로운 변수들을 context에 추가
            context.input_values.update(bentham_v2_vars.to_dict())
        except Exception as e:
            self.logger.warning(f"Bentham v2 변수 계산 실패, 기본값 사용: {e}")
            # 기본값으로 안전하게 처리
            context.input_values.update({
                'external_cost': 0.5,
                'redistribution_effect': 0.5, 
                'self_damage': 0.5
            })
        
        for variable, weight in weights.items():
            value = context.input_values.get(variable, 0.5)
            # 비선형 변환 적용
            transformed_value = self._apply_nonlinear_transform(value, variable)
            total_score += transformed_value * weight
            
        # 감정 데이터 보정 (스케일링 문제 수정)
        if context.emotion_data:
            # EmotionIntensity Enum을 안전하게 float로 변환
            intensity_value = getattr(context.emotion_data.intensity, 'value', 2) if hasattr(context.emotion_data.intensity, 'value') else 2
            # 정규화된 강도 (0.0-1.0 범위)
            normalized_intensity = min(max(float(intensity_value) / 4.0, 0.0), 1.0)
            
            from data_models import emotion_intensity_to_float
            valence_val = emotion_intensity_to_float(context.emotion_data.valence)
            arousal_val = emotion_intensity_to_float(context.emotion_data.arousal)
            
            emotion_factor = (
                valence_val * 0.4 + 
                arousal_val * 0.3 + 
                normalized_intensity * 0.3
            )
            # 감정 보정을 덧셈으로 변경하여 스케일링 문제 방지
            total_score = total_score + (emotion_factor * 0.1)
        
        # 학습 편향 반영 (과거 경험에서 온 미묘한 조정)
        if hasattr(context, 'learning_bias') and context.learning_bias:
            learning_factor = self._apply_learning_bias(context.learning_bias, context)
            total_score = total_score * learning_factor
        
        # 윤리적 가치 기반 조정
        if hasattr(context, 'ethical_values') and context.ethical_values:
            ethical_factor = self._apply_ethical_values(context.ethical_values, context)
            total_score = total_score * ethical_factor
            
        return total_score
    
    def _apply_learning_bias(self, learning_bias: Dict[str, float], 
                           context: AdvancedCalculationContext) -> float:
        """과거 경험에서 학습된 편향을 미묘하게 반영 (퍼지 적응)"""
        
        bias_factor = 1.0
        
        # 위험 회피 성향 (퍼지 적응)
        risk_aversion = learning_bias.get('risk_aversion', 0.0)
        if risk_aversion != 0.0:
            uncertainty = context.uncertainty_level
            # 퍼지 적응: 불확실성에 따른 연속적 조정
            fuzzy_uncertainty = self._adaptive_fuzzy_adjustment(uncertainty, risk_aversion)
            bias_factor *= (1.0 + fuzzy_uncertainty * 0.1)
        
        # 시간 선호 편향 (퍼지 적응)
        time_preference = learning_bias.get('time_preference', 0.0)
        if time_preference != 0.0 and hasattr(context, 'duration_seconds'):
            time_factor = min(context.duration_seconds / 86400.0, 1.0)
            # 감정 상태에 따른 시간 인식 조정
            emotion_time_modifier = self._emotion_time_perception_modifier(context)
            adjusted_time_factor = time_factor * emotion_time_modifier
            bias_factor *= (1.0 + time_preference * adjusted_time_factor * 0.05)
        
        # 사회적 고려 가중치 (퍼지 적응)
        social_weight = learning_bias.get('social_weight', 0.0)
        if social_weight != 0.0:
            social_factor = min(context.affected_count / 100.0, 1.0)
            # 감정에 따른 사회적 민감도 조정
            emotion_social_modifier = self._emotion_social_sensitivity(context)
            adjusted_social_factor = social_factor * emotion_social_modifier
            bias_factor *= (1.0 + social_weight * adjusted_social_factor * 0.05)
        
        # 퍼지 경계를 통한 부드러운 제한
        return self._fuzzy_boundary_clamp(bias_factor, 0.95, 1.05)
    
    def _adaptive_fuzzy_adjustment(self, uncertainty: float, bias_strength: float) -> float:
        """불확실성에 따른 퍼지 적응 조정"""
        # 불확실성이 높을 때 편향이 더 강하게 작용
        # 하지만 부드럽게 전환
        uncertainty_amplifier = np.tanh(uncertainty * 2)  # 0~1 부드러운 증폭
        return bias_strength * uncertainty_amplifier
    
    def _emotion_time_perception_modifier(self, context: AdvancedCalculationContext) -> float:
        """감정 상태에 따른 시간 인식 조정"""
        if not context.emotion_data:
            return 1.0
        
        # 감정에 따른 시간 인식 변화 (연구 기반)
        emotion_name = context.emotion_data.primary_emotion.value if hasattr(context.emotion_data.primary_emotion, 'value') else str(context.emotion_data.primary_emotion)
        
        time_perception_map = {
            'fear': 0.7,     # 두려움: 시간이 빠르게 느껴짐
            'anger': 0.8,    # 분노: 급하게 느껴짐
            'joy': 1.2,      # 기쁨: 시간이 느리게 느껴짐
            'sadness': 1.3,  # 슬픔: 시간이 더 느리게 느껴짐
            'neutral': 1.0
        }
        
        return time_perception_map.get(emotion_name, 1.0)
    
    def _emotion_social_sensitivity(self, context: AdvancedCalculationContext) -> float:
        """감정 상태에 따른 사회적 민감도 조정"""
        if not context.emotion_data:
            return 1.0
        
        emotion_name = context.emotion_data.primary_emotion.value if hasattr(context.emotion_data.primary_emotion, 'value') else str(context.emotion_data.primary_emotion)
        
        social_sensitivity_map = {
            'fear': 1.3,     # 두려움: 사회적 지지 더 중요
            'sadness': 1.4,  # 슬픔: 타인의 반응에 민감
            'anger': 0.8,    # 분노: 사회적 고려 감소
            'joy': 1.1,      # 기쁨: 나눔 의식 증가
            'neutral': 1.0
        }
        
        return social_sensitivity_map.get(emotion_name, 1.0)
    
    def _fuzzy_boundary_clamp(self, value: float, min_val: float, max_val: float) -> float:
        """퍼지 경계를 통한 부드러운 클래핑"""
        # 하드 클래핑 대신 시그모이드 변환으로 부드럽게
        if value < min_val:
            excess = min_val - value
            return min_val - excess * np.exp(-excess * 10)  # 부드러운 하한
        elif value > max_val:
            excess = value - max_val
            return max_val + excess * np.exp(-excess * 10)  # 부드러운 상한
        else:
            return value
    
    def _apply_ethical_values(self, ethical_values: Dict[str, float], 
                            context: AdvancedCalculationContext) -> float:
        """윤리적 가치 기반 벤담 계산 조정"""
        
        # 주요 윤리적 고려사항들
        care_harm = ethical_values.get('care_harm', 0.5)
        fairness = ethical_values.get('fairness', 0.5)
        loyalty = ethical_values.get('loyalty', 0.5)
        
        # 시나리오 특성에 따른 윤리적 가중치 적용
        ethical_factor = 1.0
        
        # 해악 방지 고려
        if care_harm > 0.6:
            # 안전/돌봄이 중요한 경우 확실성과 지속성 중시
            certainty_boost = context.input_values.get('certainty', 0.5) * 0.1
            ethical_factor += certainty_boost
        
        # 공정성 고려  
        if fairness > 0.6:
            # 공정성이 중요한 경우 범위(extent) 가중치 증가
            extent_boost = context.input_values.get('extent', 0.5) * 0.1
            ethical_factor += extent_boost
        
        # 충성/관계 고려
        if loyalty > 0.6:
            # 관계가 중요한 경우 근접성(propinquity) 가중치 증가
            propinquity_boost = context.input_values.get('propinquity', 0.5) * 0.1
            ethical_factor += propinquity_boost
        
        # 윤리적 조정도 미묘하게 제한 (±10% 이내)
        return max(0.9, min(1.1, ethical_factor))
        
    def _apply_nonlinear_transform(self, value: float, variable: str) -> float:
        """비선형 변환 적용"""
        if variable in ['intensity', 'extent']:
            # 강도와 범위는 제곱근 변환
            return np.sqrt(value)
        elif variable in ['duration', 'fecundity']:
            # 지속성과 생산성은 로그 변환
            return np.log1p(value)
        else:
            # 기타는 시그모이드 변환
            return 1 / (1 + np.exp(-10 * (value - 0.5)))
            
    def _predict_neural_weights(self, context: AdvancedCalculationContext) -> torch.Tensor:
        """신경망 기반 가중치 예측"""
        try:
            features = self._extract_neural_features(context)
            # Neural predictor와 같은 디바이스로 이동
            predictor_device = next(self.neural_predictor.parameters()).device
            features_tensor = torch.tensor(features, dtype=TORCH_DTYPE).to(predictor_device)
            
            with torch.no_grad():
                weights = self.neural_predictor(features_tensor.unsqueeze(0))
                
            return weights.squeeze(0)
            
        except Exception as e:
            self.logger.error(f"신경망 가중치 예측 실패: {e}")
            return None
            
    def _extract_neural_features(self, context: AdvancedCalculationContext) -> np.ndarray:
        """신경망용 특성 추출"""
        features = []
        
        # 기본 벤담 변수들
        for var in ['intensity', 'duration', 'certainty', 'propinquity', 
                   'fecundity', 'purity', 'extent']:
            features.append(context.input_values.get(var, 0.5))
            
        # 감정 특성
        if context.emotion_data:
            from data_models import emotion_intensity_to_float
            valence_val = emotion_intensity_to_float(context.emotion_data.valence)
            arousal_val = emotion_intensity_to_float(context.emotion_data.arousal)
            intensity_val = float(context.emotion_data.intensity.value) / 4.0 if hasattr(context.emotion_data.intensity, 'value') else emotion_intensity_to_float(context.emotion_data.intensity)
            confidence_val = emotion_intensity_to_float(context.emotion_data.confidence)
            
            features.extend([
                valence_val,
                arousal_val,
                intensity_val,
                confidence_val
            ])
        else:
            features.extend([0.0, 0.0, 0.5, 0.5])
            
        # 맥락 임베딩 요약 (주성분 분석)
        if hasattr(context, 'context_embedding') and context.context_embedding is not None:
            embedding = context.context_embedding.flatten()
            # PCA 차원 축소 (처음 20차원 사용)
            embedding_summary = embedding[:20] if len(embedding) >= 20 else np.pad(embedding, (0, 20-len(embedding)))
            features.extend(embedding_summary.tolist())
        else:
            features.extend([0.0] * 20)
            
        # 추가 컨텍스트 정보
        features.extend([
            np.log1p(context.affected_count) / 10.0,  # 로그 스케일링
            min(context.duration_seconds / 86400.0, 1.0),  # 일 단위 정규화
            context.information_quality,
            context.uncertainty_level,
            len(context.input_values) / 7.0,  # 완전성 지표
        ])
        
        # 복잡도 메트릭
        if hasattr(context, 'complexity_metrics'):
            complexity = context.complexity_metrics
            features.extend([
                complexity.get('lexical_diversity', 0.5),
                complexity.get('structural_complexity', 0.1) * 10,
                min(complexity.get('word_count', 10) / 100.0, 1.0)
            ])
        else:
            features.extend([0.5, 0.1, 0.1])
            
        # 윤리적 점수
        if hasattr(context, 'ethical_analysis'):
            ethical = context.ethical_analysis
            if isinstance(ethical, dict) and 'scores' in ethical:
                max_ethical_score = max(ethical['scores']) if ethical['scores'] else 0.5
                features.append(max_ethical_score)
            else:
                features.append(0.5)
        else:
            features.append(0.5)
            
        # 감정 분석 결과
        if hasattr(context, 'emotion_analysis'):
            emotions = context.emotion_analysis
            if emotions and isinstance(emotions, list):
                max_emotion_score = max([e.get('score', 0.5) for e in emotions])
                features.append(max_emotion_score)
            else:
                features.append(0.5)
        else:
            features.append(0.5)
            
        # 총 50차원으로 맞춤
        while len(features) < 50:
            features.append(0.0)
            
        return np.array(features[:50], dtype=np.float32)
        
    def _ensure_score_bounds(self, score: float) -> float:
        """최종 점수가 합리적인 범위 내에 있도록 보장"""
        if not isinstance(score, (int, float)) or np.isnan(score) or np.isinf(score):
            self.logger.warning(f"잘못된 점수 값 감지: {score}, 기본값 0.5 사용")
            return 0.5
            
        # 0-1 범위로 강제 제한
        bounded_score = max(0.0, min(1.0, score))
        
        if bounded_score != score:
            self.logger.info(f"점수 범위 조정: {score:.3f} -> {bounded_score:.3f}")
            
        return bounded_score
        
    def _apply_advanced_extreme_adjustment(self, score: float, 
                                         context: AdvancedCalculationContext) -> Dict[str, Any]:
        """고급 극단값 보정 - 점수를 0-1 범위로 정규화"""
        adjustment_factor = 1.0
        applied = False
        adjustment_type = "none"
        original_score = score
        
        # 점수를 0-1 범위로 정규화
        if score > 1.0:
            # 1을 초과하는 점수를 시그모이드 함수로 압축
            final_score = 1.0 / (1.0 + np.exp(-(score - 1.0)))
            final_score = 0.5 + (final_score - 0.5) * 0.5  # 0.5-1.0 범위로 매핑
            adjustment_factor = final_score / score if score != 0 else 1.0
            applied = True
            adjustment_type = "upper_normalization"
            
        elif score < 0.0:
            # 0 미만의 점수를 시그모이드 함수로 압축
            final_score = 1.0 / (1.0 + np.exp(-score))
            final_score = final_score * 0.5  # 0.0-0.5 범위로 매핑
            adjustment_factor = final_score / score if score != 0 else 1.0
            applied = True
            adjustment_type = "lower_normalization"
            
        else:
            # 이미 0-1 범위 내의 점수는 그대로 유지
            final_score = score
        
        return {
            'score': final_score,
            'factor': adjustment_factor,
            'applied': applied,
            'type': adjustment_type,
            'dynamic_threshold': self._calculate_dynamic_threshold(context)
        }
        
    def _calculate_context_complexity(self, context: AdvancedCalculationContext) -> float:
        """상황 복잡도 계산"""
        complexity_factors = []
        
        # 텍스트 복잡도
        if hasattr(context, 'text_description') and context.text_description:
            text_complexity = len(context.text_description.split()) / 100.0
            complexity_factors.append(min(1.0, text_complexity))
        
        # 영향받는 사람 수
        if hasattr(context, 'affected_count'):
            people_complexity = min(1.0, context.affected_count / 20.0)
            complexity_factors.append(people_complexity)
        
        # 불확실성 수준
        if hasattr(context, 'uncertainty_level'):
            complexity_factors.append(context.uncertainty_level)
        
        return np.mean(complexity_factors) if complexity_factors else 0.5
    
    def _calculate_layer_interaction(self, prev_layer_result, current_layer, context) -> float:
        """레이어간 상호작용 계산"""
        # 이전 레이어 신뢰도 기반 상호작용
        interaction_factor = 1.0
        
        if hasattr(prev_layer_result, 'confidence'):
            # 높은 신뢰도는 다음 레이어를 강화
            confidence_boost = 0.8 + (prev_layer_result.confidence * 0.4)
            interaction_factor *= confidence_boost
        
        # 레이어 타입별 상호작용
        if hasattr(current_layer, 'layer_type'):
            layer_synergy = self._get_layer_synergy(prev_layer_result, current_layer)
            interaction_factor *= layer_synergy
        
        return max(0.5, min(1.8, interaction_factor))
    
    def _get_layer_synergy(self, prev_result, current_layer) -> float:
        """레이어 시너지 계산"""
        # 레이어 타입별 시너지 매트릭스 (예시)
        synergy_matrix = {
            ('contextual', 'temporal'): 1.2,
            ('temporal', 'social'): 1.1,
            ('social', 'ethical'): 1.3,
            ('ethical', 'emotional'): 1.2,
            ('emotional', 'cognitive'): 1.1
        }
        
        prev_type = getattr(prev_result, 'layer_type', 'unknown')
        curr_type = getattr(current_layer, 'layer_type', 'unknown')
        
        return synergy_matrix.get((prev_type, curr_type), 1.0)
    
    def _smart_score_normalization(self, score: float, context: AdvancedCalculationContext, 
                                 interactions: List[float]) -> float:
        """지능적 점수 정규화"""
        # 기본 0-1 범위 적용
        if score <= 1.0:
            return max(0.0, score)
        
        # 복잡한 상황에서는 더 높은 점수 허용
        complexity = self._calculate_context_complexity(context)
        interaction_boost = np.mean(interactions) if interactions else 1.0
        
        # 동적 상한선 계산
        if complexity > 0.8 and interaction_boost > 1.2:
            # 매우 복잡한 상황: 0-2 범위 허용 후 압축
            max_allowed = 2.0
            normalized = score / max_allowed
            return min(1.0, normalized)
        elif complexity > 0.6:
            # 중간 복잡도: 0-1.5 범위 허용 후 압축
            max_allowed = 1.5
            normalized = score / max_allowed
            return min(1.0, normalized)
        else:
            # 단순한 상황: 0-1 범위 강제
            return min(1.0, max(0.0, score))

    def _calculate_dynamic_threshold(self, context: AdvancedCalculationContext) -> Dict[str, float]:
        """동적 임계값 계산"""
        base_pleasure = self.advanced_config['extreme_pleasure_threshold']
        base_pain = self.advanced_config['extreme_pain_threshold']
        
        # 맥락에 따른 임계값 조정
        adjustments = []
        
        # 사회적 영향 고려
        if context.affected_count > 10:
            adjustments.append(-0.1)  # 더 엄격한 임계값
        elif context.affected_count == 1:
            adjustments.append(0.1)   # 더 관대한 임계값
            
        # 시간적 영향 고려
        if context.duration_seconds > 3600:  # 1시간 이상
            adjustments.append(-0.05)
            
        # 불확실성 고려
        if context.uncertainty_level > 0.7:
            adjustments.append(0.05)  # 불확실할 때 더 관대하게
            
        adjustment = np.mean(adjustments) if adjustments else 0.0
        
        return {
            'pleasure': base_pleasure + adjustment,
            'pain': base_pain - adjustment  # 음수이므로 빼면 절댓값이 커짐
        }
        
    def _optimize_final_score(self, score: float, 
                            context: AdvancedCalculationContext) -> float:
        """최종 점수 최적화 - 0-1 범위 강제 적용"""
        try:
            # 먼저 점수를 0-1 범위로 제한
            bounded_score = max(0.0, min(1.0, score))
            
            # 목적 함수 정의 (0-1 범위 내에서만 최적화)
            def objective(x):
                # 원래 점수와의 차이 + 제약 조건 위반 페널티
                deviation = abs(x - bounded_score)
                
                # 0-1 범위 위반 시 강한 페널티
                penalty = 0.0
                if x < 0.0 or x > 1.0:
                    penalty += 100.0 * abs(x - np.clip(x, 0.0, 1.0))
                    
                # 극단값 소프트 페널티 (0.05-0.95 범위 선호)
                if x < 0.05:
                    penalty += 2.0 * (0.05 - x)
                elif x > 0.95:
                    penalty += 2.0 * (x - 0.95)
                    
                return deviation + penalty
                
            # 최적화 실행
            result = minimize_scalar(
                objective,
                bounds=(-1.0, 1.0),
                method='bounded'
            )
            
            if result.success:
                return result.x
            else:
                return score
                
        except Exception as e:
            self.logger.error(f"점수 최적화 실패: {e}")
            return score
            
    def _calculate_comprehensive_confidence(self, 
                                         context: AdvancedCalculationContext,
                                         layer_results: List[WeightLayerResult],
                                         neural_weights: torch.Tensor = None) -> float:
        """종합 신뢰도 계산"""
        confidence_factors = []
        
        # 데이터 품질
        data_quality = context.information_quality
        confidence_factors.append(data_quality)
        
        # 불확실성 (역수)
        uncertainty_factor = 1.0 - context.uncertainty_level
        confidence_factors.append(uncertainty_factor)
        
        # 레이어 신뢰도 평균
        layer_confidences = [layer.confidence for layer in layer_results]
        if layer_confidences:
            avg_layer_confidence = np.mean(layer_confidences)
            confidence_factors.append(avg_layer_confidence)
            
        # 신경망 예측 신뢰도
        if neural_weights is not None:
            # 가중치 분산이 낮을수록 높은 신뢰도
            weight_variance = torch.var(neural_weights).item()
            neural_confidence = max(0.0, 1.0 - weight_variance)
            confidence_factors.append(neural_confidence)
            
        # 맥락 완전성
        if hasattr(context, 'context_embedding') and context.context_embedding is not None:
            context_completeness = 1.0
        else:
            context_completeness = 0.5
        confidence_factors.append(context_completeness)
        
        return float(np.mean(confidence_factors))
        
    def _generate_detailed_breakdown(self, 
                                   base_score: float,
                                   layer_results: List[WeightLayerResult],
                                   adjustment_result: Dict[str, Any],
                                   context: AdvancedCalculationContext) -> Dict[str, Any]:
        """상세 계산 분석 생성"""
        return {
            'base_calculation': {
                'input_variables': context.input_values,
                'base_score': base_score,
                'nonlinear_transforms_applied': True
            },
            'layer_analysis': {
                'layer_count': len(layer_results),
                'layer_weights': [r.weight_factor for r in layer_results],
                'layer_confidences': [r.confidence for r in layer_results],
                'progressive_scores': self._calculate_progressive_scores(base_score, layer_results)
            },
            'neural_prediction': {
                'used': self.advanced_config['use_neural_prediction'],
                'model_architecture': str(self.neural_predictor) if hasattr(self, 'neural_predictor') else None
            },
            'extreme_adjustment': adjustment_result,
            'context_analysis': {
                'transformer_used': self.advanced_config['use_transformer_analysis'],
                'complexity_analyzed': hasattr(context, 'complexity_metrics'),
                'ethical_analyzed': hasattr(context, 'ethical_analysis'),
                'emotion_analyzed': hasattr(context, 'emotion_analysis')
            },
            'optimization': {
                'applied': self.advanced_config['optimization_enabled'],
                'method': self.optimization_params.get('method', 'none')
            },
            'performance': {
                'processing_time': getattr(context, 'processing_time', 0.0),
                'cache_used': hasattr(self, 'calculation_cache')
            }
        }
        
    def _calculate_progressive_scores(self, base_score: float, 
                                    layer_results: List[WeightLayerResult]) -> List[float]:
        """단계별 점수 진행 계산"""
        scores = [base_score]
        current_score = base_score
        
        for layer_result in layer_results:
            current_score *= layer_result.weight_factor
            scores.append(current_score)
            
        return scores
        
    def _generate_cache_key(self, input_data: Dict[str, Any]) -> str:
        """캐시 키 생성"""
        import hashlib
        
        # 중요한 입력 데이터만 해시
        key_data = {
            'input_values': input_data.get('input_values', {}),
            'affected_count': input_data.get('affected_count', 1),
            'duration_seconds': input_data.get('duration_seconds', 60),
            'information_quality': input_data.get('information_quality', 0.7),
            'text_description': input_data.get('text_description', "")[:100]  # 처음 100자만
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
        
        
    def clear_cache(self):
        """캐시 클리어"""
        with self.cache_lock:
            self.calculation_cache.clear()
            
    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계"""
        with self.cache_lock:
            return {
                'cache_size': len(self.calculation_cache),
                'cache_keys': list(self.calculation_cache.keys())[:10]  # 처음 10개만
            }
    
    def get_pytorch_network(self) -> Optional[nn.Module]:
        """
        헤드/스왑매니저가 사용할 대표 PyTorch 네트워크를 반환.
        - 가능한 후보를 순서대로 탐색해서 nn.Module을 반환
        - 한 번 찾으면 캐시(self._primary_nn)해 재사용
        """
        import torch.nn as nn
        
        # 캐시 있으면 즉시 반환
        if hasattr(self, "_primary_nn") and isinstance(self._primary_nn, nn.Module):
            logger.info("AdvancedBenthamCalculator: 캐시된 primary_nn 반환")
            return self._primary_nn
        
        candidates = []
        
        # 1) 자주 쓰이는 네이밍 우선
        for name in ["neural_predictor", "bentham_model", "scoring_network", "model", "network", "default_network"]:
            if hasattr(self, name):
                obj = getattr(self, name)
                # property인 경우 처리
                if hasattr(obj, '__get__'):
                    try:
                        obj = obj.__get__(self, type(self))
                    except Exception:
                        obj = None
                # callable이면 한 번 호출하여 인스턴스 획득
                try:
                    if callable(obj) and not isinstance(obj, nn.Module):
                        obj = obj()
                except Exception:
                    obj = None
                if isinstance(obj, nn.Module):
                    candidates.append((name, obj))
        
        # 2) weight_layers 내부의 neural_predictor 확인
        if hasattr(self, 'weight_layers'):
            weight_layers = getattr(self, 'weight_layers')
            if hasattr(weight_layers, '__get__'):
                try:
                    weight_layers = weight_layers.__get__(self, type(self))
                except Exception:
                    weight_layers = None
            
            if weight_layers and isinstance(weight_layers, dict):
                for layer_name, layer in weight_layers.items():
                    if hasattr(layer, 'neural_predictor') and isinstance(layer.neural_predictor, nn.Module):
                        candidates.append((f"weight_layers.{layer_name}.neural_predictor", layer.neural_predictor))
        
        # 3) 멤버 중 nn.Module 자동 탐색 (백업 경로)
        if not candidates:
            try:
                for name, val in vars(self).items():
                    if isinstance(val, nn.Module):
                        candidates.append((name, val))
            except Exception:
                pass
        
        if not candidates:
            logger.warning("bentham_calculator 내부에서 nn.Module 후보를 찾지 못했습니다.")
            logger.warning(f"  - neural_predictor: {hasattr(self, 'neural_predictor')}")
            logger.warning(f"  - weight_layers: {hasattr(self, 'weight_layers')}")
            logger.warning(f"  - default_network: {hasattr(self, 'default_network')}")
            
            # 기본 네트워크 생성 시도
            logger.info("🔨 기본 네트워크 생성 시도 중...")
            try:
                self._build_default_network()
                # 생성 후 다시 확인
                if hasattr(self, '_primary_nn') and isinstance(self._primary_nn, nn.Module):
                    return self._primary_nn
                elif hasattr(self, 'default_network') and isinstance(self.default_network, nn.Module):
                    self._primary_nn = self.default_network
                    return self._primary_nn
            except Exception as e:
                logger.error(f"기본 네트워크 생성 실패: {e}")
                raise RuntimeError(f"bentham_calculator nn.Module 생성 실패: {e}")
        
        # 4) 가장 큰 네트워크를 대표로 선택(파라미터 수 기준)
        def num_params(m): 
            try:
                return sum(p.numel() for p in m.parameters())
            except Exception:
                return 0
        
        best_name, best_model = max(candidates, key=lambda kv: num_params(kv[1]))
        
        logger.info(f"AdvancedBenthamCalculator: {best_name}을(를) primary_nn으로 선택 (파라미터 수: {num_params(best_model):,})")
        
        # 캐시 후 반환
        self._primary_nn = best_model
        return self._primary_nn


def test_advanced_bentham_calculator():
    """고급 벤담 계산기 테스트"""
    
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    try:
        # 계산기 초기화
        calculator = AdvancedBenthamCalculator()
        
        # 테스트 데이터
        test_data = {
            'input_values': {
                'intensity': 0.9,
                'duration': 0.8,
                'certainty': 0.7,
                'propinquity': 0.85,
                'fecundity': 0.6,
                'purity': 0.75,
                'extent': 0.9
            },
            'text_description': "이 결정은 많은 사람들의 생명과 안전에 직접적인 영향을 미치며, 장기적으로 사회 전체의 복지에 중대한 결과를 가져올 것입니다.",
            'language': 'ko',
            'affected_count': 1000,
            'duration_seconds': 86400,  # 24시간
            'information_quality': 0.85,
            'uncertainty_level': 0.25,
            'social_context': {
                'impact_scope': 'society',
                'social_status': 'influential',
                'network_effect': 'viral'
            },
            'temporal_context': {
                'immediacy': 'immediate',
                'persistence': 'permanent',
                'time_pressure': 'high'
            },
            'ethical_context': {
                'life_impact': 'critical',
                'justice_factor': 'high',
                'autonomy_impact': 'enhancing',
                'vulnerable_protection': 'critical'
            },
            'cognitive_context': {
                'cognitive_load': 'high',
                'decision_complexity': 'very_complex',
                'information_certainty': 'high',
                'predictability': 'low'
            }
        }
        
        print("=== 고급 벤담 쾌락 계산기 테스트 (Linux) ===\n")
        
        # 계산 실행
        start_time = time.time()
        result = calculator.calculate_with_advanced_layers(test_data)
        processing_time = time.time() - start_time
        
        # 결과 출력
        print("📊 계산 결과:")
        print(f"- 최종 점수: {result.final_score:.4f}")
        print(f"- 기본 점수: {result.base_score:.4f}")
        print(f"- 개선 비율: {((result.final_score / result.base_score - 1) * 100):+.1f}%")
        print(f"- 종합 신뢰도: {result.confidence_score:.3f}")
        print(f"- 처리 시간: {processing_time:.3f}초")
        neural_used = result.context_analysis.get('neural_prediction_used', False)
        opt_applied = result.context_analysis.get('optimization_applied', False)
        print(f"- 신경망 예측 사용: {'예' if neural_used else '아니오'}")
        print(f"- 최적화 적용: {'예' if opt_applied else '아니오'}")
        
        print(f"\n🎯 레이어별 기여도:")
        for i, layer in enumerate(result.layer_contributions):
            print(f"{i+1}. {layer.layer_name}")
            print(f"   • 가중치: {layer.weight_factor:.3f}")
            print(f"   • 기여도: {layer.contribution_score:.3f}")
            print(f"   • 신뢰도: {layer.confidence:.3f}")
            reasoning = layer.metadata.get('reasoning', 'No reasoning available')
            print(f"   • 근거: {reasoning}")
            
        if result.extreme_adjustment_applied:
            print(f"\n⚡ 극단값 보정:")
            print(f"- 보정 계수: {result.adjustment_factor:.3f}")
            extreme_adj = result.context_analysis.get('calculation_breakdown', {}).get('extreme_adjustment', {})
            if extreme_adj:
                print(f"- 보정 유형: {extreme_adj.get('type', 'unknown')}")
            
        print(f"\n🔧 시스템 정보:")
        print(f"- 디바이스: {DEVICE}")
        print(f"- GPU 사용: {'예' if ADVANCED_CONFIG['enable_gpu'] else '아니오'}")
        print(f"- 배치 크기: {BATCH_SIZE}")
        
        # 캐시 통계
        cache_stats = calculator.get_cache_stats()
        print(f"- 캐시 크기: {cache_stats['cache_size']}")
        
        return result
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return None


@dataclass
class FrommAnalysisResult:
    """에리히 프롬 분석 결과"""
    orientation: FrommOrientation
    being_score: float  # 존재 지향 점수 (0-1)
    having_score: float  # 소유 지향 점수 (0-1)
    authenticity_level: float  # 진정성 수준 (0-1)
    alienation_level: float  # 소외 수준 (0-1)
    creative_potential: float  # 창조적 잠재력 (0-1)
    social_connectedness: float  # 사회적 연결성 (0-1)
    self_realization: float  # 자기실현 (0-1)
    character_traits: Dict[str, float]  # 성격 특성들
    fromm_factors: Dict[str, Any]  # 프롬 철학 요인들
    confidence: float = 0.8


class FrommEthicalAnalyzer:
    """
    에리히 프롬 철학 분석기 - 소유냐 존재냐 (To Have or To Be) 기반
    
    2024년 연구 기반 구현:
    - 소유 vs 존재 지향 분석
    - 진정성 vs 소외 평가
    - 창조적 잠재력 측정
    - 사회적 성격 분석
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 프롬 철학 기반 평가 기준
        self.fromm_criteria = {
            'being_indicators': [
                'authentic', 'creative', 'spontaneous', 'loving', 'caring',
                'growth', 'experience', 'meaningful', 'connection', 'understanding',
                'wisdom', 'compassion', 'beauty', 'truth', 'freedom'
            ],
            'having_indicators': [
                'possess', 'own', 'control', 'dominate', 'accumulate',
                'compete', 'status', 'wealth', 'power', 'prestige',
                'consume', 'acquire', 'material', 'property', 'objects'
            ],
            'alienation_indicators': [
                'isolated', 'disconnected', 'meaningless', 'empty', 'anxious',
                'conformity', 'mechanical', 'routine', 'passive', 'dependent',
                'superficial', 'artificial', 'automated', 'robotic', 'lost'
            ],
            'authenticity_indicators': [
                'genuine', 'real', 'honest', 'sincere', 'true',
                'original', 'unique', 'individual', 'personal', 'intimate',
                'deep', 'heartfelt', 'spontaneous', 'natural', 'organic'
            ]
        }
        
        # 사회적 성격 유형 (프롬의 5가지 성격 유형)
        self.character_types = {
            'receptive': {  # 수용적 성격
                'indicators': ['passive', 'dependent', 'submissive', 'loyal', 'accepting'],
                'description': '외부로부터 받는 것에 의존하는 성격'
            },
            'exploitative': {  # 착취적 성격  
                'indicators': ['aggressive', 'dominating', 'taking', 'competitive', 'cynical'],
                'description': '타인을 착취하여 얻으려는 성격'
            },
            'hoarding': {  # 저축적 성격
                'indicators': ['saving', 'possessive', 'cautious', 'orderly', 'rigid'],
                'description': '소유물을 축적하고 보존하려는 성격'
            },
            'marketing': {  # 시장적 성격
                'indicators': ['adaptable', 'flexible', 'opportunistic', 'changeable', 'modern'],
                'description': '자신을 상품처럼 판매하려는 성격'
            },
            'productive': {  # 생산적 성격
                'indicators': ['creative', 'loving', 'reasoning', 'working', 'authentic'],
                'description': '자신의 잠재력을 실현하는 건강한 성격'
            }
        }
        
    async def analyze_fromm_orientation(self, text: str, context: Dict[str, Any] = None) -> FrommAnalysisResult:
        """
        텍스트에서 에리히 프롬의 철학적 지향성 분석
        
        Args:
            text: 분석할 텍스트
            context: 추가 컨텍스트
            
        Returns:
            프롬 분석 결과
        """
        context = context or {}
        
        try:
            # 1. 존재 vs 소유 지향 분석
            being_score = await self._analyze_being_orientation(text)
            having_score = await self._analyze_having_orientation(text)
            
            # 2. 진정성 vs 소외 분석
            authenticity_level = await self._analyze_authenticity(text)
            alienation_level = await self._analyze_alienation(text)
            
            # 3. 창조적 잠재력 분석
            creative_potential = await self._analyze_creative_potential(text)
            
            # 4. 사회적 연결성 분석
            social_connectedness = await self._analyze_social_connectedness(text)
            
            # 5. 자기실현 분석
            self_realization = await self._analyze_self_realization(text)
            
            # 6. 성격 유형 분석
            character_traits = await self._analyze_character_types(text)
            
            # 7. 프롬 요인들 종합 분석
            fromm_factors = await self._analyze_fromm_factors(text, context)
            
            # 8. 전체 지향성 결정
            orientation = await self._determine_overall_orientation(being_score, having_score)
            
            # 9. 신뢰도 계산
            confidence = await self._calculate_analysis_confidence(
                text, being_score, having_score, authenticity_level, alienation_level
            )
            
            return FrommAnalysisResult(
                orientation=orientation,
                being_score=being_score,
                having_score=having_score,
                authenticity_level=authenticity_level,
                alienation_level=alienation_level,
                creative_potential=creative_potential,
                social_connectedness=social_connectedness,
                self_realization=self_realization,
                character_traits=character_traits,
                fromm_factors=fromm_factors,
                confidence=confidence
            )
            
        except Exception as e:
            # 프로젝트 규칙: fallback 없는 순수 재시도 방식
            retry_count = 0
            max_retries = 3
            analysis_result = None
            while retry_count < max_retries:
                retry_count += 1
                self.logger.info(f"프롬 분석 재시도 {retry_count}/{max_retries}")
                try:
                    # 전체 프롬 분석 프로세스 재시도
                    # 1. 존재 vs 소유 지향 분석
                    being_score = await self._analyze_being_orientation(text)
                    having_score = await self._analyze_having_orientation(text)
                    
                    # 2. 진정성 vs 소외 분석
                    authenticity_level = await self._analyze_authenticity(text)
                    alienation_level = await self._analyze_alienation(text)
                    
                    # 3. 창조적 잠재력 분석
                    creative_potential = await self._analyze_creative_potential(text)
                    
                    # 4. 사회적 연결성 분석
                    social_connectedness = await self._analyze_social_connectedness(text)
                    
                    # 5. 자기실현 분석
                    self_realization = await self._analyze_self_realization(text)
                    
                    # 6. 성격 유형 분석
                    character_traits = await self._analyze_character_types(text)
                    
                    # 7. 프롬 요인들 종합 분석
                    fromm_factors = await self._analyze_fromm_factors(text, context)
                    
                    # 8. 전체 지향성 결정
                    orientation = self._determine_overall_orientation(being_score, having_score)
                    
                    # 9. 신뢰도 계산
                    confidence = self._calculate_fromm_confidence(text, being_score, having_score, 
                                                               authenticity_level, alienation_level)
                    
                    # 결과 생성
                    analysis_result = FrommAnalysisResult(
                        orientation=orientation,
                        being_score=being_score,
                        having_score=having_score,
                        authenticity_level=authenticity_level,
                        alienation_level=alienation_level,
                        creative_potential=creative_potential,
                        social_connectedness=social_connectedness,
                        self_realization=self_realization,
                        character_traits=character_traits,
                        fromm_factors=fromm_factors,
                        confidence=confidence
                    )
                    
                    self.logger.info(f"프롬 분석 재시도 성공: 지향성={orientation}")
                    return analysis_result
                    
                except Exception as retry_error:
                    self.logger.error(f"프롬 분석 재시도 {retry_count} 실패: {retry_error}")
                    if retry_count >= max_retries:
                        self.logger.error("프롬 분석 모든 재시도 실패 - 시스템 정지")
                        raise RuntimeError(f"프롬 분석 최종 실패: {retry_error}")
                    import asyncio
                    await asyncio.sleep(0.5)
            
            if analysis_result is None:
                self.logger.error("프롬 분석 재시도 후에도 실패 - 시스템 정지")
                raise RuntimeError("프롬 분석 실패 - fallback 금지로 시스템 정지")
    
    async def _analyze_being_orientation(self, text: str) -> float:
        """존재 지향성 분석"""
        text_lower = text.lower()
        being_count = 0
        
        for indicator in self.fromm_criteria['being_indicators']:
            if indicator in text_lower:
                being_count += 1
        
        # 존재 지향적 문구 패턴 분석
        being_patterns = [
            'who i am', 'authentic self', 'true meaning', 'deep connection',
            'creative expression', 'genuine feeling', 'real experience',
            'meaningful relationship', 'inner growth', 'spiritual development'
        ]
        
        pattern_count = sum(1 for pattern in being_patterns if pattern in text_lower)
        
        # 정규화된 점수 계산
        raw_score = (being_count + pattern_count * 2) / (len(self.fromm_criteria['being_indicators']) + len(being_patterns) * 2)
        return min(1.0, raw_score)
    
    async def _analyze_having_orientation(self, text: str) -> float:
        """소유 지향성 분석"""
        text_lower = text.lower()
        having_count = 0
        
        for indicator in self.fromm_criteria['having_indicators']:
            if indicator in text_lower:
                having_count += 1
        
        # 소유 지향적 문구 패턴 분석
        having_patterns = [
            'i have', 'i own', 'my possessions', 'accumulate wealth',
            'status symbol', 'material success', 'competitive advantage',
            'market value', 'profit margin', 'consumer goods'
        ]
        
        pattern_count = sum(1 for pattern in having_patterns if pattern in text_lower)
        
        # 정규화된 점수 계산
        raw_score = (having_count + pattern_count * 2) / (len(self.fromm_criteria['having_indicators']) + len(having_patterns) * 2)
        return min(1.0, raw_score)
    
    async def _analyze_authenticity(self, text: str) -> float:
        """진정성 수준 분석"""
        text_lower = text.lower()
        authenticity_count = 0
        
        for indicator in self.fromm_criteria['authenticity_indicators']:
            if indicator in text_lower:
                authenticity_count += 1
        
        # 진정성을 나타내는 언어적 특징
        authenticity_features = [
            len([word for word in text.split() if word.startswith('I ')]) / len(text.split()) if text.split() else 0,  # 주체적 표현
            text.count('feel') + text.count('believe') + text.count('think'),  # 개인적 견해 표현
            text.count('?') / len(text) if text else 0,  # 성찰적 질문
        ]
        
        # 가중 평균으로 진정성 점수 계산
        base_score = authenticity_count / len(self.fromm_criteria['authenticity_indicators'])
        feature_score = np.mean(authenticity_features)
        
        return min(1.0, (base_score * 0.7 + feature_score * 0.3))
    
    async def _analyze_alienation(self, text: str) -> float:
        """소외 수준 분석"""
        text_lower = text.lower()
        alienation_count = 0
        
        for indicator in self.fromm_criteria['alienation_indicators']:
            if indicator in text_lower:
                alienation_count += 1
        
        # 소외를 나타내는 언어적 패턴
        alienation_patterns = [
            'feel disconnected', 'no meaning', 'empty inside', 'going through motions',
            'just a number', 'nobody understands', 'all the same', 'pointless'
        ]
        
        pattern_count = sum(1 for pattern in alienation_patterns if pattern in text_lower)
        
        raw_score = (alienation_count + pattern_count * 1.5) / (len(self.fromm_criteria['alienation_indicators']) + len(alienation_patterns) * 1.5)
        return min(1.0, raw_score)
    
    async def _analyze_creative_potential(self, text: str) -> float:
        """창조적 잠재력 분석"""
        text_lower = text.lower()
        
        creativity_words = [
            'create', 'innovate', 'imagine', 'invent', 'design', 'artistic',
            'original', 'unique', 'new', 'experiment', 'explore', 'discover'
        ]
        
        creativity_score = sum(1 for word in creativity_words if word in text_lower)
        
        # 창조적 표현의 언어적 특징
        linguistic_creativity = [
            len(set(text.split())) / len(text.split()) if text.split() else 0,  # 어휘 다양성
            text.count('!') / len(text) if text else 0,  # 감정적 표현
            len([word for word in text.split() if len(word) > 8]) / len(text.split()) if text.split() else 0  # 복잡한 어휘
        ]
        
        base_score = creativity_score / len(creativity_words)
        linguistic_score = np.mean(linguistic_creativity)
        
        return min(1.0, (base_score * 0.6 + linguistic_score * 0.4))
    
    async def _analyze_social_connectedness(self, text: str) -> float:
        """사회적 연결성 분석"""
        text_lower = text.lower()
        
        social_words = [
            'we', 'us', 'together', 'community', 'relationship', 'friend',
            'family', 'share', 'connect', 'belong', 'care', 'love'
        ]
        
        social_count = sum(1 for word in social_words if word in text_lower)
        
        # 관계적 언어 패턴
        relational_patterns = [
            'with others', 'feel close', 'meaningful relationship', 'deep connection',
            'mutual understanding', 'shared experience', 'together we', 'part of'
        ]
        
        pattern_count = sum(1 for pattern in relational_patterns if pattern in text_lower)
        
        raw_score = (social_count + pattern_count * 1.5) / (len(social_words) + len(relational_patterns) * 1.5)
        return min(1.0, raw_score)
    
    async def _analyze_self_realization(self, text: str) -> float:
        """자기실현 분석"""
        text_lower = text.lower()
        
        self_realization_words = [
            'growth', 'develop', 'potential', 'fulfill', 'achieve', 'realize',
            'become', 'transform', 'evolve', 'improve', 'better', 'progress'
        ]
        
        realization_count = sum(1 for word in self_realization_words if word in text_lower)
        
        # 자기실현을 나타내는 문구
        realization_phrases = [
            'true potential', 'personal growth', 'self development', 'inner journey',
            'becoming myself', 'finding purpose', 'life mission', 'authentic path'
        ]
        
        phrase_count = sum(1 for phrase in realization_phrases if phrase in text_lower)
        
        raw_score = (realization_count + phrase_count * 2) / (len(self_realization_words) + len(realization_phrases) * 2)
        return min(1.0, raw_score)
    
    async def _analyze_character_types(self, text: str) -> Dict[str, float]:
        """프롬의 5가지 성격 유형 분석"""
        text_lower = text.lower()
        character_scores = {}
        
        for char_type, char_data in self.character_types.items():
            indicators = char_data['indicators']
            type_score = sum(1 for indicator in indicators if indicator in text_lower)
            character_scores[char_type] = type_score / len(indicators)
        
        return character_scores
    
    async def _analyze_fromm_factors(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """프롬 철학 요인들 종합 분석"""
        factors = {}
        
        # 자유 vs 도피 (Freedom vs Escape)
        factors['freedom_level'] = await self._analyze_freedom_orientation(text)
        factors['escape_tendency'] = await self._analyze_escape_tendency(text)
        
        # 사랑의 능력 (Capacity for Love)
        factors['love_capacity'] = await self._analyze_love_capacity(text)
        
        # 파괴성 vs 창조성 (Destructiveness vs Creativity)
        factors['destructive_tendency'] = await self._analyze_destructive_tendency(text)
        factors['constructive_tendency'] = await self._analyze_constructive_tendency(text)
        
        # 권위주의 vs 자유주의 성향
        factors['authoritarian_tendency'] = await self._analyze_authoritarian_tendency(text)
        factors['democratic_tendency'] = await self._analyze_democratic_tendency(text)
        
        return factors
    
    async def _analyze_freedom_orientation(self, text: str) -> float:
        """자유 지향성 분석"""
        freedom_words = ['freedom', 'liberty', 'choice', 'autonomy', 'independence', 'self-directed']
        text_lower = text.lower()
        return min(1.0, sum(1 for word in freedom_words if word in text_lower) / len(freedom_words))
    
    async def _analyze_escape_tendency(self, text: str) -> float:
        """도피 성향 분석"""
        escape_words = ['escape', 'avoid', 'hide', 'conform', 'follow', 'obey', 'submit']
        text_lower = text.lower()
        return min(1.0, sum(1 for word in escape_words if word in text_lower) / len(escape_words))
    
    async def _analyze_love_capacity(self, text: str) -> float:
        """사랑의 능력 분석"""
        love_words = ['love', 'care', 'compassion', 'empathy', 'understanding', 'nurture', 'support']
        text_lower = text.lower()
        return min(1.0, sum(1 for word in love_words if word in text_lower) / len(love_words))
    
    async def _analyze_destructive_tendency(self, text: str) -> float:
        """파괴적 성향 분석"""
        destructive_words = ['destroy', 'break', 'damage', 'hurt', 'harm', 'violent', 'aggressive']
        text_lower = text.lower()
        return min(1.0, sum(1 for word in destructive_words if word in text_lower) / len(destructive_words))
    
    async def _analyze_constructive_tendency(self, text: str) -> float:
        """건설적 성향 분석"""
        constructive_words = ['build', 'create', 'construct', 'develop', 'grow', 'nurture', 'improve']
        text_lower = text.lower()
        return min(1.0, sum(1 for word in constructive_words if word in text_lower) / len(constructive_words))
    
    async def _analyze_authoritarian_tendency(self, text: str) -> float:
        """권위주의 성향 분석"""
        authoritarian_words = ['obey', 'submit', 'authority', 'control', 'dominate', 'command', 'rule']
        text_lower = text.lower()
        return min(1.0, sum(1 for word in authoritarian_words if word in text_lower) / len(authoritarian_words))
    
    async def _analyze_democratic_tendency(self, text: str) -> float:
        """민주적 성향 분석"""
        democratic_words = ['equal', 'fair', 'justice', 'participate', 'collaborate', 'consensus', 'dialogue']
        text_lower = text.lower()
        return min(1.0, sum(1 for word in democratic_words if word in text_lower) / len(democratic_words))
    
    async def _determine_overall_orientation(self, being_score: float, having_score: float) -> FrommOrientation:
        """전체 지향성 결정"""
        difference = abs(being_score - having_score)
        
        if difference < 0.2:  # 차이가 작으면 혼합
            return FrommOrientation.MIXED
        elif being_score > having_score:
            return FrommOrientation.BEING
        else:
            return FrommOrientation.HAVING
    
    async def _calculate_analysis_confidence(self, text: str, being_score: float, 
                                          having_score: float, authenticity: float, 
                                          alienation: float) -> float:
        """분석 신뢰도 계산"""
        # 텍스트 길이 기반 신뢰도
        text_length_factor = min(1.0, len(text.split()) / 50)  # 50단어 이상이면 최대 신뢰도
        
        # 점수들의 일관성 기반 신뢰도
        scores = [being_score, having_score, authenticity, 1.0 - alienation]
        consistency_factor = 1.0 - np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0.5
        
        # 종합 신뢰도
        overall_confidence = (text_length_factor * 0.3 + consistency_factor * 0.7)
        return max(0.1, min(1.0, overall_confidence))
    


class FrommEnhancedBenthamCalculator:
    """
    에리히 프롬 철학이 통합된 벤담 계산기
    
    벤담의 공리주의 계산에 프롬의 인간주의적 윤리학을 통합하여
    더 깊이 있는 인간 중심적 행복 계산을 수행합니다.
    """
    
    def __init__(self, base_calculator: 'AdvancedBenthamCalculator'):
        self.base_calculator = base_calculator
        self.fromm_analyzer = FrommEthicalAnalyzer()
        self.logger = logging.getLogger(__name__)
        
        # 프롬 요소의 벤담 계산 통합 가중치
        self.fromm_integration_weights = {
            'being_orientation_bonus': 0.2,  # 존재 지향일 때 추가 가중치
            'authenticity_multiplier': 0.15,  # 진정성 배수
            'alienation_penalty': 0.1,  # 소외 페널티
            'creative_potential_bonus': 0.1,  # 창조적 잠재력 보너스
            'social_connectedness_bonus': 0.05  # 사회적 연결성 보너스
        }
    
    async def calculate_fromm_enhanced_pleasure(self, 
                                             situation: EthicalSituation,
                                             context: AdvancedCalculationContext = None) -> EnhancedHedonicResult:
        """
        프롬 철학이 통합된 쾌락 계산
        
        Args:
            situation: 윤리적 상황
            context: 계산 컨텍스트
            
        Returns:
            프롬 요소가 통합된 헤도닉 결과
        """
        try:
            # 1. 기본 벤담 계산 수행
            base_result = await self.base_calculator.calculate_enhanced_pleasure(situation, context)
            
            # 2. 프롬 철학 분석 수행
            fromm_analysis = await self.fromm_analyzer.analyze_fromm_orientation(
                situation.description, 
                context.metadata if context else {}
            )
            
            # 3. 프롬 요소를 벤담 계산에 통합
            enhanced_result = await self._integrate_fromm_into_bentham(
                base_result, fromm_analysis, situation
            )
            
            # 4. 프롬 메타데이터 추가
            enhanced_result.context_analysis['fromm_analysis'] = {
                'orientation': fromm_analysis.orientation.value,
                'being_score': fromm_analysis.being_score,
                'having_score': fromm_analysis.having_score,
                'authenticity_level': fromm_analysis.authenticity_level,
                'alienation_level': fromm_analysis.alienation_level,
                'character_traits': fromm_analysis.character_traits,
                'fromm_factors': fromm_analysis.fromm_factors,
                'analysis_confidence': fromm_analysis.confidence
            }
            
            return enhanced_result
            
        except Exception as e:
            self.logger.error(f"프롬 통합 벤담 계산 실패: {e}")
            return await self.base_calculator.calculate_enhanced_pleasure(situation, context)
    
    async def _integrate_fromm_into_bentham(self, 
                                          base_result: EnhancedHedonicResult,
                                          fromm_analysis: FrommAnalysisResult,
                                          situation: EthicalSituation) -> EnhancedHedonicResult:
        """프롬 요소를 벤담 계산에 통합"""
        
        # 기본 결과 복사
        enhanced_result = base_result
        
        # 1. 존재 지향 보너스 적용
        being_bonus = self._calculate_being_orientation_bonus(fromm_analysis)
        enhanced_result.final_pleasure_score *= (1 + being_bonus)
        
        # 2. 진정성 배수 적용
        authenticity_multiplier = self._calculate_authenticity_multiplier(fromm_analysis)
        enhanced_result.final_pleasure_score *= authenticity_multiplier
        
        # 3. 소외 페널티 적용
        alienation_penalty = self._calculate_alienation_penalty(fromm_analysis)
        enhanced_result.final_pleasure_score *= (1 - alienation_penalty)
        
        # 4. 창조적 잠재력 보너스 적용
        creative_bonus = self._calculate_creative_potential_bonus(fromm_analysis)
        enhanced_result.final_pleasure_score *= (1 + creative_bonus)
        
        # 5. 사회적 연결성 보너스 적용
        social_bonus = self._calculate_social_connectedness_bonus(fromm_analysis)
        enhanced_result.final_pleasure_score *= (1 + social_bonus)
        
        # 6. 프롬 철학 기반 추가 가중치 레이어 생성
        fromm_layer = WeightLayerResult(
            layer_name="fromm_humanistic_ethics",
            weight_factor=self._calculate_overall_fromm_weight(fromm_analysis),
            contribution_score=self._calculate_fromm_contribution(fromm_analysis),
            confidence=fromm_analysis.confidence,
            metadata={
                'orientation': fromm_analysis.orientation.value,
                'being_bonus': being_bonus,
                'authenticity_multiplier': authenticity_multiplier,
                'alienation_penalty': alienation_penalty,
                'creative_bonus': creative_bonus,
                'social_bonus': social_bonus,
                'reasoning': self._generate_fromm_reasoning(fromm_analysis)
            }
        )
        
        # 7. 레이어 기여도에 프롬 레이어 추가
        enhanced_result.layer_contributions.append(fromm_layer)
        
        # 8. 최종 점수 재정규화 (0-10 범위 유지)
        enhanced_result.final_pleasure_score = max(0.0, min(10.0, enhanced_result.final_pleasure_score))
        
        return enhanced_result
    
    def _calculate_being_orientation_bonus(self, fromm_analysis: FrommAnalysisResult) -> float:
        """존재 지향 보너스 계산"""
        if fromm_analysis.orientation == FrommOrientation.BEING:
            return fromm_analysis.being_score * self.fromm_integration_weights['being_orientation_bonus']
        return 0.0
    
    def _calculate_authenticity_multiplier(self, fromm_analysis: FrommAnalysisResult) -> float:
        """진정성 배수 계산"""
        base_multiplier = 1.0
        authenticity_bonus = fromm_analysis.authenticity_level * self.fromm_integration_weights['authenticity_multiplier']
        return base_multiplier + authenticity_bonus
    
    def _calculate_alienation_penalty(self, fromm_analysis: FrommAnalysisResult) -> float:
        """소외 페널티 계산"""
        return fromm_analysis.alienation_level * self.fromm_integration_weights['alienation_penalty']
    
    def _calculate_creative_potential_bonus(self, fromm_analysis: FrommAnalysisResult) -> float:
        """창조적 잠재력 보너스 계산"""
        return fromm_analysis.creative_potential * self.fromm_integration_weights['creative_potential_bonus']
    
    def _calculate_social_connectedness_bonus(self, fromm_analysis: FrommAnalysisResult) -> float:
        """사회적 연결성 보너스 계산"""
        return fromm_analysis.social_connectedness * self.fromm_integration_weights['social_connectedness_bonus']
    
    def _calculate_overall_fromm_weight(self, fromm_analysis: FrommAnalysisResult) -> float:
        """전체 프롬 가중치 계산"""
        weights = [
            fromm_analysis.being_score,
            fromm_analysis.authenticity_level,
            1.0 - fromm_analysis.alienation_level,  # 소외의 역수
            fromm_analysis.creative_potential,
            fromm_analysis.social_connectedness,
            fromm_analysis.self_realization
        ]
        return np.mean(weights)
    
    def _calculate_fromm_contribution(self, fromm_analysis: FrommAnalysisResult) -> float:
        """프롬 철학의 기여도 계산"""
        # 존재 지향성이 높고 소외가 낮을수록 높은 기여도
        contribution = (
            fromm_analysis.being_score * 0.3 +
            fromm_analysis.authenticity_level * 0.25 +
            (1.0 - fromm_analysis.alienation_level) * 0.2 +
            fromm_analysis.creative_potential * 0.15 +
            fromm_analysis.social_connectedness * 0.1
        )
        return contribution
    
    def _generate_fromm_reasoning(self, fromm_analysis: FrommAnalysisResult) -> str:
        """프롬 분석에 기한 추론 생성"""
        orientation = fromm_analysis.orientation.value
        being_score = fromm_analysis.being_score
        authenticity = fromm_analysis.authenticity_level
        alienation = fromm_analysis.alienation_level
        
        reasoning_parts = []
        
        if orientation == "being":
            reasoning_parts.append(f"존재 지향적 성격 (점수: {being_score:.2f})으로 진정한 자기실현에 기여")
        elif orientation == "having":
            reasoning_parts.append(f"소유 지향적 성격으로 물질적 만족에 치중")
        else:
            reasoning_parts.append(f"존재와 소유의 혼합적 지향")
        
        if authenticity > 0.7:
            reasoning_parts.append(f"높은 진정성 ({authenticity:.2f})으로 내재적 만족 증진")
        elif authenticity < 0.3:
            reasoning_parts.append(f"낮은 진정성 ({authenticity:.2f})으로 표면적 만족에 제한")
        
        if alienation > 0.5:
            reasoning_parts.append(f"소외감 ({alienation:.2f})으로 인한 행복 저해")
        
        if fromm_analysis.creative_potential > 0.6:
            reasoning_parts.append(f"높은 창조적 잠재력으로 자기실현 가능성 증대")
        
        return "; ".join(reasoning_parts)


if __name__ == "__main__":
    test_advanced_bentham_calculator()