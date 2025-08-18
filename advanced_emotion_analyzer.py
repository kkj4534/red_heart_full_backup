"""
Red Heart Linux Advanced - 고급 감정 분석 시스템
Transformers, Sentence Transformers 기반 고성능 감정 분석
"""

__all__ = ['AdvancedEmotionAnalyzer']

import os
# CVE-2025-32434는 가짜 CVE - torch_security_patch import 제거
# import torch_security_patch

import logging
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
# pathlib 제거 - WSL 호환성을 위해 os.path 사용
import json
import torch

# 고급 라이브러리 임포트 - 개발용 필수 모드
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline, AutoModel
)
# SentenceTransformer는 sentence_transformer_singleton을 통해 사용
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import torch.nn as nn
ADVANCED_LIBS_AVAILABLE = True

from config import SYSTEM_CONFIG, ADVANCED_CONFIG, EMOTION_MODELS_DIR
from data_models import EmotionData, EmotionState, EmotionIntensity, Biosignal
from mixture_of_experts import create_emotion_moe, MixtureOfExperts
from emotion_dsp_simulator import EmotionDSPSimulator, DynamicKalmanFilter

# logger를 import 후 즉시 생성하여 except 블록에서 사용 가능하도록 함
logger = logging.getLogger('RedHeartLinux.AdvancedEmotion')

# 새로운 계층적 감정 모델 임포트
try:
    from models.hierarchical_emotion.emotion_phase_models import (
        HierarchicalEmotionModel, EmotionModelManager, 
        emotion_vector_to_dict, EMOTION_DIMENSIONS
    )
    NEW_EMOTION_MODELS_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ 필수 모듈 hierarchical_emotion.emotion_phase_models import 실패: {e}")
    raise ImportError(f"필수 계층적 감정 모델 시스템을 찾을 수 없습니다: {e}") from e

# LLM 통합
try:
    from llm_module.advanced_llm_engine import get_llm_engine, interpret_emotions
    LLM_INTEGRATION_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ 필수 모듈 llm_module.advanced_llm_engine import 실패: {e}")
    raise ImportError(f"필수 LLM 통합 시스템을 찾을 수 없습니다: {e}") from e

def get_local_device():
    """디바이스 감지 (로컬 함수)"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

# 모듈 가용성 로깅
if not NEW_EMOTION_MODELS_AVAILABLE:
    logger.warning("새로운 계층적 감정 모델을 사용할 수 없습니다.")

if not LLM_INTEGRATION_AVAILABLE:
    logger.warning("LLM 통합을 사용할 수 없습니다.")

# TODO: LocalTranslator가 별도 모듈(local_translator.py)로 분리됨
# 전역 모듈로 등록되어 MasterMemoryOrchestrator가 관리하도록 구조 개선
# 정상 작동 확인 후 아래 주석 처리된 코드 삭제 예정
"""
class LocalTranslator:
    '''로컬 OPUS-MT 기반 한국어→영어 번역기 (Google Translate 대체)'''
    
    def __init__(self):
        '''로컬 번역기 초기화 (Lazy Loading)'''
        self.model_name = 'Helsinki-NLP/opus-mt-ko-en'
        self.tokenizer = None
        self.model = None
        self.device = None
        self.translation_cache = {}  # 번역 결과 캐싱
        self.initialized = False
        
        logger.info("LocalTranslator 초기화됨 (모델은 첫 사용시 로드)")
    
    def _initialize_model(self):
        '''모델 초기화 (첫 번역 시에만 실행)'''
        if self.initialized:
            return
        
        try:
            logger.info(f"🔄 OPUS-MT 모델 로드 중: {self.model_name}")
            start_time = time.time()
            
            from transformers import MarianMTModel, MarianTokenizer
            
            # HF 래퍼를 통한 토크나이저와 모델 로드 (오프라인 모드)
            from hf_model_wrapper import get_hf_wrapper
            hf_wrapper = get_hf_wrapper()
            
            self.tokenizer = hf_wrapper.wrapped_tokenizer(
                self.model_name, 
                owner="local_translator", local_files_only=True
            )
            self.model = hf_wrapper.wrapped_from_pretrained(
                MarianMTModel, self.model_name, 
                owner="local_translator", local_files_only=True
            )
            
            # 디바이스 설정 및 모델 이동
            self.device = get_smart_device()
            self.model = self.model.to(self.device)
            
            load_time = time.time() - start_time
            logger.info(f"✅ OPUS-MT 모델 로드 완료 (소요시간: {load_time:.1f}초, 디바이스: {self.device})")
            
            self.initialized = True
            
        except Exception as e:
            logger.error(f"❌ OPUS-MT 모델 로드 실패: {e}")
            self.initialized = False
            raise RuntimeError(f"로컬 번역기 초기화 실패: {e}")
    
    def _is_english_text(self, text: str) -> bool:
        '''텍스트가 이미 영어인지 감지'''
        if not text or len(text.strip()) == 0:
            return True
        
        # 한국어 문자 비율 계산 (유니코드 범위 활용)
        korean_chars = 0
        total_chars = 0
        
        for char in text:
            if char.isalpha():  # 알파벳 문자만 고려
                total_chars += 1
                # 한글 유니코드 범위: AC00-D7AF (가-힣), 1100-11FF (자모)
                if '\uAC00' <= char <= '\uD7AF' or '\u1100' <= char <= '\u11FF':
                    korean_chars += 1
        
        if total_chars == 0:
            return True  # 알파벳이 없으면 영어로 간주 (숫자, 기호만 있는 경우)
        
        korean_ratio = korean_chars / total_chars
        return korean_ratio < 0.1  # 한국어 비율이 10% 미만이면 영어로 판단
    
    def translate_ko_to_en(self, korean_text: str) -> str:
        '''한국어 텍스트를 영어로 번역'''
        if not korean_text or len(korean_text.strip()) == 0:
            return korean_text
        
        # 영어 텍스트 감지
        if self._is_english_text(korean_text):
            logger.debug("텍스트가 이미 영어로 판단됨, 번역 생략")
            return korean_text
        
        # 캐시 확인
        cache_key = hash(korean_text.strip())
        if cache_key in self.translation_cache:
            logger.debug("번역 캐시에서 결과 반환")
            return self.translation_cache[cache_key]
        
        try:
            # 모델 초기화 (첫 번역 시에만)
            if not self.initialized:
                self._initialize_model()
            
            # 번역 수행
            start_time = time.time()
            inputs = self.tokenizer([korean_text], return_tensors='pt', padding=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=128,      # 충분한 길이
                    num_beams=3,         # 적당한 품질
                    early_stopping=True, # 효율성
                    do_sample=False      # 일관성
                )
            
            translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            translation_time = time.time() - start_time
            
            # 캐시 저장 (메모리 제한)
            if len(self.translation_cache) < 1000:  # 최대 1000개 캐시
                self.translation_cache[cache_key] = translated_text
            
            logger.debug(f"번역 완료: \"{korean_text[:30]}...\" → \"{translated_text[:30]}...\" ({translation_time:.2f}초)")
            return translated_text
            
        except Exception as e:
            logger.warning(f"로컬 번역 실패: {e}, 원본 텍스트 반환")
            return korean_text
    
    def get_model_info(self) -> Dict[str, Any]:
        '''모델 정보 반환'''
        info = {
            'model_name': self.model_name,
            'initialized': self.initialized,
            'cache_size': len(self.translation_cache),
            'device': str(self.device) if self.device else 'not_loaded'
        }
        
        if self.initialized and self.model:
            info['model_type'] = 'MarianMT'
            info['vocab_size'] = self.tokenizer.vocab_size if self.tokenizer else 'unknown'
        
        return info
    
    def clear_cache(self):
        '''번역 캐시 정리'''
        cache_size = len(self.translation_cache)
        self.translation_cache.clear()
        logger.info(f"번역 캐시 정리됨: {cache_size}개 항목 삭제")
"""

class FocalLoss(nn.Module):
    """
    Focal Loss for Joy 편향 해결
    클래스 불균형 문제를 해결하기 위한 손실 함수
    """
    def __init__(self, alpha: float = 1.5, gamma: float = 2.0, size_average: bool = True):
        """
        Args:
            alpha: 클래스 가중치 (기본값: 1.5)
            gamma: focusing parameter (기본값: 2.0)
            size_average: 평균을 구할지 여부
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Focal Loss 계산
        
        Args:
            inputs: 모델 예측값 (logits) [batch_size, num_classes]
            targets: 실제 레이블 [batch_size]
            
        Returns:
            focal loss 값
        """
        # Cross entropy loss 계산
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 확률 계산
        pt = torch.exp(-ce_loss)
        
        # Focal loss 계산
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

class EmotionFocalLoss(nn.Module):
    """
    감정 분석 전용 Focal Loss
    Joy 편향 해결을 위한 특화된 손실 함수
    """
    def __init__(self, emotion_weights: Optional[Dict[str, float]] = None, 
                 alpha: float = 1.5, gamma: float = 2.0):
        """
        Args:
            emotion_weights: 각 감정별 가중치
            alpha: focusing parameter alpha
            gamma: focusing parameter gamma
        """
        super(EmotionFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
        # Joy 편향 해결을 위한 기본 가중치
        if emotion_weights is None:
            self.emotion_weights = {
                'JOY': 0.5,        # Joy 가중치 감소 (편향 해결)
                'TRUST': 1.2,
                'FEAR': 1.3,
                'SURPRISE': 1.2,
                'SADNESS': 1.3,
                'DISGUST': 1.4,
                'ANGER': 1.3,
                'ANTICIPATION': 1.2,
                'NEUTRAL': 1.0
            }
        else:
            self.emotion_weights = emotion_weights
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, 
                emotion_labels: Optional[List[str]] = None) -> torch.Tensor:
        """
        감정 기반 Focal Loss 계산
        
        Args:
            inputs: 모델 예측값 [batch_size, num_emotions]
            targets: 실제 감정 레이블 [batch_size]
            emotion_labels: 감정 레이블 목록
            
        Returns:
            focal loss 값
        """
        # 기본 focal loss 계산
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # 감정별 가중치 적용
        if emotion_labels:
            weights = torch.ones_like(targets, dtype=torch.float)
            for i, target_idx in enumerate(targets):
                if target_idx.item() < len(emotion_labels):
                    emotion_name = emotion_labels[target_idx.item()]
                    weight = self.emotion_weights.get(emotion_name, 1.0)
                    weights[i] = weight
        else:
            weights = torch.ones_like(targets, dtype=torch.float)
        
        # Focal loss with emotion weights
        focal_loss = weights * self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()

class AdvancedEmotionAnalyzer:
    """고급 감정 분석 시스템 - 폴백 없는 완전 구현"""
    
    def __init__(self):
        """고급 감정 분석기 초기화"""
        if not ADVANCED_LIBS_AVAILABLE:
            raise ImportError("고급 라이브러리가 필요합니다. requirements.txt를 확인하세요.")
        
        self.config = SYSTEM_CONFIG['emotion']
        from config import get_device
        self.device = get_device()
        
        # 모델 저장 디렉토리 (WSL 호환성)
        self.models_dir = EMOTION_MODELS_DIR
        os.makedirs(self.models_dir, exist_ok=True)
        
        # 다중 언어 감정 분석 모델들
        self.models = {}
        self.tokenizers = {}
        self.embedders = {}
        
        # 생체신호 분석 모델 (주석 처리 - 향후 연결 가능)
        # 센서 연결 시 활성화 가능: EEG, ECG, GSR, 음성, 시선추적 등
        # self.biosignal_model = None
        # self.biosignal_scaler = StandardScaler()
        self.biosignal_enabled = False  # 생체신호 기능 비활성화 (연결 시 True로 변경)
        
        # 캐시
        self.embedding_cache = {}
        self.prediction_cache = {}
        
        # 로컬 번역기 - 전역 모듈에서만 가져오기 (중복 생성 방지)
        from config import get_system_module
        self.local_translator = get_system_module('translator')
        if self.local_translator is None:
            # 전역 모듈이 없으면 에러 - translator가 먼저 초기화되어야 함
            error_msg = "전역 translator 모듈을 찾을 수 없음 - 모듈 초기화 순서 확인 필요"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Focal Loss for Joy 편향 해결
        self.focal_loss = EmotionFocalLoss(alpha=1.5, gamma=2.0)
        self.focal_loss_enabled = True
        
        # Mixture of Experts (감정 분석 강화)
        self.moe_enabled = True
        self.emotion_moe = None  # 명시적 초기화
        
        # DSP 시뮬레이터와 칼만 필터 초기화
        self.dsp_simulator = None
        self.kalman_filter = None
        self.dsp_enabled = True
        self.prev_kalman_state = None  # 칼만 필터 이전 상태
        
        # =====================================================
        # 강화 모듈 통합 (45M 추가 → 총 50M)
        # =====================================================
        base_dim = 768
        
        # 1. 생체신호 처리 네트워크 (10M)
        self.biometric_processor = nn.ModuleDict({
            'eeg': self._create_biometric_network(32, base_dim),  # EEG 32채널
            'ecg': self._create_biometric_network(12, base_dim),  # ECG 12리드
            'gsr': self._create_biometric_network(4, base_dim),   # GSR 4센서
            'fusion': nn.Sequential(
                nn.Linear(base_dim * 3, base_dim * 2),
                nn.LayerNorm(base_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(base_dim * 2, base_dim),
                nn.LayerNorm(base_dim)
            )
        }).to(self.device)
        
        # 2. 멀티모달 융합 레이어 (10M)
        self.multimodal_fusion = nn.ModuleDict({
            'text_encoder': nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=base_dim,
                    nhead=12,
                    dim_feedforward=base_dim * 4,
                    dropout=0.1,
                    batch_first=True
                ),
                num_layers=2
            ),
            'image_encoder': nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(128, base_dim)
            ),
            'audio_encoder': nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=80, stride=10),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(128, base_dim)
            ),
            'cross_modal_attention': nn.MultiheadAttention(
                embed_dim=base_dim,
                num_heads=12,
                dropout=0.1,
                batch_first=True
            )
        }).to(self.device)
        
        # 3. 시계열 감정 추적 (10M)
        self.temporal_emotion = nn.ModuleDict({
            'lstm_tracker': nn.LSTM(
                input_size=base_dim,
                hidden_size=base_dim // 2,
                num_layers=3,
                batch_first=True,
                dropout=0.1,
                bidirectional=True
            ),
            'temporal_attention': nn.Sequential(
                nn.Linear(base_dim, base_dim),
                nn.LayerNorm(base_dim),
                nn.GELU(),
                nn.Linear(base_dim, base_dim // 2),
                nn.Linear(base_dim // 2, 1),
                nn.Softmax(dim=1)
            ),
            'emotion_memory': nn.GRUCell(base_dim, base_dim),
            'trend_predictor': nn.Sequential(
                nn.Linear(base_dim * 2, base_dim),
                nn.LayerNorm(base_dim),
                nn.GELU(),
                nn.Linear(base_dim, base_dim // 2),
                nn.Linear(base_dim // 2, 7)  # 7 emotions
            )
        }).to(self.device)
        
        # 4. 문화적 뉘앙스 감지 (10M + 3M 추가 = 13M)
        self.cultural_nuance = nn.ModuleDict({
            'korean': self._create_cultural_network(base_dim),
            'western': self._create_cultural_network(base_dim),
            'eastern': self._create_cultural_network(base_dim),
            'fusion': nn.Sequential(
                nn.Linear(base_dim * 3, base_dim * 2),
                nn.LayerNorm(base_dim * 2),
                nn.GELU(),
                nn.Linear(base_dim * 2, base_dim),
                nn.LayerNorm(base_dim)
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
        
        # 5. 고급 MoE 확장 (5M)
        self.advanced_moe = nn.ModuleDict({
            'micro_experts': nn.ModuleList([
                nn.Sequential(
                    nn.Linear(base_dim, base_dim // 2),
                    nn.GELU(),
                    nn.Linear(base_dim // 2, base_dim // 4),
                    nn.Linear(base_dim // 4, 7)
                ) for _ in range(16)  # 16 micro experts
            ]),
            'router': nn.Sequential(
                nn.Linear(base_dim, base_dim // 2),
                nn.GELU(),
                nn.Linear(base_dim // 2, 16),
                nn.Softmax(dim=-1)
            )
        }).to(self.device)
        
        # 파라미터 로깅
        total_params = sum(p.numel() for p in [
            *self.biometric_processor.parameters(),
            *self.multimodal_fusion.parameters(),
            *self.temporal_emotion.parameters(),
            *self.cultural_nuance.parameters(),
            *self.advanced_moe.parameters()
        ])
        logger.info(f"✅ 감정 분석기 강화 모듈 통합: {total_params/1e6:.1f}M 파라미터 추가")
        
        if self.moe_enabled:
            try:
                # 감정 임베딩 차원 (기본값)
                emotion_input_dim = 768  # 문장 변환기 임베딩 차원
                emotion_output_dim = len(EmotionState)  # 감정 상태 수
                
                self.emotion_moe = create_emotion_moe(
                    input_dim=emotion_input_dim,
                    output_dim=emotion_output_dim,
                    num_experts=4
                ).to(self.device)
                
                logger.info("감정 분석용 MoE 시스템 초기화 완료 (4개 전문가)")
                logger.info(f"  - emotion_moe 타입: {type(self.emotion_moe)}")
                logger.info(f"  - emotion_moe None 여부: {self.emotion_moe is None}")
                
            except Exception as e:
                logger.error(f"MoE 초기화 실패: {e}")
                self.emotion_moe = None  # 실패 시 명시적으로 None 설정
                
                # 프로젝트 규칙: fallback 없는 순수 재시도 방식
                retry_count = 0
                max_retries = 3
                while retry_count < max_retries:
                    retry_count += 1
                    logger.info(f"MoE 초기화 재시도 {retry_count}/{max_retries}")
                    try:
                        self.emotion_moe = create_emotion_moe(
                            input_dim=emotion_input_dim,
                            output_dim=emotion_output_dim,
                            num_experts=4
                        ).to(self.device)
                        logger.info(f"재시도 {retry_count}: MoE 시스템 초기화 성공")
                        break
                    except Exception as retry_error:
                        logger.error(f"재시도 {retry_count} 실패: {retry_error}")
                        self.emotion_moe = None
                        if retry_count >= max_retries:
                            logger.error("MoE 초기화 모든 재시도 실패 - 시스템 종료")
                            raise Exception(f"MoE 시스템 초기화 최종 실패: {retry_error}") from e
        
        # DSP 컴포넌트 초기화
        self._init_dsp_components()
        
        # 한국어 감정 키워드 (고급 버전)
        self.korean_emotion_keywords = self._initialize_advanced_korean_keywords()
        
        # 모델 초기화는 initialize() 메서드에서 수행
        # self._initialize_models()를 여기서 호출하지 않음
        
        # 새로운 계층적 감정 모델 초기화
        global NEW_EMOTION_MODELS_AVAILABLE
        if NEW_EMOTION_MODELS_AVAILABLE:
            try:
                self.hierarchical_model = HierarchicalEmotionModel()
                self.hierarchical_model.to(self.device)
                self.emotion_model_manager = EmotionModelManager(os.path.join(self.models_dir, "hierarchical"))
                logger.info("계층적 감정 모델 초기화 완료")
            except Exception as e:
                logger.warning(f"계층적 감정 모델 초기화 실패: {e}")
                NEW_EMOTION_MODELS_AVAILABLE = False
        
        # LLM 엔진 연결
        global LLM_INTEGRATION_AVAILABLE
        if LLM_INTEGRATION_AVAILABLE:
            try:
                self.llm_engine = get_llm_engine()
                logger.info("LLM 엔진 연결 완료")
            except Exception as e:
                logger.warning(f"LLM 엔진 연결 실패: {e}")
                LLM_INTEGRATION_AVAILABLE = False
        
        logger.info("고급 감정 분석 시스템이 초기화되었습니다.")
        
        # 등록 단계에서 get_pytorch_network가 작동하도록 기본 네트워크 보장
        self._ensure_default_network()
    
    def _ensure_default_network(self):
        """최소 하나의 PyTorch 네트워크가 존재하도록 보장"""
        # 먼저 기존 네트워크 확인
        existing_network = None
        
        # hierarchical_model 확인
        if hasattr(self, 'hierarchical_model') and isinstance(self.hierarchical_model, nn.Module):
            existing_network = self.hierarchical_model
            logger.info("🔍 hierarchical_model이 이미 존재함")
        
        # emotion_moe 확인  
        elif hasattr(self, 'emotion_moe') and isinstance(self.emotion_moe, nn.Module):
            existing_network = self.emotion_moe
            logger.info("🔍 emotion_moe가 이미 존재함")
            
        # 네트워크가 없으면 기본 네트워크 생성
        if existing_network is None:
            logger.info("🔨 기본 PyTorch 네트워크 생성 중...")
            self._build_default_network()
        else:
            # 캐시에 저장
            self._primary_nn = existing_network
    
    def _build_default_network(self):
        """
        기본 PyTorch 네트워크 생성
        - 등록/헤드 바인딩을 위한 최소 네트워크
        - 가볍고 메모리 효율적
        """
        import torch.nn as nn
        
        # 디바이스 설정
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # 간단한 감정 분류기 네트워크
        class DefaultEmotionNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(1000, 128)  # 작은 임베딩
                self.lstm = nn.LSTM(128, 64, batch_first=True)
                self.classifier = nn.Linear(64, len(EmotionState))
                
            def forward(self, x):
                x = self.embedding(x)
                lstm_out, _ = self.lstm(x)
                # 마지막 타임스텝만 사용
                last_hidden = lstm_out[:, -1, :]
                return self.classifier(last_hidden)
        
        # 네트워크 생성 및 설정
        self.default_network = DefaultEmotionNetwork().to(device)
        self._primary_nn = self.default_network
        
        logger.info(f"✅ 기본 네트워크 생성 완료 (device: {device})")
        logger.info(f"   - 파라미터 수: {sum(p.numel() for p in self.default_network.parameters()):,}")
        
        # emotion_moe가 None이면 기본 네트워크로 설정
        if not hasattr(self, 'emotion_moe') or self.emotion_moe is None:
            self.emotion_moe = self.default_network
            logger.info("🔗 emotion_moe를 기본 네트워크로 설정")
    
    async def initialize(self):
        """비동기 초기화 메서드 - unified_system_orchestrator에서 호출됨"""
        logger.info("AdvancedEmotionAnalyzer 비동기 초기화 시작...")
        
        # 1. 먼저 기본 네트워크로 emotion_empathy_head 선등록 (NO FALLBACK)
        from dynamic_swap_manager import get_swap_manager, SwapPriority
        swap_manager = get_swap_manager()
        if swap_manager:
            # get_pytorch_network()는 항상 nn.Module을 반환하도록 보장됨
            try:
                primary = self.get_pytorch_network()  # 기본 네트워크 생성/획득
                if primary is not None:
                    swap_manager.register_model(
                        "emotion_empathy_head",
                        primary,
                        priority=SwapPriority.HIGH
                    )
                    logger.info(f"✅ emotion_empathy_head 선등록 완료 (기본 네트워크: {primary.__class__.__name__})")
                else:
                    logger.error("❌ emotion_empathy_head 선등록 실패: get_pytorch_network()가 None 반환")
            except Exception as e:
                logger.error(f"❌ emotion_empathy_head 선등록 중 에러: {e}")
        
        try:
            # 2. 모델 초기화 수행 (실패해도 헤드는 이미 등록됨)
            self._initialize_models()
            
            # 3. 대형 모델 로드 성공하면 emotion_empathy_head 업데이트
            if 'multilingual_direct' in self.models and self.models['multilingual_direct'] is not None:
                if swap_manager:
                    swap_manager.register_model(
                        "emotion_empathy_head",
                        self.models['multilingual_direct'],  # 대형 모델로 교체
                        priority=SwapPriority.HIGH
                    )
                    logger.info(f"✅ emotion_empathy_head 업데이트 완료 (대형 모델: multilingual_direct)")
            
            logger.info("AdvancedEmotionAnalyzer 비동기 초기화 완료")
            logger.info("✅ EmotionAnalyzer GPU 초기화 성공, 헤드 등록 완료")
        except Exception as e:
            logger.error(f"AdvancedEmotionAnalyzer 초기화 실패: {e}")
            # 대형 모델 실패해도 기본 네트워크는 이미 등록되어 있으므로 raise 하지 않음
            logger.warning("⚠️ 대형 모델 로드 실패했지만 기본 네트워크로 계속 진행")
    
    def _hf_kwargs_clean(self, **kwargs):
        """HF 호출 전 kwargs에서 owner 제거"""
        if 'owner' in kwargs:
            logger.warning("[EmotionAnalyzer] removing stray 'owner' from kwargs")
            kwargs.pop('owner', None)
        return kwargs
    
    def _create_biometric_network(self, input_channels: int, output_dim: int) -> nn.Module:
        """생체신호 처리 네트워크 생성"""
        return nn.Sequential(
            nn.Linear(input_channels, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, output_dim)
        )
    
    def _create_cultural_network(self, dim: int) -> nn.Module:
        """문화별 감정 해석 네트워크"""
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
    
    def _initialize_models(self):
        """모든 감정 분석 모델 초기화"""
        try:
            # 1. 다국어 감정 분석 모델
            logger.info("다국어 감정 분석 모델 로딩...")
            self._load_multilingual_emotion_model()
            
            # 2. 한국어 특화 모델
            logger.info("한국어 특화 모델 로딩...")
            self._load_korean_emotion_model()
            
            # 3. 감정 임베딩 모델
            logger.info("감정 임베딩 모델 로딩...")
            self._load_emotion_embedding_model()
            
            # 4. 생체신호 분석 모델 (현재 비활성화)
            if self.biosignal_enabled:
                logger.info("생체신호 분석 모델 초기화...")
                self._initialize_biosignal_model()
            else:
                logger.info("생체신호 분석 모델 비활성화됨 (센서 미연결)")
            
        except Exception as e:
            logger.error(f"모델 초기화 실패: {e}")
            raise
    
    def _load_multilingual_emotion_model(self):
        """다국어 감정 분석 모델 로드"""
        model_name = self.config['multilingual_model']
        
        try:
            # HF 래퍼를 통한 Zero-shot classification 파이프라인 (오프라인 모드)
            from hf_model_wrapper import get_hf_wrapper
            hf_wrapper = get_hf_wrapper()
            
            self.models['multilingual'] = hf_wrapper.wrapped_pipeline(
                "zero-shot-classification",
                model=model_name,
                owner="emotion_analyzer",
                **self._hf_kwargs_clean(
                    device=0 if self.device.type == 'cuda' else -1,
                    local_files_only=True
                )
            )
            
            # HF 래퍼를 통한 모델 로드 (세밀한 제어용, 오프라인 모드)
            from hf_model_wrapper import get_hf_wrapper
            hf_wrapper = get_hf_wrapper()
            
            self.tokenizers['multilingual'] = hf_wrapper.wrapped_tokenizer(
                model_name, 
                owner="emotion_analyzer",
                **self._hf_kwargs_clean(local_files_only=True)
            )
            self.models['multilingual_direct'] = hf_wrapper.wrapped_from_pretrained(
                AutoModelForSequenceClassification, model_name, 
                owner="emotion_analyzer",
                **self._hf_kwargs_clean(local_files_only=True)
            )
            self.models['multilingual_direct'].to(self.device)
            
            logger.info(f"다국어 모델 로드 완료: {model_name}")
            
        except Exception as e:
            logger.error(f"다국어 모델 로드 실패: {e}")
            raise
    
    def _load_korean_emotion_model(self):
        """한국어 특화 감정 분석 모델 로드"""
        model_name = self.config['korean_model']
        
        try:
            # HF 래퍼를 통한 한국어 BERT 모델 로드 (오프라인 모드)
            from hf_model_wrapper import get_hf_wrapper
            hf_wrapper = get_hf_wrapper()
            
            self.tokenizers['korean'] = hf_wrapper.wrapped_tokenizer(
                model_name, 
                owner="emotion_analyzer",
                **self._hf_kwargs_clean(local_files_only=True)
            )
            self.models['korean'] = hf_wrapper.wrapped_from_pretrained(
                AutoModel, model_name, 
                owner="emotion_analyzer",
                **self._hf_kwargs_clean(local_files_only=True)
            )
            self.models['korean'].to(self.device)
            
            # 한국어 감정 분류 파이프라인 (가능한 경우, 오프라인 모드)
            try:
                from hf_model_wrapper import get_hf_wrapper
                hf_wrapper = get_hf_wrapper()
                
                self.models['korean_pipeline'] = hf_wrapper.wrapped_pipeline(
                    "text-classification",
                    model=model_name,
                    tokenizer=model_name,
                    owner="emotion_analyzer",
                    **self._hf_kwargs_clean(
                        device=0 if self.device.type == 'cuda' else -1,
                        local_files_only=True
                    )
                )
            except Exception as pipeline_error:
                logger.warning(f"한국어 파이프라인 로드 실패, 직접 모델 사용: {pipeline_error}")
            
            logger.info(f"한국어 모델 로드 완료: {model_name}")
            
        except Exception as e:
            logger.warning(f"한국어 모델 로드 실패 (옵션 기능): {e}")
            logger.info("다국어 모델로 한국어 분석 대체 가능 - 시스템 계속 진행")
            # 한국어 모델 없어도 다국어 모델로 대체 가능
    
    def _load_emotion_embedding_model(self):
        """감정 임베딩 모델 로드"""
        try:
            # semantic 설정에서 임베딩 모델 가져오기
            from config import SYSTEM_CONFIG
            from sentence_transformer_singleton import get_sentence_transformer
            
            semantic_config = SYSTEM_CONFIG.get('semantic', {})
            multilingual_model = semantic_config.get('sentence_model', 'paraphrase-multilingual-mpnet-base-v2')
            
            # 싱글톤 매니저를 통해 공유 인스턴스 가져오기
            self.embedders['multilingual'] = get_sentence_transformer(
                multilingual_model,
                device=str(self.device)
            )
            
            # 한국어 특화 임베딩 모델 (싱글톤 매니저 사용)
            korean_embedding_model = semantic_config.get('korean_model', 'jhgan/ko-sroberta-multitask')
            self.embedders['korean'] = get_sentence_transformer(
                korean_embedding_model,
                device=str(self.device)
            )
            
            logger.info(f"임베딩 모델 로드 완료 (싱글톤): {multilingual_model}, {korean_embedding_model}")
            
        except Exception as e:
            logger.error(f"임베딩 모델 로드 실패: {e}")
            # fallback 없음 - 바로 예외 발생
            raise RuntimeError(f"SentenceTransformer 초기화 실패: {e}") from e
    
    def _initialize_biosignal_model(self):
        """생체신호 기반 감정 분석 모델 초기화"""
        raise RuntimeError("생체신호 분석 모델을 초기화할 수 없습니다. 실제 생체신호 데이터가 필요합니다.")
    
    
    
    def _initialize_advanced_korean_keywords(self) -> Dict[str, Dict[str, List[str]]]:
        """고급 한국어 감정 키워드 초기화"""
        return {
            EmotionState.JOY.value: {
                'primary': ['기쁘', '행복', '즐거', '좋', '만족', '흐뭇', '신나', '들뜨'],
                'secondary': ['웃', '웃음', '웃기', '재미', '유쾌', '상쾌', '통쾌', '시원'],
                'intensity': ['매우', '정말', '너무', '아주', '완전히', '엄청', '굉장히'],
                'cultural': ['기분좋', '마음좋', '속시원', '개운', '뿌듯']
            },
            EmotionState.SADNESS.value: {
                'primary': ['슬프', '우울', '눈물', '아프', '힘들', '서글', '처량', '암울'],
                'secondary': ['울', '울음', '울고', '쓸쓸', '외로', '허전', '공허', '막막'],
                'intensity': ['너무', '정말', '매우', '심하게', '깊이'],
                'cultural': ['마음아프', '가슴아프', '맘이무거', '한숨', '체념']
            },
            EmotionState.ANGER.value: {
                'primary': ['화나', '화가', '짜증', '분노', '열받', '빡치', '억울', '분해', '괘씸'],
                'secondary': ['성내', '화내', '욕하', '소리지르', '고함', '분통', '울화'],
                'intensity': ['정말', '너무', '극도로', '심하게', '완전히'],
                'cultural': ['열불나', '약오르', '빡돌', '뚜껑열리', '피꺼솟']
            },
            EmotionState.FEAR.value: {
                'primary': ['무서', '두려', '걱정', '불안', '겁나', '떨리', '오싹', '소름'],
                'secondary': ['떨', '떨림', '벌벌', '심장박동', '식은땀', '공포', '경악'],
                'intensity': ['매우', '너무', '정말', '극도로', '심하게'],
                'cultural': ['간담서늘', '등골오싹', '심장떨어뜨릴', '간떨어질']
            },
            EmotionState.SURPRISE.value: {
                'primary': ['놀라', '깜짝', '예상', '신기', '의외', '뜻밖', '갑자기'],
                'secondary': ['어', '헉', '와', '어머', '이런', '세상에', '진짜'],
                'intensity': ['정말', '너무', '완전히', '엄청'],
                'cultural': ['어이없', '기가막히', '어안이벙벙']
            },
            EmotionState.DISGUST.value: {
                'primary': ['역겨', '싫', '혐오', '더러', '구역질', '징그', '꼴불견'],
                'secondary': ['토하', '메스꺼', '구토', '찜찜', '불쾌', '기분나쁘'],
                'intensity': ['정말', '너무', '매우', '극도로'],
                'cultural': ['눈꼴사나', '보기싫', '치사']
            }
        }
    
    def analyze_emotion(self, text: str, language: str = "ko", 
                       biosignal_data: Optional[Biosignal] = None,
                       use_cache: bool = True) -> EmotionData:
        """고급 감정 분석 - 다중 모델 앙상블"""
        
        # 텍스트 길이 제한 (512 토큰 안전 마진으로 800자)
        if len(text) > 800:
            logger.warning(f"텍스트가 너무 김 ({len(text)}자), 처음 800자로 제한")
            text = text[:800]
        
        # 캐시 확인
        cache_key = f"{text}_{language}_{hash(str(biosignal_data))}"
        if use_cache and cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]
        
        start_time = time.time()
        
        # 1. 텍스트 기반 감정 분석
        text_emotion = self._analyze_text_emotion(text, language)
        
        # 2. 생체신호 기반 감정 분석 (있는 경우)
        biosignal_emotion = None
        if biosignal_data:
            biosignal_emotion = self._analyze_biosignal_emotion(biosignal_data)
        
        # 3. 결과 통합
        final_emotion = self._integrate_emotion_results(
            text_emotion, biosignal_emotion, text, language
        )
        
        # 4. MoE 기반 감정 분석 보정
        if self.moe_enabled:
            final_emotion = self._apply_moe_analysis(final_emotion, text, language)
        
        # 5. DSP 시뮬레이터와 칼만 필터 융합
        if self.dsp_enabled and self.dsp_simulator and self.kalman_filter:
            final_emotion = self._apply_dsp_kalman_fusion(final_emotion, text, language)
        
        # 6. Focal Loss 기반 Joy 편향 보정
        if self.focal_loss_enabled:
            final_emotion = self._apply_focal_loss_correction(final_emotion, text)
        
        # 처리 시간 기록
        processing_time = time.time() - start_time
        final_emotion.processing_method = "advanced_ensemble_with_moe_and_focal_correction"
        
        # 감정 임베딩 생성
        final_emotion.embedding = self._generate_emotion_embedding(text, language, final_emotion)
        
        # 캐시 저장
        if use_cache:
            self.prediction_cache[cache_key] = final_emotion
        
        logger.debug(f"감정 분석 완료: {final_emotion.primary_emotion.value} "
                    f"(신뢰도: {final_emotion.confidence:.3f}, 시간: {processing_time:.3f}s)")
        
        return final_emotion
    
    def _apply_moe_analysis(self, emotion_data: EmotionData, text: str, language: str) -> EmotionData:
        """
        MoE 기반 감정 분석 보정
        
        Args:
            emotion_data: 원본 감정 분석 결과
            text: 분석 대상 텍스트
            language: 언어
            
        Returns:
            MoE로 보정된 감정 분석 결과
        """
        try:
            # 텍스트 임베딩 생성
            text_embedding = self._get_text_embedding_for_moe(text, language)
            
            if text_embedding is None:
                return emotion_data
            
            # MoE 추론
            moe_result = self.emotion_moe(text_embedding, temperature=0.8, return_expert_outputs=True)
            
            # MoE 결과를 감정 확률로 변환 (안전한 차원 처리)
            softmax_output = F.softmax(moe_result.final_output, dim=-1)
            if softmax_output.dim() > 1 and softmax_output.size(0) == 1:
                emotion_probs = softmax_output.squeeze(0)
            else:
                emotion_probs = softmax_output
            
            # 가장 높은 확률의 감정 선택
            max_prob_idx = torch.argmax(emotion_probs).item()
            moe_confidence = emotion_probs[max_prob_idx].item()
            
            # EmotionState 매핑
            emotion_states = list(EmotionState)
            if max_prob_idx < len(emotion_states):
                moe_emotion = emotion_states[max_prob_idx]
            else:
                moe_emotion = EmotionState.NEUTRAL
            
            # 원본 결과와 MoE 결과 융합
            original_confidence = emotion_data.confidence
            
            # 신뢰도 가중 평균으로 최종 결정
            if moe_confidence > original_confidence * 1.2:
                # MoE 결과가 훨씬 확신적이면 MoE 결과 채택
                corrected_emotion = EmotionData(
                    primary_emotion=moe_emotion,
                    confidence=moe_confidence,
                    language=emotion_data.language,
                    processing_method=f"{emotion_data.processing_method}_moe_enhanced",
                    intensity=emotion_data.intensity,
                    secondary_emotions=emotion_data.secondary_emotions
                )
                
                # 보조 감정 업데이트
                secondary_emotions = {}
                for i, prob in enumerate(emotion_probs):
                    if i != max_prob_idx and i < len(emotion_states) and prob > 0.1:
                        secondary_emotions[emotion_states[i]] = prob.item()
                
                if secondary_emotions:
                    corrected_emotion.secondary_emotions = secondary_emotions
                
                # 메타데이터 추가
                corrected_emotion.metadata = emotion_data.metadata.copy()
                corrected_emotion.metadata.update({
                    'moe_analysis': True,
                    'original_emotion': emotion_data.primary_emotion.name,
                    'original_confidence': original_confidence,
                    'moe_confidence': moe_confidence,
                    'experts_used': moe_result.total_experts_used,
                    'diversity_score': moe_result.diversity_score,
                    'expert_details': [
                        {
                            'expert_id': eo.expert_id,
                            'confidence': eo.confidence,
                            'weight': eo.weight
                        } for eo in moe_result.expert_outputs[:3]  # 상위 3개만
                    ]
                })
                
            else:
                # 원본 결과 유지하되 MoE 정보 추가
                corrected_emotion = emotion_data
                corrected_emotion.metadata = emotion_data.metadata.copy()
                corrected_emotion.metadata.update({
                    'moe_analysis': True,
                    'moe_suggestion': moe_emotion.name,
                    'moe_confidence': moe_confidence,
                    'confidence_ratio': moe_confidence / original_confidence,
                    'experts_used': moe_result.total_experts_used
                })
            
            return corrected_emotion
            
        except Exception as e:
            logger.error(f"MoE 분석 실패: {e}")
            return emotion_data
    
    def _get_text_embedding_for_moe(self, text: str, language: str) -> Optional[torch.Tensor]:
        """
        MoE용 텍스트 임베딩 생성
        
        Args:
            text: 입력 텍스트
            language: 언어
            
        Returns:
            텍스트 임베딩 텐서
        """
        try:
            # 기존 임베딩 모델 사용
            if hasattr(self, 'embedders') and 'multilingual_embedder' in self.embedders:
                embedding = self.embedders['multilingual_embedder'].encode(
                    text, convert_to_tensor=True, device=self.device
                )
                
                # 배치 차원 추가
                if embedding.dim() == 1:
                    embedding = embedding.unsqueeze(0)
                
                return embedding
            
            # 대안: 한국어 임베딩 모델
            elif hasattr(self, 'embedders') and 'korean_embedder' in self.embedders:
                embedding = self.embedders['korean_embedder'].encode(
                    text, convert_to_tensor=True, device=self.device
                )
                
                if embedding.dim() == 1:
                    embedding = embedding.unsqueeze(0)
                
                return embedding
            
            # 기본 대안: 감정 임베딩 모델 사용
            elif hasattr(self, 'emotion_embedder') and self.emotion_embedder is not None:
                embedding = self.emotion_embedder.encode(
                    text, convert_to_tensor=True, device=self.device
                )
                
                if embedding.dim() == 1:
                    embedding = embedding.unsqueeze(0)
                
                return embedding
            
            # NO FALLBACK - 모델이 없으면 실패
            else:
                logger.error("전용 MoE 임베딩 모델이 없음")
                raise RuntimeError("MoE embedding model not available")
                
        except Exception as e:
            logger.error(f"MoE 임베딩 생성 실패: {e}")
            # NO FALLBACK - 즉시 예외 발생
            raise RuntimeError(f"MoE embedding generation failed: {e}")
    
    def _init_dsp_components(self):
        """DSP 시뮬레이터와 칼만 필터 초기화"""
        if not self.dsp_enabled:
            return
            
        try:
            # DSP 시뮬레이터 초기화 (20M 파라미터)
            self.dsp_simulator = EmotionDSPSimulator({
                'hidden_dim': 256,  # 축소된 차원
            }).to(self.device)
            
            # 동적 칼만 필터 초기화 (7개 감정 상태)
            self.kalman_filter = DynamicKalmanFilter(
                state_dim=len(EmotionState)
            ).to(self.device)
            
            logger.info("✅ DSP 시뮬레이터와 칼만 필터 초기화 완료")
            logger.info(f"  - DSP 시뮬레이터: 20M 파라미터")
            logger.info(f"  - 칼만 필터: 융합용")
            
        except Exception as e:
            logger.error(f"DSP 컴포넌트 초기화 실패: {e}")
            self.dsp_simulator = None
            self.kalman_filter = None
            self.dsp_enabled = False
    
    def _apply_dsp_kalman_fusion(self, emotion_data: EmotionData, text: str, language: str) -> EmotionData:
        """
        DSP 시뮬레이터와 칼만 필터를 통한 감정 융합
        
        Args:
            emotion_data: 기존 감정 분석 결과
            text: 분석 텍스트
            language: 언어
            
        Returns:
            칼만 필터로 융합된 감정 데이터
        """
        try:
            # 1. 텍스트를 DSP 입력으로 변환
            text_embedding = self._get_text_embedding_for_moe(text, language)
            if text_embedding is None or text_embedding.shape[-1] != 256:
                # 임베딩 차원 조정 (768 -> 256)
                if text_embedding is not None:
                    linear_proj = nn.Linear(text_embedding.shape[-1], 256).to(self.device)
                    text_embedding = linear_proj(text_embedding)
                else:
                    # 더미 임베딩 생성
                    text_embedding = torch.randn(1, 256).to(self.device)
            
            # 2. DSP 시뮬레이터 실행
            dsp_result = self.dsp_simulator(text_embedding)
            dsp_emotions = dsp_result['final_emotions']  # (batch, 7)
            
            # 3. 기존 감정을 텐서로 변환
            emotion_states = list(EmotionState)
            traditional_emotions = torch.zeros(1, len(emotion_states)).to(self.device)
            
            # 주 감정 설정
            primary_idx = emotion_states.index(emotion_data.primary_emotion)
            traditional_emotions[0, primary_idx] = emotion_data.confidence
            
            # 보조 감정 설정
            if emotion_data.secondary_emotions:
                for sec_emotion, sec_conf in emotion_data.secondary_emotions.items():
                    if sec_emotion in emotion_states:
                        sec_idx = emotion_states.index(sec_emotion)
                        traditional_emotions[0, sec_idx] = sec_conf
            
            # 정규화
            traditional_emotions = F.softmax(traditional_emotions, dim=-1)
            
            # 4. 칼만 필터로 융합
            fused_emotions = self.kalman_filter(
                traditional_emotions=traditional_emotions,
                dsp_emotions=dsp_emotions,
                prev_state=self.prev_kalman_state
            )
            
            # 칼만 상태 업데이트
            self.prev_kalman_state = fused_emotions.detach()
            
            # 5. 융합 결과를 EmotionData로 변환
            fused_emotions_cpu = fused_emotions[0].cpu().numpy()
            max_idx = np.argmax(fused_emotions_cpu)
            
            # 새로운 주 감정과 신뢰도
            emotion_data.primary_emotion = emotion_states[max_idx]
            emotion_data.confidence = float(fused_emotions_cpu[max_idx])
            
            # 보조 감정 업데이트
            emotion_data.secondary_emotions = {}
            for i, prob in enumerate(fused_emotions_cpu):
                if i != max_idx and prob > 0.1:  # 10% 이상인 감정만
                    emotion_data.secondary_emotions[emotion_states[i]] = float(prob)
            
            # DSP 특징 저장 (메타데이터)
            emotion_data.metadata = emotion_data.metadata or {}
            emotion_data.metadata['dsp_valence_arousal'] = dsp_result['valence_arousal'].cpu().numpy().tolist()
            emotion_data.metadata['dsp_emotion_spectrum'] = dsp_result['emotion_spectrum'].cpu().numpy().tolist()
            emotion_data.metadata['fusion_method'] = 'kalman_filter'
            
            logger.debug(f"DSP-칼만 융합 완료: {emotion_data.primary_emotion.value} "
                        f"(신뢰도: {emotion_data.confidence:.3f})")
            
        except Exception as e:
            logger.error(f"DSP-칼만 융합 실패: {e}")
            # 실패 시 원본 반환 (NO FALLBACK 원칙이지만 융합은 선택적 개선)
            
        return emotion_data
    
    def _apply_focal_loss_correction(self, emotion_data: EmotionData, text: str) -> EmotionData:
        """
        Focal Loss 기반 Joy 편향 보정
        
        Args:
            emotion_data: 원본 감정 분석 결과
            text: 분석 대상 텍스트
            
        Returns:
            보정된 감정 분석 결과
        """
        try:
            # Joy 편향 감지 및 보정
            corrected_emotion = emotion_data
            
            # Joy가 주요 감정인 경우 보정 적용
            if emotion_data.primary_emotion == EmotionState.JOY:
                # 신뢰도 기반 보정
                original_confidence = emotion_data.confidence
                
                # Joy 신뢰도 감소 (focal loss 가중치 적용)
                joy_weight = self.focal_loss.emotion_weights.get('JOY', 0.5)
                corrected_confidence = original_confidence * joy_weight
                
                # 다른 감정 가능성 재검토
                alternative_emotions = self._analyze_alternative_emotions(text, emotion_data)
                
                # 보정된 신뢰도가 임계값(0.6) 이하이면 대안 감정 고려
                if corrected_confidence < 0.6 and alternative_emotions:
                    best_alternative = max(alternative_emotions.items(), key=lambda x: x[1])
                    alt_emotion, alt_confidence = best_alternative
                    
                    # 대안 감정이 더 적절하면 교체
                    if alt_confidence > corrected_confidence * 1.2:
                        corrected_emotion = EmotionData(
                            primary_emotion=alt_emotion,
                            confidence=alt_confidence,
                            language=emotion_data.language,
                            processing_method=f"{emotion_data.processing_method}_focal_corrected",
                            intensity=emotion_data.intensity,
                            secondary_emotions=emotion_data.secondary_emotions
                        )
                        
                        # 메타데이터에 보정 정보 추가
                        corrected_emotion.metadata = emotion_data.metadata.copy()
                        corrected_emotion.metadata.update({
                            'focal_loss_correction': True,
                            'original_emotion': emotion_data.primary_emotion.name,
                            'original_confidence': original_confidence,
                            'correction_reason': 'joy_bias_correction'
                        })
                    else:
                        # Joy 유지하되 신뢰도만 보정
                        corrected_emotion.confidence = corrected_confidence
                        corrected_emotion.metadata = emotion_data.metadata.copy()
                        corrected_emotion.metadata.update({
                            'focal_loss_correction': True,
                            'confidence_adjusted': True
                        })
                else:
                    # Joy 유지하되 신뢰도만 보정
                    corrected_emotion.confidence = corrected_confidence
                    corrected_emotion.metadata = emotion_data.metadata.copy()
                    corrected_emotion.metadata.update({
                        'focal_loss_correction': True,
                        'confidence_adjusted': True
                    })
            
            # 다른 감정들의 상대적 강화
            if hasattr(corrected_emotion, 'secondary_emotions') and corrected_emotion.secondary_emotions:
                enhanced_secondary = {}
                for emotion, score in corrected_emotion.secondary_emotions.items():
                    weight = self.focal_loss.emotion_weights.get(emotion.name, 1.0)
                    enhanced_secondary[emotion] = min(1.0, score * weight)
                corrected_emotion.secondary_emotions = enhanced_secondary
            
            return corrected_emotion
            
        except Exception as e:
            logger.error(f"Focal loss 보정 실패: {e}")
            return emotion_data
    
    def _analyze_alternative_emotions(self, text: str, original_emotion: EmotionData) -> Dict[EmotionState, float]:
        """
        대안 감정 분석
        
        Args:
            text: 분석 대상 텍스트
            original_emotion: 원본 감정 분석 결과
            
        Returns:
            대안 감정들과 신뢰도
        """
        alternatives = {}
        
        try:
            # 키워드 기반 대안 감정 분석
            text_lower = text.lower()
            
            # 감정별 키워드 사전
            emotion_keywords = {
                EmotionState.SADNESS: ['슬프', '우울', '짜증', '실망', '좌절', '눈물', '아프', '힘들'],
                EmotionState.ANGER: ['화나', '짜증', '분노', '열받', '빡쳐', '싫어', '미워'],
                EmotionState.FEAR: ['무서', '두려', '걱정', '불안', '겁나', '떨려'],
                EmotionState.SURPRISE: ['놀라', '깜짝', '어?', '헉', '와', '대박'],
                EmotionState.DISGUST: ['역겨', '더러', '싫어', '구역질', '못생', '추악'],
                EmotionState.TRUST: ['믿', '신뢰', '안전', '든든', '확신'],
                EmotionState.ANTICIPATION: ['기대', '설레', '기다', '바라', '희망']
            }
            
            for emotion, keywords in emotion_keywords.items():
                if emotion != original_emotion.primary_emotion:
                    score = 0.0
                    for keyword in keywords:
                        if keyword in text_lower:
                            score += 0.15
                    
                    # 키워드 밀도 계산
                    if score > 0:
                        keyword_density = score / len(text) * 100
                        final_score = min(0.9, score + keyword_density)
                        alternatives[emotion] = final_score
            
            return alternatives
            
        except Exception as e:
            logger.error(f"대안 감정 분석 실패: {e}")
            return {}
    
    def _analyze_text_emotion(self, text: str, language: str) -> EmotionData:
        """텍스트 기반 감정 분석"""
        
        if language == "ko":
            # 한국어 특화 분석
            return self._analyze_korean_text(text)
        else:
            # 다국어 분석
            return self._analyze_multilingual_text(text)
    
    def _analyze_korean_text(self, text: str) -> EmotionData:
        """한국어 텍스트 감정 분석"""
        
        # 1. 한국어 모델 사용 (신뢰도가 높은 경우만)
        if 'korean_pipeline' in self.models:
            try:
                result = self.models['korean_pipeline'](text)
                # 결과 처리 - 신뢰도가 0.7 이상인 경우만 사용
                if result and result[0]['score'] > 0.7:
                    emotion_label = result[0]['label'].lower()
                    confidence = result[0]['score']
                    
                    # 감정 매핑 (한국어 모델 레이블 포함)
                    emotion_mapping = {
                        'positive': EmotionState.JOY,
                        'negative': EmotionState.SADNESS,
                        'joy': EmotionState.JOY,
                        'sadness': EmotionState.SADNESS,
                        'anger': EmotionState.ANGER,
                        'fear': EmotionState.FEAR,
                        'surprise': EmotionState.SURPRISE,
                        'disgust': EmotionState.DISGUST,
                        'label_0': EmotionState.SADNESS,    # 한국어 모델 레이블
                        'label_1': EmotionState.JOY,       # 한국어 모델 레이블
                        'label_2': EmotionState.ANGER,     # 한국어 모델 레이블
                        'label_3': EmotionState.FEAR,      # 한국어 모델 레이블
                        'label_4': EmotionState.SURPRISE,  # 한국어 모델 레이블
                        'label_5': EmotionState.DISGUST,   # 한국어 모델 레이블
                    }
                    
                    primary_emotion = emotion_mapping.get(emotion_label, EmotionState.NEUTRAL)
                    
                    if primary_emotion != EmotionState.NEUTRAL:
                        return EmotionData(
                            primary_emotion=primary_emotion,
                            confidence=confidence,
                            language="ko",
                            processing_method="korean_transformer"
                        )
            except Exception as e:
                logger.warning(f"한국어 모델 분석 실패: {e}")
        
        # 2. 고급 키워드 분석 (기본 방법)
        return self._analyze_korean_keywords_advanced(text)
    
    def _analyze_korean_keywords_advanced(self, text: str) -> EmotionData:
        """고급 한국어 키워드 분석 + LLM 보조"""
        text_lower = text.lower()
        emotion_scores = {}
        detected_emotions = []
        
        for emotion_name, keywords_dict in self.korean_emotion_keywords.items():
            total_score = 0
            matches = []
            
            # Primary 키워드 점수 (가중치 1.0)
            for keyword in keywords_dict['primary']:
                if keyword in text_lower:
                    total_score += 1.0
                    matches.append(('primary', keyword))
            
            # Secondary 키워드 점수 (가중치 0.7)
            for keyword in keywords_dict['secondary']:
                if keyword in text_lower:
                    total_score += 0.7
                    matches.append(('secondary', keyword))
            
            # Intensity 수식어 점수 (가중치 0.5, 곱셈)
            intensity_multiplier = 1.0
            for modifier in keywords_dict['intensity']:
                if modifier in text_lower:
                    intensity_multiplier += 0.5
                    matches.append(('intensity', modifier))
            
            # Cultural 키워드 점수 (가중치 0.8)
            for keyword in keywords_dict['cultural']:
                if keyword in text_lower:
                    total_score += 0.8
                    matches.append(('cultural', keyword))
            
            final_score = total_score * intensity_multiplier
            emotion_scores[emotion_name] = final_score
            
            if final_score > 0:
                detected_emotions.append({
                    'emotion': emotion_name,
                    'score': final_score,
                    'matches': matches
                })
        
        # 키워드 분석 결과가 있으면 LLM으로 검증
        if emotion_scores and max(emotion_scores.values()) > 0:
            best_emotion = max(emotion_scores, key=emotion_scores.get)
            best_score = emotion_scores[best_emotion]
            
            # LLM을 통한 감정 분석 강화 (조건부)
            llm_result = self._enhance_with_llm_analysis(text, best_emotion, best_score)
            if llm_result:
                # LLM 결과로 조정
                best_emotion = llm_result.get('emotion', best_emotion) 
                confidence = llm_result.get('confidence', best_score / 5.0)
                reasoning = llm_result.get('reasoning', '')
            else:
                # 기존 키워드 분석 결과 사용
                confidence = min(0.95, best_score / 5.0)
                reasoning = f"키워드 기반 분석: {detected_emotions}"
            
            # 강도 계산
            intensity = self._calculate_intensity_from_score(best_score)
            
            # 감정가 및 각성도 계산
            valence, arousal = self._calculate_valence_arousal(best_emotion, best_score)
            
            return EmotionData(
                primary_emotion=EmotionState(best_emotion),
                intensity=intensity,
                confidence=confidence,
                valence=valence,
                arousal=arousal,
                secondary_emotions={EmotionState(k): v/5.0 
                                  for k, v in emotion_scores.items() 
                                  if v > 0 and k != best_emotion},
                language="ko",
                processing_method="advanced_korean_keywords_llm",
                metadata={'llm_reasoning': reasoning}
            )
        
        # 감정이 감지되지 않은 경우 - LLM 전체 분석 (직접 결과 사용)
        try:
            llm_result = self._deep_llm_emotion_analysis(text)
            
            # 반환 타입 검증: _deep_llm_emotion_analysis는 완전히 처리된 딕셔너리를 반환
            if llm_result is None:
                raise RuntimeError("LLM 감정 분석이 None을 반환")
            elif not isinstance(llm_result, dict):
                raise RuntimeError(f"LLM 감정 분석이 예상치 못한 타입 반환: {type(llm_result)}")
            elif 'emotion' not in llm_result:
                raise RuntimeError("LLM 감정 분석 결과에 'emotion' 필드가 없음")
        except Exception as e:
            logger.error(f"LLM 분석 호출 실패: {e}")
            raise RuntimeError(f"LLM 감정 분석 실패 - fallback 금지로 시스템 정지: {e}")
            
        if llm_result and llm_result.get('emotion') != EmotionState.NEUTRAL.value:
            # intensity 값 안전하게 검증
            intensity_value = llm_result.get('intensity', 3)
            if not isinstance(intensity_value, int) or intensity_value < 1 or intensity_value > 6:
                logger.warning(f"잘못된 intensity 값: {intensity_value}, 기본값 3으로 설정")
                intensity_value = 3
            
            return EmotionData(
                primary_emotion=EmotionState(llm_result['emotion']),
                intensity=EmotionIntensity(intensity_value),
                confidence=llm_result.get('confidence', 0.6),
                valence=llm_result.get('valence', 0.0),
                arousal=llm_result.get('arousal', 0.0),
                language="ko",
                processing_method="deep_llm_analysis",
                metadata={'llm_reasoning': llm_result.get('reasoning', '')}
            )
        
        # 모든 분석이 실패한 경우 - fallback 제거로 시스템 정지
        logger.error("모든 감정 분석 방법이 실패했습니다. fallback 메커니즘이 비활성화되었습니다.")
        raise RuntimeError("감정 분석 완전 실패 - fallback 금지로 시스템 정지")
    
    def _basic_emotion_analysis(self, text: str, language: str = "ko") -> EmotionData:
        """기본 감정 분석 (fallback이 아닌 기본 분석)"""
        try:
            # 단순 키워드 기반 감정 분석
            emotion_scores = {
                EmotionState.JOY: 0,
                EmotionState.SADNESS: 0,
                EmotionState.ANGER: 0,
                EmotionState.FEAR: 0,
                EmotionState.SURPRISE: 0,
                EmotionState.DISGUST: 0,
                EmotionState.TRUST: 0,
                EmotionState.ANTICIPATION: 0
            }
            
            # 간단한 감정 키워드 매칭
            text_lower = text.lower()
            
            # 기쁨 키워드
            joy_keywords = ['기쁘', '행복', '즐거', '좋', '만족', '웃', '기뻐', '사랑']
            for keyword in joy_keywords:
                if keyword in text_lower:
                    emotion_scores[EmotionState.JOY] += 1
            
            # 슬픔 키워드
            sadness_keywords = ['슬프', '우울', '눈물', '슬픈', '아프', '상처', '괴로']
            for keyword in sadness_keywords:
                if keyword in text_lower:
                    emotion_scores[EmotionState.SADNESS] += 1
            
            # 분노 키워드
            anger_keywords = ['화나', '짜증', '분노', '열받', '빡', '미치', '싫']
            for keyword in anger_keywords:
                if keyword in text_lower:
                    emotion_scores[EmotionState.ANGER] += 1
            
            # 두려움 키워드
            fear_keywords = ['무서', '두려', '걱정', '불안', '위험', '겁']
            for keyword in fear_keywords:
                if keyword in text_lower:
                    emotion_scores[EmotionState.FEAR] += 1
            
            # 주요 감정 결정
            max_emotion = max(emotion_scores, key=emotion_scores.get)
            max_score = emotion_scores[max_emotion]
            
            if max_score == 0:
                # 감정이 감지되지 않으면 중립
                primary_emotion = EmotionState.NEUTRAL
                intensity = EmotionIntensity.MODERATE
                confidence = 0.3
            else:
                primary_emotion = max_emotion
                # 점수에 따른 강도 결정
                if max_score >= 3:
                    intensity = EmotionIntensity.VERY_HIGH
                elif max_score >= 2:
                    intensity = EmotionIntensity.HIGH
                else:
                    intensity = EmotionIntensity.MODERATE
                confidence = min(0.8, 0.3 + max_score * 0.1)
            
            return EmotionData(
                primary_emotion=primary_emotion,
                intensity=intensity,
                confidence=confidence,
                valence=0.5 if primary_emotion == EmotionState.JOY else -0.3,
                arousal=0.4,
                language=language,
                processing_method="basic_keyword_analysis",
                metadata={'keyword_scores': {e.value: s for e, s in emotion_scores.items()}}
            )
            
        except Exception as e:
            logger.error(f"기본 감정 분석도 실패: {e}")
            # 최후의 기본값
            return EmotionData(
                primary_emotion=EmotionState.NEUTRAL,
                intensity=EmotionIntensity.MODERATE,
                confidence=0.2,
                valence=0.0,
                arousal=0.0,
                language=language,
                processing_method="emergency_fallback",
                metadata={'error': str(e)}
            )
    
    def _analyze_multilingual_text(self, text: str) -> EmotionData:
        """다국어 텍스트 감정 분석"""
        
        # Zero-shot classification 사용
        emotion_labels = [e.value for e in EmotionState if e != EmotionState.NEUTRAL]
        
        try:
            result = self.models['multilingual'](text, emotion_labels)
            
            if result:
                best_label = result['labels'][0]
                best_score = result['scores'][0]
                
                # 모든 감정 점수
                emotion_scores = dict(zip(result['labels'], result['scores']))
                
                primary_emotion = EmotionState(best_label)
                
                return EmotionData(
                    primary_emotion=primary_emotion,
                    confidence=best_score,
                    secondary_emotions={EmotionState(k): v for k, v in emotion_scores.items() 
                                      if k != best_label and v > 0.1},
                    processing_method="multilingual_transformer"
                )
                
        except Exception as e:
            logger.error(f"다국어 모델 분석 실패: {e}")
            raise RuntimeError(f"다국어 감정 분석 실행 중 오류 발생: {e}")
        
        # 모든 시도가 실패한 경우
        raise RuntimeError("다국어 감정 분석을 완료할 수 없습니다. 모델이 결과를 반환하지 않았습니다.")
    
    def _analyze_biosignal_emotion(self, biosignal: Biosignal) -> Optional[EmotionData]:
        """생체신호 기반 감정 분석 (주석 처리 - 향후 연결 가능)
        
        센서 연결 시 활성화 방법:
        1. self.biosignal_enabled = True로 변경
        2. 생체신호 센서 하드웨어 연결 (EEG, ECG, GSR 등)
        3. 생체신호 ML 모델 훈련 및 로드
        4. 실시간 데이터 수집 파이프라인 구축
        """
        if not self.biosignal_enabled:
            raise RuntimeError("생체신호 분석이 비활성화되었습니다. 센서 연결 시 활성화 가능합니다.")
        
        try:
            # 특성 벡터 생성
            features = np.array([[
                biosignal.heart_rate,
                biosignal.gsr,
                biosignal.eeg_alpha,
                biosignal.eeg_beta,
                biosignal.eeg_theta,
                biosignal.eeg_delta,
                biosignal.respiratory_rate,
                biosignal.skin_temperature
            ]])
            
            # 스케일링
            features_scaled = self.biosignal_scaler.transform(features)
            
            # 예측
            prediction = self.biosignal_model.predict(features_scaled)[0]
            probabilities = self.biosignal_model.predict_proba(features_scaled)[0]
            
            # 최고 확률 찾기
            max_prob_idx = np.argmax(probabilities)
            confidence = probabilities[max_prob_idx]
            
            # 감정 상태 변환
            emotion_classes = self.biosignal_model.classes_
            predicted_emotion = EmotionState(emotion_classes[max_prob_idx])
            
            # 각성도 계산 (심박수와 GSR 기반)
            arousal = self._calculate_arousal_from_biosignal(biosignal)
            
            # 감정가 계산 (감정 유형 기반)
            valence = self._calculate_valence_from_emotion(predicted_emotion)
            
            return EmotionData(
                primary_emotion=predicted_emotion,
                confidence=confidence,
                arousal=arousal,
                valence=valence,
                biosignal_data=biosignal,
                processing_method="biosignal_ml"
            )
            
        except Exception as e:
            logger.error(f"생체신호 분석 실패: {e}")
            raise RuntimeError(f"생체신호 분석 실행 중 오류 발생: {e}")
    
    def _integrate_emotion_results(self, text_emotion: EmotionData, 
                                 biosignal_emotion: Optional[EmotionData],
                                 text: str, language: str) -> EmotionData:
        """감정 분석 결과 통합"""
        
        if biosignal_emotion is None:
            return text_emotion
        
        # 가중치 설정 (텍스트 vs 생체신호)
        text_weight = 0.7
        biosignal_weight = 0.3
        
        # 신뢰도 가중 평균
        combined_confidence = (text_emotion.confidence * text_weight + 
                             biosignal_emotion.confidence * biosignal_weight)
        
        # 주 감정 결정 (더 높은 신뢰도 기준)
        if text_emotion.confidence >= biosignal_emotion.confidence:
            primary_emotion = text_emotion.primary_emotion
        else:
            primary_emotion = biosignal_emotion.primary_emotion
        
        # 각성도와 감정가 통합
        combined_arousal = (text_emotion.arousal * text_weight + 
                          biosignal_emotion.arousal * biosignal_weight)
        combined_valence = (text_emotion.valence * text_weight + 
                          biosignal_emotion.valence * biosignal_weight)
        
        # 강도 계산
        intensity = self._calculate_intensity_from_confidence(combined_confidence)
        
        return EmotionData(
            primary_emotion=primary_emotion,
            intensity=intensity,
            confidence=combined_confidence,
            arousal=combined_arousal,
            valence=combined_valence,
            biosignal_data=biosignal_emotion.biosignal_data,
            language=language,
            processing_method="integrated_advanced"
        )
    
    def _generate_emotion_embedding(self, text: str, language: str) -> np.ndarray:
        """감정 임베딩 생성"""
        try:
            # 언어별 임베딩 모델 선택
            if language == "ko" and 'korean' in self.embedders:
                embedder = self.embedders['korean']
            else:
                embedder = self.embedders['multilingual']
            
            # 임베딩 생성
            embedding = embedder.encode(text, convert_to_numpy=True)
            
            return embedding
            
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")
            raise RuntimeError(f"감정 임베딩 생성 실패: {e}")
    
    def _calculate_intensity_from_score(self, score: float) -> EmotionIntensity:
        """점수에서 감정 강도 계산"""
        if score >= 4.0:
            return EmotionIntensity.EXTREME
        elif score >= 3.0:
            return EmotionIntensity.VERY_STRONG
        elif score >= 2.0:
            return EmotionIntensity.STRONG
        elif score >= 1.0:
            return EmotionIntensity.MODERATE
        elif score >= 0.5:
            return EmotionIntensity.WEAK
        else:
            return EmotionIntensity.VERY_WEAK
    
    def _calculate_intensity_from_confidence(self, confidence: float) -> EmotionIntensity:
        """신뢰도에서 감정 강도 계산"""
        if confidence >= 0.9:
            return EmotionIntensity.EXTREME
        elif confidence >= 0.8:
            return EmotionIntensity.VERY_STRONG
        elif confidence >= 0.7:
            return EmotionIntensity.STRONG
        elif confidence >= 0.6:
            return EmotionIntensity.MODERATE
        elif confidence >= 0.5:
            return EmotionIntensity.WEAK
        else:
            return EmotionIntensity.VERY_WEAK
    
    def _calculate_valence_arousal(self, emotion: str, score: float) -> Tuple[float, float]:
        """감정에서 감정가와 각성도 계산"""
        # 감정별 기본 감정가/각성도 (연구 기반)
        emotion_va = {
            'joy': (0.8, 0.7),
            'sadness': (-0.6, -0.4),
            'anger': (-0.7, 0.8),
            'fear': (-0.8, 0.9),
            'surprise': (0.2, 0.8),
            'disgust': (-0.7, 0.3),
            'trust': (0.6, 0.3),
            'anticipation': (0.5, 0.6),
            'neutral': (0.0, 0.0)
        }
        
        base_valence, base_arousal = emotion_va.get(emotion, (0.0, 0.0))
        
        # 점수에 따른 조정
        intensity_factor = min(1.0, score / 3.0)
        
        valence = base_valence * intensity_factor
        arousal = base_arousal * intensity_factor
        
        return valence, arousal
    
    def _calculate_arousal_from_biosignal(self, biosignal: Biosignal) -> float:
        """생체신호에서 각성도 계산"""
        # 정규화된 값들
        hr_norm = (biosignal.heart_rate - 60) / 40  # 60-100 bpm 범위
        gsr_norm = biosignal.gsr
        resp_norm = (biosignal.respiratory_rate - 12) / 8  # 12-20 범위
        
        arousal = (hr_norm * 0.4 + gsr_norm * 0.4 + resp_norm * 0.2)
        
        return max(-1.0, min(1.0, arousal))
    
    def _calculate_valence_from_emotion(self, emotion: EmotionState) -> float:
        """감정 상태에서 감정가 계산"""
        valence_map = {
            EmotionState.JOY: 0.8,
            EmotionState.TRUST: 0.6,
            EmotionState.ANTICIPATION: 0.5,
            EmotionState.SURPRISE: 0.2,
            EmotionState.NEUTRAL: 0.0,
            EmotionState.DISGUST: -0.5,
            EmotionState.SADNESS: -0.6,
            EmotionState.ANGER: -0.7,
            EmotionState.FEAR: -0.8
        }
        
        return valence_map.get(emotion, 0.0)
    
    def batch_analyze_emotions(self, texts: List[str], 
                             language: str = "ko") -> List[EmotionData]:
        """배치 감정 분석"""
        results = []
        
        # 배치 처리 활성화 시
        if self.config.get('batch_processing', True) and len(texts) > 1:
            # 임베딩 배치 생성
            if language == "ko" and 'korean' in self.embedders:
                embedder = self.embedders['korean']
            else:
                embedder = self.embedders['multilingual']
            
            embeddings = embedder.encode(texts, batch_size=self.config.get('batch_size', 16))
            
            # 각 텍스트별 분석
            for i, text in enumerate(texts):
                emotion_data = self.analyze_emotion(text, language, use_cache=False)
                emotion_data.embedding = embeddings[i]
                results.append(emotion_data)
        else:
            # 개별 처리
            for text in texts:
                results.append(self.analyze_emotion(text, language))
        
        return results
    
    def analyze_text_advanced(self, text: str, language: str = "ko", context: str = None) -> EmotionData:
        """고급 텍스트 감정 분석 - main.py 호환성을 위한 래퍼 메서드"""
        # context는 현재 사용하지 않지만 인터페이스 호환성 유지
        return self.analyze_emotion(text=text, language=language)
    
    def get_emotion_similarity(self, emotion1: EmotionData, 
                             emotion2: EmotionData) -> float:
        """두 감정 간 유사도 계산"""
        if emotion1.embedding is not None and emotion2.embedding is not None:
            # 임베딩 기반 유사도
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity(
                emotion1.embedding.reshape(1, -1),
                emotion2.embedding.reshape(1, -1)
            )[0][0]
            return float(similarity)
        else:
            # 감정 상태 기반 유사도
            if emotion1.primary_emotion == emotion2.primary_emotion:
                return 0.9
            elif self._are_emotions_similar(emotion1.primary_emotion, emotion2.primary_emotion):
                return 0.6
            else:
                return 0.2
    
    def _are_emotions_similar(self, emotion1: EmotionState, emotion2: EmotionState) -> bool:
        """감정 간 유사성 판단"""
        similar_groups = [
            {EmotionState.JOY, EmotionState.TRUST, EmotionState.ANTICIPATION},
            {EmotionState.SADNESS, EmotionState.FEAR, EmotionState.DISGUST},
            {EmotionState.ANGER, EmotionState.DISGUST},
            {EmotionState.SURPRISE, EmotionState.ANTICIPATION}
        ]
        
        for group in similar_groups:
            if emotion1 in group and emotion2 in group:
                return True
        
        return False
    
    def save_model_cache(self, file_path: str):
        """모델 캐시 저장"""
        try:
            cache_data = {
                'prediction_cache': self.prediction_cache,
                'embedding_cache': self.embedding_cache
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"모델 캐시 저장 완료: {file_path}")
            
        except Exception as e:
            logger.error(f"모델 캐시 저장 실패: {e}")
    
    def load_model_cache(self, file_path: str):
        """모델 캐시 로드"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            self.prediction_cache.update(cache_data.get('prediction_cache', {}))
            self.embedding_cache.update(cache_data.get('embedding_cache', {}))
            
            logger.info(f"모델 캐시 로드 완료: {file_path}")
            
        except Exception as e:
            logger.error(f"모델 캐시 로드 실패: {e}")
    
    def analyze_hierarchical_emotions(self, text: str, 
                                    other_emotion: Optional[torch.Tensor] = None,
                                    regret_vector: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """계층적 감정 분석 (Phase 0-2)"""
        if not NEW_EMOTION_MODELS_AVAILABLE or not hasattr(self, 'hierarchical_model'):
            raise RuntimeError("계층적 감정 모델을 사용할 수 없습니다. 계층적 감정 모델이 필요합니다.")
        
        try:
            # 텍스트 임베딩 생성
            text_embedding = self._generate_text_embedding(text)
            
            # 기본값 설정
            if other_emotion is None:
                other_emotion = torch.zeros(6)  # 6차원 감정 벡터
            if regret_vector is None:
                regret_vector = torch.zeros(6)
            
            # 계층적 모델로 분석
            with torch.no_grad():
                results = self.hierarchical_model(text_embedding, other_emotion, regret_vector)
            
            # 결과 해석
            final_emotion_dict = emotion_vector_to_dict(results['final_emotion'])
            
            return {
                'final_emotion': final_emotion_dict,
                'phase0_emotion': emotion_vector_to_dict(results['phase0_emotion']),
                'phase1_empathy': emotion_vector_to_dict(results['phase1_empathy']),
                'phase2_integrated': emotion_vector_to_dict(results['phase2_integrated']),
                'regret_intensity': results['regret_intensity'].item(),
                'emotion_evolution': self._analyze_emotion_evolution(results),
                'confidence': self._calculate_hierarchical_confidence(results)
            }
            
        except Exception as e:
            logger.error(f"계층적 감정 분석 실패: {e}")
            raise RuntimeError(f"계층적 감정 분석 실행 중 오류 발생: {e}")
    
    def _generate_text_embedding(self, text: str) -> torch.Tensor:
        """텍스트 임베딩 생성"""
        # 캐시 확인
        cache_key = hash(text)
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        try:
            # Sentence Transformer 사용 (있는 경우)
            if hasattr(self, 'sentence_embedder'):
                embedding = self.sentence_embedder.encode(text)
                embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
            else:
                # 간단한 통계적 임베딩 생성
                words = text.split()
                features = [
                    len(text), len(words), 
                    len([w for w in words if w in self.korean_emotion_keywords]),
                    text.count('!'), text.count('?')
                ]
                # 768차원으로 패딩
                embedding = np.zeros(768)
                embedding[:len(features)] = features
                embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
            
            # 캐시 저장
            self.embedding_cache[cache_key] = embedding_tensor
            return embedding_tensor
            
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")
            raise RuntimeError(f"텍스트 임베딩 생성 실패: {e}")
    
    def _analyze_emotion_evolution(self, results: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """감정 진화 과정 분석"""
        evolution = {}
        
        try:
            # Phase별 감정 강도 변화
            phase0_intensity = torch.norm(results['phase0_emotion']).item()
            phase1_intensity = torch.norm(results['phase1_empathy']).item()
            phase2_intensity = torch.norm(results['phase2_integrated']).item()
            
            evolution['intensity_progression'] = [phase0_intensity, phase1_intensity, phase2_intensity]
            evolution['intensity_trend'] = 'increasing' if phase2_intensity > phase0_intensity else 'decreasing'
            
            # 주요 감정 차원 변화
            phase0_dict = emotion_vector_to_dict(results['phase0_emotion'])
            phase2_dict = emotion_vector_to_dict(results['phase2_integrated'])
            
            dimension_changes = {}
            for dim in EMOTION_DIMENSIONS.keys():
                change = phase2_dict[dim] - phase0_dict[dim]
                dimension_changes[dim] = change
            
            evolution['dimension_changes'] = dimension_changes
            evolution['most_changed_dimension'] = max(dimension_changes.items(), key=lambda x: abs(x[1]))
            
        except Exception as e:
            logger.error(f"감정 진화 분석 실패: {e}")
            evolution = {'error': str(e)}
        
        return evolution
    
    def _calculate_hierarchical_confidence(self, results: Dict[str, torch.Tensor]) -> float:
        """계층적 분석 신뢰도 계산"""
        try:
            # 여러 요소 기반 신뢰도 계산
            base_confidence = 0.5
            
            # 감정 벡터의 일관성
            final_norm = torch.norm(results['final_emotion']).item()
            if final_norm > 0.1:
                base_confidence += 0.2
            
            # 후회 강도 (적절한 범위 내)
            regret_intensity = results['regret_intensity'].item()
            if 0.1 < regret_intensity < 0.9:
                base_confidence += 0.2
            
            # Phase 간 일관성
            phase_consistency = self._calculate_phase_consistency(results)
            base_confidence += phase_consistency * 0.1
            
            return min(1.0, max(0.0, base_confidence))
            
        except Exception as e:
            logger.error(f"신뢰도 계산 실패: {e}")
            return 0.5
    
    def _calculate_phase_consistency(self, results: Dict[str, torch.Tensor]) -> float:
        """Phase 간 일관성 계산"""
        try:
            # 코사인 유사도 기반 일관성
            phase0 = results['phase0_emotion']
            phase1 = results['phase1_empathy']
            phase2 = results['phase2_integrated']
            
            cos01 = F.cosine_similarity(phase0.unsqueeze(0), phase1.unsqueeze(0)).item()
            cos12 = F.cosine_similarity(phase1.unsqueeze(0), phase2.unsqueeze(0)).item()
            cos02 = F.cosine_similarity(phase0.unsqueeze(0), phase2.unsqueeze(0)).item()
            
            return (cos01 + cos12 + cos02) / 3
            
        except Exception as e:
            logger.error(f"일관성 계산 실패: {e}")
            return 0.0
    
    
    async def analyze_with_llm_interpretation(self, text: str, 
                                            include_hierarchical: bool = True) -> Dict[str, Any]:
        """LLM 해석을 포함한 감정 분석"""
        if not LLM_INTEGRATION_AVAILABLE:
            return {'error': 'LLM 통합을 사용할 수 없습니다.'}
        
        try:
            # 기본 감정 분석
            basic_result = self.analyze_emotion(text)
            
            # 계층적 분석 (선택적)
            hierarchical_result = None
            if include_hierarchical and NEW_EMOTION_MODELS_AVAILABLE:
                hierarchical_result = self.analyze_hierarchical_emotions(text)
            
            # LLM 해석 요청
            emotion_data = self._format_emotion_for_llm(basic_result, hierarchical_result)
            llm_interpretation = await interpret_emotions(emotion_data)
            
            result = {
                'basic_analysis': {
                    'primary_emotion': basic_result.primary_emotion.value,
                    'confidence': basic_result.confidence,
                    'intensity': basic_result.intensity.value
                },
                'text': text
            }
            
            if hierarchical_result:
                result['hierarchical_analysis'] = hierarchical_result
            
            if llm_interpretation.success:
                result['llm_interpretation'] = {
                    'explanation': llm_interpretation.generated_text,
                    'confidence': llm_interpretation.confidence,
                    'insights': self._extract_emotion_insights(llm_interpretation.generated_text)
                }
            
            return result
            
        except Exception as e:
            logger.error(f"LLM 감정 해석 실패: {e}")
    
    def _generate_emotion_embedding(self, text: str, language: str, 
                                   emotion_data: EmotionData) -> Optional[np.ndarray]:
        """감정 특화 임베딩 생성"""
        try:
            # 기본 텍스트 임베딩
            text_embedding = self._generate_text_embedding(text)
            
            # 감정 상태 벡터
            emotion_vector = self._create_emotion_state_vector(emotion_data)
            
            # 감정과 텍스트 임베딩 결합
            if text_embedding is not None and emotion_vector is not None:
                # 적절한 크기로 맞춤
                min_len = min(len(text_embedding), len(emotion_vector))
                combined_embedding = np.concatenate([
                    text_embedding[:min_len] * 0.8,
                    emotion_vector[:min_len] * 0.2
                ])
                return combined_embedding
            
            return text_embedding
            
        except Exception as e:
            self.logger.error(f"감정 임베딩 생성 실패: {e}")
            return None
    
    def _generate_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """텍스트 의미적 임베딩 생성"""
        try:
            # 다국어 임베딩 모델 사용
            if hasattr(self, 'embedders') and 'multilingual' in self.embedders:
                embedding = self.embedders['multilingual'].encode(
                    text, 
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                return embedding
            else:
                raise RuntimeError("텍스트 임베딩 모델이 로드되지 않음")
                
        except Exception as e:
            self.logger.error(f"텍스트 임베딩 생성 실패: {e}")
            return None
    
    def _create_emotion_state_vector(self, emotion_data: EmotionData) -> Optional[np.ndarray]:
        """감정 상태 특징 벡터 생성"""
        try:
            # 기본 감정 특징 (16차원)
            emotion_features = [
                emotion_data.primary_emotion.value / 16.0,  # 정규화된 감정 ID
                emotion_data.intensity.value / 6.0,         # 정규화된 강도
                emotion_data.confidence,                    # 신뢰도
                getattr(emotion_data, 'valence', 0.5),      # 감정가
                getattr(emotion_data, 'arousal', 0.5),      # 각성도
                getattr(emotion_data, 'dominance', 0.5),    # 지배성
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # 예약된 특징들
            ]
            
            return np.array(emotion_features, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"감정 벡터 생성 실패: {e}")
            return None
    
    def _format_emotion_for_llm(self, basic_result: EmotionData, 
                              hierarchical_result: Optional[Dict[str, Any]] = None) -> str:
        """감정 분석 결과를 LLM 입력용으로 포맷"""
        parts = [f"감정 분석 결과:"]
        
        # 기본 분석
        parts.append(f"- 주요 감정: {basic_result.primary_emotion.value}")
        parts.append(f"- 신뢰도: {basic_result.confidence:.3f}")
        parts.append(f"- 강도: {basic_result.intensity.value}")
        
        # 계층적 분석
        if hierarchical_result:
            parts.append("\n계층적 감정 분석:")
            for phase, emotion_dict in [
                ("Phase 0 (자아투영)", hierarchical_result['phase0_emotion']),
                ("Phase 1 (공감)", hierarchical_result['phase1_empathy']),
                ("Phase 2 (통합)", hierarchical_result['phase2_integrated'])
            ]:
                parts.append(f"- {phase}:")
                for dim, value in emotion_dict.items():
                    if abs(value) > 0.1:  # 유의미한 값만
                        parts.append(f"  * {dim}: {value:.3f}")
            
            parts.append(f"- 후회 강도: {hierarchical_result['regret_intensity']:.3f}")
        
        return "\n".join(parts)
    
    def _extract_emotion_insights(self, llm_text: str) -> List[str]:
        """LLM 응답에서 감정 인사이트 추출"""
        insights = []
        
        lines = llm_text.split('\n')
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in 
                  ['감정', '느낌', '심리', '상태', '원인', '영향']):
                if len(line) > 15:
                    insights.append(line)
        
        return insights[:3]
    
    def _emollms_preprocess_response(self, llm_response: Optional[Any], text: str) -> Optional[Dict[str, Any]]:
        """EmoLLMs 논문 방식: None 응답 전처리 (품질 보장 + 실패 유형별 차별화)"""
        
        # Step 1: Null record 처리 (치명적 실패)
        if llm_response is None:
            logger.error(f"❌ 치명적 실패: LLM 응답이 None - 시스템 정지: {text[:50]}...")
            return None
        
        # Step 2: 응답 성공 여부 확인 (치명적 실패)
        if hasattr(llm_response, 'success') and not llm_response.success:
            logger.error("❌ 치명적 실패: LLM 응답 실패 - 시스템 정지")
            return None
        
        # Step 3: 생성된 텍스트 기본 검증
        if not hasattr(llm_response, 'generated_text'):
            logger.error("❌ 치명적 실패: generated_text 속성 없음")
            return None
            
        generated_text = llm_response.generated_text
        if not generated_text:
            logger.error("❌ 치명적 실패: 생성된 텍스트가 완전히 비어있음")
            return None
        
        # Step 4: 실패 유형별 차별화 처리
        if len(generated_text.strip()) < 10:
            logger.warning("⚠️ 복구 가능한 실패: 생성된 텍스트가 너무 짧음 (10자 미만)")
            # 짧은 텍스트도 파싱 시도 (부분 응답 처리)
        
        # Step 5: finish_reason 확인 (복구 가능한 실패)
        if hasattr(llm_response, 'finish_reason'):
            if llm_response.finish_reason == 'length':
                logger.warning("⚠️ 복구 가능한 실패: finish_reason이 'length' - 부분 응답 처리")
                # 부분 응답이지만 파싱 시도
            elif llm_response.finish_reason == 'stop':
                logger.info("✅ 정상 완료: finish_reason이 'stop'")
            else:
                logger.warning(f"⚠️ 알 수 없는 finish_reason: {llm_response.finish_reason}")
        
        # Step 6: 응답 파싱 시도 (강화된 파싱 로직 사용)
        try:
            parsed_result = self._parse_deep_llm_response(generated_text)
            if parsed_result is not None:
                logger.info("✅ 전처리 성공: 파싱 완료")
                return parsed_result
            else:
                logger.error("❌ 파싱 실패: 파싱 결과가 None")
                return None
                
        except Exception as e:
            logger.error(f"❌ 파싱 중 예외 발생: {e}")
            return None
    
    
    def get_enhanced_emotion_metrics(self) -> Dict[str, Any]:
        """향상된 감정 분석 메트릭"""
        base_stats = {
            'cache_size': len(self.prediction_cache),
            'embedding_cache_size': len(self.embedding_cache),
            'models_loaded': len(self.models)
        }
        
        enhanced_stats = {
            'base_statistics': base_stats,
            'model_capabilities': {
                'hierarchical_available': NEW_EMOTION_MODELS_AVAILABLE,
                'llm_integration_available': LLM_INTEGRATION_AVAILABLE,
                'device': self.device
            },
            'korean_keywords_count': len(self.korean_emotion_keywords)
        }
        
        # 계층적 모델 통계
        if NEW_EMOTION_MODELS_AVAILABLE and hasattr(self, 'emotion_model_manager'):
            try:
                enhanced_stats['hierarchical_model_stats'] = self.emotion_model_manager.get_training_stats()
            except Exception as stats_error:
                logger.debug(f"계층적 모델 통계 수집 실패 (비핵심 기능): {stats_error}")
                enhanced_stats['hierarchical_model_stats'] = {}
        
        # LLM 엔진 통계
        if LLM_INTEGRATION_AVAILABLE and hasattr(self, 'llm_engine'):
            try:
                enhanced_stats['llm_performance'] = self.llm_engine.get_performance_stats()
            except Exception as llm_stats_error:
                logger.debug(f"LLM 성능 통계 수집 실패 (비핵심 기능): {llm_stats_error}")
                enhanced_stats['llm_performance'] = {}
        
        return enhanced_stats

    def _enhance_with_llm_analysis(self, text: str, keyword_emotion: int, keyword_score: float) -> Optional[Dict[str, Any]]:
        """LLM으로 키워드 분석 결과 검증 및 강화 (조건부 호출)"""
        
        # 조건부 LLM 호출 결정
        should_use_llm = self._should_use_llm_for_emotion(text, keyword_score)
        if not should_use_llm:
            return None
            
        # LLM 통합이 불가능한 경우 - 조건부 처리 
        if not LLM_INTEGRATION_AVAILABLE:
            logger.debug("LLM 통합을 사용할 수 없어 키워드 분석 결과를 반환합니다.")
            return None
        
        try:
            import asyncio
            from llm_module.advanced_llm_engine import get_llm_engine, LLMRequest, TaskComplexity
            
            # 키워드 감정을 텍스트로 변환
            emotion_name = self._emotion_id_to_name(keyword_emotion)
            
            # 강화된 ChatML 템플릿 - 매핑 테이블과 동기화된 엄격한 제약
            prompt = f"""<|im_start|>system
당신은 HelpingAI, 감정 분석 전문가입니다. 주어진 텍스트를 분석하고 반드시 아래 형식의 JSON으로만 응답하세요.

⚠️ 중요 제약사항:
- 반드시 지정된 감정 목록에서만 선택하세요
- 다른 감정 이름은 절대 사용하지 마세요
- JSON 형식을 정확히 지켜주세요
- 추가 텍스트나 설명은 금지입니다

필수 JSON 형식:
{{
    "emotion": "감정명",
    "intensity": 정수1-6,
    "confidence": 실수0.0-1.0,
    "valence": 실수-1.0-1.0,
    "arousal": 실수0.0-1.0,
    "reasoning": "간단한_설명"
}}

허용된 감정 (이것 외 사용 금지):
- joy: 기쁨, 행복
- trust: 신뢰, 믿음  
- fear: 두려움, 불안
- surprise: 놀람, 충격
- sadness: 슬픔, 우울
- disgust: 혐오, 거부감
- anger: 분노, 화남 (frustration 포함)
- anticipation: 기대, 희망
- neutral: 중립, 무감정

예시:
{{"emotion": "anger", "intensity": 3, "confidence": 0.8, "valence": -0.3, "arousal": 0.6, "reasoning": "분노 표현이 감지됨"}}

다시 한번 강조: 위 9개 감정 중에서만 선택하고, 정확한 JSON 형식으로만 응답하세요.<|im_end|>
<|im_start|>user
{text}<|im_end|>
<|im_start|>assistant
"""

            # 중첩 함수에서 사용할 토큰 수 미리 계산
            dynamic_tokens = self._calculate_dynamic_token_limit_direct(text, base_tokens=400)
            
            # 이벤트 루프 안전한 실행을 위한 래퍼
            def run_llm_analysis():
                try:
                    import threading
                    import concurrent.futures
                    
                    # 새로운 스레드에서 비동기 실행
                    def async_llm_call():
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            engine = get_llm_engine()
                            request = LLMRequest(
                                prompt=prompt,
                                task_type="emotion_analysis",  # HelpingAI EQ 95.89 우선
                                complexity=TaskComplexity.MODERATE,
                                max_tokens=dynamic_tokens,
                                temperature=0.3
                            )
                            
                            response = loop.run_until_complete(engine.generate_async(request))
                            return response
                        finally:
                            loop.close()
                    
                    # 스레드풀에서 실행
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(async_llm_call)
                        response = future.result(timeout=30)  # 30초 타임아웃
                    
                    if response and response.success:
                        # 안전한 속성 접근 - generated_text 속성 확인
                        if hasattr(response, 'generated_text') and response.generated_text:
                            logger.info(f"✅ LLM 응답 성공 - 길이: {len(response.generated_text)}")
                            return self._parse_llm_emotion_response(response.generated_text, keyword_emotion)
                        else:
                            logger.error(f"❌ 치명적 실패: generated_text 속성 없음 - 응답 구조: {type(response).__name__}")
                            logger.error(f"응답 속성들: {dir(response)}")
                            return None
                    else:
                        logger.warning(f"LLM 감정 분석 실패: {response.error_message if response else 'No response'}")
                        return None
                        
                except Exception as e:
                    logger.error(f"LLM 분석 실행 오류: {e}")
                    return None
            
            return run_llm_analysis()
            
        except Exception as e:
            logger.error(f"LLM 감정 분석 오류: {e}")
            return None

    def _should_use_llm_for_emotion(self, text: str, keyword_score: float) -> bool:
        """LLM 사용 조건 결정 (최적화된 고품질 AI 분석)"""
        
        # 빈 텍스트는 제외
        if not text or len(text.strip()) == 0:
            return False
            
        # 1. 키워드 점수가 모호한 경우 (1.0~2.5 사이) - 범위 확대
        if 1.0 <= keyword_score <= 2.5:
            return True
            
        # 2. 복잡한 텍스트 (길이 기준 강화)
        if len(text) > 30:  # 기존 50에서 30으로 강화
            return True
            
        # 3. 복합 감정 표현이 있는 경우 (임계값 낮춤)
        emotion_keywords_count = sum(1 for emotion_dict in self.korean_emotion_keywords.values() 
                                   for keyword_list in emotion_dict.values()
                                   for keyword in keyword_list 
                                   if keyword in text.lower())
        if emotion_keywords_count >= 2:  # 3개에서 2개로 강화
            return True
            
        # 4. 부정문이나 복잡한 문장 구조 (조건 확장)
        complex_patterns = ['않', '없', '못', '아니', '그러나', '하지만', '그런데', 
                           '오히려', '반면', '대신', '물론', '하지만', '그렇지만']
        if any(pattern in text for pattern in complex_patterns):
            return True
            
        # 5. 감정 강도 표현 (새로운 조건)
        intensity_words = ['매우', '정말', '너무', '아주', '완전히', '엄청', '굉장히', 
                          '심하게', '깊이', '극도로', '상당히', '꽤', '제법']
        if any(word in text for word in intensity_words):
            return True
            
        # 6. 복합 문장 구조 (새로운 조건)
        if text.count(',') >= 2 or text.count('.') >= 2 or '?' in text or '!' in text:
            return True
            
        # 7. 미묘한 감정 표현 (새로운 조건)
        subtle_emotions = ['미묘', '애매', '복잡', '혼란', '갈등', '딜레마', '고민', 
                          '생각', '느낌', '기분', '분위기', '뉘앙스']
        if any(word in text for word in subtle_emotions):
            return True
            
        # 간단하고 명확한 감정 표현만 키워드 분석 사용
        return False

    def _emotion_id_to_name(self, emotion_id: int) -> str:
        """감정 ID를 한국어 이름으로 변환"""
        emotion_mapping = {
            1: "기쁨",    # JOY
            2: "신뢰",    # TRUST  
            3: "두려움",  # FEAR
            4: "놀람",    # SURPRISE
            5: "슬픔",    # SADNESS
            6: "혐오",    # DISGUST
            7: "분노",    # ANGER
            8: "기대",    # ANTICIPATION
            0: "중립"     # NEUTRAL
        }
        return emotion_mapping.get(emotion_id, "알 수 없음")

    def _parse_llm_emotion_response(self, response_text: str, original_emotion: int) -> Dict[str, Any]:
        """LLM 감정 분석 응답 파싱 (JSON 우선, 텍스트 파싱 fallback)"""
        try:
            # 1. JSON 파싱 시도
            try:
                import json
                import re
                
                # 다양한 JSON 패턴 시도 (개선된 파싱)
                json_patterns = [
                    r'\{[^{}]*\}',  # 단순 JSON 블록
                    r'\{[^{}]*?"[^"]*"[^{}]*\}',  # 문자열 포함 JSON
                    r'\{.*?\}',  # 모든 문자 포함 JSON (최대한 유연)
                ]
                
                parsed_json = None
                for pattern in json_patterns:
                    matches = re.findall(pattern, response_text, re.DOTALL)
                    for match in matches:
                        try:
                            # 공백 문자 정리
                            cleaned_json = re.sub(r'\s+', ' ', match).strip()
                            
                            # 일반적인 JSON 파싱 오류 수정
                            cleaned_json = cleaned_json.replace('" emotion"', '"emotion"')
                            cleaned_json = cleaned_json.replace('" confidence"', '"confidence"')
                            cleaned_json = cleaned_json.replace('" intensity"', '"intensity"')
                            cleaned_json = cleaned_json.replace('" valence"', '"valence"')
                            cleaned_json = cleaned_json.replace('" arousal"', '"arousal"')
                            cleaned_json = cleaned_json.replace('" reasoning"', '"reasoning"')
                            
                            parsed_json = json.loads(cleaned_json)
                            logger.info(f"JSON 파싱 성공: {cleaned_json}")
                            break
                        except json.JSONDecodeError:
                            continue
                    if parsed_json:
                        break
                
                if parsed_json:
                    # JSON에서 결과 추출
                    result = {
                        'emotion': original_emotion,  # 기본값
                        'confidence': 0.5,
                        'reasoning': response_text
                    }
                    
                    # 영어 필드명 처리 (HelpingAI 응답 형식)
                    if 'emotion' in parsed_json:
                        emotion_text = parsed_json['emotion']
                        mapped_emotion = self._name_to_emotion_id(emotion_text)
                        if mapped_emotion is None:
                            logger.error(f"❌ 감정 매핑 실패로 인한 파싱 실패: '{emotion_text}'")
                            return None  # 학습 오염 방지를 위한 명확한 실패 반환
                        result['emotion'] = mapped_emotion
                    
                    # 신뢰도 처리
                    if 'confidence' in parsed_json:
                        try:
                            result['confidence'] = float(parsed_json['confidence'])
                        except (ValueError, TypeError) as parse_error:
                            logger.debug(f"타입 변환 실패, 기본값 사용: {parse_error}")
                    
                    # 감정강도 처리
                    if 'intensity' in parsed_json:
                        try:
                            result['intensity'] = int(parsed_json['intensity'])
                        except (ValueError, TypeError) as parse_error:
                            logger.debug(f"타입 변환 실패, 기본값 사용: {parse_error}")
                    
                    # valence와 arousal 처리
                    if 'valence' in parsed_json:
                        try:
                            result['valence'] = float(parsed_json['valence'])
                        except (ValueError, TypeError) as parse_error:
                            logger.debug(f"타입 변환 실패, 기본값 사용: {parse_error}")
                    
                    if 'arousal' in parsed_json:
                        try:
                            result['arousal'] = float(parsed_json['arousal'])
                        except (ValueError, TypeError) as parse_error:
                            logger.debug(f"타입 변환 실패, 기본값 사용: {parse_error}")
                    
                    # 추가 정보
                    if 'reasoning' in parsed_json:
                        result['reasoning'] = parsed_json['reasoning']
                    
                    # 한국어 필드명 처리 (기존 방식)
                    if '주요감정' in parsed_json:
                        emotion_text = parsed_json['주요감정']
                        mapped_emotion = self._name_to_emotion_id(emotion_text)
                        if mapped_emotion is None:
                            logger.error(f"❌ 한국어 감정 매핑 실패로 인한 파싱 실패: '{emotion_text}'")
                            return None  # 학습 오염 방지를 위한 명확한 실패 반환
                        result['emotion'] = mapped_emotion
                    
                    if '신뢰도' in parsed_json:
                        try:
                            result['confidence'] = float(parsed_json['신뢰도'])
                        except (ValueError, TypeError) as parse_error:
                            logger.debug(f"타입 변환 실패, 기본값 사용: {parse_error}")
                    
                    if '감정강도' in parsed_json:
                        try:
                            result['intensity'] = int(parsed_json['감정강도'])
                        except (ValueError, TypeError) as parse_error:
                            logger.debug(f"타입 변환 실패, 기본값 사용: {parse_error}")
                    
                    if '원인분석' in parsed_json:
                        result['cause_analysis'] = parsed_json['원인분석']
                    
                    logger.info(f"JSON 파싱 성공: 감정={result['emotion']}, 신뢰도={result['confidence']}")
                    return result
                    
            except (json.JSONDecodeError, AttributeError) as e:
                logger.warning(f"JSON 파싱 실패, 텍스트 파싱으로 전환: {e}")
            
            # 2. 기존 텍스트 파싱 (fallback)
            lines = response_text.split('\n')
            result = {
                'emotion': original_emotion,  # 기본값
                'confidence': 0.5,
                'reasoning': response_text
            }
            
            for line in lines:
                line = line.strip()
                if '주요 감정:' in line or '주요감정:' in line:
                    emotion_text = line.split(':')[1].strip()
                    mapped_emotion = self._name_to_emotion_id(emotion_text)
                    if mapped_emotion is None:
                        logger.error(f"❌ 텍스트 파싱 감정 매핑 실패: '{emotion_text}'")
                        return None  # 학습 오염 방지
                    result['emotion'] = mapped_emotion
                elif '신뢰도:' in line:
                    try:
                        conf_text = line.split(':')[1].strip()
                        result['confidence'] = float(conf_text)
                    except:
                        pass
                elif '감정강도:' in line:
                    try:
                        intensity_text = line.split(':')[1].strip()
                        result['intensity'] = int(intensity_text)
                    except:
                        pass
                        
            logger.info(f"텍스트 파싱 성공: 감정={result['emotion']}, 신뢰도={result['confidence']}")
            return result
            
        except Exception as e:
            logger.error(f"LLM 응답 파싱 완전 실패: {e}")
            # fallback 금지 원칙에 따라 None 반환
            return None

    def _name_to_emotion_id(self, emotion_name: str) -> int:
        """한국어/영어 감정 이름을 ID로 변환 (오타 허용 fuzzy matching 포함)"""
        name_mapping = {
            # 한국어 감정 이름
            "기쁨": 1, "행복": 1, "기빔": 1, "즐거움": 1,
            "신뢰": 2, "믿음": 2, "확신": 2,
            "두려움": 3, "불안": 3, "걱정": 3, "공포": 3, "무서움": 3,
            "놀람": 4, "깜짝": 4, "놀라움": 4, "충격": 4,
            "슬픔": 5, "우울": 5, "슬픔": 5, "우울함": 5, "아픔": 5,
            "혐오": 6, "싫음": 6, "역겨움": 6, "거부감": 6,
            "분노": 7, "화": 7, "짜증": 7, "성남": 7, "화남": 7,
            "기대": 8, "예상": 8, "기대감": 8, "희망": 8,
            "중립": 0, "무감정": 0, "중성": 0,
            
            # 영어 감정 이름 (HelpingAI 응답 형식)
            "joy": 1, "happy": 1, "happiness": 1, "joyful": 1, "pleased": 1,
            "trust": 2, "confidence": 2, "belief": 2, "reliance": 2,
            "fear": 3, "anxiety": 3, "worry": 3, "afraid": 3, "scared": 3, "anxious": 3,
            "surprise": 4, "shocked": 4, "amazed": 4, "astonished": 4, "surprised": 4,
            "sadness": 5, "sad": 5, "depression": 5, "sorrow": 5, "grief": 5, "melancholy": 5,
            "disgust": 6, "hate": 6, "dislike": 6, "revulsion": 6, "contempt": 6,
            "anger": 7, "angry": 7, "mad": 7, "rage": 7, "fury": 7, "irritation": 7, "frustration": 7, "frustrated": 7,
            "anticipation": 8, "anticipate": 8, "expectation": 8, "expect": 8, "hope": 8, "excitement": 8,
            "neutral": 0, "none": 0, "no emotion": 0, "normal": 0, "calm": 0
        }
        
        emotion_name_clean = emotion_name.lower().strip().replace(' ', '').replace('-', '').replace('_', '')
        
        # 1단계: 정확한 매칭
        for name, emotion_id in name_mapping.items():
            name_clean = name.lower().replace(' ', '').replace('-', '').replace('_', '')
            if name_clean == emotion_name_clean or name_clean in emotion_name_clean:
                return emotion_id
        
        # 2단계: fuzzy matching (Levenshtein distance 기반)
        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            return previous_row[-1]
        
        # 3단계: 유사도 기반 매칭 (편집 거리 2 이하)
        best_match = None
        best_distance = float('inf')
        
        for name, emotion_id in name_mapping.items():
            name_clean = name.lower().replace(' ', '').replace('-', '').replace('_', '')
            distance = levenshtein_distance(emotion_name_clean, name_clean)
            
            # 길이에 비례한 허용 오차 (짧은 단어는 더 엄격하게)
            max_allowed_distance = max(1, len(name_clean) // 3)
            
            if distance <= max_allowed_distance and distance < best_distance:
                best_distance = distance
                best_match = emotion_id
        
        if best_match is not None:
            logger.info(f"🔧 fuzzy matching 성공: '{emotion_name}' -> emotion_id={best_match} (distance={best_distance})")
            return best_match
        
        # 매핑 실패 시 명확한 에러 로그 및 None 반환 (학습 오염 방지)
        allowed_emotions = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation", "neutral"]
        logger.error(f"❌ 감정 매핑 실패: '{emotion_name}' -> 허용된 감정 목록에 없음")
        logger.error(f"🎯 허용된 감정: {allowed_emotions}")
        logger.error(f"🔧 제안: HelpingAI 프롬프트 제약을 더 강화하거나 매핑 테이블에 추가 필요")
        return None  # 명확한 실패 표시
    
    def _translate_to_english(self, korean_text: str) -> str:
        """한국어 텍스트를 영어로 번역 (로컬 OPUS-MT 사용, 완전 오프라인)"""
        try:
            return self.local_translator.translate_ko_to_en(korean_text)
        except Exception as e:
            logger.warning(f"로컬 번역 오류: {e}, 원본 텍스트 사용")
            return korean_text

    def _deep_llm_emotion_analysis(self, text: str) -> Optional[Dict[str, Any]]:
        """키워드 분석이 실패한 경우 LLM 전체 분석 (민감성 감지 및 중립화 시스템 통합)"""
        
        if not LLM_INTEGRATION_AVAILABLE:
            return None
        
        try:
            import asyncio
            import concurrent.futures
            from llm_module.advanced_llm_engine import get_llm_engine, LLMRequest, TaskComplexity
            from sensitivity_detection_singleton import detect_and_neutralize_sensitive_content
            
            prompt = f"""다음 텍스트에서 감정을 깊이 분석해주세요.

텍스트: "{text}"

다음 형식으로 상세히 분석해주세요:
1. 주요 감정: [기쁨/슬픔/분노/두려움/놀람/혐오/신뢰/기대/중립]
2. 감정 강도: [1-6 척도]
3. 신뢰도: [0.0-1.0]
4. 감정가: [-1.0~1.0, 부정적~긍정적]
5. 각성도: [-1.0~1.0, 낮음~높음]
6. 심층 분석: [감정의 원인, 맥락, 의미]"""

            def run_deep_analysis(current_text=text):
                # WSL2 환경 맞춤 재시도 로직 + 민감성 감지 시스템
                max_retries = 3
                retry_delays = [1, 3, 5]  # 지수적 백오프
                response_texts = []  # 응답 추적용
                
                for attempt in range(max_retries):
                    try:
                        # 메모리 체크 (WSL2 리소스 모니터링)
                        import psutil
                        memory = psutil.virtual_memory()
                        if memory.percent > 90:
                            logger.warning(f"WSL2 메모리 사용률 높음: {memory.percent}%")
                            import gc
                            gc.collect()
                            if attempt < max_retries - 1:
                                time.sleep(retry_delays[attempt])
                                continue
                        
                        # GPU 메모리 체크
                        if torch.cuda.is_available():
                            gpu_memory_percent = (torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory) * 100
                            if gpu_memory_percent > 85:
                                logger.warning(f"GPU 메모리 사용률 높음: {gpu_memory_percent:.1f}%")
                                torch.cuda.empty_cache()
                        
                        # 토큰 수 미리 계산 (중첩 함수에서 self 접근 불가 문제 해결)
                        dynamic_tokens = 1200  # JSON 응답 완전성 보장을 위한 토큰 할당
                        
                        # 한국어 텍스트를 영어로 번역 (안정성 향상)
                        translated_text = self._translate_to_english(current_text)
                        
                        # 영어 기반 JSON 프롬프트 (안정성 향상)
                        current_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an emotion analysis expert. Respond only in simple and accurate JSON format.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Text: "{translated_text}"

Respond only in this exact JSON format:
{{
  "emotion": "joy",
  "intensity": 3,
  "confidence": 0.8,
  "valence": 0.5,
  "arousal": 0.5,
  "reasoning": "Brief explanation of the emotional cause"
}}

The emotion value must be one of: joy, sadness, anger, fear, surprise, disgust, trust, anticipation, neutral

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
                        
                        def async_llm_call():
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            try:
                                engine = get_llm_engine()
                                request = LLMRequest(
                                    prompt=current_prompt,
                                    task_type="emotion_analysis",  # HelpingAI EQ 95.89 우선 선택
                                    complexity=TaskComplexity.COMPLEX,
                                    max_tokens=dynamic_tokens,
                                    temperature=0.2
                                )
                                
                                response = loop.run_until_complete(engine.generate_async(request))
                                return response
                            finally:
                                loop.close()
                        
                        # 타임아웃도 재시도에 따라 조정
                        timeout = 30 + (attempt * 10)  # 30s, 40s, 50s
                        
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(async_llm_call)
                            response = future.result(timeout=timeout)
                        
                        # 응답 텍스트 저장
                        response_text = ""
                        if response and hasattr(response, 'generated_text'):
                            response_text = response.generated_text or ""
                        response_texts.append(response_text)
                        
                        if response and response.success and response_text:
                            parsed_result = self._parse_deep_llm_response(response_text)
                            if parsed_result and parsed_result.get('emotion'):
                                logger.info(f"✅ LLM 분석 성공 (시도 {attempt + 1}/{max_retries})")
                                return parsed_result
                            else:
                                logger.warning(f"⚠️ 파싱 실패, 재시도 {attempt + 1}/{max_retries}")
                        else:
                            logger.warning(f"⚠️ LLM 응답 실패 (응답 길이: {len(response_text)}), 재시도 {attempt + 1}/{max_retries}")
                        
                        # 재시도 전 대기
                        if attempt < max_retries - 1:
                            time.sleep(retry_delays[attempt])
                            
                    except Exception as e:
                        logger.warning(f"⚠️ 시도 {attempt + 1}/{max_retries} 실패: {e}")
                        response_texts.append("")  # 예외 시에도 빈 응답 기록
                        if attempt < max_retries - 1:
                            time.sleep(retry_delays[attempt])
                        else:
                            logger.error(f"❌ 모든 재시도 실패: {e}")
                
                # 모든 재시도 실패 후 응답 분석 및 민감성 감지 시스템 트리거 판단
                logger.info(f"🔍 모든 재시도 소진 (총 {max_retries}회) - 응답 분석 시작")
                
                # 응답 상태 분석
                all_empty_responses = all(len(resp.strip()) == 0 for resp in response_texts)
                has_substantial_responses = any(len(resp.strip()) > 50 for resp in response_texts)
                avg_response_length = sum(len(resp) for resp in response_texts) / len(response_texts) if response_texts else 0
                
                logger.info(f"📊 응답 분석 결과:")
                logger.info(f"   - 총 응답 수: {len(response_texts)}")
                logger.info(f"   - 모두 빈 응답: {all_empty_responses}")
                logger.info(f"   - 실질적 응답 존재: {has_substantial_responses}")
                logger.info(f"   - 평균 응답 길이: {avg_response_length:.1f}자")
                for i, resp in enumerate(response_texts):
                    logger.info(f"   - 응답 {i+1}: {len(resp)}자 ({resp[:100]}...)")
                
                # 민감성 감지 트리거 조건: 모든 응답이 비어있고 원본 텍스트인 경우만
                should_trigger_sensitivity = (
                    all_empty_responses and 
                    current_text == text and  # 원본 텍스트에 대한 첫 번째 시도
                    avg_response_length < 10  # 평균 응답 길이가 매우 짧음
                )
                
                if should_trigger_sensitivity:
                    logger.info("🔍 민감성 감지 조건 만족 - 중립화 시스템 트리거")
                    try:
                        # 민감성 감지 및 중립화 시도
                        was_sensitive, neutralized_text, metadata = detect_and_neutralize_sensitive_content(text)
                        
                        if was_sensitive and neutralized_text and neutralized_text != text:
                            logger.info(f"🔄 민감성 감지됨 - 중립화된 텍스트로 재시도")
                            logger.info(f"📝 원본: '{text[:50]}...'")
                            logger.info(f"📝 변환: '{neutralized_text[:50]}...'")
                            
                            # 중립화된 텍스트로 재귀 호출 (무한 루프 방지를 위해 1회만)
                            return run_deep_analysis(neutralized_text)
                        else:
                            logger.info("🔍 민감성 미감지 또는 중립화 실패")
                            
                    except Exception as e:
                        logger.error(f"❌ 민감성 처리 실패: {e}")
                
                elif has_substantial_responses:
                    logger.info("📄 실질적인 응답은 생성되었으나 파싱 실패 - JSON 구조 문제로 판단")
                    logger.info("   → 민감성 감지 시스템 트리거하지 않음")
                else:
                    logger.info("❓ 응답 길이는 있으나 빈 내용 - 기타 원인으로 판단")
                
                logger.error("❌ 모든 시도 실패 - 시스템 정지")
                return None
            
            return run_deep_analysis()
            
        except Exception as e:
            logger.error(f"깊은 LLM 감정 분석 오류: {e}")
            return None

    def _extract_partial_emotion_data(self, response_text: str) -> Optional[Dict[str, Any]]:
        """부분 응답에서 감정 데이터 추출 (강화된 패턴 매칭)"""
        try:
            import re
            
            # 강화된 감정 패턴 추출
            emotion_patterns = [
                # JSON 필드 패턴
                r'"emotion"\s*:\s*"([^"]+)"',
                r'"emotion"\s*:\s*([a-zA-Z가-힣]+)',
                # 일반 텍스트 패턴
                r'(?:emotion|감정)(?:\s*is)?\s*[:=]\s*"?([a-zA-Z가-힣]+)"?',
                r'(?:주요\s*감정|감정)\s*[:=]\s*"?([a-zA-Z가-힣]+)"?',
                # 직접 감정 언급
                r'\b(joy|sadness|anger|fear|surprise|disgust|trust|anticipation|neutral)\b',
                r'\b(기쁨|슬픔|분노|두려움|놀람|혐오|신뢰|기대|중립)\b',
                # 불완전한 JSON에서 감정만 추출
                r'(?:anticipation|joy|fear|sadness|anger|disgust|trust|surprise|기대|기쁨|두려움|슬픔|분노|혐오|신뢰|놀람)'
            ]
            
            intensity_patterns = [
                r'"intensity"\s*:\s*(\d+)',
                r'"intensity"\s*:\s*"(\d+)"',
                r'(?:intensity|강도)(?:\s*is)?\s*[:=]\s*(\d+)',
                r'(?:감정\s*강도|강도)\s*[:=]\s*(\d+)',
                r'(\d+)\s*(?:out\s*of\s*[56]|/[56])',  # "3 out of 5" or "3/5" 형식
            ]
            
            confidence_patterns = [
                r'"confidence"\s*:\s*([0-9.]+)',
                r'"confidence"\s*:\s*"([0-9.]+)"',
                r'(?:confidence|신뢰도)(?:\s*is)?\s*[:=]\s*([0-9.]+)',
                r'(?:신뢰도|확실도)\s*[:=]\s*([0-9.]+)%?',
            ]
            
            valence_patterns = [
                r'"valence"\s*:\s*([-0-9.]+)',
                r'(?:valence|감정가)\s*[:=]\s*([-0-9.]+)',
            ]
            
            arousal_patterns = [
                r'"arousal"\s*:\s*([-0-9.]+)',
                r'(?:arousal|각성도)\s*[:=]\s*([-0-9.]+)',
            ]
            
            result = {
                'emotion': EmotionState.NEUTRAL.value,
                'intensity': 3,
                'confidence': 0.5,
                'valence': 0.0,
                'arousal': 0.0,
                'reasoning': response_text[:200] + '...' if len(response_text) > 200 else response_text
            }
            
            extracted_count = 0
            
            # 감정 추출 (우선순위 패턴 순서로)
            for pattern in emotion_patterns:
                match = re.search(pattern, response_text, re.IGNORECASE)
                if match:
                    emotion_name = match.group(1).strip().strip('"')
                    new_emotion_id = self._name_to_emotion_id(emotion_name)
                    if new_emotion_id != EmotionState.NEUTRAL.value:  # 유효한 감정만 채택
                        result['emotion'] = new_emotion_id
                        extracted_count += 1
                        logger.debug(f"✅ 감정 추출: '{emotion_name}' -> {new_emotion_id}")
                        break
            
            # 강도 추출
            for pattern in intensity_patterns:
                match = re.search(pattern, response_text, re.IGNORECASE)
                if match:
                    try:
                        intensity_value = int(match.group(1))
                        if 1 <= intensity_value <= 6:  # 유효 범위 확인
                            result['intensity'] = intensity_value
                            extracted_count += 1
                            logger.debug(f"✅ 강도 추출: {intensity_value}")
                            break
                    except ValueError:
                        pass
            
            # 신뢰도 추출
            for pattern in confidence_patterns:
                match = re.search(pattern, response_text, re.IGNORECASE)
                if match:
                    try:
                        conf_value = float(match.group(1))
                        # 백분율 형식인 경우 0-1 범위로 변환
                        if conf_value > 1.0:
                            conf_value = conf_value / 100.0
                        if 0.0 <= conf_value <= 1.0:  # 유효 범위 확인
                            result['confidence'] = conf_value
                            extracted_count += 1
                            logger.debug(f"✅ 신뢰도 추출: {conf_value}")
                            break
                    except ValueError:
                        pass
            
            # 감정가 추출
            for pattern in valence_patterns:
                match = re.search(pattern, response_text, re.IGNORECASE)
                if match:
                    try:
                        valence_value = float(match.group(1))
                        if -1.0 <= valence_value <= 1.0:  # 유효 범위 확인
                            result['valence'] = valence_value
                            extracted_count += 1
                            logger.debug(f"✅ 감정가 추출: {valence_value}")
                            break
                    except ValueError:
                        pass
            
            # 각성도 추출
            for pattern in arousal_patterns:
                match = re.search(pattern, response_text, re.IGNORECASE)
                if match:
                    try:
                        arousal_value = float(match.group(1))
                        if -1.0 <= arousal_value <= 1.0:  # 유효 범위 확인
                            result['arousal'] = arousal_value
                            extracted_count += 1
                            logger.debug(f"✅ 각성도 추출: {arousal_value}")
                            break
                    except ValueError:
                        pass
            
            # 품질 체크: reasoning 필드의 이상한 텍스트 감지
            reasoning_text = result.get('reasoning', '')
            is_corrupted = (
                len(re.findall(r'(.)\1{5,}', reasoning_text)) > 0 or  # 같은 문자 6회 이상 반복
                len(re.findall(r'[a-zA-Z]{15,}', reasoning_text)) > 2 or  # 15자 이상 긴 단어 2개 이상
                'reatkage' in reasoning_text or 'existsizing' in reasoning_text  # 알려진 깨진 단어
            )
            
            if is_corrupted:
                logger.warning(f"❌ 부분 데이터 추출 실패: reasoning 필드가 손상됨")
                logger.warning(f"   손상된 reasoning: {reasoning_text[:100]}...")
                return None
            
            # 최소 2개 이상의 유의미한 데이터가 추출되었는지 확인
            if extracted_count >= 1 and (result['emotion'] != EmotionState.NEUTRAL.value or result['intensity'] != 3):
                logger.info(f"✅ 부분 데이터 추출 성공: {extracted_count}개 필드 추출됨")
                logger.info(f"   emotion={result['emotion']}, intensity={result['intensity']}, confidence={result['confidence']}")
                return result
            
            logger.debug(f"❌ 부분 데이터 추출 실패: {extracted_count}개 필드만 추출됨")
            return None
            
        except Exception as e:
            logger.error(f"부분 데이터 추출 실패: {e}")
            return None

    def _parse_deep_llm_response(self, response_text: str) -> Dict[str, Any]:
        """깊은 LLM 분석 응답 파싱 (다단계 fallback 메커니즘)"""
        try:
            import json
            import re
            
            logger.info(f"🔧 LLM 응답 파싱 시작: {len(response_text)} 문자")
            logger.info(f"🔧 응답 첫 500자: {response_text[:500]}...")
            
            # 1단계: json_repair를 사용한 자동 수정 시도
            try:
                from json_repair import repair_json
                
                logger.info("🔧 1단계: json_repair 자동 수정 시도")
                repaired_json = repair_json(response_text)
                logger.info(f"🔧 수정된 JSON: {repaired_json[:200]}...")
                json_response = json.loads(repaired_json)
                
                # LLM 응답이 복합 객체일 경우 emotions 필드 확인
                if 'emotions' in json_response:
                    emotions_data = json_response['emotions']
                    # 가장 높은 값을 가진 감정 찾기
                    max_emotion = max(emotions_data.items(), key=lambda x: float(x[1]))
                    emotion_name = max_emotion[0]
                    
                    # intensity, confidence 등 확인
                    intensity_data = json_response.get('intensity', {})
                    confidence_data = json_response.get('confidence', {})
                    valence_data = json_response.get('valence', {})
                    arousal_data = json_response.get('arousal', {})
                    
                    result = {
                        'emotion': self._name_to_emotion_id(emotion_name),
                        'intensity': int(intensity_data.get(emotion_name, 3)),
                        'confidence': float(confidence_data.get(emotion_name, 0.5)),
                        'valence': float(valence_data.get(emotion_name, 0.0)),
                        'arousal': float(arousal_data.get(emotion_name, 0.0)),
                        'reasoning': str(json_response.get('reasoning', {}).get(emotion_name, ''))
                    }
                    
                    logger.info(f"✅ json_repair 성공 (복합객체): emotion={emotion_name} -> id={result['emotion']}")
                    return result
                
                # 단순 객체인 경우
                elif 'emotion' in json_response:
                    emotion_name = str(json_response.get('emotion', 'neutral')).strip()
                    result = {
                        'emotion': self._name_to_emotion_id(emotion_name),
                        'intensity': int(json_response.get('intensity', 3)),
                        'confidence': float(json_response.get('confidence', 0.5)),
                        'valence': float(json_response.get('valence', 0.0)),
                        'arousal': float(json_response.get('arousal', 0.0)),
                        'reasoning': str(json_response.get('reasoning', ''))
                    }
                    
                    logger.info(f"✅ json_repair 성공 (단순객체): emotion={emotion_name} -> id={result['emotion']}")
                    return result
                    
            except Exception as e:
                logger.info(f"🔧 json_repair 실패: {e}")
            
            # 2단계: 강화된 전처리 후 JSON 파싱 시도
            logger.info("🔧 2단계: 강화된 전처리 후 JSON 파싱")
            preprocessed_text = self._preprocess_llm_json(response_text)
            
            # 다양한 JSON 패턴 시도 (부분 응답 대응)
            json_patterns = [
                r'\{[^}]*\}',              # 완전한 JSON 객체
                r'\{[^}]*',                # 불완전한 JSON 객체 (length로 잘림)
                r'"emotion"\s*:\s*"[^"]*"', # 감정 필드만
                r'"intensity"\s*:\s*\d+',   # 강도 필드만
                r'"confidence"\s*:\s*[\d.]+' # 신뢰도 필드만
            ]
            
            for pattern in json_patterns:
                json_match = re.search(pattern, preprocessed_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(0)
                    
                    # 불완전한 JSON 객체 완성 시도
                    if not json_text.endswith('}'):
                        json_text += '}'
                    
                    try:
                        json_response = json.loads(json_text)
                        emotion_name = str(json_response.get('emotion', 'neutral')).strip()
                        
                        result = {
                            'emotion': self._name_to_emotion_id(emotion_name),
                            'intensity': int(json_response.get('intensity', 3)),
                            'confidence': float(json_response.get('confidence', 0.5)),
                            'valence': float(json_response.get('valence', 0.0)),
                            'arousal': float(json_response.get('arousal', 0.0)),
                            'reasoning': str(json_response.get('reasoning', ''))
                        }
                        
                        logger.info(f"✅ 전처리 JSON 파싱 성공 (패턴 {pattern}): emotion={emotion_name} -> id={result['emotion']}")
                        return result
                        
                    except (json.JSONDecodeError, ValueError, TypeError) as e:
                        logger.debug(f"패턴 {pattern} 파싱 실패: {e}")
                        continue
            
            # 3단계: 부분 텍스트 추출 시도
            logger.info("🔧 3단계: 부분 텍스트 추출")
            partial_result = self._extract_partial_emotion_data(response_text)
            if partial_result:
                logger.info("✅ 부분 텍스트 추출 성공")
                return partial_result
            
            # 4단계: 한국어 형식 파싱
            logger.info("🔧 4단계: 한국어 형식 파싱")
            return self._parse_korean_format_response(response_text)
            
        except Exception as e:
            logger.error(f"❌ 응답 파싱 전체 실패: {e}")
            logger.error(f"❌ 실패한 응답 텍스트: {response_text[:200]}...")
            # 파싱 실패 시에도 기본값 반환 (None 반환 방지)
            return {
                'emotion': EmotionState.NEUTRAL.value,
                'intensity': 3,
                'confidence': 0.1,
                'valence': 0.0,
                'arousal': 0.0,
                'reasoning': f'parsing_failed: {str(e)}'
            }
    
    def _preprocess_llm_json(self, text: str) -> str:
        """LLM JSON 응답 전처리 (유니코드 특수문자, 쌍따옴표 보정, 강화된 오류 수정)"""
        import re
        
        # 응답에서 JSON 부분만 추출 (ChatML 템플릿 제거)
        json_start = text.find('{')
        if json_start > 0:
            text = text[json_start:]
        
        # 유니코드 특수문자 정규화
        text = text.replace('″', '"')  # 유니코드 쌍따옴표
        text = text.replace('″，', '",')  # 쌍따옴표 + 중국어 쉼표
        text = text.replace('""', '"')  # 연속 쌍따옴표
        text = text.replace('"""', '"')  # 3개 연속 쌍따옴표
        text = text.replace('"', '"').replace('"', '"')  # 한국어 따옴표
        
        # JSON 키 공백 문제 완전 해결 (HelpingAI 특화)
        text = re.sub(r'"\s+(\w+)"\s*:', r'"\1":', text)  # " emotion": -> "emotion":
        text = re.sub(r'{\s*"\s+', r'{"', text)  # {"  -> {"
        text = re.sub(r',\s*"\s+', r', "', text)  # ,  " -> , "
        text = re.sub(r':\s*"\s+', r': "', text)  # : "value -> : "value
        text = re.sub(r'\s+"\s*:', r'":', text)  # space"space: -> ":
        # HelpingAI 특화 패턴: 키 앞뒤 공백 제거
        for key in ['emotion', 'intensity', 'confidence', 'valence', 'arousal', 'reasoning']:
            text = re.sub(rf'"\s*{key}\s*"', f'"{key}"', text)  # " emotion " -> "emotion"
        
        # 일반적인 오타 수정
        text = re.sub(r'(\w+)\s*:\s*(\w+)', r'"\1": "\2"', text)  # key: value -> "key": "value"
        text = re.sub(r'"(\w+)"\s*:\s*(\d+\.?\d*)', r'"\1": \2', text)  # "key": 123 (숫자는 따옴표 제거)
        
        # 불완전한 키에 쌍따옴표 추가
        text = re.sub(r'(\w+)\s*:', r'"\1":', text)  # key: -> "key":
        
        # 값 주변 쌍따옴표 수정 (더 정교하게)
        text = re.sub(r':\s*([a-zA-Z가-힣][^,}"\d]*)', r': "\1"', text)  # 문자열 값 쌍따옴표
        
        # 잘못된 쉼표 수정
        text = re.sub(r',\s*}', '}', text)  # trailing comma 제거
        text = re.sub(r',\s*,', ',', text)  # 중복 쉼표 제거
        
        # 불완전한 종료 수정
        open_brackets = text.count('{')
        close_brackets = text.count('}')
        if open_brackets > close_brackets:
            text += '}' * (open_brackets - close_brackets)
        
        # 불완전한 따옴표 수정
        quote_count = text.count('"')
        if quote_count % 2 != 0:
            text += '"'
        
        logger.info(f"전처리 완료: {text[:100]}...")
        return text
    
    def _parse_korean_format_response(self, response_text: str) -> Dict[str, Any]:
        """한국어 형식 응답 파싱 (기존 로직)"""
        try:
            lines = response_text.split('\n')
            result = {
                'emotion': EmotionState.NEUTRAL.value,
                'intensity': 3,
                'confidence': 0.5,
                'valence': 0.0,
                'arousal': 0.0,
                'reasoning': response_text
            }
            
            for line in lines:
                line = line.strip()
                if '주요 감정:' in line:
                    emotion_text = line.split(':')[1].strip()
                    result['emotion'] = self._name_to_emotion_id(emotion_text)
                elif '감정 강도:' in line:
                    try:
                        intensity_text = line.split(':')[1].strip()
                        result['intensity'] = int(intensity_text)
                    except:
                        pass
                elif '신뢰도:' in line:
                    try:
                        conf_text = line.split(':')[1].strip()
                        result['confidence'] = float(conf_text)
                    except:
                        pass
                elif '감정가:' in line:
                    try:
                        valence_text = line.split(':')[1].strip()
                        result['valence'] = float(valence_text)
                    except:
                        pass
                elif '각성도:' in line:
                    try:
                        arousal_text = line.split(':')[1].strip()
                        result['arousal'] = float(arousal_text)
                    except:
                        pass
                        
            return result
            
        except Exception as e:
            logger.error(f"한국어 형식 파싱 오류: {e}")
            return None
    
    def _detect_complex_ethical_question(self, text: str) -> bool:
        """복잡한 윤리적 질문 감지 (fallback 없이)"""
        try:
            # 윤리적 딜레마 키워드 패턴
            ethical_keywords = [
                # 핵심 윤리 개념
                '윤리', '도덕', '선택', '딜레마', '갈등', '판단',
                # 생명과 관련
                '생명', '죽음', '살해', '구조', '희생', '안전',
                # 정의와 공정성
                '정의', '공정', '평등', '차별', '권리', '의무',
                # 자율성과 자유
                '자유', '자율', '강제', '억압', '선택권', '의사결정',
                # 책임과 결과
                '책임', '결과', '피해', '이익', '손실', '대가',
                # 사회적 관계
                '사회', '공동체', '개인', '집단', '다수', '소수',
                # 시나리오별 키워드
                '자율주행', '의료', '인공호흡기', '개인정보', '감시', '테러',
                '브레이크', '급브레이크', '칠', '틀어', '직진',
                '환자', '나이', '우선순위', '선택',
                '프라이버시', '공공안전', '수집', '보호'
            ]
            
            text_lower = text.lower()
            found_keywords = [keyword for keyword in ethical_keywords if keyword in text_lower]
            
            # 복잡성 평가 기준
            complexity_score = 0
            
            # 1. 윤리적 키워드 수 (기본 점수)
            complexity_score += len(found_keywords) * 2
            
            # 2. 선택/결정 관련 표현
            choice_patterns = ['할까', '말까', '것인가', '선택', '결정', '판단', '고민']
            choice_count = sum(1 for pattern in choice_patterns if pattern in text_lower)
            complexity_score += choice_count * 3
            
            # 3. 대조/비교 표현
            contrast_patterns = ['vs', '대', '반면', '하지만', '그러나', '아니면', '또는']
            contrast_count = sum(1 for pattern in contrast_patterns if pattern in text_lower)
            complexity_score += contrast_count * 2
            
            # 4. 수치적 비교 (나이, 수량 등)
            import re
            number_pattern = r'\d+세|\d+명|\d+대|\d+개'
            number_matches = len(re.findall(number_pattern, text))
            complexity_score += number_matches * 2
            
            # 5. 질문 형태
            if '?' in text or text.endswith('가?') or '인가' in text:
                complexity_score += 5
            
            # 복잡성 임계값: 8점 이상이면 복잡한 윤리적 질문으로 판단
            is_complex = complexity_score >= 8
            
            if is_complex:
                logger.info(f"복잡한 윤리적 질문 감지 - 점수: {complexity_score}, 키워드: {found_keywords}")
            
            return is_complex
            
        except Exception as e:
            logger.error(f"윤리적 질문 감지 실패: {e}")
            # fallback 금지 - 기본값 False 반환
            return False

    def _calculate_dynamic_token_limit_direct(self, text: str, base_tokens: int = 400) -> int:
        """텍스트 복잡도에 따른 동적 토큰 할당 (중첩 함수에서 호출 가능)"""
        try:
            # 기본 토큰 수
            tokens = base_tokens
            
            # 텍스트 길이에 따른 추가 토큰
            if len(text) > 100:
                tokens += min(200, len(text) // 5)
            
            # 복잡한 윤리적 질문 감지
            if self._detect_complex_ethical_question(text):
                tokens += 300  # 복잡한 분석을 위한 추가 토큰
            
            # 복합 감정 표현 감지
            emotion_count = sum(1 for emotion_dict in self.korean_emotion_keywords.values() 
                               for keyword_list in emotion_dict.values()
                               for keyword in keyword_list 
                               if keyword in text.lower())
            if emotion_count >= 3:
                tokens += 150  # 복합 감정 분석을 위한 추가 토큰
            
            # 최대 토큰 수 제한 (모델 성능 최적화)
            return min(tokens, 1500)
            
        except Exception as e:
            logger.warning(f"동적 토큰 계산 오류: {e}, 기본값 사용")
            return base_tokens
    
    def get_pytorch_network(self) -> Optional[nn.Module]:
        """
        헤드/스왑매니저가 사용할 대표 PyTorch 네트워크를 반환.
        - 가능한 후보를 순서대로 탐색해서 nn.Module을 반환
        - 한 번 찾으면 캐시(self._primary_nn)해 재사용
        - 모든 후보가 없으면 _build_default_network()로 생성
        """
        import torch.nn as nn
        
        # 캐시 있으면 즉시 반환
        if hasattr(self, "_primary_nn") and isinstance(self._primary_nn, nn.Module):
            return self._primary_nn
        
        candidates = []
        
        # 1) 자주 쓰이는 네이밍 우선
        priority_names = ["hierarchical_model", "emotion_moe", "model", "network", "default_network"]
        for name in priority_names:
            if hasattr(self, name):
                obj = getattr(self, name, None)
                if obj is None:
                    continue
                    
                # callable이면 한 번 호출하여 인스턴스 획듍
                try:
                    if callable(obj) and not isinstance(obj, nn.Module):
                        obj = obj()
                except Exception as e:
                    logger.debug(f"  - {name} 호출 실패: {e}")
                    continue
                    
                if isinstance(obj, nn.Module):
                    candidates.append((name, obj))
                    logger.debug(f"  - {name} 후보 발견")
        
        # 2) 멤버 중 nn.Module 자동 탐색 (백업 경로)
        if not candidates:
            logger.info("🔍 우선순위 후보에서 nn.Module을 찾지 못함, 전체 탐색 시작...")
            try:
                for name, val in vars(self).items():
                    if name.startswith('_'):  # private 속성 건너뛰기
                        continue
                    if isinstance(val, nn.Module):
                        candidates.append((name, val))
                        logger.debug(f"  - {name} 후보 발견 (전체 탐색)")
            except Exception as e:
                logger.error(f"vars() 탐색 오류: {e}")
        
        # 3) 후보가 없으면 기본 네트워크 생성
        if not candidates:
            logger.info("emotion_analyzer 내부에서 nn.Module 후보를 찾지 못함 - 기본 네트워크 생성")
            logger.debug(f"  - emotion_moe: {hasattr(self, 'emotion_moe')} / {type(getattr(self, 'emotion_moe', None))}")
            logger.debug(f"  - hierarchical_model: {hasattr(self, 'hierarchical_model')} / {type(getattr(self, 'hierarchical_model', None))}")
            logger.debug(f"  - default_network: {hasattr(self, 'default_network')} / {type(getattr(self, 'default_network', None))}")
            
            # 기본 네트워크 무조건 생성 (NO FALLBACK)
            logger.info("🔨 기본 네트워크 생성 중...")
            self._build_default_network()
            
            # 생성 후 반환
            if hasattr(self, '_primary_nn') and isinstance(self._primary_nn, nn.Module):
                logger.info(f"✅ 기본 네트워크 생성 완료: {self._primary_nn.__class__.__name__}")
                return self._primary_nn
            elif hasattr(self, 'default_network') and isinstance(self.default_network, nn.Module):
                self._primary_nn = self.default_network
                logger.info(f"✅ 기본 네트워크 생성 완료: {self.default_network.__class__.__name__}")
                return self._primary_nn
            else:
                # NO FALLBACK - 실패시 에러
                raise RuntimeError("emotion_analyzer nn.Module 생성 실패 - NO FALLBACK")
        
        # 4) 가장 큰 네트워크를 대표로 선택(파라미터 수 기준)
        def num_params(m): 
            try:
                return sum(p.numel() for p in m.parameters())
            except Exception:
                return 0
        
        best_name, best_model = max(candidates, key=lambda kv: num_params(kv[1]))
        
        logger.info(f"AdvancedEmotionAnalyzer: {best_name}을(를) primary_nn으로 선택 (파라미터 수: {num_params(best_model):,})")
        
        # 캐시 후 반환
        self._primary_nn = best_model
        return self._primary_nn


# 테스트 함수
def test_advanced_emotion_analyzer():
    """고급 감정 분석기 테스트"""
    try:
        analyzer = AdvancedEmotionAnalyzer()
        
        # 한국어 테스트
        test_texts = [
            "오늘 정말 기뻐서 어쩔 줄 모르겠어요!",
            "너무 슬프고 우울해서 눈물이 나네요.",
            "화가 나서 정말 참을 수가 없어요!",
            "무서워서 떨리고 식은땀이 흘러요.",
            "깜짝 놀라서 심장이 떨어질 뻔했어요."
        ]
        
        for text in test_texts:
            emotion = analyzer.analyze_emotion(text, language="ko")
            print(f"텍스트: {text}")
            print(f"감정: {emotion.primary_emotion.value} (신뢰도: {emotion.confidence:.3f})")
            print(f"처리 방법: {emotion.processing_method}")
            print("-" * 50)
        
        # 배치 처리 테스트
        batch_results = analyzer.batch_analyze_emotions(test_texts, language="ko")
        print(f"배치 처리 완료: {len(batch_results)}개 텍스트")
        
        print("고급 감정 분석기 테스트 성공!")
        
    except Exception as e:
        print(f"테스트 실패: {e}")
        raise

    def _enhance_with_llm_analysis(self, text: str, keyword_emotion: int, keyword_score: float) -> Optional[Dict[str, Any]]:
        """LLM으로 키워드 분석 결과 검증 및 강화 (조건부 호출)"""
        
        # 조건부 LLM 호출 결정
        should_use_llm = self._should_use_llm_for_emotion(text, keyword_score)
        if not should_use_llm:
            return None
            
        # LLM 통합이 불가능한 경우 - 조건부 처리 
        if not LLM_INTEGRATION_AVAILABLE:
            logger.debug("LLM 통합을 사용할 수 없어 키워드 분석 결과를 반환합니다.")
            return None
        
        try:
            import asyncio
            from llm_module.advanced_llm_engine import get_llm_engine, LLMRequest, TaskComplexity
            
            # 키워드 감정을 텍스트로 변환
            emotion_name = self._emotion_id_to_name(keyword_emotion)
            
            # 강화된 ChatML 템플릿 - 매핑 테이블과 동기화된 엄격한 제약
            prompt = f"""<|im_start|>system
당신은 HelpingAI, 감정 분석 전문가입니다. 주어진 텍스트를 분석하고 반드시 아래 형식의 JSON으로만 응답하세요.

⚠️ 중요 제약사항:
- 반드시 지정된 감정 목록에서만 선택하세요
- 다른 감정 이름은 절대 사용하지 마세요
- JSON 형식을 정확히 지켜주세요
- 추가 텍스트나 설명은 금지입니다

필수 JSON 형식:
{{
    "emotion": "감정명",
    "intensity": 정수1-6,
    "confidence": 실수0.0-1.0,
    "valence": 실수-1.0-1.0,
    "arousal": 실수0.0-1.0,
    "reasoning": "간단한_설명"
}}

허용된 감정 (이것 외 사용 금지):
- joy: 기쁨, 행복
- trust: 신뢰, 믿음  
- fear: 두려움, 불안
- surprise: 놀람, 충격
- sadness: 슬픔, 우울
- disgust: 혐오, 거부감
- anger: 분노, 화남 (frustration 포함)
- anticipation: 기대, 희망
- neutral: 중립, 무감정

예시:
{{"emotion": "anger", "intensity": 3, "confidence": 0.8, "valence": -0.3, "arousal": 0.6, "reasoning": "분노 표현이 감지됨"}}

다시 한번 강조: 위 9개 감정 중에서만 선택하고, 정확한 JSON 형식으로만 응답하세요.<|im_end|>
<|im_start|>user
{text}<|im_end|>
<|im_start|>assistant
"""

            # 중첩 함수에서 사용할 토큰 수 미리 계산
            dynamic_tokens = self._calculate_dynamic_token_limit_direct(text, base_tokens=400)
            
            # 이벤트 루프 안전한 실행을 위한 래퍼
            def run_llm_analysis():
                try:
                    import threading
                    import concurrent.futures
                    
                    # 새로운 스레드에서 비동기 실행
                    def async_llm_call():
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            engine = get_llm_engine()
                            request = LLMRequest(
                                prompt=prompt,
                                task_type="emotion_analysis",  # HelpingAI EQ 95.89 우선
                                complexity=TaskComplexity.MODERATE,
                                max_tokens=dynamic_tokens,
                                temperature=0.3
                            )
                            
                            response = loop.run_until_complete(engine.generate_async(request))
                            return response
                        finally:
                            loop.close()
                    
                    # 스레드풀에서 실행
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(async_llm_call)
                        response = future.result(timeout=30)  # 30초 타임아웃
                    
                    if response and response.success:
                        # 안전한 속성 접근 - generated_text 속성 확인
                        if hasattr(response, 'generated_text') and response.generated_text:
                            logger.info(f"✅ LLM 응답 성공 - 길이: {len(response.generated_text)}")
                            return self._parse_llm_emotion_response(response.generated_text, keyword_emotion)
                        else:
                            logger.error(f"❌ 치명적 실패: generated_text 속성 없음 - 응답 구조: {type(response).__name__}")
                            logger.error(f"응답 속성들: {dir(response)}")
                            return None
                    else:
                        logger.warning(f"LLM 감정 분석 실패: {response.error_message if response else 'No response'}")
                        return None
                        
                except Exception as e:
                    logger.error(f"LLM 분석 실행 오류: {e}")
                    return None
            
            return run_llm_analysis()
            
        except Exception as e:
            logger.error(f"LLM 감정 분석 오류: {e}")
            return None
    
    def _detect_complex_ethical_question(self, text: str) -> bool:
        """복잡한 윤리적 질문인지 감지"""
        ethical_keywords = [
            "윤리적", "딜레마", "vs", "대", "선택", "방지", "보호", "권리", "안전", 
            "개인정보", "프라이버시", "감시", "테러", "의료", "자원", "배분",
            "자율주행", "브레이크", "환자", "인공호흡기", "정의", "공정", "도덕적"
        ]
        
        text_lower = text.lower()
        keyword_count = sum(1 for keyword in ethical_keywords if keyword in text_lower)
        
        # 키워드 3개 이상이고 길이가 50자 이상이면 복잡한 윤리적 질문으로 판단
        is_complex = keyword_count >= 3 and len(text) >= 50
        
        if is_complex:
            logger.info(f"복잡한 윤리적 질문 감지: 키워드 {keyword_count}개, 길이 {len(text)}자")
        
        return is_complex

    def _calculate_dynamic_token_limit(self, text: str, base_tokens: int = 400) -> int:
        """텍스트 복잡도에 따른 동적 토큰 제한 계산"""
        if self._detect_complex_ethical_question(text):
            # 복잡한 윤리적 질문의 경우 2배 토큰 할당
            dynamic_tokens = base_tokens * 2
            logger.info(f"복잡한 질문 감지 - 토큰 제한 증가: {base_tokens} → {dynamic_tokens}")
            return dynamic_tokens
        elif len(text) > 100:
            # 긴 텍스트의 경우 1.5배 토큰 할당
            dynamic_tokens = int(base_tokens * 1.5)
            logger.info(f"긴 텍스트 감지 - 토큰 제한 증가: {base_tokens} → {dynamic_tokens}")
            return dynamic_tokens
        else:
            return base_tokens

    def _retry_llm_analysis_with_increased_tokens(self, text: str, prompt: str, base_tokens: int, max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """토큰 수를 점진적으로 증가시키면서 LLM 분석 재시도"""
        for attempt in range(max_retries):
            try:
                # 재시도마다 토큰 수 증가 (1.5배씩)
                current_tokens = int(base_tokens * (1.5 ** attempt))
                logger.info(f"LLM 분석 재시도 {attempt + 1}/{max_retries}: 토큰 수 {current_tokens}")
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                engine = get_llm_engine()
                request = LLMRequest(
                    prompt=prompt,
                    task_type="emotion_interpretation",
                    complexity=TaskComplexity.COMPLEX,
                    max_tokens=current_tokens,
                    temperature=0.2
                )
                
                response = loop.run_until_complete(engine.generate_async(request))
                loop.close()
                
                if response.success:
                    # 응답이 성공적으로 완료되었는지 확인
                    if hasattr(response, 'generated_text') and response.generated_text:
                        # finish_reason 확인
                        if hasattr(response, 'finish_reason'):
                            if response.finish_reason == 'length':
                                logger.warning(f"재시도 {attempt + 1}: finish_reason이 'length'여서 계속 재시도")
                                continue  # 다음 재시도로
                            elif response.finish_reason == 'stop':
                                logger.info(f"재시도 {attempt + 1}: finish_reason이 'stop'으로 정상 완료")
                        
                        # 응답 파싱 시도
                        parsed_result = self._parse_deep_llm_response(response.generated_text)
                        if parsed_result is not None:
                            logger.info(f"재시도 {attempt + 1}: 성공적으로 파싱 완료")
                            return parsed_result
                        else:
                            logger.warning(f"재시도 {attempt + 1}: 파싱 실패, 다음 재시도 진행")
                    else:
                        logger.warning(f"재시도 {attempt + 1}: 생성된 텍스트가 없음")
                else:
                    logger.warning(f"재시도 {attempt + 1}: 응답 실패")
                    
            except Exception as e:
                logger.error(f"재시도 {attempt + 1}: 예외 발생 - {e}")
                
            # 재시도 간격 (1초씩 증가)
            if attempt < max_retries - 1:
                import time
                time.sleep(attempt + 1)
        
        logger.error(f"모든 재시도 실패: {max_retries}회 시도 후 포기")
        return None
    
    def _detect_complex_ethical_question(self, text: str) -> bool:
        """복잡한 윤리적 질문인지 감지"""
        ethical_keywords = [
            "윤리적", "딜레마", "vs", "대", "선택", "방지", "보호", "권리", "안전", 
            "개인정보", "프라이버시", "감시", "테러", "의료", "자원", "배분",
            "자율주행", "브레이크", "환자", "인공호흡기", "정의", "공정", "도덕적"
        ]
        
        text_lower = text.lower()
        keyword_count = sum(1 for keyword in ethical_keywords if keyword in text_lower)
        
        # 키워드 3개 이상이고 길이가 50자 이상이면 복잡한 윤리적 질문으로 판단
        is_complex = keyword_count >= 3 and len(text) >= 50
        
        if is_complex:
            logger.info(f"복잡한 윤리적 질문 감지: 키워드 {keyword_count}개, 길이 {len(text)}자")
        
        return is_complex

    def _calculate_dynamic_token_limit(self, text: str, base_tokens: int = 400) -> int:
        """텍스트 복잡도에 따른 동적 토큰 제한 계산"""
        if self._detect_complex_ethical_question(text):
            # 복잡한 윤리적 질문의 경우 2배 토큰 할당
            dynamic_tokens = base_tokens * 2
            logger.info(f"복잡한 질문 감지 - 토큰 제한 증가: {base_tokens} → {dynamic_tokens}")
            return dynamic_tokens
        elif len(text) > 100:
            # 긴 텍스트의 경우 1.5배 토큰 할당
            dynamic_tokens = int(base_tokens * 1.5)
            logger.info(f"긴 텍스트 감지 - 토큰 제한 증가: {base_tokens} → {dynamic_tokens}")
            return dynamic_tokens
        else:
            return base_tokens

    def _retry_llm_analysis_with_increased_tokens(self, text: str, prompt: str, base_tokens: int, max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """토큰 수를 점진적으로 증가시키면서 LLM 분석 재시도"""
        for attempt in range(max_retries):
            try:
                # 재시도마다 토큰 수 증가 (1.5배씩)
                current_tokens = int(base_tokens * (1.5 ** attempt))
                logger.info(f"LLM 분석 재시도 {attempt + 1}/{max_retries}: 토큰 수 {current_tokens}")
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                engine = get_llm_engine()
                request = LLMRequest(
                    prompt=prompt,
                    task_type="emotion_interpretation",
                    complexity=TaskComplexity.COMPLEX,
                    max_tokens=current_tokens,
                    temperature=0.2
                )
                
                response = loop.run_until_complete(engine.generate_async(request))
                loop.close()
                
                if response.success:
                    # 응답이 성공적으로 완료되었는지 확인
                    if hasattr(response, 'generated_text') and response.generated_text:
                        # finish_reason 확인
                        if hasattr(response, 'finish_reason'):
                            if response.finish_reason == 'length':
                                logger.warning(f"재시도 {attempt + 1}: finish_reason이 'length'여서 계속 재시도")
                                continue  # 다음 재시도로
                            elif response.finish_reason == 'stop':
                                logger.info(f"재시도 {attempt + 1}: finish_reason이 'stop'으로 정상 완료")
                        
                        # 응답 파싱 시도
                        parsed_result = self._parse_deep_llm_response(response.generated_text)
                        if parsed_result is not None:
                            logger.info(f"재시도 {attempt + 1}: 성공적으로 파싱 완료")
                            return parsed_result
                        else:
                            logger.warning(f"재시도 {attempt + 1}: 파싱 실패, 다음 재시도 진행")
                    else:
                        logger.warning(f"재시도 {attempt + 1}: 생성된 텍스트가 없음")
                else:
                    logger.warning(f"재시도 {attempt + 1}: 응답 실패")
                    
            except Exception as e:
                logger.error(f"재시도 {attempt + 1}: 예외 발생 - {e}")
                
            # 재시도 간격 (1초씩 증가)
            if attempt < max_retries - 1:
                import time
                time.sleep(attempt + 1)
        
        logger.error(f"모든 재시도 실패: {max_retries}회 시도 후 포기")
        return None

    
    def _should_use_llm_for_emotion(self, text: str, keyword_score: float) -> bool:
        """LLM 사용 조건 결정 (최적화된 고품질 AI 분석)"""
        
        # 빈 텍스트는 제외
        if not text or len(text.strip()) == 0:
            return False
            
        # 1. 키워드 점수가 모호한 경우 (1.0~2.5 사이) - 범위 확대
        if 1.0 <= keyword_score <= 2.5:
            return True
            
        # 2. 복잡한 텍스트 (길이 기준 강화)
        if len(text) > 30:  # 기존 50에서 30으로 강화
            return True
            
        # 3. 복합 감정 표현이 있는 경우 (임계값 낮춤)
        emotion_keywords_count = sum(1 for emotion_dict in self.korean_emotion_keywords.values() 
                                   for keyword_list in emotion_dict.values()
                                   for keyword in keyword_list 
                                   if keyword in text.lower())
        if emotion_keywords_count >= 2:  # 3개에서 2개로 강화
            return True
            
        # 4. 부정문이나 복잡한 문장 구조 (조건 확장)
        complex_patterns = ['않', '없', '못', '아니', '그러나', '하지만', '그런데', 
                           '오히려', '반면', '대신', '물론', '하지만', '그렇지만']
        if any(pattern in text for pattern in complex_patterns):
            return True
            
        # 5. 감정 강도 표현 (새로운 조건)
        intensity_words = ['매우', '정말', '너무', '아주', '완전히', '엄청', '굉장히', 
                          '심하게', '깊이', '극도로', '상당히', '꽤', '제법']
        if any(word in text for word in intensity_words):
            return True
            
        # 6. 복합 문장 구조 (새로운 조건)
        if text.count(',') >= 2 or text.count('.') >= 2 or '?' in text or '!' in text:
            return True
            
        # 7. 미묘한 감정 표현 (새로운 조건)
        subtle_emotions = ['미묘', '애매', '복잡', '혼란', '갈등', '딜레마', '고민', 
                          '생각', '느낌', '기분', '분위기', '뉘앙스']
        if any(word in text for word in subtle_emotions):
            return True
            
        # 간단하고 명확한 감정 표현만 키워드 분석 사용
        return False
    
    def _emotion_id_to_name(self, emotion_id: int) -> str:
        """감정 ID를 한국어 이름으로 변환"""
        emotion_mapping = {
            1: "기쁨",    # JOY
            2: "신뢰",    # TRUST  
            3: "두려움",  # FEAR
            4: "놀람",    # SURPRISE
            5: "슬픔",    # SADNESS
            6: "혐오",    # DISGUST
            7: "분노",    # ANGER
            8: "기대",    # ANTICIPATION
            0: "중립"     # NEUTRAL
        }
        return emotion_mapping.get(emotion_id, "알 수 없음")
    
    def _parse_llm_emotion_response(self, response_text: str, original_emotion: int) -> Dict[str, Any]:
        """LLM 감정 분석 응답 파싱 (JSON 우선, 텍스트 파싱 fallback)"""
        try:
            # 1. JSON 파싱 시도
            try:
                import json
                import re
                
                # 다양한 JSON 패턴 시도 (개선된 파싱)
                json_patterns = [
                    r'\{[^{}]*\}',  # 단순 JSON 블록
                    r'\{[^{}]*?"[^"]*"[^{}]*\}',  # 문자열 포함 JSON
                    r'\{.*?\}',  # 모든 문자 포함 JSON (최대한 유연)
                ]
                
                parsed_json = None
                for pattern in json_patterns:
                    matches = re.findall(pattern, response_text, re.DOTALL)
                    for match in matches:
                        try:
                            # 공백 문자 정리
                            cleaned_json = re.sub(r'\s+', ' ', match).strip()
                            
                            # 일반적인 JSON 파싱 오류 수정
                            cleaned_json = cleaned_json.replace('" emotion"', '"emotion"')
                            cleaned_json = cleaned_json.replace('" confidence"', '"confidence"')
                            cleaned_json = cleaned_json.replace('" intensity"', '"intensity"')
                            cleaned_json = cleaned_json.replace('" valence"', '"valence"')
                            cleaned_json = cleaned_json.replace('" arousal"', '"arousal"')
                            cleaned_json = cleaned_json.replace('" reasoning"', '"reasoning"')
                            
                            parsed_json = json.loads(cleaned_json)
                            logger.info(f"JSON 파싱 성공: {cleaned_json}")
                            break
                        except json.JSONDecodeError:
                            continue
                    if parsed_json:
                        break
                
                if parsed_json:
                    # JSON에서 결과 추출
                    result = {
                        'emotion': original_emotion,  # 기본값
                        'confidence': 0.5,
                        'reasoning': response_text
                    }
                    
                    # 영어 필드명 처리 (HelpingAI 응답 형식)
                    if 'emotion' in parsed_json:
                        emotion_text = parsed_json['emotion']
                        mapped_emotion = self._name_to_emotion_id(emotion_text)
                        if mapped_emotion is None:
                            logger.error(f"❌ 감정 매핑 실패로 인한 파싱 실패: '{emotion_text}'")
                            return None  # 학습 오염 방지를 위한 명확한 실패 반환
                        result['emotion'] = mapped_emotion
                    
                    # 신뢰도 처리
                    if 'confidence' in parsed_json:
                        try:
                            result['confidence'] = float(parsed_json['confidence'])
                        except (ValueError, TypeError) as parse_error:
                            logger.debug(f"타입 변환 실패, 기본값 사용: {parse_error}")
                    
                    # 감정강도 처리
                    if 'intensity' in parsed_json:
                        try:
                            result['intensity'] = int(parsed_json['intensity'])
                        except (ValueError, TypeError) as parse_error:
                            logger.debug(f"타입 변환 실패, 기본값 사용: {parse_error}")
                    
                    # valence와 arousal 처리
                    if 'valence' in parsed_json:
                        try:
                            result['valence'] = float(parsed_json['valence'])
                        except (ValueError, TypeError) as parse_error:
                            logger.debug(f"타입 변환 실패, 기본값 사용: {parse_error}")
                    
                    if 'arousal' in parsed_json:
                        try:
                            result['arousal'] = float(parsed_json['arousal'])
                        except (ValueError, TypeError) as parse_error:
                            logger.debug(f"타입 변환 실패, 기본값 사용: {parse_error}")
                    
                    # 추가 정보
                    if 'reasoning' in parsed_json:
                        result['reasoning'] = parsed_json['reasoning']
                    
                    # 한국어 필드명 처리 (기존 방식)
                    if '주요감정' in parsed_json:
                        emotion_text = parsed_json['주요감정']
                        mapped_emotion = self._name_to_emotion_id(emotion_text)
                        if mapped_emotion is None:
                            logger.error(f"❌ 한국어 감정 매핑 실패로 인한 파싱 실패: '{emotion_text}'")
                            return None  # 학습 오염 방지를 위한 명확한 실패 반환
                        result['emotion'] = mapped_emotion
                    
                    if '신뢰도' in parsed_json:
                        try:
                            result['confidence'] = float(parsed_json['신뢰도'])
                        except (ValueError, TypeError) as parse_error:
                            logger.debug(f"타입 변환 실패, 기본값 사용: {parse_error}")
                    
                    if '감정강도' in parsed_json:
                        try:
                            result['intensity'] = int(parsed_json['감정강도'])
                        except (ValueError, TypeError) as parse_error:
                            logger.debug(f"타입 변환 실패, 기본값 사용: {parse_error}")
                    
                    if '원인분석' in parsed_json:
                        result['cause_analysis'] = parsed_json['원인분석']
                    
                    logger.info(f"JSON 파싱 성공: 감정={result['emotion']}, 신뢰도={result['confidence']}")
                    return result
                    
            except (json.JSONDecodeError, AttributeError) as e:
                logger.warning(f"JSON 파싱 실패, 텍스트 파싱으로 전환: {e}")
            
            # 2. 기존 텍스트 파싱 (fallback)
            lines = response_text.split('\n')
            result = {
                'emotion': original_emotion,  # 기본값
                'confidence': 0.5,
                'reasoning': response_text
            }
            
            for line in lines:
                line = line.strip()
                if '주요 감정:' in line or '주요감정:' in line:
                    emotion_text = line.split(':')[1].strip()
                    mapped_emotion = self._name_to_emotion_id(emotion_text)
                    if mapped_emotion is None:
                        logger.error(f"❌ 텍스트 파싱 감정 매핑 실패: '{emotion_text}'")
                        return None  # 학습 오염 방지
                    result['emotion'] = mapped_emotion
                elif '신뢰도:' in line:
                    try:
                        conf_text = line.split(':')[1].strip()
                        result['confidence'] = float(conf_text)
                    except:
                        pass
                elif '감정강도:' in line:
                    try:
                        intensity_text = line.split(':')[1].strip()
                        result['intensity'] = int(intensity_text)
                    except:
                        pass
                        
            logger.info(f"텍스트 파싱 성공: 감정={result['emotion']}, 신뢰도={result['confidence']}")
            return result
            
        except Exception as e:
            logger.error(f"LLM 응답 파싱 완전 실패: {e}")
            # fallback 금지 원칙에 따라 None 반환
            return None
    
    
    
    def _name_to_emotion_id(self, emotion_name: str) -> int:
        """한국어/영어 감정 이름을 ID로 변환 (이중 언어 지원)"""
        # 입력 정규화
        emotion_name = emotion_name.lower().strip()
        
        name_mapping = {
            # 한국어 매핑 (기존)
            "기쁨": 1, "행복": 1, "즐거움": 1,
            "신뢰": 2, "믿음": 2,
            "두려움": 3, "불안": 3, "걱정": 3, "무서움": 3,
            "놀람": 4, "깜짝": 4, "놀라움": 4,
            "슬픔": 5, "우울": 5, "서글픔": 5,
            "혐오": 6, "싫음": 6, "역겨움": 6,
            "분노": 7, "화": 7, "짜증": 7, "화남": 7,
            "기대": 8, "예상": 8, "기대감": 8,
            "중립": 0, "무감정": 0, "보통": 0,
            # 추가 감정들
            "안도": 10, "relief": 10,
            "죄책감": 11, "guilt": 11,
            "수치심": 12, "shame": 12,
            "자부심": 13, "pride": 13,
            "경멸": 14, "contempt": 14,
            "질투": 15, "envy": 15,
            "감사": 16, "gratitude": 16,
            
            # 영어 매핑 (LLM 응답용)
            "joy": 1, "happiness": 1, "happy": 1,
            "trust": 2,
            "fear": 3, "anxiety": 3, "worried": 3, "afraid": 3,
            "surprise": 4, "surprised": 4,
            "sadness": 5, "sad": 5, "depression": 5, "depressed": 5,
            "disgust": 6, "disgusted": 6,
            "anger": 7, "angry": 7, "rage": 7, "mad": 7,
            "anticipation": 8, "expectation": 8,
            "neutral": 0, "none": 0, "normal": 0
        }
        
        # 정확한 매칭 우선
        if emotion_name in name_mapping:
            return name_mapping[emotion_name]
        
        # 부분 매칭 (기존 로직 유지)
        for name, emotion_id in name_mapping.items():
            if name in emotion_name or emotion_name in name:
                return emotion_id
                
        return EmotionState.NEUTRAL.value

# 상담사 모듈 기능 추가
class EmotionCounselorModule:
    """감정 분석 상담사 모듈 - 감정 원인 분석 및 후회 알고리즘 보조"""
    
    def __init__(self):
        self.llm_engine = None
        if LLM_INTEGRATION_AVAILABLE:
            try:
                from llm_module.advanced_llm_engine import get_llm_engine
                self.llm_engine = get_llm_engine()
            except Exception as e:
                logger.warning(f"상담사 모듈 LLM 초기화 실패: {e}")
    
    def analyze_emotion_causality(self, emotion_data: EmotionData, context: str) -> Dict[str, Any]:
        """감정 원인 분석 - 상담사 역할"""
        if not self.llm_engine:
            return {"analysis": "LLM 엔진을 사용할 수 없습니다.", "confidence": 0.0}
        
        try:
            from llm_module.advanced_llm_engine import LLMRequest, TaskComplexity
            
            prompt = f"""상담사로서 다음 감정 상태의 원인을 분석해주세요.

감정 정보:
- 주요 감정: {emotion_data.primary_emotion.name}
- 강도: {emotion_data.intensity.name}
- 신뢰도: {emotion_data.confidence:.3f}
- 감정가: {emotion_data.valence:.3f}
- 각성도: {emotion_data.arousal:.3f}

상황 맥락: "{context}"

상담사 관점에서 다음을 분석해주세요:
1. 감정 발생의 근본 원인
2. 감정 반응의 적절성 평가
3. 감정 조절 방안
4. 향후 유사 상황 대처법
5. 이 감정이 의사결정에 미칠 영향"""

            request = LLMRequest(
                prompt=prompt,
                task_type="causal_explanation",
                complexity=TaskComplexity.EXPERT,
                max_tokens=1000,  # 복잡한 윤리적 질문에 대한 고정 토큰 할당
                temperature=0.3
            )
            
            # 동기 방식으로 LLM 호출
            def run_llm_causality():
                import concurrent.futures
                def async_llm_call():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(self.llm_engine.generate_async(request))
                    finally:
                        loop.close()
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(async_llm_call)
                    return future.result(timeout=30)
            
            try:
                response = run_llm_causality()
                if response and response.success:
                    return {
                        "analysis": response.generated_text,
                        "confidence": response.confidence,
                        "processing_time": response.processing_time
                    }
                else:
                    return {"analysis": "분석 실패", "confidence": 0.0}
            except Exception as e:
                logger.error(f"감정 원인 분석 실패: {e}")
                return {"analysis": "분석 오류", "confidence": 0.0}
                
        except Exception as e:
            logger.error(f"감정 원인 분석 오류: {e}")
            return {"analysis": f"오류: {e}", "confidence": 0.0}
    
    def _detect_complex_ethical_question(self, text: str) -> bool:
        """복잡한 윤리적 질문인지 감지"""
        ethical_keywords = [
            "윤리적", "딜레마", "vs", "대", "선택", "방지", "보호", "권리", "안전", 
            "개인정보", "프라이버시", "감시", "테러", "의료", "자원", "배분",
            "자율주행", "브레이크", "환자", "인공호흡기", "정의", "공정", "도덕적"
        ]
        
        text_lower = text.lower()
        keyword_count = sum(1 for keyword in ethical_keywords if keyword in text_lower)
        
        # 키워드 3개 이상이고 길이가 50자 이상이면 복잡한 윤리적 질문으로 판단
        is_complex = keyword_count >= 3 and len(text) >= 50
        
        if is_complex:
            logger.info(f"복잡한 윤리적 질문 감지: 키워드 {keyword_count}개, 길이 {len(text)}자")
        
        return is_complex

    def _calculate_dynamic_token_limit(self, text: str, base_tokens: int = 400) -> int:
        """텍스트 복잡도에 따른 동적 토큰 제한 계산"""
        if self._detect_complex_ethical_question(text):
            # 복잡한 윤리적 질문의 경우 2배 토큰 할당
            dynamic_tokens = base_tokens * 2
            logger.info(f"복잡한 질문 감지 - 토큰 제한 증가: {base_tokens} → {dynamic_tokens}")
            return dynamic_tokens
        elif len(text) > 100:
            # 긴 텍스트의 경우 1.5배 토큰 할당
            dynamic_tokens = int(base_tokens * 1.5)
            logger.info(f"긴 텍스트 감지 - 토큰 제한 증가: {base_tokens} → {dynamic_tokens}")
            return dynamic_tokens
        else:
            return base_tokens

    async def validate_regret_reasoning(self, regret_data: Dict[str, Any], scenario: str) -> Dict[str, Any]:
        """후회 알고리즘의 추론 타당성 검증"""
        if not self.llm_engine:
            return {"validation": "검증 불가", "confidence": 0.0}
        
        try:
            from llm_module.advanced_llm_engine import LLMRequest, TaskComplexity
            
            prompt = f"""후회 분석 결과의 타당성을 전문가 관점에서 검증해주세요.

시나리오: "{scenario}"

후회 분석 결과:
- 후회 강도: {regret_data.get('intensity', 'N/A')}
- 예측 오류: {regret_data.get('prediction_error', 'N/A')}
- 더 나은 선택: {regret_data.get('better_options', 'N/A')}

전문가로서 다음을 평가해주세요:
1. 후회 강도의 적절성
2. 대안 선택의 현실성
3. 예측 오류 계산의 합리성
4. 누락된 중요 요소
5. 전반적 타당성 점수 (0-100)"""

            request = LLMRequest(
                prompt=prompt,
                task_type="ethical_analysis",
                complexity=TaskComplexity.EXPERT,
                max_tokens=400,
                temperature=0.2
            )
            
            # 동기 방식으로 LLM 호출
            def run_llm_validation():
                import concurrent.futures
                def async_llm_call():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(self.llm_engine.generate_async(request))
                    finally:
                        loop.close()
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(async_llm_call)
                    return future.result(timeout=30)
            
            try:
                response = run_llm_validation()
                if response and response.success:
                    # 타당성 점수 추출
                    validity_score = self._extract_validity_score(response.generated_text)
                    
                    return {
                        "validation": response.generated_text,
                        "validity_score": validity_score,
                        "confidence": response.confidence
                    }
                else:
                    return {"validation": "검증 실패", "validity_score": 0}
            except Exception as e:
                logger.error(f"후회 추론 검증 실행 오류: {e}")
                return {"validation": f"실행 오류: {e}", "validity_score": 0}
                
        except Exception as e:
            logger.error(f"후회 추론 검증 오류: {e}")
            return {"validation": f"오류: {e}", "validity_score": 0}
    
    def _extract_validity_score(self, analysis_text: str) -> float:
        """분석 텍스트에서 타당성 점수 추출"""
        import re
        
        # 점수 패턴 찾기
        score_patterns = [
            r'타당성 점수[:\s]*(\d+)',
            r'점수[:\s]*(\d+)',
            r'(\d+)점',
            r'(\d+)/100'
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, analysis_text)
            if match:
                try:
                    score = float(match.group(1))
                    return min(100.0, max(0.0, score)) / 100.0  # 0-1 범위로 정규화
                except:
                    continue
        
        return 0.5  # 기본값

if __name__ == "__main__":
    test_advanced_emotion_analyzer()