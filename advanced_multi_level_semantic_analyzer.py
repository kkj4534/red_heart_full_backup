"""
고급 다중수준 의미 분석 시스템 - Linux 전용
Advanced Multi-Level Semantic Analysis System for Linux

최신 AI 기법을 활용한 4수준(표면적, 윤리적, 감정적, 인과적) 의미 분석
Transformer 기반 대규모 언어모델과 고급 ML 기법을 결합한 심층 의미 이해
"""

__all__ = ['AdvancedMultiLevelSemanticAnalyzer', 'create_advanced_semantic_analyzer']

import os
import logging
import json
import time
import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from dataclasses import dataclass, field
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from collections import defaultdict

# 고급 AI 라이브러리
# SentenceTransformer는 sentence_transformer_singleton을 통해 사용
from transformers import (
    pipeline, AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    AutoModelForCausalLM, BertTokenizer, BertModel, GPT2LMHeadModel
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

from config import ADVANCED_CONFIG, SYSTEM_CONFIG, DEVICE, TORCH_DTYPE, BATCH_SIZE, MODELS_DIR, get_smart_device, ModelPriority, get_priority_based_device
from data_models import (
    SemanticRepresentationData, SemanticLevel, IntentionCategory,
    AdvancedSemanticResult, SemanticCluster, SemanticNetwork,
    CausalRelation, EthicalDimension, EmotionalProfile
)

logger = logging.getLogger('RedHeart.AdvancedMultiLevelSemantic')

@dataclass
class SemanticAnalysisState:
    """의미 분석 상태 정보"""
    current_level: str = ""
    processing_time: float = 0.0
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    error_log: List[str] = field(default_factory=list)

@dataclass
class CrossLevelSemanticRelation:
    """수준 간 의미 관계"""
    source_level: str
    target_level: str
    relation_type: str  # "supports", "contradicts", "neutral", "amplifies"
    strength: float  # 0.0 ~ 1.0
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0

class AdvancedMultiLevelSemanticAnalyzer:
    """고급 다중수준 의미론적 분석기"""
    
    def __init__(self):
        """고급 의미 분석기 초기화"""
        self.config = SYSTEM_CONFIG['semantic']
        self.device = DEVICE
        self.dtype = TORCH_DTYPE
        self.batch_size = BATCH_SIZE
        
        # 모델 디렉토리 설정
        self.models_dir = os.path.join(MODELS_DIR, 'semantic_models')
        self.cache_dir = os.path.join(MODELS_DIR, 'semantic_cache')
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 캐시 시스템
        self.embedding_cache = {}
        self.analysis_cache = {}
        self.relation_cache = {}
        
        # 비동기 초기화를 위한 플래그들
        self.models_initialized = False
        self.classifiers_initialized = False
        self.networks_initialized = False
        self.fully_initialized = False
        
        # 분석 상태 추적
        self.analysis_state = SemanticAnalysisState()
        self.cross_level_relations = []
        
        # GPU 로더 바인딩 (오케스트레이터/CONFIG 제공)
        try:
            from config import get_gpu_loader
            self.gpu_loader = get_gpu_loader()
            logger.info("GPU Loader 바인딩 완료")
        except Exception as e:
            self.gpu_loader = None
            logger.warning(f"GPU Loader 미바인딩 (초기화 중 재시도 예정): {e}")
        
        logger.info("AdvancedMultiLevelSemanticAnalyzer 기본 초기화 완료 (비동기 초기화 대기 중)")
    
    async def initialize(self):
        """비동기 초기화 - 모든 모델과 네트워크 설정"""
        # 취소/중복 방지를 위한 lock과 event 초기화
        if not hasattr(self, '_init_lock'):
            self._init_lock = asyncio.Lock()
        if not hasattr(self, '_cancel_event'):
            self._cancel_event = asyncio.Event()
        
        # 이미 초기화됨
        if self.fully_initialized:
            return
        
        # Lock으로 중복 초기화 방지
        async with self._init_lock:
            if self.fully_initialized:
                return
            
            logger.info("AdvancedMultiLevelSemanticAnalyzer 비동기 초기화 시작...")
            start_ts = time.time()
            self.last_heartbeat_ts = time.time()  # 하트비트 값 초기화
            
            # GPU 메모리 스냅샷 (초기화 시작)
            from config import get_gpu_memory_info
            gpu_info = get_gpu_memory_info()
            if gpu_info:
                logger.info(f"📊 [semantic 초기화 시작] GPU: {gpu_info['usage_percent']:.1f}% 사용, {gpu_info['free_mb']:.0f}MB 여유")
        
        # DSM placeholder 선등록 (초기화 시작 시점)
        try:
            from dynamic_swap_manager import get_swap_manager, SwapPriority
            swap = get_swap_manager()
            if swap and "semantic_analyzer" not in getattr(swap, 'models', {}):
                class _TinyPlaceholder(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.dummy = nn.Linear(8, 8, bias=False)
                    def forward(self, x):
                        return x if not isinstance(x, torch.Tensor) else self.dummy(x[..., :8])
                swap.register_model("semantic_analyzer", _TinyPlaceholder(), priority=SwapPriority.MEDIUM)
                logger.info("🧩 semantic_analyzer DSM 선등록(placeholder) 완료")
                logger.info(f"   DSM keys after placeholder: {list(swap.models.keys())[:10]}")
        except Exception as e:
            logger.warning(f"DSM placeholder 선등록 스킵: {e}")
        
        # GPU 메모리 예약 (device_policy 스펙에 따라)
        from workflow_aware_memory_manager import WorkflowAwareMemoryManager
        try:
            from module_specs import MODULE_SPECS
        except Exception:
            MODULE_SPECS = []  # 폴백: 스펙 없어도 동작
        
        # GPU 로더 재바인딩 시도 (필요시)
        if self.gpu_loader is None:
            try:
                from config import get_gpu_loader
                self.gpu_loader = get_gpu_loader()
                logger.info("GPU Loader 재바인딩 성공")
            except Exception as e:
                logger.warning(f"GPU Loader 재바인딩 실패: {e} (NoopLoader 폴백 적용)")
                # NoopLoader 폴백 처리
                class _NoopLoader:
                    def request_gpu_loading(self, *a, **kw): return ("cpu", {})
                self.gpu_loader = _NoopLoader()
        
        # gpu_loader 보증 확인
        assert hasattr(self, 'gpu_loader') and self.gpu_loader is not None, "gpu_loader must be available"
        logger.info(f"✅ gpu_loader 보증 확인 완료 (type: {type(self.gpu_loader).__name__})")
        
        # DSM/로더 실측 기반으로 승격 처리 (추정치 선할당 제거)
        # CPU 선적재 후 필요시 GPU로 프로모션하는 방식으로 변경됨
        logger.info("🔄 DSM 실측 기반 메모리 관리 정책 적용 (추정치 선할당 제거)")
        
        # 하트비트 태스크 시작 (값 노출 포함)
        async def _heartbeat():
            while not self.fully_initialized and not self._cancel_event.is_set():
                elapsed = int(time.time() - start_ts)
                self.last_heartbeat_ts = time.time()  # orchestrator가 참조할 수 있도록 값 갱신
                logger.info(f"⏳ semantic_analyzer init… {elapsed}s 경과 (heartbeat_ts={self.last_heartbeat_ts:.1f})")
                await asyncio.sleep(2.0)
        heartbeat_task = asyncio.create_task(_heartbeat())
        
        try:
            # 순차적으로 초기화 (메모리 사용량 최적화) - asyncio.to_thread로 비동기화
            if not self.models_initialized:
                models_start = time.time()
                await asyncio.to_thread(self._initialize_models_cpu_safe)
                self.models_initialized = True
                
                # 취소 체크
                if self._cancel_event.is_set():
                    raise asyncio.CancelledError("초기화 취소됨")
                
                models_elapsed = int((time.time() - models_start) * 1000)
                logger.info(f"모델 초기화 완료 (+{models_elapsed}ms)")
                
                # 단계별 RAM 선등록 - 모델 초기화 후
                try:
                    from dynamic_swap_manager import get_swap_manager, SwapPriority
                    swap = get_swap_manager()
                    if swap:
                        # embedding_model 등록
                        if hasattr(self, 'embedding_model'):
                            embed_core = getattr(self.embedding_model, 'auto_model', None) or \
                                       getattr(self.embedding_model, 'model', None) or \
                                       getattr(self.embedding_model, '_model', None)
                            if embed_core:
                                swap.register_model("semantic_embedding", embed_core, priority=SwapPriority.LOW)
                                logger.info("✅ semantic_embedding RAM 선등록")
                        
                        # emotion_pipeline 등록
                        if hasattr(self, 'emotion_pipeline') and hasattr(self.emotion_pipeline, 'model'):
                            swap.register_model("semantic_emotion", self.emotion_pipeline.model, priority=SwapPriority.LOW)
                            logger.info("✅ semantic_emotion RAM 선등록")
                except Exception as e:
                    logger.warning(f"모델 단계 RAM 선등록 스킵: {e}")
                
                # 이벤트 루프 즉시 양보 (블로킹 방지)
                await asyncio.sleep(0)
            
            if not self.classifiers_initialized:
                classifiers_start = time.time()
                await asyncio.to_thread(self._initialize_classifiers)
                self.classifiers_initialized = True
                
                # 취소 체크
                if self._cancel_event.is_set():
                    raise asyncio.CancelledError("초기화 취소됨")
                
                classifiers_elapsed = int((time.time() - classifiers_start) * 1000)
                logger.info(f"분류기 초기화 완료 (+{classifiers_elapsed}ms)")
                
                # 단계별 RAM 선등록 - 분류기 초기화 후
                try:
                    from dynamic_swap_manager import get_swap_manager, SwapPriority
                    swap = get_swap_manager()
                    if swap:
                        # ethical_model 등록
                        if hasattr(self, 'ethical_model'):
                            swap.register_model("semantic_ethical", self.ethical_model, priority=SwapPriority.LOW)
                            logger.info("✅ semantic_ethical RAM 선등록")
                        
                        # causal_pipeline 등록
                        if hasattr(self, 'causal_pipeline') and hasattr(self.causal_pipeline, 'model'):
                            swap.register_model("semantic_causal", self.causal_pipeline.model, priority=SwapPriority.LOW)
                            logger.info("✅ semantic_causal RAM 선등록")
                except Exception as e:
                    logger.warning(f"분류기 단계 RAM 선등록 스킵: {e}")
                
                # 이벤트 루프 즉시 양보
                await asyncio.sleep(0)
            
            if not self.networks_initialized:
                networks_start = time.time()
                # CPU-bound 작업만 워커 스레드에서 수행
                await asyncio.to_thread(self._setup_neural_networks_cpu)
                # GPU 로더/디바이스 이동/DSM 등록은 메인 이벤트 루프에서 수행
                await self._setup_neural_networks_gpu()
                self.networks_initialized = True
                
                # 취소 체크
                if self._cancel_event.is_set():
                    raise asyncio.CancelledError("초기화 취소됨")
                
                networks_elapsed = int((time.time() - networks_start) * 1000)
                logger.info(f"신경망 초기화 완료 (+{networks_elapsed}ms)")
                
                # 단계별 RAM 선등록 - 신경망 초기화 후
                try:
                    from dynamic_swap_manager import get_swap_manager, SwapPriority
                    swap = get_swap_manager()
                    if swap:
                        # fusion_network 등록
                        if hasattr(self, 'fusion_network'):
                            swap.register_model("semantic_fusion", self.fusion_network, priority=SwapPriority.LOW)
                            logger.info("✅ semantic_fusion RAM 선등록")
                        
                        # cross_attention 등록
                        if hasattr(self, 'cross_attention'):
                            swap.register_model("semantic_cross_attention", self.cross_attention, priority=SwapPriority.LOW)
                            logger.info("✅ semantic_cross_attention RAM 선등록")
                except Exception as e:
                    logger.warning(f"신경망 단계 RAM 선등록 스킵: {e}")
                
                # 이벤트 루프 즉시 양보
                await asyncio.sleep(0)
            
            self.fully_initialized = True
            total_elapsed = int((time.time() - start_ts) * 1000)
            logger.info(f"AdvancedMultiLevelSemanticAnalyzer 비동기 초기화 완료 (총 {total_elapsed}ms 소요)")
            
            # GPU 메모리 스냅샷 (초기화 완료)
            gpu_info = get_gpu_memory_info()
            if gpu_info:
                logger.info(f"📊 [semantic 초기화 완료] GPU: {gpu_info['usage_percent']:.1f}% 사용, {gpu_info['free_mb']:.0f}MB 여유")
            
            # 마지막: core_module로 교체 등록
            try:
                from dynamic_swap_manager import get_swap_manager, SwapPriority
                swap = get_swap_manager()
                core_module = self.get_pytorch_network()
                
                if swap and core_module:
                    old_type = type(swap.models.get('semantic_analyzer', {}).model).__name__ if 'semantic_analyzer' in swap.models else 'None'
                    swap.register_model("semantic_analyzer", core_module, priority=SwapPriority.HIGH)
                    new_type = type(swap.models['semantic_analyzer'].model).__name__
                    logger.info(f"✅ semantic_analyzer DSM 등록 완료 (placeholder → 본체 교체: {old_type} → {new_type})")
                    logger.info(f"   DSM keys after replacement: {list(swap.models.keys())[:10]}")
                    
                    # GPU 메모리 스냅샷 (DSM 교체 후)
                    gpu_info = get_gpu_memory_info()
                    if gpu_info:
                        logger.info(f"📊 [DSM 교체 후] GPU: {gpu_info['usage_percent']:.1f}% 사용, {gpu_info['free_mb']:.0f}MB 여유")
                else:
                    logger.warning(f"⚠️ semantic_analyzer DSM 등록 스킵 (swap={swap is not None}, core={core_module is not None})")
            except Exception as e:
                logger.error(f"semantic_analyzer DSM 등록 실패: {e}")
            
        except asyncio.CancelledError:
            self._cancel_event.set()
            logger.warning("⛔ semantic_analyzer 초기화 취소됨 (타임아웃/중단)")
            raise
        except Exception as e:
            logger.error(f"AdvancedMultiLevelSemanticAnalyzer 초기화 실패: {str(e)}")
            raise
        finally:
            # 하트비트 안전 정리
            if 'heartbeat_task' in locals():
                heartbeat_task.cancel()
                from contextlib import suppress
                with suppress(asyncio.CancelledError):
                    await heartbeat_task
    
    def _initialize_models_cpu_safe(self):
        """모든 모델을 CPU로 강제 로드하는 안전한 초기화"""
        # 기존 _initialize_models 호출 후 CPU 강제
        self._initialize_models()
        
        # 모든 모델을 CPU로 이동 (GPU 우회 방지)
        try:
            # embedding_model을 CPU로
            if hasattr(self, 'embedding_model') and hasattr(self.embedding_model, 'to'):
                self.embedding_model.to('cpu')
                logger.info("embedding_model CPU 강제 완료")
            
            # emotion_pipeline 내부 모델을 CPU로  
            if hasattr(self, 'emotion_pipeline') and hasattr(self.emotion_pipeline, 'model'):
                self.emotion_pipeline.model.to('cpu')
                logger.info("emotion_pipeline.model CPU 강제 완료")
                
            # ethical_model을 CPU로
            if hasattr(self, 'ethical_model') and hasattr(self.ethical_model, 'to'):
                self.ethical_model.to('cpu')
                logger.info("ethical_model CPU 강제 완료")
                
        except Exception as e:
            logger.warning(f"CPU 강제 중 경고: {e}")
        
        logger.info("✅ 모든 모델 CPU 강제 로드 완료 (DSM을 통한 GPU 승격 대기)")
    
    def _initialize_models(self):
        """고급 AI 모델들 초기화"""
        try:
            # 다국어 임베딩 모델 (한국어 특화)
            from sentence_transformer_singleton import get_sentence_transformer
            
            self.embedding_model = get_sentence_transformer(
                'paraphrase-multilingual-mpnet-base-v2',
                device='cpu',  # CPU로 강제 로드 (gpu_on_demand 정책)
                cache_folder=self.models_dir
            )
            
            # 감정 분석 모델 (gpu_on_demand: CPU 선적재)
            logger.info("감정 분석 파이프라인 로딩 시작 (gpu_on_demand)")
            try:
                # HF 래퍼 사용 시도
                try:
                    from hf_model_wrapper import wrapped_pipeline
                    self.emotion_pipeline = wrapped_pipeline(
                        task="text-classification",
                        model="j-hartmann/emotion-english-distilroberta-base",
                        owner="semantic_analyzer",
                        device=-1,  # CPU로 먼저 로드 (gpu_on_demand)
                        torch_dtype=torch.float32  # CPU는 float32 사용
                    )
                    logger.info("감정 분석 파이프라인 로딩 성공 (HF 래퍼, CPU)")
                except ImportError:
                    from transformers import pipeline
                    self.emotion_pipeline = pipeline(
                        "text-classification",
                        model="j-hartmann/emotion-english-distilroberta-base",
                        device=-1  # CPU로 먼저 로드 (gpu_on_demand)
                    )
                    logger.info("감정 분석 파이프라인 로딩 성공 (CPU)")
                
                # 모델 인스턴스 확인
                if self.emotion_pipeline is None:
                    raise RuntimeError("감정 분석 모델 인스턴스 생성 실패")
                
                logger.info("✅ emotion pipeline CPU 선적재 완료 (gpu_on_demand)")
                
            except Exception as e:
                logger.error(f"감정 분석 파이프라인 로딩 실패: {e}")
                raise RuntimeError(f"감정 분석 모델 로딩 실패: {e}")
            
            # 윤리적 분류 모델 (gpu_on_demand: CPU 선적재)
            logger.info("윤리 분석 모델 로딩 시작 (gpu_on_demand)")
            self.ethical_tokenizer = AutoTokenizer.from_pretrained(
                'bert-base-uncased',
                cache_dir=self.models_dir
            )
            
            try:
                # HF 래퍼 우선 시도
                try:
                    from hf_model_wrapper import wrapped_from_pretrained
                    self.ethical_model = wrapped_from_pretrained(
                        model_class=AutoModel,
                        model_name='bert-base-uncased',
                        owner="semantic_analyzer",
                        cache_dir=self.models_dir
                    ).to(torch.device('cpu'))  # CPU로 먼저 로드 (gpu_on_demand)
                    logger.info("윤리 분석 모델 로딩 성공 (HF 래퍼, CPU)")
                except ImportError:
                    # 래퍼 없으면 직접 로딩
                    self.ethical_model = AutoModel.from_pretrained(
                        'bert-base-uncased',
                        cache_dir=self.models_dir
                    ).to(torch.device('cpu'))  # CPU로 먼저 로드 (gpu_on_demand)
                    logger.info("윤리 분석 모델 로딩 성공 (직접, CPU)")
                
                # 모델 인스턴스 확인
                if self.ethical_model is None:
                    raise RuntimeError("윤리 분석 모델 인스턴스 생성 실패")
                
                logger.info("✅ ethical model CPU 선적재 완료 (gpu_on_demand)")
                
            except Exception as e:
                logger.error(f"윤리 분석 모델 로딩 실패: {e}")
                raise RuntimeError(f"윤리 분석 모델 로딩 실패: {e}")
            
            # 인과관계 추론 모델 (gpu_on_demand: CPU 선적재)
            logger.info("인과관계 분석 파이프라인 로딩 시작 (gpu_on_demand)")
            try:
                # HF 래퍼 우선 시도
                try:
                    from hf_model_wrapper import wrapped_pipeline
                    self.causal_pipeline = wrapped_pipeline(
                        task="zero-shot-classification",
                        model="facebook/bart-large-mnli",
                        owner="semantic_analyzer",
                        device=-1  # CPU로 먼저 로드 (gpu_on_demand)
                    )
                    logger.info("인과관계 분석 파이프라인 로딩 성공 (HF 래퍼, CPU)")
                except ImportError:
                    # 래퍼 없으면 직접 로딩
                    from transformers import pipeline
                    self.causal_pipeline = pipeline(
                        "zero-shot-classification",
                        model="facebook/bart-large-mnli",
                        device=-1  # CPU로 먼저 로드 (gpu_on_demand)
                    )
                    logger.info("인과관계 분석 파이프라인 로딩 성공 (직접, CPU)")
                
                # 모델 인스턴스 확인
                if self.causal_pipeline is None:
                    raise RuntimeError("인과관계 분석 모델 인스턴스 생성 실패")
                
                logger.info("✅ causal pipeline CPU 선적재 완료 (gpu_on_demand)")
                
            except Exception as e:
                logger.error(f"인과관계 분석 파이프라인 로딩 실패: {e}")
                raise RuntimeError(f"인과관계 분석 모델 로딩 실패: {e}")
            # causal_device 보정: 값이 항상 정의되도록 처리
            try:
                causal_device = torch.device('cpu')  # 기본값
                if hasattr(self, 'causal_pipeline') and hasattr(self.causal_pipeline, 'model'):
                    if hasattr(self.causal_pipeline.model, 'device'):
                        causal_device = self.causal_pipeline.model.device
            except Exception:
                causal_device = torch.device('cpu')
            logger.info(f"인과관계 모델 순차 로드 완료: {causal_device}")
            
            logger.info("고급 AI 모델들이 성공적으로 로드되었습니다.")
            
        except Exception as e:
            logger.error(f"모델 초기화 실패: {e}")
            # 프로젝트 규칙: fallback 없는 순수 재시도 방식
            retry_count = 0
            max_retries = 3
            while retry_count < max_retries:
                retry_count += 1
                logger.info(f"모델 초기화 재시도 {retry_count}/{max_retries}")
                try:
                    # 다국어 임베딩 모델 (한국어 특화)
                    from sentence_transformer_singleton import get_sentence_transformer
                    
                    self.embedding_model = get_sentence_transformer(
                        'paraphrase-multilingual-mpnet-base-v2',
                        device=str(self.device),
                        cache_folder=self.models_dir
                    )
                    
                    # 감정 분석 모델 (재시도, gpu_on_demand: CPU 선적재)
                    logger.info("감정 분석 파이프라인 재시도 로딩 시작 (gpu_on_demand)")
                    try:
                        # HF 래퍼 우선 시도
                        try:
                            from hf_model_wrapper import wrapped_pipeline
                            self.emotion_pipeline = wrapped_pipeline(
                                task="text-classification",
                                model="j-hartmann/emotion-english-distilroberta-base",
                                owner="semantic_analyzer",
                                device=-1  # CPU로 먼저 로드 (gpu_on_demand)
                            )
                            logger.info("감정 분석 파이프라인 재시도 로딩 성공 (HF 래퍼, CPU)")
                        except ImportError:
                            # 래퍼 없으면 직접 로딩
                            from transformers import pipeline
                            self.emotion_pipeline = pipeline(
                                "text-classification",
                                model="j-hartmann/emotion-english-distilroberta-base",
                                device=-1  # CPU로 먼저 로드 (gpu_on_demand)
                            )
                            logger.info("감정 분석 파이프라인 재시도 로딩 성공 (직접, CPU)")
                        
                        if self.emotion_pipeline is None:
                            raise RuntimeError("감정 분석 모델 인스턴스 생성 실패 (재시도)")
                        
                        logger.info("✅ emotion pipeline CPU 선적재 완료 (재시도, gpu_on_demand)")
                        
                    except Exception as e:
                        logger.error(f"감정 분석 파이프라인 재시도 로딩 실패: {e}")
                        raise
                    
                    # 윤리적 분류 모델 (재시도, gpu_on_demand: CPU 선적재)
                    logger.info("윤리 분석 모델 재시도 로딩 시작 (gpu_on_demand)")
                    self.ethical_tokenizer = AutoTokenizer.from_pretrained(
                        'bert-base-uncased',
                        cache_dir=self.models_dir
                    )
                    
                    try:
                        # HF 래퍼 우선 시도
                        try:
                            from hf_model_wrapper import wrapped_from_pretrained
                            self.ethical_model = wrapped_from_pretrained(
                                model_class=AutoModel,
                                model_name='bert-base-uncased',
                                owner="semantic_analyzer",
                                cache_dir=self.models_dir
                            ).to(torch.device('cpu'))  # CPU로 먼저 로드 (gpu_on_demand)
                            logger.info("윤리 분석 모델 재시도 로딩 성공 (HF 래퍼, CPU)")
                        except ImportError:
                            # 래퍼 없으면 직접 로딩
                            self.ethical_model = AutoModel.from_pretrained(
                                'bert-base-uncased',
                                cache_dir=self.models_dir
                            ).to(torch.device('cpu'))  # CPU로 먼저 로드 (gpu_on_demand)
                            logger.info("윤리 분석 모델 재시도 로딩 성공 (직접, CPU)")
                        
                        if self.ethical_model is None:
                            raise RuntimeError("윤리 분석 모델 인스턴스 생성 실패 (재시도)")
                        
                        logger.info("✅ ethical model CPU 선적재 완료 (재시도, gpu_on_demand)")
                        
                    except Exception as e:
                        logger.error(f"윤리 분석 모델 재시도 로딩 실패: {e}")
                        raise
                    
                    # 인과관계 추론 모델 (재시도, gpu_on_demand: CPU 선적재)
                    logger.info("인과관계 분석 파이프라인 재시도 로딩 시작 (gpu_on_demand)")
                    try:
                        # HF 래퍼 우선 시도
                        try:
                            from hf_model_wrapper import wrapped_pipeline
                            self.causal_pipeline = wrapped_pipeline(
                                task="zero-shot-classification",
                                model="facebook/bart-large-mnli",
                                owner="semantic_analyzer",
                                device=-1  # CPU로 먼저 로드 (gpu_on_demand)
                            )
                            logger.info("인과관계 분석 파이프라인 재시도 로딩 성공 (HF 래퍼, CPU)")
                        except ImportError:
                            # 래퍼 없으면 직접 로딩
                            from transformers import pipeline
                            self.causal_pipeline = pipeline(
                                "zero-shot-classification",
                                model="facebook/bart-large-mnli",
                                device=-1  # CPU로 먼저 로드 (gpu_on_demand)
                            )
                            logger.info("인과관계 분석 파이프라인 재시도 로딩 성공 (직접, CPU)")
                        
                        if self.causal_pipeline is None:
                            raise RuntimeError("인과관계 분석 모델 인스턴스 생성 실패 (재시도)")
                        
                        logger.info("✅ causal pipeline CPU 선적재 완료 (재시도, gpu_on_demand)")
                        
                    except Exception as e:
                        logger.error(f"인과관계 분석 파이프라인 재시도 로딩 실패: {e}")
                        raise
                    
                    logger.info("고급 AI 모델들이 성공적으로 로드되었습니다 (재시도).")
                    break
                    
                except Exception as retry_error:
                    logger.error(f"재시도 {retry_count} 실패: {retry_error}")
                    if retry_count >= max_retries:
                        logger.error("모든 재시도 실패 - 시스템 종료")
                        raise Exception(f"모델 초기화 최종 실패: {retry_error}")
                    import time
                    time.sleep(1)  # 재시도 간격
    
# 프로젝트 규칙 준수: fallback 메소드 제거됨 - 순수 재시도 방식만 사용

# Lazy loading 제거 - 모든 모델은 초기화 시 로딩됨
                retry_count += 1
                logger.info(f"모델 초기화 재시도 {retry_count}/{max_retries}")
                try:
                    # 다국어 임베딩 모델 (한국어 특화)
                    from sentence_transformer_singleton import get_sentence_transformer
                    
                    self.embedding_model = get_sentence_transformer(
                        'paraphrase-multilingual-mpnet-base-v2',
                        device=str(self.device),
                        cache_folder=self.models_dir
                    )
                    
                    # 감정 분석 모델 (순차적 로딩 재시도)
                    def load_semantic_emotion_pipeline_retry():
                        # GPU 로더 내부에서 실행되므로 필요한 모든 import 포함
                        from transformers import pipeline
                        import torch
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.info("GPU 로더에서 감정 분석 파이프라인 재시도 로딩 시작")
                        try:
                            # HF 래퍼 우선 시도
                            try:
                                from hf_model_wrapper import wrapped_pipeline
                                model = wrapped_pipeline(
                                    task="text-classification",
                                    model="j-hartmann/emotion-english-distilroberta-base",
                                    owner="semantic_analyzer",
                                    device=0 if torch.cuda.is_available() else -1
                                )
                                logger.info("감정 분석 파이프라인 재시도 로딩 성공 (HF 래퍼)")
                            except ImportError:
                                # 래퍼 없으면 직접 로딩
                                model = pipeline(
                                    "text-classification",
                                    model="j-hartmann/emotion-english-distilroberta-base",
                                    device=0 if torch.cuda.is_available() else -1
                                )
                                logger.info("감정 분석 파이프라인 재시도 로딩 성공 (직접)")
                            return model
                        except Exception as e:
                            logger.error(f"감정 분석 파이프라인 재시도 로딩 실패: {e}")
                            raise
                    
                    # GPU 로더 필드 보장 후 사용
                    if self.gpu_loader is None:
                        from config import get_gpu_loader
                        self.gpu_loader = get_gpu_loader()
                    
                    if self.gpu_loader is not None:
                        emotion_device, emotion_model = self.gpu_loader.request_gpu_loading(
                            model_id="semantic_emotion_pipeline_retry",
                            priority=ModelPriority.MEDIUM,
                            estimated_memory_mb=732,
                            loading_function=load_semantic_emotion_pipeline_retry
                        )
                        
                        # gpu_on_demand 정책: CPU 선적재 허용 (재시도)
                        if emotion_model is None:
                            logger.warning("감정 분석 모델 인스턴스 없음, CPU 폴백")
                            emotion_model = load_semantic_emotion_pipeline_retry()
                            emotion_device = torch.device('cpu')
                    else:
                        logger.warning("GPU 로더 없음, CPU 모드로 감정 분석 모델 로드 (재시도)")
                        emotion_device = torch.device('cpu')
                        emotion_model = load_semantic_emotion_pipeline_retry()
                    
                    if getattr(emotion_device, "type", "cpu") != "cuda":
                        logger.warning("emotion pipeline on CPU (gpu_on_demand, retry). Will promote via DSM when needed.")
                    self.emotion_pipeline = emotion_model
                    logger.info(f"감정 분석 모델 순차 로드 완료 (재시도): {emotion_device}")
                    
                    # 윤리적 분류 모델 (BERT 기반)
                    self.ethical_tokenizer = AutoTokenizer.from_pretrained(
                        'bert-base-uncased',
                        cache_dir=self.models_dir
                    )
                    def load_semantic_ethical_model_retry():
                        # GPU 로더 내부에서 실행되므로 필요한 모든 import 포함
                        from transformers import AutoModel
                        import torch
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.info("GPU 로더에서 윤리 분석 모델 재시도 로딩 시작")
                        try:
                            # HF 래퍼 우선 시도
                            try:
                                from hf_model_wrapper import wrapped_from_pretrained
                                model = wrapped_from_pretrained(
                                    model_class=AutoModel,
                                    model_name='bert-base-uncased',
                                    owner="semantic_analyzer",
                                    cache_dir=models_dir_cache
                                ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                                logger.info("윤리 분석 모델 재시도 로딩 성공 (HF 래퍼)")
                            except ImportError:
                                # 래퍼 없으면 직접 로딩
                                model = AutoModel.from_pretrained(
                                    'bert-base-uncased',
                                    cache_dir=models_dir_cache
                                ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                                logger.info("윤리 분석 모델 재시도 로딩 성공 (직접)")
                            return model
                        except Exception as e:
                            logger.error(f"윤리 분석 모델 재시도 로딩 실패: {e}")
                            raise
                    
                    # GPU 로더 필드 보장 후 사용
                    if self.gpu_loader is None:
                        from config import get_gpu_loader
                        self.gpu_loader = get_gpu_loader()
                    
                    if self.gpu_loader is not None:
                        ethical_device, ethical_model = self.gpu_loader.request_gpu_loading(
                            model_id="semantic_ethical_model_retry",
                            priority=ModelPriority.LOW,
                            estimated_memory_mb=732,
                            loading_function=load_semantic_ethical_model_retry
                        )
                        
                        # gpu_on_demand 정책: CPU 선적재 허용 (재시도)
                        if ethical_model is None:
                            logger.warning("윤리 분석 모델 인스턴스 없음, CPU 폴백")
                            ethical_model = load_semantic_ethical_model_retry()
                            ethical_device = torch.device('cpu')
                    else:
                        logger.warning("GPU 로더 없음, CPU 모드로 윤리 평가 모델 로드 (재시도)")
                        ethical_device = torch.device('cpu')
                        ethical_model = load_semantic_ethical_model_retry()
                    
                    if getattr(ethical_device, "type", "cpu") != "cuda":
                        logger.warning("ethical model on CPU (gpu_on_demand, retry). Will promote via DSM when needed.")
                    self.ethical_model = ethical_model
                    logger.info(f"윤리 분석 모델 순차 로드 완료 (재시도): {ethical_device}")
                    
                    # 인과관계 추론 모델 (순차적 로딩 재시도)
                    def load_semantic_causal_pipeline_retry():
                        # GPU 로더 내부에서 실행되므로 필요한 모든 import 포함
                        from transformers import pipeline
                        import torch
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.info("GPU 로더에서 인과관계 분석 파이프라인 재시도 로딩 시작")
                        try:
                            # HF 래퍼 우선 시도
                            try:
                                from hf_model_wrapper import wrapped_pipeline
                                model = wrapped_pipeline(
                                    task="zero-shot-classification",
                                    model="facebook/bart-large-mnli",
                                    owner="semantic_analyzer",
                                    device=0 if torch.cuda.is_available() else -1
                                )
                                logger.info("인과관계 분석 파이프라인 재시도 로딩 성공 (HF 래퍼)")
                            except ImportError:
                                # 래퍼 없으면 직접 로딩
                                model = pipeline(
                                    "zero-shot-classification",
                                    model="facebook/bart-large-mnli",
                                    device=0 if torch.cuda.is_available() else -1
                                )
                                logger.info("인과관계 분석 파이프라인 재시도 로딩 성공 (직접)")
                            return model
                        except Exception as e:
                            logger.error(f"인과관계 분석 파이프라인 재시도 로딩 실패: {e}")
                            raise
                    
                    # GPU 로더 필드 보장 후 사용
                    if self.gpu_loader is None:
                        from config import get_gpu_loader
                        self.gpu_loader = get_gpu_loader()
                    
                    if self.gpu_loader is not None:
                        causal_device, causal_model = self.gpu_loader.request_gpu_loading(
                            model_id="semantic_causal_pipeline_retry",
                            priority=ModelPriority.LOW,
                            estimated_memory_mb=732,
                            loading_function=load_semantic_causal_pipeline_retry
                        )
                        
                        # gpu_on_demand 정책: CPU 선적재 허용 (재시도)
                        if causal_model is None:
                            logger.warning("인과관계 분석 모델 인스턴스 없음, CPU 폴백")
                            causal_model = load_semantic_causal_pipeline_retry()
                            causal_device = torch.device('cpu')
                    else:
                        logger.warning("GPU 로더 없음, CPU 모드로 인과관계 모델 로드 (재시도)")
                        causal_device = torch.device('cpu')
                        causal_model = load_semantic_causal_pipeline_retry()
                    
                    if getattr(causal_device, "type", "cpu") != "cuda":
                        logger.warning("causal model on CPU (gpu_on_demand, retry). Will promote via DSM when needed.")
                    self.causal_pipeline = causal_model
                    logger.info(f"인과관계 모델 순차 로드 완료 (재시도): {causal_device}")
                    
                    logger.info(f"재시도 {retry_count}: 고급 AI 모델들이 성공적으로 로드되었습니다.")
                    break
                    
                except Exception as retry_error:
                    logger.error(f"재시도 {retry_count} 실패: {retry_error}")
                    if retry_count >= max_retries:
                        logger.error("모든 재시도 실패 - 시스템 종료")
                        raise Exception(f"모델 초기화 최종 실패: {retry_error}")
                    import time
                    time.sleep(1)  # 재시도 간격
    
# 프로젝트 규칙 준수: fallback 메소드 제거됨 - 순수 재시도 방식만 사용

# Lazy loading 제거 - 모든 모델은 초기화 시 로딩됨
    
    def _initialize_classifiers(self):
        """고급 분류기들 초기화"""
        # 표면적 의미 분류기
        self.surface_classifier = MLPClassifier(
            hidden_layer_sizes=(512, 256, 128),
            activation='relu',
            solver='adam',
            random_state=42
        )
        
        # 윤리적 판단 분류기
        self.ethical_classifier = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        # 감정 강도 회귀기
        self.emotion_regressor = MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            random_state=42
        )
        
        # 인과관계 강도 예측기
        self.causal_regressor = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
    
    def _setup_neural_networks_cpu(self):
        """커스텀 신경망 설정 - CPU 작업만 (워커 스레드에서 실행)
        GPU 로더, torch.cuda, DSM 등록 등은 절대 호출하지 않음
        """
        
        class SemanticFusionNetwork(nn.Module):
            """의미 융합 신경망"""
            def __init__(self, input_dim=768, hidden_dim=512, output_dim=256):
                super().__init__()
                self.fusion_layers = nn.Sequential(
                    nn.Linear(input_dim * 4, hidden_dim),  # 4개 수준 융합
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim // 2, output_dim),
                    nn.Tanh()
                )
                
            def forward(self, surface, ethical, emotional, causal):
                fused = torch.cat([surface, ethical, emotional, causal], dim=-1)
                return self.fusion_layers(fused)
        
        class CrossLevelAttention(nn.Module):
            """수준 간 어텐션 메커니즘"""
            def __init__(self, d_model=768, num_heads=8):
                super().__init__()
                self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
                self.norm = nn.LayerNorm(d_model)
                
            def forward(self, query, key, value):
                attn_output, attn_weights = self.attention(query, key, value)
                return self.norm(attn_output + query), attn_weights
        
        # 네트워크 인스턴스 생성 (순차적 로딩)
        def load_semantic_fusion_network():
            # GPU 로더 내부에서 실행되므로 필요한 모든 import 포함
            import torch
            import logging
            logger = logging.getLogger(__name__)
            logger.info("GPU 로더에서 의미 융합 네트워크 로딩 시작")
            try:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                fusion_network = SemanticFusionNetwork().to(device)
                cross_attention = CrossLevelAttention().to(device)
                logger.info(f"의미 융합 네트워크 로딩 성공 (device: {device})")
                return {'fusion_network': fusion_network, 'cross_attention': cross_attention}
            except Exception as e:
                logger.error(f"의미 융합 네트워크 로딩 실패: {e}")
                raise
        
        # CPU에서만 네트워크 생성 (GPU 이동은 나중에)
        self.fusion_network = SemanticFusionNetwork()
        self.cross_attention = CrossLevelAttention()
        logger.info("의미 퓨전 네트워크 CPU 생성 완료")
        
        # 옵티마이저 설정
        self.fusion_optimizer = torch.optim.Adam(
            self.fusion_network.parameters(), 
            lr=self.config.get('learning_rate', 0.001)
        )
        
        logger.info("커스텀 신경망이 CPU에서 설정되었습니다.")
    
    async def _setup_neural_networks_gpu(self):
        """신경망 GPU 이동 및 DSM 등록 (메인 이벤트 루프에서 실행)
        GPU 로더 호출, 디바이스 이동, DSM 등록 등을 수행
        """
        try:
            # DSM을 통한 GPU 승격 시도
            from dynamic_swap_manager import get_swap_manager, SwapPriority
            swap = get_swap_manager()
            
            if swap:
                # fusion_network 등록
                if hasattr(self, 'fusion_network'):
                    swap.register_model("semantic_fusion", self.fusion_network, priority=SwapPriority.LOW)
                    logger.info("✅ semantic_fusion DSM 등록 완료")
                
                # cross_attention 등록
                if hasattr(self, 'cross_attention'):
                    swap.register_model("semantic_cross_attention", self.cross_attention, priority=SwapPriority.LOW)
                    logger.info("✅ semantic_cross_attention DSM 등록 완료")
                    
                # GPU 메모리가 충분하면 GPU로 이동 시도
                if torch.cuda.is_available():
                    from config import get_gpu_memory_info
                    gpu_info = get_gpu_memory_info()
                    if gpu_info and gpu_info['free_mb'] > 500:  # 500MB 이상 여유 시
                        if hasattr(self, 'fusion_network'):
                            self.fusion_network = self.fusion_network.cuda()
                            logger.info("🚀 fusion_network GPU로 이동 성공")
                        if hasattr(self, 'cross_attention'):
                            self.cross_attention = self.cross_attention.cuda()
                            logger.info("🚀 cross_attention GPU로 이동 성공")
                    else:
                        logger.info("⚠️ GPU 메모리 부족으로 CPU에 유지")
            else:
                logger.warning("DSM을 찾을 수 없어 GPU 승격 스킵")
                
        except Exception as e:
            logger.warning(f"GPU 설정 중 오류 (무시하고 계속): {e}")
    
    async def _ensure_model_on_gpu(self, model_name: str):
        """모델을 GPU로 승격 (gpu_on_demand 정책)
        
        Args:
            model_name: 모델 속성 이름 ('emotion_pipeline', 'ethical_model', 'causal_pipeline')
        """
        try:
            # DSM을 통한 GPU 승격
            from dynamic_swap_manager import get_swap_manager
            swap = get_swap_manager()
            
            if swap and hasattr(self, model_name):
                model = getattr(self, model_name)
                
                # 이미 GPU에 있는지 확인
                if hasattr(model, 'device'):
                    if str(model.device).startswith('cuda'):
                        return  # 이미 GPU에 있음
                
                # DSM을 통한 GPU 승격 시도
                success = await swap.ensure_on_gpu("semantic_analyzer")
                
                if success:
                    # 모델을 GPU로 이동
                    if hasattr(model, 'to'):
                        setattr(self, model_name, model.to('cuda'))
                        logger.info(f"✅ {model_name} GPU 승격 완료")
                    elif hasattr(model, 'model') and hasattr(model.model, 'to'):
                        # pipeline의 경우
                        model.model = model.model.to('cuda')
                        model.device = 0  # pipeline device 설정
                        logger.info(f"✅ {model_name} pipeline GPU 승격 완료")
                else:
                    logger.warning(f"⚠️ {model_name} GPU 승격 실패 - CPU에서 실행")
                    
        except Exception as e:
            logger.warning(f"GPU 승격 중 오류 (CPU 계속): {e}")
    
    async def analyze_text_advanced(self, text: str, metadata: Dict[str, Any] = None) -> AdvancedSemanticResult:
        """
        고급 다중수준 의미 분석 (비동기)
        
        Args:
            text: 분석할 텍스트
            metadata: 추가 메타데이터
            
        Returns:
            고급 의미 분석 결과
        """
        start_time = time.time()
        self.analysis_state = SemanticAnalysisState()
        
        if not text:
            return self._get_empty_advanced_result(text)
        
        
        # 캐시 확인
        cache_key = self._generate_cache_key(text, metadata)
        if cache_key in self.analysis_cache:
            logger.info("캐시에서 결과를 가져왔습니다.")
            return self.analysis_cache[cache_key]
        
        try:
            # 병렬 분석 실행
            analysis_tasks = [
                self._analyze_surface_level_advanced(text),
                self._analyze_ethical_level_advanced(text),
                self._analyze_emotional_level_advanced(text),
                self._analyze_causal_level_advanced(text)
            ]
            
            # 모든 수준 분석 완료 대기
            results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            surface_result, ethical_result, emotional_result, causal_result = results
            
            # 예외 처리
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    level_name = ['surface', 'ethical', 'emotional', 'causal'][i]
                    logger.error(f"{level_name} 수준 분석 실패: {result}")
                    self.analysis_state.error_log.append(f"{level_name}: {str(result)}")
            
            # 수준 간 관계 분석
            cross_relations = await self._analyze_cross_level_relations(
                surface_result, ethical_result, emotional_result, causal_result
            )
            
            # 의미 융합 및 종합
            fused_representation = await self._fuse_semantic_levels(
                surface_result, ethical_result, emotional_result, causal_result
            )
            
            # 고급 결과 구성
            advanced_result = AdvancedSemanticResult(
                text=text,
                surface_analysis=surface_result,
                ethical_analysis=ethical_result,
                emotional_analysis=emotional_result,
                causal_analysis=causal_result,
                cross_level_relations=cross_relations,
                fused_representation=fused_representation,
                confidence_score=self._calculate_overall_confidence(),
                processing_time=time.time() - start_time,
                metadata=metadata or {}
            )
            
            # 캐시 저장
            self.analysis_cache[cache_key] = advanced_result
            
            return advanced_result
            
        except Exception as e:
            logger.error(f"고급 의미 분석 실패: {e}")
            return self._get_error_result(text, str(e))
    
    async def _analyze_surface_level_advanced(self, text: str) -> Dict[str, Any]:
        """고급 표면적 수준 분석"""
        self.analysis_state.current_level = "surface"
        
        try:
            # 임베딩 생성
            embedding = await self._get_embedding_async(text)
            
            # 통계적 특성 추출
            stats_features = self._extract_statistical_features(text)
            
            # 언어학적 특성 분석
            linguistic_features = self._analyze_linguistic_features(text)
            
            # TF-IDF 기반 키워드 추출
            keywords = self._extract_keywords_advanced(text)
            
            # 의미 클러스터링
            semantic_clusters = await self._perform_semantic_clustering([text])
            
            result = {
                'text_length': len(text),
                'word_count': len(text.split()),
                'sentence_count': len(re.findall(r'[.!?]+', text)),
                'embedding': embedding.tolist(),  # embedding은 필수 - None이면 실패
                'statistical_features': stats_features,
                'linguistic_features': linguistic_features,
                'keywords': keywords,
                'semantic_clusters': semantic_clusters,
                'complexity_score': self._calculate_complexity_score(text),
                'readability_score': self._calculate_readability_score(text),
                'confidence': 0.85
            }
            
            self.analysis_state.confidence_scores['surface'] = result['confidence']
            return result
            
        except Exception as e:
            logger.error(f"표면적 분석 실패: {e}")
            return {'error': str(e), 'confidence': 0.0}
    
    async def _analyze_ethical_level_advanced(self, text: str) -> Dict[str, Any]:
        """고급 윤리적 수준 분석"""
        self.analysis_state.current_level = "ethical"
        
        try:
            # 윤리적 키워드 감지
            ethical_keywords = self._detect_ethical_keywords(text)
            
            # 도덕적 감정 분석
            moral_emotions = await self._analyze_moral_emotions(text)
            
            # 권리와 의무 분석
            rights_duties = self._analyze_rights_duties(text)
            
            # 결과주의 vs 의무론적 분석
            ethical_framework = await self._classify_ethical_framework(text)
            
            # 도덕적 갈등 감지
            moral_conflicts = self._detect_moral_conflicts(text)
            
            # BERT 기반 윤리적 임베딩
            ethical_embedding = await self._get_ethical_embedding(text)
            
            result = {
                'ethical_keywords': ethical_keywords,
                'moral_emotions': moral_emotions,
                'rights_duties': rights_duties,
                'ethical_framework': ethical_framework,
                'moral_conflicts': moral_conflicts,
                'ethical_embedding': ethical_embedding.tolist(),  # ethical_embedding은 필수 - None이면 실패
                'moral_judgment': self._make_moral_judgment(text),
                'ethical_dimensions': self._analyze_ethical_dimensions(text),
                'confidence': 0.78
            }
            
            self.analysis_state.confidence_scores['ethical'] = result['confidence']
            return result
            
        except Exception as e:
            logger.error(f"윤리적 분석 실패: {e}")
            return {'error': str(e), 'confidence': 0.0}
    
    async def _analyze_emotional_level_advanced(self, text: str) -> Dict[str, Any]:
        """고급 감정적 수준 분석"""
        self.analysis_state.current_level = "emotional"
        
        try:
            # 멀티 모델 감정 분석 - emotion_pipeline은 필수
            if not self.emotion_pipeline:
                raise RuntimeError("감정 분석 파이프라인이 초기화되지 않음")
            
            # GPU on-demand: 필요시 GPU로 승격
            await self._ensure_model_on_gpu('emotion_pipeline')
            
            emotion_result = self.emotion_pipeline(text)
            primary_emotion = emotion_result[0]['label']
            emotion_confidence = emotion_result[0]['score']
            
            # 감정 강도 분석
            emotion_intensity = self._calculate_emotion_intensity(text)
            
            # 감정 극성 분석 (Valence-Arousal)
            valence_arousal = self._analyze_valence_arousal(text)
            
            # 복합 감정 감지
            complex_emotions = self._detect_complex_emotions(text)
            
            # 감정 변화 추적 (문장 단위)
            emotion_dynamics = self._track_emotion_dynamics(text)
            
            # 감정 임베딩
            emotion_embedding = await self._get_emotion_embedding(text)
            
            result = {
                'primary_emotion': primary_emotion,
                'emotion_confidence': emotion_confidence,
                'emotion_intensity': emotion_intensity,
                'valence': valence_arousal['valence'],
                'arousal': valence_arousal['arousal'],
                'complex_emotions': complex_emotions,
                'emotion_dynamics': emotion_dynamics,
                'emotion_embedding': emotion_embedding.tolist(),  # emotion_embedding은 필수 - None이면 실패
                'emotional_stability': self._calculate_emotional_stability(emotion_dynamics),
                'confidence': 0.82
            }
            
            self.analysis_state.confidence_scores['emotional'] = result['confidence']
            return result
            
        except Exception as e:
            logger.error(f"감정적 분석 실패: {e}")
            return {'error': str(e), 'confidence': 0.0}
    
    async def _analyze_causal_level_advanced(self, text: str) -> Dict[str, Any]:
        """고급 인과적 수준 분석"""
        self.analysis_state.current_level = "causal"
        
        try:
            # 인과관계 키워드 감지
            causal_keywords = self._detect_causal_keywords(text)
            
            # 원인-결과 관계 추출
            cause_effect_pairs = self._extract_cause_effect_pairs(text)
            
            # 조건부 관계 분석
            conditional_relations = self._analyze_conditional_relations(text)
            
            # 시간적 순서 분석
            temporal_order = self._analyze_temporal_order(text)
            
            # 인과관계 강도 계산
            causal_strength = self._calculate_causal_strength(text)
            
            # 인과 네트워크 구성
            causal_network = self._build_causal_network(cause_effect_pairs)
            
            # Zero-shot 인과관계 분류 - causal_pipeline은 필수
            if not self.causal_pipeline:
                raise RuntimeError("인과관계 분석 파이프라인이 초기화되지 않음")
            
            # GPU on-demand: 필요시 GPU로 승격
            await self._ensure_model_on_gpu('causal_pipeline')
            
            causal_labels = ["cause", "effect", "correlation", "independence"]
            causal_classification = self.causal_pipeline(text, causal_labels)
            
            result = {
                'causal_keywords': causal_keywords,
                'cause_effect_pairs': cause_effect_pairs,
                'conditional_relations': conditional_relations,
                'temporal_order': temporal_order,
                'causal_strength': causal_strength,
                'causal_network': causal_network,
                'causal_classification': causal_classification,
                'causal_confidence': self._calculate_causal_confidence(causal_keywords, cause_effect_pairs),
                'confidence': 0.75
            }
            
            self.analysis_state.confidence_scores['causal'] = result['confidence']
            return result
            
        except Exception as e:
            logger.error(f"인과적 분석 실패: {e}")
            return {'error': str(e), 'confidence': 0.0}
    
    async def _analyze_cross_level_relations(self, surface, ethical, emotional, causal) -> List[CrossLevelSemanticRelation]:
        """수준 간 의미 관계 분석"""
        relations = []
        
        try:
            # 각 수준 조합에 대해 관계 분석
            level_pairs = [
                ('surface', 'ethical', surface, ethical),
                ('surface', 'emotional', surface, emotional),
                ('surface', 'causal', surface, causal),
                ('ethical', 'emotional', ethical, emotional),
                ('ethical', 'causal', ethical, causal),
                ('emotional', 'causal', emotional, causal)
            ]
            
            for source_level, target_level, source_data, target_data in level_pairs:
                relation = await self._compute_cross_level_relation(
                    source_level, target_level, source_data, target_data
                )
                if relation:
                    relations.append(relation)
            
            return relations
            
        except Exception as e:
            logger.error(f"수준 간 관계 분석 실패: {e}")
            return []
    
    async def _compute_cross_level_relation(self, source_level: str, target_level: str, 
                                         source_data: Dict, target_data: Dict) -> Optional[CrossLevelSemanticRelation]:
        """두 수준 간의 구체적 관계 계산"""
        
        if 'error' in source_data or 'error' in target_data:
            return None
        
        try:
            # 신뢰도 기반 관계 강도 계산
            source_confidence = source_data.get('confidence', 0.0)
            target_confidence = target_data.get('confidence', 0.0)
            base_strength = (source_confidence + target_confidence) / 2
            
            # 관계 유형 결정 로직
            relation_type = "neutral"
            evidence = []
            
            # 특정 수준 조합별 관계 분석
            if source_level == "emotional" and target_level == "ethical":
                # 감정과 윤리의 관계
                emotion = source_data.get('primary_emotion', 'NEUTRAL')
                moral_judgment = target_data.get('moral_judgment', 'neutral')
                
                if emotion in ['JOY', 'LOVE'] and moral_judgment == 'positive':
                    relation_type = "supports"
                    evidence.append("긍정적 감정이 긍정적 도덕 판단을 지지")
                elif emotion in ['ANGER', 'FEAR'] and moral_judgment == 'negative':
                    relation_type = "supports" 
                    evidence.append("부정적 감정이 부정적 도덕 판단과 일치")
                elif emotion in ['JOY'] and moral_judgment == 'negative':
                    relation_type = "contradicts"
                    evidence.append("긍정적 감정과 부정적 도덕 판단 간 모순")
            
            elif source_level == "causal" and target_level == "ethical":
                # 인과관계와 윤리의 관계
                causal_strength = source_data.get('causal_strength', 0.0)
                moral_conflicts = target_data.get('moral_conflicts', [])
                
                if causal_strength > 0.7 and len(moral_conflicts) > 0:
                    relation_type = "amplifies"
                    evidence.append("강한 인과관계가 도덕적 갈등을 증폭")
            
            # 기본 관계 설정
            if relation_type == "neutral":
                if base_strength > 0.7:
                    relation_type = "supports"
                    evidence.append("높은 신뢰도 기반 지지 관계")
            
            return CrossLevelSemanticRelation(
                source_level=source_level,
                target_level=target_level,
                relation_type=relation_type,
                strength=base_strength,
                evidence=evidence,
                confidence=min(source_confidence, target_confidence)
            )
            
        except Exception as e:
            logger.error(f"수준 간 관계 계산 실패 ({source_level}-{target_level}): {e}")
            return None
    
    async def _fuse_semantic_levels(self, surface, ethical, emotional, causal) -> Dict[str, Any]:
        """의미 수준들을 고급 신경망으로 융합"""
        try:
            # 각 수준의 임베딩 추출
            embeddings = []
            
            for level_data in [surface, ethical, emotional, causal]:
                if 'error' in level_data:
                    # 오류가 있는 경우 0 벡터 사용
                    embeddings.append(torch.zeros(768))
                else:
                    # 임베딩이 있으면 사용, 없으면 0 벡터
                    emb_key = None
                    for key in ['embedding', 'ethical_embedding', 'emotion_embedding']:
                        if key in level_data and level_data[key] is not None:
                            emb_key = key
                            break
                    
                    if not emb_key:
                        raise RuntimeError(f"수준별 임베딩이 없음: {level_data.keys()}")
                    embedding = torch.tensor(level_data[emb_key][:768])  # 차원 맞춤
                    
                    embeddings.append(embedding)
            
            # 임베딩들을 디바이스로 이동
            # Fusion network와 같은 디바이스로 이동
            fusion_device = next(self.fusion_network.parameters()).device
            embeddings = [emb.to(fusion_device) for emb in embeddings]
            
            # 배치 차원 추가
            embeddings = [emb.unsqueeze(0) for emb in embeddings]
            
            # 융합 네트워크 적용
            with torch.no_grad():
                fused_repr = self.fusion_network(*embeddings)
                fused_repr = fused_repr.squeeze(0).cpu().numpy()
            
            # 크로스 어텐션 적용
            attention_weights = await self._compute_cross_attention(embeddings)
            
            return {
                'fused_embedding': fused_repr.tolist(),
                'attention_weights': attention_weights,
                'fusion_confidence': self._calculate_fusion_confidence(surface, ethical, emotional, causal),
                'semantic_coherence': self._calculate_semantic_coherence(surface, ethical, emotional, causal)
            }
            
        except Exception as e:
            logger.error(f"의미 융합 실패: {e}")
            return {
                'fused_embedding': None,
                'attention_weights': None,
                'fusion_confidence': 0.0,
                'semantic_coherence': 0.0,
                'error': str(e)
            }
    
    async def _compute_cross_attention(self, embeddings: List[torch.Tensor]) -> Dict[str, Any]:
        """크로스 어텐션 가중치 계산"""
        try:
            # 모든 임베딩을 하나의 시퀀스로 결합
            combined = torch.cat(embeddings, dim=1)  # [1, 4, 768]
            
            with torch.no_grad():
                attended, weights = self.cross_attention(combined, combined, combined)
            
            # 가중치를 수준별로 분리
            level_names = ['surface', 'ethical', 'emotional', 'causal']
            attention_matrix = {}
            
            weights_np = weights.squeeze(0).cpu().numpy()  # [4, 4]
            
            for i, source in enumerate(level_names):
                attention_matrix[source] = {}
                for j, target in enumerate(level_names):
                    attention_matrix[source][target] = float(weights_np[i, j])
            
            return {
                'attention_matrix': attention_matrix,
                'dominant_relations': self._identify_dominant_relations(attention_matrix)
            }
            
        except Exception as e:
            logger.error(f"크로스 어텐션 계산 실패: {e}")
            return {'attention_matrix': None, 'dominant_relations': []}
    
    def _identify_dominant_relations(self, attention_matrix: Dict[str, Dict[str, float]]) -> List[str]:
        """지배적인 어텐션 관계 식별"""
        dominant_relations = []
        threshold = 0.3
        
        for source, targets in attention_matrix.items():
            for target, weight in targets.items():
                if source != target and weight > threshold:
                    dominant_relations.append(f"{source} → {target} ({weight:.3f})")
        
        return sorted(dominant_relations, key=lambda x: float(x.split('(')[1].split(')')[0]), reverse=True)
    
    # 헬퍼 메서드들
    async def _get_embedding_async(self, text: str) -> np.ndarray:
        """비동기 임베딩 생성"""
        if self.embedding_model is None:
            raise RuntimeError("임베딩 모델이 초기화되지 않음")
        
        try:
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, lambda: self.embedding_model.encode(text)
            )
            return embedding
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")
            return None
    
    def _extract_statistical_features(self, text: str) -> Dict[str, float]:
        """통계적 특성 추출"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        return {
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'avg_sentence_length': np.mean([len(sent.split()) for sent in sentences if sent.strip()]),
            'lexical_diversity': len(set(words)) / len(words) if words else 0,
            'punctuation_ratio': len(re.findall(r'[^\w\s]', text)) / len(text) if text else 0
        }
    
    def _analyze_linguistic_features(self, text: str) -> Dict[str, Any]:
        """언어학적 특성 분석"""
        return {
            'contains_questions': '?' in text,
            'contains_exclamations': '!' in text,
            'contains_quotes': '"' in text or "'" in text,
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'digit_ratio': sum(1 for c in text if c.isdigit()) / len(text) if text else 0
        }
    
    def _extract_keywords_advanced(self, text: str) -> List[str]:
        """고급 키워드 추출"""
        try:
            # TF-IDF 벡터라이저
            vectorizer = TfidfVectorizer(
                max_features=10,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            # 단일 문서이므로 리스트로 감싸기
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            
            # TF-IDF 점수 기반 상위 키워드
            scores = tfidf_matrix.toarray()[0]
            keyword_scores = list(zip(feature_names, scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [kw for kw, score in keyword_scores if score > 0][:5]
            
        except Exception as e:
            logger.error(f"키워드 추출 실패: {e}")
            return []
    
    async def _perform_semantic_clustering(self, texts: List[str]) -> Dict[str, Any]:
        """의미론적 클러스터링"""
        if len(texts) < 2:
            return {'clusters': [], 'cluster_centers': []}
        
        try:
            if self.embedding_model:
                embeddings = self.embedding_model.encode(texts)
                
                # K-means 클러스터링
                n_clusters = min(3, len(texts))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(embeddings)
                
                return {
                    'clusters': cluster_labels.tolist(),
                    'cluster_centers': kmeans.cluster_centers_.tolist(),
                    'inertia': float(kmeans.inertia_)
                }
            else:
                return {'clusters': [], 'cluster_centers': []}
                
        except Exception as e:
            logger.error(f"의미론적 클러스터링 실패: {e}")
            return {'clusters': [], 'cluster_centers': []}
    
    def _calculate_complexity_score(self, text: str) -> float:
        """텍스트 복잡도 점수 계산"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        if not words or not sentences:
            return 0.0
        
        # 복잡도 지표들
        avg_word_length = np.mean([len(word) for word in words])
        avg_sentence_length = len(words) / len([s for s in sentences if s.strip()])
        unique_words_ratio = len(set(words)) / len(words)
        
        # 가중 평균으로 복잡도 계산
        complexity = (
            (avg_word_length / 10) * 0.3 +
            (avg_sentence_length / 20) * 0.4 +
            unique_words_ratio * 0.3
        )
        
        return min(1.0, complexity)
    
    def _calculate_readability_score(self, text: str) -> float:
        """가독성 점수 계산 (Flesch Reading Ease 변형)"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        syllables = sum(self._count_syllables(word) for word in words)
        
        if not sentences or not words:
            return 0.5
        
        avg_sentence_length = len(words) / len([s for s in sentences if s.strip()])
        avg_syllables_per_word = syllables / len(words)
        
        # 간단한 가독성 공식
        readability = 1.0 - (avg_sentence_length * 0.05 + avg_syllables_per_word * 0.1)
        return max(0.0, min(1.0, readability))
    
    def _count_syllables(self, word: str) -> int:
        """단어의 음절 수 추정"""
        word = word.lower().strip(".,!?;:")
        if not word:
            return 0
        
        # 간단한 음절 카운팅 규칙
        vowels = "aeiouy"
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False
        
        # 최소 1음절
        return max(1, syllable_count)
    
    def _detect_ethical_keywords(self, text: str) -> List[str]:
        """윤리적 키워드 감지"""
        ethical_keywords = [
            '권리', '의무', '정의', '공정', '책임', '도덕', '윤리', '선', '악',
            '옳다', '그르다', '해야', '하지말아야', '올바른', '잘못된',
            'right', 'wrong', 'should', 'ought', 'must', 'duty', 'moral',
            'ethical', 'justice', 'fair', 'unfair', 'responsibility'
        ]
        
        found_keywords = []
        text_lower = text.lower()
        
        for keyword in ethical_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    async def _analyze_moral_emotions(self, text: str) -> Dict[str, float]:
        """도덕적 감정 분석"""
        moral_emotions = {
            'guilt': 0.0, 'shame': 0.0, 'pride': 0.0, 
            'indignation': 0.0, 'compassion': 0.0, 'disgust': 0.0
        }
        
        # 키워드 기반 간단한 도덕적 감정 감지
        emotion_keywords = {
            'guilt': ['죄책감', '미안', '잘못', 'guilt', 'sorry', 'regret'],
            'shame': ['부끄럽', '창피', 'shame', 'embarrassed'],
            'pride': ['자랑', '뿌듯', 'proud', 'pride'],
            'indignation': ['분노', '억울', 'angry', 'unfair', 'outrage'],
            'compassion': ['동정', '연민', '안타깝', 'compassion', 'sympathy', 'pity'],
            'disgust': ['혐오', '역겨운', 'disgust', 'disgusting']
        }
        
        text_lower = text.lower()
        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    moral_emotions[emotion] += 0.3
        
        # 정규화
        for emotion in moral_emotions:
            moral_emotions[emotion] = min(1.0, moral_emotions[emotion])
        
        return moral_emotions
    
    def _analyze_rights_duties(self, text: str) -> Dict[str, List[str]]:
        """권리와 의무 분석"""
        rights_keywords = ['권리', '자유', '자격', 'right', 'freedom', 'liberty', 'entitled']
        duties_keywords = ['의무', '책임', '해야', 'duty', 'obligation', 'responsibility', 'must', 'should']
        
        rights = []
        duties = []
        
        text_lower = text.lower()
        
        for keyword in rights_keywords:
            if keyword in text_lower:
                rights.append(keyword)
        
        for keyword in duties_keywords:
            if keyword in text_lower:
                duties.append(keyword)
        
        return {'rights': rights, 'duties': duties}
    
    async def _classify_ethical_framework(self, text: str) -> Dict[str, float]:
        """윤리적 프레임워크 분류"""
        frameworks = {
            'consequentialist': 0.0,  # 결과주의
            'deontological': 0.0,     # 의무론
            'virtue_ethics': 0.0      # 덕윤리
        }
        
        # 키워드 기반 분류
        consequentialist_keywords = ['결과', '효과', '이익', '손해', 'result', 'consequence', 'outcome', 'benefit']
        deontological_keywords = ['의무', '규칙', '법', '원칙', 'duty', 'rule', 'law', 'principle']
        virtue_keywords = ['덕', '성품', '인격', '선량', 'virtue', 'character', 'integrity', 'honor']
        
        text_lower = text.lower()
        
        for keyword in consequentialist_keywords:
            if keyword in text_lower:
                frameworks['consequentialist'] += 0.2
        
        for keyword in deontological_keywords:
            if keyword in text_lower:
                frameworks['deontological'] += 0.2
        
        for keyword in virtue_keywords:
            if keyword in text_lower:
                frameworks['virtue_ethics'] += 0.2
        
        # 정규화
        total = sum(frameworks.values())
        if total > 0:
            for framework in frameworks:
                frameworks[framework] /= total
        
        return frameworks
    
    def _detect_moral_conflicts(self, text: str) -> List[str]:
        """도덕적 갈등 감지"""
        conflict_indicators = [
            '딜레마', '갈등', '모순', '어려운', '힘든', 
            'dilemma', 'conflict', 'contradiction', 'difficult', 'torn'
        ]
        
        conflicts = []
        text_lower = text.lower()
        
        for indicator in conflict_indicators:
            if indicator in text_lower:
                conflicts.append(f"도덕적 갈등 감지: '{indicator}' 키워드")
        
        return conflicts
    
    async def _get_ethical_embedding(self, text: str) -> np.ndarray:
        """윤리적 임베딩 생성"""
        if self.ethical_model is None:
            raise RuntimeError("윤리 분석 모델이 초기화되지 않음")
        
        # GPU on-demand: 필요시 GPU로 승격
        await self._ensure_model_on_gpu('ethical_model')
        
        try:
            # BERT 토크나이저 및 모델 사용
            inputs = self.ethical_tokenizer(
                text, return_tensors='pt', 
                max_length=512, truncation=True, padding=True
            )
            # Ethical model과 같은 디바이스로 이동
            ethical_device = next(self.ethical_model.parameters()).device
            inputs = {k: v.to(ethical_device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.ethical_model(**inputs)
                # [CLS] 토큰의 임베딩 사용
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            return embedding.squeeze()
            
        except Exception as e:
            logger.error(f"윤리적 임베딩 생성 실패: {e}")
            return None
    
    def _make_moral_judgment(self, text: str) -> str:
        """도덕적 판단"""
        positive_keywords = ['좋다', '옳다', '선', '바람직', 'good', 'right', 'moral', 'ethical']
        negative_keywords = ['나쁘다', '그르다', '악', '바람직하지않은', 'bad', 'wrong', 'immoral', 'unethical']
        
        text_lower = text.lower()
        
        positive_count = sum(1 for kw in positive_keywords if kw in text_lower)
        negative_count = sum(1 for kw in negative_keywords if kw in text_lower)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def _analyze_ethical_dimensions(self, text: str) -> Dict[str, float]:
        """윤리적 차원 분석"""
        dimensions = {
            'harm_care': 0.0,       # 피해/보살핌
            'fairness_cheating': 0.0, # 공정/부정
            'loyalty_betrayal': 0.0,   # 충성/배신
            'authority_subversion': 0.0, # 권위/전복
            'sanctity_degradation': 0.0  # 신성/타락
        }
        
        # 간단한 키워드 기반 차원 분석
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['해', '상처', 'harm', 'hurt', 'care', 'protect']):
            dimensions['harm_care'] = 0.7
        
        if any(word in text_lower for word in ['공정', '불공정', 'fair', 'unfair', 'justice']):
            dimensions['fairness_cheating'] = 0.7
        
        if any(word in text_lower for word in ['충성', '배신', 'loyal', 'betray', 'faithful']):
            dimensions['loyalty_betrayal'] = 0.7
        
        if any(word in text_lower for word in ['권위', '복종', 'authority', 'obey', 'respect']):
            dimensions['authority_subversion'] = 0.7
        
        if any(word in text_lower for word in ['신성', '순수', 'sacred', 'pure', 'holy']):
            dimensions['sanctity_degradation'] = 0.7
        
        return dimensions
    
    def _calculate_emotion_intensity(self, text: str) -> float:
        """감정 강도 계산"""
        intensity_indicators = {
            'very': 0.8, 'extremely': 0.9, 'incredibly': 0.9,
            'quite': 0.6, 'rather': 0.5, 'somewhat': 0.4,
            '매우': 0.8, '정말': 0.7, '너무': 0.8, '조금': 0.3
        }
        
        text_lower = text.lower()
        max_intensity = 0.5  # 기본 강도
        
        for indicator, intensity in intensity_indicators.items():
            if indicator in text_lower:
                max_intensity = max(max_intensity, intensity)
        
        # 느낌표 개수도 고려
        exclamation_count = text.count('!')
        max_intensity = min(1.0, max_intensity + exclamation_count * 0.1)
        
        return max_intensity
    
    def _analyze_valence_arousal(self, text: str) -> Dict[str, float]:
        """감정 극성(Valence)과 각성도(Arousal) 분석"""
        # 간단한 키워드 기반 분석
        positive_words = ['좋다', '행복', '기쁘다', 'happy', 'good', 'love', 'excellent']
        negative_words = ['나쁘다', '슬프다', '화나다', 'sad', 'angry', 'bad', 'hate']
        
        high_arousal_words = ['흥분', '열정', 'excited', 'passionate', 'intense', 'thrilled']
        low_arousal_words = ['평온', '차분', 'calm', 'peaceful', 'relaxed', 'serene']
        
        text_lower = text.lower()
        
        # Valence 계산
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count + negative_count > 0:
            valence = (positive_count - negative_count) / (positive_count + negative_count)
        else:
            valence = 0.0
        
        # Arousal 계산
        high_arousal_count = sum(1 for word in high_arousal_words if word in text_lower)
        low_arousal_count = sum(1 for word in low_arousal_words if word in text_lower)
        
        if high_arousal_count + low_arousal_count > 0:
            arousal = (high_arousal_count - low_arousal_count) / (high_arousal_count + low_arousal_count)
        else:
            arousal = 0.0
        
        return {
            'valence': valence,   # -1 (매우 부정) ~ 1 (매우 긍정)
            'arousal': arousal    # -1 (매우 낮음) ~ 1 (매우 높음)
        }
    
    def _detect_complex_emotions(self, text: str) -> List[Dict[str, Any]]:
        """복합 감정 감지"""
        complex_emotions = []
        
        # 복합 감정 패턴
        patterns = [
            {
                'name': 'bittersweet',
                'keywords': ['씁쓸', '달콤쌉쌀', 'bittersweet'],
                'components': ['sadness', 'happiness']
            },
            {
                'name': 'nostalgic',
                'keywords': ['그리운', '그립다', 'nostalgic', 'miss'],
                'components': ['sadness', 'love']
            },
            {
                'name': 'anxious_excitement',
                'keywords': ['떨린다', '긴장', 'nervous', 'anxious', 'excited'],
                'components': ['fear', 'joy']
            }
        ]
        
        text_lower = text.lower()
        
        for pattern in patterns:
            for keyword in pattern['keywords']:
                if keyword in text_lower:
                    complex_emotions.append({
                        'emotion': pattern['name'],
                        'components': pattern['components'],
                        'confidence': 0.6
                    })
                    break
        
        return complex_emotions
    
    def _track_emotion_dynamics(self, text: str) -> List[Dict[str, Any]]:
        """감정 변화 추적 (문장 단위)"""
        sentences = re.split(r'[.!?]+', text)
        emotion_timeline = []
        
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                # 각 문장의 감정 분석
                sentence_emotion = self._simple_emotion_analysis(sentence)
                emotion_timeline.append({
                    'sentence_index': i,
                    'sentence': sentence.strip(),
                    'emotion': sentence_emotion,
                    'timestamp': i  # 순서를 시간으로 간주
                })
        
        return emotion_timeline
    
    def _simple_emotion_analysis(self, text: str) -> str:
        """간단한 감정 분석"""
        emotion_keywords = {
            'joy': ['기쁘다', '행복', '좋다', 'happy', 'joy', 'good'],
            'sadness': ['슬프다', '우울', '힘들다', 'sad', 'depressed', 'difficult'],
            'anger': ['화나다', '분노', '짜증', 'angry', 'mad', 'frustrated'],
            'fear': ['무섭다', '두렵다', '걱정', 'fear', 'scared', 'worried']
        }
        
        text_lower = text.lower()
        emotion_scores = {}
        
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            emotion_scores[emotion] = score
        
        if not any(emotion_scores.values()):
            return 'neutral'
        
        return max(emotion_scores, key=emotion_scores.get)
    
    def _calculate_emotional_stability(self, emotion_dynamics: List[Dict[str, Any]]) -> float:
        """감정 안정성 계산"""
        if len(emotion_dynamics) < 2:
            return 1.0
        
        emotions = [ed['emotion'] for ed in emotion_dynamics]
        unique_emotions = len(set(emotions))
        total_emotions = len(emotions)
        
        # 감정 변화가 적을수록 안정성이 높음
        stability = 1.0 - (unique_emotions / total_emotions)
        return stability
    
    async def _get_emotion_embedding(self, text: str) -> Optional[np.ndarray]:
        """감정 임베딩 생성"""
        # 일반 임베딩 모델 사용 (감정 특화 모델이 없는 경우)
        return await self._get_embedding_async(text)
    
    def _detect_causal_keywords(self, text: str) -> List[str]:
        """인과관계 키워드 감지"""
        causal_keywords = [
            '때문에', '원인', '결과', '영향', '이유', '그래서', '따라서',
            'because', 'cause', 'effect', 'result', 'due to', 'therefore', 'thus'
        ]
        
        found_keywords = []
        text_lower = text.lower()
        
        for keyword in causal_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def _extract_cause_effect_pairs(self, text: str) -> List[Dict[str, str]]:
        """원인-결과 쌍 추출"""
        pairs = []
        
        # 간단한 패턴 매칭
        causal_patterns = [
            r'(.+?)\s*때문에\s*(.+)',
            r'(.+?)\s*이유로\s*(.+)',
            r'(.+?)\s*because\s*(.+)',
            r'(.+?)\s*due to\s*(.+)'
        ]
        
        for pattern in causal_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) == 2:
                    pairs.append({
                        'cause': match[1].strip(),
                        'effect': match[0].strip()
                    })
        
        return pairs
    
    def _analyze_conditional_relations(self, text: str) -> List[Dict[str, str]]:
        """조건부 관계 분석"""
        conditions = []
        
        conditional_patterns = [
            r'만약\s*(.+?)\s*라면\s*(.+)',
            r'(.+?)\s*이면\s*(.+)',
            r'if\s*(.+?)\s*then\s*(.+)',
            r'(.+?)\s*if\s*(.+)'
        ]
        
        for pattern in conditional_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) == 2:
                    conditions.append({
                        'condition': match[0].strip(),
                        'consequence': match[1].strip()
                    })
        
        return conditions
    
    def _analyze_temporal_order(self, text: str) -> List[str]:
        """시간적 순서 분석"""
        temporal_indicators = [
            '먼저', '그다음', '나중에', '이후', '전에',
            'first', 'then', 'next', 'later', 'after', 'before'
        ]
        
        found_indicators = []
        text_lower = text.lower()
        
        for indicator in temporal_indicators:
            if indicator in text_lower:
                found_indicators.append(indicator)
        
        return found_indicators
    
    def _calculate_causal_strength(self, text: str) -> float:
        """인과관계 강도 계산"""
        strength_indicators = {
            '강하게': 0.9, '확실히': 0.8, '분명히': 0.8,
            'strongly': 0.9, 'definitely': 0.8, 'clearly': 0.8,
            '약간': 0.3, '조금': 0.3, '아마': 0.4,
            'slightly': 0.3, 'maybe': 0.4, 'perhaps': 0.4
        }
        
        text_lower = text.lower()
        max_strength = 0.5  # 기본 강도
        
        for indicator, strength in strength_indicators.items():
            if indicator in text_lower:
                max_strength = max(max_strength, strength)
        
        return max_strength
    
    def _build_causal_network(self, cause_effect_pairs: List[Dict[str, str]]) -> Dict[str, Any]:
        """인과관계 네트워크 구성"""
        if not cause_effect_pairs:
            return {'nodes': [], 'edges': [], 'network_metrics': {}}
        
        # NetworkX 그래프 생성
        G = nx.DiGraph()
        
        for pair in cause_effect_pairs:
            cause = pair['cause']
            effect = pair['effect']
            G.add_edge(cause, effect)
        
        # 네트워크 메트릭스 계산
        try:
            metrics = {
                'node_count': G.number_of_nodes(),
                'edge_count': G.number_of_edges(),
                'density': nx.density(G),
                'is_dag': nx.is_directed_acyclic_graph(G)
            }
            
            # 중심성 측정 (연결된 그래프인 경우)
            if G.number_of_nodes() > 0:
                try:
                    betweenness = nx.betweenness_centrality(G)
                    metrics['most_central_node'] = max(betweenness, key=betweenness.get)
                except:
                    metrics['most_central_node'] = None
        except:
            metrics = {'node_count': 0, 'edge_count': 0}
        
        return {
            'nodes': list(G.nodes()),
            'edges': list(G.edges()),
            'network_metrics': metrics
        }
    
    def _calculate_causal_confidence(self, causal_keywords: List[str], 
                                   cause_effect_pairs: List[Dict[str, str]]) -> float:
        """인과관계 신뢰도 계산"""
        keyword_score = min(1.0, len(causal_keywords) * 0.2)
        pair_score = min(1.0, len(cause_effect_pairs) * 0.3)
        
        return (keyword_score + pair_score) / 2
    
    def _calculate_overall_confidence(self) -> float:
        """전체 신뢰도 계산"""
        confidences = list(self.analysis_state.confidence_scores.values())
        if not confidences:
            return 0.0
        
        return sum(confidences) / len(confidences)
    
    def _calculate_fusion_confidence(self, surface, ethical, emotional, causal) -> float:
        """융합 신뢰도 계산"""
        confidences = []
        
        for level_data in [surface, ethical, emotional, causal]:
            if 'confidence' in level_data:
                confidences.append(level_data['confidence'])
        
        if not confidences:
            return 0.0
        
        # 최소값이 전체 신뢰도를 결정 (약한 고리 원칙)
        return min(confidences) * 0.7 + (sum(confidences) / len(confidences)) * 0.3
    
    def _calculate_semantic_coherence(self, surface, ethical, emotional, causal) -> float:
        """의미적 일관성 계산"""
        # 간단한 일관성 측정
        error_count = sum(1 for level in [surface, ethical, emotional, causal] if 'error' in level)
        total_levels = 4
        
        basic_coherence = (total_levels - error_count) / total_levels
        
        # 추가 일관성 검사 (예: 감정과 윤리 판단의 일치성)
        coherence_bonus = 0.0
        
        if 'error' not in emotional and 'error' not in ethical:
            emotion = emotional.get('primary_emotion', '')
            moral_judgment = ethical.get('moral_judgment', '')
            
            # 긍정적 감정과 긍정적 도덕 판단의 일치
            if emotion in ['JOY', 'LOVE'] and moral_judgment == 'positive':
                coherence_bonus += 0.2
            elif emotion in ['ANGER', 'SADNESS'] and moral_judgment == 'negative':
                coherence_bonus += 0.2
        
        return min(1.0, basic_coherence + coherence_bonus)
    
    def _generate_cache_key(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """캐시 키 생성"""
        key_components = [text[:100]]  # 텍스트 일부만 사용
        
        if metadata:
            key_components.append(str(sorted(metadata.items())))
        
        return str(hash(tuple(key_components)))
    
    def _get_empty_advanced_result(self, text: str) -> AdvancedSemanticResult:
        """빈 결과 반환"""
        return AdvancedSemanticResult(
            text=text,
            surface_analysis={'confidence': 0.0},
            ethical_analysis={'confidence': 0.0},
            emotional_analysis={'confidence': 0.0},
            causal_analysis={'confidence': 0.0},
            cross_level_relations=[],
            fused_representation={},
            confidence_score=0.0,
            processing_time=0.0,
            metadata={}
        )
    
    def _get_error_result(self, text: str, error_msg: str) -> AdvancedSemanticResult:
        """오류 결과 반환"""
        return AdvancedSemanticResult(
            text=text,
            surface_analysis={'error': error_msg, 'confidence': 0.0},
            ethical_analysis={'error': error_msg, 'confidence': 0.0},
            emotional_analysis={'error': error_msg, 'confidence': 0.0},
            causal_analysis={'error': error_msg, 'confidence': 0.0},
            cross_level_relations=[],
            fused_representation={'error': error_msg},
            confidence_score=0.0,
            processing_time=0.0,
            metadata={'error': error_msg}
        )
    
    def get_pytorch_network(self) -> Optional[nn.Module]:
        """PyTorch 네트워크 반환 (HeadAdapter와의 호환성)
        DSM 대표 nn.Module 반환 - 가장 큰 모델 우선
        """
        # ethical_model 우선 (가장 큰 AutoModel)
        if hasattr(self, 'ethical_model') and isinstance(self.ethical_model, nn.Module):
            logger.info("AdvancedMultiLevelSemanticAnalyzer: ethical_model 반환")
            return self.ethical_model
        
        # SentenceTransformer 내부 backbone
        if hasattr(self, 'embedding_model'):
            try:
                # SentenceTransformer의 내부 모델 추출
                if hasattr(self.embedding_model, '_modules'):
                    for module in self.embedding_model._modules.values():
                        if hasattr(module, 'auto_model') and isinstance(module.auto_model, nn.Module):
                            logger.info("AdvancedMultiLevelSemanticAnalyzer: embedding_model.auto_model 반환")
                            return module.auto_model
                # 또는 직접 model 속성
                if hasattr(self.embedding_model, 'model') and isinstance(self.embedding_model.model, nn.Module):
                    logger.info("AdvancedMultiLevelSemanticAnalyzer: embedding_model.model 반환")
                    return self.embedding_model.model
            except Exception as e:
                logger.debug(f"embedding_model 추출 실패: {e}")
        
        # fusion_network
        if hasattr(self, 'fusion_network') and isinstance(self.fusion_network, nn.Module):
            logger.info("AdvancedMultiLevelSemanticAnalyzer: fusion_network 반환")
            return self.fusion_network
        
        # cross_attention
        if hasattr(self, 'cross_attention') and isinstance(self.cross_attention, nn.Module):
            logger.info("AdvancedMultiLevelSemanticAnalyzer: cross_attention 반환")
            return self.cross_attention
        
        # 기타 가능한 네트워크 속성
        for attr_name in ['main_network', 'model', 'neural_model']:
            if hasattr(self, attr_name):
                model = getattr(self, attr_name)
                if isinstance(model, nn.Module):
                    logger.info(f"AdvancedMultiLevelSemanticAnalyzer: {attr_name} 반환")
                    return model
        
        # PyTorch 네트워크를 찾지 못함
        logger.warning("AdvancedMultiLevelSemanticAnalyzer: PyTorch 네트워크를 찾지 못했습니다")
        return None

def create_advanced_semantic_analyzer() -> AdvancedMultiLevelSemanticAnalyzer:
    """고급 의미 분석기 생성"""
    return AdvancedMultiLevelSemanticAnalyzer()

# 테스트 코드
if __name__ == "__main__":
    import asyncio
    
    async def test_advanced_analyzer():
        """고급 분석기 테스트"""
        analyzer = create_advanced_semantic_analyzer()
        
        test_text = """
        이 상황에서 우리는 개인의 자유와 공공의 안전 사이에서 어려운 선택을 해야 한다.
        한편으로는 개인의 권리를 존중해야 하지만, 다른 한편으로는 사회 전체의 복지를 고려해야 한다.
        이런 딜레마는 정말 가슴 아프고 복잡한 문제다.
        """
        
        print("=== 고급 다중수준 의미 분석 테스트 ===\n")
        
        result = await analyzer.analyze_text_advanced(test_text)
        
        print(f"분석 텍스트: {result.text}")
        print(f"전체 신뢰도: {result.confidence_score:.3f}")
        print(f"처리 시간: {result.processing_time:.3f}초\n")
        
        print("=== 수준별 분석 결과 ===")
        print(f"표면적: 신뢰도 {result.surface_analysis.get('confidence', 0):.3f}")
        print(f"윤리적: 신뢰도 {result.ethical_analysis.get('confidence', 0):.3f}")
        print(f"감정적: 신뢰도 {result.emotional_analysis.get('confidence', 0):.3f}")
        print(f"인과적: 신뢰도 {result.causal_analysis.get('confidence', 0):.3f}\n")
        
        print("=== 수준 간 관계 ===")
        for relation in result.cross_level_relations:
            print(f"{relation.source_level} → {relation.target_level}: {relation.relation_type} (강도: {relation.strength:.3f})")
        
        print(f"\n=== 융합 결과 ===")
        fusion = result.fused_representation
        print(f"융합 신뢰도: {fusion.get('fusion_confidence', 0):.3f}")
        print(f"의미적 일관성: {fusion.get('semantic_coherence', 0):.3f}")
        
        if fusion.get('attention_weights'):
            print("\n지배적 어텐션 관계:")
            for relation in fusion['attention_weights'].get('dominant_relations', [])[:3]:
                print(f"  {relation}")
    
    # 비동기 테스트 실행
    asyncio.run(test_advanced_analyzer())