"""
고급 LLM 의미론적 중재 시스템 - Linux 전용
Advanced LLM Semantic Mediation System for Linux

고성능 로컬 LLM과 고급 AI 기법을 활용한 차세대 의미론적 중재 시스템
멀티모달 추론, 분산 처리, 실시간 학습을 지원
"""

import os
import json
import time
import asyncio
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import queue
import uuid
import re
from collections import defaultdict, deque
import hashlib

# 고급 LLM 라이브러리
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logging.warning("llama-cpp-python not available. Install with: pip install llama-cpp-python")

try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, AutoModel,
        pipeline, GPT2LMHeadModel, GPT2Tokenizer,
        BitsAndBytesConfig, GenerationConfig
    )
    import accelerate
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not available. Some features will be disabled.")

# 고급 AI 라이브러리
# SentenceTransformer는 sentence_transformer_singleton을 통해 사용
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

from config import ADVANCED_CONFIG, DEVICE, TORCH_DTYPE, BATCH_SIZE, MODELS_DIR
from data_models import (
    SemanticMediationRequest, SemanticMediationResult, 
    ConceptualBridge, CrossLevelSemanticRelation,
    AdvancedMediationResult, MediationStrategy, LLMResponse
)

logger = logging.getLogger('RedHeart.AdvancedLLMMediator')

@dataclass
class MediationContext:
    """중재 컨텍스트"""
    context_id: str
    session_history: List[Dict[str, Any]] = field(default_factory=list)
    domain_knowledge: Dict[str, Any] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    temporal_context: Dict[str, Any] = field(default_factory=dict)
    cultural_context: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class LLMModelInfo:
    """LLM 모델 정보"""
    model_name: str
    model_path: str
    model_type: str  # 'llama_cpp', 'transformers', 'vllm'
    capabilities: List[str]
    context_length: int
    memory_usage: float = 0.0
    inference_speed: float = 0.0
    is_loaded: bool = False

@dataclass
class MediationTask:
    """중재 작업"""
    task_id: str
    request: SemanticMediationRequest
    context: MediationContext
    priority: int = 1
    deadline: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class ConsensusResult:
    """합의 결과"""
    consensus_text: str
    confidence_score: float
    participating_models: List[str]
    disagreement_points: List[str]
    reasoning_traces: List[str]

class AdvancedLLMSemanticMediator:
    """고급 LLM 의미론적 중재기"""
    
    def __init__(self):
        """고급 LLM 중재기 초기화"""
        self.config = ADVANCED_CONFIG['llm_mediator']
        self.device = DEVICE
        self.dtype = TORCH_DTYPE
        
        # 모델 경로 설정
        self.models_dir = os.path.join(MODELS_DIR, 'llm_models')
        self.cache_dir = os.path.join(MODELS_DIR, 'llm_cache')
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # LLM 모델들 초기화
        self.llm_models = {}
        self.model_info = {}
        self._initialize_llm_models()
        
        # 임베딩 모델 초기화
        self._initialize_embedding_model()
        
        # 중재 전략 시스템 초기화
        self._initialize_mediation_strategies()
        
        # 작업 큐 및 스케줄링
        self.mediation_queue = queue.PriorityQueue()
        self.processing_threads = []
        self.is_processing = True
        self._start_processing_threads()
        
        # 캐시 시스템
        self.mediation_cache = {}
        self.reasoning_cache = {}
        self.consensus_cache = {}
        
        # 성능 모니터링
        self.performance_metrics = {
            'total_requests': 0,
            'successful_mediations': 0,
            'average_response_time': 0.0,
            'cache_hit_rate': 0.0,
            'model_usage_stats': defaultdict(int)
        }
        
        # 학습 시스템
        self.learning_buffer = deque(maxlen=1000)
        self.preference_learner = self._initialize_preference_learner()
        
        logger.info("고급 LLM 의미론적 중재기가 초기화되었습니다.")
    
    def _initialize_llm_models(self):
        """LLM 모델들 초기화"""
        try:
            # HelpingAI2-9B 모델 (기본)
            helping_ai_path = "/mnt/d/large_prj/linux_red_heart/llm_module/HelpingAI2-9B-Q4_K_M.gguf"
            if os.path.exists(helping_ai_path) and LLAMA_CPP_AVAILABLE:
                self.model_info['helping_ai'] = LLMModelInfo(
                    model_name="HelpingAI2-9B",
                    model_path=helping_ai_path,
                    model_type="llama_cpp",
                    capabilities=["text_generation", "reasoning", "korean", "ethical_analysis"],
                    context_length=4096
                )
                
                try:
                    self.llm_models['helping_ai'] = Llama(
                        model_path=helping_ai_path,
                        n_ctx=4096,
                        n_threads=4,
                        n_gpu_layers=0,  # CPU 사용
                        verbose=False
                    )
                    self.model_info['helping_ai'].is_loaded = True
                    logger.info("HelpingAI2-9B 모델이 로드되었습니다.")
                except Exception as e:
                    logger.error(f"HelpingAI2-9B 모델 로드 실패: {e}")
            
            # Transformers 기반 모델들 (선택적)
            if TRANSFORMERS_AVAILABLE:
                self._try_load_transformers_models()
            
            # 최소 하나의 모델이 로드되었는지 확인
            if not any(info.is_loaded for info in self.model_info.values()):
                logger.warning("사용 가능한 LLM 모델이 없습니다. 폴백 모드로 동작합니다.")
                self._initialize_fallback_model()
            
        except Exception as e:
            logger.error(f"LLM 모델 초기화 실패: {e}")
            self._initialize_fallback_model()
    
    def _try_load_transformers_models(self):
        """Transformers 기반 모델 로드 시도"""
        try:
            # 경량 모델들 시도
            lightweight_models = [
                "microsoft/DialoGPT-medium",
                "distilgpt2",
                "gpt2"
            ]
            
            for model_name in lightweight_models:
                try:
                    self.model_info[model_name] = LLMModelInfo(
                        model_name=model_name,
                        model_path=model_name,
                        model_type="transformers",
                        capabilities=["text_generation", "dialogue"],
                        context_length=1024
                    )
                    
                    # CPU에서 실행하도록 설정
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        cache_dir=self.models_dir,
                        trust_remote_code=True
                    )
                    
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        cache_dir=self.models_dir,
                        torch_dtype=torch.float32,
                        device_map="cpu",
                        trust_remote_code=True
                    )
                    
                    # 패딩 토큰 설정
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                    
                    self.llm_models[model_name] = {
                        'model': model,
                        'tokenizer': tokenizer
                    }
                    
                    self.model_info[model_name].is_loaded = True
                    logger.info(f"Transformers 모델 로드됨: {model_name}")
                    break  # 첫 번째 성공한 모델만 사용
                    
                except Exception as e:
                    logger.warning(f"모델 {model_name} 로드 실패: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Transformers 모델 로드 실패: {e}")
    
    def _initialize_fallback_model(self):
        """폴백 모델 초기화"""
        self.model_info['fallback'] = LLMModelInfo(
            model_name="FallbackModel",
            model_path="internal",
            model_type="fallback",
            capabilities=["basic_reasoning"],
            context_length=512,
            is_loaded=True
        )
        
        self.llm_models['fallback'] = self._create_fallback_model()
        logger.info("폴백 모델이 초기화되었습니다.")
    
    def _create_fallback_model(self):
        """기본 폴백 모델 생성"""
        class FallbackModel:
            def __init__(self):
                self.response_templates = {
                    'mediation': "분석된 결과들을 종합하면, {summary}입니다. 각 분석 간의 일관성은 {consistency}이며, 주요 갈등점은 {conflicts}입니다.",
                    'reasoning': "이 상황에서 고려해야 할 요소들은 {factors}입니다. 추천되는 접근법은 {recommendation}입니다.",
                    'consensus': "제시된 관점들을 조율한 결과, {consensus}라는 합의에 도달할 수 있습니다."
                }
            
            def __call__(self, prompt, **kwargs):
                # 간단한 키워드 기반 응답
                if "중재" in prompt or "mediation" in prompt.lower():
                    template = self.response_templates['mediation']
                    return {
                        'choices': [{
                            'text': template.format(
                                summary="다양한 관점들이 존재함",
                                consistency="보통 수준",
                                conflicts="일부 상충되는 의견들"
                            )
                        }]
                    }
                elif "추론" in prompt or "reasoning" in prompt.lower():
                    template = self.response_templates['reasoning']
                    return {
                        'choices': [{
                            'text': template.format(
                                factors="맥락적 요소들",
                                recommendation="균형잡힌 접근"
                            )
                        }]
                    }
                else:
                    template = self.response_templates['consensus']
                    return {
                        'choices': [{
                            'text': template.format(consensus="합리적인 절충안")
                        }]
                    }
        
        return FallbackModel()
    
    def _initialize_embedding_model(self):
        """임베딩 모델 초기화"""
        try:
            from sentence_transformer_singleton import get_sentence_transformer
            
            self.embedding_model = get_sentence_transformer(
                'paraphrase-multilingual-mpnet-base-v2',
                device='cpu',  # CPU 사용
                cache_folder=self.models_dir
            )
            logger.info("임베딩 모델이 초기화되었습니다.")
        except Exception as e:
            logger.error(f"임베딩 모델 초기화 실패: {e}")
            self.embedding_model = None
    
    def _initialize_mediation_strategies(self):
        """중재 전략 시스템 초기화"""
        self.mediation_strategies = {
            'consensus_building': self._consensus_building_strategy,
            'conflict_resolution': self._conflict_resolution_strategy,
            'evidence_weighing': self._evidence_weighing_strategy,
            'perspective_integration': self._perspective_integration_strategy,
            'ethical_reasoning': self._ethical_reasoning_strategy
        }
        
        # 전략 선택기
        self.strategy_selector = RandomForestClassifier(
            n_estimators=50,
            random_state=42
        )
        
        logger.info("중재 전략 시스템이 초기화되었습니다.")
    
    def _start_processing_threads(self):
        """처리 스레드들 시작"""
        num_threads = self.config.get('processing_threads', 2)
        
        for i in range(num_threads):
            thread = threading.Thread(
                target=self._process_mediation_queue,
                daemon=True,
                name=f"MediationThread-{i}"
            )
            thread.start()
            self.processing_threads.append(thread)
        
        logger.info(f"{num_threads}개의 중재 처리 스레드가 시작되었습니다.")
    
    def _process_mediation_queue(self):
        """중재 큐 처리"""
        while self.is_processing:
            try:
                # 우선순위 큐에서 작업 가져오기 (5초 타임아웃)
                priority, task = self.mediation_queue.get(timeout=5.0)
                
                if task is None:  # 종료 신호
                    break
                
                # 작업 처리
                result = self._process_mediation_task(task)
                
                # 결과를 작업 객체에 저장
                setattr(task, '_result', result)
                
                self.mediation_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"중재 큐 처리 중 오류: {e}")
    
    async def mediate_semantic_interactions_advanced(self, 
                                                   situation_description: str,
                                                   analysis_results: Dict[str, Any],
                                                   context: MediationContext = None,
                                                   strategy: str = "auto") -> AdvancedMediationResult:
        """
        고급 의미론적 상호작용 중재
        
        Args:
            situation_description: 상황 설명
            analysis_results: 분석 결과들
            context: 중재 컨텍스트
            strategy: 중재 전략
            
        Returns:
            고급 중재 결과
        """
        try:
            start_time = time.time()
            self.performance_metrics['total_requests'] += 1
            
            # 컨텍스트 생성/업데이트
            if context is None:
                context = MediationContext(
                    context_id=str(uuid.uuid4()),
                    temporal_context={'timestamp': datetime.now().isoformat()}
                )
            
            # 캐시 확인
            cache_key = self._generate_cache_key(situation_description, analysis_results, strategy)
            if cache_key in self.mediation_cache:
                cached_result = self.mediation_cache[cache_key]
                self.performance_metrics['cache_hit_rate'] += 1
                return cached_result
            
            logger.info(f"고급 의미론적 중재 시작: 전략 {strategy}")
            
            # 1단계: 분석 결과 전처리 및 정제
            preprocessed_results = await self._preprocess_analysis_results(analysis_results)
            
            # 2단계: 중재 전략 선택
            if strategy == "auto":
                selected_strategy = await self._select_optimal_strategy(
                    situation_description, preprocessed_results, context
                )
            else:
                selected_strategy = strategy
            
            # 3단계: 멀티모델 추론
            model_responses = await self._multi_model_inference(
                situation_description, preprocessed_results, context, selected_strategy
            )
            
            # 4단계: 합의 구축
            consensus_result = await self._build_consensus(model_responses, context)
            
            # 5단계: 갈등 해결
            conflict_resolution = await self._resolve_conflicts(
                consensus_result, model_responses, context
            )
            
            # 6단계: 개념적 브리지 구성
            conceptual_bridges = await self._construct_conceptual_bridges(
                preprocessed_results, consensus_result, context
            )
            
            # 7단계: 메타 추론 및 검증
            meta_reasoning = await self._perform_meta_reasoning(
                consensus_result, conflict_resolution, conceptual_bridges
            )
            
            # 8단계: 최종 중재 결과 생성
            mediation_result = AdvancedMediationResult(
                request_id=str(uuid.uuid4()),
                situation_description=situation_description,
                analysis_results=preprocessed_results,
                selected_strategy=selected_strategy,
                model_responses=model_responses,
                consensus_result=consensus_result,
                conflict_resolution=conflict_resolution,
                conceptual_bridges=conceptual_bridges,
                meta_reasoning=meta_reasoning,
                confidence_score=self._calculate_mediation_confidence(
                    consensus_result, conflict_resolution, meta_reasoning
                ),
                processing_time=time.time() - start_time,
                context=context,
                performance_metrics=self._get_current_performance_metrics()
            )
            
            # 캐시 저장
            self.mediation_cache[cache_key] = mediation_result
            
            # 학습 데이터 추가
            self._add_to_learning_buffer(mediation_result)
            
            # 성공 통계 업데이트
            self.performance_metrics['successful_mediations'] += 1
            self._update_average_response_time(mediation_result.processing_time)
            
            logger.info(f"고급 의미론적 중재 완료: 신뢰도 {mediation_result.confidence_score:.3f}")
            return mediation_result
            
        except Exception as e:
            logger.error(f"고급 의미론적 중재 실패: {e}")
            return self._create_error_result(situation_description, str(e))
    
    async def _preprocess_analysis_results(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """분석 결과 전처리 및 정제"""
        try:
            preprocessed = {}
            
            for module_name, result in analysis_results.items():
                if isinstance(result, dict):
                    # 오류 결과 필터링
                    if 'error' in result:
                        logger.warning(f"{module_name}에서 오류 발견: {result['error']}")
                        continue
                    
                    # 신뢰도 정규화
                    if 'confidence' in result:
                        result['confidence'] = max(0.0, min(1.0, float(result['confidence'])))
                    
                    # 누락된 필드 보완
                    if 'confidence' not in result:
                        result['confidence'] = 0.5  # 기본 신뢰도
                    
                    # 데이터 타입 정규화
                    for key, value in result.items():
                        if isinstance(value, np.ndarray):
                            result[key] = value.tolist()
                        elif isinstance(value, (np.integer, np.floating)):
                            result[key] = float(value)
                    
                    preprocessed[module_name] = result
                else:
                    # 비딕셔너리 결과 처리
                    preprocessed[module_name] = {
                        'value': result,
                        'confidence': 0.3  # 낮은 기본 신뢰도
                    }
            
            return preprocessed
            
        except Exception as e:
            logger.error(f"분석 결과 전처리 실패: {e}")
            return analysis_results
    
    async def _select_optimal_strategy(self, situation: str, results: Dict[str, Any], 
                                     context: MediationContext) -> str:
        """최적 중재 전략 선택"""
        try:
            # 상황 특성 분석
            situation_features = self._extract_situation_features(situation)
            
            # 결과 복잡성 분석
            complexity_features = self._analyze_result_complexity(results)
            
            # 컨텍스트 특성
            context_features = self._extract_context_features(context)
            
            # 특성 벡터 결합
            feature_vector = {**situation_features, **complexity_features, **context_features}
            
            # 규칙 기반 전략 선택
            if feature_vector.get('conflict_level', 0) > 0.7:
                return 'conflict_resolution'
            elif feature_vector.get('ethical_complexity', 0) > 0.6:
                return 'ethical_reasoning'
            elif feature_vector.get('analysis_diversity', 0) > 0.8:
                return 'perspective_integration'
            elif feature_vector.get('uncertainty_level', 0) > 0.5:
                return 'evidence_weighing'
            else:
                return 'consensus_building'
                
        except Exception as e:
            logger.error(f"전략 선택 실패: {e}")
            return 'consensus_building'  # 기본 전략
    
    def _extract_situation_features(self, situation: str) -> Dict[str, float]:
        """상황 특성 추출"""
        features = {
            'conflict_level': 0.0,
            'ethical_complexity': 0.0,
            'emotional_intensity': 0.0,
            'urgency_level': 0.0
        }
        
        situation_lower = situation.lower()
        
        # 갈등 수준
        conflict_keywords = ['갈등', '대립', '모순', 'conflict', 'oppose', 'contradiction']
        features['conflict_level'] = min(1.0, sum(1 for kw in conflict_keywords if kw in situation_lower) * 0.3)
        
        # 윤리적 복잡성
        ethical_keywords = ['윤리', '도덕', '옳다', '그르다', 'ethical', 'moral', 'right', 'wrong']
        features['ethical_complexity'] = min(1.0, sum(1 for kw in ethical_keywords if kw in situation_lower) * 0.2)
        
        # 감정적 강도
        emotion_keywords = ['화나다', '슬프다', '기쁘다', '걱정', 'angry', 'sad', 'happy', 'worried']
        features['emotional_intensity'] = min(1.0, sum(1 for kw in emotion_keywords if kw in situation_lower) * 0.25)
        
        # 긴급성 수준
        urgency_keywords = ['긴급', '빨리', '즉시', 'urgent', 'quickly', 'immediately']
        features['urgency_level'] = min(1.0, sum(1 for kw in urgency_keywords if kw in situation_lower) * 0.4)
        
        return features
    
    def _analyze_result_complexity(self, results: Dict[str, Any]) -> Dict[str, float]:
        """결과 복잡성 분석"""
        features = {
            'analysis_diversity': 0.0,
            'confidence_variance': 0.0,
            'result_count': 0.0,
            'uncertainty_level': 0.0
        }
        
        if not results:
            return features
        
        # 분석 다양성
        features['analysis_diversity'] = min(1.0, len(results) / 5.0)
        
        # 신뢰도 분산
        confidences = [r.get('confidence', 0.5) for r in results.values() if isinstance(r, dict)]
        if confidences:
            features['confidence_variance'] = float(np.var(confidences))
            features['uncertainty_level'] = 1.0 - float(np.mean(confidences))
        
        # 결과 수
        features['result_count'] = min(1.0, len(results) / 10.0)
        
        return features
    
    def _extract_context_features(self, context: MediationContext) -> Dict[str, float]:
        """컨텍스트 특성 추출"""
        features = {
            'session_length': 0.0,
            'domain_specificity': 0.0,
            'cultural_complexity': 0.0,
            'temporal_pressure': 0.0
        }
        
        # 세션 길이
        features['session_length'] = min(1.0, len(context.session_history) / 20.0)
        
        # 도메인 특수성
        features['domain_specificity'] = min(1.0, len(context.domain_knowledge) / 10.0)
        
        # 문화적 복잡성
        features['cultural_complexity'] = min(1.0, len(context.cultural_context) / 5.0)
        
        # 시간적 압박
        if context.temporal_context.get('deadline'):
            deadline = datetime.fromisoformat(context.temporal_context['deadline'])
            time_left = (deadline - datetime.now()).total_seconds()
            features['temporal_pressure'] = max(0.0, min(1.0, 1.0 - (time_left / 86400)))  # 1일 기준
        
        return features
    
    async def _multi_model_inference(self, situation: str, results: Dict[str, Any],
                                   context: MediationContext, strategy: str) -> List[LLMResponse]:
        """멀티모델 추론"""
        try:
            model_responses = []
            
            # 사용 가능한 모델들로 병렬 추론
            available_models = [name for name, info in self.model_info.items() if info.is_loaded]
            
            if not available_models:
                logger.warning("사용 가능한 모델이 없습니다.")
                return []
            
            # 각 모델에 대해 추론 수행
            inference_tasks = []
            for model_name in available_models:
                task = self._inference_with_model(model_name, situation, results, context, strategy)
                inference_tasks.append(task)
            
            # 병렬 실행
            responses = await asyncio.gather(*inference_tasks, return_exceptions=True)
            
            # 유효한 응답만 수집
            for model_name, response in zip(available_models, responses):
                if not isinstance(response, Exception) and response:
                    model_responses.append(response)
                    self.performance_metrics['model_usage_stats'][model_name] += 1
                else:
                    logger.warning(f"모델 {model_name}에서 추론 실패: {response}")
            
            return model_responses
            
        except Exception as e:
            logger.error(f"멀티모델 추론 실패: {e}")
            return []
    
    async def _inference_with_model(self, model_name: str, situation: str, 
                                  results: Dict[str, Any], context: MediationContext,
                                  strategy: str) -> Optional[LLMResponse]:
        """특정 모델로 추론"""
        try:
            # 프롬프트 생성
            prompt = self._generate_inference_prompt(situation, results, context, strategy)
            
            # 모델별 추론
            if model_name in self.llm_models:
                model = self.llm_models[model_name]
                
                if self.model_info[model_name].model_type == "llama_cpp":
                    response = await self._llama_cpp_inference(model, prompt, model_name)
                elif self.model_info[model_name].model_type == "transformers":
                    response = await self._transformers_inference(model, prompt, model_name)
                elif self.model_info[model_name].model_type == "fallback":
                    response = await self._fallback_inference(model, prompt, model_name)
                else:
                    logger.warning(f"알 수 없는 모델 타입: {self.model_info[model_name].model_type}")
                    return None
                
                return response
            
            return None
            
        except Exception as e:
            logger.error(f"모델 {model_name} 추론 실패: {e}")
            return None
    
    def _generate_inference_prompt(self, situation: str, results: Dict[str, Any],
                                 context: MediationContext, strategy: str) -> str:
        """추론 프롬프트 생성"""
        prompt_parts = []
        
        # 시스템 메시지
        prompt_parts.append(f"""당신은 윤리적 의사결정을 돕는 고급 AI 중재자입니다.
중재 전략: {strategy}

다음 분석 결과들을 종합하여 일관되고 통찰력 있는 해석을 제공해주세요:

상황: {situation}

분석 결과들:""")
        
        # 각 분석 결과 포맷
        for module_name, result in results.items():
            prompt_parts.append(f"\n### {module_name.upper()} 분석")
            if isinstance(result, dict):
                confidence = result.get('confidence', 0.5)
                prompt_parts.append(f"신뢰도: {confidence:.2f}")
                
                # 주요 결과 추출
                key_results = []
                for key, value in result.items():
                    if key != 'confidence' and not key.startswith('_'):
                        if isinstance(value, (str, int, float)):
                            key_results.append(f"{key}: {value}")
                        elif isinstance(value, list) and len(value) <= 5:
                            key_results.append(f"{key}: {value}")
                
                prompt_parts.append("\n".join(key_results[:5]))  # 상위 5개만
            else:
                prompt_parts.append(f"결과: {str(result)[:200]}")
        
        # 컨텍스트 정보
        if context.domain_knowledge:
            prompt_parts.append(f"\n### 도메인 지식")
            prompt_parts.append(str(context.domain_knowledge)[:300])
        
        # 요청사항
        prompt_parts.append(f"""

### 중재 요청
다음을 포함한 종합적 분석을 제공해주세요:
1. 통합적 해석: 모든 분석을 아우르는 일관된 이해
2. 주요 갈등점: 분석 간 불일치나 모순
3. 신뢰도 평가: 각 분석의 신뢰성
4. 실행 가능한 통찰: 의사결정에 도움이 되는 구체적 제안

응답 형식: JSON
{{
    "integrated_interpretation": "통합적 해석",
    "conflict_points": ["갈등점 1", "갈등점 2"],
    "confidence_assessment": {{"analysis1": 0.8, "analysis2": 0.6}},
    "actionable_insights": ["통찰 1", "통찰 2"],
    "reasoning_trace": "추론 과정 설명"
}}""")
        
        return "\n".join(prompt_parts)
    
    async def _llama_cpp_inference(self, model: Llama, prompt: str, model_name: str) -> LLMResponse:
        """LLaMA-cpp 모델 추론"""
        try:
            loop = asyncio.get_event_loop()
            
            # 비동기 실행
            response = await loop.run_in_executor(
                None,
                lambda: model(
                    prompt,
                    max_tokens=1024,
                    temperature=0.3,
                    top_p=0.9,
                    stop=["###", "---", "\n\n\n"],
                    echo=False
                )
            )
            
            response_text = response['choices'][0]['text'].strip()
            
            return LLMResponse(
                model_name=model_name,
                response_text=response_text,
                confidence=0.8,  # LLaMA 모델 기본 신뢰도
                reasoning_trace=[f"LLaMA-cpp inference with {model_name}"],
                metadata={
                    'tokens_generated': len(response_text.split()),
                    'model_type': 'llama_cpp'
                }
            )
            
        except Exception as e:
            logger.error(f"LLaMA-cpp 추론 실패: {e}")
            return None
    
    async def _transformers_inference(self, model_info: Dict, prompt: str, model_name: str) -> LLMResponse:
        """Transformers 모델 추론"""
        try:
            model = model_info['model']
            tokenizer = model_info['tokenizer']
            
            # 입력 토크나이징
            inputs = tokenizer.encode(prompt, return_tensors='pt', max_length=512, truncation=True)
            
            # 비동기 실행
            loop = asyncio.get_event_loop()
            
            response = await loop.run_in_executor(
                None,
                lambda: model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 256,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(inputs)
                )
            )
            
            # 디코딩
            response_text = tokenizer.decode(response[0][inputs.shape[1]:], skip_special_tokens=True)
            
            return LLMResponse(
                model_name=model_name,
                response_text=response_text.strip(),
                confidence=0.6,  # Transformers 모델 기본 신뢰도
                reasoning_trace=[f"Transformers inference with {model_name}"],
                metadata={
                    'tokens_generated': len(response_text.split()),
                    'model_type': 'transformers'
                }
            )
            
        except Exception as e:
            logger.error(f"Transformers 추론 실패: {e}")
            return None
    
    async def _fallback_inference(self, model, prompt: str, model_name: str) -> LLMResponse:
        """폴백 모델 추론"""
        try:
            response = model(prompt)
            response_text = response['choices'][0]['text']
            
            return LLMResponse(
                model_name=model_name,
                response_text=response_text,
                confidence=0.4,  # 폴백 모델 낮은 신뢰도
                reasoning_trace=[f"Fallback inference"],
                metadata={
                    'tokens_generated': len(response_text.split()),
                    'model_type': 'fallback'
                }
            )
            
        except Exception as e:
            logger.error(f"폴백 추론 실패: {e}")
            return None
    
    async def _build_consensus(self, model_responses: List[LLMResponse], 
                             context: MediationContext) -> ConsensusResult:
        """합의 구축"""
        try:
            if not model_responses:
                return ConsensusResult(
                    consensus_text="사용 가능한 모델 응답이 없습니다.",
                    confidence_score=0.0,
                    participating_models=[],
                    disagreement_points=["모델 응답 부재"],
                    reasoning_traces=[]
                )
            
            # 응답들을 파싱하여 구조화된 데이터 추출
            parsed_responses = []
            for response in model_responses:
                parsed = self._parse_model_response(response.response_text)
                parsed['model_name'] = response.model_name
                parsed['confidence'] = response.confidence
                parsed_responses.append(parsed)
            
            # 공통 요소 추출
            common_themes = self._extract_common_themes(parsed_responses)
            
            # 불일치 지점 식별
            disagreements = self._identify_disagreements(parsed_responses)
            
            # 가중 합의 생성
            consensus_text = self._generate_weighted_consensus(parsed_responses, common_themes)
            
            # 전체 신뢰도 계산
            confidence_score = self._calculate_consensus_confidence(parsed_responses, disagreements)
            
            return ConsensusResult(
                consensus_text=consensus_text,
                confidence_score=confidence_score,
                participating_models=[r['model_name'] for r in parsed_responses],
                disagreement_points=disagreements,
                reasoning_traces=[r.get('reasoning_trace', '') for r in parsed_responses]
            )
            
        except Exception as e:
            logger.error(f"합의 구축 실패: {e}")
            return ConsensusResult(
                consensus_text=f"합의 구축 중 오류 발생: {e}",
                confidence_score=0.0,
                participating_models=[],
                disagreement_points=[str(e)],
                reasoning_traces=[]
            )
    
    def _parse_model_response(self, response_text: str) -> Dict[str, Any]:
        """모델 응답 파싱"""
        try:
            # JSON 응답 파싱 시도
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
            
            # JSON이 없으면 키워드 기반 파싱
            parsed = {
                'integrated_interpretation': '',
                'conflict_points': [],
                'confidence_assessment': {},
                'actionable_insights': [],
                'reasoning_trace': response_text
            }
            
            # 간단한 키워드 매칭으로 내용 추출
            lines = response_text.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if '통합' in line or 'integrated' in line.lower():
                    current_section = 'integrated_interpretation'
                    parsed[current_section] = line
                elif '갈등' in line or 'conflict' in line.lower():
                    current_section = 'conflict_points'
                    if line:
                        parsed[current_section].append(line)
                elif '통찰' in line or 'insight' in line.lower():
                    current_section = 'actionable_insights'
                    if line:
                        parsed[current_section].append(line)
            
            return parsed
            
        except Exception as e:
            logger.error(f"모델 응답 파싱 실패: {e}")
            return {
                'integrated_interpretation': response_text[:200],
                'conflict_points': [],
                'confidence_assessment': {},
                'actionable_insights': [],
                'reasoning_trace': response_text
            }
    
    def _extract_common_themes(self, parsed_responses: List[Dict[str, Any]]) -> List[str]:
        """공통 주제 추출"""
        try:
            all_insights = []
            all_interpretations = []
            
            for response in parsed_responses:
                insights = response.get('actionable_insights', [])
                interpretation = response.get('integrated_interpretation', '')
                
                all_insights.extend(insights)
                all_interpretations.append(interpretation)
            
            # 키워드 빈도 분석
            keyword_counts = defaultdict(int)
            
            for insight in all_insights:
                if isinstance(insight, str):
                    words = insight.lower().split()
                    for word in words:
                        if len(word) > 3:  # 의미있는 단어만
                            keyword_counts[word] += 1
            
            for interpretation in all_interpretations:
                if isinstance(interpretation, str):
                    words = interpretation.lower().split()
                    for word in words:
                        if len(word) > 3:
                            keyword_counts[word] += 1
            
            # 고빈도 키워드를 공통 주제로 선택
            common_themes = [
                word for word, count in keyword_counts.items()
                if count >= len(parsed_responses) // 2  # 절반 이상에서 언급
            ]
            
            return common_themes[:10]  # 상위 10개
            
        except Exception as e:
            logger.error(f"공통 주제 추출 실패: {e}")
            return []
    
    def _identify_disagreements(self, parsed_responses: List[Dict[str, Any]]) -> List[str]:
        """불일치 지점 식별"""
        try:
            disagreements = []
            
            # 신뢰도 평가 불일치
            confidence_assessments = [r.get('confidence_assessment', {}) for r in parsed_responses]
            if len(confidence_assessments) > 1:
                for analysis_type in ['surface', 'ethical', 'emotional', 'causal']:
                    scores = [ca.get(analysis_type, 0.5) for ca in confidence_assessments if analysis_type in ca]
                    if len(scores) > 1 and np.std(scores) > 0.3:
                        disagreements.append(f"{analysis_type} 분석 신뢰도에 대한 의견 차이")
            
            # 갈등점 불일치
            all_conflicts = []
            for response in parsed_responses:
                conflicts = response.get('conflict_points', [])
                all_conflicts.extend(conflicts)
            
            if len(all_conflicts) > len(set(all_conflicts)):
                disagreements.append("갈등점 식별에서 모델 간 차이")
            
            # 통찰 다양성 검사
            all_insights = []
            for response in parsed_responses:
                insights = response.get('actionable_insights', [])
                all_insights.extend(insights)
            
            if len(all_insights) > 0:
                insight_similarity = len(set(all_insights)) / len(all_insights)
                if insight_similarity > 0.8:  # 너무 다양하면 불일치
                    disagreements.append("실행 가능한 통찰에서 의견 분산")
            
            return disagreements
            
        except Exception as e:
            logger.error(f"불일치 지점 식별 실패: {e}")
            return ["불일치 지점 분석 실패"]
    
    def _generate_weighted_consensus(self, parsed_responses: List[Dict[str, Any]], 
                                   common_themes: List[str]) -> str:
        """가중 합의 생성"""
        try:
            # 모델별 가중치 (신뢰도 기반)
            weights = []
            total_confidence = sum(r.get('confidence', 0.5) for r in parsed_responses)
            
            for response in parsed_responses:
                weight = response.get('confidence', 0.5) / max(total_confidence, 0.1)
                weights.append(weight)
            
            # 가중 평균된 해석 생성
            consensus_parts = []
            
            # 통합적 해석
            interpretations = [r.get('integrated_interpretation', '') for r in parsed_responses]
            if interpretations:
                consensus_parts.append("## 통합적 해석")
                # 가장 신뢰도 높은 해석을 기본으로 사용
                best_idx = max(range(len(weights)), key=lambda i: weights[i])
                consensus_parts.append(interpretations[best_idx])
            
            # 공통 주제 추가
            if common_themes:
                consensus_parts.append("\n## 공통 주제")
                consensus_parts.append(f"분석에서 공통적으로 나타난 주제: {', '.join(common_themes[:5])}")
            
            # 종합 통찰
            all_insights = []
            for i, response in enumerate(parsed_responses):
                insights = response.get('actionable_insights', [])
                weighted_insights = [(insight, weights[i]) for insight in insights if isinstance(insight, str)]
                all_insights.extend(weighted_insights)
            
            if all_insights:
                consensus_parts.append("\n## 종합 통찰")
                # 가중치 기반 상위 통찰 선택
                sorted_insights = sorted(all_insights, key=lambda x: x[1], reverse=True)
                top_insights = [insight for insight, weight in sorted_insights[:3]]
                consensus_parts.extend([f"- {insight}" for insight in top_insights])
            
            return "\n".join(consensus_parts)
            
        except Exception as e:
            logger.error(f"가중 합의 생성 실패: {e}")
            return "가중 합의 생성 중 오류가 발생했습니다."
    
    def _calculate_consensus_confidence(self, parsed_responses: List[Dict[str, Any]], 
                                      disagreements: List[str]) -> float:
        """합의 신뢰도 계산"""
        try:
            if not parsed_responses:
                return 0.0
            
            # 기본 신뢰도 (모델들의 평균)
            model_confidences = [r.get('confidence', 0.5) for r in parsed_responses]
            base_confidence = np.mean(model_confidences)
            
            # 불일치 패널티
            disagreement_penalty = len(disagreements) * 0.1
            
            # 모델 수 보너스 (더 많은 모델이 참여할수록 신뢰도 증가)
            model_bonus = min(0.2, len(parsed_responses) * 0.05)
            
            # 최종 신뢰도
            final_confidence = base_confidence + model_bonus - disagreement_penalty
            
            return max(0.0, min(1.0, final_confidence))
            
        except Exception as e:
            logger.error(f"합의 신뢰도 계산 실패: {e}")
            return 0.5
    
    # 중재 전략 구현들
    async def _consensus_building_strategy(self, situation: str, results: Dict[str, Any], 
                                         context: MediationContext) -> Dict[str, Any]:
        """합의 구축 전략"""
        return {"strategy": "consensus_building", "focus": "finding_common_ground"}
    
    async def _conflict_resolution_strategy(self, situation: str, results: Dict[str, Any], 
                                          context: MediationContext) -> Dict[str, Any]:
        """갈등 해결 전략"""
        return {"strategy": "conflict_resolution", "focus": "resolving_disagreements"}
    
    async def _evidence_weighing_strategy(self, situation: str, results: Dict[str, Any], 
                                        context: MediationContext) -> Dict[str, Any]:
        """증거 가중 전략"""
        return {"strategy": "evidence_weighing", "focus": "evaluating_evidence_strength"}
    
    async def _perspective_integration_strategy(self, situation: str, results: Dict[str, Any], 
                                              context: MediationContext) -> Dict[str, Any]:
        """관점 통합 전략"""
        return {"strategy": "perspective_integration", "focus": "integrating_viewpoints"}
    
    async def _ethical_reasoning_strategy(self, situation: str, results: Dict[str, Any], 
                                        context: MediationContext) -> Dict[str, Any]:
        """윤리적 추론 전략"""
        return {"strategy": "ethical_reasoning", "focus": "moral_evaluation"}
    
    # 기타 헬퍼 메서드들 (간단히 구현)
    async def _resolve_conflicts(self, consensus: ConsensusResult, responses: List[LLMResponse], 
                               context: MediationContext) -> Dict[str, Any]:
        """갈등 해결"""
        return {
            "resolved_conflicts": len(consensus.disagreement_points),
            "resolution_method": "weighted_voting",
            "confidence": consensus.confidence_score
        }
    
    async def _construct_conceptual_bridges(self, results: Dict[str, Any], 
                                          consensus: ConsensusResult,
                                          context: MediationContext) -> List[ConceptualBridge]:
        """개념적 브리지 구성"""
        bridges = []
        
        # 간단한 브리지 생성 예시
        if len(results) >= 2:
            modules = list(results.keys())
            for i in range(len(modules)):
                for j in range(i + 1, len(modules)):
                    bridge = ConceptualBridge(
                        source_concept=modules[i],
                        target_concept=modules[j],
                        relationship_type="semantic_similarity",
                        strength=0.7,
                        explanation=f"Connection between {modules[i]} and {modules[j]}",
                        evidence=[f"Common themes in consensus"]
                    )
                    bridges.append(bridge)
        
        return bridges
    
    async def _perform_meta_reasoning(self, consensus: ConsensusResult, 
                                    conflict_resolution: Dict[str, Any],
                                    bridges: List[ConceptualBridge]) -> Dict[str, Any]:
        """메타 추론"""
        return {
            "meta_confidence": consensus.confidence_score,
            "reasoning_quality": "high" if consensus.confidence_score > 0.7 else "medium",
            "integration_success": len(bridges) > 0,
            "overall_coherence": consensus.confidence_score * 0.8
        }
    
    def _calculate_mediation_confidence(self, consensus: ConsensusResult,
                                      conflict_resolution: Dict[str, Any],
                                      meta_reasoning: Dict[str, Any]) -> float:
        """중재 신뢰도 계산"""
        consensus_conf = consensus.confidence_score
        resolution_conf = conflict_resolution.get('confidence', 0.5)
        meta_conf = meta_reasoning.get('meta_confidence', 0.5)
        
        return (consensus_conf * 0.5 + resolution_conf * 0.3 + meta_conf * 0.2)
    
    def _get_current_performance_metrics(self) -> Dict[str, Any]:
        """현재 성능 메트릭 반환"""
        return {
            'cache_hit_rate': self.performance_metrics['cache_hit_rate'] / max(1, self.performance_metrics['total_requests']),
            'success_rate': self.performance_metrics['successful_mediations'] / max(1, self.performance_metrics['total_requests']),
            'average_response_time': self.performance_metrics['average_response_time']
        }
    
    def _add_to_learning_buffer(self, result: AdvancedMediationResult):
        """학습 버퍼에 추가"""
        learning_data = {
            'situation_features': self._extract_situation_features(result.situation_description),
            'strategy_used': result.selected_strategy,
            'confidence_achieved': result.confidence_score,
            'processing_time': result.processing_time
        }
        self.learning_buffer.append(learning_data)
    
    def _update_average_response_time(self, response_time: float):
        """평균 응답 시간 업데이트"""
        current_avg = self.performance_metrics['average_response_time']
        total_requests = self.performance_metrics['total_requests']
        
        if total_requests == 1:
            self.performance_metrics['average_response_time'] = response_time
        else:
            self.performance_metrics['average_response_time'] = (
                (current_avg * (total_requests - 1) + response_time) / total_requests
            )
    
    def _initialize_preference_learner(self):
        """선호도 학습기 초기화"""
        return MLPClassifier(
            hidden_layer_sizes=(64, 32),
            random_state=42
        )
    
    def _generate_cache_key(self, situation: str, results: Dict[str, Any], strategy: str) -> str:
        """캐시 키 생성"""
        key_components = [
            situation[:100],
            str(sorted(results.keys())),
            strategy
        ]
        return hashlib.md5('|'.join(key_components).encode()).hexdigest()
    
    def _create_error_result(self, situation: str, error_msg: str) -> AdvancedMediationResult:
        """오류 결과 생성"""
        return AdvancedMediationResult(
            request_id=str(uuid.uuid4()),
            situation_description=situation,
            analysis_results={},
            selected_strategy="error_handling",
            model_responses=[],
            consensus_result=ConsensusResult(
                consensus_text=f"오류 발생: {error_msg}",
                confidence_score=0.0,
                participating_models=[],
                disagreement_points=[error_msg],
                reasoning_traces=[]
            ),
            conflict_resolution={"error": error_msg},
            conceptual_bridges=[],
            meta_reasoning={"error": error_msg},
            confidence_score=0.0,
            processing_time=0.0,
            context=MediationContext(context_id=str(uuid.uuid4())),
            performance_metrics={"error": error_msg}
        )
    
    def _process_mediation_task(self, task: MediationTask) -> AdvancedMediationResult:
        """중재 작업 처리 (동기 버전)"""
        try:
            # 비동기 메서드를 동기적으로 실행
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            result = loop.run_until_complete(
                self.mediate_semantic_interactions_advanced(
                    task.request.situation_description,
                    task.request.analysis_results,
                    task.context,
                    "auto"
                )
            )
            
            loop.close()
            return result
            
        except Exception as e:
            logger.error(f"중재 작업 처리 실패: {e}")
            return self._create_error_result(
                task.request.situation_description, 
                str(e)
            )
    
    def shutdown(self):
        """중재기 종료"""
        self.is_processing = False
        
        # 처리 스레드들에 종료 신호 전송
        for _ in self.processing_threads:
            self.mediation_queue.put((0, None))
        
        # 스레드 종료 대기
        for thread in self.processing_threads:
            thread.join(timeout=5.0)
        
        logger.info("고급 LLM 의미론적 중재기가 종료되었습니다.")

def create_advanced_llm_mediator() -> AdvancedLLMSemanticMediator:
    """고급 LLM 중재기 생성"""
    return AdvancedLLMSemanticMediator()

# 테스트 코드
if __name__ == "__main__":
    import asyncio
    
    async def test_advanced_llm_mediator():
        """고급 LLM 중재기 테스트"""
        mediator = create_advanced_llm_mediator()
        
        # 테스트 데이터
        situation = """
        한 회사에서 개인정보 보호와 업무 효율성 사이에서 갈등이 발생했습니다.
        직원들의 작업 모니터링 시스템 도입을 검토하고 있지만, 
        프라이버시 침해에 대한 우려가 제기되고 있습니다.
        """
        
        analysis_results = {
            'ethical_analysis': {
                'moral_judgment': 'complex',
                'rights_duties': {'privacy_rights': 0.8, 'productivity_duty': 0.6},
                'confidence': 0.75
            },
            'emotional_analysis': {
                'primary_emotion': 'concern',
                'emotional_intensity': 0.7,
                'valence': -0.3,
                'confidence': 0.8
            },
            'causal_analysis': {
                'causal_strength': 0.6,
                'primary_factors': ['efficiency_pressure', 'privacy_concerns'],
                'confidence': 0.65
            }
        }
        
        context = MediationContext(
            context_id="test_context",
            domain_knowledge={'domain': 'workplace_ethics'},
            cultural_context={'culture': 'korean_business'}
        )
        
        print("=== 고급 LLM 의미론적 중재 테스트 ===\n")
        
        result = await mediator.mediate_semantic_interactions_advanced(
            situation, analysis_results, context
        )
        
        print(f"중재 완료: 신뢰도 {result.confidence_score:.3f}")
        print(f"선택된 전략: {result.selected_strategy}")
        print(f"처리 시간: {result.processing_time:.3f}초")
        
        print(f"\n=== 합의 결과 ===")
        print(f"신뢰도: {result.consensus_result.confidence_score:.3f}")
        print(f"참여 모델: {result.consensus_result.participating_models}")
        print(f"합의 내용:\n{result.consensus_result.consensus_text[:300]}...")
        
        if result.consensus_result.disagreement_points:
            print(f"\n불일치 지점:")
            for point in result.consensus_result.disagreement_points:
                print(f"- {point}")
        
        print(f"\n=== 개념적 브리지 ===")
        for bridge in result.conceptual_bridges[:3]:
            print(f"- {bridge.source_concept} ↔ {bridge.target_concept}: {bridge.relationship_type}")
        
        print(f"\n=== 성능 메트릭 ===")
        for metric, value in result.performance_metrics.items():
            print(f"- {metric}: {value}")
        
        # 종료
        mediator.shutdown()
        print(f"\n테스트 완료!")
    
    # 비동기 테스트 실행
    asyncio.run(test_advanced_llm_mediator())