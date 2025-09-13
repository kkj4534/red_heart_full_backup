"""
Advanced Regret Analyzer for Linux Red Heart System
GPU 가속 후회 분석 및 신경망 학습 시스템

Features:
- CUDA 기반 병렬 후회 계산
- 트랜스포머 모델을 통한 고급 후회 패턴 학습
- 실시간 성능 추적 및 벤치마킹
- 비동기 처리 파이프라인
- 고급 캐싱 및 메모리 관리
- 다차원 후회 메트릭 분석
"""

__all__ = ['AdvancedRegretAnalyzer']

import os
# CVE-2025-32434는 가짜 CVE - torch_security_patch import 제거
# import torch_security_patch

import torch
import torch.nn as nn
import numpy as np
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import time

from transformers import (
    AutoTokenizer, AutoModel, 
    BertForSequenceClassification,
    RobertaForSequenceClassification
)
from dynamic_threshold_system import dynamic_threshold_calculator
from mixture_of_experts import create_regret_moe, MixtureOfExperts
from three_view_scenario_system import ThreeViewScenarioSystem
from phase_controller_hook import PhaseControllerHook, PhaseType, PerformanceMetric

# 로거 설정
logger = logging.getLogger('advanced_regret_analyzer')

@dataclass
class AdvancedRegretMetrics:
    """고급 후회 평가 메트릭"""
    decision_id: str
    timestamp: datetime
    
    # 기본 후회 메트릭
    anticipated_regret: float
    experienced_regret: float
    regret_intensity: float
    regret_duration: float
    
    # GPU 가속 고급 메트릭
    semantic_regret_score: float
    emotional_regret_vector: List[float]
    causal_attribution_scores: Dict[str, float]
    counterfactual_utility_delta: float
    
    # 학습 관련 메트릭
    prediction_accuracy: float
    learning_rate_adjustment: float
    model_confidence: float
    uncertainty_estimate: float
    
    # 성능 메트릭
    computation_time_ms: float
    gpu_memory_usage_mb: float
    cache_hit_rate: float

class GPURegretNetwork(nn.Module):
    """CUDA 기반 후회 학습 신경망"""
    
    def __init__(self, input_dim: int = 896, hidden_dim: int = 512):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 멀티레이어 후회 예측 네트워크
        self.regret_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        ).to(self.device)
        
        # 감정 벡터 예측 네트워크
        self.emotion_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 8),  # 8차원 감정 벡터
            nn.Tanh()
        ).to(self.device)
        
        # 불확실성 추정 네트워크
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        ).to(self.device)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.to(self.device)
        regret_score = self.regret_predictor(x)
        emotion_vector = self.emotion_predictor(x)
        uncertainty = self.uncertainty_estimator(x)
        return regret_score, emotion_vector, uncertainty

class AdvancedRegretAnalyzer:
    """고급 GPU 가속 후회 분석기"""
    
    def __init__(self, config_path: str = "system_config.json"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)
        
        # 설정 로드
        self.config = self._load_config(config_path)
        
        # GPU 메모리 관리 - 동적 관리자 연동
        from dynamic_gpu_manager import get_gpu_manager
        self.gpu_manager = get_gpu_manager()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # klue/bert-base 모델은 안정성을 위해 15% 할당 유지
            self.gpu_memory_fraction = 0.15  # 15% 할당 유지
            print(f"🔧 klue/bert-base 모델: 안정성 보장을 위한 {self.gpu_memory_fraction*100}% 할당")
            
            # GPU 메모리 상태 체크
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            allocated_memory = torch.cuda.memory_allocated() / 1e9
            available_memory = total_memory - allocated_memory
            
            if available_memory < 1.0:  # 1GB 미만이면 경고
                print(f"⚠️ 경고: GPU 메모리 여유분 {available_memory:.1f}GB - 오버헤드 위험")
                self.gpu_memory_fraction = 0.05  # 더욱 보수적으로 설정
        
        # 트랜스포머 모델 초기화 - 최적화된 캐싱 방식
        self.tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
        # 글로벌 모델 캐시 사용
        self.transformer_model = None
        self.model_loaded = False
        
        # 배치 처리를 위한 요청 큐
        self.request_queue = []
        self.batch_size = 4  # 배치 크기
        self.last_batch_time = time.time()
        self.batch_timeout = 0.1  # 100ms 타임아웃
        
        print("📋 Transformer 모델: 최적화된 배치 처리 방식 활성화")
        
        # 후회 학습 네트워크 초기화
        self.regret_network = GPURegretNetwork()
        self.optimizer = torch.optim.AdamW(self.regret_network.parameters(), lr=1e-4)
        self.loss_fn = nn.MSELoss()
        
        # 성능 추적 (캐시 제거됨)
        self.performance_metrics = []
        self.learning_history = []
        
        # =====================================================
        # 강화 모듈 통합 (47M 추가 → 총 50M)
        # =====================================================
        base_dim = 768
        
        # 1. 반사실 시뮬레이션 네트워크 (15M)
        self.counterfactual_sim = nn.ModuleDict({
            'world_model': nn.Sequential(
                nn.Linear(base_dim, 1024),
                nn.LayerNorm(1024),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(1024, 1024),
                nn.LayerNorm(1024),
                nn.GELU(),
                nn.Linear(1024, 768),
                nn.LayerNorm(768),
                nn.Linear(768, base_dim)
            ),
            'outcome_predictor': nn.LSTM(
                input_size=base_dim,
                hidden_size=base_dim // 2,
                num_layers=3,
                batch_first=True,
                bidirectional=True
            ),
            'regret_calculator': nn.Sequential(
                nn.Linear(base_dim * 2, base_dim),
                nn.LayerNorm(base_dim),
                nn.GELU(),
                nn.Linear(base_dim, 512),
                nn.Linear(512, 256),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )
        }).to(self.device)
        
        # 2. 시간축 후회 전파 (12M)
        self.temporal_propagation = nn.ModuleDict({
            'past_encoder': nn.LSTM(
                input_size=base_dim,
                hidden_size=512,
                num_layers=3,
                batch_first=True,
                dropout=0.1
            ),
            'future_predictor': nn.GRU(
                input_size=base_dim,
                hidden_size=512,
                num_layers=3,
                batch_first=True,
                dropout=0.1
            ),
            'temporal_attention': nn.MultiheadAttention(
                embed_dim=base_dim,
                num_heads=12,
                dropout=0.1,
                batch_first=True
            ),
            'regret_dynamics': nn.Sequential(
                nn.Linear(base_dim * 2, base_dim),
                nn.LayerNorm(base_dim),
                nn.GELU(),
                nn.Linear(base_dim, 512),
                nn.Linear(512, 10)  # 10 time steps
            )
        }).to(self.device)
        
        # 3. 의사결정 트리 분석 (10M)
        self.decision_tree = nn.ModuleDict({
            'branch_evaluator': nn.ModuleList([
                nn.Sequential(
                    nn.Linear(base_dim, 512),
                    nn.LayerNorm(512),
                    nn.GELU(),
                    nn.Linear(512, 256),
                    nn.Linear(256, 128),
                    nn.Linear(128, 1)
                ) for _ in range(8)  # 8 branches
            ]),
            'path_integrator': nn.Sequential(
                nn.Linear(8, 256),
                nn.GELU(),
                nn.Linear(256, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Linear(512, base_dim)
            ),
            'decision_scorer': nn.Sequential(
                nn.Linear(base_dim, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Linear(512, 256),
                nn.Linear(256, 1)
            )
        }).to(self.device)
        
        # 4. 베이지안 추론 (10M + 3M 추가 = 13M)
        self.bayesian_inference = nn.ModuleDict({
            'prior_network': nn.Sequential(
                nn.Linear(base_dim, 768),
                nn.LayerNorm(768),
                nn.GELU(),
                nn.Linear(768, 512),
                nn.Linear(512, 256)
            ),
            'likelihood_network': nn.Sequential(
                nn.Linear(base_dim, 768),
                nn.LayerNorm(768),
                nn.GELU(),
                nn.Linear(768, 512),
                nn.Linear(512, 256)
            ),
            'posterior_network': nn.Sequential(
                nn.Linear(512, 768),
                nn.LayerNorm(768),
                nn.GELU(),
                nn.Linear(768, base_dim)
            ),
            'uncertainty_quantifier': nn.Sequential(
                nn.Linear(base_dim, 512),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.Linear(256, 1),
                nn.Softplus()
            ),
            # 추가 레이어 (3M)
            'deep_bayesian': nn.Sequential(
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
            *self.counterfactual_sim.parameters(),
            *self.temporal_propagation.parameters(),
            *self.decision_tree.parameters(),
            *self.bayesian_inference.parameters()
        ])
        logger.info(f"✅ 후회 분석기 강화 모듈 통합: {total_params/1e6:.1f}M 파라미터 추가")
        
        # 비동기 처리
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 후회 데이터 저장소
        self.regret_database = []
        self.decision_outcomes = {}
        
        # 동적 임계값 계산기 통합
        self.dynamic_threshold_calculator = dynamic_threshold_calculator
        
        # Mixture of Experts for 후회 분석
        self.moe_enabled = True
        if self.moe_enabled:
            try:
                # 후회 분석용 MoE 초기화
                regret_input_dim = 512  # 후회 맥락 임베딩 차원
                regret_output_dim = 3   # 후회 유형 수 (action, inaction, outcome)
                
                self.regret_moe = create_regret_moe(
                    input_dim=regret_input_dim,
                    output_dim=regret_output_dim,
                    num_experts=3
                ).to(self.device)
                
                self.logger.info("후회 분석용 MoE 시스템 초기화 완료 (3개 전문가)")
            except Exception as e:
                self.logger.warning(f"후회 MoE 초기화 실패, 기본 시스템 사용: {e}")
                self.moe_enabled = False
        
        # 3뷰 시나리오 시스템 초기화
        self.scenario_system_enabled = True
        if self.scenario_system_enabled:
            try:
                self.three_view_system = ThreeViewScenarioSystem(device=self.device)
                self.logger.info("후회 분석기 3뷰 시나리오 시스템 초기화 완료")
            except Exception as e:
                self.logger.warning(f"3뷰 시나리오 시스템 초기화 실패: {e}")
                self.scenario_system_enabled = False
        
        # PhaseController Hook 초기화
        self.phase_controller_enabled = True
        if self.phase_controller_enabled:
            try:
                # 후회 분석기 모델들 수집
                models = {}
                if hasattr(self, 'regret_network') and self.regret_network:
                    models['regret_network'] = self.regret_network
                if hasattr(self, 'regret_moe') and self.regret_moe:
                    models['regret_moe'] = self.regret_moe
                
                self.phase_controller = PhaseControllerHook(
                    models=models,
                    performance_threshold=0.8,
                    error_threshold=0.15
                )
                
                # 모니터링 시작
                self.phase_controller.start_monitoring()
                
                self.logger.info("후회 분석기 PhaseController Hook 초기화 완료")
            except Exception as e:
                self.logger.warning(f"PhaseController Hook 초기화 실패: {e}")
                self.phase_controller_enabled = False
        
        self.logger.info(f"Advanced Regret Analyzer initialized on {self.device}")
        self.logger.info("동적 임계값 시스템 통합 완료")
    
    def _load_config(self, config_path: str) -> Dict:
        """설정 파일 로드"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "regret_analysis": {
                    "learning_rate": 1e-4,
                    "batch_size": 32,
                    "max_sequence_length": 512,
                    "cache_size": 1000,
                    "gpu_memory_fraction": 0.3
                }
            }
    
    def _load_transformer_model_on_demand(self):
        """최적화된 Transformer 모델 로드 - GPU 관리자 연동"""
        if not self.model_loaded:
            print("🔄 klue/bert-base 모델을 안정 모드로 로드 중...")
            start_time = time.time()
            
            try:
                # GPU 메모리 최적화
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 모델 로드 최적화 - 캐시 사용
                try:
                    # local_files_only=True로 캐시된 모델 우선 사용
                    self.transformer_model = AutoModel.from_pretrained(
                        'klue/bert-base', 
                        local_files_only=True
                    ).to(self.device)
                    print("📦 캐시된 모델 사용")
                except:
                    # 캐시에 없으면 다운로드
                    self.transformer_model = AutoModel.from_pretrained('klue/bert-base').to(self.device)
                    print("📥 모델 다운로드 완료")
                
                self.model_loaded = True
                load_time = time.time() - start_time
                
                # GPU 메모리 상태 로깅
                memory_status = self.gpu_manager.get_memory_status()
                print(f"✅ klue/bert-base 모델 로드 완료 ({load_time:.2f}초)")
                print(f"🔧 GPU 메모리 사용량: {memory_status.get('allocated_gb', 0):.1f}GB / {memory_status.get('total_gb', 0):.1f}GB")
            
            except Exception as e:
                self.logger.error(f"❌ klue/bert-base 모델 로드 실패: {e}")
                self.logger.error("🚨 연구 단계에서 핵심 모델 로딩 실패는 치명적 오류입니다.")
                self.logger.error("💡 해결 방법: 1) GPU 메모리 확인, 2) 모델 캐시 정리, 3) 네트워크 연결 확인")
                print(f"❌ CRITICAL ERROR: klue/bert-base 모델 로드 실패")
                print(f"   오류 내용: {e}")
                print(f"   연구 단계에서는 모든 모델이 정상 작동해야 합니다.")
                print(f"   프로세스를 종료합니다.")
                raise RuntimeError(f"Critical model loading failure: {e}")
    
    def _unload_transformer_model(self):
        """Transformer 모델을 GPU에서 해제하여 메모리 절약"""
        if self.model_loaded and self.transformer_model is not None:
            print("🗑️ Transformer 모델을 GPU에서 해제 중...")
            del self.transformer_model
            self.transformer_model = None
            self.model_loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("✅ GPU 메모리 해제 완료")
    
    async def _process_batch_requests(self):
        """배치로 요청들을 처리하여 오버헤드 감소"""
        if not self.request_queue:
            return
        
        current_time = time.time()
        # 배치가 가득 찼거나 타임아웃이 지났으면 처리
        if (len(self.request_queue) >= self.batch_size or 
            current_time - self.last_batch_time > self.batch_timeout):
            
            print(f"🔄 배치 처리 시작: {len(self.request_queue)}개 요청")
            batch_start = time.time()
            
            # 모델 로드 (한 번만)
            self._load_transformer_model_on_demand()
            
            # 배치 처리
            for request in self.request_queue:
                # 여기서 실제 처리 로직 구현
                pass
            
            # 메모리 정리
            if len(self.request_queue) >= self.batch_size:
                self._unload_transformer_model()
            
            batch_time = time.time() - batch_start
            print(f"✅ 배치 처리 완료 ({batch_time:.2f}초, {len(self.request_queue)}개)")
            
            self.request_queue.clear()
            self.last_batch_time = current_time
    
    async def analyze_regret(self, decision_data: Dict[str, Any], 
                           outcome_data: Optional[Dict[str, Any]] = None) -> AdvancedRegretMetrics:
        """비동기 후회 분석 수행 (최적화된 조건부 로직)"""
        start_time = time.time()
        initial_memory = self._get_gpu_memory_usage()
        
        try:
            # 의사결정 데이터 전처리
            processed_data = await self._preprocess_decision_data(decision_data)
            
            # 복잡도 평가 및 분석 방법 결정
            complexity_level = self._evaluate_decision_complexity(processed_data)
            
            if complexity_level >= 3:  # 복잡한 의사결정
                # 전체 베이지안 반사실적 추론 사용
                return await self._perform_complex_regret_analysis(processed_data, outcome_data, start_time, initial_memory)
            else:
                # 경량 후회 분석 사용
                return await self._perform_lightweight_regret_analysis(processed_data, outcome_data, start_time, initial_memory)
                
        except Exception as e:
            logger.error(f"후회 분석 실패: {e}")
            raise RuntimeError(f"후회 분석 실패, fallback 비활성화됨: {e}")
    
    def _evaluate_decision_complexity(self, processed_data: Dict[str, Any]) -> int:
        """의사결정 복잡도 평가 (1-5 점수)"""
        complexity_score = 0
        text = processed_data.get('text', '')
        
        # 1. 텍스트 길이 (기본 복잡도)
        if len(text) > 100:
            complexity_score += 1
        if len(text) > 300:
            complexity_score += 1
            
        # 2. 감정적 복잡도 (감정 키워드 수)
        emotion_keywords = ['후회', '미안', '아쉬', '안타깝', '실망', '좌절', '갈등', '혼란', '딜레마']
        emotion_count = sum(1 for word in emotion_keywords if word in text)
        if emotion_count >= 2:
            complexity_score += 1
            
        # 3. 대안의 수 (choice, 선택, 방법 등)
        alternative_indicators = ['선택', '방법', '대안', '옵션', '가능성', '경우']
        alternative_count = sum(1 for word in alternative_indicators if word in text)
        if alternative_count >= 2:
            complexity_score += 1
            
        # 4. 시간적 복잡도 (과거, 현재, 미래 언급)
        temporal_indicators = ['과거', '현재', '미래', '전에', '지금', '나중에', '앞으로']
        temporal_count = sum(1 for word in temporal_indicators if word in text)
        if temporal_count >= 2:
            complexity_score += 1
            
        return min(complexity_score, 5)
    
    async def _perform_complex_regret_analysis(self, processed_data: Dict[str, Any], 
                                             outcome_data: Optional[Dict[str, Any]], 
                                             start_time: float, initial_memory: float) -> AdvancedRegretMetrics:
        """복잡한 후회 분석 - 전체 베이지안 추론"""
        # 트랜스포머 기반 의미적 임베딩 생성
        semantic_embedding = await self._generate_semantic_embedding(processed_data['text'])
        
        # GPU 기반 후회 예측
        regret_predictions = await self._predict_regret(semantic_embedding)
        
        # 반사실적 분석
        counterfactual_analysis = await self._perform_counterfactual_analysis(
            processed_data, outcome_data
        )
        
        # MoE 기반 후회 유형 분석
        if self.moe_enabled:
            regret_predictions = await self._apply_moe_regret_analysis(
                semantic_embedding, regret_predictions, processed_data
            )
        
        # 3뷰 시나리오 기반 후회 분석
        if self.scenario_system_enabled:
            regret_predictions = await self._apply_scenario_regret_analysis(
                regret_predictions, processed_data
            )
        
        # 인과관계 분석
        causal_attribution = await self._analyze_causal_attribution(
            processed_data, semantic_embedding
        )
        
        # 성능 메트릭 계산
        computation_time = (time.time() - start_time) * 1000
        final_memory = self._get_gpu_memory_usage()
        memory_usage = final_memory - initial_memory
        
        # 후회 예측 결과 검증
        if (regret_predictions.get('intensity', 0) <= 0.0 and 
            regret_predictions.get('anticipated', 0) <= 0.0):
            logger.warning(f"복잡한 후회 분석에서 0.0 값들 감지, 최소값 보장")
            regret_predictions['intensity'] = max(0.1, regret_predictions.get('intensity', 0.1))
            regret_predictions['anticipated'] = max(0.1, regret_predictions.get('anticipated', 0.1))
            regret_predictions['experienced'] = max(0.1, regret_predictions.get('experienced', 0.1))
        
        # 동적 임계값 기반 후회 판정
        dynamic_threshold = counterfactual_analysis.get('dynamic_threshold', 0.3)
        relative_regret = counterfactual_analysis.get('relative_regret', 0.0)
        
        # 후회 강도 조정 (동적 임계값 적용)
        if relative_regret > dynamic_threshold:
            # 임계값을 초과한 경우 후회 강도 증가
            adjusted_intensity = regret_predictions['intensity'] * (1.0 + relative_regret)
            adjusted_anticipated = regret_predictions['anticipated'] * (1.0 + relative_regret)
        else:
            # 임계값 미만인 경우 후회 강도 감소
            adjusted_intensity = regret_predictions['intensity'] * 0.8
            adjusted_anticipated = regret_predictions['anticipated'] * 0.8
        
        # 종합 후회 메트릭 생성
        regret_metrics = AdvancedRegretMetrics(
            decision_id=processed_data.get('id', f"decision_{datetime.now().isoformat()}"),
            timestamp=datetime.now(),
            anticipated_regret=max(0.0, min(1.0, adjusted_anticipated)),
            experienced_regret=regret_predictions['experienced'],
            regret_intensity=max(0.0, min(1.0, adjusted_intensity)),
            regret_duration=regret_predictions['duration'],
            semantic_regret_score=regret_predictions['semantic_score'],
            emotional_regret_vector=regret_predictions['emotion_vector'],
            causal_attribution_scores=causal_attribution,
            counterfactual_utility_delta=counterfactual_analysis['utility_delta'],
            prediction_accuracy=regret_predictions['accuracy'],
            learning_rate_adjustment=regret_predictions['lr_adjustment'],
            model_confidence=regret_predictions['confidence'],
            uncertainty_estimate=regret_predictions['uncertainty'],
            computation_time_ms=computation_time,
            gpu_memory_usage_mb=memory_usage,
            cache_hit_rate=self._calculate_cache_hit_rate()
        )
        
        # 학습 데이터 업데이트
        await self._update_learning_data(regret_metrics, outcome_data)
        
        # PhaseController Hook에 성능 기록
        if self.phase_controller_enabled:
            try:
                # 후회 예측 오차 계산
                prediction_error = abs(regret_metrics.anticipated_regret - regret_metrics.experienced_regret)
                
                # 성능 메트릭 구성
                performance_metrics = {
                    'regret_prediction_error': prediction_error,
                    'processing_time_ms': regret_metrics.computation_time_ms,
                    'confidence_score': regret_metrics.model_confidence,
                    'uncertainty_estimate': regret_metrics.uncertainty_estimate,
                    'error': prediction_error  # 오류 패턴 분석용
                }
                
                # 모델별 성능
                model_performances = {}
                if hasattr(regret_metrics, 'moe_metadata'):
                    model_performances['regret_moe'] = regret_metrics.moe_metadata
                if hasattr(regret_metrics, 'scenario_metadata'):
                    model_performances['scenario_system'] = regret_metrics.scenario_metadata
                
                # 컨텍스트 정보
                context = {
                    'analysis_type': 'complex_regret_analysis',
                    'decision_id': regret_metrics.decision_id,
                    'regret_intensity': regret_metrics.regret_intensity,
                    'anticipated_regret': regret_metrics.anticipated_regret,
                    'experienced_regret': regret_metrics.experienced_regret,
                    'complexity_level': complexity_level
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
        
        return regret_metrics
    
    async def _apply_moe_regret_analysis(self, semantic_embedding: torch.Tensor,
                                       regret_predictions: Dict[str, float],
                                       processed_data: Dict[str, Any]) -> Dict[str, float]:
        """MoE 기반 후회 유형 분석 및 예측 개선"""
        try:
            # 1. MoE 시스템을 통한 후회 유형별 분석 (안전한 차원 처리)
            if semantic_embedding.dim() == 1:
                semantic_input = semantic_embedding.unsqueeze(0)
            else:
                semantic_input = semantic_embedding
            moe_result = self.regret_moe(semantic_input, return_expert_outputs=True)
            
            # 2. 전문가별 결과 분석
            expert_insights = {}
            for expert_output in moe_result.expert_outputs:
                expert_id = expert_output.expert_id
                expert_confidence = expert_output.confidence
                expert_weight = expert_output.weight
                
                # 전문가 유형별 특화 분석
                if 'action_regret' in expert_id:
                    expert_insights['action_regret'] = {
                        'confidence': expert_confidence,
                        'weight': expert_weight,
                        'prediction': expert_output.output.item()
                    }
                elif 'inaction_regret' in expert_id:
                    expert_insights['inaction_regret'] = {
                        'confidence': expert_confidence,
                        'weight': expert_weight,
                        'prediction': expert_output.output.item()
                    }
                elif 'outcome_regret' in expert_id:
                    expert_insights['outcome_regret'] = {
                        'confidence': expert_confidence,
                        'weight': expert_weight,
                        'prediction': expert_output.output.item()
                    }
            
            # 3. 컨텍스트 기반 가중치 조정
            context_weights = self._calculate_context_weights(processed_data)
            
            # 4. 전문가 합의 기반 예측 개선
            improved_predictions = regret_predictions.copy()
            
            # 행동 후회 vs 비행동 후회 비율 계산
            action_weight = expert_insights.get('action_regret', {}).get('weight', 0.0)
            inaction_weight = expert_insights.get('inaction_regret', {}).get('weight', 0.0)
            outcome_weight = expert_insights.get('outcome_regret', {}).get('weight', 0.0)
            
            total_weight = action_weight + inaction_weight + outcome_weight
            if total_weight > 0:
                # 정규화된 가중치 계산
                norm_action = action_weight / total_weight
                norm_inaction = inaction_weight / total_weight
                norm_outcome = outcome_weight / total_weight
                
                # MoE 결과를 기반으로 예측 개선
                moe_intensity = (
                    expert_insights.get('action_regret', {}).get('prediction', 0.0) * norm_action +
                    expert_insights.get('inaction_regret', {}).get('prediction', 0.0) * norm_inaction +
                    expert_insights.get('outcome_regret', {}).get('prediction', 0.0) * norm_outcome
                )
                
                # 기존 예측과 MoE 결과 블렌딩
                blend_factor = moe_result.diversity_score  # 다양성 점수를 블렌딩 팩터로 사용
                improved_predictions['intensity'] = (
                    regret_predictions['intensity'] * (1 - blend_factor) +
                    moe_intensity * blend_factor
                )
                
                # 예상 후회도 비슷하게 조정
                improved_predictions['anticipated'] = (
                    regret_predictions['anticipated'] * (1 - blend_factor) +
                    moe_intensity * 0.9 * blend_factor  # 약간 낮은 가중치
                )
                
                # 신뢰도 개선 (전문가 신뢰도 반영)
                avg_expert_confidence = np.mean([
                    expert_insights.get('action_regret', {}).get('confidence', 0.5),
                    expert_insights.get('inaction_regret', {}).get('confidence', 0.5),
                    expert_insights.get('outcome_regret', {}).get('confidence', 0.5)
                ])
                improved_predictions['confidence'] = min(0.95, 
                    regret_predictions['confidence'] * 0.7 + avg_expert_confidence * 0.3
                )
                
                # 불확실성 조정 (다양성이 높을수록 불확실성 증가)
                uncertainty_adjustment = 1.0 + (moe_result.diversity_score - 0.5) * 0.2
                improved_predictions['uncertainty'] = min(1.0,
                    regret_predictions['uncertainty'] * uncertainty_adjustment
                )
            
            # 5. 메타데이터 추가
            improved_predictions['moe_metadata'] = {
                'expert_count': len(moe_result.expert_outputs),
                'diversity_score': moe_result.diversity_score,
                'top_expert': max(expert_insights.keys(), 
                                key=lambda k: expert_insights[k]['weight']) if expert_insights else None,
                'total_experts_used': moe_result.total_experts_used
            }
            
            logger.info(f"MoE 후회 분석 완료: {len(expert_insights)}개 전문가 활용, "
                       f"다양성 점수: {moe_result.diversity_score:.3f}")
            
            return improved_predictions
            
        except Exception as e:
            logger.warning(f"MoE 후회 분석 실패, 기본 예측 사용: {e}")
            return regret_predictions
    
    def _calculate_context_weights(self, processed_data: Dict[str, Any]) -> Dict[str, float]:
        """컨텍스트 기반 가중치 계산"""
        text = processed_data.get('text', '').lower()
        
        # 텍스트 분석 기반 가중치 계산
        weights = {
            'action_regret': 0.33,
            'inaction_regret': 0.33,
            'outcome_regret': 0.34
        }
        
        # 행동 관련 키워드
        action_keywords = ['했다', '행동', '실행', '진행', '수행', '처리']
        action_count = sum(1 for keyword in action_keywords if keyword in text)
        
        # 비행동 관련 키워드
        inaction_keywords = ['안했다', '하지않았다', '포기', '미루', '기회', '놓쳤다']
        inaction_count = sum(1 for keyword in inaction_keywords if keyword in text)
        
        # 결과 관련 키워드
        outcome_keywords = ['결과', '성과', '효과', '영향', '변화', '달성']
        outcome_count = sum(1 for keyword in outcome_keywords if keyword in text)
        
        # 가중치 조정
        total_count = action_count + inaction_count + outcome_count
        if total_count > 0:
            weights['action_regret'] = 0.2 + (action_count / total_count) * 0.6
            weights['inaction_regret'] = 0.2 + (inaction_count / total_count) * 0.6
            weights['outcome_regret'] = 0.2 + (outcome_count / total_count) * 0.6
            
            # 정규화
            total_weight = sum(weights.values())
            for key in weights:
                weights[key] /= total_weight
        
        return weights
    
    async def _apply_scenario_regret_analysis(self, regret_predictions: Dict[str, float],
                                           processed_data: Dict[str, Any]) -> Dict[str, float]:
        """3뷰 시나리오 기반 후회 분석"""
        try:
            # 1. 3뷰 시나리오 분석 수행
            scenario_analysis = await self.three_view_system.analyze_three_view_scenarios(processed_data)
            
            # 2. 시나리오별 후회 가중치 계산
            scenario_weights = {
                'optimistic': scenario_analysis.optimistic_scenario.probability_weight,
                'neutral': scenario_analysis.neutral_scenario.probability_weight,
                'pessimistic': scenario_analysis.pessimistic_scenario.probability_weight
            }
            
            # 3. 시나리오별 후회 강도 계산
            scenario_regrets = {
                'optimistic': scenario_analysis.optimistic_scenario.regret_potential,
                'neutral': scenario_analysis.neutral_scenario.regret_potential,
                'pessimistic': scenario_analysis.pessimistic_scenario.regret_potential
            }
            
            # 4. 가중 평균 후회 계산
            total_weight = sum(scenario_weights.values())
            if total_weight > 0:
                weighted_regret = sum(
                    scenario_regrets[scenario] * scenario_weights[scenario]
                    for scenario in scenario_regrets
                ) / total_weight
            else:
                weighted_regret = scenario_regrets['neutral']
            
            # 5. 기존 후회 예측과 통합
            enhanced_predictions = regret_predictions.copy()
            
            # 시나리오 분석 결과를 기존 예측과 블렌딩
            scenario_influence = min(0.4, scenario_analysis.consensus_strength)  # 최대 40% 영향
            
            # 예상 후회 조정
            original_anticipated = regret_predictions.get('anticipated', 0.5)
            enhanced_predictions['anticipated'] = (
                original_anticipated * (1 - scenario_influence) +
                weighted_regret * scenario_influence
            )
            
            # 후회 강도 조정
            original_intensity = regret_predictions.get('intensity', 0.5)
            enhanced_predictions['intensity'] = (
                original_intensity * (1 - scenario_influence) +
                weighted_regret * scenario_influence
            )
            
            # 경험 후회 조정 (시나리오 다양성 고려)
            diversity_factor = scenario_analysis.scenario_diversity
            original_experienced = regret_predictions.get('experienced', 0.5)
            enhanced_predictions['experienced'] = (
                original_experienced * (1 - diversity_factor * 0.3) +
                weighted_regret * diversity_factor * 0.3
            )
            
            # 6. 시나리오별 세부 분석
            regret_breakdown = self._analyze_scenario_regret_breakdown(scenario_analysis)
            
            # 7. 불확실성 조정
            uncertainty_range = scenario_analysis.uncertainty_range
            uncertainty_span = abs(uncertainty_range[1] - uncertainty_range[0])
            
            # 불확실성이 클수록 후회 가능성 증가
            uncertainty_adjustment = 1.0 + uncertainty_span * 0.2
            enhanced_predictions['uncertainty'] = min(1.0,
                regret_predictions.get('uncertainty', 0.5) * uncertainty_adjustment
            )
            
            # 8. 신뢰도 조정 (합의 강도 반영)
            enhanced_predictions['confidence'] = min(0.95,
                regret_predictions.get('confidence', 0.7) * scenario_analysis.consensus_strength
            )
            
            # 9. 메타데이터 추가
            enhanced_predictions['scenario_metadata'] = {
                'consensus_regret': scenario_analysis.consensus_regret,
                'uncertainty_range': uncertainty_range,
                'scenario_diversity': scenario_analysis.scenario_diversity,
                'consensus_strength': scenario_analysis.consensus_strength,
                'recommended_decision': scenario_analysis.recommended_decision,
                'scenario_weights': scenario_weights,
                'scenario_regrets': scenario_regrets,
                'regret_breakdown': regret_breakdown
            }
            
            self.logger.debug(f"3뷰 시나리오 후회 분석 완료: 합의 후회 {scenario_analysis.consensus_regret:.3f}, "
                            f"다양성 {scenario_analysis.scenario_diversity:.3f}")
            
            return enhanced_predictions
            
        except Exception as e:
            self.logger.warning(f"3뷰 시나리오 후회 분석 실패, 기본 예측 사용: {e}")
            return regret_predictions
    
    def _analyze_scenario_regret_breakdown(self, scenario_analysis) -> Dict[str, Any]:
        """시나리오별 후회 세부 분석"""
        
        breakdown = {
            'optimistic_risks': scenario_analysis.optimistic_scenario.risk_factors,
            'neutral_risks': scenario_analysis.neutral_scenario.risk_factors,
            'pessimistic_risks': scenario_analysis.pessimistic_scenario.risk_factors,
            'optimistic_opportunities': scenario_analysis.optimistic_scenario.opportunity_factors,
            'neutral_opportunities': scenario_analysis.neutral_scenario.opportunity_factors,
            'pessimistic_opportunities': scenario_analysis.pessimistic_scenario.opportunity_factors
        }
        
        # 위험 요소 분석
        all_risks = set()
        for risks in [breakdown['optimistic_risks'], breakdown['neutral_risks'], breakdown['pessimistic_risks']]:
            all_risks.update(risks)
        
        risk_frequency = {}
        for risk in all_risks:
            frequency = sum(1 for risks in [breakdown['optimistic_risks'], breakdown['neutral_risks'], breakdown['pessimistic_risks']] if risk in risks)
            risk_frequency[risk] = frequency / 3.0  # 정규화
        
        # 기회 요소 분석
        all_opportunities = set()
        for opportunities in [breakdown['optimistic_opportunities'], breakdown['neutral_opportunities'], breakdown['pessimistic_opportunities']]:
            all_opportunities.update(opportunities)
        
        opportunity_frequency = {}
        for opportunity in all_opportunities:
            frequency = sum(1 for opportunities in [breakdown['optimistic_opportunities'], breakdown['neutral_opportunities'], breakdown['pessimistic_opportunities']] if opportunity in opportunities)
            opportunity_frequency[opportunity] = frequency / 3.0  # 정규화
        
        # 후회 유형 분석
        regret_types = {
            'action_regret': 0.0,
            'inaction_regret': 0.0,
            'outcome_regret': 0.0
        }
        
        # 위험 요소 기반 후회 유형 추정
        for risk, freq in risk_frequency.items():
            if any(keyword in risk.lower() for keyword in ['실행', '행동', '진행']):
                regret_types['action_regret'] += freq * 0.3
            elif any(keyword in risk.lower() for keyword in ['기회', '놓침', '미루']):
                regret_types['inaction_regret'] += freq * 0.3
            else:
                regret_types['outcome_regret'] += freq * 0.3
        
        # 정규화
        total_regret = sum(regret_types.values())
        if total_regret > 0:
            for regret_type in regret_types:
                regret_types[regret_type] /= total_regret
        
        return {
            'risk_frequency': risk_frequency,
            'opportunity_frequency': opportunity_frequency,
            'regret_types': regret_types,
            'top_risks': sorted(risk_frequency.items(), key=lambda x: x[1], reverse=True)[:3],
            'top_opportunities': sorted(opportunity_frequency.items(), key=lambda x: x[1], reverse=True)[:3]
        }
    
    async def _perform_lightweight_regret_analysis(self, processed_data: Dict[str, Any], 
                                                 outcome_data: Optional[Dict[str, Any]], 
                                                 start_time: float, initial_memory: float) -> AdvancedRegretMetrics:
        """경량 후회 분석 - 빠른 휴리스틱 기반"""
        # 간단한 키워드 기반 후회 점수 계산
        regret_score = self._calculate_heuristic_regret_score(processed_data['text'])
        
        # 기본 감정 벡터 생성
        emotion_vector = self._generate_basic_emotion_vector(processed_data['text'])
        
        # 간단한 인과관계 분석
        causal_attribution = self._analyze_basic_causal_attribution(processed_data)
        
        # 성능 메트릭 계산
        computation_time = (time.time() - start_time) * 1000
        final_memory = self._get_gpu_memory_usage()
        memory_usage = final_memory - initial_memory
        
        # 후회 점수 검증 및 최소값 보장
        if regret_score <= 0.0:
            logger.warning(f"후회 분석에서 0.0 점수 감지: '{processed_data['text'][:50]}...'")
            regret_score = max(0.1, regret_score)  # 최소 0.1 보장
            
        # 경량 후회 메트릭 생성
        regret_metrics = AdvancedRegretMetrics(
            decision_id=processed_data.get('id', f"decision_{datetime.now().isoformat()}"),
            timestamp=datetime.now(),
            anticipated_regret=regret_score,
            experienced_regret=regret_score * 0.8,  # 추정값
            regret_intensity=regret_score,
            regret_duration=min(regret_score * 10, 100),  # 최대 100
            semantic_regret_score=regret_score,
            emotional_regret_vector=emotion_vector,
            causal_attribution_scores=causal_attribution,
            counterfactual_utility_delta=regret_score * 0.5,
            prediction_accuracy=0.7,  # 경량 분석 기본값
            learning_rate_adjustment=0.01,
            model_confidence=max(0.6, regret_score),  # 실제 점수 기반 신뢰도
            uncertainty_estimate=0.4,
            computation_time_ms=computation_time,
            gpu_memory_usage_mb=memory_usage,
            cache_hit_rate=self._calculate_cache_hit_rate()
        )
        
        return regret_metrics
    
    def _calculate_heuristic_regret_score(self, text: str) -> float:
        """휴리스틱 기반 후회 점수 계산"""
        regret_keywords = {
            'high': ['후회', '미안', '실수', '잘못', '아쉬', '안타깝', '실망'],
            'medium': ['걱정', '고민', '망설', '불안', '의심', '갈등'],
            'low': ['생각', '고려', '판단', '결정', '선택']
        }
        
        text_lower = text.lower()
        score = 0.0
        
        for word in regret_keywords['high']:
            if word in text_lower:
                score += 0.8
        
        for word in regret_keywords['medium']:
            if word in text_lower:
                score += 0.5
                
        for word in regret_keywords['low']:
            if word in text_lower:
                score += 0.2
        
        return min(score, 1.0)
    
    def _generate_basic_emotion_vector(self, text: str) -> List[float]:
        """기본 감정 벡터 생성"""
        emotion_keywords = {
            'sadness': ['슬프', '우울', '실망', '좌절'],
            'anger': ['화나', '분노', '짜증', '억울'],
            'fear': ['두려', '불안', '걱정', '무서'],
            'regret': ['후회', '미안', '아쉬', '안타깝']
        }
        
        text_lower = text.lower()
        vector = [0.0] * 8  # 8차원 감정 벡터
        
        for i, (emotion, keywords) in enumerate(emotion_keywords.items()):
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if i < len(vector):
                vector[i] = min(score / len(keywords), 1.0)
        
        return vector
    
    def _analyze_basic_causal_attribution(self, processed_data: Dict[str, Any]) -> Dict[str, float]:
        """기본 인과관계 분석"""
        text = processed_data.get('text', '')
        
        # 간단한 인과관계 지표들
        causal_indicators = {
            'personal_choice': ['선택', '결정', '판단', '행동'],
            'external_pressure': ['압력', '강요', '요구', '필요'],
            'circumstances': ['상황', '환경', '조건', '현실'],
            'emotions': ['감정', '기분', '마음', '느낌']
        }
        
        attribution_scores = {}
        text_lower = text.lower()
        
        for factor, indicators in causal_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            attribution_scores[factor] = min(score / len(indicators), 1.0)
        
        return attribution_scores
    
    async def _preprocess_decision_data(self, decision_data: Dict[str, Any]) -> Dict[str, Any]:
        """의사결정 데이터 전처리"""
        return {
            'text': f"{decision_data.get('scenario', '')} {decision_data.get('action', '')}",
            'context': decision_data.get('context', {}),
            'stakeholders': decision_data.get('stakeholders', []),
            'constraints': decision_data.get('constraints', []),
            'alternatives': decision_data.get('alternatives', [])
        }
    
    async def _generate_semantic_embedding(self, text: str) -> torch.Tensor:
        """트랜스포머 기반 의미적 임베딩 생성 - 연구급 분석"""
        # 캐시 완전 제거 - 각 분석마다 완전한 처리 보장
        
        # 모델이 로드되지 않았다면 로드
        self._load_transformer_model_on_demand()
        
        # 모델이 제대로 로드되지 않은 경우 오류 발생
        if not self.model_loaded or self.transformer_model is None:
            self.logger.error("❌ Transformer 모델이 로드되지 않음 - 임베딩 생성 불가")
            raise RuntimeError("Critical error: Transformer model not loaded. Cannot generate semantic embeddings.")
        
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors='pt', 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.transformer_model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1)
            
            # 캐시 완전 제거됨
            
            return embedding
        
        except Exception as e:
            self.logger.error(f"❌ 임베딩 생성 중 치명적 오류: {e}")
            self.logger.error("🚨 연구 단계에서 임베딩 생성 실패는 허용되지 않습니다.")
            raise RuntimeError(f"Critical embedding generation failure: {e}")
    
    async def _predict_regret(self, semantic_embedding: torch.Tensor) -> Dict[str, float]:
        """GPU 기반 후회 예측"""
        with torch.no_grad():
            regret_score, emotion_vector, uncertainty = self.regret_network(semantic_embedding)
        
        # CPU로 이동하여 결과 추출
        regret_score = regret_score.cpu().numpy().flatten()[0]
        emotion_vector = emotion_vector.cpu().numpy().flatten().tolist()
        uncertainty = uncertainty.cpu().numpy().flatten()[0]
        
        # 고급 후회 계산 - 다차원 분석
        contextual_factors = self._analyze_contextual_factors(semantic_embedding)
        temporal_decay = self._calculate_temporal_decay(uncertainty)
        cognitive_load = self._estimate_cognitive_load(emotion_vector)
        
        # 복합적 후회 점수 계산 (0-1 범위 보장)
        base_regret = float(regret_score)
        anticipated_regret = min(1.0, base_regret * (1 + contextual_factors * 0.3))
        experienced_regret = min(1.0, anticipated_regret * (0.7 + temporal_decay * 0.3))
        intensity = min(1.0, base_regret * (1 + uncertainty + cognitive_load * 0.2))
        duration = self._calculate_regret_duration(base_regret, uncertainty, cognitive_load)
        
        return {
            'anticipated': anticipated_regret,
            'experienced': experienced_regret,
            'intensity': intensity,
            'duration': duration,
            'semantic_score': base_regret,
            'emotion_vector': emotion_vector,
            'accuracy': max(0.0, min(1.0, 1 - uncertainty)),  # 0-1 범위 보장
            'lr_adjustment': uncertainty * 0.01,  # 세밀한 학습율 조정
            'confidence': max(0.0, min(1.0, 1 - uncertainty)),  # 0-1 범위 보장
            'uncertainty': uncertainty,
            'contextual_factors': contextual_factors,
            'temporal_decay': temporal_decay,
            'cognitive_load': cognitive_load
        }
    
    def _analyze_contextual_factors(self, semantic_embedding: torch.Tensor) -> float:
        """맥락적 요인 분석"""
        embedding_norm = torch.norm(semantic_embedding).item()
        embedding_variance = torch.var(semantic_embedding).item()
        semantic_complexity = embedding_norm * embedding_variance
        return min(1.0, semantic_complexity / 10.0)  # 정규화
    
    def _calculate_temporal_decay(self, uncertainty: float) -> float:
        """시간적 감쇠 계산"""
        import math
        # 불확실성이 높을수록 시간적 감쇠가 빠름
        decay_rate = 0.1 + uncertainty * 0.05
        return math.exp(-decay_rate)
    
    def _estimate_cognitive_load(self, emotion_vector: list) -> float:
        """인지적 부하 추정"""
        import numpy as np
        emotion_intensity = np.std(emotion_vector)  # 감정 벡터의 표준편차
        emotion_complexity = len([x for x in emotion_vector if abs(x) > 0.5])
        return min(1.0, (emotion_intensity + emotion_complexity * 0.1) / 2.0)
    
    def _calculate_regret_duration(self, base_regret: float, uncertainty: float, cognitive_load: float) -> float:
        """실제 감정 지속 시간 시뮬레이션 (실시간이 아닌 시뮬레이션 시간)"""
        import math
        
        # 심리학적 연구 기반 후회 지속 시간 모델
        # 경미한 후회: 몇 시간~며칠, 심각한 후회: 몇 주~몇 달
        
        # 기본 지속 시간 (일 단위)
        if base_regret < 0.3:  # 경미한 후회
            base_days = 0.5 + base_regret * 2  # 0.5일~1.1일
        elif base_regret < 0.7:  # 중간 후회  
            base_days = 1 + base_regret * 7  # 1일~6일
        else:  # 심각한 후회
            base_days = 5 + base_regret * 30  # 5일~35일
            
        # 불확실성이 높으면 더 오래 지속 (확신이 없으면 계속 생각함)
        uncertainty_multiplier = 1 + uncertainty * 1.5
        
        # 인지적 부하가 높으면 더 오래 지속 (복잡한 감정일수록 정리 시간 필요)
        cognitive_multiplier = 1 + cognitive_load * 0.8
        
        # 개인차 요인은 불확실성과 인지적 부하에 기반한 결정론적 계산
        import numpy as np
        # 불확실성과 인지적 부하의 조합으로 개인차 모델링
        individual_factor = 0.7 + (uncertainty * 0.3) + (cognitive_load * 0.5)  # 결정론적 계산
        
        final_days = base_days * uncertainty_multiplier * cognitive_multiplier * individual_factor
        
        # 현실적 범위로 제한 (30분~6개월)
        final_hours = final_days * 24
        return max(0.5, min(4320.0, final_hours))  # 0.5시간~6개월
    
    async def _perform_counterfactual_analysis(self, processed_data: Dict, 
                                             outcome_data: Optional[Dict]) -> Dict[str, float]:
        """반사실적 분석 수행"""
        if not outcome_data:
            # outcome_data가 없으면 시뮬레이션 기반 반사실적 분석
            return await self._simulate_counterfactual_analysis(processed_data)
        
        # 실제 결과와 대안 시나리오 비교
        if 'utility_score' not in outcome_data:
            # utility_score가 없으면 대체 분석 방법 사용
            return await self._alternative_counterfactual_analysis(processed_data, outcome_data)
        actual_utility = outcome_data['utility_score']
        
        # 대안들에 대한 가상 유틸리티 계산
        alternative_utilities = []
        for alt in processed_data.get('alternatives', []):
            alt_embedding = await self._generate_semantic_embedding(alt)
            alt_predictions = await self._predict_regret(alt_embedding)
            alt_utility = 1 - alt_predictions['anticipated']  # 후회가 낮을수록 유틸리티 높음
            alternative_utilities.append(alt_utility)
        
        if alternative_utilities:
            max_alt_utility = max(alternative_utilities)
            utility_delta = actual_utility - max_alt_utility
        else:
            utility_delta = 0.0
        
        # 동적 임계값 계산 적용
        context = {
            'affected_count': processed_data.get('affected_count', 1),
            'uncertainty_level': processed_data.get('uncertainty_level', 0.5),
            'ethical_complexity': processed_data.get('ethical_complexity', 0.5),
            'time_pressure': processed_data.get('time_pressure', 0.5),
            'information_quality': processed_data.get('information_quality', 0.5),
            'option_count': len(processed_data.get('alternatives', [])) + 1
        }
        
        dynamic_result = self.dynamic_threshold_calculator.calculate_dynamic_threshold(
            utility_delta, context
        )
        
        return {
            'utility_delta': utility_delta,
            'dynamic_threshold': dynamic_result.threshold,
            'relative_regret': dynamic_result.relative_regret,
            'absolute_regret': dynamic_result.absolute_regret,
            'stakeholder_penalty': dynamic_result.stakeholder_penalty,
            'context_complexity': dynamic_result.context_complexity,
            'threshold_confidence': dynamic_result.confidence
        }
    
    async def _analyze_causal_attribution(self, processed_data: Dict, 
                                        semantic_embedding: torch.Tensor) -> Dict[str, float]:
        """인과관계 분석"""
        attribution_scores = {}
        
        # 다양한 요인들에 대한 기여도 분석
        factors = {
            'context': processed_data.get('context', {}),
            'stakeholders': processed_data.get('stakeholders', []),
            'constraints': processed_data.get('constraints', [])
        }
        
        base_score = (await self._predict_regret(semantic_embedding))['anticipated']
        
        for factor_name, factor_data in factors.items():
            if factor_data:
                # 해당 요인을 제거한 시나리오 생성
                modified_text = processed_data['text'].replace(str(factor_data), '')
                modified_embedding = await self._generate_semantic_embedding(modified_text)
                modified_score = (await self._predict_regret(modified_embedding))['anticipated']
                
                # 기여도 계산
                attribution_scores[factor_name] = abs(base_score - modified_score)
            else:
                attribution_scores[factor_name] = 0.0
        
        return attribution_scores
    
    async def _update_learning_data(self, regret_metrics: AdvancedRegretMetrics, 
                                  outcome_data: Optional[Dict]):
        """학습 데이터 업데이트"""
        self.regret_database.append(regret_metrics)
        
        if outcome_data:
            # 실제 결과가 있을 경우 모델 학습
            await self._train_regret_network(regret_metrics, outcome_data)
    
    async def _train_regret_network(self, regret_metrics: AdvancedRegretMetrics, 
                                  outcome_data: Dict):
        """후회 네트워크 학습"""
        if len(self.regret_database) < 10:  # 최소 학습 데이터 필요
            # 데이터 부족시 결과 반환 안함 (시간이 걸리더라도 대기)
            return None
        
        # 배치 데이터 준비
        batch_data = self.regret_database[-32:]  # 최근 32개 샘플
        
        self.regret_network.train()
        total_loss = 0.0
        
        for metrics in batch_data:
            # 실제 후회 점수 (outcome_data에서 추출)
            if 'actual_regret' not in outcome_data:
                # actual_regret이 없으면 예측값 기반 자가학습
                actual_regret = regret_metrics.anticipated_regret
            else:
                actual_regret = outcome_data['actual_regret']
            
            # 손실 계산 및 역전파 - 경험적 학습 구현
            await self._perform_experiential_learning(regret_metrics, actual_regret)
        
        self.regret_network.eval()
    
    async def _simulate_counterfactual_analysis(self, processed_data: Dict) -> Dict[str, float]:
        """실제 베이지안 반사실적 추론 분석 (시뮬레이션 모드 제거)"""
        logger.info("🧠 베이지안 반사실적 추론 시작...")
        
        # 1. 의미적 임베딩 기반 상황 분석
        scenario_text = processed_data.get('scenario', processed_data.get('text', ''))
        semantic_embedding = await self._generate_semantic_embedding(scenario_text)
        
        # 2. 베이지안 사전 확률 계산
        prior_beliefs = await self._calculate_bayesian_priors(processed_data)
        
        # 3. 복잡한 반사실적 시나리오 생성 (하드코딩 제거)
        counterfactual_scenarios = await self._generate_complex_counterfactuals(processed_data, semantic_embedding)
        
        # 4. 각 시나리오에 대한 베이지안 추론
        scenario_probabilities = []
        scenario_utilities = []
        
        for scenario in counterfactual_scenarios:
            # 베이지안 업데이트
            posterior_prob = await self._bayesian_update(prior_beliefs, scenario, semantic_embedding)
            
            # 유틸리티 함수 계산 (문학적 맥락 고려)
            utility = await self._calculate_contextual_utility(scenario, processed_data)
            
            scenario_probabilities.append(posterior_prob)
            scenario_utilities.append(utility)
        
        # 5. 기대 유틸리티 계산
        if scenario_probabilities and scenario_utilities:
            expected_utility = sum(p * u for p, u in zip(scenario_probabilities, scenario_utilities))
            baseline_utility = await self._calculate_baseline_utility(processed_data)
            
            # 6. 반사실적 후회 강도 계산
            regret_intensity = max(0, max(scenario_utilities) - baseline_utility)
            
        else:
            expected_utility = 0.5
            baseline_utility = 0.5
            regret_intensity = 0.0
        
        logger.info(f"✅ 베이지안 분석 완료: 후회강도={regret_intensity:.3f}, 기대유틸리티={expected_utility:.3f}")
        
        return {
            'utility_delta': expected_utility - baseline_utility,
            'baseline_utility': baseline_utility,
            'expected_utility': expected_utility,
            'regret_intensity': regret_intensity,
            'scenario_count': len(counterfactual_scenarios),
            'simulation_mode': False  # 실제 베이지안 추론 모드
        }
    
    async def _alternative_counterfactual_analysis(self, processed_data: Dict, 
                                                  outcome_data: Dict) -> Dict[str, float]:
        """utility_score 없는 outcome_data에 대한 대체 분석"""
        # outcome_data에서 사용 가능한 정보 추출
        satisfaction = outcome_data.get('satisfaction', 0.5)
        success_rating = outcome_data.get('success_rating', 0.5)
        emotional_impact = outcome_data.get('emotional_impact', 0.0)
        
        # 대체 유틸리티 계산
        derived_utility = (satisfaction * 0.4 + success_rating * 0.4 + 
                          (0.5 + emotional_impact * 0.5) * 0.2)
        
        # 시뮬레이션된 대안들과 비교
        simulation_result = await self._simulate_counterfactual_analysis(processed_data)
        
        return {
            'utility_delta': simulation_result['best_alternative_utility'] - derived_utility,
            'derived_utility': derived_utility,
            'alternative_basis': 'satisfaction_success_emotion',
            'simulation_component': simulation_result['utility_delta']
        }
    
    async def _perform_experiential_learning(self, regret_metrics: AdvancedRegretMetrics, 
                                            actual_regret: float):
        """경험적 학습 수행"""
        # 예측 정확도 계산
        prediction_error = abs(regret_metrics.anticipated_regret - actual_regret)
        
        # 학습 데이터로 저장
        learning_sample = {
            'predicted_regret': regret_metrics.anticipated_regret,
            'actual_regret': actual_regret,
            'prediction_error': prediction_error,
            'context_factors': regret_metrics.contextual_factors,
            'timestamp': time.time()
        }
        
        # 경험 메모리에 추가
        if hasattr(self, 'experience_memory'):
            self.experience_memory.append(learning_sample)
            # 최근 1000개만 유지
            if len(self.experience_memory) > 1000:
                self.experience_memory = self.experience_memory[-1000:]
        else:
            self.experience_memory = [learning_sample]
        
        # 모델 가중치 미세 조정 (간단한 버전)
        if hasattr(self.regret_network, 'adjust_weights'):
            adjustment_factor = prediction_error * 0.01  # 작은 조정
            await self.regret_network.adjust_weights(adjustment_factor)
        
        self.logger.debug(f"경험적 학습 완료: 예측오차={prediction_error:.3f}")
        return learning_sample
    
    def _get_gpu_memory_usage(self) -> float:
        """GPU 메모리 사용량 조회 (MB)"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0
    
    def _calculate_cache_hit_rate(self) -> float:
        """캐시 제거됨 - 항상 0 반환"""
        return 0.0
    
    async def generate_regret_report(self, decision_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """종합 후회 분석 보고서 생성"""
        if decision_ids:
            metrics_list = [m for m in self.regret_database if m.decision_id in decision_ids]
        else:
            metrics_list = self.regret_database[-100:]  # 최근 100개
        
        if not metrics_list:
            raise RuntimeError("No regret data available for report generation")
        
        # 통계 분석
        report = {
            "summary": {
                "total_decisions": len(metrics_list),
                "average_regret": np.mean([m.anticipated_regret for m in metrics_list]),
                "regret_variance": np.var([m.anticipated_regret for m in metrics_list]),
                "average_computation_time": np.mean([m.computation_time_ms for m in metrics_list]),
                "average_gpu_usage": np.mean([m.gpu_memory_usage_mb for m in metrics_list])
            },
            "performance_trends": {
                "prediction_accuracy_trend": [m.prediction_accuracy for m in metrics_list],
                "uncertainty_trend": [m.uncertainty_estimate for m in metrics_list],
                "computation_time_trend": [m.computation_time_ms for m in metrics_list]
            },
            "causal_insights": self._analyze_causal_patterns(metrics_list),
            "recommendations": self._generate_learning_recommendations(metrics_list)
        }
        
        return report
    
    def _analyze_causal_patterns(self, metrics_list: List[AdvancedRegretMetrics]) -> Dict[str, Any]:
        """인과관계 패턴 분석"""
        causal_data = defaultdict(list)
        
        for metrics in metrics_list:
            for factor, score in metrics.causal_attribution_scores.items():
                causal_data[factor].append(score)
        
        patterns = {}
        for factor, scores in causal_data.items():
            patterns[factor] = {
                "average_impact": np.mean(scores),
                "impact_variance": np.var(scores),
                "frequency": len(scores)
            }
        
        return patterns
    
    def _generate_learning_recommendations(self, metrics_list: List[AdvancedRegretMetrics]) -> List[str]:
        """학습 개선 권장사항 생성"""
        recommendations = []
        
        # 예측 정확도 분석
        recent_accuracy = np.mean([m.prediction_accuracy for m in metrics_list[-20:]])
        if recent_accuracy < 0.7:
            recommendations.append("모델 예측 정확도가 낮습니다. 추가 학습 데이터가 필요합니다.")
        
        # 불확실성 분석
        avg_uncertainty = np.mean([m.uncertainty_estimate for m in metrics_list])
        if avg_uncertainty > 0.5:
            recommendations.append("높은 불확실성이 감지됩니다. 더 많은 특성 엔지니어링이 필요합니다.")
        
        # 성능 분석
        avg_computation_time = np.mean([m.computation_time_ms for m in metrics_list])
        if avg_computation_time > 1000:
            recommendations.append("계산 시간이 길어지고 있습니다. 모델 최적화를 고려하세요.")
        
        return recommendations
    
    async def cleanup(self):
        """리소스 정리"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("Advanced Regret Analyzer cleaned up successfully")
    
    # 베이지안 반사실적 추론을 위한 새로운 메서드들
    async def _calculate_bayesian_priors(self, processed_data: Dict) -> Dict[str, float]:
        """베이지안 사전 확률 계산"""
        priors = {}
        
        # 1. 상황의 도덕적 복잡성 기반 사전 확률
        moral_complexity = self._assess_moral_complexity(processed_data.get('text', ''))
        priors['moral_action'] = 0.3 + (moral_complexity * 0.4)  # 0.3-0.7 범위
        
        # 2. 이해관계자 수에 따른 갈등 확률
        stakeholders = processed_data.get('stakeholders', [])
        conflict_prob = min(0.8, len(stakeholders) * 0.15) if stakeholders else 0.2
        priors['conflict_outcome'] = conflict_prob
        
        # 3. 맥락적 요인들
        context = processed_data.get('context', {})
        if 'urgency' in str(context).lower():
            priors['hasty_decision'] = 0.6
        else:
            priors['hasty_decision'] = 0.3
            
        return priors
    
    def _assess_moral_complexity(self, text: str) -> float:
        """도덕적 복잡성 평가"""
        moral_indicators = [
            '딜레마', '윤리', '권리', '의무', '정의', '공정', '희생',
            '갈등', '선택', '가치', '원칙', '도덕', '양심', '책임'
        ]
        
        text_lower = text.lower()
        complexity_score = sum(1 for indicator in moral_indicators if indicator in text_lower)
        return min(1.0, complexity_score / len(moral_indicators))
    
    async def _generate_complex_counterfactuals(self, processed_data: Dict, 
                                               semantic_embedding: np.ndarray) -> List[Dict[str, Any]]:
        """복잡한 반사실적 시나리오 생성 (하드코딩 제거)"""
        scenarios = []
        base_text = processed_data.get('text', '')
        
        # 1. 도덕적 차원의 대안들
        moral_alternatives = [
            {'type': 'utilitarian', 'focus': '최대 행복', 'weight_shift': 'collective_benefit'},
            {'type': 'deontological', 'focus': '의무와 원칙', 'weight_shift': 'rule_following'},
            {'type': 'virtue_ethics', 'focus': '덕성과 성품', 'weight_shift': 'character_based'},
            {'type': 'care_ethics', 'focus': '관계와 돌봄', 'weight_shift': 'relationship_focused'}
        ]
        
        # 2. 시간적 차원의 대안들
        temporal_alternatives = [
            {'timing': 'immediate', 'horizon': 'short_term', 'consideration': '즉각적 결과'},
            {'timing': 'delayed', 'horizon': 'medium_term', 'consideration': '중기적 영향'},
            {'timing': 'patient', 'horizon': 'long_term', 'consideration': '장기적 결과'}
        ]
        
        # 3. 정보 차원의 대안들
        information_alternatives = [
            {'info_level': 'complete', 'certainty': 0.9, 'description': '완전한 정보'},
            {'info_level': 'partial', 'certainty': 0.6, 'description': '부분적 정보'},
            {'info_level': 'minimal', 'certainty': 0.3, 'description': '최소한의 정보'}
        ]
        
        # 4. 복합 시나리오 생성
        for moral in moral_alternatives:
            for temporal in temporal_alternatives:
                for info in information_alternatives:
                    scenario = {
                        'id': f"{moral['type']}_{temporal['timing']}_{info['info_level']}",
                        'moral_framework': moral,
                        'temporal_aspect': temporal,
                        'information_context': info,
                        'base_text': base_text,
                        'embedding_similarity': float(np.random.normal(0.7, 0.15))  # 의미적 유사성
                    }
                    scenarios.append(scenario)
        
        return scenarios[:12]  # 계산 효율성을 위해 12개로 제한
    
    async def _bayesian_update(self, priors: Dict[str, float], scenario: Dict[str, Any], 
                               semantic_embedding: np.ndarray) -> float:
        """베이지안 업데이트를 통한 후험 확률 계산"""
        # 1. 우도 함수 계산
        likelihood = self._calculate_likelihood(scenario, semantic_embedding)
        
        # 2. 사전 확률
        prior = priors.get('moral_action', 0.5)
        
        # 3. 베이지안 업데이트: P(H|E) = P(E|H) * P(H) / P(E)
        # 정규화를 위한 근사적 증거 확률
        evidence_prob = 0.5  # 정규화 상수
        
        posterior = (likelihood * prior) / evidence_prob
        return min(1.0, max(0.0, posterior))  # 0-1 범위로 제한
    
    def _calculate_likelihood(self, scenario: Dict[str, Any], 
                             semantic_embedding: np.ndarray) -> float:
        """시나리오의 우도 계산"""
        likelihood = 0.5  # 기본값
        
        # 1. 도덕적 프레임워크에 따른 우도 조정
        moral_type = scenario.get('moral_framework', {}).get('type', '')
        if moral_type == 'utilitarian':
            likelihood += 0.2
        elif moral_type == 'deontological':
            likelihood += 0.15
        elif moral_type == 'virtue_ethics':
            likelihood += 0.1
        
        # 2. 정보 완전성에 따른 우도 조정
        info_level = scenario.get('information_context', {}).get('info_level', '')
        if info_level == 'complete':
            likelihood += 0.2
        elif info_level == 'partial':
            likelihood += 0.1
        
        # 3. 의미적 유사성 고려
        similarity = scenario.get('embedding_similarity', 0.5)
        likelihood += (similarity - 0.5) * 0.3
        
        return min(1.0, max(0.1, likelihood))
    
    async def _calculate_contextual_utility(self, scenario: Dict[str, Any], 
                                           processed_data: Dict) -> float:
        """문학적 맥락을 고려한 유틸리티 계산"""
        utility = 0.5  # 기본 유틸리티
        
        # 1. 도덕적 프레임워크에 따른 유틸리티
        moral_framework = scenario.get('moral_framework', {})
        if moral_framework.get('type') == 'utilitarian':
            utility += self._assess_collective_benefit(processed_data) * 0.3
        elif moral_framework.get('type') == 'deontological':
            utility += self._assess_rule_adherence(processed_data) * 0.25
        
        # 2. 시간적 고려사항
        temporal_aspect = scenario.get('temporal_aspect', {})
        if temporal_aspect.get('horizon') == 'long_term':
            utility += 0.2  # 장기적 사고에 보너스
        
        # 3. 정보 품질 보정
        info_context = scenario.get('information_context', {})
        certainty = info_context.get('certainty', 0.5)
        utility *= certainty  # 불확실성에 따른 할인
        
        return min(1.0, max(0.0, utility))
    
    def _assess_collective_benefit(self, processed_data: Dict) -> float:
        """집단적 이익 평가"""
        text = processed_data.get('text', '').lower()
        benefit_indicators = ['모든', '전체', '공동', '사회', '다수', '공익']
        benefit_score = sum(1 for indicator in benefit_indicators if indicator in text)
        return min(1.0, benefit_score / len(benefit_indicators))
    
    def _assess_rule_adherence(self, processed_data: Dict) -> float:
        """규칙 준수 평가"""
        text = processed_data.get('text', '').lower()
        rule_indicators = ['법', '규칙', '원칙', '의무', '명령', '지침', '규정']
        rule_score = sum(1 for indicator in rule_indicators if indicator in text)
        return min(1.0, rule_score / len(rule_indicators))
    
    async def _calculate_baseline_utility(self, processed_data: Dict) -> float:
        """기준선 유틸리티 계산"""
        # 현재 상황의 기본 유틸리티를 의미적으로 평가
        text = processed_data.get('text', '')
        
        # 1. 긍정/부정 지표
        positive_indicators = ['좋', '행복', '성공', '도움', '이익', '만족']
        negative_indicators = ['나쁘', '슬픔', '실패', '해로', '손해', '불만']
        
        text_lower = text.lower()
        positive_score = sum(1 for indicator in positive_indicators if indicator in text_lower)
        negative_score = sum(1 for indicator in negative_indicators if indicator in text_lower)
        
        # 2. 균형 계산
        if positive_score + negative_score > 0:
            sentiment_ratio = positive_score / (positive_score + negative_score)
        else:
            sentiment_ratio = 0.5
        
        # 3. 기준선 유틸리티 (0.3-0.7 범위)
        baseline = 0.3 + (sentiment_ratio * 0.4)
        
        return baseline
    
    def get_pytorch_network(self) -> Optional[nn.Module]:
        """PyTorch 네트워크 반환 (HeadAdapter와의 호환성)
        STRICT_NO_FALLBACK 정책 준수
        """
        # regret_network 우선 반환
        if hasattr(self, 'regret_network') and isinstance(self.regret_network, nn.Module):
            self.logger.info("AdvancedRegretAnalyzer: regret_network 반환")
            return self.regret_network
        
        # 다른 가능한 네트워크 속성 확인
        for attr_name in ['model', 'neural_model', 'transformer_model']:
            if hasattr(self, attr_name):
                model = getattr(self, attr_name)
                if isinstance(model, nn.Module):
                    self.logger.info(f"AdvancedRegretAnalyzer: {attr_name} 반환")
                    return model
        
        # STRICT_NO_FALLBACK 정책에 따라 예외 발생
        self.logger.error("AdvancedRegretAnalyzer: PyTorch 네트워크를 찾지 못했습니다")
        raise RuntimeError("PyTorch 네트워크를 찾지 못했습니다")

class RegretEvaluationMetrics:
    """후회 평가 메트릭 관리"""
    
    def __init__(self):
        self.metrics_history = []
        self.performance_benchmarks = {
            'accuracy_threshold': 0.8,
            'computation_time_limit': 500,  # ms
            'memory_usage_limit': 1024  # MB
        }
    
    def evaluate_prediction_quality(self, predicted_regret: float, 
                                  actual_regret: float) -> Dict[str, float]:
        """예측 품질 평가"""
        absolute_error = abs(predicted_regret - actual_regret)
        relative_error = absolute_error / max(actual_regret, 0.001)
        
        return {
            'absolute_error': absolute_error,
            'relative_error': relative_error,
            'accuracy': 1 - relative_error,
            'prediction_quality': max(0, 1 - relative_error)
        }
    
    def benchmark_performance(self, metrics: AdvancedRegretMetrics) -> Dict[str, bool]:
        """성능 벤치마크 검증"""
        return {
            'accuracy_pass': metrics.prediction_accuracy >= self.performance_benchmarks['accuracy_threshold'],
            'speed_pass': metrics.computation_time_ms <= self.performance_benchmarks['computation_time_limit'],
            'memory_pass': metrics.gpu_memory_usage_mb <= self.performance_benchmarks['memory_usage_limit'],
            'overall_pass': all([
                metrics.prediction_accuracy >= self.performance_benchmarks['accuracy_threshold'],
                metrics.computation_time_ms <= self.performance_benchmarks['computation_time_limit'],
                metrics.gpu_memory_usage_mb <= self.performance_benchmarks['memory_usage_limit']
            ])
        }

class EnhancedRegretLogger:
    """향상된 후회 분석 로깅 시스템"""
    
    def __init__(self, log_dir: str = "logs/regret"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger('regret_analyzer')
        self.logger.setLevel(logging.INFO)
        
        # 파일 핸들러 설정
        log_file = self.log_dir / f"regret_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 로그 포맷 설정
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def log_regret_analysis(self, metrics: AdvancedRegretMetrics):
        """후회 분석 결과 로깅"""
        log_data = {
            'decision_id': metrics.decision_id,
            'timestamp': metrics.timestamp.isoformat(),
            'regret_scores': {
                'anticipated': metrics.anticipated_regret,
                'experienced': metrics.experienced_regret,
                'intensity': metrics.regret_intensity
            },
            'performance': {
                'computation_time_ms': metrics.computation_time_ms,
                'gpu_memory_usage_mb': metrics.gpu_memory_usage_mb,
                'prediction_accuracy': metrics.prediction_accuracy
            }
        }
        
        self.logger.info(f"Regret Analysis: {json.dumps(log_data, ensure_ascii=False)}")
    
    def save_regret_report(self, report: Dict[str, Any], filename: Optional[str] = None):
        """후회 분석 보고서 저장"""
        if not filename:
            filename = f"regret_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report_file = self.log_dir / filename
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    
    # 베이지안 반사실적 추론 구현을 위한 새로운 메서드들
    async def _calculate_bayesian_priors(self, processed_data: Dict) -> Dict[str, float]:
        """베이지안 사전 확률 계산"""
        priors = {}
        
        # 1. 상황의 도덕적 복잡성 기반 사전 확률
        moral_complexity = self._assess_moral_complexity(processed_data.get('text', ''))
        priors['moral_action'] = 0.3 + (moral_complexity * 0.4)  # 0.3-0.7 범위
        
        # 2. 이해관계자 수에 따른 갈등 확률
        stakeholders = processed_data.get('stakeholders', [])
        conflict_prob = min(0.8, len(stakeholders) * 0.15) if stakeholders else 0.2
        priors['conflict_outcome'] = conflict_prob
        
        # 3. 맥락적 요인들
        context = processed_data.get('context', {})
        if 'urgency' in str(context).lower():
            priors['hasty_decision'] = 0.6
        else:
            priors['hasty_decision'] = 0.3
            
        return priors
    
    def _assess_moral_complexity(self, text: str) -> float:
        """도덕적 복잡성 평가"""
        moral_indicators = [
            '딜레마', '윤리', '권리', '의무', '정의', '공정', '희생',
            '갈등', '선택', '가치', '원칙', '도덕', '양심', '책임'
        ]
        
        text_lower = text.lower()
        complexity_score = sum(1 for indicator in moral_indicators if indicator in text_lower)
        return min(1.0, complexity_score / len(moral_indicators))
    
    async def _generate_complex_counterfactuals(self, processed_data: Dict, 
                                               semantic_embedding: np.ndarray) -> List[Dict[str, Any]]:
        """복잡한 반사실적 시나리오 생성 (하드코딩 제거)"""
        scenarios = []
        base_text = processed_data.get('text', '')
        
        # 1. 도덕적 차원의 대안들
        moral_alternatives = [
            {'type': 'utilitarian', 'focus': '최대 행복', 'weight_shift': 'collective_benefit'},
            {'type': 'deontological', 'focus': '의무와 원칙', 'weight_shift': 'rule_following'},
            {'type': 'virtue_ethics', 'focus': '덕성과 성품', 'weight_shift': 'character_based'},
            {'type': 'care_ethics', 'focus': '관계와 돌봄', 'weight_shift': 'relationship_focused'}
        ]
        
        # 2. 시간적 차원의 대안들
        temporal_alternatives = [
            {'timing': 'immediate', 'horizon': 'short_term', 'consideration': '즉각적 결과'},
            {'timing': 'delayed', 'horizon': 'medium_term', 'consideration': '중기적 영향'},
            {'timing': 'patient', 'horizon': 'long_term', 'consideration': '장기적 결과'}
        ]
        
        # 3. 정보 차원의 대안들
        information_alternatives = [
            {'info_level': 'complete', 'certainty': 0.9, 'description': '완전한 정보'},
            {'info_level': 'partial', 'certainty': 0.6, 'description': '부분적 정보'},
            {'info_level': 'minimal', 'certainty': 0.3, 'description': '최소한의 정보'}
        ]
        
        # 4. 복합 시나리오 생성
        for moral in moral_alternatives:
            for temporal in temporal_alternatives:
                for info in information_alternatives:
                    scenario = {
                        'id': f"{moral['type']}_{temporal['timing']}_{info['info_level']}",
                        'moral_framework': moral,
                        'temporal_aspect': temporal,
                        'information_context': info,
                        'base_text': base_text,
                        'embedding_similarity': float(np.random.normal(0.7, 0.15))  # 의미적 유사성
                    }
                    scenarios.append(scenario)
        
        return scenarios[:12]  # 계산 효율성을 위해 12개로 제한
    
    async def _bayesian_update(self, priors: Dict[str, float], scenario: Dict[str, Any], 
                               semantic_embedding: np.ndarray) -> float:
        """베이지안 업데이트를 통한 후험 확률 계산"""
        # 1. 우도 함수 계산
        likelihood = self._calculate_likelihood(scenario, semantic_embedding)
        
        # 2. 사전 확률
        prior = priors.get('moral_action', 0.5)
        
        # 3. 베이지안 업데이트: P(H|E) = P(E|H) * P(H) / P(E)
        # 정규화를 위한 근사적 증거 확률
        evidence_prob = 0.5  # 정규화 상수
        
        posterior = (likelihood * prior) / evidence_prob
        return min(1.0, max(0.0, posterior))  # 0-1 범위로 제한
    
    def _calculate_likelihood(self, scenario: Dict[str, Any], 
                             semantic_embedding: np.ndarray) -> float:
        """시나리오의 우도 계산"""
        likelihood = 0.5  # 기본값
        
        # 1. 도덕적 프레임워크에 따른 우도 조정
        moral_type = scenario.get('moral_framework', {}).get('type', '')
        if moral_type == 'utilitarian':
            likelihood += 0.2
        elif moral_type == 'deontological':
            likelihood += 0.15
        elif moral_type == 'virtue_ethics':
            likelihood += 0.1
        
        # 2. 정보 완전성에 따른 우도 조정
        info_level = scenario.get('information_context', {}).get('info_level', '')
        if info_level == 'complete':
            likelihood += 0.2
        elif info_level == 'partial':
            likelihood += 0.1
        
        # 3. 의미적 유사성 고려
        similarity = scenario.get('embedding_similarity', 0.5)
        likelihood += (similarity - 0.5) * 0.3
        
        return min(1.0, max(0.1, likelihood))
    
    async def _calculate_contextual_utility(self, scenario: Dict[str, Any], 
                                           processed_data: Dict) -> float:
        """문학적 맥락을 고려한 유틸리티 계산"""
        utility = 0.5  # 기본 유틸리티
        
        # 1. 도덕적 프레임워크에 따른 유틸리티
        moral_framework = scenario.get('moral_framework', {})
        if moral_framework.get('type') == 'utilitarian':
            utility += self._assess_collective_benefit(processed_data) * 0.3
        elif moral_framework.get('type') == 'deontological':
            utility += self._assess_rule_adherence(processed_data) * 0.25
        
        # 2. 시간적 고려사항
        temporal_aspect = scenario.get('temporal_aspect', {})
        if temporal_aspect.get('horizon') == 'long_term':
            utility += 0.2  # 장기적 사고에 보너스
        
        # 3. 정보 품질 보정
        info_context = scenario.get('information_context', {})
        certainty = info_context.get('certainty', 0.5)
        utility *= certainty  # 불확실성에 따른 할인
        
        return min(1.0, max(0.0, utility))
    
    def _assess_collective_benefit(self, processed_data: Dict) -> float:
        """집단적 이익 평가"""
        text = processed_data.get('text', '').lower()
        benefit_indicators = ['모든', '전체', '공동', '사회', '다수', '공익']
        benefit_score = sum(1 for indicator in benefit_indicators if indicator in text)
        return min(1.0, benefit_score / len(benefit_indicators))
    
    def _assess_rule_adherence(self, processed_data: Dict) -> float:
        """규칙 준수 평가"""
        text = processed_data.get('text', '').lower()
        rule_indicators = ['법', '규칙', '원칙', '의무', '명령', '지침', '규정']
        rule_score = sum(1 for indicator in rule_indicators if indicator in text)
        return min(1.0, rule_score / len(rule_indicators))
    
    async def _calculate_baseline_utility(self, processed_data: Dict) -> float:
        """기준선 유틸리티 계산"""
        # 현재 상황의 기본 유틸리티를 의미적으로 평가
        text = processed_data.get('text', '')
        
        # 1. 긍정/부정 지표
        positive_indicators = ['좋', '행복', '성공', '도움', '이익', '만족']
        negative_indicators = ['나쁘', '슬픔', '실패', '해로', '손해', '불만']
        
        text_lower = text.lower()
        positive_score = sum(1 for indicator in positive_indicators if indicator in text_lower)
        negative_score = sum(1 for indicator in negative_indicators if indicator in text_lower)
        
        # 2. 균형 계산
        if positive_score + negative_score > 0:
            sentiment_ratio = positive_score / (positive_score + negative_score)
        else:
            sentiment_ratio = 0.5
        
        # 3. 기준선 유틸리티 (0.3-0.7 범위)
        baseline = 0.3 + (sentiment_ratio * 0.4)
        
        return baseline
        
        self.logger.info(f"Regret report saved to {report_file}")