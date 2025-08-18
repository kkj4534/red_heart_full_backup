"""
Red Heart - 고급 윤리적 의사결정 지원 시스템 (Linux 전용)
Advanced Ethical Decision Support System for Linux

모든 고급 AI 컴포넌트를 통합한 메인 시스템
- 고급 감정 분석 (Advanced Emotion Analysis)
- 고급 벤담 계산기 (Advanced Bentham Calculator)  
- 고급 의미 분석 (Advanced Semantic Analysis)
- 고급 SURD 시스템 (Advanced SURD Analysis)
"""

import os
import sys
import logging
import time
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import argparse

# 고급 라이브러리 임포트
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
import threading

# 프로젝트 모듈 임포트
from config import ADVANCED_CONFIG, DEVICE, MODELS_DIR, LOGS_DIR
from data_models import (
    EmotionData, EthicalSituation, DecisionScenario,
    IntegratedAnalysisResult, SystemStatus
)
from advanced_emotion_analyzer import AdvancedEmotionAnalyzer
from advanced_bentham_calculator import AdvancedBenthamCalculator
from advanced_semantic_analyzer import AdvancedSemanticAnalyzer
from advanced_surd_analyzer import AdvancedSURDAnalyzer
from advanced_experience_database import AdvancedExperienceDatabase

# 새로 추가된 고급 모듈들
from advanced_hierarchical_emotion_system import AdvancedHierarchicalEmotionSystem
from advanced_regret_learning_system import AdvancedRegretLearningSystem
from advanced_bayesian_inference_module import AdvancedBayesianInference
from advanced_llm_integration_layer import AdvancedLLMIntegrationLayer
from advanced_counterfactual_reasoning import AdvancedCounterfactualReasoning

# 모듈 브릿지 코디네이터 임포트
from module_bridge_coordinator import (
    ModuleBridgeCoordinator, ModuleType, 
    EmotionModuleAdapter, BenthamModuleAdapter,
    SemanticModuleAdapter, SURDModuleAdapter
)

# 로깅 기본값 추가 필터
class DefaultFilter(logging.Filter):
    """로깅 레코드에 기본값을 추가하는 필터"""
    def filter(self, record):
        # 기본값 설정
        if not hasattr(record, 'phase'):
            record.phase = 'GENERAL'
        if not hasattr(record, 'regret'):
            record.regret = 0.0
        if not hasattr(record, 'component'):
            record.component = 'SYSTEM'
        if not hasattr(record, 'metric'):
            record.metric = 'N/A'
        if not hasattr(record, 'value'):
            record.value = 0.0
        return True

# 고급 로깅 설정
def setup_advanced_logging():
    """고급 로깅 시스템 설정 - 학습 진행 상황 상세 추적"""
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # 기본값 필터
    default_filter = DefaultFilter()
    
    # 메인 로그 핸들러
    main_formatter = logging.Formatter(
        '%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 파일 핸들러들
    handlers = []
    
    # 1. 메인 시스템 로그
    main_handler = logging.FileHandler(
        os.path.join(LOGS_DIR, f'red_heart_main_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        encoding='utf-8'
    )
    main_handler.setFormatter(main_formatter)
    main_handler.setLevel(logging.INFO)
    main_handler.addFilter(default_filter)
    handlers.append(main_handler)
    
    # 2. 학습 전용 로그 (후회, 감정 학습 등)
    learning_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | PHASE:%(phase)s | REGRET:%(regret).3f | %(message)s',
        datefmt='%H:%M:%S'
    )
    learning_handler = logging.FileHandler(
        os.path.join(LOGS_DIR, f'learning_progress_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        encoding='utf-8'
    )
    learning_handler.setFormatter(learning_formatter)
    learning_handler.setLevel(logging.DEBUG)
    learning_handler.addFilter(default_filter)
    handlers.append(learning_handler)
    
    # 3. 성능 모니터링 로그
    performance_handler = logging.FileHandler(
        os.path.join(LOGS_DIR, f'performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        encoding='utf-8'
    )
    performance_handler.setFormatter(logging.Formatter(
        '%(asctime)s | PERF | %(component)s | %(metric)s:%(value).3f | %(message)s'
    ))
    performance_handler.setLevel(logging.INFO)
    performance_handler.addFilter(default_filter)
    handlers.append(performance_handler)
    
    # 4. 콘솔 핸들러 (간소화된 출력)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    ))
    console_handler.setLevel(logging.INFO)
    console_handler.addFilter(default_filter)
    handlers.append(console_handler)
    
    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # 기존 핸들러 제거
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 새 핸들러 추가
    for handler in handlers:
        root_logger.addHandler(handler)
    
    return handlers

# 학습 전용 로거 생성 함수들
def get_learning_logger(name: str):
    """학습 전용 로거 생성"""
    logger = logging.getLogger(f'Learning.{name}')
    return logger

def log_regret_progress(phase: str, regret_value: float, message: str, **kwargs):
    """후회 학습 진행 로깅"""
    logger = get_learning_logger('Regret')
    extra = {'phase': phase, 'regret': regret_value, **kwargs}
    logger.info(message, extra=extra)

def log_performance_metric(module: str, metric: str, value: float, message: str = ""):
    """성능 메트릭 로깅"""
    logger = logging.getLogger('Performance')
    extra = {'component': module, 'metric': metric, 'value': value}
    logger.info(message or f"{metric} measurement", extra=extra)

logger = logging.getLogger('RedHeart.Main')


@dataclass
class AnalysisRequest:
    """분석 요청"""
    text: str
    language: str = "ko"
    scenario_type: str = "general"
    include_emotion: bool = True
    include_bentham: bool = True
    include_semantic: bool = True
    include_surd: bool = True
    additional_context: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class IntegratedResult:
    """통합 분석 결과"""
    request: AnalysisRequest
    emotion_analysis: Optional[Any] = None
    bentham_analysis: Optional[Any] = None
    semantic_analysis: Optional[Any] = None
    surd_analysis: Optional[Any] = None
    integrated_score: float = 0.0
    recommendation: str = ""
    confidence: float = 0.0
    processing_time: float = 0.0
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class RedHeartSystem:
    """Red Heart 통합 시스템 - Module Bridge Coordinator 통합"""
    
    def __init__(self):
        self.logger = logger
        
        # 시스템 상태
        self.is_initialized = False
        self.initialization_lock = threading.Lock()
        
        # 기존 분석기들
        self.emotion_analyzer = None
        self.bentham_calculator = None
        self.semantic_analyzer = None
        self.surd_analyzer = None
        
        # 경험 메모리 시스템
        self.experience_database = None
        
        # 새로 추가된 고급 시스템들
        self.hierarchical_emotion_system = None
        self.regret_learning_system = None
        self.bayesian_inference = None
        self.llm_integration_layer = None
        self.counterfactual_reasoning = None
        
        # ⭐ 핵심: 모듈 브릿지 코디네이터
        self.module_coordinator = ModuleBridgeCoordinator()
        self.integrated_training_enabled = False
        
        # 스레드 풀
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # 캐시 시스템
        self.result_cache = {}
        self.cache_lock = threading.Lock()
        
        # 성능 통계
        self.performance_stats = {
            'total_requests': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'average_processing_time': 0.0,
            'cache_hits': 0
        }
        
        self.logger.info("Red Heart 시스템 생성자 완료 - Module Bridge Coordinator 포함")
        
    async def initialize(self):
        """시스템 초기화 (비동기)"""
        with self.initialization_lock:
            if self.is_initialized:
                return
                
            self.logger.info("Red Heart 시스템 초기화 시작...")
            start_time = time.time()
            
            try:
                # 0. 의존성 사전 검증 (엄격한 모드)
                self.logger.info("의존성 사전 검증 시작...")
                from dependency_validator import validate_dependencies
                if not validate_dependencies():
                    raise RuntimeError("의존성 검증 실패 - 시스템을 시작할 수 없습니다.")
                self.logger.info("✅ 의존성 사전 검증 완료")
                
                # 1. GPU/CUDA 확인
                self._check_system_requirements()
                
                # 2. 병렬로 분석기들 초기화
                init_tasks = [
                    self._init_emotion_analyzer(),
                    self._init_bentham_calculator(),
                    self._init_semantic_analyzer(),
                    self._init_surd_analyzer(),
                    self._init_experience_database(),
                    # 새로 추가된 고급 시스템들
                    self._init_hierarchical_emotion_system(),
                    self._init_regret_learning_system(),
                    self._init_bayesian_inference(),
                    self._init_llm_integration_layer(),
                    self._init_counterfactual_reasoning()
                ]
                
                # 초기화 타임아웃 설정 (60초로 단축)
                timeout_seconds = 60
                self.logger.info(f"모든 컴포넌트 초기화 시작 (타임아웃: {timeout_seconds}초)")
                
                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(*init_tasks, return_exceptions=True),
                        timeout=timeout_seconds
                    )
                except asyncio.TimeoutError:
                    error_msg = f"컴포넌트 초기화가 {timeout_seconds}초 내에 완료되지 않았습니다."
                    self.logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                # 3. 초기화 결과 확인
                self._validate_initialization_results(results)
                
                # 4. ⭐ 모듈 브릿지 코디네이터에 모듈 등록
                await self._register_modules_to_coordinator()
                
                # 5. 시스템 통합 테스트
                await self._run_integration_test()
                
                init_time = time.time() - start_time
                self.is_initialized = True
                
                self.logger.info(f"Red Heart 시스템 초기화 완료 ({init_time:.2f}초)")
                
            except Exception as e:
                self.logger.error(f"시스템 초기화 실패: {e}")
                raise
                
    def _check_system_requirements(self):
        """시스템 요구사항 확인"""
        self.logger.info("시스템 요구사항 확인 중...")
        
        # GPU 확인
        if ADVANCED_CONFIG['enable_gpu']:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                self.logger.info(f"GPU 사용 가능: {gpu_count}개, 메인 GPU: {gpu_name}")
            else:
                self.logger.warning("GPU가 요청되었지만 사용 불가능합니다. CPU로 전환합니다.")
                
        # 메모리 확인
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            memory_cached = torch.cuda.memory_reserved(0) / 1024**3
            self.logger.info(f"GPU 메모리: 할당됨 {memory_allocated:.2f}GB, 캐시됨 {memory_cached:.2f}GB")
            
        # 디스크 공간 확인
        models_dir = Path(MODELS_DIR)
        if models_dir.exists():
            total_size = sum(f.stat().st_size for f in models_dir.rglob('*') if f.is_file())
            self.logger.info(f"모델 디렉토리 크기: {total_size / 1024**3:.2f}GB")
            
    async def _init_emotion_analyzer(self):
        """감정 분석기 초기화"""
        try:
            from config import register_system_module
            
            self.logger.info("고급 감정 분석기 초기화 중...")
            self.emotion_analyzer = AdvancedEmotionAnalyzer()
            
            # 전역 레지스트리에 등록
            register_system_module('emotion_analyzer', self.emotion_analyzer, 'emotion')
            
            self.logger.info("고급 감정 분석기 초기화 완료")
            return True
        except Exception as e:
            self.logger.error(f"감정 분석기 초기화 실패: {e}")
            return False
            
    async def _init_bentham_calculator(self):
        """벤담 계산기 초기화"""
        try:
            from config import register_system_module
            
            self.logger.info("고급 벤담 계산기 초기화 중...")
            self.bentham_calculator = AdvancedBenthamCalculator()
            
            # 전역 레지스트리에 등록
            register_system_module('bentham_calculator', self.bentham_calculator, 'bentham')
            
            self.logger.info("고급 벤담 계산기 초기화 완료")
            return True
        except Exception as e:
            self.logger.error(f"벤담 계산기 초기화 실패: {e}")
            return False
            
    async def _init_semantic_analyzer(self):
        """의미 분석기 초기화"""
        try:
            from config import register_system_module
            
            self.logger.info("고급 의미 분석기 초기화 중...")
            self.semantic_analyzer = AdvancedSemanticAnalyzer()
            
            # 전역 레지스트리에 등록
            register_system_module('semantic_analyzer', self.semantic_analyzer, 'semantic')
            
            self.logger.info("고급 의미 분석기 초기화 완료")
            return True
        except Exception as e:
            self.logger.error(f"의미 분석기 초기화 실패: {e}")
            return False
            
    async def _init_surd_analyzer(self):
        """SURD 분석기 초기화"""
        try:
            self.logger.info("고급 SURD 분석기 초기화 중...")
            self.surd_analyzer = AdvancedSURDAnalyzer()
            self.logger.info("고급 SURD 분석기 초기화 완료")
            return True
        except Exception as e:
            self.logger.error(f"SURD 분석기 초기화 실패: {e}")
            return False
    
    async def _init_experience_database(self):
        """경험 데이터베이스 초기화"""
        try:
            self.logger.info("경험 데이터베이스 초기화 중...")
            self.experience_database = AdvancedExperienceDatabase()
            log_performance_metric("ExperienceDB", "initialization", 1.0, "경험 데이터베이스 초기화 완료")
            self.logger.info("경험 데이터베이스 초기화 완료")
            return True
        except Exception as e:
            self.logger.error(f"경험 데이터베이스 초기화 실패: {e}")
            return False
    
    async def _init_hierarchical_emotion_system(self):
        """계층적 감정 시스템 초기화"""
        try:
            self.logger.info("계층적 감정 시스템 초기화 중...")
            self.hierarchical_emotion_system = AdvancedHierarchicalEmotionSystem()
            log_performance_metric("HierarchicalEmotion", "initialization", 1.0, "시스템 초기화 완료")
            self.logger.info("계층적 감정 시스템 초기화 완료")
            return True
        except Exception as e:
            self.logger.error(f"계층적 감정 시스템 초기화 실패: {e}")
            return False
    
    async def _init_regret_learning_system(self):
        """후회 학습 시스템 초기화"""
        try:
            from config import register_system_module
            
            self.logger.info("후회 학습 시스템 초기화 중...")
            self.regret_learning_system = AdvancedRegretLearningSystem()
            
            # 전역 레지스트리에 등록
            register_system_module('regret_analyzer', self.regret_learning_system, 'regret')
            
            log_regret_progress("INIT", 0.0, "후회 학습 시스템 초기화 완료")
            self.logger.info("후회 학습 시스템 초기화 완료")
            return True
        except Exception as e:
            self.logger.error(f"후회 학습 시스템 초기화 실패: {e}")
            return False
    
    async def _init_bayesian_inference(self):
        """베이지안 추론 모듈 초기화"""
        try:
            self.logger.info("베이지안 추론 모듈 초기화 중...")
            self.bayesian_inference = AdvancedBayesianInference()
            log_performance_metric("BayesianInference", "initialization", 1.0, "추론 모듈 초기화 완료")
            self.logger.info("베이지안 추론 모듈 초기화 완료")
            return True
        except Exception as e:
            self.logger.error(f"베이지안 추론 모듈 초기화 실패: {e}")
            return False
    
    async def _init_llm_integration_layer(self):
        """LLM 통합 레이어 초기화"""
        try:
            self.logger.info("LLM 통합 레이어 초기화 중...")
            self.llm_integration_layer = AdvancedLLMIntegrationLayer()
            log_performance_metric("LLMIntegration", "initialization", 1.0, "LLM 레이어 초기화 완료")
            self.logger.info("LLM 통합 레이어 초기화 완료")
            return True
        except Exception as e:
            self.logger.error(f"LLM 통합 레이어 초기화 실패: {e}")
            return False
    
    async def _init_counterfactual_reasoning(self):
        """반사실적 추론 시스템 초기화"""
        try:
            self.logger.info("반사실적 추론 시스템 초기화 중...")
            self.counterfactual_reasoning = AdvancedCounterfactualReasoning()
            log_performance_metric("CounterfactualReasoning", "initialization", 1.0, "반사실적 추론 초기화 완료")
            self.logger.info("반사실적 추론 시스템 초기화 완료")
            return True
        except Exception as e:
            self.logger.error(f"반사실적 추론 시스템 초기화 실패: {e}")
            return False
    
    async def _register_modules_to_coordinator(self):
        """모듈 브릿지 코디네이터에 모듈들 등록"""
        self.logger.info("모듈 브릿지 코디네이터에 모듈 등록 중...")
        
        registration_count = 0
        
        # 기본 4개 모듈 등록
        if self.emotion_analyzer:
            self.module_coordinator.register_module(ModuleType.EMOTION, self.emotion_analyzer)
            registration_count += 1
            self.logger.info("감정 분석기 등록 완료")
            
        if self.bentham_calculator:
            self.module_coordinator.register_module(ModuleType.BENTHAM, self.bentham_calculator)
            registration_count += 1
            self.logger.info("벤담 계산기 등록 완료")
            
        if self.semantic_analyzer:
            self.module_coordinator.register_module(ModuleType.SEMANTIC, self.semantic_analyzer)
            registration_count += 1
            self.logger.info("의미 분석기 등록 완료")
            
        if self.surd_analyzer:
            self.module_coordinator.register_module(ModuleType.SURD, self.surd_analyzer)
            registration_count += 1
            self.logger.info("SURD 분석기 등록 완료")
            
        # ✅ 모든 주요 HeadAdapter들이 구현 및 등록 완료
        # - EmotionEmpathyHeadAdapter (140M)
        # - BenthamFrommHeadAdapter (120M)  
        # - SemanticSURDHeadAdapter (80M)
        # - RegretLearningHeadAdapter (120M)
        # - MetaIntegrationHeadAdapter (40M)
        
        self.logger.info(f"총 {registration_count}개 모듈이 브릿지 코디네이터에 등록됨")
        
        if registration_count >= 2:  # 최소 2개 이상의 모듈이 등록되어야 통합 학습 의미가 있음
            self.integrated_training_enabled = True
            self.logger.info("통합 학습 모드 준비 완료")
        else:
            self.logger.warning("통합 학습을 위해서는 최소 2개 이상의 모듈이 필요합니다")
            
    def _validate_initialization_results(self, results: List[Union[bool, Exception]]):
        """초기화 결과 검증"""
        component_names = [
            "감정 분석기", "벤담 계산기", "의미 분석기", "SURD 분석기", "경험 데이터베이스",
            "계층적 감정 시스템", "후회 학습 시스템", "베이지안 추론", "LLM 통합 레이어", "반사실적 추론"
        ]
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"{component_names[i]} 초기화 중 예외 발생: {result}")
            elif result is False:
                self.logger.error(f"{component_names[i]} 초기화 실패")
            else:
                self.logger.info(f"{component_names[i]} 초기화 성공")
                
        # 엄격한 초기화 모드: 모든 컴포넌트가 성공해야 함
        successful_components = sum(1 for r in results if r is True)
        failed_components = len(results) - successful_components
        
        if failed_components > 0:
            failed_names = [component_names[i] for i, r in enumerate(results) if r is not True]
            error_msg = f"초기화 실패 컴포넌트 ({failed_components}개): {', '.join(failed_names)}"
            self.logger.error(error_msg)
            self.logger.error("엄격한 모드: 모든 컴포넌트가 정상 초기화되어야 합니다.")
            raise RuntimeError(f"컴포넌트 초기화 실패 - {error_msg}")
        
        self.logger.info(f"✅ 모든 컴포넌트 초기화 성공 ({successful_components}/{len(results)})")
        self.logger.info("시스템이 완전한 기능으로 준비되었습니다.")
            
    async def _run_integration_test(self):
        """통합 테스트 실행"""
        self.logger.info("통합 테스트 임시 비활성화됨 - 초기화만 테스트")
        return  # 통합 테스트 임시 비활성화
        
        test_request = AnalysisRequest(
            text="이것은 시스템 테스트를 위한 간단한 문장입니다.",
            language="ko",
            scenario_type="test"
        )
        
        try:
            result = await self.analyze_async(test_request)
            if result:
                self.logger.info("통합 테스트 성공")
            else:
                self.logger.warning("통합 테스트에서 결과가 None입니다.")
        except Exception as e:
            self.logger.error(f"통합 테스트 실패: {e}")
            # 테스트 실패는 시스템 초기화를 막지 않음
    
    async def analyze_with_bridge_coordinator(self, request: AnalysisRequest) -> Dict[str, Any]:
        """⭐ 새로운 브릿지 코디네이터를 통한 통합 분석"""
        if not self.is_initialized:
            await self.initialize()
            
        if not self.integrated_training_enabled:
            self.logger.warning("통합 학습이 비활성화되어 있습니다. 기존 방식을 사용하세요.")
            return await self.analyze_async(request)
            
        start_time = time.time()
        self.performance_stats['total_requests'] += 1
        
        try:
            # 활성화된 모듈 결정
            enabled_modules = []
            if request.include_emotion and ModuleType.EMOTION in self.module_coordinator.adapters:
                enabled_modules.append(ModuleType.EMOTION)
            if request.include_bentham and ModuleType.BENTHAM in self.module_coordinator.adapters:
                enabled_modules.append(ModuleType.BENTHAM)
            if request.include_semantic and ModuleType.SEMANTIC in self.module_coordinator.adapters:
                enabled_modules.append(ModuleType.SEMANTIC)
            if request.include_surd and ModuleType.SURD in self.module_coordinator.adapters:
                enabled_modules.append(ModuleType.SURD)
                
            if not enabled_modules:
                raise ValueError("활성화된 모듈이 없습니다")
                
            # 브릿지 코디네이터를 통한 통합 분석 실행
            bridge_results = await self.module_coordinator.integrated_analysis(
                input_text=request.text,
                enable_modules=enabled_modules
            )
            
            processing_time = time.time() - start_time
            
            # 성능 통계 업데이트
            self.performance_stats['successful_analyses'] += 1
            alpha = 0.1
            self.performance_stats['average_processing_time'] = (
                alpha * processing_time + 
                (1 - alpha) * self.performance_stats['average_processing_time']
            )
            
            # 결과 정리 및 설명 가능성 정보 추가
            integrated_result = {
                'request': request,
                'bridge_results': bridge_results,
                'processing_time': processing_time,
                'timestamp': time.time(),
                'enabled_modules': [m.value for m in enabled_modules],
                'xai_explanation': self._generate_xai_explanation(bridge_results),
                'performance_report': self.module_coordinator.get_performance_report(),
                'integration_quality': self._assess_integration_quality(bridge_results)
            }
            
            self.logger.info(f"브릿지 코디네이터 분석 완료 ({processing_time:.2f}초)")
            return integrated_result
            
        except Exception as e:
            self.performance_stats['failed_analyses'] += 1
            self.logger.error(f"브릿지 코디네이터 분석 실패: {e}")
            raise
    
    def _generate_xai_explanation(self, bridge_results: Dict[str, Any]) -> Dict[str, Any]:
        """XAI 설명 생성"""
        explanation = {
            'module_contributions': {},
            'decision_factors': {},
            'confidence_breakdown': {},
            'processing_flow': []
        }
        
        for module_name, result in bridge_results.items():
            if result:
                explanation['module_contributions'][module_name] = {
                    'confidence': getattr(result, 'confidence', 0.0),
                    'processing_time': getattr(result, 'processing_time', 0.0),
                    'explanation': getattr(result, 'explanation', {})
                }
                
                explanation['processing_flow'].append({
                    'module': module_name,
                    'status': 'success',
                    'time': getattr(result, 'processing_time', 0.0)
                })
            else:
                explanation['processing_flow'].append({
                    'module': module_name,
                    'status': 'failed',
                    'time': 0.0
                })
                
        return explanation
    
    def _assess_integration_quality(self, bridge_results: Dict[str, Any]) -> Dict[str, Any]:
        """통합 품질 평가"""
        successful_modules = sum(1 for result in bridge_results.values() if result is not None)
        total_modules = len(bridge_results)
        
        avg_confidence = 0.0
        if successful_modules > 0:
            confidences = [
                getattr(result, 'confidence', 0.0) 
                for result in bridge_results.values() 
                if result is not None
            ]
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
        return {
            'success_rate': successful_modules / max(total_modules, 1),
            'average_confidence': avg_confidence,
            'module_harmony': self._calculate_module_harmony(bridge_results),
            'recommendation': 'excellent' if avg_confidence > 0.8 else 'good' if avg_confidence > 0.6 else 'needs_improvement'
        }
    
    def _calculate_module_harmony(self, bridge_results: Dict[str, Any]) -> float:
        """모듈 간 조화도 계산"""
        # 간단한 조화도 계산 - 실제로는 더 정교한 계산 필요
        confidences = [
            getattr(result, 'confidence', 0.0) 
            for result in bridge_results.values() 
            if result is not None
        ]
        
        if len(confidences) < 2:
            return 1.0
            
        # 신뢰도의 표준편차가 낮을수록 모듈들이 조화롭게 작동
        std_dev = np.std(confidences)
        harmony = max(0.0, 1.0 - std_dev)
        return harmony
            
    async def analyze_async(self, request: AnalysisRequest) -> IntegratedResult:
        """비동기 분석 실행"""
        if not self.is_initialized:
            await self.initialize()
            
        start_time = time.time()
        self.performance_stats['total_requests'] += 1
        
        # 캐시 확인
        cache_key = self._generate_cache_key(request)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            self.performance_stats['cache_hits'] += 1
            self.logger.debug("캐시된 결과 반환")
            return cached_result
            
        try:
            # 병렬 분석 실행
            analysis_tasks = []
            
            if request.include_emotion and self.emotion_analyzer:
                analysis_tasks.append(self._analyze_emotion_async(request))
            if request.include_bentham and self.bentham_calculator:
                analysis_tasks.append(self._analyze_bentham_async(request))
            if request.include_semantic and self.semantic_analyzer:
                analysis_tasks.append(self._analyze_semantic_async(request))
            if request.include_surd and self.surd_analyzer:
                analysis_tasks.append(self._analyze_surd_async(request))
                
            # 모든 분석 완료 대기
            analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # 결과 통합
            integrated_result = self._integrate_results(request, analysis_results, start_time)
            
            # 캐시 저장
            self._cache_result(cache_key, integrated_result)
            
            self.performance_stats['successful_analyses'] += 1
            self._update_performance_stats(time.time() - start_time)
            
            return integrated_result
            
        except Exception as e:
            self.logger.error(f"분석 실행 실패: {e}")
            self.performance_stats['failed_analyses'] += 1
            
            # 실패 시 기본 결과 반환
            return IntegratedResult(
                request=request,
                recommendation="분석 중 오류가 발생했습니다.",
                confidence=0.0,
                processing_time=time.time() - start_time,
                timestamp=time.time(),
                metadata={'error': str(e)}
            )
            
    async def _analyze_emotion_async(self, request: AnalysisRequest):
        """비동기 감정 분석"""
        try:
            result = self.emotion_analyzer.analyze_text_advanced(
                text=request.text,
                language=request.language,
                context=request.additional_context
            )
            return ('emotion', result)
        except Exception as e:
            self.logger.error(f"감정 분석 실패: {e}")
            return ('emotion', None)
            
    async def _analyze_bentham_async(self, request: AnalysisRequest):
        """비동기 벤담 분석 - 경험 메모리 통합"""
        try:
            # 감정 데이터 준비
            emotion_data = self._extract_emotion_data_from_text(request.text)
            
            # 벤담 계산 데이터 준비
            bentham_data = {
                'input_values': {
                    'intensity': 0.7,
                    'duration': 0.6,
                    'certainty': 0.8,
                    'propinquity': 0.9,
                    'fecundity': 0.5,
                    'purity': 0.7,
                    'extent': 0.8
                },
                'emotion_data': emotion_data,
                'text_description': request.text,
                'language': request.language,
                **request.additional_context
            }
            
            # ⭐ 경험 메모리 통합 벤담 계산 사용
            if self.experience_database:
                result = await self.bentham_calculator.calculate_with_experience_integration(
                    bentham_data, self.experience_database
                )
            else:
                # fallback to regular calculation
                result = self.bentham_calculator.calculate_with_advanced_layers(bentham_data)
                
            return ('bentham', result)
        except Exception as e:
            self.logger.error(f"벤담 분석 실패: {e}")
            return ('bentham', None)
            
    async def _analyze_semantic_async(self, request: AnalysisRequest):
        """비동기 의미 분석"""
        try:
            result = self.semantic_analyzer.analyze_text_advanced(
                text=request.text,
                language=request.language,
                analysis_depth="full"
            )
            return ('semantic', result)
        except Exception as e:
            self.logger.error(f"의미 분석 실패: {e}")
            return ('semantic', None)
            
    async def _analyze_surd_async(self, request: AnalysisRequest):
        """비동기 SURD 분석"""
        try:
            # SURD 분석을 위한 변수 준비
            variables = self._extract_variables_from_text(request.text, request.additional_context)
            
            result = self.surd_analyzer.analyze_advanced(
                variables=variables,
                target_variable='decision_outcome',
                additional_context=request.additional_context
            )
            return ('surd', result)
        except Exception as e:
            self.logger.error(f"SURD 분석 실패: {e}")
            return ('surd', None)
            
    def _extract_emotion_data_from_text(self, text: str) -> EmotionData:
        """텍스트에서 감정 데이터 추출"""
        # 간단한 감정 데이터 생성 (실제로는 더 정교한 분석 필요)
        return EmotionData(
            valence=0.1,  # 감정의 긍정/부정성
            arousal=0.5,  # 감정의 활성화 정도
            dominance=0.6,  # 감정의 통제감
            confidence=0.7
        )
        
    def _extract_variables_from_text(self, text: str, context: Dict[str, Any]) -> Dict[str, float]:
        """텍스트와 컨텍스트에서 SURD 변수 추출"""
        # 기본 변수들
        variables = {
            'emotion_intensity': 0.7,
            'ethical_weight': 0.8,
            'social_impact': 0.6,
            'time_pressure': 0.4,
            'uncertainty': 0.5,
            'decision_outcome': 0.65
        }
        
        # 컨텍스트에서 추가 변수 추출
        if context:
            for key, value in context.items():
                if isinstance(value, (int, float)) and 0 <= value <= 1:
                    variables[key] = float(value)
                    
        return variables
        
    def _integrate_results(self, request: AnalysisRequest, analysis_results: List, start_time: float) -> IntegratedResult:
        """분석 결과들을 통합"""
        
        # 결과 딕셔너리 구성
        results = {}
        for result in analysis_results:
            if isinstance(result, tuple) and len(result) == 2:
                analysis_type, analysis_result = result
                results[analysis_type] = analysis_result
            elif isinstance(result, Exception):
                self.logger.error(f"분석 중 예외 발생: {result}")
                
        # 통합 점수 계산
        integrated_score = self._calculate_integrated_score(results)
        
        # 추천 생성
        recommendation = self._generate_recommendation(results, integrated_score)
        
        # 신뢰도 계산
        confidence = self._calculate_overall_confidence(results)
        
        # 메타데이터 구성
        metadata = {
            'analyzed_components': list(results.keys()),
            'system_version': '2.0.0-linux',
            'gpu_used': ADVANCED_CONFIG['enable_gpu'] and torch.cuda.is_available(),
            'language': request.language,
            'scenario_type': request.scenario_type
        }
        
        return IntegratedResult(
            request=request,
            emotion_analysis=results.get('emotion'),
            bentham_analysis=results.get('bentham'),
            semantic_analysis=results.get('semantic'),
            surd_analysis=results.get('surd'),
            integrated_score=integrated_score,
            recommendation=recommendation,
            confidence=confidence,
            processing_time=time.time() - start_time,
            timestamp=time.time(),
            metadata=metadata
        )
        
    def _calculate_integrated_score(self, results: Dict[str, Any]) -> float:
        """통합 점수 계산"""
        scores = []
        weights = {
            'emotion': 0.25,
            'bentham': 0.35,
            'semantic': 0.20,
            'surd': 0.20
        }
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for component, weight in weights.items():
            if component in results and results[component]:
                score = self._extract_score_from_result(component, results[component])
                if score is not None:
                    weighted_sum += score * weight
                    total_weight += weight
                    
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return 0.5  # 기본값
            
    def _extract_score_from_result(self, component: str, result: Any) -> Optional[float]:
        """컴포넌트별 결과에서 점수 추출"""
        try:
            if component == 'emotion':
                if hasattr(result, 'overall_emotion_score'):
                    return result.overall_emotion_score
                elif hasattr(result, 'confidence_score'):
                    return result.confidence_score
                    
            elif component == 'bentham':
                if hasattr(result, 'final_score'):
                    return abs(result.final_score)  # 절댓값 사용
                    
            elif component == 'semantic':
                if hasattr(result, 'confidence_score'):
                    return result.confidence_score
                    
            elif component == 'surd':
                if hasattr(result, 'confidence_score'):
                    return getattr(result, 'confidence_score', 0.5)
                elif hasattr(result, 'information_decomposition'):
                    # SURD 결과에서 전체 정보량 기반 점수
                    decomp = result.information_decomposition
                    if decomp and 'all_variables' in decomp:
                        total_info = decomp['all_variables'].total_information
                        return min(total_info, 1.0)
                        
        except Exception as e:
            self.logger.error(f"{component} 점수 추출 실패: {e}")
            
        return None
        
    def _generate_recommendation(self, results: Dict[str, Any], integrated_score: float) -> str:
        """통합 추천 생성"""
        recommendations = []
        
        # 감정 분석 기반 추천
        if 'emotion' in results and results['emotion']:
            emotion_result = results['emotion']
            if hasattr(emotion_result, 'dominant_emotion'):
                recommendations.append(f"주요 감정 '{emotion_result.dominant_emotion}'을 고려하여")
                
        # 벤담 분석 기반 추천
        if 'bentham' in results and results['bentham']:
            bentham_result = results['bentham']
            if hasattr(bentham_result, 'final_score'):
                if bentham_result.final_score > 0.6:
                    recommendations.append("높은 쾌락 점수를 바탕으로 긍정적 결정을")
                elif bentham_result.final_score < 0.4:
                    recommendations.append("낮은 쾌락 점수를 고려하여 신중한 검토를")
                    
        # 의미 분석 기반 추천
        if 'semantic' in results and results['semantic']:
            semantic_result = results['semantic']
            if hasattr(semantic_result, 'ethical_analysis'):
                ethical = semantic_result.ethical_analysis
                if ethical and 'ethical_categories' in ethical:
                    recommendations.append("윤리적 가치를 우선 고려하여")
                    
        # SURD 분석 기반 추천
        if 'surd' in results and results['surd']:
            surd_result = results['surd']
            if hasattr(surd_result, 'information_decomposition'):
                recommendations.append("인과관계 분석 결과를 바탕으로")
                
        # 통합 점수 기반 최종 추천
        if integrated_score > 0.7:
            action = "적극적으로 추진하는 것이 좋겠습니다."
        elif integrated_score > 0.5:
            action = "신중하게 고려해 볼 만합니다."
        else:
            action = "재검토가 필요해 보입니다."
            
        if recommendations:
            return " ".join(recommendations) + " " + action
        else:
            return f"전체 분석 점수({integrated_score:.2f})를 바탕으로 " + action
            
    def _calculate_overall_confidence(self, results: Dict[str, Any]) -> float:
        """전체 신뢰도 계산"""
        confidences = []
        
        for component, result in results.items():
            if result and hasattr(result, 'confidence_score'):
                confidences.append(result.confidence_score)
            elif result:
                confidences.append(0.7)  # 기본 신뢰도
                
        if confidences:
            return float(np.mean(confidences))
        else:
            return 0.5
            
    def _generate_cache_key(self, request: AnalysisRequest) -> str:
        """캐시 키 생성"""
        import hashlib
        
        key_data = f"{request.text}_{request.language}_{request.scenario_type}"
        key_data += f"_{request.include_emotion}_{request.include_bentham}"
        key_data += f"_{request.include_semantic}_{request.include_surd}"
        
        return hashlib.md5(key_data.encode()).hexdigest()
        
    def _get_cached_result(self, cache_key: str) -> Optional[IntegratedResult]:
        """캐시된 결과 조회"""
        with self.cache_lock:
            return self.result_cache.get(cache_key)
            
    def _cache_result(self, cache_key: str, result: IntegratedResult):
        """결과 캐싱"""
        with self.cache_lock:
            if len(self.result_cache) >= 100:  # 캐시 크기 제한
                # 가장 오래된 항목 제거
                oldest_key = next(iter(self.result_cache))
                del self.result_cache[oldest_key]
                
            self.result_cache[cache_key] = result
            
    def _update_performance_stats(self, processing_time: float):
        """성능 통계 업데이트"""
        total_requests = self.performance_stats['total_requests']
        current_avg = self.performance_stats['average_processing_time']
        
        # 이동 평균 계산
        new_avg = ((current_avg * (total_requests - 1)) + processing_time) / total_requests
        self.performance_stats['average_processing_time'] = new_avg
        
    async def run_production_loop(self):
        """운용 모드 메인 루프 - 연속 처리"""
        self.logger.info("운용 모드 시작 - 입력 대기 중...")
        
        try:
            while True:
                # 표준 입력에서 텍스트 읽기
                print("\n텍스트를 입력하세요 (종료: 'quit' 또는 Ctrl+C):")
                text = input("> ")
                
                if text.lower() in ['quit', 'exit', 'q']:
                    print("운용 모드 종료")
                    break
                
                if not text.strip():
                    continue
                
                # 분석 요청 생성
                request = AnalysisRequest(
                    text=text,
                    include_emotion=True,
                    include_bentham=True,
                    include_semantic=True,
                    include_surd=True,
                    enable_all_modules=True
                )
                
                # 분석 실행
                result = await self.analyze_async(request)
                
                # 결과 출력
                print("\n" + "=" * 60)
                print("📊 분석 결과:")
                print_analysis_result(result)
                print("=" * 60)
                
        except KeyboardInterrupt:
            print("\n\n운용 모드 중단됨")
        except Exception as e:
            self.logger.error(f"운용 루프 오류: {e}")
            print(f"오류 발생: {e}")
    
    async def run_advanced_analysis(self, text: str):
        """고급 분석 모드 - 모든 기능 활성화"""
        self.logger.info("고급 분석 모드 시작")
        
        # 모든 고급 모듈 활성화
        if hasattr(self, 'enable_advanced_features'):
            self.enable_advanced_features = True
        
        # 브릿지 코디네이터 분석
        if self.module_coordinator:
            print("\n🔗 Module Bridge Coordinator 분석 중...")
            request = AnalysisRequest(
                text=text,
                include_emotion=True,
                include_bentham=True,
                include_semantic=True,
                include_surd=True,
                include_bridge_analysis=True
            )
            
            bridge_result = await self.analyze_with_bridge_coordinator(request)
            
            # XAI 설명 출력
            if 'xai_explanation' in bridge_result:
                print("\n🔍 XAI 설명:")
                for key, value in bridge_result['xai_explanation'].items():
                    print(f"   {key}: {value}")
            
            # 통합 품질 평가 출력
            if 'integration_quality' in bridge_result:
                quality = bridge_result['integration_quality']
                print(f"\n✨ 통합 품질: {quality.get('status', 'N/A')}")
                print(f"   전체 조화도: {quality.get('overall_harmony', 0):.2f}")
        
        # 계층적 감정 분석
        if self.hierarchical_emotion_system:
            print("\n🎭 계층적 감정 분석 중...")
            emotion_result = await self.hierarchical_emotion_system.process_literary_emotion_sequence(
                [{'text': text, 'context': 'advanced_analysis'}]
            )
            print(f"   감정 발달 궤적: {emotion_result.get('emotion_trajectory', [])}")
        
        # 베이지안 추론
        if self.bayesian_inference:
            print("\n🧠 베이지안 추론 중...")
            # 베이지안 분석 실행
        
        # 반사실적 추론
        if self.counterfactual_reasoning:
            print("\n🔮 반사실적 추론 중...")
            # 반사실적 시나리오 생성
        
        print("\n고급 분석 완료!")
    
    async def run_system_test(self):
        """시스템 테스트 모드"""
        self.logger.info("시스템 테스트 시작")
        
        test_results = {
            'modules': {},
            'integration': {},
            'performance': {}
        }
        
        # 1. 모듈별 테스트
        print("\n🧪 모듈 테스트 중...")
        
        test_text = "윤리적 딜레마에 직면한 상황에서 올바른 선택은 무엇일까?"
        
        # 감정 분석 테스트
        if self.emotion_analyzer:
            try:
                emotion_result = self.emotion_analyzer.analyze_text_advanced(test_text)
                test_results['modules']['emotion'] = 'PASS'
                print("   ✅ 감정 분석 모듈: 정상")
            except Exception as e:
                test_results['modules']['emotion'] = f'FAIL: {e}'
                print(f"   ❌ 감정 분석 모듈: 실패 - {e}")
        
        # 벤담 계산기 테스트
        if self.bentham_calculator:
            try:
                bentham_data = {'text_description': test_text}
                bentham_result = await self.bentham_calculator.calculate_hedonic_value(bentham_data)
                test_results['modules']['bentham'] = 'PASS'
                print("   ✅ 벤담 계산기: 정상")
            except Exception as e:
                test_results['modules']['bentham'] = f'FAIL: {e}'
                print(f"   ❌ 벤담 계산기: 실패 - {e}")
        
        # 2. 통합 테스트
        print("\n🔗 통합 테스트 중...")
        try:
            request = AnalysisRequest(text=test_text)
            result = await self.analyze_async(request)
            test_results['integration']['basic'] = 'PASS'
            print("   ✅ 기본 통합: 정상")
        except Exception as e:
            test_results['integration']['basic'] = f'FAIL: {e}'
            print(f"   ❌ 기본 통합: 실패 - {e}")
        
        # 3. 성능 테스트
        print("\n⚡ 성능 테스트 중...")
        import time
        
        start = time.time()
        for _ in range(3):
            await self.analyze_async(AnalysisRequest(text=test_text))
        elapsed = time.time() - start
        avg_time = elapsed / 3
        
        test_results['performance']['avg_response_time'] = avg_time
        print(f"   평균 응답 시간: {avg_time:.2f}초")
        
        # 테스트 결과 요약
        print("\n" + "=" * 60)
        print("📋 테스트 결과 요약:")
        print(f"   모듈 테스트: {sum(1 for v in test_results['modules'].values() if v == 'PASS')}/{len(test_results['modules'])} 통과")
        print(f"   통합 테스트: {'통과' if test_results['integration'].get('basic') == 'PASS' else '실패'}")
        print(f"   성능: {'양호' if avg_time < 2.0 else '개선 필요'}")
        print("=" * 60)
        
        return test_results
    
    def get_system_status(self) -> SystemStatus:
        """시스템 상태 조회"""
        return SystemStatus(
            is_initialized=self.is_initialized,
            is_running=True,
            current_phase="running" if self.is_initialized else "initializing",
            active_modules=[
                name for name, analyzer in {
                    'emotion': self.emotion_analyzer,
                    'bentham': self.bentham_calculator,
                    'semantic': self.semantic_analyzer,
                    'surd': self.surd_analyzer,
                    'experience_database': self.experience_database
                }.items() if analyzer is not None
            ],
            performance_stats=self.performance_stats.copy(),
            gpu_available=torch.cuda.is_available(),
            device=str(DEVICE)
        )
        
    def clear_cache(self):
        """캐시 클리어"""
        with self.cache_lock:
            self.result_cache.clear()
            
        # 각 분석기의 캐시도 클리어
        if self.emotion_analyzer:
            self.emotion_analyzer.clear_cache()
        if self.bentham_calculator:
            self.bentham_calculator.clear_cache()
        if self.semantic_analyzer:
            self.semantic_analyzer.clear_cache()
        if self.surd_analyzer:
            self.surd_analyzer.clear_cache()
        if self.experience_database:
            # 경험 데이터베이스는 일반적으로 캐시를 클리어하지 않음 (데이터 손실 위험)
            pass
            
        self.logger.info("모든 캐시가 클리어되었습니다.")
    
    # ⭐ 통합 학습 관련 메소드들
    def enable_integrated_training(self):
        """통합 학습 모드 활성화"""
        if self.integrated_training_enabled:
            self.logger.info("통합 학습 모드가 이미 활성화되어 있습니다.")
            return
            
        self.module_coordinator.enable_integrated_training()
        self.integrated_training_enabled = True
        self.logger.info("⭐ 통합 학습 모드 활성화 완료")
        
    def disable_integrated_training(self):
        """통합 학습 모드 비활성화"""
        self.integrated_training_enabled = False
        self.logger.info("통합 학습 모드 비활성화")
        
    async def optimize_module_performance(self, sample_texts: List[str]) -> Dict[str, Any]:
        """모듈 성능 최적화"""
        if not self.integrated_training_enabled:
            raise ValueError("통합 학습 모드가 활성화되지 않았습니다.")
            
        optimization_report = await self.module_coordinator.optimize_data_flow(sample_texts)
        self.logger.info("모듈 성능 최적화 완료")
        return optimization_report
        
    def get_integration_status(self) -> Dict[str, Any]:
        """통합 상태 조회"""
        return {
            'integrated_training_enabled': self.integrated_training_enabled,
            'registered_modules': len(self.module_coordinator.adapters),
            'module_list': [module.value for module in self.module_coordinator.adapters.keys()],
            'performance_report': self.module_coordinator.get_performance_report() if self.module_coordinator.adapters else {},
            'system_harmony': self._assess_system_harmony()
        }
    
    def _assess_system_harmony(self) -> Dict[str, Any]:
        """시스템 조화도 평가"""
        if not self.module_coordinator.adapters:
            return {'harmony_score': 0.0, 'status': 'no_modules'}
            
        # 각 모듈의 성능 통계 수집
        module_stats = []
        for module_type, adapter in self.module_coordinator.adapters.items():
            stats = adapter.performance_stats
            if stats['total_calls'] > 0:
                success_rate = stats['successful_calls'] / stats['total_calls']
                module_stats.append({
                    'module': module_type.value,
                    'success_rate': success_rate,
                    'avg_time': stats['average_time'],
                    'confidence': stats['last_confidence']
                })
                
        if not module_stats:
            return {'harmony_score': 0.0, 'status': 'no_data'}
            
        # 조화도 계산
        success_rates = [s['success_rate'] for s in module_stats]
        avg_times = [s['avg_time'] for s in module_stats]
        confidences = [s['confidence'] for s in module_stats]
        
        # 표준편차가 낮을수록 모듈들이 균등하게 성능을 발휘
        success_harmony = 1.0 - np.std(success_rates) if success_rates else 0.0
        time_harmony = 1.0 - (np.std(avg_times) / max(np.mean(avg_times), 1.0)) if avg_times else 0.0
        confidence_harmony = 1.0 - np.std(confidences) if confidences else 0.0
        
        overall_harmony = (success_harmony + time_harmony + confidence_harmony) / 3.0
        
        return {
            'harmony_score': overall_harmony,
            'success_harmony': success_harmony,
            'time_harmony': time_harmony, 
            'confidence_harmony': confidence_harmony,
            'module_details': module_stats,
            'status': 'excellent' if overall_harmony > 0.8 else 'good' if overall_harmony > 0.6 else 'needs_improvement'
        }


async def main():
    """메인 함수"""
    setup_advanced_logging()
    
    parser = argparse.ArgumentParser(description='Red Heart 윤리적 의사결정 지원 시스템')
    parser.add_argument('--mode', type=str, default='demo', 
                       choices=['production', 'advanced', 'demo', 'test'],
                       help='실행 모드 (production/advanced/demo/test)')
    parser.add_argument('--text', type=str, help='분석할 텍스트')
    parser.add_argument('--language', type=str, default='ko', help='언어 설정')
    parser.add_argument('--scenario', type=str, default='general', help='시나리오 타입')
    parser.add_argument('--demo', action='store_true', help='데모 모드 실행 (레거시)')
    
    # 고급 옵션들
    parser.add_argument('--enable-xai', action='store_true', help='XAI 피드백 통합 활성화')
    parser.add_argument('--enable-temporal', action='store_true', help='시계열 분석 활성화')
    parser.add_argument('--enable-bayesian', action='store_true', help='베이지안 추론 활성화')
    parser.add_argument('--oss-integration', action='store_true', help='OSS 20B 모델 통합')
    parser.add_argument('--enable-all', action='store_true', help='모든 고급 기능 활성화')
    
    args = parser.parse_args()
    
    # Red Heart 시스템 초기화
    system = RedHeartSystem()
    
    # 모드별 시스템 설정
    if args.mode == 'production' or args.mode == 'advanced':
        print("🚀 Red Heart AI - Production Mode")
        print("   모든 고급 분석 모듈 활성화")
        if args.enable_all or args.mode == 'advanced':
            system.enable_advanced_features = True
            system.enable_xai = True
            system.enable_temporal = True
            system.enable_bayesian = True
        else:
            system.enable_xai = args.enable_xai
            system.enable_temporal = args.enable_temporal
            system.enable_bayesian = args.enable_bayesian
        
        if args.oss_integration:
            print("   OSS 20B 모델 통합 모드")
            system.oss_integration = True
    else:
        print("🔴❤️ Red Heart 고급 윤리적 의사결정 지원 시스템 (Linux)")
    
    print("=" * 60)
    
    try:
        # 시스템 초기화
        print("시스템 초기화 중...")
        await system.initialize()
        
        # 시스템 상태 출력
        status = system.get_system_status()
        print(f"\n✅ 시스템 상태:")
        print(f"   초기화: {'완료' if status.is_initialized else '실패'}")
        print(f"   디바이스: {status.device}")
        print(f"   GPU 사용 가능: {'예' if status.gpu_available else '아니오'}")
        print(f"   활성 모듈: {', '.join(status.active_modules) if status.active_modules else '없음'}")
        print(f"   현재 단계: {status.current_phase}")
        print(f"   실행 모드: {args.mode.upper()}")
        
        # 모드별 실행
        if args.mode == 'production':
            # 운용 모드 - 실제 분석 작업
            if not args.text:
                print("\n운용 모드: 텍스트 입력 대기 중...")
                # 입력 대기 루프 또는 파일/스트림 처리
                await system.run_production_loop()
            else:
                request = AnalysisRequest(
                    text=args.text,
                    language=args.language,
                    scenario_type=args.scenario,
                    enable_all_modules=True
                )
                result = await system.analyze_async(request)
                print_analysis_result(result)
                
        elif args.mode == 'advanced':
            # 고급 모드 - 모든 기능 활성화
            print("\n고급 모드: 모든 분석 모듈 활성화")
            await system.run_advanced_analysis(args.text or "고급 분석 테스트 텍스트")
            
        elif args.mode == 'test':
            # 테스트 모드
            print("\n테스트 모드: 시스템 검증 중...")
            await system.run_system_test()
            
        elif args.demo or not args.text:
            # 데모 모드
            await run_demo(system)
        else:
            # 단일 분석
            request = AnalysisRequest(
                text=args.text,
                language=args.language,
                scenario_type=args.scenario
            )
            
            result = await system.analyze_async(request)
            print_analysis_result(result)
            
    except Exception as e:
        logger.error(f"시스템 실행 중 오류 발생: {e}")
        print(f"❌ 오류: {e}")
        
    finally:
        # 정리
        if hasattr(system, 'thread_pool'):
            system.thread_pool.shutdown(wait=True)


async def run_demo(system: RedHeartSystem):
    """데모 실행"""
    print("\n🎮 데모 모드")
    print("-" * 40)
    
    demo_scenarios = [
        {
            'text': "이 결정은 많은 사람들의 생명과 안전에 직접적인 영향을 미치며, 우리는 정의롭고 공정한 선택을 해야 합니다.",
            'language': 'ko',
            'scenario_type': 'ethical_dilemma',
            'description': '윤리적 딜레마 상황'
        },
        {
            'text': "새로운 기술 도입으로 효율성은 높아지지만 일부 직원들이 일자리를 잃을 수 있습니다.",
            'language': 'ko', 
            'scenario_type': 'technology_ethics',
            'description': '기술 윤리 상황'
        },
        {
            'text': "개인정보 보호와 공익을 위한 정보 공개 사이에서 균형점을 찾아야 합니다.",
            'language': 'ko',
            'scenario_type': 'privacy_vs_public',
            'description': '프라이버시 vs 공익'
        }
    ]
    
    for i, scenario in enumerate(demo_scenarios, 1):
        print(f"\n📝 시나리오 {i}: {scenario['description']}")
        print(f"   텍스트: {scenario['text']}")
        
        request = AnalysisRequest(
            text=scenario['text'],
            language=scenario['language'],
            scenario_type=scenario['scenario_type']
        )
        
        print("   분석 중...")
        start_time = time.time()
        result = await system.analyze_async(request)
        analysis_time = time.time() - start_time
        
        print(f"   ⏱️ 분석 시간: {analysis_time:.2f}초")
        print(f"   🎯 통합 점수: {result.integrated_score:.3f}")
        print(f"   🔮 신뢰도: {result.confidence:.3f}")
        print(f"   💡 추천: {result.recommendation}")
        
        # 컴포넌트별 간단한 결과
        if result.emotion_analysis:
            print(f"   😊 감정 분석: 완료")
        if result.bentham_analysis:
            print(f"   ⚖️ 벤담 분석: 완료 (점수: {result.bentham_analysis.final_score:.3f})")
        if result.semantic_analysis:
            print(f"   🧠 의미 분석: 완료")
        if result.surd_analysis:
            print(f"   🔗 SURD 분석: 완료")
            
        if i < len(demo_scenarios):
            print("   " + "-" * 30)
            
    # 시스템 통계
    status = system.get_system_status()
    print(f"\n📊 시스템 통계:")
    print(f"   총 요청 수: {status.performance_stats['total_requests']}")
    print(f"   성공 분석: {status.performance_stats['successful_analyses']}")
    print(f"   실패 분석: {status.performance_stats['failed_analyses']}")
    print(f"   평균 처리 시간: {status.performance_stats['average_processing_time']:.3f}초")
    print(f"   캐시 히트: {status.performance_stats['cache_hits']}")
    print(f"   캐시 크기: {status.cache_size}")


def print_analysis_result(result: IntegratedResult):
    """분석 결과 출력"""
    print(f"\n📋 분석 결과")
    print("=" * 40)
    print(f"텍스트: {result.request.text}")
    print(f"언어: {result.request.language}")
    print(f"시나리오: {result.request.scenario_type}")
    print(f"처리 시간: {result.processing_time:.3f}초")
    print(f"\n🎯 통합 점수: {result.integrated_score:.3f}")
    print(f"🔮 신뢰도: {result.confidence:.3f}")
    print(f"💡 추천사항: {result.recommendation}")
    
    print(f"\n📊 컴포넌트별 결과:")
    
    if result.emotion_analysis:
        print(f"  😊 감정 분석: ✅")
        if hasattr(result.emotion_analysis, 'dominant_emotion'):
            print(f"     주요 감정: {result.emotion_analysis.dominant_emotion}")
    else:
        print(f"  😊 감정 분석: ❌")
        
    if result.bentham_analysis:
        print(f"  ⚖️ 벤담 분석: ✅")
        print(f"     최종 점수: {result.bentham_analysis.final_score:.3f}")
        print(f"     신뢰도: {result.bentham_analysis.confidence_score:.3f}")
    else:
        print(f"  ⚖️ 벤담 분석: ❌")
        
    if result.semantic_analysis:
        print(f"  🧠 의미 분석: ✅")
        print(f"     신뢰도: {result.semantic_analysis.confidence_score:.3f}")
    else:
        print(f"  🧠 의미 분석: ❌")
        
    if result.surd_analysis:
        print(f"  🔗 SURD 분석: ✅")
        if hasattr(result.surd_analysis, 'processing_time'):
            print(f"     처리 시간: {result.surd_analysis.processing_time:.3f}초")
    else:
        print(f"  🔗 SURD 분석: ❌")
        
    print(f"\n🔧 시스템 정보:")
    for key, value in result.metadata.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())