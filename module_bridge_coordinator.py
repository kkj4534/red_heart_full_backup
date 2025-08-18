"""
Red Heart Linux - Module Bridge Coordinator
모듈 간 통합 학습을 위한 XAI 기반 중재자/어댑터 시스템

핵심 아이디어:
1. 각 모듈의 독립성 유지 (XAI 설명가능성 보존)
2. 표준화된 인터페이스를 통한 모듈 간 데이터 흐름
3. 통합 학습을 위한 그라디언트 흐름 관리
4. 실시간 성능 모니터링 및 적응형 라우팅
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import time
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from data_models import (
    EmotionData, EthicalSituation, DecisionScenario,
    IntegratedAnalysisResult, SystemStatus
)

logger = logging.getLogger('RedHeart.ModuleBridge')

class ModuleType(Enum):
    """모듈 타입 정의"""
    EMOTION = "emotion"
    BENTHAM = "bentham" 
    SEMANTIC = "semantic"
    SURD = "surd"
    REGRET = "regret"
    HIERARCHICAL_EMOTION = "hierarchical_emotion"
    BAYESIAN = "bayesian"
    LLM = "llm"
    COUNTERFACTUAL = "counterfactual"
    # 새로 추가된 모듈들
    LEGAL_EXPERT = "legal_expert"
    THREE_VIEW_SCENARIO = "three_view_scenario"
    PHASE_CONTROLLER = "phase_controller"

class DataFormat(Enum):
    """데이터 포맷 타입"""
    RAW_TEXT = "raw_text"
    EMOTION_VECTOR = "emotion_vector"
    BENTHAM_SCORES = "bentham_scores"
    SEMANTIC_EMBEDDINGS = "semantic_embeddings"
    SURD_GRAPH = "surd_graph"
    INTEGRATED_STATE = "integrated_state"
    # 새로 추가된 데이터 포맷들
    LEGAL_ANALYSIS = "legal_analysis"
    THREE_VIEW_SCENARIOS = "three_view_scenarios"
    PERFORMANCE_METRICS = "performance_metrics"

@dataclass
class ModuleInput:
    """표준화된 모듈 입력"""
    data: Any
    format_type: DataFormat
    source_module: Optional[ModuleType] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0
    
@dataclass 
class ModuleOutput:
    """표준화된 모듈 출력"""
    data: Any
    format_type: DataFormat
    source_module: ModuleType
    processing_time: float = 0.0
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    gradient_info: Optional[torch.Tensor] = None
    explanation: Dict[str, Any] = field(default_factory=dict)

class ModuleAdapter(ABC):
    """모듈 어댑터 추상 클래스"""
    
    def __init__(self, module_type: ModuleType, original_module: Any):
        self.module_type = module_type
        self.original_module = original_module
        self.is_trainable = True
        self.performance_stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'average_time': 0.0,
            'last_confidence': 0.0
        }
        
    @abstractmethod
    def standardize_input(self, input_data: Any, metadata: Dict[str, Any] = None) -> ModuleInput:
        """입력을 표준 형식으로 변환"""
        pass
        
    @abstractmethod 
    def standardize_output(self, output_data: Any, processing_time: float = 0.0) -> ModuleOutput:
        """출력을 표준 형식으로 변환"""
        pass
        
    @abstractmethod
    async def process(self, input_data: ModuleInput) -> ModuleOutput:
        """표준화된 처리 메소드"""
        pass
        
    def update_stats(self, processing_time: float, success: bool, confidence: float):
        """성능 통계 업데이트"""
        self.performance_stats['total_calls'] += 1
        if success:
            self.performance_stats['successful_calls'] += 1
            
        # 이동 평균으로 평균 시간 계산
        alpha = 0.1
        self.performance_stats['average_time'] = (
            alpha * processing_time + 
            (1 - alpha) * self.performance_stats['average_time']
        )
        self.performance_stats['last_confidence'] = confidence

class EmotionModuleAdapter(ModuleAdapter):
    """감정 분석 모듈 어댑터"""
    
    def __init__(self, emotion_analyzer):
        super().__init__(ModuleType.EMOTION, emotion_analyzer)
        
    def standardize_input(self, input_data: Any, metadata: Dict[str, Any] = None) -> ModuleInput:
        if isinstance(input_data, str):
            return ModuleInput(
                data=input_data,
                format_type=DataFormat.RAW_TEXT,
                metadata=metadata or {},
                source_module=None
            )
        else:
            return ModuleInput(
                data=input_data,
                format_type=DataFormat.EMOTION_VECTOR,
                metadata=metadata or {},
                source_module=None
            )
            
    def standardize_output(self, output_data: Any, processing_time: float = 0.0) -> ModuleOutput:
        # ⭐ EmotionData 객체를 표준 형식으로 변환
        return ModuleOutput(
            data=output_data,  # EmotionData 객체 자체를 전달
            format_type=DataFormat.EMOTION_VECTOR,
            source_module=ModuleType.EMOTION,
            processing_time=processing_time,
            confidence=getattr(output_data, 'confidence', 0.8),
            explanation={
                'emotion_breakdown': {
                    'valence': getattr(output_data, 'valence', 0.0),
                    'arousal': getattr(output_data, 'arousal', 0.0),
                    'dominance': getattr(output_data, 'dominance', 0.0)
                },
                'dominant_emotion': getattr(output_data, 'dominant_emotion', 'unknown'),
                'intensity': getattr(output_data, 'intensity', 0.0),
                'emotions': getattr(output_data, 'emotions', {})
            }
        )
        
    async def process(self, input_data: ModuleInput) -> ModuleOutput:
        start_time = time.time()
        try:
            # ⭐ 동기 메소드를 비동기로 래핑
            import asyncio
            
            def sync_analyze():
                # 실제 analyze_emotion 메소드 호출 (동기식)
                return self.original_module.analyze_emotion(
                    text=input_data.data,
                    language=input_data.metadata.get('language', 'ko'),
                    use_cache=True
                )
            
            # 동기 메소드를 스레드풀에서 실행
            result = await asyncio.get_event_loop().run_in_executor(None, sync_analyze)
                
            processing_time = time.time() - start_time
            output = self.standardize_output(result, processing_time)
            
            self.update_stats(processing_time, True, output.confidence)
            return output
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.update_stats(processing_time, False, 0.0)
            logger.error(f"감정 분석 처리 실패: {e}")
            raise

class BenthamModuleAdapter(ModuleAdapter):
    """벤담 계산 모듈 어댑터"""
    
    def __init__(self, bentham_calculator):
        super().__init__(ModuleType.BENTHAM, bentham_calculator)
        
    def standardize_input(self, input_data: Any, metadata: Dict[str, Any] = None) -> ModuleInput:
        return ModuleInput(
            data=input_data,
            format_type=DataFormat.RAW_TEXT if isinstance(input_data, str) else DataFormat.BENTHAM_SCORES,
            metadata=metadata or {},
            source_module=None
        )
        
    def standardize_output(self, output_data: Any, processing_time: float = 0.0) -> ModuleOutput:
        # ⭐ EnhancedHedonicResult 객체를 표준 형식으로 변환
        return ModuleOutput(
            data=output_data,  # EnhancedHedonicResult 객체 자체를 전달
            format_type=DataFormat.BENTHAM_SCORES,
            source_module=ModuleType.BENTHAM,
            processing_time=processing_time,
            confidence=getattr(output_data, 'confidence', 0.8),
            explanation={
                'utilitarian_score': getattr(output_data, 'final_score', 0.0),
                'base_score': getattr(output_data, 'base_score', 0.0),
                'layer_contributions': getattr(output_data, 'layer_results', []),
                'hedonic_values': getattr(output_data, 'hedonic_values', {}),
                'calculation_metadata': getattr(output_data, 'metadata', {})
            }
        )
        
    async def process(self, input_data: ModuleInput) -> ModuleOutput:
        start_time = time.time()
        try:
            import asyncio
            
            def sync_calculate():
                # ⭐ 벤담 계산을 위한 입력 데이터 준비
                if isinstance(input_data.data, str):
                    # 텍스트인 경우 기본 입력 데이터 구조 생성
                    bentham_input = {
                        'input_values': {
                            'intensity': 0.7,
                            'duration': 0.6,
                            'certainty': 0.8,
                            'propinquity': 0.9,
                            'fecundity': 0.5,
                            'purity': 0.7,
                            'extent': 0.8
                        },
                        'text_description': input_data.data,
                        'language': input_data.metadata.get('language', 'ko')
                    }
                else:
                    bentham_input = input_data.data
                    
                # 실제 calculate_with_advanced_layers 메소드 호출
                return self.original_module.calculate_with_advanced_layers(
                    input_data=bentham_input,
                    use_cache=True
                )
            
            result = await asyncio.get_event_loop().run_in_executor(None, sync_calculate)
                
            processing_time = time.time() - start_time
            output = self.standardize_output(result, processing_time)
            
            self.update_stats(processing_time, True, output.confidence)
            return output
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.update_stats(processing_time, False, 0.0)
            logger.error(f"벤담 계산 처리 실패: {e}")
            raise

class SemanticModuleAdapter(ModuleAdapter):
    """의미 분석 모듈 어댑터"""
    
    def __init__(self, semantic_analyzer):
        super().__init__(ModuleType.SEMANTIC, semantic_analyzer)
        
    def standardize_input(self, input_data: Any, metadata: Dict[str, Any] = None) -> ModuleInput:
        return ModuleInput(
            data=input_data,
            format_type=DataFormat.RAW_TEXT if isinstance(input_data, str) else DataFormat.SEMANTIC_EMBEDDINGS,
            metadata=metadata or {},
            source_module=None
        )
        
    def standardize_output(self, output_data: Any, processing_time: float = 0.0) -> ModuleOutput:
        # ⭐ AdvancedSemanticResult 객체를 표준 형식으로 변환
        return ModuleOutput(
            data=output_data,  # AdvancedSemanticResult 객체 자체를 전달
            format_type=DataFormat.SEMANTIC_EMBEDDINGS,
            source_module=ModuleType.SEMANTIC,
            processing_time=processing_time,
            confidence=getattr(output_data, 'confidence', 0.8),
            explanation={
                'semantic_levels': getattr(output_data, 'semantic_levels', {}),
                'feature_vector': getattr(output_data, 'feature_vector', None),
                'clustering_result': getattr(output_data, 'clustering_result', {}),
                'network_analysis': getattr(output_data, 'network_analysis', {}),
                'analysis_depth': getattr(output_data, 'analysis_depth', 'unknown')
            }
        )
        
    async def process(self, input_data: ModuleInput) -> ModuleOutput:
        start_time = time.time()
        try:
            import asyncio
            
            def sync_analyze():
                # 실제 analyze_text_advanced 메소드 호출
                return self.original_module.analyze_text_advanced(
                    text=input_data.data,
                    language=input_data.metadata.get('language', 'ko'),
                    analysis_depth=input_data.metadata.get('analysis_depth', 'full'),
                    use_cache=True
                )
            
            result = await asyncio.get_event_loop().run_in_executor(None, sync_analyze)
                
            processing_time = time.time() - start_time
            output = self.standardize_output(result, processing_time)
            
            self.update_stats(processing_time, True, output.confidence)
            return output
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.update_stats(processing_time, False, 0.0)
            logger.error(f"의미 분석 처리 실패: {e}")
            raise

class SURDModuleAdapter(ModuleAdapter):
    """SURD 분석 모듈 어댑터"""
    
    def __init__(self, surd_analyzer):
        super().__init__(ModuleType.SURD, surd_analyzer)
        
    def standardize_input(self, input_data: Any, metadata: Dict[str, Any] = None) -> ModuleInput:
        return ModuleInput(
            data=input_data,
            format_type=DataFormat.RAW_TEXT if isinstance(input_data, str) else DataFormat.SURD_GRAPH,
            metadata=metadata or {},
            source_module=None
        )
        
    def standardize_output(self, output_data: Any, processing_time: float = 0.0) -> ModuleOutput:
        # ⭐ AdvancedSURDResult 객체를 표준 형식으로 변환
        return ModuleOutput(
            data=output_data,  # AdvancedSURDResult 객체 자체를 전달
            format_type=DataFormat.SURD_GRAPH,
            source_module=ModuleType.SURD,
            processing_time=processing_time,
            confidence=getattr(output_data, 'confidence', 0.8),
            explanation={
                'causal_graph': getattr(output_data, 'causal_graph', {}),
                'information_decomposition': getattr(output_data, 'information_decomposition', {}),
                'synergy_scores': getattr(output_data, 'synergy_scores', {}),
                'redundancy_scores': getattr(output_data, 'redundancy_scores', {}),
                'unique_information': getattr(output_data, 'unique_information', {}),
                'llm_interpretation': getattr(output_data, 'llm_interpretation', {}),
                'surd_summary': getattr(output_data, 'summary', {})
            }
        )
        
    async def process(self, input_data: ModuleInput) -> ModuleOutput:
        start_time = time.time()
        try:
            import asyncio
            
            def sync_analyze():
                # ⭐ SURD 분석을 위한 변수 준비
                if isinstance(input_data.data, str):
                    # 텍스트인 경우 기본 변수 생성
                    variables = {
                        'emotion_intensity': 0.7,
                        'ethical_weight': 0.8,
                        'social_impact': 0.6,
                        'time_pressure': 0.4,
                        'uncertainty': 0.5,
                        'decision_outcome': 0.65
                    }
                    target_variable = 'decision_outcome'
                    additional_context = input_data.metadata or {}
                else:
                    # 딕셔너리인 경우 직접 사용
                    variables = input_data.data.get('variables', {})
                    target_variable = input_data.data.get('target_variable', 'decision_outcome')
                    additional_context = input_data.data.get('additional_context', {})
                
                # 실제 analyze_advanced 메소드 호출
                return self.original_module.analyze_advanced(
                    variables=variables,
                    target_variable=target_variable,
                    time_series_data=None,
                    additional_context=additional_context
                )
            
            result = await asyncio.get_event_loop().run_in_executor(None, sync_analyze)
                
            processing_time = time.time() - start_time
            output = self.standardize_output(result, processing_time)
            
            self.update_stats(processing_time, True, output.confidence)
            return output
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.update_stats(processing_time, False, 0.0)
            logger.error(f"SURD 분석 처리 실패: {e}")
            raise

class ModuleBridgeCoordinator:
    """모듈 간 통합 학습을 위한 중재자 시스템"""
    
    def __init__(self):
        self.adapters: Dict[ModuleType, ModuleAdapter] = {}
        self.data_flow_graph = {}
        self.training_mode = False
        self.thread_pool = ThreadPoolExecutor(max_workers=8)
        self.gradient_accumulator = {}
        self.performance_monitor = {}
        
        # 통합 학습을 위한 신경망 레이어
        self.integration_network = nn.ModuleDict({
            'emotion_transform': nn.Linear(512, 256),
            'bentham_transform': nn.Linear(256, 256), 
            'semantic_transform': nn.Linear(768, 256),
            'surd_transform': nn.Linear(512, 256),
            # 새로 추가된 모듈들을 위한 변환 레이어
            'legal_expert_transform': nn.Linear(256, 256),
            'three_view_scenario_transform': nn.Linear(512, 256),
            'phase_controller_transform': nn.Linear(128, 256),
            'fusion_layer': nn.Linear(256 * 7, 512),  # 7개 모듈로 확장
            'output_layer': nn.Linear(512, 256)
        })
        
        logger.info("ModuleBridgeCoordinator 초기화 완료")
        
    def register_module(self, module_type: ModuleType, original_module: Any) -> ModuleAdapter:
        """모듈 등록 및 어댑터 생성"""
        adapter_classes = {
            ModuleType.EMOTION: EmotionModuleAdapter,
            ModuleType.BENTHAM: BenthamModuleAdapter,
            ModuleType.SEMANTIC: SemanticModuleAdapter,
            ModuleType.SURD: SURDModuleAdapter,
            # 새로 추가된 모듈들
            ModuleType.LEGAL_EXPERT: LegalExpertAdapter,
            ModuleType.THREE_VIEW_SCENARIO: ThreeViewScenarioAdapter,
            ModuleType.PHASE_CONTROLLER: PhaseControllerAdapter,
        }
        
        if module_type in adapter_classes:
            adapter = adapter_classes[module_type](original_module)
            self.adapters[module_type] = adapter
            logger.info(f"모듈 등록 완료: {module_type.value}")
            return adapter
        else:
            raise ValueError(f"지원되지 않는 모듈 타입: {module_type}")
            
    async def integrated_analysis(self, input_text: str, 
                                 enable_modules: List[ModuleType] = None) -> Dict[str, ModuleOutput]:
        """통합 분석 실행"""
        if enable_modules is None:
            enable_modules = list(self.adapters.keys())
            
        results = {}
        
        # 병렬로 모든 모듈 실행
        tasks = []
        for module_type in enable_modules:
            if module_type in self.adapters:
                adapter = self.adapters[module_type]
                input_data = adapter.standardize_input(input_text)
                tasks.append(self._process_module(adapter, input_data, module_type))
                
        # 결과 수집
        outputs = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, module_type in enumerate(enable_modules):
            if module_type in self.adapters:
                if isinstance(outputs[i], Exception):
                    logger.error(f"모듈 {module_type.value} 처리 실패: {outputs[i]}")
                    results[module_type.value] = None
                else:
                    results[module_type.value] = outputs[i]
                    
        return results
        
    async def _process_module(self, adapter: ModuleAdapter, 
                            input_data: ModuleInput, 
                            module_type: ModuleType) -> ModuleOutput:
        """개별 모듈 처리"""
        try:
            return await adapter.process(input_data)
        except Exception as e:
            logger.error(f"모듈 {module_type.value} 처리 중 오류: {e}")
            raise
            
    def enable_integrated_training(self):
        """통합 학습 모드 활성화"""
        self.training_mode = True
        
        # 모든 어댑터를 훈련 모드로 설정
        for adapter in self.adapters.values():
            if hasattr(adapter.original_module, 'train'):
                adapter.original_module.train()
                
        # 통합 네트워크도 훈련 모드
        self.integration_network.train()
        
        logger.info("통합 학습 모드 활성화")
        
    def get_performance_report(self) -> Dict[str, Any]:
        """성능 리포트 생성"""
        report = {}
        
        for module_type, adapter in self.adapters.items():
            report[module_type.value] = {
                'stats': adapter.performance_stats.copy(),
                'success_rate': (
                    adapter.performance_stats['successful_calls'] / 
                    max(adapter.performance_stats['total_calls'], 1)
                ),
                'is_healthy': adapter.performance_stats['average_time'] < 5.0
            }
            
        return report
        
    async def optimize_data_flow(self, sample_inputs: List[str]) -> Dict[str, Any]:
        """데이터 흐름 최적화"""
        optimization_results = {}
        
        # 각 모듈의 처리 시간 측정
        for input_text in sample_inputs[:5]:  # 샘플만 테스트
            results = await self.integrated_analysis(input_text)
            
            for module_name, result in results.items():
                if result:
                    if module_name not in optimization_results:
                        optimization_results[module_name] = []
                    optimization_results[module_name].append(result.processing_time)
                    
        # 최적화 추천사항 생성
        recommendations = {}
        for module_name, times in optimization_results.items():
            avg_time = np.mean(times)
            recommendations[module_name] = {
                'average_time': avg_time,
                'recommendation': 'optimize' if avg_time > 2.0 else 'maintain'
            }
            
        return recommendations


# ===============================
# 새로 추가된 모듈들을 위한 어댑터
# ===============================

class LegalExpertAdapter(ModuleAdapter):
    """법률 전문가 시스템 어댑터"""
    
    def __init__(self, legal_expert_system):
        super().__init__(ModuleType.LEGAL_EXPERT, legal_expert_system)
        
    def standardize_input(self, input_data: Any, metadata: Dict[str, Any] = None) -> ModuleInput:
        """법률 전문가 시스템 입력 표준화"""
        # 텍스트와 도메인 정보 추출
        if isinstance(input_data, dict):
            text = input_data.get('text', '')
            domain = input_data.get('domain', 'personal')
            context_data = input_data.get('context_data', {})
        else:
            text = str(input_data)
            domain = 'personal'
            context_data = {}
            
        legal_input = {
            'text': text,
            'domain': domain,
            'context_data': context_data
        }
        
        return ModuleInput(
            data=legal_input,
            format_type=DataFormat.RAW_TEXT,
            source_module=None,
            metadata=metadata or {},
            confidence=1.0
        )
    
    def standardize_output(self, output_data: Any, processing_time: float = 0.0) -> ModuleOutput:
        """법률 전문가 시스템 출력 표준화"""
        return ModuleOutput(
            data=output_data,
            format_type=DataFormat.LEGAL_ANALYSIS,
            source_module=ModuleType.LEGAL_EXPERT,
            processing_time=processing_time,
            confidence=output_data.confidence if hasattr(output_data, 'confidence') else 0.8,
            metadata={
                'risk_level': output_data.risk_level.value if hasattr(output_data, 'risk_level') else 'unknown',
                'domain': output_data.domain if hasattr(output_data, 'domain') else 'unknown'
            },
            explanation={
                'identified_risks': output_data.identified_risks if hasattr(output_data, 'identified_risks') else [],
                'recommendations': output_data.recommendations if hasattr(output_data, 'recommendations') else []
            }
        )
    
    async def process(self, module_input: ModuleInput) -> ModuleOutput:
        """법률 전문가 시스템 처리"""
        start_time = time.time()
        
        try:
            # 도메인 매핑
            from legal_expert_system import LegalDomain
            domain_mapping = {
                'education': LegalDomain.EDUCATION,
                'business': LegalDomain.BUSINESS,
                'social': LegalDomain.SOCIAL,
                'politics': LegalDomain.POLITICS,
                'life': LegalDomain.LIFE,
                'personal': LegalDomain.LIFE  # 기본값
            }
            
            legal_domain = domain_mapping.get(module_input.data.get('domain', 'personal'), LegalDomain.LIFE)
            
            # 법률 전문가 시스템 호출
            result = self.original_module.analyze_legal_context(
                domain=legal_domain,
                text=module_input.data.get('text', ''),
                context_data=module_input.data.get('context_data', {})
            )
            
            processing_time = time.time() - start_time
            
            # 성능 통계 업데이트
            self.performance_stats['total_calls'] += 1
            if result:
                self.performance_stats['successful_calls'] += 1
                self.performance_stats['last_confidence'] = result.confidence
            
            self.performance_stats['average_time'] = (
                (self.performance_stats['average_time'] * (self.performance_stats['total_calls'] - 1) + processing_time) 
                / self.performance_stats['total_calls']
            )
            
            return self.standardize_output(result, processing_time)
            
        except Exception as e:
            logger.error(f"법률 전문가 시스템 처리 오류: {e}")
            # 오류 발생 시 기본값 반환
            processing_time = time.time() - start_time
            return ModuleOutput(
                data=None,
                format_type=DataFormat.LEGAL_ANALYSIS,
                source_module=ModuleType.LEGAL_EXPERT,
                processing_time=processing_time,
                confidence=0.0,
                metadata={'error': str(e)}
            )


class ThreeViewScenarioAdapter(ModuleAdapter):
    """3뷰 시나리오 시스템 어댑터"""
    
    def __init__(self, three_view_system):
        super().__init__(ModuleType.THREE_VIEW_SCENARIO, three_view_system)
        
    def standardize_input(self, input_data: Any, metadata: Dict[str, Any] = None) -> ModuleInput:
        """3뷰 시나리오 시스템 입력 표준화"""
        # 다양한 입력 형태 처리
        if isinstance(input_data, dict):
            scenario_input = input_data
        else:
            scenario_input = {
                'text': str(input_data),
                'description': str(input_data)
            }
        
        return ModuleInput(
            data=scenario_input,
            format_type=DataFormat.RAW_TEXT,
            source_module=None,
            metadata=metadata or {},
            confidence=1.0
        )
    
    def standardize_output(self, output_data: Any, processing_time: float = 0.0) -> ModuleOutput:
        """3뷰 시나리오 시스템 출력 표준화"""
        return ModuleOutput(
            data=output_data,
            format_type=DataFormat.THREE_VIEW_SCENARIOS,
            source_module=ModuleType.THREE_VIEW_SCENARIO,
            processing_time=processing_time,
            confidence=output_data.consensus_strength if hasattr(output_data, 'consensus_strength') else 0.7,
            metadata={
                'consensus_utility': output_data.consensus_utility if hasattr(output_data, 'consensus_utility') else 0.0,
                'consensus_regret': output_data.consensus_regret if hasattr(output_data, 'consensus_regret') else 0.0,
                'scenario_diversity': output_data.scenario_diversity if hasattr(output_data, 'scenario_diversity') else 0.0
            },
            explanation={
                'recommended_decision': output_data.recommended_decision if hasattr(output_data, 'recommended_decision') else '',
                'risk_mitigation': output_data.risk_mitigation_strategies if hasattr(output_data, 'risk_mitigation_strategies') else [],
                'opportunity_enhancement': output_data.opportunity_enhancement_strategies if hasattr(output_data, 'opportunity_enhancement_strategies') else []
            }
        )
    
    async def process(self, module_input: ModuleInput) -> ModuleOutput:
        """3뷰 시나리오 시스템 처리"""
        start_time = time.time()
        
        try:
            # 3뷰 시나리오 분석 실행
            result = await self.original_module.analyze_three_view_scenarios(module_input.data)
            
            processing_time = time.time() - start_time
            
            # 성능 통계 업데이트
            self.performance_stats['total_calls'] += 1
            if result:
                self.performance_stats['successful_calls'] += 1
                self.performance_stats['last_confidence'] = result.consensus_strength
            
            self.performance_stats['average_time'] = (
                (self.performance_stats['average_time'] * (self.performance_stats['total_calls'] - 1) + processing_time) 
                / self.performance_stats['total_calls']
            )
            
            return self.standardize_output(result, processing_time)
            
        except Exception as e:
            logger.error(f"3뷰 시나리오 시스템 처리 오류: {e}")
            # 오류 발생 시 기본값 반환
            processing_time = time.time() - start_time
            return ModuleOutput(
                data=None,
                format_type=DataFormat.THREE_VIEW_SCENARIOS,
                source_module=ModuleType.THREE_VIEW_SCENARIO,
                processing_time=processing_time,
                confidence=0.0,
                metadata={'error': str(e)}
            )


class PhaseControllerAdapter(ModuleAdapter):
    """페이즈 컨트롤러 훅 어댑터"""
    
    def __init__(self, phase_controller):
        super().__init__(ModuleType.PHASE_CONTROLLER, phase_controller)
        
    def standardize_input(self, input_data: Any, metadata: Dict[str, Any] = None) -> ModuleInput:
        """페이즈 컨트롤러 입력 표준화"""
        # 성능 메트릭 데이터 처리
        if isinstance(input_data, dict):
            metrics_input = input_data
        else:
            metrics_input = {
                'phase_type': 'inference',
                'metrics': {'accuracy': 0.5, 'loss': 0.5}
            }
        
        return ModuleInput(
            data=metrics_input,
            format_type=DataFormat.PERFORMANCE_METRICS,
            source_module=None,
            metadata=metadata or {},
            confidence=1.0
        )
    
    def standardize_output(self, output_data: Any, processing_time: float = 0.0) -> ModuleOutput:
        """페이즈 컨트롤러 출력 표준화"""
        return ModuleOutput(
            data=output_data,
            format_type=DataFormat.PERFORMANCE_METRICS,
            source_module=ModuleType.PHASE_CONTROLLER,
            processing_time=processing_time,
            confidence=0.9,  # 성능 메트릭은 일반적으로 높은 신뢰도
            metadata={
                'total_snapshots': output_data.get('summary', {}).get('total_snapshots', 0) if output_data else 0,
                'executed_actions': output_data.get('executed_actions', 0) if output_data else 0,
                'pending_actions': output_data.get('pending_actions', 0) if output_data else 0
            },
            explanation={
                'performance_trends': output_data.get('performance_trends', {}) if output_data else {},
                'error_patterns': output_data.get('error_patterns', {}) if output_data else {}
            }
        )
    
    async def process(self, module_input: ModuleInput) -> ModuleOutput:
        """페이즈 컨트롤러 처리"""
        start_time = time.time()
        
        try:
            # 페이즈 컨트롤러에서 성능 보고서 생성
            from phase_controller_hook import PhaseType
            
            # 성능 메트릭 기록
            if 'metrics' in module_input.data:
                from phase_controller_hook import PerformanceMetric
                
                phase_type = PhaseType.INFERENCE
                if module_input.data.get('phase_type') == 'training':
                    phase_type = PhaseType.TRAINING
                elif module_input.data.get('phase_type') == 'validation':
                    phase_type = PhaseType.VALIDATION
                
                self.original_module.record_performance(
                    phase_type=phase_type,
                    metrics=module_input.data['metrics'],
                    context=module_input.metadata
                )
            
            # 성능 보고서 생성
            result = self.original_module.get_performance_report()
            
            processing_time = time.time() - start_time
            
            # 성능 통계 업데이트
            self.performance_stats['total_calls'] += 1
            self.performance_stats['successful_calls'] += 1
            self.performance_stats['last_confidence'] = 0.9
            
            self.performance_stats['average_time'] = (
                (self.performance_stats['average_time'] * (self.performance_stats['total_calls'] - 1) + processing_time) 
                / self.performance_stats['total_calls']
            )
            
            return self.standardize_output(result, processing_time)
            
        except Exception as e:
            logger.error(f"페이즈 컨트롤러 처리 오류: {e}")
            # 오류 발생 시 기본값 반환
            processing_time = time.time() - start_time
            return ModuleOutput(
                data=None,
                format_type=DataFormat.PERFORMANCE_METRICS,
                source_module=ModuleType.PHASE_CONTROLLER,
                processing_time=processing_time,
                confidence=0.0,
                metadata={'error': str(e)}
            )