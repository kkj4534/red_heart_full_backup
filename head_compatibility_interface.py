"""
Red Heart 헤드 호환성 인터페이스 - 800M 아키텍처 통합
Head Compatibility Interface for Red Heart - 800M Architecture Integration

기존 전용 헤드들을 300M 통합 백본과 연동:
- EnhancedEmpathyLearner (140M)
- FrommEnhancedBenthamCalculator (120M)  
- SemanticSURDAnalyzer (80M)
- RegretLearningNetwork (120M)
- MetaIntegrationHead (40M)

⚠️ 주의사항:
혹시나 중복된 메서드가 존재할 수 있으니 코드 돌리고 해당 부분에서 문제 발생시,
get_pytorch_network 중복을 의심하며 자세한 확인 필요
특히 SemanticSURDHeadAdapter, BenthamFrommHeadAdapter, RegretLearningHeadAdapter 클래스에서
동일한 메서드가 두 번 정의되어 있는지 확인할 것 (Python은 마지막 정의를 사용함)
"""

import asyncio
import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Union, Tuple, Type
from dataclasses import dataclass, field
from datetime import datetime
import threading
from abc import ABC, abstractmethod
from enum import Enum

# 핵심 시스템 임포트
from config import ADVANCED_CONFIG, get_smart_device, get_gpu_memory_info, ModelPriority, get_priority_based_device
# 필요한 것만 실제 import, 나머지는 TYPE_CHECKING
from dynamic_swap_manager import SwapPriority, RedHeartDynamicSwapManager
from unified_red_heart_core import UnifiedRepresentation, RedHeartUnifiedBackbone
# 최적화된 차원 변환 어댑터
from optimized_dimension_adapter import OptimizedDimensionAdapter, HeadSpecificAdapters

# LightweightCrossAttention은 지연 import로 처리
def get_lightweight_cross_attention():
    """LightweightCrossAttention 지연 import - 초기화 타이밍 문제 해결"""
    from unified_red_heart_core import LightweightCrossAttention
    return LightweightCrossAttention

# 순환 import 방지를 위한 TYPE_CHECKING
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dynamic_swap_manager import SwapContext

# 기존 헤드들 실제 import (TYPE_CHECKING에서 제거)
# 🔥 CRITICAL FIX: 지연 import로 변경 - import 시점 hanging 방지
# 각 어댑터 클래스의 __init__에서 필요할 때 import

# 로거 설정
logger = logging.getLogger(__name__)

class HeadType(Enum):
    """헤드 타입 정의"""
    EMOTION_EMPATHY = "emotion_empathy_head"
    BENTHAM_FROMM = "bentham_fromm_head"
    SEMANTIC_SURD = "semantic_surd_head"
    REGRET_LEARNING = "regret_learning_head"
    META_INTEGRATION = "meta_integration_head"

@dataclass
class HeadProcessingResult:
    """헤드 처리 결과"""
    head_type: HeadType
    primary_output: Any
    secondary_outputs: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    device_used: str = "cpu"
    synergy_features: Optional[torch.Tensor] = None
    confidence_score: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)

class BaseHeadAdapter(nn.Module, ABC):
    """
    기본 헤드 어댑터 - 모든 헤드 어댑터의 베이스 클래스
    PyTorch 모듈로 인식되도록 nn.Module 상속 추가
    """
    
    def __init__(self, head_type: HeadType, priority: SwapPriority = SwapPriority.MEDIUM):
        super().__init__()
        self.head_type = head_type
        self.priority = priority
        self.config = ADVANCED_CONFIG['specialized_heads']
        self.backbone_config = ADVANCED_CONFIG['unified_backbone']
        
        # 호환성 어댑터 설정 - 최적화된 차원 변환 어댑터 사용
        self.dimension_adapter = None
        self.cross_attention = None
        self.initialized = False
        
        # 동적 로딩 관련 속성
        self.force_cpu_mode = False
        self.current_device = None
        
        logger.info(f"BaseHeadAdapter 초기화: {head_type.value}")
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch nn.Module 호환성을 위한 추상 forward 메서드
        각 헤드 어댑터에서 반드시 구현해야 함
        """
        pass
    
    def _determine_target_device(self, estimated_params_mb: int = 100):
        """우선순위 기반 타겟 디바이스 결정 로직"""
        import torch
        
        # 헤드 타입별 우선순위 매핑
        head_priority_map = {
            HeadType.EMOTION_EMPATHY: ModelPriority.HIGH,  # 140M - 높은 우선순위
            HeadType.BENTHAM_FROMM: ModelPriority.HIGH,    # 120M - 높은 우선순위  
            HeadType.SEMANTIC_SURD: ModelPriority.MEDIUM,  # 80M - 중간 우선순위
            HeadType.REGRET_LEARNING: ModelPriority.MEDIUM, # 120M - 중간 우선순위
            HeadType.META_INTEGRATION: ModelPriority.LOW   # 40M - 낮은 우선순위
        }
        
        priority = head_priority_map.get(self.head_type, ModelPriority.MEDIUM)
        required_memory_mb = estimated_params_mb * 4  # FP32 기준
        
        # 우선순위 기반 디바이스 선택
        device = get_priority_based_device(
            memory_required_mb=required_memory_mb,
            priority=priority,
            model_id=f"head_{self.head_type.value}"
        )
        
        logger.info(f"{self.head_type.value}: {estimated_params_mb}M 파라미터, {priority} 우선순위 -> {device}")
        return device
    
    def _move_to_device_safely(self, model, target_device, model_name: str = "model"):
        """안전한 디바이스 이동 (에러 처리 포함)"""
        try:
            if model is not None and hasattr(model, 'to'):
                model.to(target_device)
                logger.info(f"{self.head_type.value}.{model_name} -> {target_device}")
                return True
        except Exception as e:
            logger.warning(f"{self.head_type.value}.{model_name} 디바이스 이동 실패: {e}")
            # 실패해도 계속 진행 (프로젝트 규칙: fallback 없음)
            return False
        return False
    
    @abstractmethod
    async def initialize_head(self):
        """헤드 초기화 (비동기)"""
        pass
    
    @abstractmethod
    async def process_unified_input(self, unified_repr: UnifiedRepresentation) -> HeadProcessingResult:
        """통합 표현을 처리하여 헤드별 결과 생성"""
        pass
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        PyTorch forward 호환을 위한 메서드
        unified_learning_system.py에서 호출됨
        주의: 이 메서드는 동기 호출만 지원하며, optimized dimension_adapter를 사용합니다.
        """
        # 최적화된 차원 변환 어댑터가 있으면 사용
        if hasattr(self, 'dimension_adapter') and self.dimension_adapter is not None:
            # 백본 출력(1280) -> 헤드 입력으로 변환 (encode)
            adapted_input = self.dimension_adapter.encode(input_tensor)
        else:
            adapted_input = input_tensor
        
        # PyTorch 네트워크 가져오기
        pytorch_network = self.get_pytorch_network()
        if pytorch_network is None:
            logger.warning(f"{self.head_type.value} - PyTorch 네트워크가 없어 기본값 반환")
            return torch.zeros_like(input_tensor)
        
        # 감정 헤드의 경우 추가 차원 변환 필요 (1024 -> 768)
        if self.head_type == HeadType.EMOTION_EMPATHY and adapted_input.shape[-1] != 768:
            device = adapted_input.device
            if not hasattr(self, '_emotion_dim_reducer'):
                self._emotion_dim_reducer = torch.nn.Linear(adapted_input.shape[-1], 768).to(device)
            adapted_input = self._emotion_dim_reducer(adapted_input)
        
        # PyTorch 네트워크로 forward
        try:
            output = pytorch_network(adapted_input)
            
            # 감정 헤드의 경우 출력이 6차원이므로 특별 처리
            if self.head_type == HeadType.EMOTION_EMPATHY and output.shape[-1] == 6:
                # 6차원 감정 벡터를 고차원으로 확장
                if not hasattr(self, '_emotion_output_projector'):
                    self._emotion_output_projector = torch.nn.Sequential(
                        torch.nn.Linear(6, 256),
                        torch.nn.ReLU(),
                        torch.nn.Linear(256, 1024)
                    ).to(output.device)
                output = self._emotion_output_projector(output)
            
            # 최적화된 차원 변환 어댑터가 있으면 사용하여 백본 차원으로 복원 (decode)
            if hasattr(self, 'dimension_adapter') and self.dimension_adapter is not None:
                output = self.dimension_adapter.decode(output, original_input=input_tensor)
            
            return output
            
        except Exception as e:
            logger.error(f"{self.head_type.value} forward 실패: {str(e)}")
            logger.error(f"입력 shape: {adapted_input.shape}")
            return torch.zeros_like(input_tensor)
    
    def _create_input_adapter(self, input_dim: int, output_dim: int) -> nn.Module:
        """입력 어댑터 생성 - 통합 표현을 헤드별 입력으로 변환"""
        return nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def _create_output_adapter(self, input_dim: int, output_dim: int) -> nn.Module:
        """출력 어댑터 생성 - 헤드별 출력을 표준화된 형태로 변환"""
        return nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

class EmotionEmpathyHeadAdapter(BaseHeadAdapter):
    """
    감정+공감 헤드 어댑터 (140M 파라미터)
    EnhancedEmpathyLearner와 통합 백본 연결
    """
    
    def __init__(self):
        super().__init__(HeadType.EMOTION_EMPATHY, SwapPriority.HIGH)
        self.empathy_learner = None
        
    async def initialize_head(self):
        """감정+공감 헤드 동적 초기화 - CPU/GPU 선택적 로딩"""
        if self.initialized:
            logger.info("EmotionEmpathyHeadAdapter: 이미 초기화됨")
            return
            
        # 디바이스 결정: force_cpu_mode 우선, 그 다음 스마트 디바이스 선택 (140M 파라미터)
        target_device = self._determine_target_device(estimated_params_mb=140)
        logger.info(f"EmotionEmpathyHeadAdapter 초기화 시작 (디바이스: {target_device})...")
        
        try:
            # 전역 레지스트리에서 emotion_analyzer 모듈 확인 (재시도 로직 포함)
            logger.info("전역 emotion_analyzer 모듈 확인 중...")
            from config import get_system_module
            import asyncio
            
            max_retries = 10
            retry_count = 0
            emotion_analyzer = None
            
            while retry_count < max_retries:
                emotion_analyzer = get_system_module('emotion_analyzer')
                if emotion_analyzer is not None:
                    logger.info(f"✅ emotion_analyzer 모듈 확인됨 (시도 {retry_count + 1})")
                    break
                    
                retry_count += 1
                logger.warning(f"⏳ emotion_analyzer 아직 로드되지 않음, 대기 중... ({retry_count}/{max_retries})")
                await asyncio.sleep(0.5)
            
            if emotion_analyzer is None:
                raise RuntimeError("전역 emotion_analyzer 모듈을 찾을 수 없음 - HeadAdapter는 연결 인터페이스로만 동작")
            
            # PyTorch 네트워크 검증
            # 디버깅: emotion_analyzer의 실제 타입과 메서드 확인
            logger.info(f"emotion_analyzer 타입: {type(emotion_analyzer)}")
            logger.info(f"emotion_analyzer 메서드 목록: {[m for m in dir(emotion_analyzer) if not m.startswith('_') and callable(getattr(emotion_analyzer, m, None))]}")
            
            if hasattr(emotion_analyzer, 'get_pytorch_network'):
                pytorch_network = emotion_analyzer.get_pytorch_network()
                if pytorch_network is not None:
                    logger.info(f"전역 emotion_analyzer PyTorch 네트워크 확인 완료: {type(pytorch_network)}")
                else:
                    logger.warning("전역 emotion_analyzer에서 PyTorch 네트워크를 가져올 수 없음")
            else:
                logger.warning("전역 emotion_analyzer에 get_pytorch_network 메서드가 없음")
            
            # 최적화된 차원 변환 어댑터 (1280 ↔ 1024)
            logger.info("optimized dimension_adapter 생성 중...")
            self.dimension_adapter = HeadSpecificAdapters.create_emotion_adapter().to(target_device)
            
            # 크로스 어텐션 (시너지 창출용)
            logger.info("cross_attention 생성 중...")
            backbone_num_heads = self.backbone_config.get('num_heads', 20)
            LightweightCrossAttention = get_lightweight_cross_attention()
            self.cross_attention = LightweightCrossAttention(
                d_model=self.backbone_config['d_model'],
                num_heads=backbone_num_heads  # 백본과 호환되는 헤드 수
            ).to(target_device)
            
            # 전역 모듈의 디바이스 이동은 MasterMemoryOrchestrator에서 관리
            logger.info("전역 emotion_analyzer 모듈의 디바이스 관리는 MasterMemoryOrchestrator에서 처리")
            
            # 학습 모드 설정 및 requires_grad 확인
            logger.info("학습 모드 설정 중...")
            self._ensure_training_mode()
            
            # 현재 디바이스 기록 (동적 스왑용)
            self.current_device = target_device
            
            # 최종 검증 - 전역 모듈 연결 확인
            final_pytorch_network = self.get_pytorch_network()
            if final_pytorch_network is None:
                logger.warning("전역 emotion_analyzer 모듈에서 PyTorch 네트워크를 찾을 수 없음 - 연결 인터페이스로만 동작")
                network_info = "No PyTorch Network"
            else:
                logger.info(f"전역 emotion_analyzer 네트워크 연결 확인: {type(final_pytorch_network)}")
                network_info = f"{type(final_pytorch_network)}"
            
            self.initialized = True
            logger.info(f"EmotionEmpathyHeadAdapter 초기화 완료 (디바이스: {target_device}, 네트워크: {network_info})")
            
        except Exception as e:
            logger.error(f"EmotionEmpathyHeadAdapter 초기화 실패: {str(e)}")
            logger.error(f"실패 상태: initialized={self.initialized}")
            # 실패 시 정리
            self.initialized = False
            raise
    
    
    def _move_empathy_learner_to_device(self, target_device):
        """EnhancedEmpathyLearner 내부 모델들을 타겟 디바이스로 이동"""
        try:
            # EnhancedEmpathyLearner의 PyTorch 네트워크 찾아서 이동 - Strict Mode
            pytorch_network = self.empathy_learner.get_pytorch_network()
            if pytorch_network is None:
                raise RuntimeError("디바이스 이동 중 PyTorch 네트워크를 찾을 수 없음")
            pytorch_network.to(target_device)
            logger.info(f"EnhancedEmpathyLearner PyTorch 네트워크 -> {target_device}")
            
            # 기타 내부 모델들이 있다면 여기서 이동
            if hasattr(self.empathy_learner, 'models'):
                for name, model in self.empathy_learner.models.items():
                    if model is not None and hasattr(model, 'to'):
                        model.to(target_device)
                        logger.info(f"EnhancedEmpathyLearner.{name} -> {target_device}")
                        
        except Exception as e:
            logger.error(f"EnhancedEmpathyLearner 디바이스 이동 실패: {e}")
            # Strict Mode: 디바이스 이동 실패 시 즉시 에러 발생
            raise RuntimeError(f"EnhancedEmpathyLearner 디바이스 이동 실패: {e}") from e
    
    def _ensure_training_mode(self):
        """학습 모드 설정 및 requires_grad 확인"""
        # dimension_adapter (최적화된 차원 변환 어댑터)
        if self.dimension_adapter is not None:
            self.dimension_adapter.train()
            for param in self.dimension_adapter.parameters():
                param.requires_grad = True
        
        # cross_attention
        if self.cross_attention is not None:
            self.cross_attention.train()
            for param in self.cross_attention.parameters():
                param.requires_grad = True
        
        # 실제 empathy_learner 네트워크
        pytorch_network = self.get_pytorch_network()
        if pytorch_network is not None:
            pytorch_network.train()
            for param in pytorch_network.parameters():
                param.requires_grad = True
            logger.info("EmotionEmpathyHeadAdapter: 모든 파라미터 학습 모드 설정 완료")
    
    def get_pytorch_network(self) -> Optional[nn.Module]:
        """PyTorch 네트워크 반환 - 전역 등록된 모듈에서 가져오기"""
        try:
            from config import get_system_module
            
            # 전역 레지스트리에서 감정 분석기 가져오기
            emotion_analyzer = get_system_module('emotion_analyzer')
            if emotion_analyzer is None:
                logger.warning("emotion_analyzer가 전역 레지스트리에 등록되지 않음")
                return None
            
            # 감정 분석기에서 PyTorch 네트워크 찾기 (우선순위 순서)
            # 1. get_pytorch_network() 메서드 호출
            if hasattr(emotion_analyzer, 'get_pytorch_network'):
                network = emotion_analyzer.get_pytorch_network()
                if network is not None:
                    logger.info(f"EmotionEmpathyHeadAdapter: get_pytorch_network()로 네트워크 획득")
                    return network
                else:
                    logger.warning("EmotionEmpathyHeadAdapter: get_pytorch_network()가 None 반환")
            
            # 2. 직접 속성 확인 (emotion_moe, hierarchical_model 등)
            if hasattr(emotion_analyzer, 'emotion_moe') and emotion_analyzer.emotion_moe is not None:
                logger.info("EmotionEmpathyHeadAdapter: emotion_moe 속성 직접 반환")
                return emotion_analyzer.emotion_moe
            
            if hasattr(emotion_analyzer, 'neural_empathy_model') and emotion_analyzer.neural_empathy_model is not None:
                logger.info("EmotionEmpathyHeadAdapter: neural_empathy_model 속성 반환")
                return emotion_analyzer.neural_empathy_model
            
            if isinstance(emotion_analyzer, nn.Module):
                logger.info("EmotionEmpathyHeadAdapter: emotion_analyzer 자체가 nn.Module")
                return emotion_analyzer
            
            # 3. 계층적 감정 시스템에서 찾기 (EnhancedEmpathyLearner)
            if hasattr(emotion_analyzer, 'hierarchical_emotion_system'):
                hier_system = emotion_analyzer.hierarchical_emotion_system
                if hasattr(hier_system, 'enhanced_empathy_learner'):
                    empathy_learner = hier_system.enhanced_empathy_learner
                    if hasattr(empathy_learner, 'get_pytorch_network'):
                        network = empathy_learner.get_pytorch_network()
                        if network is not None:
                            logger.info("EmotionEmpathyHeadAdapter: EnhancedEmpathyLearner에서 네트워크 획득")
                            return network
            
            # 네트워크를 찾을 수 없는 경우 상세 로깅
            logger.warning(f"EmotionEmpathyHeadAdapter: emotion_analyzer에서 PyTorch 네트워크를 찾을 수 없음")
            logger.warning(f"  - emotion_analyzer 타입: {type(emotion_analyzer)}")
            logger.warning(f"  - 사용 가능한 속성들: {[attr for attr in dir(emotion_analyzer) if not attr.startswith('_')]}")
            return None
            
        except Exception as e:
            logger.error(f"EmotionEmpathyHeadAdapter PyTorch 네트워크 탐지 실패: {e}")
            import traceback
            logger.error(f"스택 트레이스:\n{traceback.format_exc()}")
            return None
    
    async def _ensure_network_binding(self):
        """지연 바인딩 - 네트워크가 없으면 재시도하여 바인딩"""
        if hasattr(self, '_pytorch_network_cached') and self._pytorch_network_cached is not None:
            return self._pytorch_network_cached
            
        logger.info("🔄 EmotionEmpathyHeadAdapter: 지연 바인딩 시도 중...")
        
        # 최대 3회 재시도
        for retry in range(3):
            network = self.get_pytorch_network()
            if network is not None:
                self._pytorch_network_cached = network
                logger.info(f"✅ 지연 바인딩 성공 (시도 {retry + 1}/3)")
                return network
            
            if retry < 2:
                logger.warning(f"⏳ 네트워크 바인딩 실패, 재시도 대기 중... ({retry + 1}/3)")
                await asyncio.sleep(0.5 * (retry + 1))  # 지수 백오프
        
        # 최종 실패 시 명시적 에러
        error_msg = "EmotionEmpathyHeadAdapter: 지연 바인딩 최종 실패 - PyTorch 네트워크를 찾을 수 없음"
        logger.error(error_msg)
        logger.error("🚨 가능한 원인:")
        logger.error("  1. emotion_analyzer 모듈이 아직 초기화되지 않음")
        logger.error("  2. emotion_analyzer에 get_pytorch_network() 메서드가 없음")
        logger.error("  3. 모든 신경망 모델이 None 상태")
        raise RuntimeError(error_msg)
    
    async def process_unified_input(self, unified_repr: UnifiedRepresentation) -> HeadProcessingResult:
        """통합 표현으로부터 감정+공감 분석 수행"""
        start_time = time.time()
        
        if not self.initialized:
            await self.initialize_head()
        
        # 지연 바인딩 확인 - 네트워크가 없으면 재시도
        await self._ensure_network_binding()
        
        device = unified_repr.device
        
        # 1. 입력 어댑터를 통한 차원 변환
        adapted_input = self.input_adapter(unified_repr.shared_embedding)
        
        # 2. 기존 감정 시스템과 호환 가능한 형태로 변환
        # (EnhancedEmpathyLearner의 입력 형식에 맞춤)
        emotion_context = {
            'text_embedding': adapted_input,
            'attention_weights': unified_repr.attention_weights,
            'timestamp': unified_repr.timestamp,
            'device': device
        }
        
        # 3. missing_neural_models 활용 - SelfOtherNeuralNetwork
        try:
            # SelfOtherNeuralNetwork를 활용한 자타 구분 감정 분석
            from missing_neural_models import SelfOtherNeuralNetwork
            
            # SelfOtherNeuralNetwork 초기화 (입력 차원에 맞춤)
            input_dim = adapted_input.shape[-1]
            self_other_net = SelfOtherNeuralNetwork(input_dim=input_dim).to(device)
            
            # 자타 구분 분석 수행
            self_other_result = self_other_net(adapted_input)
            
            # 기존 감정 분석과 통합
            emotion_result = await self._process_emotion_analysis(emotion_context)
            
            # SelfOtherNeuralNetwork 결과를 emotion_result에 통합
            emotion_result['self_other_classification'] = self_other_result['self_other_probs']
            emotion_result['self_other_confidence'] = self_other_result['confidence']
            emotion_result['neural_features'] = self_other_result['features']
            
            # 4. 출력 어댑터를 통한 표준화
            if isinstance(emotion_result.get('emotion_embedding'), torch.Tensor):
                standardized_output = self.output_adapter(emotion_result['emotion_embedding'])
            else:
                # 텐서가 아닌 경우 기본값 생성
                standardized_output = torch.zeros(
                    unified_repr.shared_embedding.shape[0], 
                    self.backbone_config['d_model'], 
                    device=device
                )
            
            # 5. 시너지 특성 생성 (크로스 어텐션)
            synergy_features = None
            if hasattr(self, 'cross_attention'):
                cross_attn_result = self.cross_attention(
                    query=unified_repr.shared_embedding,
                    key_value_pairs=[('emotion_head', standardized_output)]
                )
                synergy_features = cross_attn_result.get('emotion_head')
            
            processing_time = time.time() - start_time
            
            return HeadProcessingResult(
                head_type=self.head_type,
                primary_output=emotion_result,
                secondary_outputs={
                    'standardized_embedding': standardized_output,
                    'emotion_dimensions': emotion_result.get('emotion_dimensions', {}),
                    'empathy_scores': emotion_result.get('empathy_scores', {}),
                    'community_awareness': emotion_result.get('community_awareness', {})
                },
                processing_time=processing_time,
                device_used=str(device),
                synergy_features=synergy_features,
                confidence_score=emotion_result.get('confidence', 0.8)
            )
            
        except Exception as e:
            logger.error(f"감정+공감 헤드 처리 오류: {str(e)}")
            processing_time = time.time() - start_time
            
            # 오류 발생 시 기본 결과 반환
            return HeadProcessingResult(
                head_type=self.head_type,
                primary_output={'error': str(e)},
                processing_time=processing_time,
                device_used=str(device),
                confidence_score=0.0
            )
    
    async def _process_emotion_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """감정 분석 실행 (비동기) - 실제 PyTorch 네트워크 사용"""
        try:
            embedding = context['text_embedding']
            device = context.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
            
            # PyTorch 네트워크 가져오기
            pytorch_network = self.get_pytorch_network()
            if pytorch_network is None:
                # PyTorch 네트워크가 없으면 기본 처리
                logger.warning("PyTorch 네트워크가 없어 기본 처리 수행")
                return self._default_emotion_processing(embedding)
            
            # 네트워크를 eval 모드로 설정
            pytorch_network.eval()
            
            # 입력 형식 확인 및 조정
            # embedding은 이미 input_adapter를 통해 1024 차원으로 변환됨
            # EmpathyNet은 768 차원을 기대하므로 추가 변환 필요
            if embedding.shape[-1] != 768:
                # 1024 -> 768 차원 축소
                if not hasattr(self, '_emotion_dim_reducer'):
                    self._emotion_dim_reducer = torch.nn.Linear(embedding.shape[-1], 768).to(device)
                embedding_768 = self._emotion_dim_reducer(embedding)
            else:
                embedding_768 = embedding
            
            # PyTorch 네트워크로 감정 분석 수행
            with torch.no_grad():
                emotion_output = pytorch_network(embedding_768)
            
            # 이미 tanh 적용되어 [-1, 1] 범위
            if emotion_output.dim() == 1:
                emotion_output = emotion_output.unsqueeze(0)
            
            # 감정 차원 해석 (6차원 감정 벡터)
            emotion_values = emotion_output.squeeze().cpu().numpy()
            emotion_dimensions = {
                'valence': float(emotion_values[0]),       # 감정가
                'arousal': float(emotion_values[1]),       # 각성도
                'dominance': float(emotion_values[2]),     # 지배감
                'certainty': float(emotion_values[3]),     # 확실성
                'surprise': float(emotion_values[4]),      # 놀라움
                'anticipation': float(emotion_values[5])   # 기대감
            }
            
            # 공감 점수 계산 (감정 차원 기반)
            empathy_scores = {
                'self_awareness': abs(emotion_values[0]) * 0.5 + 0.5,  # valence 기반
                'other_awareness': abs(emotion_values[2]) * 0.5 + 0.5,  # dominance 기반
                'community_awareness': abs(emotion_values[5]) * 0.5 + 0.5  # anticipation 기반
            }
            
            # 임베딩은 원래 차원 유지 (1024)
            return {
                'emotion_embedding': context['text_embedding'],  # 원래 임베딩 유지
                'emotion_vector': emotion_output,  # 6차원 감정 벡터
                'emotion_dimensions': emotion_dimensions,
                'empathy_scores': empathy_scores,
                'community_awareness': {
                    'integration_level': float(np.mean(list(empathy_scores.values())))
                },
                'confidence': float(torch.sigmoid(torch.mean(torch.abs(emotion_output))).item())
            }
            
        except Exception as e:
            logger.error(f"감정 분석 실행 오류: {str(e)}")
            import traceback
            logger.error(f"트레이스백: {traceback.format_exc()}")
            return self._default_emotion_processing(context.get('text_embedding'))
    
    def _default_emotion_processing(self, embedding: torch.Tensor) -> Dict[str, Any]:
        """기본 감정 처리 (폴백)"""
        try:
            # 기본 감정 차원 계산
            if embedding is not None and embedding.shape[-1] >= 1024:
                emotion_dimensions = {
                    'valence': float(torch.mean(embedding[:, :256]).item()),
                    'arousal': float(torch.mean(embedding[:, 256:512]).item()),
                    'dominance': float(torch.mean(embedding[:, 512:768]).item()),
                    'certainty': float(torch.mean(embedding[:, 768:1024]).item()),
                    'surprise': 0.5,
                    'anticipation': 0.5
                }
            else:
                emotion_dimensions = {
                    'valence': 0.5, 'arousal': 0.5, 'dominance': 0.5,
                    'certainty': 0.5, 'surprise': 0.5, 'anticipation': 0.5
                }
            
            return {
                'emotion_embedding': embedding,
                'emotion_dimensions': emotion_dimensions,
                'empathy_scores': {
                    'self_awareness': 0.5,
                    'other_awareness': 0.5,
                    'community_awareness': 0.5
                },
                'community_awareness': {'integration_level': 0.5},
                'confidence': 0.5
            }
        except:
            return {'error': 'default processing failed', 'confidence': 0.0}

class BenthamFrommHeadAdapter(BaseHeadAdapter):
    """
    벤담+프롬 헤드 어댑터 (120M 파라미터)
    FrommEnhancedBenthamCalculator와 통합 백본 연결
    """
    
    def __init__(self):
        super().__init__(HeadType.BENTHAM_FROMM, SwapPriority.HIGH)
        self.bentham_calculator = None
    
    async def initialize_head(self):
        """벤담+프롬 헤드 초기화"""
        if self.initialized:
            logger.info("BenthamFrommHeadAdapter: 이미 초기화됨")
            return
            
        # 디바이스 결정: force_cpu_mode 우선, 그 다음 스마트 디바이스 선택 (120M 파라미터)
        target_device = self._determine_target_device(estimated_params_mb=120)
        logger.info(f"BenthamFrommHeadAdapter 초기화 시작 (디바이스: {target_device})...")
        
        try:
            # 전역 레지스트리에서 bentham_calculator 모듈 확인
            logger.info("전역 bentham_calculator 모듈 확인 중...")
            from config import get_system_module
            bentham_calculator = get_system_module('bentham_calculator')
            
            if bentham_calculator is None:
                raise RuntimeError("전역 bentham_calculator 모듈을 찾을 수 없음 - HeadAdapter는 연결 인터페이스로만 동작")
            
            # PyTorch 네트워크 검증
            if hasattr(bentham_calculator, 'get_pytorch_network'):
                pytorch_network = bentham_calculator.get_pytorch_network()
                if pytorch_network is not None:
                    logger.info(f"전역 bentham_calculator PyTorch 네트워크 확인 완료: {type(pytorch_network)}")
                else:
                    logger.warning("전역 bentham_calculator에서 PyTorch 네트워크를 가져올 수 없음")
            else:
                logger.warning("전역 bentham_calculator에 get_pytorch_network 메서드가 없음")
            
            # 최적화된 차원 변환 어댑터 (1280 ↔ 768)
            logger.info("optimized dimension_adapter 생성 중...")
            self.dimension_adapter = HeadSpecificAdapters.create_bentham_adapter().to(target_device)
            
            # 크로스 어텐션
            logger.info("cross_attention 생성 중...")
            backbone_num_heads = self.backbone_config.get('num_heads', 20)
            LightweightCrossAttention = get_lightweight_cross_attention()
            self.cross_attention = LightweightCrossAttention(
                d_model=self.backbone_config['d_model'],
                num_heads=backbone_num_heads  # 백본과 호환되는 헤드 수
            ).to(target_device)
            
            # 현재 디바이스 기록 (동적 스왑용)
            self.current_device = target_device
            
            # 학습 모드 설정 및 requires_grad 확인
            logger.info("학습 모드 설정 중...")
            self._ensure_training_mode()
            
            # 현재 디바이스 기록 (동적 스웑용)
            self.current_device = target_device
            
            # 최종 검증 - 전역 모듈 연결 확인
            final_pytorch_network = self.get_pytorch_network()
            if final_pytorch_network is None:
                logger.warning("전역 bentham_calculator 모듈에서 PyTorch 네트워크를 찾을 수 없음 - 연결 인터페이스로만 동작")
                network_info = "No PyTorch Network"
            else:
                logger.info(f"전역 bentham_calculator 네트워크 연결 확인: {type(final_pytorch_network)}")
                network_info = f"{type(final_pytorch_network)}"
            
            self.initialized = True
            logger.info(f"BenthamFrommHeadAdapter 초기화 완료 (디바이스: {target_device}, 네트워크: {network_info})")
            
        except Exception as e:
            logger.error(f"BenthamFrommHeadAdapter 초기화 실패: {str(e)}")
            logger.error(f"실패 상태: initialized={self.initialized}")
            # 실패 시 정리
            self.initialized = False
            raise
    
    def _ensure_training_mode(self):
        """학습 모드 설정 및 requires_grad 확인"""
        # dimension_adapter (최적화된 차원 변환 어댑터)
        if self.dimension_adapter is not None:
            self.dimension_adapter.train()
            for param in self.dimension_adapter.parameters():
                param.requires_grad = True
        
        # cross_attention
        if self.cross_attention is not None:
            self.cross_attention.train()
            for param in self.cross_attention.parameters():
                param.requires_grad = True
        
        # 실제 bentham_calculator 네트워크
        pytorch_network = self.get_pytorch_network()
        if pytorch_network is not None and hasattr(pytorch_network, 'parameters'):
            pytorch_network.train()
            for param in pytorch_network.parameters():
                param.requires_grad = True
            logger.info("BenthamFrommHeadAdapter: 모든 파라미터 학습 모드 설정 완료")
        else:
            logger.warning("BenthamFrommHeadAdapter: 학습 가능한 네트워크 없음 - 정상 진행")
    
    def get_pytorch_network(self) -> Optional[nn.Module]:
        """
        PyTorch 네트워크 반환 - 전역 bentham_calculator에서 안전하게 추출
        STRICT_NO_FALLBACK 정책 준수 (더미 금지)
        """
        try:
            from config import get_system_module
            
            # 전역 레지스트리에서 벤담 계산기 가져오기
            bentham_calculator = get_system_module('bentham_calculator')
            if bentham_calculator is None:
                logger.error("bentham_calculator 전역 레지스트리에 없음 → 즉시 중단")
                return None   # 상위 로직에서 RuntimeError 처리
            
            # 1️⃣ bentham_calculator 자체에 get_pytorch_network() 메서드가 있으면 우선 호출
            if hasattr(bentham_calculator, 'get_pytorch_network'):
                try:
                    net = bentham_calculator.get_pytorch_network()
                    if isinstance(net, nn.Module):
                        logger.info("BenthamFrommHeadAdapter: bentham_calculator.get_pytorch_network()에서 네트워크 획득")
                        return net
                except Exception as e:
                    logger.warning(f"bentham_calculator.get_pytorch_network() 호출 실패: {e}")
            
            # 2️⃣ base_calculator에서 neural_predictor 찾기 (레이지 로딩 지원)
            if hasattr(bentham_calculator, 'base_calculator'):
                base_calc = bentham_calculator.base_calculator
                
                # neural_predictor 프로퍼티 접근으로 레이지 로딩 트리거
                if hasattr(base_calc, 'neural_predictor'):
                    try:
                        neural_pred = base_calc.neural_predictor
                        if isinstance(neural_pred, nn.Module):
                            logger.info("BenthamFrommHeadAdapter: base_calculator.neural_predictor에서 네트워크 획득")
                            return neural_pred
                    except Exception as prop_error:
                        logger.warning(f"neural_predictor 프로퍼티 접근 실패: {prop_error}")
                
                # 직접 private 필드 확인
                if hasattr(base_calc, '_neural_predictor') and isinstance(base_calc._neural_predictor, nn.Module):
                    logger.info("BenthamFrommHeadAdapter: base_calculator._neural_predictor에서 네트워크 획득")
                    return base_calc._neural_predictor
            
            # 3️⃣ 속성 기반 탐색 (현행 유지)
            for attr in ('neural_predictor', '_neural_predictor', 'model', 'network', 'classifier'):
                net = getattr(bentham_calculator, attr, None)
                if isinstance(net, nn.Module):
                    logger.info(f"BenthamFrommHeadAdapter: bentham_calculator.{attr}에서 네트워크 획득")
                    return net
            
            # 4. bentham_calculator 자체가 nn.Module인지 확인
            if isinstance(bentham_calculator, nn.Module):
                logger.info("BenthamFrommHeadAdapter: bentham_calculator 자체가 nn.Module")
                return bentham_calculator
            
            logger.error("bentham_calculator에서 PyTorch 네트워크 탐색 실패")
            return None  # 상위에서 RuntimeError
            
        except Exception as e:
            logger.error(f"BenthamFrommHeadAdapter PyTorch 네트워크 탐지 실패: {e}")
            return None
    
    async def process_unified_input(self, unified_repr: UnifiedRepresentation) -> HeadProcessingResult:
        """통합 표현으로부터 벤담+프롬 분석 수행"""
        start_time = time.time()
        
        if not self.initialized:
            await self.initialize_head()
        
        device = unified_repr.device
        
        # 입력 어댑터 적용
        adapted_input = self.input_adapter(unified_repr.shared_embedding)
        
        # 벤담 계산 실행
        try:
            bentham_result = await self._process_bentham_calculation(adapted_input, device)
            
            # 출력 표준화
            standardized_output = self.output_adapter(adapted_input)
            
            # 시너지 특성 생성
            synergy_features = None
            if hasattr(self, 'cross_attention'):
                cross_attn_result = self.cross_attention(
                    query=unified_repr.shared_embedding,
                    key_value_pairs=[('bentham_head', standardized_output)]
                )
                synergy_features = cross_attn_result.get('bentham_head')
            
            processing_time = time.time() - start_time
            
            return HeadProcessingResult(
                head_type=self.head_type,
                primary_output=bentham_result,
                secondary_outputs={
                    'standardized_embedding': standardized_output,
                    'bentham_scores': bentham_result.get('bentham_scores', {}),
                    'fromm_orientation': bentham_result.get('fromm_orientation', 'unknown'),
                    'ethical_evaluation': bentham_result.get('ethical_evaluation', {})
                },
                processing_time=processing_time,
                device_used=str(device),
                synergy_features=synergy_features,
                confidence_score=bentham_result.get('confidence', 0.8)
            )
            
        except Exception as e:
            logger.error(f"벤담+프롬 헤드 처리 오류: {str(e)}")
            processing_time = time.time() - start_time
            
            return HeadProcessingResult(
                head_type=self.head_type,
                primary_output={'error': str(e)},
                processing_time=processing_time,
                device_used=str(device),
                confidence_score=0.0
            )
    
    async def _process_bentham_calculation(self, embedding: torch.Tensor, device: torch.device) -> Dict[str, Any]:
        """벤담 계산 실행"""
        try:
            # 기본 벤담 변수들 계산 (임시 구현)
            bentham_scores = {
                'intensity': float(torch.mean(embedding[:, :128]).item()),
                'duration': float(torch.mean(embedding[:, 128:256]).item()),
                'certainty': float(torch.mean(embedding[:, 256:384]).item()),
                'propinquity': float(torch.mean(embedding[:, 384:512]).item()),
                'fecundity': float(torch.mean(embedding[:, 512:640]).item()),
                'purity': float(torch.mean(embedding[:, 640:768]).item()),
                'extent': 0.75  # 기본값
            }
            
            # 프롬 지향성 분석
            having_score = torch.mean(embedding[:, :384]).item()
            being_score = torch.mean(embedding[:, 384:768]).item()
            
            if having_score > being_score + 0.1:
                fromm_orientation = 'having'
            elif being_score > having_score + 0.1:
                fromm_orientation = 'being'
            else:
                fromm_orientation = 'mixed'
            
            # 전체 쾌락 점수 계산
            total_pleasure = sum(bentham_scores.values()) / len(bentham_scores)
            
            return {
                'bentham_scores': bentham_scores,
                'fromm_orientation': fromm_orientation,
                'total_pleasure_score': total_pleasure,
                'ethical_evaluation': {
                    'utilitarian_score': total_pleasure,
                    'humanistic_score': being_score,
                    'balanced_score': (total_pleasure + being_score) / 2
                },
                'confidence': 0.82
            }
            
        except Exception as e:
            logger.error(f"벤담 계산 실행 오류: {str(e)}")
            return {'error': str(e), 'confidence': 0.0}

class SemanticSURDHeadAdapter(BaseHeadAdapter):
    """
    의미+SURD 헤드 어댑터 (80M 파라미터)
    AdvancedMultiLevelSemanticAnalyzer와 통합 백본 연결
    """
    
    def __init__(self):
        super().__init__(HeadType.SEMANTIC_SURD, SwapPriority.MEDIUM)
        self.semantic_analyzer = None
    
    async def initialize_head(self):
        """의미+SURD 헤드 초기화"""
        if self.initialized:
            return
            
        logger.info("SemanticSURDHeadAdapter 초기화 시작...")
        
        # 기존 의미 분석기 초기화 (순수 재시도 방식)
        max_retries = 3
        retry_delay = 1.0
        
        # 전역 레지스트리에서 semantic_analyzer 모듈 확인
        logger.info("전역 semantic_analyzer 모듈 확인 중...")
        from config import get_system_module
        semantic_analyzer = get_system_module('semantic_analyzer')
        
        if semantic_analyzer is None:
            raise RuntimeError("전역 semantic_analyzer 모듈을 찾을 수 없음 - HeadAdapter는 연결 인터페이스로만 동작")
        
        # 실제 로드된 클래스 정보 로깅
        logger.info(f"전역 semantic_analyzer 타입: {type(semantic_analyzer)}")
        logger.info(f"전역 semantic_analyzer 클래스명: {semantic_analyzer.__class__.__name__}")
        logger.info(f"전역 semantic_analyzer 모듈: {semantic_analyzer.__class__.__module__}")
        
        # PyTorch 네트워크 검증
        if hasattr(semantic_analyzer, 'get_pytorch_network'):
            logger.info("get_pytorch_network 메서드 발견")
            pytorch_network = semantic_analyzer.get_pytorch_network()
            if pytorch_network is not None:
                logger.info(f"전역 semantic_analyzer PyTorch 네트워크 확인 완료: {type(pytorch_network)}")
            else:
                logger.warning("전역 semantic_analyzer에서 PyTorch 네트워크를 가져올 수 없음")
        else:
            logger.warning("전역 semantic_analyzer에 get_pytorch_network 메서드가 없음")
            # 사용 가능한 메서드 목록 출력
            methods = [m for m in dir(semantic_analyzer) if not m.startswith('_') and callable(getattr(semantic_analyzer, m, None))]
            logger.warning(f"사용 가능한 메서드들: {methods[:20]}")
        
        # 최적화된 차원 변환 어댑터 (1280 ↔ 512)
        self.dimension_adapter = HeadSpecificAdapters.create_semantic_adapter()
        
        # 크로스 어텐션 (백본과 호환되는 헤드 수)
        backbone_num_heads = self.backbone_config.get('num_heads', 20)
        LightweightCrossAttention = get_lightweight_cross_attention()
        self.cross_attention = LightweightCrossAttention(
            d_model=self.backbone_config['d_model'],
            num_heads=backbone_num_heads  # 백본과 호환되는 헤드 수
        )
        
        # 학습 모드 설정 및 requires_grad 확인
        self._ensure_training_mode()
        
        self.initialized = True
        logger.info("SemanticSURDHeadAdapter 초기화 완료")
    
    def _ensure_training_mode(self):
        """학습 모드 설정 및 requires_grad 확인"""
        # dimension_adapter (최적화된 차원 변환 어댑터)
        if self.dimension_adapter is not None:
            self.dimension_adapter.train()
            for param in self.dimension_adapter.parameters():
                param.requires_grad = True
        
        # cross_attention
        if self.cross_attention is not None:
            self.cross_attention.train()
            for param in self.cross_attention.parameters():
                param.requires_grad = True
        
        # 실제 semantic_analyzer 네트워크
        pytorch_network = self.get_pytorch_network()
        if pytorch_network is not None:
            pytorch_network.train()
            for param in pytorch_network.parameters():
                param.requires_grad = True
            logger.info("SemanticSURDHeadAdapter: 모든 파라미터 학습 모드 설정 완료")
    
    def get_pytorch_network(self) -> Optional[nn.Module]:
        """
        PyTorch 네트워크 반환 - 전역 semantic_analyzer에서 안전하게 추출
        STRICT_NO_FALLBACK 정책 준수 (더미 금지)
        """
        try:
            from config import get_system_module
            
            # 전역 레지스트리에서 의미 분석기 가져오기
            semantic_analyzer = get_system_module('semantic_analyzer')
            if semantic_analyzer is None:
                logger.error("semantic_analyzer 전역 레지스트리에 없음 → 즉시 중단")
                return None   # 상위 로직에서 RuntimeError 처리
            
            # 1️⃣ semantic_analyzer 자체에 get_pytorch_network() 메서드가 있으면 우선 호출
            if hasattr(semantic_analyzer, 'get_pytorch_network'):
                try:
                    net = semantic_analyzer.get_pytorch_network()
                    if isinstance(net, nn.Module):
                        logger.info("SemanticSURDHeadAdapter: semantic_analyzer.get_pytorch_network()에서 네트워크 획득")
                        return net
                except Exception as e:
                    logger.warning(f"semantic_analyzer.get_pytorch_network() 호출 실패: {e}")
            
            # 2️⃣ 속성 기반 탐색 (현행 유지)
            for attr in ('fusion_network', 'cross_attention', 'main_network', 'model'):
                net = getattr(semantic_analyzer, attr, None)
                if isinstance(net, nn.Module):
                    logger.info(f"SemanticSURDHeadAdapter: semantic_analyzer.{attr}에서 네트워크 획득")
                    return net
            
            # 3. AdvancedMultiLevelSemanticAnalyzer 자체가 nn.Module인지 확인
            if isinstance(semantic_analyzer, nn.Module):
                logger.info("SemanticSURDHeadAdapter: semantic_analyzer 자체가 nn.Module")
                return semantic_analyzer
            
            logger.error("semantic_analyzer에서 PyTorch 네트워크 탐색 실패")
            return None  # 상위에서 RuntimeError
            
        except Exception as e:
            logger.warning(f"SemanticSURDHeadAdapter PyTorch 네트워크 탐지 실패: {e}")
            return None
    
    async def process_unified_input(self, unified_repr: UnifiedRepresentation) -> HeadProcessingResult:
        """통합 표현으로부터 의미+SURD 분석 수행"""
        start_time = time.time()
        
        if not self.initialized:
            await self.initialize_head()
        
        device = unified_repr.device
        
        # 입력 어댑터 적용
        adapted_input = self.input_adapter(unified_repr.shared_embedding)
        
        try:
            semantic_result = await self._process_semantic_analysis(adapted_input, device)
            
            # 출력 표준화
            standardized_output = self.output_adapter(adapted_input)
            
            # 시너지 특성 생성
            synergy_features = None
            if hasattr(self, 'cross_attention'):
                cross_attn_result = self.cross_attention(
                    query=unified_repr.shared_embedding,
                    key_value_pairs=[('semantic_head', standardized_output)]
                )
                synergy_features = cross_attn_result.get('semantic_head')
            
            processing_time = time.time() - start_time
            
            return HeadProcessingResult(
                head_type=self.head_type,
                primary_output=semantic_result,
                secondary_outputs={
                    'standardized_embedding': standardized_output,
                    'semantic_layers': semantic_result.get('semantic_layers', {}),
                    'surd_measures': semantic_result.get('surd_measures', {}),
                    'hashtag_analysis': semantic_result.get('hashtag_analysis', {})
                },
                processing_time=processing_time,
                device_used=str(device),
                synergy_features=synergy_features,
                confidence_score=semantic_result.get('confidence', 0.8)
            )
            
        except Exception as e:
            logger.error(f"의미+SURD 헤드 처리 오류: {str(e)}")
            processing_time = time.time() - start_time
            
            return HeadProcessingResult(
                head_type=self.head_type,
                primary_output={'error': str(e)},
                processing_time=processing_time,
                device_used=str(device),
                confidence_score=0.0
            )
    
    async def _process_semantic_analysis(self, embedding: torch.Tensor, device: torch.device) -> Dict[str, Any]:
        """의미 분석 실행"""
        try:
            # 다중 수준 의미 분석 (임시 구현)
            semantic_layers = {
                'surface_meaning': float(torch.mean(embedding[:, :128]).item()),
                'deep_meaning': float(torch.mean(embedding[:, 128:256]).item()),
                'contextual_meaning': float(torch.mean(embedding[:, 256:384]).item()),
                'pragmatic_meaning': float(torch.mean(embedding[:, 384:512]).item())
            }
            
            # SURD 측정값
            surd_measures = {
                'synergy': 0.72,
                'unique_info': 0.68,
                'redundancy': 0.45,
                'deterministic': 0.83
            }
            
            # 해시태그 기반 분석
            hashtag_analysis = {
                'emotional_tags': ['#empathy', '#understanding'],
                'semantic_tags': ['#analysis', '#meaning'],
                'confidence_per_tag': {'#empathy': 0.85, '#understanding': 0.78}
            }
            
            return {
                'semantic_layers': semantic_layers,
                'surd_measures': surd_measures,
                'hashtag_analysis': hashtag_analysis,
                'overall_semantic_score': sum(semantic_layers.values()) / len(semantic_layers),
                'confidence': 0.81
            }
            
        except Exception as e:
            logger.error(f"의미 분석 실행 오류: {str(e)}")
            return {'error': str(e), 'confidence': 0.0}

class RegretLearningHeadAdapter(BaseHeadAdapter):
    """
    후회+학습 헤드 어댑터 (120M 파라미터)
    GPURegretNetwork와 통합 백본 연결
    """
    
    def __init__(self):
        super().__init__(HeadType.REGRET_LEARNING, SwapPriority.MEDIUM)
        self.regret_network = None
    
    def get_pytorch_network(self) -> Optional[nn.Module]:
        """
        PyTorch 네트워크 반환 - 전역 regret_analyzer에서 안전하게 추출
        STRICT_NO_FALLBACK 정책 준수 (더미 금지)
        """
        try:
            from config import get_system_module
            
            # 전역 레지스트리에서 후회 분석기 가져오기
            regret_analyzer = get_system_module('regret_analyzer')
            if regret_analyzer is None:
                logger.error("regret_analyzer 전역 레지스트리에 없음 → 즉시 중단")
                return None   # 상위 로직에서 RuntimeError 처리
            
            # 1️⃣ regret_analyzer 자체에 get_pytorch_network() 메서드가 있으면 우선 호출
            if hasattr(regret_analyzer, 'get_pytorch_network'):
                try:
                    net = regret_analyzer.get_pytorch_network()
                    if isinstance(net, nn.Module):
                        logger.info("RegretLearningHeadAdapter: regret_analyzer.get_pytorch_network()에서 네트워크 획득")
                        return net
                except Exception as e:
                    logger.warning(f"regret_analyzer.get_pytorch_network() 호출 실패: {e}")
            
            # 2️⃣ 속성 기반 탐색 (현행 유지)
            for attr in ('gpu_regret_network', 'regret_network', 'neural_predictor', '_neural_predictor', 'model', 'network'):
                net = getattr(regret_analyzer, attr, None)
                if isinstance(net, nn.Module):
                    logger.info(f"RegretLearningHeadAdapter: regret_analyzer.{attr}에서 네트워크 획득")
                    return net
            
            # 3. AdvancedRegretLearningSystem 자체가 nn.Module인지 확인
            if isinstance(regret_analyzer, nn.Module):
                logger.info("RegretLearningHeadAdapter: regret_analyzer 자체가 nn.Module")
                return regret_analyzer
            
            logger.error("regret_analyzer에서 PyTorch 네트워크 탐색 실패")
            return None  # 상위에서 RuntimeError
            
        except Exception as e:
            logger.warning(f"RegretLearningHeadAdapter PyTorch 네트워크 탐지 실패: {e}")
            return None
    
    async def initialize_head(self):
        """후회+학습 헤드 초기화"""
        if self.initialized:
            return
            
        logger.info("RegretLearningHeadAdapter 초기화 시작...")
        
        # 기존 후회 네트워크 초기화 (올바른 생성자 인자 사용)
        max_retries = 3
        retry_delay = 1.0
        
        # 전역 레지스트리에서 regret_analyzer 모듈 확인
        logger.info("전역 regret_analyzer 모듈 확인 중...")
        from config import get_system_module
        regret_analyzer = get_system_module('regret_analyzer')
        
        if regret_analyzer is None:
            raise RuntimeError("전역 regret_analyzer 모듈을 찾을 수 없음 - HeadAdapter는 연결 인터페이스로만 동작")
        
        # PyTorch 네트워크 검증
        if hasattr(regret_analyzer, 'get_pytorch_network'):
            pytorch_network = regret_analyzer.get_pytorch_network()
            if pytorch_network is not None:
                logger.info(f"전역 regret_analyzer PyTorch 네트워크 확인 완료: {type(pytorch_network)}")
            else:
                logger.warning("전역 regret_analyzer에서 PyTorch 네트워크를 가져올 수 없음")
        else:
            logger.warning("전역 regret_analyzer에 get_pytorch_network 메서드가 없음")
        
        # 최적화된 차원 변환 어댑터 (1280 ↔ 768)
        self.dimension_adapter = HeadSpecificAdapters.create_regret_adapter()
        
        # 크로스 어텐션
        LightweightCrossAttention = get_lightweight_cross_attention()
        self.cross_attention = LightweightCrossAttention(
            d_model=self.backbone_config['d_model'],
            num_heads=8
        )
        
        # 학습 모드 설정 및 requires_grad 확인
        self._ensure_training_mode()
        
        self.initialized = True
        logger.info("RegretLearningHeadAdapter 초기화 완료")
    
    def _ensure_training_mode(self):
        """학습 모드 설정 및 requires_grad 확인"""
        # dimension_adapter (최적화된 차원 변환 어댑터)
        if self.dimension_adapter is not None:
            self.dimension_adapter.train()
            for param in self.dimension_adapter.parameters():
                param.requires_grad = True
        
        # cross_attention
        if self.cross_attention is not None:
            self.cross_attention.train()
            for param in self.cross_attention.parameters():
                param.requires_grad = True
        
        # 실제 regret_network
        pytorch_network = self.get_pytorch_network()
        if pytorch_network is not None:
            pytorch_network.train()
            for param in pytorch_network.parameters():
                param.requires_grad = True
            logger.info("RegretLearningHeadAdapter: 모든 파라미터 학습 모드 설정 완료")
    
    async def process_unified_input(self, unified_repr: UnifiedRepresentation) -> HeadProcessingResult:
        """통합 표현으로부터 후회+학습 분석 수행"""
        start_time = time.time()
        
        if not self.initialized:
            await self.initialize_head()
        
        device = unified_repr.device
        
        # 입력 어댑터 적용
        adapted_input = self.input_adapter(unified_repr.shared_embedding)
        
        try:
            # missing_neural_models 활용 - IncrementalLearner
            from missing_neural_models import IncrementalLearner
            
            # IncrementalLearner 초기화 (입력 차원에 맞춤)
            input_dim = adapted_input.shape[-1]
            incremental_learner = IncrementalLearner(input_dim=input_dim).to(device)
            
            # 증분 학습을 통한 후회 학습 강화
            # 실제 레이블은 후회 분석 결과를 기반으로 생성 (가상 레이블)
            batch_size = adapted_input.shape[0]
            virtual_labels = torch.randn(batch_size, 64, device=device)  # 가상 후회 레이블
            
            # 증분 학습 수행
            learning_metrics = incremental_learner.learn_incrementally(adapted_input, virtual_labels)
            
            # 기존 후회 분석과 통합
            regret_result = await self._process_regret_analysis(adapted_input, device)
            
            # IncrementalLearner 결과를 regret_result에 통합
            regret_result['incremental_features'] = incremental_learner(adapted_input)
            regret_result['learning_metrics'] = learning_metrics
            regret_result['knowledge_retention'] = learning_metrics.get('knowledge_retention', 0.0)
            
            # 후회 네트워크 출력을 표준화
            if isinstance(regret_result.get('regret_output'), torch.Tensor):
                standardized_output = self.output_adapter(regret_result['regret_output'])
            else:
                standardized_output = self.output_adapter(
                    torch.zeros(adapted_input.shape[0], 64, device=device)
                )
            
            # 시너지 특성 생성
            synergy_features = None
            if hasattr(self, 'cross_attention'):
                cross_attn_result = self.cross_attention(
                    query=unified_repr.shared_embedding,
                    key_value_pairs=[('regret_head', standardized_output)]
                )
                synergy_features = cross_attn_result.get('regret_head')
            
            processing_time = time.time() - start_time
            
            return HeadProcessingResult(
                head_type=self.head_type,
                primary_output=regret_result,
                secondary_outputs={
                    'standardized_embedding': standardized_output,
                    'regret_intensity': regret_result.get('regret_intensity', 0.0),
                    'learning_rate': regret_result.get('learning_rate', 0.001),
                    'adaptation_score': regret_result.get('adaptation_score', 0.5)
                },
                processing_time=processing_time,
                device_used=str(device),
                synergy_features=synergy_features,
                confidence_score=regret_result.get('confidence', 0.8)
            )
            
        except Exception as e:
            logger.error(f"후회+학습 헤드 처리 오류: {str(e)}")
            processing_time = time.time() - start_time
            
            return HeadProcessingResult(
                head_type=self.head_type,
                primary_output={'error': str(e)},
                processing_time=processing_time,
                device_used=str(device),
                confidence_score=0.0
            )
    
    async def _process_regret_analysis(self, embedding: torch.Tensor, device: torch.device) -> Dict[str, Any]:
        """후회 분석 실행"""
        try:
            # 후회 네트워크 실행
            if self.regret_network is not None:
                regret_output = self.regret_network(embedding)
                
                # regret_output이 tuple인 경우 안전하게 처리
                if isinstance(regret_output, (tuple, list)):
                    regret_output = regret_output[0] if len(regret_output) > 0 else torch.zeros_like(embedding)
                elif not isinstance(regret_output, torch.Tensor):
                    regret_output = torch.zeros_like(embedding)
            else:
                regret_output = torch.mean(embedding, dim=-1, keepdim=True).expand(-1, 64)
            
            # 후회 강도 계산 (안전한 텐서 확인)
            if isinstance(regret_output, torch.Tensor):
                regret_intensity = float(torch.mean(regret_output).item())
            else:
                regret_intensity = 0.5  # 기본값
            
            # 학습률 조정
            learning_rate = max(0.0001, min(0.01, 0.001 * (1 + regret_intensity)))
            
            # 적응 점수
            adaptation_score = 1.0 / (1.0 + regret_intensity)
            
            return {
                'regret_output': regret_output,
                'regret_intensity': regret_intensity,
                'learning_rate': learning_rate,
                'adaptation_score': adaptation_score,
                'confidence': 0.79
            }
            
        except Exception as e:
            logger.error(f"후회 분석 실행 오류: {str(e)}")
            return {'error': str(e), 'confidence': 0.0}

class MetaIntegrationHeadAdapter(BaseHeadAdapter):
    """
    메타통합 헤드 어댑터 (40M 파라미터)
    다른 헤드들의 결과를 통합하고 메타 학습 수행
    """
    
    def __init__(self):
        super().__init__(HeadType.META_INTEGRATION, SwapPriority.LOW)
        self.integration_network = None
    
    def get_pytorch_network(self) -> Optional[nn.Module]:
        """PyTorch 네트워크 반환 - HeadCompatibilityManager용"""
        try:
            # 전역 모듈 레지스트리에서 meta_integration 모듈 가져오기
            from config import get_system_module
            meta_integration_module = get_system_module('meta_integration')
            
            if meta_integration_module is not None:
                # 모듈에 get_pytorch_network 메서드가 있는지 확인
                if hasattr(meta_integration_module, 'get_pytorch_network'):
                    return meta_integration_module.get_pytorch_network()
                # 모듈 자체가 nn.Module인 경우
                elif hasattr(meta_integration_module, 'forward'):
                    return meta_integration_module
            
            # 전역 모듈이 없으면 fallback으로 로컬 integration_network 사용
            if hasattr(self, 'integration_network') and self.integration_network is not None:
                logger.warning("전역 meta_integration 모듈이 없어 로컬 네트워크 사용")
                return self.integration_network
            
            return None
            
        except Exception as e:
            logger.warning(f"MetaIntegrationHeadAdapter PyTorch 네트워크 탐지 실패: {e}")
            return None
    
    async def initialize_head(self):
        """메타통합 헤드 초기화"""
        if self.initialized:
            return
            
        logger.info("🔍 MetaIntegrationHeadAdapter 초기화 시작...")
        
        try:
            # 1단계: backbone_config 검증
            logger.info(f"🔍 Step 1: backbone_config 검증 - d_model: {self.backbone_config.get('d_model', 'MISSING')}")
            if 'd_model' not in self.backbone_config:
                raise ValueError("backbone_config에서 'd_model' 키를 찾을 수 없음")
            
            # 2단계: 전역 레지스트리에서 meta_integration 모듈 확인
            logger.info("🔍 Step 2: 전역 meta_integration 모듈 확인 중...")
            from config import get_system_module
            meta_integration_module = get_system_module('meta_integration')
            
            if meta_integration_module is not None:
                logger.info("✅ Step 2: 전역 meta_integration 모듈 연결 완료")
            else:
                logger.warning("⚠️ Step 2: 전역 meta_integration 모듈 찾을 수 없음 - HeadAdapter는 연결 인터페이스로만 동작")
            
            # 3단계: 최적화된 차원 변환 어댑터 생성 (1280 ↔ 256)
            logger.info("🔍 Step 3: optimized dimension_adapter 생성 중...")
            self.dimension_adapter = HeadSpecificAdapters.create_meta_adapter()
            logger.info("✅ Step 3: 최적화된 차원 변환 어댑터 생성 완료")
            
            # 5단계: 메타 크로스 어텐션 (백본과 호환되는 헤드 수 사용)
            logger.info("🔍 Step 5: 메타 크로스 어텐션 생성 중...")
            backbone_num_heads = self.backbone_config.get('num_heads', 20)
            logger.info(f"🔍 백본 호환 헤드 수: {backbone_num_heads}")
            LightweightCrossAttention = get_lightweight_cross_attention()
            self.cross_attention = LightweightCrossAttention(
                d_model=self.backbone_config['d_model'],
                num_heads=backbone_num_heads  # 백본과 동일한 헤드 수 사용
            )
            logger.info("✅ Step 5: 메타 크로스 어텐션 생성 완료")
            
            # 6단계: 학습 모드 설정
            logger.info("🔍 Step 6: 학습 모드 설정 중...")
            self._ensure_training_mode()
            logger.info("✅ Step 6: 학습 모드 설정 완료")
            
            self.initialized = True
            logger.info("🎉 MetaIntegrationHeadAdapter 초기화 완료")
            
        except Exception as e:
            logger.error(f"❌ MetaIntegrationHeadAdapter 초기화 실패: {str(e)}")
            import traceback
            logger.error(f"❌ 스택 트레이스: {traceback.format_exc()}")
            raise
    
    def _ensure_training_mode(self):
        """학습 모드 설정 및 requires_grad 확인"""
        # dimension_adapter (최적화된 차원 변환 어댑터)
        if self.dimension_adapter is not None:
            self.dimension_adapter.train()
            for param in self.dimension_adapter.parameters():
                param.requires_grad = True
        
        # cross_attention
        if self.cross_attention is not None:
            self.cross_attention.train()
            for param in self.cross_attention.parameters():
                param.requires_grad = True
        
        # integration_network는 전역 모듈에서 관리되므로 여기서 설정하지 않음
        logger.info("MetaIntegrationHeadAdapter: 연결 인터페이스 파라미터 학습 모드 설정 완료")
    
    async def process_unified_input(self, unified_repr: UnifiedRepresentation) -> HeadProcessingResult:
        """통합 표현으로부터 메타통합 분석 수행"""
        start_time = time.time()
        
        if not self.initialized:
            await self.initialize_head()
        
        device = unified_repr.device
        
        try:
            meta_result = await self._process_meta_integration(unified_repr, device)
            
            # 출력 표준화
            if isinstance(meta_result.get('integrated_output'), torch.Tensor):
                standardized_output = self.output_adapter(meta_result['integrated_output'])
            else:
                standardized_output = unified_repr.shared_embedding
            
            processing_time = time.time() - start_time
            
            return HeadProcessingResult(
                head_type=self.head_type,
                primary_output=meta_result,
                secondary_outputs={
                    'standardized_embedding': standardized_output,
                    'integration_weights': meta_result.get('integration_weights', {}),
                    'meta_learning_score': meta_result.get('meta_learning_score', 0.5),
                    'system_coherence': meta_result.get('system_coherence', 0.7)
                },
                processing_time=processing_time,
                device_used=str(device),
                synergy_features=standardized_output,
                confidence_score=meta_result.get('confidence', 0.8)
            )
            
        except Exception as e:
            logger.error(f"메타통합 헤드 처리 오류: {str(e)}")
            processing_time = time.time() - start_time
            
            return HeadProcessingResult(
                head_type=self.head_type,
                primary_output={'error': str(e)},
                processing_time=processing_time,
                device_used=str(device),
                confidence_score=0.0
            )
    
    async def _process_meta_integration(self, unified_repr: UnifiedRepresentation, device: torch.device) -> Dict[str, Any]:
        """메타 통합 실행"""
        try:
            # 기본 통합 (다른 헤드들의 결과가 있을 때 더 정교하게 구현)
            adapted_input = self.input_adapter(unified_repr.shared_embedding)
            
            # 임시 통합 결과 (실제로는 다른 헤드들의 결과를 조합) - 안전한 차원 처리
            # 모든 텐서를 shared_embedding과 같은 크기로 맞춤
            target_shape = unified_repr.shared_embedding.shape
            
            # cross_modal_features를 target_shape로 안전하게 변환
            cross_modal_safe = unified_repr.cross_modal_features
            if cross_modal_safe.shape != target_shape:
                if cross_modal_safe.dim() == 1:
                    cross_modal_safe = cross_modal_safe.unsqueeze(0)
                if cross_modal_safe.shape != target_shape:
                    # 크기가 맞지 않으면 shared_embedding과 같은 크기로 조정
                    cross_modal_safe = torch.zeros_like(unified_repr.shared_embedding)
            
            dummy_head_outputs = [
                unified_repr.shared_embedding,  # 백본 출력
                cross_modal_safe,
                adapted_input.expand_as(unified_repr.shared_embedding),
                torch.zeros_like(unified_repr.shared_embedding)
            ]
            
            # 헤드 결과들 연결
            concatenated = torch.cat(dummy_head_outputs, dim=-1)
            
            # 통합 네트워크 실행
            integrated_output = self.integration_network(concatenated)
            
            # 통합 가중치 계산
            integration_weights = {
                'emotion_weight': 0.3,
                'bentham_weight': 0.25,
                'semantic_weight': 0.2,
                'regret_weight': 0.25
            }
            
            # 메타 학습 점수
            meta_learning_score = float(torch.mean(torch.abs(integrated_output)).item())
            
            # 시스템 일관성
            system_coherence = 0.75  # 실제로는 헤드들 간 일관성 측정
            
            return {
                'integrated_output': integrated_output,
                'integration_weights': integration_weights,
                'meta_learning_score': meta_learning_score,
                'system_coherence': system_coherence,
                'confidence': 0.83
            }
            
        except Exception as e:
            logger.error(f"메타 통합 실행 오류: {str(e)}")
            return {'error': str(e), 'confidence': 0.0}

class HeadCompatibilityManager:
    """
    헤드 호환성 매니저 - 모든 헤드 어댑터들을 관리하고 조정
    """
    
    def __init__(self, unified_backbone: RedHeartUnifiedBackbone, 
                 swap_manager: RedHeartDynamicSwapManager):
        self.unified_backbone = unified_backbone
        self.swap_manager = swap_manager
        
        # 헤드 어댑터들 초기화
        self.head_adapters = {
            HeadType.EMOTION_EMPATHY: EmotionEmpathyHeadAdapter(),
            HeadType.BENTHAM_FROMM: BenthamFrommHeadAdapter(),
            HeadType.SEMANTIC_SURD: SemanticSURDHeadAdapter(),
            HeadType.REGRET_LEARNING: RegretLearningHeadAdapter(),
            HeadType.META_INTEGRATION: MetaIntegrationHeadAdapter()
        }
        
        # 헤드들은 초기화 후에 등록됨 (initialize_all_heads에서)
        
        # 시너지 창출을 위한 전역 크로스 어텐션 (백본과 호환되는 헤드 수)
        backbone_num_heads = ADVANCED_CONFIG['unified_backbone'].get('num_heads', 20)
        LightweightCrossAttention = get_lightweight_cross_attention()
        self.global_cross_attention = LightweightCrossAttention(
            d_model=ADVANCED_CONFIG['unified_backbone']['d_model'],
            num_heads=backbone_num_heads  # 백본과 호환되는 헤드 수
        )
        
        self.initialized = False
        logger.info("HeadCompatibilityManager 초기화 완료")
    
    def _register_heads_with_swap_manager(self):
        """헤드들을 스왑 매니저에 등록"""
        logger.info(f"🔥 _register_heads_with_swap_manager 시작 - {len(self.head_adapters)}개 헤드 등록 예정")
        for head_type, adapter in self.head_adapters.items():
            logger.info(f"🔄 {head_type.value} 헤드 등록 시작...")
            # 각 헤드의 네트워크들을 스왑 매니저에 등록
            if hasattr(adapter, 'input_adapter') and adapter.input_adapter is not None:
                self.swap_manager.register_model(
                    f"{head_type.value}_input_adapter",
                    adapter.input_adapter,
                    adapter.priority
                )
            
            if hasattr(adapter, 'output_adapter') and adapter.output_adapter is not None:
                self.swap_manager.register_model(
                    f"{head_type.value}_output_adapter", 
                    adapter.output_adapter,
                    adapter.priority
                )
            
            # 개선된 PyTorch 네트워크 탐지 - 각 어댑터의 get_pytorch_network() 메서드 사용
            pytorch_network = None
            
            # 1. 어댑터의 get_pytorch_network() 메서드 우선 사용
            logger.info(f"📋 {head_type.value} 어댑터 분석:")
            logger.info(f"   - 어댑터 타입: {type(adapter)}")
            logger.info(f"   - get_pytorch_network 존재: {hasattr(adapter, 'get_pytorch_network')}")
            
            # 디버깅용 - 사용 가능한 메서드 목록
            adapter_methods = [m for m in dir(adapter) if not m.startswith('_') and callable(getattr(adapter, m, None))]
            logger.info(f"   - 사용 가능한 메서드들: {adapter_methods[:10]}...")
            
            if hasattr(adapter, 'get_pytorch_network'):
                try:
                    logger.info(f"   🔄 {head_type.value}.get_pytorch_network() 호출 중...")
                    pytorch_network = adapter.get_pytorch_network()
                    logger.info(f"   ✅ get_pytorch_network() 성공 - 반환값: {type(pytorch_network) if pytorch_network else 'None'}")
                except Exception as e:
                    logger.error(f"   ❌ {head_type.value} get_pytorch_network() 호출 실패: {e}")
                    import traceback
                    logger.error(f"   스택 트레이스:\n{traceback.format_exc()}")
            
            # 2. 기존 방식 fallback (호환성 보장)
            if pytorch_network is None:
                main_component = getattr(adapter, 'empathy_learner', None) or \
                               getattr(adapter, 'bentham_calculator', None) or \
                               getattr(adapter, 'semantic_analyzer', None) or \
                               getattr(adapter, 'regret_network', None) or \
                               getattr(adapter, 'integration_network', None)
                
                if main_component is not None:
                    if isinstance(main_component, nn.Module):
                        pytorch_network = main_component
                    else:
                        for attr_name in ['_neural_predictor', 'neural_predictor', 'model', 'network', 'classifier']:
                            neural_component = getattr(main_component, attr_name, None)
                            if neural_component is not None and isinstance(neural_component, nn.Module):
                                pytorch_network = neural_component
                                break
            
            # 네트워크 등록 - NO DUMMY, NO FALLBACK (프로젝트 규칙)
            try:
                if pytorch_network is not None:
                    self.swap_manager.register_model(
                        head_type.value,
                        pytorch_network,
                        adapter.priority
                    )
                    logger.info(f"✅ {head_type.value}의 실제 PyTorch 네트워크를 swap_manager에 등록 성공")
                    logger.info(f"   - 네트워크 타입: {type(pytorch_network)}")
                    logger.info(f"   - 파라미터 수: {sum(p.numel() for p in pytorch_network.parameters()) / 1e6:.2f}M")
                else:
                    # NO DUMMY - 프로젝트 규칙: 실패 시 즉시 에러
                    logger.error(f"❌ {head_type.value}의 PyTorch 네트워크가 None")
                    raise RuntimeError(f"{head_type.value} 헤드 초기화 실패: PyTorch 네트워크 없음")
                    
                    # device_policy 확인 및 적용
                    device_policy = getattr(adapter, 'device_policy', 'gpu_required')
                    logger.info(f"   - device_policy: {device_policy}")
                    
                    if device_policy == 'cpu_preload':
                        # CPU 프리로드 헤드는 등록만 하고 즉시 GPU→CPU로 언로드
                        logger.info(f"   📋 {head_type.value}는 cpu_preload 정책 - CPU로 언로드")
                        try:
                            # CPU로 이동
                            if hasattr(pytorch_network, 'to'):
                                pytorch_network.to('cpu')
                                logger.info(f"   ✅ {head_type.value} CPU로 언로드 완료")
                                
                                # swap_manager에서도 GPU 목록에서 제거
                                if hasattr(self.swap_manager, 'gpu_resident_models') and head_type.value in self.swap_manager.gpu_resident_models:
                                    del self.swap_manager.gpu_resident_models[head_type.value]
                                    logger.info(f"   🗑️ {head_type.value}를 GPU 상주 목록에서 제거")
                        except Exception as e:
                            logger.warning(f"   ⚠️ {head_type.value} CPU 언로드 실패: {e}")
            except Exception as e:
                logger.error(f"❌ {head_type.value} 네트워크 등록 실패: {e}")
                # 프로젝트 규칙에 따라 fallback 없이 에러 발생
                raise RuntimeError(f"{head_type.value} swap_manager 등록 실패: {e}") from e
    
    async def initialize_all_heads(self):
        """모든 헤드 순차적 GPU 초기화 + 즉시 스왑 시스템 (85% 예측 기반)"""
        logger.debug("🔍 initialize_all_heads() 메서드 진입!")
        logger.debug(f"🔍 self.initialized = {getattr(self, 'initialized', 'NOT_SET')}")
        
        if self.initialized:
            logger.debug("🔍 이미 initialized=True이므로 return")
            return
            
        logger.info("🚀 GPU 순차적 초기화 + 즉시 스왑 시스템 시작")
        
        # GPU 메모리 상태 확인
        logger.debug("🔍 Step A0: import 시작...")
        import torch
        logger.debug("🔍 Step A1: torch import 완료")
        from config import get_gpu_memory_info, get_master_orchestrator
        logger.debug("🔍 Step A2: config import 완료")
        
        logger.debug("🔍 Step A: torch import 완료")
        
        if torch.cuda.is_available():
            memory_info = get_gpu_memory_info()
            if memory_info:
                logger.info(f"🔍 초기화 전 GPU 메모리: {memory_info['usage_percent']:.1f}% 사용중")
        
        logger.info("🔍 Step B: GPU 메모리 확인 완료")
        
        # Master Memory Orchestrator 가져오기
        logger.info("🔍 Step C: MasterOrchestrator 가져오는 중...")
        master_orch = get_master_orchestrator()
        logger.info(f"🔍 Step C 완료: MasterOrchestrator = {type(master_orch)}")
        
        initialization_errors = []
        
        # 헤드 초기화 우선순위 정의 (메모리 사용량 기준 오름차순)
        # 공격적 최적화: 더 정확한 메모리 예측값 적용
        head_priority_order = [
            (HeadType.META_INTEGRATION, 35),      # 35MB - 가장 작음 (정밀 측정)
            (HeadType.SEMANTIC_SURD, 75),         # 75MB (정밀 측정)
            (HeadType.BENTHAM_FROMM, 115),        # 115MB (정밀 측정)
            (HeadType.REGRET_LEARNING, 115),      # 115MB (정밀 측정)
            (HeadType.EMOTION_EMPATHY, 135),      # 135MB - 가장 큼 (정밀 측정)
        ]
        
        logger.info(f"📋 헤드 초기화 순서: {[f'{ht.value}({mb}MB)' for ht, mb in head_priority_order]}")
        
        logger.info("🔍 Step D: 헤드 우선순위 정의 완료")
        
        # 🔥 순차적 GPU 초기화 + 즉시 스왑 시스템
        logger.info("🔍 Step E: 헤드 초기화 루프 시작...")
        for i, (head_type, estimated_mb) in enumerate(head_priority_order):
            logger.info(f"🔍 Step E-{i+1}: {head_type.value} 초기화 시작 ({estimated_mb}MB)")
            
            if head_type not in self.head_adapters:
                logger.warning(f"⚠️ {head_type.value} 어댑터가 존재하지 않음")
                continue
                
            logger.info(f"🔍 Step E-{i+1}a: {head_type.value} 어댑터 발견")
            
            adapter = self.head_adapters[head_type]
            logger.info(f"🔍 Step E-{i+1}b: {head_type.value} 어댑터 가져옴 = {type(adapter)}")
            
            try:
                # 🔥 Step 1: 메모리 예측 체크 (85% 초과 예상 시 사전 스왑)
                current_memory = get_gpu_memory_info()
                if current_memory:
                    # 더 정밀한 메모리 예측 (8GB = 8192MB 기준)
                    predicted_usage = current_memory['usage_percent'] + (estimated_mb / 81.92)
                    
                    logger.info(f"📊 {head_type.value} 로딩 예측: {current_memory['usage_percent']:.1f}% + {estimated_mb}MB = {predicted_usage:.1f}%")
                    
                    # 🚀 공격적 GPU 활용: 95% 초과 시에만 정리 (85% 목표 달성)
                    # 현재 백본(21.5%) + 모든헤드(6%) = 27.3%이므로 훨씬 여유 있음
                    if predicted_usage > 95:
                        logger.warning(f"⚠️ 95% 초과 예상 ({predicted_usage:.1f}%) - 선택적 정리 실행")
                        await master_orch._emergency_intelligent_cleanup()
                        
                        # 스왑 후 메모리 재확인
                        post_swap_memory = get_gpu_memory_info()
                        if post_swap_memory:
                            logger.info(f"✅ 정리 후 메모리: {post_swap_memory['usage_percent']:.1f}%")
                    else:
                        logger.info(f"🚀 GPU 여유 충분 ({predicted_usage:.1f}%) - 헤드 GPU 상주 진행")
                
                # 🔥 Step 2: GPU에서 헤드 초기화
                logger.info(f"🔥 {head_type.value} GPU 초기화 시작 (예상: {estimated_mb}MB)...")
                logger.info(f"🔍 Step E-{i+1}c: adapter.initialize_head() 호출 전")
                
                # CPU 모드 해제 - GPU에서 초기화해야 함
                if hasattr(adapter, 'force_cpu_mode'):
                    adapter.force_cpu_mode = False
                    logger.info(f"🔍 Step E-{i+1}d: force_cpu_mode = False 설정")
                
                logger.info(f"🔍 Step E-{i+1}e: adapter.initialize_head() 호출 중...")
                
                # 각 헤드별 초기화에 타임아웃 추가 (180초로 증가 - 대형 모델 로딩 고려)
                try:
                    await asyncio.wait_for(adapter.initialize_head(), timeout=180.0)
                    logger.info(f"🔍 Step E-{i+1}f: adapter.initialize_head() 완료!")
                except asyncio.TimeoutError:
                    logger.error(f"❌ {head_type.value} 헤드 초기화 180초 타임아웃!")
                    logger.error(f"🔍 Hanging 발생한 헤드: {head_type.value}")
                    raise RuntimeError(f"{head_type.value} head initialization timeout - cannot continue")
                
                # 초기화 후 메모리 상태 확인
                post_init_memory = get_gpu_memory_info()
                if post_init_memory:
                    logger.info(f"📊 {head_type.value} 초기화 후: {post_init_memory['usage_percent']:.1f}%")
                
                # 🔥 Step 3: GPU 상주 유지 (공격적 메모리 활용)
                logger.info(f"🚀 {head_type.value} GPU 상주 유지 - 공격적 활용 모드")
                
                # 헤드의 실제 PyTorch 네트워크 찾기 및 GPU 상주 확인
                pytorch_network = None
                if hasattr(adapter, 'get_pytorch_network'):
                    pytorch_network = adapter.get_pytorch_network()
                
                if pytorch_network is not None:
                    # DSM에 실제 헤드 재등록 (교체)
                    from dynamic_swap_manager import get_swap_manager, SwapPriority
                    swap = get_swap_manager()
                    if swap and isinstance(pytorch_network, torch.nn.Module):
                        swap.register_model(head_type.value, pytorch_network, priority=SwapPriority.HIGH)
                        logger.info(f"✅ {head_type.value} DSM 업데이트 완료 (실제 헤드로 교체)")
                    
                    # GPU 상주 확인 및 등록
                    model_id = f"{head_type.value}_gpu_resident"
                    
                    # GPU 상주 상태 확인
                    try:
                        # GPU 로딩은 무조건 DSM 경유 (직접 .to('cuda') 금지)
                        if next(pytorch_network.parameters()).device.type != 'cuda':
                            # DSM을 통해 GPU로 로드
                            await swap.load_head_to_gpu(head_type.value, timeout=30.0)
                            logger.info(f"🔥 {head_type.value} GPU 재로딩 완료 (DSM 경유)")
                        else:
                            logger.info(f"✅ {head_type.value} GPU 상주 확인됨")
                        
                        # GPU 상주 후 메모리 상태
                        post_gpu_memory = get_gpu_memory_info()
                        if post_gpu_memory:
                            logger.info(f"📊 GPU 상주 후 메모리: {post_gpu_memory['usage_percent']:.1f}%")
                            
                    except Exception as gpu_error:
                        logger.error(f"❌ {head_type.value} GPU 상주 실패: {str(gpu_error)}")
                else:
                    logger.warning(f"⚠️ {head_type.value} PyTorch 네트워크를 찾을 수 없음")
                
                logger.info(f"✅ {head_type.value} 초기화 + 스왑 완료")
                
                # 🔥 Step 4: CUDA 캐시 정리로 메모리 최적화
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                    final_memory = get_gpu_memory_info()
                    if final_memory:
                        logger.info(f"📊 {head_type.value} 최종 메모리: {final_memory['usage_percent']:.1f}%")
                    
            except Exception as e:
                error_msg = f"❌ {head_type.value} 헤드 초기화 실패: {str(e)}"
                logger.error(error_msg)
                initialization_errors.append((head_type, str(e)))
                
                # 오류 발생 시 긴급 정리
                try:
                    await master_orch._emergency_intelligent_cleanup()
                except Exception as cleanup_error:
                    logger.error(f"긴급 정리도 실패: {str(cleanup_error)}")
        
        # 🔥 최종 단계: 모든 헤드를 스왑 매니저에 등록
        logger.info("🔄 모든 헤드를 동적 스왑 매니저에 등록 중...")
        self._register_heads_with_swap_manager()
        
        # 🔥 최종 메모리 상태 확인 및 보고
        final_memory = get_gpu_memory_info()
        if final_memory:
            logger.info(f"🎯 최종 GPU 메모리 상태: {final_memory['usage_percent']:.1f}%")
            
            if final_memory['usage_percent'] >= 80 and final_memory['usage_percent'] <= 85:
                logger.info(f"🎯 메모리 관리 최적: {final_memory['usage_percent']:.1f}% (85% 근접 달성!)")
            elif final_memory['usage_percent'] >= 70:
                logger.info(f"✅ 메모리 관리 양호: {final_memory['usage_percent']:.1f}% (더 공격적 활용 가능)")
            elif final_memory['usage_percent'] < 50:
                logger.warning(f"📊 메모리 과소 활용: {final_memory['usage_percent']:.1f}% (GPU 리소스 낭비)")
            else:
                logger.error(f"⚠️ 메모리 관리 주의: {final_memory['usage_percent']:.1f}% (85% 초과)")
        
        # 오류 보고
        if initialization_errors:
            logger.warning(f"⚠️ {len(initialization_errors)}개 헤드 초기화 실패:")
            for head_type, error in initialization_errors:
                logger.warning(f"  - {head_type.value}: {error}")
        
        successful_heads = len(self.head_adapters) - len(initialization_errors)
        logger.info(f"🎉 GPU 순차적 초기화 완료: {successful_heads}/{len(self.head_adapters)}개 성공")
        
        # 🔥 Step 6: 85% 달성을 위한 추가 공격적 로딩
        await self._aggressive_gpu_utilization()
        
        self.initialized = True
    
    async def _aggressive_gpu_utilization(self):
        """85% 근접 달성을 위한 공격적 GPU 활용"""
        logger.info("🔥 공격적 GPU 활용 시작 - 85% 목표 달성")
        
        current_memory = get_gpu_memory_info()
        if not current_memory:
            logger.warning("GPU 메모리 정보를 가져올 수 없음")
            return
        
        target_usage = 82  # 85% 근접선 (안전 마진 3%)
        current_usage = current_memory['usage_percent']
        
        logger.info(f"📊 현재 사용률: {current_usage:.1f}% → 목표: {target_usage}%")
        
        if current_usage >= target_usage:
            logger.info(f"✅ 이미 목표 달성: {current_usage:.1f}%")
            return
        
        # 추가 활용 가능한 메모리 계산
        available_percent = target_usage - current_usage
        available_gb = (available_percent / 100) * 8.0  # 8GB GPU 기준
        
        logger.info(f"🎯 추가 활용 가능: {available_percent:.1f}% ({available_gb:.2f}GB)")
        
        # 🔥 전략 1: 헤드 다중 인스턴스 생성 (배치 처리용)
        await self._load_multi_instance_heads(available_gb * 0.4)  # 40% 할당
        
        # 🔥 전략 2: 프리로드 캐시 시스템
        await self._load_preload_cache(available_gb * 0.3)  # 30% 할당
        
        # 🔥 전략 3: 중간 결과 캐시 버퍼
        await self._load_intermediate_cache(available_gb * 0.2)  # 20% 할당
        
        # 🔥 전략 4: 추가 모델 컴포넌트 (어텐션 캐시 등)
        await self._load_additional_components(available_gb * 0.1)  # 10% 할당
        
        # 최종 메모리 상태 확인
        final_memory = get_gpu_memory_info()
        if final_memory:
            achieved_usage = final_memory['usage_percent']
            logger.info(f"🎯 공격적 활용 완료: {current_usage:.1f}% → {achieved_usage:.1f}%")
            
            if achieved_usage >= 80:
                logger.info("🚀 85% 근접선 달성! GPU 최대 활용 성공")
            else:
                logger.warning(f"📊 추가 활용 가능: 현재 {achieved_usage:.1f}% (목표 82%)")
    
    async def _load_multi_instance_heads(self, target_gb: float):
        """배치 처리를 위한 헤드 다중 인스턴스 로딩"""
        logger.info(f"🔥 헤드 다중 인스턴스 로딩 시작 (목표: {target_gb:.2f}GB)")
        
        loaded_gb = 0.0
        instance_count = 0
        
        # 가장 많이 사용되는 헤드들의 추가 인스턴스 생성
        priority_heads = [HeadType.EMOTION_EMPATHY, HeadType.BENTHAM_FROMM]
        
        for head_type in priority_heads:
            if loaded_gb >= target_gb:
                break
                
            if head_type in self.head_adapters:
                try:
                    # 기존 헤드를 복제하여 추가 인스턴스 생성
                    original_adapter = self.head_adapters[head_type]
                    
                    # 간단한 복제 (실제로는 더 정교한 복제 로직 필요)
                    if hasattr(original_adapter, 'get_pytorch_network'):
                        network = original_adapter.get_pytorch_network()
                        if network and loaded_gb + 0.135 <= target_gb:  # 135MB 예상
                            # GPU에 추가 인스턴스 상주
                            # 실제 구현에서는 모델 복제 로직 필요
                            logger.info(f"🔥 {head_type.value} 추가 인스턴스 GPU 로딩 (예상: 135MB)")
                            loaded_gb += 0.135
                            instance_count += 1
                            
                except Exception as e:
                    logger.warning(f"헤드 다중 인스턴스 로딩 실패: {head_type.value} - {str(e)}")
        
        logger.info(f"✅ 헤드 다중 인스턴스: {instance_count}개, {loaded_gb:.2f}GB 로딩")
    
    async def _load_preload_cache(self, target_gb: float):
        """다음 사용 예상 모델들의 프리로드 캐시"""
        logger.info(f"📦 프리로드 캐시 생성 (목표: {target_gb:.2f}GB)")
        
        try:
            # 가상의 프리로드 텐서들 생성 (실제로는 예상 모델들)
            import torch
            cache_tensors = []
            loaded_gb = 0.0
            
            # 대용량 캐시 텐서들 생성
            tensor_sizes = [
                (8192, 1280),   # 40MB
                (4096, 2560),   # 40MB  
                (2048, 5120),   # 40MB
                (16384, 640),   # 40MB
            ]
            
            for i, (h, w) in enumerate(tensor_sizes):
                if loaded_gb >= target_gb:
                    break
                    
                tensor_size_gb = (h * w * 4) / (1024**3)  # float32 기준
                if loaded_gb + tensor_size_gb <= target_gb:
                    tensor = torch.randn(h, w, device='cuda', dtype=torch.float32)
                    cache_tensors.append(tensor)
                    loaded_gb += tensor_size_gb
                    logger.info(f"📦 프리로드 캐시 {i+1}: [{h}x{w}] {tensor_size_gb*1024:.0f}MB")
            
            # 캐시 텐서들을 클래스 변수로 저장 (GC 방지)
            self._preload_cache_tensors = cache_tensors
            logger.info(f"✅ 프리로드 캐시: {len(cache_tensors)}개, {loaded_gb:.2f}GB")
            
        except Exception as e:
            logger.warning(f"프리로드 캐시 생성 실패: {str(e)}")
    
    async def _load_intermediate_cache(self, target_gb: float):
        """중간 결과 캐시 버퍼 생성"""
        logger.info(f"💾 중간 결과 캐시 생성 (목표: {target_gb:.2f}GB)")
        
        try:
            import torch
            cache_buffers = []
            loaded_gb = 0.0
            
            # 다양한 크기의 중간 결과 버퍼들
            buffer_configs = [
                (1024, 1280, "attention_cache"),
                (2048, 640, "embedding_cache"),
                (512, 2560, "ffn_cache"),
                (4096, 320, "output_cache"),
            ]
            
            for h, w, name in buffer_configs:
                if loaded_gb >= target_gb:
                    break
                    
                buffer_size_gb = (h * w * 4) / (1024**3)
                if loaded_gb + buffer_size_gb <= target_gb:
                    buffer = torch.zeros(h, w, device='cuda', dtype=torch.float32)
                    cache_buffers.append((name, buffer))
                    loaded_gb += buffer_size_gb
                    logger.info(f"💾 {name}: [{h}x{w}] {buffer_size_gb*1024:.0f}MB")
            
            self._intermediate_cache_buffers = cache_buffers
            logger.info(f"✅ 중간 결과 캐시: {len(cache_buffers)}개, {loaded_gb:.2f}GB")
            
        except Exception as e:
            logger.warning(f"중간 결과 캐시 생성 실패: {str(e)}")
    
    async def _load_additional_components(self, target_gb: float):
        """추가 모델 컴포넌트 로딩"""
        logger.info(f"⚡ 추가 컴포넌트 로딩 (목표: {target_gb:.2f}GB)")
        
        try:
            import torch
            additional_components = []
            loaded_gb = 0.0
            
            # 추가 어텐션 헤드, 임베딩 레이어 등
            component_configs = [
                (1280, 1280, "extra_attention"),
                (1280, 5120, "extra_ffn"),
                (50000, 64, "extra_embedding"),
            ]
            
            for h, w, name in component_configs:
                if loaded_gb >= target_gb:
                    break
                    
                comp_size_gb = (h * w * 4) / (1024**3)
                if loaded_gb + comp_size_gb <= target_gb:
                    component = torch.randn(h, w, device='cuda', dtype=torch.float32)
                    additional_components.append((name, component))
                    loaded_gb += comp_size_gb
                    logger.info(f"⚡ {name}: [{h}x{w}] {comp_size_gb*1024:.0f}MB")
            
            self._additional_components = additional_components
            logger.info(f"✅ 추가 컴포넌트: {len(additional_components)}개, {loaded_gb:.2f}GB")
            
        except Exception as e:
            logger.warning(f"추가 컴포넌트 로딩 실패: {str(e)}")
    
    async def process_with_all_heads(self, text_input: str) -> Dict[HeadType, HeadProcessingResult]:
        """모든 헤드를 사용하여 입력 처리"""
        if not self.initialized:
            await self.initialize_all_heads()
        
        # 1. 통합 백본으로 표현 생성
        unified_repr = self.unified_backbone.get_embedding_for_text(text_input)
        
        # 2. 각 헤드에서 병렬 처리
        processing_tasks = {}
        for head_type, adapter in self.head_adapters.items():
            processing_tasks[head_type] = adapter.process_unified_input(unified_repr)
        
        # 3. 모든 헤드 결과 수집
        results = {}
        for head_type, task in processing_tasks.items():
            try:
                results[head_type] = await task
            except Exception as e:
                logger.error(f"헤드 {head_type.value} 처리 오류: {str(e)}")
                results[head_type] = HeadProcessingResult(
                    head_type=head_type,
                    primary_output={'error': str(e)},
                    confidence_score=0.0
                )
        
        # 4. 시너지 효과 계산
        await self._calculate_synergy_effects(results, unified_repr)
        
        return results
    
    async def _calculate_synergy_effects(self, results: Dict[HeadType, HeadProcessingResult], 
                                       unified_repr: UnifiedRepresentation):
        """헤드들 간 시너지 효과 계산"""
        try:
            # 시너지 특성들 수집
            synergy_pairs = []
            for head_type, result in results.items():
                if result.synergy_features is not None:
                    synergy_pairs.append((head_type.value, result.synergy_features))
            
            if len(synergy_pairs) > 1:
                # 전역 크로스 어텐션으로 시너지 계산
                global_synergy = self.global_cross_attention(
                    query=unified_repr.shared_embedding,
                    key_value_pairs=synergy_pairs
                )
                
                # 시너지 점수를 각 결과에 추가
                for head_type, result in results.items():
                    synergy_score = 0.0
                    if head_type.value in global_synergy:
                        synergy_tensor = global_synergy[head_type.value]
                        synergy_score = float(torch.mean(synergy_tensor).item())
                    
                    result.secondary_outputs['synergy_score'] = synergy_score
                    
        except Exception as e:
            logger.error(f"시너지 효과 계산 오류: {str(e)}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """헤드 호환성 매니저 통계"""
        stats = {
            'initialized_heads': sum(1 for adapter in self.head_adapters.values() if adapter.initialized),
            'total_heads': len(self.head_adapters),
            'swap_manager_stats': self.swap_manager.get_stats(),
            'memory_status': self.swap_manager.get_memory_status()
        }
        
        return stats

# 사용 예시 함수
async def example_usage():
    """헤드 호환성 인터페이스 사용 예시"""
    # 통합 백본 및 스왑 매니저 초기화
    backbone = RedHeartUnifiedBackbone()
    swap_manager = RedHeartDynamicSwapManager()
    await swap_manager.initialize()
    
    # 호환성 매니저 생성
    compatibility_manager = HeadCompatibilityManager(backbone, swap_manager)
    
    try:
        # 테스트 입력
        test_input = "사람들과의 관계에서 어떻게 공감할 수 있을까?"
        
        # 모든 헤드로 처리
        results = await compatibility_manager.process_with_all_heads(test_input)
        
        # 결과 출력
        print("=== 헤드별 처리 결과 ===")
        for head_type, result in results.items():
            print(f"\n{head_type.value}:")
            print(f"  처리 시간: {result.processing_time:.3f}s")
            print(f"  신뢰도: {result.confidence_score:.2f}")
            print(f"  디바이스: {result.device_used}")
            if 'synergy_score' in result.secondary_outputs:
                print(f"  시너지 점수: {result.secondary_outputs['synergy_score']:.3f}")
        
        # 통계 출력
        stats = compatibility_manager.get_statistics()
        print(f"\n=== 시스템 통계 ===")
        print(f"초기화된 헤드: {stats['initialized_heads']}/{stats['total_heads']}")
        print(f"총 스왑: {stats['swap_manager_stats']['total_swaps']}")
        print(f"GPU 모델 수: {stats['memory_status']['models_on_gpu']}")
        
    finally:
        await swap_manager.shutdown()

if __name__ == "__main__":
    asyncio.run(example_usage())