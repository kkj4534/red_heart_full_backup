"""
HuggingFace 모델 로딩 래퍼
메모리 매니저와 통합하여 모든 모델 로드를 추적
"""

import os
import logging
from typing import Any, Dict, Optional, Union, Callable
from functools import wraps
import torch
from transformers import (
    AutoModel, AutoModelForSequenceClassification, 
    AutoModelForTokenClassification, AutoModelForCausalLM,
    AutoTokenizer, pipeline
)

logger = logging.getLogger(__name__)

class HFModelWrapper:
    """HuggingFace 모델 로딩을 메모리 매니저와 통합"""
    
    def __init__(self):
        self.memory_manager = None
        self._model_registry = {}  # model_id -> (size_mb, device, owner)
        # 원본 함수들 저장
        self._original_from_pretrained = {}
        self._original_pipeline = None
        self._original_tokenizer = None
        self._is_patched = False  # 이중 패치 방지
        
    def set_memory_manager(self, memory_manager):
        """메모리 매니저 설정"""
        self.memory_manager = memory_manager
        
    def _merge_owner(self, owner: Optional[str], kwargs: Dict) -> str:
        """owner 인자 충돌 방지 - kwargs에서 owner 추출 및 병합"""
        if 'owner' in kwargs:
            kw_owner = kwargs.pop('owner')
            if owner and kw_owner != owner:
                logger.warning(f"[HFWrapper] owner conflict: arg='{owner}' vs kwargs='{kw_owner}'. "
                             f"Using kwargs value.")
            owner = kw_owner
        return owner or "unknown"
        
    def _estimate_model_size(self, model: torch.nn.Module) -> float:
        """모델 크기 추정 (MB) - dtype 인식"""
        total_bytes = 0
        requires_grad_params = 0
        
        for p in model.parameters():
            # dtype별 바이트 크기 계산
            bytes_per_param = p.element_size()  # dtype 크기 자동 반영
            param_bytes = p.numel() * bytes_per_param
            total_bytes += param_bytes
            
            # 학습 가능한 파라미터는 그래디언트 공간도 필요
            if p.requires_grad:
                total_bytes += param_bytes  # 그래디언트 공간
                requires_grad_params += p.numel()
        
        # 옵티마이저 상태 추가 (학습 모드인 경우)
        if requires_grad_params > 0:
            # Adam 옵티마이저 기준: momentum + variance
            total_bytes += requires_grad_params * 4 * 2  # float32 기준
        
        size_mb = total_bytes / (1024 * 1024)
        logger.debug(f"모델 크기 추정: {size_mb:.1f}MB (학습가능: {requires_grad_params > 0})")
        return size_mb
        
    def _estimate_model_size_predicted(self, model_class_or_task: Any, kwargs: Dict) -> float:
        """모델 로드 전 크기 예측 (MB)"""
        # 태스크/모델별 기본 크기 추정
        size_estimates = {
            # 태스크별
            'sentiment-analysis': 500,
            'text-classification': 500,
            'token-classification': 600,
            'question-answering': 800,
            'text-generation': 1000,
            'translation': 1200,
            # 모델 클래스별
            'AutoModelForSequenceClassification': 500,
            'AutoModelForTokenClassification': 600,
            'AutoModelForCausalLM': 1000,
            'AutoModel': 400,
        }
        
        # 모델 이름으로 크기 추정
        model_name = kwargs.get('model_name_or_path', '') or kwargs.get('model', '')
        if 'large' in model_name.lower():
            multiplier = 2.0
        elif 'base' in model_name.lower():
            multiplier = 1.0
        elif 'small' in model_name.lower() or 'tiny' in model_name.lower():
            multiplier = 0.5
        else:
            multiplier = 1.0
            
        # 클래스 또는 태스크로 기본 크기 결정
        if hasattr(model_class_or_task, '__name__'):
            base_size = size_estimates.get(model_class_or_task.__name__, 500)
        else:
            base_size = size_estimates.get(str(model_class_or_task), 500)
            
        # dtype 고려
        torch_dtype = kwargs.get('torch_dtype', None)
        if torch_dtype == torch.float16:
            dtype_multiplier = 0.5
        else:
            dtype_multiplier = 1.0
            
        estimated_mb = base_size * multiplier * dtype_multiplier
        
        # 학습 모드 고려
        if kwargs.get('requires_grad', True):
            estimated_mb *= 2  # 그래디언트 공간
            
        return estimated_mb
        
    async def _request_memory_async(self, module_name: str, required_mb: float, deps: list = None):
        """비동기 메모리 요청"""
        if self.memory_manager and hasattr(self.memory_manager, 'request_gpu'):
            try:
                success = await self.memory_manager.request_gpu(
                    module_name=module_name,
                    required_mb=required_mb,
                    deps=deps or [],
                    target_util=0.85
                )
                if not success:
                    logger.warning(f"⚠️ 메모리 요청 실패: {module_name} ({required_mb:.1f}MB)")
                return success
            except Exception as e:
                logger.warning(f"⚠️ 메모리 요청 중 오류: {e}")
                return False
        return True
        
    def _register_model(self, model_id: str, model: Any, owner: str, device: torch.device, force_cpu_init: bool = False):
        """모델을 메모리 매니저에 등록"""
        if isinstance(model, torch.nn.Module):
            size_mb = self._estimate_model_size(model)
        else:
            # pipeline 등의 경우 내부 모델 확인
            if hasattr(model, 'model'):
                size_mb = self._estimate_model_size(model.model)
            else:
                size_mb = 500  # 기본값
                
        self._model_registry[model_id] = {
            'size_mb': size_mb,
            'device': str(device),
            'owner': owner,
            'force_cpu_init': force_cpu_init  # FORCE_CPU_INIT 모드 추적
        }
        
        # 메모리 매니저에 등록
        if self.memory_manager:
            from workflow_aware_memory_manager import WorkflowAwareMemoryManager
            if isinstance(self.memory_manager, WorkflowAwareMemoryManager):
                # GPU 모델인 경우 소유권 태깅
                if device.type == 'cuda':
                    if hasattr(self.memory_manager, '_gpu_models'):
                        self.memory_manager._gpu_models[model_id] = {
                            'model': model,
                            'size_mb': size_mb,
                            'owner': owner,
                            'last_used': 0
                        }
                        logger.info(f"✅ 모델 '{model_id}' 메모리 매니저에 등록됨 (소유자: {owner}, 크기: {size_mb:.1f}MB)")
        
        return model
        
    def _register_tokenizer(self, tokenizer_id: str, tokenizer: Any, owner: str):
        """토크나이저를 CPU 전용으로 등록"""
        # 토크나이저는 GPU 메모리를 사용하지 않음
        self._model_registry[tokenizer_id] = {
            'size_mb': 0,  # GPU 메모리 사용 없음
            'device': 'cpu',
            'owner': owner,
            'type': 'tokenizer'  # 타입 구분
        }
        
        logger.info(f"✅ 토크나이저 '{tokenizer_id}' CPU 전용으로 등록됨 (소유자: {owner})")
        
        return tokenizer
        
    def wrapped_from_pretrained(self, model_class, model_name: str, 
                               *, owner: Optional[str] = None, **kwargs) -> Any:
        """AutoModel.from_pretrained 래퍼"""
        # Tokenizer는 원본 함수 직접 호출 (무한 루프 방지)
        if 'Tokenizer' in model_class.__name__:
            original_fn = self._original_from_pretrained.get(model_class.__name__)
            if original_fn:
                return original_fn(model_name, **kwargs)
            else:
                # 원본 함수가 없으면 직접 호출
                return model_class.from_pretrained(model_name, **kwargs)
        
        # owner 충돌 방지
        owner = self._merge_owner(owner, kwargs)
        
        # 이중 안전장치
        if 'owner' in kwargs:
            logger.warning("[HFWrapper] 'owner' remained in kwargs after merge. Forcing pop.")
            kwargs.pop('owner', None)
        
        logger.info(f"🔄 HF 모델 로딩 중: {model_name} (소유자: {owner})")
        
        # CPU 전용 owner 확인
        CPU_ONLY_OWNERS = {"translator"}
        
        # FORCE_CPU_INIT 환경변수 체크
        force_cpu_init = os.environ.get('FORCE_CPU_INIT', '0') == '1'
        
        # 디바이스 확인
        device_map = kwargs.get('device_map', None)
        if owner in CPU_ONLY_OWNERS or force_cpu_init:
            device = torch.device('cpu')  # CPU 전용 owner 또는 FORCE_CPU_INIT 모드
            if force_cpu_init:
                logger.debug(f"FORCE_CPU_INIT 모드 → device=cpu 강제 (owner: {owner})")
            else:
                logger.debug(f"CPU 전용 owner({owner}) → device=cpu 강제")
        elif device_map == "cpu":
            device = torch.device('cpu')
        elif device_map is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device('cuda')  # device_map이 있으면 보통 GPU
            
        # 모델 ID 미리 생성
        model_id = f"{owner}_{model_name.split('/')[-1]}"
        
        # 사전 메모리 요청 (GPU 모델인 경우 + CPU 전용 owner 제외 + FORCE_CPU_INIT 제외)
        if device.type == 'cuda' and self.memory_manager and owner not in CPU_ONLY_OWNERS and not force_cpu_init:
            estimated_mb = self._estimate_model_size_predicted(model_class, kwargs)
            logger.info(f"📊 모델 로드 전 메모리 요청: {model_id} ({estimated_mb:.1f}MB)")
            
            # 동기적 메모리 요청 - 필수 모듈 판단
            is_required = owner in ['emotion_analyzer', 'core_backbone', 'unified_backbone', 'bentham_calculator']
            
            # request_gpu_blocking 필수 사용 (STRICT_NO_FALLBACK)
            if not hasattr(self.memory_manager, 'request_gpu_blocking'):
                raise RuntimeError("WorkflowAwareMemoryManager에 request_gpu_blocking이 구현되지 않음")
                
            # 동기 메모리 요청 실행
            success = self.memory_manager.request_gpu_blocking(
                module_name=model_id,
                required_mb=estimated_mb,
                deps=kwargs.get('deps', []),
                target_util=0.85,
                timeout=30.0,
                is_required=is_required
            )
            
            # NO FALLBACK - 메모리 확보 실패 시 즉시 예외 발생
            if not success:
                raise RuntimeError(f"[GPU] 메모리 확보 실패: {model_id} (required={is_required})")
            
        # device 파라미터 제거 (from_pretrained은 device를 받지 않음)
        load_kwargs = kwargs.copy()
        if 'device' in load_kwargs:
            del load_kwargs['device']
        
        # 모델 로드 - 원본 함수 사용
        original_fn = self._original_from_pretrained.get(model_class.__name__)
        if original_fn:
            model = original_fn(model_name, **load_kwargs)
        else:
            # 폴백 - 경고 로그와 함께
            logger.warning(f"⚠️ 원본 함수를 찾을 수 없습니다: {model_class.__name__}. 직접 호출 사용.")
            model = model_class.from_pretrained(model_name, **load_kwargs)
        
        # 필요시 device로 이동 (Tokenizer는 제외)
        if device.type != 'cpu' and hasattr(model, 'to'):
            model = model.to(device)
        
        # 메모리 매니저에 등록 (실제 크기로 업데이트)
        self._register_model(model_id, model, owner, device, force_cpu_init)
        
        return model
        
    def wrapped_pipeline(self, task: str, model: str = None, 
                        *, owner: Optional[str] = None, **kwargs) -> Any:
        """pipeline 래퍼"""
        # owner 충돌 방지
        owner = self._merge_owner(owner, kwargs)
        
        # 이중 안전장치
        if 'owner' in kwargs:
            logger.warning("[HFWrapper] 'owner' remained in kwargs after merge. Forcing pop.")
            kwargs.pop('owner', None)
        
        logger.info(f"🔄 HF 파이프라인 생성 중: {task} (모델: {model}, 소유자: {owner})")
        
        # 디바이스 확인
        device_num = kwargs.get('device', -1)
        if device_num >= 0:
            device = torch.device(f'cuda:{device_num}')
        else:
            device = torch.device('cpu')
            
        # 파이프라인 ID 미리 생성
        model_name = model if model else f"{task}_default"
        pipe_id = f"{owner}_pipeline_{model_name.split('/')[-1]}"
        
        # 사전 메모리 요청 (GPU 파이프라인인 경우)
        if device.type == 'cuda' and self.memory_manager:
            estimated_mb = self._estimate_model_size_predicted(task, kwargs)
            logger.info(f"📊 파이프라인 로드 전 메모리 요청: {pipe_id} ({estimated_mb:.1f}MB)")
            
            # 파이프라인은 대부분 선택적이지만 일부는 필수
            is_required = owner in ['core_nli', 'semantic_search']
            
            # request_gpu_blocking 필수 사용 (STRICT_NO_FALLBACK)
            if not hasattr(self.memory_manager, 'request_gpu_blocking'):
                raise RuntimeError("WorkflowAwareMemoryManager에 request_gpu_blocking이 구현되지 않음")
                
            # 동기 메모리 요청 실행
            success = self.memory_manager.request_gpu_blocking(
                module_name=pipe_id,
                required_mb=estimated_mb,
                deps=kwargs.get('deps', []),
                target_util=0.85,
                timeout=30.0,
                is_required=is_required
            )
            
            # NO FALLBACK - 메모리 확보 실패 시 즉시 예외 발생
            if not success:
                raise RuntimeError(f"[GPU] 메모리 확보 실패: {pipe_id} (required={is_required})")
            
        # 파이프라인 생성 - 원본 함수 사용
        if self._original_pipeline:
            pipe = self._original_pipeline(task, model=model, **kwargs)
        else:
            # 폴백 - 경고 로그와 함께
            logger.warning("⚠️ 원본 pipeline 함수를 찾을 수 없습니다. 직접 호출 사용.")
            pipe = pipeline(task, model=model, **kwargs)
        
        # 메모리 매니저에 등록 (실제 크기로 업데이트)
        self._register_model(pipe_id, pipe, owner, device)
        
        return pipe
        
    def wrapped_tokenizer(self, model_name: str, *, owner: Optional[str] = None, **kwargs) -> Any:
        """AutoTokenizer.from_pretrained 래퍼"""
        # owner 충돌 방지
        owner = self._merge_owner(owner, kwargs)
        
        # 이중 안전장치 - merge 후에도 kwargs에 owner가 남아있으면 제거
        if 'owner' in kwargs:
            logger.warning("[HFWrapper] 'owner' remained in kwargs after merge. Forcing pop.")
            kwargs.pop('owner', None)
        
        logger.info(f"🔄 HF 토크나이저 로딩 중: {model_name} (소유자: {owner})")
        
        # 토크나이저는 항상 CPU에서 동작
        device = torch.device('cpu')
        
        # 토크나이저 ID 생성
        last_segment = model_name.split('/')[-1]
        tokenizer_id = f"{owner}_tokenizer_{last_segment}"
        
        # 토크나이저는 GPU 메모리를 사용하지 않으므로 메모리 요청 없음
        logger.debug(f"토크나이저는 CPU 전용, GPU 메모리 요청 스킵: {tokenizer_id}")
        
        # 토크나이저 로드 - 원본 함수 직접 사용 (패치 우회)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        
        # CPU 전용 메타데이터로 등록
        self._register_tokenizer(tokenizer_id, tokenizer, owner)
        
        return tokenizer
        
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """모델 정보 조회"""
        return self._model_registry.get(model_id)
        
    def get_total_gpu_usage(self) -> float:
        """전체 GPU 사용량 계산 (MB) - 토크나이저 제외"""
        total_mb = 0
        for model_id, info in self._model_registry.items():
            # GPU 모델이고 토크나이저가 아닌 경우만 계산
            if info['device'].startswith('cuda') and info.get('type') != 'tokenizer':
                total_mb += info['size_mb']
        return total_mb
        
    def list_models_by_owner(self, owner: str) -> Dict[str, Dict[str, Any]]:
        """특정 소유자의 모든 모델 조회"""
        return {
            model_id: info 
            for model_id, info in self._model_registry.items() 
            if info['owner'] == owner
        }

# 전역 인스턴스
_hf_wrapper = HFModelWrapper()

def get_hf_wrapper() -> HFModelWrapper:
    """전역 HF 래퍼 인스턴스 반환"""
    return _hf_wrapper

# 편의 함수들
def wrapped_auto_model(model_name: str, owner: str = "unknown", **kwargs):
    """AutoModel.from_pretrained 편의 함수"""
    # 만약 호출 측에서 kwargs에 owner를 실수로 넣었으면 제거해 일관화
    owner = kwargs.pop("owner", owner)
    return _hf_wrapper.wrapped_from_pretrained(
        AutoModel, model_name, owner=owner, **kwargs
    )

def wrapped_auto_model_for_sequence_classification(model_name: str, owner: str = "unknown", **kwargs):
    """AutoModelForSequenceClassification.from_pretrained 편의 함수"""
    # 만약 호출 측에서 kwargs에 owner를 실수로 넣었으면 제거해 일관화
    owner = kwargs.pop("owner", owner)
    return _hf_wrapper.wrapped_from_pretrained(
        AutoModelForSequenceClassification, model_name, owner=owner, **kwargs
    )

def wrapped_pipeline(task: str, model: str = None, owner: str = "unknown", **kwargs):
    """pipeline 편의 함수"""
    # 만약 호출 측에서 kwargs에 owner를 실수로 넣었으면 제거해 일관화
    owner = kwargs.pop("owner", owner)
    return _hf_wrapper.wrapped_pipeline(task, model=model, owner=owner, **kwargs)

def wrapped_tokenizer(model_name: str, owner: str = "unknown", **kwargs):
    """AutoTokenizer.from_pretrained 편의 함수"""
    # 만약 호출 측에서 kwargs에 owner를 실수로 넣었으면 제거해 일관화
    owner = kwargs.pop("owner", owner)
    return _hf_wrapper.wrapped_tokenizer(model_name, owner=owner, **kwargs)

def wrapped_from_pretrained(model_class, model_name: str, owner: str = "unknown", **kwargs):
    """범용 from_pretrained 편의 함수"""
    # 만약 호출 측에서 kwargs에 owner를 실수로 넣었으면 제거해 일관화
    owner = kwargs.pop("owner", owner)
    return _hf_wrapper.wrapped_from_pretrained(model_class, model_name, owner=owner, **kwargs)

# 기존 코드와의 호환성을 위한 monkey patching (선택적)
def enable_auto_registration():
    """자동 등록 활성화 - 기존 코드 수정 없이 사용"""
    import transformers
    from transformers import AutoTokenizer, AutoProcessor, AutoImageProcessor
    
    # 이중 패치 방지
    if _hf_wrapper._is_patched:
        logger.info("⚠️ HF 모델 자동 등록이 이미 활성화되어 있습니다.")
        return
    
    # 원본 함수 백업을 _hf_wrapper 인스턴스에 저장
    # Marian과 Electra import (견고한 경로)
    try:
        from transformers import MarianMTModel, MarianTokenizer
    except ImportError:
        logger.warning("transformers에서 직접 import 실패, 하위 모듈에서 시도")
        try:
            from transformers.models.marian import MarianMTModel, MarianTokenizer
        except ImportError as e:
            logger.error(f"MarianMTModel/MarianTokenizer import 실패: {e}")
            MarianMTModel = None
            MarianTokenizer = None
    
    try:
        from transformers import ElectraForSequenceClassification
    except ImportError:
        try:
            from transformers.models.electra import ElectraForSequenceClassification
        except ImportError as e:
            logger.error(f"ElectraForSequenceClassification import 실패: {e}")
            ElectraForSequenceClassification = None
    _hf_wrapper._original_from_pretrained = {
        'AutoModel': AutoModel.from_pretrained,
        'AutoModelForSequenceClassification': AutoModelForSequenceClassification.from_pretrained,
        'AutoModelForTokenClassification': AutoModelForTokenClassification.from_pretrained,
        'AutoModelForCausalLM': AutoModelForCausalLM.from_pretrained,
        'AutoTokenizer': AutoTokenizer.from_pretrained,
        'AutoProcessor': AutoProcessor.from_pretrained if hasattr(transformers, 'AutoProcessor') else None,
        'AutoImageProcessor': AutoImageProcessor.from_pretrained if hasattr(transformers, 'AutoImageProcessor') else None,
        'MarianMTModel': MarianMTModel.from_pretrained if MarianMTModel else None,
        'MarianTokenizer': MarianTokenizer.from_pretrained if MarianTokenizer else None,
        'ElectraForSequenceClassification': ElectraForSequenceClassification.from_pretrained if ElectraForSequenceClassification else None
    }
    _hf_wrapper._original_pipeline = transformers.pipeline
    _hf_wrapper._original_tokenizer = AutoTokenizer.from_pretrained
    
    # 래퍼로 교체
    def patched_from_pretrained(cls, *args, **kwargs):
        # 호출자 정보 추출
        import inspect
        frame = inspect.currentframe()
        caller_frame = frame.f_back
        owner = caller_frame.f_globals.get('__name__', 'unknown')
        
        return _hf_wrapper.wrapped_from_pretrained(cls, *args, owner=owner, **kwargs)
    
    def patched_pipeline(*args, **kwargs):
        # 호출자 정보 추출
        import inspect
        frame = inspect.currentframe()
        caller_frame = frame.f_back
        owner = caller_frame.f_globals.get('__name__', 'unknown')
        
        return _hf_wrapper.wrapped_pipeline(*args, owner=owner, **kwargs)
    
    # 이중 패치 방지를 위한 헬퍼 함수
    def patch_if_needed(cls, class_name):
        """이중 패치 방지하며 from_pretrained 패치"""
        if hasattr(cls, 'from_pretrained'):
            current_method = getattr(cls.from_pretrained, '__name__', '')
            if current_method != 'patched_from_pretrained':
                cls.from_pretrained = classmethod(patched_from_pretrained)
                logger.debug(f"✅ {class_name}.from_pretrained 패치됨")
            else:
                logger.debug(f"⚠️ {class_name}.from_pretrained 이미 패치됨, 스킵")
    
    # Monkey patching - Auto 클래스들 (토크나이저는 제외)
    patch_if_needed(AutoModel, 'AutoModel')
    patch_if_needed(AutoModelForSequenceClassification, 'AutoModelForSequenceClassification')
    patch_if_needed(AutoModelForTokenClassification, 'AutoModelForTokenClassification')
    patch_if_needed(AutoModelForCausalLM, 'AutoModelForCausalLM')
    # AutoTokenizer는 패치하지 않음 - 무한 재귀 방지
    # patch_if_needed(AutoTokenizer, 'AutoTokenizer')
    
    if hasattr(transformers, 'AutoProcessor'):
        patch_if_needed(AutoProcessor, 'AutoProcessor')
    if hasattr(transformers, 'AutoImageProcessor'):
        patch_if_needed(AutoImageProcessor, 'AutoImageProcessor')
    
    # Marian 모델만 패치, 토크나이저는 제외
    if MarianMTModel:
        patch_if_needed(MarianMTModel, 'MarianMTModel')
    # MarianTokenizer는 패치하지 않음 - 무한 재귀 방지
    # if MarianTokenizer:
    #     patch_if_needed(MarianTokenizer, 'MarianTokenizer')
    if ElectraForSequenceClassification:
        patch_if_needed(ElectraForSequenceClassification, 'ElectraForSequenceClassification')
    
    transformers.pipeline = patched_pipeline
    
    # 패치 완료 플래그 설정
    _hf_wrapper._is_patched = True
    
    logger.info("✅ HF 모델 자동 등록 활성화됨 (모델, 토크나이저, 프로세서 포함)")