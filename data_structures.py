"""
Red Heart I/O 파이프라인 표준 데이터 구조
Standard data structures for Red Heart I/O Pipeline

모든 모듈 간 통신에 사용되는 표준화된 메시지 구조 정의
NO FALLBACK 정책 - 모든 필드는 명시적이고 검증됨
"""

import time
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from enum import Enum
import json
import pickle
from datetime import datetime


class Priority(Enum):
    """작업 우선순위 정의"""
    CRITICAL = 0  # 최고 우선순위
    HIGH = 1
    NORMAL = 2
    LOW = 3
    IDLE = 4  # 최저 우선순위


class TaskType(Enum):
    """태스크 타입 정의"""
    EMOTION = "emotion"
    BENTHAM = "bentham"
    REGRET = "regret"
    SURD = "surd"
    CIRCUIT = "circuit"
    LLM_INITIAL = "llm_initial"
    LLM_FINAL = "llm_final"
    UNIFIED = "unified"
    NEURAL = "neural"
    ADVANCED = "advanced"


class ModuleType(Enum):
    """모듈 타입 정의"""
    UNIFIED_MODEL = "unified_model"
    NEURAL_ANALYZER = "neural_analyzer"
    ADVANCED_WRAPPER = "advanced_wrapper"
    LLM_ENGINE = "llm_engine"
    EMOTION_CIRCUIT = "emotion_circuit"
    TRANSLATOR = "translator"
    SENTENCE_TRANSFORMER = "sentence_transformer"


class DeviceType(Enum):
    """디바이스 타입"""
    CPU = "cpu"
    CUDA = "cuda"
    CUDA0 = "cuda:0"
    CUDA1 = "cuda:1"
    AUTO = "auto"


@dataclass
class TaskMessage:
    """
    모듈로 전달되는 태스크 메시지
    NO FALLBACK - 모든 필드는 명시적으로 설정되어야 함
    """
    module: ModuleType
    task_type: TaskType
    data: Dict[str, Any]
    priority: int = 0
    timestamp: float = field(default_factory=time.time)
    device: Optional[DeviceType] = None
    session_id: Optional[str] = None
    parent_task_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """검증 및 초기화"""
        if not isinstance(self.module, ModuleType):
            raise ValueError(f"module must be ModuleType, got {type(self.module)}")
        if not isinstance(self.task_type, TaskType):
            raise ValueError(f"task_type must be TaskType, got {type(self.task_type)}")
        if not isinstance(self.data, dict):
            raise ValueError(f"data must be dict, got {type(self.data)}")
        
        # 세션 ID 자동 생성 (없을 경우)
        if self.session_id is None:
            self.session_id = f"session_{int(self.timestamp)}_{id(self)}"
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'module': self.module.value,
            'task_type': self.task_type.value,
            'data': self.data,
            'priority': self.priority,
            'timestamp': self.timestamp,
            'device': self.device.value if self.device else None,
            'session_id': self.session_id,
            'parent_task_id': self.parent_task_id,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TaskMessage':
        """딕셔너리에서 생성"""
        return cls(
            module=ModuleType(d['module']),
            task_type=TaskType(d['task_type']),
            data=d['data'],
            priority=d.get('priority', 0),
            timestamp=d.get('timestamp', time.time()),
            device=DeviceType(d['device']) if d.get('device') else None,
            session_id=d.get('session_id'),
            parent_task_id=d.get('parent_task_id'),
            metadata=d.get('metadata', {})
        )
    
    def to_json(self) -> str:
        """JSON 문자열로 직렬화"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'TaskMessage':
        """JSON 문자열에서 역직렬화"""
        return cls.from_dict(json.loads(json_str))


@dataclass
class ResultMessage:
    """
    모듈에서 반환되는 결과 메시지
    NO FALLBACK - 모든 결과는 명시적으로 검증됨
    """
    module: ModuleType
    task_type: TaskType
    data: Dict[str, Any]
    success: bool = True
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    processing_time: Optional[float] = None
    session_id: Optional[str] = None
    task_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """검증 및 초기화"""
        if not isinstance(self.module, ModuleType):
            raise ValueError(f"module must be ModuleType, got {type(self.module)}")
        if not isinstance(self.task_type, TaskType):
            raise ValueError(f"task_type must be TaskType, got {type(self.task_type)}")
        if not isinstance(self.data, dict):
            raise ValueError(f"data must be dict, got {type(self.data)}")
        
        # 실패 시 에러 메시지 필수
        if not self.success and not self.error:
            raise ValueError("error message required when success=False")
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'module': self.module.value,
            'task_type': self.task_type.value,
            'data': self.data,
            'success': self.success,
            'error': self.error,
            'timestamp': self.timestamp,
            'processing_time': self.processing_time,
            'session_id': self.session_id,
            'task_id': self.task_id,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ResultMessage':
        """딕셔너리에서 생성"""
        return cls(
            module=ModuleType(d['module']),
            task_type=TaskType(d['task_type']),
            data=d['data'],
            success=d.get('success', True),
            error=d.get('error'),
            timestamp=d.get('timestamp', time.time()),
            processing_time=d.get('processing_time'),
            session_id=d.get('session_id'),
            task_id=d.get('task_id'),
            metadata=d.get('metadata', {})
        )


@dataclass
class EmotionData:
    """감정 분석 결과 표준 구조"""
    emotions: np.ndarray  # 7차원 감정 벡터 [joy, sadness, anger, fear, surprise, disgust, neutral]
    valence: float  # -1 to 1
    arousal: float  # 0 to 1
    dominance: float  # 0 to 1
    primary_emotion: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """검증"""
        if self.emotions.shape != (7,):
            raise ValueError(f"emotions must be 7-dimensional, got {self.emotions.shape}")
        if not -1 <= self.valence <= 1:
            raise ValueError(f"valence must be in [-1, 1], got {self.valence}")
        if not 0 <= self.arousal <= 1:
            raise ValueError(f"arousal must be in [0, 1], got {self.arousal}")
        if not 0 <= self.dominance <= 1:
            raise ValueError(f"dominance must be in [0, 1], got {self.dominance}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
    
    def to_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """감정 벡터를 텐서로 변환"""
        tensor = torch.from_numpy(self.emotions).float()
        if device:
            tensor = tensor.to(device)
        return tensor
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'emotions': self.emotions.tolist(),
            'valence': self.valence,
            'arousal': self.arousal,
            'dominance': self.dominance,
            'primary_emotion': self.primary_emotion,
            'confidence': self.confidence,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'EmotionData':
        """딕셔너리에서 생성"""
        return cls(
            emotions=np.array(d['emotions']),
            valence=d['valence'],
            arousal=d['arousal'],
            dominance=d['dominance'],
            primary_emotion=d['primary_emotion'],
            confidence=d['confidence'],
            metadata=d.get('metadata', {})
        )


@dataclass
class BenthamResult:
    """벤담 계산 결과 표준 구조"""
    bentham_scores: np.ndarray  # 10차원 벤담 점수
    intensity: float  # 쾌락 강도
    duration: float  # 지속 시간
    certainty: float  # 확실성
    propinquity: float  # 근접성
    fecundity: float  # 생산성
    purity: float  # 순수성
    extent: float  # 범위
    total_utility: float  # 총 효용
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """검증"""
        if self.bentham_scores.shape != (10,):
            raise ValueError(f"bentham_scores must be 10-dimensional, got {self.bentham_scores.shape}")
        
        # 모든 점수는 0-1 범위
        for attr in ['intensity', 'duration', 'certainty', 'propinquity', 
                    'fecundity', 'purity', 'extent']:
            value = getattr(self, attr)
            if not 0 <= value <= 1:
                raise ValueError(f"{attr} must be in [0, 1], got {value}")
    
    def to_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """벤담 점수를 텐서로 변환"""
        tensor = torch.from_numpy(self.bentham_scores).float()
        if device:
            tensor = tensor.to(device)
        return tensor
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'bentham_scores': self.bentham_scores.tolist(),
            'intensity': self.intensity,
            'duration': self.duration,
            'certainty': self.certainty,
            'propinquity': self.propinquity,
            'fecundity': self.fecundity,
            'purity': self.purity,
            'extent': self.extent,
            'total_utility': self.total_utility,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'BenthamResult':
        """딕셔너리에서 생성"""
        return cls(
            bentham_scores=np.array(d['bentham_scores']),
            intensity=d['intensity'],
            duration=d['duration'],
            certainty=d['certainty'],
            propinquity=d['propinquity'],
            fecundity=d['fecundity'],
            purity=d['purity'],
            extent=d['extent'],
            total_utility=d['total_utility'],
            metadata=d.get('metadata', {})
        )


@dataclass
class SURDMetrics:
    """SURD 분석 결과 표준 구조"""
    synergy: float  # 시너지 (0-1)
    uniqueness: float  # 고유성 (0-1)
    redundancy: float  # 중복성 (0-1)
    dependency: float  # 의존성 (0-1)
    causal_strength: float  # 인과 강도
    information_content: float  # 정보량
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """검증"""
        for attr in ['synergy', 'uniqueness', 'redundancy', 'dependency']:
            value = getattr(self, attr)
            if not 0 <= value <= 1:
                raise ValueError(f"{attr} must be in [0, 1], got {value}")
    
    def to_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """SURD 메트릭을 텐서로 변환"""
        metrics = np.array([self.synergy, self.uniqueness, 
                          self.redundancy, self.dependency])
        tensor = torch.from_numpy(metrics).float()
        if device:
            tensor = tensor.to(device)
        return tensor
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'synergy': self.synergy,
            'uniqueness': self.uniqueness,
            'redundancy': self.redundancy,
            'dependency': self.dependency,
            'causal_strength': self.causal_strength,
            'information_content': self.information_content,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SURDMetrics':
        """딕셔너리에서 생성"""
        return cls(
            synergy=d['synergy'],
            uniqueness=d['uniqueness'],
            redundancy=d['redundancy'],
            dependency=d['dependency'],
            causal_strength=d['causal_strength'],
            information_content=d['information_content'],
            metadata=d.get('metadata', {})
        )


@dataclass
class RegretAnalysis:
    """후회 분석 결과 표준 구조"""
    regret_score: float  # 0-1
    anticipated_regret: float  # 예상 후회
    experienced_regret: float  # 경험 후회
    counterfactual_thinking: float  # 반사실적 사고 강도
    temporal_distance: float  # 시간적 거리
    decision_importance: float  # 결정 중요도
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """검증"""
        for attr in ['regret_score', 'anticipated_regret', 'experienced_regret',
                    'counterfactual_thinking', 'decision_importance']:
            value = getattr(self, attr)
            if not 0 <= value <= 1:
                raise ValueError(f"{attr} must be in [0, 1], got {value}")
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'regret_score': self.regret_score,
            'anticipated_regret': self.anticipated_regret,
            'experienced_regret': self.experienced_regret,
            'counterfactual_thinking': self.counterfactual_thinking,
            'temporal_distance': self.temporal_distance,
            'decision_importance': self.decision_importance,
            'alternatives': self.alternatives,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'RegretAnalysis':
        """딕셔너리에서 생성"""
        return cls(
            regret_score=d['regret_score'],
            anticipated_regret=d['anticipated_regret'],
            experienced_regret=d['experienced_regret'],
            counterfactual_thinking=d['counterfactual_thinking'],
            temporal_distance=d['temporal_distance'],
            decision_importance=d['decision_importance'],
            alternatives=d.get('alternatives', []),
            metadata=d.get('metadata', {})
        )


@dataclass
class MemoryState:
    """메모리 상태 정보"""
    gpu_allocated_mb: float
    gpu_free_mb: float
    gpu_total_mb: float
    ram_used_mb: float
    ram_available_mb: float
    swap_count: int
    phase: str
    timestamp: float = field(default_factory=time.time)
    
    @property
    def gpu_usage_percent(self) -> float:
        """GPU 사용률"""
        return (self.gpu_allocated_mb / self.gpu_total_mb) * 100 if self.gpu_total_mb > 0 else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'gpu_allocated_mb': self.gpu_allocated_mb,
            'gpu_free_mb': self.gpu_free_mb,
            'gpu_total_mb': self.gpu_total_mb,
            'ram_used_mb': self.ram_used_mb,
            'ram_available_mb': self.ram_available_mb,
            'swap_count': self.swap_count,
            'phase': self.phase,
            'timestamp': self.timestamp,
            'gpu_usage_percent': self.gpu_usage_percent
        }


@dataclass
class CircuitDecision:
    """Circuit 의사결정 결과"""
    decision: str  # 최종 결정
    confidence: float  # 신뢰도 (0-1)
    ethical_score: float  # 윤리 점수 (0-1)
    emotion_influence: float  # 감정 영향도
    regret_potential: float  # 후회 가능성
    scenarios: List[Dict[str, Any]]  # 시나리오 분석
    reasoning: str  # 추론 과정
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """검증"""
        for attr in ['confidence', 'ethical_score', 'emotion_influence', 'regret_potential']:
            value = getattr(self, attr)
            if not 0 <= value <= 1:
                raise ValueError(f"{attr} must be in [0, 1], got {value}")


# 직렬화 유틸리티 함수들
def serialize_tensor(tensor: torch.Tensor) -> bytes:
    """텐서를 바이트로 직렬화"""
    buffer = pickle.dumps({
        'data': tensor.cpu().numpy(),
        'dtype': str(tensor.dtype),
        'shape': list(tensor.shape)
    })
    return buffer


def deserialize_tensor(data: bytes, device: Optional[torch.device] = None) -> torch.Tensor:
    """바이트에서 텐서 역직렬화"""
    info = pickle.loads(data)
    tensor = torch.from_numpy(info['data'])
    
    # dtype 복원
    if 'float32' in info['dtype']:
        tensor = tensor.float()
    elif 'float64' in info['dtype']:
        tensor = tensor.double()
    elif 'int32' in info['dtype']:
        tensor = tensor.int()
    elif 'int64' in info['dtype']:
        tensor = tensor.long()
    
    if device:
        tensor = tensor.to(device)
    
    return tensor


# 배치 처리를 위한 유틸리티
def batch_messages(messages: List[TaskMessage], batch_size: int = 32) -> List[List[TaskMessage]]:
    """메시지를 배치로 그룹화"""
    batches = []
    for i in range(0, len(messages), batch_size):
        batch = messages[i:i + batch_size]
        batches.append(batch)
    return batches


def merge_results(results: List[ResultMessage]) -> ResultMessage:
    """여러 결과를 하나로 병합"""
    if not results:
        raise ValueError("Cannot merge empty results")
    
    # 첫 번째 결과를 기준으로
    first = results[0]
    
    # 모든 데이터 병합
    merged_data = {}
    for result in results:
        for key, value in result.data.items():
            if key not in merged_data:
                merged_data[key] = []
            merged_data[key].append(value)
    
    # 리스트를 텐서로 변환 (가능한 경우)
    for key, values in merged_data.items():
        if all(isinstance(v, (torch.Tensor, np.ndarray)) for v in values):
            if isinstance(values[0], torch.Tensor):
                merged_data[key] = torch.stack(values)
            else:
                merged_data[key] = np.stack(values)
    
    # 병합된 결과 생성
    return ResultMessage(
        module=first.module,
        task_type=first.task_type,
        data=merged_data,
        success=all(r.success for r in results),
        error='; '.join(r.error for r in results if r.error),
        processing_time=sum(r.processing_time or 0 for r in results),
        session_id=first.session_id,
        metadata={'batch_size': len(results)}
    )