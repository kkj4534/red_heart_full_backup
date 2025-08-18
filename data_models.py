"""
데이터 모델 - 시스템에서 사용하는 주요 데이터 구조를 정의합니다.
"""

from enum import Enum
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
import datetime
import uuid
import json
import networkx as nx
import numpy as np

# 기본 열거형 타입 정의
class EmotionState(Enum):
    """기본 감정 상태"""
    JOY = 1
    TRUST = 2
    FEAR = 3
    SURPRISE = 4
    SADNESS = 5
    DISGUST = 6
    ANGER = 7
    ANTICIPATION = 8
    NEUTRAL = 0
    # 추가 감정 상태들
    ANXIETY = 9
    RELIEF = 10
    GUILT = 11
    SHAME = 12
    PRIDE = 13
    CONTEMPT = 14
    ENVY = 15
    GRATITUDE = 16

class EmotionIntensity(Enum):
    """감정 강도"""
    VERY_WEAK = 1
    WEAK = 2
    MODERATE = 3
    STRONG = 4
    VERY_STRONG = 5
    EXTREME = 6

class BenthamVariable(Enum):
    """벤담의 쾌락 계산법 변수"""
    INTENSITY = 1    # 강도
    DURATION = 2     # 지속성
    CERTAINTY = 3    # 확실성
    PROPINQUITY = 4  # 근접성
    FECUNDITY = 5    # 다산성
    PURITY = 6       # 순수성
    EXTENT = 7       # 범위
    # Bentham v2 추가 변수들
    EXTERNAL_COST = 8    # 외부비용 (E)
    REDISTRIBUTION_EFFECT = 9    # 재분배효과 (R)
    SELF_DAMAGE = 10     # 자아손상 (S)

class IntentionCategory(Enum):
    """윤리적 의도/목적 분류"""
    AVOIDING_HARM = 1        # 해악 회피
    SEEKING_GOOD = 2         # 선행 추구
    REALIZING_JUSTICE = 3    # 정의 실현
    LOYALTY_BELONGING = 4    # 충성/소속
    RESPECTING_AUTONOMY = 5  # 자율성 존중
    PROTECTING_VULNERABLE = 6 # 약자 보호
    HONESTY_TRUTH = 7        # 정직/진실성
    RESPONSIBILITY = 8       # 책임
    FAIRNESS = 9             # 공정성
    TOLERANCE_FORGIVENESS = 10 # 관용/용서

class SemanticLevel(Enum):
    """의미론적 표현 수준"""
    SURFACE = 1     # 표면적 특징
    ETHICAL = 2     # 윤리적 구조
    EMOTIONAL = 3   # 감정적 함의
    CAUSAL = 4      # 인과 관계

class Gender(Enum):
    """성별"""
    MALE = 1
    FEMALE = 2
    OTHER = 3
    UNKNOWN = 0

class AgeGroup(Enum):
    """연령대"""
    CHILD = 1        # 소년/소녀
    YOUNG_ADULT = 2  # 청년
    MIDDLE_AGED = 3  # 중년
    ELDER = 4        # 노년
    UNKNOWN = 0      # 알 수 없음

# 데이터 클래스 정의
@dataclass
class Biosignal:
    """생체신호 데이터"""
    eeg: Dict[str, float] = field(default_factory=dict)
    ecg: Dict[str, float] = field(default_factory=dict)
    gsr: Dict[str, float] = field(default_factory=dict)
    voice: Dict[str, float] = field(default_factory=dict)
    eye_tracking: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            "eeg": self.eeg,
            "ecg": self.ecg,
            "gsr": self.gsr,
            "voice": self.voice,
            "eye_tracking": self.eye_tracking,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Biosignal':
        """딕셔너리에서 생성"""
        # timestamp가 문자열이면 datetime으로 변환
        if isinstance(data.get("timestamp"), str):
            try:
                data["timestamp"] = datetime.datetime.fromisoformat(data["timestamp"])
            except ValueError:
                data["timestamp"] = datetime.datetime.now()
        
        return cls(
            eeg=data.get("eeg", {}),
            ecg=data.get("ecg", {}),
            gsr=data.get("gsr", {}),
            voice=data.get("voice", {}),
            eye_tracking=data.get("eye_tracking", {}),
            timestamp=data.get("timestamp", datetime.datetime.now())
        )

@dataclass
class EmotionData:
    """감정 상태 데이터"""
    primary_emotion: EmotionState = EmotionState.NEUTRAL
    intensity: EmotionIntensity = EmotionIntensity.MODERATE
    arousal: float = 0.0  # 각성도 (-1 ~ 1)
    valence: float = 0.0  # 정서가 (-1 ~ 1)
    dominance: float = 0.0  # 지배감 (-1 ~ 1) - PAD 모델 완성
    secondary_emotions: Dict[EmotionState, float] = field(default_factory=dict)
    confidence: float = 0.5  # 감정 분류 신뢰도
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    language: Optional[str] = None  # 언어 지원 추가
    processing_method: Optional[str] = None  # 처리 방법 추가
    metadata: Dict[str, Any] = field(default_factory=dict)  # 메타데이터 추가
    data_origin_tag: Optional['DataOriginTag'] = None  # Phase 1 개선: 데이터 출처 태그
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        result = {
            "primary_emotion": self.primary_emotion.name,
            "intensity": self.intensity.name,
            "arousal": self.arousal,
            "valence": self.valence,
            "dominance": self.dominance,
            "secondary_emotions": {e.name: v for e, v in self.secondary_emotions.items()},
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "language": self.language,
            "processing_method": self.processing_method
        }
        
        # 데이터 출처 태그 추가 (Phase 1 개선)
        if self.data_origin_tag:
            result["data_origin_tag"] = self.data_origin_tag.to_dict()
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'EmotionData':
        """딕셔너리에서 생성"""
        # Enum 타입 변환
        try:
            primary_emotion = EmotionState[data.get("primary_emotion", "NEUTRAL")]
        except (KeyError, TypeError):
            primary_emotion = EmotionState.NEUTRAL
            
        try:
            intensity = EmotionIntensity[data.get("intensity", "MODERATE")]
        except (KeyError, TypeError):
            intensity = EmotionIntensity.MODERATE
        
        # 보조 감정 변환
        secondary_emotions = {}
        for e_name, value in data.get("secondary_emotions", {}).items():
            try:
                emotion = EmotionState[e_name]
                secondary_emotions[emotion] = value
            except KeyError:
                pass
        
        # timestamp가 문자열이면 datetime으로 변환
        if isinstance(data.get("timestamp"), str):
            try:
                timestamp = datetime.datetime.fromisoformat(data["timestamp"])
            except ValueError:
                timestamp = datetime.datetime.now()
        else:
            timestamp = data.get("timestamp", datetime.datetime.now())
            
        # 데이터 출처 태그 처리 (Phase 1 개선)
        data_origin_tag = None
        if "data_origin_tag" in data:
            data_origin_tag = DataOriginTag.from_dict(data["data_origin_tag"])
        
        return cls(
            primary_emotion=primary_emotion,
            intensity=intensity,
            arousal=data.get("arousal", 0.0),
            valence=data.get("valence", 0.0),
            dominance=data.get("dominance", 0.0),
            secondary_emotions=secondary_emotions,
            confidence=data.get("confidence", 0.5),
            timestamp=timestamp,
            language=data.get("language"),
            processing_method=data.get("processing_method"),
            data_origin_tag=data_origin_tag
        )

@dataclass
class HedonicValues:
    """벤담의 쾌락 계산 값"""
    intensity: float = 0.0    # 강도
    duration: float = 0.0     # 지속성
    certainty: float = 0.0    # 확실성
    propinquity: float = 0.0  # 근접성
    fecundity: float = 0.0    # 다산성
    purity: float = 0.0       # 순수성
    extent: float = 0.0       # 범위
    # Bentham v2 추가 변수들
    external_cost: float = 0.0    # 외부비용 (E)
    redistribution_effect: float = 0.0    # 재분배효과 (R)
    self_damage: float = 0.0     # 자아손상 (S)
    hedonic_total: float = 0.0  # 종합 쾌락 값
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            "intensity": self.intensity,
            "duration": self.duration,
            "certainty": self.certainty,
            "propinquity": self.propinquity,
            "fecundity": self.fecundity,
            "purity": self.purity,
            "extent": self.extent,
            "external_cost": self.external_cost,
            "redistribution_effect": self.redistribution_effect,
            "self_damage": self.self_damage,
            "hedonic_total": self.hedonic_total
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'HedonicValues':
        """딕셔너리에서 생성"""
        return cls(
            intensity=data.get("intensity", 0.0),
            duration=data.get("duration", 0.0),
            certainty=data.get("certainty", 0.0),
            propinquity=data.get("propinquity", 0.0),
            fecundity=data.get("fecundity", 0.0),
            purity=data.get("purity", 0.0),
            extent=data.get("extent", 0.0),
            external_cost=data.get("external_cost", 0.0),
            redistribution_effect=data.get("redistribution_effect", 0.0),
            self_damage=data.get("self_damage", 0.0),
            hedonic_total=data.get("hedonic_total", 0.0)
        )

@dataclass
class EthicalSituation:
    """윤리적 상황 정보"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    options: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    source: str = "manual"  # 상황 출처 (manual, novel, scruples, vr)
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "context": self.context,
            "variables": self.variables,
            "options": self.options,
            "created_at": self.created_at.isoformat(),
            "source": self.source
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'EthicalSituation':
        """딕셔너리에서 생성"""
        # timestamp가 문자열이면 datetime으로 변환
        if isinstance(data.get("created_at"), str):
            try:
                created_at = datetime.datetime.fromisoformat(data["created_at"])
            except ValueError:
                created_at = datetime.datetime.now()
        else:
            created_at = data.get("created_at", datetime.datetime.now())
            
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            title=data.get("title", ""),
            description=data.get("description", ""),
            context=data.get("context", {}),
            variables=data.get("variables", {}),
            options=data.get("options", []),
            created_at=created_at,
            source=data.get("source", "manual")
        )

# 중복 클래스 제거됨 - 아래에 더 완성된 버전이 있음

@dataclass
class AdvancedCalculationContext:
    """고급 계산 컨텍스트"""
    scenario_text: str
    emotion_context: EmotionData
    stakeholders: List[str] = field(default_factory=list)
    cultural_context: str = ""
    urgency_level: float = 0.5

@dataclass
class Decision:
    """의사결정 정보"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    situation_id: str = ""
    choice: Any = None
    reasoning: str = ""
    confidence: float = 0.5
    reasoning_log: Dict[str, Any] = field(default_factory=dict)
    predicted_outcome: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            "id": self.id,
            "situation_id": self.situation_id,
            "choice": self.choice,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "reasoning_log": self.reasoning_log,
            "predicted_outcome": self.predicted_outcome,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Decision':
        """딕셔너리에서 생성"""
        # timestamp가 문자열이면 datetime으로 변환
        if isinstance(data.get("timestamp"), str):
            try:
                timestamp = datetime.datetime.fromisoformat(data["timestamp"])
            except ValueError:
                timestamp = datetime.datetime.now()
        else:
            timestamp = data.get("timestamp", datetime.datetime.now())
            
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            situation_id=data.get("situation_id", ""),
            choice=data.get("choice"),
            reasoning=data.get("reasoning", ""),
            confidence=data.get("confidence", 0.5),
            reasoning_log=data.get("reasoning_log", {}),
            predicted_outcome=data.get("predicted_outcome", {}),
            timestamp=timestamp
        )

@dataclass
class DecisionLog:
    """의사결정 전체 로그"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    situation: EthicalSituation = field(default_factory=EthicalSituation)
    biosignals: Biosignal = field(default_factory=Biosignal)
    emotions: EmotionData = field(default_factory=EmotionData)
    hedonic_values: HedonicValues = field(default_factory=HedonicValues)
    decision: Decision = field(default_factory=Decision)
    actual_outcome: Dict[str, Any] = field(default_factory=dict)
    has_regret: bool = False
    regret_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            "id": self.id,
            "situation": self.situation.to_dict(),
            "biosignals": self.biosignals.to_dict(),
            "emotions": self.emotions.to_dict(),
            "hedonic_values": self.hedonic_values.to_dict(),
            "decision": self.decision.to_dict(),
            "actual_outcome": self.actual_outcome,
            "has_regret": self.has_regret,
            "regret_data": self.regret_data,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DecisionLog':
        """딕셔너리에서 생성"""
        # 각 객체 변환
        situation = EthicalSituation.from_dict(data.get("situation", {}))
        biosignals = Biosignal.from_dict(data.get("biosignals", {}))
        emotions = EmotionData.from_dict(data.get("emotions", {}))
        hedonic_values = HedonicValues.from_dict(data.get("hedonic_values", {}))
        decision = Decision.from_dict(data.get("decision", {}))
        
        # timestamp가 문자열이면 datetime으로 변환
        if isinstance(data.get("timestamp"), str):
            try:
                timestamp = datetime.datetime.fromisoformat(data["timestamp"])
            except ValueError:
                timestamp = datetime.datetime.now()
        else:
            timestamp = data.get("timestamp", datetime.datetime.now())
            
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            situation=situation,
            biosignals=biosignals,
            emotions=emotions,
            hedonic_values=hedonic_values,
            decision=decision,
            actual_outcome=data.get("actual_outcome", {}),
            has_regret=data.get("has_regret", False),
            regret_data=data.get("regret_data", {}),
            timestamp=timestamp
        )

@dataclass
class SemanticRepresentationData:
    """의미론적 표현 데이터"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    levels: Dict[SemanticLevel, str] = field(default_factory=dict)
    embeddings: Dict[SemanticLevel, List[float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            "id": self.id,
            "text": self.text,
            "levels": {level.name: text for level, text in self.levels.items()},
            "embeddings": {level.name: emb for level, emb in self.embeddings.items()},
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SemanticRepresentationData':
        """딕셔너리에서 생성"""
        # Enum 타입 변환
        levels = {}
        for level_name, text in data.get("levels", {}).items():
            try:
                level = SemanticLevel[level_name]
                levels[level] = text
            except KeyError:
                pass
        
        # 임베딩 변환        
        embeddings = {}
        for level_name, emb in data.get("embeddings", {}).items():
            try:
                level = SemanticLevel[level_name]
                embeddings[level] = emb
            except KeyError:
                pass
        
        # timestamp가 문자열이면 datetime으로 변환
        if isinstance(data.get("timestamp"), str):
            try:
                timestamp = datetime.datetime.fromisoformat(data["timestamp"])
            except ValueError:
                timestamp = datetime.datetime.now()
        else:
            timestamp = data.get("timestamp", datetime.datetime.now())
            
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            text=data.get("text", ""),
            levels=levels,
            embeddings=embeddings,
            metadata=data.get("metadata", {}),
            timestamp=timestamp
        )

@dataclass
class SURDAnalysisResult:
    """SURD 인과 분석 결과"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    variables: Dict[str, float] = field(default_factory=dict)
    surd_components: Dict[str, Dict[str, float]] = field(default_factory=dict)
    causal_pathways: List[Dict[str, Any]] = field(default_factory=list)
    target_variable: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)


@dataclass
class EthicalContext:
    """윤리적 상황의 맥락"""
    scenario_description: str
    stakeholders: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    values: Dict[str, Any] = field(default_factory=dict)
    emotions: Dict[str, float] = field(default_factory=dict)
    uncertainty_level: float = 0.0


@dataclass
class BayesianScores:
    """베이지안 점수"""
    intensity: float = 0.5
    duration: float = 0.5
    certainty: float = 0.5
    propinquity: float = 0.5
    fecundity: float = 0.0
    purity: float = 0.5
    extent: float = 0.5
    expected_utility: float = 0.0
    variance: float = 0.0


@dataclass
class ExperienceData:
    """경험 데이터"""
    context: EthicalContext
    decision: str
    outcome_utility: float
    timestamp: float
    biosignals: Dict[str, Any] = field(default_factory=dict)
    semantic_analysis: Dict[str, Any] = field(default_factory=dict)
    causal_factors: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            "id": self.id,
            "variables": self.variables,
            "surd_components": self.surd_components,
            "causal_pathways": self.causal_pathways,
            "target_variable": self.target_variable,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SURDAnalysisResult':
        """딕셔너리에서 생성"""
        # timestamp가 문자열이면 datetime으로 변환
        if isinstance(data.get("timestamp"), str):
            try:
                timestamp = datetime.datetime.fromisoformat(data["timestamp"])
            except ValueError:
                timestamp = datetime.datetime.now()
        else:
            timestamp = data.get("timestamp", datetime.datetime.now())
            
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            variables=data.get("variables", {}),
            surd_components=data.get("surd_components", {}),
            causal_pathways=data.get("causal_pathways", []),
            target_variable=data.get("target_variable", ""),
            metadata=data.get("metadata", {}),
            timestamp=timestamp
        )

@dataclass
class Experience:
    """경험 데이터베이스 항목"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    decision_log_id: str = ""
    situation_summary: str = "" # 유지
    # Optional 유지, 압축 시 None 가능
    semantic_representation: Optional[SemanticRepresentationData] = field(default_factory=SemanticRepresentationData)
    hedonic_value: float = 0.0 # 유지
    # 압축 시 제거 또는 요약될 수 있음
    emotional_state: Optional[Dict[str, float]] = field(default_factory=dict)
    regret_intensity: float = 0.0 # 유지
    importance: float = 0.0 # 유지
    access_count: int = 0 # 유지
    last_accessed: datetime.datetime = field(default_factory=datetime.datetime.now)
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    is_compressed: bool = False
    category_id: Optional[str] = None

    # 핵심 상황 정보 필드 추가 (압축 시 채워짐)
    core_objects: Optional[List[str]] = None
    core_relationships: Optional[List[str]] = None
    core_variables: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        semantic_rep_dict = self.semantic_representation.to_dict() if self.semantic_representation else None
        return {
            "id": self.id,
            "decision_log_id": self.decision_log_id,
            "situation_summary": self.situation_summary,
            "semantic_representation": semantic_rep_dict, # 압축 시 None일 수 있음
            "hedonic_value": self.hedonic_value,
            "emotional_state": self.emotional_state, # 압축 시 None 또는 요약될 수 있음
            "regret_intensity": self.regret_intensity,
            "importance": self.importance,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat(),
            "created_at": self.created_at.isoformat(),
            "is_compressed": self.is_compressed,
            "category_id": self.category_id,
            # 핵심 정보 필드 추가
            "core_objects": self.core_objects,
            "core_relationships": self.core_relationships,
            "core_variables": self.core_variables
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Experience':
        """딕셔너리에서 생성"""
        semantic_rep_data = data.get("semantic_representation")
        semantic_rep = SemanticRepresentationData.from_dict(semantic_rep_data) if semantic_rep_data else None

        last_accessed = data.get("last_accessed", datetime.datetime.now())
        if isinstance(last_accessed, str):
            try: last_accessed = datetime.datetime.fromisoformat(last_accessed)
            except ValueError: last_accessed = datetime.datetime.now()

        created_at = data.get("created_at", datetime.datetime.now())
        if isinstance(created_at, str):
            try: created_at = datetime.datetime.fromisoformat(created_at)
            except ValueError: created_at = datetime.datetime.now()

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            decision_log_id=data.get("decision_log_id", ""),
            situation_summary=data.get("situation_summary", ""),
            semantic_representation=semantic_rep,
            hedonic_value=data.get("hedonic_value", 0.0),
            emotional_state=data.get("emotional_state"), # None일 수 있음
            regret_intensity=data.get("regret_intensity", 0.0),
            importance=data.get("importance", 0.0),
            access_count=data.get("access_count", 0),
            last_accessed=last_accessed,
            created_at=created_at,
            is_compressed=data.get("is_compressed", False),
            category_id=data.get("category_id"),
            # 핵심 정보 필드 로드
            core_objects=data.get("core_objects"),
            core_relationships=data.get("core_relationships"),
            core_variables=data.get("core_variables")
        )

@dataclass
class PerformanceMetrics:
    """시스템 성능 측정 메트릭"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    epoch: int = 0
    decision_count: int = 0
    accuracy: float = 0.0
    prediction_error: float = 0.0
    regret_ratio: float = 0.0
    hedonic_error: float = 0.0
    emotion_prediction_accuracy: float = 0.0
    learning_rate: float = 0.01
    avg_decision_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            "id": self.id,
            "epoch": self.epoch,
            "decision_count": self.decision_count,
            "accuracy": self.accuracy,
            "prediction_error": self.prediction_error,
            "regret_ratio": self.regret_ratio,
            "hedonic_error": self.hedonic_error,
            "emotion_prediction_accuracy": self.emotion_prediction_accuracy,
            "learning_rate": self.learning_rate,
            "avg_decision_time": self.avg_decision_time,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PerformanceMetrics':
        """딕셔너리에서 생성"""
        # timestamp가 문자열이면 datetime으로 변환
        if isinstance(data.get("timestamp"), str):
            try:
                timestamp = datetime.datetime.fromisoformat(data["timestamp"])
            except ValueError:
                timestamp = datetime.datetime.now()
        else:
            timestamp = data.get("timestamp", datetime.datetime.now())
            
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            epoch=data.get("epoch", 0),
            decision_count=data.get("decision_count", 0),
            accuracy=data.get("accuracy", 0.0),
            prediction_error=data.get("prediction_error", 0.0),
            regret_ratio=data.get("regret_ratio", 0.0),
            hedonic_error=data.get("hedonic_error", 0.0),
            emotion_prediction_accuracy=data.get("emotion_prediction_accuracy", 0.0),
            learning_rate=data.get("learning_rate", 0.01),
            avg_decision_time=data.get("avg_decision_time", 0.0),
            metadata=data.get("metadata", {}),
            timestamp=timestamp
        )

@dataclass
class DecisionScenario:
    """결정 시나리오"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    options: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)

@dataclass 
class IntegratedAnalysisResult:
    """통합 분석 결과"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    emotion_analysis: Dict[str, Any] = field(default_factory=dict)
    semantic_analysis: Dict[str, Any] = field(default_factory=dict)
    surd_analysis: Optional[SURDAnalysisResult] = None
    bentham_result: Optional[HedonicValues] = None
    confidence_score: float = 0.0
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)

@dataclass
class SystemStatus:
    """시스템 상태"""
    is_initialized: bool = False
    is_running: bool = False
    current_phase: str = "idle"
    active_modules: List[str] = field(default_factory=list)
    performance_metrics: Optional[PerformanceMetrics] = None
    performance_stats: Dict[str, Any] = field(default_factory=dict)
    last_update: datetime.datetime = field(default_factory=datetime.datetime.now)
    errors: List[str] = field(default_factory=list)
    device: str = "cpu"
    gpu_available: bool = False
    cache_size: int = 0

@dataclass
class EnhancedHedonicResult:
    """고급 벤담 계산 결과 - 실제 사용법 기반"""
    final_score: float
    base_score: float
    layer_contributions: List[Any] = field(default_factory=list)
    extreme_adjustment_applied: bool = False
    adjustment_factor: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 기존 필드들 (호환성)
    hedonic_values: HedonicValues = field(default_factory=HedonicValues)
    enhanced_layers: Dict[str, float] = field(default_factory=dict)
    ai_weights: Dict[str, float] = field(default_factory=dict)
    context_analysis: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.0
    enhancement_factors: Dict[str, float] = field(default_factory=dict)
    processing_time: float = 0.0
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    # 추가 호환성 필드들
    basic_hedonic: Optional['HedonicValues'] = None
    
    # 호환성을 위한 프로퍼티
    @property
    def confidence(self) -> float:
        return self.confidence_score

@dataclass
class AdvancedSemanticResult:
    """고급 의미 분석 결과"""
    text: str = ""
    language: str = "ko"
    analysis_depth: str = "full"
    surface_analysis: Dict[str, Any] = field(default_factory=dict)
    ethical_analysis: Dict[str, Any] = field(default_factory=dict)
    emotional_analysis: Dict[str, Any] = field(default_factory=dict)
    causal_analysis: Dict[str, Any] = field(default_factory=dict)
    feature_vector: Optional[Any] = None  # SemanticFeatureVector
    neural_encoding: Optional[Dict[str, Any]] = None
    cluster_info: Optional[Dict[str, Any]] = None
    network_info: Optional[Dict[str, Any]] = None
    semantic_representation: SemanticRepresentationData = field(default_factory=SemanticRepresentationData)
    similarity_scores: Dict[str, float] = field(default_factory=dict)
    embedding_vectors: Dict[str, List[float]] = field(default_factory=dict)
    analysis_metadata: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.0
    processing_time: float = 0.0
    timestamp: float = field(default_factory=lambda: datetime.datetime.now().timestamp())

@dataclass
class CausalGraph:
    """인과 그래프"""
    nodes: List[str] = field(default_factory=list)
    edges: List[Tuple[str, str, float]] = field(default_factory=list)
    node_attributes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    edge_attributes: Dict[Tuple[str, str], Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    def cascade(self, steps: int = 3, initial_activation: Optional[Dict[str, float]] = None) -> Dict[str, List[float]]:
        """
        Ripple-Simulator: 2-3차 효과 시뮬레이션 (마르코프 체인 기반)
        
        Args:
            steps: 시뮬레이션 단계 수 (기본값: 3)
            initial_activation: 초기 활성화 값 {node_id: activation_value}
            
        Returns:
            각 노드의 단계별 활성화 값: {node_id: [step0, step1, step2, ...]}
        """
        if not self.nodes or not self.edges:
            return {}
        
        # NetworkX 그래프 생성
        G = nx.DiGraph()
        
        # 노드 추가
        for node in self.nodes:
            node_attrs = self.node_attributes.get(node, {})
            G.add_node(node, **node_attrs)
        
        # 엣지 추가 (가중치 포함)
        for source, target, weight in self.edges:
            edge_attrs = self.edge_attributes.get((source, target), {})
            G.add_edge(source, target, weight=weight, **edge_attrs)
        
        # 초기 활성화 설정
        if initial_activation is None:
            # 모든 노드에 균등한 초기 활성화
            initial_activation = {node: 1.0 / len(self.nodes) for node in self.nodes}
        
        # 활성화 추적
        activation_history = {node: [] for node in self.nodes}
        current_activation = initial_activation.copy()
        
        # 시뮬레이션 수행
        for step in range(steps):
            # 현재 활성화 저장
            for node in self.nodes:
                activation_history[node].append(current_activation.get(node, 0.0))
            
            # 다음 단계 활성화 계산
            next_activation = {node: 0.0 for node in self.nodes}
            
            for node in self.nodes:
                current_value = current_activation.get(node, 0.0)
                
                # 감쇠 효과 (0.85 감쇠율)
                next_activation[node] = current_value * 0.85
                
                # 인접 노드로부터의 전파
                for predecessor in G.predecessors(node):
                    if predecessor in current_activation:
                        edge_data = G.get_edge_data(predecessor, node)
                        weight = edge_data.get('weight', 0.0)
                        
                        # 가중치 기반 전파 (sigmoid 정규화)
                        propagation = current_activation[predecessor] * abs(weight)
                        propagation = 1.0 / (1.0 + np.exp(-propagation))  # sigmoid
                        
                        # 방향성 고려 (양수는 증가, 음수는 감소)
                        if weight > 0:
                            next_activation[node] += propagation * 0.3
                        else:
                            next_activation[node] -= propagation * 0.3
                
                # 활성화 범위 제한 [0, 1]
                next_activation[node] = max(0.0, min(1.0, next_activation[node]))
            
            # 수렴 확인
            convergence_threshold = 1e-6
            if self._check_convergence(current_activation, next_activation, convergence_threshold):
                # 수렴 시 나머지 단계도 동일한 값으로 채움
                for remaining_step in range(step + 1, steps):
                    for node in self.nodes:
                        activation_history[node].append(next_activation.get(node, 0.0))
                break
            
            current_activation = next_activation
        
        # 메타데이터 업데이트
        self.metadata['last_cascade_result'] = {
            'steps': steps,
            'convergence_achieved': step < steps - 1,
            'final_activation': current_activation,
            'simulation_timestamp': datetime.datetime.now().isoformat()
        }
        
        return activation_history
    
    def _check_convergence(self, current: Dict[str, float], next_vals: Dict[str, float], threshold: float) -> bool:
        """수렴 확인 헬퍼 함수"""
        for node in current:
            if abs(current.get(node, 0.0) - next_vals.get(node, 0.0)) > threshold:
                return False
        return True
    
    def get_cascade_summary(self) -> Dict[str, Any]:
        """마지막 cascade 결과 요약"""
        if 'last_cascade_result' not in self.metadata:
            return {}
        
        result = self.metadata['last_cascade_result']
        
        # 2-3차 효과 강도 계산
        final_activation = result['final_activation']
        total_effect = sum(final_activation.values())
        
        # 노드별 영향도 순위
        node_impacts = sorted(final_activation.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'total_ripple_effect': total_effect,
            'convergence_achieved': result['convergence_achieved'],
            'top_affected_nodes': node_impacts[:5],
            'average_activation': total_effect / len(final_activation) if final_activation else 0.0,
            'simulation_timestamp': result['simulation_timestamp']
        }

@dataclass
class WeightLayerResult:
    """가중치 레이어 결과 - 실제 사용법 기반"""
    layer_name: str
    weight_factor: float
    contribution_score: float
    
    # 기존 필드들 (호환성)
    layer_weights: Dict[str, float] = field(default_factory=dict)
    combined_score: float = 0.0
    layer_contributions: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 추가 호환성 필드들
    weight_value: Optional[float] = None
    contribution: Optional[float] = None

@dataclass
class SemanticCluster:
    """의미 클러스터"""
    cluster_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    center_embedding: List[float] = field(default_factory=list)
    member_texts: List[str] = field(default_factory=list)
    similarity_threshold: float = 0.8
    cluster_label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CausalPath:
    """인과 경로"""
    path_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_variable: str = ""
    target_variable: str = ""
    intermediate_variables: List[str] = field(default_factory=list)
    path_strength: float = 0.0
    path_confidence: float = 0.0
    causal_mechanisms: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AdvancedCalculationContext:
    """고급 계산 컨텍스트"""
    scenario_text: str = ""
    stakeholders: List[str] = field(default_factory=list)
    cultural_context: Dict[str, Any] = field(default_factory=dict)
    temporal_factors: Dict[str, Any] = field(default_factory=dict)
    uncertainty_factors: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SemanticNetwork:
    """의미 네트워크"""
    network_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    nodes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    edges: List[Tuple[str, str, Dict[str, Any]]] = field(default_factory=list)
    clusters: List[SemanticCluster] = field(default_factory=list)
    global_metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)

@dataclass
class AdvancedSURDResult:
    """고급 SURD 분석 결과"""
    target_variable: str = ""
    synergy_score: float = 0.0
    uniqueness_score: float = 0.0
    redundancy_score: float = 0.0
    determinism_score: float = 0.0
    overall_score: float = 0.0
    causal_graph: Optional[Any] = None
    detailed_analysis: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.0
    processing_time: float = 0.0
    integration_info: Dict[str, Any] = field(default_factory=dict)
    input_variables: List[str] = field(default_factory=list)
    information_decomposition: Dict[str, Any] = field(default_factory=dict)
    neural_predictions: Optional[Dict[str, Any]] = None
    cascade_results: Optional[Dict[str, Any]] = None
    temporal_analysis: Optional[Dict[str, Any]] = None
    significance_results: Dict[str, Any] = field(default_factory=dict)
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    llm_interpretation: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0

@dataclass
class CausalRelation:
    """인과 관계"""
    relation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    cause: str = ""
    effect: str = ""
    strength: float = 0.0
    confidence: float = 0.0
    relation_type: str = "direct"
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CausalNetwork:
    """인과 네트워크"""
    network_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    relations: List[CausalRelation] = field(default_factory=list)
    nodes: List[str] = field(default_factory=list)
    adjacency_matrix: List[List[float]] = field(default_factory=list)
    network_metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)

@dataclass
class EthicalDimension:
    """윤리적 차원"""
    dimension_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    dimension_name: str = ""
    weight: float = 1.0
    ethical_values: Dict[str, float] = field(default_factory=dict)
    moral_principles: List[str] = field(default_factory=list)
    cultural_context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class InformationDecomposition:
    """정보 분해"""
    unique_information: Dict[str, float] = field(default_factory=dict)
    redundant_information: Dict[str, float] = field(default_factory=dict)  
    synergistic_information: Dict[str, float] = field(default_factory=dict)
    total_information: float = 0.0
    decomposition_method: str = "williams_beer"
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EmotionalProfile:
    """감정 프로필"""
    profile_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    dominant_emotions: Dict[str, float] = field(default_factory=dict)
    emotional_stability: float = 0.5
    empathy_level: float = 0.5
    emotional_intelligence: float = 0.5
    stress_indicators: Dict[str, float] = field(default_factory=dict)
    coping_mechanisms: List[str] = field(default_factory=list)
    emotional_patterns: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)


# =============================================================================
# Data Origin Tagging System (Phase 1 개선)
# =============================================================================

class DataOrigin(Enum):
    """데이터 출처 분류"""
    LITERATURE = "literature"          # 문학 작품 데이터
    REALTIME = "realtime"              # 실시간 사용자 입력
    SYNTHETIC = "synthetic"            # AI 생성 데이터
    ACADEMIC = "academic"              # 학술 연구 데이터
    HISTORICAL = "historical"         # 과거 경험 데이터
    BENCHMARK = "benchmark"           # 벤치마크 테스트 데이터
    UNKNOWN = "unknown"               # 출처 불명

@dataclass
class DataOriginTag:
    """데이터 출처 태그 정보"""
    origin: DataOrigin = DataOrigin.UNKNOWN
    source_detail: str = ""           # 상세 출처 (책 제목, 사용자 ID 등)
    processing_priority: int = 1      # 처리 우선순위 (1=최고, 5=최저)
    calibration_weight: float = 1.0   # Phase0 Calibrator 가중치
    reliability_score: float = 0.5    # 신뢰도 점수
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "origin": self.origin.value,
            "source_detail": self.source_detail,
            "processing_priority": self.processing_priority,
            "calibration_weight": self.calibration_weight,
            "reliability_score": self.reliability_score,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataOriginTag':
        """딕셔너리에서 생성"""
        origin = DataOrigin(data.get("origin", "unknown"))
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.datetime.fromisoformat(timestamp)
            except ValueError:
                timestamp = datetime.datetime.now()
        
        return cls(
            origin=origin,
            source_detail=data.get("source_detail", ""),
            processing_priority=data.get("processing_priority", 1),
            calibration_weight=data.get("calibration_weight", 1.0),
            reliability_score=data.get("reliability_score", 0.5),
            timestamp=timestamp,
            metadata=data.get("metadata", {})
        )

class DataOriginHelper:
    """데이터 출처 분류 및 가중치 관리 헬퍼"""
    
    # 출처별 기본 설정
    DEFAULT_CONFIGS = {
        DataOrigin.LITERATURE: {
            "processing_priority": 2,
            "calibration_weight": 0.8,  # 문학 데이터는 Phase0에서 약간 낮은 가중치
            "reliability_score": 0.9
        },
        DataOrigin.REALTIME: {
            "processing_priority": 1,
            "calibration_weight": 1.2,  # 실시간 데이터는 높은 가중치
            "reliability_score": 0.7
        },
        DataOrigin.SYNTHETIC: {
            "processing_priority": 3,
            "calibration_weight": 0.6,
            "reliability_score": 0.6
        },
        DataOrigin.ACADEMIC: {
            "processing_priority": 2,
            "calibration_weight": 1.0,
            "reliability_score": 0.95
        },
        DataOrigin.HISTORICAL: {
            "processing_priority": 4,
            "calibration_weight": 0.9,
            "reliability_score": 0.8
        },
        DataOrigin.BENCHMARK: {
            "processing_priority": 1,
            "calibration_weight": 1.0,
            "reliability_score": 1.0
        },
        DataOrigin.UNKNOWN: {
            "processing_priority": 5,
            "calibration_weight": 0.5,
            "reliability_score": 0.3
        }
    }
    
    @classmethod
    def create_tag(cls, origin: DataOrigin, source_detail: str = "", 
                   custom_config: Dict[str, Any] = None) -> DataOriginTag:
        """데이터 출처 태그 생성"""
        config = cls.DEFAULT_CONFIGS.get(origin, cls.DEFAULT_CONFIGS[DataOrigin.UNKNOWN])
        
        if custom_config:
            config = {**config, **custom_config}
            
        return DataOriginTag(
            origin=origin,
            source_detail=source_detail,
            processing_priority=config["processing_priority"],
            calibration_weight=config["calibration_weight"],
            reliability_score=config["reliability_score"]
        )
    
    @classmethod
    def detect_origin(cls, text: str, context: Dict[str, Any] = None) -> DataOrigin:
        """텍스트와 컨텍스트로부터 데이터 출처 자동 감지"""
        context = context or {}
        
        # 컨텍스트에서 명시적 출처 확인
        if "data_origin" in context:
            try:
                return DataOrigin(context["data_origin"])
            except ValueError:
                pass
                
        # 소스 정보로부터 추론
        source = context.get("source", "").lower()
        if any(keyword in source for keyword in ["novel", "book", "literature", "문학", "소설"]):
            return DataOrigin.LITERATURE
        elif any(keyword in source for keyword in ["user", "input", "realtime", "실시간", "사용자"]):
            return DataOrigin.REALTIME
        elif any(keyword in source for keyword in ["academic", "research", "paper", "학술", "연구"]):
            return DataOrigin.ACADEMIC
        elif any(keyword in source for keyword in ["synthetic", "generated", "ai", "생성", "인공"]):
            return DataOrigin.SYNTHETIC
        elif any(keyword in source for keyword in ["benchmark", "test", "벤치마크", "테스트"]):
            return DataOrigin.BENCHMARK
            
        # 파일 경로로부터 추론
        file_path = context.get("file_path", "").lower()
        if "novel" in file_path or "literature" in file_path:
            return DataOrigin.LITERATURE
        elif "test" in file_path or "benchmark" in file_path:
            return DataOrigin.BENCHMARK
            
        # 기본값
        return DataOrigin.UNKNOWN
    
    @classmethod
    def should_apply_calibration(cls, origin_tag: DataOriginTag) -> bool:
        """Phase0 Calibrator 적용 여부 결정"""
        # 문학 데이터에 대해서만 캘리브레이션 적용
        return origin_tag.origin == DataOrigin.LITERATURE
    
    @classmethod
    def get_processing_weight(cls, origin_tag: DataOriginTag, processing_phase: str = "general") -> float:
        """처리 단계별 가중치 반환"""
        base_weight = origin_tag.calibration_weight
        
        if processing_phase == "phase0_calibration":
            # Phase 0에서는 문학 데이터 가중치 조정
            if origin_tag.origin == DataOrigin.LITERATURE:
                return base_weight * 0.8  # 문학 데이터는 캘리브레이션에서 낮은 가중치
            else:
                return base_weight * 1.2  # 실시간 데이터는 높은 가중치
        elif processing_phase == "regret_learning":
            # 후회 학습에서는 실시간 데이터 우선
            if origin_tag.origin == DataOrigin.REALTIME:
                return base_weight * 1.3
            else:
                return base_weight * 0.9
        
        return base_weight


# =============================================================================
# EmotionVector Dimension Conversion Helpers (Phase 1 개선)
# =============================================================================

class EmotionDimensionHelper:
    """EmotionVector 3D↔6D 차원 변환 헬퍼 클래스"""
    
    # 6D 차원 (계층적 감정 시스템에서 사용)
    DIMENSIONS_6D = ['valence', 'arousal', 'dominance', 'certainty', 'surprise', 'anticipation']
    
    # 3D 차원 (VAD 모델 - Valence, Arousal, Dominance)
    DIMENSIONS_3D = ['valence', 'arousal', 'dominance']
    
    @classmethod
    def to_3d(cls, emotion_6d: Union[Dict[str, float], List[float]]) -> List[float]:
        """
        6D 감정 벡터를 3D VAD 모델로 변환
        
        Args:
            emotion_6d: 6차원 감정 벡터 (dict 또는 list)
        
        Returns:
            [valence, arousal, dominance] 3차원 리스트
        """
        if isinstance(emotion_6d, dict):
            # Dict 형태인 경우
            valence = emotion_6d.get('valence', 0.0)
            arousal = emotion_6d.get('arousal', 0.0)
            dominance = emotion_6d.get('dominance', 0.0)
            
            # certainty, surprise, anticipation을 VAD로 매핑
            certainty = emotion_6d.get('certainty', 0.0)
            surprise = emotion_6d.get('surprise', 0.0)
            anticipation = emotion_6d.get('anticipation', 0.0)
            
            # 추가 차원들을 VAD에 반영 (가중 평균)
            # certainty는 dominance에 영향
            dominance = dominance + (certainty * 0.3)
            
            # surprise는 arousal에 영향
            arousal = arousal + (surprise * 0.4)
            
            # anticipation은 valence에 영향  
            valence = valence + (anticipation * 0.2)
            
        elif isinstance(emotion_6d, list) and len(emotion_6d) >= 6:
            # List 형태인 경우 (순서: valence, arousal, dominance, certainty, surprise, anticipation)
            valence = emotion_6d[0] + (emotion_6d[5] * 0.2)  # valence + anticipation
            arousal = emotion_6d[1] + (emotion_6d[4] * 0.4)   # arousal + surprise  
            dominance = emotion_6d[2] + (emotion_6d[3] * 0.3) # dominance + certainty
            
        else:
            # 기본값
            valence, arousal, dominance = 0.0, 0.0, 0.0
            
        # 값 범위를 [-1, 1]로 클램핑
        valence = max(-1.0, min(1.0, valence))
        arousal = max(-1.0, min(1.0, arousal))
        dominance = max(-1.0, min(1.0, dominance))
        
        return [valence, arousal, dominance]
    
    @classmethod
    def to_6d(cls, emotion_3d: Union[Dict[str, float], List[float]], 
              preserve_extra: bool = True) -> Dict[str, float]:
        """
        3D VAD 모델을 6D 감정 벡터로 확장
        
        Args:
            emotion_3d: 3차원 감정 벡터 (dict 또는 list)
            preserve_extra: 추가 차원들을 추정할지 여부
        
        Returns:
            6차원 감정 벡터 딕셔너리
        """
        if isinstance(emotion_3d, dict):
            valence = emotion_3d.get('valence', 0.0)
            arousal = emotion_3d.get('arousal', 0.0) 
            dominance = emotion_3d.get('dominance', 0.0)
        elif isinstance(emotion_3d, list) and len(emotion_3d) >= 3:
            valence, arousal, dominance = emotion_3d[0], emotion_3d[1], emotion_3d[2]
        else:
            valence, arousal, dominance = 0.0, 0.0, 0.0
            
        result = {
            'valence': valence,
            'arousal': arousal,
            'dominance': dominance
        }
        
        if preserve_extra:
            # VAD 값으로부터 추가 차원들을 추정
            # certainty: dominance와 상관관계 (높은 dominance = 높은 certainty)
            result['certainty'] = abs(dominance) * 0.7
            
            # surprise: 높은 arousal에서 예상 가능 (긍정적 arousal = 낮은 surprise)
            result['surprise'] = max(0.0, arousal * 0.6) if arousal > 0 else abs(arousal) * 0.8
            
            # anticipation: 긍정적 valence와 상관관계
            result['anticipation'] = max(0.0, valence * 0.8)
        else:
            # 기본값으로 설정
            result.update({
                'certainty': 0.0,
                'surprise': 0.0, 
                'anticipation': 0.0
            })
            
        return result
    
    @classmethod
    def get_dimension_mode(cls, emotion_vector: Union[Dict, List]) -> str:
        """
        감정 벡터의 차원 모드를 감지
        
        Returns:
            "3d", "6d", "unknown"
        """
        if isinstance(emotion_vector, dict):
            keys = set(emotion_vector.keys())
            if keys.issuperset(set(cls.DIMENSIONS_6D)):
                return "6d"
            elif keys.issuperset(set(cls.DIMENSIONS_3D)):
                return "3d"
        elif isinstance(emotion_vector, list):
            if len(emotion_vector) >= 6:
                return "6d"
            elif len(emotion_vector) >= 3:
                return "3d"
                
        return "unknown"


# EmotionData 클래스에 헬퍼 메소드 추가를 위한 확장
def add_emotion_data_helpers():
    """EmotionData 클래스에 차원 변환 메소드를 동적으로 추가"""
    
    def to_3d(self) -> List[float]:
        """현재 EmotionData를 3D VAD 모델로 변환"""
        return [self.valence, self.arousal, self.dominance]
    
    def to_6d(self, preserve_extra: bool = True) -> Dict[str, float]:
        """현재 EmotionData를 6D 모델로 확장"""
        emotion_3d = {'valence': self.valence, 'arousal': self.arousal, 'dominance': self.dominance}
        return EmotionDimensionHelper.to_6d(emotion_3d, preserve_extra)
    
    def update_from_6d(self, emotion_6d: Dict[str, float]):
        """6D 감정 벡터로부터 EmotionData 업데이트"""
        vad_3d = EmotionDimensionHelper.to_3d(emotion_6d)
        self.valence = vad_3d[0]
        self.arousal = vad_3d[1] 
        self.dominance = vad_3d[2]
    
    # EmotionData 클래스에 메소드 추가
    EmotionData.to_3d = to_3d
    EmotionData.to_6d = to_6d
    EmotionData.update_from_6d = update_from_6d

# 모듈 로드 시 헬퍼 메소드 자동 추가
add_emotion_data_helpers()


# =============================================================================
# Advanced Rumbaugh Analyzer Data Models
# =============================================================================

@dataclass
class AdvancedStructuralResult:
    """고급 구조적 분석 결과 - Rumbaugh OMT 기반 구조 분석"""
    text: str = ""
    structural_objects: List[Any] = field(default_factory=list)
    structural_relations: List[Any] = field(default_factory=list)
    state_machines: List[Any] = field(default_factory=list)
    neural_analysis: Dict[str, Any] = field(default_factory=dict)
    graph_analysis: Dict[str, Any] = field(default_factory=dict)
    structural_patterns: List[Any] = field(default_factory=list)
    complexity_metrics: Dict[str, float] = field(default_factory=dict)
    dynamic_interactions: List['DynamicInteraction'] = field(default_factory=list)
    processing_time: float = 0.0
    confidence_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            "text": self.text,
            "structural_objects": [obj.to_dict() if hasattr(obj, 'to_dict') else str(obj) for obj in self.structural_objects],
            "structural_relations": [rel.to_dict() if hasattr(rel, 'to_dict') else str(rel) for rel in self.structural_relations],
            "state_machines": [sm.to_dict() if hasattr(sm, 'to_dict') else str(sm) for sm in self.state_machines],
            "neural_analysis": self.neural_analysis,
            "graph_analysis": self.graph_analysis,
            "structural_patterns": [pattern.to_dict() if hasattr(pattern, 'to_dict') else str(pattern) for pattern in self.structural_patterns],
            "complexity_metrics": self.complexity_metrics,
            "dynamic_interactions": [di.to_dict() if hasattr(di, 'to_dict') else str(di) for di in self.dynamic_interactions],
            "processing_time": self.processing_time,
            "confidence_score": self.confidence_score,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AdvancedStructuralResult':
        """딕셔너리에서 생성"""
        # timestamp 처리
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.datetime.fromisoformat(timestamp)
            except ValueError:
                timestamp = datetime.datetime.now()
        else:
            timestamp = timestamp or datetime.datetime.now()
            
        return cls(
            text=data.get("text", ""),
            structural_objects=data.get("structural_objects", []),
            structural_relations=data.get("structural_relations", []),
            state_machines=data.get("state_machines", []),
            neural_analysis=data.get("neural_analysis", {}),
            graph_analysis=data.get("graph_analysis", {}),
            structural_patterns=data.get("structural_patterns", []),
            complexity_metrics=data.get("complexity_metrics", {}),
            dynamic_interactions=data.get("dynamic_interactions", []),
            processing_time=data.get("processing_time", 0.0),
            confidence_score=data.get("confidence_score", 0.0),
            metadata=data.get("metadata", {}),
            timestamp=timestamp
        )

@dataclass
class DynamicInteraction:
    """동적 상호작용 - 객체 간 동적 관계 정의"""
    interaction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_object: str = ""
    target_object: str = ""
    interaction_type: str = ""
    trigger_conditions: List[str] = field(default_factory=list)
    expected_outcomes: List[str] = field(default_factory=list)
    interaction_strength: float = 0.0
    temporal_dynamics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            "interaction_id": self.interaction_id,
            "source_object": self.source_object,
            "target_object": self.target_object,
            "interaction_type": self.interaction_type,
            "trigger_conditions": self.trigger_conditions,
            "expected_outcomes": self.expected_outcomes,
            "interaction_strength": self.interaction_strength,
            "temporal_dynamics": self.temporal_dynamics,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DynamicInteraction':
        """딕셔너리에서 생성"""
        # timestamp 처리
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.datetime.fromisoformat(timestamp)
            except ValueError:
                timestamp = datetime.datetime.now()
        else:
            timestamp = timestamp or datetime.datetime.now()
            
        return cls(
            interaction_id=data.get("interaction_id", str(uuid.uuid4())),
            source_object=data.get("source_object", ""),
            target_object=data.get("target_object", ""),
            interaction_type=data.get("interaction_type", ""),
            trigger_conditions=data.get("trigger_conditions", []),
            expected_outcomes=data.get("expected_outcomes", []),
            interaction_strength=data.get("interaction_strength", 0.0),
            temporal_dynamics=data.get("temporal_dynamics", {}),
            timestamp=timestamp
        )

@dataclass
class StructuralElement:
    """구조적 요소 - 기본 구조 컴포넌트"""
    element_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    element_type: str = ""
    element_name: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            "element_id": self.element_id,
            "element_type": self.element_type,
            "element_name": self.element_name,
            "properties": self.properties,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StructuralElement':
        """딕셔너리에서 생성"""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.datetime.fromisoformat(timestamp)
            except ValueError:
                timestamp = datetime.datetime.now()
        else:
            timestamp = timestamp or datetime.datetime.now()
            
        return cls(
            element_id=data.get("element_id", str(uuid.uuid4())),
            element_type=data.get("element_type", ""),
            element_name=data.get("element_name", ""),
            properties=data.get("properties", {}),
            timestamp=timestamp
        )

@dataclass
class ObjectRelation:
    """객체 관계 - 레거시 호환성용"""
    relation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = ""
    target: str = ""
    relation_type: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            "relation_id": self.relation_id,
            "source": self.source,
            "target": self.target,
            "relation_type": self.relation_type,
            "properties": self.properties,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ObjectRelation':
        """딕셔너리에서 생성"""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.datetime.fromisoformat(timestamp)
            except ValueError:
                timestamp = datetime.datetime.now()
        else:
            timestamp = timestamp or datetime.datetime.now()
            
        return cls(
            relation_id=data.get("relation_id", str(uuid.uuid4())),
            source=data.get("source", ""),
            target=data.get("target", ""),
            relation_type=data.get("relation_type", ""),
            properties=data.get("properties", {}),
            timestamp=timestamp
        )

@dataclass
class StateMachine:
    """상태 기계 - 레거시 호환성용"""
    machine_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    machine_name: str = ""
    states: List[str] = field(default_factory=list)
    transitions: List[Dict[str, Any]] = field(default_factory=list)
    current_state: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            "machine_id": self.machine_id,
            "machine_name": self.machine_name,
            "states": self.states,
            "transitions": self.transitions,
            "current_state": self.current_state,
            "properties": self.properties,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StateMachine':
        """딕셔너리에서 생성"""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.datetime.fromisoformat(timestamp)
            except ValueError:
                timestamp = datetime.datetime.now()
        else:
            timestamp = timestamp or datetime.datetime.now()
            
        return cls(
            machine_id=data.get("machine_id", str(uuid.uuid4())),
            machine_name=data.get("machine_name", ""),
            states=data.get("states", []),
            transitions=data.get("transitions", []),
            current_state=data.get("current_state", ""),
            properties=data.get("properties", {}),
            timestamp=timestamp
        )

@dataclass
class StructuralComplexity:
    """구조적 복잡도 메트릭"""
    complexity_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    overall_complexity: float = 0.0
    structural_complexity: float = 0.0
    relational_complexity: float = 0.0
    dynamic_complexity: float = 0.0
    complexity_metrics: Dict[str, float] = field(default_factory=dict)
    calculation_method: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            "complexity_id": self.complexity_id,
            "overall_complexity": self.overall_complexity,
            "structural_complexity": self.structural_complexity,
            "relational_complexity": self.relational_complexity,
            "dynamic_complexity": self.dynamic_complexity,
            "complexity_metrics": self.complexity_metrics,
            "calculation_method": self.calculation_method,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StructuralComplexity':
        """딕셔너리에서 생성"""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.datetime.fromisoformat(timestamp)
            except ValueError:
                timestamp = datetime.datetime.now()
        else:
            timestamp = timestamp or datetime.datetime.now()
            
        return cls(
            complexity_id=data.get("complexity_id", str(uuid.uuid4())),
            overall_complexity=data.get("overall_complexity", 0.0),
            structural_complexity=data.get("structural_complexity", 0.0),
            relational_complexity=data.get("relational_complexity", 0.0),
            dynamic_complexity=data.get("dynamic_complexity", 0.0),
            complexity_metrics=data.get("complexity_metrics", {}),
            calculation_method=data.get("calculation_method", ""),
            metadata=data.get("metadata", {}),
            timestamp=timestamp
        )


# 유틸리티 함수들
def emotion_intensity_to_float(intensity) -> float:
    """EmotionIntensity Enum을 안전하게 float로 변환
    
    Args:
        intensity: EmotionIntensity enum 값 또는 다른 타입
        
    Returns:
        float: 0.0-1.0 범위의 정규화된 강도 값
    """
    if intensity is None:
        return 0.5  # 기본값
    
    # 이미 float인 경우
    if isinstance(intensity, (int, float)):
        return min(max(float(intensity) / 6.0, 0.0), 1.0)
    
    # EmotionIntensity Enum인 경우 
    if hasattr(intensity, 'value'):
        return min(max(float(intensity.value) / 6.0, 0.0), 1.0)
    
    # 문자열인 경우
    if isinstance(intensity, str):
        try:
            enum_val = EmotionIntensity[intensity.upper()]
            return min(max(float(enum_val.value) / 6.0, 0.0), 1.0)
        except (KeyError, ValueError):
            return 0.5  # 기본값
    
    # 그 외의 경우 기본값 반환
    return 0.5


def safe_float_operation(a, b, operation='add'):
    """안전한 float 연산 (EmotionIntensity와 float 간 연산 지원)
    
    Args:
        a: 첫 번째 피연산자
        b: 두 번째 피연산자  
        operation: 연산 종류 ('add', 'sub', 'mul', 'div')
        
    Returns:
        float: 연산 결과
    """
    # 안전하게 float로 변환
    a_val = emotion_intensity_to_float(a) if not isinstance(a, (int, float)) else float(a)
    b_val = emotion_intensity_to_float(b) if not isinstance(b, (int, float)) else float(b)
    
    if operation == 'add':
        return a_val + b_val
    elif operation == 'sub':
        return a_val - b_val
    elif operation == 'mul':
        return a_val * b_val
    elif operation == 'div':
        return a_val / b_val if b_val != 0 else a_val
    else:
        return a_val


# =============================================================================
# Self-Other-Community Empathy Learning Data Models (Phase 1)
# =============================================================================

@dataclass
class SelfReflectionData:
    """자기 성찰 데이터 - 자기 지향적 감정 상태 분석"""
    reflection_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    input_text: str = ""
    self_emotion_state: Dict[str, float] = field(default_factory=dict)
    self_confidence: float = 0.0
    internal_conflict_score: float = 0.0
    reflection_depth: str = "surface"  # surface, medium, deep
    personal_values_alignment: Dict[str, float] = field(default_factory=dict)
    cognitive_dissonance_score: float = 0.0
    reflection_triggers: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    data_origin_tag: Optional[DataOriginTag] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            "reflection_id": self.reflection_id,
            "input_text": self.input_text,
            "self_emotion_state": self.self_emotion_state,
            "self_confidence": self.self_confidence,
            "internal_conflict_score": self.internal_conflict_score,
            "reflection_depth": self.reflection_depth,
            "personal_values_alignment": self.personal_values_alignment,
            "cognitive_dissonance_score": self.cognitive_dissonance_score,
            "reflection_triggers": self.reflection_triggers,
            "processing_time": self.processing_time,
            "data_origin_tag": self.data_origin_tag.to_dict() if self.data_origin_tag else None,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SelfReflectionData':
        """딕셔너리에서 생성"""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.datetime.fromisoformat(timestamp)
            except ValueError:
                timestamp = datetime.datetime.now()
        else:
            timestamp = timestamp or datetime.datetime.now()
            
        data_origin_tag = None
        if data.get("data_origin_tag"):
            data_origin_tag = DataOriginTag.from_dict(data["data_origin_tag"])
            
        return cls(
            reflection_id=data.get("reflection_id", str(uuid.uuid4())),
            input_text=data.get("input_text", ""),
            self_emotion_state=data.get("self_emotion_state", {}),
            self_confidence=data.get("self_confidence", 0.0),
            internal_conflict_score=data.get("internal_conflict_score", 0.0),
            reflection_depth=data.get("reflection_depth", "surface"),
            personal_values_alignment=data.get("personal_values_alignment", {}),
            cognitive_dissonance_score=data.get("cognitive_dissonance_score", 0.0),
            reflection_triggers=data.get("reflection_triggers", []),
            processing_time=data.get("processing_time", 0.0),
            data_origin_tag=data_origin_tag,
            metadata=data.get("metadata", {}),
            timestamp=timestamp
        )

@dataclass
class EmpathySimulationData:
    """공감 시뮬레이션 데이터 - 타인 감정 예측 및 공감 시뮬레이션"""
    simulation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    input_text: str = ""
    target_perspective: str = ""  # 공감 대상 (타인, 집단 등)
    predicted_other_emotion: Dict[str, float] = field(default_factory=dict)
    empathy_accuracy: float = 0.0
    empathy_intensity: float = 0.0
    emotional_contagion_score: float = 0.0
    perspective_taking_score: float = 0.0
    compassion_level: float = 0.0
    theory_of_mind_score: float = 0.0
    simulation_confidence: float = 0.0
    mirror_neuron_activation: Dict[str, float] = field(default_factory=dict)
    processing_time: float = 0.0
    data_origin_tag: Optional[DataOriginTag] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            "simulation_id": self.simulation_id,
            "input_text": self.input_text,
            "target_perspective": self.target_perspective,
            "predicted_other_emotion": self.predicted_other_emotion,
            "empathy_accuracy": self.empathy_accuracy,
            "empathy_intensity": self.empathy_intensity,
            "emotional_contagion_score": self.emotional_contagion_score,
            "perspective_taking_score": self.perspective_taking_score,
            "compassion_level": self.compassion_level,
            "theory_of_mind_score": self.theory_of_mind_score,
            "simulation_confidence": self.simulation_confidence,
            "mirror_neuron_activation": self.mirror_neuron_activation,
            "processing_time": self.processing_time,
            "data_origin_tag": self.data_origin_tag.to_dict() if self.data_origin_tag else None,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'EmpathySimulationData':
        """딕셔너리에서 생성"""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.datetime.fromisoformat(timestamp)
            except ValueError:
                timestamp = datetime.datetime.now()
        else:
            timestamp = timestamp or datetime.datetime.now()
            
        data_origin_tag = None
        if data.get("data_origin_tag"):
            data_origin_tag = DataOriginTag.from_dict(data["data_origin_tag"])
            
        return cls(
            simulation_id=data.get("simulation_id", str(uuid.uuid4())),
            input_text=data.get("input_text", ""),
            target_perspective=data.get("target_perspective", ""),
            predicted_other_emotion=data.get("predicted_other_emotion", {}),
            empathy_accuracy=data.get("empathy_accuracy", 0.0),
            empathy_intensity=data.get("empathy_intensity", 0.0),
            emotional_contagion_score=data.get("emotional_contagion_score", 0.0),
            perspective_taking_score=data.get("perspective_taking_score", 0.0),
            compassion_level=data.get("compassion_level", 0.0),
            theory_of_mind_score=data.get("theory_of_mind_score", 0.0),
            simulation_confidence=data.get("simulation_confidence", 0.0),
            mirror_neuron_activation=data.get("mirror_neuron_activation", {}),
            processing_time=data.get("processing_time", 0.0),
            data_origin_tag=data_origin_tag,
            metadata=data.get("metadata", {}),
            timestamp=timestamp
        )

@dataclass
class CommunityContextData:
    """공동체 맥락 데이터 - 공동체 차원의 감정 및 문화적 맥락"""
    context_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    input_text: str = ""
    community_type: str = ""  # family, workplace, society, online_community 등
    cultural_background: str = ""  # korean, western, individualistic, collectivistic 등
    social_norms: Dict[str, float] = field(default_factory=dict)
    collective_emotion_state: Dict[str, float] = field(default_factory=dict)
    social_cohesion_score: float = 0.0
    cultural_alignment_score: float = 0.0
    group_influence_score: float = 0.0
    social_pressure_indicators: Dict[str, float] = field(default_factory=dict)
    community_values: Dict[str, float] = field(default_factory=dict)
    processing_time: float = 0.0
    data_origin_tag: Optional[DataOriginTag] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            "context_id": self.context_id,
            "input_text": self.input_text,
            "community_type": self.community_type,
            "cultural_background": self.cultural_background,
            "social_norms": self.social_norms,
            "collective_emotion_state": self.collective_emotion_state,
            "social_cohesion_score": self.social_cohesion_score,
            "cultural_alignment_score": self.cultural_alignment_score,
            "group_influence_score": self.group_influence_score,
            "social_pressure_indicators": self.social_pressure_indicators,
            "community_values": self.community_values,
            "processing_time": self.processing_time,
            "data_origin_tag": self.data_origin_tag.to_dict() if self.data_origin_tag else None,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CommunityContextData':
        """딕셔너리에서 생성"""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.datetime.fromisoformat(timestamp)
            except ValueError:
                timestamp = datetime.datetime.now()
        else:
            timestamp = timestamp or datetime.datetime.now()
            
        data_origin_tag = None
        if data.get("data_origin_tag"):
            data_origin_tag = DataOriginTag.from_dict(data["data_origin_tag"])
            
        return cls(
            context_id=data.get("context_id", str(uuid.uuid4())),
            input_text=data.get("input_text", ""),
            community_type=data.get("community_type", ""),
            cultural_background=data.get("cultural_background", ""),
            social_norms=data.get("social_norms", {}),
            collective_emotion_state=data.get("collective_emotion_state", {}),
            social_cohesion_score=data.get("social_cohesion_score", 0.0),
            cultural_alignment_score=data.get("cultural_alignment_score", 0.0),
            group_influence_score=data.get("group_influence_score", 0.0),
            social_pressure_indicators=data.get("social_pressure_indicators", {}),
            community_values=data.get("community_values", {}),
            processing_time=data.get("processing_time", 0.0),
            data_origin_tag=data_origin_tag,
            metadata=data.get("metadata", {}),
            timestamp=timestamp
        )

@dataclass
class MirrorNeuronData:
    """Mirror Neuron 시스템 데이터 - 2024년 Brain-Inspired AE-SNN 기반"""
    neuron_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    input_stimulus: str = ""
    observed_action: str = ""
    mirrored_response: Dict[str, float] = field(default_factory=dict)
    neural_activation_pattern: Dict[str, float] = field(default_factory=dict)
    self_other_differentiation: float = 0.0
    mirror_fidelity: float = 0.0
    empathic_resonance: float = 0.0
    pain_model_activation: Dict[str, float] = field(default_factory=dict)
    spiking_network_state: Dict[str, Any] = field(default_factory=dict)
    free_energy_prediction: float = 0.0
    processing_time: float = 0.0
    data_origin_tag: Optional[DataOriginTag] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            "neuron_id": self.neuron_id,
            "input_stimulus": self.input_stimulus,
            "observed_action": self.observed_action,
            "mirrored_response": self.mirrored_response,
            "neural_activation_pattern": self.neural_activation_pattern,
            "self_other_differentiation": self.self_other_differentiation,
            "mirror_fidelity": self.mirror_fidelity,
            "empathic_resonance": self.empathic_resonance,
            "pain_model_activation": self.pain_model_activation,
            "spiking_network_state": self.spiking_network_state,
            "free_energy_prediction": self.free_energy_prediction,
            "processing_time": self.processing_time,
            "data_origin_tag": self.data_origin_tag.to_dict() if self.data_origin_tag else None,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MirrorNeuronData':
        """딕셔너리에서 생성"""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.datetime.fromisoformat(timestamp)
            except ValueError:
                timestamp = datetime.datetime.now()
        else:
            timestamp = timestamp or datetime.datetime.now()
            
        data_origin_tag = None
        if data.get("data_origin_tag"):
            data_origin_tag = DataOriginTag.from_dict(data["data_origin_tag"])
            
        return cls(
            neuron_id=data.get("neuron_id", str(uuid.uuid4())),
            input_stimulus=data.get("input_stimulus", ""),
            observed_action=data.get("observed_action", ""),
            mirrored_response=data.get("mirrored_response", {}),
            neural_activation_pattern=data.get("neural_activation_pattern", {}),
            self_other_differentiation=data.get("self_other_differentiation", 0.0),
            mirror_fidelity=data.get("mirror_fidelity", 0.0),
            empathic_resonance=data.get("empathic_resonance", 0.0),
            pain_model_activation=data.get("pain_model_activation", {}),
            spiking_network_state=data.get("spiking_network_state", {}),
            free_energy_prediction=data.get("free_energy_prediction", 0.0),
            processing_time=data.get("processing_time", 0.0),
            data_origin_tag=data_origin_tag,
            metadata=data.get("metadata", {}),
            timestamp=timestamp
        )

@dataclass
class HierarchicalEmpathyResult:
    """계층적 공감 분석 결과 - Self-Other-Community 통합 결과"""
    result_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    input_text: str = ""
    language: str = "ko"
    
    # 계층적 공감 분석 결과
    self_reflection_result: Optional[SelfReflectionData] = None
    empathy_simulation_result: Optional[EmpathySimulationData] = None
    community_context_result: Optional[CommunityContextData] = None
    mirror_neuron_result: Optional[MirrorNeuronData] = None
    
    # 통합 점수들
    overall_empathy_score: float = 0.0
    self_awareness_score: float = 0.0
    other_understanding_score: float = 0.0
    community_integration_score: float = 0.0
    
    # 계층적 균형 분석
    self_other_balance: float = 0.0  # -1(자기중심) ~ 1(타인중심)
    individual_community_balance: float = 0.0  # -1(개인주의) ~ 1(집단주의)
    
    # 벤담 호환성 점수 (정규화됨)
    utilitarian_compatibility_score: float = 0.0
    
    # 신뢰도 및 메타데이터
    confidence_score: float = 0.0
    processing_time: float = 0.0
    analysis_depth: str = "full"  # surface, medium, full
    data_origin_tag: Optional[DataOriginTag] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            "result_id": self.result_id,
            "input_text": self.input_text,
            "language": self.language,
            "self_reflection_result": self.self_reflection_result.to_dict() if self.self_reflection_result else None,
            "empathy_simulation_result": self.empathy_simulation_result.to_dict() if self.empathy_simulation_result else None,
            "community_context_result": self.community_context_result.to_dict() if self.community_context_result else None,
            "mirror_neuron_result": self.mirror_neuron_result.to_dict() if self.mirror_neuron_result else None,
            "overall_empathy_score": self.overall_empathy_score,
            "self_awareness_score": self.self_awareness_score,
            "other_understanding_score": self.other_understanding_score,
            "community_integration_score": self.community_integration_score,
            "self_other_balance": self.self_other_balance,
            "individual_community_balance": self.individual_community_balance,
            "utilitarian_compatibility_score": self.utilitarian_compatibility_score,
            "confidence_score": self.confidence_score,
            "processing_time": self.processing_time,
            "analysis_depth": self.analysis_depth,
            "data_origin_tag": self.data_origin_tag.to_dict() if self.data_origin_tag else None,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'HierarchicalEmpathyResult':
        """딕셔너리에서 생성"""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.datetime.fromisoformat(timestamp)
            except ValueError:
                timestamp = datetime.datetime.now()
        else:
            timestamp = timestamp or datetime.datetime.now()
            
        # 하위 결과들 복원
        self_reflection_result = None
        if data.get("self_reflection_result"):
            self_reflection_result = SelfReflectionData.from_dict(data["self_reflection_result"])
            
        empathy_simulation_result = None
        if data.get("empathy_simulation_result"):
            empathy_simulation_result = EmpathySimulationData.from_dict(data["empathy_simulation_result"])
            
        community_context_result = None
        if data.get("community_context_result"):
            community_context_result = CommunityContextData.from_dict(data["community_context_result"])
            
        mirror_neuron_result = None
        if data.get("mirror_neuron_result"):
            mirror_neuron_result = MirrorNeuronData.from_dict(data["mirror_neuron_result"])
            
        data_origin_tag = None
        if data.get("data_origin_tag"):
            data_origin_tag = DataOriginTag.from_dict(data["data_origin_tag"])
            
        return cls(
            result_id=data.get("result_id", str(uuid.uuid4())),
            input_text=data.get("input_text", ""),
            language=data.get("language", "ko"),
            self_reflection_result=self_reflection_result,
            empathy_simulation_result=empathy_simulation_result,
            community_context_result=community_context_result,
            mirror_neuron_result=mirror_neuron_result,
            overall_empathy_score=data.get("overall_empathy_score", 0.0),
            self_awareness_score=data.get("self_awareness_score", 0.0),
            other_understanding_score=data.get("other_understanding_score", 0.0),
            community_integration_score=data.get("community_integration_score", 0.0),
            self_other_balance=data.get("self_other_balance", 0.0),
            individual_community_balance=data.get("individual_community_balance", 0.0),
            utilitarian_compatibility_score=data.get("utilitarian_compatibility_score", 0.0),
            confidence_score=data.get("confidence_score", 0.0),
            processing_time=data.get("processing_time", 0.0),
            analysis_depth=data.get("analysis_depth", "full"),
            data_origin_tag=data_origin_tag,
            metadata=data.get("metadata", {}),
            timestamp=timestamp
        )