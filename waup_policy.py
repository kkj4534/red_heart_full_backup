"""
WAUP (Workflow Aware Unload Policy) - 워크플로우 인지형 언로드 정책
GPT 제안 기반 구현
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Tuple
from enum import Enum
import json
import time

class WorkflowPhase(Enum):
    """워크플로우 단계 정의"""
    INGEST = "ingest"           # 입력/전처리
    EMBED = "embed"              # 임베딩/인덱싱
    EMO_DSP = "emo_dsp"         # 감정·DSP 융합
    BENTHAM = "bentham"          # 벤담 계산
    REGRET = "regret"           # 반사실·후회
    SURD = "surd"               # 인과/SURD
    INTEGRATE = "integrate"      # 통합/결정
    GEN = "gen"                 # 생성/후처리

@dataclass
class WAUPConfig:
    """WAUP 설정"""
    watermarks: Dict[str, float]
    weights: Dict[str, float]
    qos: Dict[str, float]
    phases: Dict[str, Dict]
    translator_policy: Dict
    protection: Dict
    
    @classmethod
    def get_default(cls):
        """GPT 제안 기본 설정"""
        return cls(
            watermarks={
                "low": 0.85,      # 저수위 - 언로드 중단
                "target": 0.90,   # 목표 수위
                "hard_cap": 0.92  # 상한 - 즉시 언로드
            },
            weights={
                "w_mem": 0.25,      # 메모리 크기 가중치
                "w_reload": 0.10,   # 재로드 비용 가중치
                "w_cold": 0.20,     # idle 시간 가중치
                "w_phase": 0.30,    # 단계 관련성 가중치
                "w_qos": 0.10,      # QoS 중요도 가중치
                "w_freq": 0.05      # 최근 사용 빈도 가중치
            },
            qos={
                # QoS 중요도 (0~1)
                "emotion_head": 0.85,
                "bentham_head": 0.85,
                "regret_head": 0.85,
                "surd_head": 0.85,
                "unified_backbone": 0.95,
                "translator": 0.50,
                "sentence_transformer": 0.70,
                "neural_emotion": 0.85,
                "neural_bentham": 0.85,
                "neural_regret": 0.85,
                "neural_surd": 0.85,
                "wrapper_advanced_emotion": 0.80,
                "wrapper_advanced_bentham": 0.80
            },
            phases={
                "INGEST": {
                    "pin": ["tokenizer", "router"],
                    "soft": ["light_prompt_llm"],
                    "drop_first": ["emotion_head", "regret_head", "surd_head", 
                                  "bentham_head", "translator", "neural_*", "wrapper_*"]
                },
                "EMBED": {
                    "pin": ["sentence_transformer"],
                    "soft": ["embed_alt", "index_utils"],
                    "drop_first": ["emotion_head", "regret_head", "surd_head",
                                  "bentham_head", "translator", "neural_*", "wrapper_*"]
                },
                "EMO_DSP": {
                    "pin": ["emotion_head", "neural_emotion", "wrapper_advanced_emotion"],
                    "soft": ["unified_backbone"],
                    "drop_first": ["surd_head", "bentham_head", "regret_head",
                                  "translator", "sentence_transformer"]
                },
                "BENTHAM": {
                    "pin": ["bentham_head", "neural_bentham", "wrapper_advanced_bentham"],
                    "soft": ["unified_backbone", "emotion_head"],
                    "drop_first": ["surd_head", "regret_head", "translator",
                                  "sentence_transformer", "neural_emotion"]
                },
                "REGRET": {
                    "pin": ["regret_head", "neural_regret"],
                    "soft": ["unified_backbone"],
                    "drop_first": ["surd_head", "translator", "sentence_transformer",
                                  "bentham_head", "emotion_head"]
                },
                "SURD": {
                    "pin": ["surd_head", "neural_surd"],
                    "soft": ["unified_backbone"],
                    "drop_first": ["regret_head", "translator", "sentence_transformer",
                                  "bentham_head", "emotion_head"]
                },
                "INTEGRATE": {
                    "pin": ["meta_integrator", "orchestrator", "unified_backbone"],
                    "soft": ["last_phase_cache"],
                    "drop_first": ["emotion_head", "regret_head", "surd_head",
                                  "bentham_head", "translator", "sentence_transformer",
                                  "neural_*", "wrapper_*"]
                },
                "GEN": {
                    "pin": ["gen_llm"],
                    "soft": ["light_rewriter"],
                    "drop_first": ["emotion_head", "regret_head", "surd_head",
                                  "bentham_head", "sentence_transformer", "translator",
                                  "neural_*", "wrapper_*", "unified_backbone"]
                }
            },
            translator_policy={
                "force_cpu": True,           # 항상 CPU에 유지
                "gpu_protect_below": 0.80    # GPU 80% 이하에서만 보호
            },
            protection={
                "phase_relevance_protect": 0.90,  # 단계 관련성 90% 이상 보호
                "qos_protect": 0.80,              # QoS 80% 이상 보호
                "fail_relax_after": 2             # 2회 실패 후 보호 완화
            }
        )


class WAUPManager:
    """WAUP 관리자"""
    
    def __init__(self, config: Optional[WAUPConfig] = None):
        self.config = config or WAUPConfig.get_default()
        self.current_phase = WorkflowPhase.INGEST
        self.fail_count = 0
        self.phase_start_time = time.time()
        self.model_stats = {}  # 모델별 통계
        
    def on_enter_phase(self, phase: WorkflowPhase, models: Dict):
        """단계 진입 시 우선순위 업데이트"""
        self.current_phase = phase
        self.phase_start_time = time.time()
        phase_str = phase.value.upper()
        
        if phase_str not in self.config.phases:
            return
            
        phase_policy = self.config.phases[phase_str]
        
        # 모든 모델 초기화
        for name, model in models.items():
            model.phase_relevance = 0.2  # 기본값
            model.pin_type = None
            
        # PIN 모델 설정 (관련성 1.0)
        for pattern in phase_policy.get("pin", []):
            for name, model in models.items():
                if self._match_pattern(name, pattern):
                    model.phase_relevance = 1.0
                    model.pin_type = "pin"
                    model.avoid_unload = True
                    
        # SOFT PIN 모델 설정 (관련성 0.75)
        for pattern in phase_policy.get("soft", []):
            for name, model in models.items():
                if self._match_pattern(name, pattern) and model.pin_type != "pin":
                    model.phase_relevance = 0.75
                    model.pin_type = "soft"
                    model.avoid_unload = False
                    
        # DROP_FIRST 모델 설정 (언로드 선호)
        for pattern in phase_policy.get("drop_first", []):
            for name, model in models.items():
                if self._match_pattern(name, pattern) and model.pin_type is None:
                    model.phase_relevance = 0.1
                    model.unload_preferred = True
                    
    def on_exit_phase(self, phase: WorkflowPhase, models: Dict):
        """단계 종료 시 우선순위 조정"""
        phase_str = phase.value.upper()
        
        if phase_str not in self.config.phases:
            return
            
        # soft_pin을 normal로 변경
        for name, model in models.items():
            if model.pin_type == "soft":
                model.phase_relevance = 0.2
                model.pin_type = None
                
            # 직전 단계 pin이었던 모델은 언로드 선호
            if model.pin_type == "pin":
                model.unload_preferred = True
                model.phase_relevance = 0.2
                model.pin_type = None
                
    def calculate_evict_score(self, model_name: str, model_info: any) -> float:
        """
        EvictScore 계산
        높을수록 언로드 우선순위가 높음
        """
        w = self.config.weights
        
        # 정규화 함수 (0~1)
        def norm(val, max_val):
            return min(1.0, val / max_val) if max_val > 0 else 0
            
        # 메모리 크기 (MB)
        mem_score = norm(getattr(model_info, 'size_mb', 100), 1000)
        
        # 재로드 비용 (추정)
        reload_cost = getattr(model_info, 'reload_cost', 0.5)
        
        # Idle 시간 (초)
        idle_time = time.time() - getattr(model_info, 'last_access', time.time())
        idle_score = norm(idle_time, 300)  # 5분 기준
        
        # 단계 관련성
        phase_relevance = getattr(model_info, 'phase_relevance', 0.2)
        
        # QoS 중요도
        qos = self.config.qos.get(model_name, 0.5)
        
        # 최근 사용 빈도
        access_count = getattr(model_info, 'access_count', 0)
        freq_score = 1.0 - norm(access_count, 100)  # 역수
        
        # EvictScore 계산
        score = (
            w["w_mem"] * mem_score +
            w["w_reload"] * reload_cost +
            w["w_cold"] * idle_score +
            w["w_phase"] * (1 - phase_relevance) +
            w["w_qos"] * (1 - qos) +
            w["w_freq"] * freq_score
        )
        
        # 언로드 선호 부스팅
        if getattr(model_info, 'unload_preferred', False):
            score += 0.15
            
        return score
        
    def select_unload_candidates(self, models: Dict, required_mb: float, 
                               gpu_usage: float) -> List[Tuple[str, float]]:
        """언로드 후보 선정"""
        candidates = []
        
        # GPU 사용률에 따른 전략 결정
        if gpu_usage > self.config.watermarks["hard_cap"]:
            # 긴급 모드 - 보호 완화
            protection_relaxed = True
        elif gpu_usage > self.config.watermarks["target"]:
            # 정상 언로드 모드
            protection_relaxed = (self.fail_count >= self.config.protection["fail_relax_after"])
        else:
            # 목표 달성 - 언로드 불필요
            return []
            
        # 후보 수집
        for name, model in models.items():
            # 보호 체크
            phase_rel = getattr(model, 'phase_relevance', 0.2)
            qos = self.config.qos.get(name, 0.5)
            
            protected = (
                phase_rel >= self.config.protection["phase_relevance_protect"] and
                qos >= self.config.protection["qos_protect"]
            )
            
            # 보호 완화 적용
            if protected and not protection_relaxed:
                continue
                
            # 번역기 특수 처리
            if "translator" in name.lower():
                if self.config.translator_policy["force_cpu"]:
                    continue  # 이미 CPU에 있음
                    
            # EvictScore 계산
            score = self.calculate_evict_score(name, model)
            size_mb = getattr(model, 'size_mb', 100)
            candidates.append((name, score, size_mb))
            
        # 점수 높은 순으로 정렬
        candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        # 필요한 만큼 선택
        selected = []
        freed_mb = 0
        
        for name, score, size_mb in candidates:
            if freed_mb >= required_mb:
                break
            selected.append((name, score))
            freed_mb += size_mb
            
        # 후보가 없으면 실패 카운트 증가
        if not selected and required_mb > 0:
            self.fail_count += 1
        else:
            self.fail_count = 0
            
        return selected
        
    def _match_pattern(self, name: str, pattern: str) -> bool:
        """패턴 매칭"""
        if pattern.endswith("*"):
            return name.startswith(pattern[:-1])
        return name == pattern
        
    def get_phase_models(self, phase: WorkflowPhase) -> Dict[str, Set[str]]:
        """단계별 필요 모델 반환"""
        phase_str = phase.value.upper()
        if phase_str not in self.config.phases:
            return {"pin": set(), "soft": set(), "drop_first": set()}
            
        policy = self.config.phases[phase_str]
        return {
            "pin": set(policy.get("pin", [])),
            "soft": set(policy.get("soft", [])),
            "drop_first": set(policy.get("drop_first", []))
        }