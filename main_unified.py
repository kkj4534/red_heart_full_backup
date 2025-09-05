#!/usr/bin/env python3
"""
Red Heart AI - 통합 추론 시스템 (Unified Inference System)
50 epoch으로 학습된 730M 모델 전체 활용
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import asyncio
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import glob
import warnings
warnings.filterwarnings('ignore')

# 정밀 감정-벤담 매핑 시스템 - 필수 의존성
from semantic_emotion_bentham_mapper import (
    SemanticEmotionBenthamMapper,
    NeuralEmotionBenthamAdapter,
    create_precision_mapper
)

# 3뷰 시나리오 시스템
from three_view_scenario_system import (
    ThreeViewScenarioSystem,
    ScenarioType,
    ThreeViewAnalysisResult
)

# 다원적 윤리 체계
from deep_multi_dimensional_ethics_system import (
    DeepMultiDimensionalEthicsSystem,
    EthicsSchool,
    EthicalDilemma,
    StakeholderPerspective,
    UtilitarianEngine,
    DeontologicalEngine,
    VirtueEthicsEngine,
    CareEthicsEngine,
    JusticeTheoryEngine  # MD 문서 B안: 5번째 윤리 시스템
)

# 메모리 스왑 매니저
from memory_swap_manager import SystemSwapManager, SystemType

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'training'))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('RedHeart.MainUnified')

# GPU/CPU 디바이스 설정
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"🖥️ 사용 디바이스: {DEVICE}")
if DEVICE.type == 'cuda':
    logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")


class MemoryMode(Enum):
    """메모리 모드 - MD 문서 사양 준수"""
    LIGHT = "light"        # 230M - 빠른 프로토타이핑
    MEDIUM = "medium"      # 600M - 균형잡힌 일반 사용 (재설계됨)
    HEAVY = "heavy"        # 970M - 심층 분석 (동적 스왑)
    MCP = "mcp"           # MCP 서버 모드 (HEAVY 기반)
    
    # 기존 호환성을 위한 별칭
    MINIMAL = "minimal"    # 90M (구버전)
    NORMAL = "normal"      # 400M (구버전)
    ULTRA = "ultra"        # 842M (구버전)
    EXTREME = "extreme"    # 922M (구버전)


def auto_select_memory_mode(gpu_memory_mb: int = None, batch_size: int = 1) -> MemoryMode:
    """GPU 메모리에 따른 자동 모드 선택"""
    if gpu_memory_mb is None and torch.cuda.is_available():
        gpu_memory_mb = torch.cuda.get_device_properties(0).total_memory // (1024**2)
    
    effective_memory = gpu_memory_mb - (batch_size * 500) if gpu_memory_mb else 4000
    
    if effective_memory < 3000:
        return MemoryMode.MINIMAL
    elif effective_memory < 4000:
        return MemoryMode.LIGHT
    elif effective_memory < 5000:
        return MemoryMode.NORMAL
    elif effective_memory < 6000:
        return MemoryMode.HEAVY
    elif effective_memory < 7000:
        return MemoryMode.ULTRA
    else:
        return MemoryMode.EXTREME


@dataclass
class InferenceConfig:
    """통합 추론 설정"""
    # 체크포인트 설정 - 에폭 번호로 자동 검색
    checkpoint_epoch: int = 50  # 최적 에폭 (sweet spot 분석 결과)
    checkpoint_path: Optional[str] = None  # 직접 경로 지정 (우선순위)
    checkpoint_dir: str = "training/checkpoints_final"
    
    # 모델 설정
    device: str = str(DEVICE)
    batch_size: int = 4
    max_seq_length: int = 512
    
    # 메모리 모드 설정
    memory_mode: MemoryMode = MemoryMode.NORMAL  # 기본값
    auto_memory_mode: bool = True  # 자동 모드 선택
    
    # 모듈 활성화 플래그 (메모리 모드에 따라 자동 조정)
    use_neural_analyzers: bool = True  # 368M (HEAVY+)
    use_advanced_wrappers: bool = True  # 112M (ULTRA+)
    use_dsp_simulator: bool = True  # 14M (NORMAL+)
    use_kalman_filter: bool = True  # 2.3M (NORMAL+)
    use_phase_networks: bool = True  # 4.3M (NORMAL+)
    use_regret_circuit: bool = True  # Regret Head 활성화
    
    # 새로운 모듈 플래그 (EXTREME 모드에서 활성화)
    use_meta_integration: bool = True  # 40M - 메타 통합 시스템
    use_counterfactual_reasoning: bool = True  # 15M - 반사실 추론
    use_advanced_regret_learning: bool = True  # 20M - 고급 후회 학습
    use_workflow_memory_manager: bool = True  # 5M - 워크플로우 메모리 관리
    use_temporal_propagation: bool = True  # 시계열 전파 분석
    use_experience_database: bool = True  # 경험 DB 연동
    use_emotion_hierarchy: bool = True  # 계층적 감정 처리 (공동체>타자>자아)
    use_three_view_scenario: bool = True  # 3뷰 시나리오 시스템 (낙관/중도/비관)
    use_multi_ethics_system: bool = True  # 다원적 윤리 체계 (5개 학파)
    
    # LLM 통합 옵션
    llm_mode: str = "none"  # "none", "local", "claude", "mcp"
    llm_model_path: str = "llm_module/HelpingAI2-9B.Q4_K_M.gguf"
    
    # 임베딩 설정
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  # 캐시된 모델 사용
    use_cached_embeddings: bool = True
    
    # 번역 설정
    use_translator: bool = False  # 한국어 감지 시 활성화
    translator_model: str = "facebook/m2m100_418M"
    
    # 성능 설정
    enable_xai: bool = False
    enable_monitoring: bool = True
    cache_size: int = 100
    
    # 로깅
    verbose: bool = True
    debug: bool = False


class UnifiedInferenceSystem:
    """통합 추론 시스템 - 730M~922M 모델 전체 활용"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.logger = logger
        
        # 메모리 모드 자동 설정
        if config.auto_memory_mode:
            self.config.memory_mode = auto_select_memory_mode(batch_size=config.batch_size)
            self.logger.info(f"🎛️ 자동 메모리 모드 선택: {self.config.memory_mode.value}")
            self._adjust_modules_by_memory_mode()
        
        # 기본 모델 컴포넌트들 (730M)
        self.unified_model = None
        self.neural_analyzers = None
        self.advanced_wrappers = None
        self.dsp_simulator = None
        self.kalman_filter = None
        self.phase_networks = None
        
        # 새로운 통합 모듈들 (192M) - EXTREME 모드
        self.meta_integration = None  # 40M
        self.counterfactual_reasoning = None  # 15M
        self.advanced_regret_learning = None  # 20M
        self.workflow_memory_manager = None  # 5M
        self.temporal_propagator = None  # 시계열 전파
        self.experience_database = None  # 경험 DB
        self.emotion_hierarchy_processor = None  # 계층적 감정
        
        # 정밀 감정→벤담 매퍼
        self.emotion_bentham_mapper = None  # 의미론적 매퍼
        self.neural_emotion_adapter = None  # 신경망 어댑터 (EXTREME 모드)
        
        # 3뷰 시나리오 시스템
        self.three_view_system = None
        
        # 다원적 윤리 체계
        self.multi_ethics_system = None
        self.ethics_engines = {}  # 개별 윤리 엔진들
        
        # 유휴 학습 시스템
        self.idle_learner = None
        
        # LLM 관련
        self.llm_engine = None
        self.translator = None
        
        # 메모리 스왑 매니저
        self.swap_manager = None
        
        # 체크포인트 매니저
        self.checkpoint_manager = None
        
        # 캐시
        self.cache = {}
        
        # 통계
        self.stats = {
            'total_requests': 0,
            'successful': 0,
            'failed': 0,
            'avg_time': 0.0
        }
        
        self.logger.info("✨ Red Heart AI 통합 추론 시스템 초기화")
    
    def _adjust_modules_by_memory_mode(self):
        """메모리 모드에 따른 모듈 활성화 조정 - MD 문서 사양 준수"""
        mode = self.config.memory_mode
        
        if mode == MemoryMode.LIGHT:
            # LIGHT 모드 (230M) - MD 문서 사양
            # 기본 컴포넌트만
            self.config.use_neural_analyzers = False
            self.config.use_advanced_wrappers = False
            self.config.use_dsp_simulator = False
            self.config.use_kalman_filter = False
            self.config.use_phase_networks = False
            self.config.use_meta_integration = False
            self.config.use_counterfactual_reasoning = False
            self.config.use_advanced_regret_learning = False
            self.config.use_workflow_memory_manager = True  # 메모리 관리는 LIGHT에서도 중요
            self.config.use_temporal_propagation = False
            self.config.use_experience_database = False
            self.config.use_emotion_hierarchy = False
            self.config.use_three_view_scenario = False
            self.config.use_multi_ethics_system = False  # 공리주의만
            
        elif mode == MemoryMode.MEDIUM:
            # MEDIUM 모드 (600M) - MD 문서 재설계 사양
            # Neural Analyzers 선별 (194M)
            self.config.use_neural_analyzers = True  # emotion/bentham만
            self.config.neural_analyzers_subset = ['emotion', 'bentham']  # 부분 로드
            # Advanced Wrappers 선별 (56M)
            self.config.use_advanced_wrappers = True  # emotion/bentham만
            self.config.advanced_wrappers_subset = ['advanced_emotion', 'advanced_bentham']  # 부분 로드 - 정확한 키 사용
            # DSP/Kalman (14M)
            self.config.use_dsp_simulator = True
            self.config.use_kalman_filter = True
            # LLM을 초기에 RAM으로 보내기
            self.config.llm_start_in_ram = True
            # 핵심 통합 모듈 (80M)
            self.config.use_three_view_scenario = True  # 20M
            self.config.use_multi_ethics_system = True  # 30M (3개 학파만)
            self.config.use_temporal_propagation = True  # 15M
            self.config.use_meta_integration = True  # 15M (기본 버전)
            # 선택지 생성/평가를 위한 모듈 활성화
            self.config.use_phase_networks = True  # Phase Networks 활성화 (타자-자아-공동체 감정)
            self.config.use_counterfactual_reasoning = True  # 반사실 추론 활성화 (선택지 생성)
            self.config.use_advanced_regret_learning = True  # 대안 제시 활성화 (suggest_alternatives)
            self.config.use_workflow_memory_manager = True  # 메모리 관리 활성화
            self.config.use_experience_database = False  # 경험 DB 비활성화 (MVP 테스트용)
            self.config.use_emotion_hierarchy = True  # 계층적 감정 처리
            # API 모드에서는 번역기 불필요 (API가 한국어 직접 처리)
            self.config.use_translator = False  # 초기엔 비활성화, 로컬 모드에서만 활성화
            
        elif mode == MemoryMode.HEAVY or mode == MemoryMode.MCP:
            # HEAVY 모드 (970M) - MD 문서 사양
            # 모든 모듈 활성화
            self.config.use_neural_analyzers = True
            self.config.use_advanced_wrappers = True
            self.config.use_dsp_simulator = True
            self.config.use_kalman_filter = True
            self.config.use_phase_networks = True
            self.config.use_meta_integration = True
            self.config.use_counterfactual_reasoning = True
            self.config.use_advanced_regret_learning = True
            self.config.use_three_view_scenario = True
            self.config.use_multi_ethics_system = True  # 5개 학파
            self.config.use_temporal_propagation = True
            self.config.use_workflow_memory_manager = True
            self.config.use_experience_database = True
            self.config.use_emotion_hierarchy = True
            
        # 구버전 호환성
        elif mode == MemoryMode.MINIMAL:
            # 최소 모드 (90M)
            self._set_all_modules_false()
        elif mode == MemoryMode.NORMAL:
            # 구버전 NORMAL (400M)
            self.config.use_dsp_simulator = True
            self.config.use_kalman_filter = True
        elif mode == MemoryMode.ULTRA:
            # 구버전 ULTRA (842M)
            self.config.use_neural_analyzers = True
            self.config.use_advanced_wrappers = True
        elif mode == MemoryMode.EXTREME:
            # 구버전 EXTREME (922M)
            self._set_all_modules_true()
    
    def _set_all_modules_false(self):
        """모든 모듈 비활성화"""
        for key in self.config.__dict__:
            if key.startswith('use_'):
                setattr(self.config, key, False)
    
    def _set_all_modules_true(self):
        """모든 모듈 활성화"""
        for key in self.config.__dict__:
            if key.startswith('use_'):
                setattr(self.config, key, True)
    
    def _detect_memory_mode(self):
        """GPU 메모리를 기반으로 자동으로 메모리 모드 선택"""
        try:
            if torch.cuda.is_available():
                # GPU 메모리 체크
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                
                if gpu_memory >= 24:
                    self.config.memory_mode = MemoryMode.HEAVY
                    self.logger.info(f"🎛️ 자동 메모리 모드 선택: heavy (GPU: {gpu_memory:.1f}GB)")
                elif gpu_memory >= 12:
                    self.config.memory_mode = MemoryMode.MEDIUM
                    self.logger.info(f"🎛️ 자동 메모리 모드 선택: medium (GPU: {gpu_memory:.1f}GB)")
                elif gpu_memory >= 8:
                    # 8GB GPU는 MEDIUM 모드로 설정 (동적 스왑 활용)
                    self.config.memory_mode = MemoryMode.MEDIUM
                    self.logger.info(f"🎛️ 자동 메모리 모드 선택: medium (GPU: {gpu_memory:.1f}GB, 동적 스왑 활용)")
                else:
                    self.config.memory_mode = MemoryMode.LIGHT
                    self.logger.info(f"🎛️ 자동 메모리 모드 선택: light (GPU: {gpu_memory:.1f}GB)")
            else:
                # CPU 전용
                self.config.memory_mode = MemoryMode.LIGHT
                self.logger.info("🎛️ 자동 메모리 모드 선택: light (CPU 전용)")
        except Exception as e:
            self.logger.warning(f"⚠️ 메모리 모드 자동 감지 실패: {e}")
            self.config.memory_mode = MemoryMode.LIGHT
            self.logger.info("🎛️ 기본 메모리 모드 선택: light")
        
    async def initialize(self):
        """시스템 초기화"""
        try:
            self.logger.info("=" * 70)
            self.logger.info("🚀 시스템 초기화 시작...")
            self.logger.info("=" * 70)
            
            # 0. 메모리 모드 설정
            if self.config.auto_memory_mode:
                # 자동 모드 감지
                self._detect_memory_mode()
            else:
                # 지정된 모드 사용
                self.logger.info(f"📦 지정된 메모리 모드: {self.config.memory_mode.value}")
            
            # 모드에 따른 모듈 플래그 조정
            self._adjust_modules_by_memory_mode()
            
            # DSM 초기화 (UnifiedModel 로드 전에! Claude는 제외)
            if self.config.llm_mode != "none" and self.config.llm_mode != "claude":
                await self._init_dsm_early()
            
            # 1. UnifiedModel 로드
            await self._load_unified_model()
            
            # 2. 번역기 로드 (Advanced Wrappers가 필수로 요구함)
            # API 모드에서도 translator 객체는 필요 (실제 번역은 use_translator로 제어)
            await self._load_translator()
            
            # 3. Neural Analyzers 로드 (368M)
            if self.config.use_neural_analyzers:
                await self._load_neural_analyzers()
            
            # 4. Advanced Wrappers 로드 (112M) - translator 필수
            if self.config.use_advanced_wrappers:
                await self._load_advanced_wrappers()
            
            # 5. DSP & Kalman Filter 로드
            if self.config.use_dsp_simulator:
                await self._load_dsp_components()
            
            # 6. Phase Networks 로드
            if self.config.use_phase_networks:
                await self._load_phase_networks()
            
            # 7. 새로운 통합 모듈들 (EXTREME 모드)
            if self.config.use_workflow_memory_manager:
                await self._load_workflow_memory_manager()
            
            if self.config.use_meta_integration:
                await self._load_meta_integration()
            
            if self.config.use_counterfactual_reasoning:
                await self._load_counterfactual_reasoning()
            
            if self.config.use_advanced_regret_learning:
                await self._load_advanced_regret_learning()
            
            if self.config.use_temporal_propagation:
                await self._load_temporal_propagation()
            
            if self.config.use_experience_database:
                await self._load_experience_database()
            
            if self.config.use_emotion_hierarchy:
                await self._load_emotion_hierarchy()
            
            # 8. 정밀 감정→벤담 매퍼 로드
            await self._load_precision_mapper()
            
            # 9. 3뷰 시나리오 시스템 로드
            if self.config.use_three_view_scenario:
                await self._load_three_view_scenario_system()
            
            # 10. 다원적 윤리 체계 로드
            if self.config.use_multi_ethics_system:
                await self._load_multi_ethics_system()
            
            # 11. LLM 통합 (선택적)
            if self.config.llm_mode != "none":
                await self._load_llm_integration()
            
            # 12. 유휴 학습 시스템 (MD 문서: 프로덕션 레벨까지 주석 처리)
            # TODO: 프로덕션 레벨에서 활성화
            # if self.config.memory_mode.value in ['heavy', 'ultra', 'extreme']:
            #     await self._load_idle_learner()
            
            self.logger.info("=" * 70)
            self.logger.info("✅ 시스템 초기화 완료!")
            self._print_system_status()
            self.logger.info("=" * 70)
            
        except Exception as e:
            self.logger.error(f"❌ 초기화 실패: {e}")
            raise
    
    async def _load_unified_model(self):
        """UnifiedModel 및 체크포인트 로드"""
        self.logger.info("📦 UnifiedModel 로드 중...")
        
        try:
            # UnifiedModel 임포트
            from training.unified_training_final import UnifiedModel, UnifiedTrainingConfig
            from training.enhanced_checkpoint_manager import EnhancedCheckpointManager
            
            # 체크포인트 매니저 초기화
            self.checkpoint_manager = EnhancedCheckpointManager(
                checkpoint_dir=self.config.checkpoint_dir
            )
            
            # 설정 생성
            train_config = UnifiedTrainingConfig()
            train_config.device = self.config.device
            
            # UnifiedModel 생성
            self.unified_model = UnifiedModel(config=train_config)
            
            # 체크포인트 경로 결정
            if self.config.checkpoint_path:
                # 직접 경로가 지정된 경우
                checkpoint_path = Path(self.config.checkpoint_path)
            else:
                # 에폭 번호로 자동 검색
                pattern = f"{self.config.checkpoint_dir}/checkpoint_epoch_{self.config.checkpoint_epoch:04d}_*.pt"
                matches = glob.glob(pattern)
                if matches:
                    checkpoint_path = Path(matches[0])  # 해당 에폭의 체크포인트 사용
                    self.logger.info(f"   에폭 {self.config.checkpoint_epoch} 체크포인트 자동 검색: {checkpoint_path.name}")
                else:
                    # 못 찾으면 가장 최근 체크포인트 사용
                    all_checkpoints = sorted(glob.glob(f"{self.config.checkpoint_dir}/checkpoint_epoch_*.pt"))
                    if all_checkpoints:
                        checkpoint_path = Path(all_checkpoints[-1])
                        self.logger.warning(f"   에폭 {self.config.checkpoint_epoch} 체크포인트 없음. 최신 사용: {checkpoint_path.name}")
                    else:
                        checkpoint_path = None
                        self.logger.warning(f"   체크포인트 없음. 새로운 가중치로 시작...")
            
            # 체크포인트 로드
            if checkpoint_path and checkpoint_path.exists():
                self.logger.info(f"   체크포인트 로드: {checkpoint_path.name}")
                
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location=self.config.device,
                    weights_only=False  # 전체 모델 구조 포함
                )
                
                # 체크포인트를 나중에 사용할 수 있도록 저장
                self.loaded_checkpoint = checkpoint
                
                # 모델 가중치 로드
                if 'model_state' in checkpoint:
                    self.unified_model.load_state_dict(
                        checkpoint['model_state'],
                        strict=False  # 일부 누락 허용
                    )
                    self.logger.info(f"   ✅ 모델 가중치 로드 완료")
                    self.logger.info(f"   - Epoch: {checkpoint.get('epoch', 'unknown')}")
                    best_loss = checkpoint.get('best_loss', 'unknown')
                    if isinstance(best_loss, (int, float)):
                        self.logger.info(f"   - Loss: {best_loss:.4f}")
                    else:
                        self.logger.info(f"   - Loss: {best_loss}")
                else:
                    self.logger.warning("   ⚠️ 체크포인트에 model_state가 없음")
            else:
                self.logger.warning(f"   ⚠️ 체크포인트 파일 없음: {checkpoint_path}")
                self.logger.info("   새로운 가중치로 시작...")
            
            # 평가 모드로 전환
            self.unified_model.eval()
            
            # MEDIUM 모드에서는 백본만 GPU로, 나머지는 CPU에 유지
            if self.config.memory_mode == MemoryMode.MEDIUM:
                # 백본과 헤드만 GPU로
                self.unified_model.backbone.to(self.config.device)
                self.unified_model.emotion_head.to(self.config.device)
                self.unified_model.bentham_head.to(self.config.device)
                self.unified_model.regret_head.to(self.config.device)
                self.unified_model.surd_head.to(self.config.device)
                # neural_analyzers는 이미 CPU에 있음 (MEDIUM 모드 설정)
                self.logger.info("   📌 MEDIUM 모드: 백본/헤드만 GPU, analyzers는 CPU 유지")
            else:
                # 다른 모드에서는 전체를 GPU로
                self.unified_model.to(self.config.device)
            
            # 모델 파라미터 정보
            total_params = sum(p.numel() for p in self.unified_model.parameters())
            trainable_params = sum(p.numel() for p in self.unified_model.parameters() if p.requires_grad)
            self.logger.info(f"   📊 총 파라미터: {total_params/1e6:.1f}M")
            self.logger.info(f"   📊 학습가능 파라미터: {trainable_params/1e6:.1f}M")
            
            # DSM에 UnifiedModel 등록 (DSM이 없으면 초기화)
            if not hasattr(self, 'swap_manager') or self.swap_manager is None:
                # DSM이 없으면 여기서 즉시 초기화
                if self.config.llm_mode != "none":
                    self.logger.warning("   ⚠️ DSM이 초기화되지 않음. 지금 초기화합니다...")
                    try:
                        from dynamic_swap_manager import DynamicSwapManager, set_swap_manager
                        self.swap_manager = DynamicSwapManager.get_instance()
                        set_swap_manager(self.swap_manager)
                        self.logger.info(f"   ✅ DSM 긴급 초기화 완료 (ID: {id(self.swap_manager)})")
                    except Exception as e:
                        self.logger.error(f"   ❌ DSM 긴급 초기화 실패: {e}")
            
            # DSM 등록 시도
            if self.swap_manager:
                try:
                    from dynamic_swap_manager import SwapPriority
                    
                    # 백본과 헤드들을 개별 등록 (세밀한 관리) - owner 정보 포함
                    self.swap_manager.register_model(
                        'unified_backbone', 
                        self.unified_model.backbone,
                        priority=SwapPriority.CRITICAL,
                        owner_obj=self.unified_model,
                        owner_attr='backbone'
                    )
                    self.swap_manager.register_model(
                        'emotion_head', 
                        self.unified_model.emotion_head,
                        priority=SwapPriority.HIGH,
                        owner_obj=self.unified_model,
                        owner_attr='emotion_head'
                    )
                    self.swap_manager.register_model(
                        'bentham_head', 
                        self.unified_model.bentham_head,
                        priority=SwapPriority.HIGH,
                        owner_obj=self.unified_model,
                        owner_attr='bentham_head'
                    )
                    self.swap_manager.register_model(
                        'regret_head', 
                        self.unified_model.regret_head,
                        priority=SwapPriority.HIGH,
                        owner_obj=self.unified_model,
                        owner_attr='regret_head'
                    )
                    self.swap_manager.register_model(
                        'surd_head', 
                        self.unified_model.surd_head,
                        priority=SwapPriority.HIGH,
                        owner_obj=self.unified_model,
                        owner_attr='surd_head'
                    )
                    self.logger.info("   📌 UnifiedModel DSM 등록 완료")
                except Exception as dsm_error:
                    import traceback
                    self.logger.error(f"   ❌ DSM 등록 실패: {dsm_error}")
                    self.logger.error(f"   Traceback:\n{traceback.format_exc()}")
            else:
                self.logger.warning("   ⚠️ swap_manager가 None - DSM 등록 스킵")
            
        except Exception as e:
            self.logger.error(f"   ❌ UnifiedModel 로드 실패: {e}")
            raise
    
    async def _load_neural_analyzers(self):
        """Neural Analyzers 로드 (368M 또는 부분)"""
        subset = getattr(self.config, 'neural_analyzers_subset', None)
        if subset:
            self.logger.info(f"🧠 Neural Analyzers 부분 로드 중 ({subset})...")
        else:
            self.logger.info("🧠 Neural Analyzers 전체 로드 중 (368M)...")
        
        try:
            from analyzer_neural_modules import create_neural_analyzers
            
            # Neural Analyzers 생성 - 768차원 (BERT/임베딩 차원)
            all_analyzers = create_neural_analyzers(input_dim=768)
            
            # 부분 로드 모드인 경우 필요한 것만 선택
            if subset:
                self.neural_analyzers = {k: v for k, v in all_analyzers.items() if k in subset}
                self.logger.info(f"   부분 로드 완료: {list(self.neural_analyzers.keys())}")
            else:
                self.neural_analyzers = all_analyzers
            
            # 체크포인트에서 가중치 복원 시도 (차원 호환성 체크)
            if hasattr(self.unified_model, 'neural_analyzers'):
                # 기존 neural_analyzers의 차원 확인
                old_analyzers = self.unified_model.neural_analyzers
                if old_analyzers and 'emotion' in old_analyzers:
                    # 첫 번째 레이어의 입력 차원 확인
                    first_layer = next(old_analyzers['emotion'].parameters())
                    old_dim = first_layer.shape[-1] if len(first_layer.shape) > 1 else first_layer.shape[0]
                    
                    if old_dim == 768:
                        self.logger.info("   체크포인트에서 Neural Analyzers 가중치 복원 (768차원 호환)...")
                        self.neural_analyzers = old_analyzers
                    elif old_dim == 896:
                        # 896차원 가중치를 사용하되, 768->896 프로젝션 어댑터 추가
                        self.logger.info(f"   체크포인트 Neural Analyzers 896차원 가중치 복원 중...")
                        self.neural_analyzers = old_analyzers
                        
                        # 각 analyzer에 대해 프로젝션 어댑터 추가
                        self.neural_projection_adapters = {}
                        for name in self.neural_analyzers.keys():
                            # 768 -> 896 프로젝션 레이어
                            adapter = nn.Sequential(
                                nn.Linear(768, 896),
                                nn.LayerNorm(896),
                                nn.GELU()
                            ).to(self.config.device)
                            self.neural_projection_adapters[name] = adapter
                            self.logger.info(f"   - {name}: 768→896 프로젝션 어댑터 추가")
                        
                        self.logger.info("   ✅ 896차원 Neural Analyzers 복원 완료 (프로젝션 어댑터 사용)")
                    else:
                        self.logger.warning(f"   체크포인트 Neural Analyzers 차원 불일치 ({old_dim}차원), 새로 생성된 768차원 사용")
                        # 새로 생성된 768차원 neural_analyzers 유지
            
            # 평가 모드로 전환 및 디바이스 할당 (메모리 모드별 전략)
            for name, module in self.neural_analyzers.items():
                module.eval()
                # MEDIUM 모드에서는 CPU로 초기화 (실행 시 동적 스왑)
                if self.config.memory_mode == MemoryMode.MEDIUM:
                    target_device = torch.device('cpu')
                    self.logger.info(f"   - {name}: CPU 초기화 (동적 스왑 대기)")
                else:
                    target_device = self.config.device
                module.to(target_device)
                params = sum(p.numel() for p in module.parameters())
                self.logger.info(f"   - {name}: {params/1e6:.1f}M params")
            
            self.logger.info("   ✅ Neural Analyzers 로드 완료")
            
            # UnifiedModel에 neural_analyzers 전달
            if hasattr(self, 'unified_model') and self.unified_model is not None:
                # UnifiedModel의 기존 neural_analyzers를 덮어쓰기
                self.unified_model.neural_analyzers = nn.ModuleDict(self.neural_analyzers)
                self.logger.info("   📌 UnifiedModel에 Neural Analyzers 전달 완료")
            
            # DSM에 Neural Analyzers 등록
            if hasattr(self, 'swap_manager') and self.swap_manager and self.neural_analyzers:
                try:
                    from dynamic_swap_manager import SwapPriority
                    
                    for name, analyzer in self.neural_analyzers.items():
                        # Neural Analyzers는 MEDIUM 우선순위
                        self.swap_manager.register_model(
                            f'neural_{name}', 
                            analyzer,
                            priority=SwapPriority.MEDIUM
                        )
                    self.logger.info("   📌 Neural Analyzers DSM 등록 완료")
                except Exception as dsm_error:
                    self.logger.warning(f"   ⚠️ DSM 등록 실패: {dsm_error}")
            
        except Exception as e:
            self.logger.error(f"   ❌ Neural Analyzers 로드 실패: {e}")
            raise RuntimeError(f"Neural Analyzers 로드 필수 - 실패: {e}")
    
    async def _load_advanced_wrappers(self):
        """Advanced Analyzer Wrappers 로드 (112M 또는 부분)"""
        subset = getattr(self.config, 'advanced_wrappers_subset', None)
        if subset:
            self.logger.info(f"🎯 Advanced Analyzer Wrappers 부분 로드 중 ({subset})...")
        else:
            self.logger.info("🎯 Advanced Analyzer Wrappers 전체 로드 중 (112M)...")
        
        try:
            from advanced_analyzer_wrappers import create_advanced_analyzer_wrappers
            
            # MEDIUM 모드에서는 CPU 초기화 강제
            if self.config.memory_mode == MemoryMode.MEDIUM:
                import os
                os.environ['FORCE_CPU_INIT'] = '1'
                self.logger.info("   📌 MEDIUM 모드: Advanced Wrappers CPU 초기화 설정")
            
            # Advanced Wrappers 생성
            all_wrappers = create_advanced_analyzer_wrappers()
            
            # 환경변수 정리
            if 'FORCE_CPU_INIT' in os.environ:
                del os.environ['FORCE_CPU_INIT']
            
            # 부분 로드 모드인 경우 필요한 것만 선택
            if subset:
                self.advanced_wrappers = {k: v for k, v in all_wrappers.items() if k in subset}
                self.logger.info(f"   부분 로드 완료: {list(self.advanced_wrappers.keys())}")
            else:
                self.advanced_wrappers = all_wrappers
            
            # 체크포인트에서 가중치 복원 시도
            if hasattr(self, 'loaded_checkpoint') and self.loaded_checkpoint is not None:
                if 'model_state' in self.loaded_checkpoint and 'advanced_wrappers' in self.loaded_checkpoint['model_state']:
                    self.logger.info("   체크포인트에서 Advanced Wrappers 가중치 복원...")
                    saved_wrappers = self.loaded_checkpoint['model_state']['advanced_wrappers']
                    
                    # 체크포인트의 가중치를 현재 wrapper에 로드
                    for name in self.advanced_wrappers.keys():
                        if name in saved_wrappers:
                            try:
                                # 키 리매핑 처리 - 체크포인트와 현재 코드의 키 이름 차이 해결
                                saved_state = saved_wrappers[name]
                                current_state = self.advanced_wrappers[name].state_dict()
                                remapped_state = {}
                                
                                # 현재 모델의 키를 기준으로 체크포인트에서 매칭되는 키 찾기
                                for current_key in current_state.keys():
                                    # 직접 매칭 시도
                                    if current_key in saved_state:
                                        saved_tensor = saved_state[current_key]
                                        current_tensor = current_state[current_key]
                                        
                                        # Shape 불일치 처리
                                        if saved_tensor.shape != current_tensor.shape:
                                            self.logger.info(f"       Shape 불일치 감지: {current_key}")
                                            self.logger.info(f"         체크포인트: {list(saved_tensor.shape)}")
                                            self.logger.info(f"         현재 모델: {list(current_tensor.shape)}")
                                            
                                            # diversity_layer.weight 특별 처리
                                            if 'diversity_layer.weight' in current_key:
                                                # [4, 192] → [4, 768] 또는 [4, 128] → [4, 512]
                                                if len(saved_tensor.shape) == 2 and len(current_tensor.shape) == 2:
                                                    if saved_tensor.shape[0] == current_tensor.shape[0]:
                                                        # 프로젝션: 작은 차원을 큰 차원으로 확장
                                                        expanded_tensor = torch.zeros_like(current_tensor)
                                                        min_dim = min(saved_tensor.shape[1], current_tensor.shape[1])
                                                        expanded_tensor[:, :min_dim] = saved_tensor[:, :min_dim]
                                                        
                                                        # Xavier 초기화로 나머지 부분 채우기
                                                        if min_dim < current_tensor.shape[1]:
                                                            nn.init.xavier_uniform_(expanded_tensor[:, min_dim:])
                                                        
                                                        remapped_state[current_key] = expanded_tensor
                                                        self.logger.info(f"         ✅ 프로젝션 적용: {saved_tensor.shape} → {current_tensor.shape}")
                                                    else:
                                                        self.logger.warning(f"         ❌ 첫 번째 차원 불일치, 스킵")
                                                else:
                                                    self.logger.warning(f"         ❌ 차원 수 불일치, 스킵")
                                            else:
                                                # 다른 레이어들은 스킵 (새로 초기화)
                                                self.logger.warning(f"         ⚠️ Shape 불일치로 스킵")
                                        else:
                                            remapped_state[current_key] = saved_state[current_key]
                                    # gating_network 리매핑
                                    elif 'gate' in current_key:
                                        # gate → gating_network.gating_network 매핑
                                        checkpoint_key = current_key.replace('gate', 'gating_network.gating_network')
                                        if checkpoint_key in saved_state:
                                            remapped_state[current_key] = saved_state[checkpoint_key]
                                            self.logger.debug(f"       키 리매핑: {checkpoint_key} → {current_key}")
                                    # emotion_moe.ga → emotion_moe.gating_network.gating_network 매핑
                                    elif 'emotion_moe.ga' in current_key:
                                        checkpoint_key = current_key.replace('emotion_moe.ga', 'emotion_moe.gating_network.gating_network')
                                        if checkpoint_key in saved_state:
                                            remapped_state[current_key] = saved_state[checkpoint_key]
                                            self.logger.debug(f"       키 리매핑: {checkpoint_key} → {current_key}")
                                    # bentham_ethi → bentham_deep_ethics 매핑
                                    elif 'bentham_ethi' in current_key:
                                        checkpoint_key = current_key.replace('bentham_ethi', 'bentham_deep_ethics')
                                        if checkpoint_key in saved_state:
                                            remapped_state[current_key] = saved_state[checkpoint_key]
                                            self.logger.debug(f"       키 리매핑: {checkpoint_key} → {current_key}")
                                
                                # 리매핑된 state_dict 로드
                                if remapped_state:
                                    incompatible = self.advanced_wrappers[name].load_state_dict(
                                        remapped_state, 
                                        strict=False
                                    )
                                    if incompatible.missing_keys or incompatible.unexpected_keys:
                                        self.logger.warning(f"     ⚠️ {name} 부분 복원 - 누락: {len(incompatible.missing_keys)}개, 예상외: {len(incompatible.unexpected_keys)}개")
                                    else:
                                        self.logger.info(f"     ✅ {name} 가중치 완전 복원 성공")
                                    self.logger.info(f"     📊 {name} 리매핑 성공: {len(remapped_state)}/{len(current_state)}개 키 복원")
                                else:
                                    self.logger.warning(f"     ⚠️ {name} 리매핑 실패 - 매칭되는 키 없음")
                            except Exception as e:
                                # shape mismatch 등의 경우 새로 초기화된 가중치 사용
                                error_msg = str(e)
                                self.logger.warning(f"     ⚠️ {name} 가중치 복원 실패 - 새로 초기화된 가중치 사용: {error_msg}")
                                
                                # size mismatch 에러인 경우 상세 정보 출력
                                if "size mismatch" in error_msg:
                                    # 어떤 키에서 문제가 발생했는지 파악
                                    import re
                                    match = re.search(r"size mismatch for (.+?):", error_msg)
                                    if match:
                                        problem_key = match.group(1)
                                        if problem_key in remapped_state:
                                            checkpoint_shape = list(remapped_state[problem_key].shape)
                                            current_shape = list(current_state[problem_key].shape) if problem_key in current_state else "N/A"
                                            self.logger.warning(f"       Shape 불일치: {problem_key}")
                                            self.logger.warning(f"         체크포인트: {checkpoint_shape}")
                                            self.logger.warning(f"         현재 모델: {current_shape}")
                        else:
                            self.logger.warning(f"     ⚠️ {name}이 체크포인트에 없음")
                else:
                    self.logger.info("   ⚠️ 체크포인트에 Advanced Wrappers 가중치 없음 - 랜덤 초기화 사용")
            else:
                self.logger.info("   ⚠️ 체크포인트가 로드되지 않음 - 랜덤 초기화 사용")
            
            # None 체크 추가
            if self.advanced_wrappers is None:
                self.logger.error("   ❌ Advanced Wrappers가 None입니다")
                raise ValueError("Advanced Wrappers 생성 실패")
            
            # 평가 모드로 전환
            for name, wrapper in self.advanced_wrappers.items():
                if hasattr(wrapper, 'eval'):
                    wrapper.eval()
                    # MEDIUM 모드에서는 CPU로
                    if self.config.memory_mode == MemoryMode.MEDIUM:
                        wrapper.to(torch.device('cpu'))
                    else:
                        wrapper.to(self.config.device)
                params = sum(p.numel() for p in wrapper.parameters() if hasattr(wrapper, 'parameters'))
                self.logger.info(f"   - {name}: {params/1e6:.1f}M params")
            
            self.logger.info("   ✅ Advanced Wrappers 로드 완료")
            
            # UnifiedModel에 advanced_wrappers 전달
            if hasattr(self, 'unified_model') and self.unified_model is not None:
                self.unified_model.advanced_wrappers = nn.ModuleDict(self.advanced_wrappers)
                self.logger.info("   📌 UnifiedModel에 Advanced Wrappers 전달 완료")
            
            # DSM에 Advanced Wrappers 등록
            if hasattr(self, 'swap_manager') and self.swap_manager and self.advanced_wrappers:
                try:
                    from dynamic_swap_manager import SwapPriority
                    
                    for name, wrapper in self.advanced_wrappers.items():
                        # Advanced Wrappers는 MEDIUM 우선순위
                        self.swap_manager.register_model(
                            f'wrapper_{name}', 
                            wrapper,
                            priority=SwapPriority.MEDIUM
                        )
                    self.logger.info("   📌 Advanced Wrappers DSM 등록 완료")
                except Exception as dsm_error:
                    self.logger.warning(f"   ⚠️ DSM 등록 실패: {dsm_error}")
            
        except Exception as e:
            self.logger.error(f"   ❌ Advanced Wrappers 로드 실패: {e}")
            raise RuntimeError(f"Advanced Wrappers 로드 필수 - 실패: {e}")
    
    async def _load_dsp_components(self):
        """DSP 시뮬레이터 & Kalman Filter 로드"""
        self.logger.info("📡 DSP 컴포넌트 로드 중...")
        
        try:
            from emotion_dsp_simulator import EmotionDSPSimulator
            
            # DSP 시뮬레이터 생성
            self.dsp_simulator = EmotionDSPSimulator()
            
            # 체크포인트에서 복원
            if hasattr(self.unified_model, 'dsp_simulator'):
                self.logger.info("   체크포인트에서 DSP 시뮬레이터 복원...")
                self.dsp_simulator = self.unified_model.dsp_simulator
            
            # MEDIUM 모드에서는 CPU로 초기화, 아니면 GPU로
            dsp_device = torch.device('cpu') if self.config.memory_mode == MemoryMode.MEDIUM else self.config.device
            self.dsp_simulator = self.dsp_simulator.to(dsp_device)
            self.dsp_simulator.eval()
            if self.config.memory_mode == MemoryMode.MEDIUM:
                self.logger.info("   📌 DSP 시뮬레이터 CPU 초기화 (동적 스왑 대기)")
            
            # Kalman Filter는 DSP 내부에 포함
            if hasattr(self.dsp_simulator, 'kalman_filter'):
                self.kalman_filter = self.dsp_simulator.kalman_filter
                self.logger.info("   ✅ Kalman Filter 활성화")
            
            params = sum(p.numel() for p in self.dsp_simulator.parameters())
            self.logger.info(f"   📊 DSP 파라미터: {params/1e6:.1f}M")
            self.logger.info("   ✅ DSP 컴포넌트 로드 완료")
            
        except Exception as e:
            self.logger.error(f"   ❌ DSP 컴포넌트 로드 실패: {e}")
            raise RuntimeError(f"DSP 컴포넌트 로드 필수 - 실패: {e}")
    
    async def _load_phase_networks(self):
        """Phase Networks 로드"""
        self.logger.info("🔄 Phase Networks 로드 중...")
        
        try:
            from phase_neural_networks import (
                Phase0ProjectionNet,
                Phase2CommunityNet,
                HierarchicalEmotionIntegrator
            )
            
            self.phase_networks = {
                'phase0': Phase0ProjectionNet(),
                'phase2': Phase2CommunityNet(),
                'hierarchical': HierarchicalEmotionIntegrator()
            }
            
            # MEDIUM 모드에서는 CPU로 초기화, 아니면 GPU로
            phase_device = torch.device('cpu') if self.config.memory_mode == MemoryMode.MEDIUM else self.config.device
            for name, network in self.phase_networks.items():
                self.phase_networks[name] = network.to(phase_device)
            
            if self.config.memory_mode == MemoryMode.MEDIUM:
                self.logger.info("   📌 Phase Networks CPU 초기화 (동적 스왑 대기)")
            
            # 체크포인트에서 가중치 복원 시도
            if hasattr(self, 'unified_model') and hasattr(self.unified_model, 'phase_networks'):
                self.logger.info("   체크포인트에서 Phase Networks 가중치 복원 중...")
                checkpoint_phase_nets = self.unified_model.phase_networks
                
                for name in self.phase_networks.keys():
                    if name in checkpoint_phase_nets:
                        try:
                            # 가중치 복사
                            self.phase_networks[name].load_state_dict(
                                checkpoint_phase_nets[name].state_dict()
                            )
                            self.logger.info(f"   - {name}: 가중치 복원 완료")
                        except Exception as e:
                            self.logger.warning(f"   - {name}: 가중치 복원 실패 ({e}), 새로 초기화된 가중치 사용")
                    else:
                        self.logger.info(f"   - {name}: 체크포인트에 없음, 새로 초기화된 가중치 사용")
            else:
                self.logger.info("   체크포인트에 Phase Networks 없음, 새로 초기화된 가중치 사용")
            
            # 평가 모드로 전환 및 디바이스 할당 (메모리 모드별 전략)
            for name, net in self.phase_networks.items():
                net.eval()
                # MEDIUM 모드에서는 CPU로 초기화 (실행 시 동적 스왑)
                if self.config.memory_mode == MemoryMode.MEDIUM:
                    target_device = torch.device('cpu')
                    self.logger.info(f"   - {name}: CPU 초기화 (동적 스왑 대기)")
                else:
                    target_device = self.config.device
                net.to(target_device)
                params = sum(p.numel() for p in net.parameters())
                self.logger.info(f"   - {name}: {params/1e6:.2f}M params")
            
            # Phase 출력 프로젝터 초기화 (896 → 768)
            # MEDIUM 모드에서도 프로젝터는 CPU로
            projector_device = torch.device('cpu') if self.config.memory_mode == MemoryMode.MEDIUM else self.config.device
            self.phase_output_projector = nn.Linear(896, 768).to(projector_device)
            self.logger.info("   - output_projector: 896→768 차원 변환기 초기화")
            
            self.logger.info("   ✅ Phase Networks 로드 완료")
            
        except Exception as e:
            self.logger.error(f"   ❌ Phase Networks 로드 실패: {e}")
            raise RuntimeError(f"Phase Networks 로드 필수 - 실패: {e}")
    
    async def _init_dsm_early(self):
        """DSM 초기 초기화 (UnifiedModel 로드 전)"""
        self.logger.info("🔄 Dynamic Swap Manager 초기 초기화...")
        
        try:
            from dynamic_swap_manager import DynamicSwapManager, set_swap_manager
            
            # DSM 싱글톤 인스턴스 생성 
            self.swap_manager = DynamicSwapManager.get_instance()
            set_swap_manager(self.swap_manager)
            
            self.logger.info(f"   ✅ DSM 초기화 완료 (ID: {id(self.swap_manager)})")
            
        except Exception as e:
            self.logger.warning(f"   ⚠️ DSM 초기 초기화 실패: {e}")
            # DSM 없이도 계속 진행
    
    async def _load_llm_integration(self):
        """LLM 통합 로드 및 스왑 매니저 설정"""
        self.logger.info(f"🤖 LLM 통합 로드 중 (모드: {self.config.llm_mode})...")
        
        try:
            # API 모드 목록 정의 (claude는 별도 처리)
            api_modes = ['gpt', 'perplexity', 'deepseek']
            
            # API 모드 체크를 먼저 수행
            if self.config.llm_mode in api_modes:
                # API 모드 (GPT, Perplexity, DeepSeek)
                self.logger.info(f"   🌐 API 모드 활성화: {self.config.llm_mode}")
                
                # Dynamic Swap Manager 초기화 (이미 초기화되어 있지 않은 경우만)
                if not hasattr(self, 'swap_manager') or self.swap_manager is None:
                    self.logger.info("   🔄 Dynamic Swap Manager 초기화...")
                    from dynamic_swap_manager import DynamicSwapManager, set_swap_manager
                    
                    # DSM 인스턴스 생성 및 전역 설정
                    self.swap_manager = DynamicSwapManager.get_instance()
                    set_swap_manager(self.swap_manager)  # dynamic_swap_manager.py의 전역 설정
                    
                    self.logger.info(f"   ✅ DSM 초기화 완료 (ID: {id(self.swap_manager)})")
                else:
                    self.logger.info(f"   📌 DSM 이미 초기화됨 (ID: {id(self.swap_manager)})")
                
                # LLM 엔진 초기화
                from llm_module.advanced_llm_engine import AdvancedLLMEngine, set_llm_engine
                
                self.llm_engine = AdvancedLLMEngine(use_api=self.config.llm_mode)
                
                # 전역 LLM 엔진 설정 (다른 모듈들이 사용할 수 있도록)
                set_llm_engine(self.llm_engine)
                
                # Advanced Wrappers가 이미 로드된 경우 LLM 엔진 업데이트
                if hasattr(self, 'advanced_wrappers') and self.advanced_wrappers:
                    for wrapper_name, wrapper in self.advanced_wrappers.items():
                        if hasattr(wrapper, 'llm_engine'):
                            wrapper.llm_engine = self.llm_engine
                            self.logger.info(f"   📌 {wrapper_name} LLM 엔진 업데이트 완료")
                
                self.logger.info(f"   ✅ {self.config.llm_mode.upper()} API 엔진 초기화 및 전역 설정 완료")
                
            elif self.config.llm_mode == "local":
                # 로컬 LLM (Dolphin Llama3 8B) - 스왑 매니저 사용
                self.logger.info("   메모리 스왑 매니저 초기화...")
                
                # 스왑 매니저 설정
                swap_config = {
                    'gpu_threshold': 7000,  # 8GB GPU 기준
                    'ram_threshold': 16000,
                    'llm_model_path': self.config.llm_model_path,
                    'generate_explanation': True,
                    'enable_optimization': True
                }
                
                self.swap_manager = SystemSwapManager(swap_config)
                
                # Red Heart를 RAM에 대기 (MD 문서: 초기 상태)
                await self.swap_manager.initialize(
                    red_heart_system=self,  # 현재 시스템 전달
                    llm_model=None  # LLM은 아직 로드하지 않음
                )
                
                self.logger.info("   ✅ 메모리 스왑 매니저 설정 완료")
                self.logger.info("   📌 Red Heart는 RAM에, LLM은 필요시 로드")
                
                # 로컬 LLM 엔진 초기화
                from llm_module.advanced_llm_engine import AdvancedLLMEngine, set_llm_engine
                
                self.llm_engine = AdvancedLLMEngine()
                
                # 전역 LLM 엔진 설정
                set_llm_engine(self.llm_engine)
                
                # Advanced Wrappers가 이미 로드된 경우 LLM 엔진 업데이트
                if hasattr(self, 'advanced_wrappers') and self.advanced_wrappers:
                    for wrapper_name, wrapper in self.advanced_wrappers.items():
                        if hasattr(wrapper, 'llm_engine'):
                            wrapper.llm_engine = self.llm_engine
                            self.logger.info(f"   📌 {wrapper_name} LLM 엔진 업데이트 완료")
                
                self.logger.info("   ✅ Local LLM 엔진 초기화 및 전역 설정 완료 (CPU 대기)")
                self.logger.info(f"   📌 모델: Dolphin Llama3 8B")
                
            elif self.config.llm_mode == "claude":
                # Claude API 통합 (DSM 사용하지 않음, 직접 GPU 관리)
                self.logger.info("   🌐 Claude API 모드 활성화 (DSM 비활성화)")
                self.logger.info("   🎯 GPU 직접 관리 모드로 전환")
                
                # GPU 직접 관리를 위한 헬퍼 클래스
                import gc
                class DirectGPUManager:
                    def __init__(self, logger):
                        self.logger = logger
                        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        self.models_on_gpu = {}
                        
                    def clear_gpu_cache(self):
                        """GPU 캐시 정리"""
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        gc.collect()
                        self.logger.info("   🧹 GPU 캐시 정리 완료")
                        
                    def get_gpu_memory(self):
                        """GPU 메모리 현황 확인"""
                        if torch.cuda.is_available():
                            allocated = torch.cuda.memory_allocated() / 1024**3
                            reserved = torch.cuda.memory_reserved() / 1024**3
                            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                            return allocated, reserved, total
                        return 0, 0, 0
                        
                    def move_to_gpu(self, model, name):
                        """모델을 GPU로 이동"""
                        allocated, reserved, total = self.get_gpu_memory()
                        self.logger.info(f"   📊 GPU 메모리: {allocated:.2f}/{total:.2f}GB 사용중")
                        
                        # 메모리 부족시 캐시 정리
                        if allocated > total * 0.8:
                            self.clear_gpu_cache()
                            
                        model = model.to(self.device)
                        self.models_on_gpu[name] = model
                        self.logger.info(f"   ✅ {name} GPU 로드 완료")
                        return model
                        
                    def move_to_cpu(self, model, name):
                        """모델을 CPU로 이동"""
                        model = model.cpu()
                        if name in self.models_on_gpu:
                            del self.models_on_gpu[name]
                        self.clear_gpu_cache()
                        self.logger.info(f"   ✅ {name} CPU로 언로드 완료")
                        return model
                
                # GPU 매니저 생성 (DSM 대신)
                self.gpu_manager = DirectGPUManager(self.logger)
                
                # Claude API 엔진 초기화
                from llm_module.advanced_llm_engine import AdvancedLLMEngine, set_llm_engine
                self.llm_engine = AdvancedLLMEngine(use_api='claude')
                
                # 전역 LLM 엔진 설정
                set_llm_engine(self.llm_engine)
                
                # Advanced Wrappers가 이미 로드된 경우 LLM 엔진 업데이트
                if hasattr(self, 'advanced_wrappers') and self.advanced_wrappers:
                    for wrapper_name, wrapper in self.advanced_wrappers.items():
                        if hasattr(wrapper, 'llm_engine'):
                            wrapper.llm_engine = self.llm_engine
                            self.logger.info(f"   📌 {wrapper_name} LLM 엔진 업데이트 완료")
                
                # DSM 비활성화 플래그 설정
                self.use_dsm = False
                self.swap_manager = None
                
                self.logger.info("   ✅ Claude API 엔진 초기화 완료 (직접 GPU 관리)")
                
            elif self.config.llm_mode == "mcp":
                # MCP 프로토콜
                self.logger.info("   🌐 MCP 프로토콜 모드 활성화")
                
                from llm_module.mcp_client import MCPClient, get_mcp_client
                from llm_module.advanced_llm_engine import set_llm_engine
                
                # MCP 클라이언트 생성 및 연결
                self.llm_engine = MCPClient()
                connected = await self.llm_engine.connect()
                
                if connected:
                    # 전역 설정
                    set_llm_engine(self.llm_engine)
                    
                    # Advanced Wrappers가 이미 로드된 경우 LLM 엔진 업데이트
                    if hasattr(self, 'advanced_wrappers') and self.advanced_wrappers:
                        for wrapper_name, wrapper in self.advanced_wrappers.items():
                            if hasattr(wrapper, 'llm_engine'):
                                wrapper.llm_engine = self.llm_engine
                                self.logger.info(f"   📌 {wrapper_name} LLM 엔진 업데이트 완료")
                    
                    self.logger.info("   ✅ MCP 클라이언트 초기화 및 연결 완료")
                    self.logger.info("   📌 Red Heart Ethics 서버와 통신 중")
                else:
                    self.logger.warning("   ⚠️ MCP 서버 연결 실패")
                    self.logger.info("   💡 MCP 서버 시작: python mcp_server.py")
                    raise RuntimeError("MCP 서버가 실행 중이 아닙니다. mcp_server.py를 먼저 실행하세요.")
                
        except Exception as e:
            self.logger.warning(f"   ⚠️ LLM 통합 실패: {e}")
            # llm_mode는 유지하고 DSM 등록은 계속 진행
        
        # DynamicSwapManager가 초기화되었으면 헤드들을 등록 (SystemSwapManager는 제외)
        if hasattr(self, 'swap_manager') and self.swap_manager and hasattr(self.swap_manager, 'register_model'):
            try:
                self.logger.info("   📌 UnifiedModel 헤드들을 DSM에 등록...")
                
                # 백본과 헤드들을 개별 등록
                if hasattr(self, 'unified_model') and self.unified_model:
                    from dynamic_swap_manager import SwapPriority
                    
                    # 백본은 CRITICAL 우선순위
                    self.swap_manager.register_model(
                        'unified_backbone', 
                        self.unified_model.backbone, 
                        priority=SwapPriority.CRITICAL,
                        owner_obj=self.unified_model,
                        owner_attr='backbone'
                    )
                    
                    # 헤드들은 HIGH 우선순위 - owner 정보 포함
                    self.swap_manager.register_model(
                        'emotion_head', 
                        self.unified_model.emotion_head,
                        priority=SwapPriority.HIGH,
                        owner_obj=self.unified_model,
                        owner_attr='emotion_head'
                    )
                    self.swap_manager.register_model(
                        'bentham_head', 
                        self.unified_model.bentham_head,
                        priority=SwapPriority.HIGH,
                        owner_obj=self.unified_model,
                        owner_attr='bentham_head'
                    )
                    self.swap_manager.register_model(
                        'regret_head', 
                        self.unified_model.regret_head,
                        priority=SwapPriority.HIGH,
                        owner_obj=self.unified_model,
                        owner_attr='regret_head'
                    )
                    self.swap_manager.register_model(
                        'surd_head', 
                        self.unified_model.surd_head,
                        priority=SwapPriority.HIGH,
                        owner_obj=self.unified_model,
                        owner_attr='surd_head'
                    )
                    
                    self.logger.info("   ✅ UnifiedModel DSM 등록 완료 (API 모드)")
                
                # Neural Analyzers도 등록 (이미 로드된 경우)
                if hasattr(self, 'neural_analyzers') and self.neural_analyzers:
                    for name, analyzer in self.neural_analyzers.items():
                        self.swap_manager.register_model(
                            f'neural_{name}', 
                            analyzer,
                            priority=SwapPriority.MEDIUM
                        )
                    self.logger.info("   ✅ Neural Analyzers DSM 등록 완료")
                
                # Advanced Wrappers도 등록 (이미 로드된 경우)
                if hasattr(self, 'advanced_wrappers') and self.advanced_wrappers:
                    for name, wrapper in self.advanced_wrappers.items():
                        self.swap_manager.register_model(
                            f'wrapper_{name}', 
                            wrapper,
                            priority=SwapPriority.MEDIUM
                        )
                    self.logger.info("   ✅ Advanced Wrappers DSM 등록 완료")
                    
            except Exception as dsm_error:
                self.logger.warning(f"   ⚠️ DSM 헤드 등록 실패: {dsm_error}")
    
    async def _load_translator(self):
        """번역기 로드 (선택적)"""
        self.logger.info("🌐 번역기 로드 중...")
        
        try:
            from local_translator import LocalTranslator
            from config import register_system_module
            
            self.translator = LocalTranslator()
            
            # initialize() 메서드가 있으면 호출
            if hasattr(self.translator, 'initialize'):
                await self.translator.initialize()
            
            # 시스템 모듈로 등록 (Advanced Emotion Wrapper가 찾을 수 있도록)
            register_system_module('translator', self.translator)
            
            self.logger.info("   ✅ 번역기 로드 및 등록 완료")
            
        except Exception as e:
            self.logger.error(f"   ❌ 번역기 로드 실패: {e}")
            raise RuntimeError(f"번역기 로드 필수 - 실패: {e}")
    
    async def _load_workflow_memory_manager(self):
        """워크플로우 메모리 관리자 로드 (5M)"""
        self.logger.info("🧠 워크플로우 메모리 관리자 로드 중...")
        try:
            from workflow_aware_memory_manager import WorkflowAwareMemoryManager
            
            self.workflow_memory_manager = WorkflowAwareMemoryManager()
            # initialize 메서드 없음 - 생성자로 충분
            self.logger.info("   ✅ 워크플로우 메모리 관리자 로드 완료")
        except Exception as e:
            self.logger.error(f"   ❌ 워크플로우 메모리 관리자 로드 실패: {e}")
            raise RuntimeError(f"워크플로우 메모리 관리자 로드 필수 - 실패: {e}")
    
    async def _load_meta_integration(self):
        """메타 통합 시스템 로드 (40M)"""
        self.logger.info("🔮 메타 통합 시스템 로드 중 (40M)...")
        try:
            from advanced_meta_integration_system import AdvancedMetaIntegrationSystem
            
            self.meta_integration = AdvancedMetaIntegrationSystem()
            # AdvancedMetaIntegrationSystem은 내부에 integration_network를 가지고 있음
            # 이미 생성자에서 .to(device)와 .eval() 처리됨
            
            # 내부 네트워크의 파라미터 수 확인
            if hasattr(self.meta_integration, 'integration_network'):
                params = sum(p.numel() for p in self.meta_integration.integration_network.parameters())
                self.logger.info(f"   📊 메타 통합 파라미터: {params/1e6:.1f}M")
            
            self.logger.info("   ✅ 메타 통합 시스템 로드 완료")
        except Exception as e:
            self.logger.error(f"   ❌ 메타 통합 시스템 로드 실패: {e}")
            raise RuntimeError(f"메타 통합 시스템 로드 필수 - 실패: {e}")
    
    async def _load_counterfactual_reasoning(self):
        """반사실 추론 시스템 로드 (15M)"""
        self.logger.info("💭 반사실 추론 시스템 로드 중...")
        try:
            from advanced_counterfactual_reasoning import AdvancedCounterfactualReasoning
            
            self.counterfactual_reasoning = AdvancedCounterfactualReasoning()
            self.logger.info("   ✅ 반사실 추론 시스템 로드 완료")
        except Exception as e:
            self.logger.error(f"   ❌ 반사실 추론 시스템 로드 실패: {e}")
            raise RuntimeError(f"반사실 추론 시스템 로드 필수 - 실패: {e}")
    
    async def _load_advanced_regret_learning(self):
        """고급 후회 학습 시스템 로드 (20M)"""
        self.logger.info("😔 고급 후회 학습 시스템 로드 중...")
        try:
            from advanced_regret_learning_system import AdvancedRegretLearningSystem
            
            self.advanced_regret_learning = AdvancedRegretLearningSystem()
            self.logger.info("   ✅ 고급 후회 학습 시스템 로드 완료")
        except Exception as e:
            self.logger.error(f"   ❌ 고급 후회 학습 시스템 로드 실패: {e}")
            raise RuntimeError(f"고급 후회 학습 시스템 로드 필수 - 실패: {e}")
    
    async def _load_temporal_propagation(self):
        """시계열 전파 분석기 로드"""
        self.logger.info("⏰ 시계열 전파 분석기 로드 중...")
        try:
            from temporal_event_propagation_analyzer import TemporalEventPropagationAnalyzer
            
            self.temporal_propagator = TemporalEventPropagationAnalyzer()
            self.logger.info("   ✅ 시계열 전파 분석기 로드 완료")
        except Exception as e:
            self.logger.error(f"   ❌ 시계열 전파 분석기 로드 실패: {e}")
            raise RuntimeError(f"시계열 전파 분석기 로드 필수 - 실패: {e}")
    
    async def _load_experience_database(self):
        """경험 데이터베이스 로드"""
        self.logger.info("💾 경험 데이터베이스 로드 중...")
        try:
            from advanced_experience_database import AdvancedExperienceDatabase
            
            # AdvancedExperienceDatabase는 __init__에서 모든 초기화 완료
            self.experience_database = AdvancedExperienceDatabase()
            self.logger.info("   ✅ 경험 데이터베이스 로드 완료")
        except Exception as e:
            self.logger.error(f"   ❌ 경험 데이터베이스 로드 실패: {e}")
            raise RuntimeError(f"경험 데이터베이스 로드 필수 - 실패: {e}")
    
    async def _load_emotion_hierarchy(self):
        """계층적 감정 처리기 로드"""
        self.logger.info("🎭 계층적 감정 처리기 로드 중...")
        try:
            from emotion_ethics_regret_circuit import EmotionEthicsRegretCircuit
            
            # EmotionEthicsRegretCircuit에서 계층적 로직 추출
            self.emotion_hierarchy_processor = EmotionEthicsRegretCircuit()
            self.logger.info("   ✅ 계층적 감정 처리기 로드 완료 (공동체>타자>자아)")
        except Exception as e:
            self.logger.error(f"   ❌ 계층적 감정 처리기 로드 실패: {e}")
            raise RuntimeError(f"계층적 감정 처리기 로드 필수 - 실패: {e}")
    
    async def _load_precision_mapper(self):
        """정밀 감정→벤담 매퍼 로드 - 필수 구성 요소"""
        self.logger.info("🎯 정밀 감정→벤담 매퍼 초기화...")
        
        # 의미론적 매퍼는 항상 로드 (필수) - LIGHT 모드 포함 모든 모드
        if self.config.memory_mode.value in ['light', 'medium', 'heavy', 'mcp', 'normal', 'ultra', 'extreme']:
            self.emotion_bentham_mapper = SemanticEmotionBenthamMapper()
            self.logger.info("   ✅ 의미론적 매퍼 활성화")
            
            # EXTREME 모드에서는 신경망 어댑터도 로드
            if self.config.memory_mode == MemoryMode.EXTREME:
                self.neural_emotion_adapter = NeuralEmotionBenthamAdapter()
                self.neural_emotion_adapter.eval()
                self.neural_emotion_adapter.to(self.config.device)
                self.logger.info("   ✅ 신경망 어댑터 활성화")
        else:
            # MINIMAL 등 미정의 모드도 기본 매퍼 사용
            self.emotion_bentham_mapper = SemanticEmotionBenthamMapper()
            self.logger.info("   ✅ 기본 의미론적 매퍼 활성화")
    
    async def _load_three_view_scenario_system(self):
        """3뷰 시나리오 시스템 로드 (20M)"""
        self.logger.info("🔺 3뷰 시나리오 시스템 로드 중...")
        try:
            # 3뷰 시스템은 반사실 추론과 병합하여 사용
            self.three_view_system = ThreeViewScenarioSystem(device=self.config.device)
            self.logger.info("   ✅ 3뷰 시나리오 시스템 로드 완료 (낙관/중도/비관)")
        except Exception as e:
            self.logger.error(f"   ❌ 3뷰 시나리오 시스템 로드 실패: {e}")
            raise RuntimeError(f"3뷰 시나리오 시스템 로드 필수 - 실패: {e}")
    
    async def _load_multi_ethics_system(self):
        """다원적 윤리 체계 로드 (30M) - 5개 윤리학파"""
        self.logger.info("⚖️ 다원적 윤리 체계 로드 중...")
        try:
            # 전체 시스템 로드
            self.multi_ethics_system = DeepMultiDimensionalEthicsSystem()
            
            # 개별 엔진들도 접근 가능하도록 저장 (MD 문서 B안: 5개 학파)
            self.ethics_engines = {
                'utilitarianism': UtilitarianEngine(),      # 공리주의
                'deontological': DeontologicalEngine(),     # 의무론
                'virtue_ethics': VirtueEthicsEngine(),      # 덕윤리
                'care_ethics': CareEthicsEngine(),          # 돌봄윤리
                'justice_theory': JusticeTheoryEngine()     # 정의론 (MD 문서 B안)
            }
            
            # 메모리 모드에 따른 선택적 로드
            if self.config.memory_mode in [MemoryMode.HEAVY, MemoryMode.ULTRA, MemoryMode.EXTREME]:
                # HEAVY 이상에서는 모든 엔진 활성화
                self.logger.info("   ✅ 전체 윤리 엔진 활성화 (5개 학파 - MD 문서 B안)")
            elif self.config.memory_mode in [MemoryMode.MEDIUM, MemoryMode.NORMAL]:
                # MEDIUM/NORMAL 모드에서는 핵심 3개만
                limited_engines = ['utilitarianism', 'deontological', 'virtue_ethics']
                self.ethics_engines = {k: v for k, v in self.ethics_engines.items() if k in limited_engines}
                self.logger.info("   ✅ 핵심 윤리 엔진 활성화 (3개 학파)")
            else:
                # LIGHT/MINIMAL에서는 공리주의만
                self.ethics_engines = {'utilitarianism': self.ethics_engines['utilitarianism']}
                self.logger.info("   ✅ 기본 윤리 엔진 활성화 (공리주의만)")
                
        except Exception as e:
            self.logger.error(f"   ❌ 다원적 윤리 체계 로드 실패: {e}")
            raise RuntimeError(f"다원적 윤리 체계 로드 필수 - 실패: {e}")
    
    def emotion_to_bentham_converter(self, emotion_data: Dict) -> Dict:
        """정밀 의미론적 매핑 기반 감정→벤담 변환 v2
        
        개선사항:
        - 6차원 감정과 10차원 벤담의 의미론적 연결
        - 계층적 처리 지원 (공동체>타자>자아)
        - 신경망 어댑터 옵션 (EXTREME 모드)
        """
        
        # 정밀 매퍼 필수 - fallback 없음
        if self.emotion_bentham_mapper is None:
            raise RuntimeError("SemanticEmotionBenthamMapper가 초기화되지 않음 - 정밀 매핑 시스템 필수")
            
        # 계층 레벨 확인
        hierarchy_level = 'self'
        if 'hierarchy' in emotion_data:
            if emotion_data['hierarchy'].get('community'):
                hierarchy_level = 'community'
            elif emotion_data['hierarchy'].get('other'):
                hierarchy_level = 'other'
        
        # 의미론적 매핑 수행
        bentham_params = self.emotion_bentham_mapper.map_with_hierarchy(
            emotion_data, 
            hierarchy_level
        )
        
        # EXTREME 모드에서 신경망 어댑터로 추가 정제
        if self.neural_emotion_adapter is not None and 'scores' in emotion_data:
            scores = emotion_data['scores']
            if isinstance(scores, list) and len(scores) >= 6:
                emotion_tensor = torch.tensor(scores[:6], dtype=torch.float32)
                emotion_tensor = emotion_tensor.unsqueeze(0).to(self.config.device)
                
                with torch.no_grad():
                    neural_output = self.neural_emotion_adapter(emotion_tensor)
                    neural_bentham = neural_output[0].cpu().numpy()
                
                # 의미론적 결과와 신경망 결과 혼합 (7:3 비율)
                for idx, key in enumerate(bentham_params.keys()):
                    if idx < len(neural_bentham):
                        bentham_params[key] = bentham_params[key] * 0.7 + neural_bentham[idx] * 0.3
                        
                self.logger.debug("   신경망 어댑터로 벤담 파라미터 정제 완료")
        
        # 시계열 전파가 이미 적용된 경우 보존
        if 'temporal_duration' in emotion_data:
            bentham_params['duration'] = emotion_data['temporal_duration']
        if 'temporal_fecundity' in emotion_data:
            bentham_params['fecundity'] = emotion_data['temporal_fecundity']
        
        return bentham_params
    
    async def _load_idle_learner(self):
        """유휴 시간 학습 시스템 로드 - MD 문서: 프로덕션 레벨에서 활성화"""
        # MD 문서 사양: 유휴 학습은 주석 처리하여 프로덕션 레벨에서 활성화
        self.logger.info("🌙 유휴 시간 학습 시스템 - 현재 비활성화 (프로덕션에서 활성화)")
        self.idle_learner = None  # 명시적으로 None 설정
        
        # TODO: 프로덕션 레벨에서 아래 코드 활성화
        """
        try:
            from idle_time_learner import HierarchicalIdleLearner
            
            # 유휴 학습기 생성
            self.idle_learner = HierarchicalIdleLearner(
                model=self.unified_model,
                config=self.config
            )
            
            # 학습 데이터 소스 등록
            if self.experience_database:
                self.idle_learner.register_data_source(self.experience_database)
            
            # 유휴 학습 시작
            await self.idle_learner.start()
            
            self.logger.info("   ✅ 유휴 학습 시스템 활성화")
            self.logger.info(f"   - 즉시 학습: 60초 유휴 시")
            self.logger.info(f"   - 단기 학습: 10분 유휴 시")
            self.logger.info(f"   - 중기 학습: 30분 유휴 시")
            self.logger.info(f"   - 장기 학습: 1시간 유휴 시")
            self.logger.info(f"   - 야간 학습: 8시간 유휴 시")
            
        except Exception as e:
            self.logger.warning(f"   ⚠️ 유휴 학습 시스템 로드 실패: {e}")
            self.idle_learner = None
        """
    
    def _print_system_status(self):
        """시스템 상태 출력"""
        self.logger.info("\n📊 시스템 상태:")
        self.logger.info(f"   디바이스: {self.config.device}")
        self.logger.info(f"   메모리 모드: {self.config.memory_mode.value}")
        self.logger.info(f"   UnifiedModel: {'✅' if self.unified_model else '❌'}")
        self.logger.info(f"   Neural Analyzers (368M): {'✅' if self.config.use_neural_analyzers else '❌'}")
        self.logger.info(f"   Advanced Wrappers (112M): {'✅' if self.config.use_advanced_wrappers else '❌'}")
        self.logger.info(f"   DSP Simulator (14M): {'✅' if self.config.use_dsp_simulator else '❌'}")
        self.logger.info(f"   Kalman Filter: {'✅' if self.config.use_kalman_filter else '❌'}")
        self.logger.info(f"   Phase Networks: {'✅' if self.config.use_phase_networks else '❌'}")
        self.logger.info(f"   Regret Circuit: {'✅' if self.config.use_regret_circuit else '❌'}")
        self.logger.info(f"   메타 통합 (40M): {'✅' if self.config.use_meta_integration else '❌'}")
        self.logger.info(f"   반사실 추론 (15M): {'✅' if self.config.use_counterfactual_reasoning else '❌'}")
        self.logger.info(f"   고급 후회 학습 (20M): {'✅' if self.config.use_advanced_regret_learning else '❌'}")
        self.logger.info(f"   워크플로우 관리자: {'✅' if self.config.use_workflow_memory_manager else '❌'}")
        self.logger.info(f"   시계열 전파: {'✅' if self.config.use_temporal_propagation else '❌'}")
        self.logger.info(f"   경험 DB: {'✅' if self.config.use_experience_database else '❌'}")
        self.logger.info(f"   계층적 감정: {'✅' if self.config.use_emotion_hierarchy else '❌'}")
        self.logger.info(f"   유휴 학습: {'✅' if self.idle_learner else '❌'}")
        self.logger.info(f"   LLM 모드: {self.config.llm_mode}")
        self.logger.info(f"   번역기: {'✅' if self.config.use_translator else '❌'}")
    
    async def analyze(self, text: str, **kwargs) -> Dict[str, Any]:
        """텍스트 분석 (모든 모듈 활용)"""
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        try:
            # 캐시 확인 - kwargs를 JSON 문자열로 변환해 해시 가능하게 만듦
            import json
            kwargs_str = json.dumps(kwargs, sort_keys=True, default=str)
            cache_key = f"{text[:50]}_{hash(kwargs_str)}"
            if cache_key in self.cache:
                self.logger.info("   📦 캐시 히트")
                return self.cache[cache_key]
            
            # 한국어 감지 및 번역
            original_text = text
            if self.config.use_translator and self._is_korean(text):
                self.logger.info("   🌐 한국어 감지 - 번역 중...")
                text = self.translator.translate_ko_to_en(text)
                self.logger.info(f"   번역 결과: {text}")
            
            # ========== Phase 0: LLM 초기 분석 (NEW) ==========
            # LLM이 먼저 기초 추론과 시나리오를 제공
            llm_initial_analysis = None
            llm_scenarios = []
            
            if self.config.llm_mode != "none" and hasattr(self, 'advanced_wrappers'):
                self.logger.info("\n   🤖 ========== Phase 0: LLM 초기 분석 ==========")
                self.logger.info(f"   입력: {text}")
                
                # Advanced Emotion Wrapper에서 LLM 사용
                if 'advanced_emotion' in self.advanced_wrappers:
                    try:
                        emotion_wrapper = self.advanced_wrappers['advanced_emotion']
                        # API 모드일 때는 self.llm_engine 사용
                        llm_engine_to_use = None
                        if self.config.llm_mode in ['gpt', 'claude', 'perplexity', 'deepseek', 'mcp'] and hasattr(self, 'llm_engine') and self.llm_engine:
                            llm_engine_to_use = self.llm_engine
                        elif hasattr(emotion_wrapper, 'llm_engine') and emotion_wrapper.llm_engine:
                            llm_engine_to_use = emotion_wrapper.llm_engine
                        
                        if llm_engine_to_use:
                            self.logger.info("   📝 LLM에게 초기 시나리오 생성 요청...")
                            
                            llm_prompt = f"""
Analyze the following situation and provide initial analysis:

Text: "{text}"

Provide:
1. Emotional state analysis (joy, sadness, anger, fear, surprise, disgust, neutral - scores 0-1)
2. Three possible action scenarios the person might take
3. Ethical considerations for each scenario
4. Potential regret factors

Respond in JSON format with keys:
- "emotions": dict of emotion scores
- "scenarios": list of 3 scenarios with "action", "ethical_score", "regret_potential"
- "context": brief context understanding
                            """.strip()
                            
                            # LLM 호출
                            from llm_module.advanced_llm_engine import LLMRequest, TaskComplexity
                            llm_request = LLMRequest(
                                prompt=llm_prompt,
                                task_type="initial_analysis",
                                complexity=TaskComplexity.MODERATE,
                                max_tokens=1000,
                                temperature=0.3
                            )
                            llm_response_obj = await llm_engine_to_use.generate_async(llm_request)
                            
                            # LLMResponse 객체에서 텍스트 추출
                            if llm_response_obj and llm_response_obj.success:
                                llm_response = {'text': llm_response_obj.generated_text}
                            else:
                                llm_response = None
                            
                            if llm_response and 'text' in llm_response:
                                import json
                                try:
                                    llm_initial_analysis = json.loads(llm_response['text'])
                                    self.logger.info("   ✅ LLM 초기 분석 완료")
                                    
                                    # 감정 분석 결과 추출
                                    if 'emotions' in llm_initial_analysis:
                                        self.logger.info(f"   - 감정 상태: {llm_initial_analysis['emotions']}")
                                    
                                    # 시나리오 추출
                                    if 'scenarios' in llm_initial_analysis:
                                        llm_scenarios = llm_initial_analysis['scenarios']
                                        self.logger.info(f"   - 생성된 시나리오: {len(llm_scenarios)}개")
                                        for i, scenario in enumerate(llm_scenarios[:3]):
                                            self.logger.info(f"     시나리오 {i+1}: {scenario.get('action', 'N/A')}")
                                    
                                    # 결과 저장
                                    results['llm_initial'] = llm_initial_analysis
                                    
                                except json.JSONDecodeError:
                                    # JSON 파싱 실패 시 텍스트 그대로 분석
                                    self.logger.warning("   ⚠️ LLM 응답 JSON 파싱 실패")
                                    llm_initial_analysis = {'raw_response': llm_response['text']}
                                    # 텍스트에서 시나리오 추출 시도
                                    if 'scenario' in llm_response['text'].lower():
                                        # 간단한 텍스트 파싱으로 시나리오 추출
                                        lines = llm_response['text'].split('\n')
                                        for line in lines:
                                            if 'scenario' in line.lower() or 'action' in line.lower():
                                                llm_scenarios.append({'action': line.strip()})
                                
                        else:
                            self.logger.info("   ⚠️ LLM 엔진이 없음, 건너뜀")
                            
                    except Exception as e:
                        self.logger.warning(f"   ⚠️ LLM 초기 분석 실패: {e}")
                        # 실패해도 계속 진행
                else:
                    self.logger.info("   ⚠️ Emotion Wrapper가 없음, LLM 초기 분석 건너뜀")
            else:
                self.logger.info("   ℹ️ LLM 모드 비활성화, 초기 분석 건너뜀")
            
            # 토크나이징
            inputs = self._tokenize(text)
            
            results = {}
            
            # LLM 초기 분석 결과가 있으면 results에 포함
            if llm_initial_analysis:
                results['llm_initial'] = llm_initial_analysis
                results['llm_scenarios'] = llm_scenarios
            
            # 워크플로우 관리자 시작
            if self.config.use_workflow_memory_manager and self.workflow_memory_manager:
                from workflow_aware_memory_manager import WorkflowStage
                await self.workflow_memory_manager.prepare_for_workflow(
                    "inference", WorkflowStage.EVALUATION, set()
                )
            
            with torch.no_grad():
                # ========== Phase 1: Red Heart 심층 분석 ==========
                self.logger.info("\n   🧠 ========== Phase 1: Red Heart 심층 분석 ==========")
                
                # LLM 초기 분석 결과를 활용한 Red Heart 추론
                if llm_initial_analysis:
                    self.logger.info("   📌 LLM 초기 분석 결과를 Red Heart에 통합")
                    
                    # LLM 감정 분석을 힌트로 사용
                    if 'emotions' in llm_initial_analysis:
                        results['llm_emotion_hint'] = llm_initial_analysis['emotions']
                        self.logger.info(f"   - LLM 감정 힌트: {llm_initial_analysis['emotions']}")
                    
                    # LLM 시나리오를 반사실 추론에 활용 예정
                    if llm_scenarios:
                        results['llm_scenarios_for_counterfactual'] = llm_scenarios
                        self.logger.info(f"   - 반사실 추론용 LLM 시나리오: {len(llm_scenarios)}개")
                
                # 1. UnifiedModel 백본 추론
                if self.unified_model:
                    self.logger.info("   🧠 UnifiedModel 백본 처리...")
                    # UnifiedModel은 임베딩 텐서를 입력으로 받음
                    
                    # 디바이스 확인 및 조정
                    model_device = next(self.unified_model.parameters()).device
                    if inputs['embeddings'].device != model_device:
                        self.logger.debug(f"   📍 임베딩 디바이스 조정: {inputs['embeddings'].device} → {model_device}")
                        inputs['embeddings'] = inputs['embeddings'].to(model_device)
                    
                    # 1-1. Emotion 태스크
                    emotion_outputs = self.unified_model(
                        x=inputs['embeddings'],  # 임베딩 텐서
                        task='emotion',  # 감정 태스크
                        return_all=True  # 모든 출력 반환
                    )
                    results['unified'] = self._process_unified_outputs(emotion_outputs, task='emotion')
                    
                    # 1-2. Bentham 태스크 - 학습된 bentham_head 사용
                    self.logger.info("   ⚖️ Bentham 윤리 계산 (학습된 27M 모델)...")
                    bentham_outputs = self.unified_model(
                        x=inputs['embeddings'],
                        task='bentham',  # bentham 태스크
                        return_all=True
                    )
                    bentham_results = self._process_unified_outputs(bentham_outputs, task='bentham')
                    results['bentham'] = bentham_results.get('bentham', {})
                
                # ========== Phase 2: 감정 처리 (계층적) ==========
                emotion_data = results.get('unified', {}).get('emotion', {})
                
                # 2-1. EmotionEthicsRegretCircuit 통합 처리
                circuit_result = None
                circuit_context_saved = None  # Circuit 재실행을 위해 컨텍스트 저장
                if self.config.use_emotion_hierarchy and self.emotion_hierarchy_processor:
                    self.logger.info("   🎭 감정-윤리-후회 통합 회로 처리 (초기 시도)...")
                    # CircuitDecisionContext 생성 - 풍부한 컨텍스트 제공
                    from emotion_ethics_regret_circuit import CircuitDecisionContext
                    
                    # 이해관계자 추출 (텍스트에서 언급된 대상들)
                    stakeholders = []
                    if "친구" in text:
                        stakeholders.append("친구")
                    if "가족" in text:
                        stakeholders.append("가족")
                    if "동료" in text or "회사" in text:
                        stakeholders.append("동료")
                    if not stakeholders:
                        stakeholders = ["타인", "사회"]  # 기본 이해관계자
                    
                    # emotion_data가 dict인 경우 EmotionData로 변환
                    from data_models import EmotionData, EmotionState, EmotionIntensity
                    self_emotion = None
                    if emotion_data:
                        if isinstance(emotion_data, dict):
                            emotion_id = emotion_data.get('emotion', 0)
                            primary_emotion = EmotionState(emotion_id) if emotion_id in [e.value for e in EmotionState] else EmotionState.NEUTRAL
                            intensity_val = emotion_data.get('intensity', 3)
                            intensity = EmotionIntensity(intensity_val) if intensity_val in [i.value for i in EmotionIntensity] else EmotionIntensity.MODERATE
                            
                            self_emotion = EmotionData(
                                primary_emotion=primary_emotion,
                                intensity=intensity,
                                arousal=emotion_data.get('arousal', 0.0),
                                valence=emotion_data.get('valence', 0.0),
                                dominance=emotion_data.get('dominance', 0.0),
                                confidence=emotion_data.get('confidence', 0.5),
                                language='ko'
                            )
                        else:
                            self_emotion = emotion_data
                    
                    circuit_context = CircuitDecisionContext(
                        scenario_text=text,
                        proposed_action="상황 분석 및 최적 응답 생성",
                        stakeholders=stakeholders,
                        social_context={
                            'impact_scope': 'personal' if len(stakeholders) < 3 else 'community',
                            'keywords': text.split()[:5],  # 주요 키워드
                            'urgency': 0.5  # 기본 긴급도
                        },
                        temporal_urgency=0.5,
                        self_emotion=self_emotion
                    )
                    
                    # Circuit 컨텍스트를 저장 (후반 재실행용)
                    circuit_context_saved = circuit_context
                    
                    # GPU 메모리 체크 - 부족하면 빠른 fallback
                    if torch.cuda.is_available():
                        gpu_free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                        gpu_free_gb = gpu_free / (1024**3)
                        if gpu_free_gb < 2.0:  # 2GB 미만이면 Circuit skip
                            self.logger.warning(f"   ⚠️ GPU 메모리 부족 ({gpu_free_gb:.1f}GB), Circuit 후반 실행 예약 (의도적 fallback)")
                            circuit_result = None
                        else:
                            try:
                                # 짧은 타임아웃으로 시도 (5초)
                                import asyncio
                                circuit_result = await asyncio.wait_for(
                                    self.emotion_hierarchy_processor.process_ethical_decision(circuit_context),
                                    timeout=5.0
                                )
                            except asyncio.TimeoutError:
                                self.logger.warning("   ⏱️ Circuit 초기 시도 타임아웃, 후반 재실행 예약 (의도적 fallback)")
                                circuit_result = None
                    else:
                        try:
                            circuit_result = await self.emotion_hierarchy_processor.process_ethical_decision(circuit_context)
                            
                            if circuit_result:
                                # Circuit 결과를 워크플로우에 통합
                                # 1. 감정 데이터 통합
                                if hasattr(circuit_result, 'integrated_emotion'):
                                    emotion_data['circuit_integrated'] = {
                                        'primary': circuit_result.integrated_emotion.primary_emotion.value,
                                        'intensity': circuit_result.integrated_emotion.intensity.value,
                                        'confidence': circuit_result.integrated_emotion.confidence
                                    }
                                
                                # 2. 윤리적 가치 통합
                                if hasattr(circuit_result, 'ethical_values'):
                                    results['circuit_ethics'] = circuit_result.ethical_values
                                
                                # 3. 예측된 후회 통합
                                if hasattr(circuit_result, 'predicted_regret'):
                                    results['circuit_regret'] = circuit_result.predicted_regret
                                
                                # 4. 추론 과정 저장
                                if hasattr(circuit_result, 'reasoning_trace'):
                                    results['circuit_reasoning'] = circuit_result.reasoning_trace
                                    
                                self.logger.info(f"   ✅ Circuit 처리 완료 (신뢰도: {getattr(circuit_result, 'confidence', 0):.2f})")
                        except Exception as e:
                            self.logger.warning(f"   ⚠️ Circuit 초기 처리 실패, 후반 재실행 예약 (의도적 fallback): {e}")
                            circuit_result = None
                
                # 2-2. DSP로 감정 신호 처리
                if self.config.use_dsp_simulator and self.dsp_simulator:
                    self.logger.info("   📡 DSP 감정 신호 처리...")
                    # DSP Simulator는 384차원 입력을 받음 - embedding에서 가져오기
                    unified_result = results.get('unified', {})
                    if 'embedding' in unified_result and unified_result['embedding'] is not None:
                        # embedding이 있으면 사용 (이미 384차원)
                        dsp_input = unified_result['embedding'].unsqueeze(0) if unified_result['embedding'].dim() == 1 else unified_result['embedding']
                    else:
                        # embedding이 없으면 감정 데이터를 384차원으로 확장
                        emotion_tensor = torch.zeros(1, 7).to(self.config.device)
                        if isinstance(emotion_data, dict):
                            emotion_keys = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral']
                            for i, key in enumerate(emotion_keys):
                                if key in emotion_data:
                                    emotion_tensor[0, i] = float(emotion_data[key]) if isinstance(emotion_data[key], (int, float)) else 0.5
                        
                        # 7차원을 384차원으로 투영 (선형 변환)
                        # DSP simulator의 device 확인 및 일치
                        dsp_device = next(self.dsp_simulator.parameters()).device
                        if not hasattr(self, 'emotion_to_dsp_projection'):
                            self.emotion_to_dsp_projection = nn.Linear(7, 384).to(dsp_device)
                        else:
                            self.emotion_to_dsp_projection = self.emotion_to_dsp_projection.to(dsp_device)
                        
                        # emotion_tensor를 DSP simulator와 같은 device로 이동
                        emotion_tensor = emotion_tensor.to(dsp_device)
                        dsp_input = self.emotion_to_dsp_projection(emotion_tensor)
                    
                    dsp_result = self.dsp_simulator.forward(dsp_input)
                    emotion_data['dsp_processed'] = dsp_result
                    
                    if self.config.use_kalman_filter and self.kalman_filter:
                        self.logger.info("   🔄 Kalman 필터링...")
                        emotion_data['kalman_filtered'] = self.kalman_filter.update(dsp_result)
                
                results['emotion'] = emotion_data
                
                # ========== Phase 3: 감정 → 벤담 직접 변환 ==========
                self.logger.info("   🔀 감정 → 벤담 직접 변환...")
                bentham_params = self.emotion_to_bentham_converter(emotion_data)
                
                # 3-1. 시계열 전파 → 벤담 지속성 통합
                if self.config.use_temporal_propagation and self.temporal_propagator:
                    self.logger.info("   ⏰ 시계열 전파 분석...")
                    # analyze_temporal_patterns 메서드 사용
                    temporal_patterns = self.temporal_propagator.analyze_temporal_patterns()
                    # 시계열 영향 추출
                    temporal_impact = {
                        'long_term_effect': temporal_patterns.get('LONG_TERM', {}).get('event_frequency', {}).get('average', 1.0),
                        'cascade_potential': temporal_patterns.get('cross_scale', {}).get('cascade_probability', 0.5),
                        'patterns': temporal_patterns
                    }
                    # 시계열 영향을 벤담 파라미터에 직접 반영
                    bentham_params['duration'] = temporal_impact.get('long_term_effect', bentham_params['duration'])
                    bentham_params['fecundity'] = temporal_impact.get('cascade_potential', bentham_params['fecundity'])
                    results['temporal_impact'] = temporal_impact
                
                # 3-2. 벤담 계산
                if 'bentham' in results.get('unified', {}):
                    # UnifiedModel의 벤담 헤드 결과와 병합
                    unified_bentham = results['unified']['bentham']
                    for key in bentham_params:
                        if key in unified_bentham:
                            # 가중 평균
                            bentham_params[key] = (bentham_params[key] + unified_bentham[key]) / 2
                
                results['bentham'] = bentham_params
                
                # ========== Phase 4: 반사실 추론 ==========
                if self.config.use_counterfactual_reasoning and self.counterfactual_reasoning:
                    self.logger.info("\n   💭 ========== Phase 4: 반사실 추론 ==========")
                    
                    # LLM 시나리오가 있으면 반사실 추론에 활용
                    if 'llm_scenarios_for_counterfactual' in results:
                        self.logger.info("   📌 LLM 시나리오를 반사실 추론에 통합")
                        
                    # AdvancedCounterfactualReasoning의 실제 메서드 사용
                    base_situation = {
                        'text': text,
                        'emotion_results': emotion_data,  # emotion_data 사용
                        'bentham_params': bentham_params,
                        'circuit_results': circuit_result,  # circuit_result 사용
                        'llm_scenarios': results.get('llm_scenarios_for_counterfactual', [])  # LLM 시나리오 추가
                    }
                    counterfactuals = await self.counterfactual_reasoning.analyze_counterfactual_scenarios(
                        base_situation=base_situation,
                        options={'num_hypotheses': 3, 'max_actions_per_hypothesis': 3}
                    )
                    results['counterfactuals'] = counterfactuals
                    
                    # LLM 시나리오와 반사실 추론 결과 통합
                    if counterfactuals and 'llm_scenarios_for_counterfactual' in results:
                        self.logger.info("   🔀 LLM 시나리오와 반사실 추론 결과 통합")
                        # 두 결과를 합쳐서 더 풍부한 대안 생성
                else:
                    counterfactuals = None  # LIGHT 모드에서는 반사실 추론 스킵
                
                # ========== Phase 5: 후회 계산 (이중 시스템) ==========
                regret_results = {}
                
                # 5-1. UnifiedModel RegretHead
                if 'regret' in results.get('unified', {}):
                    regret_results['unified'] = results['unified']['regret']
                
                # 5-2. 고급 후회 학습 시스템
                if self.config.use_advanced_regret_learning and self.advanced_regret_learning:
                    self.logger.info("   😔 고급 후회 학습...")
                    advanced_regret = await self.advanced_regret_learning.analyze(
                        counterfactuals=counterfactuals,
                        bentham_score=results.get('bentham', {})
                    )
                    regret_results['advanced'] = advanced_regret
                
                # 5-3. 경험 DB 검색
                if self.config.use_experience_database and self.experience_database:
                    self.logger.info("   💾 경험 데이터베이스 검색...")
                    # ExperienceQuery 생성
                    from advanced_experience_database import ExperienceQuery
                    query = ExperienceQuery(
                        query_text=text,
                        emotion_state=emotion_data if isinstance(emotion_data, dict) else None,
                        max_results=5
                    )
                    similar_experiences = await self.experience_database.search_experiences(query)
                    regret_results['experience_based'] = similar_experiences
                
                results['regret'] = regret_results
                
                # ========== Phase 6: 추가 분석 ==========
                # 6-1. Neural Analyzers (368M)
                if self.config.use_neural_analyzers and self.neural_analyzers:
                    self.logger.info("   🧠 Neural Analyzers 분석...")
                    neural_results = {}
                    # UnifiedModel 출력에서 hidden_states 추출
                    hidden_states = None
                    if 'unified' in results and 'hidden_states' in results['unified']:
                        hidden_states = results['unified']['hidden_states']
                    elif 'unified' in results and 'embedding' in results['unified']:
                        # embedding을 hidden_states로 사용
                        hidden_states = results['unified']['embedding']
                    elif 'embeddings' in inputs:
                        # 입력 임베딩 사용
                        hidden_states = inputs['embeddings']
                    
                    if hidden_states is not None:
                        # 차원 확인 및 조정
                        if isinstance(hidden_states, torch.Tensor):
                            # 4D tensor를 2D 또는 3D로 변환
                            if hidden_states.dim() == 4:
                                # [batch, seq, heads, dim] -> [batch, seq*heads*dim]
                                batch_size = hidden_states.shape[0]
                                hidden_states = hidden_states.view(batch_size, -1)
                                self.logger.info(f"   📐 4D tensor를 2D로 변환: {hidden_states.shape}")
                            elif hidden_states.dim() == 3:
                                # [batch, seq, dim] -> [batch, seq*dim] 
                                batch_size = hidden_states.shape[0]
                                hidden_states = hidden_states.view(batch_size, -1)
                                self.logger.info(f"   📐 3D tensor를 2D로 변환: {hidden_states.shape}")
                            elif hidden_states.dim() == 1:
                                # [dim] -> [1, dim]
                                hidden_states = hidden_states.unsqueeze(0)
                                self.logger.info(f"   📐 1D tensor를 2D로 변환: {hidden_states.shape}")
                            
                            # 768차원으로 맞추기 위한 프로젝션
                            if hidden_states.shape[-1] != 768:
                                # 선형 프로젝션으로 차원 맞추기
                                projection = nn.Linear(hidden_states.shape[-1], 768).to(hidden_states.device)
                                hidden_states = projection(hidden_states)
                                self.logger.info(f"   📐 차원 프로젝션: {hidden_states.shape[-1]} -> 768")
                        
                        for name, analyzer in self.neural_analyzers.items():
                            try:
                                # analyzer의 device 확인
                                analyzer_device = next(analyzer.parameters()).device
                                
                                # 프로젝션 어댑터가 있으면 사용 (896차원 가중치 복원된 경우)
                                if hasattr(self, 'neural_projection_adapters') and name in self.neural_projection_adapters:
                                    # 768 -> 896 프로젝션
                                    # hidden_states를 analyzer device로 이동
                                    hidden_states_on_device = hidden_states.to(analyzer_device)
                                    # 프로젝션 어댑터도 같은 device로 이동
                                    self.neural_projection_adapters[name] = self.neural_projection_adapters[name].to(analyzer_device)
                                    projected_hidden = self.neural_projection_adapters[name](hidden_states_on_device)
                                    neural_results[name] = analyzer(projected_hidden)
                                else:
                                    # 768차원 그대로 사용
                                    # hidden_states를 analyzer device로 이동
                                    hidden_states_on_device = hidden_states.to(analyzer_device)
                                    neural_results[name] = analyzer(hidden_states_on_device)
                            except Exception as e:
                                self.logger.warning(f"   ⚠️ {name} analyzer 실패: {e}")
                        
                        if neural_results:
                            results['neural_analysis'] = neural_results
                    else:
                        self.logger.warning("   ⚠️ Neural Analyzers: hidden_states 없음, 스킵")
                
                # 6-2. Advanced Wrappers (112M)
                if self.config.use_advanced_wrappers and self.advanced_wrappers:
                    self.logger.info("   🎯 Advanced Wrappers 분석...")
                    wrapper_results = {}
                    
                    # Advanced Wrappers는 텐서 입력을 기대함
                    # hidden_states가 이미 768차원으로 프로젝션되어 있으면 사용
                    wrapper_input = None
                    if hidden_states is not None and isinstance(hidden_states, torch.Tensor):
                        wrapper_input = hidden_states
                    elif 'embeddings' in inputs:
                        # embeddings를 사용 (384차원 -> 768차원 프로젝션 필요)
                        wrapper_input = inputs['embeddings']
                        if wrapper_input.dim() == 1:
                            wrapper_input = wrapper_input.unsqueeze(0)
                        if wrapper_input.shape[-1] != 768:
                            if not hasattr(self, 'wrapper_projection'):
                                self.wrapper_projection = nn.Linear(wrapper_input.shape[-1], 768).to(self.config.device)
                            wrapper_input = self.wrapper_projection(wrapper_input)
                    
                    if wrapper_input is not None:
                        for name, wrapper in self.advanced_wrappers.items():
                            try:
                                if hasattr(wrapper, 'forward'):
                                    wrapper_results[name] = wrapper(wrapper_input)
                            except Exception as e:
                                self.logger.warning(f"   ⚠️ {name} wrapper 실패: {e}")
                        results['wrapper_analysis'] = wrapper_results
                    else:
                        self.logger.warning("   ⚠️ Advanced Wrappers: 적절한 입력 없음, 스킵")
                
                # 6-3. Phase Networks (타자-자아-공동체 계층적 감정 처리)
                if self.config.use_phase_networks and self.phase_networks:
                    self.logger.info("   🔄 Phase Networks 처리...")
                    phase_results = {}
                    
                    # inputs는 dict이므로 embeddings 추출
                    input_embeddings = inputs['embeddings']
                    # [1, seq_len, hidden_dim] -> [1, hidden_dim] (첫 번째 시퀀스만 사용)
                    if input_embeddings.dim() == 3:
                        input_embeddings = input_embeddings[:, 0, :]  # [1, hidden_dim]
                    
                    # Phase0: 타자→자신 감정 투영 (후회를 통한 학습)
                    if 'phase0' in self.phase_networks:
                        # 감정 데이터가 있으면 타자 관점으로 변환
                        if 'emotion' in results and isinstance(results['emotion'], dict):
                            # 감정 scores를 타자 감정으로 사용
                            if 'scores' in results['emotion']:
                                emotion_scores = results['emotion']['scores']
                                # scores가 dict인 경우 리스트로 변환
                                if isinstance(emotion_scores, dict):
                                    emotion_keys = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral']
                                    emotion_scores = [emotion_scores.get(k, 0.0) for k in emotion_keys]
                                other_emotion = torch.tensor(emotion_scores, dtype=torch.float32).unsqueeze(0).to(self.config.device)
                            else:
                                # 7차원 감정 벡터 구성
                                emotion_keys = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral']
                                emotion_values = []
                                for k in emotion_keys:
                                    val = results['emotion'].get(k, 0.0)
                                    # dict나 다른 타입이면 float로 변환
                                    if isinstance(val, dict):
                                        val = val.get('score', 0.0) if 'score' in val else 0.0
                                    emotion_values.append(float(val))
                                other_emotion = torch.tensor(
                                    emotion_values,
                                    dtype=torch.float32
                                ).unsqueeze(0).to(self.config.device)
                            
                            # Phase0로 타자 감정을 자신 감정으로 투영 (embeddings 전달)
                            # Phase Networks의 device와 일치시킴
                            phase_device = next(self.phase_networks['phase0'].parameters()).device
                            other_emotion = other_emotion.to(phase_device)
                            input_embeddings_phase = input_embeddings.to(phase_device)
                            phase_results['phase0_projection'] = self.phase_networks['phase0'](other_emotion, input_embeddings_phase)
                            self.logger.info(f"      Phase0: 타자→자신 투영 완료 {phase_results['phase0_projection'].shape}")
                        else:
                            # 감정 데이터 없으면 임베딩 직접 사용
                            phase_device = next(self.phase_networks['phase0'].parameters()).device
                            input_embeddings_phase = input_embeddings.to(phase_device)
                            phase_results['phase0_projection'] = self.phase_networks['phase0'](input_embeddings_phase)
                    
                    # Phase2: 개인→공동체 감정 패턴
                    if 'phase2' in self.phase_networks:
                        # 배치 내 여러 샘플을 개인들로 간주
                        # input_embeddings: [batch_size, 768] → [1, batch_size, 768] (batch를 individuals로)
                        phase_device = next(self.phase_networks['phase2'].parameters()).device
                        if input_embeddings.dim() == 2 and input_embeddings.shape[0] > 1:
                            community_input = input_embeddings.unsqueeze(0)  # [1, batch_size, 768]
                        else:
                            # 단일 샘플인 경우 복제하여 가상의 개인들 생성
                            community_input = input_embeddings.repeat(5, 1).unsqueeze(0)  # [1, 5, 768]
                        
                        community_input = community_input.to(phase_device)
                        phase_results['phase2_community'] = self.phase_networks['phase2'](
                            community_input,
                            cultural_context='korean'  # 한국 문화 맥락
                        )
                        self.logger.info(f"      Phase2: 공동체 패턴 추출 완료 {phase_results['phase2_community'].shape}")
                    
                    # Hierarchical Integration: 계층적 통합
                    if 'hierarchical' in self.phase_networks:
                        # Phase Networks의 device 확인
                        phase_device = next(self.phase_networks['hierarchical'].parameters()).device
                        
                        # input_embeddings가 768차원인데 HierarchicalEmotionIntegrator는 896차원 기대
                        # 768 → 896 패딩 또는 프로젝션
                        if input_embeddings.shape[-1] == 768:
                            # 128차원 패딩 추가 (후회/메타 정보용 공간)
                            padded_features = F.pad(input_embeddings, (0, 128), mode='constant', value=0)
                        else:
                            padded_features = input_embeddings
                        
                        # padded_features를 device로 이동
                        padded_features = padded_features.to(phase_device)
                        
                        # Phase0, Phase2 출력 전달 (이미 phase_device에 있음)
                        phase0_out = phase_results.get('phase0_projection')
                        phase2_out = phase_results.get('phase2_community')
                        
                        integrated = self.phase_networks['hierarchical'](
                            padded_features,
                            phase0_out=phase0_out,
                            phase2_out=phase2_out
                        )
                        phase_results['hierarchical_integration'] = integrated
                        self.logger.info(f"      Hierarchical: 계층적 통합 완료 {integrated.shape}")
                        
                        # 통합된 특징을 다시 768차원으로 축소 (다른 모듈과 호환성)
                        if integrated.shape[-1] == 896:
                            # 896 → 768 프로젝션
                            if not hasattr(self, 'phase_output_projector'):
                                self.phase_output_projector = nn.Linear(896, 768).to(self.config.device)
                            phase_results['integrated_768'] = self.phase_output_projector(integrated)
                    
                    results['phase_analysis'] = phase_results
                
                # ========== Phase 7: 메타 통합 ==========
                if self.config.use_meta_integration and self.meta_integration:
                    self.logger.info("   🔮 메타 통합 시스템...")
                    
                    # 각 모듈의 출력을 텐서로 변환
                    head_tensors = {}
                    
                    # 1. Emotion tensor 추출
                    if 'emotion' in results and isinstance(results['emotion'], dict):
                        emotion_values = []
                        
                        # scores가 있으면 사용
                        if 'scores' in results['emotion']:
                            emotion_scores = results['emotion']['scores']
                            if isinstance(emotion_scores, list):
                                # [7] → [1,7]로 배치 차원 추가 (o3 제안)
                                head_tensors['emotion'] = torch.tensor(emotion_scores, dtype=torch.float32).unsqueeze(0).to(self.config.device)
                            elif isinstance(emotion_scores, torch.Tensor):
                                # 텐서도 1D면 배치 차원 추가
                                if emotion_scores.dim() == 1:
                                    head_tensors['emotion'] = emotion_scores.unsqueeze(0).to(self.config.device)
                                else:
                                    head_tensors['emotion'] = emotion_scores.to(self.config.device)
                        # scores가 없으면 감정 키들에서 값 추출
                        else:
                            emotion_keys = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral']
                            for key in emotion_keys:
                                if key in results['emotion']:
                                    value = results['emotion'][key]
                                    if isinstance(value, (int, float)):
                                        emotion_values.append(float(value))
                                    else:
                                        emotion_values.append(0.0)
                            
                            if emotion_values:
                                # [7] → [1,7]로 배치 차원 추가 (o3 제안)
                                head_tensors['emotion'] = torch.tensor(emotion_values, dtype=torch.float32).unsqueeze(0).to(self.config.device)
                    
                    # 2. Bentham tensor 추출  
                    if 'bentham' in results and isinstance(results['bentham'], dict):
                        bentham_values = []
                        # 벤담의 7개 쾌락 변수 순서대로 추출
                        bentham_keys = ['intensity', 'duration', 'certainty', 'propinquity', 
                                      'fecundity', 'purity', 'extent']
                        for key in bentham_keys:
                            if key in results['bentham']:
                                bentham_values.append(float(results['bentham'][key]))
                            else:
                                # 필수 키가 없으면 에러 로깅
                                self.logger.error(f"   ❌ bentham 결과에 필수 키 누락: {key}")
                        
                        if len(bentham_values) == 7:  # 7개 모두 있어야 유효
                            head_tensors['bentham'] = torch.tensor(bentham_values, dtype=torch.float32).unsqueeze(0).to(self.config.device)
                        else:
                            self.logger.error(f"   ❌ bentham 텐서 생성 실패: {len(bentham_values)}/7개 값만 수집됨")
                    
                    # 3. Neural analysis tensor 추출
                    if 'neural_analysis' in results and results['neural_analysis']:
                        # neural_analysis가 dict이면 첫 번째 analyzer의 출력 사용
                        if isinstance(results['neural_analysis'], dict):
                            for analyzer_name, analyzer_output in results['neural_analysis'].items():
                                if isinstance(analyzer_output, torch.Tensor):
                                    head_tensors['neural'] = analyzer_output.to(self.config.device)
                                    break
                    
                    # 4. Hidden states 또는 embedding 추출
                    if 'unified' in results:
                        if 'hidden_states' in results['unified'] and isinstance(results['unified']['hidden_states'], torch.Tensor):
                            head_tensors['hidden'] = results['unified']['hidden_states'].to(self.config.device)
                        elif 'embedding' in results['unified'] and isinstance(results['unified']['embedding'], torch.Tensor):
                            head_tensors['embedding'] = results['unified']['embedding'].to(self.config.device)
                    
                    # 5. 텐서 수집 상태 로깅
                    self.logger.info(f"   📊 수집된 텐서: {list(head_tensors.keys())}")
                    for key, tensor in head_tensors.items():
                        if isinstance(tensor, torch.Tensor):
                            is_valid = tensor.abs().sum().item() > 0
                            self.logger.info(f"      - {key}: shape={tensor.shape}, sum={tensor.abs().sum().item():.3f}, valid={is_valid}")
                    
                    # 6. 필수 텐서 체크 - GPT 제안대로 명시적 예외 발생
                    required = {'emotion', 'bentham'}
                    missing = [k for k in required if k not in head_tensors or head_tensors[k] is None]
                    
                    if missing:
                        # 근본 원인 로깅
                        self.logger.error(f"   ❌ 메타 통합 차단: 필수 텐서 누락: {missing}")
                        if 'emotion' in missing:
                            self.logger.error("      emotion 텐서 생성 실패 - Advanced Emotion Analyzer 출력 확인 필요")
                        if 'bentham' in missing:
                            self.logger.error("      bentham 텐서 생성 실패 - Advanced Bentham Calculator 출력 확인 필요")
                        raise RuntimeError(f"Meta-Integration blocked. Missing required tensors: {missing}")
                    
                    # 텐서가 있어도 유효하지 않으면(모두 0) 에러
                    invalid = [k for k in required if head_tensors[k].abs().sum().item() == 0]
                    if invalid:
                        self.logger.error(f"   ❌ 메타 통합 차단: 무효한 텐서(모두 0): {invalid}")
                        raise RuntimeError(f"Meta-Integration blocked. Invalid tensors (all zeros): {invalid}")
                    
                    # 모든 검증 통과 시에만 통합 실행
                    try:
                        integrated_result = await self.meta_integration.integrate_head_outputs(head_tensors)
                        results['meta_integrated'] = integrated_result
                        self.logger.info("   ✅ 메타 통합 완료")
                    except Exception as e:
                        self.logger.error(f"   ❌ 메타 통합 실행 중 실패: {e}")
                        raise
                    
                
                # ========== Phase 8: LLM 보강 (선택적) ==========
                if self.config.llm_mode != "none" and self.llm_engine:
                    api_modes = ['gpt', 'claude', 'perplexity', 'deepseek']
                    
                    if self.config.llm_mode in api_modes:
                        # API 모드 - GPU 스왑 불필요
                        self.logger.info(f"   🌐 LLM API 보강 ({self.config.llm_mode})...")
                    else:
                        # 로컬 모드 - GPU 스왑 필요
                        self.logger.info(f"   🤖 LLM 보강 ({self.config.llm_mode})...")
                        
                        # LLM 실행 전 Red Heart 모듈들을 RAM으로 스왑하여 GPU 메모리 확보
                        self.logger.info("   🔄 LLM을 위해 Red Heart 모듈들을 RAM으로 스왑...")
                    
                    # 1. 백본과 헤드를 RAM으로 이동
                    if self.unified_model:
                        self.unified_model.to('cpu')
                        self.logger.info("      - UnifiedModel → RAM")
                    
                    # 2. Neural Analyzers를 RAM으로 이동
                    if self.neural_analyzers:
                        for name, module in self.neural_analyzers.items():
                            module.to('cpu')
                        self.logger.info("      - Neural Analyzers → RAM")
                    
                    # 3. Advanced Wrappers를 RAM으로 이동
                    if self.advanced_wrappers:
                        for name, wrapper in self.advanced_wrappers.items():
                            if hasattr(wrapper, 'to'):
                                wrapper.to('cpu')
                        self.logger.info("      - Advanced Wrappers → RAM")
                    
                    # 4. Phase Networks를 RAM으로 이동
                    if self.phase_networks:
                        for name, network in self.phase_networks.items():
                            network.to('cpu')
                        self.logger.info("      - Phase Networks → RAM")
                    
                    # 5. DSP와 기타 모듈들을 RAM으로 이동
                    if self.dsp_simulator:
                        self.dsp_simulator.to('cpu')
                        self.logger.info("      - DSP Simulator → RAM")
                    
                    if self.kalman_filter:
                        self.kalman_filter.to('cpu')
                        self.logger.info("      - Kalman Filter → RAM")
                    
                    # 6. GPU 캐시 정리
                    torch.cuda.empty_cache()
                    if torch.cuda.is_available():
                        gpu_free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                        self.logger.info(f"   ✅ GPU 메모리 확보 완료: {gpu_free/1024**3:.1f}GB 사용 가능")
                    
                    # ========== Circuit 재실행 (GPU 여유 상태) ==========
                    if circuit_result is None and circuit_context_saved and self.emotion_hierarchy_processor:
                        self.logger.info("   🔄 Circuit 재실행 (GPU 메모리 확보됨)...")
                        try:
                            # GPU에 벤담 계산기 등 로드
                            circuit_result = await self.emotion_hierarchy_processor.process_ethical_decision(circuit_context_saved)
                            
                            if circuit_result:
                                # Circuit 결과를 워크플로우에 통합
                                # 1. 감정 데이터 통합
                                if hasattr(circuit_result, 'integrated_emotion'):
                                    results['circuit_integrated'] = {
                                        'primary': circuit_result.integrated_emotion.primary_emotion.value,
                                        'intensity': circuit_result.integrated_emotion.intensity.value,
                                        'confidence': circuit_result.integrated_emotion.confidence
                                    }
                                
                                # 2. 윤리적 가치 통합
                                if hasattr(circuit_result, 'ethical_values'):
                                    results['circuit_ethics'] = circuit_result.ethical_values
                                
                                # 3. 후회 학습 데이터 통합
                                if hasattr(circuit_result, 'regret_metrics'):
                                    results['circuit_regret'] = circuit_result.regret_metrics
                                
                                self.logger.info(f"   ✅ Circuit 재실행 성공 (신뢰도: {getattr(circuit_result, 'confidence', 0):.2f})")
                        except Exception as e:
                            self.logger.warning(f"   ⚠️ Circuit 재실행도 실패, 스킵: {e}")
                    
                    # LLMRequest 생성
                    from llm_module.advanced_llm_engine import LLMRequest, TaskComplexity
                    
                    # 결과를 프롬프트에 포함
                    context_summary = []
                    if 'emotion' in results:
                        context_summary.append(f"감정 분석: {results['emotion']}")
                    if 'bentham' in results:
                        context_summary.append(f"벤담 점수: {results['bentham']}")
                    if 'counterfactual' in results:
                        context_summary.append(f"반사실적 추론 완료")
                    
                    enhance_prompt = f"""텍스트: {text}
                    
분석 결과:
{chr(10).join(context_summary)}

위 분석 결과를 바탕으로 텍스트에 대한 심층 윤리적 평가를 제공하세요."""
                    
                    llm_request = LLMRequest(
                        prompt=enhance_prompt,
                        task_type="enhancement",
                        complexity=TaskComplexity.MODERATE,
                        context={'analysis_results': results}
                    )
                    
                    # generate_async 호출
                    llm_response = await self.llm_engine.generate_async(llm_request)
                    results['llm_enhanced'] = {
                        'text': llm_response.generated_text,  # text가 아니라 generated_text
                        'confidence': llm_response.confidence
                    }
            
            # 통합 점수 계산
            results['integrated_score'] = self._calculate_integrated_score(results)
            results['confidence'] = self._calculate_confidence(results)
            results['processing_time'] = time.time() - start_time
            
            # 워크플로우 관리자 종료
            if self.config.use_workflow_memory_manager and self.workflow_memory_manager:
                self.workflow_memory_manager.complete_workflow("inference")
            
            # 경험 DB 저장 (MD 문서 사양: 분석 결과 저장)
            if self.config.use_experience_database and self.experience_database:
                try:
                    # 경험 데이터 구성 (dict만 허용, 다른 타입은 변환)
                    import json
                    
                    def to_serializable(obj):
                        """객체를 직렬화 가능한 형태로 변환"""
                        if isinstance(obj, dict):
                            # dict 안의 각 값도 재귀적으로 변환
                            result = {}
                            for k, v in obj.items():
                                # key도 문자열로 변환
                                key = str(k) if not isinstance(k, str) else k
                                result[key] = to_serializable(v)
                            return result
                        elif isinstance(obj, (list, tuple)):
                            return [to_serializable(item) for item in obj]
                        elif hasattr(obj, '__dict__'):
                            return to_serializable(obj.__dict__)
                        elif isinstance(obj, (str, int, float, bool, type(None))):
                            return obj
                        else:
                            return str(obj)
                    
                    experience_data = {
                        'timestamp': time.time(),
                        'text': text,
                        'emotion': to_serializable(results.get('emotion', {})),
                        'bentham': to_serializable(results.get('bentham', {})),
                        'regret': to_serializable(results.get('regret', {})),
                        'integrated_score': float(results.get('integrated_score', 0)),
                        'confidence': float(results.get('confidence', 0)),
                        'meta_integrated': to_serializable(results.get('meta_integrated', {}))
                    }
                    
                    # 경험 저장
                    await self.experience_database.store_experience(
                        experience_text=text,
                        metadata=experience_data,  # experience_data 자체가 메타데이터
                        category='general',
                        importance_score=experience_data.get('confidence', 0.5)
                    )
                    self.logger.info("   💾 경험 데이터베이스에 저장 완료")
                    
                except Exception as e:
                    self.logger.warning(f"   ⚠️ 경험 DB 저장 실패: {e}")
            
            # 캐싱
            if len(self.cache) < self.config.cache_size:
                self.cache[cache_key] = results
            
            self.stats['successful'] += 1
            self.logger.info(f"   ✅ 분석 완료 ({results['processing_time']:.2f}초)")
            
            return results
            
        except Exception as e:
            self.stats['failed'] += 1
            self.logger.error(f"   ❌ 분석 실패: {e}")
            # 정확한 에러 위치 추적을 위한 traceback 추가
            import traceback
            self.logger.error(f"   📍 에러 발생 위치:\n{traceback.format_exc()}")
            return {
                'error': str(e),
                'processing_time': time.time() - start_time,
                'traceback': traceback.format_exc()
            }
    
    async def analyze_ethical_dilemma(self, llm_scenarios: List[str]) -> Dict[str, Any]:
        """비선형 윤리적 딜레마 분석 워크플로우
        
        MD 문서 사양에 따른 구현:
        1. LLM이 제시한 n개 시나리오를 3뷰로 확장 (n × 3)
        2. 각 시나리오별 감정/윤리 평가
        3. 후회 분석으로 추가 시나리오 생성
        4. 정합성 판단 (시스템 + LLM)
        5. 상위 2개 시나리오 선정
        """
        start_time = time.time()
        all_results = []
        
        self.logger.info(f"🎯 윤리적 딜레마 분석 시작 - {len(llm_scenarios)}개 시나리오")
        
        try:
            # Phase 1: 3뷰 시스템 즉시 적용
            self.logger.info("   Phase 1: 3뷰 시스템 적용 (낙관/중도/비관)")
            for idx, scenario in enumerate(llm_scenarios):
                # 3뷰 생성을 위한 컨텍스트 구성
                scenario_context = {
                    'text': scenario,
                    'scenario_id': f'original_{idx}',
                    'urgency': 0.5,  # 기본값
                    'complexity': 0.7,  # 기본값
                    'reversibility': 0.3  # 기본값
                }
                
                # 3뷰 시나리오 생성
                if self.three_view_system:
                    three_view_result = await self.three_view_system.analyze_three_view_scenarios(scenario_context)
                    
                    # 각 뷰에 대해 전체 분석 수행
                    for view_type in [ScenarioType.OPTIMISTIC, ScenarioType.NEUTRAL, ScenarioType.PESSIMISTIC]:
                        view_scenario = self._create_view_scenario(scenario, three_view_result, view_type)
                        
                        # 감정/윤리 평가
                        analysis_result = await self.analyze(view_scenario['text'])
                        
                        # 다원적 윤리 체계 평가 추가
                        ethics_analysis = {}
                        if self.multi_ethics_system:
                            # 윤리적 딜레마 구성
                            dilemma = EthicalDilemma(
                                dilemma_id=f"{idx}_{view_type.value}",
                                scenario=view_scenario['text'],
                                context="윤리적 의사결정 상황",
                                complexity_level=0.7,
                                urgency_level=0.5,
                                reversibility=0.3,
                                available_options=[scenario]
                            )
                            
                            # 각 윤리 엔진으로 평가
                            for ethics_name, engine in self.ethics_engines.items():
                                try:
                                    ethics_reasoning = engine.reason(dilemma)
                                    ethics_analysis[ethics_name] = {
                                        'recommendation': ethics_reasoning.final_recommendation,
                                        'confidence': ethics_reasoning.confidence_level,
                                        'reasoning': ethics_reasoning.reasoning_process[:100]  # 첫 100자만
                                    }
                                except Exception as e:
                                    self.logger.warning(f"   ⚠️ {ethics_name} 평가 실패: {e}")
                        
                        all_results.append({
                            'original_scenario': scenario,
                            'view_type': view_type.value,
                            'view_scenario': view_scenario,
                            'analysis': analysis_result,
                            'ethics_analysis': ethics_analysis,  # 윤리 분석 추가
                            'utility_score': three_view_result.consensus_utility if view_type == ScenarioType.NEUTRAL 
                                           else getattr(getattr(three_view_result, f"{view_type.value}_scenario"), 'utility_score'),
                            'regret_potential': getattr(getattr(three_view_result, f"{view_type.value}_scenario"), 'regret_potential'),
                            'timestamp': time.time()
                        })
                else:
                    # 3뷰 시스템 없으면 원본만 분석
                    analysis_result = await self.analyze(scenario)
                    all_results.append({
                        'original_scenario': scenario,
                        'view_type': 'original',
                        'view_scenario': {'text': scenario},
                        'analysis': analysis_result,
                        'utility_score': analysis_result.get('integrated_score', 0),
                        'regret_potential': 0.5,
                        'timestamp': time.time()
                    })
            
            self.logger.info(f"   Phase 1 완료: {len(all_results)}개 시나리오 생성")
            
            # Phase 2: 후회 분석으로 추가 시나리오 제안
            self.logger.info("   Phase 2: 후회 기반 추가 시나리오 생성")
            additional_scenarios = []
            
            if self.advanced_regret_learning:
                # 높은 후회 가능성이 있는 시나리오들을 찾아 대안 생성
                high_regret_scenarios = sorted(
                    all_results, 
                    key=lambda x: x['regret_potential'], 
                    reverse=True
                )[:3]  # 상위 3개
                
                for scenario_data in high_regret_scenarios:
                    # 후회 시스템으로 대안 생성
                    alternatives = await self.advanced_regret_learning.suggest_alternatives(
                        scenario_data['analysis']
                    )
                    
                    if alternatives and isinstance(alternatives, list):
                        additional_scenarios.extend(alternatives[:2])  # 각 시나리오당 최대 2개 대안
            
            self.logger.info(f"   Phase 2 완료: {len(additional_scenarios)}개 추가 시나리오 생성")
            
            # Phase 3: 정합성 판단 (둘 다 병행)
            self.logger.info("   Phase 3: 정합성 판단")
            plausible_scenarios = []
            
            for scenario in additional_scenarios:
                # 시스템 내부 정합성 점수 계산
                system_score = self._calculate_plausibility(scenario, context=all_results)
                
                # 점수가 낮으면 LLM 추가 검증 (향후 구현)
                if system_score < 0.7:
                    # TODO: LLM 정합성 검증 (LLM 통합 후)
                    if self.config.llm_mode != "none" and self.llm_engine:
                        llm_plausible = await self.llm_engine.check_plausibility(scenario)
                        if llm_plausible:
                            plausible_scenarios.append(scenario)
                    else:
                        # LLM 없으면 시스템 점수가 0.5 이상이면 통과
                        if system_score >= 0.5:
                            plausible_scenarios.append(scenario)
                else:
                    plausible_scenarios.append(scenario)
            
            self.logger.info(f"   Phase 3 완료: {len(plausible_scenarios)}개 정합성 통과")
            
            # Phase 4: 정합성 있는 추가 시나리오 평가
            if plausible_scenarios:
                self.logger.info("   Phase 4: 추가 시나리오 평가")
                for scenario in plausible_scenarios:
                    analysis_result = await self.analyze(scenario)
                    all_results.append({
                        'original_scenario': 'regret_generated',
                        'view_type': 'additional',
                        'view_scenario': {'text': scenario},
                        'analysis': analysis_result,
                        'utility_score': analysis_result.get('integrated_score', 0),
                        'regret_potential': 0.3,  # 대안은 후회 가능성 낮음
                        'timestamp': time.time()
                    })
            
            # Phase 5: 상위 2개 시나리오 선정
            self.logger.info("   Phase 5: 최종 시나리오 선정")
            
            # 통합 점수 기준 정렬
            sorted_results = sorted(
                all_results,
                key=lambda x: (
                    x['analysis'].get('integrated_score', 0) * 0.4 +
                    x['utility_score'] * 0.3 +
                    (1 - x['regret_potential']) * 0.3
                ),
                reverse=True
            )
            
            top_two = sorted_results[:2]
            
            # 최종 결과 구성
            result = {
                'selected_scenarios': top_two,
                'all_evaluations': all_results,
                'total_evaluated': len(all_results),
                'processing_time': time.time() - start_time,
                'recommendation': self._generate_recommendation(top_two),
                'metadata': {
                    'original_scenarios': len(llm_scenarios),
                    'three_view_expanded': len(llm_scenarios) * 3 if self.three_view_system else len(llm_scenarios),
                    'additional_generated': len(additional_scenarios),
                    'plausible_filtered': len(plausible_scenarios)
                }
            }
            
            self.logger.info(f"   ✅ 윤리적 딜레마 분석 완료 ({result['processing_time']:.2f}초)")
            self.logger.info(f"      평가된 시나리오: {result['total_evaluated']}개")
            self.logger.info(f"      최종 선택: 2개")
            
            return result
            
        except Exception as e:
            self.logger.error(f"   ❌ 윤리적 딜레마 분석 실패: {e}")
            return {
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def _create_view_scenario(self, original: str, three_view_result: ThreeViewAnalysisResult, 
                            view_type: ScenarioType) -> Dict[str, Any]:
        """3뷰 결과를 기반으로 시나리오 텍스트 생성"""
        
        # 뷰 타입에 따른 시나리오 메트릭 가져오기
        if view_type == ScenarioType.OPTIMISTIC:
            metrics = three_view_result.optimistic_scenario
            modifier = "최선의 경우: "
        elif view_type == ScenarioType.PESSIMISTIC:
            metrics = three_view_result.pessimistic_scenario
            modifier = "최악의 경우: "
        else:  # NEUTRAL
            metrics = three_view_result.neutral_scenario
            modifier = "일반적인 경우: "
        
        # 시나리오 텍스트 수정
        scenario_text = f"{modifier}{original}"
        
        # 리스크/기회 요소 추가
        if metrics.risk_factors:
            scenario_text += f" [위험: {', '.join(metrics.risk_factors[:2])}]"
        if metrics.opportunity_factors:
            scenario_text += f" [기회: {', '.join(metrics.opportunity_factors[:2])}]"
        
        return {
            'text': scenario_text,
            'metrics': metrics,
            'confidence': metrics.confidence_level,
            'ethical_implications': metrics.ethical_implications
        }
    
    def _calculate_plausibility(self, scenario: str, context: List[Dict]) -> float:
        """시나리오 정합성 점수 계산"""
        
        # 기본 점수
        score = 0.5
        
        # 컨텍스트와의 일관성 검사
        if context:
            # 기존 시나리오들과의 유사도 계산
            similarities = []
            for existing in context:
                # 간단한 텍스트 유사도 (실제로는 임베딩 사용)
                if isinstance(scenario, str) and isinstance(existing.get('original_scenario'), str):
                    common_words = set(scenario.lower().split()) & set(existing['original_scenario'].lower().split())
                    similarity = len(common_words) / max(len(scenario.split()), len(existing['original_scenario'].split()))
                    similarities.append(similarity)
            
            if similarities:
                avg_similarity = sum(similarities) / len(similarities)
                # 너무 유사하면 점수 감소 (중복), 너무 다르면 점수 감소 (비일관성)
                if 0.2 < avg_similarity < 0.8:
                    score += 0.2
                elif avg_similarity > 0.9:
                    score -= 0.2  # 거의 동일한 시나리오
                else:
                    score -= 0.1  # 너무 다른 시나리오
        
        # 길이 체크
        if isinstance(scenario, str):
            word_count = len(scenario.split())
            if 10 < word_count < 200:
                score += 0.1
            else:
                score -= 0.1
        
        # 최종 점수 정규화
        return max(0.0, min(1.0, score))
    
    def _generate_recommendation(self, top_scenarios: List[Dict]) -> str:
        """상위 시나리오 기반 추천 생성"""
        
        if not top_scenarios:
            return "시나리오 평가 실패 - 추가 분석 필요"
        
        # 첫 번째 시나리오의 점수들
        first = top_scenarios[0]
        utility = first.get('utility_score', 0)
        regret = first.get('regret_potential', 0)
        integrated = first['analysis'].get('integrated_score', 0)
        
        # 윤리 분석 결과 종합
        ethics_consensus = self._calculate_ethics_consensus(first.get('ethics_analysis', {}))
        
        # 추천 결정 로직
        if utility > 0.7 and regret < 0.3 and integrated > 0.7:
            recommendation = "적극 추진 권장 - 높은 효용과 낮은 후회 가능성"
        elif utility > 0.5 and regret < 0.5:
            recommendation = "신중한 추진 권장 - 적절한 효용, 위험 관리 필요"
        elif utility > 0.3 or regret > 0.7:
            recommendation = "재검토 권장 - 높은 후회 가능성 또는 낮은 효용"
        else:
            recommendation = "추진 비권장 - 위험이 효용을 초과"
        
        # 두 시나리오 간 차이가 작으면 추가 정보
        if len(top_scenarios) > 1:
            second = top_scenarios[1]
            diff = abs(integrated - second['analysis'].get('integrated_score', 0))
            if diff < 0.1:
                recommendation += " (상위 2개 시나리오가 유사한 평가)"
        
        return recommendation
    
    def _calculate_ethics_consensus(self, ethics_analysis: Dict[str, Dict]) -> Dict[str, Any]:
        """윤리 분석 결과 종합"""
        if not ethics_analysis:
            return {'consensus': 0.5, 'agreement': False}
        
        recommendations = []
        confidences = []
        
        for ethics_name, analysis in ethics_analysis.items():
            if 'recommendation' in analysis and 'confidence' in analysis:
                recommendations.append(analysis['recommendation'])
                confidences.append(analysis['confidence'])
        
        if not confidences:
            return {'consensus': 0.5, 'agreement': False}
        
        # 평균 신뢰도
        avg_confidence = sum(confidences) / len(confidences)
        
        # 추천 일치도 (간단한 텍스트 비교)
        agreement = False
        if recommendations:
            # 모든 추천이 '권장' 또는 '추진'을 포함하는지 확인
            positive_count = sum(1 for r in recommendations if '권장' in r or '추진' in r)
            negative_count = sum(1 for r in recommendations if '비권장' in r or '중단' in r)
            
            if positive_count > len(recommendations) * 0.7:
                agreement = True
            elif negative_count > len(recommendations) * 0.7:
                agreement = True  # 부정적 합의도 합의
        
        return {
            'consensus': avg_confidence,
            'agreement': agreement,
            'positive_ratio': positive_count / len(recommendations) if recommendations else 0
        }
    
    def _tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """텍스트를 임베딩으로 변환"""
        # sentence_transformer를 사용한 임베딩 생성
        from sentence_transformer_singleton import get_sentence_transformer
        
        # 모델 가져오기
        model = get_sentence_transformer(
            model_name=self.config.embedding_model,
            device=str(self.config.device)
        )
        
        # 텍스트를 임베딩으로 변환
        self.logger.debug(f"입력 텍스트: {text[:50]}...")
        self.logger.debug(f"텍스트 단어 수: {len(text.split())}")
        
        embeddings = model.encode([text])  # List[str] 입력
        self.logger.debug(f"encode 반환 타입: {type(embeddings)}")
        self.logger.debug(f"embeddings 길이: {len(embeddings) if hasattr(embeddings, '__len__') else 'N/A'}")
        
        if isinstance(embeddings, list) and embeddings:
            self.logger.debug(f"embeddings[0] 타입: {type(embeddings[0])}")
            if isinstance(embeddings[0], list):
                self.logger.debug(f"embeddings[0] 길이: {len(embeddings[0])}")
                self.logger.debug(f"처음 5개 값: {embeddings[0][:5]}")
        
        # 텐서로 변환 (UnifiedModel은 768차원 기대)
        embedding_tensor = torch.tensor(embeddings[0], dtype=torch.float32)
        embedding_tensor = embedding_tensor.to(self.config.device)
        self.logger.debug(f"변환 후 텐서 shape: {embedding_tensor.shape}")
        
        # 차원 조정 (UnifiedModel은 [batch, seq_len, hidden_dim] 형태 기대)
        if len(embedding_tensor.shape) == 1:
            # [hidden_dim] -> [1, 1, hidden_dim]
            embedding_tensor = embedding_tensor.unsqueeze(0).unsqueeze(0)
            self.logger.debug(f"unsqueeze 후 shape: {embedding_tensor.shape}")
            
            # 패딩을 위해 max_seq_length로 확장
            # 단순히 첫 번째 위치만 실제 임베딩, 나머지는 0 패딩
            padded_tensor = torch.zeros(
                1, self.config.max_seq_length, embedding_tensor.shape[-1],
                device=self.config.device, dtype=torch.float32
            )
            padded_tensor[:, 0, :] = embedding_tensor[0, 0, :]
            embedding_tensor = padded_tensor
            self.logger.debug(f"최종 패딩 후 shape: {embedding_tensor.shape}")
        
        return {'embeddings': embedding_tensor}
    
    def _process_unified_outputs(self, outputs: Dict, task: str = 'emotion') -> Dict:
        """UnifiedModel 출력 처리"""
        processed = {}
        
        # UnifiedModel은 return_all=True일 때 'head' 키로 반환
        if 'head' in outputs and outputs['head'] is not None:
            head_output = outputs['head']
            
            if task == 'emotion':
                # emotion 태스크 처리
                if isinstance(head_output, torch.Tensor):
                    emotion_names = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'love']
                    scores = head_output.softmax(dim=-1)[0].tolist() if head_output.dim() > 1 else head_output.softmax(dim=-1).tolist()
                    
                    # 감정 이름과 점수 매핑
                    emotion_dict = {}
                    for i, name in enumerate(emotion_names[:len(scores)]):
                        emotion_dict[name] = scores[i]
                    
                    processed['emotion'] = emotion_dict
                    # 메타 통합을 위해 scores도 추가
                    processed['emotion']['scores'] = scores
                    
            elif task == 'bentham':
                # bentham 태스크 처리 - 학습된 bentham_head 출력
                if isinstance(head_output, torch.Tensor):
                    # bentham_head는 10개 요소 출력
                    bentham_elements = [
                        'intensity', 'duration', 'certainty', 'propinquity',
                        'fecundity', 'purity', 'extent',
                        'pleasure_total', 'pain_total', 'net_pleasure'
                    ]
                    
                    # 텐서를 리스트로 변환
                    if head_output.dim() > 1:
                        scores = head_output[0].tolist()
                    else:
                        scores = head_output.tolist()
                    
                    # 벤담 요소와 점수 매핑
                    bentham_dict = {}
                    for i, name in enumerate(bentham_elements[:len(scores)]):
                        bentham_dict[name] = scores[i]
                    
                    processed['bentham'] = bentham_dict
                    
                    # 최종 쾌락 점수 계산
                    if len(scores) >= 10:
                        processed['bentham']['final_score'] = scores[9]  # net_pleasure
                    else:
                        # 7가지 기본 요소의 평균
                        processed['bentham']['final_score'] = sum(scores[:min(7, len(scores))]) / min(7, len(scores))
        
        # 다른 출력들 처리 (있다면)
        if 'dsp' in outputs and outputs['dsp'] is not None:
            # 정확한 구조 파악을 위한 상세 디버깅
            def analyze_structure(obj, name="object", depth=0, max_depth=5):
                """객체의 정확한 구조를 재귀적으로 분석"""
                indent = "  " * depth
                if depth > max_depth:
                    return f"{indent}{name}: <max depth reached>"
                
                result = []
                obj_type = type(obj).__name__
                
                if isinstance(obj, dict):
                    result.append(f"{indent}{name}: dict[{len(obj)} keys]")
                    for k, v in list(obj.items())[:5]:  # 최대 5개 키만
                        result.append(analyze_structure(v, f"['{k}']", depth + 1, max_depth))
                elif isinstance(obj, (list, tuple)):
                    type_name = 'list' if isinstance(obj, list) else 'tuple'
                    result.append(f"{indent}{name}: {type_name}[{len(obj)} items]")
                    for i, item in enumerate(obj[:3]):  # 최대 3개 아이템만
                        result.append(analyze_structure(item, f"[{i}]", depth + 1, max_depth))
                elif isinstance(obj, torch.Tensor):
                    result.append(f"{indent}{name}: Tensor(shape={list(obj.shape)}, dtype={obj.dtype}, device={obj.device})")
                elif hasattr(obj, '__dict__'):
                    attrs = list(obj.__dict__.keys())[:5]
                    result.append(f"{indent}{name}: {obj_type}(attrs={attrs})")
                else:
                    result.append(f"{indent}{name}: {obj_type}({str(obj)[:50]}...)" if len(str(obj)) > 50 else f"{indent}{name}: {obj_type}({obj})")
                
                return "\n".join(result)
            
            # DSP 출력 구조 분석
            self.logger.debug(f"DSP 출력 구조 분석:\n{analyze_structure(outputs['dsp'], 'outputs[dsp]')}")
            
            if isinstance(outputs['dsp'], dict) and 'final_emotions' in outputs['dsp']:
                final_emotions = outputs['dsp']['final_emotions']
                
                # final_emotions 타입 상세 분석
                self.logger.debug(f"final_emotions 타입 체인: {type(final_emotions)} → {type(final_emotions).__bases__ if hasattr(type(final_emotions), '__bases__') else 'no bases'}")
                
                if isinstance(final_emotions, torch.Tensor):
                    processed['dsp_emotions'] = final_emotions.tolist()
                elif isinstance(final_emotions, (list, tuple)):
                    processed['dsp_emotions'] = list(final_emotions)
                elif isinstance(final_emotions, dict):
                    self.logger.warning(f"final_emotions가 dict입니다. 구조: {analyze_structure(final_emotions, 'final_emotions')}")
                    # dict인 경우 values를 추출 시도
                    if final_emotions:
                        first_value = next(iter(final_emotions.values()))
                        if isinstance(first_value, torch.Tensor):
                            processed['dsp_emotions'] = first_value.tolist()
                else:
                    self.logger.error(f"예상치 못한 final_emotions 타입: {type(final_emotions)}")
        
        if 'neural' in outputs and outputs['neural'] is not None:
            processed['neural_analysis'] = outputs['neural']
        
        if 'wrapper' in outputs and outputs['wrapper'] is not None:
            processed['wrapper_analysis'] = outputs['wrapper']
        
        return processed
    
    def _is_korean(self, text: str) -> bool:
        """한국어 텍스트 감지"""
        import re
        korean_pattern = re.compile('[ㄱ-ㅎㅏ-ㅣ가-힣]+')
        return bool(korean_pattern.search(text))
    
    def _calculate_integrated_score(self, results: Dict) -> float:
        """통합 점수 계산"""
        score = 0.0
        weights = {
            'unified': 0.3,  # Circuit 사용시 비중 감소
            'neural_analysis': 0.25,
            'wrapper_analysis': 0.15,
            'dsp_analysis': 0.1,
            'phase_analysis': 0.05,
            'circuit_ethics': 0.15  # Circuit 윤리 점수 추가
        }
        
        for key, weight in weights.items():
            if key in results and results[key]:
                # 각 모듈의 점수 추출 (간단한 예시)
                if isinstance(results[key], dict):
                    module_score = sum(
                        v if isinstance(v, (int, float)) else 0.5
                        for v in results[key].values()
                    ) / max(len(results[key]), 1)
                    score += weight * min(max(module_score, 0), 1)
        
        return score
    
    def _calculate_confidence(self, results: Dict) -> float:
        """신뢰도 계산"""
        # 활성화된 모듈 수에 따른 신뢰도
        active_modules = sum(1 for k in [
            'unified', 'neural_analysis', 'wrapper_analysis',
            'dsp_analysis', 'phase_analysis'
        ] if k in results and results[k])
        
        return min(active_modules / 5.0, 1.0)
    
    def get_stats(self) -> Dict:
        """통계 반환"""
        if self.stats['successful'] > 0:
            self.stats['avg_time'] = self.stats['avg_time'] / self.stats['successful']
        return self.stats
    
    async def cleanup(self):
        """시스템 정리 및 종료"""
        self.logger.info("🧹 시스템 정리 중...")
        
        # 유휴 학습 시스템 정지
        if self.idle_learner:
            try:
                await self.idle_learner.stop()
                self.logger.info("   ✅ 유휴 학습 시스템 정지")
            except Exception as e:
                self.logger.warning(f"   ⚠️ 유휴 학습 정지 중 오류: {e}")
        
        # 체크포인트 저장 (필요시)
        if self.checkpoint_manager:
            try:
                # 현재 모델 상태 저장
                self.logger.info("   💾 최종 체크포인트 저장...")
            except Exception as e:
                self.logger.warning(f"   ⚠️ 체크포인트 저장 실패: {e}")
        
        self.logger.info("   ✅ 시스템 정리 완료")


def to_jsonable(x):
    """객체를 JSON 직렬화 가능한 형태로 변환"""
    import torch
    import numpy as np
    from enum import Enum
    from dataclasses import is_dataclass, asdict
    from datetime import datetime, date, timedelta
    from pathlib import Path
    import uuid
    
    # None 처리
    if x is None:
        return None
    
    # 기본 타입들
    if isinstance(x, (str, int, float, bool)):
        return x
    
    # Tensor 처리
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist() if x.dim() > 0 else x.detach().cpu().item()
    
    # NumPy 배열/스칼라 처리
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    
    # Enum 처리
    if isinstance(x, Enum):
        return x.name
    
    # UUID 처리
    if isinstance(x, uuid.UUID):
        return str(x)
    
    # Path 처리
    if isinstance(x, Path):
        return str(x)
    
    # datetime 관련 처리
    if isinstance(x, (datetime, date)):
        return x.isoformat()
    if isinstance(x, timedelta):
        return x.total_seconds()
    
    # dataclass 처리 (중요!)
    if is_dataclass(x) and not isinstance(x, type):
        # dataclass를 dict로 변환한 후 재귀적으로 처리
        try:
            return {k: to_jsonable(v) for k, v in asdict(x).items()}
        except Exception as e:
            # asdict 실패 시 __dict__ 사용
            if hasattr(x, '__dict__'):
                return {k: to_jsonable(v) for k, v in x.__dict__.items() 
                       if not k.startswith('_')}
            else:
                return str(x)
    
    # dict 처리
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    
    # list, tuple, set 처리
    if isinstance(x, (list, tuple, set)):
        converted = [to_jsonable(v) for v in x]
        if isinstance(x, tuple):
            return converted  # JSON은 tuple을 지원하지 않으므로 list로
        elif isinstance(x, set):
            return converted  # set도 list로
        else:
            return converted
    
    # 기타 객체들 - __dict__ 속성이 있으면 dict로 변환
    if hasattr(x, '__dict__'):
        return {k: to_jsonable(v) for k, v in x.__dict__.items() 
               if not k.startswith('_')}
    
    # 최후의 수단: 문자열로 변환
    try:
        return str(x)
    except:
        return None

async def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='Red Heart AI 통합 추론 시스템')
    
    # 기본 인자
    parser.add_argument('--text', type=str, help='분석할 텍스트')
    parser.add_argument('--mode', default='inference',
                       choices=['inference', 'test', 'demo', 'production'],
                       help='실행 모드')
    
    # 체크포인트 설정
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='체크포인트 직접 경로 (없으면 --epoch 사용)')
    parser.add_argument('--epoch', type=int, default=50,
                       help='로드할 체크포인트 에폭 번호 (기본: 50 - sweet spot)')
    
    # 모듈 활성화 플래그
    parser.add_argument('--no-neural', action='store_true', help='Neural Analyzers 비활성화')
    parser.add_argument('--no-wrappers', action='store_true', help='Advanced Wrappers 비활성화')
    parser.add_argument('--no-dsp', action='store_true', help='DSP 시뮬레이터 비활성화')
    parser.add_argument('--no-phase', action='store_true', help='Phase Networks 비활성화')
    
    # LLM 옵션
    parser.add_argument('--llm', 
                       choices=['none', 'local', 'claude', 'mcp', 'gpt', 'perplexity', 'deepseek'],
                       default='none', help='LLM 통합 모드 (none/local/API 이름)')
    
    # 기타
    parser.add_argument('--batch-size', type=int, default=4, help='배치 크기')
    parser.add_argument('--device', type=str, help='디바이스 (cuda/cpu)')
    parser.add_argument('--verbose', action='store_true', help='상세 로그')
    parser.add_argument('--debug', action='store_true', help='디버그 모드')
    
    # 메모리 모드 직접 선택 옵션 추가
    parser.add_argument('--memory-mode', 
                       choices=['light', 'medium', 'heavy'],
                       help='메모리 모드 선택 (기본: 자동)')
    
    args = parser.parse_args()
    
    # 설정 생성
    config = InferenceConfig(
        checkpoint_path=args.checkpoint,  # 직접 경로 (있으면 우선)
        checkpoint_epoch=args.epoch,  # 에폭 번호로 자동 검색
        batch_size=args.batch_size,
        device=args.device or str(DEVICE),
        use_neural_analyzers=not args.no_neural,
        use_advanced_wrappers=not args.no_wrappers,
        use_dsp_simulator=not args.no_dsp,
        use_phase_networks=not args.no_phase,
        llm_mode=args.llm,
        verbose=args.verbose,
        debug=args.debug
    )
    
    # 메모리 모드 설정
    if args.memory_mode:
        config.memory_mode = MemoryMode[args.memory_mode.upper()]
        config.auto_memory_mode = False  # 수동 선택시 자동 모드 비활성화
    
    # 시스템 생성 및 초기화
    system = UnifiedInferenceSystem(config)
    await system.initialize()
    
    # 모드별 실행
    if args.mode == 'test':
        # 테스트 모드
        logger.info("\n🧪 테스트 모드 실행...")
        test_texts = [
            "이 결정은 많은 사람들의 생명과 안전에 영향을 미칩니다.",
            "기술 발전과 일자리 보호 사이의 균형을 찾아야 합니다.",
            "개인정보 보호와 공익 사이의 갈등 상황입니다."
        ]
        
        for i, text in enumerate(test_texts, 1):
            logger.info(f"\n테스트 {i}: {text}")
            result = await system.analyze(text)
            logger.info(f"결과: {json.dumps(to_jsonable(result), indent=2, ensure_ascii=False)[:500]}...")
    
    elif args.mode == 'demo':
        # 데모 모드
        logger.info("\n🎮 데모 모드 - 대화형 분석")
        while True:
            try:
                text = input("\n텍스트 입력 (종료: quit): ")
                if text.lower() == 'quit':
                    break
                
                result = await system.analyze(text)
                print(f"\n📊 분석 결과:")
                print(f"   통합 점수: {result.get('integrated_score', 0):.3f}")
                print(f"   신뢰도: {result.get('confidence', 0):.3f}")
                print(f"   처리 시간: {result.get('processing_time', 0):.2f}초")
                
                if 'unified' in result:
                    print(f"\n   감정 분석: {result['unified'].get('emotion', {})}")
                    print(f"   벤담 점수: {result['unified'].get('bentham', {})}")
                    print(f"   후회 점수: {result['unified'].get('regret', {})}")
                    print(f"   SURD 분석: {result['unified'].get('surd', {})}")
                
            except KeyboardInterrupt:
                break
    
    elif args.mode == 'production':
        # 운용 모드
        logger.info("\n🚀 운용 모드 활성화")
        if args.text:
            result = await system.analyze(args.text)
            print(json.dumps(to_jsonable(result), indent=2, ensure_ascii=False))
        else:
            logger.info("텍스트를 --text 인자로 제공하세요")
    
    else:
        # 추론 모드 (기본)
        if args.text:
            logger.info(f"\n분석 텍스트: {args.text}")
            result = await system.analyze(args.text)
            print(f"\n📊 분석 결과:")
            print(json.dumps(to_jsonable(result), indent=2, ensure_ascii=False))
        else:
            logger.info("텍스트를 --text 인자로 제공하거나 --mode를 선택하세요")
    
    # 시스템 정리
    await system.cleanup()
    
    # 통계 출력
    stats = system.get_stats()
    logger.info(f"\n📈 세션 통계:")
    logger.info(f"   총 요청: {stats['total_requests']}")
    logger.info(f"   성공: {stats['successful']}")
    logger.info(f"   실패: {stats['failed']}")
    logger.info(f"   평균 시간: {stats['avg_time']:.2f}초")


if __name__ == "__main__":
    # 이벤트 루프 실행
    asyncio.run(main())