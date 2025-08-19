#!/usr/bin/env python3
"""
Red Heart AI 최종 통합 학습 시스템
730M 파라미터 모델의 60 에폭 학습 with Advanced Techniques
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import logging
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import gc
import time

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 커스텀 모듈 임포트
from training.enhanced_checkpoint_manager import EnhancedCheckpointManager
from training.lr_sweep_optimizer import LRSweepOptimizer
from training.sweet_spot_detector import SweetSpotDetector
from training.parameter_crossover_system import ParameterCrossoverSystem
from training.oom_handler import OOMHandler
from training.advanced_training_techniques import AdvancedTrainingManager

# 프로젝트 모듈
from config import ADVANCED_CONFIG, get_device
from data_loader import PreprocessedDataLoader

# 실제 모델 모듈들
from unified_backbone import RedHeartUnifiedBackbone
from unified_heads import EmotionHead, BenthamHead, RegretHead, SURDHead
from analyzer_neural_modules import create_neural_analyzers
from advanced_analyzer_wrappers import create_advanced_analyzer_wrappers
from phase_neural_networks import Phase0ProjectionNet, Phase2CommunityNet, HierarchicalEmotionIntegrator
try:
    from emotion_dsp_simulator import EmotionDSPSimulator, DynamicKalmanFilter
except ImportError:
    EmotionDSPSimulator = None
    DynamicKalmanFilter = None

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('RedHeart.UnifiedTrainingFinal')


class UnifiedTrainingConfig:
    """통합 학습 설정"""
    
    def __init__(self):
        # 모델 설정 (730M 파라미터)
        self.model_params = 730_000_000
        self.hidden_dim = 1280
        self.num_layers = 18
        self.num_heads = 20
        
        # 학습 설정
        self.total_epochs = 60
        self.micro_batch_size = 2  # 안정성을 위해 2로 시작
        self.gradient_accumulation = 32  # 유효 배치 = 64
        self.base_lr = 1e-4
        
        # LR 스윕 설정
        self.lr_sweep_enabled = True
        self.lr_sweep_range = (1e-5, 1e-2)
        self.lr_sweep_points = 5
        
        # 체크포인트 설정
        self.checkpoint_interval = 2  # 짝수 에폭마다 저장 (30개)
        self.checkpoint_dir = "training/checkpoints_final"
        
        # Advanced Training
        self.enable_label_smoothing = True
        self.enable_rdrop = True
        self.enable_ema = True
        self.enable_llrd = True
        self.label_smoothing = 0.1
        self.rdrop_alpha = 1.0
        self.ema_decay = 0.999
        
        # OOM 핸들링
        self.enable_oom_handler = True
        self.memory_threshold = 0.85
        self.min_batch_size = 1
        
        # Sweet Spot & Crossover
        self.enable_sweet_spot = True
        self.enable_crossover = True
        self.crossover_strategy = 'selective'
        
        # 데이터 설정
        self.data_dir = "for_learn_dataset"
        self.validation_split = 0.1
        self.num_workers = 4
        
        # 로깅
        self.log_interval = 10
        self.verbose = False  # 상세 출력 설정
        self.val_interval = 100


class UnifiedModel(nn.Module):
    """Red Heart AI 730M 통합 모델"""
    
    def __init__(self, config: UnifiedTrainingConfig):
        super().__init__()
        self.config = config
        
        # 백본 설정
        backbone_config = {
            'input_dim': 768,
            'd_model': 896,
            'num_layers': 8,
            'num_heads': 14,
            'feedforward_dim': 3584,
            'dropout': 0.1,
            'task_dim': 896
        }
        
        # 백본 초기화 (90.6M)
        self.backbone = RedHeartUnifiedBackbone(backbone_config)
        
        # 헤드 초기화 (153M)
        self.emotion_head = EmotionHead(input_dim=896)
        self.bentham_head = BenthamHead(input_dim=896)
        self.regret_head = RegretHead(input_dim=896) 
        self.surd_head = SURDHead(input_dim=896)
        
        # 신경망 분석기 (368M)
        self.neural_analyzers = create_neural_analyzers(input_dim=896)
        
        # Advanced 분석기 래퍼 (112M) - translator 초기화 후 생성
        self.advanced_wrappers = None  # 나중에 초기화
        
        # Phase 네트워크 (4.3M)
        self.phase0_net = Phase0ProjectionNet()
        self.phase2_net = Phase2CommunityNet()
        self.hierarchical_integrator = HierarchicalEmotionIntegrator()
        
        # DSP & 칼만 필터 (2.3M)
        if EmotionDSPSimulator is not None:
            self.dsp_simulator = EmotionDSPSimulator({'hidden_dim': 384})
            self.kalman_filter = DynamicKalmanFilter(state_dim=7)
        else:
            self.dsp_simulator = None
            self.kalman_filter = None
        
    def forward(self, x, task='emotion'):
        """순전파 - GPU 메모리 효율적 처리
        
        Args:
            x: 입력 텐서
            task: 현재 학습 중인 태스크
            
        Returns:
            해당 태스크의 출력 텐서 (dict가 아닌 tensor)
        """
        # 입력을 디바이스로 이동 (필요시)
        if x.device != self.backbone.parameters().__next__().device:
            x = x.to(self.backbone.parameters().__next__().device)
        
        # 백본 처리 (이미 GPU에 있음)
        backbone_outputs = self.backbone(x, task=task)
        
        # 태스크별 특징 추출
        if task in backbone_outputs:
            features = backbone_outputs[task]
        else:
            # 모든 태스크 출력의 평균 사용
            features = torch.stack(list(backbone_outputs.values())).mean(dim=0)
        
        # 태스크별 헤드 적용 (모두 GPU에 있음)
        if task == 'emotion':
            # EmotionHead가 dict를 반환하면 'emotions' 키 추출
            output = self.emotion_head(features)
            if isinstance(output, dict):
                output = output.get('emotions', output.get('emotion_logits', list(output.values())[0]))
        elif task == 'bentham':
            output = self.bentham_head(features)
            if isinstance(output, dict):
                output = output.get('bentham_scores', list(output.values())[0])
        elif task == 'regret':
            output = self.regret_head(features)
            if isinstance(output, dict):
                output = output.get('regret_score', list(output.values())[0])
        elif task == 'surd':
            output = self.surd_head(features)
            if isinstance(output, dict):
                output = output.get('surd_values', output.get('surd_scores', list(output.values())[0]))
        else:
            # 기본값: emotion
            output = self.emotion_head(features)
            if isinstance(output, dict):
                output = output.get('emotions', output.get('emotion_logits', list(output.values())[0]))
        
        return output


class UnifiedTrainer:
    """통합 학습 관리자"""
    
    def __init__(self, config: UnifiedTrainingConfig):
        self.config = config
        self.device = get_device()
        self.verbose = config.verbose  # V2와 동일하게 verbose 설정
        
        logger.info("=" * 70)
        logger.info("Red Heart AI 최종 통합 학습 시스템 초기화")
        logger.info(f"  - 모델 크기: {config.model_params/1e6:.0f}M 파라미터")
        logger.info(f"  - 총 에폭: {config.total_epochs}")
        logger.info(f"  - 배치 사이즈: {config.micro_batch_size} (GA={config.gradient_accumulation})")
        logger.info(f"  - 디바이스: {self.device}")
        logger.info("=" * 70)
        
        # 컴포넌트 초기화
        self._initialize_components()
        
        # 모델 초기화
        self._initialize_model()
        
        # 데이터 로더 초기화
        self._initialize_dataloaders()
        
        # 옵티마이저 초기화
        self._initialize_optimizer()
        
        # 메트릭 추적
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.no_param_update = False  # 파라미터 업데이트 플래그
        
    def _initialize_components(self):
        """컴포넌트 초기화"""
        # 체크포인트 매니저
        self.checkpoint_manager = EnhancedCheckpointManager(
            checkpoint_dir=self.config.checkpoint_dir,
            max_checkpoints=30,
            save_interval=self.config.checkpoint_interval
        )
        
        # LR 스윕 옵티마이저
        if self.config.lr_sweep_enabled:
            self.lr_sweep = LRSweepOptimizer(
                base_lr=self.config.base_lr,
                sweep_range=self.config.lr_sweep_range,
                num_sweep_points=self.config.lr_sweep_points
            )
        
        # Sweet Spot 탐지기
        if self.config.enable_sweet_spot:
            self.sweet_spot_detector = SweetSpotDetector(
                window_size=5,
                stability_threshold=0.01,
                patience=10
            )
        
        # Parameter Crossover
        if self.config.enable_crossover:
            self.crossover_system = ParameterCrossoverSystem(
                crossover_strategy=self.config.crossover_strategy,
                blend_ratio=0.7
            )
        
        # OOM 핸들러
        if self.config.enable_oom_handler:
            self.oom_handler = OOMHandler(
                initial_batch_size=self.config.micro_batch_size,
                min_batch_size=self.config.min_batch_size,
                gradient_accumulation=self.config.gradient_accumulation,
                memory_threshold=self.config.memory_threshold
            )
        
        # Advanced Training Manager
        self.training_manager = AdvancedTrainingManager(
            enable_label_smoothing=self.config.enable_label_smoothing,
            enable_rdrop=self.config.enable_rdrop,
            enable_ema=self.config.enable_ema,
            enable_llrd=self.config.enable_llrd,
            label_smoothing=self.config.label_smoothing,
            rdrop_alpha=self.config.rdrop_alpha,
            ema_decay=self.config.ema_decay
        )
        
        logger.info("✅ 모든 컴포넌트 초기화 완료")
    
    def _initialize_model(self):
        """모델 초기화 - v2 방식 차용 (순차적 GPU 로드)"""
        logger.info("🔧 모델 초기화 시작 (순차적 GPU 로드 방식)...")
        
        # 실제 730M 모델 초기화 (CPU에서 생성)
        self.model = UnifiedModel(self.config)
        
        # GPU 메모리 상태 확인
        if self.device.type == 'cuda':
            gpu_mem_before = torch.cuda.memory_allocated() / (1024**3)
            logger.info(f"  초기 GPU 메모리: {gpu_mem_before:.2f}GB")
        
        # 순차적으로 컴포넌트를 GPU로 이동 (7GB까지 활용)
        # 1. 백본 (90.6M) - 항상 GPU
        self.model.backbone = self.model.backbone.to(self.device)
        logger.info(f"  ✅ 백본 GPU 로드 (90.6M)")
        
        # 2. 모든 헤드 (153M) - GPU에 유지 (이전엔 필요시만 로드)
        self.model.emotion_head = self.model.emotion_head.to(self.device)
        logger.info(f"  ✅ 감정 헤드 GPU 로드 (38.3M)")
        
        self.model.bentham_head = self.model.bentham_head.to(self.device)
        logger.info(f"  ✅ 벤담 헤드 GPU 로드 (38.3M)")
        
        self.model.regret_head = self.model.regret_head.to(self.device)
        logger.info(f"  ✅ 후회 헤드 GPU 로드 (38.3M)")
        
        self.model.surd_head = self.model.surd_head.to(self.device)
        logger.info(f"  ✅ SURD 헤드 GPU 로드 (38.3M)")
        
        # 3. Translator 모듈 초기화 (Advanced 분석기 의존성)
        logger.info("  🔄 Translator 모듈 초기화 중...")
        try:
            from config import get_system_module, register_system_module
            existing_translator = get_system_module('translator')
            if existing_translator is None:
                from local_translator import LocalTranslator
                translator = LocalTranslator()
                register_system_module('translator', translator)
                logger.info("  ✅ LocalTranslator 초기화 및 전역 등록 완료")
            else:
                logger.info("  ℹ️ Translator가 이미 등록되어 있습니다")
        except Exception as e:
            logger.error(f"  ❌ Translator 초기화 실패: {e}")
            logger.warning("     Advanced Emotion Wrapper가 제한됩니다")
        
        # Translator 초기화 후 Advanced Wrappers 생성
        if self.model.advanced_wrappers is None:
            logger.info("  🔧 Advanced Wrappers 생성 중...")
            from advanced_analyzer_wrappers import create_advanced_analyzer_wrappers
            self.model.advanced_wrappers = create_advanced_analyzer_wrappers()
            
            # Advanced Wrappers 파라미터 수 확인
            wrapper_params = 0
            if self.model.advanced_wrappers:
                for name, wrapper in self.model.advanced_wrappers.items():
                    if hasattr(wrapper, 'parameters'):
                        params = sum(p.numel() for p in wrapper.parameters())
                        wrapper_params += params
            logger.info(f"  ✅ Advanced Wrappers 생성 완료 ({wrapper_params/1e6:.1f}M)")
        
        # 4. Neural Analyzers (368M) - GPU 여유 있으면 로드  
        if hasattr(self.model, 'neural_analyzers') and self.model.neural_analyzers:
            try:
                for name, analyzer in self.model.neural_analyzers.items():
                    analyzer.to(self.device)
                    logger.info(f"  ✅ {name} 분석기 GPU 로드")
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    logger.warning(f"  ⚠️ Neural Analyzers는 메모리 부족으로 CPU 유지")
                else:
                    raise
        
        # 5. Advanced Wrappers (112M) - GPU 여유 있으면 로드
        if hasattr(self.model, 'advanced_wrappers') and self.model.advanced_wrappers:
            try:
                for name, wrapper in self.model.advanced_wrappers.items():
                    wrapper.to(self.device)
                    logger.info(f"  ✅ {name} Wrapper GPU 로드")
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    logger.warning(f"  ⚠️ Advanced Wrappers는 메모리 부족으로 CPU 유지")
                else:
                    raise
        
        # 6. Phase Networks (4.3M) - 작으니까 GPU로
        if hasattr(self.model, 'phase0_net') and self.model.phase0_net:
            self.model.phase0_net = self.model.phase0_net.to(self.device)
            logger.info(f"  ✅ Phase0 네트워크 GPU 로드")
        
        if hasattr(self.model, 'phase2_net') and self.model.phase2_net:
            self.model.phase2_net = self.model.phase2_net.to(self.device)
            logger.info(f"  ✅ Phase2 네트워크 GPU 로드")
        
        if hasattr(self.model, 'hierarchical_integrator') and self.model.hierarchical_integrator:
            self.model.hierarchical_integrator = self.model.hierarchical_integrator.to(self.device)
            logger.info(f"  ✅ Hierarchical Integrator GPU 로드")
        
        # 7. DSP & Kalman (2.3M) - 작으니까 GPU로
        if hasattr(self.model, 'dsp_simulator') and self.model.dsp_simulator:
            self.model.dsp_simulator = self.model.dsp_simulator.to(self.device)
            logger.info(f"  ✅ DSP Simulator GPU 로드")
        
        if hasattr(self.model, 'kalman_filter') and self.model.kalman_filter:
            self.model.kalman_filter = self.model.kalman_filter.to(self.device)
            logger.info(f"  ✅ Kalman Filter GPU 로드")
        
        # GPU 메모리 사용량 확인
        if self.device.type == 'cuda':
            gpu_mem_after = torch.cuda.memory_allocated() / (1024**3)
            logger.info(f"  최종 GPU 메모리: {gpu_mem_after:.2f}GB (증가: {gpu_mem_after - gpu_mem_before:.2f}GB)")
            
            # 전체 GPU 메모리 정보
            total_gpu = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"  GPU 사용률: {gpu_mem_after/total_gpu*100:.1f}% / {total_gpu:.1f}GB")
        
        # Advanced Training 초기화
        self.training_manager.initialize(
            model=self.model,
            num_classes=6,
            base_lr=self.config.base_lr
        )
        
        # 파라미터 수 확인 (v2처럼 각 컴포넌트별로 계산)
        total_params = 0
        
        # 백본
        backbone_params = sum(p.numel() for p in self.model.backbone.parameters())
        total_params += backbone_params
        logger.info(f"  백본: {backbone_params/1e6:.1f}M")
        
        # 헤드들
        head_params = 0
        for name in ['emotion_head', 'bentham_head', 'regret_head', 'surd_head']:
            if hasattr(self.model, name):
                head = getattr(self.model, name)
                params = sum(p.numel() for p in head.parameters())
                head_params += params
                logger.info(f"  {name}: {params/1e6:.1f}M")
        total_params += head_params
        
        # Neural Analyzers
        if hasattr(self.model, 'neural_analyzers') and self.model.neural_analyzers:
            analyzer_params = 0
            for name, analyzer in self.model.neural_analyzers.items():
                params = sum(p.numel() for p in analyzer.parameters())
                analyzer_params += params
            total_params += analyzer_params
            logger.info(f"  Neural Analyzers: {analyzer_params/1e6:.1f}M")
        
        # Advanced Wrappers
        if hasattr(self.model, 'advanced_wrappers') and self.model.advanced_wrappers:
            wrapper_params = 0
            for name, wrapper in self.model.advanced_wrappers.items():
                if hasattr(wrapper, 'parameters'):
                    params = sum(p.numel() for p in wrapper.parameters())
                    wrapper_params += params
            total_params += wrapper_params
            logger.info(f"  Advanced Wrappers: {wrapper_params/1e6:.1f}M")
        
        # Phase Networks
        phase_params = 0
        for name in ['phase0_net', 'phase2_net', 'hierarchical_integrator']:
            if hasattr(self.model, name) and getattr(self.model, name) is not None:
                net = getattr(self.model, name)
                params = sum(p.numel() for p in net.parameters())
                phase_params += params
        if phase_params > 0:
            total_params += phase_params
            logger.info(f"  Phase Networks: {phase_params/1e6:.1f}M")
        
        # DSP & Kalman
        dsp_kalman_params = 0
        if hasattr(self.model, 'dsp_simulator') and self.model.dsp_simulator:
            dsp_kalman_params += sum(p.numel() for p in self.model.dsp_simulator.parameters())
        if hasattr(self.model, 'kalman_filter') and self.model.kalman_filter:
            dsp_kalman_params += sum(p.numel() for p in self.model.kalman_filter.parameters())
        if dsp_kalman_params > 0:
            total_params += dsp_kalman_params
            logger.info(f"  DSP & Kalman: {dsp_kalman_params/1e6:.1f}M")
        
        logger.info(f"✅ 모델 초기화 완료: 총 {total_params/1e6:.1f}M 파라미터")
    
    def _initialize_dataloaders(self):
        """데이터 로더 초기화"""
        # 실제 데이터 로드
        preprocessed_path = Path("claude_api_preprocessing/claude_preprocessed_complete.json")
        
        if not preprocessed_path.exists():
            # 대체 경로 시도
            preprocessed_path = Path("for_learn_dataset/claude_preprocessed_complete.json")
            if not preprocessed_path.exists():
                logger.error(f"전처리된 데이터를 찾을 수 없습니다: {preprocessed_path}")
                raise FileNotFoundError(f"전처리된 데이터 파일이 없습니다")
        
        # JSON 데이터 로드
        with open(preprocessed_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 샘플 수 제한 (테스트 모드에서)
        if hasattr(self.config, 'max_samples') and self.config.max_samples:
            data = data[:self.config.max_samples]
        
        # 데이터셋 클래스
        class RedHeartDataset(Dataset):
            def __init__(self, data_list):
                self.data = data_list
                # label 매핑 (v2에서 처럼 TargetMapper 대신 직접 처리)
                self.label_to_idx = {
                    'AUTHOR': 0,
                    'EVERYBODY': 1,
                    'INFO': 2,
                    'NOBODY': 3,
                    'OTHER': 4
                }
                # 감정 매핑 (emotions dict에서 추출)
                self.emotion_keys = ['joy', 'anger', 'surprise', 'disgust', 'sadness', 'shame', 'fear']
                
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                item = self.data[idx]
                # 텐서 변환
                text_embedding = torch.randn(100, 768)  # 실제로는 텍스트 임베딩 사용
                
                # label 문자열을 숫자로 변환
                label_str = item.get('label', 'OTHER')
                label_idx = self.label_to_idx.get(label_str, 4)  # 기본값 4 (OTHER)
                
                # emotions dict에서 감정 벡터 추출
                emotions = item.get('emotions', {})
                if isinstance(emotions, dict):
                    # 7개 기본 감정 추출
                    emotion_vector = [emotions.get(key, 0.0) for key in self.emotion_keys]
                    # 가장 높은 값의 인덱스를 라벨로
                    emotion_label = torch.argmax(torch.tensor(emotion_vector)).item()
                else:
                    emotion_label = 0  # 기본값
                
                # bentham_scores 처리 (dict -> 10차원 벡터)
                bentham_keys = [
                    'intensity', 'duration', 'certainty', 'propinquity',
                    'purity', 'extent', 'fecundity', 'remoteness', 
                    'succession', 'utility'
                ]
                
                bentham_scores = item.get('bentham_scores', {})
                if isinstance(bentham_scores, dict):
                    # dict에서 값 추출 (없으면 0.5 기본값)
                    bentham_vector = [bentham_scores.get(key, 0.5) for key in bentham_keys]
                else:
                    # dict가 아니면 기본값 사용
                    bentham_vector = [0.5] * 10
                
                return {
                    'input': text_embedding,
                    'emotion_label': torch.tensor(emotion_label, dtype=torch.long),
                    'bentham_label': torch.tensor(bentham_vector, dtype=torch.float),
                    'regret_label': torch.tensor(item.get('regret_factor', 0.0), dtype=torch.float),
                    'surd_label': torch.tensor(label_idx, dtype=torch.long)  # label을 SURD에도 사용
                }
        
        # 전체 데이터셋 생성
        dataset = RedHeartDataset(data)
        val_size = int(len(dataset) * self.config.validation_split)
        train_size = len(dataset) - val_size
        
        # 학습/검증 분할  
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.micro_batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.micro_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        logger.info(f"✅ 데이터 로더 초기화: Train={train_size}, Val={val_size}")
    
    def _initialize_optimizer(self):
        """옵티마이저 초기화"""
        # LLRD 사용 시
        if self.config.enable_llrd:
            self.optimizer = self.training_manager.get_optimizer(
                self.model,
                lr=self.config.base_lr,
                weight_decay=0.01
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.base_lr,
                weight_decay=0.01
            )
        
        # 스케줄러
        total_steps = len(self.train_loader) * self.config.total_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=total_steps // 10,
            T_mult=2,
            eta_min=self.config.base_lr * 0.01
        )
        
        logger.info("✅ 옵티마이저 및 스케줄러 초기화 완료")
    
    def run_lr_sweep(self):
        """LR 스윕 실행"""
        if not self.config.lr_sweep_enabled:
            return
        
        logger.info("\n" + "=" * 70)
        logger.info("🔍 Learning Rate Sweep 시작...")
        logger.info("=" * 70)
        
        # 간단한 손실 함수
        criterion = nn.CrossEntropyLoss()
        
        # 스윕 실행
        sweep_results = self.lr_sweep.run_sweep(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            criterion=criterion,
            device=self.device
        )
        
        # 최적 LR로 옵티마이저 재초기화
        self.config.base_lr = self.lr_sweep.best_lr
        self._initialize_optimizer()
        
        logger.info(f"✅ 최적 LR 선택: {self.config.base_lr:.1e}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """한 에폭 학습"""
        self.model.train()
        epoch_losses = []
        module_metrics = {}
        
        for batch_idx, batch in enumerate(self.train_loader):
            # OOM 핸들링
            if self.config.enable_oom_handler:
                self.oom_handler.log_memory_stats(self.global_step, 'train')
            
            # Forward pass
            try:
                loss, metrics = self._forward_step(batch)
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    if self.oom_handler.handle_oom(e):
                        # 배치 사이즈 조정 후 재시도
                        self.train_loader = self.oom_handler.adjust_dataloader(self.train_loader)
                        continue
                    else:
                        raise
                else:
                    raise
            
            # Backward pass (Gradient Accumulation)
            loss = loss / self.config.gradient_accumulation
            loss.backward()
            
            # Gradient Accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step (파라미터 업데이트 플래그 확인)
                if not self.no_param_update:
                    self.optimizer.step()
                    self.scheduler.step()
                    
                    # EMA update
                    self.training_manager.step()
                else:
                    # 검증 모드: 그라디언트만 계산하고 업데이트는 건너뜀
                    logger.debug("  [검증] 파라미터 업데이트 건너뜀")
                
                self.optimizer.zero_grad()
                self.global_step += 1
            
            epoch_losses.append(loss.item() * self.config.gradient_accumulation)
            
            # 로깅
            if batch_idx % self.config.log_interval == 0:
                avg_loss = np.mean(epoch_losses[-self.config.log_interval:])
                lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"  [Epoch {epoch}][{batch_idx}/{len(self.train_loader)}] "
                          f"Loss: {avg_loss:.4f}, LR: {lr:.1e}")
            
            # 메트릭 업데이트
            for key, value in metrics.items():
                if key not in module_metrics:
                    module_metrics[key] = []
                module_metrics[key].append(value)
        
        # 에폭 평균
        avg_metrics = {k: np.mean(v) for k, v in module_metrics.items()}
        avg_metrics['loss'] = np.mean(epoch_losses)
        
        return avg_metrics
    
    def validate(self) -> Dict[str, float]:
        """검증"""
        self.model.eval()
        val_losses = []
        module_metrics = {}
        
        with torch.no_grad():
            # EMA 적용
            if self.config.enable_ema:
                self.training_manager.apply_ema()
            
            for batch in self.val_loader:
                loss, metrics = self._forward_step(batch)
                val_losses.append(loss.item())
                
                for key, value in metrics.items():
                    if key not in module_metrics:
                        module_metrics[key] = []
                    module_metrics[key].append(value)
            
            # EMA 복원
            if self.config.enable_ema:
                self.training_manager.restore_ema()
        
        # 평균
        avg_metrics = {k: np.mean(v) for k, v in module_metrics.items()}
        avg_metrics['val_loss'] = np.mean(val_losses)
        
        return avg_metrics
    
    def _forward_step(self, batch: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        학습 스텝 - V2의 3단계 워크플로우 복원
        1. FORWARD: 데이터 → 백본 → 헤드
        2. COMPUTE: 손실 계산 + Neural Analyzers + DSP/Kalman
        3. UPDATE: 역전파 + 최적화
        """
        batch_idx = self.global_step % 100  # 로깅용
        
        # ========== STAGE 1: FORWARD ==========
        if self.verbose and batch_idx < 3:
            logger.info("    [STAGE 1] Forward Pass 시작")
        
        # 데이터 준비 - 입력 추출
        inputs = batch['input'].to(self.device)
        
        # 백본 통과
        backbone_outputs = self.model.backbone(inputs, return_all_tasks=True)
        features = backbone_outputs.get('emotion', inputs)  # 896차원
        
        if self.verbose and batch_idx < 3:
            logger.info(f"      - 백본 출력 shape: {features.shape}")
            logger.info(f"      - 백본 출력 키: {list(backbone_outputs.keys())}")
        
        # ========== STAGE 2: COMPUTE LOSSES ==========
        if self.verbose and batch_idx < 3:
            logger.info("    [STAGE 2] Compute Loss")
        
        # 헤드 손실 계산
        head_losses = []
        individual_losses = {}  # 개별 손실 저장용
        individual_accs = {}    # 개별 정확도 저장용
        
        # 감정 헤드
        if hasattr(self.model, 'emotion_head'):
            emotion_output = self.model.emotion_head(features)
            emotion_pred = emotion_output['emotions'] if isinstance(emotion_output, dict) else emotion_output
            emotion_target = batch['emotion_label'].to(self.device)
            emotion_loss = self.model.emotion_head.compute_loss(emotion_pred, emotion_target)
            head_losses.append(emotion_loss)
            individual_losses['emotion_loss'] = emotion_loss.item()
            # accuracy 계산 (classification task)
            emotion_acc = (emotion_pred.argmax(dim=-1) == emotion_target).float().mean().item()
            individual_accs['emotion_acc'] = emotion_acc
            if self.verbose and batch_idx < 3:
                logger.info(f"      - 감정 손실: {emotion_loss.item():.6f}, 정확도: {emotion_acc:.4f}")
        
        # 벤담 헤드
        if hasattr(self.model, 'bentham_head'):
            bentham_output = self.model.bentham_head(features)
            bentham_pred = bentham_output['bentham_scores'] if isinstance(bentham_output, dict) else bentham_output
            bentham_target = batch['bentham_label'].to(self.device)
            bentham_loss = self.model.bentham_head.compute_loss(bentham_pred, bentham_target)
            head_losses.append(bentham_loss)
            individual_losses['bentham_loss'] = bentham_loss.item()
            # accuracy 계산 (regression task - threshold 기반)
            bentham_acc = ((bentham_pred - bentham_target).abs() < 0.1).float().mean().item()
            individual_accs['bentham_acc'] = bentham_acc
            if self.verbose and batch_idx < 3:
                logger.info(f"      - 벤담 손실: {bentham_loss.item():.6f}, 정확도: {bentham_acc:.4f}")
        
        # 후회 헤드
        if hasattr(self.model, 'regret_head'):
            regret_output = self.model.regret_head(features)
            regret_pred = regret_output['regret_score'] if isinstance(regret_output, dict) else regret_output
            regret_target = batch['regret_label'].to(self.device)
            regret_loss = self.model.regret_head.compute_loss(regret_pred, regret_target)
            head_losses.append(regret_loss)
            individual_losses['regret_loss'] = regret_loss.item()
            # accuracy 계산 (regression task)
            regret_acc = ((regret_pred - regret_target).abs() < 0.1).float().mean().item()
            individual_accs['regret_acc'] = regret_acc
            if self.verbose and batch_idx < 3:
                logger.info(f"      - 후회 손실: {regret_loss.item():.6f}, 정확도: {regret_acc:.4f}")
        
        # SURD 헤드
        if hasattr(self.model, 'surd_head'):
            surd_output = self.model.surd_head(features)
            surd_pred = surd_output['surd_values'] if isinstance(surd_output, dict) else surd_output
            
            # SURD 타겟을 실제 데이터에서 계산
            batch_size = surd_pred.shape[0]
            surd_target = torch.zeros((batch_size, 4), device=self.device)
            
            # Synergy: 감정 다양성 (엔트로피 기반)
            if 'emotion_label' in batch:
                emotion_probs = F.one_hot(batch['emotion_label'].to(self.device), num_classes=7).float()
                emotion_entropy = -(emotion_probs * (emotion_probs + 1e-10).log()).sum(dim=1)
                surd_target[:, 0] = emotion_entropy / np.log(7)  # 정규화
            
            # Unique: 레이블 고유성 (one-hot 인코딩)
            if 'surd_label' in batch:
                label_unique = F.one_hot(batch['surd_label'].to(self.device), num_classes=5).float()
                surd_target[:, 1] = label_unique.max(dim=1)[0]  # 최대값 = 1.0
            
            # Redundant: 벤담 상관도 (평균과 분산)
            if 'bentham_label' in batch:
                bentham = batch['bentham_label'].to(self.device)
                bentham_mean = bentham.mean(dim=1)
                bentham_std = bentham.std(dim=1) + 1e-10
                surd_target[:, 2] = 1.0 - (bentham_std / (bentham_mean + 1e-10)).clamp(0, 1)
            
            # Deterministic: 후회 결정성 (절대값)
            if 'regret_label' in batch:
                regret = batch['regret_label'].to(self.device)
                if regret.dim() == 1:
                    regret = regret.unsqueeze(1)
                surd_target[:, 3] = regret.abs().squeeze()
            
            surd_loss = self.model.surd_head.compute_loss(surd_pred, surd_target)
            head_losses.append(surd_loss)
            individual_losses['surd_loss'] = surd_loss.item()
            # accuracy 계산 (multi-dimensional regression)
            surd_acc = ((surd_pred - surd_target).abs() < 0.1).float().mean().item()
            individual_accs['surd_acc'] = surd_acc
            if self.verbose and batch_idx < 3:
                logger.info(f"      - SURD 손실: {surd_loss.item():.6f}, 정확도: {surd_acc:.4f}")
        
        # ========== STAGE 2: NEURAL ANALYZERS ==========
        if self.verbose and batch_idx < 3:
            logger.info("    [STAGE 2] Neural Analyzer Processing")
        
        analyzer_losses = []
        dsp_output = None
        neural_emotion_output = None
        
        # Neural Emotion Analyzer 처리 (먼저)
        if hasattr(self.model, 'neural_analyzers') and 'emotion' in self.model.neural_analyzers:
            try:
                emotion_analyzer = self.model.neural_analyzers['emotion']
                neural_emotion_output = emotion_analyzer(features)
                
                if 'emotion_logits' in neural_emotion_output:
                    target = batch['emotion_label'].to(self.device)
                    if target.dim() == 1:
                        target = F.one_hot(target, num_classes=7).float()
                    emotion_loss = F.cross_entropy(neural_emotion_output['emotion_logits'], target)
                    analyzer_losses.append(emotion_loss)
                    if self.verbose and batch_idx < 3:
                        logger.info(f"      - neural_emotion 손실: {emotion_loss.item():.6f}")
            except Exception as e:
                logger.error(f"    ❌ neural_emotion 처리 실패: {e}")
        
        # DSP Simulator 처리
        if hasattr(self.model, 'dsp_simulator') and self.model.dsp_simulator:
            try:
                # DSP는 384차원 입력 필요
                if not hasattr(self, 'dsp_projection'):
                    self.dsp_projection = torch.nn.Linear(features.shape[-1], 384).to(self.device)
                
                dsp_input = self.dsp_projection(features)
                dsp_output = self.model.dsp_simulator(dsp_input)
                
                if self.verbose and batch_idx < 3:
                    logger.info(f"      - DSP 출력 처리됨")
            except Exception as e:
                logger.warning(f"    ⚠️ DSP 처리 실패: {e}")
        
        # Kalman Filter 처리 (neural_emotion + DSP 필요)
        if hasattr(self.model, 'kalman_filter') and self.model.kalman_filter and \
           dsp_output is not None and neural_emotion_output is not None:
            try:
                traditional_emotions = neural_emotion_output.get('emotion_logits', None)
                dsp_emotions = dsp_output.get('final_emotions', None) if isinstance(dsp_output, dict) else dsp_output
                
                if traditional_emotions is not None and dsp_emotions is not None:
                    kalman_output = self.model.kalman_filter(
                        traditional_emotions=traditional_emotions,
                        dsp_emotions=dsp_emotions
                    )
                    if self.verbose and batch_idx < 3:
                        logger.info(f"      - Kalman 필터 출력 처리됨")
            except Exception as e:
                logger.warning(f"    ⚠️ Kalman 처리 실패: {e}")
        
        # 나머지 Neural Analyzers 처리
        if hasattr(self.model, 'neural_analyzers'):
            for name, analyzer in self.model.neural_analyzers.items():
                if name == 'emotion':  # 이미 처리함
                    continue
                    
                try:
                    analyzer_output = analyzer(features)
                    
                    # 각 analyzer별 손실 계산
                    if 'bentham' in name and 'bentham_scores' in analyzer_output:
                        target = batch['bentham_label'].to(self.device)
                        analyzer_loss = F.mse_loss(analyzer_output['bentham_scores'], target)
                        analyzer_losses.append(analyzer_loss)
                        if self.verbose and batch_idx < 3:
                            logger.info(f"      - {name} 손실: {analyzer_loss.item():.6f}")
                    
                    elif 'regret' in name and 'regret_score' in analyzer_output:
                        target = batch['regret_label'].to(self.device)
                        analyzer_loss = F.smooth_l1_loss(analyzer_output['regret_score'], target)
                        analyzer_losses.append(analyzer_loss)
                        if self.verbose and batch_idx < 3:
                            logger.info(f"      - {name} 손실: {analyzer_loss.item():.6f}")
                    
                    elif 'surd' in name and 'surd_scores' in analyzer_output:
                        # SURD analyzer도 4차원 타겟 필요
                        batch_size = analyzer_output['surd_scores'].shape[0]
                        target = torch.zeros((batch_size, 4), device=self.device)
                        
                        # 실제 데이터로 SURD 계산 (위와 동일)
                        if 'emotion_label' in batch:
                            emotion_probs = F.one_hot(batch['emotion_label'].to(self.device), num_classes=7).float()
                            emotion_entropy = -(emotion_probs * (emotion_probs + 1e-10).log()).sum(dim=1)
                            target[:, 0] = emotion_entropy / np.log(7)
                        
                        if 'surd_label' in batch:
                            label_unique = F.one_hot(batch['surd_label'].to(self.device), num_classes=5).float()
                            target[:, 1] = label_unique.max(dim=1)[0]
                        
                        if 'bentham_label' in batch:
                            bentham = batch['bentham_label'].to(self.device)
                            bentham_mean = bentham.mean(dim=1)
                            bentham_std = bentham.std(dim=1) + 1e-10
                            target[:, 2] = 1.0 - (bentham_std / (bentham_mean + 1e-10)).clamp(0, 1)
                        
                        if 'regret_label' in batch:
                            regret = batch['regret_label'].to(self.device)
                            if regret.dim() == 1:
                                regret = regret.unsqueeze(1)
                            target[:, 3] = regret.abs().squeeze()
                        
                        analyzer_loss = F.mse_loss(analyzer_output['surd_scores'], target)
                        analyzer_losses.append(analyzer_loss)
                        if self.verbose and batch_idx < 3:
                            logger.info(f"      - {name} 손실: {analyzer_loss.item():.6f}")
                    
                except Exception as e:
                    if hasattr(self.config, 'debug') and self.config.debug:
                        logger.error(f"    {name} 처리 실패: {e}")
                    else:
                        logger.error(f"    {name} 처리 실패: {e}")
        
        # Advanced Wrappers 처리
        if hasattr(self.model, 'advanced_wrappers') and self.model.advanced_wrappers:
            for name, wrapper in self.model.advanced_wrappers.items():
                try:
                    wrapper_output = wrapper(features)
                    
                    # Advanced wrapper 손실 (간단히 처리)
                    if isinstance(wrapper_output, dict) and any(key in wrapper_output for key in ['emotion', 'bentham', 'regret', 'surd']):
                        # 적절한 손실 계산
                        pass  # TODO: wrapper별 손실 구현 필요
                        
                except Exception as e:
                    if self.config.debug:
                        logger.error(f"    {name} wrapper 처리 실패: {e}")
        
        # 전체 손실 통합 (V2처럼: 헤드 70%, Analyzer 30%)
        all_losses = head_losses + analyzer_losses
        
        if all_losses:
            if head_losses and analyzer_losses:
                head_loss = sum(head_losses) / len(head_losses)
                analyzer_loss = sum(analyzer_losses) / len(analyzer_losses)
                loss = 0.7 * head_loss + 0.3 * analyzer_loss
                if self.verbose and batch_idx < 3:
                    logger.info(f"      - 헤드 손실: {head_loss.item():.6f}")
                    logger.info(f"      - 분석기 손실: {analyzer_loss.item():.6f}")
                    logger.info(f"      - 전체 손실: {loss.item():.6f}")
            else:
                loss = sum(all_losses) / len(all_losses)
        else:
            # NO FALLBACK - 손실이 없으면 에러
            raise RuntimeError("손실 계산 실패: 헤드나 분석기가 손실을 생성하지 못함")
        
        # 메트릭 - 개별 모듈 손실 포함
        metrics = {
            'loss': loss.item(),
            'train_loss': loss.item(),  # 전체 손실 (backward 호환)
            'head_losses': len(head_losses),
            'analyzer_losses': len(analyzer_losses),
            'total_losses': len(all_losses)
        }
        
        # 개별 헤드 손실 및 정확도 추가
        metrics.update(individual_losses)
        metrics.update(individual_accs)
        
        # 백본 손실 (전체 손실과 동일하게 설정)
        metrics['backbone_loss'] = loss.item()
        metrics['backbone_acc'] = 0.0  # 백본은 별도 accuracy 없음
        
        # Neural Analyzer 손실
        if analyzer_losses:
            metrics['analyzer_loss'] = sum(al.item() for al in analyzer_losses) / len(analyzer_losses)
            metrics['analyzer_acc'] = 0.0  # Analyzer는 accuracy 계산이 복잡하므로 일단 0
        else:
            metrics['analyzer_loss'] = 0.0
            metrics['analyzer_acc'] = 0.0
        
        # validation 메트릭 (같은 값으로 설정 - 나중에 validate()에서 덮어씌워짐)
        metrics['val_loss'] = loss.item()
        metrics['val_acc'] = metrics.get('emotion_acc', 0.0)  # 대표로 emotion accuracy 사용
        
        return loss, metrics
    
    def train(self):
        """전체 학습 실행"""
        logger.info("\n" + "=" * 70)
        logger.info("🚀 학습 시작")
        logger.info("=" * 70)
        
        # LR 스윕 실행
        self.run_lr_sweep()
        
        # 60 에폭 학습
        for epoch in range(1, self.config.total_epochs + 1):
            self.current_epoch = epoch
            
            logger.info(f"\n📌 Epoch {epoch}/{self.config.total_epochs}")
            
            # 학습
            train_metrics = self.train_epoch(epoch)
            
            # 검증
            if epoch % 2 == 0:  # 짝수 에폭마다
                val_metrics = self.validate()
                logger.info(f"  Validation Loss: {val_metrics['val_loss']:.4f}")
            else:
                val_metrics = {}
            
            # 메트릭 통합 및 모듈별 그룹화
            all_metrics = {**train_metrics, **val_metrics}
            
            # 모듈별 메트릭으로 재구성 (sweet_spot_detector 호환)
            module_metrics = {
                'backbone': {
                    'loss': all_metrics.get('backbone_loss', all_metrics.get('train_loss', 0)),
                    'accuracy': all_metrics.get('backbone_acc', 0)
                },
                'emotion_head': {
                    'loss': all_metrics.get('emotion_loss', 0),
                    'accuracy': all_metrics.get('emotion_acc', 0)
                },
                'bentham_head': {
                    'loss': all_metrics.get('bentham_loss', 0),
                    'accuracy': all_metrics.get('bentham_acc', 0)
                },
                'regret_head': {
                    'loss': all_metrics.get('regret_loss', 0),
                    'accuracy': all_metrics.get('regret_acc', 0)
                },
                'surd_head': {
                    'loss': all_metrics.get('surd_loss', 0),
                    'accuracy': all_metrics.get('surd_acc', 0)
                },
                'neural_analyzers': {
                    'loss': all_metrics.get('analyzer_loss', 0),
                    'accuracy': all_metrics.get('analyzer_acc', 0)
                },
                'system': {
                    'loss': all_metrics.get('val_loss', all_metrics.get('train_loss', 0)),
                    'accuracy': all_metrics.get('val_acc', 0)
                }
            }
            
            # 디버그: 메트릭 검증
            if epoch == 1 and self.verbose:
                logger.info("\n  📊 메트릭 검증 (Epoch 1):")
                for module_name, metrics in module_metrics.items():
                    logger.info(f"    - {module_name}: loss={metrics['loss']:.4f}, acc={metrics['accuracy']:.4f}")
            
            # Sweet Spot 업데이트
            if self.config.enable_sweet_spot:
                self.sweet_spot_detector.update(
                    epoch=epoch,
                    module_metrics=module_metrics,
                    learning_rate=self.optimizer.param_groups[0]['lr']
                )
            
            # 체크포인트 저장
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                metrics=all_metrics,
                lr=self.optimizer.param_groups[0]['lr']
            )
            
            # Crossover 시스템에 체크포인트 추가
            if checkpoint_path and self.config.enable_crossover:
                self.crossover_system.add_checkpoint(
                    epoch=epoch,
                    checkpoint_path=checkpoint_path,
                    module_metrics=all_metrics
                )
            
            # 최고 성능 갱신
            if 'loss' in all_metrics and all_metrics['loss'] < self.best_loss:
                self.best_loss = all_metrics['loss']
                logger.info(f"  🏆 최고 성능 갱신: {self.best_loss:.4f}")
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ 60 에폭 학습 완료!")
        logger.info("=" * 70)
        
        # 최종 처리
        self._finalize_training()
    
    def _finalize_training(self):
        """학습 마무리 처리"""
        logger.info("\n🔧 최종 처리 시작...")
        
        # Sweet Spot 종합 분석 실행
        if self.config.enable_sweet_spot:
            logger.info("\n🎯 Sweet Spot 종합 분석 시작...")
            try:
                # 5가지 고급 분석 기법 적용
                analysis_results = self.sweet_spot_detector.analyze_all(
                    output_dir='training/sweet_spot_analysis'
                )
                
                # 기존 메서드도 호출 (호환성 유지)
                optimal_epochs = self.sweet_spot_detector.get_optimal_epochs()
                logger.info(f"  📊 모듈별 최적 에폭: {optimal_epochs}")
                
                # 추가 분석 결과 로깅
                for module, result in analysis_results.items():
                    rec = result['recommendation']
                    logger.info(f"    - {module}: Epoch {rec['epoch']} (신뢰도: {rec['confidence']:.1%})")
                    
            except Exception as e:
                logger.error(f"Sweet Spot 분석 실패: {e}")
                # 기본 분석만 수행
                analysis_results = self.sweet_spot_detector.export_analysis()
                logger.info(f"  📊 기본 분석 저장: {analysis_results['json_file']}")
        
        # Parameter Crossover 실행
        if self.config.enable_crossover and self.config.enable_sweet_spot:
            logger.info("\n🧬 Parameter Crossover 실행...")
            
            crossover_model = self.crossover_system.perform_crossover(
                model=self.model,
                optimal_epochs=optimal_epochs
            )
            
            # Crossover 모델 저장
            crossover_path = Path(self.config.checkpoint_dir) / "crossover_final.pth"
            self.crossover_system.save_crossover_result(
                model=crossover_model,
                save_path=str(crossover_path),
                metadata={'optimal_epochs': optimal_epochs}
            )
            logger.info(f"  💾 Crossover 모델 저장: {crossover_path}")
        
        # 학습 곡선 내보내기
        curves_file = self.checkpoint_manager.export_training_curves()
        logger.info(f"  📈 학습 곡선 저장: {curves_file}")
        
        # OOM 통계 저장
        if self.config.enable_oom_handler:
            oom_stats = self.oom_handler.save_stats()
            logger.info(f"  📊 OOM 통계 저장: {oom_stats}")
        
        logger.info("\n" + "=" * 70)
        logger.info("🎉 모든 작업 완료!")
        logger.info("=" * 70)


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Red Heart AI 최종 통합 학습")
    parser.add_argument('--test', action='store_true', help='테스트 모드')
    parser.add_argument('--epochs', type=int, default=60, help='학습 에폭')
    parser.add_argument('--batch-size', type=int, default=2, help='배치 사이즈')
    parser.add_argument('--lr', type=float, default=1e-4, help='학습률')
    parser.add_argument('--resume', type=str, help='체크포인트에서 재개')
    parser.add_argument('--no-param-update', action='store_true', help='파라미터 업데이트 건너뛰기 (검증용)')
    parser.add_argument('--debug', action='store_true', help='디버그 모드')
    parser.add_argument('--verbose', action='store_true', help='상세 로깅')
    parser.add_argument('--samples', type=int, help='테스트용 샘플 수 (에폭 수로 사용)')
    
    args = parser.parse_args()
    
    # 설정 생성
    config = UnifiedTrainingConfig()
    
    # args에서 설정 적용
    config.verbose = args.verbose
    
    # 테스트 모드
    if args.test:
        if args.samples:
            config.total_epochs = args.samples
            logger.info(f"⚠️ 테스트 모드: {args.samples} 에폭 실행")
        else:
            config.total_epochs = 2
            logger.info("⚠️ 테스트 모드: 2 에폭만 실행")
    else:
        config.total_epochs = args.epochs
    
    config.micro_batch_size = args.batch_size
    config.base_lr = args.lr
    
    # 디버그/상세 로깅 설정
    if args.debug or args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        config.log_interval = 1  # 매 스텝마다 로깅
        config.val_interval = 10  # 더 자주 검증
    
    # 트레이너 생성 및 실행
    trainer = UnifiedTrainer(config)
    trainer.no_param_update = args.no_param_update  # 파라미터 업데이트 플래그
    
    if args.no_param_update:
        logger.warning("⚠️ 파라미터 업데이트 비활성화 - 검증 모드")
    
    # 체크포인트에서 재개
    if args.resume:
        checkpoint = trainer.checkpoint_manager.load_checkpoint(args.resume)
        trainer.model.load_state_dict(checkpoint['model_state'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state'])
        trainer.current_epoch = checkpoint['epoch']
        logger.info(f"✅ 체크포인트에서 재개: Epoch {trainer.current_epoch}")
    
    # 학습 실행
    trainer.train()


if __name__ == "__main__":
    main()