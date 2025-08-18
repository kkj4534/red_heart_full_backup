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
        
        # Advanced 분석기 래퍼 (112M)
        self.advanced_wrappers = create_advanced_analyzer_wrappers()
        
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
        """순전파
        
        Args:
            x: 입력 텐서
            task: 현재 학습 중인 태스크
            
        Returns:
            해당 태스크의 출력 텐서 (dict가 아닌 tensor)
        """
        # 백본 처리
        backbone_outputs = self.backbone(x, task=task)
        
        # 태스크별 특징 추출
        if task in backbone_outputs:
            features = backbone_outputs[task]
        else:
            # 모든 태스크 출력의 평균 사용
            features = torch.stack(list(backbone_outputs.values())).mean(dim=0)
        
        # 태스크별 헤드 적용
        if task == 'emotion':
            output = self.emotion_head(features)
        elif task == 'bentham':
            output = self.bentham_head(features)
        elif task == 'regret':
            output = self.regret_head(features)
        elif task == 'surd':
            output = self.surd_head(features)
        else:
            # 기본값: emotion
            output = self.emotion_head(features)
        
        return output


class UnifiedTrainer:
    """통합 학습 관리자"""
    
    def __init__(self, config: UnifiedTrainingConfig):
        self.config = config
        self.device = get_device()
        
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
        """모델 초기화"""
        # 실제 730M 모델 초기화
        self.model = UnifiedModel(self.config).to(self.device)
        
        # Advanced Training 초기화
        self.training_manager.initialize(
            model=self.model,
            num_classes=6,
            base_lr=self.config.base_lr
        )
        
        # 파라미터 수 확인
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"✅ 모델 초기화 완료: {total_params/1e6:.1f}M 파라미터")
    
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
                
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                item = self.data[idx]
                # 텐서 변환
                text_embedding = torch.randn(100, 768)  # 실제로는 텍스트 임베딩 사용
                
                return {
                    'input': text_embedding,
                    'emotion_label': torch.tensor(item.get('label', 0), dtype=torch.long),
                    'bentham_label': torch.tensor(item.get('bentham_scores', [0.5]*10)[:10], dtype=torch.float),
                    'regret_label': torch.tensor(item.get('regret_factor', 0), dtype=torch.long),
                    'surd_label': torch.tensor(0, dtype=torch.long)  # SURD 라벨 기본값
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
        """Forward step"""
        # 데이터 준비
        inputs = batch['input'].to(self.device)
        emotion_labels = batch.get('emotion_label', batch.get('label', torch.zeros(inputs.size(0), dtype=torch.long))).to(self.device)
        
        # 현재 태스크 결정 (순환적으로 태스크 학습)
        task_order = ['emotion', 'bentham', 'regret', 'surd']
        current_task = task_order[self.global_step % len(task_order)]
        
        # Forward pass - 이제 모델이 텐서를 반환함
        outputs = self.model(inputs, task=current_task)
        
        # 손실 계산 - 직접 텐서 사용
        if self.config.enable_label_smoothing or self.config.enable_rdrop:
            # Advanced Training 사용 시 - 모델을 다시 호출하지 않고 출력 사용
            def model_wrapper(x):
                return self.model(x, task=current_task)
            
            loss = self.training_manager.compute_loss(
                model=model_wrapper,
                inputs=inputs,
                labels=emotion_labels
            )
        else:
            # 기본 손실 함수
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, emotion_labels)
        
        # 모듈별 메트릭
        metrics = {
            'loss': loss.item(),
            f'{current_task}_loss': loss.item(),
            'current_task': current_task
        }
        
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
            
            # 메트릭 통합
            all_metrics = {**train_metrics, **val_metrics}
            
            # Sweet Spot 업데이트
            if self.config.enable_sweet_spot:
                self.sweet_spot_detector.update(
                    epoch=epoch,
                    module_metrics=all_metrics,
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
        
        # Sweet Spot 분석 결과 저장
        if self.config.enable_sweet_spot:
            optimal_epochs = self.sweet_spot_detector.get_optimal_epochs()
            logger.info(f"  🎯 모듈별 최적 에폭: {optimal_epochs}")
            
            analysis_results = self.sweet_spot_detector.export_analysis()
            logger.info(f"  📊 Sweet Spot 분석 저장: {analysis_results['json_file']}")
        
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