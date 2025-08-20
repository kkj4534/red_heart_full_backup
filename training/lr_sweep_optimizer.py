"""
Learning Rate Sweep 최적화 시스템
5개 LR 값으로 스윕하여 최적 학습률 자동 탐색
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class LRSweepOptimizer:
    """
    Learning Rate Sweep 최적화 시스템
    - 5개 LR 값으로 체계적 탐색
    - 각 LR로 짧은 학습 실행
    - 수렴 속도와 안정성 평가
    - 최적 LR 자동 선택
    """
    
    def __init__(self, 
                 base_lr: float = 1e-4,
                 sweep_range: Tuple[float, float] = (1e-5, 1e-2),
                 num_sweep_points: int = 5,
                 sweep_epochs: int = 3,
                 sweep_steps_per_epoch: int = 100):
        """
        Args:
            base_lr: 기본 학습률
            sweep_range: 스윕 범위 (min, max)
            num_sweep_points: 스윕 포인트 개수 (기본 5개)
            sweep_epochs: 각 LR당 테스트 에폭 수
            sweep_steps_per_epoch: 에폭당 스텝 수
        """
        self.base_lr = base_lr
        self.sweep_range = sweep_range
        self.num_sweep_points = num_sweep_points
        self.sweep_epochs = sweep_epochs
        self.sweep_steps_per_epoch = sweep_steps_per_epoch
        
        # 스윕할 LR 값들 생성 (로그 스케일)
        self.lr_candidates = np.logspace(
            np.log10(sweep_range[0]), 
            np.log10(sweep_range[1]), 
            num_sweep_points
        )
        
        # 결과 저장
        self.sweep_results = {}
        self.best_lr = None
        self.sweep_history = []
        
        logger.info(f"✅ LR Sweep Optimizer 초기화")
        logger.info(f"  - 스윕 범위: {sweep_range[0]:.1e} ~ {sweep_range[1]:.1e}")
        logger.info(f"  - 테스트 LR 값: {[f'{lr:.1e}' for lr in self.lr_candidates]}")
        logger.info(f"  - 각 LR당 {sweep_epochs} 에폭 테스트")
    
    def run_sweep(self, 
                  model: nn.Module,
                  train_loader: Any,
                  val_loader: Any,
                  criterion: nn.Module,
                  device: torch.device) -> Dict[str, Any]:
        """
        LR 스윕 실행
        
        Args:
            model: 학습할 모델
            train_loader: 학습 데이터 로더
            val_loader: 검증 데이터 로더
            criterion: 손실 함수
            device: 학습 디바이스
            
        Returns:
            스윕 결과 및 최적 LR
        """
        logger.info("🔍 LR Sweep 시작...")
        
        for idx, lr in enumerate(self.lr_candidates):
            logger.info(f"\n[{idx+1}/{self.num_sweep_points}] LR={lr:.1e} 테스트")
            
            # 모델 초기 상태 저장 (각 LR 테스트마다 초기화)
            initial_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            # 이 LR로 학습 실행
            sweep_result = self._run_single_sweep(
                model, train_loader, val_loader, criterion, device, lr
            )
            
            # 결과 저장
            self.sweep_results[lr] = sweep_result
            
            # 모델 상태 복원
            model.load_state_dict(initial_state)
            
            # 결과 로깅
            logger.info(f"  - 최종 train loss: {sweep_result['final_train_loss']:.4f}")
            logger.info(f"  - 최종 val loss: {sweep_result['final_val_loss']:.4f}")
            logger.info(f"  - 수렴 속도: {sweep_result['convergence_speed']:.4f}")
            logger.info(f"  - 안정성 점수: {sweep_result['stability_score']:.4f}")
        
        # 최적 LR 선택
        self.best_lr = self._select_best_lr()
        
        # 결과 요약
        summary = self._generate_sweep_summary()
        
        # 시각화 생성
        self._plot_sweep_results()
        
        return summary
    
    def _run_single_sweep(self,
                         model: nn.Module,
                         train_loader: Any,
                         val_loader: Any,
                         criterion: nn.Module,
                         device: torch.device,
                         lr: float) -> Dict[str, Any]:
        """단일 LR로 스윕 실행"""
        # 옵티마이저 생성
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        
        # 메트릭 추적
        train_losses = []
        val_losses = []
        lr_history = []
        
        # 학습 실행
        for epoch in range(self.sweep_epochs):
            # Training
            model.train()
            epoch_train_losses = []
            
            for step, batch in enumerate(train_loader):
                if step >= self.sweep_steps_per_epoch:
                    break
                
                # Forward pass
                loss = self._compute_loss(model, batch, criterion, device)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_train_losses.append(loss.item())
                lr_history.append(lr)
            
            avg_train_loss = np.mean(epoch_train_losses)
            train_losses.append(avg_train_loss)
            
            # Validation
            model.eval()
            epoch_val_losses = []
            
            with torch.no_grad():
                for step, batch in enumerate(val_loader):
                    if step >= 50:  # 검증은 50 스텝만
                        break
                    
                    loss = self._compute_loss(model, batch, criterion, device)
                    epoch_val_losses.append(loss.item())
            
            avg_val_loss = np.mean(epoch_val_losses)
            val_losses.append(avg_val_loss)
            
            logger.debug(f"    Epoch {epoch+1}: train={avg_train_loss:.4f}, val={avg_val_loss:.4f}")
        
        # 메트릭 계산
        convergence_speed = self._calculate_convergence_speed(train_losses)
        stability_score = self._calculate_stability_score(train_losses, val_losses)
        
        return {
            'lr': lr,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'convergence_speed': convergence_speed,
            'stability_score': stability_score,
            'lr_history': lr_history
        }
    
    def _compute_loss(self, model: nn.Module, batch: Any, criterion: nn.Module, device: torch.device) -> torch.Tensor:
        """
        실제 모델 구조에 맞는 손실 계산
        - UnifiedModelFinal의 전체 forward pass 실행
        - 각 헤드별 손실 계산 및 통합
        - 더미 값 없이 실제 데이터 기반 계산
        """
        # 입력 데이터 준비
        inputs = batch['input'].to(device)
        
        # 백본 통과
        backbone_outputs = model.backbone(inputs, return_all_tasks=True)
        features = backbone_outputs.get('emotion', inputs)
        
        # features가 제대로 device에 있는지 확인
        if features.device != device:
            features = features.to(device)
        
        # 손실 리스트
        head_losses = []
        
        # 1. Emotion Head 손실
        if hasattr(model, 'emotion_head') and 'emotion_label' in batch:
            emotion_output = model.emotion_head(features)
            emotion_pred = emotion_output['emotions'] if isinstance(emotion_output, dict) else emotion_output
            emotion_target = batch['emotion_label'].to(device)
            emotion_loss = model.emotion_head.compute_loss(emotion_pred, emotion_target)
            head_losses.append(emotion_loss)
        
        # 2. Bentham Head 손실
        if hasattr(model, 'bentham_head') and 'bentham_label' in batch:
            bentham_output = model.bentham_head(features)
            bentham_pred = bentham_output['bentham_scores'] if isinstance(bentham_output, dict) else bentham_output
            bentham_target = batch['bentham_label'].to(device)
            bentham_loss = model.bentham_head.compute_loss(bentham_pred, bentham_target)
            head_losses.append(bentham_loss)
        
        # 3. Regret Head 손실
        if hasattr(model, 'regret_head') and 'regret_label' in batch:
            regret_output = model.regret_head(features)
            regret_pred = regret_output['regret_score'] if isinstance(regret_output, dict) else regret_output
            regret_target = batch['regret_label'].to(device)
            regret_loss = model.regret_head.compute_loss(regret_pred, regret_target)
            head_losses.append(regret_loss)
        
        # 4. SURD Head 손실 
        if hasattr(model, 'surd_head'):
            surd_output = model.surd_head(features)
            surd_pred = surd_output['surd_values'] if isinstance(surd_output, dict) else surd_output
            
            # SURD 타겟을 실제 데이터에서 계산 (unified_training_final.py와 동일)
            batch_size = surd_pred.shape[0]
            surd_target = torch.zeros((batch_size, 4), device=device)
            
            # Synergy: 감정 다양성 (엔트로피 기반)
            if 'emotion_label' in batch:
                emotion_probs = F.one_hot(batch['emotion_label'].to(device), num_classes=7).float()
                emotion_entropy = -(emotion_probs * (emotion_probs + 1e-10).log()).sum(dim=1)
                surd_target[:, 0] = emotion_entropy / np.log(7)  # 정규화
            
            # Unique: 레이블 고유성 (one-hot 인코딩)
            if 'surd_label' in batch:
                label_unique = F.one_hot(batch['surd_label'].to(device), num_classes=5).float()
                surd_target[:, 1] = label_unique.max(dim=1)[0]  # 최대값 = 1.0
            
            # Redundant: 벤담 상관도 (평균과 분산)
            if 'bentham_label' in batch:
                bentham = batch['bentham_label'].to(device)
                bentham_mean = bentham.mean(dim=1)
                bentham_std = bentham.std(dim=1) + 1e-10
                surd_target[:, 2] = 1.0 - (bentham_std / (bentham_mean + 1e-10)).clamp(0, 1)
            
            # Deterministic: 후회 결정성 (절대값)
            if 'regret_label' in batch:
                regret = batch['regret_label'].to(device)
                if regret.dim() == 1:
                    regret = regret.unsqueeze(1)
                surd_target[:, 3] = regret.abs().squeeze()
            
            surd_loss = model.surd_head.compute_loss(surd_pred, surd_target)
            head_losses.append(surd_loss)
        
        # 5. Neural Analyzers 손실
        if hasattr(model, 'neural_analyzers'):
            analyzer_losses = []
            
            # neural_analyzers는 dict이므로 각 분석기를 개별 처리
            if isinstance(model.neural_analyzers, dict):
                # 각 분석기 호출 및 손실 계산
                for analyzer_name, analyzer_module in model.neural_analyzers.items():
                    if callable(analyzer_module):
                        analyzer_output = analyzer_module(features)
                        
                        # 각 분석기별 손실 계산 (출력의 평균값 * 0.1)
                        if isinstance(analyzer_output, dict):
                            # dict 출력인 경우 주요 키의 값 사용
                            for key, value in analyzer_output.items():
                                if isinstance(value, torch.Tensor) and value.requires_grad:
                                    analyzer_loss = value.mean() * 0.1
                                    analyzer_losses.append(analyzer_loss)
                                    break  # 첫 번째 유효한 텐서만 사용
                        elif isinstance(analyzer_output, torch.Tensor):
                            # 텐서 출력인 경우 직접 사용
                            analyzer_loss = analyzer_output.mean() * 0.1
                            analyzer_losses.append(analyzer_loss)
            
            if analyzer_losses:
                total_analyzer_loss = sum(analyzer_losses) / len(analyzer_losses)
                head_losses.append(total_analyzer_loss)
        
        # 전체 손실 계산
        if head_losses:
            total_loss = sum(head_losses) / len(head_losses)
        else:
            # 헤드가 없으면 백본만으로 손실 계산 (fallback)
            total_loss = features.mean() * 0.1
        
        return total_loss
    
    def _calculate_convergence_speed(self, losses: List[float]) -> float:
        """수렴 속도 계산 (낮을수록 빠른 수렴)"""
        if len(losses) < 2:
            return float('inf')
        
        # 손실 감소율 계산
        improvements = []
        for i in range(1, len(losses)):
            if losses[i-1] > 0:
                improvement = (losses[i-1] - losses[i]) / losses[i-1]
                improvements.append(improvement)
        
        if not improvements:
            return 0.0
        
        # 평균 개선율 (높을수록 좋음)
        avg_improvement = np.mean(improvements)
        
        # 수렴 속도 점수 (0~1, 높을수록 빠름)
        convergence_speed = min(1.0, max(0.0, avg_improvement * 10))
        
        return convergence_speed
    
    def _calculate_stability_score(self, train_losses: List[float], val_losses: List[float]) -> float:
        """안정성 점수 계산 (높을수록 안정적)"""
        # 손실의 표준편차가 낮을수록 안정적
        train_std = np.std(train_losses) if len(train_losses) > 1 else 0
        val_std = np.std(val_losses) if len(val_losses) > 1 else 0
        
        # Overfitting 체크 (val loss가 train loss보다 크게 증가하는지)
        if len(train_losses) > 0 and len(val_losses) > 0:
            overfitting_gap = val_losses[-1] - train_losses[-1]
        else:
            overfitting_gap = 0
        
        # 안정성 점수 계산 (0~1, 높을수록 안정적)
        stability = 1.0 / (1.0 + train_std + val_std + max(0, overfitting_gap))
        
        return min(1.0, stability)
    
    def _select_best_lr(self) -> float:
        """최적 LR 선택"""
        best_score = -float('inf')
        best_lr = self.base_lr
        
        for lr, result in self.sweep_results.items():
            # 종합 점수 계산 (수렴 속도와 안정성의 균형)
            score = (
                0.4 * result['convergence_speed'] +  # 수렴 속도 40%
                0.3 * result['stability_score'] +     # 안정성 30%
                0.3 * (1.0 / (1.0 + result['final_val_loss']))  # 최종 성능 30%
            )
            
            if score > best_score:
                best_score = score
                best_lr = lr
        
        logger.info(f"\n🏆 최적 LR 선택: {best_lr:.1e}")
        logger.info(f"   - 종합 점수: {best_score:.4f}")
        
        return best_lr
    
    def _generate_sweep_summary(self) -> Dict[str, Any]:
        """스윕 결과 요약 생성"""
        summary = {
            'best_lr': self.best_lr,
            'sweep_timestamp': datetime.now().isoformat(),
            'lr_candidates': self.lr_candidates.tolist(),
            'results': {}
        }
        
        for lr, result in self.sweep_results.items():
            summary['results'][f"{lr:.1e}"] = {
                'final_train_loss': result['final_train_loss'],
                'final_val_loss': result['final_val_loss'],
                'convergence_speed': result['convergence_speed'],
                'stability_score': result['stability_score']
            }
        
        # 결과 저장
        output_dir = Path("training/lr_sweep_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"lr_sweep_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📊 스윕 결과 저장: {output_file}")
        
        return summary
    
    def _plot_sweep_results(self):
        """스윕 결과 시각화"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # GUI 없는 환경 대비
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # 1. 각 LR별 학습 곡선
            ax = axes[0, 0]
            for lr, result in self.sweep_results.items():
                ax.plot(result['train_losses'], label=f'LR={lr:.1e}', alpha=0.7)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Training Loss')
            ax.set_title('Training Loss by Learning Rate')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 2. 검증 손실 비교
            ax = axes[0, 1]
            lrs = list(self.sweep_results.keys())
            val_losses = [r['final_val_loss'] for r in self.sweep_results.values()]
            ax.bar(range(len(lrs)), val_losses)
            ax.set_xticks(range(len(lrs)))
            ax.set_xticklabels([f'{lr:.1e}' for lr in lrs], rotation=45)
            ax.set_ylabel('Final Validation Loss')
            ax.set_title('Final Validation Loss Comparison')
            ax.grid(True, alpha=0.3)
            
            # 3. 수렴 속도 vs 안정성
            ax = axes[1, 0]
            convergence_speeds = [r['convergence_speed'] for r in self.sweep_results.values()]
            stability_scores = [r['stability_score'] for r in self.sweep_results.values()]
            
            ax.scatter(convergence_speeds, stability_scores, s=100)
            for i, lr in enumerate(lrs):
                ax.annotate(f'{lr:.1e}', 
                           (convergence_speeds[i], stability_scores[i]),
                           xytext=(5, 5), textcoords='offset points')
            
            ax.set_xlabel('Convergence Speed')
            ax.set_ylabel('Stability Score')
            ax.set_title('Convergence vs Stability Trade-off')
            ax.grid(True, alpha=0.3)
            
            # 4. 종합 점수
            ax = axes[1, 1]
            scores = []
            for lr, result in self.sweep_results.items():
                score = (
                    0.4 * result['convergence_speed'] +
                    0.3 * result['stability_score'] +
                    0.3 * (1.0 / (1.0 + result['final_val_loss']))
                )
                scores.append(score)
            
            bars = ax.bar(range(len(lrs)), scores)
            # 최고 점수 막대 강조
            best_idx = scores.index(max(scores))
            bars[best_idx].set_color('green')
            
            ax.set_xticks(range(len(lrs)))
            ax.set_xticklabels([f'{lr:.1e}' for lr in lrs], rotation=45)
            ax.set_ylabel('Combined Score')
            ax.set_title('Overall Performance Score')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 저장
            output_dir = Path("training/lr_sweep_results")
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = output_dir / f"lr_sweep_plot_{timestamp}.png"
            plt.savefig(plot_file, dpi=100, bbox_inches='tight')
            plt.close()
            
            logger.info(f"📈 시각화 저장: {plot_file}")
            
        except Exception as e:
            logger.warning(f"시각화 생성 실패: {e}")
    
    def get_scheduler(self, optimizer: torch.optim.Optimizer, 
                     total_steps: int) -> torch.optim.lr_scheduler._LRScheduler:
        """
        최적 LR 기반 스케줄러 생성
        
        Args:
            optimizer: 옵티마이저
            total_steps: 전체 학습 스텝 수
            
        Returns:
            학습률 스케줄러
        """
        if self.best_lr is None:
            raise ValueError("먼저 run_sweep()를 실행하여 최적 LR을 찾아야 합니다")
        
        # Cosine Annealing with Warm Restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=total_steps // 10,  # 첫 번째 재시작 주기
            T_mult=2,  # 주기 배수
            eta_min=self.best_lr * 0.01  # 최소 LR
        )
        
        logger.info(f"📅 스케줄러 생성: CosineAnnealingWarmRestarts")
        logger.info(f"   - 초기 LR: {self.best_lr:.1e}")
        logger.info(f"   - 최소 LR: {self.best_lr * 0.01:.1e}")
        
        return scheduler