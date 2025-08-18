"""
Sweet Spot Detector - 모듈별 최적 에폭 탐지 시스템
각 모듈이 최고 성능을 보이는 에폭을 자동으로 탐지
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import logging
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class SweetSpotDetector:
    """
    Sweet Spot 탐지 시스템
    - 모듈별 최적 성능 에폭 탐지
    - 수렴 패턴 분석
    - 과적합 시점 감지
    - 안정성 평가
    """
    
    def __init__(self,
                 window_size: int = 5,
                 stability_threshold: float = 0.01,
                 patience: int = 10,
                 min_epochs: int = 10):
        """
        Args:
            window_size: 이동 평균 윈도우 크기
            stability_threshold: 안정성 판단 임계값
            patience: 성능 개선 없이 기다리는 에폭 수
            min_epochs: 최소 학습 에폭 (이전에는 Sweet Spot 판단 안함)
        """
        self.window_size = window_size
        self.stability_threshold = stability_threshold
        self.patience = patience
        self.min_epochs = min_epochs
        
        # 모듈별 메트릭 히스토리
        self.module_histories = defaultdict(lambda: {
            'losses': [],
            'accuracies': [],
            'epochs': [],
            'gradients': [],
            'learning_rates': []
        })
        
        # Sweet Spot 정보
        self.sweet_spots = {}
        self.convergence_points = {}
        self.overfitting_points = {}
        
        logger.info("✅ Sweet Spot Detector 초기화")
        logger.info(f"  - 윈도우 크기: {window_size}")
        logger.info(f"  - 안정성 임계값: {stability_threshold}")
        logger.info(f"  - Patience: {patience}")
    
    def update(self, 
               epoch: int,
               module_metrics: Dict[str, Dict[str, float]],
               learning_rate: float = None):
        """
        메트릭 업데이트 및 Sweet Spot 탐지
        
        Args:
            epoch: 현재 에폭
            module_metrics: 모듈별 메트릭 딕셔너리
            learning_rate: 현재 학습률
        """
        for module_name, metrics in module_metrics.items():
            history = self.module_histories[module_name]
            
            # 히스토리 업데이트
            history['epochs'].append(epoch)
            history['losses'].append(metrics.get('loss', 0))
            history['accuracies'].append(metrics.get('accuracy', 0))
            
            if 'gradient_norm' in metrics:
                history['gradients'].append(metrics['gradient_norm'])
            
            if learning_rate:
                history['learning_rates'].append(learning_rate)
            
            # Sweet Spot 탐지 (충분한 데이터가 쌓인 후)
            if epoch >= self.min_epochs:
                self._detect_sweet_spot(module_name, epoch)
                self._detect_convergence(module_name, epoch)
                self._detect_overfitting(module_name, epoch)
    
    def _detect_sweet_spot(self, module_name: str, epoch: int):
        """모듈별 Sweet Spot 탐지"""
        history = self.module_histories[module_name]
        losses = history['losses']
        
        if len(losses) < self.window_size:
            return
        
        # 최근 윈도우의 손실
        recent_losses = losses[-self.window_size:]
        
        # 조건 1: 낮은 손실
        avg_loss = np.mean(recent_losses)
        
        # 조건 2: 안정성 (낮은 분산)
        loss_std = np.std(recent_losses)
        is_stable = loss_std < self.stability_threshold
        
        # 조건 3: 수렴 (손실 감소율이 낮음)
        if len(losses) >= self.window_size * 2:
            prev_window = losses[-self.window_size*2:-self.window_size]
            improvement = (np.mean(prev_window) - avg_loss) / np.mean(prev_window)
            is_converged = abs(improvement) < 0.01  # 1% 미만 개선
        else:
            is_converged = False
        
        # 조건 4: 과적합 없음 (검증 손실이 증가하지 않음)
        # 실제 구현에서는 val_loss도 추적 필요
        is_not_overfitting = True  # 현재는 간단히 처리
        
        # Sweet Spot 판단
        if is_stable and (is_converged or avg_loss < 0.1):
            # 이전 Sweet Spot보다 나은지 확인
            if module_name not in self.sweet_spots or \
               avg_loss < self.sweet_spots[module_name]['loss']:
                
                self.sweet_spots[module_name] = {
                    'epoch': epoch,
                    'loss': avg_loss,
                    'std': loss_std,
                    'stable': is_stable,
                    'converged': is_converged
                }
                
                logger.info(f"  🎯 Sweet Spot 발견: {module_name}")
                logger.info(f"     - 에폭: {epoch}")
                logger.info(f"     - 손실: {avg_loss:.4f} (±{loss_std:.4f})")
    
    def _detect_convergence(self, module_name: str, epoch: int):
        """수렴 시점 탐지"""
        history = self.module_histories[module_name]
        losses = history['losses']
        
        if len(losses) < self.patience:
            return
        
        # 최근 patience 에폭 동안의 개선 확인
        recent_losses = losses[-self.patience:]
        best_recent = min(recent_losses)
        
        # 개선이 거의 없으면 수렴으로 판단
        improvements = []
        for i in range(1, len(recent_losses)):
            if recent_losses[i-1] > 0:
                improvement = (recent_losses[i-1] - recent_losses[i]) / recent_losses[i-1]
                improvements.append(improvement)
        
        avg_improvement = np.mean(improvements) if improvements else 0
        
        if abs(avg_improvement) < 0.001:  # 0.1% 미만 개선
            if module_name not in self.convergence_points:
                self.convergence_points[module_name] = {
                    'epoch': epoch,
                    'loss': best_recent,
                    'improvement_rate': avg_improvement
                }
                logger.info(f"  📊 수렴 감지: {module_name} @ epoch {epoch}")
    
    def _detect_overfitting(self, module_name: str, epoch: int):
        """과적합 시점 탐지"""
        history = self.module_histories[module_name]
        losses = history['losses']
        
        # 실제로는 train/val loss 비교 필요
        # 여기서는 간단한 휴리스틱 사용
        if len(losses) >= self.window_size * 3:
            # 손실이 다시 증가하기 시작하면 과적합 의심
            recent = np.mean(losses[-self.window_size:])
            previous = np.mean(losses[-self.window_size*2:-self.window_size])
            
            if recent > previous * 1.05:  # 5% 이상 증가
                if module_name not in self.overfitting_points:
                    self.overfitting_points[module_name] = {
                        'epoch': epoch - self.window_size,  # 증가 시작 시점
                        'loss_increase': (recent - previous) / previous
                    }
                    logger.warning(f"  ⚠️ 과적합 감지: {module_name} @ epoch {epoch - self.window_size}")
    
    def get_optimal_epochs(self) -> Dict[str, int]:
        """
        각 모듈의 최적 에폭 반환
        
        Returns:
            모듈별 최적 에폭 딕셔너리
        """
        optimal_epochs = {}
        
        for module_name in self.module_histories.keys():
            # Sweet Spot이 있으면 그것을 사용
            if module_name in self.sweet_spots:
                optimal_epochs[module_name] = self.sweet_spots[module_name]['epoch']
            # 수렴점이 있으면 그것을 사용
            elif module_name in self.convergence_points:
                optimal_epochs[module_name] = self.convergence_points[module_name]['epoch']
            # 과적합 직전 사용
            elif module_name in self.overfitting_points:
                optimal_epochs[module_name] = max(1, self.overfitting_points[module_name]['epoch'] - 1)
            # 기본값: 최저 손실 에폭
            else:
                losses = self.module_histories[module_name]['losses']
                if losses:
                    optimal_epochs[module_name] = losses.index(min(losses)) + 1
        
        return optimal_epochs
    
    def get_module_status(self, module_name: str) -> Dict[str, Any]:
        """
        특정 모듈의 현재 상태 반환
        
        Args:
            module_name: 모듈 이름
            
        Returns:
            모듈 상태 정보
        """
        if module_name not in self.module_histories:
            return {'status': 'not_found'}
        
        history = self.module_histories[module_name]
        status = {
            'total_epochs': len(history['epochs']),
            'current_loss': history['losses'][-1] if history['losses'] else None,
            'best_loss': min(history['losses']) if history['losses'] else None,
            'best_epoch': history['losses'].index(min(history['losses'])) + 1 if history['losses'] else None
        }
        
        # Sweet Spot 정보
        if module_name in self.sweet_spots:
            status['sweet_spot'] = self.sweet_spots[module_name]
        
        # 수렴 정보
        if module_name in self.convergence_points:
            status['convergence'] = self.convergence_points[module_name]
        
        # 과적합 정보
        if module_name in self.overfitting_points:
            status['overfitting'] = self.overfitting_points[module_name]
        
        return status
    
    def plot_module_analysis(self, module_name: str, save_path: Optional[str] = None):
        """
        모듈 분석 결과 시각화
        
        Args:
            module_name: 모듈 이름
            save_path: 저장 경로 (None이면 표시만)
        """
        if module_name not in self.module_histories:
            logger.warning(f"모듈 {module_name}의 히스토리가 없습니다")
            return
        
        history = self.module_histories[module_name]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 손실 곡선
        ax = axes[0, 0]
        ax.plot(history['epochs'], history['losses'], 'b-', label='Loss', alpha=0.7)
        
        # Sweet Spot 표시
        if module_name in self.sweet_spots:
            spot = self.sweet_spots[module_name]
            ax.axvline(x=spot['epoch'], color='g', linestyle='--', label=f"Sweet Spot (epoch {spot['epoch']})")
        
        # 수렴점 표시
        if module_name in self.convergence_points:
            conv = self.convergence_points[module_name]
            ax.axvline(x=conv['epoch'], color='orange', linestyle='--', label=f"Convergence (epoch {conv['epoch']})")
        
        # 과적합 시점 표시
        if module_name in self.overfitting_points:
            overfit = self.overfitting_points[module_name]
            ax.axvline(x=overfit['epoch'], color='r', linestyle='--', label=f"Overfitting (epoch {overfit['epoch']})")
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{module_name} - Loss Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 정확도 곡선 (있는 경우)
        ax = axes[0, 1]
        if history['accuracies']:
            ax.plot(history['epochs'], history['accuracies'], 'g-', alpha=0.7)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'{module_name} - Accuracy Curve')
            ax.grid(True, alpha=0.3)
        
        # 3. 그래디언트 노름 (있는 경우)
        ax = axes[1, 0]
        if history['gradients']:
            ax.plot(history['epochs'][:len(history['gradients'])], 
                   history['gradients'], 'r-', alpha=0.7)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Gradient Norm')
            ax.set_title(f'{module_name} - Gradient Norm')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        
        # 4. 손실 변화율
        ax = axes[1, 1]
        if len(history['losses']) > 1:
            loss_changes = np.diff(history['losses'])
            ax.plot(history['epochs'][1:], loss_changes, 'b-', alpha=0.7)
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss Change')
            ax.set_title(f'{module_name} - Loss Change Rate')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Sweet Spot Analysis: {module_name}', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"📊 분석 플롯 저장: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def export_analysis(self, output_dir: str = "training/sweet_spot_analysis"):
        """
        전체 분석 결과 내보내기
        
        Args:
            output_dir: 출력 디렉토리
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. JSON 형식으로 전체 데이터 저장
        analysis_data = {
            'timestamp': timestamp,
            'sweet_spots': self.sweet_spots,
            'convergence_points': self.convergence_points,
            'overfitting_points': self.overfitting_points,
            'optimal_epochs': self.get_optimal_epochs(),
            'module_summaries': {}
        }
        
        for module_name in self.module_histories.keys():
            analysis_data['module_summaries'][module_name] = self.get_module_status(module_name)
        
        json_file = output_dir / f"sweet_spot_analysis_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        logger.info(f"📊 Sweet Spot 분석 결과 저장: {json_file}")
        
        # 2. 각 모듈별 플롯 생성
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        for module_name in self.module_histories.keys():
            plot_file = plots_dir / f"{module_name}_{timestamp}.png"
            self.plot_module_analysis(module_name, str(plot_file))
        
        # 3. 요약 리포트 생성
        report_file = output_dir / f"sweet_spot_report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("Sweet Spot Analysis Report\n")
            f.write(f"Generated: {timestamp}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("Optimal Epochs by Module:\n")
            f.write("-" * 30 + "\n")
            for module, epoch in self.get_optimal_epochs().items():
                f.write(f"  {module}: Epoch {epoch}\n")
            
            f.write("\n")
            f.write("Sweet Spots Detected:\n")
            f.write("-" * 30 + "\n")
            for module, info in self.sweet_spots.items():
                f.write(f"  {module}:\n")
                f.write(f"    - Epoch: {info['epoch']}\n")
                f.write(f"    - Loss: {info['loss']:.4f} (±{info['std']:.4f})\n")
                f.write(f"    - Stable: {info['stable']}\n")
                f.write(f"    - Converged: {info['converged']}\n")
            
            f.write("\n")
            f.write("Convergence Points:\n")
            f.write("-" * 30 + "\n")
            for module, info in self.convergence_points.items():
                f.write(f"  {module}: Epoch {info['epoch']} (Loss: {info['loss']:.4f})\n")
            
            if self.overfitting_points:
                f.write("\n")
                f.write("⚠️ Overfitting Warnings:\n")
                f.write("-" * 30 + "\n")
                for module, info in self.overfitting_points.items():
                    f.write(f"  {module}: Started at epoch {info['epoch']}\n")
        
        logger.info(f"📄 Sweet Spot 리포트 생성: {report_file}")
        
        return {
            'json_file': str(json_file),
            'report_file': str(report_file),
            'plots_dir': str(plots_dir)
        }