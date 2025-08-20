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
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks
import pandas as pd

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
        
        # 모듈별 메트릭 히스토리 (train/val 분리)
        self.module_histories = defaultdict(lambda: {
            'train_losses': [],
            'val_losses': [],
            'train_accuracies': [],
            'val_accuracies': [],
            'epochs': [],
            'gradients': [],
            'learning_rates': [],
            'overfitting_scores': [],  # val_loss - train_loss
            'generalization_gaps': []  # val_acc - train_acc
        })
        
        # 모듈간 상호작용 메트릭
        self.interaction_metrics = defaultdict(lambda: {
            'synergy_scores': [],  # 모듈 조합 시너지
            'correlation_matrix': [],  # 모듈간 성능 상관관계
            'coupling_strength': [],  # 모듈간 결합도
            'information_flow': []  # 모듈간 정보 흐름
        })
        
        # Sweet Spot 정보
        self.sweet_spots = {}
        self.convergence_points = {}
        self.overfitting_points = {}
        self.interaction_sweet_spots = {}  # 모듈 조합 최적점
        
        logger.info("✅ Sweet Spot Detector 초기화")
        logger.info(f"  - 윈도우 크기: {window_size}")
        logger.info(f"  - 안정성 임계값: {stability_threshold}")
        logger.info(f"  - Patience: {patience}")
    
    def update(self, 
               epoch: int,
               train_module_metrics: Dict[str, Dict[str, float]],
               val_module_metrics: Dict[str, Dict[str, float]],
               learning_rate: float = None):
        """
        메트릭 업데이트 및 Sweet Spot 탐지 (train/val 분리)
        
        Args:
            epoch: 현재 에폭
            train_module_metrics: 학습 모듈별 메트릭 딕셔너리
            val_module_metrics: 검증 모듈별 메트릭 딕셔너리
            learning_rate: 현재 학습률
        """
        # 모든 모듈 이름 수집
        all_modules = set(train_module_metrics.keys()) | set(val_module_metrics.keys())
        
        for module_name in all_modules:
            history = self.module_histories[module_name]
            train_metrics = train_module_metrics.get(module_name, {})
            val_metrics = val_module_metrics.get(module_name, {})
            
            # 히스토리 업데이트
            history['epochs'].append(epoch)
            history['train_losses'].append(train_metrics.get('loss', 0))
            history['val_losses'].append(val_metrics.get('loss', 0))
            history['train_accuracies'].append(train_metrics.get('accuracy', 0))
            history['val_accuracies'].append(val_metrics.get('accuracy', 0))
            
            # 과적합 점수 계산 (val_loss - train_loss)
            overfitting_score = val_metrics.get('loss', 0) - train_metrics.get('loss', 0)
            history['overfitting_scores'].append(overfitting_score)
            
            # 일반화 갭 계산 (train_acc - val_acc)
            generalization_gap = train_metrics.get('accuracy', 0) - val_metrics.get('accuracy', 0)
            history['generalization_gaps'].append(generalization_gap)
            
            if 'gradient_norm' in train_metrics:
                history['gradients'].append(train_metrics['gradient_norm'])
            
            if learning_rate:
                history['learning_rates'].append(learning_rate)
            
            # Sweet Spot 탐지 (충분한 데이터가 쌓인 후)
            if epoch >= self.min_epochs:
                self._detect_sweet_spot(module_name, epoch)
                self._detect_convergence(module_name, epoch)
                self._detect_overfitting_improved(module_name, epoch)
        
        # 모듈간 상호작용 분석
        if epoch >= self.min_epochs:
            self._analyze_module_interactions(epoch, all_modules)
    
    def _detect_sweet_spot(self, module_name: str, epoch: int):
        """모듈별 Sweet Spot 탐지 (train/val 균형 고려)"""
        history = self.module_histories[module_name]
        val_losses = history['val_losses']
        train_losses = history['train_losses']
        
        if len(val_losses) < self.window_size:
            return
        
        # 최근 윈도우의 손실
        recent_val_losses = val_losses[-self.window_size:]
        recent_train_losses = train_losses[-self.window_size:]
        recent_overfitting = history['overfitting_scores'][-self.window_size:]
        
        # 조건 1: 낮은 검증 손실
        avg_val_loss = np.mean(recent_val_losses)
        avg_train_loss = np.mean(recent_train_losses)
        
        # 조건 2: 안정성 (낮은 분산)
        val_std = np.std(recent_val_losses)
        is_stable = val_std < self.stability_threshold
        
        # 조건 3: 수렴 (손실 감소율이 낮음)
        if len(val_losses) >= self.window_size * 2:
            prev_window = val_losses[-self.window_size*2:-self.window_size]
            improvement = (np.mean(prev_window) - avg_val_loss) / (np.mean(prev_window) + 1e-10)
            is_converged = abs(improvement) < 0.01  # 1% 미만 개선
        else:
            is_converged = False
        
        # 조건 4: 과적합 제어
        avg_overfitting = np.mean(recent_overfitting)
        is_not_overfitting = avg_overfitting < 0.1  # 10% 미만 차이
        
        # Sweet Spot 판단
        if is_stable and is_not_overfitting and (is_converged or avg_val_loss < 0.1):
            # 이전 Sweet Spot보다 나은지 확인
            if module_name not in self.sweet_spots or \
               avg_val_loss < self.sweet_spots[module_name]['val_loss']:
                
                self.sweet_spots[module_name] = {
                    'epoch': epoch,
                    'val_loss': avg_val_loss,
                    'train_loss': avg_train_loss,
                    'loss': avg_val_loss,  # 호환성 유지
                    'std': val_std,
                    'stable': is_stable,
                    'converged': is_converged,
                    'overfitting_score': avg_overfitting
                }
                
                logger.info(f"  🎯 Sweet Spot 발견: {module_name}")
                logger.info(f"     - 에폭: {epoch}")
                logger.info(f"     - Val Loss: {avg_val_loss:.4f} (±{val_std:.4f})")
                logger.info(f"     - Overfitting: {avg_overfitting:.4f}")
    
    def _detect_convergence(self, module_name: str, epoch: int):
        """수렴 시점 탐지 (val_loss 기준)"""
        history = self.module_histories[module_name]
        val_losses = history['val_losses']
        
        if len(val_losses) < self.patience:
            return
        
        # 최근 patience 에폭 동안의 개선 확인
        recent_val_losses = val_losses[-self.patience:]
        best_recent = min(recent_val_losses)
        
        # 개선이 거의 없으면 수렴으로 판단
        improvements = []
        for i in range(1, len(recent_val_losses)):
            if recent_val_losses[i-1] > 0:
                improvement = (recent_val_losses[i-1] - recent_val_losses[i]) / recent_val_losses[i-1]
                improvements.append(improvement)
        
        avg_improvement = np.mean(improvements) if improvements else 0
        
        if abs(avg_improvement) < 0.001:  # 0.1% 미만 개선
            if module_name not in self.convergence_points:
                self.convergence_points[module_name] = {
                    'epoch': epoch,
                    'val_loss': best_recent,
                    'train_loss': history['train_losses'][-1] if history['train_losses'] else 0,
                    'improvement_rate': avg_improvement
                }
                logger.info(f"  📊 수렴 감지: {module_name} @ epoch {epoch}")
    
    def _detect_overfitting_improved(self, module_name: str, epoch: int):
        """개선된 과적합 시점 탐지 (train/val 갭 기반)"""
        history = self.module_histories[module_name]
        val_losses = history['val_losses']
        train_losses = history['train_losses']
        overfitting_scores = history['overfitting_scores']
        
        if len(val_losses) < self.window_size * 2:
            return
        
        # 최근 윈도우의 과적합 점수
        recent_overfitting = np.mean(overfitting_scores[-self.window_size:])
        prev_overfitting = np.mean(overfitting_scores[-self.window_size*2:-self.window_size])
        
        # 검증 손실 증가 확인
        recent_val = np.mean(val_losses[-self.window_size:])
        prev_val = np.mean(val_losses[-self.window_size*2:-self.window_size])
        val_increase = (recent_val - prev_val) / (prev_val + 1e-10)
        
        # 학습 손실은 계속 감소하는지 확인
        recent_train = np.mean(train_losses[-self.window_size:])
        prev_train = np.mean(train_losses[-self.window_size*2:-self.window_size])
        train_decrease = (prev_train - recent_train) / (prev_train + 1e-10)
        
        # 과적합 조건: val loss 증가 & train loss 감소 & 과적합 점수 증가
        if (val_increase > 0.02 and  # val loss 2% 이상 증가
            train_decrease > 0.01 and  # train loss는 계속 감소
            recent_overfitting > prev_overfitting * 1.2):  # 과적합 점수 20% 증가
            
            if module_name not in self.overfitting_points:
                self.overfitting_points[module_name] = {
                    'epoch': epoch - self.window_size,  # 과적합 시작 시점
                    'val_increase': val_increase,
                    'train_decrease': train_decrease,
                    'overfitting_score': recent_overfitting
                }
                logger.warning(f"  ⚠️ 과적합 감지: {module_name} @ epoch {epoch - self.window_size}")
                logger.warning(f"     - Val 증가: {val_increase:.2%}, Train 감소: {train_decrease:.2%}")
    
    def _analyze_module_interactions(self, epoch: int, module_names: set):
        """모듈간 상호작용 분석"""
        import itertools
        
        # 모듈 쌍별 상관관계 계산
        correlation_matrix = {}
        synergy_scores = {}
        
        module_list = list(module_names)
        for mod1, mod2 in itertools.combinations(module_list, 2):
            if mod1 not in self.module_histories or mod2 not in self.module_histories:
                continue
                
            # 최근 손실값들의 상관관계
            losses1 = self.module_histories[mod1]['val_losses'][-self.window_size:]
            losses2 = self.module_histories[mod2]['val_losses'][-self.window_size:]
            
            if len(losses1) == len(losses2) and len(losses1) > 1:
                correlation = np.corrcoef(losses1, losses2)[0, 1]
                correlation_matrix[f"{mod1}-{mod2}"] = correlation
                
                # 시너지 점수: 음의 상관관계는 보완적, 양의 상관관계는 의존적
                if correlation < -0.3:  # 보완적 관계
                    synergy_scores[f"{mod1}-{mod2}"] = 1.0 - abs(correlation)
                elif correlation > 0.7:  # 강한 의존 관계
                    synergy_scores[f"{mod1}-{mod2}"] = correlation * 0.5
                else:  # 독립적 관계
                    synergy_scores[f"{mod1}-{mod2}"] = 0.7
        
        # 전체 모듈 조합의 시너지 계산
        if synergy_scores:
            avg_synergy = np.mean(list(synergy_scores.values()))
            
            # 상호작용 메트릭 저장
            self.interaction_metrics[epoch] = {
                'synergy_scores': synergy_scores,
                'correlation_matrix': correlation_matrix,
                'avg_synergy': avg_synergy,
                'module_count': len(module_list)
            }
            
            # Sweet Spot 조합 찾기
            if avg_synergy > 0.7 and epoch not in self.interaction_sweet_spots:
                self.interaction_sweet_spots[epoch] = {
                    'synergy': avg_synergy,
                    'best_pairs': sorted(synergy_scores.items(), 
                                        key=lambda x: x[1], reverse=True)[:3]
                }
                logger.info(f"  🔗 모듈 상호작용 Sweet Spot @ epoch {epoch}")
                logger.info(f"     - 평균 시너지: {avg_synergy:.3f}")
    
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
            # 기본값: 최저 검증 손실 에폭
            else:
                val_losses = self.module_histories[module_name]['val_losses']
                if val_losses:
                    optimal_epochs[module_name] = val_losses.index(min(val_losses)) + 1
        
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
        # train/val 분리된 손실 처리
        train_losses = history.get('train_losses', [])
        val_losses = history.get('val_losses', [])
        
        # validation loss를 주요 지표로 사용 (과적합 방지)
        primary_losses = val_losses if val_losses else train_losses
        
        status = {
            'total_epochs': len(history['epochs']),
            'current_train_loss': train_losses[-1] if train_losses else None,
            'current_val_loss': val_losses[-1] if val_losses else None,
            'best_train_loss': min(train_losses) if train_losses else None,
            'best_val_loss': min(val_losses) if val_losses else None,
            'best_epoch': val_losses.index(min(val_losses)) + 1 if val_losses else 
                         (train_losses.index(min(train_losses)) + 1 if train_losses else None),
            'overfitting_score': (val_losses[-1] - train_losses[-1]) if (val_losses and train_losses) else None
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
        
        # 1. 손실 곡선 (train/val 분리)
        ax = axes[0, 0]
        if history.get('train_losses'):
            ax.plot(history['epochs'], history['train_losses'], 'b-', label='Train Loss', alpha=0.7)
        if history.get('val_losses'):
            ax.plot(history['epochs'], history['val_losses'], 'r-', label='Val Loss', alpha=0.7)
        
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
        
        # 4. 손실 변화율 (validation 기준)
        ax = axes[1, 1]
        val_losses = history.get('val_losses', [])
        train_losses = history.get('train_losses', [])
        primary_losses = val_losses if val_losses else train_losses
        
        if len(primary_losses) > 1:
            loss_changes = np.diff(primary_losses)
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
    
    def _mann_kendall_test(self, data: List[float]) -> Dict:
        """Mann-Kendall 트렌드 테스트"""
        n = len(data)
        s = 0
        
        for i in range(n-1):
            for j in range(i+1, n):
                s += np.sign(data[j] - data[i])
        
        # Calculate variance
        var_s = n * (n - 1) * (2 * n + 5) / 18
        
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
        
        return {'statistic': z, 's': s}
    
    def _cusum_detection(self, data: List[float], threshold: float = None) -> List[int]:
        """CUSUM 변화점 탐지"""
        if threshold is None:
            threshold = np.std(data) * 2
        
        mean = np.mean(data)
        cusum_pos = np.zeros(len(data))
        cusum_neg = np.zeros(len(data))
        changes = []
        
        for i in range(1, len(data)):
            cusum_pos[i] = max(0, cusum_pos[i-1] + data[i] - mean - threshold/2)
            cusum_neg[i] = max(0, cusum_neg[i-1] + mean - data[i] - threshold/2)
            
            if cusum_pos[i] > threshold or cusum_neg[i] > threshold:
                changes.append(i)
                cusum_pos[i] = 0
                cusum_neg[i] = 0
        
        return changes
    
    def statistical_plateau_detection(self, losses: List[float]) -> Dict:
        """Statistical Plateau Detection using Mann-Kendall and CUSUM"""
        if len(losses) < 5:
            return {'detected': False}
        
        # Mann-Kendall Trend Test
        mk_result = self._mann_kendall_test(losses)
        
        # CUSUM Change Detection
        cusum_changes = self._cusum_detection(losses)
        
        # Find plateau region
        plateau_start = None
        plateau_end = None
        
        # Plateau: 트렌드가 없고 변화점이 없는 구간
        for i in range(len(losses) - 5):
            window = losses[i:i+5]
            window_trend = self._mann_kendall_test(window)
            
            if abs(window_trend['statistic']) < 0.5:  # No significant trend
                if plateau_start is None:
                    plateau_start = i
                plateau_end = i + 5
        
        if plateau_start is not None:
            plateau_center = (plateau_start + plateau_end) // 2
            plateau_mean = np.mean(losses[plateau_start:plateau_end])
            plateau_std = np.std(losses[plateau_start:plateau_end])
            
            return {
                'detected': True,
                'start': plateau_start,
                'end': plateau_end,
                'center': plateau_center,
                'mean_loss': plateau_mean,
                'std': plateau_std,
                'mk_statistic': mk_result['statistic'],
                'cusum_changes': cusum_changes
            }
        
        return {'detected': False}
    
    def calculate_task_metrics(self, module: str, metrics: Dict) -> Dict:
        """모듈별 Task-Specific 메트릭 계산"""
        task_scores = {}
        
        if 'head' in module or module == 'heads':
            # 헤드 통합 점수
            task_scores['emotion_score'] = np.mean(metrics.get('emotion_f1', [0]))
            task_scores['bentham_score'] = 1.0 - np.mean(metrics.get('bentham_rmse', [1.0]))
            task_scores['regret_score'] = np.mean(metrics.get('regret_accuracy', [0]))
            task_scores['surd_score'] = np.mean(metrics.get('surd_pid_acc', [0]))
            task_scores['combined'] = np.mean(list(task_scores.values()))
            
        elif 'analyzer' in module:
            # Analyzer 특화 메트릭 (validation 우선)
            val_losses = metrics.get('val_losses', [])
            train_losses = metrics.get('train_losses', [])
            losses = val_losses if val_losses else train_losses
            task_scores['stability'] = 1.0 / (1.0 + np.std(losses if losses else [1.0]))
            task_scores['convergence'] = self._calculate_convergence_rate(losses)
            
        elif 'kalman' in module or 'dsp' in module:
            # DSP/Kalman 특화 메트릭
            task_scores['tracking_accuracy'] = 1.0 - np.mean(metrics.get('tracking_error', [1.0]))
            task_scores['filter_stability'] = 1.0 / (1.0 + np.std(metrics.get('filter_output', [1.0])))
        
        else:
            # 기본 메트릭
            val_losses = metrics.get('val_losses', [])
            train_losses = metrics.get('train_losses', [])
            losses = val_losses if val_losses else train_losses
            task_scores['accuracy'] = np.mean(metrics.get('val_accuracies', metrics.get('accuracies', [0])))
            task_scores['loss_improvement'] = self._calculate_improvement(losses)
        
        return task_scores
    
    def _calculate_convergence_rate(self, losses: List[float]) -> float:
        """수렴 속도 계산"""
        if len(losses) < 2:
            return 0.0
        
        improvements = []
        for i in range(1, len(losses)):
            if losses[i-1] > 0:
                improvement = (losses[i-1] - losses[i]) / losses[i-1]
                improvements.append(improvement)
        
        return np.mean(improvements) if improvements else 0.0
    
    def _calculate_improvement(self, values: List[float]) -> float:
        """개선도 계산"""
        if len(values) < 2:
            return 0.0
        
        initial = np.mean(values[:3]) if len(values) >= 3 else values[0]
        final = np.mean(values[-3:]) if len(values) >= 3 else values[-1]
        
        if initial > 0:
            return (initial - final) / initial
        return 0.0
    
    def mcda_analysis(self, module: str, metrics: Dict) -> Dict:
        """Multi-Criteria Decision Analysis"""
        
        # 기준별 점수 계산 (validation 우선)
        val_losses = metrics.get('val_losses', [])
        train_losses = metrics.get('train_losses', [])
        val_accs = metrics.get('val_accuracies', [])
        train_accs = metrics.get('train_accuracies', [])
        
        losses = val_losses if val_losses else train_losses
        accuracies = val_accs if val_accs else train_accs
        
        criteria = {
            'loss': 1.0 - np.array(losses if losses else [1.0]),  # Lower is better
            'accuracy': np.array(accuracies if accuracies else [0]),
            'stability': self._calculate_stability_scores(metrics),
            'gradient_health': self._calculate_gradient_health(metrics)
        }
        
        # 정규화 (0-1 범위)
        normalized = {}
        for key, values in criteria.items():
            if len(values) > 0 and np.std(values) > 0:
                normalized[key] = (values - np.min(values)) / (np.max(values) - np.min(values))
            else:
                normalized[key] = values
        
        # 가중치
        weights = {
            'loss': 0.30,
            'accuracy': 0.40,
            'stability': 0.15,
            'gradient_health': 0.15
        }
        
        # MCDA 점수 계산
        mcda_scores = np.zeros(len(metrics.get('epochs', [])))
        for key, weight in weights.items():
            if key in normalized and len(normalized[key]) == len(mcda_scores):
                mcda_scores += weight * normalized[key]
        
        # 최적 epoch 찾기
        best_epoch_idx = np.argmax(mcda_scores) if len(mcda_scores) > 0 else 0
        
        return {
            'scores': mcda_scores.tolist(),
            'best_epoch_idx': int(best_epoch_idx),
            'best_epoch': metrics.get('epochs', [])[best_epoch_idx] if best_epoch_idx < len(metrics.get('epochs', [])) else -1,
            'best_score': float(mcda_scores[best_epoch_idx]) if len(mcda_scores) > 0 else 0.0,
            'weights': weights
        }
    
    def _calculate_stability_scores(self, metrics: Dict) -> np.ndarray:
        """안정성 점수 계산"""
        val_losses = metrics.get('val_losses', [])
        train_losses = metrics.get('train_losses', [])
        losses = val_losses if val_losses else train_losses
        if len(losses) < 3:
            return np.zeros(len(losses))
        
        stability_scores = []
        for i in range(len(losses)):
            start = max(0, i-2)
            end = min(len(losses), i+3)
            window = losses[start:end]
            
            # 낮은 분산 = 높은 안정성
            stability = 1.0 / (1.0 + np.std(window))
            stability_scores.append(stability)
        
        return np.array(stability_scores)
    
    def _calculate_gradient_health(self, metrics: Dict) -> np.ndarray:
        """Gradient Health 점수 계산"""
        grad_norms = metrics.get('gradients', [])
        if not grad_norms:
            return np.zeros(len(metrics.get('epochs', [])))
        
        health_scores = []
        for norm in grad_norms:
            # Gradient가 너무 크거나 작으면 불건전
            if norm > 0:
                if 0.001 < norm < 10.0:  # 건전한 범위
                    health = 1.0
                elif norm < 0.001:  # Vanishing
                    health = norm / 0.001
                else:  # Exploding
                    health = 10.0 / norm
            else:
                health = 0.0
            health_scores.append(health)
        
        return np.array(health_scores)
    
    def ensemble_voting(self, module: str, analyses: Dict) -> Dict:
        """여러 분석 기법의 앙상블 투표"""
        candidates = {}
        
        # 각 기법의 추천 수집
        if analyses.get('plateau', {}).get('detected'):
            candidates['plateau'] = analyses['plateau']['center']
        
        if 'best_epoch_idx' in analyses.get('mcda', {}):
            candidates['mcda'] = analyses['mcda']['best_epoch_idx']
        
        # Task metric 최고점
        task_scores = analyses.get('task_scores', {})
        if task_scores and 'combined' in task_scores:
            candidates['task'] = task_scores.get('best_idx', 0)
        
        # Minimum loss (validation 우선)
        module_history = self.module_histories.get(module, {})
        val_losses = module_history.get('val_losses', [])
        train_losses = module_history.get('train_losses', [])
        losses = val_losses if val_losses else train_losses
        if losses:
            candidates['min_loss'] = np.argmin(losses)
        
        # 투표 집계
        if not candidates:
            return {'selected_epoch': -1, 'confidence': 0.0}
        
        # 가장 많은 표를 받은 epoch
        vote_counts = Counter(candidates.values())
        winner, votes = vote_counts.most_common(1)[0]
        
        # 신뢰도 계산
        confidence = votes / len(candidates) if candidates else 0.0
        
        epochs = self.module_histories.get(module, {}).get('epochs', [])
        
        return {
            'candidates': candidates,
            'selected_epoch_idx': int(winner),
            'selected_epoch': epochs[winner] if winner < len(epochs) else -1,
            'votes': votes,
            'total_voters': len(candidates),
            'confidence': float(confidence)
        }
    
    def analyze_all(self, output_dir: str = 'analysis_results') -> Dict:
        """
        학습 완료 후 전체 분석 실행
        
        Args:
            output_dir: 결과 저장 디렉토리
            
        Returns:
            분석 결과 딕셔너리
        """
        logger.info("\n" + "=" * 70)
        logger.info("🎯 Sweet Spot 종합 분석 시작")
        logger.info("=" * 70)
        
        # 디버그: 수집된 메트릭 확인
        logger.debug("📊 수집된 메트릭 확인:")
        for module_name, history in self.module_histories.items():
            train_losses = history.get('train_losses', [])
            val_losses = history.get('val_losses', [])
            
            if train_losses or val_losses:
                if val_losses:
                    logger.debug(f"  - {module_name}: {len(val_losses)}개 에폭, "
                               f"Val: 첫={val_losses[0]:.4f}, 마지막={val_losses[-1]:.4f}")
                if train_losses:
                    logger.debug(f"    Train: 첫={train_losses[0]:.4f}, 마지막={train_losses[-1]:.4f}")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        analysis_results = {}
        
        # 모듈별 분석
        for module_name in self.module_histories.keys():
            logger.info(f"\n🔍 분석 중: {module_name}")
            
            metrics = self.module_histories[module_name]
            analyses = {}
            
            # 1. Statistical Plateau Detection (validation 우선)
            val_losses = metrics.get('val_losses', [])
            train_losses = metrics.get('train_losses', [])
            losses = val_losses if val_losses else train_losses
            analyses['plateau'] = self.statistical_plateau_detection(losses)
            
            # 2. Task-Specific Metrics
            task_scores = self.calculate_task_metrics(module_name, metrics)
            analyses['task_scores'] = task_scores
            
            # 3. MCDA
            analyses['mcda'] = self.mcda_analysis(module_name, metrics)
            
            # 4. Ensemble Voting
            analyses['voting'] = self.ensemble_voting(module_name, analyses)
            
            # 종합
            result = {
                'module': module_name,
                'metrics': metrics,
                'analyses': analyses,
                'recommendation': {
                    'epoch_idx': analyses['voting']['selected_epoch_idx'],
                    'epoch': analyses['voting']['selected_epoch'],
                    'confidence': analyses['voting']['confidence'],
                    'reasoning': self._generate_reasoning(module_name, analyses)
                }
            }
            
            analysis_results[module_name] = result
        
        # 시각화 생성
        self._generate_visualizations(analysis_results, output_path)
        
        # 결과 저장
        self._save_analysis_results(analysis_results, output_path)
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ Sweet Spot 분석 완료!")
        logger.info(f"📁 결과 저장 위치: {output_path}")
        logger.info("=" * 70)
        
        # 요약 출력
        print("\n📊 Sweet Spot Recommendations:")
        print("-" * 50)
        for module, result in analysis_results.items():
            rec = result['recommendation']
            print(f"{module:20s}: Epoch {rec['epoch']:3d} (Confidence: {rec['confidence']:.1%})")
        print("-" * 50)
        
        return analysis_results
    
    def _generate_reasoning(self, module: str, analyses: Dict) -> List[str]:
        """추천 근거 생성"""
        reasons = []
        
        if analyses.get('plateau', {}).get('detected'):
            plateau = analyses['plateau']
            reasons.append(f"Plateau 구간 탐지 (Epoch {plateau['start']}-{plateau['end']})")
        
        if analyses.get('mcda', {}).get('best_score', 0) > 0.8:
            reasons.append(f"MCDA 점수 우수 ({analyses['mcda']['best_score']:.3f})")
        
        if analyses.get('voting', {}).get('confidence', 0) > 0.6:
            reasons.append(f"높은 투표 신뢰도 ({analyses['voting']['confidence']:.1%})")
        
        task_scores = analyses.get('task_scores', {})
        if task_scores.get('combined', 0) > 0.7:
            reasons.append(f"Task 메트릭 우수 ({task_scores['combined']:.3f})")
        
        return reasons
    
    def _generate_visualizations(self, analysis_results: Dict, output_path: Path):
        """시각화 생성"""
        logger.info("\n📈 시각화 생성 중...")
        
        viz_dir = output_path / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        for module, result in analysis_results.items():
            self._plot_module_analysis(module, result, viz_dir)
        
        # 종합 히트맵
        self._plot_summary_heatmap(analysis_results, viz_dir)
    
    def _plot_module_analysis(self, module: str, result: Dict, viz_dir: Path):
        """모듈별 분석 시각화"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'{module} Sweet Spot Analysis', fontsize=16)
            
            metrics = result['metrics']
            analyses = result['analyses']
            
            # 1. Loss curve with plateau
            ax = axes[0, 0]
            epochs = metrics.get('epochs', [])
            val_losses = metrics.get('val_losses', [])
            train_losses = metrics.get('train_losses', [])
            losses = val_losses if val_losses else train_losses
            
            if epochs and losses:
                ax.plot(epochs, losses, 'b-', label='Training Loss')
                
                if analyses.get('plateau', {}).get('detected'):
                    plateau = analyses['plateau']
                    ax.axvspan(epochs[plateau['start']], epochs[plateau['end']], 
                              alpha=0.3, color='green', label='Plateau')
                    ax.axvline(epochs[plateau['center']], color='red', 
                              linestyle='--', label='Plateau Center')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 2. MCDA Scores
            ax = axes[0, 1]
            mcda_scores = analyses.get('mcda', {}).get('scores', [])
            
            if epochs and mcda_scores:
                ax.plot(epochs[:len(mcda_scores)], mcda_scores, 'g-', label='MCDA Score')
                best_idx = analyses.get('mcda', {}).get('best_epoch_idx', 0)
                if best_idx < len(epochs) and best_idx < len(mcda_scores):
                    ax.scatter(epochs[best_idx], mcda_scores[best_idx], 
                              color='red', s=100, label='Best MCDA')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('MCDA Score')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 3. Accuracies
            ax = axes[1, 0]
            accuracies = metrics.get('accuracies', [])
            
            if epochs and accuracies:
                ax.plot(epochs[:len(accuracies)], accuracies, label='Accuracy')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 4. Voting Results
            ax = axes[1, 1]
            voting = analyses.get('voting', {})
            
            if voting.get('candidates'):
                candidates = list(voting['candidates'].keys())
                values = list(voting['candidates'].values())
                colors = ['green' if v == voting['selected_epoch_idx'] else 'blue' for v in values]
                ax.bar(candidates, values, color=colors)
                ax.set_xlabel('Analysis Method')
                ax.set_ylabel('Recommended Epoch Index')
                ax.set_title(f"Final: Epoch {voting.get('selected_epoch', -1)} (Confidence: {voting.get('confidence', 0):.1%})")
            
            plt.tight_layout()
            plt.savefig(viz_dir / f'{module}_analysis.png', dpi=150)
            plt.close()
            
        except Exception as e:
            logger.warning(f"시각화 생성 실패 ({module}): {e}")
    
    def _plot_summary_heatmap(self, analysis_results: Dict, viz_dir: Path):
        """종합 히트맵"""
        try:
            modules = []
            recommended_epochs = []
            confidences = []
            
            for module, result in analysis_results.items():
                modules.append(module)
                recommended_epochs.append(result['recommendation']['epoch'])
                confidences.append(result['recommendation']['confidence'])
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Epoch recommendations
            ax1.barh(modules, recommended_epochs, color='steelblue')
            ax1.set_xlabel('Recommended Epoch')
            ax1.set_title('Sweet Spot Epochs by Module')
            ax1.grid(True, alpha=0.3)
            
            # Confidence scores
            ax2.barh(modules, confidences, color='coral')
            ax2.set_xlabel('Confidence Score')
            ax2.set_title('Recommendation Confidence')
            ax2.set_xlim([0, 1])
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(viz_dir / 'summary_sweetspots.png', dpi=150)
            plt.close()
            
        except Exception as e:
            logger.warning(f"종합 히트맵 생성 실패: {e}")
    
    def _save_analysis_results(self, analysis_results: Dict, output_path: Path):
        """분석 결과 저장"""
        # JSON 저장
        json_path = output_path / 'sweet_spot_analysis.json'
        with open(json_path, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        logger.info(f"📁 JSON 결과 저장: {json_path}")
        
        # Markdown 리포트 생성
        self._generate_markdown_report(analysis_results, output_path)
    
    def _generate_markdown_report(self, analysis_results: Dict, output_path: Path):
        """Markdown 형식 리포트 생성"""
        report_path = output_path / 'sweet_spot_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# 🎯 Sweet Spot Analysis Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary table
            f.write("## 📊 Summary\n\n")
            f.write("| Module | Recommended Epoch | Confidence | Key Reasoning |\n")
            f.write("|--------|------------------|------------|---------------|\n")
            
            for module, result in analysis_results.items():
                rec = result['recommendation']
                reasons = ', '.join(rec['reasoning'][:2]) if rec['reasoning'] else 'N/A'
                f.write(f"| {module} | {rec['epoch']} | {rec['confidence']:.1%} | {reasons} |\n")
            
            # Detailed analysis
            f.write("\n## 🔍 Detailed Analysis\n\n")
            
            for module, result in analysis_results.items():
                f.write(f"### Module: {module}\n\n")
                
                analyses = result['analyses']
                
                # Plateau
                if analyses.get('plateau', {}).get('detected'):
                    plateau = analyses['plateau']
                    f.write(f"**Plateau Detection:**\n")
                    f.write(f"- Range: Epoch {plateau['start']}-{plateau['end']}\n")
                    f.write(f"- Center: Epoch {plateau['center']}\n")
                    f.write(f"- Mean Loss: {plateau['mean_loss']:.4f} (±{plateau['std']:.4f})\n\n")
                else:
                    f.write("**Plateau Detection:** Not detected\n\n")
                
                # MCDA
                mcda = analyses.get('mcda', {})
                f.write(f"**MCDA Analysis:**\n")
                f.write(f"- Best Epoch: {mcda.get('best_epoch', -1)}\n")
                f.write(f"- Best Score: {mcda.get('best_score', 0):.3f}\n\n")
                
                # Voting
                voting = analyses.get('voting', {})
                f.write(f"**Ensemble Voting:**\n")
                f.write(f"- Selected: Epoch {voting.get('selected_epoch', -1)}\n")
                f.write(f"- Confidence: {voting.get('confidence', 0):.1%}\n")
                f.write(f"- Votes: {voting.get('votes', 0)}/{voting.get('total_voters', 0)}\n\n")
                
                f.write("---\n\n")
            
            # Threshold recommendations
            f.write("## 🎯 Recommended Thresholds for Automation\n\n")
            f.write("```python\n")
            f.write("# Based on empirical analysis\n")
            f.write("thresholds = {\n")
            
            # Calculate empirical thresholds
            all_plateau_stds = []
            for result in analysis_results.values():
                if result['analyses'].get('plateau', {}).get('detected'):
                    all_plateau_stds.append(result['analyses']['plateau']['std'])
            
            if all_plateau_stds:
                f.write(f"    'plateau_variance': {np.mean(all_plateau_stds):.4f},\n")
            else:
                f.write(f"    'plateau_variance': 0.01,  # Default\n")
            
            f.write("    'stability_window': 5,\n")
            f.write("    'mcda_weights': {\n")
            f.write("        'loss': 0.30,\n")
            f.write("        'accuracy': 0.40,\n")
            f.write("        'stability': 0.15,\n")
            f.write("        'gradient_health': 0.15\n")
            f.write("    },\n")
            
            # Average confidence
            avg_confidence = np.mean([r['recommendation']['confidence'] 
                                     for r in analysis_results.values()])
            f.write(f"    'min_confidence': {avg_confidence * 0.8:.2f}\n")
            f.write("}\n```\n\n")
            
            # Next steps
            f.write("## 📝 Next Steps\n\n")
            f.write("1. Review the recommendations above\n")
            f.write("2. Manually combine modules using recommended epochs\n")
            f.write("3. Evaluate combined model performance\n")
            f.write("4. Adjust thresholds based on results\n")
            f.write("5. Enable automated sweet spot detection\n")
        
        logger.info(f"📄 Markdown 리포트 저장: {report_path}")
    
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