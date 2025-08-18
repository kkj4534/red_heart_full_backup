"""
향상된 체크포인트 매니저
60 에폭 학습에서 30개 체크포인트 저장 및 Sweet Spot 탐지 지원
"""

import torch
import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class EnhancedCheckpointManager:
    """
    향상된 체크포인트 매니저
    - 60 에폭 중 30개 체크포인트 저장 (짝수 에폭마다)
    - 모듈별 최적 성능 추적
    - Sweet Spot 자동 탐지
    - Parameter Crossover 지원
    """
    
    def __init__(self, 
                 checkpoint_dir: str = "training/checkpoints",
                 max_checkpoints: int = 30,
                 save_interval: int = 2):
        """
        Args:
            checkpoint_dir: 체크포인트 저장 디렉토리
            max_checkpoints: 최대 체크포인트 개수 (기본 30개)
            save_interval: 저장 간격 (기본 2 에폭마다)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.save_interval = save_interval
        
        # 메트릭 추적
        self.metrics_history = {
            'global': [],  # 전체 메트릭
            'modules': {}  # 모듈별 메트릭
        }
        
        # Sweet Spot 추적
        self.sweet_spots = {}
        
        # 체크포인트 메타데이터
        self.checkpoint_metadata = []
        self.load_metadata()
        
        logger.info(f"✅ Enhanced CheckpointManager 초기화")
        logger.info(f"  - 저장 디렉토리: {self.checkpoint_dir}")
        logger.info(f"  - 최대 체크포인트: {self.max_checkpoints}개")
        logger.info(f"  - 저장 간격: {self.save_interval} 에폭마다")
    
    def load_metadata(self):
        """메타데이터 로드"""
        metadata_file = self.checkpoint_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.checkpoint_metadata = json.load(f)
                logger.info(f"  - 기존 메타데이터 로드: {len(self.checkpoint_metadata)}개 체크포인트")
    
    def save_metadata(self):
        """메타데이터 저장"""
        metadata_file = self.checkpoint_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.checkpoint_metadata, f, indent=2)
    
    def should_save_checkpoint(self, epoch: int) -> bool:
        """현재 에폭에서 체크포인트를 저장해야 하는지 확인"""
        # 짝수 에폭마다 저장 (60 에폭 중 30개)
        return epoch % self.save_interval == 0
    
    def save_checkpoint(self,
                       epoch: int,
                       model: Any,
                       optimizer: Any,
                       scheduler: Any,
                       metrics: Dict[str, Any],
                       lr: float) -> Optional[str]:
        """
        체크포인트 저장
        
        Args:
            epoch: 현재 에폭
            model: 모델 객체
            optimizer: 옵티마이저
            scheduler: 스케줄러
            metrics: 현재 메트릭
            lr: 현재 학습률
            
        Returns:
            저장된 체크포인트 경로 (저장하지 않으면 None)
        """
        if not self.should_save_checkpoint(epoch):
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_epoch_{epoch:04d}_lr_{lr:.6f}_{timestamp}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # 체크포인트 데이터 구성
        checkpoint_data = {
            'epoch': epoch,
            'lr': lr,
            'timestamp': timestamp,
            'model_state': self._extract_modular_states(model),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'sweet_spots': self.sweet_spots.copy()
        }
        
        # 저장
        torch.save(checkpoint_data, checkpoint_path)
        
        # 메타데이터 업데이트
        metadata_entry = {
            'epoch': epoch,
            'lr': lr,
            'timestamp': timestamp,
            'path': str(checkpoint_path),
            'metrics': metrics,
            'file_size_mb': checkpoint_path.stat().st_size / (1024 * 1024)
        }
        self.checkpoint_metadata.append(metadata_entry)
        self.save_metadata()
        
        # 메트릭 히스토리 업데이트
        self._update_metrics_history(epoch, metrics)
        
        # Sweet Spot 탐지
        self._detect_sweet_spots(epoch, metrics)
        
        logger.info(f"💾 체크포인트 저장: {checkpoint_path}")
        logger.info(f"   - 에폭: {epoch}, LR: {lr:.6f}")
        logger.info(f"   - 메트릭: loss={metrics.get('loss', 'N/A'):.4f}")
        
        # 오래된 체크포인트 정리
        self._cleanup_old_checkpoints()
        
        return str(checkpoint_path)
    
    def _extract_modular_states(self, model: Any) -> Dict[str, Any]:
        """모델을 모듈별로 분리하여 state_dict 추출"""
        modular_states = {}
        
        # Group A: Backbone + Heads
        group_a_modules = ['backbone', 'emotion_head', 'bentham_head', 
                          'regret_head', 'surd_head']
        for module_name in group_a_modules:
            if hasattr(model, module_name):
                module = getattr(model, module_name)
                if module is not None:
                    modular_states[module_name] = module.state_dict()
        
        # Group B: Neural Analyzers
        group_b_modules = ['neural_emotion', 'neural_bentham', 
                          'neural_regret', 'neural_surd']
        for module_name in group_b_modules:
            if hasattr(model, module_name):
                module = getattr(model, module_name)
                if module is not None:
                    modular_states[module_name] = module.state_dict()
        
        # Group C: DSP + Kalman
        group_c_modules = ['emotion_dsp', 'kalman_filter']
        for module_name in group_c_modules:
            if hasattr(model, module_name):
                module = getattr(model, module_name)
                if module is not None:
                    modular_states[module_name] = module.state_dict()
        
        # Independent: Advanced Analyzers
        independent_modules = ['advanced_emotion', 'advanced_regret', 
                              'advanced_surd', 'advanced_bentham']
        for module_name in independent_modules:
            if hasattr(model, module_name):
                module = getattr(model, module_name)
                if module is not None:
                    modular_states[module_name] = module.state_dict()
        
        return modular_states
    
    def _update_metrics_history(self, epoch: int, metrics: Dict[str, Any]):
        """메트릭 히스토리 업데이트"""
        # 전체 메트릭
        self.metrics_history['global'].append({
            'epoch': epoch,
            'metrics': metrics.copy()
        })
        
        # 모듈별 메트릭
        for key, value in metrics.items():
            if '_' in key:  # 모듈 이름이 포함된 메트릭
                module_name = key.split('_')[0]
                if module_name not in self.metrics_history['modules']:
                    self.metrics_history['modules'][module_name] = []
                self.metrics_history['modules'][module_name].append({
                    'epoch': epoch,
                    'value': value
                })
    
    def _detect_sweet_spots(self, epoch: int, metrics: Dict[str, Any]):
        """Sweet Spot 자동 탐지"""
        for module_name in self.metrics_history['modules']:
            history = self.metrics_history['modules'][module_name]
            if len(history) >= 5:  # 최소 5개 데이터포인트 필요
                recent_values = [h['value'] for h in history[-5:]]
                
                # Sweet Spot 조건: 최근 5 에폭 중 변동이 작고 성능이 좋음
                std_dev = np.std(recent_values)
                mean_value = np.mean(recent_values)
                
                # 이전 Sweet Spot과 비교
                if module_name not in self.sweet_spots or \
                   mean_value < self.sweet_spots[module_name]['value']:
                    if std_dev < 0.01:  # 안정성 조건
                        self.sweet_spots[module_name] = {
                            'epoch': epoch,
                            'value': mean_value,
                            'std': std_dev
                        }
                        logger.info(f"  🎯 Sweet Spot 발견: {module_name} @ epoch {epoch}")
    
    def _cleanup_old_checkpoints(self):
        """오래된 체크포인트 정리"""
        if len(self.checkpoint_metadata) > self.max_checkpoints:
            # 가장 오래된 체크포인트 삭제
            oldest = self.checkpoint_metadata[0]
            checkpoint_path = Path(oldest['path'])
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.info(f"  🗑️ 오래된 체크포인트 삭제: {checkpoint_path.name}")
            self.checkpoint_metadata.pop(0)
            self.save_metadata()
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """
        체크포인트 로드
        
        Args:
            checkpoint_path: 로드할 체크포인트 경로 (None이면 최신)
            
        Returns:
            체크포인트 데이터
        """
        if checkpoint_path is None:
            # 최신 체크포인트 찾기
            if not self.checkpoint_metadata:
                raise ValueError("저장된 체크포인트가 없습니다")
            checkpoint_path = self.checkpoint_metadata[-1]['path']
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {checkpoint_path}")
        
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        logger.info(f"✅ 체크포인트 로드: {checkpoint_path}")
        logger.info(f"   - 에폭: {checkpoint_data['epoch']}")
        logger.info(f"   - LR: {checkpoint_data['lr']:.6f}")
        
        return checkpoint_data
    
    def get_best_checkpoint(self, metric_name: str = 'loss') -> Optional[str]:
        """
        특정 메트릭 기준 최고 성능 체크포인트 찾기
        
        Args:
            metric_name: 평가 메트릭 이름
            
        Returns:
            최고 성능 체크포인트 경로
        """
        if not self.checkpoint_metadata:
            return None
        
        best_checkpoint = None
        best_value = float('inf')
        
        for metadata in self.checkpoint_metadata:
            if metric_name in metadata['metrics']:
                value = metadata['metrics'][metric_name]
                if value < best_value:
                    best_value = value
                    best_checkpoint = metadata['path']
        
        if best_checkpoint:
            logger.info(f"🏆 최고 성능 체크포인트: {best_checkpoint}")
            logger.info(f"   - {metric_name}: {best_value:.4f}")
        
        return best_checkpoint
    
    def get_sweet_spot_summary(self) -> Dict[str, Any]:
        """Sweet Spot 요약 정보 반환"""
        summary = {
            'detected_spots': len(self.sweet_spots),
            'modules': {}
        }
        
        for module_name, spot_info in self.sweet_spots.items():
            summary['modules'][module_name] = {
                'optimal_epoch': spot_info['epoch'],
                'metric_value': spot_info['value'],
                'stability': spot_info['std']
            }
        
        return summary
    
    def export_training_curves(self, output_dir: str = "training/plots"):
        """학습 곡선 데이터 내보내기"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 전체 메트릭 히스토리 저장
        curves_data = {
            'global_metrics': self.metrics_history['global'],
            'module_metrics': self.metrics_history['modules'],
            'sweet_spots': self.sweet_spots,
            'checkpoint_metadata': self.checkpoint_metadata
        }
        
        output_file = output_dir / f"training_curves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(curves_data, f, indent=2)
        
        logger.info(f"📊 학습 곡선 데이터 내보내기: {output_file}")
        
        return str(output_file)