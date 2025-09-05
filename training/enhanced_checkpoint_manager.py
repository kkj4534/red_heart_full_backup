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
                 save_interval: int = 1):
        """
        Args:
            checkpoint_dir: 체크포인트 저장 디렉토리
            max_checkpoints: 최대 체크포인트 개수 (기본 30개)
            save_interval: 저장 간격 (기본 1 에폭마다 - 모든 에폭 저장)
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
        # save_interval=1일 때 모든 에폭 저장
        return epoch % self.save_interval == 0
    
    def should_keep_optimizer(self, epoch: int) -> bool:
        """optimizer_state를 유지해야 하는지 결정
        
        50 에폭 전략:
        - 10, 20, 30, 40, 50: 마일스톤 유지 (재개 가능)
        - 나머지: 제거 (공간 절약, 크로스오버만 가능)
        """
        # 10 에폭 단위로 optimizer_state 저장
        if epoch % 10 == 0:
            return True
        return False
    
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
        
        # 체크포인트 데이터 구성 (CPU로 이동하여 GPU 메모리 절약)
        # Optimizer state를 CPU로 이동 (중첩된 구조 처리)
        optimizer_state_cpu = {}
        opt_state = optimizer.state_dict()
        
        # state와 param_groups 처리
        if 'state' in opt_state:
            optimizer_state_cpu['state'] = {}
            for key, state in opt_state['state'].items():
                optimizer_state_cpu['state'][key] = {
                    k: v.cpu() if torch.is_tensor(v) else v
                    for k, v in state.items()
                }
        
        if 'param_groups' in opt_state:
            optimizer_state_cpu['param_groups'] = opt_state['param_groups']
        
        # optimizer_state 저장 여부 결정
        keep_optimizer = self.should_keep_optimizer(epoch)
        
        checkpoint_data = {
            'epoch': epoch,
            'lr': lr,
            'timestamp': timestamp,
            'model_state': self._extract_modular_states(model),  # 이미 CPU로 이동됨
            'scheduler_state': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,  # 현재 에폭의 메트릭만
            # sweet_spots 제거 - 누적 방지
        }
        
        # optimizer_state는 조건부로 추가
        if keep_optimizer:
            checkpoint_data['optimizer_state'] = optimizer_state_cpu
            logger.info(f"   - Optimizer state 유지 (에폭 {epoch})")
        else:
            logger.info(f"   - Optimizer state 제거 (공간 절약)")
        
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
        # optimizer 저장 정보는 이미 위에서 출력됨
        # loss 값이 있는지 확인하고 적절한 포맷 적용
        loss_val = metrics.get('loss', 'N/A')
        if isinstance(loss_val, (int, float)) and loss_val != float('inf'):
            logger.info(f"   - 메트릭: loss={loss_val:.4f}")
        else:
            logger.info(f"   - 메트릭: loss={loss_val}")
        
        # 오래된 체크포인트 정리
        self._cleanup_old_checkpoints()
        
        return str(checkpoint_path)
    
    def _extract_modular_states(self, model: Any) -> Dict[str, Any]:
        """모델을 모듈별로 분리하여 state_dict 추출 (GPU→CPU 이동)"""
        modular_states = {}
        
        # Group A: Backbone + Heads
        group_a_modules = ['backbone', 'emotion_head', 'bentham_head', 
                          'regret_head', 'surd_head']
        for module_name in group_a_modules:
            if hasattr(model, module_name):
                module = getattr(model, module_name)
                if module is not None:
                    # GPU → CPU 이동하여 메모리 절약
                    modular_states[module_name] = {
                        k: v.cpu() for k, v in module.state_dict().items()
                    }
        
        # Neural Analyzers Dict 처리 (368M 파라미터)
        if hasattr(model, 'neural_analyzers'):
            neural_analyzers = getattr(model, 'neural_analyzers')
            # nn.ModuleDict도 처리 가능하도록 수정
            if hasattr(neural_analyzers, 'items'):  # dict-like 객체인지 확인
                # dict 전체를 하나의 모듈로 저장
                neural_states = {}
                for analyzer_name, analyzer_module in neural_analyzers.items():
                    if analyzer_module is not None:
                        # 각 analyzer의 state를 nested dict로 저장
                        neural_states[analyzer_name] = {
                            k: v.cpu() for k, v in analyzer_module.state_dict().items()
                        }
                if neural_states:
                    modular_states['neural_analyzers'] = neural_states
                    logger.debug(f"  ✓ neural_analyzers dict 저장: {len(neural_states)}개 분석기")
            else:
                # dict가 아닌 경우 기존 방식 (fallback)
                group_b_modules = ['neural_emotion', 'neural_bentham', 
                                  'neural_regret', 'neural_surd']
                for module_name in group_b_modules:
                    if hasattr(model, module_name):
                        module = getattr(model, module_name)
                        if module is not None:
                            modular_states[module_name] = {
                                k: v.cpu() for k, v in module.state_dict().items()
                            }
        
        # Group C: Phase Networks (중요: 크로스오버 필수)
        phase_modules = ['phase0_net', 'phase2_net', 'hierarchical_integrator']
        for module_name in phase_modules:
            if hasattr(model, module_name):
                module = getattr(model, module_name)
                if module is not None:
                    modular_states[module_name] = {
                        k: v.cpu() for k, v in module.state_dict().items()
                    }
                    logger.debug(f"  ✓ {module_name} 저장 완료")
        
        # Group D: DSP + Kalman
        group_d_modules = ['dsp_simulator', 'kalman_filter']
        for module_name in group_d_modules:
            if hasattr(model, module_name):
                module = getattr(model, module_name)
                if module is not None:
                    # GPU → CPU 이동하여 메모리 절약
                    modular_states[module_name] = {
                        k: v.cpu() for k, v in module.state_dict().items()
                    }
        
        # Advanced Wrappers Dict 처리 (중요: 크로스오버 필수)
        if hasattr(model, 'advanced_wrappers'):
            advanced_wrappers = getattr(model, 'advanced_wrappers')
            if advanced_wrappers is not None and hasattr(advanced_wrappers, 'items'):
                wrapper_states = {}
                for wrapper_name, wrapper_module in advanced_wrappers.items():
                    if wrapper_module is not None:
                        wrapper_states[wrapper_name] = {
                            k: v.cpu() for k, v in wrapper_module.state_dict().items()
                        }
                if wrapper_states:
                    modular_states['advanced_wrappers'] = wrapper_states
                    logger.debug(f"  ✓ advanced_wrappers dict 저장: {len(wrapper_states)}개 래퍼")
        
        # System: 전체 시스템 통합 파라미터
        # 전체 모델의 통합 성능을 위한 완전한 state_dict 저장
        # 이를 통해 모듈별 최적화와 전체 시스템 최적화를 모두 추적
        if hasattr(model, 'state_dict'):
            # 전체 모델 state_dict를 저장 (메모리 효율을 위해 선택적으로)
            # 핵심 통합 파라미터만 저장 (백본의 마지막 레이어 등)
            system_state = {}
            
            # 백본의 통합 레이어 (마지막 레이어들)
            if hasattr(model, 'backbone'):
                backbone = getattr(model, 'backbone')
                if hasattr(backbone, 'final_norm'):
                    final_norm = getattr(backbone, 'final_norm')
                    system_state['backbone_final_norm'] = {
                        k: v.cpu() for k, v in final_norm.state_dict().items()
                    }
                if hasattr(backbone, 'output_projection'):
                    output_proj = getattr(backbone, 'output_projection')
                    system_state['backbone_output_projection'] = {
                        k: v.cpu() for k, v in output_proj.state_dict().items()
                    }
            
            # 통합 메트릭 및 메타데이터
            system_state['meta'] = {
                'total_modules': len(modular_states),
                'timestamp': datetime.now().isoformat(),
                'module_names': list(modular_states.keys()),
                'total_parameters': sum(p.numel() for p in model.parameters()),
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
            }
            
            modular_states['system'] = system_state
        
        return modular_states
    
    def _update_metrics_history(self, epoch: int, metrics: Dict[str, Any]):
        """메트릭 히스토리 업데이트 - 별도 파일로 저장하여 누적 방지"""
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
        
        # 별도 파일로 저장 (체크포인트와 분리)
        history_file = self.checkpoint_dir / "metrics_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
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
                        
                        # Sweet spots도 별도 파일로 저장
                        sweet_spots_file = self.checkpoint_dir / "sweet_spots.json"
                        with open(sweet_spots_file, 'w') as f:
                            json.dump(self.sweet_spots, f, indent=2)
    
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
        logger.info(f"   - LR: {checkpoint_data.get('lr', 'N/A')}")
        
        # optimizer_state 존재 여부 확인
        if 'optimizer_state' in checkpoint_data:
            logger.info(f"   - Optimizer state: 포함 (학습 재개 가능)")
        else:
            logger.info(f"   - Optimizer state: 없음 (파라미터 크로스오버만 가능)")
        
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