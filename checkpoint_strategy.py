"""
모듈별 체크포인트 전략 구현
연동 그룹은 함께, 독립 모듈은 개별 저장
"""

import torch
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ModularCheckpointStrategy:
    """
    연동 그룹은 함께, 독립 모듈은 개별 저장하는 체크포인트 전략
    
    연동 그룹:
    1. Backbone-Heads: 백본과 헤드들이 긴밀하게 연결
    2. Phase0-1: 감정 투영과 공감 학습이 순차적 의존
    3. DSP-Kalman: DSP 출력을 칼만 필터가 필수 입력으로 사용
    
    독립 모듈:
    - Neural Analyzers: 각각 독립적으로 작동
    - Advanced Analyzers: 도메인별 특화 처리
    - Phase2: 공동체 수준 독립 처리
    """
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # 연동 그룹 정의
        self.coupled_groups = {
            'backbone_heads': {
                'modules': ['backbone', 'emotion_head', 'bentham_head', 'regret_head', 'surd_head'],
                'description': 'Backbone과 모든 헤드 (데이터 흐름 공유)',
                'params': '278M'
            },
            'phase_0_1': {
                'modules': ['phase0_calibrator', 'phase1_empathy'],
                'description': 'Phase0 투영 → Phase1 공감 (순차 의존)',
                'params': '2.23M'
            },
            'dsp_kalman': {
                'modules': ['emotion_dsp', 'kalman_filter'],
                'description': 'DSP 시뮬레이터 → 칼만 필터 (필수 입력)',
                'params': '14M'
            }
        }
        
        # 독립 모듈 정의
        self.independent_modules = {
            # Neural Analyzers
            'neural_emotion': {'description': '신경망 감정 분석기', 'params': '55M'},
            'neural_bentham': {'description': '신경망 벤담 계산기', 'params': '62M'},
            'neural_regret': {'description': '신경망 후회 분석기', 'params': '68M'},
            'neural_surd': {'description': '신경망 SURD 분석기', 'params': '47M'},
            
            # Advanced Analyzers
            'advanced_emotion': {'description': '고급 감정 분석기', 'params': '48M'},
            'advanced_regret': {'description': '고급 후회 분석기', 'params': '50M'},
            'advanced_surd': {'description': '고급 SURD 분석기', 'params': '25M'},
            'advanced_bentham': {'description': '고급 벤담 계산기', 'params': '2.5M'},
            
            # Phase2
            'phase2_community': {'description': '공동체 확장 모듈', 'params': '2.5M'}
        }
        
        # 최적 성능 추적
        self.best_metrics = {}
        self.load_best_metrics()
        
        logger.info("✅ 모듈별 체크포인트 전략 초기화")
        logger.info(f"  - 연동 그룹: {len(self.coupled_groups)}개")
        logger.info(f"  - 독립 모듈: {len(self.independent_modules)}개")
    
    def load_best_metrics(self):
        """저장된 최적 메트릭 로드"""
        metrics_file = self.checkpoint_dir / "best_metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                self.best_metrics = json.load(f)
    
    def save_best_metrics(self):
        """최적 메트릭 저장"""
        metrics_file = self.checkpoint_dir / "best_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.best_metrics, f, indent=2)
    
    def is_best_performance(self, module_name: str, current_metric: float) -> bool:
        """현재 성능이 최고인지 확인"""
        if module_name not in self.best_metrics:
            return True
        return current_metric < self.best_metrics[module_name]
    
    def save_checkpoint(self, 
                       epoch: int,
                       model: Any,
                       metrics: Dict[str, float],
                       optimizer: Optional[Any] = None,
                       scheduler: Optional[Any] = None):
        """
        체크포인트 저장
        
        Args:
            epoch: 현재 에폭
            model: 전체 모델 객체
            metrics: 각 모듈별 메트릭
            optimizer: 옵티마이저 (선택)
            scheduler: 스케줄러 (선택)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 연동 그룹 저장 (같은 에폭)
        for group_name, group_info in self.coupled_groups.items():
            group_dir = self.checkpoint_dir / group_name
            group_dir.mkdir(exist_ok=True)
            
            group_checkpoint = {
                'epoch': epoch,
                'timestamp': timestamp,
                'modules': {},
                'metrics': {}
            }
            
            # 각 모듈의 state_dict 수집
            for module_name in group_info['modules']:
                module_state = self._get_module_state(model, module_name)
                if module_state is not None:
                    group_checkpoint['modules'][module_name] = module_state
                    if module_name in metrics:
                        group_checkpoint['metrics'][module_name] = metrics[module_name]
            
            # 그룹 체크포인트 저장
            checkpoint_path = group_dir / f"epoch_{epoch:04d}.pt"
            torch.save(group_checkpoint, checkpoint_path)
            logger.info(f"  💾 {group_name} 그룹 저장: {checkpoint_path}")
        
        # 2. 독립 모듈 저장 (최적 성능 시점만)
        for module_name, module_info in self.independent_modules.items():
            if module_name in metrics:
                current_metric = metrics[module_name]
                
                if self.is_best_performance(module_name, current_metric):
                    module_dir = self.checkpoint_dir / module_name
                    module_dir.mkdir(exist_ok=True)
                    
                    module_state = self._get_module_state(model, module_name)
                    if module_state is not None:
                        checkpoint = {
                            'epoch': epoch,
                            'timestamp': timestamp,
                            'state_dict': module_state,
                            'metric': current_metric,
                            'description': module_info['description'],
                            'params': module_info['params']
                        }
                        
                        checkpoint_path = module_dir / "best.pt"
                        torch.save(checkpoint, checkpoint_path)
                        
                        # 최적 메트릭 업데이트
                        self.best_metrics[module_name] = current_metric
                        
                        logger.info(f"  🏆 {module_name} 최적 모델 저장 (metric: {current_metric:.4f})")
        
        # 3. 옵티마이저/스케줄러 저장 (선택)
        if optimizer is not None:
            optimizer_path = self.checkpoint_dir / f"optimizer_epoch_{epoch:04d}.pt"
            torch.save(optimizer.state_dict(), optimizer_path)
        
        if scheduler is not None:
            scheduler_path = self.checkpoint_dir / f"scheduler_epoch_{epoch:04d}.pt"
            torch.save(scheduler.state_dict(), scheduler_path)
        
        # 최적 메트릭 저장
        self.save_best_metrics()
    
    def _get_module_state(self, model: Any, module_name: str) -> Optional[Dict]:
        """모듈의 state_dict 추출"""
        # 백본
        if module_name == 'backbone' and hasattr(model, 'backbone'):
            return model.backbone.state_dict() if model.backbone else None
        
        # 헤드
        if '_head' in module_name:
            head_key = module_name.replace('_head', '')
            if hasattr(model, 'heads') and head_key in model.heads:
                return model.heads[head_key].state_dict()
        
        # 분석기
        if 'neural_' in module_name or 'advanced_' in module_name:
            if hasattr(model, 'analyzers') and module_name in model.analyzers:
                analyzer = model.analyzers[module_name]
                
                # Advanced Analyzer는 내부 모듈들 수집
                if 'advanced_' in module_name:
                    state_dict = {}
                    for attr_name in dir(analyzer):
                        if not attr_name.startswith('_'):
                            attr = getattr(analyzer, attr_name, None)
                            if attr is not None and isinstance(attr, torch.nn.Module):
                                state_dict[attr_name] = attr.state_dict()
                    return state_dict if state_dict else None
                else:
                    # Neural Analyzer는 직접 state_dict
                    return analyzer.state_dict() if hasattr(analyzer, 'state_dict') else None
        
        # Phase 모듈
        if 'phase' in module_name:
            # Phase 네트워크 접근 (구현 필요)
            pass
        
        # DSP/Kalman
        if module_name == 'emotion_dsp':
            if hasattr(model, 'dsp_simulator'):
                return model.dsp_simulator.state_dict()
        elif module_name == 'kalman_filter':
            if hasattr(model, 'kalman_filter'):
                return model.kalman_filter.state_dict()
        
        return None
    
    def load_checkpoint(self, 
                       model: Any,
                       checkpoint_type: str = 'best',
                       epoch: Optional[int] = None,
                       modules_to_load: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        체크포인트 로드
        
        Args:
            model: 모델 객체
            checkpoint_type: 'best' 또는 'epoch'
            epoch: 특정 에폭 (checkpoint_type='epoch'일 때)
            modules_to_load: 로드할 모듈 리스트 (None이면 전체)
            
        Returns:
            로드된 정보 딕셔너리
        """
        loaded_info = {}
        
        if checkpoint_type == 'best':
            # 최적 체크포인트 로드
            for module_name in modules_to_load or list(self.independent_modules.keys()):
                module_path = self.checkpoint_dir / module_name / "best.pt"
                if module_path.exists():
                    checkpoint = torch.load(module_path)
                    self._load_module_state(model, module_name, checkpoint['state_dict'])
                    loaded_info[module_name] = {
                        'epoch': checkpoint['epoch'],
                        'metric': checkpoint['metric']
                    }
                    logger.info(f"  ✅ {module_name} 최적 모델 로드 (epoch {checkpoint['epoch']})")
        
        elif checkpoint_type == 'epoch' and epoch is not None:
            # 특정 에폭 체크포인트 로드
            for group_name, group_info in self.coupled_groups.items():
                group_path = self.checkpoint_dir / group_name / f"epoch_{epoch:04d}.pt"
                if group_path.exists():
                    checkpoint = torch.load(group_path)
                    for module_name, state_dict in checkpoint['modules'].items():
                        if modules_to_load is None or module_name in modules_to_load:
                            self._load_module_state(model, module_name, state_dict)
                            loaded_info[module_name] = {'epoch': epoch}
                    logger.info(f"  ✅ {group_name} 그룹 로드 (epoch {epoch})")
        
        return loaded_info
    
    def _load_module_state(self, model: Any, module_name: str, state_dict: Dict):
        """모듈에 state_dict 로드"""
        # 백본
        if module_name == 'backbone' and hasattr(model, 'backbone'):
            if model.backbone:
                model.backbone.load_state_dict(state_dict)
        
        # 헤드
        elif '_head' in module_name:
            head_key = module_name.replace('_head', '')
            if hasattr(model, 'heads') and head_key in model.heads:
                model.heads[head_key].load_state_dict(state_dict)
        
        # 분석기
        elif 'neural_' in module_name or 'advanced_' in module_name:
            if hasattr(model, 'analyzers') and module_name in model.analyzers:
                analyzer = model.analyzers[module_name]
                
                # Advanced Analyzer는 내부 모듈별로 로드
                if 'advanced_' in module_name and isinstance(state_dict, dict):
                    for attr_name, attr_state in state_dict.items():
                        if hasattr(analyzer, attr_name):
                            attr = getattr(analyzer, attr_name)
                            if isinstance(attr, torch.nn.Module):
                                attr.load_state_dict(attr_state)
                else:
                    # Neural Analyzer는 직접 로드
                    if hasattr(analyzer, 'load_state_dict'):
                        analyzer.load_state_dict(state_dict)
    
    def find_optimal_combination(self) -> Dict[str, int]:
        """
        최적 에폭 조합 찾기
        
        Returns:
            각 모듈별 최적 에폭
        """
        optimal = {}
        
        # 연동 그룹은 최신 에폭 사용
        for group_name in self.coupled_groups:
            group_dir = self.checkpoint_dir / group_name
            if group_dir.exists():
                checkpoints = sorted(group_dir.glob("epoch_*.pt"))
                if checkpoints:
                    latest = checkpoints[-1]
                    epoch = int(latest.stem.split('_')[1])
                    for module in self.coupled_groups[group_name]['modules']:
                        optimal[module] = epoch
        
        # 독립 모듈은 최적 성능 에폭 사용
        for module_name in self.independent_modules:
            best_path = self.checkpoint_dir / module_name / "best.pt"
            if best_path.exists():
                checkpoint = torch.load(best_path)
                optimal[module_name] = checkpoint['epoch']
        
        return optimal
    
    def get_summary(self) -> str:
        """체크포인트 전략 요약"""
        summary = []
        summary.append("=" * 60)
        summary.append("📊 모듈별 체크포인트 전략 요약")
        summary.append("=" * 60)
        
        summary.append("\n🔗 연동 그룹 (함께 저장/로드):")
        for group_name, group_info in self.coupled_groups.items():
            summary.append(f"  • {group_name}: {group_info['description']}")
            summary.append(f"    - 모듈: {', '.join(group_info['modules'])}")
            summary.append(f"    - 파라미터: {group_info['params']}")
        
        summary.append("\n🎯 독립 모듈 (개별 최적화):")
        for module_name, module_info in self.independent_modules.items():
            best_metric = self.best_metrics.get(module_name, 'N/A')
            summary.append(f"  • {module_name}: {module_info['description']}")
            summary.append(f"    - 파라미터: {module_info['params']}")
            summary.append(f"    - 최적 메트릭: {best_metric}")
        
        summary.append("\n" + "=" * 60)
        return "\n".join(summary)