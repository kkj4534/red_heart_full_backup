#!/usr/bin/env python3
"""
Red Heart AI 모듈별 체크포인트 전략
연동 그룹과 독립 모듈을 분리하여 저장/로드
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import json
from datetime import datetime

logger = logging.getLogger('RedHeart.ModularCheckpoint')


class ModularCheckpointStrategy:
    """
    모듈별 체크포인트 관리 전략
    - 연동 그룹: 함께 업데이트되는 모듈들
    - 독립 모듈: 개별적으로 업데이트되는 모듈들
    """
    
    # 연동 그룹 정의 (함께 업데이트)
    INTEGRATED_GROUPS = {
        'core_backbone_heads': [
            'backbone',
            'emotion_head', 
            'bentham_head',
            'regret_head',
            'surd_head'
        ],
        'emotion_analyzers': [
            'neural_emotion_analyzer',
            'neural_multimodal_emotion',
            'advanced_emotion',
            'phase0_net',
            'phase2_net',
            'hierarchical_integrator'
        ],
        'ethical_analyzers': [
            'neural_bentham_analyzer',
            'neural_moral_analyzer',
            'advanced_bentham'
        ],
        'decision_analyzers': [
            'neural_regret_analyzer',
            'neural_decision_analyzer',
            'advanced_regret'
        ],
        'surd_analyzers': [
            'neural_surd_analyzer',
            'neural_semantic_analyzer',
            'advanced_surd'
        ],
        'signal_processors': [
            'dsp_simulator',
            'kalman_filter'
        ]
    }
    
    # 독립 모듈 정의 (개별 업데이트)
    INDEPENDENT_MODULES = [
        'neural_social_analyzer',
        'neural_cultural_analyzer',
        'neural_context_analyzer',
        'neural_narrative_analyzer'
    ]
    
    def __init__(self, checkpoint_dir: str = './checkpoints_modular'):
        """
        Args:
            checkpoint_dir: 체크포인트 저장 디렉토리
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # 각 그룹/모듈별 디렉토리 생성
        for group_name in self.INTEGRATED_GROUPS:
            (self.checkpoint_dir / group_name).mkdir(exist_ok=True)
        
        (self.checkpoint_dir / 'independent').mkdir(exist_ok=True)
        
        logger.info(f"✅ ModularCheckpointStrategy 초기화 완료: {self.checkpoint_dir}")
    
    def save_modular_checkpoint(
        self, 
        modules: Dict[str, nn.Module],
        epoch: int,
        global_step: int,
        loss: float,
        optimizer: Optional[torch.optim.Optimizer] = None,
        metadata: Optional[Dict] = None
    ):
        """
        모듈별 체크포인트 저장
        
        Args:
            modules: 모듈 딕셔너리
            epoch: 현재 에포크
            global_step: 전역 스텝
            loss: 현재 손실
            optimizer: 옵티마이저 (선택)
            metadata: 추가 메타데이터 (선택)
        """
        timestamp = datetime.now().isoformat()
        
        # 1. 연동 그룹별 저장
        for group_name, module_names in self.INTEGRATED_GROUPS.items():
            group_modules = {}
            
            for module_name in module_names:
                # 모듈 이름 변형 처리 (head suffix 등)
                actual_name = self._find_module(modules, module_name)
                if actual_name and modules[actual_name] is not None:
                    group_modules[actual_name] = modules[actual_name].state_dict()
            
            if group_modules:
                group_checkpoint = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'loss': loss,
                    'timestamp': timestamp,
                    'modules': group_modules,
                    'metadata': metadata or {}
                }
                
                # 옵티마이저 상태 저장 (그룹별)
                if optimizer and group_name == 'core_backbone_heads':
                    group_checkpoint['optimizer'] = optimizer.state_dict()
                
                checkpoint_path = self.checkpoint_dir / group_name / f"checkpoint_epoch_{epoch+1}.pt"
                torch.save(group_checkpoint, checkpoint_path)
                logger.info(f"  💾 {group_name} 그룹 저장: {checkpoint_path}")
        
        # 2. 독립 모듈 개별 저장
        for module_name in self.INDEPENDENT_MODULES:
            actual_name = self._find_module(modules, module_name)
            if actual_name and modules[actual_name] is not None:
                module_checkpoint = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'loss': loss,
                    'timestamp': timestamp,
                    'module_state': modules[actual_name].state_dict(),
                    'metadata': metadata or {}
                }
                
                checkpoint_path = self.checkpoint_dir / 'independent' / f"{actual_name}_epoch_{epoch+1}.pt"
                torch.save(module_checkpoint, checkpoint_path)
                logger.info(f"  💾 독립 모듈 {actual_name} 저장: {checkpoint_path}")
        
        # 3. 전체 메타데이터 저장
        meta_info = {
            'epoch': epoch,
            'global_step': global_step,
            'loss': loss,
            'timestamp': timestamp,
            'groups_saved': list(self.INTEGRATED_GROUPS.keys()),
            'independent_saved': self.INDEPENDENT_MODULES,
            'metadata': metadata or {}
        }
        
        meta_path = self.checkpoint_dir / f"meta_epoch_{epoch+1}.json"
        with open(meta_path, 'w') as f:
            json.dump(meta_info, f, indent=2)
        
        logger.info(f"✅ 모듈별 체크포인트 저장 완료 (Epoch {epoch+1})")
    
    def load_modular_checkpoint(
        self,
        modules: Dict[str, nn.Module],
        checkpoint_epoch: Optional[int] = None,
        load_optimizer: bool = True
    ) -> Dict[str, Any]:
        """
        모듈별 체크포인트 로드
        
        Args:
            modules: 모듈 딕셔너리
            checkpoint_epoch: 로드할 에포크 (None이면 최신)
            load_optimizer: 옵티마이저 상태 로드 여부
            
        Returns:
            로드된 체크포인트 정보
        """
        # 최신 체크포인트 찾기
        if checkpoint_epoch is None:
            meta_files = list(self.checkpoint_dir.glob("meta_epoch_*.json"))
            if not meta_files:
                logger.warning("체크포인트를 찾을 수 없습니다")
                return {}
            
            latest_meta = max(meta_files, key=lambda p: int(p.stem.split('_')[-1]))
            checkpoint_epoch = int(latest_meta.stem.split('_')[-1]) - 1
        
        logger.info(f"📥 Epoch {checkpoint_epoch+1} 체크포인트 로드 중...")
        
        loaded_info = {
            'epoch': checkpoint_epoch,
            'modules_loaded': [],
            'modules_failed': []
        }
        
        # 1. 연동 그룹 로드
        for group_name, module_names in self.INTEGRATED_GROUPS.items():
            checkpoint_path = self.checkpoint_dir / group_name / f"checkpoint_epoch_{checkpoint_epoch+1}.pt"
            
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                
                for module_name, state_dict in checkpoint['modules'].items():
                    actual_name = self._find_module(modules, module_name)
                    if actual_name and modules[actual_name] is not None:
                        try:
                            modules[actual_name].load_state_dict(state_dict)
                            loaded_info['modules_loaded'].append(actual_name)
                            logger.info(f"  ✅ {actual_name} 로드 완료")
                        except Exception as e:
                            loaded_info['modules_failed'].append(actual_name)
                            logger.error(f"  ❌ {actual_name} 로드 실패: {e}")
                
                # 옵티마이저 상태 로드
                if load_optimizer and 'optimizer' in checkpoint:
                    loaded_info['optimizer_state'] = checkpoint['optimizer']
                
                loaded_info['loss'] = checkpoint.get('loss', 0.0)
                loaded_info['global_step'] = checkpoint.get('global_step', 0)
        
        # 2. 독립 모듈 로드
        for module_name in self.INDEPENDENT_MODULES:
            actual_name = self._find_module(modules, module_name)
            if actual_name:
                checkpoint_path = self.checkpoint_dir / 'independent' / f"{actual_name}_epoch_{checkpoint_epoch+1}.pt"
                
                if checkpoint_path.exists() and modules[actual_name] is not None:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    try:
                        modules[actual_name].load_state_dict(checkpoint['module_state'])
                        loaded_info['modules_loaded'].append(actual_name)
                        logger.info(f"  ✅ 독립 모듈 {actual_name} 로드 완료")
                    except Exception as e:
                        loaded_info['modules_failed'].append(actual_name)
                        logger.error(f"  ❌ 독립 모듈 {actual_name} 로드 실패: {e}")
        
        logger.info(f"✅ 체크포인트 로드 완료")
        logger.info(f"  - 로드 성공: {len(loaded_info['modules_loaded'])} 모듈")
        logger.info(f"  - 로드 실패: {len(loaded_info['modules_failed'])} 모듈")
        
        return loaded_info
    
    def _find_module(self, modules: Dict[str, nn.Module], module_name: str) -> Optional[str]:
        """
        모듈 이름 찾기 (유연한 매칭)
        
        Args:
            modules: 모듈 딕셔너리
            module_name: 찾을 모듈 이름
            
        Returns:
            실제 모듈 이름 또는 None
        """
        # 정확한 매칭
        if module_name in modules:
            return module_name
        
        # 부분 매칭 (예: 'emotion_head' -> 'emotion')
        for actual_name in modules:
            if module_name.replace('_head', '') == actual_name:
                return actual_name
            if module_name.replace('neural_', '') == actual_name.replace('neural_', ''):
                return actual_name
            if module_name in actual_name or actual_name in module_name:
                return actual_name
        
        return None
    
    def get_optimizer_groups(self, modules: Dict[str, nn.Module]) -> List[Dict]:
        """
        연동/개별 모듈별 옵티마이저 파라미터 그룹 생성
        
        Args:
            modules: 모듈 딕셔너리
            
        Returns:
            옵티마이저 파라미터 그룹 리스트
        """
        param_groups = []
        
        # 1. 연동 그룹별 파라미터 그룹 생성
        for group_name, module_names in self.INTEGRATED_GROUPS.items():
            group_params = []
            
            for module_name in module_names:
                actual_name = self._find_module(modules, module_name)
                if actual_name and modules[actual_name] is not None:
                    group_params.extend(list(modules[actual_name].parameters()))
            
            if group_params:
                # 그룹별 학습률 조정 가능
                lr_scale = 1.0
                if 'analyzers' in group_name:
                    lr_scale = 0.5  # 분석기는 낮은 학습률
                elif 'signal' in group_name:
                    lr_scale = 0.8  # 신호 처리기는 중간 학습률
                
                param_groups.append({
                    'params': group_params,
                    'lr_scale': lr_scale,
                    'name': group_name
                })
                
                param_count = sum(p.numel() for p in group_params)
                logger.info(f"  파라미터 그룹 '{group_name}': {param_count:,} 파라미터 (lr_scale={lr_scale})")
        
        # 2. 독립 모듈 개별 파라미터 그룹
        for module_name in self.INDEPENDENT_MODULES:
            actual_name = self._find_module(modules, module_name)
            if actual_name and modules[actual_name] is not None:
                module_params = list(modules[actual_name].parameters())
                if module_params:
                    param_groups.append({
                        'params': module_params,
                        'lr_scale': 0.3,  # 독립 모듈은 더 낮은 학습률
                        'name': actual_name
                    })
                    
                    param_count = sum(p.numel() for p in module_params)
                    logger.info(f"  독립 모듈 '{actual_name}': {param_count:,} 파라미터 (lr_scale=0.3)")
        
        return param_groups