#!/usr/bin/env python3
"""optimizer_state 상세 분석"""

import torch
import sys
from pathlib import Path

def analyze_optimizer_state(checkpoint_path):
    """optimizer state 상세 분석"""
    
    print(f"분석 중: {checkpoint_path}")
    
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'optimizer_state' in checkpoint:
        opt_state = checkpoint['optimizer_state']
        
        total_size = 0
        
        if 'state' in opt_state:
            print(f"\nOptimizer State 분석:")
            print(f"총 상태 항목 수: {len(opt_state['state'])}")
            
            # 각 상태 크기 계산
            for idx, (key, state) in enumerate(list(opt_state['state'].items())[:5]):  # 처음 5개만
                item_size = 0
                print(f"\n상태 항목 {idx} (key={key}):")
                for k, v in state.items():
                    if torch.is_tensor(v):
                        size_mb = v.numel() * v.element_size() / (1024 * 1024)
                        item_size += size_mb
                        print(f"  - {k}: shape={v.shape}, dtype={v.dtype}, size={size_mb:.2f}MB")
                    else:
                        print(f"  - {k}: {type(v).__name__}")
                total_size += item_size
            
            # 전체 추정
            if len(opt_state['state']) > 0:
                avg_size = total_size / min(5, len(opt_state['state']))
                estimated_total = avg_size * len(opt_state['state'])
                print(f"\n예상 총 optimizer state 크기: {estimated_total:.1f}MB")
        
        if 'param_groups' in opt_state:
            print(f"\nParam groups: {len(opt_state['param_groups'])}개")
    
    del checkpoint

# 테스트
checkpoint = "training/checkpoints_final/checkpoint_epoch_0005_lr_0.000012_20250824_004444.pt"
analyze_optimizer_state(checkpoint)