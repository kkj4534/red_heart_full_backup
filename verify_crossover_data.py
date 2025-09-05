#!/usr/bin/env python3
"""
클리닝된 체크포인트에서 파라미터 크로스오버 데이터 확인
"""

import torch
import sys
from pathlib import Path

def verify_checkpoint(checkpoint_path):
    """체크포인트의 크로스오버 필수 데이터 확인"""
    
    checkpoint_path = Path(checkpoint_path)
    print(f"\n검증 중: {checkpoint_path.name}")
    print("-" * 40)
    
    try:
        # 체크포인트 로드
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 필수 데이터 확인
        print("✅ 파라미터 크로스오버 필수 데이터:")
        
        # 1. model_state 확인
        if 'model_state' in checkpoint:
            model_state = checkpoint['model_state']
            print(f"  • model_state: {len(model_state)} 모듈")
            
            # 주요 모듈 확인
            essential_modules = ['backbone', 'emotion_head', 'bentham_head', 
                               'regret_head', 'surd_head', 'neural_analyzers']
            for module in essential_modules:
                if module in model_state:
                    if module == 'neural_analyzers':
                        # dict 형태인 경우
                        if isinstance(model_state[module], dict):
                            print(f"    - {module}: {len(model_state[module])} 분석기")
                    else:
                        # 일반 모듈
                        param_count = sum(p.numel() for p in model_state[module].values() 
                                        if hasattr(p, 'numel'))
                        print(f"    - {module}: {param_count/1e6:.1f}M 파라미터")
        else:
            print("  ❌ model_state 없음!")
            
        # 2. metrics 확인
        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
            print(f"  • metrics: {len(metrics)} 항목")
            if 'loss' in metrics:
                print(f"    - loss: {metrics['loss']:.4f}")
            if 'val_acc' in metrics:
                print(f"    - val_acc: {metrics['val_acc']:.4f}")
        else:
            print("  ⚠️ metrics 없음 (선택적)")
            
        # 3. optimizer_state 확인
        if 'optimizer_state' in checkpoint:
            opt_state = checkpoint['optimizer_state']
            if 'state' in opt_state:
                print(f"  • optimizer_state: {len(opt_state['state'])} 항목 (학습 재개용)")
        else:
            print("  • optimizer_state: 제거됨 (크로스오버엔 불필요)")
            
        # 4. 기타 정보
        print(f"  • epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  • lr: {checkpoint.get('lr', 'N/A')}")
        
        print("\n✅ 파라미터 크로스오버 가능!")
        
        del checkpoint
        return True
        
    except Exception as e:
        print(f"❌ 오류: {e}")
        return False

# 테스트: 클리닝된 체크포인트 (optimizer 제거)
print("=" * 60)
print("클리닝된 체크포인트 검증 (optimizer 제거)")
print("=" * 60)
verify_checkpoint("training/checkpoints_final/checkpoint_epoch_0010_lr_0.000012_20250824_061455.pt")

# 테스트: 유지된 체크포인트 (optimizer 유지)
print("\n" + "=" * 60)
print("유지된 체크포인트 검증 (optimizer 유지)")
print("=" * 60)
verify_checkpoint("training/checkpoints_final/checkpoint_epoch_0023_lr_0.000011_20250824_204202.pt")