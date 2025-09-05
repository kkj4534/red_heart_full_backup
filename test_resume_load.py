#!/usr/bin/env python3
"""
재개 로드 테스트 - 모듈별 체크포인트 플랫 변환 확인
"""

import sys
import torch
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent))

def test_modular_to_flat():
    """모듈별 state를 플랫 구조로 변환 테스트"""
    
    checkpoint_path = "training/checkpoints_final/checkpoint_epoch_0023_lr_0.000011_20250824_204202.pt"
    
    print("=" * 60)
    print("체크포인트 구조 변환 테스트")
    print("=" * 60)
    
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_state = checkpoint['model_state']
    
    print(f"✅ 체크포인트 로드 완료")
    print(f"   - 에폭: {checkpoint['epoch']}")
    print(f"   - LR: {checkpoint.get('lr', 'N/A')}")
    
    # 구조 확인
    if isinstance(model_state, dict) and 'backbone' in model_state:
        print(f"\n📦 모듈별 구조 감지:")
        for module_name in model_state.keys():
            param_count = len(model_state[module_name])
            print(f"   - {module_name}: {param_count}개 파라미터")
        
        # 플랫 구조로 변환
        flat_state = {}
        for module_name, module_state in model_state.items():
            for param_name, param_value in module_state.items():
                flat_state[f"{module_name}.{param_name}"] = param_value
        
        print(f"\n✅ 플랫 구조로 변환 완료:")
        print(f"   - 총 파라미터: {len(flat_state)}개")
        
        # 샘플 키 출력
        print(f"\n📋 변환된 키 샘플 (처음 5개):")
        for i, key in enumerate(list(flat_state.keys())[:5]):
            print(f"   - {key}")
        
        # optimizer_state 확인
        if 'optimizer_state' in checkpoint:
            print(f"\n✅ Optimizer State: 존재 (재개 가능)")
        else:
            print(f"\n❌ Optimizer State: 없음 (재개 불가)")
            
    else:
        print(f"\n⚠️ 이미 플랫한 구조")
    
    print("\n테스트 완료!")

if __name__ == "__main__":
    test_modular_to_flat()