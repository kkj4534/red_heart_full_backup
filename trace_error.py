#!/usr/bin/env python3
"""
1x7 형태 에러 추적
"""

import torch
import sys
sys.path.append('/mnt/c/large_project/linux_red_heart')

# 직접 UnifiedModel과 백본 테스트
def test_unified_model():
    print("🔍 UnifiedModel 직접 테스트")
    print("="*50)
    
    # UnifiedModel 로드
    from training.unified_training_final import UnifiedModel
    
    # config 생성
    config = {
        'input_dim': 768,
        'd_model': 896,
        'num_layers': 8,
        'num_heads': 14,
        'feedforward_dim': 3584,
        'dropout': 0.1,
        'task_dim': 896
    }
    
    model = UnifiedModel(config)
    model.eval()
    
    # 정상 입력 테스트
    print("1. 정상 입력 (1x512x768):")
    normal_input = torch.randn(1, 512, 768)
    try:
        with torch.no_grad():
            output = model(normal_input, task='emotion', return_all=True)
        print(f"   ✅ 성공: 출력 타입 = {type(output)}")
    except Exception as e:
        print(f"   ❌ 실패: {e}")
    
    # 문제 재현 (1x7x768)
    print("\n2. 1x7x768 입력:")
    wrong_input = torch.randn(1, 7, 768)
    try:
        with torch.no_grad():
            output = model(wrong_input, task='emotion', return_all=True)
        print(f"   ✅ 성공: 출력 타입 = {type(output)}")
    except Exception as e:
        print(f"   ❌ 실패: {e}")
    
    # 백본만 테스트
    print("\n3. 백본만 테스트:")
    from unified_backbone import RedHeartUnifiedBackbone
    backbone_config = {
        'input_dim': 768,
        'd_model': 896,
        'num_layers': 8,
        'num_heads': 14,
        'feedforward_dim': 3584,
        'dropout': 0.1,
        'task_dim': 896
    }
    backbone = RedHeartUnifiedBackbone(backbone_config)
    backbone.eval()
    
    # 다양한 입력 테스트
    test_inputs = [
        (torch.randn(1, 512, 768), "1x512x768"),
        (torch.randn(1, 7, 768), "1x7x768"),
        (torch.randn(1, 1, 768), "1x1x768"),
        (torch.randn(1, 768), "1x768"),
        (torch.randn(7, 768), "7x768"),
        (torch.randn(1, 7), "1x7")  # 문제의 형태
    ]
    
    for inp, desc in test_inputs:
        print(f"\n   테스트 {desc}:")
        try:
            with torch.no_grad():
                output = backbone(inp, task='emotion')
            print(f"      ✅ 성공")
        except Exception as e:
            error_msg = str(e)
            if "mat1 and mat2" in error_msg:
                print(f"      ❌ 행렬 곱셈 에러: {error_msg[:100]}")
            else:
                print(f"      ❌ 다른 에러: {error_msg[:100]}")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    test_unified_model()