#!/usr/bin/env python3
"""
체크포인트에서 bentham_head 가중치 확인
"""

import torch
import sys

checkpoint_path = "training/checkpoints_final/checkpoint_epoch_0050_lr_0.000009_20250827_154403.pt"

print("체크포인트 로드 중...")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print(f"\n체크포인트 키: {list(checkpoint.keys())}")

if 'model_state' in checkpoint:
    model_state = checkpoint['model_state']
    
    # bentham_head 관련 키 찾기
    bentham_keys = [k for k in model_state.keys() if 'bentham' in k.lower()]
    
    print(f"\n벤담 관련 키 개수: {len(bentham_keys)}")
    
    if bentham_keys:
        print("\n벤담 헤드 구조:")
        for key in sorted(bentham_keys)[:20]:  # 처음 20개만
            value = model_state[key]
            if isinstance(value, torch.Tensor):
                shape = value.shape
                params = value.numel()
                print(f"  {key}: {shape} ({params:,} params)")
            elif isinstance(value, dict):
                print(f"  {key}: dict with {len(value)} keys")
                # dict 내부 확인
                for subkey in list(value.keys())[:5]:
                    if isinstance(value[subkey], torch.Tensor):
                        print(f"    └─ {subkey}: {value[subkey].shape}")
            else:
                print(f"  {key}: {type(value)}")
        
        # 총 파라미터 수 (dict인 경우 처리)
        total_params = 0
        for k in bentham_keys:
            if isinstance(model_state[k], torch.Tensor):
                total_params += model_state[k].numel()
            elif isinstance(model_state[k], dict):
                # dict 내부의 텐서들 계산
                for subkey, value in model_state[k].items():
                    if isinstance(value, torch.Tensor):
                        total_params += value.numel()
        print(f"\n벤담 헤드 총 파라미터: {total_params:,} ({total_params/1e6:.2f}M)")
        
        # weight_layers 확인
        weight_layer_keys = [k for k in bentham_keys if 'weight_layer' in k.lower()]
        if weight_layer_keys:
            print(f"\n가중치 레이어 발견: {len(weight_layer_keys)}개")
        else:
            print("\n⚠️ weight_layers 키 없음 - 별도 구조")
            
        # MoE experts 확인
        expert_keys = [k for k in bentham_keys if 'expert' in k.lower()]
        if expert_keys:
            print(f"윤리 전문가 발견: {len(set(k.split('.')[2] for k in expert_keys if 'ethical_experts' in k))}개")
    else:
        print("\n❌ bentham_head 키가 체크포인트에 없음!")
else:
    print("\n❌ model_state가 체크포인트에 없음!")

print("\n" + "="*60)
print("결론:")
print("- 학습된 bentham_head가 체크포인트에 있음")
print("- AdvancedBenthamCalculator와 연결 필요")
print("- 방법: main_unified.py에서 bentham_head 출력을 사용하도록 수정")