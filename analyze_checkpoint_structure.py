#!/usr/bin/env python3
"""
체크포인트 구조 분석 스크립트
- 메모리 안전하게 체크포인트 구조만 확인
- 실제 데이터 로드 없이 키와 크기만 확인
"""

import torch
import sys
import os
from pathlib import Path
import json
import traceback

def analyze_checkpoint_safely(checkpoint_path):
    """체크포인트 구조를 안전하게 분석"""
    print(f"\n=== 체크포인트 분석: {checkpoint_path} ===")
    
    # 파일 크기 확인
    file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    print(f"파일 크기: {file_size_mb:.2f} MB")
    
    try:
        # 메모리 매핑으로 안전하게 로드 (실제 메모리 사용 최소화)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        print("\n체크포인트 최상위 키:")
        for key in checkpoint.keys():
            print(f"  - {key}")
            
            # 각 키의 크기 추정
            if key == 'model_state':
                if isinstance(checkpoint[key], dict):
                    print(f"    모듈 수: {len(checkpoint[key])} 개")
                    for module_name in list(checkpoint[key].keys())[:3]:  # 처음 3개만
                        print(f"      • {module_name}")
                        
            elif key == 'optimizer_state':
                if isinstance(checkpoint[key], dict):
                    if 'state' in checkpoint[key]:
                        print(f"    optimizer state 항목 수: {len(checkpoint[key]['state'])}")
                    if 'param_groups' in checkpoint[key]:
                        print(f"    param_groups 수: {len(checkpoint[key]['param_groups'])}")
                        
            elif key == 'metrics':
                if isinstance(checkpoint[key], dict):
                    print(f"    metrics 항목 수: {len(checkpoint[key])}")
                    
            elif key == 'sweet_spots':
                if isinstance(checkpoint[key], dict):
                    print(f"    sweet_spots 항목 수: {len(checkpoint[key])}")
                    # sweet_spots 크기 추정
                    import sys
                    size_bytes = sys.getsizeof(str(checkpoint[key]))
                    print(f"    sweet_spots 예상 크기: {size_bytes / 1024:.2f} KB")
        
        # 메모리 해제
        del checkpoint
        print("\n✅ 분석 완료 (메모리 해제)")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        traceback.print_exc()

def main():
    # 작은 체크포인트부터 분석
    small_checkpoint = "training/checkpoints_final/checkpoint_epoch_0001_lr_0.000012_20250823_060212.pt"
    large_checkpoint = "training/checkpoints_final/checkpoint_epoch_0023_lr_0.000011_20250824_204202.pt"
    
    if Path(small_checkpoint).exists():
        analyze_checkpoint_safely(small_checkpoint)
    else:
        print(f"파일 없음: {small_checkpoint}")
    
    # 큰 파일은 사용자 확인 후
    print("\n" + "="*50)
    response = input("큰 체크포인트(5GB)도 분석하시겠습니까? (y/n): ")
    if response.lower() == 'y':
        if Path(large_checkpoint).exists():
            analyze_checkpoint_safely(large_checkpoint)
        else:
            print(f"파일 없음: {large_checkpoint}")

if __name__ == "__main__":
    main()