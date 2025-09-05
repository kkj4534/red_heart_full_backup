#!/usr/bin/env python3
"""
체크포인트 클리닝 스크립트 (안전 버전)
누적 데이터 제거하여 크기 축소
파라미터 크로스오버를 위한 필수 데이터는 모두 보존
"""

import torch
import sys
import os
from pathlib import Path
import json
import shutil
import gc
import traceback

def clean_checkpoint_safely(input_path, output_path=None, test_mode=False):
    """
    체크포인트에서 누적 데이터 제거
    
    Args:
        input_path: 원본 체크포인트 경로
        output_path: 출력 경로 (None이면 _clean 추가)
        test_mode: True면 저장하지 않고 크기만 확인
    """
    
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"❌ 파일 없음: {input_path}")
        return False
    
    # 파일 크기 확인
    original_size_mb = input_path.stat().st_size / (1024 * 1024)
    print(f"\n{'='*60}")
    print(f"처리 중: {input_path.name}")
    print(f"원본 크기: {original_size_mb:.1f}MB")
    
    try:
        # 체크포인트 로드
        print("체크포인트 로드 중...")
        checkpoint = torch.load(input_path, map_location='cpu')
        
        # 현재 구조 분석
        print("\n현재 체크포인트 구조:")
        for key in checkpoint.keys():
            if isinstance(checkpoint[key], dict):
                if key == 'model_state':
                    # 모델 파라미터 크기 계산
                    total_params = 0
                    for module_name, module_state in checkpoint[key].items():
                        module_params = sum(p.numel() for p in module_state.values() if hasattr(p, 'numel'))
                        total_params += module_params
                    print(f"  - {key}: {len(checkpoint[key])} 모듈, ~{total_params/1e6:.1f}M 파라미터")
                elif key == 'optimizer_state':
                    if 'state' in checkpoint[key]:
                        print(f"  - {key}: {len(checkpoint[key]['state'])} 상태 항목")
                else:
                    print(f"  - {key}: {len(checkpoint[key])} 항목")
            else:
                print(f"  - {key}: {type(checkpoint[key]).__name__}")
        
        # 필수 데이터만 추출 (파라미터 크로스오버를 위해 필요한 모든 것)
        clean_checkpoint = {
            'epoch': checkpoint.get('epoch'),
            'lr': checkpoint.get('lr'),
            'timestamp': checkpoint.get('timestamp'),
            'model_state': checkpoint.get('model_state'),  # 파라미터 크로스오버에 필수
            'optimizer_state': checkpoint.get('optimizer_state'),  # 학습 재개에 필수
            'scheduler_state': checkpoint.get('scheduler_state'),  # 학습률 스케줄 유지
            'metrics': checkpoint.get('metrics'),  # 해당 에폭의 성능 메트릭 (분석용)
        }
        
        # sweet_spots는 제거 (누적 데이터)
        # metrics_history도 있다면 제거
        
        # 추가로 보존할 데이터 (논문 분석용)
        if 'metrics' in checkpoint:
            # 해당 에폭의 메트릭만 보존 (누적 아님)
            clean_checkpoint['epoch_metrics'] = {
                'loss': checkpoint['metrics'].get('loss'),
                'train_loss': checkpoint['metrics'].get('train_loss'),
                'val_loss': checkpoint['metrics'].get('val_loss'),
                'val_acc': checkpoint['metrics'].get('val_acc'),
                'module_losses': {
                    'emotion': checkpoint['metrics'].get('emotion_loss'),
                    'bentham': checkpoint['metrics'].get('bentham_loss'),
                    'regret': checkpoint['metrics'].get('regret_loss'),
                    'surd': checkpoint['metrics'].get('surd_loss'),
                },
                'module_accs': {
                    'emotion': checkpoint['metrics'].get('emotion_acc'),
                    'bentham': checkpoint['metrics'].get('bentham_acc'),
                    'regret': checkpoint['metrics'].get('regret_acc'),
                    'surd': checkpoint['metrics'].get('surd_acc'),
                }
            }
        
        # 메모리 예측
        import pickle
        estimated_size = len(pickle.dumps(clean_checkpoint, protocol=pickle.HIGHEST_PROTOCOL))
        estimated_size_mb = estimated_size / (1024 * 1024)
        
        print(f"\n예상 클린 크기: {estimated_size_mb:.1f}MB")
        print(f"절약 공간: {original_size_mb - estimated_size_mb:.1f}MB ({(1 - estimated_size_mb/original_size_mb)*100:.1f}%)")
        
        if test_mode:
            print("\n✅ 테스트 모드 - 실제 저장하지 않음")
            del checkpoint
            del clean_checkpoint
            gc.collect()
            return True
        
        # 실제 저장
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_clean.pt"
        else:
            output_path = Path(output_path)
        
        print(f"\n저장 중: {output_path}")
        torch.save(clean_checkpoint, output_path)
        
        # 저장된 파일 크기 확인
        saved_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"저장 완료: {saved_size_mb:.1f}MB")
        
        # 메모리 정리
        del checkpoint
        del clean_checkpoint
        gc.collect()
        
        print("✅ 클리닝 완료!")
        return True
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        traceback.print_exc()
        return False

def main():
    print("=" * 80)
    print("체크포인트 클리닝 스크립트 (안전 버전)")
    print("=" * 80)
    
    # 테스트: 작은 체크포인트부터
    test_checkpoint = "training/checkpoints_final/checkpoint_epoch_0001_lr_0.000012_20250823_060212.pt"
    
    print(f"\n1단계: 테스트 모드로 확인")
    if clean_checkpoint_safely(test_checkpoint, test_mode=True):
        response = input("\n실제로 클리닝을 진행하시겠습니까? (y/n): ")
        if response.lower() == 'y':
            print("\n2단계: 실제 클리닝")
            clean_checkpoint_safely(
                test_checkpoint,
                output_path="training/checkpoints_final/test_clean_epoch01.pt"
            )
            
            # 검증
            print("\n3단계: 클리닝된 체크포인트 검증")
            cleaned = "training/checkpoints_final/test_clean_epoch01.pt"
            if Path(cleaned).exists():
                ckpt = torch.load(cleaned, map_location='cpu')
                print(f"✅ 로드 성공")
                print(f"  - 에폭: {ckpt.get('epoch')}")
                print(f"  - 모델 모듈 수: {len(ckpt.get('model_state', {}))}")
                del ckpt
                gc.collect()

if __name__ == "__main__":
    main()