#!/usr/bin/env python3
"""
체크포인트 일괄 클리닝 스크립트
전략: 1-20 에폭은 optimizer 제거, 21-23은 유지
"""

import torch
import os
import sys
import json
import gc
import shutil
from pathlib import Path
from datetime import datetime

def should_keep_optimizer(epoch):
    """optimizer 유지 여부 결정"""
    # 21, 22, 23: 재개 보험용
    if epoch in [21, 22, 23]:
        return True
    # 향후: 30, 40, 50, 60에서만 유지
    if epoch >= 30 and epoch % 10 == 0:
        return True
    return False

def clean_single_checkpoint(input_path, epoch, in_place=True):
    """단일 체크포인트 클리닝"""
    
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"  ⚠️ 파일 없음: {input_path.name}")
        return False
    
    original_size_mb = input_path.stat().st_size / (1024 * 1024)
    
    # optimizer 유지 여부 결정
    keep_optimizer = should_keep_optimizer(epoch)
    
    if keep_optimizer:
        print(f"  ✅ 에폭 {epoch}: optimizer 유지 ({original_size_mb:.1f}MB)")
        return True  # 수정 없이 유지
    
    print(f"  🔧 에폭 {epoch}: optimizer 제거 중... ({original_size_mb:.1f}MB)", end='')
    
    try:
        # 체크포인트 로드
        checkpoint = torch.load(input_path, map_location='cpu')
        
        # 클린 체크포인트 생성 (optimizer 제거)
        clean_checkpoint = {
            'epoch': checkpoint.get('epoch', epoch),
            'lr': checkpoint.get('lr'),
            'timestamp': checkpoint.get('timestamp'),
            'model_state': checkpoint.get('model_state'),
            'scheduler_state': checkpoint.get('scheduler_state'),
            'metrics': checkpoint.get('metrics'),
            # optimizer_state 제거!
        }
        
        # 임시 파일로 저장
        temp_path = input_path.parent / f"{input_path.stem}_temp.pt"
        torch.save(clean_checkpoint, temp_path)
        
        # 새 크기 확인
        new_size_mb = temp_path.stat().st_size / (1024 * 1024)
        
        if in_place:
            # 원본 파일 교체
            shutil.move(str(temp_path), str(input_path))
            print(f" → {new_size_mb:.1f}MB (절약: {original_size_mb - new_size_mb:.1f}MB)")
        else:
            # 별도 파일로 저장
            output_path = input_path.parent / f"{input_path.stem}_clean.pt"
            shutil.move(str(temp_path), str(output_path))
            print(f" → {output_path.name}: {new_size_mb:.1f}MB")
        
        # 메모리 정리
        del checkpoint
        del clean_checkpoint
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"\n  ❌ 오류: {e}")
        # 임시 파일 정리
        temp_path = input_path.parent / f"{input_path.stem}_temp.pt"
        if temp_path.exists():
            temp_path.unlink()
        return False

def main():
    print("=" * 80)
    print("체크포인트 일괄 클리닝 시작")
    print("전략: 1-20 optimizer 제거, 21-23 유지")
    print("=" * 80)
    
    # best_checkpoints_selection.json 로드
    if not Path("best_checkpoints_selection.json").exists():
        print("❌ best_checkpoints_selection.json 파일 없음")
        print("먼저 python3 identify_best_checkpoints.py 실행 필요")
        return
    
    with open("best_checkpoints_selection.json", "r") as f:
        best_checkpoints = json.load(f)
    
    # 처리 통계
    total_original = 0
    total_cleaned = 0
    success_count = 0
    
    print(f"\n총 {len(best_checkpoints)}개 체크포인트 처리 예정\n")
    
    # 안전 확인
    response = input("⚠️ 원본 파일을 직접 수정합니다. 계속하시겠습니까? (yes/no): ")
    if response.lower() != 'yes':
        print("취소됨")
        return
    
    print("\n처리 시작...")
    print("-" * 40)
    
    # 에폭별로 처리
    for epoch_str, checkpoint_info in sorted(best_checkpoints.items(), key=lambda x: int(x[0])):
        epoch = int(epoch_str)
        file_path = checkpoint_info['file']
        
        # 파일 크기 (처리 전)
        if Path(file_path).exists():
            size_before = Path(file_path).stat().st_size / (1024 * 1024)
            total_original += size_before
            
            # 클리닝 실행
            if clean_single_checkpoint(file_path, epoch, in_place=True):
                success_count += 1
                
                # 파일 크기 (처리 후)
                size_after = Path(file_path).stat().st_size / (1024 * 1024)
                total_cleaned += size_after
    
    # 결과 요약
    print("-" * 40)
    print(f"\n✅ 처리 완료: {success_count}/{len(best_checkpoints)} 성공")
    print(f"원본 총 크기: {total_original/1024:.1f}GB")
    print(f"클린 총 크기: {total_cleaned/1024:.1f}GB")
    print(f"절약된 공간: {(total_original - total_cleaned)/1024:.1f}GB")
    
    # 메타데이터 업데이트 알림
    print("\n⚠️ 다음 단계:")
    print("1. metadata.json 파일 크기 정보 업데이트 필요")
    print("2. enhanced_checkpoint_manager.py 수정")
    print("3. 학습 재개 전 검증")

if __name__ == "__main__":
    main()