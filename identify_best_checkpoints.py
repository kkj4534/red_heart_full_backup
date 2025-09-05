#!/usr/bin/env python3
"""
각 에폭별 최적 체크포인트 식별 스크립트
중복된 체크포인트 중 최신/최적 선택
"""
import os
from pathlib import Path
import json
from datetime import datetime

def identify_best_checkpoints():
    """각 에폭별 최적 체크포인트 식별"""
    
    checkpoint_dir = Path("training/checkpoints_final")
    
    # 모든 체크포인트 수집
    all_checkpoints = {}
    
    for ckpt_file in checkpoint_dir.glob("checkpoint_epoch_*.pt"):
        # 파일명에서 정보 추출
        filename = ckpt_file.name
        parts = filename.split('_')
        
        try:
            epoch = int(parts[2])
            lr = parts[4]
            timestamp = parts[5].replace('.pt', '')
            
            file_size_mb = ckpt_file.stat().st_size / (1024 * 1024)
            
            if epoch not in all_checkpoints:
                all_checkpoints[epoch] = []
            
            all_checkpoints[epoch].append({
                'file': str(ckpt_file),
                'filename': filename,
                'epoch': epoch,
                'lr': lr,
                'timestamp': timestamp,
                'size_mb': file_size_mb
            })
        except Exception as e:
            print(f"파싱 실패: {filename} - {e}")
    
    # 각 에폭별 최적 선택
    best_checkpoints = {}
    
    print("=" * 80)
    print("각 에폭별 체크포인트 분석")
    print("=" * 80)
    
    for epoch in sorted(all_checkpoints.keys()):
        candidates = all_checkpoints[epoch]
        print(f"\n에폭 {epoch:2d}: {len(candidates)}개 체크포인트")
        
        for c in candidates:
            print(f"  - {c['filename']}: {c['size_mb']:.1f}MB")
        
        # 최신 타임스탬프 선택 (가장 나중에 저장된 것)
        best = max(candidates, key=lambda x: x['timestamp'])
        best_checkpoints[epoch] = best
        print(f"  ✅ 선택: {best['filename']}")
    
    # 결과 저장
    with open("best_checkpoints_selection.json", "w") as f:
        json.dump(best_checkpoints, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"총 {len(best_checkpoints)}개 에폭의 최적 체크포인트 선택 완료")
    print("결과: best_checkpoints_selection.json")
    
    # 공간 계산
    total_current = sum(c['size_mb'] for c in best_checkpoints.values())
    estimated_clean = len(best_checkpoints) * 600  # 예상 클린 크기
    
    print(f"\n현재 총 크기: {total_current/1024:.1f}GB")
    print(f"예상 클린 크기: {estimated_clean/1024:.1f}GB")
    print(f"예상 절약 공간: {(total_current - estimated_clean)/1024:.1f}GB")
    
    return best_checkpoints

if __name__ == "__main__":
    identify_best_checkpoints()