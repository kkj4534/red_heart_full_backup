#!/usr/bin/env python3
"""큰 체크포인트 테스트"""

from clean_checkpoint_safe import clean_checkpoint_safely

# 5.2GB 크기의 체크포인트 테스트
large_checkpoint = "training/checkpoints_final/checkpoint_epoch_0023_lr_0.000011_20250824_204202.pt"

print("큰 체크포인트(5.2GB) 테스트 모드 실행...")
clean_checkpoint_safely(large_checkpoint, test_mode=True)