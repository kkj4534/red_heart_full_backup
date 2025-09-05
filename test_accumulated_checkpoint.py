#!/usr/bin/env python3
"""누적 데이터가 있는 체크포인트 테스트 (에폭 5)"""

from clean_checkpoint_safe import clean_checkpoint_safely
import sys

# 에폭 5 체크포인트 (5.2GB - 누적 데이터 포함)
test_checkpoint = "training/checkpoints_final/checkpoint_epoch_0005_lr_0.000012_20250824_004444.pt"

print("="*80)
print("에폭 5 체크포인트 테스트 (5.2GB, 누적 데이터 포함)")
print("="*80)

# 테스트 모드로 실행 (실제 저장 안함)
success = clean_checkpoint_safely(test_checkpoint, test_mode=True)

if success:
    print("\n✅ 테스트 성공! 누적 데이터 제거 가능 확인")
    print("\n다음 단계:")
    print("1. 실제 클리닝 테스트 (에폭 5만)")
    print("2. 전체 체크포인트 일괄 처리")
else:
    print("\n❌ 테스트 실패")
    sys.exit(1)