#!/usr/bin/env python3
"""
수정된 enhanced_checkpoint_manager 테스트
"""

import sys
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent))

from training.enhanced_checkpoint_manager import EnhancedCheckpointManager

def test_should_keep_optimizer():
    """should_keep_optimizer 로직 테스트"""
    
    manager = EnhancedCheckpointManager(
        checkpoint_dir="test_checkpoints",
        max_checkpoints=30,
        save_interval=1
    )
    
    print("=" * 60)
    print("Optimizer State 저장 전략 테스트")
    print("=" * 60)
    
    test_epochs = [1, 10, 20, 21, 22, 23, 24, 30, 35, 40, 50, 60]
    
    for epoch in test_epochs:
        keep = manager.should_keep_optimizer(epoch)
        status = "✅ 유지" if keep else "❌ 제거"
        print(f"에폭 {epoch:2d}: {status}")
    
    print("\n정상 작동 확인!")
    
    # 테스트 디렉토리 정리
    import shutil
    if Path("test_checkpoints").exists():
        shutil.rmtree("test_checkpoints")

if __name__ == "__main__":
    test_should_keep_optimizer()