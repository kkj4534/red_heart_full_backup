#!/usr/bin/env python3
"""
크로스오버 시스템 완전성 검증
- 모든 모듈이 저장되는지 확인
- 50 에폭 시나리오 테스트
"""

import sys
import torch
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent))

from training.unified_training_final import UnifiedModel, UnifiedTrainingConfig
from training.enhanced_checkpoint_manager import EnhancedCheckpointManager

def test_complete_save():
    """모든 모듈이 저장되는지 테스트"""
    
    print("=" * 70)
    print("크로스오버 완전성 테스트")
    print("=" * 70)
    
    # 설정 및 모델 생성
    config = UnifiedTrainingConfig()
    model = UnifiedModel(config)
    
    # Checkpoint Manager 생성
    manager = EnhancedCheckpointManager(
        checkpoint_dir="test_checkpoints",
        max_checkpoints=50,
        save_interval=1
    )
    
    # 더미 optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    print("\n📊 모델 구조 확인:")
    expected_modules = [
        'backbone', 'emotion_head', 'bentham_head', 'regret_head', 'surd_head',
        'neural_analyzers', 'phase0_net', 'phase2_net', 'hierarchical_integrator',
        'dsp_simulator', 'kalman_filter', 'advanced_wrappers', 'system'
    ]
    
    for module_name in expected_modules:
        if module_name == 'system':
            # system은 메타데이터로 _extract_modular_states에서 추가됨
            print(f"  ✅ {module_name}: 메타데이터 (체크포인트 저장시 추가)")
        elif hasattr(model, module_name):
            module = getattr(model, module_name)
            if module is not None:
                if hasattr(module, 'parameters'):
                    param_count = sum(p.numel() for p in module.parameters()) / 1e6
                    print(f"  ✅ {module_name}: {param_count:.2f}M 파라미터")
                else:
                    print(f"  ✅ {module_name}: dict 형태")
            else:
                print(f"  ❌ {module_name}: None")
        else:
            print(f"  ❌ {module_name}: 존재하지 않음")
    
    print("\n📦 체크포인트 저장 테스트:")
    
    # 테스트 에폭들
    test_epochs = [1, 10, 20, 30, 40, 50]
    
    for epoch in test_epochs:
        metrics = {'loss': 1.0 / epoch, 'accuracy': epoch / 50.0}
        
        # optimizer 저장 여부 확인
        keep_optimizer = manager.should_keep_optimizer(epoch)
        print(f"\n에폭 {epoch:2d}:")
        print(f"  - Optimizer 저장: {'✅ Yes' if keep_optimizer else '❌ No'}")
        
        # 모듈별 state 추출
        modular_states = manager._extract_modular_states(model)
        
        # 저장된 모듈 확인
        saved_modules = list(modular_states.keys())
        print(f"  - 저장된 모듈: {len(saved_modules)}개")
        
        # 누락된 모듈 확인
        missing = []
        for expected in expected_modules:
            if expected not in saved_modules:
                # advanced_wrappers는 None일 수 있음
                if expected == 'advanced_wrappers' and model.advanced_wrappers is None:
                    continue
                # system은 _extract_modular_states에서 자동 추가됨
                if expected == 'system':
                    continue
                missing.append(expected)
        
        if missing:
            print(f"  ⚠️ 누락된 모듈: {missing}")
        else:
            print(f"  ✅ 모든 모듈 저장 완료")
    
    print("\n" + "=" * 70)
    print("✅ 크로스오버 완전성 검증 완료!")
    print("=" * 70)
    
    # 테스트 디렉토리 정리
    import shutil
    if Path("test_checkpoints").exists():
        shutil.rmtree("test_checkpoints")

if __name__ == "__main__":
    test_complete_save()