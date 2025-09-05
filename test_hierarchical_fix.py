#!/usr/bin/env python3
"""
HierarchicalEmotionIntegrator 수정 테스트
후회 메커니즘 기반 계층적 감정 통합 검증
"""

import sys
import torch
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent))

from phase_neural_networks import HierarchicalEmotionIntegrator

def test_hierarchical():
    """수정된 HierarchicalEmotionIntegrator 테스트"""
    
    print("=" * 60)
    print("HierarchicalEmotionIntegrator 검증")
    print("=" * 60)
    
    # 테스트 설정
    batch_size = 4
    input_dim = 896
    
    # 모듈 생성
    integrator = HierarchicalEmotionIntegrator(input_dim=input_dim)
    
    # 파라미터 확인
    total_params = sum(p.numel() for p in integrator.parameters())
    print(f"\n📊 총 파라미터: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # 테스트 입력
    features = torch.randn(batch_size, input_dim)
    phase0_out = torch.randn(batch_size, 7)  # 감정 7차원
    phase2_out = torch.randn(batch_size, 10)  # 공동체 10차원
    
    print("\n🧪 테스트 1: features만으로 처리")
    try:
        output1 = integrator(features)
        print(f"  ✅ 출력 shape: {output1.shape}")
        print(f"  ✅ 출력 범위: [{output1.min():.3f}, {output1.max():.3f}]")
    except Exception as e:
        print(f"  ❌ 실패: {e}")
    
    print("\n🧪 테스트 2: Phase0 출력 포함")
    try:
        output2 = integrator(features, phase0_out=phase0_out)
        print(f"  ✅ 출력 shape: {output2.shape}")
        print(f"  ✅ 출력 범위: [{output2.min():.3f}, {output2.max():.3f}]")
    except Exception as e:
        print(f"  ❌ 실패: {e}")
    
    print("\n🧪 테스트 3: Phase0 + Phase2 출력 포함")
    try:
        output3 = integrator(features, phase0_out=phase0_out, phase2_out=phase2_out)
        print(f"  ✅ 출력 shape: {output3.shape}")
        print(f"  ✅ 출력 범위: [{output3.min():.3f}, {output3.max():.3f}]")
    except Exception as e:
        print(f"  ❌ 실패: {e}")
    
    print("\n🧪 테스트 4: 역전파 테스트")
    try:
        output = integrator(features)
        loss = output.mean()
        loss.backward()
        
        # 그래디언트 확인
        grad_norms = {}
        for name, param in integrator.named_parameters():
            if param.grad is not None:
                grad_norms[name] = param.grad.norm().item()
        
        print(f"  ✅ 역전파 성공")
        print(f"  ✅ regret_weight grad: {grad_norms.get('regret_weight', 0):.6f}")
        print(f"  ✅ 총 {len(grad_norms)}개 파라미터에 그래디언트 생성")
    except Exception as e:
        print(f"  ❌ 실패: {e}")
    
    print("\n📝 구조 요약:")
    print("  - 자신의 감정 (Phase 1)")
    print("  - 타자 관점 감정 (Phase 0) - 후회 메커니즘")
    print("  - 공동체 감정 (Phase 2) - 배치 평균")
    print("  - 계층적 통합 → 원래 차원 복원")
    
    print("\n" + "=" * 60)
    print("✅ 테스트 완료!")
    print("=" * 60)

if __name__ == "__main__":
    test_hierarchical()