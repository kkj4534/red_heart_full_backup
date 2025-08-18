#!/usr/bin/env python3
"""
반사실 모델만 단독 테스트
"""

import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_counterfactual_model():
    print("🧠 반사실 모델 단독 테스트")
    
    try:
        from models.counterfactual_models.counterfactual_reasoning_models import (
            CounterfactualConfig, AdvancedCounterfactualModel
        )
        
        # 설정
        config = CounterfactualConfig(input_dim=768, hidden_dims=[256, 128], latent_dim=32)
        print(f"✅ 설정 생성: input_dim={config.input_dim}")
        
        # 모델 생성
        model = AdvancedCounterfactualModel(config)
        print(f"✅ 모델 생성 성공")
        
        # 테스트 입력
        batch_size = 4
        text_embeddings = torch.randn(batch_size, 768)
        
        print(f"📊 입력 차원: {text_embeddings.shape}")
        
        # 추론 테스트
        with torch.no_grad():
            output = model(text_embeddings)
        
        print(f"✅ 추론 성공")
        print(f"📊 출력 키: {list(output.keys())}")
        if 'counterfactual_scenarios' in output:
            print(f"📊 시나리오 개수: {len(output['counterfactual_scenarios'])}")
        
        return True, output
        
    except Exception as e:
        print(f"❌ 반사실 모델 오류: {e}")
        import traceback
        traceback.print_exc()
        return False, None

if __name__ == "__main__":
    success, output = test_counterfactual_model()
    if success:
        print("🎉 반사실 모델 테스트 성공!")
    else:
        print("💥 반사실 모델 테스트 실패!")