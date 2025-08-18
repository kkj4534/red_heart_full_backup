#!/usr/bin/env python3
"""
의미 모델만 단독 테스트
"""

import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_semantic_model():
    print("🧠 의미 모델 단독 테스트")
    
    try:
        from models.semantic_models.advanced_semantic_models import (
            SemanticAnalysisConfig, AdvancedSemanticModel
        )
        
        # 설정
        config = SemanticAnalysisConfig(vocab_size=1000, embedding_dim=256)
        print(f"✅ 설정 생성: vocab_size={config.vocab_size}, embedding_dim={config.embedding_dim}")
        
        # 모델 생성
        model = AdvancedSemanticModel(config)
        print(f"✅ 모델 생성 성공")
        
        # 테스트 입력 (토큰 ID)
        batch_size = 4
        sequence_length = 50
        token_ids = torch.randint(0, 1000, (batch_size, sequence_length))
        
        print(f"📊 입력 차원: {token_ids.shape}")
        print(f"📊 입력 타입: {token_ids.dtype}")
        
        # 추론 테스트
        with torch.no_grad():
            output = model(token_ids)
        
        print(f"✅ 추론 성공")
        print(f"📊 출력 키: {list(output.keys())}")
        if 'enhanced_semantics' in output:
            print(f"📊 강화 의미 차원: {output['enhanced_semantics'].shape}")
        
        return True, output
        
    except Exception as e:
        print(f"❌ 의미 모델 오류: {e}")
        import traceback
        traceback.print_exc()
        return False, None

if __name__ == "__main__":
    success, output = test_semantic_model()
    if success:
        print("🎉 의미 모델 테스트 성공!")
    else:
        print("💥 의미 모델 테스트 실패!")