#!/usr/bin/env python3
"""
감정 모델만 단독 테스트
"""

import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_emotion_model():
    print("🧠 감정 모델 단독 테스트")
    
    try:
        from models.hierarchical_emotion.emotion_phase_models import HierarchicalEmotionModel
        
        # 모델 생성
        model = HierarchicalEmotionModel(input_dim=768)
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
        print(f"📊 최종 감정 차원: {output['final_emotion'].shape}")
        
        return True, output
        
    except Exception as e:
        print(f"❌ 감정 모델 오류: {e}")
        import traceback
        traceback.print_exc()
        return False, None

if __name__ == "__main__":
    success, output = test_emotion_model()
    if success:
        print("🎉 감정 모델 테스트 성공!")
    else:
        print("💥 감정 모델 테스트 실패!")