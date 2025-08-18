#!/usr/bin/env python3
"""
메가 스케일 모델만 단독 테스트
"""

import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_megascale_model():
    print("🧠 메가 스케일 모델 단독 테스트")
    
    try:
        from models.mega_scale_models.scalable_xai_model import create_mega_scale_model, optimize_model_for_inference
        
        # 작은 크기로 우선 테스트
        print("모델 생성 중...")
        model = create_mega_scale_model(target_params=50_000_000)  # 5천만으로 축소
        model = optimize_model_for_inference(model)
        
        actual_params = model.get_parameter_count()
        print(f"✅ 모델 생성 성공: {actual_params:,}개 파라미터")
        
        # 테스트 입력
        batch_size = 2
        seq_len = 32  # 더 작은 시퀀스
        input_dim = 1024
        
        test_input = torch.randn(batch_size, seq_len, input_dim)
        print(f"📊 입력 차원: {test_input.shape}")
        
        # 추론 테스트
        with torch.no_grad():
            outputs = model(test_input)
        
        print(f"✅ 추론 성공")
        print(f"📊 출력 키: {list(outputs.keys())}")
        if 'emotion_predictions' in outputs:
            print(f"📊 감정 예측 차원: {outputs['emotion_predictions'].shape}")
        
        return True, outputs
        
    except Exception as e:
        print(f"❌ 메가 스케일 모델 오류: {e}")
        import traceback
        traceback.print_exc()
        return False, None

if __name__ == "__main__":
    success, output = test_megascale_model()
    if success:
        print("🎉 메가 스케일 모델 테스트 성공!")
    else:
        print("💥 메가 스케일 모델 테스트 실패!")