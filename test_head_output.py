#!/usr/bin/env python3
"""
헤드 출력 형태 테스트
"""

import torch
import sys
sys.path.append('/mnt/c/large_project/linux_red_heart')

def test_head_output():
    print("🔍 헤드 출력 형태 테스트")
    print("="*50)
    
    # 백본과 헤드 테스트
    from unified_backbone import RedHeartUnifiedBackbone
    from unified_heads import EmotionHead
    
    # 백본 설정
    backbone_config = {
        'input_dim': 768,
        'd_model': 896,
        'num_layers': 8,
        'num_heads': 14,
        'feedforward_dim': 3584,
        'dropout': 0.1,
        'task_dim': 896
    }
    
    backbone = RedHeartUnifiedBackbone(backbone_config)
    backbone.eval()
    
    # 감정 헤드 설정 - EmotionHead는 input_dim만 받음
    emotion_head = EmotionHead(input_dim=896)
    emotion_head.eval()
    
    # 테스트 입력
    print("1. 백본 테스트 (1x512x768):")
    test_input = torch.randn(1, 512, 768)
    
    with torch.no_grad():
        backbone_output = backbone(test_input, task='emotion')
        print(f"   백본 출력 키: {backbone_output.keys()}")
        
        if 'emotion' in backbone_output:
            emotion_features = backbone_output['emotion']
            print(f"   emotion features shape: {emotion_features.shape}")
            
            # 헤드에 전달
            print("\n2. 헤드 테스트:")
            head_output = emotion_head(emotion_features)
            
            if isinstance(head_output, dict):
                print(f"   헤드 출력 타입: dict")
                for key, value in head_output.items():
                    if isinstance(value, torch.Tensor):
                        print(f"   - {key}: {value.shape}")
                    else:
                        print(f"   - {key}: {type(value)}")
            elif isinstance(head_output, torch.Tensor):
                print(f"   헤드 출력 shape: {head_output.shape}")
            else:
                print(f"   헤드 출력 타입: {type(head_output)}")
                
            # DSP Simulator 테스트
            print("\n3. DSP Simulator 테스트:")
            from emotion_dsp_simulator import EmotionDSPSimulator
            
            dsp_config = {
                'num_emotions': 7,
                'filter_order': 2,
                'sample_rate': 100,
                'resonance': 0.7
            }
            
            dsp = EmotionDSPSimulator(dsp_config)
            dsp.eval()
            
            # 실제 head_output을 DSP에 전달
            if isinstance(head_output, dict):
                # dict에서 첫 번째 텐서 추출
                actual_head_output = head_output.get('emotions', 
                                     head_output.get('emotion_logits', 
                                     list(head_output.values())[0] if head_output else None))
            else:
                actual_head_output = head_output
            
            print(f"   DSP 입력 shape: {actual_head_output.shape if isinstance(actual_head_output, torch.Tensor) else type(actual_head_output)}")
            
            try:
                dsp_output = dsp.forward(actual_head_output)
                print(f"   ✅ DSP 성공: 출력 shape = {dsp_output.shape if isinstance(dsp_output, torch.Tensor) else type(dsp_output)}")
            except Exception as e:
                print(f"   ❌ DSP 실패: {e}")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    test_head_output()