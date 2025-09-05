#!/usr/bin/env python3
"""
_tokenize 메서드 디버깅
"""

import sys
import torch
import logging
import asyncio
sys.path.append('/mnt/c/large_project/linux_red_heart')

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)

from main_unified import UnifiedInferenceSystem, InferenceConfig, MemoryMode

async def debug_tokenize():
    """_tokenize 메서드 디버깅"""
    print("="*60)
    print("🔍 _tokenize 메서드 디버깅")
    print("="*60)
    
    # 설정
    config = InferenceConfig(
        memory_mode=MemoryMode.LIGHT,
        auto_memory_mode=False,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=True
    )
    
    print("\n1. 시스템 초기화...")
    system = UnifiedInferenceSystem(config)
    await system.initialize()
    
    print("\n2. _tokenize 메서드 직접 호출...")
    text = "친구가 어려운 상황에 처했을 때 어떻게 도와야 할까?"
    print(f"   입력 텍스트: '{text}'")
    print(f"   단어 수: {len(text.split())}")
    
    # _tokenize 직접 호출
    inputs = system._tokenize(text)
    
    print(f"\n3. _tokenize 결과:")
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            print(f"   - {key}: shape={value.shape}, dtype={value.dtype}")
            print(f"     첫 5개 값: {value.flatten()[:5].tolist()}")
        else:
            print(f"   - {key}: {value}")
    
    print("\n4. UnifiedModel 호출 테스트...")
    try:
        with torch.no_grad():
            outputs = system.unified_model(
                x=inputs['embeddings'],
                task='emotion',
                return_all=True
            )
        print("   ✅ UnifiedModel 호출 성공!")
        print(f"   출력 키: {outputs.keys() if isinstance(outputs, dict) else type(outputs)}")
    except Exception as e:
        print(f"   ❌ UnifiedModel 호출 실패: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("✅ 디버깅 완료")

if __name__ == "__main__":
    asyncio.run(debug_tokenize())