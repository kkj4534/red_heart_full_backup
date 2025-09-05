#!/usr/bin/env python3
"""
간단한 토크나이즈 테스트
"""
import torch
import sys
sys.path.append('/mnt/c/large_project/linux_red_heart')

def test_simple():
    print("📊 간단한 토크나이즈 테스트")
    print("="*50)
    
    # SentenceTransformer 테스트
    from sentence_transformer_singleton import get_sentence_transformer
    
    model = get_sentence_transformer(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        device="cuda"
    )
    
    text = "친구가 어려운 상황에 처했을 때 어떻게 도와야 할까?"
    print(f"텍스트: {text}")
    print(f"단어 수: {len(text.split())}")
    
    # encode 호출
    embeddings = model.encode([text])
    print(f"\n1. encode 결과:")
    print(f"   타입: {type(embeddings)}")
    
    if isinstance(embeddings, list):
        print(f"   길이: {len(embeddings)}")
        if embeddings:
            first = embeddings[0]
            print(f"   embeddings[0] 타입: {type(first)}")
            if isinstance(first, list):
                print(f"   embeddings[0] 길이: {len(first)}")
                # 텐서 변환 시도
                tensor = torch.tensor(first, dtype=torch.float32)
                print(f"   텐서 shape: {tensor.shape}")
                
                # 잘못된 변환 시뮬레이션
                # 혹시 단어별로 처리했다면?
                words = text.split()
                print(f"\n2. 혹시 단어별 처리?")
                print(f"   단어 리스트: {words}")
                print(f"   단어 수: {len(words)}")
                
                # 단어를 인덱스로 변환했다면? (가정)
                word_indices = list(range(len(words)))
                wrong_tensor = torch.tensor([word_indices], dtype=torch.float32)
                print(f"   잘못된 텐서 shape: {wrong_tensor.shape}")
                
    print("\n" + "="*50)

if __name__ == "__main__":
    test_simple()