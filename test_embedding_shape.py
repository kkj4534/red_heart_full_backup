#!/usr/bin/env python3
"""
SentenceTransformer 임베딩 차원 테스트
"""

import sys
import torch
import numpy as np
sys.path.append('/mnt/c/large_project/linux_red_heart')

from sentence_transformer_singleton import get_sentence_transformer

def test_embedding_shape():
    """임베딩 형태 테스트"""
    print("📊 SentenceTransformer 임베딩 형태 테스트")
    print("="*50)
    
    # 모델 로드
    print("1. 모델 로드 중...")
    model = get_sentence_transformer(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 테스트 텍스트
    test_text = "친구가 어려운 상황에 처했을 때 어떻게 도와야 할까?"
    print(f"2. 테스트 텍스트: '{test_text}'")
    print(f"   텍스트 길이: {len(test_text)} 문자")
    print(f"   단어 수: {len(test_text.split())} 단어")
    
    # 임베딩 생성
    print("\n3. 임베딩 생성...")
    embeddings = model.encode([test_text])
    
    # 형태 확인
    print(f"\n4. 임베딩 정보:")
    print(f"   - type(embeddings): {type(embeddings)}")
    
    if isinstance(embeddings, list):
        print(f"   - len(embeddings): {len(embeddings)}")
        print(f"   - type(embeddings[0]): {type(embeddings[0])}")
        
        # 리스트의 첫 번째 요소 분석
        first_embedding = embeddings[0]
        if hasattr(first_embedding, 'shape'):
            print(f"   - embeddings[0].shape: {first_embedding.shape}")
            print(f"   - embeddings[0].dtype: {first_embedding.dtype}")
        else:
            print(f"   - embeddings[0] 내용 샘플: {first_embedding[:10] if len(first_embedding) > 10 else first_embedding}")
            print(f"   - len(embeddings[0]): {len(first_embedding) if hasattr(first_embedding, '__len__') else 'N/A'}")
    elif isinstance(embeddings, np.ndarray):
        print(f"   - embeddings.shape: {embeddings.shape}")
        print(f"   - embeddings.dtype: {embeddings.dtype}")
        print(f"   - embeddings[0][:10]: {embeddings[0][:10]}")
    
    # 텐서 변환 테스트
    print("\n5. 텐서 변환 테스트:")
    embedding_tensor = torch.tensor(embeddings[0], dtype=torch.float32)
    print(f"   - embedding_tensor.shape: {embedding_tensor.shape}")
    print(f"   - embedding_tensor.dim(): {embedding_tensor.dim()}")
    
    # 차원 확장 테스트
    print("\n6. 차원 확장 테스트:")
    if len(embedding_tensor.shape) == 1:
        print(f"   - 원본: {embedding_tensor.shape}")
        expanded = embedding_tensor.unsqueeze(0).unsqueeze(0)
        print(f"   - unsqueeze(0).unsqueeze(0): {expanded.shape}")
        
        # 패딩 테스트
        max_seq_length = 128
        padded_tensor = torch.zeros(1, max_seq_length, embedding_tensor.shape[-1])
        padded_tensor[:, 0, :] = expanded[0, 0, :]
        print(f"   - 패딩 후: {padded_tensor.shape}")
    
    print("\n7. 예상 백본 입력:")
    print(f"   - 기대: (batch_size=1, seq_len={max_seq_length}, input_dim=768)")
    print(f"   - 실제: {padded_tensor.shape}")
    
    # 백본 프로젝션 테스트
    print("\n8. 백본 프로젝션 시뮬레이션:")
    input_projection = torch.nn.Linear(768, 896)
    try:
        # 정상 케이스
        output = input_projection(padded_tensor)
        print(f"   ✅ 정상 처리: 입력 {padded_tensor.shape} -> 출력 {output.shape}")
    except Exception as e:
        print(f"   ❌ 에러: {e}")
    
    # 문제 재현 테스트 
    print("\n9. 문제 재현 테스트 (1x7 형태):")
    wrong_tensor = torch.randn(1, 7)  # 문제가 된 형태
    print(f"   - 잘못된 입력: {wrong_tensor.shape}")
    try:
        output = input_projection(wrong_tensor)
        print(f"   출력: {output.shape}")
    except Exception as e:
        print(f"   ❌ 예상된 에러: {e}")
    
    print("\n" + "="*50)
    print("✅ 테스트 완료")

if __name__ == "__main__":
    test_embedding_shape()