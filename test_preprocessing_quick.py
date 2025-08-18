#!/usr/bin/env python3
"""빠른 전처리 테스트 - 1개 샘플만"""

import json
from data_preprocessing_pipeline_v3 import HelpingAIPreprocessor

def test_single_sample():
    """단일 샘플 테스트"""
    # 테스트 데이터
    test_data = [{
        'text': "I am feeling very happy today. The sun is shining bright.",
        'source': 'test',
        'metadata': {'ethical_dilemma': 'none'}
    }]
    
    # 테스트 데이터 저장
    with open('test_sample.json', 'w') as f:
        json.dump(test_data, f)
    
    # 전처리기 초기화
    print("🔄 전처리기 초기화...")
    preprocessor = HelpingAIPreprocessor()
    
    # 단일 샘플 처리
    print("📝 단일 샘플 처리 시작...")
    preprocessor.process_dataset(
        input_file="test_sample.json",
        output_file="test_output.json",
        limit=1
    )
    
    # 결과 확인
    with open('test_output.json', 'r') as f:
        result = json.load(f)
    
    print("\n✅ 테스트 완료!")
    print(f"처리된 샘플 수: {len(result)}")
    
    if result:
        sample = result[0]
        print("\n📊 처리 결과:")
        print(f"  - 감정 벡터: {sample.get('emotion_vector', [])[:3]}...")
        print(f"  - 후회 지수: {sample.get('regret_factor', 0):.3f}")
        print(f"  - 벤담 점수 키: {list(sample.get('bentham_scores', {}).keys())}")
        print(f"  - SURD 메트릭 키: {list(sample.get('surd_metrics', {}).keys())}")
        print(f"  - 임베딩 차원: {len(sample.get('context_embedding', []))}")

if __name__ == "__main__":
    test_single_sample()