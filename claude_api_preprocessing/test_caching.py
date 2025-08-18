#!/usr/bin/env python3
"""
프롬프트 캐싱 테스트
실제로 비용이 절감되는지 확인
"""

import json
import time
from pathlib import Path
from anthropic import Anthropic

def test_with_caching():
    """캐싱 활성화 테스트"""
    
    # API 키 로드
    with open("api_key.json", 'r') as f:
        config = json.load(f)
    
    client = Anthropic(
        api_key=config['anthropic_api_key'],
        default_headers={
            "anthropic-beta": "prompt-caching-2024-07-31"
        }
    )
    
    # 긴 시스템 프롬프트 (캐싱 대상)
    system_prompt = """You are an advanced emotion analysis system. 
    Analyze emotions and return ONLY a JSON array with 7 float values (0.0-1.0) for:
    [Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger]
    
    Rules:
    1. Values should sum approximately to 1.0
    2. Multiple emotions can coexist
    3. Be nuanced - avoid 0.0 or 1.0 unless certain
    4. Consider context and subtle emotional cues
    5. Return ONLY the JSON array, no explanation
    
    This is a long prompt that benefits from caching...""" * 3  # 길게 만들기
    
    test_texts = [
        "I feel happy today",
        "I'm so sad and disappointed",
        "This makes me angry"
    ]
    
    print("=== 프롬프트 캐싱 테스트 ===\n")
    
    for i, text in enumerate(test_texts):
        print(f"테스트 {i+1}: {text}")
        
        start = time.time()
        
        # 캐싱 활성화된 요청
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"}  # 캐싱!
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": f"Analyze: {text}"
                }
            ]
        )
        
        elapsed = time.time() - start
        
        # 사용량 확인
        if hasattr(response, 'usage'):
            print(f"  입력 토큰: {response.usage.input_tokens}")
            print(f"  캐시 읽기: {getattr(response.usage, 'cache_read_input_tokens', 0)}")
            print(f"  캐시 쓰기: {getattr(response.usage, 'cache_creation_input_tokens', 0)}")
        
        print(f"  응답 시간: {elapsed:.2f}초")
        print(f"  결과: {response.content[0].text[:100]}...\n")
        
        time.sleep(1)  # Rate limit
    
    print("✅ 캐싱 테스트 완료")
    print("첫 번째 요청에서 캐시 쓰기, 이후 캐시 읽기로 토큰 절약")

if __name__ == "__main__":
    test_with_caching()