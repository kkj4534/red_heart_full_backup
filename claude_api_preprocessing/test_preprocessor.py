#!/usr/bin/env python3
"""
Claude 전처리 시스템 테스트
소량 샘플로 전체 기능 검증
"""

import json
import asyncio
import logging
from pathlib import Path
from claude_complete_preprocessor import ClaudeCompletePreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Test.Preprocessor')

async def test_small_batch():
    """소량 샘플 테스트"""
    
    # API 키 확인
    api_key_path = Path("api_key.json")
    if not api_key_path.exists():
        logger.error("api_key.json 파일이 없습니다.")
        return
    
    with open(api_key_path, 'r') as f:
        config = json.load(f)
    
    if config['anthropic_api_key'] == "YOUR_ANTHROPIC_API_KEY_HERE":
        logger.error("api_key.json에 실제 API 키를 입력하세요.")
        return
    
    # 테스트 샘플 준비
    test_samples = [
        {
            "id": "test_001",
            "text": "I feel terrible about what I did. I betrayed my best friend's trust by sharing their secret with others. The guilt is eating me alive and I don't know how to make things right.",
            "type": "anecdote"
        },
        {
            "id": "test_002", 
            "text": "Today was absolutely amazing! I got promoted at work and my family surprised me with a celebration party. I've never felt so happy and grateful in my life.",
            "type": "anecdote"
        },
        {
            "id": "test_003",
            "text": "I need to choose between: telling the truth and potentially hurting someone's feelings OR keeping quiet and living with the lie.",
            "type": "dilemma"
        }
    ]
    
    # 테스트 파일 생성
    test_file = Path("test_samples.jsonl")
    with open(test_file, 'w', encoding='utf-8') as f:
        for sample in test_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info(f"테스트 샘플 {len(test_samples)}개 준비 완료")
    
    # 전처리 실행
    preprocessor = ClaudeCompletePreprocessor()
    
    try:
        logger.info("\n=== 테스트 시작 ===")
        await preprocessor.process_batch(test_samples, output_dir="test_output")
        
        # 결과 확인
        output_file = Path("test_output/claude_preprocessed_complete.jsonl")
        if output_file.exists():
            logger.info("\n=== 테스트 결과 ===")
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    result = json.loads(line)
                    logger.info(f"\n샘플 ID: {result['id']}")
                    logger.info(f"감정: {result['emotion_labels']}")
                    logger.info(f"후회 지수: {result['regret_factor']:.2f}")
                    logger.info(f"벤담: {result['bentham_scores']}")
                    logger.info(f"SURD: {result['surd_metrics']}")
        
        # 비용 확인
        total_input, total_output = preprocessor.stats.get_total_tokens()
        cost = preprocessor.stats.get_cost_estimate()
        
        logger.info(f"\n=== 테스트 비용 ===")
        logger.info(f"입력 토큰: {total_input}")
        logger.info(f"출력 토큰: {total_output}")
        logger.info(f"비용: ${cost:.4f} ({cost*1320:.0f}원)")
        
        # 샘플당 비용 추정
        if len(test_samples) > 0:
            cost_per_sample = cost / len(test_samples)
            estimated_20k = cost_per_sample * 20000
            logger.info(f"\n=== 20,000개 예상 ===")
            logger.info(f"샘플당: ${cost_per_sample:.4f}")
            logger.info(f"총 비용: ${estimated_20k:.2f} ({estimated_20k*1320:,.0f}원)")
        
    except Exception as e:
        logger.error(f"테스트 실패: {e}")

if __name__ == "__main__":
    asyncio.run(test_small_batch())