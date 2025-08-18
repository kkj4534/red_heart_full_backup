#!/usr/bin/env python3
"""
Claude API 경량 전처리 시스템
- 불필요한 임베딩 제거
- 원본 Scruples 파일 직접 사용
- 순수 API 처리만
"""

import os
import json
import time
import logging
import asyncio
import pickle
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import anthropic
from anthropic import Anthropic, RateLimitError, APIError, AuthenticationError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Claude.LightPreprocessor')

@dataclass
class ProcessingStats:
    """처리 통계"""
    total_samples: int = 0
    processed: int = 0
    failed: int = 0
    total_cost: float = 0.0
    start_time: Optional[datetime] = None
    
    def get_cost_krw(self) -> float:
        return self.total_cost * 1320

class ClaudeLightPreprocessor:
    """Claude API 경량 전처리기"""
    
    def __init__(self):
        """초기화"""
        # API 키 로드
        api_key_path = Path("api_key.json")
        if not api_key_path.exists():
            raise ValueError("api_key.json 파일이 없습니다.")
        
        with open(api_key_path, 'r') as f:
            api_config = json.load(f)
        
        self.api_key = api_config.get('anthropic_api_key')
        if not self.api_key or self.api_key == "YOUR_ANTHROPIC_API_KEY_HERE":
            raise ValueError("api_key.json에 실제 API 키를 입력하세요.")
        
        # Claude 클라이언트
        self.client = Anthropic(api_key=self.api_key)
        
        # Rate limit
        self.rate_limit_delay = 3.5
        self.last_request_time = 0
        
        # 통계
        self.stats = ProcessingStats()
        
        # 체크포인트
        self.checkpoint_path = Path("checkpoint_light.pkl")
        self.processed_ids = set()
    
    async def wait_for_rate_limit(self):
        """Rate limit 대기"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            wait_time = self.rate_limit_delay - time_since_last
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    async def analyze_all_in_one(self, text: str) -> Optional[Dict[str, Any]]:
        """모든 분석을 한 번의 API 호출로"""
        await self.wait_for_rate_limit()
        
        # 통합 프롬프트
        prompt = f"""Analyze this text for multiple aspects. Return ONLY a JSON object with these exact fields:

1. emotions: array of 7 floats (0.0-1.0) for [Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger]
2. regret: single float (0.0-1.0) for regret level
3. bentham: object with intensity, duration, certainty, propinquity (each 0.0-1.0)
4. surd: object with selection, uncertainty, risk, decision (each 0.0-1.0)

Text: {text[:1000]}  # 토큰 절약을 위해 1000자로 제한

Return ONLY valid JSON:"""
        
        try:
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=200,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = message.content[0].text.strip()
            
            # 비용 계산 (추정)
            input_tokens = len(prompt) / 4
            output_tokens = len(response_text) / 4
            cost = (input_tokens * 3 + output_tokens * 15) / 1_000_000
            self.stats.total_cost += cost
            
            # JSON 파싱
            # JSON 블록 찾기
            json_match = re.search(r'\{[^}]*"emotions"[^}]*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    
                    # 검증
                    if 'emotions' in result and len(result['emotions']) == 7:
                        return result
                except:
                    pass
            
            # 실패 시 기본값
            return None
            
        except Exception as e:
            logger.error(f"API 오류: {e}")
            return None
    
    async def process_samples_direct(self, input_file: str, limit: int = 3):
        """원본 파일에서 직접 처리"""
        self.stats.start_time = datetime.now()
        
        # 체크포인트 로드
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
                self.processed_ids = checkpoint.get('processed_ids', set())
                logger.info(f"체크포인트: {len(self.processed_ids)}개 완료")
        
        # 출력 파일
        output_file = Path("preprocessed_light.jsonl")
        
        # 원본 파일에서 직접 읽기
        input_path = Path(input_file)
        if not input_path.exists():
            # 기본 경로 시도
            input_path = Path(f"../for_learn_dataset/scruples_real_data/anecdotes/{input_file}")
        
        if not input_path.exists():
            logger.error(f"파일 없음: {input_path}")
            return
        
        logger.info(f"입력: {input_path}")
        processed = 0
        
        with open(input_path, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'a', encoding='utf-8') as f_out:
            
            for i, line in enumerate(f_in):
                if processed >= limit:
                    break
                
                sample = json.loads(line)
                sample_id = sample.get('id', str(i))
                
                # 이미 처리됨
                if sample_id in self.processed_ids:
                    continue
                
                # 텍스트 추출
                text = sample.get('text', '')
                if not text:
                    continue
                
                logger.info(f"\n처리 중 [{processed+1}/{limit}]: {sample_id}")
                logger.info(f"텍스트: {text[:100]}...")
                
                # API 호출 (모든 분석 한번에)
                result = await self.analyze_all_in_one(text)
                
                if result:
                    # 출력 형식
                    output = {
                        'id': sample_id,
                        'text': text[:500],
                        'title': sample.get('title', ''),
                        **result,  # emotions, regret, bentham, surd
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    f_out.write(json.dumps(output, ensure_ascii=False) + '\n')
                    f_out.flush()
                    
                    self.processed_ids.add(sample_id)
                    self.stats.processed += 1
                    processed += 1
                    
                    logger.info(f"✅ 성공")
                    logger.info(f"감정: {result['emotions']}")
                    logger.info(f"후회: {result.get('regret', 'N/A')}")
                else:
                    self.stats.failed += 1
                    logger.error(f"❌ 실패")
                
                # 체크포인트
                if processed % 10 == 0:
                    with open(self.checkpoint_path, 'wb') as f:
                        pickle.dump({'processed_ids': self.processed_ids}, f)
        
        # 최종 통계
        duration = (datetime.now() - self.stats.start_time).total_seconds()
        logger.info(f"\n=== 완료 ===")
        logger.info(f"처리: {self.stats.processed}")
        logger.info(f"실패: {self.stats.failed}")
        logger.info(f"시간: {duration:.1f}초")
        logger.info(f"비용: ${self.stats.total_cost:.4f} ({self.stats.get_cost_krw():.0f}원)")
        
        if self.stats.processed > 0:
            per_sample = self.stats.total_cost / self.stats.processed
            logger.info(f"샘플당: ${per_sample:.4f}")
            logger.info(f"20,000개 예상: ${per_sample * 20000:.2f}")

async def test_quick():
    """빠른 테스트"""
    preprocessor = ClaudeLightPreprocessor()
    
    # 원본 파일에서 3개만 테스트
    await preprocessor.process_samples_direct(
        "train.scruples-anecdotes.jsonl",
        limit=3
    )

if __name__ == "__main__":
    # 체크포인트 초기화
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--reset":
        Path("checkpoint_light.pkl").unlink(missing_ok=True)
        Path("preprocessed_light.jsonl").unlink(missing_ok=True)
        print("체크포인트 초기화됨")
    
    asyncio.run(test_quick())