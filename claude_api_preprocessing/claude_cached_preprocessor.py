#!/usr/bin/env python3
"""
Claude API 캐싱 최적화 전처리 시스템
- 프롬프트 캐싱으로 90% 비용 절감
- 통합 API 호출로 횟수 최소화
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
logger = logging.getLogger('Claude.CachedPreprocessor')

@dataclass
class ProcessingStats:
    """처리 통계"""
    total_samples: int = 0
    processed: int = 0
    failed: int = 0
    
    # 토큰 추적 (캐싱 구분)
    cache_write_tokens: int = 0  # 첫 캐시 쓰기
    cache_read_tokens: int = 0   # 캐시 읽기
    regular_tokens: int = 0      # 일반 토큰
    output_tokens: int = 0       # 출력 토큰
    
    start_time: Optional[datetime] = None
    credit_exhausted: bool = False
    
    def get_cost_estimate(self) -> float:
        """캐싱 적용 비용 계산"""
        # 캐시 쓰기: $3.75/1M
        cache_write_cost = (self.cache_write_tokens / 1_000_000) * 3.75
        # 캐시 읽기: $0.30/1M (90% 절감!)
        cache_read_cost = (self.cache_read_tokens / 1_000_000) * 0.30
        # 일반 입력: $3/1M
        regular_cost = (self.regular_tokens / 1_000_000) * 3.0
        # 출력: $15/1M
        output_cost = (self.output_tokens / 1_000_000) * 15.0
        
        return cache_write_cost + cache_read_cost + regular_cost + output_cost
    
    def get_cost_krw(self) -> float:
        return self.get_cost_estimate() * 1320

class ClaudeCachedPreprocessor:
    """캐싱 최적화 전처리기"""
    
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
        
        # Claude 클라이언트 (캐싱 베타 헤더 포함)
        self.client = Anthropic(
            api_key=self.api_key,
            default_headers={
                "anthropic-beta": "prompt-caching-2024-07-31"
            }
        )
        
        # Rate limit (Tier 2: 1000/min 가능)
        self.rate_limit_delay = 0.8  # 안전하게 분당 75개
        self.last_request_time = 0
        
        # 통계
        self.stats = ProcessingStats()
        
        # 체크포인트
        self.checkpoint_path = Path("checkpoint_cached.pkl")
        self.processed_ids = set()
        
        # 통합 시스템 프롬프트 (캐싱 대상)
        self.system_prompt = self._create_unified_prompt()
    
    def _create_unified_prompt(self) -> str:
        """통합 시스템 프롬프트 생성"""
        return """You are an advanced AI system for comprehensive text analysis.

For each text, analyze ALL of the following aspects and return a single JSON object:

1. **emotions**: Array of 7 floats (0.0-1.0) for [Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger]
   - Values should sum approximately to 1.0
   - Multiple emotions can coexist
   - Be nuanced - avoid extreme values unless certain

2. **regret_factor**: Single float (0.0-1.0) for regret level
   - Consider guilt, remorse, disappointment with past actions
   - 0.0 = no regret, 1.0 = extreme regret

3. **bentham_scores**: Object with 4 aspects of hedonic calculus (each 0.0-1.0)
   - intensity: strength of happiness/pleasure
   - duration: how long-lasting
   - certainty: how likely/certain
   - propinquity: how close in time

4. **surd_metrics**: Object with 4 decision-making aspects (each 0.0-1.0)
   - selection: amount of choice/agency
   - uncertainty: level of unknown factors
   - risk: potential negative consequences
   - decision: importance of decision

Return ONLY a valid JSON object with this exact structure:
{
  "emotions": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "regret_factor": 0.0,
  "bentham_scores": {
    "intensity": 0.0,
    "duration": 0.0,
    "certainty": 0.0,
    "propinquity": 0.0
  },
  "surd_metrics": {
    "selection": 0.0,
    "uncertainty": 0.0,
    "risk": 0.0,
    "decision": 0.0
  }
}

Important: Return ONLY the JSON object, no explanations or additional text."""
    
    def load_checkpoint(self):
        """체크포인트 로드"""
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
                self.processed_ids = checkpoint.get('processed_ids', set())
                self.stats = checkpoint.get('stats', ProcessingStats())
                logger.info(f"✅ 체크포인트 로드: {len(self.processed_ids)}개 완료")
                
                if self.stats.credit_exhausted:
                    logger.warning("⚠️ 이전 세션에서 크레딧 소진으로 중단됨")
                    response = input("계속하시겠습니까? (y/n): ")
                    if response.lower() != 'y':
                        raise SystemExit("사용자가 중단함")
                    self.stats.credit_exhausted = False
    
    def save_checkpoint(self):
        """체크포인트 저장 (ID만)"""
        checkpoint = {
            'processed_ids': self.processed_ids,
            'stats': self.stats,
            'timestamp': datetime.now()
        }
        with open(self.checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
    
    async def wait_for_rate_limit(self):
        """Rate limit 대기"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            wait_time = self.rate_limit_delay - time_since_last
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    async def analyze_unified(self, text: str, is_first: bool = False) -> Optional[Dict[str, Any]]:
        """통합 분석 (모든 메트릭 한번에 + 캐싱)"""
        await self.wait_for_rate_limit()
        
        try:
            # 텍스트 길이 제한 (토큰 절약)
            text_truncated = text[:1500] if len(text) > 1500 else text
            
            # 캐싱 활성화 API 호출
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=300,
                temperature=0.7,
                system=[
                    {
                        "type": "text",
                        "text": self.system_prompt,
                        "cache_control": {"type": "ephemeral"}  # 캐싱!
                    }
                ],
                messages=[
                    {
                        "role": "user",
                        "content": f"Analyze this text:\n\n{text_truncated}"
                    }
                ]
            )
            
            response_text = message.content[0].text.strip()
            
            # 사용량 추적 (캐싱 구분)
            if hasattr(message, 'usage'):
                usage = message.usage
                
                # 캐시 관련 토큰
                cache_creation = getattr(usage, 'cache_creation_input_tokens', 0)
                cache_read = getattr(usage, 'cache_read_input_tokens', 0)
                
                if cache_creation > 0:
                    self.stats.cache_write_tokens += cache_creation
                    logger.info(f"💾 캐시 생성: {cache_creation} 토큰")
                
                if cache_read > 0:
                    self.stats.cache_read_tokens += cache_read
                    logger.debug(f"📖 캐시 읽기: {cache_read} 토큰 (90% 절감!)")
                
                # 일반 토큰 (캐시되지 않은 부분)
                regular_input = usage.input_tokens - cache_creation - cache_read
                self.stats.regular_tokens += regular_input
                
                # 출력 토큰
                self.stats.output_tokens += usage.output_tokens
            
            # JSON 파싱
            try:
                # JSON 블록 찾기
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    
                    # 검증
                    required_keys = ['emotions', 'regret_factor', 'bentham_scores', 'surd_metrics']
                    if all(key in result for key in required_keys):
                        if len(result['emotions']) == 7:
                            return result
                
                # 파싱 실패
                logger.warning(f"JSON 파싱 실패: {response_text[:200]}")
                return None
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON 디코드 에러: {e}")
                return None
            
        except AuthenticationError:
            logger.error("❌ 인증 실패 - API 키 확인")
            self.stats.credit_exhausted = True
            self.save_checkpoint()
            raise
            
        except Exception as e:
            if "credit" in str(e).lower() or "billing" in str(e).lower():
                logger.error("💳 크레딧 소진!")
                self.stats.credit_exhausted = True
                self.save_checkpoint()
                raise SystemExit("크레딧 소진 - 충전 후 재실행하세요")
            
            logger.error(f"API 오류: {e}")
            return None
    
    async def process_samples(self, input_file: str, limit: int = None):
        """샘플 처리"""
        self.stats.start_time = datetime.now()
        
        # 체크포인트 로드
        self.load_checkpoint()
        
        # 기존 JSON 파일에서 데이터 복원
        success_samples = []
        failed_samples = []
        
        success_file = Path("claude_preprocessed_complete.json")
        if success_file.exists():
            with open(success_file, 'r', encoding='utf-8') as f:
                success_samples = json.load(f)
                # 이미 처리된 ID 추가
                for sample in success_samples:
                    self.processed_ids.add(sample['id'])
                logger.info(f"✅ 기존 JSON에서 {len(success_samples)}개 복원")
        
        # 파일 경로
        input_path = Path(input_file)
        if not input_path.exists():
            input_path = Path(f"../for_learn_dataset/scruples_real_data/anecdotes/{input_file}")
        
        if not input_path.exists():
            logger.error(f"파일 없음: {input_path}")
            return
        
        # 출력 파일 (JSON 형식으로 변경)
        success_file = Path("claude_preprocessed_complete.json")
        failed_file = Path("claude_failed_complete.json")
        
        logger.info(f"\n=== 캐싱 최적화 전처리 시작 ===")
        logger.info(f"입력: {input_path}")
        logger.info(f"처리 대상: {limit if limit else '전체'}")
        
        processed_count = 0
        is_first = True  # 첫 요청 여부 (캐시 생성)
        
        # JSON 리스트로 수집
        success_samples = []
        failed_samples = []
        
        with open(input_path, 'r', encoding='utf-8') as f_in:
            
            for i, line in enumerate(f_in):
                if limit and processed_count >= limit:
                    break
                
                sample = json.loads(line)
                sample_id = sample.get('id', str(i))
                
                # 이미 처리됨
                if sample_id in self.processed_ids:
                    continue
                
                text = sample.get('text', '')
                if not text:
                    continue
                
                # 진행 상황
                if processed_count % 10 == 0 and processed_count > 0:
                    current_cost = self.stats.get_cost_estimate()
                    logger.info(f"\n진행: {processed_count}/{limit if limit else '∞'}")
                    logger.info(f"캐시 절감: {self.stats.cache_read_tokens:,} 토큰")
                    logger.info(f"현재 비용: ${current_cost:.2f} ({current_cost*1320:.0f}원)")
                    
                    # 크레딧 한도 체크 ($51)
                    if current_cost >= 51.0:
                        logger.warning(f"💰 크레딧 한도 도달! ($51)")
                        logger.info(f"총 처리: {processed_count}개")
                        break
                
                # 통합 분석 (한 번의 API 호출로 모든 메트릭)
                result = await self.analyze_unified(text, is_first)
                is_first = False  # 첫 번째 이후로는 캐시 읽기
                
                if result:
                    # 성공
                    output = {
                        'id': sample_id,
                        'text': text[:500],
                        'title': sample.get('title', ''),
                        'action': sample.get('action', {}).get('description', '') if sample.get('action') else '',
                        'label': sample.get('label', ''),
                        **result,  # emotions, regret_factor, bentham_scores, surd_metrics
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    success_samples.append(output)
                    
                    self.processed_ids.add(sample_id)
                    self.stats.processed += 1
                    processed_count += 1
                    
                    logger.info(f"✅ [{processed_count}] {sample_id}")
                    
                else:
                    # 실패
                    failed = {
                        'id': sample_id,
                        'text': text[:500],
                        'error': 'Analysis failed',
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    failed_samples.append(failed)
                    
                    self.stats.failed += 1
                    logger.error(f"❌ {sample_id}")
                
                # 30개마다 JSON + 체크포인트 동시 저장 (동기화)
                if processed_count % 30 == 0:
                    # JSON 저장
                    with open(success_file, 'w', encoding='utf-8') as f:
                        json.dump(success_samples, f, ensure_ascii=False, indent=2)
                    if failed_samples:
                        with open(failed_file, 'w', encoding='utf-8') as f:
                            json.dump(failed_samples, f, ensure_ascii=False, indent=2)
                    
                    # 체크포인트도 함께 저장 (동기화)
                    self.save_checkpoint()
                    logger.info(f"💾 저장 완료: {processed_count}개 (JSON + 체크포인트)")
        
        # JSON 파일로 저장
        logger.info(f"\n📝 JSON 파일 저장 중...")
        with open(success_file, 'w', encoding='utf-8') as f:
            json.dump(success_samples, f, ensure_ascii=False, indent=2)
        
        if failed_samples:
            with open(failed_file, 'w', encoding='utf-8') as f:
                json.dump(failed_samples, f, ensure_ascii=False, indent=2)
        
        # 최종 통계
        duration = (datetime.now() - self.stats.start_time).total_seconds()
        total_cost = self.stats.get_cost_estimate()
        
        logger.info(f"\n=== 완료 ===")
        logger.info(f"처리: {self.stats.processed}")
        logger.info(f"실패: {self.stats.failed}")
        logger.info(f"시간: {duration/60:.1f}분")
        logger.info(f"\n=== 토큰 사용량 ===")
        logger.info(f"캐시 생성: {self.stats.cache_write_tokens:,} 토큰 ($3.75/1M)")
        logger.info(f"캐시 읽기: {self.stats.cache_read_tokens:,} 토큰 ($0.30/1M) ← 90% 절감!")
        logger.info(f"일반 입력: {self.stats.regular_tokens:,} 토큰 ($3/1M)")
        logger.info(f"출력: {self.stats.output_tokens:,} 토큰 ($15/1M)")
        logger.info(f"\n=== 비용 ===")
        logger.info(f"총 비용: ${total_cost:.2f} ({total_cost*1320:.0f}원)")
        
        if self.stats.processed > 0:
            per_sample = total_cost / self.stats.processed
            savings = (self.stats.cache_read_tokens * 2.7) / 1_000_000  # 절감액
            logger.info(f"샘플당: ${per_sample:.4f}")
            logger.info(f"캐싱 절감액: ${savings:.2f} ({savings*1320:.0f}원)")
            logger.info(f"\n20,000개 예상:")
            logger.info(f"  캐싱 없이: ${per_sample * 20000 * 4:.2f}")  # 4배 (캐싱 없이)
            logger.info(f"  캐싱 적용: ${per_sample * 20000:.2f}")
        
        # 최종 체크포인트
        self.save_checkpoint()

async def test_cached():
    """캐싱 테스트"""
    preprocessor = ClaudeCachedPreprocessor()
    
    # 3개 샘플 테스트
    await preprocessor.process_samples(
        "train.scruples-anecdotes.jsonl",
        limit=3
    )

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--reset":
            # 체크포인트 초기화
            Path("checkpoint_cached.pkl").unlink(missing_ok=True)
            Path("claude_preprocessed_complete.json").unlink(missing_ok=True)
            Path("claude_failed_complete.json").unlink(missing_ok=True)
            print("✅ 캐시 관련 파일 초기화됨")
        elif sys.argv[1] == "--full":
            # 전체 실행
            asyncio.run(ClaudeCachedPreprocessor().process_samples(
                "train.scruples-anecdotes.jsonl",
                limit=20000
            ))
        else:
            # 커스텀 limit 처리 (간단한 방식)
            try:
                limit = int(sys.argv[1])
                print(f"🚀 {limit}개 샘플 처리 시작...")
                asyncio.run(ClaudeCachedPreprocessor().process_samples(
                    "../for_learn_dataset/scruples_real_data/anecdotes/train.scruples-anecdotes.jsonl",
                    limit=limit
                ))
            except ValueError:
                print(f"❌ 잘못된 인자: {sys.argv[1]}")
                print("사용법: python3 claude_cached_preprocessor.py [숫자|--reset|--full]")
    else:
        # 기본: 3개 테스트
        asyncio.run(test_cached())