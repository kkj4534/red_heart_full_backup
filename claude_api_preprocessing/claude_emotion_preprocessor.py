#!/usr/bin/env python3
"""
Claude API 기반 감정 분석 전처리 시스템
- Rate limit 안전 처리 (3.5초 간격)
- 자동 재시도 및 복구
- 프롬프트 캐싱으로 비용 절감
"""

import os
import json
import time
import logging
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import anthropic
from anthropic import Anthropic, RateLimitError, APIError
import pickle
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Claude.EmotionPreprocessor')

@dataclass
class ProcessingStats:
    """처리 통계"""
    total_samples: int = 0
    processed: int = 0
    failed: int = 0
    retried: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def get_cost_estimate(self) -> float:
        """비용 계산 (USD)"""
        input_cost = (self.total_input_tokens / 1_000_000) * 3.0  # $3/1M
        output_cost = (self.total_output_tokens / 1_000_000) * 15.0  # $15/1M
        return input_cost + output_cost
    
    def get_cost_krw(self) -> float:
        """원화 비용"""
        return self.get_cost_estimate() * 1320  # 환율 1320원 기준

class ClaudeEmotionPreprocessor:
    """Claude API 감정 분석 전처리기"""
    
    def __init__(self, api_key: str = None):
        """초기화"""
        # API 키 설정
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("API 키가 필요합니다. ANTHROPIC_API_KEY 환경변수 또는 api_key 파라미터로 제공하세요.")
        
        # Claude 클라이언트
        self.client = Anthropic(api_key=self.api_key)
        
        # Rate limit 설정
        self.rate_limit_delay = 3.5  # 초 (분당 17회, 안전 마진)
        self.last_request_time = 0
        
        # 통계
        self.stats = ProcessingStats()
        
        # 체크포인트 경로
        self.checkpoint_path = Path("checkpoint.pkl")
        self.processed_ids = set()
        
        # 시스템 프롬프트 (캐싱용)
        self.system_prompt = """You are an advanced emotion analysis system. Analyze the given text and return ONLY a JSON array with 7 float values (0.0-1.0) representing:
[Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger]

Rules:
1. Values should sum approximately to 1.0 but can vary (0.8-1.2 total)
2. Multiple emotions can coexist
3. Be nuanced - avoid 0.0 or 1.0 unless absolutely certain
4. Consider context and subtle emotional cues
5. Return ONLY the JSON array, no explanation"""
    
    def load_checkpoint(self):
        """체크포인트 로드"""
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
                self.processed_ids = checkpoint.get('processed_ids', set())
                self.stats = checkpoint.get('stats', ProcessingStats())
                logger.info(f"체크포인트 로드: {len(self.processed_ids)}개 처리 완료")
    
    def save_checkpoint(self):
        """체크포인트 저장"""
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
            logger.debug(f"Rate limit 대기: {wait_time:.1f}초")
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    async def analyze_emotion(self, text: str, sample_id: str = None, retry_count: int = 0) -> Optional[List[float]]:
        """단일 텍스트 감정 분석"""
        
        # Rate limit 대기
        await self.wait_for_rate_limit()
        
        try:
            # Claude API 호출
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=100,
                temperature=0.7 + (retry_count * 0.05),  # 재시도마다 온도 증가
                system=self.system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": f"Analyze emotions in: {text}"
                    }
                ]
            )
            
            # 응답 파싱
            response_text = message.content[0].text.strip()
            
            # 토큰 카운트 업데이트 (추정치)
            self.stats.total_input_tokens += len(self.system_prompt + text) // 4
            self.stats.total_output_tokens += len(response_text) // 4
            
            # JSON 파싱
            json_match = re.search(r'\[[\d\.,\s]+\]', response_text)
            if json_match:
                emotion_vector = json.loads(json_match.group())
                if len(emotion_vector) == 7:
                    logger.info(f"✅ {sample_id}: {emotion_vector}")
                    return emotion_vector
            
            logger.warning(f"파싱 실패: {response_text[:100]}")
            return None
            
        except RateLimitError as e:
            logger.warning(f"Rate limit 도달, 30초 대기...")
            await asyncio.sleep(30)
            if retry_count < 3:
                self.stats.retried += 1
                return await self.analyze_emotion(text, sample_id, retry_count + 1)
            
        except APIError as e:
            logger.error(f"API 에러: {e}")
            if retry_count < 3:
                self.stats.retried += 1
                await asyncio.sleep(5)
                return await self.analyze_emotion(text, sample_id, retry_count + 1)
            
        except Exception as e:
            logger.error(f"예외 발생: {e}")
        
        return None
    
    async def process_batch(self, samples: List[Dict[str, Any]], output_dir: str = "."):
        """배치 처리"""
        self.stats.total_samples = len(samples)
        self.stats.start_time = datetime.now()
        
        # 체크포인트 로드
        self.load_checkpoint()
        
        # 출력 파일 준비
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        success_file = output_dir / "claude_preprocessed_dataset.jsonl"
        failed_file = output_dir / "claude_failed_samples.jsonl"
        
        # 기존 결과 로드
        existing_results = []
        if success_file.exists():
            with open(success_file, 'r', encoding='utf-8') as f:
                for line in f:
                    existing_results.append(json.loads(line))
        
        logger.info(f"처리 시작: {len(samples)}개 샘플")
        logger.info(f"예상 시간: {len(samples) * 3.5 / 3600:.1f}시간")
        logger.info(f"예상 비용: {len(samples) * 0.0005:.2f} USD (~{len(samples) * 0.66:.0f}원)")
        
        # 처리
        success_f = open(success_file, 'a', encoding='utf-8')
        failed_f = open(failed_file, 'a', encoding='utf-8')
        
        try:
            for i, sample in enumerate(samples):
                sample_id = sample.get('id', str(i))
                
                # 이미 처리됨
                if sample_id in self.processed_ids:
                    continue
                
                # 진행 상황 출력
                if i % 10 == 0:
                    progress = (len(self.processed_ids) / len(samples)) * 100
                    elapsed = (datetime.now() - self.stats.start_time).total_seconds()
                    eta = (elapsed / max(len(self.processed_ids), 1)) * (len(samples) - len(self.processed_ids))
                    
                    logger.info(f"진행: {progress:.1f}% ({len(self.processed_ids)}/{len(samples)})")
                    logger.info(f"예상 잔여 시간: {eta/3600:.1f}시간")
                    logger.info(f"현재 비용: ${self.stats.get_cost_estimate():.2f} ({self.stats.get_cost_krw():.0f}원)")
                
                # 감정 분석
                text = sample.get('text', sample.get('content', ''))
                emotions = await self.analyze_emotion(text, sample_id)
                
                if emotions:
                    result = {
                        'id': sample_id,
                        'text': text[:200],  # 처음 200자만 저장
                        'emotions': emotions,
                        'emotion_labels': {
                            'joy': emotions[0],
                            'trust': emotions[1],
                            'fear': emotions[2],
                            'surprise': emotions[3],
                            'sadness': emotions[4],
                            'disgust': emotions[5],
                            'anger': emotions[6]
                        },
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # 성공 저장
                    success_f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    success_f.flush()
                    
                    # 통계 업데이트
                    self.processed_ids.add(sample_id)
                    self.stats.processed += 1
                else:
                    self.stats.failed += 1
                    logger.error(f"처리 실패: {sample_id}")
                    
                    # 실패 샘플 저장
                    failed_sample = {
                        'id': sample_id,
                        'text': text[:500],
                        'error': 'Failed to extract emotions',
                        'timestamp': datetime.now().isoformat()
                    }
                    failed_f.write(json.dumps(failed_sample, ensure_ascii=False) + '\n')
                    failed_f.flush()
                
                # 체크포인트 저장 (50개마다)
                if len(self.processed_ids) % 50 == 0:
                    self.save_checkpoint()
        
        finally:
            # 파일 닫기
            success_f.close()
            failed_f.close()
        
        # 최종 통계
        self.stats.end_time = datetime.now()
        duration = (self.stats.end_time - self.stats.start_time).total_seconds()
        
        logger.info("=== 처리 완료 ===")
        logger.info(f"처리: {self.stats.processed}/{self.stats.total_samples}")
        logger.info(f"실패: {self.stats.failed}")
        logger.info(f"재시도: {self.stats.retried}")
        logger.info(f"소요 시간: {duration/3600:.1f}시간")
        logger.info(f"총 비용: ${self.stats.get_cost_estimate():.2f} ({self.stats.get_cost_krw():.0f}원)")
        
        # 최종 체크포인트
        self.save_checkpoint()

async def main():
    """메인 함수"""
    # 설정 파일 읽기
    config_path = Path("config.json")
    if not config_path.exists():
        # 샘플 설정 생성
        sample_config = {
            "api_key": "YOUR_ANTHROPIC_API_KEY_HERE",
            "input_file": "scruples_prepared.jsonl",
            "output_dir": ".",
            "max_samples": 20000,
            "rate_limit_delay": 3.5,
            "use_prepared_data": True
        }
        
        with open(config_path, 'w') as f:
            json.dump(sample_config, f, indent=2)
        
        logger.info(f"설정 파일 생성됨: {config_path}")
        logger.info("config.json에 API 키를 입력하고 다시 실행하세요.")
        return
    
    # 설정 로드
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # API 키 확인
    if config['api_key'] == "YOUR_ANTHROPIC_API_KEY_HERE":
        logger.error("config.json에 실제 API 키를 입력하세요.")
        return
    
    # 샘플 로드
    input_path = Path(config['input_file'])
    if not input_path.exists():
        logger.error(f"입력 파일 없음: {input_path}")
        return
    
    samples = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
            if len(samples) >= config['max_samples']:
                break
    
    logger.info(f"로드된 샘플: {len(samples)}개")
    
    # 전처리 실행
    preprocessor = ClaudeEmotionPreprocessor(api_key=config['api_key'])
    preprocessor.rate_limit_delay = config.get('rate_limit_delay', 3.5)
    
    await preprocessor.process_batch(samples, config.get('output_dir', '.'))

if __name__ == "__main__":
    asyncio.run(main())