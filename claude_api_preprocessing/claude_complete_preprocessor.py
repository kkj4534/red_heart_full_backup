#!/usr/bin/env python3
"""
Claude API 완전 전처리 시스템
data_preprocessing_pipeline_v3.py의 모든 기능을 Claude API로 구현
- 감정 분석, 후회 지수, 벤담 점수, SURD 메트릭
- 체크포인트 및 재개 기능
- 크레딧 소진 감지 및 처리
"""

import os
import json
import time
import logging
import asyncio
import pickle
import re
import gc
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import anthropic
from anthropic import Anthropic, RateLimitError, APIError, AuthenticationError
from sentence_transformers import SentenceTransformer
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Claude.CompletePreprocessor')

@dataclass
class ProcessingStats:
    """처리 통계"""
    total_samples: int = 0
    processed: int = 0
    failed: int = 0
    retried: int = 0
    
    # 토큰 추적
    emotion_input_tokens: int = 0
    emotion_output_tokens: int = 0
    regret_input_tokens: int = 0
    regret_output_tokens: int = 0
    bentham_input_tokens: int = 0
    bentham_output_tokens: int = 0
    surd_input_tokens: int = 0
    surd_output_tokens: int = 0
    
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    credit_exhausted: bool = False
    
    def get_total_tokens(self) -> Tuple[int, int]:
        """총 토큰 수 반환"""
        total_input = (self.emotion_input_tokens + self.regret_input_tokens + 
                      self.bentham_input_tokens + self.surd_input_tokens)
        total_output = (self.emotion_output_tokens + self.regret_output_tokens + 
                       self.bentham_output_tokens + self.surd_output_tokens)
        return total_input, total_output
    
    def get_cost_estimate(self) -> float:
        """비용 계산 (USD)"""
        total_input, total_output = self.get_total_tokens()
        input_cost = (total_input / 1_000_000) * 3.0  # $3/1M
        output_cost = (total_output / 1_000_000) * 15.0  # $15/1M
        return input_cost + output_cost
    
    def get_cost_krw(self) -> float:
        """원화 비용"""
        return self.get_cost_estimate() * 1320  # 환율 1320원 기준

class ClaudeCompletePreprocessor:
    """Claude API 완전 전처리기"""
    
    def __init__(self):
        """초기화"""
        # API 키 로드
        api_key_path = Path("api_key.json")
        if not api_key_path.exists():
            raise ValueError("api_key.json 파일이 없습니다. API 키를 설정하세요.")
        
        with open(api_key_path, 'r') as f:
            api_config = json.load(f)
        
        self.api_key = api_config.get('anthropic_api_key')
        if not self.api_key or self.api_key == "YOUR_ANTHROPIC_API_KEY_HERE":
            raise ValueError("api_key.json에 실제 API 키를 입력하세요.")
        
        # Claude 클라이언트
        self.client = Anthropic(api_key=self.api_key)
        
        # Sentence Transformer (로컬 임베딩용)
        logger.info("Sentence Transformer 로딩...")
        self.sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Rate limit 설정
        self.rate_limit_delay = 3.5  # 초 (분당 17회, 안전 마진)
        self.last_request_time = 0
        
        # 통계
        self.stats = ProcessingStats()
        
        # 체크포인트 경로
        self.checkpoint_path = Path("checkpoint_complete.pkl")
        self.processed_ids = set()
        
        # 시스템 프롬프트들 (캐싱용)
        self.prompts = self._initialize_prompts()
    
    def _initialize_prompts(self) -> Dict[str, str]:
        """시스템 프롬프트 초기화"""
        return {
            'emotion': """You are an advanced emotion analysis system. Analyze emotions and return ONLY a JSON array with 7 float values (0.0-1.0) for:
[Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger]

Rules:
1. Values should sum approximately to 1.0
2. Multiple emotions can coexist
3. Be nuanced - avoid 0.0 or 1.0 unless certain
4. Return ONLY the JSON array""",
            
            'regret': """Rate the regret level in this text from 0 (no regret) to 10 (extreme regret).
Consider: guilt, remorse, disappointment with past actions.
Return ONLY a single number 0-10.""",
            
            'bentham': """Analyze hedonic aspects (Bentham's felicific calculus).
Rate each 0-10:
1. Intensity: strength of happiness/pleasure
2. Duration: how long-lasting
3. Certainty: how likely/certain
4. Propinquity: how close in time

Format:
intensity: [number]
duration: [number]
certainty: [number]
propinquity: [number]""",
            
            'surd': """Analyze decision-making aspects:
1. Selection (0-10): How much choice/agency
2. Uncertainty (0-10): Level of unknown factors
3. Risk (0-10): Potential negative consequences
4. Decision (0-10): Importance of decision

Format:
selection: [number]
uncertainty: [number]
risk: [number]
decision: [number]"""
        }
    
    def load_checkpoint(self) -> Dict[str, Any]:
        """체크포인트 로드"""
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
                self.processed_ids = checkpoint.get('processed_ids', set())
                self.stats = checkpoint.get('stats', ProcessingStats())
                logger.info(f"✅ 체크포인트 로드: {len(self.processed_ids)}개 처리 완료")
                
                # 크레딧 소진 상태 확인
                if self.stats.credit_exhausted:
                    logger.warning("⚠️ 이전 세션에서 크레딧 소진으로 중단됨")
                    logger.info("계속하려면 크레딧을 충전했는지 확인하세요.")
                    response = input("계속하시겠습니까? (y/n): ")
                    if response.lower() != 'y':
                        raise SystemExit("사용자가 중단함")
                    self.stats.credit_exhausted = False
                
                return checkpoint.get('last_results', {})
        return {}
    
    def save_checkpoint(self, last_results: Dict[str, Any] = None):
        """체크포인트 저장"""
        checkpoint = {
            'processed_ids': self.processed_ids,
            'stats': self.stats,
            'timestamp': datetime.now(),
            'last_results': last_results or {}
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
    
    async def analyze_emotion(self, text: str) -> Optional[List[float]]:
        """감정 분석"""
        await self.wait_for_rate_limit()
        
        try:
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=100,
                temperature=0.7,
                system=self.prompts['emotion'],
                messages=[{"role": "user", "content": f"Analyze: {text}"}]
            )
            
            response_text = message.content[0].text.strip()
            
            # 토큰 카운트 업데이트 (추정)
            self.stats.emotion_input_tokens += len(self.prompts['emotion'] + text) // 4
            self.stats.emotion_output_tokens += len(response_text) // 4
            
            # JSON 파싱
            json_match = re.search(r'\[[\d\.,\s]+\]', response_text)
            if json_match:
                emotion_vector = json.loads(json_match.group())
                if len(emotion_vector) == 7:
                    # 정규화
                    total = sum(emotion_vector)
                    if total > 0:
                        emotion_vector = [v/total for v in emotion_vector]
                    return emotion_vector
            
            return None
            
        except AuthenticationError as e:
            logger.error("❌ 인증 실패 - API 키 확인 필요")
            self.stats.credit_exhausted = True
            self.save_checkpoint()
            raise
            
        except Exception as e:
            if "credit" in str(e).lower() or "billing" in str(e).lower():
                logger.error("💳 크레딧 소진 감지!")
                self.stats.credit_exhausted = True
                self.save_checkpoint()
                raise SystemExit("크레딧 소진 - 충전 후 재실행하세요")
            raise
    
    async def calculate_regret(self, text: str) -> float:
        """후회 지수 계산"""
        await self.wait_for_rate_limit()
        
        try:
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=50,
                temperature=0.5,
                system=self.prompts['regret'],
                messages=[{"role": "user", "content": text}]
            )
            
            response_text = message.content[0].text.strip()
            
            # 토큰 카운트
            self.stats.regret_input_tokens += len(self.prompts['regret'] + text) // 4
            self.stats.regret_output_tokens += len(response_text) // 4
            
            # 숫자 추출
            numbers = re.findall(r'\d+(?:\.\d+)?', response_text)
            if numbers:
                score = float(numbers[0])
                if 0 <= score <= 10:
                    return score / 10.0  # 0-1로 정규화
            
            return 0.5  # 기본값
            
        except Exception as e:
            logger.error(f"후회 지수 계산 실패: {e}")
            return 0.5
    
    async def calculate_bentham(self, text: str) -> Dict[str, float]:
        """벤담 점수 계산"""
        await self.wait_for_rate_limit()
        
        try:
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=100,
                temperature=0.6,
                system=self.prompts['bentham'],
                messages=[{"role": "user", "content": text}]
            )
            
            response_text = message.content[0].text.lower()
            
            # 토큰 카운트
            self.stats.bentham_input_tokens += len(self.prompts['bentham'] + text) // 4
            self.stats.bentham_output_tokens += len(response_text) // 4
            
            # 파싱
            scores = {}
            aspects = ['intensity', 'duration', 'certainty', 'propinquity']
            
            for aspect in aspects:
                pattern = rf'{aspect}[^:]*:?\s*(\d+(?:\.\d+)?)'
                match = re.search(pattern, response_text)
                if match:
                    scores[aspect] = float(match.group(1)) / 10.0
                else:
                    scores[aspect] = 0.5
            
            return scores
            
        except Exception as e:
            logger.error(f"벤담 점수 계산 실패: {e}")
            return {'intensity': 0.5, 'duration': 0.5, 'certainty': 0.5, 'propinquity': 0.5}
    
    async def calculate_surd(self, text: str) -> Dict[str, float]:
        """SURD 메트릭 계산"""
        await self.wait_for_rate_limit()
        
        try:
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=100,
                temperature=0.6,
                system=self.prompts['surd'],
                messages=[{"role": "user", "content": text}]
            )
            
            response_text = message.content[0].text.lower()
            
            # 토큰 카운트
            self.stats.surd_input_tokens += len(self.prompts['surd'] + text) // 4
            self.stats.surd_output_tokens += len(response_text) // 4
            
            # 파싱
            scores = {}
            aspects = ['selection', 'uncertainty', 'risk', 'decision']
            
            for aspect in aspects:
                pattern = rf'{aspect}[^:]*:?\s*(\d+(?:\.\d+)?)'
                match = re.search(pattern, response_text)
                if match:
                    scores[aspect] = float(match.group(1)) / 10.0
                else:
                    scores[aspect] = 0.5
            
            return scores
            
        except Exception as e:
            logger.error(f"SURD 메트릭 계산 실패: {e}")
            return {'selection': 0.5, 'uncertainty': 0.5, 'risk': 0.5, 'decision': 0.5}
    
    def generate_context_embedding(self, text: str) -> List[float]:
        """컨텍스트 임베딩 생성 (로컬)"""
        try:
            embedding = self.sentence_model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")
            return [0.0] * 768  # 기본 차원
    
    async def process_single_sample(self, sample: Dict[str, Any], retry_count: int = 0) -> Optional[Dict[str, Any]]:
        """단일 샘플 전처리"""
        text = sample.get('text', '')
        if not text:
            return None
        
        try:
            # 1. 감정 분석
            emotions = await self.analyze_emotion(text)
            if not emotions:
                raise ValueError("감정 분석 실패")
            
            # 2. 후회 지수
            regret = await self.calculate_regret(text)
            
            # 3. 벤담 점수
            bentham = await self.calculate_bentham(text)
            
            # 4. SURD 메트릭
            surd = await self.calculate_surd(text)
            
            # 5. 컨텍스트 임베딩 (로컬)
            embedding = self.generate_context_embedding(text)
            
            return {
                'id': sample.get('id', ''),
                'text': text[:500],  # 처음 500자만 저장
                'source': sample.get('type', 'unknown'),
                'emotion_vector': emotions,
                'emotion_labels': {
                    'joy': emotions[0],
                    'trust': emotions[1],
                    'fear': emotions[2],
                    'surprise': emotions[3],
                    'sadness': emotions[4],
                    'disgust': emotions[5],
                    'anger': emotions[6]
                },
                'regret_factor': regret,
                'bentham_scores': bentham,
                'surd_metrics': surd,
                'context_embedding': embedding,
                'timestamp': datetime.now().isoformat(),
                'metadata': {
                    'title': sample.get('title', ''),
                    'action': sample.get('action', ''),
                    'label': sample.get('label', ''),
                    'type': sample.get('type', '')
                }
            }
            
        except Exception as e:
            if retry_count < 3 and "rate" not in str(e).lower():
                logger.warning(f"재시도 {retry_count + 1}/3")
                await asyncio.sleep(5)
                return await self.process_single_sample(sample, retry_count + 1)
            
            logger.error(f"샘플 처리 실패: {e}")
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
        
        success_file = output_dir / "claude_preprocessed_complete.jsonl"
        failed_file = output_dir / "claude_failed_complete.jsonl"
        
        # 예상 비용 계산
        remaining = len(samples) - len(self.processed_ids)
        estimated_cost = remaining * 1720 * 3 / 1_000_000 + remaining * 210 * 15 / 1_000_000
        estimated_krw = estimated_cost * 1320
        estimated_time = remaining * 3.5 * 4 / 3600  # 4개 API 호출
        
        logger.info(f"=== 전처리 시작 ===")
        logger.info(f"총 샘플: {len(samples)}개")
        logger.info(f"이미 처리: {len(self.processed_ids)}개")
        logger.info(f"남은 샘플: {remaining}개")
        logger.info(f"예상 비용: ${estimated_cost:.2f} (약 {estimated_krw:,.0f}원)")
        logger.info(f"예상 시간: {estimated_time:.1f}시간")
        
        # 처리
        with open(success_file, 'a', encoding='utf-8') as sf, \
             open(failed_file, 'a', encoding='utf-8') as ff:
            
            for i, sample in enumerate(samples):
                sample_id = sample.get('id', str(i))
                
                # 이미 처리됨
                if sample_id in self.processed_ids:
                    continue
                
                # 진행 상황
                if self.stats.processed % 10 == 0 and self.stats.processed > 0:
                    elapsed = (datetime.now() - self.stats.start_time).total_seconds()
                    speed = self.stats.processed / elapsed if elapsed > 0 else 0
                    eta = (remaining - self.stats.processed) / speed if speed > 0 else 0
                    
                    total_input, total_output = self.stats.get_total_tokens()
                    current_cost = self.stats.get_cost_estimate()
                    
                    logger.info(f"\n=== 진행 상황 ===")
                    logger.info(f"처리: {self.stats.processed}/{remaining} ({self.stats.processed/remaining*100:.1f}%)")
                    logger.info(f"속도: {speed*3600:.1f} samples/hour")
                    logger.info(f"ETA: {eta/3600:.1f}시간")
                    logger.info(f"토큰: 입력 {total_input:,} / 출력 {total_output:,}")
                    logger.info(f"현재 비용: ${current_cost:.2f} ({current_cost*1320:,.0f}원)")
                
                # 처리
                try:
                    result = await self.process_single_sample(sample)
                    
                    if result:
                        # 성공
                        sf.write(json.dumps(result, ensure_ascii=False) + '\n')
                        sf.flush()
                        
                        self.processed_ids.add(sample_id)
                        self.stats.processed += 1
                        
                        logger.info(f"✅ {sample_id}: 성공")
                    else:
                        # 실패
                        failed_sample = {
                            'id': sample_id,
                            'text': sample.get('text', '')[:500],
                            'error': 'Processing failed',
                            'timestamp': datetime.now().isoformat()
                        }
                        ff.write(json.dumps(failed_sample, ensure_ascii=False) + '\n')
                        ff.flush()
                        
                        self.stats.failed += 1
                        logger.error(f"❌ {sample_id}: 실패")
                    
                except AuthenticationError:
                    logger.error("💳 크레딧 문제 - 중단")
                    self.save_checkpoint({'last_sample_id': sample_id})
                    break
                    
                except Exception as e:
                    logger.error(f"예외 발생: {e}")
                    self.stats.failed += 1
                
                # 체크포인트 저장 (50개마다)
                if self.stats.processed % 50 == 0:
                    self.save_checkpoint()
                
                # 메모리 정리
                if i % 100 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        # 최종 통계
        self.stats.end_time = datetime.now()
        duration = (self.stats.end_time - self.stats.start_time).total_seconds()
        
        logger.info("\n=== 처리 완료 ===")
        logger.info(f"성공: {self.stats.processed}")
        logger.info(f"실패: {self.stats.failed}")
        logger.info(f"소요 시간: {duration/3600:.1f}시간")
        logger.info(f"총 비용: ${self.stats.get_cost_estimate():.2f} ({self.stats.get_cost_krw():,.0f}원)")
        
        # 최종 체크포인트
        self.save_checkpoint()

async def main():
    """메인 함수"""
    # 데이터 로드
    input_file = Path("scruples_prepared.jsonl")
    if not input_file.exists():
        logger.error("scruples_prepared.jsonl 파일이 없습니다.")
        logger.info("먼저 prepare_scruples_data.py를 실행하세요.")
        return
    
    samples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    
    logger.info(f"로드된 샘플: {len(samples)}개")
    
    # 전처리 실행
    preprocessor = ClaudeCompletePreprocessor()
    
    try:
        await preprocessor.process_batch(samples)
    except KeyboardInterrupt:
        logger.info("\n사용자 중단 - 체크포인트 저장")
        preprocessor.save_checkpoint()
    except SystemExit as e:
        logger.info(f"\n시스템 종료: {e}")
    except Exception as e:
        logger.error(f"\n예외 발생: {e}")
        preprocessor.save_checkpoint()

if __name__ == "__main__":
    asyncio.run(main())