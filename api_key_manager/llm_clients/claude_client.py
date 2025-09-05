"""
Claude API 클라이언트
"""

import os
import asyncio
import aiohttp
import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """LLM 응답 데이터 클래스"""
    generated_text: str
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ClaudeAPIClient:
    """Claude API 클라이언트"""
    
    def __init__(self):
        self.api_key = os.environ.get('ANTHROPIC_API_KEY', '')
        self.api_url = 'https://api.anthropic.com/v1/messages'
        self.model = 'claude-3-5-sonnet-20241022'  # 최신 모델
        self.logger = logging.getLogger(self.__class__.__name__)
        self.session = None
        
    async def initialize(self):
        """초기화"""
        if not self.api_key:
            # API 키 파일에서 읽기 시도
            key_file = os.path.join(os.path.dirname(__file__), '..', 'anthropic_key.txt')
            if os.path.exists(key_file):
                with open(key_file, 'r') as f:
                    self.api_key = f.read().strip()
            else:
                raise ValueError("Claude API 키가 설정되지 않았습니다")
                
        # 세션 생성
        self.session = aiohttp.ClientSession()
        self.logger.info("✅ Claude API 클라이언트 초기화 완료")
        
    async def generate_async(self, prompt: str, **kwargs) -> LLMResponse:
        """비동기 텍스트 생성"""
        
        if not self.session:
            await self.initialize()
            
        headers = {
            'anthropic-version': '2023-06-01',
            'x-api-key': self.api_key,
            'content-type': 'application/json'
        }
        
        data = {
            'model': self.model,
            'messages': [
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'max_tokens': kwargs.get('max_tokens', 4096),
            'temperature': kwargs.get('temperature', 0.7)
        }
        
        try:
            async with self.session.post(
                self.api_url,
                headers=headers,
                json=data
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    
                    # Claude API 응답 구조에서 텍스트 추출
                    generated_text = result['content'][0]['text']
                    
                    return LLMResponse(
                        generated_text=generated_text,
                        success=True,
                        metadata={
                            'model': self.model,
                            'usage': result.get('usage', {})
                        }
                    )
                else:
                    error_text = await response.text()
                    self.logger.error(f"Claude API 오류: {response.status} - {error_text}")
                    return LLMResponse(
                        generated_text="",
                        success=False,
                        error=f"API 오류: {response.status}"
                    )
                    
        except Exception as e:
            self.logger.error(f"Claude API 호출 실패: {e}")
            return LLMResponse(
                generated_text="",
                success=False,
                error=str(e)
            )
            
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """동기 텍스트 생성 (비동기 래퍼)"""
        return asyncio.run(self.generate_async(prompt, **kwargs))
        
    async def cleanup(self):
        """리소스 정리"""
        if self.session:
            await self.session.close()
            self.session = None
            
    def __del__(self):
        """소멸자"""
        if self.session:
            asyncio.create_task(self.cleanup())