"""
API 키 관리 및 API 클라이언트 통합 모듈
각 API 서비스별 호출 방식 통합 관리
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger('RedHeart.APIManager')

class APIProvider(Enum):
    """API 제공자"""
    GPT = "gpt"
    CLAUDE = "claude"
    PERPLEXITY = "perplexity"
    DEEPSEEK = "deepseek"
    LOCAL = "local"  # 로컬 모델 폴백

@dataclass
class APIConfig:
    """API 설정"""
    api_key: str
    model: str
    base_url: str
    max_tokens: int = 2048
    temperature: float = 0.7
    provider: APIProvider = APIProvider.GPT

class APIKeyManager:
    """API 키 관리자"""
    
    def __init__(self):
        self.config_path = Path(__file__).parent / "config.json"
        self.configs = self._load_configs()
        self.clients = {}
        logger.info(f"🔑 API 키 관리자 초기화 (설정 파일: {self.config_path})")
    
    def _load_configs(self) -> Dict[str, APIConfig]:
        """설정 파일 로드"""
        configs = {}
        
        if not self.config_path.exists():
            logger.warning(f"⚠️ API 설정 파일이 없습니다: {self.config_path}")
            return configs
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            api_keys = data.get('api_keys', {})
            for provider_name, config in api_keys.items():
                try:
                    provider = APIProvider(provider_name)
                    configs[provider] = APIConfig(
                        api_key=config['api_key'],
                        model=config['model'],
                        base_url=config['base_url'],
                        max_tokens=config.get('max_tokens', 2048),
                        temperature=config.get('temperature', 0.7),
                        provider=provider
                    )
                    
                    # API 키 유효성 체크 (마스킹)
                    if config['api_key'].startswith("YOUR_"):
                        logger.warning(f"⚠️ {provider_name} API 키가 설정되지 않음")
                    else:
                        masked_key = config['api_key'][:10] + "..." if len(config['api_key']) > 10 else "***"
                        logger.info(f"✅ {provider_name} API 키 로드됨: {masked_key}")
                        
                except Exception as e:
                    logger.error(f"❌ {provider_name} 설정 로드 실패: {e}")
                    
        except Exception as e:
            logger.error(f"❌ 설정 파일 로드 실패: {e}")
            
        return configs
    
    def get_config(self, provider: str) -> Optional[APIConfig]:
        """특정 제공자의 설정 가져오기"""
        try:
            provider_enum = APIProvider(provider.lower())
            config = self.configs.get(provider_enum)
            
            if not config:
                logger.error(f"❌ {provider} 설정을 찾을 수 없음")
                return None
            
            # API 키 체크
            if config.api_key.startswith("YOUR_"):
                logger.error(f"❌ {provider} API 키가 설정되지 않음. config.json을 확인하세요.")
                return None
                
            return config
            
        except ValueError:
            logger.error(f"❌ 알 수 없는 제공자: {provider}")
            return None
    
    def get_client(self, provider: str):
        """API 클라이언트 가져오기 (캐싱)"""
        if provider in self.clients:
            return self.clients[provider]
        
        config = self.get_config(provider)
        if not config:
            return None
        
        client = self._create_client(config)
        if client:
            self.clients[provider] = client
            
        return client
    
    def _create_client(self, config: APIConfig):
        """API 클라이언트 생성"""
        try:
            if config.provider == APIProvider.GPT:
                return self._create_openai_client(config)
            elif config.provider == APIProvider.CLAUDE:
                return self._create_anthropic_client(config)
            elif config.provider == APIProvider.PERPLEXITY:
                return self._create_perplexity_client(config)
            elif config.provider == APIProvider.DEEPSEEK:
                return self._create_deepseek_client(config)
            else:
                logger.error(f"❌ 지원되지 않는 제공자: {config.provider}")
                return None
                
        except Exception as e:
            logger.error(f"❌ 클라이언트 생성 실패 ({config.provider}): {e}")
            return None
    
    def _create_openai_client(self, config: APIConfig):
        """OpenAI 클라이언트 생성"""
        try:
            import openai
            client = openai.AsyncOpenAI(api_key=config.api_key)
            logger.info("✅ OpenAI 비동기 클라이언트 생성 완료")
            return client
        except ImportError:
            logger.error("❌ openai 패키지가 설치되지 않음. pip install openai")
            return None
    
    def _create_anthropic_client(self, config: APIConfig):
        """Anthropic Claude 클라이언트 생성"""
        try:
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=config.api_key)
            logger.info("✅ Anthropic 비동기 클라이언트 생성 완료")
            return client
        except ImportError:
            logger.error("❌ anthropic 패키지가 설치되지 않음. pip install anthropic")
            return None
    
    def _create_perplexity_client(self, config: APIConfig):
        """Perplexity 클라이언트 생성 (OpenAI 호환)"""
        try:
            import openai
            client = openai.AsyncOpenAI(
                api_key=config.api_key,
                base_url="https://api.perplexity.ai"
            )
            logger.info("✅ Perplexity 비동기 클라이언트 생성 완료")
            return client
        except ImportError:
            logger.error("❌ openai 패키지가 설치되지 않음. pip install openai")
            return None
    
    def _create_deepseek_client(self, config: APIConfig):
        """DeepSeek 클라이언트 생성 (OpenAI 호환)"""
        try:
            import openai
            client = openai.AsyncOpenAI(
                api_key=config.api_key,
                base_url="https://api.deepseek.com/v1"
            )
            logger.info("✅ DeepSeek 비동기 클라이언트 생성 완료")
            return client
        except ImportError:
            logger.error("❌ openai 패키지가 설치되지 않음. pip install openai")
            return None
    
    async def call_api(self, provider: str, prompt: str, **kwargs) -> Optional[str]:
        """통합 API 호출"""
        config = self.get_config(provider)
        if not config:
            return None
        
        client = self.get_client(provider)
        if not client:
            return None
        
        try:
            if config.provider == APIProvider.GPT:
                return await self._call_openai(client, config, prompt, **kwargs)
            elif config.provider == APIProvider.CLAUDE:
                return await self._call_anthropic(client, config, prompt, **kwargs)
            elif config.provider in [APIProvider.PERPLEXITY, APIProvider.DEEPSEEK]:
                return await self._call_openai(client, config, prompt, **kwargs)
            else:
                logger.error(f"❌ 지원되지 않는 API 호출: {config.provider}")
                return None
                
        except Exception as e:
            logger.error(f"❌ API 호출 실패 ({provider}): {e}")
            return None
    
    async def _call_openai(self, client, config: APIConfig, prompt: str, **kwargs):
        """OpenAI API 호출 (GPT, Perplexity, DeepSeek)"""
        try:
            response = await client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant specialized in ethical analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=kwargs.get('max_tokens', config.max_tokens),
                temperature=kwargs.get('temperature', config.temperature)
            )
            
            result = response.choices[0].message.content
            logger.info(f"✅ {config.provider.value} API 응답 성공: {len(result)} 문자")
            return result
            
        except Exception as e:
            logger.error(f"❌ OpenAI 형식 API 호출 실패: {e}")
            return None
    
    async def _call_anthropic(self, client, config: APIConfig, prompt: str, **kwargs):
        """Anthropic Claude API 호출"""
        try:
            response = await client.messages.create(
                model=config.model,
                max_tokens=kwargs.get('max_tokens', config.max_tokens),
                temperature=kwargs.get('temperature', config.temperature),
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            result = response.content[0].text
            logger.info(f"✅ Claude API 응답 성공: {len(result)} 문자")
            return result
            
        except Exception as e:
            logger.error(f"❌ Claude API 호출 실패: {e}")
            return None

# 싱글톤 인스턴스
_manager_instance = None

def get_api_manager() -> APIKeyManager:
    """API 관리자 싱글톤 인스턴스 가져오기"""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = APIKeyManager()
    return _manager_instance