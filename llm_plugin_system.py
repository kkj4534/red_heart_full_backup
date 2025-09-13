#!/usr/bin/env python3
"""
LLM Plugin System - 교체 가능한 LLM 백엔드 시스템

지원 백엔드:
- Local LLM (Dolphin Llama3 등)
- Claude API
- GPT API
- Perplexity API
- MCP (Model Context Protocol)

Red Heart 시스템과 독립적으로 작동하여 LLM만 교체 가능
"""

import asyncio
import logging
import json
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class LLMBackend(Enum):
    """LLM 백엔드 종류"""
    LOCAL = "local"
    CLAUDE = "claude"
    GPT = "gpt"
    PERPLEXITY = "perplexity"
    DEEPSEEK = "deepseek"
    MCP = "mcp"


@dataclass
class LLMRequest:
    """LLM 요청 데이터"""
    prompt: str
    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 0.9
    stop_sequences: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    stream: bool = False


@dataclass
class LLMResponse:
    """LLM 응답 데이터"""
    text: str
    tokens_used: int = 0
    latency_ms: float = 0
    backend: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class LLMPlugin(ABC):
    """LLM 플러그인 베이스 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 플러그인 설정
        """
        self.config = config
        self.name = self.__class__.__name__
        self.initialized = False
        self.stats = {
            'requests': 0,
            'tokens': 0,
            'errors': 0,
            'total_latency': 0
        }
        
    @abstractmethod
    async def initialize(self) -> bool:
        """플러그인 초기화"""
        pass
        
    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """텍스트 생성"""
        pass
        
    @abstractmethod
    async def cleanup(self):
        """리소스 정리"""
        pass
        
    def update_stats(self, response: LLMResponse):
        """통계 업데이트"""
        self.stats['requests'] += 1
        if response.error:
            self.stats['errors'] += 1
        else:
            self.stats['tokens'] += response.tokens_used
            self.stats['total_latency'] += response.latency_ms
            
    def get_stats(self) -> Dict:
        """통계 조회"""
        avg_latency = (
            self.stats['total_latency'] / self.stats['requests']
            if self.stats['requests'] > 0 else 0
        )
        return {
            **self.stats,
            'avg_latency_ms': avg_latency,
            'success_rate': 1 - (self.stats['errors'] / max(1, self.stats['requests']))
        }


class LocalLLMPlugin(LLMPlugin):
    """로컬 LLM 플러그인 (Dolphin Llama3 등)"""
    
    async def initialize(self) -> bool:
        """로컬 모델 로드"""
        try:
            logger.info(f"로컬 LLM 초기화: {self.config.get('model_path', 'default')}")
            
            # 실제 구현시 transformers 라이브러리 사용
            # from transformers import AutoModelForCausalLM, AutoTokenizer
            # self.model = AutoModelForCausalLM.from_pretrained(...)
            # self.tokenizer = AutoTokenizer.from_pretrained(...)
            
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"로컬 LLM 초기화 실패: {e}")
            return False
            
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """로컬 모델로 생성"""
        if not self.initialized:
            return LLMResponse(text="", error="Not initialized")
            
        start_time = time.time()
        
        try:
            # 실제 구현시 모델 추론
            # inputs = self.tokenizer(request.prompt, return_tensors="pt")
            # outputs = self.model.generate(**inputs, max_new_tokens=request.max_tokens)
            # text = self.tokenizer.decode(outputs[0])
            
            # 데모용 응답
            text = f"[Local LLM] Response to: {request.prompt[:50]}..."
            
            response = LLMResponse(
                text=text,
                tokens_used=len(text.split()),
                latency_ms=(time.time() - start_time) * 1000,
                backend="local"
            )
            
            self.update_stats(response)
            return response
            
        except Exception as e:
            logger.error(f"로컬 LLM 생성 오류: {e}")
            response = LLMResponse(text="", error=str(e))
            self.update_stats(response)
            return response
            
    async def cleanup(self):
        """모델 언로드"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        self.initialized = False
        logger.info("로컬 LLM 정리 완료")


class ClaudeLLMPlugin(LLMPlugin):
    """Claude API 플러그인"""
    
    async def initialize(self) -> bool:
        """Claude API 초기화"""
        try:
            api_key = self.config.get('api_key')
            if not api_key:
                raise ValueError("Claude API key not provided")
                
            # 실제 구현시 anthropic 라이브러리 사용
            # import anthropic
            # self.client = anthropic.Anthropic(api_key=api_key)
            
            self.initialized = True
            logger.info("Claude API 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"Claude API 초기화 실패: {e}")
            return False
            
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Claude API로 생성"""
        if not self.initialized:
            return LLMResponse(text="", error="Not initialized")
            
        start_time = time.time()
        
        try:
            # 실제 구현시 API 호출
            # response = self.client.messages.create(
            #     model="claude-3-opus-20240229",
            #     max_tokens=request.max_tokens,
            #     messages=[{"role": "user", "content": request.prompt}]
            # )
            # text = response.content[0].text
            
            # 데모용 응답
            text = f"[Claude] Analysis of: {request.prompt[:50]}..."
            
            response = LLMResponse(
                text=text,
                tokens_used=len(text.split()),
                latency_ms=(time.time() - start_time) * 1000,
                backend="claude",
                metadata={'model': 'claude-3-opus'}
            )
            
            self.update_stats(response)
            return response
            
        except Exception as e:
            logger.error(f"Claude API 오류: {e}")
            response = LLMResponse(text="", error=str(e))
            self.update_stats(response)
            return response
            
    async def cleanup(self):
        """API 클라이언트 정리"""
        if hasattr(self, 'client'):
            del self.client
        self.initialized = False
        logger.info("Claude API 정리 완료")


class GPTLLMPlugin(LLMPlugin):
    """GPT API 플러그인"""
    
    async def initialize(self) -> bool:
        """GPT API 초기화"""
        try:
            api_key = self.config.get('api_key')
            if not api_key:
                raise ValueError("OpenAI API key not provided")
                
            # 실제 구현시 openai 라이브러리 사용
            # import openai
            # openai.api_key = api_key
            
            self.initialized = True
            logger.info("GPT API 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"GPT API 초기화 실패: {e}")
            return False
            
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """GPT API로 생성"""
        if not self.initialized:
            return LLMResponse(text="", error="Not initialized")
            
        start_time = time.time()
        
        try:
            # 실제 구현시 API 호출
            # response = openai.ChatCompletion.create(
            #     model="gpt-4",
            #     messages=[{"role": "user", "content": request.prompt}],
            #     max_tokens=request.max_tokens,
            #     temperature=request.temperature
            # )
            # text = response.choices[0].message.content
            
            # 데모용 응답
            text = f"[GPT-4] Response: {request.prompt[:50]}..."
            
            response = LLMResponse(
                text=text,
                tokens_used=len(text.split()),
                latency_ms=(time.time() - start_time) * 1000,
                backend="gpt"
            )
            
            self.update_stats(response)
            return response
            
        except Exception as e:
            logger.error(f"GPT API 오류: {e}")
            response = LLMResponse(text="", error=str(e))
            self.update_stats(response)
            return response
            
    async def cleanup(self):
        """API 정리"""
        self.initialized = False
        logger.info("GPT API 정리 완료")


class LLMPluginManager:
    """LLM 플러그인 관리자
    
    플러그인 등록, 전환, 로드밸런싱 등 관리
    """
    
    def __init__(self):
        self.plugins: Dict[str, LLMPlugin] = {}
        self.active_plugin: Optional[str] = None
        self.fallback_order: List[str] = []
        
    async def register_plugin(self, name: str, plugin: LLMPlugin) -> bool:
        """플러그인 등록
        
        Args:
            name: 플러그인 이름
            plugin: 플러그인 인스턴스
            
        Returns:
            등록 성공 여부
        """
        try:
            # 초기화
            success = await plugin.initialize()
            if not success:
                logger.error(f"플러그인 초기화 실패: {name}")
                return False
                
            self.plugins[name] = plugin
            
            # 첫 플러그인이면 활성화
            if self.active_plugin is None:
                self.active_plugin = name
                
            logger.info(f"플러그인 등록: {name}")
            return True
            
        except Exception as e:
            logger.error(f"플러그인 등록 오류 ({name}): {e}")
            return False
            
    def set_active_plugin(self, name: str) -> bool:
        """활성 플러그인 설정
        
        Args:
            name: 플러그인 이름
            
        Returns:
            설정 성공 여부
        """
        if name not in self.plugins:
            logger.error(f"등록되지 않은 플러그인: {name}")
            return False
            
        self.active_plugin = name
        logger.info(f"활성 플러그인 변경: {name}")
        return True
        
    def set_fallback_order(self, order: List[str]):
        """폴백 순서 설정
        
        Args:
            order: 플러그인 이름 리스트 (우선순위 순)
        """
        self.fallback_order = [
            name for name in order 
            if name in self.plugins
        ]
        logger.info(f"폴백 순서 설정: {self.fallback_order}")
        
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """텍스트 생성 (자동 폴백 지원)
        
        Args:
            request: LLM 요청
            
        Returns:
            LLM 응답
        """
        if not self.active_plugin:
            return LLMResponse(text="", error="No active plugin")
            
        # 활성 플러그인 시도
        plugin = self.plugins[self.active_plugin]
        response = await plugin.generate(request)
        
        # 성공하면 반환
        if not response.error:
            return response
            
        # 실패시 폴백
        logger.warning(f"활성 플러그인 실패 ({self.active_plugin}), 폴백 시도...")
        
        for fallback_name in self.fallback_order:
            if fallback_name == self.active_plugin:
                continue  # 이미 시도함
                
            plugin = self.plugins.get(fallback_name)
            if plugin:
                logger.info(f"폴백 시도: {fallback_name}")
                response = await plugin.generate(request)
                if not response.error:
                    return response
                    
        # 모든 플러그인 실패
        return LLMResponse(
            text="",
            error="All plugins failed",
            metadata={'tried': [self.active_plugin] + self.fallback_order}
        )
        
    async def generate_parallel(self, request: LLMRequest, 
                              plugins: Optional[List[str]] = None) -> Dict[str, LLMResponse]:
        """병렬 생성 (여러 플러그인 동시 실행)
        
        Args:
            request: LLM 요청
            plugins: 사용할 플러그인 리스트 (None이면 전체)
            
        Returns:
            플러그인별 응답 딕셔너리
        """
        if plugins is None:
            plugins = list(self.plugins.keys())
            
        tasks = []
        for name in plugins:
            if name in self.plugins:
                plugin = self.plugins[name]
                tasks.append((name, plugin.generate(request)))
                
        # 병렬 실행
        results = {}
        for name, task in tasks:
            try:
                results[name] = await task
            except Exception as e:
                results[name] = LLMResponse(text="", error=str(e))
                
        return results
        
    def get_stats(self) -> Dict[str, Dict]:
        """모든 플러그인 통계 조회"""
        return {
            name: plugin.get_stats()
            for name, plugin in self.plugins.items()
        }
        
    async def cleanup(self):
        """모든 플러그인 정리"""
        for name, plugin in self.plugins.items():
            try:
                await plugin.cleanup()
                logger.info(f"플러그인 정리: {name}")
            except Exception as e:
                logger.error(f"플러그인 정리 오류 ({name}): {e}")
                
        self.plugins.clear()
        self.active_plugin = None


# 플러그인 팩토리
def create_plugin(backend: LLMBackend, config: Dict[str, Any]) -> Optional[LLMPlugin]:
    """플러그인 생성 팩토리
    
    Args:
        backend: 백엔드 종류
        config: 플러그인 설정
        
    Returns:
        플러그인 인스턴스
    """
    plugin_map = {
        LLMBackend.LOCAL: LocalLLMPlugin,
        LLMBackend.CLAUDE: ClaudeLLMPlugin,
        LLMBackend.GPT: GPTLLMPlugin,
        # 추가 백엔드는 여기에
    }
    
    plugin_class = plugin_map.get(backend)
    if plugin_class:
        return plugin_class(config)
        
    logger.error(f"지원하지 않는 백엔드: {backend}")
    return None


# 테스트 코드
if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(level=logging.INFO)
    
    async def test_plugin_system():
        """플러그인 시스템 테스트"""
        manager = LLMPluginManager()
        
        # 플러그인 생성 및 등록
        local_plugin = create_plugin(LLMBackend.LOCAL, {'model_path': 'test'})
        claude_plugin = create_plugin(LLMBackend.CLAUDE, {'api_key': 'test'})
        gpt_plugin = create_plugin(LLMBackend.GPT, {'api_key': 'test'})
        
        await manager.register_plugin("local", local_plugin)
        await manager.register_plugin("claude", claude_plugin)
        await manager.register_plugin("gpt", gpt_plugin)
        
        # 폴백 순서 설정
        manager.set_fallback_order(["claude", "gpt", "local"])
        
        # 단일 생성 테스트
        request = LLMRequest(
            prompt="What is the meaning of life?",
            max_tokens=100
        )
        
        response = await manager.generate(request)
        print(f"응답: {response.text}")
        print(f"백엔드: {response.backend}")
        print(f"지연시간: {response.latency_ms:.2f}ms")
        
        # 병렬 생성 테스트
        print("\n병렬 생성 테스트...")
        parallel_results = await manager.generate_parallel(request)
        for name, resp in parallel_results.items():
            print(f"{name}: {resp.text[:50]}...")
            
        # 통계 출력
        print("\n통계:")
        stats = manager.get_stats()
        for name, stat in stats.items():
            print(f"{name}: {stat}")
            
        # 정리
        await manager.cleanup()
        
    asyncio.run(test_plugin_system())