#!/usr/bin/env python3
"""
Claude API 통합 테스트
Anthropic Claude API의 동작 확인 및 특이사항 검증
"""

import asyncio
import sys
import os
import logging
from pathlib import Path

# 경로 설정
sys.path.append('/mnt/c/large_project/linux_red_heart')

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ClaudeAPITest')

async def test_claude_api():
    """Claude API 테스트"""
    
    try:
        # 1. API 매니저 임포트 및 초기화
        logger.info("=" * 50)
        logger.info("🧪 Claude API 테스트 시작")
        logger.info("=" * 50)
        
        from api_key_manager.api_manager import get_api_manager
        api_manager = get_api_manager()
        
        # 2. Claude 설정 확인
        logger.info("\n📋 Claude API 설정 확인...")
        config = api_manager.get_config('claude')
        
        if not config:
            logger.error("❌ Claude API 설정을 찾을 수 없음")
            return False
            
        logger.info(f"✅ API 설정 로드 완료")
        logger.info(f"   모델: {config.model}")
        logger.info(f"   Base URL: {config.base_url}")
        logger.info(f"   Max Tokens: {config.max_tokens}")
        
        # 3. 클라이언트 생성 확인
        logger.info("\n🔧 Claude 클라이언트 생성 중...")
        client = api_manager.get_client('claude')
        
        if not client:
            logger.error("❌ Claude 클라이언트 생성 실패")
            
            # anthropic 패키지 확인
            try:
                import anthropic
                logger.info("✅ anthropic 패키지 설치됨")
                logger.info(f"   버전: {anthropic.__version__}")
            except ImportError as e:
                logger.error(f"❌ anthropic 패키지 미설치: {e}")
                logger.info("💡 설치 명령: pip install anthropic")
                return False
        else:
            logger.info("✅ Claude 클라이언트 생성 성공")
        
        # 4. 실제 API 호출 테스트
        logger.info("\n🚀 Claude API 호출 테스트...")
        test_prompt = "AI 윤리의 핵심 원칙을 한 문장으로 설명해주세요."
        
        logger.info(f"📝 테스트 프롬프트: {test_prompt}")
        
        # API 호출
        result = await api_manager.call_api(
            'claude',
            test_prompt,
            max_tokens=100,
            temperature=0.7
        )
        
        if result:
            logger.info("✅ API 호출 성공!")
            logger.info(f"📤 응답 길이: {len(result)} 문자")
            logger.info(f"📤 응답 내용:\n{'-'*40}\n{result}\n{'-'*40}")
            
            # 5. 비동기 처리 특성 확인
            logger.info("\n⚡ 비동기 처리 검증...")
            
            # 동시 호출 테스트
            prompts = [
                "AI의 장점을 한 단어로",
                "AI의 단점을 한 단어로",
                "AI의 미래를 한 단어로"
            ]
            
            tasks = []
            for prompt in prompts:
                task = api_manager.call_api('claude', prompt, max_tokens=20)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, (prompt, res) in enumerate(zip(prompts, results)):
                if isinstance(res, Exception):
                    logger.error(f"   [{i+1}] ❌ '{prompt}': {res}")
                else:
                    logger.info(f"   [{i+1}] ✅ '{prompt}': {res}")
            
            logger.info("\n✅ Claude API 통합 테스트 완료!")
            return True
            
        else:
            logger.error("❌ API 호출 실패")
            
            # 상세 에러 진단
            logger.info("\n🔍 에러 진단 시작...")
            
            # API 키 확인
            if config.api_key.startswith("sk-ant"):
                logger.info("✅ API 키 형식 올바름")
            else:
                logger.error("❌ API 키 형식 오류")
            
            # 네트워크 연결 확인
            import socket
            try:
                socket.create_connection(("api.anthropic.com", 443), timeout=5)
                logger.info("✅ 네트워크 연결 정상")
            except:
                logger.error("❌ 네트워크 연결 실패")
            
            return False
            
    except Exception as e:
        logger.error(f"❌ 테스트 중 예외 발생: {e}", exc_info=True)
        return False

async def test_llm_engine_integration():
    """LLM 엔진 통합 테스트"""
    
    logger.info("\n" + "=" * 50)
    logger.info("🎯 LLM 엔진 통합 테스트")
    logger.info("=" * 50)
    
    try:
        # AdvancedLLMEngine 테스트
        from llm_module.advanced_llm_engine import AdvancedLLMEngine, LLMRequest, TaskComplexity
        
        logger.info("🔧 AdvancedLLMEngine 초기화 (Claude API)...")
        llm_engine = AdvancedLLMEngine(use_api='claude')
        
        # 엔진 초기화 확인
        await llm_engine.initialize()
        logger.info("✅ LLM 엔진 초기화 완료")
        
        # 요청 생성
        request = LLMRequest(
            prompt="AI 시스템의 윤리적 의사결정 원칙을 설명하세요.",
            task_type="ethical_analysis",
            complexity=TaskComplexity.COMPLEX,
            max_tokens=200,
            temperature=0.7
        )
        
        logger.info(f"📝 LLMRequest 생성: {request.task_type}")
        
        # 비동기 생성 테스트
        response = await llm_engine.generate_async(request)
        
        if response and response.success:
            logger.info("✅ LLM 엔진 응답 성공!")
            logger.info(f"   신뢰도: {response.confidence:.2f}")
            logger.info(f"   처리 시간: {response.processing_time:.2f}초")
            logger.info(f"   토큰 수: {response.token_count}")
            logger.info(f"   모델: {response.model_used}")
            logger.info(f"   응답:\n{'-'*40}\n{response.generated_text[:200]}...\n{'-'*40}")
        else:
            logger.error(f"❌ LLM 엔진 응답 실패: {response.error_message if response else 'No response'}")
            
    except Exception as e:
        logger.error(f"❌ LLM 엔진 테스트 실패: {e}", exc_info=True)

def main():
    """메인 실행"""
    logger.info("🚀 Claude API 통합 테스트 시작\n")
    
    # 이벤트 루프 실행
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # 기본 API 테스트
        success = loop.run_until_complete(test_claude_api())
        
        if success:
            # LLM 엔진 통합 테스트
            loop.run_until_complete(test_llm_engine_integration())
            
        logger.info("\n" + "=" * 50)
        logger.info("🎉 모든 테스트 완료!")
        logger.info("=" * 50)
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ 사용자에 의해 중단됨")
    finally:
        loop.close()

if __name__ == "__main__":
    main()