#!/usr/bin/env python3
"""
Hanging 디버깅을 위한 최소한의 테스트 스크립트
"""
import asyncio
import logging
import sys
import signal
import time
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.CRITICAL,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def timeout_handler(signum, frame):
    logger.critical("⏰ TIMEOUT! Hanging detected!")
    logger.critical(f"🔍 Current frame: {frame}")
    logger.critical(f"🔍 Frame filename: {frame.f_code.co_filename}")
    logger.critical(f"🔍 Frame line: {frame.f_lineno}")
    logger.critical(f"🔍 Frame function: {frame.f_code.co_name}")
    sys.exit(1)

async def minimal_test():
    """최소한의 테스트로 어디서 hanging이 발생하는지 확인"""
    logger.critical("🔍 Step 1: 테스트 시작")
    
    logger.critical("🔍 Step 2: UnifiedSystemOrchestrator import 시도...")
    try:
        from unified_system_orchestrator import UnifiedSystemOrchestrator
        logger.critical("🔍 Step 2: UnifiedSystemOrchestrator import 성공!")
    except Exception as e:
        logger.critical(f"❌ Step 2: import 실패 - {e}")
        return
    
    logger.critical("🔍 Step 3: config import 시도...")
    try:
        from config import ADVANCED_CONFIG
        logger.critical("🔍 Step 3: config import 성공!")
    except Exception as e:
        logger.critical(f"❌ Step 3: config import 실패 - {e}")
        return
    
    logger.critical("🔍 Step 4: UnifiedSystemOrchestrator 생성 시도...")
    try:
        orchestrator = UnifiedSystemOrchestrator(ADVANCED_CONFIG)
        logger.critical("🔍 Step 4: UnifiedSystemOrchestrator 생성 성공!")
    except Exception as e:
        logger.critical(f"❌ Step 4: 생성 실패 - {e}")
        return
    
    logger.critical("🔍 Step 5: initialize_system() 호출 시도...")
    try:
        # 20초 타임아웃으로 initialize_system 호출
        await asyncio.wait_for(orchestrator.initialize_system(), timeout=20.0)
        logger.critical("🔍 Step 5: initialize_system() 성공!")
    except asyncio.TimeoutError:
        logger.critical("❌ Step 5: initialize_system() 20초 타임아웃!")
        return
    except Exception as e:
        logger.critical(f"❌ Step 5: initialize_system() 실패 - {e}")
        return
    
    logger.critical("✅ 모든 단계 성공!")

def main():
    logger.critical(f"🚀 Hanging 디버깅 테스트 시작 - {datetime.now()}")
    
    # 30초 전체 타임아웃 설정
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)
    
    try:
        asyncio.run(minimal_test())
    except KeyboardInterrupt:
        logger.critical("❌ 사용자 중단")
    except Exception as e:
        logger.critical(f"❌ 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
    finally:
        signal.alarm(0)  # 타임아웃 해제
    
    logger.critical(f"🏁 테스트 종료 - {datetime.now()}")

if __name__ == "__main__":
    main()