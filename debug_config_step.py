#!/usr/bin/env python3
"""
config.py import를 단계별로 테스트
"""
import signal
import sys
import logging

logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def timeout_handler(signum, frame):
    logger.critical(f"⏰ TIMEOUT! Frame: {frame.f_code.co_filename}:{frame.f_lineno} in {frame.f_code.co_name}")
    import traceback
    traceback.print_stack(frame)
    sys.exit(1)

def test_step(step_name, code):
    logger.critical(f"🔍 Testing: {step_name}")
    signal.alarm(15)  # 15초 타임아웃
    try:
        exec(code)
        logger.critical(f"✅ SUCCESS: {step_name}")
    except Exception as e:
        logger.critical(f"❌ FAILED: {step_name} - {e}")
    finally:
        signal.alarm(0)

def main():
    signal.signal(signal.SIGALRM, timeout_handler)
    
    # Step 1: 기본 imports
    test_step("기본 imports", """
import os
import platform
import time
import asyncio
import logging
import datetime
""")
    
    # Step 2: 기본 변수들
    test_step("BASE_DIR 설정", """
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
""")
    
    # Step 3: 디렉토리 생성 부분
    test_step("디렉토리 생성", """
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
""")
    
    # Step 4: ADVANCED_CONFIG 기본 부분
    test_step("ADVANCED_CONFIG 기본", """
ADVANCED_CONFIG = {
    'model_sizes': {
        'backbone': 300_000_000,
        'total_system': 800_000_000,
    }
}
""")
    
    # Step 5: torch import 부분
    test_step("torch import만", """
import torch
""")
    
    logger.critical("🏁 All tests completed!")

if __name__ == "__main__":
    main()