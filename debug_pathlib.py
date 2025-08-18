#!/usr/bin/env python3
"""
Path.absolute() hanging 디버깅
"""
import signal
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def timeout_handler(signum, frame):
    logger.critical(f"⏰ TIMEOUT! Frame: {frame.f_code.co_filename}:{frame.f_lineno} in {frame.f_code.co_name}")
    sys.exit(1)

def main():
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(10)
    
    try:
        logger.critical("🔍 Testing Path(__file__)")
        path_obj = Path(__file__)
        logger.critical(f"✅ Path(__file__) = {path_obj}")
        
        logger.critical("🔍 Testing .parent")
        parent_path = path_obj.parent
        logger.critical(f"✅ .parent = {parent_path}")
        
        logger.critical("🔍 Testing .absolute() - THIS MAY HANG!")
        absolute_path = parent_path.absolute()
        logger.critical(f"✅ .absolute() = {absolute_path}")
        
        logger.critical("🔍 Alternative: Path.cwd()")
        cwd_path = Path.cwd()
        logger.critical(f"✅ Path.cwd() = {cwd_path}")
        
        logger.critical("✅ All path operations successful!")
        
    except Exception as e:
        logger.critical(f"❌ Error: {e}")
    finally:
        signal.alarm(0)

if __name__ == "__main__":
    main()