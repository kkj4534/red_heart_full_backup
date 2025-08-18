#!/usr/bin/env python3
"""
torch와 pathlib hanging 문제 분석
"""
import signal
import sys
import logging

logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def timeout_handler(signum, frame):
    logger.critical(f"⏰ TIMEOUT! Frame: {frame.f_code.co_filename}:{frame.f_lineno} in {frame.f_code.co_name}")
    sys.exit(1)

def test_import(name, import_statement):
    logger.critical(f"🔍 Testing: {name}")
    signal.alarm(15)  # 15초 타임아웃
    try:
        exec(import_statement)
        logger.critical(f"✅ SUCCESS: {name}")
    except Exception as e:
        logger.critical(f"❌ FAILED: {name} - {e}")
    finally:
        signal.alarm(0)

def main():
    signal.signal(signal.SIGALRM, timeout_handler)
    
    # Step 1: torch 없이 기본 imports
    test_import("basic_without_torch", """
import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple, Type
from dataclasses import dataclass, field
from datetime import datetime
import threading
from abc import ABC, abstractmethod
from enum import Enum
""")
    
    # Step 2: torch만 단독으로
    test_import("torch_only", "import torch")
    
    # Step 3: torch.nn
    test_import("torch_nn", "import torch.nn as nn")
    
    # Step 4: torch.nn.functional  
    test_import("torch_functional", "import torch.nn.functional as F")
    
    logger.critical("🏁 All torch tests completed!")

if __name__ == "__main__":
    main()