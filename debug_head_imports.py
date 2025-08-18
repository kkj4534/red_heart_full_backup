#!/usr/bin/env python3
"""
head_compatibility_interface의 import들을 하나씩 테스트
"""
import signal
import sys
import logging

logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def timeout_handler(signum, frame):
    logger.critical(f"⏰ TIMEOUT! Frame: {frame.f_code.co_filename}:{frame.f_lineno} in {frame.f_code.co_name}")
    sys.exit(1)

def test_import(module_name, import_statement):
    logger.critical(f"🔍 Testing: {import_statement}")
    signal.alarm(10)  # 10초 타임아웃
    try:
        exec(import_statement)
        logger.critical(f"✅ SUCCESS: {module_name}")
    except Exception as e:
        logger.critical(f"❌ FAILED: {module_name} - {e}")
    finally:
        signal.alarm(0)

def main():
    signal.signal(signal.SIGALRM, timeout_handler)
    
    imports_to_test = [
        ("basic imports", """
import asyncio
import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Union, Tuple, Type
from dataclasses import dataclass, field
from datetime import datetime
import threading
from abc import ABC, abstractmethod
from enum import Enum
"""),
        ("config", "from config import ADVANCED_CONFIG, get_smart_device, get_gpu_memory_info, ModelPriority, get_priority_based_device"),
        ("dynamic_swap_manager", "from dynamic_swap_manager import SwapPriority, RedHeartDynamicSwapManager"),
        ("unified_red_heart_core", "from unified_red_heart_core import UnifiedRepresentation, RedHeartUnifiedBackbone, LightweightCrossAttention"),
        ("advanced_hierarchical_emotion_system", "from advanced_hierarchical_emotion_system import EnhancedEmpathyLearner"),
        ("advanced_bentham_calculator", "from advanced_bentham_calculator import FrommEnhancedBenthamCalculator"),
        ("advanced_regret_analyzer", "from advanced_regret_analyzer import GPURegretNetwork"),
        ("advanced_multi_level_semantic_analyzer", "from advanced_multi_level_semantic_analyzer import AdvancedMultiLevelSemanticAnalyzer"),
    ]
    
    for module_name, import_statement in imports_to_test:
        test_import(module_name, import_statement)
    
    logger.critical("🏁 All import tests completed!")

if __name__ == "__main__":
    main()