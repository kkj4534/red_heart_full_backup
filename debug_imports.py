#!/usr/bin/env python3
"""
Import별 hanging 디버깅
"""
import signal
import sys
import logging

logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def timeout_handler(signum, frame):
    logger.critical(f"⏰ TIMEOUT during import! Frame: {frame.f_code.co_filename}:{frame.f_lineno} in {frame.f_code.co_name}")
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
        ("config", "from config import ADVANCED_CONFIG, get_smart_device, get_gpu_memory_info"),
        ("head_compatibility_interface", "from head_compatibility_interface import HeadType, HeadProcessingResult, HeadCompatibilityManager"),
        ("unified_red_heart_core", "from unified_red_heart_core import RedHeartUnifiedBackbone"),
        ("dynamic_swap_manager", "from dynamic_swap_manager import RedHeartDynamicSwapManager"),
        ("intelligent_synergy_system", "from intelligent_synergy_system import IntelligentSynergySystem"),
        ("unified_learning_system", "from unified_learning_system import UnifiedLearningSystem, TrainingMetrics"),
        ("advanced_usage_pattern_analyzer", "from advanced_usage_pattern_analyzer import AdvancedUsagePatternAnalyzer"),
    ]
    
    for module_name, import_statement in imports_to_test:
        test_import(module_name, import_statement)
    
    logger.critical("🏁 All imports tested!")

if __name__ == "__main__":
    main()