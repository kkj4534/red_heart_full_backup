#!/usr/bin/env python3
"""
Import 상태 진단 스크립트
Debug Import Status Script

각 모듈의 import 상태를 개별적으로 테스트하여
LEARNING_SYSTEM_AVAILABLE이 False가 되는 원인을 파악
"""

import logging
import traceback
import sys
import os

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ImportDiagnostics')

def test_basic_modules():
    """기본 모듈들 import 테스트"""
    logger.info("=== 기본 모듈 Import 테스트 ===")
    
    basic_modules = {}
    modules_to_test = [
        ('advanced_emotion_analyzer', 'AdvancedEmotionAnalyzer'),
        ('advanced_bentham_calculator', 'AdvancedBenthamCalculator'), 
        ('advanced_regret_analyzer', 'AdvancedRegretAnalyzer'),
        ('advanced_surd_analyzer', 'AdvancedSURDAnalyzer'),
        ('advanced_experience_database', 'AdvancedExperienceDatabase'),
        ('data_models', 'EthicalSituation, EmotionData, HedonicValues')
    ]
    
    for module_name, class_names in modules_to_test:
        try:
            if module_name == 'data_models':
                from data_models import EthicalSituation, EmotionData, HedonicValues
            else:
                __import__(module_name)
            
            basic_modules[module_name] = True
            logger.info(f"✅ {module_name}: SUCCESS")
        except ImportError as e:
            basic_modules[module_name] = False
            logger.error(f"❌ {module_name}: FAILED - {e}")
            logger.error(f"   스택 트레이스: {traceback.format_exc()}")
        except Exception as e:
            basic_modules[module_name] = False
            logger.error(f"❌ {module_name}: UNEXPECTED ERROR - {e}")
            logger.error(f"   스택 트레이스: {traceback.format_exc()}")
    
    return basic_modules

def test_learning_modules():
    """학습 시스템 모듈들 import 테스트"""
    logger.info("=== 학습 시스템 모듈 Import 테스트 ===")
    
    learning_modules = {}
    modules_to_test = [
        ('advanced_learning_executor', 'AdvancedLearningExecutor, LearningConfig'),
        ('advanced_regret_learning_system', 'AdvancedRegretLearningSystem, LearningPhase'),
        ('advanced_hierarchical_emotion_system', 'AdvancedHierarchicalEmotionSystem, EmotionPhase'),
        ('integrated_system_orchestrator', 'IntegratedSystemOrchestrator, IntegrationContext'),
        ('dynamic_ethical_choice_analyzer', 'DynamicEthicalChoiceAnalyzer, EthicalDilemma')
    ]
    
    for module_name, class_names in modules_to_test:
        try:
            if module_name == 'advanced_learning_executor':
                from advanced_learning_executor import AdvancedLearningExecutor, LearningConfig
            elif module_name == 'advanced_regret_learning_system':
                from advanced_regret_learning_system import AdvancedRegretLearningSystem, LearningPhase
            elif module_name == 'advanced_hierarchical_emotion_system':
                from advanced_hierarchical_emotion_system import AdvancedHierarchicalEmotionSystem, EmotionPhase
            elif module_name == 'integrated_system_orchestrator':
                from integrated_system_orchestrator import IntegratedSystemOrchestrator, IntegrationContext
            elif module_name == 'dynamic_ethical_choice_analyzer':
                from dynamic_ethical_choice_analyzer import DynamicEthicalChoiceAnalyzer, EthicalDilemma
            
            learning_modules[module_name] = True
            logger.info(f"✅ {module_name}: SUCCESS")
        except ImportError as e:
            learning_modules[module_name] = False
            logger.error(f"❌ {module_name}: FAILED - {e}")
            logger.error(f"   스택 트레이스: {traceback.format_exc()}")
        except Exception as e:
            learning_modules[module_name] = False
            logger.error(f"❌ {module_name}: UNEXPECTED ERROR - {e}")
            logger.error(f"   스택 트레이스: {traceback.format_exc()}")
    
    return learning_modules

def test_dynamic_ethical_choice_analyzer_dependencies():
    """DynamicEthicalChoiceAnalyzer의 의존성 모듈들 개별 테스트"""
    logger.info("=== DynamicEthicalChoiceAnalyzer 의존성 테스트 ===")
    
    dependencies = [
        'advanced_emotion_analyzer',
        'advanced_bentham_calculator', 
        'advanced_regret_analyzer',
        'advanced_surd_analyzer',
        'advanced_counterfactual_reasoning',
        'advanced_rumbaugh_analyzer',
        'integrated_system_orchestrator',
        'llm_module.advanced_llm_engine'
    ]
    
    for dep in dependencies:
        try:
            if dep == 'llm_module.advanced_llm_engine':
                from llm_module.advanced_llm_engine import get_llm_engine
            else:
                __import__(dep)
            logger.info(f"✅ {dep}: SUCCESS")
        except ImportError as e:
            logger.error(f"❌ {dep}: FAILED - {e}")
            logger.error(f"   스택 트레이스: {traceback.format_exc()}")
        except Exception as e:
            logger.error(f"❌ {dep}: UNEXPECTED ERROR - {e}")
            logger.error(f"   스택 트레이스: {traceback.format_exc()}")

def main():
    """메인 진단 함수"""
    logger.info("🔍 Red Heart AI Import 진단 시작")
    logger.info("=" * 80)
    
    # 환경 정보 출력
    logger.info(f"Python 경로: {sys.executable}")
    logger.info(f"가상환경: {os.environ.get('VIRTUAL_ENV', 'None')}")
    logger.info(f"Conda 환경: {os.environ.get('CONDA_DEFAULT_ENV', 'None')}")
    logger.info(f"현재 디렉토리: {os.getcwd()}")
    
    # 기본 모듈 테스트
    basic_results = test_basic_modules()
    
    # 학습 모듈 테스트
    learning_results = test_learning_modules()
    
    # DynamicEthicalChoiceAnalyzer 의존성 테스트
    test_dynamic_ethical_choice_analyzer_dependencies()
    
    # 결과 요약
    logger.info("=" * 80)
    logger.info("🎯 Import 진단 결과 요약")
    
    basic_success = all(basic_results.values())
    learning_success = all(learning_results.values())
    
    logger.info(f"기본 모듈 상태: {basic_results}")
    logger.info(f"학습 모듈 상태: {learning_results}")
    logger.info(f"MODULES_AVAILABLE 예상 값: {basic_success}")
    logger.info(f"LEARNING_SYSTEM_AVAILABLE 예상 값: {basic_success and learning_success}")
    
    if not (basic_success and learning_success):
        logger.error("❌ Import 실패로 인해 학습 시스템이 비활성화됩니다!")
        logger.error("   이것이 learning_mode=False가 되는 근본 원인입니다.")
    else:
        logger.info("✅ 모든 모듈이 성공적으로 import되었습니다.")

if __name__ == "__main__":
    main()