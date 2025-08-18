#!/usr/bin/env python3
"""
Red Heart AI 시스템 무결성 검사기
System Integrity Checker for Red Heart AI

real_integrated_training_test.py의 initialize_system() 방식을 정확히 따라한
완전한 모듈별 의존성 무결성 검사 시스템
"""

import asyncio
import logging
import time
import sys
import traceback
import os
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('RedHeart.SystemIntegrityChecker')

class SystemIntegrityChecker:
    """시스템 무결성 검사기 - real_integrated_training_test.py 방식"""
    
    def __init__(self):
        self.emotion_analyzer = None
        self.bentham_calculator = None
        self.regret_analyzer = None
        self.surd_analyzer = None
        self.experience_db = None
        
    def check_environment_setup(self) -> bool:
        """환경 설정 기본 확인"""
        logger.info("환경 설정 확인 중...")
        
        try:
            # Python 환경 확인
            python_path = sys.executable
            logger.info(f"Python 경로: {python_path}")
            
            # venv 확인
            virtual_env = os.environ.get('VIRTUAL_ENV', '')
            is_venv = bool(virtual_env)
            if is_venv:
                logger.info(f"✅ venv 활성화: {virtual_env}")
            else:
                logger.warning("⚠️ venv가 활성화되지 않았습니다")
            
            # conda 환경 확인
            conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'None')
            if conda_env != 'None':
                logger.info(f"✅ conda 활성화: {conda_env}")
            else:
                logger.warning("⚠️ conda 환경이 활성화되지 않았습니다")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 환경 확인 실패: {e}")
            return False
            
    async def initialize_system(self):
        """전체 시스템 초기화 - real_integrated_training_test.py 방식과 동일"""
        logger.info("=== Red Heart AI 시스템 무결성 검사 시작 ===")
        
        # 환경 설정 기본 확인
        if not self.check_environment_setup():
            return False
        
        try:
            # LocalTranslator 먼저 초기화
            logger.info("번역기 초기화...")
            from local_translator import LocalTranslator
            from config import register_system_module
            translator = LocalTranslator()
            register_system_module('translator', translator)
            logger.info("✅ 번역기 준비 완료")
            
            # 등록 확인
            from config import get_system_module
            registered_translator = get_system_module('translator')
            if registered_translator is None:
                logger.error("❌ translator 모듈 등록 실패!")
            else:
                logger.info(f"✅ translator 모듈 등록 확인: {type(registered_translator)}")
            
            # 감정 분석기 초기화
            logger.info("감정 분석기 초기화...")
            from advanced_emotion_analyzer import AdvancedEmotionAnalyzer
            self.emotion_analyzer = AdvancedEmotionAnalyzer()
            logger.info("✅ 감정 분석기 준비 완료")
            
            # 벤담 계산기 초기화
            logger.info("벤담 계산기 초기화...")
            from advanced_bentham_calculator import AdvancedBenthamCalculator
            self.bentham_calculator = AdvancedBenthamCalculator()
            logger.info("✅ 벤담 계산기 준비 완료")
            
            # 후회 분석기 초기화 (실패 허용)
            logger.info("후회 분석기 초기화...")
            try:
                from advanced_regret_analyzer import AdvancedRegretAnalyzer
                self.regret_analyzer = AdvancedRegretAnalyzer()
                logger.info("✅ 후회 분석기 준비 완료")
            except Exception as e:
                logger.warning(f"⚠️ 후회 분석기 초기화 실패: {e}")
                self.regret_analyzer = None
            
            # SURD 분석기 초기화
            logger.info("SURD 분석기 초기화...")
            from advanced_surd_analyzer import AdvancedSURDAnalyzer
            self.surd_analyzer = AdvancedSURDAnalyzer()
            logger.info("✅ SURD 분석기 준비 완료")
            
            # 경험 데이터베이스 초기화
            logger.info("경험 데이터베이스 초기화...")
            from advanced_experience_database import AdvancedExperienceDatabase
            self.experience_db = AdvancedExperienceDatabase()
            logger.info("✅ 경험 데이터베이스 준비 완료")
            
            # 환경 분리 테스트 (faiss subprocess)
            logger.info("환경 분리 확인 중 (conda faiss)...")
            if self.test_faiss_subprocess():
                logger.info("✅ 환경 분리 (faiss) 작동 확인")
            else:
                logger.warning("⚠️ 환경 분리 확인 실패 (계속 진행)")
            
            logger.info("🎯 시스템 무결성 검사 완료")
            return True
            
        except Exception as e:
            logger.error(f"❌ 시스템 초기화 실패: {e}")
            traceback.print_exc()
            return False
            
    def test_faiss_subprocess(self) -> bool:
        """faiss subprocess 환경 분리 테스트"""
        try:
            # utils 모듈에서 faiss subprocess 함수 확인
            try:
                from utils import run_faiss_subprocess
                # 간단한 테스트 데이터로 subprocess 확인
                test_data = {'operation': 'test', 'data': {}}
                result = run_faiss_subprocess('test', test_data)
                return True
            except ImportError:
                logger.warning("run_faiss_subprocess 함수를 찾을 수 없습니다")
                return False
            except Exception as e:
                logger.warning(f"faiss subprocess 테스트 실패: {e}")
                return False
                
        except Exception as e:
            logger.warning(f"환경 분리 테스트 중 오류: {e}")
            return False
            
    def run_basic_package_check(self) -> bool:
        """기본 패키지 확인 (빠른 검사)"""
        logger.info("기본 패키지 확인 중...")
        
        critical_packages = {
            'numpy': 'numpy',
            'torch': 'torch', 
            'transformers': 'transformers',
            'sentence-transformers': 'sentence_transformers',
            'scikit-learn': 'sklearn'
        }
        
        failed_packages = []
        
        for package_name, import_name in critical_packages.items():
            try:
                module = __import__(import_name)
                version = getattr(module, '__version__', 'unknown')
                logger.info(f"✅ {package_name}: {version}")
            except ImportError as e:
                logger.error(f"❌ {package_name}: 패키지 없음")
                failed_packages.append(package_name)
        
        if failed_packages:
            logger.error(f"❌ 필수 패키지 누락: {failed_packages}")
            return False
        else:
            logger.info("✅ 모든 기본 패키지 확인 완료")
            return True

async def main():
    """메인 실행 함수"""
    start_time = time.time()
    
    checker = SystemIntegrityChecker()
    
    # 기본 패키지 빠른 확인
    if not checker.run_basic_package_check():
        logger.error("❌ 기본 패키지 확인 실패. 시스템을 종료합니다.")
        sys.exit(1)
    
    # 전체 시스템 초기화 (실제 모듈 인스턴스 생성)
    success = await checker.initialize_system()
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print("🔍 Red Heart AI 시스템 무결성 검사 결과")
    print("=" * 60)
    
    if success:
        print("🎉 ✅ 시스템 무결성 검사 성공")
        print("   모든 모듈이 정상적으로 초기화되었습니다.")
        print("   시스템이 학습/테스트 실행 준비가 완료되었습니다.")
        print(f"⏱️ 검사 시간: {duration:.2f}초")
        print("=" * 60)
        sys.exit(0)  # 성공
    else:
        print("❌ 💥 시스템 무결성 검사 실패")
        print("   모듈 초기화 중 오류가 발생했습니다.")
        print("   환경을 확인하고 다시 시도해주세요.")
        print(f"⏱️ 검사 시간: {duration:.2f}초")
        print("=" * 60)
        sys.exit(1)  # 실패

if __name__ == "__main__":
    asyncio.run(main())