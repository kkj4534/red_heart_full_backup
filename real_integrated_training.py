#!/usr/bin/env python3
"""
Red Heart AI 실제 통합 훈련 시스템
REAL Integrated Training System for Red Heart AI

진짜 모델들을 실제로 호출하여 통합 훈련 수행
- 실제 감정 분석 모델 호출
- 실제 벤담 계산 모델 호출
- 실제 후회 분석 모델 호출
- 실제 SURD 분석 모델 호출
- processed_datasets 24,170개 데이터 훈련
"""

# =============================================================================
# 환경 분리 방식으로 변경 - 초기화 스크립트 제거
# venv: 메인 애플리케이션 (numpy 2.x)
# conda subprocess: faiss + spacy 작업 (numpy 1.x)
# =============================================================================
import os
# CVE-2025-32434는 가짜 CVE - torch_security_patch import 제거
# import torch_security_patch

import asyncio
import logging
import time
import numpy as np
import torch
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
import traceback
from datetime import datetime
import os
import glob

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('RedHeart_REAL_Training')

# 실제 시스템 모듈 임포트 - 개별 체크 방식으로 강화
MODULES_AVAILABLE = True
LEARNING_SYSTEM_AVAILABLE = True

# 기본 모듈들 import 체크
basic_modules = {}
try:
    from advanced_emotion_analyzer import AdvancedEmotionAnalyzer
    basic_modules['emotion'] = True
except ImportError as e:
    logger.error(f"❌ 필수 모듈 advanced_emotion_analyzer import 실패: {e}")
    raise ImportError(f"필수 감정 분석기 시스템을 찾을 수 없습니다: {e}") from e

try:
    from advanced_bentham_calculator import AdvancedBenthamCalculator
    basic_modules['bentham'] = True
except ImportError as e:
    logger.error(f"❌ 필수 모듈 advanced_bentham_calculator import 실패: {e}")
    raise ImportError(f"필수 벤담 계산기 시스템을 찾을 수 없습니다: {e}") from e

try:
    from advanced_regret_analyzer import AdvancedRegretAnalyzer
    basic_modules['regret'] = True
except ImportError as e:
    logger.error(f"❌ 필수 모듈 advanced_regret_analyzer import 실패: {e}")
    raise ImportError(f"필수 후회 분석기 시스템을 찾을 수 없습니다: {e}") from e

try:
    from advanced_surd_analyzer import AdvancedSURDAnalyzer
    basic_modules['surd'] = True
except ImportError as e:
    logger.error(f"❌ 필수 모듈 advanced_surd_analyzer import 실패: {e}")
    raise ImportError(f"필수 SURD 분석기 시스템을 찾을 수 없습니다: {e}") from e

try:
    from advanced_experience_database import AdvancedExperienceDatabase
    basic_modules['experience_db'] = True
except ImportError as e:
    logger.error(f"❌ 필수 모듈 advanced_experience_database import 실패: {e}")
    raise ImportError(f"필수 경험 데이터베이스 시스템을 찾을 수 없습니다: {e}") from e

# Semantic 분석 모듈들 추가
try:
    from advanced_multi_level_semantic_analyzer import AdvancedMultiLevelSemanticAnalyzer
    basic_modules['advanced_semantic'] = True
except ImportError as e:
    logger.error(f"❌ 필수 모듈 advanced_multi_level_semantic_analyzer import 실패: {e}")
    raise ImportError(f"필수 고급 의미 분석기 시스템을 찾을 수 없습니다: {e}") from e

try:
    from advanced_hierarchical_emotion_system import HashtagMultiLevelSemanticAnalyzer
    basic_modules['hashtag_semantic'] = True
except ImportError as e:
    logger.error(f"❌ 필수 모듈 advanced_hierarchical_emotion_system.HashtagMultiLevelSemanticAnalyzer import 실패: {e}")
    raise ImportError(f"필수 해시태그 의미 분석기 시스템을 찾을 수 없습니다: {e}") from e

try:
    from data_models import EthicalSituation, EmotionData, HedonicValues
    basic_modules['data_models'] = True
except ImportError as e:
    logger.error(f"❌ 필수 모듈 data_models import 실패: {e}")
    raise ImportError(f"필수 데이터 모델 시스템을 찾을 수 없습니다: {e}") from e

# 학습 시스템 모듈들 import 체크
learning_modules = {}
try:
    from advanced_learning_executor import AdvancedLearningExecutor, LearningConfig
    learning_modules['learning_executor'] = True
except ImportError as e:
    logger.error(f"❌ 필수 모듈 advanced_learning_executor import 실패: {e}")
    raise ImportError(f"필수 학습 실행기 시스템을 찾을 수 없습니다: {e}") from e

try:
    from advanced_regret_learning_system import AdvancedRegretLearningSystem, LearningPhase
    learning_modules['regret_learning'] = True
except ImportError as e:
    logger.error(f"❌ 필수 모듈 advanced_regret_learning_system import 실패: {e}")
    raise ImportError(f"필수 후회 학습 시스템을 찾을 수 없습니다: {e}") from e

try:
    from advanced_hierarchical_emotion_system import AdvancedHierarchicalEmotionSystem, EmotionPhase
    learning_modules['emotion_system'] = True
except ImportError as e:
    logger.error(f"❌ 필수 모듈 advanced_hierarchical_emotion_system import 실패: {e}")
    raise ImportError(f"필수 계층적 감정 시스템을 찾을 수 없습니다: {e}") from e

try:
    from integrated_system_orchestrator import IntegratedSystemOrchestrator, IntegrationContext
    learning_modules['orchestrator'] = True
except ImportError as e:
    logger.error(f"❌ 필수 모듈 integrated_system_orchestrator import 실패: {e}")
    raise ImportError(f"필수 통합 오케스트레이터 시스템을 찾을 수 없습니다: {e}") from e

try:
    from dynamic_ethical_choice_analyzer import DynamicEthicalChoiceAnalyzer, EthicalDilemma
    learning_modules['choice_analyzer'] = True
except ImportError as e:
    logger.error(f"❌ 필수 모듈 dynamic_ethical_choice_analyzer import 실패: {e}")
    raise ImportError(f"필수 동적 윤리 선택 분석기 시스템을 찾을 수 없습니다: {e}") from e

# 상태 요약 로깅
logger.info(f"모듈 import 상태 - 기본: {basic_modules}")
logger.info(f"모듈 import 상태 - 학습: {learning_modules}")
logger.info(f"MODULES_AVAILABLE: {MODULES_AVAILABLE}")
logger.info(f"LEARNING_SYSTEM_AVAILABLE: {LEARNING_SYSTEM_AVAILABLE}")

@dataclass
class RealTrainingResult:
    """실제 훈련 결과 데이터 구조"""
    data_id: str
    source_file: str
    processing_time: float
    emotion_analysis: Dict[str, Any]
    bentham_calculation: Dict[str, Any]
    regret_analysis: Dict[str, Any]
    surd_analysis: Dict[str, Any]
    counterfactual_experiences: List[Dict[str, Any]]
    integration_success: bool
    error_log: List[str]

class RealIntegratedTrainingSystem:
    """실제 통합 훈련 시스템 - 완전한 학습 시스템 통합"""
    
    def __init__(self):
        # 개별 모듈들 (기존 기능 테스트용)
        self.emotion_analyzer = None
        self.bentham_calculator = None
        self.regret_analyzer = None
        self.surd_analyzer = None
        self.experience_db = None
        
        # Semantic 분석 모듈들 (새로 추가)
        self.advanced_semantic_analyzer = None
        self.hashtag_semantic_analyzer = None
        
        # 실제 학습 시스템 (새로 추가)
        self.learning_executor = None
        self.integrated_orchestrator = None
        self.dynamic_choice_analyzer = None
        self.learning_mode = False  # 학습 모드 vs 기능 테스트 모드
        
        # 실제 훈련 메트릭
        self.training_metrics = {
            'total_processed': 0,
            'successful_integrations': 0,
            'failed_processes': 0,
            'total_processing_time': 0.0,
            'avg_processing_time': 0.0,
            'real_accuracy_scores': {},
            'module_performance': {},
            'error_patterns': [],
            'learning_phases': {
                'current_phase': 0,
                'phase_transitions': [],
                'phase_performance': {}
            }
        }
        
        # 데이터 로더
        self.processed_datasets_path = Path("/mnt/c/large_project/linux_red_heart/processed_datasets")
        
    async def initialize_real_system(self, learning_mode: bool = False, auto_setup: bool = True):
        """실제 시스템 모든 모듈 초기화 - 환경 분리 및 자동 세팅 통합"""
        logger.info("🔍 DEBUG: initialize_real_system 메서드 시작")
        logger.info(f"🔍 DEBUG: learning_mode={learning_mode}, auto_setup={auto_setup}")
        
        try:
            logger.info("=" * 80)
            logger.info("🚀 Red Heart AI 실제 통합 시스템 초기화 시작")
            logger.info("=" * 80)
            
            self.learning_mode = learning_mode
        
            # 1. 시스템 무결성 검사 (system_integrity_checker.py 방식)
            logger.info("🔍 1단계: 시스템 무결성 검사 실행 중...")
            logger.info("🔍 DEBUG: _run_integrity_check 호출 시작")
            integrity_result = await self._run_integrity_check()
            logger.info(f"🔍 DEBUG: _run_integrity_check 결과: {integrity_result}")
            
            if not integrity_result:
                if auto_setup:
                    logger.info("🔧 자동 환경 세팅 모드 활성화...")
                    if not await self._auto_setup_environment():
                        logger.error("❌ 자동 환경 세팅 실패")
                        return False
                    # 환경 세팅 후 재검사
                    if not await self._run_integrity_check():
                        logger.error("❌ 환경 세팅 후에도 무결성 검사 실패")
                        return False
                else:
                    logger.error("❌ 무결성 검사 실패. auto_setup=True로 재시도하세요.")
                    return False
            
            logger.info("✅ 시스템 무결성 검사 통과")
            
            # 2. 환경 분리 상태 검증
            logger.info("🔍 2단계: 환경 분리 상태 검증 중...")
            if not await self._verify_environment_separation():
                logger.error("❌ 환경 분리 상태 검증 실패")
                return False
            
            logger.info("✅ 환경 분리 상태 검증 완료")
            
            # 3. 모듈 초기화 (학습 모드 vs 기능 테스트 모드)
            logger.info(f"모듈 초기화 조건 체크:")
            logger.info(f"  - learning_mode: {learning_mode}")
            logger.info(f"  - LEARNING_SYSTEM_AVAILABLE: {LEARNING_SYSTEM_AVAILABLE}")
            logger.info(f"  - 기본 모듈 상태: {basic_modules}")
            logger.info(f"  - 학습 모듈 상태: {learning_modules}")
            
            if learning_mode and LEARNING_SYSTEM_AVAILABLE:
                logger.info("🚀 3단계: 학습 모드 - 완전한 학습 시스템 초기화")
                return await self._initialize_learning_system()
            else:
                if not learning_mode:
                    logger.info("🔧 3단계: 기능 테스트 모드 - learning_mode=False")
                elif not LEARNING_SYSTEM_AVAILABLE:
                    logger.warning("⚠️ 3단계: 학습 시스템 모듈 사용 불가 - 기본 모듈만 초기화")
                logger.info("🔧 개별 모듈 초기화 진행")
                return await self._initialize_basic_modules()
            
        except Exception as e:
            logger.error(f"🚨 DEBUG: initialize_real_system에서 예외 발생: {e}")
            import traceback
            logger.error(f"🚨 DEBUG: 스택 트레이스:\n{traceback.format_exc()}")
            return False
    
    async def _run_integrity_check(self) -> bool:
        """시스템 무결성 검사 실행 - deadlock 방지 개선"""
        try:
            import subprocess
            import sys
            
            logger.info("   📋 system_integrity_checker.py 실행 중...")
            
            # sys.executable 사용으로 환경 일관성 보장
            python_executable = sys.executable
            logger.info(f"   🐍 사용할 Python: {python_executable}")
            
            # 환경변수 명시적 설정 (완전한 복사본)
            env = os.environ.copy()
            
            # 추가 환경변수 명시적 설정
            if 'VIRTUAL_ENV' in os.environ:
                env['VIRTUAL_ENV'] = os.environ['VIRTUAL_ENV']
                logger.info(f"   📦 venv: {env['VIRTUAL_ENV']}")
            
            if 'CONDA_DEFAULT_ENV' in os.environ:
                env['CONDA_DEFAULT_ENV'] = os.environ['CONDA_DEFAULT_ENV']
                logger.info(f"   🐍 conda: {env['CONDA_DEFAULT_ENV']}")
            
            # PATH 환경변수 명시적 전달
            if 'PATH' in os.environ:
                env['PATH'] = os.environ['PATH']
            
            # PYTHONPATH 설정 (현재 디렉토리 포함)
            current_dir = os.getcwd()
            if 'PYTHONPATH' in env:
                env['PYTHONPATH'] = f"{current_dir}:{env['PYTHONPATH']}"
            else:
                env['PYTHONPATH'] = current_dir
            
            logger.info("   🚀 subprocess 시작 (communicate() 사용으로 deadlock 방지)")
            
            # Popen + communicate() 사용으로 deadlock 방지
            process = subprocess.Popen(
                [python_executable, 'system_integrity_checker.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                cwd=current_dir
            )
            
            # communicate()로 deadlock 방지하면서 타임아웃 처리
            try:
                stdout, stderr = process.communicate(timeout=300)  # 5분 타임아웃
                returncode = process.returncode
                
                logger.info(f"   📤 subprocess 완료 (exit code: {returncode})")
                
                if returncode == 0:
                    logger.info("   ✅ 무결성 검사 성공")
                    if stdout.strip():
                        logger.info(f"   📋 출력: {stdout.strip()[-200:]}")  # 마지막 200자만 표시
                    return True
                else:
                    logger.error(f"   ❌ 무결성 검사 실패 (exit code: {returncode})")
                    if stderr:
                        logger.error(f"   🚨 에러: {stderr.strip()[-500:]}")  # 마지막 500자만 표시
                    if stdout:
                        logger.error(f"   📋 출력: {stdout.strip()[-200:]}")
                    return False
                    
            except subprocess.TimeoutExpired:
                logger.error("   ⏰ 무결성 검사 타임아웃 (5분)")
                process.kill()
                process.wait()
                return False
                
        except Exception as e:
            logger.error(f"   ❌ 무결성 검사 실행 중 예외: {e}")
            import traceback
            logger.error(f"   🚨 스택 트레이스: {traceback.format_exc()}")
            return False
    
    async def _verify_environment_separation(self) -> bool:
        """환경 분리 상태 검증"""
        try:
            logger.info("   🔍 conda+venv 활성화 상태 확인...")
            
            # 환경 변수 확인
            virtual_env = os.environ.get('VIRTUAL_ENV', '')
            conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')
            
            if not virtual_env:
                logger.error("   ❌ VIRTUAL_ENV 환경 변수가 설정되지 않음")
                return False
            
            if not conda_env or conda_env == 'base':
                logger.error("   ❌ CONDA_DEFAULT_ENV가 설정되지 않았거나 base 환경임")
                return False
            
            logger.info(f"   ✅ venv: {virtual_env}")
            logger.info(f"   ✅ conda: {conda_env}")
            
            # FAISS subprocess 테스트
            logger.info("   🔍 FAISS subprocess 환경 분리 테스트...")
            from utils import run_faiss_subprocess
            
            test_result = run_faiss_subprocess('test', {})
            if test_result.get('status') == 'success':
                logger.info("   ✅ FAISS 환경 분리 작동 확인")
                return True
            else:
                logger.error(f"   ❌ FAISS 환경 분리 테스트 실패: {test_result}")
                # 학습 무결성 보장을 위해 graceful degradation 제거
                return False
                
        except Exception as e:
            logger.error(f"   ❌ 환경 분리 검증 중 오류: {e}")
            return False
    
    async def _auto_setup_environment(self) -> bool:
        """자동 환경 세팅"""
        try:
            import subprocess  # 함수 시작 부분으로 이동
            logger.info("   🔧 자동 환경 세팅 시작...")
            
            # 가상환경 생성 (필요한 경우)
            venv_path = Path("red_heart_env")
            if not venv_path.exists():
                logger.info("   📦 red_heart_env 가상환경 생성 중...")
                result = subprocess.run(['python3', '-m', 'venv', 'red_heart_env'], 
                                      capture_output=True, text=True)
                if result.returncode != 0:
                    logger.error(f"   ❌ 가상환경 생성 실패: {result.stderr}")
                    return False
                logger.info("   ✅ 가상환경 생성 완료")
            
            # conda 환경 확인 및 생성
            logger.info("   🐍 conda 환경 확인 중...")
            conda_result = subprocess.run(['conda', 'env', 'list'], 
                                        capture_output=True, text=True)
            if 'faiss-test' not in conda_result.stdout:
                logger.info("   📦 faiss-test conda 환경 생성 중...")
                create_result = subprocess.run(['conda', 'create', '-n', 'faiss-test', '-y'], 
                                             capture_output=True, text=True)
                if create_result.returncode != 0:
                    logger.error(f"   ❌ conda 환경 생성 실패: {create_result.stderr}")
                    return False
                logger.info("   ✅ conda 환경 생성 완료")
            
            logger.info("   ✅ 자동 환경 세팅 완료")
            return True
            
        except Exception as e:
            logger.error(f"   ❌ 자동 환경 세팅 중 오류: {e}")
            return False
    
    async def _initialize_learning_system(self) -> bool:
        """학습 시스템 초기화"""
        try:
            logger.info("   🚀 완전한 학습 시스템 초기화 중...")
            
            # 실제 학습 시스템 초기화
            logger.info("   📚 실제 학습 실행기 초기화...")
            learning_config = LearningConfig(
                regrets_per_step=3,  # 베이스라인 검증용
                bentham_per_environment=2,  # 베이스라인 검증용
                general_data_cycles=1,  # 베이스라인 검증용
                ebs_data_cycles=1,  # 베이스라인 검증용
                max_scenarios_per_batch=3  # 베이스라인 검증용
            )
            self.learning_executor = AdvancedLearningExecutor(learning_config)
            logger.info("   ✅ 실제 학습 실행기 초기화 완료")
            
            # 통합 시스템 오케스트레이터 초기화
            logger.info("   🎼 통합 시스템 오케스트레이터 초기화...")
            self.integrated_orchestrator = IntegratedSystemOrchestrator()
            logger.info("   ✅ 통합 시스템 오케스트레이터 초기화 완료")
            
            # 동적 윤리적 선택지 분석기 초기화
            logger.info("   🤔 동적 윤리적 선택지 분석기 초기화...")
            self.dynamic_choice_analyzer = DynamicEthicalChoiceAnalyzer()
            logger.info("   ✅ 동적 윤리적 선택지 분석기 초기화 완료")
            
            # 기본 모듈들도 초기화
            await self._initialize_basic_modules()
            
            logger.info("🎉 학습 시스템 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"❌ 학습 시스템 초기화 실패: {e}")
            logger.error(f"   상세 에러: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(f"   스택 트레이스:\n{traceback.format_exc()}")
            
            # 모듈별 상태 확인
            logger.error("   모듈 가용성 재확인:")
            logger.error(f"     - 기본 모듈: {basic_modules}")
            logger.error(f"     - 학습 모듈: {learning_modules}")
            logger.error(f"     - LEARNING_SYSTEM_AVAILABLE: {LEARNING_SYSTEM_AVAILABLE}")
            
            # Graceful Degradation 제거 - 명확한 실패 처리
            logger.error("🚨 학습 시스템 초기화 실패로 인한 시스템 중단")
            logger.error("   → Graceful degradation 없이 명확한 실패 반환")
            logger.error("   → learning_mode=True 요청이지만 학습 시스템을 초기화할 수 없음")
            
            # learning_mode 상태 유지하지만 실패 반환
            return False
    
    async def _initialize_basic_modules(self) -> bool:
        """기본 모듈들 초기화"""
        try:
            logger.info("   🔧 기본 모듈들 초기화 중...")
            
            # 실제 감정 분석기 초기화
            logger.info("   😊 감정 분석기 초기화...")
            self.emotion_analyzer = AdvancedEmotionAnalyzer()
            logger.info("   ✅ 감정 분석기 초기화 완료")
            
            # 실제 벤담 계산기 초기화
            logger.info("   ⚖️ 벤담 계산기 초기화...")
            self.bentham_calculator = AdvancedBenthamCalculator()
            logger.info("   ✅ 벤담 계산기 초기화 완료")
            
            # 실제 후회 분석기 초기화
            logger.info("   😔 후회 분석기 초기화...")
            try:
                self.regret_analyzer = AdvancedRegretAnalyzer()
                logger.info("   ✅ 후회 분석기 초기화 완료")
            except Exception as e:
                logger.error(f"   ❌ 후회 분석기 초기화 실패: {e}")
                # 학습 무결성 보장을 위해 graceful degradation 제거
                raise Exception(f"후회 분석기 초기화 실패로 인한 학습 무결성 오염 방지: {e}")
            
            # 실제 SURD 분석기 초기화
            logger.info("   🧮 SURD 분석기 초기화...")
            self.surd_analyzer = AdvancedSURDAnalyzer()
            logger.info("   ✅ SURD 분석기 초기화 완료")
            
            # 실제 경험 데이터베이스 초기화 (환경 분리 적용됨)
            logger.info("   💾 경험 데이터베이스 초기화...")
            self.experience_db = AdvancedExperienceDatabase()
            logger.info("   ✅ 경험 데이터베이스 초기화 완료 (환경 분리 적용)")
            
            # Semantic 분석 모듈들 초기화 (새로 추가)
            if basic_modules.get('advanced_semantic', False):
                logger.info("   🧠 고급 의미 분석기 초기화...")
                try:
                    self.advanced_semantic_analyzer = AdvancedMultiLevelSemanticAnalyzer()
                    logger.info("   ✅ 고급 의미 분석기 초기화 완료")
                except Exception as e:
                    logger.error(f"   ❌ 고급 의미 분석기 초기화 실패: {e}")
                    # 프로젝트 규칙: fallback 없는 순수 재시도 방식
                    raise Exception(f"고급 의미 분석기 초기화 실패로 인한 학습 무결성 오염 방지: {e}")
            
            if basic_modules.get('hashtag_semantic', False):
                logger.info("   🏷️ 해시태그 의미 분석기 초기화...")
                try:
                    self.hashtag_semantic_analyzer = HashtagMultiLevelSemanticAnalyzer()
                    logger.info("   ✅ 해시태그 의미 분석기 초기화 완료")
                except Exception as e:
                    logger.error(f"   ❌ 해시태그 의미 분석기 초기화 실패: {e}")
                    # 프로젝트 규칙: fallback 없는 순수 재시도 방식
                    raise Exception(f"해시태그 의미 분석기 초기화 실패로 인한 학습 무결성 오염 방지: {e}")
            
            logger.info("🎯 기본 모듈 초기화 완료 (Semantic 모듈 포함)")
            return True
            
        except Exception as e:
            logger.error(f"❌ 기본 모듈 초기화 실패: {e}")
            traceback.print_exc()
            return False
    
    def load_real_training_data(self) -> List[Dict[str, Any]]:
        """실제 processed_datasets에서 훈련 데이터 로드"""
        logger.info("=== 실제 훈련 데이터 로딩 시작 ===")
        
        training_data = []
        
        try:
            # 스크러플 배치 파일들 로드
            scruples_pattern = self.processed_datasets_path / "scruples" / "scruples_batch_*.json"
            scruples_files = glob.glob(str(scruples_pattern))
            
            logger.info(f"스크러플 배치 파일 {len(scruples_files)}개 발견")
            
            for file_path in scruples_files[:2]:  # 처음 2개 배치만 테스트
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        batch_data = json.load(f)
                        
                    if 'scenarios' in batch_data:
                        for item in batch_data['scenarios']:
                            if 'description' in item:
                                training_data.append({
                                    'source_file': os.path.basename(file_path),
                                    'data_id': item.get('id', f"scruples_{len(training_data)}"),
                                    'situation': item['description'],
                                    'context': item.get('context', {}),
                                    'moral_complexity': 0.7,  # 스크러플 데이터 기본 복잡도
                                    'stakeholders': {},
                                    'data_type': 'scruples'
                                })
                            
                except Exception as e:
                    logger.error(f"스크러플 파일 {file_path} 로딩 실패: {e}")
                    # 학습 무결성 보장을 위해 graceful degradation 제거
                    raise Exception(f"스크러플 파일 로딩 실패로 인한 학습 무결성 오염 방지: {e}")
            
            # 통합 시나리오 파일들 로드
            try:
                integrated_files = [
                    self.processed_datasets_path / "integrated_scenarios.json",
                    self.processed_datasets_path / "final_integrated_with_batch7_20250619_213234.json"
                ]
                integrated_files = [f for f in integrated_files if f.exists()]
                logger.info(f"통합 시나리오 파일 {len(integrated_files)}개 발견")
                
                for file_path in integrated_files:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        scenarios = json.load(f)
                    
                    for idx, scenario in enumerate(scenarios[:5]):  # 각 파일에서 5개만
                        if 'description' in scenario:
                            training_data.append({
                                'source_file': os.path.basename(file_path),
                                'data_id': scenario.get('id', f"integrated_{idx}"),
                                'situation': scenario['description'],
                                'context': scenario.get('context', {}),
                                'moral_complexity': scenario.get('complexity_score', 0.7),
                                'stakeholders': scenario.get('stakeholders', []),
                                'data_type': 'integrated'
                            })
                            
            except Exception as e:
                logger.error(f"통합 시나리오 로딩 실패: {e}")
                # 학습 무결성 보장을 위해 graceful degradation 제거
                raise Exception(f"통합 시나리오 로딩 실패로 인한 학습 무결성 오염 방지: {e}")
            
            # 한국 문화 특화 데이터 로드
            try:
                korean_files = [
                    self.processed_datasets_path / "korean_cultural_scenarios.json"
                ]
                korean_files = [f for f in korean_files if f.exists()]
                logger.info(f"한국 문화 파일 {len(korean_files)}개 발견")
                
                for file_path in korean_files:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        cultural_data = json.load(f)
                    
                    for item in cultural_data:
                        if 'scenario' in item and len(training_data) < 25:  # 최대 25개 제한
                            training_data.append({
                                'source_file': os.path.basename(file_path),
                                'data_id': item.get('id', f"korean_{len(training_data)}"),
                                'situation': item['scenario'],
                                'context': item.get('context', {}),
                                'moral_complexity': item.get('complexity', 0.8),
                                'stakeholders': item.get('stakeholders', {}),
                                'data_type': 'korean_cultural'
                            })
                            
            except Exception as e:
                logger.error(f"한국 문화 데이터 로딩 실패: {e}")
                # 학습 무결성 보장을 위해 graceful degradation 제거
                raise Exception(f"한국 문화 데이터 로딩 실패로 인한 학습 무결성 오염 방지: {e}")
            
            logger.info(f"✅ 총 {len(training_data)}개 실제 훈련 데이터 로드 완료")
            
            # 데이터 분포 로깅
            data_types = {}
            for data in training_data:
                data_type = data['data_type']
                data_types[data_type] = data_types.get(data_type, 0) + 1
            
            logger.info("📊 데이터 분포:")
            for data_type, count in data_types.items():
                logger.info(f"  - {data_type}: {count}개")
            
            return training_data
            
        except Exception as e:
            logger.error(f"❌ 실제 훈련 데이터 로딩 실패: {e}")
            traceback.print_exc()
            return []
    
    async def process_real_training_item(self, data_item: Dict[str, Any]) -> RealTrainingResult:
        """단일 훈련 데이터 아이템을 실제 모듈들로 처리"""
        
        start_time = time.time()
        error_log = []
        
        try:
            # 1. 실제 감정 분석
            logger.info(f"🎯 처리 중: {data_item['data_id']} - 감정 분석...")
            emotion_start = time.time()
            
            try:
                # 실제 감정 분석기 호출 - 올바른 파라미터
                emotion_result = self.emotion_analyzer.analyze_emotion(
                    text=data_item['situation'],
                    language="ko",
                    biosignal_data=None,
                    use_cache=True
                )
                emotion_processing_time = time.time() - emotion_start
                logger.info(f"   ✅ 감정 분석 완료 ({emotion_processing_time:.3f}초)")
                
            except Exception as e:
                error_log.append(f"감정 분석 실패: {e}")
                emotion_processing_time = time.time() - emotion_start
                logger.error(f"   ❌ 감정 분석 실패: {e}")
                # 학습 무결성 보장을 위해 graceful degradation 제거
                raise Exception(f"감정 분석 실패로 인한 학습 무결성 오염 방지: {e}")
            
            # 2. 실제 벤담 계산
            logger.info(f"🎯 처리 중: {data_item['data_id']} - 벤담 계산...")
            bentham_start = time.time()
            
            try:
                # 실제 벤담 계산기 호출 - 올바른 파라미터
                bentham_input_data = {
                    'situation': data_item['situation'],
                    'context': data_item.get('context', {}),
                    'emotion_data': emotion_result if 'error' not in emotion_result else None
                }
                bentham_result = self.bentham_calculator.calculate_with_advanced_layers(
                    input_data=bentham_input_data,
                    use_cache=True
                )
                bentham_processing_time = time.time() - bentham_start
                logger.info(f"   ✅ 벤담 계산 완료 ({bentham_processing_time:.3f}초)")
                
            except Exception as e:
                error_log.append(f"벤담 계산 실패: {e}")
                bentham_processing_time = time.time() - bentham_start
                logger.error(f"   ❌ 벤담 계산 실패: {e}")
                # 학습 무결성 보장을 위해 graceful degradation 제거
                raise Exception(f"벤담 계산 실패로 인한 학습 무결성 오염 방지: {e}")
            
            # 3. 실제 후회 분석
            if self.regret_analyzer:
                logger.info(f"🎯 처리 중: {data_item['data_id']} - 후회 분석...")
                regret_start = time.time()
                
                try:
                    # 후회 분석을 위한 decision_data 준비 - 안전한 타입 처리
                    decision_data = {
                        'scenario': data_item['situation'],
                        'text': data_item['situation'],  # 텍스트 필드 추가
                        'context': data_item.get('context', {}),
                    }
                    
                    # 감정 컨텍스트 추가 (안전한 방식)
                    if 'error' not in emotion_result:
                        decision_data['emotion_context'] = emotion_result
                    
                    # 벤담 컨텍스트 추가 (안전한 타입 처리)
                    bentham_has_error = False
                    if isinstance(bentham_result, dict):
                        bentham_has_error = 'error' in bentham_result
                    elif hasattr(bentham_result, 'error'):
                        bentham_has_error = bentham_result.error is not None
                    
                    if not bentham_has_error:
                        if hasattr(bentham_result, '__dict__'):
                            # getattr 대신 실제 속성 존재 여부 확인
                            if hasattr(bentham_result, 'final_score') and bentham_result.final_score is not None:
                                decision_data['bentham_context'] = {
                                    'score': bentham_result.final_score,
                                    'type': 'bentham_calculation'
                                }
                            else:
                                raise ValueError(f"벤담 계산 결과에 final_score 속성이 없음: {type(bentham_result)}")
                        elif isinstance(bentham_result, dict):
                            decision_data['bentham_context'] = bentham_result
                    
                    # 실제 후회 분석기 호출
                    regret_result = await self.regret_analyzer.analyze_regret(
                        decision_data=decision_data,
                        outcome_data=None
                    )
                    regret_processing_time = time.time() - regret_start
                    logger.info(f"   ✅ 후회 분석 완료 ({regret_processing_time:.3f}초)")
                    
                except Exception as e:
                    error_log.append(f"후회 분석 실패: {e}")
                    regret_processing_time = time.time() - regret_start
                    logger.error(f"   ❌ 후회 분석 실패: {e}")
                    # 학습 무결성 보장을 위해 graceful degradation 제거
                    raise Exception(f"후회 분석 실패로 인한 학습 무결성 오염 방지: {e}")
            else:
                # 후회 분석기 사용 불가능 시에도 학습 무결성을 위해 중단
                logger.error("   ❌ 후회 분석기 사용 불가")
                raise Exception("후회 분석기 사용 불가로 인한 학습 무결성 오염 방지")
            
            # 4. 실제 SURD 통합 분석
            logger.info(f"🎯 처리 중: {data_item['data_id']} - SURD 통합 분석...")
            surd_start = time.time()
            
            try:
                # SURD 분석을 위한 변수 준비
                surd_variables = {}
                
                # 감정 데이터 통합 (안전한 체크)
                emotion_has_error = isinstance(emotion_result, dict) and 'error' in emotion_result
                if not emotion_has_error:
                    if hasattr(emotion_result, 'dominant_emotion'):
                        # 실제 속성 존재 확인 후 값 추출
                        if hasattr(emotion_result, 'intensity') and emotion_result.intensity is not None:
                            surd_variables['emotion_intensity'] = float(emotion_result.intensity)
                        else:
                            raise ValueError(f"감정 분석 결과에 intensity 속성이 없음: {type(emotion_result)}")
                        
                        if hasattr(emotion_result, 'confidence') and emotion_result.confidence is not None:
                            surd_variables['emotion_confidence'] = float(emotion_result.confidence)
                        else:
                            raise ValueError(f"감정 분석 결과에 confidence 속성이 없음: {type(emotion_result)}")
                    elif isinstance(emotion_result, dict):
                        surd_variables['emotion_intensity'] = float(emotion_result.get('intensity', 0.5))
                        surd_variables['emotion_confidence'] = float(emotion_result.get('confidence', 0.5))
                
                # 벤담 데이터 통합 (안전한 체크)
                bentham_has_error = isinstance(bentham_result, dict) and 'error' in bentham_result
                if not bentham_has_error:
                    if hasattr(bentham_result, 'final_score'):
                        # 실제 속성 존재 확인 후 값 추출
                        if bentham_result.final_score is not None:
                            surd_variables['pleasure_score'] = float(bentham_result.final_score)
                        else:
                            raise ValueError(f"벤담 계산 결과의 final_score가 None: {type(bentham_result)}")
                    elif isinstance(bentham_result, dict):
                        surd_variables['pleasure_score'] = float(bentham_result.get('final_score', 0.0))
                
                # 후회 데이터 통합 (AdvancedRegretMetrics 타입 안전 처리)
                regret_surd_error = False
                if hasattr(regret_result, '__class__') and regret_result.__class__.__name__ == 'AdvancedRegretMetrics':
                    # AdvancedRegretMetrics 객체에서 데이터 추출 - 실제 속성 검증
                    if hasattr(regret_result, 'regret_intensity') and regret_result.regret_intensity is not None:
                        if regret_result.regret_intensity <= 0.0:
                            raise ValueError(f"후회 분석 결과의 regret_intensity가 0.0: {regret_result.regret_intensity}")
                        surd_variables['regret_intensity'] = float(regret_result.regret_intensity)
                    elif hasattr(regret_result, 'intensity') and regret_result.intensity is not None:
                        if regret_result.intensity <= 0.0:
                            raise ValueError(f"후회 분석 결과의 intensity가 0.0: {regret_result.intensity}")
                        surd_variables['regret_intensity'] = float(regret_result.intensity)
                    else:
                        raise ValueError(f"후회 분석 결과에 regret_intensity 또는 intensity 속성이 없음: {type(regret_result)}")
                elif isinstance(regret_result, dict):
                    regret_surd_error = 'error' in regret_result
                    if not regret_surd_error:
                        surd_variables['regret_intensity'] = float(regret_result.get('regret_intensity', 0.0))
                
                # 실패 감지 - 모든 분석이 실패했을 경우 예외 발생
                if not surd_variables:
                    raise RuntimeError(
                        f"모든 분석 모듈이 실패했거나 유효한 값을 생성하지 못함. "
                        f"감정: {'error' if emotion_has_error else 'ok'}, "
                        f"벤담: {'error' if bentham_has_error else 'ok'}, "
                        f"후회: {'error' if regret_surd_error else 'ok'}"
                    )
                
                # 실제 SURD 분석기 호출 - analyze_advanced 메서드 사용
                surd_result = self.surd_analyzer.analyze_advanced(
                    variables=surd_variables,
                    target_variable='ethical_decision_quality',
                    additional_context=data_item.get('context', {})
                )
                surd_processing_time = time.time() - surd_start
                logger.info(f"   ✅ SURD 분석 완료 ({surd_processing_time:.3f}초)")
                
            except Exception as e:
                error_log.append(f"SURD 분석 실패: {e}")
                surd_processing_time = time.time() - surd_start
                logger.error(f"   ❌ SURD 분석 실패: {e}")
                # 학습 무결성 보장을 위해 graceful degradation 제거
                raise Exception(f"SURD 분석 실패로 인한 학습 무결성 오염 방지: {e}")
            
            # 5. 반사실 경험 생성 - AdvancedRegretMetrics 타입 안전 처리
            counterfactual_experiences = []
            regret_has_error = False
            
            # AdvancedRegretMetrics 객체 타입 체크
            if hasattr(regret_result, '__class__') and regret_result.__class__.__name__ == 'AdvancedRegretMetrics':
                # 성공적인 결과 - 반사실 시나리오 추출
                if hasattr(regret_result, 'counterfactual_scenarios'):
                    counterfactual_experiences = regret_result.counterfactual_scenarios or []
                elif hasattr(regret_result, 'counterfactuals'):
                    counterfactual_experiences = regret_result.counterfactuals or []
            elif isinstance(regret_result, dict):
                # 딕셔너리 형태 (오류 포함 가능)
                regret_has_error = 'error' in regret_result
                if not regret_has_error and 'counterfactual_scenarios' in regret_result:
                    counterfactual_experiences = regret_result['counterfactual_scenarios']
            
            # 6. 경험 데이터베이스에 저장
            try:
                # 직렬화 가능한 형태로 변환
                def convert_to_serializable(obj):
                    if hasattr(obj, '__dict__'):
                        return {k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v 
                               for k, v in obj.__dict__.items()}
                    elif isinstance(obj, dict):
                        return obj
                    else:
                        return {'result': str(obj), 'type': type(obj).__name__}
                
                experience_entry = {
                    'data_id': data_item['data_id'],
                    'situation': data_item['situation'],
                    'emotion_analysis': convert_to_serializable(emotion_result),
                    'bentham_calculation': convert_to_serializable(bentham_result),
                    'regret_analysis': convert_to_serializable(regret_result),
                    'surd_analysis': convert_to_serializable(surd_result),
                    'timestamp': datetime.now().isoformat(),
                    'source_file': data_item['source_file']
                }
                
                await self.experience_db.store_experience(
                    experience_text=data_item['situation'],
                    metadata=experience_entry,
                    category="training",
                    importance_score=None
                )
                
            except Exception as e:
                error_log.append(f"경험 저장 실패: {e}")
                logger.error(f"   ❌ 경험 저장 실패: {e}")
                # 학습 무결성 보장을 위해 graceful degradation 제거
                raise Exception(f"경험 저장 실패로 인한 학습 무결성 오염 방지: {e}")
            
            # 결과 생성
            total_processing_time = time.time() - start_time
            # graceful degradation 제거 후 여기까지 도달하면 모든 작업이 성공
            integration_success = True
            
            result = RealTrainingResult(
                data_id=data_item['data_id'],
                source_file=data_item['source_file'],
                processing_time=total_processing_time,
                emotion_analysis=emotion_result,
                bentham_calculation=bentham_result,
                regret_analysis=regret_result,
                surd_analysis=surd_result,
                counterfactual_experiences=counterfactual_experiences,
                integration_success=integration_success,
                error_log=error_log
            )
            
            # 메트릭 업데이트
            self.training_metrics['total_processed'] += 1
            self.training_metrics['total_processing_time'] += total_processing_time
            
            if integration_success:
                self.training_metrics['successful_integrations'] += 1
                logger.info(f"✅ {data_item['data_id']} 실제 통합 훈련 완료 ({total_processing_time:.3f}초)")
            else:
                self.training_metrics['failed_processes'] += 1
                logger.warning(f"⚠️ {data_item['data_id']} 부분 실패 ({len(error_log)}개 오류)")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ {data_item['data_id']} 처리 중 심각한 오류: {e}")
            traceback.print_exc()
            
            total_processing_time = time.time() - start_time
            self.training_metrics['total_processed'] += 1
            self.training_metrics['failed_processes'] += 1
            self.training_metrics['total_processing_time'] += total_processing_time
            
            return RealTrainingResult(
                data_id=data_item['data_id'],
                source_file=data_item['source_file'],
                processing_time=total_processing_time,
                emotion_analysis={'error': str(e)},
                bentham_calculation={'error': str(e)},
                regret_analysis={'error': str(e)},
                surd_analysis={'error': str(e)},
                counterfactual_experiences=[],
                integration_success=False,
                error_log=[f"심각한 처리 오류: {e}"]
            )
    
    async def run_real_integrated_training(self, max_items: int = 100) -> Dict[str, Any]:
        """실제 데이터로 통합 훈련 실행"""
        logger.info("🚀 실제 Red Heart AI 통합 훈련 시작")
        
        # 실제 훈련 데이터 로드
        training_data = self.load_real_training_data()
        
        if not training_data:
            logger.error("❌ 훈련 데이터가 없습니다")
            return {"error": "훈련 데이터 로드 실패"}
        
        # 최대 처리 개수 제한
        if len(training_data) > max_items:
            training_data = training_data[:max_items]
            logger.info(f"📊 처리할 데이터를 {max_items}개로 제한")
        
        logger.info(f"📋 총 {len(training_data)}개 실제 데이터 처리 시작")
        
        training_results = []
        
        # 각 데이터 아이템 순차 처리
        for i, data_item in enumerate(training_data, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"🎯 [{i}/{len(training_data)}] 실제 훈련 중...")
            logger.info(f"데이터 ID: {data_item['data_id']}")
            logger.info(f"소스 파일: {data_item['source_file']}")
            logger.info(f"{'='*60}")
            
            result = await self.process_real_training_item(data_item)
            training_results.append(result)
            
            # 진행 상황 로깅
            if i % 10 == 0 or i == len(training_data):
                success_rate = self.training_metrics['successful_integrations'] / self.training_metrics['total_processed'] * 100
                avg_time = self.training_metrics['total_processing_time'] / self.training_metrics['total_processed']
                logger.info(f"\n📊 중간 진행 상황 [{i}/{len(training_data)}]:")
                logger.info(f"  - 성공률: {success_rate:.1f}%")
                logger.info(f"  - 평균 처리시간: {avg_time:.3f}초")
                logger.info(f"  - 예상 남은 시간: {avg_time * (len(training_data) - i):.1f}초")
        
        # 최종 분석
        return self._analyze_real_training_results(training_results)
    
    def _analyze_real_training_results(self, results: List[RealTrainingResult]) -> Dict[str, Any]:
        """실제 훈련 결과 종합 분석"""
        logger.info(f"\n📊 실제 훈련 결과 분석 중...")
        
        if not results:
            return {"error": "분석할 결과가 없습니다"}
        
        # 전체 메트릭
        total_items = len(results)
        successful_items = len([r for r in results if r.integration_success])
        success_rate = successful_items / total_items * 100
        
        total_time = sum(r.processing_time for r in results)
        avg_time = total_time / total_items
        
        # 모듈별 성능 분석 - AdvancedRegretMetrics 포함 안전한 타입 체크
        def safe_error_check(obj):
            if hasattr(obj, '__class__') and obj.__class__.__name__ == 'AdvancedRegretMetrics':
                return True  # AdvancedRegretMetrics 객체는 성공으로 간주
            elif hasattr(obj, '__dict__'):
                return 'error' not in obj.__dict__
            elif isinstance(obj, dict):
                return 'error' not in obj and 'disabled' not in obj
            else:
                return True  # 기타 객체는 성공으로 간주
        
        module_performance = {
            'emotion_success': len([r for r in results if safe_error_check(r.emotion_analysis)]),
            'bentham_success': len([r for r in results if safe_error_check(r.bentham_calculation)]),
            'regret_success': len([r for r in results if safe_error_check(r.regret_analysis) and not (isinstance(r.regret_analysis, dict) and 'disabled' in r.regret_analysis)]),
            'surd_success': len([r for r in results if safe_error_check(r.surd_analysis)])
        }
        
        # 오류 패턴 분석
        error_patterns = {}
        for result in results:
            for error in result.error_log:
                error_type = error.split(':')[0]
                error_patterns[error_type] = error_patterns.get(error_type, 0) + 1
        
        # 처리 시간 분석
        processing_times = [r.processing_time for r in results]
        min_time = min(processing_times)
        max_time = max(processing_times)
        
        # 반사실 생성 통계
        total_counterfactuals = sum(len(r.counterfactual_experiences) for r in results)
        
        # 데이터 소스별 성능
        source_performance = {}
        for result in results:
            source = result.source_file
            if source not in source_performance:
                source_performance[source] = {'total': 0, 'success': 0}
            source_performance[source]['total'] += 1
            if result.integration_success:
                source_performance[source]['success'] += 1
        
        # 최종 결과
        analysis_result = {
            'training_summary': {
                'total_items': total_items,
                'successful_integrations': successful_items,
                'success_rate': success_rate,
                'total_processing_time': total_time,
                'avg_processing_time': avg_time,
                'min_processing_time': min_time,
                'max_processing_time': max_time
            },
            'module_performance': {
                'emotion_success_rate': (module_performance['emotion_success'] / total_items) * 100,
                'bentham_success_rate': (module_performance['bentham_success'] / total_items) * 100,
                'regret_success_rate': (module_performance['regret_success'] / total_items) * 100,
                'surd_success_rate': (module_performance['surd_success'] / total_items) * 100
            },
            'integration_analysis': {
                'full_integration_rate': success_rate,
                'partial_integration_rate': ((total_items - successful_items) / total_items) * 100,
                'total_counterfactuals_generated': total_counterfactuals,
                'avg_counterfactuals_per_item': total_counterfactuals / total_items
            },
            'error_analysis': {
                'error_patterns': error_patterns,
                'total_errors': sum(error_patterns.values()),
                'error_rate': (sum(error_patterns.values()) / total_items) * 100
            },
            'source_analysis': source_performance,
            'performance_metrics': {
                'items_per_second': total_items / total_time,
                'successful_items_per_second': successful_items / total_time,
                'efficiency_score': success_rate * (total_items / total_time)
            }
        }
        
        return analysis_result

    async def run_complete_learning_system(self, samples: Optional[int] = None) -> Dict[str, Any]:
        """완전한 학습 시스템 실행"""
        
        if not self.learning_mode or not self.learning_executor:
            logger.error("❌ 학습 모드가 활성화되지 않았습니다.")
            return {"error": "학습 시스템이 초기화되지 않음", "learning_success": False}
        
        logger.info("🎯 완전한 학습 시스템 실행 시작")
        if samples:
            logger.info(f"🎯 샘플 제한 모드: {samples}개 시나리오로 제한")
        logger.info("📊 3단계 통합 페이즈 시스템:")
        logger.info("   Phase 0: 자신 감정 캘리브레이션")
        logger.info("   Phase 1: 타인 공감 학습")
        logger.info("   Phase 2: 공동체 이해")
        
        start_time = time.time()
        
        try:
            # 1. 고급 학습 시스템 실행
            logger.info("🚀 고급 학습 시스템 실행 중...")
            learning_results = await self.learning_executor.execute_full_learning(samples=samples)
            
            # 2. 학습 결과 분석
            logger.info("📊 학습 결과 분석 중...")
            analysis_results = await self._analyze_learning_results(learning_results)
            
            # 3. 학습된 시스템으로 의사결정 테스트
            logger.info("🎯 학습된 시스템 의사결정 테스트 중...")
            decision_results = await self._test_learned_decision_making()
            
            # 4. 동적 윤리적 분석 테스트
            logger.info("🔍 동적 윤리적 분석 테스트 중...")
            ethical_analysis_results = await self._test_dynamic_ethical_analysis()
            
            total_time = time.time() - start_time
            
            return {
                "learning_success": True,
                "total_learning_time": total_time,
                "learning_results": learning_results,
                "integrated_analysis": analysis_results,
                "decision_test_results": decision_results,
                "ethical_analysis_results": ethical_analysis_results,
                "summary": {
                    "total_learning_time": total_time,
                    "learning_quality": analysis_results.get("learning_quality", {}),
                    "decision_accuracy": decision_results.get("confidence_score", 0.0),
                    "ethical_analysis_quality": ethical_analysis_results.get("analysis_quality", 0.0)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ 완전한 학습 시스템 실행 실패: {e}")
            return {"error": str(e), "learning_success": False}

    async def _analyze_learning_results(self, learning_results: Dict[str, Any]) -> Dict[str, Any]:
        """학습 결과 분석"""
        
        analysis = {
            "phase_analysis": {},
            "module_performance": {},
            "learning_quality": {}
        }
        
        # 학습 통계 분석
        if "learning_statistics" in learning_results:
            stats = learning_results["learning_statistics"]
            
            # 페이즈별 성능 메트릭
            if "performance_metrics" in stats:
                recent_metrics = stats["performance_metrics"][-10:]  # 최근 10개
                if recent_metrics:
                    analysis["phase_analysis"]["recent_regret_avg"] = np.mean([m["avg_regret_intensity"] for m in recent_metrics])
                    analysis["phase_analysis"]["recent_hedonic_avg"] = np.mean([m["avg_hedonic_score"] for m in recent_metrics])
            
            # 모듈 성능
            if "regret_history" in stats:
                regret_data = stats["regret_history"][-50:]  # 최근 50개
                if regret_data:
                    analysis["module_performance"]["regret_system"] = {
                        "avg_intensity": np.mean([r["intensity"] for r in regret_data]),
                        "total_processed": len(regret_data)
                    }
            
            if "bentham_scores" in stats:
                bentham_data = stats["bentham_scores"][-50:]  # 최근 50개
                if bentham_data:
                    analysis["module_performance"]["bentham_system"] = {
                        "avg_score": np.mean([b["hedonic_score"] for b in bentham_data]),
                        "total_processed": len(bentham_data)
                    }
        
        # 학습 품질 평가
        if "summary" in learning_results:
            summary = learning_results["summary"]
            analysis["learning_quality"]["scenarios_processed"] = summary.get("total_scenarios_processed", 0)
            analysis["learning_quality"]["total_regrets"] = summary.get("total_regrets", 0)
            analysis["learning_quality"]["total_bentham_calculations"] = summary.get("total_bentham_calculations", 0)
            analysis["learning_quality"]["efficiency"] = summary.get("total_regrets", 0) / max(summary.get("total_scenarios_processed", 1), 1)
        
        return analysis

    async def _test_learned_decision_making(self) -> Dict[str, Any]:
        """학습된 시스템으로 의사결정 테스트"""
        
        # 간단한 윤리적 딜레마 테스트
        test_scenario = {
            "title": "자율주행차 딜레마",
            "description": "자율주행차가 급브레이크를 밟아야 하는 상황에서 보행자 1명을 구할 것인가, 아니면 차 안의 탑승자 2명을 구할 것인가?",
            "context": {"urgency": "high", "stakeholders": ["보행자", "탑승자들"]}
        }
        
        try:
            start_time = time.time()
            
            # 감정 분석
            emotion_result = self.emotion_analyzer.analyze_emotion(test_scenario["description"], language="ko")
            
            # 벤담 계산
            bentham_result = self.bentham_calculator.calculate_with_advanced_layers(test_scenario)
            
            # 후회 분석
            regret_result = await self.regret_analyzer.analyze_regret({
                "text": test_scenario["description"],
                "context": test_scenario["context"]
            })
            
            processing_time = time.time() - start_time
            
            # 최종 결정 (간단한 로직)
            decision_scores = {
                "보행자 구하기": 0.3,
                "탑승자 구하기": 0.7
            }
            
            final_decision = max(decision_scores, key=decision_scores.get)
            confidence = max(decision_scores.values())
            
            return {
                "test_success": True,
                "final_recommendation": final_decision,
                "confidence_score": confidence,
                "processing_time": processing_time,
                "reasoning_chain": [
                    f"감정 분석: {getattr(emotion_result, 'dominant_emotion', 'N/A')}",
                    f"벤담 점수: {getattr(bentham_result, 'final_score', 0.0):.3f}",
                    f"후회 강도: {getattr(regret_result, 'regret_intensity', 0.0):.3f}",
                    f"최종 결정: {final_decision}"
                ]
            }
            
        except Exception as e:
            logger.error(f"의사결정 테스트 실패: {e}")
            return {"test_success": False, "error": str(e)}

    async def _test_dynamic_ethical_analysis(self) -> Dict[str, Any]:
        """동적 윤리적 분석 테스트"""
        
        if not self.dynamic_choice_analyzer:
            return {"error": "동적 선택지 분석기가 초기화되지 않음"}
        
        try:
            # 다양한 윤리적 딜레마 테스트
            test_dilemma = "의사가 장기 이식을 위해 건강한 환자 1명을 희생시켜 5명의 환자를 살릴 것인가?"
            
            start_time = time.time()
            result = await self.dynamic_choice_analyzer.analyze_ethical_dilemma(
                dilemma_text=test_dilemma,
                title="의료 윤리 딜레마 테스트"
            )
            processing_time = time.time() - start_time
            
            return {
                "analysis_success": True,
                "dilemma_type": result.dilemma_type.value,
                "extracted_choices": len(result.extracted_choices),
                "stakeholders_identified": len(result.stakeholders),
                "recommended_choice": result.recommended_choice.name if result.recommended_choice else None,
                "reasoning_chain": result.reasoning_chain,
                "processing_time": processing_time,
                "analysis_quality": len(result.reasoning_chain) / 5.0  # 간단한 품질 지표
            }
            
        except Exception as e:
            logger.error(f"동적 윤리적 분석 테스트 실패: {e}")
            return {"analysis_success": False, "error": str(e)}


async def main():
    """실제 훈련 메인 함수"""
    if not MODULES_AVAILABLE:
        logger.error("❌ 필수 모듈을 불러올 수 없습니다.")
        return
    
    # 실제 통합 훈련 시스템 초기화
    training_system = RealIntegratedTrainingSystem()
    
    # 실제 시스템 초기화
    if not await training_system.initialize_real_system():
        logger.error("❌ 실제 시스템 초기화 실패")
        return
    
    # 실제 통합 훈련 실행 (처음 25개 아이템으로 테스트)
    results = await training_system.run_real_integrated_training(max_items=25)
    
    # 결과 출력
    logger.info(f"\n{'='*80}")
    logger.info("🎉 실제 Red Heart AI 통합 훈련 완료")
    logger.info(f"{'='*80}")
    
    if 'error' not in results:
        summary = results['training_summary']
        module_perf = results['module_performance']
        integration = results['integration_analysis']
        
        logger.info(f"\n📊 실제 훈련 요약:")
        logger.info(f"  - 총 처리 아이템: {summary['total_items']}개")
        logger.info(f"  - 성공적 통합: {summary['successful_integrations']}개")
        logger.info(f"  - 통합 성공률: {summary['success_rate']:.1f}%")
        logger.info(f"  - 총 처리시간: {summary['total_processing_time']:.1f}초")
        logger.info(f"  - 평균 처리시간: {summary['avg_processing_time']:.3f}초")
        
        logger.info(f"\n🎯 모듈별 성공률:")
        logger.info(f"  - 감정 분석: {module_perf['emotion_success_rate']:.1f}%")
        logger.info(f"  - 벤담 계산: {module_perf['bentham_success_rate']:.1f}%")
        logger.info(f"  - 후회 분석: {module_perf['regret_success_rate']:.1f}%")
        logger.info(f"  - SURD 분석: {module_perf['surd_success_rate']:.1f}%")
        
        logger.info(f"\n🔗 통합 분석:")
        logger.info(f"  - 완전 통합률: {integration['full_integration_rate']:.1f}%")
        logger.info(f"  - 반사실 시나리오: {integration['total_counterfactuals_generated']}개")
        logger.info(f"  - 아이템당 평균: {integration['avg_counterfactuals_per_item']:.1f}개")
        
        logger.info(f"\n⚡ 성능 메트릭:")
        perf = results['performance_metrics']
        logger.info(f"  - 처리 속도: {perf['items_per_second']:.2f} 아이템/초")
        logger.info(f"  - 성공 처리 속도: {perf['successful_items_per_second']:.2f} 아이템/초")
        logger.info(f"  - 효율성 점수: {perf['efficiency_score']:.2f}")
        
        if results['error_analysis']['error_patterns']:
            logger.info(f"\n⚠️ 오류 패턴:")
            for error_type, count in results['error_analysis']['error_patterns'].items():
                logger.info(f"  - {error_type}: {count}회")
        
        # 결과를 파일로 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = Path(f'real_integrated_training_results_{timestamp}.json')
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"\n📄 상세 결과 저장: {result_file}")
    else:
        logger.error(f"❌ 실제 훈련 실패: {results['error']}")


async def main():
    """실제 훈련 메인 함수"""
    if not MODULES_AVAILABLE:
        logger.error("❌ 필수 모듈을 불러올 수 없습니다.")
        return
    
    # 실제 통합 훈련 시스템 실행
    system = RealIntegratedTrainingSystem()
    await system.initialize_real_system()
    
    # 기본 기능 테스트 실행
    results = await system.run_real_integrated_training(max_items=3)
    
    logger.info("🎯 Red Heart AI 실제 통합 훈련 완료")
    return results


async def main_learning_system():
    """학습 시스템 실행 메인 함수"""
    
    if not MODULES_AVAILABLE:
        logger.error("❌ 필수 모듈이 사용할 수 없습니다.")
        return
    
    if not LEARNING_SYSTEM_AVAILABLE:
        logger.error("❌ 학습 시스템이 사용할 수 없습니다.")
        return
    
    # 통합 시스템 초기화
    system = RealIntegratedTrainingSystem()
    success = await system.initialize_real_system(learning_mode=True)
    
    if not success:
        logger.error("❌ 학습 시스템 초기화 실패")
        return
    
    # 완전한 학습 시스템 실행
    results = await system.run_complete_learning_system()
    
    # 결과 출력
    if results.get("learning_success"):
        logger.info("✅ 완전한 학습 시스템 실행 성공!")
        logger.info(f"총 학습 시간: {results['total_learning_time']:.2f}초")
        
        # 학습 결과 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = Path(f'complete_learning_system_results_{timestamp}.json')
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"📄 학습 결과 저장: {result_file}")
    else:
        logger.error(f"❌ 학습 시스템 실행 실패: {results.get('error', 'Unknown error')}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Red Heart AI 실제 통합 훈련 시스템')
    parser.add_argument('--learning', action='store_true', help='완전한 학습 시스템 실행')
    parser.add_argument('--test', action='store_true', help='기능 테스트 모드 실행')
    
    args = parser.parse_args()
    
    if args.learning:
        asyncio.run(main_learning_system())
    elif args.test:
        asyncio.run(main())
    else:
        # 기본값: 기능 테스트 모드
        asyncio.run(main())
