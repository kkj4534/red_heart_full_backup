"""
의존성 검증기 - Red Heart 시스템 시작 전 필수 의존성 확인
Dependency Validator for Red Heart System
"""

import sys
import importlib
import subprocess
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import traceback

logger = logging.getLogger('RedHeart.DependencyValidator')

class DependencyValidator:
    """의존성 검증 클래스"""
    
    def __init__(self):
        # 필수 패키지 (시스템 시작에 반드시 필요)
        self.critical_packages = {
            # 과학 계산 라이브러리 - 가장 중요
            'numpy': '>=1.24.0',
            'torch': '>=2.0.0',
            
            # 기본 유틸리티
            'tqdm': '>=4.65.0',
        }
        
        # 선택적 패키지 (없어도 시스템 시작 가능, 경고만 출력)
        self.optional_packages = {
            # 대용량 ML/AI 라이브러리 - 온라인 다운로드 위험
            'transformers': '>=4.30.0', 
            'sentence_transformers': '>=2.2.0',
            'sklearn': '>=1.3.0',
            'faiss': '>=1.7.0',
            
            # 과학 계산 확장
            'scipy': '>=1.10.0',
            'pandas': '>=2.0.0',
            
            # 네트워크 및 그래프
            'networkx': '>=3.0',
            'joblib': '>=1.2.0',
            'requests': '>=2.26.0',
        }
        
        self.gpu_requirements = {
            'cuda_available': False,  # 필수는 아니지만 권장
            'memory_gb': 6.0          # 최소 GPU 메모리
        }
        
        self.validation_results = {}
        
    def validate_all(self) -> bool:
        """모든 의존성 검증 (빠른 모드)"""
        logger.info("🔍 Red Heart 시스템 의존성 검증 시작 (빠른 모드)")
        
        success = True
        
        # 1. Python 버전 확인
        if not self._validate_python_version():
            success = False
            
        # 2. 필수 패키지만 먼저 확인 (블로킹 방지)
        if not self._validate_critical_packages():
            success = False
            logger.error("❌ 필수 패키지 검증 실패 - 시스템을 시작할 수 없습니다")
            return False
            
        # 3. 선택적 패키지 확인 (오류 무시, 경고만)
        self._validate_optional_packages()
        
        # 4. GPU 환경 확인 (선택적, 빠른 체크)
        self._validate_gpu_environment_fast()
        
        # 5. 파일 시스템 권한 확인 (빠른 체크)
        if not self._validate_filesystem_fast():
            logger.warning("⚠️ 파일 시스템 권한 문제 발견, 계속 진행...")
            
        # 6. 메모리 확인 (빠른 체크)
        if not self._validate_memory_fast():
            logger.warning("⚠️ 메모리 부족 가능성, 계속 진행...")
            
        if success:
            logger.info("✅ 필수 의존성 검증 통과 - 시스템 시작 가능")
        else:
            logger.error("❌ 필수 의존성 검증 실패")
            
        return success
        
    def _validate_python_version(self) -> bool:
        """Python 버전 확인"""
        min_version = (3, 8)
        current_version = sys.version_info[:2]
        
        if current_version >= min_version:
            logger.info(f"✅ Python {current_version[0]}.{current_version[1]}: 요구사항 만족")
            return True
        else:
            logger.error(f"❌ Python {current_version[0]}.{current_version[1]}: 최소 {min_version[0]}.{min_version[1]} 필요")
            return False
            
    def _validate_critical_packages(self) -> bool:
        """필수 패키지만 검증 (빠른 체크)"""
        logger.info("📦 필수 패키지 검증 중...")
        
        all_success = True
        
        for package_name, version_requirement in self.critical_packages.items():
            try:
                # 타임아웃 설정 (5초)
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"{package_name} import 타임아웃")
                
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(5)  # 5초 타임아웃
                
                try:
                    # 패키지 import 테스트
                    module = importlib.import_module(package_name)
                    
                    # 버전 확인
                    version = getattr(module, '__version__', 'unknown')
                    logger.info(f"✅ {package_name}: {version}")
                    
                    self.validation_results[package_name] = {
                        'status': 'success',
                        'version': version,
                        'requirement': version_requirement
                    }
                    
                finally:
                    signal.alarm(0)  # 타임아웃 해제
                    
            except TimeoutError as e:
                logger.error(f"❌ {package_name}: 타임아웃 - {e}")
                self.validation_results[package_name] = {
                    'status': 'timeout',
                    'error': str(e),
                    'requirement': version_requirement
                }
                all_success = False
                
            except ImportError as e:
                logger.error(f"❌ {package_name}: import 실패 - {e}")
                self.validation_results[package_name] = {
                    'status': 'import_failed',
                    'error': str(e),
                    'requirement': version_requirement
                }
                all_success = False
                
            except Exception as e:
                logger.error(f"⚠️  {package_name}: 예상치 못한 오류 - {e}")
                self.validation_results[package_name] = {
                    'status': 'error',
                    'error': str(e),
                    'requirement': version_requirement
                }
                
        return all_success
    
    def _validate_optional_packages(self):
        """선택적 패키지 검증 (오프라인 모드, 다운로드 방지)"""
        logger.info("📦 선택적 패키지 검증 중 (오프라인 모드)...")
        
        # 오프라인 모드 환경 변수 설정
        import os
        original_env = {}
        offline_env = {
            'TRANSFORMERS_OFFLINE': '1',
            'HF_HUB_OFFLINE': '1', 
            'HF_DATASETS_OFFLINE': '1',
            'SENTENCE_TRANSFORMERS_HOME': os.path.expanduser('~/.cache/torch/sentence_transformers'),
            'TRANSFORMERS_CACHE': os.path.expanduser('~/.cache/huggingface/transformers'),
        }
        
        # 환경 변수 백업 및 설정
        for key, value in offline_env.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value
        
        try:
            for package_name, version_requirement in self.optional_packages.items():
                try:
                    # 특별 처리가 필요한 패키지들
                    if package_name == 'transformers':
                        self._validate_transformers_offline()
                    elif package_name == 'sentence_transformers':
                        self._validate_sentence_transformers_offline()
                    elif package_name == 'faiss':
                        self._validate_faiss_offline() 
                    else:
                        # 일반 패키지는 빠른 import 테스트
                        self._validate_generic_package_fast(package_name, version_requirement)
                        
                except Exception as e:
                    logger.warning(f"⚠️ {package_name}: 검증 실패 (선택적) - {e}")
                    
        finally:
            # 환경 변수 복원
            for key, original_value in original_env.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value
    
    def _validate_transformers_offline(self):
        """Transformers 오프라인 검증"""
        import os
        from pathlib import Path
        
        # 로컬 캐시 확인
        cache_dir = Path.home() / '.cache' / 'huggingface' / 'transformers'
        if cache_dir.exists() and list(cache_dir.glob('*')):
            logger.info("✅ transformers: 로컬 캐시 발견, 오프라인 모드 가능")
            
            # 실제 import 테스트 (오프라인)
            try:
                import signal
                signal.alarm(3)  # 3초 타임아웃
                
                import transformers
                version = transformers.__version__
                logger.info(f"✅ transformers: {version} (오프라인)")
                
                signal.alarm(0)
                return True
                
            except Exception as e:
                logger.warning(f"⚠️ transformers: 오프라인 import 실패 - {e}")
                signal.alarm(0)
                return False
        else:
            logger.warning("⚠️ transformers: 로컬 캐시 없음, 런타임에 다운로드 필요")
            return False
    
    def _validate_sentence_transformers_offline(self):
        """Sentence Transformers 오프라인 검증"""
        import os
        from pathlib import Path
        
        # 로컬 모델 캐시 확인
        cache_dir = Path.home() / '.cache' / 'torch' / 'sentence_transformers'
        if cache_dir.exists() and list(cache_dir.glob('*')):
            logger.info("✅ sentence_transformers: 로컬 모델 캐시 발견")
            
            try:
                import signal
                signal.alarm(3)  # 3초 타임아웃
                
                import sentence_transformers
                version = sentence_transformers.__version__
                logger.info(f"✅ sentence_transformers: {version} (오프라인)")
                
                signal.alarm(0)
                return True
                
            except Exception as e:
                logger.warning(f"⚠️ sentence_transformers: 오프라인 import 실패 - {e}")
                signal.alarm(0)
                return False
        else:
            logger.warning("⚠️ sentence_transformers: 로컬 모델 없음, 런타임에 다운로드 필요")
            return False
    
    def _validate_faiss_offline(self):
        """FAISS 검증 (다운로드 없는 순수 계산 라이브러리)"""
        try:
            import signal
            signal.alarm(2)  # 2초 타임아웃
            
            import faiss
            logger.info(f"✅ faiss: 사용 가능")
            
            signal.alarm(0)
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ faiss: import 실패 - {e}")
            signal.alarm(0)
            return False
    
    def _validate_generic_package_fast(self, package_name: str, version_requirement: str):
        """일반 패키지 빠른 검증"""
        try:
            import signal
            signal.alarm(2)  # 2초 타임아웃
            
            module = importlib.import_module(package_name)
            version = getattr(module, '__version__', 'unknown')
            logger.info(f"✅ {package_name}: {version}")
            
            signal.alarm(0)
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ {package_name}: 빠른 검증 실패 - {e}")
            signal.alarm(0)
            return False
        
    def _validate_gpu_environment_fast(self) -> bool:
        """GPU 환경 빠른 확인 (타임아웃 적용)"""
        logger.info("🔧 GPU 환경 빠른 확인 중...")
        
        try:
            import signal
            signal.alarm(5)  # 5초 타임아웃
            
            import torch
            
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                if device_count > 0:
                    # 첫 번째 GPU만 빠른 체크
                    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    logger.info(f"✅ GPU 감지: {device_count}개 디바이스, 주 GPU {memory_gb:.1f}GB")
                    
                    if memory_gb >= self.gpu_requirements['memory_gb']:
                        logger.info(f"✅ GPU 메모리 충분")
                    else:
                        logger.warning(f"⚠️ GPU 메모리 부족 가능성")
                        
                signal.alarm(0)
                return True
            else:
                logger.warning("⚠️ CUDA 불가능 - CPU 모드로 진행")
                signal.alarm(0)
                return False
                
        except Exception as e:
            logger.warning(f"⚠️ GPU 확인 실패: {e}")
            signal.alarm(0)
            return False
    
    def _validate_filesystem_fast(self) -> bool:
        """파일 시스템 빠른 권한 확인"""
        logger.info("📁 파일 시스템 빠른 확인 중...")
        
        try:
            import tempfile
            import os
            
            # 임시 파일 생성 테스트
            with tempfile.NamedTemporaryFile(delete=True) as tmp:
                tmp.write(b"test")
                tmp.flush()
                
            logger.info("✅ 파일 시스템 쓰기 권한 OK")
            return True
                
        except Exception as e:
            logger.warning(f"⚠️ 파일 시스템 권한 문제: {e}")
            return False
    
    def _validate_memory_fast(self) -> bool:
        """메모리 빠른 확인"""
        logger.info("💾 메모리 빠른 확인 중...")
        
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            available_gb = memory.available / 1024**3
            
            if available_gb >= 2.0:  # 최소 2GB 필요
                logger.info(f"✅ 사용 가능 메모리: {available_gb:.1f}GB")
                return True
            else:
                logger.warning(f"⚠️ 메모리 부족: {available_gb:.1f}GB < 2.0GB")
                return False
                
        except ImportError:
            logger.warning("⚠️ psutil 없음 - 메모리 확인 스킵")
            return True  # psutil 없어도 진행
        except Exception as e:
            logger.warning(f"⚠️ 메모리 확인 실패: {e}")
            return True  # 오류 무시하고 진행
            
    def _validate_filesystem(self) -> bool:
        """파일 시스템 권한 확인"""
        logger.info("📁 파일 시스템 권한 확인 중...")
        
        try:
            # 기본 디렉토리들 확인
            from config import DATA_DIR, MODELS_DIR, LOGS_DIR
            
            directories = [DATA_DIR, MODELS_DIR, LOGS_DIR]
            
            for directory in directories:
                if not directory.exists():
                    logger.warning(f"⚠️  디렉토리가 존재하지 않음: {directory}")
                    directory.mkdir(parents=True, exist_ok=True)
                    logger.info(f"✅ 디렉토리 생성됨: {directory}")
                    
                # 읽기/쓰기 권한 테스트
                test_file = directory / '.permission_test'
                try:
                    test_file.write_text('test')
                    test_file.unlink()
                    logger.info(f"✅ {directory}: 읽기/쓰기 권한 확인")
                except Exception as e:
                    logger.error(f"❌ {directory}: 권한 없음 - {e}")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"❌ 파일 시스템 확인 실패: {e}")
            return False
            
    def _validate_memory(self) -> bool:
        """메모리 확인"""
        logger.info("💾 시스템 메모리 확인 중...")
        
        try:
            import psutil
            
            # 총 메모리
            total_memory_gb = psutil.virtual_memory().total / 1024**3
            available_memory_gb = psutil.virtual_memory().available / 1024**3
            
            logger.info(f"💾 총 메모리: {total_memory_gb:.1f}GB")
            logger.info(f"💾 사용 가능 메모리: {available_memory_gb:.1f}GB")
            
            # 최소 4GB 필요
            min_required_gb = 4.0
            if available_memory_gb >= min_required_gb:
                logger.info(f"✅ 메모리 충분: {available_memory_gb:.1f}GB >= {min_required_gb}GB")
                return True
            else:
                logger.error(f"❌ 메모리 부족: {available_memory_gb:.1f}GB < {min_required_gb}GB")
                return False
                
        except ImportError:
            logger.warning("⚠️  psutil이 설치되지 않아 메모리 확인을 건너뜁니다.")
            return True
        except Exception as e:
            logger.warning(f"⚠️  메모리 확인 중 오류: {e}")
            return True
            
    def get_validation_report(self) -> str:
        """검증 결과 리포트 생성"""
        report = ["=== Red Heart 의존성 검증 리포트 ==="]
        
        # Python 버전
        report.append(f"Python: {sys.version}")
        
        # 패키지별 상태
        report.append("\n📦 패키지 상태:")
        for package, result in self.validation_results.items():
            status = result['status']
            if status == 'success':
                report.append(f"  ✅ {package}: {result['version']}")
            elif status == 'import_failed':
                report.append(f"  ❌ {package}: import 실패")
            else:
                report.append(f"  ⚠️  {package}: {result.get('error', 'unknown error')}")
                
        return "\n".join(report)


def validate_dependencies() -> bool:
    """의존성 검증 실행"""
    validator = DependencyValidator()
    return validator.validate_all()


if __name__ == "__main__":
    # 독립 실행
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    )
    
    success = validate_dependencies()
    
    if success:
        print("\n🎉 의존성 검증 성공! Red Heart 시스템을 시작할 수 있습니다.")
        sys.exit(0)
    else:
        print("\n💥 의존성 검증 실패! 문제를 해결한 후 다시 시도하세요.")
        sys.exit(1)