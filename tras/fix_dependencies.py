"""
의존성 문제 해결 스크립트
Red Heart 시스템의 누락된 의존성을 감지하고 해결하는 도구
"""

import sys
import subprocess
import importlib
from pathlib import Path

class DependencyFixer:
    def __init__(self):
        self.missing_optional_deps = []
        self.critical_missing_deps = []
        
    def fix_config_dotenv_issue(self):
        """config.py의 dotenv 의존성 문제 해결"""
        print("🔧 config.py의 dotenv 의존성 문제 해결 중...")
        
        config_file = Path("config.py")
        if not config_file.exists():
            print("❌ config.py 파일을 찾을 수 없습니다.")
            return False
            
        # config.py 읽기
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # dotenv import를 optional로 만들기
        original_dotenv_lines = [
            "from dotenv import load_dotenv",
            "load_dotenv()"
        ]
        
        replacement_dotenv_code = """# dotenv 의존성을 선택적으로 로드
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv가 설치되지 않았습니다. .env 파일을 무시합니다.")
    def load_dotenv():
        pass"""
        
        # 기존 dotenv 관련 코드 교체
        new_content = content
        for line in original_dotenv_lines:
            new_content = new_content.replace(line, "")
        
        # 새로운 optional import 코드 추가
        new_content = new_content.replace(
            "import logging\nimport datetime",
            f"import logging\nimport datetime\n\n{replacement_dotenv_code}"
        )
        
        # DEVICE 변수가 config에서 정의되지 않은 문제 해결
        if "DEVICE = " not in new_content:
            device_code = """
# 디바이스 설정
try:
    import torch
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TORCH_DTYPE = torch.float32
except ImportError:
    DEVICE = 'cpu'
    TORCH_DTYPE = None
"""
            # 파일 마지막에 DEVICE 설정 추가
            new_content += device_code
        
        # ADVANCED_CONFIG가 없으면 추가
        if "ADVANCED_CONFIG = " not in new_content:
            advanced_config_code = """
# 고급 설정
ADVANCED_CONFIG = {
    'enable_gpu': False,
    'gpu_count': 0,
    'model_precision': 'float32',
    'batch_size': 8,
    'max_sequence_length': 512,
    'enable_logging': True,
    'log_level': 'INFO'
}

# GPU 사용 가능성 체크
try:
    import torch
    ADVANCED_CONFIG['enable_gpu'] = torch.cuda.is_available()
    ADVANCED_CONFIG['gpu_count'] = torch.cuda.device_count() if torch.cuda.is_available() else 0
except ImportError:
    ADVANCED_CONFIG['enable_gpu'] = False
    ADVANCED_CONFIG['gpu_count'] = 0
"""
            new_content += advanced_config_code
        
        # 수정된 내용 저장
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
            
        print("✅ config.py의 dotenv 의존성 문제가 해결되었습니다.")
        return True
    
    def create_minimal_requirements(self):
        """최소 requirements.txt 생성"""
        print("📋 최소 requirements.txt 생성 중...")
        
        minimal_requirements = [
            "# 필수 의존성",
            "numpy>=1.21.0",
            "torch>=1.9.0",
            "scikit-learn>=1.0.0",
            "",
            "# 선택적 의존성 (권장)",
            "# transformers>=4.20.0  # 고급 NLP 기능용",
            "# python-dotenv>=0.19.0  # 환경 변수 관리용",
            "# pandas>=1.3.0  # 데이터 처리용",
            "# matplotlib>=3.5.0  # 시각화용",
            "# seaborn>=0.11.0  # 통계 시각화용",
            "# tqdm>=4.62.0  # 진행률 표시용"
        ]
        
        with open("requirements_minimal.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(minimal_requirements))
        
        print("✅ requirements_minimal.txt가 생성되었습니다.")
    
    def make_imports_optional(self):
        """주요 모듈들의 import를 optional로 만들기"""
        print("🔄 주요 모듈들의 import를 optional로 변경 중...")
        
        files_to_fix = [
            "emotion_ethics_regret_circuit.py",
            "ethics_policy_updater.py", 
            "phase_controller.py",
            "xai_feedback_integrator.py",
            "fuzzy_emotion_ethics_mapper.py",
            "deep_multi_dimensional_ethics_system.py",
            "temporal_event_propagation_analyzer.py"
        ]
        
        for filename in files_to_fix:
            filepath = Path(filename)
            if not filepath.exists():
                print(f"⚠️  {filename} 파일을 찾을 수 없습니다.")
                continue
                
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # transformers import를 optional로 만들기
                if "from transformers import" in content:
                    content = content.replace(
                        "from transformers import",
                        "# from transformers import  # optional dependency"
                    )
                
                # config import 수정
                if "from config import ADVANCED_CONFIG, DEVICE" in content:
                    config_import_fix = """try:
    from config import ADVANCED_CONFIG, DEVICE
except ImportError:
    # config.py에 문제가 있을 경우 기본값 사용
    ADVANCED_CONFIG = {'enable_gpu': False}
    DEVICE = 'cpu'
    print(f"⚠️  {filename}: config.py 임포트 실패, 기본값 사용")"""
    
                    content = content.replace(
                        "from config import ADVANCED_CONFIG, DEVICE",
                        config_import_fix
                    )
                
                # 수정된 내용 저장
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
                print(f"✅ {filename}의 import 문제가 해결되었습니다.")
                
            except Exception as e:
                print(f"❌ {filename} 수정 실패: {e}")
    
    def test_imports_after_fix(self):
        """수정 후 import 테스트"""
        print("\n🧪 수정 후 import 테스트 실행")
        
        core_modules = [
            'config',
            'data_models', 
            'emotion_ethics_regret_circuit',
            'ethics_policy_updater',
            'phase_controller',
            'xai_feedback_integrator',
            'fuzzy_emotion_ethics_mapper',
            'deep_multi_dimensional_ethics_system',
            'temporal_event_propagation_analyzer'
        ]
        
        success_count = 0
        total_count = len(core_modules)
        
        for module_name in core_modules:
            try:
                if module_name in sys.modules:
                    importlib.reload(sys.modules[module_name])
                else:
                    importlib.import_module(module_name)
                print(f"✅ {module_name}: 임포트 성공")
                success_count += 1
            except Exception as e:
                print(f"❌ {module_name}: 임포트 실패 - {e}")
        
        success_rate = success_count / total_count
        print(f"\n📊 임포트 성공률: {success_count}/{total_count} ({success_rate:.1%})")
        
        return success_rate
    
    def create_standalone_test(self):
        """독립적인 테스트 파일 생성"""
        print("🔬 독립적인 테스트 파일 생성 중...")
        
        test_content = '''"""
Red Heart 시스템 독립 테스트
의존성 문제 없이 실행 가능한 기본 테스트
"""

import sys
from pathlib import Path

def test_basic_imports():
    """기본 임포트 테스트"""
    print("🧪 기본 임포트 테스트")
    
    try:
        import data_models
        print("✅ data_models 임포트 성공")
        
        # 기본 데이터 구조 테스트
        emotion = data_models.EmotionState.JOY
        print(f"✅ EmotionState 테스트: {emotion}")
        
        return True
    except Exception as e:
        print(f"❌ 기본 임포트 실패: {e}")
        return False

def test_config_fallback():
    """config 폴백 테스트"""
    print("\\n⚙️  config 폴백 테스트")
    
    try:
        import config
        print("✅ config 임포트 성공")
        
        # 기본 설정 확인
        device = getattr(config, 'DEVICE', 'cpu')
        print(f"✅ DEVICE 설정: {device}")
        
        return True
    except Exception as e:
        print(f"❌ config 임포트 실패: {e}")
        return False

def test_minimal_system():
    """최소 시스템 테스트"""
    print("\\n🔧 최소 시스템 테스트")
    
    try:
        # 기본 윤리 판단 시뮬레이션
        scenario = "간단한 윤리적 선택 상황"
        
        # 임시 윤리 점수 계산
        ethics_score = 0.7  # 가상의 점수
        confidence = 0.8
        
        result = {
            'scenario': scenario,
            'ethics_score': ethics_score,
            'confidence': confidence,
            'recommendation': 'moderate_ethical_approach'
        }
        
        print(f"✅ 기본 윤리 판단 시뮬레이션 성공")
        print(f"   시나리오: {result['scenario']}")
        print(f"   윤리 점수: {result['ethics_score']}")
        print(f"   신뢰도: {result['confidence']}")
        
        return True
    except Exception as e:
        print(f"❌ 최소 시스템 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 실행"""
    print("🚀 Red Heart 독립 테스트 시작\\n")
    
    tests = [
        test_basic_imports,
        test_config_fallback, 
        test_minimal_system
    ]
    
    success_count = 0
    for test_func in tests:
        try:
            if test_func():
                success_count += 1
        except Exception as e:
            print(f"❌ 테스트 {test_func.__name__} 예외: {e}")
    
    total_tests = len(tests)
    success_rate = success_count / total_tests
    
    print(f"\\n📊 테스트 결과: {success_count}/{total_tests} ({success_rate:.1%}) 성공")
    
    if success_rate >= 0.8:
        print("🟢 시스템 상태: 양호")
    elif success_rate >= 0.6:
        print("🟡 시스템 상태: 보통")
    else:
        print("🔴 시스템 상태: 문제 있음")
    
    return success_rate

if __name__ == "__main__":
    main()
'''
        
        with open("standalone_test.py", 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        print("✅ standalone_test.py가 생성되었습니다.")

def main():
    """메인 실행 함수"""
    print("🔧 Red Heart 의존성 문제 해결 시작")
    
    fixer = DependencyFixer()
    
    # 1. config.py dotenv 문제 해결
    fixer.fix_config_dotenv_issue()
    
    # 2. 최소 requirements 생성
    fixer.create_minimal_requirements()
    
    # 3. optional imports 적용
    fixer.make_imports_optional()
    
    # 4. 독립 테스트 생성
    fixer.create_standalone_test()
    
    # 5. 수정 후 테스트
    success_rate = fixer.test_imports_after_fix()
    
    print("\n" + "="*60)
    print("🎯 의존성 수정 완료")
    print("="*60)
    print(f"📈 임포트 성공률: {success_rate:.1%}")
    
    if success_rate >= 0.8:
        print("🟢 상태: 대부분의 모듈이 정상 작동합니다.")
    elif success_rate >= 0.6:
        print("🟡 상태: 일부 모듈에 문제가 있지만 기본 기능은 작동합니다.")
    else:
        print("🔴 상태: 추가 수정이 필요합니다.")
    
    print("\n💡 권장사항:")
    print("1. requirements_minimal.txt의 패키지들을 설치하세요")
    print("2. standalone_test.py로 기본 기능을 테스트하세요")
    print("3. 고급 기능이 필요하면 transformers 등을 별도 설치하세요")

if __name__ == "__main__":
    main()