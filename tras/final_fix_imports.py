"""
최종 임포트 구문 수정
중복된 except 블록 완전 제거 및 정리
"""

import os
import re
import importlib

def fix_final_imports():
    """최종 임포트 수정"""
    print("🔧 최종 임포트 수정 시작...")
    
    files_to_fix = [
        "deep_multi_dimensional_ethics_system.py",
        "temporal_event_propagation_analyzer.py", 
        "xai_feedback_integrator.py",
        "fuzzy_emotion_ethics_mapper.py",
        "emotion_ethics_regret_circuit.py",
        "ethics_policy_updater.py",
        "phase_controller.py"
    ]
    
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            fix_imports_in_file(file_path)

def fix_imports_in_file(file_path):
    """파일의 임포트 수정"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 정리된 임포트 블록
        clean_import_block = """try:
    from config import ADVANCED_CONFIG, DEVICE
except ImportError:
    # config.py에 문제가 있을 경우 기본값 사용
    ADVANCED_CONFIG = {'enable_gpu': False}
    DEVICE = 'cpu'
    print("⚠️  config.py 임포트 실패, 기본값 사용")"""
        
        # 중복된 블록을 찾아서 교체
        # 패턴: try: ... except: ... except: ...
        pattern = r'try:\s+from config import ADVANCED_CONFIG, DEVICE\s+except ImportError:\s+# config\.py에 문제가 있을 경우 기본값 사용\s+ADVANCED_CONFIG = \{\'enable_gpu\': False\}\s+DEVICE = \'cpu\'\s+print\(f?"⚠️\s+.*?config\.py 임포트 실패.*?"\)\s+except ImportError:\s+# config\.py에 문제가 있을 경우 기본값 사용\s+ADVANCED_CONFIG = \{\'enable_gpu\': False\}\s+DEVICE = \'cpu\'\s+print\("⚠️\s+.*?config\.py 임포트 실패.*?"\)'
        
        # 정규식이 복잡하므로 문자열 교체 방식 사용
        if 'except ImportError:' in content:
            # 중복된 except 블록 찾기
            lines = content.split('\n')
            new_lines = []
            i = 0
            
            while i < len(lines):
                line = lines[i]
                
                # config import를 찾으면
                if line.strip() == 'try:' and i + 1 < len(lines) and 'from config import' in lines[i + 1]:
                    # 전체 try-except 블록을 새로운 블록으로 교체
                    new_lines.extend(clean_import_block.split('\n'))
                    
                    # 기존 블록 건너뛰기
                    while i < len(lines) and not lines[i].strip().startswith('from data_models'):
                        i += 1
                    
                    # data_models import 직전까지 건너뛰었으므로 한 줄 뒤로
                    i -= 1
                else:
                    new_lines.append(line)
                
                i += 1
            
            content = '\n'.join(new_lines)
        
        # 기타 문제들 수정
        content = content.replace('), DATA_DIR, EXPERIENCE_DB_DIR', ')')
        content = content.replace('print(f"⚠️  config.py 임포트 실패, 기본값 사용")', 
                                'print("⚠️  config.py 임포트 실패, 기본값 사용")')
        
        # 파일 끝에 newline 추가
        if not content.endswith('\n'):
            content += '\n'
        
        # 변경사항이 있으면 저장
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ {file_path} 최종 수정 완료")
        else:
            print(f"➡️  {file_path} 수정 불필요")
            
    except Exception as e:
        print(f"❌ {file_path} 수정 실패: {e}")

def test_final_imports():
    """최종 임포트 테스트"""
    print("\n🧪 최종 임포트 테스트...")
    
    test_files = [
        "data_models",
        "config", 
        "emotion_ethics_regret_circuit",
        "ethics_policy_updater",
        "phase_controller",
        "xai_feedback_integrator",
        "fuzzy_emotion_ethics_mapper",
        "deep_multi_dimensional_ethics_system",
        "temporal_event_propagation_analyzer"
    ]
    
    success_count = 0
    for module_name in test_files:
        try:
            importlib.import_module(module_name)
            print(f"✅ {module_name}: 임포트 성공")
            success_count += 1
        except Exception as e:
            print(f"❌ {module_name}: 임포트 실패 - {str(e)[:100]}")
    
    print(f"\n📊 최종 임포트 성공률: {success_count}/{len(test_files)} ({success_count/len(test_files):.1%})")
    return success_count / len(test_files)

if __name__ == "__main__":
    fix_final_imports()
    test_final_imports()