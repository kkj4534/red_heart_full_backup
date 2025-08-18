"""
구문 오류 수정 스크립트
중복된 try-except 구문과 기타 syntax error 수정
"""

import os
import re
from pathlib import Path

def fix_syntax_errors():
    """구문 오류 수정"""
    print("🔧 구문 오류 수정 시작...")
    
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
            fix_file_syntax(file_path)
        else:
            print(f"⚠️  {file_path} 파일을 찾을 수 없습니다.")

def fix_file_syntax(file_path):
    """파일의 구문 오류 수정"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 1. 중복된 try-except 블록 수정
        # 패턴: try:\n    try:\n    from config ... except ...\nexcept ... 
        pattern = r'try:\s*try:\s*from config import ADVANCED_CONFIG, DEVICE\s*except ImportError:\s*# config\.py에 문제가 있을 경우 기본값 사용\s*ADVANCED_CONFIG = \{\'enable_gpu\': False\}\s*DEVICE = \'cpu\'\s*print\(f"⚠️\s+\{filename\}: config\.py 임포트 실패, 기본값 사용"\)\s*except ImportError:\s*# config\.py에 문제가 있을 경우 기본값 사용\s*ADVANCED_CONFIG = \{\'enable_gpu\': False\}\s*DEVICE = \'cpu\'\s*print\(f"⚠️\s+\{filename\}: config\.py 임포트 실패, 기본값 사용"\)'
        
        # 간단한 수정: 중복된 try-except 구조를 단일 구조로 변경
        fixed_import = """try:
    from config import ADVANCED_CONFIG, DEVICE
except ImportError:
    # config.py에 문제가 있을 경우 기본값 사용
    ADVANCED_CONFIG = {'enable_gpu': False}
    DEVICE = 'cpu'
    print(f"⚠️  config.py 임포트 실패, 기본값 사용")"""
        
        # 중복된 try 블록 찾기 및 수정
        lines = content.split('\n')
        new_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # try: 블록이 연속으로 나오는 경우 체크
            if (line.strip().startswith('try:') and 
                i + 1 < len(lines) and 
                lines[i + 1].strip().startswith('try:')):
                
                # 중복된 try 블록 건너뛰고 수정된 버전 삽입
                new_lines.append(fixed_import)
                
                # 중복된 블록의 끝까지 건너뛰기
                try_count = 0
                j = i
                while j < len(lines):
                    if 'try:' in lines[j]:
                        try_count += 1
                    if 'except ImportError:' in lines[j]:
                        try_count -= 1
                        if try_count == 0:
                            # 마지막 except 블록 다음까지 건너뛰기
                            while (j < len(lines) and 
                                   (lines[j].strip().startswith('ADVANCED_CONFIG') or
                                    lines[j].strip().startswith('DEVICE') or
                                    lines[j].strip().startswith('print(') or
                                    lines[j].strip() == '' or
                                    'config.py 임포트 실패' in lines[j])):
                                j += 1
                            break
                    j += 1
                
                i = j
                continue
            
            new_lines.append(line)
            i += 1
        
        content = '\n'.join(new_lines)
        
        # 2. 기타 syntax 오류 수정
        # 누락된 newline 문제 수정
        if not content.endswith('\n'):
            content += '\n'
        
        # LOGS_DIR 관련 오류 수정
        content = content.replace('), LOGS_DIR', ')')
        content = content.replace('print(f"⚠️  {filename}: config.py 임포트 실패, 기본값 사용")', 
                                'print("⚠️  config.py 임포트 실패, 기본값 사용")')
        
        # 변경사항이 있으면 파일 저장
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ {file_path} 구문 오류 수정 완료")
        else:
            print(f"➡️  {file_path} 수정 불필요")
            
    except Exception as e:
        print(f"❌ {file_path} 수정 실패: {e}")

def test_syntax_after_fix():
    """수정 후 구문 검사"""
    print("\n🧪 수정 후 구문 검사...")
    
    test_files = [
        "deep_multi_dimensional_ethics_system.py",
        "temporal_event_propagation_analyzer.py", 
        "xai_feedback_integrator.py",
        "fuzzy_emotion_ethics_mapper.py"
    ]
    
    success_count = 0
    for file_path in test_files:
        if os.path.exists(file_path):
            try:
                # 구문 검사 (컴파일만)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                compile(content, file_path, 'exec')
                print(f"✅ {file_path}: 구문 오류 없음")
                success_count += 1
                
            except SyntaxError as e:
                print(f"❌ {file_path}: 구문 오류 - {e}")
            except Exception as e:
                print(f"⚠️  {file_path}: 검사 실패 - {e}")
    
    print(f"\n📊 구문 검사 결과: {success_count}/{len(test_files)} 파일 통과")
    return success_count / len(test_files)

if __name__ == "__main__":
    fix_syntax_errors()
    test_syntax_after_fix()