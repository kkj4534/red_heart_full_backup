#!/usr/bin/env python3
"""
강제 클린업 및 재테스트 스크립트
웹 검색 결과 기반 완전한 해결 방법
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def complete_cleanup():
    """완전한 Python 캐시 정리"""
    print("🧹 완전한 Python 캐시 정리 시작...")
    
    base_dir = Path("/mnt/c/large_project/linux_red_heart")
    
    # 1. __pycache__ 디렉토리 삭제
    pycache_dirs = list(base_dir.rglob("__pycache__"))
    for pycache_dir in pycache_dirs:
        if pycache_dir.is_dir():
            print(f"   삭제: {pycache_dir}")
            shutil.rmtree(pycache_dir, ignore_errors=True)
    
    # 2. .pyc 파일 삭제
    pyc_files = list(base_dir.rglob("*.pyc"))
    for pyc_file in pyc_files:
        print(f"   삭제: {pyc_file}")
        pyc_file.unlink(missing_ok=True)
    
    # 3. .pyo 파일 삭제
    pyo_files = list(base_dir.rglob("*.pyo"))
    for pyo_file in pyo_files:
        print(f"   삭제: {pyo_file}")
        pyo_file.unlink(missing_ok=True)
    
    print("✅ Python 캐시 정리 완료")

def run_with_clean_environment():
    """클린 환경에서 테스트 실행"""
    print("\n🚀 클린 환경에서 통합 훈련 재실행...")
    
    # 환경 변수 설정 (바이트코드 생성 방지)
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PYTHONPATH'] = '/mnt/c/large_project/linux_red_heart'
    
    # 가상환경 활성화 및 실행
    cmd = [
        'bash', '-c',
        'source red_heart_env/bin/activate && python -B integrated_system_trainer.py'
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd='/mnt/c/large_project/linux_red_heart',
            env=env,
            capture_output=True,
            text=True,
            timeout=300  # 5분 타임아웃
        )
        
        print(f"📊 실행 결과:")
        print(f"   리턴 코드: {result.returncode}")
        print(f"   실행 완료: {'✅' if result.returncode == 0 else '❌'}")
        
        # 출력 분석
        output_lines = result.stdout.split('\n') + result.stderr.split('\n')
        
        # 중요한 로그 라인들 찾기
        important_lines = []
        for line in output_lines:
            if any(keyword in line for keyword in [
                '🎯 작업 유형', '💡 선호 모델', '✅ 선호 모델', 
                '⚠️ 선호 모델', '🔄 모델 RAM 스왑', '🦙 Llama.cpp',
                '성공률:', '손실:', '에포크', '평균'
            ]):
                important_lines.append(line)
        
        if important_lines:
            print(f"\n📋 중요 로그 (개선된 로깅 확인):")
            for line in important_lines[-10:]:  # 마지막 10줄만
                print(f"   {line}")
        
        # 성공률 체크
        success_rate_lines = [line for line in output_lines if '성공률:' in line]
        if success_rate_lines:
            last_success_rate = success_rate_lines[-1]
            print(f"\n🎯 최종 성공률: {last_success_rate}")
            
            if '0.00%' not in last_success_rate:
                print("🎉 성공률 개선 확인!")
                return True
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏰ 타임아웃 발생 (5분 초과)")
        return False
    except Exception as e:
        print(f"❌ 실행 오류: {e}")
        return False

if __name__ == "__main__":
    print("🔬 강제 클린업 및 재테스트")
    print("=" * 50)
    
    # 1. 완전한 캐시 정리
    complete_cleanup()
    
    # 2. 클린 환경에서 실행
    success = run_with_clean_environment()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 클린업 후 테스트 성공!")
    else:
        print("💥 클린업 후에도 문제 지속")
        
    exit(0 if success else 1)