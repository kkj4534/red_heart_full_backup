#!/usr/bin/env python3
"""
통합 구현 검증 스크립트
모든 컴포넌트가 제대로 연결되었는지 확인
"""

import os
import sys
import importlib.util
from pathlib import Path

# 색상 코드
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def check_module(module_path, class_name):
    """모듈 존재 및 클래스 확인"""
    if not Path(module_path).exists():
        return False, f"파일 없음: {module_path}"
    
    try:
        # 동적 임포트
        spec = importlib.util.spec_from_file_location("module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # 클래스 확인
        if hasattr(module, class_name):
            return True, f"✅ {class_name} 클래스 확인"
        else:
            return False, f"클래스 없음: {class_name}"
    except Exception as e:
        return False, f"임포트 오류: {str(e)}"

def main():
    print(f"{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}Red Heart AI 통합 구현 검증{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")
    
    # 검증할 컴포넌트 목록
    components = [
        ("training/enhanced_checkpoint_manager.py", "EnhancedCheckpointManager"),
        ("training/lr_sweep_optimizer.py", "LRSweepOptimizer"),
        ("training/sweet_spot_detector.py", "SweetSpotDetector"),
        ("training/parameter_crossover_system.py", "ParameterCrossoverSystem"),
        ("training/oom_handler.py", "OOMHandler"),
        ("training/advanced_training_techniques.py", "AdvancedTrainingManager"),
        ("training/unified_training_final.py", "UnifiedTrainer"),
        ("training/test_unified_training.py", None),  # 스크립트만 확인
    ]
    
    print(f"{YELLOW}1. 컴포넌트 파일 확인:{RESET}")
    all_success = True
    for module_path, class_name in components:
        if class_name:
            success, message = check_module(module_path, class_name)
        else:
            success = Path(module_path).exists()
            message = "✅ 스크립트 존재" if success else "❌ 스크립트 없음"
        
        status = f"{GREEN}✅{RESET}" if success else f"{RED}❌{RESET}"
        print(f"  {status} {module_path.split('/')[-1]}: {message}")
        all_success = all_success and success
    
    print(f"\n{YELLOW}2. 통합 설정 확인:{RESET}")
    
    # unified_training_final.py 내용 확인
    config_checks = []
    try:
        with open("training/unified_training_final.py", "r") as f:
            content = f.read()
            
            # 중요 설정 확인
            checks = [
                ("lr_sweep_enabled = True", "LR 스윕 활성화"),
                ("enable_sweet_spot = True", "Sweet Spot 활성화"),
                ("enable_crossover = True", "Parameter Crossover 활성화"),
                ("enable_oom_handler = True", "OOM 핸들러 활성화"),
                ("enable_label_smoothing = True", "Label Smoothing 활성화"),
                ("enable_rdrop = True", "R-Drop 활성화"),
                ("enable_ema = True", "EMA 활성화"),
                ("enable_llrd = True", "LLRD 활성화"),
                ("total_epochs = 60", "60 에폭 설정"),
                ("micro_batch_size = 2", "배치 사이즈 2"),
                ("gradient_accumulation = 32", "GA 32 설정"),
                ("self.run_lr_sweep()", "LR 스윕 실행"),
                ("self.sweet_spot_detector.update", "Sweet Spot 업데이트"),
                ("self.checkpoint_manager.save_checkpoint", "체크포인트 저장"),
                ("self.crossover_system.perform_crossover", "Crossover 실행"),
            ]
            
            for check_str, desc in checks:
                if check_str in content:
                    config_checks.append((True, desc))
                    print(f"  {GREEN}✅{RESET} {desc}")
                else:
                    config_checks.append((False, desc))
                    print(f"  {RED}❌{RESET} {desc}")
                    all_success = False
    except Exception as e:
        print(f"  {RED}❌ 설정 확인 실패: {e}{RESET}")
        all_success = False
    
    print(f"\n{YELLOW}3. run_learning.sh 통합 확인:{RESET}")
    
    # run_learning.sh 확인
    try:
        with open("run_learning.sh", "r") as f:
            content = f.read()
            
            checks = [
                ("unified-test", "unified-test 모드"),
                ("unified-train", "unified-train 모드"),
                ("training/unified_training_final.py", "최종 시스템 경로"),
                ("SAMPLES", "SAMPLES 변수 처리"),
                ("--test --epochs ${SAMPLES:-3}", "샘플 수 인자 전달"),
                ("red_heart_env/bin/activate", "가상환경 활성화"),
            ]
            
            for check_str, desc in checks:
                if check_str in content:
                    print(f"  {GREEN}✅{RESET} {desc}")
                else:
                    print(f"  {RED}❌{RESET} {desc}")
                    all_success = False
    except Exception as e:
        print(f"  {RED}❌ run_learning.sh 확인 실패: {e}{RESET}")
        all_success = False
    
    print(f"\n{YELLOW}4. 명령어 시뮬레이션:{RESET}")
    
    # 테스트 명령어들
    test_commands = [
        ("bash run_learning.sh unified-test", "기본 테스트 (2 에폭)"),
        ("SAMPLES=3 bash run_learning.sh unified-test", "3 에폭 테스트"),
        ("bash run_learning.sh unified-test --debug --verbose", "디버그 모드"),
        ("bash run_learning.sh unified-train", "60 에폭 학습"),
        ("bash run_learning.sh unified-train --epochs 30", "30 에폭 학습"),
        ("nohup timeout 1200 bash run_learning.sh unified-test --samples 3 --debug --verbose", "nohup 백그라운드"),
    ]
    
    for cmd, desc in test_commands:
        print(f"  📝 {desc}")
        print(f"     {BLUE}{cmd}{RESET}")
    
    # 최종 결과
    print(f"\n{BLUE}{'='*60}{RESET}")
    if all_success:
        print(f"{GREEN}✅ 모든 컴포넌트가 정상적으로 통합되었습니다!{RESET}")
        print(f"\n{YELLOW}실행 가능한 명령:{RESET}")
        print(f"  1. 빠른 테스트: {BLUE}bash run_learning.sh unified-test{RESET}")
        print(f"  2. 샘플 테스트: {BLUE}SAMPLES=3 bash run_learning.sh unified-test{RESET}")
        print(f"  3. 전체 학습: {BLUE}bash run_learning.sh unified-train{RESET}")
    else:
        print(f"{RED}⚠️ 일부 컴포넌트에 문제가 있습니다. 위 내용을 확인하세요.{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")
    
    return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main())