#!/usr/bin/env python3
"""
Red Heart 통합 검증 및 로깅 테스트 스크립트
Integration verification and logging test for Red Heart system
"""

import os
import sys
import json
import logging
import importlib
from datetime import datetime
from pathlib import Path

def setup_basic_logging():
    """기본 로깅 설정"""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 메인 로거
    main_logger = logging.getLogger('RedHeart.Main')
    main_logger.setLevel(logging.INFO)
    
    # 학습 로거  
    learning_logger = logging.getLogger('RedHeart.Learning')
    learning_logger.setLevel(logging.INFO)
    
    # 로그 파일 핸들러
    main_handler = logging.FileHandler(f"logs/main_verification_{timestamp}.log", encoding='utf-8')
    learning_handler = logging.FileHandler(f"logs/learning_verification_{timestamp}.log", encoding='utf-8')
    
    # 포맷터
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    main_handler.setFormatter(formatter)
    learning_handler.setFormatter(formatter)
    
    main_logger.addHandler(main_handler)
    learning_logger.addHandler(learning_handler)
    
    # 콘솔 출력도 추가
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    main_logger.addHandler(console_handler)
    
    return main_logger, learning_logger

def verify_module_imports():
    """모듈 임포트 검증"""
    print("=== 모듈 임포트 검증 시작 ===")
    
    import_results = {}
    
    # 기본 모듈들
    basic_modules = [
        'config',
        'data_models', 
        'utils'
    ]
    
    # 고급 모듈들
    advanced_modules = [
        'advanced_emotion_analyzer',
        'advanced_bentham_calculator', 
        'advanced_semantic_analyzer',
        'advanced_surd_analyzer',
        'advanced_hierarchical_emotion_system',
        'advanced_regret_learning_system',
        'advanced_bayesian_inference_module',
        'advanced_llm_integration_layer',
        'advanced_counterfactual_reasoning'
    ]
    
    all_modules = basic_modules + advanced_modules
    
    for module_name in all_modules:
        try:
            importlib.import_module(module_name)
            import_results[module_name] = "SUCCESS"
            print(f"✅ {module_name} - 임포트 성공")
        except ImportError as e:
            import_results[module_name] = f"FAILED: {str(e)}"
            print(f"❌ {module_name} - 임포트 실패: {e}")
        except Exception as e:
            import_results[module_name] = f"ERROR: {str(e)}"
            print(f"⚠️  {module_name} - 오류: {e}")
    
    return import_results

def simulate_regret_learning_logging():
    """후회 학습 로깅 시뮬레이션"""
    print("\n=== 후회 학습 로깅 시뮬레이션 ===")
    
    main_logger, learning_logger = setup_basic_logging()
    
    # 시뮬레이션 데이터
    regret_scenarios = [
        {"phase": "PHASE_0", "regret_type": "ACTION", "regret_value": 0.15, "scenario": "도덕적 딜레마 1"},
        {"phase": "PHASE_0", "regret_type": "INACTION", "regret_value": 0.23, "scenario": "윤리적 선택 1"},
        {"phase": "PHASE_1", "regret_type": "TIMING", "regret_value": 0.31, "scenario": "사회적 갈등 1"},
        {"phase": "PHASE_1", "regret_type": "EMPATHY", "regret_value": 0.18, "scenario": "감정적 상황 1"},
        {"phase": "PHASE_2", "regret_type": "PREDICTION", "regret_value": 0.09, "scenario": "예측 오류 1"}
    ]
    
    learning_progress = []
    
    for i, scenario in enumerate(regret_scenarios, 1):
        # 메인 로거에 진행 현황
        main_logger.info(f"학습 시나리오 {i}/5 진행 중: {scenario['scenario']}")
        
        # 학습 로거에 상세 후회 분석
        learning_logger.info(
            f"후회 학습 진행 | Phase: {scenario['phase']} | "
            f"후회 유형: {scenario['regret_type']} | "
            f"후회 강도: {scenario['regret_value']:.3f} | "
            f"시나리오: {scenario['scenario']}"
        )
        
        # 상세 분석 로깅
        if scenario['regret_value'] > 0.3:
            learning_logger.warning(
                f"높은 후회 감지 - 페이즈 전환 고려 필요 | "
                f"현재 후회값: {scenario['regret_value']:.3f} > 임계값: 0.30"
            )
            
        # 진행 상황 기록
        progress_data = {
            "step": i,
            "timestamp": datetime.now().isoformat(),
            "phase": scenario['phase'],
            "regret_type": scenario['regret_type'],
            "regret_value": scenario['regret_value'],
            "scenario": scenario['scenario']
        }
        learning_progress.append(progress_data)
        
        print(f"  📊 Step {i}: {scenario['regret_type']} regret = {scenario['regret_value']:.3f}")
    
    # 학습 요약 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"logs/learning_summary_{timestamp}.json"
    
    summary_data = {
        "verification_timestamp": datetime.now().isoformat(),
        "total_scenarios": len(regret_scenarios),
        "average_regret": sum(s['regret_value'] for s in regret_scenarios) / len(regret_scenarios),
        "max_regret": max(s['regret_value'] for s in regret_scenarios),
        "min_regret": min(s['regret_value'] for s in regret_scenarios),
        "phase_transitions": len([s for s in regret_scenarios if s['regret_value'] > 0.3]),
        "learning_progress": learning_progress
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    
    main_logger.info(f"학습 요약 저장 완료: {summary_file}")
    print(f"📄 학습 요약 파일 생성: {summary_file}")
    
    return summary_data

def main():
    """메인 검증 함수"""
    print("🚀 Red Heart 시스템 통합 검증 시작")
    print("=" * 50)
    
    # 1. 모듈 임포트 검증
    import_results = verify_module_imports()
    
    # 2. 로깅 시스템 테스트
    learning_summary = simulate_regret_learning_logging()
    
    # 3. 최종 결과 출력
    print("\n=== 검증 결과 요약 ===")
    successful_imports = sum(1 for result in import_results.values() if result == "SUCCESS")
    total_modules = len(import_results)
    
    print(f"모듈 임포트: {successful_imports}/{total_modules} 성공")
    print(f"평균 후회값: {learning_summary['average_regret']:.3f}")
    print(f"페이즈 전환 후보: {learning_summary['phase_transitions']}개 시나리오")
    
    if successful_imports == total_modules:
        print("✅ 모든 모듈이 성공적으로 통합되었습니다!")
    else:
        print("⚠️  일부 모듈에 문제가 있습니다. 로그를 확인하세요.")
    
    print("📊 상세 로그는 logs/ 디렉터리에서 확인할 수 있습니다.")

if __name__ == "__main__":
    main()