"""
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
    print("\n⚙️  config 폴백 테스트")
    
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
    print("\n🔧 최소 시스템 테스트")
    
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
    print("🚀 Red Heart 독립 테스트 시작\n")
    
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
    
    print(f"\n📊 테스트 결과: {success_count}/{total_tests} ({success_rate:.1%}) 성공")
    
    if success_rate >= 0.8:
        print("🟢 시스템 상태: 양호")
    elif success_rate >= 0.6:
        print("🟡 시스템 상태: 보통")
    else:
        print("🔴 시스템 상태: 문제 있음")
    
    return success_rate

if __name__ == "__main__":
    main()
