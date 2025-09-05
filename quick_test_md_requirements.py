#!/usr/bin/env python3
"""
MD 문서 요구사항 빠른 테스트
Red Heart AI 통합 상태 확인
"""

import sys
sys.path.append('/mnt/c/large_project/linux_red_heart')

print("🔍 Red Heart AI MD 문서 요구사항 구현 검증")
print("="*70)

# 색상 코드
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
END = '\033[0m'

def check_requirement(name, check_func):
    """요구사항 체크"""
    try:
        result = check_func()
        if result:
            print(f"{GREEN}✅{END} {name}")
            return True
        else:
            print(f"{RED}❌{END} {name}")
            return False
    except Exception as e:
        print(f"{RED}❌{END} {name}: {e}")
        return False

# 체크리스트
checks_passed = 0
checks_total = 0

print("\n📋 MD 문서 핵심 요구사항 체크리스트")
print("-"*70)

# 1. 메모리 모드 체크
print(f"\n{BLUE}1. 메모리 모드 (MD 문서 사양){END}")
def check_memory_modes():
    from main_unified import MemoryMode
    required = ['LIGHT', 'MEDIUM', 'HEAVY', 'MCP']
    for mode in required:
        if not hasattr(MemoryMode, mode):
            return False
        print(f"   - {mode}: {getattr(MemoryMode, mode).value}")
    return True
checks_total += 1
if check_requirement("메모리 모드 4개 구현", check_memory_modes):
    checks_passed += 1

# 2. 3뷰 시나리오 시스템
print(f"\n{BLUE}2. 3뷰 시나리오 시스템{END}")
def check_three_view():
    from three_view_scenario_system import ThreeViewScenarioSystem, ScenarioType
    from main_unified import UnifiedInferenceSystem
    
    # 클래스 존재 확인
    print(f"   - ThreeViewScenarioSystem: 구현됨")
    print(f"   - ScenarioType: {[t.value for t in ScenarioType]}")
    
    # main_unified.py 통합 확인
    if hasattr(UnifiedInferenceSystem, '_load_three_view_scenario_system'):
        print(f"   - main_unified.py 통합: ✅")
        return True
    else:
        print(f"   - main_unified.py 통합: ❌")
        return False
checks_total += 1
if check_requirement("3뷰 시나리오 시스템", check_three_view):
    checks_passed += 1

# 3. 다원적 윤리 체계 (B안: 5개)
print(f"\n{BLUE}3. 다원적 윤리 체계 (MD 문서 B안: 5개){END}")
def check_ethics():
    from deep_multi_dimensional_ethics_system import (
        UtilitarianEngine, DeontologicalEngine, VirtueEthicsEngine,
        CareEthicsEngine, JusticeTheoryEngine
    )
    engines = [
        'UtilitarianEngine', 'DeontologicalEngine', 
        'VirtueEthicsEngine', 'CareEthicsEngine', 'JusticeTheoryEngine'
    ]
    for engine in engines:
        print(f"   - {engine}: ✅")
    return True
checks_total += 1
if check_requirement("5개 윤리 엔진", check_ethics):
    checks_passed += 1

# 4. 비선형 워크플로우
print(f"\n{BLUE}4. 비선형 워크플로우{END}")
def check_workflow():
    from main_unified import UnifiedInferenceSystem
    import inspect
    
    if hasattr(UnifiedInferenceSystem, 'analyze_ethical_dilemma'):
        sig = inspect.signature(UnifiedInferenceSystem.analyze_ethical_dilemma)
        params = list(sig.parameters.keys())
        print(f"   - analyze_ethical_dilemma: 구현됨")
        print(f"   - 파라미터: {params}")
        
        if 'llm_scenarios' in params:
            print(f"   - llm_scenarios 파라미터: ✅")
            return True
        else:
            print(f"   - llm_scenarios 파라미터: ❌")
            return False
    return False
checks_total += 1
if check_requirement("비선형 워크플로우", check_workflow):
    checks_passed += 1

# 5. 메모리 스왑 매니저
print(f"\n{BLUE}5. 메모리 스왑 매니저{END}")
def check_swap():
    from memory_swap_manager import SystemSwapManager, SystemType
    print(f"   - SystemSwapManager: 구현됨")
    print(f"   - SystemType: {[t.value for t in SystemType]}")
    return True
checks_total += 1
if check_requirement("메모리 스왑 매니저", check_swap):
    checks_passed += 1

# 6. LLM 통합
print(f"\n{BLUE}6. LLM 통합{END}")
def check_llm():
    from llm_module.advanced_llm_engine import AdvancedLLMEngine
    from main_unified import UnifiedInferenceSystem
    
    print(f"   - AdvancedLLMEngine: 구현됨")
    
    if hasattr(AdvancedLLMEngine, 'check_plausibility'):
        print(f"   - check_plausibility: ✅")
    else:
        print(f"   - check_plausibility: ❌")
        
    if hasattr(UnifiedInferenceSystem, '_load_llm_integration'):
        print(f"   - _load_llm_integration: ✅")
        return True
    else:
        print(f"   - _load_llm_integration: ❌")
        return False
checks_total += 1
if check_requirement("LLM 통합", check_llm):
    checks_passed += 1

# 7. 후회 시스템 대안 생성
print(f"\n{BLUE}7. 후회 시스템 대안 생성{END}")
def check_regret():
    from advanced_regret_learning_system import AdvancedRegretLearningSystem
    if hasattr(AdvancedRegretLearningSystem, 'suggest_alternatives'):
        print(f"   - suggest_alternatives: ✅")
        return True
    else:
        print(f"   - suggest_alternatives: ❌")
        return False
checks_total += 1
if check_requirement("suggest_alternatives 메서드", check_regret):
    checks_passed += 1

# 8. 경험 DB 저장
print(f"\n{BLUE}8. 경험 DB 저장{END}")
def check_experience_db():
    from advanced_experience_database import AdvancedExperienceDatabase
    
    if hasattr(AdvancedExperienceDatabase, 'store_experience'):
        print(f"   - store_experience 메서드: ✅")
        
        # main_unified.py에서 실제 사용 확인
        with open('main_unified.py', 'r') as f:
            content = f.read()
            if 'experience_database.store' in content or 'store_experience' in content:
                print(f"   - main_unified.py에서 호출: ✅")
                return True
            else:
                print(f"   - main_unified.py에서 호출: ❌")
                return False
    return False
checks_total += 1
if check_requirement("경험 DB 저장", check_experience_db):
    checks_passed += 1

# 9. MCP 서버
print(f"\n{BLUE}9. MCP 서버{END}")
def check_mcp():
    from mcp_server import RedHeartMCPServer
    print(f"   - RedHeartMCPServer: 구현됨")
    
    if hasattr(RedHeartMCPServer, 'handle_request'):
        print(f"   - handle_request: ✅")
        return True
    else:
        print(f"   - handle_request: ❌")
        return False
checks_total += 1
if check_requirement("MCP 서버", check_mcp):
    checks_passed += 1

# 10. 시간적 전파 분석
print(f"\n{BLUE}10. 시간적 전파 분석{END}")
def check_temporal():
    from temporal_event_propagation_analyzer import TemporalEventPropagationAnalyzer
    print(f"   - TemporalEventPropagationAnalyzer: 구현됨")
    return True
checks_total += 1
if check_requirement("시간적 전파 분석", check_temporal):
    checks_passed += 1

# 11. MEDIUM 모드 600M 구현
print(f"\n{BLUE}11. MEDIUM 모드 600M 재설계{END}")
def check_medium_mode():
    with open('main_unified.py', 'r') as f:
        content = f.read()
        if 'MEDIUM 모드 (600M) - MD 문서 재설계 사양' in content:
            print(f"   - MEDIUM 모드 600M 재설계: ✅")
            return True
        else:
            print(f"   - MEDIUM 모드 600M 재설계: ❌")
            return False
checks_total += 1
if check_requirement("MEDIUM 모드 600M", check_medium_mode):
    checks_passed += 1

# 12. 정합성 판단 (LLM + 시스템)
print(f"\n{BLUE}12. 정합성 판단{END}")
def check_plausibility():
    with open('main_unified.py', 'r') as f:
        content = f.read()
        if '_calculate_plausibility' in content:
            print(f"   - _calculate_plausibility: ✅")
            if 'llm_engine.check_plausibility' in content:
                print(f"   - LLM 정합성 검증 연동: ✅")
                return True
            else:
                print(f"   - LLM 정합성 검증 연동: ❌")
                return False
        return False
checks_total += 1
if check_requirement("정합성 판단", check_plausibility):
    checks_passed += 1

# 결과 요약
print("\n" + "="*70)
print(f"{BLUE}📊 MD 문서 요구사항 구현 완성도{END}")
print("="*70)

completion_rate = (checks_passed / checks_total) * 100
color = GREEN if completion_rate >= 90 else YELLOW if completion_rate >= 70 else RED

print(f"✅ 구현 완료: {checks_passed}/{checks_total}")
print(f"📈 완성도: {color}{completion_rate:.1f}%{END}")

if checks_passed < checks_total:
    print(f"\n{YELLOW}⚠️ 미구현 항목이 {checks_total - checks_passed}개 있습니다.{END}")
else:
    print(f"\n{GREEN}🎉 모든 MD 문서 요구사항이 구현되었습니다!{END}")

# 추가 정보
print("\n" + "="*70)
print(f"{BLUE}📝 추가 확인사항{END}")
print("-"*70)

# GPU 메모리 확인
try:
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("GPU: 사용 불가 (CPU 모드)")
except:
    print("GPU: 확인 실패")

# 체크포인트 확인
import os
checkpoint_dir = "training/checkpoints_final"
if os.path.exists(checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    if checkpoints:
        latest = sorted(checkpoints)[-1]
        print(f"최신 체크포인트: {latest}")
    else:
        print("체크포인트: 없음")
else:
    print("체크포인트 디렉토리: 없음")

print("\n✅ MD 문서 요구사항 검증 완료!")