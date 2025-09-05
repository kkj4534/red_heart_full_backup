#!/usr/bin/env python3
"""
모듈 상태를 정확하게 체크하는 독립 스크립트
초기 로딩에 충분한 시간 제공 (NO FALLBACK)
"""

import sys
import os
import time

print("   ⏳ 모듈 로딩 중... (최대 60초 소요)")
start_time = time.time()

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'training'))

modules_status = {
    "UnifiedModel (730M)": False,
    "정밀 감정→벤담 매퍼": False,
    "유휴 시간 학습 시스템": False,
    "벤치마크 시스템": False,
    "Neural Analyzers (368M)": False,
    "Advanced Wrappers (112M)": False,
    "DSP Simulator (14M)": False,
    "LLM Engine": False
}

# 실제 import를 통한 정확한 체크 (NO SHORTCUTS, NO FALLBACK)
# UnifiedModel 체크
try:
    print("      로딩: UnifiedModel...", end="", flush=True)
    from training.unified_training_final import UnifiedModel
    modules_status["UnifiedModel (730M)"] = True
    print(" ✅")
except Exception as e:
    print(f" ❌ ({str(e)[:50]})")

# 정밀 매퍼 체크
try:
    print("      로딩: 정밀 감정→벤담 매퍼...", end="", flush=True)
    from semantic_emotion_bentham_mapper import SemanticEmotionBenthamMapper
    modules_status["정밀 감정→벤담 매퍼"] = True
    print(" ✅")
except Exception as e:
    print(f" ❌ ({str(e)[:50]})")

# 유휴 학습 체크
try:
    print("      로딩: 유휴 시간 학습 시스템...", end="", flush=True)
    from idle_time_learner import HierarchicalIdleLearner
    modules_status["유휴 시간 학습 시스템"] = True
    print(" ✅")
except Exception as e:
    print(f" ❌ ({str(e)[:50]})")

# 벤치마크 체크
try:
    print("      로딩: 벤치마크 시스템...", end="", flush=True)
    from benchmark_unified import UnifiedBenchmark
    modules_status["벤치마크 시스템"] = True
    print(" ✅")
except Exception as e:
    print(f" ❌ ({str(e)[:50]})")

# Neural Analyzers 체크
try:
    print("      로딩: Neural Analyzers...", end="", flush=True)
    from analyzer_neural_modules import create_neural_analyzers
    modules_status["Neural Analyzers (368M)"] = True
    print(" ✅")
except Exception as e:
    print(f" ❌ ({str(e)[:50]})")

# Advanced Wrappers 체크
try:
    print("      로딩: Advanced Wrappers...", end="", flush=True)
    from advanced_analyzer_wrappers import create_advanced_analyzer_wrappers
    modules_status["Advanced Wrappers (112M)"] = True
    print(" ✅")
except Exception as e:
    print(f" ❌ ({str(e)[:50]})")

# DSP Simulator 체크
try:
    print("      로딩: DSP Simulator...", end="", flush=True)
    from emotion_dsp_simulator import EmotionDSPSimulator
    modules_status["DSP Simulator (14M)"] = True
    print(" ✅")
except Exception as e:
    print(f" ❌ ({str(e)[:50]})")

# LLM Engine 체크
try:
    print("      로딩: LLM Engine...", end="", flush=True)
    from llm_module.advanced_llm_engine import AdvancedLLMEngine
    modules_status["LLM Engine"] = True
    print(" ✅")
except Exception as e:
    print(f" ❌ ({str(e)[:50]})")

# 결과 출력
elapsed = time.time() - start_time
total = len(modules_status)
available = sum(modules_status.values())

print(f"\n   ⏱️ 로딩 완료: {elapsed:.2f}초")
print("\n   📦 모듈 상태 (완전 검증):")
for module, status in modules_status.items():
    icon = "✅" if status else "❌"
    print(f"      {icon} {module}")

print("")
print(f"   📊 모듈 가용성: {available}/{total} ({available*100//total}%)")

if available < total:
    missing = [k for k, v in modules_status.items() if not v]
    print(f"\n   ❌ 누락된 모듈: {', '.join(missing)}")
    print("   필수 모듈이 누락되었습니다. 시스템이 제한적으로 작동합니다.")
    sys.exit(1)  # 실패 시 명확히 실패
else:
    print("\n   ✅ 모든 모듈 정상 로드 완료!")