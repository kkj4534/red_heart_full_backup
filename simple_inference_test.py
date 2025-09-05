#!/usr/bin/env python3
"""
Red Heart AI 간단한 추론 테스트
각 모드별로 빠르게 테스트
"""

import sys
import asyncio
import time
import torch
sys.path.append('/mnt/c/large_project/linux_red_heart')

from main_unified import UnifiedInferenceSystem, InferenceConfig, MemoryMode

# 색상 코드
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
END = '\033[0m'

async def test_inference(mode: MemoryMode, test_text: str):
    """단일 모드 추론 테스트"""
    print(f"\n{CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{END}")
    print(f"{BLUE}테스트: {mode.value.upper()} 모드{END}")
    print(f"{CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{END}")
    
    try:
        # 설정 생성
        config = InferenceConfig(
            memory_mode=mode,
            auto_memory_mode=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            cache_size=5,
            verbose=False
        )
        
        print(f"1️⃣ 시스템 초기화 중...")
        start_init = time.time()
        
        # 시스템 초기화
        system = UnifiedInferenceSystem(config)
        await system.initialize()
        
        init_time = time.time() - start_init
        print(f"   {GREEN}✅ 초기화 완료 ({init_time:.1f}초){END}")
        
        # GPU 메모리 확인
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            print(f"   📊 GPU 메모리: {allocated:.2f}GB")
        
        # 활성 모듈 요약
        active_modules = []
        if system.config.use_three_view_scenario:
            active_modules.append("3-View")
        if system.config.use_multi_ethics_system:
            active_modules.append("Multi-Ethics")
        if system.config.use_neural_analyzers:
            active_modules.append("Neural")
        if system.config.use_meta_integration:
            active_modules.append("Meta")
        
        print(f"   📦 활성 모듈: {', '.join(active_modules) if active_modules else 'Basic only'}")
        
        print(f"\n2️⃣ 추론 테스트")
        print(f"   📝 입력: \"{test_text}\"")
        
        start_inference = time.time()
        result = await system.analyze(test_text)
        inference_time = time.time() - start_inference
        
        print(f"   {GREEN}✅ 추론 완료 ({inference_time:.2f}초){END}")
        
        # 결과 요약
        if result and not result.get('error'):
            print(f"\n3️⃣ 결과 요약")
            print(f"   • 통합 점수: {result.get('integrated_score', 0):.3f}")
            
            if 'emotion' in result:
                emotion = result['emotion']
                # 감정 값만 추출 (dict나 특수 키는 제외)
                emotion_scores = {}
                for key, value in emotion.items():
                    if isinstance(value, (int, float)) and key not in ['dsp_processed', 'kalman_filtered', 'hierarchy']:
                        emotion_scores[key] = value
                
                if emotion_scores:
                    top_emotion = max(emotion_scores.items(), key=lambda x: x[1])
                    print(f"   • 주요 감정: {top_emotion[0]} ({top_emotion[1]:.3f})")
            
            if 'bentham' in result:
                bentham = result['bentham']
                total_pleasure = sum(v for k, v in bentham.items() if k != 'total')
                print(f"   • 벤담 쾌락: {total_pleasure:.3f}")
            
            if 'regret' in result:
                regret = result['regret']
                if isinstance(regret, dict):
                    if 'unified' in regret:
                        print(f"   • 후회 점수: {regret['unified'].get('score', 0):.3f}")
                    else:
                        print(f"   • 후회 분석: {len(regret)} 항목")
                elif isinstance(regret, (int, float)):
                    print(f"   • 후회 점수: {regret:.3f}")
            
            print(f"\n   {GREEN}✅ {mode.value.upper()} 모드 테스트 성공!{END}")
        else:
            error_msg = result.get('error', 'Unknown error') if result else 'No result'
            print(f"\n   {RED}❌ 추론 실패: {error_msg}{END}")
            
    except Exception as e:
        print(f"\n   {RED}❌ 테스트 실패: {str(e)}{END}")
        import traceback
        if '--debug' in sys.argv:
            traceback.print_exc()
    
    finally:
        # 메모리 정리
        if 'system' in locals():
            del system
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"{CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{END}")

async def test_ethical_dilemma():
    """윤리적 딜레마 분석 테스트 (HEAVY 모드)"""
    print(f"\n{YELLOW}🎯 윤리적 딜레마 분석 테스트 (비선형 워크플로우){END}")
    print(f"{CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{END}")
    
    try:
        config = InferenceConfig(
            memory_mode=MemoryMode.HEAVY,
            auto_memory_mode=False,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        print("시스템 초기화 중...")
        system = UnifiedInferenceSystem(config)
        await system.initialize()
        
        # 테스트 시나리오
        scenarios = [
            "친구를 적극적으로 도와준다",
            "상황을 지켜본 후 신중하게 판단한다",
            "최소한의 도움만 제공한다"
        ]
        
        print(f"\n시나리오 {len(scenarios)}개 분석 중...")
        start_time = time.time()
        
        result = await system.analyze_ethical_dilemma(scenarios)
        
        elapsed = time.time() - start_time
        print(f"{GREEN}✅ 분석 완료 ({elapsed:.2f}초){END}")
        
        # 결과 출력
        print(f"\n📊 분석 결과:")
        print(f"   • 평가된 시나리오: {result.get('total_evaluated', 0)}개")
        print(f"   • 처리 시간: {result.get('processing_time', 0):.2f}초")
        
        if 'selected_scenarios' in result and result['selected_scenarios']:
            print(f"\n🏆 상위 2개 시나리오:")
            for i, scenario in enumerate(result['selected_scenarios'][:2], 1):
                score = scenario['analysis'].get('integrated_score', 0)
                text = scenario.get('original_scenario', 'Unknown')
                print(f"   {i}. {text[:50]}... (점수: {score:.3f})")
        
        if 'recommendation' in result:
            print(f"\n💡 추천: {result['recommendation'][:100]}...")
        
        print(f"\n{GREEN}✅ 윤리적 딜레마 분석 성공!{END}")
        
    except Exception as e:
        print(f"{RED}❌ 윤리적 딜레마 분석 실패: {e}{END}")
    
    finally:
        if 'system' in locals():
            del system
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

async def main():
    """메인 함수"""
    print(f"{BLUE}{'='*50}{END}")
    print(f"{BLUE}🚀 Red Heart AI 추론 시스템 테스트{END}")
    print(f"{BLUE}{'='*50}{END}")
    
    # GPU 정보
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("GPU: 사용 불가 (CPU 모드)")
    
    # 테스트 텍스트
    test_text = "친구가 어려운 상황에 처했을 때 어떻게 도와야 할까?"
    
    # 모드 선택 로직 개선
    run_light = True  # 기본은 LIGHT 모드
    
    # 특정 모드가 지정되면 LIGHT 모드 건너뛰기
    if '--medium' in sys.argv or '--heavy' in sys.argv or '--dilemma' in sys.argv:
        run_light = False
    
    # --all이면 모든 모드 실행
    if '--all' in sys.argv:
        run_light = True
    
    # 1. LIGHT 모드 테스트
    if run_light:
        await test_inference(MemoryMode.LIGHT, test_text)
    
    # 2. MEDIUM 모드 테스트 (옵션)
    if '--medium' in sys.argv or '--all' in sys.argv:
        await test_inference(MemoryMode.MEDIUM, test_text)
    
    # 3. HEAVY 모드 테스트 (옵션)
    if '--heavy' in sys.argv or '--all' in sys.argv:
        await test_inference(MemoryMode.HEAVY, test_text)
    
    # 4. 윤리적 딜레마 테스트 (옵션)
    if '--dilemma' in sys.argv or '--all' in sys.argv:
        await test_ethical_dilemma()
    
    print(f"\n{GREEN}{'='*50}{END}")
    print(f"{GREEN}🎉 모든 테스트 완료!{END}")
    print(f"{GREEN}{'='*50}{END}")
    
    # 사용법 안내
    if len(sys.argv) == 1:
        print(f"\n{YELLOW}💡 추가 테스트 옵션:{END}")
        print("   --medium  : MEDIUM 모드 테스트")
        print("   --heavy   : HEAVY 모드 테스트")
        print("   --dilemma : 윤리적 딜레마 분석 테스트")
        print("   --all     : 모든 테스트 실행")
        print("   --debug   : 디버그 모드 (상세 에러 출력)")

if __name__ == "__main__":
    asyncio.run(main())