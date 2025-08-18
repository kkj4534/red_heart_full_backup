#!/usr/bin/env python3
"""
간단한 Red Heart 시스템 테스트
"""

import asyncio
import json
from main import RedHeartSystem, AnalysisRequest, setup_advanced_logging

async def simple_test():
    """간단한 테스트 실행"""
    # 로깅 설정
    setup_advanced_logging()
    
    # 시스템 초기화
    print("🔴❤️ Red Heart 시스템 간단 테스트")
    print("=" * 50)
    
    system = RedHeartSystem()
    
    try:
        print("시스템 초기화 중...")
        await system.initialize()
        
        # 테스트 데이터 로드
        with open('processed_datasets/korean_cultural_scenarios.json', 'r', encoding='utf-8') as f:
            scenarios = json.load(f)
        
        # 첫 번째 시나리오 테스트
        first_scenario = scenarios[0]
        print(f"\n테스트 시나리오: {first_scenario['title']}")
        print(f"설명: {first_scenario['description']}")
        
        # 분석 요청
        request = AnalysisRequest(
            text=first_scenario['description'],
            language="ko",
            scenario_type="korean_cultural"
        )
        
        print("\n분석 실행 중...")
        result = await system.analyze_async(request)
        
        print(f"\n✅ 분석 완료!")
        print(f"통합 점수: {result.integrated_score:.3f}")
        print(f"신뢰도: {result.confidence:.3f}")
        print(f"처리 시간: {result.processing_time:.3f}초")
        print(f"추천: {result.recommendation}")
        
        # 시스템 상태 확인
        status = system.get_system_status()
        print(f"\n📊 시스템 상태:")
        print(f"성공 분석: {status.performance_stats['successful_analyses']}")
        print(f"실패 분석: {status.performance_stats['failed_analyses']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False
    
    finally:
        # 정리
        if hasattr(system, 'thread_pool'):
            system.thread_pool.shutdown(wait=True)

if __name__ == "__main__":
    success = asyncio.run(simple_test())
    print(f"\n{'✅ 테스트 성공' if success else '❌ 테스트 실패'}")