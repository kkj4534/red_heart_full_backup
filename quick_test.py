#!/usr/bin/env python3
"""
빠른 통합 테스트 스크립트
"""

import sys
import asyncio
sys.path.insert(0, '.')

async def quick_test():
    print('빠른 통합 테스트 시작...')
    
    try:
        from main import RedHeartSystem, AnalysisRequest
        print('모듈 import 성공')
        
        system = RedHeartSystem()
        print('시스템 생성 성공')
        
        # 초기화 테스트
        await system.initialize()
        print('시스템 초기화 성공')
        
        # 간단한 분석 요청
        test_request = AnalysisRequest(
            text='간단한 테스트 텍스트입니다.',
            language='ko'
        )
        
        result = await system.analyze_async(test_request)
        print('기본 분석 완료')
        
        print('모든 테스트 통과!')
        return True
        
    except Exception as e:
        print(f'테스트 실패: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(quick_test())
    result_text = '성공' if success else '실패'
    print(f'테스트 결과: {result_text}')