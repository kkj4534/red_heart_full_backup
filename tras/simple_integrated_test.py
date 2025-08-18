#!/usr/bin/env python3
"""
Red Heart AI 시스템 통합 테스트
현재 구현된 모듈들로 10개 데이터 테스트 진행
"""

import sys
import time
import torch
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# 현재 시스템 모듈들 임포트
from config import SYSTEM_CONFIG, ADVANCED_CONFIG, get_device
from data_models import EmotionData, EmotionState, EmotionIntensity
from advanced_emotion_analyzer import AdvancedEmotionAnalyzer
from advanced_bentham_calculator import AdvancedBenthamCalculator
from llm_module.advanced_llm_engine import get_llm_engine, LLMRequest, TaskComplexity

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('RedHeartIntegratedTest')

class IntegratedTestRunner:
    """통합 테스트 실행기"""
    
    def __init__(self):
        """초기화"""
        self.device = get_device()
        self.results = []
        
        # 시스템 모듈 초기화
        print("=== Red Heart AI 시스템 초기화 ===")
        self.emotion_analyzer = AdvancedEmotionAnalyzer()
        print("✅ 감정 분석기 초기화 완료")
        
        self.bentham_calculator = AdvancedBenthamCalculator()
        print("✅ 벤담 계산기 초기화 완료")
        
        self.llm_engine = get_llm_engine()
        print("✅ LLM 엔진 초기화 완료")
        
        # 테스트 데이터
        self.test_data = [
            "오늘 정말 기분이 좋아요! 행복합니다.",
            "이 결정이 정말 옳은 걸까요? 불안해요.",
            "화가 나네요. 이런 일이 있어서는 안 되는데...",
            "슬퍼요. 모든 게 잘못된 것 같아요.",
            "놀랐어요! 예상치 못한 일이 벌어졌네요.",
            "신뢰할 수 있는 사람이 있어서 다행입니다.",
            "역겨워요. 이런 상황은 용납할 수 없어요.",
            "기대가 되네요. 좋은 일이 생길 것 같아요.",
            "죄책감이 들어요. 제가 잘못했나요?",
            "자랑스러워요. 목표를 달성했습니다!"
        ]
        
    def run_single_test(self, test_id: int, text: str) -> Dict[str, Any]:
        """단일 테스트 실행"""
        start_time = time.time()
        
        try:
            # 1단계: 감정 분석
            emotion_result = self.emotion_analyzer.analyze_emotion(text)
            
            # 2단계: 벤담 계산
            bentham_input = {
                'intensity': 4.0,
                'duration': 3.0,
                'certainty': 2.5,
                'propinquity': 3.0,
                'fecundity': 2.5,
                'purity': 3.0,
                'extent': 2.0,
                'emotional_state': str(emotion_result.primary_emotion.value),
                'emotional_intensity': emotion_result.intensity.value,
                'text_context': text
            }
            
            bentham_result = self.bentham_calculator.calculate_with_advanced_layers(
                input_data=bentham_input,
                use_cache=True
            )
            
            # 3단계: LLM 통합 분석 (비동기 호출을 동기로 처리)
            llm_analysis = None
            try:
                import asyncio
                import concurrent.futures
                
                def async_llm_call():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        request = LLMRequest(
                            prompt=f"다음 감정 분석 결과를 종합적으로 해석해주세요: 텍스트='{text}', 감정={emotion_result.primary_emotion}, 벤담 점수={bentham_result.final_score:.3f}",
                            task_type="ethical_analysis",
                            complexity=TaskComplexity.MODERATE,
                            max_tokens=200
                        )
                        return loop.run_until_complete(self.llm_engine.generate_async(request))
                    finally:
                        loop.close()
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(async_llm_call)
                    llm_response = future.result(timeout=30)
                    
                if llm_response and llm_response.success:
                    llm_analysis = llm_response.generated_text[:150] + "..."
                    
            except Exception as e:
                logger.warning(f"LLM 분석 실패: {e}")
                llm_analysis = "LLM 분석 실행 중 오류 발생"
            
            processing_time = time.time() - start_time
            
            result = {
                'test_id': test_id,
                'input_text': text,
                'emotion': {
                    'primary_emotion': str(emotion_result.primary_emotion),
                    'intensity': str(emotion_result.intensity),
                    'confidence': emotion_result.confidence,
                    'processing_method': emotion_result.processing_method
                },
                'bentham': {
                    'base_score': bentham_result.base_score,
                    'final_score': bentham_result.final_score,
                    'confidence': bentham_result.confidence
                },
                'llm_analysis': llm_analysis,
                'processing_time': processing_time,
                'success': True,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"테스트 {test_id} 실패: {e}")
            return {
                'test_id': test_id,
                'input_text': text,
                'error': str(e),
                'processing_time': time.time() - start_time,
                'success': False,
                'timestamp': datetime.now().isoformat()
            }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """모든 테스트 실행"""
        print("\n=== 10개 데이터 통합 테스트 시작 ===")
        
        start_time = time.time()
        success_count = 0
        
        for i, text in enumerate(self.test_data, 1):
            print(f"\n--- 테스트 {i}/10 ---")
            print(f"입력: {text}")
            
            result = self.run_single_test(i, text)
            self.results.append(result)
            
            if result['success']:
                success_count += 1
                print(f"✅ 성공 - 감정: {result['emotion']['primary_emotion']}, "
                      f"벤담: {result['bentham']['final_score']:.3f}, "
                      f"시간: {result['processing_time']:.2f}s")
            else:
                print(f"❌ 실패 - {result['error']}")
        
        total_time = time.time() - start_time
        
        # 결과 요약
        summary = {
            'total_tests': len(self.test_data),
            'successful_tests': success_count,
            'failed_tests': len(self.test_data) - success_count,
            'success_rate': (success_count / len(self.test_data)) * 100,
            'total_processing_time': total_time,
            'average_processing_time': total_time / len(self.test_data),
            'system_config': {
                'total_parameters': ADVANCED_CONFIG['total_parameters'],
                'gpu_enabled': ADVANCED_CONFIG['enable_gpu'],
                'device': str(self.device),
                'precision': ADVANCED_CONFIG['precision']
            },
            'results': self.results
        }
        
        return summary
    
    def save_results(self, summary: Dict[str, Any]):
        """결과 저장"""
        output_file = Path("test_results") / f"integrated_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            import json
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n📁 결과 저장됨: {output_file}")
        return output_file

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("Red Heart AI 시스템 통합 테스트")
    print("=" * 60)
    
    # 시스템 정보 출력
    print(f"디바이스: {get_device()}")
    print(f"총 파라미터: {ADVANCED_CONFIG['total_parameters']:,}")
    print(f"GPU 사용: {ADVANCED_CONFIG['enable_gpu']}")
    
    # 테스트 실행
    test_runner = IntegratedTestRunner()
    summary = test_runner.run_all_tests()
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("=== 테스트 결과 요약 ===")
    print("=" * 60)
    print(f"총 테스트: {summary['total_tests']}")
    print(f"성공: {summary['successful_tests']}")
    print(f"실패: {summary['failed_tests']}")
    print(f"성공률: {summary['success_rate']:.1f}%")
    print(f"총 처리 시간: {summary['total_processing_time']:.2f}초")
    print(f"평균 처리 시간: {summary['average_processing_time']:.2f}초")
    
    # 결과 저장
    output_file = test_runner.save_results(summary)
    
    if summary['success_rate'] >= 80:
        print("\n🎉 테스트 성공! 시스템이 정상적으로 작동합니다.")
    else:
        print("\n⚠️ 일부 테스트 실패. 시스템 점검이 필요합니다.")
    
    return summary

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자에 의해 테스트가 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 테스트 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()