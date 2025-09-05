#!/usr/bin/env python3
"""
신경망 가중치 예측 에러 추적 테스트
"""

import logging
import traceback
import sys

# 로깅 설정 - 더 자세한 정보 출력
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s'
)
logger = logging.getLogger(__name__)

def test_error_location():
    """에러 발생 위치 정확히 추적"""
    logger.info("=" * 60)
    logger.info("신경망 가중치 예측 에러 추적 시작")
    logger.info("=" * 60)
    
    try:
        # 1. 벤담 계산기 임포트 전 로깅
        logger.info("벤담 계산기 임포트 시작...")
        from advanced_bentham_calculator import AdvancedBenthamCalculator
        logger.info("벤담 계산기 임포트 완료")
        
        # 2. 초기화 전 로깅
        logger.info("벤담 계산기 초기화 시작...")
        
        # 원본 _predict_neural_weights 메서드를 래핑하여 호출 추적
        original_predict = AdvancedBenthamCalculator._predict_neural_weights
        call_count = [0]
        
        def wrapped_predict(self, context):
            call_count[0] += 1
            logger.info(f"_predict_neural_weights 호출 #{call_count[0]}")
            logger.info(f"  context type: {type(context)}")
            
            if hasattr(context, 'input_values'):
                logger.info(f"  input_values type: {type(context.input_values)}")
                if isinstance(context.input_values, list):
                    logger.info(f"  input_values is list with length: {len(context.input_values)}")
                elif isinstance(context.input_values, dict):
                    logger.info(f"  input_values is dict with keys: {list(context.input_values.keys())}")
            else:
                logger.info("  context has no input_values attribute")
            
            # 호출 스택 추적
            logger.info("  호출 스택:")
            for frame_info in traceback.extract_stack()[:-1]:
                if 'advanced_bentham' in frame_info.filename:
                    logger.info(f"    {frame_info.filename}:{frame_info.lineno} in {frame_info.name}")
            
            try:
                return original_predict(self, context)
            except Exception as e:
                logger.error(f"  _predict_neural_weights 에러: {e}")
                logger.error(f"  전체 스택:\n{traceback.format_exc()}")
                return None
        
        AdvancedBenthamCalculator._predict_neural_weights = wrapped_predict
        
        # 3. 계산기 생성
        calculator = AdvancedBenthamCalculator()
        logger.info("벤담 계산기 초기화 완료")
        
        # 4. weight_layers 접근 (이때 에러가 발생하는 것으로 추정)
        logger.info("weight_layers 프로퍼티 접근 시작...")
        layers = calculator.weight_layers
        logger.info(f"weight_layers 접근 완료: {len(layers)}개 레이어")
        
        # 5. 실제 계산 테스트
        logger.info("\n실제 계산 테스트...")
        test_input = {
            'input_values': [0.7, 0.5, 0.8, 0.6, 0.5, 0.7, 0.8],
            'text_description': '테스트'
        }
        
        result = calculator.calculate_with_advanced_layers(test_input, use_cache=False)
        logger.info(f"계산 완료: 점수={result.final_score:.3f}")
        
        logger.info(f"\n_predict_neural_weights 총 호출 횟수: {call_count[0]}")
        return True
        
    except Exception as e:
        logger.error(f"테스트 실패: {e}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_error_location()
    if success:
        logger.info("\n✅ 에러 추적 완료")
    else:
        logger.error("\n❌ 에러 추적 실패")