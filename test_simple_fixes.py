#!/usr/bin/env python3
"""
간단한 수정사항 테스트
"""

import logging
import numpy as np

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def test_bentham_only():
    """벤담 계산기만 테스트"""
    logger.info("벤담 계산기 단순 테스트")
    
    try:
        from advanced_bentham_calculator import AdvancedBenthamCalculator
        
        # 벤담 계산기 초기화
        calculator = AdvancedBenthamCalculator()
        logger.info("✅ 벤담 계산기 초기화 성공")
        
        # list 입력 테스트
        test_input = {
            'input_values': [0.7, 0.5, 0.8, 0.6, 0.5, 0.7, 0.8],
            'text_description': '테스트'
        }
        
        result = calculator.calculate_with_advanced_layers(test_input, use_cache=False)
        logger.info(f"✅ 테스트 완료: 점수={result.final_score:.3f}")
        
        # 더미 데이터 사용 확인
        logger.info("\n더미 데이터 체크:")
        
        # feature_scaler 확인
        import inspect
        source = inspect.getsource(calculator.weight_layers[0].compute_weight)
        if 'dummy_data' in source or 'random.randn' in source:
            logger.error("❌ 여전히 더미 데이터 사용 중")
        else:
            logger.info("✅ 더미 데이터 제거 확인")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_bentham_only()
    if success:
        logger.info("\n✅ 테스트 성공!")
    else:
        logger.error("\n❌ 테스트 실패!")