#!/usr/bin/env python3
"""
벤담 계산기 직접 테스트 - 에러 위치 정확히 파악
"""

import logging
import traceback
import sys
import torch

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


def test_bentham_error():
    """벤담 계산기 에러 정확한 위치 파악"""
    try:
        logger.info("=" * 60)
        logger.info("벤담 계산기 에러 추적 시작")
        logger.info("=" * 60)
        
        # 1. 벤담 계산기 임포트
        from advanced_bentham_calculator import AdvancedBenthamCalculator
        
        # 2. 초기화
        logger.info("벤담 계산기 초기화...")
        calc = AdvancedBenthamCalculator()
        logger.info("✅ 초기화 성공")
        
        # 3. weight_layers 확인
        logger.info(f"weight_layers 수: {len(calc.weight_layers)}")
        
        # 4. neural_predictor 확인
        logger.info(f"neural_predictor 타입: {type(calc.neural_predictor)}")
        
        # 5. 체크포인트 로딩 확인
        logger.info("\n체크포인트 관련 확인:")
        
        # 체크포인트 파일 존재 확인
        import os
        checkpoint_path = "training/checkpoints_final/checkpoint_epoch_0050_lr_0.000009_20250827_154403.pt"
        if os.path.exists(checkpoint_path):
            logger.info(f"✅ 체크포인트 파일 존재: {checkpoint_path}")
            
            # 체크포인트 로드 시도
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            logger.info(f"   - Epoch: {checkpoint.get('epoch', 'unknown')}")
            logger.info(f"   - Keys: {list(checkpoint.keys())[:5]}...")  # 처음 5개 키만
            
            # bentham 관련 키 확인
            bentham_keys = [k for k in checkpoint.keys() if 'bentham' in k.lower()]
            if bentham_keys:
                logger.info(f"   - Bentham 관련 키: {bentham_keys}")
            else:
                logger.warning("   ⚠️ 체크포인트에 bentham 관련 키 없음")
        else:
            logger.warning(f"⚠️ 체크포인트 파일 없음: {checkpoint_path}")
        
        # 6. 실제 계산 테스트
        logger.info("\n실제 계산 테스트:")
        
        # 테스트 입력 1: list 형태
        test_input_list = {
            'input_values': [0.7, 0.5, 0.8, 0.6, 0.5, 0.7, 0.8],
            'text_description': '테스트 리스트'
        }
        
        logger.info("테스트 1: list 입력")
        try:
            result = calc.calculate_with_advanced_layers(test_input_list, use_cache=False)
            logger.info(f"✅ list 입력 성공: {result.final_score:.3f}")
        except Exception as e:
            logger.error(f"❌ list 입력 실패: {e}")
            # 상세 에러 추적
            if "'list' object has no attribute 'get'" in str(e):
                logger.error("   => list.get() 에러 발생!")
                logger.error("   => 에러 위치 추적:")
                import traceback
                tb = traceback.format_exc()
                for line in tb.split('\n'):
                    if '.get(' in line or 'get(' in line:
                        logger.error(f"      {line}")
        
        # 테스트 입력 2: dict 형태
        test_input_dict = {
            'input_values': {
                'intensity': 0.7,
                'duration': 0.5,
                'certainty': 0.8,
                'propinquity': 0.6,
                'fecundity': 0.5,
                'purity': 0.7,
                'extent': 0.8
            },
            'text_description': '테스트 딕셔너리'
        }
        
        logger.info("\n테스트 2: dict 입력")
        try:
            result = calc.calculate_with_advanced_layers(test_input_dict, use_cache=False)
            logger.info(f"✅ dict 입력 성공: {result.final_score:.3f}")
        except Exception as e:
            logger.error(f"❌ dict 입력 실패: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"테스트 전체 실패: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_bentham_error()
    if success:
        logger.info("\n✅ 테스트 완료")
    else:
        logger.error("\n❌ 테스트 실패")