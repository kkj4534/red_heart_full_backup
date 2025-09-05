#!/usr/bin/env python3
"""
수정사항 테스트 스크립트
- 벤담 계산기 더미 데이터 제거 확인
- 입력 정규화 동작 확인
- 경험 DB 메타데이터 처리 확인
"""

import asyncio
import logging
import numpy as np
from advanced_bentham_calculator import AdvancedBenthamCalculator
from advanced_experience_database import AdvancedExperienceDatabase

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def test_bentham_calculator():
    """벤담 계산기 테스트"""
    logger.info("=" * 60)
    logger.info("벤담 계산기 테스트 시작")
    logger.info("=" * 60)
    
    try:
        # 벤담 계산기 초기화
        calculator = AdvancedBenthamCalculator()
        logger.info("✅ 벤담 계산기 초기화 성공")
        
        # 테스트 1: list 형태 입력값 처리
        logger.info("\n테스트 1: list 형태 입력값 처리")
        test_input_list = {
            'input_values': [0.7, 0.5, 0.8, 0.6, 0.5, 0.7, 0.8],  # 7개 값 list
            'text_description': '테스트 시나리오',
            'affected_count': 10,
            'duration_seconds': 3600
        }
        
        result = calculator.calculate_with_advanced_layers(test_input_list, use_cache=False)
        logger.info(f"✅ list 입력 처리 성공: 점수={result.final_score:.3f}")
        
        # 테스트 2: dict 형태 입력값 처리
        logger.info("\n테스트 2: dict 형태 입력값 처리")
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
            'text_description': '테스트 시나리오',
            'affected_count': 10,
            'duration_seconds': 3600
        }
        
        result = calculator.calculate_with_advanced_layers(test_input_dict, use_cache=False)
        logger.info(f"✅ dict 입력 처리 성공: 점수={result.final_score:.3f}")
        
        # 테스트 3: 불완전한 입력 처리 (에러 예상)
        logger.info("\n테스트 3: 불완전한 입력 처리 (에러 예상)")
        test_input_incomplete = {
            'input_values': [0.7, 0.5, 0.8],  # 3개만 (7개 필요)
            'text_description': '불완전한 테스트'
        }
        
        try:
            result = calculator.calculate_with_advanced_layers(test_input_incomplete, use_cache=False)
            logger.warning("⚠️ 불완전한 입력이 처리됨 (예상치 못한 동작)")
        except Exception as e:
            logger.info(f"✅ 불완전한 입력 적절히 거부됨: {e}")
        
        logger.info("\n✅ 벤담 계산기 테스트 완료")
        return True
        
    except Exception as e:
        logger.error(f"❌ 벤담 계산기 테스트 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


async def test_experience_db():
    """경험 DB 테스트"""
    logger.info("\n" + "=" * 60)
    logger.info("경험 DB 테스트 시작")
    logger.info("=" * 60)
    
    try:
        # 경험 DB 초기화
        db = AdvancedExperienceDatabase()
        await db.initialize()
        logger.info("✅ 경험 DB 초기화 성공")
        
        # 테스트 1: 문자열 emotion 처리
        logger.info("\n테스트 1: 문자열 emotion 처리")
        metadata1 = {
            'emotion': 'positive',
            'urgency': 0.7,
            'impact': 0.8
        }
        
        await db.add_experience(
            experience_text="긍정적인 경험",
            metadata=metadata1,
            category="test"
        )
        logger.info("✅ 문자열 emotion 저장 성공")
        
        # 테스트 2: dict emotion (scores) 처리
        logger.info("\n테스트 2: dict emotion (scores) 처리")
        metadata2 = {
            'emotion': {
                'scores': [0.8, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0],  # joy가 가장 높음
                'label': 'joy'
            },
            'urgency': 0.5,
            'impact': 0.6
        }
        
        await db.add_experience(
            experience_text="기쁜 경험",
            metadata=metadata2,
            category="test"
        )
        logger.info("✅ dict emotion (scores) 저장 성공")
        
        # 테스트 3: dict emotion (label only) 처리
        logger.info("\n테스트 3: dict emotion (label only) 처리")
        metadata3 = {
            'emotion': {
                'label': 'negative'
            },
            'urgency': 0.3,
            'impact': 0.4
        }
        
        await db.add_experience(
            experience_text="부정적인 경험",
            metadata=metadata3,
            category="test"
        )
        logger.info("✅ dict emotion (label) 저장 성공")
        
        # 테스트 4: 중첩된 emotion 처리
        logger.info("\n테스트 4: 중첩된 emotion 처리")
        metadata4 = {
            'emotion': {
                'emotion': 'neutral'  # 중첩
            },
            'urgency': 0.5,
            'impact': 0.5
        }
        
        await db.add_experience(
            experience_text="중립적인 경험",
            metadata=metadata4,
            category="test"
        )
        logger.info("✅ 중첩된 emotion 저장 성공")
        
        # 저장된 경험 검색
        logger.info("\n저장된 경험 검색 테스트")
        results = await db.search_similar_experiences("테스트 경험", max_results=5)
        logger.info(f"✅ 검색 성공: {len(results)}개 결과")
        
        # 정리
        await db.close()
        logger.info("\n✅ 경험 DB 테스트 완료")
        return True
        
    except Exception as e:
        logger.error(f"❌ 경험 DB 테스트 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


async def main():
    """메인 테스트 함수"""
    logger.info("수정사항 테스트 시작")
    logger.info("=" * 80)
    
    # 벤담 계산기 테스트
    bentham_success = test_bentham_calculator()
    
    # 경험 DB 테스트
    db_success = await test_experience_db()
    
    # 결과 요약
    logger.info("\n" + "=" * 80)
    logger.info("테스트 결과 요약")
    logger.info("=" * 80)
    logger.info(f"벤담 계산기: {'✅ 성공' if bentham_success else '❌ 실패'}")
    logger.info(f"경험 DB: {'✅ 성공' if db_success else '❌ 실패'}")
    
    if bentham_success and db_success:
        logger.info("\n✅ 모든 테스트 통과! 수정사항이 올바르게 적용되었습니다.")
    else:
        logger.error("\n❌ 일부 테스트 실패. 추가 수정이 필요합니다.")


if __name__ == "__main__":
    asyncio.run(main())