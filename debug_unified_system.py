#!/usr/bin/env python3
"""
통합 학습 시스템 디버그 스크립트
Unified Learning System Debug Script

초기화 과정에서의 문제점을 정확히 파악하기 위한 상세 디버그
"""

import asyncio
import logging
import traceback
import sys
import torch
from datetime import datetime

# 상세 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('debug_unified_system.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

# 경고 필터 설정
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

async def debug_head_initialization():
    """헤드 초기화 과정 디버그"""
    
    logger.info("=" * 80)
    logger.info("헤드 초기화 디버그 시작")
    logger.info("=" * 80)
    
    # 1. 먼저 각 헤드를 개별적으로 초기화 시도
    from head_compatibility_interface import HeadType
    
    head_types = [
        HeadType.EMOTION_EMPATHY,
        HeadType.BENTHAM_FROMM,
        HeadType.SEMANTIC_SURD,
        HeadType.REGRET_LEARNING,
        HeadType.META_INTEGRATION
    ]
    
    successful_heads = []
    failed_heads = []
    
    for head_type in head_types:
        logger.info(f"\n{'='*60}")
        logger.info(f"헤드 테스트: {head_type.value}")
        logger.info(f"{'='*60}")
        
        try:
            # 헤드 어댑터 임포트 및 생성
            if head_type == HeadType.EMOTION_EMPATHY:
                from head_compatibility_interface import EmotionEmpathyHeadAdapter
                adapter = EmotionEmpathyHeadAdapter()
            elif head_type == HeadType.BENTHAM_FROMM:
                from head_compatibility_interface import BenthamFrommHeadAdapter
                adapter = BenthamFrommHeadAdapter()
            elif head_type == HeadType.SEMANTIC_SURD:
                from head_compatibility_interface import SemanticSURDHeadAdapter
                adapter = SemanticSURDHeadAdapter()
            elif head_type == HeadType.REGRET_LEARNING:
                from head_compatibility_interface import RegretLearningHeadAdapter
                adapter = RegretLearningHeadAdapter()
            elif head_type == HeadType.META_INTEGRATION:
                from head_compatibility_interface import MetaIntegrationHeadAdapter
                adapter = MetaIntegrationHeadAdapter()
            
            logger.info(f"✅ {head_type.value} 어댑터 생성 성공")
            
            # 초기화 시도
            logger.info(f"초기화 시작...")
            await adapter.initialize_head()
            logger.info(f"✅ {head_type.value} 초기화 성공")
            
            # PyTorch 네트워크 확인
            pytorch_net = adapter.get_pytorch_network()
            if pytorch_net is not None:
                param_count = sum(p.numel() for p in pytorch_net.parameters())
                logger.info(f"✅ PyTorch 네트워크 확인: {param_count:,}개 파라미터")
            else:
                logger.warning(f"⚠️ PyTorch 네트워크가 None입니다")
            
            successful_heads.append(head_type)
            
        except Exception as e:
            logger.error(f"❌ {head_type.value} 실패: {str(e)}")
            logger.error(f"트레이스백:\n{traceback.format_exc()}")
            failed_heads.append((head_type, str(e)))
    
    # 결과 요약
    logger.info("\n" + "="*80)
    logger.info("헤드 초기화 결과 요약")
    logger.info("="*80)
    logger.info(f"성공: {len(successful_heads)}/{len(head_types)}")
    for head in successful_heads:
        logger.info(f"  ✅ {head.value}")
    
    if failed_heads:
        logger.error(f"실패: {len(failed_heads)}/{len(head_types)}")
        for head, error in failed_heads:
            logger.error(f"  ❌ {head.value}: {error}")

async def debug_unified_system():
    """통합 시스템 전체 디버그"""
    
    logger.info("=" * 80)
    logger.info("통합 학습 시스템 디버그 시작")
    logger.info("=" * 80)
    
    # GPU 메모리 상태 확인
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"총 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
        logger.info(f"할당된 메모리: {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB")
    else:
        logger.warning("CUDA를 사용할 수 없습니다")
    
    try:
        # 1. UnifiedLearningSystem 임포트
        logger.info("\n1. UnifiedLearningSystem 임포트 시도...")
        from unified_learning_system import UnifiedLearningSystem
        logger.info("✅ 임포트 성공")
        
        # 2. 인스턴스 생성
        logger.info("\n2. UnifiedLearningSystem 인스턴스 생성...")
        learning_system = UnifiedLearningSystem()
        logger.info("✅ 인스턴스 생성 성공")
        
        # 3. 시스템 초기화
        logger.info("\n3. 시스템 초기화 시작...")
        await learning_system.initialize_system()
        logger.info("✅ 시스템 초기화 성공")
        
        # 4. 캐시된 헤드 모듈 확인
        logger.info("\n4. 캐시된 헤드 모듈 확인...")
        for head_type, head_module in learning_system.cached_head_modules.items():
            logger.info(f"  - {head_type.value}: {type(head_module).__name__}")
            if hasattr(head_module, 'parameters'):
                param_count = sum(p.numel() for p in head_module.parameters())
                logger.info(f"    파라미터: {param_count:,}개")
        
        # 5. 간단한 forward 테스트
        logger.info("\n5. Forward 테스트...")
        dummy_data = {
            'text': 'Test sample',
            'batch_size': 1,
            'labels': torch.tensor([0])
        }
        
        # 단일 헤드로 테스트
        from head_compatibility_interface import HeadType
        test_heads = [HeadType.EMOTION_EMPATHY]
        
        try:
            outputs = learning_system.trainer.forward_with_checkpointing(
                dummy_data, test_heads
            )
            logger.info("✅ Forward 패스 성공")
            logger.info(f"출력 키: {list(outputs.keys())}")
            
        except Exception as e:
            logger.error(f"❌ Forward 패스 실패: {str(e)}")
            logger.error(f"트레이스백:\n{traceback.format_exc()}")
        
        logger.info("\n✅ 통합 시스템 디버그 완료")
        
    except Exception as e:
        logger.error(f"\n❌ 통합 시스템 오류: {str(e)}")
        logger.error(f"트레이스백:\n{traceback.format_exc()}")

async def debug_training_step():
    """훈련 스텝 디버그"""
    
    logger.info("=" * 80)
    logger.info("훈련 스텝 디버그")
    logger.info("=" * 80)
    
    try:
        from unified_learning_system import UnifiedLearningSystem
        from head_compatibility_interface import HeadType
        
        # 시스템 생성 및 초기화
        logger.info("시스템 초기화 중...")
        learning_system = UnifiedLearningSystem()
        await learning_system.initialize_system()
        logger.info("✅ 시스템 초기화 완료")
        
        # 가상의 데이터 로더
        class DummyDataLoader:
            def __init__(self, num_batches=5):
                self.num_batches = num_batches
            
            def __iter__(self):
                for i in range(self.num_batches):
                    yield {
                        'text': f'Training sample {i}',
                        'batch_size': 2,
                        'labels': torch.randint(0, 10, (2,))
                    }
        
        train_loader = DummyDataLoader(5)
        
        # 단일 에포크 테스트
        logger.info("\n단일 에포크 훈련 테스트...")
        
        # 간단한 설정으로 train_unified_system 호출
        await learning_system.train_unified_system(
            train_data_loader=train_loader,
            validation_data_loader=None,
            num_epochs=1
        )
        
        logger.info("✅ 훈련 테스트 완료")
        
    except Exception as e:
        logger.error(f"❌ 훈련 스텝 오류: {str(e)}")
        logger.error(f"트레이스백:\n{traceback.format_exc()}")

async def main():
    """메인 디버그 함수"""
    
    start_time = datetime.now()
    logger.info(f"디버그 시작: {start_time}")
    
    # 1. 헤드 개별 초기화 테스트
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: 헤드 개별 초기화 테스트")
    logger.info("="*80)
    await debug_head_initialization()
    
    # 2. 통합 시스템 테스트
    logger.info("\n" + "="*80)
    logger.info("PHASE 2: 통합 시스템 테스트")
    logger.info("="*80)
    await debug_unified_system()
    
    # 3. 훈련 스텝 테스트
    logger.info("\n" + "="*80)
    logger.info("PHASE 3: 훈련 스텝 테스트")
    logger.info("="*80)
    await debug_training_step()
    
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"\n디버그 완료: {end_time}")
    logger.info(f"소요 시간: {duration}")

if __name__ == "__main__":
    asyncio.run(main())