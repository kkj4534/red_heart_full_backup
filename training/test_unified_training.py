#!/usr/bin/env python3
"""
통합 학습 시스템 테스트 스크립트
작은 데이터셋으로 빠르게 시스템 검증
"""

import os
import sys
import torch
import logging
from pathlib import Path
import argparse
import time

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('RedHeart.TestTraining')


def test_components():
    """개별 컴포넌트 테스트"""
    logger.info("=" * 70)
    logger.info("🧪 컴포넌트 단위 테스트 시작")
    logger.info("=" * 70)
    
    test_results = {}
    
    # 1. CheckpointManager 테스트
    try:
        from training.enhanced_checkpoint_manager import EnhancedCheckpointManager
        manager = EnhancedCheckpointManager(
            checkpoint_dir="training/test_checkpoints",
            max_checkpoints=3,
            save_interval=1
        )
        logger.info("✅ CheckpointManager: 정상")
        test_results['checkpoint_manager'] = True
    except Exception as e:
        logger.error(f"❌ CheckpointManager: {e}")
        test_results['checkpoint_manager'] = False
    
    # 2. LRSweepOptimizer 테스트
    try:
        from training.lr_sweep_optimizer import LRSweepOptimizer
        lr_sweep = LRSweepOptimizer(
            base_lr=1e-4,
            sweep_range=(1e-5, 1e-3),
            num_sweep_points=3,
            sweep_epochs=1,
            sweep_steps_per_epoch=10
        )
        logger.info("✅ LRSweepOptimizer: 정상")
        test_results['lr_sweep'] = True
    except Exception as e:
        logger.error(f"❌ LRSweepOptimizer: {e}")
        test_results['lr_sweep'] = False
    
    # 3. SweetSpotDetector 테스트
    try:
        from training.sweet_spot_detector import SweetSpotDetector
        detector = SweetSpotDetector(
            window_size=3,
            stability_threshold=0.01,
            patience=5,
            min_epochs=3
        )
        # 테스트 데이터 업데이트
        for epoch in range(5):
            detector.update(
                epoch=epoch,
                module_metrics={'test_module': {'loss': 1.0 / (epoch + 1)}},
                learning_rate=1e-4
            )
        logger.info("✅ SweetSpotDetector: 정상")
        test_results['sweet_spot'] = True
    except Exception as e:
        logger.error(f"❌ SweetSpotDetector: {e}")
        test_results['sweet_spot'] = False
    
    # 4. ParameterCrossoverSystem 테스트
    try:
        from training.parameter_crossover_system import ParameterCrossoverSystem
        crossover = ParameterCrossoverSystem(
            crossover_strategy='selective',
            blend_ratio=0.7,
            mutation_rate=0.01
        )
        logger.info("✅ ParameterCrossoverSystem: 정상")
        test_results['crossover'] = True
    except Exception as e:
        logger.error(f"❌ ParameterCrossoverSystem: {e}")
        test_results['crossover'] = False
    
    # 5. OOMHandler 테스트
    try:
        from training.oom_handler import OOMHandler
        oom_handler = OOMHandler(
            initial_batch_size=4,
            min_batch_size=1,
            gradient_accumulation=16,
            memory_threshold=0.85
        )
        # 메모리 상태 체크
        status = oom_handler.check_memory_status()
        logger.info(f"  - CPU 메모리: {status['cpu']['percent']:.1f}%")
        if 'percent' in status.get('gpu', {}):
            logger.info(f"  - GPU 메모리: {status['gpu']['percent']:.1f}%")
        logger.info("✅ OOMHandler: 정상")
        test_results['oom_handler'] = True
    except Exception as e:
        logger.error(f"❌ OOMHandler: {e}")
        test_results['oom_handler'] = False
    
    # 6. AdvancedTrainingManager 테스트
    try:
        from training.advanced_training_techniques import AdvancedTrainingManager
        training_manager = AdvancedTrainingManager(
            enable_label_smoothing=True,
            enable_rdrop=True,
            enable_ema=True,
            enable_llrd=True,
            label_smoothing=0.1,
            rdrop_alpha=1.0,
            ema_decay=0.999,
            llrd_decay=0.8
        )
        logger.info("✅ AdvancedTrainingManager: 정상")
        test_results['advanced_training'] = True
    except Exception as e:
        logger.error(f"❌ AdvancedTrainingManager: {e}")
        test_results['advanced_training'] = False
    
    # 결과 요약
    logger.info("\n" + "=" * 70)
    logger.info("📊 컴포넌트 테스트 결과:")
    for component, result in test_results.items():
        status = "✅ 성공" if result else "❌ 실패"
        logger.info(f"  - {component}: {status}")
    
    all_passed = all(test_results.values())
    if all_passed:
        logger.info("\n🎉 모든 컴포넌트 테스트 통과!")
    else:
        logger.error("\n⚠️ 일부 컴포넌트 테스트 실패")
    
    return all_passed


def test_mini_training():
    """미니 학습 테스트 (2 에폭)"""
    logger.info("\n" + "=" * 70)
    logger.info("🚀 미니 학습 테스트 시작 (2 에폭)")
    logger.info("=" * 70)
    
    try:
        from training.unified_training_final import UnifiedTrainingConfig, UnifiedTrainer
        
        # 테스트용 설정
        config = UnifiedTrainingConfig()
        config.total_epochs = 2
        config.micro_batch_size = 2
        config.gradient_accumulation = 4  # 빠른 테스트
        config.checkpoint_interval = 1  # 매 에폭 저장
        config.checkpoint_dir = "training/test_checkpoints"
        config.lr_sweep_enabled = False  # 스윕 건너뛰기
        config.log_interval = 5
        config.val_interval = 10
        
        # 트레이너 생성
        trainer = UnifiedTrainer(config)
        
        # 학습 실행
        start_time = time.time()
        trainer.train()
        elapsed_time = time.time() - start_time
        
        logger.info(f"\n✅ 미니 학습 완료! (소요 시간: {elapsed_time:.1f}초)")
        
        # 결과 확인
        if trainer.checkpoint_manager.checkpoint_metadata:
            logger.info(f"  - 저장된 체크포인트: {len(trainer.checkpoint_manager.checkpoint_metadata)}개")
            logger.info(f"  - 최종 손실: {trainer.best_loss:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 미니 학습 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_checkpoint_loading():
    """체크포인트 로드 테스트"""
    logger.info("\n" + "=" * 70)
    logger.info("💾 체크포인트 로드 테스트")
    logger.info("=" * 70)
    
    try:
        from training.enhanced_checkpoint_manager import EnhancedCheckpointManager
        
        manager = EnhancedCheckpointManager(
            checkpoint_dir="training/test_checkpoints"
        )
        
        # 체크포인트가 있는지 확인
        if manager.checkpoint_metadata:
            # 최신 체크포인트 로드
            checkpoint_data = manager.load_checkpoint()
            logger.info(f"✅ 체크포인트 로드 성공")
            logger.info(f"  - 에폭: {checkpoint_data['epoch']}")
            logger.info(f"  - LR: {checkpoint_data['lr']:.1e}")
            
            # 최고 성능 체크포인트 찾기
            best_checkpoint = manager.get_best_checkpoint('loss')
            if best_checkpoint:
                logger.info(f"  - 최고 성능 체크포인트: {Path(best_checkpoint).name}")
            
            return True
        else:
            logger.info("  - 저장된 체크포인트 없음 (정상)")
            return True
            
    except Exception as e:
        logger.error(f"❌ 체크포인트 로드 실패: {e}")
        return False


def test_memory_monitoring():
    """메모리 모니터링 테스트"""
    logger.info("\n" + "=" * 70)
    logger.info("🔍 메모리 모니터링 테스트")
    logger.info("=" * 70)
    
    try:
        from training.oom_handler import OOMHandler
        
        handler = OOMHandler(
            initial_batch_size=4,
            min_batch_size=1,
            gradient_accumulation=16
        )
        
        # 현재 메모리 상태
        status = handler.check_memory_status()
        
        logger.info("📊 현재 메모리 상태:")
        logger.info(f"  CPU:")
        logger.info(f"    - 전체: {status['cpu']['total_gb']:.1f} GB")
        logger.info(f"    - 사용 중: {status['cpu']['used_gb']:.1f} GB")
        logger.info(f"    - 사용률: {status['cpu']['percent']:.1f}%")
        
        if 'error' not in status['gpu']:
            logger.info(f"  GPU:")
            logger.info(f"    - 전체: {status['gpu'].get('total_gb', 0):.1f} GB")
            logger.info(f"    - 할당: {status['gpu'].get('allocated_gb', 0):.1f} GB")
            logger.info(f"    - 사용률: {status['gpu'].get('percent', 0):.1f}%")
        else:
            logger.info(f"  GPU: 사용 불가")
        
        # 메모리 임계 상태 체크
        is_critical = handler.is_memory_critical()
        if is_critical:
            logger.warning("  ⚠️ 메모리가 임계 상태입니다!")
        else:
            logger.info("  ✅ 메모리 상태 양호")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 메모리 모니터링 실패: {e}")
        return False


def run_all_tests():
    """모든 테스트 실행"""
    logger.info("\n" + "=" * 70)
    logger.info("🧪 Red Heart AI 통합 학습 시스템 전체 테스트")
    logger.info("=" * 70)
    
    test_results = {
        '컴포넌트 테스트': test_components(),
        '메모리 모니터링': test_memory_monitoring(),
        '체크포인트 로딩': test_checkpoint_loading(),
        '미니 학습': test_mini_training()
    }
    
    # 최종 결과
    logger.info("\n" + "=" * 70)
    logger.info("📊 전체 테스트 결과:")
    logger.info("=" * 70)
    
    for test_name, result in test_results.items():
        status = "✅ 통과" if result else "❌ 실패"
        logger.info(f"  {test_name}: {status}")
    
    all_passed = all(test_results.values())
    
    if all_passed:
        logger.info("\n" + "🎉 " * 10)
        logger.info("🎊 모든 테스트 통과! 시스템 정상 작동 확인!")
        logger.info("🎉 " * 10)
    else:
        logger.error("\n⚠️ 일부 테스트 실패. 로그를 확인하세요.")
    
    return all_passed


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="통합 학습 시스템 테스트")
    parser.add_argument('--quick', action='store_true', help='빠른 테스트 (컴포넌트만)')
    parser.add_argument('--training', action='store_true', help='미니 학습 테스트만')
    parser.add_argument('--memory', action='store_true', help='메모리 모니터링만')
    
    args = parser.parse_args()
    
    if args.quick:
        success = test_components()
    elif args.training:
        success = test_mini_training()
    elif args.memory:
        success = test_memory_monitoring()
    else:
        success = run_all_tests()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)