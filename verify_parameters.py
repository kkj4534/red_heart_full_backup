#!/usr/bin/env python3
"""
Red Heart AI 학습 시스템 파라미터 검증
모든 모듈이 정상적으로 연결되고 학습 가능한지 확인
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any
import argparse

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module) -> int:
    """모델의 학습 가능한 파라미터 수 계산"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def verify_system():
    """전체 시스템 파라미터 검증"""
    logger.info("=" * 80)
    logger.info("🔍 Red Heart AI 학습 시스템 파라미터 검증")
    logger.info("=" * 80)
    
    total_params = 0
    module_status = {}
    
    # 1. 백본 검증
    logger.info("\n📦 백본 모듈 검증...")
    try:
        from unified_backbone import RedHeartUnifiedBackbone
        backbone = RedHeartUnifiedBackbone()
        backbone_params = count_parameters(backbone)
        logger.info(f"  ✅ 백본: {backbone_params:,} 파라미터")
        total_params += backbone_params
        module_status['backbone'] = backbone_params
    except Exception as e:
        logger.error(f"  ❌ 백본 로드 실패: {e}")
        module_status['backbone'] = 0
    
    # 2. 헤드 검증
    logger.info("\n📦 헤드 모듈 검증...")
    try:
        from unified_heads import create_heads
        heads = create_heads()
        for name, head in heads.items():
            head_params = count_parameters(head)
            logger.info(f"  ✅ {name}_head: {head_params:,} 파라미터")
            total_params += head_params
            module_status[f'{name}_head'] = head_params
    except Exception as e:
        logger.error(f"  ❌ 헤드 로드 실패: {e}")
    
    # 3. Neural Analyzers 검증
    logger.info("\n📦 Neural Analyzers 검증...")
    try:
        from analyzer_neural_modules import create_neural_analyzers
        neural_analyzers = create_neural_analyzers()
        for name, analyzer in neural_analyzers.items():
            analyzer_params = count_parameters(analyzer)
            logger.info(f"  ✅ {name}: {analyzer_params:,} 파라미터")
            total_params += analyzer_params
            module_status[name] = analyzer_params
    except Exception as e:
        logger.error(f"  ❌ Neural Analyzers 로드 실패: {e}")
    
    # 4. Advanced Analyzer Wrappers 검증
    logger.info("\n📦 Advanced Analyzer Wrappers 검증...")
    try:
        from advanced_analyzer_wrappers import create_advanced_analyzer_wrappers
        advanced_wrappers = create_advanced_analyzer_wrappers()
        for name, wrapper in advanced_wrappers.items():
            wrapper_params = count_parameters(wrapper)
            logger.info(f"  ✅ {name}: {wrapper_params:,} 파라미터")
            total_params += wrapper_params
            module_status[name] = wrapper_params
    except Exception as e:
        logger.error(f"  ❌ Advanced Wrappers 로드 실패: {e}")
    
    # 5. Phase 네트워크 검증
    logger.info("\n📦 Phase 네트워크 검증...")
    try:
        from phase_neural_networks import Phase0ProjectionNet, Phase2CommunityNet, HierarchicalEmotionIntegrator
        
        phase0 = Phase0ProjectionNet()
        phase0_params = count_parameters(phase0)
        logger.info(f"  ✅ Phase0 ProjectionNet: {phase0_params:,} 파라미터")
        total_params += phase0_params
        module_status['phase0'] = phase0_params
        
        phase2 = Phase2CommunityNet()
        phase2_params = count_parameters(phase2)
        logger.info(f"  ✅ Phase2 CommunityNet: {phase2_params:,} 파라미터")
        total_params += phase2_params
        module_status['phase2'] = phase2_params
        
        integrator = HierarchicalEmotionIntegrator()
        integrator_params = count_parameters(integrator)
        logger.info(f"  ✅ Hierarchical Integrator: {integrator_params:,} 파라미터")
        total_params += integrator_params
        module_status['hierarchical_integrator'] = integrator_params
    except Exception as e:
        logger.error(f"  ❌ Phase 네트워크 로드 실패: {e}")
    
    # 6. DSP/Kalman 검증
    logger.info("\n📦 DSP/Kalman 모듈 검증...")
    try:
        from emotion_dsp_simulator import EmotionDSPSimulator, DynamicKalmanFilter
        
        dsp = EmotionDSPSimulator()
        dsp_params = count_parameters(dsp)
        logger.info(f"  ✅ DSP Simulator: {dsp_params:,} 파라미터")
        total_params += dsp_params
        module_status['dsp_simulator'] = dsp_params
        
        kalman = DynamicKalmanFilter()
        kalman_params = count_parameters(kalman)
        logger.info(f"  ✅ Kalman Filter: {kalman_params:,} 파라미터")
        total_params += kalman_params
        module_status['kalman_filter'] = kalman_params
    except Exception as e:
        logger.error(f"  ❌ DSP/Kalman 로드 실패: {e}")
    
    # 7. 통합 시스템 검증
    logger.info("\n📦 통합 시스템 검증...")
    try:
        # 간단한 인자로 시스템 초기화
        class Args:
            def __init__(self):
                self.mode = 'test'
                self.batch_size = 16
                self.learning_rate = 1e-4
                self.epochs = 1
                self.verbose = False
                self.force_preprocess = False
                self.max_samples = 10
                self.preprocess_batch_size = 1
                self.no_param_update = True
                self.debug = False
                self.mixed_precision = False
                self.gradient_accumulation = 1
        
        from unified_training_v2 import UnifiedTrainingSystemV2
        args = Args()
        system = UnifiedTrainingSystemV2(args)
        
        # 모델 초기화
        system.initialize_models()
        
        # 옵티마이저 초기화 (파라미터 수집 확인)
        system.init_optimizer()
        
        # 옵티마이저에 등록된 파라미터 확인
        optimizer_params = sum(p.numel() for group in system.optimizer.param_groups for p in group['params'])
        logger.info(f"  ✅ 옵티마이저에 등록된 파라미터: {optimizer_params:,}")
        
        if optimizer_params != total_params:
            logger.warning(f"  ⚠️ 파라미터 불일치! 예상: {total_params:,}, 실제: {optimizer_params:,}")
        
    except Exception as e:
        logger.error(f"  ❌ 통합 시스템 검증 실패: {e}")
        import traceback
        traceback.print_exc()
    
    # 최종 요약
    logger.info("\n" + "=" * 80)
    logger.info("📊 최종 파라미터 요약")
    logger.info("=" * 80)
    
    # 카테고리별 집계
    backbone_total = module_status.get('backbone', 0)
    heads_total = sum(v for k, v in module_status.items() if '_head' in k)
    neural_total = sum(v for k, v in module_status.items() if 'neural_' in k)
    advanced_total = sum(v for k, v in module_status.items() if 'advanced_' in k)
    phase_total = sum(v for k, v in module_status.items() if 'phase' in k or 'hierarchical' in k)
    dsp_kalman_total = module_status.get('dsp_simulator', 0) + module_status.get('kalman_filter', 0)
    
    logger.info(f"  백본: {backbone_total:,} 파라미터")
    logger.info(f"  헤드: {heads_total:,} 파라미터")
    logger.info(f"  Neural Analyzers: {neural_total:,} 파라미터")
    logger.info(f"  Advanced Analyzers: {advanced_total:,} 파라미터")
    logger.info(f"  Phase Networks: {phase_total:,} 파라미터")
    logger.info(f"  DSP/Kalman: {dsp_kalman_total:,} 파라미터")
    logger.info("-" * 40)
    logger.info(f"  총 파라미터: {total_params:,}")
    
    # MD 파일 목표와 비교
    target_params = 653_000_000  # 653M
    percentage = (total_params / target_params) * 100
    logger.info(f"  목표 대비: {percentage:.1f}% ({total_params:,} / {target_params:,})")
    
    if percentage >= 95:
        logger.info("  ✅ 파라미터 목표 달성!")
    else:
        logger.warning(f"  ⚠️ 파라미터 부족: {target_params - total_params:,} 누락")
    
    logger.info("=" * 80)


if __name__ == "__main__":
    verify_system()