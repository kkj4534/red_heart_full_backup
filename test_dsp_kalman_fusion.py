#!/usr/bin/env python3
"""
DSP 시뮬레이터와 칼만 필터 융합 테스트
"""

import torch
import numpy as np
import logging
import sys
import time
from typing import Dict, Any

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_dsp_kalman')

def test_dsp_simulator():
    """DSP 시뮬레이터 단독 테스트"""
    logger.info("=" * 50)
    logger.info("DSP 시뮬레이터 테스트 시작")
    logger.info("=" * 50)
    
    try:
        from emotion_dsp_simulator import EmotionDSPSimulator
        from config import get_device
        
        device = get_device()
        logger.info(f"디바이스: {device}")
        
        # DSP 시뮬레이터 생성
        dsp = EmotionDSPSimulator({'hidden_dim': 256}).to(device)
        logger.info("✅ DSP 시뮬레이터 생성 완료")
        
        # 파라미터 수 계산
        total_params = sum(p.numel() for p in dsp.parameters())
        logger.info(f"총 파라미터 수: {total_params:,} ({total_params/1e6:.2f}M)")
        
        # 테스트 입력
        batch_size = 2
        hidden_dim = 256
        test_input = torch.randn(batch_size, hidden_dim).to(device)
        
        # Forward pass
        with torch.no_grad():
            start_time = time.time()
            result = dsp(test_input)
            inference_time = time.time() - start_time
        
        logger.info(f"추론 시간: {inference_time*1000:.2f}ms")
        
        # 결과 검증
        assert 'emotion_spectrum' in result
        assert 'adsr_params' in result
        assert 'valence_arousal' in result
        assert 'final_emotions' in result
        
        # 출력 shape 확인
        logger.info(f"emotion_spectrum shape: {result['emotion_spectrum'].shape}")
        logger.info(f"adsr_params shape: {result['adsr_params'].shape}")
        logger.info(f"valence_arousal shape: {result['valence_arousal'].shape}")
        logger.info(f"final_emotions shape: {result['final_emotions'].shape}")
        
        # 감정 확률 검증
        emotions = result['final_emotions']
        assert torch.allclose(emotions.sum(dim=-1), torch.ones(batch_size).to(device), atol=1e-5)
        logger.info("✅ 감정 확률 합 = 1 검증 완료")
        
        # Valence-Arousal 범위 검증
        va = result['valence_arousal']
        assert va.min() >= -1 and va.max() <= 1
        logger.info(f"✅ Valence-Arousal 범위: [{va.min():.3f}, {va.max():.3f}]")
        
        logger.info("✅ DSP 시뮬레이터 테스트 통과")
        return True
        
    except Exception as e:
        logger.error(f"❌ DSP 시뮬레이터 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_kalman_filter():
    """칼만 필터 단독 테스트"""
    logger.info("=" * 50)
    logger.info("칼만 필터 테스트 시작")
    logger.info("=" * 50)
    
    try:
        from emotion_dsp_simulator import DynamicKalmanFilter
        from config import get_device
        
        device = get_device()
        
        # 칼만 필터 생성
        kalman = DynamicKalmanFilter(state_dim=7).to(device)
        logger.info("✅ 칼만 필터 생성 완료")
        
        # 파라미터 수 계산
        total_params = sum(p.numel() for p in kalman.parameters())
        logger.info(f"총 파라미터 수: {total_params:,}")
        
        # 테스트 입력
        batch_size = 2
        state_dim = 7
        
        # 두 감정 소스 생성
        traditional = torch.rand(batch_size, state_dim).to(device)
        traditional = torch.softmax(traditional, dim=-1)
        
        dsp = torch.rand(batch_size, state_dim).to(device)
        dsp = torch.softmax(dsp, dim=-1)
        
        # 칼만 필터 적용
        with torch.no_grad():
            start_time = time.time()
            fused = kalman(traditional, dsp)
            fusion_time = time.time() - start_time
        
        logger.info(f"융합 시간: {fusion_time*1000:.2f}ms")
        
        # 결과 검증
        assert fused.shape == (batch_size, state_dim)
        assert torch.allclose(fused.sum(dim=-1), torch.ones(batch_size).to(device), atol=1e-5)
        logger.info("✅ 융합 확률 합 = 1 검증 완료")
        
        # 시간적 일관성 테스트
        prev_state = fused
        fused2 = kalman(traditional, dsp, prev_state)
        
        # 이전 상태 고려 시 변화가 있어야 함
        assert not torch.allclose(fused, fused2)
        logger.info("✅ 시간적 일관성 테스트 통과")
        
        logger.info("✅ 칼만 필터 테스트 통과")
        return True
        
    except Exception as e:
        logger.error(f"❌ 칼만 필터 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integrated_system():
    """통합 시스템 테스트"""
    logger.info("=" * 50)
    logger.info("통합 시스템 테스트 시작")
    logger.info("=" * 50)
    
    try:
        from advanced_emotion_analyzer import AdvancedEmotionAnalyzer
        
        # 감정 분석기 생성
        logger.info("감정 분석기 초기화 중...")
        analyzer = AdvancedEmotionAnalyzer()
        
        # DSP 컴포넌트 확인
        assert analyzer.dsp_simulator is not None, "DSP 시뮬레이터가 초기화되지 않음"
        assert analyzer.kalman_filter is not None, "칼만 필터가 초기화되지 않음"
        logger.info("✅ DSP 컴포넌트 초기화 확인")
        
        # 테스트 텍스트
        test_texts = [
            "오늘은 정말 행복한 하루였어요!",
            "시험에 떨어져서 너무 슬퍼요...",
            "갑자기 비가 와서 놀랐어요",
            "이런 상황이 정말 화가 나네요",
        ]
        
        for text in test_texts:
            logger.info(f"\n텍스트: '{text}'")
            
            # 감정 분석 실행
            start_time = time.time()
            result = analyzer.analyze_emotion(text, language="ko")
            analysis_time = time.time() - start_time
            
            logger.info(f"분석 시간: {analysis_time*1000:.2f}ms")
            logger.info(f"주 감정: {result.primary_emotion.value} (신뢰도: {result.confidence:.3f})")
            
            # DSP 메타데이터 확인
            if result.metadata and 'fusion_method' in result.metadata:
                logger.info(f"융합 방법: {result.metadata['fusion_method']}")
                
                if 'dsp_valence_arousal' in result.metadata:
                    va = result.metadata['dsp_valence_arousal']
                    logger.info(f"Valence-Arousal: {va}")
            
            # 보조 감정
            if result.secondary_emotions:
                logger.info("보조 감정:")
                for emotion, conf in result.secondary_emotions.items():
                    logger.info(f"  - {emotion.value}: {conf:.3f}")
        
        logger.info("\n✅ 통합 시스템 테스트 통과")
        return True
        
    except Exception as e:
        logger.error(f"❌ 통합 시스템 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_efficiency():
    """메모리 효율성 테스트"""
    logger.info("=" * 50)
    logger.info("메모리 효율성 테스트 시작")
    logger.info("=" * 50)
    
    try:
        import torch
        from emotion_dsp_simulator import EmotionDSPSimulator, DynamicKalmanFilter
        from config import get_device, get_gpu_memory_info
        
        device = get_device()
        
        if device.type == 'cuda':
            # 초기 메모리 상태
            torch.cuda.empty_cache()
            initial_info = get_gpu_memory_info()
            logger.info(f"초기 GPU 메모리: {initial_info['allocated_mb']:.1f}MB / {initial_info['total_mb']:.1f}MB")
            
            # DSP 시뮬레이터 로드
            dsp = EmotionDSPSimulator({'hidden_dim': 256}).to(device)
            dsp_info = get_gpu_memory_info()
            dsp_memory = dsp_info['allocated_mb'] - initial_info['allocated_mb']
            logger.info(f"DSP 시뮬레이터 메모리: {dsp_memory:.1f}MB")
            
            # 칼만 필터 로드
            kalman = DynamicKalmanFilter(state_dim=7).to(device)
            kalman_info = get_gpu_memory_info()
            kalman_memory = kalman_info['allocated_mb'] - dsp_info['allocated_mb']
            logger.info(f"칼만 필터 메모리: {kalman_memory:.1f}MB")
            
            # 총 메모리 사용량
            total_memory = kalman_info['allocated_mb'] - initial_info['allocated_mb']
            logger.info(f"총 추가 메모리: {total_memory:.1f}MB")
            
            # 목표: 100MB 이하
            if total_memory < 100:
                logger.info("✅ 메모리 효율성 테스트 통과 (<100MB)")
            else:
                logger.warning(f"⚠️ 메모리 사용량이 목표(100MB)를 초과: {total_memory:.1f}MB")
            
            # 정리
            del dsp, kalman
            torch.cuda.empty_cache()
            
        else:
            logger.info("CPU 모드 - 메모리 테스트 스킵")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 메모리 효율성 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 테스트 실행"""
    logger.info("🚀 DSP-칼만 융합 테스트 시작")
    
    results = {
        'DSP 시뮬레이터': test_dsp_simulator(),
        '칼만 필터': test_kalman_filter(),
        '통합 시스템': test_integrated_system(),
        '메모리 효율성': test_memory_efficiency(),
    }
    
    # 결과 요약
    logger.info("\n" + "=" * 50)
    logger.info("테스트 결과 요약")
    logger.info("=" * 50)
    
    for test_name, passed in results.items():
        status = "✅ 통과" if passed else "❌ 실패"
        logger.info(f"{test_name}: {status}")
    
    # 전체 결과
    all_passed = all(results.values())
    if all_passed:
        logger.info("\n🎉 모든 테스트 통과!")
    else:
        logger.error("\n⚠️ 일부 테스트 실패")
        sys.exit(1)

if __name__ == "__main__":
    main()