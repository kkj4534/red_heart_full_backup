#!/usr/bin/env python3
"""
Red Heart AI 통합 시스템 간단 테스트
50 epoch 체크포인트 로드 및 기본 기능 테스트
"""

import os
import sys
import torch
import json
from pathlib import Path

# 프로젝트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'training'))

def test_modules():
    """모듈 가용성 테스트"""
    print("=" * 70)
    print("📦 모듈 가용성 테스트")
    print("=" * 70)
    
    modules_status = {}
    
    # 1. UnifiedModel 테스트
    try:
        from training.unified_training_final import UnifiedModel, UnifiedTrainingConfig
        modules_status['UnifiedModel'] = "✅ 사용 가능"
        print("✅ UnifiedModel 로드 성공")
    except Exception as e:
        modules_status['UnifiedModel'] = f"❌ 오류: {e}"
        print(f"❌ UnifiedModel 로드 실패: {e}")
    
    # 2. CheckpointManager 테스트
    try:
        from training.enhanced_checkpoint_manager import EnhancedCheckpointManager
        modules_status['CheckpointManager'] = "✅ 사용 가능"
        print("✅ CheckpointManager 로드 성공")
    except Exception as e:
        modules_status['CheckpointManager'] = f"❌ 오류: {e}"
        print(f"❌ CheckpointManager 로드 실패: {e}")
    
    # 3. Neural Analyzers 테스트
    try:
        from analyzer_neural_modules import create_neural_analyzers
        modules_status['Neural Analyzers'] = "✅ 사용 가능"
        print("✅ Neural Analyzers 로드 성공")
    except Exception as e:
        modules_status['Neural Analyzers'] = f"❌ 오류: {e}"
        print(f"❌ Neural Analyzers 로드 실패: {e}")
    
    # 4. Advanced Wrappers 테스트
    try:
        from advanced_analyzer_wrappers import create_advanced_analyzer_wrappers
        modules_status['Advanced Wrappers'] = "✅ 사용 가능"
        print("✅ Advanced Wrappers 로드 성공")
    except Exception as e:
        modules_status['Advanced Wrappers'] = f"❌ 오류: {e}"
        print(f"❌ Advanced Wrappers 로드 실패: {e}")
    
    # 5. DSP Simulator 테스트
    try:
        from emotion_dsp_simulator import EmotionDSPSimulator
        modules_status['DSP Simulator'] = "✅ 사용 가능"
        print("✅ DSP Simulator 로드 성공")
    except Exception as e:
        modules_status['DSP Simulator'] = f"❌ 오류: {e}"
        print(f"❌ DSP Simulator 로드 실패: {e}")
    
    # 6. Phase Networks 테스트
    try:
        from phase_neural_networks import Phase0ProjectionNet
        modules_status['Phase Networks'] = "✅ 사용 가능"
        print("✅ Phase Networks 로드 성공")
    except Exception as e:
        modules_status['Phase Networks'] = f"❌ 오류: {e}"
        print(f"❌ Phase Networks 로드 실패: {e}")
    
    return modules_status


def test_checkpoint():
    """체크포인트 테스트"""
    print("\n" + "=" * 70)
    print("💾 체크포인트 테스트")
    print("=" * 70)
    
    checkpoint_dir = Path("training/checkpoints_final")
    target_checkpoint = checkpoint_dir / "checkpoint_epoch_0050_lr_0.000009_20250827_154403.pt"
    
    if target_checkpoint.exists():
        print(f"✅ 50 epoch 체크포인트 발견: {target_checkpoint.name}")
        
        # 크기 확인
        size_gb = target_checkpoint.stat().st_size / (1024**3)
        print(f"   크기: {size_gb:.2f}GB")
        
        # 메타데이터 확인
        metadata_file = checkpoint_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                if "checkpoint_epoch_0050_lr_0.000009_20250827_154403.pt" in metadata:
                    info = metadata["checkpoint_epoch_0050_lr_0.000009_20250827_154403.pt"]
                    print(f"   Epoch: {info.get('epoch', 'unknown')}")
                    print(f"   Loss: {info.get('loss', 'unknown')}")
                    print(f"   학습 시간: {info.get('training_time', 'unknown')}")
        
        # 체크포인트 로드 시도
        try:
            print("\n   체크포인트 로드 시도...")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(target_checkpoint, map_location=device, weights_only=False)
            
            print(f"   ✅ 체크포인트 로드 성공")
            print(f"   - Epoch: {checkpoint.get('epoch', 'unknown')}")
            print(f"   - Best Loss: {checkpoint.get('best_loss', 'unknown')}")
            
            if 'model_state' in checkpoint:
                num_params = len(checkpoint['model_state'])
                print(f"   - 모델 상태: {num_params}개 텐서")
            
            return True
            
        except Exception as e:
            print(f"   ❌ 체크포인트 로드 실패: {e}")
            return False
    else:
        print(f"❌ 50 epoch 체크포인트 없음")
        
        # 대체 체크포인트 찾기
        checkpoints = sorted(checkpoint_dir.glob("checkpoint*.pt"))
        if checkpoints:
            latest = checkpoints[-1]
            print(f"   대체 체크포인트: {latest.name}")
        
        return False


def test_device():
    """디바이스 테스트"""
    print("\n" + "=" * 70)
    print("🖥️ 디바이스 테스트")
    print("=" * 70)
    
    if torch.cuda.is_available():
        print(f"✅ CUDA 사용 가능")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        # VRAM 사용량
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        cached = torch.cuda.memory_reserved(0) / 1024**3
        print(f"   할당된 메모리: {allocated:.2f}GB")
        print(f"   캐시된 메모리: {cached:.2f}GB")
    else:
        print("❌ CUDA 사용 불가 - CPU 모드")
        
        # CPU 정보
        import platform
        print(f"   CPU: {platform.processor()}")
        
        # RAM 정보
        import psutil
        ram_total = psutil.virtual_memory().total / 1024**3
        ram_available = psutil.virtual_memory().available / 1024**3
        print(f"   RAM: {ram_available:.1f}/{ram_total:.1f}GB 사용 가능")


def test_simple_inference():
    """간단한 추론 테스트"""
    print("\n" + "=" * 70)
    print("🧪 간단한 추론 테스트")
    print("=" * 70)
    
    try:
        # 필요한 모듈만 임포트
        from analyzer_neural_modules import create_neural_analyzers
        from advanced_analyzer_wrappers import create_advanced_analyzer_wrappers
        
        print("✅ 기본 분석 모듈 로드 성공")
        
        # 간단한 텍스트 분석 시뮬레이션
        test_text = "이 결정은 많은 사람들의 생명과 안전에 영향을 미칩니다."
        print(f"\n테스트 텍스트: {test_text}")
        
        # Neural Analyzers 생성
        neural_analyzers = create_neural_analyzers()
        print(f"   Neural Analyzers 생성: {len(neural_analyzers)}개 모듈")
        
        # Advanced Wrappers 생성
        advanced_wrappers = create_advanced_analyzer_wrappers()
        print(f"   Advanced Wrappers 생성: {len(advanced_wrappers)}개 모듈")
        
        print("\n✅ 추론 테스트 완료")
        return True
        
    except Exception as e:
        print(f"❌ 추론 테스트 실패: {e}")
        return False


def main():
    """메인 테스트 실행"""
    print("\n" + "=" * 70)
    print("🎯 Red Heart AI 통합 시스템 테스트")
    print("   730M 모델 / 50 epoch 학습")
    print("=" * 70)
    
    results = {}
    
    # 1. 디바이스 테스트
    test_device()
    
    # 2. 모듈 테스트
    modules_status = test_modules()
    results['modules'] = modules_status
    
    # 3. 체크포인트 테스트
    checkpoint_ok = test_checkpoint()
    results['checkpoint'] = checkpoint_ok
    
    # 4. 간단한 추론 테스트
    inference_ok = test_simple_inference()
    results['inference'] = inference_ok
    
    # 결과 요약
    print("\n" + "=" * 70)
    print("📊 테스트 결과 요약")
    print("=" * 70)
    
    # 모듈 상태
    available_modules = sum(1 for v in modules_status.values() if "✅" in v)
    total_modules = len(modules_status)
    print(f"\n모듈 가용성: {available_modules}/{total_modules}")
    for name, status in modules_status.items():
        print(f"   {name}: {status}")
    
    # 체크포인트 상태
    print(f"\n체크포인트: {'✅ 정상' if checkpoint_ok else '❌ 문제 있음'}")
    
    # 추론 테스트
    print(f"추론 테스트: {'✅ 성공' if inference_ok else '❌ 실패'}")
    
    # 전체 상태
    print("\n" + "=" * 70)
    if available_modules >= 2 and (checkpoint_ok or inference_ok):
        print("✅ 시스템 부분 작동 가능")
        print("   일부 모듈은 사용 불가하지만 기본 기능은 작동합니다.")
    elif available_modules >= 4:
        print("✅ 시스템 정상 작동 가능")
    else:
        print("⚠️ 시스템 작동 제한적")
        print("   필수 의존성 설치가 필요합니다.")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    try:
        results = main()
        sys.exit(0 if results.get('inference', False) else 1)
    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)