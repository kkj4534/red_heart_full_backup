#!/usr/bin/env python3
"""
10개 샘플 테스트 학습
10 Samples Test Training
"""

import sys
import time
from pathlib import Path
import torch

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training.hybrid_distributed_trainer import HybridDistributedTrainer, HybridConfig
from training.results_analyzer import TrainingResultsAnalyzer

def main():
    """10개 샘플 테스트 학습 실행"""
    print("🧪 Red Heart XAI - 10개 샘플 테스트 학습")
    print("=" * 60)
    
    # 테스트 설정
    config = HybridConfig(
        # 테스트 모드 활성화
        test_mode=True,
        test_samples=10,
        
        # 모델 설정 (공격적 고성능 70GB)
        target_params=4_300_000_000,    # 43억 파라미터 (현실적 70GB)
        gpu_memory_gb=8.0,
        cpu_memory_gb=70.0,             # WSL 공격적 메모리 활용
        
        # 학습 설정 (공격적 고성능)
        regrets_per_step=7,             # 원래 설정 복원 (최고 품질)
        bentham_calculations_per_regret=3,  # 원래 설정 복원 (정확도 극대화)
        epochs=3,                       # 3순회 유지
        batch_size=12,                  # 대형 배치 (처리량 증가)
        micro_batch_size=3,             # 메모리 여유로 증가
        
        # 분산 설정 (공격적 병렬화)
        num_workers=8,                  # 최대 CPU 활용
        gpu_layers_ratio=0.6,           # 균형잡힌 GPU/CPU 분할
        overlap_computation=True,
        use_cpu_offload=True,           # CPU 오프로드 활성화
        enable_memory_monitoring=True,  # 실시간 메모리 모니터링
        
        # 최적화 설정
        use_mixed_precision=True,
        gradient_accumulation_steps=4,   # 대형 배치로 조정
        use_gradient_checkpointing=True, # 메모리 절약
        use_parameter_sharing=True,     # 파라미터 공유
        
        # 로깅 설정 (자주)
        log_every_n_steps=1,            # 모든 스텝 로깅
        save_checkpoint_every=2,        # 2스텝마다 저장
        max_storage_gb=10.0             # 테스트용 작은 용량
    )
    
    print(f"🧪 테스트 설정:")
    print(f"   - 샘플 수: {config.test_samples}개")
    print(f"   - 모델 파라미터: {config.target_params:,}개")
    print(f"   - 후회/스텝: {config.regrets_per_step}")
    print(f"   - 벤담/후회: {config.bentham_calculations_per_regret}")
    print(f"   - 총 벤담/스텝: {config.regrets_per_step * config.bentham_calculations_per_regret}")
    print(f"   - 에포크: {config.epochs}")
    print(f"   - 배치 크기: {config.batch_size}")
    print("=" * 60)
    
    # 시간 측정 시작
    total_start_time = time.time()
    
    try:
        # 테스트 학습기 생성
        trainer = HybridDistributedTrainer(config)
        
        # 학습 실행
        print("🎯 10개 샘플 테스트 학습 시작...")
        print(f"⏱️  시작 시간: {time.strftime('%H:%M:%S')}")
        
        report, checkpoint_path = trainer.train()
        
        total_time = time.time() - total_start_time
        
        print("\n" + "=" * 60)
        print("🎉 10개 샘플 테스트 완료!")
        print(f"📊 테스트 결과:")
        print(f"   - 총 학습 시간: {total_time:.2f}초 ({total_time/60:.2f}분)")
        print(f"   - 총 후회: {sum(trainer.training_stats['regret_count']):,}")
        print(f"   - 총 벤담 계산: {sum(trainer.training_stats['bentham_count']):,}")
        print(f"   - 총 스텝: {len(trainer.training_stats['total_loss'])}")
        print(f"   - 스텝당 평균 시간: {total_time / len(trainer.training_stats['total_loss']):.2f}초")
        print(f"💾 체크포인트: {checkpoint_path}")
        
        # 손실 개선 분석
        losses = trainer.training_stats['total_loss']
        if len(losses) > 2:
            initial_loss = losses[0]
            final_loss = losses[-1]
            improvement = ((initial_loss - final_loss) / initial_loss * 100) if initial_loss > 0 else 0
            print(f"📈 손실 개선: {improvement:.2f}% ({initial_loss:.6f} → {final_loss:.6f})")
        
        # 전체 학습 시간 예측
        samples_per_second = config.test_samples / total_time
        full_samples = 28882  # 전체 데이터
        predicted_full_time = full_samples / samples_per_second
        
        print("\n" + "=" * 60)
        print("📊 전체 학습 시간 예측:")
        print(f"   - 샘플 처리 속도: {samples_per_second:.2f} 샘플/초")
        print(f"   - 전체 데이터: {full_samples:,}개")
        print(f"   - 예상 전체 시간: {predicted_full_time:.0f}초 ({predicted_full_time/3600:.2f}시간)")
        print(f"   - 3에포크 예상: {predicted_full_time * 3 / 3600:.2f}시간")
        
        # 메모리 사용량
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / 1024**3
            print(f"   - 최대 GPU 메모리: {gpu_memory:.2f}GB")
        
        print("=" * 60)
        
        # 학습 품질 평가
        if improvement > 1:
            print("✅ 학습 효과 확인: 손실이 개선되고 있습니다!")
        elif improvement > 0:
            print("⚠️ 미미한 학습 효과: 더 많은 데이터가 필요할 수 있습니다.")
        else:
            print("❌ 학습 효과 없음: 설정 조정이 필요합니다.")
        
        if predicted_full_time < 24 * 3600:  # 24시간 이내
            print("✅ 전체 학습 시간 적절: 하루 내 완료 가능")
        elif predicted_full_time < 72 * 3600:  # 72시간 이내
            print("⚠️ 전체 학습 시간 길음: 2-3일 소요 예상")
        else:
            print("❌ 전체 학습 시간 과다: 최적화 필요")
        
        # 결과 분석 및 문서 생성
        print("\n📊 결과 분석 중...")
        analyzer = TrainingResultsAnalyzer(project_root)
        
        # 리포트 경로 찾기
        report_files = list((project_root / 'training' / 'hybrid_outputs' / 'reports').glob('hybrid_training_report_*.json'))
        if report_files:
            latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
            analysis_results = analyzer.analyze_and_generate_docs(latest_report)
            print(f"📄 분석 리포트: {analysis_results['markdown_report']}")
        
        print(f"\n🎊 10개 샘플 테스트 성공! 🎊")
        if predicted_full_time < 12 * 3600:  # 12시간 이내라면
            print("🚀 전체 학습 진행 가능합니다!")
            print("python3 training/start_hybrid_training.py 로 전체 학습을 시작하세요.")
        else:
            print("⚠️ 전체 학습 시간을 고려하여 설정 조정을 검토하세요.")
        
        return True, {
            'total_time': total_time,
            'samples_per_second': samples_per_second,
            'predicted_full_time_hours': predicted_full_time / 3600,
            'improvement_percent': improvement if len(losses) > 2 else 0,
            'total_steps': len(trainer.training_stats['total_loss']),
            'total_regrets': sum(trainer.training_stats['regret_count']),
            'total_benthams': sum(trainer.training_stats['bentham_count'])
        }
        
    except Exception as e:
        print(f"\n❌ 10개 샘플 테스트 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return False, None

if __name__ == "__main__":
    success, results = main()
    
    if success and results:
        print(f"\n⏱️ 핵심 결과:")
        print(f"   - 테스트 시간: {results['total_time']:.1f}초")
        print(f"   - 전체 예상: {results['predicted_full_time_hours']:.1f}시간")
        print(f"   - 손실 개선: {results['improvement_percent']:.1f}%")
    
    sys.exit(0 if success else 1)