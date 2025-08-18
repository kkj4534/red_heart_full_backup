#!/usr/bin/env python3
"""
하이브리드 분산 학습 시작 스크립트
Hybrid Distributed Training Start Script
"""

import sys
from pathlib import Path
import torch

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training.hybrid_distributed_trainer import HybridDistributedTrainer, HybridConfig
from training.results_analyzer import TrainingResultsAnalyzer

def main():
    """하이브리드 학습 실행"""
    print("🚀 Red Heart XAI 하이브리드 분산 학습 시작")
    print("=" * 70)
    
    # 시스템 정보
    print(f"🖥️ 시스템 정보:")
    print(f"   - CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   - GPU: {torch.cuda.get_device_name(0)}")
        print(f"   - GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    print(f"   - CPU 코어: {torch.get_num_threads()}")
    
    # 하이브리드 설정
    config = HybridConfig(
        # 모델 설정 (기존 33억 파라미터 유지)
        target_params=3_000_000_000,
        gpu_memory_gb=8.0,              # RTX 2070S
        cpu_memory_gb=128.0,            # 시스템 RAM
        
        # 학습 설정 (최적화됨)
        regrets_per_step=7,
        bentham_calculations_per_regret=3,
        epochs=3,
        batch_size=8,                   # GPU 메모리 고려
        micro_batch_size=2,
        
        # 분산 설정
        num_workers=4,                  # CPU 코어 활용
        gpu_layers_ratio=0.6,           # GPU에서 60% 처리
        overlap_computation=True,
        
        # 최적화 설정
        use_mixed_precision=True,       # FP16 사용
        gradient_accumulation_steps=4,
        
        # 로깅 설정 (더 자주)
        log_every_n_steps=5,
        save_checkpoint_every=20,
        max_storage_gb=50.0
    )
    
    print(f"📊 하이브리드 설정:")
    print(f"   - 모델 파라미터: {config.target_params:,}개")
    print(f"   - 후회/스텝: {config.regrets_per_step}")
    print(f"   - 벤담/후회: {config.bentham_calculations_per_regret}")
    print(f"   - 에포크: {config.epochs}")
    print(f"   - 배치 크기: {config.batch_size}")
    print(f"   - GPU 레이어 비율: {config.gpu_layers_ratio * 100:.0f}%")
    print(f"   - 워커 수: {config.num_workers}")
    print(f"   - Mixed Precision: {config.use_mixed_precision}")
    print(f"   - 로깅 주기: {config.log_every_n_steps}스텝")
    print("=" * 70)
    
    try:
        # 하이브리드 학습기 생성
        trainer = HybridDistributedTrainer(config)
        
        # 학습 실행
        print("🎯 하이브리드 학습 시작...")
        report, checkpoint_path = trainer.train()
        
        print("\n" + "=" * 70)
        print("🎉 하이브리드 학습 완료!")
        print(f"📊 최종 통계:")
        print(f"   - 총 후회: {sum(trainer.training_stats['regret_count']):,}")
        print(f"   - 총 벤담 계산: {sum(trainer.training_stats['bentham_count']):,}")
        print(f"   - 총 스텝: {len(trainer.training_stats['total_loss'])}")
        print(f"   - 학습 시간: {report['training_summary']['training_duration']/3600:.2f}시간")
        print(f"💾 체크포인트: {checkpoint_path}")
        print("=" * 70)
        
        # 결과 분석 및 문서 생성
        print("\n📊 결과 분석 및 문서 생성 중...")
        analyzer = TrainingResultsAnalyzer(project_root)
        
        # 리포트 경로 찾기
        report_files = list((project_root / 'training' / 'hybrid_outputs' / 'reports').glob('hybrid_training_report_*.json'))
        if report_files:
            latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
            analysis_results = analyzer.analyze_and_generate_docs(latest_report)
            
            print("\n" + "=" * 70)
            print("📚 하이브리드 학습 문서 생성 완료!")
            print(f"📄 마크다운 리포트: {analysis_results['markdown_report']}")
            print(f"🌐 HTML 리포트: {analysis_results['html_report']}")
            print(f"📊 시각화: {len(analysis_results['visualizations'])}개")
            print(f"📁 docs 폴더에 결과표가 저장되었습니다.")
            print("=" * 70)
            
            # 권장사항 출력
            recommendations = analysis_results['analysis']['recommendations']
            if recommendations:
                print("\n💡 하이브리드 시스템 권장사항:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"   {i}. {rec}")
        
        print(f"\n🎊 Red Heart XAI 하이브리드 학습 완전 성공! 🎊")
        print(f"⚡ CPU+GPU 하이브리드 시스템으로 최적화된 학습 완료!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 하이브리드 학습 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)