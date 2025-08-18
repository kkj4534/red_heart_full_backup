#!/usr/bin/env python3
"""
학습 시작 스크립트
Training Start Script
"""

import sys
from pathlib import Path
import logging

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training.regret_based_training_pipeline import RegretTrainer, RegretTrainingConfig
from training.results_analyzer import TrainingResultsAnalyzer

def main():
    """메인 학습 실행 함수"""
    print("🚀 Red Heart XAI 후회 기반 학습 시작")
    print("=" * 60)
    
    # 학습 설정
    config = RegretTrainingConfig(
        regrets_per_step=7,           # 7회 후회/스텝
        bentham_calculations_per_regret=3,  # 3회 벤담 계산/후회 (총 21회)
        epochs=3,                     # 3번 선회
        batch_size=16,
        learning_rate=1e-4,
        log_every_n_steps=20,         # 20스텝마다 로깅
        max_storage_gb=200.0,         # 200GB 한계
        model_params=200_000_000      # 2억 파라미터
    )
    
    print(f"📊 학습 설정:")
    print(f"   - 후회 횟수/스텝: {config.regrets_per_step}")
    print(f"   - 벤담 계산/후회: {config.bentham_calculations_per_regret}")
    print(f"   - 총 벤담 계산/스텝: {config.total_bentham_per_step}")
    print(f"   - 에포크: {config.epochs}")
    print(f"   - 배치 크기: {config.batch_size}")
    print(f"   - 로깅 주기: {config.log_every_n_steps}스텝")
    print(f"   - 스토리지 한계: {config.max_storage_gb}GB")
    print("=" * 60)
    
    try:
        # 학습기 생성
        trainer = RegretTrainer(config)
        
        # 학습 실행
        print("🎯 학습 시작...")
        report, checkpoint_path = trainer.train()
        
        print("\n" + "=" * 60)
        print("🎉 학습 완료!")
        print(f"📊 최종 통계:")
        print(f"   - 총 후회: {sum(trainer.training_stats['regret_count']):,}")
        print(f"   - 총 벤담 계산: {sum(trainer.training_stats['bentham_count']):,}")
        print(f"   - 총 스텝: {len(trainer.training_stats['total_loss'])}")
        print(f"💾 체크포인트: {checkpoint_path}")
        print("=" * 60)
        
        # 결과 분석 및 문서 생성
        print("\n📊 결과 분석 및 문서 생성 중...")
        analyzer = TrainingResultsAnalyzer(project_root)
        
        # 분석 실행
        analysis_results = analyzer.analyze_and_generate_docs()
        
        print("\n" + "=" * 60)
        print("📚 문서 생성 완료!")
        print(f"📄 마크다운 리포트: {analysis_results['markdown_report']}")
        print(f"🌐 HTML 리포트: {analysis_results['html_report']}")
        print(f"📊 시각화: {len(analysis_results['visualizations'])}개")
        print(f"📁 docs 폴더에 결과표가 저장되었습니다.")
        print("=" * 60)
        
        # 권장사항 출력
        recommendations = analysis_results['analysis']['recommendations']
        if recommendations:
            print("\n💡 권장사항:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        print(f"\n🎊 Red Heart XAI 학습 완전 성공! 🎊")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 학습 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)