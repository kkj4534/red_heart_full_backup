#!/usr/bin/env python3
"""
백그라운드 데이터 처리 실행 스크립트
"""

import asyncio
import sys
import signal
import logging
from pathlib import Path
import argparse
from datetime import datetime

# 프로젝트 루트를 Python path에 추가
sys.path.append(str(Path(__file__).parent))

from llm_dataset_processor import DatasetProcessor

# 로깅 설정
def setup_logging(log_level: str = "INFO"):
    """로깅 설정"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"dataset_processing_{timestamp}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger('DatasetProcessingMain')
    logger.info(f"로그 파일: {log_file}")
    return logger

# 전역 변수
processor = None
processing_task = None

def signal_handler(sig, frame):
    """시그널 핸들러 (Ctrl+C 처리)"""
    print("\n🛑 처리 중단 요청을 받았습니다...")
    if processing_task:
        processing_task.cancel()
    sys.exit(0)

async def run_processing(output_dir: str = "enhanced_training_data", chunk_size: int = 50):
    """데이터 처리 실행"""
    global processor
    
    logger = logging.getLogger('DatasetProcessingMain')
    
    try:
        # 프로세서 초기화
        processor = DatasetProcessor(output_dir=output_dir)
        processor.max_chunk_size = chunk_size
        
        logger.info("=== 데이터셋 처리 시작 ===")
        logger.info(f"출력 디렉토리: {output_dir}")
        logger.info(f"청크 크기: {chunk_size}")
        
        await processor.initialize()
        
        # 처리 시작
        start_time = datetime.now()
        processed_scenarios = await processor.process_all_sources()
        end_time = datetime.now()
        
        # 결과 요약
        processing_time = end_time - start_time
        logger.info("=== 처리 완료 ===")
        logger.info(f"총 처리된 시나리오 수: {len(processed_scenarios)}")
        logger.info(f"처리 시간: {processing_time}")
        logger.info(f"평균 시나리오당 처리 시간: {processing_time.total_seconds() / len(processed_scenarios):.2f}초")
        
        # 소스별 통계
        source_stats = {}
        for scenario in processed_scenarios:
            source_stats[scenario.source_type] = source_stats.get(scenario.source_type, 0) + 1
        
        logger.info("소스별 통계:")
        for source, count in source_stats.items():
            logger.info(f"  {source}: {count}개")
        
        print(f"\n✅ 처리 완료! 총 {len(processed_scenarios)}개 시나리오가 {output_dir}에 저장되었습니다.")
        
    except asyncio.CancelledError:
        logger.info("처리가 사용자에 의해 중단되었습니다.")
        print("처리가 중단되었습니다.")
    except Exception as e:
        logger.error(f"처리 중 오류 발생: {e}")
        print(f"❌ 오류 발생: {e}")
        raise

async def run_quick_test(max_scenarios: int = 10):
    """빠른 테스트 실행 (소량 데이터로 테스트)"""
    logger = logging.getLogger('DatasetProcessingMain')
    
    try:
        processor = DatasetProcessor(output_dir="test_output")
        processor.max_chunk_size = max_scenarios
        
        logger.info(f"=== 빠른 테스트 시작 (최대 {max_scenarios}개 시나리오) ===")
        
        await processor.initialize()
        
        # EBS 파일 하나만 테스트
        test_scenarios = await processor.process_data_source(
            'for_learn_dataset/ai_ebs/ebs_1.txt', 
            'ebs_literature'
        )
        
        logger.info(f"테스트 완료: {len(test_scenarios)}개 시나리오 처리")
        
        # 샘플 결과 출력
        if test_scenarios:
            sample = test_scenarios[0]
            print(f"\n📝 샘플 결과:")
            print(f"제목: {sample.title}")
            print(f"카테고리: {sample.category}")
            print(f"벤담 강도: {sample.bentham_factors.intensity}")
            print(f"Entailment 쌍 수: {len(sample.entailment_pairs)}")
        
        print("✅ 빠른 테스트 완료!")
        
    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {e}")
        print(f"❌ 테스트 오류: {e}")
        raise

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='LLM 데이터셋 처리기')
    parser.add_argument('--mode', choices=['full', 'test'], default='full',
                        help='실행 모드: full (전체 처리) 또는 test (빠른 테스트)')
    parser.add_argument('--output-dir', default='enhanced_training_data',
                        help='출력 디렉토리 (기본값: enhanced_training_data)')
    parser.add_argument('--chunk-size', type=int, default=50,
                        help='청크 크기 (기본값: 50)')
    parser.add_argument('--max-scenarios', type=int, default=10,
                        help='테스트 모드에서 최대 시나리오 수 (기본값: 10)')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='로그 레벨 (기본값: INFO)')
    
    args = parser.parse_args()
    
    # 로깅 설정
    logger = setup_logging(args.log_level)
    
    # 시그널 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        if args.mode == 'test':
            print("🧪 빠른 테스트 모드로 실행합니다...")
            asyncio.run(run_quick_test(args.max_scenarios))
        else:
            print("🚀 전체 처리 모드로 실행합니다...")
            print("처리 중단하려면 Ctrl+C를 누르세요.")
            global processing_task
            processing_task = asyncio.create_task(
                run_processing(args.output_dir, args.chunk_size)
            )
            asyncio.run(processing_task)
            
    except KeyboardInterrupt:
        print("\n처리가 중단되었습니다.")
    except Exception as e:
        logger.error(f"실행 중 오류: {e}")
        print(f"❌ 실행 오류: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()