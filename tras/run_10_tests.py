#!/usr/bin/env python3
"""
10회 연속 통합 학습 테스트 실행기
Continuous 10x Integrated Learning Test Runner
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
import logging

# 기존 통합 학습 테스트 임포트
from integrated_learning_test import run_learning_test

def setup_test_logging():
    """테스트 전용 로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('Continuous10Tests')
    return logger

async def run_continuous_tests(num_tests=10):
    """10회 연속 테스트 실행"""
    logger = setup_test_logging()
    
    # 결과 저장 리스트
    all_results = []
    
    logger.info(f"🚀 {num_tests}회 연속 통합 학습 테스트 시작")
    logger.info("="*80)
    
    total_start_time = time.time()
    
    for test_run in range(1, num_tests + 1):
        logger.info(f"\n📋 테스트 {test_run}/{num_tests} 시작...")
        
        run_start_time = time.time()
        
        try:
            # 개별 학습 테스트 실행
            results = await run_learning_test()
            
            if results:
                run_time = time.time() - run_start_time
                
                # 결과 요약
                test_summary = {
                    'test_run': test_run,
                    'timestamp': datetime.now().isoformat(),
                    'run_time': run_time,
                    'status': 'success',
                    'results': results[0] if isinstance(results, tuple) else results,
                    'analytics': results[1] if isinstance(results, tuple) and len(results) > 1 else None
                }
                
                all_results.append(test_summary)
                
                logger.info(f"✅ 테스트 {test_run} 완료 - 소요시간: {run_time:.2f}초")
                
                # 간단한 성능 요약
                if isinstance(results, tuple) and len(results) > 0:
                    result_data = results[0]
                    if isinstance(result_data, dict):
                        final_acc = result_data.get('final_train_accuracy', 0)
                        final_loss = result_data.get('final_train_loss', 0)
                        epochs = result_data.get('epochs_completed', 0)
                        logger.info(f"   - 최종 정확도: {final_acc:.4f}")
                        logger.info(f"   - 최종 손실: {final_loss:.4f}")
                        logger.info(f"   - 완료 에포크: {epochs}")
            else:
                logger.error(f"❌ 테스트 {test_run} 실패 - 결과 없음")
                test_summary = {
                    'test_run': test_run,
                    'timestamp': datetime.now().isoformat(),
                    'run_time': time.time() - run_start_time,
                    'status': 'failed',
                    'error': 'No results returned'
                }
                all_results.append(test_summary)
                
        except Exception as e:
            run_time = time.time() - run_start_time
            logger.error(f"❌ 테스트 {test_run} 오류: {str(e)}")
            
            test_summary = {
                'test_run': test_run,
                'timestamp': datetime.now().isoformat(),
                'run_time': run_time,
                'status': 'error',
                'error': str(e)
            }
            all_results.append(test_summary)
        
        # 테스트 간 잠깐 대기 (GPU 메모리 정리)
        if test_run < num_tests:
            logger.info("   🔄 다음 테스트 준비 중...")
            await asyncio.sleep(2)
    
    total_time = time.time() - total_start_time
    
    # 전체 결과 요약
    logger.info("\n" + "="*80)
    logger.info("📊 전체 테스트 결과 요약")
    logger.info("="*80)
    
    successful_tests = [r for r in all_results if r['status'] == 'success']
    failed_tests = [r for r in all_results if r['status'] != 'success']
    
    logger.info(f"✅ 성공한 테스트: {len(successful_tests)}/{num_tests}")
    logger.info(f"❌ 실패한 테스트: {len(failed_tests)}/{num_tests}")
    logger.info(f"⏱️ 총 소요시간: {total_time:.2f}초")
    logger.info(f"📈 평균 테스트 시간: {total_time/num_tests:.2f}초")
    
    if successful_tests:
        avg_run_time = sum(r['run_time'] for r in successful_tests) / len(successful_tests)
        logger.info(f"⚡ 성공한 테스트 평균 시간: {avg_run_time:.2f}초")
        
        # 성능 통계 (성공한 테스트만)
        successful_results = [r['results'] for r in successful_tests if r['results']]
        if successful_results:
            accuracies = [r.get('final_train_accuracy', 0) for r in successful_results]
            losses = [r.get('final_train_loss', 0) for r in successful_results]
            
            if accuracies:
                avg_accuracy = sum(accuracies) / len(accuracies)
                min_accuracy = min(accuracies)
                max_accuracy = max(accuracies)
                logger.info(f"🎯 평균 정확도: {avg_accuracy:.4f} (범위: {min_accuracy:.4f} - {max_accuracy:.4f})")
            
            if losses:
                avg_loss = sum(losses) / len(losses)
                min_loss = min(losses)
                max_loss = max(losses)
                logger.info(f"📉 평균 손실: {avg_loss:.4f} (범위: {min_loss:.4f} - {max_loss:.4f})")
    
    # 결과를 JSON 파일로 저장
    results_file = Path(f"logs/continuous_10_tests_{int(time.time())}.json")
    results_file.parent.mkdir(exist_ok=True)
    
    final_report = {
        'total_tests': num_tests,
        'successful_tests': len(successful_tests),
        'failed_tests': len(failed_tests),
        'total_time': total_time,
        'average_test_time': total_time / num_tests,
        'timestamp': datetime.now().isoformat(),
        'individual_results': all_results
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 상세 결과 저장: {results_file}")
    
    # 성공률 체크
    success_rate = len(successful_tests) / num_tests * 100
    
    if success_rate >= 80:
        logger.info(f"🏆 시스템 안정성 우수: {success_rate:.1f}% 성공률")
    elif success_rate >= 60:
        logger.info(f"✅ 시스템 안정성 양호: {success_rate:.1f}% 성공률")
    else:
        logger.warning(f"⚠️ 시스템 안정성 개선 필요: {success_rate:.1f}% 성공률")
    
    return final_report

if __name__ == "__main__":
    # 10회 연속 테스트 실행
    asyncio.run(run_continuous_tests(10))