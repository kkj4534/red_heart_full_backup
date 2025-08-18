#!/usr/bin/env python3
"""
업그레이드된 10회 연속 통합 학습 테스트 실행기
Upgraded 10x Continuous Integrated Learning Test Runner

Features:
- 동적 GPU 메모리 관리
- 견고한 로깅 시스템
- 강제 결과 파일 생성 보장
- 정확한 GPU 메모리 추적
- 실시간 성능 모니터링
"""

import asyncio
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
import logging

# 업그레이드된 시스템 임포트
from dynamic_gpu_manager import get_gpu_manager, optimize_gpu_for_learning, emergency_gpu_cleanup, allocate_gpu_memory
from robust_logging_system import get_robust_logger, test_session, add_performance_sample, generate_test_report

# 기존 통합 학습 테스트 임포트
from integrated_learning_test import run_learning_test, LearningConfig, IntegratedLearningFramework

def setup_upgraded_logging():
    """업그레이드된 로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('UpgradedContinuous10Tests')
    return logger

async def run_single_optimized_test(test_number: int, gpu_manager, robust_logger) -> dict:
    """최적화된 단일 테스트 실행"""
    
    test_name = f"integrated_learning_test_{test_number:02d}"
    
    with test_session(test_name, {"test_number": test_number}) as test_id:
        robust_logger.log("INFO", "TestRunner", f"테스트 {test_number} 시작", {"test_id": test_id})
        
        start_time = time.time()
        
        try:
            # GPU 메모리 최적화
            with allocate_gpu_memory('main_learning_pipeline', dynamic_boost=True):
                # 성능 샘플 추가
                add_performance_sample({
                    "phase": "start",
                    "test_number": test_number,
                    "gpu_status": gpu_manager.get_memory_status()
                })
                
                # 실제 학습 테스트 실행
                results = await run_learning_test()
                
                # 중간 성능 샘플
                add_performance_sample({
                    "phase": "learning_complete",
                    "test_number": test_number,
                    "gpu_status": gpu_manager.get_memory_status(),
                    "results_available": results is not None
                })
                
                if results:
                    run_time = time.time() - start_time
                    
                    # 결과 처리
                    if isinstance(results, tuple):
                        result_data = results[0]
                        analytics_data = results[1] if len(results) > 1 else None
                    else:
                        result_data = results
                        analytics_data = None
                    
                    # 최종 성능 샘플
                    add_performance_sample({
                        "phase": "complete",
                        "test_number": test_number,
                        "run_time": run_time,
                        "gpu_status": gpu_manager.get_memory_status(),
                        "result_data": result_data if isinstance(result_data, dict) else None
                    })
                    
                    # 성공 결과 반환
                    success_result = {
                        'test_number': test_number,
                        'test_id': test_id,
                        'status': 'success',
                        'run_time': run_time,
                        'timestamp': datetime.now().isoformat(),
                        'results': result_data,
                        'analytics': analytics_data,
                        'gpu_memory_usage': gpu_manager.get_memory_status()
                    }
                    
                    robust_logger.log("INFO", "TestRunner", 
                                     f"테스트 {test_number} 성공 완료 - {run_time:.2f}초",
                                     {"test_id": test_id, "run_time": run_time})
                    
                    return success_result
                    
                else:
                    # 결과 없음
                    run_time = time.time() - start_time
                    
                    failure_result = {
                        'test_number': test_number,
                        'test_id': test_id,
                        'status': 'failed',
                        'run_time': run_time,
                        'timestamp': datetime.now().isoformat(),
                        'error': 'No results returned from learning test',
                        'gpu_memory_usage': gpu_manager.get_memory_status()
                    }
                    
                    robust_logger.log("ERROR", "TestRunner",
                                     f"테스트 {test_number} 실패 - 결과 없음",
                                     {"test_id": test_id, "run_time": run_time})
                    
                    return failure_result
                    
        except Exception as e:
            # 예외 처리
            run_time = time.time() - start_time
            error_info = {
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc()
            }
            
            error_result = {
                'test_number': test_number,
                'test_id': test_id,
                'status': 'error',
                'run_time': run_time,
                'timestamp': datetime.now().isoformat(),
                'error_info': error_info,
                'gpu_memory_usage': gpu_manager.get_memory_status()
            }
            
            robust_logger.log("ERROR", "TestRunner",
                             f"테스트 {test_number} 오류: {str(e)}",
                             {"test_id": test_id, "error_info": error_info})
            
            return error_result

async def run_upgraded_continuous_tests(num_tests: int = 10):
    """업그레이드된 10회 연속 테스트 실행"""
    
    logger = setup_upgraded_logging()
    gpu_manager = get_gpu_manager()
    robust_logger = get_robust_logger()
    
    logger.info(f"🚀 업그레이드된 {num_tests}회 연속 통합 학습 테스트 시작")
    logger.info("=" * 80)
    
    # 전체 테스트 세션 시작
    with test_session(f"continuous_{num_tests}_tests", {"total_tests": num_tests}) as session_id:
        
        robust_logger.log("INFO", "ContinuousTestRunner", 
                         f"{num_tests}회 연속 테스트 세션 시작",
                         {"session_id": session_id, "total_tests": num_tests})
        
        total_start_time = time.time()
        
        # 초기 GPU 최적화
        optimization_success = optimize_gpu_for_learning()
        robust_logger.log("INFO", "GPU_Optimization", 
                         f"초기 GPU 최적화: {'성공' if optimization_success else '제한됨'}")
        
        # 결과 저장 리스트
        all_results = []
        
        for test_run in range(1, num_tests + 1):
            logger.info(f"\n📋 테스트 {test_run}/{num_tests} 실행 중...")
            
            try:
                # 개별 테스트 실행
                test_result = await run_single_optimized_test(test_run, gpu_manager, robust_logger)
                all_results.append(test_result)
                
                # 진행 상황 로깅
                if test_result['status'] == 'success':
                    logger.info(f"✅ 테스트 {test_run} 성공 - {test_result['run_time']:.2f}초")
                    if 'results' in test_result and isinstance(test_result['results'], dict):
                        result_data = test_result['results']
                        final_acc = result_data.get('final_train_accuracy', 0)
                        final_loss = result_data.get('final_train_loss', 0)
                        logger.info(f"   📊 정확도: {final_acc:.4f}, 손실: {final_loss:.4f}")
                else:
                    logger.error(f"❌ 테스트 {test_run} 실패: {test_result.get('error', 'Unknown error')}")
                
                # 메모리 상태 확인
                gpu_status = gpu_manager.get_memory_status()
                logger.info(f"   🔧 GPU 메모리: {gpu_status['allocated_gb']:.1f}/{gpu_status['total_gb']:.1f}GB")
                
            except Exception as e:
                logger.error(f"❌ 테스트 {test_run} 치명적 오류: {e}")
                
                # 응급 오류 결과 생성
                emergency_result = {
                    'test_number': test_run,
                    'test_id': f"emergency_{test_run}",
                    'status': 'fatal_error',
                    'run_time': 0,
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e),
                    'gpu_memory_usage': gpu_manager.get_memory_status()
                }
                all_results.append(emergency_result)
                
                # 응급 메모리 정리
                emergency_gpu_cleanup()
            
            # 테스트 간 대기 및 메모리 정리
            if test_run < num_tests:
                logger.info("   🔄 다음 테스트 준비 중...")
                await asyncio.sleep(3)  # 좀 더 여유있는 대기
                
                # 중간 메모리 정리
                gpu_manager.emergency_cleanup()
        
        total_time = time.time() - total_start_time
        
        # 전체 결과 분석
        logger.info("\n" + "=" * 80)
        logger.info("📊 업그레이드된 연속 테스트 결과 분석")
        logger.info("=" * 80)
        
        successful_tests = [r for r in all_results if r['status'] == 'success']
        failed_tests = [r for r in all_results if r['status'] in ['failed', 'error', 'fatal_error']]
        
        success_rate = len(successful_tests) / num_tests * 100
        
        logger.info(f"✅ 성공한 테스트: {len(successful_tests)}/{num_tests} ({success_rate:.1f}%)")
        logger.info(f"❌ 실패한 테스트: {len(failed_tests)}/{num_tests}")
        logger.info(f"⏱️ 총 소요시간: {total_time:.2f}초")
        logger.info(f"📈 평균 테스트 시간: {total_time/num_tests:.2f}초")
        
        if successful_tests:
            avg_run_time = sum(r['run_time'] for r in successful_tests) / len(successful_tests)
            logger.info(f"⚡ 성공한 테스트 평균 시간: {avg_run_time:.2f}초")
        
        # 견고한 로깅 시스템을 통한 최종 보고서 생성
        final_report = generate_test_report(num_tests)
        
        # 강제 결과 파일 생성 보장
        timestamp = int(time.time())
        
        # 상세 결과 파일
        results_file = Path(f"logs/upgraded_continuous_{num_tests}_tests_{timestamp}.json")
        results_file.parent.mkdir(exist_ok=True)
        
        comprehensive_report = {
            'session_info': {
                'session_id': session_id,
                'test_type': 'upgraded_continuous',
                'total_tests': num_tests,
                'timestamp': datetime.now().isoformat(),
                'total_time': total_time,
                'success_rate': success_rate
            },
            'gpu_optimization': {
                'initial_optimization': optimization_success,
                'final_memory_status': gpu_manager.get_memory_status()
            },
            'individual_results': all_results,
            'robust_logging_report': final_report,
            'performance_summary': {
                'successful_tests': len(successful_tests),
                'failed_tests': len(failed_tests),
                'avg_test_time': total_time / num_tests,
                'total_duration': total_time
            }
        }
        
        # 강제 파일 저장 (다중 안전장치)
        try:
            # 주 결과 파일
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(comprehensive_report, f, indent=2, ensure_ascii=False)
                f.flush()
                import os
                os.fsync(f.fileno())  # 강제 디스크 동기화
            
            # 백업 결과 파일
            backup_file = Path(f"logs/backup_continuous_{num_tests}_tests_{timestamp}.json")
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(comprehensive_report, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            
            logger.info(f"💾 결과 파일 저장 완료: {results_file}")
            logger.info(f"💾 백업 파일 저장 완료: {backup_file}")
            
        except Exception as e:
            logger.error(f"⚠️ 결과 파일 저장 오류: {e}")
            # 응급 저장 시도
            emergency_file = Path(f"emergency_results_{timestamp}.json")
            try:
                with open(emergency_file, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)
                logger.info(f"🚨 응급 결과 파일 저장: {emergency_file}")
            except Exception as e2:
                logger.error(f"🚨 응급 저장도 실패: {e2}")
        
        # 최종 평가
        if success_rate >= 90:
            logger.info(f"🏆 시스템 성능 우수: {success_rate:.1f}% 성공률")
        elif success_rate >= 70:
            logger.info(f"✅ 시스템 성능 양호: {success_rate:.1f}% 성공률")
        else:
            logger.warning(f"⚠️ 시스템 성능 개선 필요: {success_rate:.1f}% 성공률")
        
        robust_logger.log("INFO", "ContinuousTestRunner", 
                         f"연속 테스트 완료 - 성공률: {success_rate:.1f}%",
                         {"total_tests": num_tests, "successful_tests": len(successful_tests)})
        
        return comprehensive_report

if __name__ == "__main__":
    # 업그레이드된 10회 연속 테스트 실행
    print("🚀 업그레이드된 Red Heart AI 연속 학습 테스트 시작")
    print("📋 특징: 동적 GPU 관리 + 견고한 로깅 + 강제 결과 파일 생성")
    
    try:
        asyncio.run(run_upgraded_continuous_tests(10))
    except KeyboardInterrupt:
        print("\n⚠️ 사용자 중단")
    except Exception as e:
        print(f"\n❌ 치명적 오류: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 최종 정리
        try:
            emergency_gpu_cleanup()
            get_robust_logger().shutdown()
            print("✅ 시스템 정리 완료")
        except:
            pass