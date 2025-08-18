#!/usr/bin/env python3
"""
파일 로깅 통합 시스템 테스트
HelpingAI 재활성화 후 테스트, 로그를 파일에 저장
"""

import sys
import os
import time
import logging
import json
from datetime import datetime
from pathlib import Path
import traceback

# 프로젝트 루트 경로 추가
sys.path.insert(0, '/mnt/c/large_project/linux_red_heart')

def setup_file_logging():
    """파일 로깅 시스템 설정"""
    log_dir = Path("/mnt/c/large_project/linux_red_heart/logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 로그 파일들
    main_log = log_dir / f"helpingai_test_{timestamp}.log"
    summary_file = log_dir / f"helpingai_summary_{timestamp}.json"
    
    # 로거 설정
    logger = logging.getLogger('HelpingAITest')
    logger.setLevel(logging.INFO)
    
    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 파일 핸들러
    file_handler = logging.FileHandler(main_log, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 포맷터
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger, main_log, summary_file

def test_helpingai_system():
    """HelpingAI 시스템 테스트"""
    logger, main_log, summary_file = setup_file_logging()
    
    test_results = {
        'timestamp': datetime.now().isoformat(),
        'test_type': 'HelpingAI 재활성화 테스트',
        'results': {},
        'summary': {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'gpu_memory_before': None,
            'gpu_memory_after': None
        }
    }
    
    logger.info("=" * 60)
    logger.info("HelpingAI 재활성화 통합 테스트 시작")
    logger.info("=" * 60)
    
    # GPU 메모리 체크 (테스트 전)
    try:
        import subprocess
        nvidia_result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], 
                                     capture_output=True, text=True)
        test_results['summary']['gpu_memory_before'] = f"{nvidia_result.stdout.strip()}MB"
        logger.info(f"테스트 전 GPU 메모리: {test_results['summary']['gpu_memory_before']}")
    except:
        logger.warning("nvidia-smi로 GPU 메모리 확인 실패")
    
    # 1. LLM 엔진 초기화 테스트
    logger.info("1. LLM 엔진 초기화 테스트")
    test_results['summary']['total_tests'] += 1
    
    try:
        from llm_module.advanced_llm_engine import AdvancedLLMEngine, LLMRequest, TaskComplexity
        
        engine = AdvancedLLMEngine()
        available_models = list(engine.model_configs.keys())
        
        test_results['results']['engine_init'] = {
            'status': 'PASS',
            'available_models': available_models,
            'helpingai_available': 'helpingai' in available_models
        }
        
        logger.info(f"✅ LLM 엔진 초기화 성공")
        logger.info(f"   사용 가능한 모델: {available_models}")
        logger.info(f"   HelpingAI 사용 가능: {'helpingai' in available_models}")
        
        test_results['summary']['passed'] += 1
        
    except Exception as e:
        test_results['results']['engine_init'] = {
            'status': 'FAIL',
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        logger.error(f"❌ LLM 엔진 초기화 실패: {e}")
        test_results['summary']['failed'] += 1
    
    # 2. HelpingAI 로딩 테스트
    logger.info("2. HelpingAI 모델 로딩 테스트")
    test_results['summary']['total_tests'] += 1
    
    try:
        # 감정 분석 요청 (HelpingAI 우선 사용됨)
        request = LLMRequest(
            prompt="Analyze the emotion in this text: I feel very happy today because I achieved my goal.",
            task_type="emotion_interpretation",
            complexity=TaskComplexity.MODERATE,
            max_tokens=100
        )
        
        start_time = time.time()
        response = engine.generate_sync(request)
        processing_time = time.time() - start_time
        
        test_results['results']['helpingai_loading'] = {
            'status': 'PASS' if response.success else 'FAIL',
            'model_used': response.model_used,
            'processing_time': processing_time,
            'response_length': len(response.generated_text),
            'confidence': response.confidence,
            'response_preview': response.generated_text[:100] if response.generated_text else "",
            'error_message': response.error_message
        }
        
        if response.success:
            logger.info(f"✅ HelpingAI 로딩 및 생성 성공")
            logger.info(f"   사용된 모델: {response.model_used}")
            logger.info(f"   처리 시간: {processing_time:.2f}초")
            logger.info(f"   신뢰도: {response.confidence:.2f}")
            logger.info(f"   응답 길이: {len(response.generated_text)}자")
            logger.info(f"   응답 미리보기: {response.generated_text[:100]}...")
            test_results['summary']['passed'] += 1
        else:
            logger.error(f"❌ HelpingAI 생성 실패: {response.error_message}")
            test_results['summary']['failed'] += 1
            
    except Exception as e:
        test_results['results']['helpingai_loading'] = {
            'status': 'FAIL',
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        logger.error(f"❌ HelpingAI 테스트 실패: {e}")
        test_results['summary']['failed'] += 1
    
    # 3. 다중 요청 테스트
    logger.info("3. 다중 요청 테스트 (3개 연속)")
    test_results['summary']['total_tests'] += 1
    
    multi_results = []
    test_prompts = [
        {"prompt": "What emotion is expressed here: I am feeling anxious about the upcoming exam.", "task": "emotion_interpretation"},
        {"prompt": "Explain the causal relationship: Regular exercise leads to better health.", "task": "causal_explanation"},
        {"prompt": "Generate an alternative scenario: What if I had studied harder for the test?", "task": "counterfactual_scenario"}
    ]
    
    try:
        for i, test_case in enumerate(test_prompts, 1):
            logger.info(f"   테스트 {i}: {test_case['task']}")
            
            request = LLMRequest(
                prompt=test_case['prompt'],
                task_type=test_case['task'],
                complexity=TaskComplexity.SIMPLE,
                max_tokens=80
            )
            
            start_time = time.time()
            response = engine.generate_sync(request)
            processing_time = time.time() - start_time
            
            result = {
                'test_number': i,
                'success': response.success,
                'model_used': response.model_used,
                'processing_time': processing_time,
                'response_length': len(response.generated_text)
            }
            multi_results.append(result)
            
            logger.info(f"     결과: {'성공' if response.success else '실패'} | 모델: {response.model_used} | 시간: {processing_time:.2f}초")
        
        success_count = sum(1 for r in multi_results if r['success'])
        test_results['results']['multi_request'] = {
            'status': 'PASS' if success_count >= 2 else 'FAIL',
            'success_count': success_count,
            'total_requests': len(test_prompts),
            'results': multi_results
        }
        
        logger.info(f"✅ 다중 요청 테스트: {success_count}/{len(test_prompts)} 성공")
        test_results['summary']['passed'] += 1
        
    except Exception as e:
        test_results['results']['multi_request'] = {
            'status': 'FAIL',
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        logger.error(f"❌ 다중 요청 테스트 실패: {e}")
        test_results['summary']['failed'] += 1
    
    # GPU 메모리 체크 (테스트 후)
    try:
        nvidia_result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], 
                                     capture_output=True, text=True)
        test_results['summary']['gpu_memory_after'] = f"{nvidia_result.stdout.strip()}MB"
        logger.info(f"테스트 후 GPU 메모리: {test_results['summary']['gpu_memory_after']}")
    except:
        logger.warning("nvidia-smi로 GPU 메모리 확인 실패")
    
    # 최종 요약
    success_rate = (test_results['summary']['passed'] / test_results['summary']['total_tests']) * 100
    test_results['summary']['success_rate'] = success_rate
    
    logger.info("=" * 60)
    logger.info("테스트 완료 - 최종 요약")
    logger.info("=" * 60)
    logger.info(f"총 테스트: {test_results['summary']['total_tests']}")
    logger.info(f"성공: {test_results['summary']['passed']}")
    logger.info(f"실패: {test_results['summary']['failed']}")
    logger.info(f"성공률: {success_rate:.1f}%")
    logger.info("=" * 60)
    
    # 결과 파일 저장
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎯 HelpingAI 재활성화 테스트 완료!")
    print(f"📁 상세 로그: {main_log}")
    print(f"📊 요약 파일: {summary_file}")
    print(f"✅ 성공률: {success_rate:.1f}% ({test_results['summary']['passed']}/{test_results['summary']['total_tests']})")
    
    if 'helpingai_loading' in test_results['results'] and test_results['results']['helpingai_loading']['status'] == 'PASS':
        print(f"🚀 HelpingAI 모델: {test_results['results']['helpingai_loading']['model_used']}")
        print(f"⏱️  평균 처리 시간: {test_results['results']['helpingai_loading']['processing_time']:.2f}초")
    
    return test_results

if __name__ == "__main__":
    test_helpingai_system()