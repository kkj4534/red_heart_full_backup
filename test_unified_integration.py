#!/usr/bin/env python3
"""
Red Heart AI 통합 시스템 테스트
전체 파이프라인 검증 및 통합 테스트
"""

import asyncio
import torch
import time
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger('TestUnifiedIntegration')

# 프로젝트 경로 추가
import sys
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from main_unified import UnifiedInferenceSystem, InferenceConfig, MemoryMode


class UnifiedSystemTester:
    """통합 시스템 테스터"""
    
    def __init__(self, memory_mode: str = "normal", verbose: bool = False):
        self.memory_mode = MemoryMode[memory_mode.upper()]
        self.verbose = verbose
        self.results = {
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'tests': []
        }
        self.system = None
        
    async def initialize(self):
        """시스템 초기화"""
        logger.info("테스트 시스템 초기화...")
        
        config = InferenceConfig(
            memory_mode=self.memory_mode,
            auto_memory_mode=False,
            debug=self.verbose
        )
        
        self.system = UnifiedInferenceSystem(config)
        await self.system.initialize()
        
        logger.info(f"✅ 시스템 초기화 완료 (모드: {self.memory_mode.value})")
    
    async def test_basic_inference(self):
        """기본 추론 테스트"""
        test_name = "기본 추론"
        logger.info(f"🧪 테스트: {test_name}")
        
        try:
            text = "오늘 날씨가 좋아서 기분이 좋습니다."
            result = await self.system.analyze(text)
            
            # 결과 검증
            assert 'emotion' in result, "감정 분석 결과 없음"
            assert 'bentham' in result, "벤담 계산 결과 없음"
            assert 'processing_time' in result, "처리 시간 없음"
            
            if self.verbose:
                logger.info(f"   감정: {result.get('emotion', {})}")
                logger.info(f"   벤담: {result.get('bentham', {})}")
                logger.info(f"   처리 시간: {result.get('processing_time', 0):.2f}초")
            
            self.results['passed'] += 1
            self.results['tests'].append({
                'name': test_name,
                'status': 'passed',
                'time': result.get('processing_time', 0)
            })
            logger.info(f"   ✅ {test_name} 통과")
            
        except Exception as e:
            self.results['failed'] += 1
            self.results['tests'].append({
                'name': test_name,
                'status': 'failed',
                'error': str(e)
            })
            logger.error(f"   ❌ {test_name} 실패: {e}")
    
    async def test_pipeline_connection(self):
        """파이프라인 연결 테스트"""
        test_name = "파이프라인 연결"
        logger.info(f"🧪 테스트: {test_name}")
        
        try:
            text = "중요한 결정을 앞두고 고민이 많습니다."
            result = await self.system.analyze(text)
            
            # 파이프라인 단계별 검증
            checks = []
            
            # 1. 감정 → 벤담 연결
            if 'emotion' in result and 'bentham' in result:
                bentham = result['bentham']
                if isinstance(bentham, dict) and 'intensity' in bentham:
                    checks.append("감정→벤담")
            
            # 2. 반사실 추론
            if 'counterfactuals' in result:
                checks.append("반사실추론")
            
            # 3. 후회 계산
            if 'regret' in result:
                checks.append("후회계산")
            
            # 4. 시계열 전파
            if self.memory_mode.value in ['normal', 'heavy', 'ultra', 'extreme']:
                if 'temporal_impact' in result:
                    checks.append("시계열전파")
            
            # 5. 메타 통합
            if self.memory_mode.value == 'extreme':
                if 'meta_integrated' in result:
                    checks.append("메타통합")
            
            if self.verbose:
                logger.info(f"   연결된 단계: {' → '.join(checks)}")
            
            assert len(checks) >= 3, f"파이프라인 연결 부족 ({len(checks)}/5)"
            
            self.results['passed'] += 1
            self.results['tests'].append({
                'name': test_name,
                'status': 'passed',
                'pipeline': checks
            })
            logger.info(f"   ✅ {test_name} 통과 ({len(checks)}단계 연결)")
            
        except Exception as e:
            self.results['failed'] += 1
            self.results['tests'].append({
                'name': test_name,
                'status': 'failed',
                'error': str(e)
            })
            logger.error(f"   ❌ {test_name} 실패: {e}")
    
    async def test_memory_mode_modules(self):
        """메모리 모드별 모듈 활성화 테스트"""
        test_name = f"메모리 모드 모듈 ({self.memory_mode.value})"
        logger.info(f"🧪 테스트: {test_name}")
        
        try:
            # 모드별 예상 모듈
            expected_modules = {
                MemoryMode.MINIMAL: [],
                MemoryMode.LIGHT: [],
                MemoryMode.NORMAL: ['dsp_simulator', 'kalman_filter'],
                MemoryMode.HEAVY: ['dsp_simulator', 'kalman_filter', 'neural_analyzers'],
                MemoryMode.ULTRA: ['dsp_simulator', 'kalman_filter', 'neural_analyzers', 'advanced_wrappers'],
                MemoryMode.EXTREME: ['dsp_simulator', 'kalman_filter', 'neural_analyzers', 
                                    'advanced_wrappers', 'meta_integration', 'counterfactual_reasoning']
            }
            
            expected = expected_modules[self.memory_mode]
            actual = []
            
            if self.system.dsp_simulator is not None:
                actual.append('dsp_simulator')
            if self.system.kalman_filter is not None:
                actual.append('kalman_filter')
            if self.system.neural_analyzers is not None:
                actual.append('neural_analyzers')
            if self.system.advanced_wrappers is not None:
                actual.append('advanced_wrappers')
            if self.system.meta_integration is not None:
                actual.append('meta_integration')
            if self.system.counterfactual_reasoning is not None:
                actual.append('counterfactual_reasoning')
            
            if self.verbose:
                logger.info(f"   예상 모듈: {expected}")
                logger.info(f"   실제 모듈: {actual}")
            
            # 예상 모듈이 모두 로드되었는지 확인
            for module in expected:
                assert module in actual, f"{module} 모듈이 로드되지 않음"
            
            self.results['passed'] += 1
            self.results['tests'].append({
                'name': test_name,
                'status': 'passed',
                'modules': actual
            })
            logger.info(f"   ✅ {test_name} 통과 ({len(actual)}개 모듈)")
            
        except Exception as e:
            self.results['failed'] += 1
            self.results['tests'].append({
                'name': test_name,
                'status': 'failed',
                'error': str(e)
            })
            logger.error(f"   ❌ {test_name} 실패: {e}")
    
    async def test_emotion_to_bentham(self):
        """감정→벤담 변환 테스트"""
        test_name = "감정→벤담 변환"
        logger.info(f"🧪 테스트: {test_name}")
        
        try:
            # 테스트 감정 데이터
            emotion_data = {
                'scores': [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
                'hierarchy': {'community': True, 'other': True, 'self': True}
            }
            
            # 변환 실행
            bentham_params = self.system.emotion_to_bentham_converter(emotion_data)
            
            # 결과 검증
            required_keys = ['intensity', 'duration', 'certainty', 
                           'propinquity', 'fecundity', 'purity', 'extent']
            
            for key in required_keys:
                assert key in bentham_params, f"벤담 파라미터 '{key}' 없음"
                assert 0 <= bentham_params[key] <= 2, f"벤담 파라미터 '{key}' 범위 초과"
            
            # 계층적 가중치 적용 확인
            assert bentham_params['extent'] > 0.2, "계층적 가중치 미적용"
            
            if self.verbose:
                logger.info(f"   변환 결과: {bentham_params}")
            
            self.results['passed'] += 1
            self.results['tests'].append({
                'name': test_name,
                'status': 'passed',
                'bentham': bentham_params
            })
            logger.info(f"   ✅ {test_name} 통과")
            
        except Exception as e:
            self.results['failed'] += 1
            self.results['tests'].append({
                'name': test_name,
                'status': 'failed',
                'error': str(e)
            })
            logger.error(f"   ❌ {test_name} 실패: {e}")
    
    async def test_batch_processing(self):
        """배치 처리 테스트"""
        test_name = "배치 처리"
        logger.info(f"🧪 테스트: {test_name}")
        
        try:
            texts = [
                "첫 번째 테스트 문장입니다.",
                "두 번째 테스트 문장입니다.",
                "세 번째 테스트 문장입니다."
            ]
            
            start_time = time.time()
            results = []
            
            for text in texts:
                result = await self.system.analyze(text)
                results.append(result)
            
            total_time = time.time() - start_time
            avg_time = total_time / len(texts)
            
            # 모든 결과가 있는지 확인
            assert len(results) == len(texts), "배치 처리 결과 수 불일치"
            
            for i, result in enumerate(results):
                assert 'error' not in result, f"배치 {i+1} 처리 오류"
            
            if self.verbose:
                logger.info(f"   총 처리 시간: {total_time:.2f}초")
                logger.info(f"   평균 처리 시간: {avg_time:.2f}초")
            
            self.results['passed'] += 1
            self.results['tests'].append({
                'name': test_name,
                'status': 'passed',
                'batch_size': len(texts),
                'avg_time': avg_time
            })
            logger.info(f"   ✅ {test_name} 통과 (평균 {avg_time:.2f}초)")
            
        except Exception as e:
            self.results['failed'] += 1
            self.results['tests'].append({
                'name': test_name,
                'status': 'failed',
                'error': str(e)
            })
            logger.error(f"   ❌ {test_name} 실패: {e}")
    
    async def test_cache_functionality(self):
        """캐시 기능 테스트"""
        test_name = "캐시 기능"
        logger.info(f"🧪 테스트: {test_name}")
        
        try:
            text = "캐시 테스트를 위한 문장입니다."
            
            # 첫 번째 분석
            result1 = await self.system.analyze(text)
            time1 = result1.get('processing_time', 0)
            
            # 두 번째 분석 (캐시 히트 예상)
            result2 = await self.system.analyze(text)
            time2 = result2.get('processing_time', 0)
            
            # 캐시가 작동하면 두 번째가 더 빨라야 함
            assert time2 <= time1, "캐시 성능 개선 없음"
            
            if self.verbose:
                logger.info(f"   첫 번째 처리: {time1:.3f}초")
                logger.info(f"   두 번째 처리: {time2:.3f}초")
                logger.info(f"   성능 개선: {((time1-time2)/time1*100):.1f}%")
            
            self.results['passed'] += 1
            self.results['tests'].append({
                'name': test_name,
                'status': 'passed',
                'improvement': f"{((time1-time2)/time1*100):.1f}%"
            })
            logger.info(f"   ✅ {test_name} 통과")
            
        except Exception as e:
            self.results['failed'] += 1
            self.results['tests'].append({
                'name': test_name,
                'status': 'failed',
                'error': str(e)
            })
            logger.error(f"   ❌ {test_name} 실패: {e}")
    
    async def run_all_tests(self):
        """모든 테스트 실행"""
        logger.info("="*60)
        logger.info(f"🚀 통합 테스트 시작 (메모리 모드: {self.memory_mode.value})")
        logger.info("="*60)
        
        # 시스템 초기화
        await self.initialize()
        
        # 테스트 목록
        tests = [
            self.test_basic_inference,
            self.test_pipeline_connection,
            self.test_memory_mode_modules,
            self.test_emotion_to_bentham,
            self.test_batch_processing,
            self.test_cache_functionality
        ]
        
        # 테스트 실행
        for test in tests:
            await test()
            await asyncio.sleep(0.5)  # 테스트 간 짧은 대기
        
        # 결과 출력
        self.print_summary()
        
        # JSON 결과 저장
        self.save_results()
        
        return self.results['failed'] == 0
    
    def print_summary(self):
        """테스트 요약 출력"""
        logger.info("="*60)
        logger.info("📊 테스트 결과 요약")
        logger.info("="*60)
        
        total = self.results['passed'] + self.results['failed'] + self.results['skipped']
        
        logger.info(f"총 테스트: {total}")
        logger.info(f"✅ 성공: {self.results['passed']}")
        logger.info(f"❌ 실패: {self.results['failed']}")
        logger.info(f"⏭️ 건너뜀: {self.results['skipped']}")
        
        if total > 0:
            success_rate = (self.results['passed'] / total) * 100
            logger.info(f"성공률: {success_rate:.1f}%")
            
            if success_rate == 100:
                logger.info("🎉 모든 테스트 통과!")
            elif success_rate >= 80:
                logger.info("👍 대부분의 테스트 통과")
            else:
                logger.warning("⚠️ 개선 필요")
    
    def save_results(self):
        """결과를 JSON 파일로 저장"""
        results_file = f"test_results_{self.memory_mode.value}_{int(time.time())}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📁 결과 저장: {results_file}")


async def main():
    parser = argparse.ArgumentParser(description='Red Heart AI 통합 시스템 테스트')
    parser.add_argument('--memory-mode', default='normal',
                       choices=['minimal', 'light', 'normal', 'heavy', 'ultra', 'extreme'],
                       help='메모리 모드')
    parser.add_argument('--verbose', action='store_true',
                       help='상세 출력')
    
    args = parser.parse_args()
    
    # 테스터 생성 및 실행
    tester = UnifiedSystemTester(
        memory_mode=args.memory_mode,
        verbose=args.verbose
    )
    
    success = await tester.run_all_tests()
    
    # 종료 코드 반환
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    asyncio.run(main())