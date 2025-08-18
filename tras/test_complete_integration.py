"""
Linux Red Heart 시스템 완전 통합 테스트
GPU 가속 및 고급 AI 기능 포함 전체 시스템 검증
"""

import asyncio
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

# 마이그레이션된 고급 모듈들 임포트
from advanced_system_integration import AdvancedRedHeartSystem
from advanced_regret_analyzer import AdvancedRegretAnalyzer
from data_models import DecisionScenario, EmotionState
from config import SYSTEM_CONFIG

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompleteIntegrationTest:
    """완전 통합 시스템 테스트"""
    
    def __init__(self):
        self.test_results = {
            'gpu_acceleration': False,
            'transformer_integration': False,
            'regret_analysis': False,
            'real_time_processing': False,
            'system_integration': False,
            'performance_benchmarks': {},
            'error_logs': []
        }
        
        # 테스트 시나리오
        self.test_scenarios = [
            {
                'id': 'scenario_1',
                'text': '직장에서 동료의 실수를 상사에게 보고할지 말지 고민되는 상황',
                'action': '동료와 먼저 이야기하고 개선 기회를 주기로 결정',
                'context': {
                    'urgency': 'medium',
                    'impact': 'high',
                    'stakeholders': ['동료', '상사', '팀', '본인']
                },
                'expected_emotions': [EmotionState.TRUST, EmotionState.ANTICIPATION]
            },
            {
                'id': 'scenario_2', 
                'text': '친구가 부정행위를 하는 것을 목격했을 때의 딜레마',
                'action': '친구에게 직접 이야기하여 스스로 고백하도록 권유',
                'context': {
                    'urgency': 'high',
                    'impact': 'very_high', 
                    'stakeholders': ['친구', '교수', '다른 학생들', '본인']
                },
                'expected_emotions': [EmotionState.SADNESS, EmotionState.FEAR, EmotionState.TRUST]
            }
        ]
    
    async def run_complete_test(self) -> Dict[str, Any]:
        """완전 통합 테스트 실행"""
        logger.info("=== Linux Red Heart 완전 통합 테스트 시작 ===")
        
        try:
            # 1. GPU 가속 테스트
            await self.test_gpu_acceleration()
            
            # 2. 고급 시스템 초기화 테스트
            await self.test_system_initialization()
            
            # 3. 트랜스포머 통합 테스트
            await self.test_transformer_integration()
            
            # 4. 후회 분석 테스트
            await self.test_regret_analysis()
            
            # 5. 실시간 처리 테스트
            await self.test_real_time_processing()
            
            # 6. 종합 시스템 통합 테스트
            await self.test_complete_system_integration()
            
            # 7. 성능 벤치마크
            await self.test_performance_benchmarks()
            
            # 결과 보고서 생성
            report = self.generate_test_report()
            
            logger.info("=== 완전 통합 테스트 완료 ===")
            return report
            
        except Exception as e:
            logger.error(f"통합 테스트 실행 중 오류: {str(e)}")
            self.test_results['error_logs'].append(str(e))
            return self.test_results
    
    async def test_gpu_acceleration(self):
        """GPU 가속 기능 테스트"""
        logger.info("GPU 가속 테스트 시작...")
        
        try:
            import torch
            if torch.cuda.is_available():
                # GPU 메모리 테스트
                device = torch.device('cuda')
                test_tensor = torch.randn(1000, 1000).to(device)
                result = torch.matmul(test_tensor, test_tensor.T)
                
                self.test_results['gpu_acceleration'] = True
                logger.info("✅ GPU 가속 테스트 통과")
            else:
                logger.warning("⚠️ CUDA 사용 불가 - CPU 모드로 진행")
                self.test_results['gpu_acceleration'] = False
                
        except Exception as e:
            logger.error(f"❌ GPU 가속 테스트 실패: {str(e)}")
            self.test_results['error_logs'].append(f"GPU Test: {str(e)}")
    
    async def test_system_initialization(self):
        """고급 시스템 초기화 테스트"""
        logger.info("시스템 초기화 테스트 시작...")
        
        try:
            self.red_heart_system = AdvancedRedHeartSystem()
            await self.red_heart_system.initialize()
            
            # 초기화 검증
            assert hasattr(self.red_heart_system, 'emotion_analyzer')
            assert hasattr(self.red_heart_system, 'surd_analyzer')
            assert hasattr(self.red_heart_system, 'regret_analyzer')
            
            logger.info("✅ 시스템 초기화 테스트 통과")
            
        except Exception as e:
            logger.error(f"❌ 시스템 초기화 테스트 실패: {str(e)}")
            self.test_results['error_logs'].append(f"System Init: {str(e)}")
    
    async def test_transformer_integration(self):
        """트랜스포머 모델 통합 테스트"""
        logger.info("트랜스포머 통합 테스트 시작...")
        
        try:
            test_text = "이것은 트랜스포머 모델 테스트를 위한 한국어 텍스트입니다."
            
            # 시맨틱 분석 테스트
            semantic_result = await self.red_heart_system.analyze_semantic_meaning(test_text)
            
            # 결과 검증
            assert 'surface_features' in semantic_result
            assert 'semantic_embedding' in semantic_result
            assert len(semantic_result['semantic_embedding']) > 0
            
            self.test_results['transformer_integration'] = True
            logger.info("✅ 트랜스포머 통합 테스트 통과")
            
        except Exception as e:
            logger.error(f"❌ 트랜스포머 통합 테스트 실패: {str(e)}")
            self.test_results['error_logs'].append(f"Transformer: {str(e)}")
    
    async def test_regret_analysis(self):
        """고급 후회 분석 테스트"""
        logger.info("후회 분석 테스트 시작...")
        
        try:
            regret_analyzer = AdvancedRegretAnalyzer()
            
            # 테스트 의사결정 데이터
            decision_data = {
                'id': 'test_decision_001',
                'scenario': self.test_scenarios[0]['text'],
                'action': self.test_scenarios[0]['action'],
                'context': self.test_scenarios[0]['context']
            }
            
            # 후회 분석 실행
            regret_metrics = await regret_analyzer.analyze_regret(decision_data)
            
            # 결과 검증
            assert regret_metrics.decision_id == 'test_decision_001'
            assert 0 <= regret_metrics.anticipated_regret <= 1
            assert regret_metrics.computation_time_ms > 0
            assert len(regret_metrics.emotional_regret_vector) == 8
            
            self.test_results['regret_analysis'] = True
            logger.info("✅ 후회 분석 테스트 통과")
            
        except Exception as e:
            logger.error(f"❌ 후회 분석 테스트 실패: {str(e)}")
            self.test_results['error_logs'].append(f"Regret Analysis: {str(e)}")
    
    async def test_real_time_processing(self):
        """실시간 처리 성능 테스트"""
        logger.info("실시간 처리 테스트 시작...")
        
        try:
            processing_times = []
            
            for scenario in self.test_scenarios:
                start_time = time.time()
                
                # 실시간 의사결정 분석
                result = await self.red_heart_system.make_decision(
                    scenario=scenario['text'],
                    options=[scenario['action'], "다른 선택지"],
                    context=scenario['context']
                )
                
                processing_time = (time.time() - start_time) * 1000  # ms
                processing_times.append(processing_time)
                
                # 100ms 이하 목표 검증
                if processing_time <= 100:
                    logger.info(f"✅ 시나리오 처리 시간: {processing_time:.2f}ms")
                else:
                    logger.warning(f"⚠️ 처리 시간 초과: {processing_time:.2f}ms")
            
            avg_processing_time = sum(processing_times) / len(processing_times)
            self.test_results['performance_benchmarks']['avg_processing_time'] = avg_processing_time
            
            if avg_processing_time <= 100:
                self.test_results['real_time_processing'] = True
                logger.info(f"✅ 실시간 처리 테스트 통과 (평균: {avg_processing_time:.2f}ms)")
            else:
                logger.warning(f"⚠️ 실시간 처리 목표 미달 (평균: {avg_processing_time:.2f}ms)")
                
        except Exception as e:
            logger.error(f"❌ 실시간 처리 테스트 실패: {str(e)}")
            self.test_results['error_logs'].append(f"Real-time Processing: {str(e)}")
    
    async def test_complete_system_integration(self):
        """완전 시스템 통합 테스트"""
        logger.info("완전 시스템 통합 테스트 시작...")
        
        try:
            comprehensive_results = []
            
            for scenario in self.test_scenarios:
                # 종합적인 윤리적 의사결정 분석
                decision_result = await self.red_heart_system.comprehensive_ethical_analysis(
                    scenario=scenario['text'],
                    proposed_action=scenario['action'],
                    context=scenario['context']
                )
                
                # 결과 구조 검증
                required_fields = [
                    'decision_recommendation',
                    'confidence_score',
                    'ethical_analysis',
                    'emotion_analysis',
                    'regret_prediction',
                    'stakeholder_impact',
                    'reasoning'
                ]
                
                for field in required_fields:
                    assert field in decision_result, f"Missing field: {field}"
                
                comprehensive_results.append(decision_result)
                logger.info(f"✅ 시나리오 {scenario['id']} 종합 분석 완료")
            
            self.test_results['system_integration'] = True
            self.test_results['comprehensive_results'] = comprehensive_results
            logger.info("✅ 완전 시스템 통합 테스트 통과")
            
        except Exception as e:
            logger.error(f"❌ 완전 시스템 통합 테스트 실패: {str(e)}")
            self.test_results['error_logs'].append(f"System Integration: {str(e)}")
    
    async def test_performance_benchmarks(self):
        """성능 벤치마크 테스트"""
        logger.info("성능 벤치마크 테스트 시작...")
        
        try:
            import torch
            import psutil
            
            # 시스템 리소스 측정
            initial_memory = psutil.virtual_memory().percent
            if torch.cuda.is_available():
                initial_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            
            # 벤치마크 실행
            benchmark_start = time.time()
            
            # 동시 처리 테스트 (5개 시나리오 병렬 처리)
            tasks = []
            for i in range(5):
                scenario = self.test_scenarios[i % len(self.test_scenarios)]
                task = self.red_heart_system.make_decision(
                    scenario=scenario['text'],
                    options=[scenario['action']],
                    context=scenario['context']
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            benchmark_time = time.time() - benchmark_start
            
            # 리소스 사용량 측정
            final_memory = psutil.virtual_memory().percent
            memory_usage = final_memory - initial_memory
            
            if torch.cuda.is_available():
                final_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
                gpu_memory_usage = final_gpu_memory - initial_gpu_memory
            else:
                gpu_memory_usage = 0
            
            # 벤치마크 결과 저장
            self.test_results['performance_benchmarks'].update({
                'concurrent_processing_time': benchmark_time,
                'memory_usage_percent': memory_usage,
                'gpu_memory_usage_mb': gpu_memory_usage,
                'throughput_decisions_per_second': len(results) / benchmark_time,
                'successful_decisions': len([r for r in results if r is not None])
            })
            
            logger.info(f"✅ 성능 벤치마크 완료:")
            logger.info(f"   - 병렬 처리 시간: {benchmark_time:.2f}초")
            logger.info(f"   - 처리량: {len(results) / benchmark_time:.2f} 결정/초")
            logger.info(f"   - 메모리 사용량: {memory_usage:.2f}%")
            logger.info(f"   - GPU 메모리 사용량: {gpu_memory_usage:.2f}MB")
            
        except Exception as e:
            logger.error(f"❌ 성능 벤치마크 테스트 실패: {str(e)}")
            self.test_results['error_logs'].append(f"Performance Benchmark: {str(e)}")
    
    def generate_test_report(self) -> Dict[str, Any]:
        """종합 테스트 보고서 생성"""
        
        passed_tests = sum([
            self.test_results['gpu_acceleration'],
            self.test_results['transformer_integration'], 
            self.test_results['regret_analysis'],
            self.test_results['real_time_processing'],
            self.test_results['system_integration']
        ])
        
        total_tests = 5
        success_rate = (passed_tests / total_tests) * 100
        
        report = {
            'test_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': total_tests - passed_tests,
                'success_rate': f"{success_rate:.1f}%",
                'overall_status': 'PASS' if success_rate >= 80 else 'FAIL'
            },
            'test_details': self.test_results,
            'recommendations': self._generate_recommendations(),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 보고서 파일로 저장
        report_path = Path('logs') / f'integration_test_report_{int(time.time())}.json'
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"📊 테스트 보고서 저장: {report_path}")
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """개선 권장사항 생성"""
        recommendations = []
        
        if not self.test_results['gpu_acceleration']:
            recommendations.append("GPU 가속 활성화를 위해 CUDA 드라이버 설치 확인")
        
        if not self.test_results['real_time_processing']:
            recommendations.append("실시간 처리 성능 개선을 위한 모델 최적화 필요")
        
        avg_time = self.test_results['performance_benchmarks'].get('avg_processing_time', 0)
        if avg_time > 100:
            recommendations.append(f"처리 시간 최적화 필요 (현재: {avg_time:.2f}ms)")
        
        if len(self.test_results['error_logs']) > 0:
            recommendations.append("오류 로그 검토 및 안정성 개선 필요")
        
        if not recommendations:
            recommendations.append("모든 테스트 통과 - 시스템이 정상적으로 작동합니다")
        
        return recommendations

async def main():
    """메인 테스트 실행 함수"""
    tester = CompleteIntegrationTest()
    
    try:
        report = await tester.run_complete_test()
        
        print("\n" + "="*60)
        print("🧪 LINUX RED HEART 통합 테스트 결과")
        print("="*60)
        print(f"📈 성공률: {report['test_summary']['success_rate']}")
        print(f"✅ 통과: {report['test_summary']['passed_tests']}")
        print(f"❌ 실패: {report['test_summary']['failed_tests']}")
        print(f"🎯 전체 상태: {report['test_summary']['overall_status']}")
        print("\n📋 권장사항:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"   {i}. {rec}")
        print("="*60)
        
        return report['test_summary']['overall_status'] == 'PASS'
        
    except Exception as e:
        logger.error(f"테스트 실행 중 치명적 오류: {str(e)}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)