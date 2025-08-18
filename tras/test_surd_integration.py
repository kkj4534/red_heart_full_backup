"""
SURD 분석 시스템 완전 통합 테스트
Test Complete Integration of SURD Analysis System

이 스크립트는 SURD 분석 시스템과 다른 모든 주요 모듈들 간의 통합을 테스트합니다.
- 감정 분석 모듈 연동
- 벤담 계산 모듈 연동  
- LLM 모듈 연동
- 전체 시스템 통합 분석
"""

import asyncio
import logging
import time
import numpy as np
import json
from typing import Dict, Any, List
from pathlib import Path
import traceback

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SURD_Integration_Test')

# 시스템 모듈 임포트
try:
    from advanced_surd_analyzer import AdvancedSURDAnalyzer
    from advanced_emotion_analyzer import AdvancedEmotionAnalyzer
    from advanced_bentham_calculator import AdvancedBenthamCalculator
    from advanced_llm_integration_layer import AdvancedLLMIntegration
    from data_models import EthicalSituation, EmotionData, HedonicValues
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    logger.error(f"모듈 임포트 실패: {e}")

class SURDIntegrationTester:
    """SURD 시스템 통합 테스터"""
    
    def __init__(self):
        self.surd_analyzer = None
        self.emotion_analyzer = None
        self.bentham_calculator = None
        self.llm_integration = None
        
        self.test_results = {
            'emotion_integration': False,
            'bentham_integration': False, 
            'llm_integration': False,
            'full_system_integration': False,
            'performance_metrics': {},
            'error_log': []
        }
        
    async def initialize_components(self):
        """모든 컴포넌트 초기화"""
        logger.info("=== SURD 통합 시스템 초기화 ===")
        
        try:
            # SURD 분석기 초기화
            logger.info("SURD 분석기 초기화 중...")
            self.surd_analyzer = AdvancedSURDAnalyzer()
            logger.info("✅ SURD 분석기 초기화 완료")
            
            # 감정 분석기 초기화
            logger.info("감정 분석기 초기화 중...")
            try:
                self.emotion_analyzer = AdvancedEmotionAnalyzer()
                logger.info("✅ 감정 분석기 초기화 완료")
            except Exception as e:
                logger.warning(f"⚠️ 감정 분석기 초기화 실패: {e}")
                self.emotion_analyzer = None
            
            # 벤담 계산기 초기화
            logger.info("벤담 계산기 초기화 중...")
            try:
                self.bentham_calculator = AdvancedBenthamCalculator()
                logger.info("✅ 벤담 계산기 초기화 완료")
            except Exception as e:
                logger.warning(f"⚠️ 벤담 계산기 초기화 실패: {e}")
                self.bentham_calculator = None
            
            # LLM 통합 레이어 초기화
            logger.info("LLM 통합 레이어 초기화 중...")
            try:
                self.llm_integration = AdvancedLLMIntegration()
                logger.info("✅ LLM 통합 레이어 초기화 완료")
            except Exception as e:
                logger.warning(f"⚠️ LLM 통합 레이어 초기화 실패: {e}")
                self.llm_integration = None
                
        except Exception as e:
            logger.error(f"❌ 컴포넌트 초기화 실패: {e}")
            self.test_results['error_log'].append(f"초기화 실패: {e}")
    
    def create_test_scenario(self) -> Dict[str, Any]:
        """테스트 시나리오 생성"""
        return {
            'scenario_description': "윤리적 딜레마: 자율주행차 사고 상황",
            'context': {
                'situation_type': 'autonomous_vehicle_dilemma',
                'urgency_level': 0.9,
                'stakeholder_count': 5,
                'moral_complexity': 0.8,
                'time_pressure': 0.95,
                'social_impact': 0.7
            },
            'stakeholders': {
                'passenger': 0.8,
                'pedestrian': 0.9,
                'driver_family': 0.7,
                'society': 0.6,
                'legal_system': 0.5
            },
            'emotion_context': {
                'fear': 0.9,
                'anxiety': 0.8,
                'responsibility': 0.7,
                'uncertainty': 0.85,
                'empathy': 0.6
            },
            'bentham_context': {
                'intensity': 0.9,
                'duration': 0.7,
                'certainty': 0.4,
                'propinquity': 0.8,
                'fecundity': 0.3,
                'purity': 0.6,
                'extent': 0.8
            }
        }
    
    async def test_emotion_integration(self) -> bool:
        """감정 분석 모듈과의 통합 테스트"""
        logger.info("\n=== 감정 분석 모듈 통합 테스트 ===")
        
        if not self.emotion_analyzer:
            logger.warning("감정 분석기가 초기화되지 않음")
            return False
            
        try:
            # 테스트 시나리오 생성
            scenario = self.create_test_scenario()
            
            # 모의 감정 분석 데이터 생성
            emotion_data = {
                'emotion_intensities': scenario['emotion_context'],
                'emotion_states': {
                    'arousal': 0.8,
                    'valence': 0.3,
                    'dominance': 0.4
                },
                'biosignals': {
                    'heart_rate': 0.85,
                    'skin_conductance': 0.7,
                    'facial_expression': 0.6
                }
            }
            
            # SURD 분석기와 감정 데이터 통합
            start_time = time.time()
            surd_variables = self.surd_analyzer.integrate_with_emotion_analysis(emotion_data)
            integration_time = time.time() - start_time
            
            # 결과 검증
            if surd_variables and len(surd_variables) > 0:
                logger.info(f"✅ 감정 분석 통합 성공")
                logger.info(f"   - 변환된 변수 수: {len(surd_variables)}")
                logger.info(f"   - 처리 시간: {integration_time:.3f}초")
                logger.info(f"   - 변수 목록: {list(surd_variables.keys())[:5]}...")
                
                # 데이터 품질 검증
                for var_name, var_data in surd_variables.items():
                    if not isinstance(var_data, np.ndarray) or len(var_data) == 0:
                        logger.warning(f"⚠️ 변수 {var_name}의 데이터 품질 문제")
                        return False
                
                self.test_results['performance_metrics']['emotion_integration_time'] = integration_time
                return True
            else:
                logger.error("❌ 감정 분석 통합 실패: 변수 변환 없음")
                return False
                
        except Exception as e:
            logger.error(f"❌ 감정 분석 통합 테스트 실패: {e}")
            self.test_results['error_log'].append(f"감정 통합 실패: {e}")
            return False
    
    async def test_bentham_integration(self) -> bool:
        """벤담 계산 모듈과의 통합 테스트"""
        logger.info("\n=== 벤담 계산 모듈 통합 테스트 ===")
        
        if not self.bentham_calculator:
            logger.warning("벤담 계산기가 초기화되지 않음")
            return False
            
        try:
            # 테스트 시나리오 생성
            scenario = self.create_test_scenario()
            
            # 모의 벤담 계산 데이터 생성
            bentham_data = {
                'bentham_variables': scenario['bentham_context'],
                'weight_layers': {
                    'cultural_weight': 0.8,
                    'temporal_weight': 0.6,
                    'social_weight': 0.9,
                    'personal_weight': 0.7,
                    'moral_weight': 0.85,
                    'situational_weight': 0.75
                },
                'pleasure_score': 0.65,
                'neural_predictions': {
                    'predicted_satisfaction': 0.7,
                    'predicted_regret': 0.3,
                    'predicted_social_impact': 0.8
                }
            }
            
            # SURD 분석기와 벤담 데이터 통합
            start_time = time.time()
            surd_variables = self.surd_analyzer.integrate_with_bentham_calculation(bentham_data)
            integration_time = time.time() - start_time
            
            # 결과 검증
            if surd_variables and len(surd_variables) > 0:
                logger.info(f"✅ 벤담 계산 통합 성공")
                logger.info(f"   - 변환된 변수 수: {len(surd_variables)}")
                logger.info(f"   - 처리 시간: {integration_time:.3f}초")
                logger.info(f"   - 변수 목록: {list(surd_variables.keys())[:5]}...")
                
                # 벤담 변수 특화 검증
                bentham_vars = [k for k in surd_variables.keys() if k.startswith('bentham_')]
                weight_vars = [k for k in surd_variables.keys() if k.startswith('weight_')]
                
                logger.info(f"   - 벤담 변수: {len(bentham_vars)}개")
                logger.info(f"   - 가중치 변수: {len(weight_vars)}개")
                
                self.test_results['performance_metrics']['bentham_integration_time'] = integration_time
                return True
            else:
                logger.error("❌ 벤담 계산 통합 실패: 변수 변환 없음")
                return False
                
        except Exception as e:
            logger.error(f"❌ 벤담 계산 통합 테스트 실패: {e}")
            self.test_results['error_log'].append(f"벤담 통합 실패: {e}")
            return False
    
    async def test_llm_integration(self) -> bool:
        """LLM 모듈과의 통합 테스트"""
        logger.info("\n=== LLM 모듈 통합 테스트 ===")
        
        try:
            # 모의 LLM 분석 데이터 생성
            llm_data = {
                'analysis_scores': {
                    'ethical_reasoning': 0.75,
                    'contextual_understanding': 0.8,
                    'stakeholder_analysis': 0.7,
                    'consequence_prediction': 0.65,
                    'moral_reasoning': 0.82
                },
                'semantic_embeddings': np.random.normal(0, 1, 768),  # 768차원 임베딩
                'generation_quality': 0.78,
                'context_understanding': 0.85
            }
            
            # SURD 분석기와 LLM 데이터 통합
            start_time = time.time()
            surd_variables = self.surd_analyzer.integrate_with_llm_results(llm_data)
            integration_time = time.time() - start_time
            
            # 결과 검증
            if surd_variables and len(surd_variables) > 0:
                logger.info(f"✅ LLM 통합 성공")
                logger.info(f"   - 변환된 변수 수: {len(surd_variables)}")
                logger.info(f"   - 처리 시간: {integration_time:.3f}초")
                logger.info(f"   - 변수 목록: {list(surd_variables.keys())}")
                
                # LLM 변수 특화 검증
                llm_vars = [k for k in surd_variables.keys() if k.startswith('llm_')]
                logger.info(f"   - LLM 변수: {len(llm_vars)}개")
                
                self.test_results['performance_metrics']['llm_integration_time'] = integration_time
                return True
            else:
                logger.error("❌ LLM 통합 실패: 변수 변환 없음")
                return False
                
        except Exception as e:
            logger.error(f"❌ LLM 통합 테스트 실패: {e}")
            self.test_results['error_log'].append(f"LLM 통합 실패: {e}")
            return False
    
    async def test_full_system_integration(self) -> bool:
        """전체 시스템 통합 테스트"""
        logger.info("\n=== 전체 시스템 통합 테스트 ===")
        
        try:
            # 테스트 시나리오 생성
            scenario = self.create_test_scenario()
            
            # 모든 모듈의 데이터 준비
            emotion_data = {
                'emotion_intensities': scenario['emotion_context'],
                'emotion_states': {
                    'arousal': 0.8,
                    'valence': 0.3,
                    'dominance': 0.4
                },
                'biosignals': {
                    'heart_rate': 0.85,
                    'skin_conductance': 0.7,
                    'facial_expression': 0.6
                }
            }
            
            bentham_data = {
                'bentham_variables': scenario['bentham_context'],
                'weight_layers': {
                    'cultural_weight': 0.8,
                    'temporal_weight': 0.6,
                    'social_weight': 0.9,
                    'personal_weight': 0.7,
                    'moral_weight': 0.85,
                    'situational_weight': 0.75
                },
                'pleasure_score': 0.65,
                'neural_predictions': {
                    'predicted_satisfaction': 0.7,
                    'predicted_regret': 0.3,
                    'predicted_social_impact': 0.8
                }
            }
            
            llm_data = {
                'analysis_scores': {
                    'ethical_reasoning': 0.75,
                    'contextual_understanding': 0.8,
                    'stakeholder_analysis': 0.7,
                    'consequence_prediction': 0.65,
                    'moral_reasoning': 0.82
                },
                'semantic_embeddings': np.random.normal(0, 1, 768),
                'generation_quality': 0.78,
                'context_understanding': 0.85
            }
            
            # 통합 SURD 분석 수행
            logger.info("통합 SURD 분석 시작...")
            start_time = time.time()
            
            result = await self.surd_analyzer.analyze_integrated_system(
                emotion_data=emotion_data if self.emotion_analyzer else None,
                bentham_data=bentham_data if self.bentham_calculator else None,
                llm_data=llm_data,
                target_variable='ethical_decision_quality',
                additional_context=scenario['context']
            )
            
            analysis_time = time.time() - start_time
            
            # 결과 검증
            if result and hasattr(result, 'information_decomposition'):
                logger.info(f"✅ 전체 시스템 통합 분석 성공")
                logger.info(f"   - 분석 시간: {analysis_time:.3f}초")
                logger.info(f"   - 대상 변수: {result.target_variable}")
                logger.info(f"   - 입력 변수 수: {len(result.input_variables)}")
                
                # 정보 분해 결과 요약
                if result.information_decomposition:
                    for decomp_name, decomp in result.information_decomposition.items():
                        if hasattr(decomp, 'total_information'):
                            logger.info(f"   - {decomp_name}: 총 정보량 {decomp.total_information:.4f}")
                
                # 신경망 예측 결과
                if result.neural_predictions:
                    logger.info(f"   - 신경망 예측 항목: {len(result.neural_predictions)}")
                
                # 인과관계 네트워크
                if result.causal_network:
                    metrics = result.causal_network.metrics
                    logger.info(f"   - 네트워크 노드: {metrics.get('node_count', 0)}")
                    logger.info(f"   - 네트워크 엣지: {metrics.get('edge_count', 0)}")
                
                # 시간적 분석
                if result.temporal_analysis:
                    logger.info(f"   - 시간적 분석 항목: {len(result.temporal_analysis)}")
                
                # 통계적 유의성
                if result.significance_results:
                    significant_vars = sum(1 for stats in result.significance_results.values() 
                                         if stats.get('is_significant', False))
                    logger.info(f"   - 유의한 변수 수: {significant_vars}/{len(result.significance_results)}")
                
                # 성능 메트릭 저장
                self.test_results['performance_metrics'].update({
                    'full_integration_time': analysis_time,
                    'total_variables': len(result.input_variables),
                    'processing_time': result.processing_time,
                    'confidence_level': result.metadata.get('confidence_level', 0.95),
                    'integration_method': result.metadata.get('integration_method', 'unknown')
                })
                
                return True
            else:
                logger.error("❌ 전체 시스템 통합 실패: 결과 없음")
                return False
                
        except Exception as e:
            logger.error(f"❌ 전체 시스템 통합 테스트 실패: {e}")
            self.test_results['error_log'].append(f"전체 통합 실패: {e}")
            traceback.print_exc()
            return False
    
    def generate_test_report(self) -> Dict[str, Any]:
        """테스트 결과 보고서 생성"""
        
        # 전체 성공률 계산
        passed_tests = sum([
            self.test_results['emotion_integration'],
            self.test_results['bentham_integration'], 
            self.test_results['llm_integration'],
            self.test_results['full_system_integration']
        ])
        total_tests = 4
        success_rate = (passed_tests / total_tests) * 100
        
        report = {
            'test_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'success_rate': success_rate,
                'overall_status': '통과' if success_rate >= 75 else '실패'
            },
            'individual_tests': self.test_results,
            'performance_summary': self.test_results['performance_metrics'],
            'error_summary': self.test_results['error_log'],
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """개선 권장사항 생성"""
        recommendations = []
        
        # 실패한 테스트에 대한 권장사항
        if not self.test_results['emotion_integration']:
            recommendations.append("감정 분석 모듈 연동 개선 필요")
        
        if not self.test_results['bentham_integration']:
            recommendations.append("벤담 계산 모듈 연동 개선 필요")
        
        if not self.test_results['llm_integration']:
            recommendations.append("LLM 모듈 연동 개선 필요")
        
        if not self.test_results['full_system_integration']:
            recommendations.append("전체 시스템 통합 로직 재검토 필요")
        
        # 성능 기반 권장사항
        metrics = self.test_results['performance_metrics']
        
        if metrics.get('full_integration_time', 0) > 5.0:
            recommendations.append("통합 분석 성능 최적화 필요")
        
        if len(self.test_results['error_log']) > 0:
            recommendations.append("오류 로그 검토 및 예외 처리 강화 필요")
        
        if not recommendations:
            recommendations.append("모든 통합 테스트 통과 - 시스템이 정상 작동 중")
        
        return recommendations

async def run_integration_tests():
    """통합 테스트 실행 메인 함수"""
    
    if not MODULES_AVAILABLE:
        logger.error("❌ 필수 모듈을 불러올 수 없습니다. 테스트를 중단합니다.")
        return
    
    logger.info("🚀 SURD 시스템 통합 테스트 시작")
    
    # 테스터 초기화
    tester = SURDIntegrationTester()
    
    try:
        # 1. 컴포넌트 초기화
        await tester.initialize_components()
        
        # 2. 개별 모듈 통합 테스트
        tester.test_results['emotion_integration'] = await tester.test_emotion_integration()
        tester.test_results['bentham_integration'] = await tester.test_bentham_integration()
        tester.test_results['llm_integration'] = await tester.test_llm_integration()
        
        # 3. 전체 시스템 통합 테스트
        tester.test_results['full_system_integration'] = await tester.test_full_system_integration()
        
        # 4. 테스트 결과 보고서 생성
        report = tester.generate_test_report()
        
        # 5. 결과 출력
        logger.info("\n" + "="*60)
        logger.info("📊 SURD 시스템 통합 테스트 결과")
        logger.info("="*60)
        
        summary = report['test_summary']
        logger.info(f"전체 테스트: {summary['total_tests']}")
        logger.info(f"통과 테스트: {summary['passed_tests']}")
        logger.info(f"성공률: {summary['success_rate']:.1f}%")
        logger.info(f"최종 상태: {summary['overall_status']}")
        
        logger.info("\n📋 개별 테스트 결과:")
        logger.info(f"  - 감정 분석 통합: {'✅ 통과' if report['individual_tests']['emotion_integration'] else '❌ 실패'}")
        logger.info(f"  - 벤담 계산 통합: {'✅ 통과' if report['individual_tests']['bentham_integration'] else '❌ 실패'}")
        logger.info(f"  - LLM 모듈 통합: {'✅ 통과' if report['individual_tests']['llm_integration'] else '❌ 실패'}")
        logger.info(f"  - 전체 시스템 통합: {'✅ 통과' if report['individual_tests']['full_system_integration'] else '❌ 실패'}")
        
        if report['performance_summary']:
            logger.info("\n⚡ 성능 메트릭:")
            for metric, value in report['performance_summary'].items():
                if isinstance(value, float):
                    logger.info(f"  - {metric}: {value:.3f}")
                else:
                    logger.info(f"  - {metric}: {value}")
        
        if report['error_summary']:
            logger.info("\n⚠️ 오류 로그:")
            for error in report['error_summary']:
                logger.info(f"  - {error}")
        
        logger.info("\n💡 권장사항:")
        for recommendation in report['recommendations']:
            logger.info(f"  - {recommendation}")
        
        # 결과를 파일로 저장
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        report_file = Path(f'surd_integration_test_report_{timestamp}.json')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n📄 상세 보고서 저장: {report_file}")
        
        # 최종 상태 출력
        if summary['success_rate'] >= 75:
            logger.info("\n🎉 SURD 시스템 통합 테스트 성공!")
            logger.info("   시스템이 다른 모듈들과 정상적으로 연동되고 있습니다.")
        else:
            logger.warning("\n⚠️ SURD 시스템 통합에 문제가 있습니다.")
            logger.warning("   위의 권장사항을 검토하여 개선하시기 바랍니다.")
            
    except Exception as e:
        logger.error(f"❌ 통합 테스트 중 예상치 못한 오류: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # 통합 테스트 실행
    asyncio.run(run_integration_tests())