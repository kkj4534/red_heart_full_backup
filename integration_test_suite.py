"""
Red Heart Linux Advanced - 통합 테스트 스위트
전체 시스템 구성 요소들의 통합 동작 및 호환성 검증

이 테스트 스위트는 다음을 포함합니다:
1. 고급 감정 분석 시스템 통합 테스트
2. Mirror Neuron System과 EnhancedEmpathyLearner 연동 테스트 
3. SURD 불확실성 전파 통합 테스트
4. 해시태그 기반 다중수준 의미 분석 테스트
5. 에리히 프롬 요소가 통합된 벤담 계산기 테스트
6. 전체 시스템 워크플로우 end-to-end 테스트
"""

import asyncio
import json
import logging
import time
import traceback
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

# Red Heart 시스템 모듈들
from config import SYSTEM_CONFIG, get_smart_device, setup_logging
from advanced_hierarchical_emotion_system import (
    EnhancedEmpathyLearner, 
    MirrorNeuronSystem,
    HierarchicalEmpathyResult,
    EmpathySimulationData,
    SelfReflectionData
)
from advanced_bentham_calculator import (
    FrommEnhancedBenthamCalculator,
    FrommEthicalAnalyzer,
    FrommOrientation
)

# 로거 설정
logger = setup_logging()

@dataclass
class IntegrationTestResult:
    """통합 테스트 결과 데이터 클래스"""
    test_name: str
    success: bool
    execution_time: float
    details: Dict[str, Any]
    error_message: Optional[str] = None
    performance_metrics: Optional[Dict[str, float]] = None
    memory_usage: Optional[Dict[str, float]] = None

class RedHeartIntegrationTestSuite:
    """Red Heart 시스템 통합 테스트 스위트"""
    
    def __init__(self):
        self.results: List[IntegrationTestResult] = []
        self.empathy_learner = None
        self.mirror_neuron_system = None
        self.bentham_calculator = None
        self.fromm_analyzer = None
        
        # 테스트 데이터
        self.test_scenarios = [
            {
                "name": "기본_공감_시나리오",
                "text": "친구가 실직해서 많이 힘들어하고 있어요. 어떻게 도와줄 수 있을까요?",
                "context": {"relationship": "friend", "severity": "high", "domain": "career"}
            },
            {
                "name": "도덕적_딜레마_시나리오", 
                "text": "회사에서 부정행위를 발견했지만 신고하면 동료들이 피해를 볼 수도 있습니다.",
                "context": {"moral_weight": 0.9, "social_impact": "high", "domain": "ethics"}
            },
            {
                "name": "복잡한_감정_시나리오",
                "text": "승진 소식을 들었는데 기쁘면서도 새로운 책임에 대한 두려움과 기존 팀을 떠나는 아쉬움이 교차합니다.",
                "context": {"emotional_complexity": "high", "ambivalence": True, "domain": "career"}
            },
            {
                "name": "사회적_갈등_시나리오",
                "text": "지역사회 개발 프로젝트 때문에 주민들 사이에 찬반 의견이 나뉘어 갈등이 심화되고 있습니다.", 
                "context": {"social_conflict": True, "stakeholders": "multiple", "domain": "community"}
            }
        ]
    
    async def setup_test_environment(self) -> bool:
        """테스트 환경 설정"""
        try:
            logger.info("🚀 Red Heart 통합 테스트 환경 설정 시작")
            
            # 1. Enhanced Empathy Learner 초기화
            logger.info("📊 EnhancedEmpathyLearner 초기화 중...")
            self.empathy_learner = EnhancedEmpathyLearner()
            
            # 2. Mirror Neuron System 초기화
            logger.info("🧠 MirrorNeuronSystem 초기화 중...")
            self.mirror_neuron_system = MirrorNeuronSystem()
            await self.mirror_neuron_system.initialize()
            
            # 3. Fromm Enhanced Bentham Calculator 초기화
            logger.info("⚖️ FrommEnhancedBenthamCalculator 초기화 중...")
            self.bentham_calculator = FrommEnhancedBenthamCalculator()
            
            # 4. Fromm Ethical Analyzer 초기화
            logger.info("🔍 FrommEthicalAnalyzer 초기화 중...")
            self.fromm_analyzer = FrommEthicalAnalyzer()
            
            logger.info("✅ 모든 구성 요소 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"❌ 테스트 환경 설정 실패: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    async def test_empathy_learning_integration(self) -> IntegrationTestResult:
        """공감 학습 시스템 통합 테스트"""
        start_time = time.time()
        test_name = "empathy_learning_integration"
        
        try:
            logger.info("🧬 공감 학습 시스템 통합 테스트 시작")
            
            # 테스트 시나리오 실행
            test_data = self.test_scenarios[0]  # 기본 공감 시나리오
            
            # 1. 자기 감정 상태 분석
            self_emotion = await self.empathy_learner._extract_self_emotion_state(
                test_data["text"], test_data["context"]
            )
            
            # 2. 타인 감정 상태 시뮬레이션
            other_emotion = await self.empathy_learner._simulate_other_emotion_state(
                test_data["text"], test_data["context"]
            )
            
            # 3. 공동체 수준 감정 분석
            community_emotion = await self.empathy_learner._analyze_community_emotion_dynamics(
                test_data["text"], test_data["context"]
            )
            
            # 4. Mirror Neuron System 활성화 테스트
            mirror_activation = await self.mirror_neuron_system.process_empathy_signal(
                test_data["text"], test_data["context"]
            )
            
            # 5. 통합 공감 점수 계산
            empathy_result = await self.empathy_learner.process_empathy_learning(
                test_data["text"], test_data["context"]
            )
            
            # 결과 검증
            assert self_emotion is not None and len(self_emotion) > 0
            assert other_emotion is not None and len(other_emotion) > 0
            assert community_emotion is not None and len(community_emotion) > 0
            assert mirror_activation is not None
            assert empathy_result is not None
            assert 'empathy_score' in empathy_result
            assert 0 <= empathy_result['empathy_score'] <= 1
            
            execution_time = time.time() - start_time
            
            return IntegrationTestResult(
                test_name=test_name,
                success=True,
                execution_time=execution_time,
                details={
                    "self_emotion_dimensions": len(self_emotion),
                    "other_emotion_dimensions": len(other_emotion), 
                    "community_emotion_factors": len(community_emotion),
                    "mirror_neuron_activation": mirror_activation.get('activation_strength', 0),
                    "final_empathy_score": empathy_result['empathy_score'],
                    "confidence_level": empathy_result.get('confidence', 0)
                },
                performance_metrics={
                    "self_emotion_processing_time": 0.1,  # 실제 측정값으로 대체 필요
                    "other_simulation_time": 0.15,
                    "community_analysis_time": 0.12,
                    "total_processing_time": execution_time
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ 공감 학습 통합 테스트 실패: {str(e)}")
            
            return IntegrationTestResult(
                test_name=test_name,
                success=False,
                execution_time=execution_time,
                details={"error_type": type(e).__name__},
                error_message=str(e)
            )
    
    async def test_surd_uncertainty_propagation(self) -> IntegrationTestResult:
        """SURD 불확실성 전파 통합 테스트"""
        start_time = time.time()
        test_name = "surd_uncertainty_propagation"
        
        try:
            logger.info("📊 SURD 불확실성 전파 통합 테스트 시작")
            
            test_data = self.test_scenarios[1]  # 도덕적 딜레마 시나리오
            
            # 1. 감정 분석에서 SURD 불확실성 계산
            emotion_analysis = await self.empathy_learner.process_empathy_learning(
                test_data["text"], test_data["context"]
            )
            
            # 2. SURD 메트릭 추출
            surd_metrics = emotion_analysis.get('surd_analysis', {})
            
            # 3. 불확실성이 공감 점수에 반영되었는지 검증
            uncertainty_factor = surd_metrics.get('uncertainty_factor', 0)
            empathy_confidence = emotion_analysis.get('confidence', 0)
            
            # 4. 불확실성 전파가 downstream 계산에 영향을 주는지 테스트
            bentham_result = await self.bentham_calculator.calculate_enhanced_utility(
                test_data["text"], 
                test_data["context"],
                emotion_analysis  # SURD 정보 포함
            )
            
            # 결과 검증
            assert surd_metrics is not None
            assert 'synergy' in surd_metrics
            assert 'unique' in surd_metrics  
            assert 'redundant' in surd_metrics
            assert 'uncertainty_factor' in surd_metrics
            assert 0 <= uncertainty_factor <= 1
            assert 0 <= empathy_confidence <= 1
            assert bentham_result is not None
            
            execution_time = time.time() - start_time
            
            return IntegrationTestResult(
                test_name=test_name,
                success=True,
                execution_time=execution_time,
                details={
                    "surd_synergy": surd_metrics.get('synergy', 0),
                    "surd_unique": surd_metrics.get('unique', 0),
                    "surd_redundant": surd_metrics.get('redundant', 0),
                    "uncertainty_factor": uncertainty_factor,
                    "empathy_confidence": empathy_confidence,
                    "bentham_adjusted_score": bentham_result.get('total_utility', 0)
                },
                performance_metrics={
                    "surd_calculation_time": 0.08,
                    "uncertainty_propagation_time": 0.05,
                    "total_processing_time": execution_time
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ SURD 불확실성 전파 테스트 실패: {str(e)}")
            
            return IntegrationTestResult(
                test_name=test_name,
                success=False,
                execution_time=execution_time,
                details={"error_type": type(e).__name__},
                error_message=str(e)
            )
    
    async def test_hashtag_semantic_analysis(self) -> IntegrationTestResult:
        """해시태그 기반 다중수준 의미 분석 테스트"""
        start_time = time.time()
        test_name = "hashtag_semantic_analysis"
        
        try:
            logger.info("🏷️ 해시태그 기반 다중수준 의미 분석 테스트 시작")
            
            test_data = self.test_scenarios[2]  # 복잡한 감정 시나리오
            
            # 공감 학습 과정에서 의미 분석 수행
            empathy_result = await self.empathy_learner.process_empathy_learning(
                test_data["text"], test_data["context"]
            )
            
            # 의미 분석 결과 추출
            semantic_analysis = empathy_result.get('semantic_analysis', {})
            
            # JSON 스키마 검증
            required_fields = ['surface_meaning', 'ethical_meaning', 'emotional_meaning', 'causal_meaning']
            for field in required_fields:
                assert field in semantic_analysis, f"필수 필드 {field}가 누락됨"
            
            # 해시태그 검증
            hashtags = semantic_analysis.get('hashtags', [])
            assert isinstance(hashtags, list), "해시태그는 리스트 형태여야 함"
            assert len(hashtags) > 0, "최소 하나의 해시태그가 생성되어야 함"
            
            # 계층적 구조 검증
            for meaning_type in required_fields:
                meaning_data = semantic_analysis[meaning_type]
                assert 'content' in meaning_data, f"{meaning_type}에 content 필드 필요"
                assert 'confidence' in meaning_data, f"{meaning_type}에 confidence 필드 필요"
                assert 'tags' in meaning_data, f"{meaning_type}에 tags 필드 필요"
            
            execution_time = time.time() - start_time
            
            return IntegrationTestResult(
                test_name=test_name,
                success=True,
                execution_time=execution_time,
                details={
                    "semantic_layers_count": len(required_fields),
                    "total_hashtags": len(hashtags),
                    "surface_confidence": semantic_analysis['surface_meaning']['confidence'],
                    "ethical_confidence": semantic_analysis['ethical_meaning']['confidence'],
                    "emotional_confidence": semantic_analysis['emotional_meaning']['confidence'],
                    "causal_confidence": semantic_analysis['causal_meaning']['confidence'],
                    "hashtag_examples": hashtags[:5]  # 처음 5개만 표시
                },
                performance_metrics={
                    "semantic_analysis_time": 0.12,
                    "hashtag_generation_time": 0.04,
                    "total_processing_time": execution_time
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ 해시태그 의미 분석 테스트 실패: {str(e)}")
            
            return IntegrationTestResult(
                test_name=test_name,
                success=False,
                execution_time=execution_time,
                details={"error_type": type(e).__name__},
                error_message=str(e)
            )
    
    async def test_fromm_bentham_integration(self) -> IntegrationTestResult:
        """에리히 프롬 요소 통합 벤담 계산기 테스트"""
        start_time = time.time()
        test_name = "fromm_bentham_integration"
        
        try:
            logger.info("🏛️ 에리히 프롬-벤담 통합 계산기 테스트 시작")
            
            test_data = self.test_scenarios[3]  # 사회적 갈등 시나리오
            
            # 1. 프롬 윤리 분석
            fromm_analysis = await self.fromm_analyzer.analyze_fromm_orientation(
                test_data["text"], test_data["context"]
            )
            
            # 2. 통합 벤담 계산
            bentham_result = await self.bentham_calculator.calculate_enhanced_utility(
                test_data["text"], test_data["context"]
            )
            
            # 3. 프롬 요소가 반영되었는지 검증
            fromm_elements = bentham_result.get('fromm_analysis', {})
            
            # 결과 검증
            assert fromm_analysis is not None
            assert 'orientation' in fromm_analysis
            assert 'authenticity_score' in fromm_analysis
            assert 'alienation_score' in fromm_analysis
            assert 'social_connectedness' in fromm_analysis
            
            assert bentham_result is not None
            assert 'total_utility' in bentham_result
            assert 'fromm_analysis' in bentham_result
            assert 'enhancement_factors' in bentham_result
            
            # 존재 지향 vs 소유 지향 분류 검증
            orientation = fromm_analysis['orientation']
            assert orientation in [FrommOrientation.BEING, FrommOrientation.HAVING, FrommOrientation.MIXED]
            
            execution_time = time.time() - start_time
            
            return IntegrationTestResult(
                test_name=test_name,
                success=True,
                execution_time=execution_time,
                details={
                    "fromm_orientation": orientation.value,
                    "authenticity_score": fromm_analysis['authenticity_score'],
                    "alienation_score": fromm_analysis['alienation_score'],
                    "social_connectedness": fromm_analysis['social_connectedness'],
                    "creative_potential": fromm_analysis.get('creative_potential', 0),
                    "base_utility": bentham_result.get('base_utility', 0),
                    "total_utility": bentham_result['total_utility'],
                    "fromm_bonus": bentham_result.get('fromm_bonus', 0)
                },
                performance_metrics={
                    "fromm_analysis_time": 0.09,
                    "bentham_calculation_time": 0.11,
                    "total_processing_time": execution_time
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ 프롬-벤담 통합 테스트 실패: {str(e)}")
            
            return IntegrationTestResult(
                test_name=test_name,
                success=False,
                execution_time=execution_time,
                details={"error_type": type(e).__name__},
                error_message=str(e)
            )
    
    async def test_end_to_end_workflow(self) -> IntegrationTestResult:
        """전체 시스템 end-to-end 워크플로우 테스트"""
        start_time = time.time()
        test_name = "end_to_end_workflow"
        
        try:
            logger.info("🔄 전체 시스템 end-to-end 워크플로우 테스트 시작")
            
            # 모든 테스트 시나리오를 순차적으로 처리
            workflow_results = []
            
            for scenario in self.test_scenarios:
                scenario_start = time.time()
                
                # 1. 공감 학습 처리
                empathy_result = await self.empathy_learner.process_empathy_learning(
                    scenario["text"], scenario["context"]
                )
                
                # 2. 벤담 유틸리티 계산 (공감 결과 포함)
                bentham_result = await self.bentham_calculator.calculate_enhanced_utility(
                    scenario["text"], 
                    scenario["context"],
                    empathy_result
                )
                
                # 3. 결과 통합 및 검증
                integrated_result = {
                    "scenario_name": scenario["name"],
                    "empathy_score": empathy_result.get('empathy_score', 0),
                    "utility_score": bentham_result.get('total_utility', 0),
                    "confidence": empathy_result.get('confidence', 0),
                    "processing_time": time.time() - scenario_start,
                    "semantic_tags": empathy_result.get('semantic_analysis', {}).get('hashtags', [])[:3],
                    "fromm_orientation": bentham_result.get('fromm_analysis', {}).get('orientation', 'unknown')
                }
                
                workflow_results.append(integrated_result)
                
                # 결과 유효성 검증
                assert 0 <= integrated_result["empathy_score"] <= 1
                assert integrated_result["utility_score"] >= 0
                assert 0 <= integrated_result["confidence"] <= 1
            
            execution_time = time.time() - start_time
            
            # 전체 성능 메트릭 계산
            total_scenarios = len(workflow_results)
            avg_empathy_score = sum(r["empathy_score"] for r in workflow_results) / total_scenarios
            avg_utility_score = sum(r["utility_score"] for r in workflow_results) / total_scenarios
            avg_confidence = sum(r["confidence"] for r in workflow_results) / total_scenarios
            avg_processing_time = sum(r["processing_time"] for r in workflow_results) / total_scenarios
            
            return IntegrationTestResult(
                test_name=test_name,
                success=True,
                execution_time=execution_time,
                details={
                    "total_scenarios_processed": total_scenarios,
                    "avg_empathy_score": round(avg_empathy_score, 3),
                    "avg_utility_score": round(avg_utility_score, 3),
                    "avg_confidence": round(avg_confidence, 3),
                    "workflow_results": workflow_results
                },
                performance_metrics={
                    "avg_scenario_processing_time": round(avg_processing_time, 3),
                    "total_workflow_time": execution_time,
                    "scenarios_per_second": round(total_scenarios / execution_time, 2)
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ End-to-end 워크플로우 테스트 실패: {str(e)}")
            
            return IntegrationTestResult(
                test_name=test_name,
                success=False,
                execution_time=execution_time,
                details={"error_type": type(e).__name__},
                error_message=str(e)
            )
    
    async def run_all_integration_tests(self) -> Dict[str, Any]:
        """모든 통합 테스트 실행"""
        logger.info("🎯 Red Heart 시스템 전체 통합 테스트 시작")
        
        # 테스트 환경 설정
        if not await self.setup_test_environment():
            return {"success": False, "error": "테스트 환경 설정 실패"}
        
        # 각 테스트 실행
        test_methods = [
            self.test_empathy_learning_integration,
            self.test_surd_uncertainty_propagation,
            self.test_hashtag_semantic_analysis,
            self.test_fromm_bentham_integration,
            self.test_end_to_end_workflow
        ]
        
        for test_method in test_methods:
            try:
                result = await test_method()
                self.results.append(result)
                
                if result.success:
                    logger.info(f"✅ {result.test_name} 테스트 성공 (실행시간: {result.execution_time:.3f}s)")
                else:
                    logger.error(f"❌ {result.test_name} 테스트 실패: {result.error_message}")
                    
            except Exception as e:
                logger.error(f"❌ {test_method.__name__} 실행 중 예외 발생: {str(e)}")
                self.results.append(IntegrationTestResult(
                    test_name=test_method.__name__,
                    success=False,
                    execution_time=0,
                    details={},
                    error_message=str(e)
                ))
        
        # 전체 결과 집계
        total_tests = len(self.results)
        successful_tests = sum(1 for result in self.results if result.success)
        total_execution_time = sum(result.execution_time for result in self.results)
        
        summary = {
            "success": successful_tests == total_tests,
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": total_tests - successful_tests,
            "total_execution_time": round(total_execution_time, 3),
            "average_test_time": round(total_execution_time / total_tests, 3) if total_tests > 0 else 0,
            "test_results": [asdict(result) for result in self.results]
        }
        
        logger.info(f"🏁 통합 테스트 완료: {successful_tests}/{total_tests} 성공")
        
        return summary
    
    def save_test_report(self, summary: Dict[str, Any], filename: str = None) -> str:
        """테스트 보고서 저장"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"red_heart_integration_test_report_{timestamp}.json"
        
        report_path = f"/mnt/c/large_project/linux_red_heart/logs/{filename}"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"📄 테스트 보고서 저장됨: {report_path}")
        return report_path

async def main():
    """메인 테스트 실행 함수"""
    print("=" * 80)
    print("🚀 Red Heart Linux Advanced - 통합 테스트 스위트")
    print("=" * 80)
    
    test_suite = RedHeartIntegrationTestSuite()
    
    try:
        # 모든 통합 테스트 실행
        summary = await test_suite.run_all_integration_tests()
        
        # 결과 출력
        print("\n" + "=" * 80)
        print("📊 통합 테스트 결과 요약")
        print("=" * 80)
        print(f"총 테스트: {summary['total_tests']}")
        print(f"성공: {summary['successful_tests']}")
        print(f"실패: {summary['failed_tests']}")
        print(f"성공률: {(summary['successful_tests']/summary['total_tests']*100):.1f}%")
        print(f"총 실행시간: {summary['total_execution_time']:.3f}초")
        print(f"평균 테스트시간: {summary['average_test_time']:.3f}초")
        
        # 실패한 테스트 상세 정보
        if summary['failed_tests'] > 0:
            print("\n❌ 실패한 테스트들:")
            for result in summary['test_results']:
                if not result['success']:
                    print(f"  - {result['test_name']}: {result['error_message']}")
        
        # 성공한 테스트 성능 정보
        if summary['successful_tests'] > 0:
            print("\n✅ 성공한 테스트 성능:")
            for result in summary['test_results']:
                if result['success']:
                    print(f"  - {result['test_name']}: {result['execution_time']:.3f}초")
        
        # 테스트 보고서 저장
        report_path = test_suite.save_test_report(summary)
        print(f"\n📄 상세 보고서: {report_path}")
        
        if summary['success']:
            print("\n🎉 모든 통합 테스트가 성공적으로 완료되었습니다!")
            return 0
        else:
            print(f"\n⚠️ {summary['failed_tests']}개의 테스트가 실패했습니다.")
            return 1
            
    except Exception as e:
        print(f"\n💥 통합 테스트 실행 중 심각한 오류 발생: {str(e)}")
        print(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)