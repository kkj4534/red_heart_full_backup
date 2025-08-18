#!/usr/bin/env python3
"""
Red Heart AI 실제 통합 시스템 훈련 테스트
Real Integrated Training Test for Red Heart AI System

실제 모듈들을 호출하여 진짜 성능을 테스트
"""

import asyncio
import logging
import time
import numpy as np
import torch
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
import traceback
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('RedHeart_Real_Training')

# 시스템 모듈 임포트
try:
    from advanced_emotion_analyzer import AdvancedEmotionAnalyzer
    from advanced_bentham_calculator import AdvancedBenthamCalculator
    from advanced_regret_analyzer import AdvancedRegretAnalyzer
    from advanced_surd_analyzer import AdvancedSURDAnalyzer
    from advanced_experience_database import AdvancedExperienceDatabase
    from data_models import EthicalSituation, EmotionData, HedonicValues
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    logger.error(f"모듈 임포트 실패: {e}")

@dataclass
class TrainingScenario:
    """훈련 시나리오 데이터 구조"""
    id: str
    title: str
    description: str
    context: Dict[str, Any]
    stakeholders: Dict[str, float]
    optimal_choice: str
    alternative_choices: List[str]

@dataclass
class RealTrainingResult:
    """실제 훈련 결과 데이터 구조"""
    scenario_id: str
    processing_time: float
    emotion_analysis: Dict[str, Any]
    bentham_calculation: Dict[str, Any]
    regret_analysis: Dict[str, Any]
    surd_analysis: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None

class RealIntegratedTrainingTestSystem:
    """실제 통합 훈련 시스템"""
    
    def __init__(self):
        self.emotion_analyzer = None
        self.bentham_calculator = None
        self.regret_analyzer = None
        self.surd_analyzer = None
        self.experience_db = None
        
        # 훈련 메트릭
        self.training_metrics = {
            'total_scenarios': 0,
            'successful_completions': 0,
            'total_processing_time': 0,
            'module_call_times': {
                'emotion': [],
                'bentham': [],
                'regret': [],
                'surd': []
            }
        }
        
    async def initialize_system(self):
        """전체 시스템 초기화"""
        logger.info("=== Red Heart AI 실제 통합 훈련 시스템 초기화 ===")
        
        try:
            # 감정 분석기 초기화
            logger.info("감정 분석기 초기화...")
            self.emotion_analyzer = AdvancedEmotionAnalyzer()
            logger.info("✅ 감정 분석기 준비 완료")
            
            # 벤담 계산기 초기화
            logger.info("벤담 계산기 초기화...")
            self.bentham_calculator = AdvancedBenthamCalculator()
            logger.info("✅ 벤담 계산기 준비 완료")
            
            # 후회 분석기 초기화
            logger.info("후회 분석기 초기화...")
            try:
                self.regret_analyzer = AdvancedRegretAnalyzer()
                logger.info("✅ 후회 분석기 준비 완료")
            except Exception as e:
                logger.warning(f"⚠️ 후회 분석기 초기화 실패: {e}")
                self.regret_analyzer = None
            
            # SURD 분석기 초기화
            logger.info("SURD 분석기 초기화...")
            self.surd_analyzer = AdvancedSURDAnalyzer()
            logger.info("✅ SURD 분석기 준비 완료")
            
            # 경험 데이터베이스 초기화
            logger.info("경험 데이터베이스 초기화...")
            self.experience_db = AdvancedExperienceDatabase()
            logger.info("✅ 경험 데이터베이스 준비 완료")
            
            logger.info("🎯 실제 통합 시스템 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"❌ 시스템 초기화 실패: {e}")
            traceback.print_exc()
            return False
    
    def create_test_scenarios(self) -> List[TrainingScenario]:
        """3개 테스트 시나리오 생성 (간단한 버전)"""
        
        scenarios = [
            TrainingScenario(
                id="real_test_001",
                title="자율주행차 윤리적 딜레마",
                description="자율주행차가 급브레이크 실패 시 직진하여 5명을 칠 것인가, 옆길로 틀어 1명을 칠 것인가?",
                context={
                    "situation_type": "autonomous_vehicle",
                    "urgency_level": 0.95,
                    "legal_implications": 0.8,
                    "public_safety": 0.9,
                    "individual_rights": 0.7
                },
                stakeholders={
                    "passenger": 0.8,
                    "pedestrians_group": 0.9,
                    "individual_pedestrian": 0.95,
                    "society": 0.6,
                    "manufacturer": 0.5
                },
                optimal_choice="minimize_total_harm",
                alternative_choices=["protect_passenger", "random_choice", "no_action"]
            ),
            
            TrainingScenario(
                id="real_test_002", 
                title="의료 자원 배분 딜레마",
                description="코로나19 상황에서 인공호흡기 1대를 두고 90세 환자와 30세 환자 중 누구를 선택할 것인가?",
                context={
                    "situation_type": "medical_resource",
                    "urgency_level": 0.9,
                    "life_expectancy_factor": 0.8,
                    "social_contribution": 0.6,
                    "medical_priority": 0.7
                },
                stakeholders={
                    "elderly_patient": 0.9,
                    "young_patient": 0.9,
                    "families": 0.8,
                    "medical_staff": 0.7,
                    "healthcare_system": 0.6
                },
                optimal_choice="medical_priority_based",
                alternative_choices=["age_priority", "first_come_first_served", "lottery_system"]
            ),
            
            TrainingScenario(
                id="real_test_003",
                title="개인정보 vs 공공안전",
                description="테러 방지를 위해 시민들의 개인정보를 수집하고 감시할 것인가, 개인의 프라이버시를 보호할 것인가?",
                context={
                    "situation_type": "privacy_security",
                    "urgency_level": 0.8,
                    "security_threat": 0.9,
                    "privacy_rights": 0.9,
                    "democratic_values": 0.8
                },
                stakeholders={
                    "citizens": 0.9,
                    "government": 0.7,
                    "potential_victims": 0.9,
                    "civil_rights_groups": 0.8,
                    "security_agencies": 0.6
                },
                optimal_choice="balanced_approach",
                alternative_choices=["full_surveillance", "no_surveillance", "voluntary_participation"]
            )
        ]
        
        return scenarios
    
    async def _safe_module_call(self, module_name: str, func_call, context: str, max_retries: int = 2):
        """안전한 모듈 호출 (Circuit Breaker 패턴)"""
        import asyncio
        import gc
        
        for attempt in range(max_retries + 1):
            try:
                # 메모리 체크
                import psutil
                memory = psutil.virtual_memory()
                if memory.percent > 90:
                    logger.warning(f"높은 메모리 사용률 감지: {memory.percent}%, 가비지 컬렉션 실행")
                    gc.collect()
                
                # GPU 메모리 체크
                import torch
                if torch.cuda.is_available():
                    gpu_percent = (torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory) * 100
                    if gpu_percent > 85:
                        logger.warning(f"높은 GPU 메모리 사용률: {gpu_percent:.1f}%, 캐시 정리")
                        torch.cuda.empty_cache()
                
                # 실제 모듈 호출
                if asyncio.iscoroutinefunction(func_call):
                    result = await func_call()
                else:
                    result = func_call()
                
                if result:
                    logger.info(f"✅ {module_name} 모듈 성공 (시도 {attempt + 1}/{max_retries + 1})")
                    return result
                else:
                    logger.warning(f"⚠️ {module_name} 모듈 결과 없음 (시도 {attempt + 1}/{max_retries + 1})")
                    
            except Exception as e:
                logger.warning(f"⚠️ {module_name} 모듈 실패 (시도 {attempt + 1}/{max_retries + 1}): {e}")
                
                if attempt < max_retries:
                    # 재시도 전 대기 (지수적 백오프)
                    wait_time = 2 ** attempt
                    logger.info(f"🔄 {wait_time}초 후 재시도...")
                    await asyncio.sleep(wait_time)
                else:
                    # 모든 재시도 실패 시 예외 발생
                    error_msg = f"{module_name} 분석 실패 - fallback 금지로 시스템 정지: {e}"
                    logger.error(f"❌ {error_msg}")
                    raise RuntimeError(error_msg)
        
        # 결과가 없고 모든 재시도 완료 시
        error_msg = f"{module_name} 분석이 결과를 반환하지 않음 - fallback 금지로 시스템 정지"
        logger.error(f"❌ {error_msg}")
        raise RuntimeError(error_msg)
    
    async def run_real_scenario_training(self, scenario: TrainingScenario) -> RealTrainingResult:
        """단일 시나리오로 실제 통합 훈련 수행"""
        logger.info(f"\n🎯 시나리오 '{scenario.title}' 실제 통합 훈련 시작")
        
        start_time = time.time()
        
        try:
            # 1. 실제 감정 분석 수행 (재시도 로직 포함)
            logger.info("1️⃣ 실제 감정 분석 수행...")
            emotion_start = time.time()
            
            emotion_result = await self._safe_module_call(
                'emotion', 
                lambda: self.emotion_analyzer.analyze_emotion(scenario.description),
                scenario.title
            )
            emotion_time = time.time() - emotion_start
            self.training_metrics['module_call_times']['emotion'].append(emotion_time)
            
            if emotion_result:
                logger.info(f"   ✅ 감정 분석 완료 - 처리시간: {emotion_time:.2f}초")
                
                # 실제 감정 분석 결과 속성 확인 (fallback 금지)
                if not hasattr(emotion_result, 'primary_emotion'):
                    raise RuntimeError(f"감정 결과에 primary_emotion 속성이 없음: {type(emotion_result)}")
                if not hasattr(emotion_result, 'confidence'):
                    raise RuntimeError(f"감정 결과에 confidence 속성이 없음: {type(emotion_result)}")
                if not hasattr(emotion_result, 'arousal'):
                    raise RuntimeError(f"감정 결과에 arousal 속성이 없음: {type(emotion_result)}")
                if not hasattr(emotion_result, 'valence'):
                    raise RuntimeError(f"감정 결과에 valence 속성이 없음: {type(emotion_result)}")
                
                emotion_data = {
                    'emotion': emotion_result.primary_emotion.name,  # EmotionState enum의 name
                    'confidence': emotion_result.confidence,
                    'arousal': emotion_result.arousal,
                    'valence': emotion_result.valence,
                    'intensity': emotion_result.intensity.name,  # EmotionIntensity enum의 name
                    'processing_method': emotion_result.processing_method,
                    'processing_time': emotion_time,
                    'success': True
                }
                
                logger.info(f"   📊 감정 분석 실제 값:")
                logger.info(f"      주요 감정: {emotion_result.primary_emotion.name}")
                logger.info(f"      신뢰도: {emotion_result.confidence:.3f}")
                logger.info(f"      강도: {emotion_result.intensity.name}")
                logger.info(f"      각성도: {emotion_result.arousal:.3f}")
                logger.info(f"      감정가: {emotion_result.valence:.3f}")
                logger.info(f"      처리 방법: {emotion_result.processing_method}")
            else:
                # fallback 금지 원칙: 감정 분석 실패 시 시스템 정지
                error_msg = "LLM 감정 분석 실패 - fallback 금지로 시스템 정지: LLM 감정 분석이 None을 반환 - EmoLLMs 전처리 후에도 실패"
                logger.error(f"   ❌ {error_msg}")
                
                # 실패 시 예외 발생으로 시스템 정지
                raise RuntimeError(error_msg)
            
            # 2. 실제 벤담 계산 수행 (재시도 로직 포함)
            logger.info("2️⃣ 실제 벤담 계산 수행...")
            bentham_start = time.time()
            
            # EthicalSituation 객체 생성 (올바른 매개변수 사용)
            ethical_situation = EthicalSituation(
                title=scenario.title,
                description=scenario.description,
                context={
                    **scenario.context,
                    'stakeholders': scenario.stakeholders,
                    'alternatives': scenario.alternative_choices
                }
            )
            
            bentham_result = await self._safe_module_call(
                'bentham',
                lambda: self.bentham_calculator.calculate_with_ethical_reasoning(
                    input_data={
                        'situation': ethical_situation,
                        'text': scenario.description
                    },
                    community_emotion=emotion_result if emotion_result else None
                ),
                scenario.title
            )
            bentham_time = time.time() - bentham_start
            self.training_metrics['module_call_times']['bentham'].append(bentham_time)
            
            if bentham_result:
                logger.info(f"   ✅ 벤담 계산 완료 - 처리시간: {bentham_time:.2f}초")
                
                # 실제 벤담 계산 결과 속성 확인 (fallback 금지)
                if not hasattr(bentham_result, 'final_score'):
                    raise RuntimeError(f"벤담 결과에 final_score 속성이 없음: {type(bentham_result)}")
                if not hasattr(bentham_result, 'confidence_score'):
                    raise RuntimeError(f"벤담 결과에 confidence_score 속성이 없음: {type(bentham_result)}")
                if not hasattr(bentham_result, 'hedonic_values'):
                    raise RuntimeError(f"벤담 결과에 hedonic_values 속성이 없음: {type(bentham_result)}")
                
                bentham_data = {
                    'final_score': bentham_result.final_score,
                    'base_score': bentham_result.base_score,
                    'confidence_score': bentham_result.confidence_score,
                    'intensity': bentham_result.hedonic_values.intensity,
                    'duration': bentham_result.hedonic_values.duration,
                    'certainty': bentham_result.hedonic_values.certainty,
                    'purity': bentham_result.hedonic_values.purity,
                    'extent': bentham_result.hedonic_values.extent,
                    'hedonic_total': bentham_result.hedonic_values.hedonic_total,
                    'processing_time': bentham_time,
                    'success': True
                }
                
                logger.info(f"   📊 벤담 계산 실제 값:")
                logger.info(f"      최종 점수: {bentham_result.final_score:.3f}")
                logger.info(f"      기본 점수: {bentham_result.base_score:.3f}")
                logger.info(f"      신뢰도: {bentham_result.confidence_score:.3f}")
                logger.info(f"      강도: {bentham_result.hedonic_values.intensity:.3f}")
                logger.info(f"      지속성: {bentham_result.hedonic_values.duration:.3f}")
                logger.info(f"      확실성: {bentham_result.hedonic_values.certainty:.3f}")
                logger.info(f"      총 쾌락값: {bentham_result.hedonic_values.hedonic_total:.3f}")
            else:
                # fallback 금지 원칙: 벤담 계산 실패 시 시스템 정지
                error_msg = "벤담 계산 실패 - fallback 금지로 시스템 정지: 벤담 계산이 None을 반환"
                logger.error(f"   ❌ {error_msg}")
                
                # 실패 시 예외 발생으로 시스템 정지
                raise RuntimeError(error_msg)
            
            # 3. 실제 후회 분석 수행 (가능한 경우)
            regret_data = {'success': False, 'processing_time': 0.0}
            if self.regret_analyzer:
                logger.info("3️⃣ 실제 후회 분석 수행...")
                regret_start = time.time()
                
                try:
                    # 후회 분석을 위한 데이터 준비
                    decision_data = {
                        'text': scenario.description,
                        'context': scenario.context,
                        'alternatives': scenario.alternative_choices,
                        'optimal_choice': scenario.optimal_choice,
                        'stakeholders': scenario.stakeholders
                    }
                    
                    # 실제 후회 분석 수행
                    regret_result = await self.regret_analyzer.analyze_regret(decision_data)
                    regret_time = time.time() - regret_start
                    self.training_metrics['module_call_times']['regret'].append(regret_time)
                    
                    logger.info(f"   ✅ 후회 분석 완료 - 처리시간: {regret_time:.2f}초")
                    
                    # 실제 후회 분석 결과 속성 확인 (fallback 금지)
                    if not hasattr(regret_result, 'regret_intensity'):
                        raise RuntimeError(f"후회 결과에 regret_intensity 속성이 없음: {type(regret_result)}")
                    if not hasattr(regret_result, 'anticipated_regret'):
                        raise RuntimeError(f"후회 결과에 anticipated_regret 속성이 없음: {type(regret_result)}")
                    if not hasattr(regret_result, 'experienced_regret'):
                        raise RuntimeError(f"후회 결과에 experienced_regret 속성이 없음: {type(regret_result)}")
                    
                    regret_data = {
                        'regret_intensity': regret_result.regret_intensity,
                        'anticipated_regret': regret_result.anticipated_regret,
                        'experienced_regret': regret_result.experienced_regret,
                        'regret_duration': regret_result.regret_duration,
                        'semantic_regret_score': regret_result.semantic_regret_score,
                        'model_confidence': regret_result.model_confidence,
                        'uncertainty_estimate': regret_result.uncertainty_estimate,
                        'processing_time': regret_time,
                        'success': True
                    }
                    
                    logger.info(f"   📊 후회 분석 실제 값:")
                    logger.info(f"      후회 강도: {regret_result.regret_intensity:.3f}")
                    logger.info(f"      예상 후회: {regret_result.anticipated_regret:.3f}")
                    logger.info(f"      경험 후회: {regret_result.experienced_regret:.3f}")
                    logger.info(f"      후회 지속: {regret_result.regret_duration:.3f}")
                    logger.info(f"      의미적 후회: {regret_result.semantic_regret_score:.3f}")
                    logger.info(f"      모델 신뢰도: {regret_result.model_confidence:.3f}")
                except Exception as e:
                    # fallback 금지 원칙: 후회 분석 실패 시 시스템 정지
                    error_msg = f"후회 분석 실패 - fallback 금지로 시스템 정지: {e}"
                    logger.error(f"   ❌ {error_msg}")
                    
                    # 실패 시 예외 발생으로 시스템 정지
                    raise RuntimeError(error_msg)
            else:
                # fallback 금지 원칙: 후회 분석기 초기화 실패 시 시스템 정지
                error_msg = "후회 분석기 초기화 실패 - fallback 금지로 시스템 정지: 후회 분석기가 None"
                logger.error(f"   ❌ {error_msg}")
                
                # 실패 시 예외 발생으로 시스템 정지
                raise RuntimeError(error_msg)
            
            # 4. 실제 SURD 분석 수행
            logger.info("4️⃣ 실제 SURD 분석 수행...")
            surd_start = time.time()
            
            try:
                # 실제 SURD 분석 수행 (통합 시스템 분석)
                surd_result = await self.surd_analyzer.analyze_integrated_system(
                    emotion_data=emotion_data,
                    bentham_data=bentham_data,
                    target_variable='decision_quality',
                    additional_context={
                        'scenario': scenario.description,
                        'context': scenario.context,
                        'stakeholders': scenario.stakeholders
                    }
                )
                surd_time = time.time() - surd_start
                self.training_metrics['module_call_times']['surd'].append(surd_time)
                
                logger.info(f"   ✅ SURD 분석 완료 - 처리시간: {surd_time:.2f}초")
                
                # 실제 SURD 분석 결과 값 사용 (fallback 금지)
                if not hasattr(surd_result, 'synergy_score'):
                    raise RuntimeError(f"SURD 결과에 synergy_score 속성이 없음: {type(surd_result)}")
                if not hasattr(surd_result, 'uniqueness_score'):
                    raise RuntimeError(f"SURD 결과에 uniqueness_score 속성이 없음: {type(surd_result)}")
                if not hasattr(surd_result, 'redundancy_score'):
                    raise RuntimeError(f"SURD 결과에 redundancy_score 속성이 없음: {type(surd_result)}")
                if not hasattr(surd_result, 'determinism_score'):
                    raise RuntimeError(f"SURD 결과에 determinism_score 속성이 없음: {type(surd_result)}")
                
                surd_data = {
                    'synergy_score': surd_result.synergy_score,
                    'unique_score': surd_result.uniqueness_score,
                    'redundant_score': surd_result.redundancy_score,
                    'deterministic_score': surd_result.determinism_score,
                    'overall_score': surd_result.overall_score,
                    'confidence_score': surd_result.confidence_score,
                    'processing_time': surd_time,
                    'success': True
                }
                
                logger.info(f"   📊 SURD 실제 분석 값:")
                logger.info(f"      시너지: {surd_result.synergy_score:.3f}")
                logger.info(f"      고유성: {surd_result.uniqueness_score:.3f}")  
                logger.info(f"      중복성: {surd_result.redundancy_score:.3f}")
                logger.info(f"      결정성: {surd_result.determinism_score:.3f}")
                logger.info(f"      종합점수: {surd_result.overall_score:.3f}")
                logger.info(f"      신뢰도: {surd_result.confidence_score:.3f}")
            except Exception as e:
                # fallback 금지 원칙: SURD 분석 실패 시 시스템 정지
                error_msg = f"SURD 분석 실패 - fallback 금지로 시스템 정지: {e}"
                logger.error(f"   ❌ {error_msg}")
                
                # 실패 시 예외 발생으로 시스템 정지
                raise RuntimeError(error_msg)
            
            # 전체 처리 시간 계산
            total_time = time.time() - start_time
            
            # 훈련 결과 생성
            result = RealTrainingResult(
                scenario_id=scenario.id,
                processing_time=total_time,
                emotion_analysis=emotion_data,
                bentham_calculation=bentham_data,
                regret_analysis=regret_data,
                surd_analysis=surd_data,
                success=True
            )
            
            # 메트릭 업데이트 
            self.training_metrics['total_scenarios'] += 1
            self.training_metrics['successful_completions'] += 1
            self.training_metrics['total_processing_time'] += total_time
            
            logger.info(f"✅ 시나리오 '{scenario.title}' 실제 훈련 완료")
            logger.info(f"   전체 처리시간: {total_time:.3f}초")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 시나리오 '{scenario.title}' 실제 훈련 실패: {e}")
            traceback.print_exc()
            
            return RealTrainingResult(
                scenario_id=scenario.id,
                processing_time=time.time() - start_time,
                emotion_analysis={'success': False},
                bentham_calculation={'success': False},
                regret_analysis={'success': False},
                surd_analysis={'success': False},
                success=False,
                error_message=str(e)
            )
    
    async def run_real_integrated_training(self) -> Dict[str, Any]:
        """3개 시나리오로 실제 통합 훈련 실행"""
        logger.info("🚀 Red Heart AI 실제 통합 훈련 테스트 시작")
        
        # 시나리오 생성
        scenarios = self.create_test_scenarios()
        logger.info(f"📋 {len(scenarios)}개 실제 훈련 시나리오 준비 완료")
        
        training_results = []
        
        # 각 시나리오별 실제 훈련
        for i, scenario in enumerate(scenarios, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"🎯 시나리오 {i}/{len(scenarios)} 실제 훈련 중...")
            logger.info(f"{'='*60}")
            
            result = await self.run_real_scenario_training(scenario)
            training_results.append(result)
            
            # 시나리오 간 간격 (GPU 메모리 정리 시간)
            await asyncio.sleep(2.0)
        
        # 전체 훈련 결과 분석
        return self._analyze_real_training_results(training_results)
    
    def _analyze_real_training_results(self, results: List[RealTrainingResult]) -> Dict[str, Any]:
        """실제 훈련 결과 종합 분석"""
        logger.info(f"\n📊 실제 훈련 결과 분석 중...")
        
        if not results:
            return {"error": "훈련 결과가 없습니다"}
        
        successful_results = [r for r in results if r.success]
        success_rate = len(successful_results) / len(results) * 100
        
        total_processing_time = sum(r.processing_time for r in results)
        avg_processing_time = total_processing_time / len(results)
        
        # 모듈별 성능 분석
        module_performance = {
            'emotion': {
                'success_rate': sum(1 for r in results if r.emotion_analysis.get('success', False)) / len(results) * 100,
                'avg_time': np.mean(self.training_metrics['module_call_times']['emotion']) if self.training_metrics['module_call_times']['emotion'] else 0
            },
            'bentham': {
                'success_rate': sum(1 for r in results if r.bentham_calculation.get('success', False)) / len(results) * 100,
                'avg_time': np.mean(self.training_metrics['module_call_times']['bentham']) if self.training_metrics['module_call_times']['bentham'] else 0
            },
            'regret': {
                'success_rate': sum(1 for r in results if r.regret_analysis.get('success', False)) / len(results) * 100,
                'avg_time': np.mean(self.training_metrics['module_call_times']['regret']) if self.training_metrics['module_call_times']['regret'] else 0
            },
            'surd': {
                'success_rate': sum(1 for r in results if r.surd_analysis.get('success', False)) / len(results) * 100,
                'avg_time': np.mean(self.training_metrics['module_call_times']['surd']) if self.training_metrics['module_call_times']['surd'] else 0
            }
        }
        
        # 종합 결과
        analysis_result = {
            'real_training_summary': {
                'total_scenarios': len(results),
                'successful_completions': len(successful_results),
                'success_rate': success_rate,
                'total_processing_time': total_processing_time,
                'avg_processing_time': avg_processing_time
            },
            'module_performance': module_performance,
            'detailed_results': [
                {
                    'scenario_id': r.scenario_id,
                    'success': r.success,
                    'processing_time': r.processing_time,
                    'emotion_success': r.emotion_analysis.get('success', False),
                    'bentham_success': r.bentham_calculation.get('success', False),
                    'regret_success': r.regret_analysis.get('success', False),
                    'surd_success': r.surd_analysis.get('success', False),
                    'error': r.error_message
                }
                for r in results
            ],
            'performance_insights': self._generate_real_performance_insights(module_performance, success_rate)
        }
        
        return analysis_result
    
    def _generate_real_performance_insights(self, module_performance: Dict, success_rate: float) -> List[str]:
        """실제 성능 기반 개선 권장사항 생성"""
        insights = []
        
        if success_rate < 70:
            insights.append("전체 성공률이 낮습니다. 시스템 안정성 개선이 필요합니다.")
        elif success_rate >= 90:
            insights.append("우수한 성능을 보입니다. 더 복잡한 시나리오로 확장 가능합니다.")
        
        for module, perf in module_performance.items():
            if perf['success_rate'] < 70:
                insights.append(f"{module} 모듈의 성공률이 {perf['success_rate']:.1f}%로 낮습니다.")
            if perf['avg_time'] > 60:
                insights.append(f"{module} 모듈의 평균 처리시간이 {perf['avg_time']:.1f}초로 깁니다.")
        
        if not insights:
            insights.append("모든 모듈이 정상적으로 작동하고 있습니다.")
        
        return insights


async def main():
    """메인 실제 훈련 함수"""
    if not MODULES_AVAILABLE:
        logger.error("❌ 필수 모듈을 불러올 수 없습니다.")
        return
    
    # 실제 통합 훈련 시스템 초기화
    training_system = RealIntegratedTrainingTestSystem()
    
    # 시스템 초기화
    if not await training_system.initialize_system():
        logger.error("❌ 시스템 초기화 실패")
        return
    
    # 실제 통합 훈련 실행
    results = await training_system.run_real_integrated_training()
    
    # 결과 출력
    logger.info(f"\n{'='*80}")
    logger.info("🎉 Red Heart AI 실제 통합 훈련 테스트 완료")
    logger.info(f"{'='*80}")
    
    if 'error' not in results:
        summary = results['real_training_summary']
        performance = results['module_performance']
        
        logger.info(f"\n📊 실제 훈련 요약:")
        logger.info(f"  - 총 시나리오: {summary['total_scenarios']}개")
        logger.info(f"  - 성공률: {summary['success_rate']:.1f}%")
        logger.info(f"  - 평균 처리시간: {summary['avg_processing_time']:.3f}초")
        logger.info(f"  - 총 처리시간: {summary['total_processing_time']:.3f}초")
        
        logger.info(f"\n🔧 모듈별 성능:")
        for module, perf in performance.items():
            logger.info(f"  - {module}: 성공률 {perf['success_rate']:.1f}%, 평균시간 {perf['avg_time']:.2f}초")
        
        logger.info(f"\n💡 성능 인사이트:")
        for insight in results['performance_insights']:
            logger.info(f"  - {insight}")
        
        # 결과를 파일로 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = Path(f'real_integrated_training_results_{timestamp}.json')
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"\n📄 상세 결과 저장: {result_file}")
    else:
        logger.error(f"❌ 실제 훈련 실패: {results['error']}")

if __name__ == "__main__":
    asyncio.run(main())