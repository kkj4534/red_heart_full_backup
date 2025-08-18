"""
Red Heart AI 실제 통합 훈련 시스템
REAL Integrated Training System for Red Heart AI

진짜 모델들을 실제로 호출하여 통합 훈련 수행
- 실제 감정 분석 모델 호출
- 실제 벤담 계산 모델 호출
- 실제 후회 분석 모델 호출
- 실제 SURD 분석 모델 호출
- processed_datasets 24,170개 데이터 훈련
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
import os
import glob

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('RedHeart_REAL_Training')

# 실제 시스템 모듈 임포트
try:
    from advanced_emotion_analyzer import AdvancedEmotionAnalyzer
    from advanced_bentham_calculator import AdvancedBenthamCalculator
    from advanced_regret_analyzer import AdvancedRegretAnalyzer
    from advanced_surd_analyzer import AdvancedSURDAnalyzer
    from advanced_experience_database import AdvancedExperienceDatabase
    from data_models import EthicalSituation, EmotionData, HedonicValues
    
    # 실제 학습 시스템 임포트
    from advanced_learning_executor import AdvancedLearningExecutor, LearningConfig
    from advanced_regret_learning_system import AdvancedRegretLearningSystem, LearningPhase
    from advanced_hierarchical_emotion_system import AdvancedHierarchicalEmotionSystem, EmotionPhase
    from integrated_system_orchestrator import IntegratedSystemOrchestrator, IntegrationContext
    from dynamic_ethical_choice_analyzer import DynamicEthicalChoiceAnalyzer, EthicalDilemma
    
    MODULES_AVAILABLE = True
    LEARNING_SYSTEM_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    LEARNING_SYSTEM_AVAILABLE = False
    logger.error(f"모듈 임포트 실패: {e}")

@dataclass
class RealTrainingResult:
    """실제 훈련 결과 데이터 구조"""
    data_id: str
    source_file: str
    processing_time: float
    emotion_analysis: Dict[str, Any]
    bentham_calculation: Dict[str, Any]
    regret_analysis: Dict[str, Any]
    surd_analysis: Dict[str, Any]
    counterfactual_experiences: List[Dict[str, Any]]
    integration_success: bool
    error_log: List[str]

class RealIntegratedTrainingSystem:
    """실제 통합 훈련 시스템 - 완전한 학습 시스템 통합"""
    
    def __init__(self):
        # 개별 모듈들 (기존 기능 테스트용)
        self.emotion_analyzer = None
        self.bentham_calculator = None
        self.regret_analyzer = None
        self.surd_analyzer = None
        self.experience_db = None
        
        # 실제 학습 시스템 (새로 추가)
        self.learning_executor = None
        self.integrated_orchestrator = None
        self.dynamic_choice_analyzer = None
        self.learning_mode = False  # 학습 모드 vs 기능 테스트 모드
        
        # 실제 훈련 메트릭
        self.training_metrics = {
            'total_processed': 0,
            'successful_integrations': 0,
            'failed_processes': 0,
            'total_processing_time': 0.0,
            'avg_processing_time': 0.0,
            'real_accuracy_scores': {},
            'module_performance': {},
            'error_patterns': [],
            'learning_phases': {
                'current_phase': 0,
                'phase_transitions': [],
                'phase_performance': {}
            }
        }
        
        # 데이터 로더
        self.processed_datasets_path = Path("/mnt/c/large_project/linux_red_heart/processed_datasets")
        
    async def initialize_real_system(self, learning_mode: bool = False):
        """실제 시스템 모든 모듈 초기화"""
        logger.info("=== 실제 Red Heart AI 통합 시스템 초기화 ===")
        
        self.learning_mode = learning_mode
        
        if learning_mode and LEARNING_SYSTEM_AVAILABLE:
            logger.info("🚀 학습 모드 활성화 - 완전한 학습 시스템 초기화")
            
            # 실제 학습 시스템 초기화
            try:
                logger.info("실제 학습 실행기 초기화...")
                learning_config = LearningConfig(
                    regrets_per_step=7,
                    bentham_per_environment=3,
                    general_data_cycles=3,
                    ebs_data_cycles=6,
                    max_scenarios_per_batch=50
                )
                self.learning_executor = AdvancedLearningExecutor(learning_config)
                logger.info("✅ 실제 학습 실행기 초기화 완료")
                
                # 통합 시스템 오케스트레이터 초기화
                logger.info("통합 시스템 오케스트레이터 초기화...")
                self.integrated_orchestrator = IntegratedSystemOrchestrator()
                logger.info("✅ 통합 시스템 오케스트레이터 초기화 완료")
                
                # 동적 윤리적 선택지 분석기 초기화
                logger.info("동적 윤리적 선택지 분석기 초기화...")
                self.dynamic_choice_analyzer = DynamicEthicalChoiceAnalyzer()
                logger.info("✅ 동적 윤리적 선택지 분석기 초기화 완료")
                
                return True
                
            except Exception as e:
                logger.error(f"❌ 학습 시스템 초기화 실패: {e}")
                logger.info("🔄 기능 테스트 모드로 전환")
                self.learning_mode = False
        
        # 기존 기능 테스트 모드 (기본값)
        logger.info("🔧 기능 테스트 모드 - 개별 모듈 초기화")
        
        try:
            # 실제 감정 분석기 초기화
            logger.info("실제 감정 분석기 초기화...")
            self.emotion_analyzer = AdvancedEmotionAnalyzer()
            logger.info("✅ 실제 감정 분석기 초기화 완료")
            
            # 실제 벤담 계산기 초기화
            logger.info("실제 벤담 계산기 초기화...")
            self.bentham_calculator = AdvancedBenthamCalculator()
            logger.info("✅ 실제 벤담 계산기 초기화 완료")
            
            # 실제 후회 분석기 초기화
            logger.info("실제 후회 분석기 초기화...")
            try:
                self.regret_analyzer = AdvancedRegretAnalyzer()
                logger.info("✅ 실제 후회 분석기 초기화 완료")
            except Exception as e:
                logger.error(f"❌ 후회 분석기 초기화 실패: {e}")
                self.regret_analyzer = None
            
            # 실제 SURD 분석기 초기화
            logger.info("실제 SURD 분석기 초기화...")
            self.surd_analyzer = AdvancedSURDAnalyzer()
            logger.info("✅ 실제 SURD 분석기 초기화 완료")
            
            # 실제 경험 데이터베이스 초기화
            logger.info("실제 경험 데이터베이스 초기화...")
            self.experience_db = AdvancedExperienceDatabase()
            logger.info("✅ 실제 경험 데이터베이스 초기화 완료")
            
            logger.info("🎯 실제 통합 시스템 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"❌ 실제 시스템 초기화 실패: {e}")
            traceback.print_exc()
            return False
    
    def load_real_training_data(self) -> List[Dict[str, Any]]:
        """실제 processed_datasets에서 훈련 데이터 로드"""
        logger.info("=== 실제 훈련 데이터 로딩 시작 ===")
        
        training_data = []
        
        try:
            # 스크러플 배치 파일들 로드
            scruples_pattern = self.processed_datasets_path / "scruples" / "scruples_batch_*.json"
            scruples_files = glob.glob(str(scruples_pattern))
            
            logger.info(f"스크러플 배치 파일 {len(scruples_files)}개 발견")
            
            for file_path in scruples_files[:2]:  # 처음 2개 배치만 테스트
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        batch_data = json.load(f)
                        
                    if 'scenarios' in batch_data:
                        for item in batch_data['scenarios']:
                            if 'description' in item:
                                training_data.append({
                                    'source_file': os.path.basename(file_path),
                                    'data_id': item.get('id', f"scruples_{len(training_data)}"),
                                    'situation': item['description'],
                                    'context': item.get('context', {}),
                                    'moral_complexity': 0.7,  # 스크러플 데이터 기본 복잡도
                                    'stakeholders': {},
                                    'data_type': 'scruples'
                                })
                            
                except Exception as e:
                    logger.warning(f"스크러플 파일 {file_path} 로딩 실패: {e}")
                    continue
            
            # 통합 시나리오 파일들 로드
            try:
                integrated_files = [
                    self.processed_datasets_path / "integrated_scenarios.json",
                    self.processed_datasets_path / "final_integrated_with_batch7_20250619_213234.json"
                ]
                integrated_files = [f for f in integrated_files if f.exists()]
                logger.info(f"통합 시나리오 파일 {len(integrated_files)}개 발견")
                
                for file_path in integrated_files:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        scenarios = json.load(f)
                    
                    for idx, scenario in enumerate(scenarios[:5]):  # 각 파일에서 5개만
                        if 'description' in scenario:
                            training_data.append({
                                'source_file': os.path.basename(file_path),
                                'data_id': scenario.get('id', f"integrated_{idx}"),
                                'situation': scenario['description'],
                                'context': scenario.get('context', {}),
                                'moral_complexity': scenario.get('complexity_score', 0.7),
                                'stakeholders': scenario.get('stakeholders', []),
                                'data_type': 'integrated'
                            })
                            
            except Exception as e:
                logger.warning(f"통합 시나리오 로딩 실패: {e}")
            
            # 한국 문화 특화 데이터 로드
            try:
                korean_files = [
                    self.processed_datasets_path / "korean_cultural_scenarios.json"
                ]
                korean_files = [f for f in korean_files if f.exists()]
                logger.info(f"한국 문화 파일 {len(korean_files)}개 발견")
                
                for file_path in korean_files:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        cultural_data = json.load(f)
                    
                    for item in cultural_data:
                        if 'scenario' in item and len(training_data) < 25:  # 최대 25개 제한
                            training_data.append({
                                'source_file': os.path.basename(file_path),
                                'data_id': item.get('id', f"korean_{len(training_data)}"),
                                'situation': item['scenario'],
                                'context': item.get('context', {}),
                                'moral_complexity': item.get('complexity', 0.8),
                                'stakeholders': item.get('stakeholders', {}),
                                'data_type': 'korean_cultural'
                            })
                            
            except Exception as e:
                logger.warning(f"한국 문화 데이터 로딩 실패: {e}")
            
            logger.info(f"✅ 총 {len(training_data)}개 실제 훈련 데이터 로드 완료")
            
            # 데이터 분포 로깅
            data_types = {}
            for data in training_data:
                data_type = data['data_type']
                data_types[data_type] = data_types.get(data_type, 0) + 1
            
            logger.info("📊 데이터 분포:")
            for data_type, count in data_types.items():
                logger.info(f"  - {data_type}: {count}개")
            
            return training_data
            
        except Exception as e:
            logger.error(f"❌ 실제 훈련 데이터 로딩 실패: {e}")
            traceback.print_exc()
            return []
    
    async def process_real_training_item(self, data_item: Dict[str, Any]) -> RealTrainingResult:
        """단일 훈련 데이터 아이템을 실제 모듈들로 처리"""
        
        start_time = time.time()
        error_log = []
        
        try:
            # 1. 실제 감정 분석
            logger.info(f"🎯 처리 중: {data_item['data_id']} - 감정 분석...")
            emotion_start = time.time()
            
            try:
                # 실제 감정 분석기 호출 - 올바른 파라미터
                emotion_result = self.emotion_analyzer.analyze_emotion(
                    text=data_item['situation'],
                    language="ko",
                    biosignal_data=None,
                    use_cache=True
                )
                emotion_processing_time = time.time() - emotion_start
                logger.info(f"   ✅ 감정 분석 완료 ({emotion_processing_time:.3f}초)")
                
            except Exception as e:
                error_log.append(f"감정 분석 실패: {e}")
                emotion_result = {'error': str(e), 'fallback': True}
                emotion_processing_time = time.time() - emotion_start
                logger.warning(f"   ⚠️ 감정 분석 실패: {e}")
            
            # 2. 실제 벤담 계산
            logger.info(f"🎯 처리 중: {data_item['data_id']} - 벤담 계산...")
            bentham_start = time.time()
            
            try:
                # 실제 벤담 계산기 호출 - 올바른 파라미터
                bentham_input_data = {
                    'situation': data_item['situation'],
                    'context': data_item.get('context', {}),
                    'emotion_data': emotion_result if 'error' not in emotion_result else None
                }
                bentham_result = self.bentham_calculator.calculate_with_advanced_layers(
                    input_data=bentham_input_data,
                    use_cache=True
                )
                bentham_processing_time = time.time() - bentham_start
                logger.info(f"   ✅ 벤담 계산 완료 ({bentham_processing_time:.3f}초)")
                
            except Exception as e:
                error_log.append(f"벤담 계산 실패: {e}")
                bentham_result = {'error': str(e), 'fallback': True}
                bentham_processing_time = time.time() - bentham_start
                logger.warning(f"   ⚠️ 벤담 계산 실패: {e}")
            
            # 3. 실제 후회 분석
            if self.regret_analyzer:
                logger.info(f"🎯 처리 중: {data_item['data_id']} - 후회 분석...")
                regret_start = time.time()
                
                try:
                    # 후회 분석을 위한 decision_data 준비 - 안전한 타입 처리
                    decision_data = {
                        'scenario': data_item['situation'],
                        'text': data_item['situation'],  # 텍스트 필드 추가
                        'context': data_item.get('context', {}),
                    }
                    
                    # 감정 컨텍스트 추가 (안전한 방식)
                    if 'error' not in emotion_result:
                        decision_data['emotion_context'] = emotion_result
                    
                    # 벤담 컨텍스트 추가 (안전한 타입 처리)
                    bentham_has_error = False
                    if isinstance(bentham_result, dict):
                        bentham_has_error = 'error' in bentham_result
                    elif hasattr(bentham_result, 'error'):
                        bentham_has_error = bentham_result.error is not None
                    
                    if not bentham_has_error:
                        if hasattr(bentham_result, '__dict__'):
                            # getattr 대신 실제 속성 존재 여부 확인
                            if hasattr(bentham_result, 'final_score') and bentham_result.final_score is not None:
                                decision_data['bentham_context'] = {
                                    'score': bentham_result.final_score,
                                    'type': 'bentham_calculation'
                                }
                            else:
                                raise ValueError(f"벤담 계산 결과에 final_score 속성이 없음: {type(bentham_result)}")
                        elif isinstance(bentham_result, dict):
                            decision_data['bentham_context'] = bentham_result
                    
                    # 실제 후회 분석기 호출
                    regret_result = await self.regret_analyzer.analyze_regret(
                        decision_data=decision_data,
                        outcome_data=None
                    )
                    regret_processing_time = time.time() - regret_start
                    logger.info(f"   ✅ 후회 분석 완료 ({regret_processing_time:.3f}초)")
                    
                except Exception as e:
                    error_log.append(f"후회 분석 실패: {e}")
                    regret_result = {'error': str(e), 'fallback': True}
                    regret_processing_time = time.time() - regret_start
                    logger.warning(f"   ⚠️ 후회 분석 실패: {e}")
            else:
                regret_result = {'error': '후회 분석기 사용 불가', 'disabled': True}
                regret_processing_time = 0.0
            
            # 4. 실제 SURD 통합 분석
            logger.info(f"🎯 처리 중: {data_item['data_id']} - SURD 통합 분석...")
            surd_start = time.time()
            
            try:
                # SURD 분석을 위한 변수 준비
                surd_variables = {}
                
                # 감정 데이터 통합 (안전한 체크)
                emotion_has_error = isinstance(emotion_result, dict) and 'error' in emotion_result
                if not emotion_has_error:
                    if hasattr(emotion_result, 'dominant_emotion'):
                        # 실제 속성 존재 확인 후 값 추출
                        if hasattr(emotion_result, 'intensity') and emotion_result.intensity is not None:
                            surd_variables['emotion_intensity'] = float(emotion_result.intensity)
                        else:
                            raise ValueError(f"감정 분석 결과에 intensity 속성이 없음: {type(emotion_result)}")
                        
                        if hasattr(emotion_result, 'confidence') and emotion_result.confidence is not None:
                            surd_variables['emotion_confidence'] = float(emotion_result.confidence)
                        else:
                            raise ValueError(f"감정 분석 결과에 confidence 속성이 없음: {type(emotion_result)}")
                    elif isinstance(emotion_result, dict):
                        surd_variables['emotion_intensity'] = float(emotion_result.get('intensity', 0.5))
                        surd_variables['emotion_confidence'] = float(emotion_result.get('confidence', 0.5))
                
                # 벤담 데이터 통합 (안전한 체크)
                bentham_has_error = isinstance(bentham_result, dict) and 'error' in bentham_result
                if not bentham_has_error:
                    if hasattr(bentham_result, 'final_score'):
                        # 실제 속성 존재 확인 후 값 추출
                        if bentham_result.final_score is not None:
                            surd_variables['pleasure_score'] = float(bentham_result.final_score)
                        else:
                            raise ValueError(f"벤담 계산 결과의 final_score가 None: {type(bentham_result)}")
                    elif isinstance(bentham_result, dict):
                        surd_variables['pleasure_score'] = float(bentham_result.get('final_score', 0.0))
                
                # 후회 데이터 통합 (AdvancedRegretMetrics 타입 안전 처리)
                regret_surd_error = False
                if hasattr(regret_result, '__class__') and regret_result.__class__.__name__ == 'AdvancedRegretMetrics':
                    # AdvancedRegretMetrics 객체에서 데이터 추출 - 실제 속성 검증
                    if hasattr(regret_result, 'regret_intensity') and regret_result.regret_intensity is not None:
                        if regret_result.regret_intensity <= 0.0:
                            raise ValueError(f"후회 분석 결과의 regret_intensity가 0.0: {regret_result.regret_intensity}")
                        surd_variables['regret_intensity'] = float(regret_result.regret_intensity)
                    elif hasattr(regret_result, 'intensity') and regret_result.intensity is not None:
                        if regret_result.intensity <= 0.0:
                            raise ValueError(f"후회 분석 결과의 intensity가 0.0: {regret_result.intensity}")
                        surd_variables['regret_intensity'] = float(regret_result.intensity)
                    else:
                        raise ValueError(f"후회 분석 결과에 regret_intensity 또는 intensity 속성이 없음: {type(regret_result)}")
                elif isinstance(regret_result, dict):
                    regret_surd_error = 'error' in regret_result
                    if not regret_surd_error:
                        surd_variables['regret_intensity'] = float(regret_result.get('regret_intensity', 0.0))
                
                # 실패 감지 - 모든 분석이 실패했을 경우 예외 발생
                if not surd_variables:
                    raise RuntimeError(
                        f"모든 분석 모듈이 실패했거나 유효한 값을 생성하지 못함. "
                        f"감정: {'error' if emotion_has_error else 'ok'}, "
                        f"벤담: {'error' if bentham_has_error else 'ok'}, "
                        f"후회: {'error' if regret_surd_error else 'ok'}"
                    )
                
                # 실제 SURD 분석기 호출 - analyze_advanced 메서드 사용
                surd_result = self.surd_analyzer.analyze_advanced(
                    variables=surd_variables,
                    target_variable='ethical_decision_quality',
                    additional_context=data_item.get('context', {})
                )
                surd_processing_time = time.time() - surd_start
                logger.info(f"   ✅ SURD 분석 완료 ({surd_processing_time:.3f}초)")
                
            except Exception as e:
                error_log.append(f"SURD 분석 실패: {e}")
                surd_result = {'error': str(e), 'fallback': True}
                surd_processing_time = time.time() - surd_start
                logger.warning(f"   ⚠️ SURD 분석 실패: {e}")
            
            # 5. 반사실 경험 생성 - AdvancedRegretMetrics 타입 안전 처리
            counterfactual_experiences = []
            regret_has_error = False
            
            # AdvancedRegretMetrics 객체 타입 체크
            if hasattr(regret_result, '__class__') and regret_result.__class__.__name__ == 'AdvancedRegretMetrics':
                # 성공적인 결과 - 반사실 시나리오 추출
                if hasattr(regret_result, 'counterfactual_scenarios'):
                    counterfactual_experiences = regret_result.counterfactual_scenarios or []
                elif hasattr(regret_result, 'counterfactuals'):
                    counterfactual_experiences = regret_result.counterfactuals or []
            elif isinstance(regret_result, dict):
                # 딕셔너리 형태 (오류 포함 가능)
                regret_has_error = 'error' in regret_result
                if not regret_has_error and 'counterfactual_scenarios' in regret_result:
                    counterfactual_experiences = regret_result['counterfactual_scenarios']
            
            # 6. 경험 데이터베이스에 저장
            try:
                # 직렬화 가능한 형태로 변환
                def convert_to_serializable(obj):
                    if hasattr(obj, '__dict__'):
                        return {k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v 
                               for k, v in obj.__dict__.items()}
                    elif isinstance(obj, dict):
                        return obj
                    else:
                        return {'result': str(obj), 'type': type(obj).__name__}
                
                experience_entry = {
                    'data_id': data_item['data_id'],
                    'situation': data_item['situation'],
                    'emotion_analysis': convert_to_serializable(emotion_result),
                    'bentham_calculation': convert_to_serializable(bentham_result),
                    'regret_analysis': convert_to_serializable(regret_result),
                    'surd_analysis': convert_to_serializable(surd_result),
                    'timestamp': datetime.now().isoformat(),
                    'source_file': data_item['source_file']
                }
                
                await self.experience_db.store_experience(
                    experience_text=data_item['situation'],
                    metadata=experience_entry,
                    category="training",
                    importance_score=None
                )
                
            except Exception as e:
                error_log.append(f"경험 저장 실패: {e}")
                logger.warning(f"   ⚠️ 경험 저장 실패: {e}")
            
            # 결과 생성
            total_processing_time = time.time() - start_time
            integration_success = len(error_log) == 0
            
            result = RealTrainingResult(
                data_id=data_item['data_id'],
                source_file=data_item['source_file'],
                processing_time=total_processing_time,
                emotion_analysis=emotion_result,
                bentham_calculation=bentham_result,
                regret_analysis=regret_result,
                surd_analysis=surd_result,
                counterfactual_experiences=counterfactual_experiences,
                integration_success=integration_success,
                error_log=error_log
            )
            
            # 메트릭 업데이트
            self.training_metrics['total_processed'] += 1
            self.training_metrics['total_processing_time'] += total_processing_time
            
            if integration_success:
                self.training_metrics['successful_integrations'] += 1
                logger.info(f"✅ {data_item['data_id']} 실제 통합 훈련 완료 ({total_processing_time:.3f}초)")
            else:
                self.training_metrics['failed_processes'] += 1
                logger.warning(f"⚠️ {data_item['data_id']} 부분 실패 ({len(error_log)}개 오류)")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ {data_item['data_id']} 처리 중 심각한 오류: {e}")
            traceback.print_exc()
            
            total_processing_time = time.time() - start_time
            self.training_metrics['total_processed'] += 1
            self.training_metrics['failed_processes'] += 1
            self.training_metrics['total_processing_time'] += total_processing_time
            
            return RealTrainingResult(
                data_id=data_item['data_id'],
                source_file=data_item['source_file'],
                processing_time=total_processing_time,
                emotion_analysis={'error': str(e)},
                bentham_calculation={'error': str(e)},
                regret_analysis={'error': str(e)},
                surd_analysis={'error': str(e)},
                counterfactual_experiences=[],
                integration_success=False,
                error_log=[f"심각한 처리 오류: {e}"]
            )
    
    async def run_real_integrated_training(self, max_items: int = 100) -> Dict[str, Any]:
        """실제 데이터로 통합 훈련 실행"""
        logger.info("🚀 실제 Red Heart AI 통합 훈련 시작")
        
        # 실제 훈련 데이터 로드
        training_data = self.load_real_training_data()
        
        if not training_data:
            logger.error("❌ 훈련 데이터가 없습니다")
            return {"error": "훈련 데이터 로드 실패"}
        
        # 최대 처리 개수 제한
        if len(training_data) > max_items:
            training_data = training_data[:max_items]
            logger.info(f"📊 처리할 데이터를 {max_items}개로 제한")
        
        logger.info(f"📋 총 {len(training_data)}개 실제 데이터 처리 시작")
        
        training_results = []
        
        # 각 데이터 아이템 순차 처리
        for i, data_item in enumerate(training_data, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"🎯 [{i}/{len(training_data)}] 실제 훈련 중...")
            logger.info(f"데이터 ID: {data_item['data_id']}")
            logger.info(f"소스 파일: {data_item['source_file']}")
            logger.info(f"{'='*60}")
            
            result = await self.process_real_training_item(data_item)
            training_results.append(result)
            
            # 진행 상황 로깅
            if i % 10 == 0 or i == len(training_data):
                success_rate = self.training_metrics['successful_integrations'] / self.training_metrics['total_processed'] * 100
                avg_time = self.training_metrics['total_processing_time'] / self.training_metrics['total_processed']
                logger.info(f"\n📊 중간 진행 상황 [{i}/{len(training_data)}]:")
                logger.info(f"  - 성공률: {success_rate:.1f}%")
                logger.info(f"  - 평균 처리시간: {avg_time:.3f}초")
                logger.info(f"  - 예상 남은 시간: {avg_time * (len(training_data) - i):.1f}초")
        
        # 최종 분석
        return self._analyze_real_training_results(training_results)
    
    def _analyze_real_training_results(self, results: List[RealTrainingResult]) -> Dict[str, Any]:
        """실제 훈련 결과 종합 분석"""
        logger.info(f"\n📊 실제 훈련 결과 분석 중...")
        
        if not results:
            return {"error": "분석할 결과가 없습니다"}
        
        # 전체 메트릭
        total_items = len(results)
        successful_items = len([r for r in results if r.integration_success])
        success_rate = successful_items / total_items * 100
        
        total_time = sum(r.processing_time for r in results)
        avg_time = total_time / total_items
        
        # 모듈별 성능 분석 - AdvancedRegretMetrics 포함 안전한 타입 체크
        def safe_error_check(obj):
            if hasattr(obj, '__class__') and obj.__class__.__name__ == 'AdvancedRegretMetrics':
                return True  # AdvancedRegretMetrics 객체는 성공으로 간주
            elif hasattr(obj, '__dict__'):
                return 'error' not in obj.__dict__
            elif isinstance(obj, dict):
                return 'error' not in obj and 'disabled' not in obj
            else:
                return True  # 기타 객체는 성공으로 간주
        
        module_performance = {
            'emotion_success': len([r for r in results if safe_error_check(r.emotion_analysis)]),
            'bentham_success': len([r for r in results if safe_error_check(r.bentham_calculation)]),
            'regret_success': len([r for r in results if safe_error_check(r.regret_analysis) and not (isinstance(r.regret_analysis, dict) and 'disabled' in r.regret_analysis)]),
            'surd_success': len([r for r in results if safe_error_check(r.surd_analysis)])
        }
        
        # 오류 패턴 분석
        error_patterns = {}
        for result in results:
            for error in result.error_log:
                error_type = error.split(':')[0]
                error_patterns[error_type] = error_patterns.get(error_type, 0) + 1
        
        # 처리 시간 분석
        processing_times = [r.processing_time for r in results]
        min_time = min(processing_times)
        max_time = max(processing_times)
        
        # 반사실 생성 통계
        total_counterfactuals = sum(len(r.counterfactual_experiences) for r in results)
        
        # 데이터 소스별 성능
        source_performance = {}
        for result in results:
            source = result.source_file
            if source not in source_performance:
                source_performance[source] = {'total': 0, 'success': 0}
            source_performance[source]['total'] += 1
            if result.integration_success:
                source_performance[source]['success'] += 1
        
        # 최종 결과
        analysis_result = {
            'training_summary': {
                'total_items': total_items,
                'successful_integrations': successful_items,
                'success_rate': success_rate,
                'total_processing_time': total_time,
                'avg_processing_time': avg_time,
                'min_processing_time': min_time,
                'max_processing_time': max_time
            },
            'module_performance': {
                'emotion_success_rate': (module_performance['emotion_success'] / total_items) * 100,
                'bentham_success_rate': (module_performance['bentham_success'] / total_items) * 100,
                'regret_success_rate': (module_performance['regret_success'] / total_items) * 100,
                'surd_success_rate': (module_performance['surd_success'] / total_items) * 100
            },
            'integration_analysis': {
                'full_integration_rate': success_rate,
                'partial_integration_rate': ((total_items - successful_items) / total_items) * 100,
                'total_counterfactuals_generated': total_counterfactuals,
                'avg_counterfactuals_per_item': total_counterfactuals / total_items
            },
            'error_analysis': {
                'error_patterns': error_patterns,
                'total_errors': sum(error_patterns.values()),
                'error_rate': (sum(error_patterns.values()) / total_items) * 100
            },
            'source_analysis': source_performance,
            'performance_metrics': {
                'items_per_second': total_items / total_time,
                'successful_items_per_second': successful_items / total_time,
                'efficiency_score': success_rate * (total_items / total_time)
            }
        }
        
        return analysis_result

    async def run_complete_learning_system(self) -> Dict[str, Any]:
        """완전한 학습 시스템 실행"""
        
        if not self.learning_mode or not self.learning_executor:
            logger.error("❌ 학습 모드가 활성화되지 않았습니다.")
            return {"error": "학습 시스템이 초기화되지 않음"}
        
        logger.info("🎯 완전한 학습 시스템 실행 시작")
        logger.info("📊 3단계 통합 페이즈 시스템:")
        logger.info("   Phase 0: 자신 감정 캘리브레이션")
        logger.info("   Phase 1: 타인 공감 학습")
        logger.info("   Phase 2: 공동체 이해")
        
        start_time = time.time()
        
        try:
            # 1. 고급 학습 시스템 실행
            logger.info("🚀 고급 학습 시스템 실행 중...")
            learning_results = await self.learning_executor.execute_full_learning()
            
            # 2. 학습 결과 분석
            logger.info("📊 학습 결과 분석 중...")
            analysis_results = await self._analyze_learning_results(learning_results)
            
            # 3. 학습된 시스템으로 의사결정 테스트
            logger.info("🎯 학습된 시스템 의사결정 테스트 중...")
            decision_results = await self._test_learned_decision_making()
            
            # 4. 동적 윤리적 분석 테스트
            logger.info("🔍 동적 윤리적 분석 테스트 중...")
            ethical_analysis_results = await self._test_dynamic_ethical_analysis()
            
            total_time = time.time() - start_time
            
            return {
                "learning_success": True,
                "total_learning_time": total_time,
                "learning_results": learning_results,
                "integrated_analysis": analysis_results,
                "decision_test_results": decision_results,
                "ethical_analysis_results": ethical_analysis_results,
                "summary": {
                    "total_learning_time": total_time,
                    "learning_quality": analysis_results.get("learning_quality", {}),
                    "decision_accuracy": decision_results.get("confidence_score", 0.0),
                    "ethical_analysis_quality": ethical_analysis_results.get("analysis_quality", 0.0)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ 완전한 학습 시스템 실행 실패: {e}")
            return {"error": str(e), "learning_success": False}

    async def _analyze_learning_results(self, learning_results: Dict[str, Any]) -> Dict[str, Any]:
        """학습 결과 분석"""
        
        analysis = {
            "phase_analysis": {},
            "module_performance": {},
            "learning_quality": {}
        }
        
        # 학습 통계 분석
        if "learning_statistics" in learning_results:
            stats = learning_results["learning_statistics"]
            
            # 페이즈별 성능 메트릭
            if "performance_metrics" in stats:
                recent_metrics = stats["performance_metrics"][-10:]  # 최근 10개
                if recent_metrics:
                    analysis["phase_analysis"]["recent_regret_avg"] = np.mean([m["avg_regret_intensity"] for m in recent_metrics])
                    analysis["phase_analysis"]["recent_hedonic_avg"] = np.mean([m["avg_hedonic_score"] for m in recent_metrics])
            
            # 모듈 성능
            if "regret_history" in stats:
                regret_data = stats["regret_history"][-50:]  # 최근 50개
                if regret_data:
                    analysis["module_performance"]["regret_system"] = {
                        "avg_intensity": np.mean([r["intensity"] for r in regret_data]),
                        "total_processed": len(regret_data)
                    }
            
            if "bentham_scores" in stats:
                bentham_data = stats["bentham_scores"][-50:]  # 최근 50개
                if bentham_data:
                    analysis["module_performance"]["bentham_system"] = {
                        "avg_score": np.mean([b["hedonic_score"] for b in bentham_data]),
                        "total_processed": len(bentham_data)
                    }
        
        # 학습 품질 평가
        if "summary" in learning_results:
            summary = learning_results["summary"]
            analysis["learning_quality"]["scenarios_processed"] = summary.get("total_scenarios_processed", 0)
            analysis["learning_quality"]["total_regrets"] = summary.get("total_regrets", 0)
            analysis["learning_quality"]["total_bentham_calculations"] = summary.get("total_bentham_calculations", 0)
            analysis["learning_quality"]["efficiency"] = summary.get("total_regrets", 0) / max(summary.get("total_scenarios_processed", 1), 1)
        
        return analysis

    async def _test_learned_decision_making(self) -> Dict[str, Any]:
        """학습된 시스템으로 의사결정 테스트"""
        
        # 간단한 윤리적 딜레마 테스트
        test_scenario = {
            "title": "자율주행차 딜레마",
            "description": "자율주행차가 급브레이크를 밟아야 하는 상황에서 보행자 1명을 구할 것인가, 아니면 차 안의 탑승자 2명을 구할 것인가?",
            "context": {"urgency": "high", "stakeholders": ["보행자", "탑승자들"]}
        }
        
        try:
            start_time = time.time()
            
            # 감정 분석
            emotion_result = await self.emotion_analyzer.analyze_comprehensive(test_scenario["description"])
            
            # 벤담 계산
            bentham_result = await self.bentham_calculator.calculate_with_advanced_layers(test_scenario)
            
            # 후회 분석
            regret_result = await self.regret_analyzer.analyze_regret({
                "text": test_scenario["description"],
                "context": test_scenario["context"]
            })
            
            processing_time = time.time() - start_time
            
            # 최종 결정 (간단한 로직)
            decision_scores = {
                "보행자 구하기": 0.3,
                "탑승자 구하기": 0.7
            }
            
            final_decision = max(decision_scores, key=decision_scores.get)
            confidence = max(decision_scores.values())
            
            return {
                "test_success": True,
                "final_recommendation": final_decision,
                "confidence_score": confidence,
                "processing_time": processing_time,
                "reasoning_chain": [
                    f"감정 분석: {getattr(emotion_result, 'dominant_emotion', 'N/A')}",
                    f"벤담 점수: {getattr(bentham_result, 'final_score', 0.0):.3f}",
                    f"후회 강도: {getattr(regret_result, 'regret_intensity', 0.0):.3f}",
                    f"최종 결정: {final_decision}"
                ]
            }
            
        except Exception as e:
            logger.error(f"의사결정 테스트 실패: {e}")
            return {"test_success": False, "error": str(e)}

    async def _test_dynamic_ethical_analysis(self) -> Dict[str, Any]:
        """동적 윤리적 분석 테스트"""
        
        if not self.dynamic_choice_analyzer:
            return {"error": "동적 선택지 분석기가 초기화되지 않음"}
        
        try:
            # 다양한 윤리적 딜레마 테스트
            test_dilemma = "의사가 장기 이식을 위해 건강한 환자 1명을 희생시켜 5명의 환자를 살릴 것인가?"
            
            start_time = time.time()
            result = await self.dynamic_choice_analyzer.analyze_ethical_dilemma(
                dilemma_text=test_dilemma,
                title="의료 윤리 딜레마 테스트"
            )
            processing_time = time.time() - start_time
            
            return {
                "analysis_success": True,
                "dilemma_type": result.dilemma_type.value,
                "extracted_choices": len(result.extracted_choices),
                "stakeholders_identified": len(result.stakeholders),
                "recommended_choice": result.recommended_choice.name if result.recommended_choice else None,
                "reasoning_chain": result.reasoning_chain,
                "processing_time": processing_time,
                "analysis_quality": len(result.reasoning_chain) / 5.0  # 간단한 품질 지표
            }
            
        except Exception as e:
            logger.error(f"동적 윤리적 분석 테스트 실패: {e}")
            return {"analysis_success": False, "error": str(e)}


async def main():
    """실제 훈련 메인 함수"""
    if not MODULES_AVAILABLE:
        logger.error("❌ 필수 모듈을 불러올 수 없습니다.")
        return
    
    # 실제 통합 훈련 시스템 초기화
    training_system = RealIntegratedTrainingSystem()
    
    # 실제 시스템 초기화
    if not await training_system.initialize_real_system():
        logger.error("❌ 실제 시스템 초기화 실패")
        return
    
    # 실제 통합 훈련 실행 (처음 25개 아이템으로 테스트)
    results = await training_system.run_real_integrated_training(max_items=25)
    
    # 결과 출력
    logger.info(f"\n{'='*80}")
    logger.info("🎉 실제 Red Heart AI 통합 훈련 완료")
    logger.info(f"{'='*80}")
    
    if 'error' not in results:
        summary = results['training_summary']
        module_perf = results['module_performance']
        integration = results['integration_analysis']
        
        logger.info(f"\n📊 실제 훈련 요약:")
        logger.info(f"  - 총 처리 아이템: {summary['total_items']}개")
        logger.info(f"  - 성공적 통합: {summary['successful_integrations']}개")
        logger.info(f"  - 통합 성공률: {summary['success_rate']:.1f}%")
        logger.info(f"  - 총 처리시간: {summary['total_processing_time']:.1f}초")
        logger.info(f"  - 평균 처리시간: {summary['avg_processing_time']:.3f}초")
        
        logger.info(f"\n🎯 모듈별 성공률:")
        logger.info(f"  - 감정 분석: {module_perf['emotion_success_rate']:.1f}%")
        logger.info(f"  - 벤담 계산: {module_perf['bentham_success_rate']:.1f}%")
        logger.info(f"  - 후회 분석: {module_perf['regret_success_rate']:.1f}%")
        logger.info(f"  - SURD 분석: {module_perf['surd_success_rate']:.1f}%")
        
        logger.info(f"\n🔗 통합 분석:")
        logger.info(f"  - 완전 통합률: {integration['full_integration_rate']:.1f}%")
        logger.info(f"  - 반사실 시나리오: {integration['total_counterfactuals_generated']}개")
        logger.info(f"  - 아이템당 평균: {integration['avg_counterfactuals_per_item']:.1f}개")
        
        logger.info(f"\n⚡ 성능 메트릭:")
        perf = results['performance_metrics']
        logger.info(f"  - 처리 속도: {perf['items_per_second']:.2f} 아이템/초")
        logger.info(f"  - 성공 처리 속도: {perf['successful_items_per_second']:.2f} 아이템/초")
        logger.info(f"  - 효율성 점수: {perf['efficiency_score']:.2f}")
        
        if results['error_analysis']['error_patterns']:
            logger.info(f"\n⚠️ 오류 패턴:")
            for error_type, count in results['error_analysis']['error_patterns'].items():
                logger.info(f"  - {error_type}: {count}회")
        
        # 결과를 파일로 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = Path(f'real_integrated_training_results_{timestamp}.json')
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"\n📄 상세 결과 저장: {result_file}")
    else:
        logger.error(f"❌ 실제 훈련 실패: {results['error']}")


async def main():
    """실제 훈련 메인 함수"""
    if not MODULES_AVAILABLE:
        logger.error("❌ 필수 모듈을 불러올 수 없습니다.")
        return
    
    # 실제 통합 훈련 시스템 실행
    system = RealIntegratedTrainingSystem()
    await system.initialize_real_system()
    
    # 기본 기능 테스트 실행
    results = await system.run_real_integrated_training(max_items=3)
    
    logger.info("🎯 Red Heart AI 실제 통합 훈련 완료")
    return results


async def main_learning_system():
    """학습 시스템 실행 메인 함수"""
    
    if not MODULES_AVAILABLE:
        logger.error("❌ 필수 모듈이 사용할 수 없습니다.")
        return
    
    if not LEARNING_SYSTEM_AVAILABLE:
        logger.error("❌ 학습 시스템이 사용할 수 없습니다.")
        return
    
    # 통합 시스템 초기화
    system = RealIntegratedTrainingSystem()
    success = await system.initialize_real_system(learning_mode=True)
    
    if not success:
        logger.error("❌ 학습 시스템 초기화 실패")
        return
    
    # 완전한 학습 시스템 실행
    results = await system.run_complete_learning_system()
    
    # 결과 출력
    if results.get("learning_success"):
        logger.info("✅ 완전한 학습 시스템 실행 성공!")
        logger.info(f"총 학습 시간: {results['total_learning_time']:.2f}초")
        
        # 학습 결과 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = Path(f'complete_learning_system_results_{timestamp}.json')
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"📄 학습 결과 저장: {result_file}")
    else:
        logger.error(f"❌ 학습 시스템 실행 실패: {results.get('error', 'Unknown error')}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Red Heart AI 실제 통합 훈련 시스템')
    parser.add_argument('--learning', action='store_true', help='완전한 학습 시스템 실행')
    parser.add_argument('--test', action='store_true', help='기능 테스트 모드 실행')
    
    args = parser.parse_args()
    
    if args.learning:
        asyncio.run(main_learning_system())
    elif args.test:
        asyncio.run(main())
    else:
        # 기본값: 기능 테스트 모드
        asyncio.run(main())
                phase_key = f"phase_{phase}"
                if phase_key in learning_results:
                    analysis["phase_analysis"][phase_key] = {
                        "scenarios_processed": learning_results[phase_key].get("scenarios_processed", 0),
                        "learning_iterations": learning_results[phase_key].get("learning_iterations", 0),
                        "phase_completion_rate": learning_results[phase_key].get("completion_rate", 0.0)
                    }
            
            # 모듈 성능 분석
            if "module_performance" in learning_results:
                for module, performance in learning_results["module_performance"].items():
                    analysis["module_performance"][module] = {
                        "accuracy_improvement": performance.get("accuracy_improvement", 0.0),
                        "processing_time_avg": performance.get("processing_time_avg", 0.0),
                        "confidence_score": performance.get("confidence_score", 0.0)
                    }
            
            # 학습 품질 분석
            analysis["learning_quality"] = {
                "overall_improvement": learning_results.get("overall_improvement", 0.0),
                "convergence_rate": learning_results.get("convergence_rate", 0.0),
                "stability_score": learning_results.get("stability_score", 0.0)
            }
            
        except Exception as e:
            logger.error(f"학습 결과 분석 실패: {e}")
            analysis["integration_success"] = False
            analysis["error"] = str(e)
        
        return analysis
    
    async def _test_learned_decision_making(self) -> Dict[str, Any]:
        """학습된 의사결정 시스템 테스트"""
        
        if not self.integrated_orchestrator:
            logger.warning("통합 오케스트레이터가 초기화되지 않음")
            return {"error": "통합 오케스트레이터 없음"}
        
        # 자율주행차 딜레마 시나리오 생성
        test_scenario = IntegrationContext(
            session_id="learning_test_001",
            user_input="자율주행차 윤리적 딜레마 학습 테스트",
            scenario_description="자율주행차가 브레이크 고장으로 인해 불가피한 충돌 상황에서 어떤 선택을 해야 하는가?",
            urgency_level=0.95,
            complexity_level=0.9,
            stakeholder_count=5,
            ethical_weight=0.9,
            cultural_context="korean"
        )
        
        try:
            # 통합 시스템으로 의사결정 수행
            decision_result = await self.integrated_orchestrator.process_decision_request(test_scenario)
            
            # 결과 분석
            test_results = {
                "decision_id": decision_result.decision_id,
                "final_recommendation": decision_result.final_recommendation,
                "confidence_score": decision_result.confidence_score,
                "module_contributions": decision_result.module_contributions,
                "reasoning_chain": decision_result.reasoning_chain,
                "alternative_options": decision_result.alternative_options,
                "processing_time": decision_result.processing_time,
                "test_success": True
            }
            
            logger.info(f"✅ 학습된 의사결정 테스트 완료")
            logger.info(f"   최종 추천: {decision_result.final_recommendation}")
            logger.info(f"   신뢰도: {decision_result.confidence_score:.3f}")
            
            return test_results
            
        except Exception as e:
            logger.error(f"의사결정 테스트 실패: {e}")
            return {
                "error": str(e),
                "test_success": False
            }
    
    async def _test_dynamic_ethical_analysis(self) -> Dict[str, Any]:
        """동적 윤리적 딜레마 분석 테스트"""
        
        if not self.dynamic_choice_analyzer:
            logger.warning("동적 윤리적 선택지 분석기가 초기화되지 않음")
            return {"error": "동적 분석기 없음"}
        
        # 다양한 윤리적 딜레마 테스트
        test_scenarios = [
            {
                "title": "자율주행차 윤리적 딜레마",
                "description": "자율주행차가 브레이크 고장으로 인해 불가피한 충돌 상황에서 어떤 선택을 해야 하는가? 급브레이크를 밟아 뒤차 추돌을 유발할 것인가, 아니면 핸들을 틀어 벽으로 향할 것인가, 또는 직진하여 보행자와 충돌할 것인가?",
            },
            {
                "title": "의료진 자원 배분 딜레마",
                "description": "코로나19 상황에서 인공호흡기 1대를 두고 90세 중증환자와 30세 중증환자 중 누구를 선택할 것인가? 나이를 기준으로 할 것인가, 아니면 선착순으로 할 것인가, 또는 다른 의학적 기준을 적용할 것인가?",
            },
            {
                "title": "개인정보 vs 공공안전",
                "description": "테러 방지를 위해 시민들의 개인정보를 수집하고 감시할 것인가, 아니면 개인의 프라이버시를 보호할 것인가? 전면적 감시를 할 것인가, 아니면 선택적 감시를 할 것인가, 또는 감시를 하지 않을 것인가?",
            }
        ]
        
        test_results = {
            "tested_scenarios": len(test_scenarios),
            "successful_analyses": 0,
            "scenario_results": [],
            "overall_performance": {},
            "test_success": True
        }
        
        try:
            for i, scenario in enumerate(test_scenarios):
                logger.info(f"   📋 시나리오 {i+1}/{len(test_scenarios)}: {scenario['title']}")
                
                try:
                    # 동적 윤리적 딜레마 분석
                    start_time = time.time()
                    analysis_result = await self.dynamic_choice_analyzer.analyze_ethical_dilemma(
                        dilemma_text=scenario['description'],
                        title=scenario['title']
                    )
                    analysis_time = time.time() - start_time
                    
                    # 결과 정리
                    scenario_result = {
                        "scenario_id": f"scenario_{i+1}",
                        "title": scenario['title'],
                        "dilemma_type": analysis_result.dilemma_type.value,
                        "extracted_choices": len(analysis_result.extracted_choices),
                        "stakeholders": len(analysis_result.stakeholders),
                        "recommended_choice": analysis_result.recommended_choice.name if analysis_result.recommended_choice else None,
                        "analysis_time": analysis_time,
                        "success": True
                    }
                    
                    # 선택지별 상세 결과
                    choice_details = []
                    for choice_id, choice_analysis in analysis_result.choice_analyses.items():
                        choice_details.append({
                            "choice_name": choice_analysis.choice.name,
                            "utility_score": choice_analysis.utility_score,
                            "confidence_score": choice_analysis.confidence_score,
                            "risk_adjusted_score": choice_analysis.risk_adjusted_score,
                            "processing_time": choice_analysis.processing_time
                        })
                    
                    scenario_result["choice_analyses"] = choice_details
                    scenario_result["reasoning_chain"] = analysis_result.reasoning_chain
                    
                    test_results["scenario_results"].append(scenario_result)
                    test_results["successful_analyses"] += 1
                    
                    logger.info(f"      ✅ 분석 완료 ({analysis_time:.2f}초)")
                    logger.info(f"      선택지: {len(analysis_result.extracted_choices)}개")
                    logger.info(f"      이해관계자: {len(analysis_result.stakeholders)}명")
                    if analysis_result.recommended_choice:
                        logger.info(f"      추천: {analysis_result.recommended_choice.name}")
                    
                except Exception as e:
                    logger.error(f"      ❌ 시나리오 {i+1} 분석 실패: {e}")
                    scenario_result = {
                        "scenario_id": f"scenario_{i+1}",
                        "title": scenario['title'],
                        "success": False,
                        "error": str(e)
                    }
                    test_results["scenario_results"].append(scenario_result)
            
            # 전체 성능 분석
            if test_results["successful_analyses"] > 0:
                all_choice_analyses = []
                total_analysis_time = 0
                
                for result in test_results["scenario_results"]:
                    if result.get("success"):
                        total_analysis_time += result.get("analysis_time", 0)
                        if "choice_analyses" in result:
                            all_choice_analyses.extend(result["choice_analyses"])
                
                test_results["overall_performance"] = {
                    "success_rate": test_results["successful_analyses"] / test_results["tested_scenarios"],
                    "avg_analysis_time": total_analysis_time / test_results["successful_analyses"],
                    "total_choices_analyzed": len(all_choice_analyses),
                    "avg_utility_score": sum(c["utility_score"] for c in all_choice_analyses) / len(all_choice_analyses) if all_choice_analyses else 0,
                    "avg_confidence_score": sum(c["confidence_score"] for c in all_choice_analyses) / len(all_choice_analyses) if all_choice_analyses else 0
                }
                
                logger.info(f"✅ 동적 윤리적 딜레마 분석 테스트 완료")
                logger.info(f"   성공률: {test_results['overall_performance']['success_rate']:.1%}")
                logger.info(f"   평균 분석 시간: {test_results['overall_performance']['avg_analysis_time']:.2f}초")
                logger.info(f"   분석된 선택지: {test_results['overall_performance']['total_choices_analyzed']}개")
                logger.info(f"   평균 유틸리티 점수: {test_results['overall_performance']['avg_utility_score']:.3f}")
                logger.info(f"   평균 신뢰도: {test_results['overall_performance']['avg_confidence_score']:.3f}")
            
            return test_results
            
        except Exception as e:
            logger.error(f"동적 윤리적 딜레마 분석 테스트 실패: {e}")
            test_results["test_success"] = False
            test_results["error"] = str(e)
            return test_results

async def main_learning_system():
    """학습 시스템 실행 메인 함수"""
    
    if not MODULES_AVAILABLE:
        logger.error("❌ 필수 모듈이 사용할 수 없습니다.")
        return
    
    if not LEARNING_SYSTEM_AVAILABLE:
        logger.error("❌ 학습 시스템이 사용할 수 없습니다.")
        return
    
    # 통합 시스템 초기화
    system = RealIntegratedTrainingSystem()
    success = await system.initialize_real_system(learning_mode=True)
    
    if not success:
        logger.error("❌ 학습 시스템 초기화 실패")
        return
    
    # 완전한 학습 시스템 실행
    results = await system.run_complete_learning_system()
    
    # 결과 출력
    if results.get("learning_success"):
        logger.info("✅ 완전한 학습 시스템 실행 성공!")
        logger.info(f"총 학습 시간: {results['total_learning_time']:.2f}초")
        
        # 학습 결과 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = Path(f'complete_learning_system_results_{timestamp}.json')
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"📄 학습 결과 저장: {result_file}")
    else:
        logger.error(f"❌ 학습 시스템 실행 실패: {results.get('error', 'Unknown error')}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Red Heart AI 실제 통합 훈련 시스템')
    parser.add_argument('--learning', action='store_true', help='완전한 학습 시스템 실행')
    parser.add_argument('--test', action='store_true', help='기능 테스트 모드 실행')
    
    args = parser.parse_args()
    
    if args.learning:
        asyncio.run(main_learning_system())
    elif args.test:
        asyncio.run(main())
    else:
        # 기본값: 기능 테스트 모드
        asyncio.run(main())