"""
통합된 Red Heart 시스템 GPU 학습 테스트
Integrated Red Heart System GPU Learning Test

현재 통합된 모든 모듈을 사용하여 실제 데이터로 학습 및 테스트 진행
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import asyncio

# 현재 프로젝트의 통합된 모듈들 임포트
try:
    from config import DEVICE, ADVANCED_CONFIG, DATA_DIR
    from data_models import EmotionData, EmotionState, EmotionIntensity
    from emotion_ethics_regret_circuit import EmotionEthicsRegretCircuit, CircuitDecisionContext
    from ethics_policy_updater import EthicsPolicyUpdater, EthicsExperience
    from phase_controller import PhaseController, PhaseDecisionContext, Phase
    from xai_feedback_integrator import XAIFeedbackIntegrator, XAIInterpretation
    from fuzzy_emotion_ethics_mapper import FuzzyEmotionEthicsMapper
    from deep_multi_dimensional_ethics_system import DeepMultiDimensionalEthicsSystem, EthicalDilemma, StakeholderPerspective
    from temporal_event_propagation_analyzer import TemporalEventPropagationAnalyzer, TemporalEvent
    from integrated_system_orchestrator import IntegratedSystemOrchestrator, IntegrationContext
    
    # 업그레이드된 시스템 임포트
    from dynamic_gpu_manager import get_gpu_manager, allocate_gpu_memory, optimize_gpu_for_learning
    from robust_logging_system import get_robust_logger, test_session, add_performance_sample
    
    print("✅ 모든 통합 모듈 임포트 성공")
    print("✅ 동적 GPU 관리자 및 견고한 로깅 시스템 연동 완료")
except ImportError as e:
    print(f"❌ 모듈 임포트 실패: {e}")
    sys.exit(1)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('IntegratedLearningTest')

@dataclass
class LearningConfig:
    """학습 설정"""
    batch_size: int = 8
    learning_rate: float = 0.001
    num_epochs: int = 10
    device: str = str(DEVICE)
    use_gpu: bool = ADVANCED_CONFIG.get('enable_gpu', False)
    validation_split: float = 0.2
    early_stopping_patience: int = 3

@dataclass
class TestResults:
    """테스트 결과"""
    accuracy: float
    loss: float
    prediction_error: float
    processing_time: float
    gpu_utilization: float
    memory_usage: float

class IntegratedLearningFramework:
    """통합 학습 프레임워크 - 업그레이드된 GPU 관리 및 로깅"""
    
    def __init__(self, config: LearningConfig):
        self.config = config
        self.logger = logger
        
        # 업그레이드된 GPU 관리자 및 로깅 시스템 연동
        self.gpu_manager = get_gpu_manager()
        self.robust_logger = get_robust_logger()
        
        # GPU 상태 확인 및 최적화
        self.device = torch.device(config.device)
        
        if config.use_gpu and torch.cuda.is_available():
            # GPU 학습 최적화 활성화
            optimization_success = optimize_gpu_for_learning()
            
            gpu_status = self.gpu_manager.get_memory_status()
            self.logger.info(f"🚀 GPU 사용: {torch.cuda.get_device_name()}")
            self.logger.info(f"💾 총 GPU 메모리: {gpu_status['total_gb']:.1f}GB")
            self.logger.info(f"💾 사용 가능 메모리: {gpu_status['available_gb']:.1f}GB")
            self.logger.info(f"⚡ 학습 최적화: {'활성화' if optimization_success else '제한됨'}")
            
            self.robust_logger.log("INFO", "GPU_Manager", 
                                 f"GPU 학습 환경 초기화 완료 - 최적화: {optimization_success}",
                                 {"gpu_status": gpu_status})
        else:
            self.logger.info("💻 CPU 모드로 실행")
            self.robust_logger.log("INFO", "GPU_Manager", "CPU 모드로 실행")
        
        # 통합된 시스템 컴포넌트들 초기화
        self.initialize_components()
        
        # 학습 데이터 저장소
        self.training_data = []
        self.validation_data = []
        
        # 성능 추적 - 향상된 추적
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'gpu_memory': [],
            'processing_time': [],
            'gpu_utilization': [],
            'memory_efficiency': []
        }
    
    def initialize_components(self):
        """통합 시스템 컴포넌트 초기화"""
        self.logger.info("🏗️ 통합 시스템 컴포넌트 초기화...")
        
        try:
            # 핵심 회로
            self.emotion_circuit = EmotionEthicsRegretCircuit()
            self.logger.info("✅ 감정-윤리-후회 삼각회로 초기화")
            
            # 정책 업데이터
            self.policy_updater = EthicsPolicyUpdater()
            self.logger.info("✅ 윤리 정책 업데이터 초기화")
            
            # 페이즈 컨트롤러
            self.phase_controller = PhaseController()
            self.logger.info("✅ 페이즈 컨트롤러 초기화")
            
            # XAI 피드백 통합기
            self.xai_integrator = XAIFeedbackIntegrator()
            self.logger.info("✅ XAI 피드백 통합기 초기화")
            
            # 퍼지 매핑
            self.fuzzy_mapper = FuzzyEmotionEthicsMapper()
            self.logger.info("✅ 퍼지 감정-윤리 매핑 초기화")
            
            # 다차원 윤리 시스템
            self.ethics_system = DeepMultiDimensionalEthicsSystem()
            self.logger.info("✅ 다차원 윤리 시스템 초기화")
            
            # 시계열 분석기
            self.temporal_analyzer = TemporalEventPropagationAnalyzer()
            self.logger.info("✅ 시계열 사건 전파 분석기 초기화")
            
            # 통합 오케스트레이터
            self.orchestrator = IntegratedSystemOrchestrator()
            self.logger.info("✅ 통합 시스템 오케스트레이터 초기화")
            
            self.logger.info("🎯 모든 컴포넌트 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 컴포넌트 초기화 실패: {e}")
            raise
    
    def load_training_data(self, data_dir: Path = None) -> int:
        """학습 데이터 로드"""
        if data_dir is None:
            data_dir = DATA_DIR / 'decision_logs'
        
        self.logger.info(f"📂 데이터 로딩 시작: {data_dir}")
        
        json_files = list(data_dir.glob("*.json"))
        self.logger.info(f"📄 발견된 JSON 파일: {len(json_files)}개")
        
        loaded_count = 0
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 데이터 검증 및 변환
                if self.validate_data_format(data):
                    processed_data = self.preprocess_data(data)
                    self.training_data.append(processed_data)
                    loaded_count += 1
                    
            except Exception as e:
                self.logger.warning(f"⚠️ 파일 로딩 실패 {json_file.name}: {e}")
        
        # 학습/검증 분할
        if self.training_data:
            split_idx = int(len(self.training_data) * (1 - self.config.validation_split))
            self.validation_data = self.training_data[split_idx:]
            self.training_data = self.training_data[:split_idx]
        
        self.logger.info(f"✅ 데이터 로딩 완료: 학습 {len(self.training_data)}개, 검증 {len(self.validation_data)}개")
        return loaded_count
    
    def validate_data_format(self, data: Dict) -> bool:
        """데이터 형식 검증"""
        required_fields = ['situation', 'decision', 'actual_outcome']
        return all(field in data for field in required_fields)
    
    def preprocess_data(self, raw_data: Dict) -> Dict:
        """데이터 전처리"""
        situation = raw_data['situation']
        decision = raw_data['decision']
        outcome = raw_data['actual_outcome']
        
        # 감정 데이터 변환
        emotion_data = EmotionData(
            primary_emotion=EmotionState[raw_data['emotions']['primary_emotion']],
            intensity=EmotionIntensity[raw_data['emotions']['intensity']],
            valence=raw_data['emotions']['valence'],
            arousal=raw_data['emotions']['arousal'],
            confidence=raw_data['emotions']['confidence']
        )
        
        # 이해관계자 추출
        stakeholders = []
        for person in situation['context']['people_involved']:
            stakeholder = StakeholderPerspective(
                stakeholder_id=person.replace(' ', '_'),
                name=person,
                role='participant',
                power_level=0.5,
                vulnerability=0.5
            )
            stakeholders.append(stakeholder)
        
        # 윤리적 딜레마 구성
        ethical_dilemma = EthicalDilemma(
            dilemma_id=situation['id'],
            scenario=situation['description'],
            context=situation['context']['location'],
            stakeholders=stakeholders,
            available_options=[opt['text'] for opt in situation['options']]
        )
        
        return {
            'input': {
                'emotion': emotion_data,
                'dilemma': ethical_dilemma,
                'situation': situation,
                'hedonic_values': raw_data['hedonic_values']
            },
            'target': {
                'chosen_option': decision['choice'],
                'hedonic_prediction': decision['predicted_outcome']['hedonic_value'],
                'actual_hedonic': outcome['hedonic_value'],
                'actual_emotion': EmotionState[outcome['primary_emotion']],
                'regret': raw_data.get('regret_data', {}).get('intensity', 0.0)
            }
        }
    
    async def train_integrated_system(self) -> Dict[str, Any]:
        """통합 시스템 학습"""
        self.logger.info("🚀 통합 시스템 학습 시작")
        
        start_time = time.time()
        best_loss = float('inf')
        patience_counter = 0
        
        # 학습 루프
        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()
            
            # GPU 메모리 모니터링
            if self.config.use_gpu:
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated()
            
            # 배치 학습
            train_loss, train_accuracy = await self.train_epoch(epoch)
            val_loss, val_accuracy = await self.validate_epoch(epoch)
            
            # GPU 메모리 사용량 기록
            if self.config.use_gpu:
                memory_used = torch.cuda.memory_allocated() - initial_memory
                self.training_history['gpu_memory'].append(memory_used / 1e6)  # MB
            
            epoch_time = time.time() - epoch_start
            
            # 히스토리 기록
            self.training_history['loss'].append(train_loss)
            self.training_history['accuracy'].append(train_accuracy)
            self.training_history['processing_time'].append(epoch_time)
            
            self.logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    self.logger.info(f"🛑 Early stopping at epoch {epoch+1}")
                    break
        
        total_time = time.time() - start_time
        
        # 학습 결과 요약
        results = {
            'total_training_time': total_time,
            'final_train_loss': train_loss,
            'final_train_accuracy': train_accuracy,
            'final_val_loss': val_loss,
            'final_val_accuracy': val_accuracy,
            'best_loss': best_loss,
            'epochs_completed': epoch + 1,
            'training_history': self.training_history
        }
        
        self.logger.info(f"✅ 학습 완료 - 총 시간: {total_time:.2f}초, 최종 정확도: {train_accuracy:.4f}")
        return results
    
    async def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """에포크 학습"""
        total_loss = 0.0
        total_accuracy = 0.0
        batch_count = 0
        
        # 배치 처리
        for i in range(0, len(self.training_data), self.config.batch_size):
            batch_data = self.training_data[i:i + self.config.batch_size]
            
            batch_loss, batch_acc = await self.process_batch(batch_data, training=True)
            
            total_loss += batch_loss
            total_accuracy += batch_acc
            batch_count += 1
        
        return total_loss / batch_count, total_accuracy / batch_count
    
    async def validate_epoch(self, epoch: int) -> Tuple[float, float]:
        """에포크 검증"""
        if not self.validation_data:
            return 0.0, 0.0
        
        total_loss = 0.0
        total_accuracy = 0.0
        batch_count = 0
        
        # 검증 배치 처리
        for i in range(0, len(self.validation_data), self.config.batch_size):
            batch_data = self.validation_data[i:i + self.config.batch_size]
            
            batch_loss, batch_acc = await self.process_batch(batch_data, training=False)
            
            total_loss += batch_loss
            total_accuracy += batch_acc
            batch_count += 1
        
        return total_loss / batch_count, total_accuracy / batch_count
    
    async def process_batch(self, batch_data: List[Dict], training: bool = True) -> Tuple[float, float]:
        """배치 처리"""
        batch_losses = []
        batch_accuracies = []
        
        for data_point in batch_data:
            try:
                # 통합 시스템으로 예측 수행
                prediction = await self.run_integrated_prediction(data_point['input'])
                
                # 손실 계산
                loss = self.calculate_loss(prediction, data_point['target'])
                
                # 정확도 계산
                accuracy = self.calculate_accuracy(prediction, data_point['target'])
                
                batch_losses.append(loss)
                batch_accuracies.append(accuracy)
                
                # 학습 단계에서 시스템 업데이트
                if training:
                    await self.update_system_components(data_point, prediction)
                
            except Exception as e:
                self.logger.warning(f"⚠️ 배치 처리 중 오류: {e}")
                continue
        
        return (
            np.mean(batch_losses) if batch_losses else 0.0,
            np.mean(batch_accuracies) if batch_accuracies else 0.0
        )
    
    async def run_integrated_prediction(self, input_data: Dict) -> Dict:
        """통합 시스템 예측 실행"""
        
        # 1. 감정-윤리-후회 회로 처리
        circuit_context = CircuitDecisionContext(
            scenario_text=input_data['dilemma'].scenario,
            proposed_action="윤리적 의사결정",
            self_emotion=input_data['emotion'],
            stakeholders=[s.name for s in input_data['dilemma'].stakeholders]
        )
        
        circuit_result = await self.emotion_circuit.process_ethical_decision(circuit_context)
        
        # 2. 다차원 윤리 분석
        ethics_result = self.ethics_system.comprehensive_ethical_analysis(input_data['dilemma'])
        
        # 3. 퍼지 감정-윤리 매핑
        fuzzy_result = self.fuzzy_mapper.map_emotion_to_ethics(input_data['emotion'])
        
        # 4. 시계열 이벤트 등록 및 예측
        temporal_event = TemporalEvent(
            event_id=f"decision_{int(time.time())}",
            timestamp=time.time(),
            event_type="ethical_decision",
            description=input_data['dilemma'].scenario,
            emotion_state=input_data['emotion']
        )
        self.temporal_analyzer.register_event(temporal_event)
        
        # 통합 예측 결과
        prediction = {
            'ethical_score': circuit_result.final_ethical_score,
            'confidence': circuit_result.confidence,
            'ethics_weights': ethics_result.school_reasonings,
            'fuzzy_mapping': fuzzy_result.ethics_weights,
            'predicted_regret': circuit_result.predicted_regret.get('anticipated_regret', 0.0),
            'temporal_prediction': 0.5  # 간단화된 예측값
        }
        
        return prediction
    
    def calculate_loss(self, prediction: Dict, target: Dict) -> float:
        """손실 계산"""
        # 다중 목표 손실 함수
        ethical_score_loss = abs(prediction['ethical_score'] - target['actual_hedonic'])
        regret_loss = abs(prediction['predicted_regret'] - target['regret'])
        
        # 가중 평균
        total_loss = ethical_score_loss * 0.7 + regret_loss * 0.3
        return total_loss
    
    def calculate_accuracy(self, prediction: Dict, target: Dict) -> float:
        """정확도 계산"""
        # 윤리적 점수 예측 정확도
        score_accuracy = 1.0 - abs(prediction['ethical_score'] - target['actual_hedonic'])
        
        # 후회 예측 정확도
        regret_accuracy = 1.0 - abs(prediction['predicted_regret'] - target['regret'])
        
        # 평균 정확도
        return (score_accuracy + regret_accuracy) / 2.0
    
    async def update_system_components(self, data_point: Dict, prediction: Dict):
        """시스템 컴포넌트 업데이트"""
        try:
            # 윤리 정책 업데이트
            experience = EthicsExperience(
                experience_id=f"exp_{int(time.time())}",
                scenario=data_point['input']['dilemma'].scenario,
                decision_made=data_point['target']['chosen_option'],
                outcome_rating=data_point['target']['actual_hedonic'],
                emotion_state=data_point['input']['emotion'],
                stakeholders=[s.name for s in data_point['input']['dilemma'].stakeholders],
                cultural_context="korean",
                decision_urgency=0.5,
                actual_regret=data_point['target']['regret'],
                user_satisfaction=0.7,
                moral_correctness=0.8
            )
            
            self.policy_updater.add_experience(experience)
            
            # XAI 피드백 통합
            xai_interpretation = XAIInterpretation(
                interpretation_id=f"interp_{int(time.time())}",
                decision_id=f"decision_{int(time.time())}",
                feature_importance={
                    'ethical_score': prediction['ethical_score'],
                    'confidence': prediction['confidence']
                },
                explanation_confidence=prediction['confidence']
            )
            
            # 시스템 컴포넌트들
            system_components = {
                'emotion_circuit': self.emotion_circuit,
                'policy_updater': self.policy_updater,
                'fuzzy_mapper': self.fuzzy_mapper
            }
            
            self.xai_integrator.integrate_xai_feedback(xai_interpretation, system_components)
            
        except Exception as e:
            self.logger.warning(f"⚠️ 컴포넌트 업데이트 실패: {e}")
    
    def get_system_analytics(self) -> Dict[str, Any]:
        """시스템 분석 정보"""
        return {
            'emotion_circuit_status': self.emotion_circuit.get_circuit_status(),
            'policy_analytics': self.policy_updater.get_analytics(),
            'phase_analytics': self.phase_controller.get_analytics(),
            'xai_analytics': self.xai_integrator.get_feedback_analytics(),
            'fuzzy_analytics': self.fuzzy_mapper.get_mapping_analytics(),
            'ethics_analytics': self.ethics_system.get_ethics_analytics(),
            'temporal_analytics': self.temporal_analyzer.get_analytics_dashboard(),
            'training_history': self.training_history
        }

async def run_learning_test():
    """학습 테스트 실행"""
    logger.info("🎯 통합 Red Heart 시스템 GPU 학습 테스트 시작")
    
    # 설정
    config = LearningConfig(
        batch_size=4,  # 데이터가 적으므로 작은 배치
        learning_rate=0.001,
        num_epochs=10,
        use_gpu=ADVANCED_CONFIG.get('enable_gpu', False)
    )
    
    # 학습 프레임워크 초기화
    framework = IntegratedLearningFramework(config)
    
    # 데이터 로드
    data_count = framework.load_training_data()
    if data_count == 0:
        logger.error("❌ 학습 데이터가 없습니다.")
        return
    
    logger.info(f"📊 로드된 데이터: {data_count}개")
    
    # 학습 실행
    results = await framework.train_integrated_system()
    
    # 결과 출력
    logger.info("📈 학습 결과:")
    logger.info(f"- 총 학습 시간: {results['total_training_time']:.2f}초")
    logger.info(f"- 최종 정확도: {results['final_train_accuracy']:.4f}")
    logger.info(f"- 최종 손실: {results['final_train_loss']:.4f}")
    logger.info(f"- 완료된 에포크: {results['epochs_completed']}")
    
    # 시스템 분석
    analytics = framework.get_system_analytics()
    logger.info("🔍 시스템 분석:")
    
    for component, data in analytics.items():
        if isinstance(data, dict) and 'total_decisions' in data:
            logger.info(f"- {component}: {data.get('total_decisions', 0)}개 결정")
        elif isinstance(data, dict) and 'total_mappings' in data:
            logger.info(f"- {component}: {data.get('total_mappings', 0)}개 매핑")
    
    # GPU 사용량 정보
    if config.use_gpu and torch.cuda.is_available():
        max_memory = max(results['training_history']['gpu_memory']) if results['training_history']['gpu_memory'] else 0
        logger.info(f"🔥 최대 GPU 메모리 사용량: {max_memory:.1f} MB")
        logger.info(f"⚡ GPU 활용률: 높음" if max_memory > 100 else "⚡ GPU 활용률: 보통")
    
    return results, analytics

if __name__ == "__main__":
    # 비동기 실행
    results = asyncio.run(run_learning_test())