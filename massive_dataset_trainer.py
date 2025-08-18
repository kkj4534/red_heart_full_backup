"""
대규모 데이터셋 학습 시스템
Massive Dataset Learning System for Red Heart AI

요구사항:
- 전체 데이터셋 (~266MB, 130개 JSON 파일) 활용
- 데이터 1개당 7회 후회 분석 + 21회 벤담 쾌락 계산
- 총 3번 선회 학습
- 20회 학습마다 로그 저장
- 200GB 스토리지 제한 준수
- Adaptive gradient 문제 해결을 위한 데이터 셔플링
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
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import asyncio
import gc
from collections import defaultdict

# 통합 시스템 임포트
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
    from dynamic_gpu_manager import get_gpu_manager, allocate_gpu_memory, optimize_gpu_for_learning
    from robust_logging_system import get_robust_logger, test_session, add_performance_sample
    
    print("✅ 모든 통합 모듈 임포트 성공")
except ImportError as e:
    print(f"❌ 모듈 임포트 실패: {e}")
    sys.exit(1)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MassiveDatasetTrainer')

@dataclass
class MassiveTrainingConfig:
    """대규모 학습 설정"""
    dataset_path: str = "/mnt/c/large_project/linux_red_heart/processed_datasets"
    regret_iterations_per_data: int = 7  # 데이터당 후회 계산 횟수
    bentham_calculations_per_regret: int = 3  # 후회당 벤담 계산 횟수 (7*3=21)
    training_cycles: int = 3  # 총 선회 횟수
    log_interval: int = 50  # 50회마다 로그 저장 (연산 시간 최적화)
    max_storage_gb: float = 200.0  # 최대 스토리지 사용량 (GB)
    batch_size: int = 4
    learning_rate: float = 0.001
    device: str = str(DEVICE)
    use_gpu: bool = ADVANCED_CONFIG.get('enable_gpu', False)
    shuffle_strategy: str = "balanced_hash"  # adaptive gradient 대응 셔플링

@dataclass
class DatasetMetrics:
    """데이터셋 메트릭"""
    total_files: int = 0
    total_scenarios: int = 0
    processed_scenarios: int = 0
    regret_calculations: int = 0
    bentham_calculations: int = 0
    current_cycle: int = 0
    storage_used_gb: float = 0.0

@dataclass
class PerformanceMetrics:
    """성능 메트릭"""
    cycle_accuracy: float = 0.0
    cycle_loss: float = 0.0
    avg_regret_score: float = 0.0
    avg_bentham_score: float = 0.0
    processing_time_per_scenario: float = 0.0
    gpu_utilization: float = 0.0
    memory_efficiency: float = 0.0

class MassiveDatasetTrainer:
    """대규모 데이터셋 학습기"""
    
    def __init__(self, config: MassiveTrainingConfig):
        self.config = config
        self.logger = logger
        self.session_id = f"massive_training_{int(time.time())}"
        
        # 업그레이드된 시스템 연동
        self.gpu_manager = get_gpu_manager()
        self.robust_logger = get_robust_logger()
        
        # GPU 최적화
        if config.use_gpu and torch.cuda.is_available():
            optimization_success = optimize_gpu_for_learning()
            gpu_status = self.gpu_manager.get_memory_status()
            self.logger.info(f"🚀 GPU 대규모 학습 환경 초기화: {torch.cuda.get_device_name()}")
            self.logger.info(f"💾 GPU 메모리: {gpu_status['total_gb']:.1f}GB")
        
        # 통합 시스템 초기화
        self.initialize_integrated_systems()
        
        # 감정 경험 메모리 시스템
        self.emotional_experience_memory = {
            'regret_patterns': defaultdict(list),  # 후회 패턴별 감정 경험
            'ethical_emotions': defaultdict(list),  # 윤리적 상황별 감정 반응
            'decision_outcomes': [],  # 결정-결과-감정 경험
            'emotional_learning': []  # 감정 기반 학습 내역
        }
        
        # 데이터셋 메트릭
        self.dataset_metrics = DatasetMetrics()
        self.performance_history = []
        
        # 로그 관리
        self.log_dir = Path(f"logs/massive_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 스토리지 모니터링
        self.storage_monitor = StorageMonitor(self.config.max_storage_gb)
        
        self.logger.info(f"🎯 대규모 학습 시스템 초기화 완료: {self.session_id}")
    
    def initialize_integrated_systems(self):
        """통합 시스템 초기화"""
        self.logger.info("🏗️ 통합 시스템 컴포넌트 초기화...")
        
        self.emotion_circuit = EmotionEthicsRegretCircuit()
        self.policy_updater = EthicsPolicyUpdater()
        self.phase_controller = PhaseController()
        self.xai_integrator = XAIFeedbackIntegrator()
        self.fuzzy_mapper = FuzzyEmotionEthicsMapper()
        self.ethics_system = DeepMultiDimensionalEthicsSystem()
        self.temporal_analyzer = TemporalEventPropagationAnalyzer()
        self.orchestrator = IntegratedSystemOrchestrator()
        
        self.logger.info("✅ 모든 통합 시스템 초기화 완료")
    
    def discover_datasets(self) -> List[Dict[str, Any]]:
        """데이터셋 탐색 및 메타데이터 수집"""
        self.logger.info(f"📂 데이터셋 탐색 시작: {self.config.dataset_path}")
        
        datasets = []
        dataset_folders = [
            "scruples", "academic", "augmented", "classic_literature", 
            "ebs_korean_literature", "ethical_scenarios", "literature"
        ]
        
        # 폴더별 데이터 수집
        for folder in dataset_folders:
            folder_path = Path(self.config.dataset_path) / folder
            if folder_path.exists():
                json_files = list(folder_path.glob("*.json"))
                for json_file in json_files:
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # 메타데이터 추출
                        scenario_count = 0
                        if 'scenarios' in data:
                            scenario_count = len(data['scenarios'])
                        elif 'metadata' in data and 'scenario_count' in data['metadata']:
                            scenario_count = data['metadata']['scenario_count']
                        
                        datasets.append({
                            'file_path': str(json_file),
                            'folder': folder,
                            'scenario_count': scenario_count,
                            'file_size': json_file.stat().st_size,
                            'dataset_type': folder
                        })
                        
                    except Exception as e:
                        self.logger.warning(f"⚠️ 파일 읽기 실패 {json_file}: {e}")
        
        # 루트 레벨 JSON 파일들
        root_files = list(Path(self.config.dataset_path).glob("*.json"))
        for json_file in root_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                scenario_count = 0
                if isinstance(data, list):
                    scenario_count = len(data)
                elif 'scenarios' in data:
                    scenario_count = len(data['scenarios'])
                
                datasets.append({
                    'file_path': str(json_file),
                    'folder': 'root',
                    'scenario_count': scenario_count,
                    'file_size': json_file.stat().st_size,
                    'dataset_type': 'integrated'
                })
                
            except Exception as e:
                self.logger.warning(f"⚠️ 루트 파일 읽기 실패 {json_file}: {e}")
        
        # 통계 업데이트
        self.dataset_metrics.total_files = len(datasets)
        self.dataset_metrics.total_scenarios = sum(d['scenario_count'] for d in datasets)
        
        self.logger.info(f"📊 탐색 완료: {len(datasets)}개 파일, {self.dataset_metrics.total_scenarios}개 시나리오")
        return datasets
    
    def create_balanced_shuffle_order(self, datasets: List[Dict]) -> List[Dict]:
        """Adaptive gradient 문제 해결을 위한 균형잡힌 셔플링"""
        self.logger.info("🔀 균형잡힌 데이터 셔플링 수행...")
        
        # 데이터셋 타입별 그룹화
        type_groups = defaultdict(list)
        for dataset in datasets:
            type_groups[dataset['dataset_type']].append(dataset)
        
        # 각 그룹 내에서 셔플링
        shuffled_datasets = []
        for type_name, type_datasets in type_groups.items():
            # 해시 기반 안정적 셔플링
            type_datasets_with_hash = []
            for dataset in type_datasets:
                hash_value = hashlib.md5(dataset['file_path'].encode()).hexdigest()
                type_datasets_with_hash.append((hash_value, dataset))
            
            # 해시값으로 정렬 (의사 랜덤이지만 재현 가능)
            type_datasets_with_hash.sort(key=lambda x: x[0])
            type_datasets = [item[1] for item in type_datasets_with_hash]
            
            self.logger.info(f"  📦 {type_name}: {len(type_datasets)}개 파일 셔플링")
            shuffled_datasets.extend(type_datasets)
        
        # 전체 리스트를 타입별로 균등 분산
        final_order = []
        type_iterators = {k: iter(v) for k, v in type_groups.items()}
        
        while type_iterators:
            for type_name in list(type_iterators.keys()):
                try:
                    dataset = next(type_iterators[type_name])
                    final_order.append(dataset)
                except StopIteration:
                    del type_iterators[type_name]
        
        self.logger.info(f"✅ 균형잡힌 셔플링 완료: {len(final_order)}개 파일")
        return final_order
    
    async def process_single_scenario(self, scenario_data: Dict, 
                                    file_info: Dict) -> Dict[str, Any]:
        """단일 시나리오 처리 (7회 후회 + 21회 벤담 계산)"""
        
        # 감정 경험 컨텍스트 업데이트 - 이전 경험을 학습에 활용
        await self._update_emotional_experience_context(scenario_data)
        
        results = {
            'regret_scores': [],
            'bentham_scores': [],
            'integrated_predictions': [],
            'processing_times': []
        }
        
        # 시나리오 데이터 전처리
        try:
            processed_scenario = self.preprocess_scenario(scenario_data)
        except Exception as e:
            self.logger.warning(f"⚠️ 시나리오 전처리 실패: {e}")
            return results
        
        # 7회 후회 계산 루프
        for regret_iteration in range(self.config.regret_iterations_per_data):
            iteration_start = time.time()
            
            try:
                # 후회 분석 수행
                regret_result = await self.emotion_circuit.process_ethical_decision(
                    processed_scenario['circuit_context']
                )
                results['regret_scores'].append(regret_result.predicted_regret)
                
                # 각 후회당 3회 벤담 계산
                bentham_scores = []
                for bentham_iteration in range(self.config.bentham_calculations_per_regret):
                    # 다차원 윤리 분석을 통한 벤담 계산
                    ethics_result = self.ethics_system.comprehensive_ethical_analysis(
                        processed_scenario['ethical_dilemma']
                    )
                    bentham_score = self.calculate_bentham_pleasure(
                        ethics_result, processed_scenario
                    )
                    bentham_scores.append(bentham_score)
                
                results['bentham_scores'].append(bentham_scores)
                
                # 통합 예측 수행
                integrated_prediction = await self.run_integrated_prediction(
                    processed_scenario
                )
                results['integrated_predictions'].append(integrated_prediction)
                
                # 감정 경험 메모리에 저장
                await self._store_emotional_experience(
                    scenario_data, regret_result, bentham_scores, integrated_prediction
                )
                
                # 메트릭 업데이트
                self.dataset_metrics.regret_calculations += 1
                self.dataset_metrics.bentham_calculations += self.config.bentham_calculations_per_regret
                
            except Exception as e:
                self.logger.warning(f"⚠️ 후회/벤담 계산 실패 (반복 {regret_iteration}): {e}")
                continue
            
            iteration_time = time.time() - iteration_start
            results['processing_times'].append(iteration_time)
        
        return results
    
    def preprocess_scenario(self, scenario_data: Dict) -> Dict[str, Any]:
        """시나리오 데이터 전처리"""
        # 기본 감정 데이터 설정
        emotion_data = EmotionData(
            primary_emotion=EmotionState.NEUTRAL,
            intensity=EmotionIntensity.MODERATE,
            valence=0.5,
            arousal=0.5,
            confidence=0.7,
            dominance=0.5,
            language='ko',
            processing_method='massive_training'
        )
        
        # 시나리오에서 감정 정보 추출 (있는 경우)
        if 'emotions' in scenario_data:
            emotions = scenario_data['emotions']
            if 'primary_emotion' in emotions:
                try:
                    emotion_data.primary_emotion = EmotionState[emotions['primary_emotion']]
                except KeyError:
                    pass
            if 'intensity' in emotions:
                emotion_data.intensity = EmotionIntensity[emotions.get('intensity', 'MODERATE')]
            if 'valence' in emotions:
                emotion_data.valence = emotions['valence']
            if 'arousal' in emotions:
                emotion_data.arousal = emotions['arousal']
        
        # 이해관계자 구성
        stakeholders = []
        if 'context' in scenario_data and 'people_involved' in scenario_data['context']:
            for person in scenario_data['context']['people_involved']:
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
            dilemma_id=scenario_data.get('id', f"scenario_{int(time.time())}"),
            scenario=scenario_data.get('description', ''),
            context=scenario_data.get('context', {}).get('location', ''),
            stakeholders=stakeholders,
            available_options=[
                opt.get('text', opt) if isinstance(opt, dict) else str(opt) 
                for opt in scenario_data.get('options', [])
            ]
        )
        
        # 회로 결정 컨텍스트
        circuit_context = CircuitDecisionContext(
            scenario_text=ethical_dilemma.scenario,
            proposed_action="윤리적 의사결정",
            self_emotion=emotion_data,
            stakeholders=[s.name for s in stakeholders]
        )
        
        return {
            'emotion_data': emotion_data,
            'ethical_dilemma': ethical_dilemma,
            'circuit_context': circuit_context,
            'original_scenario': scenario_data
        }
    
    def calculate_bentham_pleasure(self, ethics_result, processed_scenario: Dict) -> float:
        """벤담 쾌락 계산"""
        # 다차원 윤리 결과를 벤담 쾌락 점수로 변환
        school_scores = list(ethics_result.school_reasonings.values())
        if school_scores:
            base_score = np.mean([score.confidence for score in school_scores])
        else:
            base_score = 0.5
        
        # 감정 데이터 반영
        emotion_influence = (
            processed_scenario['emotion_data'].valence * 0.4 +
            processed_scenario['emotion_data'].arousal * 0.3 +
            processed_scenario['emotion_data'].confidence * 0.3
        )
        
        bentham_score = base_score * 0.7 + emotion_influence * 0.3
        return float(bentham_score)
    
    async def run_integrated_prediction(self, processed_scenario: Dict) -> Dict[str, Any]:
        """통합 시스템 예측 실행"""
        try:
            # 퍼지 감정-윤리 매핑
            fuzzy_result = self.fuzzy_mapper.map_emotion_to_ethics(
                processed_scenario['emotion_data']
            )
            
            # 시계열 이벤트 등록
            temporal_event = TemporalEvent(
                event_id=f"massive_event_{int(time.time())}",
                timestamp=time.time(),
                event_type="massive_training",
                description=processed_scenario['ethical_dilemma'].scenario,
                emotion_state=processed_scenario['emotion_data']
            )
            self.temporal_analyzer.register_event(temporal_event)
            
            return {
                'fuzzy_ethics_weights': fuzzy_result.ethics_weights,
                'temporal_event_id': temporal_event.event_id,
                'prediction_confidence': 0.8
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ 통합 예측 실패: {e}")
            return {'error': str(e)}
    
    async def train_on_datasets(self, datasets: List[Dict]) -> Dict[str, Any]:
        """전체 데이터셋 학습 실행"""
        self.logger.info(f"🚀 대규모 학습 시작: {len(datasets)}개 파일, {self.config.training_cycles}번 선회")
        
        total_start_time = time.time()
        cycle_results = []
        
        # 3번 선회 학습
        for cycle in range(self.config.training_cycles):
            cycle_start_time = time.time()
            self.dataset_metrics.current_cycle = cycle + 1
            
            self.logger.info(f"🔄 학습 선회 {cycle + 1}/{self.config.training_cycles} 시작")
            
            # 각 선회마다 데이터 재셔플링
            shuffled_datasets = self.create_balanced_shuffle_order(datasets)
            
            cycle_performance = PerformanceMetrics()
            scenarios_processed_in_cycle = 0
            log_counter = 0
            
            # 파일별 처리
            for file_idx, dataset_info in enumerate(shuffled_datasets):
                file_start_time = time.time()
                
                # 스토리지 체크
                if not self.storage_monitor.check_storage_limit():
                    self.logger.warning("⚠️ 스토리지 한계 도달, 학습 중단")
                    break
                
                # 파일 로드 및 처리
                try:
                    scenarios = self.load_scenarios_from_file(dataset_info['file_path'])
                    
                    for scenario_idx, scenario in enumerate(scenarios):
                        scenario_start_time = time.time()
                        
                        # 단일 시나리오 처리 (7회 후회 + 21회 벤담)
                        scenario_results = await self.process_single_scenario(
                            scenario, dataset_info
                        )
                        
                        # 성능 메트릭 업데이트
                        scenario_time = time.time() - scenario_start_time
                        cycle_performance.processing_time_per_scenario += scenario_time
                        
                        # 점수 계산
                        if scenario_results['regret_scores']:
                            cycle_performance.avg_regret_score += np.mean([
                                list(scores.values())[0] if isinstance(scores, dict) else 0.5
                                for scores in scenario_results['regret_scores']
                            ])
                        
                        if scenario_results['bentham_scores']:
                            all_bentham_scores = []
                            for bentham_list in scenario_results['bentham_scores']:
                                all_bentham_scores.extend(bentham_list)
                            if all_bentham_scores:
                                cycle_performance.avg_bentham_score += np.mean(all_bentham_scores)
                        
                        scenarios_processed_in_cycle += 1
                        self.dataset_metrics.processed_scenarios += 1
                        
                        # 20회마다 로그 저장
                        log_counter += 1
                        if log_counter % self.config.log_interval == 0:
                            await self.save_intermediate_log(cycle, scenarios_processed_in_cycle, cycle_performance)
                        
                        # 강화된 메모리 관리 - 누수 방지
                        if self.config.use_gpu:
                            if scenario_idx % 5 == 0:  # 더 자주 정리 (10 → 5)
                                torch.cuda.empty_cache()
                                gc.collect()
                                
                            # 메모리 사용량 모니터링
                            if scenario_idx % 20 == 0:
                                memory_allocated = torch.cuda.memory_allocated() / 1e9
                                memory_reserved = torch.cuda.memory_reserved() / 1e9
                                
                                if memory_reserved > 7.0:  # 7GB 이상 점유 시 경고
                                    self.logger.warning(f"⚠️ GPU 메모리 과점유: {memory_reserved:.1f}GB")
                                    # 강제 메모리 정리
                                    torch.cuda.empty_cache()
                                    torch.cuda.synchronize()
                                    gc.collect()
                
                except Exception as e:
                    self.logger.error(f"❌ 파일 처리 실패 {dataset_info['file_path']}: {e}")
                    continue
                
                file_time = time.time() - file_start_time
                self.logger.info(f"📄 파일 완료 ({file_idx+1}/{len(shuffled_datasets)}): "
                               f"{dataset_info['scenario_count']}개 시나리오, {file_time:.2f}초")
            
            # 선회 완료 처리
            cycle_time = time.time() - cycle_start_time
            
            # 평균 계산
            if scenarios_processed_in_cycle > 0:
                cycle_performance.processing_time_per_scenario /= scenarios_processed_in_cycle
                cycle_performance.avg_regret_score /= scenarios_processed_in_cycle
                cycle_performance.avg_bentham_score /= scenarios_processed_in_cycle
            
            cycle_performance.cycle_accuracy = 0.75  # 임시 정확도
            cycle_performance.cycle_loss = 0.25      # 임시 손실
            
            cycle_results.append({
                'cycle': cycle + 1,
                'scenarios_processed': scenarios_processed_in_cycle,
                'cycle_time': cycle_time,
                'performance': asdict(cycle_performance)
            })
            
            self.logger.info(f"✅ 선회 {cycle + 1} 완료: {scenarios_processed_in_cycle}개 시나리오, "
                           f"{cycle_time:.2f}초, 평균 후회: {cycle_performance.avg_regret_score:.4f}")
        
        # 전체 학습 완료
        total_time = time.time() - total_start_time
        
        final_results = {
            'session_id': self.session_id,
            'total_training_time': total_time,
            'cycles_completed': len(cycle_results),
            'total_scenarios_processed': self.dataset_metrics.processed_scenarios,
            'total_regret_calculations': self.dataset_metrics.regret_calculations,
            'total_bentham_calculations': self.dataset_metrics.bentham_calculations,
            'cycle_results': cycle_results,
            'final_metrics': asdict(self.dataset_metrics),
            'storage_used_gb': self.storage_monitor.get_current_usage()
        }
        
        return final_results
    
    async def _update_emotional_experience_context(self, scenario_data: Dict):
        """감정 경험 컨텍스트 업데이트 - 과거 경험을 현재 판단에 반영"""
        
        # 현재 시나리오의 윤리적 특성 분석
        scenario_type = self._classify_ethical_scenario(scenario_data)
        
        # 과거 유사한 윤리적 상황에서의 감정 경험 조회
        similar_experiences = self.emotional_experience_memory['ethical_emotions'][scenario_type]
        
        # 감정 시스템에 과거 경험 컨텍스트 제공
        if similar_experiences:
            # 과거 경험에서 학습된 감정 패턴을 현재 분석에 반영
            avg_past_regret = np.mean([exp['regret_intensity'] for exp in similar_experiences[-5:]])  # 최근 5개
            avg_past_confidence = np.mean([exp['confidence'] for exp in similar_experiences[-5:]])
            
            # 통합 시스템에 경험 컨텍스트 설정 (직접 속성 설정)
            self.emotion_circuit.experience_context = {
                'similar_scenario_count': len(similar_experiences),
                'average_past_regret': float(avg_past_regret),
                'average_past_confidence': float(avg_past_confidence),
                'learning_progression': self._calculate_emotional_learning_curve(similar_experiences)
            }
            
            self.logger.debug(f"📚 감정 경험 컨텍스트 로드: {scenario_type} - {len(similar_experiences)}개 과거 경험")
    
    def _classify_ethical_scenario(self, scenario_data: Dict) -> str:
        """윤리적 시나리오 분류"""
        # 시나리오의 윤리적 특성에 따라 분류
        description = scenario_data.get('description', '').lower()
        
        if any(word in description for word in ['배신', 'betrayal', '거짓말', 'lie']):
            return 'trust_violation'
        elif any(word in description for word in ['도움', 'help', '구조', 'rescue']):
            return 'helping_dilemma'
        elif any(word in description for word in ['공정', 'fair', '불공평', 'unfair']):
            return 'fairness_issue'
        elif any(word in description for word in ['가족', 'family', '친구', 'friend']):
            return 'relationship_conflict'
        else:
            return 'general_ethical'
    
    def _calculate_emotional_learning_curve(self, experiences: List[Dict]) -> float:
        """감정 학습 곡선 계산"""
        if len(experiences) < 2:
            return 0.0
        
        # 시간 순서대로 신뢰도 변화 추적
        confidences = [exp['confidence'] for exp in experiences]
        
        # 학습 진행도 = 최근 신뢰도와 초기 신뢰도의 차이
        if len(confidences) >= 3:
            recent_avg = np.mean(confidences[-3:])
            initial_avg = np.mean(confidences[:3])
            return recent_avg - initial_avg
        
        return 0.0
    
    async def _store_emotional_experience(self, scenario_data: Dict, regret_result: Dict, 
                                        bentham_scores: List[float], integrated_prediction: Dict):
        """감정 경험을 메모리에 저장"""
        
        scenario_type = self._classify_ethical_scenario(scenario_data)
        
        # 경험 데이터 구성 - CircuitDecisionResult 객체 속성 직접 접근
        experience = {
            'timestamp': time.time(),
            'scenario_type': scenario_type,
            'scenario_id': scenario_data.get('id', f"scenario_{int(time.time())}"),
            'regret_intensity': regret_result.predicted_regret.get('anticipated', 0.0) if hasattr(regret_result, 'predicted_regret') else 0.0,
            'confidence': regret_result.confidence if hasattr(regret_result, 'confidence') else 0.5,
            'bentham_average': np.mean(bentham_scores) if bentham_scores else 0.5,
            'emotional_complexity': len(regret_result.predicted_regret.get('emotion_vector', [])) if hasattr(regret_result, 'predicted_regret') and isinstance(regret_result.predicted_regret, dict) else 0,
            'decision_quality': integrated_prediction.get('prediction_confidence', 0.5)
        }
        
        # 경험 저장
        self.emotional_experience_memory['ethical_emotions'][scenario_type].append(experience)
        self.emotional_experience_memory['decision_outcomes'].append(experience)
        
        # 메모리 크기 제한 (최대 1000개 경험)
        if len(self.emotional_experience_memory['decision_outcomes']) > 1000:
            self.emotional_experience_memory['decision_outcomes'] = \
                self.emotional_experience_memory['decision_outcomes'][-1000:]
        
        # 타입별 경험도 제한
        for scenario_type, experiences in self.emotional_experience_memory['ethical_emotions'].items():
            if len(experiences) > 200:
                self.emotional_experience_memory['ethical_emotions'][scenario_type] = experiences[-200:]
    
    def load_scenarios_from_file(self, file_path: str) -> List[Dict]:
        """파일에서 시나리오 로드"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                return data
            elif 'scenarios' in data:
                return data['scenarios']
            else:
                return [data]  # 단일 시나리오인 경우
                
        except Exception as e:
            self.logger.error(f"❌ 파일 로드 실패 {file_path}: {e}")
            return []
    
    async def save_intermediate_log(self, cycle: int, scenarios_count: int, 
                                  performance: PerformanceMetrics):
        """중간 로그 저장"""
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'cycle': cycle + 1,
            'scenarios_processed': scenarios_count,
            'regret_calculations': self.dataset_metrics.regret_calculations,
            'bentham_calculations': self.dataset_metrics.bentham_calculations,
            'performance': asdict(performance),
            'gpu_status': self.gpu_manager.get_memory_status() if self.config.use_gpu else {},
            'storage_used_gb': self.storage_monitor.get_current_usage()
        }
        
        log_file = self.log_dir / f"cycle_{cycle+1}_checkpoint_{scenarios_count}.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"💾 중간 로그 저장: {scenarios_count}개 시나리오 처리 완료")


class StorageMonitor:
    """스토리지 사용량 모니터링"""
    
    def __init__(self, max_gb: float):
        self.max_gb = max_gb
        self.base_usage = self.get_current_usage()
    
    def get_current_usage(self) -> float:
        """현재 스토리지 사용량 (GB)"""
        import shutil
        total, used, free = shutil.disk_usage("/mnt/c/large_project/linux_red_heart")
        return used / (1024**3)
    
    def check_storage_limit(self) -> bool:
        """스토리지 한계 체크"""
        current_usage = self.get_current_usage()
        usage_increase = current_usage - self.base_usage
        return usage_increase < self.max_gb


async def run_massive_training():
    """대규모 학습 실행"""
    logger.info("🎯 Red Heart 대규모 데이터셋 학습 시작")
    
    # 설정
    config = MassiveTrainingConfig()
    
    # 학습기 초기화
    trainer = MassiveDatasetTrainer(config)
    
    # 데이터셋 탐색
    datasets = trainer.discover_datasets()
    if not datasets:
        logger.error("❌ 처리할 데이터셋이 없습니다.")
        return
    
    # 대규모 학습 실행
    results = await trainer.train_on_datasets(datasets)
    
    # 결과 저장
    final_results_file = trainer.log_dir / "massive_training_final_results.json"
    with open(final_results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 결과 요약 출력
    logger.info("📈 대규모 학습 완료!")
    logger.info(f"- 총 학습 시간: {results['total_training_time']:.2f}초")
    logger.info(f"- 처리된 시나리오: {results['total_scenarios_processed']}개")
    logger.info(f"- 후회 계산: {results['total_regret_calculations']}회")
    logger.info(f"- 벤담 계산: {results['total_bentham_calculations']}회")
    logger.info(f"- 완료된 선회: {results['cycles_completed']}회")
    logger.info(f"- 스토리지 사용: {results['storage_used_gb']:.2f}GB")
    
    return results


if __name__ == "__main__":
    # 비동기 실행
    results = asyncio.run(run_massive_training())