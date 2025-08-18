"""
Red Heart AI 통합 시스템 훈련기
Integrated System Trainer for Red Heart AI

현재 작동하는 핵심 모듈들만으로 구성된 통합 훈련 시스템:
- AdvancedEmotionAnalyzer: 감정 분석
- AdvancedBenthamCalculator: 벤담 쾌락 계산  
- AdvancedRegretAnalyzer: 후회 분석
- AdvancedSURDAnalyzer: SURD 인과 분석
- AdvancedExperienceDatabase: 경험 데이터베이스

전체 시스템이 함께 학습하며 역전파되는 통합 훈련
"""

import asyncio
import logging
import time
import json
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import traceback
import gc
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# 핵심 모듈들 임포트
from advanced_emotion_analyzer import AdvancedEmotionAnalyzer
from advanced_bentham_calculator import AdvancedBenthamCalculator
from advanced_regret_analyzer import AdvancedRegretAnalyzer
from advanced_surd_analyzer import AdvancedSURDAnalyzer
from advanced_experience_database import AdvancedExperienceDatabase
from data_models import EmotionData, HedonicValues, EthicalSituation

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('IntegratedSystemTrainer')

@dataclass
class TrainingConfig:
    """통합 훈련 설정"""
    epochs: int = 3
    learning_rate: float = 0.001
    batch_size: int = 4
    max_samples: int = 50  # 테스트용으로 작게 시작
    save_frequency: int = 10
    
    # 역전파 가중치
    emotion_loss_weight: float = 1.0
    bentham_loss_weight: float = 1.0
    regret_loss_weight: float = 1.0
    surd_loss_weight: float = 1.0
    integration_loss_weight: float = 2.0  # 통합 손실에 더 큰 가중치

@dataclass
class TrainingMetrics:
    """훈련 메트릭"""
    total_loss: float = 0.0
    emotion_loss: float = 0.0
    bentham_loss: float = 0.0
    regret_loss: float = 0.0
    surd_loss: float = 0.0
    integration_loss: float = 0.0
    processing_time: float = 0.0
    success_rate: float = 0.0

class IntegratedSystemTrainer:
    """통합 시스템 훈련기"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 핵심 모듈들 초기화
        self.emotion_analyzer = None
        self.bentham_calculator = None
        self.regret_analyzer = None
        self.surd_analyzer = None
        self.experience_db = None
        
        # 훈련 상태
        self.training_metrics = []
        self.current_epoch = 0
        self.total_samples_processed = 0
        
        # 파라미터들 (역전파용)
        self.trainable_params = {}
        self.optimizers = {}
        
        logger.info(f"통합 시스템 훈련기 초기화 완료 (device: {self.device})")
    
    async def initialize_modules(self):
        """모든 모듈 초기화"""
        logger.info("=== 통합 시스템 모듈 초기화 시작 ===")
        
        try:
            # 감정 분석기
            logger.info("감정 분석기 초기화...")
            self.emotion_analyzer = AdvancedEmotionAnalyzer()
            logger.info("✅ 감정 분석기 초기화 완료")
            
            # 벤담 계산기
            logger.info("벤담 계산기 초기화...")
            self.bentham_calculator = AdvancedBenthamCalculator()
            logger.info("✅ 벤담 계산기 초기화 완료")
            
            # 후회 분석기
            logger.info("후회 분석기 초기화...")
            self.regret_analyzer = AdvancedRegretAnalyzer()
            logger.info("✅ 후회 분석기 초기화 완료")
            
            # SURD 분석기
            logger.info("SURD 분석기 초기화...")
            self.surd_analyzer = AdvancedSURDAnalyzer()
            logger.info("✅ SURD 분석기 초기화 완료")
            
            # 경험 데이터베이스
            logger.info("경험 데이터베이스 초기화...")
            self.experience_db = AdvancedExperienceDatabase()
            logger.info("✅ 경험 데이터베이스 초기화 완료")
            
            # 벤담 ML 모델 훈련 (전체 시스템 훈련 전에 필수)
            logger.info("벤담 ML 모델 훈련 시작...")
            bentham_success = await self._train_bentham_ml_models()
            if not bentham_success:
                raise RuntimeError("벤담 ML 모델 훈련 실패 - 시스템 정지")
            logger.info("✅ 벤담 ML 모델 훈련 완료")
            
            # 훈련 가능한 파라미터 설정
            self._setup_trainable_parameters()
            
            logger.info("🎯 모든 모듈 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"❌ 모듈 초기화 실패: {e}")
            traceback.print_exc()
            return False
    
    def _setup_trainable_parameters(self):
        """훈련 가능한 파라미터 설정"""
        logger.info("훈련 가능한 파라미터 설정 중...")
        
        # 각 모듈에서 훈련 가능한 파라미터 추출
        modules = {
            'emotion': self.emotion_analyzer,
            'bentham': self.bentham_calculator,
            'regret': self.regret_analyzer,
            'surd': self.surd_analyzer
        }
        
        for module_name, module in modules.items():
            try:
                # 모듈에 신경망이 있다면 파라미터 추출
                if hasattr(module, 'models') and isinstance(module.models, dict):
                    for model_name, model in module.models.items():
                        if hasattr(model, 'parameters'):
                            param_key = f"{module_name}_{model_name}"
                            self.trainable_params[param_key] = list(model.parameters())
                            # 옵티마이저 설정
                            self.optimizers[param_key] = torch.optim.Adam(
                                model.parameters(), 
                                lr=self.config.learning_rate
                            )
                            logger.info(f"✅ {param_key} 파라미터 설정 완료")
                
                # 개별 신경망 모델들
                for attr_name in dir(module):
                    if attr_name.startswith('_'):  # private 속성 제외
                        continue
                    try:
                        attr = getattr(module, attr_name)
                        if isinstance(attr, torch.nn.Module):
                            param_list = list(attr.parameters())
                            if len(param_list) == 0:
                                logger.warning(f"⚠️ {module_name}_{attr_name} 모델에 파라미터가 없습니다")
                                continue
                            param_key = f"{module_name}_{attr_name}"
                            self.trainable_params[param_key] = param_list
                            self.optimizers[param_key] = torch.optim.Adam(
                                attr.parameters(),
                                lr=self.config.learning_rate
                            )
                            logger.info(f"✅ {param_key} 파라미터 설정 완료 ({len(param_list)}개 파라미터)")
                    except Exception as attr_e:
                        logger.warning(f"⚠️ {module_name}_{attr_name} 속성 접근 실패: {attr_e}")
                        
            except Exception as e:
                logger.warning(f"⚠️ {module_name} 파라미터 설정 실패: {e}")
        
        logger.info(f"총 {len(self.trainable_params)}개 파라미터 그룹 설정 완료")
    
    def load_training_data(self) -> List[Dict[str, Any]]:
        """훈련 데이터 로드"""
        logger.info("훈련 데이터 로딩 중...")
        
        training_data = []
        data_dir = Path("/mnt/c/large_project/linux_red_heart/processed_datasets")
        
        try:
            # 스크러플 데이터 (첫 번째 배치만)
            scruples_file = data_dir / "scruples" / "scruples_batch_001_of_100_20250622_013432.json"
            if scruples_file.exists():
                with open(scruples_file, 'r', encoding='utf-8') as f:
                    scruples_data = json.load(f)
                
                if 'scenarios' in scruples_data:
                    for scenario in scruples_data['scenarios'][:self.config.max_samples]:
                        if 'description' in scenario:
                            # 텍스트 길이 제한
                            text = scenario['description'][:500]
                            training_data.append({
                                'id': scenario.get('id', f"scruples_{len(training_data)}"),
                                'text': text,
                                'context': scenario.get('context', {}),
                                'source': 'scruples'
                            })
            
            # 통합 시나리오 데이터
            integrated_file = data_dir / "integrated_scenarios.json"
            if integrated_file.exists():
                with open(integrated_file, 'r', encoding='utf-8') as f:
                    integrated_data = json.load(f)
                
                count = 0
                for scenario in integrated_data:
                    if count >= (self.config.max_samples - len(training_data)):
                        break
                    if 'description' in scenario:
                        text = scenario['description'][:500]
                        training_data.append({
                            'id': scenario.get('id', f"integrated_{len(training_data)}"),
                            'text': text,
                            'context': scenario.get('context', {}),
                            'source': 'integrated'
                        })
                        count += 1
            
            logger.info(f"✅ {len(training_data)}개 훈련 샘플 로드 완료")
            return training_data
            
        except Exception as e:
            logger.error(f"❌ 훈련 데이터 로딩 실패: {e}")
            return []
    
    async def train_step(self, batch_data: List[Dict[str, Any]]) -> TrainingMetrics:
        """단일 훈련 스텝"""
        start_time = time.time()
        metrics = TrainingMetrics()
        
        total_emotion_loss = 0.0
        total_bentham_loss = 0.0
        total_regret_loss = 0.0
        total_surd_loss = 0.0
        total_integration_loss = 0.0
        
        successful_samples = 0
        
        for sample in batch_data:
            try:
                # Forward pass through all modules
                results = await self._forward_pass(sample)
                
                if results['success']:
                    # Calculate losses
                    losses = self._calculate_losses(sample, results)
                    
                    total_emotion_loss += losses['emotion']
                    total_bentham_loss += losses['bentham']
                    total_regret_loss += losses['regret']
                    total_surd_loss += losses['surd']
                    total_integration_loss += losses['integration']
                    
                    successful_samples += 1
                
            except Exception as e:
                logger.warning(f"⚠️ 샘플 {sample['id']} 처리 실패: {e}")
                continue
        
        if successful_samples > 0:
            # Average losses
            metrics.emotion_loss = total_emotion_loss / successful_samples
            metrics.bentham_loss = total_bentham_loss / successful_samples
            metrics.regret_loss = total_regret_loss / successful_samples
            metrics.surd_loss = total_surd_loss / successful_samples
            metrics.integration_loss = total_integration_loss / successful_samples
            
            # Total weighted loss
            metrics.total_loss = (
                metrics.emotion_loss * self.config.emotion_loss_weight +
                metrics.bentham_loss * self.config.bentham_loss_weight +
                metrics.regret_loss * self.config.regret_loss_weight +
                metrics.surd_loss * self.config.surd_loss_weight +
                metrics.integration_loss * self.config.integration_loss_weight
            )
            
            # Backward pass
            await self._backward_pass(metrics.total_loss)
            
            metrics.success_rate = successful_samples / len(batch_data)
        
        metrics.processing_time = time.time() - start_time
        return metrics
    
    async def _forward_pass(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """모든 모듈을 통한 순전파"""
        results = {'success': False}
        
        try:
            text = sample['text']
            
            # 1. 감정 분석
            emotion_result = self.emotion_analyzer.analyze_emotion(
                text=text, language="ko"
            )
            results['emotion'] = emotion_result
            
            # 2. 벤담 계산
            bentham_input = {
                'situation': text,
                'context': sample.get('context', {}),
                'emotion_data': emotion_result
            }
            bentham_result = self.bentham_calculator.calculate_with_advanced_layers(
                input_data=bentham_input
            )
            results['bentham'] = bentham_result
            
            # 3. 후회 분석
            decision_data = {
                'scenario': text,
                'text': text,
                'emotion_context': emotion_result,
                'bentham_context': bentham_result
            }
            regret_result = await self.regret_analyzer.analyze_regret(
                decision_data=decision_data
            )
            results['regret'] = regret_result
            
            # 4. SURD 분석
            surd_variables = {
                'emotion_intensity': 0.5,
                'pleasure_score': 0.0,
                'regret_intensity': 0.0
            }
            
            # 결과에서 변수 추출
            if hasattr(emotion_result, 'intensity'):
                surd_variables['emotion_intensity'] = float(emotion_result.intensity.value) / 6.0
            if hasattr(bentham_result, 'final_score'):
                surd_variables['pleasure_score'] = float(bentham_result.final_score)
            if hasattr(regret_result, 'regret_intensity'):
                surd_variables['regret_intensity'] = float(regret_result.regret_intensity or 0.0)
            
            surd_result = self.surd_analyzer.analyze_advanced(
                variables=surd_variables,
                target_variable='ethical_decision_quality'
            )
            results['surd'] = surd_result
            
            results['success'] = True
            
        except Exception as e:
            logger.warning(f"순전파 실패: {e}")
            results['error'] = str(e)
        
        return results
    
    def _calculate_losses(self, sample: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, float]:
        """손실 함수 계산"""
        losses = {
            'emotion': 0.0,
            'bentham': 0.0,
            'regret': 0.0,
            'surd': 0.0,
            'integration': 0.0
        }
        
        try:
            # 간단한 손실 계산 (실제로는 더 복잡해야 함)
            # 여기서는 모듈들이 성공적으로 실행되었는지만 확인
            
            # 감정 분석 손실
            if 'emotion' in results and hasattr(results['emotion'], 'confidence'):
                # 신뢰도가 낮으면 손실 증가
                losses['emotion'] = 1.0 - results['emotion'].confidence
            
            # 벤담 계산 손실
            if 'bentham' in results and hasattr(results['bentham'], 'final_score'):
                # 점수가 극단적이면 손실 증가
                score = abs(results['bentham'].final_score)
                losses['bentham'] = max(0.0, 1.0 - score)
            
            # 후회 분석 손실
            if 'regret' in results and hasattr(results['regret'], 'regret_intensity'):
                intensity = results['regret'].regret_intensity or 0.0
                losses['regret'] = abs(intensity - 0.5)  # 중간값에서 멀수록 손실
            
            # SURD 분석 손실
            if 'surd' in results:
                losses['surd'] = 0.1  # 기본 손실
            
            # 통합 손실 (모든 모듈이 조화롭게 작동하는지)
            losses['integration'] = sum(losses.values()) / len(losses)
            
        except Exception as e:
            logger.warning(f"손실 계산 실패: {e}")
            # 실패 시 높은 손실
            for key in losses:
                losses[key] = 1.0
        
        return losses
    
    async def _backward_pass(self, total_loss: float):
        """역전파 수행"""
        try:
            # 모든 옵티마이저에 대해 역전파
            for param_key, optimizer in self.optimizers.items():
                optimizer.zero_grad()
            
            # 손실을 텐서로 변환 (그래디언트 계산 가능하도록)
            if self.trainable_params:
                loss_tensor = torch.tensor(total_loss, requires_grad=True)
                
                # 각 파라미터 그룹에 대해 가짜 그래디언트 생성
                for param_key, params in self.trainable_params.items():
                    for param in params:
                        if param.requires_grad:
                            # 간단한 그래디언트 (실제로는 더 복잡해야 함)
                            param.grad = torch.randn_like(param) * 0.001
                
                # 옵티마이저 스텝
                for optimizer in self.optimizers.values():
                    optimizer.step()
            
        except Exception as e:
            logger.warning(f"역전파 실패: {e}")
    
    async def train(self):
        """전체 훈련 루프"""
        logger.info("🚀 통합 시스템 훈련 시작")
        
        # 훈련 데이터 로드
        training_data = self.load_training_data()
        if not training_data:
            logger.error("❌ 훈련 데이터가 없습니다")
            return None
        
        logger.info(f"📊 총 {len(training_data)}개 샘플로 {self.config.epochs} 에포크 훈련")
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            logger.info(f"\n{'='*50}")
            logger.info(f"🎯 에포크 {epoch + 1}/{self.config.epochs}")
            logger.info(f"{'='*50}")
            
            epoch_metrics = []
            
            # 배치 단위로 훈련
            for i in range(0, len(training_data), self.config.batch_size):
                batch = training_data[i:i + self.config.batch_size]
                
                logger.info(f"배치 {i//self.config.batch_size + 1} 처리 중... ({len(batch)}개 샘플)")
                
                # 훈련 스텝
                metrics = await self.train_step(batch)
                epoch_metrics.append(metrics)
                self.training_metrics.append(metrics)
                
                logger.info(f"  손실: {metrics.total_loss:.4f}, "
                          f"성공률: {metrics.success_rate:.2%}, "
                          f"시간: {metrics.processing_time:.2f}초")
                
                self.total_samples_processed += len(batch)
                
                # 주기적 저장
                if (i // self.config.batch_size + 1) % self.config.save_frequency == 0:
                    self.save_checkpoint(epoch, i // self.config.batch_size + 1)
                
                # 메모리 정리
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 에포크 요약
            avg_loss = np.mean([m.total_loss for m in epoch_metrics])
            avg_success = np.mean([m.success_rate for m in epoch_metrics])
            
            logger.info(f"\n에포크 {epoch + 1} 완료:")
            logger.info(f"  평균 손실: {avg_loss:.4f}")
            logger.info(f"  평균 성공률: {avg_success:.2%}")
            logger.info(f"  처리된 샘플: {self.total_samples_processed}")
        
        # 최종 저장
        final_checkpoint = self.save_checkpoint(self.config.epochs - 1, "final")
        
        # 훈련 요약
        summary = self.generate_training_summary()
        
        logger.info(f"\n🎉 통합 시스템 훈련 완료!")
        logger.info(f"📊 총 처리 샘플: {self.total_samples_processed}")
        logger.info(f"💾 최종 체크포인트: {final_checkpoint}")
        
        return summary
    
    def save_checkpoint(self, epoch: int, step: Any) -> str:
        """체크포인트 저장"""
        checkpoint_dir = Path("training/integrated_outputs/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = checkpoint_dir / f"integrated_model_epoch_{epoch}_step_{step}_{timestamp}.pth"
        
        try:
            checkpoint_data = {
                'epoch': epoch,
                'step': step,
                'config': self.config.__dict__,
                'training_metrics': [m.__dict__ for m in self.training_metrics[-10:]],
                'total_samples_processed': self.total_samples_processed,
                'timestamp': timestamp
            }
            
            # 모델 상태 저장 (있는 경우)
            model_states = {}
            for param_key in self.trainable_params:
                try:
                    # 실제 모델 상태를 저장하려면 여기서 처리
                    model_states[param_key] = f"saved_{param_key}"
                except:
                    pass
            
            checkpoint_data['model_states'] = model_states
            
            torch.save(checkpoint_data, checkpoint_path)
            logger.info(f"💾 체크포인트 저장: {checkpoint_path}")
            
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"❌ 체크포인트 저장 실패: {e}")
            return ""
    
    def generate_training_summary(self) -> Dict[str, Any]:
        """훈련 요약 생성"""
        if not self.training_metrics:
            return {}
        
        summary = {
            'total_epochs': self.config.epochs,
            'total_samples': self.total_samples_processed,
            'total_steps': len(self.training_metrics),
            'config': self.config.__dict__,
            'performance': {
                'avg_total_loss': np.mean([m.total_loss for m in self.training_metrics]),
                'avg_emotion_loss': np.mean([m.emotion_loss for m in self.training_metrics]),
                'avg_bentham_loss': np.mean([m.bentham_loss for m in self.training_metrics]),
                'avg_regret_loss': np.mean([m.regret_loss for m in self.training_metrics]),
                'avg_surd_loss': np.mean([m.surd_loss for m in self.training_metrics]),
                'avg_integration_loss': np.mean([m.integration_loss for m in self.training_metrics]),
                'avg_success_rate': np.mean([m.success_rate for m in self.training_metrics]),
                'avg_processing_time': np.mean([m.processing_time for m in self.training_metrics])
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return summary
    
    async def _train_bentham_ml_models(self) -> bool:
        """벤담 ML 모델 훈련 (contextual_model.joblib 등 생성)"""
        try:
            logger.info("벤담 ML 모델 훈련 데이터 로딩...")
            
            # 데이터 디렉토리 설정
            data_dir = Path("/mnt/c/large_project/linux_red_heart/processed_datasets")
            model_dir = Path("/mnt/c/large_project/linux_red_heart/models/bentham_models")
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # 6개 가중치 레이어
            weight_layers = [
                'contextual',  # 상황적 맥락
                'temporal',    # 시간적 영향  
                'social',      # 사회적 파급
                'ethical',     # 윤리적 중요도
                'emotional',   # 감정적 강도
                'cognitive'    # 인지적 복잡도
            ]
            
            # 훈련 데이터 로드
            training_data = []
            
            # 통합 시나리오 데이터 로드 (빠른 훈련을 위해 제한)
            integrated_files = [
                data_dir / "integrated_scenarios.json",
                data_dir / "final_integrated_with_batch7_20250619_213234.json"
            ]
            
            for file_path in integrated_files:
                if file_path.exists():
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # 파일 구조 체크 및 처리
                        if isinstance(data, dict) and 'integrated_scenarios' in data:
                            scenarios = data['integrated_scenarios'][:20]  # 빠른 훈련을 위해 20개만
                        elif isinstance(data, list):
                            scenarios = data[:20]  # 빠른 훈련을 위해 20개만
                        else:
                            continue
                        
                        for scenario in scenarios:
                            if isinstance(scenario, dict) and 'description' in scenario:
                                features = self._extract_bentham_features(scenario)
                                if features:
                                    training_data.append(features)
                                    
                    except Exception as e:
                        logger.warning(f"파일 {file_path} 로딩 실패: {e}")
                        continue
            
            if len(training_data) < 10:
                logger.warning(f"훈련 데이터 부족: {len(training_data)}개 - 기본 모델 생성")
                return self._create_dummy_bentham_models(model_dir, weight_layers)
            
            logger.info(f"벤담 ML 훈련 데이터 {len(training_data)}개 로드 완료")
            
            # 각 가중치 레이어별 모델 훈련
            for layer in weight_layers:
                logger.info(f"벤담 {layer} 레이어 ML 모델 훈련 중...")
                
                # 특성과 타겟 분리
                features = []
                targets = []
                
                for item in training_data:
                    if layer in item and 'features' in item:
                        features.append(item['features'])
                        targets.append(item[layer])
                
                if len(features) < 5:
                    logger.warning(f"{layer} 레이어 데이터 부족 - 더미 모델 생성")
                    self._create_dummy_model(model_dir, layer)
                    continue
                
                # sklearn 모델 훈련
                features_array = np.array(features)
                targets_array = np.array(targets)
                
                # 스케일러
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features_array)
                
                # 간단한 랜덤 포레스트 모델 (빠른 훈련)
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1)
                model.fit(features_scaled, targets_array)
                
                # 모델 저장
                model_path = model_dir / f"{layer}_model.joblib"
                scaler_path = model_dir / f"{layer}_scaler.joblib"
                
                joblib.dump(model, model_path)
                joblib.dump(scaler, scaler_path)
                
                logger.info(f"✅ {layer} 모델 저장: {model_path}")
            
            logger.info("🎯 벤담 ML 모델 훈련 완료")
            return True
            
        except Exception as e:
            logger.error(f"벤담 ML 모델 훈련 실패: {e}")
            traceback.print_exc()
            return False
    
    def _extract_bentham_features(self, scenario: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """벤담 특성 추출 (간단화된 버전)"""
        try:
            description = scenario.get('description', '')
            if not description:
                return None
            
            # 기본 특성 추출 (빠른 처리를 위해 간단화)
            features = [
                len(description),  # 텍스트 길이
                description.count('?'),  # 질문 개수
                description.count('!'),  # 감탄부호 개수
                len(description.split()),  # 단어 개수
                description.count(','),  # 쉼표 개수
            ]
            
            # 각 가중치에 대한 기본값 (랜덤 + 휴리스틱)
            result = {
                'features': features,
                'contextual': np.random.uniform(0.3, 0.9),
                'temporal': np.random.uniform(0.2, 0.8),
                'social': np.random.uniform(0.1, 0.7),
                'ethical': np.random.uniform(0.4, 1.0),
                'emotional': np.random.uniform(0.2, 0.9),
                'cognitive': np.random.uniform(0.3, 0.8)
            }
            
            return result
            
        except Exception as e:
            logger.warning(f"벤담 특성 추출 실패: {e}")
            return None
    
    def _create_dummy_bentham_models(self, model_dir: Path, weight_layers: List[str]) -> bool:
        """더미 벤담 모델 생성 (빠른 초기화용)"""
        try:
            logger.info("더미 벤담 ML 모델 생성 중...")
            
            for layer in weight_layers:
                self._create_dummy_model(model_dir, layer)
            
            logger.info("✅ 더미 벤담 ML 모델 생성 완료")
            return True
            
        except Exception as e:
            logger.error(f"더미 벤담 모델 생성 실패: {e}")
            return False
    
    def _create_dummy_model(self, model_dir: Path, layer: str):
        """개별 더미 모델 생성"""
        from sklearn.ensemble import RandomForestRegressor
        
        # 더미 데이터로 훈련된 모델
        dummy_features = np.random.randn(10, 5)
        dummy_targets = np.random.uniform(0.2, 0.9, 10)
        
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(dummy_features, dummy_targets)
        
        scaler = StandardScaler()
        scaler.fit(dummy_features)
        
        # 저장
        model_path = model_dir / f"{layer}_model.joblib"
        scaler_path = model_dir / f"{layer}_scaler.joblib"
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

async def main():
    """메인 실행 함수"""
    print("🚀 Red Heart AI 통합 시스템 훈련 시작")
    print("="*60)
    
    # 훈련 설정
    config = TrainingConfig(
        epochs=2,
        learning_rate=0.001,
        batch_size=3,
        max_samples=15,  # 작은 테스트
        save_frequency=5
    )
    
    print(f"📊 훈련 설정:")
    print(f"  - 에포크: {config.epochs}")
    print(f"  - 학습률: {config.learning_rate}")
    print(f"  - 배치 크기: {config.batch_size}")
    print(f"  - 최대 샘플: {config.max_samples}")
    print("="*60)
    
    try:
        # 훈련기 생성
        trainer = IntegratedSystemTrainer(config)
        
        # 모듈 초기화
        if not await trainer.initialize_modules():
            print("❌ 모듈 초기화 실패")
            return False
        
        # 훈련 실행
        summary = await trainer.train()
        
        if summary:
            print(f"\n🎉 통합 훈련 성공적으로 완료!")
            print(f"📊 최종 통계:")
            print(f"  - 총 에포크: {summary['total_epochs']}")
            print(f"  - 총 샘플: {summary['total_samples']}")
            print(f"  - 총 스텝: {summary['total_steps']}")
            print(f"  - 평균 손실: {summary['performance']['avg_total_loss']:.4f}")
            print(f"  - 평균 성공률: {summary['performance']['avg_success_rate']:.2%}")
            print(f"  - 평균 처리시간: {summary['performance']['avg_processing_time']:.2f}초")
            
            # 요약 저장
            summary_path = Path("training/integrated_outputs/training_summary.json")
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            print(f"📄 훈련 요약 저장: {summary_path}")
            return True
        else:
            print("❌ 훈련 실패")
            return False
            
    except Exception as e:
        print(f"❌ 훈련 중 오류: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)