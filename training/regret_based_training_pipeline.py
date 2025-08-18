#!/usr/bin/env python3
"""
후회 기반 학습 파이프라인 (7회 후회/스텝, 3번 선회)
Regret-Based Training Pipeline (7 regrets per step, 3 epochs)
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import math
import gc
from collections import defaultdict

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 모듈 imports
from models.mega_scale_models.scalable_xai_model import create_mega_scale_model, optimize_model_for_inference
from models.hierarchical_emotion.emotion_phase_models import HierarchicalEmotionModel
from xai_core.xai_logging_system import xai_logger, xai_trace
from llm_module import llm_tracker, register_llm, ask_llm

@dataclass
class RegretTrainingConfig:
    """후회 기반 학습 설정"""
    # 후회 설정
    regrets_per_step: int = 7
    bentham_calculations_per_regret: int = 3  # 총 21번의 벤담 계산
    epochs: int = 3
    
    # 학습 설정
    batch_size: int = 16
    learning_rate: float = 1e-4
    warmup_steps: int = 100
    max_gradient_norm: float = 1.0
    
    # 로깅 설정
    log_every_n_steps: int = 20
    save_checkpoint_every: int = 100
    
    # 스토리지 설정
    max_storage_gb: float = 200.0
    cleanup_old_logs: bool = True
    
    # 모델 설정
    model_params: int = 200_000_000
    sequence_length: int = 128
    
    def __post_init__(self):
        """계산된 값들"""
        self.total_bentham_per_step = self.regrets_per_step * self.bentham_calculations_per_regret

class RegretCalculator:
    """후회 계산 모듈"""
    
    def __init__(self, config: RegretTrainingConfig):
        self.config = config
        
        # 후회 유형별 가중치
        self.regret_weights = {
            'counterfactual': 0.3,    # "만약 ~했다면"
            'temporal': 0.2,          # "그때 ~했어야 했는데"
            'moral': 0.25,            # "옳은 일을 하지 못했다"
            'opportunity': 0.15,      # "기회를 놓쳤다"
            'social': 0.1             # "다른 사람을 실망시켰다"
        }
        
        # 벤담 쾌락 계산 요소
        self.bentham_factors = {
            'intensity': 1.0,    # 강도
            'duration': 0.8,     # 지속성
            'certainty': 0.9,    # 확실성
            'propinquity': 0.7,  # 근접성
            'fecundity': 0.6,    # 다산성
            'purity': 0.8,       # 순수성
            'extent': 0.5        # 범위
        }
    
    def calculate_regret_scenarios(self, original_decision: torch.Tensor, 
                                 context: Dict[str, Any]) -> List[Dict[str, torch.Tensor]]:
        """7가지 후회 시나리오 생성"""
        scenarios = []
        
        for i, regret_type in enumerate(self.regret_weights.keys()):
            if i >= self.config.regrets_per_step:
                break
                
            # 각 후회 유형별 변형된 결정 생성
            regret_decision = self._generate_regret_decision(
                original_decision, regret_type, context
            )
            
            # 벤담 쾌락 계산
            bentham_scores = self._calculate_bentham_pleasure(
                original_decision, regret_decision, regret_type
            )
            
            scenarios.append({
                'regret_type': regret_type,
                'original_decision': original_decision,
                'regret_decision': regret_decision,
                'bentham_scores': bentham_scores,
                'regret_weight': self.regret_weights[regret_type]
            })
        
        # 부족한 경우 추가 시나리오 생성
        while len(scenarios) < self.config.regrets_per_step:
            scenarios.append(self._generate_additional_scenario(original_decision, context))
        
        return scenarios[:self.config.regrets_per_step]
    
    def _generate_regret_decision(self, original: torch.Tensor, 
                                regret_type: str, context: Dict[str, Any]) -> torch.Tensor:
        """후회 유형별 변형된 결정 생성"""
        batch_size = original.shape[0]
        
        if regret_type == 'counterfactual':
            # 반대 결정
            return -original + torch.randn_like(original) * 0.1
        elif regret_type == 'temporal':
            # 시간 지연된 결정
            return original * 0.7 + torch.randn_like(original) * 0.2
        elif regret_type == 'moral':
            # 도덕적으로 더 올바른 결정
            moral_bias = torch.ones_like(original) * 0.3
            return original + moral_bias + torch.randn_like(original) * 0.1
        elif regret_type == 'opportunity':
            # 더 적극적인 결정
            return original * 1.3 + torch.randn_like(original) * 0.15
        elif regret_type == 'social':
            # 사회적으로 더 바람직한 결정
            social_bias = torch.ones_like(original) * 0.2
            return original + social_bias + torch.randn_like(original) * 0.1
        else:
            return original + torch.randn_like(original) * 0.1
    
    def _calculate_bentham_pleasure(self, original: torch.Tensor, 
                                  regret: torch.Tensor, regret_type: str) -> Dict[str, torch.Tensor]:
        """벤담 쾌락 계산 (7가지 요소)"""
        batch_size = original.shape[0]
        scores = {}
        
        for factor, weight in self.bentham_factors.items():
            if factor == 'intensity':
                # 감정 강도
                scores[factor] = torch.abs(regret - original).mean(dim=-1) * weight
            elif factor == 'duration':
                # 지속성 (결정의 영향 지속도)
                scores[factor] = torch.sigmoid(torch.norm(regret, dim=-1)) * weight
            elif factor == 'certainty':
                # 확실성 (결정의 확신도)
                scores[factor] = (1.0 - torch.std(regret, dim=-1)) * weight
            elif factor == 'propinquity':
                # 근접성 (즉시성)
                scores[factor] = torch.exp(-torch.norm(regret - original, dim=-1)) * weight
            elif factor == 'fecundity':
                # 다산성 (추가 즐거움 생성)
                scores[factor] = torch.relu(regret.mean(dim=-1)) * weight
            elif factor == 'purity':
                # 순수성 (고통 없는 즐거움)
                scores[factor] = torch.sigmoid(regret.mean(dim=-1)) * weight
            elif factor == 'extent':
                # 범위 (영향받는 사람 수)
                scores[factor] = torch.tanh(torch.norm(regret, dim=-1)) * weight
        
        return scores
    
    def _generate_additional_scenario(self, original: torch.Tensor, 
                                    context: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """추가 시나리오 생성"""
        regret_decision = original + torch.randn_like(original) * 0.2
        bentham_scores = self._calculate_bentham_pleasure(original, regret_decision, 'additional')
        
        return {
            'regret_type': 'additional',
            'original_decision': original,
            'regret_decision': regret_decision,
            'bentham_scores': bentham_scores,
            'regret_weight': 0.1
        }

class RegretDataset(Dataset):
    """후회 기반 데이터셋"""
    
    def __init__(self, data_files: List[Path], config: RegretTrainingConfig):
        self.config = config
        self.scenarios = []
        
        # 데이터 로드
        for file_path in data_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.scenarios.extend(data)
                else:
                    self.scenarios.append(data)
        
        print(f"✅ 총 {len(self.scenarios)}개 시나리오 로드됨")
    
    def __len__(self):
        return len(self.scenarios)
    
    def __getitem__(self, idx):
        scenario = self.scenarios[idx]
        
        # 텍스트를 임베딩으로 변환 (단순화)
        description = scenario.get('description', '')
        
        # 임베딩 시뮬레이션 (메가스케일 모델 입력 차원에 맞춤)
        embedding = torch.randn(1024)  # 메가스케일 모델 입력 차원
        
        # 라벨 준비
        options = scenario.get('options', [])
        if len(options) >= 3:
            # approve, disapprove, neutral
            labels = torch.tensor([0.5, 0.3, 0.2])  # 예시 분포
        else:
            labels = torch.tensor([1.0, 0.0, 0.0])
        
        return {
            'text_embedding': embedding,
            'labels': labels,
            'scenario_id': scenario.get('id', f'scenario_{idx}'),
            'category': scenario.get('category', 'general'),
            'complexity': scenario.get('complexity_score', 0.5)
        }

class StorageMonitor:
    """스토리지 모니터링"""
    
    def __init__(self, max_gb: float, base_dir: Path):
        self.max_bytes = max_gb * 1024 * 1024 * 1024
        self.base_dir = base_dir
        
    def get_directory_size(self, directory: Path) -> int:
        """디렉토리 크기 계산"""
        total = 0
        for path in directory.rglob('*'):
            if path.is_file():
                total += path.stat().st_size
        return total
    
    def cleanup_if_needed(self):
        """필요시 정리"""
        current_size = self.get_directory_size(self.base_dir)
        
        if current_size > self.max_bytes:
            print(f"⚠️ 스토리지 한계 초과: {current_size / 1024**3:.2f}GB")
            
            # 오래된 로그 파일 삭제
            log_files = list(self.base_dir.glob('**/*.log'))
            log_files.sort(key=lambda x: x.stat().st_mtime)
            
            for log_file in log_files[:len(log_files)//2]:
                log_file.unlink()
                print(f"🗑️ 삭제됨: {log_file}")
    
    def get_size_gb(self) -> float:
        """현재 크기 GB 반환"""
        return self.get_directory_size(self.base_dir) / (1024**3)

class RegretTrainer:
    """후회 기반 학습기"""
    
    def __init__(self, config: RegretTrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 디렉토리 설정
        self.output_dir = project_root / 'training' / 'outputs'
        self.logs_dir = self.output_dir / 'logs'
        self.checkpoints_dir = self.output_dir / 'checkpoints'
        self.reports_dir = self.output_dir / 'reports'
        
        for dir_path in [self.output_dir, self.logs_dir, self.checkpoints_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 스토리지 모니터
        self.storage_monitor = StorageMonitor(config.max_storage_gb, self.output_dir)
        
        # 후회 계산기
        self.regret_calculator = RegretCalculator(config)
        
        # 학습 통계
        self.training_stats = defaultdict(list)
        self.step_count = 0
        
        # 로깅 설정
        self.setup_logging()
        
    def setup_logging(self):
        """로깅 설정"""
        log_file = self.logs_dir / f'regret_training_{int(time.time())}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('RegretTrainer')
    
    def prepare_models(self):
        """모델 준비"""
        self.logger.info("🤖 모델 준비 중...")
        
        # 메가 스케일 모델
        self.main_model = create_mega_scale_model(target_params=self.config.model_params)
        self.main_model = optimize_model_for_inference(self.main_model)
        self.main_model.to(self.device)
        
        # 감정 모델
        self.emotion_model = HierarchicalEmotionModel()
        self.emotion_model.to(self.device)
        
        # 옵티마이저
        self.optimizer = optim.AdamW(
            list(self.main_model.parameters()) + list(self.emotion_model.parameters()),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        # 스케줄러
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=1e-6
        )
        
        self.logger.info(f"✅ 모델 준비 완료: {self.main_model.get_parameter_count():,}개 파라미터")
    
    def prepare_data(self) -> DataLoader:
        """데이터 준비"""
        self.logger.info("📊 데이터 준비 중...")
        
        # 데이터 파일 찾기
        data_dir = project_root / 'processed_datasets'
        batch_files = list(data_dir.glob('full_scenarios_batch_*.json'))
        
        if not batch_files:
            raise FileNotFoundError("배치 데이터 파일을 찾을 수 없습니다.")
        
        # 데이터셋 생성
        dataset = RegretDataset(batch_files, self.config)
        
        # 데이터로더 생성
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        self.logger.info(f"✅ 데이터 준비 완료: {len(dataset)}개 시나리오")
        return dataloader
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """단일 학습 스텝"""
        self.main_model.train()
        self.emotion_model.train()
        
        text_embeddings = batch['text_embedding'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # 메인 모델 순전파
        main_outputs = self.main_model(text_embeddings.unsqueeze(1))
        
        # 감정 모델 순전파 (768차원으로 변환)
        emotion_embeddings = text_embeddings[:, :768] if text_embeddings.shape[1] > 768 else F.pad(text_embeddings, (0, 768 - text_embeddings.shape[1]))
        emotion_outputs = self.emotion_model(emotion_embeddings)
        
        # 후회 시나리오 생성 (7개)
        regret_scenarios = self.regret_calculator.calculate_regret_scenarios(
            text_embeddings, {
                'categories': batch['category'],
                'complexity': batch['complexity']
            }
        )
        
        total_loss = 0.0
        regret_losses = []
        bentham_scores = []
        
        # 각 후회 시나리오별 손실 계산
        for scenario in regret_scenarios:
            regret_decision = scenario['regret_decision'].to(self.device)
            
            # 후회 결정에 대한 모델 출력
            regret_outputs = self.main_model(regret_decision.unsqueeze(1))
            
            # 벤담 쾌락 계산 (3번씩)
            for i in range(self.config.bentham_calculations_per_regret):
                bentham_loss = self._calculate_bentham_loss(
                    main_outputs, regret_outputs, scenario['bentham_scores']
                )
                total_loss += bentham_loss * scenario['regret_weight']
                bentham_scores.append(bentham_loss.item())
        
        # 기본 분류 손실 (차원 맞춤)
        emotion_predictions = emotion_outputs['final_emotion']  # [batch_size, 6]
        emotion_avg = emotion_predictions.mean(dim=1, keepdim=True).expand(-1, 3)  # [batch_size, 3]으로 확장
        classification_loss = nn.MSELoss()(emotion_avg, labels)
        total_loss += classification_loss
        
        # 역전파
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(
            list(self.main_model.parameters()) + list(self.emotion_model.parameters()),
            self.config.max_gradient_norm
        )
        
        self.optimizer.step()
        self.scheduler.step()
        
        return {
            'total_loss': total_loss.item(),
            'classification_loss': classification_loss.item(),
            'regret_count': len(regret_scenarios),
            'bentham_count': len(bentham_scores),
            'avg_bentham_score': np.mean(bentham_scores) if bentham_scores else 0.0,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    def _calculate_bentham_loss(self, original_outputs: Dict[str, torch.Tensor],
                              regret_outputs: Dict[str, torch.Tensor],
                              bentham_scores: Dict[str, torch.Tensor]) -> torch.Tensor:
        """벤담 기반 손실 계산 (NaN 방지 안정성 개선)"""
        # 감정 예측 차이 (NaN 검사 추가)
        original_emotion = original_outputs.get('emotion_predictions', torch.zeros(1))
        regret_emotion = regret_outputs.get('emotion_predictions', torch.zeros(1))
        
        if torch.isnan(original_emotion).any() or torch.isnan(regret_emotion).any():
            logger.warning("감정 예측의 NaN 값 발견, 기본값 사용")
            emotion_diff = torch.tensor(0.1, dtype=torch.float32, requires_grad=True)
        else:
            emotion_diff = torch.abs(original_emotion - regret_emotion).mean()
            emotion_diff = torch.clamp(emotion_diff, min=0.0, max=10.0)  # 범위 제한
        
        # 의미 예측 차이 (NaN 검사 추가)
        original_semantic = original_outputs.get('semantic_predictions', torch.zeros(1))
        regret_semantic = regret_outputs.get('semantic_predictions', torch.zeros(1))
        
        if torch.isnan(original_semantic).any() or torch.isnan(regret_semantic).any():
            logger.warning("의미 예측의 NaN 값 발견, 기본값 사용")
            semantic_diff = torch.tensor(0.1, dtype=torch.float32, requires_grad=True)
        else:
            semantic_diff = torch.abs(original_semantic - regret_semantic).mean()
            semantic_diff = torch.clamp(semantic_diff, min=0.0, max=10.0)  # 범위 제한
        
        # 벤담 점수 가중합 (NaN 방지)
        try:
            bentham_values = list(bentham_scores.values())
            if bentham_values and not any(torch.isnan(v).any() if torch.is_tensor(v) else math.isnan(v) for v in bentham_values):
                bentham_weight = torch.stack([torch.tensor(v) if not torch.is_tensor(v) else v for v in bentham_values]).mean()
                bentham_weight = torch.clamp(bentham_weight, min=0.1, max=5.0)  # 범위 제한
            else:
                bentham_weight = torch.tensor(1.0, dtype=torch.float32)
        except Exception as e:
            logger.warning(f"벤담 가중치 계산 오류: {e}, 기본값 사용")
            bentham_weight = torch.tensor(1.0, dtype=torch.float32)
        
        # 최종 손실 계산 및 NaN 방지
        final_loss = (emotion_diff + semantic_diff) * bentham_weight
        final_loss = torch.clamp(final_loss, min=0.0, max=100.0)  # 범위 제한
        
        if torch.isnan(final_loss) or torch.isinf(final_loss):
            logger.error(f"벤담 손실 계산 결과가 비정상: {final_loss}")
            return torch.tensor(1.0, dtype=torch.float32, requires_grad=True)
            
        return final_loss
    
    def train_epoch(self, dataloader: DataLoader, epoch: int):
        """에포크 학습"""
        self.logger.info(f"🎯 에포크 {epoch+1}/{self.config.epochs} 시작")
        
        epoch_stats = defaultdict(list)
        
        for batch_idx, batch in enumerate(dataloader):
            step_stats = self.train_step(batch)
            
            # 통계 수집
            for key, value in step_stats.items():
                epoch_stats[key].append(value)
                self.training_stats[key].append(value)
            
            self.step_count += 1
            
            # 주기적 로깅
            if self.step_count % self.config.log_every_n_steps == 0:
                avg_loss = np.mean(epoch_stats['total_loss'][-self.config.log_every_n_steps:])
                avg_regret = np.mean(epoch_stats['regret_count'][-self.config.log_every_n_steps:])
                avg_bentham = np.mean(epoch_stats['avg_bentham_score'][-self.config.log_every_n_steps:])
                
                self.logger.info(
                    f"스텝 {self.step_count}: 손실={avg_loss:.4f}, "
                    f"후회={avg_regret:.1f}, 벤담={avg_bentham:.4f}, "
                    f"스토리지={self.storage_monitor.get_size_gb():.1f}GB"
                )
                
                # XAI 로깅
                with xai_logger.trace_operation("regret_training", f"step_{self.step_count}") as op_id:
                    xai_logger.log_llm_interaction(
                        operation_id=op_id,
                        prompt=f"Step {self.step_count} training metrics",
                        response=f"Loss: {avg_loss:.4f}, Regrets: {avg_regret}",
                        model_name="regret_trainer",
                        tokens_used=len(str(step_stats))
                    )
            
            # 체크포인트 저장
            if self.step_count % self.config.save_checkpoint_every == 0:
                self.save_checkpoint(epoch, batch_idx)
            
            # 스토리지 모니터링
            if self.step_count % 50 == 0:
                self.storage_monitor.cleanup_if_needed()
            
            # 메모리 정리
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
        
        # 에포크 요약
        avg_epoch_loss = np.mean(epoch_stats['total_loss'])
        total_regrets = sum(epoch_stats['regret_count'])
        total_benthams = sum(epoch_stats['bentham_count'])
        
        self.logger.info(
            f"✅ 에포크 {epoch+1} 완료: 평균 손실={avg_epoch_loss:.4f}, "
            f"총 후회={total_regrets}, 총 벤담 계산={total_benthams}"
        )
    
    def save_checkpoint(self, epoch: int, batch_idx: int):
        """체크포인트 저장"""
        checkpoint_path = self.checkpoints_dir / f'regret_model_epoch_{epoch}_step_{self.step_count}.pth'
        
        checkpoint = {
            'epoch': epoch,
            'step': self.step_count,
            'batch_idx': batch_idx,
            'main_model_state_dict': self.main_model.state_dict(),
            'emotion_model_state_dict': self.emotion_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': asdict(self.config),
            'training_stats': dict(self.training_stats),
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"💾 체크포인트 저장: {checkpoint_path}")
    
    def generate_training_report(self) -> Dict[str, Any]:
        """학습 리포트 생성"""
        total_steps = len(self.training_stats['total_loss'])
        total_regrets = sum(self.training_stats['regret_count'])
        total_benthams = sum(self.training_stats['bentham_count'])
        
        report = {
            'training_summary': {
                'total_steps': total_steps,
                'total_regrets': total_regrets,
                'total_bentham_calculations': total_benthams,
                'average_regrets_per_step': total_regrets / total_steps if total_steps > 0 else 0,
                'average_benthams_per_step': total_benthams / total_steps if total_steps > 0 else 0,
                'final_loss': self.training_stats['total_loss'][-1] if self.training_stats['total_loss'] else 0,
                'training_duration': time.time()
            },
            'model_info': {
                'main_model_parameters': self.main_model.get_parameter_count(),
                'target_parameters': self.config.model_params,
                'device': str(self.device)
            },
            'configuration': asdict(self.config),
            'storage_usage': {
                'final_size_gb': self.storage_monitor.get_size_gb(),
                'max_allowed_gb': self.config.max_storage_gb
            },
            'xai_integration': {
                'xai_logs_generated': len(xai_logger.logs),
                'session_id': xai_logger.session_id
            }
        }
        
        return report
    
    def train(self):
        """전체 학습 프로세스"""
        start_time = time.time()
        
        self.logger.info("🚀 후회 기반 학습 시작")
        self.logger.info(f"📊 설정: {self.config.regrets_per_step}회 후회/스텝, {self.config.epochs}번 선회")
        
        # 모델 및 데이터 준비
        self.prepare_models()
        dataloader = self.prepare_data()
        
        # 학습 실행
        for epoch in range(self.config.epochs):
            self.train_epoch(dataloader, epoch)
            
            # 에포크별 체크포인트
            self.save_checkpoint(epoch, -1)
        
        # 최종 저장
        final_checkpoint_path = self.checkpoints_dir / 'final_regret_model.pth'
        torch.save({
            'main_model_state_dict': self.main_model.state_dict(),
            'emotion_model_state_dict': self.emotion_model.state_dict(),
            'config': asdict(self.config),
            'training_stats': dict(self.training_stats),
            'timestamp': datetime.now().isoformat()
        }, final_checkpoint_path)
        
        # 학습 리포트 생성
        training_time = time.time() - start_time
        report = self.generate_training_report()
        report['training_summary']['training_duration'] = training_time
        
        # 리포트 저장
        report_path = self.reports_dir / f'regret_training_report_{int(time.time())}.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"🎉 학습 완료! 총 시간: {training_time/3600:.2f}시간")
        self.logger.info(f"📊 총 후회: {sum(self.training_stats['regret_count'])}")
        self.logger.info(f"📊 총 벤담 계산: {sum(self.training_stats['bentham_count'])}")
        self.logger.info(f"📋 리포트: {report_path}")
        
        return report, final_checkpoint_path

def create_training_config(**kwargs) -> RegretTrainingConfig:
    """학습 설정 생성 헬퍼"""
    return RegretTrainingConfig(**kwargs)

if __name__ == "__main__":
    # 기본 설정으로 테스트
    config = RegretTrainingConfig()
    trainer = RegretTrainer(config)
    
    print("🧪 후회 기반 학습 파이프라인 준비 완료")
    print(f"📊 설정: {config.regrets_per_step}회 후회/스텝, {config.epochs}번 선회")
    print(f"💾 스토리지 한계: {config.max_storage_gb}GB")
    print("준비된 학습을 시작하려면 trainer.train()을 호출하세요.")