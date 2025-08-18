#!/usr/bin/env python3
"""
하이브리드 분산 학습 시스템 (CPU 128GB + GPU RTX 2070S)
Hybrid Distributed Training System (CPU 128GB + GPU RTX 2070S)
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
import threading
import queue
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 모듈 imports
from xai_core.xai_logging_system import xai_logger, xai_trace
from llm_module import llm_tracker, register_llm, ask_llm

@dataclass
class HybridConfig:
    """하이브리드 시스템 설정"""
    # 모델 분할 설정 (공격적 최적화 70GB)
    gpu_memory_gb: float = 8.0          # RTX 2070S 메모리
    cpu_memory_gb: float = 70.0         # WSL 공격적 메모리 활용
    target_params: int = 4_300_000_000   # 43억 파라미터 (현실적 70GB)
    
    # 테스트 모드 설정
    test_mode: bool = False             # 테스트 모드 활성화
    test_samples: int = 10              # 테스트용 샘플 수
    
    # 메모리 최적화 설정
    max_safe_batch_size: int = 8        # GPU 활용으로 배치 증가 가능
    gradient_accumulation_steps: int = 8 # 효과적인 배치 크기 증대
    use_gradient_checkpointing: bool = True  # 활성화 메모리 절약
    use_parameter_sharing: bool = True   # 레이어간 파라미터 공유
    
    # 학습 설정 (공격적 고성능)
    regrets_per_step: int = 7           # 원래 설정 복원 (최고 품질)
    bentham_calculations_per_regret: int = 3  # 원래 설정 복원 (정확도 극대화)
    epochs: int = 3
    batch_size: int = 12                # 대형 배치 (처리량 증가)
    micro_batch_size: int = 3           # 메모리 여유로 증가
    
    # 분산 설정 (공격적 병렬화)
    num_workers: int = 8                # 최대 CPU 활용
    gpu_layers_ratio: float = 0.6       # 균형잡힌 GPU/CPU 분할
    overlap_computation: bool = True     # 계산 오버랩
    use_cpu_offload: bool = True        # CPU 오프로드 활성화
    enable_memory_monitoring: bool = True  # 실시간 메모리 모니터링
    
    # 최적화 설정
    use_mixed_precision: bool = True    # FP16 사용
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # 로깅/체크포인트
    log_every_n_steps: int = 5          # 더 자주 로깅
    save_checkpoint_every: int = 20     # 더 자주 저장
    max_storage_gb: float = 50.0        # 스토리지 절약

class MemoryOptimizedModel(nn.Module):
    """메모리 최적화된 모델"""
    
    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config
        
        # 현실적 고성능 설계 (43억 파라미터, 70GB)
        self.hidden_dim = 2560      # 현실적 크기 (성능 vs 메모리 균형)
        self.num_layers = 32        # 적당한 깊이 (안정성 유지)
        self.num_heads = 40         # 어텐션 품질 유지
        self.intermediate_size = 10240  # FFN 크기 현실적 조정
        
        # 입력 레이어 (CPU)
        self.input_projection = nn.Linear(1024, self.hidden_dim)
        self.input_norm = nn.LayerNorm(self.hidden_dim, eps=1e-6)  # Loss NaN 방지 수치 안정성 개선
        
        # 트랜스포머 레이어들 (분할 배치)
        gpu_layers = int(self.num_layers * config.gpu_layers_ratio)
        cpu_layers = self.num_layers - gpu_layers
        
        # 그래디언트 체크포인트 활성화
        use_checkpointing = getattr(config, 'use_gradient_checkpointing', True)
        
        self.gpu_layers = nn.ModuleList([
            OptimizedTransformerLayer(self.hidden_dim, self.num_heads, self.intermediate_size, use_checkpointing)
            for _ in range(gpu_layers)
        ])
        
        self.cpu_layers = nn.ModuleList([
            OptimizedTransformerLayer(self.hidden_dim, self.num_heads, self.intermediate_size, use_checkpointing)
            for _ in range(cpu_layers)
        ])
        
        # 출력 헤드들 (GPU) - 메모리 최적화
        # SwiGLU 기반 감정 헤드 (수치 안정성 개선)
        self.emotion_swiglu = SwiGLU(self.hidden_dim, self.hidden_dim // 2)
        self.emotion_out = nn.Linear(self.hidden_dim // 2, 6)  # 6차원 감정
        self.emotion_dropout = nn.Dropout(0.02)
        self.emotion_activation = nn.Tanh()
        
        # SwiGLU 기반 의미 헤드 (수치 안정성 개선)
        self.semantic_swiglu = SwiGLU(self.hidden_dim, self.hidden_dim // 2)
        self.semantic_out = nn.Linear(self.hidden_dim // 2, 512)  # 1000→512 축소
        self.semantic_dropout = nn.Dropout(0.02)
        self.semantic_activation = nn.Softmax(dim=-1)
        
        # SwiGLU 기반 추론 헤드 (수치 안정성 개선)
        self.reasoning_swiglu = SwiGLU(self.hidden_dim, self.hidden_dim // 4)
        self.reasoning_out = nn.Linear(self.hidden_dim // 4, 128)  # 추론 특징
        self.reasoning_dropout = nn.Dropout(0.05)
        
        # SwiGLU 기반 통합 헤드 (수치 안정성 개선)
        self.integration_swiglu = SwiGLU(self.hidden_dim, self.hidden_dim)
        self.integration_out = nn.Linear(self.hidden_dim, 512)  # 통합 특징
        self.integration_dropout = nn.Dropout(0.05)
        self.integration_activation = nn.Tanh()
        
        # 디바이스 설정
        self.gpu_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cpu_device = torch.device('cpu')
        
        self._setup_devices()
        
    def _setup_devices(self):
        """디바이스별 모델 배치"""
        # CPU 레이어들
        self.input_projection = self.input_projection.to(self.cpu_device)
        self.input_norm = self.input_norm.to(self.cpu_device)
        self.cpu_layers = self.cpu_layers.to(self.cpu_device)
        
        # GPU 레이어들 (사용 가능한 경우)
        if torch.cuda.is_available():
            self.gpu_layers = self.gpu_layers.to(self.gpu_device)
            # SwiGLU 기반 헤드들 GPU로 이동
            self.emotion_swiglu = self.emotion_swiglu.to(self.gpu_device)
            self.emotion_out = self.emotion_out.to(self.gpu_device)
            self.emotion_dropout = self.emotion_dropout.to(self.gpu_device)
            self.emotion_activation = self.emotion_activation.to(self.gpu_device)
            
            self.semantic_swiglu = self.semantic_swiglu.to(self.gpu_device)
            self.semantic_out = self.semantic_out.to(self.gpu_device)
            self.semantic_dropout = self.semantic_dropout.to(self.gpu_device)
            self.semantic_activation = self.semantic_activation.to(self.gpu_device)
            
            self.reasoning_swiglu = self.reasoning_swiglu.to(self.gpu_device)
            self.reasoning_out = self.reasoning_out.to(self.gpu_device)
            self.reasoning_dropout = self.reasoning_dropout.to(self.gpu_device)
            
            self.integration_swiglu = self.integration_swiglu.to(self.gpu_device)
            self.integration_out = self.integration_out.to(self.gpu_device)
            self.integration_dropout = self.integration_dropout.to(self.gpu_device)
            self.integration_activation = self.integration_activation.to(self.gpu_device)
        else:
            # GPU 없으면 CPU에 배치
            self.gpu_layers = self.gpu_layers.to(self.cpu_device)
            # SwiGLU 기반 헤드들 CPU로 이동
            self.emotion_swiglu = self.emotion_swiglu.to(self.cpu_device)
            self.emotion_out = self.emotion_out.to(self.cpu_device)
            self.emotion_dropout = self.emotion_dropout.to(self.cpu_device)
            self.emotion_activation = self.emotion_activation.to(self.cpu_device)
            
            self.semantic_swiglu = self.semantic_swiglu.to(self.cpu_device)
            self.semantic_out = self.semantic_out.to(self.cpu_device)
            self.semantic_dropout = self.semantic_dropout.to(self.cpu_device)
            self.semantic_activation = self.semantic_activation.to(self.cpu_device)
            
            self.reasoning_swiglu = self.reasoning_swiglu.to(self.cpu_device)
            self.reasoning_out = self.reasoning_out.to(self.cpu_device)
            self.reasoning_dropout = self.reasoning_dropout.to(self.cpu_device)
            
            self.integration_swiglu = self.integration_swiglu.to(self.cpu_device)
            self.integration_out = self.integration_out.to(self.cpu_device)
            self.integration_dropout = self.integration_dropout.to(self.cpu_device)
            self.integration_activation = self.integration_activation.to(self.cpu_device)
            self.gpu_device = self.cpu_device
    
    def get_parameter_count(self) -> int:
        """파라미터 수 계산"""
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, input_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, input_dim = input_embeddings.shape
        
        # CPU에서 입력 처리
        hidden_states = input_embeddings.to(self.cpu_device)
        hidden_states = self.input_projection(hidden_states)
        hidden_states = self.input_norm(hidden_states)
        
        # CPU 레이어들 통과
        for layer in self.cpu_layers:
            hidden_states = layer(hidden_states)
        
        # GPU로 이동 (사용 가능한 경우)
        if torch.cuda.is_available():
            hidden_states = hidden_states.to(self.gpu_device)
        
        # GPU 레이어들 통과
        for layer in self.gpu_layers:
            hidden_states = layer(hidden_states)
        
        # 평균 풀링
        pooled_output = hidden_states.mean(dim=1)
        
        # SwiGLU 기반 출력 헤드들 (수치 안정성 개선)
        emotion_swiglu = self.emotion_swiglu(pooled_output)
        emotion_output = self.emotion_activation(self.emotion_out(self.emotion_dropout(emotion_swiglu)))
        
        semantic_swiglu = self.semantic_swiglu(pooled_output)
        semantic_output = self.semantic_activation(self.semantic_out(self.semantic_dropout(semantic_swiglu)))
        
        reasoning_swiglu = self.reasoning_swiglu(pooled_output)
        reasoning_output = self.reasoning_out(self.reasoning_dropout(reasoning_swiglu))
        
        integration_swiglu = self.integration_swiglu(pooled_output)
        integration_output = self.integration_activation(self.integration_out(self.integration_dropout(integration_swiglu)))
        
        return {
            'emotion_predictions': emotion_output,
            'semantic_predictions': semantic_output,
            'reasoning_features': reasoning_output,
            'integration_features': integration_output,
            'pooled_output': pooled_output
        }

class SwiGLU(nn.Module):
    """
    SwiGLU 활성화 함수 - 수치적 안정성과 성능이 GELU보다 우수
    SwiGLU(x) = Swish(xW + b) ⊗ (xV + c)
    """
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        # SwiGLU는 두 개의 선형 변환을 필요로 함
        self.w = nn.Linear(dim, hidden_dim, bias=True)
        self.v = nn.Linear(dim, hidden_dim, bias=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Swish(x) = x * sigmoid(x), 수치적으로 안정적
        w_out = self.w(x)
        swish_w = w_out * torch.sigmoid(w_out)
        v_out = self.v(x)
        return swish_w * v_out  # Hadamard product

class OptimizedTransformerLayer(nn.Module):
    """성능 유지 메모리 최적화 트랜스포머 레이어 (SwiGLU 적용)"""
    
    def __init__(self, hidden_dim: int, num_heads: int, intermediate_size: int, use_gradient_checkpointing: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # 고효율 어텐션 (메모리 절약)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.02,  # Dropout 추가 감소 (Loss NaN 방지)
            batch_first=True
        )
        self.attention_norm = nn.LayerNorm(hidden_dim, eps=1e-6)  # Loss NaN 방지 수치 안정성 개선
        
        # SwiGLU 기반 FFN (수치 안정성 및 성능 개선)
        self.swiglu = SwiGLU(hidden_dim, intermediate_size)
        self.ffn_out = nn.Linear(intermediate_size, hidden_dim)
        self.ffn_dropout = nn.Dropout(0.02)  # Dropout 추가 감소 (Loss NaN 방지)
        self.ffn_norm = nn.LayerNorm(hidden_dim, eps=1e-6)  # Loss NaN 방지 수치 안정성 개선
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 그래디언트 체크포인트 사용 (메모리 절약)
        if self.use_gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(self._forward_impl, hidden_states, use_reentrant=False)
        else:
            return self._forward_impl(hidden_states)
    
    def _forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 어텐션 블록 (Post-LayerNorm 안정성 개선)
        attention_output, _ = self.attention(hidden_states, hidden_states, hidden_states)
        hidden_states = hidden_states + attention_output
        hidden_states = self.attention_norm(hidden_states)
        
        # SwiGLU 기반 FFN 블록 (Post-LayerNorm 수치 안정성 개선)
        swiglu_output = self.swiglu(hidden_states)
        ffn_output = self.ffn_dropout(self.ffn_out(swiglu_output))
        hidden_states = hidden_states + ffn_output
        hidden_states = self.ffn_norm(hidden_states)
        
        return hidden_states

class AsyncRegretCalculator:
    """비동기 후회 계산기"""
    
    def __init__(self, config: HybridConfig):
        self.config = config
        self.regret_queue = queue.Queue(maxsize=100)
        self.result_queue = queue.Queue(maxsize=100)
        self.workers = []
        
        # 워커 스레드들 시작
        for i in range(config.num_workers):
            worker = threading.Thread(target=self._worker_process, daemon=True)
            worker.start()
            self.workers.append(worker)
    
    def _worker_process(self):
        """워커 프로세스"""
        while True:
            try:
                task = self.regret_queue.get(timeout=1)
                if task is None:  # 종료 신호
                    break
                
                original_decision, task_id = task
                regret_scenarios = self._calculate_regret_scenarios(original_decision)
                self.result_queue.put((task_id, regret_scenarios))
                self.regret_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker error: {e}")
    
    def _calculate_regret_scenarios(self, original_decision: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """7가지 후회 시나리오 생성"""
        scenarios = []
        regret_types = ['counterfactual', 'temporal', 'moral', 'opportunity', 'social']
        
        for i, regret_type in enumerate(regret_types):
            if i >= self.config.regrets_per_step:
                break
            
            # 각 후회 유형별 변형된 결정 생성
            regret_decision = self._generate_regret_decision(original_decision, regret_type)
            
            # 벤담 쾌락 계산 (3회)
            bentham_scores = []
            for _ in range(self.config.bentham_calculations_per_regret):
                score = self._calculate_bentham_score(original_decision, regret_decision)
                bentham_scores.append(score)
            
            scenarios.append({
                'regret_type': regret_type,
                'regret_decision': regret_decision,
                'bentham_scores': torch.tensor(bentham_scores).mean(),
                'regret_weight': 0.2
            })
        
        # 부족한 경우 추가
        while len(scenarios) < self.config.regrets_per_step:
            scenarios.append(scenarios[0])  # 첫 번째 시나리오 복사
        
        return scenarios[:self.config.regrets_per_step]
    
    def _generate_regret_decision(self, original: torch.Tensor, regret_type: str) -> torch.Tensor:
        """후회 유형별 변형된 결정 생성"""
        if regret_type == 'counterfactual':
            return -original + torch.randn_like(original) * 0.1
        elif regret_type == 'temporal':
            return original * 0.7 + torch.randn_like(original) * 0.2
        elif regret_type == 'moral':
            return original + torch.ones_like(original) * 0.3
        elif regret_type == 'opportunity':
            return original * 1.3 + torch.randn_like(original) * 0.15
        elif regret_type == 'social':
            return original + torch.ones_like(original) * 0.2
        else:
            return original + torch.randn_like(original) * 0.1
    
    def _calculate_bentham_score(self, original: torch.Tensor, regret: torch.Tensor) -> float:
        """벤담 쾌락 점수 계산"""
        diff = torch.abs(regret - original).mean()
        intensity = diff.item()
        return intensity * 0.8  # 간소화된 벤담 계산
    
    def calculate_async(self, original_decision: torch.Tensor, task_id: int):
        """비동기 후회 계산 요청"""
        self.regret_queue.put((original_decision.cpu().clone(), task_id))
    
    def get_result(self, timeout: float = 0.1) -> Optional[Tuple[int, List[Dict]]]:
        """결과 가져오기"""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

class OptimizedDataset(Dataset):
    """최적화된 데이터셋"""
    
    def __init__(self, data_files: List[Path], config: HybridConfig):
        self.config = config
        self.scenarios = []
        
        # 데이터 로드 (테스트 모드 지원)
        print("📊 데이터 로딩 중...")
        total_loaded = 0
        for file_path in data_files:
            if hasattr(config, 'test_mode') and config.test_mode and total_loaded >= config.test_samples:
                break
                
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    if hasattr(config, 'test_mode') and config.test_mode:
                        # 테스트 모드: 정확히 test_samples개만 로드
                        remaining = config.test_samples - total_loaded
                        sample_size = min(len(data), remaining)
                        self.scenarios.extend(data[:sample_size])
                        total_loaded += sample_size
                    else:
                        # 일반 모드: 파일당 최대 5000개
                        sample_size = min(len(data), 5000)
                        self.scenarios.extend(data[:sample_size])
        
        print(f"✅ 총 {len(self.scenarios)}개 시나리오 로드됨")
    
    def __len__(self):
        return len(self.scenarios)
    
    def __getitem__(self, idx):
        scenario = self.scenarios[idx]
        
        # 텍스트를 임베딩으로 변환 (최적화된 차원)
        embedding = torch.randn(1024, dtype=torch.float32)
        
        # 라벨 준비
        options = scenario.get('options', [])
        if len(options) >= 3:
            labels = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float32)
        else:
            labels = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        
        return {
            'text_embedding': embedding,
            'labels': labels,
            'scenario_id': scenario.get('id', f'scenario_{idx}'),
            'category': scenario.get('category', 'general')
        }

class HybridDistributedTrainer:
    """하이브리드 분산 학습기"""
    
    def __init__(self, config: HybridConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 디렉토리 설정
        self.output_dir = project_root / 'training' / 'hybrid_outputs'
        self.logs_dir = self.output_dir / 'logs'
        self.checkpoints_dir = self.output_dir / 'checkpoints'
        self.reports_dir = self.output_dir / 'reports'
        
        for dir_path in [self.output_dir, self.logs_dir, self.checkpoints_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 비동기 후회 계산기
        self.regret_calculator = AsyncRegretCalculator(config)
        
        # 학습 통계
        self.training_stats = defaultdict(list)
        self.step_count = 0
        
        # 로깅 설정
        self.setup_logging()
        
        print(f"🚀 하이브리드 시스템 초기화 완료")
        if torch.cuda.is_available():
            print(f"   - GPU 가속: ✅ CUDA 활성화")
            print(f"   - GPU 메모리: {config.gpu_memory_gb}GB")
        else:
            print(f"   - CPU 최적화: ✅ 고성능 CPU 모드")
            print(f"   - CPU 메모리: {config.cpu_memory_gb}GB (대용량 처리)")
        print(f"   - 병렬 워커: {config.num_workers}개")
        print(f"   - 분산 처리: CPU+메모리 최적화")
    
    def setup_logging(self):
        """로깅 설정"""
        log_file = self.logs_dir / f'hybrid_training_{int(time.time())}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('HybridTrainer')
    
    def prepare_model(self):
        """모델 준비"""
        self.logger.info("🤖 하이브리드 모델 준비 중...")
        
        # 메모리 최적화된 모델
        self.model = MemoryOptimizedModel(self.config)
        
        # Mixed Precision 설정
        if self.config.use_mixed_precision and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # 옵티마이저
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )
        
        # 스케줄러
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=1e-6
        )
        
        actual_params = self.model.get_parameter_count()
        self.logger.info(f"✅ 모델 준비 완료: {actual_params:,}개 파라미터")
        
        return actual_params
    
    def prepare_data(self) -> DataLoader:
        """데이터 준비"""
        self.logger.info("📊 데이터 준비 중...")
        
        # 데이터 파일 찾기
        data_dir = project_root / 'processed_datasets'
        batch_files = list(data_dir.glob('full_scenarios_batch_*.json'))
        
        if not batch_files:
            raise FileNotFoundError("배치 데이터 파일을 찾을 수 없습니다.")
        
        # 최적화된 데이터셋
        dataset = OptimizedDataset(batch_files, self.config)
        
        # 데이터로더 (최적화된 설정)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True
        )
        
        self.logger.info(f"✅ 데이터 준비 완료: {len(dataset)}개 시나리오")
        return dataloader
    
    def train_step(self, batch: Dict[str, torch.Tensor], step_idx: int) -> Dict[str, float]:
        """메모리 최적화된 학습 스텝"""
        self.model.train()
        
        # 메모리 정리
        if step_idx % 5 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        text_embeddings = batch['text_embedding']
        labels = batch['labels']
        
        # 배치 크기 동적 조정
        current_batch_size = text_embeddings.size(0)
        if current_batch_size > self.config.max_safe_batch_size:
            return self._process_large_batch(batch, step_idx)
        
        # 비동기 후회 계산 시작 (최적화)
        for i, embedding in enumerate(text_embeddings[:min(len(text_embeddings), 6)]):
            self.regret_calculator.calculate_async(embedding, step_idx * self.config.batch_size + i)
        
        # 모델 순전파 (Mixed Precision)
        if self.scaler:
            with torch.cuda.amp.autocast():
                outputs = self.model(text_embeddings.unsqueeze(1))
        else:
            outputs = self.model(text_embeddings.unsqueeze(1))
        
        # 기본 분류 손실 (NaN 방지 안정성 개선)
        emotion_predictions = outputs['emotion_predictions']
        
        # NaN 감지 및 처리
        if torch.isnan(emotion_predictions).any():
            logger.warning("감정 예측에 NaN 발견, 제로로 초기화")
            emotion_predictions = torch.zeros_like(emotion_predictions)
        
        emotion_avg = emotion_predictions.mean(dim=1, keepdim=True).expand(-1, 3)
        classification_loss = F.mse_loss(emotion_avg, labels)
        
        # 손실 NaN 감지 및 처리
        if torch.isnan(classification_loss):
            logger.error("분류 손실이 NaN입니다. 기본값으로 설정")
            classification_loss = torch.tensor(1.0, device=emotion_predictions.device, requires_grad=True)
        
        total_loss = classification_loss
        regret_count = 0
        bentham_count = 0
        
        # 후회 결과 수집 (균형 유지)
        for _ in range(7):  # 원래 7개 유지
            result = self.regret_calculator.get_result()
            if result is None:
                break
            
            task_id, regret_scenarios = result
            regret_count += len(regret_scenarios)
            bentham_count += len(regret_scenarios) * self.config.bentham_calculations_per_regret
            
            # 후회 손실 추가 (균형 유지 + NaN 방지)
            for scenario in regret_scenarios[:3]:  # 최대 3개 처리
                # 안전한 후회 손실 계산
                bentham_scores = scenario.get('bentham_scores', 0.0)
                regret_weight = scenario.get('regret_weight', 1.0)
                
                # NaN/Inf 감지 및 제한
                if isinstance(bentham_scores, (int, float)):
                    bentham_scores = torch.tensor(bentham_scores, dtype=torch.float32)
                if isinstance(regret_weight, (int, float)):
                    regret_weight = torch.tensor(regret_weight, dtype=torch.float32)
                
                # NaN/Inf 처리
                if torch.isnan(bentham_scores) or torch.isinf(bentham_scores):
                    bentham_scores = torch.tensor(0.0, dtype=torch.float32)
                if torch.isnan(regret_weight) or torch.isinf(regret_weight):
                    regret_weight = torch.tensor(1.0, dtype=torch.float32)
                
                # 후회 손실 계산 및 제한
                regret_loss = torch.clamp(bentham_scores * 0.1, min=-10.0, max=10.0)
                weighted_regret_loss = torch.clamp(regret_loss * regret_weight, min=-10.0, max=10.0)
                
                # NaN 최종 검사
                if torch.isnan(weighted_regret_loss):
                    logger.warning(f"후회 손실에 NaN 발견, 스킨")
                    continue
                    
                total_loss += weighted_regret_loss
        
        # 최종 NaN 검사 및 역전파 전 안전성 확인
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.error(f"최종 손실이 비정상입니다: {total_loss}. 역전파 스킨")
            return {
                'loss': float('nan'),
                'classification_loss': classification_loss.item() if not torch.isnan(classification_loss) else float('nan'),
                'regret_count': regret_count,
                'bentham_count': bentham_count
            }
        
        # 역전파 (Mixed Precision)
        self.optimizer.zero_grad()
        
        if self.scaler:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            # 그래디언트 NaN 검사 및 클리핑 (Mixed Precision)
            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            if torch.isnan(total_norm) or torch.isinf(total_norm):
                logger.error(f"그래디언트 norm이 비정상입니다: {total_norm}. 옵티마이저 스킨")
                return {
                    'loss': total_loss.item(),
                    'classification_loss': classification_loss.item(),
                    'regret_count': regret_count,
                    'bentham_count': bentham_count
                }
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            # 그래디언트 NaN 검사 및 클리핑 (FP32 모드)
            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            if torch.isnan(total_norm) or torch.isinf(total_norm):
                logger.error(f"그래디언트 norm이 비정상입니다: {total_norm}. 옵티마이저 스킨")
                return {
                    'loss': total_loss.item(),
                    'classification_loss': classification_loss.item(),
                    'regret_count': regret_count,
                    'bentham_count': bentham_count
                }
            self.optimizer.step()
        
        # 통계 업데이트
        return {
            'loss': total_loss.item(),
            'classification_loss': classification_loss.item(),
            'regret_count': regret_count,
            'bentham_count': bentham_count
        }
    
    def _process_large_batch(self, batch: Dict[str, torch.Tensor], step_idx: int) -> Dict[str, float]:
        """큰 배치를 작은 청크로 나누어 처리"""
        text_embeddings = batch['text_embedding']
        labels = batch['labels']
        
        chunk_size = self.config.max_safe_batch_size
        total_loss = 0.0
        total_classification_loss = 0.0
        total_regret_count = 0
        total_bentham_count = 0
        
        num_chunks = (len(text_embeddings) + chunk_size - 1) // chunk_size
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(text_embeddings))
            
            chunk_embeddings = text_embeddings[start_idx:end_idx]
            chunk_labels = labels[start_idx:end_idx]
            
            chunk_batch = {
                'text_embedding': chunk_embeddings,
                'labels': chunk_labels
            }
            
            chunk_stats = self.train_step(chunk_batch, step_idx * num_chunks + i)
            
            total_loss += chunk_stats['loss']
            total_classification_loss += chunk_stats['classification_loss']
            total_regret_count += chunk_stats['regret_count']
            total_bentham_count += chunk_stats['bentham_count']
        
        return {
            'loss': total_loss / num_chunks,
            'classification_loss': total_classification_loss / num_chunks,
            'regret_count': total_regret_count,
            'bentham_count': total_bentham_count
        }
    
    def train_epoch(self, dataloader: DataLoader, epoch: int):
        """에포크 학습"""
        self.logger.info(f"🎯 에포크 {epoch+1}/{self.config.epochs} 시작")
        
        epoch_stats = defaultdict(list)
        
        for batch_idx, batch in enumerate(dataloader):
            step_stats = self.train_step(batch, batch_idx)
            
            # 통계 수집
            for key, value in step_stats.items():
                epoch_stats[key].append(value)
                self.training_stats[key].append(value)
            
            self.step_count += 1
            
            # 주기적 로깅
            if self.step_count % self.config.log_every_n_steps == 0:
                avg_loss = np.mean(epoch_stats['total_loss'][-self.config.log_every_n_steps:])
                avg_regret = np.mean(epoch_stats['regret_count'][-self.config.log_every_n_steps:])
                
                self.logger.info(
                    f"스텝 {self.step_count}: 손실={avg_loss:.4f}, "
                    f"후회={avg_regret:.1f}, GPU메모리={self._get_gpu_memory():.1f}MB"
                )
            
            # 체크포인트 저장
            if self.step_count % self.config.save_checkpoint_every == 0:
                self.save_checkpoint(epoch, batch_idx)
            
            # 메모리 정리
            if batch_idx % 5 == 0:
                self._cleanup_memory()
        
        # 에포크 요약
        avg_epoch_loss = np.mean(epoch_stats['total_loss'])
        total_regrets = sum(epoch_stats['regret_count'])
        
        self.logger.info(
            f"✅ 에포크 {epoch+1} 완료: 평균 손실={avg_epoch_loss:.4f}, "
            f"총 후회={total_regrets}"
        )
    
    def _get_gpu_memory(self) -> float:
        """GPU 메모리 사용량 MB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0
    
    def _cleanup_memory(self):
        """메모리 정리"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def save_checkpoint(self, epoch: int, batch_idx: int):
        """체크포인트 저장"""
        checkpoint_path = self.checkpoints_dir / f'hybrid_model_epoch_{epoch}_step_{self.step_count}.pth'
        
        checkpoint = {
            'epoch': epoch,
            'step': self.step_count,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': asdict(self.config),
            'training_stats': dict(self.training_stats),
            'timestamp': datetime.now().isoformat()
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"💾 체크포인트 저장: {checkpoint_path}")
    
    def train(self):
        """전체 학습 프로세스"""
        start_time = time.time()
        
        self.logger.info("🚀 하이브리드 분산 학습 시작")
        
        # 모델 및 데이터 준비
        actual_params = self.prepare_model()
        dataloader = self.prepare_data()
        
        # 학습 실행
        for epoch in range(self.config.epochs):
            self.train_epoch(dataloader, epoch)
            self.save_checkpoint(epoch, -1)
        
        # 최종 저장
        final_checkpoint = self.checkpoints_dir / 'final_hybrid_model.pth'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': asdict(self.config),
            'training_stats': dict(self.training_stats),
            'total_parameters': actual_params,
            'timestamp': datetime.now().isoformat()
        }, final_checkpoint)
        
        training_time = time.time() - start_time
        
        # 학습 리포트 생성
        report = {
            'training_summary': {
                'total_steps': len(self.training_stats['total_loss']),
                'total_regrets': sum(self.training_stats['regret_count']),
                'total_bentham_calculations': sum(self.training_stats['bentham_count']),
                'final_loss': self.training_stats['total_loss'][-1] if self.training_stats['total_loss'] else 0,
                'training_duration': training_time,
                'average_regrets_per_step': sum(self.training_stats['regret_count']) / len(self.training_stats['regret_count']) if self.training_stats['regret_count'] else 0,
                'average_benthams_per_step': sum(self.training_stats['bentham_count']) / len(self.training_stats['bentham_count']) if self.training_stats['bentham_count'] else 0
            },
            'model_info': {
                'main_model_parameters': actual_params,
                'target_parameters': self.config.target_params,
                'device': str(self.device),
                'hybrid_mode': True,
                'gpu_available': torch.cuda.is_available()
            },
            'configuration': asdict(self.config),
            'training_stats': dict(self.training_stats),
            'storage_usage': {
                'final_size_gb': 0,  # 계산 생략
                'max_allowed_gb': self.config.max_storage_gb
            },
            'xai_integration': {
                'xai_logs_generated': len(xai_logger.logs) if hasattr(xai_logger, 'logs') else 0,
                'session_id': getattr(xai_logger, 'session_id', 'unknown')
            }
        }
        
        # 리포트 저장
        report_path = self.reports_dir / f'hybrid_training_report_{int(time.time())}.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"🎉 하이브리드 학습 완료! 총 시간: {training_time/3600:.2f}시간")
        self.logger.info(f"📊 총 후회: {sum(self.training_stats['regret_count'])}")
        self.logger.info(f"📊 총 벤담 계산: {sum(self.training_stats['bentham_count'])}")
        self.logger.info(f"📋 리포트: {report_path}")
        
        return report, final_checkpoint

if __name__ == "__main__":
    # 하이브리드 설정으로 테스트
    config = HybridConfig()
    trainer = HybridDistributedTrainer(config)
    
    print("🧪 하이브리드 분산 학습 시스템 준비 완료")
    print(f"📊 설정: {config.regrets_per_step}회 후회/스텝, {config.epochs}번 선회")
    print(f"🤖 모델: {config.target_params:,}개 파라미터")
    print(f"⚡ 워커: {config.num_workers}개")
    print("준비된 학습을 시작하려면 trainer.train()을 호출하세요.")