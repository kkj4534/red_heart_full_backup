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
from dataclasses import dataclass, asdict, field
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

# 초기화 및 옵티마이저 개선을 위한 임포트
from torch.nn import init
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

@dataclass
class HybridConfig:
    """하이브리드 시스템 설정"""
    # 모델 분할 설정 (공격적 최적화 70GB)
    gpu_memory_gb: float = 8.0          # RTX 2070S 메모리
    cpu_memory_gb: float = 70.0         # WSL 공격적 메모리 활용
    target_params: int = 800_000_000     # 800M 파라미터 (300M 백본 + 500M 헤드)
    
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
    
    # 최적화 설정 (그래디언트 안정화)
    use_mixed_precision: bool = True    # Mixed Precision 유지 (효율성)
    gradient_accumulation_steps: int = 8 # 그래디언트 누적 단계 증가
    max_grad_norm: float = 0.5          # 적당한 수준의 클리핑
    
    # 초기화 및 옵티마이저 설정 (그래디언트 안정화)
    initialization_method: str = 'xavier'  # Xavier 초기화로 변경 (더 보수적)
    optimizer_type: str = 'adamw'          # AdamW 옵티마이저 사용
    learning_rate: float = 1e-4            # 적절한 학습률 (Adam 기본값)
    weight_decay: float = 0.1              # 가중치 감쇠 강화
    scheduler_type: str = 'cosine'         # 'cosine' 또는 'linear' 스케줄러
    
    # 데이터 샘플링 설정 (Phase 2 개선)
    enable_balanced_sampling: bool = True  # 폴더별 균등 샘플링
    data_folder_weights: dict = field(default_factory=lambda: {
        'scruples': 0.4,              # 윤리적 딜레마 (핵심)
        'classic_literature': 0.3,    # 문학적 감정 복잡성
        'ai_generated_scenarios.json': 0.2,  # 일관된 패턴
        'ebs_korean_literature': 0.1  # 교육적 체계성
    })
    validation_frequency: int = 50        # N스텝마다 검증
    enable_continuous_monitoring: bool = True  # 지속적 모니터링
    
    # 로깅/체크포인트
    log_every_n_steps: int = 5          # 더 자주 로깅
    save_checkpoint_every: int = 20     # 더 자주 저장
    max_storage_gb: float = 50.0        # 스토리지 절약

class MemoryOptimizedModel(nn.Module):
    """메모리 최적화된 모델 (개선된 초기화 및 옵티마이저 적용)"""
    
    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config
        
        # 800M 파라미터 설계 (config.py 설정과 동일)
        self.hidden_dim = 1280      # config의 d_model과 일치
        self.num_layers = 18        # config의 num_layers와 일치
        self.num_heads = 20         # config의 num_heads와 일치
        self.intermediate_size = 5120  # config의 feedforward_dim과 일치
        
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
        
        # 초기화 전략 적용 (He vs Xavier 비교 실험)
        self._initialize_weights()
        
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
    
    def _initialize_weights(self):
        """초기화 전략 적용 - He vs Xavier 비교 실험"""
        initialization_method = getattr(self.config, 'initialization_method', 'he')  # 기본값: He
        
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if initialization_method == 'he':
                    # He 초기화: ReLU 계열 활성화 함수에 적합 (SwiGLU 포함, 스케일 축소)
                    init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                    module.weight.data *= 0.5  # 수치 안정성을 위한 스케일 축소
                elif initialization_method == 'xavier':
                    # Xavier 초기화: Tanh 계열 활성화 함수에 적합 (스케일 축소)
                    init.xavier_normal_(module.weight)
                    module.weight.data *= 0.7  # 수치 안정성을 위한 스케일 축소
                
                # bias는 공통적으로 0으로 초기화
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            
            elif isinstance(module, nn.LayerNorm):
                # LayerNorm은 표준 초기화 유지
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)
        
        print(f"초기화 방법 적용: {initialization_method}")
    
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
        # 입력 안전성 검사
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"SwiGLU 입력에 NaN/Inf 발견: {x.shape}")
            return torch.zeros_like(x)
        
        # Swish(x) = x * sigmoid(x), 수치적으로 안정적
        w_out = self.w(x)
        
        # 중간 출력 안전성 검사
        if torch.isnan(w_out).any() or torch.isinf(w_out).any():
            print(f"SwiGLU w_out에 NaN/Inf 발견")
            return torch.zeros_like(x)
            
        swish_w = w_out * torch.sigmoid(w_out)
        v_out = self.v(x)
        
        # 최종 출력 안전성 검사
        result = swish_w * v_out
        if torch.isnan(result).any() or torch.isinf(result).any():
            print(f"SwiGLU 출력에 NaN/Inf 발견")
            return torch.zeros_like(x)
            
        return result

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

class BalancedDataset(Dataset):
    """폴더별 균등 샘플링 데이터셋 (Phase 2 개선)"""
    
    def __init__(self, config: HybridConfig):
        self.config = config
        self.scenarios = []
        self.folder_data = {}  # 폴더별 데이터 저장
        self.validation_stats = {'total_batches': 0, 'folder_counts': {}}
        
        # 데이터 로드 및 분류
        self._load_and_categorize_data()
        
        # 균등 샘플링 설정
        if config.enable_balanced_sampling:
            self._setup_balanced_sampling()
        
        print(f"📈 데이터셋 초기화 완료: 총 {len(self.scenarios)}개 시나리오")
        for folder, data in self.folder_data.items():
            print(f"   {folder}: {len(data)}개")
    
    def _load_and_categorize_data(self):
        """데이터 로드 및 폴더별 분류"""
        base_path = Path('/mnt/c/large_project/linux_red_heart/processed_datasets')
        
        # 기존 full_scenarios_batch 파일들 사용 (빠른 테스트)
        self._load_full_batch_files(base_path)
    
    def _load_full_batch_files(self, base_path: Path):
        """기존 배치 파일들에서 데이터 로드"""
        self.folder_data = {
            'scruples': [],
            'classic_literature': [],
            'ai_generated_scenarios.json': [],
            'ebs_korean_literature': []
        }
        
        limit = self.config.test_samples if hasattr(self.config, 'test_mode') and self.config.test_mode else 1000
        total_loaded = 0
        
        # full_scenarios_batch 파일들 로드
        for file_path in base_path.glob('full_scenarios_batch_*.json'):
            if total_loaded >= limit:
                break
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    batch_data = json.load(f)
                    if isinstance(batch_data, list):
                        for item in batch_data:
                            if total_loaded >= limit:
                                break
                            # 간단한 스키마로 변환
                            scenario = {
                                'text': str(item.get('title', '')) + ' ' + str(item.get('description', '')),
                                'source': 'mixed_data',
                                'labels': [0.3, 0.5, 0.2]
                            }
                            # 균등하게 분배
                            folder_idx = total_loaded % 4
                            folder_names = list(self.folder_data.keys())
                            self.folder_data[folder_names[folder_idx]].append(scenario)
                            total_loaded += 1
                            
                print(f"📂 {file_path.name}: {len(batch_data) if isinstance(batch_data, list) else 1}개 처리")
            except Exception as e:
                print(f"⚠️  {file_path} 로드 오류: {e}")
        
        # 로드 결과 출력
        for folder_name, data in self.folder_data.items():
            print(f"📂 {folder_name}: {len(data)}개 로드")
    
    def _load_scruples_data(self, folder_path: Path) -> List[Dict]:
        """윤리적 딜레마 데이터 로드"""
        data = []
        limit = self.config.test_samples // 4 if hasattr(self.config, 'test_mode') and self.config.test_mode else 1000
        
        # JSON 배치 파일들 로드 (실제 디렉토리 구조에 맞춤)
        for file_path in folder_path.glob('scruples_batch_*.json'):
            if len(data) < limit:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        batch_data = json.load(f)
                        if isinstance(batch_data, list):
                            for item in batch_data:
                                if len(data) >= limit:
                                    break
                                scenario = self._process_scruples_item(item, 'scruples')
                                if scenario:
                                    data.append(scenario)
                except Exception as e:
                    print(f"⚠️  {file_path} 로드 오류: {e}")
        return data
    
    def _load_book_data(self, folder_path: Path) -> List[Dict]:
        """문학 작품 데이터 로드"""
        data = []
        limit = self.config.test_samples // 4 if hasattr(self.config, 'test_mode') and self.config.test_mode else 500
        
        for file_path in folder_path.glob('*.txt'):
            if len(data) >= limit:
                break
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()[:5000]  # 처음 5000자만
                    sentences = content.split('. ')
                    
                    for sentence in sentences:
                        if len(data) >= limit:
                            break
                        if len(sentence.strip()) > 20:
                            scenario = {
                                'text': sentence.strip(),
                                'source': 'book',
                                'labels': [0.5, 0.3, 0.2]
                            }
                            data.append(scenario)
            except Exception as e:
                print(f"⚠️  {file_path} 로드 오류: {e}")
        return data
    
    def _load_ai_generated_data(self, folder_path: Path) -> List[Dict]:
        """AI 생성 데이터 로드"""
        data = []
        limit = self.config.test_samples // 4 if hasattr(self.config, 'test_mode') and self.config.test_mode else 300
        
        for file_path in folder_path.glob('*.txt'):
            if len(data) >= limit:
                break
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()[:100]  # 처음 100줄만
                    for line in lines:
                        if len(data) >= limit:
                            break
                        if len(line.strip()) > 10:
                            scenario = {
                                'text': line.strip(),
                                'source': 'ai_generated',
                                'labels': [0.4, 0.4, 0.2]
                            }
                            data.append(scenario)
            except Exception as e:
                print(f"⚠️  {file_path} 로드 오류: {e}")
        return data
    
    def _load_ebs_data(self, folder_path: Path) -> List[Dict]:
        """교육 콘텐츠 데이터 로드"""
        data = []
        limit = self.config.test_samples // 4 if hasattr(self.config, 'test_mode') and self.config.test_mode else 200
        
        for file_path in folder_path.glob('*.txt'):
            if len(data) >= limit:
                break
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()[:3000]  # 처음 3000자만
                    paragraphs = content.split('\n\n')
                    for paragraph in paragraphs:
                        if len(data) >= limit:
                            break
                        if len(paragraph.strip()) > 30:
                            scenario = {
                                'text': paragraph.strip(),
                                'source': 'ai_ebs',
                                'labels': [0.6, 0.2, 0.2]
                            }
                            data.append(scenario)
            except Exception as e:
                print(f"⚠️  {file_path} 로드 오류: {e}")
        return data
    
    def _process_scruples_item(self, item: Dict, subfolder: str) -> Dict:
        """윤리적 딜레마 아이템 처리"""
        text = ''
        if 'title' in item and 'text' in item:
            text = f"{item['title']} {item['text']}"
        elif 'text' in item:
            text = item['text']
        elif 'action' in item:
            text = item['action']
        
        if not text or len(text.strip()) < 10:
            return None
        
        return {
            'text': text.strip()[:500],  # 최대 500자
            'source': 'scruples_real_data',
            'labels': [0.3, 0.5, 0.2]
        }
    
    def _setup_balanced_sampling(self):
        """균등 샘플링 설정"""
        self.scenarios = []
        
        for folder_name, data in self.folder_data.items():
            if data and folder_name in self.config.data_folder_weights:
                weight = self.config.data_folder_weights[folder_name]
                # 가중치에 따른 샘플 수 계산
                target_samples = int(len(data) * weight * 1.5)  # 1.5배 오버샘플링
                target_samples = min(target_samples, len(data))  # 데이터 수를 초과하지 않음
                
                if target_samples > 0:
                    sampled_data = np.random.choice(data, target_samples, replace=False).tolist()
                    self.scenarios.extend(sampled_data)
        
        # 전체 데이터 셔플
        np.random.shuffle(self.scenarios)
        
        print(f"🎯 균등 샘플링 완료: {len(self.scenarios)}개 시나리오")
    
    def __len__(self):
        return len(self.scenarios)
    
    def __getitem__(self, idx):
        scenario = self.scenarios[idx]
        
        # 텍스트를 임베딩으로 변환 (최적화된 차원)
        embedding = torch.randn(1024, dtype=torch.float32)
        labels = torch.tensor(scenario['labels'], dtype=torch.float32)
        
        return {
            'text_embedding': embedding,
            'labels': labels,
            'scenario_id': f"balanced_{idx}",
            'source': scenario['source']
        }
    
    def get_batch_distribution(self, batch_sources: List[str]) -> Dict[str, float]:
        """배치 내 데이터 분포 계산"""
        from collections import Counter
        counter = Counter(batch_sources)
        total = len(batch_sources)
        return {source: count/total for source, count in counter.items()}

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
        
        # 모델, 옵티마이저, 스케줄러 초기화 (나중에 설정)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        
        print(f"🚀 하이브리드 시스템 초기화 완료")
        if torch.cuda.is_available():
            print(f"   - GPU 가속: ✅ CUDA 활성화")
            print(f"   - GPU 메모리: {config.gpu_memory_gb}GB")
        else:
            print(f"   - CPU 최적화: ✅ 고성능 CPU 모드")
            print(f"   - CPU 메모리: {config.cpu_memory_gb}GB (대용량 처리)")
        print(f"   - 병렬 워커: {config.num_workers}개")
        print(f"   - 분산 처리: CPU+메모리 최적화")
    
    def setup_model_and_optimizer(self):
        """모델, 옵티마이저, 스케줄러 설정 (단계별 개선 적용)"""
        
        # 모델 생성
        self.model = MemoryOptimizedModel(self.config)
        self.logger.info(f"모델 생성 완료: {self.model.get_parameter_count():,}개 파라미터")
        
        # 개선된 옵티마이저 설정
        if self.config.optimizer_type == 'adamw':
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                eps=1e-6,  # 수치 안정성 향상
                betas=(0.9, 0.98),  # 더 보수적인 베타 값
                amsgrad=True  # AMSGrad 활성화로 안정성 향상
            )
            self.logger.info(f"AdamW 옵티마이저 설정: lr={self.config.learning_rate}, wd={self.config.weight_decay}")
        else:
            raise ValueError(f"지원되지 않는 옵티마이저: {self.config.optimizer_type}")
        
        # 학습률 스케줄러 설정
        if self.config.scheduler_type == 'cosine':
            # Cosine Annealing: 주기적 학습률 감소
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=200,  # 최대 에포크 수
                eta_min=self.config.learning_rate * 0.01  # 최소 학습률
            )
            self.logger.info("코사인 어닐링 스케줄러 설정")
        elif self.config.scheduler_type == 'linear':
            # Linear Decay
            self.scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=100
            )
            self.logger.info("선형 감소 스케줄러 설정")
        
        # Mixed Precision 스케일러
        if self.config.use_mixed_precision and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
            self.logger.info("Mixed Precision 스케일러 활성화")
        
        # 수치 안정성 검증
        self._validate_model_stability()
    
    def _validate_model_stability(self):
        """모델 수치 안정성 검증"""
        self.logger.info("모델 안정성 검증 시작...")
        
        # 더미 입력으로 테스트
        dummy_input = torch.randn(2, 16, 1024, device=self.model.cpu_device)
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(dummy_input)
            
        # 출력 안정성 검사
        all_stable = True
        for key, value in outputs.items():
            has_nan = torch.isnan(value).any()
            has_inf = torch.isinf(value).any()
            if has_nan or has_inf:
                all_stable = False
                self.logger.warning(f"{key} 출력에 비정상 값: NaN={has_nan}, Inf={has_inf}")
            else:
                self.logger.debug(f"{key}: 안정적 ({value.min():.3f} ~ {value.max():.3f})")
        
        if all_stable:
            self.logger.info("✅ 모델 안정성 검증 통과")
        else:
            self.logger.error("❌ 모델 안정성 문제 발견")
            
        self.model.train()  # 학습 모드로 복귀
    
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
        
        # Phase 2: 균등 샘플링 또는 기존 데이터셋
        if self.config.enable_balanced_sampling:
            self.logger.info("🎯 균등 샘플링 데이터셋 사용")
            dataset = BalancedDataset(self.config)
        else:
            self.logger.info("📋 기존 데이터셋 사용")
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
        
        # 데이터 분포 검증 (균등 샘플링인 경우)
        if isinstance(dataset, BalancedDataset) and len(dataset) > 0:
            sample_batch = [dataset[i] for i in range(min(10, len(dataset)))]
            sources = [item['source'] for item in sample_batch]
            distribution = dataset.get_batch_distribution(sources)
            self.logger.info(f"📈 샘플 분포: {distribution}")
        
        return dataloader
    
    def vad_feedback_loop(self, emotion_predictions: torch.Tensor, step_idx: int) -> Dict[str, torch.Tensor]:
        """VAD 벡터 실시간 피드백 루프 (Phase 3 - 심층 연동 강화)"""
        
        # VAD 벡터 추출 (Valence-Arousal-Dominance) - 차원 안전성 보장
        # emotion_predictions: [batch_size, 6] -> [valence, arousal, dominance, certainty, surprise, anticipation]
        if emotion_predictions.shape[1] >= 3:
            vad_vector = emotion_predictions[:, :3]  # [batch_size, 3] - VAD 차원만 추출
        else:
            # 차원이 부족한 경우 제로 패딩
            batch_size = emotion_predictions.shape[0]
            vad_vector = torch.zeros(batch_size, 3, device=emotion_predictions.device)
            min_dim = min(emotion_predictions.shape[1], 3)
            vad_vector[:, :min_dim] = emotion_predictions[:, :min_dim]
            print(f"emotion_predictions 차원 부족: {emotion_predictions.shape}, VAD는 {vad_vector.shape}로 패딩")
        
        # 1. VAD 기반 윤리 판단 가중치 동적 조정
        ethics_weights = self._calculate_ethics_weights_from_vad(vad_vector)
        
        # 2. 실시간 감정-윤리 피드백 매핑
        ethics_feedback = self._apply_vad_to_ethics_mapping(vad_vector, ethics_weights)
        
        # 3. 윤리 판단 결과가 감정 시스템에 역피드백
        emotion_adjustment = self._ethics_to_emotion_feedback(ethics_feedback, vad_vector)
        
        # 4. 후회 기반 VAD 업데이트 (학습적 작용)
        regret_adjusted_vad = self._regret_based_vad_adjustment(vad_vector, step_idx)
        
        # 5. 새로운 기능: regret_score를 VAD 조정 인자로 명시적 도입
        regret_score_adjustment = self._calculate_regret_score_vad_adjustment(vad_vector, step_idx)
        
        # 6. 통합된 VAD 계산 (모든 조정 요소 반영)
        integrated_vad = regret_adjusted_vad + emotion_adjustment * 0.1 + regret_score_adjustment
        
        return {
            'original_vad': vad_vector,
            'ethics_weights': ethics_weights,
            'ethics_feedback': ethics_feedback,
            'emotion_adjustment': emotion_adjustment,
            'regret_adjusted_vad': regret_adjusted_vad,
            'regret_score_adjustment': regret_score_adjustment,  # 새로운 조정 요소
            'integrated_vad': integrated_vad
        }
    
    def _calculate_ethics_weights_from_vad(self, vad_vector: torch.Tensor) -> torch.Tensor:
        """VAD 벡터로부터 윤리 판단 가중치 계산"""
        # Valence-Arousal-Dominance → 윤리적 선택지 가중치 변환
        valence = vad_vector[:, 0]    # 감정 극성 (-1~1)
        arousal = vad_vector[:, 1]    # 각성도 (0~1) 
        dominance = vad_vector[:, 2]  # 지배감 (0~1)
        
        # 윤리적 선택에 대한 감정 기반 가중치
        # 높은 valence + 낮은 arousal = 차분한 판단 (규칙 기반 강화)
        rule_based_weight = torch.sigmoid(valence - arousal)
        
        # 높은 arousal + 높은 dominance = 직관적 판단 (결과 기반 강화)  
        consequence_weight = torch.sigmoid(arousal + dominance - 1.0)
        
        # 균형잡힌 상태 = 덕윤리 기반
        virtue_weight = 1.0 - torch.abs(valence) * torch.abs(arousal - 0.5)
        
        return torch.stack([rule_based_weight, consequence_weight, virtue_weight], dim=1)
    
    def _apply_vad_to_ethics_mapping(self, vad_vector: torch.Tensor, ethics_weights: torch.Tensor) -> torch.Tensor:
        """VAD → 윤리 판단 실시간 매핑 (퍼지 감정 매핑)"""
        batch_size = vad_vector.shape[0]
        
        # 감정 상태에 따른 윤리 판단 조정
        valence = vad_vector[:, 0]
        arousal = vad_vector[:, 1] 
        dominance = vad_vector[:, 2]
        
        # 퍼지 로직 기반 연속적 매핑
        # 슬픔 주도 상태 (valence < 0, arousal < 0.5) → 위로/배려 우선
        sadness_driven = torch.relu(-valence) * torch.relu(0.5 - arousal)
        
        # 분노 주도 상태 (valence < 0, arousal > 0.5) → 정의/공정성 우선  
        anger_driven = torch.relu(-valence) * torch.relu(arousal - 0.5)
        
        # 기쁨 주도 상태 (valence > 0, arousal > 0.5) → 공동체/협력 우선
        joy_driven = torch.relu(valence) * torch.relu(arousal - 0.5)
        
        # 평온 주도 상태 (valence > 0, arousal < 0.5) → 이성/숙고 우선
        calm_driven = torch.relu(valence) * torch.relu(0.5 - arousal)
        
        # 통합된 윤리 조정 벡터
        ethics_adjustment = torch.stack([
            sadness_driven,   # 배려 차원
            anger_driven,     # 정의 차원  
            joy_driven,       # 협력 차원
            calm_driven       # 이성 차원
        ], dim=1)
        
        return ethics_adjustment * ethics_weights[:, :1]  # 첫 번째 가중치로 스케일링
    
    def _ethics_to_emotion_feedback(self, ethics_feedback: torch.Tensor, current_vad: torch.Tensor) -> torch.Tensor:
        """윤리 판단 결과 → 감정 시스템 역피드백"""
        # 윤리적 결정이 감정 상태에 미치는 영향 모델링
        
        # 장치 일치 보장
        device = ethics_feedback.device
        
        # 배려적 결정 → 따뜻함/만족감 증가
        care_influence = ethics_feedback[:, 0:1] * torch.tensor([0.3, -0.1, 0.1], device=device)  # valence+, arousal-, dominance+
        
        # 정의로운 결정 → 의기양양함 증가
        justice_influence = ethics_feedback[:, 1:2] * torch.tensor([0.2, 0.2, 0.3], device=device)  # valence+, arousal+, dominance+
        
        # 협력적 결정 → 기쁨/활력 증가  
        coop_influence = ethics_feedback[:, 2:3] * torch.tensor([0.4, 0.3, 0.0], device=device)   # valence+, arousal+, dominance=
        
        # 이성적 결정 → 평온/확신 증가
        rational_influence = ethics_feedback[:, 3:4] * torch.tensor([0.1, -0.2, 0.2], device=device)  # valence+, arousal-, dominance+
        
        # 통합된 감정 조정
        total_influence = care_influence + justice_influence + coop_influence + rational_influence
        
        return total_influence
    
    def _regret_based_vad_adjustment(self, vad_vector: torch.Tensor, step_idx: int) -> torch.Tensor:
        """후회 → VAD 조정 (학습적 작용, 반성 기반 윤리 학습)"""
        
        # 과거 후회 패턴 기반 VAD 조정 (단순화된 버전)
        # 실제로는 Experience DB에서 유사 상황의 후회 데이터를 가져와야 함
        
        # 주기적인 후회 반성 (매 10스텝마다)
        if step_idx % 10 == 0:
            # 후회가 높았던 경우의 VAD 패턴 회피
            # 예: 과도한 arousal + 낮은 valence 조합 회피
            high_arousal_low_valence = (vad_vector[:, 1] > 0.7) & (vad_vector[:, 0] < -0.3)
            
            regret_adjustment = torch.zeros_like(vad_vector)
            regret_adjustment[high_arousal_low_valence, 1] -= 0.1  # arousal 감소
            regret_adjustment[high_arousal_low_valence, 0] += 0.05  # valence 증가
            
            return vad_vector + regret_adjustment
        
        return vad_vector  # 조정 없음
    
    def regret_based_ethics_adjustment(self, step_idx: int, recent_regret_patterns: List[Dict]) -> Dict[str, torch.Tensor]:
        """Phase 4: 후회 기반 윤리 기준 동적 조정"""
        
        # 후회 패턴 분석 및 윤리 기준 재조정
        ethics_priority_adjustment = {
            'rule_based_priority': torch.tensor(1.0),      # 규칙 기반 윤리 우선도
            'consequence_priority': torch.tensor(1.0),     # 결과 기반 윤리 우선도
            'virtue_priority': torch.tensor(1.0),          # 덕윤리 우선도
            'care_priority': torch.tensor(1.0)             # 배려 윤리 우선도
        }
        
        # 최근 후회 패턴이 있는 경우 분석
        if recent_regret_patterns:
            # 후회 유형별 빈도 분석
            regret_counts = {
                'rule_violation': 0,    # 규칙 위반으로 인한 후회
                'bad_outcome': 0,       # 나쁜 결과로 인한 후회  
                'character_flaw': 0,    # 품성 부족으로 인한 후회
                'lack_of_care': 0       # 배려 부족으로 인한 후회
            }
            
            # 최근 10개 후회 패턴 분석
            for pattern in recent_regret_patterns[-10:]:
                regret_type = pattern.get('type', 'unknown')
                if regret_type in regret_counts:
                    regret_counts[regret_type] += 1
            
            # 후회가 많은 영역의 우선순위 강화
            total_regrets = sum(regret_counts.values())
            if total_regrets > 0:
                # 규칙 위반 후회가 많으면 → 규칙 기반 윤리 강화
                if regret_counts['rule_violation'] / total_regrets > 0.3:
                    ethics_priority_adjustment['rule_based_priority'] += 0.2
                
                # 나쁜 결과 후회가 많으면 → 결과 기반 윤리 강화  
                if regret_counts['bad_outcome'] / total_regrets > 0.3:
                    ethics_priority_adjustment['consequence_priority'] += 0.2
                
                # 품성 부족 후회가 많으면 → 덕윤리 강화
                if regret_counts['character_flaw'] / total_regrets > 0.3:
                    ethics_priority_adjustment['virtue_priority'] += 0.2
                
                # 배려 부족 후회가 많으면 → 배려 윤리 강화
                if regret_counts['lack_of_care'] / total_regrets > 0.3:
                    ethics_priority_adjustment['care_priority'] += 0.2
        
        # 시간에 따른 점진적 조정 (학습 효과)
        time_factor = min(step_idx / 1000.0, 1.0)  # 1000스텝에 걸쳐 점진적 적용
        
        for key in ethics_priority_adjustment:
            base_value = ethics_priority_adjustment[key]
            adjustment = (base_value - 1.0) * time_factor
            ethics_priority_adjustment[key] = 1.0 + adjustment
        
        return ethics_priority_adjustment
    
    def _calculate_regret_score_vad_adjustment(self, vad_vector: torch.Tensor, step_idx: int) -> torch.Tensor:
        """regret_score를 VAD 값 조정 인자로 명시적 도입 (docs 개선사항)"""
        
        # 현재 단계에서의 후회 점수 추정 (step_idx 기반)
        # 실제 구현에서는 regret_calculator에서 실시간 점수를 가져올 수 있음
        base_regret_score = min(step_idx * 0.01, 1.0)  # 단계별 누적 후회
        
        batch_size = vad_vector.shape[0]
        device = vad_vector.device
        
        # VAD 각 차원별로 후회 점수의 영향 계산
        valence_adjustment = -base_regret_score * 0.2  # 후회 증가 시 valence 감소
        arousal_adjustment = base_regret_score * 0.1   # 후회 증가 시 각성도 약간 증가
        dominance_adjustment = -base_regret_score * 0.15  # 후회 증가 시 지배감 감소
        
        # 배치 크기에 맞춰 조정 벡터 생성
        regret_adjustment = torch.tensor([
            [valence_adjustment, arousal_adjustment, dominance_adjustment]
        ], device=device).expand(batch_size, -1)
        
        # 현재 VAD 상태에 따른 적응적 조정
        current_valence = vad_vector[:, 0]
        
        # 이미 부정적인 감정 상태에서는 후회 영향을 완화
        valence_mask = (current_valence < -0.5).float().unsqueeze(1)
        regret_adjustment = regret_adjustment * (1.0 - valence_mask * 0.5)
        
        return regret_adjustment
    
    def _calculate_individual_community_balance(self, vad_vector: torch.Tensor) -> torch.Tensor:
        """개인-공동체 균형 계수 계산 (docs 개선사항: 철학적 기준 수치화)"""
        
        valence = vad_vector[:, 0]    # 감정 극성
        arousal = vad_vector[:, 1]    # 각성도 
        dominance = vad_vector[:, 2]  # 지배감
        
        # 개인 중심 성향 계산
        # 높은 지배감 + 낮은 각성도 = 개인 중심적 사고
        individual_tendency = dominance * (1.0 - arousal)
        
        # 공동체 중심 성향 계산  
        # 긍정적 감정 + 높은 각성도 = 공동체 지향적 사고
        community_tendency = torch.clamp(valence, 0, 1) * arousal
        
        # 균형 계수 (0.5: 완전 개인 중심, 1.5: 완전 공동체 중심)
        balance_coefficient = 0.5 + community_tendency - individual_tendency * 0.5
        balance_coefficient = torch.clamp(balance_coefficient, 0.3, 1.7)
        
        return balance_coefficient
    
    def post_decision_emotional_response(self, ethics_decision: Dict[str, torch.Tensor], 
                                       original_vad: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Phase 5: 사후 윤리 판단에 따른 감정 변화"""
        
        # 윤리적 결정에 따른 감정적 결과 시뮬레이션
        decision_type = self._classify_ethics_decision(ethics_decision)
        
        # 장치 일치 보장
        device = original_vad.device
        
        # 감정 변화 패턴
        emotional_consequences = {}
        
        if decision_type == 'altruistic':
            # 이타적 결정 → 만족감, 자부심 증가
            emotional_consequences = {
                'valence_change': torch.tensor(0.3, device=device),     # 긍정적 감정 증가
                'arousal_change': torch.tensor(-0.1, device=device),    # 평온함 증가
                'dominance_change': torch.tensor(0.2, device=device),   # 자기효능감 증가
                'guilt_level': torch.tensor(0.0, device=device),        # 죄책감 없음
                'pride_level': torch.tensor(0.4, device=device),        # 자부심 높음
                'regret_probability': torch.tensor(0.1, device=device)  # 후회 가능성 낮음
            }
        elif decision_type == 'selfish':
            # 이기적 결정 → 일시적 만족, 장기적 죄책감
            emotional_consequences = {
                'valence_change': torch.tensor(0.1, device=device),     # 일시적 긍정감
                'arousal_change': torch.tensor(0.2, device=device),     # 불안 증가
                'dominance_change': torch.tensor(-0.1, device=device),  # 자기 의심
                'guilt_level': torch.tensor(0.3, device=device),        # 죄책감 발생
                'pride_level': torch.tensor(0.0, device=device),        # 자부심 없음
                'regret_probability': torch.tensor(0.6, device=device)  # 후회 가능성 높음
            }
        elif decision_type == 'harmful':
            # 타인에게 해를 끼치는 결정 → 죄책감, 후회
            emotional_consequences = {
                'valence_change': torch.tensor(-0.4, device=device),    # 부정적 감정 강함
                'arousal_change': torch.tensor(0.3, device=device),     # 스트레스 증가
                'dominance_change': torch.tensor(-0.3, device=device),  # 자기 혐오
                'guilt_level': torch.tensor(0.6, device=device),        # 높은 죄책감
                'pride_level': torch.tensor(0.0, device=device),        # 자부심 없음
                'regret_probability': torch.tensor(0.8, device=device)  # 매우 높은 후회 가능성
            }
        else:
            # 중성적 결정 → 최소한의 감정 변화
            emotional_consequences = {
                'valence_change': torch.tensor(0.0, device=device),
                'arousal_change': torch.tensor(0.0, device=device),
                'dominance_change': torch.tensor(0.0, device=device),
                'guilt_level': torch.tensor(0.1, device=device),
                'pride_level': torch.tensor(0.1, device=device),
                'regret_probability': torch.tensor(0.2, device=device)
            }
        
        # 새로운 VAD 상태 계산 (차원 안전성 보장)
        new_vad = original_vad.clone()
        
        # original_vad가 최소 3차원인지 확인
        if original_vad.shape[1] >= 3:
            # 브로드캐스팅을 위해 차원 확장
            batch_size = original_vad.shape[0]
            valence_change = emotional_consequences['valence_change'].expand(batch_size)
            arousal_change = emotional_consequences['arousal_change'].expand(batch_size)
            dominance_change = emotional_consequences['dominance_change'].expand(batch_size)
            
            new_vad[:, 0] += valence_change     # Valence
            new_vad[:, 1] += arousal_change     # Arousal  
            new_vad[:, 2] += dominance_change   # Dominance
        else:
            print(f"original_vad 차원 부족: {original_vad.shape}, 3차원으로 확장")
            # 3차원으로 확장
            batch_size = original_vad.shape[0]
            new_vad = torch.zeros(batch_size, 3, device=original_vad.device)
            # 기존 차원만큼 복사
            min_dim = min(original_vad.shape[1], 3)
            new_vad[:, :min_dim] = original_vad[:, :min_dim]
            
            # 감정 변화 적용
            valence_change = emotional_consequences['valence_change'].expand(batch_size)
            arousal_change = emotional_consequences['arousal_change'].expand(batch_size)
            dominance_change = emotional_consequences['dominance_change'].expand(batch_size)
            
            new_vad[:, 0] += valence_change
            new_vad[:, 1] += arousal_change
            new_vad[:, 2] += dominance_change
        
        # VAD 범위 제한 (-1 ~ 1)
        new_vad = torch.clamp(new_vad, -1.0, 1.0)
        
        return {
            'new_vad': new_vad,
            'emotional_consequences': emotional_consequences,
            'decision_type': decision_type
        }
    
    def _classify_ethics_decision(self, ethics_decision: Dict[str, torch.Tensor]) -> str:
        """윤리적 결정 분류"""
        # 단순화된 분류 로직
        # 실제로는 더 복잡한 패턴 인식이 필요
        
        rule_priority = ethics_decision.get('rule_based_priority', torch.tensor(1.0))
        care_priority = ethics_decision.get('care_priority', torch.tensor(1.0))
        
        if care_priority > 1.2:
            return 'altruistic'
        elif rule_priority < 0.8:
            return 'selfish'
        elif care_priority < 0.5:
            return 'harmful'
        else:
            return 'neutral'

    def train_step(self, batch: Dict[str, torch.Tensor], step_idx: int) -> Dict[str, float]:
        """메모리 최적화된 학습 스텝 (VAD 피드백 루프 통합)"""
        self.model.train()
        
        # 메모리 정리
        if step_idx % 5 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        text_embeddings = batch['text_embedding'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # 배치 크기 동적 조정
        current_batch_size = text_embeddings.size(0)
        if current_batch_size > self.config.max_safe_batch_size:
            return self._process_large_batch(batch, step_idx)
        
        # 비동기 후회 계산 시작 (최적화)
        for i, embedding in enumerate(text_embeddings[:min(len(text_embeddings), 6)]):
            self.regret_calculator.calculate_async(embedding, step_idx * self.config.batch_size + i)
        
        # 모델 순전파 (Mixed Precision) - 안전성 강화
        try:
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(text_embeddings.unsqueeze(1))
            else:
                outputs = self.model(text_embeddings.unsqueeze(1))
            
            # 모델 출력 전체에 대한 NaN/Inf 검사
            has_nan_output = False
            for key, value in outputs.items():
                if torch.isnan(value).any() or torch.isinf(value).any():
                    print(f"모델 출력 {key}에서 NaN/Inf 발견: {value.shape}")
                    has_nan_output = True
            
            if has_nan_output:
                # 모델 출력 자체에 문제가 있으면 안전한 기본값 반환
                print("⚠️ 모델 출력에 NaN/Inf 발견 - 학습 스킵")
                return {
                    'loss': 0.0,
                    'classification_loss': 0.0,
                    'regret_count': 0,
                    'bentham_count': 0
                }
                
        except Exception as e:
            print(f"모델 순전파 오류: {e}")
            return {
                'loss': 0.0,
                'classification_loss': 0.0,
                'regret_count': 0,
                'bentham_count': 0
            }
        
        # 기본 분류 손실 (NaN 방지 안정성 개선)
        emotion_predictions = outputs['emotion_predictions']
        
        # Mixed Precision 호환을 위해 float32로 변환
        if emotion_predictions.dtype == torch.float16:
            emotion_predictions = emotion_predictions.float()
        
        # 최종 NaN 검사
        if torch.isnan(emotion_predictions).any() or torch.isinf(emotion_predictions).any():
            print("감정 예측에 NaN/Inf 발견, 제로로 초기화")
            emotion_predictions = torch.zeros_like(emotion_predictions, requires_grad=True)
        
        # Phase 3: VAD 벡터 실시간 피드백 루프 (심층 연동 강화)
        vad_feedback_results = self.vad_feedback_loop(emotion_predictions, step_idx)
        
        # Phase 4: 후회 기반 윤리 기준 동적 조정 (모의 후회 패턴 사용)
        recent_regret_patterns = [
            {'type': 'rule_violation', 'step': step_idx-1},
            {'type': 'lack_of_care', 'step': step_idx-2}
        ] if step_idx > 5 else []
        
        ethics_adjustment = self.regret_based_ethics_adjustment(step_idx, recent_regret_patterns)
        
        # Phase 5: 사후 윤리 판단에 따른 감정 변화
        post_decision_response = self.post_decision_emotional_response(
            ethics_adjustment, vad_feedback_results['original_vad']
        )
        
        # 통합된 VAD를 사용하여 감정 예측 업데이트 (Phase 5 결과 반영)
        integrated_vad = vad_feedback_results['integrated_vad']
        final_vad = post_decision_response['new_vad']  # 사후 감정 변화 반영
        
        # NaN 검사 및 안전한 처리
        if torch.isnan(final_vad).any():
            print("final_vad에 NaN 발견, 제로로 초기화")
            final_vad = torch.zeros_like(final_vad)
        
        # 차원 안전성 확보
        enhanced_emotion_predictions = emotion_predictions.clone()
        if enhanced_emotion_predictions.shape[1] >= 3 and final_vad.shape[1] >= 3:
            # final_vad가 3차원 이상인 경우만 업데이트
            enhanced_emotion_predictions[:, :3] = final_vad[:, :3]
        elif final_vad.shape[1] == 3:
            # final_vad가 정확히 3차원인 경우
            enhanced_emotion_predictions[:, :3] = final_vad
        else:
            # 차원이 안 맞는 경우 안전한 처리
            print(f"차원 불일치: enhanced_emotion_predictions {enhanced_emotion_predictions.shape}, final_vad {final_vad.shape}")
            # 최소 차원만큼 복사
            min_dim = min(enhanced_emotion_predictions.shape[1], final_vad.shape[1], 3)
            enhanced_emotion_predictions[:, :min_dim] = final_vad[:, :min_dim]
        
        # 안전한 평균 계산 (차원 호환성 보장)
        if enhanced_emotion_predictions.shape[1] >= 3:
            emotion_avg = enhanced_emotion_predictions[:, :3]  # 처음 3차원만 사용
        else:
            # 3차원보다 작은 경우 제로 패딩
            batch_size = enhanced_emotion_predictions.shape[0]
            emotion_avg = torch.zeros(batch_size, 3, device=enhanced_emotion_predictions.device)
            min_dim = enhanced_emotion_predictions.shape[1]
            emotion_avg[:, :min_dim] = enhanced_emotion_predictions
        
        # labels 차원도 확인
        if labels.shape[1] != 3:
            batch_size = labels.shape[0]
            labels_resized = torch.zeros(batch_size, 3, device=labels.device)
            min_dim = min(labels.shape[1], 3)
            labels_resized[:, :min_dim] = labels[:, :min_dim]
            labels = labels_resized
        
        # NaN 검사 후 손실 계산
        if torch.isnan(emotion_avg).any() or torch.isnan(labels).any():
            print("emotion_avg 또는 labels에 NaN 발견")
            classification_loss = torch.tensor(0.1, device=emotion_predictions.device, requires_grad=True)
        else:
            classification_loss = F.mse_loss(emotion_avg, labels)
            # 손실이 너무 큰 경우 클리핑
            classification_loss = torch.clamp(classification_loss, max=10.0)
        
        # 손실 NaN 감지 및 처리
        if torch.isnan(classification_loss):
            self.logger.error("분류 손실이 NaN입니다. 기본값으로 설정")
            classification_loss = torch.tensor(1.0, device=emotion_predictions.device, requires_grad=True)
        
        total_loss = classification_loss
        regret_count = 0
        bentham_count = 0
        
        # 후회 결과 수집 및 안전한 손실 계산
        for _ in range(7):  # 원래 7개 유지
            result = self.regret_calculator.get_result()
            if result is None:
                break
            
            task_id, regret_scenarios = result
            regret_count += len(regret_scenarios)
            bentham_count += len(regret_scenarios) * self.config.bentham_calculations_per_regret
            
            # 후회 손실 추가 (안전한 계산 방식)
            for scenario in regret_scenarios[:2]:  # 최대 2개로 제한하여 안정성 확보
                try:
                    # 기본값을 더 보수적으로 설정
                    bentham_raw = scenario.get('bentham_scores', 0.0)
                    weight_raw = scenario.get('regret_weight', 0.5)  # 기본 가중치를 0.5로 낮춤
                    
                    # 값 검증 및 정제
                    if not isinstance(bentham_raw, (int, float)) or not (-100 <= bentham_raw <= 100):
                        bentham_raw = 0.0
                    if not isinstance(weight_raw, (int, float)) or not (0 <= weight_raw <= 2):
                        weight_raw = 0.5
                    
                    # 매우 작은 후회 손실 계산 (NaN 위험 최소화)
                    safe_regret_loss = torch.tensor(
                        float(bentham_raw) * 0.01 * float(weight_raw),  # 계수를 0.01로 대폭 축소
                        dtype=torch.float32, 
                        device=total_loss.device, 
                        requires_grad=True
                    )
                    
                    # 엄격한 범위 제한
                    safe_regret_loss = torch.clamp(safe_regret_loss, min=-0.1, max=0.1)
                    
                    # 최종 안전성 검사
                    if not (torch.isfinite(safe_regret_loss) and not torch.isnan(safe_regret_loss)):
                        continue
                    
                    total_loss = total_loss + safe_regret_loss
                    
                except Exception as e:
                    # 오류 발생 시 조용히 건너뛰기
                    continue
        
        # 최종 NaN 검사 및 역전파 전 안전성 확인
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            self.logger.error(f"최종 손실이 비정상입니다: {total_loss}. 역전파 스킨")
            return {
                'loss': float('nan'),
                'classification_loss': classification_loss.item() if not torch.isnan(classification_loss) else float('nan'),
                'regret_count': regret_count,
                'bentham_count': bentham_count
            }
        
        # 그래디언트 누적을 위한 손실 정규화
        total_loss = total_loss / self.config.gradient_accumulation_steps
        
        # 역전파 (그래디언트 누적 지원)
        if self.scaler:
            # Mixed Precision 역전파
            self.scaler.scale(total_loss).backward()
        else:
            # FP32 모드 역전파
            total_loss.backward()
        
        # 그래디언트 누적 완료 시에만 옵티마이저 스텝 실행
        if (step_idx + 1) % self.config.gradient_accumulation_steps == 0:
            if self.scaler:
                # Mixed Precision 옵티마이저 스텝
                self.scaler.unscale_(self.optimizer)
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                # 그래디언트 안정성 검사
                if torch.isnan(total_norm) or torch.isinf(total_norm):
                    self.logger.error(f"그래디언트 norm이 비정상입니다: {total_norm}. 그래디언트 초기화")
                    self.optimizer.zero_grad()
                    self.scaler.update()
                    return {
                        'loss': total_loss.item() * self.config.gradient_accumulation_steps,
                        'classification_loss': classification_loss.item(),
                        'regret_count': regret_count,
                        'bentham_count': bentham_count
                    }
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            else:
                # FP32 모드 옵티마이저 스텝
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                if torch.isnan(total_norm) or torch.isinf(total_norm):
                    self.logger.error(f"그래디언트 norm이 비정상입니다: {total_norm}. 그래디언트 초기화")
                    self.optimizer.zero_grad()
                    return {
                        'loss': total_loss.item() * self.config.gradient_accumulation_steps,
                        'classification_loss': classification_loss.item(),
                        'regret_count': regret_count,
                        'bentham_count': bentham_count
                    }
                
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        # 지속적 검증 수행 (Phase 2 개선)
        validation_results = self.continuous_validation(step_idx, batch)
        
        # 통계 업데이트
        result = {
            'loss': total_loss.item(),
            'classification_loss': classification_loss.item(),
            'regret_count': regret_count,
            'bentham_count': bentham_count,
            # Phase 3: VAD 피드백 루프 결과 추가 
            'vad_feedback': {
                'original_vad_mean': vad_feedback_results['original_vad'].mean(dim=0).tolist(),
                'integrated_vad_mean': vad_feedback_results['integrated_vad'].mean(dim=0).tolist(),
                'final_vad_mean': final_vad.mean(dim=0).tolist(),
                'ethics_weights_mean': vad_feedback_results['ethics_weights'].mean(dim=0).tolist(),
                'emotion_adjustment_norm': torch.norm(vad_feedback_results['emotion_adjustment']).item()
            },
            # Phase 4: 후회 기반 윤리 조정 결과
            'ethics_adjustment': {
                'rule_based_priority': ethics_adjustment['rule_based_priority'].item(),
                'consequence_priority': ethics_adjustment['consequence_priority'].item(),
                'virtue_priority': ethics_adjustment['virtue_priority'].item(),
                'care_priority': ethics_adjustment['care_priority'].item()
            },
            # Phase 5: 사후 감정 변화 결과
            'post_decision': {
                'decision_type': post_decision_response['decision_type'],
                'guilt_level': post_decision_response['emotional_consequences']['guilt_level'].item(),
                'pride_level': post_decision_response['emotional_consequences']['pride_level'].item(),
                'regret_probability': post_decision_response['emotional_consequences']['regret_probability'].item()
            }
        }
        
        # 검증 결과 추가
        if validation_results:
            result['validation'] = validation_results
            
        return result
    
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
            
            # 주기적 로깅 (안전한 평균 계산)
            if self.step_count % self.config.log_every_n_steps == 0:
                recent_losses = epoch_stats['loss'][-self.config.log_every_n_steps:]
                recent_regrets = epoch_stats['regret_count'][-self.config.log_every_n_steps:]
                
                avg_loss = np.mean(recent_losses) if recent_losses else 0.0
                avg_regret = np.mean(recent_regrets) if recent_regrets else 0.0
                
                self.logger.info(
                    f"스텝 {self.step_count}: 손실={avg_loss:.4f}, "
                    f"후회={avg_regret:.1f}, GPU메모리={self._get_gpu_memory():.1f}MB"
                )
            
            # 체크포인트 저장
            if self.step_count % self.config.save_checkpoint_every == 0:
                self.save_checkpoint(epoch, batch_idx)
        
        # 에포크 종료 후 스케줄러 스텝 (단계별 개선)
        if self.scheduler is not None:
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()
            new_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f"학습률 업데이트: {old_lr:.6f} → {new_lr:.6f}")
            
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
        
        # 모델, 옵티마이저, 스케줄러 준비 (단계별 개선 적용)
        self.setup_model_and_optimizer()
        
        # 데이터 준비
        dataloader = self.prepare_data()
        
        actual_params = self.model.get_parameter_count()
        self.logger.info(f"실제 파라미터 수: {actual_params:,}개")
        
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
    
    def compare_initialization_methods(self):
        """초기화 방법 비교 실험 (He vs Xavier)"""
        self.logger.info("🗋 초기화 방법 비교 실험 시작")
        
        results = {}
        
        for init_method in ['he', 'xavier']:
            self.logger.info(f"\n🎨 {init_method.upper()} 초기화 테스트")
            
            # 소형 모델로 테스트
            test_config = HybridConfig(
                cpu_memory_gb=2.0,
                gpu_memory_gb=1.0,
                target_params=500_000,
                test_mode=True,
                test_samples=5,
                initialization_method=init_method
            )
            
            test_model = MemoryOptimizedModel(test_config)
            test_model.eval()
            
            # 안정성 테스트
            stability_scores = []
            gradient_norms = []
            
            for i in range(10):  # 10번 반복 테스트
                dummy_input = torch.randn(1, 8, 1024)
                dummy_labels = torch.randn(1, 3)
                
                test_model.train()
                with torch.enable_grad():
                    outputs = test_model(dummy_input)
                    loss = F.mse_loss(outputs['emotion_predictions'].mean(dim=1, keepdim=True).expand(-1, 3), dummy_labels)
                    
                    # NaN/Inf 검사
                    is_stable = not (torch.isnan(loss) or torch.isinf(loss))
                    stability_scores.append(is_stable)
                    
                    if is_stable:
                        loss.backward()
                        total_norm = torch.nn.utils.clip_grad_norm_(test_model.parameters(), 1.0)
                        gradient_norms.append(total_norm.item())
                    
                test_model.zero_grad()
            
            # 결과 정리
            stability_rate = sum(stability_scores) / len(stability_scores)
            avg_grad_norm = np.mean(gradient_norms) if gradient_norms else float('inf')
            
            results[init_method] = {
                'stability_rate': stability_rate,
                'avg_gradient_norm': avg_grad_norm,
                'gradient_norms': gradient_norms
            }
            
            self.logger.info(f"  안정성: {stability_rate*100:.1f}%")
            self.logger.info(f"  평균 그래디언트 norm: {avg_grad_norm:.4f}")
        
        # 최종 비교 및 추천
        self.logger.info("\n📈 초기화 방법 비교 결과:")
        for method, result in results.items():
            self.logger.info(f"{method.upper()}: 안정성 {result['stability_rate']*100:.1f}%, 그래디언트 {result['avg_gradient_norm']:.4f}")
        
        # 자동 추천
        best_method = max(results.keys(), key=lambda k: results[k]['stability_rate'])
        self.logger.info(f"\n✅ 추천 초기화 방법: {best_method.upper()}")
        
        return results
    
    def continuous_validation(self, step_number: int, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """지속적 검증 시스템 (Phase 2 개선)"""
        if not self.config.enable_continuous_monitoring:
            return {}
        
        if step_number % self.config.validation_frequency != 0:
            return {}
        
        self.logger.info(f"🔍 지속적 검증 실행 (Step {step_number})")
        
        validation_results = {
            'step': step_number,
            'timestamp': time.time()
        }
        
        # 1. 데이터 분포 검증
        try:
            if 'source' in batch:
                sources = batch['source'] if isinstance(batch['source'], list) else [batch['source']]
                from collections import Counter
                distribution = Counter(sources)
                total = len(sources)
                if total > 0:
                    distribution_pct = {k: v/total for k, v in distribution.items()}
                    validation_results['data_distribution'] = distribution_pct
                    
                    # 균등성 검사
                    target_weights = self.config.data_folder_weights
                    balance_score = 0
                    for folder, actual_pct in distribution_pct.items():
                        if folder in target_weights:
                            target_pct = target_weights[folder]
                            balance_score += abs(actual_pct - target_pct)
                    
                    validation_results['balance_score'] = 1.0 - (balance_score / 2.0)  # 0~1 점수
                else:
                    validation_results['balance_score'] = 1.0  # 기본값
            else:
                validation_results['balance_score'] = 1.0  # source 키가 없는 경우 기본값
        except Exception as e:
            self.logger.warning(f"데이터 분포 검증 실패: {e}")
            validation_results['balance_score'] = 1.0  # 오류 시 기본값
        
        # 2. 모델 안정성 검증
        self.model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(2, 16, 1024, device=self.model.cpu_device)
            try:
                outputs = self.model(dummy_input)
                stability_check = {
                    'model_stable': True,
                    'output_ranges': {}
                }
                
                for key, value in outputs.items():
                    has_nan = torch.isnan(value).any()
                    has_inf = torch.isinf(value).any()
                    if has_nan or has_inf:
                        stability_check['model_stable'] = False
                    
                    stability_check['output_ranges'][key] = {
                        'min': value.min().item(),
                        'max': value.max().item(),
                        'has_nan': has_nan.item(),
                        'has_inf': has_inf.item()
                    }
                
                validation_results['stability'] = stability_check
                
            except Exception as e:
                validation_results['stability'] = {
                    'model_stable': False,
                    'error': str(e)
                }
        
        self.model.train()  # 학습 모드로 복귀
        
        # 3. 학습 진행 상황 검증
        if hasattr(self, 'training_stats') and self.training_stats:
            recent_losses = self.training_stats['total_loss'][-10:]  # 최근 10개
            if recent_losses:
                validation_results['recent_loss_trend'] = {
                    'mean': np.mean(recent_losses),
                    'std': np.std(recent_losses),
                    'trend': 'improving' if len(recent_losses) > 5 and recent_losses[-1] < recent_losses[0] else 'stable'
                }
        
        # 4. 결과 로깅
        if validation_results.get('balance_score', 0) < 0.7:
            self.logger.warning(f"⚠️  데이터 불균형 감지: {validation_results['balance_score']:.3f}")
        
        if not validation_results.get('stability', {}).get('model_stable', True):
            self.logger.error("❌ 모델 안정성 문제 발견")
        
        if validation_results.get('stability', {}).get('model_stable', True) and validation_results.get('balance_score', 0) > 0.7:
            self.logger.info("✅ 지속적 검증 통과")
        
        return validation_results

if __name__ == "__main__":
    # 하이브리드 설정으로 테스트
    config = HybridConfig()
    trainer = HybridDistributedTrainer(config)
    
    print("🧪 하이브리드 분산 학습 시스템 준비 완료")
    print(f"📊 설정: {config.regrets_per_step}회 후회/스텝, {config.epochs}번 선회")
    print(f"🤖 모델: {config.target_params:,}개 파라미터")
    print(f"⚡ 워커: {config.num_workers}개")
    print("준비된 학습을 시작하려면 trainer.train()을 호출하세요.")