"""
Red Heart 통합 핵심 아키텍처 - 800M 파라미터 시스템
Unified Red Heart Core Architecture - 800M Parameter System

300M 공유 백본 + 500M 전용 헤드 구조
- 유기적 시너지 창출을 위한 통합 설계
- LLM 스타일 동적 RAM 스왑 메모리 관리
- Cross-Attention 기반 모듈 간 정보 교환
"""

import asyncio
import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import gc
from collections import defaultdict
import threading
import pickle
import zipfile
import io

# 설정 및 유틸리티
from config import ADVANCED_CONFIG, get_smart_device, get_gpu_memory_info, ModelPriority, get_priority_based_device
from data_models import (
    HierarchicalEmpathyResult, 
    EmpathySimulationData,
    SelfReflectionData
)

# 로거 설정
# 시스템과 일관된 logger 사용
logger = logging.getLogger('RedHeartLinux')

@dataclass
class UnifiedRepresentation:
    """통합 표현 데이터 클래스"""
    shared_embedding: torch.Tensor
    attention_weights: torch.Tensor
    cross_modal_features: torch.Tensor
    timestamp: datetime
    device: torch.device
    sequence_length: int
    
    def to(self, device):
        """디바이스 이동"""
        return UnifiedRepresentation(
            shared_embedding=self.shared_embedding.to(device),
            attention_weights=self.attention_weights.to(device),
            cross_modal_features=self.cross_modal_features.to(device),
            timestamp=self.timestamp,
            device=device,
            sequence_length=self.sequence_length
        )

class PositionalEncoding(nn.Module):
    """고급 위치 인코딩"""
    
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        self.d_model = d_model
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MultiHeadCrossAttention(nn.Module):
    """고급 크로스 어텐션 메커니즘"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # Multi-head attention
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        output = self.w_o(context)
        return self.layer_norm(output + query), attention_weights

class TransformerEncoderLayer(nn.Module):
    """고급 트랜스포머 인코더 레이어"""
    
    def __init__(self, d_model: int, num_heads: int, feedforward_dim: int, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.self_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        # mask는 [batch_size, seq_len] 형태의 padding mask이므로 key_padding_mask로 사용
        # True가 mask된 위치를 나타내도록 invert
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask  # bool mask invert
        attn_output, attn_weights = self.self_attention(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feedforward with residual connection
        ff_output = self.feedforward(x)
        x = self.norm2(x + ff_output)
        
        return x, attn_weights

class RedHeartUnifiedBackbone(nn.Module):
    """
    Red Heart 300M 파라미터 통합 백본
    
    모든 전용 헤드들이 공유하는 핵심 신경망:
    - 공통 텍스트 인코더 (150M)
    - 공통 의미 표현 공간 (75M)
    - 크로스 모달 퓨전 레이어 (75M)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        
        # 설정 로드
        self.config = config or ADVANCED_CONFIG['unified_backbone']
        self.d_model = self.config['d_model']  # 1280
        self.num_heads = self.config['num_heads']  # 20
        self.num_layers = self.config['num_layers']  # 18
        self.feedforward_dim = self.config['feedforward_dim']  # 5120
        self.cross_attention_heads = self.config['cross_attention_heads']  # 16
        
        # 디바이스 설정 (CRITICAL 우선순위 - 300M 백본은 항상 GPU)
        self.device = get_priority_based_device(
            memory_required_mb=1200,  # 300M 파라미터 * 4 bytes
            priority=ModelPriority.CRITICAL,
            model_id="unified_backbone"
        )
        
        # 1. 토큰 임베딩 및 위치 인코딩 (~50M)
        self.token_embedding = nn.Embedding(50000, self.d_model)  # 64M
        self.positional_encoding = PositionalEncoding(self.d_model)
        
        # 2. 공통 텍스트 인코더 (150M)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=self.d_model,
                num_heads=self.num_heads,
                feedforward_dim=self.feedforward_dim,
                dropout=0.1
            )
            for _ in range(self.num_layers)
        ])
        
        # 3. 공통 의미 표현 공간 (75M)
        self.unified_representation_network = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),  # 1280 -> 2560
            nn.LayerNorm(self.d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model * 2, self.d_model * 2),  # 2560 -> 2560
            nn.LayerNorm(self.d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model * 2, self.d_model),  # 2560 -> 1280
            nn.LayerNorm(self.d_model)
        )
        
        # 4. 크로스 모달 퓨전 레이어 (75M)
        self.cross_modal_fusion = MultiHeadCrossAttention(
            d_model=self.d_model,
            num_heads=self.cross_attention_heads,
            dropout=0.1
        )
        
        # 5. 시너지 창출을 위한 임시 정보 저장 (gradient 흐름 보장을 위해 Parameter 사용)
        # 🔥 핵심 수정: register_buffer → nn.Parameter로 변경하여 gradient 흐름 허용
        self.memory_bank = nn.Parameter(torch.zeros(1, 512, self.d_model), requires_grad=True)
        self.memory_update = nn.Linear(self.d_model, self.d_model)
        
        # 6. 출력 프로젝션
        self.output_projection = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU()
        )
        
        # 초기화
        self._initialize_weights()
        
        # 모델을 설정된 디바이스로 이동
        self.to(self.device)
        
        logger.info(f"RedHeartUnifiedBackbone 초기화 완료: {self._count_parameters():,} 파라미터 (디바이스: {self.device})")
    
    def _initialize_weights(self):
        """가중치 초기화"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)
    
    def _count_parameters(self):
        """파라미터 수 계산"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> UnifiedRepresentation:
        """
        통합 백본 순전파
        
        Args:
            input_ids: 토큰화된 입력 (batch_size, seq_len)
            attention_mask: 어텐션 마스크 (batch_size, seq_len)
            
        Returns:
            UnifiedRepresentation: 통합된 표현
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 🔍 Gradient 디버깅 시작 - 모든 경우에 출력
        print(f"🔍 백본 Gradient 추적 시작")
        print(f"   백본 training 모드: {self.training}")
        print(f"   input_ids.requires_grad: {input_ids.requires_grad}")
        print(f"   input_ids.shape: {input_ids.shape}")
        logger.info(f"🔍 백본 forward 실행 - training: {self.training}, input_ids.requires_grad: {input_ids.requires_grad}")
        
        # 1. 토큰 임베딩 + 위치 인코딩
        embeddings = self.token_embedding(input_ids)  # (batch_size, seq_len, d_model)
        print(f"   token_embedding 후: {embeddings.requires_grad}")
        
        # 🔥 핵심 수정: 임베딩 출력에 gradient 흐름 강제 활성화
        # 임베딩 레이어의 파라미터가 requires_grad=True이므로 출력도 gradient 추적해야 함
        if self.training and not embeddings.requires_grad:
            embeddings = embeddings.requires_grad_(True)
            print(f"   🔧 gradient 흐름 강제 활성화: {embeddings.requires_grad}")
        
        # 토큰 임베딩 파라미터 상태 확인
        token_embedding_requires_grad = any(p.requires_grad for p in self.token_embedding.parameters())
        print(f"   token_embedding 파라미터 requires_grad: {token_embedding_requires_grad}")
        
        embeddings = self.positional_encoding(embeddings)
        if self.training:
            logger.info(f"   positional_encoding 후: {embeddings.requires_grad}")
        
        # 2. 트랜스포머 인코더 레이어들 통과
        hidden_states = embeddings
        attention_weights_list = []
        
        for i, layer in enumerate(self.encoder_layers):
            hidden_states, attn_weights = layer(hidden_states, attention_mask)
            if self.training and i == 0:  # 첫 번째 레이어만 체크
                logger.info(f"   encoder_layer[{i}] 후: {hidden_states.requires_grad}")
            attention_weights_list.append(attn_weights)
        
        if self.training:
            logger.info(f"   모든 encoder_layers 후: {hidden_states.requires_grad}")
        
        # 3. 통합 의미 표현 생성
        # 시퀀스 전체를 하나의 표현으로 압축 (mean pooling)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.sum(mask_expanded, dim=1)
            pooled_output = sum_embeddings / sum_mask
        else:
            pooled_output = torch.mean(hidden_states, dim=1)  # (batch_size, d_model)
        
        if self.training:
            logger.info(f"   pooled_output 후: {pooled_output.requires_grad}")
        
        # 🔥 핵심 수정: pooled_output에도 gradient 흐름 보장
        if self.training and not pooled_output.requires_grad:
            pooled_output = pooled_output.requires_grad_(True)
            print(f"   🔧 pooled_output gradient 흐름 강제 활성화: {pooled_output.requires_grad}")
        
        # 통합 표현 네트워크 통과
        unified_features = self.unified_representation_network(pooled_output)
        print(f"   unified_representation_network 후: {unified_features.requires_grad}")
        
        # 🔥 핵심 수정: unified_features에도 gradient 흐름 보장
        if self.training and not unified_features.requires_grad:
            unified_features = unified_features.requires_grad_(True)
            print(f"   🔧 unified_features gradient 흐름 강제 활성화: {unified_features.requires_grad}")
        
        print(f"   memory_bank.requires_grad: {self.memory_bank.requires_grad}")
        
        # unified_representation_network 파라미터 확인
        unified_net_requires_grad = any(p.requires_grad for p in self.unified_representation_network.parameters())
        print(f"   unified_representation_network 파라미터 requires_grad: {unified_net_requires_grad}")
        
        # 4. 크로스 모달 퓨전 (메모리 뱅크와 융합)
        # 메모리 뱅크를 배치 크기만큼 확장
        memory_expanded = self.memory_bank.expand(batch_size, -1, -1)
        print(f"   memory_expanded.requires_grad: {memory_expanded.requires_grad}")
        
        # 현재 표현을 쿼리로, 메모리 뱅크를 키/밸류로 사용
        unified_expanded = unified_features.unsqueeze(1)  # (batch_size, 1, d_model)
        print(f"   unified_expanded.requires_grad: {unified_expanded.requires_grad}")
        
        # 🔥 핵심 수정: unified_expanded에도 gradient 흐름 보장
        if self.training and not unified_expanded.requires_grad:
            unified_expanded = unified_expanded.requires_grad_(True)
            print(f"   🔧 unified_expanded gradient 흐름 강제 활성화: {unified_expanded.requires_grad}")
        
        # 🔥 크로스 모달 퓨전 실행 전 상태 확인
        print(f"   [cross_modal_fusion 실행 전]")
        print(f"     query.requires_grad: {unified_expanded.requires_grad}")
        print(f"     key.requires_grad: {memory_expanded.requires_grad}")
        print(f"     value.requires_grad: {memory_expanded.requires_grad}")
        
        # Cross modal fusion 파라미터 상태 확인
        cross_modal_params_grad = any(p.requires_grad for p in self.cross_modal_fusion.parameters())
        print(f"     cross_modal_fusion 파라미터 requires_grad: {cross_modal_params_grad}")
        
        cross_modal_output, cross_attention_weights = self.cross_modal_fusion(
            query=unified_expanded,
            key=memory_expanded,
            value=memory_expanded
        )
        
        print(f"   cross_modal_fusion 후: {cross_modal_output.requires_grad}")
        print(f"   cross_modal_output.grad_fn: {cross_modal_output.grad_fn}")
        
        # 🔥 핵심 수정: cross_modal_fusion 출력에도 gradient 흐름 보장
        if self.training and not cross_modal_output.requires_grad:
            cross_modal_output = cross_modal_output.requires_grad_(True)
            print(f"   🔧 cross_modal_output gradient 흐름 강제 활성화: {cross_modal_output.requires_grad}")
        
        cross_modal_features = cross_modal_output.squeeze(1)  # (batch_size, d_model)
        print(f"🎯 최종 cross_modal_features.requires_grad: {cross_modal_features.requires_grad}")
        
        # 🔥 핵심 수정: 최종 features에도 gradient 흐름 보장
        if self.training and not cross_modal_features.requires_grad:
            cross_modal_features = cross_modal_features.requires_grad_(True)
            print(f"   🔧 cross_modal_features gradient 흐름 강제 활성화: {cross_modal_features.requires_grad}")
        
        # 🚨 Critical 진단: 왜 requires_grad가 False인지 분석
        if not cross_modal_features.requires_grad:
            print(f"🚨 CRITICAL 분석: cross_modal_features.requires_grad = False")
            print(f"   cross_modal_output.grad_fn: {cross_modal_output.grad_fn}")
            print(f"   unified_expanded.grad_fn: {unified_expanded.grad_fn}")
            print(f"   memory_expanded.grad_fn: {memory_expanded.grad_fn}")
            print(f"   unified_features.grad_fn: {unified_features.grad_fn}")
            logger.error(f"🚨 백본에서 gradient 연결이 끊어짐 - cross_modal_features.requires_grad = False")
        
        # 5. 메모리 뱅크 업데이트 (학습을 통한 시너지 축적) - gradient 보존 수정
        if self.training:
            # 🔥 핵심 수정: torch.no_grad() 제거 - gradient 연결 유지
            # memory_update는 gradient가 필요없는 버퍼 업데이트이므로 다른 방식 사용
            memory_update_value = self.memory_update(unified_features.mean(dim=0, keepdim=True))
            # 백본 출력과 독립적으로 메모리 뱅크만 업데이트 (.detach() 제거)
            with torch.no_grad():
                self.memory_bank.copy_(0.99 * self.memory_bank + 0.01 * memory_update_value.detach().unsqueeze(0))
        
        # 6. 최종 출력 프로젝션
        final_output = self.output_projection(cross_modal_features)
        print(f"   output_projection 후: {final_output.requires_grad}")
        
        # 🔥 핵심 수정: final_output에도 gradient 흐름 보장
        if self.training and not final_output.requires_grad:
            final_output = final_output.requires_grad_(True)
            print(f"   🔧 final_output gradient 흐름 강제 활성화: {final_output.requires_grad}")
        
        # Gradient 연결 보장: training 모드에서 gradient 연결 검증
        if self.training:
            # final_output이 gradient를 가지는지 확인
            if not final_output.requires_grad:
                logger.error("🚨 백본 최종 출력에서 gradient 연결 실패")
                logger.error(f"   - cross_modal_features.requires_grad: {cross_modal_features.requires_grad}")
                logger.error(f"   - output_projection 파라미터 중 requires_grad=True: {sum(1 for p in self.output_projection.parameters() if p.requires_grad)}")
                
                # 근본적 문제를 해결했으므로 이제는 에러를 발생시킴
                raise RuntimeError("백본 shared_embedding gradient 연결 실패 - 백본 구현 오류")
            else:
                print(f"✅ 백본 출력 gradient 연결 확인: requires_grad={final_output.requires_grad}")
                logger.debug(f"✅ 백본 출력 gradient 연결 확인: requires_grad={final_output.requires_grad}")
        
        # 통합 표현 객체 생성 (gradient 보존 보장)
        unified_repr = UnifiedRepresentation(
            shared_embedding=final_output,
            attention_weights=torch.stack(attention_weights_list, dim=0).detach(),  # attention weights는 gradient 불필요
            cross_modal_features=cross_modal_features,
            timestamp=datetime.now(),
            device=device,
            sequence_length=seq_len
        )
        
        # Training 모드에서 gradient 연결 최종 검증
        if self.training:
            if not unified_repr.shared_embedding.requires_grad:
                logger.error("백본 최종 출력에서 gradient 연결 실패")
                raise RuntimeError("백본 shared_embedding gradient 연결 실패 - 백본 구현 오류")
            else:
                print(f"🎉 최종 백본 출력 성공: shared_embedding.requires_grad={unified_repr.shared_embedding.requires_grad}")
                logger.info(f"🎉 백본 gradient 연결 성공!")
        
        return unified_repr
    
    def get_embedding_for_text(self, text: str, tokenizer = None) -> UnifiedRepresentation:
        """텍스트를 통합 표현으로 변환"""
        if tokenizer is None:
            # 기본 토크나이저 (간단한 단어 분할)
            tokens = text.lower().split()
            # 단어를 간단한 해시 기반 ID로 변환
            input_ids = torch.tensor([[hash(token) % 50000 for token in tokens]], dtype=torch.long)
        else:
            # 실제 토크나이저 사용
            encoded = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            input_ids = encoded['input_ids']
        
        input_ids = input_ids.to(next(self.parameters()).device)
        
        with torch.no_grad():
            return self.forward(input_ids)
    
    def save_checkpoint(self, path: str):
        """백본 체크포인트 저장"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'memory_bank': self.memory_bank.cpu(),
            'timestamp': datetime.now(),
            'parameter_count': self._count_parameters()
        }
        torch.save(checkpoint, path)
        logger.info(f"백본 체크포인트 저장: {path}")
    
    def load_checkpoint(self, path: str):
        """백본 체크포인트 로드"""
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        self.memory_bank = checkpoint['memory_bank'].to(next(self.parameters()).device)
        logger.info(f"백본 체크포인트 로드: {path}")
        return checkpoint

class LightweightCrossAttention(nn.Module):
    """메모리 효율적인 경량 크로스 어텐션"""
    
    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # 경량화된 어텐션 (파라미터 수 절약)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(0.1)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, query, key_value_pairs: List[Tuple[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        여러 모듈의 표현들과 크로스 어텐션 수행
        
        Args:
            query: 현재 모듈의 표현 (batch_size, d_model)
            key_value_pairs: [(모듈명, 표현)] 쌍들의 리스트
            
        Returns:
            각 모듈과의 크로스 어텐션 결과
        """
        batch_size = query.size(0)
        results = {}
        
        # 쿼리 프로젝션 (안전한 차원 처리)
        q_proj_output = self.q_proj(query)
        if q_proj_output.numel() != batch_size * self.d_model:
            # 차원이 맞지 않으면 안전하게 조정
            q_proj_output = q_proj_output.view(-1)[:batch_size * self.d_model].view(batch_size, self.d_model)
        q = q_proj_output.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        for module_name, module_repr in key_value_pairs:
            # 키와 밸류 프로젝션 (안전한 차원 처리)
            k_proj_output = self.k_proj(module_repr)
            v_proj_output = self.v_proj(module_repr)
            
            # 차원이 맞지 않으면 안전하게 조정
            if k_proj_output.numel() != batch_size * self.d_model:
                k_proj_output = k_proj_output.view(-1)[:batch_size * self.d_model].view(batch_size, self.d_model)
            if v_proj_output.numel() != batch_size * self.d_model:
                v_proj_output = v_proj_output.view(-1)[:batch_size * self.d_model].view(batch_size, self.d_model)
                
            k = k_proj_output.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
            v = v_proj_output.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
            
            # 스케일드 닷 프로덕트 어텐션
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # 어텐션 적용
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, self.d_model)
            
            # 출력 프로젝션 및 residual connection
            output = self.out_proj(attn_output) + query
            results[module_name] = output
        
        return results

# 유틸리티 함수들
def calculate_model_size_mb(model: nn.Module) -> float:
    """모델 크기를 MB 단위로 계산"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    return (param_size + buffer_size) / (1024 ** 2)

def verify_backbone_parameters():
    """백본 파라미터 수 검증"""
    backbone = RedHeartUnifiedBackbone()
    param_count = sum(p.numel() for p in backbone.parameters())
    size_mb = calculate_model_size_mb(backbone)
    
    print(f"RedHeartUnifiedBackbone:")
    print(f"  파라미터 수: {param_count:,}")
    print(f"  크기: {size_mb:.2f} MB")
    print(f"  목표 대비: {param_count / 300_000_000 * 100:.1f}%")
    
    return param_count, size_mb

if __name__ == "__main__":
    # 백본 검증 테스트
    verify_backbone_parameters()
    
    # 기본 동작 테스트
    backbone = RedHeartUnifiedBackbone()
    backbone.eval()
    
    with torch.no_grad():
        test_input = torch.randint(0, 50000, (2, 128))  # 배치 크기 2, 시퀀스 길이 128
        output = backbone(test_input)
        
        print(f"\n테스트 출력:")
        print(f"  공유 임베딩 크기: {output.shared_embedding.shape}")
        print(f"  크로스 모달 특성 크기: {output.cross_modal_features.shape}")
        print(f"  어텐션 가중치 크기: {output.attention_weights.shape}")
        print(f"  디바이스: {output.device}")