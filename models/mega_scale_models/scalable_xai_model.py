"""
확장 가능한 XAI 모델 (2억 파라미터)
Scalable XAI Model (200M Parameters)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import math
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import time
from datetime import datetime

# XAI 및 LLM 모듈 import
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from xai_core.xai_logging_system import xai_logger, xai_trace, xai_decision_point
from llm_module import llm_tracker, register_llm, ask_llm, LLMConfig

@dataclass
class MegaScaleConfig:
    """메가 스케일 모델 설정"""
    # 기본 차원 설정
    input_dim: int = 1024
    hidden_dim: int = 2048
    num_layers: int = 48
    num_attention_heads: int = 32
    intermediate_size: int = 8192
    
    # 전문화 모듈 설정
    emotion_layers: int = 12
    semantic_layers: int = 12
    reasoning_layers: int = 12
    integration_layers: int = 12
    
    # 성능 최적화
    use_gradient_checkpointing: bool = True
    use_mixed_precision: bool = True
    dropout_rate: float = 0.02  # Loss NaN 방지를 위한 Dropout 감소
    layer_norm_eps: float = 1e-6  # Loss NaN 방지를 위한 수치 안정성 개선
    
    # XAI 설정
    enable_xai_tracking: bool = True
    explanation_depth: int = 5
    llm_integration: bool = True
    
    def get_total_params(self) -> int:
        """총 파라미터 수 계산"""
        # 트랜스포머 블록들
        transformer_params = self.num_layers * (
            4 * self.hidden_dim * self.hidden_dim +  # attention
            3 * self.hidden_dim * self.intermediate_size +  # FFN
            4 * self.hidden_dim  # layer norms and biases
        )
        
        # 입출력 레이어들
        io_params = (
            self.input_dim * self.hidden_dim +  # input projection
            self.hidden_dim * 6 +  # emotion output
            self.hidden_dim * 1000 +  # semantic output
            self.hidden_dim * 128 +  # reasoning output
            self.hidden_dim * 512  # integration output
        )
        
        return transformer_params + io_params

class MultiHeadAttentionXL(nn.Module):
    """확장된 멀티헤드 어텐션"""
    
    def __init__(self, config: MegaScaleConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_dim // self.num_heads
        
        assert self.head_dim * self.num_heads == self.hidden_dim
        
        # Query, Key, Value 프로젝션
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        
        # 상대적 위치 인코딩
        self.relative_position_encoding = nn.Parameter(
            torch.randn(2 * 512 - 1, self.head_dim)
        )
        
        # 어텐션 드롭아웃
        self.attention_dropout = nn.Dropout(config.dropout_rate)
        self.output_dropout = nn.Dropout(config.dropout_rate)
        
        # 스케일링 팩터
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Q, K, V 계산
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # 헤드별로 분할
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 어텐션 스코어 계산
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        
        # 상대적 위치 인코딩 추가
        if seq_len <= 512:
            rel_pos_bias = self._get_relative_position_bias(seq_len)
            attention_scores = attention_scores + rel_pos_bias.unsqueeze(0).unsqueeze(0)
        
        # 어텐션 마스크 적용
        if attention_mask is not None:
            # 마스크 차원 맞추기
            if attention_mask.dim() == 3:  # [batch, seq_len, seq_len]
                attention_mask = attention_mask.unsqueeze(1)  # [batch, 1, seq_len, seq_len]
            attention_scores = attention_scores + attention_mask
        
        # 어텐션 가중치 계산
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # 어텐션 적용
        context = torch.matmul(attention_weights, value)
        
        # 헤드 합치기
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_dim
        )
        
        # 출력 프로젝션
        output = self.o_proj(context)
        output = self.output_dropout(output)
        
        if return_attention:
            return output, attention_weights
        return output, None
    
    def _get_relative_position_bias(self, seq_len: int) -> torch.Tensor:
        """상대적 위치 편향 계산"""
        positions = torch.arange(seq_len, device=self.relative_position_encoding.device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        relative_positions = relative_positions + 512 - 1  # 최대 길이 기준
        relative_positions = torch.clamp(relative_positions, 0, 2 * 512 - 2)
        
        bias = self.relative_position_encoding[relative_positions]
        # [seq_len, seq_len, head_dim] -> [seq_len, seq_len]로 축소
        return bias.mean(dim=-1)

class FeedForwardXL(nn.Module):
    """확장된 피드포워드 네트워크"""
    
    def __init__(self, config: MegaScaleConfig):
        super().__init__()
        self.config = config
        
        # Gated Linear Unit 사용
        self.gate_proj = nn.Linear(config.hidden_dim, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_dim, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_dim, bias=False)
        
        self.dropout = nn.Dropout(config.dropout_rate)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(hidden_states))  # SwiGLU 활성화
        up = self.up_proj(hidden_states)
        intermediate = gate * up
        intermediate = self.dropout(intermediate)
        output = self.down_proj(intermediate)
        return output

class TransformerLayerXL(nn.Module):
    """확장된 트랜스포머 레이어"""
    
    def __init__(self, config: MegaScaleConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # 어텐션
        self.attention = MultiHeadAttentionXL(config)
        self.attention_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        
        # 피드포워드
        self.feed_forward = FeedForwardXL(config)
        self.ffn_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        
        # 전문화 모듈 (일부 레이어에만)
        self.specialized_module = None
        if layer_idx % 4 == 0:  # 4번째마다 전문화 모듈
            self.specialized_module = self._create_specialized_module()
    
    def _create_specialized_module(self) -> nn.Module:
        """전문화 모듈 생성"""
        if self.layer_idx < self.config.emotion_layers:
            return EmotionSpecializedModule(self.config)
        elif self.layer_idx < self.config.emotion_layers + self.config.semantic_layers:
            return SemanticSpecializedModule(self.config)
        elif self.layer_idx < self.config.emotion_layers + self.config.semantic_layers + self.config.reasoning_layers:
            return ReasoningSpecializedModule(self.config)
        else:
            return IntegrationSpecializedModule(self.config)
    
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        # 어텐션 블록 (Post-norm 안정성 개선)
        attention_output, attention_weights = self.attention(
            hidden_states, attention_mask, return_attention
        )
        hidden_states = hidden_states + attention_output
        hidden_states = self.attention_norm(hidden_states)
        
        # 피드포워드 블록 (Post-norm 안정성 개선)
        ffn_output = self.feed_forward(hidden_states)
        hidden_states = hidden_states + ffn_output
        hidden_states = self.ffn_norm(hidden_states)
        
        # 전문화 모듈 (있는 경우)
        if self.specialized_module is not None:
            specialized_output = self.specialized_module(hidden_states)
            hidden_states = hidden_states + specialized_output
        
        return hidden_states, attention_weights

class EmotionSpecializedModule(nn.Module):
    """감정 전문화 모듈"""
    
    def __init__(self, config: MegaScaleConfig):
        super().__init__()
        self.emotion_projector = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim),
            nn.Dropout(config.dropout_rate)
        )
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.emotion_projector(hidden_states) * 0.1  # 잔차 연결을 위해 작은 스케일

class SemanticSpecializedModule(nn.Module):
    """의미 전문화 모듈"""
    
    def __init__(self, config: MegaScaleConfig):
        super().__init__()
        self.semantic_analyzer = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Dropout(config.dropout_rate)
        )
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.semantic_analyzer(hidden_states) * 0.1

class ReasoningSpecializedModule(nn.Module):
    """추론 전문화 모듈"""
    
    def __init__(self, config: MegaScaleConfig):
        super().__init__()
        self.reasoning_network = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.Dropout(config.dropout_rate)
        )
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.reasoning_network(hidden_states) * 0.1

class IntegrationSpecializedModule(nn.Module):
    """통합 전문화 모듈"""
    
    def __init__(self, config: MegaScaleConfig):
        super().__init__()
        self.integration_hub = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Dropout(config.dropout_rate)
        )
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.integration_hub(hidden_states) * 0.1

class MegaScaleXAIModel(nn.Module):
    """메가 스케일 XAI 모델 (2억 파라미터)"""
    
    def __init__(self, config: MegaScaleConfig):
        super().__init__()
        self.config = config
        
        # 입력 임베딩
        self.input_projection = nn.Linear(config.input_dim, config.hidden_dim)
        self.input_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        
        # 트랜스포머 레이어들
        self.transformer_layers = nn.ModuleList([
            TransformerLayerXL(config, i) for i in range(config.num_layers)
        ])
        
        # 최종 정규화
        self.final_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        
        # 출력 헤드들
        self.emotion_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 2, 6),  # 6차원 감정
            nn.Tanh()
        )
        
        self.semantic_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, 1000),  # 의미 클래스
            nn.Softmax(dim=-1)
        )
        
        self.reasoning_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 4, 128)  # 추론 특징
        )
        
        self.integration_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, 512),  # 통합 특징
            nn.Tanh()
        )
        
        # XAI 추적기
        if config.enable_xai_tracking:
            self.xai_tracker = XAITracker(config)
        
        # LLM 통합
        if config.llm_integration:
            self.llm_integration_layer = LLMIntegrationLayer(config)
        
        # 파라미터 초기화
        self.apply(self._init_weights)
        
        print(f"✅ MegaScaleXAIModel 초기화 완료: {self.get_parameter_count():,}개 파라미터")
    
    def _init_weights(self, module):
        """가중치 초기화"""
        if isinstance(module, nn.Linear):
            # Xavier uniform 초기화
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0)
            torch.nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Parameter):
            torch.nn.init.normal_(module, mean=0.0, std=0.02)
    
    def get_parameter_count(self) -> int:
        """파라미터 수 계산"""
        return sum(p.numel() for p in self.parameters())
    
    @xai_trace("MegaScaleXAI", "inference")
    def forward(self, input_embeddings: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                return_hidden_states: bool = False,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        
        batch_size, seq_len, input_dim = input_embeddings.shape
        
        # 입력 프로젝션
        hidden_states = self.input_projection(input_embeddings)
        hidden_states = self.input_norm(hidden_states)
        
        # 어텐션 마스크 준비
        if attention_mask is None:
            attention_mask = None  # None으로 유지
        elif attention_mask.dim() == 2:
            # [batch, seq_len] -> [batch, seq_len, seq_len]
            attention_mask = attention_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        # 트랜스포머 레이어들 통과
        all_hidden_states = []
        all_attention_weights = []
        
        for i, layer in enumerate(self.transformer_layers):
            if self.config.use_gradient_checkpointing and self.training:
                # 그래디언트 체크포인팅 사용
                hidden_states, attention_weights = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, attention_mask, return_attention
                )
            else:
                hidden_states, attention_weights = layer(
                    hidden_states, attention_mask, return_attention
                )
            
            if return_hidden_states:
                all_hidden_states.append(hidden_states)
            if return_attention and attention_weights is not None:
                all_attention_weights.append(attention_weights)
            
            # XAI 추적 (일부 레이어만)
            if self.config.enable_xai_tracking and i % 8 == 0:
                self.xai_tracker.track_layer_output(i, hidden_states)
        
        # 최종 정규화
        hidden_states = self.final_norm(hidden_states)
        
        # 평균 풀링으로 문장 표현 생성
        pooled_output = hidden_states.mean(dim=1)  # 단순 평균 풀링
        
        # 각 헤드별 출력 계산
        emotion_output = self.emotion_head(pooled_output)
        semantic_output = self.semantic_head(pooled_output)
        reasoning_output = self.reasoning_head(pooled_output)
        integration_output = self.integration_head(pooled_output)
        
        results = {
            'emotion_predictions': emotion_output,
            'semantic_predictions': semantic_output,
            'reasoning_features': reasoning_output,
            'integration_features': integration_output,
            'pooled_output': pooled_output
        }
        
        # LLM 통합 (활성화된 경우)
        if self.config.llm_integration and hasattr(self, 'llm_integration_layer'):
            llm_output = self.llm_integration_layer(pooled_output, results)
            results['llm_analysis'] = llm_output
        
        # XAI 분석 추가
        if self.config.enable_xai_tracking:
            xai_analysis = self.xai_tracker.generate_explanation(results)
            results['xai_explanation'] = xai_analysis
        
        # 선택적 반환
        if return_hidden_states:
            results['hidden_states'] = all_hidden_states
        if return_attention:
            results['attention_weights'] = all_attention_weights
        
        return results

class XAITracker(nn.Module):
    """XAI 추적 모듈"""
    
    def __init__(self, config: MegaScaleConfig):
        super().__init__()
        self.config = config
        self.layer_outputs = {}
        
        # 설명 생성 네트워크
        self.explanation_generator = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.explanation_depth),
            nn.Softmax(dim=-1)
        )
        
    def track_layer_output(self, layer_idx: int, hidden_states: torch.Tensor):
        """레이어 출력 추적"""
        self.layer_outputs[layer_idx] = {
            'mean_activation': hidden_states.mean().item(),
            'std_activation': hidden_states.std().item(),
            'max_activation': hidden_states.max().item(),
            'min_activation': hidden_states.min().item(),
            'shape': hidden_states.shape
        }
    
    def generate_explanation(self, model_outputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """설명 생성"""
        pooled_output = model_outputs['pooled_output']
        
        # 설명 가중치 생성
        explanation_weights = self.explanation_generator(pooled_output)
        
        # 특징 중요도 분석
        feature_importance = {
            'emotion_contribution': torch.abs(model_outputs['emotion_predictions']).mean().item(),
            'semantic_contribution': torch.abs(model_outputs['semantic_predictions']).mean().item(),
            'reasoning_contribution': torch.abs(model_outputs['reasoning_features']).mean().item(),
            'integration_contribution': torch.abs(model_outputs['integration_features']).mean().item()
        }
        
        return {
            'explanation_weights': explanation_weights,
            'feature_importance': feature_importance,
            'layer_activations': self.layer_outputs,
            'confidence_score': self._calculate_confidence(model_outputs),
            'decision_path': self._extract_decision_path(model_outputs)
        }
    
    def _calculate_confidence(self, outputs: Dict[str, torch.Tensor]) -> float:
        """신뢰도 계산"""
        # 각 출력의 엔트로피 기반 신뢰도
        emotion_entropy = -torch.sum(torch.abs(outputs['emotion_predictions']) * 
                                   torch.log(torch.abs(outputs['emotion_predictions']) + 1e-8), dim=-1)
        semantic_entropy = -torch.sum(outputs['semantic_predictions'] * 
                                    torch.log(outputs['semantic_predictions'] + 1e-8), dim=-1)
        
        avg_entropy = (emotion_entropy + semantic_entropy).mean().item()
        confidence = 1.0 / (1.0 + avg_entropy)  # 엔트로피가 낮을수록 높은 신뢰도
        
        return confidence
    
    def _extract_decision_path(self, outputs: Dict[str, torch.Tensor]) -> List[str]:
        """의사결정 경로 추출"""
        path = []
        
        # 감정 기반 결정
        emotion_max = torch.argmax(torch.abs(outputs['emotion_predictions']), dim=-1)
        if emotion_max.numel() == 1:
            path.append(f"주요 감정 차원: {emotion_max.item()}")
        else:
            path.append(f"주요 감정 차원: {emotion_max[0].item()}")
        
        # 의미 기반 결정
        semantic_max = torch.argmax(outputs['semantic_predictions'], dim=-1)
        if semantic_max.numel() == 1:
            path.append(f"주요 의미 클래스: {semantic_max.item()}")
        else:
            path.append(f"주요 의미 클래스: {semantic_max[0].item()}")
        
        # 추론 기반 결정
        reasoning_norm = torch.norm(outputs['reasoning_features'], dim=-1)
        path.append(f"추론 강도: {reasoning_norm.mean().item():.3f}")
        
        return path

class LLMIntegrationLayer(nn.Module):
    """LLM 통합 레이어"""
    
    def __init__(self, config: MegaScaleConfig):
        super().__init__()
        self.config = config
        
        # LLM 쿼리 생성기
        self.query_generator = nn.Sequential(
            nn.Linear(config.hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # LLM 응답 인코더
        self.response_encoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, config.hidden_dim)
        )
        
    @xai_decision_point()
    def forward(self, pooled_output: torch.Tensor, 
                model_outputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        
        # 쿼리 생성
        query_features = self.query_generator(pooled_output)
        
        # LLM에게 보낼 쿼리 구성
        emotion_summary = self._summarize_emotions(model_outputs['emotion_predictions'])
        semantic_summary = self._summarize_semantics(model_outputs['semantic_predictions'])
        
        llm_query = f"""
        감정 분석 결과: {emotion_summary}
        의미 분석 결과: {semantic_summary}
        
        이 분석 결과를 바탕으로 다음을 설명해주세요:
        1. 주요 감정 패턴
        2. 의미적 특징
        3. 추론 과정
        """
        
        # LLM 호출 (비동기적으로)
        try:
            if 'gpt2' in llm_tracker.llm_models:
                llm_response = ask_llm('gpt2', llm_query)
            else:
                # 기본 LLM 등록
                register_llm('gpt2', model_name='gpt2', max_tokens=200)
                llm_response = ask_llm('gpt2', llm_query)
        except Exception as e:
            llm_response = f"[LLM 오류] {str(e)}"
        
        # LLM 응답 인코딩
        # 더미 벡터 (실제로는 LLM 응답을 임베딩해야 함)
        response_embedding = torch.randn_like(query_features)
        encoded_response = self.response_encoder(response_embedding)
        
        return {
            'llm_query': llm_query,
            'llm_response': llm_response,
            'encoded_response': encoded_response,
            'query_features': query_features
        }
    
    def _summarize_emotions(self, emotion_predictions: torch.Tensor) -> str:
        """감정 예측 요약"""
        emotions = ['valence', 'arousal', 'dominance', 'certainty', 'surprise', 'anticipation']
        values = emotion_predictions[0].detach().cpu().numpy()
        
        summary_parts = []
        for i, (emotion, value) in enumerate(zip(emotions, values)):
            if abs(value) > 0.3:  # 임계값 이상인 감정만
                summary_parts.append(f"{emotion}: {value:.2f}")
        
        return ", ".join(summary_parts) if summary_parts else "중성적 감정"
    
    def _summarize_semantics(self, semantic_predictions: torch.Tensor) -> str:
        """의미 예측 요약"""
        top_indices = torch.topk(semantic_predictions[0], 3).indices
        confidence_scores = torch.topk(semantic_predictions[0], 3).values
        
        summary_parts = []
        for idx, conf in zip(top_indices, confidence_scores):
            summary_parts.append(f"클래스{idx.item()}: {conf.item():.3f}")
        
        return ", ".join(summary_parts)

def create_mega_scale_model(target_params: int = 200_000_000) -> MegaScaleXAIModel:
    """타겟 파라미터 수에 맞는 메가 스케일 모델 생성"""
    
    # 기본 설정으로 시작
    config = MegaScaleConfig()
    
    # 파라미터 수 조정
    estimated_params = config.get_total_params()
    
    if estimated_params < target_params:
        # 파라미터 수가 부족하면 차원 증가
        scale_factor = (target_params / estimated_params) ** 0.5
        
        config.hidden_dim = int(config.hidden_dim * scale_factor)
        config.intermediate_size = int(config.intermediate_size * scale_factor)
        
        # 16의 배수로 조정 (효율성을 위해)
        config.hidden_dim = (config.hidden_dim // 16) * 16
        config.intermediate_size = (config.intermediate_size // 16) * 16
        
        # 어텐션 헤드 수도 조정 (hidden_dim의 약수가 되도록)
        head_candidates = [8, 16, 32, 64]
        for heads in head_candidates:
            if config.hidden_dim % heads == 0:
                config.num_attention_heads = heads
                break
        else:
            config.num_attention_heads = 8  # 기본값
    
    model = MegaScaleXAIModel(config)
    actual_params = model.get_parameter_count()
    
    print(f"✅ 메가 스케일 모델 생성 완료")
    print(f"   - 타겟 파라미터: {target_params:,}")
    print(f"   - 실제 파라미터: {actual_params:,}")
    print(f"   - 히든 차원: {config.hidden_dim}")
    print(f"   - 레이어 수: {config.num_layers}")
    print(f"   - 어텐션 헤드: {config.num_attention_heads}")
    
    return model

def optimize_model_for_inference(model: MegaScaleXAIModel) -> MegaScaleXAIModel:
    """추론을 위한 모델 최적화"""
    model.eval()
    
    # 그래디언트 체크포인팅 비활성화
    model.config.use_gradient_checkpointing = False
    
    # 드롭아웃 비활성화
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    
    print("✅ 모델 추론 최적화 완료")
    return model

if __name__ == "__main__":
    # 테스트 실행
    print("🚀 메가 스케일 XAI 모델 테스트")
    
    model = create_mega_scale_model(target_params=200_000_000)
    model = optimize_model_for_inference(model)
    
    # 더미 입력으로 테스트
    batch_size = 2
    seq_len = 128
    input_dim = 1024
    
    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    
    with torch.no_grad():
        outputs = model(dummy_input)
    
    print(f"✅ 추론 테스트 완료")
    print(f"   - 감정 출력: {outputs['emotion_predictions'].shape}")
    print(f"   - 의미 출력: {outputs['semantic_predictions'].shape}")
    print(f"   - 추론 출력: {outputs['reasoning_features'].shape}")
    print(f"   - 통합 출력: {outputs['integration_features'].shape}")
    
    if 'xai_explanation' in outputs:
        print(f"   - XAI 신뢰도: {outputs['xai_explanation']['confidence_score']:.3f}")