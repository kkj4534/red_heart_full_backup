"""
고급 메타 통합 시스템 - Linux 전용
Advanced Meta Integration System for Linux

다중 헤드 결과 통합 및 메타 학습 시스템:
- 40M 파라미터로 다른 헤드들의 출력 통합
- 메타 학습을 통한 동적 가중치 조정
- 앙상블 기법을 통한 결과 개선
- 크로스 어텐션 기반 헤드 간 상호작용
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
from collections import defaultdict
from enum import Enum

# 시스템 설정
from config import ADVANCED_CONFIG, DEVICE, TORCH_DTYPE, SYSTEM_CONFIG

# 로깅 설정
logger = logging.getLogger(__name__)

class IntegrationStrategy(Enum):
    """통합 전략"""
    WEIGHTED_AVERAGE = "weighted_average"      # 가중 평균
    ATTENTION_BASED = "attention_based"        # 어텐션 기반
    ENSEMBLE_VOTING = "ensemble_voting"        # 앙상블 투표
    META_LEARNED = "meta_learned"              # 메타 학습
    ADAPTIVE_FUSION = "adaptive_fusion"        # 적응적 융합

@dataclass
class MetaIntegrationResult:
    """메타 통합 결과"""
    integrated_output: torch.Tensor
    head_contributions: Dict[str, float]       # 각 헤드의 기여도
    integration_confidence: float              # 통합 신뢰도
    strategy_used: IntegrationStrategy         # 사용된 전략
    meta_insights: Dict[str, Any]              # 메타 인사이트


class MetaIntegrationNetwork(nn.Module):
    """메타 통합 신경망 (40M 파라미터)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        self.head_dim = 1280  # 백본 차원
        self.integration_dim = 512  # 통합 차원
        self.num_heads = 5  # 통합할 헤드 수
        
        # 헤드별 입력 프로젝션 (각 헤드의 출력을 통합 차원으로)
        self.head_projections = nn.ModuleDict({
            'emotion_empathy': nn.Linear(self.head_dim, self.integration_dim),
            'bentham_fromm': nn.Linear(self.head_dim, self.integration_dim),
            'semantic_surd': nn.Linear(self.head_dim, self.integration_dim),
            'regret_learning': nn.Linear(self.head_dim, self.integration_dim),
            'meta_integration': nn.Linear(self.head_dim, self.integration_dim)
        })
        
        # 크로스 어텐션 레이어 (헤드 간 상호작용)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.integration_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 통합 서킷 레이어 (config에서 지정된 대로)
        circuit_layers = config.get('circuit_integration_layers', [256, 128, 64])
        layers = []
        in_dim = self.integration_dim * self.num_heads  # 모든 헤드 concatenate
        
        for out_dim in circuit_layers:
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            in_dim = out_dim
        
        self.integration_circuit = nn.Sequential(*layers)
        
        # 최종 출력 레이어
        self.output_projection = nn.Linear(circuit_layers[-1], self.head_dim)
        
        # 메타 학습 가중치 (적응적 가중치)
        if config.get('adaptive_weights', True):
            self.head_weights = nn.Parameter(torch.ones(self.num_heads) / self.num_heads)
        else:
            self.register_buffer('head_weights', torch.ones(self.num_heads) / self.num_heads)
        
        # 전략별 특화 모듈
        self.strategy_modules = nn.ModuleDict({
            'attention_based': nn.Sequential(
                nn.Linear(self.integration_dim, 64),
                nn.ReLU(),
                nn.Linear(64, self.num_heads),
                nn.Softmax(dim=-1)
            ),
            'meta_learned': nn.LSTM(
                input_size=self.integration_dim,
                hidden_size=128,
                num_layers=2,
                batch_first=True,
                bidirectional=True
            )
        })
        
        self._init_weights()
    
    def _init_weights(self):
        """가중치 초기화"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, head_outputs: Dict[str, torch.Tensor], 
                strategy: IntegrationStrategy = IntegrationStrategy.META_LEARNED) -> MetaIntegrationResult:
        """전방향 전파"""
        
        batch_size = next(iter(head_outputs.values())).shape[0]
        device = next(iter(head_outputs.values())).device
        
        # 1. 헤드별 출력 프로젝션
        projected_outputs = {}
        for head_name, output in head_outputs.items():
            if head_name in self.head_projections:
                projected = self.head_projections[head_name](output)
                projected_outputs[head_name] = projected
        
        # 2. 크로스 어텐션을 통한 헤드 간 상호작용
        # 모든 프로젝션된 출력을 스택 (batch, num_heads, integration_dim)
        stacked_outputs = torch.stack(list(projected_outputs.values()), dim=1)
        
        # 크로스 어텐션 적용
        attended_outputs, attention_weights = self.cross_attention(
            stacked_outputs, stacked_outputs, stacked_outputs
        )
        
        # 3. 전략별 통합
        if strategy == IntegrationStrategy.WEIGHTED_AVERAGE:
            # 가중 평균
            weights = F.softmax(self.head_weights, dim=0)
            integrated = torch.sum(attended_outputs * weights.view(1, -1, 1), dim=1)
            
        elif strategy == IntegrationStrategy.ATTENTION_BASED:
            # 어텐션 기반 통합
            attention_module = self.strategy_modules['attention_based']
            # 평균 풀링으로 시퀀스 차원 제거
            pooled = attended_outputs.mean(dim=2)  # (batch, num_heads)
            dynamic_weights = attention_module(pooled)  # (batch, num_heads)
            integrated = torch.sum(
                attended_outputs * dynamic_weights.unsqueeze(-1), 
                dim=1
            )
            
        elif strategy == IntegrationStrategy.META_LEARNED:
            # LSTM 기반 메타 학습 통합
            lstm_module = self.strategy_modules['meta_learned']
            lstm_out, _ = lstm_module(attended_outputs)
            # 양방향 LSTM 출력 결합
            integrated = lstm_out.mean(dim=1)  # 시퀀스 차원 평균
            
        else:
            # 기본: 단순 평균
            integrated = attended_outputs.mean(dim=1)
        
        # 4. 통합 서킷 통과
        # 모든 attended outputs을 concatenate
        concat_features = attended_outputs.reshape(batch_size, -1)
        circuit_output = self.integration_circuit(concat_features)
        
        # 5. 최종 출력 생성
        final_output = self.output_projection(circuit_output)
        
        # 6. 헤드 기여도 계산
        head_contributions = {}
        if hasattr(attention_weights, 'mean'):
            # 어텐션 가중치에서 기여도 추출
            avg_attention = attention_weights.mean(dim=0).mean(dim=0)  # 평균 어텐션
            for i, head_name in enumerate(projected_outputs.keys()):
                head_contributions[head_name] = float(avg_attention[i].item())
        else:
            # 균등 기여도
            for head_name in projected_outputs.keys():
                head_contributions[head_name] = 1.0 / len(projected_outputs)
        
        # 7. 통합 신뢰도 계산
        # 헤드 출력 간 일관성을 기반으로
        output_std = torch.std(stacked_outputs, dim=1).mean()
        integration_confidence = 1.0 / (1.0 + output_std.item())
        
        # 8. 메타 인사이트 생성
        meta_insights = {
            'head_agreement': 1.0 - output_std.item(),
            'dominant_head': max(head_contributions, key=head_contributions.get),
            'integration_variance': float(torch.var(integrated).item()),
            'strategy_effectiveness': integration_confidence
        }
        
        return MetaIntegrationResult(
            integrated_output=final_output,
            head_contributions=head_contributions,
            integration_confidence=integration_confidence,
            strategy_used=strategy,
            meta_insights=meta_insights
        )


class AdvancedMetaIntegrationSystem:
    """고급 메타 통합 시스템"""
    
    def __init__(self):
        """시스템 초기화"""
        self.config = SYSTEM_CONFIG.get('meta_integration', {})
        self.device = torch.device(DEVICE)
        self.logger = logger
        
        # 메타 통합 네트워크 초기화
        self.integration_network = MetaIntegrationNetwork(self.config).to(self.device)
        self.integration_network.eval()  # 기본적으로 평가 모드
        
        # 성능 추적
        self.performance_history = defaultdict(list)
        self.strategy_effectiveness = defaultdict(float)
        
        # 앙상블 설정
        self.ensemble_enabled = self.config.get('ensemble_methods', True)
        self.ensemble_strategies = [
            IntegrationStrategy.WEIGHTED_AVERAGE,
            IntegrationStrategy.ATTENTION_BASED,
            IntegrationStrategy.META_LEARNED
        ]
        
        logger.info("고급 메타 통합 시스템 초기화 완료")
        logger.info(f"통합 전략: {self.config}")
        logger.info(f"디바이스: {self.device}")
    
    def get_pytorch_network(self) -> nn.Module:
        """PyTorch 네트워크 반환 (HeadAdapter와의 호환성)"""
        return self.integration_network
    
    async def integrate_head_outputs(self, 
                                    head_outputs: Dict[str, torch.Tensor],
                                    context: Optional[Dict[str, Any]] = None) -> MetaIntegrationResult:
        """헤드 출력 통합"""
        
        try:
            # 입력 검증
            if not head_outputs:
                raise ValueError("헤드 출력이 비어있습니다")
            
            # 컨텍스트 기반 전략 선택
            strategy = self._select_strategy(head_outputs, context)
            
            # 앙상블 통합
            if self.ensemble_enabled and len(head_outputs) >= 3:
                results = []
                
                # 여러 전략으로 통합 수행
                for strat in self.ensemble_strategies:
                    with torch.no_grad():
                        result = self.integration_network(head_outputs, strat)
                        results.append(result)
                
                # 앙상블 결과 결합
                final_result = self._ensemble_results(results)
                
            else:
                # 단일 전략 사용
                with torch.no_grad():
                    final_result = self.integration_network(head_outputs, strategy)
            
            # 성능 추적
            self._track_performance(final_result, strategy)
            
            return final_result
            
        except Exception as e:
            logger.error(f"메타 통합 실패: {e}")
            # 폴백: 단순 평균
            avg_output = torch.stack(list(head_outputs.values())).mean(dim=0)
            return MetaIntegrationResult(
                integrated_output=avg_output,
                head_contributions={k: 1.0/len(head_outputs) for k in head_outputs},
                integration_confidence=0.5,
                strategy_used=IntegrationStrategy.WEIGHTED_AVERAGE,
                meta_insights={'error': str(e), 'fallback': True}
            )
    
    def _select_strategy(self, head_outputs: Dict[str, torch.Tensor], 
                        context: Optional[Dict[str, Any]]) -> IntegrationStrategy:
        """컨텍스트 기반 전략 선택"""
        
        if context is None:
            # 기본: 메타 학습 전략
            return IntegrationStrategy.META_LEARNED
        
        # 헤드 출력 분산이 크면 어텐션 기반
        output_variance = torch.var(torch.stack(list(head_outputs.values())))
        if output_variance > 0.5:
            return IntegrationStrategy.ATTENTION_BASED
        
        # 빠른 응답이 필요하면 가중 평균
        if context.get('time_critical', False):
            return IntegrationStrategy.WEIGHTED_AVERAGE
        
        # 높은 정확도가 필요하면 메타 학습
        if context.get('high_accuracy', False):
            return IntegrationStrategy.META_LEARNED
        
        # 전략 효과성 기반 선택
        best_strategy = max(self.strategy_effectiveness.items(), 
                           key=lambda x: x[1], 
                           default=(IntegrationStrategy.META_LEARNED, 0))[0]
        
        return best_strategy
    
    def _ensemble_results(self, results: List[MetaIntegrationResult]) -> MetaIntegrationResult:
        """앙상블 결과 결합"""
        
        # 출력 평균
        outputs = torch.stack([r.integrated_output for r in results])
        ensemble_output = outputs.mean(dim=0)
        
        # 신뢰도 가중 평균
        confidences = torch.tensor([r.integration_confidence for r in results])
        weights = F.softmax(confidences, dim=0)
        
        weighted_output = torch.sum(outputs * weights.view(-1, 1, 1), dim=0)
        
        # 헤드 기여도 평균
        avg_contributions = defaultdict(float)
        for result in results:
            for head, contrib in result.head_contributions.items():
                avg_contributions[head] += contrib / len(results)
        
        # 통합 신뢰도는 최대값
        max_confidence = max(r.integration_confidence for r in results)
        
        # 메타 인사이트 병합
        merged_insights = {
            'ensemble_size': len(results),
            'confidence_variance': float(torch.var(confidences).item()),
            'strategies_used': [r.strategy_used.value for r in results]
        }
        
        return MetaIntegrationResult(
            integrated_output=weighted_output,
            head_contributions=dict(avg_contributions),
            integration_confidence=max_confidence,
            strategy_used=IntegrationStrategy.ENSEMBLE_VOTING,
            meta_insights=merged_insights
        )
    
    def _track_performance(self, result: MetaIntegrationResult, strategy: IntegrationStrategy):
        """성능 추적"""
        
        # 전략별 효과성 업데이트
        self.strategy_effectiveness[strategy] = (
            0.9 * self.strategy_effectiveness.get(strategy, 0.5) + 
            0.1 * result.integration_confidence
        )
        
        # 성능 이력 저장
        self.performance_history['confidence'].append(result.integration_confidence)
        self.performance_history['strategy'].append(strategy.value)
        self.performance_history['timestamp'].append(datetime.now())
        
        # 이력 크기 제한
        max_history = 1000
        for key in self.performance_history:
            if len(self.performance_history[key]) > max_history:
                self.performance_history[key] = self.performance_history[key][-max_history:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 반환"""
        
        if not self.performance_history['confidence']:
            return {'status': 'no_data'}
        
        recent_confidences = self.performance_history['confidence'][-100:]
        
        return {
            'average_confidence': np.mean(recent_confidences),
            'confidence_trend': np.polyfit(range(len(recent_confidences)), recent_confidences, 1)[0],
            'strategy_effectiveness': dict(self.strategy_effectiveness),
            'total_integrations': len(self.performance_history['confidence']),
            'most_used_strategy': max(set(self.performance_history['strategy'][-100:]), 
                                     key=self.performance_history['strategy'][-100:].count)
        }
    
    def set_training_mode(self, mode: bool = True):
        """학습 모드 설정"""
        if mode:
            self.integration_network.train()
        else:
            self.integration_network.eval()
    
    async def optimize_integration(self, feedback: Dict[str, Any]):
        """통합 최적화 (피드백 기반)"""
        
        # 피드백 기반 가중치 조정
        if hasattr(self.integration_network, 'head_weights') and self.integration_network.head_weights.requires_grad:
            
            # 긍정적 피드백이면 현재 가중치 강화
            if feedback.get('success', False):
                with torch.no_grad():
                    current_weights = self.integration_network.head_weights
                    contribution_scores = feedback.get('head_contributions', {})
                    
                    # 기여도 기반 가중치 조정
                    for i, (head_name, score) in enumerate(contribution_scores.items()):
                        if i < len(current_weights):
                            current_weights[i] = current_weights[i] * 0.9 + score * 0.1
                    
                    # 정규화
                    self.integration_network.head_weights.data = F.softmax(current_weights, dim=0)
        
        logger.info(f"통합 최적화 완료: {feedback}")
    
    def save_state(self, path: str):
        """상태 저장"""
        state = {
            'network_state': self.integration_network.state_dict(),
            'strategy_effectiveness': dict(self.strategy_effectiveness),
            'performance_history': dict(self.performance_history),
            'config': self.config
        }
        torch.save(state, path)
        logger.info(f"메타 통합 시스템 상태 저장: {path}")
    
    def load_state(self, path: str):
        """상태 로드"""
        if os.path.exists(path):
            state = torch.load(path, map_location=self.device)
            self.integration_network.load_state_dict(state['network_state'])
            self.strategy_effectiveness = defaultdict(float, state['strategy_effectiveness'])
            self.performance_history = defaultdict(list, state['performance_history'])
            logger.info(f"메타 통합 시스템 상태 로드: {path}")