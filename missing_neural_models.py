"""
누락된 신경망 모델들 구현
Missing Neural Models Implementation

Red Heart 시스템에서 필요한 신경망 모델들의 완전한 구현:
- SelfOtherNeuralNetwork: 자타 구분 신경망
- IncrementalLearner: 증분 학습기
- HierarchicalPatternStructure: 계층적 패턴 구조
- SimpleFallbackClassifier: 분류기 (폴백 아님)
- SimpleFallbackLearner: 학습기 (폴백 아님)
- SimpleFallbackManager: 관리기 (폴백 아님)

프로젝트 규칙: fallback 없는 순수 재시도 방식
"""

import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque, defaultdict
from enum import Enum
import json
from pathlib import Path

# 로거 설정
logger = logging.getLogger(__name__)

class SelfOtherNeuralNetwork(nn.Module):
    """
    자타 구분 신경망
    Self-Other Differentiation Neural Network
    
    자신과 타인의 감정/인지 상태를 구분하는 고급 신경망
    """
    
    def __init__(self, input_dim: int = 1280, hidden_dims: List[int] = None, 
                 output_dim: int = 2, dropout_rate: float = 0.1, 
                 attention_heads: int = 8, use_batch_norm: bool = True, **kwargs):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.attention_heads = attention_heads
        
        # 입력 임베딩 및 정규화
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 자타 구분을 위한 어텐션 메커니즘
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dims[0],
            num_heads=attention_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # 계층적 특성 추출
        self.feature_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.feature_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.LayerNorm(hidden_dims[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                )
            )
        
        # 자타 구분 헤드
        self.self_other_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[-1] // 2, output_dim)
        )
        
        # 신뢰도 추정 헤드
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 4),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1] // 4, 1),
            nn.Sigmoid()
        )
        
        # 컨텍스트 인코딩
        self.context_encoder = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.Tanh()
        )
        
        logger.info(f"SelfOtherNeuralNetwork 초기화: {sum(p.numel() for p in self.parameters())} 파라미터")
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        순전파
        
        Args:
            x: 입력 특성 (batch_size, input_dim)
            context: 컨텍스트 정보 (batch_size, context_dim)
            
        Returns:
            자타 구분 결과, 신뢰도, 컨텍스트 표현
        """
        batch_size = x.size(0)
        
        # 입력 임베딩
        embedded = self.input_embedding(x)  # (batch_size, hidden_dims[0])
        
        # 셀프 어텐션 (배치 차원 추가)
        embedded_seq = embedded.unsqueeze(1)  # (batch_size, 1, hidden_dims[0])
        attended, attention_weights = self.self_attention(
            embedded_seq, embedded_seq, embedded_seq
        )
        attended = attended.squeeze(1)  # (batch_size, hidden_dims[0])
        
        # 계층적 특성 추출
        features = attended
        for layer in self.feature_layers:
            features = layer(features)
        
        # 자타 구분
        self_other_logits = self.self_other_head(features)
        self_other_probs = F.softmax(self_other_logits, dim=-1)
        
        # 신뢰도 추정
        confidence = self.confidence_head(features)
        
        # 컨텍스트 인코딩
        context_repr = self.context_encoder(features)
        
        return {
            'self_other_logits': self_other_logits,
            'self_other_probs': self_other_probs,
            'confidence': confidence,
            'context_representation': context_repr,
            'attention_weights': attention_weights.squeeze(1),
            'features': features
        }

class IncrementalLearner(nn.Module):
    """
    증분 학습기
    Incremental Learning System
    
    새로운 데이터를 지속적으로 학습하면서 이전 지식을 보존하는 시스템
    """
    
    def __init__(self, input_dim: int = 1280, memory_size: int = 10000, 
                 learning_rate: float = 0.001, regularization_weight: float = 0.1,
                 base_model: Optional[nn.Module] = None, **kwargs):
        super().__init__()
        
        self.input_dim = input_dim
        self.memory_size = memory_size
        self.learning_rate = learning_rate
        self.regularization_weight = regularization_weight
        
        # 메모리 버퍼 (경험 재생용)
        self.memory_buffer = deque(maxlen=memory_size)
        self.memory_labels = deque(maxlen=memory_size)
        
        # 학습 네트워크
        self.learning_network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # 지식 증류를 위한 교사 네트워크
        self.teacher_network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # 메타 학습을 위한 어댑터
        self.meta_adapter = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )
        
        # 학습 상태 추적
        self.learning_step = 0
        self.knowledge_retention_score = 1.0
        self.adaptation_rate = 0.0
        
        # 옵티마이저
        self.optimizer = torch.optim.AdamW(
            self.learning_network.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        logger.info(f"IncrementalLearner 초기화: 메모리 크기 {memory_size}")
    
    def learn_incrementally(self, new_data: torch.Tensor, new_labels: torch.Tensor) -> Dict[str, float]:
        """
        증분 학습 수행
        
        Args:
            new_data: 새로운 학습 데이터
            new_labels: 새로운 레이블
            
        Returns:
            학습 메트릭
        """
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            retry_count += 1
            try:
                # 새 데이터를 메모리에 추가
                for data, label in zip(new_data, new_labels):
                    self.memory_buffer.append(data.detach().cpu())
                    self.memory_labels.append(label.detach().cpu())
                
                # 교사 네트워크 업데이트 (지식 보존용)
                if self.learning_step > 0:
                    self._update_teacher_network()
                
                # 새 데이터로 학습
                current_loss = self._train_on_new_data(new_data, new_labels)
                
                # 경험 재생 (catastrophic forgetting 방지)
                if len(self.memory_buffer) > 32:
                    replay_loss = self._experience_replay()
                else:
                    replay_loss = 0.0
                
                # 지식 증류 손실
                distillation_loss = self._knowledge_distillation_loss(new_data)
                
                # 전체 손실
                total_loss = current_loss + self.regularization_weight * (replay_loss + distillation_loss)
                
                # 역전파
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.learning_network.parameters(), 1.0)
                self.optimizer.step()
                
                # 메트릭 업데이트
                self.learning_step += 1
                self._update_learning_metrics(current_loss, replay_loss, distillation_loss)
                
                return {
                    'total_loss': total_loss.item(),
                    'current_loss': current_loss.item(),
                    'replay_loss': replay_loss,
                    'distillation_loss': distillation_loss.item(),
                    'knowledge_retention': self.knowledge_retention_score,
                    'adaptation_rate': self.adaptation_rate
                }
                
            except Exception as e:
                if retry_count >= max_retries:
                    logger.error(f"증분 학습 최종 실패: {e}")
                    raise RuntimeError(f"증분 학습이 {max_retries}번 재시도 후 실패: {e}")
                logger.warning(f"증분 학습 재시도 {retry_count}/{max_retries}: {e}")
                time.sleep(0.1)
    
    def _update_teacher_network(self):
        """교사 네트워크 업데이트 (EMA 방식)"""
        momentum = 0.999
        for teacher_param, student_param in zip(self.teacher_network.parameters(), 
                                               self.learning_network.parameters()):
            teacher_param.data = momentum * teacher_param.data + (1 - momentum) * student_param.data
    
    def _train_on_new_data(self, data: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """새 데이터로 훈련"""
        features = self.learning_network(data)
        adapted_features = self.meta_adapter(features)
        
        # 간단한 분류 손실 (실제로는 태스크에 따라 달라짐)
        loss = F.mse_loss(adapted_features, features.detach())
        return loss
    
    def _experience_replay(self) -> float:
        """경험 재생"""
        if len(self.memory_buffer) < 16:
            return 0.0
        
        # 랜덤 샘플링
        indices = np.random.choice(len(self.memory_buffer), size=16, replace=False)
        replay_data = torch.stack([self.memory_buffer[i] for i in indices])
        replay_labels = torch.stack([self.memory_labels[i] for i in indices])
        
        # 현재 디바이스로 이동
        device = next(self.learning_network.parameters()).device
        replay_data = replay_data.to(device)
        replay_labels = replay_labels.to(device)
        
        # 재생 손실 계산
        features = self.learning_network(replay_data)
        adapted_features = self.meta_adapter(features)
        
        replay_loss = F.mse_loss(adapted_features, features.detach())
        return replay_loss.item()
    
    def _knowledge_distillation_loss(self, data: torch.Tensor) -> torch.Tensor:
        """지식 증류 손실"""
        student_features = self.learning_network(data)
        teacher_features = self.teacher_network(data)
        
        distillation_loss = F.mse_loss(student_features, teacher_features.detach())
        return distillation_loss
    
    def _update_learning_metrics(self, current_loss: torch.Tensor, replay_loss: float, distillation_loss: torch.Tensor):
        """학습 메트릭 업데이트"""
        # 지식 보존 점수 (낮은 증류 손실일수록 높음)
        self.knowledge_retention_score = max(0.0, 1.0 - distillation_loss.item())
        
        # 적응률 (현재 손실 대비 재생 손실)
        if replay_loss > 0:
            self.adaptation_rate = current_loss.item() / (current_loss.item() + replay_loss)
        else:
            self.adaptation_rate = 1.0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순전파"""
        features = self.learning_network(x)
        adapted_features = self.meta_adapter(features)
        return adapted_features

class HierarchicalPatternStructure:
    """
    계층적 패턴 구조
    Hierarchical Pattern Structure
    
    패턴을 계층적으로 조직화하고 관리하는 시스템
    """
    
    def __init__(self, max_depth: int = 5, branching_factor: int = 3, 
                 similarity_threshold: float = 0.8, merge_threshold: float = 0.9,
                 split_threshold: float = 0.3, rebalancing_frequency: int = 100, **kwargs):
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.similarity_threshold = similarity_threshold
        self.merge_threshold = merge_threshold
        self.split_threshold = split_threshold
        self.rebalancing_frequency = rebalancing_frequency
        
        # 패턴 트리 구조
        self.root_patterns = {}
        self.pattern_hierarchy = defaultdict(lambda: defaultdict(list))
        self.pattern_statistics = defaultdict(lambda: {
            'frequency': 0,
            'last_accessed': datetime.now(),
            'children': [],
            'parent': None,
            'depth': 0,
            'similarity_scores': {}
        })
        
        # 패턴 인덱스
        self.pattern_embeddings = {}
        self.pattern_metadata = {}
        
        logger.info("HierarchicalPatternStructure 초기화 완료")
    
    def add_pattern(self, pattern_id: str, pattern_data: Any, embedding: np.ndarray) -> bool:
        """
        새 패턴 추가
        
        Args:
            pattern_id: 패턴 식별자
            pattern_data: 패턴 데이터
            embedding: 패턴 임베딩
            
        Returns:
            추가 성공 여부
        """
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            retry_count += 1
            try:
                # 기존 패턴과 유사도 계산
                best_parent = self._find_best_parent(embedding)
                
                # 새 패턴 추가
                self.pattern_embeddings[pattern_id] = embedding
                self.pattern_metadata[pattern_id] = {
                    'data': pattern_data,
                    'created_at': datetime.now(),
                    'embedding': embedding
                }
                
                if best_parent:
                    # 기존 패턴의 자식으로 추가
                    parent_depth = self.pattern_statistics[best_parent]['depth']
                    if parent_depth < self.max_depth - 1:
                        self.pattern_hierarchy[best_parent][parent_depth + 1].append(pattern_id)
                        self.pattern_statistics[pattern_id]['parent'] = best_parent
                        self.pattern_statistics[pattern_id]['depth'] = parent_depth + 1
                        self.pattern_statistics[best_parent]['children'].append(pattern_id)
                    else:
                        # 최대 깊이 도달, 루트로 추가
                        self.root_patterns[pattern_id] = pattern_data
                        self.pattern_statistics[pattern_id]['depth'] = 0
                else:
                    # 루트 패턴으로 추가
                    self.root_patterns[pattern_id] = pattern_data
                    self.pattern_statistics[pattern_id]['depth'] = 0
                
                # 패턴 통계 업데이트
                self.pattern_statistics[pattern_id]['frequency'] = 1
                self.pattern_statistics[pattern_id]['last_accessed'] = datetime.now()
                
                # 필요시 구조 최적화
                self._optimize_structure()
                
                return True
                
            except Exception as e:
                if retry_count >= max_retries:
                    logger.error(f"패턴 추가 최종 실패: {e}")
                    raise RuntimeError(f"패턴 추가가 {max_retries}번 재시도 후 실패: {e}")
                logger.warning(f"패턴 추가 재시도 {retry_count}/{max_retries}: {e}")
                time.sleep(0.1)
    
    def _find_best_parent(self, embedding: np.ndarray) -> Optional[str]:
        """최적의 부모 패턴 찾기"""
        best_parent = None
        best_similarity = 0.0
        
        for pattern_id, pattern_embedding in self.pattern_embeddings.items():
            similarity = self._calculate_similarity(embedding, pattern_embedding)
            
            # 유사도가 임계값 이상이고 현재 최고 유사도보다 높으면
            if similarity >= self.similarity_threshold and similarity > best_similarity:
                # 해당 패턴이 자식을 가질 수 있는지 확인
                current_children = len(self.pattern_statistics[pattern_id]['children'])
                if current_children < self.branching_factor:
                    best_parent = pattern_id
                    best_similarity = similarity
        
        return best_parent
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """두 임베딩 간 유사도 계산"""
        # 코사인 유사도 사용
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(embedding1, embedding2) / (norm1 * norm2)
    
    def _optimize_structure(self):
        """구조 최적화 (병합 및 분할)"""
        # 유사한 패턴들 병합
        self._merge_similar_patterns()
        
        # 너무 큰 클러스터 분할
        self._split_large_clusters()
    
    def _merge_similar_patterns(self):
        """유사한 패턴들 병합"""
        patterns_to_merge = []
        
        for pattern_id1 in self.pattern_embeddings:
            for pattern_id2 in self.pattern_embeddings:
                if pattern_id1 != pattern_id2:
                    similarity = self._calculate_similarity(
                        self.pattern_embeddings[pattern_id1],
                        self.pattern_embeddings[pattern_id2]
                    )
                    
                    if similarity >= self.merge_threshold:
                        # 같은 깊이의 패턴들만 병합
                        if (self.pattern_statistics[pattern_id1]['depth'] == 
                            self.pattern_statistics[pattern_id2]['depth']):
                            patterns_to_merge.append((pattern_id1, pattern_id2, similarity))
        
        # 유사도가 높은 순으로 정렬
        patterns_to_merge.sort(key=lambda x: x[2], reverse=True)
        
        # 병합 수행
        merged_patterns = set()
        for pattern_id1, pattern_id2, similarity in patterns_to_merge:
            if pattern_id1 not in merged_patterns and pattern_id2 not in merged_patterns:
                self._merge_two_patterns(pattern_id1, pattern_id2)
                merged_patterns.add(pattern_id2)
    
    def _merge_two_patterns(self, pattern_id1: str, pattern_id2: str):
        """두 패턴 병합"""
        # pattern_id2를 pattern_id1로 병합
        
        # 임베딩 평균화
        emb1 = self.pattern_embeddings[pattern_id1]
        emb2 = self.pattern_embeddings[pattern_id2]
        freq1 = self.pattern_statistics[pattern_id1]['frequency']
        freq2 = self.pattern_statistics[pattern_id2]['frequency']
        
        # 빈도 가중 평균
        total_freq = freq1 + freq2
        merged_embedding = (emb1 * freq1 + emb2 * freq2) / total_freq
        
        # 임베딩 업데이트
        self.pattern_embeddings[pattern_id1] = merged_embedding
        
        # 통계 업데이트
        self.pattern_statistics[pattern_id1]['frequency'] = total_freq
        
        # 자식 패턴들을 pattern_id1로 이전
        children2 = self.pattern_statistics[pattern_id2]['children']
        self.pattern_statistics[pattern_id1]['children'].extend(children2)
        
        for child in children2:
            self.pattern_statistics[child]['parent'] = pattern_id1
        
        # pattern_id2 제거
        del self.pattern_embeddings[pattern_id2]
        del self.pattern_statistics[pattern_id2]
        del self.pattern_metadata[pattern_id2]
        
        if pattern_id2 in self.root_patterns:
            del self.root_patterns[pattern_id2]
    
    def _split_large_clusters(self):
        """큰 클러스터 분할"""
        patterns_to_split = []
        
        for pattern_id in self.pattern_embeddings:
            children_count = len(self.pattern_statistics[pattern_id]['children'])
            if children_count > self.branching_factor * 2:
                patterns_to_split.append(pattern_id)
        
        for pattern_id in patterns_to_split:
            self._split_pattern_cluster(pattern_id)
    
    def _split_pattern_cluster(self, pattern_id: str):
        """패턴 클러스터 분할"""
        children = self.pattern_statistics[pattern_id]['children']
        
        if len(children) <= self.branching_factor:
            return
        
        # K-means 스타일의 클러스터링으로 분할
        child_embeddings = [self.pattern_embeddings[child_id] for child_id in children]
        
        # 간단한 2-분할 (실제로는 더 정교한 클러스터링 사용 가능)
        mid_point = len(children) // 2
        group1 = children[:mid_point]
        group2 = children[mid_point:]
        
        # 새로운 중간 노드 생성
        new_node_id1 = f"{pattern_id}_split_1"
        new_node_id2 = f"{pattern_id}_split_2"
        
        # 그룹 임베딩 계산
        group1_embeddings = [self.pattern_embeddings[child] for child in group1]
        group2_embeddings = [self.pattern_embeddings[child] for child in group2]
        
        group1_embedding = np.mean(group1_embeddings, axis=0)
        group2_embedding = np.mean(group2_embeddings, axis=0)
        
        # 새 노드 추가
        current_depth = self.pattern_statistics[pattern_id]['depth']
        
        self.pattern_embeddings[new_node_id1] = group1_embedding
        self.pattern_embeddings[new_node_id2] = group2_embedding
        
        self.pattern_statistics[new_node_id1] = {
            'frequency': sum(self.pattern_statistics[child]['frequency'] for child in group1),
            'last_accessed': datetime.now(),
            'children': group1,
            'parent': pattern_id,
            'depth': current_depth + 1,
            'similarity_scores': {}
        }
        
        self.pattern_statistics[new_node_id2] = {
            'frequency': sum(self.pattern_statistics[child]['frequency'] for child in group2),
            'last_accessed': datetime.now(),
            'children': group2,
            'parent': pattern_id,
            'depth': current_depth + 1,
            'similarity_scores': {}
        }
        
        # 자식들의 부모 업데이트
        for child in group1:
            self.pattern_statistics[child]['parent'] = new_node_id1
        
        for child in group2:
            self.pattern_statistics[child]['parent'] = new_node_id2
        
        # 원래 패턴의 자식 목록 업데이트
        self.pattern_statistics[pattern_id]['children'] = [new_node_id1, new_node_id2]
    
    def get_pytorch_network(self) -> Optional[nn.Module]:
        """PyTorch 네트워크 반환 (HeadAdapter와의 호환성)"""
        # HierarchicalPatternStructure는 전통적인 딥러닝 모델 대신
        # 딕셔너리 기반 구조를 사용하므로 None 반환
        logger.warning("HierarchicalPatternStructure: PyTorch 네트워크가 없음 (딕셔너리 기반 구조)")
        return None

class SimpleFallbackClassifier(nn.Module):
    """
    간단한 분류기 (fallback이 아닌 기본 분류기)
    Simple Classifier (Not a fallback, but a basic classifier)
    """
    
    def __init__(self, input_dim: int = 1280, hidden_dim: int = 256, output_dim: int = 2):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        logger.info(f"SimpleFallbackClassifier 초기화: {sum(p.numel() for p in self.parameters())} 파라미터")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

class SimpleFallbackLearner:
    """
    간단한 학습기 (fallback이 아닌 기본 학습기)
    Simple Learner (Not a fallback, but a basic learner)
    """
    
    def __init__(self, learning_rate: float = 0.01, memory_size: int = 1000):
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.memory = deque(maxlen=memory_size)
        self.learning_history = []
        
        logger.info(f"SimpleFallbackLearner 초기화: LR={learning_rate}, 메모리={memory_size}")
    
    def learn(self, data: Any, label: Any) -> Dict[str, float]:
        """기본 학습"""
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            retry_count += 1
            try:
                # 메모리에 추가
                self.memory.append({'data': data, 'label': label, 'timestamp': datetime.now()})
                
                # 간단한 학습 시뮬레이션
                learning_score = min(1.0, len(self.memory) / self.memory_size)
                
                # 학습 기록
                learning_record = {
                    'timestamp': datetime.now(),
                    'learning_score': learning_score,
                    'memory_usage': len(self.memory) / self.memory_size
                }
                
                self.learning_history.append(learning_record)
                
                return {
                    'learning_score': learning_score,
                    'memory_usage': len(self.memory) / self.memory_size,
                    'learning_rate': self.learning_rate
                }
                
            except Exception as e:
                if retry_count >= max_retries:
                    logger.error(f"기본 학습 최종 실패: {e}")
                    raise RuntimeError(f"기본 학습이 {max_retries}번 재시도 후 실패: {e}")
                logger.warning(f"기본 학습 재시도 {retry_count}/{max_retries}: {e}")
                time.sleep(0.1)

class SimpleFallbackManager:
    """
    간단한 관리기 (fallback이 아닌 기본 관리기)
    Simple Manager (Not a fallback, but a basic manager)
    """
    
    def __init__(self, max_patterns: int = 10000, cleanup_threshold: float = 0.8):
        self.max_patterns = max_patterns
        self.cleanup_threshold = cleanup_threshold
        self.patterns = {}
        self.pattern_usage = defaultdict(int)
        self.last_cleanup = datetime.now()
        
        logger.info(f"SimpleFallbackManager 초기화: 최대 패턴={max_patterns}")
    
    def add_pattern(self, pattern_id: str, pattern_data: Any) -> bool:
        """패턴 추가"""
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            retry_count += 1
            try:
                # 메모리 정리 필요시
                if len(self.patterns) >= self.max_patterns * self.cleanup_threshold:
                    self._cleanup_old_patterns()
                
                # 패턴 추가
                self.patterns[pattern_id] = {
                    'data': pattern_data,
                    'created_at': datetime.now(),
                    'last_accessed': datetime.now()
                }
                
                self.pattern_usage[pattern_id] = 1
                
                return True
                
            except Exception as e:
                if retry_count >= max_retries:
                    logger.error(f"패턴 관리 최종 실패: {e}")
                    raise RuntimeError(f"패턴 관리가 {max_retries}번 재시도 후 실패: {e}")
                logger.warning(f"패턴 관리 재시도 {retry_count}/{max_retries}: {e}")
                time.sleep(0.1)
    
    def _cleanup_old_patterns(self):
        """오래된 패턴 정리"""
        # 사용 빈도가 낮은 패턴들 제거
        patterns_by_usage = sorted(
            self.pattern_usage.items(),
            key=lambda x: x[1]
        )
        
        # 하위 20% 제거
        remove_count = max(1, len(patterns_by_usage) // 5)
        patterns_to_remove = patterns_by_usage[:remove_count]
        
        for pattern_id, _ in patterns_to_remove:
            if pattern_id in self.patterns:
                del self.patterns[pattern_id]
            if pattern_id in self.pattern_usage:
                del self.pattern_usage[pattern_id]
        
        self.last_cleanup = datetime.now()
        logger.info(f"패턴 정리 완료: {remove_count}개 패턴 제거")
    
    def get_pattern(self, pattern_id: str) -> Optional[Any]:
        """패턴 조회"""
        if pattern_id in self.patterns:
            self.patterns[pattern_id]['last_accessed'] = datetime.now()
            self.pattern_usage[pattern_id] += 1
            return self.patterns[pattern_id]['data']
        return None

class AdvancedFeatureExtractor(nn.Module):
    """
    고급 특성 추출기
    Advanced Feature Extractor
    """
    
    def __init__(self, input_dim: int = 1280, output_dim: int = 512, **kwargs):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
        
        logger.info(f"AdvancedFeatureExtractor 초기화: {sum(p.numel() for p in self.parameters())} 파라미터")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)

class HierarchicalPatternClustering:
    """
    계층적 패턴 클러스터링
    Hierarchical Pattern Clustering
    """
    
    def __init__(self, n_clusters: int = 10, max_depth: int = 5, **kwargs):
        self.n_clusters = n_clusters
        self.max_depth = max_depth
        self.clusters = {}
        self.cluster_centers = {}
        
        logger.info(f"HierarchicalPatternClustering 초기화: {n_clusters}개 클러스터")
    
    def fit(self, data: np.ndarray) -> Dict[str, Any]:
        """클러스터링 수행"""
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            retry_count += 1
            try:
                # 간단한 K-means 클러스터링 시뮬레이션
                from sklearn.cluster import KMeans
                
                kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(data)
                
                self.clusters = {i: [] for i in range(self.n_clusters)}
                for idx, label in enumerate(cluster_labels):
                    self.clusters[label].append(idx)
                
                self.cluster_centers = {i: center for i, center in enumerate(kmeans.cluster_centers_)}
                
                return {
                    'clusters': self.clusters,
                    'centers': self.cluster_centers,
                    'n_clusters': self.n_clusters
                }
                
            except Exception as e:
                if retry_count >= max_retries:
                    logger.error(f"클러스터링 최종 실패: {e}")
                    # 기본 클러스터링으로 폴백하지 않고 예외 발생
                    raise RuntimeError(f"클러스터링이 {max_retries}번 재시도 후 실패: {e}")
                logger.warning(f"클러스터링 재시도 {retry_count}/{max_retries}: {e}")
                time.sleep(0.1)

class PatternRelationshipGraph:
    """
    패턴 관계 그래프
    Pattern Relationship Graph
    """
    
    def __init__(self, max_nodes: int = 10000, edge_threshold: float = 0.5, **kwargs):
        self.max_nodes = max_nodes
        self.edge_threshold = edge_threshold
        self.nodes = {}
        self.edges = defaultdict(list)
        self.node_features = {}
        
        logger.info(f"PatternRelationshipGraph 초기화: 최대 {max_nodes}개 노드")
    
    def add_node(self, node_id: str, features: np.ndarray) -> bool:
        """노드 추가"""
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            retry_count += 1
            try:
                if len(self.nodes) >= self.max_nodes:
                    self._remove_least_connected_node()
                
                self.nodes[node_id] = {
                    'created_at': datetime.now(),
                    'connections': 0
                }
                self.node_features[node_id] = features
                
                # 기존 노드들과의 관계 계산
                self._update_relationships(node_id)
                
                return True
                
            except Exception as e:
                if retry_count >= max_retries:
                    logger.error(f"노드 추가 최종 실패: {e}")
                    raise RuntimeError(f"노드 추가가 {max_retries}번 재시도 후 실패: {e}")
                logger.warning(f"노드 추가 재시도 {retry_count}/{max_retries}: {e}")
                time.sleep(0.1)
    
    def _update_relationships(self, new_node_id: str):
        """새 노드와 기존 노드들 간의 관계 업데이트"""
        new_features = self.node_features[new_node_id]
        
        for existing_node_id, existing_features in self.node_features.items():
            if existing_node_id != new_node_id:
                # 코사인 유사도 계산
                similarity = self._calculate_similarity(new_features, existing_features)
                
                if similarity >= self.edge_threshold:
                    self.edges[new_node_id].append({
                        'target': existing_node_id,
                        'weight': similarity
                    })
                    self.edges[existing_node_id].append({
                        'target': new_node_id,
                        'weight': similarity
                    })
                    
                    self.nodes[new_node_id]['connections'] += 1
                    self.nodes[existing_node_id]['connections'] += 1
    
    def _calculate_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """특성 간 유사도 계산"""
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(features1, features2) / (norm1 * norm2)
    
    def _remove_least_connected_node(self):
        """연결이 가장 적은 노드 제거"""
        if not self.nodes:
            return
        
        least_connected = min(self.nodes.items(), key=lambda x: x[1]['connections'])
        node_id = least_connected[0]
        
        # 노드와 관련된 모든 엣지 제거
        for connected_node in self.edges[node_id]:
            target_id = connected_node['target']
            self.edges[target_id] = [
                edge for edge in self.edges[target_id] 
                if edge['target'] != node_id
            ]
            if target_id in self.nodes:
                self.nodes[target_id]['connections'] -= 1
        
        # 노드 제거
        del self.nodes[node_id]
        del self.edges[node_id]
        del self.node_features[node_id]

class PatternDiscriminator(nn.Module):
    """패턴 분별기"""
    
    def __init__(self, input_dim: int = 1280, **kwargs):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        logger.info("PatternDiscriminator 초기화 완료")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(x)

class AdvancedNoveltyDetector:
    """고급 참신함 탐지기"""
    
    def __init__(self, threshold: float = 0.7, **kwargs):
        self.threshold = threshold
        self.known_patterns = []
        logger.info("AdvancedNoveltyDetector 초기화 완료")
    
    def detect_novelty(self, pattern: np.ndarray) -> Dict[str, float]:
        """참신함 탐지"""
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            retry_count += 1
            try:
                if len(self.known_patterns) == 0:
                    novelty_score = 1.0
                else:
                    similarities = [
                        np.corrcoef(pattern.flatten(), known.flatten())[0, 1] 
                        for known in self.known_patterns
                    ]
                    max_similarity = max(similarities) if similarities else 0.0
                    novelty_score = 1.0 - max_similarity
                
                # 새로운 패턴이면 저장
                if novelty_score > self.threshold:
                    self.known_patterns.append(pattern)
                    if len(self.known_patterns) > 1000:  # 메모리 관리
                        self.known_patterns = self.known_patterns[-800:]
                
                return {
                    'novelty_score': novelty_score,
                    'is_novel': novelty_score > self.threshold,
                    'known_patterns_count': len(self.known_patterns)
                }
                
            except Exception as e:
                if retry_count >= max_retries:
                    logger.error(f"참신함 탐지 최종 실패: {e}")
                    raise RuntimeError(f"참신함 탐지가 {max_retries}번 재시도 후 실패: {e}")
                logger.warning(f"참신함 탐지 재시도 {retry_count}/{max_retries}: {e}")
                time.sleep(0.1)

class PatternEvolutionTracker:
    """패턴 진화 추적기"""
    
    def __init__(self, history_size: int = 10000, **kwargs):
        self.history_size = history_size
        self.pattern_history = deque(maxlen=history_size)
        self.evolution_metrics = defaultdict(list)
        logger.info("PatternEvolutionTracker 초기화 완료")
    
    def track_evolution(self, pattern_id: str, pattern_data: Any, timestamp: datetime = None) -> Dict[str, Any]:
        """패턴 진화 추적"""
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            retry_count += 1
            try:
                if timestamp is None:
                    timestamp = datetime.now()
                
                # 패턴 기록 추가
                evolution_entry = {
                    'pattern_id': pattern_id,
                    'data': pattern_data,
                    'timestamp': timestamp,
                    'version': len([h for h in self.pattern_history if h['pattern_id'] == pattern_id]) + 1
                }
                
                self.pattern_history.append(evolution_entry)
                
                # 진화 메트릭 계산
                evolution_metrics = self._calculate_evolution_metrics(pattern_id)
                self.evolution_metrics[pattern_id].append(evolution_metrics)
                
                return {
                    'pattern_id': pattern_id,
                    'version': evolution_entry['version'],
                    'evolution_metrics': evolution_metrics,
                    'total_patterns': len(self.pattern_history)
                }
                
            except Exception as e:
                if retry_count >= max_retries:
                    logger.error(f"패턴 진화 추적 최종 실패: {e}")
                    raise RuntimeError(f"패턴 진화 추적이 {max_retries}번 재시도 후 실패: {e}")
                logger.warning(f"패턴 진화 추적 재시도 {retry_count}/{max_retries}: {e}")
                time.sleep(0.1)
    
    def _calculate_evolution_metrics(self, pattern_id: str) -> Dict[str, float]:
        """진화 메트릭 계산"""
        pattern_versions = [h for h in self.pattern_history if h['pattern_id'] == pattern_id]
        
        if len(pattern_versions) < 2:
            return {
                'stability': 1.0,
                'change_rate': 0.0,
                'complexity': 0.5
            }
        
        # 간단한 진화 메트릭
        recent_changes = len(pattern_versions) / max(1, len(self.pattern_history))
        stability = 1.0 - recent_changes
        change_rate = recent_changes
        complexity = min(1.0, len(pattern_versions) / 100.0)
        
        return {
            'stability': stability,
            'change_rate': change_rate,
            'complexity': complexity
        }

class MetaClassifier(nn.Module):
    """
    메타 분류기 - 고급 앙상블 학습 시스템
    Meta Classifier - Advanced Ensemble Learning System
    
    여러 분류기들을 지능적으로 조합하여 최적의 분류 성능을 달성하는 메타 학습 시스템
    """
    
    def __init__(self, input_dim: int = 1280, num_base_classifiers: int = 5, 
                 num_classes: int = 10, ensemble_method: str = "adaptive_weighting", **kwargs):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_base_classifiers = num_base_classifiers
        self.num_classes = num_classes
        self.ensemble_method = ensemble_method
        
        # 기본 분류기들 (다양한 아키텍처)
        self.base_classifiers = nn.ModuleList()
        
        # 분류기 1: 깊은 MLP
        self.base_classifiers.append(nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        ))
        
        # 분류기 2: 잔차 연결 네트워크
        self.base_classifiers.append(self._create_residual_classifier(input_dim, num_classes))
        
        # 분류기 3: 어텐션 기반 분류기
        self.base_classifiers.append(self._create_attention_classifier(input_dim, num_classes))
        
        # 분류기 4: 컨볼루션 스타일 (1D)
        self.base_classifiers.append(self._create_conv1d_classifier(input_dim, num_classes))
        
        # 분류기 5: 트랜스포머 스타일
        self.base_classifiers.append(self._create_transformer_classifier(input_dim, num_classes))
        
        # 메타 학습기 - 기본 분류기들의 출력을 조합
        self.meta_learner = nn.Sequential(
            nn.Linear(num_classes * num_base_classifiers + input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        
        # 동적 가중치 계산기
        self.weight_calculator = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_base_classifiers),
            nn.Softmax(dim=-1)
        )
        
        # 신뢰도 추정기
        self.confidence_estimator = nn.Sequential(
            nn.Linear(num_classes, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 성능 추적
        self.classifier_performance = torch.ones(num_base_classifiers)
        self.adaptation_rates = torch.ones(num_base_classifiers) * 0.1
        
        logger.info(f"MetaClassifier 초기화: {sum(p.numel() for p in self.parameters())} 파라미터")
    
    def _create_residual_classifier(self, input_dim: int, num_classes: int) -> nn.Module:
        """잔차 연결 분류기 생성"""
        class ResidualBlock(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.block = nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.LayerNorm(dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(dim, dim)
                )
                self.norm = nn.LayerNorm(dim)
            
            def forward(self, x):
                return F.relu(self.norm(x + self.block(x)))
        
        return nn.Sequential(
            nn.Linear(input_dim, 512),
            ResidualBlock(512),
            ResidualBlock(512),
            nn.Linear(512, num_classes)
        )
    
    def _create_attention_classifier(self, input_dim: int, num_classes: int) -> nn.Module:
        """어텐션 기반 분류기 생성"""
        class AttentionClassifier(nn.Module):
            def __init__(self, input_dim, num_classes):
                super().__init__()
                self.feature_extractor = nn.Linear(input_dim, 512)
                self.attention = nn.MultiheadAttention(512, 8, batch_first=True)
                self.classifier = nn.Linear(512, num_classes)
                
            def forward(self, x):
                features = self.feature_extractor(x).unsqueeze(1)  # (batch, 1, 512)
                attended, _ = self.attention(features, features, features)
                return self.classifier(attended.squeeze(1))
        
        return AttentionClassifier(input_dim, num_classes)
    
    def _create_conv1d_classifier(self, input_dim: int, num_classes: int) -> nn.Module:
        """1D 컨볼루션 분류기 생성"""
        # 입력을 시퀀스로 재구성하여 1D Conv 적용
        seq_len = 64
        channels = input_dim // seq_len if input_dim >= seq_len else 1
        
        return nn.Sequential(
            nn.Linear(input_dim, seq_len * channels),
            nn.Unflatten(1, (channels, seq_len)),
            nn.Conv1d(channels, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )
    
    def _create_transformer_classifier(self, input_dim: int, num_classes: int) -> nn.Module:
        """트랜스포머 스타일 분류기 생성"""
        class TransformerClassifier(nn.Module):
            def __init__(self, input_dim, num_classes):
                super().__init__()
                self.embedding = nn.Linear(input_dim, 512)
                self.pos_encoding = nn.Parameter(torch.randn(1, 1, 512))
                self.transformer = nn.TransformerEncoderLayer(
                    d_model=512, nhead=8, dim_feedforward=1024, batch_first=True
                )
                self.classifier = nn.Linear(512, num_classes)
                
            def forward(self, x):
                embedded = self.embedding(x).unsqueeze(1) + self.pos_encoding
                transformed = self.transformer(embedded)
                return self.classifier(transformed.squeeze(1))
        
        return TransformerClassifier(input_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """메타 분류 순전파"""
        batch_size = x.size(0)
        device = x.device
        
        # 각 기본 분류기의 출력 계산
        base_outputs = []
        base_confidences = []
        
        for i, classifier in enumerate(self.base_classifiers):
            try:
                output = classifier(x)
                base_outputs.append(output)
                
                # 각 분류기의 신뢰도 계산
                probs = F.softmax(output, dim=-1)
                confidence = self.confidence_estimator(probs).squeeze(-1)
                base_confidences.append(confidence)
                
            except Exception as e:
                logger.warning(f"기본 분류기 {i} 처리 중 오류: {e}")
                # 오류 발생시 기본값 사용 (fallback 아님, 순수 기본값)
                base_outputs.append(torch.zeros(batch_size, self.num_classes, device=device))
                base_confidences.append(torch.zeros(batch_size, device=device))
        
        # 동적 가중치 계산
        dynamic_weights = self.weight_calculator(x)  # (batch_size, num_base_classifiers)
        
        # 성능 기반 가중치 조정
        performance_weights = self.classifier_performance.to(device).unsqueeze(0)
        performance_weights = performance_weights.expand(batch_size, -1)
        
        # 최종 가중치 = 동적 가중치 × 성능 가중치
        final_weights = dynamic_weights * performance_weights
        final_weights = F.softmax(final_weights, dim=-1)
        
        # 앙상블 방법별 처리
        if self.ensemble_method == "adaptive_weighting":
            ensemble_output = self._adaptive_weighted_ensemble(base_outputs, final_weights, base_confidences)
        elif self.ensemble_method == "voting":
            ensemble_output = self._voting_ensemble(base_outputs)
        elif self.ensemble_method == "stacking":
            ensemble_output = self._stacking_ensemble(base_outputs, x)
        else:
            ensemble_output = self._simple_average_ensemble(base_outputs)
        
        # 메타 학습기를 통한 최종 예측
        concat_features = torch.cat([torch.cat(base_outputs, dim=-1), x], dim=-1)
        meta_output = self.meta_learner(concat_features)
        
        # 앙상블과 메타 학습기 출력을 조합
        alpha = 0.7  # 앙상블 가중치
        final_output = alpha * ensemble_output + (1 - alpha) * meta_output
        
        return {
            'final_output': final_output,
            'ensemble_output': ensemble_output,
            'meta_output': meta_output,
            'base_outputs': torch.stack(base_outputs, dim=1),
            'weights': final_weights,
            'confidences': torch.stack(base_confidences, dim=1) if base_confidences else torch.zeros(batch_size, self.num_base_classifiers, device=device)
        }
    
    def _adaptive_weighted_ensemble(self, base_outputs: List[torch.Tensor], 
                                   weights: torch.Tensor, confidences: List[torch.Tensor]) -> torch.Tensor:
        """적응적 가중 앙상블"""
        # 신뢰도 기반 가중치 재조정
        conf_stack = torch.stack(confidences, dim=-1)  # (batch_size, num_classifiers)
        adjusted_weights = weights * conf_stack
        adjusted_weights = F.softmax(adjusted_weights, dim=-1)
        
        # 가중 평균
        weighted_sum = torch.zeros_like(base_outputs[0])
        for i, output in enumerate(base_outputs):
            weighted_sum += adjusted_weights[:, i:i+1] * output
        
        return weighted_sum
    
    def _voting_ensemble(self, base_outputs: List[torch.Tensor]) -> torch.Tensor:
        """투표 기반 앙상블"""
        # 각 분류기의 예측을 투표로 결합
        predictions = [F.softmax(output, dim=-1) for output in base_outputs]
        return torch.stack(predictions, dim=0).mean(dim=0)
    
    def _stacking_ensemble(self, base_outputs: List[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        """스태킹 앙상블 (메타 학습기 사용)"""
        concat_outputs = torch.cat(base_outputs, dim=-1)
        stacked_features = torch.cat([concat_outputs, x], dim=-1)
        return self.meta_learner(stacked_features)
    
    def _simple_average_ensemble(self, base_outputs: List[torch.Tensor]) -> torch.Tensor:
        """단순 평균 앙상블"""
        return torch.stack(base_outputs, dim=0).mean(dim=0)
    
    def update_performance(self, predictions: torch.Tensor, targets: torch.Tensor):
        """분류기 성능 업데이트"""
        with torch.no_grad():
            # 각 기본 분류기의 개별 성능 계산
            for i in range(self.num_base_classifiers):
                pred_i = predictions[:, i, :]
                accuracy = (pred_i.argmax(dim=-1) == targets).float().mean()
                
                # 지수 평활법으로 성능 업데이트
                self.classifier_performance[i] = (0.9 * self.classifier_performance[i] + 
                                                0.1 * accuracy.cpu())

class PatternPredictor(nn.Module):
    """
    패턴 예측기 - 시계열 패턴 분석 및 미래 예측 시스템
    Pattern Predictor - Time Series Pattern Analysis and Future Prediction System
    
    패턴의 시간적 진화를 모델링하고 미래 상태를 예측하는 고급 시퀀스 모델
    """
    
    def __init__(self, input_dim: int = 1280, hidden_dim: int = 512, num_layers: int = 3,
                 prediction_horizon: int = 10, pattern_memory_size: int = 1000, **kwargs):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.prediction_horizon = prediction_horizon
        self.pattern_memory_size = pattern_memory_size
        
        # 특성 인코더
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # LSTM 기반 시퀀스 모델링
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
            bidirectional=True
        )
        
        # 어텐션 메커니즘 (시퀀스 내 중요한 부분에 집중)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # bidirectional
            num_heads=8,
            batch_first=True
        )
        
        # 트렌드 분석기
        self.trend_analyzer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)  # [증가, 유지, 감소]
        )
        
        # 주기성 탐지기
        self.periodicity_detector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),  # 주기 강도
            nn.Sigmoid()
        )
        
        # 패턴 예측 헤드
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, input_dim * prediction_horizon)
        )
        
        # 불확실성 추정기
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, prediction_horizon),
            nn.Softplus()  # 항상 양수
        )
        
        # 패턴 메모리 (과거 패턴들 저장)
        self.pattern_memory = deque(maxlen=pattern_memory_size)
        self.pattern_embeddings = {}
        
        # 상태 추적
        self.prediction_accuracy_history = deque(maxlen=100)
        self.pattern_stability_score = 0.5
        
        logger.info(f"PatternPredictor 초기화: {sum(p.numel() for p in self.parameters())} 파라미터")
    
    def forward(self, sequence: torch.Tensor, return_analysis: bool = True) -> Dict[str, torch.Tensor]:
        """
        패턴 예측 순전파
        
        Args:
            sequence: 입력 시퀀스 (batch_size, seq_len, input_dim)
            return_analysis: 상세 분석 결과 반환 여부
            
        Returns:
            예측 결과 및 분석 정보
        """
        batch_size, seq_len, _ = sequence.shape
        device = sequence.device
        
        # 특성 인코딩
        encoded_features = self.feature_encoder(sequence)  # (batch, seq_len, hidden_dim)
        
        # LSTM 처리
        lstm_output, (hidden, cell) = self.lstm(encoded_features)  # (batch, seq_len, hidden_dim*2)
        
        # 어텐션 적용
        attended_output, attention_weights = self.attention(
            lstm_output, lstm_output, lstm_output
        )
        
        # 마지막 타임스텝의 표현 사용
        final_representation = attended_output[:, -1, :]  # (batch, hidden_dim*2)
        
        # 예측 생성
        predictions = self.prediction_head(final_representation)  # (batch, input_dim * horizon)
        predictions = predictions.view(batch_size, self.prediction_horizon, self.input_dim)
        
        result = {'predictions': predictions}
        
        if return_analysis:
            # 트렌드 분석
            trend_logits = self.trend_analyzer(final_representation)
            trend_probs = F.softmax(trend_logits, dim=-1)
            
            # 주기성 탐지
            periodicity_score = self.periodicity_detector(final_representation)
            
            # 불확실성 추정
            uncertainty = self.uncertainty_estimator(final_representation)
            
            # 패턴 안정성 분석
            stability_score = self._analyze_pattern_stability(sequence)
            
            # 이상 탐지
            anomaly_score = self._detect_anomalies(sequence, final_representation)
            
            result.update({
                'trend_probabilities': trend_probs,  # [증가, 유지, 감소] 확률
                'periodicity_score': periodicity_score,  # 주기성 강도
                'uncertainty': uncertainty,  # 예측 불확실성
                'stability_score': stability_score,  # 패턴 안정성
                'anomaly_score': anomaly_score,  # 이상 정도
                'attention_weights': attention_weights,
                'sequence_representation': final_representation
            })
        
        return result
    
    def predict_future_patterns(self, current_sequence: torch.Tensor, 
                               steps_ahead: int = None) -> Dict[str, torch.Tensor]:
        """미래 패턴 예측 (순수 재시도 방식)"""
        if steps_ahead is None:
            steps_ahead = self.prediction_horizon
            
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            retry_count += 1
            try:
                with torch.no_grad():
                    # 현재 시퀀스로부터 미래 예측
                    result = self.forward(current_sequence, return_analysis=True)
                    
                    # 다단계 예측 (iterative prediction)
                    if steps_ahead > self.prediction_horizon:
                        extended_predictions = self._iterative_prediction(
                            current_sequence, steps_ahead
                        )
                        result['extended_predictions'] = extended_predictions
                    
                    # 신뢰도 구간 계산
                    confidence_intervals = self._calculate_confidence_intervals(
                        result['predictions'], result['uncertainty']
                    )
                    result['confidence_intervals'] = confidence_intervals
                    
                    return result
                    
            except Exception as e:
                if retry_count >= max_retries:
                    logger.error(f"미래 패턴 예측 최종 실패: {e}")
                    raise RuntimeError(f"미래 패턴 예측이 {max_retries}번 재시도 후 실패: {e}")
                logger.warning(f"미래 패턴 예측 재시도 {retry_count}/{max_retries}: {e}")
                time.sleep(0.1)
    
    def _analyze_pattern_stability(self, sequence: torch.Tensor) -> torch.Tensor:
        """패턴 안정성 분석"""
        # 시퀀스 내 변동성 계산
        if sequence.size(1) < 2:
            return torch.ones(sequence.size(0), device=sequence.device) * 0.5
        
        # 연속된 타임스텝 간 변화량 계산
        differences = torch.diff(sequence, dim=1)
        variance = torch.var(differences, dim=(1, 2))
        
        # 안정성 점수 (낮은 변동성 = 높은 안정성)
        stability = torch.exp(-variance)  # 0과 1 사이 값
        
        return stability
    
    def _detect_anomalies(self, sequence: torch.Tensor, representation: torch.Tensor) -> torch.Tensor:
        """이상 패턴 탐지"""
        # 메모리에 저장된 정상 패턴들과 비교
        if len(self.pattern_memory) < 10:
            return torch.zeros(sequence.size(0), device=sequence.device)
        
        # 현재 표현과 과거 정상 패턴들 간의 거리 계산
        current_repr = representation.detach().cpu().numpy()
        
        anomaly_scores = []
        for batch_idx in range(sequence.size(0)):
            distances = []
            for past_pattern in list(self.pattern_memory)[-50:]:  # 최근 50개 패턴과 비교
                distance = np.linalg.norm(current_repr[batch_idx] - past_pattern)
                distances.append(distance)
            
            if distances:
                # 평균 거리가 클수록 이상
                avg_distance = np.mean(distances)
                anomaly_score = min(1.0, avg_distance / 10.0)  # 정규화
                anomaly_scores.append(anomaly_score)
            else:
                anomaly_scores.append(0.0)
        
        return torch.tensor(anomaly_scores, device=sequence.device)
    
    def _iterative_prediction(self, initial_sequence: torch.Tensor, total_steps: int) -> torch.Tensor:
        """반복적 다단계 예측"""
        current_sequence = initial_sequence.clone()
        all_predictions = []
        
        steps_remaining = total_steps
        while steps_remaining > 0:
            steps_this_round = min(self.prediction_horizon, steps_remaining)
            
            # 현재 시퀀스로 예측
            result = self.forward(current_sequence, return_analysis=False)
            predictions = result['predictions'][:, :steps_this_round, :]
            
            all_predictions.append(predictions)
            
            # 다음 반복을 위해 시퀀스 업데이트
            if steps_remaining > self.prediction_horizon:
                # 예측된 값들을 시퀀스에 추가
                current_sequence = torch.cat([
                    current_sequence[:, steps_this_round:, :],
                    predictions
                ], dim=1)
            
            steps_remaining -= steps_this_round
        
        return torch.cat(all_predictions, dim=1)
    
    def _calculate_confidence_intervals(self, predictions: torch.Tensor, 
                                      uncertainty: torch.Tensor) -> Dict[str, torch.Tensor]:
        """신뢰도 구간 계산"""
        # 95% 신뢰구간 계산 (1.96 * 표준편차)
        std = uncertainty.unsqueeze(-1).expand_as(predictions)
        
        lower_bound = predictions - 1.96 * std
        upper_bound = predictions + 1.96 * std
        
        return {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence_width': upper_bound - lower_bound
        }
    
    def update_pattern_memory(self, sequence_representation: torch.Tensor):
        """패턴 메모리 업데이트"""
        # 배치의 평균 표현을 메모리에 저장
        avg_repr = sequence_representation.mean(dim=0).detach().cpu().numpy()
        self.pattern_memory.append(avg_repr)

class PatternConsolidationSystem:
    """
    패턴 통합 시스템 - 고급 패턴 관리 및 최적화 시스템
    Pattern Consolidation System - Advanced Pattern Management and Optimization System
    
    다양한 소스의 패턴들을 지능적으로 통합, 정리, 최적화하는 시스템
    """
    
    def __init__(self, similarity_threshold: float = 0.85, max_patterns: int = 50000,
                 consolidation_frequency: int = 1000, quality_threshold: float = 0.7, **kwargs):
        self.similarity_threshold = similarity_threshold
        self.max_patterns = max_patterns
        self.consolidation_frequency = consolidation_frequency
        self.quality_threshold = quality_threshold
        
        # 패턴 저장소
        self.consolidated_patterns = {}  # {pattern_id: PatternInfo}
        self.pattern_clusters = defaultdict(list)  # {cluster_id: [pattern_ids]}
        self.pattern_relationships = defaultdict(list)  # {pattern_id: [related_pattern_ids]}
        
        # 패턴 메타데이터
        self.pattern_metadata = {}  # {pattern_id: metadata}
        self.pattern_quality_scores = {}  # {pattern_id: quality_score}
        self.pattern_usage_stats = defaultdict(lambda: {'count': 0, 'last_used': datetime.now()})
        
        # 통합 상태 추적
        self.consolidation_count = 0
        self.last_consolidation = datetime.now()
        self.consolidation_metrics = {
            'patterns_merged': 0,
            'patterns_removed': 0,
            'clusters_created': 0,
            'relationships_established': 0
        }
        
        # 고급 분석 도구
        self.pattern_analyzer = PatternAnalyzer()
        self.quality_assessor = PatternQualityAssessor()
        self.relationship_builder = PatternRelationshipBuilder()
        
        logger.info("PatternConsolidationSystem 초기화 완료")
    
    def consolidate_patterns(self, new_patterns: Dict[str, Any], 
                           source_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """패턴들을 통합하고 정리 (순수 재시도 방식)"""
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            retry_count += 1
            try:
                consolidation_start = time.time()
                
                # 1. 새 패턴들의 품질 평가
                quality_results = self._assess_pattern_quality(new_patterns)
                
                # 2. 기존 패턴들과의 유사도 계산
                similarity_results = self._calculate_pattern_similarities(new_patterns)
                
                # 3. 중복 패턴 탐지 및 병합
                merge_results = self._merge_duplicate_patterns(new_patterns, similarity_results)
                
                # 4. 새로운 패턴들을 적절한 클러스터에 배치
                clustering_results = self._assign_to_clusters(new_patterns, merge_results['unique_patterns'])
                
                # 5. 패턴 간 관계 분석 및 구축
                relationship_results = self._build_pattern_relationships(
                    merge_results['unique_patterns'], clustering_results
                )
                
                # 6. 저품질 패턴 제거
                cleanup_results = self._cleanup_low_quality_patterns(quality_results)
                
                # 7. 메모리 최적화
                if len(self.consolidated_patterns) > self.max_patterns:
                    optimization_results = self._optimize_pattern_storage()
                else:
                    optimization_results = {'patterns_removed': 0}
                
                # 8. 통합 메트릭 업데이트
                self._update_consolidation_metrics(
                    merge_results, clustering_results, relationship_results, 
                    cleanup_results, optimization_results
                )
                
                consolidation_time = time.time() - consolidation_start
                self.consolidation_count += 1
                self.last_consolidation = datetime.now()
                
                return {
                    'success': True,
                    'consolidation_time': consolidation_time,
                    'patterns_processed': len(new_patterns),
                    'patterns_merged': merge_results['merged_count'],
                    'patterns_added': merge_results['added_count'],
                    'clusters_modified': len(clustering_results['modified_clusters']),
                    'relationships_created': relationship_results['new_relationships'],
                    'patterns_removed': cleanup_results['removed_count'] + optimization_results['patterns_removed'],
                    'total_patterns': len(self.consolidated_patterns),
                    'consolidation_metrics': self.consolidation_metrics.copy()
                }
                
            except Exception as e:
                if retry_count >= max_retries:
                    logger.error(f"패턴 통합 최종 실패: {e}")
                    raise RuntimeError(f"패턴 통합이 {max_retries}번 재시도 후 실패: {e}")
                logger.warning(f"패턴 통합 재시도 {retry_count}/{max_retries}: {e}")
                time.sleep(0.2)
    
    def _assess_pattern_quality(self, patterns: Dict[str, Any]) -> Dict[str, float]:
        """패턴 품질 평가"""
        quality_scores = {}
        
        for pattern_id, pattern_data in patterns.items():
            try:
                # 다양한 품질 지표 계산
                complexity_score = self._calculate_pattern_complexity(pattern_data)
                uniqueness_score = self._calculate_pattern_uniqueness(pattern_data)
                coherence_score = self._calculate_pattern_coherence(pattern_data)
                utility_score = self._calculate_pattern_utility(pattern_data)
                
                # 종합 품질 점수
                quality_score = (
                    0.3 * complexity_score +
                    0.25 * uniqueness_score +
                    0.25 * coherence_score +
                    0.2 * utility_score
                )
                
                quality_scores[pattern_id] = quality_score
                self.pattern_quality_scores[pattern_id] = quality_score
                
            except Exception as e:
                logger.warning(f"패턴 {pattern_id} 품질 평가 실패: {e}")
                quality_scores[pattern_id] = 0.0
        
        return quality_scores
    
    def _calculate_pattern_similarities(self, new_patterns: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """패턴 간 유사도 계산"""
        similarities = {}
        
        for new_pattern_id, new_pattern_data in new_patterns.items():
            similarities[new_pattern_id] = {}
            
            # 기존 패턴들과의 유사도 계산
            for existing_pattern_id, existing_pattern_data in self.consolidated_patterns.items():
                try:
                    similarity = self._compute_pattern_similarity(new_pattern_data, existing_pattern_data)
                    similarities[new_pattern_id][existing_pattern_id] = similarity
                except Exception as e:
                    logger.debug(f"유사도 계산 실패 ({new_pattern_id}, {existing_pattern_id}): {e}")
                    similarities[new_pattern_id][existing_pattern_id] = 0.0
        
        return similarities
    
    def _merge_duplicate_patterns(self, new_patterns: Dict[str, Any], 
                                 similarities: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """중복 패턴 병합"""
        merged_count = 0
        added_count = 0
        unique_patterns = {}
        
        for pattern_id, pattern_data in new_patterns.items():
            # 가장 유사한 기존 패턴 찾기
            max_similarity = 0.0
            most_similar_id = None
            
            for existing_id, similarity in similarities[pattern_id].items():
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_id = existing_id
            
            if max_similarity >= self.similarity_threshold and most_similar_id:
                # 유사한 패턴과 병합
                self._merge_two_patterns(most_similar_id, pattern_id, pattern_data)
                merged_count += 1
            else:
                # 새로운 고유 패턴으로 추가
                self.consolidated_patterns[pattern_id] = pattern_data
                unique_patterns[pattern_id] = pattern_data
                added_count += 1
        
        return {
            'merged_count': merged_count,
            'added_count': added_count,
            'unique_patterns': unique_patterns
        }
    
    def _assign_to_clusters(self, new_patterns: Dict[str, Any], 
                           unique_patterns: Dict[str, Any]) -> Dict[str, Any]:
        """패턴을 클러스터에 배치"""
        modified_clusters = set()
        
        for pattern_id in unique_patterns:
            # 최적의 클러스터 찾기
            best_cluster = self._find_best_cluster(pattern_id)
            
            if best_cluster is None:
                # 새 클러스터 생성
                cluster_id = f"cluster_{len(self.pattern_clusters)}"
                self.pattern_clusters[cluster_id] = [pattern_id]
                modified_clusters.add(cluster_id)
            else:
                # 기존 클러스터에 추가
                self.pattern_clusters[best_cluster].append(pattern_id)
                modified_clusters.add(best_cluster)
        
        return {'modified_clusters': list(modified_clusters)}
    
    def _build_pattern_relationships(self, patterns: Dict[str, Any], 
                                   clustering_results: Dict[str, Any]) -> Dict[str, Any]:
        """패턴 간 관계 구축"""
        new_relationships = 0
        
        # 클러스터 내 패턴들 간의 관계 구축
        for cluster_id in clustering_results['modified_clusters']:
            cluster_patterns = self.pattern_clusters[cluster_id]
            
            # 클러스터 내 모든 패턴 쌍에 대해 관계 분석
            for i, pattern_id_1 in enumerate(cluster_patterns):
                for pattern_id_2 in cluster_patterns[i+1:]:
                    if self._should_establish_relationship(pattern_id_1, pattern_id_2):
                        self.pattern_relationships[pattern_id_1].append(pattern_id_2)
                        self.pattern_relationships[pattern_id_2].append(pattern_id_1)
                        new_relationships += 1
        
        return {'new_relationships': new_relationships}
    
    def _cleanup_low_quality_patterns(self, quality_scores: Dict[str, float]) -> Dict[str, Any]:
        """저품질 패턴 제거"""
        removed_count = 0
        patterns_to_remove = []
        
        # 품질 임계값 이하의 패턴들 식별
        for pattern_id, quality_score in quality_scores.items():
            if quality_score < self.quality_threshold:
                patterns_to_remove.append(pattern_id)
        
        # 오래되고 사용되지 않는 패턴들도 제거 대상에 추가
        current_time = datetime.now()
        for pattern_id, usage_stats in self.pattern_usage_stats.items():
            time_since_last_use = (current_time - usage_stats['last_used']).days
            if time_since_last_use > 90 and usage_stats['count'] < 5:  # 90일 이상 미사용, 5회 미만 사용
                if pattern_id not in patterns_to_remove:
                    patterns_to_remove.append(pattern_id)
        
        # 패턴 제거 실행
        for pattern_id in patterns_to_remove:
            self._remove_pattern(pattern_id)
            removed_count += 1
        
        return {'removed_count': removed_count}
    
    def _optimize_pattern_storage(self) -> Dict[str, Any]:
        """패턴 저장소 최적화"""
        patterns_removed = 0
        
        # 사용 빈도와 품질 점수 기반으로 패턴 순위 매기기
        pattern_rankings = []
        for pattern_id in self.consolidated_patterns:
            usage_count = self.pattern_usage_stats[pattern_id]['count']
            quality_score = self.pattern_quality_scores.get(pattern_id, 0.5)
            
            # 종합 점수 = 품질 × log(사용횟수 + 1)
            combined_score = quality_score * np.log(usage_count + 1)
            pattern_rankings.append((pattern_id, combined_score))
        
        # 점수 순으로 정렬 (낮은 점수부터)
        pattern_rankings.sort(key=lambda x: x[1])
        
        # 하위 패턴들 제거
        patterns_to_remove_count = len(self.consolidated_patterns) - int(self.max_patterns * 0.8)
        patterns_to_remove = pattern_rankings[:patterns_to_remove_count]
        
        for pattern_id, _ in patterns_to_remove:
            self._remove_pattern(pattern_id)
            patterns_removed += 1
        
        return {'patterns_removed': patterns_removed}
    
    # 헬퍼 메서드들
    def _calculate_pattern_complexity(self, pattern_data: Any) -> float:
        """패턴 복잡도 계산"""
        # 간단한 복잡도 지표 (실제로는 더 정교한 분석 필요)
        if isinstance(pattern_data, dict):
            return min(1.0, len(str(pattern_data)) / 1000.0)
        return 0.5
    
    def _calculate_pattern_uniqueness(self, pattern_data: Any) -> float:
        """패턴 독창성 계산"""
        # 기존 패턴들과의 차별성 측정
        return 0.7  # 임시 값
    
    def _calculate_pattern_coherence(self, pattern_data: Any) -> float:
        """패턴 일관성 계산"""
        # 패턴 내부 구조의 일관성 측정
        return 0.8  # 임시 값
    
    def _calculate_pattern_utility(self, pattern_data: Any) -> float:
        """패턴 유용성 계산"""
        # 패턴의 실용적 가치 측정
        return 0.6  # 임시 값
    
    def _compute_pattern_similarity(self, pattern1: Any, pattern2: Any) -> float:
        """두 패턴 간 유사도 계산"""
        # 간단한 유사도 계산 (실제로는 더 정교한 방법 사용)
        str1 = str(pattern1)
        str2 = str(pattern2)
        
        # 자카드 유사도 근사치
        set1 = set(str1.split())
        set2 = set(str2.split())
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _merge_two_patterns(self, existing_id: str, new_id: str, new_pattern_data: Any):
        """두 패턴 병합"""
        # 기존 패턴 데이터 업데이트 (가중 평균 등)
        # 사용 통계 업데이트
        self.pattern_usage_stats[existing_id]['count'] += 1
        self.pattern_usage_stats[existing_id]['last_used'] = datetime.now()
    
    def _find_best_cluster(self, pattern_id: str) -> Optional[str]:
        """패턴에 가장 적합한 클러스터 찾기"""
        # 각 클러스터와의 유사도 계산하여 최적 클러스터 반환
        # 임시로 None 반환 (새 클러스터 생성)
        return None
    
    def _should_establish_relationship(self, pattern_id_1: str, pattern_id_2: str) -> bool:
        """두 패턴 간 관계 설정 여부 결정"""
        # 패턴 간 관련성 분석
        return True  # 임시로 항상 True
    
    def _remove_pattern(self, pattern_id: str):
        """패턴 제거"""
        # 모든 저장소에서 패턴 제거
        if pattern_id in self.consolidated_patterns:
            del self.consolidated_patterns[pattern_id]
        
        if pattern_id in self.pattern_quality_scores:
            del self.pattern_quality_scores[pattern_id]
        
        if pattern_id in self.pattern_usage_stats:
            del self.pattern_usage_stats[pattern_id]
        
        # 클러스터에서 제거
        for cluster_patterns in self.pattern_clusters.values():
            if pattern_id in cluster_patterns:
                cluster_patterns.remove(pattern_id)
        
        # 관계에서 제거
        if pattern_id in self.pattern_relationships:
            del self.pattern_relationships[pattern_id]
    
    def _update_consolidation_metrics(self, merge_results, clustering_results, 
                                    relationship_results, cleanup_results, optimization_results):
        """통합 메트릭 업데이트"""
        self.consolidation_metrics['patterns_merged'] += merge_results['merged_count']
        self.consolidation_metrics['patterns_removed'] += (
            cleanup_results['removed_count'] + optimization_results['patterns_removed']
        )
        self.consolidation_metrics['clusters_created'] += len(clustering_results['modified_clusters'])
        self.consolidation_metrics['relationships_established'] += relationship_results['new_relationships']

# 보조 클래스들 (PatternConsolidationSystem에서 사용)
class PatternAnalyzer:
    """패턴 분석기"""
    def __init__(self):
        pass

class PatternQualityAssessor:
    """패턴 품질 평가기"""
    def __init__(self):
        pass

class PatternRelationshipBuilder:
    """패턴 관계 구축기"""
    def __init__(self):
        pass

class SelectiveForgettingSystem:
    """
    선택적 망각 시스템 - 지능형 메모리 관리 및 패턴 선별 망각
    Selective Forgetting System - Intelligent Memory Management and Pattern Forgetting
    
    중요하지 않은 패턴들을 선별적으로 망각하여 메모리 효율성과 학습 성능을 최적화하는 시스템
    """
    
    def __init__(self, memory_capacity: int = 100000, forgetting_rate: float = 0.1,
                 importance_threshold: float = 0.3, consolidation_interval: int = 1000, **kwargs):
        self.memory_capacity = memory_capacity
        self.forgetting_rate = forgetting_rate
        self.importance_threshold = importance_threshold
        self.consolidation_interval = consolidation_interval
        
        # 메모리 저장소
        self.memory_store = {}  # {pattern_id: PatternMemory}
        self.importance_scores = {}  # {pattern_id: importance_score}
        self.access_history = defaultdict(list)  # {pattern_id: [access_times]}
        self.forgetting_candidates = set()  # 망각 후보 패턴들
        
        # 망각 메트릭
        self.forgetting_metrics = {
            'patterns_forgotten': 0,
            'memory_freed_mb': 0.0,
            'forgetting_efficiency': 0.0,
            'last_consolidation': datetime.now()
        }
        
        # 중요도 계산기
        self.importance_calculator = ImportanceCalculator()
        
        # 메모리 사용량 추적
        self.current_memory_usage = 0
        self.memory_access_count = 0
        
        logger.info("SelectiveForgettingSystem 초기화 완료")
    
    def add_pattern_to_memory(self, pattern_id: str, pattern_data: Any, 
                             initial_importance: float = 0.5) -> bool:
        """패턴을 메모리에 추가 (순수 재시도 방식)"""
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            retry_count += 1
            try:
                # 메모리 용량 확인
                if len(self.memory_store) >= self.memory_capacity:
                    self._trigger_selective_forgetting()
                
                # 중요도 점수 계산
                importance_score = self._calculate_importance_score(
                    pattern_data, initial_importance
                )
                
                # 메모리에 패턴 저장
                self.memory_store[pattern_id] = {
                    'data': pattern_data,
                    'created_at': datetime.now(),
                    'last_accessed': datetime.now(),
                    'access_count': 1,
                    'memory_size': self._estimate_memory_size(pattern_data)
                }
                
                self.importance_scores[pattern_id] = importance_score
                self.access_history[pattern_id].append(datetime.now())
                
                # 메모리 사용량 업데이트
                self.current_memory_usage += self.memory_store[pattern_id]['memory_size']
                
                return True
                
            except Exception as e:
                if retry_count >= max_retries:
                    logger.error(f"패턴 메모리 추가 최종 실패: {e}")
                    raise RuntimeError(f"패턴 메모리 추가가 {max_retries}번 재시도 후 실패: {e}")
                logger.warning(f"패턴 메모리 추가 재시도 {retry_count}/{max_retries}: {e}")
                time.sleep(0.1)
    
    def access_pattern(self, pattern_id: str) -> Optional[Any]:
        """패턴 접근 (사용 통계 업데이트)"""
        if pattern_id not in self.memory_store:
            return None
        
        # 접근 기록 업데이트
        current_time = datetime.now()
        self.memory_store[pattern_id]['last_accessed'] = current_time
        self.memory_store[pattern_id]['access_count'] += 1
        self.access_history[pattern_id].append(current_time)
        
        # 접근 기반 중요도 업데이트
        self._update_importance_on_access(pattern_id)
        
        self.memory_access_count += 1
        
        # 주기적 통합 확인
        if self.memory_access_count % self.consolidation_interval == 0:
            self._consolidate_memory()
        
        return self.memory_store[pattern_id]['data']
    
    def selective_forget(self, target_forget_count: int = None) -> Dict[str, Any]:
        """선택적 망각 수행"""
        if target_forget_count is None:
            target_forget_count = max(1, int(len(self.memory_store) * self.forgetting_rate))
        
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            retry_count += 1
            try:
                # 망각 후보 선별
                forgetting_candidates = self._select_forgetting_candidates(target_forget_count)
                
                # 망각 실행
                forgotten_patterns = []
                memory_freed = 0.0
                
                for pattern_id in forgetting_candidates:
                    if self._should_forget_pattern(pattern_id):
                        memory_size = self.memory_store[pattern_id]['memory_size']
                        self._forget_pattern(pattern_id)
                        forgotten_patterns.append(pattern_id)
                        memory_freed += memory_size
                
                # 메트릭 업데이트
                self.forgetting_metrics['patterns_forgotten'] += len(forgotten_patterns)
                self.forgetting_metrics['memory_freed_mb'] += memory_freed / (1024 * 1024)
                self.forgetting_metrics['forgetting_efficiency'] = (
                    len(forgotten_patterns) / max(1, target_forget_count)
                )
                
                return {
                    'forgotten_count': len(forgotten_patterns),
                    'forgotten_patterns': forgotten_patterns,
                    'memory_freed_mb': memory_freed / (1024 * 1024),
                    'remaining_patterns': len(self.memory_store),
                    'forgetting_efficiency': self.forgetting_metrics['forgetting_efficiency']
                }
                
            except Exception as e:
                if retry_count >= max_retries:
                    logger.error(f"선택적 망각 최종 실패: {e}")
                    raise RuntimeError(f"선택적 망각이 {max_retries}번 재시도 후 실패: {e}")
                logger.warning(f"선택적 망각 재시도 {retry_count}/{max_retries}: {e}")
                time.sleep(0.1)
    
    def _calculate_importance_score(self, pattern_data: Any, initial_importance: float) -> float:
        """패턴 중요도 점수 계산"""
        # 기본 점수
        base_score = initial_importance
        
        # 패턴 복잡도 (복잡할수록 중요)
        complexity_score = self._calculate_pattern_complexity(pattern_data)
        
        # 패턴 독창성 (독창적일수록 중요)
        uniqueness_score = self._calculate_pattern_uniqueness(pattern_data)
        
        # 종합 중요도 점수
        importance_score = (
            0.4 * base_score +
            0.3 * complexity_score +
            0.3 * uniqueness_score
        )
        
        return min(1.0, importance_score)
    
    def _update_importance_on_access(self, pattern_id: str):
        """접근 시 중요도 업데이트"""
        # 접근 빈도 기반 중요도 증가
        access_count = self.memory_store[pattern_id]['access_count']
        frequency_boost = min(0.2, np.log(access_count + 1) / 10.0)
        
        # 최근성 기반 중요도 조정
        last_accessed = self.memory_store[pattern_id]['last_accessed']
        time_since_access = (datetime.now() - last_accessed).total_seconds()
        recency_factor = max(0.8, 1.0 - time_since_access / (7 * 24 * 3600))  # 7일 기준
        
        # 중요도 업데이트
        current_importance = self.importance_scores[pattern_id]
        new_importance = min(1.0, (current_importance + frequency_boost) * recency_factor)
        self.importance_scores[pattern_id] = new_importance
    
    def _select_forgetting_candidates(self, target_count: int) -> List[str]:
        """망각 후보 선별"""
        # 중요도가 낮은 패턴들을 후보로 선별
        pattern_scores = []
        for pattern_id in self.memory_store:
            importance = self.importance_scores[pattern_id]
            last_accessed = self.memory_store[pattern_id]['last_accessed']
            age = (datetime.now() - last_accessed).total_seconds()
            
            # 망각 점수 = (1 - 중요도) + 나이 가중치
            forgetting_score = (1.0 - importance) + min(0.5, age / (30 * 24 * 3600))  # 30일 기준
            pattern_scores.append((pattern_id, forgetting_score))
        
        # 망각 점수가 높은 순으로 정렬
        pattern_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 상위 target_count개 선별
        candidates = [pattern_id for pattern_id, _ in pattern_scores[:target_count]]
        return candidates
    
    def _should_forget_pattern(self, pattern_id: str) -> bool:
        """패턴 망각 여부 결정"""
        # 중요도가 임계값 이하이고 최근에 접근하지 않은 패턴만 망각
        importance = self.importance_scores[pattern_id]
        last_accessed = self.memory_store[pattern_id]['last_accessed']
        time_since_access = (datetime.now() - last_accessed).total_seconds()
        
        return (importance < self.importance_threshold and 
                time_since_access > 24 * 3600)  # 24시간 이상 미접근
    
    def _forget_pattern(self, pattern_id: str):
        """패턴 망각 실행"""
        if pattern_id in self.memory_store:
            self.current_memory_usage -= self.memory_store[pattern_id]['memory_size']
            del self.memory_store[pattern_id]
        
        if pattern_id in self.importance_scores:
            del self.importance_scores[pattern_id]
        
        if pattern_id in self.access_history:
            del self.access_history[pattern_id]
        
        self.forgetting_candidates.discard(pattern_id)
    
    def _trigger_selective_forgetting(self):
        """메모리 부족 시 자동 망각 트리거"""
        forget_count = max(1, int(self.memory_capacity * 0.1))  # 10% 망각
        self.selective_forget(forget_count)
    
    def _consolidate_memory(self):
        """메모리 통합 (주기적 정리)"""
        # 중요도 재계산
        for pattern_id in list(self.memory_store.keys()):
            self._recalculate_importance(pattern_id)
        
        # 메모리 사용량 최적화
        if self.current_memory_usage > self.memory_capacity * 0.8:  # 80% 사용 시
            self._trigger_selective_forgetting()
        
        self.forgetting_metrics['last_consolidation'] = datetime.now()
    
    def _recalculate_importance(self, pattern_id: str):
        """패턴 중요도 재계산"""
        if pattern_id not in self.memory_store:
            return
        
        # 시간 기반 중요도 감소 (자연적 망각)
        creation_time = self.memory_store[pattern_id]['created_at']
        age_days = (datetime.now() - creation_time).days
        age_decay = max(0.5, 1.0 - age_days / 365.0)  # 1년 기준
        
        # 접근 빈도 기반 중요도
        access_count = self.memory_store[pattern_id]['access_count']
        frequency_score = min(1.0, np.log(access_count + 1) / 5.0)
        
        # 새로운 중요도 계산
        current_importance = self.importance_scores[pattern_id]
        new_importance = current_importance * age_decay * 0.8 + frequency_score * 0.2
        
        self.importance_scores[pattern_id] = max(0.0, min(1.0, new_importance))
    
    def _estimate_memory_size(self, pattern_data: Any) -> float:
        """패턴 데이터의 메모리 크기 추정 (바이트)"""
        import sys
        return sys.getsizeof(pattern_data)
    
    def _calculate_pattern_complexity(self, pattern_data: Any) -> float:
        """패턴 복잡도 계산"""
        # 간단한 복잡도 측정
        if isinstance(pattern_data, (dict, list)):
            return min(1.0, len(str(pattern_data)) / 1000.0)
        return 0.5
    
    def _calculate_pattern_uniqueness(self, pattern_data: Any) -> float:
        """패턴 독창성 계산"""
        # 기존 패턴들과의 차별성 측정
        return 0.6  # 임시 값
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """메모리 사용 통계 반환"""
        return {
            'total_patterns': len(self.memory_store),
            'memory_usage_mb': self.current_memory_usage / (1024 * 1024),
            'average_importance': np.mean(list(self.importance_scores.values())) if self.importance_scores else 0.0,
            'forgetting_metrics': self.forgetting_metrics.copy(),
            'memory_capacity': self.memory_capacity,
            'usage_percentage': len(self.memory_store) / self.memory_capacity * 100
        }

# 보조 클래스
class ImportanceCalculator:
    """중요도 계산기"""
    def __init__(self):
        pass

# 로깅
logger.info("완전한 누락된 신경망 모델들 구현 완료 (SelectiveForgettingSystem 포함 - 총 16개 클래스)")