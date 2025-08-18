"""
고급 Rumbaugh 구조적 분석 시스템 - Linux 전용
Advanced Rumbaugh Structural Analysis System for Linux

Object Modeling Technique (OMT)를 활용한 고급 윤리적 의사결정 구조 분석
신경망과 그래프 이론을 결합한 차세대 구조적 모델링
"""

import os
import json
import time
import asyncio
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from collections import defaultdict, Counter
import uuid

# 고급 AI 및 그래프 라이브러리
import networkx as nx
# SentenceTransformer는 sentence_transformer_singleton을 통해 사용
# import spacy  # 직접 임포트 제거 - subprocess 사용
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from config import ADVANCED_CONFIG, DEVICE, TORCH_DTYPE, BATCH_SIZE, MODELS_DIR
from utils import run_spacy_subprocess
from data_models import (
    Decision, EthicalSituation, StructuralElement,
    ObjectRelation, StateMachine, StructuralComplexity,
    AdvancedStructuralResult, DynamicInteraction
)

logger = logging.getLogger('RedHeart.AdvancedRumbaugh')

class ObjectType(Enum):
    """구조적 객체 유형"""
    ACTOR = "actor"              # 행위자
    RESOURCE = "resource"        # 자원
    CONSTRAINT = "constraint"    # 제약
    GOAL = "goal"               # 목표
    ACTION = "action"           # 행동
    RELATIONSHIP = "relationship" # 관계
    CONTEXT = "context"         # 맥락
    OUTCOME = "outcome"         # 결과

class RelationType(Enum):
    """관계 유형"""
    DEPENDENCY = "dependency"    # 의존성
    ASSOCIATION = "association"  # 연관
    AGGREGATION = "aggregation"  # 집합
    COMPOSITION = "composition"  # 구성
    INHERITANCE = "inheritance"  # 상속
    CAUSALITY = "causality"     # 인과관계
    CONFLICT = "conflict"       # 갈등
    COOPERATION = "cooperation"  # 협력

class StateType(Enum):
    """상태 유형"""
    INITIAL = "initial"         # 초기 상태
    ACTIVE = "active"          # 활성 상태
    PASSIVE = "passive"        # 수동 상태
    TERMINAL = "terminal"      # 종료 상태
    ERROR = "error"           # 오류 상태
    PENDING = "pending"       # 대기 상태

@dataclass
class StructuralObject:
    """구조적 객체"""
    object_id: str
    object_type: ObjectType
    name: str
    description: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    state: StateType = StateType.INITIAL
    importance: float = 0.0
    confidence: float = 0.0
    extracted_from: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class StructuralRelation:
    """구조적 관계"""
    relation_id: str
    source_object: str
    target_object: str
    relation_type: RelationType
    strength: float = 0.0
    confidence: float = 0.0
    properties: Dict[str, Any] = field(default_factory=dict)
    bidirectional: bool = False
    temporal_order: Optional[int] = None

@dataclass
class ObjectStateMachine:
    """객체 상태 기계"""
    object_id: str
    states: List[StateType]
    transitions: List[Dict[str, Any]]
    current_state: StateType
    state_probabilities: Dict[StateType, float] = field(default_factory=dict)
    transition_matrix: Optional[np.ndarray] = None

@dataclass
class StructuralPattern:
    """구조적 패턴"""
    pattern_id: str
    pattern_type: str
    objects_involved: List[str]
    relations_involved: List[str]
    pattern_frequency: int = 0
    significance_score: float = 0.0
    description: str = ""

class AdvancedNeuralStructureAnalyzer(nn.Module):
    """고급 신경망 기반 구조 분석기"""
    
    def __init__(self, embedding_dim=768, hidden_dim=512, num_object_types=8, num_relation_types=8):
        super().__init__()
        
        # 객체 분류 네트워크
        self.object_classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_object_types),
            nn.Softmax(dim=-1)
        )
        
        # 관계 분류 네트워크
        self.relation_classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_relation_types),
            nn.Softmax(dim=-1)
        )
        
        # 중요도 예측 네트워크
        self.importance_predictor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # 그래프 신경망 (GNN) 레이어들
        self.gnn_layers = nn.ModuleList([
            GNNLayer(embedding_dim, hidden_dim) for _ in range(3)
        ])
        
        # 어텐션 메커니즘
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
    def forward(self, node_embeddings, edge_indices=None, edge_weights=None):
        """순전파"""
        batch_size, num_nodes, embed_dim = node_embeddings.shape
        
        # 객체 분류
        object_classifications = self.object_classifier(node_embeddings)
        
        # 중요도 예측
        importance_scores = self.importance_predictor(node_embeddings)
        
        # GNN을 통한 구조적 임베딩 업데이트
        gnn_output = node_embeddings
        for gnn_layer in self.gnn_layers:
            gnn_output = gnn_layer(gnn_output, edge_indices, edge_weights)
        
        # 어텐션 적용
        attended_output, attention_weights = self.attention(
            gnn_output, gnn_output, gnn_output
        )
        
        # 관계 분류 (모든 노드 쌍에 대해)
        relation_classifications = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                pair_embedding = torch.cat([
                    attended_output[:, i, :],
                    attended_output[:, j, :]
                ], dim=-1)
                relation_pred = self.relation_classifier(pair_embedding)
                relation_classifications.append(relation_pred)
        
        if relation_classifications:
            relation_classifications = torch.stack(relation_classifications, dim=1)
        else:
            relation_classifications = torch.empty(batch_size, 0, self.relation_classifier[-2].out_features)
        
        return {
            'object_classifications': object_classifications,
            'relation_classifications': relation_classifications,
            'importance_scores': importance_scores,
            'structural_embeddings': attended_output,
            'attention_weights': attention_weights
        }

class GNNLayer(nn.Module):
    """그래프 신경망 레이어"""
    
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear_self = nn.Linear(input_dim, hidden_dim)
        self.linear_neighbor = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, node_features, edge_indices=None, edge_weights=None):
        """GNN 레이어 순전파"""
        # 자기 자신의 특성
        self_features = self.linear_self(node_features)
        
        if edge_indices is not None:
            # 이웃 노드들의 특성 집계
            neighbor_features = self._aggregate_neighbors(
                node_features, edge_indices, edge_weights
            )
            neighbor_features = self.linear_neighbor(neighbor_features)
        else:
            # 엣지 정보가 없으면 전체 평균 사용
            neighbor_features = self.linear_neighbor(
                node_features.mean(dim=1, keepdim=True).expand_as(node_features)
            )
        
        # 결합 및 활성화
        combined = self_features + neighbor_features
        output = self.activation(combined)
        output = self.norm(output)
        
        return output
    
    def _aggregate_neighbors(self, node_features, edge_indices, edge_weights):
        """이웃 노드 특성 집계"""
        batch_size, num_nodes, feature_dim = node_features.shape
        
        # 간단한 평균 집계 (실제로는 더 복잡한 집계 함수 사용 가능)
        aggregated = torch.zeros_like(node_features)
        
        if edge_weights is not None:
            # 가중 평균
            for batch_idx in range(batch_size):
                for edge_idx, (src, tgt) in enumerate(edge_indices):
                    weight = edge_weights[edge_idx]
                    aggregated[batch_idx, tgt] += weight * node_features[batch_idx, src]
        else:
            # 단순 평균
            for batch_idx in range(batch_size):
                for src, tgt in edge_indices:
                    aggregated[batch_idx, tgt] += node_features[batch_idx, src]
        
        return aggregated

class AdvancedRumbaughAnalyzer:
    """고급 Rumbaugh 구조적 분석기"""
    
    def __init__(self):
        """고급 Rumbaugh 분석기 초기화"""
        self.config = ADVANCED_CONFIG['rumbaugh']
        self.device = DEVICE
        self.dtype = TORCH_DTYPE
        
        # 모델 경로 설정
        self.models_dir = os.path.join(MODELS_DIR, 'rumbaugh_models')
        os.makedirs(self.models_dir, exist_ok=True)
        
        # NLP 모델 초기화
        self._initialize_nlp_models()
        
        # 신경망 분석기 초기화
        self._initialize_neural_analyzer()
        
        # 그래프 분석 도구 초기화
        self._initialize_graph_tools()
        
        # 패턴 인식 시스템 초기화
        self._initialize_pattern_recognition()
        
        # 구조적 요소 저장소
        self.structural_objects = {}
        self.structural_relations = {}
        self.state_machines = {}
        self.structural_patterns = {}
        
        # 분석 캐시
        self.analysis_cache = {}
        self.pattern_cache = {}
        
        logger.info("고급 Rumbaugh 구조적 분석기가 초기화되었습니다.")
    
    def _initialize_nlp_models(self):
        """NLP 모델들 초기화"""
        try:
            # Sentence Transformer
            from sentence_transformer_singleton import get_sentence_transformer
            
            self.embedding_model = get_sentence_transformer(
                'paraphrase-multilingual-mpnet-base-v2',
                device=str(self.device),
                cache_folder=self.models_dir
            )
            
            # spaCy 모델 (영어) - subprocess로 처리
            try:
                # spacy 모델 로딩 테스트
                test_result = run_spacy_subprocess('process_text_nlp', {
                    'texts': ['test'],
                    'model_name': 'en_core_web_sm',
                    'operations': ['tokenize']
                })
                if test_result['status'] == 'success':
                    logger.info("spaCy 모델 (subprocess) 초기화 완료")
                    self.nlp_en = "subprocess"  # subprocess 사용 표시
                else:
                    logger.warning("spaCy subprocess 초기화 실패. 기본 처리를 사용합니다.")
                    self.nlp_en = None
            except Exception as e:
                logger.warning(f"spaCy subprocess 초기화 실패: {e}. 기본 처리를 사용합니다.")
                self.nlp_en = None
            
            # 한국어 처리를 위한 간단한 토크나이저
            self.korean_patterns = {
                'actor_patterns': [r'([가-힣]+)이', r'([가-힣]+)가', r'([가-힣]+)은', r'([가-힣]+)는'],
                'action_patterns': [r'([가-힣]+)하다', r'([가-힣]+)한다', r'([가-힣]+)했다'],
                'resource_patterns': [r'([가-힣]+)을', r'([가-힣]+)를', r'([가-힣]+)의'],
                'goal_patterns': [r'([가-힣]+)하기위해', r'([가-힣]+)하려고', r'목표는\s*([가-힣]+)']
            }
            
            logger.info("NLP 모델들이 초기화되었습니다.")
            
        except Exception as e:
            logger.error(f"NLP 모델 초기화 실패: {e}")
            self.embedding_model = None
            self.nlp_en = None
    
    def _initialize_neural_analyzer(self):
        """신경망 분석기 초기화"""
        try:
            embedding_dim = 768  # Sentence Transformer 차원
            
            self.neural_analyzer = AdvancedNeuralStructureAnalyzer(
                embedding_dim=embedding_dim,
                hidden_dim=self.config.get('neural_hidden_dim', 512),
                num_object_types=len(ObjectType),
                num_relation_types=len(RelationType)
            ).to(self.device)
            
            # 옵티마이저
            self.neural_optimizer = torch.optim.Adam(
                self.neural_analyzer.parameters(),
                lr=self.config.get('learning_rate', 0.001)
            )
            
            # 손실 함수들
            self.classification_loss = nn.CrossEntropyLoss()
            self.importance_loss = nn.MSELoss()
            
            # 체크포인트 로드
            checkpoint_path = os.path.join(self.models_dir, 'neural_analyzer.pt')
            if os.path.exists(checkpoint_path):
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                    self.neural_analyzer.load_state_dict(checkpoint['model_state'])
                    self.neural_optimizer.load_state_dict(checkpoint['optimizer_state'])
                    logger.info("신경망 분석기 체크포인트를 로드했습니다.")
                except Exception as e:
                    logger.warning(f"체크포인트 로드 실패: {e}")
            
            logger.info("신경망 분석기가 초기화되었습니다.")
            
        except Exception as e:
            logger.error(f"신경망 분석기 초기화 실패: {e}")
            self.neural_analyzer = None
    
    def _initialize_graph_tools(self):
        """그래프 분석 도구 초기화"""
        # 그래프 메트릭 계산기들
        self.graph_metrics = {
            'centrality_measures': ['betweenness', 'closeness', 'degree', 'eigenvector'],
            'clustering_algorithms': ['leiden', 'louvain', 'spectral'],
            'path_algorithms': ['shortest_path', 'all_pairs_shortest_path'],
            'flow_algorithms': ['max_flow', 'min_cost_flow']
        }
        
        # 복잡도 측정기
        self.complexity_calculator = StructuralComplexityCalculator()
        
        logger.info("그래프 분석 도구가 초기화되었습니다.")
    
    def _initialize_pattern_recognition(self):
        """패턴 인식 시스템 초기화"""
        # 구조적 패턴 템플릿
        self.pattern_templates = {
            'hierarchical': {
                'description': '계층적 구조 패턴',
                'required_relations': [RelationType.INHERITANCE, RelationType.COMPOSITION],
                'min_objects': 3
            },
            'collaborative': {
                'description': '협력적 구조 패턴',
                'required_relations': [RelationType.COOPERATION, RelationType.ASSOCIATION],
                'min_objects': 2
            },
            'conflictual': {
                'description': '갈등적 구조 패턴',
                'required_relations': [RelationType.CONFLICT],
                'min_objects': 2
            },
            'causal_chain': {
                'description': '인과 사슬 패턴',
                'required_relations': [RelationType.CAUSALITY],
                'min_objects': 3
            }
        }
        
        # 패턴 분류기
        self.pattern_classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        
        logger.info("패턴 인식 시스템이 초기화되었습니다.")
    
    async def analyze_structure(self, text: str, metadata: Dict[str, Any] = None) -> AdvancedStructuralResult:
        """
        고급 구조적 분석 수행
        
        Args:
            text: 분석할 텍스트
            metadata: 추가 메타데이터
            
        Returns:
            고급 구조적 분석 결과
        """
        try:
            start_time = time.time()
            
            # 캐시 확인
            cache_key = self._generate_cache_key(text, metadata)
            if cache_key in self.analysis_cache:
                return self.analysis_cache[cache_key]
            
            logger.info("구조적 분석을 시작합니다...")
            
            # 1단계: 구조적 객체 추출
            structural_objects = await self._extract_structural_objects(text)
            
            # 2단계: 관계 분석
            structural_relations = await self._analyze_structural_relations(
                text, structural_objects
            )
            
            # 3단계: 상태 기계 구성
            state_machines = await self._build_state_machines(
                structural_objects, structural_relations
            )
            
            # 4단계: 신경망 기반 고급 분석
            neural_analysis = await self._perform_neural_analysis(
                structural_objects, structural_relations
            )
            
            # 5단계: 그래프 구조 분석
            graph_analysis = await self._analyze_graph_structure(
                structural_objects, structural_relations
            )
            
            # 6단계: 패턴 인식
            structural_patterns = await self._recognize_structural_patterns(
                structural_objects, structural_relations
            )
            
            # 7단계: 복잡도 계산
            complexity_metrics = await self._calculate_structural_complexity(
                structural_objects, structural_relations, graph_analysis
            )
            
            # 8단계: 동적 상호작용 분석
            dynamic_interactions = await self._analyze_dynamic_interactions(
                state_machines, structural_relations
            )
            
            # 결과 구성
            result = AdvancedStructuralResult(
                text=text,
                structural_objects=structural_objects,
                structural_relations=structural_relations,
                state_machines=state_machines,
                neural_analysis=neural_analysis,
                graph_analysis=graph_analysis,
                structural_patterns=structural_patterns,
                complexity_metrics=complexity_metrics,
                dynamic_interactions=dynamic_interactions,
                processing_time=time.time() - start_time,
                confidence_score=self._calculate_overall_confidence(
                    structural_objects, structural_relations, neural_analysis
                ),
                metadata=metadata or {}
            )
            
            # 캐시 저장
            self.analysis_cache[cache_key] = result
            
            logger.info(f"구조적 분석 완료: {len(structural_objects)}개 객체, {len(structural_relations)}개 관계")
            return result
            
        except Exception as e:
            logger.error(f"구조적 분석 실패: {e}")
            return self._get_error_result(text, str(e))
    
    async def _extract_structural_objects(self, text: str) -> List[StructuralObject]:
        """구조적 객체 추출"""
        objects = []
        
        try:
            # 임베딩 생성
            if self.embedding_model:
                text_embedding = self.embedding_model.encode(text)
            else:
                text_embedding = np.zeros(768)
            
            # 영어 NLP 처리
            if self.nlp_en:
                doc = self.nlp_en(text)
                
                # 개체명 인식
                for ent in doc.ents:
                    if ent.label_ in ['PERSON', 'ORG']:
                        obj = StructuralObject(
                            object_id=str(uuid.uuid4()),
                            object_type=ObjectType.ACTOR,
                            name=ent.text,
                            description=f"{ent.label_}: {ent.text}",
                            attributes={'entity_type': ent.label_, 'confidence': 0.8},
                            extracted_from=text
                        )
                        objects.append(obj)
                
                # 동사에서 행동 추출
                for token in doc:
                    if token.pos_ == 'VERB' and not token.is_stop:
                        obj = StructuralObject(
                            object_id=str(uuid.uuid4()),
                            object_type=ObjectType.ACTION,
                            name=token.lemma_,
                            description=f"Action: {token.text}",
                            attributes={'pos': token.pos_, 'confidence': 0.6},
                            extracted_from=text
                        )
                        objects.append(obj)
                
                # 명사에서 자원/목표 추출
                for token in doc:
                    if token.pos_ == 'NOUN' and not token.is_stop:
                        obj_type = ObjectType.RESOURCE
                        if any(goal_word in token.text.lower() for goal_word in ['goal', 'objective', 'target']):
                            obj_type = ObjectType.GOAL
                        
                        obj = StructuralObject(
                            object_id=str(uuid.uuid4()),
                            object_type=obj_type,
                            name=token.text,
                            description=f"{obj_type.value}: {token.text}",
                            attributes={'pos': token.pos_, 'confidence': 0.5},
                            extracted_from=text
                        )
                        objects.append(obj)
            
            # 한국어 패턴 기반 추출
            for obj_type, patterns in self.korean_patterns.items():
                for pattern in patterns:
                    matches = re.findall(pattern, text)
                    for match in matches:
                        if len(match) > 1:  # 의미있는 길이
                            object_type_map = {
                                'actor_patterns': ObjectType.ACTOR,
                                'action_patterns': ObjectType.ACTION,
                                'resource_patterns': ObjectType.RESOURCE,
                                'goal_patterns': ObjectType.GOAL
                            }
                            
                            obj = StructuralObject(
                                object_id=str(uuid.uuid4()),
                                object_type=object_type_map[obj_type],
                                name=match,
                                description=f"Korean {obj_type}: {match}",
                                attributes={'pattern_based': True, 'confidence': 0.7},
                                extracted_from=text
                            )
                            objects.append(obj)
            
            # 키워드 기반 제약사항 추출
            constraint_keywords = ['제약', '한계', '문제', 'constraint', 'limitation', 'restriction']
            for keyword in constraint_keywords:
                if keyword in text.lower():
                    obj = StructuralObject(
                        object_id=str(uuid.uuid4()),
                        object_type=ObjectType.CONSTRAINT,
                        name=keyword,
                        description=f"Constraint keyword: {keyword}",
                        attributes={'keyword_based': True, 'confidence': 0.6},
                        extracted_from=text
                    )
                    objects.append(obj)
            
            # 신경망 기반 중요도 계산
            if self.neural_analyzer and objects:
                await self._calculate_neural_importance(objects, text_embedding)
            
            # 중복 제거 및 정제
            objects = self._deduplicate_objects(objects)
            
            return objects
            
        except Exception as e:
            logger.error(f"구조적 객체 추출 실패: {e}")
            return []
    
    async def _analyze_structural_relations(self, text: str, 
                                          objects: List[StructuralObject]) -> List[StructuralRelation]:
        """구조적 관계 분석"""
        relations = []
        
        try:
            if len(objects) < 2:
                return relations
            
            # 객체 쌍별 관계 분석
            for i, obj1 in enumerate(objects):
                for j, obj2 in enumerate(objects[i + 1:], i + 1):
                    relation = await self._analyze_object_pair_relation(text, obj1, obj2)
                    if relation:
                        relations.append(relation)
            
            # 신경망 기반 관계 분류
            if self.neural_analyzer and relations:
                await self._classify_relations_neural(relations, objects)
            
            # 텍스트 패턴 기반 관계 강화
            relations = await self._enhance_relations_with_patterns(text, relations)
            
            # 관계 필터링 (신뢰도 기반)
            relations = [rel for rel in relations if rel.confidence > 0.3]
            
            return relations
            
        except Exception as e:
            logger.error(f"구조적 관계 분석 실패: {e}")
            return []
    
    async def _analyze_object_pair_relation(self, text: str, 
                                          obj1: StructuralObject, 
                                          obj2: StructuralObject) -> Optional[StructuralRelation]:
        """객체 쌍 간의 관계 분석"""
        try:
            # 기본 관계 유형 결정
            relation_type = self._determine_basic_relation_type(obj1, obj2)
            
            # 텍스트에서 관계 강도 계산
            strength = self._calculate_relation_strength(text, obj1.name, obj2.name)
            
            # 관계 신뢰도 계산
            confidence = (obj1.confidence + obj2.confidence) / 2 * strength
            
            if confidence > 0.3:  # 최소 신뢰도
                relation = StructuralRelation(
                    relation_id=str(uuid.uuid4()),
                    source_object=obj1.object_id,
                    target_object=obj2.object_id,
                    relation_type=relation_type,
                    strength=strength,
                    confidence=confidence,
                    properties={
                        'source_name': obj1.name,
                        'target_name': obj2.name,
                        'extracted_from': text[:100]
                    }
                )
                return relation
            
            return None
            
        except Exception as e:
            logger.error(f"객체 쌍 관계 분석 실패: {e}")
            return None
    
    def _determine_basic_relation_type(self, obj1: StructuralObject, 
                                     obj2: StructuralObject) -> RelationType:
        """기본 관계 유형 결정"""
        # 객체 유형에 따른 기본 관계 매핑
        type_relations = {
            (ObjectType.ACTOR, ObjectType.ACTION): RelationType.ASSOCIATION,
            (ObjectType.ACTOR, ObjectType.GOAL): RelationType.DEPENDENCY,
            (ObjectType.ACTOR, ObjectType.RESOURCE): RelationType.ASSOCIATION,
            (ObjectType.ACTION, ObjectType.GOAL): RelationType.CAUSALITY,
            (ObjectType.ACTION, ObjectType.RESOURCE): RelationType.DEPENDENCY,
            (ObjectType.RESOURCE, ObjectType.GOAL): RelationType.DEPENDENCY,
            (ObjectType.CONSTRAINT, ObjectType.ACTION): RelationType.CONFLICT,
            (ObjectType.CONSTRAINT, ObjectType.GOAL): RelationType.CONFLICT,
        }
        
        # 순서 상관없이 매핑 확인
        key1 = (obj1.object_type, obj2.object_type)
        key2 = (obj2.object_type, obj1.object_type)
        
        return type_relations.get(key1, type_relations.get(key2, RelationType.ASSOCIATION))
    
    def _calculate_relation_strength(self, text: str, name1: str, name2: str) -> float:
        """관계 강도 계산"""
        # 텍스트에서 두 이름의 근접성 계산
        text_lower = text.lower()
        name1_lower = name1.lower()
        name2_lower = name2.lower()
        
        if name1_lower not in text_lower or name2_lower not in text_lower:
            return 0.1
        
        # 이름들 사이의 거리 계산
        pos1 = text_lower.find(name1_lower)
        pos2 = text_lower.find(name2_lower)
        distance = abs(pos1 - pos2)
        
        # 거리가 가까울수록 강한 관계
        max_distance = len(text)
        strength = 1.0 - (distance / max_distance)
        
        # 관계 키워드 존재 여부로 강도 조정
        relation_keywords = [
            '관련', '연결', '의존', '영향', '결과', '원인',
            'related', 'connected', 'depends', 'affects', 'causes'
        ]
        
        keyword_bonus = 0.0
        for keyword in relation_keywords:
            if keyword in text_lower:
                keyword_bonus += 0.2
        
        return min(1.0, strength + keyword_bonus)
    
    async def _build_state_machines(self, objects: List[StructuralObject], 
                                   relations: List[StructuralRelation]) -> List[ObjectStateMachine]:
        """상태 기계 구성"""
        state_machines = []
        
        try:
            # 동적 객체들 (ACTOR, ACTION)에 대해 상태 기계 생성
            dynamic_objects = [obj for obj in objects if obj.object_type in [ObjectType.ACTOR, ObjectType.ACTION]]
            
            for obj in dynamic_objects:
                # 기본 상태들 정의
                states = [StateType.INITIAL, StateType.ACTIVE, StateType.TERMINAL]
                
                # 관련 관계들을 기반으로 전이 정의
                transitions = []
                related_relations = [rel for rel in relations 
                                   if rel.source_object == obj.object_id or rel.target_object == obj.object_id]
                
                for rel in related_relations:
                    if rel.relation_type == RelationType.CAUSALITY:
                        transitions.append({
                            'from_state': StateType.INITIAL,
                            'to_state': StateType.ACTIVE,
                            'trigger': f"causal_trigger_{rel.relation_id}",
                            'probability': rel.strength
                        })
                    elif rel.relation_type == RelationType.CONFLICT:
                        transitions.append({
                            'from_state': StateType.ACTIVE,
                            'to_state': StateType.ERROR,
                            'trigger': f"conflict_trigger_{rel.relation_id}",
                            'probability': rel.strength * 0.8
                        })
                
                # 기본 전이 추가
                if not transitions:
                    transitions = [
                        {
                            'from_state': StateType.INITIAL,
                            'to_state': StateType.ACTIVE,
                            'trigger': 'default_activation',
                            'probability': 0.7
                        },
                        {
                            'from_state': StateType.ACTIVE,
                            'to_state': StateType.TERMINAL,
                            'trigger': 'completion',
                            'probability': 0.8
                        }
                    ]
                
                # 상태 확률 계산
                state_probabilities = self._calculate_state_probabilities(states, transitions)
                
                # 전이 매트릭스 구성
                transition_matrix = self._build_transition_matrix(states, transitions)
                
                state_machine = ObjectStateMachine(
                    object_id=obj.object_id,
                    states=states,
                    transitions=transitions,
                    current_state=StateType.INITIAL,
                    state_probabilities=state_probabilities,
                    transition_matrix=transition_matrix
                )
                
                state_machines.append(state_machine)
            
            return state_machines
            
        except Exception as e:
            logger.error(f"상태 기계 구성 실패: {e}")
            return []
    
    def _calculate_state_probabilities(self, states: List[StateType], 
                                     transitions: List[Dict[str, Any]]) -> Dict[StateType, float]:
        """상태 확률 계산"""
        probabilities = {state: 1.0 / len(states) for state in states}
        
        # 전이 확률을 기반으로 정상 상태 확률 계산
        for transition in transitions:
            from_state = transition['from_state']
            to_state = transition['to_state']
            prob = transition['probability']
            
            # 간단한 확률 업데이트
            probabilities[to_state] += prob * 0.1
            probabilities[from_state] -= prob * 0.05
        
        # 정규화
        total = sum(probabilities.values())
        if total > 0:
            for state in probabilities:
                probabilities[state] /= total
        
        return probabilities
    
    def _build_transition_matrix(self, states: List[StateType], 
                               transitions: List[Dict[str, Any]]) -> np.ndarray:
        """전이 매트릭스 구성"""
        n_states = len(states)
        matrix = np.zeros((n_states, n_states))
        
        state_to_idx = {state: i for i, state in enumerate(states)}
        
        for transition in transitions:
            from_idx = state_to_idx[transition['from_state']]
            to_idx = state_to_idx[transition['to_state']]
            prob = transition['probability']
            
            matrix[from_idx, to_idx] = prob
        
        # 행 정규화 (각 상태에서 나가는 전이 확률의 합 = 1)
        for i in range(n_states):
            row_sum = matrix[i].sum()
            if row_sum > 0:
                matrix[i] /= row_sum
            else:
                matrix[i, i] = 1.0  # 자기 자신으로의 전이
        
        return matrix
    
    async def _perform_neural_analysis(self, objects: List[StructuralObject], 
                                     relations: List[StructuralRelation]) -> Dict[str, Any]:
        """신경망 기반 고급 분석"""
        if self.neural_analyzer is None or not objects:
            return {'error': '신경망 분석기가 사용 불가능합니다.'}
        
        try:
            # 객체 임베딩 생성
            object_embeddings = []
            for obj in objects:
                if self.embedding_model:
                    embedding = self.embedding_model.encode(obj.description)
                else:
                    embedding = np.random.normal(0, 1, 768)
                object_embeddings.append(embedding)
            
            if not object_embeddings:
                return {'error': '임베딩 생성 실패'}
            
            # 텐서 변환
            embeddings_tensor = torch.tensor(
                np.array(object_embeddings),
                dtype=self.dtype,
                device=self.device
            ).unsqueeze(0)  # 배치 차원 추가
            
            # 엣지 인덱스 구성
            edge_indices = []
            edge_weights = []
            
            for rel in relations:
                # 객체 ID를 인덱스로 변환
                source_idx = None
                target_idx = None
                
                for i, obj in enumerate(objects):
                    if obj.object_id == rel.source_object:
                        source_idx = i
                    if obj.object_id == rel.target_object:
                        target_idx = i
                
                if source_idx is not None and target_idx is not None:
                    edge_indices.append([source_idx, target_idx])
                    edge_weights.append(rel.strength)
            
            # 신경망 순전파
            with torch.no_grad():
                outputs = self.neural_analyzer(
                    embeddings_tensor,
                    edge_indices=edge_indices if edge_indices else None,
                    edge_weights=torch.tensor(edge_weights) if edge_weights else None
                )
            
            # 결과 처리
            neural_analysis = {
                'object_classifications': outputs['object_classifications'].squeeze().cpu().numpy().tolist(),
                'importance_scores': outputs['importance_scores'].squeeze().cpu().numpy().tolist(),
                'structural_embeddings': outputs['structural_embeddings'].squeeze().cpu().numpy().tolist(),
                'attention_weights': outputs['attention_weights'].squeeze().cpu().numpy().tolist(),
                'network_connectivity': len(edge_indices) / max(1, len(objects) * (len(objects) - 1) / 2),
                'structural_coherence': self._calculate_structural_coherence(outputs)
            }
            
            return neural_analysis
            
        except Exception as e:
            logger.error(f"신경망 분석 실패: {e}")
            return {'error': str(e)}
    
    def _calculate_structural_coherence(self, neural_outputs: Dict[str, torch.Tensor]) -> float:
        """구조적 일관성 계산"""
        try:
            # 어텐션 가중치의 엔트로피 계산
            attention_weights = neural_outputs['attention_weights']
            entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8))
            
            # 일관성 = 1 - 정규화된 엔트로피
            max_entropy = torch.log(torch.tensor(attention_weights.size(-1), dtype=torch.float))
            coherence = 1.0 - (entropy / max_entropy).item()
            
            return max(0.0, min(1.0, coherence))
            
        except Exception as e:
            logger.error(f"구조적 일관성 계산 실패: {e}")
            return 0.5
    
    async def _analyze_graph_structure(self, objects: List[StructuralObject], 
                                     relations: List[StructuralRelation]) -> Dict[str, Any]:
        """그래프 구조 분석"""
        try:
            # NetworkX 그래프 생성
            G = nx.DiGraph()
            
            # 노드 추가
            for obj in objects:
                G.add_node(obj.object_id, 
                          name=obj.name,
                          type=obj.object_type.value,
                          importance=obj.importance)
            
            # 엣지 추가
            for rel in relations:
                G.add_edge(rel.source_object, rel.target_object,
                          type=rel.relation_type.value,
                          strength=rel.strength,
                          confidence=rel.confidence)
            
            # 그래프 메트릭 계산
            graph_metrics = {}
            
            if G.number_of_nodes() > 0:
                # 중심성 측정
                try:
                    graph_metrics['betweenness_centrality'] = nx.betweenness_centrality(G)
                    graph_metrics['closeness_centrality'] = nx.closeness_centrality(G)
                    graph_metrics['degree_centrality'] = nx.degree_centrality(G)
                except:
                    graph_metrics['centrality_error'] = 'Centrality calculation failed'
                
                # 클러스터링 계수
                try:
                    graph_metrics['clustering_coefficient'] = nx.average_clustering(G.to_undirected())
                except:
                    graph_metrics['clustering_coefficient'] = 0.0
                
                # 밀도
                graph_metrics['density'] = nx.density(G)
                
                # 강연결성분
                try:
                    graph_metrics['strongly_connected_components'] = len(list(nx.strongly_connected_components(G)))
                except:
                    graph_metrics['strongly_connected_components'] = 1
                
                # 경로 분석
                try:
                    if nx.is_weakly_connected(G):
                        graph_metrics['average_shortest_path'] = nx.average_shortest_path_length(G.to_undirected())
                    else:
                        graph_metrics['average_shortest_path'] = float('inf')
                except:
                    graph_metrics['average_shortest_path'] = float('inf')
            
            # 구조적 특성
            structural_properties = {
                'node_count': G.number_of_nodes(),
                'edge_count': G.number_of_edges(),
                'is_dag': nx.is_directed_acyclic_graph(G),
                'is_connected': nx.is_weakly_connected(G) if G.number_of_nodes() > 0 else False,
                'max_in_degree': max([d for n, d in G.in_degree()]) if G.number_of_nodes() > 0 else 0,
                'max_out_degree': max([d for n, d in G.out_degree()]) if G.number_of_nodes() > 0 else 0
            }
            
            return {
                'graph_metrics': graph_metrics,
                'structural_properties': structural_properties,
                'graph_object': G  # 추가 분석을 위해 그래프 객체 포함
            }
            
        except Exception as e:
            logger.error(f"그래프 구조 분석 실패: {e}")
            return {'error': str(e)}
    
    async def _recognize_structural_patterns(self, objects: List[StructuralObject], 
                                           relations: List[StructuralRelation]) -> List[StructuralPattern]:
        """구조적 패턴 인식"""
        patterns = []
        
        try:
            # 템플릿 기반 패턴 매칭
            for pattern_name, template in self.pattern_templates.items():
                pattern = await self._match_pattern_template(
                    pattern_name, template, objects, relations
                )
                if pattern:
                    patterns.append(pattern)
            
            # 빈도 기반 패턴 발견
            frequency_patterns = await self._discover_frequency_patterns(objects, relations)
            patterns.extend(frequency_patterns)
            
            # 구조적 모티프 감지
            motif_patterns = await self._detect_structural_motifs(objects, relations)
            patterns.extend(motif_patterns)
            
            return patterns
            
        except Exception as e:
            logger.error(f"구조적 패턴 인식 실패: {e}")
            return []
    
    async def _match_pattern_template(self, pattern_name: str, template: Dict[str, Any],
                                    objects: List[StructuralObject], 
                                    relations: List[StructuralRelation]) -> Optional[StructuralPattern]:
        """패턴 템플릿 매칭"""
        try:
            # 최소 객체 수 확인
            if len(objects) < template['min_objects']:
                return None
            
            # 필요한 관계 유형 확인
            relation_types = [rel.relation_type for rel in relations]
            required_relations = template['required_relations']
            
            if not any(req_rel in relation_types for req_rel in required_relations):
                return None
            
            # 패턴 매칭 성공
            involved_objects = [obj.object_id for obj in objects]
            involved_relations = [rel.relation_id for rel in relations 
                                if rel.relation_type in required_relations]
            
            # 의미도 점수 계산
            significance = len(involved_relations) / max(1, len(relations))
            
            pattern = StructuralPattern(
                pattern_id=str(uuid.uuid4()),
                pattern_type=pattern_name,
                objects_involved=involved_objects,
                relations_involved=involved_relations,
                pattern_frequency=1,
                significance_score=significance,
                description=template['description']
            )
            
            return pattern
            
        except Exception as e:
            logger.error(f"패턴 템플릿 매칭 실패: {e}")
            return None
    
    async def _discover_frequency_patterns(self, objects: List[StructuralObject], 
                                         relations: List[StructuralRelation]) -> List[StructuralPattern]:
        """빈도 기반 패턴 발견"""
        patterns = []
        
        try:
            # 객체 유형 빈도 분석
            object_type_counts = Counter([obj.object_type for obj in objects])
            
            # 관계 유형 빈도 분석
            relation_type_counts = Counter([rel.relation_type for rel in relations])
            
            # 고빈도 패턴 식별
            for obj_type, count in object_type_counts.items():
                if count >= 3:  # 최소 3개 이상
                    pattern = StructuralPattern(
                        pattern_id=str(uuid.uuid4()),
                        pattern_type=f"frequent_{obj_type.value}",
                        objects_involved=[obj.object_id for obj in objects if obj.object_type == obj_type],
                        relations_involved=[],
                        pattern_frequency=count,
                        significance_score=count / len(objects),
                        description=f"High frequency of {obj_type.value} objects"
                    )
                    patterns.append(pattern)
            
            for rel_type, count in relation_type_counts.items():
                if count >= 2:  # 최소 2개 이상
                    pattern = StructuralPattern(
                        pattern_id=str(uuid.uuid4()),
                        pattern_type=f"frequent_{rel_type.value}",
                        objects_involved=[],
                        relations_involved=[rel.relation_id for rel in relations if rel.relation_type == rel_type],
                        pattern_frequency=count,
                        significance_score=count / len(relations) if relations else 0,
                        description=f"High frequency of {rel_type.value} relations"
                    )
                    patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"빈도 기반 패턴 발견 실패: {e}")
            return []
    
    async def _detect_structural_motifs(self, objects: List[StructuralObject], 
                                      relations: List[StructuralRelation]) -> List[StructuralPattern]:
        """구조적 모티프 감지"""
        patterns = []
        
        try:
            # 삼각형 모티프 (3개 객체가 서로 연결)
            triangle_motifs = self._find_triangle_motifs(objects, relations)
            patterns.extend(triangle_motifs)
            
            # 스타 모티프 (중심 객체와 여러 주변 객체)
            star_motifs = self._find_star_motifs(objects, relations)
            patterns.extend(star_motifs)
            
            # 체인 모티프 (연속적 연결)
            chain_motifs = self._find_chain_motifs(objects, relations)
            patterns.extend(chain_motifs)
            
            return patterns
            
        except Exception as e:
            logger.error(f"구조적 모티프 감지 실패: {e}")
            return []
    
    def _find_triangle_motifs(self, objects: List[StructuralObject], 
                            relations: List[StructuralRelation]) -> List[StructuralPattern]:
        """삼각형 모티프 찾기"""
        patterns = []
        
        # 인접 리스트 구성
        adj_list = defaultdict(set)
        for rel in relations:
            adj_list[rel.source_object].add(rel.target_object)
            adj_list[rel.target_object].add(rel.source_object)  # 무방향으로 처리
        
        # 삼각형 찾기
        object_ids = [obj.object_id for obj in objects]
        
        for i, obj1 in enumerate(object_ids):
            for j, obj2 in enumerate(object_ids[i+1:], i+1):
                if obj2 in adj_list[obj1]:
                    for k, obj3 in enumerate(object_ids[j+1:], j+1):
                        if obj3 in adj_list[obj1] and obj3 in adj_list[obj2]:
                            # 삼각형 발견
                            pattern = StructuralPattern(
                                pattern_id=str(uuid.uuid4()),
                                pattern_type="triangle_motif",
                                objects_involved=[obj1, obj2, obj3],
                                relations_involved=[],  # 관련 관계들을 추가할 수 있음
                                pattern_frequency=1,
                                significance_score=0.8,
                                description="Triangular structural motif"
                            )
                            patterns.append(pattern)
        
        return patterns
    
    def _find_star_motifs(self, objects: List[StructuralObject], 
                        relations: List[StructuralRelation]) -> List[StructuralPattern]:
        """스타 모티프 찾기"""
        patterns = []
        
        # 각 객체의 연결 수 계산
        connection_counts = defaultdict(int)
        for rel in relations:
            connection_counts[rel.source_object] += 1
            connection_counts[rel.target_object] += 1
        
        # 높은 연결성을 가진 중심 객체 찾기
        for obj_id, count in connection_counts.items():
            if count >= 3:  # 최소 3개 연결
                # 연결된 객체들 찾기
                connected_objects = set()
                for rel in relations:
                    if rel.source_object == obj_id:
                        connected_objects.add(rel.target_object)
                    elif rel.target_object == obj_id:
                        connected_objects.add(rel.source_object)
                
                if len(connected_objects) >= 3:
                    pattern = StructuralPattern(
                        pattern_id=str(uuid.uuid4()),
                        pattern_type="star_motif",
                        objects_involved=[obj_id] + list(connected_objects),
                        relations_involved=[],
                        pattern_frequency=1,
                        significance_score=count / len(objects),
                        description=f"Star motif with center {obj_id}"
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _find_chain_motifs(self, objects: List[StructuralObject], 
                         relations: List[StructuralRelation]) -> List[StructuralPattern]:
        """체인 모티프 찾기"""
        patterns = []
        
        # 인과관계 체인 찾기
        causal_relations = [rel for rel in relations if rel.relation_type == RelationType.CAUSALITY]
        
        if len(causal_relations) >= 2:
            # 연속적 인과관계 체인 구성
            chains = []
            
            for rel1 in causal_relations:
                for rel2 in causal_relations:
                    if rel1.target_object == rel2.source_object:
                        # 체인 발견 (A -> B -> C)
                        chain = [rel1.source_object, rel1.target_object, rel2.target_object]
                        chains.append(chain)
            
            for chain in chains:
                pattern = StructuralPattern(
                    pattern_id=str(uuid.uuid4()),
                    pattern_type="causal_chain_motif",
                    objects_involved=chain,
                    relations_involved=[],
                    pattern_frequency=1,
                    significance_score=0.7,
                    description="Causal chain motif"
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _calculate_structural_complexity(self, objects: List[StructuralObject], 
                                             relations: List[StructuralRelation],
                                             graph_analysis: Dict[str, Any]) -> Dict[str, float]:
        """구조적 복잡도 계산"""
        try:
            complexity_metrics = {}
            
            # 기본 복잡도 메트릭
            n_objects = len(objects)
            n_relations = len(relations)
            
            # 노드 복잡도
            complexity_metrics['node_complexity'] = n_objects / 10.0  # 정규화
            
            # 엣지 복잡도
            max_possible_edges = n_objects * (n_objects - 1) / 2
            complexity_metrics['edge_complexity'] = n_relations / max(1, max_possible_edges)
            
            # 구조적 다양성
            object_types = set([obj.object_type for obj in objects])
            relation_types = set([rel.relation_type for rel in relations])
            complexity_metrics['type_diversity'] = (len(object_types) + len(relation_types)) / 16.0  # 총 유형 수
            
            # 그래프 기반 복잡도
            if 'graph_metrics' in graph_analysis:
                metrics = graph_analysis['graph_metrics']
                
                # 중심성 복잡도
                if 'betweenness_centrality' in metrics:
                    centrality_values = list(metrics['betweenness_centrality'].values())
                    complexity_metrics['centrality_complexity'] = np.std(centrality_values) if centrality_values else 0
                
                # 클러스터링 복잡도
                clustering_coeff = metrics.get('clustering_coefficient', 0)
                complexity_metrics['clustering_complexity'] = clustering_coeff
                
                # 경로 복잡도
                avg_path = metrics.get('average_shortest_path', float('inf'))
                if avg_path != float('inf'):
                    complexity_metrics['path_complexity'] = min(1.0, avg_path / n_objects)
                else:
                    complexity_metrics['path_complexity'] = 1.0
            
            # 전체 복잡도 점수
            complexity_metrics['overall_complexity'] = np.mean([
                complexity_metrics.get('node_complexity', 0),
                complexity_metrics.get('edge_complexity', 0),
                complexity_metrics.get('type_diversity', 0),
                complexity_metrics.get('centrality_complexity', 0),
                complexity_metrics.get('clustering_complexity', 0),
                complexity_metrics.get('path_complexity', 0)
            ])
            
            return complexity_metrics
            
        except Exception as e:
            logger.error(f"구조적 복잡도 계산 실패: {e}")
            return {'overall_complexity': 0.5}
    
    async def _analyze_dynamic_interactions(self, state_machines: List[ObjectStateMachine], 
                                          relations: List[StructuralRelation]) -> List[DynamicInteraction]:
        """동적 상호작용 분석"""
        interactions = []
        
        try:
            # 상태 기계 간의 상호작용 분석
            for i, sm1 in enumerate(state_machines):
                for j, sm2 in enumerate(state_machines[i+1:], i+1):
                    # 두 상태 기계 간의 관계 찾기
                    connecting_relations = [
                        rel for rel in relations
                        if (rel.source_object == sm1.object_id and rel.target_object == sm2.object_id) or
                           (rel.source_object == sm2.object_id and rel.target_object == sm1.object_id)
                    ]
                    
                    if connecting_relations:
                        for rel in connecting_relations:
                            interaction = DynamicInteraction(
                                interaction_id=str(uuid.uuid4()),
                                source_object=sm1.object_id,
                                target_object=sm2.object_id,
                                interaction_type=rel.relation_type.value,
                                trigger_conditions=[f"state_change_{sm1.object_id}", f"state_change_{sm2.object_id}"],
                                expected_outcomes=[f"synchronized_behavior"],
                                interaction_strength=rel.strength,
                                temporal_dynamics={
                                    'duration': 'variable',
                                    'frequency': 'event_driven',
                                    'synchronization': rel.strength > 0.7
                                }
                            )
                            interactions.append(interaction)
            
            return interactions
            
        except Exception as e:
            logger.error(f"동적 상호작용 분석 실패: {e}")
            return []
    
    # 헬퍼 메서드들
    async def _calculate_neural_importance(self, objects: List[StructuralObject], text_embedding: np.ndarray):
        """신경망 기반 중요도 계산"""
        if self.neural_analyzer is None:
            return
        
        try:
            for obj in objects:
                # 객체 설명의 임베딩 생성
                if self.embedding_model:
                    obj_embedding = self.embedding_model.encode(obj.description)
                else:
                    obj_embedding = np.random.normal(0, 1, 768)
                
                # 신경망으로 중요도 예측
                with torch.no_grad():
                    embedding_tensor = torch.tensor(
                        obj_embedding,
                        dtype=self.dtype,
                        device=self.device
                    ).unsqueeze(0)
                    
                    importance = self.neural_analyzer.importance_predictor(embedding_tensor)
                    obj.importance = float(importance.squeeze())
                    
        except Exception as e:
            logger.error(f"신경망 중요도 계산 실패: {e}")
    
    def _deduplicate_objects(self, objects: List[StructuralObject]) -> List[StructuralObject]:
        """중복 객체 제거"""
        unique_objects = {}
        
        for obj in objects:
            # 이름과 유형으로 키 생성
            key = (obj.name.lower(), obj.object_type)
            
            if key not in unique_objects:
                unique_objects[key] = obj
            else:
                # 기존 객체의 신뢰도가 더 높으면 유지
                if obj.confidence > unique_objects[key].confidence:
                    unique_objects[key] = obj
        
        return list(unique_objects.values())
    
    async def _classify_relations_neural(self, relations: List[StructuralRelation], 
                                       objects: List[StructuralObject]):
        """신경망 기반 관계 분류"""
        if self.neural_analyzer is None or not relations:
            return
        
        try:
            # 관계별로 신경망 분류 수행
            # (실제 구현에서는 배치 처리가 더 효율적)
            pass  # 여기서는 기본 분류를 유지
            
        except Exception as e:
            logger.error(f"신경망 관계 분류 실패: {e}")
    
    async def _enhance_relations_with_patterns(self, text: str, 
                                             relations: List[StructuralRelation]) -> List[StructuralRelation]:
        """패턴 기반 관계 강화"""
        try:
            # 텍스트에서 관계 강화 패턴 검색
            enhancement_patterns = {
                RelationType.CAUSALITY: [r'때문에', r'원인', r'결과', r'because', r'due to', r'results in'],
                RelationType.CONFLICT: [r'갈등', r'반대', r'충돌', r'conflict', r'oppose', r'against'],
                RelationType.COOPERATION: [r'협력', r'함께', r'협업', r'cooperate', r'together', r'collaborate'],
                RelationType.DEPENDENCY: [r'의존', r'필요', r'요구', r'depend', r'need', r'require']
            }
            
            for relation in relations:
                patterns = enhancement_patterns.get(relation.relation_type, [])
                pattern_count = sum(1 for pattern in patterns if pattern in text.lower())
                
                if pattern_count > 0:
                    # 패턴 발견시 관계 강도 증가
                    relation.strength = min(1.0, relation.strength + pattern_count * 0.2)
                    relation.confidence = min(1.0, relation.confidence + pattern_count * 0.1)
            
            return relations
            
        except Exception as e:
            logger.error(f"패턴 기반 관계 강화 실패: {e}")
            return relations
    
    def _calculate_overall_confidence(self, objects: List[StructuralObject], 
                                    relations: List[StructuralRelation],
                                    neural_analysis: Dict[str, Any]) -> float:
        """전체 신뢰도 계산"""
        try:
            # 객체 신뢰도 평균
            obj_confidence = np.mean([obj.confidence for obj in objects]) if objects else 0.0
            
            # 관계 신뢰도 평균
            rel_confidence = np.mean([rel.confidence for rel in relations]) if relations else 0.0
            
            # 신경망 분석 신뢰도
            neural_confidence = 0.5
            if 'structural_coherence' in neural_analysis:
                neural_confidence = neural_analysis['structural_coherence']
            
            # 가중 평균
            overall_confidence = (
                obj_confidence * 0.4 +
                rel_confidence * 0.4 +
                neural_confidence * 0.2
            )
            
            return float(overall_confidence)
            
        except Exception as e:
            logger.error(f"전체 신뢰도 계산 실패: {e}")
            return 0.5
    
    def _generate_cache_key(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """캐시 키 생성"""
        import hashlib
        
        key_components = [text[:200]]  # 텍스트 일부
        if metadata:
            key_components.append(str(sorted(metadata.items())))
        
        return hashlib.md5('|'.join(key_components).encode()).hexdigest()
    
    def _get_error_result(self, text: str, error_msg: str) -> AdvancedStructuralResult:
        """오류 결과 반환"""
        return AdvancedStructuralResult(
            text=text,
            structural_objects=[],
            structural_relations=[],
            state_machines=[],
            neural_analysis={'error': error_msg},
            graph_analysis={'error': error_msg},
            structural_patterns=[],
            complexity_metrics={'overall_complexity': 0.0},
            dynamic_interactions=[],
            processing_time=0.0,
            confidence_score=0.0,
            metadata={'error': error_msg}
        )

class StructuralComplexityCalculator:
    """구조적 복잡도 계산기"""
    
    def __init__(self):
        self.complexity_weights = {
            'node_count': 0.2,
            'edge_count': 0.2,
            'type_diversity': 0.15,
            'centrality_variance': 0.15,
            'clustering': 0.15,
            'path_length': 0.15
        }
    
    def calculate_complexity(self, graph_analysis: Dict[str, Any]) -> float:
        """복잡도 계산"""
        try:
            complexity_scores = {}
            
            # 각 복잡도 요소 계산
            structural_props = graph_analysis.get('structural_properties', {})
            graph_metrics = graph_analysis.get('graph_metrics', {})
            
            # 노드 수 복잡도
            node_count = structural_props.get('node_count', 0)
            complexity_scores['node_count'] = min(1.0, node_count / 20.0)
            
            # 엣지 수 복잡도
            edge_count = structural_props.get('edge_count', 0)
            max_edges = node_count * (node_count - 1) / 2 if node_count > 1 else 1
            complexity_scores['edge_count'] = edge_count / max_edges
            
            # 기타 복잡도 요소들은 기본값 사용
            complexity_scores['type_diversity'] = 0.5
            complexity_scores['centrality_variance'] = 0.5
            complexity_scores['clustering'] = graph_metrics.get('clustering_coefficient', 0.5)
            complexity_scores['path_length'] = 0.5
            
            # 가중 평균 계산
            weighted_complexity = sum(
                score * self.complexity_weights.get(metric, 0)
                for metric, score in complexity_scores.items()
            )
            
            return weighted_complexity
            
        except Exception as e:
            logger.error(f"복잡도 계산 실패: {e}")
            return 0.5

def create_advanced_rumbaugh_analyzer() -> AdvancedRumbaughAnalyzer:
    """고급 Rumbaugh 분석기 생성"""
    return AdvancedRumbaughAnalyzer()

# 테스트 코드
if __name__ == "__main__":
    import asyncio
    
    async def test_advanced_rumbaugh():
        """고급 Rumbaugh 분석기 테스트"""
        analyzer = create_advanced_rumbaugh_analyzer()
        
        test_text = """
        이 상황에서 프로젝트 매니저는 팀원들과 협력하여 새로운 소프트웨어를 개발해야 한다.
        하지만 예산 제약과 시간 압박이라는 문제가 있다. 
        팀의 목표는 고품질 제품을 출시하는 것이지만, 현실적인 제약사항들이 이를 방해한다.
        매니저는 이러한 갈등을 해결하기 위해 우선순위를 정하고 자원을 효율적으로 배분해야 한다.
        """
        
        print("=== 고급 Rumbaugh 구조적 분석 테스트 ===\n")
        
        result = await analyzer.analyze_structure(test_text)
        
        print(f"분석 텍스트: {result.text}")
        print(f"처리 시간: {result.processing_time:.3f}초")
        print(f"전체 신뢰도: {result.confidence_score:.3f}\n")
        
        print("=== 구조적 객체들 ===")
        for obj in result.structural_objects:
            print(f"- {obj.name} ({obj.object_type.value}): 중요도 {obj.importance:.3f}, 신뢰도 {obj.confidence:.3f}")
        
        print(f"\n=== 구조적 관계들 ===")
        for rel in result.structural_relations:
            print(f"- {rel.relation_type.value}: 강도 {rel.strength:.3f}, 신뢰도 {rel.confidence:.3f}")
        
        print(f"\n=== 상태 기계들 ===")
        for sm in result.state_machines:
            print(f"- 객체 {sm.object_id}: {len(sm.states)}개 상태, {len(sm.transitions)}개 전이")
        
        print(f"\n=== 구조적 패턴들 ===")
        for pattern in result.structural_patterns:
            print(f"- {pattern.pattern_type}: {pattern.description}")
        
        print(f"\n=== 복잡도 메트릭 ===")
        for metric, value in result.complexity_metrics.items():
            print(f"- {metric}: {value:.3f}")
        
        if result.neural_analysis and 'error' not in result.neural_analysis:
            print(f"\n=== 신경망 분석 ===")
            print(f"- 네트워크 연결성: {result.neural_analysis.get('network_connectivity', 0):.3f}")
            print(f"- 구조적 일관성: {result.neural_analysis.get('structural_coherence', 0):.3f}")
        
        print(f"\n테스트 완료!")
    
    # 비동기 테스트 실행
    asyncio.run(test_advanced_rumbaugh())