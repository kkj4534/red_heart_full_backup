"""
고급 경험 데이터베이스 시스템 - Linux 전용
Advanced Experience Database System for Linux

고급 AI 기법을 활용한 경험 저장, 검색, 압축, 학습 시스템
벡터 데이터베이스와 신경망 기반 경험 추론을 결합
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
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import sqlite3
from collections import defaultdict, deque
import hashlib
import uuid

# 고급 AI 라이브러리
# SentenceTransformer는 sentence_transformer_singleton을 통해 사용
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
import networkx as nx
# 고성능 벡터 검색 - subprocess로 분리 처리
# import faiss  # 직접 임포트 제거 - subprocess 사용
FAISS_AVAILABLE = True
import warnings
warnings.filterwarnings('ignore')

from config import ADVANCED_CONFIG, get_device, TORCH_DTYPE, BATCH_SIZE, MODELS_DIR
from utils import run_faiss_subprocess
# 데이터 모델 선택적 임포트
try:
    from data_models import (
        Decision, DecisionOutcome, ExperienceMetadata, 
        AdvancedExperience, ExperienceCluster, ExperiencePattern,
        LearningProgress, AdaptationStrategy
    )
    DATA_MODELS_AVAILABLE = True
except ImportError as e:
    # 기본 데이터 모델 사용
    DATA_MODELS_AVAILABLE = False
    from dataclasses import dataclass
    
    @dataclass
    class Decision:
        id: str
        situation: str
        action: str
    
    @dataclass  
    class DecisionOutcome:
        decision_id: str
        result: str
        satisfaction: float
    
    @dataclass
    class ExperienceMetadata:
        timestamp: str = ""
        tags: list = field(default_factory=list)
    
    @dataclass  
    class AdvancedExperience:
        id: str = ""
        content: str = ""
        metadata: dict = field(default_factory=dict)
    
    @dataclass
    class ExperienceCluster:
        id: str = ""
        experiences: list = field(default_factory=list)
    
    @dataclass
    class ExperiencePattern:
        pattern_id: str = ""
        pattern_type: str = ""
        confidence: float = 0.0
    
    @dataclass
    class LearningProgress:
        progress: float = 0.0
        stage: str = ""
    
    @dataclass
    class AdaptationStrategy:
        strategy_id: str = ""
        description: str = ""

logger = logging.getLogger('RedHeart.AdvancedExperienceDB')

@dataclass
class ExperienceVector:
    """경험 벡터 표현"""
    experience_id: str
    embedding: np.ndarray
    metadata_features: np.ndarray
    timestamp: datetime
    category: str
    importance_score: float = 0.0
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)

@dataclass
class ExperienceQuery:
    """경험 검색 쿼리"""
    query_text: str = ""
    query_embedding: Optional[np.ndarray] = None
    category_filter: Optional[str] = None
    time_range: Optional[Tuple[datetime, datetime]] = None
    similarity_threshold: float = 0.7
    max_results: int = 10
    include_metadata: bool = True
    boost_recent: bool = True

@dataclass
class ExperienceLearningState:
    """경험 학습 상태"""
    total_experiences: int = 0
    compressed_experiences: int = 0
    active_patterns: int = 0
    learning_accuracy: float = 0.0
    adaptation_rate: float = 0.0
    memory_efficiency: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)

class AdvancedNeuralMemory(nn.Module):
    """고급 신경망 기반 메모리 시스템"""
    
    def __init__(self, input_dim=768, hidden_dim=512, memory_slots=1000):
        super().__init__()
        
        # 메모리 인코더
        self.memory_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
        # 어텐션 메커니즘
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim // 4,
            num_heads=8,
            batch_first=True
        )
        
        # 경험 분류기
        self.experience_classifier = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 8, 10)  # 10개 카테고리
        )
        
        # 중요도 예측기
        self.importance_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.ReLU(),
            nn.Linear(hidden_dim // 8, 1),
            nn.Sigmoid()
        )
        
        # 메모리 슬롯
        self.memory_slots = memory_slots
        self.memory_bank = nn.Parameter(
            torch.randn(memory_slots, hidden_dim // 4)
        )
        
    def forward(self, experience_embeddings):
        """순전파"""
        # 메모리 인코딩
        encoded = self.memory_encoder(experience_embeddings)
        
        # 어텐션 적용
        attended, attention_weights = self.attention(
            encoded, self.memory_bank.unsqueeze(0).repeat(encoded.size(0), 1, 1), 
            self.memory_bank.unsqueeze(0).repeat(encoded.size(0), 1, 1)
        )
        
        # 분류 및 중요도 예측
        categories = self.experience_classifier(attended.mean(dim=1))
        importance = self.importance_predictor(attended.mean(dim=1))
        
        return {
            'encoded': encoded,
            'attended': attended,
            'attention_weights': attention_weights,
            'categories': categories,
            'importance': importance
        }

class AdvancedExperienceDatabase:
    """고급 경험 데이터베이스"""
    
    def __init__(self):
        """고급 경험 데이터베이스 초기화"""
        # 기본 설정 사용 (config에 없는 경우)
        self.config = ADVANCED_CONFIG.get('experience_db', {
            'max_experiences': 10000,
            'compression_threshold': 1000,
            'similarity_threshold': 0.8,
            'clustering_method': 'kmeans'
        })
        self.device = get_device()
        self.dtype = TORCH_DTYPE
        
        # 데이터베이스 경로 설정
        self.db_dir = os.path.join(MODELS_DIR, 'experience_db')
        self.vector_db_path = os.path.join(self.db_dir, 'vectors')
        self.metadata_db_path = os.path.join(self.db_dir, 'metadata.db')
        self.models_path = os.path.join(self.db_dir, 'models')
        
        os.makedirs(self.db_dir, exist_ok=True)
        os.makedirs(self.vector_db_path, exist_ok=True)
        os.makedirs(self.models_path, exist_ok=True)
        
        # GPU 순차 처리 및 배치 시스템 초기화 (무한 대기 이슈 해결)
        self.gpu_semaphore = asyncio.Semaphore(1)  # GPU 동시 접근 제한
        self.embedding_batch_queue = asyncio.Queue()  # 배치 처리 큐
        self.embedding_results = {}  # 요청 ID -> 결과 매핑
        self.batch_size = 32  # 동적 배치 크기
        self.batch_timeout = 1.0  # 1초 타임아웃 (더 나은 전략)
        self.current_batch = []  # 현재 배치
        self.batch_lock = asyncio.Lock()  # 배치 조작 동기화
        self.batch_processor_task = None  # 배치 처리 태스크
        
        # ThreadPoolExecutor 초기화
        from concurrent.futures import ThreadPoolExecutor
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="EmbeddingExecutor")
        
        # 임베딩 모델 초기화
        self._initialize_embedding_model()
        
        # 벡터 데이터베이스 초기화 (FAISS)
        self._initialize_vector_db()
        
        # 배치 처리 워커 시작
        self._start_batch_processor()
        
        # 메타데이터 SQLite 데이터베이스 초기화
        self._initialize_metadata_db()
        
        # 신경망 메모리 시스템 초기화
        self._initialize_neural_memory()
        
        # 학습 시스템 초기화
        self._initialize_learning_systems()
        
        # 경험 저장소
        self.experience_vectors = {}
        self.experience_metadata = {}
        self.experience_clusters = {}
        self.learning_patterns = {}
        
        # 학습 상태
        self.learning_state = ExperienceLearningState()
        
        # 캐시 시스템
        self.query_cache = {}
        self.embedding_cache = {}
        
        # 실시간 학습을 위한 버퍼
        self.learning_buffer = deque(maxlen=1000)
        
        logger.info("고급 경험 데이터베이스가 초기화되었습니다.")
    
    def _initialize_embedding_model(self):
        """임베딩 모델 초기화 (싱글톤 매니저 사용)"""
        try:
            from sentence_transformer_singleton import get_sentence_transformer
            
            # 싱글톤 매니저를 통해 공유 인스턴스 가져오기
            self.embedding_model = get_sentence_transformer(
                'paraphrase-multilingual-mpnet-base-v2',
                device=str(self.device),
                cache_folder=self.models_path
            )
            
            # 임베딩 차원
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            
            logger.info(f"임베딩 모델 로드 완료 (싱글톤, 차원: {self.embedding_dim})")
            
        except Exception as e:
            logger.error(f"임베딩 모델 초기화 실패: {e}")
            # fallback 없음 - 바로 예외 발생
            raise RuntimeError(f"SentenceTransformer 초기화 실패: {e}") from e
    
    def _initialize_vector_db(self):
        """FAISS 벡터 데이터베이스 초기화 - subprocess 분리 처리"""
        try:
            # subprocess를 통해 FAISS 인덱스 생성
            result = run_faiss_subprocess('create_index', {
                'dimension': self.embedding_dim,
                'index_type': 'IndexFlatL2'
            })
            
            if result['status'] == 'success':
                logger.info(f"FAISS 인덱스 생성 완료: {result['index_type']}, 차원={result['dimension']}")
                # 인덱스 상태를 내부적으로 관리
                self.vector_index_info = {
                    'index_type': result['index_type'],
                    'dimension': result['dimension'],
                    'total_vectors': 0
                }
                self.vector_ids = []
                logger.info("FAISS 벡터 데이터베이스 초기화 완료 (subprocess)")
            else:
                raise RuntimeError(f"FAISS 인덱스 생성 실패: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"FAISS 초기화 실패: {e}")
            raise RuntimeError(f"벡터 데이터베이스 초기화 실패: {e}")
    
    def _initialize_metadata_db(self):
        """SQLite 메타데이터 데이터베이스 초기화"""
        try:
            self.metadata_conn = sqlite3.connect(
                self.metadata_db_path, 
                check_same_thread=False
            )
            
            # 테이블 생성
            self.metadata_conn.execute('''
                CREATE TABLE IF NOT EXISTS experiences (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    category TEXT,
                    importance_score REAL,
                    access_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP,
                    last_accessed TIMESTAMP,
                    is_compressed BOOLEAN DEFAULT FALSE,
                    cluster_id TEXT,
                    metadata_json TEXT
                )
            ''')
            
            self.metadata_conn.execute('''
                CREATE TABLE IF NOT EXISTS experience_relations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT,
                    target_id TEXT,
                    relation_type TEXT,
                    strength REAL,
                    created_at TIMESTAMP,
                    FOREIGN KEY (source_id) REFERENCES experiences (id),
                    FOREIGN KEY (target_id) REFERENCES experiences (id)
                )
            ''')
            
            self.metadata_conn.execute('''
                CREATE TABLE IF NOT EXISTS learning_progress (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT,
                    metric_value REAL,
                    timestamp TIMESTAMP
                )
            ''')
            
            # 인덱스 생성
            self.metadata_conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_category ON experiences (category)
            ''')
            self.metadata_conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_created_at ON experiences (created_at)
            ''')
            self.metadata_conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_importance ON experiences (importance_score)
            ''')
            
            self.metadata_conn.commit()
            
            logger.info("메타데이터 데이터베이스가 초기화되었습니다.")
            
        except Exception as e:
            logger.error(f"메타데이터 데이터베이스 초기화 실패: {e}")
            self.metadata_conn = None
    
    def _initialize_neural_memory(self):
        """신경망 메모리 시스템 초기화"""
        try:
            self.neural_memory = AdvancedNeuralMemory(
                input_dim=self.embedding_dim,
                hidden_dim=self.config.get('neural_hidden_dim', 512),
                memory_slots=self.config.get('memory_slots', 1000)
            ).to(self.device)
            
            # 옵티마이저
            self.memory_optimizer = torch.optim.Adam(
                self.neural_memory.parameters(),
                lr=self.config.get('learning_rate', 0.001)
            )
            
            # 손실 함수들
            self.classification_loss = nn.CrossEntropyLoss()
            self.importance_loss = nn.MSELoss()
            
            # 모델 체크포인트 로드
            checkpoint_path = os.path.join(self.models_path, 'neural_memory.pt')
            if os.path.exists(checkpoint_path):
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                    self.neural_memory.load_state_dict(checkpoint['model_state'])
                    self.memory_optimizer.load_state_dict(checkpoint['optimizer_state'])
                    logger.info("신경망 메모리 체크포인트를 로드했습니다.")
                except:
                    logger.warning("체크포인트 로드 실패, 새 모델을 사용합니다.")
            
            logger.info("신경망 메모리 시스템이 초기화되었습니다.")
            
        except Exception as e:
            logger.error(f"신경망 메모리 초기화 실패: {e}")
            self.neural_memory = None
    
    def _initialize_learning_systems(self):
        """학습 시스템들 초기화"""
        # 이상 탐지기 (새로운 경험 패턴 감지)
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        
        # 클러스터링 알고리즘
        self.clusterer = DBSCAN(
            eps=0.3,
            min_samples=3,
            metric='cosine'
        )
        
        # 패턴 분류기
        self.pattern_classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        
        # 근사 이웃 검색기
        self.neighbor_searcher = NearestNeighbors(
            n_neighbors=10,
            metric='cosine'
        )
        
        logger.info("학습 시스템들이 초기화되었습니다.")
    
    async def store_experience(self, experience_text: str, 
                             metadata: Dict[str, Any] = None,
                             category: str = "general",
                             importance_score: Optional[float] = None) -> str:
        """
        고급 경험 저장
        
        Args:
            experience_text: 경험 텍스트
            metadata: 추가 메타데이터
            category: 카테고리
            importance_score: 중요도 점수 (None이면 자동 계산)
            
        Returns:
            경험 ID
        """
        try:
            # 경험 ID 생성
            experience_id = str(uuid.uuid4())
            
            # 임베딩 생성
            embedding = await self._generate_embedding(experience_text)
            
            # 중요도 점수 계산 (지정되지 않은 경우)
            if importance_score is None:
                importance_score = await self._calculate_importance_score(
                    experience_text, embedding, metadata
                )
            
            # 메타데이터 특성 추출
            metadata_features = self._extract_metadata_features(metadata or {})
            
            # 경험 벡터 생성
            experience_vector = ExperienceVector(
                experience_id=experience_id,
                embedding=embedding,
                metadata_features=metadata_features,
                timestamp=datetime.now(),
                category=category,
                importance_score=importance_score
            )
            
            # 벡터 데이터베이스에 저장 - subprocess 환경 분리 방식 사용
            if self.vector_index_info is not None:
                try:
                    # FAISS subprocess를 통해 벡터 추가
                    result = run_faiss_subprocess('add_vectors', {
                        'vectors': [embedding.tolist()],
                        'dimension': self.embedding_dim,
                        'index_type': self.vector_index_info['index_type']
                    })
                    
                    if result['status'] == 'success':
                        self.vector_index_info['total_vectors'] = result['total_vectors']
                        self.vector_ids.append(experience_id)
                        logger.debug(f"벡터가 FAISS에 추가됨: {experience_id}")
                    else:
                        logger.warning(f"FAISS 벡터 추가 실패: {result.get('error', 'Unknown error')}")
                        # 폴백: 메모리 기반 저장
                        self.vectors_memory.append(embedding)
                        self.vector_ids.append(experience_id)
                        
                except Exception as e:
                    logger.warning(f"FAISS subprocess 오류: {e}, 메모리 기반으로 저장")
                    # 폴백: 메모리 기반 저장
                    self.vectors_memory.append(embedding)
                    self.vector_ids.append(experience_id)
            else:
                # 메모리 기반 저장
                self.vectors_memory.append(embedding)
                self.vector_ids.append(experience_id)
            
            # 메타데이터 데이터베이스에 저장
            if self.metadata_conn:
                self.metadata_conn.execute('''
                    INSERT INTO experiences 
                    (id, text, category, importance_score, created_at, last_accessed, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    experience_id,
                    experience_text,
                    category,
                    importance_score,
                    datetime.now(),
                    datetime.now(),
                    json.dumps(metadata or {})
                ))
                self.metadata_conn.commit()
            
            # 경험 저장소에 추가
            self.experience_vectors[experience_id] = experience_vector
            self.experience_metadata[experience_id] = {
                'text': experience_text,
                'metadata': metadata or {},
                'category': category,
                'importance_score': importance_score,
                'created_at': datetime.now()
            }
            
            # 실시간 학습 버퍼에 추가
            self.learning_buffer.append({
                'id': experience_id,
                'embedding': embedding,
                'importance': importance_score,
                'category': category
            })
            
            # 학습 상태 업데이트
            self.learning_state.total_experiences += 1
            
            # 주기적 학습 트리거
            if len(self.learning_buffer) >= self.config.get('learning_batch_size', 50):
                asyncio.create_task(self._update_learning_systems())
            
            logger.info(f"경험이 저장되었습니다: {experience_id}")
            return experience_id
            
        except Exception as e:
            logger.error(f"경험 저장 실패: {e}")
            raise
    
    async def search_experiences(self, query: ExperienceQuery) -> List[Dict[str, Any]]:
        """
        고급 경험 검색
        
        Args:
            query: 검색 쿼리
            
        Returns:
            검색 결과 리스트
        """
        try:
            # 캐시 확인
            cache_key = self._generate_query_cache_key(query)
            if cache_key in self.query_cache:
                cached_result = self.query_cache[cache_key]
                # 캐시가 너무 오래되지 않았는지 확인
                if (datetime.now() - cached_result['timestamp']).seconds < 300:  # 5분
                    return cached_result['results']
            
            # 쿼리 임베딩 생성
            if query.query_embedding is None and query.query_text:
                query_embedding = await self._generate_embedding(query.query_text)
            else:
                query_embedding = query.query_embedding
            
            if query_embedding is None:
                return []
            
            # 벡터 검색
            similar_experiences = await self._vector_search(
                query_embedding, 
                query.max_results * 2  # 필터링을 위해 더 많이 검색
            )
            
            # 메타데이터 필터링
            filtered_results = await self._apply_metadata_filters(similar_experiences, query)
            
            # 최근성 부스트 적용
            if query.boost_recent:
                filtered_results = self._apply_recency_boost(filtered_results)
            
            # 결과 정렬 및 제한
            final_results = sorted(
                filtered_results, 
                key=lambda x: x['relevance_score'], 
                reverse=True
            )[:query.max_results]
            
            # 메타데이터 포함 여부에 따라 결과 구성
            if query.include_metadata:
                enriched_results = await self._enrich_results_with_metadata(final_results)
            else:
                enriched_results = final_results
            
            # 캐시 저장
            self.query_cache[cache_key] = {
                'results': enriched_results,
                'timestamp': datetime.now()
            }
            
            # 접근 횟수 업데이트
            for result in enriched_results:
                await self._update_access_count(result['experience_id'])
            
            return enriched_results
            
        except Exception as e:
            logger.error(f"경험 검색 실패: {e}")
            return []
    
    async def compress_experiences(self, compression_threshold: float = 0.1) -> Dict[str, Any]:
        """
        경험 압축 - 유사한 경험들을 클러스터링하여 메모리 효율성 향상
        
        Args:
            compression_threshold: 압축 임계값
            
        Returns:
            압축 결과 통계
        """
        try:
            logger.info("경험 압축을 시작합니다...")
            
            # 모든 경험 벡터 수집
            all_embeddings = []
            all_ids = []
            
            for exp_id, exp_vector in self.experience_vectors.items():
                all_embeddings.append(exp_vector.embedding)
                all_ids.append(exp_id)
            
            if len(all_embeddings) < 10:
                return {'message': '압축에 충분한 경험이 없습니다.', 'compressed': 0}
            
            embeddings_array = np.array(all_embeddings)
            
            # 클러스터링 수행
            cluster_labels = self.clusterer.fit_predict(embeddings_array)
            
            # 클러스터별 압축
            compression_stats = {
                'total_experiences': len(all_embeddings),
                'clusters_formed': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
                'compressed_experiences': 0,
                'compression_ratio': 0.0,
                'cluster_details': {}
            }
            
            for cluster_id in set(cluster_labels):
                if cluster_id == -1:  # 노이즈 포인트들
                    continue
                
                # 클러스터 멤버 찾기
                cluster_members = [all_ids[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
                
                if len(cluster_members) >= 3:  # 최소 3개 이상일 때만 압축
                    # 클러스터 대표 경험 선택 (중요도가 가장 높은 것)
                    representative_id = max(
                        cluster_members,
                        key=lambda x: self.experience_vectors[x].importance_score
                    )
                    
                    # 나머지 경험들을 압축
                    compressed_members = [m for m in cluster_members if m != representative_id]
                    
                    # 클러스터 정보 생성
                    cluster_info = ExperienceCluster(
                        cluster_id=f"cluster_{cluster_id}",
                        representative_id=representative_id,
                        member_ids=cluster_members,
                        compressed_ids=compressed_members,
                        centroid=np.mean([self.experience_vectors[m].embedding for m in cluster_members], axis=0),
                        compression_ratio=len(compressed_members) / len(cluster_members),
                        created_at=datetime.now()
                    )
                    
                    self.experience_clusters[cluster_info.cluster_id] = cluster_info
                    
                    # 압축된 경험들을 메타데이터 DB에서 압축 상태로 표시
                    if self.metadata_conn:
                        for compressed_id in compressed_members:
                            self.metadata_conn.execute('''
                                UPDATE experiences 
                                SET is_compressed = TRUE, cluster_id = ?
                                WHERE id = ?
                            ''', (cluster_info.cluster_id, compressed_id))
                        
                        self.metadata_conn.commit()
                    
                    compression_stats['compressed_experiences'] += len(compressed_members)
                    compression_stats['cluster_details'][cluster_info.cluster_id] = {
                        'size': len(cluster_members),
                        'compressed': len(compressed_members),
                        'representative': representative_id
                    }
            
            # 압축 비율 계산
            if compression_stats['total_experiences'] > 0:
                compression_stats['compression_ratio'] = (
                    compression_stats['compressed_experiences'] / 
                    compression_stats['total_experiences']
                )
            
            # 학습 상태 업데이트
            self.learning_state.compressed_experiences = compression_stats['compressed_experiences']
            self.learning_state.memory_efficiency = compression_stats['compression_ratio']
            
            logger.info(f"경험 압축 완료: {compression_stats['compressed_experiences']}개 압축됨")
            return compression_stats
            
        except Exception as e:
            logger.error(f"경험 압축 실패: {e}")
            return {'error': str(e), 'compressed': 0}
    
    async def learn_patterns(self) -> Dict[str, Any]:
        """
        경험 패턴 학습 - 신경망을 통한 고급 패턴 인식
        
        Returns:
            학습 결과
        """
        try:
            if self.neural_memory is None:
                return {'error': '신경망 메모리가 초기화되지 않았습니다.'}
            
            logger.info("경험 패턴 학습을 시작합니다...")
            
            # 학습 데이터 준비
            training_data = list(self.learning_buffer)
            if len(training_data) < 10:
                return {'message': '학습에 충분한 데이터가 없습니다.'}
            
            # 배치 데이터 구성
            embeddings = torch.tensor(
                [item['embedding'] for item in training_data],
                dtype=self.dtype,
                device=self.device
            )
            
            importance_targets = torch.tensor(
                [item['importance'] for item in training_data],
                dtype=self.dtype,
                device=self.device
            ).unsqueeze(1)
            
            # 카테고리 레이블 인코딩
            unique_categories = list(set(item['category'] for item in training_data))
            category_to_idx = {cat: idx for idx, cat in enumerate(unique_categories)}
            category_targets = torch.tensor(
                [category_to_idx[item['category']] for item in training_data],
                dtype=torch.long,
                device=self.device
            )
            
            # 신경망 학습
            self.neural_memory.train()
            total_loss = 0.0
            num_batches = (len(training_data) + self.config.get('batch_size', 32) - 1) // self.config.get('batch_size', 32)
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.config.get('batch_size', 32)
                end_idx = min(start_idx + self.config.get('batch_size', 32), len(training_data))
                
                batch_embeddings = embeddings[start_idx:end_idx]
                batch_importance = importance_targets[start_idx:end_idx]
                batch_categories = category_targets[start_idx:end_idx]
                
                # 순전파
                outputs = self.neural_memory(batch_embeddings)
                
                # 손실 계산
                classification_loss = self.classification_loss(
                    outputs['categories'], 
                    batch_categories
                )
                importance_loss = self.importance_loss(
                    outputs['importance'], 
                    batch_importance
                )
                
                total_loss_batch = classification_loss + importance_loss
                
                # 역전파
                self.memory_optimizer.zero_grad()
                total_loss_batch.backward()
                self.memory_optimizer.step()
                
                total_loss += total_loss_batch.item()
            
            avg_loss = total_loss / num_batches
            
            # 학습 정확도 계산
            self.neural_memory.eval()
            with torch.no_grad():
                outputs = self.neural_memory(embeddings)
                predicted_categories = torch.argmax(outputs['categories'], dim=1)
                accuracy = (predicted_categories == category_targets).float().mean().item()
            
            # 패턴 추출
            patterns = await self._extract_learned_patterns(training_data, outputs)
            
            # 학습 상태 업데이트
            self.learning_state.learning_accuracy = accuracy
            self.learning_state.active_patterns = len(patterns)
            self.learning_state.last_update = datetime.now()
            
            # 모델 체크포인트 저장
            checkpoint_path = os.path.join(self.models_path, 'neural_memory.pt')
            torch.save({
                'model_state': self.neural_memory.state_dict(),
                'optimizer_state': self.memory_optimizer.state_dict(),
                'loss': avg_loss,
                'accuracy': accuracy
            }, checkpoint_path)
            
            # 학습 버퍼 초기화
            self.learning_buffer.clear()
            
            learning_result = {
                'training_samples': len(training_data),
                'average_loss': avg_loss,
                'accuracy': accuracy,
                'patterns_learned': len(patterns),
                'model_saved': True
            }
            
            logger.info(f"패턴 학습 완료: 정확도 {accuracy:.3f}, 손실 {avg_loss:.3f}")
            return learning_result
            
        except Exception as e:
            logger.error(f"패턴 학습 실패: {e}")
            return {'error': str(e)}
    
    async def get_experience_insights(self, experience_id: str) -> Dict[str, Any]:
        """
        특정 경험에 대한 인사이트 분석
        
        Args:
            experience_id: 경험 ID
            
        Returns:
            경험 인사이트
        """
        try:
            if experience_id not in self.experience_vectors:
                return {'error': '경험을 찾을 수 없습니다.'}
            
            experience = self.experience_vectors[experience_id]
            metadata = self.experience_metadata[experience_id]
            
            # 유사 경험 찾기
            similar_experiences = await self._vector_search(
                experience.embedding, 
                max_results=5
            )
            
            # 클러스터 정보
            cluster_info = None
            for cluster_id, cluster in self.experience_clusters.items():
                if experience_id in cluster.member_ids:
                    cluster_info = {
                        'cluster_id': cluster_id,
                        'is_representative': experience_id == cluster.representative_id,
                        'cluster_size': len(cluster.member_ids),
                        'compression_ratio': cluster.compression_ratio
                    }
                    break
            
            # 신경망 분석 (가능한 경우)
            neural_analysis = None
            if self.neural_memory is not None:
                with torch.no_grad():
                    embedding_tensor = torch.tensor(
                        experience.embedding,
                        dtype=self.dtype,
                        device=self.device
                    ).unsqueeze(0)
                    
                    outputs = self.neural_memory(embedding_tensor)
                    
                    neural_analysis = {
                        'predicted_importance': float(outputs['importance'].squeeze()),
                        'category_probabilities': torch.softmax(outputs['categories'], dim=1).squeeze().cpu().numpy().tolist(),
                        'attention_weights': outputs['attention_weights'].squeeze().cpu().numpy().tolist()
                    }
            
            # 경험 영향도 분석
            impact_analysis = await self._analyze_experience_impact(experience_id)
            
            insights = {
                'experience_id': experience_id,
                'basic_info': {
                    'category': experience.category,
                    'importance_score': experience.importance_score,
                    'access_count': experience.access_count,
                    'created_at': experience.timestamp.isoformat(),
                    'last_accessed': experience.last_accessed.isoformat()
                },
                'similar_experiences': similar_experiences,
                'cluster_info': cluster_info,
                'neural_analysis': neural_analysis,
                'impact_analysis': impact_analysis,
                'text_preview': metadata['text'][:200] + "..." if len(metadata['text']) > 200 else metadata['text']
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"경험 인사이트 분석 실패: {e}")
            return {'error': str(e)}
    
    async def optimize_database(self) -> Dict[str, Any]:
        """
        데이터베이스 최적화 - 성능 개선 및 메모리 정리
        
        Returns:
            최적화 결과
        """
        try:
            logger.info("데이터베이스 최적화를 시작합니다...")
            
            optimization_results = {
                'cache_cleared': 0,
                'index_rebuilt': False,
                'compression_performed': False,
                'old_data_archived': 0,
                'performance_improvement': 0.0
            }
            
            # 캐시 정리
            old_cache_size = len(self.query_cache) + len(self.embedding_cache)
            self.query_cache.clear()
            self.embedding_cache.clear()
            optimization_results['cache_cleared'] = old_cache_size
            
            # FAISS 인덱스 재구성 (필요한 경우)
            if self.vector_index is not None and self.vector_index.ntotal > 1000:
                try:
                    # 인덱스 저장
                    index_file = os.path.join(self.vector_db_path, 'experience_index.faiss')
                    faiss.write_index(self.vector_index, index_file)
                    optimization_results['index_rebuilt'] = True
                except Exception as e:
                    logger.warning(f"인덱스 저장 실패: {e}")
            
            # 오래된 데이터 아카이브 (90일 이상)
            cutoff_date = datetime.now() - timedelta(days=90)
            archived_count = 0
            
            if self.metadata_conn:
                cursor = self.metadata_conn.execute('''
                    SELECT id FROM experiences 
                    WHERE created_at < ? AND access_count = 0
                ''', (cutoff_date,))
                
                old_experience_ids = [row[0] for row in cursor.fetchall()]
                
                for exp_id in old_experience_ids:
                    # 메타데이터에서 아카이브 표시
                    self.metadata_conn.execute('''
                        UPDATE experiences 
                        SET metadata_json = json_set(metadata_json, '$.archived', 'true')
                        WHERE id = ?
                    ''', (exp_id,))
                    
                    # 메모리에서 제거 (선택적)
                    if exp_id in self.experience_vectors:
                        del self.experience_vectors[exp_id]
                    if exp_id in self.experience_metadata:
                        del self.experience_metadata[exp_id]
                    
                    archived_count += 1
                
                self.metadata_conn.commit()
                optimization_results['old_data_archived'] = archived_count
            
            # 경험 압축 수행
            compression_result = await self.compress_experiences()
            if 'compressed_experiences' in compression_result:
                optimization_results['compression_performed'] = True
            
            # SQLite 최적화
            if self.metadata_conn:
                self.metadata_conn.execute('VACUUM')
                self.metadata_conn.execute('ANALYZE')
            
            logger.info(f"데이터베이스 최적화 완료: {archived_count}개 아카이브됨")
            return optimization_results
            
        except Exception as e:
            logger.error(f"데이터베이스 최적화 실패: {e}")
            return {'error': str(e)}
    
    # =====================================
    # 🚀 동적 배치 처리 시스템 (무한 대기 이슈 해결)
    # =====================================
    
    def _start_batch_processor(self):
        """배치 처리 워커 시작"""
        if self.batch_processor_task is None:
            self.batch_processor_task = asyncio.create_task(self._process_embedding_queue())
            logger.info("GPU 배치 처리 워커 시작됨 (무한 대기 이슈 해결)")
    
    async def _process_embedding_queue(self):
        """백그라운드 배치 처리 워커 - 더 나은 전략 구현"""
        while True:
            try:
                # 배치 수집 대기
                await asyncio.sleep(0.01)  # CPU 사용률 조절
                
                async with self.batch_lock:
                    if not self.current_batch:
                        continue
                    
                    # 동적 배치 처리 조건 (더 나은 전략)
                    should_process = (
                        len(self.current_batch) >= self.batch_size  # 32개 도달 -> 즉시 처리
                        or (self.current_batch and 
                            time.time() - self.current_batch[0]['timestamp'] >= self.batch_timeout)  # 1초 경과 -> 현재 배치 처리
                    )
                    
                    if should_process:
                        batch_to_process = self.current_batch.copy()
                        self.current_batch.clear()
                    else:
                        continue
                
                # GPU 순차 처리 (세마포어로 보장)
                await self._process_batch(batch_to_process)
                
            except Exception as e:
                logger.error(f"배치 처리 워커 오류: {e}")
                await asyncio.sleep(1)  # 오류 시 잠시 대기
    
    async def _process_batch(self, batch_requests):
        """실제 배치 처리 - GPU 순차 접근 보장"""
        if not batch_requests:
            return
        
        # GPU 세마포어로 순차 접근 보장
        async with self.gpu_semaphore:
            try:
                # 배치에서 텍스트만 추출
                texts = [req['text'] for req in batch_requests]
                
                # GPU에서 배치 임베딩 생성 (블로킹 방지를 위해 executor 사용)
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    embeddings = await loop.run_in_executor(
                        executor, 
                        self.embedding_model.encode, 
                        texts
                    )
                
                # 결과를 각 요청에 매핑
                for i, req in enumerate(batch_requests):
                    embedding = embeddings[i]
                    req_id = req['request_id']
                    text = req['text']
                    
                    # 캐시에 저장
                    self.embedding_cache[text] = embedding
                    
                    # 결과 저장 및 이벤트 신호
                    self.embedding_results[req_id] = embedding
                    req['event'].set()
                
                logger.debug(f"배치 처리 완료: {len(batch_requests)}개 임베딩")
                
            except Exception as e:
                logger.error(f"배치 처리 실패: {e}")
                # 실패한 요청들에 오류 신호 (폴백 없음, 즉시 실패)
                for req in batch_requests:
                    req_id = req['request_id']
                    # 실패 결과를 저장하지 않음 - RuntimeError가 발생하도록 함
                    req['event'].set()
    
    # =====================================
    # 📝 기존 메서드 (인터페이스 유지하며 배치 처리로 변경)
    # =====================================
    
    # 헬퍼 메서드들
    async def _generate_embedding(self, text: str) -> np.ndarray:
        """텍스트 임베딩 생성 - GPU 세마포어로 무한 대기 이슈 해결"""
        # 캐시 확인 (빠른 반환)
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        # 임베딩 모델이 없는 경우 즉시 실패
        if self.embedding_model is None:
            raise RuntimeError("임베딩 모델이 초기화되지 않았습니다. 폴백 모드 비활성화됨.")
        
        # GPU 세마포어로 순차 처리
        async with self.gpu_semaphore:
            try:
                # ThreadPoolExecutor로 블로킹 방지
                loop = asyncio.get_event_loop()
                
                # partial 함수로 키워드 인자 처리
                from functools import partial
                encode_func = partial(
                    self.embedding_model.encode,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                
                embedding = await loop.run_in_executor(
                    self.executor,
                    encode_func,
                    text
                )
                
                # 캐시에 저장
                self.embedding_cache[text] = embedding
                return embedding
                
            except Exception as e:
                logger.error(f"임베딩 생성 실패: {e}")
                # fallback 없음 - 바로 예외 발생
                raise RuntimeError(f"임베딩 생성 실패: {e}") from e
    
    async def _calculate_importance_score(self, text: str, embedding: np.ndarray, 
                                        metadata: Dict[str, Any]) -> float:
        """중요도 점수 계산"""
        # 기본 점수
        base_score = 0.5
        
        # 텍스트 길이 기반 점수
        length_score = min(1.0, len(text) / 1000)
        
        # 메타데이터 기반 점수
        metadata_score = 0.0
        if metadata:
            if metadata.get('urgency', 0) > 0.7:
                metadata_score += 0.3
            if metadata.get('impact', 0) > 0.7:
                metadata_score += 0.3
        
        # 유일성 점수 (기존 경험과의 차이)
        uniqueness_score = await self._calculate_uniqueness_score(embedding)
        
        # 종합 점수
        final_score = (
            base_score * 0.2 +
            length_score * 0.2 +
            metadata_score * 0.3 +
            uniqueness_score * 0.3
        )
        
        return min(1.0, max(0.0, final_score))
    
    async def _calculate_uniqueness_score(self, embedding: np.ndarray) -> float:
        """유일성 점수 계산"""
        if not self.experience_vectors:
            return 1.0  # 첫 번째 경험
        
        # 기존 임베딩들과의 유사도 계산
        existing_embeddings = np.array([
            exp.embedding for exp in self.experience_vectors.values()
        ])
        
        similarities = cosine_similarity([embedding], existing_embeddings)[0]
        max_similarity = np.max(similarities)
        
        # 유일성 = 1 - 최대 유사도
        return 1.0 - max_similarity
    
    def _extract_metadata_features(self, metadata: Dict[str, Any]) -> np.ndarray:
        """메타데이터 특성 추출"""
        # 기본 특성 벡터 (크기 10으로 고정)
        features = np.zeros(10)
        
        # 수치형 메타데이터 추출
        if 'urgency' in metadata:
            features[0] = float(metadata['urgency'])
        if 'impact' in metadata:
            features[1] = float(metadata['impact'])
        if 'confidence' in metadata:
            features[2] = float(metadata['confidence'])
        
        # 범주형 메타데이터를 수치로 변환
        if 'emotion' in metadata:
            emotion_map = {'positive': 1.0, 'negative': -1.0, 'neutral': 0.0}
            features[3] = emotion_map.get(metadata['emotion'], 0.0)
        
        # 시간 관련 특성
        if 'time_sensitivity' in metadata:
            features[4] = float(metadata['time_sensitivity'])
        
        return features
    
    async def _vector_search(self, query_embedding: np.ndarray, 
                           max_results: int = 10) -> List[Dict[str, Any]]:
        """벡터 검색 수행 - subprocess 환경 분리 방식 사용"""
        if self.vector_index_info is not None and self.vector_index_info['total_vectors'] > 0:
            try:
                # FAISS subprocess를 통해 검색
                result = run_faiss_subprocess('search_vectors', {
                    'vectors': self.vectors_memory,  # 기존 벡터들
                    'query_vectors': [query_embedding.tolist()],
                    'dimension': self.embedding_dim,
                    'index_type': self.vector_index_info['index_type'],
                    'k': max_results
                })
                
                if result['status'] == 'success':
                    search_results = []
                    distances = result['distances'][0]  # 첫 번째 쿼리 결과
                    indices = result['indices'][0]
                    
                    for i, (distance, idx) in enumerate(zip(distances, indices)):
                        if idx >= 0 and idx < len(self.vector_ids):  # 유효한 인덱스
                            similarity = 1.0 / (1.0 + distance)  # 거리를 유사도로 변환
                            search_results.append({
                                'experience_id': self.vector_ids[idx],
                                'similarity': float(similarity),
                                'relevance_score': float(similarity)
                            })
                    
                    return search_results
                else:
                    logger.warning(f"FAISS 검색 실패: {result.get('error', 'Unknown error')}")
                    # 폴백: 메모리 기반 검색
                    
            except Exception as e:
                logger.warning(f"FAISS subprocess 검색 오류: {e}, 메모리 기반으로 검색")
                # 폴백: 메모리 기반 검색
        
        # 폴백: 메모리 기반 검색
        if not self.vectors_memory:
            return []
        
        similarities = cosine_similarity([query_embedding], self.vectors_memory)[0]
        
        # 상위 결과 선택
        top_indices = np.argsort(similarities)[::-1][:max_results]
        
        results = []
        for idx in top_indices:
            if idx < len(self.vector_ids) and similarities[idx] > 0.1:  # 최소 유사도
                results.append({
                    'experience_id': self.vector_ids[idx],
                    'similarity': float(similarities[idx]),
                    'relevance_score': float(similarities[idx])
                })
        
        return results
    
    async def _apply_metadata_filters(self, results: List[Dict[str, Any]], 
                                    query: ExperienceQuery) -> List[Dict[str, Any]]:
        """메타데이터 필터 적용"""
        filtered = []
        
        for result in results:
            # 카테고리 필터
            if query.category_filter:
                exp_id = result.get('experience_id')
                if exp_id and exp_id in self.experience_vectors:
                    exp_category = self.experience_vectors[exp_id].category
                    if exp_category != query.category_filter:
                        continue
            
            # 시간 범위 필터
            if query.time_range:
                exp_id = result.get('experience_id')
                if exp_id and exp_id in self.experience_vectors:
                    exp_time = self.experience_vectors[exp_id].timestamp
                    if not (query.time_range[0] <= exp_time <= query.time_range[1]):
                        continue
            
            # 유사도 임계값
            if result['similarity'] >= query.similarity_threshold:
                filtered.append(result)
        
        return filtered
    
    def _apply_recency_boost(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """최근성 부스트 적용"""
        current_time = datetime.now()
        
        for result in results:
            exp_id = result.get('experience_id')
            if exp_id and exp_id in self.experience_vectors:
                exp_time = self.experience_vectors[exp_id].timestamp
                days_ago = (current_time - exp_time).days
                
                # 최근 경험일수록 높은 부스트
                recency_boost = max(0.1, 1.0 - (days_ago / 365))  # 1년 기준
                result['relevance_score'] *= recency_boost
        
        return results
    
    async def _enrich_results_with_metadata(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """결과에 메타데이터 추가"""
        enriched = []
        
        for result in results:
            exp_id = result.get('experience_id')
            if exp_id and exp_id in self.experience_metadata:
                metadata = self.experience_metadata[exp_id]
                result.update({
                    'text': metadata['text'],
                    'category': metadata['category'],
                    'importance_score': metadata['importance_score'],
                    'created_at': metadata['created_at'].isoformat(),
                    'metadata': metadata['metadata']
                })
            
            enriched.append(result)
        
        return enriched
    
    async def _update_access_count(self, experience_id: str):
        """접근 횟수 업데이트"""
        if experience_id in self.experience_vectors:
            self.experience_vectors[experience_id].access_count += 1
            self.experience_vectors[experience_id].last_accessed = datetime.now()
        
        if self.metadata_conn:
            self.metadata_conn.execute('''
                UPDATE experiences 
                SET access_count = access_count + 1, last_accessed = ?
                WHERE id = ?
            ''', (datetime.now(), experience_id))
            self.metadata_conn.commit()
    
    def _generate_query_cache_key(self, query: ExperienceQuery) -> str:
        """쿼리 캐시 키 생성"""
        key_components = [
            query.query_text[:50],  # 텍스트 일부
            query.category_filter or "",
            str(query.similarity_threshold),
            str(query.max_results)
        ]
        return hashlib.md5("|".join(key_components).encode()).hexdigest()
    
    async def _update_learning_systems(self):
        """학습 시스템 업데이트"""
        try:
            # 패턴 학습 수행
            learning_result = await self.learn_patterns()
            
            # 이상 탐지 모델 업데이트
            if len(self.experience_vectors) >= 50:
                embeddings = np.array([exp.embedding for exp in self.experience_vectors.values()])
                self.anomaly_detector.fit(embeddings)
            
            logger.info("학습 시스템이 업데이트되었습니다.")
            
        except Exception as e:
            logger.error(f"학습 시스템 업데이트 실패: {e}")
    
    async def _extract_learned_patterns(self, training_data: List[Dict], 
                                      neural_outputs: Dict) -> List[ExperiencePattern]:
        """학습된 패턴 추출"""
        patterns = []
        
        try:
            # 어텐션 가중치 분석
            attention_weights = neural_outputs['attention_weights'].cpu().numpy()
            
            # 높은 어텐션을 받는 메모리 슬롯 식별
            avg_attention = np.mean(attention_weights, axis=0)
            high_attention_slots = np.where(avg_attention > np.percentile(avg_attention, 80))[0]
            
            for slot_idx in high_attention_slots:
                pattern = ExperiencePattern(
                    pattern_id=f"pattern_{slot_idx}",
                    pattern_type="attention_based",
                    description=f"고어텐션 메모리 슬롯 {slot_idx}",
                    confidence=float(avg_attention[slot_idx]),
                    examples=[item['id'] for item in training_data[:5]],  # 예시
                    created_at=datetime.now()
                )
                patterns.append(pattern)
        
        except Exception as e:
            logger.error(f"패턴 추출 실패: {e}")
        
        return patterns
    
    async def _analyze_experience_impact(self, experience_id: str) -> Dict[str, Any]:
        """경험 영향도 분석"""
        try:
            if experience_id not in self.experience_vectors:
                return {'error': '경험을 찾을 수 없습니다.'}
            
            experience = self.experience_vectors[experience_id]
            
            # 직접적 영향 (유사 경험들)
            similar_experiences = await self._vector_search(experience.embedding, max_results=10)
            
            # 간접적 영향 (참조된 횟수, 클러스터 내 위치 등)
            indirect_impact = {
                'access_frequency': experience.access_count,
                'recency_impact': (datetime.now() - experience.timestamp).days,
                'importance_percentile': 0.0
            }
            
            # 중요도 백분위 계산
            all_importance_scores = [exp.importance_score for exp in self.experience_vectors.values()]
            if all_importance_scores:
                importance_rank = sorted(all_importance_scores, reverse=True).index(experience.importance_score)
                indirect_impact['importance_percentile'] = (1 - importance_rank / len(all_importance_scores)) * 100
            
            return {
                'direct_impact': {
                    'similar_experiences_count': len(similar_experiences),
                    'average_similarity': np.mean([sim['similarity'] for sim in similar_experiences]) if similar_experiences else 0.0
                },
                'indirect_impact': indirect_impact,
                'overall_impact_score': self._calculate_overall_impact_score(experience, similar_experiences, indirect_impact)
            }
            
        except Exception as e:
            logger.error(f"영향도 분석 실패: {e}")
            return {'error': str(e)}
    
    def _calculate_overall_impact_score(self, experience: ExperienceVector, 
                                      similar_experiences: List[Dict], 
                                      indirect_impact: Dict) -> float:
        """전체 영향도 점수 계산"""
        # 기본 중요도
        base_score = experience.importance_score
        
        # 유사 경험 영향
        similarity_score = len(similar_experiences) / 10.0  # 정규화
        
        # 접근 빈도 영향
        access_score = min(1.0, experience.access_count / 10.0)
        
        # 최근성 영향 (최근일수록 높음)
        recency_days = indirect_impact['recency_impact']
        recency_score = max(0.1, 1.0 - (recency_days / 365))
        
        # 가중 평균
        overall_score = (
            base_score * 0.4 +
            similarity_score * 0.2 +
            access_score * 0.2 +
            recency_score * 0.2
        )
        
        return min(1.0, overall_score)
    
    def close(self):
        """데이터베이스 연결 종료"""
        if self.metadata_conn:
            self.metadata_conn.close()
        
        # FAISS 인덱스 저장
        if self.vector_index is not None:
            try:
                index_file = os.path.join(self.vector_db_path, 'experience_index.faiss')
                faiss.write_index(self.vector_index, index_file)
            except Exception as e:
                logger.error(f"인덱스 저장 실패: {e}")
        
        logger.info("고급 경험 데이터베이스가 종료되었습니다.")

def create_advanced_experience_database() -> AdvancedExperienceDatabase:
    """고급 경험 데이터베이스 생성"""
    return AdvancedExperienceDatabase()

# 테스트 코드
if __name__ == "__main__":
    import asyncio
    
    async def test_advanced_database():
        """고급 데이터베이스 테스트"""
        db = create_advanced_experience_database()
        
        print("=== 고급 경험 데이터베이스 테스트 ===\n")
        
        # 경험 저장 테스트
        test_experiences = [
            "오늘 어려운 윤리적 결정을 내려야 했다. 개인의 이익과 공공의 이익 사이에서 고민했지만, 결국 공공의 이익을 선택했다.",
            "팀 프로젝트에서 갈등이 발생했다. 서로 다른 의견을 조율하는 과정에서 많은 것을 배웠다.",
            "새로운 기술을 배우는 과정에서 실패를 많이 경험했지만, 결국 성공할 수 있었다.",
            "도덕적 딜레마 상황에서 원칙을 지키는 것의 중요성을 깨달았다.",
            "협력과 소통의 중요성을 실감했던 경험이었다."
        ]
        
        stored_ids = []
        for i, exp_text in enumerate(test_experiences):
            exp_id = await db.store_experience(
                exp_text,
                metadata={'test_id': i, 'urgency': 0.5 + i * 0.1},
                category='learning' if i % 2 == 0 else 'ethical'
            )
            stored_ids.append(exp_id)
            print(f"경험 저장됨: {exp_id[:8]}...")
        
        print(f"\n총 {len(stored_ids)}개 경험이 저장되었습니다.\n")
        
        # 검색 테스트
        query = ExperienceQuery(
            query_text="윤리적 결정과 도덕적 딜레마",
            similarity_threshold=0.3,
            max_results=3
        )
        
        search_results = await db.search_experiences(query)
        print(f"검색 결과: {len(search_results)}개")
        for result in search_results:
            print(f"- 유사도: {result['relevance_score']:.3f}, 텍스트: {result['text'][:50]}...")
        
        # 압축 테스트
        print(f"\n경험 압축 수행...")
        compression_result = await db.compress_experiences()
        print(f"압축 결과: {compression_result}")
        
        # 패턴 학습 테스트
        print(f"\n패턴 학습 수행...")
        learning_result = await db.learn_patterns()
        print(f"학습 결과: {learning_result}")
        
        # 인사이트 분석 테스트
        if stored_ids:
            print(f"\n경험 인사이트 분석...")
            insights = await db.get_experience_insights(stored_ids[0])
            print(f"인사이트: {insights['basic_info']}")
        
        # 최적화 테스트
        print(f"\n데이터베이스 최적화...")
        optimization_result = await db.optimize_database()
        print(f"최적화 결과: {optimization_result}")
        
        # 종료
        db.close()
        print(f"\n테스트 완료!")
    
    # 비동기 테스트 실행
    asyncio.run(test_advanced_database())