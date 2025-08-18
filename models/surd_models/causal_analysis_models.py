"""
SURD 인과 분석 모델들
SURD (Synergy, Unique, Redundant, Deterministic) Causal Analysis Models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle
from datetime import datetime
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from concurrent.futures import ThreadPoolExecutor
import itertools

@dataclass
class SURDConfig:
    """SURD 분석 설정"""
    num_variables: int = 4
    embedding_dim: int = 768
    hidden_dims: List[int] = None
    k_neighbors: int = 5
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256, 128, 64]

@dataclass
class InformationMeasures:
    """정보 이론적 측정값들"""
    mutual_information: float
    conditional_mutual_information: float
    transfer_entropy: float
    partial_information_decomposition: Dict[str, float]
    causal_strength: float
    confidence_interval: Tuple[float, float]

class KraskovEstimator:
    """Kraskov-Stögbauer-Grassberger 상호정보량 추정기"""
    
    def __init__(self, k: int = 5):
        self.k = k
        
    def estimate_mi(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Kraskov 추정법으로 상호정보량 계산"""
        n = len(X)
        if n == 0:
            return 0.0
            
        # 데이터 정규화
        X = (X - np.mean(X)) / (np.std(X) + 1e-8)
        Y = (Y - np.mean(Y)) / (np.std(Y) + 1e-8)
        
        # 결합 공간에서의 kNN
        XY = np.column_stack([X, Y])
        nbrs_XY = NearestNeighbors(n_neighbors=self.k+1, metric='chebyshev').fit(XY)
        distances_XY, _ = nbrs_XY.kneighbors(XY)
        
        # X와 Y 각각의 공간에서 거리 계산
        nbrs_X = NearestNeighbors(metric='chebyshev').fit(X.reshape(-1, 1))
        nbrs_Y = NearestNeighbors(metric='chebyshev').fit(Y.reshape(-1, 1))
        
        mi = 0.0
        for i in range(n):
            eps = distances_XY[i, self.k]  # k번째 이웃까지의 거리
            
            # X 공간에서 eps 거리 내의 점들
            nx = len(nbrs_X.radius_neighbors([X[i].reshape(-1)], eps)[1][0]) - 1
            # Y 공간에서 eps 거리 내의 점들
            ny = len(nbrs_Y.radius_neighbors([Y[i].reshape(-1)], eps)[1][0]) - 1
            
            mi += digamma(self.k) - digamma(nx + 1) - digamma(ny + 1) + digamma(n)
        
        return max(0, mi / n / np.log(2))  # 비트 단위로 변환
    
    def estimate_conditional_mi(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> float:
        """조건부 상호정보량 I(X;Y|Z) 계산"""
        # I(X;Y|Z) = I(X;Y,Z) - I(X;Z)
        YZ = np.column_stack([Y, Z])
        mi_xyz = self.estimate_mi(X, YZ.flatten() if YZ.shape[1] == 1 else YZ.mean(axis=1))
        mi_xz = self.estimate_mi(X, Z)
        
        return max(0, mi_xyz - mi_xz)
    
    def estimate_transfer_entropy(self, X: np.ndarray, Y: np.ndarray, lag: int = 1) -> float:
        """전이 엔트로피 계산"""
        if len(X) <= lag or len(Y) <= lag:
            return 0.0
            
        # Y(t+1), Y(t), X(t) 준비
        Y_future = Y[lag:]
        Y_past = Y[:-lag]
        X_past = X[:-lag]
        
        # TE(X→Y) = I(Y(t+1); X(t) | Y(t))
        return self.estimate_conditional_mi(Y_future, X_past, Y_past)

class NeuralCausalModel(nn.Module):
    """신경망 기반 인과관계 모델"""
    
    def __init__(self, config: SURDConfig):
        super().__init__()
        self.config = config
        
        # 변수별 인코더
        self.variable_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.embedding_dim, config.hidden_dims[0]),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(config.hidden_dims[0], config.hidden_dims[1]),
                nn.ReLU(),
                nn.Linear(config.hidden_dims[1], config.hidden_dims[2])
            ) for _ in range(config.num_variables)
        ])
        
        # 인과관계 어텐션
        self.causal_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dims[2],
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 시너지 예측 네트워크
        self.synergy_predictor = nn.Sequential(
            nn.Linear(config.hidden_dims[2] * config.num_variables, config.hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_dims[1], config.hidden_dims[3]),
            nn.ReLU(),
            nn.Linear(config.hidden_dims[3], 1),
            nn.Sigmoid()
        )
        
        # 유니크 정보 예측기
        self.unique_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_dims[2], config.hidden_dims[3]),
                nn.ReLU(),
                nn.Linear(config.hidden_dims[3], 1),
                nn.Sigmoid()
            ) for _ in range(config.num_variables)
        ])
        
        # 리던던시 예측기
        self.redundancy_predictor = nn.Sequential(
            nn.Linear(config.hidden_dims[2] * config.num_variables, config.hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(config.hidden_dims[1], config.hidden_dims[3]),
            nn.ReLU(),
            nn.Linear(config.hidden_dims[3], 1),
            nn.Sigmoid()
        )
        
        # 인과 강도 예측기
        self.causal_strength_predictor = nn.Sequential(
            nn.Linear(config.hidden_dims[2] * config.num_variables, config.hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(config.hidden_dims[1], config.num_variables * config.num_variables),
            nn.Sigmoid()
        )
        
    def forward(self, variable_embeddings: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        if len(variable_embeddings) != self.config.num_variables:
            raise ValueError(f"Expected {self.config.num_variables} variables, got {len(variable_embeddings)}")
        
        # 각 변수 인코딩
        encoded_variables = []
        for i, embedding in enumerate(variable_embeddings):
            encoded = self.variable_encoders[i](embedding)
            encoded_variables.append(encoded)
        
        # 변수들을 스택하여 어텐션 적용
        stacked_variables = torch.stack(encoded_variables, dim=1)  # [batch, num_vars, hidden_dim]
        
        # 인과관계 어텐션
        attended_variables, attention_weights = self.causal_attention(
            stacked_variables, stacked_variables, stacked_variables
        )
        
        # 모든 변수 정보 결합
        combined_variables = attended_variables.view(attended_variables.size(0), -1)
        
        # 시너지 예측
        synergy = self.synergy_predictor(combined_variables)
        
        # 각 변수별 유니크 정보 예측
        unique_info = []
        for i, encoded_var in enumerate(encoded_variables):
            unique = self.unique_predictors[i](encoded_var)
            unique_info.append(unique)
        unique_info = torch.stack(unique_info, dim=1)  # [batch, num_vars, 1]
        
        # 리던던시 예측
        redundancy = self.redundancy_predictor(combined_variables)
        
        # 인과 강도 예측 (변수 간 페어별)
        causal_strengths = self.causal_strength_predictor(combined_variables)
        causal_matrix = causal_strengths.view(-1, self.config.num_variables, self.config.num_variables)
        
        return {
            'synergy': synergy.squeeze(-1),
            'unique_info': unique_info.squeeze(-1),
            'redundancy': redundancy.squeeze(-1),
            'causal_matrix': causal_matrix,
            'attention_weights': attention_weights,
            'encoded_variables': encoded_variables
        }

class PIDDecomposition:
    """Partial Information Decomposition (PID) 분해"""
    
    def __init__(self, estimator: KraskovEstimator):
        self.estimator = estimator
    
    def decompose(self, X1: np.ndarray, X2: np.ndarray, Y: np.ndarray) -> Dict[str, float]:
        """PID 분해 수행"""
        # I(X1,X2;Y) - 전체 상호정보량
        X_combined = np.column_stack([X1, X2])
        total_mi = self.estimator.estimate_mi(X_combined.mean(axis=1), Y)
        
        # I(X1;Y) - X1과 Y의 상호정보량
        mi_x1_y = self.estimator.estimate_mi(X1, Y)
        
        # I(X2;Y) - X2와 Y의 상호정보량
        mi_x2_y = self.estimator.estimate_mi(X2, Y)
        
        # I(X1;Y|X2) - X2가 주어졌을 때 X1과 Y의 조건부 상호정보량
        unique_x1 = self.estimator.estimate_conditional_mi(X1, Y, X2)
        
        # I(X2;Y|X1) - X1이 주어졌을 때 X2와 Y의 조건부 상호정보량
        unique_x2 = self.estimator.estimate_conditional_mi(X2, Y, X1)
        
        # Williams-Beer 최소값 원리로 Redundancy 계산
        redundancy = min(mi_x1_y, mi_x2_y)
        
        # Synergy 계산
        synergy = total_mi - unique_x1 - unique_x2 - redundancy
        
        return {
            'synergy': max(0, synergy),
            'unique_x1': max(0, unique_x1),
            'unique_x2': max(0, unique_x2),
            'redundancy': max(0, redundancy),
            'total_mi': total_mi
        }

class AdvancedSURDAnalyzer(nn.Module):
    """고급 SURD 분석기"""
    
    def __init__(self, estimator_or_config):
        super().__init__()
        
        # 유연한 초기화: config 또는 estimator 받기
        if isinstance(estimator_or_config, KraskovEstimator):
            self.kraskov_estimator = estimator_or_config
            # 기본 config 생성
            self.config = SURDConfig()
        else:
            self.config = estimator_or_config
            # Kraskov 추정기
            self.kraskov_estimator = KraskovEstimator(k=self.config.k_neighbors)
        
        # PID 분해기
        self.pid_decomposer = PIDDecomposition(self.kraskov_estimator)
        
        # 신경망 모델
        self.neural_model = NeuralCausalModel(self.config)
        
        # 부트스트랩을 위한 스레드 풀
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
    def forward(self, variable_embeddings: List[torch.Tensor], 
                variable_data: Optional[List[np.ndarray]] = None) -> Dict[str, Any]:
        
        # 신경망 기반 분석
        neural_output = self.neural_model(variable_embeddings)
        
        result = {
            'neural_analysis': neural_output,
            'synergy': neural_output['synergy'],
            'unique_info': neural_output['unique_info'],
            'redundancy': neural_output['redundancy'],
            'causal_matrix': neural_output['causal_matrix']
        }
        
        # 실제 데이터가 제공된 경우 Kraskov 추정 수행
        if variable_data and len(variable_data) >= 2:
            kraskov_results = self._analyze_with_kraskov(variable_data)
            result['kraskov_analysis'] = kraskov_results
            
        return result
    
    def _analyze_with_kraskov(self, variable_data: List[np.ndarray]) -> Dict[str, Any]:
        """Kraskov 추정기를 사용한 실제 정보 분석"""
        results = {}
        
        # 모든 변수 쌍에 대해 상호정보량 계산
        mi_matrix = np.zeros((len(variable_data), len(variable_data)))
        
        for i in range(len(variable_data)):
            for j in range(i + 1, len(variable_data)):
                mi = self.kraskov_estimator.estimate_mi(variable_data[i], variable_data[j])
                mi_matrix[i, j] = mi_matrix[j, i] = mi
        
        results['mutual_info_matrix'] = mi_matrix
        
        # 2개 이상의 변수가 있는 경우 PID 분해
        if len(variable_data) >= 3:
            # 첫 3개 변수로 PID 분해 (X1, X2 → Y)
            pid_result = self.pid_decomposer.decompose(
                variable_data[0], variable_data[1], variable_data[2]
            )
            results['pid_decomposition'] = pid_result
        
        # Transfer Entropy 계산 (시계열 데이터인 경우)
        if len(variable_data) >= 2:
            te_matrix = np.zeros((len(variable_data), len(variable_data)))
            
            for i in range(len(variable_data)):
                for j in range(len(variable_data)):
                    if i != j:
                        te = self.kraskov_estimator.estimate_transfer_entropy(
                            variable_data[i], variable_data[j]
                        )
                        te_matrix[i, j] = te
            
            results['transfer_entropy_matrix'] = te_matrix
        
        return results
    
    def bootstrap_confidence_interval(self, variable_data: List[np.ndarray], 
                                    analysis_func, **kwargs) -> Tuple[float, Tuple[float, float]]:
        """부트스트랩 신뢰구간 계산"""
        n_samples = len(variable_data[0])
        bootstrap_results = []
        
        for _ in range(self.config.bootstrap_samples):
            # 부트스트랩 샘플링
            indices = np.random.choice(n_samples, n_samples, replace=True)
            bootstrap_data = [data[indices] for data in variable_data]
            
            # 분석 함수 실행
            result = analysis_func(bootstrap_data, **kwargs)
            bootstrap_results.append(result)
        
        # 신뢰구간 계산
        mean_result = np.mean(bootstrap_results)
        alpha = 1 - self.config.confidence_level
        lower = np.percentile(bootstrap_results, 100 * alpha / 2)
        upper = np.percentile(bootstrap_results, 100 * (1 - alpha / 2))
        
        return mean_result, (lower, upper)

class CausalNetworkBuilder:
    """인과 네트워크 구성기"""
    
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
    
    def build_causal_graph(self, causal_matrix: np.ndarray, 
                          variable_names: List[str]) -> nx.DiGraph:
        """인과 그래프 구성"""
        G = nx.DiGraph()
        
        # 노드 추가
        for name in variable_names:
            G.add_node(name)
        
        # 엣지 추가 (임계값 이상의 인과관계만)
        for i in range(len(variable_names)):
            for j in range(len(variable_names)):
                if i != j and causal_matrix[i, j] > self.threshold:
                    G.add_edge(variable_names[i], variable_names[j], 
                             weight=causal_matrix[i, j])
        
        return G
    
    def analyze_causal_structure(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """인과 구조 분석"""
        return {
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'density': nx.density(graph),
            'centrality': nx.degree_centrality(graph),
            'pagerank': nx.pagerank(graph),
            'strongly_connected_components': list(nx.strongly_connected_components(graph)),
            'is_dag': nx.is_directed_acyclic_graph(graph)
        }

class SURDModelManager:
    """SURD 모델 관리자"""
    
    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.config = None
        
    def save_model(self, model: AdvancedSURDAnalyzer, config: SURDConfig,
                   epoch: int, metrics: Dict[str, float]):
        """모델 저장"""
        model_path = self.models_dir / f"surd_model_epoch_{epoch}.pth"
        config_path = self.models_dir / "surd_config.json"
        
        # 모델 저장 (신경망 부분만)
        save_data = {
            'neural_model_state_dict': model.neural_model.state_dict(),
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        torch.save(save_data, model_path)
        
        # 설정 저장
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(config), f, indent=2, ensure_ascii=False)
            
    def load_model(self, model_path: Optional[Path] = None) -> AdvancedSURDAnalyzer:
        """모델 로드"""
        # 설정 로드
        config_path = self.models_dir / "surd_config.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            self.config = SURDConfig(**config_dict)
        else:
            self.config = SURDConfig()
        
        if model_path is None:
            model_files = list(self.models_dir.glob("surd_model_epoch_*.pth"))
            if not model_files:
                raise FileNotFoundError("저장된 SURD 모델이 없습니다.")
            
            model_path = max(model_files, key=lambda p: p.stat().st_mtime)
        
        checkpoint = torch.load(model_path, map_location='cpu')
        
        model = AdvancedSURDAnalyzer(self.config)
        model.neural_model.load_state_dict(checkpoint['neural_model_state_dict'])
        
        self.model = model
        return model

def create_surd_config(**kwargs) -> SURDConfig:
    """SURD 설정 생성 헬퍼"""
    return SURDConfig(**kwargs)

def extract_causal_insights(surd_results: Dict[str, Any]) -> List[str]:
    """SURD 결과에서 인과관계 인사이트 추출"""
    insights = []
    
    # 시너지 분석
    if 'synergy' in surd_results:
        synergy = surd_results['synergy']
        if isinstance(synergy, torch.Tensor):
            synergy = synergy.mean().item()
        
        if synergy > 0.7:
            insights.append("변수들 간에 강한 시너지 효과가 관찰됩니다.")
        elif synergy > 0.3:
            insights.append("변수들 간에 중간 수준의 시너지가 있습니다.")
        else:
            insights.append("변수들 간 시너지 효과가 제한적입니다.")
    
    # 리던던시 분석
    if 'redundancy' in surd_results:
        redundancy = surd_results['redundancy']
        if isinstance(redundancy, torch.Tensor):
            redundancy = redundancy.mean().item()
        
        if redundancy > 0.6:
            insights.append("변수들 간에 높은 중복성이 있습니다.")
        elif redundancy > 0.3:
            insights.append("변수들 간에 적당한 중복성이 관찰됩니다.")
    
    # 인과 강도 분석
    if 'causal_matrix' in surd_results:
        causal_matrix = surd_results['causal_matrix']
        if isinstance(causal_matrix, torch.Tensor):
            causal_matrix = causal_matrix.detach().numpy()
        
        max_causal = np.max(causal_matrix)
        if max_causal > 0.8:
            insights.append("강한 인과관계가 감지되었습니다.")
        elif max_causal > 0.5:
            insights.append("중간 정도의 인과관계가 있습니다.")
    
    return insights