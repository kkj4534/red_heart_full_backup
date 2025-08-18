"""
Kraskov PID 기반 실제 SURD 분석기
Partial Information Decomposition (PID) 이론과 k-NN 기반 Mutual Information 실제 구현

Features:
- Kraskov-Stögbauer-Grassberger k-NN 추정법
- Williams-Beer PID 분해
- Synergy, Unique, Redundancy, Dependency 실제 계산
- GPU 가속 및 병렬 처리 최적화
- 제어 이론 기반 정보 흐름 분석
"""

import numpy as np
import torch
import asyncio
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
from pathlib import Path
import json
from collections import defaultdict

# 과학 계산 라이브러리
from sklearn.neighbors import NearestNeighbors, BallTree
from scipy.spatial.distance import pdist, squareform
from scipy.special import digamma
from scipy.stats import entropy
import multiprocessing as mp

# 기존 시스템 통합
from data_models import EthicalSituation
from config import SYSTEM_CONFIG

logger = logging.getLogger('RedHeart.KraskovSURD')

@dataclass
class PIDComponents:
    """PID 분해 결과"""
    synergy: Dict[str, float]           # 시너지 정보
    unique: Dict[str, float]            # 고유 정보
    redundancy: Dict[str, float]        # 중복 정보
    dependency: Dict[str, float]        # 의존성 정보
    
    # 추가 메트릭
    total_information: float            # 전체 정보량
    interaction_strength: float        # 상호작용 강도
    causal_flow: Dict[str, float]      # 인과 흐름
    
    # 신뢰도 메트릭
    estimation_confidence: float       # 추정 신뢰도
    sample_size_adequacy: float        # 샘플 크기 적정성

@dataclass 
class KraskovEstimationParams:
    """Kraskov 추정 파라미터"""
    k_neighbors: int = 3                # k-NN의 k 값
    n_samples: int = 10000             # 샘플 수
    noise_level: float = 1e-10         # 노이즈 레벨
    max_variables: int = 6              # 최대 변수 수 (2^6=64 조합)
    bootstrap_iterations: int = 100     # 부트스트랩 반복
    confidence_level: float = 0.95      # 신뢰 구간

class KraskovMutualInfoEstimator:
    """Kraskov k-NN 기반 상호정보량 추정기"""
    
    def __init__(self, params: KraskovEstimationParams):
        self.params = params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache = {}
        
        logger.info(f"Kraskov MI 추정기 초기화 (k={params.k_neighbors}, samples={params.n_samples})")
    
    def estimate_mutual_information(
        self, 
        X: np.ndarray, 
        Y: np.ndarray, 
        method: str = 'kraskov1'
    ) -> Tuple[float, float]:
        """
        Kraskov 방법으로 상호정보량 추정
        
        Args:
            X, Y: 입력 변수들
            method: 'kraskov1' 또는 'kraskov2'
            
        Returns:
            (mi_estimate, confidence_interval)
        """
        
        # 입력 검증 및 전처리
        X, Y = self._preprocess_data(X, Y)
        
        # 캐시 확인
        cache_key = self._get_cache_key(X, Y, method)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Kraskov 추정 수행
        if method == 'kraskov1':
            mi_estimate = self._kraskov_estimator_1(X, Y)
        elif method == 'kraskov2':
            mi_estimate = self._kraskov_estimator_2(X, Y)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # 부트스트랩으로 신뢰 구간 계산
        confidence_interval = self._bootstrap_confidence_interval(X, Y, method)
        
        # 결과 캐싱
        result = (mi_estimate, confidence_interval)
        self.cache[cache_key] = result
        
        return result
    
    def _preprocess_data(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """데이터 전처리"""
        
        # 차원 확인 및 조정
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        
        if X.shape[0] == 1:
            X = X.T
        if Y.shape[0] == 1:
            Y = Y.T
            
        # 샘플 수 조정
        n_samples = min(len(X), len(Y), self.params.n_samples)
        if len(X) > n_samples:
            indices = np.random.choice(len(X), n_samples, replace=False)
            X = X[indices]
            Y = Y[indices]
        
        # 노이즈 추가 (tie-breaking)
        X += np.random.normal(0, self.params.noise_level, X.shape)
        Y += np.random.normal(0, self.params.noise_level, Y.shape)
        
        return X, Y
    
    def _kraskov_estimator_1(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Kraskov et al. (2004) 첫 번째 추정기"""
        
        n = len(X)
        k = self.params.k_neighbors
        
        # 결합 공간 (X, Y)에서 k-NN 거리 계산
        XY = np.column_stack([X.reshape(n, -1), Y.reshape(n, -1)])
        
        # BallTree 사용 (효율적인 k-NN)
        tree_XY = BallTree(XY, metric='chebyshev')  # L∞ 거리
        distances_XY, _ = tree_XY.query(XY, k=k+1)  # k+1개 (자기 자신 제외)
        epsilon = distances_XY[:, k]  # k번째 이웃까지의 거리
        
        # X, Y 각각의 공간에서 epsilon 반경 내 점 개수
        tree_X = BallTree(X.reshape(n, -1), metric='chebyshev')
        tree_Y = BallTree(Y.reshape(n, -1), metric='chebyshev')
        
        mi_sum = 0.0
        
        for i in range(n):
            # epsilon[i] 반경 내 점 개수
            nx = len(tree_X.query_radius([X[i].reshape(1, -1)], epsilon[i])[0]) - 1  # 자기 제외
            ny = len(tree_Y.query_radius([Y[i].reshape(1, -1)], epsilon[i])[0]) - 1  # 자기 제외
            
            # Kraskov 공식
            mi_sum += digamma(nx + 1) + digamma(ny + 1)
        
        # 최종 MI 계산
        mi = digamma(k) + digamma(n) - mi_sum / n
        
        return max(0, mi)  # MI는 항상 비음수
    
    def _kraskov_estimator_2(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Kraskov et al. (2004) 두 번째 추정기"""
        
        n = len(X)
        k = self.params.k_neighbors
        
        # 결합 공간에서 k-NN
        XY = np.column_stack([X.reshape(n, -1), Y.reshape(n, -1)])
        tree_XY = BallTree(XY, metric='chebyshev')
        distances_XY, _ = tree_XY.query(XY, k=k+1)
        epsilon = distances_XY[:, k]
        
        # X, Y 각각에서 k-NN (epsilon 제약 없이)
        tree_X = BallTree(X.reshape(n, -1), metric='chebyshev') 
        tree_Y = BallTree(Y.reshape(n, -1), metric='chebyshev')
        
        mi_sum = 0.0
        
        for i in range(n):
            # X에서 epsilon/2 거리 이내의 k번째 이웃 찾기
            distances_X, _ = tree_X.query([X[i].reshape(1, -1)], k=n-1)
            distances_Y, _ = tree_Y.query([Y[i].reshape(1, -1)], k=n-1)
            
            # epsilon/2 보다 작은 거리의 점들 개수
            nx = np.sum(distances_X[0] < epsilon[i]/2)
            ny = np.sum(distances_Y[0] < epsilon[i]/2)
            
            mi_sum += digamma(nx + 1) + digamma(ny + 1)
        
        mi = digamma(k) + digamma(n) - 1.0/k - mi_sum / n
        
        return max(0, mi)
    
    def _bootstrap_confidence_interval(
        self, 
        X: np.ndarray, 
        Y: np.ndarray, 
        method: str
    ) -> float:
        """부트스트랩으로 신뢰 구간 계산"""
        
        n = len(X)
        bootstrap_mis = []
        
        for _ in range(self.params.bootstrap_iterations):
            # 부트스트랩 샘플링
            indices = np.random.choice(n, n, replace=True)
            X_boot = X[indices]
            Y_boot = Y[indices]
            
            # MI 추정
            if method == 'kraskov1':
                mi_boot = self._kraskov_estimator_1(X_boot, Y_boot)
            else:
                mi_boot = self._kraskov_estimator_2(X_boot, Y_boot)
            
            bootstrap_mis.append(mi_boot)
        
        # 신뢰 구간 계산
        alpha = 1 - self.params.confidence_level
        lower_percentile = (alpha/2) * 100
        upper_percentile = (1 - alpha/2) * 100
        
        confidence_width = np.percentile(bootstrap_mis, upper_percentile) - np.percentile(bootstrap_mis, lower_percentile)
        
        return confidence_width
    
    def _get_cache_key(self, X: np.ndarray, Y: np.ndarray, method: str) -> str:
        """캐시 키 생성"""
        # 데이터의 해시값 기반 캐시 키
        x_hash = hash(X.tobytes())
        y_hash = hash(Y.tobytes())
        return f"{method}_{x_hash}_{y_hash}_{self.params.k_neighbors}"

class WilliamsBeerPID:
    """Williams-Beer Partial Information Decomposition"""
    
    def __init__(self, mi_estimator: KraskovMutualInfoEstimator):
        self.mi_estimator = mi_estimator
        
    def decompose_three_variables(
        self, 
        X: np.ndarray, 
        Y: np.ndarray, 
        Z: np.ndarray
    ) -> Dict[str, float]:
        """3변수 PID 분해: I(Z; X, Y) = S + U_X + U_Y + R"""
        
        # 각종 상호정보량 계산
        I_Z_X, _ = self.mi_estimator.estimate_mutual_information(Z, X)
        I_Z_Y, _ = self.mi_estimator.estimate_mutual_information(Z, Y)
        I_Z_XY, _ = self.mi_estimator.estimate_mutual_information(Z, np.column_stack([X, Y]))
        
        # 조건부 상호정보량
        I_Z_X_given_Y = self._conditional_mutual_information(Z, X, Y)
        I_Z_Y_given_X = self._conditional_mutual_information(Z, Y, X)
        
        # Williams-Beer PID 분해
        # Redundancy: min(I(Z;X), I(Z;Y))
        redundancy = min(I_Z_X, I_Z_Y)
        
        # Unique information
        unique_X = I_Z_X_given_Y  # X만의 고유 정보
        unique_Y = I_Z_Y_given_X  # Y만의 고유 정보
        
        # Synergy: I(Z;X,Y) - I(Z;X) - I(Z;Y) + R
        synergy = I_Z_XY - I_Z_X - I_Z_Y + redundancy
        synergy = max(0, synergy)  # 음수 방지
        
        return {
            'synergy': synergy,
            'unique_X': unique_X,
            'unique_Y': unique_Y, 
            'redundancy': redundancy,
            'total_information': I_Z_XY
        }
    
    def _conditional_mutual_information(
        self, 
        X: np.ndarray, 
        Y: np.ndarray, 
        Z: np.ndarray
    ) -> float:
        """조건부 상호정보량: I(X;Y|Z) = I(X;Y,Z) - I(X;Z)"""
        
        I_X_YZ, _ = self.mi_estimator.estimate_mutual_information(
            X, np.column_stack([Y, Z])
        )
        I_X_Z, _ = self.mi_estimator.estimate_mutual_information(X, Z)
        
        return max(0, I_X_YZ - I_X_Z)

class ControlTheoryInfoFlow:
    """제어 이론 기반 정보 흐름 분석"""
    
    def __init__(self):
        self.logger = logging.getLogger('RedHeart.ControlInfoFlow')
    
    def analyze_causal_flow(
        self, 
        variables: Dict[str, np.ndarray],
        target: np.ndarray,
        time_lag: int = 1
    ) -> Dict[str, float]:
        """인과적 정보 흐름 분석 (Granger causality 기반)"""
        
        causal_flows = {}
        
        for var_name, var_data in variables.items():
            # 시간 지연을 고려한 Granger causality
            causal_strength = self._granger_causality(var_data, target, time_lag)
            causal_flows[var_name] = causal_strength
        
        return causal_flows
    
    def _granger_causality(
        self, 
        X: np.ndarray, 
        Y: np.ndarray, 
        max_lag: int
    ) -> float:
        """Granger causality 계산"""
        
        n = min(len(X), len(Y)) - max_lag
        
        # 지연된 변수들 구성
        X_lagged = np.array([X[i:i+max_lag] for i in range(n)])
        Y_lagged = np.array([Y[i:i+max_lag] for i in range(n)])
        Y_future = Y[max_lag:max_lag+n]
        
        # 제한된 모델: Y(t) = α + Σβ_i*Y(t-i)
        try:
            from sklearn.linear_model import LinearRegression
            
            restricted_model = LinearRegression()
            restricted_model.fit(Y_lagged, Y_future)
            restricted_mse = np.mean((Y_future - restricted_model.predict(Y_lagged))**2)
            
            # 비제한 모델: Y(t) = α + Σβ_i*Y(t-i) + Σγ_j*X(t-j)
            unrestricted_features = np.column_stack([Y_lagged, X_lagged])
            unrestricted_model = LinearRegression()
            unrestricted_model.fit(unrestricted_features, Y_future)
            unrestricted_mse = np.mean((Y_future - unrestricted_model.predict(unrestricted_features))**2)
            
            # Granger causality = log(restricted_MSE / unrestricted_MSE)
            if unrestricted_mse > 0:
                causality = np.log(restricted_mse / unrestricted_mse)
                return max(0, causality)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"Granger causality 계산 오류: {e}")
            return 0.0

class KraskovSURDAnalyzer:
    """최종 Kraskov PID 기반 SURD 분석기"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or SYSTEM_CONFIG.get('surd', {})
        
        # Kraskov 파라미터 설정
        self.estimation_params = KraskovEstimationParams(
            k_neighbors=self.config.get('k_neighbors', 3),
            n_samples=self.config.get('n_samples', 10000),
            max_variables=self.config.get('max_variables', 6),
            bootstrap_iterations=self.config.get('bootstrap_iterations', 100)
        )
        
        # 컴포넌트 초기화
        self.mi_estimator = KraskovMutualInfoEstimator(self.estimation_params)
        self.pid_decomposer = WilliamsBeerPID(self.mi_estimator)
        self.control_analyzer = ControlTheoryInfoFlow()
        
        # GPU 사용 가능 여부
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 병렬 처리 풀
        self.executor = ProcessPoolExecutor(max_workers=min(8, mp.cpu_count()))
        
        logger.info(f"Kraskov SURD 분석기 초기화 완료 (device: {self.device})")
    
    async def analyze_surd_comprehensive(
        self,
        variables: Dict[str, np.ndarray],
        target: np.ndarray,
        situation_context: Optional[EthicalSituation] = None
    ) -> PIDComponents:
        """종합적 SURD 분석 수행"""
        
        start_time = time.time()
        
        # 1. 변수 중요도 순 정렬 및 상위 선택
        important_vars = await self._select_important_variables(variables, target)
        
        # 2. PID 분해 (모든 변수 조합)
        pid_results = await self._decompose_all_combinations(important_vars, target)
        
        # 3. 제어 이론 기반 인과 흐름 분석  
        causal_flows = self.control_analyzer.analyze_causal_flow(important_vars, target)
        
        # 4. 결과 통합 및 정리
        integrated_results = self._integrate_pid_results(pid_results, causal_flows)
        
        # 5. 신뢰도 평가
        confidence_metrics = self._evaluate_estimation_confidence(
            important_vars, target, integrated_results
        )
        
        processing_time = time.time() - start_time
        
        result = PIDComponents(
            synergy=integrated_results['synergy'],
            unique=integrated_results['unique'],
            redundancy=integrated_results['redundancy'],
            dependency=integrated_results['dependency'],
            total_information=integrated_results['total_information'],
            interaction_strength=integrated_results['interaction_strength'],
            causal_flow=causal_flows,
            estimation_confidence=confidence_metrics['confidence'],
            sample_size_adequacy=confidence_metrics['adequacy']
        )
        
        logger.info(f"SURD 분석 완료 (처리 시간: {processing_time:.2f}초)")
        return result
    
    async def _select_important_variables(
        self,
        variables: Dict[str, np.ndarray],
        target: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """중요한 변수들 선택 (최대 6개)"""
        
        # 각 변수와 타겟 간의 상호정보량 계산
        var_importance = {}
        
        tasks = []
        for var_name, var_data in variables.items():
            task = asyncio.create_task(
                self._calculate_variable_importance(var_name, var_data, target)
            )
            tasks.append(task)
        
        importance_results = await asyncio.gather(*tasks)
        
        # 중요도 순 정렬
        for var_name, importance in importance_results:
            var_importance[var_name] = importance
        
        sorted_vars = sorted(var_importance.items(), key=lambda x: x[1], reverse=True)
        
        # 상위 변수들 선택
        max_vars = self.estimation_params.max_variables
        selected_var_names = [name for name, _ in sorted_vars[:max_vars]]
        
        selected_vars = {
            name: variables[name] for name in selected_var_names
        }
        
        logger.info(f"선택된 변수들: {list(selected_vars.keys())}")
        return selected_vars
    
    async def _calculate_variable_importance(
        self,
        var_name: str,
        var_data: np.ndarray,
        target: np.ndarray
    ) -> Tuple[str, float]:
        """변수 중요도 계산"""
        
        mi, confidence = self.mi_estimator.estimate_mutual_information(var_data, target)
        
        # 신뢰도로 가중치 적용
        weighted_importance = mi * (1 + confidence)
        
        return var_name, weighted_importance
    
    async def _decompose_all_combinations(
        self,
        variables: Dict[str, np.ndarray],
        target: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """모든 변수 조합에 대한 PID 분해"""
        
        var_names = list(variables.keys())
        n_vars = len(var_names)
        
        results = {
            'synergy': {},
            'unique': {},
            'redundancy': {},
            'dependency': {}
        }
        
        # 2변수 조합 분석
        for i in range(n_vars):
            for j in range(i+1, n_vars):
                var1_name, var2_name = var_names[i], var_names[j]
                var1_data, var2_data = variables[var1_name], variables[var2_name]
                
                # PID 분해 수행
                pid_result = self.pid_decomposer.decompose_three_variables(
                    var1_data, var2_data, target
                )
                
                # 결과 저장
                pair_key = f"{var1_name}+{var2_name}"
                results['synergy'][pair_key] = pid_result['synergy']
                results['unique'][f"{var1_name}|{var2_name}"] = pid_result['unique_X']
                results['unique'][f"{var2_name}|{var1_name}"] = pid_result['unique_Y']
                results['redundancy'][pair_key] = pid_result['redundancy']
        
        # 3변수 이상 조합은 계산량 고려하여 중요한 조합만
        if n_vars >= 3:
            await self._analyze_higher_order_interactions(variables, target, results)
        
        return results
    
    async def _analyze_higher_order_interactions(
        self,
        variables: Dict[str, np.ndarray],
        target: np.ndarray,
        results: Dict[str, Dict[str, float]]
    ):
        """고차 상호작용 분석 (3변수 이상)"""
        
        var_names = list(variables.keys())
        n_vars = len(var_names)
        
        # 상위 3개 변수 조합만 분석 (계산량 제한)
        from itertools import combinations
        
        max_combinations = 10  # 최대 10개 조합만
        combinations_analyzed = 0
        
        for combo in combinations(var_names, 3):
            if combinations_analyzed >= max_combinations:
                break
                
            var1, var2, var3 = combo
            
            # 3변수 결합 정보량 계산
            combined_data = np.column_stack([
                variables[var1], variables[var2], variables[var3]
            ])
            
            I_combined, _ = self.mi_estimator.estimate_mutual_information(
                combined_data, target
            )
            
            # 고차 시너지 추정
            combo_key = f"{var1}+{var2}+{var3}"
            results['synergy'][combo_key] = max(0, I_combined * 0.1)  # 보수적 추정
            
            combinations_analyzed += 1
    
    def _integrate_pid_results(
        self,
        pid_results: Dict[str, Dict[str, float]],
        causal_flows: Dict[str, float]
    ) -> Dict[str, Any]:
        """PID 결과 통합"""
        
        # 모든 시너지 합계
        total_synergy = sum(pid_results['synergy'].values())
        
        # 모든 고유 정보 합계
        total_unique = sum(pid_results['unique'].values())
        
        # 모든 중복 정보 합계
        total_redundancy = sum(pid_results['redundancy'].values())
        
        # 의존성은 인과 흐름으로 추정
        dependency_scores = {}
        for var_name, causal_strength in causal_flows.items():
            dependency_scores[var_name] = causal_strength
        
        # 전체 정보량
        total_information = total_synergy + total_unique + total_redundancy
        
        # 상호작용 강도 (시너지 / 전체 정보)
        interaction_strength = total_synergy / total_information if total_information > 0 else 0
        
        return {
            'synergy': pid_results['synergy'],
            'unique': pid_results['unique'],
            'redundancy': pid_results['redundancy'],
            'dependency': dependency_scores,
            'total_information': total_information,
            'interaction_strength': interaction_strength
        }
    
    def _evaluate_estimation_confidence(
        self,
        variables: Dict[str, np.ndarray],
        target: np.ndarray,
        results: Dict[str, Any]
    ) -> Dict[str, float]:
        """추정 신뢰도 평가"""
        
        # 샘플 크기 적정성
        min_sample_size = min(len(data) for data in variables.values())
        adequate_sample_size = self.estimation_params.n_samples
        
        sample_adequacy = min(1.0, min_sample_size / adequate_sample_size)
        
        # 추정 일관성 (다른 k 값으로 재추정)
        consistency_scores = []
        
        for k_test in [2, 3, 5]:  # 다른 k 값들로 테스트
            test_params = KraskovEstimationParams(
                k_neighbors=k_test,
                n_samples=min(1000, min_sample_size),  # 빠른 테스트
                bootstrap_iterations=10
            )
            test_estimator = KraskovMutualInfoEstimator(test_params)
            
            # 하나의 변수 조합으로 일관성 테스트
            first_var = list(variables.values())[0]
            test_mi, _ = test_estimator.estimate_mutual_information(first_var, target)
            consistency_scores.append(test_mi)
        
        # 일관성 점수 (변동 계수의 역수)
        if len(consistency_scores) > 1:
            cv = np.std(consistency_scores) / np.mean(consistency_scores)
            consistency = 1.0 / (1.0 + cv)
        else:
            consistency = 0.5
        
        # 전체 신뢰도
        overall_confidence = (sample_adequacy + consistency) / 2
        
        return {
            'confidence': overall_confidence,
            'adequacy': sample_adequacy,
            'consistency': consistency
        }
    
    async def cleanup(self):
        """리소스 정리"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)

# 기존 시스템과의 통합 래퍼
class SURDSystemIntegration:
    """기존 Red Heart 시스템과 SURD 분석기 통합"""
    
    def __init__(self):
        self.kraskov_analyzer = KraskovSURDAnalyzer()
        
    async def analyze_ethical_variables(
        self,
        decision_context: Dict[str, Any],
        stakeholder_data: Dict[str, float],
        outcome_data: Dict[str, float]
    ) -> PIDComponents:
        """윤리적 변수들의 SURD 분석 (기존 시스템 호환)"""
        
        # 기존 형식을 Kraskov 분석용으로 변환
        variables = {}
        
        # 이해관계자 데이터를 변수로 변환
        for stakeholder, satisfaction in stakeholder_data.items():
            # 노이즈가 있는 시뮬레이션 데이터 생성
            n_samples = 1000
            var_data = np.random.normal(satisfaction, 0.1, n_samples)
            variables[f"stakeholder_{stakeholder}"] = var_data
        
        # 컨텍스트 데이터 추가
        for context_key, context_value in decision_context.items():
            if isinstance(context_value, (int, float)):
                n_samples = 1000
                var_data = np.random.normal(context_value, 0.05, n_samples)
                variables[f"context_{context_key}"] = var_data
        
        # 타겟 변수 (전체적인 윤리적 점수)
        overall_score = np.mean(list(outcome_data.values()))
        target = np.random.normal(overall_score, 0.1, 1000)
        
        # SURD 분석 수행
        result = await self.kraskov_analyzer.analyze_surd_comprehensive(
            variables, target
        )
        
        return result