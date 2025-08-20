"""
Parameter Crossover System
서로 다른 에폭의 최적 파라미터를 결합하여 최종 모델 생성
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import logging
from datetime import datetime
from collections import defaultdict
import copy

logger = logging.getLogger(__name__)


class ParameterCrossoverSystem:
    """
    파라미터 교차 시스템
    - 모듈별 최적 에폭의 파라미터 선택
    - 가중치 평균화 및 앙상블
    - 유전 알고리즘 기반 최적화
    - 파라미터 보간
    """
    
    def __init__(self,
                 crossover_strategy: str = 'selective',
                 blend_ratio: float = 0.7,
                 mutation_rate: float = 0.01):
        """
        Args:
            crossover_strategy: 교차 전략 ('selective', 'weighted', 'genetic', 'interpolate')
            blend_ratio: 블렌딩 비율 (주 모델의 가중치)
            mutation_rate: 유전 알고리즘의 변이율
        """
        self.crossover_strategy = crossover_strategy
        self.blend_ratio = blend_ratio
        self.mutation_rate = mutation_rate
        
        # 체크포인트 저장소
        self.checkpoint_pool = {}
        self.module_best_epochs = {}
        
        # 교차 결과
        self.crossover_results = []
        self.best_combination = None
        
        logger.info("✅ Parameter Crossover System 초기화")
        logger.info(f"  - 전략: {crossover_strategy}")
        logger.info(f"  - 블렌드 비율: {blend_ratio}")
    
    def add_checkpoint(self, 
                      epoch: int,
                      checkpoint_path: str,
                      module_metrics: Dict[str, float]):
        """
        체크포인트 풀에 추가
        
        Args:
            epoch: 에폭 번호
            checkpoint_path: 체크포인트 파일 경로
            module_metrics: 모듈별 메트릭
        """
        self.checkpoint_pool[epoch] = {
            'path': checkpoint_path,
            'metrics': module_metrics
        }
        
        # 모듈별 최고 성능 에폭 업데이트
        for module_name, metric_value in module_metrics.items():
            if module_name not in self.module_best_epochs or \
               metric_value < self.module_best_epochs[module_name]['metric']:
                self.module_best_epochs[module_name] = {
                    'epoch': epoch,
                    'metric': metric_value
                }
    
    def perform_crossover(self,
                         model: nn.Module,
                         optimal_epochs: Dict[str, int],
                         validation_fn: Optional[Any] = None) -> nn.Module:
        """
        파라미터 교차 수행
        
        Args:
            model: 기본 모델 구조
            optimal_epochs: 모듈별 최적 에폭
            validation_fn: 검증 함수 (선택적)
            
        Returns:
            교차된 최종 모델
        """
        logger.info("🧬 Parameter Crossover 시작...")
        logger.info(f"  - 최적 에폭: {optimal_epochs}")
        
        if self.crossover_strategy == 'selective':
            return self._selective_crossover(model, optimal_epochs)
        elif self.crossover_strategy == 'weighted':
            return self._weighted_crossover(model, optimal_epochs)
        elif self.crossover_strategy == 'genetic':
            return self._genetic_crossover(model, optimal_epochs, validation_fn)
        elif self.crossover_strategy == 'interpolate':
            return self._interpolate_crossover(model, optimal_epochs)
        else:
            raise ValueError(f"Unknown crossover strategy: {self.crossover_strategy}")
    
    def _selective_crossover(self, 
                           model: nn.Module,
                           optimal_epochs: Dict[str, int]) -> nn.Module:
        """
        선택적 교차: 각 모듈마다 최적 에폭의 파라미터 선택
        메모리 효율적인 방식으로 state_dict만 교체
        """
        logger.info("  📌 선택적 교차 수행 중...")
        
        # 현재 모델의 state_dict 저장 (deepcopy 대신)
        current_state = model.state_dict()
        crossover_state = {}  # 새로운 state_dict 구성
        
        # 기본적으로 현재 state를 복사
        for key, value in current_state.items():
            crossover_state[key] = value.clone()
        
        # 모듈별로 최적 에폭의 파라미터 로드
        for module_name, optimal_epoch in optimal_epochs.items():
            if optimal_epoch not in self.checkpoint_pool:
                logger.warning(f"    ⚠️ 에폭 {optimal_epoch}의 체크포인트가 없습니다: {module_name}")
                continue
            
            checkpoint_path = self.checkpoint_pool[optimal_epoch]['path']
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # 모듈별 state_dict 추출
            if 'model_state' in checkpoint:
                checkpoint_state = checkpoint['model_state']
            elif 'model' in checkpoint:
                checkpoint_state = checkpoint['model']
            else:
                checkpoint_state = checkpoint
            
            # 해당 모듈의 파라미터만 업데이트
            updated_keys = []
            
            # Special case 1: neural_analyzers dict 처리
            if module_name == 'neural_analyzers' and module_name in checkpoint_state:
                neural_states = checkpoint_state[module_name]
                if isinstance(neural_states, dict):
                    # nested dict 구조 처리
                    for analyzer_name, analyzer_state in neural_states.items():
                        if isinstance(analyzer_state, dict):
                            for param_key, param_value in analyzer_state.items():
                                full_key = f"neural_analyzers.{analyzer_name}.{param_key}"
                                if full_key in crossover_state:
                                    crossover_state[full_key] = param_value.cpu() if torch.is_tensor(param_value) else param_value
                                    updated_keys.append(full_key)
                    if not updated_keys:
                        # 키가 안맞으면 다른 형식 시도 (dict of dict이 아닌 경우)
                        for key, value in neural_states.items():
                            if torch.is_tensor(value):
                                # 직접 매핑 시도
                                crossover_state[f"neural_analyzers.{key}"] = value.cpu()
                                updated_keys.append(f"neural_analyzers.{key}")
            
            # Special case 2: system 처리 (통합 파라미터)
            elif module_name == 'system' and module_name in checkpoint_state:
                system_state = checkpoint_state[module_name]
                if isinstance(system_state, dict) and 'meta' not in system_state:
                    # system의 백본 통합 레이어 처리
                    for sub_module, sub_state in system_state.items():
                        if sub_module != 'meta' and isinstance(sub_state, dict):
                            for key, value in sub_state.items():
                                # backbone.final_norm 등의 실제 키로 매핑
                                if 'backbone_final_norm' in sub_module:
                                    full_key = f"backbone.final_norm.{key}"
                                elif 'backbone_output_projection' in sub_module:
                                    full_key = f"backbone.output_projection.{key}"
                                else:
                                    full_key = f"{sub_module}.{key}"
                                
                                if full_key in crossover_state:
                                    crossover_state[full_key] = value.cpu() if torch.is_tensor(value) else value
                                    updated_keys.append(full_key)
            
            # 일반적인 경우: checkpoint_state가 모듈별로 저장된 경우
            elif module_name in checkpoint_state:
                module_state = checkpoint_state[module_name]
                # 모듈의 state_dict를 crossover_state에 추가
                for key, value in module_state.items():
                    full_key = f"{module_name}.{key}"
                    if full_key in crossover_state:
                        # CPU에서 작업 (이미 CPU에 있지만 명시적으로)
                        crossover_state[full_key] = value.cpu() if torch.is_tensor(value) else value
                        updated_keys.append(full_key)
            else:
                # 전체 state_dict가 플랫하게 저장된 경우 (기존 방식)
                module_prefix = f"{module_name}."
                for key, value in checkpoint_state.items():
                    if key.startswith(module_prefix):
                        if key in crossover_state:
                            # CPU에서 작업 (이미 CPU에 있지만 명시적으로)
                            crossover_state[key] = value.cpu() if torch.is_tensor(value) else value
                            updated_keys.append(key)
            
            if updated_keys:
                logger.info(f"    ✓ {module_name}: 에폭 {optimal_epoch}에서 {len(updated_keys)}개 파라미터 로드")
            else:
                logger.warning(f"    ⚠️ {module_name}: 매칭되는 파라미터 없음")
        
        # 새로운 state_dict를 모델에 로드
        try:
            model.load_state_dict(crossover_state, strict=False)
            logger.info("  ✅ Parameter Crossover 완료")
        except Exception as e:
            logger.error(f"  ❌ State dict 로드 실패: {e}")
            # 실패 시 원래 state 복원
            model.load_state_dict(current_state)
            logger.info("  ↩️ 원래 state로 복원됨")
        
        return model
    
    def _weighted_crossover(self,
                          model: nn.Module,
                          optimal_epochs: Dict[str, int]) -> nn.Module:
        """
        가중 평균 교차: 여러 에폭의 파라미터를 가중 평균
        """
        logger.info("  📊 가중 평균 교차 수행 중...")
        
        crossover_model = copy.deepcopy(model)
        
        # 각 모듈별로 처리
        for module_name in optimal_epochs.keys():
            if not hasattr(crossover_model, module_name):
                continue
            
            module = getattr(crossover_model, module_name)
            
            # 최적 에폭 주변의 체크포인트 수집
            optimal_epoch = optimal_epochs[module_name]
            nearby_epochs = [
                e for e in range(max(1, optimal_epoch - 2), optimal_epoch + 3)
                if e in self.checkpoint_pool
            ]
            
            if not nearby_epochs:
                continue
            
            # 가중치 계산 (최적 에폭에 가까울수록 높은 가중치)
            weights = []
            state_dicts = []
            
            for epoch in nearby_epochs:
                distance = abs(epoch - optimal_epoch)
                weight = 1.0 / (1.0 + distance)  # 거리에 반비례
                weights.append(weight)
                
                checkpoint = torch.load(self.checkpoint_pool[epoch]['path'], map_location='cpu')
                if 'model_state' in checkpoint and module_name in checkpoint['model_state']:
                    state_dicts.append(checkpoint['model_state'][module_name])
            
            if not state_dicts:
                continue
            
            # 가중치 정규화
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            # 가중 평균 계산
            averaged_state = {}
            for key in state_dicts[0].keys():
                averaged_state[key] = sum(
                    w * sd[key] for w, sd in zip(weights, state_dicts)
                )
            
            module.load_state_dict(averaged_state)
            logger.info(f"    ✓ {module_name}: {len(nearby_epochs)}개 에폭 가중 평균")
        
        return crossover_model
    
    def _genetic_crossover(self,
                         model: nn.Module,
                         optimal_epochs: Dict[str, int],
                         validation_fn: Optional[Any] = None) -> nn.Module:
        """
        유전 알고리즘 기반 교차
        """
        logger.info("  🧬 유전 알고리즘 교차 수행 중...")
        
        population_size = 10
        generations = 5
        
        # 초기 개체군 생성
        population = []
        for _ in range(population_size):
            individual = self._create_random_combination(model, optimal_epochs)
            population.append(individual)
        
        # 진화 과정
        for gen in range(generations):
            # 적합도 평가
            if validation_fn:
                fitness_scores = [validation_fn(ind) for ind in population]
            else:
                # 더미 적합도 (실제로는 검증 손실 등 사용)
                fitness_scores = [np.random.random() for _ in population]
            
            # 선택 (상위 50%)
            sorted_indices = np.argsort(fitness_scores)[:population_size // 2]
            parents = [population[i] for i in sorted_indices]
            
            # 교차 및 변이
            offspring = []
            for i in range(0, len(parents) - 1, 2):
                child1, child2 = self._crossover_individuals(parents[i], parents[i+1])
                child1 = self._mutate_individual(child1)
                child2 = self._mutate_individual(child2)
                offspring.extend([child1, child2])
            
            # 다음 세대
            population = parents + offspring
            
            logger.info(f"    Generation {gen+1}: Best fitness = {min(fitness_scores):.4f}")
        
        # 최고 개체 선택
        if validation_fn:
            fitness_scores = [validation_fn(ind) for ind in population]
            best_idx = np.argmin(fitness_scores)
        else:
            best_idx = 0
        
        return population[best_idx]
    
    def _interpolate_crossover(self,
                             model: nn.Module,
                             optimal_epochs: Dict[str, int]) -> nn.Module:
        """
        보간 기반 교차: 에폭 간 파라미터를 부드럽게 보간
        """
        logger.info("  🔄 보간 교차 수행 중...")
        
        crossover_model = copy.deepcopy(model)
        
        for module_name, optimal_epoch in optimal_epochs.items():
            if not hasattr(crossover_model, module_name):
                continue
            
            module = getattr(crossover_model, module_name)
            
            # 보간할 두 체크포인트 선택
            if optimal_epoch > 1 and (optimal_epoch - 1) in self.checkpoint_pool:
                epoch1 = optimal_epoch - 1
                epoch2 = optimal_epoch
            elif optimal_epoch < 60 and (optimal_epoch + 1) in self.checkpoint_pool:
                epoch1 = optimal_epoch
                epoch2 = optimal_epoch + 1
            else:
                continue
            
            # 체크포인트 로드
            cp1 = torch.load(self.checkpoint_pool[epoch1]['path'], map_location='cpu')
            cp2 = torch.load(self.checkpoint_pool[epoch2]['path'], map_location='cpu')
            
            if 'model_state' not in cp1 or 'model_state' not in cp2:
                continue
            
            if module_name not in cp1['model_state'] or module_name not in cp2['model_state']:
                continue
            
            state1 = cp1['model_state'][module_name]
            state2 = cp2['model_state'][module_name]
            
            # 선형 보간
            interpolated_state = {}
            alpha = 0.5  # 보간 비율
            
            for key in state1.keys():
                interpolated_state[key] = (1 - alpha) * state1[key] + alpha * state2[key]
            
            module.load_state_dict(interpolated_state)
            logger.info(f"    ✓ {module_name}: 에폭 {epoch1}-{epoch2} 보간")
        
        return crossover_model
    
    def _create_random_combination(self,
                                  model: nn.Module,
                                  optimal_epochs: Dict[str, int]) -> nn.Module:
        """랜덤 파라미터 조합 생성"""
        individual = copy.deepcopy(model)
        
        for module_name in optimal_epochs.keys():
            # 랜덤하게 에폭 선택
            available_epochs = list(self.checkpoint_pool.keys())
            if available_epochs:
                random_epoch = np.random.choice(available_epochs)
                
                checkpoint = torch.load(
                    self.checkpoint_pool[random_epoch]['path'],
                    map_location='cpu'
                )
                
                if hasattr(individual, module_name) and 'model_state' in checkpoint:
                    if module_name in checkpoint['model_state']:
                        module = getattr(individual, module_name)
                        try:
                            module.load_state_dict(checkpoint['model_state'][module_name])
                        except:
                            pass
        
        return individual
    
    def _crossover_individuals(self,
                             parent1: nn.Module,
                             parent2: nn.Module) -> Tuple[nn.Module, nn.Module]:
        """두 개체 교차"""
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        
        # 모듈별로 50% 확률로 교환
        for name, module in parent1.named_modules():
            if np.random.random() < 0.5:
                # child1은 parent2의 모듈을, child2는 parent1의 모듈을 받음
                if hasattr(child1, name) and hasattr(parent2, name):
                    child1_module = getattr(child1, name)
                    parent2_module = getattr(parent2, name)
                    child1_module.load_state_dict(parent2_module.state_dict())
                
                if hasattr(child2, name) and hasattr(parent1, name):
                    child2_module = getattr(child2, name)
                    parent1_module = getattr(parent1, name)
                    child2_module.load_state_dict(parent1_module.state_dict())
        
        return child1, child2
    
    def _mutate_individual(self, individual: nn.Module) -> nn.Module:
        """개체 변이"""
        for name, param in individual.named_parameters():
            if np.random.random() < self.mutation_rate:
                # 작은 노이즈 추가
                noise = torch.randn_like(param) * 0.01
                param.data.add_(noise)
        
        return individual
    
    def save_crossover_result(self,
                            model: nn.Module,
                            save_path: str,
                            metadata: Optional[Dict] = None):
        """
        교차 결과 저장
        
        Args:
            model: 교차된 모델
            save_path: 저장 경로
            metadata: 추가 메타데이터
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'crossover_strategy': self.crossover_strategy,
            'blend_ratio': self.blend_ratio,
            'module_best_epochs': self.module_best_epochs,
            'timestamp': datetime.now().isoformat()
        }
        
        if metadata:
            checkpoint['metadata'] = metadata
        
        torch.save(checkpoint, save_path)
        logger.info(f"💾 Crossover 모델 저장: {save_path}")
        
        # 메타데이터 별도 저장
        meta_file = save_path.with_suffix('.json')
        with open(meta_file, 'w') as f:
            json.dump({
                'crossover_strategy': self.crossover_strategy,
                'module_best_epochs': self.module_best_epochs,
                'timestamp': checkpoint['timestamp'],
                'metadata': metadata or {}
            }, f, indent=2)
    
    def analyze_crossover_impact(self,
                                original_model: nn.Module,
                                crossover_model: nn.Module) -> Dict[str, Any]:
        """
        교차 전후 모델 비교 분석
        
        Args:
            original_model: 원본 모델
            crossover_model: 교차된 모델
            
        Returns:
            분석 결과
        """
        analysis = {
            'parameter_changes': {},
            'magnitude_changes': {},
            'similarity_scores': {}
        }
        
        for name, param in original_model.named_parameters():
            if name in dict(crossover_model.named_parameters()):
                original_param = param.data
                crossover_param = dict(crossover_model.named_parameters())[name].data
                
                # 파라미터 변화량
                change = (crossover_param - original_param).abs().mean().item()
                analysis['parameter_changes'][name] = change
                
                # 크기 변화
                original_norm = original_param.norm().item()
                crossover_norm = crossover_param.norm().item()
                magnitude_change = (crossover_norm - original_norm) / original_norm if original_norm > 0 else 0
                analysis['magnitude_changes'][name] = magnitude_change
                
                # 코사인 유사도
                if original_param.numel() > 1:
                    original_flat = original_param.flatten()
                    crossover_flat = crossover_param.flatten()
                    cosine_sim = torch.nn.functional.cosine_similarity(
                        original_flat.unsqueeze(0),
                        crossover_flat.unsqueeze(0)
                    ).item()
                    analysis['similarity_scores'][name] = cosine_sim
        
        # 요약 통계
        analysis['summary'] = {
            'avg_parameter_change': np.mean(list(analysis['parameter_changes'].values())),
            'avg_magnitude_change': np.mean(list(analysis['magnitude_changes'].values())),
            'avg_similarity': np.mean(list(analysis['similarity_scores'].values()))
        }
        
        logger.info("📊 Crossover Impact Analysis:")
        logger.info(f"  - 평균 파라미터 변화: {analysis['summary']['avg_parameter_change']:.4f}")
        logger.info(f"  - 평균 크기 변화: {analysis['summary']['avg_magnitude_change']:.4f}")
        logger.info(f"  - 평균 유사도: {analysis['summary']['avg_similarity']:.4f}")
        
        return analysis