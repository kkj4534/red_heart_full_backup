"""
Bentham v2 Calculator - 외부비용, 재분배효과, 자아손상 계산 로직
Advanced logic for calculating External Cost, Redistribution Effect, and Self-damage
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger('RedHeart.BenthamV2Calculator')

@dataclass
class BenthamV2Variables:
    """Bentham v2 확장 변수들"""
    external_cost: float = 0.0
    redistribution_effect: float = 0.0
    self_damage: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'external_cost': self.external_cost,
            'redistribution_effect': self.redistribution_effect,
            'self_damage': self.self_damage
        }

class BenthamV2Calculator:
    """Bentham v2 확장 변수 계산기"""
    
    def __init__(self):
        self.logger = logger
        
    def calculate_external_cost(self, context: Dict[str, Any]) -> float:
        """외부비용 계산 (E)
        
        - 환경 영향, 사회적 비용, 미래 세대 부담 등 고려
        - 범위: 0.0 (외부비용 없음) ~ 1.0 (높은 외부비용)
        """
        cost_factors = []
        
        # 1. 환경 영향도
        if 'environmental_impact' in context:
            env_impact = context['environmental_impact']
            cost_factors.append(env_impact * 0.3)
        
        # 2. 사회적 파급효과
        affected_count = context.get('affected_count', 1)
        if affected_count > 10:
            social_cost = min(np.log(affected_count) / np.log(1000), 1.0)
            cost_factors.append(social_cost * 0.4)
        
        # 3. 장기적 결과 비용
        duration = context.get('duration_seconds', 0)
        if duration > 86400:  # 1일 이상
            long_term_cost = min(duration / (365 * 86400), 1.0)  # 1년 기준 정규화
            cost_factors.append(long_term_cost * 0.3)
        
        # 4. 윤리적 딜레마 복잡도
        ethical_complexity = context.get('ethical_complexity', 0.5)
        cost_factors.append(ethical_complexity * 0.2)
        
        # 5. 불확실성 비용
        uncertainty = context.get('uncertainty_level', 0.5)
        if uncertainty > 0.7:
            uncertainty_cost = (uncertainty - 0.7) / 0.3
            cost_factors.append(uncertainty_cost * 0.2)
        
        # 가중 평균 계산
        if cost_factors:
            external_cost = np.mean(cost_factors)
        else:
            external_cost = 0.5  # 기본값
        
        return min(max(external_cost, 0.0), 1.0)
    
    def calculate_redistribution_effect(self, context: Dict[str, Any]) -> float:
        """재분배효과 계산 (R)
        
        - 부의 재분배, 권력 이동, 사회적 이동성 등 고려
        - 범위: 0.0 (재분배 없음) ~ 1.0 (강한 재분배)
        """
        redistribution_factors = []
        
        # 1. 경제적 재분배
        if 'economic_redistribution' in context:
            econ_redistribution = context['economic_redistribution']
            redistribution_factors.append(econ_redistribution * 0.4)
        
        # 2. 권력 구조 변화
        if 'power_structure_change' in context:
            power_change = context['power_structure_change']
            redistribution_factors.append(power_change * 0.3)
        
        # 3. 사회적 형평성 개선
        equity_improvement = context.get('equity_improvement', 0.5)
        redistribution_factors.append(equity_improvement * 0.3)
        
        # 4. 취약계층 보호 효과
        vulnerable_protection = context.get('vulnerable_protection', 0.5)
        redistribution_factors.append(vulnerable_protection * 0.2)
        
        # 5. 사회적 계층 이동성
        social_mobility = context.get('social_mobility', 0.5)
        redistribution_factors.append(social_mobility * 0.1)
        
        # 가중 평균 계산
        if redistribution_factors:
            redistribution_effect = np.mean(redistribution_factors)
        else:
            redistribution_effect = 0.5  # 기본값
        
        return min(max(redistribution_effect, 0.0), 1.0)
    
    def calculate_self_damage(self, context: Dict[str, Any]) -> float:
        """자아손상 계산 (S)
        
        - 정신적 피해, 도덕적 손상, 자아정체성 훼손 등 고려
        - 범위: 0.0 (자아손상 없음) ~ 1.0 (심각한 자아손상)
        """
        damage_factors = []
        
        # 1. 도덕적 갈등 강도
        moral_conflict = context.get('moral_conflict', 0.5)
        damage_factors.append(moral_conflict * 0.3)
        
        # 2. 가치관 충돌 정도
        value_conflict = context.get('value_conflict', 0.5)
        damage_factors.append(value_conflict * 0.3)
        
        # 3. 정신적 스트레스
        mental_stress = context.get('mental_stress', 0.5)
        damage_factors.append(mental_stress * 0.2)
        
        # 4. 자아정체성 위협
        identity_threat = context.get('identity_threat', 0.5)
        damage_factors.append(identity_threat * 0.2)
        
        # 5. 장기적 심리적 영향
        if 'long_term_psychological_impact' in context:
            psychological_impact = context['long_term_psychological_impact']
            damage_factors.append(psychological_impact * 0.3)
        
        # 6. 사회적 관계 손상
        relationship_damage = context.get('relationship_damage', 0.5)
        damage_factors.append(relationship_damage * 0.1)
        
        # 가중 평균 계산
        if damage_factors:
            self_damage = np.mean(damage_factors)
        else:
            self_damage = 0.5  # 기본값
        
        return min(max(self_damage, 0.0), 1.0)
    
    def calculate_surd_based_weights(self, surd_graph: Optional[Any]) -> Dict[str, float]:
        """SURD Graph centrality 기반 동적 가중치 계산
        
        Args:
            surd_graph: SURD 분석 결과 그래프
            
        Returns:
            Dict[str, float]: 변수별 동적 가중치
        """
        if surd_graph is None:
            # 기본 가중치 반환
            return {
                'external_cost': 0.10,
                'redistribution_effect': 0.05,
                'self_damage': 0.05
            }
        
        # SURD Graph에서 중요도 추출
        try:
            # 노드 중요도 분석
            node_importance = {}
            if hasattr(surd_graph, 'nodes'):
                for node in surd_graph.nodes():
                    # 중심성 계산 (degree, betweenness, closeness)
                    degree_centrality = surd_graph.degree(node)
                    node_importance[node] = degree_centrality
            
            # 변수별 중요도 매핑
            external_cost_importance = node_importance.get('external_effects', 0.5)
            redistribution_importance = node_importance.get('redistribution', 0.5)
            self_damage_importance = node_importance.get('self_impact', 0.5)
            
            # 정규화 (총 합이 0.2가 되도록)
            total_importance = external_cost_importance + redistribution_importance + self_damage_importance
            if total_importance > 0:
                normalization_factor = 0.2 / total_importance
                return {
                    'external_cost': external_cost_importance * normalization_factor,
                    'redistribution_effect': redistribution_importance * normalization_factor,
                    'self_damage': self_damage_importance * normalization_factor
                }
            else:
                return {
                    'external_cost': 0.10,
                    'redistribution_effect': 0.05,
                    'self_damage': 0.05
                }
                
        except Exception as e:
            logger.warning(f"SURD 기반 가중치 계산 실패: {e}")
            return {
                'external_cost': 0.10,
                'redistribution_effect': 0.05,
                'self_damage': 0.05
            }
    
    def calculate_bentham_v2_variables(self, context: Dict[str, Any], 
                                     surd_graph: Optional[Any] = None) -> BenthamV2Variables:
        """Bentham v2 확장 변수들 종합 계산
        
        Args:
            context: 계산 컨텍스트
            surd_graph: SURD 분석 결과 (선택적)
            
        Returns:
            BenthamV2Variables: 계산된 확장 변수들
        """
        try:
            # 동적 가중치 계산
            dynamic_weights = self.calculate_surd_based_weights(surd_graph)
            
            # 각 변수 계산
            external_cost = self.calculate_external_cost(context)
            redistribution_effect = self.calculate_redistribution_effect(context)
            self_damage = self.calculate_self_damage(context)
            
            # 동적 가중치 적용
            external_cost *= dynamic_weights['external_cost'] * 10  # 가중치 증폭
            redistribution_effect *= dynamic_weights['redistribution_effect'] * 20  # 가중치 증폭
            self_damage *= dynamic_weights['self_damage'] * 20  # 가중치 증폭
            
            # 범위 제한
            external_cost = min(max(external_cost, 0.0), 1.0)
            redistribution_effect = min(max(redistribution_effect, 0.0), 1.0)
            self_damage = min(max(self_damage, 0.0), 1.0)
            
            return BenthamV2Variables(
                external_cost=external_cost,
                redistribution_effect=redistribution_effect,
                self_damage=self_damage
            )
            
        except Exception as e:
            logger.error(f"Bentham v2 변수 계산 실패: {e}")
            return BenthamV2Variables(
                external_cost=0.5,
                redistribution_effect=0.5,
                self_damage=0.5
            )

# 전역 계산기 인스턴스
bentham_v2_calculator = BenthamV2Calculator()