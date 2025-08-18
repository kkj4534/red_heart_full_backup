#!/usr/bin/env python3
"""
계층형 윤리 가중치 조정기 (Ethics Policy Updater)
Hierarchical Ethics Weight Adjuster for Red Heart AI

docs/추가 조정.txt 개선사항 구현:
- regret_score나 satisfaction_score를 감정 벡터 (VAD)의 값 조정 인자로 명시적 도입
- ethics_policy_updater.py 같은 계층형 윤리 가중치 조정기를 만들고, 경험 데이터베이스와도 직접 연결
"""

import torch
import torch.nn as nn
import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import logging
from collections import defaultdict, deque

@dataclass
class EthicsPolicyConfig:
    """윤리 정책 설정"""
    learning_rate: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 0.01
    experience_buffer_size: int = 1000
    update_frequency: int = 10
    regret_threshold: float = 0.6
    satisfaction_threshold: float = 0.7
    
    # 윤리 카테고리 가중치 (초기값)
    rule_based_weight: float = 1.0
    consequence_weight: float = 1.0  
    virtue_weight: float = 1.0
    care_weight: float = 1.0

class ExperienceDatabase:
    """경험 데이터베이스 - 윤리적 결정과 그 결과 저장"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.experiences = deque(maxlen=max_size)
        self.regret_patterns = defaultdict(list)
        self.satisfaction_patterns = defaultdict(list)
        
    def add_experience(self, experience: Dict[str, Any]):
        """경험 추가"""
        experience['timestamp'] = datetime.now().isoformat()
        self.experiences.append(experience)
        
        # 패턴별 분류
        decision_type = experience.get('decision_type', 'unknown')
        regret_score = experience.get('regret_score', 0.0)
        satisfaction_score = experience.get('satisfaction_score', 0.0)
        
        if regret_score > 0.5:
            self.regret_patterns[decision_type].append(experience)
        if satisfaction_score > 0.5:
            self.satisfaction_patterns[decision_type].append(experience)
    
    def get_regret_patterns(self, decision_type: str = None, limit: int = 100) -> List[Dict]:
        """후회 패턴 조회"""
        if decision_type:
            return list(self.regret_patterns[decision_type])[-limit:]
        else:
            all_patterns = []
            for patterns in self.regret_patterns.values():
                all_patterns.extend(patterns)
            return all_patterns[-limit:]
    
    def get_satisfaction_patterns(self, decision_type: str = None, limit: int = 100) -> List[Dict]:
        """만족 패턴 조회"""
        if decision_type:
            return list(self.satisfaction_patterns[decision_type])[-limit:]
        else:
            all_patterns = []
            for patterns in self.satisfaction_patterns.values():
                all_patterns.extend(patterns)
            return all_patterns[-limit:]
    
    def analyze_trends(self) -> Dict[str, Any]:
        """경험 트렌드 분석"""
        if not self.experiences:
            return {}
        
        recent_experiences = list(self.experiences)[-100:]  # 최근 100개
        
        # 후회/만족 점수 평균
        regret_scores = [exp.get('regret_score', 0.0) for exp in recent_experiences]
        satisfaction_scores = [exp.get('satisfaction_score', 0.0) for exp in recent_experiences]
        
        # 결정 유형별 분석
        decision_counts = defaultdict(int)
        decision_regrets = defaultdict(list)
        decision_satisfactions = defaultdict(list)
        
        for exp in recent_experiences:
            dec_type = exp.get('decision_type', 'unknown')
            decision_counts[dec_type] += 1
            decision_regrets[dec_type].append(exp.get('regret_score', 0.0))
            decision_satisfactions[dec_type].append(exp.get('satisfaction_score', 0.0))
        
        return {
            'total_experiences': len(recent_experiences),
            'avg_regret': np.mean(regret_scores) if regret_scores else 0.0,
            'avg_satisfaction': np.mean(satisfaction_scores) if satisfaction_scores else 0.0,
            'decision_types': dict(decision_counts),
            'decision_regret_avg': {k: np.mean(v) for k, v in decision_regrets.items()},
            'decision_satisfaction_avg': {k: np.mean(v) for k, v in decision_satisfactions.items()}
        }

class EthicsPolicyUpdater(nn.Module):
    """계층형 윤리 가중치 조정기"""
    
    def __init__(self, config: EthicsPolicyConfig):
        super().__init__()
        self.config = config
        self.experience_db = ExperienceDatabase(config.experience_buffer_size)
        self.update_count = 0
        
        # 윤리 카테고리 가중치 (학습 가능한 파라미터)
        self.ethics_weights = nn.Parameter(torch.tensor([
            config.rule_based_weight,    # 규칙 기반 윤리
            config.consequence_weight,   # 결과 기반 윤리  
            config.virtue_weight,        # 덕윤리
            config.care_weight          # 배려 윤리
        ]))
        
        # VAD 조정 네트워크 (regret/satisfaction → VAD 변화)
        self.vad_adjustment_net = nn.Sequential(
            nn.Linear(2, 16),  # regret_score, satisfaction_score 입력
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(), 
            nn.Linear(8, 3)    # VAD 조정값 출력
        )
        
        # 윤리 가중치 조정 네트워크
        self.weight_adjustment_net = nn.Sequential(
            nn.Linear(6, 32),  # VAD + regret + satisfaction + decision_outcome
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4)   # 4개 윤리 카테고리 조정값
        )
        
        # 옵티마이저
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.logger = logging.getLogger('EthicsPolicyUpdater')
    
    def record_decision_outcome(self, vad_vector: torch.Tensor, decision_type: str,
                              regret_score: float, satisfaction_score: float,
                              decision_outcome: str):
        """윤리적 결정과 그 결과 기록"""
        experience = {
            'vad_vector': vad_vector.detach().cpu().numpy().tolist(),
            'decision_type': decision_type,
            'regret_score': regret_score,
            'satisfaction_score': satisfaction_score,
            'decision_outcome': decision_outcome,
            'ethics_weights': self.ethics_weights.detach().cpu().numpy().tolist()
        }
        
        self.experience_db.add_experience(experience)
        self.logger.debug(f"경험 기록: {decision_type}, 후회={regret_score:.3f}, 만족={satisfaction_score:.3f}")
    
    def calculate_vad_adjustment(self, regret_score: float, satisfaction_score: float) -> torch.Tensor:
        """regret_score와 satisfaction_score를 VAD 조정 인자로 변환"""
        input_tensor = torch.tensor([regret_score, satisfaction_score], dtype=torch.float32)
        
        with torch.no_grad():
            vad_adjustment = self.vad_adjustment_net(input_tensor)
        
        return vad_adjustment
    
    def update_ethics_weights(self, recent_experiences: List[Dict[str, Any]]) -> Dict[str, float]:
        """경험 기반 윤리 가중치 업데이트"""
        if len(recent_experiences) < 5:  # 최소 경험 수
            return self._get_current_weights()
        
        # 경험에서 특징 추출
        features = []
        targets = []
        
        for exp in recent_experiences:
            vad = exp.get('vad_vector', [0, 0, 0])
            regret = exp.get('regret_score', 0.0)
            satisfaction = exp.get('satisfaction_score', 0.0)
            outcome = 1.0 if exp.get('decision_outcome') == 'positive' else 0.0
            
            # 특징: VAD + regret + satisfaction + outcome
            feature = vad + [regret, satisfaction, outcome]
            features.append(feature)
            
            # 타겟: 이상적인 윤리 가중치 (후회 최소화, 만족 최대화 방향)
            target_weights = self._calculate_ideal_weights(regret, satisfaction, exp.get('decision_type', ''))
            targets.append(target_weights)
        
        if not features:
            return self._get_current_weights()
        
        # 학습 데이터 준비
        features_tensor = torch.tensor(features, dtype=torch.float32)
        targets_tensor = torch.tensor(targets, dtype=torch.float32)
        
        # 가중치 조정 학습
        self.optimizer.zero_grad()
        
        predicted_adjustments = self.weight_adjustment_net(features_tensor)
        current_weights = self.ethics_weights.unsqueeze(0).expand(len(features), -1)
        predicted_weights = current_weights + predicted_adjustments * 0.1  # 점진적 조정
        
        # 손실 계산 (예측된 가중치와 이상적 가중치 간의 차이)
        loss = nn.MSELoss()(predicted_weights, targets_tensor)
        
        loss.backward()
        self.optimizer.step()
        
        # 가중치 업데이트 (클램핑으로 안정성 확보)
        with torch.no_grad():
            avg_adjustment = predicted_adjustments.mean(dim=0)
            self.ethics_weights.data += avg_adjustment * 0.05  # 매우 점진적 업데이트
            self.ethics_weights.data = torch.clamp(self.ethics_weights.data, 0.1, 3.0)
        
        self.update_count += 1
        self.logger.info(f"윤리 가중치 업데이트 완료 (#{self.update_count}), 손실: {loss.item():.4f}")
        
        return self._get_current_weights()
    
    def _calculate_ideal_weights(self, regret: float, satisfaction: float, decision_type: str) -> List[float]:
        """후회/만족 점수 기반 이상적 윤리 가중치 계산"""
        base_weights = [1.0, 1.0, 1.0, 1.0]  # rule, consequence, virtue, care
        
        # 높은 후회 → 해당 결정 유형의 반대 방향 강화
        if regret > self.config.regret_threshold:
            if decision_type == 'rule_violation':
                base_weights[0] += 0.3  # 규칙 기반 강화
            elif decision_type == 'bad_outcome':
                base_weights[1] += 0.3  # 결과 기반 강화
            elif decision_type == 'character_flaw':
                base_weights[2] += 0.3  # 덕윤리 강화
            elif decision_type == 'lack_of_care':
                base_weights[3] += 0.3  # 배려 윤리 강화
        
        # 높은 만족 → 해당 결정 유형 유지/강화
        if satisfaction > self.config.satisfaction_threshold:
            if decision_type == 'altruistic':
                base_weights[3] += 0.2  # 배려 윤리 강화
            elif decision_type == 'principled':
                base_weights[0] += 0.2  # 규칙 기반 강화
            elif decision_type == 'pragmatic':
                base_weights[1] += 0.2  # 결과 기반 강화
            elif decision_type == 'virtuous':
                base_weights[2] += 0.2  # 덕윤리 강화
        
        return base_weights
    
    def _get_current_weights(self) -> Dict[str, float]:
        """현재 윤리 가중치 반환"""
        return {
            'rule_based_weight': float(self.ethics_weights[0]),
            'consequence_weight': float(self.ethics_weights[1]),
            'virtue_weight': float(self.ethics_weights[2]),
            'care_weight': float(self.ethics_weights[3])
        }
    
    def get_adjusted_vad(self, original_vad: torch.Tensor, regret_score: float, 
                        satisfaction_score: float) -> torch.Tensor:
        """VAD 벡터를 regret/satisfaction 점수로 조정"""
        adjustment = self.calculate_vad_adjustment(regret_score, satisfaction_score)
        
        # 배치 처리를 위한 차원 맞춤
        if original_vad.dim() == 2:
            adjustment = adjustment.unsqueeze(0).expand(original_vad.size(0), -1)
        
        adjusted_vad = original_vad + adjustment * 0.1  # 점진적 조정
        adjusted_vad = torch.clamp(adjusted_vad, -1.0, 1.0)  # VAD 범위 제한
        
        return adjusted_vad
    
    def should_update(self) -> bool:
        """업데이트 수행 여부 판단"""
        return len(self.experience_db.experiences) >= self.config.update_frequency
    
    def perform_policy_update(self) -> Dict[str, Any]:
        """정책 업데이트 수행"""
        if not self.should_update():
            return {'updated': False, 'reason': 'insufficient_experiences'}
        
        # 최근 경험들 가져오기
        recent_experiences = list(self.experience_db.experiences)[-50:]  # 최근 50개
        
        # 윤리 가중치 업데이트
        updated_weights = self.update_ethics_weights(recent_experiences)
        
        # 트렌드 분석
        trends = self.experience_db.analyze_trends()
        
        return {
            'updated': True,
            'new_weights': updated_weights,
            'trends': trends,
            'update_count': self.update_count,
            'experiences_used': len(recent_experiences)
        }
    
    def save_policy(self, path: Path):
        """정책 저장"""
        policy_data = {
            'ethics_weights': self.ethics_weights.detach().cpu().numpy().tolist(),
            'config': asdict(self.config),
            'update_count': self.update_count,
            'model_state_dict': self.state_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(policy_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"윤리 정책 저장: {path}")
    
    def load_policy(self, path: Path):
        """정책 로드"""
        with open(path, 'r', encoding='utf-8') as f:
            policy_data = json.load(f)
        
        # 가중치 복원
        self.ethics_weights.data = torch.tensor(policy_data['ethics_weights'])
        self.update_count = policy_data.get('update_count', 0)
        
        # 모델 상태 복원
        if 'model_state_dict' in policy_data:
            self.load_state_dict(policy_data['model_state_dict'])
        
        self.logger.info(f"윤리 정책 로드: {path}")

def create_ethics_policy_updater(config: Optional[EthicsPolicyConfig] = None) -> EthicsPolicyUpdater:
    """윤리 정책 업데이터 생성 헬퍼 함수"""
    if config is None:
        config = EthicsPolicyConfig()
    
    return EthicsPolicyUpdater(config)

# 사용 예시
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 윤리 정책 업데이터 생성
    config = EthicsPolicyConfig(
        learning_rate=0.001,
        experience_buffer_size=1000,
        update_frequency=20
    )
    
    updater = create_ethics_policy_updater(config)
    
    # 가상의 경험 기록
    vad_sample = torch.tensor([[0.5, 0.3, 0.7]])  # 예시 VAD 벡터
    
    updater.record_decision_outcome(
        vad_vector=vad_sample,
        decision_type='altruistic',
        regret_score=0.2,
        satisfaction_score=0.8,
        decision_outcome='positive'
    )
    
    # VAD 조정 예시
    adjusted_vad = updater.get_adjusted_vad(vad_sample, 0.3, 0.7)
    print(f"원본 VAD: {vad_sample}")
    print(f"조정된 VAD: {adjusted_vad}")
    
    # 현재 윤리 가중치 출력
    current_weights = updater._get_current_weights()
    print(f"현재 윤리 가중치: {current_weights}")