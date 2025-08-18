"""
Three-View Scenario System for Red Heart AI
벤담 계산과 후회 분석을 위한 3뷰 시나리오 시스템

핵심 기능:
1. 낙관적/중도적/비관적 시나리오 생성
2. 벤담 계산에서 시나리오별 쾌락 계산
3. 후회 분석에서 시나리오 고려
4. 통계적 분포 기반 결과 예측
5. MoE 시스템과의 통합
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from datetime import datetime
import asyncio
from scipy import stats

logger = logging.getLogger('RedHeart.ThreeViewScenarioSystem')

class ScenarioType(Enum):
    """시나리오 유형"""
    OPTIMISTIC = "optimistic"      # 낙관적 시나리오 (μ+σ)
    NEUTRAL = "neutral"            # 중도적 시나리오 (μ)
    PESSIMISTIC = "pessimistic"    # 비관적 시나리오 (μ-σ)

@dataclass
class ScenarioMetrics:
    """시나리오별 메트릭"""
    scenario_type: ScenarioType
    probability_weight: float      # 시나리오 발생 확률
    
    # 벤담 계산 관련
    expected_pleasure: float       # 예상 쾌락
    expected_pain: float          # 예상 고통  
    utility_score: float          # 효용 점수
    
    # 후회 분석 관련
    regret_potential: float       # 후회 가능성
    regret_intensity: float       # 후회 강도
    
    # 윤리적 고려사항
    ethical_implications: Dict[str, float] = field(default_factory=dict)
    
    # 위험 및 기회 요소
    risk_factors: List[str] = field(default_factory=list)
    opportunity_factors: List[str] = field(default_factory=list)
    
    # 메타데이터
    confidence_level: float = 0.7
    uncertainty_level: float = 0.3
    generation_timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ThreeViewAnalysisResult:
    """3뷰 분석 결과"""
    optimistic_scenario: ScenarioMetrics
    neutral_scenario: ScenarioMetrics
    pessimistic_scenario: ScenarioMetrics
    
    # 통합 결과
    consensus_utility: float        # 합의 효용
    consensus_regret: float        # 합의 후회 점수
    uncertainty_range: Tuple[float, float]  # 불확실성 범위
    
    # 권장사항
    recommended_decision: str
    risk_mitigation_strategies: List[str]
    opportunity_enhancement_strategies: List[str]
    
    # 분석 품질 지표
    scenario_diversity: float      # 시나리오 다양성
    consensus_strength: float      # 합의 강도
    
    # 메타데이터
    analysis_duration_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class ScenarioDistributionModel:
    """시나리오 분포 모델"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        
        # 분포 추정 네트워크
        self.distribution_net = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 9)  # μ, σ, skew for pleasure, pain, regret
        ).to(self.device)
        
        # 시나리오 가중치 네트워크
        self.weight_net = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # 3개 시나리오 가중치
            nn.Softmax(dim=-1)
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            list(self.distribution_net.parameters()) + list(self.weight_net.parameters()), 
            lr=1e-4
        )
        
    def estimate_scenario_distributions(self, context_embedding: torch.Tensor) -> Dict[str, Dict[str, float]]:
        """컨텍스트 기반 시나리오 분포 추정"""
        self.distribution_net.eval()
        self.weight_net.eval()
        
        with torch.no_grad():
            # 분포 파라미터 추정
            dist_params = self.distribution_net(context_embedding)
            scenario_weights = self.weight_net(context_embedding)
            
            # 쾌락, 고통, 후회에 대한 분포 파라미터
            pleasure_mu = torch.sigmoid(dist_params[0]).item()
            pleasure_sigma = torch.sigmoid(dist_params[1]).item() * 0.3
            pleasure_skew = torch.tanh(dist_params[2]).item()
            
            pain_mu = torch.sigmoid(dist_params[3]).item()
            pain_sigma = torch.sigmoid(dist_params[4]).item() * 0.3
            pain_skew = torch.tanh(dist_params[5]).item()
            
            regret_mu = torch.sigmoid(dist_params[6]).item()
            regret_sigma = torch.sigmoid(dist_params[7]).item() * 0.3
            regret_skew = torch.tanh(dist_params[8]).item()
            
            # 시나리오별 가중치
            opt_weight, neu_weight, pes_weight = scenario_weights.tolist()
            
            # 3뷰 시나리오 생성
            scenarios = {}
            
            # 낙관적 시나리오 (μ+σ)
            scenarios['optimistic'] = {
                'pleasure': min(1.0, pleasure_mu + pleasure_sigma * (1 + abs(pleasure_skew))),
                'pain': max(0.0, pain_mu - pain_sigma * (1 + abs(pain_skew))),
                'regret': max(0.0, regret_mu - regret_sigma * (1 + abs(regret_skew))),
                'weight': opt_weight,
                'confidence': 0.7 + abs(pleasure_skew) * 0.2
            }
            
            # 중도적 시나리오 (μ)
            scenarios['neutral'] = {
                'pleasure': pleasure_mu,
                'pain': pain_mu,
                'regret': regret_mu,
                'weight': neu_weight,
                'confidence': 0.9  # 중도적 시나리오는 높은 신뢰도
            }
            
            # 비관적 시나리오 (μ-σ)
            scenarios['pessimistic'] = {
                'pleasure': max(0.0, pleasure_mu - pleasure_sigma * (1 + abs(pleasure_skew))),
                'pain': min(1.0, pain_mu + pain_sigma * (1 + abs(pain_skew))),
                'regret': min(1.0, regret_mu + regret_sigma * (1 + abs(regret_skew))),
                'weight': pes_weight,
                'confidence': 0.7 + abs(pain_skew) * 0.2
            }
            
            return scenarios

class ThreeViewScenarioSystem:
    """3뷰 시나리오 시스템"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        
        # GPU 메모리 관리자
        try:
            from dynamic_gpu_manager import DynamicGPUManager
            self.gpu_manager = DynamicGPUManager()
            self.gpu_optimization_enabled = True
            logger.info("3뷰 시나리오 시스템 GPU 메모리 관리 활성화")
        except ImportError:
            self.gpu_manager = None
            self.gpu_optimization_enabled = False
            logger.warning("Dynamic GPU Manager를 찾을 수 없습니다. 기본 메모리 관리 사용")
        
        # 온디맨드 로딩을 위한 모델 상태
        self.distribution_model = None
        self.is_model_loaded = False
        
        # 도메인별 위험/기회 요소
        self.domain_factors = {
            'education': {
                'risks': ['학습 효과 미흡', '시간 낭비', '잘못된 방향', '기회비용'],
                'opportunities': ['지식 습득', '역량 개발', '네트워크 구축', '성장 기회']
            },
            'business': {
                'risks': ['재정 손실', '시장 변화', '경쟁 심화', '규제 변화'],
                'opportunities': ['수익 증대', '시장 확대', '혁신 기회', '파트너십']
            },
            'social': {
                'risks': ['관계 악화', '신뢰 손상', '사회적 고립', '평판 훼손'],
                'opportunities': ['관계 개선', '영향력 확대', '네트워크 성장', '사회적 기여']
            },
            'personal': {
                'risks': ['스트레스 증가', '건강 악화', '만족도 감소', '자아 손상'],
                'opportunities': ['자아실현', '행복 증진', '성취감', '개인 성장']
            },
            'health': {
                'risks': ['건강 악화', '부작용', '의료비 증가', '생활 제약'],
                'opportunities': ['건강 개선', '삶의 질 향상', '활동성 증가', '장수']
            }
        }
        
        # 시나리오 히스토리 (학습용)
        self.scenario_history = []
        
        logger.info("3뷰 시나리오 시스템 초기화 완료 (온디맨드 로딩 방식)")
    
    def _load_model(self):
        """모델 온디맨드 로딩"""
        if self.is_model_loaded:
            return
            
        try:
            if self.gpu_optimization_enabled:
                # GPU 메모리 관리자를 통한 로딩
                with self.gpu_manager.allocate_memory('three_view_scenario_system', dynamic_boost=True):
                    self.distribution_model = ScenarioDistributionModel(self.device)
                    self.is_model_loaded = True
                    logger.info("3뷰 시나리오 시스템 모델 로딩 완료 (GPU 최적화)")
            else:
                # 기본 로딩
                self.distribution_model = ScenarioDistributionModel(self.device)
                self.is_model_loaded = True
                logger.info("3뷰 시나리오 시스템 모델 로딩 완료 (기본 방식)")
                
        except Exception as e:
            logger.error(f"3뷰 시나리오 시스템 모델 로딩 실패: {e}")
            self.is_model_loaded = False
            self.distribution_model = None
            
    def _unload_model(self):
        """모델 언로딩"""
        if not self.is_model_loaded:
            return
            
        try:
            if self.distribution_model:
                # 메모리 정리
                if hasattr(self.distribution_model, 'distribution_net'):
                    del self.distribution_model.distribution_net
                if hasattr(self.distribution_model, 'weight_net'):
                    del self.distribution_model.weight_net
                if hasattr(self.distribution_model, 'optimizer'):
                    del self.distribution_model.optimizer
                    
                del self.distribution_model
                self.distribution_model = None
                
            # GPU 메모리 정리
            if self.device == 'cuda':
                import torch
                torch.cuda.empty_cache()
                
            self.is_model_loaded = False
            logger.info("3뷰 시나리오 시스템 모델 언로딩 완료")
            
        except Exception as e:
            logger.error(f"3뷰 시나리오 시스템 모델 언로딩 실패: {e}")
        
    def _extract_context_features(self, input_data: Dict[str, Any]) -> torch.Tensor:
        """컨텍스트 특성 추출"""
        features = []
        
        # 1. 텍스트 기반 특성
        text = input_data.get('text', '') or input_data.get('description', '')
        if text:
            text_lower = text.lower()
            
            # 감정 키워드
            emotion_keywords = ['기쁨', '슬픔', '분노', '두려움', '놀람', '혐오', '신뢰', '기대']
            for keyword in emotion_keywords:
                features.append(float(keyword in text_lower))
            
            # 시간 키워드
            time_keywords = ['즉시', '빠르게', '천천히', '장기적', '단기적', '영구적', '일시적']
            for keyword in time_keywords:
                features.append(float(keyword in text_lower))
            
            # 확실성 키워드
            certainty_keywords = ['확실', '불확실', '아마도', '틀림없이', '의심', '확신']
            for keyword in certainty_keywords:
                features.append(float(keyword in text_lower))
            
            # 영향 범위 키워드
            scope_keywords = ['개인적', '사회적', '공적', '사적', '집단적', '전체적']
            for keyword in scope_keywords:
                features.append(float(keyword in text_lower))
        else:
            features.extend([0] * 28)  # 텍스트 없음
        
        # 2. 벤담 변수들
        bentham_vars = [
            'intensity', 'duration', 'certainty', 'propinquity',
            'productivity', 'purity', 'extent'
        ]
        
        for var in bentham_vars:
            value = input_data.get(var, 0.5)
            features.append(float(value))
        
        # 3. 감정 데이터
        if 'emotion_data' in input_data:
            emotion_data = input_data['emotion_data']
            if hasattr(emotion_data, 'primary_emotion'):
                # 감정 원-핫 인코딩
                emotions = ['joy', 'sadness', 'anger', 'fear', 'trust', 'disgust', 'surprise', 'anticipation']
                primary = str(emotion_data.primary_emotion).lower()
                for emotion in emotions:
                    features.append(float(emotion in primary))
                
                # 감정 강도
                intensity = getattr(emotion_data, 'intensity', 3)
                if hasattr(intensity, 'value'):
                    intensity = intensity.value
                features.append(float(intensity) / 6.0)
                
                # 감정 신뢰도
                confidence = getattr(emotion_data, 'confidence', 0.5)
                features.append(float(confidence))
            else:
                features.extend([0] * 10)
        else:
            features.extend([0] * 10)
        
        # 4. 윤리적 가치들
        if 'ethical_values' in input_data:
            ethical_values = input_data['ethical_values']
            ethics = ['care_harm', 'fairness', 'loyalty', 'authority', 'sanctity', 'liberty']
            for ethic in ethics:
                features.append(float(ethical_values.get(ethic, 0.5)))
        else:
            features.extend([0.5] * 6)
        
        # 5. 추가 컨텍스트 특성
        additional_features = [
            input_data.get('complexity', 0.5),
            input_data.get('urgency', 0.5),
            input_data.get('reversibility', 0.5),
            input_data.get('social_impact', 0.5),
            input_data.get('personal_impact', 0.5),
            input_data.get('risk_level', 0.5),
            input_data.get('opportunity_level', 0.5)
        ]
        features.extend(additional_features)
        
        # 패딩하여 512차원으로 맞춤
        while len(features) < 512:
            features.append(0.0)
        
        return torch.tensor(features[:512], dtype=torch.float32, device=self.device)
    
    def _identify_domain(self, input_data: Dict[str, Any]) -> str:
        """도메인 식별"""
        text = input_data.get('text', '') or input_data.get('description', '')
        if not text:
            return 'personal'
        
        text_lower = text.lower()
        
        domain_keywords = {
            'education': ['학습', '공부', '교육', '수업', '과목', '시험', '학교', '대학'],
            'business': ['일', '사업', '회사', '프로젝트', '업무', '투자', '수익', '사업'],
            'social': ['관계', '친구', '가족', '사람', '소통', '만남', '사회', '공동체'],
            'health': ['건강', '병원', '의료', '치료', '운동', '식단', '약', '질병'],
            'personal': ['개인', '자신', '나', '내', '취미', '여가', '성장', '목표']
        }
        
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            domain_scores[domain] = score
        
        # 가장 높은 점수를 가진 도메인 반환
        best_domain = max(domain_scores, key=domain_scores.get)
        return best_domain if domain_scores[best_domain] > 0 else 'personal'
    
    def _generate_scenario_metrics(self, scenario_type: ScenarioType, 
                                 scenario_data: Dict[str, float],
                                 domain: str, 
                                 input_data: Dict[str, Any]) -> ScenarioMetrics:
        """시나리오 메트릭 생성"""
        
        # 도메인별 위험/기회 요소
        domain_info = self.domain_factors.get(domain, self.domain_factors['personal'])
        
        # 시나리오 유형별 특성
        if scenario_type == ScenarioType.OPTIMISTIC:
            risk_factors = domain_info['risks'][:2]  # 위험 요소 적게
            opportunity_factors = domain_info['opportunities']  # 기회 요소 많이
        elif scenario_type == ScenarioType.PESSIMISTIC:
            risk_factors = domain_info['risks']  # 위험 요소 많이
            opportunity_factors = domain_info['opportunities'][:1]  # 기회 요소 적게
        else:  # NEUTRAL
            risk_factors = domain_info['risks'][:len(domain_info['risks'])//2]
            opportunity_factors = domain_info['opportunities'][:len(domain_info['opportunities'])//2]
        
        # 효용 점수 계산
        utility_score = scenario_data['pleasure'] - scenario_data['pain']
        
        # 윤리적 함의 계산
        ethical_implications = {}
        if 'ethical_values' in input_data:
            ethical_values = input_data['ethical_values']
            
            # 시나리오별 윤리적 가중치 조정
            for ethic, base_value in ethical_values.items():
                if scenario_type == ScenarioType.OPTIMISTIC:
                    # 낙관적 시나리오: 긍정적 윤리 가치 강화
                    if ethic in ['care_harm', 'fairness', 'liberty']:
                        ethical_implications[ethic] = min(1.0, base_value * 1.2)
                    else:
                        ethical_implications[ethic] = base_value
                elif scenario_type == ScenarioType.PESSIMISTIC:
                    # 비관적 시나리오: 부정적 결과 고려
                    if ethic in ['care_harm', 'fairness']:
                        ethical_implications[ethic] = max(0.0, base_value * 0.8)
                    else:
                        ethical_implications[ethic] = base_value
                else:  # NEUTRAL
                    ethical_implications[ethic] = base_value
        
        return ScenarioMetrics(
            scenario_type=scenario_type,
            probability_weight=scenario_data['weight'],
            expected_pleasure=scenario_data['pleasure'],
            expected_pain=scenario_data['pain'],
            utility_score=utility_score,
            regret_potential=scenario_data['regret'],
            regret_intensity=scenario_data['regret'] * 0.8,  # 강도는 약간 낮게
            ethical_implications=ethical_implications,
            risk_factors=risk_factors,
            opportunity_factors=opportunity_factors,
            confidence_level=scenario_data['confidence'],
            uncertainty_level=1.0 - scenario_data['confidence']
        )
    
    def _calculate_consensus_metrics(self, optimistic: ScenarioMetrics,
                                   neutral: ScenarioMetrics,
                                   pessimistic: ScenarioMetrics) -> Dict[str, float]:
        """합의 메트릭 계산"""
        
        # 가중 평균 계산
        total_weight = optimistic.probability_weight + neutral.probability_weight + pessimistic.probability_weight
        
        if total_weight == 0:
            return {'consensus_utility': 0.0, 'consensus_regret': 0.0}
        
        consensus_utility = (
            optimistic.utility_score * optimistic.probability_weight +
            neutral.utility_score * neutral.probability_weight +
            pessimistic.utility_score * pessimistic.probability_weight
        ) / total_weight
        
        consensus_regret = (
            optimistic.regret_potential * optimistic.probability_weight +
            neutral.regret_potential * neutral.probability_weight +
            pessimistic.regret_potential * pessimistic.probability_weight
        ) / total_weight
        
        # 불확실성 범위
        utilities = [optimistic.utility_score, neutral.utility_score, pessimistic.utility_score]
        uncertainty_range = (min(utilities), max(utilities))
        
        # 시나리오 다양성 (분산 기반)
        scenario_diversity = float(np.var(utilities))
        
        # 합의 강도 (다양성의 역수)
        consensus_strength = 1.0 / (1.0 + scenario_diversity)
        
        return {
            'consensus_utility': consensus_utility,
            'consensus_regret': consensus_regret,
            'uncertainty_range': uncertainty_range,
            'scenario_diversity': scenario_diversity,
            'consensus_strength': consensus_strength
        }
    
    def _generate_recommendations(self, optimistic: ScenarioMetrics,
                                neutral: ScenarioMetrics,
                                pessimistic: ScenarioMetrics,
                                consensus_metrics: Dict[str, float]) -> Tuple[str, List[str], List[str]]:
        """권장사항 생성"""
        
        consensus_utility = consensus_metrics['consensus_utility']
        consensus_regret = consensus_metrics['consensus_regret']
        scenario_diversity = consensus_metrics['scenario_diversity']
        
        # 권장 결정
        if consensus_utility > 0.3 and consensus_regret < 0.4:
            if scenario_diversity < 0.1:
                recommended_decision = "적극 추진 권장 (높은 합의)"
            else:
                recommended_decision = "신중한 추진 권장 (불확실성 존재)"
        elif consensus_utility > 0.1 and consensus_regret < 0.6:
            recommended_decision = "조건부 추진 권장 (위험 관리 필요)"
        elif consensus_utility < -0.1 or consensus_regret > 0.7:
            recommended_decision = "추진 비권장 (높은 위험)"
        else:
            recommended_decision = "추가 분석 필요 (모호한 결과)"
        
        # 위험 완화 전략
        all_risks = set(optimistic.risk_factors + neutral.risk_factors + pessimistic.risk_factors)
        risk_mitigation = []
        
        for risk in list(all_risks)[:3]:  # 최대 3개
            if '손실' in risk:
                risk_mitigation.append(f"재정 리스크 관리: {risk} 대비 예비 자금 확보")
            elif '관계' in risk:
                risk_mitigation.append(f"관계 리스크 관리: {risk} 대비 사전 소통 강화")
            elif '건강' in risk:
                risk_mitigation.append(f"건강 리스크 관리: {risk} 대비 예방 조치")
            else:
                risk_mitigation.append(f"리스크 모니터링: {risk} 지속 관찰")
        
        # 기회 향상 전략
        all_opportunities = set(optimistic.opportunity_factors + neutral.opportunity_factors + pessimistic.opportunity_factors)
        opportunity_enhancement = []
        
        for opportunity in list(all_opportunities)[:3]:  # 최대 3개
            if '성장' in opportunity:
                opportunity_enhancement.append(f"성장 기회 활용: {opportunity} 집중 투자")
            elif '네트워크' in opportunity:
                opportunity_enhancement.append(f"네트워크 기회 활용: {opportunity} 적극 확대")
            elif '수익' in opportunity:
                opportunity_enhancement.append(f"수익 기회 활용: {opportunity} 최적화")
            else:
                opportunity_enhancement.append(f"기회 최대화: {opportunity} 전략적 접근")
        
        return recommended_decision, risk_mitigation, opportunity_enhancement
    
    async def analyze_three_view_scenarios(self, input_data: Dict[str, Any]) -> ThreeViewAnalysisResult:
        """3뷰 시나리오 분석"""
        
        # 온디맨드 모델 로딩
        self._load_model()
        
        if not self.is_model_loaded or not self.distribution_model:
            logger.error("3뷰 시나리오 시스템 모델 로딩 실패")
            raise RuntimeError("3뷰 시나리오 시스템 모델 로딩 실패")
        
        start_time = time.time()
        
        try:
            # 1. 컨텍스트 특성 추출
            context_features = self._extract_context_features(input_data)
            
            # 2. 도메인 식별
            domain = self._identify_domain(input_data)
            
            # 3. 시나리오 분포 추정
            scenario_distributions = self.distribution_model.estimate_scenario_distributions(context_features)
            
            # 4. 시나리오별 메트릭 생성
            optimistic_metrics = self._generate_scenario_metrics(
                ScenarioType.OPTIMISTIC, 
                scenario_distributions['optimistic'],
                domain, 
                input_data
            )
            
            neutral_metrics = self._generate_scenario_metrics(
                ScenarioType.NEUTRAL,
                scenario_distributions['neutral'],
                domain,
                input_data
            )
            
            pessimistic_metrics = self._generate_scenario_metrics(
                ScenarioType.PESSIMISTIC,
                scenario_distributions['pessimistic'],
                domain,
                input_data
            )
            
            # 5. 합의 메트릭 계산
            consensus_metrics = self._calculate_consensus_metrics(
                optimistic_metrics, neutral_metrics, pessimistic_metrics
            )
            
            # 6. 권장사항 생성
            recommended_decision, risk_mitigation, opportunity_enhancement = self._generate_recommendations(
                optimistic_metrics, neutral_metrics, pessimistic_metrics, consensus_metrics
            )
            
            # 7. 결과 통합
            result = ThreeViewAnalysisResult(
                optimistic_scenario=optimistic_metrics,
                neutral_scenario=neutral_metrics,
                pessimistic_scenario=pessimistic_metrics,
                consensus_utility=consensus_metrics['consensus_utility'],
                consensus_regret=consensus_metrics['consensus_regret'],
                uncertainty_range=consensus_metrics['uncertainty_range'],
                recommended_decision=recommended_decision,
                risk_mitigation_strategies=risk_mitigation,
                opportunity_enhancement_strategies=opportunity_enhancement,
                scenario_diversity=consensus_metrics['scenario_diversity'],
                consensus_strength=consensus_metrics['consensus_strength'],
                analysis_duration_ms=(time.time() - start_time) * 1000,
                metadata={
                    'domain': domain,
                    'input_id': input_data.get('id', 'unknown'),
                    'generation_timestamp': datetime.now().isoformat()
                }
            )
            
            # 8. 히스토리 저장 (학습용)
            self.scenario_history.append(result)
            if len(self.scenario_history) > 1000:  # 최대 1000개 유지
                self.scenario_history = self.scenario_history[-1000:]
            
            logger.info(f"3뷰 시나리오 분석 완료: {result.analysis_duration_ms:.2f}ms, "
                       f"합의 효용: {result.consensus_utility:.3f}, "
                       f"합의 후회: {result.consensus_regret:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"3뷰 시나리오 분석 실패: {e}")
            raise RuntimeError(f"시나리오 분석 실패: {e}")
    
    def get_scenario_statistics(self) -> Dict[str, Any]:
        """시나리오 통계 정보"""
        if not self.scenario_history:
            return {}
        
        utilities = [r.consensus_utility for r in self.scenario_history]
        regrets = [r.consensus_regret for r in self.scenario_history]
        diversities = [r.scenario_diversity for r in self.scenario_history]
        
        return {
            'total_analyses': len(self.scenario_history),
            'avg_consensus_utility': float(np.mean(utilities)),
            'avg_consensus_regret': float(np.mean(regrets)),
            'avg_scenario_diversity': float(np.mean(diversities)),
            'utility_std': float(np.std(utilities)),
            'regret_std': float(np.std(regrets)),
            'diversity_std': float(np.std(diversities))
        }