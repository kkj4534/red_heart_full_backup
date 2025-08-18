"""
Legal Expert System for Red Heart AI
내장 모델 기반 법률 전문가 시스템

핵심 기능:
1. 교육/기업/사회/정치/생활 도메인별 법률 분석
2. 학습 시 비활성화, 운용 시 활성화
3. 내장 모델 기반 법률 해석 및 권고
4. 컨텍스트 기반 법률 위험 평가
5. 실시간 법률 규정 검토

주의사항:
- 이 시스템은 법률 자문을 제공하지 않음
- 참고용 정보만 제공
- 실제 법률 문제는 전문가 상담 필요
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from datetime import datetime
import os
from pathlib import Path

logger = logging.getLogger('RedHeart.LegalExpertSystem')

class LegalDomain(Enum):
    """법률 도메인"""
    EDUCATION = "education"          # 교육
    BUSINESS = "business"            # 기업
    SOCIAL = "social"                # 사회
    POLITICS = "politics"            # 정치
    LIFE = "life"                    # 생활

class LegalRiskLevel(Enum):
    """법률 위험 수준"""
    LOW = "low"                      # 낮음
    MEDIUM = "medium"                # 보통
    HIGH = "high"                    # 높음
    CRITICAL = "critical"            # 심각

class OperationMode(Enum):
    """운용 모드"""
    TRAINING = "training"            # 학습 모드 (비활성화)
    INFERENCE = "inference"          # 추론 모드 (활성화)
    EVALUATION = "evaluation"        # 평가 모드 (활성화)

@dataclass
class LegalAnalysisContext:
    """법률 분석 컨텍스트"""
    domain: LegalDomain
    text: str
    context_data: Dict[str, Any] = field(default_factory=dict)
    
    # 분석 범위
    jurisdiction: str = "south_korea"  # 관할권
    applicable_laws: List[str] = field(default_factory=list)
    
    # 메타데이터
    timestamp: datetime = field(default_factory=datetime.now)
    request_id: str = ""

@dataclass
class LegalAnalysisResult:
    """법률 분석 결과"""
    risk_level: LegalRiskLevel
    confidence: float
    
    # 위험 요소
    identified_risks: List[str] = field(default_factory=list)
    potential_violations: List[str] = field(default_factory=list)
    
    # 권고사항
    recommendations: List[str] = field(default_factory=list)
    precautions: List[str] = field(default_factory=list)
    
    # 관련 법률
    relevant_laws: List[str] = field(default_factory=list)
    legal_references: List[str] = field(default_factory=list)
    
    # 분석 세부사항
    analysis_details: Dict[str, Any] = field(default_factory=dict)
    
    # 메타데이터
    analysis_duration_ms: float = 0.0
    model_version: str = "1.0"
    disclaimer: str = "이 분석은 참고용이며, 실제 법률 자문을 대체하지 않습니다."

class LegalKnowledgeBase:
    """법률 지식베이스"""
    
    def __init__(self):
        self.legal_rules = {
            LegalDomain.EDUCATION: {
                'privacy_laws': [
                    "개인정보보호법",
                    "정보통신망 이용촉진 및 정보보호 등에 관한 법률",
                    "교육기본법"
                ],
                'student_rights': [
                    "학생 개인정보 보호",
                    "교육 차별 금지",
                    "학습권 보장"
                ],
                'risk_keywords': [
                    "개인정보", "학생정보", "성적", "출결", "상담기록",
                    "차별", "괴롭힘", "체벌", "인권침해"
                ]
            },
            LegalDomain.BUSINESS: {
                'commercial_laws': [
                    "상법",
                    "공정거래법",
                    "근로기준법",
                    "개인정보보호법",
                    "전자상거래 등에서의 소비자보호에 관한 법률"
                ],
                'compliance_areas': [
                    "근로자 권리 보호",
                    "공정거래 준수",
                    "소비자 보호",
                    "개인정보 처리"
                ],
                'risk_keywords': [
                    "계약", "노동", "임금", "해고", "차별", "개인정보",
                    "광고", "표시", "불공정", "독점", "담합"
                ]
            },
            LegalDomain.SOCIAL: {
                'social_laws': [
                    "민법",
                    "형법",
                    "국가인권위원회법",
                    "장애인차별금지 및 권리구제 등에 관한 법률",
                    "성폭력방지 및 피해자보호 등에 관한 법률"
                ],
                'social_rights': [
                    "평등권",
                    "인격권",
                    "사생활 보호",
                    "표현의 자유"
                ],
                'risk_keywords': [
                    "차별", "혐오", "명예훼손", "모욕", "스토킹",
                    "성희롱", "괴롭힘", "사생활침해", "인권침해"
                ]
            },
            LegalDomain.POLITICS: {
                'political_laws': [
                    "헌법",
                    "공직선거법",
                    "정치자금법",
                    "공무원법",
                    "정보공개법"
                ],
                'political_rights': [
                    "선거권",
                    "피선거권",
                    "정치활동의 자유",
                    "알권리"
                ],
                'risk_keywords': [
                    "선거", "정치자금", "후보", "유권자", "매수",
                    "허위사실", "비방", "정치적중립", "공무원"
                ]
            },
            LegalDomain.LIFE: {
                'life_laws': [
                    "민법",
                    "소비자기본법",
                    "주택임대차보호법",
                    "도로교통법",
                    "의료법"
                ],
                'life_rights': [
                    "소비자 권리",
                    "주거권",
                    "생명권",
                    "안전권"
                ],
                'risk_keywords': [
                    "계약", "임대차", "소비자", "의료", "교통",
                    "안전", "피해", "손해", "보상", "책임"
                ]
            }
        }
        
        # 위험 패턴 정의
        self.risk_patterns = {
            'high_risk': [
                "불법", "위법", "형사처벌", "벌금", "과태료",
                "손해배상", "법적책임", "고발", "고소"
            ],
            'medium_risk': [
                "주의", "검토필요", "상담권장", "확인필요",
                "규정위반", "위반가능성", "리스크"
            ],
            'low_risk': [
                "참고", "고려사항", "권장사항", "일반적",
                "관례", "바람직", "추천"
            ]
        }

class LegalAnalysisEngine:
    """법률 분석 엔진"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.knowledge_base = LegalKnowledgeBase()
        
        # 법률 위험 분류 모델
        self.risk_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # 4개 위험 수준
            nn.Softmax(dim=-1)
        ).to(self.device)
        
        # 도메인 분류 모델
        self.domain_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 5),  # 5개 도메인
            nn.Softmax(dim=-1)
        ).to(self.device)
        
        # 법률 임베딩 생성기
        self.legal_embedding = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        ).to(self.device)
        
        # 모델 초기화
        self._initialize_weights()
        
    def _initialize_weights(self):
        """가중치 초기화"""
        for module in [self.risk_classifier, self.domain_classifier, self.legal_embedding]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def _extract_legal_features(self, context: LegalAnalysisContext) -> torch.Tensor:
        """법률 특성 추출"""
        features = []
        
        text = context.text.lower()
        
        # 1. 도메인별 키워드 특성
        domain_rules = self.knowledge_base.legal_rules[context.domain]
        risk_keywords = domain_rules['risk_keywords']
        
        for keyword in risk_keywords:
            features.append(float(keyword in text))
        
        # 2. 위험 패턴 특성
        for risk_level, patterns in self.knowledge_base.risk_patterns.items():
            pattern_count = sum(1 for pattern in patterns if pattern in text)
            features.append(min(pattern_count / len(patterns), 1.0))
        
        # 3. 법률 용어 특성
        legal_terms = [
            "법률", "규정", "조항", "위반", "준수", "의무",
            "권리", "책임", "처벌", "제재", "허가", "신고"
        ]
        
        for term in legal_terms:
            features.append(float(term in text))
        
        # 4. 텍스트 길이 및 복잡도
        features.extend([
            min(len(text) / 1000, 1.0),  # 텍스트 길이 정규화
            len(text.split()) / 100,     # 단어 수 정규화
            text.count('.') / 10,        # 문장 수 정규화
            text.count('?') / 5,         # 질문 수 정규화
        ])
        
        # 5. 컨텍스트 데이터
        context_data = context.context_data
        features.extend([
            context_data.get('urgency', 0.0),
            context_data.get('impact_level', 0.0),
            context_data.get('public_interest', 0.0),
            context_data.get('stakeholder_count', 0.0) / 100,
            context_data.get('financial_impact', 0.0) / 1000000,  # 백만원 단위
        ])
        
        # 패딩하여 512차원으로 맞춤
        while len(features) < 512:
            features.append(0.0)
        
        return torch.tensor(features[:512], dtype=torch.float32, device=self.device)
    
    def analyze_legal_risk(self, context: LegalAnalysisContext) -> LegalAnalysisResult:
        """법률 위험 분석"""
        
        # 특성 추출
        features = self._extract_legal_features(context)
        
        # 위험 수준 분류
        with torch.no_grad():
            risk_probs = self.risk_classifier(features.unsqueeze(0)).squeeze(0)
            risk_level_idx = torch.argmax(risk_probs).item()
            confidence = risk_probs[risk_level_idx].item()
        
        # 위험 수준 매핑
        risk_levels = [LegalRiskLevel.LOW, LegalRiskLevel.MEDIUM, 
                      LegalRiskLevel.HIGH, LegalRiskLevel.CRITICAL]
        risk_level = risk_levels[risk_level_idx]
        
        # 세부 분석
        analysis_details = self._detailed_analysis(context, features)
        
        # 권고사항 생성
        recommendations = self._generate_recommendations(context, risk_level, analysis_details)
        
        # 관련 법률 식별
        relevant_laws = self._identify_relevant_laws(context, analysis_details)
        
        # 결과 생성
        result = LegalAnalysisResult(
            risk_level=risk_level,
            confidence=confidence,
            identified_risks=analysis_details['identified_risks'],
            potential_violations=analysis_details['potential_violations'],
            recommendations=recommendations,
            precautions=analysis_details['precautions'],
            relevant_laws=relevant_laws,
            analysis_details=analysis_details,
            model_version="1.0"
        )
        
        return result
    
    def _detailed_analysis(self, context: LegalAnalysisContext, 
                         features: torch.Tensor) -> Dict[str, Any]:
        """세부 분석"""
        
        text = context.text.lower()
        domain_rules = self.knowledge_base.legal_rules[context.domain]
        
        # 위험 요소 식별
        identified_risks = []
        potential_violations = []
        precautions = []
        
        # 도메인별 위험 키워드 검사
        for keyword in domain_rules['risk_keywords']:
            if keyword in text:
                identified_risks.append(f"{keyword} 관련 위험")
        
        # 위험 패턴 검사
        for risk_level, patterns in self.knowledge_base.risk_patterns.items():
            for pattern in patterns:
                if pattern in text:
                    if risk_level == 'high_risk':
                        potential_violations.append(f"{pattern} 관련 법적 위험")
                    elif risk_level == 'medium_risk':
                        precautions.append(f"{pattern} 관련 주의사항")
        
        # 컨텍스트 기반 분석
        context_analysis = {}
        if context.context_data:
            if context.context_data.get('public_interest', 0) > 0.7:
                context_analysis['public_interest'] = "공익성 높음 - 추가 검토 필요"
            
            if context.context_data.get('financial_impact', 0) > 1000000:
                context_analysis['financial_impact'] = "재정적 영향 큼 - 신중한 검토 필요"
        
        return {
            'identified_risks': identified_risks[:5],  # 최대 5개
            'potential_violations': potential_violations[:3],  # 최대 3개
            'precautions': precautions[:5],  # 최대 5개
            'context_analysis': context_analysis,
            'risk_keywords_found': len(identified_risks),
            'high_risk_patterns': len([p for p in potential_violations if '법적 위험' in p])
        }
    
    def _generate_recommendations(self, context: LegalAnalysisContext, 
                                risk_level: LegalRiskLevel,
                                analysis_details: Dict[str, Any]) -> List[str]:
        """권고사항 생성"""
        
        recommendations = []
        
        # 위험 수준별 기본 권고사항
        if risk_level == LegalRiskLevel.CRITICAL:
            recommendations.extend([
                "즉시 법률 전문가와 상담하십시오",
                "관련 활동을 일시 중단하고 법적 검토를 받으십시오",
                "내부 법무팀 또는 외부 법률사무소와 긴급 협의하십시오"
            ])
        elif risk_level == LegalRiskLevel.HIGH:
            recommendations.extend([
                "법률 전문가 상담을 권장합니다",
                "관련 법령을 상세히 검토하십시오",
                "내부 컴플라이언스 팀과 협의하십시오"
            ])
        elif risk_level == LegalRiskLevel.MEDIUM:
            recommendations.extend([
                "관련 법령 및 규정을 확인하십시오",
                "필요시 전문가 상담을 고려하십시오",
                "내부 정책과의 일치성을 검토하십시오"
            ])
        else:  # LOW
            recommendations.extend([
                "일반적인 법적 원칙을 준수하십시오",
                "관련 가이드라인을 참고하십시오"
            ])
        
        # 도메인별 특화 권고사항
        domain_specific = self._get_domain_specific_recommendations(context.domain, analysis_details)
        recommendations.extend(domain_specific)
        
        return recommendations[:5]  # 최대 5개
    
    def _get_domain_specific_recommendations(self, domain: LegalDomain, 
                                          analysis_details: Dict[str, Any]) -> List[str]:
        """도메인별 특화 권고사항"""
        
        if domain == LegalDomain.EDUCATION:
            return [
                "학생 개인정보 보호 원칙을 준수하십시오",
                "교육 차별 금지 원칙을 확인하십시오"
            ]
        elif domain == LegalDomain.BUSINESS:
            return [
                "근로기준법 준수 여부를 확인하십시오",
                "공정거래 관련 규정을 검토하십시오"
            ]
        elif domain == LegalDomain.SOCIAL:
            return [
                "인권 침해 요소가 없는지 확인하십시오",
                "차별 금지 원칙을 준수하십시오"
            ]
        elif domain == LegalDomain.POLITICS:
            return [
                "정치적 중립성을 유지하십시오",
                "선거법 위반 요소를 확인하십시오"
            ]
        elif domain == LegalDomain.LIFE:
            return [
                "소비자 권리 보호를 고려하십시오",
                "안전 규정을 준수하십시오"
            ]
        
        return []
    
    def _identify_relevant_laws(self, context: LegalAnalysisContext, 
                              analysis_details: Dict[str, Any]) -> List[str]:
        """관련 법률 식별"""
        
        domain_rules = self.knowledge_base.legal_rules[context.domain]
        relevant_laws = []
        
        # 도메인별 기본 법률
        if context.domain == LegalDomain.EDUCATION:
            relevant_laws.extend(domain_rules['privacy_laws'])
        elif context.domain == LegalDomain.BUSINESS:
            relevant_laws.extend(domain_rules['commercial_laws'])
        elif context.domain == LegalDomain.SOCIAL:
            relevant_laws.extend(domain_rules['social_laws'])
        elif context.domain == LegalDomain.POLITICS:
            relevant_laws.extend(domain_rules['political_laws'])
        elif context.domain == LegalDomain.LIFE:
            relevant_laws.extend(domain_rules['life_laws'])
        
        # 위험 수준에 따른 추가 법률
        if analysis_details['high_risk_patterns'] > 0:
            relevant_laws.extend(["형법", "민법"])
        
        return list(set(relevant_laws))[:5]  # 중복 제거 및 최대 5개

class LegalExpertSystem:
    """법률 전문가 시스템"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        
        # GPU 메모리 관리자
        try:
            from dynamic_gpu_manager import DynamicGPUManager
            self.gpu_manager = DynamicGPUManager()
            self.gpu_optimization_enabled = True
            logger.info("법률 전문가 시스템 GPU 메모리 관리 활성화")
        except ImportError:
            self.gpu_manager = None
            self.gpu_optimization_enabled = False
            logger.warning("Dynamic GPU Manager를 찾을 수 없습니다. 기본 메모리 관리 사용")
        
        # 온디맨드 로딩을 위한 모델 상태
        self.analysis_engine = None
        self.is_model_loaded = False
        
        # 운용 모드 (기본값: 추론 모드)
        self.operation_mode = OperationMode.INFERENCE
        
        # 활성화 상태
        self.is_active = True
        
        # 분석 히스토리
        self.analysis_history = []
        
        logger.info("법률 전문가 시스템 초기화 완료 (온디맨드 로딩 방식)")
    
    def _load_model(self):
        """모델 온디맨드 로딩"""
        if self.is_model_loaded:
            return
            
        try:
            if self.gpu_optimization_enabled:
                # GPU 메모리 관리자를 통한 로딩
                with self.gpu_manager.allocate_memory('legal_expert_system', dynamic_boost=True):
                    self.analysis_engine = LegalAnalysisEngine(self.device)
                    self.is_model_loaded = True
                    logger.info("법률 전문가 시스템 모델 로딩 완료 (GPU 최적화)")
            else:
                # 기본 로딩
                self.analysis_engine = LegalAnalysisEngine(self.device)
                self.is_model_loaded = True
                logger.info("법률 전문가 시스템 모델 로딩 완료 (기본 방식)")
                
        except Exception as e:
            logger.error(f"법률 전문가 시스템 모델 로딩 실패: {e}")
            self.is_model_loaded = False
            self.analysis_engine = None
            
    def _unload_model(self):
        """모델 언로딩"""
        if not self.is_model_loaded:
            return
            
        try:
            if self.analysis_engine:
                # 메모리 정리
                if hasattr(self.analysis_engine, 'risk_classifier'):
                    del self.analysis_engine.risk_classifier
                if hasattr(self.analysis_engine, 'domain_classifier'):
                    del self.analysis_engine.domain_classifier
                if hasattr(self.analysis_engine, 'legal_embedding'):
                    del self.analysis_engine.legal_embedding
                    
                del self.analysis_engine
                self.analysis_engine = None
                
            # GPU 메모리 정리
            if self.device == 'cuda':
                import torch
                torch.cuda.empty_cache()
                
            self.is_model_loaded = False
            logger.info("법률 전문가 시스템 모델 언로딩 완료")
            
        except Exception as e:
            logger.error(f"법률 전문가 시스템 모델 언로딩 실패: {e}")
    
    def set_operation_mode(self, mode: OperationMode):
        """운용 모드 설정"""
        self.operation_mode = mode
        
        # 학습 모드에서는 비활성화
        if mode == OperationMode.TRAINING:
            self.is_active = False
            logger.info("학습 모드 - 법률 전문가 시스템 비활성화")
        else:
            self.is_active = True
            logger.info(f"{mode.value} 모드 - 법률 전문가 시스템 활성화")
    
    def analyze_legal_context(self, domain: LegalDomain, 
                            text: str, 
                            context_data: Optional[Dict[str, Any]] = None) -> Optional[LegalAnalysisResult]:
        """법률 컨텍스트 분석"""
        
        # 비활성화 상태 또는 학습 모드에서는 None 반환
        if not self.is_active or self.operation_mode == OperationMode.TRAINING:
            logger.debug("법률 전문가 시스템 비활성화 상태")
            return None
        
        # 온디맨드 모델 로딩
        self._load_model()
        
        if not self.is_model_loaded or not self.analysis_engine:
            logger.error("법률 전문가 시스템 모델 로딩 실패")
            return None
        
        start_time = time.time()
        
        try:
            # 분석 컨텍스트 생성
            context = LegalAnalysisContext(
                domain=domain,
                text=text,
                context_data=context_data or {},
                request_id=f"legal_{int(time.time() * 1000)}"
            )
            
            # 법률 분석 수행
            result = self.analysis_engine.analyze_legal_risk(context)
            
            # 처리 시간 기록
            result.analysis_duration_ms = (time.time() - start_time) * 1000
            
            # 히스토리 저장
            self.analysis_history.append({
                'context': context,
                'result': result,
                'timestamp': datetime.now()
            })
            
            # 최근 100개만 유지
            if len(self.analysis_history) > 100:
                self.analysis_history = self.analysis_history[-100:]
            
            logger.info(f"법률 분석 완료: {domain.value}, 위험도: {result.risk_level.value}, "
                       f"처리시간: {result.analysis_duration_ms:.2f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"법률 분석 실패: {e}")
            return None
    
    def get_domain_guidelines(self, domain: LegalDomain) -> Dict[str, Any]:
        """도메인별 가이드라인"""
        
        if not self.is_active:
            return {}
        
        domain_rules = self.analysis_engine.knowledge_base.legal_rules[domain]
        
        guidelines = {
            'domain': domain.value,
            'key_laws': domain_rules.get('privacy_laws', []) + 
                       domain_rules.get('commercial_laws', []) + 
                       domain_rules.get('social_laws', []) + 
                       domain_rules.get('political_laws', []) + 
                       domain_rules.get('life_laws', []),
            'risk_keywords': domain_rules['risk_keywords'],
            'compliance_areas': domain_rules.get('compliance_areas', []),
            'general_recommendations': [
                "관련 법령을 정기적으로 검토하십시오",
                "내부 컴플라이언스 체계를 구축하십시오",
                "전문가 상담을 정기적으로 받으십시오"
            ]
        }
        
        return guidelines
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 조회"""
        
        return {
            'is_active': self.is_active,
            'operation_mode': self.operation_mode.value,
            'analysis_count': len(self.analysis_history),
            'supported_domains': [domain.value for domain in LegalDomain],
            'last_analysis': self.analysis_history[-1]['timestamp'].isoformat() if self.analysis_history else None,
            'model_version': "1.0",
            'disclaimer': "이 시스템은 법률 자문을 제공하지 않으며, 참고용 정보만 제공합니다."
        }
    
    def reset_system(self):
        """시스템 리셋"""
        self.analysis_history.clear()
        logger.info("법률 전문가 시스템 리셋 완료")

# 전역 인스턴스 (싱글톤 패턴)
_legal_expert_system = None

def get_legal_expert_system(device: str = 'cpu') -> LegalExpertSystem:
    """법률 전문가 시스템 인스턴스 반환"""
    global _legal_expert_system
    
    if _legal_expert_system is None:
        _legal_expert_system = LegalExpertSystem(device)
    
    return _legal_expert_system