"""
동적 윤리적 선택지 분석 시스템
Dynamic Ethical Choice Analyzer System

모든 윤리적 딜레마에 대해 동적으로 선택지를 추출하고 분석하는 범용 시스템
- 자연어 처리를 통한 선택지 자동 추출
- 럼바우 구조적 분석으로 이해관계자 및 제약사항 파악
- 각 선택지별 12개 모듈 병렬 분석
- 반사실적 추론을 통한 대안 시나리오 생성
"""

import asyncio
import logging
import time
import json
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import uuid
from enum import Enum

# 고급 분석 모듈들
try:
    from advanced_emotion_analyzer import AdvancedEmotionAnalyzer
    from advanced_bentham_calculator import AdvancedBenthamCalculator
    from advanced_regret_analyzer import AdvancedRegretAnalyzer
    from advanced_surd_analyzer import AdvancedSURDAnalyzer
    from advanced_counterfactual_reasoning import AdvancedCounterfactualReasoning
    from advanced_rumbaugh_analyzer import AdvancedRumbaughAnalyzer
    from integrated_system_orchestrator import IntegratedSystemOrchestrator, IntegrationContext
    from llm_module.advanced_llm_engine import get_llm_engine
    ADVANCED_MODULES_AVAILABLE = True
except ImportError as e:
    ADVANCED_MODULES_AVAILABLE = False
    logging.error(f"고급 모듈 임포트 실패: {e}")

logger = logging.getLogger('DynamicEthicalChoiceAnalyzer')

class DilemmaType(Enum):
    """딜레마 유형"""
    TROLLEY_PROBLEM = "trolley_problem"
    RESOURCE_ALLOCATION = "resource_allocation"
    PRIVACY_SECURITY = "privacy_security"
    MEDICAL_ETHICS = "medical_ethics"
    ENVIRONMENTAL_CHOICE = "environmental_choice"
    PERSONAL_RELATIONSHIP = "personal_relationship"
    PROFESSIONAL_DUTY = "professional_duty"
    SACRIFICE_CHOICE = "sacrifice_choice"
    TRUTH_VS_KINDNESS = "truth_vs_kindness"
    INDIVIDUAL_VS_COLLECTIVE = "individual_vs_collective"
    UNKNOWN = "unknown"

@dataclass
class EthicalChoice:
    """윤리적 선택지"""
    id: str
    name: str
    description: str
    action_type: str  # "action", "inaction", "compromise"
    stakeholders_affected: List[str]
    expected_outcomes: Dict[str, Any]
    moral_weight: float = 0.0
    feasibility: float = 1.0
    risk_level: float = 0.0
    time_sensitivity: float = 0.0
    
    # 럼바우 구조 분석 결과
    structural_elements: Dict[str, Any] = field(default_factory=dict)
    object_relationships: List[Dict[str, Any]] = field(default_factory=list)
    constraint_factors: List[str] = field(default_factory=list)

@dataclass
class EthicalDilemma:
    """윤리적 딜레마"""
    id: str
    title: str
    description: str
    context: Dict[str, Any]
    dilemma_type: DilemmaType
    
    # 동적 추출된 정보
    extracted_choices: List[EthicalChoice] = field(default_factory=list)
    stakeholders: Dict[str, float] = field(default_factory=dict)
    moral_complexity: float = 0.0
    urgency_level: float = 0.0
    
    # 분석 결과
    choice_analyses: Dict[str, Any] = field(default_factory=dict)
    recommended_choice: Optional[EthicalChoice] = None
    reasoning_chain: List[str] = field(default_factory=list)
    
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ChoiceAnalysisResult:
    """선택지 분석 결과"""
    choice: EthicalChoice
    emotion_analysis: Dict[str, Any]
    bentham_analysis: Dict[str, Any]
    regret_analysis: Dict[str, Any]
    surd_analysis: Dict[str, Any]
    counterfactual_scenarios: List[Dict[str, Any]]
    
    # 통합 점수
    utility_score: float = 0.0
    confidence_score: float = 0.0
    risk_adjusted_score: float = 0.0
    
    processing_time: float = 0.0

class DynamicEthicalChoiceAnalyzer:
    """동적 윤리적 선택지 분석 시스템"""
    
    def __init__(self):
        self.emotion_analyzer = None
        self.bentham_calculator = None
        self.regret_analyzer = None
        self.surd_analyzer = None
        self.counterfactual_reasoner = None
        self.rumbaugh_analyzer = None
        self.integrated_orchestrator = None
        self.llm_engine = None
        
        # 선택지 추출 패턴
        self.choice_patterns = [
            r'(?:선택|옵션|방법|방안).*?(?:\d+|하나|둘|셋)',
            r'(?:해야|할지|것인가|인가).*?(?:아니면|또는|혹은)',
            r'(?:밟을지|틀지|할지|갈지).*?(?:아니면|또는|혹은)',
            r'(?:보호|선택|우선|고려).*?(?:vs|대|versus)',
            r'(?:포기|희생|양보).*?(?:위해|하고|해서)',
        ]
        
        # 이해관계자 추출 패턴
        self.stakeholder_patterns = [
            r'(?:승객|탑승자|운전자|보행자|시민|환자|의사|가족|친구|동료|회사|정부|사회|공동체|개인|집단)',
            r'(?:어린이|아이|청소년|성인|노인|장애인|임산부)',
            r'(?:직원|관리자|CEO|상사|부하|동료|파트너)',
            r'(?:학생|교사|교수|연구자|전문가)',
        ]
        
        # 딜레마 유형 키워드
        self.dilemma_keywords = {
            DilemmaType.TROLLEY_PROBLEM: ['트롤리', '자율주행', '브레이크', '충돌', '사고'],
            DilemmaType.RESOURCE_ALLOCATION: ['자원', '배분', '분배', '할당', '예산'],
            DilemmaType.PRIVACY_SECURITY: ['개인정보', '프라이버시', '보안', '감시', '정보'],
            DilemmaType.MEDICAL_ETHICS: ['의료', '치료', '환자', '생명', '건강'],
            DilemmaType.ENVIRONMENTAL_CHOICE: ['환경', '기후', '오염', '자연', '생태'],
            DilemmaType.PERSONAL_RELATIONSHIP: ['친구', '가족', '연인', '관계', '신뢰'],
            DilemmaType.PROFESSIONAL_DUTY: ['직업', '업무', '의무', '책임', '윤리'],
            DilemmaType.SACRIFICE_CHOICE: ['희생', '포기', '양보', '손실', '대가'],
            DilemmaType.TRUTH_VS_KINDNESS: ['진실', '거짓말', '친절', '상처', '솔직'],
            DilemmaType.INDIVIDUAL_VS_COLLECTIVE: ['개인', '집단', '공동체', '사회', '이익']
        }
        
        self._initialize_modules()
    
    def _initialize_modules(self):
        """모든 분석 모듈 초기화 - CPU 기반 + Lazy GPU Loading"""
        if not ADVANCED_MODULES_AVAILABLE:
            logger.error("고급 모듈을 사용할 수 없습니다.")
            return
        
        try:
            logger.info("동적 윤리적 선택지 분석 시스템 초기화 (CPU 기반)...")
            
            # 개별 분석 모듈들은 이미 초기화된 인스턴스를 재사용하거나 필요시 생성
            # 실제 분석 시에만 GPU 컨텍스트 매니저 사용
            logger.info("  📋 분석 모듈들을 레이지 로딩 모드로 설정...")
            
            # 모듈들을 None으로 초기화 (실제 사용시 로드)
            self.emotion_analyzer = None
            self.bentham_calculator = None  
            self.regret_analyzer = None
            self.surd_analyzer = None
            self.counterfactual_reasoner = None
            self.rumbaugh_analyzer = None
            
            # 통합 시스템도 lazy loading
            self.integrated_orchestrator = None
            
            # LLM 엔진도 필요시에만 로드
            self.llm_engine = None
            
            logger.info("✅ 동적 윤리적 선택지 분석 시스템 초기화 완료 (Lazy Loading 모드)")
            
        except Exception as e:
            logger.error(f"모듈 초기화 실패: {e}")
            raise
    
    def _get_emotion_analyzer(self):
        """감정 분석기 lazy loading"""
        if self.emotion_analyzer is None:
            logger.info("  🔄 감정 분석기 로딩...")
            self.emotion_analyzer = AdvancedEmotionAnalyzer()
        return self.emotion_analyzer
    
    def _get_bentham_calculator(self):
        """벤담 계산기 lazy loading"""
        if self.bentham_calculator is None:
            logger.info("  🔄 벤담 계산기 로딩...")
            self.bentham_calculator = AdvancedBenthamCalculator()
        return self.bentham_calculator
    
    def _get_regret_analyzer(self):
        """후회 분석기 lazy loading"""
        if self.regret_analyzer is None:
            logger.info("  🔄 후회 분석기 로딩...")
            self.regret_analyzer = AdvancedRegretAnalyzer()
        return self.regret_analyzer
    
    def _get_surd_analyzer(self):
        """SURD 분석기 lazy loading"""
        if self.surd_analyzer is None:
            logger.info("  🔄 SURD 분석기 로딩...")
            self.surd_analyzer = AdvancedSURDAnalyzer()
        return self.surd_analyzer
    
    def _get_counterfactual_reasoner(self):
        """반사실적 추론기 lazy loading"""
        if self.counterfactual_reasoner is None:
            logger.info("  🔄 반사실적 추론기 로딩...")
            self.counterfactual_reasoner = AdvancedCounterfactualReasoning()
        return self.counterfactual_reasoner
    
    def _get_rumbaugh_analyzer(self):
        """럼바우 분석기 lazy loading"""
        if self.rumbaugh_analyzer is None:
            logger.info("  🔄 럼바우 분석기 로딩...")
            self.rumbaugh_analyzer = AdvancedRumbaughAnalyzer()
        return self.rumbaugh_analyzer
    
    def _get_integrated_orchestrator(self):
        """통합 오케스트레이터 lazy loading"""
        if self.integrated_orchestrator is None:
            logger.info("  🔄 통합 오케스트레이터 로딩...")
            self.integrated_orchestrator = IntegratedSystemOrchestrator()
        return self.integrated_orchestrator
    
    def _get_llm_engine(self):
        """LLM 엔진 lazy loading"""
        if self.llm_engine is None:
            logger.info("  🔄 LLM 엔진 로딩...")
            self.llm_engine = get_llm_engine()
        return self.llm_engine
    
    async def analyze_ethical_dilemma(self, dilemma_text: str, title: str = "", context: Dict[str, Any] = None) -> EthicalDilemma:
        """윤리적 딜레마 종합 분석"""
        
        logger.info(f"🎯 윤리적 딜레마 분석 시작: {title or '제목 없음'}")
        
        # 1. 딜레마 기본 정보 생성
        dilemma = EthicalDilemma(
            id=str(uuid.uuid4()),
            title=title or "윤리적 딜레마",
            description=dilemma_text,
            context=context or {},
            dilemma_type=self._classify_dilemma_type(dilemma_text)
        )
        
        # 2. 동적 선택지 추출
        logger.info("📋 동적 선택지 추출...")
        dilemma.extracted_choices = await self._extract_choices(dilemma_text)
        logger.info(f"   추출된 선택지: {len(dilemma.extracted_choices)}개")
        
        # 3. 이해관계자 추출
        logger.info("👥 이해관계자 추출...")
        dilemma.stakeholders = self._extract_stakeholders(dilemma_text)
        logger.info(f"   식별된 이해관계자: {len(dilemma.stakeholders)}명")
        
        # 4. 럼바우 구조적 분석
        logger.info("🏗️ 럼바우 구조적 분석...")
        await self._rumbaugh_structural_analysis(dilemma)
        
        # 5. 각 선택지별 심층 분석
        logger.info("🔍 각 선택지별 심층 분석...")
        for choice in dilemma.extracted_choices:
            analysis_result = await self._analyze_individual_choice(dilemma, choice)
            dilemma.choice_analyses[choice.id] = analysis_result
            logger.info(f"   ✅ {choice.name} 분석 완료 (유틸리티: {analysis_result.utility_score:.3f})")
        
        # 6. 최적 선택지 추천
        logger.info("🎯 최적 선택지 추천...")
        dilemma.recommended_choice = self._recommend_optimal_choice(dilemma)
        
        # 7. 추론 체인 생성
        logger.info("🧠 추론 체인 생성...")
        dilemma.reasoning_chain = await self._generate_reasoning_chain(dilemma)
        
        logger.info(f"✅ 윤리적 딜레마 분석 완료: {dilemma.title}")
        if dilemma.recommended_choice:
            logger.info(f"   🎯 최종 추천: {dilemma.recommended_choice.name}")
        
        return dilemma
    
    async def _extract_choices(self, text: str) -> List[EthicalChoice]:
        """텍스트에서 선택지 동적 추출"""
        
        choices = []
        
        # 1. 패턴 기반 선택지 추출
        for pattern in self.choice_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                choices.extend(self._parse_choice_from_match(match, text))
        
        # 2. LLM 기반 선택지 추출 (더 정확한 추출)
        if self.llm_engine:
            llm_choices = await self._extract_choices_with_llm(text)
            choices.extend(llm_choices)
        
        # 3. 중복 제거 및 정규화
        unique_choices = self._deduplicate_choices(choices)
        
        # 4. 선택지가 없으면 기본 선택지 생성
        if not unique_choices:
            unique_choices = self._generate_default_choices(text)
        
        return unique_choices
    
    async def _extract_choices_with_llm(self, text: str) -> List[EthicalChoice]:
        """LLM을 사용한 선택지 추출"""
        
        if not self.llm_engine:
            return []
        
        prompt = f"""
다음 윤리적 딜레마에서 가능한 선택지들을 추출하세요:

상황: {text}

가능한 선택지들을 JSON 형태로 반환하세요:
{{
    "choices": [
        {{
            "name": "선택지 이름",
            "description": "선택지 상세 설명",
            "action_type": "action/inaction/compromise",
            "stakeholders_affected": ["이해관계자1", "이해관계자2"],
            "expected_outcomes": {{"결과1": "설명1", "결과2": "설명2"}}
        }}
    ]
}}
"""
        
        try:
            response = await self.llm_engine.generate_response(prompt)
            # JSON 파싱 및 EthicalChoice 객체 생성
            # 실제 구현에서는 JSON 파싱 및 객체 생성 로직 추가
            return []
        except Exception as e:
            logger.error(f"LLM 선택지 추출 실패: {e}")
            return []
    
    def _parse_choice_from_match(self, match: str, full_text: str) -> List[EthicalChoice]:
        """매치된 텍스트에서 선택지 파싱"""
        
        choices = []
        
        # 간단한 선택지 파싱 로직
        if "아니면" in match or "또는" in match or "혹은" in match:
            parts = re.split(r'(?:아니면|또는|혹은)', match)
            for i, part in enumerate(parts):
                if part.strip():
                    choice = EthicalChoice(
                        id=f"choice_{i}_{uuid.uuid4().hex[:8]}",
                        name=part.strip(),
                        description=f"선택지: {part.strip()}",
                        action_type="action",
                        stakeholders_affected=[],
                        expected_outcomes={}
                    )
                    choices.append(choice)
        
        return choices
    
    def _deduplicate_choices(self, choices: List[EthicalChoice]) -> List[EthicalChoice]:
        """중복 선택지 제거"""
        
        seen_names = set()
        unique_choices = []
        
        for choice in choices:
            if choice.name not in seen_names:
                seen_names.add(choice.name)
                unique_choices.append(choice)
        
        return unique_choices
    
    def _generate_default_choices(self, text: str) -> List[EthicalChoice]:
        """기본 선택지 생성"""
        
        return [
            EthicalChoice(
                id=f"default_action_{uuid.uuid4().hex[:8]}",
                name="적극적 행동",
                description="상황에 적극적으로 개입하여 행동한다",
                action_type="action",
                stakeholders_affected=[],
                expected_outcomes={"개입": "상황에 직접적 영향"}
            ),
            EthicalChoice(
                id=f"default_inaction_{uuid.uuid4().hex[:8]}",
                name="소극적 대응",
                description="상황을 지켜보며 최소한의 개입만 한다",
                action_type="inaction",
                stakeholders_affected=[],
                expected_outcomes={"관찰": "상황 변화 관찰"}
            ),
            EthicalChoice(
                id=f"default_compromise_{uuid.uuid4().hex[:8]}",
                name="타협적 해결",
                description="여러 이해관계자의 입장을 고려한 타협안을 찾는다",
                action_type="compromise",
                stakeholders_affected=[],
                expected_outcomes={"타협": "부분적 만족"}
            )
        ]
    
    def _extract_stakeholders(self, text: str) -> Dict[str, float]:
        """이해관계자 추출"""
        
        stakeholders = {}
        
        for pattern in self.stakeholder_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # 기본 중요도 할당
                importance = 0.8
                if any(keyword in match for keyword in ['어린이', '아이', '임산부', '장애인']):
                    importance = 0.9
                elif any(keyword in match for keyword in ['환자', '피해자']):
                    importance = 0.95
                
                stakeholders[match] = importance
        
        return stakeholders
    
    def _classify_dilemma_type(self, text: str) -> DilemmaType:
        """딜레마 유형 분류"""
        
        text_lower = text.lower()
        
        for dilemma_type, keywords in self.dilemma_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return dilemma_type
        
        return DilemmaType.UNKNOWN
    
    async def _rumbaugh_structural_analysis(self, dilemma: EthicalDilemma):
        """럼바우 구조적 분석 - Lazy Loading + GPU Context"""
        
        try:
            # Lazy loading으로 분석기 로드
            rumbaugh_analyzer = self._get_rumbaugh_analyzer()
            
            # 각 선택지에 대한 구조적 분석
            for choice in dilemma.extracted_choices:
                # GPU context manager 사용
                from config import gpu_model_context
                
                # 필요시 GPU 사용, 기본은 CPU
                with gpu_model_context(rumbaugh_analyzer, memory_required_mb=300) as gpu_analyzer:
                    analysis_result = await gpu_analyzer.analyze_ethical_structure(
                        situation=dilemma.description,
                        choice_context=choice.description,
                        stakeholders=list(dilemma.stakeholders.keys())
                    )
                
                choice.structural_elements = analysis_result.get('structural_elements', {})
                choice.object_relationships = analysis_result.get('object_relationships', [])
                choice.constraint_factors = analysis_result.get('constraint_factors', [])
                
        except Exception as e:
            logger.error(f"럼바우 구조적 분석 실패: {e}")
    
    async def _analyze_individual_choice(self, dilemma: EthicalDilemma, choice: EthicalChoice) -> ChoiceAnalysisResult:
        """개별 선택지 심층 분석"""
        
        start_time = time.time()
        
        # 선택지 컨텍스트 구성
        choice_context = f"""
상황: {dilemma.description}
선택지: {choice.name}
상세 설명: {choice.description}
이해관계자: {', '.join(dilemma.stakeholders.keys())}
"""
        
        # 병렬 분석 실행
        analyses = await asyncio.gather(
            self._analyze_choice_emotion(choice_context),
            self._analyze_choice_bentham(choice_context, dilemma),
            self._analyze_choice_regret(choice_context, dilemma, choice),
            self._analyze_choice_surd(choice_context, dilemma),
            self._generate_choice_counterfactuals(choice_context, dilemma, choice),
            return_exceptions=True
        )
        
        # 결과 처리
        emotion_analysis = analyses[0] if not isinstance(analyses[0], Exception) else {}
        bentham_analysis = analyses[1] if not isinstance(analyses[1], Exception) else {}
        regret_analysis = analyses[2] if not isinstance(analyses[2], Exception) else {}
        surd_analysis = analyses[3] if not isinstance(analyses[3], Exception) else {}
        counterfactual_scenarios = analyses[4] if not isinstance(analyses[4], Exception) else []
        
        # 통합 점수 계산
        utility_score = self._calculate_utility_score(emotion_analysis, bentham_analysis, regret_analysis)
        confidence_score = self._calculate_confidence_score(emotion_analysis, bentham_analysis, regret_analysis)
        risk_adjusted_score = utility_score * confidence_score * (1 - choice.risk_level)
        
        processing_time = time.time() - start_time
        
        return ChoiceAnalysisResult(
            choice=choice,
            emotion_analysis=emotion_analysis,
            bentham_analysis=bentham_analysis,
            regret_analysis=regret_analysis,
            surd_analysis=surd_analysis,
            counterfactual_scenarios=counterfactual_scenarios,
            utility_score=utility_score,
            confidence_score=confidence_score,
            risk_adjusted_score=risk_adjusted_score,
            processing_time=processing_time
        )
    
    async def _analyze_choice_emotion(self, choice_context: str) -> Dict[str, Any]:
        """선택지 감정 분석 - Lazy Loading + GPU Context"""
        
        try:
            # Lazy loading으로 감정 분석기 로드
            emotion_analyzer = self._get_emotion_analyzer()
            
            # GPU context manager 사용 (감정 분석은 메모리 사용량이 높음)
            from config import gpu_model_context
            
            with gpu_model_context(emotion_analyzer, memory_required_mb=800) as gpu_analyzer:
                emotion_result = gpu_analyzer.analyze_emotion(
                    text=choice_context,
                    language="ko",
                    use_cache=True
                )
            
            return {
                'emotion': getattr(emotion_result, 'dominant_emotion', 'unknown'),
                'intensity': getattr(emotion_result, 'intensity', 0.0),
                'confidence': getattr(emotion_result, 'confidence', 0.0),
                'arousal': getattr(emotion_result, 'arousal', 0.0),
                'valence': getattr(emotion_result, 'valence', 0.0)
            }
        except Exception as e:
            logger.error(f"감정 분석 실패: {e}")
            return {}
    
    async def _analyze_choice_bentham(self, choice_context: str, dilemma: EthicalDilemma) -> Dict[str, Any]:
        """선택지 벤담 분석 - Lazy Loading + GPU Context"""
        
        try:
            # Lazy loading으로 벤담 계산기 로드
            bentham_calculator = self._get_bentham_calculator()
            
            bentham_input = {
                'situation': choice_context,
                'context': dilemma.context,
                'stakeholders': dilemma.stakeholders
            }
            
            # GPU context manager 사용 (벤담 계산은 중간 정도 메모리 사용)
            from config import gpu_model_context
            
            with gpu_model_context(bentham_calculator, memory_required_mb=400) as gpu_calculator:
                bentham_result = gpu_calculator.calculate_with_advanced_layers(
                    input_data=bentham_input,
                    use_cache=True
                )
            
            return {
                'final_score': getattr(bentham_result, 'final_score', 0.0),
                'base_score': getattr(bentham_result, 'base_score', 0.0),
                'confidence': getattr(bentham_result, 'confidence', 0.0),
                'intensity': getattr(bentham_result, 'intensity', 0.0),
                'duration': getattr(bentham_result, 'duration', 0.0),
                'certainty': getattr(bentham_result, 'certainty', 0.0)
            }
        except Exception as e:
            logger.error(f"벤담 분석 실패: {e}")
            return {}
    
    async def _analyze_choice_regret(self, choice_context: str, dilemma: EthicalDilemma, choice: EthicalChoice) -> Dict[str, Any]:
        """선택지 후회 분석 - Lazy Loading + GPU Context"""
        
        try:
            # Lazy loading으로 후회 분석기 로드
            regret_analyzer = self._get_regret_analyzer()
            
            regret_input = {
                'scenario': dilemma.description,
                'text': choice_context,
                'chosen_action': choice.name,
                'alternative_actions': [c.name for c in dilemma.extracted_choices if c.id != choice.id],
                'context': dilemma.context
            }
            
            # GPU context manager 사용 (후회 분석은 중간 정도 메모리 사용)
            from config import gpu_model_context
            
            with gpu_model_context(regret_analyzer, memory_required_mb=450) as gpu_analyzer:
                regret_result = await gpu_analyzer.analyze_regret(
                    decision_data=regret_input,
                    outcome_data=None
                )
            
            return {
                'regret_intensity': getattr(regret_result, 'regret_intensity', 0.0),
                'anticipated_regret': getattr(regret_result, 'anticipated_regret', 0.0),
                'experienced_regret': getattr(regret_result, 'experienced_regret', 0.0),
                'model_confidence': getattr(regret_result, 'model_confidence', 0.0)
            }
        except Exception as e:
            logger.error(f"후회 분석 실패: {e}")
            return {}
    
    async def _analyze_choice_surd(self, choice_context: str, dilemma: EthicalDilemma) -> Dict[str, Any]:
        """선택지 SURD 분석 - Lazy Loading + GPU Context"""
        
        try:
            # Lazy loading으로 SURD 분석기 로드
            surd_analyzer = self._get_surd_analyzer()
            
            surd_variables = {
                'ethical_decision_quality': 0.5,
                'stakeholder_satisfaction': 0.5,
                'moral_complexity': dilemma.moral_complexity or 0.5
            }
            
            # GPU context manager 사용 (SURD 분석은 상대적으로 가벼움)
            from config import gpu_model_context
            
            with gpu_model_context(surd_analyzer, memory_required_mb=300) as gpu_analyzer:
                surd_result = gpu_analyzer.analyze_advanced(
                    variables=surd_variables,
                    target_variable='ethical_decision_quality',
                    additional_context=dilemma.context
                )
            
            return {
                'synergy': getattr(surd_result, 'synergy', 0.0),
                'uniqueness': getattr(surd_result, 'uniqueness', 0.0),
                'redundancy': getattr(surd_result, 'redundancy', 0.0),
                'determinism': getattr(surd_result, 'determinism', 0.0)
            }
        except Exception as e:
            logger.error(f"SURD 분석 실패: {e}")
            return {}
    
    async def _generate_choice_counterfactuals(self, choice_context: str, dilemma: EthicalDilemma, choice: EthicalChoice) -> List[Dict[str, Any]]:
        """선택지 반사실적 시나리오 생성"""
        
        try:
            if not self.counterfactual_reasoner:
                return []
            
            # 반사실적 시나리오 생성
            scenarios = await self.counterfactual_reasoner.generate_counterfactual_scenarios(
                base_situation=dilemma.description,
                chosen_action=choice.name,
                alternative_actions=[c.name for c in dilemma.extracted_choices if c.id != choice.id],
                context=dilemma.context
            )
            
            return scenarios or []
        except Exception as e:
            logger.error(f"반사실적 시나리오 생성 실패: {e}")
            return []
    
    def _calculate_utility_score(self, emotion_analysis: Dict, bentham_analysis: Dict, regret_analysis: Dict) -> float:
        """유틸리티 점수 계산"""
        
        bentham_score = bentham_analysis.get('final_score', 0.0)
        emotion_intensity = emotion_analysis.get('intensity', 0.0)
        regret_intensity = regret_analysis.get('regret_intensity', 0.0)
        
        # 유틸리티 = 벤담 점수 + 감정 강도 - 후회 강도
        utility = bentham_score + emotion_intensity - regret_intensity
        return max(0.0, min(1.0, utility))
    
    def _calculate_confidence_score(self, emotion_analysis: Dict, bentham_analysis: Dict, regret_analysis: Dict) -> float:
        """신뢰도 점수 계산"""
        
        confidences = []
        
        if 'confidence' in emotion_analysis:
            confidences.append(emotion_analysis['confidence'])
        if 'confidence' in bentham_analysis:
            confidences.append(bentham_analysis['confidence'])
        if 'model_confidence' in regret_analysis:
            confidences.append(regret_analysis['model_confidence'])
        
        return sum(confidences) / len(confidences) if confidences else 0.5
    
    def _recommend_optimal_choice(self, dilemma: EthicalDilemma) -> Optional[EthicalChoice]:
        """최적 선택지 추천"""
        
        if not dilemma.choice_analyses:
            return None
        
        # 위험 조정 점수 기준으로 최적 선택지 선택
        best_choice = None
        best_score = -1
        
        for choice_id, analysis in dilemma.choice_analyses.items():
            if analysis.risk_adjusted_score > best_score:
                best_score = analysis.risk_adjusted_score
                best_choice = analysis.choice
        
        return best_choice
    
    async def _generate_reasoning_chain(self, dilemma: EthicalDilemma) -> List[str]:
        """추론 체인 생성"""
        
        reasoning = []
        
        # 1. 상황 분석
        reasoning.append(f"상황 분석: {dilemma.dilemma_type.value} 유형의 윤리적 딜레마")
        
        # 2. 선택지 분석
        reasoning.append(f"식별된 선택지: {len(dilemma.extracted_choices)}개")
        
        # 3. 이해관계자 분석
        reasoning.append(f"주요 이해관계자: {', '.join(dilemma.stakeholders.keys())}")
        
        # 4. 최적 선택지 근거
        if dilemma.recommended_choice:
            best_analysis = dilemma.choice_analyses.get(dilemma.recommended_choice.id)
            if best_analysis:
                reasoning.append(f"최적 선택지: {dilemma.recommended_choice.name}")
                reasoning.append(f"유틸리티 점수: {best_analysis.utility_score:.3f}")
                reasoning.append(f"신뢰도: {best_analysis.confidence_score:.3f}")
                reasoning.append(f"위험 조정 점수: {best_analysis.risk_adjusted_score:.3f}")
        
        return reasoning

# 테스트 함수
async def test_dynamic_analyzer():
    """동적 분석기 테스트"""
    
    analyzer = DynamicEthicalChoiceAnalyzer()
    
    # 다양한 윤리적 딜레마 테스트
    test_scenarios = [
        {
            "title": "의료진 자원 배분 딜레마",
            "description": "코로나19 상황에서 인공호흡기 1대를 두고 90세 환자와 30세 환자 중 누구를 선택할 것인가? 나이를 고려할 것인가, 아니면 선착순으로 할 것인가?",
        },
        {
            "title": "개인정보 vs 공공안전",
            "description": "테러 방지를 위해 시민들의 개인정보를 수집하고 감시할 것인가, 아니면 개인의 프라이버시를 보호할 것인가?",
        },
        {
            "title": "친구의 부정행위",
            "description": "시험에서 친구가 부정행위를 하는 것을 목격했다. 신고해야 할까, 아니면 친구를 보호해야 할까?",
        }
    ]
    
    for scenario in test_scenarios:
        logger.info(f"\n{'='*60}")
        logger.info(f"테스트 시나리오: {scenario['title']}")
        logger.info(f"{'='*60}")
        
        result = await analyzer.analyze_ethical_dilemma(
            dilemma_text=scenario['description'],
            title=scenario['title']
        )
        
        logger.info(f"✅ 분석 완료")
        logger.info(f"딜레마 유형: {result.dilemma_type.value}")
        logger.info(f"추출된 선택지: {len(result.extracted_choices)}개")
        logger.info(f"이해관계자: {len(result.stakeholders)}명")
        
        if result.recommended_choice:
            logger.info(f"🎯 최종 추천: {result.recommended_choice.name}")
            
            best_analysis = result.choice_analyses.get(result.recommended_choice.id)
            if best_analysis:
                logger.info(f"   유틸리티 점수: {best_analysis.utility_score:.3f}")
                logger.info(f"   신뢰도: {best_analysis.confidence_score:.3f}")
        
        logger.info(f"🧠 추론 체인:")
        for i, reason in enumerate(result.reasoning_chain):
            logger.info(f"   {i+1}. {reason}")

if __name__ == "__main__":
    asyncio.run(test_dynamic_analyzer())