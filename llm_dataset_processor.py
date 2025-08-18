"""
LLM 기반 데이터셋 통합 처리 시스템
for_learn_dataset → 벤담 변수 + entailment 라벨 추출
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
# pathlib 제거 - WSL 호환성을 위해 os.path 사용
from dataclasses import dataclass, field
import re
from datetime import datetime
# import aiofiles  # 사용하지 않음
import uuid

# 프로젝트 모듈
from llm_module.advanced_llm_engine import get_llm_engine, LLMRequest, TaskComplexity

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('LLMDatasetProcessor')

@dataclass
class BenthamFactors:
    """벤담 쾌락 계산 7개 변수"""
    intensity: float        # 강도 (0-1)
    duration: float         # 지속성 (0-1) 
    certainty: float        # 확실성 (0-1)
    propinquity: float      # 근접성 (0-1)
    fecundity: float        # 다산성 (0-1)
    purity: float           # 순수성 (0-1)
    extent: float           # 범위 (0-1)

@dataclass
class EntailmentPair:
    """Entailment 학습용 데이터"""
    premise: str
    hypothesis: str
    label: str  # ENTAILS, CONTRADICTS, NEUTRAL

@dataclass
class ProcessedScenario:
    """처리된 시나리오 데이터"""
    id: str
    title: str
    description: str
    source_type: str
    category: str
    stakeholders: List[str]
    ethical_themes: List[str]
    
    # LLM으로 추출된 데이터
    bentham_factors: BenthamFactors
    entailment_pairs: List[EntailmentPair] = field(default_factory=list)
    binary_label: Optional[str] = None  # RIGHT/WRONG
    complexity_score: float = 0.5
    controversy_score: float = 0.5
    
    # 메타데이터
    cultural_context: str = "general"
    language: str = "ko"
    processing_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

class DatasetSourceParser:
    """다양한 데이터 소스 파싱 클래스"""
    
    def __init__(self):
        self.logger = logging.getLogger(f'{__name__}.DatasetSourceParser')
    
    def parse_ebs_literature(self, file_path: str) -> List[Dict[str, Any]]:
        """EBS 한국문학 데이터 파싱"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        scenarios = []
        # 작품별로 분할 (작품명이 첫 줄에 있음)
        sections = content.split('\n\n')
        
        current_scenario = {}
        for section in sections:
            lines = section.strip().split('\n')
            if not lines:
                continue
                
            # 작품명으로 시작하는 새 시나리오
            if len(lines) > 1 and '상황설명:' in section:
                if current_scenario:
                    scenarios.append(current_scenario)
                
                current_scenario = {
                    'title': lines[0].strip(),
                    'description': '',
                    'stakeholders': [],
                    'ethical_dilemma': '',
                    'emotions': {},
                    'choices': []
                }
                
                # 상황설명 추출
                for line in lines:
                    if line.startswith('상황설명:'):
                        current_scenario['description'] = line.replace('상황설명:', '').strip()
                    elif line.startswith('이해관계자:'):
                        stakeholders_text = line.replace('이해관계자:', '').strip()
                        current_scenario['stakeholders'] = [s.strip().strip("'\"") for s in stakeholders_text.split(',')]
                    elif line.startswith('윤리적 딜레마:'):
                        current_scenario['ethical_dilemma'] = line.replace('윤리적 딜레마:', '').strip()
                    elif line.startswith('선택지'):
                        current_scenario['choices'].append(line.strip())
        
        if current_scenario:
            scenarios.append(current_scenario)
            
        return scenarios
    
    def parse_ai_generated(self, file_path: str) -> List[Dict[str, Any]]:
        """AI 생성 데이터 파싱"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        scenarios = []
        # "사례 N:" 패턴으로 분할
        case_pattern = r'사례 \d+:'
        cases = re.split(case_pattern, content)
        
        for i, case_content in enumerate(cases[1:], 1):  # 첫 번째는 빈 문자열
            lines = case_content.strip().split('\n')
            if len(lines) < 5:
                continue
                
            scenario = {
                'title': f"AI 생성 사례 {i}",
                'description': '',
                'stakeholders': [],
                'situation_title': '',
                'values': [],
                'choices': []
            }
            
            current_section = None
            for line in lines:
                line = line.strip()
                if '상황 제목:' in line:
                    scenario['situation_title'] = line.split('상황 제목:', 1)[1].strip()
                elif '상황 설명:' in line:
                    current_section = 'description'
                    desc_start = line.split('상황 설명:', 1)
                    if len(desc_start) > 1:
                        scenario['description'] = desc_start[1].strip()
                elif '관련된 핵심 가치/원칙:' in line:
                    current_section = 'values'
                elif '선택 가능한 옵션들:' in line or '의사결정 과정' in line:
                    current_section = 'choices'
                elif current_section == 'description' and line:
                    scenario['description'] += ' ' + line
                elif current_section == 'values' and line:
                    scenario['values'].append(line)
                elif current_section == 'choices' and line:
                    scenario['choices'].append(line)
            
            scenarios.append(scenario)
        
        return scenarios
    
    def parse_classic_literature(self, file_path: str) -> List[Dict[str, Any]]:
        """고전 문학 데이터 파싱 (큰 텍스트를 청크로 분할)"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Project Gutenberg 헤더 제거
        if 'Project Gutenberg' in content:
            parts = content.split('***')
            if len(parts) >= 3:
                content = parts[1]  # 실제 텍스트 부분만
        
        # 책 제목 추출
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        book_title = file_name.replace('_', ' ').title()
        
        # 텍스트를 적당한 크기로 청크 분할 (윤리적 상황 추출용)
        chunk_size = 2000  # 2000자 단위
        chunks = []
        
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i + chunk_size]
            if len(chunk.strip()) > 500:  # 너무 작은 청크는 제외
                chunks.append({
                    'title': f"{book_title} - 구간 {i//chunk_size + 1}",
                    'description': chunk.strip(),
                    'source_file': file_name,
                    'chunk_index': i//chunk_size
                })
        
        return chunks
    
    def parse_scruples_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """Scruples JSONL 데이터 파싱"""
        scenarios = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data = json.loads(line.strip())
                        scenario = {
                            'title': data.get('title', f'Scruples Case {line_num}'),
                            'description': data.get('text', ''),
                            'action_description': data.get('action', {}).get('description', ''),
                            'label': data.get('label', ''),
                            'binarized_label': data.get('binarized_label', ''),
                            'post_type': data.get('post_type', ''),
                            'original_id': data.get('id', '')
                        }
                        scenarios.append(scenario)
                    except json.JSONDecodeError:
                        logger.warning(f"JSON 파싱 실패: {file_path}:{line_num}")
        
        return scenarios

class LLMDataProcessor:
    """LLM을 사용한 데이터 처리 클래스"""
    
    def __init__(self):
        self.llm_engine = None
        self.logger = logging.getLogger(f'{__name__}.LLMDataProcessor')
    
    async def initialize(self):
        """LLM 엔진 초기화"""
        try:
            self.llm_engine = get_llm_engine()
            self.logger.info("LLM 엔진 초기화 완료")
        except Exception as e:
            self.logger.error(f"LLM 엔진 초기화 실패: {e}")
            raise
    
    async def extract_bentham_factors(self, scenario_text: str) -> BenthamFactors:
        """텍스트에서 벤담 쾌락 계산 변수 추출"""
        prompt = f"""
다음 윤리적 상황을 분석하여 벤담의 쾌락 계산법 7개 변수를 0.0~1.0 사이의 값으로 평가해주세요.

상황: {scenario_text[:1500]}

다음 변수들을 평가해주세요:
1. intensity (강도): 쾌락이나 고통의 강도가 얼마나 큰가?
2. duration (지속성): 그 영향이 얼마나 오래 지속되는가?
3. certainty (확실성): 그 결과가 일어날 확률이 얼마나 높은가?
4. propinquity (근접성): 그 결과가 얼마나 빨리 나타나는가?
5. fecundity (다산성): 비슷한 결과를 낳을 가능성이 얼마나 높은가?
6. purity (순수성): 반대 효과 없이 순수한 결과를 낳을 가능성이 얼마나 높은가?
7. extent (범위): 얼마나 많은 사람들이 영향을 받는가?

JSON 형식으로 응답해주세요:
{{"intensity": 0.7, "duration": 0.5, "certainty": 0.8, "propinquity": 0.6, "fecundity": 0.4, "purity": 0.5, "extent": 0.9}}
"""
        
        try:
            request = LLMRequest(
                prompt=prompt,
                task_type="bentham_analysis",
                complexity=TaskComplexity.MODERATE
            )
            response = self.llm_engine.generate_sync(request)
            response_text = response.generated_text
            
            # JSON 추출
            json_match = re.search(r'\{[^}]+\}', response_text)
            if json_match:
                factors_dict = json.loads(json_match.group())
                return BenthamFactors(
                    intensity=float(factors_dict.get('intensity', 0.5)),
                    duration=float(factors_dict.get('duration', 0.5)),
                    certainty=float(factors_dict.get('certainty', 0.5)),
                    propinquity=float(factors_dict.get('propinquity', 0.5)),
                    fecundity=float(factors_dict.get('fecundity', 0.5)),
                    purity=float(factors_dict.get('purity', 0.5)),
                    extent=float(factors_dict.get('extent', 0.5))
                )
            else:
                self.logger.warning("벤담 변수 JSON 추출 실패, 기본값 사용")
                return BenthamFactors(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
                
        except Exception as e:
            self.logger.error(f"벤담 변수 추출 실패: {e}")
            return BenthamFactors(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    
    async def generate_entailment_pairs(self, scenario_text: str) -> List[EntailmentPair]:
        """시나리오에서 entailment 학습용 데이터 생성"""
        prompt = f"""
다음 윤리적 상황에서 entailment 학습용 데이터를 생성해주세요.

상황: {scenario_text[:1000]}

premise-hypothesis 쌍을 3개 생성하고, 각각에 대해 ENTAILS, CONTRADICTS, NEUTRAL 중 하나의 라벨을 부여해주세요.

JSON 형식으로 응답해주세요:
[
  {{"premise": "상황 설명 또는 행동", "hypothesis": "결과나 판단", "label": "ENTAILS"}},
  {{"premise": "다른 상황", "hypothesis": "다른 결과", "label": "CONTRADICTS"}},
  {{"premise": "또 다른 상황", "hypothesis": "중립적 결과", "label": "NEUTRAL"}}
]
"""
        
        try:
            request = LLMRequest(
                prompt=prompt,
                task_type="entailment_generation",
                complexity=TaskComplexity.MODERATE
            )
            response = self.llm_engine.generate_sync(request)
            response_text = response.generated_text
            
            # JSON 배열 추출
            json_match = re.search(r'\[[^\]]+\]', response_text, re.DOTALL)
            if json_match:
                pairs_list = json.loads(json_match.group())
                entailment_pairs = []
                for pair in pairs_list:
                    entailment_pairs.append(EntailmentPair(
                        premise=pair.get('premise', ''),
                        hypothesis=pair.get('hypothesis', ''),
                        label=pair.get('label', 'NEUTRAL')
                    ))
                return entailment_pairs
            else:
                self.logger.warning("Entailment 데이터 JSON 추출 실패")
                return []
                
        except Exception as e:
            self.logger.error(f"Entailment 데이터 생성 실패: {e}")
            return []
    
    async def extract_metadata(self, scenario_text: str, source_type: str) -> Dict[str, Any]:
        """메타데이터 추출 (카테고리, 이해관계자, 윤리적 주제 등)"""
        prompt = f"""
다음 윤리적 상황을 분석하여 메타데이터를 추출해주세요.

상황: {scenario_text[:1000]}

다음 정보를 추출해주세요:
1. category: legal, medical, business, personal, social, academic 중 하나
2. stakeholders: 관련된 이해관계자들 (최대 5개)
3. ethical_themes: 윤리적 주제들 (예: autonomy, justice, beneficence, 등)
4. complexity_score: 상황의 복잡도 (0.0~1.0)
5. controversy_score: 논란의 정도 (0.0~1.0)
6. binary_label: RIGHT 또는 WRONG (명확한 경우만)

JSON 형식으로 응답해주세요:
{{
  "category": "personal",
  "stakeholders": ["개인", "가족", "친구"],
  "ethical_themes": ["autonomy", "loyalty", "honesty"],
  "complexity_score": 0.7,
  "controversy_score": 0.5,
  "binary_label": "RIGHT"
}}
"""
        
        try:
            request = LLMRequest(
                prompt=prompt,
                task_type="metadata_extraction",
                complexity=TaskComplexity.MODERATE
            )
            response = self.llm_engine.generate_sync(request)
            response_text = response.generated_text
            
            json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
            if json_match:
                metadata = json.loads(json_match.group())
                return {
                    'category': metadata.get('category', 'general'),
                    'stakeholders': metadata.get('stakeholders', []),
                    'ethical_themes': metadata.get('ethical_themes', []),
                    'complexity_score': float(metadata.get('complexity_score', 0.5)),
                    'controversy_score': float(metadata.get('controversy_score', 0.5)),
                    'binary_label': metadata.get('binary_label')
                }
            else:
                self.logger.warning("메타데이터 JSON 추출 실패, 기본값 사용")
                return {
                    'category': 'general',
                    'stakeholders': [],
                    'ethical_themes': [],
                    'complexity_score': 0.5,
                    'controversy_score': 0.5,
                    'binary_label': None
                }
                
        except Exception as e:
            self.logger.error(f"메타데이터 추출 실패: {e}")
            return {
                'category': 'general',
                'stakeholders': [],
                'ethical_themes': [],
                'complexity_score': 0.5,
                'controversy_score': 0.5,
                'binary_label': None
            }

class DatasetProcessor:
    """전체 데이터셋 처리 통합 클래스"""
    
    def __init__(self, output_dir: str = "enhanced_training_data"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.parser = DatasetSourceParser()
        self.llm_processor = LLMDataProcessor()
        self.logger = logging.getLogger(f'{__name__}.DatasetProcessor')
        
        # 청크 크기 설정 (컨텍스트 제한 방지)
        self.max_chunk_size = 50  # 한 번에 처리할 시나리오 수
    
    async def initialize(self):
        """초기화"""
        await self.llm_processor.initialize()
        self.logger.info("데이터셋 프로세서 초기화 완료")
    
    async def process_data_source(self, source_path: str, source_type: str) -> List[ProcessedScenario]:
        """데이터 소스별 처리"""
        self.logger.info(f"{source_type} 처리 시작: {source_path}")
        
        # 데이터 파싱
        raw_scenarios = []
        if source_type == "ebs_literature":
            raw_scenarios = self.parser.parse_ebs_literature(source_path)
        elif source_type == "ai_generated":
            raw_scenarios = self.parser.parse_ai_generated(source_path)
        elif source_type == "classic_literature":
            raw_scenarios = self.parser.parse_classic_literature(source_path)
        elif source_type == "scruples_jsonl":
            raw_scenarios = self.parser.parse_scruples_jsonl(source_path)
        
        self.logger.info(f"파싱된 시나리오 수: {len(raw_scenarios)}")
        
        # 청크별 처리 (컨텍스트 제한 방지)
        processed_scenarios = []
        for i in range(0, len(raw_scenarios), self.max_chunk_size):
            chunk = raw_scenarios[i:i + self.max_chunk_size]
            chunk_results = await self._process_chunk(chunk, source_type, i)
            processed_scenarios.extend(chunk_results)
            
            # 중간 저장
            await self._save_chunk_results(chunk_results, source_type, i)
            
            self.logger.info(f"청크 {i//self.max_chunk_size + 1} 처리 완료 ({len(chunk)} 시나리오)")
        
        return processed_scenarios
    
    async def _process_chunk(self, raw_scenarios: List[Dict], source_type: str, chunk_index: int) -> List[ProcessedScenario]:
        """청크 단위 시나리오 처리"""
        processed = []
        
        for idx, raw_scenario in enumerate(raw_scenarios):
            try:
                # 텍스트 준비
                main_text = raw_scenario.get('description', '') or raw_scenario.get('text', '')
                if not main_text:
                    continue
                
                # LLM으로 데이터 추출
                bentham_factors = await self.llm_processor.extract_bentham_factors(main_text)
                entailment_pairs = await self.llm_processor.generate_entailment_pairs(main_text)
                metadata = await self.llm_processor.extract_metadata(main_text, source_type)
                
                # ProcessedScenario 생성
                scenario = ProcessedScenario(
                    id=f"{source_type}_{chunk_index}_{idx}_{uuid.uuid4().hex[:8]}",
                    title=raw_scenario.get('title', f'{source_type} 시나리오 {idx+1}'),
                    description=main_text,
                    source_type=source_type,
                    category=metadata['category'],
                    stakeholders=metadata['stakeholders'],
                    ethical_themes=metadata['ethical_themes'],
                    bentham_factors=bentham_factors,
                    entailment_pairs=entailment_pairs,
                    binary_label=metadata['binary_label'],
                    complexity_score=metadata['complexity_score'],
                    controversy_score=metadata['controversy_score']
                )
                
                processed.append(scenario)
                
                # 처리 진행상황 로깅
                if (idx + 1) % 10 == 0:
                    self.logger.info(f"청크 내 {idx + 1}/{len(raw_scenarios)} 시나리오 처리 완료")
                    
            except Exception as e:
                self.logger.error(f"시나리오 처리 실패 ({source_type}_{chunk_index}_{idx}): {e}")
                continue
        
        return processed
    
    async def _save_chunk_results(self, scenarios: List[ProcessedScenario], source_type: str, chunk_index: int):
        """청크 결과 저장"""
        chunk_file = os.path.join(self.output_dir, f"{source_type}_chunk_{chunk_index:03d}.json")
        
        # dataclass를 dict로 변환
        scenarios_dict = []
        for scenario in scenarios:
            scenario_dict = {
                'id': scenario.id,
                'title': scenario.title,
                'description': scenario.description,
                'source_type': scenario.source_type,
                'category': scenario.category,
                'stakeholders': scenario.stakeholders,
                'ethical_themes': scenario.ethical_themes,
                'bentham_factors': {
                    'intensity': scenario.bentham_factors.intensity,
                    'duration': scenario.bentham_factors.duration,
                    'certainty': scenario.bentham_factors.certainty,
                    'propinquity': scenario.bentham_factors.propinquity,
                    'fecundity': scenario.bentham_factors.fecundity,
                    'purity': scenario.bentham_factors.purity,
                    'extent': scenario.bentham_factors.extent
                },
                'entailment_pairs': [
                    {
                        'premise': pair.premise,
                        'hypothesis': pair.hypothesis,
                        'label': pair.label
                    } for pair in scenario.entailment_pairs
                ],
                'binary_label': scenario.binary_label,
                'complexity_score': scenario.complexity_score,
                'controversy_score': scenario.controversy_score,
                'cultural_context': scenario.cultural_context,
                'language': scenario.language,
                'processing_timestamp': scenario.processing_timestamp
            }
            scenarios_dict.append(scenario_dict)
        
        with open(chunk_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(scenarios_dict, ensure_ascii=False, indent=2))
        
        self.logger.info(f"청크 저장 완료: {chunk_file}")
    
    async def process_all_sources(self):
        """모든 데이터 소스 처리"""
        source_configs = [
            # EBS 한국문학
            {
                'path': 'for_learn_dataset/ai_ebs/ebs_1.txt',
                'type': 'ebs_literature'
            },
            {
                'path': 'for_learn_dataset/ai_ebs/ebs_2.txt', 
                'type': 'ebs_literature'
            },
            {
                'path': 'for_learn_dataset/ai_ebs/ebs_3.txt',
                'type': 'ebs_literature'
            },
            
            # AI 생성 데이터
            {
                'path': 'for_learn_dataset/ai_generated_dataset/raw_data_by_claude_210.txt',
                'type': 'ai_generated'
            },
            {
                'path': 'for_learn_dataset/ai_generated_dataset/raw_data_novel_123.txt',
                'type': 'ai_generated' 
            },
            
            # 고전 문학 (작은 것들만 우선)
            {
                'path': 'for_learn_dataset/book/romeo_and_juliet.txt',
                'type': 'classic_literature'
            },
            
            # Scruples 데이터
            {
                'path': 'for_learn_dataset/scruples_real_data/anecdotes/train.scruples-anecdotes.jsonl',
                'type': 'scruples_jsonl'
            },
            {
                'path': 'for_learn_dataset/scruples_real_data/dilemmas/train.scruples-dilemmas.jsonl',
                'type': 'scruples_jsonl'
            }
        ]
        
        all_processed = []
        
        for config in source_configs:
            source_path = config['path']
            source_type = config['type']
            
            if os.path.exists(source_path):
                try:
                    processed = await self.process_data_source(source_path, source_type)
                    all_processed.extend(processed)
                    self.logger.info(f"{source_type} 처리 완료: {len(processed)} 시나리오")
                except Exception as e:
                    self.logger.error(f"{source_type} 처리 실패: {e}")
            else:
                self.logger.warning(f"파일 없음: {source_path}")
        
        # 최종 통합 파일 저장
        await self._save_final_results(all_processed)
        
        return all_processed
    
    async def _save_final_results(self, all_scenarios: List[ProcessedScenario]):
        """최종 결과 저장"""
        final_file = os.path.join(self.output_dir, "enhanced_training_scenarios.json")
        
        # 통계 정보
        stats = {
            'total_scenarios': len(all_scenarios),
            'by_source_type': {},
            'by_category': {},
            'processing_timestamp': datetime.now().isoformat()
        }
        
        scenarios_dict = []
        for scenario in all_scenarios:
            # 통계 업데이트
            stats['by_source_type'][scenario.source_type] = stats['by_source_type'].get(scenario.source_type, 0) + 1
            stats['by_category'][scenario.category] = stats['by_category'].get(scenario.category, 0) + 1
            
            # 시나리오 딕셔너리 변환
            scenario_dict = {
                'id': scenario.id,
                'title': scenario.title,
                'description': scenario.description,
                'source_type': scenario.source_type,
                'category': scenario.category,
                'stakeholders': scenario.stakeholders,
                'ethical_themes': scenario.ethical_themes,
                'bentham_factors': {
                    'intensity': scenario.bentham_factors.intensity,
                    'duration': scenario.bentham_factors.duration,
                    'certainty': scenario.bentham_factors.certainty,
                    'propinquity': scenario.bentham_factors.propinquity,
                    'fecundity': scenario.bentham_factors.fecundity,
                    'purity': scenario.bentham_factors.purity,
                    'extent': scenario.bentham_factors.extent
                },
                'entailment_pairs': [
                    {
                        'premise': pair.premise,
                        'hypothesis': pair.hypothesis,
                        'label': pair.label
                    } for pair in scenario.entailment_pairs
                ],
                'binary_label': scenario.binary_label,
                'complexity_score': scenario.complexity_score,
                'controversy_score': scenario.controversy_score,
                'cultural_context': scenario.cultural_context,
                'language': scenario.language,
                'processing_timestamp': scenario.processing_timestamp
            }
            scenarios_dict.append(scenario_dict)
        
        # 최종 파일 저장
        final_data = {
            'metadata': stats,
            'scenarios': scenarios_dict
        }
        
        with open(final_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(final_data, ensure_ascii=False, indent=2))
        
        # 통계 파일 저장
        stats_file = os.path.join(self.output_dir, "processing_statistics.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(stats, ensure_ascii=False, indent=2))
        
        self.logger.info(f"최종 결과 저장 완료: {final_file}")
        self.logger.info(f"총 {len(all_scenarios)}개 시나리오 처리 완료")

async def main():
    """메인 실행 함수"""
    processor = DatasetProcessor()
    
    try:
        await processor.initialize()
        await processor.process_all_sources()
        print("✅ 모든 데이터 소스 처리 완료")
    except Exception as e:
        print(f"❌ 처리 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())