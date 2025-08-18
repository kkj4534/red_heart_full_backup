"""
고품질 데이터 유지하면서 최적화된 LLM 데이터셋 처리 시스템
High-Quality Data Processing with Performance Optimization
"""

import os
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
import re
from datetime import datetime
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 프로젝트 모듈
from llm_module.advanced_llm_engine import get_llm_engine, LLMRequest, TaskComplexity

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('OptimizedDatasetProcessor')

@dataclass
class BenthamFactors:
    """벤담 쾌락 계산 7개 변수"""
    intensity: float = 0.5
    duration: float = 0.5
    certainty: float = 0.5
    propinquity: float = 0.5
    fecundity: float = 0.5
    purity: float = 0.5
    extent: float = 0.5

@dataclass
class EnhancedScenario:
    """고품질 처리된 시나리오"""
    id: str
    title: str
    description: str
    source_type: str
    
    # 고품질 추출 데이터
    bentham_factors: BenthamFactors
    category: str = "general"
    stakeholders: List[str] = field(default_factory=list)
    ethical_themes: List[str] = field(default_factory=list)
    binary_label: Optional[str] = None
    complexity_score: float = 0.5
    controversy_score: float = 0.5
    
    # 메타데이터
    processing_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    processing_time: float = 0.0
    quality_score: float = 1.0

class SmartDataParser:
    """스마트 데이터 파싱 - 파싱 최적화"""
    
    def __init__(self):
        self.logger = logging.getLogger(f'{__name__}.SmartDataParser')
    
    def parse_ebs_literature_optimized(self, file_path: str, max_scenarios: int = None) -> List[Dict[str, Any]]:
        """EBS 문학 데이터 최적화 파싱"""
        self.logger.info(f"EBS 문학 파싱 시작: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        scenarios = []
        current_scenario = {}
        
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 새 작품 시작 (첫 글자가 한글이고 괄호 포함)
            if re.match(r'^[가-힣].+\([^)]+\)$', line):
                if current_scenario and current_scenario.get('description'):
                    scenarios.append(current_scenario)
                    if max_scenarios and len(scenarios) >= max_scenarios:
                        break
                
                current_scenario = {
                    'title': line,
                    'description': '',
                    'stakeholders': [],
                    'raw_text': line + '\n'
                }
            
            elif line.startswith('상황설명:'):
                current_scenario['description'] = line.replace('상황설명:', '').strip()
                current_scenario['raw_text'] += line + '\n'
            
            elif line.startswith('이해관계자:'):
                stakeholders_text = line.replace('이해관계자:', '').strip()
                current_scenario['stakeholders'] = [
                    s.strip().strip("'\"") for s in stakeholders_text.split(',')
                ]
                current_scenario['raw_text'] += line + '\n'
            
            elif current_scenario:
                current_scenario['raw_text'] += line + '\n'
        
        # 마지막 시나리오 추가
        if current_scenario and current_scenario.get('description'):
            scenarios.append(current_scenario)
        
        self.logger.info(f"파싱 완료: {len(scenarios)}개 시나리오")
        return scenarios

class HighQualityLLMProcessor:
    """고품질 LLM 처리 - 품질 우선"""
    
    def __init__(self):
        self.llm_engine = None
        self.logger = logging.getLogger(f'{__name__}.HighQualityLLMProcessor')
        
        # 품질 관리 설정
        self.quality_thresholds = {
            'min_response_length': 50,
            'required_json_fields': ['intensity', 'duration', 'certainty'],
            'max_retry_attempts': 2
        }
        
        # 프롬프트 템플릿 최적화
        self.bentham_prompt_template = """
다음은 윤리적 상황입니다. 벤담의 쾌락 계산법 7개 변수를 정확히 분석해주세요.

상황: {scenario_text}

각 변수를 0.0~1.0 사이 값으로 평가:
- intensity: 쾌락/고통의 강도
- duration: 영향의 지속 시간  
- certainty: 결과의 확실성
- propinquity: 시간적 근접성
- fecundity: 유사한 결과를 낳을 가능성
- purity: 반대 효과 없이 순수한 결과일 가능성
- extent: 영향받는 사람의 수

반드시 다음 JSON 형식으로만 응답하세요:
{{"intensity": 0.0, "duration": 0.0, "certainty": 0.0, "propinquity": 0.0, "fecundity": 0.0, "purity": 0.0, "extent": 0.0}}
"""
    
    def initialize(self):
        """LLM 엔진 초기화"""
        try:
            self.llm_engine = get_llm_engine()
            self.logger.info("고품질 LLM 엔진 초기화 완료")
        except Exception as e:
            self.logger.error(f"LLM 엔진 초기화 실패: {e}")
            raise
    
    def extract_bentham_factors_hq(self, scenario_text: str) -> Tuple[BenthamFactors, float]:
        """고품질 벤담 변수 추출"""
        if not self.llm_engine:
            raise RuntimeError("LLM 엔진이 초기화되지 않았습니다.")
        
        # 텍스트 길이 최적화 (너무 길면 잘라내기)
        if len(scenario_text) > 1000:
            scenario_text = scenario_text[:1000] + "..."
        
        prompt = self.bentham_prompt_template.format(scenario_text=scenario_text)
        
        for attempt in range(self.quality_thresholds['max_retry_attempts'] + 1):
            try:
                start_time = time.time()
                
                request = LLMRequest(
                    prompt=prompt,
                    task_type="bentham_analysis",
                    complexity=TaskComplexity.MODERATE
                )
                
                response = self.llm_engine.generate_sync(request)
                processing_time = time.time() - start_time
                
                # 품질 검증 및 파싱
                factors, quality_score = self._parse_and_validate_bentham_response(
                    response.generated_text
                )
                
                if quality_score >= 0.8:  # 고품질 기준
                    self.logger.debug(f"고품질 벤담 추출 완료 (품질점수: {quality_score:.2f})")
                    return factors, quality_score
                elif attempt < self.quality_thresholds['max_retry_attempts']:
                    self.logger.warning(f"품질 미달 (시도 {attempt + 1}), 재시도...")
                    continue
                else:
                    self.logger.warning(f"최종 시도 - 낮은 품질로 진행 (품질점수: {quality_score:.2f})")
                    return factors, quality_score
                    
            except Exception as e:
                if attempt < self.quality_thresholds['max_retry_attempts']:
                    self.logger.warning(f"벤담 추출 실패 (시도 {attempt + 1}): {e}, 재시도...")
                    continue
                else:
                    self.logger.error(f"벤담 추출 최종 실패: {e}")
                    return BenthamFactors(), 0.0
        
        return BenthamFactors(), 0.0
    
    def _parse_and_validate_bentham_response(self, response_text: str) -> Tuple[BenthamFactors, float]:
        """벤담 응답 파싱 및 품질 검증"""
        quality_score = 0.0
        
        # 기본 응답 길이 검증
        if len(response_text) < self.quality_thresholds['min_response_length']:
            return BenthamFactors(), 0.1
        
        # JSON 추출
        json_match = re.search(r'\{[^}]+\}', response_text)
        if not json_match:
            return BenthamFactors(), 0.2
        
        try:
            factors_dict = json.loads(json_match.group())
            quality_score += 0.3  # JSON 파싱 성공
            
            # 필수 필드 존재 검증
            required_fields = ['intensity', 'duration', 'certainty', 'propinquity', 'fecundity', 'purity', 'extent']
            missing_fields = [field for field in required_fields if field not in factors_dict]
            
            if not missing_fields:
                quality_score += 0.3  # 모든 필드 존재
            else:
                quality_score += 0.1  # 일부 필드만 존재
            
            # 값 범위 검증 (0.0~1.0)
            valid_values = 0
            for field in required_fields:
                value = factors_dict.get(field, 0.5)
                try:
                    float_value = float(value)
                    if 0.0 <= float_value <= 1.0:
                        valid_values += 1
                except (ValueError, TypeError):
                    pass
            
            quality_score += (valid_values / len(required_fields)) * 0.4  # 값 유효성
            
            # BenthamFactors 객체 생성
            factors = BenthamFactors(
                intensity=float(factors_dict.get('intensity', 0.5)),
                duration=float(factors_dict.get('duration', 0.5)),
                certainty=float(factors_dict.get('certainty', 0.5)),
                propinquity=float(factors_dict.get('propinquity', 0.5)),
                fecundity=float(factors_dict.get('fecundity', 0.5)),
                purity=float(factors_dict.get('purity', 0.5)),
                extent=float(factors_dict.get('extent', 0.5))
            )
            
            return factors, quality_score
            
        except json.JSONDecodeError:
            return BenthamFactors(), 0.2

class OptimizedDatasetProcessor:
    """최적화된 데이터셋 처리기 - 품질과 성능의 균형"""
    
    def __init__(self, output_dir: str = "high_quality_training_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.parser = SmartDataParser()
        self.llm_processor = HighQualityLLMProcessor()
        self.logger = logging.getLogger(f'{__name__}.OptimizedDatasetProcessor')
        
        # 성능 최적화 설정
        self.max_concurrent_llm_calls = 1  # LLM은 순차 처리 (안정성)
        self.chunk_size = 10  # 작은 청크로 자주 저장
        self.enable_parallel_parsing = True  # 파싱은 병렬화
        
        # 품질 통계
        self.quality_stats = {
            'total_processed': 0,
            'high_quality': 0,
            'medium_quality': 0,
            'low_quality': 0,
            'failed': 0,
            'average_quality': 0.0,
            'average_processing_time': 0.0
        }
    
    def initialize(self):
        """초기화"""
        self.llm_processor.initialize()
        self.logger.info("최적화된 데이터셋 프로세서 초기화 완료")
    
    def process_single_scenario(self, raw_scenario: Dict[str, Any], source_type: str, index: int) -> EnhancedScenario:
        """단일 시나리오 고품질 처리"""
        start_time = time.time()
        
        try:
            # 텍스트 준비
            main_text = raw_scenario.get('description', '') or raw_scenario.get('raw_text', '')
            if not main_text:
                raise ValueError("처리할 텍스트가 없습니다.")
            
            # 고품질 벤담 변수 추출
            bentham_factors, quality_score = self.llm_processor.extract_bentham_factors_hq(main_text)
            
            processing_time = time.time() - start_time
            
            # EnhancedScenario 생성
            scenario = EnhancedScenario(
                id=f"{source_type}_{index}_{uuid.uuid4().hex[:8]}",
                title=raw_scenario.get('title', f'{source_type} 시나리오 {index+1}'),
                description=main_text[:2000],  # 길이 제한
                source_type=source_type,
                bentham_factors=bentham_factors,
                stakeholders=raw_scenario.get('stakeholders', []),
                processing_time=processing_time,
                quality_score=quality_score
            )
            
            # 품질 통계 업데이트
            self._update_quality_stats(quality_score, processing_time)
            
            return scenario
            
        except Exception as e:
            self.logger.error(f"시나리오 처리 실패 ({source_type}_{index}): {e}")
            self.quality_stats['failed'] += 1
            
            # 실패 시에도 기본 시나리오 반환
            return EnhancedScenario(
                id=f"{source_type}_{index}_failed",
                title=raw_scenario.get('title', 'Failed Scenario'),
                description=raw_scenario.get('description', '')[:500],
                source_type=source_type,
                bentham_factors=BenthamFactors(),
                quality_score=0.0,
                processing_time=time.time() - start_time
            )
    
    def _update_quality_stats(self, quality_score: float, processing_time: float):
        """품질 통계 업데이트"""
        self.quality_stats['total_processed'] += 1
        
        if quality_score >= 0.8:
            self.quality_stats['high_quality'] += 1
        elif quality_score >= 0.5:
            self.quality_stats['medium_quality'] += 1
        else:
            self.quality_stats['low_quality'] += 1
        
        # 평균 계산
        total = self.quality_stats['total_processed']
        self.quality_stats['average_quality'] = (
            (self.quality_stats['average_quality'] * (total - 1) + quality_score) / total
        )
        self.quality_stats['average_processing_time'] = (
            (self.quality_stats['average_processing_time'] * (total - 1) + processing_time) / total
        )
    
    def process_ebs_file(self, file_path: str, max_scenarios: int = 20) -> List[EnhancedScenario]:
        """EBS 파일 처리 - 제한된 개수로 고품질 처리"""
        self.logger.info(f"EBS 파일 처리 시작: {file_path} (최대 {max_scenarios}개)")
        
        # 파싱
        raw_scenarios = self.parser.parse_ebs_literature_optimized(file_path, max_scenarios)
        self.logger.info(f"파싱 완료: {len(raw_scenarios)}개 시나리오")
        
        processed_scenarios = []
        
        # 순차 처리 (LLM 안정성 위해)
        for idx, raw_scenario in enumerate(raw_scenarios):
            if idx >= max_scenarios:
                break
                
            try:
                scenario = self.process_single_scenario(raw_scenario, "ebs_literature", idx)
                processed_scenarios.append(scenario)
                
                # 진행상황 로깅
                self.logger.info(f"처리 완료: {idx + 1}/{len(raw_scenarios)} "
                                f"(품질: {scenario.quality_score:.2f}, "
                                f"시간: {scenario.processing_time:.1f}초)")
                
                # 청크 단위 저장
                if (idx + 1) % self.chunk_size == 0:
                    self._save_chunk(processed_scenarios[-self.chunk_size:], f"ebs_chunk_{idx//self.chunk_size}")
                    
            except Exception as e:
                self.logger.error(f"시나리오 {idx} 처리 중 오류: {e}")
                continue
        
        return processed_scenarios
    
    def _save_chunk(self, scenarios: List[EnhancedScenario], chunk_name: str):
        """청크 저장"""
        chunk_file = self.output_dir / f"{chunk_name}.json"
        
        scenarios_dict = []
        for scenario in scenarios:
            scenario_dict = {
                'id': scenario.id,
                'title': scenario.title,
                'description': scenario.description,
                'source_type': scenario.source_type,
                'bentham_factors': {
                    'intensity': scenario.bentham_factors.intensity,
                    'duration': scenario.bentham_factors.duration,
                    'certainty': scenario.bentham_factors.certainty,
                    'propinquity': scenario.bentham_factors.propinquity,
                    'fecundity': scenario.bentham_factors.fecundity,
                    'purity': scenario.bentham_factors.purity,
                    'extent': scenario.bentham_factors.extent
                },
                'stakeholders': scenario.stakeholders,
                'quality_score': scenario.quality_score,
                'processing_time': scenario.processing_time,
                'processing_timestamp': scenario.processing_timestamp
            }
            scenarios_dict.append(scenario_dict)
        
        with open(chunk_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(scenarios_dict, ensure_ascii=False, indent=2))
        
        self.logger.info(f"청크 저장: {chunk_file} ({len(scenarios)}개 시나리오)")
    
    def save_final_results(self, all_scenarios: List[EnhancedScenario]):
        """최종 결과 및 품질 통계 저장"""
        # 최종 시나리오 파일
        final_file = self.output_dir / "high_quality_scenarios.json"
        
        scenarios_dict = []
        for scenario in all_scenarios:
            scenario_dict = {
                'id': scenario.id,
                'title': scenario.title,
                'description': scenario.description,
                'source_type': scenario.source_type,
                'bentham_factors': {
                    'intensity': scenario.bentham_factors.intensity,
                    'duration': scenario.bentham_factors.duration,
                    'certainty': scenario.bentham_factors.certainty,
                    'propinquity': scenario.bentham_factors.propinquity,
                    'fecundity': scenario.bentham_factors.fecundity,
                    'purity': scenario.bentham_factors.purity,
                    'extent': scenario.bentham_factors.extent
                },
                'stakeholders': scenario.stakeholders,
                'quality_score': scenario.quality_score,
                'processing_time': scenario.processing_time,
                'processing_timestamp': scenario.processing_timestamp
            }
            scenarios_dict.append(scenario_dict)
        
        final_data = {
            'metadata': {
                'total_scenarios': len(all_scenarios),
                'quality_statistics': self.quality_stats,
                'processing_timestamp': datetime.now().isoformat()
            },
            'scenarios': scenarios_dict
        }
        
        with open(final_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(final_data, ensure_ascii=False, indent=2))
        
        # 품질 통계 파일
        stats_file = self.output_dir / "quality_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.quality_stats, ensure_ascii=False, indent=2))
        
        self.logger.info(f"최종 결과 저장: {final_file}")
        self.logger.info(f"품질 통계 저장: {stats_file}")
        
        # 품질 리포트 출력
        self._print_quality_report()
    
    def _print_quality_report(self):
        """품질 리포트 출력"""
        stats = self.quality_stats
        total = stats['total_processed']
        
        if total == 0:
            self.logger.warning("처리된 시나리오가 없습니다.")
            return
        
        print(f"\n{'='*60}")
        print(f"🏆 고품질 데이터 처리 완료 리포트")
        print(f"{'='*60}")
        print(f"📊 전체 처리: {total}개 시나리오")
        print(f"✨ 고품질 (≥0.8): {stats['high_quality']}개 ({stats['high_quality']/total*100:.1f}%)")
        print(f"🔶 중품질 (≥0.5): {stats['medium_quality']}개 ({stats['medium_quality']/total*100:.1f}%)")
        print(f"🔸 저품질 (<0.5): {stats['low_quality']}개 ({stats['low_quality']/total*100:.1f}%)")
        print(f"❌ 실패: {stats['failed']}개 ({stats['failed']/total*100:.1f}%)")
        print(f"📈 평균 품질 점수: {stats['average_quality']:.3f}")
        print(f"⏱️ 평균 처리 시간: {stats['average_processing_time']:.2f}초")
        print(f"{'='*60}\n")

def main():
    """메인 실행 함수"""
    processor = OptimizedDatasetProcessor()
    
    try:
        print("🚀 고품질 데이터 처리 시작...")
        processor.initialize()
        
        # EBS 파일 하나만 고품질로 처리 (테스트)
        scenarios = processor.process_ebs_file(
            'for_learn_dataset/ai_ebs/ebs_1.txt', 
            max_scenarios=10  # 테스트용 10개만
        )
        
        processor.save_final_results(scenarios)
        
        print("✅ 고품질 데이터 처리 완료!")
        
    except Exception as e:
        print(f"❌ 처리 중 오류: {e}")
        logger.error(f"처리 실패: {e}")
        raise

if __name__ == "__main__":
    main()