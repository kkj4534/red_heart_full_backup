"""
단일 9B 모델만 사용하는 간단한 LLM 처리기
Simple LLM Processor using only the 9B model
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

# llama-cpp-python 직접 사용
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("llama-cpp-python이 설치되지 않았습니다.")

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('SimpleLLMProcessor')

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
class ProcessedScenario:
    """처리된 시나리오"""
    id: str
    title: str
    description: str
    source_type: str
    bentham_factors: BenthamFactors
    category: str = "general"
    stakeholders: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    quality_score: float = 1.0

class Simple9BProcessor:
    """9B 모델만 사용하는 간단한 프로세서"""
    
    def __init__(self):
        self.llama_model = None
        self.logger = logging.getLogger(f'{__name__}.Simple9BProcessor')
        
        # 프롬프트 템플릿 최적화 (더 짧고 명확하게)
        self.bentham_prompt = """다음 상황의 벤담 7개 변수를 0.0~1.0으로 평가하세요.

상황: {text}

JSON 형식으로만 답하세요:
{{"intensity": 0.0, "duration": 0.0, "certainty": 0.0, "propinquity": 0.0, "fecundity": 0.0, "purity": 0.0, "extent": 0.0}}"""
    
    def initialize(self):
        """9B 모델만 로드"""
        if not LLAMA_CPP_AVAILABLE:
            raise RuntimeError("llama-cpp-python이 필요합니다: pip install llama-cpp-python")
        
        gguf_path = Path("llm_module/HelpingAI2-9B.Q4_K_M.gguf")
        if not gguf_path.exists():
            raise RuntimeError(f"모델 파일을 찾을 수 없습니다: {gguf_path}")
        
        try:
            self.logger.info("9B 모델 로딩 중...")
            self.llama_model = Llama(
                model_path=str(gguf_path),
                n_ctx=2048,  # 컨텍스트 길이 줄임
                n_batch=256,  # 배치 크기 줄임
                n_threads=4,  # 스레드 수 조정
                verbose=False
            )
            self.logger.info("9B 모델 로드 완료")
        except Exception as e:
            self.logger.error(f"9B 모델 로드 실패: {e}")
            raise
    
    def extract_bentham_factors(self, text: str) -> Tuple[BenthamFactors, float]:
        """9B 모델로 벤담 변수 추출"""
        if not self.llama_model:
            raise RuntimeError("9B 모델이 초기화되지 않았습니다.")
        
        # 텍스트 길이 제한 (속도 향상)
        if len(text) > 800:
            text = text[:800] + "..."
        
        prompt = self.bentham_prompt.format(text=text)
        
        try:
            start_time = time.time()
            
            # 9B 모델로 생성 (파라미터 최적화)
            response = self.llama_model(
                prompt,
                max_tokens=200,  # 토큰 수 제한
                temperature=0.3,  # 온도 낮춤 (일관성)
                top_p=0.9,
                stop=["}", "\n\n"],  # 조기 중단
                echo=False
            )
            
            processing_time = time.time() - start_time
            response_text = response['choices'][0]['text']
            
            # JSON 파싱
            factors, quality_score = self._parse_bentham_response(response_text)
            
            self.logger.debug(f"벤담 추출 완료 ({processing_time:.2f}초, 품질: {quality_score:.2f})")
            
            return factors, quality_score
            
        except Exception as e:
            self.logger.error(f"벤담 추출 실패: {e}")
            return BenthamFactors(), 0.0
    
    def _parse_bentham_response(self, response_text: str) -> Tuple[BenthamFactors, float]:
        """벤담 응답 파싱"""
        quality_score = 0.0
        
        # JSON 추출
        json_match = re.search(r'\{[^}]*\}', response_text)
        if not json_match:
            return BenthamFactors(), 0.1
        
        try:
            factors_dict = json.loads(json_match.group())
            quality_score = 0.5
            
            # 필드 검증
            required_fields = ['intensity', 'duration', 'certainty', 'propinquity', 'fecundity', 'purity', 'extent']
            valid_count = 0
            
            for field in required_fields:
                if field in factors_dict:
                    try:
                        value = float(factors_dict[field])
                        if 0.0 <= value <= 1.0:
                            valid_count += 1
                    except (ValueError, TypeError):
                        pass
            
            quality_score = 0.3 + (valid_count / len(required_fields)) * 0.7
            
            # BenthamFactors 생성
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
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON 파싱 실패: {e}")
            return BenthamFactors(), 0.2

class SimpleDataParser:
    """간단한 데이터 파서"""
    
    def __init__(self):
        self.logger = logging.getLogger(f'{__name__}.SimpleDataParser')
    
    def parse_ebs_file(self, file_path: str, max_scenarios: int = 10) -> List[Dict[str, Any]]:
        """EBS 파일 파싱"""
        self.logger.info(f"EBS 파싱: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        scenarios = []
        current_scenario = {}
        
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 새 작품 시작
            if re.match(r'^[가-힣].+\([^)]+\)$', line):
                if current_scenario and current_scenario.get('description'):
                    scenarios.append(current_scenario)
                    if len(scenarios) >= max_scenarios:
                        break
                
                current_scenario = {
                    'title': line,
                    'description': '',
                    'stakeholders': []
                }
            
            elif line.startswith('상황설명:'):
                current_scenario['description'] = line.replace('상황설명:', '').strip()
            
            elif line.startswith('이해관계자:'):
                stakeholders_text = line.replace('이해관계자:', '').strip()
                current_scenario['stakeholders'] = [
                    s.strip().strip("'\"") for s in stakeholders_text.split(',')
                ]
        
        if current_scenario and current_scenario.get('description'):
            scenarios.append(current_scenario)
        
        self.logger.info(f"파싱 완료: {len(scenarios)}개 시나리오")
        return scenarios

class Simple9BDataProcessor:
    """9B 모델만 사용하는 데이터 처리기"""
    
    def __init__(self, output_dir: str = "simple_9b_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.parser = SimpleDataParser()
        self.llm_processor = Simple9BProcessor()
        self.logger = logging.getLogger(f'{__name__}.Simple9BDataProcessor')
        
        # 통계
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'average_time': 0.0,
            'average_quality': 0.0
        }
    
    def initialize(self):
        """초기화"""
        self.llm_processor.initialize()
        self.logger.info("9B 데이터 처리기 초기화 완료")
    
    def process_ebs_file(self, file_path: str, max_scenarios: int = 10) -> List[ProcessedScenario]:
        """EBS 파일 처리"""
        self.logger.info(f"9B 모델로 처리 시작: {file_path} (최대 {max_scenarios}개)")
        
        # 파싱
        raw_scenarios = self.parser.parse_ebs_file(file_path, max_scenarios)
        processed_scenarios = []
        
        for idx, raw_scenario in enumerate(raw_scenarios):
            try:
                start_time = time.time()
                
                # 9B 모델로 벤담 변수 추출
                bentham_factors, quality_score = self.llm_processor.extract_bentham_factors(
                    raw_scenario['description']
                )
                
                processing_time = time.time() - start_time
                
                scenario = ProcessedScenario(
                    id=f"ebs_9b_{idx}_{uuid.uuid4().hex[:8]}",
                    title=raw_scenario['title'],
                    description=raw_scenario['description'],
                    source_type="ebs_literature_9b",
                    bentham_factors=bentham_factors,
                    stakeholders=raw_scenario.get('stakeholders', []),
                    processing_time=processing_time,
                    quality_score=quality_score
                )
                
                processed_scenarios.append(scenario)
                
                # 통계 업데이트
                self.stats['total_processed'] += 1
                if quality_score > 0.5:
                    self.stats['successful'] += 1
                else:
                    self.stats['failed'] += 1
                
                # 평균 계산
                total = self.stats['total_processed']
                self.stats['average_time'] = (
                    (self.stats['average_time'] * (total - 1) + processing_time) / total
                )
                self.stats['average_quality'] = (
                    (self.stats['average_quality'] * (total - 1) + quality_score) / total
                )
                
                self.logger.info(f"처리: {idx + 1}/{len(raw_scenarios)} "
                                f"(시간: {processing_time:.1f}초, 품질: {quality_score:.2f})")
                
            except Exception as e:
                self.logger.error(f"시나리오 {idx} 처리 실패: {e}")
                self.stats['failed'] += 1
                continue
        
        return processed_scenarios
    
    def save_results(self, scenarios: List[ProcessedScenario]):
        """결과 저장"""
        output_file = self.output_dir / "9b_processed_scenarios.json"
        
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
                'processing_time': scenario.processing_time,
                'quality_score': scenario.quality_score
            }
            scenarios_dict.append(scenario_dict)
        
        final_data = {
            'metadata': {
                'total_scenarios': len(scenarios),
                'processing_statistics': self.stats,
                'model_used': '9B_only',
                'timestamp': datetime.now().isoformat()
            },
            'scenarios': scenarios_dict
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(final_data, ensure_ascii=False, indent=2))
        
        self.logger.info(f"결과 저장: {output_file}")
        self._print_report(len(scenarios))
    
    def _print_report(self, total_scenarios: int):
        """결과 리포트"""
        stats = self.stats
        print(f"\n{'='*60}")
        print(f"🚀 9B 모델 전용 처리 완료 리포트")
        print(f"{'='*60}")
        print(f"📊 처리된 시나리오: {total_scenarios}개")
        print(f"✅ 성공: {stats['successful']}개 ({stats['successful']/total_scenarios*100:.1f}%)")
        print(f"❌ 실패: {stats['failed']}개 ({stats['failed']/total_scenarios*100:.1f}%)")
        print(f"⏱️ 평균 처리 시간: {stats['average_time']:.2f}초")
        print(f"🎯 평균 품질 점수: {stats['average_quality']:.3f}")
        print(f"🧠 사용 모델: HelpingAI2-9B (단일)")
        print(f"{'='*60}\n")

def main():
    """메인 실행 - 9B 모델만 사용"""
    print("🧠 9B 모델 전용 데이터 처리 시작...")
    
    processor = Simple9BDataProcessor()
    
    try:
        processor.initialize()
        
        # EBS 파일 처리 (5개만 테스트)
        scenarios = processor.process_ebs_file(
            'for_learn_dataset/ai_ebs/ebs_1.txt',
            max_scenarios=5
        )
        
        processor.save_results(scenarios)
        
        print("✅ 9B 모델 처리 완료!")
        
    except Exception as e:
        print(f"❌ 처리 중 오류: {e}")
        logger.error(f"처리 실패: {e}")

if __name__ == "__main__":
    main()