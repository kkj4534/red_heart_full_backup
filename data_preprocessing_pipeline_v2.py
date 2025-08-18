#!/usr/bin/env python3
"""
Red Heart AI 데이터 전처리 파이프라인 v2
- sentence-transformers를 통한 실제 임베딩 생성
- LLM 기반 완전한 메트릭 생성 (더미 데이터 제거)
- parsed_raw_datasets.json 활용
- 프로젝트 규칙 준수: fallback/simplification 없음
"""

import os
import json
import torch
import numpy as np
import logging
import gc
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer

# 프로젝트 모듈
from llm_module.advanced_llm_engine import (
    LLMRequest, LLMResponse, 
    TaskComplexity,
    AdvancedLLMEngine
)
from config import ADVANCED_CONFIG, get_device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('RedHeart.DataPreprocessingV2')

@dataclass
class EnrichedDataPoint:
    """전처리된 데이터 포인트"""
    text: str
    original_label: str
    
    # LLM이 생성한 enriched 데이터
    emotion_vector: List[float]      # 7차원 감정 벡터
    regret_factor: float            # 후회 지수
    bentham_scores: Dict[str, float]  # 벤담 쾌락 계산
    surd_metrics: Dict[str, float]    # SURD 분석
    context_embedding: List[float]     # 768차원 문맥 임베딩
    
    # 메타데이터
    processing_time: float
    llm_confidence: float
    timestamp: str
    source: str  # 데이터 출처

class ImprovedDataPreprocessor:
    """개선된 데이터 전처리 파이프라인 - 프로젝트 규칙 완전 준수"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        초기화
        Args:
            model_name: sentence-transformers 모델명
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"디바이스: {self.device}")
        
        # Sentence Transformer 로드
        logger.info(f"Loading sentence transformer: {model_name}")
        self.sentence_model = SentenceTransformer(model_name, device=self.device)
        
        # LLM 엔진 초기화
        self.llm_engine = None
        self.is_llm_initialized = False
        
        logger.info("✅ 개선된 전처리 파이프라인 초기화 완료")
    
    def initialize_llm(self):
        """LLM 초기화"""
        if self.is_llm_initialized:
            logger.info("LLM이 이미 초기화되어 있음")
            return
        
        logger.info("🔄 LLM 엔진 초기화 중...")
        
        self.llm_engine = AdvancedLLMEngine()
        self.is_llm_initialized = True
        
        logger.info("✅ LLM 초기화 완료")
    
    def generate_context_embedding(self, text: str) -> List[float]:
        """
        Sentence-Transformers를 사용한 실제 임베딩 생성
        
        Args:
            text: 입력 텍스트
            
        Returns:
            768차원 임베딩 벡터
        """
        with torch.no_grad():
            embedding = self.sentence_model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    def analyze_emotions_with_llm(self, text: str) -> Dict[str, Any]:
        """
        LLM을 통한 감정 분석
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            감정 분석 결과
        """
        prompt = f"""다음 텍스트의 감정을 분석하세요. 각 감정을 0-1 사이의 값으로 평가하세요.
모든 감정의 합이 1이 되도록 정규화하세요.

텍스트: {text[:500]}

다음 형식으로 답변하세요:
{{
    "joy": 0.x,
    "sadness": 0.x,
    "anger": 0.x,
    "fear": 0.x,
    "surprise": 0.x,
    "disgust": 0.x,
    "trust": 0.x
}}"""
        
        request = LLMRequest(
            prompt=prompt,
            task_type="emotion_analysis",
            complexity=TaskComplexity.MODERATE
        )
        response = self.llm_engine.generate_sync(request)
        
        # JSON 파싱
        import re
        json_match = re.search(r'\{[^}]+\}', response.generated_text)
        if not json_match:
            raise ValueError("LLM이 유효한 JSON 형식으로 응답하지 않음")
        
        emotion_dict = json.loads(json_match.group())
        emotion_vector = [
            emotion_dict.get('joy', 0.0),
            emotion_dict.get('sadness', 0.0),
            emotion_dict.get('anger', 0.0),
            emotion_dict.get('fear', 0.0),
            emotion_dict.get('surprise', 0.0),
            emotion_dict.get('disgust', 0.0),
            emotion_dict.get('trust', 0.0)
        ]
        
        # 정규화
        total = sum(emotion_vector)
        if total > 0:
            emotion_vector = [e/total for e in emotion_vector]
        else:
            raise ValueError("감정 벡터 합이 0")
        
        return {"vector": emotion_vector, "confidence": response.confidence}
    
    def calculate_bentham_scores(self, text: str, emotion_vector: List[float]) -> Dict[str, float]:
        """
        LLM을 통한 벤담 쾌락 계산
        
        Args:
            text: 분석할 텍스트
            emotion_vector: 감정 벡터
            
        Returns:
            벤담 점수
        """
        prompt = f"""다음 텍스트를 벤담의 쾌락 계산법으로 분석하세요.

텍스트: {text[:500]}
현재 감정 상태: {emotion_vector}

다음 4가지 차원을 0-1 사이의 값으로 평가하세요:
1. intensity (강도): 행복/쾌락의 강도
2. duration (지속성): 효과의 지속 시간
3. certainty (확실성): 발생 가능성
4. propinquity (근접성): 시간적/공간적 근접성

형식:
{{
    "intensity": 0.x,
    "duration": 0.x,
    "certainty": 0.x,
    "propinquity": 0.x
}}"""
        
        request = LLMRequest(
            prompt=prompt,
            task_type="bentham_analysis",
            complexity=TaskComplexity.COMPLEX
        )
        response = self.llm_engine.generate_sync(request)
        
        # JSON 파싱
        import re
        json_match = re.search(r'\{[^}]+\}', response.generated_text)
        if not json_match:
            raise ValueError("LLM이 유효한 벤담 점수를 반환하지 않음")
        
        scores = json.loads(json_match.group())
        return {
            'intensity': float(scores['intensity']),
            'duration': float(scores['duration']),
            'certainty': float(scores['certainty']),
            'propinquity': float(scores['propinquity'])
        }
    
    def calculate_regret_factor(self, text: str) -> float:
        """
        LLM을 통한 후회 지수 계산
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            후회 지수 (0-1)
        """
        prompt = f"""다음 텍스트에서 후회의 정도를 0에서 1 사이의 값으로 평가하세요.
0은 후회 없음, 1은 극도의 후회를 의미합니다.

텍스트: {text[:500]}

숫자만 답하세요 (예: 0.7)"""
        
        request = LLMRequest(
            prompt=prompt,
            task_type="regret_analysis",
            complexity=TaskComplexity.SIMPLE
        )
        response = self.llm_engine.generate_sync(request)
        
        # 숫자 추출
        import re
        numbers = re.findall(r'0\.\d+|1\.0|0|1', response.generated_text)
        if not numbers:
            raise ValueError("LLM이 유효한 후회 지수를 반환하지 않음")
        
        return float(numbers[0])
    
    def calculate_surd_metrics(self, text: str, metadata: Dict[str, Any]) -> Dict[str, float]:
        """
        LLM을 통한 SURD (충분성, 이해가능성, 복원력, 결정성) 메트릭 계산
        
        Args:
            text: 분석할 텍스트
            metadata: 추가 컨텍스트 정보
            
        Returns:
            SURD 메트릭
        """
        # 메타데이터에서 추가 정보 추출
        ethical_dilemma = metadata.get('ethical_dilemma', '')
        stakeholders = metadata.get('stakeholders', '')
        
        prompt = f"""다음 텍스트를 SURD 프레임워크로 분석하세요.

텍스트: {text[:500]}
{f"윤리적 딜레마: {ethical_dilemma[:200]}" if ethical_dilemma else ""}
{f"이해관계자: {stakeholders[:100]}" if stakeholders else ""}

다음 4가지 차원을 0-1 사이의 값으로 평가하세요:

1. Sufficiency (충분성): 윤리적 판단을 위한 정보의 충분성
   - 상황이 얼마나 완전하게 설명되었는가?
   - 모든 관련 측면이 고려되었는가?

2. Understandability (이해가능성): 상황의 명확성과 이해 용이성
   - 딜레마가 얼마나 명확하게 표현되었는가?
   - 이해관계자들의 입장이 분명한가?

3. Resilience (복원력): 결정의 견고성과 적응력
   - 결정이 변화하는 상황에서도 유효한가?
   - 예상치 못한 결과에 대응할 수 있는가?

4. Decisiveness (결정성): 명확한 선택과 행동 가능성
   - 구체적인 행동 방향이 제시되는가?
   - 결정이 실행 가능한가?

형식:
{{
    "sufficiency": 0.x,
    "understandability": 0.x,
    "resilience": 0.x,
    "decisiveness": 0.x
}}"""
        
        request = LLMRequest(
            prompt=prompt,
            task_type="surd_analysis",
            complexity=TaskComplexity.COMPLEX
        )
        response = self.llm_engine.generate_sync(request)
        
        # JSON 파싱
        import re
        json_match = re.search(r'\{[^}]+\}', response.generated_text)
        if not json_match:
            raise ValueError("LLM이 유효한 SURD 메트릭을 반환하지 않음")
        
        metrics = json.loads(json_match.group())
        return {
            'sufficiency': float(metrics['sufficiency']),
            'understandability': float(metrics['understandability']),
            'resilience': float(metrics['resilience']),
            'decisiveness': float(metrics['decisiveness'])
        }
    
    def process_single_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        단일 데이터 아이템 처리 - 예외 처리 없음 (프로젝트 규칙)
        
        Args:
            item: 원본 데이터 아이템
            
        Returns:
            전처리된 데이터
        """
        text = item.get('text', '')
        if not text:
            raise ValueError("텍스트가 비어있음")
        
        start_time = time.time()
        
        # 1. Sentence-Transformer로 실제 임베딩 생성
        context_embedding = self.generate_context_embedding(text)
        
        # 2. LLM으로 감정 분석
        emotion_result = self.analyze_emotions_with_llm(text)
        emotion_vector = emotion_result['vector']
        
        # 3. 후회 지수 계산
        regret_factor = self.calculate_regret_factor(text)
        
        # 4. 벤담 점수 계산 (LLM)
        bentham_scores = self.calculate_bentham_scores(text, emotion_vector)
        
        # 5. SURD 메트릭 계산 (LLM, 메타데이터 활용)
        surd_metrics = self.calculate_surd_metrics(text, item.get('metadata', {}))
        
        processing_time = time.time() - start_time
        
        return {
            'text': text,
            'source': item.get('source', 'unknown'),
            'emotion_vector': emotion_vector,
            'regret_factor': regret_factor,
            'bentham_scores': bentham_scores,
            'surd_metrics': surd_metrics,
            'context_embedding': context_embedding,
            'processing_time': processing_time,
            'llm_confidence': emotion_result.get('confidence', 0.5),
            'timestamp': datetime.now().isoformat(),
            'metadata': item.get('metadata', {})
        }
    
    def process_dataset(self, input_file: str = "parsed_raw_datasets.json",
                        output_file: str = "preprocessed_dataset_v3.json",
                        limit: Optional[int] = None):
        """
        전체 데이터셋 처리
        
        Args:
            input_file: 입력 파일 경로
            output_file: 출력 파일 경로
            limit: 처리할 최대 샘플 수
        """
        logger.info(f"📚 데이터셋 처리 시작: {input_file}")
        
        # 데이터 로드
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        if limit:
            raw_data = raw_data[:limit]
        
        logger.info(f"총 {len(raw_data)}개 샘플 처리 예정")
        
        # LLM 초기화
        self.initialize_llm()
        
        # 처리
        processed_data = []
        failed_count = 0
        
        for i, item in enumerate(raw_data):
            if i % 10 == 0:
                logger.info(f"진행률: {i}/{len(raw_data)} ({i*100/len(raw_data):.1f}%)")
            
            try:
                result = self.process_single_item(item)
                processed_data.append(result)
            except Exception as e:
                logger.error(f"샘플 {i} 처리 실패: {e}")
                failed_count += 1
                # 프로젝트 규칙: fallback 없음, 실패는 실패
                raise
            
            # 메모리 관리
            if i % 50 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ 처리 완료: {len(processed_data)}개 샘플 저장 → {output_file}")
        
        # 통계 출력
        self.print_statistics(processed_data)
    
    def print_statistics(self, data: List[Dict]):
        """데이터 통계 출력"""
        if not data:
            return
        
        print("\n📊 데이터셋 통계:")
        print(f"  - 총 샘플 수: {len(data)}")
        
        # 소스별 분포
        sources = {}
        for item in data:
            src = item.get('source', 'unknown')
            sources[src] = sources.get(src, 0) + 1
        
        print("\n  소스별 분포:")
        for src, count in sources.items():
            print(f"    - {src}: {count}개")
        
        # 평균 지표
        avg_regret = sum(item['regret_factor'] for item in data) / len(data)
        print(f"\n  평균 후회 지수: {avg_regret:.3f}")
        
        # 임베딩 차원
        if data[0].get('context_embedding'):
            print(f"  임베딩 차원: {len(data[0]['context_embedding'])}")
        
        # SURD 메트릭 평균
        avg_surd = {
            'sufficiency': sum(item['surd_metrics']['sufficiency'] for item in data) / len(data),
            'understandability': sum(item['surd_metrics']['understandability'] for item in data) / len(data),
            'resilience': sum(item['surd_metrics']['resilience'] for item in data) / len(data),
            'decisiveness': sum(item['surd_metrics']['decisiveness'] for item in data) / len(data)
        }
        print("\n  평균 SURD 메트릭:")
        for key, value in avg_surd.items():
            print(f"    - {key}: {value:.3f}")

def main():
    """메인 실행 함수"""
    preprocessor = ImprovedDataPreprocessor(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # 테스트 모드 (처음 10개만)
    preprocessor.process_dataset(
        input_file="parsed_raw_datasets.json",
        output_file="preprocessed_dataset_v3.json",
        limit=10  # 테스트용
    )

if __name__ == "__main__":
    main()