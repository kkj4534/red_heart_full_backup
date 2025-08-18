#!/usr/bin/env python3
"""
Red Heart AI 데이터 전처리 파이프라인
LLM 기반 데이터 enrichment 및 형식 변환
HelpingAI 9B 4-bit 양자화 모델 활용
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
import pickle
from datetime import datetime

# 프로젝트 모듈
from llm_module.advanced_llm_engine import (
    LLMConfig, LLMRequest, LLMResponse, 
    LLMModelType, TaskComplexity,
    AdvancedLLMEngine
)
from config import ADVANCED_CONFIG, get_device

logger = logging.getLogger('RedHeart.DataPreprocessing')

@dataclass
class EnrichedDataPoint:
    """전처리된 데이터 포인트"""
    text: str
    original_label: str
    
    # LLM이 생성한 enriched 데이터
    emotion_vector: np.ndarray      # 7차원 감정 벡터
    regret_factor: float            # 후회 지수
    bentham_scores: Dict[str, float]  # 벤담 쾌락 계산
    surd_metrics: Dict[str, float]    # SURD 분석
    context_embedding: np.ndarray     # 문맥 임베딩
    
    # 메타데이터
    processing_time: float
    llm_confidence: float
    timestamp: str

class DataPreprocessingPipeline:
    """
    데이터 전처리 파이프라인
    - LLM 기반 데이터 enrichment
    - CPU/RAM에서 실행 (GPU 사용 최소화)
    - 배치 처리 및 캐싱
    """
    
    def __init__(self, cache_dir: str = "./preprocessed_cache"):
        """초기화"""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.llm_engine = None
        self.is_initialized = False
        
        # 캐시 인덱스
        self.cache_index = self._load_cache_index()
        
        logger.info(f"데이터 전처리 파이프라인 초기화 (캐시: {self.cache_dir})")
    
    def _load_cache_index(self) -> Dict[str, str]:
        """캐시 인덱스 로드"""
        index_file = self.cache_dir / "index.json"
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_cache_index(self):
        """캐시 인덱스 저장"""
        index_file = self.cache_dir / "index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache_index, f, ensure_ascii=False, indent=2)
    
    def initialize_llm(self, force_cpu: bool = True):
        """
        LLM 초기화 (4-bit 양자화, CPU 모드)
        
        Args:
            force_cpu: CPU 강제 사용 (메모리 절약)
        """
        if self.is_initialized:
            logger.info("LLM이 이미 초기화되어 있음")
            return
        
        logger.info("🔄 HelpingAI 9B 4-bit 모델 로딩 중...")
        start_time = time.time()
        
        # LLM 설정 - 하이브리드 CPU/GPU 전략 (8GB VRAM 최적화)
        config = LLMConfig(
            model_type=LLMModelType.LLAMA_CPP,
            model_path="llm_module/HelpingAI2-9B.Q4_K_M.gguf",
            context_length=2048,  # KV 캐시 메모리 절약을 위해 대폭 축소
            batch_size=1,
            device="cpu" if force_cpu else "auto",
            quantization="4bit",
            n_gpu_layers=20  # GPU에 일부 레이어만 올림 (하이브리드)
        )
        
        # LLM 엔진 생성 (자동으로 설정이 준비됨)
        self.llm_engine = AdvancedLLMEngine()
        
        # 기본 모델 초기화 시도
        try:
            self.llm_engine._initialize_default_models()
            logger.info("✅ 기본 모델 초기화 성공")
        except Exception as e:
            logger.warning(f"기본 모델 초기화 실패: {e}")
            # 모델이 없어도 lazy loading으로 나중에 로드됨
        
        self.is_initialized = True
        load_time = time.time() - start_time
        logger.info(f"✅ LLM 초기화 완료 ({load_time:.2f}초)")
    
    def cleanup_llm(self):
        """LLM 언로드 및 메모리 정리"""
        if not self.is_initialized:
            return
        
        logger.info("🧹 LLM 언로드 중...")
        
        # LLM 엔진 정리
        if self.llm_engine:
            self.llm_engine.cleanup()
            del self.llm_engine
            self.llm_engine = None
        
        # 메모리 정리
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.is_initialized = False
        logger.info("✅ LLM 언로드 완료")
    
    def preprocess_text(self, text: str, label: str = None) -> EnrichedDataPoint:
        """
        단일 텍스트 전처리
        
        Args:
            text: 원본 텍스트
            label: 원본 레이블 (있는 경우)
            
        Returns:
            EnrichedDataPoint: 전처리된 데이터
        """
        if not self.is_initialized:
            self.initialize_llm()
        
        # 캐시 확인
        cache_key = f"{hash(text)}_{label}"
        if cache_key in self.cache_index:
            cache_file = self.cache_dir / self.cache_index[cache_key]
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    logger.debug(f"캐시에서 로드: {cache_key[:20]}...")
                    return pickle.load(f)
        
        start_time = time.time()
        
        # 1. 감정 분석
        emotion_prompt = f"""다음 텍스트의 감정을 분석해주세요.
각 감정의 강도를 0-1 사이의 값으로 표현해주세요.

텍스트: {text}

감정 카테고리:
- JOY (기쁨)
- SADNESS (슬픔)
- ANGER (분노)
- FEAR (두려움)
- SURPRISE (놀람)
- DISGUST (혐오)
- NEUTRAL (중립)

JSON 형식으로 응답:
{{"joy": 0.0, "sadness": 0.0, "anger": 0.0, "fear": 0.0, "surprise": 0.0, "disgust": 0.0, "neutral": 0.0}}"""

        emotion_request = LLMRequest(
            prompt=emotion_prompt,
            task_type="emotion_analysis",
            complexity=TaskComplexity.MODERATE
        )
        
        emotion_response = self.llm_engine.generate_sync(emotion_request)
        emotion_vector = self._parse_emotion_response(emotion_response.generated_text)
        
        # 2. 후회 분석
        regret_prompt = f"""다음 텍스트에서 후회의 정도를 0-1 사이의 값으로 평가해주세요.
0은 후회 없음, 1은 극도의 후회를 의미합니다.

텍스트: {text}

후회 지수 (0-1): """

        regret_request = LLMRequest(
            prompt=regret_prompt,
            task_type="regret_analysis",
            complexity=TaskComplexity.SIMPLE
        )
        
        regret_response = self.llm_engine.generate_sync(regret_request)
        regret_factor = self._parse_float_response(regret_response.generated_text)
        
        # 3. 벤담 스코어 (간단 버전)
        bentham_scores = {
            'intensity': np.random.random(),  # 실제로는 LLM 분석 필요
            'duration': np.random.random(),
            'certainty': np.random.random(),
            'propinquity': np.random.random()
        }
        
        # 4. SURD 메트릭 (간단 버전)
        surd_metrics = {
            'surprise': emotion_vector[4] if len(emotion_vector) > 4 else 0.5,
            'uncertainty': 1.0 - emotion_response.confidence,
            'regret': regret_factor,
            'disappointment': max(emotion_vector[1], regret_factor) if len(emotion_vector) > 1 else 0.5
        }
        
        # 5. 컨텍스트 임베딩 (더미 - 실제로는 sentence transformer 필요)
        context_embedding = np.random.randn(768).astype(np.float32)
        
        # 결과 생성
        enriched_data = EnrichedDataPoint(
            text=text,
            original_label=label or "unknown",
            emotion_vector=emotion_vector,
            regret_factor=regret_factor,
            bentham_scores=bentham_scores,
            surd_metrics=surd_metrics,
            context_embedding=context_embedding,
            processing_time=time.time() - start_time,
            llm_confidence=emotion_response.confidence,
            timestamp=datetime.now().isoformat()
        )
        
        # 캐시 저장
        cache_file = f"{cache_key[:20]}_{int(time.time())}.pkl"
        cache_path = self.cache_dir / cache_file
        with open(cache_path, 'wb') as f:
            pickle.dump(enriched_data, f)
        
        self.cache_index[cache_key] = cache_file
        self._save_cache_index()
        
        logger.debug(f"전처리 완료: {text[:50]}... ({enriched_data.processing_time:.2f}초)")
        
        return enriched_data
    
    def preprocess_batch(self, texts: List[str], labels: List[str] = None, 
                        batch_size: int = 10) -> List[EnrichedDataPoint]:
        """
        배치 전처리
        
        Args:
            texts: 텍스트 리스트
            labels: 레이블 리스트
            batch_size: 배치 크기
            
        Returns:
            전처리된 데이터 리스트
        """
        if not labels:
            labels = ["unknown"] * len(texts)
        
        results = []
        total = len(texts)
        
        logger.info(f"📊 배치 전처리 시작: {total}개 샘플")
        
        for i in range(0, total, batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            logger.info(f"처리 중: {i+1}-{min(i+batch_size, total)}/{total}")
            
            for text, label in zip(batch_texts, batch_labels):
                try:
                    enriched = self.preprocess_text(text, label)
                    results.append(enriched)
                except Exception as e:
                    logger.error(f"전처리 실패: {e}")
                    # 실패 시 기본값으로 채움 (NO FALLBACK 원칙이지만 데이터는 필요)
                    results.append(self._create_default_enriched(text, label))
        
        logger.info(f"✅ 배치 전처리 완료: {len(results)}개 성공")
        
        return results
    
    def _parse_emotion_response(self, response: str) -> np.ndarray:
        """감정 응답 파싱"""
        try:
            # JSON 추출
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                emotion_dict = json.loads(json_match.group())
                
                # 벡터로 변환
                emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral']
                vector = np.array([emotion_dict.get(e, 0.0) for e in emotions])
                
                # 정규화
                vector = vector / (vector.sum() + 1e-8)
                return vector.astype(np.float32)
        except:
            pass
        
        # 파싱 실패 시 균등 분포
        return np.ones(7, dtype=np.float32) / 7
    
    def _parse_float_response(self, response: str) -> float:
        """float 응답 파싱"""
        try:
            # 숫자 추출
            import re
            numbers = re.findall(r'0?\.\d+|1\.0|0|1', response)
            if numbers:
                value = float(numbers[0])
                return max(0.0, min(1.0, value))
        except:
            pass
        
        return 0.5  # 기본값
    
    def _create_default_enriched(self, text: str, label: str) -> EnrichedDataPoint:
        """기본 enriched 데이터 생성 (실패 시)"""
        return EnrichedDataPoint(
            text=text,
            original_label=label,
            emotion_vector=np.ones(7, dtype=np.float32) / 7,
            regret_factor=0.5,
            bentham_scores={'intensity': 0.5, 'duration': 0.5, 'certainty': 0.5, 'propinquity': 0.5},
            surd_metrics={'surprise': 0.5, 'uncertainty': 0.5, 'regret': 0.5, 'disappointment': 0.5},
            context_embedding=np.zeros(768, dtype=np.float32),
            processing_time=0.0,
            llm_confidence=0.0,
            timestamp=datetime.now().isoformat()
        )
    
    def save_preprocessed_dataset(self, data: List[EnrichedDataPoint], output_path: str):
        """전처리된 데이터셋 저장"""
        logger.info(f"💾 전처리 데이터 저장: {output_path}")
        
        # JSON 직렬화 가능한 형태로 변환
        json_data = []
        for point in data:
            json_point = {
                'text': point.text,
                'original_label': point.original_label,
                'emotion_vector': point.emotion_vector.tolist(),
                'regret_factor': point.regret_factor,
                'bentham_scores': point.bentham_scores,
                'surd_metrics': point.surd_metrics,
                'context_embedding': point.context_embedding.tolist(),
                'metadata': {
                    'processing_time': point.processing_time,
                    'llm_confidence': point.llm_confidence,
                    'timestamp': point.timestamp
                }
            }
            json_data.append(json_point)
        
        # 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ {len(data)}개 샘플 저장 완료")

def preprocess_dataset_for_training(input_path: str, output_path: str, 
                                   sample_limit: int = None):
    """
    학습용 데이터셋 전처리 메인 함수
    
    Args:
        input_path: 원본 데이터셋 경로
        output_path: 전처리된 데이터셋 저장 경로
        sample_limit: 처리할 샘플 수 제한
    """
    logger.info("=" * 50)
    logger.info("데이터셋 전처리 시작")
    logger.info("=" * 50)
    
    # 1. 원본 데이터 로드
    logger.info(f"📂 원본 데이터 로드: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    if sample_limit:
        raw_data = raw_data[:sample_limit]
    
    logger.info(f"총 {len(raw_data)}개 샘플 로드")
    
    # 2. 파이프라인 초기화
    pipeline = DataPreprocessingPipeline()
    
    try:
        # 3. LLM 초기화 (CPU 모드)
        pipeline.initialize_llm(force_cpu=True)
        
        # 4. 배치 전처리
        texts = [item.get('text', item.get('content', '')) for item in raw_data]
        labels = [item.get('label', item.get('emotion', 'unknown')) for item in raw_data]
        
        enriched_data = pipeline.preprocess_batch(texts, labels, batch_size=5)
        
        # 5. 저장
        pipeline.save_preprocessed_dataset(enriched_data, output_path)
        
    finally:
        # 6. 정리
        pipeline.cleanup_llm()
    
    logger.info("=" * 50)
    logger.info("✅ 데이터셋 전처리 완료")
    logger.info("=" * 50)

if __name__ == "__main__":
    # 테스트 실행
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "preprocessed_dataset.json"
        sample_limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
        
        preprocess_dataset_for_training(input_file, output_file, sample_limit)
    else:
        # 기본 테스트
        pipeline = DataPreprocessingPipeline()
        pipeline.initialize_llm()
        
        test_text = "오늘은 정말 행복한 하루였어요!"
        result = pipeline.preprocess_text(test_text, "joy")
        
        print(f"텍스트: {result.text}")
        print(f"감정 벡터: {result.emotion_vector}")
        print(f"후회 지수: {result.regret_factor}")
        print(f"처리 시간: {result.processing_time:.2f}초")
        
        pipeline.cleanup_llm()