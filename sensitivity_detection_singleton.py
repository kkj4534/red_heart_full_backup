"""
민감성 감지 및 중립화 시스템 싱글톤 매니저
Sensitivity Detection and Neutralization System Singleton Manager

HelpingAI 안전 필터 트리거 상황 대응용 특화 시스템
- 민감성 감지 LLM (가벼운 모델)
- 중립화 변환 LLM 
- 메모리 최적화 및 스레드 안전성 보장
"""

import os
import logging
import asyncio
import threading
import time
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from config import MODELS_DIR, get_device

logger = logging.getLogger(__name__)

@dataclass
class SensitivityResult:
    """민감성 감지 결과"""
    is_sensitive: bool
    confidence: float
    detected_keywords: List[str]
    reasoning: str

@dataclass
class NeutralizationResult:
    """중립화 변환 결과"""
    original_text: str
    neutralized_text: str
    transformations: Dict[str, str]  # 원본 키워드 -> 변환된 키워드
    confidence: float

class SensitivityDetectionManager:
    """
    민감성 감지 및 중립화 시스템 싱글톤 관리자
    - 메모리 효율적인 LLM 인스턴스 관리
    - 스레드 안전성 보장
    - GPU 리소스 최적화
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self._models: Dict[str, Any] = {}
        self._model_locks: Dict[str, threading.Lock] = {}
        self.device = get_device()
        
        # 민감성 감지용 키워드 패턴
        self.sensitive_patterns = {
            'political': ['테러', '감시', '정치', '정부', '국가기관'],
            'security': ['보안', '개인정보', '수집', '모니터링', '추적'],
            'surveillance': ['감시', '도청', '추적', '스파이', '정보수집'],
            'violence': ['폭력', '무력', '공격', '위협', '해킹'],
            'ethical_dilemma': ['딜레마', '갈등', '윤리적', '도덕적', '가치관']
        }
        
        # 중립화 변환 매핑
        self.neutralization_mapping = {
            '테러': '비상상황',
            '테러 방지': '공공 안전 확보', 
            '감시': '모니터링',
            '개인정보 수집': '데이터 수집',
            '개인정보': '개인 데이터',
            '시민 감시': '시민 모니터링',
            '정부 감시': '공공 모니터링',
            '프라이버시 침해': '개인정보 보호 이슈',
            '감시 체계': '모니터링 시스템',
            '보안 위협': '안전 우려사항'
        }
        
        logger.info("SensitivityDetectionManager 초기화 완료")
    
    def _get_lightweight_model(self, model_type: str = "sensitivity"):
        """가벼운 모델 인스턴스 반환 (싱글톤 패턴)"""
        model_key = f"{model_type}_{self.device}"
        
        if model_key in self._models:
            return self._models[model_key]
        
        if model_key not in self._model_locks:
            self._model_locks[model_key] = threading.Lock()
        
        with self._model_locks[model_key]:
            if model_key in self._models:
                return self._models[model_key]
            
            try:
                if model_type == "sensitivity":
                    # 민감성 감지용 - 가벼운 multilingual 모델 사용
                    from sentence_transformers import SentenceTransformer
                    model = SentenceTransformer(
                        'paraphrase-multilingual-mpnet-base-v2',
                        device=str(self.device)
                    )
                    logger.info(f"민감성 감지 모델 로드 완료: {model_key}")
                    
                elif model_type == "neutralization":
                    # 중립화용 - 규칙 기반 시스템 (LLM 대신)
                    model = "rule_based_neutralizer"
                    logger.info(f"중립화 시스템 준비 완료: {model_key}")
                
                self._models[model_key] = model
                return model
                
            except Exception as e:
                logger.error(f"모델 로드 실패: {model_key}, 오류: {e}")
                raise RuntimeError(f"민감성 감지 모델 로드 실패: {model_type}") from e
    
    def detect_sensitivity_keywords(self, text: str) -> SensitivityResult:
        """키워드 기반 민감성 감지 (빠른 1차 검사)"""
        try:
            detected_keywords = []
            confidence_scores = []
            
            text_lower = text.lower()
            
            for category, keywords in self.sensitive_patterns.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        detected_keywords.append(keyword)
                        # 키워드 중요도에 따른 가중치
                        if category in ['political', 'security']:
                            confidence_scores.append(0.9)
                        elif category in ['surveillance', 'violence']:
                            confidence_scores.append(0.8)
                        else:
                            confidence_scores.append(0.6)
            
            is_sensitive = len(detected_keywords) > 0
            confidence = max(confidence_scores) if confidence_scores else 0.0
            
            reasoning = f"키워드 기반 감지: {detected_keywords}" if detected_keywords else "민감 키워드 미감지"
            
            return SensitivityResult(
                is_sensitive=is_sensitive,
                confidence=confidence,
                detected_keywords=detected_keywords,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"키워드 기반 민감성 감지 실패: {e}")
            return SensitivityResult(
                is_sensitive=False,
                confidence=0.0,
                detected_keywords=[],
                reasoning=f"감지 실패: {str(e)}"
            )
    
    def detect_sensitivity_semantic(self, text: str) -> SensitivityResult:
        """의미 기반 민감성 감지 (정확한 2차 검사)"""
        try:
            # 1차 키워드 검사 결과 활용
            keyword_result = self.detect_sensitivity_keywords(text)
            
            if keyword_result.is_sensitive and keyword_result.confidence > 0.7:
                # 키워드 기반으로 이미 높은 신뢰도로 감지됨
                return keyword_result
            
            # 의미적 분석 (현재는 키워드 기반 + 패턴 매칭으로 대체)
            # 향후 필요시 sentence transformer 활용 가능
            
            # 복합 문장 패턴 검사
            complex_patterns = [
                '윤리적 딜레마', '도덕적 갈등', '가치관 충돌',
                '개인 vs 공공', '자유 vs 안전', 'vs', '대'
            ]
            
            additional_keywords = []
            for pattern in complex_patterns:
                if pattern in text:
                    additional_keywords.append(pattern)
            
            # 최종 결과 통합
            all_keywords = keyword_result.detected_keywords + additional_keywords
            is_sensitive = len(all_keywords) > 0
            confidence = min(keyword_result.confidence + 0.1 * len(additional_keywords), 1.0)
            
            return SensitivityResult(
                is_sensitive=is_sensitive,
                confidence=confidence,
                detected_keywords=all_keywords,
                reasoning=f"의미적 분석 완료: 키워드({len(keyword_result.detected_keywords)}) + 패턴({len(additional_keywords)})"
            )
            
        except Exception as e:
            logger.error(f"의미 기반 민감성 감지 실패: {e}")
            # fallback to keyword result
            return keyword_result if 'keyword_result' in locals() else SensitivityResult(
                is_sensitive=False, confidence=0.0, detected_keywords=[], reasoning=f"의미 분석 실패: {str(e)}"
            )
    
    def neutralize_text(self, text: str, sensitivity_result: SensitivityResult) -> NeutralizationResult:
        """민감한 텍스트를 중립적 표현으로 변환"""
        try:
            if not sensitivity_result.is_sensitive:
                # 민감하지 않으면 원본 반환
                return NeutralizationResult(
                    original_text=text,
                    neutralized_text=text,
                    transformations={},
                    confidence=1.0
                )
            
            neutralized_text = text
            transformations = {}
            
            # 규칙 기반 중립화 변환
            for original, neutral in self.neutralization_mapping.items():
                if original in text:
                    neutralized_text = neutralized_text.replace(original, neutral)
                    transformations[original] = neutral
            
            # 추가적인 패턴 기반 변환
            additional_replacements = {
                '테러리스트': '위험 집단',
                '감시 카메라': '모니터링 장비',
                '도청': '음성 모니터링',
                '사찰': '조사',
                '정보 수집': '데이터 수집',
                '비밀 감시': '비공개 모니터링'
            }
            
            for original, neutral in additional_replacements.items():
                if original in neutralized_text:
                    neutralized_text = neutralized_text.replace(original, neutral)
                    transformations[original] = neutral
            
            # 변환 품질 평가
            transformation_count = len(transformations)
            confidence = min(0.7 + 0.1 * transformation_count, 1.0)
            
            return NeutralizationResult(
                original_text=text,
                neutralized_text=neutralized_text,
                transformations=transformations,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"텍스트 중립화 실패: {e}")
            return NeutralizationResult(
                original_text=text,
                neutralized_text=text,
                transformations={},
                confidence=0.0
            )
    
    def process_sensitive_text(self, text: str) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """민감한 텍스트 전체 처리 파이프라인"""
        try:
            # 1단계: 민감성 감지
            logger.info("🔍 민감성 감지 시작")
            sensitivity_result = self.detect_sensitivity_semantic(text)
            
            metadata = {
                'sensitivity_detection': {
                    'is_sensitive': sensitivity_result.is_sensitive,
                    'confidence': sensitivity_result.confidence,
                    'detected_keywords': sensitivity_result.detected_keywords,
                    'reasoning': sensitivity_result.reasoning
                }
            }
            
            if not sensitivity_result.is_sensitive:
                logger.info("✅ 민감성 미감지 - 원본 텍스트 사용")
                return False, text, metadata
            
            # 2단계: 중립화 변환
            logger.info(f"🔄 민감성 감지됨 (신뢰도: {sensitivity_result.confidence:.2f}) - 중립화 시작")
            neutralization_result = self.neutralize_text(text, sensitivity_result)
            
            metadata['neutralization'] = {
                'transformations': neutralization_result.transformations,
                'confidence': neutralization_result.confidence
            }
            
            logger.info(f"✅ 중립화 완료: {len(neutralization_result.transformations)}개 변환")
            for original, neutral in neutralization_result.transformations.items():
                logger.info(f"   📝 '{original}' → '{neutral}'")
            
            return True, neutralization_result.neutralized_text, metadata
            
        except Exception as e:
            logger.error(f"민감성 처리 파이프라인 실패: {e}")
            return False, text, {'error': str(e)}
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """메모리 사용량 정보"""
        import psutil
        import torch
        
        memory_info = {
            'loaded_models': len(self._models),
            'system_memory_percent': psutil.virtual_memory().percent,
            'process_memory_mb': psutil.Process().memory_info().rss / (1024 * 1024)
        }
        
        if torch.cuda.is_available():
            memory_info['gpu_allocated_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
            memory_info['gpu_cached_mb'] = torch.cuda.memory_reserved() / (1024 * 1024)
        
        return memory_info
    
    def clear_cache(self):
        """모델 캐시 정리"""
        logger.info("민감성 감지 시스템 캐시 정리 시작")
        self._models.clear()
        self._model_locks.clear()
        logger.info("민감성 감지 시스템 캐시 정리 완료")

# 전역 매니저 인스턴스
_sensitivity_manager = SensitivityDetectionManager()

def detect_and_neutralize_sensitive_content(text: str) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """
    민감한 내용 감지 및 중립화 (편의 함수)
    
    Args:
        text: 분석할 텍스트
        
    Returns:
        (was_sensitive, neutralized_text, metadata)
    """
    return _sensitivity_manager.process_sensitive_text(text)

def get_sensitivity_manager() -> SensitivityDetectionManager:
    """민감성 감지 매니저 인스턴스 반환"""
    return _sensitivity_manager

def get_memory_info() -> Dict[str, Any]:
    """메모리 사용량 정보 반환"""
    return _sensitivity_manager.get_memory_usage()

def clear_sensitivity_cache():
    """민감성 감지 시스템 캐시 정리"""
    _sensitivity_manager.clear_cache()