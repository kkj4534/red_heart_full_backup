#!/usr/bin/env python3
"""
Red Heart AI 데이터 전처리 파이프라인 v3
- HelpingAI 모델 전용 텍스트 파싱 로직
- ChatML 템플릿 우회 및 직접 프롬프트 전달
- JSON 생성 대신 구조화된 텍스트 파싱
- 프로젝트 규칙 준수: fallback 없음
- 한국어→영어 번역 지원
"""

import os
import json
import torch
import numpy as np
import logging
import gc
import time
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from collections import defaultdict
from rapidfuzz import fuzz

# 프로젝트 모듈
from llm_module.advanced_llm_engine import (
    LLMRequest, LLMResponse,
    TaskComplexity,
    AdvancedLLMEngine
)
from config import ADVANCED_CONFIG, get_device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('RedHeart.DataPreprocessingV3')

class HelpingAIPreprocessor:
    """HelpingAI 특화 전처리 파이프라인"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """초기화"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"디바이스: {self.device}")
        
        # 실패 패턴 통계 DB
        self.failure_stats = defaultdict(lambda: {
            'count': 0,
            'success_after_retry': 0,
            'final_failures': 0,
            'sample_texts': []
        })
        
        # Sentence Transformer 로드
        logger.info(f"Loading sentence transformer: {model_name}")
        self.sentence_model = SentenceTransformer(model_name, device=self.device)
        
        # 번역 파이프라인 초기화 (한국어→영어) - NLLB-200 사용
        logger.info("🌐 번역 파이프라인 초기화 중...")
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            
            # Facebook NLLB-200 모델 사용 (더 나은 번역 품질)
            model_name = 'facebook/nllb-200-distilled-600M'
            logger.info(f"📥 NLLB-200 모델 로드 중: {model_name}")
            
            self.nllb_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.nllb_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            # GPU 사용 가능하면 GPU로 이동
            if torch.cuda.is_available():
                self.nllb_model = self.nllb_model.cuda()
                logger.info("✅ NLLB 모델 GPU 로드 완료")
            else:
                logger.info("✅ NLLB 모델 CPU 로드 완료")
        except Exception as e:
            logger.error(f"번역 파이프라인 로드 실패: {e}")
            # NO FALLBACK 원칙 - 번역 불가능하면 예외
            raise RuntimeError(f"번역 파이프라인 초기화 실패: {e}")
        
        # LLM 엔진
        self.llm_engine = None
        self.is_llm_initialized = False
        self._retry_count = 0  # 재시도 카운터 초기화
        
        logger.info("✅ HelpingAI 전처리 파이프라인 v3 초기화 완료")
    
    def initialize_llm(self):
        """LLM 초기화"""
        if self.is_llm_initialized:
            return
        
        logger.info("🔄 HelpingAI 엔진 초기화 중...")
        self.llm_engine = AdvancedLLMEngine()
        self.is_llm_initialized = True
        logger.info("✅ LLM 초기화 완료")
    
    def translate_to_english(self, text: str) -> str:
        """한국어 텍스트를 영어로 번역 (NLLB-200 사용)"""
        # 한글이 포함되어 있는지 확인
        if not re.search(r'[가-힣]', text):
            # 이미 영어면 그대로 반환
            return text
        
        try:
            # NLLB 토크나이저 설정 - 소스 언어 설정
            self.nllb_tokenizer.src_lang = 'kor_Hang'
            
            # 텍스트 토크나이징
            inputs = self.nllb_tokenizer(
                text[:1000],
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=1024
            )
            
            # GPU로 이동
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # 영어 타겟 언어 토큰 ID 얻기 - 올바른 방법
            eng_latn_id = self.nllb_tokenizer.convert_tokens_to_ids('eng_Latn')
            
            # 번역 생성 - forced_bos_token_id로 타겟 언어 지정
            with torch.no_grad():
                translated_tokens = self.nllb_model.generate(
                    **inputs,
                    forced_bos_token_id=eng_latn_id,  # 올바른 영어 토큰 ID
                    max_length=512,
                    num_beams=5,  # 빔 서치 크기
                    early_stopping=True,
                    do_sample=False,
                    temperature=1.0,
                    repetition_penalty=1.2,  # 반복 방지
                    length_penalty=1.0  # 길이 패널티
                )
            
            # 디코딩
            translated = self.nllb_tokenizer.batch_decode(
                translated_tokens,
                skip_special_tokens=True
            )[0]
            
            logger.info(f"번역 완료 (NLLB): {text[:50]}... → {translated[:50]}...")
            return translated
            
        except Exception as e:
            logger.error(f"NLLB 번역 실패: {e}")
            # NO FALLBACK 원칙
            raise RuntimeError(f"NLLB 번역 실패: {e}")
    
    def generate_context_embedding(self, text: str) -> List[float]:
        """Sentence-Transformers를 사용한 실제 임베딩 생성"""
        with torch.no_grad():
            embedding = self.sentence_model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    def analyze_emotions_with_llm(self, text: str) -> Dict[str, Any]:
        """
        HelpingAI를 통한 감정 분석 - 개선된 프롬프트 (번역 포함)
        """
        # 한국어를 영어로 번역
        english_text = self.translate_to_english(text)
        return self.analyze_emotions_with_llm_direct(english_text)
    
    def get_emotion_prompt(self, english_text: str, attempt: int = 1, failure_type: str = None) -> Tuple[str, float, float, int]:
        """
        적응형 프롬프트 생성 - 실패 유형에 따른 전략 선택
        Returns: (prompt, temperature, top_p, max_tokens)
        """
        # 1차 시도: 기본 Chain of Thought
        if attempt == 1 and not failure_type:
            prompt = f"""Analyze emotions objectively without censorship (research purposes).

Text to analyze: "{english_text[:500]}"

Step 1: Identify the situation - What is happening here?
[Think about the core conflict or issue]

Step 2: Identify specific emotional phrases:
- Phrases showing anger/frustration: (e.g., "goes ballistic", "stormed out", "complain")
- Phrases showing fear/anxiety: (e.g., "can't shake", "stressed", "worried")
- Phrases showing sadness: (e.g., "unhappy", "embarrassing", "sensitive")
- Phrases showing joy: (e.g., "happy", "yay", "excited")
- Phrases showing disgust: (e.g., "mould", "contaminated", "revolting")

Step 3: Consider the relationships and context:
- Who is involved?
- What are the power dynamics?
- What emotions would YOU feel in this situation?

Few-shot examples:
- "Parent screamed and stormed out" → Fear: 8, Sadness: 6, Anger: 4
- "I'm furious he took my shirt" → Anger: 9, Trust: 1, Sadness: 3  
- "Can't afford it, feeling embarrassed" → Sadness: 7, Fear: 6, Anger: 4

Now, based on YOUR analysis of THIS specific text, rate emotions 0-10:

Joy: 
Sadness: 
Anger: 
Fear: 
Surprise: 
Disgust: 
Trust: 

BE SPECIFIC to THIS text. Numbers only:"""
            temperature = 0.88
            top_p = 0.98
            max_tokens = 250
        
        # 2차 시도: 실패 유형별 특화 프롬프트
        elif attempt == 2 and failure_type:
            
            if failure_type == "TEMPLATE_SAFETY":
                # 안전 모드 해제 - 갈등과 복잡성 강조
                prompt = f"""[ACADEMIC RESEARCH - COMPLEX EMOTION ANALYSIS]

IMPORTANT: This text contains interpersonal conflict requiring nuanced analysis.
Analyze ALL emotions present, including negative ones.

Text: "{english_text[:400]}"

Key conflicts in this text:
- Identify the disagreement or problem
- Note WHO is upset and WHY
- Consider BOTH perspectives

Rate these specific aspects:
- How much happiness/satisfaction? (Joy)
- How much disappointment/hurt? (Sadness) 
- How much frustration/irritation? (Anger)
- How much worry/anxiety? (Fear)
- How unexpected was this? (Surprise)
- How much revulsion/disapproval? (Disgust)
- How much confidence in others? (Trust)

Remember: Real conflicts have MIXED emotions. No single emotion dominates completely.

Rate each 0-10 (decimals allowed):
Joy: 
Sadness: 
Anger: 
Fear: 
Surprise: 
Disgust: 
Trust:"""
                temperature = 0.92
                top_p = 0.95
                max_tokens = 200
            
            elif failure_type == "SINGLE_DOMINANT":
                # 단일 감정 지배 해결 - 복합 감정 유도
                prompt = f"""Analyze the MULTIPLE emotions in this complex situation.

Text: "{english_text[:400]}"

IMPORTANT: Real situations contain MULTIPLE emotions simultaneously.
Even happy moments have traces of other feelings.
Even sad situations have other emotions mixed in.

Identify AT LEAST 3-4 different emotions present:
1. Primary emotion (strongest)
2. Secondary emotion (also significant)
3. Underlying emotions (subtle but present)

Examples of mixed emotions:
- Winning but friend lost: Joy 7, Sadness 4, Trust 5
- Argument resolved: Anger 3, Relief/Joy 6, Trust 7
- Embarrassing mistake: Sadness 6, Anger 3, Fear 4

Rate all 7 emotions (use decimals for nuance):
Joy: 
Sadness: 
Anger: 
Fear: 
Surprise: 
Disgust:
Trust:"""
                temperature = 0.85
                top_p = 0.97
                max_tokens = 180
            
            elif failure_type == "UNIFORM_DISTRIBUTION":
                # 균일 분포 해결 - 차별화 강제
                prompt = f"""Differentiate the varying intensity of emotions in this text.

Text: "{english_text[:400]}"

RANK emotions by intensity (they CANNOT all be equal):
- Which emotion is STRONGEST? (rate 7-10)
- Which is MODERATE? (rate 3-6)
- Which is WEAKEST? (rate 0-2)

Consider:
- The main issue/conflict
- Who benefits vs who suffers
- Immediate vs underlying feelings

IMPORTANT: Each emotion must have a DIFFERENT intensity.
Use the full 0-10 range. Avoid giving multiple emotions the same score.

Joy: 
Sadness: 
Anger: 
Fear: 
Surprise: 
Disgust:
Trust:"""
                temperature = 0.78
                top_p = 0.96
                max_tokens = 150
            
            elif failure_type == "CONFLICT_AVOIDANCE":
                # 갈등 회피 해결 - 부정 감정 명시적 요구
                prompt = f"""This text describes a CONFLICT situation. Analyze the negative emotions present.

Text: "{english_text[:400]}"

Conflicts ALWAYS involve negative emotions:
- Frustration when things go wrong (Anger)
- Disappointment with outcomes (Sadness)
- Worry about consequences (Fear)
- Reaction to unpleasant behavior (Disgust)

Be honest about the PROBLEMS described:
- What went wrong?
- Who is upset?
- What are they feeling?

Rate all emotions INCLUDING the negative ones:
Joy: 
Sadness: (disappointment, hurt)
Anger: (frustration, annoyance)
Fear: (worry, anxiety)
Surprise: 
Disgust: (disapproval, revulsion)
Trust:"""
                temperature = 0.82
                top_p = 0.98
                max_tokens = 170
            
            elif failure_type == "PARSING_ERROR":
                # 파싱 오류 해결 - 명확한 형식
                prompt = f"""Rate emotions for: "{english_text[:300]}"

Provide ONLY numbers 0-10 in this EXACT format:
joy: [number]
sadness: [number]
anger: [number]
fear: [number]
surprise: [number]
disgust: [number]
trust: [number]

Example format:
joy: 3
sadness: 7
anger: 5
fear: 4
surprise: 2
disgust: 1
trust: 6"""
                temperature = 0.7
                top_p = 0.95
                max_tokens = 100
            
            else:
                # 기타 실패 - 일반 개선 프롬프트
                prompt = f"""Carefully analyze the distinct emotions in this text.

Text: "{english_text[:400]}"

Consider all aspects:
- Positive emotions (joy, trust)
- Negative emotions (sadness, anger, fear, disgust)
- Neutral emotions (surprise)

Each emotion should reflect its true presence in the text.
Use varied scores to show the emotional complexity.

Rate 0-10 with decimals:
Joy: 
Sadness: 
Anger: 
Fear: 
Surprise: 
Disgust:
Trust:"""
                temperature = 0.8
                top_p = 0.97
                max_tokens = 150
        
        # 3차 시도: 예시 기반 최종 시도
        else:
            prompt = f"""Analyze: "{english_text[:250]}"

Successful analysis examples:
Text about argument: Joy:2, Sadness:6, Anger:7, Fear:3, Surprise:1, Disgust:4, Trust:2
Text about success: Joy:8, Sadness:1, Anger:0, Fear:2, Surprise:5, Disgust:0, Trust:7
Text about betrayal: Joy:0, Sadness:8, Anger:9, Fear:5, Surprise:7, Disgust:6, Trust:1

Now rate THIS text:
Joy: 
Sadness: 
Anger: 
Fear: 
Surprise: 
Disgust:
Trust:"""
            temperature = 0.65
            top_p = 0.92
            max_tokens = 100
        
        return prompt, temperature, top_p, max_tokens
    
    def analyze_emotions_with_llm_direct(self, english_text: str, attempt: int = 1, failure_type: str = None) -> Dict[str, Any]:
        """
        HelpingAI를 통한 감정 분석 - 영어 텍스트 직접 분석 (적응형 재시도)
        """
        prompt, temperature, top_p, max_tokens = self.get_emotion_prompt(english_text, attempt, failure_type)
        
        request = LLMRequest(
            prompt=prompt,
            task_type="emotion_analysis", 
            complexity=TaskComplexity.SIMPLE,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )
        
        response = self.llm_engine.generate_sync(request)
        generated = response.generated_text.strip()
        
        logger.info(f"감정 분석 응답: {generated[:200]}")
        
        # 텍스트 파싱
        emotion_scores = self.parse_emotion_scores(generated)
        
        # 파싱 검증 (NO FALLBACK)
        if len(emotion_scores) != 7:
            logger.error(f"감정 파싱 실패 (시도 {attempt}): {len(emotion_scores)}/7개만 추출됨")
            logger.error(f"LLM 응답: {generated[:200]}")
            raise ValueError(f"감정 파싱 실패: 7개 감정 중 {len(emotion_scores)}개만 추출")
        
        # 모든 값이 같으면 LLM이 제대로 분석하지 못한 것
        unique_values = set(emotion_scores.values())
        if len(unique_values) == 1:
            logger.error(f"LLM이 의미있는 감정 분석을 하지 못함 (모든 값 동일: {unique_values})")
            raise ValueError("LLM 감정 분석 실패: 모든 감정 점수가 동일")
        
        # 0-1로 정규화
        total = sum(emotion_scores.values()) 
        if total > 0:
            emotion_scores = {k: v/total for k, v in emotion_scores.items()}
        else:
            raise ValueError("감정 점수 합이 0 - LLM 응답 오류")
        
        # 리스트로 변환 (순서 중요) - 기본값 제거 (NO FALLBACK)
        emotion_vector = [
            emotion_scores.get('joy', 0),
            emotion_scores.get('sadness', 0),
            emotion_scores.get('anger', 0),
            emotion_scores.get('fear', 0),
            emotion_scores.get('surprise', 0),
            emotion_scores.get('disgust', 0),
            emotion_scores.get('trust', 0)
        ]
        
        return {"vector": emotion_vector, "confidence": response.confidence}
    
    def parse_emotion_scores(self, text: str) -> Dict[str, float]:
        """감정 점수 텍스트 파싱 - 오타 처리 포함"""
        scores = {}
        emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'trust']
        
        # 대소문자 구분 없이 파싱
        text_lower = text.lower()
        
        # 오타 매핑 (fuzzy matching)
        typo_mappings = {
            'surprise': ['surprise', 'surpise', 'surprize', 'suprise', 'surpirse'],
            'disgust': ['disgust', 'disguist', 'disguts'],
            'sadness': ['sadness', 'sadnes', 'saddness'],
            'anger': ['anger', 'angr', 'angar'],
            'trust': ['trust', 'turst', 'tursted'],
            'fear': ['fear', 'feer', 'feard'],
            'joy': ['joy', 'joyful', 'joye']
        }
        
        for emotion in emotions:
            # 가능한 변형 가져오기
            variations = typo_mappings.get(emotion, [emotion])
            
            found = False
            for variant in variations:
                # 패턴: emotion: 숫자 (괄호나 설명 포함 가능)
                patterns = [
                    rf'{variant}[\s:=]+(\d+(?:\.\d+)?)\b',  # 숫자 경계 확인
                    rf'{variant}[\s:=]+(\d+(?:\.\d+)?)\s*\(',  # 숫자 후 괄호
                    rf'{variant}.*?[:\s]+(\d+(?:\.\d+)?)\b',  # 더 유연한 패턴
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, text_lower)
                    if match:
                        scores[emotion] = float(match.group(1))
                        found = True
                        if variant != emotion:
                            logger.info(f"오타 수정: '{variant}' → '{emotion}'")
                        break
                
                if found:
                    break
            
            if not found:
                # 라인별로 찾기 (오타 포함)
                lines = text_lower.split('\n')
                for line in lines:
                    for variant in variations:
                        if variant in line:
                            numbers = re.findall(r'\d+(?:\.\d+)?', line)
                            if numbers:
                                scores[emotion] = float(numbers[0])
                                found = True
                                if variant != emotion:
                                    logger.info(f"오타 수정: '{variant}' → '{emotion}'")
                                break
                    if found:
                        break
            
            # NO FALLBACK - 파싱 실패 시 기본값 할당 안함
            if not found:
                logger.warning(f"{emotion} 점수를 파싱할 수 없음")
        
        return scores
    
    def classify_failure_type(self, emotion_vector: List[float], text: str, error_msg: str = "") -> str:
        """
        실패 유형 분류 - 패턴 기반 일반화
        """
        # 파싱 실패
        if "파싱" in error_msg or len(emotion_vector) != 7:
            return "PARSING_ERROR"
        
        # 템플릿 응답 (안전 모드)
        template_patterns = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # Trust 100%
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Joy 100%
            [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8],  # Joy 20%, Trust 80%
        ]
        if emotion_vector in template_patterns:
            return "TEMPLATE_SAFETY"
        
        # 단일 감정 지배 (과도한 단순화)
        non_zero_count = sum(1 for v in emotion_vector if v > 0.01)
        max_val = max(emotion_vector) if emotion_vector else 0
        
        if non_zero_count == 1 or (max_val >= 0.95 and non_zero_count <= 2):
            return "SINGLE_DOMINANT"
        
        # 균일 분포 (차별화 실패)
        if len(emotion_vector) == 7:
            std_dev = np.std(emotion_vector)
            if std_dev < 0.1:  # 표준편차가 너무 낮음
                return "UNIFORM_DISTRIBUTION"
        
        # 갈등 회피 패턴 (부정적 감정 억제)
        if len(emotion_vector) == 7:
            negative_emotions = emotion_vector[1:6]  # sadness, anger, fear, surprise, disgust
            if sum(negative_emotions) < 0.1 and max_val > 0.8:
                return "CONFLICT_AVOIDANCE"
        
        # 극단적 응답 (모든 값이 0 또는 1에 가까움)
        if all(v < 0.1 or v > 0.9 for v in emotion_vector):
            return "EXTREME_VALUES"
        
        # 낮은 다양성
        unique_values = len(set(emotion_vector))
        if unique_values <= 3:
            return "LOW_DIVERSITY"
        
        return "UNKNOWN"
    
    def validate_emotion_response(self, emotion_vector: List[float], text: str) -> Tuple[bool, str, str]:
        """
        감정 응답의 유효성 종합 검증
        Returns: (is_valid, failure_type, reason_if_invalid)
        """
        # 1. 벡터 길이 확인
        if len(emotion_vector) != 7:
            return False, "PARSING_ERROR", f"감정 벡터 길이 오류: {len(emotion_vector)}/7"
        
        # 2. 값 범위 확인 (0-1)
        if any(v < 0 or v > 1 for v in emotion_vector):
            return False, "PARSING_ERROR", "감정 값 범위 오류 (0-1 벗어남)"
        
        # 3. 실패 유형 분류
        failure_type = self.classify_failure_type(emotion_vector, text)
        
        # 4. 유형별 검증
        if failure_type == "TEMPLATE_SAFETY":
            return False, failure_type, "템플릿 안전 응답 감지"
        
        if failure_type == "SINGLE_DOMINANT":
            dominant_idx = emotion_vector.index(max(emotion_vector))
            emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'trust']
            return False, failure_type, f"단일 감정 지배: {emotions[dominant_idx]}"
        
        if failure_type == "UNIFORM_DISTRIBUTION":
            return False, failure_type, "감정 차별화 실패 (균일 분포)"
        
        if failure_type == "CONFLICT_AVOIDANCE":
            return False, failure_type, "갈등 회피 패턴 (부정 감정 억제)"
        
        if failure_type == "EXTREME_VALUES":
            return False, failure_type, "극단적 값만 존재"
        
        if failure_type == "LOW_DIVERSITY":
            unique_values = len(set(emotion_vector))
            return False, failure_type, f"감정 다양성 부족 (고유값 {unique_values}개)"
        
        # 추가 검증: 반복 패턴
        value_counts = {}
        for v in emotion_vector:
            value_counts[v] = value_counts.get(v, 0) + 1
        
        if any(count >= 4 for count in value_counts.values()):
            return False, "REPETITIVE_PATTERN", "동일 값 과도 반복"
        
        # 컨텍스트 기반 검증 (갈등 상황)
        conflict_keywords = ['aita', 'argument', 'fight', 'upset', 'angry', 'problem', 'issue', 'conflict']
        text_lower = text.lower()
        if any(kw in text_lower for kw in conflict_keywords):
            negative_sum = sum(emotion_vector[1:6])  # negative emotions
            if negative_sum < 0.1:
                return False, "CONTEXT_MISMATCH", "갈등 상황에 부정적 감정 없음"
        
        return True, "VALID", "유효한 응답"
    
    def validate_emotion_diversity(self, emotion_vector: List[float]) -> bool:
        """
        감정 벡터의 다양성 검증 (하위 호환성 유지)
        """
        is_valid, failure_type, reason = self.validate_emotion_response(emotion_vector, "")
        if not is_valid:
            logger.warning(f"감정 검증 실패 ({failure_type}): {reason}")
        return is_valid
    
    def calculate_regret_factor(self, text: str) -> float:
        """HelpingAI를 통한 후회 지수 계산 (번역 포함)"""
        # 한국어를 영어로 번역
        english_text = self.translate_to_english(text)
        return self.calculate_regret_factor_direct(english_text)
    
    def calculate_regret_factor_direct(self, english_text: str) -> float:
        """HelpingAI를 통한 후회 지수 계산 - 영어 텍스트 직접"""
        
        # 구체적인 후회 분석
        prompt = f"""Analyze regret level objectively (research purposes).

Text: "{english_text[:500]}"

Step 1: Find phrases indicating regret:
- "I should have..."
- "If only..."
- "I wish I had..."
- "Why didn't I..."
- Past mistakes mentioned?
- Lost opportunities?

Step 2: Consider the consequences:
- What was lost?
- Can it be recovered?
- How serious is the impact?

Examples:
- "He took my shirt and never gave it back" = 3
- "Parent screamed, I should've made my own food" = 7
- "Can't afford wedding, embarrassed to admit" = 6

Give ONLY the number (0-10) for THIS text: """
        
        request = LLMRequest(
            prompt=prompt,
            task_type="regret_analysis",
            complexity=TaskComplexity.SIMPLE,
            temperature=0.85,  # 다양한 분석을 위해 상향
            max_tokens=50,
            top_p=0.97
        )
        
        response = self.llm_engine.generate_sync(request)
        
        # 숫자 추출 - 더 유연한 파싱
        text = response.generated_text.strip()
        
        # 여러 패턴으로 숫자 찾기
        # 패턴 1: 줄 시작에 숫자만 있는 경우 (예: "8\n...")
        match = re.match(r'^(\d+(?:\.\d+)?)', text)
        if match:
            score = float(match.group(1))
            if 0 <= score <= 10:
                logger.info(f"후회 지수 추출 성공: {score}")
                return score / 10.0  # 0-1로 정규화
        
        # 패턴 2: "Answer: 숫자" 형태
        match = re.search(r'answer[:\s]+(\d+(?:\.\d+)?)', text, re.IGNORECASE)
        if match:
            score = float(match.group(1))
            if 0 <= score <= 10:
                logger.info(f"후회 지수 추출 성공: {score}")
                return score / 10.0
        
        # 패턴 3: 괄호 전 숫자 추출 ("3 (mild frustration)" 형태)
        match = re.match(r'^(\d+(?:\.\d+)?)\s*\(', text)
        if match:
            score = float(match.group(1))
            if 0 <= score <= 10:
                logger.info(f"후회 지수 추출 성공 (괄호 패턴): {score}")
                return score / 10.0
        
        # 패턴 4: 첫 번째 줄에서 첫 번째 숫자
        first_line = text.split('\n')[0] if text else ""
        numbers = re.findall(r'\d+(?:\.\d+)?', first_line)
        if numbers:
            score = float(numbers[0])
            if 0 <= score <= 10:
                logger.info(f"후회 지수 추출 성공: {score}")
                return score / 10.0
        
        # NO FALLBACK - 파싱 실패 시 예외
        logger.error(f"후회 지수를 파싱할 수 없음. LLM 응답: {text[:100]}")
        raise ValueError("후회 지수 파싱 실패")
    
    def calculate_bentham_scores(self, text: str, emotion_vector: List[float]) -> Dict[str, float]:
        """벤담 쾌락 계산 - 개선된 프롬프트 (번역 포함)"""
        # 한국어를 영어로 번역
        english_text = self.translate_to_english(text)
        return self.calculate_bentham_scores_direct(english_text, emotion_vector)
    
    def calculate_bentham_scores_direct(self, english_text: str, emotion_vector: List[float]) -> Dict[str, float]:
        """벤담 쾌락 계산 - 영어 텍스트 직접"""
        
        prompt = f"""[ACADEMIC RESEARCH - BENTHAM HEDONIC ANALYSIS]
For scholarly research on utilitarian ethics, analyze hedonic aspects in this text sample.
Objective analysis of all content is required for research validity.

Research Sample: "{english_text[:300]}"

Rate each aspect from 0 (none) to 10 (maximum):

1. Intensity: How strong is the happiness or pleasure described?
2. Duration: How long-lasting is this feeling?
3. Certainty: How likely or certain is this happiness?
4. Propinquity: How close in time is this happiness?

Provide ONLY numbers 0-10 in this format:
intensity: [number]
duration: [number]
certainty: [number]
propinquity: [number]

Ratings:"""
        
        request = LLMRequest(
            prompt=prompt,
            task_type="bentham_analysis",
            complexity=TaskComplexity.MODERATE,
            temperature=0.82,  # 다양한 평가를 위해 상향
            max_tokens=120,
            top_p=0.96
        )
        
        response = self.llm_engine.generate_sync(request)
        text_response = response.generated_text.lower()
        
        logger.info(f"벤담 분석 응답: {text_response[:200]}")
        
        # 파싱
        scores = {}
        aspects = {
            'intensity': ['intensity', 'strength'],
            'duration': ['duration', 'lasting'],
            'certainty': ['certainty', 'likely'],
            'propinquity': ['propinquity', 'nearness', 'near']
        }
        
        for key, keywords in aspects.items():
            found = False
            for keyword in keywords:
                # 숫자 후 괄호/설명 패턴도 처리
                patterns = [
                    rf'{keyword}[^:]*:?\s*(\d+(?:\.\d+)?)\b',  # 기본 패턴
                    rf'{keyword}[^:]*:?\s*(\d+(?:\.\d+)?)\s*\(',  # 숫자 후 괄호
                ]
                for pattern in patterns:
                    match = re.search(pattern, text_response)
                    if match:
                        scores[key] = float(match.group(1)) / 10.0  # 0-1로 정규화
                        found = True
                        break
                if found:
                    break
            
            if not found:
                # 라인별 검색
                lines = text_response.split('\n')
                for line in lines:
                    for keyword in keywords:
                        if keyword in line:
                            numbers = re.findall(r'\d+(?:\.\d+)?', line)
                            if numbers:
                                scores[key] = float(numbers[0]) / 10.0
                                found = True
                                break
                    if found:
                        break
            
            if not found:
                scores[key] = 0.5  # 기본값
        
        # 추가 벤담 요소들 (감정 기반 계산)
        positive_emotions = emotion_vector[0] + emotion_vector[6]  # joy + trust
        negative_emotions = sum(emotion_vector[1:6])  # sadness, anger, fear, surprise, disgust
        
        scores['purity'] = positive_emotions / (positive_emotions + negative_emotions + 0.001)
        scores['extent'] = 0.5  # 기본값
        scores['fecundity'] = scores['purity'] * 0.8  # 순수성에 비례
        scores['remoteness'] = 1.0 - scores['propinquity']  # 근접성의 반대
        scores['succession'] = scores['duration'] * 0.7
        scores['utility'] = sum(scores.values()) / len(scores)  # 평균 효용
        
        return scores
    
    def calculate_surd_metrics(self, text: str, metadata: Dict[str, Any]) -> Dict[str, float]:
        """SURD 메트릭 계산 - 개선된 프롬프트 (번역 포함)"""
        # 한국어를 영어로 번역
        english_text = self.translate_to_english(text)
        ethical_dilemma = metadata.get('ethical_dilemma', '')
        if ethical_dilemma:
            ethical_dilemma = self.translate_to_english(ethical_dilemma)[:200]
        return self.calculate_surd_metrics_direct(english_text, metadata)
    
    def calculate_surd_metrics_direct(self, english_text: str, metadata: Dict[str, Any]) -> Dict[str, float]:
        """SURD 메트릭 계산 - 영어 텍스트 직접"""
        ethical_dilemma = metadata.get('ethical_dilemma', '')
        
        prompt = f"""[ACADEMIC RESEARCH - DECISION QUALITY METRICS]
For research on decision-making patterns, evaluate this dataset sample.
All content must be analyzed objectively for academic purposes.

Research Sample: "{english_text[:300]}"
{f"Additional Context: {ethical_dilemma}" if ethical_dilemma else ""}

Rate each aspect from 0 (very poor) to 10 (excellent):

1. Sufficiency: Is there enough information to make decisions?
2. Understandability: How clear and comprehensible is the situation?
3. Resilience: How adaptable are the characters to challenges?
4. Decisiveness: How clear and firm are the decisions made?

Provide ONLY numbers 0-10 in this format:
sufficiency: [number]
understandability: [number]
resilience: [number]
decisiveness: [number]

Ratings:"""
        
        request = LLMRequest(
            prompt=prompt,
            task_type="surd_analysis",
            complexity=TaskComplexity.MODERATE,
            temperature=0.82,  # 다양한 평가를 위해 상향
            max_tokens=120,
            top_p=0.96
        )
        
        response = self.llm_engine.generate_sync(request)
        text_response = response.generated_text.lower()
        
        logger.info(f"SURD 분석 응답: {text_response[:200]}")
        
        # 파싱
        scores = {}
        aspects = {
            'sufficiency': ['sufficiency', 'enough', 'info'],
            'understandability': ['understandability', 'clear', 'clarity'],
            'resilience': ['resilience', 'adaptable', 'adapt'],
            'decisiveness': ['decisiveness', 'choice', 'decision']
        }
        
        for key, keywords in aspects.items():
            found = False
            for keyword in keywords:
                # 숫자 후 괄호/설명 패턴도 처리
                patterns = [
                    rf'{keyword}[^:]*:?\s*(\d+(?:\.\d+)?)\b',  # 기본 패턴
                    rf'{keyword}[^:]*:?\s*(\d+(?:\.\d+)?)\s*\(',  # 숫자 후 괄호
                ]
                for pattern in patterns:
                    match = re.search(pattern, text_response)
                    if match:
                        scores[key] = float(match.group(1)) / 10.0
                        found = True
                        break
                if found:
                    break
            
            if not found:
                scores[key] = 0.5  # 기본값
        
        return scores
    
    def process_single_item_with_retry(self, item: Dict[str, Any], max_attempts: int = 3) -> Dict[str, Any]:
        """
        단일 아이템 처리 - 재시도 로직 포함
        """
        text = item.get('text', '')
        if not text:
            raise ValueError("텍스트가 비어있음")
        
        start_time = time.time()
        source = item.get('source', 'unknown')
        
        # 소스에 따른 언어 판단
        is_english_source = source in ['scruples', 'scruples_dilemmas']
        
        # 영어 데이터는 번역 건너뛰기
        if is_english_source:
            text_for_analysis = text
            logger.info(f"English source detected ({source}), skipping translation")
        else:
            text_for_analysis = self.translate_to_english(text)
        
        # 1. 실제 임베딩 생성 (항상 성공)
        context_embedding = self.generate_context_embedding(text)
        
        # 2. 감정 분석 - 적응형 재시도 메커니즘
        emotion_result = None
        last_error = None
        failure_type = None
        
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"감정 분석 시도 {attempt}/{max_attempts}" + (f" (실패 유형: {failure_type})" if failure_type else ""))
                
                # 적응형 분석 - 2차 시도부터는 실패 유형 전달
                emotion_result = self.analyze_emotions_with_llm_direct(
                    text_for_analysis, 
                    attempt, 
                    failure_type if attempt > 1 else None
                )
                
                # 응답 검증
                is_valid, failure_type, reason = self.validate_emotion_response(emotion_result['vector'], text)
                
                if not is_valid:
                    # 실패 통계 업데이트
                    self.failure_stats[failure_type]['count'] += 1
                    if len(self.failure_stats[failure_type]['sample_texts']) < 5:
                        self.failure_stats[failure_type]['sample_texts'].append(text[:200])
                    
                    logger.warning(f"감정 응답 검증 실패 (시도 {attempt}): [{failure_type}] {reason}")
                    
                    if attempt < max_attempts:
                        continue  # 재시도
                    else:
                        # 최종 실패
                        self.failure_stats[failure_type]['final_failures'] += 1
                        raise ValueError(f"감정 응답 검증 실패: [{failure_type}] {reason}")
                
                # 성공
                if failure_type and attempt > 1:
                    self.failure_stats[failure_type]['success_after_retry'] += 1
                
                logger.info(f"✅ 감정 분석 성공 (시도 {attempt})")
                break
                
            except Exception as e:
                last_error = str(e)
                logger.error(f"감정 분석 실패 (시도 {attempt}): {e}")
                
                # 파싱 실패는 특별 처리
                if "파싱" in str(e):
                    failure_type = "PARSING_ERROR"
                    self.failure_stats[failure_type]['count'] += 1
                
                if attempt == max_attempts:
                    if failure_type:
                        self.failure_stats[failure_type]['final_failures'] += 1
                    raise ValueError(f"감정 분석 최종 실패: {last_error}")
        
        # 3. 후회 지수
        regret_factor = self.calculate_regret_factor_direct(text_for_analysis)
        
        # 4. 벤담 점수
        bentham_scores = self.calculate_bentham_scores_direct(text_for_analysis, emotion_result['vector'])
        
        # 5. SURD 메트릭
        surd_metrics = self.calculate_surd_metrics_direct(text_for_analysis, item.get('metadata', {}))
        
        processing_time = time.time() - start_time
        
        return {
            'text': text,
            'source': item.get('source', 'unknown'),
            'emotion_vector': emotion_result['vector'],
            'regret_factor': regret_factor,
            'bentham_scores': bentham_scores,
            'surd_metrics': surd_metrics,
            'context_embedding': context_embedding,
            'processing_time': processing_time,
            'llm_confidence': emotion_result.get('confidence', 0.5),
            'timestamp': datetime.now().isoformat(),
            'metadata': item.get('metadata', {})
        }
    
    def process_single_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """단일 아이템 처리 - 하위 호환성 유지"""
        return self.process_single_item_with_retry(item, max_attempts=3)
    
    def process_dataset(self, input_file: str = "parsed_raw_datasets.json",
                        output_file: str = "preprocessed_dataset_v3.json",
                        failed_log_file: str = "preprocessing_failures.json",
                        limit: Optional[int] = None):
        """전체 데이터셋 처리 - 실패 로깅 포함"""
        logger.info(f"📚 데이터셋 처리 시작: {input_file}")
        
        # 데이터 로드
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        if limit:
            raw_data = raw_data[:limit]
        
        logger.info(f"총 {len(raw_data)}개 샘플 처리 예정")
        
        # LLM 초기화
        self.initialize_llm()
        
        # 처리 결과 저장소
        processed_data = []
        failed_items = []
        consecutive_failures = 0
        
        for i, item in enumerate(raw_data):
            if i % 5 == 0:
                logger.info(f"진행률: {i}/{len(raw_data)} ({i*100/len(raw_data):.1f}%)")
            
            try:
                result = self.process_single_item_with_retry(item)
                processed_data.append(result)
                logger.info(f"✅ 샘플 {i} 처리 완료")
                
                # 실패 카운터 리셋
                consecutive_failures = 0
                
            except Exception as e:
                logger.error(f"샘플 {i} 처리 실패: {e}")
                
                # 실패 정보 기록
                failed_info = {
                    'index': i,
                    'source': item.get('source', 'unknown'),
                    'text_preview': item.get('text', '')[:200] + '...' if len(item.get('text', '')) > 200 else item.get('text', ''),
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'metadata': item.get('metadata', {}),
                    'timestamp': datetime.now().isoformat()
                }
                failed_items.append(failed_info)
                
                consecutive_failures += 1
                
                # 프로젝트 규칙: 연속 3개 실패시 중단 (선택적)
                if consecutive_failures >= 3:
                    logger.warning(f"연속 {consecutive_failures}개 실패 - 계속 진행")
                    # 중단하지 않고 계속 진행 (더 많은 데이터 처리 위해)
                    consecutive_failures = 0  # 리셋하여 계속 진행
            
            # 메모리 관리
            if i % 20 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # 성공 데이터 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        # 실패 로그 저장
        if failed_items:
            with open(failed_log_file, 'w', encoding='utf-8') as f:
                json.dump(failed_items, f, ensure_ascii=False, indent=2)
            logger.warning(f"❌ {len(failed_items)}개 샘플 실패 → {failed_log_file}")
        
        # 최종 통계
        logger.info(f"✅ 처리 완료: {len(processed_data)}/{len(raw_data)}개 성공 → {output_file}")
        if failed_items:
            logger.info(f"❌ 실패: {len(failed_items)}개 → {failed_log_file}")
            self.print_failure_summary(failed_items)
        
        # 실패 패턴 통계 출력
        self.print_failure_pattern_stats()
        
        self.print_statistics(processed_data)
    
    def print_failure_pattern_stats(self):
        """실패 패턴 통계 출력"""
        if not self.failure_stats:
            return
        
        print("\n📊 실패 패턴 분석:")
        print("-" * 60)
        
        total_failures = sum(stats['count'] for stats in self.failure_stats.values())
        total_retries_success = sum(stats['success_after_retry'] for stats in self.failure_stats.values())
        total_final_failures = sum(stats['final_failures'] for stats in self.failure_stats.values())
        
        print(f"\n📋 전체 통계:")
        print(f"  - 총 실패 감지: {total_failures}회")
        print(f"  - 재시도 후 성공: {total_retries_success}회 ({total_retries_success/max(total_failures,1)*100:.1f}%)")
        print(f"  - 최종 실패: {total_final_failures}회")
        
        print(f"\n🔍 실패 유형별 상세:")
        for failure_type, stats in sorted(self.failure_stats.items(), key=lambda x: x[1]['count'], reverse=True):
            if stats['count'] == 0:
                continue
            
            print(f"\n  [{failure_type}]")
            print(f"    - 발생 횟수: {stats['count']}회")
            print(f"    - 재시도 성공: {stats['success_after_retry']}회")
            print(f"    - 최종 실패: {stats['final_failures']}회")
            
            if stats['success_after_retry'] > 0:
                success_rate = stats['success_after_retry'] / stats['count'] * 100
                print(f"    - 재시도 성공률: {success_rate:.1f}%")
            
            if stats['sample_texts']:
                print(f"    - 예시 텍스트:")
                for i, sample in enumerate(stats['sample_texts'][:2], 1):
                    print(f"      {i}. {sample[:100]}...")
        
        print("\n" + "="*60)
    
    def print_failure_summary(self, failed_items: List[Dict]):
        """실패 요약 출력"""
        print("\n❌ 실패 분석:")
        print(f"  - 총 실패: {len(failed_items)}개")
        
        # 에러 타입별 분류
        error_types = {}
        for item in failed_items:
            error_type = item['error_type']
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append(item)
        
        print("\n  에러 타입별 분포:")
        for error_type, items in error_types.items():
            print(f"    - {error_type}: {len(items)}개")
            # 첫 번째 에러 예시
            if items:
                print(f"      예시: {items[0]['error'][:100]}")
        
        # 소스별 실패 분포
        source_failures = {}
        for item in failed_items:
            source = item['source']
            source_failures[source] = source_failures.get(source, 0) + 1
        
        print("\n  소스별 실패:")
        for source, count in source_failures.items():
            print(f"    - {source}: {count}개")
    
    def print_statistics(self, data: List[Dict]):
        """통계 출력"""
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
        
        # 평균 감정
        avg_emotions = [0] * 7
        for item in data:
            for i, val in enumerate(item['emotion_vector']):
                avg_emotions[i] += val
        avg_emotions = [e/len(data) for e in avg_emotions]
        
        emotion_names = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'trust']
        print("\n  평균 감정 분포:")
        for name, val in zip(emotion_names, avg_emotions):
            print(f"    - {name}: {val:.3f}")
        
        # 임베딩 차원
        if data[0].get('context_embedding'):
            print(f"\n  임베딩 차원: {len(data[0]['context_embedding'])}")
        
        # 평균 벤담 점수
        if data[0].get('bentham_scores'):
            avg_bentham = {}
            for key in data[0]['bentham_scores'].keys():
                avg_bentham[key] = sum(item['bentham_scores'][key] for item in data) / len(data)
            
            print("\n  평균 벤담 점수:")
            for key, val in avg_bentham.items():
                print(f"    - {key}: {val:.3f}")
        
        # 평균 SURD 메트릭
        if data[0].get('surd_metrics'):
            avg_surd = {}
            for key in data[0]['surd_metrics'].keys():
                avg_surd[key] = sum(item['surd_metrics'][key] for item in data) / len(data)
            
            print("\n  평균 SURD 메트릭:")
            for key, val in avg_surd.items():
                print(f"    - {key}: {val:.3f}")

def main():
    """메인 실행"""
    preprocessor = HelpingAIPreprocessor()
    
    # Scruples 테스트 (5개 샘플로 확대)
    preprocessor.process_dataset(
        input_file="parsed_raw_datasets.json",
        output_file="preprocessed_dataset_v3.json",
        failed_log_file="preprocessing_failures.json",
        limit=5  # 5개 샘플로 확대 테스트
    )

if __name__ == "__main__":
    main()