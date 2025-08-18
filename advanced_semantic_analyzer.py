"""
고급 의미 분석 시스템 - Linux 전용
Advanced Semantic Analysis System for Linux

다중수준 의미론적 표현을 통해 텍스트를 표면적, 윤리적, 감정적, 인과적 
수준에서 분석하고 고급 AI 모델을 활용한 심층 의미 이해를 제공합니다.
"""

import os
import logging
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from pathlib import Path
from dataclasses import dataclass, field
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import re

# 고급 라이브러리 임포트
# SentenceTransformer는 sentence_transformer_singleton을 통해 사용
from transformers import (
    pipeline, AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    AutoModelForCausalLM, BertTokenizer, BertModel
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform
import networkx as nx

from config import ADVANCED_CONFIG, DEVICE, TORCH_DTYPE, BATCH_SIZE, MODELS_DIR
from data_models import (
    SemanticRepresentationData, SemanticLevel, IntentionCategory,
    AdvancedSemanticResult, SemanticCluster, SemanticNetwork,
    CausalRelation, EthicalDimension, EmotionalProfile
)

# 고급 라이브러리 가용성 확인
ADVANCED_LIBS_AVAILABLE = True
try:
    import torch
    import transformers
    import sentence_transformers
    import sklearn
    import scipy
    import networkx
    assert torch.cuda.is_available() if ADVANCED_CONFIG['enable_gpu'] else True
except ImportError as e:
    ADVANCED_LIBS_AVAILABLE = False
    raise ImportError(f"고급 라이브러리가 필요합니다: {e}")

logger = logging.getLogger('RedHeart.AdvancedSemanticAnalyzer')


@dataclass
class SemanticFeatureVector:
    """의미론적 특성 벡터"""
    surface_features: np.ndarray
    ethical_features: np.ndarray
    emotional_features: np.ndarray
    causal_features: np.ndarray
    integrated_features: np.ndarray
    feature_names: List[str] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)


class NeuralSemanticEncoder(nn.Module):
    """신경망 기반 의미론적 인코더"""
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 512, output_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 다층 인코더
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Tanh()
        )
        
        # 레벨별 특화 헤드
        self.surface_head = nn.Linear(output_dim, output_dim // 4)
        self.ethical_head = nn.Linear(output_dim, output_dim // 4)
        self.emotional_head = nn.Linear(output_dim, output_dim // 4)
        self.causal_head = nn.Linear(output_dim, output_dim // 4)
        
        # 어텐션 메커니즘
        self.attention = nn.MultiheadAttention(output_dim, num_heads=8, batch_first=True)
        
    def forward(self, x):
        # 기본 인코딩
        encoded = self.encoder(x)
        
        # 어텐션 적용
        attended, _ = self.attention(encoded.unsqueeze(1), encoded.unsqueeze(1), encoded.unsqueeze(1))
        attended = attended.squeeze(1)
        
        # 레벨별 특성 추출
        surface = torch.sigmoid(self.surface_head(attended))
        ethical = torch.sigmoid(self.ethical_head(attended))
        emotional = torch.tanh(self.emotional_head(attended))
        causal = torch.sigmoid(self.causal_head(attended))
        
        return {
            'integrated': attended,
            'surface': surface,
            'ethical': ethical,
            'emotional': emotional,
            'causal': causal
        }


class AdvancedTransformerProcessor:
    """고급 트랜스포머 기반 처리기"""
    
    def __init__(self):
        if not ADVANCED_LIBS_AVAILABLE:
            raise ImportError("고급 라이브러리가 필요합니다.")
            
        self.device = DEVICE
        
        # 다중 언어 모델들
        self.models = {}
        self.tokenizers = {}
        
        # 1. 멀티링구얼 의미 임베딩 (싱글톤 매니저 사용)
        from sentence_transformer_singleton import get_sentence_transformer
        
        self.semantic_model = get_sentence_transformer(
            'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
            device=str(self.device),
            cache_folder=os.path.join(MODELS_DIR, 'sentence_transformers')
        )
        
        # 2. 한국어 특화 모델 (싱글톤 매니저 사용)
        try:
            self.korean_model = get_sentence_transformer(
                'jhgan/ko-sroberta-multitask',
                device=str(self.device),
                cache_folder=os.path.join(MODELS_DIR, 'sentence_transformers')
            )
        except Exception as e:
            logger.error(f"한국어 모델 로드 실패: {e}")
            # fallback 없음 - 바로 예외 발생
            raise RuntimeError(f"한국어 SentenceTransformer 초기화 실패: {e}") from e
            
        # 3. 감정 분석 모델 (오프라인 모드) - HF 래퍼 사용
        try:
            from hf_model_wrapper import wrapped_pipeline
            self.emotion_classifier = wrapped_pipeline(
                task="text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                owner="semantic_analyzer",
                device=0 if self.device == 'cuda' else -1,
                return_all_scores=True,
                local_files_only=True,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
            )
            logger.info("✅ 감정 분석 모델 로드 완료 (HF 래퍼)")
        except ImportError:
            # HF 래퍼 없으면 기존 방식
            self.emotion_classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=0 if self.device == 'cuda' else -1,
                return_all_scores=True,
                local_files_only=True
            )
        
        # 4. 윤리적 분류 모델 (오프라인 모드) - HF 래퍼 사용
        try:
            from hf_model_wrapper import wrapped_pipeline
            self.ethical_classifier = wrapped_pipeline(
                task="zero-shot-classification",
                model="facebook/bart-large-mnli",
                owner="semantic_analyzer",
                device=0 if self.device == 'cuda' else -1,
                local_files_only=True,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
            )
            logger.info("✅ 윤리적 분류 모델 로드 완료 (HF 래퍼)")
        except ImportError:
            self.ethical_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if self.device == 'cuda' else -1,
                local_files_only=True
            )
        
        # 5. 인과관계 분석 모델 (오프라인 모드) - HF 래퍼 사용
        try:
            from hf_model_wrapper import wrapped_pipeline
            self.causal_model = wrapped_pipeline(
                task="text-classification",
                model="microsoft/DialoGPT-medium",
                owner="semantic_analyzer",
                device=0 if self.device == 'cuda' else -1,
                local_files_only=True,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
            )
            logger.info("✅ 인과관계 분석 모델 로드 완료 (HF 래퍼)")
        except ImportError:
            self.causal_model = pipeline(
                "text-classification",
                model="microsoft/DialoGPT-medium",
                device=0 if self.device == 'cuda' else -1,
                local_files_only=True
            )
        
        # 6. 한국어 NER 모델 (오프라인 모드) - HF 래퍼 사용
        try:
            from hf_model_wrapper import wrapped_pipeline
            self.ner_model = wrapped_pipeline(
                task="ner",
                model="klue/bert-base-kor-ner",
                owner="semantic_analyzer",
                device=0 if self.device == 'cuda' else -1,
                local_files_only=True,
                aggregation_strategy="simple",
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
            )
            logger.info("✅ 한국어 NER 모델 로드 완료 (HF 래퍼)")
        except:
            self.ner_model = None
            
        # 7. 의존성 파싱 (영어)
        try:
            import spacy
            self.nlp_en = spacy.load("en_core_web_sm")
        except:
            self.nlp_en = None
            
        logger.info("고급 트랜스포머 처리기 초기화 완료")
        
    def analyze_surface_level(self, text: str, language: str = "ko") -> Dict[str, Any]:
        """표면적 수준 분석"""
        results = {}
        
        try:
            # 1. 개체명 인식
            if self.ner_model and language == "ko":
                entities = self.ner_model(text)
                results['entities'] = entities
            elif self.nlp_en and language == "en":
                doc = self.nlp_en(text)
                entities = [{"entity": ent.label_, "word": ent.text, "start": ent.start_char, "end": ent.end_char} for ent in doc.ents]
                results['entities'] = entities
            else:
                results['entities'] = []
                
            # 2. 키워드 추출
            keywords = self._extract_keywords(text, language)
            results['keywords'] = keywords
            
            # 3. 구문 분석
            if self.nlp_en and language == "en":
                doc = self.nlp_en(text)
                syntax = [{"token": token.text, "pos": token.pos_, "dep": token.dep_} for token in doc]
                results['syntax'] = syntax
            else:
                results['syntax'] = []
                
            # 4. 문장 분할 및 중요도
            sentences = self._segment_sentences(text)
            sentence_importance = self._calculate_sentence_importance(sentences)
            results['sentences'] = list(zip(sentences, sentence_importance))
            
        except Exception as e:
            logger.error(f"표면적 분석 실패: {e}")
            results = {'entities': [], 'keywords': [], 'syntax': [], 'sentences': [(text, 1.0)]}
            
        return results
        
    def analyze_ethical_level(self, text: str) -> Dict[str, Any]:
        """윤리적 수준 분석"""
        results = {}
        
        try:
            # 1. 윤리적 카테고리 분류
            ethical_labels = [
                "생명과 안전 (Life and Safety)",
                "정의와 공정성 (Justice and Fairness)", 
                "자율성과 자유 (Autonomy and Freedom)",
                "취약계층 보호 (Protection of Vulnerable)",
                "사회적 책임 (Social Responsibility)",
                "정직과 진실 (Honesty and Truth)",
                "관용과 용서 (Tolerance and Forgiveness)",
                "충성과 소속 (Loyalty and Belonging)"
            ]
            
            ethical_classification = self.ethical_classifier(text, ethical_labels)
            results['ethical_categories'] = ethical_classification
            
            # 2. 도덕적 감정 분석
            moral_emotions = self._analyze_moral_emotions(text)
            results['moral_emotions'] = moral_emotions
            
            # 3. 가치 충돌 감지
            value_conflicts = self._detect_value_conflicts(text)
            results['value_conflicts'] = value_conflicts
            
            # 4. 윤리적 딜레마 식별
            ethical_dilemmas = self._identify_ethical_dilemmas(text)
            results['ethical_dilemmas'] = ethical_dilemmas
            
        except Exception as e:
            logger.error(f"윤리적 분석 실패: {e}")
            results = {'ethical_categories': [], 'moral_emotions': [], 'value_conflicts': [], 'ethical_dilemmas': []}
            
        return results
        
    def analyze_emotional_level(self, text: str, language: str = "ko") -> Dict[str, Any]:
        """감정적 수준 분석"""
        results = {}
        
        try:
            # 1. 기본 감정 분류
            if language == "en":
                emotions = self.emotion_classifier(text)
                results['basic_emotions'] = emotions
            else:
                # 한국어는 대체 방법 사용
                emotions = self._korean_emotion_analysis(text)
                results['basic_emotions'] = emotions
                
            # 2. 감정 강도 분석
            emotion_intensity = self._analyze_emotion_intensity(text)
            results['emotion_intensity'] = emotion_intensity
            
            # 3. 감정 진행 분석
            emotion_progression = self._analyze_emotion_progression(text)
            results['emotion_progression'] = emotion_progression
            
            # 4. 복합 감정 분석
            complex_emotions = self._analyze_complex_emotions(text)
            results['complex_emotions'] = complex_emotions
            
            # 5. 감정적 극성 (Valence & Arousal)
            valence_arousal = self._calculate_valence_arousal(text, language)
            results['valence_arousal'] = valence_arousal
            
        except Exception as e:
            logger.error(f"감정적 분석 실패: {e}")
            results = {'basic_emotions': [], 'emotion_intensity': 0.5, 'emotion_progression': [], 'complex_emotions': [], 'valence_arousal': {'valence': 0.0, 'arousal': 0.0}}
            
        return results
        
    def analyze_causal_level(self, text: str) -> Dict[str, Any]:
        """인과적 수준 분석"""
        results = {}
        
        try:
            # 1. 인과관계 추출
            causal_relations = self._extract_causal_relations(text)
            results['causal_relations'] = causal_relations
            
            # 2. 조건부 관계 분석
            conditional_relations = self._analyze_conditional_relations(text)
            results['conditional_relations'] = conditional_relations
            
            # 3. 시간적 순서 분석
            temporal_sequence = self._analyze_temporal_sequence(text)
            results['temporal_sequence'] = temporal_sequence
            
            # 4. 결과 예측
            consequence_predictions = self._predict_consequences(text)
            results['consequence_predictions'] = consequence_predictions
            
            # 5. 인과 네트워크 구성
            causal_network = self._build_causal_network(causal_relations)
            results['causal_network'] = causal_network
            
        except Exception as e:
            logger.error(f"인과적 분석 실패: {e}")
            results = {'causal_relations': [], 'conditional_relations': [], 'temporal_sequence': [], 'consequence_predictions': [], 'causal_network': {}}
            
        return results
        
    def _extract_keywords(self, text: str, language: str) -> List[str]:
        """키워드 추출"""
        try:
            # TF-IDF 기반 키워드 추출
            vectorizer = TfidfVectorizer(
                max_features=20,
                ngram_range=(1, 2),
                stop_words='english' if language == 'en' else None
            )
            
            # 단일 문서를 여러 문장으로 분할하여 처리
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if len(sentences) < 2:
                sentences = [text, text + " 키워드"]  # 최소 2개 문서 필요
                
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            
            # 첫 번째 문서(원본 텍스트)의 TF-IDF 점수
            scores = tfidf_matrix[0].toarray()[0]
            
            # 상위 키워드 선택
            top_indices = np.argsort(scores)[::-1][:10]
            keywords = [feature_names[i] for i in top_indices if scores[i] > 0]
            
            return keywords
            
        except Exception as e:
            logger.error(f"키워드 추출 실패: {e}")
            return text.split()[:5]
            
    def _segment_sentences(self, text: str) -> List[str]:
        """문장 분할"""
        # 한국어와 영어 모두 처리 가능한 간단한 분할
        sentences = []
        for delimiter in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
            text = text.replace(delimiter, '<SENT_BREAK>')
        
        raw_sentences = text.split('<SENT_BREAK>')
        for sent in raw_sentences:
            sent = sent.strip()
            if len(sent) > 3:  # 최소 길이 확인
                sentences.append(sent)
                
        return sentences if sentences else [text]
        
    def _calculate_sentence_importance(self, sentences: List[str]) -> List[float]:
        """문장 중요도 계산"""
        if len(sentences) <= 1:
            return [1.0] * len(sentences)
            
        try:
            # 문장 임베딩 생성
            embeddings = self.semantic_model.encode(sentences)
            
            # 중심성 기반 중요도 계산
            similarity_matrix = cosine_similarity(embeddings)
            
            # PageRank 알고리즘 적용
            G = nx.from_numpy_array(similarity_matrix)
            pagerank_scores = nx.pagerank(G)
            
            importance_scores = [pagerank_scores[i] for i in range(len(sentences))]
            
            # 정규화
            max_score = max(importance_scores) if importance_scores else 1.0
            importance_scores = [score / max_score for score in importance_scores]
            
            return importance_scores
            
        except Exception as e:
            logger.error(f"문장 중요도 계산 실패: {e}")
            return [1.0] * len(sentences)
            
    def _analyze_moral_emotions(self, text: str) -> List[Dict[str, Any]]:
        """도덕적 감정 분석"""
        moral_emotion_keywords = {
            'guilt': ['죄책감', '미안', '후회', '잘못'],
            'shame': ['부끄러움', '창피', '수치'],
            'pride': ['자부심', '자랑', '뿌듯'],
            'empathy': ['공감', '이해', '동정', '연민'],
            'indignation': ['분개', '의분', '분노'],
            'gratitude': ['감사', '고마움', '감동'],
            'admiration': ['존경', '감탄', '경외']
        }
        
        results = []
        text_lower = text.lower()
        
        for emotion, keywords in moral_emotion_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            if count > 0:
                results.append({
                    'emotion': emotion,
                    'intensity': min(count / len(keywords), 1.0),
                    'keywords_found': [kw for kw in keywords if kw in text_lower]
                })
                
        return results
        
    def _detect_value_conflicts(self, text: str) -> List[Dict[str, Any]]:
        """가치 충돌 감지"""
        conflict_patterns = [
            ('개인 vs 집단', ['개인', '자유'], ['집단', '공동체', '사회']),
            ('정의 vs 자비', ['정의', '공정'], ['자비', '용서', '관용']),
            ('안전 vs 자유', ['안전', '보안'], ['자유', '권리']),
            ('효율 vs 공정', ['효율', '성과'], ['공정', '평등']),
            ('전통 vs 진보', ['전통', '관습'], ['변화', '혁신', '진보'])
        ]
        
        conflicts = []
        text_lower = text.lower()
        
        for conflict_name, side1_keywords, side2_keywords in conflict_patterns:
            side1_count = sum(1 for kw in side1_keywords if kw in text_lower)
            side2_count = sum(1 for kw in side2_keywords if kw in text_lower)
            
            if side1_count > 0 and side2_count > 0:
                conflicts.append({
                    'conflict_type': conflict_name,
                    'side1_strength': side1_count,
                    'side2_strength': side2_count,
                    'intensity': (side1_count + side2_count) / (len(side1_keywords) + len(side2_keywords))
                })
                
        return conflicts
        
    def _identify_ethical_dilemmas(self, text: str) -> List[Dict[str, Any]]:
        """윤리적 딜레마 식별"""
        dilemma_indicators = [
            '딜레마', '갈등', '선택', '결정', '어려움',
            '고민', '망설임', '혼란', 'vs', '대립'
        ]
        
        dilemmas = []
        text_lower = text.lower()
        
        for indicator in dilemma_indicators:
            if indicator in text_lower:
                # 딜레마 컨텍스트 추출
                sentences = self._segment_sentences(text)
                for i, sentence in enumerate(sentences):
                    if indicator in sentence.lower():
                        dilemmas.append({
                            'indicator': indicator,
                            'context': sentence,
                            'position': i,
                            'confidence': 0.7  # 기본 신뢰도
                        })
                        
        return dilemmas
        
    def _korean_emotion_analysis(self, text: str) -> List[Dict[str, Any]]:
        """한국어 감정 분석"""
        korean_emotions = {
            'joy': ['기쁨', '즐거움', '행복', '만족', '기뻐', '좋아', '웃음'],
            'sadness': ['슬픔', '우울', '슬퍼', '눈물', '아픔', '상실'],
            'anger': ['분노', '화', '짜증', '분함', '억울', '격분'],
            'fear': ['두려움', '무서움', '걱정', '불안', '공포', '우려'],
            'surprise': ['놀라움', '놀라', '충격', '깜짝', '의외'],
            'disgust': ['혐오', '역겨움', '싫음', '불쾌', '거북'],
            'trust': ['신뢰', '믿음', '의지', '확신'],
            'anticipation': ['기대', '기다림', '희망', '예상']
        }
        
        results = []
        text_lower = text.lower()
        
        for emotion, keywords in korean_emotions.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                results.append({
                    'label': emotion,
                    'score': min(score / len(keywords), 1.0)
                })
                
        # 점수로 정렬
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
        
    def _analyze_emotion_intensity(self, text: str) -> float:
        """감정 강도 분석"""
        intensity_indicators = [
            ('매우', 2.0), ('정말', 1.8), ('너무', 1.8), ('극도로', 2.5),
            ('굉장히', 1.5), ('상당히', 1.3), ('꽤', 1.2), ('조금', 0.8),
            ('약간', 0.7), ('살짝', 0.6), ('!!!', 2.0), ('!!', 1.5), ('!', 1.2)
        ]
        
        base_intensity = 1.0
        text_lower = text.lower()
        
        for indicator, multiplier in intensity_indicators:
            if indicator in text_lower:
                base_intensity *= multiplier
                
        return min(base_intensity, 3.0) / 3.0  # 0-1 범위로 정규화
        
    def _analyze_emotion_progression(self, text: str) -> List[Dict[str, Any]]:
        """감정 진행 분석"""
        sentences = self._segment_sentences(text)
        progression = []
        
        for i, sentence in enumerate(sentences):
            emotions = self._korean_emotion_analysis(sentence)
            if emotions:
                progression.append({
                    'position': i,
                    'sentence': sentence,
                    'dominant_emotion': emotions[0]['label'],
                    'intensity': emotions[0]['score']
                })
                
        return progression
        
    def _analyze_complex_emotions(self, text: str) -> List[Dict[str, Any]]:
        """복합 감정 분석"""
        complex_patterns = {
            'bittersweet': (['기쁨', '행복'], ['슬픔', '아픔']),
            'mixed_feelings': (['좋아', '만족'], ['걱정', '불안']),
            'conflicted': (['원해', '바라'], ['싫어', '거부']),
            'ambivalent': (['사랑', '좋아'], ['미워', '싫어'])
        }
        
        results = []
        text_lower = text.lower()
        
        for pattern_name, (positive_words, negative_words) in complex_patterns.items():
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count > 0 and neg_count > 0:
                results.append({
                    'pattern': pattern_name,
                    'positive_strength': pos_count,
                    'negative_strength': neg_count,
                    'complexity_score': (pos_count + neg_count) / (len(positive_words) + len(negative_words))
                })
                
        return results
        
    def _calculate_valence_arousal(self, text: str, language: str) -> Dict[str, float]:
        """감정의 정서가(Valence)와 각성도(Arousal) 계산"""
        # 정서가 (긍정-부정)
        positive_words = ['좋아', '행복', '기쁨', '만족', '성공', '승리', '사랑']
        negative_words = ['나쁘', '슬픔', '분노', '실패', '좌절', '미워', '고통']
        
        # 각성도 (활성화-비활성화)  
        high_arousal_words = ['흥분', '열정', '분노', '공포', '놀라', '급한', '빠른']
        low_arousal_words = ['평온', '차분', '조용', '느린', '편안', '안정', '릴렉스']
        
        text_lower = text.lower()
        
        pos_score = sum(1 for word in positive_words if word in text_lower)
        neg_score = sum(1 for word in negative_words if word in text_lower)
        high_arousal_score = sum(1 for word in high_arousal_words if word in text_lower)
        low_arousal_score = sum(1 for word in low_arousal_words if word in text_lower)
        
        # 정규화된 점수 계산
        total_valence_words = len(positive_words) + len(negative_words)
        total_arousal_words = len(high_arousal_words) + len(low_arousal_words)
        
        valence = (pos_score - neg_score) / total_valence_words if total_valence_words > 0 else 0.0
        arousal = (high_arousal_score - low_arousal_score) / total_arousal_words if total_arousal_words > 0 else 0.0
        
        return {
            'valence': np.clip(valence, -1.0, 1.0),
            'arousal': np.clip(arousal, -1.0, 1.0)
        }
        
    def _extract_causal_relations(self, text: str) -> List[Dict[str, Any]]:
        """인과관계 추출"""
        causal_patterns = [
            (r'(.+?)(때문에|로 인해|인해서)(.+)', 'cause_effect'),
            (r'(.+?)(결과|결과적으로)(.+)', 'cause_effect'),
            (r'(.+?)(따라서|그래서|그러므로)(.+)', 'cause_effect'),
            (r'(.+?)(만약|만일)(.+?)(면|라면)(.+)', 'condition_result'),
            (r'(.+?)(원인은)(.+)', 'effect_cause'),
            (r'(.+?)(이유는)(.+)', 'effect_cause')
        ]
        
        relations = []
        
        for pattern, relation_type in causal_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                if relation_type == 'cause_effect' and len(groups) >= 3:
                    relations.append({
                        'type': 'causal',
                        'cause': groups[0].strip(),
                        'effect': groups[2].strip(),
                        'confidence': 0.8,
                        'position': match.start()
                    })
                elif relation_type == 'condition_result' and len(groups) >= 5:
                    relations.append({
                        'type': 'conditional',
                        'condition': groups[2].strip(),
                        'result': groups[4].strip(),
                        'confidence': 0.7,
                        'position': match.start()
                    })
                elif relation_type == 'effect_cause' and len(groups) >= 3:
                    relations.append({
                        'type': 'causal',
                        'cause': groups[2].strip(),
                        'effect': groups[0].strip(),
                        'confidence': 0.8,
                        'position': match.start()
                    })
                    
        return relations
        
    def _analyze_conditional_relations(self, text: str) -> List[Dict[str, Any]]:
        """조건부 관계 분석"""
        conditional_patterns = [
            r'(.+?)(만약|만일|혹시)(.+?)(면|라면|다면)(.+)',
            r'(.+?)(경우에는|때는)(.+)',
            r'(.+?)(조건으로|전제로)(.+)'
        ]
        
        relations = []
        
        for pattern in conditional_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                if len(groups) >= 3:
                    relations.append({
                        'condition': groups[2].strip() if len(groups) > 2 else groups[1].strip(),
                        'consequence': groups[-1].strip(),
                        'confidence': 0.7,
                        'type': 'conditional'
                    })
                    
        return relations
        
    def _analyze_temporal_sequence(self, text: str) -> List[Dict[str, Any]]:
        """시간적 순서 분석"""
        temporal_markers = [
            ('먼저', 1), ('처음에', 1), ('첫째', 1),
            ('그다음', 2), ('다음에', 2), ('둘째', 2),
            ('마지막으로', 3), ('결국', 3), ('최종적으로', 3),
            ('이전에', 0), ('나중에', 4), ('이후에', 4)
        ]
        
        sequences = []
        sentences = self._segment_sentences(text)
        
        for i, sentence in enumerate(sentences):
            for marker, order in temporal_markers:
                if marker in sentence:
                    sequences.append({
                        'sentence': sentence,
                        'position': i,
                        'temporal_order': order,
                        'marker': marker
                    })
                    
        # 시간 순서로 정렬
        sequences.sort(key=lambda x: x['temporal_order'])
        return sequences
        
    def _predict_consequences(self, text: str) -> List[Dict[str, Any]]:
        """결과 예측"""
        # 간단한 규칙 기반 결과 예측
        consequence_indicators = [
            ('위험', ['사고', '손상', '피해', '문제']),
            ('노력', ['성공', '발전', '개선', '달성']),
            ('갈등', ['분열', '대립', '문제', '해결']),
            ('협력', ['성과', '발전', '성공', '화합'])
        ]
        
        predictions = []
        text_lower = text.lower()
        
        for cause, possible_effects in consequence_indicators:
            if cause in text_lower:
                for effect in possible_effects:
                    predictions.append({
                        'cause': cause,
                        'predicted_effect': effect,
                        'probability': 0.6,  # 기본 확률
                        'reasoning': f"'{cause}' 패턴 기반 예측"
                    })
                    
        return predictions
        
    def _build_causal_network(self, causal_relations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """인과 네트워크 구성"""
        try:
            G = nx.DiGraph()
            
            for relation in causal_relations:
                if relation['type'] == 'causal':
                    cause = relation['cause'][:50]  # 길이 제한
                    effect = relation['effect'][:50]
                    confidence = relation['confidence']
                    
                    G.add_edge(cause, effect, weight=confidence)
                    
            # 네트워크 분석
            network_info = {
                'nodes': list(G.nodes()),
                'edges': list(G.edges(data=True)),
                'node_count': G.number_of_nodes(),
                'edge_count': G.number_of_edges(),
                'density': nx.density(G),
                'central_nodes': []
            }
            
            # 중심성 계산
            if G.number_of_nodes() > 0:
                centrality = nx.degree_centrality(G)
                sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
                network_info['central_nodes'] = sorted_centrality[:5]
                
            return network_info
            
        except Exception as e:
            logger.error(f"인과 네트워크 구성 실패: {e}")
            return {'nodes': [], 'edges': [], 'node_count': 0, 'edge_count': 0}


class AdvancedSemanticAnalyzer:
    """고급 의미 분석 시스템 - Linux 전용 AI 강화 버전"""
    
    def __init__(self):
        if not ADVANCED_LIBS_AVAILABLE:
            raise ImportError("고급 라이브러리가 필요합니다. requirements.txt를 확인하세요.")
            
        self.logger = logger
        self.device = DEVICE
        
        # 트랜스포머 처리기
        self.transformer_processor = AdvancedTransformerProcessor()
        
        # 신경망 의미 인코더
        self.neural_encoder = NeuralSemanticEncoder().to(self.device)
        
        # 모델 및 캐시 디렉토리 설정
        self.model_dir = os.path.join(MODELS_DIR, 'semantic_models')
        self.cache_dir = os.path.join(MODELS_DIR, 'semantic_cache')
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 고급 설정
        self.advanced_config = {
            'use_neural_encoding': True,
            'use_clustering': True,
            'use_network_analysis': True,
            'batch_processing': True,
            'parallel_analysis': True,
            'cache_embeddings': True,
            'similarity_threshold': 0.7,
            'max_cache_size': 10000
        }
        
        # 임베딩 캐시 (고급)
        self.embedding_cache = {}
        self.cache_lock = threading.Lock()
        
        # 클러스터링 모델
        self.kmeans_model = None
        self.pca_model = None
        self.scaler = StandardScaler()
        
        # 의미론적 레벨별 가중치
        self.level_weights = {
            SemanticLevel.SURFACE: 0.3,
            SemanticLevel.ETHICAL: 0.25,
            SemanticLevel.EMOTIONAL: 0.25,
            SemanticLevel.CAUSAL: 0.2
        }
        
        # 스레드 풀
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        self.logger.info("고급 의미 분석 시스템 초기화 완료")
        
    def analyze_text_advanced(self, 
                            text: str, 
                            language: str = "ko",
                            analysis_depth: str = "full",
                            use_cache: bool = True) -> AdvancedSemanticResult:
        """고급 텍스트 의미 분석"""
        
        if not text or not text.strip():
            logger.warning("빈 텍스트가 입력되었습니다.")
            return AdvancedSemanticResult(text="", analysis_depth="none")
            
        start_time = time.time()
        
        # 캐시 확인
        cache_key = self._generate_cache_key(text, language, analysis_depth)
        if use_cache and cache_key in self.embedding_cache:
            self.logger.debug("캐시된 분석 결과 반환")
            return self.embedding_cache[cache_key]
            
        try:
            # 병렬 분석 실행
            if self.advanced_config['parallel_analysis']:
                analysis_results = self._parallel_analysis(text, language, analysis_depth)
            else:
                analysis_results = self._sequential_analysis(text, language, analysis_depth)
                
            # 신경망 인코딩
            if self.advanced_config['use_neural_encoding']:
                neural_features = self._neural_encode_text(text, language)
            else:
                neural_features = None
                
            # 특성 벡터 생성
            feature_vector = self._create_feature_vector(analysis_results, neural_features)
            
            # 클러스터링 (선택적)
            cluster_info = None
            if self.advanced_config['use_clustering']:
                cluster_info = self._perform_clustering(feature_vector)
                
            # 의미 네트워크 분석
            network_info = None
            if self.advanced_config['use_network_analysis']:
                network_info = self._analyze_semantic_network(analysis_results)
                
            # 종합 결과 생성
            result = AdvancedSemanticResult(
                text=text,
                language=language,
                analysis_depth=analysis_depth,
                surface_analysis=analysis_results.get('surface', {}),
                ethical_analysis=analysis_results.get('ethical', {}),
                emotional_analysis=analysis_results.get('emotional', {}),
                causal_analysis=analysis_results.get('causal', {}),
                feature_vector=feature_vector,
                neural_encoding=neural_features,
                cluster_info=cluster_info,
                network_info=network_info,
                confidence_score=self._calculate_overall_confidence(analysis_results),
                processing_time=time.time() - start_time,
                timestamp=time.time()
            )
            
            # 캐시 저장
            if use_cache:
                self._cache_result(cache_key, result)
                
            return result
            
        except Exception as e:
            self.logger.error(f"고급 의미 분석 실패: {e}")
            return self._fallback_analysis(text, language)
            
    def _parallel_analysis(self, text: str, language: str, depth: str) -> Dict[str, Any]:
        """병렬 분석 실행"""
        futures = {}
        
        # 각 레벨 분석을 병렬로 실행
        if depth in ['full', 'surface']:
            futures['surface'] = self.thread_pool.submit(
                self.transformer_processor.analyze_surface_level, text, language
            )
            
        if depth in ['full', 'ethical']:
            futures['ethical'] = self.thread_pool.submit(
                self.transformer_processor.analyze_ethical_level, text
            )
            
        if depth in ['full', 'emotional']:
            futures['emotional'] = self.thread_pool.submit(
                self.transformer_processor.analyze_emotional_level, text, language
            )
            
        if depth in ['full', 'causal']:
            futures['causal'] = self.thread_pool.submit(
                self.transformer_processor.analyze_causal_level, text
            )
            
        # 결과 수집
        results = {}
        for level, future in futures.items():
            try:
                results[level] = future.result(timeout=30)
            except Exception as e:
                self.logger.error(f"{level} 분석 실패: {e}")
                results[level] = {}
                
        return results
        
    def _sequential_analysis(self, text: str, language: str, depth: str) -> Dict[str, Any]:
        """순차 분석 실행"""
        results = {}
        
        try:
            if depth in ['full', 'surface']:
                results['surface'] = self.transformer_processor.analyze_surface_level(text, language)
                
            if depth in ['full', 'ethical']:
                results['ethical'] = self.transformer_processor.analyze_ethical_level(text)
                
            if depth in ['full', 'emotional']:
                results['emotional'] = self.transformer_processor.analyze_emotional_level(text, language)
                
            if depth in ['full', 'causal']:
                results['causal'] = self.transformer_processor.analyze_causal_level(text)
                
        except Exception as e:
            self.logger.error(f"순차 분석 실패: {e}")
            
        return results
        
    def _neural_encode_text(self, text: str, language: str) -> Dict[str, np.ndarray]:
        """신경망 기반 텍스트 인코딩"""
        try:
            # 기본 임베딩 생성
            if language == "ko":
                embedding = self.transformer_processor.korean_model.encode(text)
            else:
                embedding = self.transformer_processor.semantic_model.encode(text)
                
            # 임베딩 차원 확인 및 조정
            if isinstance(embedding, (int, float)):
                # 스칼라 값인 경우 기본 벡터로 확장
                self.logger.warning("임베딩이 스칼라 값입니다. 기본 벡터로 확장합니다.")
                embedding = np.full(768, float(embedding))
            elif hasattr(embedding, 'shape') and len(embedding.shape) == 0:
                # 0차원 배열인 경우
                self.logger.warning("임베딩이 0차원 배열입니다. 기본 벡터로 확장합니다.")
                embedding = np.full(768, float(embedding))
            elif hasattr(embedding, 'shape') and embedding.shape[0] < 256:
                # 차원이 너무 작은 경우 패딩
                expected_dim = 768
                if embedding.shape[0] < expected_dim:
                    padding_size = expected_dim - embedding.shape[0]
                    embedding = np.pad(embedding, (0, padding_size), mode='constant', constant_values=0)
                    self.logger.info(f"임베딩 차원을 {embedding.shape[0]}에서 {expected_dim}으로 확장했습니다.")
                
            # 신경망 인코더 적용
            with torch.no_grad():
                input_tensor = torch.tensor(embedding, dtype=TORCH_DTYPE).to(self.device)
                # 배치 차원 추가 (1, embedding_dim)
                if len(input_tensor.shape) == 1:
                    input_tensor = input_tensor.unsqueeze(0)
                neural_features = self.neural_encoder(input_tensor)
                
            # NumPy 배열로 변환
            encoded_features = {}
            for level, features in neural_features.items():
                encoded_features[level] = features.cpu().numpy()
                
            return encoded_features
            
        except Exception as e:
            self.logger.error(f"신경망 인코딩 실패: {e}")
            # 기본 더미 특성 반환
            return {
                'surface': np.zeros((1, 64)),
                'ethical': np.zeros((1, 64)),
                'emotional': np.zeros((1, 64)),
                'causal': np.zeros((1, 64))
            }
            
    def _create_feature_vector(self, 
                             analysis_results: Dict[str, Any],
                             neural_features: Dict[str, np.ndarray] = None) -> SemanticFeatureVector:
        """종합 특성 벡터 생성"""
        
        # 각 레벨별 특성 추출
        surface_features = self._extract_surface_features(analysis_results.get('surface', {}))
        ethical_features = self._extract_ethical_features(analysis_results.get('ethical', {}))
        emotional_features = self._extract_emotional_features(analysis_results.get('emotional', {}))
        causal_features = self._extract_causal_features(analysis_results.get('causal', {}))
        
        # 통합 특성 벡터 생성
        integrated_features = np.concatenate([
            surface_features, ethical_features, 
            emotional_features, causal_features
        ])
        
        # 신경망 특성 추가 (가능한 경우)
        if neural_features:
            neural_integrated = neural_features.get('integrated', np.array([]))
            if neural_integrated.size > 0:
                integrated_features = np.concatenate([integrated_features, neural_integrated])
                
        # 특성 이름 생성
        feature_names = (
            [f"surface_{i}" for i in range(len(surface_features))] +
            [f"ethical_{i}" for i in range(len(ethical_features))] +
            [f"emotional_{i}" for i in range(len(emotional_features))] +
            [f"causal_{i}" for i in range(len(causal_features))]
        )
        
        # 신뢰도 점수 계산
        confidence_scores = {
            'surface': self._calculate_level_confidence(analysis_results.get('surface', {})),
            'ethical': self._calculate_level_confidence(analysis_results.get('ethical', {})),
            'emotional': self._calculate_level_confidence(analysis_results.get('emotional', {})),
            'causal': self._calculate_level_confidence(analysis_results.get('causal', {}))
        }
        
        return SemanticFeatureVector(
            surface_features=surface_features,
            ethical_features=ethical_features,
            emotional_features=emotional_features,
            causal_features=causal_features,
            integrated_features=integrated_features,
            feature_names=feature_names,
            confidence_scores=confidence_scores
        )
        
    def _extract_surface_features(self, surface_analysis: Dict[str, Any]) -> np.ndarray:
        """표면적 특성 추출"""
        features = []
        
        # 개체 수
        entities = surface_analysis.get('entities', [])
        features.append(len(entities))
        
        # 키워드 수
        keywords = surface_analysis.get('keywords', [])
        features.append(len(keywords))
        
        # 문장 수
        sentences = surface_analysis.get('sentences', [])
        features.append(len(sentences))
        
        # 평균 문장 중요도
        if sentences:
            avg_importance = np.mean([importance for _, importance in sentences])
            features.append(avg_importance)
        else:
            features.append(0.0)
            
        # 구문 복잡도
        syntax = surface_analysis.get('syntax', [])
        features.append(len(syntax))
        
        # 고정 길이로 패딩/트러케이팅
        target_length = 10
        if len(features) < target_length:
            features.extend([0.0] * (target_length - len(features)))
        else:
            features = features[:target_length]
            
        return np.array(features, dtype=np.float32)
        
    def _extract_ethical_features(self, ethical_analysis: Dict[str, Any]) -> np.ndarray:
        """윤리적 특성 추출"""
        features = []
        
        # 윤리적 카테고리 점수
        categories = ethical_analysis.get('ethical_categories', {})
        if isinstance(categories, dict) and 'scores' in categories:
            category_scores = categories['scores']
            features.extend(category_scores[:8])  # 최대 8개 카테고리
        else:
            features.extend([0.0] * 8)
            
        # 도덕적 감정 강도
        moral_emotions = ethical_analysis.get('moral_emotions', [])
        if moral_emotions:
            avg_intensity = np.mean([emotion['intensity'] for emotion in moral_emotions])
            features.append(avg_intensity)
        else:
            features.append(0.0)
            
        # 가치 충돌 수
        value_conflicts = ethical_analysis.get('value_conflicts', [])
        features.append(len(value_conflicts))
        
        # 윤리적 딜레마 수
        ethical_dilemmas = ethical_analysis.get('ethical_dilemmas', [])
        features.append(len(ethical_dilemmas))
        
        # 고정 길이로 조정
        target_length = 12
        if len(features) < target_length:
            features.extend([0.0] * (target_length - len(features)))
        else:
            features = features[:target_length]
            
        return np.array(features, dtype=np.float32)
        
    def _extract_emotional_features(self, emotional_analysis: Dict[str, Any]) -> np.ndarray:
        """감정적 특성 추출"""
        features = []
        
        # 기본 감정 점수
        basic_emotions = emotional_analysis.get('basic_emotions', [])
        emotion_scores = [0.0] * 8  # 8가지 기본 감정
        
        for i, emotion in enumerate(basic_emotions[:8]):
            if isinstance(emotion, dict) and 'score' in emotion:
                emotion_scores[i] = emotion['score']
                
        features.extend(emotion_scores)
        
        # 감정 강도
        intensity = emotional_analysis.get('emotion_intensity', 0.0)
        features.append(intensity)
        
        # 복합 감정 수
        complex_emotions = emotional_analysis.get('complex_emotions', [])
        features.append(len(complex_emotions))
        
        # 정서가와 각성도
        valence_arousal = emotional_analysis.get('valence_arousal', {'valence': 0.0, 'arousal': 0.0})
        features.append(valence_arousal['valence'])
        features.append(valence_arousal['arousal'])
        
        # 고정 길이로 조정
        target_length = 12
        if len(features) < target_length:
            features.extend([0.0] * (target_length - len(features)))
        else:
            features = features[:target_length]
            
        return np.array(features, dtype=np.float32)
        
    def _extract_causal_features(self, causal_analysis: Dict[str, Any]) -> np.ndarray:
        """인과적 특성 추출"""
        features = []
        
        # 인과관계 수
        causal_relations = causal_analysis.get('causal_relations', [])
        features.append(len(causal_relations))
        
        # 조건부 관계 수
        conditional_relations = causal_analysis.get('conditional_relations', [])
        features.append(len(conditional_relations))
        
        # 시간적 순서 수
        temporal_sequence = causal_analysis.get('temporal_sequence', [])
        features.append(len(temporal_sequence))
        
        # 결과 예측 수
        predictions = causal_analysis.get('consequence_predictions', [])
        features.append(len(predictions))
        
        # 네트워크 밀도
        network = causal_analysis.get('causal_network', {})
        density = network.get('density', 0.0)
        features.append(density)
        
        # 중심 노드 수
        central_nodes = network.get('central_nodes', [])
        features.append(len(central_nodes))
        
        # 평균 신뢰도
        if causal_relations:
            avg_confidence = np.mean([rel.get('confidence', 0.0) for rel in causal_relations])
            features.append(avg_confidence)
        else:
            features.append(0.0)
            
        # 고정 길이로 조정
        target_length = 10
        if len(features) < target_length:
            features.extend([0.0] * (target_length - len(features)))
        else:
            features = features[:target_length]
            
        return np.array(features, dtype=np.float32)
        
    def _perform_clustering(self, feature_vector: SemanticFeatureVector) -> Dict[str, Any]:
        """클러스터링 수행"""
        try:
            # 특성 정규화
            features = feature_vector.integrated_features.reshape(1, -1)
            features_scaled = self.scaler.fit_transform(features)
            
            # K-means 클러스터링 (단일 샘플이므로 더미 클러스터링)
            # 실제로는 여러 텍스트가 축적된 후에 의미있는 클러스터링 수행
            cluster_info = {
                'cluster_id': 0,
                'cluster_center': features_scaled[0].tolist(),
                'distance_to_center': 0.0,
                'cluster_size': 1,
                'cluster_coherence': 1.0
            }
            
            return cluster_info
            
        except Exception as e:
            self.logger.error(f"클러스터링 실패: {e}")
            return {}
            
    def _analyze_semantic_network(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """의미 네트워크 분석"""
        try:
            # 각 레벨의 키 개념들 추출
            concepts = set()
            
            # 표면적 레벨에서 키워드 추출
            surface = analysis_results.get('surface', {})
            keywords = surface.get('keywords', [])
            concepts.update(keywords)
            
            # 윤리적 레벨에서 가치 추출
            ethical = analysis_results.get('ethical', {})
            categories = ethical.get('ethical_categories', {})
            if isinstance(categories, dict) and 'labels' in categories:
                concepts.update(categories['labels'][:3])  # 상위 3개 라벨
                
            # 감정적 레벨에서 감정 추출
            emotional = analysis_results.get('emotional', {})
            emotions = emotional.get('basic_emotions', [])
            for emotion in emotions[:3]:  # 상위 3개 감정
                if isinstance(emotion, dict) and 'label' in emotion:
                    concepts.add(emotion['label'])
                    
            # 인과적 레벨에서 관계 추출
            causal = analysis_results.get('causal', {})
            relations = causal.get('causal_relations', [])
            
            # 네트워크 구성
            G = nx.Graph()
            
            # 개념 노드 추가
            for concept in concepts:
                G.add_node(concept)
                
            # 관계 엣지 추가
            for relation in relations:
                if 'cause' in relation and 'effect' in relation:
                    cause = relation['cause'][:20]  # 길이 제한
                    effect = relation['effect'][:20]
                    if cause in concepts or effect in concepts:
                        G.add_edge(cause, effect, weight=relation.get('confidence', 0.5))
                        
            # 네트워크 분석
            network_info = {
                'node_count': G.number_of_nodes(),
                'edge_count': G.number_of_edges(),
                'density': nx.density(G),
                'clustering_coefficient': nx.average_clustering(G),
                'diameter': 0,
                'central_concepts': []
            }
            
            # 연결된 그래프인 경우 직경 계산
            if nx.is_connected(G) and G.number_of_nodes() > 1:
                network_info['diameter'] = nx.diameter(G)
                
            # 중심성 계산
            if G.number_of_nodes() > 0:
                centrality = nx.degree_centrality(G)
                sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
                network_info['central_concepts'] = sorted_centrality[:5]
                
            return network_info
            
        except Exception as e:
            self.logger.error(f"의미 네트워크 분석 실패: {e}")
            return {}
            
    def _calculate_level_confidence(self, level_analysis: Dict[str, Any]) -> float:
        """레벨별 신뢰도 계산"""
        if not level_analysis:
            return 0.0
            
        confidence_factors = []
        
        # 분석 결과의 완전성
        non_empty_results = sum(1 for v in level_analysis.values() if v)
        total_results = len(level_analysis)
        completeness = non_empty_results / total_results if total_results > 0 else 0.0
        confidence_factors.append(completeness)
        
        # 개별 결과의 신뢰도
        for key, value in level_analysis.items():
            if isinstance(value, list) and value:
                # 리스트 항목들의 신뢰도
                confidences = [item.get('confidence', 0.5) for item in value if isinstance(item, dict)]
                if confidences:
                    confidence_factors.append(np.mean(confidences))
            elif isinstance(value, dict) and 'confidence' in value:
                confidence_factors.append(value['confidence'])
                
        return float(np.mean(confidence_factors)) if confidence_factors else 0.5
        
    def _calculate_overall_confidence(self, analysis_results: Dict[str, Any]) -> float:
        """전체 신뢰도 계산"""
        level_confidences = []
        
        for level, weight in self.level_weights.items():
            level_name = level.name.lower()
            if level_name in analysis_results:
                level_conf = self._calculate_level_confidence(analysis_results[level_name])
                level_confidences.append(level_conf * weight)
                
        return float(np.sum(level_confidences)) if level_confidences else 0.5
        
    def _generate_cache_key(self, text: str, language: str, depth: str) -> str:
        """캐시 키 생성"""
        import hashlib
        
        key_data = f"{text[:100]}_{language}_{depth}"  # 텍스트 일부만 사용
        return hashlib.md5(key_data.encode()).hexdigest()
        
    def _cache_result(self, cache_key: str, result: AdvancedSemanticResult):
        """결과 캐싱"""
        with self.cache_lock:
            if len(self.embedding_cache) >= self.advanced_config['max_cache_size']:
                # 가장 오래된 항목 제거
                oldest_key = next(iter(self.embedding_cache))
                del self.embedding_cache[oldest_key]
                
            self.embedding_cache[cache_key] = result
            
    def _fallback_analysis(self, text: str, language: str) -> AdvancedSemanticResult:
        """대체 분석 (오류 시)"""
        return AdvancedSemanticResult(
            text=text,
            language=language,
            analysis_depth="fallback",
            surface_analysis={'keywords': text.split()[:5], 'sentences': [(text, 1.0)]},
            ethical_analysis={},
            emotional_analysis={'basic_emotions': [{'label': 'neutral', 'score': 0.5}]},
            causal_analysis={},
            confidence_score=0.3,
            processing_time=0.0,
            timestamp=time.time()
        )
        
    def calculate_advanced_similarity(self, 
                                   result1: AdvancedSemanticResult,
                                   result2: AdvancedSemanticResult) -> Dict[str, float]:
        """고급 유사도 계산"""
        similarities = {}
        
        try:
            # 특성 벡터 기반 유사도
            if result1.feature_vector and result2.feature_vector:
                vec1 = result1.feature_vector.integrated_features
                vec2 = result2.feature_vector.integrated_features
                
                if vec1.size > 0 and vec2.size > 0:
                    # 벡터 길이 맞춤
                    min_len = min(len(vec1), len(vec2))
                    vec1_truncated = vec1[:min_len]
                    vec2_truncated = vec2[:min_len]
                    
                    # 코사인 유사도
                    cosine_sim = np.dot(vec1_truncated, vec2_truncated) / (
                        np.linalg.norm(vec1_truncated) * np.linalg.norm(vec2_truncated)
                    )
                    similarities['feature_similarity'] = float(cosine_sim)
                    
            # 레벨별 유사도
            for level in ['surface', 'ethical', 'emotional', 'causal']:
                level_sim = self._calculate_level_similarity(
                    getattr(result1, f'{level}_analysis', {}),
                    getattr(result2, f'{level}_analysis', {})
                )
                similarities[f'{level}_similarity'] = level_sim
                
            # 가중 평균 전체 유사도
            weighted_sim = sum(
                similarities[f'{level}_similarity'] * self.level_weights[SemanticLevel[level.upper()]]
                for level in ['surface', 'ethical', 'emotional', 'causal']
                if f'{level}_similarity' in similarities
            )
            similarities['overall_similarity'] = weighted_sim
            
        except Exception as e:
            self.logger.error(f"고급 유사도 계산 실패: {e}")
            similarities = {'overall_similarity': 0.0}
            
        return similarities
        
    def _calculate_level_similarity(self, analysis1: Dict[str, Any], analysis2: Dict[str, Any]) -> float:
        """레벨별 유사도 계산"""
        if not analysis1 or not analysis2:
            return 0.0
            
        try:
            # 공통 키워드/개념 기반 유사도
            concepts1 = self._extract_concepts_from_analysis(analysis1)
            concepts2 = self._extract_concepts_from_analysis(analysis2)
            
            if not concepts1 or not concepts2:
                return 0.0
                
            # Jaccard 유사도
            intersection = len(concepts1.intersection(concepts2))
            union = len(concepts1.union(concepts2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"레벨 유사도 계산 실패: {e}")
            return 0.0
            
    def _extract_concepts_from_analysis(self, analysis: Dict[str, Any]) -> Set[str]:
        """분석 결과에서 개념 추출"""
        concepts = set()
        
        # 리스트 형태의 결과에서 개념 추출
        for key, value in analysis.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        # 라벨이나 키워드 추출
                        for field in ['label', 'keyword', 'word', 'entity', 'emotion']:
                            if field in item:
                                concepts.add(str(item[field]).lower())
                    elif isinstance(item, str):
                        concepts.add(item.lower())
            elif isinstance(value, str):
                concepts.add(value.lower())
                
        return concepts
        
    def clear_cache(self):
        """캐시 클리어"""
        with self.cache_lock:
            self.embedding_cache.clear()
            
    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계"""
        with self.cache_lock:
            return {
                'cache_size': len(self.embedding_cache),
                'max_cache_size': self.advanced_config['max_cache_size'],
                'cache_hit_rate': 0.0  # 실제 구현에서는 히트율 추적
            }


def test_advanced_semantic_analyzer():
    """고급 의미 분석 시스템 테스트"""
    
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    try:
        # 분석기 초기화
        analyzer = AdvancedSemanticAnalyzer()
        
        # 테스트 텍스트
        test_texts = [
            "이 결정은 많은 사람들의 생명과 안전에 직접적인 영향을 미치며, 우리는 정의롭고 공정한 선택을 해야 합니다. 하지만 동시에 개인의 자유와 권리도 존중해야 하는 어려운 딜레마에 직면해 있습니다.",
            "새로운 기술의 도입으로 인해 많은 사람들이 일자리를 잃을 수 있습니다. 그러나 이 기술은 전체 사회의 발전과 효율성을 크게 향상시킬 것입니다. 우리는 어떤 선택을 해야 할까요?",
            "갑작스러운 자연재해로 인해 즉각적인 구조 작업이 필요합니다. 제한된 자원으로 모든 사람을 구할 수는 없지만, 최대한 많은 생명을 구하기 위해 노력해야 합니다."
        ]
        
        print("=== 고급 의미 분석 시스템 테스트 (Linux) ===\n")
        
        results = []
        
        for i, text in enumerate(test_texts, 1):
            print(f"📝 테스트 {i}: {text[:50]}...")
            
            # 분석 실행
            start_time = time.time()
            result = analyzer.analyze_text_advanced(
                text=text,
                language="ko",
                analysis_depth="full",
                use_cache=True
            )
            analysis_time = time.time() - start_time
            
            results.append(result)
            
            # 결과 출력
            print(f"   ⏱️ 처리 시간: {analysis_time:.3f}초")
            print(f"   🎯 신뢰도: {result.confidence_score:.3f}")
            print(f"   📊 특성 벡터 크기: {len(result.feature_vector.integrated_features) if result.feature_vector else 0}")
            
            # 표면적 분석
            surface = result.surface_analysis
            if surface:
                entities = surface.get('entities', [])
                keywords = surface.get('keywords', [])
                print(f"   🏷️ 개체 수: {len(entities)}, 키워드 수: {len(keywords)}")
                
            # 윤리적 분석
            ethical = result.ethical_analysis
            if ethical:
                categories = ethical.get('ethical_categories', {})
                conflicts = ethical.get('value_conflicts', [])
                print(f"   ⚖️ 윤리적 카테고리: {len(categories) if isinstance(categories, list) else 1}")
                print(f"   ⚡ 가치 충돌: {len(conflicts)}개")
                
            # 감정적 분석
            emotional = result.emotional_analysis
            if emotional:
                emotions = emotional.get('basic_emotions', [])
                intensity = emotional.get('emotion_intensity', 0.0)
                valence_arousal = emotional.get('valence_arousal', {})
                print(f"   😊 감정 수: {len(emotions)}, 강도: {intensity:.2f}")
                print(f"   💫 정서가: {valence_arousal.get('valence', 0):.2f}, 각성도: {valence_arousal.get('arousal', 0):.2f}")
                
            # 인과적 분석
            causal = result.causal_analysis
            if causal:
                relations = causal.get('causal_relations', [])
                network = causal.get('causal_network', {})
                print(f"   🔗 인과관계: {len(relations)}개")
                print(f"   🕸️ 네트워크 밀도: {network.get('density', 0):.3f}")
                
            print()
            
        # 유사도 분석
        if len(results) >= 2:
            print("🔍 유사도 분석:")
            for i in range(len(results)):
                for j in range(i+1, len(results)):
                    similarities = analyzer.calculate_advanced_similarity(results[i], results[j])
                    overall_sim = similarities.get('overall_similarity', 0.0)
                    print(f"   텍스트 {i+1} vs {j+1}: {overall_sim:.3f}")
                    
        # 시스템 정보
        print(f"\n🔧 시스템 정보:")
        print(f"   디바이스: {analyzer.device}")
        print(f"   GPU 사용: {'예' if ADVANCED_CONFIG['enable_gpu'] else '아니오'}")
        
        # 캐시 통계
        cache_stats = analyzer.get_cache_stats()
        print(f"   캐시 크기: {cache_stats['cache_size']}/{cache_stats['max_cache_size']}")
        
        return results
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    def get_pytorch_network(self) -> Optional[nn.Module]:
        """PyTorch 네트워크 반환 (HeadAdapter와의 호환성)"""
        # SentenceTransformer 모델은 PyTorch 모델이지만 직접 반환하지 않음
        # 대신 다른 PyTorch 모델이 있는지 확인
        
        for attr_name in ['model', 'neural_model', 'semantic_encoder', 'neural_encoder']:
            if hasattr(self, attr_name):
                model = getattr(self, attr_name)
                if isinstance(model, nn.Module):
                    self.logger.info(f"AdvancedSemanticAnalyzer: {attr_name} 반환")
                    return model
        
        # TransformerProcessor의 모델들도 확인
        if hasattr(self, 'transformer_processor'):
            transformer = self.transformer_processor
            # SentenceTransformer는 하위에 실제 PyTorch 모델이 있음
            if hasattr(transformer, 'semantic_model') and hasattr(transformer.semantic_model, 'modules'):
                # SentenceTransformer의 내부 모델 추출
                modules = transformer.semantic_model.modules()
                for module in modules:
                    if isinstance(module, nn.Module) and not isinstance(module, nn.ModuleList):
                        self.logger.info("AdvancedSemanticAnalyzer: SentenceTransformer 내부 모델 반환")
                        return module
        
        self.logger.warning("AdvancedSemanticAnalyzer: PyTorch 네트워크를 찾을 수 없음")
        return None


if __name__ == "__main__":
    test_advanced_semantic_analyzer()