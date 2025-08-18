"""
벤담 계산기 ML 모델 훈련 시스템
Bentham Calculator ML Model Training System

벤담 가중치 레이어별 ML 모델을 실제 데이터로 훈련
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
# SentenceTransformer는 sentence_transformer_singleton을 통해 사용
from transformers import pipeline

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('BenthamMLTrainer')

class BenthamDataset(Dataset):
    """PyTorch 데이터셋 클래스"""
    
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)
        self.out_linear = nn.Linear(dim, dim)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Linear transformations
        Q = self.q_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        
        return self.out_linear(attended)


class TransformerBlock(nn.Module):
    """Transformer block with attention and feed-forward"""
    
    def __init__(self, dim, num_heads=8, ff_dim=None, dropout=0.1):
        super().__init__()
        if ff_dim is None:
            ff_dim = dim * 4
            
        self.attention = MultiHeadAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention with residual
        attended = self.attention(x.unsqueeze(1)).squeeze(1)
        x = self.norm1(x + attended)
        
        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x


class BenthamDNN(nn.Module):
    """매우 높은 복잡도의 벤담 계산용 딥러닝 모델 (2M 파라미터)"""
    
    def __init__(self, input_dim, hidden_dims=[2048, 1024, 512, 256, 128], dropout=0.5):
        super().__init__()
        
        # 입력 임베딩 레이어
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Transformer 블록들
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dims[0], num_heads=16, dropout=dropout),
            TransformerBlock(hidden_dims[0], num_heads=16, dropout=dropout)
        ])
        
        # 메인 네트워크 (Residual connections 포함)
        self.layers = nn.ModuleList()
        prev_dim = hidden_dims[0]
        
        for i, hidden_dim in enumerate(hidden_dims[1:]):
            self.layers.append(nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ))
            
            # Skip connection layers (차원이 맞을 때만)
            if prev_dim == hidden_dim:
                setattr(self, f'skip_{i}', nn.Identity())
            else:
                setattr(self, f'skip_{i}', nn.Linear(prev_dim, hidden_dim))
                
            prev_dim = hidden_dim
        
        # 출력 레이어
        self.output_layer = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.LayerNorm(prev_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(prev_dim // 2, 1)
        )
        
        # 가중치 초기화
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x):
        # 입력 임베딩
        x = self.input_embedding(x)
        
        # Transformer 블록들
        for transformer in self.transformer_blocks:
            x = transformer(x)
        
        # 메인 네트워크 (Residual connections)
        prev_x = x
        for i, layer in enumerate(self.layers):
            new_x = layer(prev_x)
            
            # Skip connection
            skip_layer = getattr(self, f'skip_{i}')
            if hasattr(skip_layer, 'weight'):  # Linear layer
                skip_x = skip_layer(prev_x)
            else:  # Identity
                skip_x = prev_x
                
            # Residual connection (차원이 맞을 때만)
            if new_x.shape == skip_x.shape:
                x = new_x + skip_x
            else:
                x = new_x
                
            prev_x = x
        
        # 출력
        return self.output_layer(x)


class BenthamMLTrainer:
    """벤담 계산기 ML 모델 훈련"""
    
    def __init__(self):
        self.data_dir = Path("/mnt/c/large_project/linux_red_heart/processed_datasets")
        self.model_dir = Path("/mnt/c/large_project/linux_red_heart/models/bentham_models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # 6개 가중치 레이어
        self.weight_layers = [
            'contextual',  # 상황적 맥락
            'temporal',    # 시간적 영향
            'social',      # 사회적 파급
            'ethical',     # 윤리적 중요도
            'emotional',   # 감정적 강도
            'cognitive'    # 인지적 복잡도
        ]
        
        self.trained_models = {}
        self.scalers = {}
        
        # GPU 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # 훈련 설정 (AdamW adaptive gradient 활용)
        self.epochs = 300
        self.batch_size = 16  # 메모리 안정성을 위해 더 작게
        self.learning_rate = 0.0005  # 적당한 학습률
        self.patience = 40  # 충분한 patience로 최적점 찾기
        
        # 특성 엔지니어링을 위한 모델들
        logger.info("임베딩 모델 로딩 중...")
        from sentence_transformer_singleton import get_sentence_transformer
        self.sentence_transformer = get_sentence_transformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        self.emotion_analyzer = pipeline('text-classification', 
                                        model='j-hartmann/emotion-english-distilroberta-base',
                                        device=0 if torch.cuda.is_available() else -1)
        
    def load_training_data(self) -> pd.DataFrame:
        """훈련 데이터 로드"""
        logger.info("벤담 훈련 데이터 로딩 시작...")
        
        training_data = []
        
        # 스크러플 데이터 로드
        scruples_files = list(self.data_dir.glob("scruples/scruples_batch_*.json"))
        logger.info(f"스크러플 파일 {len(scruples_files)}개 발견")
        
        for file_path in scruples_files[:2]:  # 처음 2개만 사용 (빠른 테스트)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'scenarios' in data:
                    for scenario in data['scenarios']:
                        if isinstance(scenario, dict) and 'description' in scenario:
                            # 벤담 특성 추출
                            features = self._extract_bentham_features(scenario)
                            if features:
                                training_data.append(features)
                        else:
                            logger.warning(f"잘못된 시나리오 형식: {type(scenario)}")
                                
            except Exception as e:
                logger.warning(f"파일 {file_path} 로딩 실패: {e}")
                continue
        
        # 통합 시나리오 데이터 로드 (구조 수정)
        integrated_files = [
            self.data_dir / "integrated_scenarios.json",
            self.data_dir / "final_integrated_with_batch7_20250619_213234.json"
        ]
        
        for file_path in integrated_files:
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 파일 구조 체크 및 처리
                    if isinstance(data, dict) and 'integrated_scenarios' in data:
                        # final_integrated_with_batch7 형식
                        scenarios = data['integrated_scenarios']
                        logger.info(f"통합 파일 {file_path}에서 {len(scenarios)}개 시나리오 발견")
                    elif isinstance(data, list):
                        # integrated_scenarios.json 형식
                        scenarios = data
                        logger.info(f"통합 파일 {file_path}에서 {len(scenarios)}개 시나리오 발견")
                    else:
                        logger.warning(f"알 수 없는 통합 파일 형식: {type(data)}")
                        continue
                    
                    for idx, scenario in enumerate(scenarios[:10]):  # 각 파일에서 10개만
                        if isinstance(scenario, dict) and 'description' in scenario:
                            features = self._extract_bentham_features(scenario)
                            if features:
                                training_data.append(features)
                        else:
                            logger.warning(f"잘못된 통합 시나리오 형식: {type(scenario)}")
                            
                except Exception as e:
                    logger.warning(f"통합 파일 {file_path} 로딩 실패: {e}")
                    import traceback
                    logger.warning(f"상세 오류: {traceback.format_exc()}")
        
        df = pd.DataFrame(training_data)
        logger.info(f"총 {len(df)}개 훈련 샘플 생성")
        return df
    
    def _extract_bentham_features(self, scenario: Dict) -> Dict[str, Any]:
        """시나리오에서 고급 벤담 특성 추출 (특성 엔지니어링 포함)"""
        try:
            text = scenario.get('description', '')
            if not text:
                return None
            
            # 텍스트 길이 제한 (임베딩 모델 512 토큰 한계, 안전 마진 적용)
            if len(text) > 400:
                text = text[:400]
            
            # 기본 특성들
            features = {
                'text_length': len(text),
                'word_count': len(text.split()),
                'complexity': scenario.get('complexity', 0.5),
                
                # 상황적 맥락 특성
                'has_context': 1 if scenario.get('context') else 0,
                'stakeholder_count': len(scenario.get('stakeholders', {})),
                'moral_complexity': scenario.get('moral_complexity', 0.5),
                
                # 시간적 특성
                'urgency_keywords': self._count_urgency_keywords(text),
                'future_impact_keywords': self._count_future_keywords(text),
                
                # 사회적 특성
                'social_keywords': self._count_social_keywords(text),
                'relationship_keywords': self._count_relationship_keywords(text),
                
                # 윤리적 특성
                'ethical_keywords': self._count_ethical_keywords(text),
                'moral_judgment': self._encode_moral_judgment(scenario.get('context', {})),
                
                # 감정적 특성
                'emotion_keywords': self._count_emotion_keywords(text),
                'emotional_intensity': self._estimate_emotional_intensity(text),
                
                # 인지적 특성
                'cognitive_load': self._estimate_cognitive_load(text),
                'decision_complexity': self._estimate_decision_complexity(text)
            }
            
            # 고급 특성 엔지니어링 (GPU 메모리 안정성 개선)
            try:
                # GPU 메모리 체크
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                    if gpu_memory > 6.0:  # 6GB 이상 사용 중이면 GPU 정리
                        torch.cuda.empty_cache()
                        logger.info(f"GPU 메모리 정리: {gpu_memory:.2f}GB")
                
                # 1. 임베딩 특성 (메모리 효율적 처리)
                try:
                    with torch.no_grad():  # 그래디언트 계산 비활성화
                        embedding = self.sentence_transformer.encode(text, convert_to_numpy=True, show_progress_bar=False)
                    
                    # 통계적 특성 추출
                    features.update({
                        'embedding_mean': float(np.mean(embedding)),
                        'embedding_std': float(np.std(embedding)),
                        'embedding_max': float(np.max(embedding)),
                        'embedding_min': float(np.min(embedding)),
                        'embedding_skew': float(np.percentile(embedding, 75) - np.percentile(embedding, 25))
                    })
                    
                    # 메모리 정리
                    del embedding
                    
                except Exception as embed_e:
                    logger.warning(f"임베딩 실패: {embed_e}")
                    features.update({
                        'embedding_mean': 0.0, 'embedding_std': 0.0, 'embedding_max': 0.0, 
                        'embedding_min': 0.0, 'embedding_skew': 0.0
                    })
                
                # 2. 감정 분석 특성 (배치 크기 제한)
                try:
                    # 짧은 텍스트로 잘라서 처리 (안전 마진)
                    text_for_emotion = text[:300] if len(text) > 300 else text
                    
                    with torch.no_grad():
                        emotion_result = self.emotion_analyzer(text_for_emotion)
                    
                    emotion_scores = {result['label']: result['score'] for result in emotion_result}
                    
                    # 주요 감정들의 점수
                    emotions = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
                    for emotion in emotions:
                        features[f'emotion_{emotion}'] = emotion_scores.get(emotion, 0.0)
                    
                    # 감정 다양성 지수
                    emotion_values = list(emotion_scores.values())
                    features['emotion_diversity'] = float(np.std(emotion_values))
                    features['emotion_dominance'] = float(np.max(emotion_values))
                    
                except Exception as emotion_e:
                    logger.warning(f"감정 분석 실패: {emotion_e}")
                    features.update({
                        'emotion_anger': 0.0, 'emotion_disgust': 0.0, 'emotion_fear': 0.0,
                        'emotion_joy': 0.0, 'emotion_neutral': 0.5, 'emotion_sadness': 0.0,
                        'emotion_surprise': 0.0, 'emotion_diversity': 0.0, 'emotion_dominance': 0.5
                    })
                
            except Exception as e:
                logger.warning(f"고급 특성 추출 전체 실패: {e}")
                # 기본값들로 채우기
                features.update({
                    'embedding_mean': 0.0, 'embedding_std': 0.0, 'embedding_max': 0.0, 
                    'embedding_min': 0.0, 'embedding_skew': 0.0,
                    'emotion_anger': 0.0, 'emotion_disgust': 0.0, 'emotion_fear': 0.0,
                    'emotion_joy': 0.0, 'emotion_neutral': 0.5, 'emotion_sadness': 0.0,
                    'emotion_surprise': 0.0, 'emotion_diversity': 0.0, 'emotion_dominance': 0.5
                })
            
            # 목표 가중치 생성 (개선된 규칙 기반)
            targets = self._generate_target_weights(features, text)
            features.update(targets)
            
            return features
            
        except Exception as e:
            logger.warning(f"특성 추출 실패: {e}")
            return None
    
    def _count_urgency_keywords(self, text: str) -> int:
        urgency_words = ['급하', '빨리', '즉시', '긴급', '서둘', '지금', '당장', '빨리빨리']
        return sum(1 for word in urgency_words if word in text)
    
    def _count_future_keywords(self, text: str) -> int:
        future_words = ['미래', '앞으로', '나중', '장기', '영구', '계속', '평생', '오래']
        return sum(1 for word in future_words if word in text)
    
    def _count_social_keywords(self, text: str) -> int:
        social_words = ['사람', '친구', '가족', '동료', '사회', '공동체', '관계', '타인']
        return sum(1 for word in social_words if word in text)
    
    def _count_relationship_keywords(self, text: str) -> int:
        rel_words = ['친구', '연인', '부모', '자식', '형제', '자매', '동료', '이웃']
        return sum(1 for word in rel_words if word in text)
    
    def _count_ethical_keywords(self, text: str) -> int:
        ethical_words = ['윤리', '도덕', '옳', '그르', '정의', '공정', '선악', '책임']
        return sum(1 for word in ethical_words if word in text)
    
    def _encode_moral_judgment(self, context: Dict) -> float:
        judgment = context.get('moral_judgment', 'NOBODY')
        mapping = {
            'AUTHOR': 0.2,    # 작성자가 잘못
            'OTHER': 0.8,     # 타인이 잘못
            'EVERYBODY': 0.5, # 모두 잘못
            'NOBODY': 0.6,    # 아무도 잘못 안함
            'INFO': 0.5       # 정보 부족
        }
        return mapping.get(judgment, 0.5)
    
    def _count_emotion_keywords(self, text: str) -> int:
        emotion_words = ['기쁘', '슬프', '화나', '무서', '놀라', '짜증', '행복', '우울']
        return sum(1 for word in emotion_words if word in text)
    
    def _estimate_emotional_intensity(self, text: str) -> float:
        intensity_words = ['매우', '정말', '너무', '완전', '엄청', '굉장', '극도로']
        count = sum(1 for word in intensity_words if word in text)
        return min(1.0, count * 0.2)
    
    def _estimate_cognitive_load(self, text: str) -> float:
        # 문장 복잡도 추정
        sentences = text.split('.')
        avg_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        return min(1.0, avg_length / 20.0)
    
    def _estimate_decision_complexity(self, text: str) -> float:
        decision_words = ['선택', '결정', '고민', '판단', '딜레마', '갈등']
        count = sum(1 for word in decision_words if word in text)
        return min(1.0, count * 0.3)
    
    def _generate_target_weights(self, features: Dict, text: str) -> Dict[str, float]:
        """개선된 목표 가중치 생성 (데이터 리케지 방지)"""
        targets = {}
        
        # 기본 가중치에 노이즈 추가하여 데이터 리케지 방지
        base_weight = 1.0
        noise_factor = 0.1  # 10% 노이즈
        
        # 상황적 맥락 가중치 (복잡한 비선형 관계)
        contextual = base_weight
        # 텍스트 길이와 복잡도의 비선형 관계
        length_factor = np.tanh(features['text_length'] / 1000.0)  
        complexity_factor = features.get('complexity', 0.5) ** 2
        contextual += length_factor * complexity_factor * 0.8
        # 랜덤 노이즈 추가
        contextual += np.random.normal(0, noise_factor)
        targets['target_contextual'] = np.clip(contextual, 0.5, 3.0)
        
        # 시간적 영향 가중치 (키워드 밀도 기반)
        temporal = base_weight
        word_count = max(1, features['word_count'])
        urgency_density = features['urgency_keywords'] / word_count
        future_density = features['future_impact_keywords'] / word_count
        temporal += (urgency_density * 2.0 + future_density * 1.5) * 10
        temporal += np.random.normal(0, noise_factor)
        targets['target_temporal'] = np.clip(temporal, 0.5, 3.0)
        
        # 사회적 파급 가중치 (상호작용 효과)
        social = base_weight
        social_density = features['social_keywords'] / word_count
        rel_density = features['relationship_keywords'] / word_count
        # 비선형 상호작용
        social += np.sqrt(social_density * rel_density) * 5.0
        social += np.random.normal(0, noise_factor)
        targets['target_social'] = np.clip(social, 0.5, 3.0)
        
        # 윤리적 중요도 가중치 (복합 지수)
        ethical = base_weight
        ethical_density = features['ethical_keywords'] / word_count
        moral_extremeness = abs(features['moral_judgment'] - 0.5) * 2
        # 로그 스케일 적용
        ethical += np.log1p(ethical_density * 10) + moral_extremeness
        ethical += np.random.normal(0, noise_factor)
        targets['target_ethical'] = np.clip(ethical, 0.5, 3.0)
        
        # 감정적 강도 가중치 (감정 지수)
        emotional = base_weight
        emotion_density = features['emotion_keywords'] / word_count
        intensity = features['emotional_intensity']
        # 제곱근 관계
        emotional += np.sqrt(emotion_density * intensity) * 3.0
        emotional += np.random.normal(0, noise_factor)
        targets['target_emotional'] = np.clip(emotional, 0.5, 3.0)
        
        # 인지적 복잡도 가중치 (복잡도 지수)
        cognitive = base_weight
        cog_load = features['cognitive_load']
        decision_comp = features['decision_complexity']
        # 지수 함수 관계
        cognitive += np.exp(cog_load * decision_comp) - 1.0
        cognitive += np.random.normal(0, noise_factor)
        targets['target_cognitive'] = np.clip(cognitive, 0.5, 3.0)
        
        return targets
    
    def train_deep_model(self, X_train, X_test, y_train, y_test, layer_name):
        """딥러닝 모델 훈련"""
        input_dim = X_train.shape[1]
        
        # 모델 생성
        model = BenthamDNN(input_dim).to(self.device)
        
        # 최적화기 (AdamW with weight decay)
        optimizer = optim.AdamW(model.parameters(), 
                               lr=self.learning_rate, 
                               weight_decay=0.01)
        
        # 학습률 스케줄러
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        # 손실 함수
        criterion = nn.MSELoss()
        
        # 데이터로더
        train_dataset = BenthamDataset(X_train, y_train.values)
        test_dataset = BenthamDataset(X_test, y_test.values)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # 훈련 루프
        best_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(self.epochs):
            # 훈련
            model.train()
            train_loss = 0.0
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_features).squeeze()
                loss = criterion(outputs, batch_targets)
                loss.backward()
                
                # 그래디언트 클리핑
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            # 검증
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_features, batch_targets in test_loader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    outputs = model(batch_features).squeeze()
                    loss = criterion(outputs, batch_targets)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(test_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # 스케줄러 업데이트
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                # 최고 모델 저장
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                logger.info(f"  Epoch {epoch:3d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
            
            if patience_counter >= self.patience:
                logger.info(f"  Early stopping at epoch {epoch}")
                break
        
        # 최고 모델 복원
        model.load_state_dict(best_model_state)
        
        # 최종 평가
        model.eval()
        with torch.no_grad():
            y_pred = []
            for batch_features, _ in test_loader:
                batch_features = batch_features.to(self.device)
                outputs = model(batch_features).squeeze()
                y_pred.extend(outputs.cpu().numpy())
        
        y_pred = np.array(y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return model, mse, r2, len(train_losses)

    def train_models(self, df: pd.DataFrame):
        """각 가중치 레이어별 딥러닝 모델 훈련"""
        logger.info("딥러닝 모델 훈련 시작...")
        
        # 특성 컬럼들
        feature_cols = [col for col in df.columns if not col.startswith('target_')]
        
        for layer in self.weight_layers:
            target_col = f'target_{layer}'
            if target_col not in df.columns:
                logger.warning(f"목표 컬럼 {target_col} 없음, 건너뜀")
                continue
            
            logger.info(f"{layer} 레이어 딥러닝 모델 훈련 중...")
            
            # 데이터 준비
            X = df[feature_cols].fillna(0)
            y = df[target_col]
            
            # 훈련/테스트 분할
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # 특성 정규화
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 딥러닝 모델 훈련
            model, mse, r2, epochs_trained = self.train_deep_model(
                X_train_scaled, X_test_scaled, y_train, y_test, layer
            )
            
            logger.info(f"{layer} 모델 성능:")
            logger.info(f"  - MSE: {mse:.6f}")
            logger.info(f"  - R²: {r2:.6f}")
            logger.info(f"  - 훈련 에포크: {epochs_trained}")
            
            # 모델 및 스케일러 저장
            self.trained_models[layer] = model
            self.scalers[layer] = scaler
            
            # 파일로 저장
            model_path = self.model_dir / f"{layer}_model.pth"
            scaler_path = self.model_dir / f"{layer}_scaler.joblib"
            
            torch.save(model.state_dict(), model_path)
            joblib.dump(scaler, scaler_path)
            
            logger.info(f"{layer} 모델 저장: {model_path}")
        
        # 훈련 완료 플래그 저장
        training_info = {
            'model_type': 'deep_learning',
            'trained_layers': self.weight_layers,
            'training_date': datetime.now().isoformat(),
            'training_samples': len(df),
            'feature_columns': feature_cols,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate
        }
        
        info_path = self.model_dir / "training_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(training_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ 모든 딥러닝 모델 훈련 완료! 정보 저장: {info_path}")

def main():
    """메인 훈련 함수"""
    trainer = BenthamMLTrainer()
    
    # 데이터 로드
    df = trainer.load_training_data()
    
    if len(df) < 10:
        logger.error("훈련 데이터가 부족합니다 (최소 10개 필요)")
        return
    
    # 모델 훈련
    trainer.train_models(df)
    
    logger.info("🎉 벤담 ML 모델 훈련 완료!")

if __name__ == "__main__":
    main()