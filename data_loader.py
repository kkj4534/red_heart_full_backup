"""
Red Heart AI 데이터 로더
Claude API로 전처리된 데이터를 학습용으로 로드
"""

import json
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PreprocessedDataLoader:
    """Claude API로 전처리된 데이터 로더"""
    
    def __init__(self, data_path: str = "/mnt/c/large_project/linux_red_heart/claude_api_preprocessing/claude_preprocessed_complete.json"):
        """
        Args:
            data_path: 전처리된 JSON 파일 경로
        """
        self.data_path = Path(data_path)
        self.data = None
        self.embeddings_cache = {}
        
        # 데이터 로드
        self._load_data()
    
    def _load_data(self):
        """전처리된 데이터 로드"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"전처리 데이터 파일을 찾을 수 없습니다: {self.data_path}")
        
        logger.info(f"📥 전처리 데이터 로드 중: {self.data_path}")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # 데이터 구조 파악
        if isinstance(raw_data, list):
            self.data = raw_data
        elif isinstance(raw_data, dict) and 'samples' in raw_data:
            self.data = raw_data['samples']
        else:
            raise ValueError("지원하지 않는 데이터 형식입니다")
        
        logger.info(f"✅ {len(self.data)}개 샘플 로드 완료")
        
        # 첫 번째 샘플 구조 출력
        if self.data:
            sample = self.data[0]
            logger.info("📊 샘플 구조:")
            for key in sample.keys():
                if isinstance(sample[key], dict):
                    logger.info(f"  - {key}: {list(sample[key].keys())}")
                elif isinstance(sample[key], list):
                    logger.info(f"  - {key}: List[{len(sample[key])}]")
                else:
                    logger.info(f"  - {key}: {type(sample[key]).__name__}")
    
    def get_batch(self, indices: List[int]) -> Dict[str, torch.Tensor]:
        """배치 데이터 추출 및 텐서 변환
        
        Args:
            indices: 샘플 인덱스 리스트
            
        Returns:
            배치 데이터 딕셔너리 (텐서)
        """
        batch_data = {
            'texts': [],
            'embeddings': [],
            'emotion_labels': [],
            'bentham_scores': [],
            'surd_metrics': [],
            'regret_factors': []
        }
        
        for idx in indices:
            sample = self.data[idx]
            
            # 텍스트
            if 'text' in sample:
                batch_data['texts'].append(sample['text'])
            elif 'post' in sample:
                batch_data['texts'].append(sample['post'])
            
            # 임베딩 (없으면 생성 필요)
            if 'embedding' in sample:
                batch_data['embeddings'].append(sample['embedding'])
            else:
                # 임시로 768차원 제로 벡터 (나중에 실제 인코더 사용)
                batch_data['embeddings'].append(np.zeros(768))
            
            # 감정 라벨
            if 'emotions' in sample:
                emotions = sample['emotions']
                if isinstance(emotions, dict):
                    # 7개 기본 감정 추출
                    emotion_vector = [
                        emotions.get('joy', 0.0),
                        emotions.get('anger', 0.0),
                        emotions.get('surprise', 0.0),
                        emotions.get('disgust', 0.0),
                        emotions.get('sadness', 0.0),
                        emotions.get('shame', 0.0),
                        emotions.get('fear', 0.0)
                    ]
                else:
                    emotion_vector = emotions[:7]
                batch_data['emotion_labels'].append(emotion_vector)
            
            # 벤담 점수
            if 'bentham_scores' in sample:
                bentham = sample['bentham_scores']
                if isinstance(bentham, dict):
                    bentham_vector = [
                        bentham.get('intensity', 0.5),
                        bentham.get('duration', 0.5),
                        bentham.get('certainty', 0.5),
                        bentham.get('propinquity', 0.5),
                        bentham.get('fecundity', 0.5),
                        bentham.get('purity', 0.5),
                        bentham.get('extent', 0.5)
                    ]
                else:
                    bentham_vector = bentham
                batch_data['bentham_scores'].append(bentham_vector)
            
            # SURD 메트릭
            if 'surd_metrics' in sample:
                surd = sample['surd_metrics']
                if isinstance(surd, dict):
                    surd_vector = [
                        surd.get('selection', 0.5),
                        surd.get('uncertainty', 0.5),
                        surd.get('risk', 0.5),
                        surd.get('decision', 0.5)
                    ]
                else:
                    surd_vector = surd
                batch_data['surd_metrics'].append(surd_vector)
            
            # 후회 팩터
            if 'regret_factor' in sample:
                batch_data['regret_factors'].append([sample['regret_factor']])
            elif 'regret' in sample:
                batch_data['regret_factors'].append([sample['regret']])
            else:
                batch_data['regret_factors'].append([0.5])
        
        # 텐서 변환
        result = {}
        
        # 텍스트는 리스트로 유지
        if batch_data['texts']:
            result['texts'] = batch_data['texts']
        
        # 나머지는 텐서로 변환
        if batch_data['embeddings']:
            result['embeddings'] = torch.tensor(batch_data['embeddings'], dtype=torch.float32)
        
        if batch_data['emotion_labels']:
            result['emotion_labels'] = torch.tensor(batch_data['emotion_labels'], dtype=torch.float32)
        
        if batch_data['bentham_scores']:
            result['bentham_scores'] = torch.tensor(batch_data['bentham_scores'], dtype=torch.float32)
        
        if batch_data['surd_metrics']:
            result['surd_metrics'] = torch.tensor(batch_data['surd_metrics'], dtype=torch.float32)
        
        if batch_data['regret_factors']:
            result['regret_factors'] = torch.tensor(batch_data['regret_factors'], dtype=torch.float32)
        
        return result
    
    def split_data(self, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[List, List, List]:
        """데이터셋 분할
        
        Args:
            train_ratio: 학습 데이터 비율
            val_ratio: 검증 데이터 비율
            
        Returns:
            (train_data, val_data, test_data)
        """
        n = len(self.data)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        # 인덱스 셔플
        indices = np.random.permutation(n)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train+n_val]
        test_indices = indices[n_train+n_val:]
        
        train_data = [self.data[i] for i in train_indices]
        val_data = [self.data[i] for i in val_indices]
        test_data = [self.data[i] for i in test_indices]
        
        logger.info(f"📊 데이터 분할 완료:")
        logger.info(f"  - 학습: {len(train_data)} 샘플")
        logger.info(f"  - 검증: {len(val_data)} 샘플")
        logger.info(f"  - 테스트: {len(test_data)} 샘플")
        
        return train_data, val_data, test_data
    
    def __len__(self):
        return len(self.data) if self.data else 0
    
    def __getitem__(self, idx):
        return self.data[idx]


class TargetMapper:
    """배치 데이터에서 타깃 추출"""
    
    @staticmethod
    def extract_emotion_target(batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """감정 타깃 추출
        
        Args:
            batch_data: 배치 데이터 딕셔너리
            
        Returns:
            감정 타깃 텐서 (분류: argmax 인덱스 또는 회귀: 벡터)
        """
        if 'emotion_labels' in batch_data:
            # 회귀 태스크: 전체 감정 벡터 반환
            return batch_data['emotion_labels']
        else:
            # 기본값: 균등 분포
            batch_size = len(batch_data.get('texts', [1]))
            return torch.ones(batch_size, 7) / 7.0
    
    @staticmethod
    def extract_bentham_target(batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """벤담 타깃 추출"""
        if 'bentham_scores' in batch_data:
            return batch_data['bentham_scores']
        else:
            batch_size = len(batch_data.get('texts', [1]))
            return torch.ones(batch_size, 7) * 0.5
    
    @staticmethod
    def extract_surd_target(batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """SURD 타깃 추출"""
        if 'surd_metrics' in batch_data:
            return batch_data['surd_metrics']
        else:
            batch_size = len(batch_data.get('texts', [1]))
            return torch.ones(batch_size, 4) * 0.5
    
    @staticmethod
    def extract_regret_target(batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """후회 타깃 추출"""
        if 'regret_factors' in batch_data:
            return batch_data['regret_factors']
        else:
            batch_size = len(batch_data.get('texts', [1]))
            return torch.ones(batch_size, 1) * 0.5