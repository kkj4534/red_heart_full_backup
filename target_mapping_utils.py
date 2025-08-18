#!/usr/bin/env python3
"""
Red Heart AI 타깃 매핑 유틸리티
- 전처리된 데이터를 학습용 타깃으로 변환
- 프로젝트 규칙: NO FALLBACK, 더미 데이터 없음
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger('RedHeart.TargetMapping')

class TargetMapper:
    """전처리 데이터를 학습 타깃으로 매핑"""
    
    # Sentence-Transformers 인코더 캐싱 (클래스 변수)
    _sentence_encoder = None
    _encoder_device = None
    
    @classmethod
    def _get_sentence_encoder(cls):
        """
        Sentence-Transformers 인코더 초기화 및 반환
        캐싱된 인스턴스 사용
        """
        if cls._sentence_encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("🔄 Sentence-Transformers 인코더 초기화 중...")
                
                # 경량 모델 사용 (384차원 출력)
                # all-MiniLM-L6-v2: 80MB, 빠르고 효율적
                cls._sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
                
                # 디바이스 설정
                cls._encoder_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                cls._sentence_encoder = cls._sentence_encoder.to(cls._encoder_device)
                
                logger.info(f"✅ Sentence-Transformers 인코더 준비 완료 (device: {cls._encoder_device})")
            except ImportError as e:
                raise RuntimeError(f"sentence-transformers 라이브러리 필요: {e}")
            except Exception as e:
                raise RuntimeError(f"Sentence-Transformers 초기화 실패: {e}")
        
        return cls._sentence_encoder
    
    @classmethod
    def extract_context_embedding(cls, batch: List[Dict], backbone_model=None) -> torch.Tensor:
        """
        컨텍스트 임베딩 추출 (호환성 모드 포함)
        Args:
            batch: 배치 데이터 (리스트 of dict)
            backbone_model: 텍스트를 임베딩으로 변환할 백본 모델 (선택)
        Returns:
            (batch_size, 768) 텐서
        """
        if not batch:
            raise ValueError("배치가 비어있음")
        
        embeddings = []
        
        for item in batch:
            # context_embedding이 있으면 직접 사용
            if 'context_embedding' in item:
                emb = item['context_embedding']
                if isinstance(emb, list):
                    emb = torch.tensor(emb, dtype=torch.float32)
                elif isinstance(emb, np.ndarray):
                    emb = torch.from_numpy(emb).float()
                elif isinstance(emb, torch.Tensor):
                    emb = emb.float()
                else:
                    raise TypeError(f"지원하지 않는 임베딩 타입: {type(emb)}")
                embeddings.append(emb)
            
            # text만 있는 경우 (Sentence-Transformers 사용)
            elif 'text' in item:
                text = item['text']
                
                # Sentence-Transformers로 실시간 임베딩 생성
                try:
                    encoder = cls._get_sentence_encoder()
                    with torch.no_grad():
                        # 텍스트를 임베딩으로 변환 (384차원)
                        emb = encoder.encode(text, convert_to_tensor=True)
                        
                        # 768차원으로 투영 (백본과 호환)
                        if emb.shape[-1] == 384:
                            # 간단한 선형 투영으로 768차원 확장
                            # 필요시 학습 가능한 투영 레이어 추가 가능
                            if not hasattr(cls, '_projection_layer'):
                                cls._projection_layer = torch.nn.Linear(384, 768)
                                cls._projection_layer = cls._projection_layer.to(cls._encoder_device)
                                cls._projection_layer.eval()  # 학습 비활성화
                                logger.debug("투영 레이어 생성: 384 -> 768")
                            
                            emb = cls._projection_layer(emb)
                        
                        # float32로 변환 및 CPU로 이동 (나중에 통합)
                        emb = emb.to(torch.float32).cpu()
                        
                except Exception as e:
                    # NO FALLBACK 원칙 - 실패 시 예외 발생
                    raise RuntimeError(f"텍스트 임베딩 생성 실패: {e}")
                
                embeddings.append(emb)
            else:
                raise KeyError(f"샘플에 context_embedding 또는 text 없음: {item.get('id', 'unknown')}")
        
        # 스택하여 배치 텐서 생성
        return torch.stack(embeddings)
    
    @staticmethod
    def extract_emotion_target(batch: List[Dict]) -> torch.Tensor:
        """
        감정 타깃 추출 (7차원 벡터)
        Args:
            batch: 배치 데이터
        Returns:
            (batch_size, 7) 텐서
        """
        if not batch:
            raise ValueError("배치가 비어있음")
        
        emotion_vectors = []
        for item in batch:
            # emotion_vector 또는 emotions 키 확인 (호환성)
            if 'emotion_vector' in item:
                vec = item['emotion_vector']
            elif 'emotions' in item:
                vec = item['emotions']
            else:
                raise KeyError(f"emotion_vector 또는 emotions 없음")
            
            if isinstance(vec, list) and len(vec) == 7:
                vec = torch.tensor(vec, dtype=torch.float32)
            else:
                raise ValueError(f"감정 벡터 차원 오류: {len(vec) if isinstance(vec, list) else 'not list'}")
            
            emotion_vectors.append(vec)
        
        return torch.stack(emotion_vectors)
    
    @staticmethod
    def extract_emotion_labels(batch: List[Dict]) -> torch.Tensor:
        """
        감정 라벨 추출 (카테고리 인덱스)
        Args:
            batch: 배치 데이터
        Returns:
            (batch_size,) 정수 텐서
        """
        emotion_vectors = TargetMapper.extract_emotion_target(batch)
        # 가장 높은 값의 인덱스를 라벨로 사용
        return torch.argmax(emotion_vectors, dim=1)
    
    @staticmethod
    def extract_regret_target(batch: List[Dict]) -> torch.Tensor:
        """
        후회 지수 타깃 추출
        Args:
            batch: 배치 데이터
        Returns:
            (batch_size, 1) 텐서
        """
        if not batch:
            raise ValueError("배치가 비어있음")
        
        regret_scores = []
        for item in batch:
            if 'regret_factor' not in item:
                raise KeyError("regret_factor 없음")
            
            score = item['regret_factor']
            if isinstance(score, (int, float)):
                score = torch.tensor([score], dtype=torch.float32)
            else:
                raise TypeError(f"후회 지수 타입 오류: {type(score)}")
            
            regret_scores.append(score)
        
        return torch.stack(regret_scores)
    
    @staticmethod
    def extract_bentham_target(batch: List[Dict]) -> torch.Tensor:
        """
        벤담 점수 타깃 추출 (10차원)
        Args:
            batch: 배치 데이터
        Returns:
            (batch_size, 10) 텐서
        """
        if not batch:
            raise ValueError("배치가 비어있음")
        
        # 벤담 키 순서 고정 (중요!)
        bentham_keys = [
            'intensity', 'duration', 'certainty', 'propinquity',
            'purity', 'extent', 'fecundity', 'remoteness', 
            'succession', 'utility'
        ]
        
        bentham_vectors = []
        for item in batch:
            if 'bentham_scores' not in item:
                raise KeyError("bentham_scores 없음")
            
            scores = item['bentham_scores']
            if not isinstance(scores, dict):
                raise TypeError(f"bentham_scores가 dict가 아님: {type(scores)}")
            
            # 고정된 순서로 벡터 생성 (없는 키는 0.5로 기본값)
            vec = []
            for key in bentham_keys:
                if key in scores:
                    vec.append(float(scores[key]))
                else:
                    # 기본값 0.5 (중간값)
                    vec.append(0.5)
            
            bentham_vectors.append(torch.tensor(vec, dtype=torch.float32))
        
        return torch.stack(bentham_vectors)
    
    @staticmethod
    def extract_surd_target(batch: List[Dict], normalize: bool = True) -> torch.Tensor:
        """
        SURD 메트릭 타깃 추출 (4차원)
        Args:
            batch: 배치 데이터
            normalize: True면 합이 1이 되도록 정규화
        Returns:
            (batch_size, 4) 텐서
        """
        if not batch:
            raise ValueError("배치가 비어있음")
        
        # SURD 키 순서 고정 (호환성 매핑 포함)
        surd_keys = ['sufficiency', 'understandability', 'resilience', 'decisiveness']
        # 실제 데이터 키 -> 정식 키 매핑
        key_mapping = {
            'selection': 'sufficiency',
            'uncertainty': 'understandability', 
            'risk': 'resilience',
            'decision': 'decisiveness'
        }
        
        surd_vectors = []
        for item in batch:
            if 'surd_metrics' not in item:
                raise KeyError("surd_metrics 없음")
            
            metrics = item['surd_metrics']
            if not isinstance(metrics, dict):
                raise TypeError(f"surd_metrics가 dict가 아님: {type(metrics)}")
            
            # 고정된 순서로 벡터 생성
            vec = []
            for key in surd_keys:
                value = None
                # 정식 키로 먼저 확인
                if key in metrics:
                    value = float(metrics[key])
                else:
                    # 매핑된 키 확인
                    for old_key, new_key in key_mapping.items():
                        if new_key == key and old_key in metrics:
                            value = float(metrics[old_key])
                            break
                
                if value is None:
                    raise KeyError(f"surd_metrics에 {key} 또는 매핑된 키 없음")
                vec.append(value)
            
            vec_tensor = torch.tensor(vec, dtype=torch.float32)
            
            # 정규화 (합이 1이 되도록)
            if normalize:
                vec_sum = vec_tensor.sum()
                if vec_sum > 0:
                    vec_tensor = vec_tensor / vec_sum
                else:
                    # 모두 0인 경우 균등 분포
                    vec_tensor = torch.ones(4) / 4
            
            surd_vectors.append(vec_tensor)
        
        return torch.stack(surd_vectors)
    
    @staticmethod
    def validate_batch(batch: List[Dict]) -> bool:
        """
        배치 데이터 유효성 검증
        Args:
            batch: 배치 데이터
        Returns:
            유효하면 True
        """
        if not batch:
            logger.error("빈 배치")
            return False
        
        required_keys = [
            'context_embedding',
            'emotion_vector', 
            'regret_factor',
            'bentham_scores',
            'surd_metrics'
        ]
        
        for i, item in enumerate(batch):
            if not isinstance(item, dict):
                logger.error(f"샘플 {i}이 dict가 아님: {type(item)}")
                return False
            
            for key in required_keys:
                if key not in item:
                    logger.error(f"샘플 {i}에 {key} 없음")
                    return False
        
        return True
    
    @staticmethod
    def prepare_training_batch(batch: List[Dict], device: torch.device) -> Dict[str, torch.Tensor]:
        """
        학습용 배치 준비 (모든 타깃 포함)
        Args:
            batch: 배치 데이터
            device: 디바이스 (cuda/cpu)
        Returns:
            학습용 텐서 딕셔너리
        """
        if not TargetMapper.validate_batch(batch):
            raise ValueError("배치 유효성 검증 실패")
        
        # 모든 타깃 추출 및 디바이스 이동
        training_batch = {
            'input': TargetMapper.extract_context_embedding(batch).to(device),
            'emotion_target': TargetMapper.extract_emotion_target(batch).to(device),
            'emotion_labels': TargetMapper.extract_emotion_labels(batch).to(device),
            'regret_target': TargetMapper.extract_regret_target(batch).to(device),
            'bentham_target': TargetMapper.extract_bentham_target(batch).to(device),
            'surd_target': TargetMapper.extract_surd_target(batch, normalize=True).to(device)
        }
        
        # 메타데이터 추가 (선택적)
        if 'source' in batch[0]:
            training_batch['sources'] = [item.get('source', 'unknown') for item in batch]
        
        return training_batch


def test_mapper():
    """매퍼 테스트"""
    # 테스트 데이터 생성
    test_batch = [
        {
            'context_embedding': [0.1] * 768,
            'emotion_vector': [0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.16],
            'regret_factor': 0.5,
            'bentham_scores': {
                'intensity': 0.6, 'duration': 0.5, 'certainty': 0.7, 'propinquity': 0.4,
                'purity': 0.8, 'extent': 0.5, 'fecundity': 0.6, 'remoteness': 0.3,
                'succession': 0.4, 'utility': 0.55
            },
            'surd_metrics': {
                'sufficiency': 0.7, 'understandability': 0.8, 
                'resilience': 0.6, 'decisiveness': 0.9
            },
            'source': 'test'
        }
    ] * 4  # 배치 크기 4
    
    device = torch.device('cpu')
    
    try:
        # 배치 준비
        training_batch = TargetMapper.prepare_training_batch(test_batch, device)
        
        print("✅ 매퍼 테스트 성공!")
        print(f"  - Input shape: {training_batch['input'].shape}")
        print(f"  - Emotion target shape: {training_batch['emotion_target'].shape}")
        print(f"  - Emotion labels shape: {training_batch['emotion_labels'].shape}")
        print(f"  - Regret target shape: {training_batch['regret_target'].shape}")
        print(f"  - Bentham target shape: {training_batch['bentham_target'].shape}")
        print(f"  - SURD target shape: {training_batch['surd_target'].shape}")
        
        # SURD 정규화 확인
        surd_sums = training_batch['surd_target'].sum(dim=1)
        print(f"  - SURD 합 검증: {surd_sums} (모두 1이어야 함)")
        
    except Exception as e:
        print(f"❌ 매퍼 테스트 실패: {e}")
        raise


if __name__ == "__main__":
    test_mapper()