#!/usr/bin/env python3
"""
간단한 학습 테스트 스크립트
Simple Learning Test Script
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import time
import math
import random
from datetime import datetime

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SimpleLearningTest')

def check_dependencies():
    """의존성 확인"""
    dependencies = {
        'builtin_only': True,
        'torch': False,
        'transformers': False,
        'sklearn': False
    }
    
    logger.info("✅ Python 기본 라이브러리만 사용 (fallback 모드)")
    
    try:
        import torch
        dependencies['torch'] = True
        logger.info(f"✅ PyTorch {torch.__version__} 사용 가능")
    except ImportError:
        logger.warning("❌ PyTorch 사용 불가")
    
    try:
        import transformers
        dependencies['transformers'] = True
        logger.info(f"✅ Transformers {transformers.__version__} 사용 가능")
    except ImportError:
        logger.warning("❌ Transformers 사용 불가")
    
    try:
        import sklearn
        dependencies['sklearn'] = True
        logger.info(f"✅ Scikit-learn {sklearn.__version__} 사용 가능")
    except ImportError:
        logger.warning("❌ Scikit-learn 사용 불가")
    
    return dependencies

def load_dataset_sample(dataset_path: Path, max_samples: int = 5) -> List[Dict[str, Any]]:
    """데이터셋 샘플 로드"""
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        scenarios = data.get('scenarios', [])
        sample_scenarios = scenarios[:max_samples]
        
        logger.info(f"✅ {dataset_path.name}에서 {len(sample_scenarios)}개 시나리오 로드")
        return sample_scenarios
        
    except Exception as e:
        logger.error(f"❌ 데이터셋 로드 실패 {dataset_path}: {e}")
        return []

def extract_features_simple(scenario: Dict[str, Any]) -> Dict[str, float]:
    """간단한 특징 추출"""
    features = {}
    
    # 텍스트 기본 특징
    description = scenario.get('description', '')
    features['text_length'] = len(description)
    features['word_count'] = len(description.split())
    features['exclamation_count'] = description.count('!')
    features['question_count'] = description.count('?')
    
    # 감정 특징
    emotions = scenario.get('context', {}).get('emotions', {})
    for emotion, value in emotions.items():
        features[f'emotion_{emotion.lower()}'] = float(value)
    
    # 기타 특징
    stakeholders = scenario.get('context', {}).get('stakeholders', [])
    features['stakeholder_count'] = len(stakeholders)
    
    return features

def simple_emotion_analysis(text: str) -> Dict[str, float]:
    """간단한 감정 분석 (키워드 기반)"""
    
    # 기본 한국어 감정 키워드
    emotion_keywords = {
        'joy': ['기쁘', '행복', '즐거', '좋', '웃', '미소', '기뻐'],
        'sadness': ['슬프', '우울', '눈물', '아프', '괴로', '슬픔'],
        'anger': ['화나', '분노', '짜증', '억울', '화가', '열받'],
        'fear': ['무서', '두려', '걱정', '불안', '떨리', '겁나'],
        'surprise': ['놀라', '깜짝', '어머', '헉', '와', '어?'],
        'disgust': ['역겨', '싫어', '지겨', '혐오', '구역']
    }
    
    text_lower = text.lower()
    emotion_scores = {}
    
    for emotion, keywords in emotion_keywords.items():
        score = 0
        for keyword in keywords:
            score += text_lower.count(keyword)
        
        # 정규화 (0-1 범위)
        emotion_scores[emotion] = min(score / 10, 1.0)
    
    return emotion_scores

def simple_regret_analysis(scenario: Dict[str, Any]) -> float:
    """간단한 후회 분석"""
    regret_keywords = ['후회', '아쉽', '실수', '잘못', '미안', '죄송']
    
    text = scenario.get('description', '') + ' ' + scenario.get('context', {}).get('regret_info', '')
    text_lower = text.lower()
    
    regret_score = 0
    for keyword in regret_keywords:
        regret_score += text_lower.count(keyword)
    
    return min(regret_score / 5, 1.0)

def simple_surd_analysis(features: Dict[str, float]) -> Dict[str, float]:
    """간단한 SURD 분석 (상관관계 기반)"""
    
    # 특징들을 배열로 변환
    feature_values = list(features.values())
    
    if len(feature_values) < 2:
        return {'synergy': 0.0, 'redundancy': 0.0, 'unique': 0.5}
    
    # 간단한 상관관계 계산 (numpy 없이)
    mean_val = sum(feature_values) / len(feature_values)
    variance = sum((x - mean_val) ** 2 for x in feature_values) / len(feature_values)
    std_val = math.sqrt(variance)
    
    # SURD 지표 계산 (간단한 휴리스틱)
    synergy = min(std_val / (mean_val + 0.001), 1.0)
    redundancy = max(0, 1 - synergy)
    unique = abs(mean_val - 0.5)
    
    return {
        'synergy': synergy,
        'redundancy': redundancy,
        'unique': unique
    }

class SimpleLearningSystem:
    """간단한 학습 시스템"""
    
    def __init__(self):
        self.emotion_weights = [random.random() * 0.1 + 0.5 for _ in range(6)]
        self.regret_weights = [random.random() * 0.1 + 0.5 for _ in range(3)]
        self.learning_rate = 0.01
        self.training_history = []
    
    def predict_emotion(self, features: Dict[str, float]) -> Dict[str, float]:
        """감정 예측"""
        # 단순한 선형 결합
        emotion_features = [v for k, v in features.items() if k.startswith('emotion_')]
        
        if len(emotion_features) >= 3:
            prediction = sum(a * b for a, b in zip(emotion_features[:3], self.emotion_weights[:3]))
        else:
            feature_values = list(features.values())[:3]
            prediction = sum(feature_values) / len(feature_values) if feature_values else 0.5
        
        return {
            'predicted_valence': math.tanh(prediction),
            'confidence': min(abs(prediction), 1.0)
        }
    
    def predict_regret(self, features: Dict[str, float]) -> float:
        """후회 예측"""
        text_features = [
            features.get('text_length', 0) / 1000,
            features.get('word_count', 0) / 100,
            features.get('stakeholder_count', 0) / 10
        ]
        
        prediction = sum(a * b for a, b in zip(text_features, self.regret_weights))
        return 1 / (1 + math.exp(-prediction))  # sigmoid function
    
    def train_step(self, features: Dict[str, float], target_emotion: float, target_regret: float):
        """단일 훈련 스텝"""
        
        # 예측
        emotion_pred = self.predict_emotion(features)
        regret_pred = self.predict_regret(features)
        
        # 오차 계산
        emotion_error = target_emotion - emotion_pred['predicted_valence']
        regret_error = target_regret - regret_pred
        
        # 가중치 업데이트 (간단한 그래디언트)
        for i in range(len(self.emotion_weights)):
            self.emotion_weights[i] += self.learning_rate * emotion_error * 0.1
        for i in range(len(self.regret_weights)):
            self.regret_weights[i] += self.learning_rate * regret_error * 0.1
        
        # 이력 저장
        self.training_history.append({
            'emotion_error': abs(emotion_error),
            'regret_error': abs(regret_error),
            'emotion_weights': self.emotion_weights[:],  # copy list
            'regret_weights': self.regret_weights[:]     # copy list
        })
    
    def get_training_stats(self) -> Dict[str, Any]:
        """훈련 통계"""
        if not self.training_history:
            return {}
        
        emotion_errors = [h['emotion_error'] for h in self.training_history]
        regret_errors = [h['regret_error'] for h in self.training_history]
        
        return {
            'total_steps': len(self.training_history),
            'avg_emotion_error': sum(emotion_errors) / len(emotion_errors),
            'avg_regret_error': sum(regret_errors) / len(regret_errors),
            'final_emotion_weights': self.emotion_weights,
            'final_regret_weights': self.regret_weights,
            'improvement': emotion_errors[0] - emotion_errors[-1] if len(emotion_errors) > 1 else 0
        }

def run_simple_learning_test():
    """간단한 학습 테스트 실행"""
    logger.info("🚀 간단한 학습 테스트 시작")
    
    # 1. 의존성 확인
    logger.info("📋 1. 의존성 확인")
    deps = check_dependencies()
    
    # 2. 데이터셋 로드
    logger.info("📂 2. 데이터셋 로드")
    datasets_dir = project_root / 'processed_datasets'
    
    # 여러 데이터 소스 테스트
    test_datasets = [
        datasets_dir / 'scruples' / 'scruples_batch_001_of_100_20250622_013432.json',
        datasets_dir / 'scruples' / 'scruples_batch_002_of_100_20250622_013432.json',
        datasets_dir / 'classic_literature' / 'classic_literature_batch_001_of_001_20250622_013442.json',
        datasets_dir / 'korean_cultural_scenarios.json'
    ]
    
    all_scenarios = []
    for dataset_path in test_datasets:
        scenarios = load_dataset_sample(dataset_path, 2)  # 각 데이터셋에서 2개씩
        all_scenarios.extend(scenarios)
    
    if not all_scenarios:
        logger.error("❌ 로드할 수 있는 시나리오가 없습니다.")
        return
    
    logger.info(f"✅ 총 {len(all_scenarios)}개 시나리오 로드 완료")
    
    # 3. 특징 추출 및 분석
    logger.info("🔬 3. 특징 추출 및 분석")
    learning_system = SimpleLearningSystem()
    
    for i, scenario in enumerate(all_scenarios):
        logger.info(f"\n--- 시나리오 {i+1}: {scenario.get('title', 'Unknown')} ---")
        
        # 특징 추출
        features = extract_features_simple(scenario)
        logger.info(f"추출된 특징: {len(features)}개")
        
        # 감정 분석
        emotion_analysis = simple_emotion_analysis(scenario.get('description', ''))
        logger.info(f"감정 분석: {emotion_analysis}")
        
        # 후회 분석
        regret_score = simple_regret_analysis(scenario)
        logger.info(f"후회 점수: {regret_score:.3f}")
        
        # SURD 분석
        surd_analysis = simple_surd_analysis(features)
        logger.info(f"SURD 분석: {surd_analysis}")
        
        # 학습 (실제 값을 대략적으로 추정)
        target_emotion = list(emotion_analysis.values())[0] if emotion_analysis else 0.5
        target_regret = regret_score
        
        learning_system.train_step(features, target_emotion, target_regret)
        
        # 예측 테스트
        emotion_pred = learning_system.predict_emotion(features)
        regret_pred = learning_system.predict_regret(features)
        
        logger.info(f"감정 예측: {emotion_pred}")
        logger.info(f"후회 예측: {regret_pred:.3f}")
    
    # 4. 학습 결과
    logger.info("\n📊 4. 학습 결과")
    training_stats = learning_system.get_training_stats()
    
    for key, value in training_stats.items():
        if isinstance(value, float):
            logger.info(f"{key}: {value:.4f}")
        elif isinstance(value, list) and len(value) < 10:
            logger.info(f"{key}: {[f'{v:.3f}' for v in value]}")
        else:
            logger.info(f"{key}: {value}")
    
    # 5. 결과 저장
    logger.info("💾 5. 결과 저장")
    results = {
        'test_info': {
            'timestamp': datetime.now().isoformat(),
            'scenarios_processed': len(all_scenarios),
            'dependencies': deps
        },
        'training_stats': training_stats,
        'scenarios_analyzed': [
            {
                'title': s.get('title', 'Unknown'),
                'source': s.get('context', {}).get('source', 'unknown')
            } for s in all_scenarios
        ]
    }
    
    results_path = project_root / 'logs' / f'simple_learning_test_{int(time.time())}.json'
    results_path.parent.mkdir(exist_ok=True)
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ 결과 저장 완료: {results_path}")
    logger.info("🎉 간단한 학습 테스트 완료!")
    
    return results

if __name__ == "__main__":
    try:
        results = run_simple_learning_test()
        print("\n" + "="*50)
        print("✅ 학습 테스트 성공!")
        print("="*50)
    except Exception as e:
        print(f"\n❌ 학습 테스트 실패: {e}")
        import traceback
        traceback.print_exc()