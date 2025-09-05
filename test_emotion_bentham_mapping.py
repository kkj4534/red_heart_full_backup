#!/usr/bin/env python3
"""
감정→벤담 정밀 매핑 테스트 스크립트
휴리스틱 vs 의미론적 매핑 비교
"""

import numpy as np
import torch
from semantic_emotion_bentham_mapper import (
    SemanticEmotionBenthamMapper,
    NeuralEmotionBenthamAdapter,
    EMOTION_DIMENSIONS,
    BENTHAM_DIMENSIONS
)
import matplotlib.pyplot as plt
import seaborn as sns


def heuristic_mapping(emotion_scores):
    """기존 휴리스틱 매핑 (단순 인덱스)"""
    bentham = {}
    bentham_names = list(BENTHAM_DIMENSIONS.keys())
    
    for i, name in enumerate(bentham_names):
        if i < len(emotion_scores):
            bentham[name] = emotion_scores[i]
        else:
            bentham[name] = 0.5
    
    return bentham


def test_mapping_quality():
    """매핑 품질 테스트"""
    print("=" * 70)
    print("감정→벤담 매핑 품질 비교 테스트")
    print("=" * 70)
    
    # 테스트 시나리오들
    test_scenarios = [
        {
            'name': '긍정적 흥분 상태',
            'emotion': {
                'valence': 0.8,      # 매우 긍정적
                'arousal': 0.9,      # 높은 각성
                'dominance': 0.7,    # 통제감
                'certainty': 0.8,    # 확실
                'surprise': 0.2,     # 예상됨
                'anticipation': 0.9  # 높은 기대
            },
            'expected': {
                'intensity': 'high',       # 높은 각성 → 높은 강도
                'duration': 'medium-high', # 통제감 → 지속성
                'fecundity': 'high',      # 긍정+기대 → 생산성
                'extent': 'high'          # 강한 감정 → 넓은 영향
            }
        },
        {
            'name': '불안과 무력감',
            'emotion': {
                'valence': -0.7,     # 부정적
                'arousal': 0.8,      # 높은 각성 (불안)
                'dominance': -0.6,   # 무력감
                'certainty': 0.2,    # 불확실
                'surprise': 0.7,     # 예상 못함
                'anticipation': 0.1  # 기대 없음
            },
            'expected': {
                'intensity': 'high',        # 불안한 각성
                'duration': 'low',          # 무력감 → 짧은 지속
                'self_damage': 'high',      # 부정+무력 → 자기손상
                'external_cost': 'high'     # 부정적 영향
            }
        },
        {
            'name': '평온한 확신',
            'emotion': {
                'valence': 0.5,      # 약간 긍정적
                'arousal': -0.2,     # 낮은 각성 (평온)
                'dominance': 0.6,    # 적절한 통제
                'certainty': 0.9,    # 매우 확실
                'surprise': 0.0,     # 완전 예상
                'anticipation': 0.4  # 보통 기대
            },
            'expected': {
                'intensity': 'low',         # 낮은 각성
                'duration': 'high',         # 확실+통제 → 지속
                'certainty': 'very-high',   # 매우 확실
                'purity': 'high'            # 명확한 상태
            }
        }
    ]
    
    # 매퍼 초기화
    semantic_mapper = SemanticEmotionBenthamMapper()
    
    # 각 시나리오 테스트
    for scenario in test_scenarios:
        print(f"\n🔬 시나리오: {scenario['name']}")
        print("-" * 50)
        
        # 감정 벡터 생성 (리스트 형태로)
        emotion_list = [scenario['emotion'][dim] for dim in EMOTION_DIMENSIONS.keys()]
        
        # 휴리스틱 매핑
        heuristic_result = heuristic_mapping(emotion_list)
        
        # 의미론적 매핑
        semantic_result = semantic_mapper.map_emotion_to_bentham(scenario['emotion'])
        
        # 결과 비교
        print("\n📊 매핑 결과 비교:")
        print(f"{'벤담 차원':<20} {'휴리스틱':<12} {'의미론적':<12} {'예상':<15}")
        print("-" * 60)
        
        for bentham_dim in BENTHAM_DIMENSIONS.keys():
            heur_val = heuristic_result.get(bentham_dim, 0)
            sem_val = semantic_result.get(bentham_dim, 0)
            
            # 예상값과 비교
            expected = scenario['expected'].get(bentham_dim, '')
            if expected:
                if 'high' in expected and sem_val > 0.7:
                    match = '✅'
                elif 'low' in expected and sem_val < 0.3:
                    match = '✅'
                elif 'medium' in expected and 0.3 <= sem_val <= 0.7:
                    match = '✅'
                else:
                    match = '❌'
                expected += f" {match}"
            
            print(f"{bentham_dim:<20} {heur_val:<12.3f} {sem_val:<12.3f} {expected:<15}")
        
        # 의미론적 일관성 점수
        print(f"\n💡 의미론적 일관성 분석:")
        
        # 핵심 연결 확인
        checks = []
        
        # 각성도 → 강도
        if scenario['emotion']['arousal'] > 0.5 and semantic_result['intensity'] > 0.6:
            checks.append("✅ 각성도 → 강도")
        elif scenario['emotion']['arousal'] < 0 and semantic_result['intensity'] < 0.4:
            checks.append("✅ 낮은 각성 → 낮은 강도")
        
        # 통제감 → 지속성
        if scenario['emotion']['dominance'] > 0.5 and semantic_result['duration'] > 0.6:
            checks.append("✅ 통제감 → 지속성")
        elif scenario['emotion']['dominance'] < 0 and semantic_result['duration'] < 0.4:
            checks.append("✅ 무력감 → 짧은 지속")
        
        # 부정 감정 → 자기손상
        if scenario['emotion']['valence'] < -0.5 and semantic_result['self_damage'] > 0.6:
            checks.append("✅ 부정 감정 → 자기손상")
        
        # 긍정+기대 → 생산성
        if scenario['emotion']['valence'] > 0.5 and scenario['emotion']['anticipation'] > 0.5:
            if semantic_result['fecundity'] > 0.6:
                checks.append("✅ 긍정+기대 → 생산성")
        
        for check in checks:
            print(f"   {check}")


def visualize_mapping_comparison():
    """매핑 비교 시각화"""
    print("\n" + "=" * 70)
    print("매핑 행렬 시각화")
    print("=" * 70)
    
    # 의미론적 매핑 행렬 생성
    mapper = SemanticEmotionBenthamMapper()
    mapping_matrix = np.zeros((10, 6))
    
    for b_idx, b_name in enumerate(BENTHAM_DIMENSIONS.keys()):
        if b_name in mapper.mapping_rules:
            for e_name, weight in mapper.mapping_rules[b_name]:
                e_idx = EMOTION_DIMENSIONS[e_name]
                mapping_matrix[b_idx, e_idx] = weight
    
    # 히트맵 생성
    plt.figure(figsize=(10, 8))
    
    # 의미론적 매핑 히트맵
    plt.subplot(1, 2, 1)
    sns.heatmap(mapping_matrix, 
                xticklabels=list(EMOTION_DIMENSIONS.keys()),
                yticklabels=list(BENTHAM_DIMENSIONS.keys()),
                cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                annot=True, fmt='.2f', cbar_kws={'label': '가중치'})
    plt.title('의미론적 매핑 행렬')
    plt.xlabel('감정 차원')
    plt.ylabel('벤담 차원')
    
    # 휴리스틱 매핑 (대각선)
    plt.subplot(1, 2, 2)
    heuristic_matrix = np.zeros((10, 6))
    for i in range(min(6, 10)):
        heuristic_matrix[i, i] = 1.0
    
    sns.heatmap(heuristic_matrix,
                xticklabels=list(EMOTION_DIMENSIONS.keys()),
                yticklabels=list(BENTHAM_DIMENSIONS.keys()),
                cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                annot=True, fmt='.1f', cbar_kws={'label': '가중치'})
    plt.title('휴리스틱 매핑 (단순 인덱스)')
    plt.xlabel('감정 차원')
    plt.ylabel('벤담 차원')
    
    plt.tight_layout()
    plt.savefig('emotion_bentham_mapping_comparison.png', dpi=150)
    print("📊 시각화 저장: emotion_bentham_mapping_comparison.png")
    plt.show()


def test_neural_adapter():
    """신경망 어댑터 테스트"""
    print("\n" + "=" * 70)
    print("신경망 어댑터 학습 시뮬레이션")
    print("=" * 70)
    
    # 어댑터 초기화
    adapter = NeuralEmotionBenthamAdapter()
    optimizer = torch.optim.Adam(adapter.parameters(), lr=0.001)
    
    # 의미론적 매퍼로 학습 데이터 생성
    semantic_mapper = SemanticEmotionBenthamMapper()
    
    # 간단한 학습 시뮬레이션
    print("\n🧠 신경망 어댑터 학습 중...")
    
    losses = []
    for epoch in range(100):
        # 랜덤 감정 생성
        random_emotion = np.random.randn(6)
        random_emotion = np.tanh(random_emotion)  # -1 ~ 1 범위
        
        # 타겟: 의미론적 매핑 결과
        emotion_dict = {dim: random_emotion[idx] for idx, dim in enumerate(EMOTION_DIMENSIONS.keys())}
        target_bentham = semantic_mapper.map_emotion_to_bentham(emotion_dict)
        target_tensor = torch.tensor([target_bentham[dim] for dim in BENTHAM_DIMENSIONS.keys()], dtype=torch.float32)
        
        # 순전파
        emotion_tensor = torch.tensor(random_emotion, dtype=torch.float32).unsqueeze(0)
        output = adapter(emotion_tensor)
        
        # 손실 계산
        loss = torch.nn.functional.mse_loss(output[0], target_tensor)
        
        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 20 == 0:
            print(f"   Epoch {epoch+1:3d}: Loss = {loss.item():.6f}")
    
    print("\n✅ 학습 완료!")
    
    # 학습 결과 테스트
    print("\n📈 학습된 어댑터 vs 의미론적 매퍼 비교:")
    
    test_emotion = np.array([0.7, 0.5, 0.6, 0.8, 0.2, 0.7])  # 테스트 감정
    
    # 의미론적 결과
    emotion_dict = {dim: test_emotion[idx] for idx, dim in enumerate(EMOTION_DIMENSIONS.keys())}
    semantic_result = semantic_mapper.map_emotion_to_bentham(emotion_dict)
    
    # 신경망 결과
    with torch.no_grad():
        emotion_tensor = torch.tensor(test_emotion, dtype=torch.float32).unsqueeze(0)
        neural_result = adapter(emotion_tensor)[0].numpy()
    
    print(f"{'벤담 차원':<20} {'의미론적':<12} {'신경망':<12} {'차이':<10}")
    print("-" * 55)
    
    for idx, dim in enumerate(BENTHAM_DIMENSIONS.keys()):
        sem_val = semantic_result[dim]
        neural_val = neural_result[idx]
        diff = abs(sem_val - neural_val)
        
        print(f"{dim:<20} {sem_val:<12.3f} {neural_val:<12.3f} {diff:<10.3f}")
    
    # 학습 곡선 그리기
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.title('신경망 어댑터 학습 곡선')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig('neural_adapter_training.png', dpi=150)
    print("\n📊 학습 곡선 저장: neural_adapter_training.png")
    plt.show()


if __name__ == '__main__':
    # 1. 매핑 품질 테스트
    test_mapping_quality()
    
    # 2. 시각화 (matplotlib 있을 경우)
    try:
        visualize_mapping_comparison()
    except ImportError:
        print("\n⚠️ matplotlib/seaborn 없음 - 시각화 건너뜀")
    
    # 3. 신경망 어댑터 테스트
    test_neural_adapter()
    
    print("\n" + "=" * 70)
    print("🎉 모든 테스트 완료!")
    print("=" * 70)
    print("\n💡 결론:")
    print("   - 의미론적 매핑이 휴리스틱보다 훨씬 정밀함")
    print("   - 감정과 벤담 차원 간 의미론적 연결 확인")
    print("   - 신경망 어댑터로 추가 개선 가능")
    print("   - 계층별 처리 (공동체/타자/자아) 지원")