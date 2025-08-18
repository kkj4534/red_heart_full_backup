#!/usr/bin/env python3

"""
빠른 감정 분석 테스트 - 핵심 문제 진단
"""

import sys
sys.path.append('.')

from advanced_emotion_analyzer import AdvancedEmotionAnalyzer
from data_models import EmotionState

def test_keyword_analysis():
    """키워드 분석만 직접 테스트"""
    
    analyzer = AdvancedEmotionAnalyzer()
    test_text = '매우 걱정되고 화가 납니다'
    
    print(f'=== 키워드 분석 직접 테스트 ===')
    print(f'텍스트: "{test_text}"')
    
    # 키워드 매칭 직접 확인
    text_lower = test_text.lower()
    emotion_scores = {}
    
    for emotion_id, keywords_dict in analyzer.korean_emotion_keywords.items():
        total_score = 0
        matches = []
        
        # Primary 키워드
        for keyword in keywords_dict['primary']:
            if keyword in text_lower:
                total_score += 1.0
                matches.append(f'primary:{keyword}')
        
        # Intensity 수식어
        intensity_multiplier = 1.0
        for modifier in keywords_dict['intensity']:
            if modifier in text_lower:
                intensity_multiplier += 0.5
                matches.append(f'intensity:{modifier}')
        
        final_score = total_score * intensity_multiplier
        if final_score > 0:
            emotion_scores[emotion_id] = final_score
            emotion_name = analyzer._emotion_id_to_name(emotion_id) if hasattr(analyzer, '_emotion_id_to_name') else str(emotion_id)
            print(f'감정 #{emotion_id} ({emotion_name}): {final_score} - 매치: {matches}')
    
    if emotion_scores:
        best_emotion_id = max(emotion_scores, key=emotion_scores.get)
        best_score = emotion_scores[best_emotion_id]
        
        print(f'\n🏆 최고 점수: 감정 #{best_emotion_id} = {best_score}')
        
        try:
            emotion_state = EmotionState(best_emotion_id)
            print(f'📌 EmotionState: {emotion_state}')
            
            # 간단한 감정 데이터 생성
            from data_models import EmotionData, EmotionIntensity
            
            confidence = min(0.95, best_score / 5.0)
            
            result = EmotionData(
                primary_emotion=emotion_state,
                confidence=confidence,
                intensity=EmotionIntensity.MODERATE,
                language="ko",
                processing_method="direct_keyword_test"
            )
            
            print(f'\n✅ 직접 생성 결과:')
            print(f'Primary: {result.primary_emotion}')
            print(f'Confidence: {result.confidence:.3f}')
            print(f'Intensity: {result.intensity}')
            
        except Exception as e:
            print(f'❌ EmotionState 변환 오류: {e}')
    
    else:
        print('❌ 매칭된 감정이 없습니다.')

if __name__ == "__main__":
    test_keyword_analysis()