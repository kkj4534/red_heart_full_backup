import traceback
import logging

# main_unified.py의 analyze 메서드를 시뮬레이션
class TestUnified:
    def __init__(self):
        self.logger = logging.getLogger('Test')
        
    def analyze(self):
        try:
            # 1. UnifiedModel이 dict 반환 (정상)
            emotion_outputs = {
                'head': None,  # tensor일 수도 있음
                'advanced': None,  # tensor일 수도 있음
                'neural': None,
                'dsp': None
            }
            
            # 2. _process_unified_outputs 호출 (정상)
            results = {'unified': self._process_unified_outputs(emotion_outputs)}
            
            # 3. 어딘가에서 dict를 tensor로 착각?
            # 예를 들어 emotion_outputs를 다시 사용하려고?
            print(f"emotion_outputs type: {type(emotion_outputs)}")
            
            # 만약 emotion_outputs를 텐서로 착각하고 .to() 호출하면?
            # emotion_outputs.to('cpu')  # <- 이런 코드가 어딘가 있을 것
            
            return results
            
        except Exception as e:
            self.logger.error(f"분석 실패: {e}")
            print(f"\n실제 에러 위치 (traceback):")
            traceback.print_exc()
            return {'error': str(e)}
    
    def _process_unified_outputs(self, outputs):
        # 정상 처리
        return {'processed': True}

# 테스트
test = TestUnified()
result = test.analyze()
print(f"\n결과: {result}")
