#!/usr/bin/env python3
"""
main_unified.py에서 bentham_head를 사용하도록 수정하는 패치
"""

# main_unified.py의 analyze() 함수에서 수정해야 할 부분:

# 현재 코드 (1051-1056줄):
"""
outputs = self.unified_model(
    x=inputs['embeddings'],
    task='emotion',  # 기본 태스크
    return_all=True
)
results['unified'] = self._process_unified_outputs(outputs)
"""

# 수정 코드:
"""
# emotion과 bentham 둘 다 실행
emotion_outputs = self.unified_model(
    x=inputs['embeddings'],
    task='emotion',
    return_all=True
)
results['unified_emotion'] = self._process_unified_outputs(emotion_outputs, task='emotion')

# bentham_head 사용 - 학습된 가중치 활용
bentham_outputs = self.unified_model(
    x=inputs['embeddings'],
    task='bentham',  # bentham 태스크로 실행
    return_all=True
)
results['unified_bentham'] = self._process_unified_outputs(bentham_outputs, task='bentham')
"""

# _process_unified_outputs 함수도 수정 필요:
"""
def _process_unified_outputs(self, outputs: Dict, task: str = 'emotion') -> Dict:
    processed = {}
    
    if 'head' in outputs and outputs['head'] is not None:
        head_output = outputs['head']
        
        if task == 'emotion':
            # 기존 emotion 처리
            if isinstance(head_output, torch.Tensor):
                emotion_names = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'love']
                scores = head_output.softmax(dim=-1)[0].tolist()
                emotion_dict = {name: scores[i] for i, name in enumerate(emotion_names[:len(scores)])}
                processed['emotion'] = emotion_dict
                processed['emotion']['scores'] = scores
                
        elif task == 'bentham':
            # bentham 처리 추가
            if isinstance(head_output, torch.Tensor):
                # bentham_head는 10개 요소 출력
                bentham_elements = [
                    'intensity', 'duration', 'certainty', 'propinquity',
                    'fecundity', 'purity', 'extent', 
                    'pleasure_total', 'pain_total', 'net_pleasure'
                ]
                
                scores = head_output[0].tolist() if head_output.dim() > 1 else head_output.tolist()
                bentham_dict = {name: scores[i] for i, name in enumerate(bentham_elements[:len(scores)])}
                processed['bentham'] = bentham_dict
                
                # 전체 쾌락 점수 계산
                if len(scores) >= 10:
                    processed['bentham']['final_score'] = scores[9]  # net_pleasure
                else:
                    # 간단한 계산
                    processed['bentham']['final_score'] = sum(scores[:7]) / 7.0
    
    # 다른 출력들 처리
    if 'dsp' in outputs and outputs['dsp'] is not None:
        if isinstance(outputs['dsp'], dict) and 'final_emotions' in outputs['dsp']:
            processed['dsp_emotions'] = outputs['dsp']['final_emotions'].tolist()
    
    if 'neural' in outputs and outputs['neural'] is not None:
        processed['neural_analysis'] = outputs['neural']
    
    if 'wrapper' in outputs and outputs['wrapper'] is not None:
        processed['wrapper_analysis'] = outputs['wrapper']
    
    return processed
"""

print("수정 방법:")
print("1. main_unified.py의 analyze() 함수에서 UnifiedModel을 task='bentham'으로도 호출")
print("2. _process_unified_outputs() 함수에 bentham 처리 추가")
print("3. 이렇게 하면 학습된 bentham_head (27M 파라미터)를 사용하게 됨")
print("\n경험 DB 비활성화:")
print("config.use_experience_database = False로 설정")