#!/usr/bin/env python3
import json

# 데이터 로드
with open('preprocessed_dataset_v2.json', 'r') as f:
    data = json.load(f)

print(f'총 샘플 수: {len(data)}')
print(f'첫 번째 샘플 text 길이: {len(data[0]["text"])}')
print(f'context_embedding 길이: {len(data[0]["context_embedding"])}')

# context_embedding이 0이 아닌 샘플이 있는지 확인
non_zero_count = 0
for item in data:
    if any(v != 0.0 for v in item['context_embedding']):
        non_zero_count += 1

print(f'context_embedding이 0이 아닌 샘플: {non_zero_count}/{len(data)}')

# bentham_scores 확인
unique_bentham = set()
for item in data:
    unique_bentham.add(str(item['bentham_scores']))
print(f'고유한 bentham_scores 패턴: {len(unique_bentham)}개')

# 첫 5개 샘플의 텍스트 확인
print('\n첫 5개 샘플의 텍스트:')
for i, item in enumerate(data[:5]):
    text = item.get('text', '')
    print(f'  {i+1}. "{text[:50]}..." (길이: {len(text)})')