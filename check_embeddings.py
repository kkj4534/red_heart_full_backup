#!/usr/bin/env python3
"""빠른 임베딩 상태 확인 스크립트"""

import json
import sys

print("임베딩 파일 분석 중...")

with open('/mnt/c/large_project/linux_red_heart/claude_api_preprocessing/claude_preprocessed_complete.embedded.json', 'r') as f:
    data = json.load(f)

total = len(data)
embedded = sum(1 for item in data if item.get('embedding') is not None)

# 마지막 임베딩 인덱스
last_idx = -1
for i in range(len(data)-1, -1, -1):
    if data[i].get('embedding') is not None:
        last_idx = i
        break

print(f'전체 데이터: {total:,}개')
print(f'임베딩 완료: {embedded:,}개')
print(f'마지막 임베딩 인덱스: {last_idx}')
print(f'진행률: {embedded/total*100:.2f}%')

# 다음 시작 지점
next_idx = last_idx + 1 if last_idx >= 0 else 0
print(f'\n다음 시작 인덱스: {next_idx}')
print(f'남은 데이터: {total - embedded:,}개')