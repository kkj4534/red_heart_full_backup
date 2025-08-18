#!/usr/bin/env python3
"""
Scruples 데이터셋 준비 스크립트
- anecdotes와 dilemmas 데이터 통합
- Claude API 전처리용 형식으로 변환
"""

import json
import random
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Scruples.Prepare')

def load_scruples_data(max_samples: int = 20000):
    """Scruples 데이터 로드 및 변환"""
    
    base_path = Path("../for_learn_dataset/scruples_real_data")
    
    # 수집할 샘플들
    all_samples = []
    
    # 1. Anecdotes (AITA 포스트)
    anecdotes_path = base_path / "anecdotes" / "train.scruples-anecdotes.jsonl"
    if anecdotes_path.exists():
        logger.info(f"Anecdotes 로드: {anecdotes_path}")
        with open(anecdotes_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_samples // 2:  # 절반은 anecdotes
                    break
                
                data = json.loads(line)
                # action 안전 처리
                action_desc = ''
                if data.get('action'):
                    action_desc = data['action'].get('description', '')
                
                sample = {
                    'id': f"anecdote_{data['id']}",
                    'text': data['text'],  # AITA 포스트 본문
                    'title': data.get('title', ''),
                    'action': action_desc,
                    'label': data.get('label', ''),
                    'type': 'anecdote'
                }
                all_samples.append(sample)
        
        logger.info(f"Anecdotes 로드: {len(all_samples)}개")
    
    # 2. Dilemmas (윤리적 딜레마)
    dilemmas_path = base_path / "dilemmas" / "train.scruples-dilemmas.jsonl"
    if dilemmas_path.exists():
        logger.info(f"Dilemmas 로드: {dilemmas_path}")
        current_count = len(all_samples)
        
        with open(dilemmas_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if len(all_samples) >= max_samples:
                    break
                
                data = json.loads(line)
                
                # 두 액션을 하나의 텍스트로 결합
                action1 = data.get('actions', [{}])[0].get('description', '')
                action2 = data.get('actions', [{}])[1].get('description', '') if len(data.get('actions', [])) > 1 else ''
                
                text = f"I need to choose between: {action1} OR {action2}"
                if data.get('context'):
                    text = f"Context: {data['context']}. {text}"
                
                sample = {
                    'id': f"dilemma_{data['id']}",
                    'text': text,
                    'context': data.get('context', ''),
                    'actions': data.get('actions', []),
                    'label': data.get('label', ''),
                    'type': 'dilemma'
                }
                all_samples.append(sample)
        
        logger.info(f"Dilemmas 로드: {len(all_samples) - current_count}개")
    
    # 3. 셔플
    random.shuffle(all_samples)
    
    # 4. 저장
    output_path = Path("scruples_prepared.jsonl")
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info(f"✅ 총 {len(all_samples)}개 샘플 저장: {output_path}")
    
    # 통계
    anecdote_count = sum(1 for s in all_samples if s['type'] == 'anecdote')
    dilemma_count = sum(1 for s in all_samples if s['type'] == 'dilemma')
    
    logger.info(f"통계: Anecdotes {anecdote_count}개, Dilemmas {dilemma_count}개")
    
    # 샘플 미리보기
    logger.info("\n=== 샘플 미리보기 ===")
    for i in range(min(3, len(all_samples))):
        sample = all_samples[i]
        logger.info(f"\n샘플 {i+1} ({sample['type']}):")
        logger.info(f"ID: {sample['id']}")
        logger.info(f"Text: {sample['text'][:200]}...")

if __name__ == "__main__":
    load_scruples_data(max_samples=20000)