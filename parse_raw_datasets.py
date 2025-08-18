#!/usr/bin/env python3
"""
Red Heart AI 원본 데이터셋 파싱 유틸리티
다양한 형식의 비정형 데이터를 통합 형식으로 변환
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetParser:
    """다양한 형식의 데이터셋 파서"""
    
    def __init__(self, base_path: str = "for_learn_dataset"):
        self.base_path = Path(base_path)
        self.parsed_data = []
    
    def parse_ebs_format(self, file_path: Path) -> List[Dict]:
        """EBS 문학 데이터 파싱"""
        results = []
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 작품별로 분리 (구분자: 긴 대시 라인)
        works = re.split(r'ㅡ{40,}', content)
        
        for work in works:
            if not work.strip():
                continue
            
            lines = work.strip().split('\n')
            data = {
                'source': 'ebs_literature',
                'type': 'korean_literature',
                'text': '',
                'metadata': {}
            }
            
            current_field = None
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 필드 추출
                if line.endswith(')') and '(' in line and not ':' in line:
                    # 작품명
                    data['title'] = line
                elif line.startswith('상황설명:'):
                    current_field = 'situation'
                    data['text'] = line.replace('상황설명:', '').strip()
                elif line.startswith('이해관계자:'):
                    data['metadata']['stakeholders'] = line.replace('이해관계자:', '').strip()
                elif line.startswith('윤리적 딜레마:'):
                    data['metadata']['ethical_dilemma'] = line.replace('윤리적 딜레마:', '').strip()
                elif line.startswith('주요 감정:'):
                    current_field = 'emotions'
                elif line.startswith('감정 원인:'):
                    current_field = 'emotion_causes'
                elif line.startswith('선택지'):
                    current_field = 'choices'
                elif line.startswith('덕목윤리:'):
                    data['metadata']['virtue_ethics'] = line.replace('덕목윤리:', '').strip()
                elif line.startswith('의무윤리:'):
                    data['metadata']['duty_ethics'] = line.replace('의무윤리:', '').strip()
                elif line.startswith('결과윤리:'):
                    data['metadata']['consequence_ethics'] = line.replace('결과윤리:', '').strip()
                else:
                    # 감정 점수 파싱
                    if current_field == 'emotions' and ':' in line:
                        emotion, score = line.split(':')
                        if 'emotions' not in data['metadata']:
                            data['metadata']['emotions'] = {}
                        try:
                            data['metadata']['emotions'][emotion.strip()] = int(score.strip())
                        except:
                            pass
            
            if data['text']:  # 텍스트가 있는 경우만 추가
                results.append(data)
        
        return results
    
    def parse_claude_format(self, file_path: Path) -> List[Dict]:
        """Claude가 생성한 데이터 파싱"""
        results = []
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 사례별로 분리
        cases = re.split(r'사례 \d+:', content)
        
        for case in cases[1:]:  # 첫 번째는 빈 문자열
            if not case.strip():
                continue
            
            data = {
                'source': 'ai_generated',
                'type': 'ethical_dilemma',
                'text': '',
                'metadata': {}
            }
            
            # 제목 추출
            title_match = re.search(r'상황 제목:\s*(.+)', case)
            if title_match:
                data['title'] = title_match.group(1).strip()
            
            # 상황 설명 추출
            desc_match = re.search(r'상황 설명:\s*(.+?)(?=관련된 핵심|$)', case, re.DOTALL)
            if desc_match:
                data['text'] = desc_match.group(1).strip()
            
            # 핵심 가치 추출
            values_match = re.search(r'관련된 핵심 가치/원칙:\s*(.+?)(?=\d+\.|$)', case, re.DOTALL)
            if values_match:
                values = values_match.group(1).strip()
                data['metadata']['core_values'] = [v.strip() for v in values.split('\n') if v.strip()]
            
            # 감정 상태 추출
            emotion_match = re.search(r'주요 감정:\s*(.+)', case)
            if emotion_match:
                data['metadata']['emotions'] = emotion_match.group(1).strip()
            
            emotion_intensity_match = re.search(r'감정 강도:\s*(\d+)/10', case)
            if emotion_intensity_match:
                data['metadata']['emotion_intensity'] = int(emotion_intensity_match.group(1))
            
            if data['text']:
                results.append(data)
        
        return results
    
    def parse_scruples_format(self, file_path: Path) -> List[Dict]:
        """Scruples JSONL 데이터 파싱"""
        results = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    data = {
                        'source': 'scruples',
                        'type': 'moral_judgment',
                        'title': item.get('title', ''),
                        'text': item.get('text', ''),
                        'metadata': {
                            'action': item.get('action', {}),
                            'label': item.get('label', ''),
                            'binarized_label': item.get('binarized_label', ''),
                            'post_type': item.get('post_type', '')
                        }
                    }
                    
                    if data['text']:
                        results.append(data)
                except:
                    continue
        
        return results
    
    def parse_all_datasets(self) -> List[Dict]:
        """모든 데이터셋 파싱"""
        all_data = []
        
        # EBS 데이터
        ebs_dir = self.base_path / "ai_ebs"
        if ebs_dir.exists():
            for file in ebs_dir.glob("*.txt"):
                logger.info(f"파싱 중: {file}")
                data = self.parse_ebs_format(file)
                all_data.extend(data)
                logger.info(f"  - {len(data)}개 샘플 추출")
        
        # AI 생성 데이터
        ai_dir = self.base_path / "ai_generated_dataset"
        if ai_dir.exists():
            for file in ai_dir.glob("*.txt"):
                logger.info(f"파싱 중: {file}")
                data = self.parse_claude_format(file)
                all_data.extend(data)
                logger.info(f"  - {len(data)}개 샘플 추출")
        
        # Scruples 데이터
        scruples_dir = self.base_path / "scruples_real_data"
        if scruples_dir.exists():
            for subdir in ['anecdotes', 'dilemmas']:
                sub_path = scruples_dir / subdir
                if sub_path.exists():
                    for file in sub_path.glob("*.jsonl"):
                        if 'train' in file.name:  # 학습 데이터만
                            logger.info(f"파싱 중: {file}")
                            data = self.parse_scruples_format(file)
                            # 더 많은 데이터 사용
                            sample_size = min(1000, len(data))  # 최대 1000개
                            all_data.extend(data[:sample_size])
                            logger.info(f"  - {sample_size}개 샘플 추출")
        
        logger.info(f"총 {len(all_data)}개 샘플 파싱 완료")
        return all_data
    
    def save_parsed_data(self, output_path: str = "parsed_raw_datasets.json"):
        """파싱된 데이터 저장"""
        data = self.parse_all_datasets()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"데이터 저장 완료: {output_path}")
        return data

if __name__ == "__main__":
    parser = DatasetParser()
    data = parser.save_parsed_data()
    
    # 통계 출력
    print(f"\n📊 데이터셋 통계:")
    print(f"  - 총 샘플 수: {len(data)}")
    
    sources = {}
    for item in data:
        source = item.get('source', 'unknown')
        sources[source] = sources.get(source, 0) + 1
    
    print(f"  - 소스별 분포:")
    for source, count in sources.items():
        print(f"    • {source}: {count}개")