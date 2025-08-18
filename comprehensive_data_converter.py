#!/usr/bin/env python3
"""
종합 데이터 변환기 - Red Heart 학습용 데이터 변환
Comprehensive Data Converter for Red Heart Learning System

모든 유형의 학습 데이터를 Red Heart 시스템이 이해할 수 있는 형태로 변환
손실 최소화하면서 대용량 데이터는 여러 파일로 분할
"""

import os
import json
import re
import uuid
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import asyncio
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict
import math

# 프로젝트 모듈
from config import ADVANCED_CONFIG, PROCESSED_DATASETS_DIR
from data_models import (
    EthicalSituation, DecisionScenario, EmotionData, EmotionState, 
    HedonicValues, BenthamVariable, IntentionCategory
)

logger = logging.getLogger('RedHeart.DataConverter')

@dataclass
class ConversionStats:
    """변환 통계"""
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    total_scenarios: int = 0
    split_files: int = 0
    data_loss_percentage: float = 0.0
    
class DataType:
    """데이터 유형 상수"""
    EBS_KOREAN_LITERATURE = "ebs_korean_literature"
    SCRUPLES_ANECDOTES = "scruples_anecdotes"
    SCRUPLES_DILEMMAS = "scruples_dilemmas"
    CLASSIC_LITERATURE = "classic_literature"
    AI_GENERATED = "ai_generated"
    UNKNOWN = "unknown"

class ComprehensiveDataConverter:
    """종합 데이터 변환기"""
    
    def __init__(self):
        self.stats = ConversionStats()
        self.max_scenarios_per_file = 1000  # 파일당 최대 시나리오 수
        self.max_file_size_mb = 50  # 최대 파일 크기 (MB)
        
        # 감정 키워드 매핑 (사용 가능한 EmotionState만)
        self.emotion_keywords = {
            '슬픔': EmotionState.SADNESS,
            '분노': EmotionState.ANGER,
            '애처로움': EmotionState.SADNESS,
            '씁쓸함': EmotionState.DISGUST,
            '기쁨': EmotionState.JOY,
            '행복': EmotionState.JOY,
            '두려움': EmotionState.FEAR,
            '놀라움': EmotionState.SURPRISE,
            '혐오': EmotionState.DISGUST,
            '사랑': EmotionState.JOY,  # LOVE가 없으므로 JOY로 매핑
            '죄책감': EmotionState.SADNESS,  # GUILT가 없으므로 SADNESS로 매핑
            '수치심': EmotionState.SADNESS,  # SHAME이 없으므로 SADNESS로 매핑
            '자부심': EmotionState.JOY,  # PRIDE가 없으므로 JOY로 매핑
            '질투': EmotionState.ANGER  # ENVY가 없으므로 ANGER로 매핑
        }
        
    async def convert_all_datasets(self, source_dir: str) -> Dict[str, Any]:
        """모든 데이터셋 변환"""
        logger.info(f"🚀 종합 데이터 변환 시작: {source_dir}")
        
        source_path = Path(source_dir)
        if not source_path.exists():
            raise FileNotFoundError(f"소스 디렉토리가 존재하지 않습니다: {source_dir}")
        
        conversion_results = {}
        
        # 1. EBS 한국 문학 데이터 변환
        ebs_dir = source_path / "ai_ebs"
        if ebs_dir.exists():
            logger.info("📚 EBS 한국 문학 데이터 변환 중...")
            ebs_result = await self._convert_ebs_data(ebs_dir)
            conversion_results['ebs_korean_literature'] = ebs_result
            
        # 2. Scruples 데이터 변환
        scruples_dir = source_path / "scruples_real_data"
        if scruples_dir.exists():
            logger.info("⚖️ Scruples 윤리 데이터 변환 중...")
            scruples_result = await self._convert_scruples_data(scruples_dir)
            conversion_results['scruples'] = scruples_result
            
        # 3. 고전 문학 데이터 변환
        books_dir = source_path / "book"
        if books_dir.exists():
            logger.info("📖 고전 문학 데이터 변환 중...")
            books_result = await self._convert_classic_literature(books_dir)
            conversion_results['classic_literature'] = books_result
            
        # 4. AI 생성 데이터 변환
        ai_dir = source_path / "ai_generated_dataset"
        if ai_dir.exists():
            logger.info("🤖 AI 생성 데이터 변환 중...")
            ai_result = await self._convert_ai_generated_data(ai_dir)
            conversion_results['ai_generated'] = ai_result
        
        # 5. 변환 통계 저장
        await self._save_conversion_report(conversion_results)
        
        logger.info(f"✅ 모든 데이터 변환 완료: {self.stats.total_scenarios}개 시나리오 생성")
        return conversion_results
    
    async def _convert_ebs_data(self, ebs_dir: Path) -> Dict[str, Any]:
        """EBS 한국 문학 데이터 변환"""
        scenarios = []
        
        for file_path in ebs_dir.glob("*.txt"):
            try:
                logger.info(f"  처리 중: {file_path.name}")
                content = file_path.read_text(encoding='utf-8')
                
                # 작품별로 분리
                works = content.split('ㅡ' * 50)
                
                for work_content in works:
                    if not work_content.strip():
                        continue
                        
                    scenario = await self._parse_korean_literature_work(work_content.strip())
                    if scenario:
                        scenarios.append(scenario)
                        
                self.stats.processed_files += 1
                
            except Exception as e:
                logger.error(f"EBS 파일 처리 실패 {file_path}: {e}")
                self.stats.failed_files += 1
        
        # 파일 분할 및 저장
        return await self._save_scenarios_with_splitting(scenarios, "ebs_korean_literature")
    
    async def _parse_korean_literature_work(self, content: str) -> Optional[DecisionScenario]:
        """한국 문학 작품 파싱"""
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if not lines:
            return None
            
        try:
            # 제목 추출
            title = lines[0] if lines else "한국 문학 작품"
            
            # 주요 정보 추출
            work_data = {}
            for line in lines:
                if line.startswith('상황설명:'):
                    work_data['situation'] = line[5:].strip()
                elif line.startswith('이해관계자:'):
                    work_data['stakeholders'] = [s.strip() for s in line[6:].split(',')]
                elif line.startswith('윤리적 딜레마:'):
                    work_data['ethical_dilemma'] = line[7:].strip()
                elif line.startswith('선택지'):
                    if '선택지1:' in line:
                        work_data['option1'] = line.split('선택지1:')[1].strip()
                    elif '선택지2:' in line:
                        work_data['option2'] = line.split('선택지2:')[1].strip()
                    elif '선택지3:' in line:
                        work_data['option3'] = line.split('선택지3:')[1].strip()
                elif '후회' in line:
                    work_data['regret'] = line.strip()
            
            # 감정 데이터 추출
            emotions = self._extract_emotions_from_korean_text(content)
            
            # DecisionScenario 생성
            scenario = DecisionScenario(
                title=title,
                description=work_data.get('situation', ''),
                context={
                    'source': 'korean_literature',
                    'stakeholders': work_data.get('stakeholders', []),
                    'ethical_dilemma': work_data.get('ethical_dilemma', ''),
                    'emotions': emotions,
                    'regret_info': work_data.get('regret', ''),
                    'cultural_context': 'korean'
                },
                options=[
                    work_data.get('option1', ''),
                    work_data.get('option2', ''),
                    work_data.get('option3', '')
                ],
                metadata={
                    'data_type': DataType.EBS_KOREAN_LITERATURE,
                    'processing_quality': 'high',
                    'emotion_count': len(emotions),
                    'language': 'korean'
                }
            )
            
            self.stats.total_scenarios += 1
            return scenario
            
        except Exception as e:
            logger.error(f"한국 문학 작품 파싱 실패: {e}")
            return None
    
    async def _convert_scruples_data(self, scruples_dir: Path) -> Dict[str, Any]:
        """Scruples 데이터 변환"""
        scenarios = []
        
        # anecdotes와 dilemmas 처리
        for subdir in ['anecdotes', 'dilemmas']:
            subdir_path = scruples_dir / subdir
            if not subdir_path.exists():
                continue
                
            for file_path in subdir_path.glob("*.jsonl"):
                try:
                    logger.info(f"  처리 중: {file_path.name}")
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f, 1):
                            if not line.strip():
                                continue
                                
                            try:
                                data = json.loads(line)
                                scenario = await self._parse_scruples_entry(data, subdir)
                                if scenario:
                                    scenarios.append(scenario)
                                    
                            except json.JSONDecodeError as e:
                                logger.warning(f"JSON 파싱 실패 {file_path}:{line_num}: {e}")
                    
                    self.stats.processed_files += 1
                    
                except Exception as e:
                    logger.error(f"Scruples 파일 처리 실패 {file_path}: {e}")
                    self.stats.failed_files += 1
        
        return await self._save_scenarios_with_splitting(scenarios, "scruples")
    
    async def _parse_scruples_entry(self, data: Dict, subdir: str) -> Optional[DecisionScenario]:
        """Scruples 항목 파싱"""
        try:
            # 기본 정보
            title = data.get('title', 'Ethical Scenario')
            text = data.get('text', '')
            action_desc = data.get('action', {}).get('description', '')
            
            # 라벨 정보
            label = data.get('label', 'UNKNOWN')
            label_scores = data.get('label_scores', {})
            binarized_label = data.get('binarized_label', 'UNKNOWN')
            
            # 감정 추정 (텍스트 기반)
            emotions = self._estimate_emotions_from_english_text(text)
            
            scenario = DecisionScenario(
                title=title,
                description=text,
                context={
                    'source': f'scruples_{subdir}',
                    'action_description': action_desc,
                    'moral_judgment': label,
                    'label_scores': label_scores,
                    'binarized_judgment': binarized_label,
                    'emotions': emotions,
                    'language': 'english',
                    'post_type': data.get('post_type', 'UNKNOWN')
                },
                options=[
                    "Follow through with the action",
                    "Avoid the action", 
                    "Seek alternative approach"
                ],
                metadata={
                    'data_type': DataType.SCRUPLES_ANECDOTES if subdir == 'anecdotes' else DataType.SCRUPLES_DILEMMAS,
                    'processing_quality': 'medium',
                    'original_id': data.get('id', ''),
                    'post_id': data.get('post_id', ''),
                    'moral_complexity': len(label_scores)
                }
            )
            
            self.stats.total_scenarios += 1
            return scenario
            
        except Exception as e:
            logger.error(f"Scruples 항목 파싱 실패: {e}")
            return None
    
    async def _convert_classic_literature(self, books_dir: Path) -> Dict[str, Any]:
        """고전 문학 변환"""
        scenarios = []
        
        for file_path in books_dir.glob("*.txt"):
            try:
                logger.info(f"  처리 중: {file_path.name}")
                content = file_path.read_text(encoding='utf-8')
                
                # 대용량 책을 장별로 분할
                book_scenarios = await self._extract_scenarios_from_book(
                    content, file_path.stem
                )
                scenarios.extend(book_scenarios)
                
                self.stats.processed_files += 1
                
            except Exception as e:
                logger.error(f"고전 문학 처리 실패 {file_path}: {e}")
                self.stats.failed_files += 1
        
        return await self._save_scenarios_with_splitting(scenarios, "classic_literature")
    
    async def _extract_scenarios_from_book(self, content: str, book_title: str) -> List[DecisionScenario]:
        """책에서 윤리적 시나리오 추출"""
        scenarios = []
        
        # 책을 단락별로 분할 (간단한 휴리스틱)
        paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 100]
        
        # 윤리적 갈등이 있을 것 같은 단락들 식별
        ethical_paragraphs = []
        ethical_keywords = [
            'should', 'ought', 'right', 'wrong', 'moral', 'decision', 'choice',
            'dilemma', 'conflict', 'conscience', 'duty', 'responsibility'
        ]
        
        for para in paragraphs[:50]:  # 처음 50개 단락만 처리 (성능상)
            if any(keyword in para.lower() for keyword in ethical_keywords):
                ethical_paragraphs.append(para)
        
        # 선별된 단락들을 시나리오로 변환
        for i, para in enumerate(ethical_paragraphs[:10]):  # 최대 10개
            try:
                scenario = DecisionScenario(
                    title=f"{book_title} - Ethical Scenario {i+1}",
                    description=para[:1000] + "..." if len(para) > 1000 else para,
                    context={
                        'source': 'classic_literature',
                        'book_title': book_title,
                        'chapter_estimate': i + 1,
                        'language': 'english',
                        'literary_period': self._estimate_literary_period(book_title)
                    },
                    options=[
                        "Follow moral duty",
                        "Choose personal benefit", 
                        "Seek compromise solution"
                    ],
                    metadata={
                        'data_type': DataType.CLASSIC_LITERATURE,
                        'processing_quality': 'low',  # 자동 추출이므로 품질 낮음
                        'extraction_method': 'keyword_based',
                        'paragraph_length': len(para)
                    }
                )
                scenarios.append(scenario)
                self.stats.total_scenarios += 1
                
            except Exception as e:
                logger.warning(f"책 시나리오 생성 실패: {e}")
        
        return scenarios
    
    async def _convert_ai_generated_data(self, ai_dir: Path) -> Dict[str, Any]:
        """AI 생성 데이터 변환"""
        scenarios = []
        
        for file_path in ai_dir.glob("*.txt"):
            try:
                logger.info(f"  처리 중: {file_path.name}")
                content = file_path.read_text(encoding='utf-8')
                
                # AI 생성 데이터는 구조화되어 있다고 가정
                ai_scenarios = await self._parse_ai_generated_scenarios(content)
                scenarios.extend(ai_scenarios)
                
                self.stats.processed_files += 1
                
            except Exception as e:
                logger.error(f"AI 생성 데이터 처리 실패 {file_path}: {e}")
                self.stats.failed_files += 1
        
        return await self._save_scenarios_with_splitting(scenarios, "ai_generated")
    
    async def _parse_ai_generated_scenarios(self, content: str) -> List[DecisionScenario]:
        """AI 생성 시나리오 파싱"""
        scenarios = []
        
        # "사례"로 시작하는 부분들을 찾아 분할
        cases = re.split(r'사례 \d+:', content)
        
        for i, case_content in enumerate(cases[1:], 1):  # 첫 번째는 보통 빈 내용
            try:
                lines = [line.strip() for line in case_content.split('\n') if line.strip()]
                if not lines:
                    continue
                
                # 제목과 상황 설명 추출
                title = f"AI 생성 시나리오 {i}"
                situation = ""
                options = []
                
                for line in lines:
                    if line.startswith('상황 설명'):
                        situation = line.split(':', 1)[1].strip() if ':' in line else line
                    elif line.startswith('선택 가능한 옵션'):
                        # 다음 몇 줄이 옵션들
                        continue
                    elif re.match(r'^\d+\.', line):
                        options.append(line)
                
                if situation:
                    scenario = DecisionScenario(
                        title=title,
                        description=situation,
                        context={
                            'source': 'ai_generated',
                            'generation_method': 'claude_ai',
                            'language': 'korean',
                            'complexity_level': 'medium'
                        },
                        options=options[:5],  # 최대 5개 옵션
                        metadata={
                            'data_type': DataType.AI_GENERATED,
                            'processing_quality': 'high',
                            'case_number': i,
                            'options_count': len(options)
                        }
                    )
                    scenarios.append(scenario)
                    self.stats.total_scenarios += 1
                    
            except Exception as e:
                logger.warning(f"AI 시나리오 파싱 실패 (사례 {i}): {e}")
        
        return scenarios
    
    async def _save_scenarios_with_splitting(self, scenarios: List[DecisionScenario], dataset_name: str) -> Dict[str, Any]:
        """시나리오들을 분할하여 저장"""
        if not scenarios:
            return {'files_created': 0, 'total_scenarios': 0}
        
        # 출력 디렉토리 생성
        output_dir = PROCESSED_DATASETS_DIR / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 파일 분할 계산
        total_scenarios = len(scenarios)
        files_needed = math.ceil(total_scenarios / self.max_scenarios_per_file)
        
        logger.info(f"  📦 {total_scenarios}개 시나리오를 {files_needed}개 파일로 분할")
        
        created_files = []
        
        for file_idx in range(files_needed):
            start_idx = file_idx * self.max_scenarios_per_file
            end_idx = min(start_idx + self.max_scenarios_per_file, total_scenarios)
            
            file_scenarios = scenarios[start_idx:end_idx]
            
            # 파일명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{dataset_name}_batch_{file_idx+1:03d}_of_{files_needed:03d}_{timestamp}.json"
            file_path = output_dir / filename
            
            # 시나리오들을 딕셔너리로 변환 (datetime 처리)
            scenarios_dicts = []
            for scenario in file_scenarios:
                scenario_dict = asdict(scenario)
                # datetime 객체를 ISO 형식 문자열로 변환
                if 'created_at' in scenario_dict:
                    scenario_dict['created_at'] = scenario_dict['created_at'].isoformat()
                scenarios_dicts.append(scenario_dict)
            
            scenarios_data = {
                'metadata': {
                    'dataset_name': dataset_name,
                    'batch_info': f"{file_idx+1}/{files_needed}",
                    'scenario_count': len(file_scenarios),
                    'creation_time': datetime.now().isoformat(),
                    'converter_version': '1.0'
                },
                'scenarios': scenarios_dicts
            }
            
            # JSON 파일로 저장
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(scenarios_data, f, ensure_ascii=False, indent=2)
            
            created_files.append(str(file_path))
            logger.info(f"    ✅ 저장: {filename} ({len(file_scenarios)}개 시나리오)")
        
        self.stats.split_files += len(created_files)
        
        return {
            'files_created': len(created_files),
            'total_scenarios': total_scenarios,
            'files_list': created_files,
            'average_scenarios_per_file': total_scenarios / len(created_files)
        }
    
    def _extract_emotions_from_korean_text(self, text: str) -> Dict[str, float]:
        """한국어 텍스트에서 감정 추출"""
        emotions = {}
        
        # 감정 점수가 명시된 경우
        emotion_pattern = r'(\w+):\s*(\d+)'
        matches = re.findall(emotion_pattern, text)
        
        for emotion_name, score in matches:
            if emotion_name in self.emotion_keywords:
                emotion_state = self.emotion_keywords[emotion_name]
                emotions[emotion_state.name] = float(score) / 10.0  # 0-1 범위로 정규화
        
        return emotions
    
    def _estimate_emotions_from_english_text(self, text: str) -> Dict[str, float]:
        """영어 텍스트에서 감정 추정 (간단한 키워드 기반)"""
        emotions = {}
        
        # 간단한 감정 키워드 매핑
        emotion_keywords_en = {
            'angry': (EmotionState.ANGER, 0.7),
            'sad': (EmotionState.SADNESS, 0.7),
            'happy': (EmotionState.JOY, 0.7),
            'scared': (EmotionState.FEAR, 0.7),
            'disgusted': (EmotionState.DISGUST, 0.7),
            'surprised': (EmotionState.SURPRISE, 0.7),
            'frustrated': (EmotionState.ANGER, 0.5),
            'worried': (EmotionState.FEAR, 0.5),
            'disappointed': (EmotionState.SADNESS, 0.5),
            'excited': (EmotionState.JOY, 0.6)
        }
        
        text_lower = text.lower()
        for keyword, (emotion_state, intensity) in emotion_keywords_en.items():
            if keyword in text_lower:
                emotions[emotion_state.name] = intensity
        
        return emotions
    
    def _estimate_literary_period(self, book_title: str) -> str:
        """책 제목으로 문학 시대 추정"""
        title_lower = book_title.lower()
        
        if any(classic in title_lower for classic in ['frankenstein', 'dracula']):
            return 'gothic_romantic'
        elif any(classic in title_lower for classic in ['gatsby', 'mockingbird']):
            return 'modern_american'
        elif any(classic in title_lower for classic in ['shakespeare', 'romeo']):
            return 'elizabethan'
        elif any(classic in title_lower for classic in ['dickens', 'austen']):
            return 'victorian'
        else:
            return 'unknown'
    
    async def _save_conversion_report(self, results: Dict[str, Any]) -> None:
        """변환 보고서 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = PROCESSED_DATASETS_DIR / f"conversion_report_{timestamp}.json"
        
        # 통계 계산
        total_files_created = sum(r.get('files_created', 0) for r in results.values())
        total_scenarios = sum(r.get('total_scenarios', 0) for r in results.values())
        
        report = {
            'conversion_summary': {
                'timestamp': datetime.now().isoformat(),
                'total_datasets': len(results),
                'total_files_processed': self.stats.processed_files,
                'total_files_failed': self.stats.failed_files,
                'total_scenarios_created': total_scenarios,
                'total_output_files': total_files_created,
                'success_rate': (self.stats.processed_files / (self.stats.processed_files + self.stats.failed_files)) * 100 if (self.stats.processed_files + self.stats.failed_files) > 0 else 0
            },
            'dataset_details': results,
            'processing_stats': asdict(self.stats)
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📊 변환 보고서 저장: {report_path}")
        logger.info(f"✅ 변환 완료: {total_scenarios}개 시나리오, {total_files_created}개 파일 생성")

async def main():
    """메인 실행 함수"""
    logger.info("🚀 Red Heart 종합 데이터 변환기 시작")
    
    converter = ComprehensiveDataConverter()
    
    try:
        # 소스 데이터 디렉토리
        source_dir = "/mnt/d/large_prj/linux_red_heart/for_learn_dataset"
        
        # 변환 실행
        results = await converter.convert_all_datasets(source_dir)
        
        print("\n" + "="*60)
        print("🎉 데이터 변환 완료!")
        print("="*60)
        print(f"📊 처리된 파일: {converter.stats.processed_files}")
        print(f"❌ 실패한 파일: {converter.stats.failed_files}")
        print(f"🎯 생성된 시나리오: {converter.stats.total_scenarios}")
        print(f"📁 출력 파일: {converter.stats.split_files}")
        print("="*60)
        
        return results
        
    except Exception as e:
        logger.error(f"데이터 변환 실패: {e}")
        raise

if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    )
    
    # 변환 실행
    asyncio.run(main())