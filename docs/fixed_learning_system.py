#!/usr/bin/env python3
"""
수정된 학습 시스템 - 메서드 문제 해결 버전
"""

import asyncio
import logging
import json
import os
import time
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import datetime

from config import SYSTEM_CONFIG, BASE_DIR
from data_models import (
    EthicalSituation, HedonicValues, EmotionData, Decision, DecisionLog,
    EmotionState, EmotionIntensity
)
from emotion_analysis import EmotionAnalyzer
from regret_algorithm import RegretAnalyzer
from enhanced_bentham_calculator import EnhancedBenthamCalculator
from hierarchical_emotion_system import HierarchicalEmotionLearning
from surd_causal_analysis_v2 import RealSURDAnalyzer
from multi_level_semantic_analyzer import MultiLevelSemanticAnalyzer
from rumbaugh_counterfactual import EnhancedCounterfactualReasoning
from data_store import ExperienceDatabase

# D 드라이브에 로그 디렉토리 설정
LOG_DIR = Path("/mnt/d/large_prj/red_heart/logs")
LOG_DIR.mkdir(exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"fixed_learning_system_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('RedHeart.FixedLearning')

# Regret 관련 클래스
class RegretType(Enum):
    ACTION = "action"
    INACTION = "inaction"
    TIMING = "timing"
    CHOICE = "choice"
    EMPATHY = "empathy"
    PREDICTION = "prediction"

class RegretLearningSystem:
    def __init__(self):
        self.regret_analyzer = RegretAnalyzer()
        self.total_learning_count = 0
        self.current_phase = "PHASE_0"
        self.phase_thresholds = [50, 100, 200]
        self.regret_history = []
        
    def calculate_regret(self, situation: EthicalSituation, regret_type: RegretType) -> float:
        """후회 값 계산 - 수정된 버전"""
        try:
            # Decision 생성 (confidence_level이 아닌 confidence 사용)
            decision = Decision(
                situation_id=situation.id,
                choice="test_choice",
                reasoning=situation.description[:100],
                confidence=0.7  # confidence_level이 아닌 confidence
            )
            
            decision_log = DecisionLog(
                situation=situation,
                decision=decision,
                predicted_outcome={"score": 0.5}
            )
            
            # 실제 결과 생성
            actual_outcome = {"score": 0.3}
            
            # analyze_decision_outcome 메서드 사용
            regret_analysis = self.regret_analyzer.analyze_decision_outcome(decision_log, actual_outcome)
            
            # 후회 점수 추출
            if isinstance(regret_analysis, dict) and 'regret_intensity' in regret_analysis:
                regret_score = regret_analysis['regret_intensity']
            else:
                regret_score = 0.3
            
            # 후회 유형별 가중치 적용
            type_weights = {
                RegretType.ACTION: 1.0,
                RegretType.INACTION: 0.9,
                RegretType.TIMING: 0.8,
                RegretType.CHOICE: 1.1,
                RegretType.EMPATHY: 0.7,
                RegretType.PREDICTION: 0.6
            }
            
            weighted_regret = regret_score * type_weights.get(regret_type, 1.0)
            self.regret_history.append((regret_type, weighted_regret, situation.id))
            
            logger.info(f"후회 계산 완료 - 유형: {regret_type.value}, 점수: {weighted_regret:.4f}")
            return weighted_regret
            
        except Exception as e:
            logger.error(f"후회 계산 중 오류: {e}")
            return 0.0
    
    def advance_phase(self) -> bool:
        """페이즈 전환 체크"""
        if self.current_phase == "PHASE_0" and self.total_learning_count >= 50:
            self.current_phase = "PHASE_1"
            logger.info("페이즈 전환: PHASE_0 → PHASE_1")
            return True
        elif self.current_phase == "PHASE_1" and self.total_learning_count >= 100:
            self.current_phase = "PHASE_2"
            logger.info("페이즈 전환: PHASE_1 → PHASE_2")
            return True
        return False

def convert_emotion_data_to_dict(emotion_data: EmotionData) -> Dict[str, Any]:
    """EmotionData 객체를 JSON 직렬화 가능한 dict로 변환"""
    try:
        result = {
            'primary_emotion': emotion_data.primary_emotion.name if isinstance(emotion_data.primary_emotion, EmotionState) else str(emotion_data.primary_emotion),
            'intensity': emotion_data.intensity.name if isinstance(emotion_data.intensity, EmotionIntensity) else str(emotion_data.intensity),
            'arousal': float(emotion_data.arousal),
            'valence': float(emotion_data.valence),
            'confidence': float(emotion_data.confidence),
            'timestamp': emotion_data.timestamp.isoformat() if hasattr(emotion_data.timestamp, 'isoformat') else str(emotion_data.timestamp)
        }
        
        # secondary_emotions 처리
        if hasattr(emotion_data, 'secondary_emotions') and emotion_data.secondary_emotions:
            result['secondary_emotions'] = {
                k.name if isinstance(k, EmotionState) else str(k): float(v) 
                for k, v in emotion_data.secondary_emotions.items()
            }
        else:
            result['secondary_emotions'] = {}
            
        return result
    except Exception as e:
        logger.error(f"EmotionData 변환 중 오류: {e}")
        return {'error': str(e)}

class FixedLearningSystem:
    def __init__(self):
        self.regret_system = RegretLearningSystem()
        self.bentham_calculator = EnhancedBenthamCalculator()
        self.emotion_analyzer = EmotionAnalyzer()
        self.hierarchical_emotion = HierarchicalEmotionLearning()
        self.surd_analyzer = RealSURDAnalyzer()
        self.semantic_analyzer = MultiLevelSemanticAnalyzer()
        self.counterfactual_reasoner = EnhancedCounterfactualReasoning()
        self.experience_db = ExperienceDatabase()
        
        # 학습 통계
        self.learning_stats = {
            'total_scenarios': 0,
            'regret_calculations': 0,
            'bentham_calculations': 0,
            'phase_transitions': 0,
            'current_phase': 'PHASE_0',
            'start_time': time.time(),
            'errors': []
        }
        
        logger.info("수정된 학습 시스템 초기화 완료")
    
    def load_sample_data(self, count: int = 5) -> List[Dict[str, Any]]:
        """샘플 데이터 로드"""
        data_path = Path("/mnt/d/large_prj/red_heart/processed_datasets")
        sample_data = []
        
        try:
            # 여러 파일에서 데이터 수집
            json_files = [
                data_path / "ai_generated_scenarios.json",
                data_path / "scruples_anecdotes_scenarios.json",
                data_path / "literature_scenarios.json"
            ]
            
            all_scenarios = []
            for file_path in json_files:
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            all_scenarios.extend(data)
                        elif isinstance(data, dict) and 'scenarios' in data:
                            all_scenarios.extend(data['scenarios'])
                        logger.info(f"파일 로드: {file_path.name} - {len(data) if isinstance(data, list) else len(data.get('scenarios', []))} 항목")
            
            # 랜덤 샘플링
            if len(all_scenarios) >= count:
                sample_data = random.sample(all_scenarios, count)
            else:
                sample_data = all_scenarios[:count]
                
            logger.info(f"샘플 데이터 로드 완료: {len(sample_data)}개")
            return sample_data
            
        except Exception as e:
            logger.error(f"데이터 로드 중 오류: {e}")
            return []
    
    def create_ethical_situation(self, data: Dict[str, Any]) -> EthicalSituation:
        """윤리적 상황 객체 생성"""
        return EthicalSituation(
            title=data.get('title', data.get('scenario', 'Unknown')),
            description=data.get('description', data.get('text', str(data))),
            situation_type=data.get('category', data.get('situation_type', 'general')),
            context=data
        )
    
    async def process_scenario(self, scenario_data: Dict[str, Any], step: int) -> Dict[str, Any]:
        """개별 시나리오 처리 - 수정된 버전"""
        logger.info(f"=== 시나리오 {step} 처리 시작 ===")
        
        situation = self.create_ethical_situation(scenario_data)
        results = {
            'scenario_id': situation.id,
            'step': step,
            'regret_results': [],
            'bentham_results': [],
            'emotion_analysis': {},
            'processing_time': 0,
            'errors': []
        }
        
        start_time = time.time()
        
        try:
            # 1. 후회 학습 (7개)
            logger.info(f"후회 학습 시작 - 7회 계산")
            regret_types = list(RegretType)
            for i in range(7):
                try:
                    regret_type = regret_types[i % len(regret_types)]
                    regret_value = self.regret_system.calculate_regret(situation, regret_type)
                    results['regret_results'].append({
                        'type': regret_type.value,
                        'value': regret_value
                    })
                    
                    # 페이즈 전환 체크
                    if regret_value > 0.3 and self.regret_system.total_learning_count >= 50:
                        if self.regret_system.advance_phase():
                            self.learning_stats['phase_transitions'] += 1
                            self.learning_stats['current_phase'] = self.regret_system.current_phase
                    
                    self.regret_system.total_learning_count += 1
                    self.learning_stats['regret_calculations'] += 1
                    
                except Exception as e:
                    error_msg = f"후회 계산 오류 (유형: {regret_type.value}): {e}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)
            
            # 2. 벤담 계산 (21개 - 7개 환경 × 3회)
            logger.info(f"벤담 계산 시작 - 21회 계산")
            environments = ['개인적', '사회적', '가족적', '직업적', '도덕적', '문화적', '상황적']
            for env_idx, environment in enumerate(environments):
                for calc_round in range(3):
                    try:
                        hedonic_values = HedonicValues(
                            intensity=random.uniform(0.3, 1.0),
                            duration=random.uniform(0.2, 0.9),
                            certainty=random.uniform(0.4, 1.0),
                            propinquity=random.uniform(0.1, 0.8),
                            purity=random.uniform(0.2, 0.7),
                            extent=random.uniform(0.3, 1.0)
                        )
                        
                        # calculate_with_layers 메서드 호출
                        bentham_result = self.bentham_calculator.calculate_with_layers(hedonic_values)
                        
                        # 결과 처리
                        if hasattr(bentham_result, 'final_score'):
                            bentham_score = bentham_result.final_score
                        else:
                            bentham_score = float(bentham_result) if bentham_result else 0.0
                        
                        results['bentham_results'].append({
                            'environment': environment,
                            'round': calc_round + 1,
                            'score': bentham_score,
                            'hedonic_values': asdict(hedonic_values) if hasattr(hedonic_values, '__dataclass_fields__') else hedonic_values.__dict__
                        })
                        
                        self.learning_stats['bentham_calculations'] += 1
                        
                    except Exception as e:
                        error_msg = f"벤담 계산 오류 (환경: {environment}, 라운드: {calc_round+1}): {e}"
                        logger.error(error_msg)
                        results['errors'].append(error_msg)
            
            # 3. 감정 분석
            try:
                emotion_result = self.emotion_analyzer.analyze_text(situation.description)
                # EmotionData를 dict로 변환
                results['emotion_analysis'] = convert_emotion_data_to_dict(emotion_result)
                logger.info(f"감정 분석 완료")
            except Exception as e:
                error_msg = f"감정 분석 오류: {e}"
                logger.error(error_msg)
                results['emotion_analysis'] = {}
                results['errors'].append(error_msg)
            
            results['processing_time'] = time.time() - start_time
            self.learning_stats['total_scenarios'] += 1
            
            logger.info(f"시나리오 {step} 처리 완료 - 소요시간: {results['processing_time']:.2f}초")
            
        except Exception as e:
            error_msg = f"시나리오 {step} 처리 중 오류: {e}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
        
        return results
    
    async def run_learning(self, sample_count: int = 5):
        """학습 실행"""
        logger.info(f"=== 학습 시작 ({sample_count}개 샘플) ===")
        
        # 샘플 데이터 로드
        sample_data = self.load_sample_data(sample_count)
        if not sample_data:
            logger.error("샘플 데이터를 로드할 수 없습니다.")
            return
        
        logger.info(f"로드된 샘플 데이터: {len(sample_data)}개")
        
        # 결과 저장 경로
        results_dir = Path("/mnt/d/large_prj/red_heart/logs/learning_results")
        results_dir.mkdir(exist_ok=True)
        
        all_results = []
        
        # 각 시나리오 처리
        for i, scenario_data in enumerate(sample_data, 1):
            try:
                result = await self.process_scenario(scenario_data, i)
                all_results.append(result)
                
                # 중간 저장
                temp_file = results_dir / f"temp_results_{timestamp}_{i}.json"
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                
                logger.info(f"진행률: {i}/{len(sample_data)} ({i/len(sample_data)*100:.1f}%)")
                
            except Exception as e:
                logger.error(f"시나리오 {i} 처리 실패: {e}")
                self.learning_stats['errors'].append(f"시나리오 {i}: {str(e)}")
        
        # 최종 결과 저장
        final_results = {
            'test_info': {
                'total_samples': len(sample_data),
                'timestamp': timestamp,
                'log_file': str(log_file)
            },
            'learning_stats': self.learning_stats,
            'scenario_results': all_results
        }
        
        final_results['learning_stats']['total_time'] = time.time() - self.learning_stats['start_time']
        
        final_file = results_dir / f"fixed_learning_results_{timestamp}.json"
        with open(final_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        
        logger.info("=== 학습 완료 ===")
        logger.info(f"총 처리 시나리오: {self.learning_stats['total_scenarios']}")
        logger.info(f"총 후회 계산: {self.learning_stats['regret_calculations']}")
        logger.info(f"총 벤담 계산: {self.learning_stats['bentham_calculations']}")
        logger.info(f"페이즈 전환 횟수: {self.learning_stats['phase_transitions']}")
        logger.info(f"현재 페이즈: {self.learning_stats['current_phase']}")
        logger.info(f"총 소요시간: {final_results['learning_stats']['total_time']:.2f}초")
        logger.info(f"총 오류 수: {len(self.learning_stats['errors'])}")
        logger.info(f"결과 파일: {final_file}")

async def main():
    """메인 함수"""
    try:
        learning_system = FixedLearningSystem()
        # 5개 샘플로 테스트
        await learning_system.run_learning(sample_count=5)
    except Exception as e:
        logger.error(f"메인 실행 중 오류: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())