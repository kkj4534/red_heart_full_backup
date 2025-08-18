#!/usr/bin/env python3
"""
Red Heart 학습 및 시뮬레이션 실행 스크립트
Learning and Simulation Runner with Advanced Logging

이 스크립트는 Red Heart 시스템의 학습 과정을 실행하고
상세한 로그 리포트를 생성합니다.
"""

import asyncio
import logging
import time
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

from main import (
    RedHeartSystem, AnalysisRequest, setup_advanced_logging,
    log_regret_progress, log_performance_metric, get_learning_logger
)
from advanced_hierarchical_emotion_system import (
    AdvancedHierarchicalEmotionSystem, EmotionVector, EmotionDimension
)
from advanced_regret_learning_system import (
    AdvancedRegretLearningSystem, RegretMemory, RegretType, LearningPhase
)
from advanced_bayesian_inference_module import (
    AdvancedBayesianInference, Evidence, BeliefType
)
from config import LOGS_DIR


class LearningSimulationRunner:
    """학습 시뮬레이션 실행기"""
    
    def __init__(self):
        self.logger = get_learning_logger("SimulationRunner")
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {
            'session_id': self.session_id,
            'start_time': datetime.now().isoformat(),
            'phases': [],
            'regret_timeline': [],
            'performance_metrics': {},
            'learning_statistics': {}
        }
        
        # 시스템 인스턴스들
        self.main_system = None
        self.emotion_system = None
        self.regret_system = None
        self.bayesian_system = None
        
        self.logger.info(f"학습 시뮬레이션 세션 시작: {self.session_id}", extra={'phase': 'INIT', 'regret': 0.0})
    
    async def initialize_systems(self):
        """시스템들 초기화"""
        self.logger.info("=== 시스템 초기화 단계 ===", extra={'phase': 'INIT', 'regret': 0.0})
        
        # 메인 시스템
        self.logger.info("메인 Red Heart 시스템 초기화 중...", extra={'phase': 'INIT', 'regret': 0.0})
        self.main_system = RedHeartSystem()
        await self.main_system.initialize()
        
        # 개별 고급 시스템들 (더 상세한 제어를 위해)
        self.logger.info("계층적 감정 시스템 초기화 중...", extra={'phase': 'INIT', 'regret': 0.0})
        self.emotion_system = AdvancedHierarchicalEmotionSystem()
        
        self.logger.info("후회 학습 시스템 초기화 중...", extra={'phase': 'INIT', 'regret': 0.0})
        self.regret_system = AdvancedRegretLearningSystem()
        
        self.logger.info("베이지안 추론 시스템 초기화 중...", extra={'phase': 'INIT', 'regret': 0.0})
        self.bayesian_system = AdvancedBayesianInference()
        
        self.logger.info("모든 시스템 초기화 완료", extra={'phase': 'INIT', 'regret': 0.0})
        log_performance_metric("Initialization", "systems_ready", 1.0, "전체 시스템 준비 완료")
    
    async def run_hierarchical_emotion_learning(self, num_episodes: int = 150):
        """계층적 감정 학습 실행"""
        self.logger.info(f"=== 계층적 감정 학습 시작 ({num_episodes}회) ===", extra={'phase': 'EMOTION_LEARNING', 'regret': 0.0})
        
        # 테스트용 문학 시나리오 생성
        literary_scenarios = self._generate_literary_scenarios()
        
        phase_timeline = []
        
        for episode in range(num_episodes):
            scenario = literary_scenarios[episode % len(literary_scenarios)]
            
            # 학습 수행
            result = await self.emotion_system.process_literary_emotion_sequence(
                [scenario], time_series_mode=True
            )
            
            # 현재 페이즈 상태 확인
            current_phase = "PHASE_0"  # 기본값
            
            # 결과 로깅
            if result.get('learning_summary'):
                summary = result['learning_summary']
                
                # 후회 수준 계산 (Phase 1 학습이 있는 경우)
                regret_level = 0.0
                if result.get('phase1_learnings'):
                    regret_levels = [l.regret_intensity for l in result['phase1_learnings']]
                    regret_level = sum(regret_levels) / len(regret_levels) if regret_levels else 0.0
                
                # 상세 로깅
                log_regret_progress(
                    current_phase, 
                    regret_level,
                    f"에피소드 {episode+1}: 감정 학습 완료",
                    episode=episode+1,
                    scenario_type=scenario.get('genre', 'unknown'),
                    calibrations=summary.get('total_calibrations', 0),
                    empathy_learnings=summary.get('total_empathy_learnings', 0)
                )
                
                # 타임라인에 추가
                phase_timeline.append({
                    'episode': episode + 1,
                    'phase': current_phase,
                    'regret_level': regret_level,
                    'timestamp': datetime.now().isoformat(),
                    'scenario': scenario.get('genre', 'unknown'),
                    'metrics': summary
                })
                
                # 진행률 표시
                if (episode + 1) % 10 == 0:
                    progress = (episode + 1) / num_episodes * 100
                    self.logger.info(f"감정 학습 진행률: {progress:.1f}% ({episode+1}/{num_episodes})", extra={'phase': current_phase, 'regret': 0.0})
                    log_performance_metric(
                        "EmotionLearning", "progress_percent", progress,
                        f"에피소드 {episode+1} 완료"
                    )
        
        self.results['emotion_learning'] = {
            'episodes': num_episodes,
            'timeline': phase_timeline,
            'final_metrics': result.get('learning_summary', {})
        }
        
        self.logger.info(f"계층적 감정 학습 완료: {num_episodes}회 에피소드", extra={'phase': 'EMOTION_COMPLETE', 'regret': 0.0})
        return phase_timeline
    
    async def run_regret_learning_simulation(self, num_scenarios: int = 120):
        """후회 학습 시뮬레이션 실행"""
        self.logger.info(f"=== 후회 학습 시뮬레이션 시작 ({num_scenarios}회) ===", extra={'phase': 'REGRET_LEARNING', 'regret': 0.0})
        
        # 후회 시나리오 생성
        regret_scenarios = self._generate_regret_scenarios()
        
        regret_timeline = []
        phase_transitions = []
        
        for scenario_idx in range(num_scenarios):
            scenario = regret_scenarios[scenario_idx % len(regret_scenarios)]
            
            # 후회 학습 수행
            regret_memory = await self.regret_system.process_regret(
                situation=scenario['situation'],
                outcome=scenario['outcome'],
                alternatives=scenario['alternatives'],
                literary_context=scenario.get('literary_context')
            )
            
            # 현재 학습 상태 확인
            current_phase = self.regret_system.learning_state.current_phase.name
            regret_level = regret_memory.intensity
            
            # 페이즈 전환 체크
            if len(self.regret_system.learning_state.phase_transitions) > len(phase_transitions):
                new_transition = self.regret_system.learning_state.phase_transitions[-1]
                phase_transitions.append({
                    'from_phase': new_transition.from_phase.name,
                    'to_phase': new_transition.to_phase.name,
                    'scenario_number': scenario_idx + 1,
                    'trigger': new_transition.trigger_condition,
                    'metrics': new_transition.metrics_at_transition,
                    'timestamp': new_transition.transition_time.isoformat()
                })
                
                self.logger.info(f"🔄 페이즈 전환 발생: {new_transition.from_phase.name} → {new_transition.to_phase.name}", extra={'phase': new_transition.to_phase.name, 'regret': current_regret})
                log_regret_progress(
                    new_transition.to_phase.name,
                    regret_level,
                    f"페이즈 전환: {new_transition.from_phase.name} → {new_transition.to_phase.name}",
                    transition_trigger=new_transition.trigger_condition,
                    scenario_number=scenario_idx + 1
                )
            
            # 상세 로깅
            log_regret_progress(
                current_phase,
                regret_level,
                f"시나리오 {scenario_idx+1}: {regret_memory.regret_type.value} 후회 학습",
                scenario_number=scenario_idx + 1,
                regret_type=regret_memory.regret_type.value,
                learning_phase=current_phase,
                confidence=regret_memory.confidence_level if hasattr(regret_memory, 'confidence_level') else 0.5
            )
            
            # 타임라인에 추가
            regret_timeline.append({
                'scenario': scenario_idx + 1,
                'phase': current_phase,
                'regret_type': regret_memory.regret_type.value,
                'regret_intensity': regret_level,
                'timestamp': datetime.now().isoformat(),
                'situation_type': scenario['situation'].get('type', 'unknown')
            })
            
            # 진행률 표시
            if (scenario_idx + 1) % 10 == 0:
                progress = (scenario_idx + 1) / num_scenarios * 100
                self.logger.info(f"후회 학습 진행률: {progress:.1f}% ({scenario_idx+1}/{num_scenarios})", extra={'phase': current_phase.name if hasattr(current_phase, 'name') else 'REGRET_LEARNING', 'regret': current_regret})
                log_performance_metric(
                    "RegretLearning", "progress_percent", progress,
                    f"시나리오 {scenario_idx+1} 완료, 현재 페이즈: {current_phase}"
                )
        
        # 최종 리포트 생성
        final_report = await self.regret_system.generate_regret_report()
        
        self.results['regret_learning'] = {
            'scenarios': num_scenarios,
            'timeline': regret_timeline,
            'phase_transitions': phase_transitions,
            'final_report': final_report
        }
        
        self.logger.info(f"후회 학습 시뮬레이션 완료: {num_scenarios}회 시나리오, {len(phase_transitions)}회 페이즈 전환", extra={'phase': 'REGRET_COMPLETE', 'regret': 0.0})
        return regret_timeline, phase_transitions
    
    async def run_bayesian_inference_learning(self, num_inferences: int = 80):
        """베이지안 추론 학습 실행"""
        self.logger.info(f"=== 베이지안 추론 학습 시작 ({num_inferences}회) ===", extra={'phase': 'BAYESIAN_LEARNING', 'regret': 0.0})
        
        # 추론 시나리오 생성
        inference_scenarios = self._generate_inference_scenarios()
        
        inference_timeline = []
        
        for inference_idx in range(num_inferences):
            scenario = inference_scenarios[inference_idx % len(inference_scenarios)]
            
            # 증거 추가
            for evidence_data in scenario['evidences']:
                evidence = Evidence(
                    node_id=evidence_data['node'],
                    value=evidence_data['value'],
                    strength=evidence_data['strength'],
                    source="simulation"
                )
                await self.bayesian_system.add_evidence(evidence)
            
            # 추론 실행
            result = await self.bayesian_system.infer(
                query_node=scenario['query_node'],
                given_evidence=scenario.get('given_evidence', {}),
                context=scenario.get('context', {})
            )
            
            # 결과 학습
            if scenario.get('actual_outcome'):
                await self.bayesian_system.update_from_outcome(
                    scenario['query_node'],
                    max(result.posterior_distribution.items(), key=lambda x: x[1])[0],
                    scenario['actual_outcome'],
                    scenario.get('context', {})
                )
            
            # 로깅
            uncertainty = result.uncertainty
            confidence = result.confidence
            
            log_performance_metric(
                "BayesianInference", "uncertainty", uncertainty,
                f"추론 {inference_idx+1}: 불확실성 {uncertainty:.3f}"
            )
            
            # 타임라인에 추가
            inference_timeline.append({
                'inference': inference_idx + 1,
                'query_node': scenario['query_node'],
                'uncertainty': uncertainty,
                'confidence': confidence,
                'posterior': result.posterior_distribution,
                'timestamp': datetime.now().isoformat()
            })
            
            # 진행률 표시
            if (inference_idx + 1) % 10 == 0:
                progress = (inference_idx + 1) / num_inferences * 100
                self.logger.info(f"베이지안 추론 진행률: {progress:.1f}% ({inference_idx+1}/{num_inferences})", extra={'phase': 'BAYESIAN_LEARNING', 'regret': 0.0})
        
        self.results['bayesian_learning'] = {
            'inferences': num_inferences,
            'timeline': inference_timeline
        }
        
        self.logger.info(f"베이지안 추론 학습 완료: {num_inferences}회 추론", extra={'phase': 'BAYESIAN_COMPLETE', 'regret': 0.0})
        return inference_timeline
    
    async def run_integrated_analysis(self, num_analyses: int = 50):
        """통합 분석 실행"""
        self.logger.info(f"=== 통합 분석 실행 시작 ({num_analyses}회) ===", extra={'phase': 'INTEGRATION', 'regret': 0.0})
        
        # 복합 시나리오 생성
        integrated_scenarios = self._generate_integrated_scenarios()
        
        analysis_timeline = []
        
        for analysis_idx in range(num_analyses):
            scenario = integrated_scenarios[analysis_idx % len(integrated_scenarios)]
            
            # 통합 분석 실행
            request = AnalysisRequest(
                text=scenario['text'],
                language=scenario.get('language', 'ko'),
                scenario_type=scenario.get('type', 'general'),
                additional_context=scenario.get('context', {})
            )
            
            start_time = time.time()
            result = await self.main_system.analyze_async(request)
            processing_time = time.time() - start_time
            
            # 성능 로깅
            log_performance_metric(
                "IntegratedAnalysis", "processing_time", processing_time,
                f"분석 {analysis_idx+1}: {processing_time:.3f}초"
            )
            
            log_performance_metric(
                "IntegratedAnalysis", "integrated_score", result.integrated_score,
                f"통합 점수: {result.integrated_score:.3f}"
            )
            
            # 타임라인에 추가
            analysis_timeline.append({
                'analysis': analysis_idx + 1,
                'scenario_type': scenario.get('type', 'general'),
                'processing_time': processing_time,
                'integrated_score': result.integrated_score,
                'confidence': result.confidence,
                'components_used': len([c for c in [
                    result.emotion_analysis, result.bentham_analysis,
                    result.semantic_analysis, result.surd_analysis
                ] if c is not None]),
                'timestamp': datetime.now().isoformat()
            })
            
            # 진행률 표시
            if (analysis_idx + 1) % 10 == 0:
                progress = (analysis_idx + 1) / num_analyses * 100
                avg_time = sum(a['processing_time'] for a in analysis_timeline) / len(analysis_timeline)
                self.logger.info(f"통합 분석 진행률: {progress:.1f}% ({analysis_idx+1}/{num_analyses}), 평균 처리시간: {avg_time:.3f}초", extra={'phase': 'INTEGRATION', 'regret': 0.0})
        
        self.results['integrated_analysis'] = {
            'analyses': num_analyses,
            'timeline': analysis_timeline
        }
        
        self.logger.info(f"통합 분석 완료: {num_analyses}회 분석", extra={'phase': 'INTEGRATION_COMPLETE', 'regret': 0.0})
        return analysis_timeline
    
    def _generate_literary_scenarios(self) -> List[Dict[str, Any]]:
        """문학적 시나리오 생성"""
        return [
            {
                'character_emotion': {
                    'valence': -0.8, 'arousal': 0.7, 'dominance': -0.5,
                    'source': 'hamlet'
                },
                'reader_emotion': {
                    'valence': -0.5, 'arousal': 0.5, 'dominance': 0.0
                },
                'context': {
                    'situation_type': 'tragedy',
                    'cultural_context': 'western'
                },
                'genre': 'tragedy'
            },
            {
                'character_emotion': {
                    'valence': 0.3, 'arousal': 0.4, 'dominance': -0.2,
                    'source': 'chunhyang'
                },
                'reader_emotion': {
                    'valence': 0.5, 'arousal': 0.3, 'dominance': 0.1
                },
                'community_emotions': {
                    'reader1': {'valence': 0.5, 'arousal': 0.3},
                    'reader2': {'valence': 0.4, 'arousal': 0.4},
                    'reader3': {'valence': 0.6, 'arousal': 0.2}
                },
                'context': {
                    'situation_type': 'romance',
                    'cultural_context': 'korean_traditional'
                },
                'cultural_context': 'korean_traditional',
                'genre': 'romance'
            },
            {
                'character_emotion': {
                    'valence': 0.7, 'arousal': 0.8, 'dominance': 0.3,
                    'source': 'don_quixote'
                },
                'reader_emotion': {
                    'valence': 0.6, 'arousal': 0.6, 'dominance': 0.2
                },
                'context': {
                    'situation_type': 'comedy',
                    'cultural_context': 'western'
                },
                'genre': 'comedy'
            }
        ]
    
    def _generate_regret_scenarios(self) -> List[Dict[str, Any]]:
        """후회 시나리오 생성"""
        return [
            {
                'situation': {
                    'type': 'moral_dilemma',
                    'emotion_type': 'conflict',
                    'chosen_action': {'type': 'compromise'},
                    'is_critical': False
                },
                'outcome': {
                    'negativity': 0.6,
                    'actual_value': 0.3,
                    'empathy_failure': False
                },
                'alternatives': [
                    {'action': 'stand_firm', 'expected_value': 0.7},
                    {'action': 'full_cooperation', 'expected_value': 0.5}
                ],
                'literary_context': {
                    'source': 'les_miserables',
                    'genre': 'moral_literature'
                }
            },
            {
                'situation': {
                    'type': 'relationship_conflict',
                    'emotion_type': 'sadness',
                    'chosen_action': {'type': 'avoidance'},
                    'is_critical': True
                },
                'outcome': {
                    'negativity': 0.8,
                    'actual_value': 0.2,
                    'empathy_failure': True
                },
                'alternatives': [
                    {'action': 'direct_communication', 'expected_value': 0.8},
                    {'action': 'seek_mediation', 'expected_value': 0.6}
                ]
            },
            {
                'situation': {
                    'type': 'career_decision',
                    'emotion_type': 'anxiety',
                    'chosen_action': {'type': 'safe_choice'},
                    'is_critical': False
                },
                'outcome': {
                    'negativity': 0.4,
                    'actual_value': 0.6,
                    'prediction_failure': True
                },
                'alternatives': [
                    {'action': 'risk_taking', 'expected_value': 0.9},
                    {'action': 'wait_and_see', 'expected_value': 0.4}
                ]
            }
        ]
    
    def _generate_inference_scenarios(self) -> List[Dict[str, Any]]:
        """추론 시나리오 생성"""
        return [
            {
                'query_node': 'action_tendency',
                'evidences': [
                    {'node': 'emotional_state', 'value': 'negative', 'strength': 0.8}
                ],
                'given_evidence': {'moral_judgment': 'right'},
                'context': {'literary_context': {'belief_type': 'tragic_fate', 'source': 'hamlet'}},
                'actual_outcome': 'avoid'
            },
            {
                'query_node': 'outcome_prediction',
                'evidences': [
                    {'node': 'emotional_state', 'value': 'positive', 'strength': 0.7},
                    {'node': 'social_approval', 'value': 'approve', 'strength': 0.6}
                ],
                'context': {'literary_context': {'belief_type': 'redemption', 'source': 'les_miserables'}},
                'actual_outcome': 'success'
            }
        ]
    
    def _generate_integrated_scenarios(self) -> List[Dict[str, Any]]:
        """통합 분석 시나리오 생성"""
        return [
            {
                'text': '이 결정은 많은 사람들의 생명과 안전에 직접적인 영향을 미치며, 우리는 정의롭고 공정한 선택을 해야 합니다.',
                'type': 'ethical_dilemma',
                'context': {'urgency': 'high', 'stakeholders': 'many'}
            },
            {
                'text': '새로운 기술 도입으로 효율성은 높아지지만 일부 직원들이 일자리를 잃을 수 있습니다.',
                'type': 'technology_ethics',
                'context': {'impact': 'economic', 'timeframe': 'long_term'}
            },
            {
                'text': '개인정보 보호와 공익을 위한 정보 공개 사이에서 균형점을 찾아야 합니다.',
                'type': 'privacy_vs_public',
                'context': {'legal_framework': 'gdpr', 'public_interest': 'high'}
            },
            {
                'text': '환경 보호를 위해 경제 성장을 일부 제한해야 할지 고민이 됩니다.',
                'type': 'environment_vs_economy',
                'context': {'urgency': 'medium', 'global_impact': True}
            }
        ]
    
    async def generate_final_report(self):
        """최종 리포트 생성"""
        self.logger.info("=== 최종 학습 리포트 생성 ===", extra={'phase': 'REPORT', 'regret': 0.0})
        
        self.results['end_time'] = datetime.now().isoformat()
        self.results['duration'] = (
            datetime.fromisoformat(self.results['end_time']) - 
            datetime.fromisoformat(self.results['start_time'])
        ).total_seconds()
        
        # 종합 통계 계산
        total_episodes = (
            self.results.get('emotion_learning', {}).get('episodes', 0) +
            self.results.get('regret_learning', {}).get('scenarios', 0) +
            self.results.get('bayesian_learning', {}).get('inferences', 0) +
            self.results.get('integrated_analysis', {}).get('analyses', 0)
        )
        
        self.results['summary'] = {
            'total_learning_episodes': total_episodes,
            'phase_transitions': len(self.results.get('regret_learning', {}).get('phase_transitions', [])),
            'final_phase': (
                self.results.get('regret_learning', {})
                .get('phase_transitions', [{}])[-1]
                .get('to_phase', 'PHASE_0') if self.results.get('regret_learning', {}).get('phase_transitions') else 'PHASE_0'
            ),
            'avg_regret_final': self._calculate_average_final_regret(),
            'performance_improvement': self._calculate_performance_improvement()
        }
        
        # 리포트 파일 저장
        report_file = Path(LOGS_DIR) / f"learning_simulation_report_{self.session_id}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"최종 리포트 저장: {report_file}", extra={'phase': 'REPORT', 'regret': 0.0})
        
        # 요약 출력
        self.logger.info("=== 학습 시뮬레이션 완료 요약 ===", extra={'phase': 'COMPLETE', 'regret': 0.0})
        self.logger.info(f"세션 ID: {self.session_id}", extra={'phase': 'COMPLETE', 'regret': 0.0})
        self.logger.info(f"총 학습 에피소드: {total_episodes}", extra={'phase': 'COMPLETE', 'regret': 0.0})
        self.logger.info(f"페이즈 전환 횟수: {self.results['summary']['phase_transitions']}", extra={'phase': 'COMPLETE', 'regret': 0.0})
        self.logger.info(f"최종 도달 페이즈: {self.results['summary']['final_phase']}", extra={'phase': 'COMPLETE', 'regret': 0.0})
        self.logger.info(f"평균 최종 후회 수준: {self.results['summary']['avg_regret_final']:.3f}", extra={'phase': 'COMPLETE', 'regret': self.results['summary']['avg_regret_final']})
        self.logger.info(f"전체 실행 시간: {self.results['duration']:.1f}초", extra={'phase': 'COMPLETE', 'regret': 0.0})
        
        return self.results
    
    def _calculate_average_final_regret(self) -> float:
        """최종 후회 수준 평균 계산"""
        regret_timeline = self.results.get('regret_learning', {}).get('timeline', [])
        if not regret_timeline:
            return 0.0
        
        # 마지막 20개 에피소드의 평균
        recent_regrets = [r['regret_intensity'] for r in regret_timeline[-20:]]
        return sum(recent_regrets) / len(recent_regrets) if recent_regrets else 0.0
    
    def _calculate_performance_improvement(self) -> float:
        """성능 개선도 계산"""
        regret_timeline = self.results.get('regret_learning', {}).get('timeline', [])
        if len(regret_timeline) < 20:
            return 0.0
        
        # 초기 20개와 마지막 20개 비교
        initial_regrets = [r['regret_intensity'] for r in regret_timeline[:20]]
        final_regrets = [r['regret_intensity'] for r in regret_timeline[-20:]]
        
        initial_avg = sum(initial_regrets) / len(initial_regrets)
        final_avg = sum(final_regrets) / len(final_regrets)
        
        if initial_avg > 0:
            return (initial_avg - final_avg) / initial_avg
        return 0.0


async def main():
    """메인 실행 함수"""
    print("🔴❤️ Red Heart 학습 시뮬레이션 시작")
    print("=" * 60)
    
    # 고급 로깅 설정
    setup_advanced_logging()
    
    # 시뮬레이션 실행기 생성
    runner = LearningSimulationRunner()
    
    try:
        # 시스템 초기화
        await runner.initialize_systems()
        
        # 학습 시뮬레이션 실행
        print("\n📚 계층적 감정 학습 실행 중...")
        await runner.run_hierarchical_emotion_learning(num_episodes=100)
        
        print("\n💭 후회 학습 시뮬레이션 실행 중...")
        await runner.run_regret_learning_simulation(num_scenarios=80)
        
        print("\n🔮 베이지안 추론 학습 실행 중...")
        await runner.run_bayesian_inference_learning(num_inferences=60)
        
        print("\n🔄 통합 분석 실행 중...")
        await runner.run_integrated_analysis(num_analyses=40)
        
        # 최종 리포트 생성
        print("\n📊 최종 리포트 생성 중...")
        final_results = await runner.generate_final_report()
        
        print("\n✅ 학습 시뮬레이션 완료!")
        print(f"📁 결과 파일: logs/learning_simulation_report_{runner.session_id}.json")
        print(f"📁 상세 로그: logs/learning_progress_{runner.session_id}.log")
        
    except Exception as e:
        print(f"\n❌ 시뮬레이션 실행 중 오류: {e}")
        logging.exception("시뮬레이션 실행 실패")


if __name__ == "__main__":
    asyncio.run(main())