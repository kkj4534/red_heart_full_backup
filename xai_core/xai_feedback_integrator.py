#!/usr/bin/env python3
"""
XAI 피드백 통합기 (XAI Feedback Integrator)
XAI 해석 결과를 윤리/감정 시스템에 피드백하는 구조

docs/추가 조정.txt 개선사항 구현:
- XAI 시스템의 피드백 연동 부족 해결
- 로그 결과를 기반으로 윤리 정책 강화 or 조정하는 루틴 설계
- XAI 로그의 feature importance 결과를 가중치 변경에 반영하는 lightweight fine-tuning 모듈
"""

import torch
import torch.nn as nn
import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import logging
from collections import defaultdict, deque
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

# 프로젝트 모듈 import
from .xai_logging_system import xai_logger, xai_trace
from ..models.ethics_policy_updater import EthicsPolicyUpdater, EthicsPolicyConfig

@dataclass
class XAIFeedbackConfig:
    """XAI 피드백 설정"""
    feature_importance_threshold: float = 0.3
    feedback_update_frequency: int = 50
    max_feedback_queue_size: int = 1000
    lightweight_learning_rate: float = 0.0001
    attention_weight_decay: float = 0.95
    interpretability_weight: float = 0.7
    
    # XAI 분석 카테고리
    analyze_attention_weights: bool = True
    analyze_gradient_importance: bool = True  
    analyze_activation_patterns: bool = True
    analyze_decision_pathways: bool = True

class XAIFeatureAnalyzer:
    """XAI 특징 분석기"""
    
    def __init__(self, config: XAIFeedbackConfig):
        self.config = config
        self.feature_importance_history = deque(maxlen=1000)
        self.attention_patterns = defaultdict(list)
        self.gradient_patterns = defaultdict(list)
        self.decision_pathways = defaultdict(list)
        
    def analyze_attention_weights(self, attention_weights: torch.Tensor, 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """어텐션 가중치 분석"""
        if not self.config.analyze_attention_weights:
            return {}
        
        # 어텐션 패턴 분석
        attention_np = attention_weights.detach().cpu().numpy()
        
        # 중요 특징 식별 (임계값 이상)
        important_features = np.where(attention_np > self.config.feature_importance_threshold)[0]
        
        # 어텐션 분포 통계
        attention_stats = {
            'mean_attention': float(np.mean(attention_np)),
            'max_attention': float(np.max(attention_np)),
            'attention_entropy': float(-np.sum(attention_np * np.log(attention_np + 1e-9))),
            'important_feature_count': len(important_features),
            'important_features': important_features.tolist()
        }
        
        # 패턴 저장
        decision_type = context.get('decision_type', 'unknown')
        self.attention_patterns[decision_type].append({
            'attention_stats': attention_stats,
            'timestamp': datetime.now().isoformat(),
            'context': context
        })
        
        return {
            'attention_analysis': attention_stats,
            'recommended_adjustments': self._get_attention_based_adjustments(attention_stats, decision_type)
        }
    
    def analyze_gradient_importance(self, gradients: torch.Tensor,
                                  parameters: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """그래디언트 중요도 분석"""
        if not self.config.analyze_gradient_importance:
            return {}
        
        gradient_importance = {}
        
        for name, param in parameters.items():
            if param.grad is not None:
                grad_norm = torch.norm(param.grad).item()
                param_norm = torch.norm(param).item()
                
                # 상대적 중요도 계산
                relative_importance = grad_norm / (param_norm + 1e-9)
                gradient_importance[name] = {
                    'gradient_norm': grad_norm,
                    'parameter_norm': param_norm,
                    'relative_importance': relative_importance
                }
        
        # 가장 중요한 파라미터들 식별
        sorted_importance = sorted(
            gradient_importance.items(),
            key=lambda x: x[1]['relative_importance'],
            reverse=True
        )
        
        return {
            'gradient_importance': gradient_importance,
            'top_important_params': sorted_importance[:5],
            'recommended_focus_areas': [item[0] for item in sorted_importance[:3]]
        }
    
    def analyze_activation_patterns(self, activations: Dict[str, torch.Tensor],
                                  decision_context: Dict[str, Any]) -> Dict[str, Any]:
        """활성화 패턴 분석"""
        if not self.config.analyze_activation_patterns:
            return {}
        
        activation_analysis = {}
        
        for layer_name, activation in activations.items():
            activation_np = activation.detach().cpu().numpy()
            
            # 활성화 통계
            activation_stats = {
                'mean_activation': float(np.mean(activation_np)),
                'std_activation': float(np.std(activation_np)),
                'sparsity': float(np.mean(activation_np == 0)),
                'max_activation': float(np.max(activation_np)),
                'dead_neurons': int(np.sum(np.all(activation_np == 0, axis=0)))
            }
            
            activation_analysis[layer_name] = activation_stats
        
        return {
            'activation_patterns': activation_analysis,
            'health_indicators': self._assess_activation_health(activation_analysis)
        }
    
    def analyze_decision_pathways(self, decision_flow: List[Dict[str, Any]]) -> Dict[str, Any]:
        """의사결정 경로 분석"""
        if not self.config.analyze_decision_pathways:
            return {}
        
        pathway_analysis = {
            'decision_steps': len(decision_flow),
            'pathway_efficiency': self._calculate_pathway_efficiency(decision_flow),
            'critical_decision_points': self._identify_critical_points(decision_flow),
            'alternative_pathways': self._suggest_alternative_pathways(decision_flow)
        }
        
        return pathway_analysis
    
    def _get_attention_based_adjustments(self, attention_stats: Dict, decision_type: str) -> Dict[str, float]:
        """어텐션 기반 조정 권장사항"""
        adjustments = {
            'ethics_attention_weight': 1.0,
            'emotion_attention_weight': 1.0,
            'regret_attention_weight': 1.0
        }
        
        # 어텐션이 너무 집중된 경우 분산 권장
        if attention_stats['attention_entropy'] < 1.0:
            adjustments['ethics_attention_weight'] = 0.8
            adjustments['emotion_attention_weight'] = 1.2
        
        # 중요 특징이 적은 경우 주의 집중 권장
        if attention_stats['important_feature_count'] < 3:
            adjustments['ethics_attention_weight'] = 1.3
            adjustments['regret_attention_weight'] = 1.2
        
        return adjustments
    
    def _assess_activation_health(self, activation_analysis: Dict) -> Dict[str, Any]:
        """활성화 건강도 평가"""
        health_indicators = {
            'overall_health': 'good',
            'issues': [],
            'recommendations': []
        }
        
        for layer_name, stats in activation_analysis.items():
            # 죽은 뉴런 검사
            if stats['dead_neurons'] > 0:
                health_indicators['issues'].append(f"{layer_name}: {stats['dead_neurons']} dead neurons")
                health_indicators['recommendations'].append(f"Consider lower learning rate for {layer_name}")
            
            # 희소성 검사
            if stats['sparsity'] > 0.8:
                health_indicators['issues'].append(f"{layer_name}: High sparsity ({stats['sparsity']:.2f})")
                health_indicators['recommendations'].append(f"Increase activation strength in {layer_name}")
        
        if health_indicators['issues']:
            health_indicators['overall_health'] = 'needs_attention'
        
        return health_indicators
    
    def _calculate_pathway_efficiency(self, decision_flow: List[Dict]) -> float:
        """의사결정 경로 효율성 계산"""
        if not decision_flow:
            return 0.0
        
        # 단순화된 효율성 메트릭: 결정 스텝 대비 확신도
        total_confidence = sum(step.get('confidence', 0.5) for step in decision_flow)
        return total_confidence / len(decision_flow)
    
    def _identify_critical_points(self, decision_flow: List[Dict]) -> List[Dict]:
        """중요한 결정 지점 식별"""
        critical_points = []
        
        for i, step in enumerate(decision_flow):
            confidence = step.get('confidence', 0.5)
            uncertainty = step.get('uncertainty', 0.5)
            
            # 낮은 확신도 또는 높은 불확실성 = 중요한 결정 지점
            if confidence < 0.6 or uncertainty > 0.4:
                critical_points.append({
                    'step_index': i,
                    'step_type': step.get('type', 'unknown'),
                    'confidence': confidence,
                    'uncertainty': uncertainty,
                    'importance': 1.0 - confidence + uncertainty
                })
        
        return sorted(critical_points, key=lambda x: x['importance'], reverse=True)
    
    def _suggest_alternative_pathways(self, decision_flow: List[Dict]) -> List[str]:
        """대안적 결정 경로 제안"""
        suggestions = []
        
        # 비효율적 스텝이 있는 경우 대안 제안
        for i, step in enumerate(decision_flow):
            if step.get('confidence', 0.5) < 0.5:
                suggestions.append(f"Step {i}: Consider emotion-first approach instead of logic-first")
            
            if step.get('regret_probability', 0.0) > 0.6:
                suggestions.append(f"Step {i}: Add regret mitigation checkpoint")
        
        return suggestions

class XAIFeedbackIntegrator:
    """XAI 피드백 통합기 - 메인 클래스"""
    
    def __init__(self, config: XAIFeedbackConfig, ethics_policy_updater: EthicsPolicyUpdater):
        self.config = config
        self.ethics_policy_updater = ethics_policy_updater
        self.feature_analyzer = XAIFeatureAnalyzer(config)
        
        # 피드백 큐
        self.feedback_queue = queue.Queue(maxsize=config.max_feedback_queue_size)
        self.processing_thread = None
        self.is_running = False
        
        # 경량 파인튜닝 모듈
        self.lightweight_tuner = LightweightFineTuner(config)
        
        # 피드백 통계
        self.feedback_stats = defaultdict(int)
        self.improvement_metrics = defaultdict(list)
        
        self.logger = logging.getLogger('XAIFeedbackIntegrator')
    
    def start_feedback_processing(self):
        """피드백 처리 시작"""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._process_feedback_loop, daemon=True)
        self.processing_thread.start()
        self.logger.info("XAI 피드백 처리 시작")
    
    def stop_feedback_processing(self):
        """피드백 처리 중단"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        self.logger.info("XAI 피드백 처리 중단")
    
    def submit_xai_analysis(self, xai_result: Dict[str, Any], context: Dict[str, Any]):
        """XAI 분석 결과 제출"""
        feedback_item = {
            'xai_result': xai_result,
            'context': context,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            self.feedback_queue.put_nowait(feedback_item)
            self.feedback_stats['submitted'] += 1
        except queue.Full:
            self.logger.warning("피드백 큐가 가득참 - 오래된 항목 제거")
            # 오래된 항목 제거하고 새 항목 추가
            try:
                self.feedback_queue.get_nowait()
                self.feedback_queue.put_nowait(feedback_item)
            except queue.Empty:
                pass
    
    def _process_feedback_loop(self):
        """피드백 처리 루프"""
        batch_items = []
        
        while self.is_running:
            try:
                # 배치 수집 (최대 1초 대기)
                try:
                    item = self.feedback_queue.get(timeout=1.0)
                    batch_items.append(item)
                except queue.Empty:
                    continue
                
                # 배치 크기에 도달하거나 주기적으로 처리
                if len(batch_items) >= 10:
                    self._process_feedback_batch(batch_items)
                    batch_items = []
                    
            except Exception as e:
                self.logger.error(f"피드백 처리 오류: {e}")
        
        # 종료시 남은 배치 처리
        if batch_items:
            self._process_feedback_batch(batch_items)
    
    def _process_feedback_batch(self, batch_items: List[Dict]):
        """피드백 배치 처리"""
        self.logger.debug(f"피드백 배치 처리: {len(batch_items)}개 항목")
        
        # XAI 분석 수행
        aggregated_analysis = self._aggregate_xai_analysis(batch_items)
        
        # 윤리/감정 시스템 조정 권장사항 생성
        adjustment_recommendations = self._generate_adjustments(aggregated_analysis)
        
        # 경량 파인튜닝 수행
        if adjustment_recommendations:
            self._apply_lightweight_tuning(adjustment_recommendations)
        
        # 윤리 정책 업데이터에 피드백
        self._update_ethics_policy(aggregated_analysis, adjustment_recommendations)
        
        self.feedback_stats['processed'] += len(batch_items)
        
        # 개선 메트릭 기록
        self._record_improvement_metrics(aggregated_analysis, adjustment_recommendations)
    
    def _aggregate_xai_analysis(self, batch_items: List[Dict]) -> Dict[str, Any]:
        """XAI 분석 결과 집계"""
        aggregated = {
            'attention_patterns': [],
            'gradient_importance': [],
            'activation_health': [],
            'decision_pathways': [],
            'feature_importance_trends': []
        }
        
        for item in batch_items:
            xai_result = item['xai_result']
            context = item['context']
            
            # 어텐션 분석
            if 'attention_weights' in xai_result:
                attention_analysis = self.feature_analyzer.analyze_attention_weights(
                    xai_result['attention_weights'], context
                )
                aggregated['attention_patterns'].append(attention_analysis)
            
            # 그래디언트 분석
            if 'gradients' in xai_result and 'parameters' in xai_result:
                gradient_analysis = self.feature_analyzer.analyze_gradient_importance(
                    xai_result['gradients'], xai_result['parameters']
                )
                aggregated['gradient_importance'].append(gradient_analysis)
            
            # 활성화 분석
            if 'activations' in xai_result:
                activation_analysis = self.feature_analyzer.analyze_activation_patterns(
                    xai_result['activations'], context
                )
                aggregated['activation_health'].append(activation_analysis)
            
            # 결정 경로 분석
            if 'decision_flow' in xai_result:
                pathway_analysis = self.feature_analyzer.analyze_decision_pathways(
                    xai_result['decision_flow']
                )
                aggregated['decision_pathways'].append(pathway_analysis)
        
        return aggregated
    
    def _generate_adjustments(self, aggregated_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """조정 권장사항 생성"""
        adjustments = {
            'ethics_weight_adjustments': {},
            'emotion_attention_adjustments': {},
            'model_parameter_adjustments': {},
            'training_recommendations': []
        }
        
        # 어텐션 패턴 기반 조정
        attention_patterns = aggregated_analysis.get('attention_patterns', [])
        if attention_patterns:
            ethics_attention_adj = {}
            for pattern in attention_patterns:
                if 'recommended_adjustments' in pattern:
                    for key, value in pattern['recommended_adjustments'].items():
                        ethics_attention_adj[key] = ethics_attention_adj.get(key, 0) + value
            
            # 평균 계산
            count = len(attention_patterns)
            adjustments['ethics_weight_adjustments'] = {
                k: v / count for k, v in ethics_attention_adj.items()
            }
        
        # 그래디언트 중요도 기반 조정
        gradient_importance = aggregated_analysis.get('gradient_importance', [])
        if gradient_importance:
            important_params = set()
            for analysis in gradient_importance:
                focus_areas = analysis.get('recommended_focus_areas', [])
                important_params.update(focus_areas)
            
            adjustments['model_parameter_adjustments'] = {
                'focus_parameters': list(important_params),
                'learning_rate_multiplier': 1.2  # 중요한 파라미터들의 학습률 증가
            }
        
        # 활성화 건강도 기반 권장사항
        activation_health = aggregated_analysis.get('activation_health', [])
        recommendations = []
        for health in activation_health:
            health_indicators = health.get('health_indicators', {})
            recommendations.extend(health_indicators.get('recommendations', []))
        
        adjustments['training_recommendations'] = list(set(recommendations))
        
        return adjustments
    
    def _apply_lightweight_tuning(self, adjustments: Dict[str, Any]):
        """경량 파인튜닝 적용"""
        try:
            self.lightweight_tuner.apply_adjustments(adjustments)
            self.logger.info("경량 파인튜닝 적용 완료")
        except Exception as e:
            self.logger.error(f"경량 파인튜닝 오류: {e}")
    
    def _update_ethics_policy(self, analysis: Dict[str, Any], adjustments: Dict[str, Any]):
        """윤리 정책 업데이트"""
        # 분석 결과를 경험으로 변환
        experience = {
            'analysis_summary': self._summarize_analysis(analysis),
            'adjustment_applied': adjustments,
            'feedback_source': 'xai_analysis',
            'improvement_potential': self._estimate_improvement_potential(analysis)
        }
        
        # 윤리 정책 업데이터에 피드백
        # (실제 구현에서는 ethics_policy_updater의 적절한 메서드 호출)
        self.logger.info("윤리 정책에 XAI 피드백 반영")
    
    def _record_improvement_metrics(self, analysis: Dict, adjustments: Dict):
        """개선 메트릭 기록"""
        metrics = {
            'attention_efficiency': self._calculate_attention_efficiency(analysis),
            'gradient_focus': self._calculate_gradient_focus(analysis),
            'decision_clarity': self._calculate_decision_clarity(analysis),
            'timestamp': datetime.now().isoformat()
        }
        
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.improvement_metrics[metric_name].append(value)
    
    def _summarize_analysis(self, analysis: Dict) -> Dict[str, Any]:
        """분석 결과 요약"""
        summary = {
            'attention_patterns_count': len(analysis.get('attention_patterns', [])),
            'gradient_analyses_count': len(analysis.get('gradient_importance', [])),
            'activation_health_issues': 0,
            'decision_pathway_efficiency': 0.0
        }
        
        # 활성화 건강도 이슈 계산
        for health in analysis.get('activation_health', []):
            indicators = health.get('health_indicators', {})
            summary['activation_health_issues'] += len(indicators.get('issues', []))
        
        # 결정 경로 효율성 평균
        pathways = analysis.get('decision_pathways', [])
        if pathways:
            efficiencies = [p.get('pathway_efficiency', 0.0) for p in pathways]
            summary['decision_pathway_efficiency'] = np.mean(efficiencies)
        
        return summary
    
    def _estimate_improvement_potential(self, analysis: Dict) -> float:
        """개선 가능성 추정"""
        # 단순화된 개선 가능성 메트릭
        potential_score = 0.5  # 기본값
        
        # 어텐션 패턴 개선 가능성
        attention_patterns = analysis.get('attention_patterns', [])
        if attention_patterns:
            avg_entropy = np.mean([
                p.get('attention_analysis', {}).get('attention_entropy', 2.0)
                for p in attention_patterns
            ])
            if avg_entropy < 1.5:  # 낮은 엔트로피 = 개선 가능성 높음
                potential_score += 0.2
        
        # 활성화 건강도 개선 가능성
        activation_health = analysis.get('activation_health', [])
        total_issues = sum(
            len(h.get('health_indicators', {}).get('issues', []))
            for h in activation_health
        )
        if total_issues > 0:
            potential_score += min(total_issues * 0.1, 0.3)
        
        return min(potential_score, 1.0)
    
    def _calculate_attention_efficiency(self, analysis: Dict) -> float:
        """어텐션 효율성 계산"""
        patterns = analysis.get('attention_patterns', [])
        if not patterns:
            return 0.5
        
        entropies = [
            p.get('attention_analysis', {}).get('attention_entropy', 0.0)
            for p in patterns
        ]
        return min(np.mean(entropies) / 3.0, 1.0)  # 정규화
    
    def _calculate_gradient_focus(self, analysis: Dict) -> float:
        """그래디언트 집중도 계산"""
        gradient_analyses = analysis.get('gradient_importance', [])
        if not gradient_analyses:
            return 0.5
        
        focus_scores = []
        for analysis in gradient_analyses:
            top_params = analysis.get('top_important_params', [])
            if len(top_params) >= 3:
                # 상위 3개 파라미터의 중요도 합
                top_importance = sum(p[1]['relative_importance'] for p in top_params[:3])
                focus_scores.append(min(top_importance, 1.0))
        
        return np.mean(focus_scores) if focus_scores else 0.5
    
    def _calculate_decision_clarity(self, analysis: Dict) -> float:
        """결정 명확성 계산"""
        pathways = analysis.get('decision_pathways', [])
        if not pathways:
            return 0.5
        
        efficiencies = [p.get('pathway_efficiency', 0.0) for p in pathways]
        return np.mean(efficiencies)
    
    def get_feedback_statistics(self) -> Dict[str, Any]:
        """피드백 통계 반환"""
        return {
            'feedback_stats': dict(self.feedback_stats),
            'improvement_metrics': {
                k: {
                    'latest': v[-1] if v else None,
                    'average': np.mean(v) if v else None,
                    'trend': np.polyfit(range(len(v)), v, 1)[0] if len(v) > 1 else 0
                }
                for k, v in self.improvement_metrics.items()
            },
            'queue_size': self.feedback_queue.qsize(),
            'is_running': self.is_running
        }

class LightweightFineTuner:
    """경량 파인튜닝 모듈"""
    
    def __init__(self, config: XAIFeedbackConfig):
        self.config = config
        self.adjustment_history = deque(maxlen=100)
        
    def apply_adjustments(self, adjustments: Dict[str, Any]):
        """조정사항 적용"""
        # 실제 구현에서는 모델 파라미터에 직접 적용
        # 여기서는 로깅만 수행
        
        ethics_adjustments = adjustments.get('ethics_weight_adjustments', {})
        if ethics_adjustments:
            logging.getLogger('LightweightFineTuner').info(
                f"윤리 가중치 조정: {ethics_adjustments}"
            )
        
        param_adjustments = adjustments.get('model_parameter_adjustments', {})
        if param_adjustments:
            logging.getLogger('LightweightFineTuner').info(
                f"모델 파라미터 조정: {param_adjustments}"
            )
        
        # 조정 이력 저장
        self.adjustment_history.append({
            'adjustments': adjustments,
            'timestamp': datetime.now().isoformat()
        })

# 헬퍼 함수들
def create_xai_feedback_integrator(
    config: Optional[XAIFeedbackConfig] = None,
    ethics_policy_updater: Optional[EthicsPolicyUpdater] = None
) -> XAIFeedbackIntegrator:
    """XAI 피드백 통합기 생성"""
    if config is None:
        config = XAIFeedbackConfig()
    
    if ethics_policy_updater is None:
        from ..models.ethics_policy_updater import create_ethics_policy_updater
        ethics_policy_updater = create_ethics_policy_updater()
    
    return XAIFeedbackIntegrator(config, ethics_policy_updater)

# 사용 예시
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # XAI 피드백 통합기 생성
    config = XAIFeedbackConfig(
        feature_importance_threshold=0.3,
        feedback_update_frequency=50
    )
    
    integrator = create_xai_feedback_integrator(config)
    integrator.start_feedback_processing()
    
    # 가상의 XAI 결과 제출
    dummy_xai_result = {
        'attention_weights': torch.rand(10),
        'gradients': torch.rand(5, 10),
        'parameters': {'layer1': torch.rand(5, 10)},
        'activations': {'layer1': torch.rand(2, 10)},
        'decision_flow': [
            {'type': 'emotion_analysis', 'confidence': 0.7, 'uncertainty': 0.3},
            {'type': 'ethics_evaluation', 'confidence': 0.6, 'uncertainty': 0.4}
        ]
    }
    
    context = {
        'decision_type': 'altruistic',
        'scenario_type': 'ethical_dilemma',
        'complexity': 'high'
    }
    
    integrator.submit_xai_analysis(dummy_xai_result, context)
    
    # 통계 출력
    import time
    time.sleep(2)  # 처리 대기
    stats = integrator.get_feedback_statistics()
    print(f"피드백 통계: {stats}")
    
    integrator.stop_feedback_processing()