"""
XAI 추적 가능한 로깅 시스템
XAI Traceable Logging System
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
import logging
import threading
from collections import defaultdict
import numpy as np
import torch
from contextlib import contextmanager

@dataclass
class XAILogEntry:
    """XAI 로그 엔트리"""
    timestamp: str
    session_id: str
    operation_id: str
    module_name: str
    operation_type: str
    input_shape: Optional[tuple] = None
    output_shape: Optional[tuple] = None
    parameters: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    memory_usage: Optional[float] = None
    explanation_data: Optional[Dict[str, Any]] = None
    llm_interaction: Optional[Dict[str, Any]] = None
    attention_weights: Optional[List[float]] = None
    gradient_norm: Optional[float] = None
    decision_path: Optional[List[str]] = None
    confidence_score: Optional[float] = None

@dataclass
class XAIDecisionTrace:
    """XAI 의사결정 추적"""
    decision_id: str
    timestamp: str
    model_components: List[str]
    input_features: Dict[str, Any]
    intermediate_outputs: Dict[str, Any]
    final_decision: Dict[str, Any]
    explanation_chain: List[Dict[str, Any]]
    llm_reasoning: Optional[str] = None
    uncertainty_measures: Optional[Dict[str, float]] = None

class XAILogger:
    """XAI 전용 로거"""
    
    def __init__(self, log_dir: Path = None):
        self.log_dir = log_dir or Path("logs/xai")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.session_id = str(uuid.uuid4())
        self.logs = []
        self.decision_traces = []
        self.lock = threading.Lock()
        
        # 성능 추적
        self.performance_metrics = defaultdict(list)
        self.memory_tracker = defaultdict(list)
        
        # 설정
        self.max_logs_in_memory = 10000
        self.auto_save_interval = 100
        self.log_counter = 0
        
        # 표준 로거 설정
        self.logger = logging.getLogger('XAI_Logger')
        handler = logging.FileHandler(self.log_dir / f"xai_session_{self.session_id}.log")
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    @contextmanager
    def trace_operation(self, module_name: str, operation_type: str, **kwargs):
        """연산 추적 컨텍스트 매니저"""
        operation_id = str(uuid.uuid4())
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            yield operation_id
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            entry = XAILogEntry(
                timestamp=datetime.now().isoformat(),
                session_id=self.session_id,
                operation_id=operation_id,
                module_name=module_name,
                operation_type=operation_type,
                execution_time=end_time - start_time,
                memory_usage=end_memory - start_memory,
                **kwargs
            )
            
            self.add_log_entry(entry)
    
    def add_log_entry(self, entry: XAILogEntry):
        """로그 엔트리 추가"""
        with self.lock:
            self.logs.append(entry)
            self.log_counter += 1
            
            # 성능 메트릭 업데이트
            if entry.execution_time:
                self.performance_metrics[entry.module_name].append(entry.execution_time)
            if entry.memory_usage:
                self.memory_tracker[entry.module_name].append(entry.memory_usage)
            
            # 자동 저장
            if self.log_counter % self.auto_save_interval == 0:
                self._auto_save()
            
            # 메모리 관리
            if len(self.logs) > self.max_logs_in_memory:
                self._flush_to_disk()
    
    def add_decision_trace(self, trace: XAIDecisionTrace):
        """의사결정 추적 추가"""
        with self.lock:
            self.decision_traces.append(trace)
            
            # JSON으로 저장
            trace_file = self.log_dir / f"decision_trace_{trace.decision_id}.json"
            with open(trace_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(trace), f, indent=2, ensure_ascii=False)
    
    def log_llm_interaction(self, operation_id: str, prompt: str, response: str, 
                           model_name: str = "unknown", tokens_used: int = None):
        """LLM 상호작용 로그"""
        llm_data = {
            'model_name': model_name,
            'prompt': prompt[:500],  # 처음 500자만
            'response': response[:1000],  # 처음 1000자만
            'tokens_used': tokens_used,
            'timestamp': datetime.now().isoformat()
        }
        
        # 해당 operation에 LLM 데이터 추가
        for entry in reversed(self.logs):
            if entry.operation_id == operation_id:
                entry.llm_interaction = llm_data
                break
    
    def log_attention_weights(self, operation_id: str, weights: torch.Tensor):
        """어텐션 가중치 로그"""
        if weights is not None:
            attention_data = weights.detach().cpu().numpy().tolist()
            for entry in reversed(self.logs):
                if entry.operation_id == operation_id:
                    entry.attention_weights = attention_data[:100]  # 처음 100개만
                    break
    
    def log_gradient_info(self, operation_id: str, model: torch.nn.Module):
        """그래디언트 정보 로그"""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        for entry in reversed(self.logs):
            if entry.operation_id == operation_id:
                entry.gradient_norm = total_norm
                break
    
    def create_explanation_chain(self, model_outputs: Dict[str, Any], 
                                input_data: Any, model_name: str) -> Dict[str, Any]:
        """설명 체인 생성"""
        explanation = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'input_summary': self._summarize_input(input_data),
            'output_summary': self._summarize_outputs(model_outputs),
            'feature_importance': self._calculate_feature_importance(model_outputs),
            'decision_confidence': self._calculate_confidence(model_outputs)
        }
        return explanation
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 반환"""
        summary = {}
        for module, times in self.performance_metrics.items():
            summary[module] = {
                'avg_time': np.mean(times),
                'max_time': np.max(times),
                'min_time': np.min(times),
                'total_calls': len(times)
            }
        return summary
    
    def export_xai_report(self, output_path: Path = None) -> Path:
        """XAI 리포트 내보내기"""
        if output_path is None:
            output_path = self.log_dir / f"xai_report_{self.session_id}.json"
        
        report = {
            'session_info': {
                'session_id': self.session_id,
                'start_time': self.logs[0].timestamp if self.logs else None,
                'end_time': self.logs[-1].timestamp if self.logs else None,
                'total_operations': len(self.logs),
                'total_decisions': len(self.decision_traces)
            },
            'performance_summary': self.get_performance_summary(),
            'model_interactions': self._get_model_interaction_summary(),
            'llm_usage_summary': self._get_llm_usage_summary(),
            'decision_patterns': self._analyze_decision_patterns(),
            'explainability_metrics': self._calculate_explainability_metrics()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    def _get_memory_usage(self) -> float:
        """메모리 사용량 반환"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return 0.0
    
    def _summarize_input(self, input_data: Any) -> Dict[str, Any]:
        """입력 데이터 요약"""
        if isinstance(input_data, torch.Tensor):
            return {
                'type': 'tensor',
                'shape': list(input_data.shape),
                'dtype': str(input_data.dtype),
                'mean': float(input_data.mean()) if input_data.numel() > 0 else 0,
                'std': float(input_data.std()) if input_data.numel() > 1 else 0
            }
        elif isinstance(input_data, (list, tuple)):
            return {
                'type': type(input_data).__name__,
                'length': len(input_data),
                'first_element_type': type(input_data[0]).__name__ if input_data else None
            }
        else:
            return {
                'type': type(input_data).__name__,
                'summary': str(input_data)[:100]
            }
    
    def _summarize_outputs(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """출력 데이터 요약"""
        summary = {}
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                summary[key] = {
                    'shape': list(value.shape),
                    'mean': float(value.mean()),
                    'std': float(value.std())
                }
            else:
                summary[key] = {
                    'type': type(value).__name__,
                    'value': str(value)[:50]
                }
        return summary
    
    def _calculate_feature_importance(self, outputs: Dict[str, Any]) -> Dict[str, float]:
        """특징 중요도 계산"""
        importance = {}
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor) and value.numel() > 0:
                importance[key] = float(torch.abs(value).mean())
        return importance
    
    def _calculate_confidence(self, outputs: Dict[str, Any]) -> float:
        """신뢰도 계산"""
        confidences = []
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor) and 'prob' in key.lower():
                max_prob = float(torch.max(value))
                confidences.append(max_prob)
        return np.mean(confidences) if confidences else 0.5
    
    def _get_model_interaction_summary(self) -> Dict[str, Any]:
        """모델 상호작용 요약"""
        module_counts = defaultdict(int)
        operation_counts = defaultdict(int)
        
        for entry in self.logs:
            module_counts[entry.module_name] += 1
            operation_counts[entry.operation_type] += 1
        
        return {
            'module_usage': dict(module_counts),
            'operation_types': dict(operation_counts)
        }
    
    def _get_llm_usage_summary(self) -> Dict[str, Any]:
        """LLM 사용량 요약"""
        llm_interactions = [entry for entry in self.logs if entry.llm_interaction]
        
        if not llm_interactions:
            return {'total_interactions': 0}
        
        total_tokens = sum(
            entry.llm_interaction.get('tokens_used', 0) 
            for entry in llm_interactions 
            if entry.llm_interaction.get('tokens_used')
        )
        
        models_used = set(
            entry.llm_interaction.get('model_name', 'unknown')
            for entry in llm_interactions
        )
        
        return {
            'total_interactions': len(llm_interactions),
            'total_tokens': total_tokens,
            'models_used': list(models_used),
            'avg_tokens_per_interaction': total_tokens / len(llm_interactions) if llm_interactions else 0
        }
    
    def _analyze_decision_patterns(self) -> Dict[str, Any]:
        """의사결정 패턴 분석"""
        if not self.decision_traces:
            return {'patterns_found': 0}
        
        patterns = {
            'avg_components_used': np.mean([
                len(trace.model_components) for trace in self.decision_traces
            ]),
            'common_decision_paths': self._find_common_paths(),
            'confidence_distribution': self._analyze_confidence_distribution()
        }
        
        return patterns
    
    def _find_common_paths(self) -> List[Dict[str, Any]]:
        """공통 의사결정 경로 찾기"""
        path_counts = defaultdict(int)
        for trace in self.decision_traces:
            path = ' -> '.join(trace.model_components)
            path_counts[path] += 1
        
        # 상위 5개 경로 반환
        common_paths = sorted(path_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        return [{'path': path, 'count': count} for path, count in common_paths]
    
    def _analyze_confidence_distribution(self) -> Dict[str, float]:
        """신뢰도 분포 분석"""
        confidences = []
        for trace in self.decision_traces:
            if trace.uncertainty_measures and 'confidence' in trace.uncertainty_measures:
                confidences.append(trace.uncertainty_measures['confidence'])
        
        if not confidences:
            return {'mean': 0.0, 'std': 0.0}
        
        return {
            'mean': float(np.mean(confidences)),
            'std': float(np.std(confidences)),
            'min': float(np.min(confidences)),
            'max': float(np.max(confidences))
        }
    
    def _calculate_explainability_metrics(self) -> Dict[str, float]:
        """설명가능성 메트릭 계산"""
        total_operations = len(self.logs)
        explained_operations = len([
            entry for entry in self.logs 
            if entry.explanation_data or entry.llm_interaction
        ])
        
        return {
            'explanation_coverage': explained_operations / total_operations if total_operations > 0 else 0,
            'avg_explanation_depth': self._calculate_avg_explanation_depth(),
            'llm_integration_rate': len([
                entry for entry in self.logs if entry.llm_interaction
            ]) / total_operations if total_operations > 0 else 0
        }
    
    def _calculate_avg_explanation_depth(self) -> float:
        """평균 설명 깊이 계산"""
        depths = []
        for trace in self.decision_traces:
            depth = len(trace.explanation_chain)
            depths.append(depth)
        return float(np.mean(depths)) if depths else 0.0
    
    def _auto_save(self):
        """자동 저장"""
        save_file = self.log_dir / f"logs_backup_{int(time.time())}.json"
        with open(save_file, 'w', encoding='utf-8') as f:
            logs_data = [asdict(entry) for entry in self.logs[-self.auto_save_interval:]]
            json.dump(logs_data, f, indent=2, ensure_ascii=False)
    
    def _flush_to_disk(self):
        """디스크로 플러시"""
        overflow = len(self.logs) - self.max_logs_in_memory + 1000
        overflow_logs = self.logs[:overflow]
        
        flush_file = self.log_dir / f"overflow_{int(time.time())}.json"
        with open(flush_file, 'w', encoding='utf-8') as f:
            logs_data = [asdict(entry) for entry in overflow_logs]
            json.dump(logs_data, f, indent=2, ensure_ascii=False)
        
        self.logs = self.logs[overflow:]

# 전역 XAI 로거 인스턴스
xai_logger = XAILogger()

# 데코레이터들
def xai_trace(module_name: str, operation_type: str = "inference"):
    """XAI 추적 데코레이터"""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            with xai_logger.trace_operation(module_name, operation_type) as operation_id:
                result = func(*args, **kwargs)
                
                # 결과가 텐서인 경우 shape 로그
                if isinstance(result, torch.Tensor):
                    for entry in reversed(xai_logger.logs):
                        if entry.operation_id == operation_id:
                            entry.output_shape = tuple(result.shape)
                            break
                elif isinstance(result, dict):
                    # 딕셔너리 결과인 경우 설명 데이터 생성
                    explanation = xai_logger.create_explanation_chain(
                        result, args[1] if len(args) > 1 else None, module_name
                    )
                    for entry in reversed(xai_logger.logs):
                        if entry.operation_id == operation_id:
                            entry.explanation_data = explanation
                            break
                
                return result
        return wrapper
    return decorator

def xai_decision_point(decision_id: str = None):
    """XAI 의사결정 지점 데코레이터"""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            nonlocal decision_id
            if decision_id is None:
                decision_id = str(uuid.uuid4())
            
            # 입력 캡처
            input_features = {}
            if args:
                input_features['args'] = [
                    xai_logger._summarize_input(arg) for arg in args[:3]  # 처음 3개만
                ]
            if kwargs:
                input_features['kwargs'] = {
                    k: xai_logger._summarize_input(v) for k, v in list(kwargs.items())[:5]
                }
            
            # 함수 실행
            result = func(*args, **kwargs)
            
            # 의사결정 추적 생성
            trace = XAIDecisionTrace(
                decision_id=decision_id,
                timestamp=datetime.now().isoformat(),
                model_components=[func.__module__, func.__name__],
                input_features=input_features,
                intermediate_outputs={},
                final_decision=xai_logger._summarize_input(result),
                explanation_chain=[]
            )
            
            xai_logger.add_decision_trace(trace)
            return result
        return wrapper
    return decorator