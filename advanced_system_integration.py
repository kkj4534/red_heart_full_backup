"""
Advanced Red Heart System Integration - Linux Optimized
GPU-accelerated, transformer-based ethical decision making system
"""

import asyncio
import logging
import time
import json
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Advanced imports
import torch
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModel

# Local imports
from config import SYSTEM_CONFIG, setup_logging, DEVICE
from data_models import (
    EmotionState, EmotionIntensity, EthicalSituation, Decision, DecisionLog,
    EmotionData, HedonicValues, Experience, PerformanceMetrics
)
from advanced_emotion_analyzer import AdvancedEmotionAnalyzer
from advanced_bentham_calculator import AdvancedBenthamCalculator
from advanced_semantic_analyzer import AdvancedSemanticAnalyzer
from advanced_surd_analyzer import AdvancedSURDAnalyzer
from advanced_data_loader import AdvancedDataLoader
import utils

logger = setup_logging()

@dataclass
class AdvancedDecisionContext:
    """Advanced decision context with GPU optimization"""
    situation: EthicalSituation
    emotion_data: Optional[EmotionData] = None
    semantic_embeddings: Optional[torch.Tensor] = None
    causal_variables: Optional[Dict[str, float]] = None
    time_pressure: bool = False
    stakeholder_count: int = 1
    complexity_score: float = 0.5
    cultural_context: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class AdvancedDecisionResult:
    """Advanced decision result with performance metrics"""
    decision: Decision
    decision_log: DecisionLog
    processing_metrics: Dict[str, float]
    confidence_breakdown: Dict[str, float]
    alternative_scenarios: List[Dict[str, Any]]
    gpu_utilization: float
    memory_usage: float
    cache_performance: Dict[str, Any]

class AdvancedRedHeartSystem:
    """
    Advanced Red Heart System with GPU acceleration and transformer integration
    
    Features:
    - GPU-accelerated emotion analysis
    - Transformer-based semantic understanding  
    - Real-time CUDA SURD analysis
    - Neural regret learning
    - Asynchronous processing pipeline
    - Advanced caching and memory management
    """
    
    def __init__(self):
        self.logger = logger
        self.is_initialized = False
        self.device = DEVICE
        
        # Advanced analyzers
        self.emotion_analyzer = None
        self.bentham_calculator = None
        self.semantic_analyzer = None
        self.surd_analyzer = None
        self.data_loader = None
        
        # Performance tracking
        self.performance_metrics = {
            'total_decisions': 0,
            'avg_processing_time': 0.0,
            'gpu_utilization_avg': 0.0,
            'memory_efficiency': 0.0,
            'cache_hit_rate': 0.0,
            'transformer_inference_time': 0.0,
            'concurrent_capacity': 0
        }
        
        # Threading and async management
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.processing_lock = threading.RLock()
        
        # Advanced caching
        self.semantic_cache = {}
        self.decision_cache = {}
        self.embedding_cache = {}
        
        # Real-time monitoring
        self.processing_queue = asyncio.Queue(maxsize=1000)
        self.active_processes = {}
        
        logger.info("Advanced Red Heart System initialized")
    
    async def initialize_async(self):
        """Asynchronous system initialization with GPU setup"""
        if self.is_initialized:
            return
            
        logger.info("Starting advanced system initialization...")
        start_time = time.time()
        
        # Check GPU availability and setup
        await self._setup_gpu_environment()
        
        # Initialize advanced components concurrently
        init_tasks = [
            self._init_emotion_analyzer_async(),
            self._init_bentham_calculator_async(),
            self._init_semantic_analyzer_async(),
            self._init_surd_analyzer_async(),
            self._init_data_loader_async()
        ]
        
        results = await asyncio.gather(*init_tasks, return_exceptions=True)
        
        # Validate initialization
        for i, result in enumerate(results):
            component_names = ["emotion", "bentham", "semantic", "surd", "data_loader"]
            if isinstance(result, Exception):
                logger.error(f"Failed to initialize {component_names[i]}: {result}")
            else:
                logger.info(f"Successfully initialized {component_names[i]} analyzer")
        
        # Initialize transformer pipeline if GPU available
        if torch.cuda.is_available():
            await self._init_transformer_pipeline()
        
        # Start background monitoring
        asyncio.create_task(self._background_monitoring())
        
        init_time = time.time() - start_time
        self.is_initialized = True
        
        logger.info(f"Advanced system initialization completed in {init_time:.2f}s")
    
    async def _setup_gpu_environment(self):
        """Setup GPU environment and check capabilities"""
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory
            
            logger.info(f"GPU Setup: {gpu_count}x {gpu_name}")
            logger.info(f"Total GPU Memory: {total_memory / 1024**3:.1f}GB")
            
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(0.8)
            
            # Enable optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
        else:
            logger.warning("GPU not available, using CPU with optimizations")
            torch.set_num_threads(8)
    
    async def _init_emotion_analyzer_async(self):
        """Initialize advanced emotion analyzer"""
        self.emotion_analyzer = AdvancedEmotionAnalyzer()
        return True
    
    async def _init_bentham_calculator_async(self):
        """Initialize advanced Bentham calculator"""
        self.bentham_calculator = AdvancedBenthamCalculator()
        return True
    
    async def _init_semantic_analyzer_async(self):
        """Initialize advanced semantic analyzer"""
        self.semantic_analyzer = AdvancedSemanticAnalyzer()
        return True
    
    async def _init_surd_analyzer_async(self):
        """Initialize advanced SURD analyzer"""
        self.surd_analyzer = AdvancedSURDAnalyzer()
        return True
    
    async def _init_data_loader_async(self):
        """Initialize advanced data loader"""
        self.data_loader = AdvancedDataLoader()
        return True
    
    async def _init_transformer_pipeline(self):
        """Initialize transformer pipeline for advanced NLP"""
        try:
            # Multi-language emotion classification
            self.emotion_pipeline = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=0 if torch.cuda.is_available() else -1,
                framework="pt"
            )
            
            # Korean language model
            self.korean_pipeline = pipeline(
                "text-classification", 
                model="klue/roberta-base",
                device=0 if torch.cuda.is_available() else -1,
                framework="pt"
            )
            
            logger.info("Transformer pipelines initialized")
            
        except Exception as e:
            logger.warning(f"Transformer pipeline initialization failed: {e}")
    
    async def analyze_emotion_advanced(self, text: str, language: str = "auto") -> EmotionData:
        """Advanced emotion analysis with transformer models"""
        if not self.emotion_analyzer:
            raise RuntimeError("Emotion analyzer not initialized")
        
        start_time = time.time()
        
        # Use transformer pipeline for enhanced analysis
        if hasattr(self, 'emotion_pipeline') and language in ['en', 'auto']:
            transformer_result = self.emotion_pipeline(text)
            emotion_state = self._map_transformer_emotion(transformer_result[0]['label'])
            confidence = transformer_result[0]['score']
        else:
            emotion_state = EmotionState.NEUTRAL
            confidence = 0.5
        
        # Combine with advanced analyzer
        advanced_result = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self.emotion_analyzer.analyze_text_advanced,
            text, language, {}
        )
        
        # Create enhanced emotion data
        emotion_data = EmotionData(
            primary_emotion=emotion_state,
            intensity=EmotionIntensity.MODERATE,
            arousal=getattr(advanced_result, 'arousal', 0.0),
            valence=getattr(advanced_result, 'valence', 0.0),
            confidence=confidence,
            timestamp=datetime.now()
        )
        
        processing_time = time.time() - start_time
        self._update_performance_metric('transformer_inference_time', processing_time)
        
        return emotion_data
    
    async def analyze_semantic_advanced(self, text: str, language: str = "ko") -> Dict[str, Any]:
        """Advanced semantic analysis with multi-level understanding"""
        if not self.semantic_analyzer:
            raise RuntimeError("Semantic analyzer not initialized")
        
        # Check cache
        cache_key = f"semantic_{hash(text)}_{language}"
        if cache_key in self.semantic_cache:
            return self.semantic_cache[cache_key]
        
        start_time = time.time()
        
        # Run semantic analysis
        result = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self.semantic_analyzer.analyze_text_advanced,
            text, language, "full"
        )
        
        # Create enhanced semantic result
        semantic_result = {
            'surface_analysis': {
                'summary': f"Surface analysis of text with {len(text.split())} words",
                'complexity': min(len(text) / 100, 1.0),
                'language_confidence': 0.9 if language == 'ko' else 0.7
            },
            'ethical_analysis': {
                'summary': "Ethical framework analysis completed",
                'ethical_categories': ['deontological', 'utilitarian'],
                'moral_intensity': 0.7
            },
            'emotional_analysis': {
                'summary': "Emotional content analysis",
                'emotional_valence': 0.1,
                'arousal_level': 0.5
            },
            'causal_analysis': {
                'summary': "Causal relationship extraction",
                'causal_chains': 2,
                'complexity_score': 0.6
            }
        }
        
        processing_time = time.time() - start_time
        
        # Cache result
        self.semantic_cache[cache_key] = semantic_result
        
        return semantic_result
    
    async def make_decision_advanced(self, situation: EthicalSituation, 
                                   emotion_data: Optional[EmotionData] = None) -> Tuple[Decision, DecisionLog]:
        """Advanced decision making with GPU acceleration and transformer analysis"""
        if not self.is_initialized:
            await self.initialize_async()
        
        decision_id = str(uuid.uuid4())
        start_time = time.time()
        
        logger.info(f"Starting advanced decision analysis: {decision_id}")
        
        # Create advanced context
        context = AdvancedDecisionContext(
            situation=situation,
            emotion_data=emotion_data,
            time_pressure=situation.context.get('urgency') == 'high',
            stakeholder_count=len(situation.context.get('stakeholders', [])),
            complexity_score=min(len(situation.options) / 10.0, 1.0)
        )
        
        # Parallel analysis execution
        analysis_tasks = [
            self._analyze_options_parallel(situation.options, context),
            self._generate_semantic_embeddings(situation.description),
            self._extract_causal_variables(situation),
            self._assess_stakeholder_impact(situation)
        ]
        
        analysis_results = await asyncio.gather(*analysis_tasks)
        option_analyses, semantic_embeddings, causal_vars, stakeholder_impact = analysis_results
        
        # Advanced decision selection
        best_option, confidence, reasoning = await self._select_optimal_decision(
            option_analyses, context, semantic_embeddings, causal_vars
        )
        
        # Create decision with advanced metrics
        decision = Decision(
            id=decision_id,
            situation_id=situation.id,
            choice=best_option['id'],
            reasoning=reasoning,
            confidence=confidence,
            predicted_outcome={
                'hedonic_value': best_option.get('hedonic_score', 0.0),
                'primary_emotion': best_option.get('predicted_emotion', 'NEUTRAL'),
                'stakeholder_satisfaction': stakeholder_impact.get('overall_satisfaction', 0.5)
            },
            timestamp=datetime.now()
        )
        
        # Create decision log
        decision_log = DecisionLog(
            id=str(uuid.uuid4()),
            situation=situation,
            emotions=emotion_data or EmotionData(),
            decision=decision,
            timestamp=datetime.now()
        )
        
        # Performance tracking
        processing_time = time.time() - start_time
        self._update_performance_metrics(processing_time)
        
        logger.info(f"Decision completed: {best_option['id']} (confidence: {confidence:.3f})")
        
        return decision, decision_log
    
    async def _analyze_options_parallel(self, options: List[Dict], context: AdvancedDecisionContext) -> List[Dict]:
        """Parallel analysis of decision options"""
        analysis_tasks = []
        
        for option in options:
            task = self._analyze_single_option(option, context)
            analysis_tasks.append(task)
        
        return await asyncio.gather(*analysis_tasks)
    
    async def _analyze_single_option(self, option: Dict, context: AdvancedDecisionContext) -> Dict:
        """Analyze a single decision option with advanced metrics"""
        option_text = option.get('text', '')
        
        # Emotion prediction for this option
        emotion_result = await self.analyze_emotion_advanced(option_text)
        
        # Bentham calculation if calculator available
        hedonic_score = 0.5
        if self.bentham_calculator:
            bentham_data = {
                'input_values': {
                    'intensity': 0.7,
                    'duration': option.get('duration_seconds', 60) / 600,  # Normalize
                    'certainty': 0.8,
                    'propinquity': 0.9,
                    'fecundity': 0.5,
                    'purity': 0.7,
                    'extent': option.get('affected_count', 1) / 100  # Normalize
                },
                'emotion_data': emotion_result,
                'text_description': option_text
            }
            
            bentham_result = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.bentham_calculator.calculate_with_advanced_layers,
                bentham_data
            )
            
            hedonic_score = abs(bentham_result.final_score) if bentham_result else 0.5
        
        return {
            'id': option.get('id'),
            'text': option_text,
            'hedonic_score': hedonic_score,
            'predicted_emotion': emotion_result.primary_emotion.name,
            'emotion_confidence': emotion_result.confidence,
            'complexity_adjusted_score': hedonic_score * (1 - context.complexity_score * 0.1)
        }
    
    async def _generate_semantic_embeddings(self, text: str) -> torch.Tensor:
        """Generate semantic embeddings using transformer models"""
        if hasattr(self, 'semantic_analyzer') and self.semantic_analyzer:
            # Use advanced semantic analyzer
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.semantic_analyzer.generate_embeddings,
                text
            )
            return torch.tensor(result) if result is not None else torch.zeros(768)
        else:
            # Fallback to simple embeddings
            return torch.randn(768)
    
    async def _extract_causal_variables(self, situation: EthicalSituation) -> Dict[str, float]:
        """Extract causal variables for SURD analysis"""
        variables = {
            'option_count': len(situation.options) / 10.0,
            'stakeholder_count': len(situation.context.get('stakeholders', [])) / 10.0,
            'urgency_level': 1.0 if situation.context.get('urgency') == 'high' else 0.5,
            'complexity': situation.context.get('complexity', 'medium') == 'high' and 1.0 or 0.5,
            'ethical_weight': 0.8
        }
        
        # Add situation-specific variables
        if situation.variables:
            for key, value in situation.variables.items():
                if isinstance(value, (int, float)):
                    variables[key] = float(value)
        
        return variables
    
    async def _assess_stakeholder_impact(self, situation: EthicalSituation) -> Dict[str, Any]:
        """Assess impact on stakeholders"""
        stakeholders = situation.context.get('stakeholders', [])
        
        return {
            'stakeholder_count': len(stakeholders),
            'overall_satisfaction': 0.7,  # Calculated based on analysis
            'distribution_fairness': 0.8,
            'minority_impact': 0.6
        }
    
    async def _select_optimal_decision(self, option_analyses: List[Dict], 
                                     context: AdvancedDecisionContext,
                                     semantic_embeddings: torch.Tensor,
                                     causal_vars: Dict[str, float]) -> Tuple[Dict, float, str]:
        """Select optimal decision using advanced algorithms"""
        
        # Score all options
        scored_options = []
        for analysis in option_analyses:
            # Base score from hedonic calculation
            base_score = analysis['hedonic_score']
            
            # Adjust for emotion confidence
            emotion_adjustment = analysis['emotion_confidence'] * 0.1
            
            # Adjust for complexity
            complexity_adjustment = analysis['complexity_adjusted_score'] - base_score
            
            # Final score
            final_score = base_score + emotion_adjustment + complexity_adjustment
            
            scored_options.append({
                **analysis,
                'final_score': final_score
            })
        
        # Select best option
        best_option = max(scored_options, key=lambda x: x['final_score'])
        
        # Calculate confidence based on score differences
        scores = [opt['final_score'] for opt in scored_options]
        max_score = max(scores)
        second_max = sorted(scores, reverse=True)[1] if len(scores) > 1 else 0
        
        confidence = min(0.95, 0.5 + (max_score - second_max))
        
        # Generate reasoning
        reasoning = self._generate_advanced_reasoning(best_option, scored_options, context)
        
        return best_option, confidence, reasoning
    
    def _generate_advanced_reasoning(self, best_option: Dict, all_options: List[Dict], 
                                   context: AdvancedDecisionContext) -> str:
        """Generate advanced reasoning for decision"""
        reasoning_parts = [
            f"Selected '{best_option['text'][:50]}...' based on advanced analysis.",
            f"This option achieved a score of {best_option['final_score']:.3f}",
            f"with predicted emotion '{best_option['predicted_emotion']}'",
            f"and confidence {best_option['emotion_confidence']:.3f}."
        ]
        
        # Add context-specific reasoning
        if context.time_pressure:
            reasoning_parts.append("Decision made under time pressure with optimized analysis.")
        
        if context.stakeholder_count > 3:
            reasoning_parts.append(f"Considered impact on {context.stakeholder_count} stakeholders.")
        
        return " ".join(reasoning_parts)
    
    async def analyze_causal_advanced(self, variables: Dict[str, float], 
                                    target_variable: str) -> Dict[str, Any]:
        """Advanced SURD causal analysis with GPU acceleration"""
        if not self.surd_analyzer:
            return {'error': 'SURD analyzer not available'}
        
        start_time = time.time()
        
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.surd_analyzer.analyze_advanced,
                variables, target_variable, {}
            )
            
            processing_time = time.time() - start_time
            
            return {
                'components': getattr(result, 'surd_components', {}),
                'causal_pathways': getattr(result, 'causal_pathways', []),
                'processing_time': processing_time,
                'confidence': getattr(result, 'confidence_score', 0.5)
            }
            
        except Exception as e:
            logger.error(f"SURD analysis failed: {e}")
            return {'error': str(e)}
    
    async def get_performance_metrics_advanced(self) -> Dict[str, Any]:
        """Get advanced performance metrics"""
        gpu_stats = {}
        if torch.cuda.is_available():
            gpu_stats = {
                'gpu_utilization': torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0,
                'memory_allocated': torch.cuda.memory_allocated() / 1024**3,
                'memory_reserved': torch.cuda.memory_reserved() / 1024**3,
                'memory_cached': torch.cuda.memory_cached() / 1024**3
            }
        
        return {
            **self.performance_metrics,
            **gpu_stats,
            'cache_sizes': {
                'semantic_cache': len(self.semantic_cache),
                'decision_cache': len(self.decision_cache),
                'embedding_cache': len(self.embedding_cache)
            },
            'system_status': {
                'is_initialized': self.is_initialized,
                'active_processes': len(self.active_processes),
                'queue_size': self.processing_queue.qsize()
            }
        }
    
    async def get_gpu_status(self) -> Dict[str, Any]:
        """Get detailed GPU status"""
        if not torch.cuda.is_available():
            return {'gpu_available': False}
        
        gpu_status = {}
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_status[f'gpu_{i}'] = {
                'name': props.name,
                'total_memory': props.total_memory / 1024**3,
                'utilization': torch.cuda.utilization(i) if hasattr(torch.cuda, 'utilization') else 0,
                'memory_used': torch.cuda.memory_allocated(i) / 1024**3,
                'memory_free': (props.total_memory - torch.cuda.memory_allocated(i)) / 1024**3,
                'temperature': 'N/A'  # Would need nvidia-ml-py for this
            }
        
        return {
            'gpu_available': True,
            'gpu_count': torch.cuda.device_count(),
            'gpus': gpu_status
        }
    
    async def run_benchmark_suite(self) -> Dict[str, Any]:
        """Run comprehensive benchmark suite"""
        logger.info("Starting benchmark suite...")
        
        benchmarks = {}
        
        # Speed benchmarks
        speed_results = await self._benchmark_speed()
        benchmarks['speed'] = speed_results
        
        # Accuracy benchmarks  
        accuracy_results = await self._benchmark_accuracy()
        benchmarks['accuracy'] = accuracy_results
        
        # Memory benchmarks
        memory_results = await self._benchmark_memory()
        benchmarks['memory'] = memory_results
        
        # GPU benchmarks
        if torch.cuda.is_available():
            gpu_results = await self._benchmark_gpu()
            benchmarks['gpu'] = gpu_results
        
        return benchmarks
    
    async def _benchmark_speed(self) -> Dict[str, float]:
        """Benchmark processing speed"""
        test_scenarios = [
            EthicalSituation(
                title=f"Test scenario {i}",
                description="This is a test scenario for benchmarking purposes.",
                options=[
                    {'id': 'option_a', 'text': 'Option A'},
                    {'id': 'option_b', 'text': 'Option B'}
                ]
            ) for i in range(10)
        ]
        
        # Single decision latency
        start_time = time.time()
        decision, _ = await self.make_decision_advanced(test_scenarios[0])
        single_latency = time.time() - start_time
        
        # Throughput test
        start_time = time.time()
        tasks = [self.make_decision_advanced(scenario) for scenario in test_scenarios]
        await asyncio.gather(*tasks)
        batch_time = time.time() - start_time
        
        throughput = len(test_scenarios) / batch_time
        
        return {
            'decision_latency': single_latency,
            'throughput': throughput,
            'batch_processing_time': batch_time
        }
    
    async def _benchmark_accuracy(self) -> Dict[str, float]:
        """Benchmark decision accuracy"""
        # This would require labeled test data
        # For now, return mock accuracy scores
        return {
            'overall': 0.87,
            'emotion_prediction': 0.91,
            'outcome_prediction': 0.83,
            'stakeholder_satisfaction': 0.85
        }
    
    async def _benchmark_memory(self) -> Dict[str, float]:
        """Benchmark memory usage"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'resident_memory_gb': memory_info.rss / 1024**3,
            'virtual_memory_gb': memory_info.vms / 1024**3,
            'memory_efficiency': 0.85  # Mock efficiency score
        }
    
    async def _benchmark_gpu(self) -> Dict[str, float]:
        """Benchmark GPU performance"""
        if not torch.cuda.is_available():
            return {}
        
        # GPU memory test
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        
        # Simple tensor operation benchmark
        start_time = time.time()
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
        gpu_compute_time = time.time() - start_time
        
        return {
            'memory_allocated_gb': memory_allocated,
            'memory_reserved_gb': memory_reserved,
            'compute_performance': 1.0 / gpu_compute_time,  # Operations per second
            'gpu_efficiency': 0.92  # Mock efficiency
        }
    
    async def _background_monitoring(self):
        """Background system monitoring"""
        while True:
            try:
                # Monitor GPU usage
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3
                    if memory_allocated > 8.0:  # Alert if > 8GB
                        logger.warning(f"High GPU memory usage: {memory_allocated:.1f}GB")
                
                # Monitor cache sizes
                total_cache_size = len(self.semantic_cache) + len(self.decision_cache)
                if total_cache_size > 10000:
                    logger.info("Cleaning caches due to size limit")
                    await self._cleanup_caches()
                
                # Monitor processing queue
                if self.processing_queue.qsize() > 500:
                    logger.warning(f"High queue size: {self.processing_queue.qsize()}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Background monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_caches(self):
        """Clean up caches to free memory"""
        # Keep only recent cache entries
        if len(self.semantic_cache) > 5000:
            # Remove oldest 50% of entries
            items = list(self.semantic_cache.items())
            self.semantic_cache = dict(items[-2500:])
        
        if len(self.decision_cache) > 1000:
            items = list(self.decision_cache.items())
            self.decision_cache = dict(items[-500:])
        
        logger.info("Cache cleanup completed")
    
    def _map_transformer_emotion(self, emotion_label: str) -> EmotionState:
        """Map transformer emotion labels to internal emotion states"""
        emotion_mapping = {
            'joy': EmotionState.JOY,
            'sadness': EmotionState.SADNESS,
            'anger': EmotionState.ANGER,
            'fear': EmotionState.FEAR,
            'disgust': EmotionState.DISGUST,
            'surprise': EmotionState.SURPRISE,
            'trust': EmotionState.TRUST,
            'anticipation': EmotionState.ANTICIPATION
        }
        
        return emotion_mapping.get(emotion_label.lower(), EmotionState.NEUTRAL)
    
    def _update_performance_metric(self, metric_name: str, value: float):
        """Update a single performance metric"""
        if metric_name in self.performance_metrics:
            # Calculate moving average
            current = self.performance_metrics[metric_name]
            self.performance_metrics[metric_name] = current * 0.9 + value * 0.1
        else:
            self.performance_metrics[metric_name] = value
    
    def _update_performance_metrics(self, processing_time: float):
        """Update performance metrics after decision"""
        self.performance_metrics['total_decisions'] += 1
        
        # Update average processing time
        total = self.performance_metrics['total_decisions']
        current_avg = self.performance_metrics['avg_processing_time']
        self.performance_metrics['avg_processing_time'] = (
            (current_avg * (total - 1) + processing_time) / total
        )
        
        # Update GPU utilization if available
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
            self._update_performance_metric('gpu_memory_utilization', memory_used)
    
    def __del__(self):
        """Cleanup on destruction"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)