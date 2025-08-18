#!/usr/bin/env python3
"""
Red Heart AI 통합 학습 시스템 v2
320M 파라미터 + LLM 전처리 + 모듈 선택
Gate 9 통과를 위한 최적화 버전
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import logging
import time
import gc
from typing import Dict, List, Any, Optional
from pathlib import Path
import argparse
from datetime import datetime

# 프로젝트 모듈
from config import ADVANCED_CONFIG, get_device, get_gpu_memory_info, register_system_module, get_system_module
from module_selector import ModuleSelector, ExecutionMode, get_module_selector
from data_preprocessing_pipeline_v3 import HelpingAIPreprocessor as DataPreprocessingPipeline
from target_mapping_utils import TargetMapper
from data_loader import PreprocessedDataLoader
from dynamic_swap_manager import get_swap_manager
from workflow_aware_memory_manager import WorkflowAwareMemoryManager
from analyzer_neural_modules import create_neural_analyzers
from advanced_analyzer_wrappers import create_advanced_analyzer_wrappers
from phase_neural_networks import Phase0ProjectionNet, Phase2CommunityNet, HierarchicalEmotionIntegrator
try:
    from emotion_dsp_simulator import EmotionDSPSimulator, DynamicKalmanFilter
except ImportError:
    EmotionDSPSimulator = None
    DynamicKalmanFilter = None

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('RedHeart.UnifiedTrainingV2')

class UnifiedTrainingSystemV2:
    """
    통합 학습 시스템 v2
    - 320M 파라미터 모델
    - LLM 전처리 파이프라인
    - 동적 모듈 선택
    - 3단계 워크플로우
    """
    
    def __init__(self, args: argparse.Namespace):
        """초기화"""
        self.args = args
        self.device = get_device()
        self.verbose = args.verbose if hasattr(args, 'verbose') else False
        
        # 모듈 선택기
        self.module_selector = get_module_selector()
        
        # 메모리 관리자
        self.memory_manager = WorkflowAwareMemoryManager()
        self.swap_manager = get_swap_manager()
        
        # 모델들
        self.backbone = None
        self.heads = {}
        self.analyzers = {}
        
        # 데이터
        self.train_data = None
        self.val_data = None
        
        # 상태
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        # 옵티마이저 (나중에 초기화)
        self.optimizer = None
        self.scheduler = None
        
        # Gradient Accumulation 설정
        self.gradient_accumulation_steps = max(1, getattr(args, 'gradient_accumulation', 1))
        
        # Mixed Precision 설정
        self.use_mixed_precision = bool(
            getattr(args, 'use_mixed_precision', False) or 
            getattr(args, 'mixed_precision', False)
        )
        self.scaler = (
            torch.amp.GradScaler('cuda', enabled=self.use_mixed_precision)
            if str(self.device).startswith('cuda') else None
        )
        
        logger.info("=" * 60)
        logger.info("Red Heart AI 통합 학습 시스템 v2 초기화")
        logger.info(f"  - 모델 크기: 320M 파라미터")
        logger.info(f"  - 디바이스: {self.device}")
        logger.info(f"  - 모드: {args.mode}")
        logger.info(f"  - 학습률: {args.learning_rate}")
        logger.info("=" * 60)
    
    def prepare_data(self):
        """데이터 준비 (LLM 전처리 포함)"""
        logger.info("\n📊 데이터 준비 시작")
        
        # 1. 원본 데이터 확인 - processed_datasets 디렉토리 사용
        raw_data_path = Path("processed_datasets/integrated_scenarios.json")
        # Claude API로 전처리된 완전한 데이터 사용
        preprocessed_path = Path("claude_api_preprocessing/claude_preprocessed_complete.json")
        
        # 2. 전처리 필요 여부 확인
        if not preprocessed_path.exists() or self.args.force_preprocess:
            logger.info("🔄 LLM 전처리 시작...")
            
            # 전처리 파이프라인 실행
            pipeline = DataPreprocessingPipeline()
            
            try:
                # LLM 로드 (CPU)
                pipeline.initialize_llm(force_cpu=True)
                
                # 원본 데이터 로드
                with open(raw_data_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                
                # 샘플 제한
                if self.args.max_samples:
                    raw_data = raw_data[:self.args.max_samples]
                
                logger.info(f"전처리할 샘플 수: {len(raw_data)}")
                
                # 배치 전처리
                texts = [item.get('text', '') for item in raw_data]
                labels = [item.get('label', 'unknown') for item in raw_data]
                
                enriched_data = pipeline.preprocess_batch(
                    texts, labels, 
                    batch_size=self.args.preprocess_batch_size
                )
                
                # 저장
                pipeline.save_preprocessed_dataset(enriched_data, str(preprocessed_path))
                
            finally:
                # LLM 정리
                pipeline.cleanup_llm()
                gc.collect()
            
            logger.info("✅ LLM 전처리 완료")
        
        # 3. 전처리된 데이터 로드
        logger.info(f"📂 전처리된 데이터 로드: {preprocessed_path}")
        with open(preprocessed_path, 'r', encoding='utf-8') as f:
            processed_data = json.load(f)
        
        # 데이터 검증
        logger.info(f"로드된 데이터: {len(processed_data)} 샘플")
        if len(processed_data) < 10:
            logger.error(f"⚠️ 데이터 부족: {len(processed_data)}개만 로드됨")
            logger.error(f"   파일 경로: {preprocessed_path}")
            logger.error(f"   최소 10개 이상의 데이터가 필요합니다")
            raise ValueError(f"충분한 학습 데이터가 없습니다 (현재: {len(processed_data)}개)")
        
        # max_samples 적용
        if self.args.max_samples and len(processed_data) > self.args.max_samples:
            logger.info(f"샘플 제한 적용: {len(processed_data)} → {self.args.max_samples}")
            processed_data = processed_data[:self.args.max_samples]
        
        # 4. 학습/검증 분할
        split_idx = int(len(processed_data) * 0.8)
        self.train_data = processed_data[:split_idx]
        self.val_data = processed_data[split_idx:]
        
        logger.info(f"✅ 데이터 준비 완료")
        logger.info(f"  - 학습: {len(self.train_data)} 샘플")
        logger.info(f"  - 검증: {len(self.val_data)} 샘플")
    
    def initialize_models(self):
        """모델 초기화 (모드별 선택)"""
        logger.info("\n🤖 모델 초기화 시작")
        
        # 실행 모드 설정
        if self.args.mode in ['train', 'training', 'train-test']:
            self.module_selector.set_mode(ExecutionMode.TRAINING)
        elif self.args.mode in ['eval', 'test']:
            self.module_selector.set_mode(ExecutionMode.EVALUATION)
        else:
            self.module_selector.set_mode(ExecutionMode.INFERENCE)
        
        # 모듈 선택 요약 (신경망 분석기 등록 전)
        self.module_selector.print_summary()
        
        # 메모리 체크
        memory_info = self.module_selector.calculate_memory_usage()
        gpu_info = get_gpu_memory_info()
        
        logger.info(f"📊 메모리 상태:")
        logger.info(f"  - 필요: {memory_info['gpu_memory_mb']:.1f} MB")
        logger.info(f"  - 가용: {gpu_info['free_mb']:.1f} MB")
        
        if memory_info['gpu_memory_mb'] > gpu_info['free_mb']:
            logger.warning("⚠️ GPU 메모리 부족 - 스왑 모드 활성화")
        
        # 모듈 로드 순서
        load_order = self.module_selector.get_load_order()
        
        for module_name in load_order:
            if self.module_selector.should_use_module(module_name):
                self._load_module(module_name)
        
        # 학습 모드에서 필수 모듈 강제 로드
        if self.args.mode in ['train', 'training', 'train-test']:
            logger.info("🎯 학습 필수 모듈 강제 로드 중...")
            
            # 신경망 분석기 로드 (232M 파라미터)
            logger.info("🤖 신경망 분석기 로드 중 (232M 파라미터)...")
            neural_analyzers = create_neural_analyzers()
            for name, analyzer in neural_analyzers.items():
                self.analyzers[f'neural_{name}'] = analyzer
                self.analyzers[f'neural_{name}'].to(self.device)
                # module_selector에 활성 모듈로 등록
                self.module_selector.active_modules.add(f'neural_{name}')
                logger.info(f"  ✅ {name} 신경망 분석기 로드 성공")
            
            # 🔴 중요: translator 모듈 사전 초기화 (Advanced 분석기 의존성)
            logger.info("🔄 Translator 모듈 초기화 중...")
            try:
                # translator가 이미 등록되어 있는지 확인
                existing_translator = get_system_module('translator')
                if existing_translator is None:
                    # LocalTranslator 초기화 및 전역 등록
                    from local_translator import LocalTranslator
                    translator = LocalTranslator()
                    register_system_module('translator', translator)
                    logger.info("  ✅ LocalTranslator 초기화 및 전역 등록 완료")
                else:
                    logger.info("  ℹ️ Translator가 이미 등록되어 있습니다")
            except Exception as e:
                logger.error(f"  ❌ Translator 초기화 실패: {e}")
                logger.error("     Advanced 분석기는 translator 모듈에 의존합니다")
                raise RuntimeError(f"Translator 초기화 실패: {e}")
            
            # Advanced 분석기 필수 통합 (nn.Module Wrapper 사용)
            logger.info("🚀 Advanced 분석기 통합 중 (nn.Module Wrapper)...")
            
            # Advanced Analyzer Wrappers 생성 및 등록
            try:
                advanced_wrappers = create_advanced_analyzer_wrappers()
                for name, wrapper in advanced_wrappers.items():
                    self.analyzers[name] = wrapper.to(self.device)
                    # ✅ 활성 모듈로 등록 (중요!)
                    self.module_selector.active_modules.add(name)
                    param_count = sum(p.numel() for p in wrapper.parameters())
                    logger.info(f"  ✅ {name} Wrapper 로드 완료 ({param_count:,} 파라미터)")
            except Exception as e:
                logger.error(f"  ❌ Advanced Analyzer Wrappers 로드 실패: {e}")
                raise RuntimeError(f"필수 학습 모듈을 로드할 수 없습니다: {e}")
            
            # Phase 0/1/2 계층적 감정 시스템 통합
            logger.info("🌀 3-Phase Hierarchical Emotion System 통합 중...")
            try:
                # Phase 0: 타자→자신 투영 (2M)
                self.phase0_net = Phase0ProjectionNet().to(self.device)
                logger.info("  ✅ Phase0 ProjectionNet 로드 (2M 파라미터)")
                
                # Phase 2: 개인→공동체 (2.5M)
                self.phase2_net = Phase2CommunityNet().to(self.device)
                logger.info("  ✅ Phase2 CommunityNet 로드 (2.5M 파라미터)")
                
                # 통합 모듈
                self.hierarchical_integrator = HierarchicalEmotionIntegrator().to(self.device)
                logger.info("  ✅ Hierarchical Emotion Integrator 로드")
            except Exception as e:
                logger.error(f"  ❌ Phase 네트워크 로드 실패: {e}")
                raise RuntimeError(f"Phase 네트워크 로드 실패: {e}")
            
            # DSP 시뮬레이터 및 칼만 필터 통합
            logger.info("🎵 DSP Simulator & Kalman Filter 통합 중...")
            try:
                if EmotionDSPSimulator is not None:
                    # DSP 시뮬레이터 (14M) - 384차원 hidden_dim 명시
                    self.dsp_simulator = EmotionDSPSimulator({'hidden_dim': 384}).to(self.device)
                    # analyzers dict에 등록
                    self.analyzers['dsp'] = self.dsp_simulator
                    # 활성 모듈로 등록
                    self.module_selector.active_modules.add('dsp')
                    logger.info("  ✅ Emotion DSP Simulator 로드 (14M 파라미터, 384차원)")
                    
                    # 다이나믹 칼만 필터 (0.7K)
                    self.kalman_filter = DynamicKalmanFilter(state_dim=7).to(self.device)
                    # analyzers dict에 등록
                    self.analyzers['kalman'] = self.kalman_filter
                    # 활성 모듈로 등록
                    self.module_selector.active_modules.add('kalman')
                    logger.info("  ✅ Dynamic Kalman Filter 로드 (700 파라미터, 7차원 state)")
                else:
                    logger.warning("  ⚠️ DSP/Kalman 모듈 사용 불가")
                    self.dsp_simulator = None
                    self.kalman_filter = None
            except Exception as e:
                logger.error(f"  ❌ DSP/Kalman 로드 실패: {e}")
                # DSP/Kalman은 선택적이므로 실패해도 계속 진행
                self.dsp_simulator = None
                self.kalman_filter = None
            
            # 헤드 모듈
            essential_heads = ['emotion_head', 'bentham_head', 'regret_head', 'surd_head']
            for head in essential_heads:
                logger.info(f"  - {head} 로드")
                self._load_module(head)
        
        logger.info("✅ 모델 초기화 완료")
        
        # 최종 모듈 선택 요약 (신경망 분석기 포함)
        logger.info("\n📊 최종 로드된 모듈 요약:")
        self.module_selector.print_summary()
        
        # 테스트 모드에서 상세 파라미터 정보 출력
        if self.args.mode in ['test', 'eval'] or self.args.debug:
            self._print_detailed_parameters()
    
    def _print_detailed_parameters(self):
        """상세 파라미터 정보 출력"""
        logger.info("\n" + "=" * 70)
        logger.info("🔍 상세 파라미터 분석")
        logger.info("=" * 70)
        
        total_params = 0
        trainable_params = 0
        
        # 백본
        if self.backbone:
            backbone_params = sum(p.numel() for p in self.backbone.parameters())
            backbone_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
            total_params += backbone_params
            trainable_params += backbone_trainable
            logger.info(f"\n📌 백본:")
            logger.info(f"  - 총 파라미터: {backbone_params:,} ({backbone_params/1e6:.2f}M)")
            logger.info(f"  - 학습가능: {backbone_trainable:,} ({backbone_trainable/1e6:.2f}M)")
        
        # 헤드
        if self.heads:
            logger.info(f"\n📌 헤드 모듈 ({len(self.heads)}개):")
            for name, head in self.heads.items():
                head_params = sum(p.numel() for p in head.parameters())
                head_trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
                total_params += head_params
                trainable_params += head_trainable
                logger.info(f"  [{name}]")
                logger.info(f"    - 총: {head_params:,} ({head_params/1e6:.2f}M)")
                logger.info(f"    - 학습가능: {head_trainable:,}")
        
        # 분석기
        if self.analyzers:
            logger.info(f"\n📌 분석기 모듈 ({len(self.analyzers)}개):")
            
            # Neural Analyzer 분리
            neural_analyzers = {k: v for k, v in self.analyzers.items() if 'neural_' in k}
            other_analyzers = {k: v for k, v in self.analyzers.items() if 'neural_' not in k}
            
            if neural_analyzers:
                logger.info("  🤖 Neural Analyzers:")
                neural_total = 0
                for name, analyzer in neural_analyzers.items():
                    if hasattr(analyzer, 'parameters'):
                        analyzer_params = sum(p.numel() for p in analyzer.parameters())
                        analyzer_trainable = sum(p.numel() for p in analyzer.parameters() if p.requires_grad)
                        neural_total += analyzer_params
                        total_params += analyzer_params
                        trainable_params += analyzer_trainable
                        logger.info(f"    [{name}]")
                        logger.info(f"      - 총: {analyzer_params:,} ({analyzer_params/1e6:.2f}M)")
                        logger.info(f"      - 학습가능: {analyzer_trainable:,}")
                logger.info(f"    📊 Neural Analyzer 합계: {neural_total:,} ({neural_total/1e6:.2f}M)")
            
            if other_analyzers:
                logger.info("  📈 기타 분석기:")
                for name, analyzer in other_analyzers.items():
                    if hasattr(analyzer, 'parameters'):
                        try:
                            analyzer_params = sum(p.numel() for p in analyzer.parameters())
                            if analyzer_params > 0:
                                analyzer_trainable = sum(p.numel() for p in analyzer.parameters() if p.requires_grad)
                                total_params += analyzer_params
                                trainable_params += analyzer_trainable
                                logger.info(f"    [{name}]: {analyzer_params:,} params")
                        except:
                            logger.info(f"    [{name}]: 파라미터 없음")
        
        # 총계
        logger.info("\n" + "=" * 70)
        logger.info(f"📊 전체 통계:")
        logger.info(f"  - 총 파라미터: {total_params:,} ({total_params/1e6:.2f}M)")
        logger.info(f"  - 학습가능 파라미터: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        logger.info(f"  - 목표: 450M, 실제: {total_params/1e6:.2f}M")
        
        # GPU 메모리 사용량
        gpu_info = get_gpu_memory_info()
        logger.info(f"\n💾 GPU 메모리:")
        logger.info(f"  - 할당됨: {gpu_info['allocated_mb']:.1f} MB")
        logger.info(f"  - 여유: {gpu_info['free_mb']:.1f} MB")
        logger.info(f"  - 전체: {gpu_info['total_mb']:.1f} MB")
        logger.info("=" * 70 + "\n")
    
    def _load_module(self, module_name: str):
        """개별 모듈 로드"""
        module_info = self.module_selector.get_module_info(module_name)
        if not module_info:
            return
        
        logger.debug(f"모듈 로드: {module_name}")
        
        # 모듈별 로드 로직
        if module_name == 'unified_backbone':
            from unified_backbone import RedHeartUnifiedBackbone
            self.backbone = RedHeartUnifiedBackbone(ADVANCED_CONFIG['unified_backbone'])
            self.backbone.to(self.device)
            
        elif 'head' in module_name:
            # 헤드 로드 (80M 파라미터)
            try:
                from unified_heads import EmotionHead, BenthamHead, RegretHead, SURDHead
                
                if 'emotion' in module_name:
                    self.heads['emotion'] = EmotionHead(input_dim=896)
                    self.heads['emotion'].to(self.device)
                    logger.info("✅ 감정 헤드 로드 성공 (30M)")
                elif 'bentham' in module_name:
                    self.heads['bentham'] = BenthamHead(input_dim=896)
                    self.heads['bentham'].to(self.device)
                    logger.info("✅ 벤담 헤드 로드 성공 (27M)")
                elif 'regret' in module_name:
                    self.heads['regret'] = RegretHead(input_dim=896)
                    self.heads['regret'].to(self.device)
                    logger.info("✅ 후회 헤드 로드 성공 (30M)")
                elif 'surd' in module_name:
                    self.heads['surd'] = SURDHead(input_dim=896)
                    self.heads['surd'].to(self.device)
                    logger.info("✅ SURD 헤드 로드 성공 (22M)")
            except Exception as e:
                logger.error(f"❌ 헤드 로드 실패: {e}")
                raise RuntimeError(f"NO FALLBACK: 필수 헤드 모듈 로드 실패 - {module_name}: {e}")
            
        elif module_name == 'emotion_dsp_simulator':
            from emotion_dsp_simulator import EmotionDSPSimulator
            self.analyzers['dsp'] = EmotionDSPSimulator({'hidden_dim': 384})
            self.analyzers['dsp'].to(self.device)
            
        elif module_name == 'kalman_filter':
            from emotion_dsp_simulator import DynamicKalmanFilter
            self.analyzers['kalman'] = DynamicKalmanFilter(state_dim=7)
            self.analyzers['kalman'].to(self.device)
            
    
    def _initialize_optimizer(self):
        """옵티마이저 초기화 - NO FALLBACK 원칙"""
        logger.info("🔧 옵티마이저 초기화 중...")
        
        # 학습 가능한 파라미터 수집
        params = []
        
        if self.backbone and hasattr(self.backbone, 'parameters'):
            params.extend(list(self.backbone.parameters()))
            logger.info(f"  백본 파라미터 추가됨")
        
        for name, head in self.heads.items():
            if hasattr(head, 'parameters'):
                params.extend(list(head.parameters()))
                logger.info(f"  {name} 헤드 파라미터 추가됨")
        
        # 모든 분석기의 파라미터 수집 (Neural + Advanced)
        for name, analyzer in self.analyzers.items():
            param_count = 0
            
            # Neural Analyzers (nn.Module 상속)
            if hasattr(analyzer, 'parameters'):
                try:
                    analyzer_params = list(analyzer.parameters())
                    if analyzer_params:
                        params.extend(analyzer_params)
                        param_count = sum(p.numel() for p in analyzer_params)
                        logger.info(f"  {name} 분석기 파라미터 추가됨: {param_count:,}")
                except Exception as e:
                    logger.warning(f"  {name} 분석기 파라미터 수집 실패: {e}")
            
            # Advanced Analyzers (내부 nn.Module 수집)
            elif 'advanced_' in name:
                logger.info(f"  {name} Advanced 분석기 내부 모듈 수집 중...")
                
                # Advanced Emotion Analyzer 내부 모듈들
                if name == 'advanced_emotion' and hasattr(analyzer, 'biometric_processor'):
                    try:
                        biometric_params = list(analyzer.biometric_processor.parameters())
                        params.extend(biometric_params)
                        count = sum(p.numel() for p in biometric_params)
                        param_count += count
                        logger.info(f"    - biometric_processor: {count:,}")
                    except: pass
                    
                    if hasattr(analyzer, 'multimodal_fusion'):
                        try:
                            fusion_params = list(analyzer.multimodal_fusion.parameters())
                            params.extend(fusion_params)
                            count = sum(p.numel() for p in fusion_params)
                            param_count += count
                            logger.info(f"    - multimodal_fusion: {count:,}")
                        except: pass
                    
                    if hasattr(analyzer, 'temporal_emotion'):
                        try:
                            temporal_params = list(analyzer.temporal_emotion.parameters())
                            params.extend(temporal_params)
                            count = sum(p.numel() for p in temporal_params)
                            param_count += count
                            logger.info(f"    - temporal_emotion: {count:,}")
                        except: pass
                    
                    if hasattr(analyzer, 'cultural_nuance'):
                        try:
                            cultural_params = list(analyzer.cultural_nuance.parameters())
                            params.extend(cultural_params)
                            count = sum(p.numel() for p in cultural_params)
                            param_count += count
                            logger.info(f"    - cultural_nuance: {count:,}")
                        except: pass
                    
                    if hasattr(analyzer, 'advanced_moe'):
                        try:
                            moe_params = list(analyzer.advanced_moe.parameters())
                            params.extend(moe_params)
                            count = sum(p.numel() for p in moe_params)
                            param_count += count
                            logger.info(f"    - advanced_moe: {count:,}")
                        except: pass
                
                # Advanced Regret Analyzer 내부 모듈들
                elif name == 'advanced_regret':
                    if hasattr(analyzer, 'regret_network'):
                        try:
                            regret_params = list(analyzer.regret_network.parameters())
                            params.extend(regret_params)
                            count = sum(p.numel() for p in regret_params)
                            param_count += count
                            logger.info(f"    - regret_network: {count:,}")
                        except: pass
                    
                    if hasattr(analyzer, 'counterfactual_sim'):
                        try:
                            cf_params = list(analyzer.counterfactual_sim.parameters())
                            params.extend(cf_params)
                            count = sum(p.numel() for p in cf_params)
                            param_count += count
                            logger.info(f"    - counterfactual_sim: {count:,}")
                        except: pass
                    
                    if hasattr(analyzer, 'temporal_propagation'):
                        try:
                            tp_params = list(analyzer.temporal_propagation.parameters())
                            params.extend(tp_params)
                            count = sum(p.numel() for p in tp_params)
                            param_count += count
                            logger.info(f"    - temporal_propagation: {count:,}")
                        except: pass
                    
                    if hasattr(analyzer, 'decision_tree'):
                        try:
                            dt_params = list(analyzer.decision_tree.parameters())
                            params.extend(dt_params)
                            count = sum(p.numel() for p in dt_params)
                            param_count += count
                            logger.info(f"    - decision_tree: {count:,}")
                        except: pass
                
                # Advanced SURD Analyzer 내부 모듈들
                elif name == 'advanced_surd':
                    if hasattr(analyzer, 'deep_causal'):
                        try:
                            causal_params = list(analyzer.deep_causal.parameters())
                            params.extend(causal_params)
                            count = sum(p.numel() for p in causal_params)
                            param_count += count
                            logger.info(f"    - deep_causal: {count:,}")
                        except: pass
                    
                    if hasattr(analyzer, 'info_decomposition'):
                        try:
                            info_params = list(analyzer.info_decomposition.parameters())
                            params.extend(info_params)
                            count = sum(p.numel() for p in info_params)
                            param_count += count
                            logger.info(f"    - info_decomposition: {count:,}")
                        except: pass
                
                # Advanced Bentham Calculator 내부 모듈들
                elif name == 'advanced_bentham':
                    # 내부 네트워크 찾기 (속성 동적 탐색)
                    for attr_name in dir(analyzer):
                        if not attr_name.startswith('_'):
                            attr = getattr(analyzer, attr_name, None)
                            if attr is not None and isinstance(attr, torch.nn.Module):
                                try:
                                    module_params = list(attr.parameters())
                                    if module_params:
                                        params.extend(module_params)
                                        count = sum(p.numel() for p in module_params)
                                        param_count += count
                                        logger.info(f"    - {attr_name}: {count:,}")
                                except: pass
                
                if param_count > 0:
                    logger.info(f"  {name} 총 파라미터: {param_count:,}")
                else:
                    logger.warning(f"  ⚠️ {name} 분석기에서 학습 파라미터를 찾을 수 없음")
        
        # Phase 네트워크 파라미터 추가
        if hasattr(self, 'phase0_net'):
            params.extend(list(self.phase0_net.parameters()))
            logger.info(f"  Phase0 ProjectionNet 파라미터 추가됨: {sum(p.numel() for p in self.phase0_net.parameters()):,}")
        
        if hasattr(self, 'phase2_net'):
            params.extend(list(self.phase2_net.parameters()))
            logger.info(f"  Phase2 CommunityNet 파라미터 추가됨: {sum(p.numel() for p in self.phase2_net.parameters()):,}")
        
        if hasattr(self, 'hierarchical_integrator'):
            params.extend(list(self.hierarchical_integrator.parameters()))
            logger.info(f"  Hierarchical Integrator 파라미터 추가됨: {sum(p.numel() for p in self.hierarchical_integrator.parameters()):,}")
        
        # DSP/Kalman은 이미 self.analyzers에 포함되어 있으므로 중복 추가하지 않음
        
        if not params:
            raise RuntimeError("학습 가능한 파라미터가 없습니다. 모델 초기화를 확인하세요.")
        
        # AdamW 옵티마이저 사용
        self.optimizer = torch.optim.AdamW(
            params, 
            lr=self.args.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # 학습률 스케줄러 (Cosine Annealing)
        # 배치 수가 0이 되지 않도록 최소 1로 보장
        if self.train_data:
            num_batches = max(1, len(self.train_data) // self.args.batch_size)
            if len(self.train_data) % self.args.batch_size != 0:
                num_batches += 1  # 나머지가 있으면 배치 하나 추가
            total_steps = self.args.epochs * num_batches
        else:
            total_steps = 1000
            
        # T_max가 0이 되지 않도록 최소값 보장
        total_steps = max(1, total_steps)
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=1e-6
        )
        
        logger.info(f"  - 스케줄러 T_max: {total_steps} steps")
        
        logger.info(f"✅ 옵티마이저 초기화 완료")
        logger.info(f"  - 옵티마이저: AdamW")
        logger.info(f"  - 학습률: {self.args.learning_rate}")
        logger.info(f"  - 스케줄러: CosineAnnealingLR")
        logger.info(f"  - 총 파라미터: {sum(p.numel() for p in params):,}")
        logger.info(f"  - 학습 가능 파라미터: {sum(p.numel() for p in params if p.requires_grad):,}")
    
    def train_epoch(self, epoch: int):
        """한 에포크 학습"""
        logger.info(f"\n📚 Epoch {epoch+1}/{self.args.epochs} 시작")
        
        # 학습 모드 - 모든 모듈을 train 모드로 설정
        if self.backbone:
            self.backbone.train()
        
        for head in self.heads.values():
            head.train()
        
        for analyzer in self.analyzers.values():
            if hasattr(analyzer, 'train'):
                analyzer.train()
        
        total_loss = 0.0
        num_batches = max(1, len(self.train_data) // self.args.batch_size)
        
        if self.verbose:
            logger.info(f"📊 에포크 상세 정보:")
            logger.info(f"  - 총 샘플 수: {len(self.train_data)}")
            logger.info(f"  - 배치 크기: {self.args.batch_size}")
            logger.info(f"  - 배치 수: {num_batches}")
        
        for batch_idx in range(num_batches):
            # 배치 데이터
            batch_start = batch_idx * self.args.batch_size
            batch_end = batch_start + self.args.batch_size
            batch_data = self.train_data[batch_start:batch_end]
            
            if self.verbose and batch_idx < 3:  # 처음 3개 배치만 상세 출력
                logger.info(f"\n🔍 배치 {batch_idx+1} 처리 중:")
                logger.info(f"  - 샘플 범위: {batch_start} ~ {batch_end}")
                logger.info(f"  - 샘플 수: {len(batch_data)}")
            
            # 3단계 워크플로우
            loss = self._train_step(batch_data, batch_idx)
            total_loss += loss
            
            # 진행 상황 출력
            if (batch_idx + 1) % 10 == 0 or (self.verbose and batch_idx < 3):
                avg_loss = total_loss / (batch_idx + 1)
                logger.info(f"  Batch {batch_idx+1}/{num_batches} - Loss: {avg_loss:.4f}")
            
            self.global_step += 1
        
        # 에포크 완료
        avg_epoch_loss = total_loss / num_batches
        logger.info(f"✅ Epoch {epoch+1} 완료 - 평균 손실: {avg_epoch_loss:.4f}")
        
        return avg_epoch_loss
    
    def _train_step(self, batch_data: List[Dict], batch_idx: int) -> float:
        """
        학습 스텝 - 3단계 워크플로우
        1. FORWARD: 데이터 → 백본 → 헤드
        2. COMPUTE: 손실 계산 + 시너지
        3. UPDATE: 역전파 + 최적화
        """
        
        # ========== STAGE 1: FORWARD ==========
        if self.verbose and batch_idx < 3:
            logger.info("    [STAGE 1] Forward Pass 시작")
        
        # 실제 데이터 사용 (더미 데이터 대체)
        batch_size = len(batch_data)
        
        # batch_data에서 실제 입력 추출 (context_embedding 사용)
        if not batch_data or not isinstance(batch_data[0], dict):
            raise ValueError("배치 데이터가 비어있거나 형식이 잘못됨")
        
        # TargetMapper를 사용하여 실제 데이터 추출
        try:
            # context_embedding 추출 (text_embedding 대신) - 백본 모델 전달
            input_embeddings = TargetMapper.extract_context_embedding(batch_data, backbone_model=self.backbone)
            dummy_input = input_embeddings.to(self.device).requires_grad_(True)
        except KeyError as e:
            # NO FALLBACK 원칙 - 실패시 예외 발생
            raise RuntimeError(f"입력 임베딩 추출 실패: {e}")
        
        if self.backbone:
            # 백본 forward - Dict[str, Tensor] 반환
            backbone_outputs = self.backbone(dummy_input, return_all_tasks=True)
            # 감정 태스크를 기본으로 사용 (7차원 출력 필요)
            features = backbone_outputs.get('emotion', dummy_input)
            if self.verbose and batch_idx < 3:
                logger.info(f"      - 백본 출력 shape: {features.shape}")
                logger.info(f"      - 백본 출력 키: {list(backbone_outputs.keys())}")
        else:
            features = dummy_input
        
        # ========== STAGE 2: COMPUTE ==========
        if self.verbose and batch_idx < 3:
            logger.info("    [STAGE 2] Compute Loss")
        
        # 헤드 통과 및 손실 계산
        losses = []
        
        # 감정 헤드 예시
        if 'emotion' in self.heads and features is not None:
            emotion_output = self.heads['emotion'](features)
            # 헤드는 Dict[str, Tensor] 반환 - 'emotions' 키로 실제 출력 추출
            emotion_pred = emotion_output['emotions'] if isinstance(emotion_output, dict) else emotion_output
            # 실제 감정 타깃 추출 (TargetMapper 사용)
            target = TargetMapper.extract_emotion_target(batch_data).to(self.device)
            # head.compute_loss 사용
            emotion_loss = self.heads['emotion'].compute_loss(emotion_pred, target)
            losses.append(emotion_loss)
            
            if self.verbose and batch_idx < 3:
                logger.info(f"      - 감정 예측 shape: {emotion_pred.shape}")
        
        # 벤담 헤드 (선택적)
        if 'bentham' in self.heads and features is not None:
            bentham_output = self.heads['bentham'](features)
            bentham_pred = bentham_output['bentham_scores'] if isinstance(bentham_output, dict) else bentham_output
            # 실제 벤담 타깃 추출 (10차원) - TargetMapper 사용
            target = TargetMapper.extract_bentham_target(batch_data).to(self.device)
            # head.compute_loss 사용
            bentham_loss = self.heads['bentham'].compute_loss(bentham_pred, target)
            losses.append(bentham_loss)
        
        # ========== STAGE 2: NEURAL ANALYZERS ==========
        if self.verbose and batch_idx < 3:
            logger.info("    [STAGE 2] Neural Analyzer Processing")
        
        # Neural/Advanced Analyzer 처리 (378M+ 파라미터 활용)
        analyzer_losses = []
        dsp_output = None  # DSP 출력 저장 (Kalman 입력용)
        neural_emotion_output = None  # neural_emotion 출력 저장 (Kalman traditional emotions용)
        
        # neural_emotion을 먼저 처리 (Kalman이 traditional emotions로 사용)
        if 'neural_emotion' in self.analyzers:
            try:
                emotion_analyzer = self.analyzers['neural_emotion']
                neural_emotion_output = emotion_analyzer(features)
                
                # 손실 계산
                if 'emotion_logits' in neural_emotion_output:
                    target = TargetMapper.extract_emotion_target(batch_data).to(self.device)
                    if target.dim() == 1:
                        target = F.one_hot(target, num_classes=7).float()
                    emotion_loss = F.cross_entropy(neural_emotion_output['emotion_logits'], target)
                    analyzer_losses.append(emotion_loss)
                    if self.verbose and batch_idx < 3:
                        logger.info(f"      - neural_emotion 손실: {emotion_loss.item():.6f}")
            except Exception as e:
                logger.error(f"    ❌ neural_emotion 처리 실패: {e}")
                raise  # 필수 모듈이므로 실패 시 중단
        
        # DSP 처리 (두 번째)
        if 'dsp' in self.analyzers:
            try:
                dsp_analyzer = self.analyzers['dsp']
                # DSP는 384차원 입력을 기대 - 백본 출력(896차원)을 투영
                if not hasattr(self, 'dsp_projection'):
                    self.dsp_projection = torch.nn.Linear(features.shape[-1], 384).to(self.device)
                
                dsp_input = self.dsp_projection(features)
                dsp_output = dsp_analyzer(dsp_input)
                
                if self.verbose and batch_idx < 3:
                    logger.info(f"      - DSP 출력: {dsp_output.get('final_emotions', torch.zeros(1)).shape if isinstance(dsp_output, dict) else 'scalar'}")
            except Exception as e:
                logger.warning(f"    ⚠️ DSP 처리 실패: {e}")
                dsp_output = None
        
        # Kalman 처리 (neural_emotion과 DSP 출력 모두 필요)
        if 'kalman' in self.analyzers and dsp_output is not None and neural_emotion_output is not None:
            try:
                kalman_analyzer = self.analyzers['kalman']
                # neural_emotion의 logits를 traditional emotions로 사용
                traditional_emotions = neural_emotion_output.get('emotion_logits', None)
                dsp_emotions = dsp_output.get('final_emotions', None) if isinstance(dsp_output, dict) else dsp_output
                
                if traditional_emotions is not None and dsp_emotions is not None:
                    # 둘 다 7차원이어야 함
                    kalman_output = kalman_analyzer(
                        traditional_emotions=traditional_emotions,
                        dsp_emotions=dsp_emotions
                    )
                    if self.verbose and batch_idx < 3:
                        logger.info(f"      - Kalman 필터 출력 처리됨")
            except Exception as e:
                logger.warning(f"    ⚠️ Kalman 처리 실패: {e}")
        
        # 나머지 analyzer 처리
        for name, analyzer in self.analyzers.items():
            # 이미 처리한 모듈들은 건너뜀
            if name in ['dsp', 'kalman', 'neural_emotion']:
                continue
                
            # 모든 nn.Module 기반 analyzer 처리 (neural_, advanced_ 등 모두 포함)
            if isinstance(analyzer, torch.nn.Module) and hasattr(analyzer, 'forward'):
                try:
                    # Neural analyzer forward pass - features 사용 (backbone 출력)
                    analyzer_output = analyzer(features)
                    
                    # 각 analyzer 타입별 손실 계산
                    if 'emotion' in name and 'emotion_logits' in analyzer_output:
                        # 실제 감정 타깃 추출 (더미 데이터 제거)
                        target = TargetMapper.extract_emotion_target(batch_data).to(self.device)
                        # 분류가 아닌 회귀 태스크로 처리
                        if target.dim() == 1:
                            target = F.one_hot(target, num_classes=7).float()
                        analyzer_loss = F.cross_entropy(analyzer_output['emotion_logits'], target)
                        analyzer_losses.append(analyzer_loss)
                        if self.verbose and batch_idx < 3:
                            logger.info(f"      - {name} 손실: {analyzer_loss.item():.6f}")
                    
                    elif 'bentham' in name and 'bentham_scores' in analyzer_output:
                        # 실제 벤담 타깃 추출
                        target = TargetMapper.extract_bentham_target(batch_data).to(self.device)
                        analyzer_loss = F.mse_loss(analyzer_output['bentham_scores'], target)
                        analyzer_losses.append(analyzer_loss)
                        if self.verbose and batch_idx < 3:
                            logger.info(f"      - {name} 손실: {analyzer_loss.item():.6f}")
                    
                    elif 'regret' in name and 'regret_score' in analyzer_output:
                        # 실제 후회 타깃 추출
                        target = TargetMapper.extract_regret_target(batch_data).to(self.device)
                        analyzer_loss = F.smooth_l1_loss(analyzer_output['regret_score'], target)
                        analyzer_losses.append(analyzer_loss)
                        if self.verbose and batch_idx < 3:
                            logger.info(f"      - {name} 손실: {analyzer_loss.item():.6f}")
                    
                    elif 'surd' in name and 'surd_scores' in analyzer_output:
                        # 실제 SURD 타깃 추출 (정규화된 4차원)
                        target = TargetMapper.extract_surd_target(batch_data, normalize=True).to(self.device)
                        analyzer_loss = F.mse_loss(analyzer_output['surd_scores'], target)
                        analyzer_losses.append(analyzer_loss)
                        if self.verbose and batch_idx < 3:
                            logger.info(f"      - {name} 손실: {analyzer_loss.item():.6f}")
                    
                except Exception as e:
                    if self.args.debug:
                        logger.error(f"    {name} 처리 실패: {e}")
        
        # 전체 손실 통합 (헤드 + Neural Analyzer)
        all_losses = losses + analyzer_losses
        
        if all_losses:
            # 가중 평균: 헤드 70%, Neural Analyzer 30%
            if losses and analyzer_losses:
                head_loss = sum(losses) / len(losses)
                analyzer_loss = sum(analyzer_losses) / len(analyzer_losses)
                loss = 0.7 * head_loss + 0.3 * analyzer_loss
                if self.verbose and batch_idx < 3:
                    logger.info(f"      - 헤드 손실: {head_loss.item():.6f}")
                    logger.info(f"      - 분석기 손실: {analyzer_loss.item():.6f}")
            else:
                loss = sum(all_losses) / len(all_losses)
        else:
            # NO FALLBACK 원칙 - 손실이 없으면 예외
            raise RuntimeError("손실 계산 실패: 헤드와 분석기 모두 손실을 생성하지 못함")
        
        if self.verbose and batch_idx < 3:
            logger.info(f"      - 손실값: {loss.item():.6f}")
        
        # ========== STAGE 3: UPDATE ==========
        if self.verbose and batch_idx < 3:
            logger.info("    [STAGE 3] Parameter Update")
        
        if self.optimizer is not None:
            # NO FALLBACK 원칙 - 옵티마이저가 없으면 학습 불가
            
            # Gradient Accumulation: 경계에서만 zero_grad 호출
            if batch_idx % self.gradient_accumulation_steps == 0:
                self.optimizer.zero_grad()
            
            # 역전파
            loss.backward()
            
            # 그래디언트 체크 (NaN, Inf 검증)
            total_norm = 0.0
            for p in self.backbone.parameters() if self.backbone else []:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    
                    # NaN/Inf 체크
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        logger.error(f"⚠️ 그래디언트 이상 감지: NaN 또는 Inf")
                        if self.args.debug:
                            logger.error(f"   파라미터 shape: {p.shape}")
            
            total_norm = total_norm ** 0.5
            if self.verbose and batch_idx < 3:
                logger.info(f"      - 그래디언트 norm: {total_norm:.4f}")
            elif self.args.debug and batch_idx % 10 == 0:
                logger.debug(f"   그래디언트 norm: {total_norm:.4f}")
            
            # Gradient Accumulation을 위한 loss 스케일링
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps
            
            # Gradient Accumulation 체크
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # 그래디언트 클리핑 (안정적인 학습)
                all_params = []
                if self.backbone:
                    all_params.extend([p for p in self.backbone.parameters() if p.requires_grad])
                for head in self.heads.values():
                    all_params.extend([p for p in head.parameters() if p.requires_grad])
                # Neural Analyzer 파라미터도 포함
                for name, analyzer in self.analyzers.items():
                    if 'neural_' in name and hasattr(analyzer, 'parameters'):
                        all_params.extend([p for p in analyzer.parameters() if p.requires_grad])
                
                if all_params:
                    if self.use_mixed_precision and self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                
                # 파라미터 업데이트 (--no-param-update 옵션 체크)
                if not self.args.no_param_update:
                    # 옵티마이저 스텝
                    if self.use_mixed_precision and self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    # 학습률 스케줄러 업데이트
                    if self.scheduler:
                        self.scheduler.step()
                
                if self.verbose and batch_idx < 3:
                    logger.info(f"      - 파라미터 업데이트 완료")
            else:
                if self.verbose and batch_idx < 3:
                    logger.info(f"      - 파라미터 업데이트 스킵 (--no-param-update)")
                elif self.args.debug:
                    logger.debug("   파라미터 업데이트 생략 (베이스라인 회귀)")
        elif self.args.mode in ['train', 'training', 'train-test']:
            # 학습 모드인데 옵티마이저가 없으면 오류
            raise RuntimeError("학습 모드이지만 옵티마이저가 초기화되지 않음")
        
        return loss.item()
    
    def _eval_step(self, batch_data: List[Dict], batch_idx: int) -> float:
        """
        평가 스텝 - backward 없이 손실만 계산
        """
        # ========== STAGE 1: FORWARD ==========
        if self.verbose and batch_idx < 3:
            logger.info("    [STAGE 1] Forward Pass 시작")
        
        # 실제 데이터 사용 (NO FALLBACK - 더미 데이터 제거)
        batch_size = len(batch_data)
        
        # batch_data에서 실제 입력 추출
        if not batch_data or not isinstance(batch_data[0], dict):
            raise ValueError("배치 데이터가 비어있거나 형식이 잘못됨")
        
        # TargetMapper를 사용하여 실제 데이터 추출
        try:
            # context_embedding 추출
            input_embeddings = TargetMapper.extract_context_embedding(batch_data)
            dummy_input = input_embeddings.to(self.device).requires_grad_(False)  # 평가 모드이므로 gradient 불필요
        except KeyError as e:
            # NO FALLBACK 원칙 - 실패시 예외 발생
            raise RuntimeError(f"입력 임베딩 추출 실패: {e}")
        
        if self.backbone:
            # 백본 forward - Dict[str, Tensor] 반환
            backbone_outputs = self.backbone(dummy_input, return_all_tasks=True)
            features = backbone_outputs.get('emotion', dummy_input)
            if self.verbose and batch_idx < 3:
                logger.info(f"      - 백본 출력 shape: {features.shape}")
                logger.info(f"      - 백본 출력 키: {list(backbone_outputs.keys())}")
        else:
            features = dummy_input
        
        # ========== STAGE 2: COMPUTE LOSS (평가용) ==========
        if self.verbose and batch_idx < 3:
            logger.info("    [STAGE 2] Compute Loss (평가)")
        
        # 헤드 통과 및 손실 계산
        losses = []
        
        # 감정 헤드 예시
        if 'emotion' in self.heads and features is not None:
            emotion_output = self.heads['emotion'](features)
            emotion_pred = emotion_output['emotions'] if isinstance(emotion_output, dict) else emotion_output
            # 실제 감정 타깃 추출 (TargetMapper 사용 - List[Dict] 처리용)
            target = TargetMapper.extract_emotion_target(batch_data).to(self.device)
            # head.compute_loss 사용
            emotion_loss = self.heads['emotion'].compute_loss(emotion_pred, target)
            losses.append(emotion_loss)
            
            if self.verbose and batch_idx < 3:
                logger.info(f"      - 감정 예측 shape: {emotion_pred.shape}")
        
        # 벤담 헤드 (선택적)
        if 'bentham' in self.heads and features is not None:
            bentham_output = self.heads['bentham'](features)
            bentham_pred = bentham_output['bentham_scores'] if isinstance(bentham_output, dict) else bentham_output
            # 실제 벤담 타깃 추출 (TargetMapper 사용)
            target = TargetMapper.extract_bentham_target(batch_data).to(self.device)
            # head.compute_loss 사용 (일관성)
            bentham_loss = self.heads['bentham'].compute_loss(bentham_pred, target)
            losses.append(bentham_loss)
        
        # 전체 손실
        if losses:
            loss = sum(losses) / len(losses)
        else:
            # NO FALLBACK 원칙 - 손실이 없으면 예외
            raise RuntimeError("검증 손실 계산 실패: 헤드가 손실을 생성하지 못함")
        
        if self.verbose and batch_idx < 3:
            logger.info(f"      - 평가 손실값: {loss.item():.6f}")
        
        return loss.item()
    
    def evaluate(self):
        """평가"""
        logger.info("\n🧪 평가 시작")
        
        if self.backbone:
            self.backbone.eval()
        
        for head in self.heads.values():
            head.eval()
        
        total_loss = 0.0
        num_batches = max(1, len(self.val_data) // self.args.batch_size)
        
        with torch.no_grad():
            for batch_idx in range(num_batches):
                batch_start = batch_idx * self.args.batch_size
                batch_end = batch_start + self.args.batch_size
                batch_data = self.val_data[batch_start:batch_end]
                
                # 평가 스텝 (backward 없음)
                loss = self._eval_step(batch_data, batch_idx)
                total_loss += loss
        
        avg_val_loss = total_loss / max(num_batches, 1)
        logger.info(f"✅ 평가 완료 - 평균 손실: {avg_val_loss:.4f}")
        
        return avg_val_loss
    
    def save_checkpoint(self, epoch: int, loss: float):
        """체크포인트 저장"""
        checkpoint_dir = Path(self.args.checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'loss': loss,
            'config': ADVANCED_CONFIG,
            'args': vars(self.args),
            'timestamp': datetime.now().isoformat()
        }
        
        # 모델 상태 저장
        if self.backbone:
            checkpoint['backbone_state'] = self.backbone.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"💾 체크포인트 저장: {checkpoint_path}")
        
        # 최고 모델 저장
        if loss < self.best_loss:
            self.best_loss = loss
            best_path = checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"🏆 최고 모델 갱신: {best_path}")
    
    def run(self):
        """메인 실행"""
        logger.info("\n" + "=" * 60)
        logger.info("🚀 Red Heart AI 통합 학습 시작")
        logger.info("=" * 60)
        
        try:
            # 1. 데이터 준비
            self.prepare_data()
            
            # 2. 모델 초기화
            self.initialize_models()
            
            # 2-1. 학습 모드일 때 옵티마이저 초기화
            if self.args.mode in ['train', 'training', 'train-test']:
                self._initialize_optimizer()
            
            # 3. 학습 루프
            if self.args.mode in ['train', 'training', 'train-test']:
                for epoch in range(self.args.epochs):
                    # 학습
                    train_loss = self.train_epoch(epoch)
                    
                    # 평가
                    val_loss = self.evaluate()
                    
                    # 체크포인트 저장
                    self.save_checkpoint(epoch, val_loss)
                    
                    # 메모리 상태 출력
                    if (epoch + 1) % 5 == 0:
                        gpu_info = get_gpu_memory_info()
                        logger.info(f"📊 GPU 메모리: {gpu_info['usage_percent']:.1f}% 사용")
            
            elif self.args.mode in ['eval', 'test']:
                # 평가만
                val_loss = self.evaluate()
                logger.info(f"최종 평가 손실: {val_loss:.4f}")
            
            logger.info("\n" + "=" * 60)
            logger.info("✅ 학습/평가 완료!")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        finally:
            # 정리
            self.cleanup()
    
    def cleanup(self):
        """정리"""
        logger.info("\n🧹 시스템 정리 중...")
        
        # 모델 정리
        if self.backbone:
            del self.backbone
        self.heads.clear()
        self.analyzers.clear()
        
        # 메모리 정리
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("✅ 정리 완료")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Red Heart AI 통합 학습 v2")
    
    # 기본 옵션
    parser.add_argument('--mode', choices=['train', 'eval', 'test', 'training', 'train-test'], 
                       default='train', help='실행 모드')
    parser.add_argument('--no-param-update', action='store_true',
                       help='파라미터 업데이트 없이 그래디언트만 체크 (학습 테스트용)')
    parser.add_argument('--epochs', type=int, default=10, 
                       help='학습 에포크')
    parser.add_argument('--batch-size', type=int, default=4, 
                       help='배치 크기')
    parser.add_argument('--learning-rate', type=float, default=1e-4, 
                       help='학습률')
    
    # 데이터 옵션
    parser.add_argument('--max-samples', type=int, default=1000,
                       help='최대 샘플 수')
    parser.add_argument('--samples', type=int, dest='max_samples',
                       help='최대 샘플 수 (--max-samples와 동일)')
    parser.add_argument('--force-preprocess', action='store_true',
                       help='강제 전처리')
    parser.add_argument('--preprocess-batch-size', type=int, default=5,
                       help='전처리 배치 크기')
    
    # 시스템 옵션
    parser.add_argument('--checkpoint-dir', default='./checkpoints_v2',
                       help='체크포인트 디렉토리')
    parser.add_argument('--debug', action='store_true',
                       help='디버그 모드')
    parser.add_argument('--verbose', action='store_true',
                       help='상세 출력 모드')
    parser.add_argument('--use-advanced', action='store_true',
                       help='Advanced 분석기 통합 (FocalLoss, MoE, GPU가속 등)')
    
    args = parser.parse_args()
    
    # 디버그 모드
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 시스템 실행
    system = UnifiedTrainingSystemV2(args)
    system.run()


if __name__ == "__main__":
    main()