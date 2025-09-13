#!/usr/bin/env python3
"""
Claude API 전용 독립 추론 시스템
DSM과 완전히 분리된 독립적인 워크플로우
"""

import asyncio
import logging
import torch
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import json
import sys
import os

# 경로 설정
sys.path.append('/mnt/c/large_project/linux_red_heart')

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('ClaudeInference')

class ClaudeInferenceSystem:
    """Claude API 전용 추론 시스템 - DSM 없이 독립 실행"""
    
    def __init__(self, epoch: int = 50, debug: bool = False):
        self.epoch = epoch
        self.debug = debug
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 컴포넌트 초기화 플래그
        self.unified_model = None
        self.translator = None
        self.neural_analyzers = {}
        self.advanced_wrappers = {}
        self.llm_engine = None
        self.emotion_hierarchy_processor = None  # Regret Circuit 추가
        
        logger.info("=" * 70)
        logger.info("🚀 Claude API 독립 추론 시스템 초기화")
        logger.info("=" * 70)
        logger.info(f"📌 에폭: {epoch}")
        logger.info(f"📌 디바이스: {self.device}")
        logger.info(f"📌 디버그 모드: {debug}")
        
    async def initialize(self):
        """시스템 초기화 - 순차적 로드"""
        try:
            logger.info("\n📦 Phase 1: UnifiedModel 로드...")
            await self._load_unified_model()
            
            logger.info("\n📦 Phase 2: 번역기 로드...")
            await self._load_translator()
            
            logger.info("\n📦 Phase 3: Neural Analyzers 로드...")
            await self._load_neural_analyzers()
            
            logger.info("\n📦 Phase 4: Advanced Wrappers 로드...")
            await self._load_advanced_wrappers()
            
            logger.info("\n📦 Phase 5: Emotion-Ethics-Regret Circuit 초기화...")
            await self._load_emotion_circuit()
            
            logger.info("\n📦 Phase 6: Claude API 엔진 초기화...")
            await self._init_claude_api()
            
            logger.info("\n✅ 모든 컴포넌트 초기화 완료!")
            
        except Exception as e:
            logger.error(f"❌ 초기화 실패: {e}", exc_info=True)
            raise
    
    async def _load_unified_model(self):
        """UnifiedModel 로드 - DSM 없이"""
        from training.unified_training_final import UnifiedModel, UnifiedTrainingConfig
        from training.enhanced_checkpoint_manager import EnhancedCheckpointManager
        
        # 체크포인트 매니저
        checkpoint_manager = EnhancedCheckpointManager(
            checkpoint_dir="training/checkpoints_final",
            max_checkpoints=30
        )
        
        # 학습 설정 생성
        train_config = UnifiedTrainingConfig()
        
        # 모델 초기화
        self.unified_model = UnifiedModel(
            config=train_config,
            device=self.device
        )
        
        # 체크포인트 매니저 연결
        self.unified_model.checkpoint_manager = checkpoint_manager
        
        # 체크포인트 로드
        checkpoint_path = f"training/checkpoints_final/checkpoint_epoch_{self.epoch:04d}_*.pt"
        checkpoints = list(Path(".").glob(checkpoint_path))
        
        if checkpoints:
            checkpoint_file = str(checkpoints[0])
            logger.info(f"   체크포인트 로드: {checkpoint_file}")
            
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            
            # 체크포인트 구조에 맞게 로드
            if 'model_state' in checkpoint:
                self.unified_model.load_state_dict(checkpoint['model_state'], strict=False)
                logger.info("   ✅ UnifiedModel 로드 완료")
                logger.info(f"   - Epoch: {checkpoint.get('epoch', 'unknown')}")
                logger.info(f"   - LR: {checkpoint.get('lr', 'unknown')}")
            else:
                logger.error("   ❌ 체크포인트에 'model_state' 키가 없습니다")
                logger.error(f"   체크포인트 키: {list(checkpoint.keys())}")
                raise KeyError("체크포인트 구조가 예상과 다릅니다")
        else:
            logger.warning(f"   ⚠️ 에폭 {self.epoch} 체크포인트 없음")
        
        self.unified_model.eval()
        
        # advanced_wrappers를 UnifiedModel에 전달 (이것을 나중에 로드한 advanced_wrappers로 설정)
        self.unified_model.advanced_wrappers = None  # 나중에 _load_advanced_wrappers에서 설정
        
    async def _load_translator(self):
        """번역기 로드 - Claude 모드에서는 스킵"""
        import os
        
        # Claude 모드에서는 번역기 불필요 (Claude API가 다국어 지원)
        if os.getenv('REDHEART_CLAUDE_MODE') == '1':
            logger.info("   ⚠️ Claude 모드 - 번역기 로드 스킵 (Claude API 다국어 지원)")
            self.translator = None
            return
        
        from config import get_system_module
        
        # main_unified.py에서 이미 등록된 translator를 가져오기
        self.translator = get_system_module('translator')
        
        if self.translator is None:
            # 혹시 등록되지 않았다면 직접 생성
            from local_translator import LocalTranslator
            from config import register_system_module
            
            logger.warning("전역 translator 없음 - 직접 생성")
            self.translator = LocalTranslator()
            register_system_module('translator', self.translator)
            logger.info("   ✅ 번역기 직접 로드 및 등록 완료 (CPU)")
        else:
            logger.info("   ✅ 전역 번역기 재사용 (CPU)")
    
    async def _load_neural_analyzers(self):
        """Neural Analyzers 로드 - CPU에서만"""
        from analyzer_neural_modules import (
            NeuralEmotionAnalyzer,
            NeuralBenthamCalculator,
            NeuralRegretAnalyzer,
            NeuralSURDAnalyzer
        )
        
        # 초기화 (input_dim 기본값 사용)
        self.neural_analyzers = {
            'emotion': NeuralEmotionAnalyzer(),
            'bentham': NeuralBenthamCalculator(),
            'regret': NeuralRegretAnalyzer(),
            'surd': NeuralSURDAnalyzer()
        }
        
        # CPU로 이동
        for analyzer in self.neural_analyzers.values():
            analyzer.to('cpu')
        
        logger.info("   ✅ Neural Analyzers 로드 완료 (CPU)")
        
    async def _load_advanced_wrappers(self):
        """Advanced Wrappers 로드 - CPU에서만"""
        from advanced_analyzer_wrappers import (
            AdvancedEmotionAnalyzerWrapper,
            AdvancedBenthamCalculatorWrapper,
            AdvancedRegretAnalyzerWrapper,
            AdvancedSURDAnalyzerWrapper
        )
        
        # Wrapper 초기화 (파라미터 없이)
        self.advanced_wrappers = {
            'advanced_emotion': AdvancedEmotionAnalyzerWrapper(),
            'advanced_bentham': AdvancedBenthamCalculatorWrapper(),
            'advanced_regret': AdvancedRegretAnalyzerWrapper(),
            'advanced_surd': AdvancedSURDAnalyzerWrapper()
        }
        
        # Wrappers에 필요한 컴포넌트 설정
        for wrapper_name, wrapper in self.advanced_wrappers.items():
            # analyzer 내부에 이미 포함되어 있음
            if hasattr(wrapper, 'analyzer'):
                # translator 설정
                if hasattr(wrapper.analyzer, 'translator'):
                    wrapper.analyzer.translator = self.translator
        
        logger.info("   ✅ Advanced Wrappers 로드 완료 (CPU)")
        
        # UnifiedModel에 advanced_wrappers 전달
        if hasattr(self, 'unified_model'):
            self.unified_model.advanced_wrappers = self.advanced_wrappers
            logger.info("   ✅ UnifiedModel에 advanced_wrappers 연결 완료")
    
    async def _init_claude_api(self):
        """Claude API 엔진 초기화"""
        from llm_module.advanced_llm_engine import AdvancedLLMEngine
        
        # Claude API 엔진 초기화
        self.llm_engine = AdvancedLLMEngine(use_api='claude')
        # AdvancedLLMEngine에는 initialize() 메서드가 없음 - 제거
        
        # Wrappers에 LLM 엔진 연결
        for wrapper in self.advanced_wrappers.values():
            wrapper.llm_engine = self.llm_engine
        
        logger.info("   ✅ Claude API 엔진 초기화 완료")
    
    async def _load_emotion_circuit(self):
        """Emotion-Ethics-Regret Circuit 로드"""
        try:
            from emotion_ethics_regret_circuit import EmotionEthicsRegretCircuit
            
            # Circuit 초기화
            self.emotion_hierarchy_processor = EmotionEthicsRegretCircuit()
            logger.info("   ✅ Emotion-Ethics-Regret Circuit 초기화 완료")
            logger.info("      - 감정-윤리-후회 통합 분석 가능")
            logger.info("      - Circuit 활성화 조건: config.use_emotion_hierarchy=True")
            
        except ImportError as e:
            logger.warning(f"   ⚠️ Emotion Circuit 로드 실패 (선택적): {e}")
            self.emotion_hierarchy_processor = None
        except Exception as e:
            logger.error(f"   ❌ Emotion Circuit 초기화 실패: {e}")
            self.emotion_hierarchy_processor = None
    
    async def inference(self, text: str) -> Dict[str, Any]:
        """추론 실행"""
        logger.info("\n" + "=" * 70)
        logger.info("🎯 추론 시작")
        logger.info("=" * 70)
        logger.info(f"📝 입력 텍스트: {text}")
        
        try:
            # 1. 번역 (한국어 → 영어) - Claude 모드에서는 스킵
            import os
            if os.getenv('REDHEART_CLAUDE_MODE') == '1':
                # Claude 모드: 번역 스킵 (Claude가 다국어 지원)
                en_text = text
                logger.info("⚡ Claude 모드 - 번역 스킵 (원문 그대로 사용)")
            elif self.translator:
                en_text = await self.translator.translate_async(text)
                logger.info(f"🔄 번역 완료: {en_text[:100]}...")
            else:
                en_text = text
            
            # 1.5. LLM을 통한 초기 반사실적 시나리오 요청 (필수)
            counterfactual_scenarios = []
            if not self.llm_engine:
                logger.error("\n❌ LLM 엔진이 초기화되지 않음")
                raise RuntimeError("LLM 엔진이 없으면 반사실적 시나리오 생성 불가. 시스템 무결성 오류")
            
            if self.llm_engine:
                logger.info("\n🎯 LLM 초기 반사실적 시나리오 생성...")
                from llm_module.advanced_llm_engine import LLMRequest, TaskComplexity
                
                scenario_prompt = f"""다음 상황에 대한 3가지 반사실적(counterfactual) 시나리오를 생성하세요.

상황: {text}

각 시나리오는 다른 선택이나 행동을 했을 때의 가능한 결과를 보여주어야 합니다:
1. 낙관적 시나리오: 최선의 선택을 했을 때의 결과
2. 중립적 시나리오: 일반적인 선택을 했을 때의 결과  
3. 비관적 시나리오: 최악의 선택을 했을 때의 결과

반드시 아래와 같은 JSON 배열 형식으로만 응답하세요. 다른 설명이나 텍스트 없이 JSON만 반환하세요:
[
  {{"type": "optimistic", "action": "구체적 행동", "outcome": "예상 결과", "ethical_implications": "윤리적 함의"}},
  {{"type": "neutral", "action": "구체적 행동", "outcome": "예상 결과", "ethical_implications": "윤리적 함의"}},
  {{"type": "pessimistic", "action": "구체적 행동", "outcome": "예상 결과", "ethical_implications": "윤리적 함의"}}
]

중요: JSON 형식만 반환하고, 추가 설명이나 텍스트는 포함하지 마세요."""
                
                scenario_request = LLMRequest(
                    prompt=scenario_prompt,
                    task_type="scenario_generation",
                    complexity=TaskComplexity.MODERATE,
                    max_tokens=2000,  # 3개 시나리오를 완전히 담을 수 있도록 증가
                    context={'mode': 'counterfactual_initial'}
                )
                
                try:
                    scenario_response = await self.llm_engine.generate_async(scenario_request)
                    if scenario_response and scenario_response.success:
                        # JSON 파싱 시도 (개선된 버전)
                        import json
                        import re
                        text_content = scenario_response.generated_text
                        
                        # 여러 JSON 추출 방법 시도
                        json_parsed = False
                        
                        # 방법 1: 전체 텍스트가 JSON인 경우
                        try:
                            # 디버깅: 전체 응답 로깅
                            logger.debug(f"   📝 LLM 전체 응답 ({len(text_content)}자): {text_content}")
                            counterfactual_scenarios = json.loads(text_content)
                            json_parsed = True
                            logger.info("   ✅ 방법 1(전체 텍스트 JSON)로 파싱 성공")
                        except json.JSONDecodeError as e:
                            logger.debug(f"   ⚠️ 방법 1 실패: {e}")
                            pass
                        
                        # 방법 2: JSON 배열 패턴 찾기 (개선된 정규식)
                        if not json_parsed:
                            # 더 유연한 JSON 배열 추출 (중첩된 객체 고려)
                            json_patterns = [
                                r'\[\s*\{[^[\]]*\}\s*\]',  # 간단한 패턴
                                r'\[[\s\S]*?\](?=\s*$|\s*[^\[\]\{\}])',  # 문서 끝의 JSON
                                r'(?:^|\n)\s*\[[\s\S]*?\]\s*(?:\n|$)'  # 줄바꿈 사이의 JSON
                            ]
                            
                            for pattern in json_patterns:
                                matches = re.findall(pattern, text_content, re.MULTILINE | re.DOTALL)
                                for match in matches:
                                    try:
                                        # JSON 문자열 정리
                                        cleaned_json = match.strip()
                                        # 이스케이프 처리 제거 (이미 올바른 JSON이어야 함)
                                        counterfactual_scenarios = json.loads(cleaned_json)
                                        json_parsed = True
                                        logger.info(f"   ✅ 방법 2(정규식 패턴 {idx+1})로 파싱 성공")
                                        break
                                    except json.JSONDecodeError:
                                        continue
                                if json_parsed:
                                    break
                        
                        # 방법 3: 코드 블록 내 JSON 찾기
                        if not json_parsed:
                            code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text_content)
                            if code_block_match:
                                try:
                                    counterfactual_scenarios = json.loads(code_block_match.group(1).strip())
                                    json_parsed = True
                                    logger.info("   ✅ 방법 3(코드 블록)으로 파싱 성공")
                                except json.JSONDecodeError as e:
                                    logger.debug(f"   ⚠️ 방법 3 실패: {e}")
                                    pass
                        
                        # JSON 파싱 성공 여부에 따른 처리
                        if json_parsed and isinstance(counterfactual_scenarios, list) and len(counterfactual_scenarios) > 0:
                            # 시나리오 형식 검증
                            required_fields = ['type', 'action', 'outcome', 'ethical_implications']
                            for idx, scenario in enumerate(counterfactual_scenarios):
                                if not isinstance(scenario, dict):
                                    raise RuntimeError(f"시나리오 {idx}가 딕셔너리가 아님: {type(scenario)}")
                                missing_fields = [f for f in required_fields if f not in scenario]
                                if missing_fields:
                                    raise RuntimeError(f"시나리오 {idx}에 필수 필드 누락: {missing_fields}")
                            
                            logger.info(f"   ✅ {len(counterfactual_scenarios)}개 반사실적 시나리오 생성 및 검증 완료")
                            for scenario in counterfactual_scenarios:
                                logger.info(f"      - {scenario.get('type', 'unknown')}: {scenario.get('action', '')[:50]}...")
                        else:
                            # JSON 파싱 실패 - 시스템 중단
                            logger.error("   ❌ JSON 파싱 실패 - 반사실적 시나리오 생성 불가")
                            logger.error(f"   → LLM 응답: {text_content[:200]}...")
                            raise RuntimeError("반사실적 시나리오 JSON 파싱 실패: LLM이 올바른 형식의 JSON을 반환하지 않음")
                    else:
                        # LLM 응답 실패 - 시스템 중단
                        logger.error("   ❌ LLM 응답 실패 - 반사실적 시나리오 생성 불가")
                        if scenario_response:
                            logger.error(f"   → 에러: {scenario_response.error}")
                        raise RuntimeError(f"LLM 응답 실패: {scenario_response.error if scenario_response else 'Unknown error'}")
                except Exception as e:
                    logger.error(f"   ❌ 초기 시나리오 생성 실패: {e}")
                    import traceback
                    logger.error(f"   → 상세 에러: {traceback.format_exc()}")
                    raise RuntimeError(f"반사실적 시나리오 생성 중 예외 발생: {e}")
            
            # 2. UnifiedModel 추론 (GPU)
            logger.info("\n📊 UnifiedModel 추론...")
            with torch.no_grad():
                # GPU로 임시 이동
                self.unified_model = self.unified_model.to(self.device)
                
                # 실제 텍스트를 임베딩으로 변환
                from sentence_transformer_singleton import SentenceTransformerManager
                stm = SentenceTransformerManager()
                
                # get_model()을 통해 모델 프록시 획득
                logger.info("   📝 텍스트 임베딩 생성 중...")
                model_proxy = stm.get_model('sentence-transformers/paraphrase-multilingual-mpnet-base-v2', device=str(self.device))
                # convert_to_tensor=False로 설정하여 리스트로 받고 직접 텐서 변환
                embeddings = model_proxy.encode(en_text, convert_to_tensor=False)
                # 리스트를 텐서로 변환
                embeddings = torch.tensor(embeddings).to(self.device)
                embeddings = embeddings.unsqueeze(0) if len(embeddings.shape) == 1 else embeddings
                
                # 각 태스크별로 실제 추론 수행
                logger.info("   🎭 감정 분석 추론...")
                emotion_outputs = self.unified_model(embeddings, task='emotion', return_all=True)
                emotion_output = emotion_outputs.get('head', list(emotion_outputs.values())[0]).cpu().numpy()
                
                logger.info("   ⚖️ 벤담 계산 추론...")
                bentham_outputs = self.unified_model(embeddings, task='bentham', return_all=True)
                bentham_output = bentham_outputs.get('head', list(bentham_outputs.values())[0]).cpu().numpy()
                
                logger.info("   😔 후회 분석 추론...")
                regret_outputs = self.unified_model(embeddings, task='regret', return_all=True)
                regret_output = regret_outputs.get('head', list(regret_outputs.values())[0]).cpu().numpy()
                
                logger.info("   🔍 SURD 분석 추론...")
                surd_outputs = self.unified_model(embeddings, task='surd', return_all=True)
                surd_output = surd_outputs.get('head', list(surd_outputs.values())[0]).cpu().numpy()
                
                # 임베딩을 CPU로 이동하여 보존 (Advanced Analysis에서 사용)
                embeddings_cpu = embeddings.cpu()
                
                # 모델을 다시 CPU로
                self.unified_model = self.unified_model.to('cpu')
                torch.cuda.empty_cache()
                
            logger.info("   ✅ UnifiedModel 추론 완료")
            
            # 3. Advanced Wrappers 분석 (CPU)
            logger.info("\n🧠 Advanced Analysis...")
            
            results = {
                'text': text,
                'translated': en_text,
                'unified_outputs': {
                    'emotion': emotion_output.tolist(),
                    'bentham': bentham_output.tolist(),
                    'regret': regret_output.tolist(),
                    'surd': surd_output.tolist()
                },
                'advanced_analysis': {}
            }
            
            # 각 Wrapper 실행 - nn.Module의 forward 메서드 호출
            # 모든 wrapper에 임베딩(896차원)을 전달해야 함
            if self.advanced_wrappers.get('advanced_emotion'):
                try:
                    # 임베딩과 텍스트를 함께 전달
                    emotion_result = self.advanced_wrappers['advanced_emotion'](embeddings_cpu, text=text, language='ko')
                    results['advanced_analysis']['emotion'] = {k: v.tolist() if torch.is_tensor(v) else v for k, v in emotion_result.items()}
                    logger.info("   ✅ 감정 분석 완료")
                except Exception as e:
                    logger.error(f"   ❌ 감정 분석 실패: {e}")
                    import traceback
                    logger.error(f"   → 상세 에러: {traceback.format_exc()}")
                    # 감정 분석은 Circuit의 필수 의존성이므로 실패 시 중단
                    raise RuntimeError(f"Advanced Emotion Analyzer 실행 실패: {e}")
            
            if self.advanced_wrappers.get('advanced_bentham'):
                try:
                    # 임베딩을 전달 (896차원)
                    bentham_result = self.advanced_wrappers['advanced_bentham'](embeddings_cpu)
                    results['advanced_analysis']['bentham'] = {k: v.tolist() if torch.is_tensor(v) else v for k, v in bentham_result.items()}
                    logger.info("   ✅ 벤담 계산 완료")
                except Exception as e:
                    logger.error(f"   ❌ 벤담 계산 실패: {e}")
                    import traceback
                    logger.error(f"   → 상세 에러: {traceback.format_exc()}")
                    # 벤담 계산은 윤리적 평가의 핵심이므로 실패 시 중단
                    raise RuntimeError(f"Advanced Bentham Calculator 실행 실패: {e}")
            
            # Regret Wrapper 호출 추가 (반사실적 시나리오 포함)
            if self.advanced_wrappers.get('advanced_regret'):
                logger.info("   🔄 Regret 분석 중...")
                # 임베딩을 전달 (896차원)
                try:
                    # 반사실적 시나리오가 있으면 전달
                    if counterfactual_scenarios:
                        regret_result = self.advanced_wrappers['advanced_regret'](
                            embeddings_cpu, 
                            scenarios=counterfactual_scenarios,
                            text=text
                        )
                        logger.info(f"      - {len(counterfactual_scenarios)}개 반사실적 시나리오와 함께 분석")
                    else:
                        regret_result = self.advanced_wrappers['advanced_regret'](embeddings_cpu)
                    
                    results['advanced_analysis']['regret'] = {k: v.tolist() if torch.is_tensor(v) else v for k, v in regret_result.items()}
                    
                    # 반사실적 시나리오 결과 저장
                    if counterfactual_scenarios:
                        results['counterfactual_analysis'] = {
                            'initial_scenarios': counterfactual_scenarios,
                            'regret_based_evaluation': regret_result.get('counterfactual_evaluation', {})
                        }
                    
                    logger.info("   ✅ Regret 분석 완료")
                except Exception as e:
                    logger.error(f"   ❌ Regret 분석 실패: {e}")
                    import traceback
                    logger.error(f"   → 상세 에러: {traceback.format_exc()}")
                    # Regret 분석은 반사실적 추론의 핵심이므로 실패 시 중단
                    raise RuntimeError(f"Advanced Regret Analyzer 실행 실패: {e}")
            
            # SURD Wrapper 호출 추가
            if self.advanced_wrappers.get('advanced_surd'):
                logger.info("   🔄 SURD 분석 중...")
                # 임베딩을 전달 (896차원)
                try:
                    surd_result = self.advanced_wrappers['advanced_surd'](embeddings_cpu)
                    results['advanced_analysis']['surd'] = {k: v.tolist() if torch.is_tensor(v) else v for k, v in surd_result.items()}
                    logger.info("   ✅ SURD 분석 완료")
                except Exception as e:
                    logger.error(f"   ❌ SURD 분석 실패: {e}")
                    import traceback
                    logger.error(f"   → 상세 에러: {traceback.format_exc()}")
                    # SURD 분석은 불확실성 평가의 핵심이므로 실패 시 중단
                    raise RuntimeError(f"Advanced SURD Analyzer 실행 실패: {e}")
            
            # Three View Analysis Result를 dict로 변환하는 헬퍼 함수
            def three_view_to_dict(obj):
                """ThreeViewAnalysisResult 객체를 JSON 직렬화 가능한 dict로 변환"""
                from dataclasses import fields, is_dataclass
                from enum import Enum
                
                if is_dataclass(obj):
                    # dataclass를 dict로 변환 (재귀적으로)
                    result = {}
                    for field in fields(obj):
                        field_value = getattr(obj, field.name)
                        result[field.name] = three_view_to_dict(field_value)
                    return result
                elif isinstance(obj, Enum):
                    return obj.value
                elif isinstance(obj, dict):
                    return {k: three_view_to_dict(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [three_view_to_dict(item) for item in obj]
                else:
                    return obj
            
            # 3.4. Three View System 분석 (추가)
            three_view_results = None
            try:
                from three_view_scenario_system import ThreeViewScenarioSystem
                logger.info("\n🔮 Three View Scenario System 분석...")
                
                three_view_system = ThreeViewScenarioSystem()
                
                # 반사실적 시나리오나 텍스트로 3뷰 분석 수행
                if counterfactual_scenarios:
                    # 각 시나리오에 대한 3뷰 분석
                    logger.info(f"   📊 {len(counterfactual_scenarios)}개 시나리오에 대한 3뷰 분석 중...")
                    scenario_analyses = []
                    for scenario in counterfactual_scenarios:
                        scenario_text = f"{scenario.get('action', '')} - {scenario.get('outcome', '')}"
                        analysis = await three_view_system.analyze_three_view_scenarios({
                            'text': scenario_text,
                            'scenario_type': scenario.get('type', 'unknown')
                        })
                        scenario_analyses.append({
                            'scenario_type': scenario.get('type'),
                            'three_views': three_view_to_dict(analysis)
                        })
                    three_view_results = scenario_analyses
                    logger.info("   ✅ 반사실적 시나리오 3뷰 분석 완료")
                else:
                    # 원본 텍스트에 대한 3뷰 분석
                    logger.info("   📊 원본 텍스트에 대한 3뷰 분석 중...")
                    analysis_result = await three_view_system.analyze_three_view_scenarios({
                        'text': text,
                        'mode': 'direct_analysis'
                    })
                    three_view_results = three_view_to_dict(analysis_result)
                    logger.info("   ✅ 원본 텍스트 3뷰 분석 완료")
                
                # 결과 저장
                if three_view_results:
                    results['three_view_analysis'] = three_view_results
                    logger.info(f"   💡 3뷰 시스템 분석 결과 저장 완료")
                    
            except ImportError as e:
                logger.warning(f"   ⚠️ Three View System 모듈 로드 실패: {e}")
            except Exception as e:
                logger.error(f"   ❌ Three View System 분석 실패: {e}")
                # 3뷰 시스템 실패는 전체 실패가 아니므로 계속 진행
            
            # 3.5. Emotion-Ethics-Regret Circuit 실행 (추가)
            circuit_result = None
            if self.emotion_hierarchy_processor and results.get('advanced_analysis', {}).get('emotion'):
                logger.info("\n🎭 Emotion-Ethics-Regret Circuit 처리...")
                try:
                    from emotion_ethics_regret_circuit import CircuitDecisionContext
                    from data_models import EmotionData, EmotionState, EmotionIntensity
                    
                    # 감정 데이터 준비
                    emotion_analysis = results['advanced_analysis']['emotion']
                    emotions_array = emotion_analysis.get('emotions', [[0,0,0,0,0,0,0.5]])[0]
                    
                    # 주요 감정 식별
                    emotion_labels = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral']
                    max_idx = np.argmax(emotions_array)
                    primary_emotion = EmotionState(max_idx if max_idx < 7 else 6)
                    
                    # EmotionData 생성
                    emotion_data = EmotionData(
                        primary_emotion=primary_emotion,
                        intensity=EmotionIntensity.MODERATE,
                        arousal=float(np.mean(emotions_array[:6])),  # 중립 제외 평균
                        valence=float(emotions_array[0] - emotions_array[1]),  # 기쁨 - 슬픔
                        dominance=float(emotions_array[2]),  # 분노를 dominance로
                        confidence=float(np.max(emotions_array)),
                        language='ko'
                    )
                    
                    # 이해관계자 추출
                    stakeholders = []
                    if "친구" in text:
                        stakeholders.append("친구")
                    if "가족" in text:
                        stakeholders.append("가족")
                    if "동료" in text or "회사" in text:
                        stakeholders.append("동료")
                    if not stakeholders:
                        stakeholders = ["타인", "사회"]
                    
                    # Circuit 컨텍스트 생성
                    circuit_context = CircuitDecisionContext(
                        scenario_text=text,
                        proposed_action="상황 분석 및 윤리적 평가",
                        stakeholders=stakeholders,
                        social_context={
                            'impact_scope': 'personal' if len(stakeholders) < 3 else 'community',
                            'keywords': text.split()[:5],
                            'urgency': 0.5
                        },
                        temporal_urgency=0.5,
                        self_emotion=emotion_data
                    )
                    
                    # Circuit 실행 (짧은 타임아웃)
                    import asyncio
                    circuit_result = await asyncio.wait_for(
                        self.emotion_hierarchy_processor.process_ethical_decision(circuit_context),
                        timeout=3.0  # 3초 타임아웃
                    )
                    
                    if circuit_result:
                        # Circuit 결과를 results에 추가
                        results['circuit_analysis'] = {
                            'integrated_emotion': {
                                'primary': circuit_result.integrated_emotion.primary_emotion.value if hasattr(circuit_result, 'integrated_emotion') else None,
                                'intensity': circuit_result.integrated_emotion.intensity.value if hasattr(circuit_result, 'integrated_emotion') else None,
                                'confidence': circuit_result.integrated_emotion.confidence if hasattr(circuit_result, 'integrated_emotion') else 0
                            } if hasattr(circuit_result, 'integrated_emotion') else {},
                            'ethical_values': circuit_result.ethical_values if hasattr(circuit_result, 'ethical_values') else {},
                            'predicted_regret': circuit_result.predicted_regret if hasattr(circuit_result, 'predicted_regret') else {},
                            'regret_metrics': circuit_result.regret_metrics if hasattr(circuit_result, 'regret_metrics') else {},
                            'reasoning_trace': circuit_result.reasoning_trace if hasattr(circuit_result, 'reasoning_trace') else []
                        }
                        logger.info(f"   ✅ Circuit 처리 완료 (신뢰도: {circuit_result.confidence if hasattr(circuit_result, 'confidence') else 0:.2f})")
                        
                        # Circuit 결과 로깅
                        if hasattr(circuit_result, 'integrated_emotion'):
                            logger.info(f"      - 통합 감정: {circuit_result.integrated_emotion.primary_emotion.name}")
                        if hasattr(circuit_result, 'predicted_regret'):
                            logger.info(f"      - 예측 후회: {circuit_result.predicted_regret}")
                        if hasattr(circuit_result, 'ethical_values'):
                            logger.info(f"      - 윤리적 가치: {circuit_result.ethical_values}")
                    else:
                        logger.warning("   ⚠️ Circuit 결과 없음")
                        
                except asyncio.TimeoutError:
                    logger.warning("   ⏱️ Circuit 처리 타임아웃 (3초 초과)")
                except ImportError as e:
                    logger.warning(f"   ⚠️ Circuit 모듈 임포트 실패: {e}")
                except Exception as e:
                    logger.error(f"   ❌ Circuit 처리 실패: {e}")
                    # Circuit 실패는 전체 실패가 아니므로 계속 진행
            else:
                # 감정 분석 결과가 없어서 Circuit 실행 불가
                if self.emotion_hierarchy_processor:
                    if not results.get('advanced_analysis'):
                        logger.error("   ❌ Circuit 실행 불가: Advanced Analysis 결과가 없습니다")
                        logger.error("      → Advanced Emotion Analyzer가 실패했을 가능성이 있습니다")
                    elif not results.get('advanced_analysis', {}).get('emotion'):
                        logger.error("   ❌ Circuit 실행 불가: 감정 분석 결과가 없습니다")
                        logger.error("      → emotion 필드가 비어있거나 실패했습니다")
                        logger.error(f"      → Advanced Analysis 키: {list(results.get('advanced_analysis', {}).keys())}")
                    # Circuit은 필수 구성요소이므로 실행 불가 시 시스템 중단
                    raise RuntimeError("Circuit 실행 불가: 감정 분석 데이터가 없음. Advanced Emotion Analyzer 실패로 인한 연쇄 실패")
                else:
                    logger.info("   ℹ️ Circuit 프로세서가 초기화되지 않음")
            
            # 4. Claude API 보강
            if self.llm_engine:
                logger.info("\n🌐 Claude API 보강...")
                from llm_module.advanced_llm_engine import LLMRequest, TaskComplexity
                
                # Red Heart 분석 결과를 통합한 프롬프트 생성
                analysis_summary = []
                
                # UnifiedModel 결과 포함
                if results.get('unified_outputs'):
                    unified_outputs = results['unified_outputs']
                    if 'emotion' in unified_outputs:
                        analysis_summary.append(f"감정 분석: {unified_outputs['emotion']}")
                    if 'bentham' in unified_outputs:
                        analysis_summary.append(f"공리주의 점수: {unified_outputs['bentham']}")
                    if 'regret' in unified_outputs:
                        analysis_summary.append(f"후회 예측: {unified_outputs['regret']}")
                    if 'surd' in unified_outputs:
                        analysis_summary.append(f"SURD 메트릭: {unified_outputs['surd']}")
                
                # Advanced Wrapper 결과 포함 (모든 분석이 성공해야 함)
                if not results.get('advanced_analysis'):
                    raise RuntimeError("Advanced Analysis 결과가 없음: 시스템 무결성 오류")
                
                adv = results['advanced_analysis']
                required_analyses = ['emotion', 'bentham', 'regret', 'surd']
                missing_analyses = [a for a in required_analyses if a not in adv or adv[a] is None]
                
                if missing_analyses:
                    raise RuntimeError(f"필수 분석 누락: {', '.join(missing_analyses)}. 시스템 무결성 오류")
                
                analysis_summary.append(f"고급 감정 분석: {adv['emotion']}")
                analysis_summary.append(f"고급 벤담 분석: {adv['bentham']}")
                analysis_summary.append(f"고급 후회 분석: {adv['regret']}")
                analysis_summary.append(f"고급 SURD 분석: {adv['surd']}")
                
                # Circuit 분석 결과 포함
                if results.get('circuit_analysis'):
                    circuit = results['circuit_analysis']
                    if 'integrated_emotion' in circuit and circuit['integrated_emotion']:
                        analysis_summary.append(f"Circuit 통합 감정: {circuit['integrated_emotion']}")
                    if 'ethical_values' in circuit and circuit['ethical_values']:
                        analysis_summary.append(f"Circuit 윤리 가치: {circuit['ethical_values']}")
                    if 'predicted_regret' in circuit and circuit['predicted_regret']:
                        analysis_summary.append(f"Circuit 예측 후회: {circuit['predicted_regret']}")
                
                # 반사실적 시나리오 분석 결과 포함
                if results.get('counterfactual_analysis'):
                    cf_analysis = results['counterfactual_analysis']
                    if 'initial_scenarios' in cf_analysis:
                        scenarios_text = []
                        for scenario in cf_analysis['initial_scenarios']:
                            scenarios_text.append(f"- {scenario.get('type', '')}: {scenario.get('action', '')[:50]}...")
                        analysis_summary.append(f"반사실적 시나리오: {chr(10).join(scenarios_text)}")
                
                # 3뷰 시스템 분석 결과 포함
                if results.get('three_view_analysis'):
                    analysis_summary.append(f"3뷰 시스템 분석: 낙관적/중립적/비관적 관점 분석 완료")
                
                # 통합된 프롬프트 생성
                integrated_prompt = f"""다음 텍스트와 Red Heart AI의 심층 분석 결과를 바탕으로 종합적인 윤리적 평가를 제공하세요.

텍스트: {text}

Red Heart AI 다층 분석 결과:
{chr(10).join(analysis_summary) if analysis_summary else '분석 결과 없음'}

반사실적 시나리오 기반 분석:
{len(counterfactual_scenarios)}개의 대안적 행동 시나리오가 생성되고 평가되었습니다.

위 모든 분석을 통합하여 다음을 포함한 최종 윤리적 평가를 제공하세요:
1. 주요 윤리적 쟁점과 딜레마
2. 각 시나리오별 잠재적 결과와 영향
3. 최적의 행동 방향 권장 (3뷰 분석 기반)
4. 고려해야 할 가치 충돌과 trade-off
5. 반사실적 분석에서 도출된 핵심 통찰"""
                
                request = LLMRequest(
                    prompt=integrated_prompt,
                    task_type="ethical_analysis",
                    complexity=TaskComplexity.COMPLEX,
                    max_tokens=500,
                    context={'red_heart_analysis': results}  # 컨텍스트에 Red Heart 분석 결과 포함
                )
                
                llm_response = await self.llm_engine.generate_async(request)
                
                if llm_response and llm_response.success:
                    results['claude_analysis'] = {
                        'text': llm_response.generated_text,
                        'confidence': llm_response.confidence,
                        'processing_time': llm_response.processing_time
                    }
                    logger.info("   ✅ Claude API 분석 완료")
                else:
                    logger.warning("   ⚠️ Claude API 응답 실패")
            
            # 5. 결과 저장 (마이크로초 및 PID 포함하여 유일성 보장)
            import os
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 밀리초까지 포함
            pid = os.getpid()
            output_file = f"claude_inference_result_{timestamp}_pid{pid}.json"
            self.output_file = output_file  # 나중에 참조할 수 있도록 저장
            
            # datetime 객체를 문자열로 변환하는 커스텀 JSON encoder
            class DateTimeEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, datetime):
                        return obj.isoformat()
                    return super().default(obj)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, cls=DateTimeEncoder)
            
            logger.info(f"\n💾 결과 저장: {output_file}")
            
            # 최신 결과를 가리키는 심볼릭 링크 생성 (선택적)
            try:
                latest_link = "claude_inference_result_latest.json"
                if os.path.exists(latest_link) or os.path.islink(latest_link):
                    os.remove(latest_link)
                os.symlink(output_file, latest_link)
                logger.info(f"   🔗 최신 결과 링크: {latest_link} -> {output_file}")
            except (OSError, NotImplementedError) as e:
                # WSL이나 권한 문제로 심볼릭 링크 생성 실패 시 무시
                logger.debug(f"   ⚠️ 심볼릭 링크 생성 실패 (무시됨): {e}")
            
            # 6. 요약 출력
            logger.info("\n" + "=" * 70)
            logger.info("📊 추론 완료 요약")
            logger.info("=" * 70)
            
            if 'emotion' in results['advanced_analysis']:
                emotions = results['advanced_analysis']['emotion'].get('emotions', [])
                if emotions and len(emotions) > 0 and len(emotions[0]) > 0:
                    # 감정 레이블 매핑 (7차원)
                    emotion_labels = ['기쁨', '슬픔', '분노', '두려움', '놀람', '혐오', '중립']
                    emotion_values = emotions[0] if isinstance(emotions[0], list) else emotions
                    
                    # 가장 높은 값의 감정 찾기
                    max_idx = np.argmax(emotion_values)
                    max_score = emotion_values[max_idx]
                    primary_emotion = emotion_labels[max_idx]
                    
                    logger.info(f"🎭 주요 감정: {primary_emotion} (점수: {max_score:.3f})")
                else:
                    logger.info(f"🎭 주요 감정: N/A")
            
            if 'bentham' in results['advanced_analysis']:
                bentham_scores = results['advanced_analysis']['bentham'].get('bentham_scores', [])
                if bentham_scores and len(bentham_scores) > 0 and len(bentham_scores[0]) > 0:
                    # 벤담 점수들의 평균을 공리주의 점수로 사용
                    scores = bentham_scores[0] if isinstance(bentham_scores[0], list) else bentham_scores
                    utility_score = np.mean(scores)
                    logger.info(f"⚖️ 공리주의 점수: {utility_score:.3f}")
                else:
                    logger.info(f"⚖️ 공리주의 점수: N/A")
            
            # Circuit 결과 출력 추가
            if 'circuit_analysis' in results:
                circuit = results['circuit_analysis']
                if circuit.get('integrated_emotion'):
                    ie = circuit['integrated_emotion']
                    logger.info(f"🎭 Circuit 통합 감정: 주요={ie.get('primary', 'N/A')}, 강도={ie.get('intensity', 'N/A')}, 신뢰도={ie.get('confidence', 0):.3f}")
                if circuit.get('predicted_regret'):
                    logger.info(f"😔 Circuit 예측 후회: {circuit['predicted_regret']}")
                if circuit.get('ethical_values'):
                    logger.info(f"⚖️ Circuit 윤리 가치: {circuit['ethical_values']}")
            
            # Regret 분석 결과 출력 추가
            if 'regret' in results['advanced_analysis']:
                regret_scores = results['advanced_analysis']['regret']
                if regret_scores:
                    logger.info(f"😔 고급 후회 분석: {regret_scores}")
            
            if 'claude_analysis' in results:
                claude_text = results['claude_analysis']['text'][:200]
                logger.info(f"🤖 Claude 분석:\n{claude_text}...")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ 추론 실패: {e}", exc_info=True)
            return {'error': str(e)}
    
    async def cleanup(self):
        """리소스 정리"""
        logger.info("\n🧹 리소스 정리 중...")
        
        # 모델 정리
        if self.unified_model:
            del self.unified_model
        
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("✅ 정리 완료")


async def main(args):
    """메인 실행 함수"""
    system = ClaudeInferenceSystem(
        epoch=args.epoch,
        debug=args.debug
    )
    
    try:
        # 초기화
        await system.initialize()
        
        # 추론 실행
        results = await system.inference(args.text)
        
        # 정리
        await system.cleanup()
        
        logger.info("\n🎉 Claude API 추론 완료!")
        
    except Exception as e:
        logger.error(f"❌ 실행 실패: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Claude API 독립 추론 시스템')
    parser.add_argument('--text', type=str, required=True, help='추론할 텍스트')
    parser.add_argument('--epoch', type=int, default=50, help='체크포인트 에폭')
    parser.add_argument('--debug', action='store_true', help='디버그 모드')
    
    args = parser.parse_args()
    
    # 비동기 실행
    asyncio.run(main(args))