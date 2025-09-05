#!/usr/bin/env python3
"""
Claude API 전용 워크플로우
- DSM 사용하지 않고 직접 GPU 메모리 관리
- 분할 초기화로 메모리 최적화
- Claude API로 직접 처리 (번역기 불필요)
"""

import os
import sys
import asyncio
import argparse
import logging
import torch
import torch.nn as nn
import gc
import time
import json
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 환경 설정
os.environ['TORCH_HOME'] = str(PROJECT_ROOT / '.cache' / 'torch')
os.environ['HF_HOME'] = str(PROJECT_ROOT / '.cache' / 'huggingface')
os.environ['TRANSFORMERS_CACHE'] = str(PROJECT_ROOT / '.cache' / 'transformers')
os.environ['FORCE_CPU_INIT'] = '1'  # SentenceTransformer는 항상 CPU

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class GPUMemoryManager:
    """하드코딩된 GPU 메모리 관리자"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cpu_device = torch.device('cpu')
        self.models_on_gpu = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def get_gpu_memory_info(self) -> Tuple[float, float, float]:
        """GPU 메모리 정보 반환 (used_mb, free_mb, total_mb)"""
        if not torch.cuda.is_available():
            return 0.0, 0.0, 0.0
            
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        free = total - reserved
        
        return allocated, free, total
        
    def clear_cache(self):
        """GPU 캐시 정리"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
    def move_to_gpu(self, model: nn.Module, name: str, max_memory_mb: float = 4000) -> nn.Module:
        """모델을 GPU로 이동 (메모리 제한 확인)"""
        used, free, total = self.get_gpu_memory_info()
        
        if free < max_memory_mb:
            self.logger.warning(f"⚠️ GPU 메모리 부족: {free:.1f}MB < {max_memory_mb:.1f}MB")
            # 가장 오래된 모델 언로드
            if self.models_on_gpu:
                oldest_name = list(self.models_on_gpu.keys())[0]
                self.move_to_cpu(self.models_on_gpu[oldest_name], oldest_name)
                self.clear_cache()
                
        model = model.to(self.device)
        self.models_on_gpu[name] = model
        
        used, free, total = self.get_gpu_memory_info()
        self.logger.info(f"✅ {name} GPU 로드 완료: {used:.1f}/{total:.1f}MB 사용중")
        
        return model
        
    def move_to_cpu(self, model: nn.Module, name: str) -> nn.Module:
        """모델을 CPU로 이동"""
        model = model.to(self.cpu_device)
        
        if name in self.models_on_gpu:
            del self.models_on_gpu[name]
            
        self.clear_cache()
        
        used, free, total = self.get_gpu_memory_info()
        self.logger.info(f"✅ {name} CPU 언로드 완료: {used:.1f}/{total:.1f}MB 사용중")
        
        return model


class ClaudeAPIWorkflow:
    """Claude API 전용 워크플로우"""
    
    def __init__(self, args):
        self.args = args
        self.logger = logging.getLogger(self.__class__.__name__)
        self.gpu_manager = GPUMemoryManager()
        
        # 모델 컴포넌트 (지연 로딩)
        self.sentence_transformer = None
        self.unified_model = None
        self.emotion_analyzer = None
        self.bentham_calculator = None
        self.regret_system = None
        self.counterfactual = None
        self.circuit = None
        self.claude_client = None
        
        # 체크포인트 경로
        self.checkpoint_path = PROJECT_ROOT / 'training' / 'checkpoints_final'
        
    async def initialize(self):
        """분할 초기화 - 단계별로 메모리 관리"""
        self.logger.info("=" * 60)
        self.logger.info("🚀 Claude API 워크플로우 초기화 시작")
        self.logger.info("=" * 60)
        
        try:
            # 1단계: SentenceTransformer (CPU 전용)
            await self._init_sentence_transformer()
            
            # 2단계: Claude API 클라이언트
            await self._init_claude_api()
            
            # 3단계: UnifiedModel (필요시에만 GPU 로드)
            await self._init_unified_model()
            
            # 4단계: 분석 모듈들 (필요시에만 로드)
            # 나중에 필요할 때 개별적으로 로드
            
            self.logger.info("✅ 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 초기화 실패: {e}")
            raise
            
    async def _init_sentence_transformer(self):
        """SentenceTransformer 초기화 (CPU 전용)"""
        self.logger.info("\n📝 SentenceTransformer 초기화 (CPU)...")
        
        from sentence_transformer_singleton import SentenceTransformerManager
        
        # CPU 강제 설정
        self.sentence_transformer = SentenceTransformerManager(device='cpu')
        
        self.logger.info("✅ SentenceTransformer CPU 초기화 완료")
        
    async def _init_claude_api(self):
        """Claude API 클라이언트 초기화"""
        self.logger.info("\n🤖 Claude API 초기화...")
        
        from api_key_manager.llm_clients.claude_client import ClaudeAPIClient
        
        self.claude_client = ClaudeAPIClient()
        await self.claude_client.initialize()
        
        self.logger.info("✅ Claude API 초기화 완료")
        
    async def _init_unified_model(self):
        """UnifiedModel 초기화 (CPU로 먼저 로드)"""
        self.logger.info("\n🧠 UnifiedModel 초기화...")
        
        from unified_model import UnifiedModel
        from config import Config
        
        # 설정 로드
        config = Config()
        
        # CPU에서 먼저 로드
        self.unified_model = UnifiedModel(config)
        
        # 체크포인트 로드
        checkpoint_file = self.checkpoint_path / 'accumulated_checkpoint.pt'
        if checkpoint_file.exists():
            self.logger.info(f"📂 체크포인트 로드: {checkpoint_file}")
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
            
            if 'unified_model_state' in checkpoint:
                self.unified_model.load_state_dict(checkpoint['unified_model_state'])
                self.logger.info("✅ UnifiedModel 체크포인트 로드 완료")
            else:
                self.logger.warning("⚠️ 체크포인트에 UnifiedModel 상태 없음")
                
        self.unified_model.eval()
        self.logger.info("✅ UnifiedModel 초기화 완료 (CPU)")
        
    async def _load_analyzer_on_demand(self, analyzer_name: str):
        """필요시 분석기 로드"""
        self.logger.info(f"\n🔄 {analyzer_name} 분석기 로드...")
        
        if analyzer_name == 'emotion' and not self.emotion_analyzer:
            from advanced_emotion_analyzer import AdvancedEmotionAnalyzer
            self.emotion_analyzer = AdvancedEmotionAnalyzer()
            
        elif analyzer_name == 'bentham' and not self.bentham_calculator:
            from advanced_bentham_calculator import AdvancedBenthamCalculator
            self.bentham_calculator = AdvancedBenthamCalculator()
            
        elif analyzer_name == 'regret' and not self.regret_system:
            from advanced_regret_learning_system import AdvancedRegretLearningSystem
            self.regret_system = AdvancedRegretLearningSystem()
            
        elif analyzer_name == 'counterfactual' and not self.counterfactual:
            from advanced_counterfactual_reasoning import AdvancedCounterfactualReasoning
            self.counterfactual = AdvancedCounterfactualReasoning()
            
        self.logger.info(f"✅ {analyzer_name} 로드 완료")
        
    async def inference(self, text: str) -> Dict[str, Any]:
        """추론 실행 - 단계별 GPU 관리"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info(f"🎯 추론 시작: {text[:50]}...")
        self.logger.info("=" * 60)
        
        results = {}
        
        try:
            # Phase 1: 텍스트 임베딩 (CPU)
            self.logger.info("\n📝 Phase 1: 텍스트 임베딩...")
            embeddings = await self._get_embeddings(text)
            results['embeddings'] = embeddings
            
            # Phase 2: Claude API 초기 분석
            self.logger.info("\n🤖 Phase 2: Claude API 분석...")
            claude_analysis = await self._claude_analysis(text)
            results['claude_initial'] = claude_analysis
            
            # Phase 3: UnifiedModel 처리 (GPU 로드/언로드)
            self.logger.info("\n🧠 Phase 3: UnifiedModel 처리...")
            unified_results = await self._unified_model_inference(embeddings)
            results['unified'] = unified_results
            
            # Phase 4: 심층 분석기 (선택적, GPU 관리)
            if self.args.deep_analysis:
                self.logger.info("\n🔍 Phase 4: 심층 분석...")
                deep_results = await self._deep_analysis(embeddings, unified_results)
                results['deep_analysis'] = deep_results
                
            # Phase 5: Circuit 통합
            self.logger.info("\n⚡ Phase 5: Circuit 통합...")
            circuit_results = await self._circuit_integration(results)
            results['circuit'] = circuit_results
            
            # Phase 6: Claude API 최종 보정
            self.logger.info("\n🎯 Phase 6: 최종 보정...")
            final_results = await self._claude_final_correction(results)
            results['final'] = final_results
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ 추론 실패: {e}")
            raise
            
    async def _get_embeddings(self, text: str) -> torch.Tensor:
        """텍스트 임베딩 생성 (CPU)"""
        if not self.sentence_transformer:
            await self._init_sentence_transformer()
            
        embeddings = self.sentence_transformer.encode(
            [text],
            convert_to_tensor=True
        )
        
        self.logger.info(f"✅ 임베딩 생성 완료: shape={embeddings.shape}")
        return embeddings
        
    async def _claude_analysis(self, text: str) -> Dict[str, Any]:
        """Claude API로 초기 분석"""
        prompt = f"""
        다음 텍스트에 대한 윤리적, 감정적 분석을 수행하세요:
        
        텍스트: {text}
        
        분석 항목:
        1. 감정 상태 및 강도
        2. 윤리적 딜레마 포인트
        3. 가능한 시나리오들
        4. 잠재적 결과
        
        JSON 형식으로 응답하세요.
        """
        
        response = await self.claude_client.generate_async(prompt)
        
        try:
            analysis = json.loads(response.generated_text)
        except json.JSONDecodeError:
            analysis = {"raw_response": response.generated_text}
            
        self.logger.info("✅ Claude 초기 분석 완료")
        return analysis
        
    async def _unified_model_inference(self, embeddings: torch.Tensor) -> Dict[str, Any]:
        """UnifiedModel 추론 (GPU 로드/언로드)"""
        results = {}
        
        # GPU로 모델 이동
        self.unified_model = self.gpu_manager.move_to_gpu(
            self.unified_model, 
            "UnifiedModel",
            max_memory_mb=3000  # 3GB 제한
        )
        
        # 임베딩도 GPU로
        embeddings = embeddings.to(self.gpu_manager.device)
        
        try:
            with torch.no_grad():
                # Emotion 태스크
                emotion_out = self.unified_model(
                    x=embeddings,
                    task_type='emotion',
                    epoch=self.args.epoch
                )
                results['emotion'] = emotion_out
                
                # Bentham 태스크
                bentham_out = self.unified_model(
                    x=embeddings,
                    task_type='bentham',
                    epoch=self.args.epoch
                )
                results['bentham'] = bentham_out
                
                # Regret 태스크
                regret_out = self.unified_model(
                    x=embeddings,
                    task_type='regret',
                    epoch=self.args.epoch
                )
                results['regret'] = regret_out
                
        finally:
            # GPU에서 언로드
            self.unified_model = self.gpu_manager.move_to_cpu(
                self.unified_model,
                "UnifiedModel"
            )
            
        self.logger.info("✅ UnifiedModel 처리 완료")
        return results
        
    async def _deep_analysis(self, embeddings: torch.Tensor, unified_results: Dict) -> Dict[str, Any]:
        """심층 분석 (선택적)"""
        results = {}
        
        # 감정 분석
        if self.args.use_emotion:
            await self._load_analyzer_on_demand('emotion')
            emotion_result = await self.emotion_analyzer.analyze(
                embeddings, 
                unified_results.get('emotion')
            )
            results['emotion'] = emotion_result
            
        # Bentham 계산
        if self.args.use_bentham:
            await self._load_analyzer_on_demand('bentham')
            bentham_result = await self.bentham_calculator.calculate(
                unified_results.get('bentham')
            )
            results['bentham'] = bentham_result
            
        return results
        
    async def _circuit_integration(self, all_results: Dict) -> Dict[str, Any]:
        """Circuit 통합"""
        if not self.circuit:
            from emotion_ethics_regret_circuit import EmotionEthicsRegretCircuit
            self.circuit = EmotionEthicsRegretCircuit()
            
        # Circuit은 CPU에서 실행
        integrated = await self.circuit.integrate(all_results)
        
        self.logger.info("✅ Circuit 통합 완료")
        return integrated
        
    async def _claude_final_correction(self, results: Dict) -> Dict[str, Any]:
        """Claude API로 최종 보정"""
        prompt = f"""
        다음 AI 분석 결과들을 종합하여 최종 윤리적 판단을 내려주세요:
        
        분석 결과:
        {json.dumps(results, indent=2, ensure_ascii=False)}
        
        최종 판단:
        1. 종합적 윤리 평가
        2. 권장 행동
        3. 주의사항
        4. 신뢰도 점수
        
        JSON 형식으로 응답하세요.
        """
        
        response = await self.claude_client.generate_async(prompt)
        
        try:
            final = json.loads(response.generated_text)
        except json.JSONDecodeError:
            final = {"raw_response": response.generated_text}
            
        self.logger.info("✅ 최종 보정 완료")
        return final
        
    async def cleanup(self):
        """리소스 정리"""
        self.logger.info("\n🧹 리소스 정리...")
        
        # 모든 모델 CPU로 이동
        if self.unified_model:
            self.unified_model = self.unified_model.cpu()
            
        # GPU 캐시 정리
        self.gpu_manager.clear_cache()
        
        # SentenceTransformer 정리
        if self.sentence_transformer:
            pass  # SentenceTransformerManager는 자동으로 정리됨
            
        self.logger.info("✅ 정리 완료")


async def main():
    parser = argparse.ArgumentParser(description='Claude API 워크플로우')
    
    # 기본 인자
    parser.add_argument('--text', type=str, required=True, help='분석할 텍스트')
    parser.add_argument('--epoch', type=int, default=50, help='에폭 번호')
    parser.add_argument('--debug', action='store_true', help='디버그 모드')
    
    # 분석 옵션
    parser.add_argument('--deep-analysis', action='store_true', help='심층 분석 수행')
    parser.add_argument('--use-emotion', action='store_true', help='감정 분석 사용')
    parser.add_argument('--use-bentham', action='store_true', help='Bentham 계산 사용')
    parser.add_argument('--use-regret', action='store_true', help='후회 학습 사용')
    
    # 출력 옵션
    parser.add_argument('--output', type=str, help='결과 저장 파일')
    
    args = parser.parse_args()
    
    # 디버그 모드 설정
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # 워크플로우 실행
    workflow = ClaudeAPIWorkflow(args)
    
    try:
        # 초기화
        await workflow.initialize()
        
        # 추론 실행
        results = await workflow.inference(args.text)
        
        # 결과 출력
        print("\n" + "=" * 60)
        print("📊 최종 결과:")
        print("=" * 60)
        
        if 'final' in results:
            print(json.dumps(results['final'], indent=2, ensure_ascii=False))
        else:
            print(json.dumps(results, indent=2, ensure_ascii=False))
            
        # 파일 저장
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 결과 저장: {output_path}")
            
    except Exception as e:
        logger.error(f"❌ 실행 실패: {e}")
        raise
        
    finally:
        # 정리
        await workflow.cleanup()
        

if __name__ == '__main__':
    asyncio.run(main())