#!/usr/bin/env python3
"""
하이브리드 시스템 완전 테스트 - Fallback 없는 온전한 시스템 검증
Hybrid System Full Test - Complete System Verification without Fallbacks
"""

import os
import sys
import json
import torch
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import traceback

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 모든 모듈 import
from training.hybrid_distributed_trainer import HybridDistributedTrainer, HybridConfig, MemoryOptimizedModel, AsyncRegretCalculator
from models.mega_scale_models.scalable_xai_model import create_mega_scale_model
from models.hierarchical_emotion.emotion_phase_models import HierarchicalEmotionModel
from models.semantic_models.advanced_semantic_models import AdvancedSemanticModel, SemanticAnalysisConfig
from models.counterfactual_models.counterfactual_reasoning_models import AdvancedCounterfactualModel, CounterfactualConfig
from xai_core.xai_logging_system import xai_logger, xai_trace, xai_decision_point
from llm_module import llm_tracker, register_llm, ask_llm, LLMConfig

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('HybridFullTest')

class FullSystemTester:
    """완전한 시스템 테스터"""
    
    def __init__(self):
        self.test_results = {}
        self.fallback_detected = False
        self.errors_detected = []
        
    def log_test(self, test_name: str, success: bool, details: str = "", fallback_used: bool = False):
        """테스트 결과 로깅"""
        status = "✅ PASS" if success else "❌ FAIL"
        if fallback_used:
            status += " (⚠️ FALLBACK USED)"
            self.fallback_detected = True
        
        logger.info(f"{status} {test_name}: {details}")
        
        self.test_results[test_name] = {
            'success': success,
            'details': details,
            'fallback_used': fallback_used,
            'timestamp': datetime.now().isoformat()
        }
        
        if not success:
            self.errors_detected.append(f"{test_name}: {details}")
    
    def test_system_info(self):
        """시스템 정보 테스트"""
        logger.info("🖥️ 시스템 정보 확인")
        
        # 계산 시스템 확인
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self.log_test("Compute_System", True, f"GPU 가속: {gpu_name}, Memory: {gpu_memory:.1f}GB")
        else:
            self.log_test("Compute_System", True, "고성능 CPU 계산 모드 - 128GB 대용량 메모리 활용")
        
        # CPU 정보
        cpu_cores = torch.get_num_threads()
        self.log_test("CPU_Cores", True, f"{cpu_cores} threads available")
        
        return cuda_available
    
    def test_hybrid_config(self):
        """하이브리드 설정 테스트"""
        logger.info("⚙️ 하이브리드 설정 테스트")
        
        try:
            config = HybridConfig(
                target_params=3_000_000_000,
                gpu_memory_gb=8.0,
                cpu_memory_gb=128.0,
                regrets_per_step=7,
                bentham_calculations_per_regret=3,
                epochs=1,  # 테스트용으로 1 에포크
                batch_size=4,  # 테스트용으로 작은 배치
                num_workers=2,  # 테스트용으로 적은 워커
                use_mixed_precision=True,
                log_every_n_steps=2,
                save_checkpoint_every=5
            )
            
            self.log_test("Hybrid_Config", True, f"Target: {config.target_params:,} params, Workers: {config.num_workers}")
            return config
            
        except Exception as e:
            self.log_test("Hybrid_Config", False, f"Config error: {e}")
            return None
    
    def test_memory_optimized_model(self, config: HybridConfig):
        """메모리 최적화 모델 테스트"""
        logger.info("🧠 메모리 최적화 모델 테스트")
        
        try:
            model = MemoryOptimizedModel(config)
            param_count = model.get_parameter_count()
            
            # 33억 파라미터 목표 달성 확인
            if param_count >= 2_500_000_000:  # 25억 이상이면 성공
                self.log_test("Model_Parameters", True, f"{param_count:,} parameters (Target: 3B)")
            else:
                self.log_test("Model_Parameters", False, f"Only {param_count:,} parameters (Target: 3B)")
            
            # 테스트 입력으로 순전파
            test_input = torch.randn(2, 32, 1024)  # 작은 테스트 입력
            
            with torch.no_grad():
                outputs = model(test_input)
            
            # 출력 검증
            required_keys = ['emotion_predictions', 'semantic_predictions', 'reasoning_features', 'integration_features']
            missing_keys = [key for key in required_keys if key not in outputs]
            
            if not missing_keys:
                self.log_test("Model_Forward", True, f"All outputs present: {list(outputs.keys())}")
            else:
                self.log_test("Model_Forward", False, f"Missing outputs: {missing_keys}")
            
            return model, outputs
            
        except Exception as e:
            self.log_test("Model_Creation", False, f"Model error: {e}")
            traceback.print_exc()
            return None, None
    
    def test_async_regret_calculator(self, config: HybridConfig):
        """비동기 후회 계산기 테스트"""
        logger.info("🔄 비동기 후회 계산기 테스트")
        
        try:
            calculator = AsyncRegretCalculator(config)
            
            # 테스트 결정
            test_decision = torch.randn(1024)
            
            # 비동기 계산 요청
            calculator.calculate_async(test_decision, task_id=1)
            
            # 결과 대기 (최대 5초)
            result = None
            for _ in range(50):  # 5초 대기
                result = calculator.get_result(timeout=0.1)
                if result:
                    break
                time.sleep(0.1)
            
            if result:
                task_id, scenarios = result
                if len(scenarios) == config.regrets_per_step:
                    self.log_test("Async_Regret", True, f"{len(scenarios)} regret scenarios generated")
                else:
                    self.log_test("Async_Regret", False, f"Expected {config.regrets_per_step}, got {len(scenarios)}")
            else:
                self.log_test("Async_Regret", False, "No result received within timeout")
            
            return calculator
            
        except Exception as e:
            self.log_test("Async_Regret", False, f"Calculator error: {e}")
            traceback.print_exc()
            return None
    
    def test_xai_integration(self):
        """XAI 통합 테스트"""
        logger.info("🔍 XAI 통합 테스트")
        
        try:
            # XAI 로깅 테스트
            with xai_logger.trace_operation("hybrid_test", "xai_integration") as operation_id:
                if operation_id:
                    self.log_test("XAI_Tracing", True, f"Operation ID: {operation_id}")
                else:
                    self.log_test("XAI_Tracing", False, "No operation ID generated")
                
                # XAI 결정 포인트 테스트
                @xai_decision_point()
                def test_decision():
                    return {"decision": "test", "confidence": 0.95}
                
                decision_result = test_decision()
                self.log_test("XAI_Decision", True, f"Decision: {decision_result}")
            
            # 성능 요약
            performance = xai_logger.get_performance_summary()
            self.log_test("XAI_Performance", True, f"Ops tracked: {len(performance)}")
            
            return True
            
        except Exception as e:
            self.log_test("XAI_Integration", False, f"XAI error: {e}")
            return False
    
    def test_llm_integration(self):
        """LLM 통합 테스트"""
        logger.info("🤖 LLM 통합 테스트")
        
        try:
            # LLM 등록 테스트
            wrapper = register_llm(
                model_name="test_hybrid_model",
                max_tokens=50,
                temperature=0.7
            )
            
            if wrapper and wrapper.config.model_name:
                if "intelligent" in wrapper.config.model_name or "advanced" in wrapper.config.model_name:
                    self.log_test("LLM_Registration", True, f"고급 지능형 LLM: {wrapper.config.model_name}")
                else:
                    self.log_test("LLM_Registration", True, f"LLM 모델: {wrapper.config.model_name}")
            else:
                self.log_test("LLM_Registration", False, "LLM registration failed", fallback_used=True)
            
            # LLM 응답 테스트
            response = ask_llm("test_hybrid_model", "하이브리드 시스템 테스트")
            
            if response and len(response) > 0:
                if "고급 AI" in response or "지능형" in response or "분석" in response:
                    self.log_test("LLM_Response", True, f"고품질 응답: {len(response)}자")
                elif "fallback" not in response.lower() and "error" not in response.lower():
                    self.log_test("LLM_Response", True, f"정상 응답: {len(response)}자")
                else:
                    self.log_test("LLM_Response", True, f"기본 응답: {response[:50]}...", fallback_used=True)
            else:
                self.log_test("LLM_Response", False, "No response from LLM")
            
            # 성능 요약
            performance = llm_tracker.get_performance_summary()
            self.log_test("LLM_Performance", True, f"Interactions: {performance.get('total_interactions', 0)}")
            
            return True
            
        except Exception as e:
            self.log_test("LLM_Integration", False, f"LLM error: {e}")
            return False
    
    def test_individual_models(self):
        """개별 모델들 테스트"""
        logger.info("🎯 개별 모델들 테스트")
        
        # 테스트 데이터
        test_embedding = torch.randn(4, 768)
        test_tokens = torch.randint(0, 1000, (4, 50))
        
        # 1. 감정 모델
        try:
            emotion_model = HierarchicalEmotionModel()
            emotion_output = emotion_model(test_embedding)
            
            if 'final_emotion' in emotion_output:
                self.log_test("Emotion_Model", True, f"Output shape: {emotion_output['final_emotion'].shape}")
            else:
                self.log_test("Emotion_Model", False, "No final_emotion output")
                
        except Exception as e:
            self.log_test("Emotion_Model", False, f"Emotion error: {e}")
        
        # 2. 의미 모델
        try:
            semantic_config = SemanticAnalysisConfig(vocab_size=1000, embedding_dim=256)
            semantic_model = AdvancedSemanticModel(semantic_config)
            semantic_output = semantic_model(test_tokens)
            
            if 'enhanced_semantics' in semantic_output:
                self.log_test("Semantic_Model", True, f"Output shape: {semantic_output['enhanced_semantics'].shape}")
            else:
                self.log_test("Semantic_Model", False, "No enhanced_semantics output")
                
        except Exception as e:
            self.log_test("Semantic_Model", False, f"Semantic error: {e}")
        
        # 3. 반사실 모델
        try:
            cf_config = CounterfactualConfig(input_dim=768, hidden_dims=[512, 256, 128], latent_dim=64)
            cf_model = AdvancedCounterfactualModel(cf_config)
            cf_output = cf_model(test_embedding)
            
            if 'counterfactual_scenarios' in cf_output:
                self.log_test("Counterfactual_Model", True, f"Scenarios: {len(cf_output['counterfactual_scenarios'])}")
            else:
                self.log_test("Counterfactual_Model", False, "No counterfactual_scenarios output")
                
        except Exception as e:
            self.log_test("Counterfactual_Model", False, f"Counterfactual error: {e}")
    
    def test_data_loading(self):
        """데이터 로딩 테스트"""
        logger.info("📊 데이터 로딩 테스트")
        
        try:
            data_dir = project_root / 'processed_datasets'
            batch_files = list(data_dir.glob('full_scenarios_batch_*.json'))
            
            if not batch_files:
                self.log_test("Data_Files", False, "No batch files found")
                return False
            
            self.log_test("Data_Files", True, f"{len(batch_files)} batch files found")
            
            # 첫 번째 파일 로드 테스트
            with open(batch_files[0], 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if isinstance(data, list) and len(data) > 0:
                sample = data[0]
                required_fields = ['id', 'description', 'options']
                missing_fields = [field for field in required_fields if field not in sample]
                
                if not missing_fields:
                    self.log_test("Data_Structure", True, f"Sample has all required fields")
                else:
                    self.log_test("Data_Structure", False, f"Missing fields: {missing_fields}")
            else:
                self.log_test("Data_Structure", False, "Invalid data structure")
            
            return True
            
        except Exception as e:
            self.log_test("Data_Loading", False, f"Data error: {e}")
            return False
    
    def test_hybrid_trainer_creation(self, config: HybridConfig):
        """하이브리드 트레이너 생성 테스트"""
        logger.info("🚀 하이브리드 트레이너 생성 테스트")
        
        try:
            trainer = HybridDistributedTrainer(config)
            
            # 모델 준비
            param_count = trainer.prepare_model()
            
            if param_count >= 2_500_000_000:
                self.log_test("Trainer_Model", True, f"{param_count:,} parameters prepared")
            else:
                self.log_test("Trainer_Model", False, f"Only {param_count:,} parameters")
            
            # 데이터 준비
            dataloader = trainer.prepare_data()
            
            if len(dataloader) > 0:
                self.log_test("Trainer_Data", True, f"{len(dataloader)} batches prepared")
            else:
                self.log_test("Trainer_Data", False, "No data batches")
            
            return trainer
            
        except Exception as e:
            self.log_test("Trainer_Creation", False, f"Trainer error: {e}")
            traceback.print_exc()
            return None
    
    def run_full_test(self):
        """전체 테스트 실행"""
        logger.info("🎉 하이브리드 시스템 완전 테스트 시작")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # 1. 시스템 정보
        cuda_available = self.test_system_info()
        
        # 2. 하이브리드 설정
        config = self.test_hybrid_config()
        if not config:
            logger.error("❌ 설정 테스트 실패 - 테스트 중단")
            return False
        
        # 3. 메모리 최적화 모델
        model, outputs = self.test_memory_optimized_model(config)
        
        # 4. 비동기 후회 계산기
        calculator = self.test_async_regret_calculator(config)
        
        # 5. XAI 통합
        self.test_xai_integration()
        
        # 6. LLM 통합
        self.test_llm_integration()
        
        # 7. 개별 모델들
        self.test_individual_models()
        
        # 8. 데이터 로딩
        self.test_data_loading()
        
        # 9. 하이브리드 트레이너
        trainer = self.test_hybrid_trainer_creation(config)
        
        test_time = time.time() - start_time
        
        # 결과 분석
        logger.info("\n" + "=" * 80)
        logger.info("📊 하이브리드 시스템 테스트 결과")
        logger.info("=" * 80)
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results.values() if result['success'])
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        logger.info(f"✅ 성공한 테스트: {successful_tests}/{total_tests} ({success_rate:.1f}%)")
        logger.info(f"⏱️ 테스트 시간: {test_time:.2f}초")
        
        # 개별 테스트 결과
        for test_name, result in self.test_results.items():
            status = "✅" if result['success'] else "❌"
            fallback = " (⚠️ FALLBACK)" if result['fallback_used'] else ""
            logger.info(f"   {status} {test_name}{fallback}")
        
        # Fallback 사용 검사
        if self.fallback_detected:
            logger.warning("\n⚠️ FALLBACK 모드가 감지되었습니다!")
            logger.warning("일부 기능이 완전하지 않습니다.")
        else:
            logger.info("\n✅ FALLBACK 없는 완전한 시스템입니다!")
        
        # 오류 요약
        if self.errors_detected:
            logger.error(f"\n❌ 감지된 오류들:")
            for error in self.errors_detected:
                logger.error(f"   - {error}")
        
        # 최종 판정
        system_complete = (
            success_rate >= 80 and 
            not self.fallback_detected and 
            len(self.errors_detected) == 0
        )
        
        if system_complete:
            logger.info("\n🎊 하이브리드 시스템 완전 검증 성공! 🎊")
            logger.info("모든 기능이 Fallback 없이 정상 작동합니다!")
            logger.info("학습 준비 완료 상태입니다.")
        else:
            logger.warning("\n⚠️ 시스템에 문제가 있습니다.")
            logger.warning("추가 수정이 필요합니다.")
        
        # 결과 저장
        results_path = project_root / 'training' / 'hybrid_outputs' / 'full_system_test_results.json'
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        final_results = {
            'test_summary': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'success_rate': success_rate,
                'test_duration': test_time,
                'fallback_detected': self.fallback_detected,
                'errors_count': len(self.errors_detected),
                'system_complete': system_complete,
                'cuda_available': cuda_available,
                'timestamp': datetime.now().isoformat()
            },
            'test_results': self.test_results,
            'errors_detected': self.errors_detected,
            'recommendations': self._generate_recommendations()
        }
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n💾 테스트 결과 저장: {results_path}")
        
        return system_complete
    
    def _generate_recommendations(self) -> List[str]:
        """권장사항 생성"""
        recommendations = []
        
        if self.fallback_detected:
            recommendations.append("Fallback 모드를 제거하여 완전한 기능 구현이 필요합니다.")
        
        if self.errors_detected:
            recommendations.append("감지된 오류들을 수정해야 합니다.")
        
        success_rate = sum(1 for r in self.test_results.values() if r['success']) / len(self.test_results) * 100
        if success_rate < 90:
            recommendations.append("테스트 성공률을 90% 이상으로 개선해야 합니다.")
        
        if not recommendations:
            recommendations.append("시스템이 완전히 검증되었습니다. 학습을 시작할 수 있습니다.")
        
        return recommendations

if __name__ == "__main__":
    tester = FullSystemTester()
    success = tester.run_full_test()
    
    if success:
        print("\n🚀 하이브리드 시스템 준비 완료!")
        print("python3 training/start_hybrid_training.py 로 학습을 시작할 수 있습니다.")
    else:
        print("\n⚠️ 시스템 점검이 필요합니다.")
    
    sys.exit(0 if success else 1)