#!/usr/bin/env python3
"""
Red Heart AI 모듈 선택기
학습/추론 모드에 따른 동적 모듈 활성화
"""

import logging
from typing import Dict, List, Set, Any, Optional
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger('RedHeart.ModuleSelector')

class ExecutionMode(Enum):
    """실행 모드"""
    TRAINING = "training"        # 학습 모드 (최소 모듈)
    INFERENCE = "inference"      # 추론 모드 (모든 모듈)
    EVALUATION = "evaluation"    # 평가 모드 (중간)
    DEBUG = "debug"             # 디버그 모드

@dataclass
class ModuleInfo:
    """모듈 정보"""
    name: str
    category: str
    parameters: int  # 파라미터 수
    memory_mb: float  # 메모리 사용량 (MB)
    required_for_training: bool
    required_for_inference: bool
    dependencies: List[str] = field(default_factory=list)
    gpu_resident: bool = False

class ModuleSelector:
    """
    모듈 선택기
    실행 모드에 따라 필요한 모듈만 활성화
    """
    
    def __init__(self):
        """초기화"""
        self.modules = self._initialize_module_registry()
        self.active_modules: Set[str] = set()
        self.mode = ExecutionMode.TRAINING
        
        logger.info("모듈 선택기 초기화 완료")
    
    def _initialize_module_registry(self) -> Dict[str, ModuleInfo]:
        """모듈 레지스트리 초기화"""
        
        modules = {
            # ========== 핵심 학습 모듈 (320M) ==========
            'unified_backbone': ModuleInfo(
                name='unified_backbone',
                category='core',
                parameters=104_000_000,
                memory_mb=400,
                required_for_training=True,
                required_for_inference=True,
                gpu_resident=True
            ),
            
            'emotion_empathy_head': ModuleInfo(
                name='emotion_empathy_head',
                category='head',
                parameters=48_600_000,
                memory_mb=186,
                required_for_training=True,
                required_for_inference=True,
                dependencies=['unified_backbone']
            ),
            
            'bentham_fromm_head': ModuleInfo(
                name='bentham_fromm_head',
                category='head',
                parameters=41_700_000,
                memory_mb=160,
                required_for_training=True,
                required_for_inference=True,
                dependencies=['unified_backbone']
            ),
            
            'semantic_surd_head': ModuleInfo(
                name='semantic_surd_head',
                category='head',
                parameters=27_700_000,
                memory_mb=106,
                required_for_training=True,
                required_for_inference=True,
                dependencies=['unified_backbone']
            ),
            
            'regret_learning_head': ModuleInfo(
                name='regret_learning_head',
                category='head',
                parameters=41_700_000,
                memory_mb=160,
                required_for_training=True,
                required_for_inference=True,
                dependencies=['unified_backbone']
            ),
            
            'meta_integration_head': ModuleInfo(
                name='meta_integration_head',
                category='head',
                parameters=14_000_000,
                memory_mb=54,
                required_for_training=True,
                required_for_inference=True,
                dependencies=['unified_backbone']
            ),
            
            # ========== 전문 분석 모듈 (학습 필수) ==========
            'emotion_dsp_simulator': ModuleInfo(
                name='emotion_dsp_simulator',
                category='analyzer',
                parameters=1_180_000,
                memory_mb=5,
                required_for_training=True,
                required_for_inference=True,
                dependencies=[]
            ),
            
            'kalman_filter': ModuleInfo(
                name='kalman_filter',
                category='filter',
                parameters=742,
                memory_mb=0.1,
                required_for_training=True,
                required_for_inference=True,
                dependencies=['emotion_dsp_simulator']
            ),
            
            # bentham_calculator 제거 - NeuralBenthamCalculator(78M)로 대체됨
            
            # ========== 신경망 분석기 모듈 (학습 필수, 378M) ==========
            'neural_emotion': ModuleInfo(
                name='neural_emotion',
                category='neural_analyzer',
                parameters=133_360_719,
                memory_mb=512,
                required_for_training=True,
                required_for_inference=True,
                dependencies=[],
                gpu_resident=True
            ),
            
            'neural_bentham': ModuleInfo(
                name='neural_bentham',
                category='neural_analyzer',
                parameters=78_019_458,
                memory_mb=300,
                required_for_training=True,
                required_for_inference=True,
                dependencies=[],
                gpu_resident=True
            ),
            
            'neural_regret': ModuleInfo(
                name='neural_regret',
                category='neural_analyzer',
                parameters=153_610_522,
                memory_mb=590,
                required_for_training=True,
                required_for_inference=True,
                dependencies=[],
                gpu_resident=True
            ),
            
            'neural_surd': ModuleInfo(
                name='neural_surd',
                category='neural_analyzer',
                parameters=13_276_424,
                memory_mb=51,
                required_for_training=True,
                required_for_inference=True,
                dependencies=[],
                gpu_resident=True
            ),
            
            # ========== 추론 전용 모듈 (학습 제외) ==========
            'advanced_emotion_analyzer': ModuleInfo(
                name='advanced_emotion_analyzer',
                category='analyzer',
                parameters=5_000_000,  # MoE 등
                memory_mb=20,
                required_for_training=False,
                required_for_inference=True,
                dependencies=['emotion_dsp_simulator', 'kalman_filter']
            ),
            
            'advanced_surd_analyzer': ModuleInfo(
                name='advanced_surd_analyzer',
                category='analyzer',
                parameters=3_000_000,
                memory_mb=12,
                required_for_training=False,
                required_for_inference=True,
                dependencies=[]
            ),
            
            'advanced_rumbaugh_analyzer': ModuleInfo(
                name='advanced_rumbaugh_analyzer',
                category='analyzer',
                parameters=2_000_000,
                memory_mb=8,
                required_for_training=False,
                required_for_inference=True,
                dependencies=[]
            ),
            
            'counterfactual_reasoning': ModuleInfo(
                name='counterfactual_reasoning',
                category='reasoning',
                parameters=4_000_000,
                memory_mb=15,
                required_for_training=False,
                required_for_inference=True,
                dependencies=[]
            ),
            
            # ========== 외부 모델 (스왑 필요) ==========
            'sentence_transformer': ModuleInfo(
                name='sentence_transformer',
                category='external',
                parameters=400_000_000,
                memory_mb=1500,
                required_for_training=False,
                required_for_inference=True,
                dependencies=[]
            ),
            
            'marian_translator': ModuleInfo(
                name='marian_translator',
                category='external',
                parameters=300_000_000,
                memory_mb=1200,
                required_for_training=False,
                required_for_inference=True,
                dependencies=[]
            ),
            
            'llm_engine': ModuleInfo(
                name='llm_engine',
                category='external',
                parameters=9_000_000_000,  # 9B
                memory_mb=4500,  # 4-bit 양자화
                required_for_training=False,
                required_for_inference=False,  # 전처리용
                dependencies=[]
            ),
        }
        
        return modules
    
    def set_mode(self, mode: ExecutionMode):
        """실행 모드 설정"""
        self.mode = mode
        logger.info(f"실행 모드 변경: {mode.value}")
        
        # 모드에 따른 자동 모듈 선택
        self._auto_select_modules()
    
    def _auto_select_modules(self):
        """모드에 따른 자동 모듈 선택"""
        self.active_modules.clear()
        
        if self.mode == ExecutionMode.TRAINING:
            # 학습 모드: 필수 모듈만
            for name, info in self.modules.items():
                if info.required_for_training:
                    self.active_modules.add(name)
                    
        elif self.mode == ExecutionMode.INFERENCE:
            # 추론 모드: 모든 모듈
            for name, info in self.modules.items():
                if info.required_for_inference:
                    self.active_modules.add(name)
                    
        elif self.mode == ExecutionMode.EVALUATION:
            # 평가 모드: 학습 + 일부 추론
            for name, info in self.modules.items():
                if info.required_for_training or info.category == 'analyzer':
                    self.active_modules.add(name)
                    
        elif self.mode == ExecutionMode.DEBUG:
            # 디버그 모드: 최소 모듈
            self.active_modules.update(['unified_backbone', 'emotion_empathy_head'])
        
        # 의존성 해결
        self._resolve_dependencies()
        
        logger.info(f"활성 모듈 수: {len(self.active_modules)}")
        logger.debug(f"활성 모듈: {sorted(self.active_modules)}")
    
    def _resolve_dependencies(self):
        """의존성 해결"""
        changed = True
        while changed:
            changed = False
            for module in list(self.active_modules):
                if module in self.modules:
                    for dep in self.modules[module].dependencies:
                        if dep not in self.active_modules:
                            self.active_modules.add(dep)
                            changed = True
    
    def get_active_modules(self) -> List[str]:
        """활성 모듈 목록 반환"""
        return sorted(self.active_modules)
    
    def get_module_info(self, module_name: str) -> Optional[ModuleInfo]:
        """모듈 정보 반환"""
        return self.modules.get(module_name)
    
    def calculate_memory_usage(self) -> Dict[str, float]:
        """메모리 사용량 계산"""
        total_params = 0
        total_memory = 0.0
        gpu_memory = 0.0
        
        for module in self.active_modules:
            if module in self.modules:
                info = self.modules[module]
                total_params += info.parameters
                total_memory += info.memory_mb
                
                if info.gpu_resident:
                    gpu_memory += info.memory_mb
        
        return {
            'total_parameters': total_params,
            'total_memory_mb': total_memory,
            'gpu_memory_mb': gpu_memory,
            'cpu_memory_mb': total_memory - gpu_memory
        }
    
    def get_load_order(self) -> List[str]:
        """모듈 로드 순서 결정 (의존성 고려)"""
        loaded = set()
        order = []
        
        def can_load(module: str) -> bool:
            if module not in self.modules:
                return False
            deps = self.modules[module].dependencies
            return all(dep in loaded for dep in deps)
        
        # 의존성 순서대로 로드
        remaining = self.active_modules.copy()
        
        while remaining:
            for module in sorted(remaining):
                if can_load(module):
                    order.append(module)
                    loaded.add(module)
                    remaining.remove(module)
                    break
            else:
                # 순환 의존성 또는 해결 불가
                logger.warning(f"의존성 해결 불가: {remaining}")
                order.extend(sorted(remaining))
                break
        
        return order
    
    def should_use_module(self, module_name: str) -> bool:
        """모듈 사용 여부 확인"""
        return module_name in self.active_modules
    
    def get_swap_candidates(self) -> List[str]:
        """스왑 가능한 모듈 목록 (외부 모델)"""
        candidates = []
        
        for module in self.active_modules:
            if module in self.modules:
                info = self.modules[module]
                if info.category == 'external' and not info.gpu_resident:
                    candidates.append(module)
        
        return candidates
    
    def print_summary(self):
        """모듈 선택 요약 출력"""
        memory_info = self.calculate_memory_usage()
        
        print("\n" + "=" * 60)
        print(f"모듈 선택기 요약 - 모드: {self.mode.value}")
        print("=" * 60)
        
        print(f"\n📊 메모리 사용량:")
        print(f"  - 총 파라미터: {memory_info['total_parameters']:,}")
        print(f"  - 총 메모리: {memory_info['total_memory_mb']:.1f} MB")
        print(f"  - GPU 메모리: {memory_info['gpu_memory_mb']:.1f} MB")
        print(f"  - CPU 메모리: {memory_info['cpu_memory_mb']:.1f} MB")
        
        print(f"\n✅ 활성 모듈 ({len(self.active_modules)}개):")
        
        # 카테고리별 정리
        by_category = {}
        dynamic_modules = []  # self.modules에 없는 동적 모듈들
        
        for module in self.active_modules:
            if module in self.modules:
                cat = self.modules[module].category
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append(module)
            else:
                # self.modules에 등록되지 않은 동적 모듈 (advanced_*, dsp, kalman 등)
                dynamic_modules.append(module)
        
        # 등록된 모듈 출력
        for category, modules in sorted(by_category.items()):
            print(f"\n  [{category}]")
            for module in sorted(modules):
                info = self.modules[module]
                print(f"    - {module}: {info.parameters:,} params, {info.memory_mb:.1f} MB")
        
        # 동적 모듈 출력
        if dynamic_modules:
            print(f"\n  [Dynamic Modules]")
            for module in sorted(dynamic_modules):
                print(f"    - {module}: (dynamically loaded)")
        
        # 스왑 필요 모듈
        swap_modules = self.get_swap_candidates()
        if swap_modules:
            print(f"\n⚠️ 스왑 필요 모듈:")
            for module in swap_modules:
                info = self.modules[module]
                print(f"  - {module}: {info.memory_mb:.1f} MB")
        
        print("=" * 60 + "\n")


# 전역 인스턴스
_module_selector = ModuleSelector()

def get_module_selector() -> ModuleSelector:
    """전역 모듈 선택기 반환"""
    return _module_selector

def set_execution_mode(mode: ExecutionMode):
    """실행 모드 설정"""
    _module_selector.set_mode(mode)

def should_use_module(module_name: str) -> bool:
    """모듈 사용 여부 확인"""
    return _module_selector.should_use_module(module_name)


if __name__ == "__main__":
    # 테스트
    selector = ModuleSelector()
    
    print("\n🎯 학습 모드:")
    selector.set_mode(ExecutionMode.TRAINING)
    selector.print_summary()
    
    print("\n🚀 추론 모드:")
    selector.set_mode(ExecutionMode.INFERENCE)
    selector.print_summary()
    
    print("\n📊 메모리 비교:")
    for mode in ExecutionMode:
        selector.set_mode(mode)
        mem = selector.calculate_memory_usage()
        print(f"{mode.value:10s}: {mem['total_parameters']/1e6:6.1f}M params, "
              f"{mem['gpu_memory_mb']:7.1f} MB GPU")