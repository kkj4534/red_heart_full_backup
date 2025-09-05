"""
Memory Swap Manager for LLM and Red Heart System
LLM과 Red Heart 시스템 간 GPU 메모리 스왑 관리

MD 문서 사양에 따른 구현:
1. 초기: Red Heart를 RAM에 대기
2. LLM을 GPU로 로드 → 상황 해석
3. LLM → RAM, Red Heart → GPU (스왑)
4. Red Heart 추론 수행
5. 결과를 LLM에 전달 필요시 다시 스왑
"""

import torch
import logging
import asyncio
import time
import gc
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import psutil
# GPUtil 제거 - torch.cuda로 대체

logger = logging.getLogger('RedHeart.MemorySwapManager')

class SystemType(Enum):
    """시스템 타입"""
    LLM = "llm"
    RED_HEART = "red_heart"
    NONE = "none"

@dataclass
class MemoryStatus:
    """메모리 상태"""
    gpu_total: int  # MB
    gpu_used: int   # MB
    gpu_free: int   # MB
    ram_total: int  # MB
    ram_used: int   # MB
    ram_free: int   # MB
    current_on_gpu: SystemType
    current_on_ram: SystemType
    timestamp: float

class SystemSwapManager:
    """LLM과 Red Heart 간 메모리 스왑 관리
    
    MD 문서의 스왑 전략 구현:
    - 초기: Red Heart를 RAM에 대기
    - LLM과 Red Heart를 필요에 따라 GPU/RAM 간 스왑
    - 메모리 효율적인 추론 파이프라인
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = logger
        self.config = config or {}
        
        # 시스템 인스턴스
        self.llm_model = None
        self.red_heart_system = None
        
        # 현재 상태
        self.current_on_gpu = SystemType.NONE
        self.current_on_ram = SystemType.NONE
        
        # 스왑 히스토리
        self.swap_history = []
        self.swap_count = 0
        
        # 메모리 임계값 (MB)
        self.gpu_threshold = self.config.get('gpu_threshold', 7000)  # 8GB GPU 기준
        self.ram_threshold = self.config.get('ram_threshold', 16000)  # 16GB RAM 기준
        
        # 스왑 최적화 설정
        self.enable_optimization = self.config.get('enable_optimization', True)
        self.cache_states = self.config.get('cache_states', True)
        
        # 상태 캐시
        self._state_cache = {}
        
        self.logger.info("Memory Swap Manager 초기화 완료")
        self.logger.info(f"  GPU 임계값: {self.gpu_threshold}MB")
        self.logger.info(f"  RAM 임계값: {self.ram_threshold}MB")
    
    def get_gpu_memory_status(self):
        """GPU 메모리 상태 확인 (GPUtil 대신 torch 사용)"""
        if torch.cuda.is_available():
            # 현재 GPU 메모리 상태
            gpu_total = torch.cuda.get_device_properties(0).total_memory // (1024**2)  # MB
            gpu_reserved = torch.cuda.memory_reserved(0) // (1024**2)  # MB
            gpu_allocated = torch.cuda.memory_allocated(0) // (1024**2)  # MB
            gpu_free = gpu_total - gpu_allocated
            
            return {
                'total': gpu_total,
                'used': gpu_allocated,
                'reserved': gpu_reserved,
                'free': gpu_free
            }
        return None
    
    async def initialize(self, red_heart_system=None, llm_model=None):
        """초기화 - Red Heart는 RAM, LLM은 미로드
        
        MD 문서: "초기: Red Heart를 RAM에 대기"
        """
        self.logger.info("시스템 초기화 시작...")
        
        # Red Heart 시스템 설정
        if red_heart_system:
            self.red_heart_system = red_heart_system
            if hasattr(self.red_heart_system, 'to'):
                self.red_heart_system.to('cpu')  # RAM에 대기
                self.current_on_ram = SystemType.RED_HEART
                self.logger.info("  Red Heart 시스템을 RAM에 로드")
        
        # LLM은 아직 로드하지 않음
        self.llm_model = llm_model  # 참조만 저장
        self.current_on_gpu = SystemType.NONE
        
        # 초기 메모리 상태 기록
        await self._log_memory_status("초기화 완료")
        
        self.logger.info("시스템 초기화 완료")
    
    async def process_with_llm(self, text: str) -> Dict[str, Any]:
        """LLM으로 초기 처리
        
        MD 문서 워크플로우:
        1. LLM을 GPU로
        2. LLM으로 상황 해석 및 시나리오 생성
        3. Red Heart로 스왑
        4. Red Heart 추론
        5. 필요시 LLM으로 다시 스왑하여 자연어 생성
        """
        start_time = time.time()
        result = {}
        
        try:
            # Step 1: LLM을 GPU로
            self.logger.info("Step 1: LLM을 GPU로 로드")
            await self.swap_to_gpu(SystemType.LLM)
            
            # Step 2: LLM으로 상황 해석 및 시나리오 생성
            self.logger.info("Step 2: LLM으로 시나리오 생성")
            if self.llm_model:
                scenarios = await self._generate_scenarios_with_llm(text)
                result['scenarios'] = scenarios
            else:
                # LLM이 없으면 기본 시나리오 생성
                scenarios = [text]
                result['scenarios'] = scenarios
            
            # Step 3: Red Heart로 스왑
            self.logger.info("Step 3: Red Heart로 스왑")
            await self.swap_to_gpu(SystemType.RED_HEART)
            
            # Step 4: Red Heart 추론
            self.logger.info("Step 4: Red Heart 윤리적 분석")
            if self.red_heart_system and hasattr(self.red_heart_system, 'analyze_ethical_dilemma'):
                analysis = await self.red_heart_system.analyze_ethical_dilemma(scenarios)
                result['analysis'] = analysis
            else:
                result['analysis'] = {'error': 'Red Heart 시스템 미초기화'}
            
            # Step 5: 필요시 LLM으로 다시 스왑하여 자연어 생성
            if self.config.get('generate_explanation', True):
                self.logger.info("Step 5: LLM으로 설명 생성")
                await self.swap_to_gpu(SystemType.LLM)
                
                if self.llm_model:
                    explanation = await self._generate_explanation_with_llm(result['analysis'])
                    result['explanation'] = explanation
            
            result['processing_time'] = time.time() - start_time
            result['swap_count'] = self.swap_count
            
            self.logger.info(f"처리 완료: {result['processing_time']:.2f}초, 스왑 {self.swap_count}회")
            
            return result
            
        except Exception as e:
            self.logger.error(f"처리 실패: {e}")
            return {
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    async def swap_to_gpu(self, target: SystemType):
        """지정된 시스템을 GPU로 스왑
        
        메모리 효율적인 스왑 전략:
        1. 현재 GPU 점유 시스템을 RAM으로
        2. GPU 메모리 정리
        3. 타겟 시스템을 GPU로
        """
        if self.current_on_gpu == target:
            self.logger.debug(f"{target.value}는 이미 GPU에 있음")
            return  # 이미 GPU에 있음
        
        self.logger.info(f"스왑 시작: {target.value} → GPU")
        swap_start = time.time()
        
        # 1. 현재 GPU 점유 시스템을 RAM으로
        if self.current_on_gpu == SystemType.LLM:
            await self._move_llm_to_ram()
        elif self.current_on_gpu == SystemType.RED_HEART:
            await self._move_red_heart_to_ram()
        
        # 2. GPU 메모리 정리
        await self._cleanup_gpu_memory()
        
        # 3. 타겟을 GPU로
        if target == SystemType.LLM:
            await self._move_llm_to_gpu()
        elif target == SystemType.RED_HEART:
            await self._move_red_heart_to_gpu()
        
        # 스왑 히스토리 기록
        swap_time = time.time() - swap_start
        self.swap_history.append({
            'from': self.current_on_gpu.value,
            'to': target.value,
            'time': swap_time,
            'timestamp': time.time()
        })
        self.swap_count += 1
        
        self.current_on_gpu = target
        
        self.logger.info(f"스왑 완료: {swap_time:.2f}초")
        await self._log_memory_status(f"스왑 후 ({target.value} on GPU)")
    
    async def _move_llm_to_gpu(self):
        """LLM을 GPU로 이동"""
        if self.llm_model is None:
            # LLM 최초 로드
            await self._load_llm()
        
        if self.llm_model and hasattr(self.llm_model, 'to'):
            self.llm_model = self.llm_model.to('cuda')
            self.logger.debug("LLM을 GPU로 이동")
        elif self.llm_model:
            # llama-cpp 모델의 경우 재로드 필요
            await self._reload_llm_on_gpu()
    
    async def _move_llm_to_ram(self):
        """LLM을 RAM으로 이동"""
        if self.llm_model and hasattr(self.llm_model, 'to'):
            self.llm_model = self.llm_model.to('cpu')
            self.logger.debug("LLM을 RAM으로 이동")
        elif self.llm_model:
            # 상태 저장 후 언로드
            if self.cache_states:
                self._cache_llm_state()
            self._unload_llm()
    
    async def _move_red_heart_to_gpu(self):
        """Red Heart를 GPU로 이동"""
        if self.red_heart_system and hasattr(self.red_heart_system, 'to'):
            self.red_heart_system.to('cuda')
            self.logger.debug("Red Heart를 GPU로 이동")
    
    async def _move_red_heart_to_ram(self):
        """Red Heart를 RAM으로 이동"""
        if self.red_heart_system and hasattr(self.red_heart_system, 'to'):
            self.red_heart_system.to('cpu')
            self.logger.debug("Red Heart를 RAM으로 이동")
    
    async def _load_llm(self):
        """LLM 최초 로드"""
        self.logger.info("LLM 모델 로드 중...")
        
        try:
            # 실제 LLM 로드 로직
            from llm_module.advanced_llm_engine import AdvancedLLMEngine
            
            model_path = self.config.get('llm_model_path', 
                                        'llm_module/HelpingAI2-9B.Q4_K_M.gguf')
            
            self.llm_model = AdvancedLLMEngine(model_path=model_path)
            await self.llm_model.initialize()
            
            self.logger.info("LLM 모델 로드 완료")
            
        except Exception as e:
            self.logger.error(f"LLM 로드 실패: {e}")
            self.llm_model = None
    
    async def _reload_llm_on_gpu(self):
        """LLM을 GPU에 재로드 (llama-cpp 등)"""
        self.logger.debug("LLM GPU 재로드")
        
        # 기존 모델 언로드
        if self.llm_model:
            self._unload_llm()
        
        # GPU에 재로드
        await self._load_llm()
    
    def _unload_llm(self):
        """LLM 언로드"""
        if self.llm_model:
            # 메모리 해제
            if hasattr(self.llm_model, 'unload'):
                self.llm_model.unload()
            del self.llm_model
            self.llm_model = None
            self.logger.debug("LLM 언로드")
    
    def _cache_llm_state(self):
        """LLM 상태 캐시"""
        if self.llm_model and hasattr(self.llm_model, 'get_state'):
            self._state_cache['llm'] = self.llm_model.get_state()
            self.logger.debug("LLM 상태 캐시됨")
    
    async def _cleanup_gpu_memory(self):
        """GPU 메모리 정리"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # 가비지 컬렉션
        gc.collect()
        
        # 짧은 대기
        await asyncio.sleep(0.1)
        
        self.logger.debug("GPU 메모리 정리 완료")
    
    async def _generate_scenarios_with_llm(self, text: str) -> List[str]:
        """LLM으로 시나리오 생성"""
        if not self.llm_model:
            return [text]
        
        try:
            # LLM에게 윤리적 딜레마 시나리오 생성 요청
            prompt = f"""
다음 상황에 대해 3개의 다른 행동 시나리오를 제시하세요:

상황: {text}

시나리오 1 (적극적):
시나리오 2 (중도적):
시나리오 3 (보수적):
"""
            
            if hasattr(self.llm_model, 'generate'):
                response = await self.llm_model.generate(prompt)
                # 시나리오 파싱
                scenarios = self._parse_scenarios(response)
                return scenarios if scenarios else [text]
            else:
                return [text]
                
        except Exception as e:
            self.logger.error(f"시나리오 생성 실패: {e}")
            return [text]
    
    async def _generate_explanation_with_llm(self, analysis: Dict) -> str:
        """LLM으로 분석 결과 설명 생성"""
        if not self.llm_model:
            return "설명 생성 불가"
        
        try:
            # 분석 결과를 자연어로 설명
            selected = analysis.get('selected_scenarios', [])
            if selected:
                top_scenario = selected[0]
                prompt = f"""
다음 윤리적 분석 결과를 쉽게 설명해주세요:

선택된 시나리오: {top_scenario.get('original_scenario', '')}
효용 점수: {top_scenario.get('utility_score', 0):.2f}
후회 가능성: {top_scenario.get('regret_potential', 0):.2f}
추천: {analysis.get('recommendation', '')}

한국어로 2-3문장으로 요약:
"""
                
                if hasattr(self.llm_model, 'generate'):
                    explanation = await self.llm_model.generate(prompt)
                    return explanation
            
            return "분석 결과를 설명할 수 없습니다."
            
        except Exception as e:
            self.logger.error(f"설명 생성 실패: {e}")
            return f"설명 생성 오류: {e}"
    
    def _parse_scenarios(self, response: str) -> List[str]:
        """LLM 응답에서 시나리오 파싱"""
        scenarios = []
        
        # 간단한 파싱 로직
        lines = response.split('\n')
        current_scenario = []
        
        for line in lines:
            if '시나리오' in line and ':' in line:
                if current_scenario:
                    scenarios.append(' '.join(current_scenario))
                    current_scenario = []
                # 콜론 이후 텍스트 추가
                parts = line.split(':', 1)
                if len(parts) > 1:
                    current_scenario.append(parts[1].strip())
            elif current_scenario:
                current_scenario.append(line.strip())
        
        # 마지막 시나리오 추가
        if current_scenario:
            scenarios.append(' '.join(current_scenario))
        
        return scenarios[:3]  # 최대 3개
    
    async def _log_memory_status(self, context: str = ""):
        """메모리 상태 로깅"""
        status = self.get_memory_status()
        
        self.logger.info(f"메모리 상태 {context}:")
        self.logger.info(f"  GPU: {status.gpu_used}/{status.gpu_total}MB ({status.gpu_free}MB 여유)")
        self.logger.info(f"  RAM: {status.ram_used}/{status.ram_total}MB ({status.ram_free}MB 여유)")
        self.logger.info(f"  GPU: {status.current_on_gpu.value}, RAM: {status.current_on_ram.value}")
    
    def get_memory_status(self) -> MemoryStatus:
        """현재 메모리 상태 조회"""
        # GPU 메모리
        gpu_total = 0
        gpu_used = 0
        gpu_free = 0
        
        if torch.cuda.is_available():
            gpu_total = torch.cuda.get_device_properties(0).total_memory // (1024**2)
            gpu_used = torch.cuda.memory_allocated() // (1024**2)
            gpu_free = gpu_total - gpu_used
        
        # RAM 메모리
        ram = psutil.virtual_memory()
        ram_total = ram.total // (1024**2)
        ram_used = ram.used // (1024**2)
        ram_free = ram.available // (1024**2)
        
        return MemoryStatus(
            gpu_total=gpu_total,
            gpu_used=gpu_used,
            gpu_free=gpu_free,
            ram_total=ram_total,
            ram_used=ram_used,
            ram_free=ram_free,
            current_on_gpu=self.current_on_gpu,
            current_on_ram=self.current_on_ram,
            timestamp=time.time()
        )
    
    def get_swap_statistics(self) -> Dict[str, Any]:
        """스왑 통계 조회"""
        if not self.swap_history:
            return {
                'total_swaps': 0,
                'avg_swap_time': 0,
                'total_swap_time': 0
            }
        
        swap_times = [s['time'] for s in self.swap_history]
        
        return {
            'total_swaps': self.swap_count,
            'avg_swap_time': sum(swap_times) / len(swap_times),
            'total_swap_time': sum(swap_times),
            'swap_history': self.swap_history[-10:]  # 최근 10개
        }


# 테스트 코드
async def test_swap_manager():
    """스왑 매니저 테스트"""
    import sys
    sys.path.append('/mnt/c/large_project/linux_red_heart')
    
    config = {
        'gpu_threshold': 7000,
        'ram_threshold': 16000,
        'llm_model_path': 'llm_module/HelpingAI2-9B.Q4_K_M.gguf',
        'generate_explanation': True
    }
    
    manager = SystemSwapManager(config)
    
    # 초기화 (Red Heart만 RAM에)
    await manager.initialize()
    
    # 테스트 텍스트
    test_text = "친구의 비밀을 알게 되었는데, 그것이 다른 사람에게 해를 끼칠 수 있는 내용입니다."
    
    # 처리
    result = await manager.process_with_llm(test_text)
    
    print("처리 결과:")
    print(f"  시나리오 수: {len(result.get('scenarios', []))}")
    print(f"  처리 시간: {result.get('processing_time', 0):.2f}초")
    print(f"  스왑 횟수: {result.get('swap_count', 0)}")
    
    # 통계
    stats = manager.get_swap_statistics()
    print("\n스왑 통계:")
    print(f"  총 스왑: {stats['total_swaps']}")
    print(f"  평균 스왑 시간: {stats['avg_swap_time']:.2f}초")
    
    # 메모리 상태
    status = manager.get_memory_status()
    print("\n최종 메모리 상태:")
    print(f"  GPU: {status.gpu_used}/{status.gpu_total}MB")
    print(f"  RAM: {status.ram_used}/{status.ram_total}MB")


if __name__ == "__main__":
    # 테스트 실행
    asyncio.run(test_swap_manager())