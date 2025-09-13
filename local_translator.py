#!/usr/bin/env python3
"""
로컬 번역 모듈 - OPUS-MT 기반 한국어→영어 번역
독립적인 모듈로 분리하여 MasterMemoryOrchestrator가 관리

이 모듈은 전역 모듈로 등록되어:
1. 초기화 시점이 예측 가능
2. GPU 메모리 관리 정책 준수
3. 다른 모듈들과 일관된 관리
"""

import torch
import time
import logging
from typing import Optional, Dict
from config import get_smart_device

logger = logging.getLogger(__name__)


class LocalTranslator:
    """로컬 OPUS-MT 기반 한국어→영어 번역기
    
    특징:
    - 완전 오프라인 작동
    - MasterMemoryOrchestrator 통합
    - 캐싱을 통한 성능 최적화
    """
    
    def __init__(self, lazy_load: bool = True):
        """로컬 번역기 초기화 (lazy loading 지원)
        
        Args:
            lazy_load: True면 실제 번역이 필요할 때 모델 로드
        """
        self.model_name = 'Helsinki-NLP/opus-mt-ko-en'
        self.tokenizer = None
        self.model = None
        self.device = None
        self.translation_cache = {}  # 번역 결과 캐싱
        self.initialized = False
        self.lazy_load = lazy_load
        
        logger.info(f"LocalTranslator 생성 - {'lazy loading 모드' if lazy_load else '즉시 로드 모드'}")
        
        # lazy_load가 False일 때만 즉시 초기화
        if not self.lazy_load:
            self._initialize_model()
    
    def _initialize_model(self):
        """모델 초기화 - 전역 모듈 등록 시 즉시 실행"""
        if self.initialized:
            return
        
        try:
            logger.info(f"🔄 OPUS-MT 모델 로드 중: {self.model_name}")
            start_time = time.time()
            
            from transformers import MarianMTModel, MarianTokenizer
            
            # HF 래퍼는 모델에만 사용 (토크나이저는 직접 로드)
            from hf_model_wrapper import get_hf_wrapper, enable_auto_registration
            
            # HF 모델 자동 등록 활성화 (원본 함수 저장)
            enable_auto_registration()
            
            hf_wrapper = get_hf_wrapper()
            
            # 토크나이저는 직접 로드 (패치 문제 우회)
            self.tokenizer = MarianTokenizer.from_pretrained(
                self.model_name,
                local_files_only=True
            )
            
            # 모델은 래퍼 사용 (메모리 추적) - CPU에서만 실행
            self.model = hf_wrapper.wrapped_from_pretrained(
                MarianMTModel, self.model_name, 
                owner="translator",
                local_files_only=True,
                device_map="cpu"  # CPU 전용으로 명시
            )
            
            # CPU에서만 초기화 (GPU 메모리 절약)
            self.device = torch.device('cpu')  # CPU 고정
            self.model = self.model.to(self.device)  # CPU에 유지
            
            # 평가 모드 설정
            self.model.eval()
            
            load_time = time.time() - start_time
            logger.info(f"✅ OPUS-MT 모델 로드 완료 (소요시간: {load_time:.1f}초, 디바이스: {self.device})")
            
            # CPU 사용 로깅
            logger.info("📊 Translator는 CPU에서 초기화됨 (GPU 메모리 절약)")
            
            # DSM에 등록 (CPU resident로 시작)
            try:
                from dynamic_swap_manager import get_swap_manager, SwapPriority
                import asyncio
                
                swap_manager = get_swap_manager()
                if swap_manager:
                    # 이벤트 루프 확인 및 생성
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        # 루프가 없으면 새로 생성
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        logger.debug("DSM 등록을 위한 이벤트 루프 생성")
                    
                    # compression_enabled가 백그라운드 작업을 시작하므로 임시 비활성화
                    original_compression = swap_manager.compression_enabled
                    swap_manager.compression_enabled = False
                    
                    # 동기적으로 모델 등록
                    swap_manager.register_model(
                        "translator",
                        self.model,
                        priority=SwapPriority.HIGH  # NO FALLBACK - lazy 제거
                    )
                    
                    # 압축 설정 복구
                    swap_manager.compression_enabled = original_compression
                    
                    logger.info("✅ Translator를 DSM에 등록 (HIGH priority)")
            except Exception as e:
                logger.warning(f"DSM 등록 실패 (계속 진행): {e}")
            
            self.initialized = True
            
        except Exception as e:
            logger.error(f"❌ OPUS-MT 모델 로드 실패: {e}")
            self.initialized = False
            raise RuntimeError(f"로컬 번역기 초기화 실패: {e}")
    
    def _is_english_text(self, text: str) -> bool:
        """텍스트가 이미 영어인지 감지"""
        if not text or len(text.strip()) == 0:
            return True
        
        # 한국어 문자 비율 계산 (유니코드 범위 활용)
        korean_chars = 0
        total_chars = 0
        
        for char in text:
            if char.isalpha():  # 알파벳 문자만 고려
                total_chars += 1
                # 한글 유니코드 범위: AC00-D7AF (가-힣), 1100-11FF (자모)
                if '\uAC00' <= char <= '\uD7AF' or '\u1100' <= char <= '\u11FF':
                    korean_chars += 1
        
        if total_chars == 0:
            return True  # 알파벳이 없으면 영어로 간주 (숫자, 기호만 있는 경우)
        
        korean_ratio = korean_chars / total_chars
        return korean_ratio < 0.1  # 한국어 비율이 10% 미만이면 영어로 판단
    
    def load_to_gpu(self) -> bool:
        """필요시 모델을 GPU로 승격"""
        try:
            if not self.initialized or self.model is None:
                logger.warning("모델이 초기화되지 않음")
                return False
            
            # 이미 GPU에 있으면 스킵
            if next(self.model.parameters()).is_cuda:
                return True
            
            # GPU 가용성 확인
            if not torch.cuda.is_available():
                logger.warning("GPU 사용 불가능")
                return False
            
            # WorkflowAwareMemoryManager를 통해 GPU 메모리 확보
            logger.info("🔄 Translator GPU 메모리 요청 중...")
            from workflow_aware_memory_manager import WorkflowAwareMemoryManager
            mem_manager = WorkflowAwareMemoryManager()
            
            # DSM 실측치 사용 (허수 예약치 제거)
            from dynamic_swap_manager import get_swap_manager
            swap = get_swap_manager()
            required_mb = 0.0
            if swap and "translator" in swap.models:
                required_mb = max(0.0, swap.models["translator"].size_mb)  # 실측치
                logger.info(f"📊 Translator 실측 크기: {required_mb:.1f}MB")
            else:
                # DSM에 없으면 기본값 사용
                required_mb = 300  # 최소값
                logger.info(f"📊 Translator 기본 크기 사용: {required_mb}MB")
            
            # 동기 방식으로 GPU 메모리 요청 (DSM 실측치)
            mem_ok = mem_manager.request_gpu_blocking(
                module_name="translator",
                required_mb=required_mb,
                deps=[],
                target_util=0.85,
                timeout=30.0,
                is_required=False  # 필수가 아님
            )
            
            if not mem_ok:
                logger.error("GPU 메모리 확보 실패")
                raise RuntimeError("GPU space for translator not available")
            
            # GPU로 이동
            logger.info("🚀 Translator를 GPU로 승격 중...")
            self.model = self.model.to(torch.device('cuda'))
            self.device = torch.device('cuda')
            
            # 메모리 사용량 로깅
            allocated_mb = torch.cuda.memory_allocated(self.device) / 1024 / 1024
            logger.info(f"✅ Translator GPU 승격 완료 (메모리: {allocated_mb:.1f}MB)")
            return True
            
        except Exception as e:
            logger.error(f"❌ GPU 승격 실패: {e}")
            return False
    
    def unload_from_gpu(self) -> bool:
        """GPU에서 CPU로 언로드"""
        try:
            if not self.initialized or self.model is None:
                return False
            
            # 이미 CPU에 있으면 스킵
            if not next(self.model.parameters()).is_cuda:
                return True
            
            # CPU로 이동
            logger.info("⬇️ Translator를 CPU로 언로드 중...")
            self.model = self.model.to(torch.device('cpu'))
            self.device = torch.device('cpu')
            
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("✅ Translator CPU 언로드 완료")
            return True
            
        except Exception as e:
            logger.error(f"❌ CPU 언로드 실패: {e}")
            return False
    
    def translate_ko_to_en(self, korean_text: str) -> str:
        """한국어 텍스트를 영어로 번역
        
        Args:
            korean_text: 번역할 한국어 텍스트
            
        Returns:
            번역된 영어 텍스트
        """
        if not korean_text or len(korean_text.strip()) == 0:
            return korean_text
        
        # 영어 텍스트 감지
        if self._is_english_text(korean_text):
            logger.debug("텍스트가 이미 영어로 판단됨, 번역 생략")
            return korean_text
        
        # 캐시 확인
        cache_key = hash(korean_text.strip())
        if cache_key in self.translation_cache:
            logger.debug("번역 캐시에서 결과 반환")
            return self.translation_cache[cache_key]
        
        # Lazy loading: 초기화되지 않았으면 여기서 초기화
        if not self.initialized:
            if self.lazy_load:
                logger.info("🔄 Lazy loading: 번역이 필요해서 모델을 로드합니다...")
                self._initialize_model()
                if not self.initialized:
                    logger.error("LocalTranslator 초기화 실패")
                    raise RuntimeError("LocalTranslator initialization failed")
            else:
                logger.error("LocalTranslator가 초기화되지 않음")
                raise RuntimeError("LocalTranslator not initialized")
        
        try:
            # 번역 수행
            start_time = time.time()
            inputs = self.tokenizer([korean_text], return_tensors='pt', padding=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=128,      # 충분한 길이
                    num_beams=3,         # 적당한 품질
                    early_stopping=True, # 효율성
                    do_sample=False      # 일관성
                )
            
            translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            translation_time = time.time() - start_time
            
            # 캐시 저장
            self.translation_cache[cache_key] = translated_text
            
            logger.debug(f"번역 완료: \"{korean_text[:30]}...\" → \"{translated_text[:30]}...\" ({translation_time:.2f}초)")
            return translated_text
            
        except Exception as e:
            logger.error(f"번역 중 오류 발생: {e}")
            raise RuntimeError(f"Translation failed: {e}")
    
    def clear_cache(self):
        """번역 캐시 초기화"""
        self.translation_cache.clear()
        logger.info("번역 캐시 초기화됨")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """현재 메모리 사용량 반환"""
        if not self.initialized or self.device is None:
            return {"allocated_mb": 0, "cached_mb": 0}
        
        if self.device.type == 'cuda':
            allocated_mb = torch.cuda.memory_allocated(self.device) / 1024 / 1024
            cached_mb = torch.cuda.memory_reserved(self.device) / 1024 / 1024
            return {
                "allocated_mb": allocated_mb,
                "cached_mb": cached_mb
            }
        else:
            return {"allocated_mb": 0, "cached_mb": 0}
    
    def to(self, device):
        """모델을 다른 디바이스로 이동 (MasterMemoryOrchestrator 호환)"""
        if self.initialized and self.model is not None:
            self.device = device
            self.model = self.model.to(device)
            logger.info(f"LocalTranslator 모델을 {device}로 이동")
        return self
    
    def get_pytorch_network(self):
        """PyTorch 네트워크 반환 (HeadAdapter와의 호환성)"""
        if not self.initialized:
            logger.warning("LocalTranslator: 모델이 아직 초기화되지 않음")
            self._initialize_model()
        
        if self.model is not None:
            logger.info("✅ LocalTranslator: MarianMT 모델 반환")
            return self.model
        
        # STRICT_NO_FALLBACK
        raise RuntimeError("LocalTranslator: get_pytorch_network 실패 - 모델 없음")
    
    async def translate_async(self, text: str) -> str:
        """비동기 번역 메서드 - claude_inference.py 호환용
        
        Args:
            text: 번역할 텍스트 (한국어)
            
        Returns:
            번역된 영어 텍스트
        """
        import asyncio
        
        # 동기 메서드를 비동기로 래핑
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.translate_ko_to_en, text)
    
    def __repr__(self):
        return f"LocalTranslator(model={self.model_name}, initialized={self.initialized}, device={self.device})"