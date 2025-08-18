"""
SentenceTransformer Singleton Manager - GPU 메모리 최적화
동일한 모델의 중복 로딩을 방지하고 공유 인스턴스 제공
"""

import os
import logging
import asyncio
import threading
# signal 제거 - WSL 호환성을 위해 threading.Timer 사용
import subprocess
import time
from typing import Dict, Optional, Any
from sentence_transformers import SentenceTransformer
from config import MODELS_DIR, get_device

logger = logging.getLogger(__name__)

def load_sentence_transformer_with_wsl_timeout(model_name, device, cache_folder, timeout_seconds=15):
    """
WSL 호환 SentenceTransformer 로딩 (기존 리소스 관리 전략 유지)
- multiprocessing.Process 기반 강력한 타임아웃 (WSL 환경 안정성)
- 순차적 로딩과 재시도 로직 유지
    """
    import multiprocessing
    import queue
    import pickle
    import tempfile
    
    def loading_worker(model_name, device, cache_folder, result_path, error_path):
        """별도 프로세스에서 실행되는 로더 함수"""
        try:
            # 프로세스 내에서 필요한 import
            import logging
            import os
            import sys
            
            logger = logging.getLogger(__name__)
            logger.info(f"[Process] SentenceTransformer 로딩 시작: {model_name}")
            
            # 모델 파일 존재 여부 사전 확인
            model_path = os.path.join(cache_folder, f"models--sentence-transformers--{model_name.replace('/', '--')}")
            if os.path.exists(model_path):
                logger.info(f"[Process] 캐시된 모델 발견: {model_path}")
            
            # 환경 변수 설정으로 MaskFormer 관련 의존성 문제 회피
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_DATASETS_OFFLINE'] = '1'
            
            # Import를 보호된 환경에서 수행
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"[Process] SentenceTransformer import 성공")
            
            model = SentenceTransformer(
                model_name,
                device=device,
                cache_folder=cache_folder
            )
            
            logger.info(f"[Process] SentenceTransformer 로딩 성공: {model_name}")
            
            # 결과를 파일로 저장 (프로세스 간 통신)
            with open(result_path, 'wb') as f:
                pickle.dump(model, f)
                
        except Exception as e:
            import traceback
            logger = logging.getLogger(__name__)
            logger.error(f"[Process] SentenceTransformer 로딩 실패: {model_name}")
            logger.error(f"[Process] 에러: {str(e)}")
            logger.error(f"[Process] 트레이스백: {traceback.format_exc()}")
            
            # 에러를 파일로 저장
            with open(error_path, 'wb') as f:
                pickle.dump((e, traceback.format_exc()), f)
    
    # 임시 파일 경로 생성
    with tempfile.NamedTemporaryFile(delete=False) as result_file:
        result_path = result_file.name
    with tempfile.NamedTemporaryFile(delete=False) as error_file:
        error_path = error_file.name
    
    try:
        # 별도 프로세스에서 로딩 실행
        logger.info(f"멀티프로세싱 기반 SentenceTransformer 로딩 시작: {model_name}")
        
        process = multiprocessing.Process(
            target=loading_worker,
            args=(model_name, device, cache_folder, result_path, error_path),
            daemon=True
        )
        
        process.start()
        process.join(timeout=timeout_seconds)
        
        if process.is_alive():
            # 타임아웃 발생 - 프로세스 강제 종료
            logger.error(f"SentenceTransformer 로딩 타임아웃 ({timeout_seconds}초): {model_name}")
            process.terminate()
            process.join(timeout=5)  # 종료 대기
            if process.is_alive():
                process.kill()  # 강제 종료
            raise TimeoutError(f"SentenceTransformer 로딩 타임아웃: {model_name}")
        
        # 결과 확인
        if os.path.exists(result_path) and os.path.getsize(result_path) > 0:
            with open(result_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"SentenceTransformer 로딩 완료: {model_name}")
            return model
        elif os.path.exists(error_path) and os.path.getsize(error_path) > 0:
            with open(error_path, 'rb') as f:
                error, traceback_str = pickle.load(f)
            logger.error(f"로딩 중 에러 발생: {error}")
            logger.error(f"트레이스백: {traceback_str}")
            raise error
        else:
            raise RuntimeError(f"SentenceTransformer 로딩 실패 - 결과 없음: {model_name}")
            
    finally:
        # 임시 파일 정리
        for path in [result_path, error_path]:
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except:
                    pass

class SentenceTransformerManager:
    """
    SentenceTransformer 싱글톤 관리자
    - 동일한 모델의 중복 로딩 방지
    - GPU 메모리 최적화
    - 스레드 안전성 보장
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self._models: Dict[str, SentenceTransformer] = {}
        self._model_locks: Dict[str, threading.Lock] = {}
        self._gpu_semaphore = asyncio.Semaphore(2)  # 2개 동시 GPU 연산 허용
        
        logger.info("SentenceTransformerManager 초기화 완료")
    
    def get_model(self, model_name: str, device: str = None, cache_folder: str = None) -> SentenceTransformer:
        """
        모델 인스턴스 반환 (싱글톤 패턴)
        
        Args:
            model_name: 모델 이름
            device: 디바이스 설정 (없으면 자동 결정)
            cache_folder: 캐시 폴더 경로
            
        Returns:
            SentenceTransformer 인스턴스
        """
        # 모델별 고유 키 생성
        if device is None:
            device = str(get_device())
        
        model_key = f"{model_name}_{device}"
        
        # 모델이 이미 로드되어 있으면 반환
        if model_key in self._models:
            logger.info(f"기존 모델 반환: {model_key}")
            return self._models[model_key]
        
        # 모델별 락 생성
        if model_key not in self._model_locks:
            self._model_locks[model_key] = threading.Lock()
        
        # 스레드 안전하게 모델 로드 (30초 타임아웃)
        lock_acquired = self._model_locks[model_key].acquire(timeout=30.0)
        if not lock_acquired:
            logger.error(f"모델 락 획득 실패 (30초 타임아웃): {model_key}")
            raise RuntimeError(f"SentenceTransformer 모델 락 타임아웃: {model_name}")
        
        try:
            # 다시 한 번 확인 (다른 스레드에서 로드했을 수 있음)
            if model_key in self._models:
                logger.info(f"대기 중 다른 스레드에서 로드 완료: {model_key}")
                return self._models[model_key]
            
            try:
                logger.info(f"새 모델 로드 시작: {model_key}")
                
                # 캐시 폴더 설정
                if cache_folder is None:
                    cache_folder = os.path.join(MODELS_DIR, 'sentence_transformers')
                os.makedirs(cache_folder, exist_ok=True)
                
                logger.info(f"SentenceTransformer 생성 중: {model_name} (device: {device})")
                
                # WSL 호환 타임아웃으로 SentenceTransformer 로드
                logger.info(f"SentenceTransformer 로드 시작: {model_key}")
                
                # threading.Timer 기반 WSL 호환 타임아웃 (15초)
                model = load_sentence_transformer_with_wsl_timeout(
                    model_name=model_name,
                    device=device,
                    cache_folder=cache_folder,
                    timeout_seconds=15
                )
                logger.info(f"SentenceTransformer 로드 성공: {model_key}")
                
                # GPU에서 FP16 변환으로 메모리 50% 절약
                if device == 'cuda' and hasattr(model, '_modules'):
                    try:
                        logger.info(f"FP16 변환 시작: {model_key}")
                        model = model.half()  # FP16 변환
                        logger.info(f"FP16 변환 완료: {model_key}")
                    except Exception as e:
                        logger.warning(f"FP16 변환 실패 (FP32로 계속): {model_key}, 오류: {e}")
                        # FP16 실패 시 FP32로 계속 진행
                
                self._models[model_key] = model
                logger.info(f"모델 캐시 저장 완료: {model_key}")
                
                return model
                
            except TimeoutError as e:
                logger.error(f"모델 로드 타임아웃: {model_key}, 오류: {e}")
                raise RuntimeError(f"SentenceTransformer 모델 로드 타임아웃: {model_name}") from e
            except Exception as e:
                logger.error(f"모델 로드 실패: {model_key}, 오류: {e}")
                logger.error(f"오류 타입: {type(e).__name__}")
                import traceback
                logger.error(f"스택 트레이스: {traceback.format_exc()}")
                # fallback 없음 - 바로 예외 발생
                raise RuntimeError(f"SentenceTransformer 모델 로드 실패: {model_name}") from e
        finally:
            # 락 항상 해제
            self._model_locks[model_key].release()
    
    async def get_gpu_semaphore(self):
        """GPU 세마포어 반환"""
        return self._gpu_semaphore
    
    def get_loaded_models(self) -> Dict[str, str]:
        """로드된 모델 목록 반환"""
        return {key: str(type(model).__name__) for key, model in self._models.items()}
    
    def get_memory_usage(self) -> Dict[str, float]:
        """메모리 사용량 반환 (MB)"""
        import psutil
        import torch
        
        memory_info = {
            'system_used': psutil.virtual_memory().percent,
            'process_memory': psutil.Process().memory_info().rss / (1024 * 1024),
            'loaded_models': len(self._models)
        }
        
        if torch.cuda.is_available():
            memory_info['gpu_allocated'] = torch.cuda.memory_allocated() / (1024 * 1024)
            memory_info['gpu_cached'] = torch.cuda.memory_reserved() / (1024 * 1024)
        
        return memory_info
    
    def clear_cache(self):
        """모델 캐시 정리"""
        logger.info("모델 캐시 정리 시작")
        self._models.clear()
        self._model_locks.clear()
        logger.info("모델 캐시 정리 완료")

# 전역 매니저 인스턴스
_manager = SentenceTransformerManager()

def get_sentence_transformer(model_name: str, device: str = None, cache_folder: str = None) -> SentenceTransformer:
    """
    공유 SentenceTransformer 인스턴스 반환
    
    Args:
        model_name: 모델 이름
        device: 디바이스 설정
        cache_folder: 캐시 폴더
        
    Returns:
        SentenceTransformer 인스턴스
    """
    return _manager.get_model(model_name, device, cache_folder)

async def get_gpu_semaphore():
    """GPU 세마포어 반환"""
    return await _manager.get_gpu_semaphore()

def get_model_info():
    """로드된 모델 정보 반환"""
    return _manager.get_loaded_models()

def get_memory_info():
    """메모리 사용량 정보 반환"""
    return _manager.get_memory_usage()

def clear_model_cache():
    """모델 캐시 정리"""
    _manager.clear_cache()