"""
SentenceTransformer Singleton Manager - Persistent Subprocess Server Architecture

새로운 아키텍처:
- Persistent subprocess server (RAM 상주 모델)
- 완전한 프로세스 격리 (signal handler inheritance 완전 차단)
- JSON IPC 통신
- 성능 최적화 (한번 로딩 후 지속적 서비스)
- WSL 호환성 및 안정성 보장
"""

import os
import logging
import asyncio
import threading
import time
from typing import Dict, Optional, Any, List
from sentence_transformer_client import SentenceTransformerClient
from config import MODELS_DIR, get_device

logger = logging.getLogger(__name__)

class SentenceTransformerManager:
    """
    SentenceTransformer 싱글톤 관리자 - Persistent Subprocess Server 기반
    
    새로운 아키텍처:
    - Persistent subprocess server (모델 RAM 상주)
    - 완전한 프로세스 격리 (signal handler inheritance 차단)
    - JSON IPC 통신
    - 자동 재연결 및 복구
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
        self._clients: Dict[str, SentenceTransformerClient] = {}  # 모델별 클라이언트
        self._model_locks: Dict[str, threading.Lock] = {}
        self._gpu_semaphore = asyncio.Semaphore(2)  # 2개 동시 GPU 연산 허용
        
        # 서버 스크립트 경로 설정
        self._server_script_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "sentence_transformer_server.py"
        )
        
        logger.info("SentenceTransformerManager 초기화 완료 (Subprocess Server 기반)")
    
    def get_model(self, model_name: str, device: str = None, cache_folder: str = None):
        """
        모델 클라이언트 반환 (Subprocess Server 기반)
        
        Args:
            model_name: 모델 이름
            device: 디바이스 설정 (없으면 자동 결정)
            cache_folder: 캐시 폴더 경로
            
        Returns:
            SentenceTransformerProxy 객체 (SentenceTransformer 호환 인터페이스)
        """
        # 모델별 고유 키 생성
        if device is None:
            device = str(get_device())
        
        model_key = f"{model_name}_{device}"
        
        # 클라이언트가 이미 존재하면 반환
        if model_key in self._clients:
            # Health check로 연결 상태 확인
            try:
                health_result = self._clients[model_key].health_check(force=False)
                if health_result.get("status") == "success":
                    logger.info(f"기존 클라이언트 재사용: {model_key}")
                    return SentenceTransformerProxy(self._clients[model_key])
                else:
                    logger.warning(f"기존 클라이언트 불안정 - 재생성: {model_key}")
                    # 기존 클라이언트 정리
                    self._clients[model_key].stop_server()
                    del self._clients[model_key]
            except Exception as e:
                logger.warning(f"Health check 실패 - 클라이언트 재생성: {model_key}, 오류: {e}")
                # 기존 클라이언트 정리
                try:
                    self._clients[model_key].stop_server()
                except Exception:
                    pass
                del self._clients[model_key]
        
        # 모델별 락 생성
        if model_key not in self._model_locks:
            self._model_locks[model_key] = threading.Lock()
        
        # 스레드 안전하게 클라이언트 생성 (60초 타임아웃)
        lock_acquired = self._model_locks[model_key].acquire(timeout=60.0)
        if not lock_acquired:
            logger.error(f"모델 락 획득 실패 (60초 타임아웃): {model_key}")
            raise RuntimeError(f"SentenceTransformer 모델 락 타임아웃: {model_name}")
        
        try:
            # 다시 한 번 확인 (다른 스레드에서 생성했을 수 있음)
            if model_key in self._clients:
                logger.info(f"대기 중 다른 스레드에서 생성 완료: {model_key}")
                return SentenceTransformerProxy(self._clients[model_key])
            
            try:
                logger.info(f"새 클라이언트 생성 시작: {model_key}")
                
                # 캐시 폴더 설정
                if cache_folder is None:
                    cache_folder = os.path.join(MODELS_DIR, 'sentence_transformers')
                os.makedirs(cache_folder, exist_ok=True)
                
                # 새 클라이언트 생성
                client = SentenceTransformerClient(
                    server_script_path=self._server_script_path,
                    startup_timeout=60.0,  # 충분한 시간
                    request_timeout=120.0
                )
                
                # 서버 시작 및 모델 로딩
                logger.info(f"서버 시작 및 모델 로딩: {model_name} (device: {device})")
                
                # 서버 시작
                if not client.start_server():
                    raise RuntimeError(f"서버 시작 실패: {model_name}")
                
                # 모델 로딩
                load_result = client.load_model(
                    model_name=model_name,
                    device=device,
                    cache_folder=cache_folder
                )
                
                if load_result.get("status") != "success":
                    error_msg = load_result.get("error", "Unknown error")
                    raise RuntimeError(f"모델 로딩 실패: {error_msg}")
                
                # 클라이언트 저장
                self._clients[model_key] = client
                logger.info(f"클라이언트 생성 및 모델 로딩 성공: {model_key}")
                
                return SentenceTransformerProxy(client)
                
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
        loaded_models = {}
        for key, client in self._clients.items():
            try:
                health_result = client.health_check(force=False)
                if health_result.get("status") == "success":
                    model_info = health_result.get("result", {})
                    model_name = model_info.get("model_name", "Unknown")
                    loaded_models[key] = f"SentenceTransformerProxy({model_name})"
                else:
                    loaded_models[key] = "Disconnected"
            except Exception:
                loaded_models[key] = "Error"
        return loaded_models
    
    def get_memory_usage(self) -> Dict[str, float]:
        """메모리 사용량 반환 (MB)"""
        import psutil
        import torch
        
        memory_info = {
            'system_used': psutil.virtual_memory().percent,
            'process_memory': psutil.Process().memory_info().rss / (1024 * 1024),
            'loaded_clients': len(self._clients)
        }
        
        # 각 클라이언트의 메모리 사용량 수집
        total_server_memory = 0
        for key, client in self._clients.items():
            try:
                health_result = client.health_check(force=False)
                if health_result.get("status") == "success":
                    server_info = health_result.get("result", {})
                    server_memory = server_info.get("memory_usage", 0)
                    total_server_memory += server_memory
            except Exception:
                pass
        
        memory_info['server_processes_memory'] = total_server_memory
        
        if torch.cuda.is_available():
            memory_info['gpu_allocated'] = torch.cuda.memory_allocated() / (1024 * 1024)
            memory_info['gpu_cached'] = torch.cuda.memory_reserved() / (1024 * 1024)
        
        return memory_info
    
    def clear_cache(self):
        """모델 캐시 정리"""
        logger.info("클라이언트 캐시 정리 시작")
        
        # 모든 클라이언트 서버 종료
        for key, client in self._clients.items():
            try:
                logger.info(f"클라이언트 종료 중: {key}")
                client.stop_server()
            except Exception as e:
                logger.warning(f"클라이언트 종료 실패: {key}, 오류: {e}")
        
        self._clients.clear()
        self._model_locks.clear()
        logger.info("클라이언트 캐시 정리 완료")

class SentenceTransformerProxy:
    """
    SentenceTransformer 호환 인터페이스 - Subprocess Client Wrapper
    
    기존 SentenceTransformer API와 동일한 인터페이스 제공
    내부적으로는 subprocess server와 통신
    """
    
    def __init__(self, client: SentenceTransformerClient):
        self.client = client
        self._model_info = None
        
        # 모델 정보 캐시
        try:
            health_result = client.health_check(force=True)
            if health_result.get("status") == "success":
                self._model_info = health_result.get("result", {})
        except Exception as e:
            logger.warning(f"모델 정보 캐시 실패: {e}")
    
    def encode(self, sentences: List[str], **kwargs) -> List[List[float]]:
        """
        텍스트를 임베딩으로 변환 (SentenceTransformer 호환)
        
        Args:
            sentences: 인코딩할 텍스트 리스트
            **kwargs: 추가 인자
            
        Returns:
            임베딩 리스트
        """
        # 단일 문자열을 리스트로 변환
        if isinstance(sentences, str):
            sentences = [sentences]
            return_single = True
        else:
            return_single = False
        
        # 서버에 인코딩 요청
        response = self.client.encode_texts(sentences, **kwargs)
        
        if response.get("status") != "success":
            error_msg = response.get("error", "Unknown error")
            raise RuntimeError(f"텍스트 인코딩 실패: {error_msg}")
        
        embeddings = response.get("result", {}).get("embeddings", [])
        
        # 단일 문자열 입력이었다면 단일 결과 반환
        if return_single and embeddings:
            return embeddings[0]
        
        return embeddings
    
    @property
    def max_seq_length(self) -> Optional[int]:
        """모델의 최대 시퀀스 길이"""
        if self._model_info:
            return self._model_info.get("model_max_seq_length")
        return None
    
    @property
    def device(self) -> str:
        """모델이 로드된 디바이스"""
        if self._model_info:
            return self._model_info.get("device", "unknown")
        return "unknown"
    
    def get_sentence_embedding_dimension(self) -> int:
        """모델의 임베딩 차원 반환"""
        if self._model_info:
            # embedding_dimension 또는 hidden_size 키 사용
            dim = self._model_info.get("embedding_dimension")
            if dim is not None:
                return dim
            # fallback으로 hidden_size 확인
            dim = self._model_info.get("hidden_size")
            if dim is not None:
                return dim
        # 기본값 반환 (일반적인 임베딩 차원)
        return 768
    
    def __repr__(self) -> str:
        model_name = self._model_info.get("model_name", "Unknown") if self._model_info else "Unknown"
        device = self.device
        return f"SentenceTransformerProxy(model='{model_name}', device='{device}')"

# 전역 매니저 인스턴스
_manager = SentenceTransformerManager()

def get_sentence_transformer(model_name: str, device: str = None, cache_folder: str = None):
    """
    공유 SentenceTransformer 인스턴스 반환 (Subprocess Server 기반)
    
    Args:
        model_name: 모델 이름
        device: 디바이스 설정
        cache_folder: 캐시 폴더
        
    Returns:
        SentenceTransformerProxy 인스턴스 (SentenceTransformer 호환)
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

def get_client_info():
    """활성 클라이언트 정보 반환"""
    client_info = {}
    for key, client in _manager._clients.items():
        try:
            health_result = client.health_check(force=False)
            client_info[key] = health_result
        except Exception as e:
            client_info[key] = {"status": "error", "error": str(e)}
    return client_info