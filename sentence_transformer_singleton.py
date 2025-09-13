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
import torch
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
            # FORCE_CPU_INIT 환경변수 체크 추가
            if os.environ.get('FORCE_CPU_INIT', '0') == '1':
                device = 'cpu'
                logger.info("📌 FORCE_CPU_INIT 감지: CPU 디바이스 강제 설정")
            else:
                device = str(get_device())
        
        # CPU 모드에서는 직접 로드 (subprocess 우회)
        if device == 'cpu' or device == 'cpu:0':
            logger.info(f"📌 CPU 모드: {model_name} 직접 로드 (subprocess 우회)")
            from sentence_transformers import SentenceTransformer
            if cache_folder is None:
                cache_folder = os.path.join(MODELS_DIR, 'sentence_transformers')
            os.makedirs(cache_folder, exist_ok=True)
            
            # 직접 CPU에서 로드
            model = SentenceTransformer(model_name, device='cpu', cache_folder=cache_folder)
            
            # SimpleCPUProxy로 래핑하여 호환성 유지
            class SimpleCPUProxy:
                def __init__(self, model):
                    self.model = model
                    
                def encode(self, sentences, **kwargs):
                    # convert_to_tensor 처리
                    convert_to_tensor = kwargs.get('convert_to_tensor', False)
                    result = self.model.encode(sentences, **kwargs)
                    
                    # 텐서로 변환 요청이지만 이미 텐서인 경우 그대로 반환
                    if convert_to_tensor and torch.is_tensor(result):
                        return result
                    # 텐서가 아닌데 텐서로 변환 요청인 경우
                    elif convert_to_tensor and not torch.is_tensor(result):
                        import numpy as np
                        if isinstance(result, (list, np.ndarray)):
                            return torch.tensor(result)
                    # 텐서인데 텐서 변환 요청이 아닌 경우
                    elif not convert_to_tensor and torch.is_tensor(result):
                        return result.cpu().numpy() if result.is_cuda else result.numpy()
                    
                    return result
                
                def get_sentence_embedding_dimension(self):
                    return self.model.get_sentence_embedding_dimension()
                    
                @property
                def device(self):
                    return torch.device('cpu')
                    
                @property
                def max_seq_length(self):
                    return self.model.max_seq_length
            
            return SimpleCPUProxy(model)
        
        model_key = f"{model_name}_{device}"
        
        # 클라이언트가 이미 존재하면 반환
        if model_key in self._clients:
            # Health check로 연결 상태 확인
            try:
                health_result = self._clients[model_key].health_check(force=False)
                if health_result.get("status") == "success":
                    # 모델 로드 상태 확인
                    model_info = health_result.get("result", {})
                    if model_info.get("model_loaded", False):
                        logger.info(f"기존 클라이언트 재사용 (모델 로드됨): {model_key}")
                        return SentenceTransformerProxy(self._clients[model_key], model_key, model_name, device, cache_folder)
                    else:
                        # 서버는 살아있지만 모델이 로드되지 않음 - 모델 로드 시도
                        logger.info(f"기존 클라이언트 서버는 살아있지만 모델 미로드 - 모델 로드 시도: {model_key}")
                        load_result = self._clients[model_key].load_model(
                            model_name=model_name,
                            device=device,
                            cache_folder=cache_folder
                        )
                        if load_result.get("status") == "success":
                            logger.info(f"모델 로드 성공: {model_key}")
                            return SentenceTransformerProxy(self._clients[model_key], model_key, model_name, device, cache_folder)
                        else:
                            logger.warning(f"모델 로드 실패 - 클라이언트 재생성: {model_key}")
                            self._clients[model_key].stop_server()
                            del self._clients[model_key]
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
                return SentenceTransformerProxy(self._clients[model_key], model_key, model_name, device, cache_folder)
            
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
                    request_timeout=180.0  # 3분 (메모리 정리 후 충분한 시간)
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
                
                return SentenceTransformerProxy(client, model_key, model_name, device, cache_folder)
                
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
    
    def restart_server(self, model_name: str, device: str = None):
        """특정 모델의 서버 재시작"""
        if device is None:
            device = str(get_device())
        
        model_key = f"{model_name}_{device}"
        
        # 기존 클라이언트 종료
        if model_key in self._clients:
            try:
                logger.info(f"서버 재시작 중: {model_key}")
                self._clients[model_key].stop_server()
                del self._clients[model_key]
            except Exception as e:
                logger.warning(f"서버 종료 실패: {e}")
        
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"서버 재시작 완료: {model_key}")
    
    def unload_model_from_gpu(self, model_key: str):
        """특정 모델을 GPU에서 RAM으로 스왑 (서버는 유지)"""
        if model_key not in self._clients:
            return
        
        try:
            # GPU 모델인 경우에만 스왑
            if 'cuda' in model_key or 'gpu' in model_key:
                logger.info(f"모델을 GPU에서 RAM으로 스왑: {model_key}")
                client = self._clients[model_key]
                
                # GPU→CPU 스왑 (서버는 유지, 모델만 이동)
                swap_result = client.swap_to_cpu()
                
                if swap_result.get("status") == "success":
                    logger.info(f"GPU→RAM 스왑 완료: {model_key}")
                    # GPU 캐시 정리
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    logger.warning(f"GPU→RAM 스왑 실패: {swap_result.get('error')}")
        except Exception as e:
            logger.warning(f"GPU 스왑 실패: {model_key}, 오류: {e}")
    
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
    
    def __init__(self, client: SentenceTransformerClient, model_key: str = None, 
                 model_name: str = None, device: str = None, cache_folder: str = None):
        self.client = client
        self._model_info = None
        self._model_key = model_key  # GPU 언로드를 위한 키 저장
        self._model_name = model_name  # 재연결을 위한 모델 이름 저장
        self._device = device  # 재연결을 위한 디바이스 저장
        self._cache_folder = cache_folder  # 재연결을 위한 캐시 폴더 저장
        
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
        GPU 사용 후 자동으로 RAM으로 스왑 (서버는 유지)
        
        Args:
            sentences: 인코딩할 텍스트 리스트
            **kwargs: 추가 인자 (auto_swap=False로 스왑 비활성화 가능)
            
        Returns:
            임베딩 리스트
        """
        # auto_swap 옵션 확인 (기본값: GPU는 True, CPU는 False)
        if 'cuda' in str(self.device) or 'gpu' in str(self.device):
            auto_swap = kwargs.pop('auto_swap', True)  # GPU는 기본적으로 스왑
            # 레거시 호환성: auto_unload가 있으면 auto_swap으로 매핑
            if 'auto_unload' in kwargs:
                auto_swap = kwargs.pop('auto_unload')
        else:
            auto_swap = kwargs.pop('auto_swap', False)  # CPU는 스왑 안함
            kwargs.pop('auto_unload', None)  # 레거시 파라미터 제거
        
        # 단일 문자열을 리스트로 변환
        if isinstance(sentences, str):
            sentences = [sentences]
            return_single = True
        else:
            return_single = False
        
        try:
            # GPU 모델인 경우 필요시 GPU로 스왑
            if auto_swap and ('cuda' in str(self.device) or 'gpu' in str(self.device)):
                # 현재 모델이 CPU에 있으면 GPU로 이동
                try:
                    # Health check로 현재 device 확인
                    health_result = self.client.health_check(force=True)
                    if health_result.get("status") == "success":
                        current_device = health_result.get("result", {}).get("device", "unknown")
                        if current_device == "cpu":
                            logger.info(f"모델이 CPU에 있음, GPU로 스왑: {self._model_key}")
                            swap_result = self.client.swap_to_gpu(self._device)
                            if swap_result.get("status") != "success":
                                logger.warning(f"GPU 스왑 실패, CPU에서 계속 진행: {swap_result.get('error')}")
                except Exception as e:
                    logger.warning(f"GPU 스왑 체크 중 오류, 계속 진행: {e}")
            
            # 최대 2번 시도 (첫 시도 실패 시 재연결 후 재시도)
            max_attempts = 2
            
            for attempt in range(max_attempts):
                try:
                    # 서버에 인코딩 요청
                    response = self.client.encode_texts(sentences, **kwargs)
                    
                    if response.get("status") == "success":
                        # 성공하면 바로 반환
                        break
                    else:
                        error_msg = response.get("error", "Unknown error")
                        
                        # 모델이 로드되지 않은 경우 재연결 시도
                        if "모델이 로드되지 않음" in error_msg or "not loaded" in error_msg.lower():
                            if attempt < max_attempts - 1 and self._model_name and self._model_key:
                                logger.warning(f"모델 미로드 감지, 재연결 시도 ({attempt+1}/{max_attempts}): {self._model_key}")
                                # 매니저를 통해 새 클라이언트 획득 (모델 로딩 포함)
                                new_proxy = _manager.get_model(self._model_name, self._device, self._cache_folder)
                                # 새 클라이언트로 교체
                                self.client = new_proxy.client
                                self._model_info = new_proxy._model_info
                                logger.info(f"클라이언트 재연결 및 모델 로드 성공: {self._model_key}")
                                continue  # 다음 시도로
                        
                        # 재연결해도 안 되거나 다른 에러면 예외 발생
                        raise RuntimeError(f"텍스트 인코딩 실패: {error_msg}")
                        
                except Exception as e:
                    # 연결 자체가 실패한 경우
                    if attempt < max_attempts - 1 and self._model_name and self._model_key:
                        logger.warning(f"인코딩 실패, 재연결 시도 ({attempt+1}/{max_attempts}): {e}")
                        try:
                            # 매니저를 통해 새 클라이언트 획득 (모델 로딩 포함)
                            new_proxy = _manager.get_model(self._model_name, self._device, self._cache_folder)
                            # 새 클라이언트로 교체
                            self.client = new_proxy.client
                            self._model_info = new_proxy._model_info
                            logger.info(f"클라이언트 재연결 및 모델 로드 성공: {self._model_key}")
                            continue  # 다음 시도로
                        except Exception as reconnect_error:
                            logger.error(f"재연결 실패: {reconnect_error}")
                            raise RuntimeError(f"클라이언트 재연결 실패: {reconnect_error}") from e
                    else:
                        # 마지막 시도이거나 모델 정보가 없으면 에러 발생
                        raise RuntimeError(f"텍스트 인코딩 최종 실패: {e}") from e
            
            embeddings = response.get("result", {}).get("embeddings", [])
            
            # 단일 문자열 입력이었다면 단일 결과 반환
            # 텐서와 리스트 모두 처리 가능하도록 수정
            if return_single:
                # 텐서인 경우
                if torch.is_tensor(embeddings):
                    if embeddings.numel() > 0:  # 텐서가 비어있지 않은 경우
                        return embeddings[0] if embeddings.dim() > 1 else embeddings
                # 리스트나 배열인 경우
                elif embeddings and len(embeddings) > 0:
                    return embeddings[0]
            
            return embeddings
            
        finally:
            # GPU 사용 후 자동 스왑 (GPU 모델인 경우에만 RAM으로 이동)
            if auto_swap and self._model_key and ('cuda' in str(self.device) or 'gpu' in str(self.device)):
                try:
                    logger.debug(f"임베딩 완료, 모델을 RAM으로 스왑: {self._model_key}")
                    # 클라이언트를 통해 GPU→CPU 스왑 (서버는 유지)
                    swap_result = self.client.swap_to_cpu()
                    if swap_result.get("status") == "success":
                        logger.debug(f"GPU→RAM 스왑 성공: {self._model_key}")
                        # GPU 캐시도 정리
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()  # GPU 연산 완료 대기
                    else:
                        logger.warning(f"GPU→RAM 스왑 실패: {swap_result.get('error')}")
                except Exception as e:
                    logger.warning(f"GPU 자동 스왑 실패: {e}")
    
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