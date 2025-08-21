"""
SentenceTransformer Client - subprocess 서버와의 IPC 통신
Red Heart AI System용 클라이언트 통신 모듈

주요 기능:
- subprocess.Popen으로 sentence_transformer_server.py와 통신
- JSON 기반 request/response 처리
- Process lifecycle 관리 (startup/shutdown/restart)
- Health check 및 자동 복구
- Thread-safe 통신
"""

import subprocess
import json
import threading
import time
import logging
import os
import sys
from typing import Dict, List, Any, Optional
import queue

logger = logging.getLogger(__name__)

class SentenceTransformerClient:
    """
    SentenceTransformer Server와 통신하는 클라이언트
    - subprocess.Popen으로 서버 프로세스 관리
    - JSON 기반 IPC 통신
    - 자동 재연결 및 복구
    """
    
    def __init__(self, 
                 server_script_path: str = "sentence_transformer_server.py",
                 python_executable: str = None,
                 startup_timeout: float = 30.0,
                 request_timeout: float = 60.0):
        """
        클라이언트 초기화
        
        Args:
            server_script_path: 서버 스크립트 경로
            python_executable: 파이썬 실행 파일 (기본: sys.executable)
            startup_timeout: 서버 시작 타임아웃 (초)
            request_timeout: 요청 타임아웃 (초)
        """
        self.server_script_path = server_script_path
        
        # venv 환경의 python 실행파일 자동 감지
        if python_executable:
            self.python_executable = python_executable
        else:
            self.python_executable = self._detect_venv_python()
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        
        # 프로세스 관리
        self.process: Optional[subprocess.Popen] = None
        self.is_connected = False
        self.connection_lock = threading.RLock()
        
        # 통신 관리
        self.request_lock = threading.RLock()
        self.last_health_check = 0
        self.health_check_interval = 10.0  # 10초마다 health check
        
        logger.info(f"SentenceTransformer Client 초기화: {server_script_path}")
        logger.info(f"Python 실행파일: {self.python_executable}")
    
    def _detect_venv_python(self) -> str:
        """
        venv 환경의 python 실행파일 자동 감지
        
        Returns:
            python 실행파일 경로
        """
        # 1. 현재 sys.executable이 venv 환경인지 확인
        current_python = sys.executable
        
        # venv 환경 표시자들 확인
        if any(indicator in current_python for indicator in ['red_heart_env', 'venv', 'env']):
            logger.info(f"현재 Python이 venv 환경: {current_python}")
            return current_python
        
        # 2. VIRTUAL_ENV 환경변수 확인
        virtual_env = os.environ.get('VIRTUAL_ENV')
        if virtual_env:
            venv_python = os.path.join(virtual_env, 'bin', 'python3')
            if os.path.exists(venv_python):
                logger.info(f"VIRTUAL_ENV에서 python 발견: {venv_python}")
                return venv_python
        
        # 3. red_heart_env 폴더 직접 확인
        script_dir = os.path.dirname(os.path.abspath(self.server_script_path))
        red_heart_python = os.path.join(script_dir, 'red_heart_env', 'bin', 'python3')
        
        if os.path.exists(red_heart_python):
            logger.info(f"red_heart_env python 발견: {red_heart_python}")
            return red_heart_python
        
        # 4. 상위 디렉토리에서 red_heart_env 찾기
        parent_dir = os.path.dirname(script_dir)
        parent_red_heart_python = os.path.join(parent_dir, 'red_heart_env', 'bin', 'python3')
        
        if os.path.exists(parent_red_heart_python):
            logger.info(f"상위 디렉토리에서 red_heart_env python 발견: {parent_red_heart_python}")
            return parent_red_heart_python
        
        # 5. Fallback: 현재 sys.executable 사용
        logger.warning(f"venv python을 찾을 수 없음. fallback 사용: {current_python}")
        return current_python
    
    def start_server(self) -> bool:
        """
        서버 프로세스 시작
        
        Returns:
            시작 성공 여부
        """
        with self.connection_lock:
            if self.process and self.process.poll() is None:
                logger.info("서버가 이미 실행 중입니다")
                return True
            
            try:
                logger.info(f"SentenceTransformer 서버 시작: {self.server_script_path}")
                
                # 환경변수 설정 (venv 환경 정보 전달)
                env = os.environ.copy()
                
                # PYTHONPATH 설정으로 패키지 찾기 보장
                current_dir = os.path.dirname(os.path.abspath(self.server_script_path))
                if 'PYTHONPATH' in env:
                    env['PYTHONPATH'] = f"{current_dir}:{env['PYTHONPATH']}"
                else:
                    env['PYTHONPATH'] = current_dir
                
                # venv 환경 감지 시 추가 설정
                if 'red_heart_env' in self.python_executable:
                    venv_path = os.path.dirname(os.path.dirname(self.python_executable))  # .../red_heart_env
                    env['VIRTUAL_ENV'] = venv_path
                    # venv의 site-packages 경로 추가
                    site_packages = os.path.join(venv_path, 'lib', 'python3.12', 'site-packages')
                    if os.path.exists(site_packages):
                        env['PYTHONPATH'] = f"{site_packages}:{env['PYTHONPATH']}"
                
                logger.info(f"PYTHONPATH: {env.get('PYTHONPATH', 'None')}")
                logger.info(f"VIRTUAL_ENV: {env.get('VIRTUAL_ENV', 'None')}")
                
                # 서버 프로세스 시작 (환경변수 전달)
                self.process = subprocess.Popen(
                    [self.python_executable, self.server_script_path],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,  # 라인 버퍼링
                    universal_newlines=True,
                    cwd=os.path.dirname(os.path.abspath(self.server_script_path)) or ".",
                    env=env  # 환경변수 전달
                )
                
                # 서버 시작 대기 및 health check
                start_time = time.time()
                while time.time() - start_time < self.startup_timeout:
                    if self.process.poll() is not None:
                        # 프로세스가 종료됨
                        stderr_output = self.process.stderr.read() if self.process.stderr else ""
                        raise RuntimeError(f"서버 프로세스가 예기치 않게 종료됨: {stderr_output}")
                    
                    # Health check 시도
                    try:
                        health_response = self._send_request_raw({"action": "health"}, timeout=30.0)
                        if health_response and health_response.get("status") == "success":
                            self.is_connected = True
                            logger.info("서버 연결 성공")
                            return True
                    except Exception as e:
                        logger.debug(f"Health check 실패 (재시도 중): {e}")
                    
                    time.sleep(0.5)
                
                raise TimeoutError(f"서버 시작 타임아웃 ({self.startup_timeout}초)")
                
            except Exception as e:
                logger.error(f"서버 시작 실패: {e}")
                self._cleanup_process()
                return False
    
    def stop_server(self, graceful_timeout: float = 5.0) -> bool:
        """
        서버 프로세스 종료
        
        Args:
            graceful_timeout: graceful shutdown 타임아웃 (초)
            
        Returns:
            종료 성공 여부
        """
        with self.connection_lock:
            if not self.process:
                return True
            
            try:
                logger.info("서버 종료 중...")
                
                # Graceful shutdown 시도
                try:
                    self._send_request_raw({"action": "shutdown"}, timeout=graceful_timeout)
                except Exception:
                    logger.debug("Graceful shutdown 요청 실패")
                
                # 프로세스 종료 대기
                try:
                    self.process.wait(timeout=graceful_timeout)
                    logger.info("서버 정상 종료")
                except subprocess.TimeoutExpired:
                    logger.warning("Graceful shutdown 타임아웃 - 강제 종료")
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        logger.warning("강제 종료 실패 - kill 시도")
                        self.process.kill()
                
                return True
                
            except Exception as e:
                logger.error(f"서버 종료 중 오류: {e}")
                return False
            finally:
                self._cleanup_process()
    
    def _cleanup_process(self):
        """프로세스 리소스 정리"""
        if self.process:
            if self.process.stdin:
                self.process.stdin.close()
            if self.process.stdout:
                self.process.stdout.close()
            if self.process.stderr:
                self.process.stderr.close()
            self.process = None
        
        self.is_connected = False
    
    def _send_request_raw(self, request: Dict[str, Any], timeout: float = None) -> Optional[Dict[str, Any]]:
        """
        원시 요청 전송 (재연결 로직 없음)
        
        Args:
            request: 요청 딕셔너리
            timeout: 타임아웃 (초)
            
        Returns:
            응답 딕셔너리 또는 None
        """
        if not self.process or self.process.poll() is not None:
            raise RuntimeError("서버 프로세스가 실행되지 않음")
        
        timeout = timeout or self.request_timeout
        
        try:
            # 요청 전송
            request_json = json.dumps(request, ensure_ascii=False) + '\n'
            self.process.stdin.write(request_json)
            self.process.stdin.flush()
            
            # 응답 수신 (타임아웃 처리)
            start_time = time.time()
            while time.time() - start_time < timeout:
                if self.process.poll() is not None:
                    raise RuntimeError("서버 프로세스가 예기치 않게 종료됨")
                
                # non-blocking read 시뮬레이션
                try:
                    # subprocess는 timeout을 직접 지원하지 않으므로 polling 사용
                    import select
                    ready, _, _ = select.select([self.process.stdout], [], [], 0.1)
                    if ready:
                        response_line = self.process.stdout.readline()
                        if response_line:
                            return json.loads(response_line.strip())
                except Exception as e:
                    # select가 지원되지 않는 환경에서는 블로킹 read 사용
                    response_line = self.process.stdout.readline()
                    if response_line:
                        return json.loads(response_line.strip())
                    break
            
            raise TimeoutError(f"응답 타임아웃 ({timeout}초)")
            
        except Exception as e:
            logger.error(f"요청 전송 실패: {e}")
            raise
    
    def send_request(self, request: Dict[str, Any], auto_reconnect: bool = True) -> Dict[str, Any]:
        """
        요청 전송 (재연결 없이 실패 시 즉시 종료)
        
        Args:
            request: 요청 딕셔너리
            auto_reconnect: (더 이상 사용되지 않음, 호환성을 위해 유지)
            
        Returns:
            응답 딕셔너리
        """
        with self.request_lock:
            # 재연결 로직 제거 - 실패 시 즉시 에러 발생
            try:
                # 연결 상태 확인
                if not self.is_connected:
                    if not self.start_server():
                        raise RuntimeError("서버 시작 실패")
                
                # 요청 전송
                response = self._send_request_raw(request)
                if response:
                    return response
                else:
                    raise RuntimeError("응답 없음")
            
            except Exception as e:
                logger.error(f"요청 전송 실패: {e}")
                # 재연결 시도 없이 즉시 에러 발생
                raise RuntimeError(f"요청 실패 (재연결 없음): {e}") from e
    
    def load_model(self, model_name: str, device: str = None, cache_folder: str = None) -> Dict[str, Any]:
        """
        모델 로딩 요청
        
        Args:
            model_name: 모델 이름
            device: 디바이스
            cache_folder: 캐시 폴더
            
        Returns:
            로딩 결과
        """
        # device가 None인 경우 스마트 디바이스 선택
        if device is None:
            try:
                from config import get_smart_device
                smart_device = get_smart_device(memory_required_mb=1000)  # SentenceTransformer는 약 1GB 필요
                device = str(smart_device).split(':')[0]  # 'cuda:0' -> 'cuda', 'cpu' -> 'cpu'
                logger.info(f"스마트 디바이스 선택: {device}")
            except Exception as e:
                logger.warning(f"스마트 디바이스 선택 실패, CPU 사용: {e}")
                device = "cpu"
        
        request = {
            "action": "load",
            "data": {
                "model_name": model_name,
                "device": device,
                "cache_folder": cache_folder
            }
        }
        
        logger.info(f"모델 로딩 요청: {model_name} (device: {device})")
        response = self.send_request(request)
        
        if response.get("status") == "success":
            logger.info(f"모델 로딩 성공: {model_name}")
        else:
            logger.error(f"모델 로딩 실패: {response.get('error', 'Unknown error')}")
        
        return response
    
    def encode_texts(self, texts: List[str], **kwargs) -> Dict[str, Any]:
        """
        텍스트 인코딩 요청
        
        Args:
            texts: 인코딩할 텍스트 리스트
            **kwargs: 추가 인자
            
        Returns:
            인코딩 결과
        """
        request = {
            "action": "embed",
            "data": {
                "texts": texts,
                "kwargs": kwargs
            }
        }
        
        logger.debug(f"텍스트 인코딩 요청: {len(texts)}개 텍스트")
        response = self.send_request(request)
        
        if response.get("status") == "success":
            logger.debug(f"텍스트 인코딩 성공: {len(texts)}개")
        else:
            logger.error(f"텍스트 인코딩 실패: {response.get('error', 'Unknown error')}")
        
        return response
    
    def health_check(self, force: bool = False) -> Dict[str, Any]:
        """
        서버 상태 확인
        
        Args:
            force: 강제 체크 (캐시 무시)
            
        Returns:
            상태 정보
        """
        current_time = time.time()
        
        # 캐시된 결과 사용 (너무 자주 체크하지 않음)
        if not force and (current_time - self.last_health_check) < self.health_check_interval:
            return {"status": "success", "result": {"cached": True}}
        
        request = {"action": "health"}
        
        try:
            response = self.send_request(request, auto_reconnect=True)
            self.last_health_check = current_time
            return response
        except Exception as e:
            logger.error(f"Health check 실패: {e}")
            return {
                "status": "error", 
                "error": f"Health check 실패: {e}"
            }
    
    def __enter__(self):
        """Context manager 진입"""
        if not self.start_server():
            raise RuntimeError("서버 시작 실패")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.stop_server()
    
    def __del__(self):
        """소멸자 - 프로세스 정리"""
        try:
            self.stop_server()
        except Exception:
            pass