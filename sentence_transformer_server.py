#!/usr/bin/env python3
"""
Persistent SentenceTransformer Server - 완전 격리 + RAM 상주
Red Heart AI System용 독립적 SentenceTransformer 서버

주요 기능:
- SentenceTransformer 모델을 RAM에 영구 상주
- stdin/stdout JSON 통신 인터페이스
- 완전한 프로세스 격리 (signal handler inheritance 완전 회피)
- 지속적 서비스 제공 (한번 로딩 후 재사용)

통신 프로토콜:
- Request: {"action": "load|embed|health", "data": {...}}
- Response: {"status": "success|error", "result": ..., "error": "..."}
"""

import sys
import json
import logging
import traceback
import os
from typing import Dict, List, Any, Optional
import numpy as np

# 환경 변수 설정 - 오프라인 모드 활성화 (캐시된 모델만 사용)
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'  # HuggingFace Hub 오프라인 모드

def setup_logging():
    """서버 전용 로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='[SentenceTransformer-Server] %(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/sentence_transformer_server.log'),
            logging.StreamHandler(sys.stderr)  # stdout는 통신용이므로 stderr 사용
        ]
    )
    return logging.getLogger(__name__)

class SentenceTransformerServer:
    """
    Persistent SentenceTransformer Server
    - 모델을 메모리에 영구 상주
    - JSON 기반 request/response 처리
    - 완전한 프로세스 격리
    """
    
    def __init__(self):
        self.logger = setup_logging()
        self.model = None
        self.model_name = None
        self.device = None
        self.cache_folder = None
        self.is_running = True
        
        self.logger.info("SentenceTransformer Server 초기화 완료")
    
    def load_model(self, model_name: str, device: str = None, cache_folder: str = None) -> Dict[str, Any]:
        """
        SentenceTransformer 모델 로딩 및 RAM 상주
        
        Args:
            model_name: 모델 이름
            device: 디바이스 ('cpu' 또는 'cuda')
            cache_folder: 캐시 폴더 경로
            
        Returns:
            로딩 결과 딕셔너리
        """
        try:
            # device가 None인 경우 스마트 디바이스 선택
            if device is None:
                try:
                    import sys
                    # config 모듈 import를 위한 경로 추가 (os는 이미 전역 import됨)
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    if current_dir not in sys.path:
                        sys.path.append(current_dir)
                    
                    from config import get_smart_device
                    smart_device = get_smart_device(memory_required_mb=1000)  # SentenceTransformer는 약 1GB 필요
                    device = str(smart_device).split(':')[0]  # 'cuda:0' -> 'cuda', 'cpu' -> 'cpu'
                    self.logger.info(f"스마트 디바이스 선택: {device}")
                except Exception as e:
                    self.logger.warning(f"스마트 디바이스 선택 실패, CPU 사용: {e}")
                    device = "cpu"
            
            self.logger.info(f"모델 로딩 시작: {model_name} (device: {device})")
            
            # 기존 모델이 같으면 재사용
            if (self.model is not None and 
                self.model_name == model_name and 
                self.device == device):
                self.logger.info(f"기존 모델 재사용: {model_name}")
                return {
                    "status": "success",
                    "result": {
                        "model_name": model_name,
                        "device": device,
                        "reused": True
                    }
                }
            
            # 새 모델 로딩
            self.logger.info(f"새 모델 로딩 중: {model_name}")
            
            # SentenceTransformer import를 지연 로딩
            from sentence_transformers import SentenceTransformer
            
            # 캐시 폴더 설정 - 기존 Hugging Face 캐시 활용
            if cache_folder is None:
                # 기존 캐시된 모델이 있는 위치 사용
                cache_folder = os.path.join("models", "sentence_transformers")
            os.makedirs(cache_folder, exist_ok=True)
            
            # 환경 변수로 Hugging Face 캐시 디렉토리 설정
            os.environ['HUGGINGFACE_HUB_CACHE'] = cache_folder
            os.environ['TRANSFORMERS_CACHE'] = cache_folder
            
            # 모델 로딩 (프로세스 격리 환경에서 안전)
            self.model = SentenceTransformer(
                model_name,
                device=device,
                cache_folder=cache_folder
            )
            
            self.model_name = model_name
            self.device = device
            self.cache_folder = cache_folder
            
            self.logger.info(f"모델 로딩 성공: {model_name}")
            
            return {
                "status": "success", 
                "result": {
                    "model_name": model_name,
                    "device": device,
                    "reused": False,
                    "model_max_seq_length": getattr(self.model, 'max_seq_length', None)
                }
            }
            
        except Exception as e:
            error_msg = f"모델 로딩 실패: {model_name} - {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(f"트레이스백: {traceback.format_exc()}")
            return {
                "status": "error",
                "error": error_msg,
                "traceback": traceback.format_exc()
            }
    
    def encode_texts(self, texts: List[str], **kwargs) -> Dict[str, Any]:
        """
        텍스트 리스트를 임베딩으로 변환
        
        Args:
            texts: 인코딩할 텍스트 리스트
            **kwargs: SentenceTransformer.encode() 추가 인자
            
        Returns:
            인코딩 결과 딕셔너리
        """
        try:
            if self.model is None:
                return {
                    "status": "error",
                    "error": "모델이 로드되지 않음. 먼저 load action을 실행하세요."
                }
            
            self.logger.info(f"텍스트 인코딩 시작: {len(texts)}개 텍스트")
            
            # 임베딩 생성
            embeddings = self.model.encode(texts, **kwargs)
            
            # numpy array를 list로 변환 (JSON 직렬화용)
            if isinstance(embeddings, np.ndarray):
                embeddings_list = embeddings.tolist()
            else:
                embeddings_list = embeddings
            
            self.logger.info(f"텍스트 인코딩 완료: {len(texts)}개 → {len(embeddings_list)}개 임베딩")
            
            return {
                "status": "success",
                "result": {
                    "embeddings": embeddings_list,
                    "num_texts": len(texts),
                    "embedding_dim": len(embeddings_list[0]) if embeddings_list else 0
                }
            }
            
        except Exception as e:
            error_msg = f"텍스트 인코딩 실패: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(f"트레이스백: {traceback.format_exc()}")
            return {
                "status": "error",
                "error": error_msg,
                "traceback": traceback.format_exc()
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        서버 상태 및 건강도 체크
        
        Returns:
            서버 상태 딕셔너리
        """
        try:
            import psutil
            import torch
            
            status = {
                "status": "success",
                "result": {
                    "server_running": self.is_running,
                    "model_loaded": self.model is not None,
                    "model_name": self.model_name,
                    "device": self.device,
                    "process_id": os.getpid(),
                    "memory_usage": psutil.Process().memory_info().rss / (1024 * 1024),  # MB
                    "cpu_percent": psutil.Process().cpu_percent()
                }
            }
            
            # 모델이 로드되어 있으면 추가 정보 포함
            if self.model is not None:
                try:
                    # 임베딩 차원 정보
                    status["result"]["embedding_dimension"] = self.model.get_sentence_embedding_dimension()
                except:
                    # fallback으로 다른 속성들 시도
                    try:
                        if hasattr(self.model, 'model') and hasattr(self.model.model, 'config'):
                            status["result"]["hidden_size"] = self.model.model.config.hidden_size
                    except:
                        pass
                
                # 최대 시퀀스 길이
                try:
                    if hasattr(self.model, 'max_seq_length'):
                        status["result"]["model_max_seq_length"] = self.model.max_seq_length
                except:
                    pass
            
            # GPU 정보 추가
            if torch.cuda.is_available():
                status["result"]["gpu_available"] = True
                status["result"]["gpu_allocated"] = torch.cuda.memory_allocated() / (1024 * 1024)
                status["result"]["gpu_cached"] = torch.cuda.memory_reserved() / (1024 * 1024)
            else:
                status["result"]["gpu_available"] = False
            
            return status
            
        except Exception as e:
            error_msg = f"상태 체크 실패: {str(e)}"
            self.logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "traceback": traceback.format_exc()
            }
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        요청 처리 메인 함수
        
        Args:
            request: JSON 요청 딕셔너리
            
        Returns:
            응답 딕셔너리
        """
        try:
            action = request.get("action")
            data = request.get("data", {})
            
            if action == "load":
                model_name = data.get("model_name")
                device = data.get("device")
                cache_folder = data.get("cache_folder")
                
                if not model_name:
                    return {
                        "status": "error",
                        "error": "model_name이 필요합니다"
                    }
                
                return self.load_model(model_name, device, cache_folder)
            
            elif action == "embed":
                texts = data.get("texts")
                kwargs = data.get("kwargs", {})
                
                if not texts or not isinstance(texts, list):
                    return {
                        "status": "error",
                        "error": "texts 리스트가 필요합니다"
                    }
                
                return self.encode_texts(texts, **kwargs)
            
            elif action == "health":
                return self.get_health_status()
            
            elif action == "shutdown":
                self.logger.info("서버 종료 요청 수신")
                self.is_running = False
                return {
                    "status": "success",
                    "result": {"message": "서버 종료 중..."}
                }
            
            else:
                return {
                    "status": "error",
                    "error": f"알 수 없는 action: {action}. 지원되는 action: load, embed, health, shutdown"
                }
                
        except Exception as e:
            error_msg = f"요청 처리 실패: {str(e)}"
            self.logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "traceback": traceback.format_exc()
            }
    
    def run(self):
        """
        서버 메인 루프 - stdin에서 JSON 요청을 받아 stdout으로 응답
        """
        self.logger.info("SentenceTransformer Server 시작 - stdin/stdout JSON 통신 대기")
        
        try:
            while self.is_running:
                try:
                    # stdin에서 JSON 요청 읽기
                    line = sys.stdin.readline()
                    if not line:  # EOF
                        break
                    
                    line = line.strip()
                    if not line:
                        continue
                    
                    # JSON 파싱
                    try:
                        request = json.loads(line)
                    except json.JSONDecodeError as e:
                        response = {
                            "status": "error",
                            "error": f"JSON 파싱 오류: {str(e)}"
                        }
                        print(json.dumps(response, ensure_ascii=False))
                        sys.stdout.flush()
                        continue
                    
                    # 요청 처리
                    response = self.process_request(request)
                    
                    # 응답 출력
                    print(json.dumps(response, ensure_ascii=False))
                    sys.stdout.flush()
                    
                except KeyboardInterrupt:
                    self.logger.info("키보드 인터럽트 수신 - 서버 종료")
                    break
                except Exception as e:
                    self.logger.error(f"요청 처리 중 오류: {str(e)}")
                    error_response = {
                        "status": "error",
                        "error": f"서버 내부 오류: {str(e)}"
                    }
                    print(json.dumps(error_response, ensure_ascii=False))
                    sys.stdout.flush()
        
        except Exception as e:
            self.logger.error(f"서버 실행 중 치명적 오류: {str(e)}")
            self.logger.error(f"트레이스백: {traceback.format_exc()}")
        
        finally:
            self.logger.info("SentenceTransformer Server 종료")

def main():
    """메인 함수"""
    
    # 로그 디렉토리 생성
    os.makedirs('logs', exist_ok=True)
    
    # 서버 시작
    server = SentenceTransformerServer()
    server.run()

if __name__ == "__main__":
    main()