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
import torch

# 환경 변수 설정 - 오프라인 모드 활성화 (캐시된 모델만 사용)
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'  # HuggingFace Hub 오프라인 모드
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # 경고 메시지 최소화
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'  # 텔레메트리 비활성화

# HuggingFace 캐시 디렉토리 명시적 설정
HF_HOME = os.path.expanduser('~/.cache/huggingface')
os.environ['HF_HOME'] = HF_HOME
os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(HF_HOME, 'hub')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(HF_HOME, 'hub')

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
            # cache_folder가 전달된 경우 사용, 아니면 기본 경로 사용
            if cache_folder:
                hf_cache_dir = cache_folder
            else:
                hf_cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
            
            # 캐시된 모델 경로 확인
            model_path = None
            # HuggingFace Hub 캐시 형식으로 확인
            hub_model_name = f"models--{model_name.replace('/', '--')}"
            hub_path = os.path.join(hf_cache_dir, hub_model_name)
            
            self.logger.info(f"HuggingFace 캐시 디렉토리: {hf_cache_dir}")
            self.logger.info(f"모델 캐시 경로 확인 중: {hub_path}")
            
            if os.path.exists(hub_path):
                # 스냅샷 찾기
                snapshots_dir = os.path.join(hub_path, 'snapshots')
                if os.path.exists(snapshots_dir):
                    snapshot_dirs = [d for d in os.listdir(snapshots_dir) 
                                   if os.path.isdir(os.path.join(snapshots_dir, d))]
                    if snapshot_dirs:
                        # 가장 최근 스냅샷 사용
                        model_path = os.path.join(snapshots_dir, snapshot_dirs[0])
                        self.logger.info(f"캐시된 모델 발견: {model_path}")
                        
                        # 모델 파일 존재 확인
                        model_files = os.listdir(model_path)
                        self.logger.info(f"모델 파일 목록: {model_files}")
                else:
                    self.logger.warning(f"스냅샷 디렉토리 없음: {snapshots_dir}")
            else:
                self.logger.warning(f"모델 캐시 경로 없음: {hub_path}")
            
            # 모델 로딩 - 캐시된 경로에서 직접 로드
            if model_path and os.path.exists(model_path):
                try:
                    self.logger.info(f"캐시된 모델 직접 로드 중: {model_path}")
                    
                    # 방법 1: 캐시 경로에서 직접 로드 (local_files_only 추가)
                    self.model = SentenceTransformer(
                        model_path,
                        device=device,
                        local_files_only=True
                    )
                    
                    self.logger.info(f"캐시된 모델 로드 성공: {model_path}")
                    
                except Exception as e:
                    self.logger.warning(f"직접 로드 실패, 대체 방법 시도: {e}")
                    
                    # 방법 2: transformers 라이브러리로 수동 로드
                    from transformers import AutoModel, AutoTokenizer
                    from sentence_transformers import models
                    
                    # 토크나이저와 모델을 개별적으로 로드
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_path,
                        local_files_only=True
                    )
                    
                    transformer = AutoModel.from_pretrained(
                        model_path,
                        local_files_only=True
                    )
                    
                    # SentenceTransformer 모듈 수동 구성
                    # 1_Pooling 디렉토리에서 설정 읽기
                    pooling_config_path = os.path.join(model_path, '1_Pooling', 'config.json')
                    if os.path.exists(pooling_config_path):
                        import json
                        with open(pooling_config_path, 'r') as f:
                            pooling_config = json.load(f)
                        
                        word_embedding_model = models.Transformer(
                            model_name_or_path=model_path,
                            max_seq_length=pooling_config.get('max_seq_length', 512)
                        )
                        
                        pooling_model = models.Pooling(
                            word_embedding_dimension=pooling_config.get('word_embedding_dimension', 384),
                            pooling_mode_cls_token=pooling_config.get('pooling_mode_cls_token', False),
                            pooling_mode_mean_tokens=pooling_config.get('pooling_mode_mean_tokens', True),
                            pooling_mode_max_tokens=pooling_config.get('pooling_mode_max_tokens', False),
                            pooling_mode_mean_sqrt_len_tokens=pooling_config.get('pooling_mode_mean_sqrt_len_tokens', False)
                        )
                    else:
                        # 기본 설정 사용
                        word_embedding_model = models.Transformer(model_name_or_path=model_path)
                        pooling_model = models.Pooling(
                            word_embedding_model.get_word_embedding_dimension(),
                            pooling_mode_mean_tokens=True,
                            pooling_mode_cls_token=False,
                            pooling_mode_max_tokens=False
                        )
                    
                    # 정규화 레이어 추가 (all-MiniLM-L6-v2는 정규화 사용)
                    normalize_layer = models.Normalize()
                    
                    self.model = SentenceTransformer(
                        modules=[word_embedding_model, pooling_model, normalize_layer],
                        device=device
                    )
                    
                    self.logger.info(f"대체 방법으로 모델 로드 성공")
            else:
                # 캐시된 모델이 없는 경우 에러
                error_msg = f"캐시된 모델을 찾을 수 없습니다: {model_name}\n"
                error_msg += f"확인된 경로: {hub_path}\n"
                error_msg += "모델을 먼저 다운로드하거나 캐시 경로를 확인하세요."
                raise FileNotFoundError(error_msg)
            
            self.model_name = model_name
            self.device = device
            self.cache_folder = hf_cache_dir
            
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
            
            # 임베딩 생성 - gradient 추적 비활성화
            with torch.no_grad():
                embeddings = self.model.encode(texts, **kwargs)
            
            # numpy array를 list로 변환 (JSON 직렬화용)
            if isinstance(embeddings, np.ndarray):
                embeddings_list = embeddings.tolist()
            else:
                embeddings_list = embeddings
            
            # GPU 메모리 정리 (메모리 누수 방지)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
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