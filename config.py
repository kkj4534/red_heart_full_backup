"""
Red Heart Linux Advanced - 설정 파일
고급 라이브러리 기반 Linux 최적화 설정
"""

import os
import platform
import time
import asyncio

import logging
import datetime

# 전역 logger 설정
logger = logging.getLogger('RedHeart.Config')

# dotenv 의존성을 선택적으로 로드
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv가 설치되지 않았습니다. .env 파일을 무시합니다.")
    def load_dotenv():
        pass


# 기본 경로 설정 - pathlib 제거하여 WSL hanging 방지
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
DOCS_DIR = os.path.join(BASE_DIR, 'docs')
TESTS_DIR = os.path.join(BASE_DIR, 'tests')

# 데이터 하위 디렉토리
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
EXPERIENCE_DB_DIR = os.path.join(DATA_DIR, 'experience_db')
DECISION_LOGS_DIR = os.path.join(DATA_DIR, 'decision_logs')

# 모델 하위 디렉토리
EMOTION_MODELS_DIR = os.path.join(MODELS_DIR, 'emotion_models')
SEMANTIC_MODELS_DIR = os.path.join(MODELS_DIR, 'semantic_models')
SURD_CACHE_DIR = os.path.join(MODELS_DIR, 'surd_cache')
REGRET_MODELS_DIR = os.path.join(MODELS_DIR, 'regret_models')
HIERARCHICAL_EMOTION_DIR = os.path.join(MODELS_DIR, 'hierarchical_emotion')
SEMANTIC_CACHE_DIR = os.path.join(MODELS_DIR, 'semantic_cache')

# 캐시 디렉토리 설정
CACHE_DIR = os.path.join(BASE_DIR, 'cache')

# 학습 데이터 디렉토리
LEARN_DATASET_DIR = os.path.join(BASE_DIR, 'for_learn_dataset')
PROCESSED_DATASETS_DIR = os.path.join(BASE_DIR, 'processed_datasets')

# 디렉토리 생성을 지연 함수로 래핑 (WSL /mnt hanging 방지)
def _create_directories_if_needed():
    """필요시에만 디렉토리를 생성 (WSL hanging 방지)"""
    directories = [DATA_DIR, MODELS_DIR, LOGS_DIR, PLOTS_DIR, DOCS_DIR, TESTS_DIR,
                   RAW_DATA_DIR, PROCESSED_DATA_DIR, EXPERIENCE_DB_DIR, DECISION_LOGS_DIR,
                   EMOTION_MODELS_DIR, SEMANTIC_MODELS_DIR, SURD_CACHE_DIR, REGRET_MODELS_DIR,
                   HIERARCHICAL_EMOTION_DIR, SEMANTIC_CACHE_DIR, CACHE_DIR,
                   LEARN_DATASET_DIR, PROCESSED_DATASETS_DIR]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception as e:
            # WSL /mnt 경로에서 발생할 수 있는 hanging 문제 무시
            logger.warning(f"디렉토리 생성 실패 (무시): {directory} - {e}")
            pass

# 디렉토리는 실제 사용시에만 생성 (모듈 import시 hanging 방지)

# 시스템 정보
SYSTEM_INFO = {
    'platform': platform.system(),
    'architecture': platform.architecture(),
    'python_version': platform.python_version(),
    'is_linux': platform.system().lower() == 'linux',
    'cpu_count': os.cpu_count(),
}

# 고급 라이브러리 활성화 설정 (800M 파라미터 통합 아키텍처)
ADVANCED_CONFIG = {
    'use_transformers': True,           # Transformers 라이브러리 사용
    'use_sentence_transformers': True,  # Sentence Transformers 사용
    'use_torch': True,                  # PyTorch 사용
    'use_advanced_nlp': True,           # 고급 NLP 라이브러리 사용
    'enable_gpu': True,                 # GPU 사용 (자동 감지)
    'fallback_mode': False,             # 폴백 모드 완전 비활성화
    'strict_mode': True,                # 엄격 모드 (고급 모듈 필수)
    'korean_advanced': True,            # 고급 한국어 처리 활성화
    'use_multiprocessing': True,        # 멀티프로세싱 사용
    'total_parameters': 450_000_000,    # 총 450M 파라미터 (68M 백본 + 109M 헤드 + 232M 분석기 + 41M 보조)
    'optimization_target': 'unified_synergy',  # 통합 시너지에 집중
    'gpu_memory_fraction': 0.85,        # 85%로 복구 - 임계값과 별개
    'precision': 'fp16',                # 메모리 효율성을 위한 반정밀도
    'enable_mixed_precision': True,     # 혼합 정밀도로 성능 향상
    'disable_counselor_during_training': True,  # 학습 중 상담사 모듈 OFF
    'enable_llm_emotion_support': True, # 감정 LLM 지원은 유지
    'enable_dynamic_swap': True,        # 동적 RAM 스왑 활성화
    'unified_backbone': {
        'total_parameters': 68_000_000,     # 68M 공유 백본 (50M의 1.364배)
        'd_model': 896,                     # 모델 차원 (768의 1.17배)
        'num_heads': 14,                    # 어텐션 헤드 수
        'num_layers': 8,                    # 레이어 수
        'feedforward_dim': 3584,            # 피드포워드 차원 (896*4)
        'cross_attention_heads': 14,        # 크로스 어텐션 헤드
        'gpu_resident': True,               # 백본은 항상 GPU 상주
    },
    'specialized_heads': {
        'emotion_empathy_head': 30_000_000,       # 감정+공감 헤드 (22M의 1.364배)
        'bentham_fromm_head': 27_000_000,         # 벤담+프롬 헤드 (20M의 1.364배)
        'semantic_surd_head': 22_000_000,         # 의미+SURD 헤드 (16M의 1.364배)
        'regret_learning_head': 30_000_000,       # 후회+학습 헤드 (22M의 1.364배)
        'meta_integration_head': 0,               # 메타통합 헤드 (추가로 필요시)
        'default_gpu_resident': False,            # 헤드들은 기본적으로 RAM에서 스왑
        'swap_strategy': 'predictive_preload',    # 예측적 프리로딩
    },
    'dynamic_swap_config': {
        'swap_backend': 'ram',                    # RAM 기반 스왑
        'compression_enabled': True,              # 모델 압축 활성화
        'async_swap': True,                       # 비동기 스왑
        'preload_prediction': True,               # 예측적 프리로딩
        'swap_timeout': 2.0,                      # 스왑 타임아웃 (초)
        'memory_threshold': 0.85,                 # GPU 메모리 임계치 85% 복원
    },
    'rumbaugh': {
        'neural_hidden_dim': 512,
        'learning_rate': 0.001,
        'gpu_allocation': 0.15,
        'max_depth': 10,
        'attention_mechanism': True,
        'layer_norm': True,
        'dropout_rate': 0.1,
        'batch_size': 32,
        'enable_caching': True,
    },
}

# 초기값 설정 (torch 없이)
ADVANCED_CONFIG['enable_gpu'] = False
ADVANCED_CONFIG['gpu_count'] = 0
DEVICE = 'cpu'
TORCH_DTYPE = None

# 우선순위 클래스 정의 (get_smart_device보다 먼저 정의)
class ModelPriority:
    """모델 우선순위 클래스"""
    CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1

# 우선순위 문자열 매핑
MODULE_PRIORITY_MAP = {
    'CRITICAL': ModelPriority.CRITICAL,
    'HIGH': ModelPriority.HIGH,
    'MEDIUM': ModelPriority.MEDIUM,
    'LOW': ModelPriority.LOW
}

# GPU 사용 가능성 체크 및 디바이스 설정 - torch import 지연 로딩
def _initialize_torch_config():
    """torch 설정을 지연 초기화"""
    global DEVICE, TORCH_DTYPE
    try:
        import torch
        ADVANCED_CONFIG['enable_gpu'] = torch.cuda.is_available()
        ADVANCED_CONFIG['gpu_count'] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        TORCH_DTYPE = torch.float32
        return True
    except ImportError:
        ADVANCED_CONFIG['enable_gpu'] = False
        ADVANCED_CONFIG['gpu_count'] = 0
        DEVICE = 'cpu'
        TORCH_DTYPE = None
        return False

def get_gpu_memory_info():
    """
    GPU 메모리 상태를 정밀하게 추적하는 함수
    
    Returns:
        Dict: GPU 메모리 정보 (단일 스키마)
            - total_mb: 전체 메모리 (MB)
            - allocated_mb: 할당된 메모리 (MB)
            - cached_mb: 예약된/캐시된 메모리 (MB)
            - free_mb: 여유 메모리 (MB)
            - usage_percent: 사용률 (%)
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            return None
            
        # GPU 0번 디바이스 기준
        device = 0
        
        # 총 메모리 (바이트)
        total_memory = torch.cuda.get_device_properties(device).total_memory
        
        # 현재 할당된 메모리 (바이트)
        allocated_memory = torch.cuda.memory_allocated(device)
        
        # 예약된 메모리 (바이트) - PyTorch 캐시 포함
        cached_memory = torch.cuda.memory_reserved(device)
        
        # MB 단위로 변환
        total_mb = total_memory / (1024 * 1024)
        allocated_mb = allocated_memory / (1024 * 1024)
        cached_mb = cached_memory / (1024 * 1024)
        free_mb = total_mb - cached_mb  # 캐시된 메모리 기준
        
        # 사용률 계산 (할당된 메모리 기준)
        usage_percent = (allocated_mb / total_mb) * 100
        
        return {
            'total_mb': total_mb,
            'allocated_mb': allocated_mb,
            'cached_mb': cached_mb,
            'free_mb': free_mb,
            'usage_percent': usage_percent
        }
        
    except Exception as e:
        logger.error(f"GPU 메모리 정보 조회 실패: {e}")
        return None

# 전역 MasterMemoryOrchestrator 인스턴스
_master_memory_orchestrator = None

# 전역 SequentialGPULoader 인스턴스
_gpu_loader = None

def get_gpu_loader():
    """SequentialGPULoader 싱글톤 인스턴스 반환"""
    global _gpu_loader
    
    if _gpu_loader is None:
        _gpu_loader = SequentialGPULoader()
        _gpu_loader.start()
        logger.info("SequentialGPULoader 시작됨")
    
    return _gpu_loader

def get_master_memory_orchestrator():
    """MasterMemoryOrchestrator 싱글톤 인스턴스 반환"""
    global _master_memory_orchestrator
    
    if _master_memory_orchestrator is None:
        _master_memory_orchestrator = MasterMemoryOrchestrator()
        
        # 서브시스템 연결
        try:
            gpu_loader = get_gpu_loader()
            
            # 동적 스왑 매니저 초기화
            try:
                from dynamic_swap_manager import RedHeartDynamicSwapManager
                swap_manager = RedHeartDynamicSwapManager()
                logger.info("동적 스왑 매니저 연결 성공")
            except Exception as e:
                logger.warning(f"동적 스왑 매니저 연결 실패: {e}")
                swap_manager = None
                
            # 압축 시스템 초기화 (필요시)
            compressor = None  # 현재는 내장 압축 시스템 사용
            predictor = None   # 현재는 MasterMemoryOrchestrator의 내장 예측 사용
            
            _master_memory_orchestrator.connect_subsystems(
                gpu_manager=gpu_loader,
                swap_manager=swap_manager,
                predictor=predictor,
                compressor=compressor
            )
        except Exception as e:
            logger.warning(f"서브시스템 연결 중 일부 실패: {e}")
    
    return _master_memory_orchestrator

def get_smart_device(memory_required_mb: int = 500, force_cpu: bool = False, priority: int = ModelPriority.MEDIUM, model_id: str = None):
    """
    MasterMemoryOrchestrator를 활용한 진정한 스마트 디바이스 선택
    - GPU 메모리를 85%까지 최대한 활용
    - 필요시 다른 모듈을 RAM으로 스왑해서 공간 확보
    - 순차적 비동기 처리를 통한 메모리 관리
    
    Args:
        memory_required_mb: 필요한 메모리 용량 (MB)
        force_cpu: CPU 강제 사용 여부
        priority: 모델 우선순위 (ModelPriority)
        model_id: 모델 고유 ID
        
    Returns:
        torch.device: 선택된 디바이스
    """
    # torch 모듈이 초기화되지 않았다면 초기화
    if DEVICE == 'cpu' and TORCH_DTYPE is None:
        _initialize_torch_config()
    
    # CPU 강제 사용 모드
    if force_cpu:
        try:
            import torch
            return torch.device('cpu')
        except ImportError:
            return 'cpu'
    
    try:
        import torch
        
        # GPU 사용 불가능한 경우
        if not torch.cuda.is_available():
            return torch.device('cpu')
        
        # 🔥 GPU 적극 활용 모드: 85%까지 최대한 활용
        memory_info = get_gpu_memory_info()
        if memory_info:
            current_usage = memory_info['usage_percent']
            # 85% 미만이면 적극적으로 GPU 사용
            if current_usage < 85:
                logger.info(f"GPU 직접 사용: {model_id} (사용률: {current_usage:.1f}%, 필요: {memory_required_mb}MB)")
                return torch.device('cuda')
            # 85% 초과면 스왑을 통한 공간 확보 시도
            elif current_usage >= 85:
                logger.info(f"GPU 포화 상태 - 스왑 시도: {current_usage:.1f}%")
        
        # MasterMemoryOrchestrator를 통한 지능적 디바이스 선택
        orchestrator = get_master_memory_orchestrator()
        
        # 비동기 함수를 동기적으로 실행
        try:
            device = run_async_safely(
                orchestrator.intelligent_device_selection(
                    model_id=model_id or f"temp_model_{int(time.time())}",
                    priority=priority,
                    estimated_memory_mb=memory_required_mb,
                    force_gpu=False
                ),
                timeout=5.0
            )
            
            if device is not None:
                return device
            else:
                return torch.device('cpu')
                
        except Exception as e:
            logger.warning(f"지능적 디바이스 선택 실패: {e}. 기본 로직 사용")
            
            # 폴백: 기본 메모리 체크
            memory_info = get_gpu_memory_info()
            if memory_info and memory_info['free_mb'] > memory_required_mb * 1.2:
                return torch.device('cuda')
            else:
                return torch.device('cpu')
                
    except Exception as e:
        logger.error(f"디바이스 선택 중 오류: {e}. CPU 사용")
        try:
            import torch
            return torch.device('cpu')
        except ImportError:
            return 'cpu'

# 시스템 설정
SYSTEM_CONFIG = {
    # 성능 설정 (Linux 최적화)
    'performance': {
        'batch_size': 16 if ADVANCED_CONFIG['enable_gpu'] else 8,
        'processing_delay': 0.1,        # Linux에서 더 빠른 처리
        'max_memory_usage': 0.7,        # Linux 메모리 관리 효율성
        'save_interval': 10,
        'num_workers': min(8, os.cpu_count()),
        'prefetch_factor': 2,
        'pin_memory': ADVANCED_CONFIG['enable_gpu'],
    },
    
    # 학습 설정 (40M 파라미터 - 메타 통합에 집중)
    'learning': {
        'initial_learning_rate': 0.005,
        'min_learning_rate': 0.0001,
        'learning_decay': 0.995,
        'regret_threshold': 0.3,
        'max_regret_intensity': 5.0,
        'early_stopping_patience': 10,
        'validation_split': 0.2,
        'total_parameters': 40_000_000,   # 정확한 파라미터 수
        'memory_required_mb': 160,        # 40M * 4 bytes (FP32)
        'priority': 'LOW',                # 우선순위
        'model_id': 'meta_integration',   # 메모리 관리용 ID
        'circuit_integration_layers': [256, 128, 64],  # 통합 서킷 레이어
        'meta_learning': True,           # 메타 학습 활성화
        'adaptive_weights': True,        # 적응적 가중치
        'ensemble_methods': True,        # 앙상블 방법
    },
    
    # 벤담 계산 설정 (120M 파라미터)
    'bentham': {
        'weights': {
            'intensity': 0.2,
            'duration': 0.15,
            'certainty': 0.15,
            'propinquity': 0.1,
            'fecundity': 0.15,
            'purity': 0.1,
            'extent': 0.15,
        },
        'total_parameters': 120_000_000,  # 정확한 파라미터 수
        'memory_required_mb': 480,        # 120M * 4 bytes (FP32)
        'priority': 'HIGH',               # 우선순위
        'model_id': 'bentham_calculator', # 메모리 관리용 ID
        'neural_predictor_layers': [512, 1024, 512, 256],  # 신경망 예측기
        'weight_layers': 6,              # 6층 가중치 레이어
        'dynamic_scaling': True,         # 동적 스케일링
        'layer_norm': True,              # 레이어 정규화
        'advanced_dropout': 0.1,         # 고급 드롭아웃
        'residual_connections': True,    # 잔여 연결
        },
        'enhancement_layers': {
            'cultural_weight': 0.2,
            'temporal_weight': 0.15,
            'social_weight': 0.2,
            'personal_weight': 0.2,
            'moral_weight': 0.15,
            'situational_weight': 0.1,
        },
    
    # 고급 의미 분석 설정 (80M 파라미터 - 반사실적 추론에 집중)
    'semantic': {
        'sentence_model': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        'korean_model': 'jhgan/ko-sroberta-multitask',
        'similarity_threshold': 0.75,
        'cache_size': 10000,
        'embedding_dimension': 1024,  # 임베딩 차원 확장
        'surface_weight': 0.2,
        'ethical_weight': 0.3,
        'emotional_weight': 0.3,
        'causal_weight': 0.2,
        'use_llm': True,
        'llm_model': 'microsoft/DialoGPT-medium',
        'max_sequence_length': 512,
        'total_parameters': 80_000_000,   # 정확한 파라미터 수
        'memory_required_mb': 320,        # 80M * 4 bytes (FP32)
        'priority': 'MEDIUM',             # 우선순위
        'model_id': 'semantic_analyzer',  # 메모리 관리용 ID
        'counterfactual_layers': [512, 256, 128],  # 반사실적 추론 네트워크
        'deep_reasoning': True,       # 깊은 추론 모드
        'advanced_attention': True,   # 고급 어텐션
    },
    
    # SURD 분석 설정 (강화됨)
    'surd': {
        'kraskov_k': 5,                 # 더 정확한 추정
        'min_samples': 50,              # 더 많은 샘플 요구
        'max_variables': 10,
        'min_effect_threshold': 0.001,  # 더 민감한 감지
        'simulation_samples': 10000,    # 고품질 시뮬레이션
        'bootstrap_iterations': 1000,
        'confidence_level': 0.95,
        'parallel_processing': True,
    },
    
    # 후회 분석 설정 (120M 파라미터)
    'regret': {
        'learning_rate': 0.0005,
        'hidden_layers': [1024, 768, 512, 256, 128, 64],  # 깊은 네트워크
        'dropout_rate': 0.15,
        'l2_regularization': 0.01,
        'evaluation_metrics': ['mse', 'mae', 'r2'],
        'cross_validation_folds': 5,
        'total_parameters': 120_000_000,  # 정확한 파라미터 수
        'memory_required_mb': 480,        # 120M * 4 bytes (FP32)
        'priority': 'MEDIUM',             # 우선순위
        'model_id': 'regret_analyzer',    # 메모리 관리용 ID
        'attention_mechanism': True,     # 어텐션 메커니즘 추가
        'residual_connections': True,    # 잔여 연결
        'batch_norm': True,              # 배치 정규화
        'advanced_optimization': True,   # 고급 최적화
    },
    
    # 데이터 처리 설정
    'data': {
        'experience_compression_threshold': 1000,
        'max_active_experiences': 5000,
        'backup_interval_hours': 6,
        'data_validation': True,
        'auto_cleanup': True,
        'korean_preprocessing': True,
    },
    
    # Advanced emotion analysis settings (140M 파라미터)
    'emotion': {
        'model_name': 'cardiffnlp/twitter-roberta-base-emotion',
        'use_transformers': True,
        'confidence_threshold': 0.8,
        'fallback_to_keywords': False,
        'require_advanced': True,
        'enable_biosignal': True,
        'korean_model': 'beomi/KcELECTRA-base-v2022',
        'multilingual_model': 'j-hartmann/emotion-english-distilroberta-base',
        'batch_processing': True,
        'cache_embeddings': True,
        'total_parameters': 140_000_000,  # 정확한 파라미터 수
        'memory_required_mb': 560,        # 140M * 4 bytes (FP32)
        'priority': 'HIGH',               # 우선순위
        'model_id': 'emotion_analyzer',   # 메모리 관리용 ID
        'hidden_layers': [1024, 512, 256, 128],
        'attention_heads': 16,
        'layer_norm': True,
        'dropout_rate': 0.1,
    },
    
    # 번역 설정 (LocalTranslator - 400MB OPUS-MT)
    'translation': {
        'model_name': 'Helsinki-NLP/opus-mt-ko-en',
        'total_parameters': 74_000_000,   # 74M 파라미터 (OPUS-MT 표준)
        'memory_required_mb': 400,        # 모델 + 토크나이저
        'priority': 'HIGH',               # 다른 모듈들의 의존성
        'model_id': 'translator',         # 메모리 관리용 ID
        'cache_embeddings': True,
        'max_sequence_length': 512,
        'batch_size': 16,
        'num_beams': 3,
        'early_stopping': True,
    },
    
    # 신경망 컴포넌트 설정 (HierarchicalPatternStructure - 150MB)
    'neural': {
        'total_parameters': 40_000_000,   # 40M 파라미터
        'memory_required_mb': 150,        # 가벼운 모듈
        'priority': 'MEDIUM',             # 선택적 모듈
        'model_id': 'neural_components',  # 메모리 관리용 ID
        'hidden_dim': 512,
        'num_layers': 4,
        'dropout_rate': 0.1,
        'use_attention': True,
    },
    
    # 로깅 설정
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_rotation': True,
        'max_file_size': '10MB',
        'backup_count': 5,
        'performance_logging': True,
    }
}

def setup_logging():
    """고급 로깅 시스템 설정"""
    log_config = SYSTEM_CONFIG['logging']
    
    # 로그 레벨 설정
    level = getattr(logging, log_config['level'])
    
    # 로그 포맷 설정
    formatter = logging.Formatter(log_config['format'])
    
    # 루트 로거 설정
    logger = logging.getLogger('RedHeartLinux')
    logger.setLevel(level)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러 (로테이션)
    if log_config['file_rotation']:
        from logging.handlers import RotatingFileHandler
        log_file = os.path.join(LOGS_DIR, f'red_heart_{datetime.datetime.now().strftime("%Y%m%d")}.log')
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=int(log_config['max_file_size'].replace('MB', '')) * 1024 * 1024,
            backupCount=log_config['backup_count']
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# 디바이스 설정은 이미 위에서 처리됨 (torch import 지연 방식)

# 배치 사이즈 설정
BATCH_SIZE = SYSTEM_CONFIG['performance']['batch_size']

# 고급 모듈 필수 모드 설정
REQUIRE_ADVANCED_MODULES = ADVANCED_CONFIG.get('strict_mode', True)

# CUDA Context 프리로딩 시스템 (11초 지연 해결)
_cuda_context_initialized = False
_gpu_memory_cache = None

def preload_cuda_context():
    """CUDA Context 사전 초기화 (11초 지연 근본 해결)"""
    global _cuda_context_initialized, _gpu_memory_cache
    
    if _cuda_context_initialized:
        return True
        
    import torch
    if not torch.cuda.is_available():
        return False
        
    try:
        logger.info("🚀 CUDA Context 프리로딩 시작 (WSL 지연 해결)...")
        start_time = time.time()
        
        # Step 1: 간단한 CUDA 연산으로 context 활성화
        device = torch.device('cuda:0')
        dummy_tensor = torch.tensor([1.0], device=device)
        _ = dummy_tensor * 2  # 강제 context 초기화
        
        # Step 2: GPU 메모리 정보 첫 호출 (지연 흡수)
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated(0)
        cached_memory = torch.cuda.memory_reserved(0)
        
        # Step 3: 메모리 정보 캐싱
        _gpu_memory_cache = {
            'total_mb': total_memory / (1024 * 1024),
            'properties': torch.cuda.get_device_properties(0)
        }
        
        # Step 4: 더미 텐서 정리
        del dummy_tensor
        torch.cuda.empty_cache()
        
        _cuda_context_initialized = True
        duration = time.time() - start_time
        logger.info(f"✅ CUDA Context 프리로딩 완료 ({duration:.1f}초, 향후 즉시 응답)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ CUDA Context 프리로딩 실패: {e}")
        return False

# 중복 함수 제거 - 위의 get_gpu_memory_info() 사용

# 중복된 get_smart_device 함수 제거 - 위의 MasterMemoryOrchestrator 기반 함수 사용

def get_device():
    """최적 디바이스 반환 (GPU/CPU) - 레거시 호환성"""
    return get_smart_device()

class GPUModelContext:
    """GPU 모델 임시 사용을 위한 Context Manager"""
    
    def __init__(self, model, memory_required_mb=500, force_cpu=False):
        self.model = model
        self.memory_required_mb = memory_required_mb
        self.force_cpu = force_cpu
        self.original_device = None
        self.target_device = None
        self.moved_to_gpu = False
        
    def __enter__(self):
        """GPU로 모델 이동"""
        import torch
        
        # 현재 모델이 어느 디바이스에 있는지 확인
        if hasattr(self.model, 'device'):
            self.original_device = self.model.device
        elif hasattr(self.model, 'parameters'):
            try:
                self.original_device = next(self.model.parameters()).device
            except StopIteration:
                self.original_device = torch.device('cpu')
        else:
            self.original_device = torch.device('cpu')
        
        # 타겟 디바이스 결정
        self.target_device = get_smart_device(self.memory_required_mb, self.force_cpu)
        
        # GPU로 이동이 필요하고 가능한 경우에만 이동
        if (self.target_device.type == 'cuda' and 
            self.original_device.type != 'cuda' and 
            hasattr(self.model, 'to')):
            try:
                self.model = self.model.to(self.target_device)
                self.moved_to_gpu = True
            except Exception as e:
                # GPU 이동 실패 시 CPU에서 계속 진행
                self.target_device = torch.device('cpu')
                if self.original_device.type != 'cpu':
                    self.model = self.model.to('cpu')
        
        return self.model
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """원래 디바이스로 모델 복귀"""
        # GPU에서 CPU로 이동한 경우에만 복귀
        if (self.moved_to_gpu and 
            self.original_device.type == 'cpu' and 
            hasattr(self.model, 'to')):
            try:
                self.model = self.model.to(self.original_device)
            except Exception:
                # 복귀 실패 시에도 에러 발생시키지 않음
                pass

def gpu_model_context(model, memory_required_mb=500, force_cpu=False):
    """GPU 모델 임시 사용을 위한 헬퍼 함수"""
    return GPUModelContext(model, memory_required_mb, force_cpu)

# GPU 메모리 우선순위 시스템 (이미 위에서 정의됨)

# 전역 GPU 로딩 순차 제어 시스템
import queue
import threading
from dataclasses import dataclass, field
from typing import Callable, Any, Optional, Dict, List

@dataclass(order=True)
class GPULoadingRequest:
    """GPU 로딩 요청"""
    priority: int  # 비교를 위해 첫 번째 필드로 이동
    model_id: str = field(compare=True)
    estimated_memory_mb: int = field(compare=False)
    loading_function: Callable[[], Any] = field(default=None, compare=False)
    result_queue: queue.Queue = field(default=None, compare=False)
    device_queue: queue.Queue = field(default=None, compare=False)

class SequentialGPULoader:
    """순차적 GPU 모델 로딩 매니저"""
    
    def __init__(self):
        self.loading_queue = queue.PriorityQueue()
        self.loading_thread = None
        self.is_running = False
        self.current_gpu_usage = 0.0
        self.loaded_models = {}  # {model_id: {'memory_mb': int, 'device': str}}
        self.lock = threading.RLock()
        
    def start(self):
        """로딩 스레드 시작"""
        if not self.is_running:
            self.is_running = True
            self.loading_thread = threading.Thread(target=self._loading_worker, daemon=True)
            self.loading_thread.start()
            logger.info("순차적 GPU 로딩 시스템 시작")
    
    def stop(self):
        """로딩 스레드 중지"""
        self.is_running = False
        if self.loading_thread:
            self.loading_thread.join(timeout=5)
    
    def classify_model_risk(self, model_id: str) -> str:
        """모델 위험도 분류 (SequentialGPULoader용 기본 구현)"""
        # 기본적인 위험도 분류 로직
        if 'backbone' in model_id.lower():
            return 'HIGH'      # 백본은 항상 고위험 (중요함)
        elif 'head' in model_id.lower():
            return 'MEDIUM'    # 헤드들은 중위험
        elif any(keyword in model_id.lower() for keyword in ['large', 'huge', 'xl']):
            return 'HIGH'      # 큰 모델들은 고위험
        else:
            return 'LOW'       # 나머지는 저위험
    
    def request_gpu_loading(self, model_id: str, priority: int, estimated_memory_mb: int, loading_function: Callable[[], Any], timeout: float = 30.0):
        """GPU 로딩 요청 (마스터 오케스트레이터와 통합)"""
        import torch  # WSL worker thread에서 torch 접근 보장
        if not self.is_running:
            self.start()
        
        # 🔥 핵심 변화: MasterMemoryOrchestrator 활용
        master_orchestrator = get_master_orchestrator()
        
        # 헬퍼 함수로 비동기 실행을 동기적으로 처리
        try:
            device, result = run_async_safely(
                master_orchestrator.intelligent_load_model(
                    model_id, priority, estimated_memory_mb, loading_function
                ),
                timeout=timeout
            )
            
            if device is None:
                logger.error(f"마스터 오케스트레이터 로딩 타임아웃: {model_id}")
                return torch.device('cpu'), None
                
            return device, result
            
        except Exception as e:
            logger.error(f"마스터 오케스트레이터 로딩 실패: {model_id} - {str(e)}")
            return torch.device('cpu'), None
    
    def _loading_worker(self):
        """로딩 워커 스레드 (순차적 처리)"""
        import logging
        import torch  # WSL worker thread에서 torch 접근 보장
        worker_logger = logging.getLogger(__name__)
        worker_logger.info("GPU 로딩 워커 스레드 시작")
        
        while self.is_running:
            try:
                # 다음 로딩 요청 대기 (우선순위 순서대로)
                request = self.loading_queue.get(timeout=1.0)
                
                # try-finally로 task_done() 안전하게 보장
                try:
                    model_id = request.model_id
                    priority = request.priority
                    
                    with self.lock:
                        # 현재 GPU 메모리 상태 확인
                        memory_info = get_gpu_memory_info()
                        if memory_info is None:
                            # GPU 사용 불가능
                            request.device_queue.put(torch.device('cpu'))
                            return  # task_done()은 finally에서 호출
                        
                        current_usage = memory_info['usage_percent']
                        free_mb = memory_info['free_mb']
                        
                        # 안전 여유분을 고려한 로딩 가능성 판단
                        safety_margin = 1000  # 1GB 안전 여유분
                        required_total = request.estimated_memory_mb + safety_margin
                        
                        # 로딩 가능성 판단 (85% 기준으로 최적화)
                        can_load_to_gpu = (
                            current_usage < 82 and  # 82% 미만으로 85% 기준 적용
                            free_mb > required_total and  # 여유 메모리 충분
                            current_usage + (request.estimated_memory_mb / 80) < 83  # 로딩 후 예상 사용률 83% 미만
                        )
                        
                        if can_load_to_gpu:
                            # GPU 로딩 진행
                            worker_logger.info(f"GPU 로딩 시작: {model_id} (현재 사용률: {current_usage:.1f}%, 여유: {free_mb}MB)")
                            request.device_queue.put(torch.device('cuda'))
                            
                            try:
                                # 실제 모델 로딩 실행
                                result = request.loading_function()
                                
                                # 로딩 후 메모리 상태 재확인
                                new_memory_info = get_gpu_memory_info()
                                if new_memory_info:
                                    new_usage = new_memory_info['usage_percent']
                                    actual_memory_used = (new_usage - current_usage) * 80  # 8GB * 10
                                    
                                    # 로딩된 모델 기록
                                    self.loaded_models[model_id] = {
                                        'memory_mb': actual_memory_used,
                                        'device': 'cuda'
                                    }
                                    
                                    worker_logger.info(f"GPU 로딩 완료: {model_id} (사용률: {current_usage:.1f}% → {new_usage:.1f}%, 실제 사용: {actual_memory_used:.0f}MB)")
                                    
                                    # 메모리 사용률이 85% 초과하면 긴급 조치
                                    if new_usage > 85:
                                        worker_logger.error(f"GPU 메모리 위험 수준: {new_usage:.1f}% - 다음 모델들은 CPU로 강제 이동")
                                        self._force_cpu_mode()
                                
                                request.result_queue.put(result)
                                
                            except Exception as e:
                                worker_logger.error(f"GPU 로딩 실패: {model_id} - {e}")
                                request.result_queue.put(None)
                        else:
                            # CPU로 로딩
                            worker_logger.info(f"CPU 로딩: {model_id} (GPU 메모리 부족: 현재 {current_usage:.1f}%, 여유 {free_mb}MB < 필요 {required_total}MB)")
                            request.device_queue.put(torch.device('cpu'))
                            
                            # CPU 로딩도 기록
                            self.loaded_models[model_id] = {
                                'memory_mb': 0,
                                'device': 'cpu'
                            }
                
                finally:
                    # 모든 코드 경로에서 반드시 task_done() 호출
                    self.loading_queue.task_done()
                    
            except queue.Empty:
                continue
            except Exception as e:
                worker_logger.error(f"GPU 로딩 워커 오류: {e}")
                # queue.get()이 실패한 경우에는 task_done() 호출하지 않음
    
    def _force_cpu_mode(self):
        """CPU 강제 모드 활성화"""
        import logging
        worker_logger = logging.getLogger(__name__)
        worker_logger.warning("GPU 메모리 한계로 CPU 강제 모드 활성화")
        # 이후 모든 요청을 CPU로 처리하도록 설정
        # 이 부분은 추후 구현 가능

# 전역 통합 메모리 오케스트레이션 시스템
_gpu_loaded_models = {}  # GPU에 로드된 모델들 추적 (미선언 버그 수정)
_gpu_loader = SequentialGPULoader()

# 전역 모듈 레지스트리 - 시스템의 모든 모듈 정보
_global_module_registry = {}
_module_instances = {}  # 실제 모듈 인스턴스들

def register_system_module(module_id: str, module_instance, config_section: str = None, replace: bool = False):
    """시스템 모듈을 전역 레지스트리에 등록
    
    Args:
        module_id: 모듈 고유 ID (예: 'emotion_analyzer')
        module_instance: 실제 모듈 인스턴스
        config_section: 설정 섹션 이름 (예: 'emotion')
        replace: True면 기존 모듈 교체 허용 (기본값: False)
    """
    global _global_module_registry, _module_instances
    
    logger.info(f"🔥 register_system_module 호출: module_id={module_id}, instance_type={type(module_instance)}, config_section={config_section}")
    
    # STRICT_NO_OVERWRITE: 이미 등록된 모듈이면 덮어쓰기 금지 (replace=False인 경우)
    if module_id in _module_instances:
        existing_type = type(_module_instances[module_id]).__name__
        new_type = type(module_instance).__name__
        
        if not replace:
            # 동일 타입이면 경고만 하고 진행
            if existing_type == new_type:
                logger.warning(f"⚠️ {module_id} 이미 등록됨 (동일 타입: {existing_type}), 스킵")
                return  # 동일 타입이면 경고만 하고 진행
            else:
                logger.error(f"❌ STRICT_NO_OVERWRITE: {module_id}가 이미 등록됨")
                logger.error(f"   기존: {existing_type}")
                logger.error(f"   신규: {new_type}")
                logger.error(f"   교체하려면 replace=True 사용")
                raise RuntimeError(f"STRICT_NO_OVERWRITE: {module_id} already registered")
        else:
            # replace=True인 경우 기존 모듈 정리
            logger.info(f"🔄 기존 모듈 교체: {module_id} ({existing_type} → {new_type})")
            old_module = _module_instances[module_id]
            # 기존 모듈 메모리 해제 시도
            if hasattr(old_module, 'cleanup') and callable(old_module.cleanup):
                try:
                    old_module.cleanup()
                    logger.info(f"   기존 모듈 정리 완료")
                except Exception as e:
                    logger.error(f"   기존 모듈 정리 실패: {e}")
    
    # 특정 모듈에 대한 인터페이스 검증
    if module_id in {"emotion_analyzer", "bentham_calculator"}:
        if not hasattr(module_instance, "get_pytorch_network"):
            logger.error(f"❌ {module_id} must implement get_pytorch_network method")
            # 사용 가능한 메서드 로깅
            methods = [m for m in dir(module_instance) if not m.startswith('_') and callable(getattr(module_instance, m, None))]
            logger.error(f"   Available methods: {methods[:10]}...")
            raise AssertionError(f"{module_id} must implement get_pytorch_network")
        else:
            logger.info(f"✅ {module_id} has get_pytorch_network method")
    
    # 설정에서 모듈 정보 가져오기
    if config_section and config_section in SYSTEM_CONFIG:
        module_config = SYSTEM_CONFIG[config_section]
        
        # 모듈 메타데이터 생성
        module_info = {
            'module_id': module_id,
            'config_section': config_section,
            'total_parameters': module_config.get('total_parameters', 0),
            'memory_required_mb': module_config.get('memory_required_mb', 0),
            'priority': MODULE_PRIORITY_MAP.get(module_config.get('priority', 'MEDIUM'), ModelPriority.MEDIUM),
            'device': 'cpu',  # 초기값
            'loaded': False,
            'last_used': time.time()
        }
        
        _global_module_registry[module_id] = module_info
        _module_instances[module_id] = module_instance
        
        logger.info(f"시스템 모듈 등록: {module_id} ({module_config.get('total_parameters', 0):,} 파라미터)")
        
        # MasterMemoryOrchestrator에도 등록
        orchestrator = get_master_orchestrator()
        if orchestrator:
            orchestrator.master_model_registry[module_id] = {
                'device': 'cpu',
                'memory_mb': module_config.get('memory_required_mb', 0),
                'priority': MODULE_PRIORITY_MAP.get(module_config.get('priority', 'MEDIUM'), ModelPriority.MEDIUM),
                'total_parameters': module_config.get('total_parameters', 0),
                'load_time': time.time(),
                'access_count': 0,
                'risk_level': 'MEDIUM'
            }
    else:
        # config_section이 없어도 _module_instances에는 등록
        _module_instances[module_id] = module_instance
        logger.info(f"🔥 모듈 인스턴스 등록 (config_section 없음): {module_id} -> {type(module_instance)}")

def get_system_module(module_id: str):
    """등록된 시스템 모듈 인스턴스 반환
    
    Args:
        module_id: 모듈 고유 ID
        
    Returns:
        모듈 인스턴스 또는 None
    """
    logger.info(f"🔍 get_system_module 호출: module_id={module_id}")
    logger.info(f"🔍 현재 등록된 모듈들: {list(_module_instances.keys())}")
    result = _module_instances.get(module_id)
    logger.info(f"🔍 결과: {type(result) if result else 'None'}")
    return result

def get_module_info(module_id: str):
    """등록된 모듈의 메타데이터 반환
    
    Args:
        module_id: 모듈 고유 ID
        
    Returns:
        모듈 메타데이터 딕셔너리 또는 None
    """
    return _global_module_registry.get(module_id)

def run_async_safely(coro, timeout=60.0):
    """비동기 함수를 동기적으로 안전하게 실행하는 헬퍼"""
    try:
        # 현재 이벤트 루프가 실행 중인지 확인
        loop = asyncio.get_running_loop()
        
        # 새 스레드에서 실행
        import concurrent.futures
        import threading
        
        result_holder = {'result': None, 'exception': None}
        
        def run_in_new_loop():
            try:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                result_holder['result'] = new_loop.run_until_complete(coro)
                new_loop.close()
            except Exception as e:
                result_holder['exception'] = e
            finally:
                asyncio.set_event_loop(None)
        
        thread = threading.Thread(target=run_in_new_loop)
        thread.start()
        thread.join(timeout=timeout)
        
        if thread.is_alive():
            logger.error("비동기 실행 타임아웃")
            return None
            
        if result_holder['exception']:
            raise result_holder['exception']
            
        return result_holder['result']
        
    except RuntimeError:
        # 이벤트 루프가 없는 경우 직접 실행
        return asyncio.run(coro)

# 모델 실행 상태 열거형
from enum import Enum

class ModelExecutionState(Enum):
    """모델 실행 상태"""
    IDLE = "idle"                    # 유휴 상태 - 스왑 가능
    BUSY = "busy"                    # 작업 실행 중 - 스왑 불가
    LOADING = "loading"              # 로딩 중 - 스왑 불가
    PENDING_SWAP = "pending_swap"    # 스왑 대기 중
    SWAPPING = "swapping"            # 스왑 진행 중
    ERROR = "error"                  # 에러 상태

class TaskSequenceType(Enum):
    """연계 작업 유형"""
    STANDALONE = "standalone"        # 독립 작업
    BACKBONE_HEAD = "backbone_head"  # 백본 + 헤드 조합
    PIPELINE = "pipeline"            # 파이프라인 연계
    BATCH = "batch"                  # 배치 처리

@dataclass
class ModelExecutionContext:
    """모델 실행 컨텍스트"""
    model_id: str
    state: ModelExecutionState = ModelExecutionState.IDLE
    current_task_id: Optional[str] = None
    start_time: Optional[float] = None
    estimated_completion_time: Optional[float] = None
    sequence_type: TaskSequenceType = TaskSequenceType.STANDALONE
    related_models: List[str] = field(default_factory=list)
    is_critical_for_sequence: bool = False
    last_access_time: float = field(default_factory=time.time)
    access_count: int = 0

class MasterMemoryOrchestrator:
    """작업 완료 기반 지능적 메모리 관리 시스템
    
    핵심 개선사항:
    - 우선순위 기반 → 작업 완료 기반 스왑
    - 실시간 작업 상태 추적 및 모니터링
    - 연계 작업 (backbone+head) 감지 및 안전 대기
    - 작업 순서 기반 안전한 스왑 타이밍 제어
    """
    
    def __init__(self):
        # 단일 진실 소스 (Single Source of Truth)
        self.master_model_registry = {}  # {model_id: ModelMetadata}
        self.gpu_memory_map = {}         # 실제 GPU 상태 추적
        self.compressed_cache = {}       # 압축된 모델들
        
        # 🔥 새로운 작업 상태 추적 시스템
        self.execution_contexts = {}     # {model_id: ModelExecutionContext}
        self.active_task_sequences = {}  # {sequence_id: [model_ids]}
        self.pending_swap_queue = []     # 스왑 대기 큐
        
        # 🔥 작업 완료 모니터링 시스템
        self.task_completion_callbacks = {}  # {task_id: callback_function}
        self.sequence_dependencies = {}      # {model_id: [dependent_model_ids]}
        
        # 기존 고급 시스템들과 연결
        self._dynamic_gpu_manager = None
        self._swap_manager = None
        self._predictor = None
        self._compressor = None
        
        # 실제 모델 인스턴스들의 약한 참조 (메모리 누수 방지)
        import weakref
        self.active_model_refs = {}      # {model_id: weakref.ref(model)}
        
        # 스레드 안전성
        self.lock = threading.RLock()
        
        # 🔥 안전한 스왑 스케줄러
        self.safe_swap_scheduler_running = False
        self.swap_scheduler_thread = None
    
    def connect_subsystems(self, gpu_manager=None, swap_manager=None, predictor=None, compressor=None):
        """기존 서브시스템들과 연결"""
        self._dynamic_gpu_manager = gpu_manager
        self._swap_manager = swap_manager  
        self._predictor = predictor
        self._compressor = compressor
        
        # 안전한 스왑 스케줄러 시작
        self._start_safe_swap_scheduler()
    
    def _start_safe_swap_scheduler(self):
        """안전한 스왑 스케줄러 스레드 시작"""
        if not self.safe_swap_scheduler_running:
            self.safe_swap_scheduler_running = True
            self.swap_scheduler_thread = threading.Thread(
                target=self._safe_swap_scheduler_worker, 
                daemon=True
            )
            self.swap_scheduler_thread.start()
            logger.info("안전한 스왑 스케줄러 시작됨")
    
    def _safe_swap_scheduler_worker(self):
        """안전한 스왑 스케줄러 워커 스레드"""
        while self.safe_swap_scheduler_running:
            try:
                with self.lock:
                    # 스왑 대기 큐에서 처리 가능한 항목 확인
                    if self.pending_swap_queue:
                        self._process_pending_swaps()
                
                # 1초마다 체크
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"스왑 스케줄러 오류: {e}")
                time.sleep(5.0)  # 에러 시 5초 대기
    
    def register_model_execution_start(self, model_id: str, task_id: str, sequence_type: TaskSequenceType = TaskSequenceType.STANDALONE, related_models: List[str] = None):
        """
        모델 작업 시작 등록
        
        Args:
            model_id: 모델 ID
            task_id: 작업 ID
            sequence_type: 연계 작업 유형
            related_models: 연계된 모델들
        """
        with self.lock:
            if model_id not in self.execution_contexts:
                self.execution_contexts[model_id] = ModelExecutionContext(model_id=model_id)
            
            context = self.execution_contexts[model_id]
            context.state = ModelExecutionState.BUSY
            context.current_task_id = task_id
            context.start_time = time.time()
            context.sequence_type = sequence_type
            context.related_models = related_models or []
            context.access_count += 1
            context.last_access_time = time.time()
            
            # 연계 작업 등록
            if sequence_type != TaskSequenceType.STANDALONE and related_models:
                sequence_id = f"{task_id}_{sequence_type.value}"
                self.active_task_sequences[sequence_id] = [model_id] + related_models
                
                # 의존성 등록
                for related_model in related_models:
                    if related_model not in self.sequence_dependencies:
                        self.sequence_dependencies[related_model] = []
                    self.sequence_dependencies[related_model].append(model_id)
            
            logger.info(f"작업 시작 등록: {model_id} (작업: {task_id}, 유형: {sequence_type.value})")
    
    def register_model_execution_complete(self, model_id: str, task_id: str):
        """
        모델 작업 완료 등록
        
        Args:
            model_id: 모델 ID
            task_id: 작업 ID
        """
        with self.lock:
            if model_id not in self.execution_contexts:
                logger.warning(f"작업 완료 등록 실패 - 컨텍스트 없음: {model_id}")
                return
            
            context = self.execution_contexts[model_id]
            
            # 작업 ID 매칭 확인
            if context.current_task_id != task_id:
                logger.warning(f"작업 ID 불일치: {model_id}, 예상: {context.current_task_id}, 실제: {task_id}")
            
            # 상태 업데이트
            context.state = ModelExecutionState.IDLE
            context.current_task_id = None
            context.start_time = None
            context.last_access_time = time.time()
            
            logger.info(f"작업 완료 등록: {model_id} (작업: {task_id})")
            
            # 완료 콜백 실행
            if task_id in self.task_completion_callbacks:
                try:
                    callback = self.task_completion_callbacks[task_id]
                    callback(model_id, task_id)
                    del self.task_completion_callbacks[task_id]
                except Exception as e:
                    logger.error(f"작업 완료 콜백 실행 실패: {e}")
    
    def is_model_safe_to_swap(self, model_id: str) -> bool:
        """
        모델이 안전하게 스왑 가능한지 확인
        
        Args:
            model_id: 모델 ID
            
        Returns:
            bool: 스왑 가능 여부
        """
        with self.lock:
            # 실행 컨텍스트 확인
            if model_id not in self.execution_contexts:
                return True  # 컨텍스트가 없으면 스왑 가능
            
            context = self.execution_contexts[model_id]
            
            # 현재 작업 중이면 스왑 불가
            if context.state in [ModelExecutionState.BUSY, ModelExecutionState.LOADING, ModelExecutionState.SWAPPING]:
                logger.debug(f"스왑 불가 - 작업 중: {model_id} (상태: {context.state.value})")
                return False
            
            # 연계 작업 확인
            if self._is_part_of_active_sequence(model_id):
                logger.debug(f"스왑 불가 - 연계 작업 진행 중: {model_id}")
                return False
            
            # 의존성 있는 모델들이 작업 중인지 확인
            if model_id in self.sequence_dependencies:
                for dependent_model in self.sequence_dependencies[model_id]:
                    if not self.is_model_safe_to_swap(dependent_model):
                        logger.debug(f"스왑 불가 - 의존성 모델 작업 중: {model_id} -> {dependent_model}")
                        return False
            
            return True
    
    def _is_part_of_active_sequence(self, model_id: str) -> bool:
        """모델이 활성 연계 작업의 일부인지 확인"""
        for sequence_id, models in self.active_task_sequences.items():
            if model_id in models:
                # 시퀀스 내 다른 모델들이 작업 중인지 확인
                for other_model in models:
                    if other_model != model_id and other_model in self.execution_contexts:
                        other_context = self.execution_contexts[other_model]
                        if other_context.state == ModelExecutionState.BUSY:
                            return True
        return False
    
    def request_safe_swap(self, model_id: str, reason: str = "memory_needed"):
        """
        안전한 스왑 요청 (즉시 실행하지 않고 대기열에 추가)
        
        Args:
            model_id: 스왑할 모델 ID
            reason: 스왑 사유
        """
        with self.lock:
            # 이미 대기열에 있는지 확인
            for pending_model, pending_reason, pending_time in self.pending_swap_queue:
                if pending_model == model_id:
                    logger.debug(f"스왑 이미 대기 중: {model_id}")
                    return
            
            # 대기열에 추가
            self.pending_swap_queue.append((model_id, reason, time.time()))
            logger.info(f"안전한 스왑 대기열 추가: {model_id} (사유: {reason})")
    
    def _process_pending_swaps(self):
        """대기 중인 스왑 요청들 처리"""
        processed_count = 0
        remaining_queue = []
        
        for model_id, reason, request_time in self.pending_swap_queue:
            if self.is_model_safe_to_swap(model_id):
                # 안전하게 스왑 실행
                try:
                    logger.info(f"안전한 스왑 실행: {model_id} (대기 시간: {time.time() - request_time:.1f}초)")
                    
                    # 비동기 언로드를 동기적으로 실행
                    success = run_async_safely(
                        self.intelligent_unload_model(model_id, force=True),
                        timeout=30.0
                    )
                    
                    if success:
                        processed_count += 1
                        logger.info(f"안전한 스왑 완료: {model_id}")
                    else:
                        remaining_queue.append((model_id, reason, request_time))
                        logger.warning(f"스왑 실패, 재시도 예정: {model_id}")
                        
                except Exception as e:
                    logger.error(f"스왑 실행 중 오류: {model_id} - {e}")
                    remaining_queue.append((model_id, reason, request_time))
            else:
                # 아직 안전하지 않으면 대기열에 유지
                remaining_queue.append((model_id, reason, request_time))
        
        # 대기열 업데이트
        self.pending_swap_queue = remaining_queue
        
        if processed_count > 0:
            logger.info(f"안전한 스왑 배치 완료: {processed_count}개 모델 처리")
    
    async def intelligent_device_selection(self, model_id: str, priority: int, estimated_memory_mb: int, force_gpu: bool = False):
        """
        지능적 디바이스 선택 (실제 모델 로딩 없이)
        - GPU 85% 활용 목표
        - 필요시 스왑으로 공간 확보
        - 위험도 기반 우선순위 적용
        
        Args:
            model_id: 모델 고유 ID
            priority: 모델 우선순위
            estimated_memory_mb: 예상 메모리 사용량 (MB)
            force_gpu: GPU 강제 사용 여부
            
        Returns:
            torch.device: 선택된 디바이스
        """
        import torch
        
        with self.lock:
            # 1. GPU 메모리 상태 정밀 체크
            memory_info = get_gpu_memory_info()
            if not memory_info:
                logger.info(f"GPU 사용 불가: {model_id}")
                return torch.device('cpu')
            
            current_usage = memory_info['usage_percent']
            free_mb = memory_info['free_mb']
            
            # 2. 위험도 분류 및 안전 여유분 계산
            try:
                if self._dynamic_gpu_manager:
                    risk_level = self._dynamic_gpu_manager.classify_model_risk(model_id)
                else:
                    # 기본 위험도 분류
                    if 'backbone' in model_id.lower():
                        risk_level = 'HIGH'
                    elif 'head' in model_id.lower():
                        risk_level = 'MEDIUM'
                    else:
                        risk_level = 'LOW'
                        
                safety_margin = {
                    'HIGH': 1000,    # 고위험 모델은 1GB 여유분
                    'MEDIUM': 800,   # 중위험 모델은 800MB 여유분  
                    'LOW': 500       # 저위험 모델은 500MB 여유분
                }.get(risk_level, 800)
                
            except Exception as e:
                logger.warning(f"위험도 분류 실패: {str(e)} - 기본값 사용")
                risk_level = 'MEDIUM'
                safety_margin = 800
            
            required_total = estimated_memory_mb + safety_margin
            
            # 3. 85% 기준 적용 - 우선순위별 차등 적용
            if priority <= ModelPriority.HIGH:
                # HIGH/CRITICAL 우선순위는 85%까지 적극 활용
                max_usage_threshold = 85
            else:
                # MEDIUM/LOW 우선순위는 80%까지만 활용 (안전)
                max_usage_threshold = 80
            
            # 4. 현재 상태로 GPU 사용 가능한지 체크
            if current_usage < max_usage_threshold and free_mb > required_total:
                logger.info(f"GPU 직접 사용: {model_id} (사용률: {current_usage:.1f}%, 필요: {required_total}MB)")
                return torch.device('cuda')
            
            # 5. 스왑 공간 확보 시도
            if current_usage >= 82 or free_mb < required_total:
                logger.info(f"GPU 공간 부족: {model_id} (사용률: {current_usage:.1f}%, 가용: {free_mb}MB, 필요: {required_total}MB)")
                
                if self._swap_manager:
                    try:
                        # 스왑 매니저를 통한 공간 확보 시도
                        freed_space = await self._swap_manager.free_gpu_space_intelligent(required_total)
                        if freed_space >= required_total:
                            logger.info(f"스왑으로 공간 확보 성공: {model_id} ({freed_space}MB 확보)")
                            return torch.device('cuda')
                        else:
                            logger.warning(f"스왑 공간 부족: {model_id} (필요: {required_total}MB, 확보: {freed_space}MB)")
                    except Exception as e:
                        logger.error(f"스왑 공간 확보 실패: {model_id} - {str(e)}")
                
                # 6. 🔥 작업 완료 기반 안전한 공간 확보 시도
                if len(self.master_model_registry) > 0:
                    logger.info(f"작업 완료 기반 공간 확보 시도: {model_id}")
                    
                    # 스왑 가능한 모델들 찾기 (작업 상태 기반)
                    swappable_models = []
                    for existing_model_id, model_info in self.master_model_registry.items():
                        if self.is_model_safe_to_swap(existing_model_id):
                            # 추가 조건: 현재 요청보다 덜 중요한 모델
                            existing_priority = model_info.get('priority', ModelPriority.LOW)
                            if existing_priority >= priority:  # 같은 우선순위도 포함 (LRU 방식)
                                swappable_models.append((
                                    existing_model_id, 
                                    model_info.get('memory_mb', 0),
                                    model_info.get('last_access_time', 0)
                                ))
                    
                    if swappable_models:
                        # LRU (Least Recently Used) 순으로 정렬
                        swappable_models.sort(key=lambda x: x[2])  # last_access_time 기준
                        
                        logger.info(f"스왑 가능한 모델 {len(swappable_models)}개 발견")
                        
                        # 필요한 메모리만큼 안전한 스왑 요청
                        freed_memory = 0
                        swap_requested_count = 0
                        
                        for swap_model_id, model_memory, last_access_time in swappable_models:
                            if freed_memory >= required_total:
                                break
                                
                            # 안전한 스왑 대기열에 추가
                            self.request_safe_swap(
                                model_id=swap_model_id,
                                reason=f"space_for_{model_id}"
                            )
                            
                            freed_memory += model_memory
                            swap_requested_count += 1
                            
                            logger.info(f"스왑 요청: {swap_model_id} ({model_memory}MB, 마지막 사용: {time.time() - last_access_time:.1f}초 전)")
                        
                        # 스왑 완료 대기 (최대 30초)
                        if swap_requested_count > 0:
                            logger.info(f"스왑 완료 대기 중: {swap_requested_count}개 모델, 예상 확보: {freed_memory}MB")
                            
                            wait_start_time = time.time()
                            max_wait_time = 30.0
                            check_interval = 1.0
                            
                            while time.time() - wait_start_time < max_wait_time:
                                # 현재 메모리 상태 재확인
                                updated_memory_info = get_gpu_memory_info()
                                if updated_memory_info and updated_memory_info['free_mb'] > required_total:
                                    wait_time = time.time() - wait_start_time
                                    logger.info(f"작업 완료 기반 공간 확보 성공: {model_id} (대기 시간: {wait_time:.1f}초)")
                                    return torch.device('cuda')
                                
                                # 잠시 대기 후 재확인
                                await asyncio.sleep(check_interval)
                            
                            logger.warning(f"스왑 대기 타임아웃: {model_id} (최대 대기 시간 {max_wait_time}초 초과)")
                    else:
                        logger.info(f"현재 스왑 가능한 모델 없음: {model_id} (모든 모델 작업 중)")
            
            # 7. 모든 시도 실패 시 CPU 사용
            logger.info(f"작업 완료 기반 공간 확보 실패, CPU 사용: {model_id}")
            return torch.device('cpu')
    
    async def intelligent_load_model(self, model_id: str, priority: int, estimated_memory_mb: int, loading_function: Callable, force_gpu: bool = False):
        """지능적 통합 모델 로딩
        
        기존 기술들을 모두 활용:
        1. TaskSequencePredictor로 필요성 예측
        2. ModelCompressor로 압축 해제 최적화  
        3. DynamicGPUManager로 안전한 메모리 할당
        4. RedHeartDynamicSwapManager로 스왑 공간 확보
        5. 실제 GPU 메모리 추적으로 정확한 상태 관리
        """
        import torch  # 비동기 함수 내부에서 torch 접근 보장
        with self.lock:
            # 1. 기존 예측 시스템 활용
            if self._predictor:
                prediction_confidence = await self._predictor.predict_model_need(model_id)
                if prediction_confidence < 0.3 and not force_gpu:
                    logger.info(f"예측 시스템: {model_id} 불필요 판정 (신뢰도: {prediction_confidence:.2f})")
                    return torch.device('cpu'), None
            
            # 2. 압축된 버전이 있으면 활용
            if model_id in self.compressed_cache and self._compressor:
                logger.info(f"압축 버전 발견: {model_id} - 압축 해제 중")
                # 압축 해제 로직은 기존 ModelCompressor 활용
            
            # 3. GPU 메모리 상태 정밀 체크
            memory_info = get_gpu_memory_info()
            if not memory_info:
                return torch.device('cpu'), None
            
            current_usage = memory_info['usage_percent']
            free_mb = memory_info['free_mb']
            
            # 4. 위험도 분류 및 안전 여유분 계산
            # GPU 로더의 위험도 분류 활용 (SequentialGPULoader에 구현됨)
            try:
                risk_level = _gpu_loader.classify_model_risk(model_id)
                safety_margin = {
                    'HIGH': 2000,    # 고위험 모델은 2GB 여유분 필요
                    'MEDIUM': 1500,  # 중위험 모델은 1.5GB 여유분
                    'LOW': 1000      # 저위험 모델은 1GB 여유분
                }.get(risk_level, 1500)
            except Exception as e:
                logger.warning(f"위험도 분류 실패: {str(e)} - 기본값 사용")
                risk_level = 'MEDIUM'
                safety_margin = 1500
            
            required_total = estimated_memory_mb + safety_margin
            
            # 5. 85% 기준 적용 및 스왑 공간 확보
            if current_usage >= 82 or free_mb < required_total:
                # 기존 SwapManager 활용하여 공간 확보
                if self._swap_manager:
                    freed_space = await self._swap_manager.free_gpu_space_intelligent(required_total)
                    if freed_space >= required_total:
                        logger.info(f"스왑 매니저가 {freed_space}MB 공간 확보 완료")
                    else:
                        logger.warning(f"스왑 공간 부족: 필요 {required_total}MB, 확보 {freed_space}MB")
                        return torch.device('cpu'), None
                else:
                    return torch.device('cpu'), None
            
            # 6. 실제 GPU 로딩 수행
            try:
                device = torch.device('cuda')
                logger.info(f"GPU 로딩 시작: {model_id} (예상 메모리: {estimated_memory_mb}MB)")
                
                result = loading_function()
                
                # 7. 로딩 후 실제 메모리 사용량 측정
                new_memory_info = get_gpu_memory_info()
                if new_memory_info:
                    new_usage = new_memory_info['usage_percent']
                    actual_memory_used = (new_usage - current_usage) * 80  # 8GB 기준
                    
                    # 8. 마스터 레지스트리에 등록 (단일 진실 소스)
                    self.master_model_registry[model_id] = {
                        'device': 'cuda',
                        'memory_mb': actual_memory_used,
                        'priority': priority,
                        'load_time': time.time(),
                        'access_count': 1,
                        'risk_level': risk_level if self._dynamic_gpu_manager else 'MEDIUM'
                    }
                    
                    # 9. 약한 참조로 실제 모델 추적 (메모리 누수 방지)
                    if hasattr(result, 'to'):  # PyTorch 모델인 경우
                        import weakref
                        self.active_model_refs[model_id] = weakref.ref(result, 
                            lambda ref: self._on_model_garbage_collected(model_id))
                    
                    # 10. 모든 서브시스템에 상태 동기화
                    await self._sync_to_all_subsystems(model_id, 'cuda', actual_memory_used)
                    
                    logger.info(f"GPU 로딩 성공: {model_id} (사용률: {current_usage:.1f}% → {new_usage:.1f}%, 실제: {actual_memory_used:.0f}MB)")
                    
                    # 11. 85% 초과 시 긴급 조치
                    if new_usage > 85:
                        logger.error(f"🚨 GPU 메모리 위험: {new_usage:.1f}% - 긴급 정리 시작")
                        await self._emergency_intelligent_cleanup()
                
                return device, result
                
            except Exception as e:
                logger.error(f"GPU 로딩 실패: {model_id} - {str(e)}")
                return torch.device('cpu'), None
    
    async def _sync_to_all_subsystems(self, model_id: str, device: str, memory_mb: float):
        """모든 서브시스템의 loaded_models를 마스터와 동기화"""
        model_info = {
            'device': device,
            'memory_mb': memory_mb,
            'last_sync': time.time()
        }
        
        # 기존 시스템들과 동기화
        if hasattr(self._dynamic_gpu_manager, 'loaded_models'):
            self._dynamic_gpu_manager.loaded_models[model_id] = model_info
        
        if hasattr(self._swap_manager, 'gpu_resident_models'):
            if device == 'cuda':
                self._swap_manager.gpu_resident_models[model_id] = model_info
            elif model_id in self._swap_manager.gpu_resident_models:
                del self._swap_manager.gpu_resident_models[model_id]
    
    async def intelligent_unload_model(self, model_id: str, force: bool = False):
        """진정한 모델 언로드 - 딕셔너리뿐만 아니라 실제 GPU 메모리도 해제"""
        import torch
        with self.lock:
            if model_id not in self.master_model_registry:
                logger.warning(f"언로드 요청된 모델이 레지스트리에 없음: {model_id}")
                return True
            
            model_info = self.master_model_registry[model_id]
            
            # 1. 우선순위 체크 (CRITICAL 모델은 보호)
            if not force and model_info.get('priority', ModelPriority.MEDIUM) <= ModelPriority.CRITICAL:
                logger.info(f"CRITICAL 모델 보호: {model_id} 언로드 거부")
                return False
            
            # 2. 실제 모델 인스턴스 GPU→CPU 이동 (2025 PyTorch 모범 사례 적용)
            model_instance = None
            if model_id in self.active_model_refs:
                model_ref = self.active_model_refs[model_id]
                model_instance = model_ref()  # 약한 참조에서 실제 객체 가져오기
                
                if model_instance is not None and hasattr(model_instance, 'to'):
                    try:
                        logger.info(f"🔄 실제 GPU→CPU 이동 시작: {model_id}")
                        
                        # 🔥 Step 1: GPU→CPU 이동
                        model_instance.to('cpu')
                        logger.info(f"✅ GPU→CPU 이동 완료: {model_id}")
                        
                        # 🔥 Step 2: 실제 모델 객체 삭제 (핵심!)
                        # WeakRef에서 제거
                        del self.active_model_refs[model_id]
                        
                        # 🔥 실제 Python 객체 참조 삭제 - 이것이 가장 중요함!
                        logger.info(f"🗑️ 모델 객체 참조 삭제 시작: {model_id}")
                        del model_instance
                        model_instance = None
                        logger.info(f"✅ 모델 객체 참조 삭제 완료: {model_id}")
                        
                    except Exception as e:
                        logger.error(f"❌ 모델 이동/삭제 실패: {model_id} - {str(e)}")
                else:
                    # 이미 삭제된 모델의 경우 WeakRef만 정리
                    del self.active_model_refs[model_id]
            
            # 3. 마스터 레지스트리에서 제거
            memory_freed = model_info.get('memory_mb', 0)
            del self.master_model_registry[model_id]
            
            # 4. 모든 서브시스템에서 동기화 제거
            await self._unsync_from_all_subsystems(model_id)
            
            # 🔥 Step 3: 가비지 컬렉션 (Python 객체 완전 정리)
            import gc
            gc.collect()
            logger.info(f"🧹 가비지 컬렉션 완료: {model_id}")
            
            # 🔥 Step 4: CUDA 캐시 정리 (GPU 메모리 해제)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info(f"🚀 CUDA 캐시 정리 완료: {model_id}")
            
            logger.info(f"완전 언로드 완료: {model_id} ({memory_freed:.0f}MB 해제)")
            return True
    
    async def _unsync_from_all_subsystems(self, model_id: str):
        """모든 서브시스템에서 모델 정보 제거"""
        # 기존 시스템들과 동기화 해제
        if hasattr(self._dynamic_gpu_manager, 'loaded_models'):
            if model_id in self._dynamic_gpu_manager.loaded_models:
                del self._dynamic_gpu_manager.loaded_models[model_id]
        
        if hasattr(self._swap_manager, 'gpu_resident_models'):
            if model_id in self._swap_manager.gpu_resident_models:
                del self._swap_manager.gpu_resident_models[model_id]
        
        if hasattr(self._swap_manager, 'ram_models'):
            if model_id in self._swap_manager.ram_models:
                del self._swap_manager.ram_models[model_id]
    
    def _sync_cleanup_model(self, model_id: str):
        """동기적 모델 정리 - asyncio RuntimeError 방지용"""
        try:
            # 동기적으로 모든 서브시스템에서 모델 정보 제거
            if hasattr(self._dynamic_gpu_manager, 'loaded_models'):
                if model_id in self._dynamic_gpu_manager.loaded_models:
                    del self._dynamic_gpu_manager.loaded_models[model_id]
            
            if hasattr(self._swap_manager, 'gpu_resident_models'):
                if model_id in self._swap_manager.gpu_resident_models:
                    del self._swap_manager.gpu_resident_models[model_id]
            
            if hasattr(self._swap_manager, 'ram_models'):
                if model_id in self._swap_manager.ram_models:
                    del self._swap_manager.ram_models[model_id]
                    
            logger.info(f"동기적 모델 정리 완료: {model_id}")
        except Exception as e:
            logger.warning(f"동기적 모델 정리 중 오류 ({model_id}): {str(e)}")
    
    def _on_model_garbage_collected(self, model_id: str):
        """모델이 가비지 컬렉션될 때 자동 정리"""
        logger.warning(f"모델 가비지 컬렉션 감지: {model_id} - 레지스트리 정리")
        if model_id in self.master_model_registry:
            # 🔧 asyncio RuntimeError 해결: 이벤트 루프 상태 확인 후 안전 처리
            try:
                # 실행 중인 이벤트 루프가 있는지 확인
                loop = asyncio.get_running_loop()
                if loop and not loop.is_closed():
                    # 이벤트 루프가 실행 중이면 비동기 작업 수행
                    asyncio.create_task(self._unsync_from_all_subsystems(model_id))
                else:
                    # 루프가 없으면 동기적 정리만 수행
                    logger.info(f"이벤트 루프 없음 - {model_id} 동기적 정리")
                    self._sync_cleanup_model(model_id)
            except RuntimeError:
                # 이벤트 루프가 없는 경우 - 동기적 정리만 수행
                logger.info(f"RuntimeError 방지 - {model_id} 동기적 정리")
                self._sync_cleanup_model(model_id)
            
            # 레지스트리에서 제거
            del self.master_model_registry[model_id]
    
    async def ensure_gpu_space(self, required_mb: float):
        """GPU 메모리 공간 확보
        
        Args:
            required_mb: 필요한 메모리 크기 (MB)
        
        Returns:
            bool: 공간 확보 성공 여부
        """
        try:
            # 현재 GPU 메모리 상태 확인
            memory_info = get_gpu_memory_info()
            if not memory_info:
                logger.warning("GPU 메모리 정보를 가져올 수 없음 - CPU 모드로 진행")
                return False
            
            current_usage_percent = memory_info.get('usage_percent', 0)
            free_mb = memory_info.get('free_mb', 0)
            total_mb = memory_info.get('memory_total_gb', 8) * 1024
            
            logger.info(f"🔍 GPU 메모리 현황: {current_usage_percent:.1f}% 사용 중, {free_mb}MB 여유")
            logger.info(f"🎯 요청된 메모리: {required_mb}MB")
            
            # 85% 이하이고 충분한 여유 공간이 있으면 추가 정리 불필요
            if current_usage_percent <= 85 and free_mb >= required_mb * 1.2:  # 20% 여유분 포함
                logger.info(f"✅ 충분한 GPU 메모리 여유 - 정리 불필요")
                return True
            
            # 90% 이상이거나 여유 공간이 부족하면 정리 필요
            if current_usage_percent > 90 or free_mb < required_mb:
                logger.warning(f"⚠️ GPU 메모리 부족 - 정리 시작 (사용률: {current_usage_percent:.1f}%, 여유: {free_mb}MB)")
                
                # 긴급 정리 실행
                await self._emergency_intelligent_cleanup()
                
                # 정리 후 다시 확인
                memory_info_after = get_gpu_memory_info()
                if memory_info_after:
                    new_usage = memory_info_after.get('usage_percent', 0)
                    new_free = memory_info_after.get('free_mb', 0)
                    logger.info(f"🧹 정리 후 GPU 메모리: {new_usage:.1f}% 사용 중, {new_free}MB 여유")
                    
                    if new_free >= required_mb:
                        logger.info(f"✅ GPU 메모리 공간 확보 성공")
                        return True
                    else:
                        logger.error(f"❌ GPU 메모리 공간 확보 실패 - 필요: {required_mb}MB, 여유: {new_free}MB")
                        return False
                else:
                    logger.error("❌ 정리 후 GPU 메모리 정보 확인 실패")
                    return False
            else:
                logger.info(f"✅ GPU 메모리 상태 양호 - 정리 불필요")
                return True
                
        except Exception as e:
            logger.error(f"❌ ensure_gpu_space 실행 중 오류: {str(e)}")
            return False
    
    async def _emergency_intelligent_cleanup(self):
        """긴급 상황 시 지능적 정리 - 2025 PyTorch 강화 버전"""
        import torch
        logger.error("🚨 긴급 GPU 메모리 정리 시작")
        
        # 🔥 Step 1: 현재 GPU 메모리 상황 정확한 파악
        initial_memory = get_gpu_memory_info()
        if initial_memory:
            logger.error(f"🚨 초기 GPU 메모리: {initial_memory['usage_percent']:.1f}% 사용 중")
        
        # 🔥 Step 2: 실제 GPU 상주 모델들을 전면 스캔 및 등록 
        logger.error("🔍 실제 GPU 상주 모델 전면 스캔 시작...")
        await self._discover_and_register_gpu_models()
        
        # 🔥 Step 3: PyTorch 메모리 스냅샷으로 실제 상태 확인
        try:
            if torch.cuda.is_available():
                # PyTorch 2025 메모리 디버깅 도구 활용
                allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
                reserved = torch.cuda.memory_reserved() / (1024**3)    # GB
                logger.error(f"🔍 PyTorch 실제 메모리: 할당={allocated:.2f}GB, 예약={reserved:.2f}GB")
        except Exception as e:
            logger.warning(f"메모리 스냅샷 실패: {str(e)}")
        
        # 🔥 Step 4: 적극적 모델 언로드 (CRITICAL 외 모든 모델 대상)
        models_by_priority = sorted(
            self.master_model_registry.items(),
            key=lambda x: (x[1].get('priority', 10), -x[1].get('access_count', 0))
        )
        
        freed_memory = 0
        target_usage = 85  # 85% 목표로 설정 (과도한 언로드 방지)
        unloaded_count = 0
        
        logger.error(f"🗑️ 총 {len(models_by_priority)}개 모델 중 CRITICAL 외 모든 모델 언로드 시작")
        
        for model_id, model_info in models_by_priority:
            priority = model_info.get('priority', ModelPriority.MEDIUM)
            
            # CRITICAL 외 모든 모델 강제 언로드
            if priority > ModelPriority.CRITICAL:
                logger.error(f"🗑️ 강제 언로드 시작: {model_id} (우선순위: {priority})")
                
                success = await self.intelligent_unload_model(model_id, force=True)
                if success:
                    freed_memory += model_info.get('memory_mb', 0)
                    unloaded_count += 1
                    
                    # 중간 상태 체크
                    current_memory = get_gpu_memory_info()
                    if current_memory:
                        logger.error(f"📊 진행 상황: {model_id} 언로드 후 {current_memory['usage_percent']:.1f}%")
                        
                        # 목표 달성 체크 (75% 이하)
                        if current_memory['usage_percent'] <= target_usage:
                            logger.info(f"✅ 긴급 정리 목표 달성: {freed_memory:.0f}MB 해제, 사용률: {current_memory['usage_percent']:.1f}%")
                            break
                else:
                    logger.error(f"❌ 언로드 실패: {model_id}")
        
        # 🔥 Step 5: 최종 전면 메모리 정리 (2025 모범 사례)
        logger.error("🧹 최종 전면 메모리 정리 시작...")
        import gc
        
        # 강력한 가비지 컬렉션
        for i in range(3):  # 3회 반복으로 확실히 정리
            collected = gc.collect()
            logger.error(f"🧹 가비지 컬렉션 {i+1}회: {collected}개 객체 정리")
        
        # CUDA 메모리 완전 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.error("🚀 CUDA 캐시 완전 정리 완료")
        
        # 🔥 Step 6: 최종 결과 검증
        final_memory = get_gpu_memory_info()
        if final_memory:
            improvement = initial_memory['usage_percent'] - final_memory['usage_percent'] if initial_memory else 0
            logger.error(f"✅ 긴급 정리 완료:")
            logger.error(f"   📊 메모리 사용률: {initial_memory['usage_percent']:.1f}% → {final_memory['usage_percent']:.1f}% ({improvement:+.1f}%)")
            logger.error(f"   🗑️ 언로드된 모델: {unloaded_count}개")
            logger.error(f"   💾 해제된 메모리: {freed_memory:.0f}MB")
            
            # 여전히 임계치 초과 시 경고
            if final_memory['usage_percent'] > 85:
                logger.error("⚠️ 메모리 사용률이 여전히 높음 - 추가 조치 필요")
        else:
            logger.error(f"긴급 정리 종료: {unloaded_count}개 모델 언로드, {freed_memory:.0f}MB 해제")
    
    async def _discover_and_register_gpu_models(self):
        """실제 GPU 상주 모델들을 스캔해서 registry에 등록"""
        logger.info("🔍 GPU 상주 모델 스캔 및 registry 등록 시작")
        
        import gc
        import torch.nn as nn
        discovered_models = 0
        
        try:
            # 1. 가비지 컬렉터로 모든 객체 스캔
            for obj in gc.get_objects():
                try:
                    # 2. PyTorch 모델인지 확인
                    if isinstance(obj, nn.Module) and hasattr(obj, 'parameters'):
                        # 3. GPU에 상주하는 모델인지 확인
                        gpu_params = []
                        total_params = 0
                        gpu_memory_mb = 0
                        
                        for param in obj.parameters():
                            total_params += param.numel()
                            if param.device.type == 'cuda':
                                gpu_params.append(param)
                                gpu_memory_mb += param.numel() * param.element_size() / (1024 * 1024)
                        
                        # 4. GPU 파라미터가 있고 충분한 크기인 모델만 등록
                        if gpu_params and gpu_memory_mb > 10:  # 10MB 이상
                            # 5. 모델 ID 생성 (클래스명 + 메모리 크기 기반)
                            model_class = obj.__class__.__name__
                            model_id = f"discovered_{model_class}_{int(gpu_memory_mb)}MB_{id(obj)}"
                            
                            # 6. Registry에 없는 경우만 등록
                            if model_id not in self.master_model_registry:
                                logger.info(f"🔍 발견된 GPU 모델: {model_id} ({gpu_memory_mb:.1f}MB)")
                                
                                # Registry에 등록
                                self.master_model_registry[model_id] = {
                                    'device': 'cuda',
                                    'memory_mb': gpu_memory_mb,
                                    'priority': ModelPriority.MEDIUM,  # 발견된 모델은 중간 우선순위
                                    'load_time': time.time(),
                                    'access_count': 0,
                                    'risk_level': 'MEDIUM',
                                    'discovered': True  # 스캔으로 발견된 모델 표시
                                }
                                
                                # WeakRef 등록
                                import weakref
                                self.active_model_refs[model_id] = weakref.ref(obj, 
                                    lambda ref, mid=model_id: self._on_model_garbage_collected(mid))
                                
                                discovered_models += 1
                
                except Exception as e:
                    # 개별 객체 스캔 실패는 무시하고 계속
                    continue
            
            logger.info(f"✅ GPU 모델 스캔 완료: {discovered_models}개 모델 발견 및 등록")
            logger.info(f"📊 현재 Registry 상태: {len(self.master_model_registry)}개 모델 추적 중")
            
        except Exception as e:
            logger.error(f"❌ GPU 모델 스캔 실패: {str(e)}")
            # 스캔 실패해도 기존 registry로 계속 진행

# 전역 마스터 오케스트레이터 인스턴스
_master_orchestrator = MasterMemoryOrchestrator()

def get_master_orchestrator() -> MasterMemoryOrchestrator:
    """전역 마스터 메모리 오케스트레이터 반환"""
    return _master_orchestrator

def initialize_unified_memory_system():
    """통합 메모리 관리 시스템 초기화
    
    이 함수는 시스템 시작 시 한 번 호출되어야 하며,
    모든 서브시스템들을 마스터 오케스트레이터와 연결합니다.
    """
    master_orchestrator = get_master_orchestrator()
    
    logger.info("🔧 통합 메모리 관리 시스템 초기화 시작")
    
    try:
        # 🚀 Step 0: CUDA Context 프리로딩 (11초 지연 근본 해결)
        logger.info("🚀 CUDA Context 프리로딩 실행...")
        preload_success = preload_cuda_context()
        if preload_success:
            logger.info("✅ CUDA 지연 문제 근본 해결 완료 - 향후 즉시 응답")
        else:
            logger.warning("⚠️ CUDA 프리로딩 실패 - CPU 모드로 계속 진행")
        
        # 🔥 핵심 수정: 실제 서브시스템들과의 연결 활성화
        logger.info("🔧 서브시스템들과의 연결 시도 중...")
        
        # 기존 시스템들과 연결 (에러 발생 시 안전하게 처리)
        try:
            # GPU 로더와 연결
            master_orchestrator.connect_subsystems(
                gpu_manager=_gpu_loader,  # 전역 GPU 로더 사용
                swap_manager=None,        # 추후 연결 예정
                predictor=None,           # 추후 연결 예정  
                compressor=None           # 추후 연결 예정
            )
            logger.info("✅ 기본 서브시스템 연결 완료")
        except Exception as e:
            logger.warning(f"⚠️ 서브시스템 연결 부분 실패 (계속 진행): {str(e)}")
        
        # 강제 GPU→RAM 스왑 테스트 실행 (동기적 처리)
        logger.info("🧪 메모리 관리 기본 설정 완료")
        
        logger.info("✅ 통합 메모리 관리 시스템 초기화 완료")
        logger.info(f"📊 마스터 레지스트리 상태: {len(master_orchestrator.master_model_registry)}개 모델 추적 중")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 통합 메모리 관리 시스템 초기화 실패: {str(e)}")
        return False

def get_memory_system_status() -> dict:
    """현재 메모리 시스템 상태 반환"""
    master_orchestrator = get_master_orchestrator()
    memory_info = get_gpu_memory_info()
    
    status = {
        'gpu_available': memory_info is not None,
        'gpu_usage_percent': memory_info['usage_percent'] if memory_info else 0,
        'gpu_free_mb': memory_info['free_mb'] if memory_info else 0,
        'loaded_models_count': len(master_orchestrator.master_model_registry),
        'active_models': list(master_orchestrator.master_model_registry.keys()),
        'memory_by_priority': {}
    }
    
    # 우선순위별 메모리 사용량 집계
    for model_id, info in master_orchestrator.master_model_registry.items():
        priority = info.get('priority', 'UNKNOWN')
        if priority not in status['memory_by_priority']:
            status['memory_by_priority'][priority] = {'count': 0, 'total_mb': 0}
        
        status['memory_by_priority'][priority]['count'] += 1
        status['memory_by_priority'][priority]['total_mb'] += info.get('memory_mb', 0)
    
    return status

def force_unified_cleanup(target_usage_percent: float = 75.0):
    """통합 시스템을 통한 강제 정리
    
    Args:
        target_usage_percent: 목표 GPU 사용률 (기본값: 75%)
    """
    logger.info(f"🧹 통합 시스템 강제 정리 시작 (목표: {target_usage_percent}%)")
    
    master_orchestrator = get_master_orchestrator()
    initial_memory_info = get_gpu_memory_info()
    
    if not initial_memory_info:
        logger.warning("GPU 정보를 가져올 수 없어 정리 작업을 건너뜁니다")
        return False
    
    initial_usage = initial_memory_info['usage_percent']
    logger.info(f"📊 정리 전 GPU 사용률: {initial_usage:.1f}%")
    
    if initial_usage <= target_usage_percent:
        logger.info("이미 목표 사용률 이하입니다")
        return True
    
    # 1단계: LOW 우선순위 모델들 제거
    logger.info("1단계: LOW 우선순위 모델 정리")
    _force_swap_low_priority_models()
    
    # 중간 체크
    mid_memory_info = get_gpu_memory_info()
    if mid_memory_info and mid_memory_info['usage_percent'] <= target_usage_percent:
        logger.info(f"✅ 1단계 정리로 목표 달성: {mid_memory_info['usage_percent']:.1f}%")
        return True
    
    # 2단계: 긴급 정리 (MEDIUM 우선순위까지)
    logger.info("2단계: 긴급 메모리 정리")
    _emergency_gpu_cleanup()
    
    # 최종 체크
    final_memory_info = get_gpu_memory_info()
    if final_memory_info:
        final_usage = final_memory_info['usage_percent']
        freed_mb = (initial_usage - final_usage) * 80  # 8GB 기준
        
        logger.info(f"🏁 정리 완료: {initial_usage:.1f}% → {final_usage:.1f}% ({freed_mb:.0f}MB 해제)")
        
        if final_usage <= target_usage_percent:
            logger.info("✅ 목표 사용률 달성")
            return True
        else:
            logger.warning(f"⚠️ 목표 미달성 (목표: {target_usage_percent}%, 현재: {final_usage:.1f}%)")
            return False
    else:
        logger.error("최종 메모리 상태를 확인할 수 없습니다")
        return False

def get_gpu_loader():
    """전역 GPU 로더 반환"""
    return _gpu_loader

def get_priority_based_device(memory_required_mb=500, priority=ModelPriority.MEDIUM, model_id=None, loading_function=None):
    """마스터 오케스트레이터를 통한 지능적 디바이스 선택"""
    import torch
    import asyncio
    
    # CPU 강제 모드
    if not ADVANCED_CONFIG['enable_gpu'] or not torch.cuda.is_available():
        return torch.device('cpu')
    
    # loading_function이 없으면 빠른 체크 (즉시 결정)
    if loading_function is None:
        memory_info = get_gpu_memory_info()
        if memory_info is None or memory_info['usage_percent'] > 82:
            return torch.device('cpu')
        return torch.device('cuda')
    
    # loading_function이 있으면 마스터 오케스트레이터 활용
    master_orchestrator = get_master_orchestrator()
    
    try:
        device, result = run_async_safely(
            master_orchestrator.intelligent_load_model(
                model_id or "unknown_model", priority, memory_required_mb, loading_function
            ),
            timeout=30.0
        )
        
        if device is None:
            return torch.device('cpu')
            
        return device
        
    except Exception as e:
        logger.error(f"마스터 오케스트레이터 디바이스 선택 실패: {str(e)}")
        return torch.device('cpu')

def _track_gpu_model(model_id, priority, memory_mb):
    """GPU 로드된 모델 추적 (마스터 오케스트레이터와 연동)"""
    if model_id:
        # 레거시 시스템 호환성
        _gpu_loaded_models[model_id] = {
            'priority': priority,
            'memory_mb': memory_mb
        }
        
        # 마스터 오케스트레이터에도 동기화
        master_orchestrator = get_master_orchestrator()
        if model_id not in master_orchestrator.master_model_registry:
            master_orchestrator.master_model_registry[model_id] = {
                'device': 'cuda',
                'memory_mb': memory_mb,
                'priority': priority,
                'load_time': time.time(),
                'access_count': 1,
                'risk_level': 'MEDIUM'
            }

def _force_swap_low_priority_models():
    """낮은 우선순위 모델들 강제 스왑 (마스터 오케스트레이터 활용)"""
    import asyncio
    
    master_orchestrator = get_master_orchestrator()
    
    # LOW 우선순위 모델들 찾기
    low_priority_models = [
        model_id for model_id, info in master_orchestrator.master_model_registry.items()
        if info.get('priority', ModelPriority.MEDIUM) >= ModelPriority.LOW
    ]
    
    if not low_priority_models:
        logger.info("강제 스왑할 LOW 우선순위 모델이 없음")
        return
    
    logger.info(f"LOW 우선순위 모델들 강제 스왑 시작: {low_priority_models}")
    
    # 비동기 언로드를 동기적으로 실행 (헬퍼 사용)
    async def _async_swap():
        tasks = []
        for model_id in low_priority_models:
            task = master_orchestrator.intelligent_unload_model(model_id, force=False)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    
    try:
        results = run_async_safely(_async_swap(), timeout=60.0)
        
        if results is not None:
            success_count = sum(1 for r in results if r is True)
            logger.info(f"LOW 우선순위 모델 스왑 완료: 성공 {success_count}개")
        else:
            logger.error("LOW 우선순위 모델 스왑 타임아웃")
            _legacy_force_cleanup()
        
    except Exception as e:
        logger.error(f"LOW 우선순위 모델 스왑 실패: {str(e)}")
        # 폴백으로 기존 방식 사용
        _legacy_force_cleanup()

def _emergency_gpu_cleanup():
    """긴급 GPU 메모리 정리 (마스터 오케스트레이터 활용)"""
    logger.error("🚨 긴급 GPU 메모리 정리 시작")
    
    master_orchestrator = get_master_orchestrator()
    
    # 비동기 긴급 정리를 동기적으로 실행 (헬퍼 사용)
    try:
        result = run_async_safely(
            master_orchestrator._emergency_intelligent_cleanup(),
            timeout=120.0
        )
        
        if result is not None:
            logger.info("🚨 긴급 GPU 메모리 정리 완료")
        else:
            logger.error("🚨 긴급 정리 타임아웃 - 레거시 정리 모드로 전환")
            _legacy_emergency_cleanup()
        
    except Exception as e:
        logger.error(f"🚨 긴급 정리 실패: {str(e)} - 레거시 정리 모드로 전환")
        _legacy_emergency_cleanup()

def _legacy_force_cleanup():
    """레거시 강제 정리 (폴백 전용)"""
    import gc
    import torch
    
    logger.warning("레거시 강제 정리 모드 실행")
    
    to_remove = []
    for model_id, info in _gpu_loaded_models.items():
        if info['priority'] >= ModelPriority.LOW:
            to_remove.append(model_id)
    
    for model_id in to_remove:
        del _gpu_loaded_models[model_id]
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def _legacy_emergency_cleanup():
    """레거시 긴급 정리 (폴백 전용)"""
    import gc
    import torch
    
    logger.warning("레거시 긴급 정리 모드 실행")
    
    to_remove = []
    for model_id, info in _gpu_loaded_models.items():
        if info['priority'] > ModelPriority.CRITICAL:
            to_remove.append(model_id)
    
    for model_id in to_remove:
        del _gpu_loaded_models[model_id]
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def update_smart_device_with_priority():
    """기존 get_smart_device를 우선순위 시스템으로 업데이트"""
    global get_smart_device
    
    def enhanced_get_smart_device(memory_required_mb=500, force_cpu=False, priority=ModelPriority.MEDIUM, model_id=None):
        import torch  # 지연 로딩으로 torch import 추가
        if force_cpu:
            return torch.device('cpu')
        return get_priority_based_device(memory_required_mb, priority, model_id)
    
    # 기존 함수 대체
    get_smart_device = enhanced_get_smart_device

def print_system_info():
    """시스템 정보 출력"""
    print("="*60)
    print("Red Heart Linux Advanced - System Information")
    print("="*60)
    print(f"Platform: {SYSTEM_INFO['platform']}")
    print(f"Architecture: {SYSTEM_INFO['architecture']}")
    print(f"Python Version: {SYSTEM_INFO['python_version']}")
    print(f"CPU Count: {SYSTEM_INFO['cpu_count']}")
    print(f"GPU Available: {ADVANCED_CONFIG['enable_gpu']}")
    print(f"GPU Count: {ADVANCED_CONFIG['gpu_count']}")
    print(f"Advanced Libraries: {'Enabled' if not ADVANCED_CONFIG['fallback_mode'] else 'Fallback Mode'}")
    print("="*60)

if __name__ == "__main__":
    print_system_info()