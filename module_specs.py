"""
Red Heart AI 모듈 스펙 정의
각 모듈의 메타데이터를 중앙 집중식으로 관리
"""

# 모듈 초기화 우선순위 및 예상 메모리 사용량 (MB)
# translator를 먼저 초기화하여 다른 모듈들이 전역에서 찾을 수 있도록 함
# emotion_analyzer를 두 번째로 초기화하여 HeadCompatibilityManager 등록 보장
MODULE_SPECS = [
    {
        'name': 'translator',  # 가장 먼저 초기화 - 다른 모듈들의 의존성
        'class_path': 'local_translator.LocalTranslator',
        'category': 'translation',
        'estimated_mb': 900,  # OPUS-MT 모델 실제 크기 (로그 기준)
        'priority': 'HIGH',  # HIGH - 중요하지만 필요시 언로드 가능
        'timeout': 120,  # 모델 로드 시간 고려하여 대폭 증가 (2분)
        'device_policy': 'gpu_on_demand',  # 초기 CPU 로드, 필요시 GPU 승격 (메모리 피크 억제)
        'required': True  # 필수 모듈
    },
    {
        'name': 'emotion_analyzer',  # 두 번째 초기화 - HeadCompatibilityManager 등록 보장
        'class_path': 'advanced_emotion_analyzer.AdvancedEmotionAnalyzer',
        'category': 'emotion',
        'estimated_mb': 700,  # 초기 예측용 추정치 (실측 전까지 사용)
        'priority': 'HIGH',  # HIGH - 언로드 가능
        'timeout': 120,  # 다중 모델 로드 시간 고려하여 대폭 증가 (2분)
        'needs_initialize': True,  # 비동기 초기화 필수
        'device_policy': 'gpu_on_demand',  # CPU 프리로드 후 DSM으로 GPU 승격 (초기화 후 즉시 필요)
        'required': True  # 필수 모듈
    },
    {
        'name': 'bentham_calculator', 
        'class_path': 'advanced_bentham_calculator.AdvancedBenthamCalculator',
        'category': 'bentham',
        'estimated_mb': 200,  # 상대적으로 가벼움
        'priority': 'HIGH',  # HIGH - 중요하지만 언로드 가능
        'timeout': 60,  # 안전하게 1분으로 증가
        'needs_initialize': True,  # 비동기 초기화 필수
        'device_policy': 'gpu_on_demand',  # CPU 사전 로드 가능
        'required': True  # 필수 모듈
    },
    {
        'name': 'semantic_analyzer',
        'class_path': 'advanced_multi_level_semantic_analyzer.AdvancedMultiLevelSemanticAnalyzer', 
        'category': 'semantic',
        'estimated_mb': 700,  # sentence-transformers 기반
        'priority': 'HIGH',  # HIGH - 중요하지만 언로드 가능
        'timeout': 120,  # 대용량 모델 고려하여 대폭 증가 (2분)
        'needs_initialize': True,  # 비동기 초기화 필수
        'device_policy': 'gpu_on_demand',  # CPU 프리로드 후 DSM으로 GPU 승격
        'required': True  # 필수 모듈
    },
    {
        'name': 'regret_analyzer',
        'class_path': 'advanced_regret_analyzer.AdvancedRegretAnalyzer',
        'category': 'regret', 
        'estimated_mb': 300,
        'priority': 'MEDIUM',  # MEDIUM - 선택적 모듈
        'timeout': 60,  # 안전하게 1분으로 증가
        'needs_initialize': True,  # 비동기 초기화 필수
        'device_policy': 'gpu_on_demand',  # CPU 프리로드 후 DSM으로 GPU 승격
        'required': False  # 선택 모듈
    },
    {
        'name': 'neural_components',
        'class_path': 'missing_neural_models.HierarchicalPatternStructure',
        'category': 'neural',
        'estimated_mb': 150,
        'priority': 'LOW',  # LOW - 우선순위 낮음
        'timeout': 60,  # 안전하게 1분으로 증가
        'device_policy': 'gpu_on_demand',  # CPU 사전 로드 가능
        'required': False  # 선택 모듈
    },
    {
        'name': 'meta_integration',
        'class_path': 'advanced_meta_integration_system.AdvancedMetaIntegrationSystem',
        'category': 'meta_integration',
        'estimated_mb': 200,  # 40M 파라미터 모델
        'priority': 'MEDIUM',  # MEDIUM - 선택적 모듈
        'timeout': 60,  # 안전하게 1분으로 증가
        'device_policy': 'gpu_on_demand',  # CPU 사전 로드 가능
        'required': False  # 선택 모듈
    },
    {
        'name': 'surd_analyzer',
        'class_path': 'advanced_surd_analyzer.AdvancedSurdAnalyzer',
        'category': 'surd',
        'estimated_mb': 250,
        'priority': 'MEDIUM',  # MEDIUM - 선택적
        'timeout': 60,
        'device_policy': 'gpu_on_demand',  # CPU 프리로드 후 DSM으로 GPU 승격
        'required': False,  # 선택 모듈
        'retry_on_error': 1  # 오류 시 1회 재시도
    },
    {
        'name': 'bayesian_engine',
        'class_path': 'advanced_bayesian_inference_module.AdvancedBayesianInferenceModule',
        'category': 'inference',
        'estimated_mb': 150,
        'priority': 'LOW',  # LOW - 우선순위 낮음
        'timeout': 60,
        'device_policy': 'gpu_on_demand',  # CPU 사전 로드 가능
        'required': False,  # 선택 모듈
        'retry_on_error': 1  # 오류 시 1회 재시도
    },
    {
        'name': 'llm_engine',
        'class_path': 'llm_module.advanced_llm_engine.AdvancedLLMEngine',
        'category': 'llm',
        'estimated_mb': 800,  # LLM 모델 크기
        'priority': 'LOW',  # LOW - 메모리 큼
        'timeout': 180,  # 3분
        'device_policy': 'gpu_on_demand',  # CPU 프리로드 후 DSM으로 GPU 승격
        'required': False  # 선택 모듈
    },
    {
        'name': 'experience_database',
        'class_path': 'advanced_experience_database.AdvancedExperienceDatabase',
        'category': 'database',
        'estimated_mb': 100,
        'priority': 'LOW',  # LOW - 우선순위 낮음
        'timeout': 60,
        'device_policy': 'gpu_on_demand',  # CPU 사전 로드 가능
        'required': False  # 선택 모듈
    },
    {
        'name': 'hierarchical_emotion',
        'class_path': 'advanced_hierarchical_emotion_system.AdvancedHierarchicalEmotionSystem',
        'category': 'emotion',
        'estimated_mb': 350,
        'priority': 'MEDIUM',  # MEDIUM - 선택적
        'timeout': 90,
        'device_policy': 'gpu_on_demand',  # CPU 프리로드 후 DSM으로 GPU 승격
        'required': False  # 선택 모듈
    },
    {
        'name': 'usage_pattern_analyzer',
        'class_path': 'advanced_usage_pattern_analyzer.AdvancedUsagePatternAnalyzer',
        'category': 'pattern',
        'estimated_mb': 200,
        'priority': 'LOW',  # LOW - 우선순위 낮음
        'timeout': 60,
        'device_policy': 'gpu_on_demand',  # CPU 사전 로드 가능
        'required': False  # 선택 모듈
    }
]

# 모듈 이름으로 빠르게 검색하기 위한 인덱스
MODULE_SPECS_INDEX = {spec['name']: spec for spec in MODULE_SPECS}

def get_module_spec(module_name: str) -> dict:
    """모듈 이름으로 스펙 조회"""
    return MODULE_SPECS_INDEX.get(module_name, None)

def get_module_estimated_mb(module_name: str) -> float:
    """모듈의 예상 메모리 사용량 조회"""
    spec = get_module_spec(module_name)
    return spec.get('estimated_mb', 0) if spec else 0