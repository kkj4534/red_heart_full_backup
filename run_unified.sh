#!/bin/bash
# Red Heart AI 통합 시스템 실행 스크립트
# Unified System with Precision Mapping, Idle Learning, and Benchmarking
# 730M+ 모델 with 정밀 감정→벤담 매핑 통합

set -e  # 오류 발생 시 즉시 종료

echo "🚀 Red Heart AI 통합 시스템 v2.0"
echo "==========================================="
echo "   730M+ 파라미터 모델"
echo "   정밀 감정→벤담 매핑 시스템 통합"
echo "   유휴 시간 학습 시스템 포함"
echo "==========================================="

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 함수 정의
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_module() {
    echo -e "${PURPLE}[MODULE]${NC} $1"
}

print_feature() {
    echo -e "${CYAN}[FEATURE]${NC} $1"
}

# 현재 디렉토리 확인
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

print_status "작업 디렉토리: $SCRIPT_DIR"

# 가상환경 활성화 함수
activate_environment() {
    print_status "가상환경 확인 중..."
    
    # red_heart_env 우선 확인
    if [ -f "red_heart_env/bin/activate" ]; then
        source red_heart_env/bin/activate
        print_success "✅ red_heart_env 가상환경 활성화"
        print_status "   Python: $(which python)"
        print_status "   Python 버전: $(python --version 2>&1)"
    elif [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        print_success "✅ venv 가상환경 활성화"
        print_status "   Python: $(which python)"
    else
        print_warning "⚠️ 가상환경 없음 - 시스템 Python 사용"
        print_warning "   권장: python3 -m venv red_heart_env"
    fi
}

# 환경 검증 함수
check_environment() {
    print_status "환경 검증 중..."
    
    # Python 버전 확인
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    print_status "Python 버전: $python_version"
    
    # 필수 패키지 확인
    local missing_packages=()
    
    # PyTorch 확인
    if python3 -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null; then
        print_success "✅ PyTorch 설치됨"
        
        # GPU 확인
        if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
            gpu_info=$(python3 -c "import torch; print(f'{torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB)')" 2>/dev/null)
            print_success "✅ GPU 사용 가능: $gpu_info"
            export DEVICE="cuda"
        else
            print_warning "⚠️ GPU 없음 - CPU 모드로 실행"
            export DEVICE="cpu"
        fi
    else
        missing_packages+=("torch")
    fi
    
    # Transformers 확인
    if python3 -c "import transformers" 2>/dev/null; then
        print_success "✅ Transformers 설치됨"
    else
        missing_packages+=("transformers")
    fi
    
    # NumPy 확인 (정밀 매핑에 필수)
    if python3 -c "import numpy" 2>/dev/null; then
        print_success "✅ NumPy 설치됨"
    else
        missing_packages+=("numpy")
    fi
    
    # Sentence Transformers 확인 (선택)
    if python3 -c "import sentence_transformers" 2>/dev/null; then
        print_success "✅ Sentence Transformers 설치됨"
    else
        print_warning "⚠️ Sentence Transformers 없음 (선택 사항)"
    fi
    
    # 누락된 패키지 처리
    if [ ${#missing_packages[@]} -gt 0 ]; then
        print_error "❌ 필수 패키지 누락: ${missing_packages[*]}"
        print_warning "다음 명령어로 설치하세요:"
        print_warning "pip install ${missing_packages[*]}"
        return 1
    fi
    
    # 체크포인트 확인
    if [ -d "training/checkpoints_final" ]; then
        latest_checkpoint=$(ls -t training/checkpoints_final/*.pt 2>/dev/null | head -n1)
        if [ -n "$latest_checkpoint" ]; then
            print_success "✅ 체크포인트 발견: $(basename $latest_checkpoint)"
        else
            print_warning "⚠️ 체크포인트 없음 - 새로운 가중치 사용"
        fi
    fi
    
    return 0
}

# 모듈 상태 체크 함수
check_modules() {
    print_status "핵심 모듈 상태 확인 (초기 로딩 최대 60초)..."
    echo ""
    
    # 충분한 시간을 주고 정확한 체크 수행 (NO FALLBACK)
    if [ -f "check_modules.py" ]; then
        # 60초 타임아웃 - 초기 CUDA/torch 로딩에 충분한 시간
        timeout 60 python3 check_modules.py
        
        # 실패하면 실패 (NO FALLBACK)
        if [ $? -ne 0 ]; then
            print_error "❌ 모듈 체크 실패 - 필수 모듈이 누락되었습니다"
            print_error "   필요한 패키지를 설치하세요:"
            print_error "   pip install numpy torch transformers"
            exit 1
        fi
    else
        print_error "❌ check_modules.py 파일이 없습니다"
        print_error "   시스템 무결성이 손상되었습니다"
        exit 1
    fi
    
    echo ""
}

# HuggingFace 오프라인 모드 설정
setup_offline_mode() {
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_VERBOSITY=error
    export HF_HUB_DISABLE_TELEMETRY=1
    print_status "🔐 HuggingFace 오프라인 모드 활성화"
}

# 메인 실행 함수
run_unified_system() {
    local mode="${1:-inference}"
    shift 2>/dev/null || true
    
    print_status "Red Heart AI 통합 시스템 실행..."
    print_status "실행 모드: $mode"
    echo ""
    
    case "$mode" in
        # ================== 기본 추론 모드 ==================
        "inference"|"infer")
            print_module "🎯 추론 모드 - 730M+ 전체 모델"
            print_feature "✨ 정밀 감정→벤담 매핑 활성화"
            if [ -n "$1" ]; then
                python main_unified.py --text "$@"
            else
                python main_unified.py "$@"
            fi
            ;;
            
        # ================== 대화형 모드 ==================
        "interactive"|"chat")
            print_module "💬 대화형 모드"
            print_feature "✨ 실시간 분석 + 정밀 매핑"
            print_status "   종료: 'quit' 또는 Ctrl+C"
            python main_unified.py --mode interactive "$@"
            ;;
            
        # ================== 운용 모드 ==================
        "production"|"prod")
            print_module "🚀 운용 모드"
            print_feature "✨ 모든 최적화 활성화"
            print_feature "✨ 유휴 학습 시스템 활성화"
            python main_unified.py --mode production "$@"
            ;;
            
        # ================== 벤치마크 모드 ==================
        "benchmark"|"bench")
            print_module "📊 벤치마크 모드"
            print_feature "✨ 성능 측정 및 비교"
            
            # 벤치마크 스크립트 실행
            if [ -f "benchmark_unified.py" ]; then
                python benchmark_unified.py "$@"
            else
                print_warning "벤치마크 스크립트 없음 - 간단한 테스트 실행"
                for i in {1..5}; do
                    echo -n "   테스트 $i/5... "
                    time python main_unified.py --text "벤치마크 테스트 $i" 2>&1 | tail -n1
                done
            fi
            ;;
            
        # ================== 테스트 모드 ==================
        "test")
            print_module "🧪 통합 테스트 모드"
            print_feature "✨ 모든 컴포넌트 검증"
            
            if [ -f "test_final_integration.py" ]; then
                python test_final_integration.py "$@"
            else
                python main_unified.py --text "테스트 문장" --verbose "$@"
            fi
            ;;
            
        # ================== 메모리 모드별 실행 ==================
        "minimal"|"light"|"normal"|"heavy"|"ultra"|"extreme")
            print_module "🎚️ 메모리 모드: $mode"
            
            case "$mode" in
                "minimal")
                    print_feature "최소 모드 - UnifiedModel만"
                    export MEMORY_MODE="minimal"
                    ;;
                "light")
                    print_feature "경량 모드 - +DSP, Kalman"
                    export MEMORY_MODE="light"
                    ;;
                "normal")
                    print_feature "일반 모드 - +정밀 매퍼, Phase"
                    export MEMORY_MODE="normal"
                    ;;
                "heavy")
                    print_feature "고성능 모드 - +Neural Analyzers, 유휴 학습"
                    export MEMORY_MODE="heavy"
                    ;;
                "ultra")
                    print_feature "울트라 모드 - +Advanced Wrappers, 시계열"
                    export MEMORY_MODE="ultra"
                    ;;
                "extreme")
                    print_feature "익스트림 모드 - 모든 모듈 + 신경망 어댑터"
                    export MEMORY_MODE="extreme"
                    ;;
            esac
            
            python main_unified.py --memory-mode $mode "$@"
            ;;
            
        # ================== LLM 통합 모드 ==================
        "llm-local")
            print_module "🤖 로컬 LLM 통합 모드"
            print_feature "HelpingAI 9B + 정밀 매핑"
            python main_unified.py --llm local "$@"
            ;;
            
        "llm-claude")
            print_module "🤖 Claude API 통합 모드"
            print_feature "Claude API + 캐싱 최적화"
            python main_unified.py --llm claude "$@"
            ;;
            
        # ================== 유휴 학습 모드 ==================
        "idle"|"idle-learning")
            print_module "🌙 유휴 학습 시스템"
            print_feature "계층적 유휴 학습 활성화"
            python -c "
from idle_time_learner import HierarchicalIdleLearner
from main_unified import UnifiedInferenceSystem, InferenceConfig
import asyncio

async def run_idle():
    config = InferenceConfig()
    system = UnifiedInferenceSystem(config)
    await system.initialize()
    
    learner = HierarchicalIdleLearner(system.unified_model, config)
    await learner.start()
    print('유휴 학습 시스템 실행 중... (Ctrl+C로 종료)')
    
    try:
        await asyncio.sleep(86400)  # 24시간
    except KeyboardInterrupt:
        await learner.stop()
        print('유휴 학습 종료')

asyncio.run(run_idle())
" "$@"
            ;;
            
        # ================== 정밀 매핑 테스트 ==================
        "mapping"|"precision")
            print_module "🎯 정밀 매핑 테스트"
            print_feature "감정→벤담 의미론적 매핑 검증"
            
            if [ -f "test_emotion_bentham_mapping.py" ]; then
                python test_emotion_bentham_mapping.py "$@"
            else
                print_warning "매핑 테스트 스크립트 없음"
            fi
            ;;
            
        # ================== 시스템 상태 ==================
        "status"|"info")
            print_module "📈 시스템 상태"
            
            # GPU 상태
            if command -v nvidia-smi &> /dev/null; then
                print_status "GPU 상태:"
                nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
                    --format=csv,noheader,nounits | while read line; do
                    echo "      $line"
                done
            fi
            
            # 메모리 상태
            print_status "시스템 메모리:"
            free -h | grep "^Mem:" | awk '{print "      전체: "$2", 사용: "$3", 여유: "$4}'
            
            # 모듈 체크
            check_modules
            ;;
            
        # ================== 설정 ==================
        "setup"|"install")
            print_module "🔧 환경 설정"
            
            # 가상환경 생성
            if [ ! -d "red_heart_env" ]; then
                print_status "가상환경 생성 중..."
                python3 -m venv red_heart_env
            fi
            
            source red_heart_env/bin/activate
            
            print_status "필수 패키지 설치 안내:"
            echo "      pip install numpy torch transformers"
            echo "      pip install sentence-transformers matplotlib seaborn"
            echo "      pip install pandas jinja2 markdown"
            
            print_warning "프로젝트 규칙에 따라 자동 설치는 불가합니다."
            print_warning "위 명령어를 수동으로 실행해주세요."
            ;;
            
        # ================== 도움말 ==================
        "help"|"-h"|"--help")
            show_unified_help
            ;;
            
        *)
            print_warning "알 수 없는 모드: $mode"
            print_status "추론 모드로 실행합니다."
            python main_unified.py --text "$mode" "$@"
            ;;
    esac
}

# 도움말 함수
show_unified_help() {
    echo ""
    echo "🚀 Red Heart AI 통합 시스템 사용법 v2.0"
    echo "========================================"
    echo ""
    echo "기본 사용법:"
    echo "  $0 [모드] [옵션...] [--text \"분석할 텍스트\"]"
    echo ""
    echo "🎯 주요 모드:"
    echo "  inference         # 기본 추론 (정밀 매핑 포함)"
    echo "  interactive       # 대화형 모드"
    echo "  production        # 운용 모드 (유휴 학습 포함)"
    echo "  benchmark         # 성능 벤치마크"
    echo "  test             # 통합 테스트"
    echo ""
    echo "🎚️ 메모리 모드:"
    echo "  minimal          # 최소 (UnifiedModel만)"
    echo "  light            # 경량 (+DSP, Kalman)"
    echo "  normal           # 일반 (+정밀 매퍼)"
    echo "  heavy            # 고성능 (+Neural Analyzers, 유휴 학습)"
    echo "  ultra            # 울트라 (+Advanced Wrappers)"
    echo "  extreme          # 익스트림 (모든 모듈)"
    echo ""
    echo "✨ 새로운 기능:"
    echo "  mapping          # 정밀 감정→벤담 매핑 테스트"
    echo "  idle             # 유휴 학습 시스템 실행"
    echo "  status           # 시스템 상태 확인"
    echo ""
    echo "🤖 LLM 통합:"
    echo "  llm-local        # 로컬 LLM (HelpingAI)"
    echo "  llm-claude       # Claude API"
    echo ""
    echo "💡 예시:"
    echo "  $0 inference --text \"분석할 텍스트\""
    echo "  $0 interactive"
    echo "  $0 production --text \"텍스트\" --verbose"
    echo "  $0 benchmark --samples 100"
    echo "  $0 heavy --text \"고성능 분석\""
    echo "  $0 mapping  # 매핑 테스트"
    echo ""
    echo "📊 시스템 구성:"
    echo "  - UnifiedModel: 730M"
    echo "  - 정밀 감정→벤담 매퍼 (의미론적)"
    echo "  - 유휴 시간 학습 (5단계)"
    echo "  - 벤치마크 시스템"
    echo "  - Neural Analyzers: 368M"
    echo "  - Advanced Wrappers: 112M"
    echo ""
    echo "🔧 고급 옵션:"
    echo "  --checkpoint PATH # 체크포인트 경로"
    echo "  --batch-size N   # 배치 크기"
    echo "  --device cuda/cpu# 디바이스 선택"
    echo "  --no-neural      # Neural Analyzers 비활성화"
    echo "  --no-wrappers    # Advanced Wrappers 비활성화"
    echo "  --no-dsp         # DSP Simulator 비활성화"
    echo "  --llm MODE       # LLM 모드"
    echo "  --verbose        # 상세 로그"
    echo "  --debug          # 디버그 모드"
    echo ""
    echo "📈 예상 성능 (NORMAL 모드):"
    echo "  - 지연시간: ~200ms"
    echo "  - 처리량: 5 req/s"
    echo "  - VRAM: 4.5GB"
    echo "  - 정확도: 85%+"
    echo ""
}

# 정리 함수
cleanup() {
    print_status "시스템 정리 중..."
    
    # GPU 메모리 정리
    if command -v nvidia-smi &> /dev/null; then
        python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    fi
    
    # 임시 파일 정리
    find . -name "*.pyc" -delete 2>/dev/null || true
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    
    print_success "정리 완료"
}

# 신호 처리
trap cleanup EXIT
trap 'print_warning "실행 중단됨"; cleanup; exit 130' INT TERM

# 메인 실행
main() {
    local mode="${1:-help}"
    
    # 도움말 요청 처리
    case "$mode" in
        --help|-h|help)
            show_unified_help
            exit 0
            ;;
    esac
    
    echo ""
    print_success "🎯 Red Heart AI 통합 시스템 v2.0"
    print_status "   정밀 감정→벤담 매핑 통합"
    print_status "   유휴 시간 학습 시스템 포함"
    echo ""
    
    # 환경 활성화
    activate_environment
    
    # 환경 검증
    if ! check_environment; then
        print_error "❌ 환경 검증 실패"
        print_warning "필수 패키지를 설치하세요:"
        print_warning "pip install numpy torch transformers"
        exit 1
    fi
    
    echo ""
    
    # 모듈 체크
    check_modules
    
    # 오프라인 모드 설정
    setup_offline_mode
    
    echo "==========================================="
    echo ""
    
    # 시스템 실행
    run_unified_system "$@"
    
    echo ""
    print_success "🎉 Red Heart AI 통합 시스템 완료!"
    echo ""
}

# 스크립트 실행
main "$@"