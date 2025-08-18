#!/bin/bash

# Red Heart Linux 실행 스크립트
# Advanced Ethical Decision Support System for Linux

set -e  # 오류 발생 시 스크립트 중단

echo "🔴❤️ Red Heart 고급 윤리적 의사결정 지원 시스템 (Linux)"
echo "==============================================================="

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# 현재 디렉토리 확인
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

print_status "작업 디렉토리: $SCRIPT_DIR"

# Python 및 가상환경 확인
check_python() {
    print_status "Python 환경 확인 중..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1)
        print_success "Python 발견: $PYTHON_VERSION"
    else
        print_error "Python3가 설치되어 있지 않습니다."
        exit 1
    fi
    
    # 가상환경 활성화
    if [ -d "red_heart_env" ]; then
        print_status "Red Heart 가상환경 활성화 중..."
        source red_heart_env/bin/activate
        print_success "가상환경 활성화됨: $(which python)"
    elif [ -d "venv" ]; then
        print_status "기본 가상환경 활성화 중..."
        source venv/bin/activate
        print_success "가상환경 활성화됨"
    else
        print_warning "가상환경이 없습니다. 시스템 Python을 사용합니다."
        print_status "권장사항: ./run_red_heart.sh setup 으로 환경을 설정하세요."
    fi
}

# GPU/CUDA 확인
check_gpu() {
    print_status "GPU/CUDA 환경 확인 중..."
    
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
        GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        print_success "GPU 발견: $GPU_INFO (${GPU_MEMORY}MB)"
        
        # CUDA 버전 확인
        if command -v nvcc &> /dev/null; then
            CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
            print_success "CUDA 버전: $CUDA_VERSION"
        else
            print_warning "CUDA 컴파일러(nvcc)를 찾을 수 없습니다."
        fi
    else
        print_warning "NVIDIA GPU가 감지되지 않았습니다. CPU 모드로 실행됩니다."
    fi
}

# 의존성 확인
check_dependencies() {
    print_status "Python 패키지 의존성 확인 중..."
    
    if [ -f "requirements.txt" ]; then
        # 주요 패키지 확인
        CRITICAL_PACKAGES=("torch" "transformers" "sentence-transformers" "sklearn" "numpy")
        
        for package in "${CRITICAL_PACKAGES[@]}"; do
            # 패키지명 매핑 (pip 이름 → import 이름)
            import_name="$package"
            if [ "$package" = "sentence-transformers" ]; then
                import_name="sentence_transformers"
            elif [ "$package" = "sklearn" ]; then
                import_name="sklearn"
            fi
            
            if python3 -c "import $import_name" 2>/dev/null; then
                PACKAGE_VERSION=$(python3 -c "import $import_name; print(getattr($import_name, '__version__', 'unknown'))" 2>/dev/null || echo "버전 정보 없음")
                print_success "$package 설치됨 (버전: $PACKAGE_VERSION)"
            else
                print_error "$package가 설치되어 있지 않습니다."
                print_status "의존성 설치를 위해 다음 명령을 실행하세요:"
                print_status "pip install -r requirements.txt"
                exit 1
            fi
        done
    else
        print_warning "requirements.txt 파일을 찾을 수 없습니다."
    fi
}

# 모델 디렉토리 확인
check_models() {
    print_status "모델 디렉토리 확인 중..."
    
    if [ ! -d "models" ]; then
        print_status "모델 디렉토리 생성 중..."
        mkdir -p models/{emotion_models,hierarchical_emotion,regret_models,semantic_models,semantic_cache,surd_cache,surd_models}
        print_success "모델 디렉토리 생성됨"
    else
        print_success "모델 디렉토리 존재함"
    fi
    
    # 로그 디렉토리 확인
    if [ ! -d "logs" ]; then
        print_status "로그 디렉토리 생성 중..."
        mkdir -p logs
        print_success "로그 디렉토리 생성됨"
    fi
}

# 메모리 사용량 확인
check_memory() {
    print_status "시스템 메모리 확인 중..."
    
    TOTAL_MEM=$(free -m | awk 'NR==2{printf "%.1f", $2/1024}')
    AVAILABLE_MEM=$(free -m | awk 'NR==2{printf "%.1f", $7/1024}')
    
    print_success "총 메모리: ${TOTAL_MEM}GB, 사용 가능: ${AVAILABLE_MEM}GB"
    
    # 최소 메모리 요구사항 확인 (4GB)
    if (( $(echo "$AVAILABLE_MEM < 4.0" | bc -l) )); then
        print_warning "사용 가능한 메모리가 4GB 미만입니다. 성능이 저하될 수 있습니다."
    fi
}

# 시스템 정보 출력
print_system_info() {
    echo ""
    print_status "시스템 정보:"
    echo "  OS: $(uname -s) $(uname -r)"
    echo "  아키텍처: $(uname -m)"
    echo "  CPU: $(nproc) 코어"
    echo "  Python: $(python3 --version 2>&1)"
    
    if command -v nvidia-smi &> /dev/null; then
        echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
    fi
    echo ""
}

# 메인 실행 함수
run_main() {
    local mode="${1:-help}"
    shift 2>/dev/null || true
    
    print_status "Red Heart AI 시스템 시작 중..."
    
    case "$mode" in
        "learn"|"learning"|"train"|"training")
            print_status "학습 시스템 실행..."
            ./run_learning.sh "$@"
            ;;
        "test"|"testing")
            print_status "테스트 시스템 실행..."
            ./run_test.sh "$@"
            ;;
        "env"|"environment"|"status")
            print_status "시스템 무결성 검사..."
            python3 system_integrity_checker.py
            ;;
        "demo"|"main")
            print_status "기본 시스템 실행..."
            if [ -f "main.py" ]; then
                python3 main.py "$@"
            else
                print_warning "main.py를 찾을 수 없습니다. 테스트 모드로 실행합니다."
                ./run_test.sh quick "$@"
            fi
            ;;
        "setup"|"install")
            print_status "환경 설정..."
            setup_environment
            ;;
        *)
            show_help
            ;;
    esac
}

# 도움말 출력
show_help() {
    echo "❤️  Red Heart AI 마스터 실행 스크립트"
    echo "==========================================="
    echo ""
    echo "사용법: $0 <명령어> [옵션...]"
    echo ""
    echo "명령어:"
    echo "  learn, learning     # 학습 시스템 실행"
    echo "  test, testing       # 테스트 시스템 실행"
    echo "  env, environment    # 환경 상태 확인"
    echo "  setup, install      # 초기 환경 설정"
    echo "  demo, main          # 기본 시스템 실행"
    echo "  --check-only        # 시스템 확인만 수행"
    echo "  help, --help, -h    # 이 도움말 표시"
    echo ""
    echo "예시:"
    echo "  $0 setup             # 초기 환경 설정"
    echo "  $0 learn             # 기본 학습 실행"
    echo "  $0 learn complete    # 완전한 학습 시스템 실행"
    echo "  $0 test quick        # 빠른 테스트"
    echo "  $0 test integrated   # 통합 테스트"
    echo "  $0 env               # 환경 상태 확인"
    echo "  $0 --check-only      # 시스템 확인만"
    echo ""
    echo "고급 사용법:"
    echo "  $0 learn --samples 100 --verbose"
    echo "  $0 test baseline --log-level DEBUG"
    echo ""
    echo "문제 해결:"
    echo "  1. 환경 설정: $0 setup"
    echo "  2. 환경 확인: $0 env"
    echo "  3. 빠른 테스트: $0 test quick"
    echo ""
}

# 환경 설정 함수
setup_environment() {
    print_status "Red Heart AI 환경 설정 시작..."
    
    # 가상환경 생성
    if [ ! -d "red_heart_env" ]; then
        print_status "가상환경 생성 중..."
        python3 -m venv red_heart_env
        print_success "가상환경 생성 완료"
    fi
    
    # 가상환경 활성화
    source red_heart_env/bin/activate
    print_success "가상환경 활성화 완료"
    
    # pip 업그레이드
    print_status "pip 업그레이드 중..."
    pip install --upgrade pip
    
    # 필수 패키지 설치
    print_status "필수 패키지 설치 중..."
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        # 기본 패키지 설치
        pip install torch torchvision transformers numpy scipy scikit-learn
        pip install matplotlib seaborn pandas psutil python-dotenv json_repair tqdm
    fi
    
    # 스크립트 실행 권한 설정
    chmod +x run_learning.sh run_test.sh system_integrity_checker.py
    
    print_success "환경 설정 완료!"
    print_status "이제 다음 명령어로 시스템을 실행할 수 있습니다:"
    print_status "  ./run_red_heart.sh learn    # 학습 시스템"
    print_status "  ./run_red_heart.sh test     # 테스트 시스템"
    print_status "  ./run_red_heart.sh env      # 환경 상태 확인"
}

# 정리 함수
cleanup() {
    print_status "정리 작업 수행 중..."
    
    # 임시 파일 정리
    find . -name "*.pyc" -delete 2>/dev/null || true
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    
    # 손상된 패키지 정리
    if [ -d "red_heart_env/lib/python3.12/site-packages" ]; then
        rm -rf red_heart_env/lib/python3.12/site-packages/~* 2>/dev/null || true
    fi
    
    # GPU 메모리 정리 (nvidia-smi가 있는 경우)
    if command -v nvidia-smi &> /dev/null; then
        print_status "GPU 메모리 정리 중..."
        # 여기에 GPU 메모리 정리 코드 추가 가능
    fi
    
    print_success "정리 작업 완료"
}

# 신호 처리 (Ctrl+C 등)
trap cleanup EXIT
trap 'print_warning "중단됨"; exit 130' INT TERM

# 메인 실행 로직
main() {
    # 인수 파싱
    case "${1:-}" in
        --help|-h)
            show_help
            exit 0
            ;;
        --check-only)
            print_status "시스템 확인 모드"
            check_python
            check_gpu
            check_dependencies
            check_models
            check_memory
            print_system_info
            print_success "모든 시스템 확인 완료!"
            exit 0
            ;;
    esac
    
    # 시스템 확인
    check_python
    check_gpu
    check_dependencies
    check_models
    check_memory
    print_system_info
    
    # 메인 프로그램 실행
    run_main "$@"
}

# 스크립트 실행
main "$@"