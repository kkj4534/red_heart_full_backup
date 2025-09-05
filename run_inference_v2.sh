#!/bin/bash
# Red Heart AI 통합 추론 시스템 v2.0 실행 스크립트
# 5단계 메모리 모드 지원 (922M 전체 통합)
# Created: 2025-08-28

set -e  # 오류 발생 시 즉시 종료

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 기본 설정
MEMORY_MODE="auto"
ACTION_MODE="inference"
TEXT=""
BATCH_SIZE=4
USE_CACHE=true
DEBUG=false

# 함수 정의
print_header() {
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║       ${GREEN}Red Heart AI 통합 추론 시스템 v2.0${CYAN}          ║${NC}"
    echo -e "${CYAN}║         ${YELLOW}922M 파라미터 전체 통합 모드${CYAN}             ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════╝${NC}"
    echo ""
}

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

# 사용법 출력
usage() {
    print_header
    echo -e "${CYAN}사용법:${NC} $0 [메모리모드] [동작모드] [옵션]"
    echo ""
    echo -e "${PURPLE}━━━ 메모리 모드 ━━━${NC}"
    echo "  minimal    - 90M  (Backbone만)"
    echo "  light      - 230M (+ Heads)"
    echo "  normal     - 400M (+ DSP/Kalman)"
    echo "  heavy      - 600M (+ Neural Analyzers)"
    echo "  ultra      - 842M (+ Advanced Analyzers)"
    echo "  extreme    - 922M (+ Meta/Regret/CF 전체)"
    echo "  auto       - GPU 메모리 자동 감지"
    echo ""
    echo -e "${CYAN}━━━ 동작 모드 ━━━${NC}"
    echo "  inference  - 텍스트 분석"
    echo "  test       - 시스템 테스트"
    echo "  demo       - 데모 실행"
    echo "  benchmark  - 성능 벤치마크"
    echo ""
    echo -e "${YELLOW}━━━ 옵션 ━━━${NC}"
    echo "  --text \"텍스트\"     - 분석할 텍스트"
    echo "  --batch-size N      - 배치 크기 (기본: 4)"
    echo "  --no-cache          - 캐시 비활성화"
    echo "  --debug             - 디버그 모드"
    echo ""
    echo -e "${GREEN}예시:${NC}"
    echo "  $0 auto inference --text \"오늘 기분이 좋아요\""
    echo "  $0 extreme test"
    echo "  $0 normal benchmark --batch-size 8"
    echo ""
    exit 0
}

# GPU 메모리 확인
check_gpu_memory() {
    python3 -c "
import torch
if torch.cuda.is_available():
    mb = torch.cuda.get_device_properties(0).total_memory // (1024**2)
    print(mb)
else:
    print(0)
" 2>/dev/null
}

# 메모리 모드 자동 선택
auto_select_memory_mode() {
    local gpu_mem=$(check_gpu_memory)
    print_status "GPU 메모리: ${gpu_mem}MB"
    
    if [ $gpu_mem -eq 0 ]; then
        print_warning "GPU 없음 - MINIMAL 모드 선택"
        echo "minimal"
    elif [ $gpu_mem -lt 3000 ]; then
        echo "minimal"
    elif [ $gpu_mem -lt 4000 ]; then
        echo "light"
    elif [ $gpu_mem -lt 5000 ]; then
        echo "normal"
    elif [ $gpu_mem -lt 6000 ]; then
        echo "heavy"
    elif [ $gpu_mem -lt 7000 ]; then
        echo "ultra"
    else
        echo "extreme"
    fi
}

# 파라미터 파싱
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            # 메모리 모드
            minimal|light|normal|heavy|ultra|extreme|auto)
                MEMORY_MODE="$1"
                shift
                ;;
            # 동작 모드
            inference|infer|test|demo|benchmark)
                ACTION_MODE="$1"
                shift
                ;;
            # 옵션
            --text)
                TEXT="$2"
                shift 2
                ;;
            --batch-size)
                BATCH_SIZE="$2"
                shift 2
                ;;
            --no-cache)
                USE_CACHE=false
                shift
                ;;
            --debug)
                DEBUG=true
                shift
                ;;
            --help|-h)
                usage
                ;;
            *)
                print_error "알 수 없는 옵션: $1"
                usage
                ;;
        esac
    done
}

# 환경 검증
check_environment() {
    print_status "환경 검증 중..."
    
    # Python 버전 확인
    if ! python3 --version &>/dev/null; then
        print_error "Python3가 설치되지 않았습니다"
        exit 1
    fi
    
    # PyTorch 확인
    if ! python3 -c "import torch" 2>/dev/null; then
        print_error "PyTorch가 설치되지 않았습니다"
        exit 1
    fi
    
    # GPU 정보 출력
    if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        gpu_info=$(python3 -c "import torch; print(f'{torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB)')")
        print_success "✅ GPU: $gpu_info"
    else
        print_warning "⚠️ GPU 없음 - CPU 모드"
    fi
    
    print_success "✅ 환경 검증 완료"
}

# 추론 실행
run_inference() {
    local memory_mode="$1"
    local text="$2"
    
    print_module "추론 모드 실행"
    print_status "메모리 모드: $memory_mode"
    print_status "텍스트: $text"
    
    python3 main_unified.py \
        --mode inference \
        --memory-mode "$memory_mode" \
        --text "$text" \
        --batch-size "$BATCH_SIZE" \
        $([ "$USE_CACHE" = false ] && echo "--no-cache") \
        $([ "$DEBUG" = true ] && echo "--debug")
}

# 테스트 실행
run_test() {
    local memory_mode="$1"
    
    print_module "테스트 모드 실행"
    print_status "메모리 모드: $memory_mode"
    
    python3 test_unified_integration.py \
        --memory-mode "$memory_mode" \
        --batch-size "$BATCH_SIZE" \
        $([ "$DEBUG" = true ] && echo "--verbose")
}

# 데모 실행
run_demo() {
    local memory_mode="$1"
    
    print_module "데모 모드 실행"
    print_status "메모리 모드: $memory_mode"
    
    # 데모 텍스트 목록
    demo_texts=(
        "오늘 날씨가 정말 좋아서 기분이 좋아요"
        "시험에 떨어져서 너무 속상합니다"
        "친구들과 함께 시간을 보내니 행복해요"
        "미래가 불확실해서 걱정됩니다"
        "새로운 프로젝트를 시작하게 되어 설레요"
    )
    
    for text in "${demo_texts[@]}"; do
        echo ""
        print_status "분석: $text"
        run_inference "$memory_mode" "$text"
        sleep 2
    done
}

# 벤치마크 실행
run_benchmark() {
    local memory_mode="$1"
    
    print_module "벤치마크 모드 실행"
    print_status "메모리 모드: $memory_mode"
    
    python3 benchmark_unified.py \
        --memory-mode "$memory_mode" \
        --batch-size "$BATCH_SIZE" \
        --num-iterations 10
}

# 메인 실행
main() {
    # 현재 디렉토리로 이동
    cd "$(dirname "${BASH_SOURCE[0]}")"
    
    print_header
    
    # 인자 파싱
    parse_arguments "$@"
    
    # auto 모드일 경우 자동 선택
    if [ "$MEMORY_MODE" = "auto" ]; then
        MEMORY_MODE=$(auto_select_memory_mode)
        print_success "자동 선택된 메모리 모드: $MEMORY_MODE"
    fi
    
    # 환경 검증
    check_environment
    
    # 메모리 모드별 예상 사용량 출력
    case $MEMORY_MODE in
        minimal)
            print_status "메모리 사용: ~90M (Backbone만)"
            ;;
        light)
            print_status "메모리 사용: ~230M (+ Heads)"
            ;;
        normal)
            print_status "메모리 사용: ~400M (+ DSP/Kalman)"
            ;;
        heavy)
            print_status "메모리 사용: ~600M (+ Neural)"
            ;;
        ultra)
            print_status "메모리 사용: ~842M (+ Advanced)"
            ;;
        extreme)
            print_status "메모리 사용: ~922M (전체 통합)"
            ;;
    esac
    
    echo ""
    print_status "========== 실행 시작 =========="
    
    # 동작 모드별 실행
    case $ACTION_MODE in
        inference|infer)
            if [ -z "$TEXT" ]; then
                print_error "텍스트를 입력해주세요 (--text \"텍스트\")"
                exit 1
            fi
            run_inference "$MEMORY_MODE" "$TEXT"
            ;;
        test)
            run_test "$MEMORY_MODE"
            ;;
        demo)
            run_demo "$MEMORY_MODE"
            ;;
        benchmark)
            run_benchmark "$MEMORY_MODE"
            ;;
        *)
            print_error "알 수 없는 동작 모드: $ACTION_MODE"
            exit 1
            ;;
    esac
    
    echo ""
    print_success "========== 실행 완료 =========="
}

# 실행
main "$@"