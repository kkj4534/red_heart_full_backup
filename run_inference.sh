#!/bin/bash
# Red Heart AI 통합 추론 시스템 실행 스크립트
# Unified Inference System Execution Script
# 50 epoch으로 학습된 730M 모델 운용

set -e  # 오류 발생 시 즉시 종료

echo "🚀 Red Heart AI 통합 추론 시스템 시작"
echo "==========================================="
echo "   730M 파라미터 모델 (50 epoch 학습 완료)"
echo "==========================================="

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
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

# 현재 디렉토리 확인
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

print_status "작업 디렉토리: $SCRIPT_DIR"

# 환경 검증 함수
check_environment() {
    print_status "환경 검증 중..."
    
    # Python 버전 확인
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    print_status "Python 버전: $python_version"
    
    # PyTorch 확인
    if python3 -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null; then
        print_success "✅ PyTorch 설치됨"
        
        # GPU 확인
        if python3 -c "import torch; print('CUDA' if torch.cuda.is_available() else 'CPU')" | grep -q "CUDA"; then
            gpu_info=$(python3 -c "import torch; print(f'{torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB)')" 2>/dev/null)
            print_success "✅ GPU 사용 가능: $gpu_info"
        else
            print_warning "⚠️ GPU 없음 - CPU 모드로 실행됩니다"
        fi
    else
        print_error "❌ PyTorch가 설치되지 않았습니다"
        return 1
    fi
    
    # Transformers 확인
    if python3 -c "import transformers" 2>/dev/null; then
        print_success "✅ Transformers 설치됨"
    else
        print_error "❌ Transformers가 설치되지 않았습니다"
        return 1
    fi
    
    # 체크포인트 확인
    checkpoint_path="training/checkpoints_final/checkpoint_epoch_0050_lr_0.000009_20250827_154403.pt"
    if [ -f "$checkpoint_path" ]; then
        checkpoint_size=$(du -h "$checkpoint_path" | cut -f1)
        print_success "✅ 50 epoch 체크포인트 확인 ($checkpoint_size)"
    else
        print_warning "⚠️ 50 epoch 체크포인트 없음 - 다른 체크포인트 검색 중..."
        
        # 대체 체크포인트 찾기
        if [ -d "training/checkpoints_final" ]; then
            latest_checkpoint=$(ls -t training/checkpoints_final/*.pt 2>/dev/null | head -n1)
            if [ -n "$latest_checkpoint" ]; then
                print_status "대체 체크포인트 발견: $(basename $latest_checkpoint)"
            else
                print_warning "체크포인트 없음 - 새로운 가중치로 시작합니다"
            fi
        fi
    fi
    
    return 0
}

# 가상환경 활성화 함수
activate_environment() {
    print_status "가상환경 확인 중..."
    
    # venv 환경 확인
    if [ -f "red_heart_env/bin/activate" ]; then
        source red_heart_env/bin/activate
        print_success "✅ red_heart_env 가상환경 활성화"
        print_status "   Python: $(which python)"
    else
        print_warning "가상환경 없음 - 시스템 Python 사용"
    fi
}

# 모듈 상태 체크 함수
check_modules() {
    print_status "핵심 모듈 상태 확인..."
    echo ""
    
    # Python에서 모듈 체크
    python3 << 'EOF'
import sys
import os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'training'))

modules_status = {
    "UnifiedModel (730M)": False,
    "Neural Analyzers (368M)": False,
    "Advanced Wrappers (112M)": False,
    "DSP Simulator (14M)": False,
    "Phase Networks (4.3M)": False,
    "LLM Engine": False
}

# UnifiedModel 체크
try:
    from training.unified_training_final import UnifiedModel
    modules_status["UnifiedModel (730M)"] = True
except: pass

# Neural Analyzers 체크
try:
    from analyzer_neural_modules import create_neural_analyzers
    modules_status["Neural Analyzers (368M)"] = True
except: pass

# Advanced Wrappers 체크
try:
    from advanced_analyzer_wrappers import create_advanced_analyzer_wrappers
    modules_status["Advanced Wrappers (112M)"] = True
except: pass

# DSP Simulator 체크
try:
    from emotion_dsp_simulator import EmotionDSPSimulator
    modules_status["DSP Simulator (14M)"] = True
except: pass

# Phase Networks 체크
try:
    from phase_neural_networks import Phase0ProjectionNet
    modules_status["Phase Networks (4.3M)"] = True
except: pass

# LLM Engine 체크
try:
    from llm_module.advanced_llm_engine import AdvancedLLMEngine
    modules_status["LLM Engine"] = True
except: pass

# 결과 출력
total = len(modules_status)
available = sum(modules_status.values())

for module, status in modules_status.items():
    icon = "✅" if status else "❌"
    print(f"   {icon} {module}")

print("")
print(f"   📊 모듈 가용성: {available}/{total} ({available*100//total}%)")

if available < 3:
    print("   ⚠️ 핵심 모듈이 부족합니다. 기본 모드로 실행됩니다.")
EOF
    
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

# 추론 시스템 실행 함수
run_inference_system() {
    local mode="${1:-inference}"
    shift 2>/dev/null || true
    
    print_status "Red Heart AI 추론 시스템 실행..."
    print_status "실행 모드: $mode"
    echo ""
    
    case "$mode" in
        # ================== 기본 추론 모드 ==================
        "inference"|"infer")
            print_module "🎯 추론 모드 - 730M 전체 모델 사용"
            print_status "   - UnifiedModel + 모든 보조 모듈"
            print_status "   - 50 epoch 체크포인트 로드"
            if [ -n "$1" ]; then
                python main_unified.py --mode inference --text "$@"
            else
                python main_unified.py --mode inference "$@"
            fi
            ;;
            
        # ================== 테스트 모드 ==================
        "test")
            print_module "🧪 테스트 모드"
            print_status "   - 3개 샘플로 시스템 검증"
            print_status "   - 모든 모듈 작동 확인"
            python main_unified.py --mode test --verbose "$@"
            ;;
            
        # ================== 데모 모드 ==================
        "demo"|"interactive")
            print_module "🎮 대화형 데모 모드"
            print_status "   - 실시간 텍스트 분석"
            print_status "   - 종료: quit 입력"
            python main_unified.py --mode demo "$@"
            ;;
            
        # ================== 운용 모드 ==================
        "production"|"prod")
            print_module "🚀 운용 모드"
            print_status "   - 완전한 730M 모델"
            print_status "   - 모든 최적화 활성화"
            python main_unified.py --mode production "$@"
            ;;
            
        # ================== 고급 모드 (모듈별 제어) ==================
        "advanced"|"custom")
            print_module "⚙️ 고급 모드 - 모듈 선택적 활성화"
            echo ""
            echo "사용 가능한 옵션:"
            echo "   --no-neural    : Neural Analyzers 비활성화"
            echo "   --no-wrappers  : Advanced Wrappers 비활성화"
            echo "   --no-dsp       : DSP Simulator 비활성화"
            echo "   --no-phase     : Phase Networks 비활성화"
            echo "   --llm local    : 로컬 LLM 활성화"
            echo "   --llm claude   : Claude API 활성화"
            echo ""
            python main_unified.py --mode inference "$@"
            ;;
            
        # ================== LLM 통합 모드 ==================
        "llm-local")
            print_module "🤖 로컬 LLM 통합 모드"
            print_status "   - HelpingAI 9B 모델 사용"
            print_status "   - 4-bit 양자화"
            python main_unified.py --mode production --llm local "$@"
            ;;
            
        "llm-claude")
            print_module "🤖 Claude API 통합 모드"
            print_status "   - Claude API 캐싱 활용"
            print_status "   - 90% 비용 절감"
            python main_unified.py --mode production --llm claude "$@"
            ;;
            
        "llm-mcp")
            print_module "🔌 MCP 프로토콜 모드"
            print_status "   - Model Context Protocol 지원"
            print_status "   - (개발 중)"
            python main_unified.py --mode production --llm mcp "$@"
            ;;
            
        # ================== 경량 모드 ==================
        "light"|"fast")
            print_module "⚡ 경량 모드"
            print_status "   - 기본 UnifiedModel만 사용"
            print_status "   - 빠른 추론 속도"
            python main_unified.py --mode inference \
                --no-neural --no-wrappers --no-dsp --no-phase "$@"
            ;;
            
        # ================== 벤치마크 모드 ==================
        "benchmark"|"bench")
            print_module "📊 벤치마크 모드"
            print_status "   - 성능 측정"
            print_status "   - 처리 시간 분석"
            
            echo ""
            echo "벤치마크 시작..."
            
            # 10개 샘플로 벤치마크
            for i in {1..10}; do
                echo -n "   테스트 $i/10... "
                time_start=$(date +%s.%N)
                python main_unified.py --mode inference \
                    --text "벤치마크 테스트 텍스트 $i" \
                    --verbose 2>/dev/null | tail -n1
                time_end=$(date +%s.%N)
                elapsed=$(echo "$time_end - $time_start" | bc)
                echo "완료 (${elapsed}초)"
            done
            ;;
            
        # ================== 모니터링 모드 ==================
        "monitor"|"status")
            print_module "📈 시스템 모니터링"
            
            # GPU 상태
            if command -v nvidia-smi &> /dev/null; then
                print_status "GPU 상태:"
                nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
                    --format=csv,noheader,nounits | while read line; do
                    echo "   $line"
                done
            fi
            
            # 메모리 상태
            print_status "시스템 메모리:"
            free -h | grep "^Mem:" | awk '{print "   전체: "$2", 사용: "$3", 여유: "$4}'
            
            # 프로세스 상태
            print_status "Python 프로세스:"
            ps aux | grep python | grep -E "(main_unified|unified_training)" | head -n3
            ;;
            
        # ================== 도움말 ==================
        "help"|"-h"|"--help")
            show_inference_help
            ;;
            
        *)
            print_warning "알 수 없는 모드: $mode"
            print_status "기본 추론 모드로 실행합니다."
            python main_unified.py --mode inference "$mode" "$@"
            ;;
    esac
}

# 도움말 함수
show_inference_help() {
    echo ""
    echo "🚀 Red Heart AI 통합 추론 시스템 사용법"
    echo "========================================"
    echo ""
    echo "기본 사용법:"
    echo "  $0 [모드] [옵션...] [--text \"분석할 텍스트\"]"
    echo ""
    echo "🎯 주요 모드:"
    echo "  inference, infer   # 기본 추론 (730M 전체)"
    echo "  test              # 시스템 테스트"
    echo "  demo              # 대화형 데모"
    echo "  production        # 운용 모드"
    echo ""
    echo "🤖 LLM 통합:"
    echo "  llm-local         # 로컬 LLM (HelpingAI 9B)"
    echo "  llm-claude        # Claude API"
    echo "  llm-mcp           # MCP 프로토콜 (개발 중)"
    echo ""
    echo "⚙️ 특수 모드:"
    echo "  advanced          # 모듈 선택적 활성화"
    echo "  light             # 경량 모드 (빠른 추론)"
    echo "  benchmark         # 성능 벤치마크"
    echo "  monitor           # 시스템 모니터링"
    echo ""
    echo "💡 예시:"
    echo "  $0 inference --text \"분석할 텍스트\""
    echo "  $0 demo"
    echo "  $0 test --verbose"
    echo "  $0 production --text \"운용 텍스트\" --llm local"
    echo "  $0 light --text \"빠른 분석\""
    echo "  $0 advanced --no-dsp --no-phase --text \"선택적 분석\""
    echo ""
    echo "📊 모델 구성 (730M):"
    echo "  - UnifiedModel: 243.6M (Backbone + 4 Heads)"
    echo "  - Neural Analyzers: 368M"
    echo "  - Advanced Wrappers: 112M"
    echo "  - DSP Simulator: 14M"
    echo "  - Kalman Filter: 2.3M"
    echo "  - Phase Networks: 4.3M"
    echo ""
    echo "🔧 고급 옵션:"
    echo "  --checkpoint PATH  # 체크포인트 경로"
    echo "  --batch-size N     # 배치 크기"
    echo "  --device cuda/cpu  # 디바이스 선택"
    echo "  --no-neural        # Neural Analyzers 비활성화"
    echo "  --no-wrappers      # Advanced Wrappers 비활성화"
    echo "  --no-dsp           # DSP Simulator 비활성화"
    echo "  --no-phase         # Phase Networks 비활성화"
    echo "  --llm MODE         # LLM 모드 (none/local/claude/mcp)"
    echo "  --verbose          # 상세 로그"
    echo "  --debug            # 디버그 모드"
    echo ""
}

# 정리 함수
cleanup_inference() {
    print_status "시스템 정리 중..."
    
    # GPU 메모리 정리
    if command -v nvidia-smi &> /dev/null; then
        python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    fi
    
    print_success "정리 완료"
}

# 신호 처리
trap cleanup_inference EXIT
trap 'print_warning "추론 중단됨"; cleanup_inference; exit 130' INT TERM

# 메인 실행
main() {
    local mode="${1:-help}"
    
    # 도움말 요청 처리
    case "$mode" in
        --help|-h|help)
            show_inference_help
            exit 0
            ;;
    esac
    
    echo ""
    print_success "🎯 Red Heart AI 통합 추론 시스템"
    print_status "   730M 파라미터 / 50 epoch 학습 완료"
    echo ""
    
    # 환경 활성화
    activate_environment
    
    # 환경 검증
    if ! check_environment; then
        print_error "환경 검증 실패"
        exit 1
    fi
    
    echo ""
    
    # 모듈 체크
    check_modules
    
    # 오프라인 모드 설정
    setup_offline_mode
    
    echo "==========================================="
    echo ""
    
    # 추론 시스템 실행
    run_inference_system "$@"
    
    echo ""
    print_success "🎉 Red Heart AI 추론 완료!"
    echo ""
}

# 스크립트 실행
main "$@"