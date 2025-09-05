#!/bin/bash
# Red Heart AI 모드별 테스트 스크립트
# MD 문서 사양에 따른 메모리 모드 테스트

set -e

echo "🚀 Red Heart AI 모드별 테스트"
echo "==========================================="
date

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

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

# 스크립트 디렉토리
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 가상환경 활성화
activate_environment() {
    if [ -f "red_heart_env/bin/activate" ]; then
        source red_heart_env/bin/activate
        print_success "✅ red_heart_env 가상환경 활성화"
    elif [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        print_success "✅ venv 가상환경 활성화"
    else
        print_error "❌ 가상환경 없음"
        exit 1
    fi
    
    # Conda 환경 확인 (faiss용)
    if command -v conda &> /dev/null; then
        conda activate faiss-test 2>/dev/null || true
        print_status "Conda faiss-test 환경 활성화 시도"
    fi
}

# 테스트 모드 함수
test_mode() {
    local mode=$1
    local description=$2
    
    echo ""
    echo "==========================================="
    echo -e "${CYAN}테스트: $mode 모드${NC}"
    echo "$description"
    echo "==========================================="
    
    python3 << EOF
import sys
import torch
import asyncio
sys.path.append('$SCRIPT_DIR')

from main_unified import UnifiedInferenceSystem, InferenceConfig, MemoryMode

async def test_mode():
    mode = "$mode"
    print(f"\\n🔍 {mode.upper()} 모드 테스트 시작...")
    
    try:
        # 설정 생성
        config = InferenceConfig(
            memory_mode=MemoryMode.$mode,
            auto_memory_mode=False,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # 시스템 초기화
        system = UnifiedInferenceSystem(config)
        await system.initialize()
        
        # 메모리 사용량 확인
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"   GPU 메모리: {allocated:.2f}GB 할당 / {reserved:.2f}GB 예약")
        
        # 활성화된 모듈 확인
        print("\\n   활성화된 모듈:")
        modules = {
            'Neural Analyzers': system.config.use_neural_analyzers,
            'Advanced Wrappers': system.config.use_advanced_wrappers,
            'DSP Simulator': system.config.use_dsp_simulator,
            'Kalman Filter': system.config.use_kalman_filter,
            '3-View Scenario': system.config.use_three_view_scenario,
            'Multi-Ethics': system.config.use_multi_ethics_system,
            'Counterfactual': system.config.use_counterfactual_reasoning,
            'Regret Learning': system.config.use_advanced_regret_learning,
            'Meta Integration': system.config.use_meta_integration
        }
        
        for name, active in modules.items():
            status = "✅" if active else "❌"
            print(f"      {status} {name}")
        
        # 간단한 추론 테스트
        print("\\n   추론 테스트:")
        test_text = "친구가 어려운 상황에 처했을 때 도와야 할까?"
        
        import time
        start_time = time.time()
        result = await system.analyze(test_text)
        elapsed = time.time() - start_time
        
        print(f"      추론 시간: {elapsed:.2f}초")
        print(f"      통합 점수: {result.get('integrated_score', 0):.3f}")
        
        # 윤리적 딜레마 분석 테스트 (HEAVY/MCP 모드만)
        if mode in ['HEAVY', 'MCP']:
            print("\\n   윤리적 딜레마 분석 테스트:")
            scenarios = [
                "적극적으로 도와준다",
                "상황을 지켜본 후 판단한다",
                "최소한의 도움만 제공한다"
            ]
            
            dilemma_result = await system.analyze_ethical_dilemma(scenarios)
            print(f"      평가된 시나리오: {dilemma_result.get('total_evaluated', 0)}개")
            print(f"      처리 시간: {dilemma_result.get('processing_time', 0):.2f}초")
        
        print(f"\\n✅ {mode.upper()} 모드 테스트 성공!")
        
    except Exception as e:
        print(f"\\n❌ {mode.upper()} 모드 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

# 테스트 실행
asyncio.run(test_mode())
EOF
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        print_success "✅ $mode 모드 테스트 완료"
    else
        print_error "❌ $mode 모드 테스트 실패"
    fi
    
    return $exit_code
}

# 메인 실행
main() {
    print_status "테스트 준비 중..."
    
    # 가상환경 활성화
    activate_environment
    
    # Python 버전 확인
    python_version=$(python3 --version 2>&1)
    print_status "Python 버전: $python_version"
    
    # GPU 확인
    if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        gpu_info=$(python3 -c "import torch; print(f'{torch.cuda.get_device_name(0)}')" 2>/dev/null)
        print_success "GPU 사용 가능: $gpu_info"
    else
        print_warning "GPU 없음 - CPU 모드로 실행"
    fi
    
    # 테스트할 모드 선택
    if [ $# -eq 0 ]; then
        # 인자가 없으면 모든 모드 테스트
        print_status "모든 모드 테스트 시작..."
        
        test_mode "LIGHT" "230M - 빠른 프로토타이핑"
        test_mode "MEDIUM" "600M - 균형잡힌 일반 사용"
        test_mode "HEAVY" "970M - 심층 분석 (동적 스왑)"
        
        # MCP 모드는 선택적
        read -p "MCP 모드도 테스트하시겠습니까? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            test_mode "MCP" "MCP 서버 모드 (HEAVY 기반)"
        fi
        
    else
        # 특정 모드만 테스트
        mode=$(echo "$1" | tr '[:lower:]' '[:upper:]')
        case $mode in
            LIGHT)
                test_mode "LIGHT" "230M - 빠른 프로토타이핑"
                ;;
            MEDIUM)
                test_mode "MEDIUM" "600M - 균형잡힌 일반 사용"
                ;;
            HEAVY)
                test_mode "HEAVY" "970M - 심층 분석 (동적 스왑)"
                ;;
            MCP)
                test_mode "MCP" "MCP 서버 모드 (HEAVY 기반)"
                ;;
            *)
                print_error "알 수 없는 모드: $mode"
                echo "사용 가능한 모드: LIGHT, MEDIUM, HEAVY, MCP"
                exit 1
                ;;
        esac
    fi
    
    echo ""
    print_success "🎉 모든 테스트 완료!"
    date
}

# 스크립트 실행
main "$@"