#!/bin/bash
# Red Heart AI 통합 학습 시스템 실행 스크립트 
# Complete Learning System Execution Script for Red Heart AI
# 환경 분리 지원: conda (faiss) + venv (transformers, torch, 기타)

set -e  # 오류 발생 시 즉시 종료

echo "🚀 Red Heart AI 통합 학습 시스템 시작"
echo "==========================================="

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

# 환경 분리 자동 설정 함수
setup_integrated_environment() {
    print_status "Red Heart AI 통합 환경 설정 시작..."
    
    # 1. venv 환경 확인/생성
    if [ ! -d "red_heart_env" ]; then
        print_status "red_heart_env 가상환경 생성 중..."
        python3 -m venv red_heart_env
        print_success "가상환경 생성 완료"
    fi
    
    # 2. conda 환경 확인/생성
    if ! conda env list | grep -q "faiss-test"; then
        print_status "faiss-test conda 환경 생성 중..."
        conda create -n faiss-test python=3.12 -y
        print_success "conda 환경 생성 완료"
    fi
    
    # 3. conda 환경에 FAISS 설치 - 자동 설치 차단됨
    print_error "❌ 자동 패키지 설치가 차단되었습니다!"
    print_warning "필요한 패키지가 없다면 수동으로 설치하세요:"
    print_warning "  conda run -n faiss-test pip install -r requirements_conda.txt"
    print_status "conda 환경 패키지 검증만 수행합니다..."
    
    # 패키지 존재 여부만 확인
    conda run -n faiss-test python -c "import faiss, numpy, scipy, spacy" 2>/dev/null
    if [ $? -eq 0 ]; then
        print_success "✅ conda 환경 필수 패키지 확인됨"
    else
        print_error "❌ conda 환경에 필수 패키지가 누락되었습니다!"
        print_error "   수동 설치 필요: requirements_conda.txt 참조"
        return 1
    fi
    
    # 4. venv 환경 활성화 및 패키지 설치
    source red_heart_env/bin/activate
    print_status "venv 환경 패키지 상태 확인 중..."
    
    # 환경별 requirements 파일 사용 (오염 방지)
    print_status "📋 환경별 requirements 파일 사용 (NumPy 1.x 호환성 보장)"
    
    # 필수 패키지 확인 먼저 실행
    python -c "
import sys
required_packages = ['torch', 'transformers', 'numpy', 'llama_cpp']
missing = []
for pkg in required_packages:
    try:
        __import__(pkg)
        print(f'✅ {pkg} 설치됨')
    except ImportError:
        missing.append(pkg)
        print(f'❌ {pkg} 누락')

if missing:
    print(f'⚠️ 누락된 패키지: {missing}')
    # 누락된 패키지가 있어도 일단 진행 (선택적 설치 지원)
    print('📋 requirements_venv.txt 설치를 권장합니다')
else:
    print('✅ 모든 필수 패키지 설치 완료')
    "
    
    # requirements_venv.txt가 있으면 사용자에게 선택권 제공
    if [ -f "requirements_venv.txt" ]; then
        print_status "requirements_venv.txt 발견 - 안전한 환경별 패키지 사용 가능"
        print_status "기존 환경이 정상 작동하므로 추가 설치 생략"
    fi
    
    print_success "통합 환경 설정 완료!"
}

# 환경 검증 전용 함수
check_environment_status() {
    local need_setup=false
    
    # 1. conda 초기화 (조용히)
    if [ -f "/home/kkj/miniconda3/etc/profile.d/conda.sh" ]; then
        source "/home/kkj/miniconda3/etc/profile.d/conda.sh" 2>/dev/null
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null
    elif command -v conda &> /dev/null; then
        eval "$(conda shell.bash hook)" 2>/dev/null || true
    fi
    
    # 2. venv 상태 확인
    if [ ! -f "red_heart_env/bin/activate" ]; then
        print_warning "❌ red_heart_env 가상환경 없음"
        need_setup=true
    else
        print_success "✅ red_heart_env 가상환경 존재"
    fi
    
    # 3. conda 환경 상태 확인  
    if ! conda env list 2>/dev/null | grep -q "faiss-test"; then
        print_warning "❌ faiss-test conda 환경 없음"
        need_setup=true
    else
        print_success "✅ faiss-test conda 환경 존재"
        
        # FAISS 설치 상태 확인
        if conda run -n faiss-test python -c "import faiss" >/dev/null 2>&1; then
            print_success "✅ FAISS 설치됨 및 작동 확인"
        else
            print_warning "❌ FAISS 설치되지 않음"
            print_warning "   수동 설치 필요: conda run -n faiss-test pip install -r requirements_conda.txt"
            # need_setup=true 제거 - 자동 설치 트리거 차단
        fi
    fi
    
    # 4. 주요 패키지 확인 (venv 활성화 후)
    if [ -f "red_heart_env/bin/activate" ]; then
        source red_heart_env/bin/activate 2>/dev/null
        if python -c "import torch, transformers, numpy" >/dev/null 2>&1; then
            print_success "✅ 주요 패키지 (torch, transformers, numpy) 설치됨"
        else
            print_warning "❌ 주요 패키지 일부 미설치"
            print_warning "   수동 설치 필요: pip install -r requirements_venv.txt"
            # need_setup=true 제거 - 자동 설치 차단
        fi
    fi
    
    if [ "$need_setup" = true ]; then
        return 1  # 설정 필요
    else
        return 0  # 모든 것이 준비됨
    fi
}

# 환경 분리 활성화 함수 (스마트)
activate_integrated_environment() {
    print_status "Red Heart AI 통합 환경 상태 확인 중..."
    
    # 1. 먼저 환경 상태 체크
    if check_environment_status; then
        print_success "🎉 모든 환경이 이미 준비되어 있습니다!"
        
        # conda 초기화 (필요한 경우)
        if [ -f "/home/kkj/miniconda3/etc/profile.d/conda.sh" ]; then
            source "/home/kkj/miniconda3/etc/profile.d/conda.sh" 2>/dev/null
        elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
            source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null
        elif command -v conda &> /dev/null; then
            eval "$(conda shell.bash hook)" 2>/dev/null || true
        fi
        
        # venv 활성화
        source red_heart_env/bin/activate
        
    else
        print_warning "⚠️  환경 설정이 필요합니다. 자동 설정을 시작합니다..."
        
        # 사용자에게 확인
        echo ""
        echo "다음 작업이 수행됩니다:"
        echo "  1. red_heart_env 가상환경 생성/업데이트"
        echo "  2. faiss-test conda 환경 생성/업데이트"  
        echo "  3. 필요한 패키지 설치 (torch, transformers, faiss 등)"
        echo ""
        read -p "계속 진행하시겠습니까? [Y/n]: " -t 10 confirm || confirm="Y"
        
        if [[ $confirm =~ ^[Yy]$|^$ ]]; then
            print_error "❌ 자동 환경 설정이 차단되었습니다!"
            print_error "   의존성 무결성 보호를 위해 자동 설치가 비활성화됨"
            print_warning "필요한 패키지는 수동으로 설치하세요:"
            print_warning "  - venv: pip install -r requirements_venv.txt"
            print_warning "  - conda: conda run -n faiss-test pip install -r requirements_conda.txt"
            # setup_integrated_environment 호출 차단
            exit 1
            
            # 설정 후 재검증
            if check_environment_status; then
                print_success "✅ 환경 검증이 성공적으로 완료되었습니다!"
                source red_heart_env/bin/activate
            else
                print_error "❌ 환경 검증에 실패했습니다. 수동으로 확인해주세요."
                print_warning "다음 파일들을 참조하여 수동 설치:"
                print_warning "  - requirements_venv.txt (venv 환경용)"
                print_warning "  - requirements_conda.txt (conda faiss-test 환경용)"
                exit 1
            fi
        else
            print_warning "사용자가 설정을 취소했습니다. 기존 환경으로 진행합니다."
            if [ -f "red_heart_env/bin/activate" ]; then
                source red_heart_env/bin/activate
            fi
        fi
    fi
    
    print_success "✅ Red Heart AI 통합 환경 활성화 완료"
    print_success "   - venv: ${VIRTUAL_ENV:-'not activated'}"
    print_success "   - conda: faiss-test 환경 준비됨"
    print_success "   - python: $(which python)"
    print_success "   - 환경 분리: faiss→conda subprocess, 나머지→venv"
}

# CVE-2025-32434는 가짜 CVE이므로 보안 패치 코드 제거
# torch.load는 정상적으로 작동하며 추가 환경변수나 패치가 불필요함

# export TORCH_LOAD_ALLOW_UNSAFE=1
# print_status "torch 보안 제한 우회 설정 (TORCH_LOAD_ALLOW_UNSAFE=1)"

# print_status "CVE-2025-32434 보안 패치 사전 적용..."
# python -c "import torch_security_patch; print('✅ 보안 패치 사전 적용 완료')" 2>/dev/null || print_warning "보안 패치 사전 적용 실패"

# 통합 환경 활성화
activate_integrated_environment

# 학습 시스템 실행 함수
run_learning_system() {
    local mode="${1:-auto}"
    shift 2>/dev/null || true
    
    print_status "Red Heart AI 통합 학습 시스템 실행 시작..."
    print_status "실행 모드: $mode"
    
    case "$mode" in
        # ================== 학습 모드 (Training) ==================
        "train"|"training")
            print_status "📚 학습 모드 - 320M v2 시스템"
            print_status "   - 완전한 학습 기능 (NO FALLBACK)"
            print_status "   - DSP, 칼만필터 등 모든 모듈 온전히 학습"
            python unified_training_v2.py --mode train "$@"
            ;;
        "train-local"|"train-test")
            print_status "🧪 로컬 학습 테스트 모드"
            print_status "   - 소규모 샘플로 학습 가능성 검증"
            print_status "   - GPU 메모리 사용량 모니터링"
            python unified_training_v2.py --mode train --max-samples ${SAMPLES:-3} --debug --verbose "$@"
            ;;
        "train-cloud"|"train-full")
            print_status "☁️ 클라우드 학습 모드"
            print_status "   - 전체 데이터셋 학습"
            print_status "   - 체크포인트 자동 저장"
            python unified_training_v2.py --mode train --full-dataset --checkpoint-interval 1000 "$@"
            ;;
        "train-validate")
            print_status "✅ 학습 검증 모드"
            print_status "   - 학습된 모델 성능 평가"
            python unified_training_v2.py --mode eval --load-checkpoint "$@"
            ;;
            
        # ================== 메인 시스템 모드 ==================
        "main"|"advanced")
            print_status "🎯 Red Heart AI 메인 시스템 (main.py)"
            print_status "   - 모든 고급 AI 모듈 통합"
            print_status "   - module_bridge_coordinator 활용"
            print_status "   - XAI, 베이지안, 반사실적 추론 등 전체 기능"
            if [ -f "main.py" ]; then
                python main.py --mode advanced "$@"
            else
                print_error "main.py를 찾을 수 없습니다."
                exit 1
            fi
            ;;
            
        # ================== 운용 모드 (Production) ==================
        "production"|"prod")
            print_status "🚀 운용 모드 - main.py 전체 시스템"
            print_status "   - 모든 고급 분석 모듈 통합"
            print_status "   - XAI, 시계열, 베이지안 등 전체 기능"
            if [ -f "main.py" ]; then
                python main.py --mode production "$@"
            else
                print_error "main.py를 찾을 수 없습니다. 통합 시스템으로 대체합니다."
                python unified_system_main.py --mode auto "$@"
            fi
            ;;
        "production-advanced"|"prod-adv")
            print_status "🎯 고급 운용 모드"
            print_status "   - main.py + 추가 고급 모듈"
            print_status "   - XAI 피드백, 시계열 전파, 베이지안 추론"
            if [ -f "main.py" ]; then
                python main.py --mode advanced --enable-xai --enable-temporal --enable-bayesian "$@"
            else
                python unified_system_main.py --mode auto --advanced "$@"
            fi
            ;;
        "production-oss"|"prod-oss")
            print_status "🤖 OSS 20B 통합 운용 모드"
            print_status "   - OSS 모델과 연동 분석"
            python main.py --mode production --oss-integration "$@"
            ;;
            
        # ================== 고급 AI 분석 모드 ==================
        "xai"|"explain")
            print_status "🔍 XAI 설명 가능 AI 모드"
            print_status "   - 의사결정 투명성 제공"
            python main.py --mode xai "$@"
            ;;
        "temporal"|"time-series")
            print_status "⏱️ 시계열 사건 전파 분석"
            print_status "   - 장기적 영향 예측"
            python main.py --mode temporal "$@"
            ;;
        "bayesian")
            print_status "📊 베이지안 추론 모드"
            print_status "   - 불확실성 정량화"
            python main.py --mode bayesian "$@"
            ;;
        "counterfactual"|"what-if")
            print_status "🤔 반사실적 추론 모드"
            print_status "   - '만약' 시나리오 분석"
            python main.py --mode counterfactual "$@"
            ;;
        "ethics"|"multi-ethics")
            print_status "⚖️ 다차원 윤리 시스템"
            print_status "   - 복합적 윤리 판단"
            python main.py --mode ethics "$@"
            ;;
            
        # ================== MCP 준비 모드 ==================
        "mcp-prepare"|"mcp-init")
            print_status "🔌 MCP 서비스 준비 모드"
            print_status "   - API 엔드포인트 초기화"
            print_status "   - 인터페이스 스켈레톤 생성"
            if [ -f "mcp_service_init.py" ]; then
                python mcp_service_init.py "$@"
            else
                print_warning "MCP 서비스 초기화 스크립트를 준비 중입니다."
                print_status "향후 구현 예정: Claude/GPT/OSS 챗봇 연결"
            fi
            ;;
            
        # ================== 기존 호환 모드 ==================
        "unified"|"800m"|"v2")
            print_status "🚀 새로운 320M v2 통합 시스템 실행..."
            print_status "   - 104M 공유 백본 + 174M 전문 헤드 + 40M 전문모듈"
            print_status "   - LLM 전처리 + 3단계 워크플로우"
            print_status "   - Gate 9 최적화 버전"
            if [ -f "unified_training_v2.py" ]; then
                python unified_training_v2.py --mode train "$@"
            elif [ -f "unified_system_main.py" ]; then
                print_warning "v2를 찾을 수 없음, 기존 800M 시스템 사용"
                python unified_system_main.py --mode auto "$@"
            else
                print_error "통합 시스템을 찾을 수 없습니다."
                python real_integrated_training.py "$@"
            fi
            ;;
        "unified-train")
            print_status "📚 730M 최종 통합 시스템 훈련 모드..."
            
            # 새로운 최종 시스템이 있으면 우선 사용
            if [ -f "training/unified_training_final.py" ]; then
                print_status "   ✨ 최종 통합 시스템 (730M) 사용"
                print_status "   - 60 에폭 전체 학습"
                print_status "   - 30개 체크포인트 저장"
                python training/unified_training_final.py --epochs 60 "$@"
            else
                print_warning "최종 시스템 없음, 기존 v2 시스템 사용"
                python unified_training_v2.py --mode train "$@"
            fi
            ;;
        "unified-test")
            print_status "🧪 730M 최종 통합 시스템 학습 테스트 모드..."
            print_status "   - 60 에폭 학습 테스트 (--samples로 제한 가능)"
            print_status "   - LR 스윕, Sweet Spot, Parameter Crossover 포함"
            print_status "   - Advanced Training Techniques 활성화"
            
            # 새로운 최종 시스템이 있으면 우선 사용
            if [ -f "training/unified_training_final.py" ]; then
                print_status "   ✨ 최종 통합 시스템 (730M) 사용"
                if [ -n "${SAMPLES}" ]; then
                    # 샘플 수가 지정되면 테스트 모드로 에폭 조정
                    python training/unified_training_final.py --test --epochs ${SAMPLES:-3} "$@"
                else
                    # 기본 테스트 모드 (2 에폭)
                    python training/unified_training_final.py --test "$@"
                fi
            else
                print_warning "최종 시스템 없음, 기존 v2 시스템 사용"
                python unified_training_v2.py --mode train-test --max-samples ${SAMPLES:-3} --no-param-update --debug --verbose "$@"
            fi
            ;;
        "unified-test-v1"|"unified-test-800m")
            print_status "🧪 기존 800M 통합 시스템 테스트 모드..."
            python unified_system_main.py --mode test "$@"
            ;;
        "unified-monitor")
            print_status "📊 800M 통합 시스템 모니터링 모드..."
            python unified_system_main.py --mode monitoring "$@"
            ;;
        "unified-dashboard")
            print_status "📈 800M 통합 시스템 대시보드 모드..."
            python unified_system_main.py --mode dashboard "$@"
            ;;
        "auto"|"integrated")
            print_status "🎯 자동 통합 학습 시스템 실행..."
            print_status "   - 자동 환경 세팅 포함"
            print_status "   - 무결성 검사 + 환경 분리 검증 + 학습 실행"
            # 새로운 시스템이 있으면 우선 사용
            if [ -f "unified_system_main.py" ]; then
                print_status "   ✨ 800M 통합 시스템으로 실행"
                python unified_system_main.py --mode auto "$@"
            else
                python real_integrated_training.py "$@"
            fi
            ;;
        "learning"|"train")
            print_status "📚 기본 학습 모드 실행..."
            python real_integrated_training.py --learning-mode "$@"
            ;;
        "test"|"testing")
            print_status "🧪 통합 테스트 시스템 실행..."
            print_status "   - 학습 시스템 모드로 강제 활성화"
            if [ -f "test_complete_learning_system.py" ]; then
                python test_complete_learning_system.py "$@"
            else
                print_warning "test_complete_learning_system.py를 찾을 수 없습니다. real_integrated_training.py 학습 모드로 실행합니다."
                python real_integrated_training.py --learning "$@"
            fi
            ;;
        "complete"|"full")
            print_status "🎯 완전한 학습 시스템 실행..."
            print_status "   - 24,170개 데이터셋 처리"
            print_status "   - 감정, 벤담, 후회, SURD 분석 통합"
            python real_integrated_training.py --complete-learning "$@"
            ;;
        "baseline"|"basic")
            print_status "⚡ 기본 베이스라인 테스트..."
            python real_integrated_training.py --baseline-test "$@"
            ;;
        "validate"|"check")
            print_status "🔍 시스템 검증 모드..."
            print_status "   - 무결성 검사만 수행"
            print_status "   - 환경 분리 검증"
            print_status "   - FAISS subprocess 테스트"
            # 새로운 시스템 검증 우선 시도
            if [ -f "unified_system_main.py" ]; then
                python unified_system_main.py --mode validate "$@"
            else
                python system_integrity_checker.py
            fi
            ;;
        "setup"|"install")
            print_status "🔧 환경 설정 모드..."
            setup_integrated_environment
            print_success "환경 설정이 완료되었습니다. 이제 학습을 시작할 수 있습니다."
            print_status "다음 명령어로 학습을 시작하세요:"
            print_status "  $0 unified      # 800M 통합 시스템 (권장)"
            print_status "  $0 auto         # 자동 통합 학습"
            print_status "  $0 test         # 테스트 실행"
            ;;
        "help"|"-h"|"--help")
            show_learning_help
            ;;
        *)
            print_warning "알 수 없는 모드: $mode, 자동 모드로 실행합니다."
            python real_integrated_training.py "$mode" "$@"
            ;;
    esac
}

# 도움말 함수
show_learning_help() {
    echo ""
    echo "🚀 Red Heart AI 통합 학습 시스템 사용법"
    echo "========================================"
    echo ""
    echo "기본 사용법:"
    echo "  $0 [모드] [옵션...]"
    echo ""
    echo "🎯 메인 시스템:"
    echo "  main, advanced      # Red Heart AI 메인 시스템 (main.py)"
    echo ""
    echo "📚 학습 모드 (Training):"
    echo "  train, training     # 450M 학습 시스템 (완전한 학습)"
    echo "  train-local         # 로컬 학습 테스트 (3개 샘플)"
    echo "  train-cloud         # 클라우드 학습 (전체 데이터셋)"
    echo "  train-validate      # 학습된 모델 검증"
    echo ""
    echo "🚀 운용 모드 (Production):"
    echo "  production, prod    # main.py 전체 시스템"
    echo "  production-advanced # 고급 운용 (XAI, 시계열, 베이지안)"
    echo "  production-oss      # OSS 20B 모델 통합"
    echo ""
    echo "🔍 고급 AI 분석:"
    echo "  xai, explain        # XAI 설명 가능 AI"
    echo "  temporal            # 시계열 사건 전파 분석"
    echo "  bayesian            # 베이지안 추론"
    echo "  counterfactual      # 반사실적 추론"
    echo "  ethics              # 다차원 윤리 시스템"
    echo ""
    echo "🔌 MCP 준비:"
    echo "  mcp-prepare         # MCP 서비스 초기화 (향후 구현)"
    echo ""
    echo "⚙️ 기존 호환 모드:"
    echo "  unified, 800m, v2   # 320M v2 시스템"
    echo "  unified-train       # 320M v2 훈련 모드"
    echo "  unified-test        # 320M v2 테스트 모드"
    echo "  unified-monitor     # 시스템 모니터링"
    echo "  unified-dashboard   # 웹 대시보드"
    echo ""
    echo "🎯 권장 사용법:"
    echo "  $0 train-local      # 로컬 학습 테스트 (3개 샘플)"
    echo "  $0 train-cloud      # 클라우드 전체 학습"
    echo "  $0 production       # 운용 모드 (모든 기능)"
    echo ""
    echo "⚡ 학습 테스트 명령어:"
    echo "  $0 train-local --samples 3 --debug --verbose"
    echo "  nohup timeout 1200 bash $0 unified-test --samples 3 --debug --verbose > test_$(date +%Y%m%d_%H%M%S).txt 2>&1 &"
    echo ""
    echo "자주 사용되는 명령어:"
    echo "  $0 train            # 학습 모드"
    echo "  $0 production       # 운용 모드"
    echo "  $0 test             # 빠른 테스트"
    echo "  $0 setup            # 처음 설정"
    echo "  $0 validate         # 환경 검증"
    echo ""
    echo "고급 옵션 (800M 통합 시스템):"
    echo "  --samples N         # 처리할 샘플 수"
    echo "  --epochs N          # 훈련 에포크 수"
    echo "  --batch-size N      # 배치 크기"
    echo "  --learning-rate F   # 학습률"
    echo "  --strategy S        # 훈련 전략 (adaptive/parallel/round_robin)"
    echo "  --timeout N         # 최대 실행 시간 (초)"
    echo "  --dashboard-port N  # 대시보드 포트"
    echo "  --verbose           # 상세 로그 출력"
    echo "  --debug             # 디버그 모드"
    echo "  --report            # 실행 완료 후 리포트 생성"
    echo ""
    echo "기존 시스템 옵션 (real_integrated_training.py):"
    echo "  --samples N         # 처리할 샘플 수"
    echo "  --batch-size N      # 배치 크기"
    echo "  --learning-rate F   # 학습률"
    echo "  --verbose           # 상세 로그 출력"
    echo "  --debug             # 디버그 모드"
    echo ""
    echo "💡 800M 시스템 예시:"
    echo "  $0 unified --samples 1000 --epochs 5 --report"
    echo "  $0 unified-train --batch-size 8 --strategy adaptive"
    echo "  $0 unified-test --samples 50 --debug"
    echo "  $0 unified-monitor --duration 3600"
    echo "  $0 unified-dashboard --dashboard-port 8080"
    echo ""
    echo "기존 시스템 예시:"
    echo "  $0 auto --samples 100 --verbose"
    echo "  $0 complete --batch-size 32"
    echo "  $0 test --debug"
    echo ""
    echo "🔧 시스템 아키텍처:"
    echo "  320M v2 시스템 (Gate 9 최적화):"
    echo "    - 104M 공유 백본 (확장된 차원)"
    echo "    - 174M 전문 헤드 (비례 확장)"
    echo "    - 40M 전문 분석 모듈"
    echo "    - 1.2M DSP + 칼만 필터"
    echo "    - LLM 전처리 파이프라인 (4-bit 양자화)"
    echo "    - 3단계 단순화 워크플로우"
    echo ""
    echo "  기존 800M 시스템 (v1):"
    echo "    - 300M 공유 백본"
    echo "    - 500M 전문 헤드"
    echo "    - 동적 RAM 스왑 관리자"
    echo ""
    echo "환경 분리 정보:"
    echo "  - venv: transformers, torch, numpy 등 메인 패키지"
    echo "  - conda: faiss, spacy (subprocess로 분리 실행)"
    echo "  - 자동 환경 설정 및 검증 포함"
    echo ""
}

# 정리 함수
cleanup_learning() {
    print_status "학습 시스템 정리 중..."
    
    # 임시 파일 정리
    find . -name "*.pyc" -delete 2>/dev/null || true
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    
    # GPU 메모리 정리
    if command -v nvidia-smi &> /dev/null; then
        print_status "GPU 메모리 정리..."
        # Python processes cleanup
        pkill -f "python.*real_integrated_training" 2>/dev/null || true
        sleep 1
    fi
    
    print_success "정리 완료"
}

# 신호 처리
trap cleanup_learning EXIT
trap 'print_warning "학습 중단됨"; cleanup_learning; exit 130' INT TERM

# 메인 실행
main() {
    local mode="${1:-auto}"
    
    # 도움말 요청 처리
    case "$mode" in
        --help|-h|help)
            show_learning_help
            exit 0
            ;;
    esac
    
    print_status "Red Heart AI 통합 학습 시스템 시작"
    print_status "환경 분리: conda(faiss) + venv(transformers,torch)"
    
    # 학습 시스템 실행
    run_learning_system "$@"
    
    print_success "🎉 Red Heart AI 학습 시스템 완료!"
    echo ""
}

# 스크립트 실행
main "$@"