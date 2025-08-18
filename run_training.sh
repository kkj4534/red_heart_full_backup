#!/bin/bash

# Red Heart AI 통합 학습 실행 스크립트
# 60 에폭 학습 with Advanced Techniques

set -e  # 에러 발생 시 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 타임스탬프
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    Red Heart AI Training System${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 옵션 파싱
MODE="train"
EPOCHS=60
BATCH_SIZE=2
LR=1e-4
RESUME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            MODE="test"
            shift
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        --help)
            echo "사용법: $0 [옵션]"
            echo ""
            echo "옵션:"
            echo "  --test           테스트 모드 (2 에폭)"
            echo "  --epochs N       학습 에폭 수 (기본: 60)"
            echo "  --batch-size N   배치 사이즈 (기본: 2)"
            echo "  --lr N           학습률 (기본: 1e-4)"
            echo "  --resume PATH    체크포인트에서 재개"
            echo "  --help           도움말 표시"
            exit 0
            ;;
        *)
            echo -e "${RED}알 수 없는 옵션: $1${NC}"
            exit 1
            ;;
    esac
done

# 환경 확인
echo -e "${YELLOW}🔍 환경 확인 중...${NC}"

# Python 버전 확인
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
echo -e "  Python: $PYTHON_VERSION"

# PyTorch 확인
if python3 -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
    echo -e "  PyTorch: $TORCH_VERSION"
    
    # GPU 확인
    if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
        GPU_MEMORY=$(python3 -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')")
        echo -e "  GPU: ${GREEN}$GPU_NAME ($GPU_MEMORY)${NC}"
    else
        echo -e "  GPU: ${YELLOW}사용 불가 (CPU 모드)${NC}"
    fi
else
    echo -e "${RED}❌ PyTorch가 설치되지 않았습니다!${NC}"
    exit 1
fi

# 메모리 확인
TOTAL_MEM=$(free -h | grep "^Mem:" | awk '{print $2}')
AVAIL_MEM=$(free -h | grep "^Mem:" | awk '{print $7}')
echo -e "  메모리: $AVAIL_MEM / $TOTAL_MEM 사용 가능"

echo ""

# 모드에 따른 실행
if [ "$MODE" == "test" ]; then
    echo -e "${YELLOW}🧪 테스트 모드 실행${NC}"
    echo -e "  - 컴포넌트 테스트"
    echo -e "  - 미니 학습 (2 에폭)"
    echo ""
    
    # 테스트 실행
    python3 training/test_unified_training.py
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ 테스트 성공!${NC}"
    else
        echo -e "${RED}❌ 테스트 실패!${NC}"
        exit 1
    fi
    
else
    echo -e "${GREEN}🚀 학습 시작${NC}"
    echo -e "  - 에폭: $EPOCHS"
    echo -e "  - 배치 사이즈: $BATCH_SIZE"
    echo -e "  - 학습률: $LR"
    
    if [ ! -z "$RESUME" ]; then
        echo -e "  - 재개: $RESUME"
    fi
    
    echo ""
    
    # 로그 디렉토리 생성
    LOG_DIR="training/logs"
    mkdir -p $LOG_DIR
    LOG_FILE="$LOG_DIR/training_${TIMESTAMP}.log"
    
    echo -e "${BLUE}📝 로그 파일: $LOG_FILE${NC}"
    echo ""
    
    # 학습 명령 구성
    CMD="python3 training/unified_training_final.py"
    CMD="$CMD --epochs $EPOCHS"
    CMD="$CMD --batch-size $BATCH_SIZE"
    CMD="$CMD --lr $LR"
    
    if [ ! -z "$RESUME" ]; then
        CMD="$CMD --resume $RESUME"
    fi
    
    # 학습 실행 (로그 저장 + 화면 출력)
    echo -e "${YELLOW}실행 명령: $CMD${NC}"
    echo ""
    
    # tee를 사용하여 화면과 파일에 동시 출력
    $CMD 2>&1 | tee $LOG_FILE
    
    # 결과 확인
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo ""
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}🎉 학습 완료!${NC}"
        echo -e "${GREEN}========================================${NC}"
        
        # 체크포인트 확인
        CHECKPOINT_DIR="training/checkpoints_final"
        if [ -d "$CHECKPOINT_DIR" ]; then
            NUM_CHECKPOINTS=$(ls -1 $CHECKPOINT_DIR/*.pt 2>/dev/null | wc -l)
            echo -e "  💾 저장된 체크포인트: $NUM_CHECKPOINTS개"
            
            # Crossover 모델 확인
            if [ -f "$CHECKPOINT_DIR/crossover_final.pth" ]; then
                echo -e "  🧬 Crossover 모델: ${GREEN}생성 완료${NC}"
            fi
        fi
        
        # Sweet Spot 분석 결과 확인
        ANALYSIS_DIR="training/sweet_spot_analysis"
        if [ -d "$ANALYSIS_DIR" ]; then
            echo -e "  📊 Sweet Spot 분석: ${GREEN}완료${NC}"
        fi
        
        echo ""
        echo -e "${BLUE}다음 단계:${NC}"
        echo -e "  1. 학습 곡선 확인: training/plots/"
        echo -e "  2. Sweet Spot 분석: training/sweet_spot_analysis/"
        echo -e "  3. 최종 모델 사용: training/checkpoints_final/crossover_final.pth"
        
    else
        echo ""
        echo -e "${RED}❌ 학습 실패!${NC}"
        echo -e "  로그 확인: $LOG_FILE"
        exit 1
    fi
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    작업 완료 - $(date)${NC}"
echo -e "${BLUE}========================================${NC}"