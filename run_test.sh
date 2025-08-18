#!/bin/bash
# Red Heart AI 테스트 시스템 실행 스크립트
# Test System Execution Script for Red Heart AI

set -e  # 오류 발생 시 즉시 종료

echo "🧪 Red Heart AI 테스트 시스템 시작"
echo "================================"

# 현재 디렉토리 확인
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# venv 환경 활성화
if [ ! -f "red_heart_env/bin/activate" ]; then
    echo "❌ 오류: red_heart_env 가상환경을 찾을 수 없습니다."
    exit 1
fi

echo "🔧 환경 활성화 중..."
source red_heart_env/bin/activate

# 시스템 무결성 검사 (real_integrated_training_test.py 방식 - 필수)
echo "🔍 시스템 무결성 검사 중..."
echo "   📋 전체 모듈별 의존성 무결성 검사 실행..."
python system_integrity_checker.py
INTEGRITY_STATUS=$?

if [ $INTEGRITY_STATUS -eq 0 ]; then
    echo "✅ 시스템 무결성 검사 통과 - 테스트 시작"
else
    echo "❌ 시스템 무결성 검사 실패. 테스트를 중단합니다."
    echo "   환경을 수정하고 다시 시도해주세요."
    exit 1
fi

# 테스트 실행
echo "🎯 테스트 실행 중..."

# 실행 모드 결정
TEST_TYPE="${1:-integrated}"
shift 2>/dev/null || true

case "$TEST_TYPE" in
    "integrated"|"integration")
        echo "🔧 통합 테스트 실행..."
        python real_integrated_training_test.py "$@"
        ;;
    "learning"|"complete")
        echo "📚 완전한 학습 시스템 테스트..."
        python test_complete_learning_system.py "$@"
        ;;
    "baseline"|"basic")
        echo "📊 기본 베이스라인 테스트..."
        python real_integrated_training.py --test-mode --samples 3 "$@"
        ;;
    "quick"|"fast")
        echo "⚡ 빠른 테스트 (1개 샘플)..."
        python real_integrated_training_test.py --samples 1 --quick "$@"
        ;;
    "integrity"|"check")
        echo "🔍 시스템 무결성 검사 실행..."
        python system_integrity_checker.py
        ;;
    "help"|"-h"|"--help")
        echo "사용법:"
        echo "  $0 integrated   # 통합 테스트 실행 (기본값)"
        echo "  $0 learning     # 완전한 학습 시스템 테스트"
        echo "  $0 baseline     # 기본 베이스라인 테스트 (3개 샘플)"
        echo "  $0 quick        # 빠른 테스트 (1개 샘플)"
        echo "  $0 integrity    # 시스템 무결성 검사만 실행"
        echo "  $0 help         # 이 도움말 표시"
        echo ""
        echo "참고:"
        echo "  모든 테스트는 시작 전에 시스템 무결성 검사를 필수로 수행합니다."
        echo ""
        echo "예시:"
        echo "  $0 integrated --verbose"
        echo "  $0 quick"
        echo "  $0 baseline --log-level DEBUG"
        ;;
    *)
        echo "📖 사용자 정의 테스트: $TEST_TYPE"
        if [ -f "$TEST_TYPE" ]; then
            python "$TEST_TYPE" "$@"
        else
            python real_integrated_training_test.py --mode="$TEST_TYPE" "$@"
        fi
        ;;
esac

echo ""
echo "✅ Red Heart AI 테스트 완료"
echo "=========================="