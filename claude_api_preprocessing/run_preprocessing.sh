#!/bin/bash
# Claude API 전처리 실행 스크립트

echo "=== Claude API 완전 전처리 시스템 ==="
echo ""

# API 키 확인
if [ ! -f "api_key.json" ]; then
    echo "❌ api_key.json 파일이 없습니다."
    echo "파일을 생성하고 API 키를 입력하세요."
    exit 1
fi

# API 키가 설정되었는지 확인
if grep -q "YOUR_ANTHROPIC_API_KEY_HERE" api_key.json; then
    echo "❌ api_key.json에 실제 API 키를 입력하세요."
    exit 1
fi

# 데이터 확인
if [ ! -f "scruples_prepared.jsonl" ]; then
    echo "📊 Scruples 데이터 준비 중..."
    python3 prepare_scruples_data.py
    
    if [ $? -ne 0 ]; then
        echo "❌ 데이터 준비 실패"
        exit 1
    fi
fi

# 샘플 수 확인
SAMPLE_COUNT=$(wc -l < scruples_prepared.jsonl)
echo "✅ 준비된 샘플: $SAMPLE_COUNT개"

# 비용 예상
ESTIMATED_COST=$(echo "scale=2; $SAMPLE_COUNT * 1720 * 3 / 1000000 + $SAMPLE_COUNT * 210 * 15 / 1000000" | bc)
ESTIMATED_KRW=$(echo "scale=0; $ESTIMATED_COST * 1320" | bc)
ESTIMATED_HOURS=$(echo "scale=1; $SAMPLE_COUNT * 14 / 3600" | bc)

echo ""
echo "=== 예상 비용 및 시간 ==="
echo "💰 비용: \$$ESTIMATED_COST (약 ${ESTIMATED_KRW}원)"
echo "⏱️  시간: ${ESTIMATED_HOURS}시간"
echo ""

# 체크포인트 확인
if [ -f "checkpoint_complete.pkl" ]; then
    echo "📌 체크포인트 발견 - 이전 작업을 이어서 진행합니다."
    echo ""
fi

# 확인
read -p "계속하시겠습니까? (y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "취소되었습니다."
    exit 0
fi

# 실행
echo ""
echo "🚀 전처리 시작..."
echo "중단하려면 Ctrl+C (체크포인트 자동 저장)"
echo ""

# 로그와 함께 실행
python3 -u claude_complete_preprocessor.py 2>&1 | tee preprocessing_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "✅ 전처리 완료!"
echo ""

# 결과 통계
if [ -f "claude_preprocessed_complete.jsonl" ]; then
    SUCCESS_COUNT=$(wc -l < claude_preprocessed_complete.jsonl)
    echo "성공: $SUCCESS_COUNT개"
fi

if [ -f "claude_failed_complete.jsonl" ]; then
    FAILED_COUNT=$(wc -l < claude_failed_complete.jsonl)
    echo "실패: $FAILED_COUNT개"
fi