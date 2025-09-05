# MCP 서버 활성화 가이드

## 🎯 구현 완료 사항

### 1. Circuit Fallback 가속화 ✅
- **초기 시도**: GPU 메모리 체크 (2GB 미만시 skip)
- **타임아웃**: 5초로 제한 (빠른 fallback)
- **컨텍스트 저장**: `circuit_context_saved` 변수에 저장

### 2. Circuit 재실행 로직 ✅
- **위치**: Phase 8 GPU 정리 직후
- **조건**: 초기 실행 실패 시 재시도
- **결과 통합**: emotion, ethics, regret 데이터 워크플로우 통합

### 3. 변수 유지 및 결과 통합 ✅
- Circuit 컨텍스트 변수 유지
- Red Heart 출력물 변수 유지
- 재실행 결과 워크플로우 통합

## 📌 MCP 서버 활성화 방법

### 1단계: MCP 설정 생성
```bash
cd /mnt/c/large_project/linux_red_heart
python mcp_server.py --create-config
```

### 2단계: MCP 서버 시작
```bash
# 터미널 1에서 실행
python mcp_server.py
```
- 포트: 8765
- URL: http://localhost:8765
- 매니페스트: http://localhost:8765/mcp/manifest

### 3단계: Claude Desktop 연동
1. Claude Desktop 설정 열기
2. MCP Servers 섹션 찾기
3. 생성된 설정 파일 경로 추가:
   - `~/.config/claude/mcp_config.json`

### 4단계: 테스트
```bash
# 터미널 2에서 테스트
curl -X POST http://localhost:8765/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "친구의 비밀을 지켜야 할지 공익을 위해 알려야 할지 고민입니다",
    "mode": "heavy"
  }'
```

## 🔧 main_unified.py 테스트

### 기본 테스트 (Circuit 재실행 확인)
```bash
python main_unified.py \
  --text "친구의 거짓말을 폭로해야 하는 상황입니다" \
  --memory_mode medium \
  --debug
```

### 예상 로그 패턴
```
1. 🎭 감정-윤리-후회 통합 회로 처리 (초기 시도)...
2. ⚠️ GPU 메모리 부족 (X.XGB), Circuit 후반 실행 예약
3. Phase 2-7 진행...
4. 🔄 LLM을 위해 Red Heart 모듈들을 RAM으로 스왑...
5. ✅ GPU 메모리 확보 완료: X.XGB 사용 가능
6. 🔄 Circuit 재실행 (GPU 메모리 확보됨)...
7. ✅ Circuit 재실행 성공 (신뢰도: X.XX)
```

## 📊 성능 개선 예상

- **초기 Circuit 실패**: 30초 → 5초 (6배 단축)
- **전체 처리 시간**: 유지 또는 개선
- **GPU 메모리 효율**: 92.7% → 최적화
- **Circuit 성공률**: 0% → 90%+

## 🚀 LLM MCP 전환 (선택사항)

### Phase 8 대체 방법
1. `main_unified.py`에서 Phase 8 비활성화:
   ```python
   config.llm_mode = "none"
   ```

2. MCP 서버를 통해 LLM 처리:
   - Red Heart 분석 결과를 MCP로 전송
   - Claude Desktop에서 윤리적 판단
   - 결과 통합

## ⚠️ 주의사항

1. **메모리 모드**: MEDIUM 모드 권장 (GPU/CPU 혼합)
2. **포트 충돌**: 8765 포트 확인
3. **Python 경로**: PYTHONPATH 설정 확인

## 📝 다음 단계

1. 테스트 실행
2. 로그 확인 (Circuit 재실행 성공 여부)
3. LLM 품질 개선 (temperature 조정)
4. MCP 서버 상시 운영 설정 (systemd 등)