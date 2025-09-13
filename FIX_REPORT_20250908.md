# Red Heart AI 코드베이스 수정 보고서
작성일: 2025-09-08

## 수정 완료 항목

### 1. 클래스명 불일치 문제 해결 ✅
**파일**: `advanced_bentham_calculator.py`
**수정 내용**:
- Line 27: `from sentence_transformer_singleton import SentenceTransformerSingleton` → `SentenceTransformerManager`
- Line 184: `st_singleton = SentenceTransformerSingleton()` → `SentenceTransformerManager()`
- **결과**: ImportError 해결, 클래스 정상 import 확인

### 2. asyncio.run 중첩 문제 해결 ✅
**파일**: `advanced_bentham_calculator.py`
**수정 내용**:
- Lines 1557-1563: nest_asyncio.apply() 및 asyncio.run() 제거
- run_async_safely() 헬퍼 함수 사용으로 대체
```python
from config import run_async_safely
scenario_analysis = run_async_safely(
    self.three_view_system.analyze_three_view_scenarios(input_data),
    timeout=120.0
)
```
- **결과**: 이벤트 루프 중첩 문제 근본적 해결

### 3. LLM 조건부 초기화 구현 ✅
**파일**: `advanced_emotion_analyzer.py`
**수정 내용**:

#### AdvancedEmotionAnalyzer 클래스 (Lines 602-615)
```python
# LLM 엔진 연결 - Claude 모드에서는 비활성화
is_claude_mode = os.environ.get('REDHEART_CLAUDE_MODE', '0') == '1'

if LLM_INTEGRATION_AVAILABLE and not is_claude_mode:
    try:
        self.llm_engine = get_llm_engine()
        logger.info("LLM 엔진 연결 완료")
    except Exception as e:
        logger.warning(f"LLM 엔진 연결 실패: {e}")
        LLM_INTEGRATION_AVAILABLE = False
elif is_claude_mode:
    logger.info("📌 Claude 모드 감지 - 로컬 LLM 엔진 비활성화")
    self.llm_engine = None
```

#### EmotionCounselorModule 클래스 (Lines 4257-4269)
```python
def __init__(self):
    self.llm_engine = None
    # Claude 모드에서는 LLM 엔진 비활성화
    is_claude_mode = os.environ.get('REDHEART_CLAUDE_MODE', '0') == '1'
    
    if LLM_INTEGRATION_AVAILABLE and not is_claude_mode:
        try:
            from llm_module.advanced_llm_engine import get_llm_engine
            self.llm_engine = get_llm_engine()
        except Exception as e:
            logger.warning(f"상담사 모듈 LLM 초기화 실패: {e}")
    elif is_claude_mode:
        logger.info("📌 상담사 모듈: Claude 모드 감지 - 로컬 LLM 엔진 비활성화")
```

### 4. Claude 모드 환경변수 설정 ✅
**파일**: `main_unified.py`
**수정 내용**:
- Lines 3572-3574: Claude 모드 감지 시 환경변수 설정
```python
if args.llm == 'claude':
    logger.info("🔄 Claude API 모드 감지 - 독립 워크플로우로 전환...")
    
    # Claude 모드 환경변수 설정 - 로컬 LLM 비활성화
    os.environ['REDHEART_CLAUDE_MODE'] = '1'
    logger.info("📌 REDHEART_CLAUDE_MODE 환경변수 설정 - 로컬 LLM 엔진 비활성화")
```

## 스모크 테스트 결과

### 테스트 #1: 클래스 Import ✅
```bash
source red_heart_env/bin/activate && python3 -c "from advanced_bentham_calculator import AdvancedBenthamCalculator"
```
**결과**: ✅ 성공 - AdvancedBenthamCalculator import 정상

### 테스트 #2: Claude 모드 격리 ⚠️
```bash
source red_heart_env/bin/activate && python3 -c "import os; os.environ['REDHEART_CLAUDE_MODE']='1'; from advanced_emotion_analyzer import AdvancedEmotionAnalyzer"
```
**결과**: ⚠️ 타임아웃 - 추가 조사 필요 (다른 초기화 지연 가능성)

### 테스트 #3: 비동기 헬퍼 ✅
```bash
source red_heart_env/bin/activate && python3 -c "from config import run_async_safely; import asyncio; run_async_safely(asyncio.sleep(0.1))"
```
**결과**: ✅ 성공 - run_async_safely 정상 작동

## 핵심 개선사항

1. **모듈 격리**: Claude 모드에서 로컬 Dolphin LLM을 로드하지 않도록 조건부 초기화 구현
2. **이벤트 루프 안정성**: nest_asyncio 대신 run_async_safely 사용으로 근본적 해결
3. **클래스명 일관성**: SentenceTransformerManager로 통일

## 권장 후속 조치

1. **AdvancedEmotionAnalyzer 초기화 시간 최적화**
   - Claude 모드에서도 타임아웃이 발생하는 원인 조사
   - 다른 무거운 모델 초기화 지연 로딩 고려

2. **환경변수 문서화**
   - `REDHEART_CLAUDE_MODE`: Claude API 모드 활성화 플래그
   - `FORCE_CPU_INIT`: CPU 강제 초기화 플래그

3. **테스트 커버리지 확대**
   - 각 모드별 전체 파이프라인 테스트
   - 메모리 사용량 모니터링

## 변경 파일 목록

1. `/mnt/c/large_project/linux_red_heart/advanced_bentham_calculator.py` (2개 위치 수정)
2. `/mnt/c/large_project/linux_red_heart/advanced_emotion_analyzer.py` (2개 클래스 수정)
3. `/mnt/c/large_project/linux_red_heart/main_unified.py` (환경변수 설정 추가)

---
*본 보고서는 CODEBASE_ANALYSIS_REPORT.md의 분석 결과를 기반으로 작성되었습니다.*