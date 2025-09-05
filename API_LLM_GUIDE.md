# API LLM 통합 가이드

## 🎯 개요
로컬 LLM의 GPU 메모리 문제를 해결하기 위해 외부 API를 통한 LLM 통합을 지원합니다.

## 📋 지원 API 및 프로토콜
- **GPT**: OpenAI GPT-4 Turbo
- **Claude**: Anthropic Claude 3 Opus
- **Perplexity**: Llama 3 Sonar (온라인 검색 지원)
- **DeepSeek**: DeepSeek Chat
- **MCP**: Model Context Protocol (Red Heart Ethics 서버)

## 🔑 API 키 설정

### 1. 설정 파일 수정
`api_key_manager/config.json` 파일을 열어 API 키를 입력:

```json
{
  "api_keys": {
    "gpt": {
      "api_key": "sk-xxxxxxxxxxxxx",  // OpenAI API 키
      "model": "gpt-4-turbo-preview",
      "base_url": "https://api.openai.com/v1",
      "max_tokens": 2048,
      "temperature": 0.7
    },
    "claude": {
      "api_key": "sk-ant-xxxxxxxxxxxxx",  // Anthropic API 키
      "model": "claude-3-opus-20240229",
      "base_url": "https://api.anthropic.com/v1",
      "max_tokens": 2048,
      "temperature": 0.7
    },
    "perplexity": {
      "api_key": "pplx-xxxxxxxxxxxxx",  // Perplexity API 키
      "model": "llama-3-sonar-large-32k-online",
      "base_url": "https://api.perplexity.ai",
      "max_tokens": 2048,
      "temperature": 0.7
    }
  }
}
```

### 2. 보안 주의사항
- ⚠️ `config.json`은 `.gitignore`에 자동 추가됨
- 절대 API 키를 git에 커밋하지 마세요
- 프로덕션에서는 환경 변수 사용 권장

## 🚀 사용 방법

### GPT-4 사용
```bash
python main_unified.py \
  --text "AI 윤리적 문제를 해결해야 합니다" \
  --memory-mode medium \
  --llm gpt \
  --debug
```

### Claude 3 사용
```bash
python main_unified.py \
  --text "AI 윤리적 문제를 해결해야 합니다" \
  --memory-mode medium \
  --llm claude \
  --debug
```

### Perplexity 사용 (온라인 검색 포함)
```bash
python main_unified.py \
  --text "최신 AI 윤리 가이드라인은 무엇인가요?" \
  --memory-mode medium \
  --llm perplexity \
  --debug
```

### DeepSeek 사용
```bash
python main_unified.py \
  --text "AI 윤리적 문제를 해결해야 합니다" \
  --memory-mode medium \
  --llm deepseek \
  --debug
```

### MCP (Model Context Protocol) 사용
MCP는 Red Heart 시스템 자체를 윤리적 추론 엔진으로 사용하는 프로토콜입니다.

#### 1. MCP 서버 시작
```bash
# 별도 터미널에서 실행
python mcp_server.py
```

#### 2. MCP 모드로 실행
```bash
python main_unified.py \
  --text "트롤리 딜레마를 해결해야 합니다" \
  --memory-mode medium \
  --llm mcp \
  --debug
```

## 📊 비교 분석

| 모드 | 장점 | 단점 | 추천 상황 |
|------|------|------|-----------|
| **local** | 무료, 오프라인 | GPU 메모리 필요, 품질 낮음 | 개발 테스트 |
| **gpt** | 최고 품질, 안정적 | 유료, 인터넷 필요 | 프로덕션, 발표 |
| **claude** | 윤리 특화, 한국어 우수 | 유료, 속도 느림 | 윤리적 분석 |
| **perplexity** | 실시간 검색, 최신 정보 | 유료 | 최신 정보 필요시 |
| **deepseek** | 저렴, 코딩 특화 | 한국어 약함 | 기술 문서 |
| **mcp** | Red Heart 자체 추론, 무료 | 서버 실행 필요 | 깊은 윤리 분석 |

## 🔧 필요 패키지 설치

```bash
# OpenAI (GPT, Perplexity, DeepSeek)
pip install openai

# Claude
pip install anthropic

# 또는 한번에
pip install openai anthropic
```

## 📈 성능 비교

### 로컬 모델 (Dolphin Llama3 8B)
- 처리 시간: ~230초
- GPU 메모리: 8GB 필요
- 응답 품질: 낮음
- 비용: 무료

### API 모델 (GPT-4)
- 처리 시간: ~3-5초
- GPU 메모리: 0 (불필요)
- 응답 품질: 최상
- 비용: ~$0.01/요청

## 🎯 학술 발표용 추천 설정

```bash
# 최고 품질 데모
python main_unified.py \
  --text "트롤리 딜레마에서 5명을 구하기 위해 1명을 희생시키는 것이 윤리적인가?" \
  --memory-mode heavy \
  --llm gpt \
  --debug

# 실시간 검색 데모
python main_unified.py \
  --text "2024년 최신 AI 윤리 규제는 무엇인가?" \
  --memory-mode medium \
  --llm perplexity \
  --debug
```

## ⚠️ 트러블슈팅

### API 키 오류
```
❌ gpt API 키가 설정되지 않음. config.json을 확인하세요.
```
→ `api_key_manager/config.json`에서 해당 API 키 입력

### 패키지 없음
```
❌ openai 패키지가 설치되지 않음. pip install openai
```
→ 필요 패키지 설치

### 네트워크 오류
```
❌ API 호출 실패: Connection error
```
→ 인터넷 연결 확인, API 서비스 상태 확인

## 📊 비용 예측

- **GPT-4**: ~$0.01-0.03 / 요청
- **Claude 3**: ~$0.015-0.025 / 요청
- **Perplexity**: ~$0.005-0.01 / 요청
- **DeepSeek**: ~$0.001-0.002 / 요청

학술 발표 데모 (100회): 약 $1-3

## 🚀 다음 단계

1. API 키 발급 (각 서비스 웹사이트)
2. `config.json` 수정
3. 테스트 실행
4. 학술 발표 준비

---

**주의**: API 사용 시 Red Heart GPU 스왑이 자동으로 비활성화되어 더 빠른 처리가 가능합니다.