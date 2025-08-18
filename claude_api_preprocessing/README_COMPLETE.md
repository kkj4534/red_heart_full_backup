# Claude API 완전 전처리 시스템

## 개요
`data_preprocessing_pipeline_v3.py`의 모든 기능을 Claude API로 구현한 완전한 전처리 시스템

## 주요 기능
- ✅ **완전한 전처리**: 감정 분석 + 후회 지수 + 벤담 점수 + SURD 메트릭
- ✅ **체크포인트 시스템**: 중단 후 재개 가능
- ✅ **크레딧 소진 감지**: 자동 중단 및 재개
- ✅ **Rate Limit 안전 처리**: 3.5초 간격
- ✅ **실시간 비용 추적**: 토큰 및 비용 모니터링
- ✅ **성공/실패 분리 저장**

## 설치

```bash
pip install anthropic asyncio sentence-transformers torch
```

## 사용법

### 1. API 키 설정

`api_key.json` 파일 편집:
```json
{
  "anthropic_api_key": "sk-ant-api03-...",
  "note": "Replace with your actual Anthropic API key"
}
```

### 2. 데이터 준비

```bash
# Scruples 데이터셋 준비 (2만개)
python prepare_scruples_data.py
```

### 3. 테스트 실행 (권장)

```bash
# 3개 샘플로 전체 기능 테스트
python test_preprocessor.py
```

### 4. 전체 실행

```bash
# 20,000개 샘플 전처리
python claude_complete_preprocessor.py
```

## 출력 파일

- `claude_preprocessed_complete.jsonl`: 성공한 샘플
- `claude_failed_complete.jsonl`: 실패한 샘플
- `checkpoint_complete.pkl`: 체크포인트

## 출력 형식

```json
{
  "id": "anecdote_xxx",
  "text": "원본 텍스트...",
  "source": "anecdote",
  "emotion_vector": [0.1, 0.2, 0.15, 0.05, 0.35, 0.1, 0.05],
  "emotion_labels": {
    "joy": 0.1,
    "trust": 0.2,
    "fear": 0.15,
    "surprise": 0.05,
    "sadness": 0.35,
    "disgust": 0.1,
    "anger": 0.05
  },
  "regret_factor": 0.75,
  "bentham_scores": {
    "intensity": 0.3,
    "duration": 0.8,
    "certainty": 0.6,
    "propinquity": 0.5
  },
  "surd_metrics": {
    "selection": 0.7,
    "uncertainty": 0.6,
    "risk": 0.5,
    "decision": 0.8
  },
  "context_embedding": [768차원 벡터],
  "timestamp": "2025-08-16T00:00:00",
  "metadata": {
    "title": "AITA for...",
    "action": "hiding my controller",
    "label": "OTHER",
    "type": "anecdote"
  }
}
```

## 비용 계산

### 샘플당 토큰
- 입력: ~1,720 토큰 (4개 프롬프트)
- 출력: ~210 토큰

### 20,000개 샘플
- 입력: 34.4M 토큰 × $3/1M = $103.2
- 출력: 4.2M 토큰 × $15/1M = $63.0
- **기본 비용**: $166.2 (약 220,000원)

### 최적화 적용
- 프롬프트 캐싱: -80% 입력 비용
- 배치 처리: -50% 전체 비용
- **최종 예상**: $40-50 (약 53,000-66,000원)

## 시간 예상
- 샘플당: 3.5초 × 4 API 호출 = 14초
- 20,000개: 약 78시간 (3.2일)

## 중단 및 재개

### 크레딧 소진 시
1. 시스템이 자동으로 감지하고 중단
2. 체크포인트 자동 저장
3. 크레딧 충전 후:
```bash
python claude_complete_preprocessor.py
# "계속하시겠습니까? (y/n):" 프롬프트에 y 입력
```

### 수동 중단
- `Ctrl+C`로 중단 (체크포인트 자동 저장)
- 재실행하면 자동으로 이어서 처리

## 모니터링

10개 샘플마다 진행 상황 출력:
```
=== 진행 상황 ===
처리: 100/20000 (0.5%)
속도: 257.1 samples/hour
ETA: 77.2시간
토큰: 입력 172,000 / 출력 21,000
현재 비용: $0.83 (1,096원)
```

## 주의사항

1. **API 키 보안**: `api_key.json`을 git에 커밋하지 마세요
2. **비용 확인**: 테스트로 먼저 소량 실행 권장
3. **Rate Limit**: 분당 17회로 제한됨 (안전 마진)
4. **메모리**: 임베딩 생성으로 RAM 8GB+ 권장

## 문제 해결

### "Credit exhausted" 오류
- Anthropic 대시보드에서 크레딧 충전
- 재실행 시 자동으로 재개

### Rate limit 오류
- 자동으로 재시도됨
- 계속 발생 시 `rate_limit_delay` 증가

### 메모리 부족
- 100개 샘플마다 자동 정리
- 그래도 부족하면 배치 크기 감소