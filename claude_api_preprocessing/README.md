# Claude API 감정 분석 전처리 시스템

## 특징
- ✅ Rate limit 안전 처리 (3.5초 간격, 분당 17회)
- ✅ 자동 재시도 메커니즘
- ✅ 체크포인트 저장 (중단 후 재개 가능)
- ✅ 실시간 진행 상황 및 비용 추적
- ✅ 성공/실패 샘플 분리 저장
- ✅ Scruples 데이터셋 직접 연결

## 설치
```bash
pip install anthropic asyncio
```

## 사용법

### 1. 데이터 준비
```bash
# Scruples 데이터셋 준비 (2만개 샘플)
python prepare_scruples_data.py
```

### 2. API 키 설정
처음 실행 시 `config.json` 파일이 생성됩니다:
```json
{
  "api_key": "YOUR_ANTHROPIC_API_KEY_HERE",
  "input_file": "../for_learn_dataset/scruples_real_data/anecdotes/train.scruples-anecdotes.jsonl",
  "output_dir": ".",
  "max_samples": 20000,
  "rate_limit_delay": 3.5,
  "dataset_type": "anecdotes"
}
```

### 3. 실행
```bash
python claude_emotion_preprocessor.py
```

## 비용 예상
- 2만개 샘플: 약 $8-10 (배치 처리 시)
- 처리 시간: 약 19.4시간 (3.5초/샘플)

## 출력 파일
- `claude_preprocessed_dataset.jsonl`: 성공한 샘플
- `claude_failed_samples.jsonl`: 실패한 샘플
- `checkpoint.pkl`: 체크포인트 (자동 재개용)

## 출력 형식
### 성공 샘플
```json
{
  "id": "anecdote_yqk9ZSYu3Vd9A6sUBIGPLJ0mTCxcsv74",
  "text": "Backstory: So, I got an Xbox one for Christmas...",
  "emotions": [0.1, 0.2, 0.15, 0.05, 0.35, 0.1, 0.05],
  "emotion_labels": {
    "joy": 0.1,
    "trust": 0.2,
    "fear": 0.15,
    "surprise": 0.05,
    "sadness": 0.35,
    "disgust": 0.1,
    "anger": 0.05
  },
  "timestamp": "2025-08-15T22:00:00"
}
```

### 실패 샘플
```json
{
  "id": "sample_001",
  "text": "원본 텍스트...",
  "error": "Failed to extract emotions",
  "timestamp": "2025-08-15T22:00:00"
}
```

## 체크포인트
- 50개 샘플마다 자동 저장
- 중단 시 `checkpoint.pkl` 파일에서 재개
- 이미 처리된 샘플은 건너뜀

## 주의사항
- API 키를 안전하게 보관하세요
- Rate limit 초과 시 자동으로 30초 대기
- 네트워크 오류 시 3회까지 재시도