# Red Heart XAI 후회 기반 학습 시스템

## 🎯 학습 준비 완료

시스템이 학습 가능한 상태로 준비되었습니다.

### 📊 학습 설정

- **후회 횟수**: 7회/스텝
- **벤담 쾌락 계산**: 3회/후회 (총 21회/스텝)
- **학습 선회**: 3번 (3 epochs)
- **로깅 주기**: 20스텝마다
- **스토리지 한계**: 200GB
- **모델 파라미터**: 2억개

### 🚀 학습 시작 방법

학습을 시작하려면 다음 명령을 실행하세요:

```bash
cd /mnt/c/large_project/linux_red_heart
python training/start_training.py
```

또는 Python에서 직접:

```python
from training.regret_based_training_pipeline import RegretTrainer, RegretTrainingConfig

# 설정 생성
config = RegretTrainingConfig(
    regrets_per_step=7,
    bentham_calculations_per_regret=3,
    epochs=3,
    log_every_n_steps=20,
    max_storage_gb=200.0
)

# 학습 실행
trainer = RegretTrainer(config)
report, checkpoint = trainer.train()
```

### 📊 데이터셋 정보

- **총 시나리오**: 28,882개
- **데이터 소스**: Scruples, 문학, 한국 문화, AI 생성
- **복잡도**: 69.4% 높은 복잡도, 30.6% 중간 복잡도

### 🔧 주요 구성 요소

#### 1. 후회 계산 시스템
- **5가지 후회 유형**: 반사실적, 시간적, 도덕적, 기회, 사회적
- **벤담 쾌락 계산**: 7가지 요소 (강도, 지속성, 확실성, 근접성, 다산성, 순수성, 범위)

#### 2. 모델 구조
- **메가 스케일 XAI 모델**: 33억 파라미터
- **계층적 감정 모델**: 다단계 감정 처리
- **XAI 추적**: 실시간 설명 생성
- **LLM 통합**: 언어 모델 연동

#### 3. 학습 최적화
- **그래디언트 체크포인팅**: 메모리 효율성
- **동적 스토리지 관리**: 자동 로그 정리
- **배치 처리**: 효율적인 데이터 로딩

### 📈 모니터링

학습 중 다음 메트릭이 모니터링됩니다:

- **손실 트렌드**: 총 손실, 분류 손실
- **후회 생성**: 스텝당 후회 수, 효율성
- **벤담 계산**: 계산 수, 평균 점수
- **시스템 자원**: 메모리, 스토리지 사용량
- **XAI 추적**: 로그 생성, 신뢰도

### 📚 결과 분석

학습 완료 후 자동으로 생성되는 문서:

1. **마크다운 리포트** (`docs/training_report_*.md`)
2. **HTML 대시보드** (`docs/training_report_*.html`)
3. **시각화** (`docs/visualizations/`)
4. **요약 JSON** (`docs/latest_training_summary.json`)

### 🔍 분석 내용

- **기본 메트릭**: 스텝 수, 후회 수, 벤담 계산, 학습 시간
- **효율성 분석**: 시간당 처리량, 자원 사용 효율성
- **후회 패턴**: 후회 생성 패턴, 벤담 계산 분포
- **성능 트렌드**: 손실 개선, 수렴 분석
- **권장사항**: 최적화 제안, 문제점 지적

### ⚠️ 주의사항

- **스토리지**: 200GB 한계 모니터링
- **메모리**: GPU 메모리 관리
- **중단**: Ctrl+C로 안전 중단 가능
- **체크포인트**: 자동 저장으로 중단 지점부터 재시작 가능

### 🛠️ 문제 해결

#### 메모리 부족
```python
# 배치 크기 감소
config.batch_size = 8
```

#### 스토리지 부족
```python
# 로깅 주기 증가
config.log_every_n_steps = 50
```

#### 학습 속도 개선
```python
# 그래디언트 체크포인팅 비활성화 (메모리 충분한 경우)
config.use_gradient_checkpointing = False
```

---

**준비 완료! 언제든지 학습을 시작하세요.**

```bash
python training/start_training.py
```