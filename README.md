# Red Heart - 윤리적 의사결정 지원 시스템 (Linux)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Red Heart**는 다중 알고리즘을 통합한 윤리적 의사결정 분석 시스템입니다. 계층적 감정 학습, 후회 기반 학습, 베이지안 추론, 반사실적 추론, 벤담 공리주의 계산을 조합하여 윤리적 상황을 분석합니다.

## 📢 최신 업데이트 (2025-08-18)

### 🚀 730M 파라미터 통합 학습 시스템 구현 완료

- **60 에폭 학습 파이프라인**: LR 스윕 → 학습 → Sweet Spot 탐지 → Parameter Crossover
- **Advanced Training Techniques**: Label Smoothing, R-Drop, EMA, LLRD 통합
- **스마트 체크포인트**: 30개 체크포인트 자동 저장 (짝수 에폭마다)
- **OOM 핸들링**: 메모리 부족 시 자동 관리
- **자세한 실행 가이드**: [TRAINING_MODES_GUIDE.md](TRAINING_MODES_GUIDE.md) 참조

#### 빠른 시작:
```bash
# 시스템 검증 (파라미터 업데이트 없이)
SAMPLES=3 bash run_learning.sh unified-test --no-param-update --debug

# 전체 학습 (60 에폭, 2-3일 소요)
nohup bash run_learning.sh unified-train > training.log 2>&1 &
```

## 🔧 **주요 구성 요소**
- **3단계 계층적 감정 시스템**: Phase 0(자기 캘리브레이션) → Phase 1(타인 공감) → Phase 2(공동체 이해)
- **페이즈 기반 후회 학습**: 학습 횟수(50회) + 후회 임계값(0.3) 기반 전환
- **베이지안 추론 모듈**: 문학 기반 믿음 네트워크, Junction Tree 추론
- **LLM 통합 레이어**: 데이터 보강, 상황 시뮬레이션, 패턴 발견
- **벤담 계산기**: 7가지 기본 변수 + AI 기반 가중치 예측

## 🌟 주요 특징

### 🧠 고급 AI 기반 분석
- **트랜스포머 모델**: 멀티링구얼 BERT, RoBERTa, Korean-specific models
- **신경망 인과 모델**: 딥러닝 기반 인과관계 예측
- **Sentence Transformers**: 고차원 의미 임베딩
- **어텐션 메커니즘**: 중요 정보에 집중하는 분석

### 📊 핵심 분석 엔진

#### 1. 계층적 감정 시스템 (3-Phase Learning)
- **6차원 감정 벡터**: Valence, Arousal, Dominance, Certainty, Surprise, Anticipation
- **Phase 0 (감정 캘리브레이션)**: 타자 감정을 자신에게 투영, 캘리브레이션 계수 학습
- **Phase 1 (공감 학습)**: 후회 알고리즘 기반 타자 경험 학습, 예측 오차 개선
- **Phase 2 (공동체 확장)**: 개인 감정을 공동체 수준으로 확장, 합의도 분석
- **문학적 감정 데이터베이스**: tragedy, comedy, romance 패턴 기반 학습

#### 2. 후회 학습 시스템 (Regret-Based Learning)
- **6가지 후회 유형**: ACTION, INACTION, TIMING, CHOICE, EMPATHY, PREDICTION
- **페이즈 전환 조건**: 최소 50회 학습 + 후회 임계값 0.3 이하
- **다층적 학습**: 상위 페이즈에서 하위 페이즈 후회도 지속 반영 (가중치 감소)
- **시계열 패턴**: 문학 데이터 기반 시간적 학습 추적
- **메타 패턴 발견**: 패턴 공존, 패턴 시퀀스, 이상치 탐지

#### 3. 베이지안 추론 모듈 (Independent Bayesian Inference)
- **5가지 믿음 노드**: FACTUAL, MORAL, EMOTIONAL, SOCIAL, PREDICTIVE
- **문학적 사전확률**: tragic_fate, redemption, love_conquers, karma 패턴
- **Variable Elimination**: Junction Tree 알고리즘 기반 정확한 추론
- **불확실성 정량화**: 엔트로피 기반 불확실성 측정
- **상호정보량 분석**: 노드 간 영향력 정량 평가

#### 4. 벤담 계산기 (AI-Enhanced Bentham Calculator)
- **7가지 기본 변수**: 강도, 지속성, 확실성, 근접성, 풍요성, 순수성, 확장성
- **NeuralWeightPredictor**: 6개 가중치 레이어의 AI 기반 예측 (0.3~2.5 범위)
- **TransformerContextAnalyzer**: BERT, RoBERTa, KcELECTRA 기반 맥락 분석
- **6개 추가 레이어**: 문화적, 시간적, 사회적, 개인적, 도덕적, 상황적 가중치
- **극단값 보정**: 복합 상황에서 수치 안정성 보장

#### 5. SURD 시스템 (Synergy, Unique, Redundant, Deterministic)
- **Kraskov k-NN 상호정보량**: k=5 기준 정확한 추정
- **Williams-Beer PID 분해**: Synergy, Unique, Redundant, Deterministic 요소
- **NeuralCausalModel**: 128-64-32 차원 신경망 기반 인과관계 예측
- **Transfer Entropy**: 시간적 인과관계 분석
- **인과 네트워크**: NetworkX 기반 복잡한 인과구조 시각화

#### 6. LLM 통합 레이어 (Data Enhancement & Pattern Discovery)
- **DataEnrichmentLLM**: 누락된 감정 차원 추론, 시간적 동태 예측
- **RumbaughSimulationLLM**: OMT 기반 상황 시뮬레이션, 5단계 진화 모델
- **PatternDiscoveryLLM**: 시간적/구조적/의미적 패턴 발견, 메타 패턴 추출
- **데이터 손실 방지**: 정보 보강을 통한 모듈 간 연결 강화

## 🚀 빠른 시작

### 시스템 요구사항

#### 최소 요구사항
- **OS**: Ubuntu 20.04+ / CentOS 8+ / Debian 11+ / WSL2
- **Python**: 3.8 이상
- **RAM**: 8GB 이상
- **Storage**: 20GB 이상 (모델 다운로드 포함)

#### 권장 사양 (테스트 환경)
- **OS**: Ubuntu 22.04 LTS / Windows 11 WSL2
- **Python**: 3.10+
- **RAM**: 16GB (현재 테스트 환경)
- **GPU**: NVIDIA GPU (선택사항, CPU 전용 모드 지원)
- **Storage**: 70GB 여유 공간 (현재 테스트 환경)

### 1. 저장소 클론

```bash
git clone https://github.com/kkj4534/red_heart_full_backup.git
cd red_heart_full_backup
```

### 2. 가상환경 설정

```bash
# Python 가상환경 생성
python3 -m venv venv

# 가상환경 활성화
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 3. 시스템 확인

```bash
# 시스템 환경 확인
./run_red_heart.sh --check-only
```

### 4. 실행

```bash
# 데모 모드 실행
./run_red_heart.sh --demo

# 텍스트 분석
./run_red_heart.sh --text "분석할 윤리적 상황을 입력하세요"

# Python으로 직접 실행
python main.py --demo
```

## 📖 사용법

### 기본 사용법

```python
from main import RedHeartSystem, AnalysisRequest
import asyncio

async def analyze_situation():
    # 시스템 초기화
    system = RedHeartSystem()
    await system.initialize()
    
    # 분석 요청 생성
    request = AnalysisRequest(
        text="이 결정은 많은 사람들의 생명과 안전에 영향을 미칩니다.",
        language="ko",
        scenario_type="ethical_dilemma",
        include_emotion=True,
        include_bentham=True,
        include_semantic=True,
        include_surd=True
    )
    
    # 분석 실행
    result = await system.analyze_async(request)
    
    # 결과 출력
    print(f"통합 점수: {result.integrated_score:.3f}")
    print(f"신뢰도: {result.confidence:.3f}")
    print(f"추천: {result.recommendation}")

# 실행
asyncio.run(analyze_situation())
```

### 고급 사용법

```python
# 개별 분석기 사용
from advanced_emotion_analyzer import AdvancedEmotionAnalyzer
from advanced_bentham_calculator import AdvancedBenthamCalculator

# 감정 분석만 실행
emotion_analyzer = AdvancedEmotionAnalyzer()
emotion_result = emotion_analyzer.analyze_text_advanced(
    text="복잡한 감정이 담긴 텍스트",
    language="ko",
    context={"domain": "ethics"}
)

# 벤담 계산만 실행  
bentham_calculator = AdvancedBenthamCalculator()
bentham_result = bentham_calculator.calculate_with_advanced_layers({
    'input_values': {
        'intensity': 0.8,
        'duration': 0.7,
        'certainty': 0.9,
        # ... 기타 변수들
    },
    'text_description': "윤리적 딜레마 상황",
    'language': 'ko'
})
```

## 🏗️ 시스템 아키텍처

```
Red Heart System (139 Python 모듈)
├── 🧠 계층적 감정 시스템
│   ├── Phase0EmotionCalibrator (타자→자신 투영)
│   ├── Phase1EmpathyLearner (공감 학습 + 후회 알고리즘)
│   ├── Phase2CommunityExpander (공동체 확장)
│   └── SentenceTransformer (다국어 임베딩)
├── 📚 후회 학습 시스템
│   ├── RegretMemory (6가지 후회 유형)
│   ├── PhaseTransition (페이즈 전환 로직)
│   ├── 시계열 패턴 분석
│   └── 메타 패턴 발견
├── 🔮 베이지안 추론 모듈
│   ├── BayesianNode (5가지 믿음 유형)
│   ├── LiteraryBeliefNetwork (문학적 사전확률)
│   ├── Variable Elimination
│   └── 불확실성 정량화
├── ⚖️ 벤담 계산기
│   ├── NeuralWeightPredictor (AI 가중치)
│   ├── TransformerContextAnalyzer (BERT/RoBERTa)
│   ├── 7가지 기본 변수
│   └── 6개 추가 레이어
├── 🔗 SURD 분석기
│   ├── Kraskov k-NN 추정기
│   ├── NeuralCausalModel (128-64-32)
│   ├── Williams-Beer PID
│   └── Transfer Entropy
├── 🤖 LLM 통합 레이어
│   ├── DataEnrichmentLLM
│   ├── RumbaughSimulationLLM
│   └── PatternDiscoveryLLM
└── 📊 통합 시스템
    ├── RedHeartSystem (메인 통합)
    ├── 비동기 처리
    ├── GPU/CPU 자동 감지
    └── 캐시 시스템
```

## 📊 성능 벤치마크

### 예상 성능 (테스트 환경 기준)

| 구성 요소 | CPU 전용 | GPU 가속 | 주요 병목 |
|----------|---------|----------|----------|
| **계층적 감정 학습** | 1-3초 | 0.5-1초 | SentenceTransformer 임베딩 |
| **후회 학습 시스템** | 0.5-1초 | 0.2-0.5초 | 패턴 분석 알고리즘 |
| **베이지안 추론** | 0.3-0.8초 | 0.1-0.3초 | Variable Elimination |
| **벤담 계산기** | 1-2초 | 0.3-0.8초 | Transformer 모델 |
| **SURD 분석** | 3-8초 | 1-3초 | Kraskov k-NN 계산 |
| **LLM 통합 레이어** | 2-5초 | 0.8-2초 | 패턴 발견 알고리즘 |

### 메모리 사용량 (현재 구현 기준)

| 모드 | 시스템 RAM | 모델 크기 | 데이터셋 |
|------|------------|----------|----------|
| **기본 모드** | 4-6GB | 2-4GB | 기본 학습 데이터 |
| **표준 모드** | 8-12GB | 6-10GB | 전체 문학 데이터 |
| **고급 모드** | 12-16GB | 10-15GB | 확장 데이터셋 |

**현재 테스트 환경**: 16GB RAM, 70GB 저장공간 여유

## 🔧 구성 옵션

### config.py 주요 설정

```python
# GPU 사용 설정
ADVANCED_CONFIG = {
    'enable_gpu': True,              # GPU 사용 여부
    'use_transformers': True,        # 트랜스포머 모델 사용
    'use_neural_causal_model': True, # 신경망 인과 모델 사용
    'parallel_processing': True,     # 병렬 처리 사용
    'batch_size': 32,               # 배치 크기
    'fallback_mode': False,         # 폴백 모드 (고급 기능 비활성화)
}

# 모델 경로
MODELS_DIR = "./models"
LOGS_DIR = "./logs"
CACHE_DIR = "./cache"

# 성능 튜닝
PERFORMANCE_CONFIG = {
    'max_sequence_length': 512,     # 최대 시퀀스 길이
    'num_workers': 4,               # 워커 스레드 수
    'cache_size': 1000,             # 캐시 크기
    'timeout_seconds': 300,         # 타임아웃 (초)
}
```

## 🧪 테스트

### 단위 테스트 실행

```bash
# 모든 테스트 실행
python -m pytest tests/ -v

# 특정 모듈 테스트
python -m pytest tests/test_emotion_analyzer.py -v
python -m pytest tests/test_bentham_calculator.py -v
python -m pytest tests/test_semantic_analyzer.py -v
python -m pytest tests/test_surd_analyzer.py -v
```

### 성능 테스트

```bash
# 벤치마크 실행
python benchmark.py --iterations 100

# 메모리 프로파일링
python -m memory_profiler main.py --demo
```

### 통합 테스트

```bash
# 전체 시스템 테스트
python test_integration.py

# 특정 시나리오 테스트
python test_scenarios.py --scenario ethical_dilemma
```

## 📚 API 문서

### 주요 클래스

#### RedHeartSystem
메인 통합 시스템

```python
class RedHeartSystem:
    async def initialize()                    # 시스템 초기화
    async def analyze_async(request)          # 비동기 분석
    def get_system_status()                   # 시스템 상태 조회
    def clear_cache()                         # 캐시 클리어
```

#### AnalysisRequest
분석 요청 데이터

```python
@dataclass
class AnalysisRequest:
    text: str                                 # 분석할 텍스트
    language: str = "ko"                      # 언어
    scenario_type: str = "general"            # 시나리오 타입
    include_emotion: bool = True              # 감정 분석 포함
    include_bentham: bool = True              # 벤담 계산 포함
    include_semantic: bool = True             # 의미 분석 포함
    include_surd: bool = True                 # SURD 분석 포함
```

#### IntegratedResult
통합 분석 결과

```python
@dataclass
class IntegratedResult:
    request: AnalysisRequest                  # 원본 요청
    emotion_analysis: Optional[Any]           # 감정 분석 결과
    bentham_analysis: Optional[Any]           # 벤담 분석 결과
    semantic_analysis: Optional[Any]          # 의미 분석 결과
    surd_analysis: Optional[Any]              # SURD 분석 결과
    integrated_score: float                   # 통합 점수
    recommendation: str                       # 추천사항
    confidence: float                         # 신뢰도
    processing_time: float                    # 처리 시간
```

## 🐛 문제 해결

### 일반적인 문제들

#### 1. GPU 메모리 부족
```bash
# 배치 크기 줄이기
export CUDA_VISIBLE_DEVICES=0
python main.py --demo --batch-size 8
```

#### 2. 모델 다운로드 실패
```bash
# 수동 모델 다운로드
python -c "
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained('klue/bert-base')
tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
"
```

#### 3. 의존성 충돌
```bash
# 가상환경 재생성
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 4. 성능 저하
```bash
# 성능 모니터링
pip install psutil nvidia-ml-py3
python monitor_performance.py
```

### 로그 분석

```bash
# 실시간 로그 모니터링
tail -f logs/red_heart_linux.log

# 에러 로그 필터링
grep "ERROR" logs/red_heart_linux.log

# 성능 로그 분석
grep "processing_time" logs/red_heart_linux.log | awk '{print $NF}'
```

## 🤝 기여하기

### 개발 환경 설정

```bash
# 개발용 의존성 설치
pip install -r requirements-dev.txt

# 프리커밋 훅 설치
pre-commit install

# 코드 포맷팅
black .
isort .
flake8 .
```

### 기여 가이드라인

1. **이슈 생성**: 버그 리포트나 기능 요청
2. **포크 & 브랜치**: 새로운 브랜치에서 작업
3. **테스트**: 모든 테스트 통과 확인
4. **PR 제출**: 상세한 설명과 함께

### 코딩 표준

- **Python**: PEP 8 준수
- **타입 힌트**: 모든 함수에 타입 어노테이션
- **독스트링**: Google 스타일 독스트링
- **테스트**: 최소 80% 코드 커버리지

## 📄 라이선스

MIT License - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 🙏 감사의 말

이 프로젝트는 다음 오픈소스 프로젝트들을 기반으로 합니다:

- [Transformers](https://github.com/huggingface/transformers) - Hugging Face
- [PyTorch](https://pytorch.org/) - Meta AI
- [Sentence Transformers](https://www.sbert.net/) - UKP Lab
- [scikit-learn](https://scikit-learn.org/) - scikit-learn developers
- [NetworkX](https://networkx.org/) - NetworkX Developers

## 📞 지원 및 연락

- **이슈 트래커**: [GitHub Issues](https://github.com/kkj4534/red_heart_full_backup/issues)
- **토론**: [GitHub Discussions](https://github.com/kkj4534/red_heart_full_backup/discussions)
- **이메일**: memento1087@gmail.com

---

**Red Heart**로 더 나은 윤리적 의사결정을 시작하세요! 🔴❤️