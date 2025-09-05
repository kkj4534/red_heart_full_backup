# Red Heart AI 통합 계획서
작성일: 2025-08-28

## 📋 현황 분석 및 통합 방향

### 1. EmotionEthicsRegretCircuit 재확인 결과

#### 실제 구현 상태
**파일**: `emotion_ethics_regret_circuit.py`
- **상태**: 완전 구현된 구식 시스템
- **구성**:
  - AdvancedEmotionAnalyzer (원본 파일 존재 ✅)
  - AdvancedBenthamCalculator (원본 파일 존재 ✅)
  - AdvancedRegretAnalyzer (GPU 가속, 원본 파일 존재 ✅)
- **핵심 원칙**:
  - 우선순위: 공동체 > 타자 > 자아
  - 치명적 손실 시 우선순위 역전
  - 경험 데이터베이스 연동

### 2. Advanced Analyzers 원본 파일 확인 ✅

**모든 원본 파일 실제 존재**:
- `advanced_emotion_analyzer.py` ✅
- `advanced_bentham_calculator.py` ✅
- `advanced_regret_analyzer.py` ✅ (GPU 가속 포함)
- `advanced_surd_analyzer.py` ✅
- `advanced_semantic_analyzer.py` ✅

따라서 Advanced Wrappers (112M)가 정상적으로 학습될 수 있었음

### 3. 두 가지 후회 시스템 비교

#### RegretHead (신규, UnifiedModel 내장)
```python
class RegretHead(MultiTaskHead):  # 30M 파라미터
    - 반사실적 추론 GRU (5.5M)
    - 3뷰 시나리오 네트워크 (8M)
    - 불확실성 정량화 (2M)
    - 시간 전파 LSTM (2M)
    
    # 50 epoch 학습됨
    # main_unified.py에서 사용 중
```

#### EmotionEthicsRegretCircuit (구식, 별도)
```python
class EmotionEthicsRegretCircuit:
    - AdvancedRegretAnalyzer (GPU 가속)
    - 감정-윤리-후회 삼각 회로
    - 공동체/타자/자아 계층 구조
    
    # main_unified.py에 통합 안 됨
    # 별도 시스템으로 존재
```

### 4. 문제점 분석

#### 현재 파이프라인 문제
```
텍스트 → UnifiedModel → 각 Head가 독립적으로 작동
              ↓
         EmotionHead → ? (연결 없음)
              ↓
         BenthamHead → ? (연결 없음)
              ↓
         RegretHead → ? (벤담 점수 입력 없음)
```

**핵심 문제**:
- EmotionHead → BenthamHead 직접 연결 없음
- RegretHead가 벤담 점수를 입력받지 않음
- 반사실 추론이 독립적으로 작동
- 시계열 전파가 통합되지 않음

---

## 🔧 통합 수정 방향

### 1. 통합 전략: 하이브리드 접근

```python
class UnifiedInferenceSystem:
    def __init__(self):
        # 기존 학습된 모델
        self.unified_model = UnifiedModel()  # 730M
        
        # Advanced Analyzers 전체 포함 (112M) - 필요시 활성화 아니라 항상 포함
        self.advanced_emotion = AdvancedEmotionAnalyzer()
        self.advanced_bentham = AdvancedBenthamCalculator()
        self.advanced_regret = AdvancedRegretAnalyzer()
        self.advanced_surd = AdvancedSURDAnalyzer()
        self.advanced_semantic = AdvancedSemanticAnalyzer()
        
        # 계층적 감정 처리 로직 (Circuit에서 차용)
        self.emotion_hierarchy = EmotionHierarchyProcessor()
        
        # 시계열 전파 분석기
        self.temporal_propagator = TemporalEventPropagationAnalyzer()
```

### 2. 수정된 파이프라인

```python
async def analyze(self, text):
    # 1. 기본 처리 (UnifiedModel)
    hidden_states = self.unified_model.backbone(text)
    
    # 2. 계층적 감정 분석 (Circuit 로직 + EmotionHead + Advanced)
    emotion_result = {
        'self': self.unified_model.emotion_head(hidden_states),
        'other': self.advanced_emotion.analyze_other(text),
        'community': self.advanced_emotion.analyze_community(text)
    }
    integrated_emotion = self.emotion_hierarchy.integrate(emotion_result)
    
    # 3. 감정 → 벤담 직접 연결 ✅
    bentham_input = self.emotion_to_bentham_converter(integrated_emotion)
    
    # 4. 시계열 전파 → 벤담 지속성 통합 ✅
    temporal_impact = self.temporal_propagator.analyze(bentham_input)
    bentham_input['duration'] = temporal_impact.long_term_effect
    bentham_input['fecundity'] = temporal_impact.cascade_potential
    
    # 5. 벤담 계산 (Head + Advanced 병렬)
    bentham_score = {
        'unified': self.unified_model.bentham_head(bentham_input),
        'advanced': self.advanced_bentham.calculate_enhanced(bentham_input)
    }
    final_bentham = self.merge_bentham_results(bentham_score)
    
    # 6. 반사실 추론
    counterfactuals = self.advanced_regret.generate_counterfactuals(
        decision=bentham_input,
        bentham_score=final_bentham
    )
    
    # 7. 후회 계산 (RegretHead + AdvancedRegret 병렬)
    regret_score = {
        'unified': self.unified_model.regret_head(counterfactuals),
        'advanced': self.advanced_regret.analyze_regret(counterfactuals)
    }
    
    # 8. SURD 분석
    surd_result = {
        'unified': self.unified_model.surd_head(hidden_states),
        'advanced': self.advanced_surd.analyze(text)
    }
    
    return self.integrate_all_results(
        emotion=integrated_emotion,
        bentham=final_bentham,
        regret=regret_score,
        surd=surd_result,
        temporal=temporal_impact
    )
```

### 3. 메모리 관리 전략

#### 4단계 메모리 모드
```python
class MemoryMode(Enum):
    LIGHT = "light"      # 230M (UnifiedModel Heads만)
    NORMAL = "normal"    # 400M (+ DSP/Kalman)
    HEAVY = "heavy"      # 600M (+ Neural Analyzers)
    ULTRA = "ultra"      # 730M + 112M = 842M (전체)

# 자동 모드 전환
def auto_select_mode(gpu_memory_available):
    if gpu_memory_available < 4000:  # 4GB
        return MemoryMode.LIGHT
    elif gpu_memory_available < 6000:  # 6GB
        return MemoryMode.NORMAL
    elif gpu_memory_available < 7000:  # 7GB
        return MemoryMode.HEAVY
    else:  # 8GB+
        return MemoryMode.ULTRA
```

### 4. 후회 학습 시스템

#### 유휴 시간 배치 학습
```python
class IdleTimeLearner:
    def __init__(self):
        self.idle_threshold = 3600  # 1시간
        self.last_interaction = time.time()
        self.regret_buffer = []
        
    async def monitor_and_learn(self):
        while True:
            await asyncio.sleep(60)  # 1분마다 체크
            
            if time.time() - self.last_interaction > self.idle_threshold:
                if len(self.regret_buffer) > 0:
                    # 배치 학습 수행
                    await self.batch_update_regret_policy(self.regret_buffer)
                    self.regret_buffer.clear()
```

---

## 📝 구체적 수정 작업 목록

### Phase 1: 즉시 수정 (1일)

#### 1. 감정 → 벤담 직접 연결 구현
```python
def emotion_to_bentham_converter(emotion):
    return {
        'intensity': emotion['joy'] - emotion['sadness'],
        'duration': emotion['stability'],
        'certainty': emotion['confidence'],
        'propinquity': emotion['urgency'],
        'fecundity': emotion['productivity'],
        'purity': emotion['purity'],
        'extent': emotion['scope']
    }
```

#### 2. 시계열 전파 → 벤담 통합
- temporal_event_propagation_analyzer.py 임포트
- 벤담 duration, fecundity 파라미터에 연결

#### 3. Advanced Analyzers 전체 활성화
- 112M 전체를 main_unified.py에 통합
- Wrapper 제거하고 원본 직접 사용

### Phase 2: 파이프라인 연결 (2일)

#### 4. 계층적 감정 처리 통합
- EmotionEthicsRegretCircuit에서 로직 추출
- 공동체 > 타자 > 자아 우선순위 구현
- 치명적 손실 감지 로직

#### 5. 반사실 → 후회 연결
- counterfactual 결과를 RegretHead 입력으로
- Advanced와 Unified 결과 병합

#### 6. 경험 데이터베이스 연동
- AdvancedExperienceDatabase 활성화
- 유사 경험 검색 및 활용

### Phase 3: 최적화 (3일)

#### 7. 메모리 모드 시스템 구현
- 4단계 모드 자동 전환
- Dynamic Swap Manager 통합

#### 8. 유휴 시간 학습 구현
- IdleTimeLearner 클래스
- 후회 버퍼 관리

#### 9. 성능 모니터링
- 각 모듈별 처리 시간 측정
- 병목 지점 식별 및 최적화

---

## 🗑️ 제거 대상

- **Fuzzy Emotion-Ethics Mapper**: 불필요한 디퓨전 유발
- **Ethics Policy Updater**: 후회 학습에 이미 포함
- **Advanced Wrappers**: 원본 직접 사용으로 대체

---

## 🏗️ 최종 아키텍처

```
입력 텍스트
    ↓
UnifiedModel Backbone (90.6M)
    ↓
[병렬 처리]
├─ EmotionHead (30M) + AdvancedEmotion + 계층적 통합
├─ BenthamHead (27M) + AdvancedBentham + 시계열 지속성
├─ RegretHead (30M) + AdvancedRegret (GPU 가속)
└─ SURDHead (22M) + AdvancedSURD
    ↓
통합 결과 (감정→벤담→반사실→후회 완전 연결)
    ↓
유휴 시간 학습 (배치 업데이트)
```

**총 파라미터**: 730M + 112M (Advanced) = 842M
**GPU 요구사항**: 8GB (Dynamic Swap으로 관리)

---

## 💡 핵심 인사이트

### 사용자 피드백 반영
1. **시계열 전파**: 벤담의 지속성(duration) 파라미터에 직접 통합
2. **Fuzzy Emotion Mapper**: 제거 (디퓨전만 유발)
3. **Advanced Analyzers**: 필요시 활성화가 아니라 **항상 포함**
4. **감정 → 벤담**: 직접 연결 (쾌락 = 감정의 함수)
5. **후회 학습**: 유휴 시간 배치 업데이트 (1시간 이상 대화 없을 때)

### 메모리 영향 분석
현재 사용: 539.9M / 730M

통합 시 추가:
- 계층적 감정 로직: ~5M (코드 로직)
- 경험 DB 연동: ~10M
- 시계열 전파: ~20M
- Advanced Analyzers: 112M

**총합**: ~687M (8GB GPU에서 충분)

---

## 🚀 실행 명령

```bash
# NumPy 설치 (가상환경 활성화 후)
source red_heart_env/bin/activate
pip install numpy==1.24.3

# 통합 시스템 실행
./run_inference.sh ultra  # 전체 모드 (842M)
./run_inference.sh heavy  # Neural Analyzers 포함 (600M)
./run_inference.sh normal # 기본 모드 (400M)
./run_inference.sh light  # 경량 모드 (230M)
```

---

## 📅 예상 일정

- **Day 1**: 감정→벤담 연결, 시계열 통합, Advanced 활성화
- **Day 2-3**: 파이프라인 연결, 계층적 감정, 반사실→후회
- **Day 4-6**: 메모리 최적화, 유휴 학습, 성능 모니터링

**총 예상 기간**: 6일

---

## ✅ 체크리스트

- [ ] NumPy 설치
- [ ] 감정 → 벤담 직접 연결
- [ ] 시계열 전파 → 벤담 지속성
- [ ] Advanced Analyzers 전체 활성화
- [ ] 계층적 감정 처리 (공동체>타자>자아)
- [ ] 반사실 → 후회 연결
- [ ] 경험 DB 연동
- [ ] 메모리 모드 시스템
- [ ] 유휴 시간 학습
- [ ] 성능 모니터링
- [ ] 테스트 및 검증