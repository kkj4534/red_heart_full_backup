# 🎯 Sweet Spot Analysis & Parameter Crossover Workflow

## 📋 Overview
730M 파라미터 모델의 모듈별 최적 성능 지점(Sweet Spot)을 탐색하고, 독립적인 모듈 그룹별로 최적 파라미터 셋을 조합하여 시스템 성능을 극대화하는 워크플로우

## 🏗️ 모듈 구조 (7개 독립 Sweet Spot)
```
1. Backbone (90.6M) - Group A 핵심
2. Heads 통합 (153M) - Group A 보조
3. Neural Analyzers (368M) - Group B
4. DSP + Kalman (16.3M) - Group C  
5-7. Advanced Wrappers (112M) - Independent × 4
```

## 📊 Phase 1: 데이터 수집 (60 Epoch 학습)
- [x] 2 epoch마다 checkpoint 저장 (총 30개)
- [x] 모듈별 독립적 state_dict 저장
- [x] Validation loss + Task-specific metrics 기록
- [x] Gradient norm, learning rate 추적

## 🔍 Phase 2: Sweet Spot 분석 (학습 완료 후)

### 2.1 자동 메트릭 분석 시스템
```python
# 실행 명령
python training/analyze_sweetspots.py --checkpoint-dir training/checkpoints_final --output analysis_results/

# 출력 파일
- analysis_results/module_metrics.json  # 모든 메트릭 데이터
- analysis_results/analysis_report.md   # 분석 리포트
- analysis_results/visualizations/      # 그래프 및 히트맵
```

### 2.2 분석 기법 (5가지)
1. **Statistical Plateau Detection**
   - Mann-Kendall 트렌드 테스트
   - CUSUM 변화점 탐지
   - Plateau 구간 중심점

2. **Task-Specific Metrics**
   - Emotion: F1-score, Confusion stability
   - Bentham: RMSE, Correlation
   - SURD: PID accuracy, Mutual information
   - Regret: Counterfactual accuracy

3. **Multi-Criteria Decision Analysis (MCDA)**
   - Loss (30%), Task (40%), Stability (15%), Gradient (15%)
   - 가중치 점수 통합

4. **Gradient Flow Health**
   - Gradient norm ratio
   - Update/parameter ratio
   - Dead neuron detection

5. **Ensemble Voting**
   - 각 기법의 추천 epoch
   - 신뢰도 가중 투표

## 🎯 Phase 3: 수동 분석 및 조합

### 3.1 AI 분석 요청
```bash
# 분석 결과를 AI에게 전달
"analysis_results/module_metrics.json 파일을 분석하여 
각 모듈별 최적 sweet spot을 근거와 함께 제시해주세요"
```

### 3.2 예상 분석 결과 형식
```
[Module: Backbone]
- 추천 Epoch: 44
- 근거:
  1. Plateau 구간 (42-48) 중심부
  2. Task F1-score 최고점 (0.924)
  3. Gradient stability 양호
- 신뢰도: 92%

[Module: Emotion Head]
- 추천 Epoch: 40
- 근거: ...
```

### 3.3 수동 파라미터 조합
```python
# 선택된 sweet spot으로 모델 조합
python training/combine_modules.py \
  --backbone checkpoint_epoch_44.pt \
  --heads checkpoint_epoch_40.pt \
  --analyzers checkpoint_epoch_46.pt \
  --output combined_model.pt
```

## 📈 Phase 4: Threshold 역산 및 자동화

### 4.1 성능 검증
```python
# 조합 모델 평가
python training/evaluate_combined.py --model combined_model.pt

# 비교군
- Baseline: 단일 epoch (30, 40, 50)
- Proposed: Sweet spot 조합
- Average: Plateau 평균
```

### 4.2 Threshold 값 역산
```python
# 실제 sweet spot 기반으로 threshold 계산
calibrated_thresholds = {
    'plateau_variance': 0.012,      # 실제 plateau 분산
    'improvement_rate': 0.008,      # epoch당 개선율
    'stability_window': 5,           # 안정성 판단 윈도우
    'mcda_weights': {
        'loss': 0.35,
        'task': 0.45,
        'stability': 0.10,
        'gradient': 0.10
    },
    'voting_confidence': 0.75        # 투표 신뢰도
}
```

### 4.3 자동화 시스템 활성화
```python
# threshold 값 적용하여 자동 sweet spot 탐색
python training/auto_sweetspot.py --thresholds calibrated_thresholds.json
```

## 🎯 Phase 5: 최종 결과

### 5.1 예상 성능 향상
- Validation Loss: -8~12% 개선
- Task Metrics: +5~10% 향상
- Generalization: 과적합 20% 감소

### 5.2 공모전 제출 포인트
1. **통계적 엄밀성**: Mann-Kendall, CUSUM 등 검증된 방법
2. **Empirical Validation**: 이론값이 아닌 실증 기반 threshold
3. **Modular Optimization**: 모듈별 독립 최적화
4. **Reproducibility**: 모든 과정 자동화 가능

## ⚠️ 주의사항
- GPU 메모리: 조합 시 일시적으로 2배 사용
- 검증 시간: 조합별 5-10분 소요
- Checkpoint 용량: 약 10GB (30개 × 350MB)

## 📝 TODO After Training
1. [ ] `analyze_sweetspots.py` 실행하여 메트릭 수집
2. [ ] AI 분석 요청 및 sweet spot 선정
3. [ ] 3-5개 조합 테스트
4. [ ] 최적 조합 확정
5. [ ] Threshold 값 역산
6. [ ] 자동화 시스템 검증
7. [ ] 최종 문서화

---
*Last Updated: 2025-08-19*
*Model: 730M RED_HEART_AI*
*Strategy: Parameter Crossover with Statistical Sweet Spot Detection*