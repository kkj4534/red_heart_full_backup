# 🎯 Hierarchical Learning Rate Sweep Strategy
## 5-5-5-5 Coarse-to-Fine Optimization

### 📊 전체 개요
점진적 범위 축소를 통한 효율적인 학습률 탐색 전략

```
Stage 0 (Base): 5 points  → 4 intervals
Stage 1: 5 points → Select top 2 intervals → 2 refined intervals  
Stage 2: 5 points → Select top 2 intervals → 2 refined intervals
Stage 3: 5 points → Select top 2 intervals → 2 refined intervals
Stage 4: 5 points → Final precision tuning
---
Total: 25 data points (vs 500+ for full grid search)
```

### 🔍 단계별 상세 전략

#### Stage 0: Initial Sweep (완료)
```python
initial_lrs = [1e-5, 5.6e-5, 3.2e-4, 1.8e-3, 1e-2]
# 결과: 5개 데이터 포인트 수집
```

#### Stage 1: First Refinement
```python
# 상위 2개 구간 선택 (예: [5.6e-5, 3.2e-4], [3.2e-4, 1.8e-3])
# 각 구간을 2-3개로 세분화
stage1_lrs = np.logspace(
    np.log10(5.6e-5), 
    np.log10(1.8e-3), 
    5
)
# 경계 중복 제거 후 5개 새 포인트
```

#### Stage 2-4: Progressive Refinement
```python
def generate_next_stage(best_intervals, n_points=5):
    """
    상위 성능 구간에서 새로운 탐색점 생성
    - 경계 중복 방지
    - 로그 스케일 균등 분포
    """
    new_points = []
    for (low, high) in best_intervals:
        points = np.logspace(
            np.log10(low),
            np.log10(high),
            n_points // len(best_intervals) + 1
        )[1:-1]  # 경계 제외
        new_points.extend(points)
    return new_points[:n_points]
```

### 📈 성능 메트릭 수집

각 단계에서 수집할 데이터:
1. **Loss Metrics**
   - Train/Val loss per epoch
   - Module-specific losses
   
2. **Convergence Indicators**
   - Loss reduction rate
   - Gradient norms
   - Stability scores

3. **Efficiency Metrics**
   - Time per epoch
   - Memory usage
   - Parameter utilization

### 🎨 시각화 계획

```python
# 1. Convergence Heatmap
# X축: Stage (0-4)
# Y축: Learning Rate (log scale)
# Color: Performance metric

# 2. Search Path Visualization
# 각 단계에서 선택된 구간 표시
# 최종 수렴 경로 하이라이트

# 3. Efficiency Comparison
# Grid Search: 500+ evaluations
# Hierarchical: 25 evaluations
# Time saved: ~95%
```

### 📊 논문 활용 방안

#### Methods Section
```latex
\subsection{Hierarchical Learning Rate Optimization}
We employed a coarse-to-fine adaptive grid search strategy,
progressively narrowing the search space based on empirical 
performance metrics. This approach reduced computational cost 
by approximately 95\% compared to exhaustive grid search 
(25 vs 500+ evaluations).
```

#### Results Section
- Stage-wise performance improvement graphs
- Variance reduction across stages
- Final optimal LR with confidence intervals

### 🚀 구현 계획

#### Phase 1: Infrastructure (30분)
```python
class HierarchicalLRSweep:
    def __init__(self, stages=4, points_per_stage=5):
        self.stages = stages
        self.points_per_stage = points_per_stage
        self.stage_results = []
        
    def analyze_stage(self, results):
        """현재 단계 결과 분석 및 다음 단계 구간 선택"""
        
    def generate_next_stage(self, top_intervals):
        """다음 단계 LR 포인트 생성"""
```

#### Phase 2: Execution (1시간)
- Stage 1: 15분 (5 points × 3 epochs)
- Stage 2: 15분
- Stage 3: 15분  
- Stage 4: 15분

#### Phase 3: Analysis (30분)
- 결과 집계 및 시각화
- 최적 LR 결정
- 보고서 생성

### 🎯 예상 결과

1. **정밀도**: ±5% 이내 최적 LR 발견
2. **효율성**: 95% 계산 비용 절감
3. **재현성**: 명확한 단계별 프로세스
4. **학술성**: 방법론 섹션 강화

### ⚙️ 기술적 고려사항

1. **중복 방지**: 이전 단계에서 평가한 LR 제외
2. **경계 처리**: 구간 경계는 한 번만 평가
3. **수렴 조건**: 성능 개선 < 1% 시 조기 종료
4. **메모리 효율**: 각 단계 결과만 저장 (누적 X)

### 📝 참고문헌
- Adaptive Grid Search methodologies
- Bayesian Optimization principles  
- Coarse-to-fine optimization strategies

---
*Last Updated: 2025-08-20*
*Status: Planning Phase*