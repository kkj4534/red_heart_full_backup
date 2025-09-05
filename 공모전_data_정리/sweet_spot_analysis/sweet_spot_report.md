# 🎯 Sweet Spot Analysis Report

Generated: 2025-08-27 15:44:53

## 📊 Summary

| Module | Recommended Epoch | Confidence | Key Reasoning |
|--------|------------------|------------|---------------|
| neural_analyzers | 5 | 50.0% | N/A |
| regret_head | 39 | 25.0% | Plateau 구간 탐지 (Epoch 28-48) |
| emotion_head | 50 | 66.7% | MCDA 점수 우수 (0.850), 높은 투표 신뢰도 (66.7%) |
| surd_head | 40 | 25.0% | Plateau 구간 탐지 (Epoch 29-49), MCDA 점수 우수 (0.845) |
| backbone | 50 | 100.0% | 높은 투표 신뢰도 (100.0%) |
| system | 5 | 50.0% | MCDA 점수 우수 (0.906) |
| bentham_head | 48 | 66.7% | MCDA 점수 우수 (0.849), 높은 투표 신뢰도 (66.7%) |

## 🔍 Detailed Analysis

### Module: neural_analyzers

**Plateau Detection:** Not detected

**MCDA Analysis:**
- Best Epoch: 5
- Best Score: 0.784

**Ensemble Voting:**
- Selected: Epoch 5
- Confidence: 50.0%
- Votes: 1/2

---

### Module: regret_head

**Plateau Detection:**
- Range: Epoch 28-48
- Center: Epoch 38
- Mean Loss: 0.0004 (±0.0001)

**MCDA Analysis:**
- Best Epoch: 28
- Best Score: 0.586

**Ensemble Voting:**
- Selected: Epoch 39
- Confidence: 25.0%
- Votes: 1/4

---

### Module: emotion_head

**Plateau Detection:** Not detected

**MCDA Analysis:**
- Best Epoch: 50
- Best Score: 0.850

**Ensemble Voting:**
- Selected: Epoch 50
- Confidence: 66.7%
- Votes: 2/3

---

### Module: surd_head

**Plateau Detection:**
- Range: Epoch 29-49
- Center: Epoch 39
- Mean Loss: 0.0512 (±0.0001)

**MCDA Analysis:**
- Best Epoch: 28
- Best Score: 0.845

**Ensemble Voting:**
- Selected: Epoch 40
- Confidence: 25.0%
- Votes: 1/4

---

### Module: backbone

**Plateau Detection:** Not detected

**MCDA Analysis:**
- Best Epoch: 50
- Best Score: 0.600

**Ensemble Voting:**
- Selected: Epoch 50
- Confidence: 100.0%
- Votes: 2/2

---

### Module: system

**Plateau Detection:** Not detected

**MCDA Analysis:**
- Best Epoch: 5
- Best Score: 0.906

**Ensemble Voting:**
- Selected: Epoch 5
- Confidence: 50.0%
- Votes: 1/2

---

### Module: bentham_head

**Plateau Detection:** Not detected

**MCDA Analysis:**
- Best Epoch: 48
- Best Score: 0.849

**Ensemble Voting:**
- Selected: Epoch 48
- Confidence: 66.7%
- Votes: 2/3

---

## 🎯 Recommended Thresholds for Automation

```python
# Based on empirical analysis
thresholds = {
    'plateau_variance': 0.0001,
    'stability_window': 5,
    'mcda_weights': {
        'loss': 0.30,
        'accuracy': 0.40,
        'stability': 0.15,
        'gradient_health': 0.15
    },
    'min_confidence': 0.44
}
```

## 📝 Next Steps

1. Review the recommendations above
2. Manually combine modules using recommended epochs
3. Evaluate combined model performance
4. Adjust thresholds based on results
5. Enable automated sweet spot detection
