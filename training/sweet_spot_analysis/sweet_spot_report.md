# 🎯 Sweet Spot Analysis Report

Generated: 2025-08-23 09:53:12

## 📊 Summary

| Module | Recommended Epoch | Confidence | Key Reasoning |
|--------|------------------|------------|---------------|
| regret_head | 1 | 100.0% | MCDA 점수 우수 (0.850), 높은 투표 신뢰도 (100.0%) |
| backbone | 3 | 100.0% | 높은 투표 신뢰도 (100.0%) |
| emotion_head | 1 | 66.7% | 높은 투표 신뢰도 (66.7%) |
| surd_head | 2 | 66.7% | MCDA 점수 우수 (0.850), 높은 투표 신뢰도 (66.7%) |
| neural_analyzers | 1 | 50.0% | N/A |
| bentham_head | 1 | 66.7% | 높은 투표 신뢰도 (66.7%) |
| system | 3 | 100.0% | MCDA 점수 우수 (1.000), 높은 투표 신뢰도 (100.0%) |

## 🔍 Detailed Analysis

### Module: regret_head

**Plateau Detection:** Not detected

**MCDA Analysis:**
- Best Epoch: 1
- Best Score: 0.850

**Ensemble Voting:**
- Selected: Epoch 1
- Confidence: 100.0%
- Votes: 3/3

---

### Module: backbone

**Plateau Detection:** Not detected

**MCDA Analysis:**
- Best Epoch: 3
- Best Score: 0.600

**Ensemble Voting:**
- Selected: Epoch 3
- Confidence: 100.0%
- Votes: 2/2

---

### Module: emotion_head

**Plateau Detection:** Not detected

**MCDA Analysis:**
- Best Epoch: 1
- Best Score: nan

**Ensemble Voting:**
- Selected: Epoch 1
- Confidence: 66.7%
- Votes: 2/3

---

### Module: surd_head

**Plateau Detection:** Not detected

**MCDA Analysis:**
- Best Epoch: 2
- Best Score: 0.850

**Ensemble Voting:**
- Selected: Epoch 2
- Confidence: 66.7%
- Votes: 2/3

---

### Module: neural_analyzers

**Plateau Detection:** Not detected

**MCDA Analysis:**
- Best Epoch: 1
- Best Score: nan

**Ensemble Voting:**
- Selected: Epoch 1
- Confidence: 50.0%
- Votes: 1/2

---

### Module: bentham_head

**Plateau Detection:** Not detected

**MCDA Analysis:**
- Best Epoch: 1
- Best Score: 0.550

**Ensemble Voting:**
- Selected: Epoch 1
- Confidence: 66.7%
- Votes: 2/3

---

### Module: system

**Plateau Detection:** Not detected

**MCDA Analysis:**
- Best Epoch: 3
- Best Score: 1.000

**Ensemble Voting:**
- Selected: Epoch 3
- Confidence: 100.0%
- Votes: 2/2

---

## 🎯 Recommended Thresholds for Automation

```python
# Based on empirical analysis
thresholds = {
    'plateau_variance': 0.01,  # Default
    'stability_window': 5,
    'mcda_weights': {
        'loss': 0.30,
        'accuracy': 0.40,
        'stability': 0.15,
        'gradient_health': 0.15
    },
    'min_confidence': 0.63
}
```

## 📝 Next Steps

1. Review the recommendations above
2. Manually combine modules using recommended epochs
3. Evaluate combined model performance
4. Adjust thresholds based on results
5. Enable automated sweet spot detection
