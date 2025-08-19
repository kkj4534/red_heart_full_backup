# 🎯 Sweet Spot Analysis Report

Generated: 2025-08-19 16:11:24

## 📊 Summary

| Module | Recommended Epoch | Confidence | Key Reasoning |
|--------|------------------|------------|---------------|
| backbone | 1 | 100.0% | 높은 투표 신뢰도 (100.0%) |
| emotion_head | 1 | 100.0% | 높은 투표 신뢰도 (100.0%) |
| bentham_head | 1 | 100.0% | 높은 투표 신뢰도 (100.0%) |
| regret_head | 1 | 100.0% | 높은 투표 신뢰도 (100.0%) |
| surd_head | 1 | 100.0% | 높은 투표 신뢰도 (100.0%) |
| neural_analyzers | 1 | 100.0% | 높은 투표 신뢰도 (100.0%) |
| system | 1 | 100.0% | 높은 투표 신뢰도 (100.0%) |

## 🔍 Detailed Analysis

### Module: backbone

**Plateau Detection:** Not detected

**MCDA Analysis:**
- Best Epoch: 1
- Best Score: 0.450

**Ensemble Voting:**
- Selected: Epoch 1
- Confidence: 100.0%
- Votes: 2/2

---

### Module: emotion_head

**Plateau Detection:** Not detected

**MCDA Analysis:**
- Best Epoch: 1
- Best Score: 0.450

**Ensemble Voting:**
- Selected: Epoch 1
- Confidence: 100.0%
- Votes: 3/3

---

### Module: bentham_head

**Plateau Detection:** Not detected

**MCDA Analysis:**
- Best Epoch: 1
- Best Score: 0.450

**Ensemble Voting:**
- Selected: Epoch 1
- Confidence: 100.0%
- Votes: 3/3

---

### Module: regret_head

**Plateau Detection:** Not detected

**MCDA Analysis:**
- Best Epoch: 1
- Best Score: 0.450

**Ensemble Voting:**
- Selected: Epoch 1
- Confidence: 100.0%
- Votes: 3/3

---

### Module: surd_head

**Plateau Detection:** Not detected

**MCDA Analysis:**
- Best Epoch: 1
- Best Score: 0.450

**Ensemble Voting:**
- Selected: Epoch 1
- Confidence: 100.0%
- Votes: 3/3

---

### Module: neural_analyzers

**Plateau Detection:** Not detected

**MCDA Analysis:**
- Best Epoch: 1
- Best Score: 0.450

**Ensemble Voting:**
- Selected: Epoch 1
- Confidence: 100.0%
- Votes: 2/2

---

### Module: system

**Plateau Detection:** Not detected

**MCDA Analysis:**
- Best Epoch: 1
- Best Score: 0.411

**Ensemble Voting:**
- Selected: Epoch 1
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
    'min_confidence': 0.80
}
```

## 📝 Next Steps

1. Review the recommendations above
2. Manually combine modules using recommended epochs
3. Evaluate combined model performance
4. Adjust thresholds based on results
5. Enable automated sweet spot detection
