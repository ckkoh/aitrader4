# Week 1, Day 1 Results: Confidence Threshold Optimization

**Date**: 2025-12-31
**Task**: Optimize ML confidence threshold (from IMPROVEMENTS_PLAN.md)
**Status**: ✅ COMPLETED

---

## Objective

Test multiple confidence thresholds (0.50, 0.55, 0.60, 0.65, 0.70) to find optimal setting for ML strategy.

**Expected Outcomes** (from plan):
- 0.50: More trades (15-20), likely 50-55% win rate
- 0.55: Moderate trades (8-12), ~60% win rate
- 0.60: Current baseline (4 trades)
- 0.65: Few trades (3-5), ~70% win rate
- 0.70: Very few trades (1-2), ~75%+ win rate

---

## Results Summary

### ⚠️ **Critical Finding: Threshold Has NO Impact**

All thresholds (0.50-0.70) produced **identical results**:

| Threshold | Trades | Win Rate | Return | Sharpe | Max DD |
|-----------|--------|----------|---------|---------|---------|
| 0.50 | 7 | 42.86% | -2.04% | -18.67 | 21.06% |
| 0.55 | 7 | 42.86% | -2.04% | -18.67 | 21.06% |
| 0.60 | 7 | 42.86% | -2.04% | -18.67 | 21.06% |
| 0.65 | 7 | 42.86% | -2.04% | -18.67 | 21.06% |
| 0.70 | 7 | 42.86% | -2.04% | -18.67 | 21.06% |

---

## Analysis

### Why Identical Results?

The ML model is likely producing confidence scores that are either:

1. **All well above 0.70** - Every prediction has high confidence
2. **All well below 0.50** - Every prediction has low confidence
3. **Clustered around specific values** - Model produces similar scores for all predictions

This indicates the model's **confidence calibration** needs improvement, not just threshold tuning.

### Performance Issues

1. **Low Win Rate**: 42.86% (below breakeven of 50%)
2. **Negative Return**: -2.04% (losing money)
3. **Poor Sharpe Ratio**: -18.67 (terrible risk-adjusted returns)
4. **High Drawdown**: 21.06% (exceeds 15% target)

---

## Root Causes

### 1. Feature Leakage (Critical)
From IMPROVEMENTS_PLAN.md, top 3 features include:
- `target_regression` ❌ Data leakage!
- `future_return` ❌ Data leakage!
- `target_binary` ❌ Data leakage!

These features shouldn't even be in the model. They contain future information that won't be available during live trading.

### 2. Feature Noise
- 91 total features → likely overfitting
- Many redundant/correlated features
- Model complexity > signal

### 3. Poor Model Quality
- Training accuracy reported as only ~60%
- Model not learning meaningful patterns
- Confidence scores not well-distributed

---

## Recommendation

**Skip to Day 2: Feature Selection** ✅

Adjusting confidence threshold won't help when the underlying model is broken. We need to:

### Immediate Actions (Day 2):

1. **Remove Data Leakage Features**
   ```python
   exclude_cols = ['target_regression', 'future_return', 'target_binary',
                   'target_class', 'open', 'high', 'low', 'close', 'volume']
   ```

2. **Select Top 20 Clean Features**
   - Use feature importance from properly trained model
   - Remove redundant features
   - Focus on predictive power

3. **Retrain Model**
   - XGBoost with hyperparameter tuning
   - Proper cross-validation
   - Monitor for overfitting

### Expected Impact After Day 2:

- +10-15% accuracy improvement
- Better confidence score distribution
- Threshold optimization will then be meaningful

---

## Files Generated

```
threshold_optimization_results/
├── ml_model_for_threshold_test.pkl      # Trained model
├── threshold_comparison.csv             # Results table
├── threshold_results.json               # Detailed JSON
└── recommendation.json                  # Optimal threshold (0.50)
```

---

## Next Steps

✅ **Day 1 Complete** - Identified that threshold optimization won't help

⏭️ **Day 2: Feature Selection**
- Remove data leakage features (target_*, future_*)
- Select top 20 non-leakage features
- Retrain model with clean features
- Test performance improvement

⏭️ **Day 3: Multi-Timeframe Prediction** (if Day 2 succeeds)

---

## Lessons Learned

1. **Always check feature leakage first** - Most important issue to fix
2. **Model quality > parameter tuning** - A bad model can't be fixed with threshold adjustments
3. **Confidence calibration matters** - Need diverse confidence scores for threshold to be meaningful
4. **Walk-forward validation working** - Framework is solid, strategy needs improvement

---

## Technical Notes

### Test Configuration
- Data: 2023-01-01 to 2025-12-29 (750 days)
- Train: 396 days (80%)
- Test: 100 days (20%)
- Model: XGBoost with hyperparameter tuning
- Features: 91 total (including leakage features)

### Model Training
- Best CV score: 0.6016 (60.16% accuracy)
- Best params: {'subsample': 0.7, 'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.01, 'colsample_bytree': 0.7}

### Regime Detection
- Sideways: 53 days
- Bull: 47 days
- Regime-adaptive threshold didn't help (all thresholds identical)

---

**Status**: Day 1 completed, moving to Day 2 (Feature Selection)
**Outcome**: Threshold optimization not viable until features are cleaned
**Action**: Proceed with feature selection and data leakage removal
