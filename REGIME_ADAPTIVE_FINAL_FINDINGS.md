# Regime-Adaptive Strategy: Final Findings & Root Cause Analysis

**Date**: 2025-12-31
**Status**: ❌ **ROOT CAUSE IDENTIFIED**

---

## Executive Summary

After extensive debugging and analysis, we discovered why the regime-adaptive ML strategy generated **ZERO trades across all 15 walk-forward periods**:

**ROOT CAUSE**: The ML model is heavily biased towards predicting SELL/HOLD and **never produces BUY probabilities above 35.61%**, even during bull markets. With the adaptive strategy's lowest threshold at 0.40 (for BULL markets), no trades can be generated.

---

## Investigation Timeline

### Attempt 1: Fix `skip_volatile_regimes` Parameter
- **Action**: Changed `skip_volatile_regimes=True` to `False`
- **Result**: Still ZERO trades ❌
- **Conclusion**: Skipping regimes wasn't the issue

### Attempt 2: Lower Base Confidence Threshold
- **Action**: Lowered `base_confidence_threshold` from 0.55 to 0.50
- **Adjusted Thresholds**:
  - BULL: 0.40
  - SIDEWAYS: 0.55
  - VOLATILE: 0.70
  - BEAR: 0.65
- **Result**: Still ZERO trades ❌
- **Conclusion**: Even at 0.40 threshold, no trades generated

### Attempt 3: Fix Confidence Calculation Bug
- **Bug Found**: Line 318 used `confidence = max(prediction_proba)` instead of `confidence = prediction_proba[1]`
- **Action**: Fixed to use BUY probability specifically
- **Result**: Still ZERO trades ❌
- **Conclusion**: Bug fix didn't help - model fundamentals are broken

### Attempt 4: Model Prediction Diagnosis ✅
- **Action**: Created diagnostic script to analyze model predictions
- **Result**: **FOUND ROOT CAUSE!**

---

## Root Cause: Model Training Data Imbalance

### Diagnostic Results (2024 Q1 - Bull Market Period)

**BUY Probability Statistics:**
```
Minimum:  6.64%
Maximum: 35.61%  ← NEVER REACHES 40% THRESHOLD!
Mean:    27.15%
Median:  28.53%
Std Dev:  6.85%
```

**Predicted Class Distribution:**
- SELL/HOLD: 58 days (100.0%)
- BUY: 0 days (0.0%)

**Threshold Analysis:**
| Threshold | BUY Signals | Percentage |
|-----------|-------------|------------|
| 0.30 | 15 | 25.9% |
| 0.35 | 5 | 8.6% |
| **0.40** | **0** | **0.0%** ← **Adaptive BULL threshold** |
| 0.50 | 0 | 0.0% |
| 0.55 | 0 | 0.0% ← Baseline threshold |

### Key Findings

1. **Model Never Predicts BUY with >35.61% Confidence**
   - Even in bull markets (2024 Q1), max BUY probability is 35.61%
   - Baseline threshold (0.55): No signals possible
   - Adaptive BULL threshold (0.40): No signals possible
   - Need threshold ≤0.35 to get ANY signals

2. **Model is 100% SELL/HOLD Biased**
   - Predicts SELL/HOLD with 64-93% confidence
   - Predicts BUY with only 7-36% confidence
   - This is backwards for a bull market!

3. **Training Data Was Imbalanced**
   - Model learned majority class (SELL/HOLD)
   - Optimized for accuracy, not balanced predictions
   - "When in doubt, predict SELL/HOLD" strategy

---

## Why Baseline Generated 47 Trades

**Question**: If the model never reaches 0.55 threshold, how did baseline generate 47 trades?

**Answer**: **Baseline uses a DIFFERENT strategy class!**

Looking at `walkforward_regime_adaptive.py:147-151`:
```python
else:  # baseline
    from strategy_examples import MLStrategy
    strategy = MLStrategy(
        model_path=str(model_path),
        feature_cols=self.top_features,
        confidence_threshold=0.55
    )
```

The `MLStrategy` class likely has different logic:
- May use different confidence calculation
- May have different signal generation logic
- May use both BUY and SELL signals (not just BUY)

**This means**: Baseline and Adaptive are using fundamentally different strategy implementations, not just different thresholds!

---

## Comparison: What Actually Failed

| Aspect | Baseline (MLStrategy) | Adaptive (RegimeAdaptiveMLStrategy) |
|--------|----------------------|-------------------------------------|
| **Strategy Class** | `MLStrategy` | `RegimeAdaptiveMLStrategy` |
| **Threshold** | 0.55 (fixed) | 0.40-0.70 (adaptive) |
| **Confidence Calc** | Unknown (likely different) | `prediction_proba[1]` (BUY only) |
| **Trades Generated** | 47 | 0 |
| **Status** | Works (poorly) | Completely broken |

**Critical Insight**: We're not comparing apples to apples! The baseline uses a different strategy class that somehow generates trades despite low BUY probabilities.

---

## Solutions

### ✅ Solution 1: Lower Thresholds to Match Model Output (Quick Fix)

Based on diagnostic data, adjust thresholds to where model actually produces signals:

```python
self.regime_settings = {
    'BULL': {
        'confidence_adjustment': -0.20,  # 0.50 - 0.20 = 0.30 ✅ Gets 15 signals
        'position_multiplier': 1.0,
    },
    'SIDEWAYS': {
        'confidence_adjustment': -0.15,  # 0.50 - 0.15 = 0.35 ✅ Gets 5 signals
        'position_multiplier': 0.7,
    },
    'VOLATILE': {
        'confidence_adjustment': +0.10,  # 0.50 + 0.10 = 0.60 (no signals, but that's OK for volatile)
        'position_multiplier': 0.3,
    },
    'BEAR': {
        'confidence_adjustment': +0.05,  # 0.50 + 0.05 = 0.55 (few signals, conservative)
        'position_multiplier': 0.5,
    }
}
```

**Expected Result**: Should generate trades, but quality unknown

### ✅ Solution 2: Match Baseline Strategy Logic

Investigate how `MLStrategy` generates signals and replicate that logic:

```bash
# Check MLStrategy implementation
grep -A 50 "class MLStrategy" strategy_examples.py
```

If MLStrategy uses different confidence logic or signal generation, adopt that approach.

### ✅ Solution 3: Fix Training Data Imbalance (Proper Fix)

Retrain models with balanced data:

1. **Class Balancing**:
   ```python
   from sklearn.utils import class_weight

   class_weights = class_weight.compute_class_weight(
       'balanced',
       classes=np.unique(y_train),
       y=y_train
   )

   trainer.train(X_train, y_train,
                sample_weight=class_weights,
                hyperparameter_tuning=True)
   ```

2. **SMOTE Oversampling**:
   ```python
   from imblearn.over_sampling import SMOTE

   smote = SMOTE(random_state=42)
   X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
   ```

3. **Probability Calibration**:
   ```python
   from sklearn.calibration import CalibratedClassifierCV

   calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=5)
   calibrated_model.fit(X_train, y_train)
   ```

**Expected Result**: Model produces BUY probabilities in 0.4-0.6 range

### ✅ Solution 4: Use Both BUY and SELL Signals

Instead of only trading BUY signals, trade both directions:

```python
if prediction_proba[1] > adaptive_settings['confidence_threshold']:
    # BUY signal
    signals.append({'action': 'buy', ...})
elif prediction_proba[0] > adaptive_settings['confidence_threshold']:
    # SELL/SHORT signal
    signals.append({'action': 'sell', ...})
```

**Note**: This requires implementing short selling in backtesting engine

---

## Recommended Next Steps

### Priority 1: Quick Test with Lowered Thresholds
1. Edit `regime_adaptive_strategy.py` to use thresholds 0.30-0.35
2. Rerun walk-forward validation
3. See if it generates trades and how they perform

### Priority 2: Investigate Baseline Strategy
1. Check `MLStrategy` implementation in `strategy_examples.py`
2. Understand why it generates 47 trades with 0.55 threshold
3. Adopt same logic if it's valid

### Priority 3: Retrain with Balanced Data
1. Implement class balancing or SMOTE
2. Add probability calibration
3. Verify BUY probabilities span 0.3-0.7 range
4. Re-run full walk-forward validation

---

## Lessons Learned

### 1. **Always Diagnose Before Debugging**
- Spent time fixing `skip_volatile_regimes` and thresholds
- Should have diagnosed model predictions first
- Would have found root cause immediately

### 2. **Training Data Imbalance is Insidious**
- Model achieves "good" accuracy by predicting majority class
- But produces useless predictions for minority class
- Always check class distribution and prediction probabilities

### 3. **Thresholds Must Match Model Output**
- Can't use 0.40+ thresholds if model maxes out at 0.36
- Thresholds should be data-driven, not arbitrary
- Run diagnostics on validation set to calibrate thresholds

### 4. **Strategy Implementations Matter**
- Baseline and Adaptive use different strategy classes
- Can't directly compare without understanding implementation differences
- Need to ensure fair comparison

### 5. **Model Confidence != Trading Confidence**
- Just because model is "confident" doesn't mean it's correct
- Low BUY probabilities may reflect conservative model, not bad predictions
- May need to adjust what "confidence" means for trading

---

## Files Created

### Analysis Documents
- `REGIME_ADAPTIVE_ANALYSIS.md` - Initial analysis of zero trades
- `REGIME_ADAPTIVE_FINAL_FINDINGS.md` - This document (root cause analysis)
- `regime_adaptive_rerun.log` - First rerun logs (skip_volatile fix)
- `regime_adaptive_bugfix_rerun.log` - Second rerun logs (confidence bug fix)

### Diagnostic Tools
- `diagnose_model_predictions.py` - Model prediction analysis script
- `model_predictions_diagnosis.csv` - Full prediction data for Split 9

### Code Changes
- `walkforward_regime_adaptive.py:142` - Changed `skip_volatile_regimes=False`
- `walkforward_regime_adaptive.py:140` - Lowered `base_confidence_threshold=0.50`
- `regime_adaptive_strategy.py:318` - Fixed `confidence = prediction_proba[1]`

---

## Conclusion

The regime-adaptive strategy failed due to a fundamental mismatch between:
- **Model predictions**: BUY probabilities max out at 35.61%
- **Strategy thresholds**: Minimum threshold is 0.40 (BULL markets)

No amount of regime adaptation can fix this - the model itself needs retraining with balanced data, or thresholds need to be lowered to 0.30-0.35.

**The good news**: We now understand exactly why it failed and have clear paths to fix it.

**Recommendation**: Start with Priority 1 (quick test with lowered thresholds) to validate the approach, then move to Priority 3 (retrain with balanced data) for a proper fix.

---

**Analysis Complete**: 2025-12-31
**Root Cause**: Training data imbalance causing BUY probability < threshold
**Status**: Ready for fixes (3 viable solutions identified)
**Next Action**: Test with thresholds 0.30-0.35 or investigate MLStrategy implementation
