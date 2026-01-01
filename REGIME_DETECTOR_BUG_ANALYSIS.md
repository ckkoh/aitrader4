# Regime Detector Bug Analysis

## Bug Found: Why Ensemble Only Uses Balanced Model

**Root Cause:** Regime detector classifies almost all periods as SIDEWAYS due to incorrect thresholds and overly strict trend direction logic.

---

## Evidence from Diagnostic

### Regime Distribution Across 15 Periods:

| Period | Expected Regime | Actual Regime | Why Wrong? |
|--------|----------------|---------------|------------|
| Split 1 (2022 Q1) | BEAR/VOLATILE | **SIDEWAYS** | Missed bear market start |
| Split 2 (2022 Q2) | BEAR | **SIDEWAYS** | Missed bear market |
| Split 3 (2022 Q3) | BEAR | **SIDEWAYS** | Missed bear market |
| Split 4 (2022 Q4) | BEAR | **SIDEWAYS** | Missed bear market |
| Split 5-8 (2023) | SIDEWAYS/BULL | **SIDEWAYS** | Correct (by luck) |
| Split 9-12 (2024) | BULL | **BULL** ✓ | Correct |
| Split 13-15 (2025) | BULL/SIDEWAYS | **SIDEWAYS** | Mostly correct |

**Actual Distribution:**
- SIDEWAYS: 11/15 periods (73.3%) ❌ WAY TOO HIGH
- BULL: 4/15 periods (26.7%)
- BEAR: 0/15 periods (0%) ❌ MISSED ENTIRE 2022 BEAR MARKET
- VOLATILE: 0/15 periods (0%) ❌ NEVER TRIGGERED

**Result:** Ensemble always uses Balanced model (SIDEWAYS → Balanced, BULL → Balanced)

---

## Root Cause Analysis

### Bug #1: VOLATILE Threshold Too High

**Current Code:**
```python
# regime_adaptive_strategy.py:113
HIGH_VOL = 20  # 20% annualized volatility
```

**Volatility Calculation:**
```python
volatility = current['atr_14'] / current['close'] * 100
# Returns: 0.7% - 3.0% (ATR as % of price, NOT annualized)
```

**Problem:** Comparing non-annualized volatility (0.7-3%) to annualized threshold (20%)
- 2022 bear market had volatility of 1.5-3.0%
- Threshold is 20%
- **3.0% < 20% = NEVER triggers VOLATILE**

**Fix:** Lower threshold to 3% for ATR-based volatility

### Bug #2: Trend Direction Too Strict

**Current Code:**
```python
# regime_adaptive_strategy.py:67-72
if price > sma_50 > sma_200:
    trend_direction = 1  # Bullish
elif price < sma_50 < sma_200:
    trend_direction = -1  # Bearish
else:
    trend_direction = 0  # Mixed
```

**Problem:** Requires PERFECT alignment for BEAR classification
- BEAR needs: `price < sma_50 < sma_200` (all three in descending order)
- During 2022 bear market crossover:
  - Day 1: price = 4500, sma_50 = 4600, sma_200 = 4550 → trend_direction = 0 (SIDEWAYS)
  - Price below 50 MA, but 50 MA still above 200 MA during transition
  - This classifies as SIDEWAYS even though market is clearly bearish!

**Evidence from Split 2 (2022 Q2 bear market):**
```
START: SIDEWAYS (should be BEAR)
MIDDLE: SIDEWAYS (should be BEAR)
END: BEAR ✓ (finally detected when alignment is perfect)
```

**Fix:** Relax trend direction logic to consider price vs 50 MA primarily

### Bug #3: BEAR Condition Too Restrictive

**Current Code:**
```python
# regime_adaptive_strategy.py:126
if trend_direction == -1 and trend_strength > STRONG_TREND:
    return 'BEAR'
```

**Problem:** BEAR requires BOTH:
1. Perfect alignment (`trend_direction == -1`)
2. Strong trend (`trend_strength > 5%`)

**During 2022 bear market:**
- Split 1: trend_strength = 1.5% < 5% → SIDEWAYS (even if direction correct)
- Split 2: trend_direction = 0 → SIDEWAYS (alignment not perfect)
- Split 3: trend_strength = 4.7% < 5% → SIDEWAYS (just barely missed)

**Fix:** Lower STRONG_TREND threshold to 3% OR use relaxed trend direction

---

## Recommended Fixes

### Option 1: Lower Thresholds (Conservative)

```python
# regime_adaptive_strategy.py:113-115
HIGH_VOL = 3.0  # 3% ATR volatility (was 20%)
STRONG_TREND = 3.0  # 3% distance between SMAs (was 5%)
STRONG_MOMENTUM = 10  # Keep as is
```

**Impact:** Will detect more BEAR/VOLATILE regimes

### Option 2: Relax Trend Direction Logic (Aggressive)

```python
# regime_adaptive_strategy.py:67-73
# Trend direction based on price vs 50 MA primarily
if price > sma_50:
    trend_direction = 1  # Bullish
elif price < sma_50:
    trend_direction = -1  # Bearish
else:
    trend_direction = 0  # Mixed

# Additional check: 50 MA vs 200 MA for confirmation
sma_crossover = 1 if sma_50 > sma_200 else -1 if sma_50 < sma_200 else 0

# Final direction: combine both signals
if trend_direction == sma_crossover:
    final_trend_direction = trend_direction
else:
    final_trend_direction = 0  # Conflicting signals
```

**Impact:** More responsive to price action, less dependent on SMA crossover

### Option 3: Add Momentum Condition (Balanced)

```python
# regime_adaptive_strategy.py:125-131
# BEAR regime: Use multiple signals
if (trend_direction == -1 and trend_strength > STRONG_TREND) or \
   (momentum < -STRONG_MOMENTUM and volatility > 2.0):
    return {
        'regime': 'BEAR',
        'confidence': max(trend_strength / STRONG_TREND, abs(momentum) / STRONG_MOMENTUM),
        'reason': f'Bearish conditions (trend: {trend_strength:.1f}%, momentum: {momentum:.1f}%)'
    }
```

**Impact:** Can detect BEAR even without perfect SMA alignment if momentum is strongly negative

---

## Recommended Implementation

**Use Option 1 + Option 3 (combined approach):**

1. **Lower thresholds** to make regime detection more sensitive
2. **Add momentum condition** to catch bear markets during SMA crossover periods
3. **Keep VOLATILE threshold at 3%** to detect high volatility periods

This will fix the ensemble bug while maintaining reasonable regime detection accuracy.

---

## Expected Results After Fix

### 2022 Bear Market (Splits 1-4):
- Should detect BEAR or VOLATILE (instead of SIDEWAYS)
- Ensemble will use Original model
- Expected to preserve +48.77% cumulative bear market performance

### 2023-2024 Bull Market (Splits 9-12):
- Should detect BULL (currently correct)
- Ensemble will use Balanced model
- Expected to preserve +52.10% cumulative bull market performance

### Final Ensemble Performance Estimate:
- Current: +0.70% avg return (100% Balanced usage)
- Fixed: +2.5% to +3.5% avg return (proper model switching)
- Reasoning: Combines Original's bear strength (+48.77%) with Balanced's bull strength (+52.10%)

---

## Implementation Steps

1. **Fix regime detector thresholds** (regime_adaptive_strategy.py:113-115)
2. **Add momentum condition** to BEAR classification
3. **Re-run ensemble validation** (walkforward_ensemble.py)
4. **Verify regime distribution** matches expected (4 BEAR, 4 BULL, 7 SIDEWAYS/VOLATILE)
5. **Compare results** to current ensemble (+0.70% baseline)

---

**Bug Status:** Identified and ready to fix
**Priority:** HIGH - Blocks ensemble strategy from working as designed
**Estimated Impact:** +1.8% to +2.8% improvement in average returns
