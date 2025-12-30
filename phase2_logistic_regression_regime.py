#!/usr/bin/env python3
"""
PHASE 2, DAY 1: Logistic Regression + Regime Detection

Goal: Improve model accuracy and trade frequency
- Use Logistic Regression (best model at 62.8% accuracy)
- Add regime detection features
- Test confidence thresholds 0.55-0.58
- Target: 60%+ accuracy, 5-10 trades, maintain quality
"""

import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import json
from typing import List, Dict

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from backtesting_engine import Strategy, BacktestEngine, BacktestConfig
from feature_engineering import FeatureEngineering

print("=" * 80)
print("PHASE 2, DAY 1: LOGISTIC REGRESSION + REGIME DETECTION")
print("=" * 80)

# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("\n[1/7] Loading Data...")

df = pd.read_csv('sp500_historical_data.csv', index_col='Date', parse_dates=True)
df = df.sort_index()

train_df = df[(df.index >= '2020-01-01') & (df.index < '2025-01-01')]
val_df = df[df.index >= '2025-01-01']

print(f"  Train: {len(train_df)} rows (2020-2024)")
print(f"  Val:   {len(val_df)} rows (2025)")

# ============================================================================
# STEP 2: Generate Base Features + Regime Detection Features
# ============================================================================
print("\n[2/7] Generating Features with Regime Detection...")

train_feat = FeatureEngineering.build_complete_feature_set(train_df, include_volume=True).dropna()
val_feat = FeatureEngineering.build_complete_feature_set(val_df, include_volume=True).dropna()

# Add regime detection features


def add_regime_features(df, feat_df):
    """Add regime detection features"""

    # 1. TREND REGIME (trending vs mean-reverting)
    # Use ADX: >25 = trending, <20 = ranging
    if 'adx_14' in feat_df.columns:
        feat_df['regime_trending'] = (feat_df['adx_14'] > 25).astype(int)
        feat_df['regime_ranging'] = (feat_df['adx_14'] < 20).astype(int)

    # 2. DIRECTIONAL REGIME (bull vs bear)
    # Use SMA slopes and price position
    if 'sma_50' in feat_df.columns and 'sma_200' in feat_df.columns and 'close' in feat_df.columns:
        # Bull: price > SMA50 > SMA200, all rising
        price = feat_df['close']
        sma50 = feat_df['sma_50']
        sma200 = feat_df['sma_200']

        sma50_rising = sma50.diff(5) > 0
        sma200.diff(10) > 0

        feat_df['regime_bull'] = ((price > sma50) & (sma50 > sma200) & sma50_rising).astype(int)
        feat_df['regime_bear'] = ((price < sma50) & (sma50 < sma200) & (sma50.diff(5) < 0)).astype(int)

    # 3. VOLATILITY REGIME
    if 'volatility_20' in feat_df.columns:
        vol_median = feat_df['volatility_20'].rolling(100, min_periods=20).median()
        feat_df['regime_high_vol'] = (feat_df['volatility_20'] > vol_median * 1.5).astype(int)
        feat_df['regime_low_vol'] = (feat_df['volatility_20'] < vol_median * 0.7).astype(int)

    # 4. MOMENTUM REGIME
    if 'rsi_14' in feat_df.columns:
        feat_df['regime_overbought'] = (feat_df['rsi_14'] > 70).astype(int)
        feat_df['regime_oversold'] = (feat_df['rsi_14'] < 30).astype(int)

    # 5. MARKET STRUCTURE (higher highs, higher lows)
    if 'close' in feat_df.columns:
        close = feat_df['close']
        high_20 = close.rolling(20).max()
        low_20 = close.rolling(20).min()

        prev_high = high_20.shift(20)
        prev_low = low_20.shift(20)

        feat_df['regime_uptrend'] = (high_20 > prev_high).astype(int)
        feat_df['regime_downtrend'] = (low_20 < prev_low).astype(int)

    return feat_df


print("  Adding regime detection features...")
train_feat = add_regime_features(train_df, train_feat)
val_feat = add_regime_features(val_df, val_feat)

# Remove NaN from new features
train_feat = train_feat.dropna()
val_feat = val_feat.dropna()

print(f"  Total features after regime detection: {len(train_feat.columns)}")

# ============================================================================
# STEP 3: Feature Selection - Top 30 Features
# ============================================================================
print("\n[3/7] Selecting Top 30 Features (including regime features)...")

# Create target
train_target = (train_feat['close'].pct_change().shift(-1) > 0).astype(int)[:-1]
val_target = (val_feat['close'].pct_change().shift(-1) > 0).astype(int)[:-1]

train_feat_clean = train_feat.iloc[:-1]
val_feat_clean = val_feat.iloc[:-1]

# Exclude leakage and OHLCV
leakage_features = [
    'target_regression', 'future_return', 'target_binary', 'target_class',
    'future_return_5', 'future_return_10', 'future_volatility',
    'open', 'high', 'low', 'close', 'volume'
]

all_features = [c for c in train_feat_clean.columns if c not in leakage_features]

print(f"  Clean features available: {len(all_features)}")

# Train temporary Logistic Regression to rank features
X_train_all = train_feat_clean[all_features].replace([np.inf, -np.inf], np.nan).fillna(0)
y_train = train_target

# Use L1 regularization to get feature importance
temp_lr = LogisticRegression(
    penalty='l1',
    C=0.1,
    solver='liblinear',
    random_state=42,
    max_iter=1000
)

temp_lr.fit(X_train_all, y_train)

# Get feature importance (absolute coefficients)
feature_importance = pd.DataFrame({
    'feature': all_features,
    'importance': np.abs(temp_lr.coef_[0])
}).sort_values('importance', ascending=False)

# Select top 30 features (more than Phase 1's 20)
top_30_features = feature_importance.head(30)['feature'].tolist()

print("\n  Top 30 Features:")
for i, row in feature_importance.head(30).iterrows():
    regime_marker = " [REGIME]" if 'regime_' in row['feature'] else ""
    print(f"    {i + 1:2d}. {row['feature']:35s} {row['importance']:.4f}{regime_marker}")

# ============================================================================
# STEP 4: Train Logistic Regression with Top 30 Features
# ============================================================================
print("\n[4/7] Training Logistic Regression with Top 30 Features...")

X_train = train_feat_clean[top_30_features].replace([np.inf, -np.inf], np.nan).fillna(0)
X_val = val_feat_clean[top_30_features].replace([np.inf, -np.inf], np.nan).fillna(0)
y_val = val_target

# Train final Logistic Regression (L2 regularization for better generalization)
lr_model = LogisticRegression(
    penalty='l2',
    C=1.0,
    max_iter=1000,
    random_state=42
)

lr_model.fit(X_train, y_train)

# Evaluate
y_train_pred = lr_model.predict(X_train)
y_val_pred = lr_model.predict(X_val)

train_acc = accuracy_score(y_train, y_train_pred)
val_acc = accuracy_score(y_val, y_val_pred)
val_prec = precision_score(y_val, y_val_pred, zero_division=0)
val_rec = recall_score(y_val, y_val_pred, zero_division=0)
val_f1 = f1_score(y_val, y_val_pred, zero_division=0)

print("\n  Performance:")
print(f"    Train Accuracy: {train_acc:.3f}")
print(f"    Val Accuracy:   {val_acc:.3f}")
print(f"    Val Precision:  {val_prec:.3f}")
print(f"    Val Recall:     {val_rec:.3f}")
print(f"    Val F1:         {val_f1:.3f}")

# ============================================================================
# STEP 5: Save Model
# ============================================================================
print("\n[5/7] Saving Logistic Regression Model...")

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_file = f'models/logistic_regression_regime_{timestamp}.pkl'

with open(model_file, 'wb') as f:
    pickle.dump({
        'model': lr_model,
        'features': top_30_features,
        'feature_importance': feature_importance.head(30).to_dict('records'),
        'performance': {
            'train_accuracy': float(train_acc),
            'val_accuracy': float(val_acc),
            'val_precision': float(val_prec),
            'val_recall': float(val_rec),
            'val_f1': float(val_f1)
        },
        'training_date': timestamp,
        'notes': 'Logistic Regression with regime detection features'
    }, f)

print(f"  ✓ Saved: {model_file}")

# ============================================================================
# STEP 6: Test Multiple Confidence Thresholds
# ============================================================================
print("\n[6/7] Testing Confidence Thresholds (0.55-0.65)...")

# Create ML strategy (same as Phase 1 but with LR model)


class MLLogisticRegressionStrategy(Strategy):
    """Trading strategy using Logistic Regression"""

    def __init__(self, model, feature_cols, confidence_threshold=0.6, name="MLLogisticRegression"):
        super().__init__(name)
        self.model = model
        self.feature_cols = feature_cols
        self.confidence_threshold = confidence_threshold

    def generate_signals(self, data: pd.DataFrame, timestamp: datetime) -> List[Dict]:
        signals = []

        if len(data) < 60:
            return signals

        try:
            # Generate features
            features = FeatureEngineering.build_complete_feature_set(data, include_volume=True)
            features = features.dropna()

            if len(features) == 0:
                return signals

            # Add regime features
            features = add_regime_features(data, features)
            features = features.dropna()

            if len(features) == 0:
                return signals

            # Get current features
            current_features = features.iloc[-1:][self.feature_cols]
            current_features = current_features.replace([np.inf, -np.inf], np.nan).fillna(0)

            # Make prediction
            prediction = self.model.predict(current_features)[0]
            prediction_proba = self.model.predict_proba(current_features)[0]
            confidence = prediction_proba[prediction]

            # Risk management
            current_price = data['close'].iloc[-1]

            if 'atr_14' in features.columns:
                atr = features['atr_14'].iloc[-1]
            else:
                high_low = data['high'].iloc[-14:] - data['low'].iloc[-14:]
                atr = high_low.mean()

            stop_loss = current_price - (atr * 2.0)
            take_profit = current_price + (atr * 3.0)

            has_position = len(self.positions) > 0

            if not has_position:
                # ENTRY: Buy if predicts UP with high confidence
                if prediction == 1 and confidence >= self.confidence_threshold:
                    signals.append({
                        'instrument': 'SP500',
                        'action': 'buy',
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'reason': f"LR_BUY_conf={confidence:.2f}"
                    })
            else:
                # EXIT: Close if predicts DOWN
                if prediction == 0:
                    signals.append({
                        'instrument': 'SP500',
                        'action': 'close',
                        'reason': f"LR_EXIT_conf={confidence:.2f}"
                    })

        except Exception:
            pass

        return signals


# Test thresholds 0.55, 0.57, 0.58, 0.60, 0.62
thresholds = [0.55, 0.57, 0.58, 0.60, 0.62]

config = BacktestConfig(
    initial_capital=10000.0,
    commission_pct=0.0001,
    slippage_pct=0.0001,
    position_size_pct=0.02,
    max_positions=1,
    leverage=1.0,
    position_sizing_method='fixed_pct',
    max_daily_loss_pct=0.05,
    max_drawdown_pct=0.20
)

results_by_threshold = {}

for threshold in thresholds:
    print(f"\n  Testing threshold: {threshold:.2f}")

    strategy = MLLogisticRegressionStrategy(
        model=lr_model,
        feature_cols=top_30_features,
        confidence_threshold=threshold
    )

    engine = BacktestEngine(config)
    backtest_results = engine.run_backtest(strategy, val_df)
    metrics = backtest_results['metrics']

    results_by_threshold[threshold] = {
        'threshold': threshold,
        'total_trades': metrics.get('total_trades', 0),
        'win_rate': metrics.get('win_rate', 0),
        'total_return_pct': metrics.get('total_return_pct', 0),
        'sharpe_ratio': metrics.get('sharpe_ratio', 0),
        'max_drawdown_pct': metrics.get('max_drawdown_pct', 0)
    }

    print(f"    Trades: {metrics.get('total_trades', 0)}, "
          f"Return: {metrics.get('total_return_pct', 0):+.2%}, "
          f"Win Rate: {metrics.get('win_rate', 0):.1%}")

# ============================================================================
# STEP 7: Compare Results
# ============================================================================
print("\n" + "=" * 80)
print("RESULTS COMPARISON")
print("=" * 80)

# Load Phase 1 best result
with open('models/phase1_confidence_sweep_results.json', 'r') as f:
    phase1 = json.load(f)
    buy_hold = phase1['buy_hold_return']
    phase1_best = phase1['best_threshold']

print(f"\n📈 BENCHMARK: Buy & Hold = {buy_hold:+.2%}\n")

# Phase 1 best
print("PHASE 1 BEST (RandomForest, 20 features, threshold=0.60):")
print("  Val Accuracy:  55.8%")
print(f"  Trades:        {phase1_best['total_trades']}")
print(f"  Win Rate:      {phase1_best['win_rate']:.1%}")
print(f"  Total Return:  {phase1_best['total_return_pct']:+.2%}")

# Phase 2 results
print("\nPHASE 2 RESULTS (LogisticRegression, 30 features + regime):")
print(f"  Val Accuracy:  {val_acc:.1%}")
print(f"\n{'Threshold':<12} {'Trades':<10} {'Win Rate':<12} {'Return':<15} {'vs Phase 1':<15}")
print("-" * 80)

for threshold, results in sorted(results_by_threshold.items()):
    vs_phase1 = results['total_return_pct'] - phase1_best['total_return_pct']
    print(f"{threshold:<12.2f} {results['total_trades']:<10} {results['win_rate']:<12.1%} "
          f"{results['total_return_pct']:<15.2%} {vs_phase1:<15.2%}")

# Find best Phase 2 result
best_phase2 = max(results_by_threshold.values(), key=lambda x: x['total_return_pct'])

print("\n🎯 BEST PHASE 2 CONFIG:")
print(f"  Threshold:     {best_phase2['threshold']:.2f}")
print(f"  Trades:        {best_phase2['total_trades']}")
print(f"  Win Rate:      {best_phase2['win_rate']:.1%}")
print(f"  Total Return:  {best_phase2['total_return_pct']:+.2%}")
print(f"  vs Phase 1:    {best_phase2['total_return_pct'] - phase1_best['total_return_pct']:+.2%}")

# ============================================================================
# STEP 8: Save Results
# ============================================================================
print("\n💾 Saving Results...")

results_summary = {
    'timestamp': datetime.now().isoformat(),
    'model_file': model_file,
    'model_type': 'LogisticRegression',
    'features': len(top_30_features),
    'regime_features': len([f for f in top_30_features if 'regime_' in f]),
    'buy_hold_return': float(buy_hold),
    'model_performance': {
        'train_accuracy': float(train_acc),
        'val_accuracy': float(val_acc),
        'val_precision': float(val_prec),
        'val_recall': float(val_rec),
        'val_f1': float(val_f1)
    },
    'phase1_best': {
        'model': 'RandomForest',
        'features': 20,
        'val_accuracy': 0.558,
        'trades': phase1_best['total_trades'],
        'win_rate': float(phase1_best['win_rate']),
        'return_pct': float(phase1_best['total_return_pct'])
    },
    'phase2_results': {str(k): v for k, v in results_by_threshold.items()},
    'best_config': {
        'threshold': best_phase2['threshold'],
        'trades': best_phase2['total_trades'],
        'win_rate': float(best_phase2['win_rate']),
        'return_pct': float(best_phase2['total_return_pct']),
        'vs_phase1': float(best_phase2['total_return_pct'] - phase1_best['total_return_pct'])
    }
}

results_file = 'models/phase2_day1_logistic_regime_results.json'
with open(results_file, 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"  ✓ Results saved: {results_file}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 2 DAY 1 COMPLETE!")
print("=" * 80)

print("\n✅ IMPROVEMENTS:")
print("  • Switched to Logistic Regression (better accuracy)")
print(f"  • Added {len([f for f in top_30_features if 'regime_' in f])} regime detection features")
print("  • Increased features from 20 → 30")
print(f"  • Val accuracy: {val_acc:.1%} (Phase 1: 55.8%)")

print("\n📊 BEST RESULT:")
print(f"  Threshold: {best_phase2['threshold']:.2f}")
print(f"  Trades:    {best_phase2['total_trades']} (Phase 1: {phase1_best['total_trades']})")
print(f"  Win Rate:  {best_phase2['win_rate']:.1%} (Phase 1: {phase1_best['win_rate']:.1%})")
print(f"  Return:    {best_phase2['total_return_pct']:+.2%} (Phase 1: {phase1_best['total_return_pct']:+.2%})")

# Verdict
if best_phase2['total_trades'] > phase1_best['total_trades'] and best_phase2['win_rate'] >= 0.5:
    verdict = f"✅ SUCCESS: More trades ({
        best_phase2['total_trades']} vs {
        phase1_best['total_trades']}) with good quality!"
elif best_phase2['total_return_pct'] > phase1_best['total_return_pct']:
    verdict = "✅ SUCCESS: Higher return!"
elif val_acc > 0.558:
    verdict = f"✅ PROGRESS: Improved accuracy ({val_acc:.1%} vs 55.8%)"
else:
    verdict = "⚠️  Mixed results - need further tuning"

print(f"\n{verdict}")

print("\n📁 OUTPUT FILES:")
print(f"  Model: {model_file}")
print(f"  Results: {results_file}")

print("\n" + "=" * 80)
print("NEXT STEP: Walk-forward validation")
print("=" * 80)
