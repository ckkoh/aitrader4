#!/usr/bin/env python3
"""
PHASE 1, DAY 2: Test Multiple Confidence Thresholds

Goal: Find optimal confidence threshold for clean ML model
- Original model used 0.6 threshold → only 4 trades
- Test thresholds: 0.50, 0.55, 0.60, 0.65, 0.70
- Measure: trades, return, win rate, Sharpe ratio
- Find sweet spot: enough trades + positive return
"""

import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import json
from typing import List, Dict

from backtesting_engine import Strategy, BacktestEngine, BacktestConfig
from feature_engineering import FeatureEngineering

print("=" * 80)
print("PHASE 1, DAY 2: CONFIDENCE THRESHOLD SWEEP")
print("=" * 80)

# ============================================================================
# STEP 1: Load Clean Model
# ============================================================================
print("\n[1/5] Loading Clean Model...")

model_file = 'models/randomforest_CLEAN_top20_20251230_230009.pkl'
with open(model_file, 'rb') as f:
    model_data = pickle.load(f)
    clean_model = model_data['model']
    feature_cols = model_data['features']

print(f"✓ Model loaded: {model_file}")
print(f"  Features: {len(feature_cols)}")
print(f"  Validation Accuracy: {model_data['performance']['val_accuracy']:.1%}")

# ============================================================================
# STEP 2: Load 2025 Validation Data
# ============================================================================
print("\n[2/5] Loading 2025 Validation Data...")

df = pd.read_csv('sp500_historical_data.csv', index_col='Date', parse_dates=True)
df = df.sort_index()
val_df = df[df.index >= '2025-01-01'].copy()

print(f"✓ Loaded {len(val_df)} rows")
print(f"  Period: {val_df.index[0].date()} to {val_df.index[-1].date()}")

# Generate features
print("  Generating features...")
val_features = FeatureEngineering.build_complete_feature_set(val_df, include_volume=True)
val_features = val_features.dropna()
X_val = val_features[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

print(f"✓ Features generated: {X_val.shape}")

# ============================================================================
# STEP 3: Create ML Strategy Class
# ============================================================================
print("\n[3/5] Creating ML Strategy Class...")


class MLRandomForestStrategy(Strategy):
    """ML Trading Strategy with configurable confidence threshold"""

    def __init__(self, model, feature_cols, confidence_threshold=0.6, name="MLRandomForest"):
        super().__init__(name)
        self.model = model
        self.feature_cols = feature_cols
        self.confidence_threshold = confidence_threshold
        self.last_features = None

    def generate_signals(self, data: pd.DataFrame, timestamp: datetime) -> List[Dict]:
        """Generate trading signals based on ML predictions"""
        signals = []

        # Need enough data for features
        if len(data) < 60:
            return signals

        # Generate features for current data
        try:
            features = FeatureEngineering.build_complete_feature_set(data, include_volume=True)
            features = features.dropna()

            if len(features) == 0:
                return signals

            # Get current row features
            current_features = features.iloc[-1:][self.feature_cols]
            current_features = current_features.replace([np.inf, -np.inf], np.nan).fillna(0)

            # Make prediction
            prediction = self.model.predict(current_features)[0]
            prediction_proba = self.model.predict_proba(current_features)[0]

            # Get confidence (probability of predicted class)
            confidence = prediction_proba[prediction]

            # Get current price and ATR for risk management
            current_price = data['close'].iloc[-1]

            # Calculate ATR if available
            if 'atr_14' in features.columns:
                atr = features['atr_14'].iloc[-1]
            else:
                # Fallback: use simple price-based ATR estimate
                high_low = data['high'].iloc[-14:] - data['low'].iloc[-14:]
                atr = high_low.mean()

            # Stop loss and take profit based on ATR
            stop_loss = current_price - (atr * 2.0)
            take_profit = current_price + (atr * 3.0)

            # Check if we have a position
            has_position = len(self.positions) > 0

            if not has_position:
                # ENTRY: Buy if model predicts UP with high confidence
                if prediction == 1 and confidence >= self.confidence_threshold:
                    signals.append({
                        'instrument': 'SP500',
                        'action': 'buy',
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'reason': f"ML_BUY_conf={confidence:.2f}"
                    })
            else:
                # EXIT: Close if model predicts DOWN
                if prediction == 0:
                    signals.append({
                        'instrument': 'SP500',
                        'action': 'close',
                        'reason': f"ML_EXIT_prediction=DOWN_conf={confidence:.2f}"
                    })

        except Exception:
            # Silently handle errors (e.g., feature calculation issues)
            pass

        return signals


print("✓ Strategy class created")

# ============================================================================
# STEP 4: Test Multiple Confidence Thresholds
# ============================================================================
print("\n[4/5] Testing Multiple Confidence Thresholds...")

# Configure backtest
config = BacktestConfig(
    initial_capital=10000.0,
    commission_pct=0.0001,      # 0.01% commission
    slippage_pct=0.0001,         # 0.01% slippage
    position_size_pct=0.02,      # 2% risk per trade
    max_positions=1,
    leverage=1.0,
    position_sizing_method='fixed_pct',
    max_daily_loss_pct=0.05,
    max_drawdown_pct=0.20
)

# Test these thresholds
thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]

results_by_threshold = {}

for threshold in thresholds:
    print(f"\n  Testing threshold: {threshold:.2f}")

    # Create strategy with this threshold
    strategy = MLRandomForestStrategy(
        model=clean_model,
        feature_cols=feature_cols,
        confidence_threshold=threshold
    )

    # Create backtest engine
    engine = BacktestEngine(config)

    # Run backtest
    backtest_results = engine.run_backtest(strategy, val_df)

    # Extract metrics
    metrics = backtest_results['metrics']
    trades = backtest_results['trades']

    # Store results
    results_by_threshold[threshold] = {
        'threshold': threshold,
        'total_trades': metrics.get('total_trades', 0),
        'win_rate': metrics.get('win_rate', 0),
        'total_return_pct': metrics.get('total_return_pct', 0),
        'sharpe_ratio': metrics.get('sharpe_ratio', 0),
        'sortino_ratio': metrics.get('sortino_ratio', 0),
        'max_drawdown_pct': metrics.get('max_drawdown_pct', 0),
        'profit_factor': metrics.get('profit_factor', 0),
        'expectancy': metrics.get('expectancy', 0),
        'avg_win': metrics.get('avg_win', 0),
        'avg_loss': metrics.get('avg_loss', 0),
        'winning_trades': metrics.get('winning_trades', 0),
        'losing_trades': metrics.get('losing_trades', 0),
    }

    print(f"    Trades: {metrics.get('total_trades', 0)}")
    print(f"    Return: {metrics.get('total_return_pct', 0):+.2%}")
    print(f"    Win Rate: {metrics.get('win_rate', 0):.1%}")

# ============================================================================
# STEP 5: Compare Results and Find Best Threshold
# ============================================================================
print("\n[5/5] Comparing Results...")

# Convert to DataFrame for easier comparison
results_df = pd.DataFrame(results_by_threshold.values())

# Calculate buy & hold return
buy_hold_return = (val_df['close'].iloc[-1] - val_df['close'].iloc[0]) / val_df['close'].iloc[0]

print(f"\n{'=' * 80}")
print("CONFIDENCE THRESHOLD COMPARISON")
print(f"{'=' * 80}")

print(f"\nBuy & Hold Return: {buy_hold_return:+.2%}\n")

# Display table
print(f"{'Threshold':<12} {'Trades':<8} {'Win%':<8} {'Return':<10} {'Sharpe':<8} {'Max DD':<10}")
print(f"{'-' * 80}")

for _, row in results_df.iterrows():
    print(f"{row['threshold']:<12.2f} {int(row['total_trades']):<8} "
          f"{row['win_rate']:<8.1%} {row['total_return_pct']:<10.2%} "
          f"{row['sharpe_ratio']:<8.2f} {row['max_drawdown_pct']:<10.2%}")

# Find best threshold
# Criteria: Most trades with positive return and highest Sharpe
profitable_results = results_df[results_df['total_return_pct'] > 0]

if len(profitable_results) > 0:
    # Best = highest Sharpe among profitable
    best_idx = profitable_results['sharpe_ratio'].idxmax()
    best_threshold = profitable_results.loc[best_idx]
else:
    # No profitable thresholds - pick one with most trades
    best_idx = results_df['total_trades'].idxmax()
    best_threshold = results_df.loc[best_idx]

print(f"\n{'=' * 80}")
print("BEST THRESHOLD")
print(f"{'=' * 80}")
print(f"  Confidence Threshold: {best_threshold['threshold']:.2f}")
print(f"  Total Trades:         {int(best_threshold['total_trades'])}")
print(f"  Win Rate:             {best_threshold['win_rate']:.1%}")
print(f"  Total Return:         {best_threshold['total_return_pct']:+.2%}")
print(f"  Sharpe Ratio:         {best_threshold['sharpe_ratio']:.2f}")
print(f"  Max Drawdown:         {best_threshold['max_drawdown_pct']:.2%}")
print(f"  Profit Factor:        {best_threshold['profit_factor']:.2f}")

# ============================================================================
# STEP 6: Save Results
# ============================================================================
print("\n💾 Saving Results...")

# Save comparison
comparison = {
    'timestamp': datetime.now().isoformat(),
    'model': model_file,
    'test_period': f"{val_df.index[0].date()} to {val_df.index[-1].date()}",
    'buy_hold_return': float(buy_hold_return),
    'thresholds_tested': thresholds,
    'results': {str(k): {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                         for kk, vv in v.items()}
                for k, v in results_by_threshold.items()},
    'best_threshold': {
        'threshold': float(best_threshold['threshold']),
        'total_trades': int(best_threshold['total_trades']),
        'win_rate': float(best_threshold['win_rate']),
        'total_return_pct': float(best_threshold['total_return_pct']),
        'sharpe_ratio': float(best_threshold['sharpe_ratio']),
        'max_drawdown_pct': float(best_threshold['max_drawdown_pct'])
    }
}

results_file = 'models/phase1_confidence_sweep_results.json'
with open(results_file, 'w') as f:
    json.dump(comparison, f, indent=2)

print(f"  ✓ Results saved: {results_file}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 1 DAY 2 COMPLETE!")
print("=" * 80)

print(f"\n✅ TESTED {len(thresholds)} CONFIDENCE THRESHOLDS")
print(f"  Range: {min(thresholds):.2f} - {max(thresholds):.2f}")

print(f"\n🎯 BEST THRESHOLD: {best_threshold['threshold']:.2f}")
print(f"  Generated {int(best_threshold['total_trades'])} trades")
print(f"  Return: {best_threshold['total_return_pct']:+.2%} (vs {buy_hold_return:+.2%} buy & hold)")

# Verdict
if best_threshold['total_return_pct'] > buy_hold_return:
    verdict = "✅ OUTPERFORMED buy & hold!"
elif best_threshold['total_return_pct'] > 0:
    verdict = "⚠️  Positive but underperformed buy & hold"
else:
    verdict = "❌ Negative return - needs more work"

print(f"\n{verdict}")

print(f"\n📁 OUTPUT: {results_file}")

print("\n" + "=" * 80)
print("NEXT STEP: Implement multi-timeframe prediction")
print("Run: python phase1_multi_timeframe.py")
print("=" * 80)
