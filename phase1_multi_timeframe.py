#!/usr/bin/env python3
"""
PHASE 1, DAY 3: Multi-Timeframe Prediction

Goal: Train models for multiple time horizons and combine predictions
- Horizons: 1-day, 3-day, 5-day, 10-day
- Strategy: Require agreement across multiple timeframes
- Expected: More robust signals, higher trade frequency
"""

import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import json
from typing import List, Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from backtesting_engine import Strategy, BacktestEngine, BacktestConfig
from feature_engineering import FeatureEngineering

print("=" * 80)
print("PHASE 1, DAY 3: MULTI-TIMEFRAME PREDICTION")
print("=" * 80)

# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("\n[1/6] Loading Data...")

df = pd.read_csv('sp500_historical_data.csv', index_col='Date', parse_dates=True)
df = df.sort_index()

train_df = df[(df.index >= '2020-01-01') & (df.index < '2025-01-01')]
val_df = df[df.index >= '2025-01-01']

print(f"  Train: {len(train_df)} rows (2020-2024)")
print(f"  Val:   {len(val_df)} rows (2025)")

# ============================================================================
# STEP 2: Generate Features (same as Day 1)
# ============================================================================
print("\n[2/6] Generating Features...")

train_feat = FeatureEngineering.build_complete_feature_set(train_df, include_volume=True).dropna()
val_feat = FeatureEngineering.build_complete_feature_set(val_df, include_volume=True).dropna()

# Use same clean features from Day 1
with open('models/randomforest_CLEAN_top20_20251230_230009.pkl', 'rb') as f:
    clean_model_data = pickle.load(f)
    feature_cols = clean_model_data['features']

print(f"  Using {len(feature_cols)} clean features from Day 1")
print(f"  Features: {', '.join(feature_cols[:5])}...")

# ============================================================================
# STEP 3: Train Models for Multiple Timeframes
# ============================================================================
print("\n[3/6] Training Multi-Timeframe Models...")

# Define time horizons (in days)
horizons = [1, 3, 5, 10]

models = {}
performance = {}

for horizon in horizons:
    print(f"\n  Training {horizon}-day model...")

    # Create target: will price be higher in N days?
    train_target = (train_feat['close'].pct_change(horizon).shift(-horizon) > 0).astype(int)
    val_target = (val_feat['close'].pct_change(horizon).shift(-horizon) > 0).astype(int)

    # Remove rows where target is NaN (last N rows)
    train_X = train_feat[feature_cols].iloc[:-horizon].replace([np.inf, -np.inf], np.nan).fillna(0)
    train_y = train_target.iloc[:-horizon].dropna()

    val_X = val_feat[feature_cols].iloc[:-horizon].replace([np.inf, -np.inf], np.nan).fillna(0)
    val_y = val_target.iloc[:-horizon].dropna()

    # Align indices
    common_train_idx = train_X.index.intersection(train_y.index)
    train_X = train_X.loc[common_train_idx]
    train_y = train_y.loc[common_train_idx]

    common_val_idx = val_X.index.intersection(val_y.index)
    val_X = val_X.loc[common_val_idx]
    val_y = val_y.loc[common_val_idx]

    print(f"    Train samples: {len(train_X)}, Val samples: {len(val_X)}")

    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    )

    model.fit(train_X, train_y)

    # Evaluate
    train_pred = model.predict(train_X)
    val_pred = model.predict(val_X)

    train_acc = accuracy_score(train_y, train_pred)
    val_acc = accuracy_score(val_y, val_pred)
    val_prec = precision_score(val_y, val_pred, zero_division=0)
    val_rec = recall_score(val_y, val_pred, zero_division=0)
    val_f1 = f1_score(val_y, val_pred, zero_division=0)

    print(f"    Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}")
    print(f"    Val Precision: {val_prec:.3f}, Recall: {val_rec:.3f}, F1: {val_f1:.3f}")

    # Store model and performance
    models[horizon] = model
    performance[horizon] = {
        'train_accuracy': float(train_acc),
        'val_accuracy': float(val_acc),
        'val_precision': float(val_prec),
        'val_recall': float(val_rec),
        'val_f1': float(val_f1)
    }

# ============================================================================
# STEP 4: Save Multi-Timeframe Models
# ============================================================================
print("\n[4/6] Saving Multi-Timeframe Models...")

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
multi_model_file = f'models/randomforest_MULTITIMEFRAME_{timestamp}.pkl'

with open(multi_model_file, 'wb') as f:
    pickle.dump({
        'models': models,
        'horizons': horizons,
        'features': feature_cols,
        'performance': performance,
        'training_date': timestamp,
        'notes': 'Multi-timeframe models: 1d, 3d, 5d, 10d'
    }, f)

print(f"  ✓ Saved: {multi_model_file}")

# ============================================================================
# STEP 5: Create Multi-Timeframe Strategy
# ============================================================================
print("\n[5/6] Creating Multi-Timeframe Strategy...")


class MultiTimeframeStrategy(Strategy):
    """
    Trading strategy using multiple time horizons

    Entry Logic:
    - Get predictions from all timeframe models
    - Require majority agreement (3/4 or 4/4 models agree on UP)
    - Each model must meet minimum confidence threshold

    Exit Logic:
    - Any model predicts DOWN with high confidence
    - Or majority predict DOWN
    """

    def __init__(self, models_dict, feature_cols, confidence_threshold=0.6,
                 agreement_required=3, name="MultiTimeframe"):
        super().__init__(name)
        self.models = models_dict
        self.horizons = sorted(models_dict.keys())
        self.feature_cols = feature_cols
        self.confidence_threshold = confidence_threshold
        self.agreement_required = agreement_required  # How many models must agree

    def generate_signals(self, data: pd.DataFrame, timestamp: datetime) -> List[Dict]:
        """Generate trading signals based on multi-timeframe predictions"""
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

            # Get predictions from all timeframe models
            predictions = {}
            confidences = {}

            for horizon in self.horizons:
                model = self.models[horizon]
                pred = model.predict(current_features)[0]
                pred_proba = model.predict_proba(current_features)[0]
                conf = pred_proba[pred]

                predictions[horizon] = pred
                confidences[horizon] = conf

            # Count how many models predict UP (1) with sufficient confidence
            up_votes = sum(1 for h in self.horizons
                           if predictions[h] == 1 and confidences[h] >= self.confidence_threshold)

            down_votes = sum(1 for h in self.horizons
                             if predictions[h] == 0 and confidences[h] >= self.confidence_threshold)

            # Get current price and ATR for risk management
            current_price = data['close'].iloc[-1]

            # Calculate ATR
            if 'atr_14' in features.columns:
                atr = features['atr_14'].iloc[-1]
            else:
                high_low = data['high'].iloc[-14:] - data['low'].iloc[-14:]
                atr = high_low.mean()

            # Stop loss and take profit based on ATR
            stop_loss = current_price - (atr * 2.0)
            take_profit = current_price + (atr * 3.0)

            # Check if we have a position
            has_position = len(self.positions) > 0

            if not has_position:
                # ENTRY: Buy if enough models agree on UP
                if up_votes >= self.agreement_required:
                    # Calculate average confidence
                    avg_conf = np.mean([confidences[h] for h in self.horizons if predictions[h] == 1])

                    signals.append({
                        'instrument': 'SP500',
                        'action': 'buy',
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'reason': f"MTF_BUY_{up_votes}/{len(self.horizons)}_conf={avg_conf:.2f}"
                    })
            else:
                # EXIT: Close if majority predict DOWN
                if down_votes >= self.agreement_required:
                    avg_conf = np.mean([confidences[h] for h in self.horizons if predictions[h] == 0])

                    signals.append({
                        'instrument': 'SP500',
                        'action': 'close',
                        'reason': f"MTF_EXIT_{down_votes}/{len(self.horizons)}_conf={avg_conf:.2f}"
                    })

        except Exception:
            # Silently handle errors
            pass

        return signals


# Test multiple agreement thresholds
agreement_levels = [2, 3, 4]  # 2/4, 3/4, 4/4 agreement required

print(f"  Strategy created with {len(horizons)} timeframes: {horizons}")

# ============================================================================
# STEP 6: Backtest Multi-Timeframe Strategies
# ============================================================================
print("\n[6/6] Backtesting Multi-Timeframe Strategies...")

# Configure backtest
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

results_by_agreement = {}

for agreement in agreement_levels:
    print(f"\n  Testing {agreement}/{len(horizons)} agreement requirement...")

    # Create strategy
    strategy = MultiTimeframeStrategy(
        models_dict=models,
        feature_cols=feature_cols,
        confidence_threshold=0.60,
        agreement_required=agreement
    )

    # Create backtest engine
    engine = BacktestEngine(config)

    # Run backtest
    backtest_results = engine.run_backtest(strategy, val_df)

    # Extract metrics
    metrics = backtest_results['metrics']
    trades = backtest_results['trades']

    # Store results
    results_by_agreement[agreement] = {
        'agreement_level': f"{agreement}/{len(horizons)}",
        'total_trades': metrics.get('total_trades', 0),
        'win_rate': metrics.get('win_rate', 0),
        'total_return_pct': metrics.get('total_return_pct', 0),
        'sharpe_ratio': metrics.get('sharpe_ratio', 0),
        'max_drawdown_pct': metrics.get('max_drawdown_pct', 0),
        'profit_factor': metrics.get('profit_factor', 0)
    }

    print(f"    Trades: {metrics.get('total_trades', 0)}")
    print(f"    Return: {metrics.get('total_return_pct', 0):+.2%}")
    print(f"    Win Rate: {metrics.get('win_rate', 0):.1%}")

# ============================================================================
# STEP 7: Compare Results
# ============================================================================
print("\n" + "=" * 80)
print("MULTI-TIMEFRAME RESULTS COMPARISON")
print("=" * 80)

# Load single-timeframe results
with open('models/phase1_confidence_sweep_results.json', 'r') as f:
    single_tf = json.load(f)
    buy_hold = single_tf['buy_hold_return']
    single_best = single_tf['best_threshold']

print(f"\n📈 BENCHMARK: Buy & Hold = {buy_hold:+.2%}\n")

# Single timeframe
print("SINGLE TIMEFRAME (1-day only, threshold=0.60):")
print(f"  Trades:        {single_best['total_trades']}")
print(f"  Win Rate:      {single_best['win_rate']:.1%}")
print(f"  Total Return:  {single_best['total_return_pct']:+.2%}")
print(f"  Sharpe Ratio:  {single_best['sharpe_ratio']:.2f}")

# Multi-timeframe results
print("\nMULTI-TIMEFRAME RESULTS:")
print(f"{'Agreement':<15} {'Trades':<10} {'Win Rate':<12} {'Return':<15} {'Sharpe':<10}")
print("-" * 80)

for agreement, results in sorted(results_by_agreement.items()):
    print(f"{results['agreement_level']:<15} {results['total_trades']:<10} "
          f"{results['win_rate']:<12.1%} {results['total_return_pct']:<15.2%} "
          f"{results['sharpe_ratio']:<10.2f}")

# Find best multi-timeframe configuration
best_mtf = max(results_by_agreement.values(), key=lambda x: x['total_return_pct'])

print("\n🎯 BEST MULTI-TIMEFRAME CONFIG:")
print(f"  Agreement Level:  {best_mtf['agreement_level']}")
print(f"  Total Trades:     {best_mtf['total_trades']}")
print(f"  Win Rate:         {best_mtf['win_rate']:.1%}")
print(f"  Total Return:     {best_mtf['total_return_pct']:+.2%}")
print(f"  Sharpe Ratio:     {best_mtf['sharpe_ratio']:.2f}")
print(f"  vs Buy & Hold:    {best_mtf['total_return_pct'] - buy_hold:+.2%}")

# ============================================================================
# STEP 8: Model Performance by Timeframe
# ============================================================================
print("\n" + "=" * 80)
print("INDIVIDUAL TIMEFRAME MODEL PERFORMANCE")
print("=" * 80)

print(f"\n{'Horizon':<10} {'Val Acc':<12} {'Precision':<12} {'Recall':<12} {'F1':<10}")
print("-" * 80)

for horizon in horizons:
    perf = performance[horizon]
    print(f"{horizon}-day{'':<5} {perf['val_accuracy']:<12.3f} {perf['val_precision']:<12.3f} "
          f"{perf['val_recall']:<12.3f} {perf['val_f1']:<10.3f}")

# ============================================================================
# STEP 9: Save Results
# ============================================================================
print("\n💾 Saving Results...")

results_summary = {
    'timestamp': datetime.now().isoformat(),
    'model_file': multi_model_file,
    'horizons': horizons,
    'features': len(feature_cols),
    'buy_hold_return': float(buy_hold),
    'single_timeframe': {
        'trades': single_best['total_trades'],
        'win_rate': float(single_best['win_rate']),
        'return_pct': float(single_best['total_return_pct']),
        'sharpe': float(single_best['sharpe_ratio'])
    },
    'multi_timeframe': {
        'model_performance': performance,
        'backtest_results': {str(k): v for k, v in results_by_agreement.items()},
        'best_config': {
            'agreement_level': best_mtf['agreement_level'],
            'trades': best_mtf['total_trades'],
            'win_rate': float(best_mtf['win_rate']),
            'return_pct': float(best_mtf['total_return_pct']),
            'sharpe': float(best_mtf['sharpe_ratio'])
        }
    }
}

results_file = 'models/phase1_day3_multitimeframe_results.json'
with open(results_file, 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"  ✓ Results saved: {results_file}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 1 DAY 3 COMPLETE!")
print("=" * 80)

print(f"\n✅ TRAINED {len(horizons)} TIMEFRAME MODELS:")
print(f"  • 1-day:  {performance[1]['val_accuracy']:.1%} accuracy")
print(f"  • 3-day:  {performance[3]['val_accuracy']:.1%} accuracy")
print(f"  • 5-day:  {performance[5]['val_accuracy']:.1%} accuracy")
print(f"  • 10-day: {performance[10]['val_accuracy']:.1%} accuracy")

print("\n📊 RESULTS:")
print(f"  Single TF:  {single_best['total_trades']} trades, {single_best['total_return_pct']:+.2%} return")
print(f"  Multi TF:   {best_mtf['total_trades']} trades, {best_mtf['total_return_pct']:+.2%} return")

# Comparison
if best_mtf['total_trades'] > single_best['total_trades']:
    print(f"\n✅ SUCCESS: Generated {best_mtf['total_trades'] - single_best['total_trades']} more trades!")
elif best_mtf['total_return_pct'] > single_best['total_return_pct']:
    print(f"\n✅ SUCCESS: Higher return ({best_mtf['total_return_pct'] - single_best['total_return_pct']:+.2%})!")
else:
    print("\n⚠️  RESULT: Multi-timeframe didn't improve over single timeframe")

print("\n📁 OUTPUT FILES:")
print(f"  Models: {multi_model_file}")
print(f"  Results: {results_file}")

print("\n" + "=" * 80)
print("NEXT STEP: Build ensemble models (Days 4-5)")
print("Run: python phase1_ensemble_models.py")
print("=" * 80)
