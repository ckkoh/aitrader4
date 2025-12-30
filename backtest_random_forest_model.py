#!/usr/bin/env python3
"""
Backtest Random Forest Model Predictions
Tests the trained ML model on 2025 validation data
"""

import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import json
from typing import List, Dict

# Import trading system components
from backtesting_engine import Strategy, BacktestEngine, BacktestConfig
from feature_engineering import FeatureEngineering

print("=" * 80)
print("BACKTESTING RANDOM FOREST MODEL")
print("=" * 80)

# ============================================================================
# STEP 1: Load Trained Model
# ============================================================================
print("\n[1/5] Loading Trained Random Forest Model...")

model_file = 'models/randomforest_2020-2024train_2025val_20251230_210110.pkl'
with open(model_file, 'rb') as f:
    model_data = pickle.load(f)
    rf_model = model_data['model']
    feature_cols = model_data['features']

print(f"✓ Model loaded: {model_file}")
print(f"  Features: {len(feature_cols)}")
print(f"  Model type: {type(rf_model).__name__}")

# ============================================================================
# STEP 2: Load and Prepare 2025 Validation Data
# ============================================================================
print("\n[2/5] Loading 2025 Validation Data...")

df = pd.read_csv('sp500_historical_data.csv', index_col='Date', parse_dates=True)
df = df.sort_index()

# Get 2025 data (validation period)
val_df = df[df.index >= '2025-01-01'].copy()

print(f"✓ Loaded {len(val_df)} rows")
print(f"  Period: {val_df.index[0].date()} to {val_df.index[-1].date()}")
print(f"  Price range: ${val_df['close'].min():.2f} - ${val_df['close'].max():.2f}")

# Generate features
print("\n  Generating features...")
val_features = FeatureEngineering.build_complete_feature_set(val_df, include_volume=True)
val_features = val_features.dropna()

# Clean data
X_val = val_features[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

print(f"✓ Features generated: {X_val.shape}")

# ============================================================================
# STEP 3: Create ML Trading Strategy
# ============================================================================
print("\n[3/5] Creating ML Trading Strategy...")


class MLRandomForestStrategy(Strategy):
    """
    Trading strategy based on Random Forest ML model predictions

    Strategy Logic:
    - Use RF model to predict next day's direction
    - Enter long when model predicts up with high confidence
    - Exit when model predicts down or confidence drops
    - Use ATR for stop loss and take profit
    """

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


# Create strategy
ml_strategy = MLRandomForestStrategy(
    model=rf_model,
    feature_cols=feature_cols,
    confidence_threshold=0.6
)

print(f"✓ Strategy created: {ml_strategy.name}")
print(f"  Confidence threshold: {ml_strategy.confidence_threshold}")

# ============================================================================
# STEP 4: Run Backtest
# ============================================================================
print("\n[4/5] Running Backtest on 2025 Data...")

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

# Create backtest engine
engine = BacktestEngine(config)

# Run backtest
print("\n  Executing backtest...")
results = engine.run_backtest(ml_strategy, val_df)

# Extract metrics
metrics = results['metrics']
trades = results['trades']
equity_curve = results['equity_curve']

print("\n✓ Backtest Complete")

# ============================================================================
# STEP 5: Display Results
# ============================================================================
print("\n[5/5] Results Analysis")
print("=" * 80)

print("\n📊 PERFORMANCE METRICS")
print(f"{'─' * 80}")
print(f"  Total Trades:        {metrics.get('total_trades', 0)}")
print(f"  Winning Trades:      {metrics.get('winning_trades', 0)}")
print(f"  Losing Trades:       {metrics.get('losing_trades', 0)}")
print(f"  Win Rate:            {metrics.get('win_rate', 0):.2%}")
print(f"\n  Total Return:        {metrics.get('total_return_pct', 0):.2%}")
print(f"  Annual Return:       {metrics.get('annual_return_pct', 0):.2%}")
print(f"  Sharpe Ratio:        {metrics.get('sharpe_ratio', 0):.3f}")
print(f"  Sortino Ratio:       {metrics.get('sortino_ratio', 0):.3f}")
print(f"  Calmar Ratio:        {metrics.get('calmar_ratio', 0):.3f}")
print(f"\n  Max Drawdown:        {metrics.get('max_drawdown_pct', 0):.2%}")
print(f"  Current Drawdown:    {metrics.get('current_drawdown_pct', 0):.2%}")
print(f"\n  Profit Factor:       {metrics.get('profit_factor', 0):.2f}")
print(f"  Avg Win:             ${metrics.get('avg_win', 0):.2f}")
print(f"  Avg Loss:            ${metrics.get('avg_loss', 0):.2f}")
print(f"  Expectancy:          ${metrics.get('expectancy', 0):.2f}")

# Buy & Hold comparison
buy_hold_return = (val_df['close'].iloc[-1] - val_df['close'].iloc[0]) / val_df['close'].iloc[0]
print(f"\n  Buy & Hold Return:   {buy_hold_return:.2%}")
print(f"  Strategy vs B&H:     {(metrics.get('total_return_pct', 0) - buy_hold_return):.2%}")

# Trade details
if trades:
    print("\n📈 TRADE DETAILS")
    print(f"{'─' * 80}")

    trades_df = pd.DataFrame([{
        'Entry': t.entry_time.strftime('%Y-%m-%d'),
        'Exit': t.exit_time.strftime('%Y-%m-%d'),
        'Direction': t.direction,
        'Entry $': f"${t.entry_price:.2f}",
        'Exit $': f"${t.exit_price:.2f}",
        'P&L': f"${t.pnl:.2f}",
        'P&L %': f"{t.pnl_percent:.2%}",
        'Reason': t.entry_reason[:30]
    } for t in trades[:10]])  # Show first 10 trades

    print(trades_df.to_string(index=False))

    if len(trades) > 10:
        print(f"\n  ... and {len(trades) - 10} more trades")

# Save results
print("\n💾 SAVING RESULTS")
print(f"{'─' * 80}")

# Save to JSON
results_summary = {
    'timestamp': datetime.now().isoformat(),
    'model': model_file,
    'strategy': ml_strategy.name,
    'test_period': f"{val_df.index[0].date()} to {val_df.index[-1].date()}",
    'test_days': len(val_df),
    'metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in metrics.items()},
    'buy_hold_return': float(buy_hold_return),
    'total_trades': len(trades)
}

results_file = 'models/backtest_results_rf_2025.json'
with open(results_file, 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"  Results saved: {results_file}")

# Save equity curve
equity_df = pd.DataFrame({
    'timestamp': equity_curve.index,
    'equity': equity_curve.values
})
equity_file = 'models/equity_curve_rf_2025.csv'
equity_df.to_csv(equity_file, index=False)
print(f"  Equity curve saved: {equity_file}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("BACKTEST SUMMARY")
print("=" * 80)

print("\n🤖 Model: Random Forest (trained on 2020-2024)")
print(f"📅 Test Period: 2025 YTD ({len(val_df)} days)")
print(f"💰 Initial Capital: ${config.initial_capital:,.0f}")
print(f"📊 Final Equity: ${equity_curve.iloc[-1]:,.2f}")
print(f"📈 Total Return: {metrics.get('total_return_pct', 0):.2%}")
print(f"🎯 Win Rate: {metrics.get('win_rate', 0):.1%}")
print(f"📉 Max Drawdown: {metrics.get('max_drawdown_pct', 0):.1%}")
print(f"⚡ Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")

# Verdict
total_return = metrics.get('total_return_pct', 0)
win_rate = metrics.get('win_rate', 0)
sharpe = metrics.get('sharpe_ratio', 0)
total_trades = metrics.get('total_trades', 0)

if total_trades == 0:
    verdict = "❌ NO TRADES - Model too conservative or confidence threshold too high"
elif total_return > buy_hold_return and sharpe > 1.0:
    verdict = "✅ EXCELLENT - Outperformed buy & hold with good Sharpe ratio"
elif total_return > 0 and win_rate > 0.5:
    verdict = "✅ GOOD - Profitable with positive win rate"
elif total_return > 0:
    verdict = "⚠️  MARGINAL - Profitable but needs improvement"
else:
    verdict = "❌ POOR - Strategy needs optimization"

print(f"\n{verdict}")

print("\n" + "=" * 80)
print("Next Steps:")
print("  1. Review individual trades for patterns")
print("  2. Try different confidence thresholds (0.5, 0.7)")
print("  3. Combine with momentum indicators")
print("  4. Test on different time periods")
print("=" * 80)
