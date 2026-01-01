#!/usr/bin/env python3
"""
Diagnostic Script: Zero Trades Issue
Test adaptive strategy on single period with debug logging
"""

import pandas as pd
import numpy as np
from pathlib import Path

from feature_engineering import FeatureEngineering
from ml_training_pipeline import MLModelTrainer
from backtesting_engine import BacktestEngine, BacktestConfig
from regime_adaptive_strategy import RegimeAdaptiveMLStrategy

# Top 20 features
TOP_20_FEATURES = [
    'bullish_engulfing', 'stoch_d_3', 'week_of_year', 'atr_14', 'regime',
    'roc_20', 'obv', 'parkinson_vol_10', 'volatility_200d', 'momentum_5',
    'macd_signal', 'adx_14', 'month_sin', 'hl_ratio', 'rsi_14',
    'stoch_k_14', 'bb_position_20', 'momentum_oscillator', 'pvt', 'price_acceleration'
]

def main():
    print("="*80)
    print("DIAGNOSTIC: Zero Trades Issue")
    print("Testing Split 9: 2023-12-29 to 2024-04-01 (Bull Market)")
    print("="*80)

    # 1. Load data
    print("\n[1/6] Loading S&P 500 data...")
    df = pd.read_csv('sp500_historical_data.csv', index_col='Date', parse_dates=True)
    df = df.loc['2020-01-01':]
    print(f"  ✅ Loaded {len(df)} days")

    # 2. Generate features
    print("\n[2/6] Generating features...")
    df_features = FeatureEngineering.build_complete_feature_set(df, include_volume=True)
    df_features = df_features.dropna()
    print(f"  ✅ Clean data points: {len(df_features)}")

    # 3. Create Split 9 manually
    print("\n[3/6] Creating Split 9...")
    days_per_month = 21
    train_days = 12 * days_per_month  # 12 months
    test_days = 3 * days_per_month    # 3 months
    step_days = 3 * days_per_month    # 3 month step

    # Split 9 = 9th iteration
    start_idx = 8 * step_days  # 8 steps forward
    train_start = start_idx
    train_end = start_idx + train_days
    test_start = train_end
    test_end = test_start + test_days

    train_data = df_features.iloc[train_start:train_end]
    test_data = df_features.iloc[test_start:test_end]

    print(f"  Train: {train_data.index[0]} to {train_data.index[-1]} ({len(train_data)} days)")
    print(f"  Test:  {test_data.index[0]} to {test_data.index[-1]} ({len(test_data)} days)")

    # 4. Train model
    print("\n[4/6] Training model...")
    df_train = train_data.copy()
    df_train['target'] = (df_train['close'].shift(-1) > df_train['close']).astype(int)
    df_train = df_train.dropna()

    X_train = df_train[TOP_20_FEATURES].values
    y_train = df_train['target'].values

    print(f"  Training samples: {len(X_train)}")
    print(f"  Target distribution: {np.bincount(y_train)}")

    trainer = MLModelTrainer(model_type='xgboost', task='classification')
    trainer.train(X_train, y_train, hyperparameter_tuning=True)

    model_path = 'regime_adaptive_results/diagnostic_model_split9.pkl'
    trainer.save_model(model_path)
    print(f"  ✅ Model trained and saved")

    # 5. Test adaptive strategy
    print("\n[5/6] Testing Adaptive Strategy...")
    print("="*80)
    print("DEBUG OUTPUT (from generate_signals):")
    print("="*80)

    strategy = RegimeAdaptiveMLStrategy(
        model_path=model_path,
        feature_cols=TOP_20_FEATURES,
        base_confidence_threshold=0.50,
        enable_regime_adaptation=True,
        skip_volatile_regimes=False,
        skip_bear_regimes=False
    )

    # Backtest configuration
    backtest_config = BacktestConfig(
        initial_capital=10000.0,
        commission_pct=0.001,
        slippage_pct=0.0002,
        position_size_pct=0.02,
        max_position_value_pct=0.02,
        max_positions=1,
        max_daily_loss_pct=0.03,
        max_drawdown_pct=0.15,
        position_sizing_method='volatility',
    )

    # Run backtest
    combined_data = pd.concat([train_data, test_data])
    engine = BacktestEngine(backtest_config)
    test_start_date = test_data.index[0]

    results = engine.run_backtest(
        strategy,
        combined_data,
        trading_start_date=test_start_date
    )

    print("="*80)
    print("\n[6/6] Results:")
    print("="*80)

    metrics = results['metrics']
    trades = results['trades']

    print(f"\n📊 Performance:")
    print(f"  Total Trades: {metrics.get('total_trades', 0)}")
    print(f"  Win Rate: {metrics.get('win_rate', 0):.1%}")
    print(f"  Total Return: {metrics.get('total_return_pct', 0):+.2%}")
    print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"  Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2%}")

    if len(trades) > 0:
        print(f"\n✅ SUCCESS! Generated {len(trades)} trades")
        print("\nTrade Details:")
        for i, trade in enumerate(trades[:5], 1):
            print(f"  {i}. {trade['entry_date'].date()}: {trade['action'].upper()} @ {trade['entry_price']:.2f} "
                  f"→ {trade['exit_date'].date()}: EXIT @ {trade['exit_price']:.2f} "
                  f"({trade['profit_loss_pct']:+.2%})")
        if len(trades) > 5:
            print(f"  ... and {len(trades) - 5} more trades")
    else:
        print(f"\n❌ FAILURE! Generated 0 trades")
        print("\nCheck DEBUG output above to identify the blocking condition!")

    # Show regime statistics
    print("\n📈 Regime Statistics:")
    regime_stats = strategy.get_regime_statistics()
    print(regime_stats.to_string(index=False))

    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
