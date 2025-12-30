#!/usr/bin/env python3
"""
Debug the actual backtest execution
"""

import pandas as pd
from momentum_strategy_80_20 import MomentumStrategy
from backtesting_engine import BacktestEngine, BacktestConfig

# Load split data
dataset_name = "Dataset_1_Recent"
split_num = 1

train_file = f'walkforward_results/{dataset_name}/split_{split_num}_train.csv'
test_file = f'walkforward_results/{dataset_name}/split_{split_num}_test.csv'

train_data = pd.read_csv(train_file, index_col='Date', parse_dates=True)
test_data = pd.read_csv(test_file, index_col='Date', parse_dates=True)

# Combine data
combined_data = pd.concat([train_data, test_data])
test_start_date = test_data.index[0]

print("=" * 80)
print("BACKTEST EXECUTION DEBUG")
print("=" * 80)

print(f"\nCombined data: {len(combined_data)} rows")
print(f"Test start date: {test_start_date}")
print(f"Test data index: {test_data.index.tolist()}")

# Create strategy
strategy = MomentumStrategy()

# Create config
config = BacktestConfig(
    initial_capital=10000.0,
    commission_pct=0.0001,
    slippage_pct=0.0001,
    position_size_pct=0.02,
    max_positions=1,
    leverage=1.0,
    position_sizing_method='volatility',
    max_daily_loss_pct=0.05,
    max_drawdown_pct=0.20
)

# Create engine
engine = BacktestEngine(config)

print(f"\nRunning backtest with start_date={test_start_date}...")

# Run backtest
results = engine.run_backtest(strategy, combined_data, start_date=test_start_date)

print("\nBacktest Results:")
print(f"  Total Trades: {len(results['trades'])}")
print(f"  Equity Curve Points: {len(results['equity_curve'])}")

if results['trades']:
    print("\nTrades:")
    for i, trade in enumerate(results['trades'], 1):
        print(
            f"  {i}. {
                trade.entry_time}: {
                trade.direction} @ ${
                trade.entry_price:.2f} → ${
                    trade.exit_price:.2f}, PnL: {
                        trade.pnl:.2f}")
else:
    print("\n  No trades executed!")

    # Debug: manually check signal generation
    print("\n" + "=" * 80)
    print("MANUAL SIGNAL CHECK")
    print("=" * 80)

    for i, (timestamp, row) in enumerate(combined_data.iterrows()):
        if timestamp >= test_start_date:
            historical_data = combined_data.iloc[:i + 1]
            signals = strategy.generate_signals(historical_data, timestamp)

            if signals:
                print(f"\n{timestamp}: {len(signals)} signal(s) generated")
                for sig in signals:
                    print(f"  Action: {sig['action']}, Reason: {sig.get('reason', 'N/A')}")
                print("  But trade was NOT executed - investigating...")
