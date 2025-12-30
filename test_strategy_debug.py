#!/usr/bin/env python3
"""
Debug script to understand why no trades are generated
"""

import pandas as pd
from momentum_strategy_80_20 import MomentumStrategy

# Load a split
dataset_name = "Dataset_1_Recent"
split_num = 1

train_file = f'walkforward_results/{dataset_name}/split_{split_num}_train.csv'
test_file = f'walkforward_results/{dataset_name}/split_{split_num}_test.csv'

train_data = pd.read_csv(train_file, index_col='Date', parse_dates=True)
test_data = pd.read_csv(test_file, index_col='Date', parse_dates=True)

# Combine data
combined_data = pd.concat([train_data, test_data])

print("=" * 80)
print("DEBUGGING MOMENTUM STRATEGY")
print("=" * 80)

print(f"\nTrain data: {len(train_data)} rows from {train_data.index[0]} to {train_data.index[-1]}")
print(f"Test data: {len(test_data)} rows from {test_data.index[0]} to {test_data.index[-1]}")
print(f"Combined data: {len(combined_data)} rows from {combined_data.index[0]} to {combined_data.index[-1]}")

# Create strategy
strategy = MomentumStrategy(
    short_window=10,
    long_window=20,
    roc_period=10,
    roc_threshold=0.5
)

# Calculate indicators on combined data
df_with_indicators = strategy.calculate_indicators(combined_data)

print("\n" + "=" * 80)
print("INDICATORS CALCULATED")
print("=" * 80)
print(df_with_indicators[['close', 'sma_short', 'sma_long', 'roc', 'atr', 'trend_strength']].tail(15))

# Test signal generation for each test period timestamp
print("\n" + "=" * 80)
print("SIGNAL GENERATION DURING TEST PERIOD")
print("=" * 80)

for timestamp in test_data.index:
    # Get data up to this timestamp
    historical_data = combined_data[combined_data.index <= timestamp]

    # Generate signals
    signals = strategy.generate_signals(historical_data, timestamp)

    # Get current values
    current = df_with_indicators.loc[timestamp]

    print(f"\n{timestamp}:")
    print(f"  Close: ${current['close']:.2f}")
    print(f"  SMA Short: {current['sma_short']:.2f}, SMA Long: {current['sma_long']:.2f}")
    print(f"  ROC: {current['roc']:.2f}%")
    print(f"  Trend Strength: {current['trend_strength']:.2f}%")
    print(f"  Has Position: {len(strategy.positions) > 0}")
    print(f"  Signals Generated: {len(signals)}")

    if signals:
        for sig in signals:
            print(f"    → {sig['action'].upper()}: {sig.get('reason', 'N/A')}")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

# Check why no signals
test_indicators = df_with_indicators.loc[test_data.index]

print("\nDuring test period:")
print(
    f"  SMA crossovers: {
        (
            (test_indicators['sma_short'] > test_indicators['sma_long']).shift(1) != (
                test_indicators['sma_short'] > test_indicators['sma_long'])).sum()}")
print(f"  Positive ROC > 0.5%: {(test_indicators['roc'] > 0.5).sum()} / {len(test_indicators)}")
print(f"  Price > SMA Long: {(test_indicators['close'] > test_indicators['sma_long']).sum()} / {len(test_indicators)}")
print(f"  Positive trend: {(test_indicators['trend_strength'] > 0).sum()} / {len(test_indicators)}")

print("\nROC values during test period:")
print(test_indicators['roc'].values)

print("\nTrend strength during test period:")
print(test_indicators['trend_strength'].values)
