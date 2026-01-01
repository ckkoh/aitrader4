#!/usr/bin/env python3
"""
Debug Ensemble Regime Detection
Test why ensemble always uses balanced model
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from feature_engineering import FeatureEngineering
from regime_adaptive_strategy import RegimeDetector
from ensemble_regime_strategy import EnsembleRegimeStrategy

# Top 20 features
TOP_20_FEATURES = [
    'bullish_engulfing', 'stoch_d_3', 'week_of_year', 'atr_14', 'regime',
    'roc_20', 'obv', 'parkinson_vol_10', 'volatility_200d', 'momentum_5',
    'macd_signal', 'adx_14', 'month_sin', 'hl_ratio', 'rsi_14',
    'stoch_k_14', 'bb_position_20', 'momentum_oscillator', 'pvt', 'price_acceleration'
]


def test_regime_detection():
    """Test regime detection across all periods"""
    print("="*80)
    print("REGIME DETECTION DIAGNOSTIC")
    print("="*80)

    # Load data
    print("\n[1/4] Loading data...")
    df = pd.read_csv('sp500_historical_data.csv', index_col='Date', parse_dates=True)
    df = df.loc['2020-01-01':]

    # Generate features
    print("\n[2/4] Generating features...")
    df_features = FeatureEngineering.build_complete_feature_set(df, include_volume=True)
    df_features = df_features.dropna()

    # Create splits
    print("\n[3/4] Creating splits...")
    days_per_month = 21
    train_days = 12 * days_per_month
    test_days = 3 * days_per_month
    step_days = 3 * days_per_month

    splits = []
    start_idx = 0

    while start_idx + train_days + test_days <= len(df_features):
        train_start = start_idx
        train_end = start_idx + train_days
        test_start = train_end
        test_end = test_start + test_days

        train_data = df_features.iloc[train_start:train_end]
        test_data = df_features.iloc[test_start:test_end]

        splits.append({
            'split_num': len(splits) + 1,
            'train_data': train_data,
            'test_data': test_data,
            'test_start': test_data.index[0],
            'test_end': test_data.index[-1],
        })

        start_idx += step_days

    print(f"   Created {len(splits)} splits")

    # Test regime detection on each period
    print("\n[4/4] Testing regime detection on all periods...")
    print("="*80)

    regime_counts = {'BULL': 0, 'BEAR': 0, 'VOLATILE': 0, 'SIDEWAYS': 0, 'UNKNOWN': 0}

    for split in splits[:15]:  # Test first 15 splits
        split_num = split['split_num']
        test_data = split['test_data']

        print(f"\n{'='*80}")
        print(f"Split {split_num}: {split['test_start']} to {split['test_end']}")
        print(f"{'='*80}")

        # Test regime detection at multiple points in the period
        regimes_in_period = []

        # Test at start, middle, and end of period
        test_points = [0, len(test_data)//2, -1]

        for i, point in enumerate(test_points):
            # Get data up to this point
            if point == -1:
                data_slice = pd.concat([split['train_data'], test_data])
            else:
                data_slice = pd.concat([split['train_data'], test_data.iloc[:point+1]])

            # Detect regime
            regime_info = RegimeDetector.detect_regime(data_slice, lookback=50)
            regimes_in_period.append(regime_info['regime'])

            point_name = ['START', 'MIDDLE', 'END'][i]
            print(f"\n  {point_name} of period ({data_slice.index[-1]}):")
            print(f"    Regime: {regime_info['regime']}")
            print(f"    Volatility: {regime_info['volatility']:.6f}")
            print(f"    Trend Strength: {regime_info['trend_strength']:.6f}")
            print(f"    Trend Direction: {regime_info.get('trend_direction', 'N/A')}")

        # Count most common regime in period
        most_common_regime = max(set(regimes_in_period), key=regimes_in_period.count)
        regime_counts[most_common_regime] += 1

        print(f"\n  → Dominant regime: {most_common_regime}")
        print(f"  → Regimes detected: {regimes_in_period}")

    # Summary
    print("\n" + "="*80)
    print("REGIME DISTRIBUTION SUMMARY")
    print("="*80)

    for regime, count in regime_counts.items():
        pct = count / 15 * 100 if count > 0 else 0
        print(f"  {regime}: {count}/15 periods ({pct:.1f}%)")

    print("\n" + "="*80)
    print("EXPECTED vs ACTUAL")
    print("="*80)

    print("\nExpected:")
    print("  - 2022 (Splits 1-4): BEAR or VOLATILE (bear market)")
    print("  - 2023-2024 (Splits 5-15): BULL or SIDEWAYS (bull/recovery)")

    print("\nActual:")
    total_bear_volatile = regime_counts['BEAR'] + regime_counts['VOLATILE']
    total_bull_sideways = regime_counts['BULL'] + regime_counts['SIDEWAYS']
    print(f"  - BEAR/VOLATILE: {total_bear_volatile}/15 ({total_bear_volatile/15*100:.1f}%)")
    print(f"  - BULL/SIDEWAYS: {total_bull_sideways}/15 ({total_bull_sideways/15*100:.1f}%)")

    if total_bull_sideways > 12:
        print("\n❌ BUG CONFIRMED: Regime detector classifies most periods as BULL/SIDEWAYS")
        print("   This explains why ensemble always uses Balanced model!")
    else:
        print("\n✅ Regime detection looks reasonable")


def test_ensemble_model_selection():
    """Test ensemble model selection logic"""
    print("\n\n" + "="*80)
    print("ENSEMBLE MODEL SELECTION LOGIC TEST")
    print("="*80)

    # Create dummy strategy to test logic
    original_model = 'ensemble_results/model_split_1_original.pkl'
    balanced_model = 'ensemble_results/model_split_1_balanced.pkl'

    if not Path(original_model).exists() or not Path(balanced_model).exists():
        print("\n⚠️  Model files not found, skipping this test")
        return

    strategy = EnsembleRegimeStrategy(
        original_model_path=original_model,
        balanced_model_path=balanced_model,
        feature_cols=TOP_20_FEATURES,
        base_confidence_threshold=0.50,
        enable_regime_adaptation=True
    )

    print("\nTesting get_adaptive_settings() for each regime:")
    print("="*80)

    test_regimes = ['BULL', 'BEAR', 'VOLATILE', 'SIDEWAYS', 'UNKNOWN']

    for regime in test_regimes:
        regime_info = {'regime': regime, 'volatility': 0.02, 'trend_strength': 0.5, 'trend_direction': 1}
        settings = strategy.get_adaptive_settings(regime_info)

        model_name = 'Balanced' if settings['use_balanced'] else 'Original'

        print(f"\n  {regime}:")
        print(f"    Model: {model_name}")
        print(f"    Confidence threshold: {settings['confidence_threshold']:.2f}")
        print(f"    Position multiplier: {settings['position_multiplier']:.1f}x")
        print(f"    Description: {settings['description']}")

    print("\n" + "="*80)
    print("EXPECTED MODEL SELECTION:")
    print("="*80)
    print("  BULL → Balanced ✓")
    print("  BEAR → Original ✓")
    print("  VOLATILE → Original ✓")
    print("  SIDEWAYS → Balanced ✓")
    print("  UNKNOWN → Balanced ✓")


def test_signal_generation():
    """Test signal generation on Split 1 (2022 bear market)"""
    print("\n\n" + "="*80)
    print("SIGNAL GENERATION TEST - Split 1 (2022 Bear Market)")
    print("="*80)

    # Load data
    df = pd.read_csv('sp500_historical_data.csv', index_col='Date', parse_dates=True)
    df = df.loc['2020-01-01':]
    df_features = FeatureEngineering.build_complete_feature_set(df, include_volume=True)
    df_features = df_features.dropna()

    # Create Split 1
    days_per_month = 21
    train_days = 12 * days_per_month
    test_days = 3 * days_per_month

    train_data = df_features.iloc[0:train_days]
    test_data = df_features.iloc[train_days:train_days+test_days]

    print(f"\nTrain: {train_data.index[0]} to {train_data.index[-1]}")
    print(f"Test:  {test_data.index[0]} to {test_data.index[-1]}")

    # Load models
    original_model = 'ensemble_results/model_split_1_original.pkl'
    balanced_model = 'ensemble_results/model_split_1_balanced.pkl'

    if not Path(original_model).exists() or not Path(balanced_model).exists():
        print("\n⚠️  Model files not found, skipping signal generation test")
        return

    strategy = EnsembleRegimeStrategy(
        original_model_path=original_model,
        balanced_model_path=balanced_model,
        feature_cols=TOP_20_FEATURES,
        base_confidence_threshold=0.50,
        enable_regime_adaptation=True
    )

    print("\nGenerating signals for test period...")
    print("="*80)

    signal_count = 0
    regime_usage = {'Balanced': 0, 'Original': 0}

    # Simulate day-by-day signal generation
    for i in range(len(test_data)):
        # Get data up to current day
        current_data = pd.concat([train_data, test_data.iloc[:i+1]])
        current_timestamp = test_data.index[i]

        # Detect regime
        regime_info = RegimeDetector.detect_regime(current_data, lookback=50)
        settings = strategy.get_adaptive_settings(regime_info)

        model_used = 'Balanced' if settings['use_balanced'] else 'Original'
        regime_usage[model_used] += 1

        # Generate signal (just for counting)
        signals = strategy.generate_signals(current_data, current_timestamp)

        if signals:
            signal_count += len(signals)
            print(f"\n  Day {i+1} ({current_timestamp.date()}):")
            print(f"    Regime: {regime_info['regime']}")
            print(f"    Model used: {model_used}")
            print(f"    Signals: {len(signals)}")
            for signal in signals:
                print(f"      - {signal['action'].upper()} ({signal['reason']})")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nTotal signals: {signal_count}")
    print(f"\nModel usage:")
    for model, count in regime_usage.items():
        pct = count / len(test_data) * 100
        print(f"  {model}: {count}/{len(test_data)} days ({pct:.1f}%)")

    if regime_usage['Balanced'] > regime_usage['Original'] * 2:
        print("\n❌ BUG: Balanced model used much more than Original in 2022 bear market!")
    else:
        print("\n✅ Model usage looks reasonable")


def main():
    """Run all diagnostic tests"""

    try:
        # Test 1: Regime detection
        test_regime_detection()

        # Test 2: Model selection logic
        test_ensemble_model_selection()

        # Test 3: Signal generation
        test_signal_generation()

        print("\n\n" + "="*80)
        print("DIAGNOSTIC COMPLETE")
        print("="*80)
        print("\nCheck the output above to identify the bug:")
        print("1. Is regime detection working? (should see BEAR in 2022)")
        print("2. Is model selection logic correct? (BEAR → Original)")
        print("3. Which model is actually being used during signal generation?")
        print("="*80)

    except Exception as e:
        print(f"\n❌ Error during diagnostic: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
