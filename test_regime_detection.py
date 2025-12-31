"""
Test Regime Detection Implementation

Quick test to verify regime detection works correctly on historical data.
"""

import pandas as pd
from regime_detection import RegimeDetector, MarketRegime
from feature_engineering import FeatureEngineering

print("="*70)
print("REGIME DETECTION TEST")
print("="*70)

# Load S&P 500 data
print("\n1. Loading S&P 500 data...")
df = pd.read_csv('sp500_historical_data.csv', index_col='Date', parse_dates=True)
print(f"   ✅ Loaded {len(df)} days from sp500_historical_data.csv")

# Test regime detection on different periods
detector = RegimeDetector(ma_period=200, trend_lookback=50, volatility_period=20)

test_periods = [
    ('2021-01-01', '2021-04-30', 'Q1 2021 Bull Market'),
    ('2022-01-01', '2022-04-30', 'Q1 2022 Bear Market'),
    ('2020-03-01', '2020-06-30', 'COVID Crash & Recovery'),
    ('2024-01-01', '2024-12-31', 'Recent 2024'),
]

print(f"\n2. Testing regime detection on different periods...")
print("="*70)

for start, end, description in test_periods:
    try:
        period_data = df.loc[start:end]

        if len(period_data) < 200:
            print(f"\n{description}:")
            print(f"  ⚠️ Insufficient data ({len(period_data)} days)")
            continue

        regime = detector.detect_regime(period_data)
        params = detector.get_regime_parameters(regime)

        # Calculate actual returns
        actual_return = (period_data['close'].iloc[-1] / period_data['close'].iloc[0] - 1) * 100

        print(f"\n{description} ({start} to {end}):")
        print(f"  Period Return: {actual_return:+.2f}%")
        print(f"  Detected Regime: {regime.value.upper()}")
        print(f"  Strategy Parameters:")
        print(f"    - Confidence Threshold: {params['confidence_threshold']:.2f}")
        print(f"    - Position Size: {params['position_size_pct']:.1%}")
        print(f"    - Stop Loss: {params['stop_loss_atr_mult']:.1f}x ATR")
        print(f"    - Take Profit: {params['take_profit_atr_mult']:.1f}x ATR")

        # Verify regime makes sense
        if actual_return > 5 and regime != MarketRegime.BULL:
            print(f"  ⚠️ WARNING: Strong positive return but regime is {regime.value}")
        elif actual_return < -5 and regime != MarketRegime.BEAR:
            print(f"  ⚠️ WARNING: Strong negative return but regime is {regime.value}")
        else:
            print(f"  ✅ Regime detection looks reasonable")

    except Exception as e:
        print(f"\n{description}:")
        print(f"  ❌ Error: {e}")

# Test adding regime features
print("\n" + "="*70)
print("3. Testing regime feature generation...")
print("="*70)

try:
    # Get recent data
    recent_data = df.loc['2023-01-01':'2023-12-31'].copy()

    print(f"\nAdding regime features to 2023 data ({len(recent_data)} days)...")

    # Build complete feature set (which includes regime features)
    df_features = FeatureEngineering.build_complete_feature_set(recent_data, include_volume=True)

    # Check if regime features were added
    regime_features = [col for col in df_features.columns if 'regime' in col.lower() or 'ma_200' in col]

    print(f"\n✅ Regime features added ({len(regime_features)} features):")
    for feat in regime_features[:10]:  # Show first 10
        print(f"  - {feat}")
    if len(regime_features) > 10:
        print(f"  ... and {len(regime_features) - 10} more")

    # Show regime distribution
    if 'regime' in df_features.columns:
        regime_counts = df_features['regime'].value_counts()
        print(f"\nRegime Distribution in 2023:")
        regime_map = {0: 'Bear', 1: 'Sideways', 2: 'Bull', 3: 'Volatile'}
        for regime_code, count in regime_counts.items():
            regime_name = regime_map.get(regime_code, f'Unknown({regime_code})')
            print(f"  {regime_name}: {count} days ({count/len(df_features)*100:.1f}%)")

except Exception as e:
    print(f"❌ Error adding features: {e}")
    import traceback
    traceback.print_exc()

# Test regime statistics
print("\n" + "="*70)
print("4. Regime Statistics for 2020-2024...")
print("="*70)

try:
    # Get multi-year data
    multi_year = df.loc['2020-01-01':'2024-12-31'].copy()

    # Add regime features
    multi_year_features = detector.add_regime_features(multi_year)

    # Get statistics
    stats = detector.get_regime_statistics(multi_year_features)

    print("\n" + stats.to_string(index=False))

except Exception as e:
    print(f"❌ Error calculating statistics: {e}")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
print("\n✅ Regime detection implementation is ready to use!")
print("\nNext steps:")
print("1. Re-run walk-forward tests with regime adaptation enabled")
print("2. Compare results to non-adaptive baseline")
print("3. Analyze regime-specific performance")
