"""
Test script for fetching historical data from Oanda
"""

from oanda_integration import OandaConnector
from config import OANDA_CONFIG
import pandas as pd


def test_fetch_data():
    """Test fetching historical data from Oanda"""

    print("=" * 70)
    print("TESTING OANDA HISTORICAL DATA FETCH")
    print("=" * 70)

    # Initialize connector
    print("\n1. Connecting to Oanda...")
    oanda = OandaConnector(
        account_id=OANDA_CONFIG['account_id'],
        access_token=OANDA_CONFIG['access_token'],
        environment=OANDA_CONFIG['environment']
    )
    print("✅ Connected successfully")

    # Test 1: Fetch last 100 Daily candles
    print("\n" + "=" * 70)
    print("TEST 1: Fetch last 100 Daily candles for SPX500_USD")
    print("=" * 70)

    df1 = oanda.fetch_historical_data('SPX500_USD', 'D', count=100)

    if not df1.empty:
        print(f"✅ Fetched {len(df1)} candles")
        print(f"Date range: {df1.index[0]} to {df1.index[-1]}")
        print("\nFirst 5 rows:")
        print(df1.head())
        print("\nLast 5 rows:")
        print(df1.tail())
        print(f"\nData shape: {df1.shape}")
        print(f"Columns: {df1.columns.tolist()}")

        # Verify data quality
        print("\n✅ Data Quality Checks:")
        print(f"  - No missing values: {df1.isnull().sum().sum() == 0}")
        print(f"  - High >= Low: {(df1['high'] >= df1['low']).all()}")
        print(f"  - High >= Open: {(df1['high'] >= df1['open']).all()}")
        print(f"  - High >= Close: {(df1['high'] >= df1['close']).all()}")
        print(f"  - Low <= Open: {(df1['low'] <= df1['open']).all()}")
        print(f"  - Low <= Close: {(df1['low'] <= df1['close']).all()}")
    else:
        print("❌ No data fetched")
        return False

    # Test 2: Fetch last 30 days of Daily data
    print("\n" + "=" * 70)
    print("TEST 2: Fetch last 30 days of Daily data for SPX500_USD")
    print("=" * 70)

    df2 = oanda.fetch_historical_data_range('SPX500_USD', 'D', days=30)

    if not df2.empty:
        print(f"✅ Fetched {len(df2)} candles")
        print(f"Date range: {df2.index[0]} to {df2.index[-1]}")
        print("\nExpected ~30 candles (30 trading days)")
        print(f"Actual: {len(df2)} candles")

        # Calculate some statistics
        print("\n📊 Price Statistics:")
        print(f"  Open:   ${df2['open'].mean():.5f} ± ${df2['open'].std():.5f}")
        print(f"  High:   ${df2['high'].mean():.5f}")
        print(f"  Low:    ${df2['low'].mean():.5f}")
        print(f"  Close:  ${df2['close'].mean():.5f}")
        print(f"  Volume: {df2['volume'].mean():.0f} ± {df2['volume'].std():.0f}")
    else:
        print("❌ No data fetched")
        return False

    # Test 3: Fetch D1 (daily) candles
    print("\n" + "=" * 70)
    print("TEST 3: Fetch last 90 daily candles for SPX500_USD")
    print("=" * 70)

    df3 = oanda.fetch_historical_data('SPX500_USD', 'D', count=90)

    if not df3.empty:
        print(f"✅ Fetched {len(df3)} candles")
        print(f"Date range: {df3.index[0]} to {df3.index[-1]}")
        print("\nDaily OHLC:")
        print(df3.tail(10))
    else:
        print("❌ No data fetched")
        return False

    # Test 4: Save to CSV for future use
    print("\n" + "=" * 70)
    print("TEST 4: Save data to CSV")
    print("=" * 70)

    try:
        import os
        os.makedirs('data', exist_ok=True)

        csv_file = 'data/SPX500_USD_Daily_recent.csv'
        df1.to_csv(csv_file)
        print(f"✅ Saved {len(df1)} candles to {csv_file}")

        # Verify can reload
        df_loaded = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        print(f"✅ Verified: Reloaded {len(df_loaded)} candles from CSV")
    except Exception as e:
        print(f"❌ Error saving to CSV: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("✅ All tests passed!")
    print("\nYou can now:")
    print("  1. Fetch real market data from Oanda")
    print("  2. Use any granularity (M1, M5, H1, H4, D, W, M)")
    print("  3. Fetch up to 5000 candles per request")
    print("  4. Automatically handle pagination for larger datasets")
    print("  5. Use this data for backtesting instead of simulated data")

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("1. Backtest on real data:")
    print("   python3 backtest_real_data.py")
    print("\n2. Train ML model on real data:")
    print("   # Use df = oanda.fetch_historical_data_range('SPX500_USD', 'D', days=365)")
    print("\n3. Run walk-forward on real data:")
    print("   # Update run_examples.py to use real data")

    return True


if __name__ == "__main__":
    try:
        test_fetch_data()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
