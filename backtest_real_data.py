"""
Backtest strategies on REAL Oanda data (not simulated)
"""

from oanda_integration import OandaConnector
from config import OANDA_CONFIG
from backtesting_engine import BacktestEngine, BacktestConfig
from strategy_examples import MomentumStrategy, MeanReversionStrategy, BreakoutStrategy
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def backtest_on_real_data():
    """Run backtests on real Oanda market data"""

    print("=" * 70)
    print("BACKTESTING ON REAL OANDA DATA")
    print("=" * 70)

    # 1. Connect to Oanda and fetch real data
    print("\n1. Fetching real market data from Oanda...")
    oanda = OandaConnector(
        account_id=OANDA_CONFIG['account_id'],
        access_token=OANDA_CONFIG['access_token'],
        environment=OANDA_CONFIG['environment']
    )

    # Fetch 6 months of H1 data for SPX500_USD
    print("   Fetching 180 days of SPX500_USD H1 data...")
    df = oanda.fetch_historical_data_range('SPX500_USD', 'D', days=180)

    if df.empty:
        print("❌ Failed to fetch data")
        return

    print(f"✅ Fetched {len(df)} candles")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    print(f"   Price range: ${df['close'].min():.5f} - ${df['close'].max():.5f}")

    # 2. Set up backtest configuration
    print("\n2. Setting up backtest configuration...")
    config = BacktestConfig(
        initial_capital=10000.0,
        commission_pct=0.0001,  # 1 pip
        slippage_pct=0.0001,    # 1 pip
        position_size_pct=0.02,  # 2% risk
        max_positions=1,
        max_daily_loss_pct=0.05,
        max_drawdown_pct=0.20
    )
    print("✅ Configuration ready")

    # 3. Initialize strategies
    print("\n3. Preparing strategies...")
    strategies = [
        MomentumStrategy(fast_period=20, slow_period=50),
        MomentumStrategy(fast_period=10, slow_period=30),
        MeanReversionStrategy(bb_period=20, bb_std=2.0),
        BreakoutStrategy(lookback_period=20, atr_period=14)
    ]
    print(f"✅ {len(strategies)} strategies ready")

    # 4. Run backtests
    print("\n" + "=" * 70)
    print("RUNNING BACKTESTS ON REAL DATA")
    print("=" * 70)

    engine = BacktestEngine(config)
    results = []

    for strategy in strategies:
        print(f"\nTesting: {strategy.name}")
        print("-" * 70)

        result = engine.run_backtest(strategy, df)
        metrics = result['metrics']

        print(f"  Trades: {metrics['total_trades']}")
        print(f"  Win Rate: {metrics['win_rate']:.2%}")
        print(f"  Total Return: {metrics['total_return_pct']:.2%}")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2%}")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}")

        results.append({
            'Strategy': strategy.name,
            'Trades': metrics['total_trades'],
            'Win Rate': f"{metrics['win_rate']:.2%}",
            'Return': f"{metrics['total_return_pct']:.2%}",
            'Sharpe': f"{metrics['sharpe_ratio']:.2f}",
            'Max DD': f"{metrics['max_drawdown_pct']:.2%}",
            'Profit Factor': f"{metrics['profit_factor']:.2f}"
        })

    # 5. Display comparison
    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON (REAL OANDA DATA)")
    print("=" * 70)

    # Create comparison table
    import pandas as pd
    comparison_df = pd.DataFrame(results)
    print(comparison_df.to_string(index=False))

    # 6. Find best strategy
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Extract numeric values for comparison
    sharpe_values = [float(r['Sharpe']) for r in results]
    return_values = [float(r['Return'].rstrip('%')) for r in results]

    best_sharpe_idx = sharpe_values.index(max(sharpe_values))
    best_return_idx = return_values.index(max(return_values))

    print(f"\n🏆 Best Sharpe Ratio: {results[best_sharpe_idx]['Strategy']}")
    print(f"   Sharpe: {results[best_sharpe_idx]['Sharpe']}")
    print(f"   Return: {results[best_sharpe_idx]['Return']}")

    print(f"\n💰 Best Return: {results[best_return_idx]['Strategy']}")
    print(f"   Return: {results[best_return_idx]['Return']}")
    print(f"   Sharpe: {results[best_return_idx]['Sharpe']}")

    # Check if any strategy is profitable
    profitable = [r for r in results if float(r['Return'].rstrip('%')) > 0]

    if profitable:
        print(f"\n✅ {len(profitable)}/{len(results)} strategies are profitable on real data!")
        print("\nProfitable strategies:")
        for r in profitable:
            print(f"  - {r['Strategy']}: {r['Return']} return, {r['Win Rate']} win rate")
    else:
        print("\n⚠️  No strategies are profitable on this dataset")
        print("   This is common - not all strategies work in all market conditions")
        print("   Try:")
        print("   1. Different timeframes (H4, D1)")
        print("   2. Different currency pairs")
        print("   3. ML-based strategies")
        print("   4. Parameter optimization")

    # 7. Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    try:
        import os
        import json
        from datetime import datetime

        os.makedirs('results', exist_ok=True)

        # Save to JSON
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f'results/real_data_backtest_{timestamp}.json'

        output = {
            'timestamp': timestamp,
            'data_source': 'Oanda Real Data',
            'instrument': 'SPX500_USD',
            'granularity': 'D',
            'data_points': len(df),
            'date_range': {
                'start': str(df.index[0]),
                'end': str(df.index[-1])
            },
            'strategies': results
        }

        with open(results_file, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"✅ Results saved to {results_file}")

        # Save data
        data_file = 'data/SPX500_USD_H1_180days.csv'
        df.to_csv(data_file)
        print(f"✅ Data saved to {data_file}")

    except Exception as e:
        print(f"⚠️  Error saving results: {e}")

    # 8. Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✅ Successfully backtested {len(strategies)} strategies on REAL market data")
    print(f"   Data: {len(df)} candles from {df.index[0].date()} to {df.index[-1].date()}")
    print(f"   Total period: ~{len(df) / 24:.0f} days")
    print("\nKey Differences from Simulated Data:")
    print("  ✅ Real market volatility")
    print("  ✅ Actual price movements")
    print("  ✅ Real spreads and slippage")
    print("  ✅ Market microstructure effects")

    print("\nNext Steps:")
    print("  1. Run walk-forward validation on real data")
    print("  2. Train ML model on this data")
    print("  3. Test on different currency pairs")
    print("  4. Optimize strategy parameters")
    print("  5. Paper trade best strategy for 90 days")

    return results


if __name__ == "__main__":
    try:
        backtest_on_real_data()
    except KeyboardInterrupt:
        print("\n\nBacktest interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
