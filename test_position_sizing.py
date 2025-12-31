"""Test position sizing fix with S&P 500 data"""

import pandas as pd
from backtesting_engine import BacktestEngine, BacktestConfig
from strategy_examples import MomentumStrategy, MeanReversionStrategy, BreakoutStrategy

# Load S&P 500 data
df = pd.read_csv('sp500_ytd_2025.csv', index_col='Date', parse_dates=True)

print("=" * 70)
print("TESTING POSITION SIZING FIX")
print("=" * 70)
print(f"\nData: S&P 500 YTD 2025")
print(f"Rows: {len(df)}")
print(f"Period: {df.index[0]} to {df.index[-1]}")

# Conservative configuration
config = BacktestConfig(
    initial_capital=10000.0,
    commission_pct=0.001,  # 0.1%
    position_size_pct=0.02,  # 2% risk
    max_position_value_pct=0.05,  # Max 5% of capital per position
    max_positions=1,
    max_daily_loss_pct=0.05,
    max_drawdown_pct=0.20
)

print("\nPosition Sizing Config:")
print(f"  Risk per trade: {config.position_size_pct:.1%}")
print(f"  Max position value: {config.max_position_value_pct:.1%} of capital")
print(f"  Initial capital: ${config.initial_capital:,.0f}")
print(f"  Max drawdown: {config.max_drawdown_pct:.0%}")

# Test strategies
strategies = [
    MomentumStrategy(fast_period=10, slow_period=30),
    MeanReversionStrategy(bb_period=20, bb_std=2.0),
    BreakoutStrategy(lookback_period=20, atr_period=14)
]

engine = BacktestEngine(config)
results = []

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

for strategy in strategies:
    result = engine.run_backtest(strategy, df)
    metrics = result['metrics']
    
    print(f"\n{strategy.name}:")
    print(f"  Trades: {metrics['total_trades']}")
    print(f"  Win Rate: {metrics['win_rate']:.1%}")
    print(f"  Return: {metrics['total_return_pct']:.2%}")
    print(f"  Sharpe: {metrics['sharpe_ratio']:.2f}")
    print(f"  Max DD: {metrics['max_drawdown_pct']:.2%}")
    print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    
    results.append({
        'Strategy': strategy.name,
        'Trades': metrics['total_trades'],
        'Win Rate': metrics['win_rate'],
        'Return': metrics['total_return_pct'],
        'Max DD': metrics['max_drawdown_pct']
    })

# Summary
results_df = pd.DataFrame(results)
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(results_df.to_string(index=False))

profitable = results_df[results_df['Return'] > 0]
print(f"\n✅ {len(profitable)}/{len(results_df)} strategies profitable")
print(f"Max drawdown range: {results_df['Max DD'].min():.1%} to {results_df['Max DD'].max():.1%}")

if results_df['Max DD'].max() < 0.50:
    print("\n🎯 SUCCESS: All drawdowns under 50% (position sizing working!)")
else:
    print(f"\n⚠️  Highest drawdown: {results_df['Max DD'].max():.1%}")
