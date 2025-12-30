"""
Practical Examples - Ready to Run
Copy and execute these examples to learn the system
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# EXAMPLE 1: Simple Backtest
# ============================================================================

def example1_simple_backtest():
    """
    Example 1: Run a simple momentum strategy backtest
    Demonstrates: Basic backtesting workflow
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Simple Momentum Strategy Backtest")
    print("=" * 60 + "\n")

    from backtesting_engine import BacktestEngine, BacktestConfig
    from strategy_examples import MomentumStrategy

    # Generate sample data
    dates = pd.date_range('2023-01-01', periods=3000, freq='1H')
    np.random.seed(42)

    returns = np.random.randn(3000) * 0.0005
    price = 1.1 + np.cumsum(returns)

    df = pd.DataFrame({
        'open': price,
        'high': price + np.abs(np.random.randn(3000)) * 0.0002,
        'low': price - np.abs(np.random.randn(3000)) * 0.0002,
        'close': price,
        'volume': np.random.randint(5000, 15000, 3000)
    }, index=dates)

    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    # Create strategy
    strategy = MomentumStrategy(fast_period=20, slow_period=50)

    # Configure backtest
    config = BacktestConfig(
        initial_capital=10000,
        commission_pct=0.0001,
        slippage_pct=0.0001,
        position_size_pct=0.02
    )

    # Run backtest
    engine = BacktestEngine(config)
    result = engine.run_backtest(strategy, df)

    # Display results
    metrics = result['metrics']
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Total Return: {metrics['total_return_pct']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")

    return result


# ============================================================================
# EXAMPLE 2: Compare Multiple Strategies
# ============================================================================

def example2_compare_strategies():
    """
    Example 2: Compare multiple strategies on same data
    Demonstrates: Strategy comparison and selection
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Compare Multiple Strategies")
    print("=" * 60 + "\n")

    from backtesting_engine import BacktestEngine, BacktestConfig
    from strategy_examples import (MomentumStrategy, MeanReversionStrategy,
                                   BreakoutStrategy)

    # Generate data
    dates = pd.date_range('2023-01-01', periods=5000, freq='1H')
    np.random.seed(42)

    returns = np.random.randn(5000) * 0.0005
    price = 1.1 + np.cumsum(returns)

    df = pd.DataFrame({
        'open': price,
        'high': price + np.abs(np.random.randn(5000)) * 0.0002,
        'low': price - np.abs(np.random.randn(5000)) * 0.0002,
        'close': price,
        'volume': np.random.randint(5000, 15000, 5000)
    }, index=dates)

    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    # Define strategies to test
    strategies = [
        ('Momentum 20/50', MomentumStrategy(20, 50)),
        ('Momentum 10/30', MomentumStrategy(10, 30)),
        ('Mean Reversion', MeanReversionStrategy(20, 2.0)),
        ('Breakout', BreakoutStrategy(20))
    ]

    # Backtest each strategy
    config = BacktestConfig(initial_capital=10000)
    engine = BacktestEngine(config)

    results = []

    for name, strategy in strategies:
        print(f"\nTesting: {name}")
        result = engine.run_backtest(strategy, df)

        metrics = result['metrics']
        results.append({
            'Strategy': name,
            'Trades': metrics['total_trades'],
            'Win Rate': f"{metrics['win_rate']:.1%}",
            'Return': f"{metrics['total_return_pct']:.2f}%",
            'Sharpe': f"{metrics['sharpe_ratio']:.2f}",
            'Max DD': f"{metrics['max_drawdown_pct']:.2f}%"
        })

    # Display comparison
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("STRATEGY COMPARISON")
    print("=" * 60)
    print(results_df.to_string(index=False))

    return results_df


# ============================================================================
# EXAMPLE 3: Feature Engineering
# ============================================================================

def example3_feature_engineering():
    """
    Example 3: Create features from price data
    Demonstrates: Feature engineering workflow
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Feature Engineering")
    print("=" * 60 + "\n")

    from feature_engineering import FeatureEngineering

    # Generate sample data
    dates = pd.date_range('2023-01-01', periods=2000, freq='1H')
    np.random.seed(42)

    returns = np.random.randn(2000) * 0.0005
    price = 1.1 + np.cumsum(returns)

    df = pd.DataFrame({
        'open': price,
        'high': price + np.abs(np.random.randn(2000)) * 0.0002,
        'low': price - np.abs(np.random.randn(2000)) * 0.0002,
        'close': price,
        'volume': np.random.randint(5000, 15000, 2000)
    }, index=dates)

    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    print(f"Original data shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}\n")

    # Build features
    df_features = FeatureEngineering.build_complete_feature_set(df)

    print(f"After feature engineering: {df_features.shape}")
    print(f"Total features created: {len(df_features.columns)}\n")

    # Show sample features
    print("Sample features created:")
    feature_cols = [col for col in df_features.columns
                    if col not in ['open', 'high', 'low', 'close', 'volume']]

    for i, feature in enumerate(feature_cols[:20], 1):
        print(f"  {i:2d}. {feature}")

    print(f"  ... and {len(feature_cols) - 20} more features\n")

    # Show feature statistics
    print("Feature statistics:")
    print(df_features[['rsi_14', 'macd', 'bb_position_20', 'volatility_20']].describe())

    return df_features


# ============================================================================
# EXAMPLE 4: Train ML Model
# ============================================================================

def example4_train_ml_model():
    """
    Example 4: Train a machine learning model
    Demonstrates: ML training pipeline
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Train Machine Learning Model")
    print("=" * 60 + "\n")

    from ml_training_pipeline import MLTradingPipeline

    # Generate realistic data
    dates = pd.date_range('2022-01-01', periods=8000, freq='1H')
    np.random.seed(42)

    # More realistic price simulation
    trend = np.linspace(0, 0.02, 8000)
    returns = np.random.randn(8000) * 0.0005
    for i in range(1, len(returns)):
        returns[i] += -0.1 * returns[i - 1]  # Mean reversion

    price = 1.1 + trend + np.cumsum(returns)

    df = pd.DataFrame({
        'open': price,
        'high': price + np.abs(np.random.randn(8000)) * 0.0002,
        'low': price - np.abs(np.random.randn(8000)) * 0.0002,
        'close': price,
        'volume': np.random.randint(5000, 15000, 8000)
    }, index=dates)

    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    print("Training ML model on 8000 hours of data...\n")

    # Initialize pipeline
    pipeline = MLTradingPipeline()

    # Prepare features
    _df_features = pipeline.load_and_prepare_data(df, include_volume=True)  # noqa: F841

    print(f"Features created: {len(pipeline.feature_cols)}\n")

    # Train model (without hyperparameter tuning for speed)
    results = pipeline.train_model(
        model_type='xgboost',
        hyperparameter_tuning=False,
        cross_validation=True
    )

    # Display results
    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)

    test_metrics = results['test_metrics']
    print("\nTest Set Performance:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1 Score: {test_metrics['f1_score']:.4f}")

    if 'cv_results' in results:
        cv_results = results['cv_results']
        print("\nCross-Validation Results:")
        print(f"  Accuracy: {cv_results['accuracy_mean']:.4f} (+/- {cv_results['accuracy_std']:.4f})")
        print(f"  F1 Score: {cv_results['f1_mean']:.4f} (+/- {cv_results['f1_std']:.4f})")

    print("\nTop 10 Important Features:")
    feature_importance = results['feature_importance']
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:<25} {row['importance']:.4f}")

    print(f"\nModel saved: {results['model_filename']}")

    return results


# ============================================================================
# EXAMPLE 5: Walk-Forward Analysis
# ============================================================================

def example5_walk_forward_analysis():
    """
    Example 5: Perform walk-forward validation
    Demonstrates: Robust strategy validation
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Walk-Forward Analysis")
    print("=" * 60 + "\n")

    from backtesting_engine import BacktestEngine, BacktestConfig
    from strategy_examples import MomentumStrategy

    # Generate longer data for walk-forward
    dates = pd.date_range('2022-01-01', periods=12000, freq='1H')
    np.random.seed(42)

    returns = np.random.randn(12000) * 0.0005
    price = 1.1 + np.cumsum(returns)

    df = pd.DataFrame({
        'open': price,
        'high': price + np.abs(np.random.randn(12000)) * 0.0002,
        'low': price - np.abs(np.random.randn(12000)) * 0.0002,
        'close': price,
        'volume': np.random.randint(5000, 15000, 12000)
    }, index=dates)

    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    # Create strategy
    strategy = MomentumStrategy(20, 50)

    # Configure backtest
    config = BacktestConfig(initial_capital=10000)
    engine = BacktestEngine(config)

    # Run walk-forward analysis
    print("Running walk-forward analysis...")
    print("(This validates strategy across multiple time periods)\n")

    wf_results = engine.walk_forward_analysis(
        strategy=strategy,
        data=df,
        train_period_days=180,  # 6 months training
        test_period_days=60,    # 2 months testing
        step_days=60            # Move forward 2 months
    )

    # Summarize results
    print("\n" + "=" * 60)
    print("WALK-FORWARD RESULTS SUMMARY")
    print("=" * 60)

    wf_summary = []
    for i, result in enumerate(wf_results, 1):
        metrics = result['metrics']
        wf_summary.append({
            'Period': i,
            'Trades': metrics['total_trades'],
            'Win Rate': f"{metrics['win_rate']:.1%}",
            'Return': f"{metrics['total_return_pct']:.2f}%",
            'Sharpe': f"{metrics['sharpe_ratio']:.2f}"
        })

    wf_df = pd.DataFrame(wf_summary)
    print(wf_df.to_string(index=False))

    # Calculate consistency
    sharpe_ratios = [r['metrics']['sharpe_ratio'] for r in wf_results]
    print(f"\nAverage Sharpe: {np.mean(sharpe_ratios):.2f}")
    print(f"Sharpe Std Dev: {np.std(sharpe_ratios):.2f}")
    print(f"Profitable Periods: {sum(1 for s in sharpe_ratios if s > 0)}/{len(sharpe_ratios)}")

    return wf_results


# ============================================================================
# EXAMPLE 6: Deploy to Dashboard
# ============================================================================

def example6_deploy_to_dashboard():
    """
    Example 6: Add backtest results to monitoring dashboard
    Demonstrates: Integration with dashboard
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Deploy Results to Dashboard")
    print("=" * 60 + "\n")

    from backtesting_engine import BacktestEngine, BacktestConfig
    from strategy_examples import MomentumStrategy
    from trading_dashboard_main import DatabaseManager

    # Run a quick backtest
    dates = pd.date_range('2023-01-01', periods=2000, freq='1H')
    np.random.seed(42)

    returns = np.random.randn(2000) * 0.0005
    price = 1.1 + np.cumsum(returns)

    df = pd.DataFrame({
        'open': price,
        'high': price + np.abs(np.random.randn(2000)) * 0.0002,
        'low': price - np.abs(np.random.randn(2000)) * 0.0002,
        'close': price,
        'volume': np.random.randint(5000, 15000, 2000)
    }, index=dates)

    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    strategy = MomentumStrategy(20, 50)
    config = BacktestConfig(initial_capital=10000)
    engine = BacktestEngine(config)

    print("Running backtest...")
    result = engine.run_backtest(strategy, df)

    # Add to dashboard
    print("Adding trades to dashboard database...")
    db = DatabaseManager()

    trades_added = 0
    for trade in result['trades']:
        db.add_trade({
            'trade_id': f"EXAMPLE_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}",
            'instrument': trade.instrument,
            'direction': trade.direction,
            'entry_time': trade.entry_time.isoformat(),
            'exit_time': trade.exit_time.isoformat(),
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'size': trade.size,
            'pnl': trade.pnl,
            'pnl_percent': trade.pnl_percent,
            'commission': trade.commission,
            'slippage': trade.slippage,
            'strategy': 'Example_Momentum_20_50',
            'status': 'closed'
        })
        trades_added += 1

    print(f"\n✅ Successfully added {trades_added} trades to dashboard!")
    print("\nTo view in dashboard, run:")
    print("  streamlit run trading_dashboard_main.py")

    return trades_added


# ============================================================================
# RUN ALL EXAMPLES
# ============================================================================

def run_all_examples():
    """Run all examples in sequence"""
    print("\n" + "=" * 60)
    print("RUNNING ALL EXAMPLES")
    print("=" * 60)

    try:
        example1_simple_backtest()
        input("\nPress Enter to continue to Example 2...")

        example2_compare_strategies()
        input("\nPress Enter to continue to Example 3...")

        example3_feature_engineering()
        input("\nPress Enter to continue to Example 4...")

        example4_train_ml_model()
        input("\nPress Enter to continue to Example 5...")

        example5_walk_forward_analysis()
        input("\nPress Enter to continue to Example 6...")

        example6_deploy_to_dashboard()

        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED!")
        print("=" * 60)
        print("\nYou now understand:")
        print("  ✓ Backtesting strategies")
        print("  ✓ Comparing multiple strategies")
        print("  ✓ Feature engineering")
        print("  ✓ Training ML models")
        print("  ✓ Walk-forward validation")
        print("  ✓ Dashboard integration")
        print("\nNext: Build your own strategy and test it!")

    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user.")
    except Exception as e:
        print(f"\n\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run Trading System Examples')
    parser.add_argument('--example', type=int, choices=[1, 2, 3, 4, 5, 6],
                        help='Run specific example (1-6)')
    parser.add_argument('--all', action='store_true',
                        help='Run all examples in sequence')

    args = parser.parse_args()

    if args.all:
        run_all_examples()
    elif args.example:
        if args.example == 1:
            example1_simple_backtest()
        elif args.example == 2:
            example2_compare_strategies()
        elif args.example == 3:
            example3_feature_engineering()
        elif args.example == 4:
            example4_train_ml_model()
        elif args.example == 5:
            example5_walk_forward_analysis()
        elif args.example == 6:
            example6_deploy_to_dashboard()
    else:
        # Default: run example 1
        print("Running Example 1 (use --help for more options)")
        example1_simple_backtest()
