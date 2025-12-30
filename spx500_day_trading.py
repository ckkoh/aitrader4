"""
S&P 500 Day Trading - Complete Pipeline
Fetch data, train ML model, and backtest intraday strategies
"""

from oanda_integration import OandaConnector
from config import OANDA_CONFIG, STRATEGY_CONFIG
from backtesting_engine import BacktestEngine, BacktestConfig
from strategy_examples import MomentumStrategy, MeanReversionStrategy, BreakoutStrategy
from feature_engineering import FeatureEngineering
from ml_training_pipeline import MLTradingPipeline
import pandas as pd
import logging
import os
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_spx500_data(days=90, granularity='M15'):
    """Fetch S&P 500 historical data"""

    print("=" * 80)
    print(f"FETCHING S&P 500 DATA ({granularity}, {days} days)")
    print("=" * 80)

    oanda = OandaConnector(** OANDA_CONFIG)

    print(f"\nFetching SPX500_USD {granularity} data for last {days} days...")
    df = oanda.fetch_historical_data_range('SPX500_USD', granularity, days=days)

    if df.empty:
        logger.error("Failed to fetch data!")
        return None

    print(f"✅ Fetched {len(df)} candles")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"   Current price: ${df['close'].iloc[-1]:.2f}")

    # Save to CSV
    os.makedirs('data', exist_ok=True)
    filename = f'data/SPX500_USD_{granularity}_{days}days.csv'
    df.to_csv(filename)
    print(f"   Saved to: {filename}")

    return df


def train_spx500_ml_model(df):
    """Train ML model on S&P 500 data"""

    print("\n" + "=" * 80)
    print("TRAINING ML MODEL ON S&P 500 DATA")
    print("=" * 80)

    # Generate features
    print("\n1. Generating features...")
    df_features = FeatureEngineering.build_complete_feature_set(df.copy(), include_volume=True)

    # Create target: 1 if price goes up in next period, 0 otherwise
    df_features['target_binary'] = (df_features['close'].shift(-1) > df_features['close']).astype(int)

    # Remove NaN
    df_features = df_features.dropna()

    print(f"   Features: {len(df_features.columns) - 6} (after removing OHLCV + target)")
    print(f"   Samples: {len(df_features)}")

    # Initialize ML pipeline
    print("\n2. Initializing ML pipeline...")
    pipeline = MLTradingPipeline()

    # Prepare data
    feature_cols = [col for col in df_features.columns
                    if col not in ['open', 'high', 'low', 'close', 'volume', 'target_binary']]

    print(f"   Using {len(feature_cols)} features")

    # Train model
    print("\n3. Training XGBoost model...")
    print("   (This may take a few minutes...)")

    results = pipeline.train_model(
        df_features,
        feature_cols=feature_cols,
        target_col='target_binary',
        model_type='xgboost',
        test_size=0.2,
        hyperparameter_tuning=False,  # Set to True for better results (takes longer)
        cross_validation=True
    )

    # Display results
    print("\n" + "=" * 80)
    print("TRAINING RESULTS")
    print("=" * 80)

    if results and 'test_metrics' in results:
        metrics = results['test_metrics']
        print("\nTest Set Performance:")
        print(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}")
        print(f"  Precision: {metrics.get('precision', 0):.4f}")
        print(f"  Recall:    {metrics.get('recall', 0):.4f}")
        print(f"  F1 Score:  {metrics.get('f1_score', 0):.4f}")

        if 'cv_scores' in results:
            cv_scores = results['cv_scores']
            print("\nCross-Validation (5-fold):")
            print(f"  Accuracy: {cv_scores['accuracy'].mean():.4f} (+/- {cv_scores['accuracy'].std():.4f})")
            print(f"  F1 Score: {cv_scores['f1'].mean():.4f} (+/- {cv_scores['f1'].std():.4f})")

    # Get feature importance
    if hasattr(pipeline.trainer, 'feature_importance') and pipeline.trainer.feature_importance is not None:
        importance = pipeline.trainer.feature_importance
        if hasattr(importance, '__len__') and len(importance) == len(feature_cols):
            feat_imp = pd.DataFrame({
                'feature': feature_cols,
                'importance': importance
            }).sort_values('importance', ascending=False)

            print("\nTop 15 Important Features:")
            for idx, row in feat_imp.head(15).iterrows():
                print(f"  {row['feature']:30s} {row['importance']:.4f}")

    # Save model
    os.makedirs('models', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f'models/spx500_xgboost_{timestamp}.pkl'

    if hasattr(pipeline, 'trainer') and hasattr(pipeline.trainer, 'save_model'):
        pipeline.trainer.save_model(model_path, feature_cols)
        print(f"\n✅ Model saved to: {model_path}")

    return pipeline, feature_cols


def backtest_spx500_strategies(df):
    """Backtest day trading strategies on S&P 500"""

    print("\n" + "=" * 80)
    print("BACKTESTING S&P 500 DAY TRADING STRATEGIES")
    print("=" * 80)

    # Configure backtesting for S&P 500
    config = BacktestConfig(
        initial_capital=25000.0,        # PDT rule requirement
        commission_pct=0.0001,          # 0.01% commission
        slippage_pct=0.0002,            # 0.02% slippage (higher for indices)
        position_size_pct=0.02,         # 2% risk per trade
        max_positions=1,                # Day trading - one position at a time
        max_daily_loss_pct=0.03,        # 3% max daily loss for day trading
        max_drawdown_pct=0.10           # 10% max drawdown
    )

    # Create day trading strategies
    strategies = [
        MomentumStrategy(
            fast_period=STRATEGY_CONFIG['momentum']['fast_period'],
            slow_period=STRATEGY_CONFIG['momentum']['slow_period']
        ),
        MeanReversionStrategy(
            bb_period=STRATEGY_CONFIG['mean_reversion']['bb_period'],
            bb_std=STRATEGY_CONFIG['mean_reversion']['bb_std']
        ),
        BreakoutStrategy(
            lookback_period=STRATEGY_CONFIG['breakout']['lookback_period'],
            atr_period=STRATEGY_CONFIG['breakout']['atr_period']
        )
    ]

    # Run backtests
    engine = BacktestEngine(config)
    results = []

    for strategy in strategies:
        print(f"\nTesting: {strategy.name}")
        print("-" * 80)

        result = engine.run_backtest(strategy, df)
        metrics = result['metrics']

        print(f"  Trades:        {metrics['total_trades']}")
        print(f"  Win Rate:      {metrics['win_rate']:.2%}")
        print(f"  Total Return:  {metrics['total_return_pct']:.2%}")
        print(f"  Sharpe Ratio:  {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:  {metrics['max_drawdown_pct']:.2%}")
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

    # Display comparison
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON")
    print("=" * 80)

    comparison_df = pd.DataFrame(results)
    print(comparison_df.to_string(index=False))

    # Save results
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'results/spx500_backtest_{timestamp}.json'

    import json
    output = {
        'timestamp': timestamp,
        'instrument': 'SPX500_USD',
        'data_points': len(df),
        'date_range': {
            'start': str(df.index[0]),
            'end': str(df.index[-1])
        },
        'strategies': results
    }

    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Results saved to: {results_file}")

    return results


def main():
    """Main pipeline"""

    print("\n" + "=" * 80)
    print("S&P 500 DAY TRADING - COMPLETE PIPELINE")
    print("=" * 80)
    print("\nThis will:")
    print("  1. Fetch 90 days of SPX500_USD M15 data from Oanda")
    print("  2. Train an ML model on S&P 500 patterns")
    print("  3. Backtest day trading strategies")
    print("\n" + "=" * 80)

    # Step 1: Fetch data
    df = fetch_spx500_data(days=90, granularity='M15')

    if df is None or df.empty:
        print("\n❌ Failed to fetch data. Exiting.")
        return

    # Step 2: Train ML model
    try:
        pipeline, feature_cols = train_spx500_ml_model(df)
    except Exception as e:
        logger.error(f"ML training failed: {e}")
        logger.info("Continuing with backtesting...")

    # Step 3: Backtest strategies
    results = backtest_spx500_strategies(df)

    # Summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)

    # Check for profitable strategies
    profitable = [r for r in results if float(r['Return'].rstrip('%')) > 0]

    if profitable:
        print(f"\n✅ {len(profitable)}/{len(results)} strategies are profitable!")
        print("\nProfitable Strategies:")
        for r in profitable:
            print(f"  • {r['Strategy']:20s} Return: {r['Return']:>8s}  Sharpe: {r['Sharpe']}")

        print("\n🎯 NEXT STEPS:")
        print("  1. Paper trade the best strategy for 30 days")
        print("  2. Monitor performance daily")
        print("  3. Adjust parameters if needed")
        print("  4. Consider live trading after successful paper trading")
    else:
        print("\n⚠️  No strategies are profitable on this dataset")
        print("\n💡 SUGGESTIONS:")
        print("  1. Try different timeframes (M5, M30, H1)")
        print("  2. Optimize strategy parameters")
        print("  3. Use the trained ML model")
        print("  4. Test on different market conditions")
        print("  5. Consider combining strategies (ensemble)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
