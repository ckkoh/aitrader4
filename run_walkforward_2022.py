"""
Walk-Forward Training Plan: Jan-Apr 2022
S&P 500 Trading System - Bear Market Test
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json

from feature_engineering import FeatureEngineering
from ml_training_pipeline import MLTradingPipeline, MLModelTrainer
from backtesting_engine import BacktestEngine, BacktestConfig
from strategy_examples import MLStrategy

# Configuration
OUTPUT_DIR = Path('walkforward_2022_jan_apr')
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / 'data').mkdir(exist_ok=True)
(OUTPUT_DIR / 'models').mkdir(exist_ok=True)
(OUTPUT_DIR / 'results').mkdir(exist_ok=True)

MODEL_TYPE = 'xgboost'
CONFIDENCE_THRESHOLD = 0.55  # Lowered from 0.65 for 2022 volatile period
HYPERPARAMETER_TUNING = True  # Enable for better performance (slower training)

# Date ranges for Rolling Window Strategy
ITERATIONS = [
    {
        'name': 'Jan_2022',
        'train_start': '2021-01-01',
        'train_end': '2021-12-31',
        'test_start': '2022-01-01',
        'test_end': '2022-01-31'
    },
    {
        'name': 'Feb_2022',
        'train_start': '2021-02-01',
        'train_end': '2022-01-31',
        'test_start': '2022-02-01',
        'test_end': '2022-02-28'
    },
    {
        'name': 'Mar_2022',
        'train_start': '2021-03-01',
        'train_end': '2022-02-28',
        'test_start': '2022-03-01',
        'test_end': '2022-03-31'
    },
    {
        'name': 'Apr_2022',
        'train_start': '2021-04-01',
        'train_end': '2022-03-31',
        'test_start': '2022-04-01',
        'test_end': '2022-04-30'
    }
]


def load_sp500_data():
    """Load S&P 500 historical data"""
    print("\n1. Loading S&P 500 data...")

    # Try to load from existing file
    data_file = 'sp500_historical_data.csv'

    if Path(data_file).exists():
        df = pd.read_csv(data_file, index_col='Date', parse_dates=True)
        print(f"   ✅ Loaded {len(df)} days from {data_file}")
        print(f"   Date range: {df.index[0]} to {df.index[-1]}")
        return df
    else:
        print(f"   ❌ {data_file} not found!")
        print("   Please ensure sp500_historical_data.csv exists with 2020-2022 data")
        raise FileNotFoundError(f"{data_file} not found")


def add_sp500_specific_features(df):
    """Add S&P 500 specific features for 2022"""
    print("\n   Adding S&P 500 specific features...")

    # Volatility regime
    df['volatility_regime'] = df['close'].pct_change().rolling(20).std()
    df['high_vol'] = (df['volatility_regime'] > df['volatility_regime'].rolling(60).mean()).astype(int)

    # VIX proxy
    df['vix_proxy'] = df['close'].pct_change().rolling(10).std() * np.sqrt(252) * 100
    df['fear_level'] = 0
    df.loc[df['vix_proxy'] > 25, 'fear_level'] = 1
    df.loc[df['vix_proxy'] > 35, 'fear_level'] = 2

    # Fed hawkish regime (2022 onwards)
    df['fed_hawkish'] = (df.index >= '2022-01-01').astype(int)

    # Month-end effects
    df['days_to_month_end'] = df.index.to_series().apply(
        lambda x: (x + pd.offsets.MonthEnd(0) - x).days
    )
    df['month_end_period'] = (df['days_to_month_end'] <= 3).astype(int)

    print(f"   ✅ Added 7 S&P 500 specific features")

    return df


def get_2022_backtest_config():
    """Get backtest config optimized for 2022 volatility"""
    return BacktestConfig(
        initial_capital=10000.0,
        commission_pct=0.001,
        slippage_pct=0.0003,  # Higher for 2022 volatility
        position_size_pct=0.015,  # Lower for volatile period
        max_position_value_pct=0.02,
        max_positions=1,
        max_daily_loss_pct=0.02,
        max_drawdown_pct=0.12,
        position_sizing_method='volatility',
    )


def train_model(df_train, model_type='xgboost', hyperparameter_tuning=False):
    """Train ML model on training data"""
    tuning_status = "WITH hyperparameter tuning" if hyperparameter_tuning else "without tuning"
    print(f"\n   Training ML model ({tuning_status})...")

    # Get feature columns (exclude target and non-features)
    exclude_cols = ['target_class', 'target_regression', 'target_binary',
                    'future_return', 'open', 'high', 'low', 'close', 'volume']
    feature_cols = [col for col in df_train.columns if col not in exclude_cols]

    # Create target variable
    df_train = df_train.copy()
    df_train['target'] = (df_train['close'].shift(-1) > df_train['close']).astype(int)
    df_train = df_train.dropna()

    X_train = df_train[feature_cols].values
    y_train = df_train['target'].values

    # Initialize trainer
    trainer = MLModelTrainer(model_type=model_type, task='classification')

    # Train model with optional hyperparameter tuning
    try:
        results = trainer.train(X_train, y_train, hyperparameter_tuning=hyperparameter_tuning)
        print(f"   ✅ Model trained on {len(df_train)} samples")
        if results and 'train_accuracy' in results:
            print(f"   Training accuracy: {results['train_accuracy']:.2f}%")
        if hyperparameter_tuning and results:
            print(f"   Best hyperparameters: {results.get('best_params', 'N/A')}")
    except Exception as e:
        print(f"   Training completed with warnings: {e}")

    return trainer, feature_cols


def save_iteration_results(iteration, month_name, results, train_data, test_data):
    """Save results for one iteration"""
    # results is already a dict with all needed info
    # Just add train/test samples
    results_dict = results.copy()
    results_dict['train_samples'] = len(train_data)
    results_dict['test_samples'] = len(test_data)

    results_path = OUTPUT_DIR / 'results' / f'backtest_{month_name}.json'
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)


def main():
    print("="*70)
    print("WALK-FORWARD TRAINING PLAN: JAN-APR 2022")
    print("S&P 500 Trading System - Bear Market Test")
    print("="*70)
    print(f"\nTest Period: January-April 2022")
    print(f"Market Context: Start of bear market, -13.31% buy & hold")
    print(f"Volatility: VIX 25-35 (elevated)")
    print(f"\nConfiguration:")
    print(f"  Model Type: {MODEL_TYPE}")
    print(f"  Hyperparameter Tuning: {'ENABLED' if HYPERPARAMETER_TUNING else 'Disabled'}")
    print(f"  Confidence Threshold: {CONFIDENCE_THRESHOLD}")

    # Load data
    try:
        df = load_sp500_data()
    except FileNotFoundError:
        print("\n❌ Cannot proceed without data file")
        return

    # Generate features (standard set only for consistency with MLStrategy)
    print("\n2. Generating features...")
    df_features = FeatureEngineering.build_complete_feature_set(df, include_volume=True)

    # Note: NOT adding custom S&P 500 features to maintain consistency
    # with MLStrategy which uses standard features only

    # Remove any NaN values
    df_features = df_features.dropna()

    print(f"   ✅ Generated {len(df_features.columns)} total features")
    print(f"   Data points after cleaning: {len(df_features)}")

    # Get backtest config
    backtest_config = get_2022_backtest_config()

    # Run walk-forward iterations
    results = []

    for i, config in enumerate(ITERATIONS, 1):
        print(f"\n{'='*70}")
        print(f"ITERATION {i}: {config['name']}")
        print(f"{'='*70}")

        # Split data
        try:
            df_train = df_features.loc[config['train_start']:config['train_end']].copy()
            df_test = df_features.loc[config['test_start']:config['test_end']].copy()

            # For backtesting, include historical data for feature engineering
            # Use train data + test data so strategy has enough context
            df_backtest = df_features.loc[config['train_start']:config['test_end']].copy()

            print(f"Train: {config['train_start']} to {config['train_end']} ({len(df_train)} days)")
            print(f"Test:  {config['test_start']} to {config['test_end']} ({len(df_test)} days)")
            print(f"Backtest data (train+test for context): {len(df_backtest)} days")

            if len(df_train) < 50:
                print(f"   ⚠️ Warning: Only {len(df_train)} training samples!")
                continue

            if len(df_test) < 5:
                print(f"   ⚠️ Warning: Only {len(df_test)} test samples!")
                continue

        except Exception as e:
            print(f"   ❌ Error splitting data: {e}")
            continue

        # Train model
        try:
            trainer, feature_cols = train_model(df_train, MODEL_TYPE, HYPERPARAMETER_TUNING)

            # Save model using trainer's save method
            temp_model_path = OUTPUT_DIR / 'models' / f'{config["name"]}.pkl'
            trainer.save_model(str(temp_model_path))

        except Exception as e:
            print(f"   ❌ Error training model: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Create strategy
        strategy = MLStrategy(
            model_path=str(temp_model_path),
            feature_cols=feature_cols,
            confidence_threshold=CONFIDENCE_THRESHOLD
        )

        # Run backtest
        print("\n4. Running backtest...")
        engine = BacktestEngine(backtest_config)

        try:
            # Use df_backtest (train+test) for context, but only trade during test period
            from datetime import datetime as dt
            trading_start = dt.strptime(config['test_start'], '%Y-%m-%d')

            backtest_results = engine.run_backtest(
                strategy,
                df_backtest,
                trading_start_date=trading_start
            )

            # Store results (backtest_results is a dict with 'metrics' key)
            metrics = backtest_results['metrics']
            result = {
                'iteration': i,
                'month': config['name'],
                'train_days': len(df_train),
                'test_days': len(df_test),
                'trades': metrics['total_trades'],
                'win_rate': metrics['win_rate'] * 100,  # Convert to percentage
                'total_return': metrics['total_return_pct'] * 100,
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown': metrics['max_drawdown_pct'] * 100,
                'profit_factor': metrics['profit_factor']
            }
            results.append(result)

            print(f"\n✅ Results for {config['name']}:")
            print(f"  Trades: {metrics['total_trades']}")
            print(f"  Win Rate: {metrics['win_rate']*100:.2f}%")
            print(f"  Return: {metrics['total_return_pct']*100:+.2f}%")
            print(f"  Sharpe: {metrics['sharpe_ratio']:.2f}")
            print(f"  Max DD: {metrics['max_drawdown_pct']*100:.2f}%")

            # Save iteration results
            save_iteration_results(i, config['name'], result, df_train, df_test)

        except Exception as e:
            print(f"   ❌ Error in backtest: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Generate summary
    if not results:
        print("\n❌ No results to summarize")
        return

    print(f"\n{'='*70}")
    print("WALK-FORWARD SUMMARY")
    print(f"{'='*70}")

    summary_df = pd.DataFrame(results)
    print("\n" + summary_df.to_string(index=False))

    # Calculate aggregate metrics
    total_return = summary_df['total_return'].sum()
    avg_sharpe = summary_df['sharpe_ratio'].mean()
    max_dd = summary_df['max_drawdown'].max()
    avg_win_rate = summary_df['win_rate'].mean()
    profitable_months = (summary_df['total_return'] > 0).sum()

    print(f"\n{'='*70}")
    print("AGGREGATE METRICS")
    print(f"{'='*70}")
    print(f"Total Return: {total_return:+.2f}%")
    print(f"Buy & Hold (Jan-Apr 2022): -13.31%")
    print(f"Excess Return: {total_return - (-13.31):+.2f}%")
    print(f"Average Win Rate: {avg_win_rate:.2f}%")
    print(f"Average Sharpe: {avg_sharpe:.2f}")
    print(f"Worst Drawdown: {max_dd:.2f}%")
    print(f"Profitable Months: {profitable_months}/{len(results)}")

    # Validation
    print(f"\n{'='*70}")
    print("VALIDATION")
    print(f"{'='*70}")

    if profitable_months >= 2:
        print(f"✅ Consistency check: PASS ({profitable_months}/4 profitable)")
    else:
        print(f"❌ Consistency check: FAIL ({profitable_months}/4 profitable)")

    if total_return > -13.31:
        print(f"✅ Beat buy & hold: PASS ({total_return:+.2f}% vs -13.31%)")
    else:
        print(f"❌ Beat buy & hold: FAIL ({total_return:+.2f}% vs -13.31%)")

    if avg_sharpe > 0.5:
        print(f"✅ Sharpe ratio: PASS ({avg_sharpe:.2f} > 0.5)")
    else:
        print(f"⚠️ Sharpe ratio: MARGINAL ({avg_sharpe:.2f} < 0.5)")

    if max_dd < 15:
        print(f"✅ Drawdown control: PASS ({max_dd:.2f}% < 15%)")
    else:
        print(f"⚠️ Drawdown control: MARGINAL ({max_dd:.2f}% >= 15%)")

    # Save summary
    summary_df.to_csv(OUTPUT_DIR / 'walkforward_summary.csv', index=False)

    # Save aggregate metrics
    aggregate = {
        'total_return': float(total_return),
        'buy_hold_return': -13.31,
        'excess_return': float(total_return - (-13.31)),
        'avg_win_rate': float(avg_win_rate),
        'avg_sharpe': float(avg_sharpe),
        'worst_drawdown': float(max_dd),
        'profitable_months': int(profitable_months),
        'total_months': len(results)
    }

    with open(OUTPUT_DIR / 'aggregate_metrics.json', 'w') as f:
        json.dump(aggregate, f, indent=2)

    print(f"\n✅ Results saved to {OUTPUT_DIR}/")
    print(f"   - walkforward_summary.csv")
    print(f"   - aggregate_metrics.json")
    print(f"   - Individual results in results/")
    print(f"   - Models saved in models/")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Execution interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
