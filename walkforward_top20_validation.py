#!/usr/bin/env python3
"""
Walk-Forward Validation for Top 20 ML Model
Tests model robustness across multiple time periods
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List

from feature_engineering import FeatureEngineering
from ml_training_pipeline import MLModelTrainer
from backtesting_engine import BacktestEngine, BacktestConfig
from strategy_examples import MLStrategy


class WalkForwardMLValidator:
    """
    Walk-forward validation for Top 20 ML model
    """

    def __init__(self, top_features: List[str], output_dir: str = 'walkforward_ml_results'):
        self.top_features = top_features
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Backtest configuration
        self.backtest_config = BacktestConfig(
            initial_capital=10000.0,
            commission_pct=0.001,
            slippage_pct=0.0002,
            position_size_pct=0.02,
            max_position_value_pct=0.02,
            max_positions=1,
            max_daily_loss_pct=0.03,
            max_drawdown_pct=0.15,
            position_sizing_method='volatility',
        )

    def load_data(self) -> pd.DataFrame:
        """Load S&P 500 historical data"""
        print("\n1. Loading S&P 500 data...")

        data_file = 'sp500_historical_data.csv'

        if not Path(data_file).exists():
            raise FileNotFoundError(f"{data_file} not found!")

        df = pd.read_csv(data_file, index_col='Date', parse_dates=True)

        # Use data from 2020 onwards for walk-forward
        df = df.loc['2020-01-01':]

        print(f"   ✅ Loaded {len(df)} days from {data_file}")
        print(f"   Date range: {df.index[0]} to {df.index[-1]}")

        return df

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all features and select top 20"""
        print("\n2. Generating features...")

        df_features = FeatureEngineering.build_complete_feature_set(
            df,
            include_volume=True
        )

        df_features = df_features.dropna()

        print(f"   ✅ Generated {len(df_features.columns)} total features")
        print(f"   ✅ Using top 20 selected features")
        print(f"   Clean data points: {len(df_features)}")

        return df_features

    def create_splits(self, df: pd.DataFrame,
                     train_months: int = 12,
                     test_months: int = 3,
                     step_months: int = 3) -> List[Dict]:
        """
        Create walk-forward train/test splits

        Args:
            df: DataFrame with features
            train_months: Training period in months (default: 12)
            test_months: Testing period in months (default: 3)
            step_months: Step size in months (default: 3)

        Returns:
            List of split definitions
        """
        print(f"\n3. Creating walk-forward splits...")
        print(f"   Train: {train_months} months, Test: {test_months} months, Step: {step_months} months")

        # Approximate days per month
        days_per_month = 21  # Trading days

        train_days = train_months * days_per_month
        test_days = test_months * days_per_month
        step_days = step_months * days_per_month

        splits = []
        start_idx = 0

        while start_idx + train_days + test_days <= len(df):
            train_start = start_idx
            train_end = start_idx + train_days
            test_start = train_end
            test_end = test_start + test_days

            # Get data
            train_data = df.iloc[train_start:train_end]
            test_data = df.iloc[test_start:test_end]

            splits.append({
                'split_num': len(splits) + 1,
                'train_data': train_data,
                'test_data': test_data,
                'train_start': train_data.index[0],
                'train_end': train_data.index[-1],
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1],
            })

            start_idx += step_days

        print(f"   ✅ Created {len(splits)} walk-forward splits")

        return splits

    def train_and_test_split(self, split: Dict, split_num: int) -> Dict:
        """
        Train model on train period, test on test period

        Args:
            split: Split definition with train/test data
            split_num: Split number

        Returns:
            Dict with results
        """
        print(f"\n{'='*80}")
        print(f"Split {split_num}")
        print(f"Train: {split['train_start']} to {split['train_end']} ({len(split['train_data'])} days)")
        print(f"Test:  {split['test_start']} to {split['test_end']} ({len(split['test_data'])} days)")
        print(f"{'='*80}")

        train_data = split['train_data']
        test_data = split['test_data']

        # Prepare training data
        df_train = train_data.copy()
        df_train['target'] = (df_train['close'].shift(-1) > df_train['close']).astype(int)
        df_train = df_train.dropna()

        X_train = df_train[self.top_features].values
        y_train = df_train['target'].values

        print(f"\n   [1/4] Training model...")
        print(f"         Features: {len(self.top_features)}")
        print(f"         Samples: {len(X_train)}")

        # Train model
        trainer = MLModelTrainer(model_type='xgboost', task='classification')
        train_results = trainer.train(X_train, y_train, hyperparameter_tuning=True)

        print(f"         Train accuracy: {train_results.get('train_accuracy', 0):.2f}%")

        # Save model
        model_path = self.output_dir / f'model_split_{split_num}.pkl'
        trainer.save_model(str(model_path))

        # Create strategy
        print(f"\n   [2/4] Creating ML strategy...")
        strategy = MLStrategy(
            model_path=str(model_path),
            feature_cols=self.top_features,
            confidence_threshold=0.55  # Middle threshold
        )

        # Combine train + test for indicators
        combined_data = pd.concat([train_data, test_data])

        # Run backtest
        print(f"\n   [3/4] Running backtest...")
        engine = BacktestEngine(self.backtest_config)
        test_start_date = test_data.index[0]

        backtest_results = engine.run_backtest(
            strategy,
            combined_data,
            trading_start_date=test_start_date
        )

        metrics = backtest_results['metrics']

        print(f"\n   [4/4] Results:")
        print(f"         Trades: {metrics.get('total_trades', 0)}")
        print(f"         Win Rate: {metrics.get('win_rate', 0):.1%}")
        print(f"         Return: {metrics.get('total_return_pct', 0):+.2%}")
        print(f"         Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"         Max DD: {metrics.get('max_drawdown_pct', 0):.2%}")

        return {
            'split_num': split_num,
            'train_start': str(split['train_start']),
            'train_end': str(split['train_end']),
            'test_start': str(split['test_start']),
            'test_end': str(split['test_end']),
            'train_samples': len(X_train),
            'train_accuracy': train_results.get('train_accuracy', 0),
            'trades': metrics.get('total_trades', 0),
            'win_rate': metrics.get('win_rate', 0),
            'total_return': metrics.get('total_return_pct', 0),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
            'max_drawdown': metrics.get('max_drawdown_pct', 0),
            'profit_factor': metrics.get('profit_factor', 0),
            'avg_win': metrics.get('avg_win', 0),
            'avg_loss': metrics.get('avg_loss', 0)
        }

    def run_validation(self) -> List[Dict]:
        """Run complete walk-forward validation"""
        print("="*80)
        print("WALK-FORWARD VALIDATION - TOP 20 ML MODEL")
        print("="*80)

        # Load and prepare data
        df = self.load_data()
        df_features = self.prepare_features(df)

        # Create splits
        splits = self.create_splits(df_features,
                                    train_months=12,
                                    test_months=3,
                                    step_months=3)

        # Run validation on each split
        print(f"\n4. Running validation on {len(splits)} splits...")
        print("="*80)

        results = []
        for split in splits:
            try:
                result = self.train_and_test_split(split, split['split_num'])
                results.append(result)
            except Exception as e:
                print(f"\n   ❌ Error on split {split['split_num']}: {e}")
                import traceback
                traceback.print_exc()
                continue

        return results

    def analyze_results(self, results: List[Dict]):
        """Analyze and display validation results"""
        print("\n" + "="*80)
        print("WALK-FORWARD VALIDATION RESULTS")
        print("="*80)

        if not results:
            print("\n❌ No results to analyze")
            return

        # Create DataFrame
        results_df = pd.DataFrame(results)

        # Display all splits
        print("\n📊 Individual Split Results:")
        print("="*80)

        display_cols = ['split_num', 'trades', 'win_rate', 'total_return',
                       'sharpe_ratio', 'max_drawdown']

        for _, row in results_df.iterrows():
            print(f"\nSplit {row['split_num']}: {row['test_start']} to {row['test_end']}")
            print(f"  Trades: {row['trades']}")
            print(f"  Win Rate: {row['win_rate']:.1%}")
            print(f"  Return: {row['total_return']:+.2%}")
            print(f"  Sharpe: {row['sharpe_ratio']:.2f}")
            print(f"  Max DD: {row['max_drawdown']:.2%}")

        # Aggregate statistics
        print("\n" + "="*80)
        print("AGGREGATE STATISTICS")
        print("="*80)

        total_trades = results_df['trades'].sum()
        avg_win_rate = results_df['win_rate'].mean()
        avg_return = results_df['total_return'].mean()
        median_return = results_df['total_return'].median()
        std_return = results_df['total_return'].std()
        avg_sharpe = results_df['sharpe_ratio'].mean()
        avg_max_dd = results_df['max_drawdown'].mean()
        max_max_dd = results_df['max_drawdown'].max()

        positive_periods = (results_df['total_return'] > 0).sum()
        positive_pct = (positive_periods / len(results_df)) * 100

        sharpe_above_1 = (results_df['sharpe_ratio'] > 1.0).sum()
        sharpe_above_1_pct = (sharpe_above_1 / len(results_df)) * 100

        print(f"\nTotal Splits: {len(results_df)}")
        print(f"Total Trades: {total_trades}")
        print(f"Avg Trades/Split: {total_trades / len(results_df):.1f}")
        print(f"\nAvg Win Rate: {avg_win_rate:.1%}")
        print(f"Avg Return: {avg_return:+.2%}")
        print(f"Median Return: {median_return:+.2%}")
        print(f"Std Dev Return: {std_return:.2%}")
        print(f"Avg Sharpe Ratio: {avg_sharpe:.2f}")
        print(f"Avg Max Drawdown: {avg_max_dd:.2%}")
        print(f"Max Max Drawdown: {max_max_dd:.2%}")
        print(f"\nPositive Return Periods: {positive_periods}/{len(results_df)} ({positive_pct:.1f}%)")
        print(f"Sharpe > 1.0 Periods: {sharpe_above_1}/{len(results_df)} ({sharpe_above_1_pct:.1f}%)")

        # Validation criteria
        print("\n" + "="*80)
        print("VALIDATION CRITERIA")
        print("="*80)

        criteria = [
            ("Minimum 3 walk-forward periods", len(results_df) >= 3),
            ("Avg Win Rate > 50%", avg_win_rate > 0.50),
            ("Avg Return > 0%", avg_return > 0),
            ("Positive returns in >50% of periods", positive_pct > 50),
            ("Avg Max Drawdown < 15%", avg_max_dd < 0.15),
            ("Max Max Drawdown < 20%", max_max_dd < 0.20),
        ]

        passed = 0
        for criterion, result in criteria:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status}: {criterion}")
            if result:
                passed += 1

        print(f"\nOverall: {passed}/{len(criteria)} criteria passed ({passed/len(criteria)*100:.0f}%)")

        if passed == len(criteria):
            print("\n🎉 MODEL VALIDATION PASSED! Ready for paper trading.")
        elif passed >= len(criteria) * 0.7:
            print("\n⚠️  Model shows promise but needs refinement.")
        else:
            print("\n❌ Model needs significant improvement.")

        # Save results
        results_df.to_csv(self.output_dir / 'validation_results.csv', index=False)

        summary = {
            'total_splits': len(results_df),
            'total_trades': int(total_trades),
            'avg_win_rate': float(avg_win_rate),
            'avg_return': float(avg_return),
            'median_return': float(median_return),
            'std_return': float(std_return),
            'avg_sharpe': float(avg_sharpe),
            'avg_max_dd': float(avg_max_dd),
            'max_max_dd': float(max_max_dd),
            'positive_periods_pct': float(positive_pct),
            'sharpe_above_1_pct': float(sharpe_above_1_pct),
            'criteria_passed': passed,
            'criteria_total': len(criteria),
            'validation_date': datetime.now().isoformat()
        }

        with open(self.output_dir / 'validation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n✅ Results saved to {self.output_dir}/")
        print("   - validation_results.csv")
        print("   - validation_summary.json")


def main():
    """Main execution"""

    # Top 20 features from Day 2
    TOP_20_FEATURES = [
        'bullish_engulfing',
        'stoch_d_3',
        'week_of_year',
        'atr_14',
        'regime',
        'roc_20',
        'obv',
        'parkinson_vol_10',
        'volatility_200d',
        'momentum_5',
        'macd_signal',
        'adx_14',
        'month_sin',
        'hl_ratio',
        'rsi_14',
        'stoch_k_14',
        'bb_position_20',
        'momentum_oscillator',
        'pvt',
        'price_acceleration'
    ]

    print("="*80)
    print("TOP 20 ML MODEL - WALK-FORWARD VALIDATION")
    print("="*80)
    print(f"\nUsing {len(TOP_20_FEATURES)} selected features")
    print("Testing robustness across multiple time periods")

    try:
        validator = WalkForwardMLValidator(TOP_20_FEATURES)
        results = validator.run_validation()
        validator.analyze_results(results)

        print("\n" + "="*80)
        print("✅ VALIDATION COMPLETE")
        print("="*80)

    except KeyboardInterrupt:
        print("\n\n⚠️  Execution interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
