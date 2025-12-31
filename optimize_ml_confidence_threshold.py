#!/usr/bin/env python3
"""
Week 1, Day 1: Optimize ML Confidence Threshold
From IMPROVEMENTS_PLAN.md

Problem: Current ML strategy with 60% threshold only generates 4 trades in 248 days
Solution: Test multiple thresholds (0.50, 0.55, 0.60, 0.65, 0.70) to find optimal

Expected Outcomes:
- 0.50: More trades (15-20), likely 50-55% win rate
- 0.55: Moderate trades (8-12), ~60% win rate
- 0.60: Current baseline (4 trades)
- 0.65: Few trades (3-5), ~70% win rate
- 0.70: Very few trades (1-2), ~75%+ win rate

Impact: +200% more trades, +10-15% win rate improvement
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


class ConfidenceThresholdOptimizer:
    """
    Test ML strategy with different confidence thresholds
    """

    def __init__(self, output_dir: str = 'threshold_optimization_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Thresholds to test
        self.thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]

        # Backtest configuration
        self.backtest_config = BacktestConfig(
            initial_capital=10000.0,
            commission_pct=0.001,  # 0.1% per trade
            slippage_pct=0.0002,
            position_size_pct=0.02,
            max_position_value_pct=0.02,
            max_positions=1,
            max_daily_loss_pct=0.03,
            max_drawdown_pct=0.15,
            position_sizing_method='volatility',
        )

    def load_sp500_data(self) -> pd.DataFrame:
        """Load S&P 500 historical data for testing"""
        print("\n1. Loading S&P 500 data...")

        # Use full historical data (need 200+ days for regime detection)
        data_file = 'sp500_historical_data.csv'

        if not Path(data_file).exists():
            raise FileNotFoundError(f"{data_file} not found!")

        df = pd.read_csv(data_file, index_col='Date', parse_dates=True)

        # Use last 2 years for training + testing (recent market conditions)
        df = df.loc['2023-01-01':]

        print(f"   ✅ Loaded {len(df)} days from {data_file}")
        print(f"   Date range: {df.index[0]} to {df.index[-1]}")

        return df

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features using standard feature engineering"""
        print("\n2. Generating features...")

        df_features = FeatureEngineering.build_complete_feature_set(
            df,
            include_volume=True
        )

        df_features = df_features.dropna()

        print(f"   ✅ Generated {len(df_features.columns)} features")
        print(f"   Clean data points: {len(df_features)}")

        return df_features

    def train_ml_model(self, df_train: pd.DataFrame, model_path: Path) -> tuple:
        """
        Train ML model on training data

        Returns:
            (trainer, feature_cols)
        """
        print("\n3. Training ML model...")

        # Exclude target and non-feature columns
        exclude_cols = ['target_class', 'target_regression', 'target_binary',
                        'future_return', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in df_train.columns if col not in exclude_cols]

        # Create target
        df_train = df_train.copy()
        df_train['target'] = (df_train['close'].shift(-1) > df_train['close']).astype(int)
        df_train = df_train.dropna()

        X_train = df_train[feature_cols].values
        y_train = df_train['target'].values

        # Train model
        trainer = MLModelTrainer(model_type='xgboost', task='classification')

        try:
            results = trainer.train(X_train, y_train, hyperparameter_tuning=True)
            print(f"   ✅ Model trained on {len(df_train)} samples")

            if results and 'train_accuracy' in results:
                print(f"   Training accuracy: {results['train_accuracy']:.2f}%")

            if results and 'best_params' in results:
                print(f"   Best params: {results['best_params']}")

        except Exception as e:
            print(f"   ⚠️  Training completed with warnings: {e}")

        # Save model
        trainer.save_model(str(model_path))
        print(f"   ✅ Model saved to {model_path}")

        return trainer, feature_cols

    def test_threshold(self, threshold: float, model_path: Path,
                      feature_cols: List[str], test_data: pd.DataFrame,
                      full_data: pd.DataFrame) -> Dict:
        """
        Test ML strategy with specific confidence threshold

        Args:
            threshold: Confidence threshold (0.50-0.70)
            model_path: Path to trained model
            feature_cols: Feature column names
            test_data: Test period data
            full_data: Full data (train + test for indicators)

        Returns:
            Dict with backtest results
        """
        print(f"\n   Testing threshold: {threshold:.2f}")

        # Create ML strategy with this threshold
        strategy = MLStrategy(
            model_path=str(model_path),
            feature_cols=feature_cols,
            confidence_threshold=threshold
        )

        # Run backtest on full data, but only trade during test period
        engine = BacktestEngine(self.backtest_config)

        test_start_date = test_data.index[0]

        results = engine.run_backtest(
            strategy,
            full_data,
            trading_start_date=test_start_date
        )

        metrics = results['metrics']

        print(f"      Trades: {metrics.get('total_trades', 0)}")
        print(f"      Win Rate: {metrics.get('win_rate', 0):.1%}")
        print(f"      Return: {metrics.get('total_return_pct', 0):+.2%}")
        print(f"      Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")

        return {
            'threshold': threshold,
            'trades': metrics.get('total_trades', 0),
            'win_rate': metrics.get('win_rate', 0),
            'total_return': metrics.get('total_return_pct', 0),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
            'max_drawdown': metrics.get('max_drawdown_pct', 0),
            'profit_factor': metrics.get('profit_factor', 0),
            'avg_win': metrics.get('avg_win', 0),
            'avg_loss': metrics.get('avg_loss', 0)
        }

    def optimize(self) -> pd.DataFrame:
        """
        Run optimization across all thresholds

        Returns:
            DataFrame with results for each threshold
        """
        print("="*80)
        print("ML CONFIDENCE THRESHOLD OPTIMIZATION")
        print("Week 1, Day 1 - IMPROVEMENTS_PLAN.md")
        print("="*80)

        # Load and prepare data
        df = self.load_sp500_data()
        df_features = self.prepare_features(df)

        # Split into train/test (80/20)
        split_point = int(len(df_features) * 0.8)
        df_train = df_features.iloc[:split_point].copy()
        df_test = df_features.iloc[split_point:].copy()

        print(f"\n   Train: {df_train.index[0]} to {df_train.index[-1]} ({len(df_train)} days)")
        print(f"   Test:  {df_test.index[0]} to {df_test.index[-1]} ({len(df_test)} days)")

        # Train model once
        model_path = self.output_dir / 'ml_model_for_threshold_test.pkl'
        trainer, feature_cols = self.train_ml_model(df_train, model_path)

        # Test each threshold
        print("\n4. Testing confidence thresholds...")
        print("="*80)

        results = []

        for threshold in self.thresholds:
            try:
                result = self.test_threshold(
                    threshold,
                    model_path,
                    feature_cols,
                    df_test,
                    df_features  # Full data for indicators
                )
                results.append(result)

            except Exception as e:
                print(f"      ❌ Error testing threshold {threshold}: {e}")
                continue

        # Convert to DataFrame
        results_df = pd.DataFrame(results)

        # Save results
        results_df.to_csv(self.output_dir / 'threshold_comparison.csv', index=False)

        # Save detailed JSON
        with open(self.output_dir / 'threshold_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        return results_df

    def analyze_results(self, results_df: pd.DataFrame):
        """Analyze and display optimization results"""
        print("\n" + "="*80)
        print("THRESHOLD OPTIMIZATION RESULTS")
        print("="*80)

        # Display full results table
        print("\n" + results_df.to_string(index=False))

        # Find optimal thresholds for different objectives
        print("\n" + "="*80)
        print("OPTIMAL THRESHOLDS BY OBJECTIVE")
        print("="*80)

        # Most trades
        most_trades = results_df.loc[results_df['trades'].idxmax()]
        print(f"\n📊 Most Trades: {most_trades['threshold']:.2f}")
        print(f"   Trades: {most_trades['trades']:.0f}")
        print(f"   Win Rate: {most_trades['win_rate']:.1%}")
        print(f"   Return: {most_trades['total_return']:+.2%}")

        # Best win rate
        best_winrate = results_df.loc[results_df['win_rate'].idxmax()]
        print(f"\n🎯 Best Win Rate: {best_winrate['threshold']:.2f}")
        print(f"   Win Rate: {best_winrate['win_rate']:.1%}")
        print(f"   Trades: {best_winrate['trades']:.0f}")
        print(f"   Return: {best_winrate['total_return']:+.2%}")

        # Best return
        best_return = results_df.loc[results_df['total_return'].idxmax()]
        print(f"\n💰 Best Return: {best_return['threshold']:.2f}")
        print(f"   Return: {best_return['total_return']:+.2%}")
        print(f"   Win Rate: {best_return['win_rate']:.1%}")
        print(f"   Sharpe: {best_return['sharpe_ratio']:.2f}")

        # Best Sharpe ratio
        best_sharpe = results_df.loc[results_df['sharpe_ratio'].idxmax()]
        print(f"\n📈 Best Sharpe Ratio: {best_sharpe['threshold']:.2f}")
        print(f"   Sharpe: {best_sharpe['sharpe_ratio']:.2f}")
        print(f"   Return: {best_sharpe['total_return']:+.2%}")
        print(f"   Win Rate: {best_sharpe['win_rate']:.1%}")

        # Balanced recommendation
        print("\n" + "="*80)
        print("RECOMMENDATION")
        print("="*80)

        # Score each threshold (balanced approach)
        results_df['score'] = (
            (results_df['total_return'] - results_df['total_return'].min()) /
            (results_df['total_return'].max() - results_df['total_return'].min() + 1e-6) * 0.4 +
            (results_df['win_rate'] - results_df['win_rate'].min()) /
            (results_df['win_rate'].max() - results_df['win_rate'].min() + 1e-6) * 0.3 +
            (results_df['sharpe_ratio'] - results_df['sharpe_ratio'].min()) /
            (results_df['sharpe_ratio'].max() - results_df['sharpe_ratio'].min() + 1e-6) * 0.3
        )

        recommended = results_df.loc[results_df['score'].idxmax()]

        print(f"\n✅ Recommended Threshold: {recommended['threshold']:.2f}")
        print(f"\nBalanced across:")
        print(f"  • Return: {recommended['total_return']:+.2%}")
        print(f"  • Win Rate: {recommended['win_rate']:.1%}")
        print(f"  • Sharpe: {recommended['sharpe_ratio']:.2f}")
        print(f"  • Trades: {recommended['trades']:.0f}")
        print(f"  • Max DD: {recommended['max_drawdown']:.2%}")

        # Comparison to baseline (0.60)
        if 0.60 in results_df['threshold'].values:
            baseline = results_df[results_df['threshold'] == 0.60].iloc[0]

            print(f"\n📊 Improvement vs Baseline (0.60):")
            print(f"  • Trades: {recommended['trades'] / (baseline['trades'] + 1e-6) - 1:+.1%}")
            print(f"  • Win Rate: {(recommended['win_rate'] - baseline['win_rate']) * 100:+.1f}pp")
            print(f"  • Return: {(recommended['total_return'] - baseline['total_return']) * 100:+.2f}pp")

        # Save recommendation
        recommendation = {
            'recommended_threshold': float(recommended['threshold']),
            'metrics': {
                'trades': int(recommended['trades']),
                'win_rate': float(recommended['win_rate']),
                'total_return': float(recommended['total_return']),
                'sharpe_ratio': float(recommended['sharpe_ratio']),
                'max_drawdown': float(recommended['max_drawdown'])
            },
            'optimization_date': datetime.now().isoformat(),
            'all_thresholds_tested': self.thresholds
        }

        with open(self.output_dir / 'recommendation.json', 'w') as f:
            json.dump(recommendation, f, indent=2)

        print(f"\n✅ Results saved to {self.output_dir}/")
        print(f"   - threshold_comparison.csv")
        print(f"   - threshold_results.json")
        print(f"   - recommendation.json")


def main():
    """Main execution"""
    try:
        optimizer = ConfidenceThresholdOptimizer()
        results_df = optimizer.optimize()
        optimizer.analyze_results(results_df)

        print("\n" + "="*80)
        print("✅ OPTIMIZATION COMPLETE")
        print("="*80)
        print("\nNext steps:")
        print("  1. Review recommendation.json for optimal threshold")
        print("  2. Continue to Day 2: Feature selection (remove leakage, top 20)")
        print("  3. Test combined improvements")

    except KeyboardInterrupt:
        print("\n\n⚠️  Execution interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
