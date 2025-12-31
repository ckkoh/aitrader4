#!/usr/bin/env python3
"""
Week 1, Day 2: Feature Selection & Data Leakage Removal
From IMPROVEMENTS_PLAN.md

Problem: 91 features including data leakage (target_*, future_return)
Solution: Remove leakage, select top 20 clean features

Expected Impact: +5-10% accuracy, faster training, less overfitting
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple

from feature_engineering import FeatureEngineering
from ml_training_pipeline import MLModelTrainer
from backtesting_engine import BacktestEngine, BacktestConfig
from strategy_examples import MLStrategy


class FeatureSelector:
    """
    Remove data leakage and select top features for ML model
    """

    def __init__(self, output_dir: str = 'feature_selection_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Data leakage features to ALWAYS exclude
        self.LEAKAGE_FEATURES = [
            'target_regression',
            'future_return',
            'target_binary',
            'target_class',
            'target'  # Any variation
        ]

        # Non-feature columns (price data, not features)
        self.NON_FEATURES = [
            'open', 'high', 'low', 'close', 'volume'
        ]

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

        # Use last 2 years for training + testing
        df = df.loc['2023-01-01':]

        print(f"   ✅ Loaded {len(df)} days from {data_file}")
        print(f"   Date range: {df.index[0]} to {df.index[-1]}")

        return df

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all features (including leakage for comparison)"""
        print("\n2. Generating features...")

        df_features = FeatureEngineering.build_complete_feature_set(
            df,
            include_volume=True
        )

        df_features = df_features.dropna()

        print(f"   ✅ Generated {len(df_features.columns)} total features")
        print(f"   Clean data points: {len(df_features)}")

        return df_features

    def identify_leakage(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Identify and separate leakage vs clean features

        Returns:
            (leakage_features, clean_features)
        """
        print("\n3. Identifying data leakage features...")

        all_cols = df.columns.tolist()

        # Find leakage features (case-insensitive partial match)
        leakage_found = []
        for col in all_cols:
            col_lower = col.lower()
            for leak in self.LEAKAGE_FEATURES:
                if leak.lower() in col_lower:
                    leakage_found.append(col)
                    break

        # Find non-feature columns
        non_features_found = [col for col in all_cols if col in self.NON_FEATURES]

        # Clean features = all - leakage - non_features
        exclude_all = set(leakage_found + non_features_found)
        clean_features = [col for col in all_cols if col not in exclude_all]

        print(f"\n   📊 Feature Breakdown:")
        print(f"   Total columns: {len(all_cols)}")
        print(f"   🔴 Leakage features: {len(leakage_found)}")
        print(f"   🔵 Non-features (OHLCV): {len(non_features_found)}")
        print(f"   ✅ Clean features: {len(clean_features)}")

        if leakage_found:
            print(f"\n   🚨 LEAKAGE DETECTED:")
            for leak in leakage_found:
                print(f"      - {leak}")

        # Save breakdown
        breakdown = {
            'total_columns': len(all_cols),
            'leakage_features': leakage_found,
            'non_features': non_features_found,
            'clean_features': clean_features,
            'leakage_count': len(leakage_found),
            'clean_count': len(clean_features)
        }

        with open(self.output_dir / 'feature_breakdown.json', 'w') as f:
            json.dump(breakdown, f, indent=2)

        return leakage_found, clean_features

    def analyze_feature_importance(self, df: pd.DataFrame,
                                   feature_cols: List[str]) -> pd.DataFrame:
        """
        Train model and analyze feature importance

        Args:
            df: DataFrame with features
            feature_cols: List of feature columns to use

        Returns:
            DataFrame with feature importances
        """
        print(f"\n4. Analyzing feature importance ({len(feature_cols)} features)...")

        # Create target
        df_train = df.copy()
        df_train['target'] = (df_train['close'].shift(-1) > df_train['close']).astype(int)
        df_train = df_train.dropna()

        X = df_train[feature_cols].values
        y = df_train['target'].values

        # Train model
        trainer = MLModelTrainer(model_type='xgboost', task='classification')

        print("   Training model for feature importance analysis...")
        results = trainer.train(X, y, hyperparameter_tuning=False)  # Faster without tuning

        # Get feature importances
        feature_importances = trainer.model.feature_importances_

        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': feature_importances
        }).sort_values('importance', ascending=False)

        # Add percentage
        total_importance = importance_df['importance'].sum()
        importance_df['importance_pct'] = (importance_df['importance'] / total_importance * 100)

        print(f"\n   ✅ Feature importance calculated")
        print(f"   Training accuracy: {results.get('train_accuracy', 0):.2f}%")

        return importance_df

    def select_top_features(self, importance_df: pd.DataFrame,
                           n_features: int = 20) -> List[str]:
        """
        Select top N features by importance

        Args:
            importance_df: DataFrame with feature importances
            n_features: Number of features to select

        Returns:
            List of top feature names
        """
        print(f"\n5. Selecting top {n_features} features...")

        top_features_df = importance_df.head(n_features)
        top_features_list = top_features_df['feature'].tolist()

        print(f"\n   🎯 Top {n_features} Features:")
        print("   " + "="*60)

        for i in range(min(n_features, len(top_features_df))):
            feature = top_features_df.iloc[i]['feature']
            importance = top_features_df.iloc[i]['importance_pct']
            print(f"   {i+1:2d}. {feature:30s} {importance:5.2f}%")

        # Calculate cumulative importance
        cumulative = top_features_df['importance_pct'].sum()
        print(f"\n   Cumulative importance: {cumulative:.1f}%")

        return top_features_list

    def train_and_compare_models(self, df: pd.DataFrame,
                                 all_features: List[str],
                                 clean_features: List[str],
                                 top_features: List[str]) -> Dict:
        """
        Train 3 models and compare:
        1. With leakage features (baseline)
        2. All clean features (no leakage)
        3. Top 20 clean features (optimal)

        Returns:
            Dict with comparison results
        """
        print("\n6. Training and comparing models...")
        print("   " + "="*60)

        # Split data
        split_point = int(len(df) * 0.8)
        df_train = df.iloc[:split_point].copy()
        df_test = df.iloc[split_point:].copy()

        print(f"   Train: {len(df_train)} days")
        print(f"   Test:  {len(df_test)} days")

        results = {}

        # Model 1: With leakage (baseline - shows why we need to fix)
        print("\n   📊 Model 1: WITH LEAKAGE (baseline)")
        results['with_leakage'] = self._train_single_model(
            df_train, df_test, df, all_features, "model_with_leakage.pkl"
        )

        # Model 2: All clean features (no leakage)
        print("\n   📊 Model 2: ALL CLEAN FEATURES (no leakage)")
        results['all_clean'] = self._train_single_model(
            df_train, df_test, df, clean_features, "model_all_clean.pkl"
        )

        # Model 3: Top 20 features (optimal)
        print("\n   📊 Model 3: TOP 20 FEATURES (optimal)")
        results['top_20'] = self._train_single_model(
            df_train, df_test, df, top_features, "model_top20.pkl"
        )

        return results

    def _train_single_model(self, df_train: pd.DataFrame,
                           df_test: pd.DataFrame,
                           df_full: pd.DataFrame,
                           feature_cols: List[str],
                           model_name: str) -> Dict:
        """Train a single model and evaluate"""

        # Prepare data
        df_train_prep = df_train.copy()
        df_train_prep['target'] = (df_train_prep['close'].shift(-1) > df_train_prep['close']).astype(int)
        df_train_prep = df_train_prep.dropna()

        X_train = df_train_prep[feature_cols].values
        y_train = df_train_prep['target'].values

        # Train model
        trainer = MLModelTrainer(model_type='xgboost', task='classification')
        train_results = trainer.train(X_train, y_train, hyperparameter_tuning=True)

        # Save model
        model_path = self.output_dir / model_name
        trainer.save_model(str(model_path))

        print(f"      Training accuracy: {train_results.get('train_accuracy', 0):.2f}%")
        print(f"      Features used: {len(feature_cols)}")

        # Backtest
        strategy = MLStrategy(
            model_path=str(model_path),
            feature_cols=feature_cols,
            confidence_threshold=0.55  # Use middle threshold
        )

        engine = BacktestEngine(self.backtest_config)
        test_start_date = df_test.index[0]

        backtest_results = engine.run_backtest(
            strategy,
            df_full,
            trading_start_date=test_start_date
        )

        metrics = backtest_results['metrics']

        print(f"      Trades: {metrics.get('total_trades', 0)}")
        print(f"      Win Rate: {metrics.get('win_rate', 0):.1%}")
        print(f"      Return: {metrics.get('total_return_pct', 0):+.2%}")
        print(f"      Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")

        return {
            'model_name': model_name,
            'features_count': len(feature_cols),
            'train_accuracy': train_results.get('train_accuracy', 0),
            'trades': metrics.get('total_trades', 0),
            'win_rate': metrics.get('win_rate', 0),
            'total_return': metrics.get('total_return_pct', 0),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
            'max_drawdown': metrics.get('max_drawdown_pct', 0),
            'profit_factor': metrics.get('profit_factor', 0)
        }

    def generate_report(self, results: Dict, importance_df: pd.DataFrame,
                       top_features: List[str]):
        """Generate comparison report"""
        print("\n" + "="*80)
        print("FEATURE SELECTION RESULTS")
        print("="*80)

        # Create comparison table
        comparison_data = []
        for model_type, result in results.items():
            comparison_data.append({
                'Model': model_type,
                'Features': result['features_count'],
                'Train Acc': f"{result['train_accuracy']:.1f}%",
                'Trades': result['trades'],
                'Win Rate': f"{result['win_rate']:.1%}",
                'Return': f"{result['total_return']:+.2%}",
                'Sharpe': f"{result['sharpe_ratio']:.2f}",
                'Max DD': f"{result['max_drawdown']:.2%}"
            })

        comparison_df = pd.DataFrame(comparison_data)

        print("\n" + comparison_df.to_string(index=False))

        # Calculate improvements
        baseline = results['with_leakage']
        optimal = results['top_20']

        print("\n" + "="*80)
        print("IMPROVEMENT: Top 20 vs Baseline (with leakage)")
        print("="*80)

        improvements = {
            'win_rate_change': (optimal['win_rate'] - baseline['win_rate']) * 100,
            'return_change': (optimal['total_return'] - baseline['total_return']) * 100,
            'sharpe_change': optimal['sharpe_ratio'] - baseline['sharpe_ratio'],
            'trades_change': optimal['trades'] - baseline['trades']
        }

        print(f"Win Rate:  {baseline['win_rate']:.1%} → {optimal['win_rate']:.1%} ({improvements['win_rate_change']:+.1f}pp)")
        print(f"Return:    {baseline['total_return']:+.2%} → {optimal['total_return']:+.2%} ({improvements['return_change']:+.2f}pp)")
        print(f"Sharpe:    {baseline['sharpe_ratio']:.2f} → {optimal['sharpe_ratio']:.2f} ({improvements['sharpe_change']:+.2f})")
        print(f"Trades:    {baseline['trades']} → {optimal['trades']} ({improvements['trades_change']:+d})")

        # Save results
        comparison_df.to_csv(self.output_dir / 'model_comparison.csv', index=False)

        # Save top features
        top_features_df = importance_df.head(20)
        top_features_df.to_csv(self.output_dir / 'top_20_features.csv', index=False)

        # Save detailed results
        with open(self.output_dir / 'detailed_results.json', 'w') as f:
            json.dump({
                'results': results,
                'improvements': improvements,
                'top_features': top_features,
                'analysis_date': datetime.now().isoformat()
            }, f, indent=2)

        print(f"\n✅ Results saved to {self.output_dir}/")
        print("   - model_comparison.csv")
        print("   - top_20_features.csv")
        print("   - detailed_results.json")
        print("   - feature_breakdown.json")


def main():
    """Main execution"""
    print("="*80)
    print("WEEK 1, DAY 2: FEATURE SELECTION & DATA LEAKAGE REMOVAL")
    print("="*80)

    try:
        selector = FeatureSelector()

        # Load data
        df = selector.load_data()

        # Generate features
        df_features = selector.generate_features(df)

        # Identify leakage
        leakage_features, clean_features = selector.identify_leakage(df_features)

        # Get all features (including leakage for comparison)
        all_features = leakage_features + clean_features

        # Analyze importance on CLEAN features only
        importance_df = selector.analyze_feature_importance(df_features, clean_features)

        # Select top 20
        top_features = selector.select_top_features(importance_df, n_features=20)

        # Train and compare models
        results = selector.train_and_compare_models(
            df_features,
            all_features,
            clean_features,
            top_features
        )

        # Generate report
        selector.generate_report(results, importance_df, top_features)

        print("\n" + "="*80)
        print("✅ DAY 2 COMPLETE")
        print("="*80)
        print("\nKey Findings:")
        print("  1. Removed data leakage features (target_*, future_return)")
        print("  2. Selected top 20 predictive features")
        print("  3. Retrained model with clean features")
        print("  4. Compared performance: leakage vs clean vs top 20")
        print("\nNext: Day 3 - Multi-timeframe prediction")

    except KeyboardInterrupt:
        print("\n\n⚠️  Execution interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
