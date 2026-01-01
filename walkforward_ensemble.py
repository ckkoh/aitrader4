#!/usr/bin/env python3
"""
Walk-Forward Validation: Ensemble Strategy
Combines Original (bear market champion) + Balanced (bull market champion)
Tests if ensemble captures best of both worlds
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List
from sklearn.utils import class_weight
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

from feature_engineering import FeatureEngineering
from ml_training_pipeline import MLModelTrainer
from backtesting_engine import BacktestEngine, BacktestConfig
from ensemble_regime_strategy import EnsembleRegimeStrategy


class EnsembleValidator:
    """
    Validate ensemble strategy combining original and balanced models
    """

    def __init__(self, top_features: List[str], output_dir: str = 'ensemble_results'):
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
        df = pd.read_csv(data_file, index_col='Date', parse_dates=True)
        df = df.loc['2020-01-01':]

        print(f"   ✅ Loaded {len(df)} days")
        print(f"   Date range: {df.index[0]} to {df.index[-1]}")

        return df

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features"""
        print("\n2. Generating features...")

        df_features = FeatureEngineering.build_complete_feature_set(df, include_volume=True)
        df_features = df_features.dropna()

        print(f"   ✅ Clean data points: {len(df_features)}")

        return df_features

    def create_splits(self, df: pd.DataFrame) -> List[Dict]:
        """Create walk-forward splits"""
        print(f"\n3. Creating walk-forward splits...")

        train_months, test_months, step_months = 12, 3, 3
        days_per_month = 21

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

        print(f"   ✅ Created {len(splits)} splits")

        return splits

    def train_both_models(self, split: Dict) -> tuple:
        """
        Train both original and balanced models

        Returns:
            (original_model_path, balanced_model_path)
        """
        train_data = split['train_data']
        split_num = split['split_num']

        # Prepare training data
        df_train = train_data.copy()
        df_train['target'] = (df_train['close'].shift(-1) > df_train['close']).astype(int)
        df_train = df_train.dropna()

        X_train = df_train[self.top_features].values
        y_train = df_train['target'].values

        print(f"  Training both models (split {split_num})...")
        print(f"  Class distribution: {np.bincount(y_train)} (0=SELL, 1=BUY)")

        # 1. Train ORIGINAL model (no balancing)
        trainer_orig = MLModelTrainer(model_type='xgboost', task='classification')
        trainer_orig.train(X_train, y_train, hyperparameter_tuning=True)
        original_path = str(self.output_dir / f'model_split_{split_num}_original.pkl')
        trainer_orig.save_model(original_path)

        # 2. Train BALANCED model (with class weights)
        classes = np.unique(y_train)
        class_weights = class_weight.compute_class_weight('balanced', classes=classes, y=y_train)
        sample_weights = np.zeros(len(y_train))
        for i, cls in enumerate(classes):
            sample_weights[y_train == cls] = class_weights[i]

        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 4, 6],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }

        base_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=0)
        grid_search.fit(X_train, y_train, sample_weight=sample_weights)

        trainer_balanced = MLModelTrainer(model_type='xgboost', task='classification')
        trainer_balanced.model = grid_search.best_estimator_
        balanced_path = str(self.output_dir / f'model_split_{split_num}_balanced.pkl')
        trainer_balanced.save_model(balanced_path)

        print(f"  ✅ Both models trained")

        return original_path, balanced_path

    def test_ensemble(self, split: Dict, original_path: str, balanced_path: str) -> Dict:
        """
        Test ensemble strategy on a split

        Args:
            split: Split definition
            original_path: Path to original model
            balanced_path: Path to balanced model

        Returns:
            Dict with results
        """
        train_data = split['train_data']
        test_data = split['test_data']
        split_num = split['split_num']

        # Create ensemble strategy
        strategy = EnsembleRegimeStrategy(
            original_model_path=original_path,
            balanced_model_path=balanced_path,
            feature_cols=self.top_features,
            base_confidence_threshold=0.50,
            enable_regime_adaptation=True
        )

        # Backtest
        combined_data = pd.concat([train_data, test_data])
        engine = BacktestEngine(self.backtest_config)
        test_start_date = test_data.index[0]

        backtest_results = engine.run_backtest(
            strategy,
            combined_data,
            trading_start_date=test_start_date
        )

        metrics = backtest_results['metrics']

        return {
            'split_num': split_num,
            'strategy': 'ensemble',
            'test_start': str(split['test_start']),
            'test_end': str(split['test_end']),
            'trades': metrics.get('total_trades', 0),
            'win_rate': metrics.get('win_rate', 0),
            'total_return': metrics.get('total_return_pct', 0),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
            'max_drawdown': metrics.get('max_drawdown_pct', 0),
            'profit_factor': metrics.get('profit_factor', 0)
        }

    def run_validation(self) -> List[Dict]:
        """Run ensemble validation"""
        print("="*80)
        print("ENSEMBLE STRATEGY VALIDATION")
        print("Combining Original (bear champion) + Balanced (bull champion)")
        print("="*80)

        df = self.load_data()
        df_features = self.prepare_features(df)
        splits = self.create_splits(df_features)

        print(f"\n4. Running ensemble validation on {len(splits)} splits...")
        print("="*80)

        ensemble_results = []

        for split in splits:
            split_num = split['split_num']
            print(f"\n{'='*80}")
            print(f"Split {split_num}: {split['test_start']} to {split['test_end']}")
            print(f"{'='*80}")

            try:
                # Train both models
                original_path, balanced_path = self.train_both_models(split)

                # Test ensemble
                print(f"\n  Testing ENSEMBLE strategy...")
                ensemble_result = self.test_ensemble(split, original_path, balanced_path)
                ensemble_results.append(ensemble_result)
                print(f"  ✅ Trades: {ensemble_result['trades']}, "
                      f"Win Rate: {ensemble_result['win_rate']:.1%}, "
                      f"Return: {ensemble_result['total_return']:+.2%}")

            except Exception as e:
                print(f"  ❌ Error: {e}")
                import traceback
                traceback.print_exc()
                continue

        return ensemble_results

    def analyze_results(self, ensemble_results: List[Dict]):
        """Analyze ensemble results"""
        print("\n" + "="*80)
        print("ENSEMBLE STRATEGY RESULTS")
        print("="*80)

        ensemble_df = pd.DataFrame(ensemble_results)

        # Aggregate statistics
        stats = {
            'Strategy': 'Ensemble',
            'Splits': len(ensemble_df),
            'Total Trades': ensemble_df['trades'].sum(),
            'Avg Trades/Split': f"{ensemble_df['trades'].mean():.1f}",
            'Avg Win Rate': f"{ensemble_df['win_rate'].mean():.1%}",
            'Avg Return': f"{ensemble_df['total_return'].mean():+.2%}",
            'Median Return': f"{ensemble_df['total_return'].median():+.2%}",
            'Avg Sharpe': f"{ensemble_df['sharpe_ratio'].mean():.2f}",
            'Avg Max DD': f"{ensemble_df['max_drawdown'].mean():.2%}",
            'Max Max DD': f"{ensemble_df['max_drawdown'].max():.2%}",
            'Positive Periods': f"{(ensemble_df['total_return'] > 0).sum()}/{len(ensemble_df)} ({(ensemble_df['total_return'] > 0).sum()/len(ensemble_df)*100:.1f}%)"
        }

        print("\nPerformance Summary:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # Validation criteria
        print("\n" + "="*80)
        print("VALIDATION CRITERIA (Ensemble)")
        print("="*80)

        avg_win_rate = ensemble_df['win_rate'].mean()
        avg_return = ensemble_df['total_return'].mean()
        positive_pct = (ensemble_df['total_return'] > 0).sum() / len(ensemble_df) * 100
        avg_max_dd = ensemble_df['max_drawdown'].mean()
        max_max_dd = ensemble_df['max_drawdown'].max()

        criteria = [
            ("Minimum 3 periods", len(ensemble_df) >= 3),
            ("Avg Win Rate > 50%", avg_win_rate > 0.50),
            ("Avg Return > 0%", avg_return > 0),
            ("Positive periods >50%", positive_pct > 50),
            ("Avg Max DD < 15%", avg_max_dd < 0.15),
            ("Max Max DD < 20%", max_max_dd < 0.20),
        ]

        passed = sum([1 for _, result in criteria if result])

        for criterion, result in criteria:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status}: {criterion}")

        print(f"\nOverall: {passed}/{len(criteria)} criteria passed ({passed/len(criteria)*100:.0f}%)")

        if passed >= 5:
            print("\n🎉 ENSEMBLE STRATEGY PASSED VALIDATION!")
            print("   Ready for paper trading!")
        elif passed >= 4:
            print("\n⚠️  Ensemble strategy shows promise but needs refinement.")
        else:
            print("\n❌ Ensemble strategy still needs work.")

        # Save results
        ensemble_df.to_csv(self.output_dir / 'ensemble_results.csv', index=False)

        summary = {
            'ensemble': {
                'avg_win_rate': float(ensemble_df['win_rate'].mean()),
                'avg_return': float(ensemble_df['total_return'].mean()),
                'avg_sharpe': float(ensemble_df['sharpe_ratio'].mean()),
                'positive_periods_pct': float((ensemble_df['total_return'] > 0).sum() / len(ensemble_df) * 100),
                'total_trades': int(ensemble_df['trades'].sum()),
                'avg_trades_per_split': float(ensemble_df['trades'].mean())
            },
            'criteria_passed': passed,
            'criteria_total': len(criteria),
            'validation_date': datetime.now().isoformat()
        }

        with open(self.output_dir / 'ensemble_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n✅ Results saved to {self.output_dir}/")

        # Load comparison data if available
        self.compare_with_others(ensemble_df)

    def compare_with_others(self, ensemble_df: pd.DataFrame):
        """Compare ensemble with baseline, original, and balanced"""
        print("\n" + "="*80)
        print("COMPARISON WITH OTHER STRATEGIES")
        print("="*80)

        try:
            # Load other results
            baseline_df = pd.read_csv('regime_adaptive_results/baseline_results.csv')
            original_df = pd.read_csv('regime_adaptive_results/adaptive_results.csv')
            balanced_df = pd.read_csv('balanced_model_results/balanced_results.csv')

            # Create comparison table
            comparison = {
                'Strategy': ['Baseline', 'Original Adaptive', 'Balanced Adaptive', '**Ensemble**'],
                'Trades': [
                    baseline_df['trades'].sum(),
                    original_df['trades'].sum(),
                    balanced_df['trades'].sum(),
                    ensemble_df['trades'].sum()
                ],
                'Win Rate': [
                    f"{baseline_df['win_rate'].mean():.1%}",
                    f"{original_df['win_rate'].mean():.1%}",
                    f"{balanced_df['win_rate'].mean():.1%}",
                    f"{ensemble_df['win_rate'].mean():.1%}"
                ],
                'Avg Return': [
                    f"{baseline_df['total_return'].mean():+.2%}",
                    f"{original_df['total_return'].mean():+.2%}",
                    f"{balanced_df['total_return'].mean():+.2%}",
                    f"{ensemble_df['total_return'].mean():+.2%}"
                ],
                'Sharpe': [
                    f"{baseline_df['sharpe_ratio'].mean():.2f}",
                    f"{original_df['sharpe_ratio'].mean():.2f}",
                    f"{balanced_df['sharpe_ratio'].mean():.2f}",
                    f"{ensemble_df['sharpe_ratio'].mean():.2f}"
                ],
                'Max DD': [
                    f"{baseline_df['max_drawdown'].max():.2%}",
                    f"{original_df['max_drawdown'].max():.2%}",
                    f"{balanced_df['max_drawdown'].max():.2%}",
                    f"{ensemble_df['max_drawdown'].max():.2%}"
                ],
                'Positive': [
                    f"{(baseline_df['total_return'] > 0).sum()}/15",
                    f"{(original_df['total_return'] > 0).sum()}/15",
                    f"{(balanced_df['total_return'] > 0).sum()}/15",
                    f"{(ensemble_df['total_return'] > 0).sum()}/15"
                ]
            }

            comp_df = pd.DataFrame(comparison)
            print("\n" + comp_df.to_string(index=False))

        except FileNotFoundError:
            print("\n  (Comparison data not available)")


def main():
    """Main execution"""

    TOP_20_FEATURES = [
        'bullish_engulfing', 'stoch_d_3', 'week_of_year', 'atr_14', 'regime',
        'roc_20', 'obv', 'parkinson_vol_10', 'volatility_200d', 'momentum_5',
        'macd_signal', 'adx_14', 'month_sin', 'hl_ratio', 'rsi_14',
        'stoch_k_14', 'bb_position_20', 'momentum_oscillator', 'pvt', 'price_acceleration'
    ]

    try:
        validator = EnsembleValidator(TOP_20_FEATURES)
        ensemble_results = validator.run_validation()
        validator.analyze_results(ensemble_results)

        print("\n" + "="*80)
        print("✅ ENSEMBLE VALIDATION COMPLETE")
        print("="*80)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
