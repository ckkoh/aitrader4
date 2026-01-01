#!/usr/bin/env python3
"""
Walk-Forward Validation: Regime-Adaptive ML Strategy with BALANCED CLASS WEIGHTS
Fix SELL bias by training models with balanced class weights
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List
from sklearn.utils import class_weight

from feature_engineering import FeatureEngineering
from ml_training_pipeline import MLModelTrainer
from backtesting_engine import BacktestEngine, BacktestConfig
from regime_adaptive_strategy import RegimeAdaptiveMLStrategy


class BalancedModelValidator:
    """
    Validate regime-adaptive strategy with balanced class weight training
    """

    def __init__(self, top_features: List[str], output_dir: str = 'balanced_model_results'):
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

    def test_strategy_balanced(self, split: Dict) -> Dict:
        """
        Test strategy on a split with BALANCED class weights

        Args:
            split: Split definition

        Returns:
            Dict with results
        """
        train_data = split['train_data']
        test_data = split['test_data']
        split_num = split['split_num']

        # Train model with BALANCED CLASS WEIGHTS
        df_train = train_data.copy()
        df_train['target'] = (df_train['close'].shift(-1) > df_train['close']).astype(int)
        df_train = df_train.dropna()

        X_train = df_train[self.top_features].values
        y_train = df_train['target'].values

        # Calculate class weights for balancing
        classes = np.unique(y_train)
        class_weights = class_weight.compute_class_weight(
            'balanced',
            classes=classes,
            y=y_train
        )

        # Create sample weights array
        sample_weights = np.zeros(len(y_train))
        for i, cls in enumerate(classes):
            sample_weights[y_train == cls] = class_weights[i]

        print(f"  Class distribution: {np.bincount(y_train)} (0=SELL, 1=BUY)")
        print(f"  Class weights: {dict(zip(classes, class_weights))}")
        print(f"  Sample weight range: {sample_weights.min():.3f} - {sample_weights.max():.3f}")

        # Train with balanced weights
        trainer = MLModelTrainer(model_type='xgboost', task='classification')

        # XGBoost hyperparameter grid (same as original)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 4, 6],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }

        # Train with sample weights for class balancing
        from sklearn.model_selection import GridSearchCV
        import xgboost as xgb

        base_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=0
        )

        # Fit with sample weights
        grid_search.fit(X_train, y_train, sample_weight=sample_weights)

        trainer.model = grid_search.best_estimator_
        print(f"  Best parameters: {grid_search.best_params_}")
        print(f"  Best CV score: {grid_search.best_score_:.4f}")

        # Save model
        model_path = self.output_dir / f'model_split_{split_num}_balanced.pkl'
        trainer.save_model(str(model_path))

        # Create strategy
        strategy = RegimeAdaptiveMLStrategy(
            model_path=str(model_path),
            feature_cols=self.top_features,
            base_confidence_threshold=0.50,
            enable_regime_adaptation=True,
            skip_volatile_regimes=False,
            skip_bear_regimes=False
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
            'strategy': 'balanced',
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
        """Run validation with balanced models"""
        print("="*80)
        print("BALANCED CLASS WEIGHT VALIDATION")
        print("="*80)

        df = self.load_data()
        df_features = self.prepare_features(df)
        splits = self.create_splits(df_features)

        print(f"\n4. Running validation on {len(splits)} splits...")
        print("="*80)

        balanced_results = []

        for split in splits:
            split_num = split['split_num']
            print(f"\n{'='*80}")
            print(f"Split {split_num}: {split['test_start']} to {split['test_end']}")
            print(f"{'='*80}")

            try:
                # Test with balanced class weights
                print(f"\n  Testing BALANCED model...")
                balanced_result = self.test_strategy_balanced(split)
                balanced_results.append(balanced_result)
                print(f"  ✅ Trades: {balanced_result['trades']}, "
                      f"Win Rate: {balanced_result['win_rate']:.1%}, "
                      f"Return: {balanced_result['total_return']:+.2%}")

            except Exception as e:
                print(f"  ❌ Error: {e}")
                import traceback
                traceback.print_exc()
                continue

        return balanced_results

    def analyze_results(self, balanced_results: List[Dict]):
        """Analyze balanced model results"""
        print("\n" + "="*80)
        print("BALANCED MODEL RESULTS")
        print("="*80)

        balanced_df = pd.DataFrame(balanced_results)

        # Aggregate statistics
        stats = {
            'Strategy': 'Balanced',
            'Splits': len(balanced_df),
            'Total Trades': balanced_df['trades'].sum(),
            'Avg Trades/Split': f"{balanced_df['trades'].mean():.1f}",
            'Avg Win Rate': f"{balanced_df['win_rate'].mean():.1%}",
            'Avg Return': f"{balanced_df['total_return'].mean():+.2%}",
            'Median Return': f"{balanced_df['total_return'].median():+.2%}",
            'Avg Sharpe': f"{balanced_df['sharpe_ratio'].mean():.2f}",
            'Avg Max DD': f"{balanced_df['max_drawdown'].mean():.2%}",
            'Max Max DD': f"{balanced_df['max_drawdown'].max():.2%}",
            'Positive Periods': f"{(balanced_df['total_return'] > 0).sum()}/{len(balanced_df)} ({(balanced_df['total_return'] > 0).sum()/len(balanced_df)*100:.1f}%)"
        }

        print("\nPerformance Summary:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # Validation criteria
        print("\n" + "="*80)
        print("VALIDATION CRITERIA (Balanced Model)")
        print("="*80)

        avg_win_rate = balanced_df['win_rate'].mean()
        avg_return = balanced_df['total_return'].mean()
        positive_pct = (balanced_df['total_return'] > 0).sum() / len(balanced_df) * 100
        avg_max_dd = balanced_df['max_drawdown'].mean()
        max_max_dd = balanced_df['max_drawdown'].max()

        criteria = [
            ("Minimum 3 periods", len(balanced_df) >= 3),
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
            print("\n🎉 BALANCED MODEL PASSED VALIDATION!")
        elif passed >= 4:
            print("\n⚠️  Balanced model shows improvement but needs refinement.")
        else:
            print("\n❌ Balanced model still needs work.")

        # Save results
        balanced_df.to_csv(self.output_dir / 'balanced_results.csv', index=False)

        summary = {
            'balanced': {
                'avg_win_rate': float(balanced_df['win_rate'].mean()),
                'avg_return': float(balanced_df['total_return'].mean()),
                'avg_sharpe': float(balanced_df['sharpe_ratio'].mean()),
                'positive_periods_pct': float((balanced_df['total_return'] > 0).sum() / len(balanced_df) * 100),
                'total_trades': int(balanced_df['trades'].sum()),
                'avg_trades_per_split': float(balanced_df['trades'].mean())
            },
            'criteria_passed': passed,
            'criteria_total': len(criteria),
            'validation_date': datetime.now().isoformat()
        }

        with open(self.output_dir / 'balanced_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n✅ Results saved to {self.output_dir}/")


def main():
    """Main execution"""

    TOP_20_FEATURES = [
        'bullish_engulfing', 'stoch_d_3', 'week_of_year', 'atr_14', 'regime',
        'roc_20', 'obv', 'parkinson_vol_10', 'volatility_200d', 'momentum_5',
        'macd_signal', 'adx_14', 'month_sin', 'hl_ratio', 'rsi_14',
        'stoch_k_14', 'bb_position_20', 'momentum_oscillator', 'pvt', 'price_acceleration'
    ]

    try:
        validator = BalancedModelValidator(TOP_20_FEATURES)
        balanced_results = validator.run_validation()
        validator.analyze_results(balanced_results)

        print("\n" + "="*80)
        print("✅ BALANCED MODEL VALIDATION COMPLETE")
        print("="*80)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
