#!/usr/bin/env python3
"""
Walk-Forward Validation: Regime-Adaptive ML Strategy
Compare adaptive strategy vs baseline Top 20 model
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
from regime_adaptive_strategy import RegimeAdaptiveMLStrategy


class RegimeAdaptiveValidator:
    """
    Validate regime-adaptive strategy across multiple periods
    """

    def __init__(self, top_features: List[str], output_dir: str = 'regime_adaptive_results'):
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

    def test_strategy(self, split: Dict, strategy_type: str) -> Dict:
        """
        Test strategy on a split

        Args:
            split: Split definition
            strategy_type: 'baseline' or 'adaptive'

        Returns:
            Dict with results
        """
        train_data = split['train_data']
        test_data = split['test_data']
        split_num = split['split_num']

        # Train model
        df_train = train_data.copy()
        df_train['target'] = (df_train['close'].shift(-1) > df_train['close']).astype(int)
        df_train = df_train.dropna()

        X_train = df_train[self.top_features].values
        y_train = df_train['target'].values

        trainer = MLModelTrainer(model_type='xgboost', task='classification')
        trainer.train(X_train, y_train, hyperparameter_tuning=True)

        model_path = self.output_dir / f'model_split_{split_num}_{strategy_type}.pkl'
        trainer.save_model(str(model_path))

        # Create strategy
        if strategy_type == 'adaptive':
            strategy = RegimeAdaptiveMLStrategy(
                model_path=str(model_path),
                feature_cols=self.top_features,
                base_confidence_threshold=0.50,  # Lowered from 0.55
                enable_regime_adaptation=True,
                skip_volatile_regimes=False,  # FIX: Don't skip, use threshold adjustment instead
                skip_bear_regimes=False  # Trade conservatively in bear
            )
        else:  # baseline
            from strategy_examples import MLStrategy
            strategy = MLStrategy(
                model_path=str(model_path),
                feature_cols=self.top_features,
                confidence_threshold=0.55
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
            'strategy': strategy_type,
            'test_start': str(split['test_start']),
            'test_end': str(split['test_end']),
            'trades': metrics.get('total_trades', 0),
            'win_rate': metrics.get('win_rate', 0),
            'total_return': metrics.get('total_return_pct', 0),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
            'max_drawdown': metrics.get('max_drawdown_pct', 0),
            'profit_factor': metrics.get('profit_factor', 0)
        }

    def run_comparison(self) -> tuple:
        """Run comparison between baseline and adaptive strategies"""
        print("="*80)
        print("REGIME-ADAPTIVE STRATEGY VALIDATION")
        print("="*80)

        df = self.load_data()
        df_features = self.prepare_features(df)
        splits = self.create_splits(df_features)

        print(f"\n4. Running comparison on {len(splits)} splits...")
        print("="*80)

        baseline_results = []
        adaptive_results = []

        for split in splits:
            split_num = split['split_num']
            print(f"\n{'='*80}")
            print(f"Split {split_num}: {split['test_start']} to {split['test_end']}")
            print(f"{'='*80}")

            try:
                # Test baseline
                print(f"\n  [1/2] Testing BASELINE strategy...")
                baseline_result = self.test_strategy(split, 'baseline')
                baseline_results.append(baseline_result)
                print(f"        Trades: {baseline_result['trades']}, "
                      f"Win Rate: {baseline_result['win_rate']:.1%}, "
                      f"Return: {baseline_result['total_return']:+.2%}")

                # Test adaptive
                print(f"\n  [2/2] Testing ADAPTIVE strategy...")
                adaptive_result = self.test_strategy(split, 'adaptive')
                adaptive_results.append(adaptive_result)
                print(f"        Trades: {adaptive_result['trades']}, "
                      f"Win Rate: {adaptive_result['win_rate']:.1%}, "
                      f"Return: {adaptive_result['total_return']:+.2%}")

                # Show improvement
                win_rate_improvement = (adaptive_result['win_rate'] - baseline_result['win_rate']) * 100
                return_improvement = (adaptive_result['total_return'] - baseline_result['total_return']) * 100

                print(f"\n  📊 Improvement:")
                print(f"     Win Rate: {win_rate_improvement:+.1f}pp")
                print(f"     Return: {return_improvement:+.2f}pp")

            except Exception as e:
                print(f"  ❌ Error: {e}")
                continue

        return baseline_results, adaptive_results

    def analyze_comparison(self, baseline_results: List[Dict], adaptive_results: List[Dict]):
        """Analyze and compare results"""
        print("\n" + "="*80)
        print("COMPARISON RESULTS")
        print("="*80)

        baseline_df = pd.DataFrame(baseline_results)
        adaptive_df = pd.DataFrame(adaptive_results)

        # Aggregate statistics
        stats = []
        for name, df in [('Baseline', baseline_df), ('Adaptive', adaptive_df)]:
            stats.append({
                'Strategy': name,
                'Splits': len(df),
                'Total Trades': df['trades'].sum(),
                'Avg Trades/Split': f"{df['trades'].mean():.1f}",
                'Avg Win Rate': f"{df['win_rate'].mean():.1%}",
                'Avg Return': f"{df['total_return'].mean():+.2%}",
                'Median Return': f"{df['total_return'].median():+.2%}",
                'Avg Sharpe': f"{df['sharpe_ratio'].mean():.2f}",
                'Avg Max DD': f"{df['max_drawdown'].mean():.2%}",
                'Max Max DD': f"{df['max_drawdown'].max():.2%}",
                'Positive Periods': f"{(df['total_return'] > 0).sum()}/{len(df)} ({(df['total_return'] > 0).sum()/len(df)*100:.1f}%)"
            })

        stats_df = pd.DataFrame(stats)

        print("\n" + stats_df.to_string(index=False))

        # Calculate improvements
        print("\n" + "="*80)
        print("IMPROVEMENTS (Adaptive vs Baseline)")
        print("="*80)

        improvements = {
            'win_rate': (adaptive_df['win_rate'].mean() - baseline_df['win_rate'].mean()) * 100,
            'return': (adaptive_df['total_return'].mean() - baseline_df['total_return'].mean()) * 100,
            'sharpe': adaptive_df['sharpe_ratio'].mean() - baseline_df['sharpe_ratio'].mean(),
            'max_dd': (baseline_df['max_drawdown'].max() - adaptive_df['max_drawdown'].max()) * 100,
            'positive_periods': (adaptive_df['total_return'] > 0).sum() - (baseline_df['total_return'] > 0).sum()
        }

        print(f"\nWin Rate: {improvements['win_rate']:+.1f}pp")
        print(f"Avg Return: {improvements['return']:+.2f}pp")
        print(f"Sharpe Ratio: {improvements['sharpe']:+.2f}")
        print(f"Max Drawdown Reduction: {improvements['max_dd']:+.2f}pp")
        print(f"Additional Positive Periods: {improvements['positive_periods']:+d}")

        # Validation criteria
        print("\n" + "="*80)
        print("VALIDATION CRITERIA (Adaptive Strategy)")
        print("="*80)

        avg_win_rate = adaptive_df['win_rate'].mean()
        avg_return = adaptive_df['total_return'].mean()
        positive_pct = (adaptive_df['total_return'] > 0).sum() / len(adaptive_df) * 100
        avg_max_dd = adaptive_df['max_drawdown'].mean()
        max_max_dd = adaptive_df['max_drawdown'].max()

        criteria = [
            ("Minimum 3 periods", len(adaptive_df) >= 3),
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
            print("\n🎉 ADAPTIVE STRATEGY PASSED VALIDATION!")
        elif passed >= 4:
            print("\n⚠️  Adaptive strategy shows improvement but needs refinement.")
        else:
            print("\n❌ Adaptive strategy still needs work.")

        # Save results
        baseline_df.to_csv(self.output_dir / 'baseline_results.csv', index=False)
        adaptive_df.to_csv(self.output_dir / 'adaptive_results.csv', index=False)
        stats_df.to_csv(self.output_dir / 'comparison_summary.csv', index=False)

        summary = {
            'baseline': {
                'avg_win_rate': float(baseline_df['win_rate'].mean()),
                'avg_return': float(baseline_df['total_return'].mean()),
                'avg_sharpe': float(baseline_df['sharpe_ratio'].mean()),
                'positive_periods_pct': float((baseline_df['total_return'] > 0).sum() / len(baseline_df) * 100)
            },
            'adaptive': {
                'avg_win_rate': float(adaptive_df['win_rate'].mean()),
                'avg_return': float(adaptive_df['total_return'].mean()),
                'avg_sharpe': float(adaptive_df['sharpe_ratio'].mean()),
                'positive_periods_pct': float((adaptive_df['total_return'] > 0).sum() / len(adaptive_df) * 100)
            },
            'improvements': improvements,
            'criteria_passed': passed,
            'criteria_total': len(criteria),
            'validation_date': datetime.now().isoformat()
        }

        with open(self.output_dir / 'comparison_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)

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
        validator = RegimeAdaptiveValidator(TOP_20_FEATURES)
        baseline_results, adaptive_results = validator.run_comparison()
        validator.analyze_comparison(baseline_results, adaptive_results)

        print("\n" + "="*80)
        print("✅ VALIDATION COMPLETE")
        print("="*80)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
