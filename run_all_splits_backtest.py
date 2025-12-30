#!/usr/bin/env python3
"""
Run Momentum Strategy Backtests on All Walk-Forward Splits

Comprehensive testing plan:
1. Load all 80/20 splits for both datasets
2. Run momentum strategy backtest on each split
3. Aggregate results across all splits
4. Generate performance comparison reports
5. Validate strategy robustness across different market regimes
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import List, Dict
from momentum_strategy_80_20 import run_momentum_backtest


class WalkForwardBacktestRunner:
    """
    Orchestrates backtesting across all walk-forward splits
    """

    def __init__(self, base_dir: str = 'walkforward_results'):
        self.base_dir = base_dir
        self.results = []

    def load_split_data(self, dataset_name: str, split_num: int) -> tuple:
        """Load train and test data for a split"""
        dataset_dir = f'{self.base_dir}/{dataset_name}'
        train_file = f'{dataset_dir}/split_{split_num}_train.csv'
        test_file = f'{dataset_dir}/split_{split_num}_test.csv'

        train_data = pd.read_csv(train_file, index_col='Date', parse_dates=True)
        test_data = pd.read_csv(test_file, index_col='Date', parse_dates=True)

        return train_data, test_data

    def run_dataset_backtests(self, dataset_name: str, max_splits: int = None) -> List[Dict]:
        """
        Run backtests on all splits for a dataset

        Parameters:
        -----------
        dataset_name : str
            Dataset directory name
        max_splits : int, optional
            Maximum number of splits to test (None = all)

        Returns:
        --------
        List[Dict] : Results for all splits
        """
        print(f"\n{'=' * 80}")
        print(f"BACKTESTING DATASET: {dataset_name}")
        print(f"{'=' * 80}")

        # Count available splits
        dataset_dir = f'{self.base_dir}/{dataset_name}'
        split_files = [f for f in os.listdir(dataset_dir) if f.startswith('split_') and f.endswith('_train.csv')]
        num_splits = len(split_files)

        if max_splits:
            num_splits = min(num_splits, max_splits)

        print(f"Found {num_splits} splits to backtest")

        dataset_results = []

        for split_num in range(1, num_splits + 1):
            try:
                # Load data
                train_data, test_data = self.load_split_data(dataset_name, split_num)

                # Run backtest
                result = run_momentum_backtest(train_data, test_data, split_num, dataset_name)

                dataset_results.append(result)

            except Exception as e:
                print(f"Error processing split {split_num}: {e}")
                continue

        return dataset_results

    def aggregate_results(self, results: List[Dict]) -> Dict:
        """
        Aggregate metrics across all splits

        Parameters:
        -----------
        results : List[Dict]
            Results from all splits

        Returns:
        --------
        Dict : Aggregated metrics and statistics
        """
        if not results:
            return {}

        # Extract metrics from all splits
        metrics_list = [r['metrics'] for r in results if r.get('metrics')]

        # Calculate aggregate statistics
        aggregate = {
            'total_splits': len(results),
            'avg_total_return': np.mean([m.get('total_return', 0) for m in metrics_list]),
            'median_total_return': np.median([m.get('total_return', 0) for m in metrics_list]),
            'std_total_return': np.std([m.get('total_return', 0) for m in metrics_list]),
            'avg_sharpe_ratio': np.mean([m.get('sharpe_ratio', 0) for m in metrics_list]),
            'median_sharpe_ratio': np.median([m.get('sharpe_ratio', 0) for m in metrics_list]),
            'avg_win_rate': np.mean([m.get('win_rate', 0) for m in metrics_list]),
            'avg_max_drawdown': np.mean([m.get('max_drawdown', 0) for m in metrics_list]),
            'max_max_drawdown': max([m.get('max_drawdown', 0) for m in metrics_list]),
            'avg_profit_factor': np.mean([m.get('profit_factor', 0) for m in metrics_list
                                          if m.get('profit_factor', 0) > 0]),
            'total_trades': sum([m.get('total_trades', 0) for m in metrics_list]),
            'positive_returns_pct': (sum([1 for m in metrics_list if m.get('total_return', 0) > 0])
                                     / len(metrics_list) * 100),
            'sharpe_above_1_pct': (sum([1 for m in metrics_list if m.get('sharpe_ratio', 0) > 1.0])
                                   / len(metrics_list) * 100),
        }

        # Consistency metrics
        returns = [m.get('total_return', 0) for m in metrics_list]
        sharpes = [m.get('sharpe_ratio', 0) for m in metrics_list]
        [m.get('win_rate', 0) for m in metrics_list]

        aggregate['return_consistency'] = 1 - (np.std(returns) / (abs(np.mean(returns)) + 1e-6))
        aggregate['sharpe_consistency'] = 1 - (np.std(sharpes) / (abs(np.mean(sharpes)) + 1e-6))

        return aggregate

    def generate_comparison_report(self, dataset1_results: List[Dict],
                                   dataset2_results: List[Dict]) -> pd.DataFrame:
        """
        Generate comparison report between two datasets

        Parameters:
        -----------
        dataset1_results : List[Dict]
            Results from dataset 1
        dataset2_results : List[Dict]
            Results from dataset 2

        Returns:
        --------
        pd.DataFrame : Comparison table
        """
        agg1 = self.aggregate_results(dataset1_results)
        agg2 = self.aggregate_results(dataset2_results)

        comparison_data = {
            'Metric': [
                'Total Splits',
                'Total Trades',
                'Avg Return (%)',
                'Median Return (%)',
                'Std Dev Return (%)',
                'Avg Sharpe Ratio',
                'Median Sharpe Ratio',
                'Avg Win Rate (%)',
                'Avg Max Drawdown (%)',
                'Max Max Drawdown (%)',
                'Avg Profit Factor',
                'Positive Returns (%)',
                'Sharpe > 1.0 (%)',
                'Return Consistency',
                'Sharpe Consistency'
            ],
            'Dataset_1_Recent': [
                agg1.get('total_splits', 0),
                agg1.get('total_trades', 0),
                f"{agg1.get('avg_total_return', 0) * 100:.2f}",
                f"{agg1.get('median_total_return', 0) * 100:.2f}",
                f"{agg1.get('std_total_return', 0) * 100:.2f}",
                f"{agg1.get('avg_sharpe_ratio', 0):.3f}",
                f"{agg1.get('median_sharpe_ratio', 0):.3f}",
                f"{agg1.get('avg_win_rate', 0) * 100:.2f}",
                f"{agg1.get('avg_max_drawdown', 0) * 100:.2f}",
                f"{agg1.get('max_max_drawdown', 0) * 100:.2f}",
                f"{agg1.get('avg_profit_factor', 0):.2f}",
                f"{agg1.get('positive_returns_pct', 0):.1f}",
                f"{agg1.get('sharpe_above_1_pct', 0):.1f}",
                f"{agg1.get('return_consistency', 0):.3f}",
                f"{agg1.get('sharpe_consistency', 0):.3f}"
            ],
            'Dataset_2_Historical': [
                agg2.get('total_splits', 0),
                agg2.get('total_trades', 0),
                f"{agg2.get('avg_total_return', 0) * 100:.2f}",
                f"{agg2.get('median_total_return', 0) * 100:.2f}",
                f"{agg2.get('std_total_return', 0) * 100:.2f}",
                f"{agg2.get('avg_sharpe_ratio', 0):.3f}",
                f"{agg2.get('median_sharpe_ratio', 0):.3f}",
                f"{agg2.get('avg_win_rate', 0) * 100:.2f}",
                f"{agg2.get('avg_max_drawdown', 0) * 100:.2f}",
                f"{agg2.get('max_max_drawdown', 0) * 100:.2f}",
                f"{agg2.get('avg_profit_factor', 0):.2f}",
                f"{agg2.get('positive_returns_pct', 0):.1f}",
                f"{agg2.get('sharpe_above_1_pct', 0):.1f}",
                f"{agg2.get('return_consistency', 0):.3f}",
                f"{agg2.get('sharpe_consistency', 0):.3f}"
            ]
        }

        return pd.DataFrame(comparison_data)

    def save_results(self, all_results: Dict, comparison_df: pd.DataFrame):
        """Save all results to files"""
        output_dir = 'walkforward_results/backtest_results'
        os.makedirs(output_dir, exist_ok=True)

        # Save detailed results as JSON
        results_file = f'{output_dir}/detailed_results.json'
        with open(results_file, 'w') as f:
            # Convert non-serializable objects
            serializable_results = {}
            for dataset, results in all_results.items():
                serializable_results[dataset] = []
                for result in results:
                    r = result.copy()
                    r.pop('equity_curve', None)  # Remove Series object
                    serializable_results[dataset].append(r)

            json.dump(serializable_results, f, indent=2, default=str)

        print(f"\n✓ Saved detailed results to: {results_file}")

        # Save comparison report
        comparison_file = f'{output_dir}/comparison_report.csv'
        comparison_df.to_csv(comparison_file, index=False)
        print(f"✓ Saved comparison report to: {comparison_file}")

        # Save summary
        summary_file = f'{output_dir}/summary.txt'
        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("WALK-FORWARD BACKTEST SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(comparison_df.to_string(index=False))
            f.write("\n\n" + "=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        print(f"✓ Saved summary to: {summary_file}")


def main():
    """
    Main execution: Run all backtests and generate reports
    """
    print("=" * 80)
    print("80/20 WALK-FORWARD MOMENTUM STRATEGY BACKTEST")
    print("=" * 80)

    runner = WalkForwardBacktestRunner()

    # Run backtests on both datasets
    print("\n[1/3] Running backtests on Dataset 1 (Recent: 2020-2025)...")
    dataset1_results = runner.run_dataset_backtests('Dataset_1_Recent', max_splits=None)

    print("\n[2/3] Running backtests on Dataset 2 (Historical: 2010-2019)...")
    dataset2_results = runner.run_dataset_backtests('Dataset_2_Historical', max_splits=None)

    # Generate comparison report
    print("\n[3/3] Generating comparison report...")
    comparison_df = runner.generate_comparison_report(dataset1_results, dataset2_results)

    # Display comparison
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    print(comparison_df.to_string(index=False))

    # Save results
    all_results = {
        'Dataset_1_Recent': dataset1_results,
        'Dataset_2_Historical': dataset2_results
    }
    runner.save_results(all_results, comparison_df)

    # Final validation checklist
    print("\n" + "=" * 80)
    print("VALIDATION CHECKLIST")
    print("=" * 80)

    agg1 = runner.aggregate_results(dataset1_results)
    agg2 = runner.aggregate_results(dataset2_results)

    checks = [
        ("Minimum 3 walk-forward periods per dataset",
         agg1['total_splits'] >= 3 and agg2['total_splits'] >= 3),
        ("Average Sharpe ratio > 1.0 in at least one dataset",
         agg1['avg_sharpe_ratio'] > 1.0 or agg2['avg_sharpe_ratio'] > 1.0),
        ("Positive returns in >50% of splits (Dataset 1)",
         agg1['positive_returns_pct'] > 50),
        ("Positive returns in >50% of splits (Dataset 2)",
         agg2['positive_returns_pct'] > 50),
        ("Max drawdown < 20% on average (Dataset 1)",
         agg1['avg_max_drawdown'] < 0.20),
        ("Max drawdown < 20% on average (Dataset 2)",
         agg2['avg_max_drawdown'] < 0.20),
    ]

    for check_name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {check_name}")

    passed_checks = sum([1 for _, passed in checks if passed])
    print(f"\nOverall: {passed_checks}/{len(checks)} checks passed")

    if passed_checks == len(checks):
        print("\n🎉 Strategy validation PASSED! Consider paper trading.")
    elif passed_checks >= len(checks) * 0.7:
        print("\n⚠️  Strategy shows promise but needs refinement.")
    else:
        print("\n❌ Strategy needs significant improvement before deployment.")

    print("\n" + "=" * 80)
    print("✓ Complete! Check 'walkforward_results/backtest_results/' for detailed output")
    print("=" * 80)


if __name__ == "__main__":
    main()
