#!/usr/bin/env python3
"""
Walk-Forward 80/20 Testing Plan for Momentum Strategy

Design Principles:
------------------
1. Sequential time-series splits (NO random shuffling)
2. 8 weeks training + 2 weeks testing = 10-week rolling window
3. 80/20 ratio maintained throughout
4. Two distinct datasets for robust validation
5. Proper accounting for trading days vs calendar days

Implementation:
--------------
- Dataset 1: Recent period (2020-2025) - Modern market conditions
- Dataset 2: Historical period (2010-2019) - Different market regime
- Walk-forward windows: Non-overlapping 10-week periods
- Trading days: ~252 days/year, ~21 days/month, ~5 days/week
"""

import pandas as pd
from datetime import timedelta
import json
from typing import List, Dict, Tuple
import os


class WalkForwardDataSplitter:
    """
    Creates sequential 80/20 train/test splits for time-series data

    Key Features:
    - Respects temporal ordering (critical for trading strategies)
    - Configurable train/test periods
    - Generates multiple non-overlapping windows
    - Validates data integrity
    """

    def __init__(self, data: pd.DataFrame, train_weeks: int = 8, test_weeks: int = 2):
        """
        Initialize splitter

        Parameters:
        -----------
        data : pd.DataFrame
            Time-series data with DatetimeIndex
        train_weeks : int
            Training period in weeks (default: 8 = 80%)
        test_weeks : int
            Testing period in weeks (default: 2 = 20%)
        """
        self.data = data.copy()
        self.train_weeks = train_weeks
        self.test_weeks = test_weeks
        self.window_weeks = train_weeks + test_weeks

        # Convert weeks to approximate trading days (5 days/week)
        self.train_days = train_weeks * 5
        self.test_days = test_weeks * 5
        self.window_days = self.window_weeks * 5

        print("Walk-Forward Configuration:")
        print(f"  Training: {train_weeks} weeks (~{self.train_days} trading days)")
        print(f"  Testing:  {test_weeks} weeks (~{self.test_days} trading days)")
        print(f"  Window:   {self.window_weeks} weeks (~{self.window_days} trading days)")
        print(f"  Ratio:    {train_weeks}:{test_weeks} = "
              f"{train_weeks / (train_weeks + test_weeks) * 100:.0f}/"
              f"{test_weeks / (train_weeks + test_weeks) * 100:.0f}")

    def generate_splits(self, step_weeks: int = None) -> List[Dict]:
        """
        Generate sequential train/test splits

        Parameters:
        -----------
        step_weeks : int, optional
            How many weeks to move forward each iteration
            If None, uses non-overlapping windows (step = window_weeks)

        Returns:
        --------
        List[Dict] : List of split definitions with train/test data
        """
        if step_weeks is None:
            step_weeks = self.window_weeks  # Non-overlapping

        step_days = step_weeks * 5

        splits = []
        start_idx = 0
        total_rows = len(self.data)

        print(f"\nGenerating splits (step size: {step_weeks} weeks)...")

        while start_idx + self.window_days <= total_rows:
            # Define train and test indices
            train_start = start_idx
            train_end = start_idx + self.train_days
            test_start = train_end
            test_end = test_start + self.test_days

            # Extract data
            train_data = self.data.iloc[train_start:train_end]
            test_data = self.data.iloc[test_start:test_end]

            # Create split info
            split = {
                'split_num': len(splits) + 1,
                'train_data': train_data,
                'test_data': test_data,
                'train_start': train_data.index[0],
                'train_end': train_data.index[-1],
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1],
                'train_rows': len(train_data),
                'test_rows': len(test_data),
                'actual_ratio': len(train_data) / (len(train_data) + len(test_data))
            }

            splits.append(split)

            # Move forward
            start_idx += step_days

        print(f"✓ Generated {len(splits)} walk-forward splits")
        return splits

    def validate_splits(self, splits: List[Dict]) -> Dict:
        """
        Validate splits for proper temporal ordering and data integrity
        """
        issues = []

        for i, split in enumerate(splits):
            # Check temporal ordering
            if split['train_end'] >= split['test_start']:
                issues.append(f"Split {i + 1}: Train end >= Test start")

            # Check for data leakage
            train_dates = set(split['train_data'].index)
            test_dates = set(split['test_data'].index)
            overlap = train_dates.intersection(test_dates)
            if overlap:
                issues.append(f"Split {i + 1}: {len(overlap)} dates overlap between train/test")

            # Check ratio
            expected_ratio = self.train_weeks / self.window_weeks
            actual_ratio = split['actual_ratio']
            if abs(actual_ratio - expected_ratio) > 0.05:  # 5% tolerance
                issues.append(f"Split {i + 1}: Ratio {actual_ratio:.2f} differs from expected {expected_ratio:.2f}")

        # Check for gaps between splits
        for i in range(len(splits) - 1):
            current_end = splits[i]['test_end']
            next_start = splits[i + 1]['train_start']
            if next_start > current_end + timedelta(days=7):  # Allow for weekends
                issues.append(f"Gap between split {i + 1} and {i + 2}")

        validation = {
            'valid': len(issues) == 0,
            'issues': issues,
            'total_splits': len(splits),
            'date_range': f"{splits[0]['train_start']} to {splits[-1]['test_end']}" if splits else "N/A"
        }

        if validation['valid']:
            print("✓ All splits validated successfully")
        else:
            print(f"⚠ Found {len(issues)} validation issues:")
            for issue in issues:
                print(f"  - {issue}")

        return validation

    def summary_report(self, splits: List[Dict]) -> pd.DataFrame:
        """Generate summary report of all splits"""
        summary_data = []

        for split in splits:
            summary_data.append({
                'Split': split['split_num'],
                'Train Start': split['train_start'].strftime('%Y-%m-%d'),
                'Train End': split['train_end'].strftime('%Y-%m-%d'),
                'Test Start': split['test_start'].strftime('%Y-%m-%d'),
                'Test End': split['test_end'].strftime('%Y-%m-%d'),
                'Train Rows': split['train_rows'],
                'Test Rows': split['test_rows'],
                'Ratio': f"{split['actual_ratio']:.1%}"
            })

        return pd.DataFrame(summary_data)


def prepare_two_datasets(data_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create two distinct datasets from S&P 500 data

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame] : (dataset1, dataset2)
        - Dataset 1: Recent period (2020-2025)
        - Dataset 2: Historical period (2010-2019)
    """
    print("Loading S&P 500 data...")
    df = pd.read_csv(data_file, index_col='Date', parse_dates=True)
    df = df.sort_index()

    print(f"Total data: {len(df)} rows from {df.index[0]} to {df.index[-1]}")

    # Dataset 1: Recent period (2020-2025)
    dataset1 = df[df.index >= '2020-01-01']
    print(f"\nDataset 1 (Recent): {len(dataset1)} rows from {dataset1.index[0]} to {dataset1.index[-1]}")

    # Dataset 2: Historical period (2010-2019)
    dataset2 = df[(df.index >= '2010-01-01') & (df.index < '2020-01-01')]
    print(f"Dataset 2 (Historical): {len(dataset2)} rows from {dataset2.index[0]} to {dataset2.index[-1]}")

    return dataset1, dataset2


def main():
    """
    Main execution: Create and validate 80/20 walk-forward splits
    """
    print("=" * 80)
    print("WALK-FORWARD 80/20 DATA TESTING PLAN")
    print("=" * 80)

    # Load and prepare datasets
    dataset1, dataset2 = prepare_two_datasets('sp500_historical_data.csv')

    # Create results directory
    os.makedirs('walkforward_results', exist_ok=True)

    # Process both datasets
    datasets = {
        'Dataset_1_Recent': dataset1,
        'Dataset_2_Historical': dataset2
    }

    all_results = {}

    for name, data in datasets.items():
        print("\n" + "=" * 80)
        print(f"Processing: {name}")
        print("=" * 80)

        # Create splitter (8 weeks train, 2 weeks test)
        splitter = WalkForwardDataSplitter(data, train_weeks=8, test_weeks=2)

        # Generate non-overlapping splits
        splits = splitter.generate_splits(step_weeks=None)  # Non-overlapping

        # Validate splits
        validation = splitter.validate_splits(splits)

        # Generate summary report
        summary = splitter.summary_report(splits)
        print("\n" + summary.to_string(index=False))

        # Save splits to disk
        output_dir = f'walkforward_results/{name}'
        os.makedirs(output_dir, exist_ok=True)

        for split in splits:
            split_num = split['split_num']
            split['train_data'].to_csv(f'{output_dir}/split_{split_num}_train.csv')
            split['test_data'].to_csv(f'{output_dir}/split_{split_num}_test.csv')

        # Save summary
        summary.to_csv(f'{output_dir}/splits_summary.csv', index=False)

        # Save validation report
        with open(f'{output_dir}/validation_report.json', 'w') as f:
            json.dump(validation, f, indent=2, default=str)

        all_results[name] = {
            'splits': len(splits),
            'validation': validation,
            'summary': summary
        }

        print(f"\n✓ Saved {len(splits)} splits to: {output_dir}/")

    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for name, results in all_results.items():
        print(f"\n{name}:")
        print(f"  Total Splits: {results['splits']}")
        print(f"  Validation: {'✓ PASSED' if results['validation']['valid'] else '✗ FAILED'}")
        print(f"  Date Range: {results['validation']['date_range']}")

    print("\n✓ Walk-forward data preparation complete!")
    print("Next step: Run momentum strategy backtests on these splits")


if __name__ == "__main__":
    main()
