#!/usr/bin/env python3
"""
Test with DEBUG logging enabled
"""

import pandas as pd
import logging
from momentum_strategy_80_20 import run_momentum_backtest

# Enable DEBUG logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')

# Load split data
dataset_name = "Dataset_1_Recent"
split_num = 1

train_file = f'walkforward_results/{dataset_name}/split_{split_num}_train.csv'
test_file = f'walkforward_results/{dataset_name}/split_{split_num}_test.csv'

train_data = pd.read_csv(train_file, index_col='Date', parse_dates=True)
test_data = pd.read_csv(test_file, index_col='Date', parse_dates=True)

# Run backtest
results = run_momentum_backtest(train_data, test_data, split_num, dataset_name)

print(f"\nTotal Trades: {results['metrics']['total_trades']}")
