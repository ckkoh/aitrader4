#!/usr/bin/env python3
"""
Momentum Strategy for 80/20 Walk-Forward Testing

Strategy Logic:
--------------
1. Calculate price momentum over multiple timeframes
2. Use rate of change (ROC) and moving average crossovers
3. ATR-based stop losses and take profits
4. Only trade when momentum is strong and confirmed

Entry Conditions:
- Short-term SMA crosses above long-term SMA (bullish momentum)
- Rate of change is positive and above threshold
- Price > 20-day SMA (trend filter)

Exit Conditions:
- Stop loss hit (2x ATR)
- Take profit hit (3x ATR)
- SMA crossover reverses
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict
import sys
import os

# Import backtesting framework
from backtesting_engine import Strategy, BacktestEngine, BacktestConfig


class MomentumStrategy(Strategy):
    """
    Simple momentum strategy using moving averages and rate of change

    Parameters:
    -----------
    short_window : int
        Short-term moving average period (default: 10 days)
    long_window : int
        Long-term moving average period (default: 20 days)
    roc_period : int
        Rate of change lookback period (default: 10 days)
    roc_threshold : float
        Minimum ROC % to trigger entry (default: 0.5%)
    atr_period : int
        ATR period for stop loss/take profit (default: 14 days)
    stop_loss_atr_multiplier : float
        Stop loss distance in ATR units (default: 2.0)
    take_profit_atr_multiplier : float
        Take profit distance in ATR units (default: 3.0)
    """

    def __init__(self,
                 name: str = "MomentumStrategy",
                 short_window: int = 10,
                 long_window: int = 20,
                 roc_period: int = 10,
                 roc_threshold: float = 0.5,
                 atr_period: int = 14,
                 stop_loss_atr_multiplier: float = 2.0,
                 take_profit_atr_multiplier: float = 3.0):

        super().__init__(name)
        self.short_window = short_window
        self.long_window = long_window
        self.roc_period = roc_period
        self.roc_threshold = roc_threshold
        self.atr_period = atr_period
        self.stop_loss_atr_multiplier = stop_loss_atr_multiplier
        self.take_profit_atr_multiplier = take_profit_atr_multiplier

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        df = data.copy()

        # Moving averages
        df['sma_short'] = df['close'].rolling(window=self.short_window).mean()
        df['sma_long'] = df['close'].rolling(window=self.long_window).mean()

        # Rate of change (momentum)
        df['roc'] = ((df['close'] - df['close'].shift(self.roc_period)) /
                     df['close'].shift(self.roc_period)) * 100

        # ATR for stop loss/take profit
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(window=self.atr_period).mean()

        # Trend strength
        df['trend_strength'] = (df['sma_short'] - df['sma_long']) / df['sma_long'] * 100

        return df

    def generate_signals(self, data: pd.DataFrame, timestamp: datetime) -> List[Dict]:
        """
        Generate trading signals based on momentum

        Args:
            data: Historical price data up to timestamp
            timestamp: Current timestamp

        Returns:
            List of signal dictionaries
        """
        signals = []

        # Need minimum data for indicators
        if len(data) < max(self.long_window, self.roc_period, self.atr_period) + 1:
            return signals

        # Calculate indicators
        df = self.calculate_indicators(data)

        # Get current and previous values
        current = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else current

        # Check for valid indicator values
        if pd.isna(current['sma_short']) or pd.isna(current['atr']):
            return signals

        current_price = current['close']
        atr = current['atr']

        # Calculate stop loss and take profit levels
        stop_loss = current_price - (atr * self.stop_loss_atr_multiplier)
        take_profit = current_price + (atr * self.take_profit_atr_multiplier)

        # Check if we have an open position
        has_position = len(self.positions) > 0

        if not has_position:
            # ENTRY LOGIC: Simplified bidirectional momentum strategy

            # Calculate conditions
            sma_short_above_long = current['sma_short'] > current['sma_long']
            strong_positive_momentum = current['roc'] > self.roc_threshold
            strong_negative_momentum = current['roc'] < -self.roc_threshold
            current_price > current['sma_long']
            current_price < current['sma_long']

            # BUY SIGNAL: Strong positive momentum in an uptrend
            if strong_positive_momentum and sma_short_above_long:
                signals.append({
                    'instrument': 'SP500',
                    'action': 'buy',
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'reason': f"Bullish_Momentum_ROC={current['roc']:.2f}%"
                })

            # SELL (SHORT) SIGNAL: Strong negative momentum in a downtrend
            elif strong_negative_momentum and not sma_short_above_long:
                # For short positions, invert stop loss and take profit
                short_stop_loss = current_price + (atr * self.stop_loss_atr_multiplier)
                short_take_profit = current_price - (atr * self.take_profit_atr_multiplier)

                signals.append({
                    'instrument': 'SP500',
                    'action': 'sell',
                    'stop_loss': short_stop_loss,
                    'take_profit': short_take_profit,
                    'reason': f"Bearish_Momentum_ROC={current['roc']:.2f}%"
                })

        else:
            # EXIT LOGIC: Check exit conditions

            # Exit condition 1: Bearish crossover (short crosses below long)
            bearish_crossover = (
                current['sma_short'] < current['sma_long'] and
                previous['sma_short'] >= previous['sma_long']
            )

            # Exit condition 2: Negative ROC (momentum reversal)
            momentum_reversal = current['roc'] < -self.roc_threshold

            # Exit condition 3: Trend strength turned negative
            trend_reversal = current['trend_strength'] < -0.5

            # Close position if any exit condition is met
            if bearish_crossover or momentum_reversal or trend_reversal:
                reason = "SMA_Crossover" if bearish_crossover else \
                    "Momentum_Reversal" if momentum_reversal else "Trend_Reversal"

                signals.append({
                    'instrument': 'SP500',
                    'action': 'close',
                    'reason': reason
                })

        return signals


def run_momentum_backtest(train_data: pd.DataFrame,
                          test_data: pd.DataFrame,
                          split_num: int,
                          dataset_name: str) -> Dict:
    """
    Run momentum strategy backtest on a single train/test split

    Parameters:
    -----------
    train_data : pd.DataFrame
        Training data (used for indicator calculation)
    test_data : pd.DataFrame
        Testing data (out-of-sample performance)
    split_num : int
        Split number
    dataset_name : str
        Dataset identifier

    Returns:
    --------
    Dict : Backtest results
    """
    print(f"\n{'=' * 60}")
    print(f"Split {split_num} - {dataset_name}")
    print(f"Train: {train_data.index[0]} to {train_data.index[-1]} ({len(train_data)} rows)")
    print(f"Test:  {test_data.index[0]} to {test_data.index[-1]} ({len(test_data)} rows)")
    print(f"{'=' * 60}")

    # Create strategy
    strategy = MomentumStrategy(
        short_window=10,
        long_window=20,
        roc_period=10,
        roc_threshold=0.5,
        atr_period=14,
        stop_loss_atr_multiplier=2.0,
        take_profit_atr_multiplier=3.0
    )

    # Configure backtest
    config = BacktestConfig(
        initial_capital=10000.0,
        commission_pct=0.0001,  # 1 pip = 0.01%
        slippage_pct=0.0001,
        position_size_pct=0.02,  # 2% risk per trade
        max_positions=1,
        leverage=1.0,
        position_sizing_method='fixed_pct',  # Fixed percentage risk-based sizing
        max_daily_loss_pct=0.05,
        max_drawdown_pct=0.20
    )

    # Create backtest engine
    engine = BacktestEngine(config)

    # CRITICAL FIX: Combine train + test data for indicator calculation
    # But only trade during test period (out-of-sample)
    combined_data = pd.concat([train_data, test_data])
    test_start_date = test_data.index[0]

    # Run backtest on COMBINED data, but only trade during TEST period
    # Use trading_start_date to ensure trades only execute during test period
    results = engine.run_backtest(strategy, combined_data, trading_start_date=test_start_date)

    # Extract key metrics
    metrics = results['metrics']

    print("\nResults:")
    print(f"  Total Trades: {metrics.get('total_trades', 0)}")
    print(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
    print(f"  Total Return: {metrics.get('total_return', 0):.2%}")
    print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
    print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
    print(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")

    return {
        'split_num': split_num,
        'dataset': dataset_name,
        'train_start': str(train_data.index[0]),
        'train_end': str(train_data.index[-1]),
        'test_start': str(test_data.index[0]),
        'test_end': str(test_data.index[-1]),
        'metrics': metrics,
        'trades': results.get('trades', []),
        'equity_curve': results.get('equity_curve', pd.Series())
    }


if __name__ == "__main__":
    print("=" * 80)
    print("MOMENTUM STRATEGY - 80/20 WALK-FORWARD BACKTEST")
    print("=" * 80)

    # Test with a single split as demonstration
    dataset_name = "Dataset_1_Recent"
    split_num = 1

    # Load train/test data
    base_dir = f'walkforward_results/{dataset_name}'
    train_file = f'{base_dir}/split_{split_num}_train.csv'
    test_file = f'{base_dir}/split_{split_num}_test.csv'

    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print("Error: Split files not found. Please run walkforward_80_20_plan.py first.")
        sys.exit(1)

    train_data = pd.read_csv(train_file, index_col='Date', parse_dates=True)
    test_data = pd.read_csv(test_file, index_col='Date', parse_dates=True)

    # Run single backtest
    results = run_momentum_backtest(train_data, test_data, split_num, dataset_name)

    print("\n✓ Single split backtest complete")
    print("Run 'python3 run_all_splits_backtest.py' to backtest all splits")
