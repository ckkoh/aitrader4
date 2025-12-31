"""
Professional Backtesting Engine for Forex Trading
Includes realistic transaction costs, position sizing, and walk-forward analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class PositionSide(Enum):
    """Position direction"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class Trade:
    """Represents a completed trade"""
    entry_time: datetime
    exit_time: datetime
    instrument: str
    direction: str
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_percent: float
    commission: float
    slippage: float
    strategy: str
    entry_reason: str = ""
    exit_reason: str = ""
    max_adverse_excursion: float = 0.0
    max_favorable_excursion: float = 0.0


@dataclass
class Position:
    """Represents an open position"""
    entry_time: datetime
    instrument: str
    direction: PositionSide
    entry_price: float
    size: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0
    max_adverse_excursion: float = 0.0
    max_favorable_excursion: float = 0.0


@dataclass
class BacktestConfig:
    """Configuration for backtest"""
    initial_capital: float = 10000.0
    commission_pct: float = 0.0001  # 1 pip = 0.01%
    slippage_pct: float = 0.0001    # 1 pip slippage
    position_size_pct: float = 0.02  # 2% risk per trade
    max_position_value_pct: float = 0.02  # Max 2% of capital per position (notional value)
    max_positions: int = 1
    leverage: float = 1.0
    risk_free_rate: float = 0.02  # For Sharpe calculation

    # Position sizing methods: 'fixed_pct', 'volatility', 'kelly'
    position_sizing_method: str = 'fixed_pct'

    # Risk management
    max_daily_loss_pct: float = 0.05  # 5% max daily loss
    max_drawdown_pct: float = 0.20    # 20% max drawdown

    # Realism features
    use_bid_ask_spread: bool = True
    spread_pips: float = 1.0  # Average spread in pips


class BacktestMetrics:
    """Calculate comprehensive backtest metrics"""

    @staticmethod
    def calculate_metrics(trades: List[Trade], equity_curve: pd.Series,
                          initial_capital: float, config: BacktestConfig) -> Dict:
        """Calculate all performance metrics"""

        if not trades:
            return BacktestMetrics._empty_metrics()

        trades_df = pd.DataFrame([vars(t) for t in trades])

        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # P&L metrics
        total_pnl = trades_df['pnl'].sum()
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())

        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if losing_trades > 0 else 0

        # Advanced metrics
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        # Returns calculation
        returns = equity_curve.pct_change().dropna()

        # Sharpe Ratio
        excess_returns = returns - (config.risk_free_rate / 252)
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if len(returns) > 0 else 0

        # Sortino Ratio (only downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_std if downside_std > 0 else 0

        # Drawdown analysis
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        max_drawdown_duration = BacktestMetrics._calculate_max_dd_duration(drawdown)

        # Current drawdown
        current_drawdown = abs(drawdown.iloc[-1]) if len(drawdown) > 0 else 0

        # Calmar Ratio (return / max drawdown)
        total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]
        calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0

        # Trade duration
        trades_df['duration'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 3600
        avg_trade_duration = trades_df['duration'].mean()

        # Consecutive wins/losses
        trades_df['win'] = (trades_df['pnl'] > 0).astype(int)
        max_consecutive_wins = BacktestMetrics._max_consecutive(trades_df['win'].values, 1)
        max_consecutive_losses = BacktestMetrics._max_consecutive(trades_df['win'].values, 0)

        # MAE/MFE analysis
        avg_mae = trades_df['max_adverse_excursion'].mean()
        avg_mfe = trades_df['max_favorable_excursion'].mean()

        # Recovery factor
        recovery_factor = total_pnl / abs(max_drawdown * initial_capital) if max_drawdown > 0 else 0

        # Risk-adjusted return
        annual_return = total_return * (252 / len(equity_curve)) if len(equity_curve) > 0 else 0

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return_pct': total_return * 100,
            'annual_return_pct': annual_return * 100,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown_pct': max_drawdown * 100,
            'current_drawdown_pct': current_drawdown * 100,
            'max_drawdown_duration_days': max_drawdown_duration,
            'avg_trade_duration_hours': avg_trade_duration,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'avg_mae': avg_mae,
            'avg_mfe': avg_mfe,
            'recovery_factor': recovery_factor,
            'final_equity': equity_curve.iloc[-1]
        }

    @staticmethod
    def _empty_metrics() -> Dict:
        """Return empty metrics dictionary"""
        return {key: 0 for key in [
            'total_trades', 'winning_trades', 'losing_trades', 'win_rate',
            'total_pnl', 'total_return_pct', 'annual_return_pct', 'gross_profit',
            'gross_loss', 'avg_win', 'avg_loss', 'profit_factor', 'expectancy',
            'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'max_drawdown_pct',
            'current_drawdown_pct', 'max_drawdown_duration_days',
            'avg_trade_duration_hours', 'max_consecutive_wins',
            'max_consecutive_losses', 'avg_mae', 'avg_mfe', 'recovery_factor',
            'final_equity'
        ]}

    @staticmethod
    def _calculate_max_dd_duration(drawdown: pd.Series) -> int:
        """Calculate maximum drawdown duration in days"""
        in_drawdown = drawdown < 0
        drawdown_periods = (in_drawdown != in_drawdown.shift()).cumsum()

        if not in_drawdown.any():
            return 0

        durations = drawdown_periods[in_drawdown].value_counts()
        return durations.max() if len(durations) > 0 else 0

    @staticmethod
    def _max_consecutive(arr: np.ndarray, value: int) -> int:
        """Calculate maximum consecutive occurrences"""
        max_count = 0
        current_count = 0

        for item in arr:
            if item == value:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0

        return max_count


class PositionSizer:
    """Calculate position sizes using various methods"""

    @staticmethod
    def fixed_percentage(capital: float, risk_pct: float, entry_price: float,
                         stop_loss: float, max_position_value_pct: float = 0.05) -> float:
        """
        Fixed percentage risk per trade

        Args:
            capital: Current account capital
            risk_pct: Percentage of capital to risk (0.02 = 2%)
            entry_price: Entry price for the trade
            stop_loss: Stop loss price
            max_position_value_pct: Max position value as % of capital (0.10 = 10%)

        Returns:
            Position size (capped by max position value)
        """
        risk_amount = capital * risk_pct
        price_risk = abs(entry_price - stop_loss)

        if price_risk == 0:
            return 0

        position_size = risk_amount / price_risk

        # Cap position size by maximum notional value
        max_notional_value = capital * max_position_value_pct
        max_position_size = max_notional_value / entry_price
        position_size = min(position_size, max_position_size)

        return position_size

    @staticmethod
    def volatility_adjusted(capital: float, risk_pct: float, atr: float,
                            entry_price: float, atr_multiplier: float = 2.0,
                            max_position_value_pct: float = 0.05) -> float:
        """
        Position sizing based on ATR volatility

        Args:
            capital: Current account capital
            risk_pct: Percentage of capital to risk
            atr: Average True Range
            entry_price: Entry price
            atr_multiplier: ATR multiplier for stop distance
            max_position_value_pct: Max position value as % of capital (0.10 = 10%)

        Returns:
            Position size (capped by max position value)
        """
        risk_amount = capital * risk_pct
        stop_distance = atr * atr_multiplier

        if stop_distance == 0:
            return 0

        position_size = risk_amount / stop_distance

        # Cap position size by maximum notional value
        max_notional_value = capital * max_position_value_pct
        max_position_size = max_notional_value / entry_price
        position_size = min(position_size, max_position_size)

        return position_size

    @staticmethod
    def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float,
                        fraction: float = 0.5) -> float:
        """
        Kelly Criterion for position sizing

        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade
            avg_loss: Average losing trade (absolute value)
            fraction: Fraction of Kelly to use (0.5 = half Kelly)

        Returns:
            Percentage of capital to risk
        """
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return 0.01  # Default to 1%

        win_loss_ratio = avg_win / avg_loss
        kelly_pct = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio

        # Use fractional Kelly and cap at reasonable limits
        kelly_pct = kelly_pct * fraction
        kelly_pct = np.clip(kelly_pct, 0.001, 0.05)  # 0.1% to 5%

        return kelly_pct


class Strategy(ABC):
    """Base class for trading strategies"""

    def __init__(self, name: str):
        self.name = name
        self.positions: Dict[str, Position] = {}

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, timestamp: datetime) -> List[Dict]:
        """
        Generate trading signals

        Args:
            data: Historical price data up to timestamp
            timestamp: Current timestamp

        Returns:
            List of signal dictionaries with keys:
                - instrument: str
                - action: 'buy', 'sell', or 'close'
                - stop_loss: float (optional)
                - take_profit: float (optional)
                - reason: str (optional)
        """

    def get_position(self, instrument: str) -> Optional[Position]:
        """Get current position for instrument"""
        return self.positions.get(instrument)

    def has_position(self, instrument: str) -> bool:
        """Check if position exists"""
        return instrument in self.positions


class BacktestEngine:
    """
    Main backtesting engine with realistic execution and walk-forward analysis
    """

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.equity_curve = []
        self.trades: List[Trade] = []
        self.current_capital = self.config.initial_capital
        self.current_positions: Dict[str, Position] = {}
        self.daily_pnl = {}
        self.trading_start_date = None

    def run_backtest(self, strategy: Strategy, data: pd.DataFrame,
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None,
                     trading_start_date: Optional[datetime] = None) -> Dict:
        """
        Run backtest on historical data

        Args:
            strategy: Strategy instance to backtest
            data: DataFrame with OHLCV data and timestamp index
            start_date: Start date for DATA (optional, for filtering)
            end_date: End date for DATA (optional, for filtering)
            trading_start_date: Start date for TRADING (signals before this are ignored)

        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Starting backtest for strategy: {strategy.name}")

        # Reset state
        self._reset()

        # Store trading constraints
        self.trading_start_date = trading_start_date

        # Filter data by date range (if needed)
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]

        if len(data) == 0:
            logger.error("No data available for backtesting")
            return self._empty_results()

        # Ensure data is sorted
        data = data.sort_index()

        # Iterate through each timestamp
        for i, (timestamp, row) in enumerate(data.iterrows()):
            # Get historical data up to current point (avoid look-ahead bias)
            historical_data = data.iloc[:i + 1]

            # Check risk limits
            if not self._check_risk_limits(timestamp):
                logger.warning(f"Risk limits exceeded at {timestamp}, halting trading")
                break

            # Update unrealized P&L for open positions
            self._update_positions(timestamp, row)

            # Check for stop loss / take profit hits
            self._check_exit_conditions(timestamp, row)

            # Generate signals
            try:
                signals = strategy.generate_signals(historical_data, timestamp)
                if signals:
                    logger.debug(f"{timestamp}: Generated {len(signals)} signal(s)")
            except Exception as e:
                logger.error(f"Error generating signals at {timestamp}: {e}")
                signals = []

            # Execute signals (only if past trading_start_date)
            for signal in signals:
                if self.trading_start_date and timestamp < self.trading_start_date:
                    logger.debug(f"Skipping signal before trading start date: {timestamp}")
                    continue
                logger.debug(f"Executing signal: {signal}")
                self._execute_signal(signal, timestamp, row, strategy.name)

            # Record equity
            total_equity = self._calculate_total_equity(row)
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': total_equity
            })

            # Track daily P&L
            date_key = timestamp.date()
            if date_key not in self.daily_pnl:
                self.daily_pnl[date_key] = 0

        # Close any remaining positions at end
        self._close_all_positions(data.iloc[-1].name, data.iloc[-1])

        # Calculate metrics
        equity_series = pd.Series(
            [e['equity'] for e in self.equity_curve],
            index=[e['timestamp'] for e in self.equity_curve]
        )

        metrics = BacktestMetrics.calculate_metrics(
            self.trades, equity_series, self.config.initial_capital, self.config
        )

        logger.info(f"Backtest complete. Total trades: {metrics['total_trades']}, "
                    f"Win rate: {metrics['win_rate']:.2%}, "
                    f"Sharpe: {metrics['sharpe_ratio']:.2f}")

        return {
            'metrics': metrics,
            'trades': self.trades,
            'equity_curve': equity_series,
            'strategy_name': strategy.name
        }

    def walk_forward_analysis(self, strategy: Strategy, data: pd.DataFrame,
                              train_period_days: int = 180,
                              test_period_days: int = 60,
                              step_days: int = 60) -> List[Dict]:
        """
        Perform walk-forward analysis

        Args:
            strategy: Strategy to test
            data: Full dataset
            train_period_days: Days for training period
            test_period_days: Days for testing period
            step_days: Days to step forward each iteration

        Returns:
            List of backtest results for each period
        """
        logger.info("Starting walk-forward analysis...")

        results = []
        start_date = data.index[0]
        end_date = data.index[-1]

        current_start = start_date

        while current_start + timedelta(days=train_period_days + test_period_days) <= end_date:
            train_end = current_start + timedelta(days=train_period_days)
            test_start = train_end
            test_end = test_start + timedelta(days=test_period_days)

            logger.info(f"Walk-forward period: Train {current_start.date()} to {train_end.date()}, "
                        f"Test {test_start.date()} to {test_end.date()}")

            # Extract train and test data
            train_data = data[(data.index >= current_start) & (data.index < train_end)]
            test_data = data[(data.index >= test_start) & (data.index < test_end)]

            # Train strategy if it has a train method
            if hasattr(strategy, 'train'):
                try:
                    strategy.train(train_data)
                except Exception as e:
                    logger.error(f"Error training strategy: {e}")

            # Run backtest on test period
            result = self.run_backtest(strategy, test_data, test_start, test_end)
            result['train_start'] = current_start
            result['train_end'] = train_end
            result['test_start'] = test_start
            result['test_end'] = test_end

            results.append(result)

            # Move forward
            current_start = current_start + timedelta(days=step_days)

        # Aggregate results
        self._print_walk_forward_summary(results)

        return results

    def _reset(self):
        """Reset backtest state"""
        self.equity_curve = []
        self.trades = []
        self.current_capital = self.config.initial_capital
        self.current_positions = {}
        self.daily_pnl = {}

    def _execute_signal(self, signal: Dict, timestamp: datetime,
                        current_prices: pd.Series, strategy_name: str):
        """Execute a trading signal"""
        instrument = signal['instrument']
        action = signal['action'].lower()

        # Check if we can open new positions
        if len(self.current_positions) >= self.config.max_positions and action in ['buy', 'sell']:
            return

        if action == 'close':
            if instrument in self.current_positions:
                self._close_position(instrument, timestamp, current_prices, signal.get('reason', 'signal'))

        elif action in ['buy', 'sell']:
            # Don't open if already have position
            if instrument in self.current_positions:
                return

            direction = PositionSide.LONG if action == 'buy' else PositionSide.SHORT

            # Get prices with spread
            if direction == PositionSide.LONG:
                entry_price = self._get_ask_price(current_prices['close'])
            else:
                entry_price = self._get_bid_price(current_prices['close'])

            # Apply slippage
            slippage = self._calculate_slippage(entry_price)
            entry_price += slippage if direction == PositionSide.LONG else -slippage

            # Calculate position size
            stop_loss = signal.get('stop_loss')
            position_size = self._calculate_position_size(
                entry_price,
                stop_loss,
                current_prices.get('atr', None)
            )

            logger.debug(f"Signal: {action} {instrument} @ {entry_price:.2f}, SL: {stop_loss}, Size: {position_size}")

            if position_size <= 0:
                logger.debug(f"Position size {position_size} <= 0, skipping trade")
                return

            # Open position
            position = Position(
                entry_time=timestamp,
                instrument=instrument,
                direction=direction,
                entry_price=entry_price,
                size=position_size,
                stop_loss=signal.get('stop_loss'),
                take_profit=signal.get('take_profit')
            )

            self.current_positions[instrument] = position
            logger.debug(f"Opened {direction.value} position: {instrument} @ {entry_price:.5f}")

    def _close_position(self, instrument: str, timestamp: datetime,
                        current_prices: pd.Series, reason: str = ""):
        """Close an open position"""
        if instrument not in self.current_positions:
            return

        position = self.current_positions[instrument]

        # Get exit price with spread
        if position.direction == PositionSide.LONG:
            exit_price = self._get_bid_price(current_prices['close'])
        else:
            exit_price = self._get_ask_price(current_prices['close'])

        # Apply slippage
        slippage = self._calculate_slippage(exit_price)
        exit_price -= slippage if position.direction == PositionSide.LONG else -slippage

        # Calculate P&L
        if position.direction == PositionSide.LONG:
            pnl = (exit_price - position.entry_price) * position.size
        else:
            pnl = (position.entry_price - exit_price) * position.size

        # Subtract commission
        commission = abs(position.size) * position.entry_price * self.config.commission_pct
        commission += abs(position.size) * exit_price * self.config.commission_pct

        pnl -= commission

        # Calculate P&L percentage
        position_value = abs(position.size) * position.entry_price
        pnl_percent = (pnl / position_value) * 100 if position_value > 0 else 0

        # Update capital
        self.current_capital += pnl

        # Create trade record
        trade = Trade(
            entry_time=position.entry_time,
            exit_time=timestamp,
            instrument=instrument,
            direction=position.direction.value,
            entry_price=position.entry_price,
            exit_price=exit_price,
            size=position.size,
            pnl=pnl,
            pnl_percent=pnl_percent,
            commission=commission,
            slippage=abs(slippage * position.size),
            strategy=position.direction.value,
            entry_reason="",
            exit_reason=reason,
            max_adverse_excursion=position.max_adverse_excursion,
            max_favorable_excursion=position.max_favorable_excursion
        )

        self.trades.append(trade)

        # Remove position
        del self.current_positions[instrument]

        logger.debug(f"Closed position: {instrument}, P&L: ${pnl:.2f}")

    def _update_positions(self, timestamp: datetime, current_prices: pd.Series):
        """Update unrealized P&L and MAE/MFE for open positions"""
        for instrument, position in self.current_positions.items():
            # Get current price
            if position.direction == PositionSide.LONG:
                current_price = self._get_bid_price(current_prices['close'])
            else:
                current_price = self._get_ask_price(current_prices['close'])

            # Calculate unrealized P&L
            if position.direction == PositionSide.LONG:
                unrealized_pnl = (current_price - position.entry_price) * position.size
            else:
                unrealized_pnl = (position.entry_price - current_price) * position.size

            position.unrealized_pnl = unrealized_pnl

            # Update MAE/MFE
            if unrealized_pnl < position.max_adverse_excursion:
                position.max_adverse_excursion = unrealized_pnl
            if unrealized_pnl > position.max_favorable_excursion:
                position.max_favorable_excursion = unrealized_pnl

    def _check_exit_conditions(self, timestamp: datetime, current_prices: pd.Series):
        """Check if any positions should be closed due to stop loss or take profit"""
        positions_to_close = []

        for instrument, position in self.current_positions.items():
            high_price = current_prices['high']
            low_price = current_prices['low']

            # Check stop loss
            if position.stop_loss:
                if position.direction == PositionSide.LONG and low_price <= position.stop_loss:
                    positions_to_close.append((instrument, 'stop_loss'))
                elif position.direction == PositionSide.SHORT and high_price >= position.stop_loss:
                    positions_to_close.append((instrument, 'stop_loss'))

            # Check take profit
            if position.take_profit:
                if position.direction == PositionSide.LONG and high_price >= position.take_profit:
                    positions_to_close.append((instrument, 'take_profit'))
                elif position.direction == PositionSide.SHORT and low_price <= position.take_profit:
                    positions_to_close.append((instrument, 'take_profit'))

        # Close positions
        for instrument, reason in positions_to_close:
            self._close_position(instrument, timestamp, current_prices, reason)

    def _close_all_positions(self, timestamp: datetime, current_prices: pd.Series):
        """Close all open positions"""
        instruments = list(self.current_positions.keys())
        for instrument in instruments:
            self._close_position(instrument, timestamp, current_prices, "end_of_backtest")

    def _calculate_total_equity(self, current_prices: pd.Series) -> float:
        """Calculate total equity including unrealized P&L"""
        total = self.current_capital

        for position in self.current_positions.values():
            total += position.unrealized_pnl

        return total

    def _calculate_position_size(self, entry_price: float,
                                 stop_loss: Optional[float],
                                 atr: Optional[float]) -> float:
        """Calculate position size based on configured method"""

        if self.config.position_sizing_method == 'fixed_pct':
            if stop_loss:
                return PositionSizer.fixed_percentage(
                    self.current_capital,
                    self.config.position_size_pct,
                    entry_price,
                    stop_loss,
                    self.config.max_position_value_pct
                )
            else:
                # Default to fixed percentage of capital (capped)
                max_notional = self.current_capital * self.config.max_position_value_pct
                return max_notional / entry_price

        elif self.config.position_sizing_method == 'volatility' and atr:
            return PositionSizer.volatility_adjusted(
                self.current_capital,
                self.config.position_size_pct,
                atr,
                entry_price,
                2.0,
                self.config.max_position_value_pct
            )

        elif self.config.position_sizing_method == 'kelly' and len(self.trades) > 20:
            # Need trade history for Kelly
            recent_trades = pd.DataFrame([vars(t) for t in self.trades[-100:]])
            win_rate = len(recent_trades[recent_trades['pnl'] > 0]) / len(recent_trades)
            avg_win = recent_trades[recent_trades['pnl'] > 0]['pnl'].mean()
            avg_loss = abs(recent_trades[recent_trades['pnl'] < 0]['pnl'].mean())

            kelly_pct = PositionSizer.kelly_criterion(win_rate, avg_win, avg_loss)
            # Cap Kelly by max position value
            kelly_size = (self.current_capital * kelly_pct) / entry_price
            max_notional = self.current_capital * self.config.max_position_value_pct
            max_size = max_notional / entry_price
            return min(kelly_size, max_size)

        # Default (capped)
        max_notional = self.current_capital * self.config.max_position_value_pct
        return max_notional / entry_price

    def _get_bid_price(self, mid_price: float) -> float:
        """Get bid price accounting for spread"""
        if self.config.use_bid_ask_spread:
            spread = mid_price * (self.config.spread_pips * 0.0001)
            return mid_price - (spread / 2)
        return mid_price

    def _get_ask_price(self, mid_price: float) -> float:
        """Get ask price accounting for spread"""
        if self.config.use_bid_ask_spread:
            spread = mid_price * (self.config.spread_pips * 0.0001)
            return mid_price + (spread / 2)
        return mid_price

    def _calculate_slippage(self, price: float) -> float:
        """Calculate slippage"""
        return price * self.config.slippage_pct

    def _check_risk_limits(self, timestamp: datetime) -> bool:
        """Check if risk limits are exceeded"""
        # Check daily loss
        date_key = timestamp.date()
        if date_key in self.daily_pnl:
            daily_loss_pct = abs(self.daily_pnl[date_key]) / self.config.initial_capital
            if daily_loss_pct > self.config.max_daily_loss_pct:
                return False

        # Check max drawdown
        if self.equity_curve:
            equity_series = pd.Series([e['equity'] for e in self.equity_curve])
            running_max = equity_series.max()
            current_equity = equity_series.iloc[-1]
            drawdown = (running_max - current_equity) / running_max

            if drawdown > self.config.max_drawdown_pct:
                return False

        return True

    def _empty_results(self) -> Dict:
        """Return empty results"""
        return {
            'metrics': BacktestMetrics._empty_metrics(),
            'trades': [],
            'equity_curve': pd.Series(),
            'strategy_name': ''
        }

    def _print_walk_forward_summary(self, results: List[Dict]):
        """Print summary of walk-forward analysis"""
        logger.info("\n" + "=" * 60)
        logger.info("WALK-FORWARD ANALYSIS SUMMARY")
        logger.info("=" * 60)

        for i, result in enumerate(results, 1):
            metrics = result['metrics']
            logger.info(f"\nPeriod {i}: {result['test_start'].date()} to {result['test_end'].date()}")
            logger.info(f"  Trades: {metrics['total_trades']}")
            logger.info(f"  Win Rate: {metrics['win_rate']:.2%}")
            logger.info(f"  Total P&L: ${metrics['total_pnl']:.2f}")
            logger.info(f"  Sharpe: {metrics['sharpe_ratio']:.2f}")
            logger.info(f"  Max DD: {metrics['max_drawdown_pct']:.2f}%")

        # Overall statistics
        all_metrics = [r['metrics'] for r in results]
        avg_sharpe = np.mean([m['sharpe_ratio'] for m in all_metrics])
        avg_win_rate = np.mean([m['win_rate'] for m in all_metrics])
        total_trades = sum([m['total_trades'] for m in all_metrics])

        logger.info(f"\n{'=' * 60}")
        logger.info("OVERALL STATISTICS")
        logger.info(f"{'=' * 60}")
        logger.info(f"Total Periods: {len(results)}")
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Average Win Rate: {avg_win_rate:.2%}")
        logger.info(f"Average Sharpe: {avg_sharpe:.2f}")
        logger.info("=" * 60 + "\n")
