"""
Sample Trade Data Generator for Testing Dashboard
Generates realistic trading data for backtesting and dashboard visualization
"""

import random
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
import uuid


class TradeGenerator:
    """Generates realistic sample trading data"""

    def __init__(self, initial_capital: float = 10000, seed: int = 42):
        """
        Initialize the trade generator

        Args:
            initial_capital: Starting capital
            seed: Random seed for reproducibility
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        random.seed(seed)
        np.random.seed(seed)

        self.instruments = [
            'SPX500_USD', 'SPX500_USD', 'SPX500_USD', 'SPX500_USD',
            'SPX500_USD', 'SPX500_USD', 'SPX500_USD', 'SPX500_USD'
        ]

        self.strategies = [
            'momentum', 'mean_reversion', 'breakout',
            'trend_following', 'ml_model_v1'
        ]

    def generate_trade(self, date: datetime, win_probability: float = 0.55) -> Dict:
        """
        Generate a single trade

        Args:
            date: Trade date/time
            win_probability: Probability of winning trade (0-1)

        Returns:
            Trade dictionary
        """
        instrument = random.choice(self.instruments)
        direction = random.choice(['long', 'short'])
        strategy = random.choice(self.strategies)

        # Entry and exit prices (realistic S&P 500 range)
        if 'JPY' in instrument:
            base_price = random.uniform(100, 150)
            volatility = random.uniform(0.1, 0.5)
        else:
            base_price = random.uniform(0.8, 1.5)
            volatility = random.uniform(0.001, 0.005)

        entry_price = base_price

        # Determine if winning trade
        is_win = random.random() < win_probability

        if is_win:
            # Winning trade - positive P&L
            price_change = random.uniform(0.3, 2.0) * volatility
            if direction == 'long':
                exit_price = entry_price + price_change
            else:
                exit_price = entry_price - price_change
        else:
            # Losing trade - negative P&L
            price_change = random.uniform(0.3, 1.5) * volatility
            if direction == 'long':
                exit_price = entry_price - price_change
            else:
                exit_price = entry_price + price_change

        # Position sizing (1-3% of capital)
        risk_pct = random.uniform(0.01, 0.03)
        position_value = self.current_capital * risk_pct

        # Calculate units
        if 'JPY' in instrument:
            units = position_value / (entry_price * 0.01)  # Standard lot calculation
        else:
            units = position_value / entry_price * 1000  # Mini lots

        # Calculate P&L
        if direction == 'long':
            pnl = (exit_price - entry_price) * units
        else:
            pnl = (entry_price - exit_price) * units

        # Add realistic commission and slippage
        commission = abs(units) * 0.00005  # 0.5 pip per unit
        slippage = random.uniform(0, 0.0002) * abs(units)

        pnl = pnl - commission - slippage
        pnl_percent = (pnl / position_value) * 100

        # Update capital
        self.current_capital += pnl

        # Trade duration (minutes to hours)
        duration_minutes = int(random.expovariate(1 / 120))  # Avg 2 hours
        exit_time = date + timedelta(minutes=duration_minutes)

        return {
            'trade_id': str(uuid.uuid4()),
            'instrument': instrument,
            'direction': direction,
            'entry_time': date.isoformat(),
            'exit_time': exit_time.isoformat(),
            'entry_price': round(entry_price, 5),
            'exit_price': round(exit_price, 5),
            'size': round(units, 2),
            'pnl': round(pnl, 2),
            'pnl_percent': round(pnl_percent, 2),
            'commission': round(commission, 2),
            'slippage': round(slippage, 2),
            'strategy': strategy,
            'status': 'closed'
        }

    def generate_trade_sequence(self, num_trades: int, start_date: datetime,
                                win_probability: float = 0.55,
                                time_distribution: str = 'random') -> List[Dict]:
        """
        Generate a sequence of trades

        Args:
            num_trades: Number of trades to generate
            start_date: Starting date for trades
            win_probability: Overall win rate
            time_distribution: 'random', 'hourly', or 'daily'

        Returns:
            List of trade dictionaries
        """
        trades = []
        current_date = start_date

        for i in range(num_trades):
            # Add some variability to win rate (simulate changing market conditions)
            if i % 20 == 0:  # Every 20 trades, adjust win rate slightly
                win_prob = np.clip(win_probability + random.uniform(-0.1, 0.1), 0.3, 0.8)
            else:
                win_prob = win_probability

            trade = self.generate_trade(current_date, win_prob)
            trades.append(trade)

            # Advance time based on distribution
            if time_distribution == 'hourly':
                current_date += timedelta(hours=random.randint(1, 6))
            elif time_distribution == 'daily':
                current_date += timedelta(days=1, hours=random.randint(0, 23))
            else:  # random
                current_date += timedelta(
                    hours=random.randint(0, 48),
                    minutes=random.randint(0, 59)
                )

        return trades

    def generate_realistic_campaign(self, days: int = 90) -> List[Dict]:
        """
        Generate a realistic trading campaign with varying performance

        Args:
            days: Number of days to simulate

        Returns:
            List of trades
        """
        trades = []
        current_date = datetime.now() - timedelta(days=days)

        # Simulate different market phases
        phases = [
            {'name': 'learning', 'days': days * 0.2, 'win_rate': 0.45, 'trades_per_day': 2},
            {'name': 'improvement', 'days': days * 0.3, 'win_rate': 0.52, 'trades_per_day': 3},
            {'name': 'profitable', 'days': days * 0.3, 'win_rate': 0.58, 'trades_per_day': 4},
            {'name': 'drawdown', 'days': days * 0.1, 'win_rate': 0.40, 'trades_per_day': 2},
            {'name': 'recovery', 'days': days * 0.1, 'win_rate': 0.55, 'trades_per_day': 3}
        ]

        for phase in phases:
            phase_days = int(phase['days'])
            total_trades = int(phase_days * phase['trades_per_day'])

            phase_trades = self.generate_trade_sequence(
                num_trades=total_trades,
                start_date=current_date,
                win_probability=phase['win_rate'],
                time_distribution='random'
            )

            trades.extend(phase_trades)

            # Move to next phase
            if phase_trades:
                last_trade_date = datetime.fromisoformat(phase_trades[-1]['exit_time'])
                current_date = last_trade_date + timedelta(hours=1)

        return trades

    def generate_open_positions(self, num_positions: int = 3) -> List[Dict]:
        """
        Generate sample open positions

        Args:
            num_positions: Number of open positions to generate

        Returns:
            List of position dictionaries
        """
        positions = []

        for _ in range(num_positions):
            instrument = random.choice(self.instruments)
            direction = random.choice(['long', 'short'])

            # Entry price
            if 'JPY' in instrument:
                entry_price = random.uniform(100, 150)
            else:
                entry_price = random.uniform(0.8, 1.5)

            # Current price (simulate unrealized P&L)
            price_change_pct = random.uniform(-0.02, 0.03)  # -2% to +3%
            if direction == 'long':
                current_price = entry_price * (1 + price_change_pct)
            else:
                current_price = entry_price * (1 - price_change_pct)

            # Position size
            position_value = self.current_capital * random.uniform(0.01, 0.03)
            if 'JPY' in instrument:
                units = position_value / (entry_price * 0.01)
            else:
                units = position_value / entry_price * 1000

            # Calculate unrealized P&L
            if direction == 'long':
                unrealized_pnl = (current_price - entry_price) * units
            else:
                unrealized_pnl = (entry_price - current_price) * units

            # Stop loss and take profit
            if direction == 'long':
                stop_loss = entry_price * 0.98  # 2% stop
                take_profit = entry_price * 1.04  # 4% target
            else:
                stop_loss = entry_price * 1.02
                take_profit = entry_price * 0.96

            entry_time = datetime.now() - timedelta(hours=random.randint(1, 48))

            positions.append({
                'position_id': str(uuid.uuid4()),
                'instrument': instrument,
                'direction': direction,
                'entry_time': entry_time.isoformat(),
                'entry_price': round(entry_price, 5),
                'size': round(units, 2),
                'current_price': round(current_price, 5),
                'unrealized_pnl': round(unrealized_pnl, 2),
                'stop_loss': round(stop_loss, 5),
                'take_profit': round(take_profit, 5)
            })

        return positions


def populate_dashboard_with_sample_data(db_manager, days: int = 90):
    """
    Populate dashboard database with sample trading data

    Args:
        db_manager: DatabaseManager instance from main dashboard
        days: Number of days of trading history to generate
    """
    print(f"Generating {days} days of sample trading data...")

    # Initialize generator
    generator = TradeGenerator(initial_capital=10000)

    # Generate realistic trading campaign
    trades = generator.generate_realistic_campaign(days=days)

    print(f"Generated {len(trades)} trades")
    print(f"Final capital: ${generator.current_capital:,.2f}")
    print(f"Total P&L: ${generator.current_capital - generator.initial_capital:+,.2f}")

    # Add trades to database
    for trade in trades:
        db_manager.add_trade(trade)

    # Generate and add open positions
    positions = generator.generate_open_positions(num_positions=3)
    print(f"\nGenerated {len(positions)} open positions")

    for position in positions:
        db_manager.update_position(position)

    # Generate some sample alerts
    from trading_dashboard_main import Alert

    sample_alerts = [
        Alert(
            timestamp=datetime.now() - timedelta(hours=2),
            level='warning',
            category='Performance',
            message='Win rate dropped below 50% in last 20 trades'
        ),
        Alert(
            timestamp=datetime.now() - timedelta(hours=5),
            level='info',
            category='System',
            message='Daily sync completed successfully'
        )
    ]

    for alert in sample_alerts:
        db_manager.add_alert(alert)

    print("\n✅ Sample data successfully loaded into dashboard!")
    print("Run the dashboard with: streamlit run trading_dashboard_main.py")


if __name__ == "__main__":
    # Example usage
    from trading_dashboard_main import DatabaseManager

    # Initialize database
    db = DatabaseManager()

    # Populate with 90 days of sample data
    populate_dashboard_with_sample_data(db, days=90)

    # Quick stats
    trades_df = db.get_all_trades()
    print(f"\nDatabase contains {len(trades_df)} trades")
    print(f"Total P&L: ${trades_df['pnl'].sum():,.2f}")
    print(f"Win Rate: {len(trades_df[trades_df['pnl'] > 0]) / len(trades_df) * 100:.1f}%")
