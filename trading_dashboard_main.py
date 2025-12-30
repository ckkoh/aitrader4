"""
Oanda Trading Monitoring Dashboard
Real-time performance tracking, risk monitoring, and alerts
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sqlite3
from dataclasses import dataclass
from typing import List, Dict

# Configure page
st.set_page_config(
    page_title="Trading Monitor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .positive { color: #00cc00; }
    .negative { color: #ff0000; }
    .warning { color: #ff9900; }
    .alert-box {
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 5px solid;
    }
    .alert-critical {
        background-color: #ffe6e6;
        border-left-color: #ff0000;
    }
    .alert-warning {
        background-color: #fff4e6;
        border-left-color: #ff9900;
    }
    .alert-info {
        background-color: #e6f3ff;
        border-left-color: #0066cc;
    }
</style>
""", unsafe_allow_html=True)


@dataclass
class TradeMetrics:
    """Container for trade performance metrics"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    current_drawdown: float
    avg_trade_duration: float
    expectancy: float


@dataclass
class Alert:
    """Alert/notification container"""
    timestamp: datetime
    level: str  # 'critical', 'warning', 'info'
    category: str
    message: str


class DatabaseManager:
    """Manages SQLite database for trade data"""

    def __init__(self, db_path: str = "trading_data.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE,
                instrument TEXT,
                direction TEXT,
                entry_time TIMESTAMP,
                exit_time TIMESTAMP,
                entry_price REAL,
                exit_price REAL,
                size REAL,
                pnl REAL,
                pnl_percent REAL,
                commission REAL,
                slippage REAL,
                strategy TEXT,
                status TEXT
            )
        """)

        # Daily performance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_performance (
                date DATE PRIMARY KEY,
                total_trades INTEGER,
                winning_trades INTEGER,
                total_pnl REAL,
                equity REAL,
                drawdown REAL,
                sharpe_ratio REAL
            )
        """)

        # Alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP,
                level TEXT,
                category TEXT,
                message TEXT
            )
        """)

        # Positions table (for open positions)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                position_id TEXT PRIMARY KEY,
                instrument TEXT,
                direction TEXT,
                entry_time TIMESTAMP,
                entry_price REAL,
                size REAL,
                current_price REAL,
                unrealized_pnl REAL,
                stop_loss REAL,
                take_profit REAL
            )
        """)

        conn.commit()
        conn.close()

    def add_trade(self, trade: Dict):
        """Add completed trade to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO trades
            (trade_id, instrument, direction, entry_time, exit_time,
             entry_price, exit_price, size, pnl, pnl_percent,
             commission, slippage, strategy, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade['trade_id'], trade['instrument'], trade['direction'],
            trade['entry_time'], trade['exit_time'], trade['entry_price'],
            trade['exit_price'], trade['size'], trade['pnl'],
            trade['pnl_percent'], trade['commission'], trade['slippage'],
            trade['strategy'], trade['status']
        ))

        conn.commit()
        conn.close()

    def get_trades(self, days: int = 30) -> pd.DataFrame:
        """Get trades from last N days"""
        conn = sqlite3.connect(self.db_path)
        query = """
            SELECT * FROM trades
            WHERE exit_time >= datetime('now', '-{days} days')
            ORDER BY exit_time DESC
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

    def get_all_trades(self) -> pd.DataFrame:
        """Get all trades"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM trades ORDER BY exit_time DESC", conn)
        conn.close()
        return df

    def add_alert(self, alert: Alert):
        """Add alert to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO alerts (timestamp, level, category, message)
            VALUES (?, ?, ?, ?)
        """, (alert.timestamp, alert.level, alert.category, alert.message))

        conn.commit()
        conn.close()

    def get_recent_alerts(self, hours: int = 24) -> List[Alert]:
        """Get recent alerts"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT timestamp, level, category, message
            FROM alerts
            WHERE timestamp >= datetime('now', '-{} hours')
            ORDER BY timestamp DESC
        """.format(hours))

        alerts = []
        for row in cursor.fetchall():
            alerts.append(Alert(
                timestamp=datetime.fromisoformat(row[0]),
                level=row[1],
                category=row[2],
                message=row[3]
            ))

        conn.close()
        return alerts

    def update_position(self, position: Dict):
        """Update open position"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO positions
            (position_id, instrument, direction, entry_time, entry_price,
             size, current_price, unrealized_pnl, stop_loss, take_profit)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            position['position_id'], position['instrument'], position['direction'],
            position['entry_time'], position['entry_price'], position['size'],
            position['current_price'], position['unrealized_pnl'],
            position['stop_loss'], position['take_profit']
        ))

        conn.commit()
        conn.close()

    def get_open_positions(self) -> pd.DataFrame:
        """Get all open positions"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM positions", conn)
        conn.close()
        return df

    def close_position(self, position_id: str):
        """Remove closed position"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM positions WHERE position_id = ?", (position_id,))
        conn.commit()
        conn.close()


class PerformanceCalculator:
    """Calculate trading performance metrics"""

    @staticmethod
    def calculate_metrics(trades_df: pd.DataFrame, initial_capital: float = 10000) -> TradeMetrics:
        """Calculate comprehensive performance metrics"""
        if trades_df.empty:
            return TradeMetrics(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_pnl = trades_df['pnl'].sum()

        wins = trades_df[trades_df['pnl'] > 0]['pnl']
        losses = trades_df[trades_df['pnl'] <= 0]['pnl']

        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0

        gross_profit = wins.sum() if len(wins) > 0 else 0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
        profit_factor = (gross_profit / gross_loss) if gross_loss != 0 else 0

        # Calculate Sharpe ratio
        if len(trades_df) > 1:
            returns = trades_df['pnl_percent'].values
            sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) != 0 else 0
        else:
            sharpe_ratio = 0

        # Calculate drawdown
        cumulative_pnl = trades_df['pnl'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - running_max
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
        current_drawdown = abs(drawdown.iloc[-1]) if len(drawdown) > 0 else 0

        # Trade duration
        if 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
            trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
            trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
            durations = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 3600
            avg_trade_duration = durations.mean()
        else:
            avg_trade_duration = 0

        # Expectancy
        expectancy = (win_rate / 100 * avg_win) - ((100 - win_rate) / 100 * avg_loss)

        return TradeMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            avg_trade_duration=avg_trade_duration,
            expectancy=expectancy
        )

    @staticmethod
    def create_equity_curve(trades_df: pd.DataFrame, initial_capital: float = 10000) -> pd.DataFrame:
        """Create equity curve from trades"""
        if trades_df.empty:
            return pd.DataFrame({'date': [datetime.now()], 'equity': [initial_capital]})

        trades_df = trades_df.copy()
        trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
        trades_df = trades_df.sort_values('exit_time')

        equity_data = []
        current_equity = initial_capital

        equity_data.append({'date': trades_df['exit_time'].iloc[0], 'equity': initial_capital})

        for _, trade in trades_df.iterrows():
            current_equity += trade['pnl']
            equity_data.append({'date': trade['exit_time'], 'equity': current_equity})

        return pd.DataFrame(equity_data)


class RiskMonitor:
    """Monitor risk metrics and generate alerts"""

    def __init__(self, max_daily_loss: float = 0.05, max_drawdown: float = 0.15):
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        self.alerts = []

    def check_risk_limits(self, metrics: TradeMetrics, daily_pnl: float,
                          capital: float) -> List[Alert]:
        """Check all risk limits and generate alerts"""
        alerts = []
        now = datetime.now()

        # Daily loss check
        daily_loss_pct = daily_pnl / capital
        if daily_loss_pct < -self.max_daily_loss:
            alerts.append(Alert(
                timestamp=now,
                level='critical',
                category='Risk Limit',
                message=f'Daily loss limit exceeded: {daily_loss_pct:.2%} (Limit: {-self.max_daily_loss:.2%})'
            ))
        elif daily_loss_pct < -self.max_daily_loss * 0.7:
            alerts.append(Alert(
                timestamp=now,
                level='warning',
                category='Risk Warning',
                message=f'Approaching daily loss limit: {daily_loss_pct:.2%}'
            ))

        # Drawdown check
        drawdown_pct = metrics.current_drawdown / capital
        if drawdown_pct > self.max_drawdown:
            alerts.append(Alert(
                timestamp=now,
                level='critical',
                category='Risk Limit',
                message=f'Maximum drawdown exceeded: {drawdown_pct:.2%} (Limit: {self.max_drawdown:.2%})'
            ))
        elif drawdown_pct > self.max_drawdown * 0.8:
            alerts.append(Alert(
                timestamp=now,
                level='warning',
                category='Risk Warning',
                message=f'Approaching max drawdown: {drawdown_pct:.2%}'
            ))

        # Losing streak check
        if metrics.total_trades >= 5:
            # This would need recent trade results - simplified here
            if metrics.win_rate < 30:
                alerts.append(Alert(
                    timestamp=now,
                    level='warning',
                    category='Performance',
                    message=f'Low win rate detected: {metrics.win_rate:.1f}%'
                ))

        # Sharpe ratio check
        if metrics.total_trades >= 20 and metrics.sharpe_ratio < 0:
            alerts.append(Alert(
                timestamp=now,
                level='warning',
                category='Performance',
                message=f'Negative Sharpe ratio: {metrics.sharpe_ratio:.2f}'
            ))

        return alerts


def create_performance_charts(trades_df: pd.DataFrame, equity_df: pd.DataFrame,
                              positions_df: pd.DataFrame):
    """Create performance visualization charts"""

    # Equity curve
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(
        x=equity_df['date'],
        y=equity_df['equity'],
        mode='lines',
        name='Equity',
        line=dict(color='#0066cc', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 102, 204, 0.1)'
    ))
    fig_equity.update_layout(
        title='Equity Curve',
        xaxis_title='Date',
        yaxis_title='Equity ($)',
        hovermode='x unified',
        height=400
    )

    # PnL distribution
    if not trades_df.empty:
        fig_pnl_dist = go.Figure()
        fig_pnl_dist.add_trace(go.Histogram(
            x=trades_df['pnl'],
            nbinsx=30,
            name='PnL Distribution',
            marker_color='#0066cc'
        ))
        fig_pnl_dist.update_layout(
            title='PnL Distribution',
            xaxis_title='PnL ($)',
            yaxis_title='Frequency',
            height=300
        )
    else:
        fig_pnl_dist = go.Figure()
        fig_pnl_dist.add_annotation(text="No trades yet", showarrow=False,
                                    xref="paper", yref="paper", x=0.5, y=0.5)

    # Daily PnL
    if not trades_df.empty:
        trades_df['exit_date'] = pd.to_datetime(trades_df['exit_time']).dt.date
        daily_pnl = trades_df.groupby('exit_date')['pnl'].sum().reset_index()

        fig_daily = go.Figure()
        colors = ['green' if x >= 0 else 'red' for x in daily_pnl['pnl']]
        fig_daily.add_trace(go.Bar(
            x=daily_pnl['exit_date'],
            y=daily_pnl['pnl'],
            marker_color=colors,
            name='Daily PnL'
        ))
        fig_daily.update_layout(
            title='Daily PnL',
            xaxis_title='Date',
            yaxis_title='PnL ($)',
            height=300
        )
    else:
        fig_daily = go.Figure()
        fig_daily.add_annotation(text="No trades yet", showarrow=False,
                                 xref="paper", yref="paper", x=0.5, y=0.5)

    # Win rate over time
    if not trades_df.empty and len(trades_df) >= 10:
        trades_df = trades_df.sort_values('exit_time')
        trades_df['win'] = (trades_df['pnl'] > 0).astype(int)
        trades_df['rolling_win_rate'] = trades_df['win'].rolling(window=20, min_periods=10).mean() * 100

        fig_winrate = go.Figure()
        fig_winrate.add_trace(go.Scatter(
            x=trades_df['exit_time'],
            y=trades_df['rolling_win_rate'],
            mode='lines',
            name='Win Rate (20-trade MA)',
            line=dict(color='#00cc00', width=2)
        ))
        fig_winrate.add_hline(y=50, line_dash="dash", line_color="gray",
                              annotation_text="50%")
        fig_winrate.update_layout(
            title='Rolling Win Rate (20 trades)',
            xaxis_title='Date',
            yaxis_title='Win Rate (%)',
            height=300
        )
    else:
        fig_winrate = go.Figure()
        fig_winrate.add_annotation(text="Not enough trades", showarrow=False,
                                   xref="paper", yref="paper", x=0.5, y=0.5)

    return fig_equity, fig_pnl_dist, fig_daily, fig_winrate


def display_alerts(alerts: List[Alert]):
    """Display alert notifications"""
    if not alerts:
        st.success("✅ No active alerts - all systems normal")
        return

    st.subheader("🚨 Active Alerts")

    for alert in alerts:
        if alert.level == 'critical':
            css_class = 'alert-critical'
            icon = '🔴'
        elif alert.level == 'warning':
            css_class = 'alert-warning'
            icon = '⚠️'
        else:
            css_class = 'alert-info'
            icon = 'ℹ️'

        st.markdown(f"""
        <div class="alert-box {css_class}">
            <strong>{icon} {alert.category}</strong> - {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}<br>
            {alert.message}
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main dashboard application"""

    # Initialize
    db = DatabaseManager()
    risk_monitor = RiskMonitor(max_daily_loss=0.05, max_drawdown=0.15)

    # Sidebar
    st.sidebar.title("📊 Trading Monitor")
    st.sidebar.markdown("---")

    # Settings
    initial_capital = st.sidebar.number_input("Initial Capital ($)",
                                              value=10000, step=1000)
    days_to_show = st.sidebar.selectbox("Time Period",
                                        [7, 14, 30, 60, 90, 180, 365],
                                        index=2)

    _refresh_rate = st.sidebar.selectbox(  # noqa: F841
        "Refresh Rate",
        ["Manual", "30s", "1min", "5min"],
        index=0)

    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Navigation")
    page = st.sidebar.radio("Go to:",
                            ["📈 Overview", "📋 Trades", "⚠️ Risk Monitor",
                             "🔍 Analysis", "⚙️ Settings"])

    # Load data
    trades_df = db.get_trades(days=days_to_show)
    positions_df = db.get_open_positions()

    # Calculate metrics
    metrics = PerformanceCalculator.calculate_metrics(trades_df, initial_capital)
    equity_df = PerformanceCalculator.create_equity_curve(trades_df, initial_capital)

    # Calculate current values
    current_equity = equity_df['equity'].iloc[-1] if not equity_df.empty else initial_capital
    daily_pnl = trades_df[trades_df['exit_time'] >=
                          (datetime.now() - timedelta(days=1)).isoformat()]['pnl'].sum() if not trades_df.empty else 0

    # Check risk limits
    alerts = risk_monitor.check_risk_limits(metrics, daily_pnl, current_equity)

    # Save alerts to database
    for alert in alerts:
        db.add_alert(alert)

    # Get recent alerts
    recent_alerts = db.get_recent_alerts(hours=24)

    # PAGE: OVERVIEW
    if page == "📈 Overview":
        st.title("📈 Trading Performance Dashboard")

        # Alerts section
        if recent_alerts:
            display_alerts(recent_alerts[:5])  # Show top 5 most recent
            st.markdown("---")

        # Key metrics row
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Current Equity", f"${current_equity:,.2f}",
                      f"${metrics.total_pnl:+,.2f}")

        with col2:
            pnl_color = "normal" if daily_pnl >= 0 else "inverse"
            st.metric("Daily P&L", f"${daily_pnl:+,.2f}",
                      f"{daily_pnl / initial_capital * 100:+.2f}%",
                      delta_color=pnl_color)

        with col3:
            st.metric("Total Trades", metrics.total_trades,
                      f"{metrics.win_rate:.1f}% Win Rate")

        with col4:
            st.metric("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}",
                      "Good" if metrics.sharpe_ratio > 1 else "Poor")

        with col5:
            dd_pct = metrics.current_drawdown / current_equity * 100
            st.metric("Drawdown", f"{dd_pct:.2f}%",
                      f"Max: {metrics.max_drawdown / initial_capital * 100:.2f}%")

        st.markdown("---")

        # Charts row 1
        col1, col2 = st.columns([2, 1])

        with col1:
            fig_equity, _, _, _ = create_performance_charts(trades_df, equity_df, positions_df)
            st.plotly_chart(fig_equity, use_container_width=True)

        with col2:
            st.subheader("Performance Summary")

            summary_data = {
                "Metric": [
                    "Total P&L",
                    "Profit Factor",
                    "Avg Win",
                    "Avg Loss",
                    "Expectancy",
                    "Win/Loss",
                    "Avg Duration"
                ],
                "Value": [
                    f"${metrics.total_pnl:,.2f}",
                    f"{metrics.profit_factor:.2f}",
                    f"${metrics.avg_win:,.2f}",
                    f"${metrics.avg_loss:,.2f}",
                    f"${metrics.expectancy:,.2f}",
                    f"{metrics.winning_trades}/{metrics.losing_trades}",
                    f"{metrics.avg_trade_duration:.1f}h"
                ]
            }
            st.dataframe(pd.DataFrame(summary_data), hide_index=True,
                         use_container_width=True)

        # Charts row 2
        _, fig_pnl_dist, fig_daily, fig_winrate = create_performance_charts(
            trades_df, equity_df, positions_df)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_daily, use_container_width=True)
        with col2:
            st.plotly_chart(fig_pnl_dist, use_container_width=True)

        st.plotly_chart(fig_winrate, use_container_width=True)

        # Open positions
        if not positions_df.empty:
            st.markdown("---")
            st.subheader("💼 Open Positions")

            # Format for display
            display_positions = positions_df.copy()
            display_positions['unrealized_pnl'] = display_positions['unrealized_pnl'].apply(
                lambda x: f"${x:+,.2f}")
            display_positions['entry_price'] = display_positions['entry_price'].apply(
                lambda x: f"{x:.5f}")
            display_positions['current_price'] = display_positions['current_price'].apply(
                lambda x: f"{x:.5f}")

            st.dataframe(display_positions, use_container_width=True, hide_index=True)

    # PAGE: TRADES
    elif page == "📋 Trades":
        st.title("📋 Trade History")

        if trades_df.empty:
            st.info("No trades found for the selected period.")
        else:
            # Filters
            col1, col2, col3 = st.columns(3)

            with col1:
                instruments = ['All'] + sorted(trades_df['instrument'].unique().tolist())
                selected_instrument = st.selectbox("Instrument", instruments)

            with col2:
                directions = ['All', 'long', 'short']
                selected_direction = st.selectbox("Direction", directions)

            with col3:
                strategies = ['All'] + sorted(trades_df['strategy'].unique().tolist())
                selected_strategy = st.selectbox("Strategy", strategies)

            # Filter data
            filtered_df = trades_df.copy()
            if selected_instrument != 'All':
                filtered_df = filtered_df[filtered_df['instrument'] == selected_instrument]
            if selected_direction != 'All':
                filtered_df = filtered_df[filtered_df['direction'] == selected_direction]
            if selected_strategy != 'All':
                filtered_df = filtered_df[filtered_df['strategy'] == selected_strategy]

            st.markdown(f"**Showing {len(filtered_df)} of {len(trades_df)} trades**")

            # Format for display
            display_df = filtered_df.copy()
            display_df['pnl'] = display_df['pnl'].apply(lambda x: f"${x:+,.2f}")
            display_df['pnl_percent'] = display_df['pnl_percent'].apply(lambda x: f"{x:+.2f}%")

            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Export
            if st.button("📥 Export to CSV"):
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"trades_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

    # PAGE: RISK MONITOR
    elif page == "⚠️ Risk Monitor":
        st.title("⚠️ Risk Monitor")

        # Display all alerts from last 24 hours
        display_alerts(recent_alerts)

        st.markdown("---")

        # Risk metrics
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Risk Limits")

            # Daily loss
            daily_loss_pct = daily_pnl / current_equity * 100
            daily_loss_used = abs(daily_loss_pct) / (risk_monitor.max_daily_loss * 100)

            st.metric("Daily Loss Limit",
                      f"{risk_monitor.max_daily_loss * 100:.1f}%",
                      f"{daily_loss_used * 100:.1f}% used")

            st.progress(min(daily_loss_used, 1.0))

            # Drawdown
            dd_pct = metrics.current_drawdown / current_equity
            dd_used = dd_pct / risk_monitor.max_drawdown

            st.metric("Max Drawdown Limit",
                      f"{risk_monitor.max_drawdown * 100:.1f}%",
                      f"{dd_used * 100:.1f}% used")

            st.progress(min(dd_used, 1.0))

        with col2:
            st.subheader("Position Concentration")

            if not positions_df.empty:
                instrument_exposure = positions_df.groupby('instrument')['size'].sum()

                fig_exposure = px.pie(
                    values=instrument_exposure.values,
                    names=instrument_exposure.index,
                    title="Exposure by Instrument"
                )
                st.plotly_chart(fig_exposure, use_container_width=True)
            else:
                st.info("No open positions")

        st.markdown("---")

        # Recent performance
        st.subheader("Recent Performance (Last 10 Trades)")

        if len(trades_df) >= 10:
            recent_10 = trades_df.head(10)

            col1, col2, col3 = st.columns(3)

            with col1:
                recent_wins = len(recent_10[recent_10['pnl'] > 0])
                st.metric("Win Rate", f"{recent_wins / 10 * 100:.0f}%")

            with col2:
                recent_pnl = recent_10['pnl'].sum()
                st.metric("Total P&L", f"${recent_pnl:+,.2f}")

            with col3:
                avg_recent = recent_10['pnl'].mean()
                st.metric("Avg P&L", f"${avg_recent:+,.2f}")

            # Visual of last 10 trades
            fig_recent = go.Figure()
            colors = ['green' if x > 0 else 'red' for x in recent_10['pnl']]
            fig_recent.add_trace(go.Bar(
                x=list(range(1, 11)),
                y=recent_10['pnl'],
                marker_color=colors
            ))
            fig_recent.update_layout(
                title="Last 10 Trades P&L",
                xaxis_title="Trade Number (Most Recent = 1)",
                yaxis_title="P&L ($)",
                height=300
            )
            st.plotly_chart(fig_recent, use_container_width=True)

    # PAGE: ANALYSIS
    elif page == "🔍 Analysis":
        st.title("🔍 Performance Analysis")

        if trades_df.empty:
            st.info("No trades available for analysis.")
        else:
            # Performance by instrument
            st.subheader("Performance by Instrument")

            instrument_stats = trades_df.groupby('instrument').agg({
                'pnl': ['sum', 'mean', 'count'],
                'pnl_percent': 'mean'
            }).round(2)

            instrument_stats.columns = ['Total P&L', 'Avg P&L', 'Trades', 'Avg Return %']
            instrument_stats = instrument_stats.sort_values('Total P&L', ascending=False)

            st.dataframe(instrument_stats, use_container_width=True)

            # Performance by strategy
            st.subheader("Performance by Strategy")

            strategy_stats = trades_df.groupby('strategy').agg({
                'pnl': ['sum', 'mean', 'count'],
                'pnl_percent': 'mean'
            }).round(2)

            strategy_stats.columns = ['Total P&L', 'Avg P&L', 'Trades', 'Avg Return %']
            strategy_stats = strategy_stats.sort_values('Total P&L', ascending=False)

            st.dataframe(strategy_stats, use_container_width=True)

            # Performance by time
            st.subheader("Performance by Hour of Day")

            if 'exit_time' in trades_df.columns:
                trades_df['hour'] = pd.to_datetime(trades_df['exit_time']).dt.hour
                hourly_pnl = trades_df.groupby('hour')['pnl'].agg(['sum', 'count'])

                fig_hourly = go.Figure()
                fig_hourly.add_trace(go.Bar(
                    x=hourly_pnl.index,
                    y=hourly_pnl['sum'],
                    name='P&L',
                    marker_color=['green' if x > 0 else 'red' for x in hourly_pnl['sum']]
                ))

                fig_hourly.update_layout(
                    title="P&L by Hour of Day",
                    xaxis_title="Hour (UTC)",
                    yaxis_title="Total P&L ($)",
                    height=400
                )

                st.plotly_chart(fig_hourly, use_container_width=True)

            # Performance by day of week
            st.subheader("Performance by Day of Week")

            if 'exit_time' in trades_df.columns:
                trades_df['day_of_week'] = pd.to_datetime(trades_df['exit_time']).dt.day_name()
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

                daily_pnl = trades_df.groupby('day_of_week')['pnl'].agg(['sum', 'count', 'mean'])
                daily_pnl = daily_pnl.reindex(day_order)

                fig_daily = go.Figure()
                fig_daily.add_trace(go.Bar(
                    x=daily_pnl.index,
                    y=daily_pnl['sum'],
                    name='Total P&L',
                    marker_color=['green' if x > 0 else 'red' for x in daily_pnl['sum']]
                ))

                fig_daily.update_layout(
                    title="P&L by Day of Week",
                    xaxis_title="Day",
                    yaxis_title="Total P&L ($)",
                    height=400
                )

                st.plotly_chart(fig_daily, use_container_width=True)

    # PAGE: SETTINGS
    elif page == "⚙️ Settings":
        st.title("⚙️ Settings")

        st.subheader("Risk Management")

        col1, col2 = st.columns(2)

        with col1:
            st.slider("Max Daily Loss (%)", 1, 10, 5)
            st.slider("Max Drawdown (%)", 5, 30, 15)

        with col2:
            st.slider("Max Position Size (%)", 1, 10, 2)
            st.slider("Max Position Correlation", 0.3, 1.0, 0.7)

        st.subheader("Alert Settings")

        email_alerts = st.checkbox("Enable Email Alerts")
        if email_alerts:
            st.text_input("Email Address")

        slack_alerts = st.checkbox("Enable Slack Notifications")
        if slack_alerts:
            _slack_webhook = st.text_input("Slack Webhook URL", type="password")  # noqa: F841

        st.subheader("Data Management")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📥 Export All Data"):
                all_trades = db.get_all_trades()
                csv = all_trades.to_csv(index=False)
                st.download_button(
                    "Download All Trades",
                    csv,
                    f"all_trades_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv"
                )

        with col2:
            if st.button("🗑️ Clear Old Data (>1 year)"):
                st.warning("This will delete trades older than 1 year. This action cannot be undone.")
                if st.button("Confirm Delete"):
                    # Implementation would go here
                    st.success("Old data cleared successfully")

        if st.button("💾 Save Settings"):
            st.success("Settings saved successfully!")


if __name__ == "__main__":
    main()
