"""
Integration Example: Add Health Monitoring to Your Trading System
Shows how to integrate failure detection into existing codebase
"""

from model_failure_recovery import (
    ModelFailureDetector, RecoveryStrategy,
    AdaptivePositionSizer, HealthStatus
)
from trading_dashboard_main import DatabaseManager
from oanda_integration import OandaConnector
from backtesting_engine import BacktestEngine, BacktestConfig
import pandas as pd
import time
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MonitoredTradingBot:
    """
    Enhanced trading bot with health monitoring and automatic recovery
    """

    def __init__(self, strategy, oanda_config, initial_capital=10000):
        # Core components
        self.strategy = strategy
        self.oanda = OandaConnector(**oanda_config)
        self.db = DatabaseManager()

        # Health monitoring
        self.health_monitor = ModelFailureDetector(
            min_win_rate=0.45,
            min_sharpe=0.5,
            max_consecutive_losses=5,
            max_drawdown=0.15,
            min_confidence=0.55
        )

        # Recovery system
        self.recovery = RecoveryStrategy()

        # Adaptive position sizing
        self.position_sizer = AdaptivePositionSizer(base_risk_pct=0.02)

        # Configuration
        self.initial_capital = initial_capital
        self.is_running = True
        self.last_health_check = None

    def set_baseline_from_backtest(self, backtest_results):
        """
        Set baseline metrics from backtesting results

        Args:
            backtest_results: Dictionary from BacktestEngine.run_backtest()
        """
        metrics = backtest_results['metrics']

        self.health_monitor.set_baseline(
            win_rate=metrics['win_rate'],
            sharpe=metrics['sharpe_ratio'],
            profit_factor=metrics['profit_factor']
        )

        logger.info("✅ Baseline metrics set from backtest:")
        logger.info(f"   Win Rate: {metrics['win_rate']:.2%}")
        logger.info(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"   Profit Factor: {metrics['profit_factor']:.2f}")

    def run(self, health_check_interval_minutes=15):
        """
        Main trading loop with integrated health monitoring

        Args:
            health_check_interval_minutes: How often to check system health
        """
        logger.info("🚀 Starting monitored trading bot...")

        last_health_check = datetime.now()

        while self.is_running:
            try:
                # 1. HEALTH CHECK (every N minutes)
                if (datetime.now() - last_health_check).seconds > health_check_interval_minutes * 60:
                    health_status = self._perform_health_check()

                    if health_status == HealthStatus.FAILED:
                        logger.critical("❌ System health failed - STOPPING")
                        break

                    last_health_check = datetime.now()

                # 2. CHECK IF TRADING IS PAUSED
                if self.recovery.is_trading_paused:
                    logger.info("⏸ Trading paused - waiting for recovery")
                    time.sleep(60)
                    continue

                # 3. GET MARKET DATA
                data = self._get_recent_data()

                # 4. GENERATE SIGNALS
                signals = self.strategy.generate_signals(data, datetime.now())

                # 5. EXECUTE SIGNALS (with adaptive sizing)
                for signal in signals:
                    self._execute_signal_with_monitoring(signal, data)

                # 6. UPDATE POSITIONS
                self._update_open_positions()

                # 7. SYNC TO DASHBOARD
                self._sync_to_dashboard()

                # Sleep before next iteration
                time.sleep(60)  # Check every minute

            except KeyboardInterrupt:
                logger.info("🛑 Shutting down gracefully...")
                self._shutdown()
                break

            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(60)

    def _perform_health_check(self):
        """
        Perform comprehensive health check
        """
        logger.info("🏥 Performing health check...")

        # Get recent trades from database
        trades_df = self.db.get_trades(days=90)

        if len(trades_df) < 10:
            logger.warning("⚠️ Not enough trades for health check")
            return HealthStatus.HEALTHY

        # Run health check
        health = self.health_monitor.check_health(trades_df)

        # Log results
        self._log_health_status(health)

        # Take action if needed
        if health.status != HealthStatus.HEALTHY:
            positions = self.oanda.get_open_positions()
            action = self.recovery.execute_recovery(health, positions)

            # Log recovery action
            logger.warning(f"🔧 Recovery action: {action['action']}")
            for detail in action['details']:
                logger.warning(f"   {detail}")

            # Send alert
            self._send_alert(health, action)

        self.last_health_check = health
        return health.status

    def _execute_signal_with_monitoring(self, signal, data):
        """
        Execute trading signal with adaptive position sizing

        Args:
            signal: Trading signal dictionary
            data: Recent market data
        """
        if signal['action'] not in ['buy', 'sell']:
            return

        # Get recent trade history
        trades_df = self.db.get_trades(days=30)

        # Calculate adaptive position size
        stop_loss_distance = abs(signal['entry_price'] - signal['stop_loss'])
        current_capital = self._get_current_capital()

        base_size = self.position_sizer.calculate_size(
            capital=current_capital,
            trades_df=trades_df,
            stop_loss_distance=stop_loss_distance
        )

        # Apply recovery multiplier if active
        adjusted_size = base_size * self.recovery.current_position_size_multiplier

        logger.info("📊 Position sizing:")
        logger.info(f"   Base: {base_size:.2f}")
        logger.info(f"   Multiplier: {self.recovery.current_position_size_multiplier:.2f}")
        logger.info(f"   Final: {adjusted_size:.2f}")

        # Execute order
        try:
            if signal['action'] == 'buy':
                units = int(adjusted_size)
            else:
                units = -int(adjusted_size)

            _response = self.oanda.place_market_order(  # noqa: F841
                instrument=signal['instrument'],
                units=units,
                stop_loss=signal.get('stop_loss'),
                take_profit=signal.get('take_profit')
            )

            logger.info(f"✅ Order executed: {signal['instrument']} {units} units")

        except Exception as e:
            logger.error(f"❌ Order execution failed: {e}")

    def _log_health_status(self, health):
        """Log detailed health status"""

        status_emoji = {
            HealthStatus.HEALTHY: "✅",
            HealthStatus.WARNING: "⚠️",
            HealthStatus.CRITICAL: "🔴",
            HealthStatus.FAILED: "❌"
        }

        logger.info("=" * 60)
        logger.info(f"{status_emoji[health.status]} HEALTH STATUS: {health.status.value.upper()}")
        logger.info("=" * 60)
        logger.info(f"Win Rate (30): {health.win_rate_30:.1%}")
        logger.info(f"Win Rate (100): {health.win_rate_100:.1%}")
        logger.info(f"Sharpe (30d): {health.sharpe_30d:.2f}")
        logger.info(f"Drawdown: {health.current_drawdown:.1%}")
        logger.info(f"Consecutive Losses: {health.consecutive_losses}")
        logger.info(f"Model Drift: {health.model_drift_score:.2f}")

        if health.warnings:
            logger.warning(f"\n⚠️ WARNINGS ({len(health.warnings)}):")
            for warning in health.warnings:
                logger.warning(f"  - {warning}")

        logger.info("=" * 60)

    def _send_alert(self, health, action):
        """
        Send alert notifications

        In production, integrate with:
        - Email (SMTP)
        - SMS (Twilio)
        - Slack (Webhook)
        - Discord (Webhook)
        """

        # Example: Email alert
        message = """
        TRADING SYSTEM ALERT

        Status: {health.status.value.upper()}
        Action Taken: {action['action']}

        Metrics:
        - Win Rate (30): {health.win_rate_30:.1%}
        - Sharpe (30d): {health.sharpe_30d:.2f}
        - Drawdown: {health.current_drawdown:.1%}

        Warnings:
        {chr(10).join(['- ' + w for w in health.warnings])}

        Time: {datetime.now()}
        """

        logger.critical(message)

        # TODO: Implement actual email/SMS/Slack notification
        # send_email(to="your@email.com", subject="Trading Alert", body=message)
        # send_slack(webhook_url="...", message=message)

    def _get_recent_data(self):
        """Get recent market data"""
        # In production, fetch from Oanda
        # For now, return placeholder
        return pd.DataFrame()

    def _update_open_positions(self):
        """Update unrealized P&L for open positions"""
        positions = self.oanda.get_open_positions()

        for position in positions:
            self.db.update_position(position)

    def _sync_to_dashboard(self):
        """Sync latest data to dashboard"""
        # This happens automatically via DatabaseManager

    def _get_current_capital(self):
        """Calculate current capital"""
        account = self.oanda.get_account_summary()
        return float(account.get('balance', self.initial_capital))

    def _shutdown(self):
        """Graceful shutdown"""
        logger.info("Closing all positions...")

        positions = self.oanda.get_open_positions()
        for pos in positions:
            try:
                self.oanda.close_position(pos['instrument'])
                logger.info(f"Closed: {pos['instrument']}")
            except Exception as e:
                logger.error(f"Error closing {pos['instrument']}: {e}")

        logger.info("✅ Shutdown complete")
        self.is_running = False


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def main():
    """
    Example: Running monitored trading bot
    """

    # 1. CONFIGURATION
    from config import OANDA_CONFIG
    from strategy_examples import MomentumStrategy

    # Create strategy
    strategy = MomentumStrategy(fast_period=20, slow_period=50)

    # 2. RUN BACKTEST TO SET BASELINE
    logger.info("Step 1: Running backtest to establish baseline...")

    from complete_workflow import TradingSystemPipeline

    pipeline = TradingSystemPipeline()
    pipeline.load_oanda_data(instrument='SPX500_USD', days=365)

    # Get historical data
    df = pipeline.data

    # Run backtest
    config = BacktestConfig(initial_capital=10000)
    engine = BacktestEngine(config)
    backtest_results = engine.run_backtest(strategy, df)

    logger.info(f"✅ Backtest complete: Sharpe={backtest_results['metrics']['sharpe_ratio']:.2f}")

    # 3. INITIALIZE MONITORED BOT
    logger.info("\nStep 2: Initializing monitored trading bot...")

    bot = MonitoredTradingBot(
        strategy=strategy,
        oanda_config=OANDA_CONFIG,
        initial_capital=10000
    )

    # Set baseline from backtest
    bot.set_baseline_from_backtest(backtest_results)

    # 4. RUN BOT WITH MONITORING
    logger.info("\nStep 3: Starting live trading with health monitoring...")
    logger.info("Health checks will run every 15 minutes")
    logger.info("Press Ctrl+C to stop\n")

    try:
        bot.run(health_check_interval_minutes=15)
    except KeyboardInterrupt:
        logger.info("\nStopping bot...")


if __name__ == "__main__":
    main()
