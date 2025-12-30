"""
Model Failure Detection and Recovery System
Critical for preventing catastrophic losses in live trading
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """System health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"


class RecoveryAction(Enum):
    """Possible recovery actions"""
    CONTINUE = "continue_trading"
    REDUCE_SIZE = "reduce_position_size"
    PAUSE_NEW_TRADES = "pause_new_trades"
    CLOSE_POSITIONS = "close_all_positions"
    STOP_TRADING = "stop_trading_completely"
    RETRAIN_MODEL = "retrain_model"
    SWITCH_STRATEGY = "switch_to_backup_strategy"


@dataclass
class HealthMetrics:
    """Container for system health metrics"""
    timestamp: datetime
    win_rate_30: float  # Last 30 trades
    win_rate_100: float  # Last 100 trades
    sharpe_30d: float  # 30-day Sharpe
    current_drawdown: float
    consecutive_losses: int
    prediction_confidence: float
    model_drift_score: float
    correlation_breakdown: float
    status: HealthStatus
    recommended_action: RecoveryAction
    warnings: List[str]


class ModelFailureDetector:
    """
    Detects when trading models are failing or degrading

    Key Failure Signals:
    1. Win rate significantly below expected
    2. Sharpe ratio turning negative
    3. Consecutive losing trades
    4. Prediction confidence dropping
    5. Feature distribution drift
    6. Correlation breakdown
    """

    def __init__(self,
                 min_win_rate: float = 0.45,
                 min_sharpe: float = 0.5,
                 max_consecutive_losses: int = 5,
                 max_drawdown: float = 0.15,
                 min_confidence: float = 0.55):
        """
        Initialize failure detector with thresholds

        Args:
            min_win_rate: Minimum acceptable win rate
            min_sharpe: Minimum acceptable Sharpe ratio
            max_consecutive_losses: Max consecutive losses before alert
            max_drawdown: Maximum acceptable drawdown
            min_confidence: Minimum model confidence threshold
        """
        self.min_win_rate = min_win_rate
        self.min_sharpe = min_sharpe
        self.max_consecutive_losses = max_consecutive_losses
        self.max_drawdown = max_drawdown
        self.min_confidence = min_confidence

        # Baseline metrics (from backtesting/validation)
        self.baseline_win_rate = None
        self.baseline_sharpe = None
        self.baseline_profit_factor = None

    def set_baseline(self, win_rate: float, sharpe: float, profit_factor: float):
        """Set baseline metrics from backtesting"""
        self.baseline_win_rate = win_rate
        self.baseline_sharpe = sharpe
        self.baseline_profit_factor = profit_factor
        logger.info(f"Baseline set: WR={win_rate:.2%}, Sharpe={sharpe:.2f}")

    def check_health(self, trades_df: pd.DataFrame,
                     recent_predictions: Optional[pd.DataFrame] = None) -> HealthMetrics:
        """
        Comprehensive health check

        Args:
            trades_df: DataFrame with recent trades
            recent_predictions: DataFrame with model predictions and confidence

        Returns:
            HealthMetrics with diagnosis and recommendations
        """
        warnings = []

        # 1. Win Rate Analysis
        win_rate_30 = self._calculate_win_rate(trades_df, window=30)
        win_rate_100 = self._calculate_win_rate(trades_df, window=100)

        # 2. Sharpe Ratio (rolling 30 days)
        sharpe_30d = self._calculate_rolling_sharpe(trades_df, days=30)

        # 3. Current Drawdown
        current_drawdown = self._calculate_current_drawdown(trades_df)

        # 4. Consecutive Losses
        consecutive_losses = self._count_consecutive_losses(trades_df)

        # 5. Model Confidence (if available)
        avg_confidence = 0.0
        if recent_predictions is not None and 'confidence' in recent_predictions.columns:
            avg_confidence = recent_predictions['confidence'].tail(20).mean()

        # 6. Model Drift Detection
        drift_score = self._detect_model_drift(trades_df)

        # 7. Correlation Breakdown
        correlation_breakdown = self._check_correlation_breakdown(trades_df)

        # Determine Health Status
        status, action = self._determine_status(
            win_rate_30, sharpe_30d, current_drawdown,
            consecutive_losses, avg_confidence, drift_score
        )

        # Generate warnings
        if win_rate_30 < self.min_win_rate:
            warnings.append(f"Win rate dropped to {win_rate_30:.1%} (threshold: {self.min_win_rate:.1%})")

        if sharpe_30d < self.min_sharpe:
            warnings.append(f"Sharpe ratio at {sharpe_30d:.2f} (threshold: {self.min_sharpe:.2f})")

        if consecutive_losses >= self.max_consecutive_losses:
            warnings.append(f"Consecutive losses: {consecutive_losses} (threshold: {self.max_consecutive_losses})")

        if current_drawdown > self.max_drawdown:
            warnings.append(f"Drawdown at {current_drawdown:.1%} (threshold: {self.max_drawdown:.1%})")

        if avg_confidence > 0 and avg_confidence < self.min_confidence:
            warnings.append(f"Model confidence dropped to {avg_confidence:.1%}")

        if drift_score > 0.3:
            warnings.append(f"Model drift detected: {drift_score:.2f}")

        if correlation_breakdown > 0.5:
            warnings.append(f"Correlation breakdown: {correlation_breakdown:.2f}")

        return HealthMetrics(
            timestamp=datetime.now(),
            win_rate_30=win_rate_30,
            win_rate_100=win_rate_100,
            sharpe_30d=sharpe_30d,
            current_drawdown=current_drawdown,
            consecutive_losses=consecutive_losses,
            prediction_confidence=avg_confidence,
            model_drift_score=drift_score,
            correlation_breakdown=correlation_breakdown,
            status=status,
            recommended_action=action,
            warnings=warnings
        )

    def _calculate_win_rate(self, trades_df: pd.DataFrame, window: int) -> float:
        """Calculate win rate for last N trades"""
        if len(trades_df) < window:
            window = len(trades_df)

        if window == 0:
            return 0.0

        recent = trades_df.tail(window)
        wins = len(recent[recent['pnl'] > 0])
        return wins / window

    def _calculate_rolling_sharpe(self, trades_df: pd.DataFrame, days: int) -> float:
        """Calculate rolling Sharpe ratio"""
        if len(trades_df) < 20:
            return 0.0

        cutoff = datetime.now() - timedelta(days=days)
        recent = trades_df[trades_df['exit_time'] >= cutoff.isoformat()]

        if len(recent) < 10:
            return 0.0

        returns = recent['pnl_percent'].values
        if np.std(returns) == 0:
            return 0.0

        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
        return sharpe

    def _calculate_current_drawdown(self, trades_df: pd.DataFrame) -> float:
        """Calculate current drawdown from peak"""
        if len(trades_df) == 0:
            return 0.0

        cumulative_pnl = trades_df['pnl'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = (cumulative_pnl - running_max) / running_max.abs()

        return abs(drawdown.iloc[-1]) if len(drawdown) > 0 else 0.0

    def _count_consecutive_losses(self, trades_df: pd.DataFrame) -> int:
        """Count current consecutive losing trades"""
        if len(trades_df) == 0:
            return 0

        count = 0
        for _, trade in trades_df.iloc[::-1].iterrows():
            if trade['pnl'] <= 0:
                count += 1
            else:
                break

        return count

    def _detect_model_drift(self, trades_df: pd.DataFrame) -> float:
        """
        Detect if model performance is drifting from baseline

        Returns:
            Drift score (0-1, higher = more drift)
        """
        if self.baseline_win_rate is None or len(trades_df) < 50:
            return 0.0

        recent_30 = self._calculate_win_rate(trades_df, 30)
        recent_100 = self._calculate_win_rate(trades_df, 100)

        # Compare to baseline
        drift_30 = abs(recent_30 - self.baseline_win_rate) / self.baseline_win_rate
        drift_100 = abs(recent_100 - self.baseline_win_rate) / self.baseline_win_rate

        # Weighted average (recent drift is more important)
        drift_score = 0.7 * drift_30 + 0.3 * drift_100

        return min(drift_score, 1.0)

    def _check_correlation_breakdown(self, trades_df: pd.DataFrame) -> float:
        """
        Check if live performance correlation with backtest is breaking down

        Returns:
            Breakdown score (0-1, higher = worse)
        """
        if len(trades_df) < 30:
            return 0.0

        # Compare recent metrics to baseline
        recent_win_rate = self._calculate_win_rate(trades_df, 30)
        recent_sharpe = self._calculate_rolling_sharpe(trades_df, 30)

        if self.baseline_win_rate and self.baseline_sharpe:
            wr_diff = abs(recent_win_rate - self.baseline_win_rate)
            sharpe_diff = abs(recent_sharpe - self.baseline_sharpe) / max(abs(self.baseline_sharpe), 1.0)

            breakdown = (wr_diff + sharpe_diff) / 2
            return min(breakdown, 1.0)

        return 0.0

    def _determine_status(self, win_rate: float, sharpe: float,
                          drawdown: float, consecutive_losses: int,
                          confidence: float, drift: float) -> Tuple[HealthStatus, RecoveryAction]:
        """Determine overall health status and recommended action"""

        # CRITICAL: Immediate stop needed
        if (consecutive_losses >= self.max_consecutive_losses or
            drawdown > self.max_drawdown or
                win_rate < 0.3):
            return HealthStatus.FAILED, RecoveryAction.STOP_TRADING

        # CRITICAL: Close positions and pause
        if (sharpe < -0.5 or
            win_rate < 0.35 or
                drift > 0.5):
            return HealthStatus.CRITICAL, RecoveryAction.CLOSE_POSITIONS

        # WARNING: Reduce risk
        if (sharpe < self.min_sharpe or
            win_rate < self.min_win_rate or
            consecutive_losses >= 3 or
            drift > 0.3 or
                confidence < self.min_confidence):
            return HealthStatus.WARNING, RecoveryAction.REDUCE_SIZE

        # HEALTHY: Continue
        return HealthStatus.HEALTHY, RecoveryAction.CONTINUE


class RecoveryStrategy:
    """
    Implements recovery actions when models fail
    """

    def __init__(self):
        self.recovery_history = []
        self.current_position_size_multiplier = 1.0
        self.is_trading_paused = False

    def execute_recovery(self, health: HealthMetrics,
                         current_positions: List[Dict]) -> Dict:
        """
        Execute recovery action based on health metrics

        Args:
            health: Current health metrics
            current_positions: List of open positions

        Returns:
            Dictionary with recovery actions taken
        """
        action_taken = {
            'timestamp': datetime.now(),
            'status': health.status.value,
            'action': health.recommended_action.value,
            'details': []
        }

        logger.warning(f"Health Status: {health.status.value}")
        logger.warning(f"Recommended Action: {health.recommended_action.value}")

        # Execute based on recommendation
        if health.recommended_action == RecoveryAction.STOP_TRADING:
            result = self._stop_trading(current_positions, health)
            action_taken['details'] = result

        elif health.recommended_action == RecoveryAction.CLOSE_POSITIONS:
            result = self._close_all_positions(current_positions, health)
            action_taken['details'] = result

        elif health.recommended_action == RecoveryAction.REDUCE_SIZE:
            result = self._reduce_position_size(health)
            action_taken['details'] = result

        elif health.recommended_action == RecoveryAction.PAUSE_NEW_TRADES:
            result = self._pause_new_trades(current_positions, health)
            action_taken['details'] = result

        elif health.recommended_action == RecoveryAction.RETRAIN_MODEL:
            result = self._trigger_retraining(health)
            action_taken['details'] = result

        elif health.recommended_action == RecoveryAction.CONTINUE:
            # Check if we should increase size back to normal
            if self.current_position_size_multiplier < 1.0:
                if health.win_rate_30 > 0.52 and health.sharpe_30d > 1.0:
                    self._restore_normal_size()
                    action_taken['details'].append("Restored normal position size")

        # Log recovery action
        self.recovery_history.append(action_taken)

        return action_taken

    def _stop_trading(self, positions: List[Dict], health: HealthMetrics) -> List[str]:
        """Stop all trading immediately"""
        actions = []

        # Close all positions
        for pos in positions:
            logger.critical(f"CLOSING POSITION: {pos['instrument']} due to {health.status.value}")
            actions.append(f"Closed position: {pos['instrument']}")

        # Disable trading
        self.is_trading_paused = True
        actions.append("❌ ALL TRADING STOPPED")
        actions.append(f"Reason: {', '.join(health.warnings)}")

        # Send emergency alert
        self._send_emergency_alert(health)
        actions.append("🚨 Emergency alert sent")

        return actions

    def _close_all_positions(self, positions: List[Dict], health: HealthMetrics) -> List[str]:
        """Close all positions but allow new trades"""
        actions = []

        for pos in positions:
            logger.warning(f"Closing position: {pos['instrument']}")
            actions.append(f"Closed: {pos['instrument']}")

        # Pause new trades temporarily
        self.is_trading_paused = True
        actions.append("⏸ New trades paused for review")
        actions.append(f"Triggers: {', '.join(health.warnings)}")

        return actions

    def _reduce_position_size(self, health: HealthMetrics) -> List[str]:
        """Reduce position sizing"""
        actions = []

        # Progressive reduction based on severity
        if health.status == HealthStatus.WARNING:
            reduction_factor = 0.5  # 50% size
        else:
            reduction_factor = 0.25  # 25% size

        self.current_position_size_multiplier = reduction_factor

        actions.append(f"⚠️ Position size reduced to {reduction_factor * 100:.0f}%")
        actions.append(f"Reason: {', '.join(health.warnings)}")

        logger.warning(f"Position size reduced to {reduction_factor * 100:.0f}%")

        return actions

    def _pause_new_trades(self, positions: List[Dict], health: HealthMetrics) -> List[str]:
        """Pause new trades, keep existing positions"""
        actions = []

        self.is_trading_paused = True
        actions.append("⏸ New trades paused")
        actions.append(f"Open positions: {len(positions)} maintained")
        actions.append(f"Monitoring: {', '.join(health.warnings)}")

        return actions

    def _trigger_retraining(self, health: HealthMetrics) -> List[str]:
        """Trigger model retraining"""
        actions = []

        actions.append("🔄 Model retraining triggered")
        actions.append(f"Reason: Model drift = {health.model_drift_score:.2f}")

        # In production, this would trigger actual retraining pipeline
        logger.info("Model retraining scheduled")

        return actions

    def _restore_normal_size(self):
        """Restore normal position sizing after recovery"""
        self.current_position_size_multiplier = 1.0
        self.is_trading_paused = False
        logger.info("✅ Normal position sizing restored")

    def _send_emergency_alert(self, health: HealthMetrics):
        """Send emergency notification"""
        # In production, integrate with email/SMS/Slack
        logger.critical("=" * 60)
        logger.critical("🚨 EMERGENCY: TRADING SYSTEM HALTED")
        logger.critical("=" * 60)
        logger.critical(f"Status: {health.status.value}")
        logger.critical(f"Win Rate (30): {health.win_rate_30:.1%}")
        logger.critical(f"Sharpe (30d): {health.sharpe_30d:.2f}")
        logger.critical(f"Drawdown: {health.current_drawdown:.1%}")
        logger.critical(f"Consecutive Losses: {health.consecutive_losses}")
        logger.critical("Warnings:")
        for warning in health.warnings:
            logger.critical(f"  - {warning}")
        logger.critical("=" * 60)


class AdaptivePositionSizer:
    """
    Dynamically adjusts position sizes based on recent performance
    """

    def __init__(self, base_risk_pct: float = 0.02):
        self.base_risk_pct = base_risk_pct
        self.current_multiplier = 1.0

    def calculate_size(self, capital: float, trades_df: pd.DataFrame,
                       stop_loss_distance: float) -> float:
        """
        Calculate position size with dynamic adjustment

        Args:
            capital: Current capital
            trades_df: Recent trade history
            stop_loss_distance: Distance to stop loss

        Returns:
            Adjusted position size
        """
        # Base position size
        base_size = (capital * self.base_risk_pct) / stop_loss_distance

        # Calculate performance multiplier
        multiplier = self._calculate_multiplier(trades_df)

        # Apply multiplier
        adjusted_size = base_size * multiplier

        return adjusted_size

    def _calculate_multiplier(self, trades_df: pd.DataFrame) -> float:
        """
        Calculate position size multiplier based on recent performance

        Strategy:
        - High win rate + positive Sharpe → Increase size (up to 1.5x)
        - Low win rate or negative Sharpe → Decrease size (down to 0.25x)
        """
        if len(trades_df) < 20:
            return 0.5  # Conservative until proven

        # Recent performance
        last_20 = trades_df.tail(20)
        win_rate = len(last_20[last_20['pnl'] > 0]) / 20

        returns = last_20['pnl_percent'].values
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0

        # Calculate multiplier
        if win_rate > 0.60 and sharpe > 1.5:
            multiplier = 1.5  # Increase size
        elif win_rate > 0.55 and sharpe > 1.0:
            multiplier = 1.2
        elif win_rate > 0.50 and sharpe > 0.5:
            multiplier = 1.0  # Normal size
        elif win_rate > 0.45 and sharpe > 0:
            multiplier = 0.75  # Reduce size
        elif win_rate > 0.40:
            multiplier = 0.5  # Significantly reduce
        else:
            multiplier = 0.25  # Minimal size

        return multiplier


# Example usage
if __name__ == "__main__":
    # Simulate trade data
    np.random.seed(42)

    # Simulating deteriorating performance
    trades_data = []
    for i in range(100):
        # First 50 trades: Good performance (60% win rate)
        if i < 50:
            win = np.random.random() < 0.60
        # Last 50 trades: Deteriorating (35% win rate)
        else:
            win = np.random.random() < 0.35

        pnl = np.random.uniform(50, 150) if win else -np.random.uniform(40, 120)
        pnl_pct = pnl / 10000 * 100

        trades_data.append({
            'exit_time': (datetime.now() - timedelta(days=100 - i)).isoformat(),
            'pnl': pnl,
            'pnl_percent': pnl_pct
        })

    trades_df = pd.DataFrame(trades_data)

    # Initialize detector
    detector = ModelFailureDetector(
        min_win_rate=0.45,
        min_sharpe=0.5,
        max_consecutive_losses=5,
        max_drawdown=0.15
    )

    # Set baseline from "backtesting"
    detector.set_baseline(win_rate=0.58, sharpe=1.2, profit_factor=1.8)

    # Check health
    health = detector.check_health(trades_df)

    # Print results
    print("\n" + "=" * 60)
    print("HEALTH CHECK RESULTS")
    print("=" * 60)
    print(f"Status: {health.status.value.upper()}")
    print(f"Recommended Action: {health.recommended_action.value}")
    print("\nMetrics:")
    print(f"  Win Rate (30 trades): {health.win_rate_30:.1%}")
    print(f"  Win Rate (100 trades): {health.win_rate_100:.1%}")
    print(f"  Sharpe Ratio (30 days): {health.sharpe_30d:.2f}")
    print(f"  Current Drawdown: {health.current_drawdown:.1%}")
    print(f"  Consecutive Losses: {health.consecutive_losses}")
    print(f"  Model Drift Score: {health.model_drift_score:.2f}")

    print(f"\n⚠️ WARNINGS ({len(health.warnings)}):")
    for warning in health.warnings:
        print(f"  - {warning}")

    # Execute recovery
    print("\n" + "=" * 60)
    print("EXECUTING RECOVERY")
    print("=" * 60)

    recovery = RecoveryStrategy()
    action = recovery.execute_recovery(health, current_positions=[])

    print(f"\nAction Taken: {action['action']}")
    print("Details:")
    for detail in action['details']:
        print(f"  {detail}")
