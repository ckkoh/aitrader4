"""
Example Trading Strategies: Rule-based and ML-powered
Ready to use with the backtesting engine
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
import logging

from backtesting_engine import Strategy, PositionSide
from ml_training_pipeline import MLModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MomentumStrategy(Strategy):
    """
    Simple momentum strategy based on moving average crossover
    """

    def __init__(self, fast_period: int = 20, slow_period: int = 50,
                 rsi_period: int = 14):
        super().__init__(name=f"Momentum_{fast_period}_{slow_period}")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.rsi_period = rsi_period

    def generate_signals(self, data: pd.DataFrame, timestamp: datetime) -> List[Dict]:
        """Generate momentum-based signals"""
        signals = []

        if len(data) < self.slow_period + 1:
            return signals

        instrument = 'SPX500_USD'  # Can be parameterized

        # Calculate indicators
        fast_ma = data['close'].rolling(window=self.fast_period).mean()
        slow_ma = data['close'].rolling(window=self.slow_period).mean()

        # RSI for filter
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # Current values
        fast_current = fast_ma.iloc[-1]
        slow_current = slow_ma.iloc[-1]
        fast_prev = fast_ma.iloc[-2]
        slow_prev = slow_ma.iloc[-2]
        rsi_current = rsi.iloc[-1]

        current_price = data['close'].iloc[-1]
        atr = data['close'].rolling(window=14).std().iloc[-1]

        # Check if we have a position
        has_position = self.has_position(instrument)

        # Entry signals
        if not has_position:
            # Bullish crossover
            if fast_prev <= slow_prev and fast_current > slow_current and rsi_current < 70:
                stop_loss = current_price - (2 * atr)
                take_profit = current_price + (3 * atr)

                signals.append({
                    'instrument': instrument,
                    'action': 'buy',
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'reason': 'bullish_crossover'
                })

            # Bearish crossover
            elif fast_prev >= slow_prev and fast_current < slow_current and rsi_current > 30:
                stop_loss = current_price + (2 * atr)
                take_profit = current_price - (3 * atr)

                signals.append({
                    'instrument': instrument,
                    'action': 'sell',
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'reason': 'bearish_crossover'
                })

        # Exit signals
        else:
            position = self.get_position(instrument)

            # Exit long if fast MA crosses below slow MA
            if position.direction == PositionSide.LONG:
                if fast_current < slow_current:
                    signals.append({
                        'instrument': instrument,
                        'action': 'close',
                        'reason': 'exit_long_crossover'
                    })

            # Exit short if fast MA crosses above slow MA
            elif position.direction == PositionSide.SHORT:
                if fast_current > slow_current:
                    signals.append({
                        'instrument': instrument,
                        'action': 'close',
                        'reason': 'exit_short_crossover'
                    })

        return signals


class MeanReversionStrategy(Strategy):
    """
    Mean reversion strategy using Bollinger Bands
    """

    def __init__(self, bb_period: int = 20, bb_std: float = 2.0,
                 rsi_period: int = 14):
        super().__init__(name=f"MeanReversion_{bb_period}")
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period

    def generate_signals(self, data: pd.DataFrame, timestamp: datetime) -> List[Dict]:
        """Generate mean reversion signals"""
        signals = []

        if len(data) < self.bb_period + 1:
            return signals

        instrument = 'SPX500_USD'

        # Bollinger Bands
        sma = data['close'].rolling(window=self.bb_period).mean()
        std = data['close'].rolling(window=self.bb_period).std()
        upper_band = sma + (self.bb_std * std)
        lower_band = sma - (self.bb_std * std)

        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        current_price = data['close'].iloc[-1]
        upper = upper_band.iloc[-1]
        lower = lower_band.iloc[-1]
        middle = sma.iloc[-1]
        rsi_current = rsi.iloc[-1]

        atr = data['close'].rolling(window=14).std().iloc[-1]
        has_position = self.has_position(instrument)

        if not has_position:
            # Buy when price touches lower band and RSI oversold
            if current_price <= lower and rsi_current < 30:
                stop_loss = current_price - (1.5 * atr)
                take_profit = middle  # Target middle band

                signals.append({
                    'instrument': instrument,
                    'action': 'buy',
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'reason': 'oversold_mean_reversion'
                })

            # Sell when price touches upper band and RSI overbought
            elif current_price >= upper and rsi_current > 70:
                stop_loss = current_price + (1.5 * atr)
                take_profit = middle

                signals.append({
                    'instrument': instrument,
                    'action': 'sell',
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'reason': 'overbought_mean_reversion'
                })

        else:
            position = self.get_position(instrument)

            # Exit long at middle band
            if position.direction == PositionSide.LONG and current_price >= middle:
                signals.append({
                    'instrument': instrument,
                    'action': 'close',
                    'reason': 'target_reached'
                })

            # Exit short at middle band
            elif position.direction == PositionSide.SHORT and current_price <= middle:
                signals.append({
                    'instrument': instrument,
                    'action': 'close',
                    'reason': 'target_reached'
                })

        return signals


class BreakoutStrategy(Strategy):
    """
    Breakout strategy using price channels
    """

    def __init__(self, lookback_period: int = 20, atr_period: int = 14):
        super().__init__(name=f"Breakout_{lookback_period}")
        self.lookback_period = lookback_period
        self.atr_period = atr_period

    def generate_signals(self, data: pd.DataFrame, timestamp: datetime) -> List[Dict]:
        """Generate breakout signals"""
        signals = []

        if len(data) < self.lookback_period + 1:
            return signals

        instrument = 'SPX500_USD'

        # Calculate channel
        high_channel = data['high'].rolling(window=self.lookback_period).max()
        low_channel = data['low'].rolling(window=self.lookback_period).min()

        # ATR for stops
        high = data['high']
        low = data['low']
        close = data['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean()

        current_price = data['close'].iloc[-1]
        high_channel_current = high_channel.iloc[-2]  # Use previous to avoid look-ahead
        low_channel_current = low_channel.iloc[-2]
        atr_current = atr.iloc[-1]

        has_position = self.has_position(instrument)

        if not has_position:
            # Bullish breakout
            if current_price > high_channel_current:
                stop_loss = current_price - (2 * atr_current)
                take_profit = current_price + (3 * atr_current)

                signals.append({
                    'instrument': instrument,
                    'action': 'buy',
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'reason': 'bullish_breakout'
                })

            # Bearish breakout
            elif current_price < low_channel_current:
                stop_loss = current_price + (2 * atr_current)
                take_profit = current_price - (3 * atr_current)

                signals.append({
                    'instrument': instrument,
                    'action': 'sell',
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'reason': 'bearish_breakout'
                })

        return signals


class MLStrategy(Strategy):
    """
    Machine Learning powered strategy
    Uses trained ML model to generate signals
    """

    def __init__(self, model_path: str, feature_cols: List[str],
                 confidence_threshold: float = 0.6):
        super().__init__(name="ML_Strategy")
        self.model_path = model_path
        self.feature_cols = feature_cols
        self.confidence_threshold = confidence_threshold
        self.trainer = None
        self.last_features = None

        # Load model
        self._load_model()

    def _load_model(self):
        """Load trained ML model"""
        try:
            self.trainer = MLModelTrainer.load_model(self.model_path)
            logger.info(f"ML model loaded: {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading ML model: {e}")
            raise

    def train(self, data: pd.DataFrame):
        """
        Train/update model on new data
        This would be called during walk-forward analysis
        """
        from ml_training_pipeline import MLTradingPipeline

        logger.info("Training ML model on new data...")

        # Create features
        pipeline = MLTradingPipeline()
        _df_features = pipeline.load_and_prepare_data(data, include_volume=False)  # noqa: F841

        # Train
        _results = pipeline.train_model(  # noqa: F841
            model_type='xgboost',
            hyperparameter_tuning=False,
            cross_validation=False
        )

        # Update model
        self.trainer = pipeline.trainer
        self.feature_cols = pipeline.feature_cols

        logger.info("ML model training complete")

    def generate_signals(self, data: pd.DataFrame, timestamp: datetime) -> List[Dict]:
        """Generate ML-based signals"""
        signals = []

        if len(data) < 200:  # Need enough data for features
            return signals

        instrument = 'SPX500_USD'

        try:
            # Get current features
            from feature_engineering import FeatureEngineering

            # Calculate features for current data (must match training!)
            data_with_features = FeatureEngineering.build_complete_feature_set(
                data.copy(), include_volume=True
            )

            # Get last row features
            latest = data_with_features.iloc[-1]

            # Extract feature values
            X = np.array([latest[self.feature_cols].values])

            # Get prediction and probability
            prediction = self.trainer.predict(X)[0]
            proba = self.trainer.predict_proba(X)[0]

            # Confidence for the predicted class
            confidence = proba[prediction]

            logger.debug(f"ML Prediction: {prediction}, Confidence: {confidence:.3f}")

            # Only trade if confidence is high enough
            if confidence < self.confidence_threshold:
                return signals

            current_price = data['close'].iloc[-1]
            atr = data['close'].rolling(window=14).std().iloc[-1]
            has_position = self.has_position(instrument)

            if not has_position:
                # Buy signal (prediction = 1)
                if prediction == 1:
                    stop_loss = current_price - (2 * atr)
                    take_profit = current_price + (3 * atr)

                    signals.append({
                        'instrument': instrument,
                        'action': 'buy',
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'reason': f'ml_buy_confidence_{confidence:.2f}'
                    })

                # Sell signal (prediction = 0 or -1)
                elif prediction == 0 or prediction == -1:
                    stop_loss = current_price + (2 * atr)
                    take_profit = current_price - (3 * atr)

                    signals.append({
                        'instrument': instrument,
                        'action': 'sell',
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'reason': f'ml_sell_confidence_{confidence:.2f}'
                    })

            else:
                # Exit logic - exit if prediction reverses
                position = self.get_position(instrument)

                if position.direction == PositionSide.LONG and prediction != 1:
                    signals.append({
                        'instrument': instrument,
                        'action': 'close',
                        'reason': 'ml_exit_long'
                    })

                elif position.direction == PositionSide.SHORT and prediction == 1:
                    signals.append({
                        'instrument': instrument,
                        'action': 'close',
                        'reason': 'ml_exit_short'
                    })

        except Exception as e:
            logger.error(f"Error generating ML signals: {e}")

        return signals


class EnsembleStrategy(Strategy):
    """
    Ensemble strategy that combines multiple strategies
    """

    def __init__(self, strategies: List[Strategy], voting_method: str = 'majority'):
        super().__init__(name="Ensemble_Strategy")
        self.strategies = strategies
        self.voting_method = voting_method

    def generate_signals(self, data: pd.DataFrame, timestamp: datetime) -> List[Dict]:
        """Generate signals by combining multiple strategies"""

        # Get signals from all strategies
        all_signals = []
        for strategy in self.strategies:
            signals = strategy.generate_signals(data, timestamp)
            all_signals.extend(signals)

        if not all_signals:
            return []

        # Count votes for each action
        buy_votes = sum(1 for s in all_signals if s['action'] == 'buy')
        sell_votes = sum(1 for s in all_signals if s['action'] == 'sell')
        close_votes = sum(1 for s in all_signals if s['action'] == 'close')

        total_strategies = len(self.strategies)

        # Majority voting
        if self.voting_method == 'majority':
            threshold = total_strategies / 2

            if buy_votes > threshold:
                # Average the stop loss and take profit from buy signals
                buy_signals = [s for s in all_signals if s['action'] == 'buy']
                avg_stop = np.mean([s['stop_loss'] for s in buy_signals])
                avg_tp = np.mean([s['take_profit'] for s in buy_signals])

                return [{
                    'instrument': 'SPX500_USD',
                    'action': 'buy',
                    'stop_loss': avg_stop,
                    'take_profit': avg_tp,
                    'reason': f'ensemble_buy_{buy_votes}/{total_strategies}'
                }]

            elif sell_votes > threshold:
                sell_signals = [s for s in all_signals if s['action'] == 'sell']
                avg_stop = np.mean([s['stop_loss'] for s in sell_signals])
                avg_tp = np.mean([s['take_profit'] for s in sell_signals])

                return [{
                    'instrument': 'SPX500_USD',
                    'action': 'sell',
                    'stop_loss': avg_stop,
                    'take_profit': avg_tp,
                    'reason': f'ensemble_sell_{sell_votes}/{total_strategies}'
                }]

            elif close_votes > threshold:
                return [{
                    'instrument': 'SPX500_USD',
                    'action': 'close',
                    'reason': f'ensemble_close_{close_votes}/{total_strategies}'
                }]

        # Unanimous voting - all strategies must agree
        elif self.voting_method == 'unanimous':
            if buy_votes == total_strategies:
                buy_signals = [s for s in all_signals if s['action'] == 'buy']
                avg_stop = np.mean([s['stop_loss'] for s in buy_signals])
                avg_tp = np.mean([s['take_profit'] for s in buy_signals])

                return [{
                    'instrument': 'SPX500_USD',
                    'action': 'buy',
                    'stop_loss': avg_stop,
                    'take_profit': avg_tp,
                    'reason': 'ensemble_unanimous_buy'
                }]

            elif sell_votes == total_strategies:
                sell_signals = [s for s in all_signals if s['action'] == 'sell']
                avg_stop = np.mean([s['stop_loss'] for s in sell_signals])
                avg_tp = np.mean([s['take_profit'] for s in sell_signals])

                return [{
                    'instrument': 'SPX500_USD',
                    'action': 'sell',
                    'stop_loss': avg_stop,
                    'take_profit': avg_tp,
                    'reason': 'ensemble_unanimous_sell'
                }]

        return []


# Example: Adaptive strategy that changes parameters based on volatility
class AdaptiveMomentumStrategy(Strategy):
    """
    Adaptive momentum strategy that adjusts parameters based on market volatility
    """

    def __init__(self):
        super().__init__(name="Adaptive_Momentum")
        self.current_fast = 20
        self.current_slow = 50

    def generate_signals(self, data: pd.DataFrame, timestamp: datetime) -> List[Dict]:
        """Generate adaptive signals"""

        if len(data) < 100:
            return []

        # Calculate current volatility regime
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=20).std().iloc[-1]

        # Adapt parameters based on volatility
        if volatility > 0.002:  # High volatility
            self.current_fast = 10
            self.current_slow = 30
        else:  # Low volatility
            self.current_fast = 30
            self.current_slow = 70

        # Use momentum strategy with adapted parameters
        adapted_strategy = MomentumStrategy(self.current_fast, self.current_slow)
        adapted_strategy.positions = self.positions

        return adapted_strategy.generate_signals(data, timestamp)
