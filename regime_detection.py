"""
Market Regime Detection Module

Detects market regimes (bull, bear, sideways) to adapt trading strategy.
Uses multiple indicators including 200-day MA, trend strength, and volatility.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"


class RegimeDetector:
    """
    Detects market regimes using multiple indicators

    Primary indicator: 200-day MA
    Secondary: Trend strength (ADX-like), volatility
    """

    def __init__(self,
                 ma_period: int = 200,
                 trend_lookback: int = 50,
                 volatility_period: int = 20):
        """
        Initialize regime detector

        Args:
            ma_period: Moving average period for trend detection (default 200)
            trend_lookback: Days to calculate trend strength (default 50)
            volatility_period: Period for volatility calculation (default 20)
        """
        self.ma_period = ma_period
        self.trend_lookback = trend_lookback
        self.volatility_period = volatility_period

    def detect_regime(self, data: pd.DataFrame) -> MarketRegime:
        """
        Detect current market regime

        Args:
            data: DataFrame with OHLC data (needs 'close' column)

        Returns:
            MarketRegime enum
        """
        if len(data) < self.ma_period:
            logger.warning(f"Insufficient data for regime detection (need {self.ma_period}, got {len(data)})")
            return MarketRegime.SIDEWAYS  # Default to neutral

        # Calculate 200-day MA
        ma_200 = data['close'].rolling(window=self.ma_period).mean()

        # Current price vs MA
        current_price = data['close'].iloc[-1]
        current_ma = ma_200.iloc[-1]

        # Price position relative to MA
        price_vs_ma = (current_price - current_ma) / current_ma

        # MA slope (trend direction and strength)
        ma_slope = (ma_200.iloc[-1] - ma_200.iloc[-self.trend_lookback]) / ma_200.iloc[-self.trend_lookback]

        # Volatility (for detecting choppy markets)
        volatility = data['close'].pct_change().rolling(window=self.volatility_period).std().iloc[-1]
        avg_volatility = data['close'].pct_change().rolling(window=self.ma_period).std().mean()

        # Trend strength (% of days price was above MA in lookback)
        recent_prices = data['close'].iloc[-self.trend_lookback:]
        recent_ma = ma_200.iloc[-self.trend_lookback:]
        trend_strength = (recent_prices > recent_ma).sum() / self.trend_lookback

        logger.debug(f"Regime Detection Metrics:")
        logger.debug(f"  Price vs MA: {price_vs_ma:.2%}")
        logger.debug(f"  MA Slope: {ma_slope:.2%}")
        logger.debug(f"  Trend Strength: {trend_strength:.2%}")
        logger.debug(f"  Volatility: {volatility:.4f} (avg: {avg_volatility:.4f})")

        # High volatility regime
        if volatility > avg_volatility * 1.5:
            logger.info("Regime: VOLATILE (high volatility)")
            return MarketRegime.VOLATILE

        # Bull market criteria
        if (price_vs_ma > 0.02 and  # Price >2% above MA
            ma_slope > 0.01 and      # MA rising >1%
            trend_strength > 0.6):   # Price above MA 60%+ of time
            logger.info("Regime: BULL")
            return MarketRegime.BULL

        # Bear market criteria
        if (price_vs_ma < -0.02 and  # Price <2% below MA
            ma_slope < -0.01 and      # MA falling >1%
            trend_strength < 0.4):    # Price below MA 60%+ of time
            logger.info("Regime: BEAR")
            return MarketRegime.BEAR

        # Default to sideways
        logger.info("Regime: SIDEWAYS (no clear trend)")
        return MarketRegime.SIDEWAYS

    def add_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add regime-related features to DataFrame

        Args:
            data: DataFrame with OHLC data

        Returns:
            DataFrame with added regime features
        """
        df = data.copy()

        # 200-day MA
        df['ma_200'] = df['close'].rolling(window=self.ma_period).mean()

        # Price vs MA features
        df['price_vs_ma_200'] = (df['close'] - df['ma_200']) / df['ma_200']
        df['above_ma_200'] = (df['close'] > df['ma_200']).astype(int)

        # MA slope (trend strength)
        df['ma_200_slope'] = df['ma_200'].pct_change(periods=self.trend_lookback)

        # Trend strength (rolling % above MA)
        df['trend_strength'] = df['above_ma_200'].rolling(window=self.trend_lookback).mean()

        # Volatility features
        df['volatility_20d'] = df['close'].pct_change().rolling(window=self.volatility_period).std()
        df['volatility_200d'] = df['close'].pct_change().rolling(window=self.ma_period).std()
        df['volatility_regime'] = df['volatility_20d'] / df['volatility_200d']

        # Regime classification (for each row)
        df['regime'] = self._classify_regime_series(df)

        logger.info(f"Added {7} regime features to DataFrame")

        return df

    def _classify_regime_series(self, df: pd.DataFrame) -> pd.Series:
        """
        Classify regime for each row in DataFrame

        Returns:
            Series with regime classification (0=bear, 1=sideways, 2=bull, 3=volatile)
        """
        regime = pd.Series(1, index=df.index)  # Default sideways

        # Bull: price >2% above MA, MA rising, strong trend
        bull_mask = (
            (df['price_vs_ma_200'] > 0.02) &
            (df['ma_200_slope'] > 0.01) &
            (df['trend_strength'] > 0.6)
        )
        regime[bull_mask] = 2

        # Bear: price <2% below MA, MA falling, weak trend
        bear_mask = (
            (df['price_vs_ma_200'] < -0.02) &
            (df['ma_200_slope'] < -0.01) &
            (df['trend_strength'] < 0.4)
        )
        regime[bear_mask] = 0

        # Volatile: high volatility relative to average
        volatile_mask = (df['volatility_regime'] > 1.5)
        regime[volatile_mask] = 3

        return regime

    def get_regime_parameters(self, regime: MarketRegime) -> Dict:
        """
        Get trading parameters optimized for current regime

        Args:
            regime: Current market regime

        Returns:
            Dictionary with recommended parameters
        """
        if regime == MarketRegime.BULL:
            return {
                'confidence_threshold': 0.50,  # Lower threshold (more trades)
                'position_size_pct': 0.025,    # Larger positions
                'stop_loss_atr_mult': 2.5,     # Wider stops (let winners run)
                'take_profit_atr_mult': 4.0,   # Higher targets
                'max_daily_loss_pct': 0.04,    # Less restrictive
                'description': 'Bull market: Momentum-following, wider stops'
            }

        elif regime == MarketRegime.BEAR:
            return {
                'confidence_threshold': 0.65,  # Higher threshold (selective)
                'position_size_pct': 0.015,    # Smaller positions
                'stop_loss_atr_mult': 1.5,     # Tighter stops (protect capital)
                'take_profit_atr_mult': 2.0,   # Lower targets (take quick profits)
                'max_daily_loss_pct': 0.02,    # Very restrictive
                'description': 'Bear market: Defensive, tight risk control'
            }

        elif regime == MarketRegime.VOLATILE:
            return {
                'confidence_threshold': 0.70,  # Very high threshold
                'position_size_pct': 0.010,    # Small positions
                'stop_loss_atr_mult': 3.0,     # Wide stops (avoid whipsaws)
                'take_profit_atr_mult': 2.5,   # Moderate targets
                'max_daily_loss_pct': 0.015,   # Very restrictive
                'description': 'Volatile market: Minimal exposure, avoid whipsaws'
            }

        else:  # SIDEWAYS
            return {
                'confidence_threshold': 0.55,  # Standard threshold
                'position_size_pct': 0.020,    # Standard position
                'stop_loss_atr_mult': 2.0,     # Standard stops
                'take_profit_atr_mult': 3.0,   # Standard targets
                'max_daily_loss_pct': 0.03,    # Standard
                'description': 'Sideways market: Standard mean-reversion'
            }

    def get_regime_statistics(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate statistics for each regime period

        Args:
            data: DataFrame with regime features added

        Returns:
            DataFrame with regime statistics
        """
        if 'regime' not in data.columns:
            data = self.add_regime_features(data)

        regime_map = {0: 'Bear', 1: 'Sideways', 2: 'Bull', 3: 'Volatile'}

        stats = []
        for regime_code, regime_name in regime_map.items():
            regime_data = data[data['regime'] == regime_code]

            if len(regime_data) > 0:
                returns = regime_data['close'].pct_change()

                stats.append({
                    'regime': regime_name,
                    'days': len(regime_data),
                    'pct_of_total': len(regime_data) / len(data) * 100,
                    'avg_return': returns.mean() * 100,
                    'volatility': returns.std() * 100,
                    'sharpe': (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
                    'max_gain': returns.max() * 100,
                    'max_loss': returns.min() * 100
                })

        return pd.DataFrame(stats)


def detect_current_regime(data: pd.DataFrame, ma_period: int = 200) -> Tuple[MarketRegime, Dict]:
    """
    Convenience function to detect current regime and get parameters

    Args:
        data: DataFrame with OHLC data
        ma_period: Moving average period (default 200)

    Returns:
        Tuple of (regime, parameters_dict)
    """
    detector = RegimeDetector(ma_period=ma_period)
    regime = detector.detect_regime(data)
    params = detector.get_regime_parameters(regime)

    return regime, params


if __name__ == "__main__":
    # Example usage and testing
    import matplotlib.pyplot as plt

    # Test with sample data (would normally load real data)
    print("Regime Detection Module")
    print("=" * 60)
    print("\nExample regime parameters:")

    detector = RegimeDetector()

    for regime in MarketRegime:
        params = detector.get_regime_parameters(regime)
        print(f"\n{regime.value.upper()} Market:")
        print(f"  Confidence Threshold: {params['confidence_threshold']:.2f}")
        print(f"  Position Size: {params['position_size_pct']:.1%}")
        print(f"  Stop Loss: {params['stop_loss_atr_mult']:.1f}x ATR")
        print(f"  Take Profit: {params['take_profit_atr_mult']:.1f}x ATR")
        print(f"  Description: {params['description']}")
