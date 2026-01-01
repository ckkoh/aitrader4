#!/usr/bin/env python3
"""
Regime-Adaptive ML Strategy
From IMPROVEMENTS_PLAN.md - Phase 2

Adapts trading behavior based on market regime:
- Bull markets: Aggressive (lower threshold, full position size)
- Bear markets: Conservative (higher threshold, reduced size or skip)
- Volatile markets: Very conservative (very high threshold, minimal size)
- Sideways markets: Moderate (medium threshold, reduced size)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

from backtesting_engine import Strategy
from ml_training_pipeline import MLModelTrainer


class RegimeDetector:
    """
    Detect current market regime based on multiple indicators
    """

    @staticmethod
    def detect_regime(data: pd.DataFrame, lookback: int = 50) -> Dict:
        """
        Detect current market regime

        Args:
            data: DataFrame with OHLC data and features
            lookback: Lookback period for regime detection

        Returns:
            Dict with regime info
        """
        if len(data) < lookback:
            return {
                'regime': 'UNKNOWN',
                'confidence': 0.0,
                'volatility': 0.0,
                'trend_strength': 0.0,
                'reason': 'Insufficient data'
            }

        recent = data.iloc[-lookback:]
        current = data.iloc[-1]

        # 1. Calculate volatility (ATR-based)
        if 'atr_14' in data.columns:
            volatility = current['atr_14'] / current['close'] * 100
        else:
            returns = recent['close'].pct_change()
            volatility = returns.std() * np.sqrt(252) * 100

        # 2. Calculate trend strength
        # Use 50-day vs 200-day SMA if available, otherwise simple trend
        if 'sma_50' in data.columns and 'sma_200' in data.columns:
            sma_50 = current['sma_50']
            sma_200 = current['sma_200']
            price = current['close']

            # Trend direction
            if price > sma_50 > sma_200:
                trend_direction = 1  # Bullish
            elif price < sma_50 < sma_200:
                trend_direction = -1  # Bearish
            else:
                trend_direction = 0  # Mixed

            # Trend strength (distance between SMAs)
            trend_strength = abs(sma_50 - sma_200) / sma_200 * 100

        else:
            # Fallback: linear regression slope
            prices = recent['close'].values
            x = np.arange(len(prices))
            slope, _ = np.polyfit(x, prices, 1)
            trend_direction = 1 if slope > 0 else -1 if slope < 0 else 0
            trend_strength = abs(slope) / prices.mean() * 100 * lookback

        # 3. Calculate momentum (ROC)
        if 'roc_20' in data.columns:
            momentum = current['roc_20']
        else:
            momentum = (current['close'] - data.iloc[-20]['close']) / data.iloc[-20]['close'] * 100

        # 4. Regime classification
        regime_info = RegimeDetector._classify_regime(
            volatility, trend_strength, trend_direction, momentum
        )

        regime_info['volatility'] = volatility
        regime_info['trend_strength'] = trend_strength
        regime_info['trend_direction'] = trend_direction
        regime_info['momentum'] = momentum

        return regime_info

    @staticmethod
    def _classify_regime(volatility: float, trend_strength: float,
                        trend_direction: int, momentum: float) -> Dict:
        """
        Classify market regime based on indicators

        Returns:
            Dict with regime, confidence, and reason
        """
        # Thresholds (FIXED: lowered from 20% to 3% for ATR-based volatility)
        HIGH_VOL = 3.0  # 3% ATR volatility (not annualized)
        STRONG_TREND = 3.0  # 3% distance between SMAs (was 5%)
        STRONG_MOMENTUM = 10  # 10% momentum

        # 1. VOLATILE regime (highest priority)
        if volatility > HIGH_VOL:
            return {
                'regime': 'VOLATILE',
                'confidence': min(volatility / HIGH_VOL, 2.0),
                'reason': f'High volatility ({volatility:.1f}%)'
            }

        # 2. BEAR regime (FIXED: added momentum condition for crossover periods)
        if (trend_direction == -1 and trend_strength > STRONG_TREND) or \
           (momentum < -STRONG_MOMENTUM and volatility > 2.0):
            return {
                'regime': 'BEAR',
                'confidence': max(
                    min(trend_strength / STRONG_TREND, 2.0) if trend_direction == -1 else 0,
                    min(abs(momentum) / STRONG_MOMENTUM, 2.0) if momentum < -STRONG_MOMENTUM else 0
                ),
                'reason': f'Bearish conditions (trend: {trend_strength:.1f}%, momentum: {momentum:.1f}%)'
            }

        # 3. BULL regime
        if trend_direction == 1 and trend_strength > STRONG_TREND:
            return {
                'regime': 'BULL',
                'confidence': min(trend_strength / STRONG_TREND, 2.0),
                'reason': f'Strong uptrend (strength: {trend_strength:.1f}%)'
            }

        # 4. SIDEWAYS regime (default)
        return {
            'regime': 'SIDEWAYS',
            'confidence': 1.0,
            'reason': f'No clear trend (strength: {trend_strength:.1f}%)'
        }


class RegimeAdaptiveMLStrategy(Strategy):
    """
    ML Strategy with regime-based adaptation

    Key Features:
    - Detects current market regime
    - Adjusts confidence thresholds by regime
    - Adjusts position sizing by regime
    - Can skip trading in unfavorable regimes
    """

    def __init__(self,
                 model_path: str,
                 feature_cols: List[str],
                 base_confidence_threshold: float = 0.55,
                 enable_regime_adaptation: bool = True,
                 skip_volatile_regimes: bool = True,
                 skip_bear_regimes: bool = False):
        """
        Initialize regime-adaptive ML strategy

        Args:
            model_path: Path to trained ML model
            feature_cols: List of feature column names
            base_confidence_threshold: Base threshold (adjusted by regime)
            enable_regime_adaptation: Enable adaptive behavior
            skip_volatile_regimes: Skip trading in volatile markets
            skip_bear_regimes: Skip trading in bear markets
        """
        super().__init__(name="RegimeAdaptiveML")

        self.model_path = model_path
        self.feature_cols = feature_cols
        self.base_confidence_threshold = base_confidence_threshold
        self.enable_regime_adaptation = enable_regime_adaptation
        self.skip_volatile_regimes = skip_volatile_regimes
        self.skip_bear_regimes = skip_bear_regimes

        # Load model (keep trainer object, not raw model)
        self.trainer = self._load_model()

        # Regime-specific settings
        self.regime_settings = {
            'BULL': {
                'confidence_adjustment': -0.10,  # Lower threshold (more aggressive)
                'position_multiplier': 1.0,  # Full position size
                'description': 'Aggressive trading in uptrend'
            },
            'BEAR': {
                'confidence_adjustment': +0.15,  # Higher threshold (conservative)
                'position_multiplier': 0.5,  # Half position size
                'description': 'Conservative trading in downtrend'
            },
            'VOLATILE': {
                'confidence_adjustment': +0.20,  # Much higher threshold
                'position_multiplier': 0.3,  # 30% position size
                'description': 'Very conservative in high volatility'
            },
            'SIDEWAYS': {
                'confidence_adjustment': +0.05,  # Slightly higher threshold
                'position_multiplier': 0.7,  # 70% position size
                'description': 'Moderate trading in ranging market'
            },
            'UNKNOWN': {
                'confidence_adjustment': +0.10,  # Conservative default
                'position_multiplier': 0.5,  # Reduced size
                'description': 'Conservative due to uncertainty'
            }
        }

        # Statistics tracking
        self.regime_stats = {
            'BULL': {'signals': 0, 'trades': 0},
            'BEAR': {'signals': 0, 'trades': 0},
            'VOLATILE': {'signals': 0, 'trades': 0},
            'SIDEWAYS': {'signals': 0, 'trades': 0},
            'UNKNOWN': {'signals': 0, 'trades': 0}
        }

    def _load_model(self):
        """Load trained ML model (returns trainer object)"""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # load_model() is a STATIC method that returns a new trainer
        trainer = MLModelTrainer.load_model(self.model_path)
        return trainer

    def get_adaptive_settings(self, regime_info: Dict) -> Dict:
        """
        Get adaptive settings for current regime

        Args:
            regime_info: Regime detection results

        Returns:
            Dict with adjusted settings
        """
        regime = regime_info['regime']
        settings = self.regime_settings.get(regime, self.regime_settings['UNKNOWN'])

        # Calculate adjusted confidence threshold
        adjusted_threshold = self.base_confidence_threshold + settings['confidence_adjustment']
        adjusted_threshold = max(0.5, min(0.9, adjusted_threshold))  # Clamp to [0.5, 0.9]

        return {
            'regime': regime,
            'confidence_threshold': adjusted_threshold,
            'position_multiplier': settings['position_multiplier'],
            'description': settings['description'],
            'skip_trading': self._should_skip_regime(regime)
        }

    def _should_skip_regime(self, regime: str) -> bool:
        """Determine if we should skip trading in this regime"""
        if not self.enable_regime_adaptation:
            return False

        if self.skip_volatile_regimes and regime == 'VOLATILE':
            return True

        if self.skip_bear_regimes and regime == 'BEAR':
            return True

        return False

    def generate_signals(self, data: pd.DataFrame, timestamp: datetime) -> List[Dict]:
        """
        Generate trading signals with regime adaptation

        Args:
            data: Historical data up to timestamp
            timestamp: Current timestamp

        Returns:
            List of trading signals
        """
        signals = []

        # DEBUG: Track signal generation attempts
        debug_mode = True  # Set to False to disable debug logging

        # Need minimum data
        if len(data) < 200:
            if debug_mode:
                print(f"DEBUG [{timestamp.date()}]: Insufficient data ({len(data)} < 200)")
            return signals

        # 1. Detect current regime
        regime_info = RegimeDetector.detect_regime(data, lookback=50)

        # 2. Get adaptive settings
        adaptive_settings = self.get_adaptive_settings(regime_info)

        if debug_mode:
            print(f"DEBUG [{timestamp.date()}]: Regime={regime_info['regime']}, "
                  f"Threshold={adaptive_settings['confidence_threshold']:.2f}, "
                  f"PosMultiplier={adaptive_settings['position_multiplier']:.1f}")

        # Track regime
        self.regime_stats[regime_info['regime']]['signals'] += 1

        # 3. Check if we should skip this regime
        if adaptive_settings['skip_trading']:
            if debug_mode:
                print(f"DEBUG [{timestamp.date()}]: Skipping {regime_info['regime']} regime")
            return signals

        # 4. Get current data point
        current = data.iloc[-1]

        # 5. Extract features
        try:
            # Check each feature individually first
            missing_features = [f for f in self.feature_cols if f not in current.index]
            if missing_features:
                if debug_mode:
                    print(f"ERROR [{timestamp.date()}]: Missing {len(missing_features)} features!")
                    print(f"  Missing: {missing_features[:5]}...")
                return signals

            # Extract features (avoid pandas fancy indexing broadcasting issues)
            feature_values = [current[f] for f in self.feature_cols]
            features = np.array(feature_values, dtype=np.float64).reshape(1, -1)
            if debug_mode:
                nan_count = np.isnan(features).sum()
                print(f"DEBUG [{timestamp.date()}]: Features extracted. "
                      f"Shape={features.shape}, NaN count={nan_count}")
                if nan_count > 0:
                    print(f"  WARNING: Features contain {nan_count} NaN values!")
        except Exception as e:
            if debug_mode:
                print(f"ERROR [{timestamp.date()}]: Feature extraction failed! {type(e).__name__}: {e}")
                print(f"  Checking each feature individually:")
                for feat in self.feature_cols:
                    try:
                        val = current[feat]
                        print(f"    {feat}: {type(val)} = {val}")
                    except Exception as fe:
                        print(f"    {feat}: ERROR - {fe}")
            return signals

        # 6. Get ML prediction (use raw model to avoid "not trained" checks)
        try:
            prediction_proba = self.trainer.model.predict_proba(features)[0]
            predicted_class = int(prediction_proba[1] > 0.5)  # 1 = buy, 0 = sell/hold
            confidence = prediction_proba[predicted_class]  # FIX: Use probability of PREDICTED class

            if debug_mode:
                print(f"DEBUG [{timestamp.date()}]: Prediction success!")
                print(f"  Proba: [SELL={prediction_proba[0]:.3f}, BUY={prediction_proba[1]:.3f}]")
                print(f"  Predicted class: {predicted_class} ({'BUY' if predicted_class == 1 else 'SELL'})")
                print(f"  Confidence: {confidence:.3f}")
        except Exception as e:
            if debug_mode:
                print(f"ERROR [{timestamp.date()}]: Prediction failed! {type(e).__name__}: {e}")
                print(f"  Features shape: {features.shape}")
                print(f"  Features contain NaN: {np.isnan(features).any()}")
                print(f"  First 5 feature values: {features[0][:5]}")
            return signals

        # 7. Check if confidence meets adjusted threshold
        if confidence < adaptive_settings['confidence_threshold']:
            if debug_mode:
                print(f"DEBUG [{timestamp.date()}]: Confidence {confidence:.3f} < threshold {adaptive_settings['confidence_threshold']:.3f} - NO SIGNAL")
            return signals

        # 8. Check if we have an open position
        has_position = len(self.positions) > 0

        if debug_mode:
            print(f"DEBUG [{timestamp.date()}]: Confidence check PASSED! Has position: {has_position}")

        if not has_position:
            # ENTRY LOGIC
            current_price = current['close']

            # Calculate ATR for stop loss/take profit
            if 'atr_14' in data.columns:
                atr = current['atr_14']
            else:
                # Fallback ATR calculation
                high_low = data['high'] - data['low']
                atr = high_low.rolling(14).mean().iloc[-1]

            # Regime-adaptive stop loss/take profit
            if regime_info['regime'] == 'VOLATILE':
                # Wider stops in volatile markets
                stop_multiplier = 3.0
                profit_multiplier = 2.0  # Lower target
            elif regime_info['regime'] == 'BULL':
                # Normal stops in bull markets
                stop_multiplier = 1.5
                profit_multiplier = 3.0
            elif regime_info['regime'] == 'BEAR':
                # Tighter stops in bear markets
                stop_multiplier = 1.0
                profit_multiplier = 2.0
            else:  # SIDEWAYS
                stop_multiplier = 2.0
                profit_multiplier = 2.5

            if predicted_class == 1:  # Buy signal
                stop_loss = current_price - (atr * stop_multiplier)
                take_profit = current_price + (atr * profit_multiplier)

                signal = {
                    'instrument': 'SPX500_USD',
                    'action': 'buy',
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'position_size_multiplier': adaptive_settings['position_multiplier'],
                    'reason': f"{regime_info['regime']}_ML_BUY_{confidence:.2f}"
                }
                signals.append(signal)

                if debug_mode:
                    print(f"✅ SIGNAL GENERATED [{timestamp.date()}]: BUY @ {current_price:.2f}")
                    print(f"  Stop: {stop_loss:.2f}, Target: {take_profit:.2f}")
                    print(f"  Reason: {signal['reason']}")

                self.regime_stats[regime_info['regime']]['trades'] += 1

            elif predicted_class == 0:  # Sell signal
                stop_loss = current_price + (atr * stop_multiplier)
                take_profit = current_price - (atr * profit_multiplier)

                signal = {
                    'instrument': 'SPX500_USD',
                    'action': 'sell',
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'position_size_multiplier': adaptive_settings['position_multiplier'],
                    'reason': f"{regime_info['regime']}_ML_SELL_{confidence:.2f}"
                }
                signals.append(signal)

                if debug_mode:
                    print(f"✅ SIGNAL GENERATED [{timestamp.date()}]: SELL @ {current_price:.2f}")
                    print(f"  Stop: {stop_loss:.2f}, Target: {take_profit:.2f}")
                    print(f"  Reason: {signal['reason']}")

                self.regime_stats[regime_info['regime']]['trades'] += 1

        else:
            # EXIT LOGIC
            # In volatile/bear markets, exit more quickly on reversal signals
            if regime_info['regime'] in ['VOLATILE', 'BEAR']:
                exit_threshold = 0.50  # Exit if reverse signal has 50%+ confidence
            else:
                exit_threshold = 0.55  # More lenient in bull/sideways

            # Exit if we have a reversal signal with high confidence
            # (This will trigger when model predicts opposite direction with high confidence)
            if confidence >= exit_threshold:
                signals.append({
                    'instrument': 'SPX500_USD',
                    'action': 'close',
                    'reason': f"{regime_info['regime']}_Exit_Reversal_{confidence:.2f}"
                })

        return signals

    def get_regime_statistics(self) -> pd.DataFrame:
        """Get statistics on regime-based trading"""
        stats_data = []
        for regime, stats in self.regime_stats.items():
            trade_rate = stats['trades'] / stats['signals'] * 100 if stats['signals'] > 0 else 0
            stats_data.append({
                'Regime': regime,
                'Signals': stats['signals'],
                'Trades': stats['trades'],
                'Trade Rate': f"{trade_rate:.1f}%"
            })

        return pd.DataFrame(stats_data)


def main():
    """Example usage"""
    print("="*80)
    print("REGIME-ADAPTIVE ML STRATEGY")
    print("="*80)

    # Top 20 features from Day 2
    TOP_20_FEATURES = [
        'bullish_engulfing',
        'stoch_d_3',
        'week_of_year',
        'atr_14',
        'regime',
        'roc_20',
        'obv',
        'parkinson_vol_10',
        'volatility_200d',
        'momentum_5',
        'macd_signal',
        'adx_14',
        'month_sin',
        'hl_ratio',
        'rsi_14',
        'stoch_k_14',
        'bb_position_20',
        'momentum_oscillator',
        'pvt',
        'price_acceleration'
    ]

    # Create strategy
    strategy = RegimeAdaptiveMLStrategy(
        model_path='feature_selection_results/model_top20.pkl',
        feature_cols=TOP_20_FEATURES,
        base_confidence_threshold=0.55,
        enable_regime_adaptation=True,
        skip_volatile_regimes=True,  # Skip volatile markets
        skip_bear_regimes=False  # Still trade in bear, but conservatively
    )

    print("\n✅ Regime-Adaptive Strategy Created")
    print("\nRegime Settings:")
    for regime, settings in strategy.regime_settings.items():
        print(f"\n  {regime}:")
        print(f"    Confidence adj: {settings['confidence_adjustment']:+.2f}")
        print(f"    Position size: {settings['position_multiplier']:.1%}")
        print(f"    Description: {settings['description']}")

    print("\n" + "="*80)
    print("Ready for walk-forward validation!")
    print("Run: python3 walkforward_regime_adaptive.py")
    print("="*80)


if __name__ == "__main__":
    main()
