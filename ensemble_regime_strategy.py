#!/usr/bin/env python3
"""
Ensemble Regime-Adaptive Strategy
Combines Original (best in bear) + Balanced (best in bull) models
Switches between models based on regime detection
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict
from pathlib import Path

from backtesting_engine import Strategy
from ml_training_pipeline import MLModelTrainer
from regime_adaptive_strategy import RegimeDetector


class EnsembleRegimeStrategy(Strategy):
    """
    Ensemble strategy that switches between Original and Balanced models
    based on market regime

    Strategy Selection:
    - BULL regime → Balanced Model (best bull performance: +52% cumulative)
    - BEAR regime → Original Model (best bear performance: +49% cumulative)
    - VOLATILE regime → Original Model (85% win rate in volatile markets)
    - SIDEWAYS regime → Balanced Model (better overall returns)
    """

    def __init__(self,
                 original_model_path: str,
                 balanced_model_path: str,
                 feature_cols: List[str],
                 base_confidence_threshold: float = 0.50,
                 enable_regime_adaptation: bool = True):
        """
        Initialize ensemble strategy

        Args:
            original_model_path: Path to original (unbalanced) model
            balanced_model_path: Path to balanced model
            feature_cols: List of feature column names
            base_confidence_threshold: Base threshold (adjusted by regime)
            enable_regime_adaptation: Enable adaptive behavior
        """
        super().__init__(name="EnsembleRegime")

        self.original_model_path = original_model_path
        self.balanced_model_path = balanced_model_path
        self.feature_cols = feature_cols
        self.base_confidence_threshold = base_confidence_threshold
        self.enable_regime_adaptation = enable_regime_adaptation

        # Load both models
        self.original_trainer = self._load_model(original_model_path)
        self.balanced_trainer = self._load_model(balanced_model_path)

        # Regime-specific settings (same as RegimeAdaptiveMLStrategy)
        self.regime_settings = {
            'BULL': {
                'confidence_adjustment': -0.10,
                'position_multiplier': 1.0,
                'use_balanced': True,  # Use balanced model in bull markets
                'description': 'Balanced model in uptrend'
            },
            'BEAR': {
                'confidence_adjustment': +0.15,
                'position_multiplier': 0.5,
                'use_balanced': False,  # Use original model in bear markets
                'description': 'Original model in downtrend'
            },
            'VOLATILE': {
                'confidence_adjustment': +0.20,
                'position_multiplier': 0.3,
                'use_balanced': False,  # Use original model in volatile markets
                'description': 'Original model in high volatility'
            },
            'SIDEWAYS': {
                'confidence_adjustment': +0.05,
                'position_multiplier': 0.7,
                'use_balanced': True,  # Use balanced model in ranging markets
                'description': 'Balanced model in ranging market'
            },
            'UNKNOWN': {
                'confidence_adjustment': +0.10,
                'position_multiplier': 0.5,
                'use_balanced': True,  # Default to balanced
                'description': 'Balanced model (default)'
            }
        }

        # Statistics tracking
        self.regime_stats = {
            'BULL': {'signals': 0, 'trades': 0, 'model_used': 'Balanced'},
            'BEAR': {'signals': 0, 'trades': 0, 'model_used': 'Original'},
            'VOLATILE': {'signals': 0, 'trades': 0, 'model_used': 'Original'},
            'SIDEWAYS': {'signals': 0, 'trades': 0, 'model_used': 'Balanced'},
            'UNKNOWN': {'signals': 0, 'trades': 0, 'model_used': 'Balanced'}
        }

    def _load_model(self, model_path: str):
        """Load trained ML model"""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        trainer = MLModelTrainer.load_model(model_path)
        return trainer

    def get_adaptive_settings(self, regime_info: Dict) -> Dict:
        """
        Get adaptive settings for current regime

        Args:
            regime_info: Regime detection results

        Returns:
            Dict with adjusted settings including model selection
        """
        regime = regime_info['regime']
        settings = self.regime_settings.get(regime, self.regime_settings['UNKNOWN'])

        # Calculate adjusted confidence threshold
        adjusted_threshold = self.base_confidence_threshold + settings['confidence_adjustment']
        adjusted_threshold = max(0.5, min(0.9, adjusted_threshold))

        return {
            'regime': regime,
            'confidence_threshold': adjusted_threshold,
            'position_multiplier': settings['position_multiplier'],
            'use_balanced': settings['use_balanced'],
            'description': settings['description'],
            'skip_trading': False  # Never skip in ensemble
        }

    def generate_signals(self, data: pd.DataFrame, timestamp: datetime) -> List[Dict]:
        """
        Generate trading signals using ensemble approach

        Args:
            data: Historical data up to timestamp
            timestamp: Current timestamp

        Returns:
            List of trading signals
        """
        signals = []

        # Need minimum data
        if len(data) < 200:
            return signals

        # 1. Detect current regime
        regime_info = RegimeDetector.detect_regime(data, lookback=50)

        # 2. Get adaptive settings (includes model selection)
        adaptive_settings = self.get_adaptive_settings(regime_info)

        # Track regime
        self.regime_stats[regime_info['regime']]['signals'] += 1

        # 3. Select model based on regime
        if adaptive_settings['use_balanced']:
            model = self.balanced_trainer.model
            model_name = 'Balanced'
        else:
            model = self.original_trainer.model
            model_name = 'Original'

        # 4. Get current data point
        current = data.iloc[-1]

        # 5. Extract features
        try:
            # Check features exist
            missing_features = [f for f in self.feature_cols if f not in current.index]
            if missing_features:
                return signals

            # Extract features (avoid pandas fancy indexing issues)
            feature_values = [current[f] for f in self.feature_cols]
            features = np.array(feature_values, dtype=np.float64).reshape(1, -1)

        except Exception as e:
            return signals

        # 6. Get ML prediction
        try:
            prediction_proba = model.predict_proba(features)[0]
            predicted_class = int(prediction_proba[1] > 0.5)
            confidence = prediction_proba[predicted_class]

        except Exception as e:
            return signals

        # 7. Check if confidence meets adjusted threshold
        if confidence < adaptive_settings['confidence_threshold']:
            return signals

        # 8. Check if we have an open position
        has_position = len(self.positions) > 0

        if not has_position:
            # ENTRY LOGIC
            current_price = current['close']

            # Calculate ATR for stop loss/take profit
            if 'atr_14' in data.columns:
                atr = current['atr_14']
            else:
                high_low = data['high'] - data['low']
                atr = high_low.rolling(14).mean().iloc[-1]

            # Regime-adaptive stop loss/take profit
            if regime_info['regime'] == 'VOLATILE':
                stop_multiplier = 3.0
                profit_multiplier = 2.0
            elif regime_info['regime'] == 'BULL':
                stop_multiplier = 1.5
                profit_multiplier = 3.0
            elif regime_info['regime'] == 'BEAR':
                stop_multiplier = 1.0
                profit_multiplier = 2.0
            else:  # SIDEWAYS
                stop_multiplier = 2.0
                profit_multiplier = 2.5

            if predicted_class == 1:  # Buy signal
                stop_loss = current_price - (atr * stop_multiplier)
                take_profit = current_price + (atr * profit_multiplier)

                signals.append({
                    'instrument': 'SPX500_USD',
                    'action': 'buy',
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'position_size_multiplier': adaptive_settings['position_multiplier'],
                    'reason': f"{regime_info['regime']}_{model_name}_BUY_{confidence:.2f}"
                })

                self.regime_stats[regime_info['regime']]['trades'] += 1

            elif predicted_class == 0:  # Sell signal
                stop_loss = current_price + (atr * stop_multiplier)
                take_profit = current_price - (atr * profit_multiplier)

                signals.append({
                    'instrument': 'SPX500_USD',
                    'action': 'sell',
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'position_size_multiplier': adaptive_settings['position_multiplier'],
                    'reason': f"{regime_info['regime']}_{model_name}_SELL_{confidence:.2f}"
                })

                self.regime_stats[regime_info['regime']]['trades'] += 1

        else:
            # EXIT LOGIC
            if regime_info['regime'] in ['VOLATILE', 'BEAR']:
                exit_threshold = 0.50
            else:
                exit_threshold = 0.55

            if confidence >= exit_threshold:
                signals.append({
                    'instrument': 'SPX500_USD',
                    'action': 'close',
                    'reason': f"{regime_info['regime']}_{model_name}_Exit_{confidence:.2f}"
                })

        return signals

    def get_regime_statistics(self) -> pd.DataFrame:
        """Get statistics on regime-based trading"""
        stats_data = []
        for regime, stats in self.regime_stats.items():
            trade_rate = stats['trades'] / stats['signals'] * 100 if stats['signals'] > 0 else 0
            stats_data.append({
                'Regime': regime,
                'Model Used': stats['model_used'],
                'Signals': stats['signals'],
                'Trades': stats['trades'],
                'Trade Rate': f"{trade_rate:.1f}%"
            })

        return pd.DataFrame(stats_data)


def main():
    """Example usage"""
    print("="*80)
    print("ENSEMBLE REGIME-ADAPTIVE STRATEGY")
    print("="*80)

    TOP_20_FEATURES = [
        'bullish_engulfing', 'stoch_d_3', 'week_of_year', 'atr_14', 'regime',
        'roc_20', 'obv', 'parkinson_vol_10', 'volatility_200d', 'momentum_5',
        'macd_signal', 'adx_14', 'month_sin', 'hl_ratio', 'rsi_14',
        'stoch_k_14', 'bb_position_20', 'momentum_oscillator', 'pvt', 'price_acceleration'
    ]

    # Example paths (would be actual model paths in practice)
    original_model = 'regime_adaptive_results/model_split_9_adaptive.pkl'
    balanced_model = 'balanced_model_results/model_split_9_balanced.pkl'

    strategy = EnsembleRegimeStrategy(
        original_model_path=original_model,
        balanced_model_path=balanced_model,
        feature_cols=TOP_20_FEATURES,
        base_confidence_threshold=0.50,
        enable_regime_adaptation=True
    )

    print("\n✅ Ensemble Strategy Created")
    print("\nModel Selection by Regime:")
    for regime, settings in strategy.regime_settings.items():
        model_name = 'Balanced' if settings['use_balanced'] else 'Original'
        print(f"\n  {regime}:")
        print(f"    Model: {model_name}")
        print(f"    Confidence adj: {settings['confidence_adjustment']:+.2f}")
        print(f"    Position size: {settings['position_multiplier']:.1%}")
        print(f"    Description: {settings['description']}")

    print("\n" + "="*80)
    print("Ready for walk-forward validation!")
    print("Run: python3 walkforward_ensemble.py")
    print("="*80)


if __name__ == "__main__":
    main()
