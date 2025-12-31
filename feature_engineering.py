"""
Feature Engineering for Trading ML Models
Comprehensive technical indicators and feature transformations
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Calculate technical indicators for price data"""

    @staticmethod
    def add_sma(df: pd.DataFrame, periods: List[int] = [20, 50, 200],
                price_col: str = 'close') -> pd.DataFrame:
        """Add Simple Moving Averages"""
        for period in periods:
            df[f'sma_{period}'] = df[price_col].rolling(window=period).mean()
        return df

    @staticmethod
    def add_ema(df: pd.DataFrame, periods: List[int] = [12, 26, 50],
                price_col: str = 'close') -> pd.DataFrame:
        """Add Exponential Moving Averages"""
        for period in periods:
            df[f'ema_{period}'] = df[price_col].ewm(span=period, adjust=False).mean()
        return df

    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = 14,
                price_col: str = 'close') -> pd.DataFrame:
        """Add Relative Strength Index"""
        delta = df[price_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26,
                 signal: int = 9, price_col: str = 'close') -> pd.DataFrame:
        """Add MACD indicator"""
        ema_fast = df[price_col].ewm(span=fast, adjust=False).mean()
        ema_slow = df[price_col].ewm(span=slow, adjust=False).mean()

        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        return df

    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame, period: int = 20,
                            std_dev: int = 2, price_col: str = 'close') -> pd.DataFrame:
        """Add Bollinger Bands"""
        df[f'bb_middle_{period}'] = df[price_col].rolling(window=period).mean()
        std = df[price_col].rolling(window=period).std()

        df[f'bb_upper_{period}'] = df[f'bb_middle_{period}'] + (std * std_dev)
        df[f'bb_lower_{period}'] = df[f'bb_middle_{period}'] - (std * std_dev)
        df[f'bb_width_{period}'] = df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']
        df[f'bb_position_{period}'] = (df[price_col] - df[f'bb_lower_{period}']) / df[f'bb_width_{period}']

        return df

    @staticmethod
    def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df[f'atr_{period}'] = tr.rolling(window=period).mean()

        # Normalized ATR
        df[f'atr_percent_{period}'] = (df[f'atr_{period}'] / close) * 100

        return df

    @staticmethod
    def add_stochastic(df: pd.DataFrame, k_period: int = 14,
                       d_period: int = 3) -> pd.DataFrame:
        """Add Stochastic Oscillator"""
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()

        df[f'stoch_k_{k_period}'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df[f'stoch_d_{d_period}'] = df[f'stoch_k_{k_period}'].rolling(window=d_period).mean()

        return df

    @staticmethod
    def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average Directional Index"""
        high = df['high']
        low = df['low']
        close = df['close']

        # Calculate +DM and -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Smoothed values
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df[f'adx_{period}'] = dx.rolling(window=period).mean()
        df[f'plus_di_{period}'] = plus_di
        df[f'minus_di_{period}'] = minus_di

        return df

    @staticmethod
    def add_obv(df: pd.DataFrame) -> pd.DataFrame:
        """Add On-Balance Volume"""
        obv = [0]

        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i - 1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i - 1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])

        df['obv'] = obv
        df['obv_ema'] = df['obv'].ewm(span=20, adjust=False).mean()

        return df

    @staticmethod
    def add_cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Add Commodity Channel Index"""
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())

        df[f'cci_{period}'] = (tp - sma_tp) / (0.015 * mad)

        return df

    @staticmethod
    def add_williams_r(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Williams %R"""
        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()

        df[f'williams_r_{period}'] = -100 * (high_max - df['close']) / (high_max - low_min)

        return df


class FeatureEngineering:
    """Advanced feature engineering for ML models"""

    @staticmethod
    def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Lagged returns
        for lag in [1, 2, 3, 5, 10]:
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)

        # Price momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1

        # High-Low range
        df['hl_ratio'] = (df['high'] - df['low']) / df['close']
        df['co_ratio'] = (df['close'] - df['open']) / df['close']

        # Gap
        df['gap'] = df['open'] / df['close'].shift(1) - 1

        return df

    @staticmethod
    def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features"""
        # Historical volatility
        for period in [10, 20, 50]:
            df[f'volatility_{period}'] = df['returns'].rolling(window=period).std() * np.sqrt(252)

        # Parkinson volatility (uses high-low range)
        for period in [10, 20]:
            df[f'parkinson_vol_{period}'] = np.sqrt(
                (1 / (4 * period * np.log(2))) *
                ((np.log(df['high'] / df['low']) ** 2).rolling(window=period).sum())
            )

        # Volatility of volatility
        df['vol_of_vol'] = df['volatility_20'].rolling(window=20).std()

        return df

    @staticmethod
    def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        if 'volume' not in df.columns:
            return df

        # Volume ratios
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']

        # Volume momentum
        df['volume_momentum'] = df['volume'] / df['volume'].shift(5) - 1

        # Price-Volume trend
        df['pvt'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1) * df['volume']).cumsum()

        return df

    @staticmethod
    def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            return df

        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['week_of_year'] = df.index.isocalendar().week
        df['month'] = df.index.month

        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Trading session (simplified)
        # Asian: 0-8, European: 8-16, US: 16-24 (UTC)
        df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['european_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['us_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)

        return df

    @staticmethod
    def add_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick pattern features"""
        # Doji
        body = abs(df['close'] - df['open'])
        df['doji'] = (body / (df['high'] - df['low']) < 0.1).astype(int)

        # Hammer/Hanging Man
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        df['hammer'] = ((lower_shadow > 2 * body) & (upper_shadow < body)).astype(int)

        # Engulfing patterns
        df['bullish_engulfing'] = (
            (df['close'] > df['open']) &
            (df['open'].shift(1) > df['close'].shift(1)) &
            (df['close'] > df['open'].shift(1)) &
            (df['open'] < df['close'].shift(1))
        ).astype(int)

        df['bearish_engulfing'] = (
            (df['open'] > df['close']) &
            (df['close'].shift(1) > df['open'].shift(1)) &
            (df['open'] > df['close'].shift(1)) &
            (df['close'] < df['open'].shift(1))
        ).astype(int)

        return df

    @staticmethod
    def add_market_regime_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive market regime features using 200-day MA"""
        try:
            from regime_detection import RegimeDetector

            # Initialize regime detector
            detector = RegimeDetector(ma_period=200, trend_lookback=50, volatility_period=20)

            # Add all regime features
            df = detector.add_regime_features(df)

            logger.info("Added regime detection features (200-day MA based)")

        except Exception as e:
            logger.warning(f"Could not add regime features: {e}, using basic features")

            # Fallback: basic features if regime_detection not available
            # Trend strength
            df['trend_strength'] = abs(df['ema_12'] - df['ema_26']) / df['close'] if 'ema_12' in df.columns else 0

            # Price distance from moving averages
            for period in [20, 50, 200]:
                if f'sma_{period}' in df.columns:
                    df[f'price_vs_sma_{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}']

            # Moving average slopes
            for period in [20, 50]:
                if f'sma_{period}' in df.columns:
                    df[f'sma_{period}_slope'] = df[f'sma_{period}'].diff(5) / df[f'sma_{period}']

            # Volatility regime (high/low based on percentile)
            if 'volatility_20' in df.columns:
                df['vol_percentile'] = df['volatility_20'].rolling(window=100).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1]
                )
                df['high_vol_regime'] = (df['vol_percentile'] > 0.7).astype(int)
                df['low_vol_regime'] = (df['vol_percentile'] < 0.3).astype(int)

        return df

    @staticmethod
    def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based features"""
        # Rate of change
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period)

        # Momentum oscillator
        df['momentum_oscillator'] = df['close'] - df['close'].shift(10)

        # Acceleration (second derivative)
        df['price_acceleration'] = df['returns'].diff()

        return df

    @staticmethod
    def create_target_variable(df: pd.DataFrame,
                               forward_periods: int = 5,
                               threshold_pct: float = 0.001) -> pd.DataFrame:
        """
        Create target variable for ML training

        Args:
            df: DataFrame with price data
            forward_periods: Number of periods to look forward
            threshold_pct: Minimum return threshold for classification

        Returns:
            DataFrame with target variables
        """
        # Future returns
        df['future_return'] = df['close'].shift(-forward_periods) / df['close'] - 1

        # Classification target (1: up, -1: down, 0: sideways)
        df['target_class'] = 0
        df.loc[df['future_return'] > threshold_pct, 'target_class'] = 1
        df.loc[df['future_return'] < -threshold_pct, 'target_class'] = -1

        # Binary target (1: up, 0: down)
        df['target_binary'] = (df['future_return'] > 0).astype(int)

        # Regression target (actual future return)
        df['target_regression'] = df['future_return']

        return df

    @staticmethod
    def build_complete_feature_set(df: pd.DataFrame,
                                   include_volume: bool = False) -> pd.DataFrame:
        """
        Build complete feature set with all indicators

        Args:
            df: DataFrame with OHLCV data
            include_volume: Whether to include volume features

        Returns:
            DataFrame with all features
        """
        logger.info("Building complete feature set...")

        # Make a copy to avoid modifying original
        df = df.copy()

        # Price features
        df = FeatureEngineering.add_price_features(df)

        # Technical indicators
        df = TechnicalIndicators.add_sma(df, [10, 20, 50, 200])
        df = TechnicalIndicators.add_ema(df, [12, 26, 50])
        df = TechnicalIndicators.add_rsi(df, 14)
        df = TechnicalIndicators.add_macd(df)
        df = TechnicalIndicators.add_bollinger_bands(df, 20)
        df = TechnicalIndicators.add_atr(df, 14)
        df = TechnicalIndicators.add_stochastic(df, 14, 3)
        df = TechnicalIndicators.add_adx(df, 14)
        df = TechnicalIndicators.add_cci(df, 20)
        df = TechnicalIndicators.add_williams_r(df, 14)

        # Volume features
        if include_volume and 'volume' in df.columns:
            df = TechnicalIndicators.add_obv(df)
            df = FeatureEngineering.add_volume_features(df)

        # Volatility features
        df = FeatureEngineering.add_volatility_features(df)

        # Time features
        df = FeatureEngineering.add_time_features(df)

        # Pattern features
        df = FeatureEngineering.add_pattern_features(df)

        # Momentum features
        df = FeatureEngineering.add_momentum_features(df)

        # Market regime features
        df = FeatureEngineering.add_market_regime_features(df)

        # Create target variable (for training)
        df = FeatureEngineering.create_target_variable(df, forward_periods=5)

        logger.info(f"Feature set complete. Total features: {len(df.columns)}")

        return df

    @staticmethod
    def select_features(df: pd.DataFrame,
                        feature_importance: Optional[pd.Series] = None,
                        top_n: int = 50) -> List[str]:
        """
        Select most important features

        Args:
            df: DataFrame with features
            feature_importance: Series with feature importance scores
            top_n: Number of features to select

        Returns:
            List of selected feature names
        """
        # Exclude target and price columns
        exclude_cols = ['open', 'high', 'low', 'close', 'volume',
                        'future_return', 'target_class', 'target_binary',
                        'target_regression']

        feature_cols = [col for col in df.columns if col not in exclude_cols]

        if feature_importance is not None:
            # Sort by importance
            important_features = feature_importance.sort_values(ascending=False)
            selected_features = important_features.head(top_n).index.tolist()
        else:
            # Use variance-based selection
            variances = df[feature_cols].var()
            selected_features = variances.nlargest(top_n).index.tolist()

        return selected_features


class DataPreprocessor:
    """Preprocess data for ML models"""

    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean data by removing NaNs and infinities"""
        # Remove infinities
        df = df.replace([np.inf, -np.inf], np.nan)

        # Forward fill first, then backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')

        # Drop any remaining NaNs
        df = df.dropna()

        return df

    @staticmethod
    def normalize_features(df: pd.DataFrame,
                           feature_cols: List[str],
                           method: str = 'standardize') -> Tuple[pd.DataFrame, Dict]:
        """
        Normalize features

        Args:
            df: DataFrame with features
            feature_cols: List of feature column names
            method: 'standardize' or 'minmax'

        Returns:
            Tuple of (normalized DataFrame, normalization parameters)
        """
        from sklearn.preprocessing import StandardScaler, MinMaxScaler

        df = df.copy()

        if method == 'standardize':
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()

        df[feature_cols] = scaler.fit_transform(df[feature_cols])

        # Store normalization parameters
        norm_params = {
            'scaler': scaler,
            'method': method,
            'feature_cols': feature_cols
        }

        return df, norm_params

    @staticmethod
    def apply_normalization(df: pd.DataFrame, norm_params: Dict) -> pd.DataFrame:
        """Apply saved normalization parameters"""
        df = df.copy()
        scaler = norm_params['scaler']
        feature_cols = norm_params['feature_cols']

        df[feature_cols] = scaler.transform(df[feature_cols])

        return df


# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
    np.random.seed(42)

    df = pd.DataFrame({
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 101,
        'low': np.random.randn(1000).cumsum() + 99,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000)
    }, index=dates)

    # Build feature set
    df_features = FeatureEngineering.build_complete_feature_set(df, include_volume=True)

    print(f"Original columns: {len(df.columns)}")
    print(f"Features created: {len(df_features.columns)}")
    print(f"\nSample features: {list(df_features.columns[:20])}")

    # Clean data
    df_clean = DataPreprocessor.clean_data(df_features)
    print(f"\nRows after cleaning: {len(df_clean)}")
