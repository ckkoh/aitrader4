# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚠️ CRITICAL: S&P 500 ONLY TRADING SYSTEM

**THIS SYSTEM TRADES S&P 500 EXCLUSIVELY. DO NOT USE FOR OTHER INSTRUMENTS.**

## System Overview

This is a **production-ready algorithmic S&P 500 trading system** with ML-powered strategies, comprehensive backtesting, real-time monitoring, and automated failure detection. The system combines traditional technical analysis with machine learning to trade the S&P 500 index (SPX500_USD) through the Oanda API.

**Core Capabilities:**
- Backtest strategies with realistic transaction costs (commission, slippage)
- Train ML models on 50+ technical indicators optimized for S&P 500
- Live trade S&P 500 via Oanda API (practice and live accounts)
- Monitor performance via 5-page Streamlit dashboard
- Walk-forward validation to prevent overfitting
- Automated health monitoring and failure recovery

## S&P 500 TRADING CONFIGURATION

### ⚡ Quick Reference

**Instrument:** SPX500_USD (S&P 500 Index via Oanda)
**Primary Timeframe:** **D (Daily)** ← RECOMMENDED for swing trading
**Secondary Timeframe:** **M15 (15-minute)** ← For intraday only
**❌ NOT M5:** System uses Daily or M15, NOT M5 (5-minute)
**Trading Hours:** 9:30 AM - 4:00 PM ET (US market hours only)
**Data Source:** Oanda API or Yahoo Finance (^GSPC)

**Total Indicators:** 50+ technical indicators (see list below)
**Volume:** MANDATORY - Always use `include_volume=True`
**Position Size:** Max 2% of capital notional value
**Commission:** 0.1% per trade (vs 0.01% for forex)

### Optimal S&P 500 Parameters

**BacktestConfig for S&P 500:**
```python
config = BacktestConfig(
    initial_capital=10000.0,
    commission_pct=0.001,  # 0.1% per trade (typical for CFD/index trading)
    slippage_pct=0.0002,   # 0.02% slippage
    position_size_pct=0.02,  # Risk 2% per trade
    max_position_value_pct=0.02,  # Max 2% of capital per position (CRITICAL for S&P 500)
    max_positions=1,  # Single position only for S&P 500
    max_daily_loss_pct=0.03,  # 3% max daily loss (tighter for indices)
    max_drawdown_pct=0.15,  # 15% max drawdown (tighter than forex)
    position_sizing_method='volatility',  # Use ATR-based sizing for S&P 500
)
```

**Key Differences from Forex:**
- **Volume is CRITICAL**: Always use `include_volume=True` for S&P 500
- **Lower position sizes**: S&P 500 at ~$6000 requires smaller % allocations
- **Tighter stops**: Indices move faster than forex; use 1-1.5x ATR stops
- **Trading hours matter**: Only trade during US session (9:30 AM - 4:00 PM ET)
- **Commission structure**: Fixed per trade or percentage, not pips

### S&P 500 Optimal Indicators

**Primary Indicators (Proven for S&P 500):**
1. **SMA/EMA**: 20, 50, 200-day moving averages (critical trend filters)
2. **RSI(14)**: Overbought >70, oversold <30
3. **MACD(12,26,9)**: Trend confirmation
4. **ATR(14)**: Volatility-based position sizing and stops
5. **Bollinger Bands(20,2)**: Volatility breakouts
6. **Volume**: Above/below 20-day average (critical for S&P 500!)

**Secondary Indicators:**
- ADX(14): Trend strength >25 = strong trend
- Stochastic(14,3): Momentum confirmation
- Volume ratio: Current volume / 20-day avg
- VIX correlation: High VIX = avoid longs

### 📊 Complete Indicator List (50+ Total)

**11 Core Technical Indicators:**
1. SMA - Simple Moving Averages (10, 20, 50, 200)
2. EMA - Exponential Moving Averages (12, 26, 50)
3. RSI - Relative Strength Index (14)
4. MACD - Moving Average Convergence Divergence (12/26/9)
5. Bollinger Bands - (20, 2 std)
6. ATR - Average True Range (14)
7. ADX - Average Directional Index (14)
8. Stochastic Oscillator - (14/3)
9. CCI - Commodity Channel Index (20)
10. Williams %R - (14)
11. OBV - On-Balance Volume

**Feature Engineering Categories (40+ derived features):**
- **Price Features:** OHLC ratios, price vs SMA/EMA, returns, log returns, price acceleration
- **Momentum Features:** ROC, momentum indicators, price velocity
- **Volatility Features:** ATR-based, rolling std, Bollinger width, volatility ratios
- **Volume Features:** Volume ratios, OBV, volume trends, volume MA (CRITICAL for S&P 500!)
- **Pattern Features:** Candlestick patterns, support/resistance levels
- **Market Regime:** Trending/ranging detection, bull/bear market classification
- **Time Features:** Hour, day of week, month, quarter (for M15 intraday)

**S&P 500 Feature Set:**
```python
# ALWAYS include volume for S&P 500
df_features = FeatureEngineering.build_complete_feature_set(df, include_volume=True)

# S&P 500 specific features to add:
- Trading hour filters (only 9:30 AM - 4 PM ET)
- Month-end effects (last 3 days of month)
- FOMC meeting days (high volatility)
- Earnings season indicators
```

### S&P 500 Strategy Recommendations

**Best Strategies for S&P 500 (in order):**

1. **Trend Following (Daily timeframe)**
   - Use 50/200 SMA crossover as primary signal
   - Confirm with MACD and volume
   - ATR-based stops (1.5x ATR)
   - Works best in trending markets (2023-2024)

2. **Breakout Strategy (M15 for intraday)**
   - 20-period high/low breakouts
   - Confirm with volume surge (>1.5x average)
   - Trade first 2 hours after open (highest volume)
   - See `spx500_day_trading.py`

3. **ML-Based Swing Trading (Daily)**
   - Train on 5-year daily S&P 500 data
   - Use XGBoost with top 20 features
   - Confidence threshold: 0.65+ (higher than forex)
   - Include volume features (mandatory)

**Avoid for S&P 500:**
- Mean reversion (S&P 500 trends strongly)
- High-frequency strategies (commission erosion)
- Overnight holds during earnings season
- Trading during low-volume periods (<10 AM or >3 PM ET)

### S&P 500 Position Sizing

**CRITICAL: S&P 500 price ~$6000 requires careful sizing**

```python
# For $10,000 account trading S&P 500:
max_position_value_pct = 0.02  # 2% = $200 max notional
max_units = $200 / $6000 = 0.033 units (DO NOT exceed this!)

# With 2% risk per trade and 1.5% ATR:
# Risk amount = $10,000 * 0.02 = $200
# Stop distance = $6000 * 0.015 = $90
# Position size = $200 / $90 = 2.22 units
# BUT capped at 0.033 units by max_position_value_pct!

# Actual position: 0.033 units = $200 notional (safe)
```

**Never exceed 2% of capital in notional S&P 500 value!**

### S&P 500 Data Files

**Use these data files for S&P 500:**
- `sp500_historical_data.csv`: Full history (2020-2024, ~1200 days)
- `sp500_ytd_2025.csv`: YTD 2025 data (~248 days)
- Download fresh data: `python download_sp500_data.py`

**Oanda S&P 500 fetch:**
```python
oanda = OandaConnector(...)
df = oanda.fetch_historical_data_range('SPX500_USD', 'D', days=365)
# For intraday: granularity='M15'
```

### S&P 500 Walk-Forward Settings

```python
# For Daily S&P 500 data:
results = engine.walk_forward_analysis(
    strategy=strategy,
    data=df,
    train_period_days=365,  # 1 year training
    test_period_days=90,    # 3 months testing
    step_days=30           # Move forward 1 month
)

# Minimum 3 walk-forward periods = need 2+ years data
```

## Build & Run Commands

### Installation & Setup
```bash
# Initial setup (creates directories, generates requirements.txt)
python requirements_setup.py

# Install dependencies
pip install pandas numpy scikit-learn xgboost streamlit plotly oandapyV20 python-dateutil

# Configuration
# Create config.py with Oanda credentials (use practice account first!)
# See .env for template structure
```

### Running Examples
```bash
# Run specific example (1-6)
python run_examples.py --example 1  # Simple momentum backtest
python run_examples.py --example 2  # Compare multiple strategies
python run_examples.py --example 3  # Feature engineering demo
python run_examples.py --example 4  # Train ML model
python run_examples.py --example 5  # Walk-forward validation
python run_examples.py --example 6  # Dashboard deployment

# Run all examples
python run_examples.py --all
```

### Workflows
```bash
# Quick backtest (generates sample data + runs momentum strategy)
python complete_workflow.py --mode quick

# Full pipeline (data -> features -> ML -> backtest -> validate)
python complete_workflow.py --mode full

# Generate test data for dashboard
python sample_data_generator.py

# Launch monitoring dashboard
streamlit run trading_dashboard_main.py
```

### Testing
```bash
# No formal test suite - validation done through:
# 1. Run examples (--all flag)
# 2. Walk-forward analysis
# 3. Paper trading (90+ days recommended)

# Verify installation
python -c "import pandas, numpy, sklearn, xgboost, streamlit, plotly"
```

## High-Level Architecture

### Module Organization (4 Tiers)

**Tier 1 - Foundation (no internal dependencies):**
- `backtesting_engine.py` (850 lines): Core backtesting framework with walk-forward analysis, position sizing (3 methods: fixed %, ATR-based, Kelly criterion), realistic cost modeling
- `feature_engineering.py` (650 lines): 50+ technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, Stochastic, ADX, OBV, CCI, Williams %R), pattern recognition, market regime detection

**Tier 2 - Core Components (depend on Tier 1):**
- `ml_training_pipeline.py` (550 lines): ML model training with hyperparameter tuning, time-series cross-validation, feature importance analysis. Depends on `feature_engineering.py`
- `oanda_integration.py` (350 lines): Oanda v20 API connector for account data, positions, trade history, order placement
- `trading_dashboard_main.py` (950 lines): Streamlit dashboard (5 pages: Overview, Trades, Risk Monitor, Analysis, Settings), SQLite database management

**Tier 3 - Strategies (depend on Tier 1 & 2):**
- `strategy_examples.py` (500 lines): 6 pre-built strategies (MomentumStrategy, MeanReversionStrategy, BreakoutStrategy, MLStrategy, EnsembleStrategy, AdaptiveMomentumStrategy). Depends on `backtesting_engine.py` and `ml_training_pipeline.py`

**Tier 4 - Integration (depend on all tiers):**
- `complete_workflow.py` (600 lines): End-to-end pipeline orchestration
- `run_examples.py` (500 lines): 6 runnable examples demonstrating system capabilities
- `sample_data_generator.py` (300 lines): Realistic test data generation for dashboard
- `requirements_setup.py` (250 lines): Automated installation

**Additional Modules:**
- `monitoring_integration.py`: Live trading bot with automatic health monitoring
- `model_failure_recovery.py`: Model degradation detection and recovery strategies
- `model_accuracy_maintenance.py`: Model performance tracking and retraining automation

### Key Data Flow

```
Oanda API / CSV → feature_engineering → ml_training_pipeline → strategy_examples
                                                  ↓
                                          backtesting_engine (walk-forward)
                                                  ↓
                                          trading_dashboard_main (monitoring)
                                                  ↓
                                          oanda_integration (live trading)
```

### Critical Classes & Interfaces

**backtesting_engine.py:**
- `Strategy` (ABC): Base class for all strategies. Subclass and implement `generate_signals(data, timestamp)` returning list of signal dicts with keys: instrument, action (buy/sell/close), stop_loss, take_profit, reason
- `BacktestEngine`: Main backtesting engine. Key methods: `run_backtest(strategy, data)`, `walk_forward_analysis(strategy, data, train_period_days, test_period_days, step_days)`
- `BacktestConfig`: Configuration dataclass for backtest parameters (initial_capital, commission_pct, slippage_pct, position_size_pct, max_positions, max_daily_loss_pct, max_drawdown_pct)
- `PositionSizer`: Static methods for position sizing (fixed_percentage, volatility_adjusted, kelly_criterion)
- `Trade` & `Position`: Dataclasses representing trades and positions

**feature_engineering.py:**
- `TechnicalIndicators`: Static methods for calculating technical indicators. All methods return modified DataFrame with new columns
- `FeatureEngineering`: Static methods for building feature sets. Key method: `build_complete_feature_set(df, include_volume=True)` returns DataFrame with 50+ features
- `DataPreprocessor`: Static methods for data cleaning and normalization

**ml_training_pipeline.py:**
- `MLModelTrainer`: Handles model training, evaluation, persistence. Methods: `prepare_data()`, `train()`, `predict()`, `evaluate()`, `cross_validate()`, `save_model()`, `load_model()`
- `MLTradingPipeline`: High-level pipeline. Key methods: `load_and_prepare_data()`, `train_model(model_type, hyperparameter_tuning, cross_validation)`, `compare_models()`

**trading_dashboard_main.py:**
- `DatabaseManager`: SQLite database operations. Methods: `add_trade()`, `get_trades()`, `add_alert()`, `update_position()`, `get_open_positions()`
- `PerformanceCalculator`: Static methods for calculating 20+ metrics (Sharpe, Sortino, Calmar ratios, win rate, profit factor, max drawdown, etc.)
- `RiskMonitor`: Check risk limits and generate alerts

**oanda_integration.py:**
- `OandaConnector`: Oanda API wrapper. Methods: `get_account_summary()`, `get_open_positions()`, `get_trade_history()`, `place_market_order()`, `close_position()`
- `DashboardDataSync`: Sync Oanda data to dashboard. Methods: `sync_positions()`, `sync_trade_history()`, `sync_all()`, `start_auto_sync()`

### Strategy Development Pattern

To create a custom strategy:

1. Subclass `Strategy` from `backtesting_engine.py`
2. Implement `generate_signals(data: pd.DataFrame, timestamp: datetime) -> List[Dict]`
3. Signal dict format: `{'instrument': str, 'action': 'buy'|'sell'|'close', 'stop_loss': float, 'take_profit': float, 'reason': str}`
4. Test with `BacktestEngine.run_backtest()`
5. Validate with `BacktestEngine.walk_forward_analysis()` (CRITICAL - prevents overfitting)
6. Paper trade for 90+ days before live deployment

Example:
```python
class MyStrategy(Strategy):
    def __init__(self, param1):
        super().__init__(name="MyStrategy")
        self.param1 = param1

    def generate_signals(self, data: pd.DataFrame, timestamp: datetime) -> List[Dict]:
        signals = []
        if your_condition:
            signals.append({
                'instrument': 'SPX500_USD',  # S&P 500 ONLY
                'action': 'buy',
                'stop_loss': price - atr * 1.5,  # Tighter stops for S&P 500
                'take_profit': price + atr * 2.5,
                'reason': 'my_signal'
            })
        return signals
```

### ML Model Training Pattern

1. Load data with OHLCV columns (open, high, low, close, volume)
2. Use `FeatureEngineering.build_complete_feature_set()` to generate features
3. Initialize `MLTradingPipeline()`
4. Call `pipeline.train_model(model_type='xgboost', hyperparameter_tuning=True, cross_validation=True)`
5. Model automatically saved to `models/` directory with timestamp
6. Load model in `MLStrategy(model_path='models/xgboost_20250630_120000.pkl', feature_cols=cols, confidence_threshold=0.6)`

### Walk-Forward Validation (Critical)

Walk-forward analysis is the **most important validation step** to prevent overfitting:

```python
engine = BacktestEngine(config)
results = engine.walk_forward_analysis(
    strategy=strategy,
    data=df,
    train_period_days=180,  # Train on 6 months
    test_period_days=60,    # Test on 2 months
    step_days=30            # Move forward 1 month each iteration
)
```

Validation criteria:
- Minimum 3 walk-forward periods
- Sharpe ratio > 1.0 in each period
- Win rate > 50% in at least 75% of periods
- Max drawdown < 15% in all periods

### Risk Management

**Built-in safeguards:**
- Position sizing limits (2% default risk per trade)
- Daily loss limits (5% max)
- Max drawdown limits (20% max)
- Stop loss required on every position
- Transaction costs included (commission, slippage, bid-ask spread)

**Pre-live deployment checklist:**
- 90+ days successful paper trading
- Sharpe ratio > 1.5 in paper trading
- Max drawdown < 10%
- Walk-forward validation passed
- All risk limits tested
- Start with small capital ($500-$1000)

## Important Notes

### Configuration
- **NEVER commit config.py or .env** - contains Oanda API credentials
- Use practice account initially (set `environment: 'practice'` in config)
- API token format: Long alphanumeric string from Oanda API settings
- Account ID format: ###-###-#######-###

### Data Requirements
- OHLCV format required (columns: open, high, low, close, volume)
- DateTime index required for backtesting
- Minimum 5000 samples recommended for ML training
- Walk-forward requires sufficient data (e.g., 2 years for 6-month train + 2-month test periods)

### Time-Series Constraints
- **No random shuffling** - always use `TimeSeriesSplit` for cross-validation
- **No look-ahead bias** - features must only use past data
- **Strict temporal ordering** - training data must precede test data

### Common Pitfalls
- Overfitting: Use regularization, cross-validation, walk-forward analysis
- Look-ahead bias: Verify features don't use future data
- Ignoring costs: Always include commission, slippage, spread (realistic: 1 pip = 0.0001 = 0.01%)
- Insufficient testing: Minimum 3 walk-forward periods required
- Skipping paper trading: 90+ days paper trading mandatory before live

### Performance Expectations
- Typical Sharpe ratios: 1.0-2.0 (>1.5 considered good)
- Typical win rates: 50-60%
- Typical max drawdown: 10-15%
- Backtest speed: ~1000 trades/second
- ML training: <5 minutes for 100k samples
- Feature generation: 50+ indicators in <1 second

### File Locations
- Trained models: `models/` (auto-created, .pkl files with timestamps)
- Backtest results: `results/` (auto-created, .json files)
- Database: `trading_data.db` (auto-created SQLite)
- Logs: `logs/` (if configured)
- Historical data: `data/` (user-provided CSVs)

### Development Workflow
1. Develop strategy locally
2. Backtest with `BacktestEngine`
3. Validate with walk-forward analysis
4. Add to dashboard with `sample_data_generator.py` for visualization
5. Paper trade via Oanda practice account (90+ days)
6. Monitor via `trading_dashboard_main.py`
7. Only then consider live trading with minimal capital

### Extending the System
- New strategies: Subclass `Strategy` in `strategy_examples.py`
- New indicators: Add to `TechnicalIndicators` or `FeatureEngineering` classes
- New ML models: Add to `MLModelTrainer._get_model()` method
- Dashboard pages: Modify `trading_dashboard_main.py` (uses Streamlit multipage pattern)
- Risk monitors: Extend `RiskMonitor.check_risk_limits()`

## System Status

**Production-Ready:** All 10 core files complete and functional (5,500+ lines)
**Missing:** Only user-specific `config.py` with Oanda credentials
**Dependencies:** Python 3.8+, pandas, numpy, scikit-learn, xgboost, streamlit, plotly, oandapyV20
