# Complete Trading System Guide
## From Backtesting to Live Trading with ML

This guide covers the complete workflow for building, testing, and deploying a professional trading system with machine learning.

---

## 📁 System Architecture

```
trading_system/
├── data/                          # Historical price data
├── models/                        # Trained ML models
├── results/                       # Backtest results and reports
├── backtesting_engine.py         # Core backtesting framework
├── feature_engineering.py        # Technical indicators & features
├── ml_training_pipeline.py       # ML model training
├── strategy_examples.py          # Pre-built strategies
├── complete_workflow.py          # End-to-end pipeline
├── oanda_integration.py          # Oanda API connector
├── trading_dashboard_main.py     # Monitoring dashboard
├── sample_data_generator.py      # Test data generation
└── config.py                     # Configuration (API keys, etc.)
```

---

## 🚀 Quick Start (5 Minutes)

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn xgboost streamlit plotly oandapyV20
```

### 2. Run Quick Backtest Example
```bash
python complete_workflow.py --mode quick
```

This will:
- Generate sample price data
- Run momentum strategy backtest
- Display performance metrics

### 3. View Results in Dashboard
```bash
# Generate sample trades
python sample_data_generator.py

# Launch dashboard
streamlit run trading_dashboard_main.py
```

---

## 📊 Complete Workflow (Production)

### Step 1: Data Collection

**Option A: Use Oanda API (Real Data)**
```python
from oanda_integration import OandaConnector
from config import OANDA_CONFIG

# Initialize connector
oanda = OandaConnector(
    account_id=OANDA_CONFIG['account_id'],
    access_token=OANDA_CONFIG['access_token'],
    environment='practice'
)

# Get historical data (implement fetch_historical method)
# df = oanda.fetch_historical_data('SPX500_USD', 'H1', days=365)
```

**Option B: Load from CSV**
```python
import pandas as pd

df = pd.read_csv('your_data.csv', index_col='timestamp', parse_dates=True)
# Ensure columns: open, high, low, close, volume
```

### Step 2: Feature Engineering

```python
from feature_engineering import FeatureEngineering

# Build complete feature set
df_features = FeatureEngineering.build_complete_feature_set(
    df, 
    include_volume=True
)

print(f"Created {len(df_features.columns)} features")
```

**Features Created:**
- **Price Features**: Returns, momentum, gaps
- **Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, Stochastic, ADX, CCI, Williams %R
- **Volatility**: Historical volatility, Parkinson volatility
- **Volume**: Volume ratios, OBV, volume momentum
- **Time**: Hour, day of week, trading sessions
- **Patterns**: Doji, hammer, engulfing patterns
- **Market Regime**: Trend strength, volatility regime

### Step 3: Train ML Models

```python
from ml_training_pipeline import MLTradingPipeline

# Initialize pipeline
pipeline = MLTradingPipeline()

# Prepare data
df_features = pipeline.load_and_prepare_data(df, include_volume=True)

# Compare multiple models
comparison = pipeline.compare_models(
    model_types=['logistic', 'random_forest', 'xgboost', 'gradient_boosting']
)
print(comparison)

# Train best model with hyperparameter tuning
results = pipeline.train_model(
    model_type='xgboost',
    hyperparameter_tuning=True,
    cross_validation=True
)

# Model is automatically saved in models/
```

**Model Selection Criteria:**
- F1 Score > 0.55
- Cross-validation performance stable
- Feature importance makes sense
- No overfitting (train vs test gap < 5%)

### Step 4: Backtest Strategies

```python
from backtesting_engine import BacktestEngine, BacktestConfig
from strategy_examples import MomentumStrategy, MeanReversionStrategy

# Configure backtest parameters
config = BacktestConfig(
    initial_capital=10000.0,
    commission_pct=0.001,  # 0.1% for S&P 500 = 0.01%
    slippage_pct=0.0002,  # 0.02% slippage slippage
    position_size_pct=0.02,     # 2% risk per trade
    max_positions=1,
    use_bid_ask_spread=True,
    spread_pips=1.0,
    max_daily_loss_pct=0.05,   # 5% daily loss limit
    max_drawdown_pct=0.20       # 20% max drawdown
)

# Initialize engine
engine = BacktestEngine(config)

# Test strategy
strategy = MomentumStrategy(fast_period=20, slow_period=50)
result = engine.run_backtest(strategy, df)

# View metrics
metrics = result['metrics']
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Win Rate: {metrics['win_rate']:.2%}")
print(f"Total Return: {metrics['total_return_pct']:.2f}%")
print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
```

**Key Metrics Calculated:**
- Sharpe Ratio, Sortino Ratio, Calmar Ratio
- Win Rate, Profit Factor, Expectancy
- Maximum Drawdown, Recovery Factor
- MAE/MFE (Max Adverse/Favorable Excursion)
- Consecutive wins/losses

### Step 5: Walk-Forward Analysis

```python
# Perform walk-forward analysis (most important validation!)
wf_results = engine.walk_forward_analysis(
    strategy=strategy,
    data=df,
    train_period_days=180,    # 6 months training
    test_period_days=60,      # 2 months testing
    step_days=30              # Move forward 1 month each time
)

# Analyze consistency across periods
for i, result in enumerate(wf_results, 1):
    print(f"Period {i}: Sharpe={result['metrics']['sharpe_ratio']:.2f}")
```

**Validation Criteria:**
- Minimum 3 walk-forward periods
- Sharpe Ratio > 1.0 in each period
- Win rate > 50% in at least 75% of periods
- Max drawdown < 15% in all periods

### Step 6: Deploy to Dashboard

```python
from trading_dashboard_main import DatabaseManager

db = DatabaseManager()

# Add backtest trades to dashboard
for trade in result['trades']:
    db.add_trade({
        'trade_id': f"BT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'instrument': trade.instrument,
        'direction': trade.direction,
        'entry_time': trade.entry_time.isoformat(),
        'exit_time': trade.exit_time.isoformat(),
        'entry_price': trade.entry_price,
        'exit_price': trade.exit_price,
        'size': trade.size,
        'pnl': trade.pnl,
        'pnl_percent': trade.pnl_percent,
        'commission': trade.commission,
        'slippage': trade.slippage,
        'strategy': 'Momentum_20_50',
        'status': 'closed'
    })

print("✅ Trades added to dashboard")
```

Launch dashboard:
```bash
streamlit run trading_dashboard_main.py
```

---

## 🎯 Strategy Examples

### 1. Momentum Strategy (MA Crossover)
```python
from strategy_examples import MomentumStrategy

strategy = MomentumStrategy(fast_period=20, slow_period=50, rsi_period=14)
# Enters when fast MA crosses slow MA, filtered by RSI
```

### 2. Mean Reversion Strategy
```python
from strategy_examples import MeanReversionStrategy

strategy = MeanReversionStrategy(bb_period=20, bb_std=2.0)
# Buys at lower Bollinger Band, sells at upper
```

### 3. Breakout Strategy
```python
from strategy_examples import BreakoutStrategy

strategy = BreakoutStrategy(lookback_period=20, atr_period=14)
# Enters on breakout above/below channel
```

### 4. ML-Powered Strategy
```python
from strategy_examples import MLStrategy

strategy = MLStrategy(
    model_path='models/xgboost_20240115_120000.pkl',
    feature_cols=pipeline.feature_cols,
    confidence_threshold=0.6
)
# Uses trained ML model for predictions
```

### 5. Ensemble Strategy
```python
from strategy_examples import EnsembleStrategy

strategies = [
    MomentumStrategy(20, 50),
    MeanReversionStrategy(20, 2.0),
    BreakoutStrategy(20)
]

ensemble = EnsembleStrategy(strategies, voting_method='majority')
# Combines signals from multiple strategies
```

### 6. Create Your Own Strategy
```python
from backtesting_engine import Strategy
import pandas as pd
from datetime import datetime

class MyCustomStrategy(Strategy):
    def __init__(self, param1, param2):
        super().__init__(name="MyStrategy")
        self.param1 = param1
        self.param2 = param2
    
    def generate_signals(self, data: pd.DataFrame, timestamp: datetime):
        signals = []
        
        # Your logic here
        if some_condition:
            signals.append({
                'instrument': 'SPX500_USD',
                'action': 'buy',  # or 'sell' or 'close'
                'stop_loss': price - atr * 2,
                'take_profit': price + atr * 3,
                'reason': 'my_signal'
            })
        
        return signals
```

---

## 📈 Paper Trading (Critical Step!)

**Never skip paper trading!** Trade on Oanda practice account for minimum 3 months.

```python
from oanda_integration import OandaConnector, DashboardDataSync
from trading_dashboard_main import DatabaseManager

# Initialize
oanda = OandaConnector(
    account_id='your-practice-account-id',
    access_token='your-token',
    environment='practice'
)

db = DatabaseManager()
sync = DashboardDataSync(oanda, db)

# Start continuous sync (runs in background)
sync.start_auto_sync(interval_seconds=60)
```

**Paper Trading Checklist:**
- [ ] Run for minimum 90 days
- [ ] Sharpe ratio > 1.0
- [ ] Max drawdown < 15%
- [ ] Win rate > 50%
- [ ] Consistent monthly returns
- [ ] No system failures or bugs
- [ ] Risk management working correctly

---

## 🔧 Position Sizing Methods

The system supports 3 position sizing methods:

### 1. Fixed Percentage Risk
```python
config = BacktestConfig(
    position_sizing_method='fixed_pct',
    position_size_pct=0.02  # Risk 2% per trade
)
```

### 2. Volatility-Based (ATR)
```python
config = BacktestConfig(
    position_sizing_method='volatility',
    position_size_pct=0.02
)
```

### 3. Kelly Criterion
```python
config = BacktestConfig(
    position_sizing_method='kelly',
    position_size_pct=0.02  # Max allocation
)
# Uses historical win rate and avg win/loss
```

---

## ⚠️ Risk Management

### Built-in Risk Controls

1. **Position Limits**
   - Max 2% risk per trade
   - Max 1-3 positions simultaneously
   - Position sizing based on stop loss

2. **Daily Limits**
   - 5% max daily loss (configurable)
   - Trading halts if exceeded

3. **Drawdown Limits**
   - 20% max drawdown (configurable)
   - Automatic trading stop

4. **Stop Loss/Take Profit**
   - Every position must have SL
   - TP based on risk:reward ratio (1:1.5 minimum)

### Manual Kill Switch Conditions

Stop trading immediately if:
- 5 consecutive losing trades
- Daily loss > 5%
- Drawdown > 20%
- Sharpe ratio drops below 0 (30-day rolling)
- Win rate drops below 40% (100 trade sample)

---

## 📊 Model Validation Checklist

Before deploying any model:

- [ ] **Cross-validation performance**
  - Use TimeSeriesSplit (no random shuffling!)
  - 5-fold minimum
  - Consistent metrics across folds

- [ ] **Walk-forward analysis**
  - Minimum 3 periods
  - Each period profitable
  - Sharpe > 1.0 in all periods

- [ ] **Out-of-sample testing**
  - Test on data model never saw
  - Performance within 20% of training

- [ ] **Feature importance**
  - Top features make intuitive sense
  - No data leakage
  - Features stable across time

- [ ] **Realistic assumptions**
  - Transaction costs included
  - Slippage modeled
  - Bid-ask spread considered
  - No look-ahead bias

---

## 🔄 Continuous Improvement Cycle

```
1. Collect Data (Oanda API)
       ↓
2. Engineer Features
       ↓
3. Train/Update ML Model
       ↓
4. Backtest New Strategy
       ↓
5. Walk-Forward Validate
       ↓
6. Paper Trade (90 days)
       ↓
7. Monitor Dashboard Daily
       ↓
8. Analyze Performance
       ↓
9. Adjust Parameters
       ↓
   (Back to Step 3)
```

**Retraining Schedule:**
- Weekly: Update with new data
- Monthly: Full model retraining
- Quarterly: Strategy review and optimization

---

## 🚨 Common Pitfalls to Avoid

1. **Overfitting**
   - ❌ Using too many features (>50)
   - ❌ Over-optimizing parameters
   - ✅ Use regularization, cross-validation

2. **Look-Ahead Bias**
   - ❌ Using future data in features
   - ❌ Including target in feature calculation
   - ✅ Strict time-series splits

3. **Survivorship Bias**
   - ❌ Only testing on surviving instruments
   - ✅ Include delisted/removed instruments

4. **Ignoring Costs**
   - ❌ Assuming zero transaction costs
   - ✅ Include commission, slippage, spread

5. **Insufficient Testing**
   - ❌ Only backtesting once
   - ✅ Multiple time periods, walk-forward

6. **Skipping Paper Trading**
   - ❌ Going live after backtest
   - ✅ 3+ months paper trading first

---

## 📱 Integration with Trading Bot

```python
# In your trading bot
from trading_dashboard_main import DatabaseManager
from ml_training_pipeline import MLModelTrainer

class TradingBot:
    def __init__(self, strategy, oanda_connector):
        self.strategy = strategy
        self.oanda = oanda_connector
        self.db = DatabaseManager()
        
        # Load ML model if using ML strategy
        if isinstance(strategy, MLStrategy):
            self.model = MLModelTrainer.load_model(strategy.model_path)
    
    def run(self):
        while True:
            # Get latest data
            data = self.oanda.get_recent_data()
            
            # Generate signals
            signals = self.strategy.generate_signals(data, datetime.now())
            
            # Execute trades
            for signal in signals:
                self.execute_signal(signal)
            
            # Update dashboard
            self.sync_to_dashboard()
            
            # Check risk limits
            if self.check_risk_limits():
                logger.critical("Risk limits exceeded! Stopping.")
                break
            
            time.sleep(60)  # Run every minute
    
    def execute_signal(self, signal):
        # Place order via Oanda API
        response = self.oanda.place_market_order(...)
        
        # Log to dashboard
        self.db.add_trade(...)
```

---

## 📚 Additional Resources

**Recommended Reading:**
- "Evidence-Based Technical Analysis" by David Aronson
- "Advances in Financial Machine Learning" by Marcos López de Prado
- "Quantitative Trading" by Ernest Chan

**Online Resources:**
- Oanda API Docs: https://developer.oanda.com/
- Quantopian Lectures: https://www.quantopian.com/lectures
- QuantConnect Community: https://www.quantconnect.com/

---

## 🆘 Troubleshooting

### Issue: Low Sharpe Ratio (<1.0)
- Increase position size (within risk limits)
- Tighten stop losses
- Add more filters to reduce bad trades
- Consider different instruments/timeframes

### Issue: High Drawdown (>20%)
- Reduce position size
- Widen stop losses
- Add correlation filters
- Implement portfolio heat monitoring

### Issue: Model Not Profitable
- Check for overfitting (cross-validation)
- Verify no look-ahead bias
- Ensure sufficient training data (>5000 samples)
- Try simpler models first (logistic regression)

### Issue: Inconsistent Results
- Market regime may have changed
- Retrain model on recent data
- Consider adaptive strategies
- Review feature importance shifts

---

## ✅ Production Deployment Checklist

Before going live with real money:

- [ ] 90+ days successful paper trading
- [ ] Sharpe ratio > 1.5 in paper trading
- [ ] Max drawdown < 10% in paper trading
- [ ] All risk limits functioning correctly
- [ ] Dashboard monitoring setup
- [ ] Alert system configured
- [ ] Kill switch conditions tested
- [ ] Data backup system in place
- [ ] Started with small capital ($500-$1000)
- [ ] Risk per trade ≤ 1% of capital
- [ ] Documented strategy rules
- [ ] Emergency contact plan

---

## 📞 Support

For issues or questions:
1. Check documentation above
2. Review code comments in source files
3. Test with sample data first
4. Use paper trading before live

**Remember: No system guarantees profits. Always trade responsibly!**

---

*System Version 1.0 | Last Updated: October 2025*
