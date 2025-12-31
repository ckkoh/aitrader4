# Machine Learning Framework Documentation

> Deep dive into the ML capabilities, feature engineering, and model training pipeline

---

## 📚 Table of Contents

1. [Overview](#overview)
2. [Feature Engineering](#feature-engineering)
3. [ML Models](#ml-models)
4. [Training Pipeline](#training-pipeline)
5. [Model Validation](#model-validation)
6. [Deployment](#deployment)
7. [Performance Benchmarks](#performance-benchmarks)
8. [Troubleshooting](#troubleshooting)

---

## Overview

### **ML Architecture**

```
Raw Data (OHLCV)
    ↓
Feature Engineering (50+ indicators)
    ↓
Data Preprocessing (cleaning, normalization)
    ↓
Model Training (5 algorithms)
    ↓
Hyperparameter Tuning (RandomizedSearchCV)
    ↓
Validation (Time-series CV + Walk-forward)
    ↓
Model Selection (best performing)
    ↓
Deployment (save model + metadata)
    ↓
Live Prediction (with confidence scores)
    ↓
Monitoring (drift detection, retraining triggers)
```

### **Key Components**

| Component | Location | Purpose |
|-----------|----------|---------|
| **Feature Engineering** | `core/feature_engineering.py` | Generate 50+ technical indicators |
| **ML Pipeline** | `core/ml_training_pipeline.py` | Train and validate models |
| **ML Strategy** | `strategies/strategy_examples.py` | Use ML for trading signals |
| **Model Monitoring** | `core/model_failure_recovery.py` | Detect drift and failures |

---

## Feature Engineering

### **Categories of Features (50+ total)**

#### **1. Price Features (15+)**

```python
# Returns
- returns               # Simple returns
- log_returns          # Logarithmic returns
- returns_lag_1        # Lagged returns (1-10 periods)

# Momentum
- momentum_5           # 5-period momentum
- momentum_10          # 10-period momentum
- momentum_20          # 20-period momentum

# Ratios
- hl_ratio             # (High - Low) / Close
- co_ratio             # (Close - Open) / Close
- gap                  # Open / Previous Close - 1
```

#### **2. Moving Averages (10+)**

```python
# Simple Moving Averages
- sma_10, sma_20, sma_50, sma_200

# Exponential Moving Averages
- ema_12, ema_26, ema_50

# Crossovers
- price_vs_sma_20      # Distance from SMA
- sma_20_slope         # SMA slope (trend)
```

#### **3. Volatility Indicators (8)**

```python
# Standard Volatility
- volatility_10        # 10-period historical vol
- volatility_20        # 20-period historical vol
- volatility_50        # 50-period historical vol

# Advanced Volatility
- parkinson_vol_10     # Parkinson volatility
- parkinson_vol_20     # Uses high-low range
- vol_of_vol           # Volatility of volatility

# ATR
- atr_14               # Average True Range
- atr_percent_14       # ATR as % of price
```

#### **4. Momentum Oscillators (7)**

```python
# RSI
- rsi_14               # Relative Strength Index

# MACD
- macd                 # MACD line
- macd_signal          # Signal line
- macd_hist            # Histogram

# Stochastic
- stoch_k_14           # %K line
- stoch_d_3            # %D line

# Others
- cci_20               # Commodity Channel Index
- williams_r_14        # Williams %R
```

#### **5. Trend Indicators (5)**

```python
# ADX
- adx_14               # Average Directional Index
- plus_di_14           # +DI
- minus_di_14          # -DI

# Bollinger Bands
- bb_width_20          # Band width
- bb_position_20       # Price position in bands
```

#### **6. Volume Indicators (4)**

```python
- volume_ratio         # Volume / 20-day avg
- volume_momentum      # Volume change
- obv                  # On-Balance Volume
- pvt                  # Price-Volume Trend
```

#### **7. Time Features (12)**

```python
# Raw Time
- hour                 # Hour of day (0-23)
- day_of_week          # Day of week (0-6)
- month                # Month (1-12)

# Cyclical Encoding (prevents discontinuity)
- hour_sin, hour_cos
- day_sin, day_cos
- month_sin, month_cos

# Trading Sessions
- asian_session        # 0-8 UTC
- european_session     # 8-16 UTC
- us_session           # 16-24 UTC
```

#### **8. Pattern Recognition (5)**

```python
- doji                 # Doji candlestick
- hammer               # Hammer pattern
- bullish_engulfing    # Bullish engulfing
- bearish_engulfing    # Bearish engulfing
```

#### **9. Market Regime (5)**

```python
- trend_strength       # |EMA12 - EMA26| / Close
- high_vol_regime      # Volatility > 70th percentile
- low_vol_regime       # Volatility < 30th percentile
- vol_percentile       # Volatility percentile rank
```

### **Target Variables**

```python
# For Classification
- target_binary        # 1 if price up, 0 if down
- target_class         # 1 (up), 0 (sideways), -1 (down)

# For Regression
- target_regression    # Actual future return
- future_return        # N-period forward return
```

### **Feature Generation Example**

```python
from core.feature_engineering import FeatureEngineering

# Load OHLCV data
df = pd.read_csv('SPX500_USD_H1.csv', index_col='Date', parse_dates=True)

# Generate all features
df_features = FeatureEngineering.build_complete_feature_set(
    df, 
    include_volume=True
)

# Result: Original 5 columns → 60+ columns
print(f"Features created: {len(df_features.columns)}")
# Features created: 62

# Top features by importance (after training)
important_features = [
    'rsi_14',
    'macd_hist',
    'volatility_20',
    'bb_position_20',
    'momentum_10',
    'atr_percent_14',
    'stoch_k_14',
    'ema_12',
    'hour_sin',
    'trend_strength'
]
```

---

## ML Models

### **Model Comparison**

| Model | Pros | Cons | Training Time | Best Use Case |
|-------|------|------|---------------|---------------|
| **XGBoost** | • Best overall performance<br>• Handles non-linear patterns<br>• Feature importance | • Can overfit<br>• Slower than LR | Medium | General purpose |
| **Random Forest** | • Robust<br>• Less overfitting<br>• Feature importance | • Slower prediction<br>• Large model size | Slow | Conservative |
| **Gradient Boosting** | • Good performance<br>• Handles complexity | • Can overfit<br>• Slower training | Medium | Complex patterns |
| **Logistic Regression** | • Fast<br>• Interpretable<br>• Baseline | • Linear only<br>• Lower accuracy | Fast | Baseline comparison |
| **Voting Ensemble** | • Most robust<br>• Best Sharpe ratio | • Slowest<br>• Complex to debug | Very Slow | Production systems |

### **Hyperparameter Grids**

#### **XGBoost**

```python
param_grid = {
    'max_depth': [4, 6, 8],           # Tree depth
    'learning_rate': [0.01, 0.05, 0.1], # Learning rate
    'n_estimators': [100, 200, 300],   # Number of trees
    'subsample': [0.7, 0.8, 0.9],      # Row sampling
    'colsample_bytree': [0.7, 0.8, 0.9] # Column sampling
}

# Optimal often found at:
# max_depth=6, learning_rate=0.05, n_estimators=200
# subsample=0.8, colsample_bytree=0.8
```

#### **Random Forest**

```python
param_grid = {
    'n_estimators': [50, 100, 200],        # Number of trees
    'max_depth': [5, 10, 15, None],        # Tree depth
    'min_samples_split': [10, 20, 50],     # Min samples to split
    'min_samples_leaf': [5, 10, 20]        # Min samples per leaf
}

# Optimal often found at:
# n_estimators=100, max_depth=10
# min_samples_split=20, min_samples_leaf=10
```

### **Model Training Example**

```python
from core.ml_training_pipeline import MLTradingPipeline

# Initialize pipeline
pipeline = MLTradingPipeline()

# Load data with features
df_features = pipeline.load_and_prepare_data(data, include_volume=True)

# Train with hyperparameter tuning
results = pipeline.train_model(
    model_type='xgboost',
    target_col='target_binary',
    test_size=0.2,
    hyperparameter_tuning=True,  # Use RandomizedSearchCV
    cross_validation=True         # 5-fold time-series CV
)

# Access results
print(f"Test Accuracy: {results['test_metrics']['accuracy']:.2%}")
print(f"F1 Score: {results['test_metrics']['f1_score']:.3f}")
print(f"ROC AUC: {results['test_metrics'].get('roc_auc', 'N/A')}")

# Top features
print("\nTop 10 Features:")
for idx, row in results['feature_importance'].head(10).iterrows():
    print(f"{row['feature']}: {row['importance']:.4f}")

# Model saved automatically
print(f"\nModel saved: {results['model_filename']}")
```

### **Expected Performance by Model**

```python
# Typical metrics on EUR/USD daily data

XGBoost:
├─ Accuracy: 55-60%
├─ F1 Score: 0.57-0.62
├─ Win Rate: 54-58%
└─ Sharpe: 1.2-1.8

Random Forest:
├─ Accuracy: 53-58%
├─ F1 Score: 0.55-0.60
├─ Win Rate: 52-56%
└─ Sharpe: 1.0-1.5

Logistic Regression:
├─ Accuracy: 51-55%
├─ F1 Score: 0.52-0.56
├─ Win Rate: 50-54%
└─ Sharpe: 0.8-1.2

Ensemble:
├─ Accuracy: 56-62%
├─ F1 Score: 0.58-0.64
├─ Win Rate: 55-60%
└─ Sharpe: 1.3-2.0
```

---

## Training Pipeline

### **Complete Training Flow**

#### **Step 1: Data Preparation**

```python
# Time-series split (NO SHUFFLING!)
train_size = int(len(df) * 0.8)
train_data = df.iloc[:train_size]
test_data = df.iloc[train_size:]

# Further split training into train/validation
val_size = int(len(train_data) * 0.2)
train_split = train_data.iloc[:-val_size]
val_split = train_data.iloc[-val_size:]
```

#### **Step 2: Feature Selection**

```python
# Remove low-variance features
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.01)
selected_features = selector.fit_transform(X_train)

# Or use feature importance after initial training
initial_model = xgb.XGBClassifier()
initial_model.fit(X_train, y_train)

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': initial_model.feature_importances_
}).sort_values('importance', ascending=False)

# Keep top 30-50 features
top_features = feature_importance.head(40)['feature'].tolist()
```

#### **Step 3: Normalization**

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler with model
import pickle
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
```

#### **Step 4: Model Training**

```python
# XGBoost with early stopping
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    random_state=42
)

model.fit(
    X_train_scaled, 
    y_train,
    eval_set=[(X_val_scaled, y_val)],
    early_stopping_rounds=50,
    verbose=False
)
```

#### **Step 5: Validation**

```python
# Cross-validation
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    model.fit(X_tr, y_tr)
    score = model.score(X_val, y_val)
    cv_scores.append(score)
    
print(f"CV Accuracy: {np.mean(cv_scores):.2%} (+/- {np.std(cv_scores):.2%})")
```

---

## Model Validation

### **Validation Strategy**

**Three-Layer Validation:**

1. **Time-Series Cross-Validation** (In-sample)
   - 5-fold TimeSeriesSplit
   - Ensures model consistency

2. **Out-of-Sample Testing** (Hold-out set)
   - Last 20% of data
   - Never seen during training

3. **Walk-Forward Analysis** (Realistic)
   - Train on 6 months → Test on 2 months
   - Step forward 1 month → Repeat
   - Most realistic validation

### **Validation Metrics**

```python
from sklearn.metrics import classification_report

# Predictions
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# Detailed metrics
print(classification_report(y_test, y_pred))

# Custom trading metrics
from core.backtesting_engine import BacktestEngine

# Convert predictions to trades and backtest
# This gives you Sharpe, win rate, drawdown, etc.
```

### **Model Acceptance Criteria**

**Minimum Requirements:**

```python
✅ Accuracy > 52% (on test set)
✅ F1 Score > 0.54
✅ Sharpe Ratio > 1.0 (in backtest)
✅ Win Rate > 50%
✅ Max Drawdown < 15%
✅ Profit Factor > 1.3
✅ CV std < 0.05 (consistency)
✅ Out-of-sample within 80% of in-sample
```

**Failure Indicators:**

```python
❌ Train accuracy >> Test accuracy (overfitting)
❌ Negative Sharpe on test set
❌ High CV variance (inconsistent)
❌ Random-looking feature importance
❌ Fails walk-forward validation
```

---

## Deployment

### **Model Deployment Workflow**

#### **1. Save Trained Model**

```python
# Automatic in pipeline
model_filename = f"xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"

model_data = {
    'model': trained_model,
    'scaler': scaler,
    'feature_cols': feature_cols,
    'training_metrics': metrics,
    'training_date': datetime.now().isoformat(),
    'data_range': {
        'start': train_data.index[0],
        'end': train_data.index[-1]
    }
}

with open(f'models/{model_filename}', 'wb') as f:
    pickle.dump(model_data, f)
```

#### **2. Load Model for Trading**

```python
from strategies.strategy_examples import MLStrategy

# Load trained model
strategy = MLStrategy(
    model_path='models/xgboost_20250629_143022.pkl',
    feature_cols=feature_list,
    confidence_threshold=0.60  # Only trade high-confidence predictions
)

# Use in backtesting
from core.backtesting_engine import BacktestEngine

engine = BacktestEngine()
results = engine.run_backtest(strategy, test_data)
```

#### **3. Paper Trading**

```python
from tools.monitoring_integration import MonitoredTradingBot

bot = MonitoredTradingBot(
    strategy=strategy,
    oanda_config=OANDA_CONFIG,
    initial_capital=10000
)

# Set baseline from backtesting
bot.set_baseline_from_backtest(backtest_results)

# Run with monitoring (paper account)
bot.run(health_check_interval_minutes=15)
```

#### **4. Model Monitoring**

```python
# Continuous monitoring
from core.model_failure_recovery import ModelFailureDetector

detector = ModelFailureDetector()
detector.set_baseline(
    win_rate=0.55,
    sharpe=1.2,
    profit_factor=1.8
)

# Check health every 15 minutes
health = detector.check_health(recent_trades_df)

if health.status != HealthStatus.HEALTHY:
    # Trigger recovery protocol
    take_action(health.recommended_action)
```

### **Retraining Schedule**

```python
Trigger Retraining When:
├─ Monthly: Scheduled maintenance
├─ Feature drift > 0.3
├─ Win rate drops > 10% below baseline
├─ Sharpe < 0.5 for 30 days
└─ Market regime change detected

Retraining Process:
1. Use last 6 months of data
2. Full pipeline (features + training)
3. Validate with walk-forward
4. A/B test: new vs old model (2 weeks)
5. Deploy if new model > old model
```

---

## Performance Benchmarks

### **Training Performance**

**Hardware: 16GB RAM, i7 CPU**

| Task | Data Size | Time | Notes |
|------|-----------|------|-------|
| Feature Generation | 10,000 bars | 0.8s | 50+ features |
| XGBoost Training | 10,000 samples | 3.2s | No tuning |
| XGBoost + Tuning | 10,000 samples | 4.5min | 20 iterations |
| Random Forest | 10,000 samples | 8.5s | No tuning |
| Cross-Validation | 10,000 samples | 15s | 5 folds |
| Walk-Forward (1yr) | 8,760 bars | 2.3min | 6 periods |

### **Prediction Performance**

```python
# Real-time prediction speed
Model           | Predictions/sec
----------------|----------------
XGBoost         | 15,000
Random Forest   | 8,000
Logistic Reg    | 50,000
Ensemble        | 5,000
```

### **Memory Usage**

```python
Component              | RAM Usage
-----------------------|-----------
Feature DataFrame      | ~50 MB per 10k bars
XGBoost Model          | ~5 MB
Random Forest Model    | ~25 MB
Ensemble Model         | ~80 MB
Full Pipeline          | ~150 MB
```

---

## Troubleshooting

### **Common Issues**

#### **Issue 1: Low Accuracy (< 52%)**

**Possible Causes:**
- Not enough features
- Features not normalized
- Overfitting to training data
- Wrong target variable

**Solutions:**
```python
# Add more features
df_features = FeatureEngineering.build_complete_feature_set(df)

# Check for data leakage
# Make sure no future data in features

# Use simpler model
model = LogisticRegression()  # Start simple

# Check target distribution
print(y.value_counts(normalize=True))
# Should be roughly balanced (40-60% each class)
```

#### **Issue 2: Overfitting**

**Symptoms:**
- Train accuracy: 80%, Test accuracy: 52%
- Great backtest, terrible live trading

**Solutions:**
```python
# Increase regularization
model = xgb.XGBClassifier(
    max_depth=4,        # Reduce from 6
    min_child_weight=5, # Increase
    gamma=0.1,          # Add regularization
    subsample=0.7,      # More aggressive sampling
    colsample_bytree=0.7
)

# Use fewer features
important_features = feature_importance.head(20)  # Top 20 only

# More cross-validation folds
tscv = TimeSeriesSplit(n_splits=10)

# Simpler model
model = RandomForestClassifier(max_depth=5)
```

#### **Issue 3: Model Drift**

**Symptoms:**
- Model worked well initially
- Performance degrading over time
- Feature distributions changing

**Solutions:**
```python
# Detect drift
from core.model_failure_recovery import ModelFailureDetector

detector = ModelFailureDetector()
health = detector.check_health(trades_df)

if health.model_drift_score > 0.3:
    # Retrain on recent data
    recent_data = df.tail(5000)  # Last 6 months
    pipeline.train_model(recent_data)
```

#### **Issue 4: Slow Predictions**

**Symptoms:**
- Missing trading opportunities
- High latency

**Solutions:**
```python
# Use faster model
model = LogisticRegression()  # 10x faster than XGBoost

# Reduce features
features = important_features[:20]  # Use top 20 only

# Pre-compute features
# Cache recent calculations
# Update incrementally instead of recalculating
```

### **Debugging Checklist**

```python
# 1. Check data
print(df.isnull().sum())  # No NaNs?
print(df.describe())      # Reasonable values?

# 2. Check features
print(f"Features: {len(feature_cols)}")
print(X_train.shape, y_train.shape)  # Shapes match?

# 3. Check target
print(y_train.value_counts())  # Balanced?
print(y_train.isna().sum())    # No NaNs?

# 4. Check model
print(model.get_params())      # Parameters correct?
print(model.feature_importances_[:10])  # Sensible?

# 5. Check predictions
y_pred = model.predict(X_test)
print(np.unique(y_pred, return_counts=True))  # Predicting both classes?
```

---

## Advanced Topics

### **Custom Feature Engineering**

```python
# Add your own features
class CustomFeatures:
    @staticmethod
    def add_custom_indicator(df):
        # Example: Custom momentum indicator
        df['custom_momentum'] = (
            df['close'].rolling(10).mean() / 
            df['close'].rolling(30).mean() - 1
        )
        return df

# Use it
df = CustomFeatures.add_custom_indicator(df)
```

### **Ensemble Strategies**

```python
# Combine multiple models
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('rf', rf_model),
        ('lr', lr_model)
    ],
    voting='soft',  # Use probabilities
    weights=[2, 1, 1]  # XGBoost gets 2x weight
)

ensemble.fit(X_train, y_train)
```

### **Feature Importance Analysis**

```python
import matplotlib.pyplot as plt

# Get importances
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot
plt.figure(figsize=(10, 6))
plt.bar(range(20), importances[indices[:20]])
plt.xticks(range(20), [feature_cols[i] for i in indices[:20]], rotation=90)
plt.title('Top 20 Feature Importances')
plt.tight_layout()
plt.show()
```

---

## Summary

**Key Takeaways:**

✅ **50+ Features**: Comprehensive technical analysis  
✅ **5 ML Models**: From simple to ensemble  
✅ **Proper Validation**: Time-series CV + walk-forward  
✅ **Production Ready**: Monitoring + retraining  
✅ **Realistic Testing**: Transaction costs + slippage  
✅ **Continuous Improvement**: Regular retraining schedule  

**Best Practices:**

1. Always use time-series validation (no shuffling!)
2. Test on out-of-sample data
3. Monitor for overfitting
4. Retrain monthly or on drift detection
5. Start simple, add complexity gradually
6. Paper trade for 90+ days before live

---

**For more details, see:**
- `core/ml_training_pipeline.py` - Full implementation
- `core/feature_engineering.py` - All features
- `tools/run_examples.py` - Working examples
- `docs/Complete_System_Integration_Guide.md` - Full documentation

---

*Last Updated: December 29, 2025*
