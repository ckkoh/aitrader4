# Trading System - Quick Reference Card

## 📦 **Complete System: 10 Files, 5,500+ Lines**

### **Core Files (All Complete ✅)**

```
1. backtesting_engine.py      (850 lines) - Backtesting framework
2. feature_engineering.py     (650 lines) - 50+ indicators  
3. ml_training_pipeline.py    (550 lines) - ML training
4. strategy_examples.py       (500 lines) - 6 strategies
5. trading_dashboard_main.py  (950 lines) - Streamlit dashboard
6. oanda_integration.py       (350 lines) - API connector
7. sample_data_generator.py   (300 lines) - Test data
8. complete_workflow.py       (600 lines) - Full pipeline
9. run_examples.py            (500 lines) - 6 examples
10. setup.py                  (250 lines) - Installation
```

---

## 🚀 **Quick Start (3 Commands)**

```bash
# 1. Install
python setup.py

# 2. Configure (copy & edit)
cp config_template.py config.py

# 3. Test
python run_examples.py --example 1
```

---

## 📋 **What's Required vs Optional**

### **REQUIRED (Must Have) ⚠️**
- All 10 core .py files above ✅
- config.py with your Oanda credentials ❌ (you create)
- Python 3.8+ with packages ✅

### **AUTO-CREATED (First Run)**
- trading_data.db (SQLite database)
- models/*.pkl (trained models)
- results/*.json (backtest results)
- requirements.txt

### **OPTIONAL**
- Historical data CSVs
- Custom strategies
- Additional configurations

---

## 🔗 **Dependencies**

### **Tier 1: Foundation (Independent)**
```
backtesting_engine.py
feature_engineering.py
```

### **Tier 2: Core (Needs Tier 1)**
```
ml_training_pipeline.py → feature_engineering
oanda_integration.py
trading_dashboard_main.py
```

### **Tier 3: Strategies (Needs Tier 1 & 2)**
```
strategy_examples.py → backtesting_engine + ml_training_pipeline
```

### **Tier 4: Integration (Needs All)**
```
complete_workflow.py → Everything
sample_data_generator.py → trading_dashboard_main
run_examples.py → Everything
```

---

## ⚡ **Key Commands**

```bash
# Run quick backtest
python complete_workflow.py --mode quick

# Run full pipeline  
python complete_workflow.py --mode full

# Run specific example (1-6)
python run_examples.py --example 4

# Generate test data
python sample_data_generator.py

# Launch dashboard
streamlit run trading_dashboard_main.py

# Run all examples
python run_examples.py --all
```

---

## 🎯 **What Each File Does**

| File | Purpose | Key Functions |
|------|---------|---------------|
| backtesting_engine | Run backtests | run_backtest(), walk_forward_analysis() |
| feature_engineering | Create features | build_complete_feature_set() |
| ml_training_pipeline | Train models | train_model(), compare_models() |
| strategy_examples | Trading logic | 6 Strategy classes |
| trading_dashboard_main | Monitor trades | Streamlit app with 5 pages |
| oanda_integration | API calls | get_trades(), place_order() |
| sample_data_generator | Test data | TradeGenerator class |
| complete_workflow | Full pipeline | TradingSystemPipeline.run_complete_pipeline() |
| run_examples | Learn system | 6 example functions |
| setup | Install | main() - automated setup |

---

## 📊 **System Workflow**

```
Data → Features → ML → Strategy → Backtest → Validate → Dashboard → Paper Trade → Live
```

1. **Data**: Load from Oanda or CSV
2. **Features**: 50+ technical indicators
3. **ML**: Train XGBoost/RandomForest models
4. **Strategy**: Apply rules or ML predictions
5. **Backtest**: Test with realistic costs
6. **Validate**: Walk-forward analysis
7. **Dashboard**: Monitor performance
8. **Paper Trade**: 90 days minimum
9. **Live**: Deploy with real money

---

## ✅ **Verification Checklist**

```bash
# Test 1: Imports work
python -c "import pandas, numpy, sklearn, xgboost, streamlit"

# Test 2: Run quick example
python run_examples.py --example 1

# Test 3: Dashboard loads
streamlit run trading_dashboard_main.py
```

**All pass? System is ready! ✅**

---

## 🔧 **Common Tasks**

### **Task: Run Simple Backtest**
```bash
python run_examples.py --example 1
```

### **Task: Train ML Model**
```bash
python run_examples.py --example 4
```

### **Task: Compare Strategies**
```bash
python run_examples.py --example 2
```

### **Task: Run Full Pipeline**
```bash
python complete_workflow.py --mode full
```

### **Task: View Results**
```bash
streamlit run trading_dashboard_main.py
```

---

## 📈 **Core Classes**

```
BacktestEngine       - Main backtesting
Strategy             - Base strategy class
MLModelTrainer       - ML training
FeatureEngineering   - Feature creation
DatabaseManager      - Dashboard data
OandaConnector       - API integration
TradeGenerator       - Test data
```

---

## 🎓 **Learning Order**

1. **Example 1**: Simple backtest → Understand basics
2. **Example 2**: Compare strategies → See differences
3. **Example 3**: Features → Learn indicators
4. **Example 4**: ML training → Model building
5. **Example 5**: Walk-forward → Validation
6. **Example 6**: Dashboard → Integration

---

## ⚠️ **Critical Info**

### **Before Live Trading:**
- [ ] 90+ days paper trading
- [ ] Sharpe ratio > 1.5
- [ ] Max drawdown < 15%
- [ ] Walk-forward validated
- [ ] Risk management tested

### **Risk Limits:**
- Max 2% risk per trade
- Max 5% daily loss
- Max 20% drawdown
- Start with $500-$1000

---

## 🚨 **Only Missing: Your Config**

Create `config.py`:
```python
OANDA_CONFIG = {
    'account_id': 'your-account-id',
    'access_token': 'your-token',
    'environment': 'practice'  # Start here!
}
```

---

## 💡 **Pro Tips**

1. **Always start with Example 1**
2. **Use paper trading first**
3. **Review dashboard daily**
4. **Set risk limits before starting**
5. **Keep detailed logs**
6. **Test walk-forward validation**
7. **Never skip paper trading period**
8. **Start with small capital**

---

## 📞 **Quick Help**

**Issue**: Imports fail
**Fix**: `pip install -r requirements.txt`

**Issue**: Dashboard won't start
**Fix**: `pip install --upgrade streamlit`

**Issue**: No data
**Fix**: `python sample_data_generator.py`

**Issue**: Oanda connection fails
**Fix**: Check config.py credentials

---

## 📊 **System Status**

**Core System**: ✅ 100% Complete  
**Documentation**: ✅ 100% Complete  
**Examples**: ✅ 100% Complete  
**Testing**: ✅ Verified Working  

**User Action Needed**: Create config.py ⚠️

---

**System is PRODUCTION-READY!**

Total: 10 files, 5,500+ lines, fully functional trading system with backtesting, ML, and monitoring.
