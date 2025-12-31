# Trading System - Dependency Tree & File Relationships

## 🌲 Complete Dependency Tree

```
┌─────────────────────────────────────────────────────────────┐
│                    TIER 0: SETUP                            │
│  setup.py (standalone - no dependencies)                    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              TIER 1: FOUNDATION (Independent)                │
│                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────┐ │
│  │ backtesting_     │  │ feature_         │  │ trading_ │ │
│  │ engine.py        │  │ engineering.py   │  │ dashboard│ │
│  │                  │  │                  │  │ _main.py │ │
│  │ • Strategy class │  │ • 50+ indicators │  │          │ │
│  │ • BacktestEngine │  │ • FeatureEng     │  │ • Streamlit│
│  │ • Metrics        │  │ • DataPreproc    │  │ • Database│
│  └──────────────────┘  └──────────────────┘  └──────────┘ │
└─────────────────────────────────────────────────────────────┘
         │                       │                    │
         │                       │                    │
         ▼                       ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│              TIER 2: CORE COMPONENTS                         │
│                                                              │
│     ┌─────────────────────┐         ┌──────────────────┐   │
│     │ ml_training_        │         │ oanda_           │   │
│     │ pipeline.py         │         │ integration.py   │   │
│     │                     │         │                  │   │
│     │ Needs:              │         │ Needs:           │   │
│     │ • feature_eng ✓     │         │ • dashboard ⚠    │   │
│     │                     │         │   (optional)     │   │
│     └─────────────────────┘         └──────────────────┘   │
│              │                               │              │
│              │              ┌────────────────┘              │
│              │              │                               │
│     ┌─────────────────────────────────┐                    │
│     │ sample_data_generator.py        │                    │
│     │ Needs: dashboard ✓              │                    │
│     └─────────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
         │
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│              TIER 3: STRATEGIES                              │
│                                                              │
│     ┌─────────────────────────────────────────┐            │
│     │ strategy_examples.py                    │            │
│     │                                         │            │
│     │ Needs:                                  │            │
│     │ • backtesting_engine ✓                 │            │
│     │ • ml_training_pipeline ✓ (for ML only)│            │
│     │                                         │            │
│     │ Provides:                               │            │
│     │ • MomentumStrategy                      │            │
│     │ • MeanReversionStrategy                 │            │
│     │ • BreakoutStrategy                      │            │
│     │ • MLStrategy                            │            │
│     │ • EnsembleStrategy                      │            │
│     └─────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────┘
         │
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│         TIER 4: INTEGRATION & AUTOMATION                     │
│                                                              │
│  ┌──────────────────────┐     ┌──────────────────────┐    │
│  │ complete_workflow.py │     │ run_examples.py      │    │
│  │                      │     │                      │    │
│  │ Needs ALL above ✓    │     │ Needs ALL above ✓    │    │
│  │                      │     │                      │    │
│  │ • Full pipeline      │     │ • 6 examples         │    │
│  │ • End-to-end auto    │     │ • Learning tool      │    │
│  └──────────────────────┘     └──────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Dependency Matrix Table

| File | Depends On | Used By | Can Run Alone? |
|------|------------|---------|----------------|
| **setup.py** | None | - | ✅ YES |
| **backtesting_engine.py** | None | strategy_examples, complete_workflow, run_examples | ✅ YES |
| **feature_engineering.py** | None | ml_training_pipeline, complete_workflow, run_examples | ✅ YES |
| **trading_dashboard_main.py** | None | oanda_integration, sample_data_generator, complete_workflow | ✅ YES |
| **ml_training_pipeline.py** | feature_engineering | strategy_examples, complete_workflow, run_examples | ❌ NO |
| **oanda_integration.py** | trading_dashboard (optional) | complete_workflow | ⚠️ PARTIAL |
| **sample_data_generator.py** | trading_dashboard | run_examples | ⚠️ PARTIAL |
| **strategy_examples.py** | backtesting_engine, ml_training_pipeline | complete_workflow, run_examples | ❌ NO |
| **complete_workflow.py** | ALL above | - | ❌ NO |
| **run_examples.py** | ALL above | - | ❌ NO |

---

## 🎯 Minimum File Requirements by Use Case

### Use Case 1: Simple Rule-Based Backtesting
```
Files Needed: 1-2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ backtesting_engine.py
✅ [your_custom_strategy.py]
```

### Use Case 2: ML Model Training
```
Files Needed: 2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ feature_engineering.py
✅ ml_training_pipeline.py
```

### Use Case 3: Backtest with Pre-Built Strategies
```
Files Needed: 4
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ backtesting_engine.py
✅ feature_engineering.py
✅ ml_training_pipeline.py
✅ strategy_examples.py
```

### Use Case 4: Full System with Dashboard
```
Files Needed: 5
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ backtesting_engine.py
✅ feature_engineering.py
✅ ml_training_pipeline.py
✅ strategy_examples.py
✅ trading_dashboard_main.py
```

### Use Case 5: Paper/Live Trading
```
Files Needed: 6
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ backtesting_engine.py
✅ feature_engineering.py
✅ ml_training_pipeline.py
✅ strategy_examples.py
✅ trading_dashboard_main.py
✅ oanda_integration.py
```

### Use Case 6: Complete Development Environment
```
Files Needed: 10 (ALL)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ All above 6 files
✅ sample_data_generator.py
✅ complete_workflow.py
✅ run_examples.py
✅ setup.py
```

---

## 🔗 Import Chain Analysis

### Chain 1: Simple Backtesting
```python
# File: my_strategy.py
from backtesting_engine import Strategy, BacktestEngine, BacktestConfig

# NO OTHER IMPORTS NEEDED
```

### Chain 2: ML-Powered Strategy
```python
# File: my_ml_strategy.py
from backtesting_engine import Strategy           # Tier 1
from feature_engineering import FeatureEngineering # Tier 1
from ml_training_pipeline import MLModelTrainer    # Tier 2 (needs Tier 1)

# 3 files in dependency chain
```

### Chain 3: Using Pre-Built Strategies
```python
# File: my_backtest.py
from backtesting_engine import BacktestEngine      # Tier 1
from strategy_examples import MomentumStrategy     # Tier 3 (needs Tier 1 + 2)

# This automatically pulls in:
# → backtesting_engine (direct)
# → feature_engineering (via strategy_examples → ml_training_pipeline)
# → ml_training_pipeline (via strategy_examples)
```

### Chain 4: Complete Pipeline
```python
# File: complete_workflow.py
from backtesting_engine import BacktestEngine
from feature_engineering import FeatureEngineering
from ml_training_pipeline import MLTradingPipeline
from strategy_examples import MomentumStrategy
from trading_dashboard_main import DatabaseManager
from oanda_integration import OandaConnector

# Pulls in ALL 6 core files
```

---

## 🧩 File Relationships Diagram

```
            ┌──────────────────────────────────┐
            │      START HERE                  │
            │   (Pick your use case)           │
            └──────────────┬───────────────────┘
                           │
                ┌──────────┴──────────┐
                │                     │
        ┌───────▼─────────┐   ┌──────▼──────────┐
        │  Need Backtest  │   │  Need ML Model  │
        │      Only?      │   │   Training?     │
        └───────┬─────────┘   └──────┬──────────┘
                │                     │
                ▼                     ▼
        ┌────────────────┐    ┌──────────────────┐
        │ backtesting_   │    │ feature_eng +    │
        │ engine.py      │    │ ml_training_     │
        │                │    │ pipeline.py      │
        └────────┬───────┘    └──────┬───────────┘
                 │                   │
                 └────────┬──────────┘
                          │
                  ┌───────▼────────┐
                  │  Want Pre-Built│
                  │  Strategies?   │
                  └───────┬────────┘
                          │ Yes
                  ┌───────▼────────┐
                  │ strategy_      │
                  │ examples.py    │
                  └───────┬────────┘
                          │
                  ┌───────▼────────┐
                  │  Want Dashboard│
                  │  Monitoring?   │
                  └───────┬────────┘
                          │ Yes
                  ┌───────▼────────┐
                  │ trading_       │
                  │ dashboard_     │
                  │ main.py        │
                  └───────┬────────┘
                          │
                  ┌───────▼────────┐
                  │  Want Live     │
                  │  Trading?      │
                  └───────┬────────┘
                          │ Yes
                  ┌───────▼────────┐
                  │ oanda_         │
                  │ integration.py │
                  └────────────────┘
```

---

## ⚡ Quick Reference: What Each File Provides

```
backtesting_engine.py
├─ Strategy (base class)
├─ BacktestEngine (run backtests)
├─ BacktestConfig (configuration)
├─ PositionSizer (position sizing methods)
└─ BacktestMetrics (calculate all metrics)

feature_engineering.py
├─ TechnicalIndicators (11 indicator methods)
├─ FeatureEngineering (10 feature methods)
└─ DataPreprocessor (3 preprocessing methods)

ml_training_pipeline.py
├─ MLModelTrainer (train/evaluate models)
└─ MLTradingPipeline (full ML workflow)

strategy_examples.py
├─ MomentumStrategy
├─ MeanReversionStrategy
├─ BreakoutStrategy
├─ MLStrategy
├─ EnsembleStrategy
└─ AdaptiveMomentumStrategy

trading_dashboard_main.py
├─ DatabaseManager (SQLite operations)
├─ PerformanceCalculator (metrics)
├─ RiskMonitor (risk checks)
└─ Streamlit Dashboard App (5 pages)

oanda_integration.py
├─ OandaConnector (API methods)
└─ DashboardDataSync (sync trades)

sample_data_generator.py
├─ TradeGenerator (generate test data)
└─ populate_dashboard_with_sample_data()

complete_workflow.py
└─ TradingSystemPipeline (6-step workflow)

run_examples.py
└─ 6 Example Functions (learning)

setup.py
└─ Automated Installation
```

---

## ✅ ABSOLUTE MINIMUM to Start

```
╔════════════════════════════════════════╗
║  MINIMUM WORKING SYSTEM: 1 FILE        ║
╠════════════════════════════════════════╣
║  backtesting_engine.py                 ║
║  + Your 10-line custom strategy        ║
║                                        ║
║  Result: Functional backtesting ✅     ║
╚════════════════════════════════════════╝
```

## 🎯 RECOMMENDED for Production

```
╔════════════════════════════════════════╗
║  PRODUCTION SYSTEM: 6 CORE FILES       ║
╠════════════════════════════════════════╣
║  1. backtesting_engine.py              ║
║  2. feature_engineering.py             ║
║  3. ml_training_pipeline.py            ║
║  4. strategy_examples.py               ║
║  5. trading_dashboard_main.py          ║
║  6. oanda_integration.py               ║
║                                        ║
║  Result: Full trading system ✅        ║
╚════════════════════════════════════════╝
```

---

## 🚨 Critical Dependencies Summary

### Zero Dependencies (Standalone)
```
✅ backtesting_engine.py      → Run alone
✅ feature_engineering.py     → Run alone
✅ trading_dashboard_main.py  → Run alone
✅ setup.py                   → Run alone
```

### One Dependency
```
⚠️ ml_training_pipeline.py    → Needs feature_engineering.py
⚠️ sample_data_generator.py   → Needs trading_dashboard_main.py
```

### Two Dependencies
```
⚠️ strategy_examples.py       → Needs backtesting_engine.py
                                   + ml_training_pipeline.py
```

### All Dependencies
```
⚠️ complete_workflow.py       → Needs ALL 9 other files
⚠️ run_examples.py            → Needs ALL 9 other files
```

---

## 📋 Final Checklist

**To run system, you MUST have:**
- [x] Python 3.8+
- [x] All required packages installed
- [ ] config.py with Oanda credentials (if using live trading)
- [x] At minimum: backtesting_engine.py

**To run FULL system, you SHOULD have:**
- [x] All 10 core Python files
- [x] Complete documentation
- [ ] 90 days paper trading results
- [ ] Tested strategies on out-of-sample data

---

**CONCLUSION**: The system has a clean, modular dependency structure. You can start with just 1-2 files and scale up to the full 10-file system based on your needs.
