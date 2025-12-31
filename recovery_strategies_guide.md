# Recovery Strategies for Low Win Rate & Model Failure

## 🚨 Critical Understanding

**Win rate dropping is NOT always model failure!**  
Market conditions change, and even good strategies have losing streaks.

**The key questions:**
1. Is this normal variance or actual failure?
2. Is the market regime changing?
3. Is the model overfitted to past data?

---

## 📊 PART 1: Early Warning Signals

### **Level 1: Yellow Flags** ⚠️ (Watch Closely)

| Metric | Warning Threshold | Action |
|--------|------------------|--------|
| Win rate (30 trades) | < 48% | Monitor daily |
| Sharpe ratio (30 days) | < 0.8 | Review strategy |
| Drawdown | > 10% | Reduce position size 25% |
| Consecutive losses | 3 in a row | Pause and review |
| Model confidence | < 60% | Check feature drift |

**What to do:**
```python
# Increase monitoring frequency
# Review last 20 trades manually
# Check if market conditions changed
# DO NOT panic - this is normal variance
```

### **Level 2: Orange Flags** 🟧 (Take Action)

| Metric | Critical Threshold | Action |
|--------|-------------------|--------|
| Win rate (30 trades) | < 45% | Reduce size 50% |
| Sharpe ratio (30 days) | < 0.5 | Pause new trades |
| Drawdown | > 15% | Close 50% of positions |
| Consecutive losses | 5 in a row | STOP trading |
| Model confidence | < 55% | Consider retraining |

**What to do:**
```python
# Reduce position sizes immediately
# Pause opening new positions
# Analyze losing trades for patterns
# Check if strategy assumptions still valid
```

### **Level 3: Red Flags** 🔴 (Emergency Stop)

| Metric | Emergency Threshold | Action |
|--------|-------------------|--------|
| Win rate (30 trades) | < 35% | STOP all trading |
| Sharpe ratio (30 days) | < 0 (negative) | Close ALL positions |
| Drawdown | > 20% | Halt system |
| Consecutive losses | 7+ in a row | Full review needed |
| Model confidence | < 50% | Model failed |

**What to do:**
```python
# IMMEDIATELY STOP ALL TRADING
# Close all open positions
# Send emergency alerts
# Full system review required before resuming
```

---

## 🔍 PART 2: Diagnosing the Root Cause

### **Diagnosis Framework**

```python
def diagnose_failure(trades_df, features_df):
    """
    Step-by-step diagnosis
    """
    
    # 1. NORMAL VARIANCE CHECK
    if len(trades_df) < 30:
        return "INSUFFICIENT_DATA - Need more trades"
    
    # 2. STATISTICAL SIGNIFICANCE
    from scipy import stats
    recent_wr = calculate_win_rate(trades_df, 30)
    baseline_wr = 0.55  # From backtesting
    
    # Binomial test
    p_value = stats.binom_test(wins, total_trades, baseline_wr)
    if p_value > 0.05:
        return "NORMAL_VARIANCE - Not statistically significant"
    
    # 3. MARKET REGIME CHANGE
    volatility_now = features_df['atr_14'].tail(30).mean()
    volatility_baseline = features_df['atr_14'].mean()
    
    if abs(volatility_now - volatility_baseline) / volatility_baseline > 0.3:
        return "MARKET_REGIME_CHANGE - Volatility shift detected"
    
    # 4. FEATURE DRIFT
    for feature in important_features:
        now_mean = features_df[feature].tail(100).mean()
        baseline_mean = features_df[feature].mean()
        
        drift = abs(now_mean - baseline_mean) / abs(baseline_mean)
        if drift > 0.2:
            return f"FEATURE_DRIFT - {feature} drifted {drift:.1%}"
    
    # 5. OVERFITTING
    if recent_wr < baseline_wr * 0.7:
        return "OVERFITTING - Model likely overfit to training data"
    
    return "UNKNOWN - Further investigation needed"
```

### **Common Root Causes**

**1. Normal Variance** ✅ (Not a problem)
- **Symptoms**: Win rate 48-52%, Sharpe 0.8-1.2
- **Cause**: Statistical fluctuation
- **Action**: Continue trading, monitor

**2. Market Regime Change** 🔄
- **Symptoms**: All strategies failing, high volatility
- **Cause**: Market structure shifted (e.g., crisis, new regulations)
- **Action**: Pause, retrain on recent data, adjust parameters

**3. Feature Drift** 📉
- **Symptoms**: Model confidence dropping, random-looking results
- **Cause**: Price patterns changed, indicators no longer predictive
- **Action**: Retrain model with fresh data

**4. Overfitting** ⚠️
- **Symptoms**: Great backtest, terrible live performance
- **Cause**: Model learned noise, not signal
- **Action**: Simplify model, use more regularization, get more data

**5. Data Quality Issues** 🔧
- **Symptoms**: Erratic behavior, unexplained losses
- **Cause**: Missing data, incorrect prices, API errors
- **Action**: Validate data pipeline, check API connection

**6. Execution Issues** ⚙️
- **Symptoms**: Losses on trades that should win
- **Cause**: High slippage, incorrect order execution
- **Action**: Check order execution logs, reduce position sizes

---

## 🛠️ PART 3: Recovery Strategies

### **Strategy 1: Reduce Position Size** (First Response)

```python
class AdaptiveRiskManager:
    def adjust_size(self, recent_performance):
        """
        Dynamically adjust position size based on performance
        """
        win_rate = recent_performance.win_rate_30
        sharpe = recent_performance.sharpe_30d
        
        # Normal performance: 100% size
        if win_rate > 0.52 and sharpe > 1.0:
            return 1.0
        
        # Slightly underperforming: 75% size
        elif win_rate > 0.48 and sharpe > 0.7:
            return 0.75
        
        # Underperforming: 50% size
        elif win_rate > 0.45 and sharpe > 0.5:
            return 0.5
        
        # Significantly underperforming: 25% size
        elif win_rate > 0.40:
            return 0.25
        
        # Failing: STOP
        else:
            return 0.0
```

**Benefits:**
- Preserves capital during drawdowns
- Allows strategy to prove itself again
- Reduces emotional stress

**When to use:** First sign of trouble (Level 1 warnings)

### **Strategy 2: Pause New Trades** (Defensive)

```python
def should_pause_trading(health_metrics):
    """
    Determine if new trades should be paused
    """
    # Pause if:
    if health_metrics.consecutive_losses >= 4:
        return True, "Too many consecutive losses"
    
    if health_metrics.sharpe_30d < 0.3:
        return True, "Sharpe ratio too low"
    
    if health_metrics.model_drift_score > 0.4:
        return True, "Model drift detected"
    
    return False, "Continue trading"
```

**Benefits:**
- Stops bleeding while keeping existing positions
- Time to analyze without pressure
- Prevents compounding losses

**When to use:** Level 2 warnings, unclear failure mode

### **Strategy 3: Reduce Trade Frequency** (Conservative)

```python
def filter_trades_by_confidence(signals, min_confidence=0.65):
    """
    Only take highest confidence trades
    """
    high_confidence = []
    
    for signal in signals:
        if signal['confidence'] >= min_confidence:
            high_confidence.append(signal)
    
    # During trouble: Only take top 30% of signals
    if is_underperforming:
        top_30_pct = int(len(high_confidence) * 0.3)
        return sorted(high_confidence, 
                     key=lambda x: x['confidence'], 
                     reverse=True)[:top_30_pct]
    
    return high_confidence
```

**Benefits:**
- Higher quality trades only
- Reduces exposure
- Better risk/reward

**When to use:** Model confidence dropping, unclear signals

### **Strategy 4: Increase Stop Loss Distance** (Risk Reduction)

```python
def adjust_stop_loss(normal_atr_distance, performance):
    """
    Widen stops during uncertainty
    """
    if performance.win_rate_30 < 0.48:
        # Wider stops = fewer stop-outs = higher win rate
        return normal_atr_distance * 1.5
    
    return normal_atr_distance
```

**Benefits:**
- Reduces premature stop-outs
- Gives trades more room
- Can improve win rate

**Risks:**
- Larger losses when wrong
- Lower risk/reward ratio

**When to use:** High volatility regime, frequent stop-outs

### **Strategy 5: Switch to Backup Strategy** (Failover)

```python
class StrategyManager:
    def __init__(self):
        self.primary = MLStrategy()
        self.backup = MomentumStrategy()  # Simple, robust
        self.current = self.primary
    
    def check_and_switch(self, performance):
        """
        Switch to backup if primary fails
        """
        if self.current == self.primary:
            if performance.win_rate_100 < 0.40:
                logger.warning("Switching to backup strategy")
                self.current = self.backup
                return True
        
        # Switch back if primary recovers
        elif performance.win_rate_100 > 0.55:
            logger.info("Primary strategy recovered, switching back")
            self.current = self.primary
            return True
        
        return False
```

**Benefits:**
- Continues trading with proven backup
- Automatic failover
- Diversification

**When to use:** Primary strategy clearly failing, backup validated

### **Strategy 6: Retrain Model** (Adaptation)

```python
def should_retrain_model(trades_df, features_df):
    """
    Determine if retraining is needed
    """
    # Check 1: Performance degradation
    if recent_sharpe < baseline_sharpe * 0.5:
        return True, "Performance degraded"
    
    # Check 2: Feature drift
    drift_score = calculate_feature_drift(features_df)
    if drift_score > 0.3:
        return True, "Feature drift detected"
    
    # Check 3: Time since last training
    if days_since_training > 30:
        return True, "Regular retraining schedule"
    
    # Check 4: Market regime change
    if detect_regime_change(features_df):
        return True, "Market regime changed"
    
    return False, "Model still valid"
```

**Retraining Protocol:**
1. **Use recent data only** (last 6 months)
2. **Keep validation strict** (walk-forward)
3. **Compare new vs old** (A/B test)
4. **Deploy gradually** (25% → 50% → 100%)

**When to use:** 
- Feature drift detected
- Market regime changed
- Monthly maintenance

### **Strategy 7: Take a Break** (Reset)

```python
def implement_cooling_off_period(duration_days=7):
    """
    Complete trading pause for system review
    """
    actions = []
    
    # 1. Close all positions
    close_all_positions()
    actions.append("All positions closed")
    
    # 2. Disable trading
    trading_enabled = False
    actions.append(f"Trading paused for {duration_days} days")
    
    # 3. Full system review
    actions.append("Scheduled: Full system review")
    
    # 4. Analyze all losing trades
    actions.append("Scheduled: Loss analysis")
    
    # 5. Retrain on fresh data
    actions.append("Scheduled: Model retraining")
    
    return actions
```

**Benefits:**
- Mental reset
- Time for thorough analysis
- Prevents emotional trading

**When to use:**
- Major drawdown (>15%)
- Unclear failure mode
- Emotional stress high

---

## 🎯 PART 4: Specific Recovery Playbooks

### **Playbook A: Win Rate 40-45% (Moderate Decline)**

```
Day 1:
✅ Reduce position size to 50%
✅ Review last 30 trades manually
✅ Check feature distributions
✅ Increase monitoring to real-time

Day 2-3:
✅ Run walk-forward validation on recent data
✅ Compare to baseline performance
✅ Check for data quality issues

Day 4-7:
✅ If no improvement: Pause new trades
✅ Retrain model on last 3 months
✅ A/B test new vs old model

Action: Continue with caution, close monitoring
```

### **Playbook B: Win Rate 35-40% (Serious Decline)**

```
IMMEDIATE:
🚨 Reduce position size to 25%
🚨 Close 50% of open positions
🚨 Pause new trades

Day 1:
✅ Full diagnostic review
✅ Check all data pipelines
✅ Review execution logs
✅ Calculate statistical significance

Day 2-7:
✅ Identify root cause
✅ If market regime change: Retrain completely
✅ If overfitting: Simplify model
✅ If data issues: Fix pipeline

Action: Minimal trading until diagnosed
```

### **Playbook C: Win Rate <35% (Critical Failure)**

```
IMMEDIATE:
❌ STOP ALL TRADING
❌ Close ALL positions
❌ Send emergency alerts

Day 1-3:
✅ Complete system audit
✅ Validate all code
✅ Check data integrity
✅ Review all assumptions

Day 4-14:
✅ Redesign strategy if needed
✅ Retrain from scratch
✅ Extended walk-forward validation
✅ Paper trade for 2 weeks minimum

Action: Full system rebuild required
```

---

## 📈 PART 5: Progressive Recovery Protocol

### **Phase 1: Identify (Days 1-2)**

```python
# Immediate actions
1. Calculate current metrics
2. Compare to baseline
3. Determine statistical significance
4. Identify root cause category
5. Take immediate protective action
```

### **Phase 2: Stabilize (Days 3-7)**

```python
# Stop the bleeding
1. Reduce position sizes
2. Tighten entry criteria
3. Close underperforming positions
4. Increase monitoring frequency
5. Document all changes
```

### **Phase 3: Diagnose (Days 8-14)**

```python
# Deep analysis
1. Review all losing trades
2. Check feature importance shifts
3. Validate data pipeline
4. Test on out-of-sample data
5. Consult baseline assumptions
```

### **Phase 4: Repair (Days 15-30)**

```python
# Fix the issue
1. If overfitting → Simplify model
2. If drift → Retrain on recent data
3. If regime change → New strategy
4. If execution → Fix infrastructure
5. Validate fix thoroughly
```

### **Phase 5: Test (Days 31-60)**

```python
# Prove it works
1. Paper trade for 30 days minimum
2. Must hit performance targets
3. Walk-forward validate
4. Compare to baseline
5. Get statistical confidence
```

### **Phase 6: Resume (Days 61+)**

```python
# Gradual return
1. Start with 25% position sizes
2. Increase to 50% after 20 trades
3. Increase to 75% after 50 trades
4. Return to 100% after 100 trades
5. Maintain enhanced monitoring
```

---

## 🔔 PART 6: Automated Monitoring System

```python
class TradingHealthMonitor:
    """
    Continuous health monitoring with automatic actions
    """
    
    def __init__(self, db_manager, oanda_connector):
        self.db = db_manager
        self.oanda = oanda_connector
        self.detector = ModelFailureDetector()
        self.recovery = RecoveryStrategy()
        
        # Set baselines from backtesting
        self.detector.set_baseline(
            win_rate=0.55,
            sharpe=1.2,
            profit_factor=1.8
        )
    
    def run_continuous_monitoring(self, check_interval_minutes=15):
        """
        Run continuous health checks
        """
        while True:
            try:
                # Get recent trades
                trades_df = self.db.get_trades(days=90)
                
                # Get predictions if ML model
                predictions_df = self.get_recent_predictions()
                
                # Health check
                health = self.detector.check_health(
                    trades_df, 
                    predictions_df
                )
                
                # Log health metrics
                self.log_health(health)
                
                # Take action if needed
                if health.status != HealthStatus.HEALTHY:
                    positions = self.oanda.get_open_positions()
                    action = self.recovery.execute_recovery(
                        health, 
                        positions
                    )
                    
                    # Send alerts
                    self.send_alert(health, action)
                
                # Sleep
                time.sleep(check_interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(60)
```

---

## ✅ PART 7: Checklist - "Is My Model Failing?"

### **Quick Diagnostic (2 minutes)**

- [ ] Win rate last 30 trades < 45%?
- [ ] Sharpe ratio last 30 days < 0.5?
- [ ] 4+ consecutive losses?
- [ ] Drawdown > 12%?
- [ ] Model confidence < 60%?

**If YES to 2+: Take action NOW**

### **Detailed Diagnostic (30 minutes)**

- [ ] Statistical significance test (binomial)
- [ ] Feature drift analysis
- [ ] Execution quality review
- [ ] Data pipeline validation
- [ ] Backtest vs live comparison
- [ ] Market regime check
- [ ] Volatility analysis

**If multiple issues: Stop and fix**

### **Root Cause Analysis (2 hours)**

- [ ] Review all losing trades manually
- [ ] Check feature importance changes
- [ ] Compare training vs live distributions
- [ ] Analyze by time of day
- [ ] Analyze by instrument
- [ ] Review code changes
- [ ] Check for overfitting signs

---

## 📊 PART 8: Key Metrics to Track

### **Daily Monitoring**

```python
daily_metrics = {
    'win_rate_10': 0.0,      # Last 10 trades
    'daily_pnl': 0.0,        # Today's P&L
    'open_positions': 0,      # Current exposure
    'avg_confidence': 0.0,    # Model confidence
    'execution_quality': 0.0  # Slippage analysis
}
```

### **Weekly Review**

```python
weekly_metrics = {
    'win_rate_30': 0.0,
    'sharpe_7d': 0.0,
    'max_dd_week': 0.0,
    'trade_count': 0,
    'avg_hold_time': 0.0
}
```

### **Monthly Analysis**

```python
monthly_metrics = {
    'win_rate_100': 0.0,
    'sharpe_30d': 0.0,
    'profit_factor': 0.0,
    'calmar_ratio': 0.0,
    'feature_drift_score': 0.0,
    'vs_baseline': 0.0
}
```

---

## 🎯 SUMMARY: Decision Tree

```
Win Rate Dropped?
├─ Still > 48%?
│  └─ Monitor closely ✅
│
├─ Between 45-48%?
│  └─ Reduce size 50% ⚠️
│
├─ Between 40-45%?
│  ├─ Statistically significant?
│  │  ├─ No → Continue with caution
│  │  └─ Yes → Pause & diagnose
│  └─
│
└─ Below 40%?
   └─ STOP TRADING 🚨
      └─ Full system review required
```

---

## 💡 FINAL TIPS

**DO:**
✅ Monitor continuously
✅ React gradually (reduce size first)
✅ Keep detailed logs
✅ Statistical testing before panic
✅ Have backup strategies ready
✅ Set clear stop-loss rules for the system itself

**DON'T:**
❌ Panic after 3 losing trades
❌ Constantly retrain model
❌ Abandon strategy without analysis
❌ Increase size during drawdown
❌ Trade emotionally
❌ Ignore warning signals

---

**Remember**: Even the best strategies lose 40-45% of the time. The key is knowing when normal variance becomes actual failure.
