# Quick Reference: Model Failure & Recovery

## 🚦 Traffic Light System

### 🟢 GREEN (Healthy)
- Win rate: > 50%
- Sharpe: > 1.0
- Drawdown: < 10%
- **Action**: Continue normal trading

### 🟡 YELLOW (Warning)
- Win rate: 45-50%
- Sharpe: 0.5-1.0
- Drawdown: 10-15%
- Consecutive losses: 3-4
- **Action**: Reduce size 50%, monitor closely

### 🟠 ORANGE (Critical)
- Win rate: 40-45%
- Sharpe: 0-0.5
- Drawdown: 15-20%
- Consecutive losses: 5-6
- **Action**: Reduce size 75%, pause new trades

### 🔴 RED (Failed)
- Win rate: < 40%
- Sharpe: < 0
- Drawdown: > 20%
- Consecutive losses: 7+
- **Action**: STOP ALL TRADING

---

## ⚡ Immediate Actions by Situation

| Situation | Immediate Action | Timeline |
|-----------|-----------------|----------|
| 3 losses in row | Monitor closely | Hours |
| 4 losses in row | Reduce size 50% | Immediate |
| 5 losses in row | Stop new trades | Immediate |
| 7+ losses in row | STOP ALL | Immediate |
| Win rate < 48% (30 trades) | Reduce size 25% | 1 day |
| Win rate < 45% (30 trades) | Reduce size 50% | Immediate |
| Win rate < 40% (30 trades) | STOP ALL | Immediate |
| Sharpe < 0.5 (30 days) | Pause & review | 2 days |
| Sharpe < 0 (30 days) | STOP ALL | Immediate |
| Drawdown > 12% | Reduce size 50% | Immediate |
| Drawdown > 15% | Close 50% positions | Immediate |
| Drawdown > 20% | STOP ALL | Immediate |

---

## 🔍 Diagnosis Checklist

**Is it actually failing? (5 minutes)**
```
1. [ ] Calculate win rate last 30 trades
2. [ ] Calculate Sharpe ratio last 30 days  
3. [ ] Check consecutive losses
4. [ ] Calculate current drawdown
5. [ ] Compare to baseline metrics

If 2+ metrics in RED zone → Take action
If all in YELLOW zone → Might be normal variance
```

**What's the root cause? (30 minutes)**
```
1. [ ] Run statistical significance test
2. [ ] Check feature drift score
3. [ ] Compare volatility: now vs baseline
4. [ ] Review execution quality (slippage)
5. [ ] Check data pipeline integrity

Common causes:
- Normal variance (wait it out)
- Market regime change (retrain)
- Overfitting (simplify model)
- Data issues (fix pipeline)
```

---

## 🛠️ Recovery Action Matrix

| Root Cause | Immediate | Short-term (1 week) | Long-term (1 month) |
|------------|-----------|-------------------|-------------------|
| **Normal Variance** | Monitor | Continue | Continue |
| **Market Regime Change** | Reduce size 50% | Retrain model | Deploy new model |
| **Overfitting** | Reduce size 75% | Simplify model | Retrain with regularization |
| **Feature Drift** | Pause new trades | Retrain on recent data | Regular retraining schedule |
| **Data Issues** | STOP | Fix pipeline | Automated validation |
| **Execution Problems** | Reduce size 50% | Fix infrastructure | Better broker/API |

---

## 📊 Key Metrics to Track

**Every Day:**
- Win rate (last 10 trades)
- Daily P&L
- Open position count
- Model confidence

**Every Week:**
- Win rate (last 30 trades)
- Sharpe ratio (7 days)
- Max weekly drawdown
- Trade count

**Every Month:**
- Win rate (last 100 trades)
- Sharpe ratio (30 days)
- Feature drift score
- vs Baseline comparison

---

## 🎯 Position Size Adjustment

```
Performance         → Position Size
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Excellent (WR>55%, Sharpe>1.5) → 150%
Good (WR>52%, Sharpe>1.0)      → 100%
Acceptable (WR>50%, Sharpe>0.7)→ 75%
Warning (WR>45%, Sharpe>0.5)   → 50%
Critical (WR>40%)              → 25%
Failed (WR<40%)                → 0% (STOP)
```

---

## ⏱️ Recovery Timeline

**Day 1:**
- Identify problem
- Take protective action
- Start diagnosis

**Days 2-7:**
- Complete root cause analysis
- Implement fix (retrain/adjust)
- Test fix thoroughly

**Days 8-30:**
- Paper trade new system
- Monitor closely
- Validate performance

**Days 31+:**
- Gradual capital deployment
- 25% → 50% → 75% → 100%
- Enhanced monitoring

---

## 🚨 Emergency Contacts

**When to escalate:**
- Drawdown > 15%
- 7+ consecutive losses
- Sharpe < 0 for 7+ days
- Unclear failure mode
- System errors

**Who to contact:**
- Risk manager
- Technical lead
- Broker support
- Your future self (write detailed notes!)

---

## 📝 Daily Health Check (2 minutes)

```python
# Morning routine
1. Check overnight performance
2. Review open positions
3. Calculate daily metrics
4. Compare to baseline
5. Adjust position sizes if needed

# Questions to ask:
- Any new warnings?
- Are metrics trending wrong direction?
- Is anything outside normal range?
- Do I understand what's happening?
```

---

## 🎓 Common Mistakes to Avoid

**DON'T:**
❌ Panic after 2-3 losses
❌ Keep trading during clear failure
❌ Increase size during drawdown
❌ Constantly retrain model
❌ Ignore warning signals
❌ Trade emotionally
❌ Skip paper trading after changes

**DO:**
✅ Have predefined rules
✅ Reduce size progressively
✅ Keep detailed logs
✅ Use statistical tests
✅ Paper trade fixes
✅ Maintain discipline
✅ Review regularly

---

## 📞 Quick Decision Tree

```
Am I losing money?
│
├─ No → Continue ✅
│
└─ Yes → How many consecutive losses?
    │
    ├─ 1-2 → Normal, continue ✅
    │
    ├─ 3-4 → Reduce size 50% ⚠️
    │
    ├─ 5-6 → Stop new trades 🟠
    │
    └─ 7+ → STOP EVERYTHING 🔴
```

```
Is performance worse than backtest?
│
├─ Within 20% → Normal variance ✅
│
├─ 20-40% worse → Investigate ⚠️
│
└─ 40%+ worse → Stop & fix 🔴
```

---

## 💡 Golden Rules

1. **Set clear rules BEFORE trading**
2. **Follow rules without emotion**
3. **Reduce risk FIRST, investigate second**
4. **Never trade through uncertainty**
5. **Paper trade all changes**
6. **Keep detailed records**
7. **When in doubt, sit out**

---

## 🔧 Essential Code Snippet

```python
# Add to your trading loop
def should_i_trade():
    """Quick check before each trade"""
    
    recent = get_last_30_trades()
    
    win_rate = calculate_win_rate(recent)
    drawdown = calculate_current_drawdown()
    consecutive_losses = count_consecutive_losses(recent)
    
    # Stop conditions
    if win_rate < 0.40:
        return False, "Win rate too low"
    
    if drawdown > 0.15:
        return False, "Drawdown too high"
    
    if consecutive_losses >= 5:
        return False, "Too many losses in row"
    
    # Warning conditions  
    if win_rate < 0.48:
        reduce_position_size(0.5)
    
    return True, "OK to trade"
```

---

## 📱 Emergency Protocol

**If Sharpe < 0 or Drawdown > 20%:**

1. **Immediately**: Close all positions
2. **Within 1 hour**: Notify stakeholders
3. **Within 24 hours**: Complete diagnosis
4. **Within 1 week**: Implement fix
5. **Within 1 month**: Validate fix via paper trading
6. **Resume only after**: 30+ days successful paper trading

---

**Remember**: It's better to miss profits than to lose capital. When in doubt, reduce risk or stop trading.
