# Trading Dashboard Setup Guide

Complete guide to set up your Oanda trading monitoring dashboard.

## 📋 Table of Contents
1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Oanda API Setup](#oanda-api-setup)
4. [Quick Start](#quick-start)
5. [Dashboard Features](#dashboard-features)
6. [Integration with Live Trading](#integration)
7. [Troubleshooting](#troubleshooting)

---

## Requirements

### Python Requirements
Create a `requirements.txt` file:

```txt
# Core Dashboard
streamlit>=1.30.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0

# Oanda API
oandapyV20>=0.7.2

# Database
sqlite3  # Built into Python

# ML & Analysis (optional, for future features)
scikit-learn>=1.3.0
xgboost>=2.0.0

# Utilities
python-dateutil>=2.8.0
```

### System Requirements
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- Internet connection for Oanda API
- Modern web browser (Chrome, Firefox, Safari, Edge)

---

## Installation

### Step 1: Clone or Download Files

Organize your project directory:
```
trading_dashboard/
├── trading_dashboard_main.py       # Main dashboard application
├── oanda_integration.py           # Oanda API connector
├── sample_data_generator.py       # Test data generator
├── requirements.txt               # Python dependencies
├── trading_data.db               # SQLite database (auto-created)
└── README.md                     # This file
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Oanda API Setup

### Step 1: Get Oanda Account

1. Go to [Oanda](https://www.oanda.com/)
2. Sign up for a **Practice Account** (free)
   - Do NOT start with a live account
3. Once registered, log in to fxTrade Practice

### Step 2: Generate API Token

1. In fxTrade, go to **Manage API Access**
2. Click **Generate** to create a new API token
3. **IMPORTANT**: Copy and save this token immediately - you can't view it again!
4. Note your Account ID (format: `###-###-#######-###`)

### Step 3: Configure Credentials

Create a `config.py` file:

```python
# config.py
# KEEP THIS FILE SECURE - DO NOT COMMIT TO GIT

OANDA_CONFIG = {
    'account_id': 'your-account-id-here',
    'access_token': 'your-access-token-here',
    'environment': 'practice'  # or 'live' for production
}
```

Add to `.gitignore`:
```
config.py
trading_data.db
__pycache__/
venv/
*.pyc
```

---

## Quick Start

### Option 1: Test with Sample Data (Recommended First)

```bash
# Generate 90 days of sample trading data
python sample_data_generator.py

# Launch dashboard
streamlit run trading_dashboard_main.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Option 2: Connect to Oanda (Real Data)

```python
# sync_oanda_data.py
from oanda_integration import OandaConnector, DashboardDataSync
from trading_dashboard_main import DatabaseManager
from config import OANDA_CONFIG

# Initialize
oanda = OandaConnector(
    account_id=OANDA_CONFIG['account_id'],
    access_token=OANDA_CONFIG['access_token'],
    environment=OANDA_CONFIG['environment']
)

db = DatabaseManager()
sync = DashboardDataSync(oanda, db)

# Sync data
print("Syncing data from Oanda...")
sync.sync_all()
print("✅ Sync complete!")

# Start auto-sync (optional - runs continuously)
# sync.start_auto_sync(interval_seconds=60)
```

Run sync then launch dashboard:
```bash
python sync_oanda_data.py
streamlit run trading_dashboard_main.py
```

---

## Dashboard Features

### 📈 Overview Page
- **Real-time Metrics**: Current equity, daily P&L, total trades
- **Equity Curve**: Visual representation of account growth
- **Performance Summary**: Sharpe ratio, profit factor, win rate
- **Daily P&L Chart**: Bar chart of daily performance
- **Open Positions**: Current positions with unrealized P&L

### 📋 Trades Page
- **Complete Trade History**: All closed trades
- **Filters**: By instrument, direction, strategy
- **Export**: Download trades as CSV
- **Detailed Information**: Entry/exit prices, P&L, duration

### ⚠️ Risk Monitor
- **Alert System**: Critical, warning, and info alerts
- **Risk Limits**: Daily loss and max drawdown tracking
- **Position Concentration**: Exposure by instrument
- **Recent Performance**: Last 10 trades analysis
- **Kill Switch Indicators**: Automatic trading halt conditions

### 🔍 Analysis Page
- **Performance by Instrument**: Which pairs are most profitable
- **Performance by Strategy**: Compare strategy effectiveness
- **Time-based Analysis**: Best hours/days for trading
- **Statistical Breakdown**: Comprehensive metrics

### ⚙️ Settings Page
- **Risk Parameters**: Configure risk limits
- **Alert Configuration**: Email/Slack notifications
- **Data Management**: Export and cleanup tools

---

## Integration with Live Trading

### Automated Sync Script

Create a continuous sync service:

```python
# continuous_sync.py
from oanda_integration import OandaConnector, DashboardDataSync
from trading_dashboard_main import DatabaseManager
from config import OANDA_CONFIG
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    # Initialize
    oanda = OandaConnector(
        account_id=OANDA_CONFIG['account_id'],
        access_token=OANDA_CONFIG['access_token'],
        environment=OANDA_CONFIG['environment']
    )
    
    db = DatabaseManager()
    sync = DashboardDataSync(oanda, db)
    
    # Continuous sync every 60 seconds
    logging.info("Starting continuous sync...")
    
    while True:
        try:
            sync.sync_all()
            time.sleep(60)  # Sync every minute
        except KeyboardInterrupt:
            logging.info("Sync stopped by user")
            break
        except Exception as e:
            logging.error(f"Sync error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
```

Run in background:
```bash
# On Linux/Mac with screen
screen -S oanda_sync
python continuous_sync.py
# Detach with Ctrl+A, D

# On Windows with pythonw (no console)
pythonw continuous_sync.py
```

### Integration with Trading Bot

```python
# In your trading bot code
from trading_dashboard_main import DatabaseManager

class TradingBot:
    def __init__(self):
        self.db = DatabaseManager()
        # ... other initialization
    
    def on_trade_closed(self, trade_data):
        """Called when a trade closes"""
        # Log trade to dashboard
        self.db.add_trade({
            'trade_id': trade_data['id'],
            'instrument': trade_data['instrument'],
            'direction': trade_data['direction'],
            'entry_time': trade_data['entry_time'],
            'exit_time': trade_data['exit_time'],
            'entry_price': trade_data['entry_price'],
            'exit_price': trade_data['exit_price'],
            'size': trade_data['size'],
            'pnl': trade_data['pnl'],
            'pnl_percent': trade_data['pnl_percent'],
            'commission': trade_data['commission'],
            'slippage': trade_data['slippage'],
            'strategy': 'your_strategy_name',
            'status': 'closed'
        })
    
    def check_risk_limits(self):
        """Check if risk limits exceeded"""
        from trading_dashboard_main import PerformanceCalculator, RiskMonitor
        
        trades_df = self.db.get_trades(days=1)  # Today's trades
        metrics = PerformanceCalculator.calculate_metrics(trades_df)
        
        risk_monitor = RiskMonitor()
        alerts = risk_monitor.check_risk_limits(
            metrics, 
            daily_pnl=trades_df['pnl'].sum(),
            capital=10000  # Your current capital
        )
        
        # If critical alerts, halt trading
        critical_alerts = [a for a in alerts if a.level == 'critical']
        if critical_alerts:
            self.stop_trading()
            for alert in critical_alerts:
                self.db.add_alert(alert)
```

---

## Troubleshooting

### Issue: Dashboard won't start
```bash
# Check if all dependencies installed
pip list | grep streamlit

# Try reinstalling
pip install --upgrade streamlit

# Check Python version
python --version  # Should be 3.8+
```

### Issue: Oanda connection fails
```python
# Test connection
from oanda_integration import OandaConnector
from config import OANDA_CONFIG

oanda = OandaConnector(**OANDA_CONFIG)
account = oanda.get_account_summary()
print(account)  # Should print account details
```

**Common fixes:**
- Verify API token is correct (no extra spaces)
- Check account ID format
- Ensure using 'practice' environment for practice accounts
- Check internet connection
- Verify Oanda API is not under maintenance

### Issue: Database errors
```bash
# Reset database (WARNING: deletes all data)
rm trading_data.db
python sample_data_generator.py  # Recreate with sample data
```

### Issue: Charts not displaying
- Clear browser cache
- Try different browser
- Check browser console for JavaScript errors (F12)
- Ensure plotly installed: `pip install --upgrade plotly`

### Issue: Performance is slow
- Reduce data displayed (filter by fewer days)
- Check system resources (RAM usage)
- Close other applications
- Consider running on more powerful machine

---

## Production Deployment

### Running on Server

```bash
# Install screen or tmux
sudo apt-get install screen  # Ubuntu/Debian

# Start persistent session
screen -S trading_dashboard

# Launch dashboard on specific port
streamlit run trading_dashboard_main.py --server.port 8501

# Detach: Ctrl+A, D
# Reattach: screen -r trading_dashboard
```

### Nginx Reverse Proxy (Optional)

```nginx
# /etc/nginx/sites-available/trading-dashboard
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

### Systemd Service (Linux)

```ini
# /etc/systemd/system/trading-dashboard.service
[Unit]
Description=Trading Dashboard
After=network.target

[Service]
Type=simple
User=yourusername
WorkingDirectory=/path/to/trading_dashboard
ExecStart=/path/to/venv/bin/streamlit run trading_dashboard_main.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable trading-dashboard
sudo systemctl start trading-dashboard
sudo systemctl status trading-dashboard
```

---

## Security Best Practices

1. **Never commit credentials** - Use environment variables or config files in .gitignore
2. **Use HTTPS** - For production deployments
3. **Restrict access** - Use authentication (Streamlit supports this)
4. **Regular backups** - Backup `trading_data.db` regularly
5. **Monitor API usage** - Stay within Oanda rate limits
6. **Practice first** - Never test on live account initially

---

## Next Steps

1. ✅ Set up dashboard with sample data
2. ✅ Connect to Oanda practice account
3. ✅ Run paper trading for 3 months minimum
4. ⚠️ Analyze results and adjust strategies
5. 📊 Track key metrics daily
6. 🚀 Only then consider live trading with small capital

---

## Support & Resources

- **Oanda API Documentation**: https://developer.oanda.com/
- **Streamlit Documentation**: https://docs.streamlit.io/
- **Pandas Documentation**: https://pandas.pydata.org/docs/

---

## License

This dashboard is provided as-is for educational and personal use. 

**DISCLAIMER**: Trading S&P 500 and CFDs involves significant risk. This dashboard is a monitoring tool and does not guarantee profits. Always trade responsibly and never risk more than you can afford to lose.

---

**Created for monitoring Oanda trading performance**
Version 1.0 | October 2025
