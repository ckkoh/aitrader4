# Development & Deployment Plan
## AI S&P 500 trading System

**System Status**: 98% Complete (~8,000 lines of Python code)
**Created**: 2025-12-30
**Purpose**: Roadmap for completing, testing, and deploying the trading system

---

## Executive Summary

The AI S&P 500 trading System is functionally complete with all core components implemented:
- ✅ Backtesting engine with walk-forward validation
- ✅ Feature engineering (50+ technical indicators)
- ✅ ML training pipeline (5 models)
- ✅ 6 pre-built strategies
- ✅ Real-time Streamlit dashboard
- ✅ Oanda API integration
- ✅ Model failure detection and recovery
- ✅ Comprehensive documentation

**Remaining Work**: Small gaps in integration, testing infrastructure, and deployment automation.

---

## Phase 1: Complete Missing Core Features
**Timeline**: 1-2 days
**Priority**: HIGH

### 1.1 Implement Historical Data Fetching
**File**: `oanda_integration.py`
**Status**: Placeholder exists, needs implementation
**Impact**: Required for automated data collection

**Current Code** (line ~50):
```python
def fetch_historical_data(self, instrument, granularity, days):
    # TODO: Implement using Oanda API
    pass
```

**Required Implementation**:
```python
import oandapyV20.endpoints.instruments as instruments

def fetch_historical_data(self, instrument: str, granularity: str = 'H1',
                         days: int = 365) -> pd.DataFrame:
    """
    Fetch historical OHLCV data from Oanda

    Args:
        instrument: e.g., 'SPX500_USD'
        granularity: 'M1', 'M5', 'H1', 'H4', 'D'
        days: Number of days of history

    Returns:
        DataFrame with OHLCV data
    """
    from_time = (datetime.now() - timedelta(days=days)).isoformat() + 'Z'
    to_time = datetime.now().isoformat() + 'Z'

    params = {
        "granularity": granularity,
        "from": from_time,
        "to": to_time
    }

    r = instruments.InstrumentsCandles(instrument=instrument, params=params)
    response = self.client.request(r)

    candles = []
    for candle in response.get('candles', []):
        if candle['complete']:  # Only use complete candles
            candles.append({
                'time': candle['time'],
                'open': float(candle['mid']['o']),
                'high': float(candle['mid']['h']),
                'low': float(candle['mid']['l']),
                'close': float(candle['mid']['c']),
                'volume': int(candle['volume'])
            })

    df = pd.DataFrame(candles)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    logger.info(f"Fetched {len(df)} candles for {instrument}")
    return df
```

**Testing**:
- Test with practice account
- Verify data quality (no gaps, correct OHLCV format)
- Test different granularities (M1, H1, D)
- Handle API rate limits (max 500 candles per request)

### 1.2 Implement Email/SMS Alerts
**File**: `monitoring_integration.py`
**Status**: Structure exists (line 274), needs implementation
**Impact**: Critical for live trading notifications

**Current Code**:
```python
def send_alert(self, message: str, level: str = 'info'):
    # TODO: Implement actual email/SMS/Slack notification
    pass
```

**Required Implementation**:
```python
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email_alert(self, message: str, level: str = 'info'):
    """Send email alert"""
    if not self.config.get('email_alerts_enabled'):
        return

    subject = f"Trading Alert [{level.upper()}]: {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    msg = MIMEMultipart()
    msg['From'] = self.config['email_from']
    msg['To'] = self.config['email_to']
    msg['Subject'] = subject

    body = f"""
    Trading System Alert

    Level: {level.upper()}
    Time: {datetime.now()}

    Message:
    {message}

    ---
    AI S&P 500 trading System
    """

    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port'])
        server.starttls()
        server.login(self.config['email_from'], self.config['email_password'])
        server.send_message(msg)
        server.quit()
        logger.info(f"Email alert sent: {level}")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")

def send_slack_alert(self, message: str, level: str = 'info'):
    """Send Slack notification"""
    import requests

    if not self.config.get('slack_webhook_url'):
        return

    color = {
        'critical': '#ff0000',
        'warning': '#ff9900',
        'info': '#0066cc'
    }.get(level, '#808080')

    payload = {
        "attachments": [{
            "color": color,
            "title": f"Trading Alert [{level.upper()}]",
            "text": message,
            "footer": "AI S&P 500 trading System",
            "ts": int(datetime.now().timestamp())
        }]
    }

    try:
        response = requests.post(self.config['slack_webhook_url'], json=payload)
        if response.status_code == 200:
            logger.info(f"Slack alert sent: {level}")
    except Exception as e:
        logger.error(f"Failed to send Slack notification: {e}")
```

**Configuration Required** (add to `config.py`):
```python
ALERT_CONFIG = {
    'email_alerts_enabled': True,
    'email_from': 'your-email@gmail.com',
    'email_to': 'your-email@gmail.com',
    'email_password': 'your-app-password',  # Use app-specific password
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,

    'slack_webhook_url': 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL',
}
```

### 1.3 Create Configuration Validator
**File**: New file `config_validator.py`
**Status**: Does not exist
**Impact**: Prevents runtime errors from missing/invalid config

**Implementation**:
```python
"""
Configuration Validator
Ensures all required config values are present and valid before starting
"""

from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class ConfigValidator:
    """Validates trading system configuration"""

    REQUIRED_OANDA_KEYS = ['account_id', 'access_token', 'environment']
    REQUIRED_BACKTEST_KEYS = ['initial_capital', 'commission_pct', 'slippage_pct']
    REQUIRED_RISK_KEYS = ['max_daily_loss_pct', 'max_drawdown_pct', 'position_size_pct']

    @staticmethod
    def validate_oanda_config(config: Dict) -> Tuple[bool, List[str]]:
        """Validate Oanda configuration"""
        errors = []

        for key in ConfigValidator.REQUIRED_OANDA_KEYS:
            if key not in config or not config[key]:
                errors.append(f"Missing required key: {key}")

        if 'environment' in config and config['environment'] not in ['practice', 'live']:
            errors.append("environment must be 'practice' or 'live'")

        if 'account_id' in config:
            # Validate account ID format (###-###-#######-###)
            parts = config['account_id'].split('-')
            if len(parts) != 4:
                errors.append("account_id format invalid (should be ###-###-#######-###)")

        return (len(errors) == 0, errors)

    @staticmethod
    def validate_backtest_config(config: Dict) -> Tuple[bool, List[str]]:
        """Validate backtest configuration"""
        errors = []

        for key in ConfigValidator.REQUIRED_BACKTEST_KEYS:
            if key not in config:
                errors.append(f"Missing required key: {key}")

        if config.get('initial_capital', 0) <= 0:
            errors.append("initial_capital must be > 0")

        if config.get('commission_pct', -1) < 0:
            errors.append("commission_pct cannot be negative")

        return (len(errors) == 0, errors)

    @staticmethod
    def validate_risk_config(config: Dict) -> Tuple[bool, List[str]]:
        """Validate risk management configuration"""
        errors = []

        for key in ConfigValidator.REQUIRED_RISK_KEYS:
            if key not in config:
                errors.append(f"Missing required key: {key}")

        if config.get('max_daily_loss_pct', 0) <= 0 or config.get('max_daily_loss_pct', 1) > 0.5:
            errors.append("max_daily_loss_pct should be between 0 and 0.5 (50%)")

        if config.get('position_size_pct', 0) <= 0 or config.get('position_size_pct', 1) > 0.1:
            errors.append("position_size_pct should be between 0 and 0.1 (10%)")

        return (len(errors) == 0, errors)

    @staticmethod
    def validate_all(oanda_config: Dict, backtest_config: Dict, risk_config: Dict) -> bool:
        """Validate all configurations"""
        all_valid = True

        valid, errors = ConfigValidator.validate_oanda_config(oanda_config)
        if not valid:
            logger.error("Oanda config validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            all_valid = False

        valid, errors = ConfigValidator.validate_backtest_config(backtest_config)
        if not valid:
            logger.error("Backtest config validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            all_valid = False

        valid, errors = ConfigValidator.validate_risk_config(risk_config)
        if not valid:
            logger.error("Risk config validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            all_valid = False

        if all_valid:
            logger.info("✅ All configurations valid")

        return all_valid

# Usage example
if __name__ == "__main__":
    from config import OANDA_CONFIG, BACKTEST_CONFIG, RISK_CONFIG

    validator = ConfigValidator()

    if validator.validate_all(OANDA_CONFIG, BACKTEST_CONFIG, RISK_CONFIG):
        print("✅ Configuration is valid - ready to trade")
    else:
        print("❌ Configuration has errors - please fix before running")
```

**Deliverables**:
- [ ] Complete `fetch_historical_data()` method
- [ ] Implement email alerts with Gmail/SMTP
- [ ] Implement Slack alerts with webhook
- [ ] Create `config_validator.py`
- [ ] Test all implementations with practice account

---

## Phase 2: Testing Infrastructure
**Timeline**: 2-3 days
**Priority**: HIGH

### 2.1 Create Pytest Test Suite
**Status**: Does not exist
**Impact**: Critical for code quality and reliability

**Structure**:
```
tests/
├── __init__.py
├── conftest.py                    # Shared fixtures
├── test_backtesting_engine.py     # Core backtesting tests
├── test_feature_engineering.py    # Feature generation tests
├── test_ml_training.py            # ML pipeline tests
├── test_strategies.py             # Strategy logic tests
├── test_oanda_integration.py      # API integration tests (mocked)
├── test_dashboard.py              # Database operations tests
└── test_risk_management.py        # Risk checks tests
```

**Priority Test Files**:

**tests/conftest.py**:
```python
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing"""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='H')
    np.random.seed(42)

    data = pd.DataFrame({
        'open': 1.1000 + np.random.randn(len(dates)) * 0.001,
        'high': 1.1010 + np.random.randn(len(dates)) * 0.001,
        'low': 1.0990 + np.random.randn(len(dates)) * 0.001,
        'close': 1.1000 + np.random.randn(len(dates)) * 0.001,
        'volume': np.random.randint(100, 1000, len(dates))
    }, index=dates)

    # Ensure OHLC relationships are valid
    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)

    return data

@pytest.fixture
def backtest_config():
    """Standard backtest configuration"""
    from backtesting_engine import BacktestConfig
    return BacktestConfig(
        initial_capital=10000,
        commission_pct=0.0001,
        slippage_pct=0.0001,
        position_size_pct=0.02
    )
```

**tests/test_backtesting_engine.py**:
```python
import pytest
from backtesting_engine import BacktestEngine, BacktestConfig, Strategy
from strategy_examples import MomentumStrategy

def test_backtest_engine_initialization(backtest_config):
    """Test BacktestEngine initializes correctly"""
    engine = BacktestEngine(backtest_config)
    assert engine.config.initial_capital == 10000
    assert engine.config.commission_pct == 0.0001

def test_simple_backtest(sample_ohlcv_data, backtest_config):
    """Test basic backtesting functionality"""
    strategy = MomentumStrategy(fast_period=20, slow_period=50)
    engine = BacktestEngine(backtest_config)

    result = engine.run_backtest(strategy, sample_ohlcv_data)

    assert 'metrics' in result
    assert 'trades' in result
    assert 'equity_curve' in result
    assert result['metrics']['total_trades'] >= 0

def test_metrics_calculation(sample_ohlcv_data, backtest_config):
    """Test performance metrics are calculated correctly"""
    strategy = MomentumStrategy(fast_period=20, slow_period=50)
    engine = BacktestEngine(backtest_config)

    result = engine.run_backtest(strategy, sample_ohlcv_data)
    metrics = result['metrics']

    # Verify all expected metrics exist
    required_metrics = [
        'total_trades', 'win_rate', 'profit_factor',
        'sharpe_ratio', 'max_drawdown_pct', 'total_return_pct'
    ]
    for metric in required_metrics:
        assert metric in metrics

    # Verify metric validity
    assert 0 <= metrics['win_rate'] <= 1
    assert metrics['total_trades'] >= 0
    assert metrics['max_drawdown_pct'] <= 0

def test_walk_forward_analysis(sample_ohlcv_data, backtest_config):
    """Test walk-forward validation"""
    strategy = MomentumStrategy(fast_period=20, slow_period=50)
    engine = BacktestEngine(backtest_config)

    results = engine.walk_forward_analysis(
        strategy=strategy,
        data=sample_ohlcv_data,
        train_period_days=180,
        test_period_days=60,
        step_days=30
    )

    assert len(results) > 0
    for result in results:
        assert 'metrics' in result
        assert 'train_period' in result
        assert 'test_period' in result

def test_position_sizing_methods(sample_ohlcv_data):
    """Test different position sizing methods"""
    from backtesting_engine import PositionSizer

    capital = 10000
    risk_pct = 0.02
    entry_price = 1.1000
    stop_loss = 1.0950
    atr = 0.0020

    # Test fixed percentage
    size1 = PositionSizer.fixed_percentage(capital, risk_pct, entry_price, stop_loss)
    assert size1 > 0

    # Test volatility adjusted
    size2 = PositionSizer.volatility_adjusted(capital, risk_pct, atr, entry_price)
    assert size2 > 0

    # Test Kelly criterion
    size3 = PositionSizer.kelly_criterion(capital, win_rate=0.55, avg_win=100, avg_loss=80)
    assert size3 > 0
```

**tests/test_feature_engineering.py**:
```python
import pytest
from feature_engineering import TechnicalIndicators, FeatureEngineering

def test_sma_calculation(sample_ohlcv_data):
    """Test SMA indicator calculation"""
    df = TechnicalIndicators.add_sma(sample_ohlcv_data.copy(), periods=[20, 50])

    assert 'sma_20' in df.columns
    assert 'sma_50' in df.columns
    assert df['sma_20'].notna().sum() > 0
    assert df['sma_50'].notna().sum() > 0

def test_rsi_calculation(sample_ohlcv_data):
    """Test RSI indicator calculation"""
    df = TechnicalIndicators.add_rsi(sample_ohlcv_data.copy(), period=14)

    assert 'rsi_14' in df.columns
    # RSI should be between 0 and 100
    assert df['rsi_14'].min() >= 0
    assert df['rsi_14'].max() <= 100

def test_complete_feature_set(sample_ohlcv_data):
    """Test complete feature generation"""
    df = FeatureEngineering.build_complete_feature_set(
        sample_ohlcv_data.copy(),
        include_volume=True
    )

    # Should have significantly more columns than original
    assert len(df.columns) > 30

    # Check for key feature categories
    feature_names = df.columns.tolist()
    assert any('sma' in col for col in feature_names)
    assert any('rsi' in col for col in feature_names)
    assert any('macd' in col for col in feature_names)

def test_feature_selection(sample_ohlcv_data):
    """Test feature selection functionality"""
    df = FeatureEngineering.build_complete_feature_set(
        sample_ohlcv_data.copy(),
        include_volume=True
    )

    # Create dummy target
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna()

    # Select top features
    selected_features, importances = FeatureEngineering.select_features(
        df,
        target_col='target',
        n_features=20
    )

    assert len(selected_features) == 20
    assert len(importances) == 20
```

**Run Configuration** (`pytest.ini`):
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    -v
    --tb=short
    --strict-markers
    --disable-warnings
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
```

**Deliverables**:
- [ ] Create test directory structure
- [ ] Implement conftest.py with fixtures
- [ ] Write tests for backtesting_engine (10+ tests)
- [ ] Write tests for feature_engineering (8+ tests)
- [ ] Write tests for ML pipeline (6+ tests)
- [ ] Write tests for strategies (5+ tests)
- [ ] Achieve >80% code coverage
- [ ] Set up GitHub Actions for CI (optional)

### 2.2 Integration Tests with Paper Trading
**Timeline**: 1 day
**Priority**: MEDIUM

Create integration test that:
1. Connects to Oanda practice account
2. Fetches historical data
3. Trains a simple model
4. Runs backtest
5. Makes 1-2 paper trades
6. Monitors for 1 hour
7. Verifies all components work together

**File**: `tests/test_integration_end_to_end.py`

---

## Phase 3: Deployment Automation
**Timeline**: 2-3 days
**Priority**: MEDIUM

### 3.1 Docker Containerization
**Status**: Does not exist
**Impact**: Easier deployment and environment consistency

**Dockerfile**:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py ./
COPY config.py ./

# Create directories
RUN mkdir -p data models results logs

# Expose Streamlit port
EXPOSE 8501

# Default command: Run dashboard
CMD ["streamlit", "run", "trading_dashboard_main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**docker-compose.yml**:
```yaml
version: '3.8'

services:
  trading-dashboard:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./results:/app/results
      - ./logs:/app/logs
      - ./trading_data.db:/app/trading_data.db
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped

  trading-bot:
    build: .
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./results:/app/results
      - ./logs:/app/logs
      - ./trading_data.db:/app/trading_data.db
    environment:
      - PYTHONUNBUFFERED=1
    command: python monitoring_integration.py
    restart: unless-stopped
    depends_on:
      - trading-dashboard
```

### 3.2 Systemd Service Files (Linux)
**Status**: Does not exist
**Impact**: Production deployment on Linux servers

**File**: `systemd/trading-dashboard.service`
```ini
[Unit]
Description=Trading Dashboard (Streamlit)
After=network.target

[Service]
Type=simple
User=trader
WorkingDirectory=/opt/trading_system
ExecStart=/opt/trading_system/venv/bin/streamlit run trading_dashboard_main.py --server.port=8501
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

**File**: `systemd/trading-bot.service`
```ini
[Unit]
Description=Trading Bot with Health Monitoring
After=network.target

[Service]
Type=simple
User=trader
WorkingDirectory=/opt/trading_system
ExecStart=/opt/trading_system/venv/bin/python monitoring_integration.py
Restart=always
RestartSec=30
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### 3.3 Deployment Scripts

**File**: `deploy/install.sh`
```bash
#!/bin/bash
# Production installation script

set -e

echo "🚀 Installing AI S&P 500 trading System"

# Check Python version
python3 --version || { echo "Python 3.8+ required"; exit 1; }

# Create installation directory
INSTALL_DIR="/opt/trading_system"
sudo mkdir -p $INSTALL_DIR
sudo chown $USER:$USER $INSTALL_DIR

# Copy files
echo "📦 Copying application files..."
cp *.py $INSTALL_DIR/
cp requirements.txt $INSTALL_DIR/
cp -r data models results logs $INSTALL_DIR/ 2>/dev/null || true

# Create virtual environment
echo "🐍 Creating virtual environment..."
cd $INSTALL_DIR
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create config from template
if [ ! -f config.py ]; then
    echo "⚙️  Creating config file..."
    echo "IMPORTANT: Edit config.py with your Oanda credentials!"
    cp config_template.py config.py
fi

# Set up systemd services
echo "🔧 Setting up systemd services..."
sudo cp systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload

echo "✅ Installation complete!"
echo ""
echo "Next steps:"
echo "1. Edit $INSTALL_DIR/config.py with your Oanda credentials"
echo "2. sudo systemctl enable trading-dashboard trading-bot"
echo "3. sudo systemctl start trading-dashboard trading-bot"
echo "4. Access dashboard at http://localhost:8501"
```

**File**: `deploy/backup.sh`
```bash
#!/bin/bash
# Backup script for database and models

BACKUP_DIR="/opt/trading_system/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup database
echo "💾 Backing up database..."
cp trading_data.db "$BACKUP_DIR/trading_data_$TIMESTAMP.db"

# Backup models
echo "💾 Backing up models..."
tar -czf "$BACKUP_DIR/models_$TIMESTAMP.tar.gz" models/

# Backup results
echo "💾 Backing up results..."
tar -czf "$BACKUP_DIR/results_$TIMESTAMP.tar.gz" results/

# Keep only last 30 days of backups
find $BACKUP_DIR -name "*.db" -mtime +30 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete

echo "✅ Backup complete: $BACKUP_DIR"
```

**Deliverables**:
- [ ] Create Dockerfile and docker-compose.yml
- [ ] Create systemd service files
- [ ] Create installation script
- [ ] Create backup script
- [ ] Create monitoring script
- [ ] Test deployment on clean Ubuntu server

---

## Phase 4: Production Hardening
**Timeline**: 2-3 days
**Priority**: MEDIUM

### 4.1 Comprehensive Logging
**Status**: Basic logging exists, needs enhancement

**File**: `utils/logging_config.py`
```python
"""
Centralized logging configuration for production
"""

import logging
import logging.handlers
from pathlib import Path
from datetime import datetime

def setup_logging(log_dir: str = "logs", level: str = "INFO"):
    """
    Set up comprehensive logging

    Creates separate log files for:
    - All activity (trading.log)
    - Errors only (errors.log)
    - Trades only (trades.log)
    - Health checks (health.log)
    """

    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    root_logger.handlers = []

    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler (INFO and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Rotating file handler - All logs
    all_handler = logging.handlers.RotatingFileHandler(
        log_path / 'trading.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=10
    )
    all_handler.setLevel(logging.DEBUG)
    all_handler.setFormatter(formatter)
    root_logger.addHandler(all_handler)

    # Error file handler
    error_handler = logging.handlers.RotatingFileHandler(
        log_path / 'errors.log',
        maxBytes=5*1024*1024,  # 5MB
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)

    # Trade logger (separate)
    trade_logger = logging.getLogger('trades')
    trade_handler = logging.handlers.RotatingFileHandler(
        log_path / 'trades.log',
        maxBytes=10*1024*1024,
        backupCount=20
    )
    trade_handler.setFormatter(formatter)
    trade_logger.addHandler(trade_handler)

    # Health check logger
    health_logger = logging.getLogger('health')
    health_handler = logging.handlers.RotatingFileHandler(
        log_path / 'health.log',
        maxBytes=5*1024*1024,
        backupCount=10
    )
    health_handler.setFormatter(formatter)
    health_logger.addHandler(health_handler)

    logging.info("Logging configured successfully")
```

### 4.2 Performance Monitoring

**File**: `utils/performance_monitor.py`
```python
"""
System performance monitoring
Track CPU, memory, API latency, database performance
"""

import psutil
import time
from datetime import datetime
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor system performance metrics"""

    def __init__(self):
        self.start_time = time.time()
        self.api_call_times = []
        self.db_query_times = []

    def get_system_metrics(self) -> Dict:
        """Get current system resource usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        return {
            'timestamp': datetime.now(),
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'disk_percent': disk.percent,
            'uptime_hours': (time.time() - self.start_time) / 3600
        }

    def log_api_call(self, duration_ms: float):
        """Log API call duration"""
        self.api_call_times.append(duration_ms)
        if len(self.api_call_times) > 1000:
            self.api_call_times.pop(0)

    def get_api_stats(self) -> Dict:
        """Get API performance statistics"""
        if not self.api_call_times:
            return {}

        import numpy as np
        return {
            'avg_ms': np.mean(self.api_call_times),
            'median_ms': np.median(self.api_call_times),
            'p95_ms': np.percentile(self.api_call_times, 95),
            'max_ms': max(self.api_call_times),
            'count': len(self.api_call_times)
        }

    def check_health(self) -> Tuple[bool, List[str]]:
        """Check if system is healthy"""
        issues = []

        metrics = self.get_system_metrics()

        if metrics['cpu_percent'] > 90:
            issues.append(f"High CPU usage: {metrics['cpu_percent']}%")

        if metrics['memory_percent'] > 90:
            issues.append(f"High memory usage: {metrics['memory_percent']}%")

        if metrics['disk_percent'] > 90:
            issues.append(f"Low disk space: {metrics['disk_percent']}% used")

        api_stats = self.get_api_stats()
        if api_stats and api_stats['p95_ms'] > 5000:
            issues.append(f"Slow API calls: P95={api_stats['p95_ms']:.0f}ms")

        return (len(issues) == 0, issues)
```

### 4.3 Rate Limiting for Oanda API

**File**: Update `oanda_integration.py` with rate limiter

```python
from functools import wraps
import time
from collections import deque

class RateLimiter:
    """Rate limiter for Oanda API calls"""

    def __init__(self, max_calls: int = 100, time_window: int = 60):
        """
        Args:
            max_calls: Maximum calls allowed in time window
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = deque()

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()

            # Remove old calls outside time window
            while self.calls and self.calls[0] < now - self.time_window:
                self.calls.popleft()

            # Check if we can make another call
            if len(self.calls) >= self.max_calls:
                sleep_time = self.time_window - (now - self.calls[0])
                if sleep_time > 0:
                    logger.warning(f"Rate limit reached, sleeping {sleep_time:.1f}s")
                    time.sleep(sleep_time)
                    # Clear old calls after sleep
                    while self.calls and self.calls[0] < time.time() - self.time_window:
                        self.calls.popleft()

            # Record this call
            self.calls.append(time.time())

            # Execute function
            return func(*args, **kwargs)

        return wrapper

# Apply to OandaConnector methods
class OandaConnector:
    # ... existing code ...

    @RateLimiter(max_calls=100, time_window=60)  # 100 calls per minute
    def get_account_summary(self):
        # ... existing implementation ...

    @RateLimiter(max_calls=100, time_window=60)
    def get_current_prices(self, instruments):
        # ... existing implementation ...
```

**Deliverables**:
- [ ] Implement centralized logging configuration
- [ ] Create performance monitoring utilities
- [ ] Add rate limiting to Oanda API calls
- [ ] Create system health check script
- [ ] Set up log rotation and archiving
- [ ] Create performance dashboard (optional)

---

## Phase 5: Documentation & Knowledge Transfer
**Timeline**: 1-2 days
**Priority**: LOW (already well documented)

### 5.1 Update Existing Documentation
- [x] CLAUDE.md (already created)
- [ ] Update README.md with deployment instructions
- [ ] Add troubleshooting section to setup_guide.md
- [ ] Create API_REFERENCE.md for all classes/methods
- [ ] Add architecture diagrams (optional)

### 5.2 Video Tutorials (Optional)
- [ ] System overview (5 min)
- [ ] Running first backtest (10 min)
- [ ] Training ML model (15 min)
- [ ] Setting up paper trading (10 min)
- [ ] Deploying to production (15 min)

---

## Phase 6: Optional Enhancements
**Timeline**: 1-2 weeks
**Priority**: LOW

### 6.1 Multi-Instrument Portfolio
Currently system trades single instrument. Add support for:
- Multiple S&P 500 index simultaneously
- Portfolio-level risk management
- Correlation-based position sizing
- Diversification metrics

### 6.2 Advanced ML Models
- LSTM/GRU for time series
- Transformer models
- Ensemble with deep learning
- AutoML for hyperparameter optimization

### 6.3 Additional Data Sources
- Economic calendar integration
- News sentiment analysis
- Alternative data (COT reports, etc.)
- Order flow data

### 6.4 Backtesting Enhancements
- Multi-timeframe analysis
- Monte Carlo simulation
- Parameter optimization (grid search, genetic algorithms)
- Advanced slippage models

### 6.5 Dashboard Enhancements
- Real-time charts with WebSocket updates
- Mobile app (React Native or Flutter)
- Trade journal with screenshots
- Performance comparison vs benchmarks

---

## Timeline Summary

### Immediate (Week 1)
- Phase 1: Complete missing core features (1-2 days)
- Phase 2.1: Create pytest test suite (2-3 days)

### Short-term (Week 2-3)
- Phase 2.2: Integration testing (1 day)
- Phase 3: Deployment automation (2-3 days)
- Phase 4: Production hardening (2-3 days)

### Medium-term (Month 1-2)
- Paper trading validation (90 days)
- Performance monitoring and optimization
- Bug fixes and improvements

### Long-term (Month 3+)
- Phase 6: Optional enhancements
- Live trading with minimal capital
- Scaling and optimization

---

## Success Criteria

### Phase 1 Complete When:
- [ ] Historical data can be fetched from Oanda
- [ ] Email/Slack alerts working
- [ ] Configuration validated before startup
- [ ] All manual testing passes

### Phase 2 Complete When:
- [ ] 50+ pytest tests passing
- [ ] Code coverage >80%
- [ ] End-to-end integration test passes
- [ ] No critical bugs in test results

### Phase 3 Complete When:
- [ ] Docker deployment works
- [ ] Systemd services work
- [ ] Installation script tested on clean Ubuntu
- [ ] Backup/restore tested

### Phase 4 Complete When:
- [ ] Comprehensive logging implemented
- [ ] Performance monitoring active
- [ ] Rate limiting prevents API bans
- [ ] System health checks working

### Production Ready When:
- [ ] All phases 1-4 complete
- [ ] 90+ days successful paper trading
- [ ] Sharpe ratio >1.5 in paper trading
- [ ] Max drawdown <10% in paper trading
- [ ] All alerts and monitoring tested
- [ ] Documentation complete

---

## Risk Management

### Technical Risks
1. **Oanda API changes**: Monitor API documentation, implement versioning
2. **Data quality issues**: Validate all incoming data, log anomalies
3. **Model degradation**: Continuous monitoring, automated retraining
4. **System failures**: Redundancy, monitoring, automatic recovery

### Financial Risks
1. **Start with practice account**: Never skip paper trading
2. **Position sizing limits**: Enforce maximum 2% risk per trade
3. **Daily loss limits**: Stop trading after 5% daily loss
4. **Diversification**: Don't trade single pair only

### Operational Risks
1. **Internet outage**: VPS with good uptime SLA
2. **Hardware failure**: Cloud deployment preferred
3. **Human error**: Configuration validation, read-only production
4. **Security**: Encrypted credentials, secure API access

---

## Maintenance Schedule

### Daily
- Check dashboard for alerts
- Review overnight performance
- Verify system health

### Weekly
- Backup database and models
- Review strategy performance
- Check logs for errors
- Update market data

### Monthly
- Retrain ML models
- Review risk parameters
- Analyze performance metrics
- Update documentation

### Quarterly
- Full system audit
- Walk-forward validation
- Security review
- Dependency updates

---

## Appendix: Quick Reference

### Essential Commands
```bash
# Run tests
pytest tests/ -v

# Start dashboard
streamlit run trading_dashboard_main.py

# Run backtest
python run_examples.py --example 1

# Train model
python run_examples.py --example 4

# Deploy with Docker
docker-compose up -d

# Check logs
journalctl -u trading-bot -f
```

### Key Files
- `config.py`: Configuration (MUST CREATE)
- `backtesting_engine.py`: Core backtesting
- `ml_training_pipeline.py`: ML training
- `monitoring_integration.py`: Live trading
- `trading_dashboard_main.py`: Dashboard

### Support Resources
- Documentation: `docs/` folder
- Examples: `run_examples.py`
- Quick Reference: `quick_reference.md`
- System Guide: `complete_system_guide.md`

---

**End of Development Plan**
