# Docker Deployment Guide - AI Trader 4

Complete guide for deploying the Balanced Adaptive ML trading strategy using Docker for paper trading.

---

## Quick Start (5 Minutes)

```bash
# 1. Clone repository
git clone https://github.com/ckkoh/aitrader4.git
cd aitrader4

# 2. Configure credentials
cp .env.example .env
vim .env  # Add your Oanda practice account credentials

# 3. Deploy!
./docker-deploy.sh start

# 4. Access dashboard
open http://localhost:8501
```

**That's it!** Your trading bot is now running in Docker.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Configuration](#configuration)
3. [Deployment](#deployment)
4. [Monitoring](#monitoring)
5. [Troubleshooting](#troubleshooting)
6. [Production Deployment](#production-deployment)
7. [FAQ](#faq)

---

## Prerequisites

### 1. Install Docker

**Linux:**
```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker --version
docker-compose --version
```

**macOS:**
```bash
# Install Docker Desktop
brew install --cask docker
# Start Docker Desktop from Applications

# Verify
docker --version
docker-compose --version
```

**Windows:**
```powershell
# Install Docker Desktop from https://www.docker.com/products/docker-desktop
# Start Docker Desktop

# Verify (in PowerShell)
docker --version
docker-compose --version
```

### 2. Get Oanda Practice Account

1. Sign up at https://www.oanda.com/
2. Create a **Practice Account** (free, unlimited funds)
3. Generate API token: Account → Manage API Access → Generate Token
4. Note your Account ID (format: 001-001-1234567-001)

**Important:** Use PRACTICE account for testing! Never start with live money.

---

## Configuration

### Step 1: Clone Repository

```bash
git clone https://github.com/ckkoh/aitrader4.git
cd aitrader4
```

### Step 2: Configure Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit with your Oanda credentials
vim .env  # or nano .env, or any text editor
```

**Required settings in .env:**
```bash
# CRITICAL: Set these values
OANDA_ACCOUNT_ID=001-001-1234567-001     # Your practice account ID
OANDA_API_TOKEN=abc123def456...          # Your API token
OANDA_ENVIRONMENT=practice               # MUST be 'practice' for paper trading

# Strategy (recommended defaults)
STRATEGY_NAME=BalancedAdaptive
BASE_CONFIDENCE_THRESHOLD=0.50
ENABLE_REGIME_ADAPTATION=true

# Risk Management (conservative defaults for paper trading)
INITIAL_CAPITAL=10000.0
POSITION_SIZE_PCT=0.02
MAX_POSITION_VALUE_PCT=0.02
MAX_DAILY_LOSS_PCT=0.03
MAX_DRAWDOWN_PCT=0.15
```

### Step 3: Verify Configuration

```bash
# Check .env file
cat .env | grep -E "OANDA_ACCOUNT_ID|OANDA_API_TOKEN|OANDA_ENVIRONMENT"

# Make sure OANDA_ENVIRONMENT=practice !!
```

---

## Deployment

### Option 1: Use Deployment Script (Recommended)

```bash
# Start everything
./docker-deploy.sh start

# View status
./docker-deploy.sh status

# Follow logs
./docker-deploy.sh logs

# Stop everything
./docker-deploy.sh stop
```

### Option 2: Manual Docker Compose

```bash
# Build images
docker-compose build

# Start services (detached)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### What Gets Deployed?

The deployment creates **2 containers**:

1. **aitrader4-bot** - Trading bot
   - Monitors market every minute
   - Generates ML predictions
   - Places/manages trades via Oanda API
   - Logs all activity

2. **aitrader4-dashboard** - Streamlit dashboard
   - Real-time performance monitoring
   - Trade history visualization
   - Risk metrics
   - Accessible at http://localhost:8501

---

## Monitoring

### Access Dashboard

```bash
# Local deployment
open http://localhost:8501

# Remote server
ssh -L 8501:localhost:8501 user@server
# Then access http://localhost:8501 on your local machine
```

### View Logs

```bash
# Follow bot logs
docker-compose logs -f aitrader4-bot

# Follow dashboard logs
docker-compose logs -f aitrader4-dashboard

# View last 100 lines
docker-compose logs --tail=100 aitrader4-bot

# View logs from specific time
docker-compose logs --since 2024-01-01T10:00:00 aitrader4-bot
```

### Check Container Status

```bash
# Status of all containers
docker-compose ps

# Detailed resource usage
docker stats

# Health check status
docker inspect aitrader4_trading_bot | grep -A 5 Health
```

### Monitor Performance

**Key metrics to watch:**

1. **Win Rate** (target: >50%)
   - Dashboard → Overview tab
   - Check daily, weekly, monthly

2. **Total Return** (target: positive)
   - Dashboard → Performance chart
   - Should match backtest expectations (~+1.7%)

3. **Max Drawdown** (target: <15%)
   - Dashboard → Risk Monitor
   - Alert if exceeds 15%

4. **Trade Frequency** (expected: ~6 trades/month)
   - Dashboard → Trades tab
   - Compare to 15-period validation (94 trades / 15 = 6.3/month)

5. **System Health**
   - Check logs for errors
   - Verify container is running: `docker ps`

---

## Troubleshooting

### Container Won't Start

**Symptom:** `docker-compose up` fails

**Solution:**
```bash
# Check Docker daemon
docker info

# Check logs for errors
docker-compose logs

# Rebuild from scratch
docker-compose down -v
docker-compose build --no-cache
docker-compose up
```

### "Invalid Credentials" Error

**Symptom:** Logs show Oanda API authentication failed

**Solution:**
```bash
# Verify credentials in .env
cat .env | grep OANDA

# Make sure:
# 1. Account ID is correct (001-001-XXXXXXX-001)
# 2. API token is valid (generate new one if needed)
# 3. Environment is 'practice' not 'live'

# Restart after fixing
docker-compose restart
```

### Dashboard Not Accessible

**Symptom:** http://localhost:8501 doesn't load

**Solution:**
```bash
# Check if container is running
docker ps | grep dashboard

# Check dashboard logs
docker-compose logs aitrader4-dashboard

# Check port mapping
docker port aitrader4_dashboard

# Try rebuilding
docker-compose restart aitrader4-dashboard
```

### No Trades Being Placed

**Symptom:** Bot running but no trades in 24+ hours

**Solution:**
```bash
# Check bot logs
docker-compose logs aitrader4-bot | grep -i "signal\|trade"

# Verify:
# 1. Market is open (trading hours: 9:30 AM - 4 PM ET)
# 2. Confidence threshold not too high (check .env: BASE_CONFIDENCE_THRESHOLD=0.50)
# 3. No errors in logs
# 4. Oanda account has sufficient margin

# Test with lower confidence (TEMPORARY)
# Edit .env: BASE_CONFIDENCE_THRESHOLD=0.45
docker-compose restart aitrader4-bot
```

### High Memory Usage

**Symptom:** Container using >2GB RAM

**Solution:**
```bash
# Check current usage
docker stats aitrader4_trading_bot

# Set memory limit in docker-compose.yml:
services:
  aitrader4-bot:
    mem_limit: 1g
    memswap_limit: 1g

# Restart
docker-compose up -d
```

### Database Locked Error

**Symptom:** "database is locked" in logs

**Solution:**
```bash
# Stop both containers
docker-compose down

# Check if database file is accessible
ls -la trading_data.db

# Restart
docker-compose up -d

# If persists, recreate database
rm trading_data.db
docker-compose up -d
```

---

## Production Deployment

### Remote Server Deployment

**1. Copy files to server:**
```bash
# From local machine
scp -r aitrader4/ user@server:/home/user/

# Or clone directly on server
ssh user@server
git clone https://github.com/ckkoh/aitrader4.git
cd aitrader4
```

**2. Configure environment:**
```bash
# On server
cp .env.example .env
vim .env  # Add credentials

# Verify
cat .env | grep OANDA_ENVIRONMENT
# Should be: OANDA_ENVIRONMENT=practice
```

**3. Deploy:**
```bash
./docker-deploy.sh start
```

**4. Access dashboard remotely:**
```bash
# From local machine (creates SSH tunnel)
ssh -L 8501:localhost:8501 user@server

# Access http://localhost:8501 in browser
```

### Auto-Start on System Reboot

**Add to crontab:**
```bash
crontab -e

# Add this line
@reboot cd /home/user/aitrader4 && docker-compose up -d
```

**Or use systemd:**
```bash
# Create service file
sudo vim /etc/systemd/system/aitrader4.service

# Content:
[Unit]
Description=AI Trader 4 Docker Containers
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/user/aitrader4
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target

# Enable
sudo systemctl enable aitrader4
sudo systemctl start aitrader4
```

### Security Hardening

**1. Use read-only config:**
```yaml
# In docker-compose.yml
volumes:
  - ./config.py:/app/config.py:ro  # Read-only!
```

**2. Limit resources:**
```yaml
# In docker-compose.yml
services:
  aitrader4-bot:
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
```

**3. Network isolation:**
```yaml
# Only dashboard needs external access
services:
  aitrader4-bot:
    networks:
      - internal
  aitrader4-dashboard:
    networks:
      - internal
    ports:
      - "127.0.0.1:8501:8501"  # Only localhost can access
```

**4. Regular backups:**
```bash
# Backup database daily
0 0 * * * cd /home/user/aitrader4 && cp trading_data.db backups/trading_data_$(date +\%Y\%m\%d).db

# Backup logs weekly
0 0 * * 0 cd /home/user/aitrader4 && tar -czf backups/logs_$(date +\%Y\%m\%d).tar.gz logs/
```

---

## FAQ

### Can I run multiple strategies simultaneously?

Yes! Use separate directories:

```bash
# Strategy 1: Balanced Adaptive (S&P 500)
cd ~/aitrader4-balanced
vim .env  # Account 1, SPX500_USD
./docker-deploy.sh start

# Strategy 2: Different model (EUR/USD)
cd ~/aitrader4-forex
vim .env  # Account 2, EUR_USD
vim docker-compose.yml  # Change ports to 8502
./docker-deploy.sh start
```

### How much does it cost to run?

**Oanda Practice Account:** FREE (unlimited virtual funds)
**Server costs:**
- Local (free): $0/month
- VPS (1 CPU, 2GB RAM): $5-10/month
- Cloud (AWS t3.micro): ~$8/month

**Recommended:** Start local, move to VPS after 90 days successful paper trading.

### When should I switch to live trading?

**ONLY after:**
- ✅ 90+ days successful paper trading
- ✅ Performance matches backtest expectations
- ✅ Max drawdown < 10%
- ✅ Win rate > 50% or positive returns
- ✅ No system errors or crashes
- ✅ Comfortable with risk

**Start with:** Minimal capital ($500-$1000 max)

### How do I update the code?

```bash
# Stop containers
docker-compose down

# Update code
git pull

# Rebuild and restart
./docker-deploy.sh update
```

### Can I customize the strategy?

Yes! Edit parameters in `.env`:

```bash
# More conservative
BASE_CONFIDENCE_THRESHOLD=0.60  # Fewer trades, higher confidence
POSITION_SIZE_PCT=0.01          # Smaller positions (1% risk)

# More aggressive (NOT recommended)
BASE_CONFIDENCE_THRESHOLD=0.45  # More trades, lower confidence
POSITION_SIZE_PCT=0.03          # Larger positions (3% risk)
```

### How do I check if it's working?

**Daily checks:**
```bash
# 1. Container status
docker ps

# 2. Recent activity
docker-compose logs --tail=50 aitrader4-bot

# 3. Dashboard
open http://localhost:8501

# 4. Health
docker inspect aitrader4_trading_bot | grep "Health"
```

---

## Support

**Issues:** https://github.com/ckkoh/aitrader4/issues
**Documentation:** See other .md files in repository
**Logs:** Always check `docker-compose logs` first

---

## Quick Reference

```bash
# Start
./docker-deploy.sh start

# Stop
./docker-deploy.sh stop

# Status
./docker-deploy.sh status

# Logs
./docker-deploy.sh logs

# Restart
./docker-deploy.sh restart

# Update
./docker-deploy.sh update

# Clean
./docker-deploy.sh clean
```

**Dashboard:** http://localhost:8501
**Logs location:** `./logs/`
**Database:** `./trading_data.db`

---

**Last Updated:** 2026-01-01
**Recommended Strategy:** Balanced Adaptive (+1.72% avg return in validation)
**Deployment:** Docker (practice account only for first 90 days)
