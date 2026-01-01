# Quick Start - Docker Deployment

Deploy AI Trader 4 (Balanced Adaptive Strategy) in 5 minutes using Docker.

---

## Prerequisites

- Docker installed (https://docs.docker.com/get-docker/)
- Oanda practice account (https://www.oanda.com/)

---

## 5-Minute Setup

### 1. Clone Repository

```bash
git clone https://github.com/ckkoh/aitrader4.git
cd aitrader4
```

### 2. Configure Credentials

```bash
# Copy environment template
cp .env.example .env

# Edit and add your Oanda practice credentials
nano .env  # or vim .env
```

**Required in .env:**
```bash
OANDA_ACCOUNT_ID=001-001-XXXXXXX-001  # Your practice account ID
OANDA_API_TOKEN=your_token_here       # Your API token
OANDA_ENVIRONMENT=practice            # MUST be 'practice'!
```

### 3. Deploy

```bash
# Start everything
./docker-deploy.sh start
```

### 4. Access Dashboard

Open in browser: **http://localhost:8501**

---

## That's It!

Your trading bot is now:
- ✅ Running in Docker containers
- ✅ Paper trading on Oanda practice account
- ✅ Monitoring S&P 500 (SPX500_USD)
- ✅ Using Balanced Adaptive strategy (+1.72% avg return)

---

## Quick Commands

```bash
# View logs
./docker-deploy.sh logs

# Check status
./docker-deploy.sh status

# Stop
./docker-deploy.sh stop

# Restart
./docker-deploy.sh restart
```

---

## Next Steps

1. **Monitor for 24 hours** - Verify bot is working correctly
2. **Check dashboard daily** - Watch performance metrics
3. **Review logs weekly** - Look for errors or issues
4. **Paper trade 90+ days** - Validate strategy before live trading

---

## Expected Performance

Based on 15-period walk-forward validation:

- **Average Return:** +1.72% per 3-month period
- **Win Rate:** 39.6%
- **Trade Frequency:** ~6 trades/month
- **Max Drawdown:** <15%

---

## Need Help?

- **Full Documentation:** See [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md)
- **Troubleshooting:** Check Docker logs: `docker-compose logs`
- **Issues:** https://github.com/ckkoh/aitrader4/issues

---

## Safety Reminders

⚠️ **CRITICAL:**
- Use **practice account** for first 90 days minimum
- Never deploy to **live account** without extensive paper trading
- Start with **minimal capital** ($500-1000 max) when going live
- Monitor **daily** for first month

---

**Happy Paper Trading! 📈**
