# Docker Installation & Deployment Guide for Windows 11

Complete step-by-step guide to deploy AI Trader 4 on Windows 11.

---

## Step 1: Install Docker Desktop for Windows

### Download Docker Desktop

1. **Go to Docker Desktop download page:**
   - https://www.docker.com/products/docker-desktop

2. **Click "Download for Windows"**
   - File: Docker Desktop Installer.exe
   - Size: ~500MB

3. **Run the installer**
   - Double-click Docker Desktop Installer.exe
   - Click "OK" to accept UAC prompt
   - Installation takes 5-10 minutes

4. **Configuration during installation:**
   - ✅ Enable WSL 2 backend (recommended)
   - ✅ Add Docker to PATH
   - Click "Install"

5. **Restart computer** (required after installation)

### First-Time Setup

1. **Start Docker Desktop**
   - Search for "Docker Desktop" in Windows Start menu
   - Click to launch
   - Wait for Docker Engine to start (~30 seconds)

2. **Accept License Agreement**
   - Docker Desktop will prompt on first run
   - Click "Accept"

3. **Sign in to Docker Hub** (optional)
   - You can skip this step
   - Or create free account at https://hub.docker.com

4. **Verify Installation**
   - Open PowerShell or Windows Terminal
   ```powershell
   docker --version
   docker-compose --version
   ```
   - Should show versions (e.g., Docker version 24.0.x)

---

## Step 2: Verify WSL2 Backend

Your system is already running WSL2 (I can see you're in `/home/ck/aitrader4`), which is perfect!

### Check WSL2 Integration

1. **Open Docker Desktop**
2. **Go to Settings (gear icon)**
3. **Resources → WSL Integration**
4. **Enable integration with your WSL distro** (usually Ubuntu)
5. **Apply & Restart**

### Verify Docker in WSL

```bash
# In your WSL terminal (where you are now)
docker --version
docker-compose --version
docker info
```

All commands should work without errors.

---

## Step 3: Configure Environment

### Create .env File

You're already in the right directory (`/home/ck/aitrader4`), so:

```bash
# Copy example file
cp .env.example .env

# Edit with nano (easier for beginners)
nano .env
```

### Required Configuration

Add your Oanda practice account credentials:

```bash
# CRITICAL: Set these values
OANDA_ACCOUNT_ID=001-001-XXXXXXX-001     # Your practice account ID
OANDA_API_TOKEN=abc123def456...          # Your API token (long string)
OANDA_ENVIRONMENT=practice               # MUST be 'practice'!

# Keep defaults for rest
STRATEGY_NAME=BalancedAdaptive
BASE_CONFIDENCE_THRESHOLD=0.50
ENABLE_REGIME_ADAPTATION=true
INITIAL_CAPITAL=10000.0
POSITION_SIZE_PCT=0.02
MAX_POSITION_VALUE_PCT=0.02
MAX_DAILY_LOSS_PCT=0.03
MAX_DRAWDOWN_PCT=0.15
LOG_LEVEL=INFO
DASHBOARD_PORT=8501
```

### Save and Exit Nano

- Press `Ctrl+X` to exit
- Press `Y` to confirm save
- Press `Enter` to confirm filename

### Verify Configuration

```bash
# Check .env was created
cat .env | grep OANDA_ENVIRONMENT

# Should show: OANDA_ENVIRONMENT=practice
```

---

## Step 4: Deploy with Docker

### Option A: Using Deployment Script (Recommended)

```bash
# Make sure you're in the project directory
cd /home/ck/aitrader4

# Run deployment script
./docker-deploy.sh start
```

The script will:
1. Check Docker is installed ✓
2. Validate .env configuration ✓
3. Create necessary directories ✓
4. Build Docker images (~5 minutes first time)
5. Start containers ✓

### Option B: Manual Docker Compose

If the script has issues:

```bash
# Build images
docker-compose build

# Start containers in background
docker-compose up -d

# Check status
docker-compose ps
```

---

## Step 5: Access Dashboard (Windows)

### Method 1: Localhost (Simplest)

1. **Wait 30 seconds** for containers to fully start
2. **Open browser** (Chrome, Edge, Firefox)
3. **Go to:** http://localhost:8501
4. **Dashboard should load** with AI Trader 4 interface

### Method 2: WSL IP Address (If localhost doesn't work)

```bash
# Get WSL IP address
hostname -I | awk '{print $1}'
```

Then access: http://[WSL_IP]:8501 (e.g., http://172.20.10.2:8501)

### Method 3: Port Forwarding (Alternative)

If neither works, Windows might be blocking WSL ports:

```powershell
# In PowerShell (as Administrator)
netsh interface portproxy add v4tov4 listenport=8501 listenaddress=0.0.0.0 connectport=8501 connectaddress=localhost

# Then access
http://localhost:8501
```

---

## Step 6: Verify Deployment

### Check Containers Are Running

```bash
# In WSL terminal
docker ps

# Should show 2 containers:
# - aitrader4_trading_bot
# - aitrader4_dashboard
```

Expected output:
```
CONTAINER ID   IMAGE                  STATUS         PORTS
abc123def456   aitrader4_aitrader4-bot       Up 2 minutes
def456abc789   aitrader4_aitrader4-dashboard Up 2 minutes   0.0.0.0:8501->8501/tcp
```

### Check Logs

```bash
# View bot logs
docker-compose logs aitrader4-bot

# Should see:
# - "Loading configuration..."
# - "Connecting to Oanda..."
# - "Monitoring market..."
```

### Check Dashboard

1. Open http://localhost:8501 in browser
2. Should see 5 tabs: Overview, Trades, Risk Monitor, Analysis, Settings
3. Overview should show:
   - Account balance: $10,000 (practice account)
   - Total trades: 0 (initially)
   - Strategy: Balanced Adaptive

---

## Troubleshooting Windows-Specific Issues

### Issue 1: Docker Desktop Not Starting

**Symptoms:** "Docker Desktop is not running" error

**Solution:**
1. Open Docker Desktop from Start menu
2. Wait for whale icon in system tray to stabilize
3. Click whale icon → Dashboard → Ensure "Starting" becomes "Running"
4. If stuck, restart Docker Desktop
5. If still stuck, restart Windows

### Issue 2: WSL Integration Not Working

**Symptoms:** `docker: command not found` in WSL

**Solution:**
1. Open Docker Desktop
2. Settings → Resources → WSL Integration
3. Enable for your distro (Ubuntu)
4. Click "Apply & Restart"
5. Restart WSL terminal
6. Try: `docker --version`

### Issue 3: Port 8501 Already in Use

**Symptoms:** "port is already allocated" error

**Solution:**
```bash
# Check what's using port 8501
netstat -ano | findstr :8501

# Kill the process (in PowerShell as Admin)
taskkill /PID [PID_NUMBER] /F

# Or change port in docker-compose.yml
# Edit: ports: - "8502:8501"
# Then access http://localhost:8502
```

### Issue 4: Dashboard Shows "Connection Error"

**Symptoms:** Dashboard loads but shows API connection errors

**Solution:**
1. Verify .env has correct credentials
2. Check Oanda account is practice (not live)
3. Verify API token is valid (regenerate if needed)
4. Restart containers:
```bash
docker-compose restart
```

### Issue 5: Firewall Blocking Docker

**Symptoms:** "Cannot connect to Docker daemon"

**Solution:**
1. Open Windows Security → Firewall & network protection
2. Allow Docker Desktop through firewall
3. Restart Docker Desktop

### Issue 6: Performance Issues

**Symptoms:** Docker running very slow

**Solution:**
1. Allocate more resources to WSL2:
   - Create/edit `C:\Users\YourName\.wslconfig`
   ```ini
   [wsl2]
   memory=4GB
   processors=2
   ```
2. Restart WSL: `wsl --shutdown` (in PowerShell)
3. Restart Docker Desktop

---

## Common Commands (Windows/WSL)

### Managing Containers

```bash
# Start
./docker-deploy.sh start

# Stop
./docker-deploy.sh stop

# Restart
./docker-deploy.sh restart

# View status
./docker-deploy.sh status

# View logs
./docker-deploy.sh logs
```

### Direct Docker Commands

```bash
# See running containers
docker ps

# See all containers (including stopped)
docker ps -a

# View logs (live)
docker-compose logs -f

# Stop containers
docker-compose down

# Restart containers
docker-compose restart

# Remove everything and start fresh
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d
```

### Monitoring

```bash
# Resource usage
docker stats

# Container health
docker inspect aitrader4_trading_bot | grep -A 5 Health

# Logs from last hour
docker-compose logs --since 1h aitrader4-bot
```

---

## Expected Behavior on Windows

### First 5 Minutes
- Docker Desktop starting (~30 seconds)
- Building images (~3-5 minutes first time)
- Starting containers (~30 seconds)
- Dashboard loading (http://localhost:8501)

### After Successful Deployment
- **Docker Desktop:** Whale icon in system tray (green = running)
- **Containers:** 2 running (check with `docker ps`)
- **Dashboard:** Accessible at http://localhost:8501
- **CPU Usage:** 5-10% idle, 50-80% during model training
- **Memory:** ~1GB total (both containers)
- **Logs:** Clean, no errors

### Windows Performance Notes
- WSL2 adds ~10-15% overhead vs native Linux
- Docker Desktop uses ~500MB-1GB RAM when idle
- First build takes longer (~5-10 minutes)
- Subsequent starts are fast (~30 seconds)

---

## Stopping Deployment

### Graceful Shutdown

```bash
# Stop containers (preserves data)
./docker-deploy.sh stop

# Or manually
docker-compose down
```

### Complete Cleanup

```bash
# Stop and remove everything
./docker-deploy.sh clean

# Or manually
docker-compose down -v  # Warning: removes volumes!
```

### Pause for Later

Just stop Docker Desktop:
1. Right-click whale icon in system tray
2. Click "Quit Docker Desktop"
3. Containers will stop but data is preserved
4. Next time: Start Docker Desktop → Containers auto-resume

---

## Next Steps After Successful Deployment

1. ✅ **Verify dashboard works** (http://localhost:8501)
2. ✅ **Check logs are clean** (`docker-compose logs`)
3. ✅ **Monitor for 24 hours** (should see "Monitoring market..." every minute)
4. ✅ **Check for first trade** (may take 1-2 days depending on market)
5. ✅ **Review performance weekly** (compare to validation: +1.72% avg return)

---

## FAQ - Windows Specific

**Q: Can I use PowerShell instead of WSL?**
A: Yes, but WSL is recommended. The deploy script uses bash, so in PowerShell you'd need to run `docker-compose` commands directly.

**Q: Do I need to keep Docker Desktop open?**
A: Yes, Docker Desktop must be running for containers to work. It runs in the background (system tray).

**Q: Can I close WSL terminal after deployment?**
A: Yes! Containers run independently. You can close the terminal and they'll keep running.

**Q: How much disk space does this use?**
A: ~2-3GB total (Docker Desktop + images + data). Make sure you have 5GB free.

**Q: Will this slow down my computer?**
A: Minimal impact when idle (<5% CPU, ~1GB RAM). Training spikes CPU to 50-80% for a few minutes occasionally.

**Q: Can I run this on battery (laptop)?**
A: Yes, but for 24/7 operation, keep laptop plugged in or deploy to VPS.

---

## Helpful Windows Commands

### Check if Docker is Running

```powershell
# In PowerShell
docker info

# Or check Docker Desktop system tray icon
```

### Open Dashboard from PowerShell

```powershell
# Open browser to dashboard
Start-Process "http://localhost:8501"
```

### View Files from Windows Explorer

```powershell
# Open WSL home in Windows Explorer
explorer.exe \\wsl$\Ubuntu\home\ck\aitrader4

# Or from WSL
cd /home/ck/aitrader4
explorer.exe .
```

### Edit Files in Windows

You can edit `.env` and other files using Windows Notepad:
1. Open File Explorer
2. Navigate to: `\\wsl$\Ubuntu\home\ck\aitrader4`
3. Right-click `.env` → Open with Notepad
4. Save changes
5. Restart containers: `docker-compose restart`

---

## Summary Checklist

Before starting, ensure:
- [ ] Docker Desktop installed on Windows 11
- [ ] Docker Desktop is running (whale icon in system tray)
- [ ] WSL2 integration enabled in Docker settings
- [ ] In WSL terminal: `cd /home/ck/aitrader4`
- [ ] `.env` file created with practice credentials
- [ ] `OANDA_ENVIRONMENT=practice` in .env

Then deploy:
```bash
./docker-deploy.sh start
```

Access dashboard:
- http://localhost:8501

---

**Windows-Specific Help:** See Docker Desktop documentation
**General Help:** See DOCKER_DEPLOYMENT.md
**Quick Start:** See QUICKSTART_DOCKER.md
