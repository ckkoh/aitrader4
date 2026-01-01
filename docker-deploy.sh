#!/bin/bash
# AI Trader 4 - Docker Deployment Script
# Simplifies Docker deployment with safety checks

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Functions
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "ℹ️  $1"
}

# Check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed!"
        echo "Install from: https://docs.docker.com/get-docker/"
        exit 1
    fi
    print_success "Docker installed"

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed!"
        echo "Install from: https://docs.docker.com/compose/install/"
        exit 1
    fi
    print_success "Docker Compose installed"

    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running!"
        echo "Start Docker and try again"
        exit 1
    fi
    print_success "Docker daemon running"
}

# Check configuration
check_configuration() {
    print_info "Checking configuration..."

    # Check for .env file
    if [ ! -f ".env" ]; then
        print_warning ".env file not found!"
        echo ""
        echo "Creating .env from .env.example..."

        if [ -f ".env.example" ]; then
            cp .env.example .env
            print_warning "Please edit .env and add your Oanda credentials:"
            echo "  1. OANDA_ACCOUNT_ID"
            echo "  2. OANDA_API_TOKEN"
            echo "  3. Verify OANDA_ENVIRONMENT=practice"
            echo ""
            read -p "Press Enter after editing .env file..."
        else
            print_error ".env.example not found! Cannot create .env"
            exit 1
        fi
    fi

    # Check for config.py
    if [ ! -f "config.py" ]; then
        print_warning "config.py not found!"
        echo ""
        echo "Creating config.py template..."
        cat > config.py <<'EOF'
# AI Trader 4 - Configuration File
# This file is mounted into Docker container

import os

OANDA_CONFIG = {
    'account_id': os.getenv('OANDA_ACCOUNT_ID', 'your_account_id'),
    'api_token': os.getenv('OANDA_API_TOKEN', 'your_api_token'),
    'environment': os.getenv('OANDA_ENVIRONMENT', 'practice')
}

# Strategy settings
STRATEGY_CONFIG = {
    'name': os.getenv('STRATEGY_NAME', 'BalancedAdaptive'),
    'base_confidence_threshold': float(os.getenv('BASE_CONFIDENCE_THRESHOLD', '0.50')),
    'enable_regime_adaptation': os.getenv('ENABLE_REGIME_ADAPTATION', 'true').lower() == 'true'
}

# Risk management
RISK_CONFIG = {
    'initial_capital': float(os.getenv('INITIAL_CAPITAL', '10000.0')),
    'position_size_pct': float(os.getenv('POSITION_SIZE_PCT', '0.02')),
    'max_position_value_pct': float(os.getenv('MAX_POSITION_VALUE_PCT', '0.02')),
    'max_daily_loss_pct': float(os.getenv('MAX_DAILY_LOSS_PCT', '0.03')),
    'max_drawdown_pct': float(os.getenv('MAX_DRAWDOWN_PCT', '0.15'))
}
EOF
        print_success "config.py created (reads from environment variables)"
    fi

    # Verify .env has credentials
    if grep -q "your_oanda_api_token_here" .env 2>/dev/null; then
        print_error ".env still contains placeholder values!"
        echo "Please edit .env and add your real Oanda credentials"
        exit 1
    fi

    print_success "Configuration files ready"
}

# Create necessary directories
create_directories() {
    print_info "Creating necessary directories..."

    mkdir -p logs
    mkdir -p data
    mkdir -p balanced_model_results
    mkdir -p ensemble_results

    print_success "Directories created"
}

# Build Docker images
build_images() {
    print_info "Building Docker images..."
    echo ""

    docker-compose build

    print_success "Docker images built successfully"
}

# Start services
start_services() {
    print_info "Starting AI Trader 4 services..."
    echo ""

    docker-compose up -d

    print_success "Services started!"
    echo ""
    print_info "Container status:"
    docker-compose ps

    echo ""
    print_info "Dashboard will be available at:"
    echo "  http://localhost:8501"
    echo ""
    print_info "View logs:"
    echo "  docker-compose logs -f aitrader4-bot"
    echo "  docker-compose logs -f aitrader4-dashboard"
}

# Stop services
stop_services() {
    print_info "Stopping AI Trader 4 services..."

    docker-compose down

    print_success "Services stopped"
}

# Show status
show_status() {
    print_info "AI Trader 4 Status:"
    echo ""

    docker-compose ps

    echo ""
    print_info "Recent logs (bot):"
    docker-compose logs --tail=20 aitrader4-bot || true

    echo ""
    print_info "Recent logs (dashboard):"
    docker-compose logs --tail=20 aitrader4-dashboard || true
}

# Show logs
show_logs() {
    service="${1:-aitrader4-bot}"
    print_info "Showing logs for $service (Ctrl+C to exit)..."

    docker-compose logs -f "$service"
}

# Restart services
restart_services() {
    print_info "Restarting services..."

    docker-compose restart

    print_success "Services restarted"
}

# Update deployment
update_deployment() {
    print_info "Updating deployment..."

    # Pull latest code
    if [ -d ".git" ]; then
        print_info "Pulling latest code from git..."
        git pull
    fi

    # Rebuild images
    print_info "Rebuilding Docker images..."
    docker-compose build --no-cache

    # Restart services
    print_info "Restarting services..."
    docker-compose up -d

    print_success "Deployment updated successfully"
}

# Main menu
show_menu() {
    echo ""
    echo "╔════════════════════════════════════════╗"
    echo "║   AI Trader 4 - Docker Deployment     ║"
    echo "╚════════════════════════════════════════╝"
    echo ""
    echo "Commands:"
    echo "  start     - Start trading bot and dashboard"
    echo "  stop      - Stop all services"
    echo "  restart   - Restart all services"
    echo "  status    - Show service status and logs"
    echo "  logs      - Follow logs (bot/dashboard)"
    echo "  update    - Update code and rebuild"
    echo "  rebuild   - Rebuild Docker images"
    echo "  clean     - Stop and remove containers"
    echo ""
}

# Parse command
case "${1:-}" in
    start)
        check_prerequisites
        check_configuration
        create_directories
        build_images
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs "${2:-aitrader4-bot}"
        ;;
    update)
        update_deployment
        ;;
    rebuild)
        build_images
        restart_services
        ;;
    clean)
        print_warning "This will stop and remove all containers"
        read -p "Are you sure? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker-compose down -v
            print_success "Containers removed"
        fi
        ;;
    *)
        show_menu
        exit 1
        ;;
esac

echo ""
print_success "Done!"
