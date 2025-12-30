"""
Setup Script for Trading System
Installs dependencies and performs initial setup
"""

import subprocess
import sys
from pathlib import Path

# Requirements
REQUIREMENTS = """
# Core Dependencies
pandas>=2.0.0
numpy>=1.24.0
python-dateutil>=2.8.0

# Machine Learning
scikit-learn>=1.3.0
xgboost>=2.0.0

# Data Visualization
plotly>=5.18.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Dashboard
streamlit>=1.30.0

# Oanda API
oandapyV20>=0.7.2

# Utilities
requests>=2.31.0
tqdm>=4.66.0

# Optional: Advanced features
# tensorflow>=2.14.0  # For deep learning models
# pytorch>=2.0.0      # Alternative to TensorFlow
# ta-lib>=0.4.0       # Additional technical indicators
"""


def create_directory_structure():
    """Create necessary directories"""
    print("Creating directory structure...")

    directories = [
        'data',
        'models',
        'results',
        'logs',
        'configs'
    ]

    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  ✓ Created {directory}/")

    print()


def install_requirements():
    """Install required packages"""
    print("Installing dependencies...")
    print("This may take a few minutes...\n")

    # Save requirements to file
    with open('requirements.txt', 'w') as f:
        f.write(REQUIREMENTS)

    # Install using pip
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ])
        print("\n✓ All dependencies installed successfully\n")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error installing dependencies: {e}")
        print("Please install manually using: pip install -r requirements.txt")
        sys.exit(1)


def create_config_template():
    """Create configuration template"""
    print("Creating configuration template...")

    config_content = """# Configuration File
# Copy this to config.py and fill in your details

OANDA_CONFIG = {
    'account_id': 'your-account-id-here',
    'access_token': 'your-access-token-here',
    'environment': 'practice'  # or 'live'
}

# Backtest Configuration
BACKTEST_CONFIG = {
    'initial_capital': 10000.0,
    'commission_pct': 0.0001,  # 1 pip
    'slippage_pct': 0.0001,    # 1 pip
    'position_size_pct': 0.02,  # 2% risk per trade
    'max_positions': 1,
    'use_bid_ask_spread': True,
    'spread_pips': 1.0
}

# Risk Management
RISK_CONFIG = {
    'max_daily_loss_pct': 0.05,  # 5%
    'max_drawdown_pct': 0.20,    # 20%
    'max_position_size_pct': 0.02,  # 2%
    'max_correlation': 0.7
}

# ML Model Configuration
ML_CONFIG = {
    'model_type': 'xgboost',
    'test_size': 0.2,
    'validation_size': 0.2,
    'cross_validation_folds': 5,
    'hyperparameter_tuning': True,
    'feature_selection_threshold': 0.01
}

# Dashboard Configuration
DASHBOARD_CONFIG = {
    'refresh_interval': 60,  # seconds
    'max_trades_display': 100,
    'timezone': 'UTC'
}
"""

    with open('config_template.py', 'w') as f:
        f.write(config_content)

    print("  ✓ Created config_template.py")
    print("  → Copy to config.py and add your Oanda credentials\n")


def create_gitignore():
    """Create .gitignore file"""
    print("Creating .gitignore...")

    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Data & Models
data/*.csv
models/*.pkl
results/*.json
logs/*.log

# Secrets
config.py
.env

# Database
*.db
trading_data.db

# OS
.DS_Store
Thumbs.db

# Jupyter Notebooks
.ipynb_checkpoints/
*.ipynb
"""

    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)

    print("  ✓ Created .gitignore\n")


def create_readme():
    """Create quick start README"""
    print("Creating README...")

    readme_content = """# Forex Trading System with ML

Professional trading system with backtesting, ML models, and live monitoring.

## Quick Start

### 1. Install Dependencies
```bash
python setup.py
```

### 2. Configure
```bash
cp config_template.py config.py
# Edit config.py with your Oanda credentials
```

### 3. Run Quick Example
```bash
python complete_workflow.py --mode quick
```

### 4. Launch Dashboard
```bash
streamlit run trading_dashboard_main.py
```

## System Components

- **backtesting_engine.py**: Professional backtesting with realistic costs
- **feature_engineering.py**: 50+ technical indicators and features
- **ml_training_pipeline.py**: Train and compare ML models
- **strategy_examples.py**: Pre-built trading strategies
- **trading_dashboard_main.py**: Real-time monitoring dashboard
- **oanda_integration.py**: Oanda API connector

## Documentation

See `Complete System Integration Guide` for full documentation.

## Important

⚠️ **Always paper trade for 3+ months before going live**
⚠️ **Never risk more than 2% per trade**
⚠️ **Past performance doesn't guarantee future results**

## License

Educational use only. Trade at your own risk.
"""

    with open('README.md', 'w') as f:
        f.write(readme_content)

    print("  ✓ Created README.md\n")


def verify_installation():
    """Verify all packages are installed correctly"""
    print("Verifying installation...")

    required_packages = [
        'pandas',
        'numpy',
        'sklearn',
        'xgboost',
        'streamlit',
        'plotly',
        'oandapyV20'
    ]

    all_installed = True

    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - NOT FOUND")
            all_installed = False

    print()

    if all_installed:
        print("✓ All packages installed successfully!\n")
    else:
        print("✗ Some packages missing. Run: pip install -r requirements.txt\n")

    return all_installed


def print_next_steps():
    """Print next steps for user"""
    print("=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    print()
    print("Next Steps:")
    print()
    print("1. Configure your Oanda credentials:")
    print("   cp config_template.py config.py")
    print("   # Edit config.py with your API key and account ID")
    print()
    print("2. Run a quick backtest example:")
    print("   python complete_workflow.py --mode quick")
    print()
    print("3. Generate sample data and view dashboard:")
    print("   python sample_data_generator.py")
    print("   streamlit run trading_dashboard_main.py")
    print()
    print("4. Train ML models:")
    print("   python complete_workflow.py --mode full")
    print()
    print("5. Read the documentation:")
    print("   See 'Complete System Integration Guide'")
    print()
    print("=" * 60)
    print()
    print("⚠️  IMPORTANT REMINDERS:")
    print("  • Always start with paper trading (practice account)")
    print("  • Test for minimum 90 days before going live")
    print("  • Never risk more than 2% per trade")
    print("  • Monitor dashboard daily")
    print("  • Set up proper risk management")
    print()
    print("Good luck, and trade responsibly!")
    print()


def main():
    """Main setup function"""
    print()
    print("=" * 60)
    print("FOREX TRADING SYSTEM SETUP")
    print("=" * 60)
    print()

    # Create directory structure
    create_directory_structure()

    # Install dependencies
    install_requirements()

    # Create configuration files
    create_config_template()
    create_gitignore()
    create_readme()

    # Verify installation
    if verify_installation():
        print_next_steps()
    else:
        print("Setup incomplete. Please install missing packages.")
        sys.exit(1)


if __name__ == "__main__":
    main()
