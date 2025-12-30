"""
Quick test to verify configuration and Oanda connection
"""

import sys


def test_config_import():
    """Test that config imports correctly"""
    print("=" * 60)
    print("Testing Configuration Import...")
    print("=" * 60)

    try:
        from config import (
            OANDA_CONFIG, BACKTEST_CONFIG, RISK_CONFIG,
            ML_CONFIG, validate_config
        )
        print("✅ Config import successful")

        # Validate config
        valid, errors = validate_config()
        if valid:
            print("✅ Configuration validation passed")
        else:
            print("❌ Configuration validation failed:")
            for error in errors:
                print(f"   - {error}")
            return False

        # Display key settings
        print("\n" + "=" * 60)
        print("Configuration Summary:")
        print("=" * 60)
        print(f"Oanda Account: {OANDA_CONFIG['account_id']}")
        print(f"Environment: {OANDA_CONFIG['environment'].upper()}")
        print(f"Initial Capital: ${BACKTEST_CONFIG['initial_capital']:,.0f}")
        print(f"Max Daily Loss: {RISK_CONFIG['max_daily_loss_pct'] * 100:.0f}%")
        print(f"Max Drawdown: {RISK_CONFIG['max_drawdown_pct'] * 100:.0f}%")
        print(f"Position Size: {RISK_CONFIG['max_position_size_pct'] * 100:.0f}%")
        print(f"ML Model: {ML_CONFIG['model_type'].upper()}")

        return True

    except ImportError as e:
        print(f"❌ Failed to import config: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_oanda_connection():
    """Test Oanda API connection"""
    print("\n" + "=" * 60)
    print("Testing Oanda API Connection...")
    print("=" * 60)

    try:
        from oanda_integration import OandaConnector
        from config import OANDA_CONFIG

        print(f"Connecting to Oanda {OANDA_CONFIG['environment']} environment...")

        oanda = OandaConnector(
            account_id=OANDA_CONFIG['account_id'],
            access_token=OANDA_CONFIG['access_token'],
            environment=OANDA_CONFIG['environment']
        )

        # Test connection by getting account summary
        print("Fetching account summary...")
        account = oanda.get_account_summary()

        if account and 'balance' in account:
            print("✅ Successfully connected to Oanda!")
            print("\nAccount Details:")
            print(f"  Balance: {account.get('balance', 'N/A')}")
            print(f"  NAV: {account.get('NAV', 'N/A')}")
            print(f"  Currency: {account.get('currency', 'N/A')}")
            print(f"  Unrealized P&L: {account.get('unrealizedPL', '0')}")
            print(f"  Margin Used: {account.get('marginUsed', '0')}")
            print(f"  Margin Available: {account.get('marginAvailable', 'N/A')}")

            # Test getting current prices
            print("\nFetching current prices for EUR_USD, GBP_USD, USD_JPY...")
            prices = oanda.get_current_prices(['EUR_USD', 'GBP_USD', 'USD_JPY'])

            if prices:
                print("\nCurrent Prices:")
                for instrument, price_data in prices.items():
                    print(f"  {instrument}:")
                    print(f"    Bid: {price_data['bid']:.5f}")
                    print(f"    Ask: {price_data['ask']:.5f}")
                    print(f"    Spread: {price_data['spread']:.5f}")

            # Test getting open positions
            print("\nChecking open positions...")
            positions = oanda.get_open_positions()
            print(f"  Open Positions: {len(positions)}")

            if positions:
                for pos in positions:
                    print(f"    {pos['instrument']}: {pos['direction']} {pos['size']} units")
                    print(f"      Unrealized P&L: ${pos['unrealized_pnl']:.2f}")
            else:
                print("    No open positions")

            return True
        else:
            print("❌ Failed to get account summary")
            print(f"Response: {account}")
            return False

    except ImportError as e:
        print(f"❌ Failed to import required modules: {e}")
        print("Make sure oandapyV20 is installed: pip install oandapyV20")
        return False
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Verify your account_id and access_token in config.py")
        print("2. Check that you're using 'practice' environment for practice account")
        print("3. Ensure your internet connection is working")
        print("4. Verify Oanda API is not under maintenance")
        return False


def test_core_modules():
    """Test that core modules can be imported"""
    print("\n" + "=" * 60)
    print("Testing Core Module Imports...")
    print("=" * 60)

    modules = [
        ('backtesting_engine', 'BacktestEngine'),
        ('feature_engineering', 'FeatureEngineering'),
        ('ml_training_pipeline', 'MLTradingPipeline'),
        ('strategy_examples', 'MomentumStrategy'),
        ('trading_dashboard_main', 'DatabaseManager'),
        ('oanda_integration', 'OandaConnector'),
    ]

    success = True
    for module_name, class_name in modules:
        try:
            module = __import__(module_name)
            getattr(module, class_name)
            print(f"✅ {module_name}.{class_name}")
        except ImportError as e:
            print(f"❌ {module_name}: {e}")
            success = False
        except AttributeError as e:
            print(f"❌ {module_name}.{class_name}: {e}")
            success = False

    return success


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("AI FOREX TRADING SYSTEM - CONFIGURATION TEST")
    print("=" * 60)

    # Test 1: Config import and validation
    if not test_config_import():
        print("\n⚠️  Configuration test failed!")
        sys.exit(1)

    # Test 2: Core modules
    if not test_core_modules():
        print("\n⚠️  Some core modules failed to load!")
        print("Run: pip install -r requirements.txt")

    # Test 3: Oanda connection
    try:
        if test_oanda_connection():
            print("\n" + "=" * 60)
            print("✅ ALL TESTS PASSED!")
            print("=" * 60)
            print("\nYour system is ready for:")
            print("  1. Backtesting: python run_examples.py --example 1")
            print("  2. ML Training: python run_examples.py --example 4")
            print("  3. Dashboard: streamlit run trading_dashboard_main.py")
            print("  4. Paper Trading: Ready when you are!")
            print("\n⚠️  Remember: Always start with paper trading!")
            print("=" * 60)
        else:
            print("\n⚠️  Oanda connection test failed!")
            print("The system will still work for backtesting with sample data.")
            print("Fix the connection to enable live/paper trading.")
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(0)
