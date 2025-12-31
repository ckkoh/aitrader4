"""
Complete Trading System Workflow
Demonstrates end-to-end usage: Data -> Features -> ML Training -> Backtesting -> Dashboard
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
from typing import Optional, Dict, List

from backtesting_engine import BacktestEngine, BacktestConfig
from ml_training_pipeline import MLTradingPipeline
from feature_engineering import FeatureEngineering
from strategy_examples import (MomentumStrategy, MeanReversionStrategy,
                               BreakoutStrategy)
from trading_dashboard_main import DatabaseManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradingSystemPipeline:
    """
    Complete pipeline from data to deployment
    """

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path
        self.data = None
        self.backtest_results = {}
        self.ml_pipeline = None
        self.db = DatabaseManager()

        # Create directories
        self.models_dir = Path('models')
        self.results_dir = Path('results')
        self.models_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)

    def load_oanda_data(self, instrument: str = 'SPX500_USD',
                        granularity: str = 'D',
                        days: int = 365) -> pd.DataFrame:
        """
        Load data from Oanda API

        Args:
            instrument: Trading instrument
            granularity: Candle granularity (M5, H1, D, etc.)
            days: Number of days of historical data

        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Loading {days} days of {instrument} data from Oanda...")

        try:
            from oanda_integration import OandaConnector
            from config import OANDA_CONFIG

            OandaConnector(**OANDA_CONFIG)

            # Calculate date range
            end_date = datetime.now()
            _start_date = end_date - timedelta(days=days)  # noqa: F841

            # Fetch data using Oanda API
            # Note: You'd implement the actual fetch method in OandaConnector
            # This is a placeholder showing the structure

            # For now, use simulated data
            logger.warning("Using simulated data. Implement actual Oanda fetch in production.")
            self.data = self._generate_realistic_data(days, instrument)

        except ImportError:
            logger.warning("Oanda integration not available. Using simulated data.")
            self.data = self._generate_realistic_data(days, instrument)

        return self.data

    def _generate_realistic_data(self, days: int, instrument: str) -> pd.DataFrame:
        """Generate realistic price data for testing"""
        logger.info("Generating realistic simulated data...")

        # Generate hourly data
        periods = days * 24
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='1H')

        # Simulate realistic S&P 500 price movements
        np.random.seed(42)

        # Trend component
        trend = np.linspace(0, 0.02, periods)

        # Random walk with mean reversion
        returns = np.random.randn(periods) * 0.0005
        for i in range(1, len(returns)):
            # Mean reversion factor
            returns[i] += -0.1 * returns[i - 1]

        # Price
        base_price = 1.1000 if 'EUR' in instrument else 1.3000
        price = base_price + trend + np.cumsum(returns)

        # OHLC with realistic intrabar movement
        noise = np.random.randn(periods) * 0.0001

        df = pd.DataFrame({
            'open': price + noise,
            'high': price + np.abs(np.random.randn(periods)) * 0.0003,
            'low': price - np.abs(np.random.randn(periods)) * 0.0003,
            'close': price,
            'volume': np.random.randint(5000, 15000, periods)
        }, index=dates)

        # Ensure high is highest and low is lowest
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)

        return df

    def step1_prepare_features(self) -> pd.DataFrame:
        """
        Step 1: Prepare features for ML training
        """
        logger.info("\n" + "=" * 60)
        logger.info("STEP 1: FEATURE ENGINEERING")
        logger.info("=" * 60)

        if self.data is None:
            raise ValueError("No data loaded. Call load_oanda_data first.")

        # Build complete feature set
        df_features = FeatureEngineering.build_complete_feature_set(
            self.data.copy(),
            include_volume=True
        )

        logger.info(f"Original data shape: {self.data.shape}")
        logger.info(f"Features shape: {df_features.shape}")
        logger.info(f"Number of features created: {len(df_features.columns)}")

        return df_features

    def step2_train_ml_models(self, df_features: pd.DataFrame) -> Dict:
        """
        Step 2: Train and compare ML models
        """
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: ML MODEL TRAINING")
        logger.info("=" * 60)

        # Initialize ML pipeline
        self.ml_pipeline = MLTradingPipeline()
        self.ml_pipeline.df_features = df_features

        # Get feature columns
        exclude_cols = ['open', 'high', 'low', 'close', 'volume',
                        'future_return', 'target_class', 'target_binary',
                        'target_regression']
        self.ml_pipeline.feature_cols = [col for col in df_features.columns
                                         if col not in exclude_cols]

        # Compare multiple models
        logger.info("\nComparing models...")
        comparison = self.ml_pipeline.compare_models(
            model_types=['logistic', 'random_forest', 'xgboost'],
            target_col='target_binary'
        )

        # Train best model with hyperparameter tuning
        best_model = comparison.iloc[0]['model']
        logger.info(f"\nTraining best model ({best_model}) with optimization...")

        results = self.ml_pipeline.train_model(
            model_type=best_model,
            target_col='target_binary',
            hyperparameter_tuning=True,
            cross_validation=True
        )

        logger.info(f"\nModel saved: {results['model_filename']}")

        return results

    def step3_backtest_strategies(self, df_features: pd.DataFrame) -> Dict:
        """
        Step 3: Backtest multiple strategies
        """
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: STRATEGY BACKTESTING")
        logger.info("=" * 60)

        # Configure backtest
        config = BacktestConfig(
            initial_capital=10000.0,
            commission_pct=0.0001,
            slippage_pct=0.0001,
            position_size_pct=0.02,
            max_positions=1,
            use_bid_ask_spread=True,
            spread_pips=1.0,
            max_daily_loss_pct=0.05,
            max_drawdown_pct=0.20
        )

        # Initialize backtest engine
        engine = BacktestEngine(config)

        # Test multiple strategies
        strategies = [
            MomentumStrategy(fast_period=20, slow_period=50),
            MomentumStrategy(fast_period=10, slow_period=30),
            MeanReversionStrategy(bb_period=20, bb_std=2.0),
            BreakoutStrategy(lookback_period=20)
        ]

        results = {}

        for strategy in strategies:
            logger.info(f"\nBacktesting: {strategy.name}")

            result = engine.run_backtest(
                strategy=strategy,
                data=self.data,
                start_date=self.data.index[0] + timedelta(days=60),  # Skip initial period
                end_date=self.data.index[-1]
            )

            results[strategy.name] = result

            metrics = result['metrics']
            logger.info(f"  Total Trades: {metrics['total_trades']}")
            logger.info(f"  Win Rate: {metrics['win_rate']:.2%}")
            logger.info(f"  Total P&L: ${metrics['total_pnl']:.2f}")
            logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            logger.info(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")

        # Save results
        self._save_backtest_results(results)

        return results

    def step4_walk_forward_analysis(self, best_strategy_name: str) -> List[Dict]:
        """
        Step 4: Perform walk-forward analysis on best strategy
        """
        logger.info("\n" + "=" * 60)
        logger.info("STEP 4: WALK-FORWARD ANALYSIS")
        logger.info("=" * 60)

        # Get best strategy
        if best_strategy_name == 'Momentum_20_50':
            strategy = MomentumStrategy(20, 50)
        elif best_strategy_name == 'MeanReversion_20':
            strategy = MeanReversionStrategy(20, 2.0)
        else:
            strategy = BreakoutStrategy(20)

        # Configure backtest
        config = BacktestConfig(
            initial_capital=10000.0,
            commission_pct=0.0001,
            slippage_pct=0.0001,
            position_size_pct=0.02,
            use_bid_ask_spread=True
        )

        engine = BacktestEngine(config)

        # Walk-forward analysis
        wf_results = engine.walk_forward_analysis(
            strategy=strategy,
            data=self.data,
            train_period_days=180,
            test_period_days=60,
            step_days=30
        )

        return wf_results

    def step5_deploy_to_dashboard(self, backtest_results: Dict):
        """
        Step 5: Deploy results to monitoring dashboard
        """
        logger.info("\n" + "=" * 60)
        logger.info("STEP 5: DEPLOYING TO DASHBOARD")
        logger.info("=" * 60)

        # Get best performing strategy
        best_strategy = max(
            backtest_results.items(),
            key=lambda x: x[1]['metrics']['sharpe_ratio']
        )

        strategy_name = best_strategy[0]
        strategy_data = best_strategy[1]

        logger.info(f"Best strategy: {strategy_name}")
        logger.info(f"Sharpe Ratio: {strategy_data['metrics']['sharpe_ratio']:.2f}")

        # Add trades to dashboard database
        trades = strategy_data['trades']
        logger.info(f"Adding {len(trades)} trades to dashboard...")

        for trade in trades:
            self.db.add_trade({
                'trade_id': f"BT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}",
                'instrument': trade.instrument,
                'direction': trade.direction,
                'entry_time': trade.entry_time.isoformat(),
                'exit_time': trade.exit_time.isoformat(),
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'size': trade.size,
                'pnl': trade.pnl,
                'pnl_percent': trade.pnl_percent,
                'commission': trade.commission,
                'slippage': trade.slippage,
                'strategy': strategy_name,
                'status': 'closed'
            })

        logger.info("✅ Trades successfully added to dashboard!")
        logger.info("Run 'streamlit run trading_dashboard_main.py' to view results")

    def step6_generate_report(self, backtest_results: Dict) -> str:
        """
        Step 6: Generate comprehensive PDF/HTML report
        """
        logger.info("\n" + "=" * 60)
        logger.info("STEP 6: GENERATING REPORT")
        logger.info("=" * 60)

        # Create markdown report
        report = self._create_markdown_report(backtest_results)

        # Save report
        report_path = self.results_dir / f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_path, 'w') as f:
            f.write(report)

        logger.info(f"Report saved: {report_path}")

        return str(report_path)

    def _save_backtest_results(self, results: Dict):
        """Save backtest results to JSON"""
        # Convert non-serializable objects
        serializable_results = {}

        for strategy_name, result in results.items():
            serializable_results[strategy_name] = {
                'metrics': result['metrics'],
                'num_trades': len(result['trades']),
                'strategy_name': result['strategy_name']
            }

        results_path = self.results_dir / f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Results saved: {results_path}")

    def _create_markdown_report(self, backtest_results: Dict) -> str:
        """Create comprehensive markdown report"""

        report = """# Trading System Backtest Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

"""

        # Find best strategy
        _best_strategy = max(  # noqa: F841
            backtest_results.items(),
            key=lambda x: x[1]['metrics']['sharpe_ratio']
        )

        report += """
**Recommended Strategy:** {best_strategy[0]}
- **Sharpe Ratio:** {best_strategy[1]['metrics']['sharpe_ratio']:.2f}
- **Total Return:** {best_strategy[1]['metrics']['total_return_pct']:.2f}%
- **Max Drawdown:** {best_strategy[1]['metrics']['max_drawdown_pct']:.2f}%
- **Win Rate:** {best_strategy[1]['metrics']['win_rate']:.2%}

---

## Strategy Comparison

| Strategy | Trades | Win Rate | Sharpe | Return % | Max DD % | Profit Factor |
|----------|--------|----------|--------|----------|----------|---------------|
"""

        for name, result in backtest_results.items():
            m = result['metrics']
            report += f"| {name} | {
                m['total_trades']} | {
                m['win_rate']:.1%} | {
                m['sharpe_ratio']:.2f} | {
                m['total_return_pct']:.2f} | {
                    m['max_drawdown_pct']:.2f} | {
                        m['profit_factor']:.2f} |\n"

        report += "\n---\n\n## Detailed Strategy Analysis\n\n"

        for name, result in backtest_results.items():
            m = result['metrics']

            report += """
### {name}

**Performance Metrics:**
- Total Trades: {m['total_trades']}
- Winning Trades: {m['winning_trades']}
- Losing Trades: {m['losing_trades']}
- Win Rate: {m['win_rate']:.2%}

**Returns:**
- Total Return: {m['total_return_pct']:.2f}%
- Annual Return: {m['annual_return_pct']:.2f}%
- Profit Factor: {m['profit_factor']:.2f}
- Expectancy: ${m['expectancy']:.2f}

**Risk Metrics:**
- Sharpe Ratio: {m['sharpe_ratio']:.2f}
- Sortino Ratio: {m['sortino_ratio']:.2f}
- Calmar Ratio: {m['calmar_ratio']:.2f}
- Max Drawdown: {m['max_drawdown_pct']:.2f}%
- Recovery Factor: {m['recovery_factor']:.2f}

**Trade Statistics:**
- Average Win: ${m['avg_win']:.2f}
- Average Loss: ${m['avg_loss']:.2f}
- Avg Trade Duration: {m['avg_trade_duration_hours']:.1f} hours
- Max Consecutive Wins: {m['max_consecutive_wins']}
- Max Consecutive Losses: {m['max_consecutive_losses']}

---
"""

        report += """
## Next Steps

1. ✅ Review backtest results and select best strategy
2. ⚠️ Perform walk-forward analysis to validate robustness
3. 📊 Run on out-of-sample data from different time periods
4. 🧪 Test on paper trading account for 3 months minimum
5. 📈 Monitor performance in dashboard daily
6. 🚀 Consider live deployment only after consistent results

## Risk Warnings

- Past performance does not guarantee future results
- Market conditions change and strategies may stop working
- Always use proper position sizing and risk management
- Never risk more than 2% of capital per trade
- Monitor drawdown limits strictly

---

*Report generated by Trading System Pipeline*
"""

        return report

    def run_complete_pipeline(self):
        """
        Run complete pipeline from start to finish
        """
        logger.info("\n" + "=" * 70)
        logger.info("STARTING COMPLETE TRADING SYSTEM PIPELINE")
        logger.info("=" * 70)

        try:
            # Load data
            self.load_oanda_data(instrument='SPX500_USD', days=365)

            # Step 1: Feature engineering
            df_features = self.step1_prepare_features()

            # Step 2: Train ML models
            ml_results = self.step2_train_ml_models(df_features)

            # Step 3: Backtest strategies
            backtest_results = self.step3_backtest_strategies(df_features)

            # Step 4: Walk-forward analysis (on best strategy)
            best_strategy_name = max(
                backtest_results.items(),
                key=lambda x: x[1]['metrics']['sharpe_ratio']
            )[0]
            wf_results = self.step4_walk_forward_analysis(best_strategy_name)

            # Step 5: Deploy to dashboard
            self.step5_deploy_to_dashboard(backtest_results)

            # Step 6: Generate report
            report_path = self.step6_generate_report(backtest_results)

            logger.info("\n" + "=" * 70)
            logger.info("✅ PIPELINE COMPLETE!")
            logger.info("=" * 70)
            logger.info(f"\nReport saved to: {report_path}")
            logger.info("\nTo view dashboard:")
            logger.info("  streamlit run trading_dashboard_main.py")
            logger.info("\n" + "=" * 70)

            return {
                'ml_results': ml_results,
                'backtest_results': backtest_results,
                'wf_results': wf_results,
                'report_path': report_path
            }

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


def quick_backtest_example():
    """
    Quick example showing how to run a simple backtest
    """
    logger.info("Running quick backtest example...")

    # Generate sample data
    dates = pd.date_range('2023-01-01', periods=5000, freq='1H')
    np.random.seed(42)

    returns = np.random.randn(5000) * 0.0005
    price = 1.1 + np.cumsum(returns)

    df = pd.DataFrame({
        'open': price + np.random.randn(5000) * 0.0001,
        'high': price + np.abs(np.random.randn(5000)) * 0.0002,
        'low': price - np.abs(np.random.randn(5000)) * 0.0002,
        'close': price,
        'volume': np.random.randint(5000, 15000, 5000)
    }, index=dates)

    # Ensure high/low are correct
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    # Create strategy
    strategy = MomentumStrategy(fast_period=20, slow_period=50)

    # Configure backtest
    config = BacktestConfig(
        initial_capital=10000,
        commission_pct=0.0001,
        slippage_pct=0.0001,
        position_size_pct=0.02
    )

    # Run backtest
    engine = BacktestEngine(config)
    result = engine.run_backtest(strategy, df)

    # Print results
    metrics = result['metrics']
    logger.info("\n" + "=" * 50)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 50)
    logger.info(f"Strategy: {strategy.name}")
    logger.info(f"Period: {df.index[0]} to {df.index[-1]}")
    logger.info(f"Total Trades: {metrics['total_trades']}")
    logger.info(f"Win Rate: {metrics['win_rate']:.2%}")
    logger.info(f"Total Return: {metrics['total_return_pct']:.2f}%")
    logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    logger.info(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    logger.info(f"Profit Factor: {metrics['profit_factor']:.2f}")
    logger.info("=" * 50)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Trading System Pipeline')
    parser.add_argument('--mode', choices=['full', 'quick'], default='quick',
                        help='Run mode: full pipeline or quick example')
    parser.add_argument('--days', type=int, default=365,
                        help='Days of historical data to load')

    args = parser.parse_args()

    if args.mode == 'full':
        # Run complete pipeline
        pipeline = TradingSystemPipeline()
        results = pipeline.run_complete_pipeline()

    else:
        # Run quick example
        quick_backtest_example()
