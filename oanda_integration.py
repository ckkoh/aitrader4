"""
Oanda API Integration for Trading Dashboard
Handles live trading data, position monitoring, and trade execution
"""

import oandapyV20
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.instruments as instruments
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Optional
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OandaConnector:
    """
    Connector for Oanda API v20
    Handles authentication, data retrieval, and order management
    """

    def __init__(self, account_id: str, access_token: str, environment: str = "practice"):
        """
        Initialize Oanda connection

        Args:
            account_id: Your Oanda account ID
            access_token: Your Oanda API access token
            environment: 'practice' or 'live'
        """
        self.account_id = account_id
        self.access_token = access_token

        if environment == "practice":
            self.client = oandapyV20.API(
                access_token=access_token,
                environment="practice"
            )
        else:
            self.client = oandapyV20.API(
                access_token=access_token,
                environment="live"
            )

        logger.info(f"Connected to Oanda {environment} environment")

    def get_account_summary(self) -> Dict:
        """Get account summary information"""
        try:
            r = accounts.AccountSummary(accountID=self.account_id)
            response = self.client.request(r)
            return response['account']
        except Exception as e:
            logger.error(f"Error fetching account summary: {e}")
            return {}

    def get_open_positions(self) -> List[Dict]:
        """
        Get all open positions

        Returns:
            List of position dictionaries formatted for dashboard
        """
        try:
            r = positions.OpenPositions(accountID=self.account_id)
            response = self.client.request(r)

            position_list = []
            for pos in response.get('positions', []):
                # Handle both long and short positions
                if pos['long']['units'] != '0':
                    units = float(pos['long']['units'])
                    avg_price = float(pos['long']['averagePrice'])
                    unrealized_pnl = float(pos['long']['unrealizedPL'])
                    direction = 'long'
                elif pos['short']['units'] != '0':
                    units = float(pos['short']['units'])
                    avg_price = float(pos['short']['averagePrice'])
                    unrealized_pnl = float(pos['short']['unrealizedPL'])
                    direction = 'short'
                else:
                    continue

                position_list.append({
                    'position_id': f"{pos['instrument']}_{direction}",
                    'instrument': pos['instrument'],
                    'direction': direction,
                    'entry_time': datetime.now().isoformat(),  # Oanda doesn't provide this directly
                    'entry_price': avg_price,
                    'size': abs(units),
                    'current_price': avg_price + (unrealized_pnl / abs(units)),
                    'unrealized_pnl': unrealized_pnl,
                    'stop_loss': None,  # Would need to get from associated orders
                    'take_profit': None
                })

            return position_list

        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return []

    def get_trade_history(self, count: int = 50) -> List[Dict]:
        """
        Get closed trades history

        Args:
            count: Number of trades to retrieve

        Returns:
            List of trade dictionaries formatted for dashboard
        """
        try:
            params = {
                "count": count,
                "state": "CLOSED"
            }
            r = trades.TradesList(accountID=self.account_id, params=params)
            response = self.client.request(r)

            trade_list = []
            for trade in response.get('trades', []):
                # Calculate P&L
                realized_pnl = float(trade.get('realizedPL', 0))
                units = float(trade['initialUnits'])

                trade_list.append({
                    'trade_id': trade['id'],
                    'instrument': trade['instrument'],
                    'direction': 'long' if units > 0 else 'short',
                    'entry_time': trade.get('openTime', ''),
                    'exit_time': trade.get('closeTime', datetime.now().isoformat()),
                    'entry_price': float(trade.get('price', 0)),
                    'exit_price': float(trade.get('averageClosePrice', 0)),
                    'size': abs(units),
                    'pnl': realized_pnl,
                    'pnl_percent': (realized_pnl / (abs(units) * float(trade.get('price', 1)))) * 100,
                    'commission': float(trade.get('financing', 0)),
                    'slippage': 0,  # Calculate if needed
                    'strategy': 'manual',  # Tag with your strategy name
                    'status': 'closed'
                })

            return trade_list

        except Exception as e:
            logger.error(f"Error fetching trade history: {e}")
            return []

    def get_current_prices(self, instruments: List[str]) -> Dict[str, Dict]:
        """
        Get current bid/ask prices for instruments

        Args:
            instruments: List of instrument names (e.g., ['EUR_USD', 'GBP_USD'])

        Returns:
            Dictionary mapping instrument to price data
        """
        try:
            params = {
                "instruments": ",".join(instruments)
            }
            r = pricing.PricingInfo(accountID=self.account_id, params=params)
            response = self.client.request(r)

            prices = {}
            for price in response.get('prices', []):
                prices[price['instrument']] = {
                    'bid': float(price['bids'][0]['price']),
                    'ask': float(price['asks'][0]['price']),
                    'spread': float(price['asks'][0]['price']) - float(price['bids'][0]['price']),
                    'timestamp': price['time']
                }

            return prices

        except Exception as e:
            logger.error(f"Error fetching prices: {e}")
            return {}

    def place_market_order(self, instrument: str, units: int,
                           stop_loss: Optional[float] = None,
                           take_profit: Optional[float] = None) -> Dict:
        """
        Place a market order

        Args:
            instrument: Trading instrument (e.g., 'EUR_USD')
            units: Number of units (positive for long, negative for short)
            stop_loss: Stop loss price
            take_profit: Take profit price

        Returns:
            Order response dictionary
        """
        try:
            order_data = {
                "order": {
                    "type": "MARKET",
                    "instrument": instrument,
                    "units": str(units),
                    "timeInForce": "FOK",
                    "positionFill": "DEFAULT"
                }
            }

            # Add stop loss if provided
            if stop_loss:
                order_data["order"]["stopLossOnFill"] = {
                    "price": str(stop_loss)
                }

            # Add take profit if provided
            if take_profit:
                order_data["order"]["takeProfitOnFill"] = {
                    "price": str(take_profit)
                }

            r = orders.OrderCreate(accountID=self.account_id, data=order_data)
            response = self.client.request(r)

            logger.info(f"Order placed: {instrument} {units} units")
            return response

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {"error": str(e)}

    def close_position(self, instrument: str) -> Dict:
        """
        Close an open position

        Args:
            instrument: Instrument to close (e.g., 'EUR_USD')

        Returns:
            Response dictionary
        """
        try:
            # Close long position
            data = {"longUnits": "ALL"}
            r = positions.PositionClose(
                accountID=self.account_id,
                instrument=instrument,
                data=data
            )
            response = self.client.request(r)

            logger.info(f"Position closed: {instrument}")
            return response

        except oandapyV20.exceptions.V20Error:
            # Try closing short position if long fails
            try:
                data = {"shortUnits": "ALL"}
                r = positions.PositionClose(
                    accountID=self.account_id,
                    instrument=instrument,
                    data=data
                )
                response = self.client.request(r)
                logger.info(f"Position closed: {instrument}")
                return response
            except Exception as e2:
                logger.error(f"Error closing position: {e2}")
                return {"error": str(e2)}

    def fetch_historical_data(self, instrument: str, granularity: str = 'H1',
                              count: Optional[int] = None,
                              start_time: Optional[str] = None,
                              end_time: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from Oanda

        Args:
            instrument: Trading pair (e.g., 'EUR_USD')
            granularity: Candle size - 'M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D', 'W', 'M'
            count: Number of candles to fetch (max 5000). If None, uses start_time/end_time
            start_time: ISO format datetime string (e.g., '2023-01-01T00:00:00Z')
            end_time: ISO format datetime string (e.g., '2023-12-31T23:59:59Z')

        Returns:
            DataFrame with columns: open, high, low, close, volume
            Index: datetime

        Examples:
            # Get last 500 H1 candles
            df = oanda.fetch_historical_data('EUR_USD', 'H1', count=500)

            # Get specific date range
            df = oanda.fetch_historical_data(
                'EUR_USD',
                'H1',
                start_time='2023-01-01T00:00:00Z',
                end_time='2023-12-31T23:59:59Z'
            )
        """
        try:
            # Build parameters
            params = {
                "granularity": granularity,
            }

            if count is not None:
                # Use count (simpler, max 5000)
                if count > 5000:
                    logger.warning(f"Count {count} exceeds max 5000, using 5000")
                    count = 5000
                params["count"] = count
            else:
                # Use date range
                if start_time:
                    params["from"] = start_time
                if end_time:
                    params["to"] = end_time

            # Make API request
            logger.info(f"Fetching {instrument} {granularity} data...")
            r = instruments.InstrumentsCandles(instrument=instrument, params=params)
            response = self.client.request(r)

            # Parse candles
            candles_data = []
            for candle in response.get('candles', []):
                # Only use complete candles
                if candle.get('complete', True):
                    candles_data.append({
                        'time': candle['time'],
                        'open': float(candle['mid']['o']),
                        'high': float(candle['mid']['h']),
                        'low': float(candle['mid']['l']),
                        'close': float(candle['mid']['c']),
                        'volume': int(candle['volume'])
                    })

            # Convert to DataFrame
            df = pd.DataFrame(candles_data)

            if len(df) == 0:
                logger.warning(f"No data returned for {instrument}")
                return pd.DataFrame()

            # Set datetime index
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)

            logger.info(f"Fetched {len(df)} candles for {instrument}")
            logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")

            return df

        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()

    def fetch_historical_data_range(self, instrument: str, granularity: str = 'H1',
                                    days: int = 365) -> pd.DataFrame:
        """
        Fetch historical data for a specific number of days
        Handles pagination if needed (Oanda max 5000 candles per request)

        Args:
            instrument: Trading pair (e.g., 'EUR_USD')
            granularity: Candle size
            days: Number of days of history to fetch

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Calculate time range
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days)

            # Format for Oanda API
            start_str = start_time.strftime('%Y-%m-%dT%H:%M:%S.000000000Z')
            end_str = end_time.strftime('%Y-%m-%dT%H:%M:%S.000000000Z')

            # Calculate approximate number of candles
            granularity_minutes = {
                'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
                'H1': 60, 'H4': 240, 'D': 1440, 'W': 10080
            }

            minutes_per_candle = granularity_minutes.get(granularity, 60)
            total_minutes = days * 24 * 60
            approx_candles = total_minutes / minutes_per_candle

            # If more than 5000 candles, need to paginate
            if approx_candles > 5000:
                logger.info(f"Need ~{approx_candles:.0f} candles, will paginate...")
                return self._fetch_with_pagination(instrument, granularity, start_str, end_str)
            else:
                # Can fetch in single request
                return self.fetch_historical_data(
                    instrument, granularity,
                    start_time=start_str,
                    end_time=end_str
                )

        except Exception as e:
            logger.error(f"Error fetching historical data range: {e}")
            return pd.DataFrame()

    def _fetch_with_pagination(self, instrument: str, granularity: str,
                               start_time: str, end_time: str) -> pd.DataFrame:
        """
        Fetch data with pagination for large date ranges

        Args:
            instrument: Trading pair
            granularity: Candle size
            start_time: Start datetime string
            end_time: End datetime string

        Returns:
            Combined DataFrame
        """
        all_data = []
        current_start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        final_end = datetime.fromisoformat(end_time.replace('Z', '+00:00'))

        logger.info(f"Fetching data from {current_start} to {final_end} with pagination...")

        while current_start < final_end:
            # Fetch chunk of 5000 candles
            chunk_start = current_start.strftime('%Y-%m-%dT%H:%M:%S.000000000Z')

            df_chunk = self.fetch_historical_data(
                instrument, granularity,
                count=5000,
                start_time=chunk_start
            )

            if df_chunk.empty:
                break

            all_data.append(df_chunk)

            # Move to next chunk (start after last candle)
            current_start = df_chunk.index[-1] + timedelta(seconds=1)

            # Rate limiting - avoid hitting API limits
            time.sleep(0.1)

            logger.info(f"Fetched chunk up to {df_chunk.index[-1]}")

        if not all_data:
            logger.warning("No data fetched")
            return pd.DataFrame()

        # Combine all chunks
        df_combined = pd.concat(all_data)

        # Remove duplicates and sort
        df_combined = df_combined[~df_combined.index.duplicated(keep='first')]
        df_combined = df_combined.sort_index()

        logger.info(f"Total candles fetched: {len(df_combined)}")

        return df_combined


class DashboardDataSync:
    """
    Synchronizes Oanda data with the dashboard database
    """

    def __init__(self, oanda_connector: OandaConnector, db_manager):
        self.oanda = oanda_connector
        self.db = db_manager
        self.last_sync = None

    def sync_positions(self):
        """Sync open positions to dashboard"""
        try:
            positions = self.oanda.get_open_positions()

            for position in positions:
                self.db.update_position(position)

            logger.info(f"Synced {len(positions)} positions")

        except Exception as e:
            logger.error(f"Error syncing positions: {e}")

    def sync_trade_history(self, count: int = 100):
        """Sync trade history to dashboard"""
        try:
            trades = self.oanda.get_trade_history(count=count)

            for trade in trades:
                self.db.add_trade(trade)

            logger.info(f"Synced {len(trades)} trades")

        except Exception as e:
            logger.error(f"Error syncing trades: {e}")

    def sync_all(self):
        """Perform full sync"""
        logger.info("Starting full data sync...")
        self.sync_positions()
        self.sync_trade_history()
        self.last_sync = datetime.now()
        logger.info("Full sync completed")

    def start_auto_sync(self, interval_seconds: int = 60):
        """
        Start automatic syncing at regular intervals

        Args:
            interval_seconds: Seconds between syncs
        """
        logger.info(f"Auto-sync started (interval: {interval_seconds}s)")

        while True:
            try:
                self.sync_all()
                time.sleep(interval_seconds)
            except KeyboardInterrupt:
                logger.info("Auto-sync stopped")
                break
            except Exception as e:
                logger.error(f"Error in auto-sync: {e}")
                time.sleep(interval_seconds)


# Example usage
if __name__ == "__main__":
    # Configuration
    ACCOUNT_ID = "your-account-id"
    ACCESS_TOKEN = "your-access-token"
    ENVIRONMENT = "practice"  # or "live"

    # Initialize connector
    oanda = OandaConnector(
        account_id=ACCOUNT_ID,
        access_token=ACCESS_TOKEN,
        environment=ENVIRONMENT
    )

    # Get account summary
    account = oanda.get_account_summary()
    print(f"Balance: {account.get('balance', 'N/A')}")
    print(f"NAV: {account.get('NAV', 'N/A')}")
    print(f"Unrealized P&L: {account.get('unrealizedPL', 'N/A')}")

    # Get open positions
    open_positions = oanda.get_open_positions()
    print(f"\nOpen Positions: {len(open_positions)}")
    for pos in open_positions:
        print(f"  {pos['instrument']}: {pos['direction']} {pos['size']} units, P&L: ${pos['unrealized_pnl']:.2f}")

    # Get trade history
    trade_history = oanda.get_trade_history(count=10)
    print(f"\nRecent Trades: {len(trade_history)}")
    for trade in trade_history[:5]:
        print(f"  {trade['instrument']}: ${trade['pnl']:.2f}")

    # Get current prices
    instrument_list = ['EUR_USD', 'GBP_USD', 'USD_JPY']
    prices = oanda.get_current_prices(instrument_list)
    print("\nCurrent Prices:")
    for inst, price_data in prices.items():
        print(f"  {inst}: Bid={price_data['bid']:.5f}, Ask={price_data['ask']:.5f}, Spread={price_data['spread']:.5f}")

    # Initialize database manager (from main dashboard code)
    from trading_dashboard_main import DatabaseManager
    db = DatabaseManager()

    # Setup auto-sync
    sync = DashboardDataSync(oanda, db)

    # Perform one-time sync
    sync.sync_all()

    # Or start automatic syncing (runs continuously)
    # sync.start_auto_sync(interval_seconds=60)
