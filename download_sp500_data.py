#!/usr/bin/env python3
"""
Download historical S&P 500 data from Yahoo Finance
"""

import yfinance as yf
import pandas as pd


def download_sp500_data(start_date=None, end_date=None, period='max'):
    """
    Download S&P 500 historical data

    Parameters:
    -----------
    start_date : str, optional
        Start date in format 'YYYY-MM-DD'
    end_date : str, optional
        End date in format 'YYYY-MM-DD'
    period : str, optional
        Time period if start_date not specified: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max

    Returns:
    --------
    pandas.DataFrame : OHLCV data
    """
    # S&P 500 ticker symbol on Yahoo Finance
    ticker = "^GSPC"

    print("Downloading S&P 500 (^GSPC) historical data...")

    # Download data
    if start_date and end_date:
        data = yf.download(ticker, start=start_date, end=end_date, progress=True)
    else:
        data = yf.download(ticker, period=period, progress=True)

    # Remove any multi-level column indexing first
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Rename columns to match trading system format (lowercase)
    data.columns = [str(col).lower() for col in data.columns]

    print(f"\nDownloaded {len(data)} rows of data")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    print(f"\nColumns: {list(data.columns)}")
    print("\nFirst few rows:")
    print(data.head())
    print("\nLast few rows:")
    print(data.tail())

    return data


if __name__ == "__main__":
    # Download all available historical data
    sp500_data = download_sp500_data(period='max')

    # Save to CSV
    output_file = 'sp500_historical_data.csv'
    sp500_data.to_csv(output_file)
    print(f"\n✓ Data saved to: {output_file}")

    # Also download YTD 2025 data separately
    ytd_data = download_sp500_data(start_date='2025-01-01', end_date='2025-12-30')
    ytd_file = 'sp500_ytd_2025.csv'
    ytd_data.to_csv(ytd_file)
    print(f"✓ YTD 2025 data saved to: {ytd_file}")

    # Calculate YTD return
    if len(ytd_data) > 0:
        start_price = ytd_data['close'].iloc[0]
        end_price = ytd_data['close'].iloc[-1]
        ytd_return = ((end_price - start_price) / start_price) * 100
        print(f"\n2025 YTD Return: {ytd_return:.2f}%")
