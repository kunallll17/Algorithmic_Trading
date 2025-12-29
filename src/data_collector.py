"""
Data collection module for downloading historical stock data using yfinance.
"""
import yfinance as yf
import pandas as pd
import os
from typing import List, Optional


class DataCollector:
    """Handles downloading and saving historical stock data."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def download_stock_data(
        self,
        symbol: str,
        start_date: str = "2015-01-01",
        end_date: str = "2024-12-31",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Download historical stock data for a given symbol.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Data interval ('1d', '1h', etc.)
        
        Returns:
            DataFrame with OHLCV data
        """
        print(f"Downloading data for {symbol}...")
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval=interval)
        
        if df.empty:
            raise ValueError(f"No data retrieved for {symbol}")
        
        # Rename columns to lowercase for consistency
        df.columns = [col.lower() for col in df.columns]
        
        # Save to CSV
        filepath = os.path.join(self.data_dir, f"{symbol}.csv")
        df.to_csv(filepath)
        print(f"Data saved to {filepath}")
        
        return df
    
    def download_multiple_stocks(
        self,
        symbols: List[str],
        start_date: str = "2015-01-01",
        end_date: str = "2024-12-31"
    ) -> dict:
        """
        Download data for multiple stocks.
        
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        data_dict = {}
        for symbol in symbols:
            try:
                data_dict[symbol] = self.download_stock_data(
                    symbol, start_date, end_date
                )
            except Exception as e:
                print(f"Error downloading {symbol}: {e}")
        
        return data_dict
    
    def load_stock_data(self, symbol: str) -> pd.DataFrame:
        """Load previously downloaded stock data from CSV."""
        filepath = os.path.join(self.data_dir, f"{symbol}.csv")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        df.columns = [col.lower() for col in df.columns]
        return df




