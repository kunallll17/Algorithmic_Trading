"""
Data Download Script
Downloads historical stock data from Yahoo Finance for the trading strategies.

Usage:
    python download_data.py
"""
from data_collector import DataCollector

def main():
    """Download historical data for AAPL, GOOGL, and MSFT."""
    print("=" * 60)
    print("DATA DOWNLOAD SCRIPT")
    print("=" * 60)
    
    # Initialize data collector
    collector = DataCollector(data_dir="data")
    
    # Define symbols to download
    symbols = ["AAPL", "GOOGL", "MSFT"]
    
    # Date range for the study (2015-2023)
    start_date = "2015-01-01"
    end_date = "2024-12-31"
    
    print(f"\nDownloading data for: {', '.join(symbols)}")
    print(f"Date range: {start_date} to {end_date}\n")
    
    # Download data for all symbols
    data_dict = collector.download_multiple_stocks(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date
    )
    
    print("\n" + "=" * 60)
    print(f"Successfully downloaded data for {len(data_dict)} symbols")
    print("=" * 60)
    
    # Print summary
    for symbol, df in data_dict.items():
        print(f"{symbol}: {len(df)} rows, from {df.index[0]} to {df.index[-1]}")

if __name__ == "__main__":
    main()

