import pandas as pd
import argparse
import os
from strategies.buy_hold import BuyHoldStrategy
from strategies.momentum import MomentumStrategy
from backtester.engine import BacktesterEngine

def load_data(data_dir="data", selected_symbols=None):
    """
    Loads data from CSVs in the directory.
    Returns: dict {symbol: dataframe}
    """
    data_dict = {}
    
    # Just list a few files manualy to start with if selected_symbols is None
    # Or scan directory
    
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    for f in files:
        symbol = f.replace('.csv', '').upper()
        
        # If user specified symbols, filter
        if selected_symbols and symbol not in selected_symbols:
            continue
            
        path = os.path.join(data_dir, f)
        try:
            df = pd.read_csv(path)
            # Normalize columns to lowercase
            df.columns = [c.lower() for c in df.columns]
            
            # Ensure 'date' is datetime index
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], utc=True)
                # Remove timezone info for simplicity
                df['date'] = df['date'].dt.tz_localize(None)
                df.set_index('date', inplace=True)
                df = df.sort_index()
                
            data_dict[symbol] = df
            print(f"Loaded {symbol}: {len(df)} rows")
        except Exception as e:
            print(f"Error loading {path}: {e}")
            
    return data_dict

def main():
    parser = argparse.ArgumentParser(description='Algorithmic Trading Comparative Study')
    parser.add_argument('--tickers', type=str, default='AAPL,GOOGL,MSFT', help='Comma separated list of tickers')
    parser.add_argument('--strategy', type=str, default='buy_hold', help='Strategy to run (buy_hold, momentum, etc.)')
    parser.add_argument('--train', action='store_true', help='Train the strategy (if applicable) before running')
    
    args = parser.parse_args()
    
    tickers = args.tickers.split(',')
    print(f"--- Running Comparative Study for {tickers} ---")
    
    # 1. Load Data
    data_dict = load_data("data", tickers)
    if not data_dict:
        print("No data loaded.")
        return

    # 2. Select Strategy
    strategy = None
    if args.strategy == 'buy_hold':
        strategy = BuyHoldStrategy()
    elif args.strategy == 'momentum':
        strategy = MomentumStrategy()
    elif args.strategy == 'perfect_order':
        # Lazy import to avoid 'ta' dependency if not using it? No, imports are top level usually.
        # But we can import inside.
        from strategies.perfect_order import PerfectOrderStrategy
        strategy = PerfectOrderStrategy()
    elif args.strategy == 'pairs_trading':
        from strategies.pairs_trading import PairsTradingStrategy
        # Default to GOOGL/MSFT or parse from tickers
        t1, t2 = tickers[:2] if len(tickers) >= 2 else ('GOOGL', 'MSFT')
        strategy = PairsTradingStrategy(ticker_a=t1, ticker_b=t2)
    elif args.strategy == 'rpmm':
        from strategies.rpmm import RPMMStrategy
        strategy = RPMMStrategy()
    elif args.strategy == 'tdqn':
        from strategies.tdqn_agent import TDQNAgent
        strategy = TDQNAgent()
        # Check if we need to train
        # Quick hack: check for separate training flag or just Train if model missing?
        # Let's assume user trains via a separate call or we add a --train flag.
        # For now, let's just Instantiate.
    else:
        print(f"Strategy {args.strategy} not implemented yet.")
        return
        
    # Optional Training Mode
    if args.train and hasattr(strategy, 'train'):
        strategy.train(data_dict)
        
    # 3. Run Backtest
    # Pass 'strategy' which now might have a trained model
    engine = BacktesterEngine(initial_balance=10000)
    metrics, history_df = engine.run(data_dict, strategy)
    
    # 4. Report Results
    print("\n--- Performance Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    # Save results to CSV (optional)
    # history_df.to_csv(f"results/{args.strategy}_portfolio.csv")

if __name__ == "__main__":
    main()
