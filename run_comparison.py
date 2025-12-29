import pandas as pd
import matplotlib.pyplot as plt
import os
from main import load_data
from backtester.engine import BacktesterEngine
from backtester.performance import Performance

# Import Strategies
from strategies.buy_hold import BuyHoldStrategy
from strategies.momentum import MomentumStrategy
from strategies.perfect_order import PerfectOrderStrategy
from strategies.pairs_trading import PairsTradingStrategy
from strategies.rpmm import RPMMStrategy
from strategies.tdqn_agent import TDQNAgent

def run_comparison():
    print("Loading Data...")
    # Load all data or a subset?
    # Let's load the big 3 for general strategies
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    data_dict = load_data(selected_symbols=tickers)
    
    # Define Strategies to Test
    strategies = {
        "Buy & Hold": BuyHoldStrategy(),
        "Momentum (Paper 3)": MomentumStrategy(),
        "Perfect Order (Paper 4)": PerfectOrderStrategy(),
        "TDQN AI (Paper 1)": TDQNAgent(), # Will load pre-trained model if exists
        "RPMM (Paper 2)": RPMMStrategy()
    }
    
    # Pairs Trading needs specific pair
    # We can run it on the same data dict if it contains the pair
    strategies["Pairs Trading (Paper 2)"] = PairsTradingStrategy(ticker_a='GOOGL', ticker_b='MSFT')
    
    results = []
    equity_curves = {}
    
    print(f"Running Backtests for {len(strategies)} strategies...")
    
    for name, strategy in strategies.items():
        print(f"Running {name}...")
        try:
            engine = BacktesterEngine(initial_balance=10000)
            metrics, history = engine.run(data_dict, strategy)
            
            # Store Metrics
            metrics['Strategy'] = name
            results.append(metrics)
            
            # Store Equity Curve (Portfolio Value)
            # The engine stores it as 'TotalValue'
            equity_curves[name] = history['TotalValue']
            
        except Exception as e:
            print(f"Failed to run {name}: {e}")
            
    # --- Generate Outputs ---
    os.makedirs("results", exist_ok=True)
    
    # 1. Metrics CSV
    results_df = pd.DataFrame(results)
    # Reorder columns
    cols = ['Strategy', 'Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Geometric Mean Return', 'Annualized Volatility']
    # Filter for cols that exist
    cols = [c for c in cols if c in results_df.columns]
    results_df = results_df[cols]
    
    csv_path = "results/comparison_metrics.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Saved metrics to {csv_path}")
    print(results_df)
    
    # 2. Equity Curve Plot
    from src.visualizer import Visualizer
    # We assume Visualizer has a method for this, or we plot directly here using matplotlib
    
    plt.figure(figsize=(12, 6))
    for name, series in equity_curves.items():
        plt.plot(series.index, series.values, label=name)
        
    plt.title("Strategy Comparison: Equity Curves")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)
    
    plot_path = "results/equity_curves.png"
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")
    
if __name__ == "__main__":
    run_comparison()
