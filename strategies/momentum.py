from .base_strategy import BaseStrategy
import pandas as pd
import numpy as np

class MomentumStrategy(BaseStrategy):
    """
    Implements 'Returns to Buying Winners and Selling Losers' (Jegadeesh & Titman, 1993).
    
    Parameters:
    - J (Lookback Period): 12 months
    - K (Holding Period): 3 months
    - Skip Week: Exclude last week updates to avoid microstructure noise.
    - January Effect: Avoid long positions in January.
    """
    
    def __init__(self, j_months=12, k_months=3, top_percentile=0.1):
        super().__init__("Momentum (J=12, K=3)")
        self.j_months = j_months
        self.k_months = k_months
        self.top_percentile = top_percentile
        
        self.rebalance_tracker = 0 # Months since last rebalance
        self.last_rebalance_month = None

    def generate_signal(self, current_date, data_dict, current_positions, portfolio_value):
        """
        Executes momentum logic.
        """
        orders = []
        
        # 1. Check if we should rebalance
        # Logic: Rebalance at the start of the simulation, and then every K months
        # Simplified: Check if month changed. If it's a "rebalance month", act.
        
        current_month = current_date.month
        current_year = current_date.year
        
        # Track unique months seen to handle Rebalance Frequency K
        # A simple approximation: Rebalance on the 1st trading day of the month
        if self.last_rebalance_month == current_month:
            return [] # Already processed this month
            
        # It's a new month. check K period.
        # Ideally we track robustly, but for now let's rebalance EVERY MONTH (K=1 sliding)
        # Or strictly every K. Let's do K=3 fixed blocks for fidelity to the paper?
        # The paper says "Overlapping portfolios" or "Fixed holding".
        # Let's implement: Rebalance Every Month to hold the "Top Decile" as determined 
        # by the J-month lookback. This is a common modern interpretation (monthly rebalance).
        
        self.last_rebalance_month = current_month
        
        # 2. January Effect Filter
        # "If Month == January: Close Long positions"
        if current_month == 1:
            # print(f"[{current_date.date()}] January Element: Closing Longs/Staying Cash.")
            return [{'action': 'close_all'}]
            
        # 3. Calculate Momentum for all assets
        scores = []
        
        for symbol, df in data_dict.items():
            # Need history up to current_date
            # Slicing: df.loc[:current_date]
            
            # Optimization: use get_loc ensures we don't copy the whole DF
            # But we need a window.
            
            # Lookback: J=12 months approx 252 days.
            # Skip week: 5 days.
            
            # Let's get the price 1 week ago and 12 months ago
            
            # Efficient lookup:
            try:
                # Get index location of current date
                # We need exact index or nearest past
                curr_idx_loc = df.index.get_indexer([current_date], method='pad')[0]
                if curr_idx_loc == -1: continue
                
                # 1 Week Ago index
                week_ago_idx = curr_idx_loc - 5
                # 12 Months Ago index (approx 252 days)
                year_ago_idx = curr_idx_loc - 252
                
                if year_ago_idx < 0: continue # Not enough history
                
                price_now = df.iloc[curr_idx_loc]['close']
                price_1wk_ago = df.iloc[week_ago_idx]['close']
                price_1yr_ago = df.iloc[year_ago_idx]['close']
                
                # Momentum Return = (Price[1_Week_Ago] / Price[12_Months_Ago]) - 1
                # We use 1 week ago as the 'Recent' price to skip the bid-ask bounce
                
                momentum_score = (price_1wk_ago / price_1yr_ago) - 1
                
                scores.append((symbol, momentum_score))
                
            except Exception as e:
                # Index out of bounds or data issues
                continue
                
        if not scores:
            return []
            
        # 4. Rank and Select
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        num_assets = len(scores)
        top_n = max(1, int(num_assets * self.top_percentile))
        
        winners = [x[0] for x in scores[:top_n]]
        
        # print(f"[{current_date.date()}] Winners: {winners}")
        
        # 5. Generate Rebalance Orders
        # Sell everything currently held that is NOT a winner
        # Buy Winners equal weight
        
        # First, sell non-winners
        for symbol in list(current_positions.keys()):
            if symbol not in winners:
                orders.append({'symbol': symbol, 'action': 'sell'})
                
        # Determine Cash Allocation per Winner
        # We want to be fully invested in Winners
        # Total Equity = portfolio_value
        # Allocation per asset = Total Equity / top_n
        
        target_alloc = portfolio_value * 0.95 / top_n # 95% invested to leave buffer for comms
        
        for win_sym in winners:
            # If we don't have it, buy it
            if win_sym not in current_positions:
                orders.append({
                    'symbol': win_sym, 
                    'action': 'buy', 
                    'cash_amount': target_alloc
                })
            else:
                # Rebalance? (Optional, skip for simplicity or just hold)
                # If we already have it, we could adjust to match target_alloc
                # Let's Hold for now to minimize turnover costs
                pass
                
        return orders
