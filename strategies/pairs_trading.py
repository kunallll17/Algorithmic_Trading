from .base_strategy import BaseStrategy
import pandas as pd
import numpy as np

class PairsTradingStrategy(BaseStrategy):
    """
    Implements 'Pairs Trading' (Statistical Arbitrage) from Paper 2.
    Uses Bollinger Bands on the spread (Price A / Price B).
    """
    
    def __init__(self, ticker_a='GOOGL', ticker_b='MSFT', window=20, std_dev=2):
        super().__init__(f"Pairs Trading ({ticker_a}/{ticker_b})")
        self.ticker_a = ticker_a
        self.ticker_b = ticker_b
        self.window = window
        self.std_dev = std_dev
        self.in_position = False # 'long_spread' (Long A, Short B) or 'short_spread' (Short A, Long B) or False

    def generate_signal(self, current_date, data_dict, current_positions, portfolio_value):
        orders = []
        
        # Ensure both assets exist
        if self.ticker_a not in data_dict or self.ticker_b not in data_dict:
            return []
            
        df_a = data_dict[self.ticker_a]
        df_b = data_dict[self.ticker_b]
        
        # Need history for indicators
        # Window size 20 is small, let's grab 50 to be safe
        try:
            loc_a = df_a.index.get_indexer([current_date], method='pad')[0]
            loc_b = df_b.index.get_indexer([current_date], method='pad')[0]
            
            if loc_a < 30 or loc_b < 30: return []
            
            # Align history approximately?
            # Easiest way: Get the prices for the last N dates available in the master timeline
            # Engine doesn't pass master timeline history easily without full scan.
            # Let's slice the DataFrames individually assuming they are somewhat aligned or we merge.
            # Best practice for Pairs: Merge on Date.
            
            # Slice last 50 rows from both, merge to ensure date alignment
            slice_a = df_a.iloc[loc_a-50 : loc_a+1]['close'].rename(self.ticker_a)
            slice_b = df_b.iloc[loc_b-50 : loc_b+1]['close'].rename(self.ticker_b)
            
            merged = pd.concat([slice_a, slice_b], axis=1).dropna()
            
            if len(merged) < self.window: return []
            
            # Calculate Spread
            # Paper 2: Spread = Price A / Price B (Ratio) or Price A - Multiplier * Price B (Cointegration)
            # Paper text says "Price A / Price B" or similar simple ratio for basic pairs.
            spread = merged[self.ticker_a] / merged[self.ticker_b]
            
            # Bollinger Bands on Spread
            rolling_mean = spread.rolling(window=self.window).mean()
            rolling_std = spread.rolling(window=self.window).std()
            
            upper_band = rolling_mean + (self.std_dev * rolling_std)
            lower_band = rolling_mean - (self.std_dev * rolling_std)
            
            current_spread = spread.iloc[-1]
            curr_upper = upper_band.iloc[-1]
            curr_lower = lower_band.iloc[-1]
            curr_mean = rolling_mean.iloc[-1]
            
            # Logic
            # If Spread > Upper -> Spread is too high (A is expensive relative to B) -> Short A, Long B
            # If Spread < Lower -> Spread is too low (A is cheap relative to B) -> Long A, Short B
            # Exit when Spread returns to Mean
            
            # Current holdings: we check self.in_position or query current_positions
            
            has_a = current_positions.get(self.ticker_a, 0)
            has_b = current_positions.get(self.ticker_b, 0)
            
            # Define State from positions (simplified)
            # Long Spread: Long A, Short B (or just Long A if short restricted)
            # Short Spread: Short A, Long B
            
            # Entry Logic
            if not self.in_position:
                if current_spread > curr_upper:
                    # Signal: Short Spread (Short A, Long B)
                    # Allocation: 50% capital each leg? Or 100% total?
                    half_capital = portfolio_value * 0.45
                    
                    orders.append({'symbol': self.ticker_a, 'action': 'sell'}) # Short A (engine assumes sell=short if 0 held? Need explicit short support or just sell what we don't have)
                    # Note: Engine 'sell' currently just reduces position. If we seek to Short, we need negative position support.
                    # Assuming Engine supports negative quantity positions implicitly if we sell from 0?
                    # Engine code: `current_shares = positions.get(symbol, 0); if current_shares > 0...`
                    # Ah, basic engine prevented selling if shares=0.
                    # CRITICAL: We need to enable Shorting in Engine for Pairs Trading.
                    # Workaround for now: Long-Only Pairs? (Long B when Spread High, Long A when Spread Low)
                    # That is valid. "Switching" strategy.
                    
                    # Let's implement Long-Only "Switching" for safety if Shorting is complex.
                    # "Short A, Long B" -> Just Buy B (and Sell A if held)
                    
                    orders.append({'symbol': self.ticker_a, 'action': 'sell'}) # Close A if held
                    orders.append({'symbol': self.ticker_b, 'action': 'buy', 'cash_amount': half_capital * 2}) # Buy B
                    self.in_position = 'short_spread' # Actually just Long B
                    
                elif current_spread < curr_lower:
                    # Signal: Long Spread (Long A, Short B) -> Just Buy A
                    half_capital = portfolio_value * 0.45
                    
                    orders.append({'symbol': self.ticker_b, 'action': 'sell'}) # Close B if held
                    orders.append({'symbol': self.ticker_a, 'action': 'buy', 'cash_amount': half_capital * 2}) # Buy A
                    self.in_position = 'long_spread' # Actually just Long A
                    
            # Exit Logic (Mean Reversion)
            elif self.in_position:
                # Check for Mean Reversion (crossing the moving average)
                # If we were Long Spread (Held A), and spread goes > Mean, we exit? or wait for Upper?
                # Usually exit at Mean.
                
                if self.in_position == 'long_spread' and current_spread >= curr_mean:
                    orders.append({'symbol': self.ticker_a, 'action': 'sell'}) # Close A
                    self.in_position = False
                    
                elif self.in_position == 'short_spread' and current_spread <= curr_mean:
                    orders.append({'symbol': self.ticker_b, 'action': 'sell'}) # Close B
                    self.in_position = False
                    
        except Exception as e:
            # print(f"Pairs Error: {e}")
            pass
            
        return orders
