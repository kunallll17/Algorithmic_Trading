from .base_strategy import BaseStrategy
import numpy as np

class RPMMStrategy(BaseStrategy):
    """
    Implements 'Modified RPMM' (Reservation Price) from Paper 2.
    """
    
    def __init__(self, window=10):
        super().__init__("RPMM (Modified)")
        self.window = window

    def generate_signal(self, current_date, data_dict, current_positions, portfolio_value):
        orders = []
        
        # Iterate over all assets (RPMM is single-asset logic applied to many)
        for symbol, df in data_dict.items():
            try:
                # Need history
                loc = df.index.get_indexer([current_date], method='pad')[0]
                if loc < self.window: continue
                
                # Get Window
                history = df.iloc[loc - self.window : loc + 1]
                
                # Calculate M (Max) and m (Min) in window
                # 'High' and 'Low' columns preferred? Or just Close?
                # Paper uses High/Low of the period.
                
                if 'high' in history.columns and 'low' in history.columns:
                    M = history['high'].max()
                    m = history['low'].min()
                else:
                    M = history['close'].max()
                    m = history['close'].min()
                    
                # Reservation Price q*
                q_star = np.sqrt(M * m)
                
                current_price = history.iloc[-1]['close']
                
                has_position = symbol in current_positions
                
                # Rule:
                # Buy if Price <= q*
                # Sell if Price >= q*
                
                cash_alloc = portfolio_value * 0.95 / len(data_dict) # Split capital
                
                if current_price <= q_star and not has_position:
                    orders.append({
                        'symbol': symbol,
                        'action': 'buy',
                        'cash_amount': cash_alloc
                    })
                    
                elif current_price >= q_star and has_position:
                    orders.append({
                        'symbol': symbol,
                        'action': 'sell'
                    })
                    
            except Exception as e:
                continue
                
        return orders
