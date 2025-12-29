from .base_strategy import BaseStrategy

class BuyHoldStrategy(BaseStrategy):
    """
    Simplest strategy: Buy at the beginning and hold forever.
    Updated for Multi-Asset Engine.
    """
    
    def __init__(self):
        super().__init__("Buy & Hold")
        self.bought = False

    def generate_signal(self, current_date, data_dict, current_positions, portfolio_value):
        """
        Returns buy signals for all available assets on the first tick, then nothing.
        """
        if not self.bought:
            self.bought = True
            
            # Buy everything available equally
            orders = []
            available_symbols = [s for s, df in data_dict.items() if current_date in df.index]
            
            if not available_symbols:
                self.bought = False # Try again next tick unique case
                return []
                
            allocation = portfolio_value * 0.95 / len(available_symbols)
            
            for sym in available_symbols:
                orders.append({
                    'symbol': sym,
                    'action': 'buy',
                    'cash_amount': allocation
                })
                
            return orders
        
        return []
