from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    """
    Abstract Base Class for all trading strategies.
    Ensures compatibility with the Backtester Engine.
    """
    
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def generate_signal(self, current_date, data_dict, current_positions, portfolio_value):
        """
        Generates trading signals for the current date.
        
        Args:
            current_date (pd.Timestamp): The current simulation time.
            data_dict (dict): Dictionary of all DataFrames available.
                              Strategy should query data_dict[symbol].loc[:current_date]
            current_positions (dict): {symbol: shares}
            portfolio_value (float): Total Net Worth (Cash + Stock)
            
        Returns:
            list: List of order dictionaries.
                  Example: [{'symbol': 'AAPL', 'action': 'buy', 'cash_amount': 5000}]
        """
        pass
