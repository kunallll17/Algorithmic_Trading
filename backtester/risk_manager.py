import numpy as np

class RiskManager:
    """
    Implements risk management and position sizing logic from 
    'A Study on Algorithmic Trading' and 'Perfect Order' strategies.
    
    Ref: Paper 4, Section 2.5 & 3.3.2
    """
    
    def __init__(self, risk_per_trade_percent=0.02, max_position_size_percent=1.0):
        """
        Args:
            risk_per_trade_percent (float): Fraction of account to risk per trade (e.g., 0.02 for 2%).
            max_position_size_percent (float): Max fraction of account for a single position.
        """
        self.risk_per_trade_percent = risk_per_trade_percent
        self.max_position_size_percent = max_position_size_percent

    def calculate_position_size(self, account_balance, entry_price, stop_loss_price):
        """
        Calculates the number of shares to buy based on risk percentage.
        
        Formula (Paper 4, Eq 5):
        Lot Size = (Account Balance * Risk %) / (Stop Loss Distance)
        
        Args:
            account_balance (float): Current trading capital.
            entry_price (float): Intended buy price.
            stop_loss_price (float): Exit price for stopping loss.
            
        Returns:
            float: Number of shares to purchase.
        """
        if entry_price <= stop_loss_price:
            # Short selling logic or invalid input protection
            # For this implementation, we assume long-only or pre-validated inputs for direction
            # If entry == stop, distance is 0, avoid division by zero
            return 0.0

        risk_amount = account_balance * self.risk_per_trade_percent
        stop_loss_distance = entry_price - stop_loss_price
        
        if stop_loss_distance <= 0:
            return 0.0
            
        raw_position_size = risk_amount / stop_loss_distance
        
        # Upper bound check: Do not exceed max allowed allocation
        max_position_value = account_balance * self.max_position_size_percent
        max_shares_by_value = max_position_value / entry_price
        
        final_position_size = min(raw_position_size, max_shares_by_value)
        
        return final_position_size

    def calculate_atr_stop_loss(self, entry_price, atr_value, multiplier=2.0, direction='long'):
        """
        Calculates stop loss price based on ATR (Average True Range).
        Paper 4 Suggestion: Stop Loss = Entry - (2 * ATR)
        
        Args:
            entry_price (float): Current price.
            atr_value (float): Current ATR value.
            multiplier (float): Multiplier for ATR (default 2.0).
            direction (str): 'long' or 'short'.
        
        Returns:
            float: Stop loss price.
        """
        if direction == 'long':
            return entry_price - (atr_value * multiplier)
        else:
            return entry_price + (atr_value * multiplier)
