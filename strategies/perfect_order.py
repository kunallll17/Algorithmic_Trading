from .base_strategy import BaseStrategy
import pandas as pd
import numpy as np
import ta

class PerfectOrderStrategy(BaseStrategy):
    """
    Implements 'Perfect Order' Strategy (Paper 4).
    Trend Following with Multiple SMA confirmations and ADX.
    """
    
    def __init__(self):
        super().__init__("Perfect Order (5 SMAs + ADX)")
        self.consecutive_days = 0
        self.last_signal = None 
        # For simplicity in this multi-asset engine, we track consecutive days PER ASSET.
        # But `consecutive_days` as a single int won't work for multi-asset.
        self.asset_states = {} # {symbol: {'consecutive': 0, 'trend': 'none'}}

    def generate_signal(self, current_date, data_dict, current_positions, portfolio_value):
        orders = []
        
        # Allocate per asset (naive equal weight for now, or dynamic)
        # Strategy says: Risk Management calculates size. 
        # Here we just trigger the entry, engine/risk_manager sizing logic handles quantity?
        # Our engine expects 'cash_amount' for buys.
        
        # Max concurrent positions? Let's say 5.
        max_positions = 5
        current_holdings_count = len(current_positions)
        
        for symbol, df in data_dict.items():
            if symbol not in self.asset_states:
                self.asset_states[symbol] = {'consecutive': 0, 'trend': 'none'}
                
            state = self.asset_states[symbol]
            
            # Need history for indicators
            # SMAs need 200 days. ADX needs 14 + smoothing.
            # Let's grab 300 days to be safe.
            
            try:
                # Efficient lookup of "upto now"
                curr_loc = df.index.get_indexer([current_date], method='pad')[0]
                if curr_loc < 250: continue # Not enough data
                
                # Slice last 250 rows
                history = df.iloc[curr_loc-250 : curr_loc+1].copy()
                
                # Check for NaNs using 'Close'
                if history['close'].isnull().values.any():
                    # clean or skip
                    history = history.dropna()
                    if len(history) < 200: continue

                # Calculate Indicators
                # Note: Calculating full TA series every step is inefficient for backtesting entire history loop.
                # Optimization: In a real backtester, we calc indicators ONCE for the whole dataframe at start.
                # But our interface passes `data_dict` which might be growing. 
                # For `BacktesterEngine` as implemented, `data_dict` contains FULL DFs.
                # So we SHOULD calc indicators once in `__init__` or `setup` if possible.
                # However, `generate_signal` is called daily.
                # Let's assume we can pre-calculate if we want, but to keep it simple and robust to "streaming",
                # we calculate on the fly (slow) or look up pre-calculated columns.
                
                # Fast path: Check if 'sma_200' exists in columns. If not, we can't assume.
                # Let's do a lightweight calculation on the tail, or assume user/main adds them.
                # For this implementation, let's Calculate on the fly (simple, but slow).
                
                # To make it faster: Only calculate last few values? TA lib usually computes whole series.
                # Better approach for this loop:
                # Retrieve the VALUES from pre-calculated columns if they exist.
                # If they don't, calculate them now.
                
                # Let's perform calculation on the specific window passed.
                
                close = history['close']
                high = history['high']
                low = history['low']
                
                # SMAs
                sma_10 = ta.trend.sma_indicator(close, window=10).iloc[-1]
                sma_20 = ta.trend.sma_indicator(close, window=20).iloc[-1]
                sma_50 = ta.trend.sma_indicator(close, window=50).iloc[-1]
                sma_100 = ta.trend.sma_indicator(close, window=100).iloc[-1]
                sma_200 = ta.trend.sma_indicator(close, window=200).iloc[-1]
                
                # ADX
                adx = ta.trend.ADXIndicator(high, low, close, window=14).adx().iloc[-1]
                
                # ATR (for risk management)
                atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]
                
                # 1. Check Perfect Order Conditions
                
                # BUY: 10 > 20 > 50 > 100 > 200 AND ADX >= 20
                is_buy_setup = (sma_10 > sma_20 > sma_50 > sma_100 > sma_200) and (adx >= 20)
                
                # SELL: 10 < 20 < 50 < 100 < 200 AND ADX >= 20
                is_sell_setup = (sma_10 < sma_20 < sma_50 < sma_100 < sma_200) and (adx >= 20)
                
                # 2. Confirmation Logic (5 bars)
                if is_buy_setup:
                    if state['trend'] == 'buy_setup':
                        state['consecutive'] += 1
                    else:
                        state['trend'] = 'buy_setup'
                        state['consecutive'] = 1
                elif is_sell_setup:
                    if state['trend'] == 'sell_setup':
                        state['consecutive'] += 1
                    else:
                        state['trend'] = 'sell_setup'
                        state['consecutive'] = 1
                else:
                    state['consecutive'] = 0
                    state['trend'] = 'none'
                
                # 3. Execution Logic
                
                # Entry Signal
                if state['trend'] == 'buy_setup' and state['consecutive'] >= 5:
                    if symbol not in current_positions:
                         if current_holdings_count < max_positions:
                             # Position Sizing
                             # Risk 2% of Account
                             # Stop Loss = 2 * ATR
                             stop_loss = close.iloc[-1] - (2 * atr)
                             stop_dist = close.iloc[-1] - stop_loss
                             
                             risk_amt = portfolio_value * 0.02
                             
                             # If stop dist is tiny, cap size
                             if stop_dist > 0:
                                 target_cash = (risk_amt / stop_dist) * close.iloc[-1]
                                 
                                 # Cap at 20% of portfolio
                                 max_alloc = portfolio_value * 0.20
                                 target_cash = min(target_cash, max_alloc)
                                 
                                 orders.append({
                                     'symbol': symbol,
                                     'action': 'buy',
                                     'cash_amount': target_cash,
                                     'stop_loss': stop_loss
                                 })
                                 
                # Exit Logic (Simple Trend Breakdown)
                # If we hold it, and setup breaks, do we sell?
                # Paper 4 suggests Trailing Stop or Trend Reversal.
                # Let's use the ATR Stop Loss implicit in "Risk Manager" 
                # BUT the engine handles "Risk Manager" for quantity, not exit monitoring yet?
                # A real engine monitors stops. Our simple engine doesn't have "pending stop orders".
                # We must self-monitor.
                
                if symbol in current_positions:
                    # Check Soft Stop
                    # We don't know our entry price here easily without position tracking enhancement.
                    # Simplified: If Trend reverses (Sell Setup confirmed), close Longs.
                    
                    if state['trend'] == 'sell_setup' and state['consecutive'] >= 2: # Fast exit
                         orders.append({'symbol': symbol, 'action': 'sell'})
                         
            except Exception as e:
                # print(f"Error {symbol}: {e}")
                continue
                
        return orders
