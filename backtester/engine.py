import pandas as pd
import numpy as np
from .risk_manager import RiskManager
from .performance import Performance

class BacktesterEngine:
    """
    Event-driven backtester engine.
    Refactored for Multi-Asset support.
    """
    
    def __init__(self, initial_balance=10000.0, commission_percent=0.001):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.commission_percent = commission_percent
        
        # Positions: {'AAPL': 10, 'GOOGL': 5}
        self.positions = {} 
        self.history = []   
        
        self.risk_manager = RiskManager()
        
    def run(self, data_dict, strategy):
        """
        Runs the backtest over a portfolio of assets.
        
        Args:
            data_dict (dict): { 'Symbol': pd.DataFrame }. 
                              Each DF must have 'close' and datetime index.
            strategy (BaseStrategy): Strategy instance.
            
        Returns:
            dict: Performance metrics.
            pd.DataFrame: History log.
        """
        print(f"Starting Multi-Asset Backtest for: {strategy.name}")
        self.current_balance = self.initial_balance
        self.positions = {}
        self.history = []
        
        # 1. Align Data
        # Get the union of all indices to create a master timeline
        all_dates = pd.Index([])
        for symbol, df in data_dict.items():
            # Ensure index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                print(f"Warning: Index for {symbol} is not DatetimeIndex. Attempting convert.")
                df.index = pd.to_datetime(df.index)
            all_dates = all_dates.union(df.index)
            
        all_dates = all_dates.sort_values().unique()
        
        print(f"Timeline established: {len(all_dates)} trading days.")
        
        # 2. Main Event Loop
        for current_date in all_dates:
            
            # Construct the "Market State" for this date
            # It contains the current row for every asset (if it exists today)
            # and potentially past history
            
            # Optimization: 
            # Passing full history dict every tick is heavy. 
            # We'll pass a lightweight object or rely on the strategy to look up data 
            # provided we give it the current_date.
            
            # Let's pass the global data_dict and the current index pointer/date
            # The strategy should know how to slice it.
            
            # 2.1 Calculate Portfolio Value (Mark-to-Market)
            holdings_value = 0.0
            portfolio_snapshot = {}
            
            available_prices = {}
            
            for symbol, shares in self.positions.items():
                df = data_dict.get(symbol)
                if df is not None:
                    # Find price at or before current date
                    # Using 'asof' or simply reindexing beforehand is faster, but this is safe
                    try:
                        # Exact match
                        if current_date in df.index:
                            price = df.loc[current_date]['close']
                        else:
                            # Forward fill logic (use yesterday's price)
                            # For backtesting, we strictly use "known" prices. 
                            # If no trade today, maybe price didn't change or exchange closed.
                            # We'll lookup nearest past date
                            idx_loc = df.index.get_indexer([current_date], method='pad')[0]
                            if idx_loc != -1:
                                price = df.iloc[idx_loc]['close']
                            else:
                                price = 0.0 # Asset didn't exist yet?
                                
                        holdings_value += shares * price
                        available_prices[symbol] = price
                        
                    except KeyError:
                        pass
            
            total_value = self.current_balance + holdings_value
            
            # 2.2 Get Strategy Signals
            # Strategy receives: available market data up to this point
            # For efficiency, we just pass the date and the full dict, strategy handles lookback
            
            signals = strategy.generate_signal(current_date, data_dict, self.positions, total_value)
            
            # Signal format expected: 
            # [
            #   {'symbol': 'AAPL', 'action': 'buy', 'amount': 1000 (cash) or 'shares': 10},
            #   {'symbol': 'MSFT', 'action': 'sell', 'percent': 1.0}
            # ]
            
            # 2.3 Execute Signals
            if signals:
                for order in signals:
                    symbol = order.get('symbol')
                    action = order.get('action')
                    
                    df = data_dict.get(symbol)
                    if df is None: continue
                    
                    # Get execution price (Close of today, assuming we trade at Close)
                    # Realistically: Trade at Next Open. For this academic exercise, Close is standard.
                    if current_date in df.index:
                        exec_price = df.loc[current_date]['close']
                    else:
                        continue # Can't trade if no price
                        
                    if action == 'buy':
                        # Determine size (Cash amount or Share count)
                        cash_alloc = order.get('cash_amount', 0)
                        
                        # Apply risk management if not explicitly sized?
                        # For momentum, usually we allocate "10% of portfolio".
                        
                        if cash_alloc > 0:
                            # Commission
                            # cash_needed = trade_value + comm
                            # trade_value = shares * price
                            # comm = trade_value * comm_pct
                            # cost = trade_value * (1 + comm_pct)
                            
                            # max_trade_value = cash_alloc / (1 + comm_pct)
                            shares_to_buy = (cash_alloc / (1 + self.commission_percent)) // exec_price
                            
                            cost = shares_to_buy * exec_price
                            comm = cost * self.commission_percent
                            total_cost = cost + comm
                            
                            if total_cost <= self.current_balance and shares_to_buy > 0:
                                self.current_balance -= total_cost
                                self.positions[symbol] = self.positions.get(symbol, 0) + shares_to_buy
                                # print(f"[{current_date.date()}] BUY {symbol} {shares_to_buy} @ {exec_price:.2f}")

                    elif action == 'sell':
                        # Sell all or specific amount
                        current_shares = self.positions.get(symbol, 0)
                        if current_shares > 0:
                            revenue = current_shares * exec_price
                            comm = revenue * self.commission_percent
                            
                            self.current_balance += (revenue - comm)
                            del self.positions[symbol]
                            # print(f"[{current_date.date()}] SELL {symbol} {current_shares} @ {exec_price:.2f}")

                    elif action == 'close_all':
                         for sym, shares in list(self.positions.items()):
                             if shares > 0:
                                 # Need price
                                 if sym in data_dict and current_date in data_dict[sym].index:
                                     p = data_dict[sym].loc[current_date]['close']
                                     rev = shares * p
                                     comm = rev * self.commission_percent
                                     self.current_balance += (rev - comm)
                         self.positions = {}

            # 3. Record History
            self.history.append({
                'Date': current_date,
                'Cash': self.current_balance,
                'TotalValue': total_value
            })

        # Final Report
        history_df = pd.DataFrame(self.history).set_index('Date')
        metrics = Performance.calculate_metrics(history_df)
        
        return metrics, history_df
