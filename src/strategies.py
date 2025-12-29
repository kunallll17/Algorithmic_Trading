"""
Trading strategy implementations for Backtrader.
"""
import backtrader as bt
import pandas as pd


class MomentumStrategy(bt.Strategy):
    """
    Momentum Strategy:
    - Buy when price > Moving Average
    - Sell when price < Moving Average
    """
    params = (
        ("period", 50),
        ("printlog", False),
    )
    
    def __init__(self):
        self.ma = bt.indicators.SimpleMovingAverage(
            self.data.close,
            period=self.params.period
        )
        self.order = None
        self.buyprice = None
        self.buycomm = None
    
    def log(self, txt, dt=None):
        """Logging function for strategy."""
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f"{dt.isoformat()}, {txt}")
    
    def notify_order(self, order):
        """Called when order status changes."""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f"BUY EXECUTED, Price: {order.executed.price:.2f}, "
                    f"Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}"
                )
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:
                self.log(
                    f"SELL EXECUTED, Price: {order.executed.price:.2f}, "
                    f"Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}"
                )
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order Canceled/Margin/Rejected")
        
        self.order = None
    
    def notify_trade(self, trade):
        """Called when trade is closed."""
        if not trade.isclosed:
            return
        
        self.log(f"OPERATION PROFIT, GROSS: {trade.pnl:.2f}, NET: {trade.pnlcomm:.2f}")
    
    def next(self):
        """Called for each bar."""
        if self.order:
            return
        
        if not self.position:
            if self.data.close[0] > self.ma[0]:
                self.log(f"BUY CREATE, {self.data.close[0]:.2f}")
                self.order = self.buy()
        else:
            if self.data.close[0] < self.ma[0]:
                self.log(f"SELL CREATE, {self.data.close[0]:.2f}")
                self.order = self.sell()


class MeanReversionStrategy(bt.Strategy):
    """
    Mean Reversion Strategy using RSI:
    - Buy when RSI < 30 (oversold)
    - Sell when RSI > 70 (overbought)
    """
    params = (
        ("rsi_period", 14),
        ("rsi_low", 30),
        ("rsi_high", 70),
        ("printlog", False),
    )
    
    def __init__(self):
        self.rsi = bt.indicators.RSI(
            self.data.close,
            period=self.params.rsi_period
        )
        self.order = None
        self.buyprice = None
        self.buycomm = None
    
    def log(self, txt, dt=None):
        """Logging function for strategy."""
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f"{dt.isoformat()}, {txt}")
    
    def notify_order(self, order):
        """Called when order status changes."""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f"BUY EXECUTED, Price: {order.executed.price:.2f}, "
                    f"Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}"
                )
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:
                self.log(
                    f"SELL EXECUTED, Price: {order.executed.price:.2f}, "
                    f"Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}"
                )
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order Canceled/Margin/Rejected")
        
        self.order = None
    
    def notify_trade(self, trade):
        """Called when trade is closed."""
        if not trade.isclosed:
            return
        
        self.log(f"OPERATION PROFIT, GROSS: {trade.pnl:.2f}, NET: {trade.pnlcomm:.2f}")
    
    def next(self):
        """Called for each bar."""
        if self.order:
            return
        
        if not self.position:
            if self.rsi[0] < self.params.rsi_low:
                self.log(f"BUY CREATE, {self.data.close[0]:.2f}, RSI: {self.rsi[0]:.2f}")
                self.order = self.buy()
        else:
            if self.rsi[0] > self.params.rsi_high:
                self.log(f"SELL CREATE, {self.data.close[0]:.2f}, RSI: {self.rsi[0]:.2f}")
                self.order = self.sell()


class PairsTradingStrategy(bt.Strategy):
    """
    Pairs Trading Strategy:
    - Trade the spread between two correlated stocks
    - Buy when spread is too low, sell when spread is too high
    """
    params = (
        ("lookback", 20),
        ("entry_threshold", 2.0),
        ("exit_threshold", 0.5),
        ("printlog", False),
    )
    
    def __init__(self):
        # Assuming two data feeds: data0 and data1
        if len(self.datas) < 2:
            raise ValueError("Pairs trading requires 2 data feeds")
        
        self.spread = self.data0.close - self.data1.close
        self.spread_mean = bt.indicators.SMA(self.spread, period=self.params.lookback)
        self.spread_std = bt.indicators.StandardDeviation(self.spread, period=self.params.lookback)
        self.zscore = (self.spread - self.spread_mean) / self.spread_std
        
        self.order = None
        self.buyprice = None
        self.buycomm = None
    
    def log(self, txt, dt=None):
        """Logging function for strategy."""
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f"{dt.isoformat()}, {txt}")
    
    def notify_order(self, order):
        """Called when order status changes."""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f"BUY EXECUTED, Price: {order.executed.price:.2f}, "
                    f"Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}"
                )
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:
                self.log(
                    f"SELL EXECUTED, Price: {order.executed.price:.2f}, "
                    f"Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}"
                )
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order Canceled/Margin/Rejected")
        
        self.order = None
    
    def notify_trade(self, trade):
        """Called when trade is closed."""
        if not trade.isclosed:
            return
        
        self.log(f"OPERATION PROFIT, GROSS: {trade.pnl:.2f}, NET: {trade.pnlcomm:.2f}")
    
    def next(self):
        """Called for each bar."""
        if self.order:
            return
        
        zscore = self.zscore[0]
        
        if not self.position:
            # Enter long when spread is too low (zscore < -entry_threshold)
            if zscore < -self.params.entry_threshold:
                self.log(f"BUY CREATE (Pairs), Z-Score: {zscore:.2f}")
                self.order = self.buy()
            # Enter short when spread is too high (zscore > entry_threshold)
            elif zscore > self.params.entry_threshold:
                self.log(f"SELL CREATE (Pairs), Z-Score: {zscore:.2f}")
                self.order = self.sell()
        else:
            # Exit when spread returns to mean (|zscore| < exit_threshold)
            if abs(zscore) < self.params.exit_threshold:
                self.log(f"CLOSE POSITION (Pairs), Z-Score: {zscore:.2f}")
                self.order = self.close()




