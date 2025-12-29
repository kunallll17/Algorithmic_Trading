"""
Backtesting framework using Backtrader.
"""
import backtrader as bt
import pandas as pd
import os
from typing import Dict, Any, Optional
from src.strategies import MomentumStrategy, MeanReversionStrategy, PairsTradingStrategy


class Backtester:
    """Main backtesting engine using Backtrader."""
    
    def __init__(
        self,
        initial_cash: float = 100000.0,
        commission: float = 0.001,
        data_dir: str = "data"
    ):
        self.initial_cash = initial_cash
        self.commission = commission
        self.data_dir = data_dir
    
    def prepare_data(self, symbol: str) -> bt.feeds.PandasData:
        """Load and prepare data for Backtrader."""
        filepath = os.path.join(self.data_dir, f"{symbol}.csv")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        df.columns = [col.lower() for col in df.columns]
        
        # Ensure required columns exist
        required_cols = ["open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Create Backtrader data feed
        data = bt.feeds.PandasData(
            dataname=df,
            datetime=None,
            open=0,
            high=1,
            low=2,
            close=3,
            volume=4,
            openinterest=-1
        )
        
        return data
    
    def run_backtest(
        self,
        strategy_class,
        symbol: str,
        strategy_params: Optional[Dict] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run a backtest for a given strategy.
        
        Args:
            strategy_class: Strategy class to use
            symbol: Stock symbol to backtest
            strategy_params: Parameters for the strategy
            start_date: Start date filter (optional)
            end_date: End date filter (optional)
        
        Returns:
            Dictionary with backtest results and metrics
        """
        cerebro = bt.Cerebro()
        
        # Add data
        data = self.prepare_data(symbol)
        if start_date or end_date:
            # Filter data by date if specified
            df = pd.read_csv(
                os.path.join(self.data_dir, f"{symbol}.csv"),
                index_col=0,
                parse_dates=True
            )
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
            
            data = bt.feeds.PandasData(
                dataname=df,
                datetime=None,
                open=0,
                high=1,
                low=2,
                close=3,
                volume=4,
                openinterest=-1
            )
        
        cerebro.adddata(data)
        
        # Add strategy
        if strategy_params:
            cerebro.addstrategy(strategy_class, **strategy_params)
        else:
            cerebro.addstrategy(strategy_class)
        
        # Set initial cash and commission
        cerebro.broker.setcash(self.initial_cash)
        cerebro.broker.setcommission(commission=self.commission)
        
        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        cerebro.addanalyzer(bt.analyzers.SQN, _name="sqn")
        
        # Run backtest
        print(f"Running backtest for {symbol} with {strategy_class.__name__}...")
        results = cerebro.run()
        
        # Extract results
        strat = results[0]
        
        # Calculate metrics
        final_value = cerebro.broker.getvalue()
        total_return = (final_value - self.initial_cash) / self.initial_cash
        
        sharpe_analysis = strat.analyzers.sharpe.get_analysis()
        drawdown_analysis = strat.analyzers.drawdown.get_analysis()
        returns_analysis = strat.analyzers.returns.get_analysis()
        trades_analysis = strat.analyzers.trades.get_analysis()
        
        results_dict = {
            "strategy": strategy_class.__name__,
            "symbol": symbol,
            "initial_cash": self.initial_cash,
            "final_value": final_value,
            "total_return": total_return,
            "sharpe_ratio": sharpe_analysis.get("sharperatio", None),
            "max_drawdown": drawdown_analysis.get("max", {}).get("drawdown", None),
            "max_drawdown_period": drawdown_analysis.get("max", {}).get("len", None),
            "annual_return": returns_analysis.get("rnorm100", None),
            "total_trades": trades_analysis.get("total", {}).get("total", 0),
            "winning_trades": trades_analysis.get("won", {}).get("total", 0),
            "losing_trades": trades_analysis.get("lost", {}).get("total", 0),
            "win_rate": (
                trades_analysis.get("won", {}).get("total", 0) /
                max(trades_analysis.get("total", {}).get("total", 1), 1)
            ) * 100,
            "avg_win": trades_analysis.get("won", {}).get("pnl", {}).get("average", 0),
            "avg_loss": trades_analysis.get("lost", {}).get("pnl", {}).get("average", 0),
            "profit_factor": trades_analysis.get("pnl", {}).get("net", {}).get("average", 0),
        }
        
        return results_dict
    
    def run_pairs_backtest(
        self,
        symbol1: str,
        symbol2: str,
        strategy_params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Run backtest for pairs trading strategy."""
        cerebro = bt.Cerebro()
        
        # Add both data feeds
        data1 = self.prepare_data(symbol1)
        data2 = self.prepare_data(symbol2)
        
        cerebro.adddata(data1, name=symbol1)
        cerebro.adddata(data2, name=symbol2)
        
        # Add strategy
        if strategy_params:
            cerebro.addstrategy(PairsTradingStrategy, **strategy_params)
        else:
            cerebro.addstrategy(PairsTradingStrategy)
        
        # Set initial cash and commission
        cerebro.broker.setcash(self.initial_cash)
        cerebro.broker.setcommission(commission=self.commission)
        
        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        
        # Run backtest
        print(f"Running pairs backtest for {symbol1} and {symbol2}...")
        results = cerebro.run()
        
        # Extract results
        strat = results[0]
        
        final_value = cerebro.broker.getvalue()
        total_return = (final_value - self.initial_cash) / self.initial_cash
        
        sharpe_analysis = strat.analyzers.sharpe.get_analysis()
        drawdown_analysis = strat.analyzers.drawdown.get_analysis()
        returns_analysis = strat.analyzers.returns.get_analysis()
        trades_analysis = strat.analyzers.trades.get_analysis()
        
        results_dict = {
            "strategy": "PairsTradingStrategy",
            "symbols": f"{symbol1}-{symbol2}",
            "initial_cash": self.initial_cash,
            "final_value": final_value,
            "total_return": total_return,
            "sharpe_ratio": sharpe_analysis.get("sharperatio", None),
            "max_drawdown": drawdown_analysis.get("max", {}).get("drawdown", None),
            "max_drawdown_period": drawdown_analysis.get("max", {}).get("len", None),
            "annual_return": returns_analysis.get("rnorm100", None),
            "total_trades": trades_analysis.get("total", {}).get("total", 0),
            "winning_trades": trades_analysis.get("won", {}).get("total", 0),
            "losing_trades": trades_analysis.get("lost", {}).get("total", 0),
            "win_rate": (
                trades_analysis.get("won", {}).get("total", 0) /
                max(trades_analysis.get("total", {}).get("total", 1), 1)
            ) * 100,
        }
        
        return results_dict




