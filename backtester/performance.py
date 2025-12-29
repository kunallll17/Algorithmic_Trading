import numpy as np
import pandas as pd

class Performance:
    """
    Calculates performance metrics for trading strategies.
    References:
    - Paper 2 (Geometric Mean, Sharpe Ratio)
    - Standard Financial Metrics
    """
    
    @staticmethod
    def calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
        """
        Calculates the Annualized Sharpe Ratio of a returns series.
        
        Args:
            returns (pd.Series or np.array): Period returns (e.g., daily).
            risk_free_rate (float): Annualized risk-free rate.
            periods_per_year (int): Trading periods per year (252 for daily).
        
        Returns:
            float: Annualized Sharpe Ratio.
        """
        if len(returns) < 2:
            return 0.0
        
        # Convert annual risk-free rate to per-period rate
        rf_per_period = risk_free_rate / periods_per_year
        
        excess_returns = returns - rf_per_period
        mean_excess_return = np.mean(excess_returns)
        std_dev = np.std(excess_returns, ddof=1)
        
        if std_dev == 0:
            return 0.0
        
        # Annualize the Sharpe Ratio
        # Sharpe = (mean_return * sqrt(periods)) / (std * sqrt(periods)) = mean/std * sqrt(periods)
        daily_sharpe = mean_excess_return / std_dev
        annualized_sharpe = daily_sharpe * np.sqrt(periods_per_year)
            
        return annualized_sharpe

    @staticmethod
    def calculate_max_drawdown(equity_curve):
        """
        Calculates the Maximum Drawdown of an equity curve.
        
        Args:
            equity_curve (pd.Series or np.array): Series of portfolio values/prices.
            
        Returns:
            float: Maximum percentage drop from a peak.
        """
        if len(equity_curve) < 1:
            return 0.0
            
        # Convert to numpy for speed
        values = np.array(equity_curve)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(values)
        
        # Calculate drawdown for each point
        drawdowns = (values - running_max) / running_max
        
        # Max drawdown is the minimum value (most negative)
        max_dd = np.min(drawdowns)
        
        return max_dd

    @staticmethod
    def calculate_geometric_mean_return(returns):
        """
        Calculates Geometric Mean Return (AGR) as suggested in Paper 2.
        AGR = (Product(1 + r))^(1/n) - 1
        
        Args:
            returns (pd.Series or np.array): Series of fractional returns (e.g., 0.01 for 1%).
            
        Returns:
            float: Geometric Mean Return per period.
        """
        if len(returns) < 1:
            return 0.0
            
        # Add 1 to returns to get growth factors
        growth_factors = 1 + returns
        
        # Ensure no negative values if returns < -100% (bankruptcy)
        if np.any(growth_factors <= 0):
            return -1.0 # Lost everything
            
        # Calculate geometric mean
        # Using exp(mean(log(x))) is numerically more stable for large n
        log_growth = np.log(growth_factors)
        mean_log_growth = np.mean(log_growth)
        geometric_mean = np.exp(mean_log_growth) - 1
        
        return geometric_mean

    @staticmethod
    def calculate_metrics(history_df):
        """
        Computes a comprehensive metrics dictionary from a history DataFrame.
        Expected columns: 'TotalValue' (indexed by time).
        """
        if history_df.empty:
            return {}
        
        equity_curve = history_df['TotalValue']
        
        # Calculate returns
        returns = equity_curve.pct_change().dropna()
        
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        sharpe = Performance.calculate_sharpe_ratio(returns)
        max_dd = Performance.calculate_max_drawdown(equity_curve)
        geom_mean = Performance.calculate_geometric_mean_return(returns)
        
        # Annualized volatility (assuming daily data, 252 days)
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0.0
        
        return {
            "Total Return": total_return,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_dd,
            "Geometric Mean Return": geom_mean,
            "Annualized Volatility": volatility
        }
