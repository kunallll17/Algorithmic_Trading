"""
Parameter optimization module using grid search.
"""
import pandas as pd
from typing import Dict, List, Any, Callable
from src.backtester import Backtester


class Optimizer:
    """Handles parameter optimization for trading strategies."""
    
    def __init__(self, backtester: Backtester):
        self.backtester = backtester
        self.optimization_results = []
    
    def grid_search(
        self,
        strategy_class,
        symbol: str,
        param_grid: Dict[str, List[Any]],
        metric: str = "sharpe_ratio"
    ) -> pd.DataFrame:
        """
        Perform grid search optimization.
        
        Args:
            strategy_class: Strategy class to optimize
            symbol: Stock symbol
            param_grid: Dictionary mapping parameter names to lists of values
            metric: Metric to optimize (default: sharpe_ratio)
        
        Returns:
            DataFrame with optimization results
        """
        import itertools
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))
        
        results = []
        
        print(f"Running grid search with {len(combinations)} combinations...")
        
        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))
            
            try:
                result = self.backtester.run_backtest(
                    strategy_class=strategy_class,
                    symbol=symbol,
                    strategy_params=params
                )
                
                # Add parameter values to result
                for name, value in zip(param_names, combo):
                    result[f"param_{name}"] = value
                
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    print(f"Completed {i + 1}/{len(combinations)} combinations...")
            
            except Exception as e:
                print(f"Error with parameters {params}: {e}")
                continue
        
        # Convert to DataFrame
        df_results = pd.DataFrame(results)
        
        if df_results.empty:
            print("No valid results from optimization!")
            return df_results
        
        # Sort by metric (descending)
        if metric in df_results.columns:
            df_results = df_results.sort_values(by=metric, ascending=False, na_last=True)
        
        self.optimization_results = df_results
        
        return df_results
    
    def get_best_params(self, metric: str = "sharpe_ratio") -> Dict:
        """Get the best parameters from optimization results."""
        if self.optimization_results.empty:
            raise ValueError("No optimization results available. Run grid_search first.")
        
        best_row = self.optimization_results.iloc[0]
        
        # Extract parameter columns
        param_cols = [col for col in self.optimization_results.columns if col.startswith("param_")]
        best_params = {}
        
        for col in param_cols:
            param_name = col.replace("param_", "")
            best_params[param_name] = best_row[col]
        
        return best_params
    
    def print_optimization_summary(self, top_n: int = 10):
        """Print summary of top N optimization results."""
        if self.optimization_results.empty:
            print("No optimization results available.")
            return
        
        print("\n" + "="*80)
        print("OPTIMIZATION SUMMARY")
        print("="*80)
        
        top_results = self.optimization_results.head(top_n)
        
        for idx, row in top_results.iterrows():
            print(f"\nRank {idx + 1}:")
            print(f"  Strategy: {row.get('strategy', 'N/A')}")
            
            # Print parameters
            param_cols = [col for col in top_results.columns if col.startswith("param_")]
            for col in param_cols:
                param_name = col.replace("param_", "")
                print(f"  {param_name}: {row[col]}")
            
            # Print key metrics
            print(f"  Sharpe Ratio: {row.get('sharpe_ratio', 'N/A'):.4f}" if pd.notna(row.get('sharpe_ratio')) else "  Sharpe Ratio: N/A")
            print(f"  Total Return: {row.get('total_return', 'N/A'):.2%}" if pd.notna(row.get('total_return')) else "  Total Return: N/A")
            print(f"  Max Drawdown: {row.get('max_drawdown', 'N/A'):.2%}" if pd.notna(row.get('max_drawdown')) else "  Max Drawdown: N/A")
            print(f"  Win Rate: {row.get('win_rate', 'N/A'):.2f}%" if pd.notna(row.get('win_rate')) else "  Win Rate: N/A")
        
        print("\n" + "="*80)




