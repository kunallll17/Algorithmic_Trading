"""
Visualization module for backtest results.
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from typing import Dict, List, Optional
import os

# Set style
try:
    plt.style.use("seaborn-v0_8-darkgrid")
except:
    try:
        plt.style.use("seaborn-darkgrid")
    except:
        plt.style.use("default")
sns.set_palette("husl")


class Visualizer:
    """Handles visualization of backtest results and metrics."""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
    
    def plot_equity_curve(
        self,
        equity_data: pd.DataFrame,
        title: str = "Equity Curve",
        save_path: Optional[str] = None
    ):
        """
        Plot equity curve over time.
        
        Args:
            equity_data: DataFrame with datetime index and equity values
            title: Plot title
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(equity_data.index, equity_data.values, linewidth=2, label="Portfolio Value")
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Portfolio Value ($)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Equity curve saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_drawdown(
        self,
        drawdown_data: pd.DataFrame,
        title: str = "Drawdown Curve",
        save_path: Optional[str] = None
    ):
        """Plot drawdown curve."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.fill_between(
            drawdown_data.index,
            drawdown_data.values,
            0,
            alpha=0.3,
            color="red",
            label="Drawdown"
        )
        ax.plot(drawdown_data.index, drawdown_data.values, linewidth=1, color="darkred")
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Drawdown (%)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Drawdown curve saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_rolling_sharpe(
        self,
        returns: pd.Series,
        window: int = 252,
        title: str = "Rolling Sharpe Ratio",
        save_path: Optional[str] = None
    ):
        """Plot rolling Sharpe ratio."""
        rolling_sharpe = returns.rolling(window=window).mean() / returns.rolling(window=window).std() * np.sqrt(252)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2, label=f"Rolling Sharpe ({window} days)")
        ax.axhline(y=1, color="r", linestyle="--", alpha=0.5, label="Sharpe = 1")
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Sharpe Ratio", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Rolling Sharpe plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_strategy_comparison(
        self,
        results_df: pd.DataFrame,
        metrics: List[str] = ["sharpe_ratio", "total_return", "max_drawdown", "win_rate"],
        save_path: Optional[str] = None
    ):
        """Plot comparison of multiple strategies."""
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
        
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            if metric not in results_df.columns:
                continue
            
            data = results_df[[metric, "strategy"]].dropna()
            if data.empty:
                continue
            
            axes[idx].bar(data["strategy"], data[metric], alpha=0.7)
            axes[idx].set_title(f"{metric.replace('_', ' ').title()}", fontsize=12, fontweight="bold")
            axes[idx].set_ylabel(metric.replace("_", " ").title(), fontsize=10)
            axes[idx].tick_params(axis="x", rotation=45)
            axes[idx].grid(True, alpha=0.3, axis="y")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Strategy comparison saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_optimization_results(
        self,
        optimization_df: pd.DataFrame,
        param_name: str,
        metric: str = "sharpe_ratio",
        save_path: Optional[str] = None
    ):
        """Plot optimization results for a single parameter."""
        param_col = f"param_{param_name}"
        
        if param_col not in optimization_df.columns:
            print(f"Parameter {param_name} not found in optimization results")
            return
        
        data = optimization_df[[param_col, metric]].dropna()
        data = data.sort_values(by=param_col)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(data[param_col], data[metric], marker="o", linewidth=2, markersize=8)
        ax.set_xlabel(param_name.replace("_", " ").title(), fontsize=12)
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
        ax.set_title(f"Optimization: {param_name.replace('_', ' ').title()} vs {metric.replace('_', ' ').title()}", 
                     fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Optimization plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def create_performance_dashboard(
        self,
        results_list: List[Dict],
        save_path: Optional[str] = None
    ):
        """Create a comprehensive performance dashboard."""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        results_df = pd.DataFrame(results_list)
        
        # 1. Strategy comparison - Sharpe Ratio
        ax1 = fig.add_subplot(gs[0, 0])
        if "sharpe_ratio" in results_df.columns:
            data = results_df[["strategy", "sharpe_ratio"]].dropna()
            ax1.bar(data["strategy"], data["sharpe_ratio"], alpha=0.7, color="steelblue")
            ax1.set_title("Sharpe Ratio Comparison", fontweight="bold")
            ax1.set_ylabel("Sharpe Ratio")
            ax1.tick_params(axis="x", rotation=45)
            ax1.grid(True, alpha=0.3, axis="y")
        
        # 2. Strategy comparison - Total Return
        ax2 = fig.add_subplot(gs[0, 1])
        if "total_return" in results_df.columns:
            data = results_df[["strategy", "total_return"]].dropna()
            ax2.bar(data["strategy"], data["total_return"] * 100, alpha=0.7, color="green")
            ax2.set_title("Total Return Comparison", fontweight="bold")
            ax2.set_ylabel("Return (%)")
            ax2.tick_params(axis="x", rotation=45)
            ax2.grid(True, alpha=0.3, axis="y")
        
        # 3. Strategy comparison - Max Drawdown
        ax3 = fig.add_subplot(gs[0, 2])
        if "max_drawdown" in results_df.columns:
            data = results_df[["strategy", "max_drawdown"]].dropna()
            ax3.bar(data["strategy"], data["max_drawdown"], alpha=0.7, color="red")
            ax3.set_title("Max Drawdown Comparison", fontweight="bold")
            ax3.set_ylabel("Drawdown (%)")
            ax3.tick_params(axis="x", rotation=45)
            ax3.grid(True, alpha=0.3, axis="y")
        
        # 4. Win Rate
        ax4 = fig.add_subplot(gs[1, 0])
        if "win_rate" in results_df.columns:
            data = results_df[["strategy", "win_rate"]].dropna()
            ax4.bar(data["strategy"], data["win_rate"], alpha=0.7, color="orange")
            ax4.set_title("Win Rate Comparison", fontweight="bold")
            ax4.set_ylabel("Win Rate (%)")
            ax4.tick_params(axis="x", rotation=45)
            ax4.grid(True, alpha=0.3, axis="y")
        
        # 5. Total Trades
        ax5 = fig.add_subplot(gs[1, 1])
        if "total_trades" in results_df.columns:
            data = results_df[["strategy", "total_trades"]].dropna()
            ax5.bar(data["strategy"], data["total_trades"], alpha=0.7, color="purple")
            ax5.set_title("Total Trades", fontweight="bold")
            ax5.set_ylabel("Number of Trades")
            ax5.tick_params(axis="x", rotation=45)
            ax5.grid(True, alpha=0.3, axis="y")
        
        # 6. Risk-Return Scatter
        ax6 = fig.add_subplot(gs[1, 2])
        if "total_return" in results_df.columns and "max_drawdown" in results_df.columns:
            data = results_df[["strategy", "total_return", "max_drawdown"]].dropna()
            scatter = ax6.scatter(
                data["max_drawdown"],
                data["total_return"] * 100,
                s=100,
                alpha=0.6,
                c=range(len(data)),
                cmap="viridis"
            )
            for idx, row in data.iterrows():
                ax6.annotate(
                    row["strategy"],
                    (row["max_drawdown"], row["total_return"] * 100),
                    fontsize=8
                )
            ax6.set_xlabel("Max Drawdown (%)")
            ax6.set_ylabel("Total Return (%)")
            ax6.set_title("Risk-Return Profile", fontweight="bold")
            ax6.grid(True, alpha=0.3)
        
        # 7-9. Summary table
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis("tight")
        ax7.axis("off")
        
        # Prepare summary table
        summary_cols = ["strategy", "sharpe_ratio", "total_return", "max_drawdown", "win_rate", "total_trades"]
        available_cols = [col for col in summary_cols if col in results_df.columns]
        summary_df = results_df[available_cols].copy()
        
        # Format numeric columns
        if "total_return" in summary_df.columns:
            summary_df["total_return"] = summary_df["total_return"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
        if "sharpe_ratio" in summary_df.columns:
            summary_df["sharpe_ratio"] = summary_df["sharpe_ratio"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
        if "max_drawdown" in summary_df.columns:
            summary_df["max_drawdown"] = summary_df["max_drawdown"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
        if "win_rate" in summary_df.columns:
            summary_df["win_rate"] = summary_df["win_rate"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
        
        table = ax7.table(
            cellText=summary_df.values,
            colLabels=summary_df.columns,
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        ax7.set_title("Performance Summary", fontsize=14, fontweight="bold", pad=20)
        
        plt.suptitle("Backtest Performance Dashboard", fontsize=16, fontweight="bold", y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Performance dashboard saved to {save_path}")
        else:
            plt.show()
        
        plt.close()

