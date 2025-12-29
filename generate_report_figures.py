"""
Generate polished figures for the final report.
Creates publication-quality visualizations.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.figsize'] = (12, 7)

# Create results directory if not exists
os.makedirs('results', exist_ok=True)

# Load the comparison results
metrics_df = pd.read_csv('results/comparison_metrics.csv')
print("Loaded metrics:")
print(metrics_df)

# Define consistent colors for each strategy
STRATEGY_COLORS = {
    'Buy & Hold': '#808080',           # Gray (benchmark - dashed)
    'Momentum (Paper 3)': '#D4A84B',   # Gold/Olive
    'Perfect Order (Paper 4)': '#2E8B57',  # Sea Green
    'TDQN AI (Paper 1)': '#1E90FF',    # Bright Blue (highlight)
    'RPMM (Paper 2)': '#4169E1',       # Royal Blue
    'Pairs Trading (Paper 2)': '#DA70D6'  # Orchid/Magenta
}

STRATEGY_LINESTYLES = {
    'Buy & Hold': '--',  # Dashed for benchmark
    'Momentum (Paper 3)': '-',
    'Perfect Order (Paper 4)': '-',
    'TDQN AI (Paper 1)': '-',
    'RPMM (Paper 2)': '-',
    'Pairs Trading (Paper 2)': '-'
}

STRATEGY_LINEWIDTHS = {
    'Buy & Hold': 2.0,
    'Momentum (Paper 3)': 1.5,
    'Perfect Order (Paper 4)': 1.5,
    'TDQN AI (Paper 1)': 3.0,  # Thick for highlight
    'RPMM (Paper 2)': 1.5,
    'Pairs Trading (Paper 2)': 1.5
}

def load_equity_data():
    """Re-run backtest to get equity curves data."""
    from main import load_data
    from backtester.engine import BacktesterEngine
    from strategies.buy_hold import BuyHoldStrategy
    from strategies.momentum import MomentumStrategy
    from strategies.perfect_order import PerfectOrderStrategy
    from strategies.pairs_trading import PairsTradingStrategy
    from strategies.rpmm import RPMMStrategy
    from strategies.tdqn_agent import TDQNAgent
    
    print("Loading data...")
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    data_dict = load_data(selected_symbols=tickers)
    
    strategies = {
        "Buy & Hold": BuyHoldStrategy(),
        "Momentum (Paper 3)": MomentumStrategy(),
        "Perfect Order (Paper 4)": PerfectOrderStrategy(),
        "TDQN AI (Paper 1)": TDQNAgent(),
        "RPMM (Paper 2)": RPMMStrategy(),
        "Pairs Trading (Paper 2)": PairsTradingStrategy(ticker_a='GOOGL', ticker_b='MSFT')
    }
    
    equity_curves = {}
    
    for name, strategy in strategies.items():
        print(f"Running {name}...")
        try:
            engine = BacktesterEngine(initial_balance=10000)
            metrics, history = engine.run(data_dict, strategy)
            equity_curves[name] = history['TotalValue']
        except Exception as e:
            print(f"Error with {name}: {e}")
    
    return equity_curves

def plot_equity_curves(equity_curves):
    """Create polished equity curves figure."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot order: Put TDQN and Buy & Hold last so they're on top
    plot_order = [
        'RPMM (Paper 2)',
        'Pairs Trading (Paper 2)',
        'Perfect Order (Paper 4)',
        'Momentum (Paper 3)',
        'Buy & Hold',
        'TDQN AI (Paper 1)'
    ]
    
    for name in plot_order:
        if name in equity_curves:
            series = equity_curves[name]
            ax.plot(series.index, series.values,
                   label=name,
                   color=STRATEGY_COLORS.get(name, 'black'),
                   linestyle=STRATEGY_LINESTYLES.get(name, '-'),
                   linewidth=STRATEGY_LINEWIDTHS.get(name, 1.5),
                   alpha=0.9 if name in ['TDQN AI (Paper 1)', 'Buy & Hold'] else 0.7)
    
    ax.set_title('Strategy Comparison: Equity Curves (2015-2024)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    
    # Format y-axis with dollar signs
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Add legend outside plot
    ax.legend(loc='upper left', framealpha=0.95, fontsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Highlight key events with annotations
    ax.axvspan(pd.Timestamp('2020-02-15'), pd.Timestamp('2020-04-01'), 
               alpha=0.15, color='red', label='COVID Crash')
    ax.axvspan(pd.Timestamp('2022-01-01'), pd.Timestamp('2022-12-31'),
               alpha=0.1, color='orange', label='2022 Bear Market')
    
    plt.tight_layout()
    plt.savefig('results/equity_curves_polished.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: results/equity_curves_polished.png")

def plot_performance_comparison(metrics_df):
    """Create bar chart comparing key metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    strategies = metrics_df['Strategy'].tolist()
    colors = [STRATEGY_COLORS.get(s, 'gray') for s in strategies]
    
    # 1. Total Return
    ax1 = axes[0, 0]
    returns = metrics_df['Total Return'] * 100  # Convert to percentage
    bars1 = ax1.bar(range(len(strategies)), returns, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_title('Total Return (%)', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(strategies)))
    ax1.set_xticklabels([s.split(' (')[0] for s in strategies], rotation=45, ha='right', fontsize=9)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_ylabel('Return (%)')
    # Add value labels
    for bar, val in zip(bars1, returns):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=9)
    
    # 2. Sharpe Ratio
    ax2 = axes[0, 1]
    sharpe = metrics_df['Sharpe Ratio']
    bars2 = ax2.bar(range(len(strategies)), sharpe, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_title('Sharpe Ratio (Annualized)', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(strategies)))
    ax2.set_xticklabels([s.split(' (')[0] for s in strategies], rotation=45, ha='right', fontsize=9)
    ax2.axhline(y=1.0, color='green', linestyle='--', linewidth=1, label='Good (>1.0)')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.legend(loc='upper right', fontsize=8)
    for bar, val in zip(bars2, sharpe):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Maximum Drawdown
    ax3 = axes[1, 0]
    drawdown = metrics_df['Max Drawdown'] * 100  # Convert to percentage
    bars3 = ax3.bar(range(len(strategies)), drawdown, color=colors, edgecolor='black', linewidth=0.5)
    ax3.set_title('Maximum Drawdown (%)', fontsize=14, fontweight='bold')
    ax3.set_xticks(range(len(strategies)))
    ax3.set_xticklabels([s.split(' (')[0] for s in strategies], rotation=45, ha='right', fontsize=9)
    ax3.set_ylabel('Drawdown (%)')
    ax3.axhline(y=-20, color='orange', linestyle='--', linewidth=1, label='Moderate Risk')
    ax3.legend(loc='lower right', fontsize=8)
    for bar, val in zip(bars3, drawdown):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 2,
                f'{val:.1f}%', ha='center', va='top', fontsize=9)
    
    # 4. Annualized Volatility
    ax4 = axes[1, 1]
    volatility = metrics_df['Annualized Volatility'] * 100  # Convert to percentage
    bars4 = ax4.bar(range(len(strategies)), volatility, color=colors, edgecolor='black', linewidth=0.5)
    ax4.set_title('Annualized Volatility (%)', fontsize=14, fontweight='bold')
    ax4.set_xticks(range(len(strategies)))
    ax4.set_xticklabels([s.split(' (')[0] for s in strategies], rotation=45, ha='right', fontsize=9)
    ax4.set_ylabel('Volatility (%)')
    for bar, val in zip(bars4, volatility):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Strategy Performance Comparison (2015-2024)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results/performance_metrics.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: results/performance_metrics.png")

def plot_risk_return_scatter(metrics_df):
    """Create risk-return scatter plot."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for idx, row in metrics_df.iterrows():
        strategy = row['Strategy']
        x = abs(row['Max Drawdown']) * 100  # Risk (positive for plotting)
        y = row['Total Return'] * 100  # Return
        
        color = STRATEGY_COLORS.get(strategy, 'gray')
        size = 200 if 'TDQN' in strategy else 120
        
        ax.scatter(x, y, s=size, c=color, edgecolors='black', linewidth=1.5,
                  label=strategy, alpha=0.8, zorder=5)
    
    # Add quadrant lines
    ax.axhline(y=300, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=25, color='gray', linestyle=':', alpha=0.5)
    
    # Add annotations for quadrants
    ax.text(10, 550, 'HIGH RETURN\nLOW RISK\n(Ideal)', fontsize=9, ha='center', 
           color='green', fontweight='bold', alpha=0.7)
    ax.text(40, 550, 'HIGH RETURN\nHIGH RISK', fontsize=9, ha='center',
           color='orange', fontweight='bold', alpha=0.7)
    ax.text(10, 150, 'LOW RETURN\nLOW RISK\n(Conservative)', fontsize=9, ha='center',
           color='blue', fontweight='bold', alpha=0.7)
    
    ax.set_xlabel('Maximum Drawdown (%) - Risk', fontsize=12)
    ax.set_ylabel('Total Return (%) - Reward', fontsize=12)
    ax.set_title('Risk-Return Profile of Trading Strategies', fontsize=14, fontweight='bold')
    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/risk_return_scatter.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: results/risk_return_scatter.png")

def plot_drawdown_analysis(equity_curves):
    """Plot drawdown over time for each strategy."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for name, equity in equity_curves.items():
        # Calculate drawdown
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max * 100
        
        ax.fill_between(drawdown.index, drawdown.values, 0,
                       alpha=0.3 if 'TDQN' not in name else 0.5,
                       color=STRATEGY_COLORS.get(name, 'gray'),
                       label=name)
    
    ax.set_title('Drawdown Analysis Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.legend(loc='lower left', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Highlight major drawdown periods
    ax.axvspan(pd.Timestamp('2020-02-15'), pd.Timestamp('2020-04-01'),
               alpha=0.2, color='red')
    ax.text(pd.Timestamp('2020-03-01'), -5, 'COVID\nCrash', fontsize=8, ha='center')
    
    plt.tight_layout()
    plt.savefig('results/drawdown_analysis.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: results/drawdown_analysis.png")

def create_summary_table_image(metrics_df):
    """Create a formatted table image for the report."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    
    # Prepare data
    table_data = []
    for _, row in metrics_df.iterrows():
        table_data.append([
            row['Strategy'].split(' (')[0],
            f"{row['Total Return']*100:.1f}%",
            f"{row['Sharpe Ratio']:.2f}",
            f"{row['Max Drawdown']*100:.1f}%",
            f"{row['Annualized Volatility']*100:.1f}%"
        ])
    
    columns = ['Strategy', 'Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Volatility']
    
    table = ax.table(cellText=table_data, colLabels=columns,
                    loc='center', cellLoc='center',
                    colColours=['#4472C4']*5)
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Highlight best values
    for i, row in enumerate(table_data):
        # Best return (highest)
        if 'TDQN' in metrics_df.iloc[i]['Strategy']:
            table[(i+1, 1)].set_facecolor('#C6EFCE')  # Green
            table[(i+1, 2)].set_facecolor('#C6EFCE')
        # Best drawdown (closest to 0)
        if 'RPMM' in metrics_df.iloc[i]['Strategy']:
            table[(i+1, 3)].set_facecolor('#C6EFCE')
    
    plt.title('Performance Summary Table', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('results/summary_table.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: results/summary_table.png")

if __name__ == "__main__":
    print("Generating polished report figures...")
    
    # Load equity curves data
    equity_curves = load_equity_data()
    
    # Generate all figures
    plot_equity_curves(equity_curves)
    plot_performance_comparison(metrics_df)
    plot_risk_return_scatter(metrics_df)
    plot_drawdown_analysis(equity_curves)
    create_summary_table_image(metrics_df)
    
    print("\nAll figures generated successfully!")
    print("Files saved in results/ directory:")
    print("  - equity_curves_polished.png")
    print("  - performance_metrics.png")
    print("  - risk_return_scatter.png")
    print("  - drawdown_analysis.png")
    print("  - summary_table.png")


