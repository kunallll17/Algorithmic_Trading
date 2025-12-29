# Algorithmic Trading Comparative Study

A comprehensive implementation and comparison of classical trading strategies versus Deep Reinforcement Learning for algorithmic trading.

## Project Overview

This project implements and backtests six distinct algorithmic trading strategies:
1. **Buy & Hold** (Benchmark)
2. **Momentum Strategy** (Jegadeesh & Titman, 1993)
3. **Perfect Order Strategy** (Technical Analysis)
4. **TDQN AI Agent** (Théate & Ernst, 2020)
5. **RPMM Strategy** (Hägg, 2023)
6. **Pairs Trading** (Hägg, 2023)

All strategies are evaluated on a unified dataset (AAPL, GOOGL, MSFT) over a 9-year period (2015-2023) using a custom event-driven backtesting engine.

## Project Structure

```
AlgoTrading_Project/
├── strategies/              # Trading strategy implementations
│   ├── base_strategy.py     # Abstract base class
│   ├── buy_hold.py          # Benchmark strategy
│   ├── momentum.py          # Jegadeesh & Titman (1993)
│   ├── perfect_order.py     # Technical Analysis
│   ├── pairs_trading.py     # Hägg (2023)
│   ├── rpmm.py              # Hägg (2023)
│   ├── tdqn_agent.py        # Théate & Ernst (2020)
│   └── data_augmentation.py # Trajectory generation for RL
├── backtester/              # Backtesting engine
│   ├── engine.py            # Main simulation loop
│   ├── performance.py       # Metrics calculation
│   └── risk_manager.py      # Position sizing
├── data/                    # Historical stock data (CSV files)
│   ├── AAPL.csv
│   ├── GOOGL.csv
│   └── MSFT.csv
├── models/                  # Trained ML models
│   ├── tdqn_model.zip       # Trained TDQN agent
│   └── ppo_AAPL.zip         # Additional model
├── results/                 # Output files
│   ├── comparison_metrics.csv
│   ├── backtest_results.csv
│   └── *.png                # Visualization figures
├── main.py                  # Entry point for individual strategies
├── run_comparison.py        # Run all strategies and compare
├── generate_report_figures.py  # Generate polished visualizations
├── download_data.py         # Data acquisition script
├── data_collector.py        # Data collection utility
├── requirements.txt         # Python dependencies
├── report.pdf               # Final report (3.5MB)
└── README.md                # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone or navigate to the project directory:**
   ```bash
   cd AlgoTrading_Project
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Download Data (Optional)

If you need to download fresh data:
```bash
python download_data.py
```

**Note:** The `data/` directory already contains pre-downloaded CSV files for AAPL, GOOGL, and MSFT.

### 2. Train TDQN Agent (If Model Missing)

If `models/tdqn_model.zip` doesn't exist, train the model:
```bash
python main.py --strategy tdqn --train --tickers AAPL,GOOGL,MSFT
```

**Expected Duration:** 5-15 minutes (depending on hardware)

### 3. Run Individual Strategy

Test a single strategy:
```bash
# Buy & Hold
python main.py --strategy buy_hold --tickers AAPL,GOOGL,MSFT

# Momentum
python main.py --strategy momentum --tickers AAPL,GOOGL,MSFT

# Perfect Order
python main.py --strategy perfect_order --tickers AAPL,GOOGL,MSFT

# Pairs Trading
python main.py --strategy pairs_trading --tickers GOOGL,MSFT

# RPMM
python main.py --strategy rpmm --tickers AAPL,GOOGL,MSFT

# TDQN (requires trained model)
python main.py --strategy tdqn --tickers AAPL,GOOGL,MSFT
```

### 4. Run Complete Comparison (Main Demo)

Run all strategies and generate comparison results:
```bash
python run_comparison.py
```

**Expected Duration:** 2-5 minutes

**Output:**
- `results/comparison_metrics.csv` - Performance metrics table
- `results/equity_curves.png` - Basic equity curves plot

### 5. Generate Polished Visualizations

Create publication-quality figures for the report:
```bash
python generate_report_figures.py
```

**Expected Duration:** 3-6 minutes

**Output:**
- `results/equity_curves_polished.png`
- `results/performance_metrics.png`
- `results/risk_return_scatter.png`
- `results/drawdown_analysis.png`
- `results/summary_table.png`

## Key Results

Based on the 9-year backtest (2015-2023):

| Strategy | Total Return | Sharpe Ratio | Max Drawdown |
|----------|--------------|--------------|--------------|
| **TDQN AI** | 653.5% | 1.02 | -34.4% |
| Buy & Hold | 612.4% | 1.01 | -34.7% |
| Momentum | 374.6% | 0.81 | -34.5% |
| Perfect Order | 238.6% | 0.92 | -23.5% |
| Pairs Trading | 271.7% | 0.90 | -26.8% |
| RPMM | 177.7% | 0.81 | -16.9% |

## Academic References

The strategies implemented are based on the following academic papers:

1. **Théate, T., & Ernst, D. (2020).** "An Application of Deep Reinforcement Learning to Algorithmic Trading." Expert Systems with Applications, 173, 114632.

2. **Hägg, A. (2023).** "A Study on Algorithmic Trading: Statistical Arbitrage and Reservation Price Methods." KTH Royal Institute of Technology (Bachelor's Thesis).

3. **Jegadeesh, N., & Titman, S. (1993).** "Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency." The Journal of Finance, 48(1), 65-91.

4. **Technical Analysis (Perfect Order):** Based on Moving Average trend-following strategies from technical analysis literature.

## Technical Details

### Backtesting Engine
- **Type:** Event-driven (prevents look-ahead bias)
- **Initial Capital:** $10,000
- **Transaction Cost:** 0.1% per trade
- **Execution:** Close price on signal day
- **Data Frequency:** Daily

### Performance Metrics
- Total Return
- Compound Annual Growth Rate (CAGR)
- Sharpe Ratio (annualized)
- Maximum Drawdown
- Geometric Mean Return
- Annualized Volatility
- Win Rate

## File Descriptions

### Core Scripts
- **main.py:** Entry point for running individual strategies
- **run_comparison.py:** Runs all strategies and generates comparison
- **generate_report_figures.py:** Creates polished visualizations
- **download_data.py:** Downloads historical data from Yahoo Finance

### Strategy Files
All strategies inherit from `base_strategy.py` and implement the `generate_signal()` method.

### Backtester Module
- **engine.py:** Main event-driven simulation loop
- **performance.py:** Calculates performance metrics
- **risk_manager.py:** Handles position sizing and risk management

## Troubleshooting

### Issue: "No module named 'stable_baselines3'"
**Solution:** 
```bash
pip install stable-baselines3[extra]
```

### Issue: "Model file not found" for TDQN
**Solution:** Train the model first:
```bash
python main.py --strategy tdqn --train --tickers AAPL,GOOGL,MSFT
```

### Issue: "No data loaded"
**Solution:** Verify CSV files exist in `data/` directory with correct format (columns: date, open, high, low, close, volume)

### Issue: Import errors
**Solution:** Ensure you're running scripts from the project root directory and all `__init__.py` files are present.

## Dependencies

Key Python packages (see `requirements.txt` for full list):
- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- stable-baselines3[extra] >= 2.1.0
- gymnasium >= 0.29.0
- yfinance >= 0.2.28
- ta >= 0.11.0

## Authors

- Kunal Sharma (2022610)
- Sarthak Kandpal (2022453)

**Supervised by:** Professor Pankaj Vajpayee

## License

This project is part of an Independent Study Project. All code and results are provided for academic purposes.




