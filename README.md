A complete quantitative trading research project implementing:

✔ Exponential Moving Average (EMA) crossover strategy
✔ Custom-built event-driven backtesting engine
✔ Parameter optimization using Differential Evolution (DE)
✔ Full performance analysis (Sharpe, CAGR, Max Drawdown, Win Rate, etc.)
✔ Modular Python architecture (research-ready)

This project serves as Project 1 in my Quant Research portfolio.

Overview
Trend-following strategies like moving average crossovers are widely researched in systematic trading.
This project:
Computes EMA-based buy/sell signals
Generates trades using a deterministic trade engine
Optimizes EMA parameters over the training window
Evaluates performance on out-of-sample data
Produces detailed analytics and summary reports
The design focuses on clarity, modularity, and research extensibility.

Project Structure
Quant-Project-1-Moving-Average-Backtest/
│
├── src/
│   ├── EMA_Crossover_DE_Optimize.py     # Main optimizer + runner
│   ├── trade_generator_BT.py            # Custom trade engine
│   ├── PerformanceAnalysis1.py          # Performance metrics module
│   ├── myutils.py                       # Helper utilities
│
├── output/                              # Auto-generated backtest results
│   ├── optimal_params.csv
│   └── EMA_Summary_Report.xlsx
│
├── README.md
├── requirements.txt
└── .gitignore


Strategy Logic
Buy (Enter Long)
When:
EMA_fast > EMA_slow
and previous_fast <= previous_slow

Sell (Exit Long / Enter Short)
When:
EMA_fast < EMA_slow
and previous_fast >= previous_slow


Signals:
+1 → Long
-1 → Short
0 → Flat


Backtesting Engine
The engine is fully custom, built for clarity and control over trading logic:
Bar-by-bar simulation
Long/short position handling
Realized & mark-to-market PnL
Slippage & transaction cost modeling
Entry/exit event detection
Capital allocation rules (fixed capital option)
This makes the strategy’s behaviour transparent and reproducible.


Performance Metrics
Implemented in PerformanceAnalysis1.py:
CAGR
Sharpe Ratio
Calmar Ratio
Max Drawdown
Win Rate
Total Trades
Avg Profit / Avg Loss
Max Profit / Max Loss
Consecutive Losses
Volatility / Std Dev of returns
These metrics allow evaluating robustness, risk-adjusted performance, and sensitivity.


Parameter Optimization (Differential Evolution)
The strategy's EMA windows (fast, slow) are optimized using SciPy Differential Evolution, a global optimization algorithm.

Objective:
Maximise Sharpe Ratio on the training dataset.

DE Details:
Strategy: best1bin
Seed: 42 (reproducible)

Bounds:
fast ∈ [5, 30]
slow ∈ [20, 200]
Workflow:
Search across parameter space
Select best in-sample parameters
Apply to out-of-sample dataset
Report final performance


📊 Generated Outputs
1. optimal_params.csv
Contains optimized EMA parameters + Sharpe, runtime, and diagnostics for each ticker.

2. EMA_Summary_Report.xlsx
A detailed summary report including:
System information
Tickers
Train/Test split
Best parameters
Full performance metrics
Runtime statistics


How to Run This Project
Install dependencies
pip install -r requirements.txt

Run the optimizer
python src/EMA_Crossover_DE_Optimize.py \
    --tickers RELIANCE \
    --data_path ./data \
    --train_ratio 0.8

Output will be saved in:
output/optimal_params.csv
output/EMA_Summary_Report.xlsx


Future Improvements
I plan to extend this project with:
ATR-based position sizing
Stop-loss / trailing-stop variants
Regime filtering (ADX, volatility filters)
Walk-forward optimization
Multi-asset portfolio aggregation
Comparison with SMA, MACD, and RSI trend strategies
Transaction cost modelling
Live data execution pipeline
(If you want to see these versions, star ⭐ the repo!)


Author
Abhishek N
Quant Research | Python | Algorithmic Trading | CA
📧 Email: nagarajabhishek3@gmail.com
🔗 LinkedIn: https://linkedin.com/in/ca-abhishek-n-47ba09242
🐙 GitHub: https://github.com/nagarajabhishek3
