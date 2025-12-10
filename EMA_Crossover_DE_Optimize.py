# ============================================
# EMA Crossover Strategy with DE Optimization
# General, Multi-Ticker Implementation
# ============================================
# ============================================
# EMA Crossover Strategy with DE Optimization
# General, Multi-Ticker Implementation
# --------------------------------------------
# Uses your helpers:
# - myutils.read_stock_data
# - trade_generator_BT.generate_trades_v1
# - PerformanceAnalysis1.all_performance_statistics / performance_statistic
# ============================================

import argparse
import time
import sys
import platform
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

import myutils as mu
import trade_generator_BT as tg
import PerformanceAnalysis1 as pa


# -----------------------
# Config defaults
# -----------------------
DEFAULT_TICKERS = ["RELIANCE"]
DEFAULT_DATA_PATH = r"C:/Users/nagar/Documents/IIQF PGPAT/Project/Quant-Project-1-Moving-Average-Backtest"
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_INIT_CAPITAL = 1_000_000
DEFAULT_RISKFREE = 0.05

# Trade engine settings (must be > 0 as per trade_generator_BT)
DEFAULT_MAX_CAPITAL_DEPLOY = 1.0     # deploy 100% of capital per position (since fixedcapital can be True)
DEFAULT_BUY_MARGIN = 1.0             # assume full cash product (no leverage) for longs
DEFAULT_SELL_MARGIN = 1.0            # assume full cash product (no leverage) for shorts
DEFAULT_SLIPPAGE = 0.002             # 0.2% slippage assumption
DEFAULT_BUY_TRANSCOST = 0.0001       # 0.01% buy cost
DEFAULT_SELL_TRANSCOST = 0.0005      # 0.05% sell cost

# For pure crossover exits we keep target/stop very large, but strictly > 0
DEFAULT_PNL_TARGET = 10.0            # effectively disabled in normal markets
DEFAULT_PNL_STOPLOSS = 5.0          # effectively disabled in normal markets

# DE bounds for (fast_ema, slow_ema)
DEFAULT_BOUNDS: List[Tuple[int, int]] = [(5, 30), (20, 200)]


@dataclass
class RunConfig:
    tickers: List[str]
    data_path: str
    train_ratio: float
    init_capital: float
    riskfree_rate: float
    bounds: List[Tuple[int, int]]
    fixedcapital: bool = True  # keep capital fixed between trades when True


# -----------------------
# Utilities
# -----------------------

def compute_years_from_index(df: pd.DataFrame, date_col: str = None) -> float:
    """Compute total time in years from first to last row.
    If index is datetime, use it; else use provided date_col.
    """
    if df is None or len(df) < 2:
        return 0.0
    try:
        if isinstance(df.index, pd.DatetimeIndex):
            start = df.index[0]
            end = df.index[-1]
        else:
            col = date_col or "Date"
            start = pd.to_datetime(df[col].iloc[0])
            end = pd.to_datetime(df[col].iloc[-1])
        days = max((end - start).days, 1)
        return days / 365.25
    except Exception:
        return 0.0


def ensure_date_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df is sorted by date ascending and has a Date index for consistency."""
    if df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])  # may already be datetime
            df = df.set_index("Date")
    df = df.sort_index()
    return df


# -----------------------
# Signals
# -----------------------

def generate_signal_ema(data: pd.DataFrame, fast_win: int, slow_win: int, close_column: str = "Close") -> pd.DataFrame:
    """Create EMA crossover signals.
    signal = +1 on bullish cross (fast above slow), -1 on bearish cross, else 0.
    Also returns buy_signal/sell_signal columns for compatibility.
    """
    if data is None or data.empty:
        return pd.DataFrame()
    if fast_win < 2 or slow_win <= fast_win:
        return pd.DataFrame()

    df = data.copy()
    if close_column not in df.columns:
        raise ValueError(f"close_column '{close_column}' not found in dataframe columns: {df.columns.tolist()}")

    # Compute EMAs
    df["EMA_fast"] = df[close_column].ewm(span=int(fast_win), adjust=False).mean()
    df["EMA_slow"] = df[close_column].ewm(span=int(slow_win), adjust=False).mean()

    # Define crossovers
    prev_fast = df["EMA_fast"].shift(1)
    prev_slow = df["EMA_slow"].shift(1)

    cond_buy = (df["EMA_fast"] > df["EMA_slow"]) & (prev_fast <= prev_slow)
    cond_sell = (df["EMA_fast"] < df["EMA_slow"]) & (prev_fast >= prev_slow)

    df["buy_signal"] = np.where(cond_buy, 1.0, 0.0)
    df["sell_signal"] = np.where(cond_sell, -1.0, 0.0)
    df["signal"] = df["buy_signal"] + df["sell_signal"]

    return df


# -----------------------
# Data split
# -----------------------

def split_train_test(df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()
    n = len(df)
    cut = max(min(int(n * train_ratio), n - 1), 1)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


# -----------------------
# Backtest wrapper
# -----------------------

def backtest_signaled_df(
    df_signaled: pd.DataFrame,
    init_capital: float,
    fixedcapital: bool = True,
    close_column: str = "Close",
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Run trade generator on a dataframe that already contains 'signal'.
    Returns (final_capital, trade_pnl, mtm_pnl).
    """
    if df_signaled is None or df_signaled.empty:
        return init_capital, np.array([]), np.array([])

    # ✅ Use generate_trades_v2 which supports all the parameters
    cap, trade_pnl, mtm_pnl = tg.generate_trades_v2(
        df_signaled,
        init_capital,
        DEFAULT_MAX_CAPITAL_DEPLOY,
        DEFAULT_BUY_MARGIN,
        DEFAULT_SELL_MARGIN,
        DEFAULT_PNL_TARGET,
        DEFAULT_PNL_STOPLOSS,
        fixedcapital=fixedcapital,
        datecol="Date",
        closecol=close_column,
        slippage=DEFAULT_SLIPPAGE,
        buy_transcost=DEFAULT_BUY_TRANSCOST,
        sell_transcost=DEFAULT_SELL_TRANSCOST,
    )
    
    # Ensure numpy arrays
    trade_pnl = np.array(trade_pnl) if not isinstance(trade_pnl, np.ndarray) else trade_pnl
    mtm_pnl = np.array(mtm_pnl) if not isinstance(mtm_pnl, np.ndarray) else mtm_pnl
    return cap, trade_pnl, mtm_pnl


# -----------------------
# Objective for DE (maximize Sharpe on TRAIN)
# -----------------------

def objective_sharpe(params: np.ndarray, df_train: pd.DataFrame, init_capital: float, riskfree_rate: float, fixedcapital: bool) -> float:
    fast, slow = int(round(params[0])), int(round(params[1]))
    # Penalize invalid combos
    if fast < 2 or slow <= fast:
        return 1e9

    df_sig = generate_signal_ema(df_train, fast, slow)
    if df_sig.empty:
        return 1e9

    years = compute_years_from_index(df_sig)
    if years <= 0:
        return 1e9

    _, _, mtm_pnl = backtest_signaled_df(df_sig, init_capital, fixedcapital=fixedcapital)
    if mtm_pnl is None or len(mtm_pnl) < 5:
        return 1e9

    sharpe = pa.performance_statistic("SharpeRatio", init_capital, mtm_pnl, riskfree_rate, years, fixedcapital)
    # Differential Evolution minimizes; return negative Sharpe to maximize it
    return -float(sharpe)



def run_optimization(df_train: pd.DataFrame, cfg: RunConfig) -> Tuple[int, int, Dict[str, Any]]:
    """Run DE to find best (fast, slow). Returns best_fast, best_slow, and scipy result dict.
    This version uses serial DE (no workers/updating) to avoid SciPy map-like errors.
    """
    # small bounds sanity check
    bounds = cfg.bounds if cfg.bounds else DEFAULT_BOUNDS

    def wrapped_objective(p):
        # ensure we always return a scalar float (penalty for invalid combos)
        try:
            val = objective_sharpe(p, df_train, cfg.init_capital, cfg.riskfree_rate, cfg.fixedcapital)
            return float(val)
        except Exception as e:
            # Return a very large penalty if objective crashes for some sample p
            # This prevents DE from blowing up while allowing us to inspect logs.
            print(f"[OBJECTIVE ERROR] params={p} -> {e}")
            return 1e9

    try:
        # SERIAL DE: no workers, no updating parameter — most robust across SciPy versions
        result = differential_evolution(
            func=wrapped_objective,
            bounds=bounds,
            maxiter=60,
            tol=1e-3,
            polish=True,
            strategy="best1bin",
            seed=42
        )
    except Exception as ex:
        # Bubble up the exception info in a structured way so caller can report it
        # We'll re-raise for now so the outer try/except captures it and logs the message,
        # but also print the traceback for debugging.
        import traceback
        print("[DE ERROR] differential_evolution raised an exception:")
        traceback.print_exc()
        raise

    best_fast, best_slow = int(round(result.x[0])), int(round(result.x[1]))
    return best_fast, best_slow, {"fun": float(result.fun), "nit": int(getattr(result, "nit", 0)),
                                 "nfev": int(getattr(result, "nfev", 0)), "message": str(getattr(result, "message", ""))}


# -----------------------
# Evaluation on TEST (full stats)
# -----------------------

def evaluate_params(df_test: pd.DataFrame, fast: int, slow: int, cfg: RunConfig) -> Dict[str, Any]:
    df_sig = generate_signal_ema(df_test, fast, slow)
    years = compute_years_from_index(df_sig)
    final_capital, trade_pnl, mtm_pnl = backtest_signaled_df(df_sig, cfg.init_capital, fixedcapital=cfg.fixedcapital)

    stats = pa.all_performance_statistics(
        cfg.init_capital,
        mtm_pnl if mtm_pnl is not None else np.array([]),
        cfg.riskfree_rate,
        years,
        fixedcapital=cfg.fixedcapital,
        mindatapoints=5,
    )
    # all_performance_statistics returns a dict per your module
    return {
        "FinalCapital": float(final_capital) if final_capital is not None else float(cfg.init_capital),
        "TradePnLCount": int(len(trade_pnl)) if trade_pnl is not None else 0,
        "MTMPnLCount": int(len(mtm_pnl)) if mtm_pnl is not None else 0,
        "Years": float(years),
        **(stats if isinstance(stats, dict) else {}),
    }


# -----------------------
# Main loop
# -----------------------

def process_single_ticker(ticker: str, cfg: RunConfig, close_column: str = "Close") -> Dict[str, Any]:
    # 1) Load
    df = mu.read_stock_data(ticker, cfg.data_path, date_column="Date", close_column=close_column, setindex_date_column=True)
    if df is None or df.empty:
        print(f"[WARN] No data for {ticker} at {cfg.data_path}")
        return {"Ticker": ticker, "Status": "NO_DATA"}
    df = ensure_date_index(df)

    # 2) Split
    df_train, df_test = split_train_test(df, cfg.train_ratio)
    if df_train.empty or df_test.empty:
        print(f"[WARN] Insufficient split for {ticker}")
        return {"Ticker": ticker, "Status": "INSUFFICIENT_DATA"}

    # 3) Optimize on TRAIN
    t0 = time.time()
    best_fast, best_slow, opt_info = run_optimization(df_train, cfg)
    t1 = time.time()

    # 4) Evaluate on TEST
    eval_stats = evaluate_params(df_test, best_fast, best_slow, cfg)

    # 5) Assemble row
    row: Dict[str, Any] = {
        "Ticker": ticker,
        "Best_FastEMA": int(best_fast),
        "Best_SlowEMA": int(best_slow),
        "Opt_FuncValue": float(opt_info.get("fun", np.nan)),
        "Opt_Iterations": int(opt_info.get("nit", 0)),
        "Opt_Evals": int(opt_info.get("nfev", 0)),
        "Opt_Message": opt_info.get("message", ""),
        "Opt_TimeSec": round(t1 - t0, 3),
        "Status": "OK",
    }

    # Merge evaluation statistics keys into row (if present)
    if isinstance(eval_stats, dict):
        # Flatten primary stats dict (keys like 'Trades', 'Sharpe Ratio', etc.)
        for k, v in eval_stats.items():
            row[str(k)] = v

    return row


import traceback

def run(cfg: RunConfig) -> pd.DataFrame:
    results: List[Dict[str, Any]] = []
    for tk in cfg.tickers:
        print(f"\n=== Processing {tk} ===")
        try:
            res = process_single_ticker(tk, cfg)
        except Exception as ex:
            tb = traceback.format_exc()
            print(f"[ERROR] {tk}: {ex}\nTraceback:\n{tb}")
            # store first N chars of traceback to avoid huge CSV fields
            short_tb = tb if len(tb) < 2000 else tb[:1999] + " ...[truncated]"
            res = {"Ticker": tk, "Status": f"ERROR: {str(ex)}", "Traceback": short_tb}
        results.append(res)

    df_out = pd.DataFrame(results)
    return df_out



# -----------------------
# CLI
# -----------------------

def parse_args() -> RunConfig:
    p = argparse.ArgumentParser(description="EMA Crossover + Differential Evolution Optimizer (multi-ticker)")
    p.add_argument("--tickers", type=str, default=",".join(DEFAULT_TICKERS), help="Comma-separated tickers e.g. RELIANCE,TCS,INFY")
    p.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH, help="Folder path containing <TICKER>.csv files")
    p.add_argument("--train_ratio", type=float, default=DEFAULT_TRAIN_RATIO, help="Train split ratio (0-1)")
    p.add_argument("--init_capital", type=float, default=DEFAULT_INIT_CAPITAL, help="Initial capital")
    p.add_argument("--riskfree", type=float, default=DEFAULT_RISKFREE, help="Risk-free rate per trade step (as used in your PA module)")
    p.add_argument("--bounds", type=str, default="5:30,20:200", help="EMA window bounds 'fast_lo:fast_hi,slow_lo:slow_hi'")
    p.add_argument("--fixedcapital", action="store_true", help="Use fixed capital between trades (default ON if set)")

    args = p.parse_args(args=None if sys.argv[0].endswith('.py') else [])

    # Parse tickers
    tickers = [x.strip() for x in args.tickers.split(",") if x.strip()]

    # Parse bounds
    def parse_pair(pair: str) -> Tuple[int, int]:
        lo, hi = pair.split(":")
        return (int(lo), int(hi))

    try:
        b_fast, b_slow = args.bounds.split(",")
        bounds = [parse_pair(b_fast), parse_pair(b_slow)]
    except Exception:
        bounds = DEFAULT_BOUNDS

    cfg = RunConfig(
        tickers=tickers or DEFAULT_TICKERS,
        data_path=args.data_path or DEFAULT_DATA_PATH,
        train_ratio=float(args.train_ratio),
        init_capital=float(args.init_capital),
        riskfree_rate=float(args.riskfree),
        bounds=bounds,
        fixedcapital=bool(args.fixedcapital) if args.fixedcapital else True,
    )
    return cfg


# -----------------------
# Entry
# -----------------------

def main():
    cfg = parse_args()

    print("\n>>> Config:")
    print(cfg)

    t0 = time.time()
    df_results = run(cfg)
    out_file = "optimal_params.csv"
    df_results.to_csv(out_file, index=False)
    t1 = time.time()

    print(f"\nSaved results to {out_file}")
    print(f"Total runtime: {round(t1 - t0, 2)} sec")
    print("Machine:", platform.platform(), "| Python:", sys.version.split(" ")[0])

from datetime import datetime

def generate_summary_excel(cfg, output_csv="optimal_params.csv", start_time=None, end_time=None):
    """
    Creates an Excel summary similar to 'EMA_Summary_Report_Template.pdf'.
    """
    total_runtime = round((end_time - start_time), 2) if (start_time and end_time) else None
    
    summary_data = {
        "Section": [
            "EMA Crossover Strategy - Summary Report",
            "",
            "Machine / Environment Details",
            "Platform",
            "Python Version",
            "Processor",
            "Machine",
            "",
            "Optimization Timing",
            "Start Time",
            "End Time",
            "Total Runtime (sec)",
            "",
            "Dataset Details",
            "Tickers",
            "Data Path",
            "Train/Test Split",
            "",
            "Results Summary",
            "Results saved to",
            "",
            "Prepared By"
        ],
        "Details": [
            "",
            "",
            "",
            platform.platform(),
            sys.version.split(" ")[0],
            platform.processor(),
            platform.machine(),
            "",
            "",
            datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S") if start_time else "",
            datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S") if end_time else "",
            total_runtime,
            "",
            "",
            ", ".join(cfg.tickers),
            cfg.data_path,
            f"{cfg.train_ratio:.2f}",
            "",
            "",
            output_csv,
            "",
            "" 
        ],
    }

    df_summary = pd.DataFrame(summary_data)
    excel_path = "EMA_Summary_Report.xlsx"
    df_summary.to_excel(excel_path, index=False)
    print(f"✅ Summary Excel saved: {excel_path}")


if __name__ == "__main__":
    t0 = time.time()
    cfg = parse_args()
    df_results = run(cfg)
    df_results.to_csv("optimal_params.csv", index=False)
    t1 = time.time()

    # Generate Excel Summary
    generate_summary_excel(cfg, output_csv="optimal_params.csv", start_time=t0, end_time=t1)



print("EMA Crossover DE Optimize script placeholder")
