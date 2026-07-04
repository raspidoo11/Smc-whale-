"""Offline backtester.

Replays historical candles through the *live* signal engine (strategy.get_signal)
and a realistic fill simulator so strategy changes can be evaluated in minutes
instead of days of live paper trading. The simulation core (`simulate`) is pure
and network-free — it takes DataFrames in and returns trades + metrics — so it
is unit-testable; `backtest_symbol` wraps it with ccxt data fetching.

Fills model:
  * entry at the signal candle's close
  * SL / TP checked against subsequent candle highs/lows
  * if a candle straddles both SL and TP, SL is assumed first (conservative)
  * on TP, a ratcheting trailing stop takes over — mirroring live/paper behavior
  * taker fees applied on entry and exit

Usage:
    python backtester.py "BTC/USDT:USDT" --candles 3000 --tf5 5m --tf15 15m
"""

import argparse
import logging

import numpy as np
import pandas as pd

import strategy
from config import START_BALANCE

logger = logging.getLogger(__name__)

FEE_RATE = 0.0004          # taker fee per side
RISK_FRACTION = 0.05       # fraction of running equity risked per trade
TRAIL_PERCENT = 0.5
WARMUP = 60                # bars of context before the first possible signal
WINDOW = 250               # rolling df length handed to get_signal


def _simulate_exit(direction, sl, tp, highs, lows, closes, trail_pct):
    """Walk future candles; return (exit_price, reason, bars_held)."""
    trailing = False
    anchor = None
    n = len(highs)

    for k in range(n):
        h, l = highs[k], lows[k]

        if not trailing:
            if direction == "LONG":
                if l <= sl:
                    return sl, "Stop Loss Hit", k + 1
                if h >= tp:
                    trailing, anchor = True, h
            else:
                if h >= sl:
                    return sl, "Stop Loss Hit", k + 1
                if l <= tp:
                    trailing, anchor = True, l

        if trailing:
            if direction == "LONG":
                anchor = max(anchor, h)
                stop = anchor * (1 - trail_pct / 100)
                if l <= stop:
                    return stop, "Trailing Stop Hit", k + 1
            else:
                anchor = min(anchor, l)
                stop = anchor * (1 + trail_pct / 100)
                if h >= stop:
                    return stop, "Trailing Stop Hit", k + 1

    # Never exited within available data -> close at final candle's close.
    return float(closes[-1]) if n else None, "Open at data end", n


def simulate(symbol, df_5m, df_15m, use_xgboost=False):
    """Pure, offline replay. Returns (trades, metrics)."""
    # Feed the signal engine simulated history + candle-time, then restore.
    orig_hist = strategy.get_trade_history
    orig_now = strategy.NOW_FN
    orig_xgb = strategy.USE_XGBOOST
    strategy.USE_XGBOOST = use_xgboost

    sim_history = []
    equity = float(START_BALANCE)
    equity_curve = [equity]
    trades = []

    highs = df_5m["high"].to_numpy()
    lows = df_5m["low"].to_numpy()
    closes = df_5m["close"].to_numpy()
    index = df_5m.index

    try:
        strategy.get_trade_history = lambda: list(sim_history)

        i = WARMUP
        n = len(df_5m)
        while i < n - 1:
            bar_time = index[i]
            strategy.NOW_FN = lambda bt=bar_time: bt.to_pydatetime()

            df5_slice = df_5m.iloc[max(0, i - WINDOW): i + 1]
            df15_slice = df_15m[df_15m.index <= bar_time]
            if len(df15_slice) < 30 or len(df5_slice) < 30:
                i += 1
                continue

            signal = strategy.get_signal(symbol, df15_slice.copy(), df5_slice.copy())
            if not signal:
                i += 1
                continue

            entry = float(signal["entry"])
            sl = float(signal["sl"])
            tp = float(signal["tp"])
            direction = signal["direction"]

            risk_usd = equity * RISK_FRACTION
            per_unit = abs(entry - sl)
            if per_unit <= 0:
                i += 1
                continue
            qty = risk_usd / per_unit

            exit_price, reason, bars_held = _simulate_exit(
                direction, sl, tp,
                highs[i + 1:], lows[i + 1:], closes[i + 1:],
                TRAIL_PERCENT,
            )
            if exit_price is None:
                break

            gross = (exit_price - entry) * qty if direction == "LONG" else (entry - exit_price) * qty
            fees = (entry * qty + exit_price * qty) * FEE_RATE
            pnl = gross - fees
            equity += pnl
            equity_curve.append(equity)

            status = "WIN" if pnl > 0 else "LOSS"
            r_mult = ((exit_price - entry) if direction == "LONG" else (entry - exit_price)) / per_unit

            closed = {
                **signal,
                "symbol": symbol,
                "status": status,
                "exit_price": exit_price,
                "exit_reason": reason,
                "pnl": round(pnl, 4),
                "qty": qty,
                "bars_held": bars_held,
                "realized_r": round(r_mult, 4),
                "entry_time": str(bar_time),
            }
            trades.append(closed)
            sim_history.append(closed)

            # Single position per symbol: resume scanning after the exit bar.
            i += max(1, bars_held) + 1

    finally:
        strategy.get_trade_history = orig_hist
        strategy.NOW_FN = orig_now
        strategy.USE_XGBOOST = orig_xgb

    metrics = compute_metrics(trades, equity_curve)
    return trades, metrics


def compute_metrics(trades, equity_curve):
    n = len(trades)
    if n == 0:
        return {"n_trades": 0}

    pnls = np.array([t["pnl"] for t in trades], dtype=float)
    rs = np.array([t["realized_r"] for t in trades], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    equity = np.array(equity_curve, dtype=float)
    running_max = np.maximum.accumulate(equity)
    dd = (running_max - equity) / np.where(running_max == 0, 1, running_max)

    ret_std = pnls.std(ddof=1) if n > 1 else 0.0
    sharpe = float(pnls.mean() / ret_std * np.sqrt(n)) if ret_std > 0 else 0.0

    return {
        "n_trades": n,
        "win_rate": round(len(wins) / n, 3),
        "total_return_pct": round((equity[-1] / equity[0] - 1) * 100, 2),
        "final_equity": round(float(equity[-1]), 2),
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf"),
        "expectancy_usd": round(float(pnls.mean()), 4),
        "avg_R": round(float(rs.mean()), 3),
        "max_drawdown_pct": round(float(dd.max()) * 100, 2),
        "sharpe": round(sharpe, 2),
        "avg_bars_held": round(float(np.mean([t["bars_held"] for t in trades])), 1),
    }


def fetch_ohlcv_paginated(symbol, timeframe, total):
    """Fetch `total` candles via ccxt, paginating with `since`."""
    from exchange import get_exchange
    ex = get_exchange()
    tf_ms = ex.parse_timeframe(timeframe) * 1000
    now = ex.milliseconds()
    since = now - total * tf_ms
    rows = []
    while len(rows) < total:
        batch = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not batch:
            break
        rows += batch
        since = batch[-1][0] + tf_ms
        if len(batch) < 1000:
            break
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp")


def backtest_symbol(symbol, candles=3000, tf5="5m", tf15="15m", use_xgboost=False):
    logger.info(f"Fetching {candles} candles for {symbol} ({tf5}/{tf15})...")
    df_5m = fetch_ohlcv_paginated(symbol, tf5, candles)
    df_15m = fetch_ohlcv_paginated(symbol, tf15, max(candles // 3, 200))
    logger.info(f"Fetched {len(df_5m)} x {tf5}, {len(df_15m)} x {tf15}")
    return simulate(symbol, df_5m, df_15m, use_xgboost=use_xgboost)


def _print_report(symbol, trades, metrics):
    print("=" * 52)
    print(f"BACKTEST — {symbol}")
    print("=" * 52)
    if metrics.get("n_trades", 0) == 0:
        print("No trades generated.")
        return
    for k, v in metrics.items():
        print(f"  {k:<20}: {v}")
    print("-" * 52)
    print("  last 5 trades:")
    for t in trades[-5:]:
        print(f"    {t['entry_time']} {t['direction']:<5} {t['status']:<4} "
              f"R={t['realized_r']:+.2f} pnl={t['pnl']:+.2f} ({t['exit_reason']})")
    print("=" * 52)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description="Backtest the SMC Whale strategy.")
    p.add_argument("symbol", help="ccxt symbol, e.g. 'BTC/USDT:USDT'")
    p.add_argument("--candles", type=int, default=3000, help="number of 5m candles")
    p.add_argument("--tf5", default="5m")
    p.add_argument("--tf15", default="15m")
    p.add_argument("--xgboost", action="store_true", help="enable the AI probability layer")
    args = p.parse_args()

    trades, metrics = backtest_symbol(
        args.symbol, candles=args.candles, tf5=args.tf5, tf15=args.tf15, use_xgboost=args.xgboost
    )
    _print_report(args.symbol, trades, metrics)


if __name__ == "__main__":
    main()
