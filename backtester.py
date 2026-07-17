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
from config import (
    START_BALANCE,
    MAX_HOLD_MINUTES,
    TRAIL_PERCENT,
    TRAIL_ACTIVATION_RATIO,
    ENTRY_MODE,
    LIMIT_TTL_MINUTES,
    SLIPPAGE_PCT,
    MAKER_FEE_RATE,
    TAKER_FEE_RATE,
    INVALIDATE_PENDING_ON_STRUCTURE,
)

logger = logging.getLogger(__name__)
RISK_FRACTION = 0.05       # fraction of running equity risked per trade
WARMUP = 60                # bars of context before the first possible signal
WINDOW = 250               # rolling df length handed to get_signal
LIMIT_TTL_BARS = max(1, int(LIMIT_TTL_MINUTES / 5))  # 5m bars per resting order


def _simulate_limit_fill(
    direction,
    limit_price,
    highs,
    lows,
    ttl_bars,
    invalidation_price=None,
):
    """Walk forward up to ttl_bars looking for price to trade through the
    resting limit level. Returns the 0-based bar offset of the fill, or None
    if the order expires unfilled or structure invalidates first (mirrors
    live trade_monitor behaviour)."""
    inv = invalidation_price
    for k in range(min(ttl_bars, len(highs))):
        if INVALIDATE_PENDING_ON_STRUCTURE and inv is not None:
            if direction == "LONG" and lows[k] < float(inv):
                return None
            if direction == "SHORT" and highs[k] > float(inv):
                return None
        if direction == "LONG" and lows[k] <= limit_price:
            return k
        if direction == "SHORT" and highs[k] >= limit_price:
            return k
    return None


def _simulate_exit(direction, entry, sl, tp, highs, lows, closes, trail_pct, activation_ratio,
                   max_hold_bars=0):
    """Walk future candles; return (exit_price, reason, bars_held).

    Mirrors live behavior: the hard TP is replaced by a trailing stop once price
    reaches `activation_ratio` of the way to TP (e.g. 97%), so a winner can run
    PAST tp instead of being capped there. max_hold_bars > 0 adds the time
    stop: a trade that has neither stopped out nor started trailing by then is
    closed at that bar's close (same eviction the live monitor applies).
    """
    # Price at which the trailing stop takes over (just short of TP).
    if direction == "LONG":
        activation = entry + (tp - entry) * activation_ratio
    else:
        activation = entry - (entry - tp) * activation_ratio

    trailing = False
    anchor = None
    n = len(highs)

    for k in range(n):
        h, l = highs[k], lows[k]

        if not trailing:
            if direction == "LONG":
                if l <= sl:
                    return sl, "Stop Loss Hit", k + 1
                if h >= activation:
                    trailing, anchor = True, h
            else:
                if h >= sl:
                    return sl, "Stop Loss Hit", k + 1
                if l <= activation:
                    trailing, anchor = True, l

        if trailing:
            # Floor/ceiling at entry once trailing is armed: a pure percent trail
            # can sit past breakeven on tight stops (trail distance > locked-in
            # progress to TP), which used to mark trail hits as LOSSes even
            # though the trade reached the arm zone. Match paper_trader /
            # trade_monitor.trail_stop_price (fee buffer applied on close).
            if direction == "LONG":
                anchor = max(anchor, h)
                stop = max(anchor * (1 - trail_pct / 100), entry)
                if l <= stop:
                    return stop, "Trailing Stop Hit", k + 1
            else:
                anchor = min(anchor, l)
                stop = min(anchor * (1 + trail_pct / 100), entry)
                if h >= stop:
                    return stop, "Trailing Stop Hit", k + 1

        # Time stop: same eviction the live monitor applies — only for trades
        # that are neither stopped out nor trailing by the max-hold bar.
        if max_hold_bars and not trailing and (k + 1) >= max_hold_bars:
            return float(closes[k]), "Time Stop (max hold)", k + 1

    # Never exited within available data -> close at final candle's close.
    return float(closes[-1]) if n else None, "Open at data end", n


def simulate(symbol, df_5m, df_15m, use_xgboost=False, context_provider=None):
    """Pure, offline replay. Returns (trades, metrics).

    context_provider: optional historical_context.HistoricalContextProvider
    (already preloaded). When given, replayed signals carry the REAL funding /
    OI / BTC-trend / Fear&Greed values of the bar being replayed instead of
    neutral defaults — so backfilled training rows teach the model its
    market-context features. Replay stays network-free either way.
    """
    # Feed the signal engine simulated history + candle-time, then restore.
    orig_hist = strategy.get_trade_history
    orig_now = strategy.NOW_FN
    orig_xgb = strategy.USE_XGBOOST
    orig_ctx = strategy.MARKET_CONTEXT_FN
    strategy.USE_XGBOOST = use_xgboost
    if context_provider is not None:
        strategy.MARKET_CONTEXT_FN = context_provider

    sim_history = []
    equity = float(START_BALANCE)
    equity_curve = [equity]
    trades = []
    unfilled = 0  # limit orders that expired without a fill

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

            direction = signal["direction"]
            sl = float(signal["sl"])
            slip = SLIPPAGE_PCT / 100
            fill_offset = 0  # bars between signal and actual entry

            if ENTRY_MODE == "limit":
                # Mirror live behavior: rest at the prediction zone, wait for a
                # touch, expire unfilled orders after the TTL, cancel if
                # structure invalidates. Limit fills get NO adverse slippage.
                entry = float(signal.get("limit_price", signal["entry"]))
                # Prefer the SL/TP already re-anchored to the limit in strategy.
                sl = float(signal.get("sl", sl))
                tp = float(signal.get("tp", entry))
                if abs(tp - entry) < 1e-12:
                    rr = float(signal.get("rr_multiplier", 1.5))
                    tp = (
                        entry + (entry - sl) * rr
                        if direction == "LONG"
                        else entry - (sl - entry) * rr
                    )
                inv = signal.get("invalidation_price", signal.get("structure_swing"))
                touched = _simulate_limit_fill(
                    direction,
                    entry,
                    highs[i + 1:],
                    lows[i + 1:],
                    LIMIT_TTL_BARS,
                    invalidation_price=inv,
                )
                if touched is None:
                    unfilled += 1
                    i += 1
                    continue
                fill_offset = touched
            else:
                # Market entry at the signal close, with adverse slippage.
                entry = float(signal["entry"]) * (1 + slip if direction == "LONG" else 1 - slip)
                tp = float(signal["tp"])

            risk_usd = equity * RISK_FRACTION
            per_unit = abs(entry - sl)
            if per_unit <= 0:
                i += 1
                continue
            qty = risk_usd / per_unit

            # Exit walk starts at the fill bar itself (a limit fill and an SL
            # breach can share a candle — conservatively, SL wins).
            start = i + 1 + fill_offset
            exit_price, reason, bars_held = _simulate_exit(
                direction, entry, sl, tp,
                highs[start:], lows[start:], closes[start:],
                TRAIL_PERCENT, TRAIL_ACTIVATION_RATIO,
                max_hold_bars=int(MAX_HOLD_MINUTES / 5) if MAX_HOLD_MINUTES > 0 else 0,
            )
            if exit_price is None:
                break
            bars_held += fill_offset

            # Stop-outs and trailing exits leave at market -> adverse slippage.
            exit_price = exit_price * (1 - slip if direction == "LONG" else 1 + slip)

            gross = (exit_price - entry) * qty if direction == "LONG" else (entry - exit_price) * qty
            # Maker/taker split: resting limit fills pay maker on entry; market
            # entries and ALL exits (SL/trailing fire at market) pay taker.
            entry_fee_rate = MAKER_FEE_RATE if ENTRY_MODE == "limit" else TAKER_FEE_RATE
            fees = entry * qty * entry_fee_rate + exit_price * qty * TAKER_FEE_RATE
            pnl = gross - fees
            equity += pnl
            equity_curve.append(equity)

            status = "WIN" if pnl > 0 else "LOSS"
            r_mult = ((exit_price - entry) if direction == "LONG" else (entry - exit_price)) / per_unit

            closed = {
                **signal,
                "symbol": symbol,
                # Override with the ACTUAL fill economics (limit entry price /
                # recomputed TP), not the signal-close values from **signal —
                # realized_r and any later training on this record depend on it.
                "entry": entry,
                "tp": tp,
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
        strategy.MARKET_CONTEXT_FN = orig_ctx

    metrics = compute_metrics(trades, equity_curve)
    metrics["unfilled_limit_orders"] = unfilled
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
