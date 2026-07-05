"""Parameter optimizer — replaces hand-picked thresholds with measured ones.

Grid-searches the strategy's tunable knobs against historical data and ranks
combinations by profit factor (tie-break: expectancy), so settings like
"confidence 40" or "trail 0.5%" are chosen because they EARN, not because they
felt right.

Two-phase design (this is what makes a 144-combo grid finish in minutes, not
hours): in pure-SMC mode the signal decision depends only on the confidence
bar — none of the trail/retrace/entry-mode knobs change WHICH signals fire.
So phase 1 runs the full signal engine ONCE per symbol at the lowest
confidence in the grid and records every signal with its bar index; phase 2
replays fills and exits for each combo over that cached list, filtering by
confidence and recomputing the limit level per retrace value.

Usage:
    python optimize.py "BTC/USDT:USDT" "ETH/USDT:USDT" --candles 3000
    python optimize.py "SOL/USDT:USDT" --entry-modes limit market

Notes:
  * SMC (non-XGBoost) path only — the ML layer retrains continuously, so
    tuning against a frozen model would mislead.
  * A combo needs >= --min-trades across all symbols to be ranked; a 3-trade
    "100% win rate" is noise, not signal.
"""

import argparse
import itertools
import logging

import strategy
from backtester import (
    _simulate_exit,
    _simulate_limit_fill,
    compute_metrics,
    fetch_ohlcv_paginated,
    RISK_FRACTION,
    WARMUP,
    WINDOW,
)
from config import (
    START_BALANCE,
    SLIPPAGE_PCT,
    LIMIT_TTL_MINUTES,
    MAKER_FEE_RATE,
    TAKER_FEE_RATE,
)

logger = logging.getLogger(__name__)

GRID = {
    "CONFIDENCE_REQUIRED_SMC": [35, 40, 45, 50],
    "RETRACE_ATR_FRACTION": [0.25, 0.35, 0.50],
    "TRAIL_PERCENT": [0.3, 0.5, 0.8],
    "TRAIL_ACTIVATION_RATIO": [0.90, 0.97],
}

LIMIT_TTL_BARS = max(1, int(LIMIT_TTL_MINUTES / 5))


# ---------------------------------------------------------------------------
# Phase 1 — collect every signal once, at the loosest confidence in the grid
# ---------------------------------------------------------------------------

def collect_signals(symbol, df_5m, df_15m, min_confidence):
    """One full pass of the live signal engine. Returns [(bar_index, signal)].
    Exact for SMC mode: confidence = SMC score, which is history-independent,
    so collecting with empty history loses nothing."""
    orig_hist = strategy.get_trade_history
    orig_now = strategy.NOW_FN
    orig_xgb = strategy.USE_XGBOOST
    orig_conf = strategy.CONFIDENCE_REQUIRED_SMC
    strategy.USE_XGBOOST = False
    strategy.CONFIDENCE_REQUIRED_SMC = min_confidence
    strategy.recent_signals.clear()

    out = []
    index = df_5m.index
    try:
        strategy.get_trade_history = lambda: []
        for i in range(WARMUP, len(df_5m) - 1):
            bar_time = index[i]
            strategy.NOW_FN = lambda bt=bar_time: bt.to_pydatetime()
            df5_slice = df_5m.iloc[max(0, i - WINDOW): i + 1]
            df15_slice = df_15m[df_15m.index <= bar_time]
            if len(df15_slice) < 30 or len(df5_slice) < 30:
                continue
            s = strategy.get_signal(symbol, df15_slice.copy(), df5_slice.copy())
            if s:
                out.append((i, s))
    finally:
        strategy.get_trade_history = orig_hist
        strategy.NOW_FN = orig_now
        strategy.USE_XGBOOST = orig_xgb
        strategy.CONFIDENCE_REQUIRED_SMC = orig_conf
        strategy.recent_signals.clear()

    logger.info(f"   {symbol}: {len(out)} raw signals collected at confidence>={min_confidence}")
    return out


# ---------------------------------------------------------------------------
# Phase 2 — cheap replay of fills/exits for one parameter combo
# ---------------------------------------------------------------------------

def _limit_price_for(signal, retrace):
    """Recompute the retrace limit level for a given retrace fraction. FVG
    signals use the stored gap midpoint (retrace-independent); non-FVG signals
    use the ATR pullback with the same clamps strategy.py applies."""
    if signal.get("fvg"):
        return float(signal["limit_price"])
    entry, sl, atr = float(signal["entry"]), float(signal["sl"]), float(signal["atr"])
    if signal["direction"] == "LONG":
        lp = min(entry - atr * retrace, entry)
        return max(lp, sl + 0.25 * (entry - sl))
    lp = max(entry + atr * retrace, entry)
    return min(lp, sl - 0.25 * (sl - entry))


def replay(signals, highs, lows, closes, conf, retrace, trail, activation, entry_mode):
    """Replay one combo over the cached signals. Mirrors backtester.simulate's
    trade economics (slippage, fees, single position per symbol)."""
    equity = float(START_BALANCE)
    equity_curve = [equity]
    trades = []
    slip = SLIPPAGE_PCT / 100
    next_free_bar = 0

    for i, s in signals:
        if i < next_free_bar or s["confidence"] < conf:
            continue

        direction = s["direction"]
        sl = float(s["sl"])
        fill_offset = 0

        if entry_mode == "limit":
            entry = _limit_price_for(s, retrace)
            rr = float(s.get("rr_multiplier", 1.5))
            tp = entry + (entry - sl) * rr if direction == "LONG" else entry - (sl - entry) * rr
            touched = _simulate_limit_fill(direction, entry, highs[i + 1:], lows[i + 1:], LIMIT_TTL_BARS)
            if touched is None:
                next_free_bar = i + LIMIT_TTL_BARS
                continue
            fill_offset = touched
        else:
            entry = float(s["entry"]) * (1 + slip if direction == "LONG" else 1 - slip)
            tp = float(s["tp"])

        per_unit = abs(entry - sl)
        if per_unit <= 0:
            continue
        qty = equity * RISK_FRACTION / per_unit

        start = i + 1 + fill_offset
        exit_price, reason, bars_held = _simulate_exit(
            direction, entry, sl, tp,
            highs[start:], lows[start:], closes[start:],
            trail, activation,
        )
        if exit_price is None:
            break
        exit_price = exit_price * (1 - slip if direction == "LONG" else 1 + slip)

        gross = (exit_price - entry) * qty if direction == "LONG" else (entry - exit_price) * qty
        # Resting limit fills pay maker on entry; market entries + all exits pay taker.
        entry_fee_rate = MAKER_FEE_RATE if entry_mode == "limit" else TAKER_FEE_RATE
        pnl = gross - (entry * qty * entry_fee_rate + exit_price * qty * TAKER_FEE_RATE)
        equity += pnl
        equity_curve.append(equity)
        trades.append({
            "pnl": round(pnl, 6),
            "realized_r": round(((exit_price - entry) if direction == "LONG" else (entry - exit_price)) / per_unit, 4),
            "bars_held": bars_held + fill_offset,
        })
        next_free_bar = start + max(1, bars_held) + 1

    return trades, equity_curve


def run_grid(datasets, entry_modes, min_trades):
    """datasets: list of (symbol, signals, highs, lows, closes)."""
    names = list(GRID.keys())
    combos = list(itertools.product(*(GRID[n] for n in names)))
    total = len(combos) * len(entry_modes)
    logger.info(f"Replaying {total} combinations over {len(datasets)} symbol(s)...")

    results = []
    for entry_mode in entry_modes:
        for combo in combos:
            params = dict(zip(names, combo))
            all_trades, curve = [], [float(START_BALANCE)]
            for _, signals, highs, lows, closes in datasets:
                trades, _ = replay(
                    signals, highs, lows, closes,
                    conf=params["CONFIDENCE_REQUIRED_SMC"],
                    retrace=params["RETRACE_ATR_FRACTION"],
                    trail=params["TRAIL_PERCENT"],
                    activation=params["TRAIL_ACTIVATION_RATIO"],
                    entry_mode=entry_mode,
                )
                for t in trades:
                    curve.append(curve[-1] + t["pnl"])
                all_trades += trades

            m = compute_metrics(all_trades, curve)
            if m.get("n_trades", 0) >= min_trades:
                results.append({"entry_mode": entry_mode, **params, **m})

    results.sort(
        key=lambda r: (r.get("profit_factor", 0), r.get("expectancy_usd", 0)),
        reverse=True,
    )
    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    # get_signal prints a full SIGNAL SCAN block per setup — thousands of lines
    # over a collection pass. Keep the optimizer's own progress lines only.
    logging.getLogger("strategy").setLevel(logging.WARNING)

    p = argparse.ArgumentParser(description="Grid-search strategy parameters against historical data.")
    p.add_argument("symbols", nargs="+", help="ccxt symbols, e.g. 'BTC/USDT:USDT'")
    p.add_argument("--candles", type=int, default=3000)
    p.add_argument("--entry-modes", nargs="+", default=["limit", "market"], choices=["limit", "market"])
    p.add_argument("--min-trades", type=int, default=20)
    p.add_argument("--top", type=int, default=15)
    args = p.parse_args()

    min_conf = min(GRID["CONFIDENCE_REQUIRED_SMC"])
    datasets = []
    for sym in args.symbols:
        logger.info(f"Fetching {args.candles} x 5m candles for {sym}...")
        df5 = fetch_ohlcv_paginated(sym, "5m", args.candles)
        df15 = fetch_ohlcv_paginated(sym, "15m", max(args.candles // 3, 200))
        logger.info(f"   {sym}: {len(df5)} x 5m, {len(df15)} x 15m — collecting signals (one-time pass)...")
        signals = collect_signals(sym, df5, df15, min_conf)
        datasets.append((
            sym, signals,
            df5["high"].to_numpy(), df5["low"].to_numpy(), df5["close"].to_numpy(),
        ))

    results = run_grid(datasets, args.entry_modes, args.min_trades)

    if not results:
        total_signals = sum(len(d[1]) for d in datasets)
        print(f"\nNo combination produced >= {args.min_trades} trades "
              f"({total_signals} raw signals collected). Try more candles/symbols "
              f"or --min-trades lower.")
        return

    cols = ["entry_mode", *GRID.keys(), "n_trades", "win_rate",
            "profit_factor", "expectancy_usd", "avg_R", "max_drawdown_pct", "sharpe"]
    print("\n" + "=" * 130)
    print(f"TOP {min(args.top, len(results))} of {len(results)} qualifying combinations "
          f"(ranked by profit factor, then expectancy)")
    print("=" * 130)
    print(" | ".join(f"{c[:14]:>14}" for c in cols))
    print("-" * 130)
    for r in results[:args.top]:
        print(" | ".join(f"{str(r.get(c, ''))[:14]:>14}" for c in cols))
    print("=" * 130)
    best = results[0]
    print("\nBest combo as env vars (set these on Railway to adopt it):")
    print(f"  ENTRY_MODE={best['entry_mode']}")
    for name in GRID:
        print(f"  {name}={best[name]}")


if __name__ == "__main__":
    main()
