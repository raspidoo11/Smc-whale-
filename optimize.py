"""Parameter optimizer — replaces hand-picked thresholds with measured ones.

Grid-searches the strategy's tunable knobs against the offline backtester and
ranks combinations by profit factor (tie-break: expectancy), so settings like
"confidence 40" or "trail 0.5%" are chosen because they EARN, not because they
felt right. Data is fetched once per symbol and reused across the whole grid.

Usage:
    python optimize.py "BTC/USDT:USDT" "ETH/USDT:USDT" --candles 3000
    python optimize.py "SOL/USDT:USDT" --entry-modes limit market

Notes:
  * Results are per the SMC (non-XGBoost) path — the ML layer retrains
    continuously, so tuning against a frozen model would mislead.
  * A combo needs >= --min-trades across all symbols to be ranked; a 3-trade
    "100% win rate" is noise, not signal.
"""

import argparse
import itertools
import logging

import strategy
import backtester
from backtester import simulate, fetch_ohlcv_paginated, compute_metrics

logger = logging.getLogger(__name__)

# knob -> (module that holds it, values to try)
GRID = {
    "CONFIDENCE_REQUIRED_SMC": (strategy, [35, 40, 45, 50]),
    "RETRACE_ATR_FRACTION": (strategy, [0.25, 0.35, 0.50]),
    "TRAIL_PERCENT": (backtester, [0.3, 0.5, 0.8]),
    "TRAIL_ACTIVATION_RATIO": (backtester, [0.90, 0.97]),
}


def run_grid(datasets, entry_modes, min_trades):
    """datasets: list of (symbol, df_5m, df_15m). Returns ranked result rows."""
    names = list(GRID.keys())
    originals = {n: getattr(GRID[n][0], n) for n in names}
    orig_entry_mode = backtester.ENTRY_MODE
    results = []

    combos = list(itertools.product(*(GRID[n][1] for n in names)))
    total = len(combos) * len(entry_modes)
    logger.info(f"Running {total} combinations over {len(datasets)} symbol(s)...")

    try:
        for entry_mode in entry_modes:
            backtester.ENTRY_MODE = entry_mode
            for combo in combos:
                for name, value in zip(names, combo):
                    setattr(GRID[name][0], name, value)

                all_trades, equity_curve = [], [100.0]
                for sym, df5, df15 in datasets:
                    trades, _ = simulate(sym, df5, df15, use_xgboost=False)
                    for t in trades:
                        equity_curve.append(equity_curve[-1] + t["pnl"])
                    all_trades += trades

                m = compute_metrics(all_trades, equity_curve)
                if m.get("n_trades", 0) >= min_trades:
                    results.append({
                        "entry_mode": entry_mode,
                        **dict(zip(names, combo)),
                        **m,
                    })
    finally:
        for name, value in originals.items():
            setattr(GRID[name][0], name, value)
        backtester.ENTRY_MODE = orig_entry_mode

    results.sort(
        key=lambda r: (r.get("profit_factor", 0), r.get("expectancy_usd", 0)),
        reverse=True,
    )
    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description="Grid-search strategy parameters against the backtester.")
    p.add_argument("symbols", nargs="+", help="ccxt symbols, e.g. 'BTC/USDT:USDT'")
    p.add_argument("--candles", type=int, default=3000)
    p.add_argument("--entry-modes", nargs="+", default=["limit", "market"], choices=["limit", "market"])
    p.add_argument("--min-trades", type=int, default=20)
    p.add_argument("--top", type=int, default=15)
    args = p.parse_args()

    datasets = []
    for sym in args.symbols:
        logger.info(f"Fetching {args.candles} candles for {sym}...")
        df5 = fetch_ohlcv_paginated(sym, "5m", args.candles)
        df15 = fetch_ohlcv_paginated(sym, "15m", max(args.candles // 3, 200))
        datasets.append((sym, df5, df15))

    results = run_grid(datasets, args.entry_modes, args.min_trades)

    if not results:
        print(f"\nNo combination produced >= {args.min_trades} trades. "
              f"Try more candles or more symbols.")
        return

    cols = ["entry_mode", *GRID.keys(), "n_trades", "win_rate",
            "profit_factor", "expectancy_usd", "avg_R", "max_drawdown_pct", "sharpe"]
    print("\n" + "=" * 118)
    print(f"TOP {min(args.top, len(results))} of {len(results)} qualifying combinations "
          f"(ranked by profit factor, then expectancy)")
    print("=" * 118)
    print(" | ".join(f"{c[:14]:>14}" for c in cols))
    print("-" * 118)
    for r in results[:args.top]:
        print(" | ".join(f"{str(r.get(c, ''))[:14]:>14}" for c in cols))
    print("=" * 118)
    best = results[0]
    print("\nBest combo as env vars (set these on Railway to adopt it):")
    print(f"  ENTRY_MODE={best['entry_mode']}")
    for name in GRID:
        print(f"  {name}={best[name]}")


if __name__ == "__main__":
    main()
