"""Warm-start the model with backtest-generated training data.

Runs the offline backtester (the SAME live signal engine + realistic fills)
across a basket of symbols and prepends the resulting closed trades to trade
history, tagged with source="backtest". The trainer then:

  * trains them at BACKTEST_SAMPLE_WEIGHT (0.5x) so a simulated fill never
    outvotes a real one,
  * keeps real trades at the tail of the rolling window (backfilled rows are
    PREPENDED, so they age out first as real trades accumulate),
  * and strategy.ai_blend_weight ignores them entirely when deciding how much
    say the model gets — synthetic data warms up the model, it does not earn
    it trust.

Honest limitations (by design, not accident):
  * Market-context features (funding / OI / BTC trend / spread) are neutral in
    backtest rows — the backtester is network-free. The model learns those
    only from real trades.
  * Labels come from simulated fills (slippage + maker/taker fees modeled,
    but still a simulation).

Usage:
    python backfill_from_backtest.py                      # default basket
    python backfill_from_backtest.py --symbols "BTC/USDT:USDT" --candles 3000
    python backfill_from_backtest.py --replace            # refresh old backfill
    railway run python backfill_from_backtest.py          # against production
"""

import argparse
import json
import logging
import os
from datetime import datetime, timezone

from config import DATA_DIR

logger = logging.getLogger(__name__)

DEFAULT_SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT",
    "XRP/USDT:USDT", "ADA/USDT:USDT", "LINK/USDT:USDT", "AVAX/USDT:USDT",
]


def prepare_backfill(trades_by_symbol, max_trades):
    """Tag, chronologically sort, and cap simulated trades. Pure function —
    unit-tested without network. Keeps the MOST RECENT trades when capping,
    since they best reflect current market behavior."""
    tagged = []
    for symbol, trades in trades_by_symbol.items():
        for t in trades:
            tagged.append({**t, "symbol": t.get("symbol", symbol), "source": "backtest"})

    tagged.sort(key=lambda t: t.get("entry_time", ""))
    if max_trades and len(tagged) > max_trades:
        tagged = tagged[-max_trades:]
    return tagged


def run_backfill(symbols=None, candles=3000, max_trades=150, replace=False):
    """Core backfill, callable from the CLI below or from main.py at startup
    (BACKFILL_ON_START=true). Returns a one-line ASCII summary of what
    happened. Idempotent: if backfilled rows already exist and replace is
    False, it does nothing — safe to leave enabled across restarts."""
    symbols = symbols or DEFAULT_SYMBOLS

    # Existing-backfill guard runs BEFORE any network import/fetch.
    from trade_manager import get_trade_history, save_trade_history

    history = get_trade_history()
    existing_backtest = [t for t in history if t.get("source") == "backtest"]
    real_rows = [t for t in history if t.get("source") != "backtest"]

    if existing_backtest and not replace:
        msg = (f"History already contains {len(existing_backtest)} backfilled trades "
               f"- skipping (use replace to refresh)")
        logger.info(msg)
        return msg

    from backtester import simulate, fetch_ohlcv_paginated
    from historical_context import HistoricalContextProvider

    provider = HistoricalContextProvider()
    preloaded_global = False

    trades_by_symbol = {}
    for sym in symbols:
        try:
            logger.info(f"Backtesting {sym} ({candles} x 5m candles)...")
            df5 = fetch_ohlcv_paginated(sym, "5m", candles)
            df15 = fetch_ohlcv_paginated(sym, "15m", max(candles // 3, 200))

            # Historical funding / OI / BTC trend / Fear&Greed, so backfilled
            # rows carry REAL context features instead of neutral zeros.
            start_ms = int(df5.index[0].timestamp() * 1000)
            end_ms = int(df5.index[-1].timestamp() * 1000)
            if not preloaded_global:
                provider.preload_global(start_ms, end_ms)
                preloaded_global = True
            provider.preload(sym, start_ms, end_ms)

            trades, metrics = simulate(sym, df5, df15, use_xgboost=False,
                                       context_provider=provider)
            trades_by_symbol[sym] = trades
            logger.info(f"   {sym}: {len(trades)} simulated trades "
                        f"(win rate {metrics.get('win_rate', 0):.0%})")
        except Exception as e:
            logger.warning(f"   {sym} failed ({e}) — skipping")

    backfill = prepare_backfill(trades_by_symbol, max_trades)
    if not backfill:
        msg = "No simulated trades produced - nothing to backfill"
        logger.info(msg)
        return msg

    # Backup before touching history (mirrors reset_model_state.py).
    backup_dir = os.path.join(
        DATA_DIR, "backups", datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    )
    os.makedirs(backup_dir, exist_ok=True)
    with open(os.path.join(backup_dir, "trade_history.json"), "w") as f:
        json.dump(history, f, indent=2, default=str)

    # PREPEND: real trades stay at the tail, so the trainer's rolling window
    # sheds synthetic rows first as real history grows.
    save_trade_history(backfill + real_rows)

    wins = sum(1 for t in backfill if t.get("status") == "WIN")
    # ASCII-only summary: emoji crash on Windows cp1252 consoles.
    msg = (f"[OK] Backfilled {len(backfill)} simulated trades "
           f"({wins} W / {len(backfill) - wins} L) ahead of {len(real_rows)} real trades"
           + (f", replacing {len(existing_backtest)} old backfilled rows" if existing_backtest else "")
           + f". Backup: {backup_dir}")
    logger.info(msg)
    return msg


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logging.getLogger("strategy").setLevel(logging.WARNING)

    p = argparse.ArgumentParser(description="Backfill trade history with backtest-simulated trades.")
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--candles", type=int, default=3000)
    p.add_argument("--max-trades", type=int, default=150,
                   help="Cap on backfilled rows (rolling window is 220 — leave room for real trades)")
    p.add_argument("--replace", action="store_true",
                   help="Remove previously backfilled rows before adding the new batch")
    args = p.parse_args()

    print(run_backfill(
        symbols=args.symbols, candles=args.candles,
        max_trades=args.max_trades, replace=args.replace,
    ))
    print("Next scheduled retrain will pick them up automatically "
          "(or run: python -c \"from xgboost_trainer import train_model_incremental; train_model_incremental()\")")


if __name__ == "__main__":
    main()
