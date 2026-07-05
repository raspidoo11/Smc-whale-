"""Historical market-context provider for backtests and pretraining.

Live signals are enriched with funding rate / OI change / BTC trend / Fear &
Greed via market_context.py. Backtest-generated training rows used to carry
NEUTRAL values for all of these (the backtester is network-free at replay
time), which meant the model's most predictive features were unlearnable
offline. This module fixes that: it pre-fetches the HISTORICAL series for a
symbol/time range up front, then acts as a drop-in MARKET_CONTEXT_FN whose
lookups are aligned to the bar being replayed (via strategy.NOW_FN), so
synthetic trades carry the real values those features had at that moment.

  provider = HistoricalContextProvider()
  provider.preload(symbol, start_ms, end_ms)   # once per symbol
  simulate(..., context_provider=provider)      # replay is offline after this

All series are lookup-by-bisect ("most recent value at or before t").
spread_pct stays None — historical orderbooks can't be reconstructed.
Every fetch is best-effort: a failure degrades that feature to neutral,
exactly like live.
"""

import json
import logging
import time
import urllib.request
from bisect import bisect_right

import pandas as pd

logger = logging.getLogger(__name__)

FNG_HISTORY_URL = "https://api.alternative.me/fng/?limit=0"


def _series_lookup(timestamps, values, t_ms):
    """Most recent value at or before t_ms; None if t_ms precedes the series."""
    i = bisect_right(timestamps, t_ms) - 1
    return values[i] if i >= 0 else None


def compute_bos_series(df, lookback=10, confirm=2):
    """Vectorized bullish/bearish BOS over a whole OHLCV frame, with EXACTLY
    strategy.bullish_bos/bearish_bos semantics: swing level from a window that
    excludes the last `confirm` bars; all `confirm` closes must break it.
    Returns a +1/-1/0 int series (bull/bear/neutral)."""
    swing_high = df["high"].rolling(lookback).max().shift(confirm)
    swing_low = df["low"].rolling(lookback).min().shift(confirm)

    bull = pd.Series(True, index=df.index)
    bear = pd.Series(True, index=df.index)
    for i in range(confirm):
        bull &= df["close"].shift(i) > swing_high
        bear &= df["close"].shift(i) < swing_low

    trend = pd.Series(0, index=df.index)
    trend[bull.fillna(False)] = 1
    trend[bear.fillna(False)] = -1
    return trend


class HistoricalContextProvider:
    def __init__(self):
        # per-symbol: {"ts": [...], "vals": [...]} sorted ascending
        self._funding = {}
        self._oi = {}
        self._btc = None   # {"ts": [...], "vals": [...]}
        self._fng = None

    # ------------------------------------------------------------------
    # Preloading (network happens here, once)
    # ------------------------------------------------------------------

    def _fetch_paginated(self, fetch_page, since_ms, end_ms, what, symbol):
        rows, cursor = [], since_ms
        try:
            while cursor < end_ms:
                page = fetch_page(cursor)
                if not page:
                    break
                rows += page
                last = page[-1].get("timestamp") or 0
                if last <= cursor:
                    break
                cursor = last + 1
                time.sleep(0.05)
        except Exception as e:
            logger.warning(f"historical {what} fetch failed for {symbol}: {e}")
        return rows

    def preload(self, symbol, start_ms, end_ms):
        from exchange import get_exchange
        ex = get_exchange()

        # Funding rate history (Bybit: one point every 8h).
        rows = self._fetch_paginated(
            lambda since: ex.fetch_funding_rate_history(symbol, since=since, limit=200),
            start_ms - 8 * 3600 * 1000, end_ms, "funding", symbol,
        )
        pairs = sorted(
            (r["timestamp"], float(r["fundingRate"]))
            for r in rows if r.get("timestamp") and r.get("fundingRate") is not None
        )
        self._funding[symbol] = {
            "ts": [p[0] for p in pairs], "vals": [p[1] for p in pairs],
        }

        # Open interest history at 1h; lookup computes pct change vs the
        # previous point (live uses 12x5m ≈ 1h — same horizon).
        # NOTE: Bybit serves OI history NEWEST-FIRST and caps at 200/page, so
        # forward `since` pagination silently returns only the most recent
        # ~8 days. Paginate BACKWARD via endTime until the window is covered.
        rows = []
        try:
            cursor_end = end_ms
            for _ in range(60):  # hard cap: 60 pages = 500 days of 1h points
                page = ex.fetch_open_interest_history(
                    symbol, "1h", limit=200, params={"endTime": cursor_end}
                )
                if not page:
                    break
                page.sort(key=lambda r: r.get("timestamp") or 0)
                rows = page + rows
                oldest = page[0].get("timestamp") or 0
                if oldest <= start_ms - 2 * 3600 * 1000 or oldest >= cursor_end:
                    break
                cursor_end = oldest - 1
                time.sleep(0.05)
        except Exception as e:
            logger.warning(f"historical open interest fetch failed for {symbol}: {e}")
        by_ts = {}
        for r in rows:
            val = r.get("openInterestValue") or r.get("openInterestAmount") or 0
            if r.get("timestamp") and val:
                by_ts[r["timestamp"]] = float(val)  # dedupes overlapping pages
        pairs = sorted(by_ts.items())
        ts = [p[0] for p in pairs]
        vals = [p[1] for p in pairs]
        changes = [
            (vals[i] - vals[i - 1]) / vals[i - 1] * 100 if i > 0 and vals[i - 1] > 0 else None
            for i in range(len(vals))
        ]
        self._oi[symbol] = {"ts": ts, "vals": changes}

        logger.info(f"   context preloaded for {symbol}: "
                    f"{len(self._funding[symbol]['ts'])} funding pts, {len(ts)} OI pts")

    def preload_global(self, start_ms, end_ms):
        """BTC trend + Fear & Greed — shared across all symbols, load once."""
        try:
            from backtester import fetch_ohlcv_paginated
            n_bars = int((end_ms - start_ms) / (15 * 60 * 1000)) + 50
            btc = fetch_ohlcv_paginated("BTC/USDT:USDT", "15m", n_bars)
            trend = compute_bos_series(btc)
            self._btc = {
                "ts": [int(t.timestamp() * 1000) for t in btc.index],
                "vals": trend.tolist(),
            }
            logger.info(f"   context preloaded: BTC trend over {len(btc)} x 15m bars")
        except Exception as e:
            logger.warning(f"historical BTC trend preload failed: {e}")

        try:
            with urllib.request.urlopen(FNG_HISTORY_URL, timeout=15) as resp:
                data = json.loads(resp.read().decode())
            pairs = sorted(
                (int(r["timestamp"]) * 1000, float(r["value"]))
                for r in data.get("data", []) if r.get("timestamp") and r.get("value")
            )
            self._fng = {"ts": [p[0] for p in pairs], "vals": [p[1] for p in pairs]}
            logger.info(f"   context preloaded: {len(pairs)} Fear&Greed daily points")
        except Exception as e:
            logger.warning(f"historical Fear&Greed preload failed: {e}")

    # ------------------------------------------------------------------
    # MARKET_CONTEXT_FN interface — called by strategy.get_signal during
    # replay; "now" is the replayed bar's time via strategy.NOW_FN.
    # ------------------------------------------------------------------

    def __call__(self, symbol):
        import strategy
        t_ms = int(strategy.NOW_FN().timestamp() * 1000)

        def look(series):
            if not series or not series["ts"]:
                return None
            return _series_lookup(series["ts"], series["vals"], t_ms)

        return {
            "funding_rate": look(self._funding.get(symbol)),
            "oi_change_pct": look(self._oi.get(symbol)),
            "btc_trend": look(self._btc),
            "fng": look(self._fng),
            "spread_pct": None,  # historical orderbooks can't be reconstructed
        }
