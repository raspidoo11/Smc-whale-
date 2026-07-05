"""Market-context enrichment for signals.

Fetches the features that price candles alone can't provide — the ones with
real predictive teeth on perp futures:

  * funding_rate   : positive = longs pay shorts (crowded long side)
  * oi_change_pct  : open-interest change over the last hour (conviction vs
                     squeeze fuel behind the move)
  * btc_trend      : +1 / -1 / 0 — BTC's own 15m structure; alts rarely
                     sustain a move against BTC
  * spread_pct     : current bid-ask spread as % of mid (execution quality)

strategy.py consumes this through an injectable seam (MARKET_CONTEXT_FN), so
the backtester stays network-free (it simply never wires this module in and
the features default to neutral). Everything is cached so a 30-symbol scan
doesn't triple its API usage; get_signal additionally only calls this for
setups that already passed the cheap SMC pre-filter.
"""

import json
import time
import logging
import urllib.request

logger = logging.getLogger(__name__)

_CACHE = {}
_TTL = {"funding": 300, "oi": 300, "btc": 300, "spread": 60, "fng": 3600}

FNG_URL = "https://api.alternative.me/fng/?limit=1"


def _cached(kind, key, ttl, fn):
    now = time.time()
    hit = _CACHE.get((kind, key))
    if hit and now - hit[0] < ttl:
        return hit[1]
    try:
        val = fn()
    except Exception as e:
        logger.debug(f"market_context {kind}({key}) failed: {e}")
        val = None
    _CACHE[(kind, key)] = (now, val)
    return val


def get_funding_rate(symbol):
    def fetch():
        from exchange import get_exchange
        data = get_exchange().fetch_funding_rate(symbol)
        rate = data.get("fundingRate")
        return float(rate) if rate is not None else None
    return _cached("funding", symbol, _TTL["funding"], fetch)


def get_oi_change_pct(symbol):
    """Open-interest % change over roughly the last hour (12 x 5m points)."""
    def fetch():
        from exchange import get_exchange
        rows = get_exchange().fetch_open_interest_history(symbol, "5m", limit=13)
        vals = [
            float(r.get("openInterestValue") or r.get("openInterestAmount") or 0)
            for r in rows
        ]
        vals = [v for v in vals if v > 0]
        if len(vals) < 2:
            return None
        return (vals[-1] - vals[0]) / vals[0] * 100
    return _cached("oi", symbol, _TTL["oi"], fetch)


def get_btc_trend():
    """BTC 15m structure via the same BOS logic the strategy uses:
    +1 bullish break, -1 bearish break, 0 neutral."""
    def fetch():
        from scanner import get_ohlcv
        from strategy import bullish_bos, bearish_bos
        df = get_ohlcv("BTC/USDT:USDT", "15m", 60)
        if df is None or len(df) < 20:
            return None
        if bullish_bos(df):
            return 1
        if bearish_bos(df):
            return -1
        return 0
    return _cached("btc", "BTCUSDT", _TTL["btc"], fetch)


def get_spread_pct(symbol):
    def fetch():
        from exchange import get_exchange
        t = get_exchange().fetch_ticker(symbol)
        bid, ask = t.get("bid") or 0, t.get("ask") or 0
        if bid <= 0 or ask <= 0 or ask < bid:
            return None
        mid = (bid + ask) / 2
        return (ask - bid) / mid * 100
    return _cached("spread", symbol, _TTL["spread"], fetch)


def get_fng():
    """Crypto Fear & Greed index (0=extreme fear, 100=extreme greed), daily,
    from alternative.me. Sentiment regime — contrarian gold at the extremes."""
    def fetch():
        with urllib.request.urlopen(FNG_URL, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        rows = data.get("data") or []
        return float(rows[0]["value"]) if rows else None
    return _cached("fng", "global", _TTL["fng"], fetch)


def get_market_context(symbol):
    """Full enrichment dict for a candidate signal. Missing values come back as
    None; the featurizer maps None to neutral defaults, so an API hiccup can
    never block a signal — it just degrades to candle-only features."""
    return {
        "funding_rate": get_funding_rate(symbol),
        "oi_change_pct": get_oi_change_pct(symbol),
        "btc_trend": get_btc_trend(),
        "spread_pct": get_spread_pct(symbol),
        "fng": get_fng(),
    }
