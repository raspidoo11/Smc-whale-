"""Portfolio-level risk gate.

Per-trade sizing already caps the loss on any single position, but nothing
stopped the bot from stacking 10 highly-correlated alt longs — which is really
one big leveraged bet, not 10 independent ones. This module enforces
portfolio-wide limits before a new trade is allowed to open:

  * total open risk ("heat") as a % of balance
  * max positions in the same direction (net directional exposure)
  * max concurrent alt positions (correlation concentration proxy)
  * one position per symbol

`can_open_trade()` returns (allowed: bool, reason: str) so the caller can log
exactly why a candidate was rejected.
"""

import logging

from config import (
    MAX_PORTFOLIO_RISK_PCT,
    MAX_TRADES_PER_DIRECTION,
    MAX_ALT_POSITIONS,
)

logger = logging.getLogger(__name__)

# Liquid majors are treated as low-correlation anchors; everything else counts
# toward the alt-concentration cap.
MAJORS = {"BTC", "ETH", "BNB", "SOL"}

_QUOTES = ("USDT", "USDC", "USD", "FDUSD", "BUSD")


def base_asset(symbol):
    """'BTC/USDT:USDT' or 'BTCUSDT' -> 'BTC'."""
    s = str(symbol).upper()
    if "/" in s:
        return s.split("/")[0]
    s = s.split(":")[0]
    for q in _QUOTES:
        if s.endswith(q):
            return s[: -len(q)]
    return s


def trade_risk_usd(trade):
    """Capital at risk if this trade hits its stop: qty * |entry - sl|."""
    try:
        entry = float(trade.get("entry", 0))
        sl = float(trade.get("sl", 0))
        qty = float(trade.get("qty", 0))
        return abs(entry - sl) * qty
    except (TypeError, ValueError):
        return 0.0


def portfolio_risk_usd(open_trades):
    return sum(trade_risk_usd(t) for t in open_trades if t.get("status") == "OPEN")


def can_open_trade(new_trade, open_trades, balance):
    open_now = [t for t in open_trades if t.get("status") == "OPEN"]

    symbol = new_trade.get("symbol")
    if any(t.get("symbol") == symbol for t in open_now):
        return False, f"already have an open position on {symbol}"

    direction = new_trade.get("direction")
    same_dir = sum(1 for t in open_now if t.get("direction") == direction)
    if same_dir >= MAX_TRADES_PER_DIRECTION:
        return False, f"max {direction} positions reached ({same_dir}/{MAX_TRADES_PER_DIRECTION})"

    if base_asset(symbol) not in MAJORS:
        alt_count = sum(1 for t in open_now if base_asset(t.get("symbol")) not in MAJORS)
        if alt_count >= MAX_ALT_POSITIONS:
            return False, f"max alt positions reached ({alt_count}/{MAX_ALT_POSITIONS})"

    new_risk = trade_risk_usd(new_trade)
    total_risk = portfolio_risk_usd(open_now) + new_risk
    cap = float(balance) * MAX_PORTFOLIO_RISK_PCT / 100.0
    if total_risk > cap:
        return False, (
            f"portfolio heat {total_risk:.2f} USDT would exceed cap "
            f"{cap:.2f} USDT ({MAX_PORTFOLIO_RISK_PCT}% of balance)"
        )

    return True, "ok"
