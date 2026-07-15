"""Tests for the precision upgrades: retrace limit entries, the pending-order
lifecycle (paper), market-context features, and the backtester's limit-fill
simulation."""

import sys
import asyncio
import types
from datetime import datetime, timedelta, timezone

import pytest

# trade_monitor calls exchange.get_exchange() at import time (network). Stub
# the exchange module BEFORE importing it so tests stay offline.
if "exchange" not in sys.modules or not hasattr(sys.modules.get("exchange"), "_TEST_STUB"):
    stub = types.ModuleType("exchange")
    stub._TEST_STUB = True
    stub.get_exchange = lambda: None
    stub.get_trade_client = lambda: None
    sys.modules["exchange"] = stub

import strategy
import trade_monitor
from xgboost_trainer import extract_pro_features_from_trade
from backtester import _simulate_limit_fill
from risk_manager import can_open_trade
from test_feature_parity import _breakout_df


@pytest.fixture(autouse=True)
def _clear_recent_signals():
    strategy.recent_signals.clear()
    yield
    strategy.recent_signals.clear()


# ---------------------------------------------------------------------------
# Retrace limit price on the signal
# ---------------------------------------------------------------------------

def test_signal_carries_limit_price_below_close_for_long(monkeypatch):
    monkeypatch.setattr(strategy, "get_trade_history", lambda: [])
    df = _breakout_df()
    s = strategy.get_signal("BTCUSDT", df.copy(), df.copy())
    assert s is not None and s["direction"] == "LONG"
    # Retrace entry: below the signal close, but safely above the stop.
    assert s["limit_price"] <= s["entry"]
    assert s["limit_price"] >= s["sl"] + 0.25 * (s["entry"] - s["sl"]) - 1e-9
    assert "rr_multiplier" in s


def test_signal_market_context_defaults_are_none_safe(monkeypatch):
    monkeypatch.setattr(strategy, "get_trade_history", lambda: [])
    # Default seam returns {} -> context fields present but None.
    df = _breakout_df()
    s = strategy.get_signal("BTCUSDT", df.copy(), df.copy())
    assert s is not None
    assert "funding_rate" in s and "btc_trend" in s and "symbol_win_rate" in s


def test_signal_symbol_win_rate_from_history(monkeypatch):
    hist = (
        [{"symbol": "BTCUSDT", "status": "WIN", "pnl": 1.0}] * 6
        + [{"symbol": "BTCUSDT", "status": "LOSS", "pnl": -1.0}] * 2
        + [{"symbol": "OTHERUSDT", "status": "LOSS", "pnl": -1.0}] * 5
    )
    monkeypatch.setattr(strategy, "get_trade_history", lambda: hist)
    df = _breakout_df()
    s = strategy.get_signal("BTCUSDT", df.copy(), df.copy())
    assert s is not None
    assert s["symbol_win_rate"] == pytest.approx(6 / 8)


# ---------------------------------------------------------------------------
# Featurizer: market-context features
# ---------------------------------------------------------------------------

def test_featurizer_market_context_features():
    trade = {
        "direction": "LONG", "entry": 100.0, "sl": 98.0, "tp": 104.0,
        "funding_rate": 0.0005, "oi_change_pct": 3.2, "btc_trend": 1,
        "spread_pct": 0.02, "symbol_win_rate": 0.7,
    }
    f = extract_pro_features_from_trade(trade)
    assert f["funding_rate"] == pytest.approx(0.0005)
    assert f["oi_change_pct"] == pytest.approx(3.2)
    assert f["btc_aligned"] == 1          # LONG with btc_trend +1
    assert f["funding_vs_direction"] == pytest.approx(0.0005)
    assert f["symbol_win_rate"] == pytest.approx(0.7)


def test_featurizer_handles_missing_context():
    trade = {"direction": "SHORT", "entry": 100.0, "sl": 102.0, "tp": 96.0,
             "funding_rate": None, "btc_trend": None}
    f = extract_pro_features_from_trade(trade)
    assert f["funding_rate"] == 0.0
    assert f["btc_trend"] == 0.0
    assert f["btc_aligned"] == 0
    assert f["symbol_win_rate"] == 0.5


# ---------------------------------------------------------------------------
# Pending-order lifecycle (paper)
# ---------------------------------------------------------------------------

async def _noop_alert(msg):
    return None


def _pending_trade(direction="LONG", limit=99.0, minutes_ago=0):
    placed = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    # SL / invalidation on the correct side of the limit so structure
    # invalidation never false-cancels healthy pending orders.
    if direction == "LONG":
        sl, inv, tp = 97.0, 97.5, 103.0
    else:
        sl, inv, tp = 103.0, 102.5, 97.0
    return {
        "symbol": "BTC/USDT:USDT", "direction": direction, "status": "PENDING",
        "entry": limit, "sl": sl, "tp": tp, "qty": 1.0, "trade_no": 7,
        "invalidation_price": inv, "structure_swing": inv,
        "placed_at": placed.isoformat(),
    }


def test_paper_limit_fills_on_touch(monkeypatch):
    monkeypatch.setattr(trade_monitor, "EXECUTE_TRADES", False)
    monkeypatch.setattr(trade_monitor, "send_alert", _noop_alert)
    trade = _pending_trade(limit=99.0, minutes_ago=5)
    book = [trade]
    changed = asyncio.run(
        trade_monitor._handle_pending_order(trade, trade["symbol"], "LONG", 98.9, book)
    )
    assert changed is True
    assert trade["status"] == "OPEN"
    assert "filled_at" in trade
    assert trade in book  # filled orders stay in the book as open positions


def test_paper_limit_expires_unfilled(monkeypatch):
    monkeypatch.setattr(trade_monitor, "EXECUTE_TRADES", False)
    monkeypatch.setattr(trade_monitor, "send_alert", _noop_alert)
    monkeypatch.setattr(trade_monitor, "LIMIT_TTL_MINUTES", 30)
    trade = _pending_trade(limit=99.0, minutes_ago=45)
    book = [trade]
    changed = asyncio.run(
        trade_monitor._handle_pending_order(trade, trade["symbol"], "LONG", 101.0, book)
    )
    assert changed is True
    assert trade not in book          # expired orders leave the book
    assert trade["status"] == "PENDING"  # never opened -> never in history


def test_paper_limit_waits_while_fresh(monkeypatch):
    monkeypatch.setattr(trade_monitor, "EXECUTE_TRADES", False)
    monkeypatch.setattr(trade_monitor, "send_alert", _noop_alert)
    monkeypatch.setattr(trade_monitor, "LIMIT_TTL_MINUTES", 30)
    trade = _pending_trade(limit=99.0, minutes_ago=5)
    book = [trade]
    changed = asyncio.run(
        trade_monitor._handle_pending_order(trade, trade["symbol"], "LONG", 101.0, book)
    )
    assert changed is False
    assert trade["status"] == "PENDING" and trade in book


def test_paper_short_limit_fills_on_touch_above(monkeypatch):
    monkeypatch.setattr(trade_monitor, "EXECUTE_TRADES", False)
    monkeypatch.setattr(trade_monitor, "send_alert", _noop_alert)
    trade = _pending_trade(direction="SHORT", limit=101.0, minutes_ago=1)
    book = [trade]
    changed = asyncio.run(
        trade_monitor._handle_pending_order(trade, trade["symbol"], "SHORT", 101.2, book)
    )
    assert changed is True and trade["status"] == "OPEN"


# ---------------------------------------------------------------------------
# Backtester limit-fill walk + risk manager PENDING handling
# ---------------------------------------------------------------------------

def test_simulate_limit_fill_long():
    #                 bar0   bar1
    highs = [101.0, 100.5]; lows = [100.0, 98.5]
    assert _simulate_limit_fill("LONG", 99.0, highs, lows, ttl_bars=6) == 1


def test_simulate_limit_fill_expires():
    highs = [101.0, 102.0]; lows = [100.0, 100.5]
    assert _simulate_limit_fill("LONG", 99.0, highs, lows, ttl_bars=2) is None


def test_pending_counts_toward_risk_caps():
    pending = {"symbol": "BTC/USDT:USDT", "status": "PENDING", "direction": "LONG",
               "entry": 100, "sl": 99, "qty": 1}
    dup = {"symbol": "BTC/USDT:USDT", "direction": "LONG", "entry": 100, "sl": 99, "qty": 1}
    ok, reason = can_open_trade(dup, [pending], balance=100000)
    assert not ok and "already have an open position" in reason
