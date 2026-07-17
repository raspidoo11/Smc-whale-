"""Scalp pacing: the time stop (MAX_HOLD_MINUTES) and the stop-distance cap
(SL_MAX_ATR_MULT). Both default OFF — these tests exercise the enabled paths."""

import sys
import types
from datetime import datetime, timedelta, timezone

import pytest

if "exchange" not in sys.modules or not hasattr(sys.modules.get("exchange"), "_TEST_STUB"):
    stub = types.ModuleType("exchange")
    stub._TEST_STUB = True
    stub.get_exchange = lambda: None
    stub.get_trade_client = lambda: None
    sys.modules["exchange"] = stub

import strategy
import trade_monitor
from strategy import cap_stop_distance
from backtester import _simulate_exit


# ---------------------------------------------------------------------------
# Stop-distance cap
# ---------------------------------------------------------------------------

def test_cap_disabled_leaves_stop_alone(monkeypatch):
    monkeypatch.setattr(strategy, "SL_MAX_ATR_MULT", 0)
    assert cap_stop_distance("LONG", 90.0, 100.0, atr=2.0) == 90.0


def test_cap_bounds_long_stop(monkeypatch):
    monkeypatch.setattr(strategy, "SL_MAX_ATR_MULT", 1.5)
    # Structural stop 5 ATR away -> clamped to 1.5 ATR below reference.
    assert cap_stop_distance("LONG", 90.0, 100.0, atr=2.0) == 100.0 - 3.0
    # Structural stop already inside the cap -> untouched.
    assert cap_stop_distance("LONG", 98.5, 100.0, atr=2.0) == 98.5


def test_cap_bounds_short_stop(monkeypatch):
    monkeypatch.setattr(strategy, "SL_MAX_ATR_MULT", 1.5)
    assert cap_stop_distance("SHORT", 110.0, 100.0, atr=2.0) == 100.0 + 3.0
    assert cap_stop_distance("SHORT", 101.5, 100.0, atr=2.0) == 101.5


# ---------------------------------------------------------------------------
# Time stop: hold-expiry logic (monitor)
# ---------------------------------------------------------------------------

def _open_trade(minutes_ago, trailing=False):
    t = {
        "symbol": "BTC/USDT:USDT", "direction": "LONG", "status": "OPEN",
        "entry": 100.0, "sl": 97.0, "tp": 105.0, "qty": 1.0,
        "filled_at": (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat(),
    }
    if trailing:
        t["trailing_stop_active"] = True
    return t


def test_hold_expired_after_max_hold(monkeypatch):
    monkeypatch.setattr(trade_monitor, "MAX_HOLD_MINUTES", 120)
    assert trade_monitor._hold_expired(_open_trade(minutes_ago=121)) is True
    assert trade_monitor._hold_expired(_open_trade(minutes_ago=60)) is False


def test_hold_never_expires_when_disabled(monkeypatch):
    monkeypatch.setattr(trade_monitor, "MAX_HOLD_MINUTES", 0)
    assert trade_monitor._hold_expired(_open_trade(minutes_ago=100000)) is False


def test_trailing_winner_is_never_evicted(monkeypatch):
    monkeypatch.setattr(trade_monitor, "MAX_HOLD_MINUTES", 120)
    assert trade_monitor._hold_expired(_open_trade(minutes_ago=999, trailing=True)) is False


# ---------------------------------------------------------------------------
# Time stop in the backtester's exit walk
# ---------------------------------------------------------------------------

def test_simulate_exit_time_stop_fires():
    # Price drifts sideways: no SL (97), no activation (near 105).
    highs = [101, 101, 101, 101]
    lows = [99, 99, 99, 99]
    closes = [100, 100.2, 99.8, 100.1]
    price, reason, bars = _simulate_exit(
        "LONG", 100, 97, 105, highs, lows, closes, 0.3, 0.9, max_hold_bars=3
    )
    assert reason == "Time Stop (max hold)"
    assert bars == 3
    assert price == 99.8  # the close of the eviction bar


def test_simulate_exit_time_stop_does_not_preempt_sl():
    highs = [101, 101]
    lows = [96.5, 99]  # SL 97 breached on bar 0
    closes = [97, 100]
    price, reason, bars = _simulate_exit(
        "LONG", 100, 97, 105, highs, lows, closes, 0.3, 0.9, max_hold_bars=1
    )
    assert reason == "Stop Loss Hit" and price == 97


def test_simulate_exit_disabled_time_stop_unchanged():
    highs = [101] * 6
    lows = [99] * 6
    closes = [100] * 6
    price, reason, bars = _simulate_exit(
        "LONG", 100, 97, 105, highs, lows, closes, 0.3, 0.9
    )
    assert reason == "Open at data end"
