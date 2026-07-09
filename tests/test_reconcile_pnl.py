"""Reconcile close alerts must always carry a real W/L and PnL — the
zero-PnL fallback made every exchange-side close read 'LOSS +0.00'."""

import sys
import types

# Stub the exchange module before reconcile's import chain touches it.
if "exchange" not in sys.modules or not hasattr(sys.modules.get("exchange"), "_TEST_STUB"):
    stub = types.ModuleType("exchange")
    stub._TEST_STUB = True
    stub.get_exchange = lambda: None
    stub.get_trade_client = lambda: None
    sys.modules["exchange"] = stub

from reconcile import _fallback_pnl
from alerts import format_close_alert


def test_fallback_pnl_long_win():
    trade = {"direction": "LONG", "entry": 100.0, "qty": 2.0}
    assert _fallback_pnl(trade, 103.0) == 6.0


def test_fallback_pnl_short_win():
    trade = {"direction": "SHORT", "entry": 100.0, "qty": 2.0}
    assert _fallback_pnl(trade, 97.0) == 6.0


def test_fallback_pnl_long_loss():
    trade = {"direction": "LONG", "entry": 100.0, "qty": 1.5}
    assert _fallback_pnl(trade, 98.0) == -3.0


def test_close_alert_shows_win_and_profit():
    trade = {"trade_no": 12, "symbol": "SOLUSDT", "direction": "LONG",
             "entry": 100.0, "sl": 98.0, "tp": 104.0, "qty": 2.0, "leverage": 10}
    msg = format_close_alert(trade, 103.0, "Closed on exchange (SL/TP)", 6.0, 250.0)
    assert "WIN" in msg
    assert "+6.00 USDT" in msg
    assert "#12" in msg


def test_close_alert_shows_loss_and_negative_pnl():
    trade = {"trade_no": 13, "symbol": "SOLUSDT", "direction": "LONG",
             "entry": 100.0, "sl": 98.0, "tp": 104.0, "qty": 1.5, "leverage": 10}
    msg = format_close_alert(trade, 98.0, "Closed on exchange (SL/TP)", -3.0, 240.0)
    assert "LOSS" in msg
    assert "-3.00 USDT" in msg


def test_add_daily_pnl_touches_breaker_not_balance(monkeypatch):
    """Exchange-side closes must count toward the daily circuit breaker
    without re-adjusting balance (wallet equity is already synced)."""
    import trade_manager as tm

    state = {"balance": 500.0, "daily_pnl": -10.0}
    monkeypatch.setattr(tm, "get_balance", lambda: dict(state))
    saved = {}
    monkeypatch.setattr(tm, "save_balance", lambda d: saved.update(d))

    tm.add_daily_pnl(-25.0)

    assert saved["daily_pnl"] == -35.0   # breaker sees the loss
    assert saved["balance"] == 500.0     # balance untouched (equity-synced)
