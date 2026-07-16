"""Trailing-stop price rounding + breakeven floor (paper trail hits != LOSS)."""

import asyncio
import sys
import types

from bybit_executor import round_price
from backtester import _simulate_exit
from paper_trader import FEE_RATE, close_paper_trade_with_fees

# trade_monitor calls exchange.get_exchange() at import time (network). Stub
# the exchange module BEFORE importing it so tests stay offline.
if "exchange" not in sys.modules or not hasattr(sys.modules.get("exchange"), "_TEST_STUB"):
    stub = types.ModuleType("exchange")
    stub._TEST_STUB = True
    stub.get_exchange = lambda: None
    stub.get_trade_client = lambda: None
    sys.modules["exchange"] = stub

from trade_monitor import trail_stop_price, _handle_paper_trailing


def test_round_price_tick_step():
    m = {"precision": {"price": 0.0001}}
    assert round_price(0.0365567, m) == 0.0366


def test_round_price_decimal_places():
    m = {"precision": {"price": 2}}
    assert round_price(123.4567, m) == 123.46


def test_round_price_no_market_falls_back():
    assert round_price(1.23456789, None) == round(1.23456789, 6)


def test_trail_stop_floors_at_breakeven_long():
    # Tight stop: pure 0.3% trail from activation sits *below* entry.
    # Floor must lift it to fee-aware BE so a trail hit cannot be a loss.
    entry = 100.0
    anchor = 100.27  # ~90% of way to a 0.3% TP
    raw = anchor * (1 - 0.3 / 100)
    assert raw < entry
    stop = trail_stop_price("LONG", anchor, 0.3, entry, FEE_RATE)
    assert stop >= entry
    assert stop == entry / (1.0 - FEE_RATE)


def test_trail_stop_ceilings_at_breakeven_short():
    entry = 100.0
    anchor = 99.73
    raw = anchor * (1 + 0.3 / 100)
    assert raw > entry
    stop = trail_stop_price("SHORT", anchor, 0.3, entry, FEE_RATE)
    assert stop <= entry
    assert stop == entry / (1.0 + FEE_RATE)


def test_trail_stop_does_not_raise_when_locked_profit_exceeds_trail():
    # Winner that ran: trail sits well above entry — leave it alone.
    stop = trail_stop_price("LONG", anchor=110.0, trail_percent=0.5, entry=100.0, fee_rate=FEE_RATE)
    assert stop == 110.0 * (1 - 0.5 / 100)
    assert stop > 100.0


def test_backtester_trail_hit_never_exits_below_entry_long():
    # Activate near a tight TP, then retrace hard — fill must be >= entry.
    highs = [100.27, 100.20]
    lows = [100.20, 99.50]
    closes = [100.22, 99.60]
    price, reason, _ = _simulate_exit(
        "LONG", 100.0, 99.8, 100.3, highs, lows, closes, 0.3, 0.90
    )
    assert reason == "Trailing Stop Hit"
    assert price >= 100.0


def test_paper_trail_hit_records_win_not_loss(monkeypatch):
    """Armed trail on a tight setup retraces through the raw trail level
    (below entry) — fill is floored at BE and status is WIN."""
    captured = {}

    def fake_close(symbol, exit_price, status, extra_fields=None):
        captured["status"] = status
        captured["exit_price"] = exit_price
        captured["extra"] = extra_fields
        return {"symbol": symbol, "status": status, **(extra_fields or {})}

    monkeypatch.setattr("paper_trader.close_trade", fake_close)
    monkeypatch.setattr("paper_trader.update_balance", lambda pnl: None)

    trade = {
        "symbol": "SOL/USDT:USDT",
        "direction": "LONG",
        "entry": 100.0,
        "sl": 99.8,
        "tp": 100.3,
        "qty": 5.0,
        "entry_fee": 0.02,
        "trail_percent": 0.3,
        "trail_anchor": 100.27,
        "trailing_stop_active": True,
        "status": "OPEN",
    }
    open_trades = [trade]
    # Price well below the raw trail (and below entry) — would have been a LOSS
    # before the BE floor.
    closed = asyncio.run(
        _handle_paper_trailing(trade, trade["symbol"], "LONG", 99.50, open_trades)
    )
    assert closed is True
    assert open_trades == []
    assert captured["status"] == "WIN"
    assert captured["exit_price"] >= 100.0
    assert (captured["extra"] or {}).get("exit_reason") == "Trailing Stop Hit"


def test_close_trail_scratch_at_entry_is_win(monkeypatch):
    """Even if fees nibble a hair past zero, favorable-side trail exit = WIN."""
    captured = {}

    def fake_close(symbol, exit_price, status, extra_fields=None):
        captured["status"] = status
        return {"symbol": symbol, "status": status, **(extra_fields or {})}

    monkeypatch.setattr("paper_trader.close_trade", fake_close)
    monkeypatch.setattr("paper_trader.update_balance", lambda pnl: None)

    trade = {
        "symbol": "BTC/USDT:USDT",
        "direction": "LONG",
        "entry": 100.0,
        "qty": 1.0,
        "entry_fee": 0.04,
    }
    # Exit exactly at entry: raw pnl 0, exit fee > 0 -> pnl_after_fees < 0,
    # but trail + favorable side must still count as WIN.
    close_paper_trade_with_fees(trade, 100.0, "Trailing Stop Hit")
    assert captured["status"] == "WIN"
