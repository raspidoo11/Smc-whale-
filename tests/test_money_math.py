"""Money-math and state-machine tests — the parts where a silent bug costs
real capital. No network, no model files required."""

import paper_trader
from paper_trader import (
    calculate_qty,
    calculate_entry_fee,
    calculate_exit_fee,
    close_paper_trade_with_fees,
)
from xgboost_trainer import calculate_realized_r, calculate_historical_context


def test_calculate_qty_caps_risk_at_5usd(monkeypatch):
    # $5 max risk, $1 risk-per-unit -> ~5 units (minus the entry-fee haircut).
    monkeypatch.setattr(paper_trader, "get_balance", lambda: {"balance": 100.0})
    qty = calculate_qty(entry=100.0, sl=99.0)
    assert 4.9 < qty < 5.0


def test_calculate_qty_respects_leverage_cap(monkeypatch):
    # Tiny stop distance would ask for a huge position; leverage cap must bind.
    monkeypatch.setattr(paper_trader, "get_balance", lambda: {"balance": 100.0})
    qty = calculate_qty(entry=100.0, sl=99.999)
    # max notional = balance * leverage = 1000 -> max qty = 1000/100 = 10
    assert qty == 10.0


def test_calculate_qty_zero_on_degenerate_stop(monkeypatch):
    monkeypatch.setattr(paper_trader, "get_balance", lambda: {"balance": 100.0})
    assert calculate_qty(entry=100.0, sl=100.0) == 0.0


def test_fees_are_symmetric_and_correct():
    assert calculate_entry_fee(100.0, 5.0) == 0.2   # 500 notional * 0.0004
    assert calculate_exit_fee(100.0, 5.0) == 0.2


def test_realized_r_long_and_short():
    long_win = {"direction": "LONG", "entry": 100, "sl": 98, "exit_price": 104}
    assert calculate_realized_r(long_win) == 2.0  # gained 2x the 2-unit risk

    short_win = {"direction": "SHORT", "entry": 100, "sl": 102, "exit_price": 96}
    assert calculate_realized_r(short_win) == 2.0

    long_loss = {"direction": "LONG", "entry": 100, "sl": 98, "exit_price": 98}
    assert calculate_realized_r(long_loss) == -1.0


def test_realized_r_zero_when_no_risk():
    assert calculate_realized_r({"direction": "LONG", "entry": 100, "sl": 100}) == 0.0


def test_realized_r_survives_null_exit_price():
    # Legacy/open rows can carry exit_price=None; must not raise.
    assert calculate_realized_r(
        {"direction": "LONG", "entry": 100, "sl": 98, "exit_price": None}
    ) == 0.0


def test_historical_context_survives_null_pnl():
    # Pre-fix history rows persisted pnl as null; context math must not crash.
    history = [
        {"status": "WIN", "pnl": None},
        {"status": "LOSS", "pnl": None},
        {"status": "WIN", "pnl": 2.0},
        {"status": "WIN", "pnl": None},
        {"status": "LOSS", "pnl": -1.0},
        {"status": "WIN", "pnl": 3.0},
    ]
    ctx = calculate_historical_context(history)
    assert "recent_win_rate" in ctx
    assert isinstance(ctx["cumulative_pnl"], float)


def test_close_pnl_status_derived_from_pnl_not_exit_reason(monkeypatch):
    """Status must come from the PnL sign, never from matching exit_reason
    strings — the old string-match labeled every trade LOSS."""
    captured = {}

    def fake_close_trade(symbol, exit_price, status, extra_fields=None):
        captured["status"] = status
        captured["extra"] = extra_fields
        return {"symbol": symbol, "status": status, **(extra_fields or {})}

    monkeypatch.setattr(paper_trader, "update_balance", lambda pnl: None)
    monkeypatch.setattr(paper_trader, "close_trade", fake_close_trade)

    trade = {"symbol": "BTCUSDT", "entry": 100.0, "qty": 1.0, "direction": "LONG"}
    # exit above entry -> WIN regardless of the exit_reason wording
    close_paper_trade_with_fees(trade, exit_price=110.0, exit_reason="Some Novel Reason")
    assert captured["status"] == "WIN"
    assert captured["extra"]["pnl"] is not None

    # Real stop-out still LOSS
    close_paper_trade_with_fees(trade, exit_price=98.0, exit_reason="Stop Loss Hit")
    assert captured["status"] == "LOSS"
