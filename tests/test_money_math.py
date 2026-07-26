"""Money-math and state-machine tests — the parts where a silent bug costs
real capital. No network, no model files required."""

import pytest

import paper_trader
from paper_trader import (
    calculate_qty,
    calculate_entry_fee,
    calculate_exit_fee,
    close_paper_trade_with_fees,
)
from xgboost_trainer import calculate_realized_r, calculate_historical_context


def _balance(monkeypatch, amount):
    """Pin the balance for BOTH paper_trader and the shared risk helper —
    calculate_qty now sizes off trade_manager.get_risk_amount()."""
    monkeypatch.setattr(paper_trader, "get_balance", lambda: {"balance": amount})
    import trade_manager
    monkeypatch.setattr(trade_manager, "get_balance", lambda: {"balance": amount})


def test_calculate_qty_risks_configured_percent_of_balance(monkeypatch):
    # RISK_PERCENT=0.5 of a $100 balance -> $0.50 at risk, $1 risk-per-unit
    # -> ~0.5 units (minus the entry-fee haircut). Previously this was a flat
    # $5 regardless of balance, i.e. ~10x the live path's risk.
    from config import RISK_PERCENT
    _balance(monkeypatch, 100.0)
    qty = calculate_qty(entry=100.0, sl=99.0)
    expected = (100.0 * RISK_PERCENT / 100) / 1.0
    assert expected * 0.98 < qty <= expected


def test_calculate_qty_scales_with_balance(monkeypatch):
    """Percent-of-balance sizing must compound; the old flat-$5 rule did not."""
    _balance(monkeypatch, 100.0)
    small = calculate_qty(entry=100.0, sl=99.0)
    _balance(monkeypatch, 400.0)
    large = calculate_qty(entry=100.0, sl=99.0)
    assert large == pytest.approx(small * 4, rel=1e-3)


def test_paper_and_live_share_one_risk_source(monkeypatch):
    """Regression: paper risked a flat $5 while live risked 0.5% of balance, so
    paper equity curves never described live behaviour."""
    import trade_manager
    _balance(monkeypatch, 250.0)
    risk_per_unit = 2.0
    qty = calculate_qty(entry=100.0, sl=100.0 - risk_per_unit)
    implied_risk = qty * risk_per_unit
    assert implied_risk == pytest.approx(trade_manager.get_risk_amount(), rel=0.01)


def test_calculate_qty_respects_leverage_cap(monkeypatch):
    # Tiny stop distance would ask for a huge position; leverage cap must bind.
    _balance(monkeypatch, 100.0)
    qty = calculate_qty(entry=100.0, sl=99.99999)
    # max notional = balance * leverage = 1000 -> max qty = 1000/100 = 10
    assert qty == 10.0


def test_calculate_qty_zero_on_degenerate_stop(monkeypatch):
    _balance(monkeypatch, 100.0)
    assert calculate_qty(entry=100.0, sl=100.0) == 0.0


def test_fees_use_the_real_maker_taker_split():
    from config import MAKER_FEE_RATE, TAKER_FEE_RATE
    # Entry pays maker in limit mode; exits always fire at market -> taker.
    assert calculate_entry_fee(100.0, 5.0) == round(500 * paper_trader.FEE_RATE, 2)
    assert calculate_exit_fee(100.0, 5.0) == round(500 * TAKER_FEE_RATE, 2)
    assert paper_trader.EXIT_FEE_RATE == TAKER_FEE_RATE
    assert paper_trader.FEE_RATE in (MAKER_FEE_RATE, TAKER_FEE_RATE)


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
