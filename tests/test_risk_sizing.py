"""Live-path position sizing: one risk source, and a hard ceiling on the
AI confidence multiplier.

Regression context: base risk is RISK_PERCENT of balance, but the AI multiplier
(get_ai_risk_percent / 0.5) scales it up to 5x with nothing bounding the result,
so raising RISK_PERCENT silently raised the real maximum too.
"""

import pytest

import bybit_executor
import trade_manager
from config import MAX_RISK_PERCENT, RISK_PERCENT
from xgboost_trainer import get_ai_risk_percent


BALANCE = 1000.0


@pytest.fixture(autouse=True)
def _fixed_balance(monkeypatch):
    monkeypatch.setattr(trade_manager, "get_balance", lambda: {"balance": BALANCE})
    # bybit_executor imported get_risk_amount by name, so patch it there too.
    monkeypatch.setattr(
        bybit_executor, "get_risk_amount", lambda: round(BALANCE * RISK_PERCENT / 100, 2)
    )


@pytest.fixture(autouse=True)
def _fake_market(monkeypatch):
    monkeypatch.setattr(
        bybit_executor, "get_symbol_info",
        lambda symbol: {
            "precision": {"amount": 0.000001},
            "limits": {"amount": {"min": 0.0, "max": 10**9}},
        },
    )


def implied_risk_pct(qty, entry, sl):
    """Percent of balance actually at risk if the stop is hit."""
    return qty * abs(entry - sl) / BALANCE * 100


def test_base_sizing_risks_configured_percent():
    # ai_prob=50 -> get_ai_risk_percent returns base*0.6 = 0.3 -> 0.6x multiplier
    qty = bybit_executor.calculate_proper_qty("BTC/USDT:USDT", 100.0, 98.0, ai_prob=50)
    assert qty is not None
    expected = RISK_PERCENT * (get_ai_risk_percent(50) / 0.5)
    assert implied_risk_pct(qty, 100.0, 98.0) == pytest.approx(expected, rel=0.02)


def test_high_confidence_scales_risk_up():
    low = bybit_executor.calculate_proper_qty("BTC/USDT:USDT", 100.0, 98.0, ai_prob=50)
    high = bybit_executor.calculate_proper_qty("BTC/USDT:USDT", 100.0, 98.0, ai_prob=85)
    assert high > low, "a confident model should size up"


def test_risk_is_capped_at_max_risk_percent():
    """The 5x multiplier must not push risk past the configured ceiling."""
    qty = bybit_executor.calculate_proper_qty(
        "BTC/USDT:USDT", 100.0, 98.0, ai_prob=95, regime="trending", recent_drawdown=0.0
    )
    assert qty is not None
    assert implied_risk_pct(qty, 100.0, 98.0) <= MAX_RISK_PERCENT + 1e-6


def test_cap_binds_even_if_risk_percent_is_raised(monkeypatch):
    """Raising base risk must not raise the ceiling — that was the actual bug."""
    monkeypatch.setattr(bybit_executor, "get_risk_amount", lambda: BALANCE * 2.0 / 100)
    monkeypatch.setattr(
        "config.RISK_PERCENT", 2.0, raising=False
    )
    qty = bybit_executor.calculate_proper_qty(
        "BTC/USDT:USDT", 100.0, 98.0, ai_prob=95, regime="trending"
    )
    assert qty is not None
    # With RISK_PERCENT=2.0 and a 5x multiplier this would be 10% of balance
    # uncapped; the ceiling must hold it at MAX_RISK_PERCENT.
    assert implied_risk_pct(qty, 100.0, 98.0) <= MAX_RISK_PERCENT + 1e-6


def test_degenerate_stop_returns_none():
    assert bybit_executor.calculate_proper_qty("BTC/USDT:USDT", 100.0, 100.0) is None


def test_drawdown_reduces_risk():
    calm = bybit_executor.calculate_proper_qty(
        "BTC/USDT:USDT", 100.0, 98.0, ai_prob=75, recent_drawdown=0.0
    )
    stressed = bybit_executor.calculate_proper_qty(
        "BTC/USDT:USDT", 100.0, 98.0, ai_prob=75, recent_drawdown=7.0
    )
    assert stressed < calm, "drawdown should shrink size"


def test_paper_and_live_agree_on_base_risk(monkeypatch):
    """Both paths must derive from the same helper — the divergence that made
    paper ~10x more aggressive than live."""
    import paper_trader
    monkeypatch.setattr(paper_trader, "get_balance", lambda: {"balance": BALANCE})
    paper_qty = paper_trader.calculate_qty(entry=100.0, sl=98.0)
    paper_risk = implied_risk_pct(paper_qty, 100.0, 98.0)
    # Paper applies no AI multiplier, so it should sit at the base rate.
    assert paper_risk == pytest.approx(RISK_PERCENT, rel=0.02)
