"""Close-once guard (no double balance/alerts) and the configurable AI vote."""

import paper_trader
from paper_trader import close_paper_trade_with_fees
from strategy import ai_blend_weight


def _trade():
    return {"symbol": "SOLUSDT", "direction": "LONG", "entry": 100.0,
            "sl": 98.0, "tp": 104.0, "qty": 2.0, "entry_fee": 0.05}


def test_close_skips_balance_when_already_closed(monkeypatch):
    balance_calls = []
    monkeypatch.setattr(paper_trader, "close_trade", lambda *a, **k: None)
    monkeypatch.setattr(paper_trader, "update_balance", lambda pnl: balance_calls.append(pnl))

    result = close_paper_trade_with_fees(_trade(), 103.0, "Trailing Stop Hit")

    assert result is None          # signals caller to skip the alert
    assert balance_calls == []     # balance NOT double-counted


def test_close_updates_balance_when_close_succeeds(monkeypatch):
    balance_calls = []
    monkeypatch.setattr(paper_trader, "close_trade", lambda *a, **k: {"symbol": "SOLUSDT"})
    monkeypatch.setattr(paper_trader, "update_balance", lambda pnl: balance_calls.append(pnl))

    result = close_paper_trade_with_fees(_trade(), 103.0, "Trailing Stop Hit")

    assert result is not None and result > 0
    assert len(balance_calls) == 1
    assert balance_calls[0] == result


def test_ai_weight_ceiling_is_configurable():
    # Explicit ceiling: at full ramp, weight == the configured max.
    assert ai_blend_weight(200, max_weight=0.70) == 0.70
    # Ramp still applies below full_at even with a higher ceiling.
    mid = ai_blend_weight(90, max_weight=0.70)
    assert 0.0 < mid < 0.70
    # And zero below the minimum evidence bar, regardless of ceiling.
    assert ai_blend_weight(30, max_weight=0.70) == 0.0
