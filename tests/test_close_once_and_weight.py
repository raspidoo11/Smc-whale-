"""Close-once guard (no double balance/alerts) and the configurable AI vote."""

import paper_trader
from paper_trader import close_paper_trade_with_fees
from strategy import ai_blend_weight


def _trade():
    return {"symbol": "SOLUSDT", "direction": "LONG", "entry": 100.0,
            "sl": 98.0, "tp": 104.0, "qty": 2.0, "entry_fee": 0.05}


def test_close_skips_balance_when_already_closed(monkeypatch):
    captured = {}

    def fake_close(*a, **k):
        captured["called"] = True
        captured["balance_delta"] = k.get("balance_delta")
        return None  # already closed elsewhere

    monkeypatch.setattr(paper_trader, "close_trade", fake_close)

    result = close_paper_trade_with_fees(_trade(), 103.0, "Trailing Stop Hit")

    assert result is None          # signals caller to skip the alert
    assert captured.get("called") is True
    # balance_delta is offered, but close_trade returning None means it
    # must NOT apply it (no double-count).


def test_close_passes_balance_delta_atomically(monkeypatch):
    captured = {}

    def fake_close(symbol, exit_price, status, extra_fields=None, balance_delta=None, trade_no=None):
        captured["balance_delta"] = balance_delta
        captured["extra"] = extra_fields
        captured["status"] = status
        return {"symbol": symbol, "status": status, **(extra_fields or {})}

    monkeypatch.setattr(paper_trader, "close_trade", fake_close)

    result = close_paper_trade_with_fees(_trade(), 103.0, "Trailing Stop Hit")

    assert result is not None and result > 0
    # PnL must ride in on the same close_trade call (atomic with history).
    assert captured["balance_delta"] == result
    assert captured["extra"]["pnl"] is not None
    assert captured["status"] == "WIN"


def test_ai_weight_ceiling_is_configurable():
    # Explicit ceiling: at full ramp, weight == the configured max.
    assert ai_blend_weight(200, max_weight=0.70) == 0.70
    # Ramp still applies below full_at even with a higher ceiling.
    mid = ai_blend_weight(90, max_weight=0.70)
    assert 0.0 < mid < 0.70
    # And zero below the minimum evidence bar, regardless of ceiling.
    assert ai_blend_weight(30, max_weight=0.70) == 0.0
